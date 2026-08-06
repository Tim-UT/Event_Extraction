#!/usr/bin/env python3
"""Train a leakage-safe temporal-presence model with constrained classification.

The model predicts two binary facts from the five source columns:

* has_start_time
* has_end_time

Classification is then derived deterministically:
both -> event, start only -> notice, end only -> deadline, neither -> not related.
Model/threshold selection uses only the grouped validation split.  The grouped
test split is evaluated once after the configuration is frozen.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "outputs" / "event_extraction_refined_20260710" / "augmented_dataset.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "event_extraction_refined_20260710" / "constrained_temporal_model"
LABELS = ["not related", "notice", "deadline", "event"]
SEED = 20260710

DATE_PATTERN = re.compile(
    r"\b(?:today|tomorrow|tonight|this\s+(?:morning|afternoon|evening))\b|"
    r"\b(?:mon(?:day)?|tue(?:s(?:day)?)?|wed(?:nesday)?|thu(?:rs(?:day)?)?|"
    r"fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b|"
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\.?\s*\d{1,2}(?:st|nd|rd|th)?\b|"
    r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b|\b20\d{2}-\d{1,2}-\d{1,2}\b",
    re.I,
)
TIME_PATTERN = re.compile(
    r"\b(?:[01]?\d|2[0-3])\s*:\s*[0-5]\d\s*(?:a\.?m\.?|p\.?m\.?)?\b|"
    r"\b\d{1,2}(?::[0-5]\d)?\s*(?:a\.?m\.?|p\.?m\.?|[ap])\b|"
    r"\b(?:noon|midnight)\b",
    re.I,
)
RANGE_PATTERN = re.compile(
    r"(?:\b(?:from|between)\b.{0,24})?"
    r"\b\d{1,2}(?::[0-5]\d)?\s*(?:a\.?m\.?|p\.?m\.?|[ap])?\s*"
    r"(?:-|–|—|to|until|through)\s*"
    r"\d{1,2}(?::[0-5]\d)?\s*(?:a\.?m\.?|p\.?m\.?|[ap])\b",
    re.I,
)
DUE_PATTERN = re.compile(
    r"\b(?:due|deadline|submit|submission|complete\s+by|respond\s+by|"
    r"register\s+by|apply\s+by|closes?|close\s+at|until|last\s+day|"
    r"no\s+later\s+than|must\s+be\s+(?:received|completed|submitted))\b",
    re.I,
)
START_PATTERN = re.compile(
    r"\b(?:opens?|available|released?|posted|published|begins?|starts?|"
    r"goes?\s+live|launch(?:es|ed)?|will\s+be\s+available)\b",
    re.I,
)
EVENT_PATTERN = re.compile(
    r"\b(?:session|lecture|tutorial|exam|test|midterm|quiz|office\s+hours?|"
    r"workshop|meeting|presentation|review|seminar|webinar|class|lab|"
    r"showcase|check-?in|held|take\s+place|join\s+us|attend)\b",
    re.I,
)
CANCEL_PATTERN = re.compile(r"\b(?:cancelled|canceled|postponed|rescheduled|no\s+class)\b", re.I)


def clean(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def temporal_signature(text: str) -> str:
    """Emit explicit structural tokens alongside lexical TF-IDF features."""
    tokens: list[str] = []
    checks = [
        (DATE_PATTERN.search(text), "HAS_DATE"),
        (TIME_PATTERN.search(text), "HAS_TIME"),
        (RANGE_PATTERN.search(text), "HAS_TIME_RANGE"),
        (DUE_PATTERN.search(text), "HAS_DUE_CUE"),
        (START_PATTERN.search(text), "HAS_START_CUE"),
        (EVENT_PATTERN.search(text), "HAS_EVENT_CUE"),
        (CANCEL_PATTERN.search(text), "HAS_CANCEL_CUE"),
    ]
    for matched, token in checks:
        if matched:
            tokens.extend([token] * 4)
    if not DATE_PATTERN.search(text) and not TIME_PATTERN.search(text):
        tokens.extend(["NO_TEMPORAL_MENTION"] * 4)
    return " ".join(tokens)


def compose_text(frame: pd.DataFrame) -> pd.Series:
    values = []
    for _, row in frame.iterrows():
        title = clean(row.get("announcement_title"))
        body = clean(row.get("body_text"))
        course = clean(row.get("course_name"))
        source = f"TITLE {title} TITLE {title} COURSE {course} BODY {body}"
        values.append(f"{source} STRUCTURE {temporal_signature(source)}")
    return pd.Series(values, index=frame.index)


def class_from_presence(has_start: np.ndarray, has_end: np.ndarray) -> np.ndarray:
    return np.where(
        has_start & has_end,
        "event",
        np.where(has_start, "notice", np.where(has_end, "deadline", "not related")),
    )


def class_metrics(truth: pd.Series | np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    report = classification_report(truth, prediction, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=LABELS, average="macro", zero_division=0)),
        "per_class_f1": {label: float(report[label]["f1-score"]) for label in LABELS},
        "report": report,
    }


def binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
    }


def evaluate(
    frame: pd.DataFrame,
    start_scores: np.ndarray,
    end_scores: np.ndarray,
    start_threshold: float,
    end_threshold: float,
) -> dict[str, object]:
    true_start = frame["start_time"].ne("").to_numpy()
    true_end = frame["end_time"].ne("").to_numpy()
    pred_start = start_scores >= start_threshold
    pred_end = end_scores >= end_threshold
    prediction = class_from_presence(pred_start, pred_end)
    return {
        "classification": class_metrics(frame["classification"], prediction),
        "start_presence": binary_metrics(true_start, pred_start),
        "end_presence": binary_metrics(true_end, pred_end),
        "joint_presence_accuracy": float(np.mean((true_start == pred_start) & (true_end == pred_end))),
    }


def make_vectorizer(kind: str) -> TfidfVectorizer:
    if kind == "word":
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.997,
            max_features=180_000,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 6),
        min_df=2,
        max_df=0.997,
        max_features=220_000,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )


def write_predictions(
    path: Path,
    frame: pd.DataFrame,
    start_scores: np.ndarray,
    end_scores: np.ndarray,
    start_threshold: float,
    end_threshold: float,
) -> None:
    output = frame[["reviewed_row_id", "source_row", "source_group_id", "classification", "start_time", "end_time"]].copy()
    output["predicted_has_start"] = start_scores >= start_threshold
    output["predicted_has_end"] = end_scores >= end_threshold
    output["predicted_classification"] = class_from_presence(
        output["predicted_has_start"].to_numpy(), output["predicted_has_end"].to_numpy()
    )
    output["correct_classification"] = output["classification"].eq(output["predicted_classification"])
    output["start_score"] = start_scores
    output["end_score"] = end_scores
    output.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data, dtype=str, keep_default_na=False)
    train = data[data["dataset_split"].eq("train")].copy().reset_index(drop=True)
    validation = data[
        data["dataset_split"].eq("validation") & data["row_origin"].eq("reviewed_real")
    ].copy().reset_index(drop=True)
    test = data[
        data["dataset_split"].eq("test") & data["row_origin"].eq("reviewed_real")
    ].copy().reset_index(drop=True)

    groups = {name: set(frame["source_group_id"]) for name, frame in [("train", train), ("validation", validation), ("test", test)]}
    overlap = {
        "train_validation": len(groups["train"] & groups["validation"]),
        "train_test": len(groups["train"] & groups["test"]),
        "validation_test": len(groups["validation"] & groups["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Group leakage detected: {overlap}")

    train_text = compose_text(train)
    validation_text = compose_text(validation)
    test_text = compose_text(test)
    true_train = {
        "start": train["start_time"].ne("").astype(int).to_numpy(),
        "end": train["end_time"].ne("").astype(int).to_numpy(),
    }

    c_values = [0.10, 0.20, 0.35, 0.55, 0.85, 1.30, 2.00, 3.00]
    has_synthetic = train["row_origin"].eq("synthetic_paraphrase").any()
    synthetic_weights = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0] if has_synthetic else [0.0]
    threshold_values = np.arange(-0.80, 0.81, 0.05)
    candidates: list[dict[str, object]] = []
    fitted_by_kind: dict[str, dict[str, object]] = {}

    for kind in ("word", "char"):
        print(f"Vectorizing {kind} features", flush=True)
        vectorizer = make_vectorizer(kind)
        train_matrix = vectorizer.fit_transform(train_text)
        validation_matrix = vectorizer.transform(validation_text)
        test_matrix = vectorizer.transform(test_text)
        fitted_by_kind[kind] = {
            "vectorizer": vectorizer,
            "train_matrix": train_matrix,
            "validation_matrix": validation_matrix,
            "test_matrix": test_matrix,
        }
        for synthetic_weight in synthetic_weights:
            sample_weight = np.where(
                train["row_origin"].eq("synthetic_paraphrase").to_numpy(), synthetic_weight, 1.0
            )
            for c_value in c_values:
                models: dict[str, LinearSVC] = {}
                validation_scores: dict[str, np.ndarray] = {}
                for target in ("start", "end"):
                    model = LinearSVC(
                        C=c_value,
                        class_weight="balanced",
                        dual="auto",
                        max_iter=30_000,
                        random_state=SEED,
                        tol=1e-5,
                    )
                    model.fit(train_matrix, true_train[target], sample_weight=sample_weight)
                    models[target] = model
                    validation_scores[target] = model.decision_function(validation_matrix)
                best_thresholds = (0.0, 0.0)
                best_metrics: dict[str, object] | None = None
                best_key: tuple[float, ...] | None = None
                validation_true_start = validation["start_time"].ne("").to_numpy()
                validation_true_end = validation["end_time"].ne("").to_numpy()
                validation_true_class = validation["classification"].to_numpy()
                for start_threshold in threshold_values:
                    for end_threshold in threshold_values:
                        predicted_start = validation_scores["start"] >= start_threshold
                        predicted_end = validation_scores["end"] >= end_threshold
                        predicted_class = class_from_presence(predicted_start, predicted_end)
                        start_accuracy = float(
                            np.mean(validation_true_start == predicted_start)
                        )
                        end_accuracy = float(
                            np.mean(validation_true_end == predicted_end)
                        )
                        joint_accuracy = float(
                            np.mean(
                                (validation_true_start == predicted_start)
                                & (validation_true_end == predicted_end)
                            )
                        )
                        key = (
                            joint_accuracy,
                            min(start_accuracy, end_accuracy),
                            (start_accuracy + end_accuracy) / 2.0,
                            float(f1_score(validation_true_class, predicted_class, labels=LABELS, average="macro", zero_division=0)),
                            float(np.mean(validation_true_class == predicted_class)),
                        )
                        if best_key is None or key > best_key:
                            best_key = key
                            best_thresholds = (float(start_threshold), float(end_threshold))
                best_metrics = evaluate(
                    validation,
                    validation_scores["start"],
                    validation_scores["end"],
                    best_thresholds[0],
                    best_thresholds[1],
                )
                candidates.append(
                    {
                        "kind": kind,
                        "C": c_value,
                        "synthetic_weight": synthetic_weight,
                        "start_threshold": best_thresholds[0],
                        "end_threshold": best_thresholds[1],
                        "validation": best_metrics,
                    }
                )
                print(
                    f"{kind} sw={synthetic_weight:.2f} C={c_value:.2f} "
                    f"val_acc={best_metrics['classification']['accuracy']:.4f} "
                    f"val_macro={best_metrics['classification']['macro_f1']:.4f}",
                    flush=True,
                )

    candidates.sort(
        key=lambda item: (
            item["validation"]["joint_presence_accuracy"],
            min(
                item["validation"]["start_presence"]["accuracy"],
                item["validation"]["end_presence"]["accuracy"],
            ),
            (
                item["validation"]["start_presence"]["accuracy"]
                + item["validation"]["end_presence"]["accuracy"]
            )
            / 2.0,
            item["validation"]["classification"]["macro_f1"],
        ),
        reverse=True,
    )
    selected = candidates[0]
    kind = str(selected["kind"])
    matrices = fitted_by_kind[kind]
    sample_weight = np.where(
        train["row_origin"].eq("synthetic_paraphrase").to_numpy(),
        float(selected["synthetic_weight"]),
        1.0,
    )
    final_models: dict[str, LinearSVC] = {}
    scores: dict[str, dict[str, np.ndarray]] = {"validation": {}, "test": {}}
    for target in ("start", "end"):
        model = LinearSVC(
            C=float(selected["C"]),
            class_weight="balanced",
            dual="auto",
            max_iter=30_000,
            random_state=SEED,
            tol=1e-5,
        )
        model.fit(matrices["train_matrix"], true_train[target], sample_weight=sample_weight)
        final_models[target] = model
        scores["validation"][target] = model.decision_function(matrices["validation_matrix"])
        scores["test"][target] = model.decision_function(matrices["test_matrix"])

    start_threshold = float(selected["start_threshold"])
    end_threshold = float(selected["end_threshold"])
    final_metrics = {
        "architecture": "two independent TF-IDF LinearSVC temporal-presence heads plus deterministic classification constraint",
        "classification_rule": {
            "start_and_end": "event",
            "start_only": "notice",
            "end_only": "deadline",
            "neither": "not related",
        },
        "selection": {
            "metric": "validation joint endpoint presence, then minimum and mean endpoint accuracy; hard-topology class is diagnostic",
            "feature_kind": kind,
            "C": float(selected["C"]),
            "synthetic_weight": float(selected["synthetic_weight"]),
            "start_threshold": start_threshold,
            "end_threshold": end_threshold,
        },
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "group_overlap": overlap,
        "validation": evaluate(
            validation,
            scores["validation"]["start"],
            scores["validation"]["end"],
            start_threshold,
            end_threshold,
        ),
        "test": evaluate(
            test,
            scores["test"]["start"],
            scores["test"]["end"],
            start_threshold,
            end_threshold,
        ),
        "top_validation_candidates": candidates[:20],
    }
    bundle = {
        "architecture": final_metrics["architecture"],
        "vectorizer": matrices["vectorizer"],
        "models": final_models,
        "start_threshold": start_threshold,
        "end_threshold": end_threshold,
    }
    joblib.dump(bundle, args.output_dir / "constrained_temporal_presence.joblib")
    (args.output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n", encoding="utf-8")
    write_predictions(
        args.output_dir / "validation_predictions.csv",
        validation,
        scores["validation"]["start"],
        scores["validation"]["end"],
        start_threshold,
        end_threshold,
    )
    write_predictions(
        args.output_dir / "test_predictions.csv",
        test,
        scores["test"]["start"],
        scores["test"]["end"],
        start_threshold,
        end_threshold,
    )
    print(json.dumps({"selection": final_metrics["selection"], "validation": final_metrics["validation"], "test": final_metrics["test"]}, indent=2))


if __name__ == "__main__":
    main()
