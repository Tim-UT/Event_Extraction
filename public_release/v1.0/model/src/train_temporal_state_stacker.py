#!/usr/bin/env python3
"""Train a grouped-validation endpoint stacker over text and temporal evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import constrained_temporal_extractor as temporal
import hybrid_classification
import train_constrained_temporal_model as presence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "outputs" / "event_extraction_refined_20260710" / "augmented_dataset.csv"
DEFAULT_PRESENCE = ROOT / "outputs" / "event_extraction_refined_20260710" / "constrained_temporal_model" / "constrained_temporal_presence.joblib"
DEFAULT_CLASS = ROOT / "outputs" / "event_extraction_refined_20260710" / "classification_ensemble" / "classification_ensemble.joblib"
DEFAULT_OUTPUT = ROOT / "outputs" / "event_extraction_refined_20260710" / "temporal_state_stacker"
LABELS = ["not related", "notice", "deadline", "event"]
SEED = 20260710


def fast_macro_f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    values = []
    for label in LABELS:
        tp = int(np.sum((truth == label) & (prediction == label)))
        fp = int(np.sum((truth != label) & (prediction == label)))
        fn = int(np.sum((truth == label) & (prediction != label)))
        denominator = 2 * tp + fp + fn
        values.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(values))


def engineered_features(frame: pd.DataFrame, presence_bundle: dict, class_bundle: dict) -> np.ndarray:
    matrix = presence_bundle["vectorizer"].transform(presence.compose_text(frame))
    start_score = presence_bundle["models"]["start"].decision_function(matrix)
    end_score = presence_bundle["models"]["end"].decision_function(matrix)
    _, probabilities = hybrid_classification.predict_rows(class_bundle, frame.to_dict("records"))
    positions = {label: index for index, label in enumerate(class_bundle["labels"])}
    rows = []
    for row_index, (_, row) in enumerate(frame.iterrows()):
        record = row.to_dict()
        title = temporal.clean(row["announcement_title"])
        body = temporal.clean(row["body_text"])
        text = f"{title}\n{body}"
        posted = temporal.posted_at_toronto(row["posted_at"])
        dates_all = temporal.date_mentions(text, posted)
        dates_title = temporal.date_mentions(title, posted)
        dates_body = temporal.date_mentions(body, posted)
        ranges = temporal.time_ranges(text)
        singles = temporal.single_times(text, [(item.start, item.end) for item in ranges])
        candidates = temporal.build_candidates(record)
        role_scores = {
            role: max((temporal.candidate_score(item, role) for item in candidates), default=-5.0)
            for role in ("notice", "deadline", "event")
        }
        rows.append(
            [
                float(start_score[row_index]),
                float(end_score[row_index]),
                *[float(probabilities[row_index, positions[label]]) for label in LABELS],
                float(len(dates_all)),
                float(len(dates_title)),
                float(len(dates_body)),
                float(len(ranges)),
                float(len(singles)),
                float(bool(temporal.DUE_RE.search(text))),
                float(bool(temporal.START_RE.search(text))),
                float(bool(temporal.EVENT_RE.search(text))),
                float(bool(temporal.CANCEL_RE.search(text))),
                float(bool(temporal.RELATIVE_RE.search(text))),
                float(bool(temporal.MONTH_FIRST_RE.search(text) or temporal.DAY_FIRST_RE.search(text) or temporal.NUMERIC_RE.search(text))),
                role_scores["notice"],
                role_scores["deadline"],
                role_scores["event"],
                math_log1p(len(body)),
                float(posted.hour) / 24.0,
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def math_log1p(value: int) -> float:
    return float(np.log1p(value))


def make_model(c_value: float, balanced: bool):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced" if balanced else None,
            max_iter=20_000,
            random_state=SEED,
            solver="liblinear",
        ),
    )


def endpoint_oof(features: np.ndarray, truth: np.ndarray, groups: np.ndarray, configs: list[tuple[float, bool]]):
    splitter = GroupKFold(n_splits=5)
    output = {}
    for config in configs:
        values = np.zeros(len(truth), dtype=np.float64)
        for train_index, validation_index in splitter.split(features, truth, groups):
            model = make_model(*config)
            model.fit(features[train_index], truth[train_index])
            values[validation_index] = model.predict_proba(features[validation_index])[:, 1]
        output[config] = values
    return output


def choose_configuration(frame: pd.DataFrame, features: np.ndarray):
    truth_start = frame["start_time"].ne("").to_numpy()
    truth_end = frame["end_time"].ne("").to_numpy()
    truth_class = frame["classification"].to_numpy()
    groups = frame["source_group_id"].to_numpy()
    configs = [(c, balanced) for c in (0.03, 0.10, 0.30, 1.0, 3.0, 10.0, 30.0) for balanced in (False, True)]
    start_oof = endpoint_oof(features, truth_start, groups, configs)
    end_oof = endpoint_oof(features, truth_end, groups, configs)
    thresholds = np.linspace(0.20, 0.80, 31)
    best = None
    for start_config in configs:
        for end_config in configs:
            for start_threshold in thresholds:
                pred_start = start_oof[start_config] >= start_threshold
                for end_threshold in thresholds:
                    pred_end = end_oof[end_config] >= end_threshold
                    pred_class = presence.class_from_presence(pred_start, pred_end)
                    start_accuracy = float(np.mean(truth_start == pred_start))
                    end_accuracy = float(np.mean(truth_end == pred_end))
                    joint_accuracy = float(
                        np.mean(
                            (truth_start == pred_start)
                            & (truth_end == pred_end)
                        )
                    )
                    key = (
                        joint_accuracy,
                        min(start_accuracy, end_accuracy),
                        (start_accuracy + end_accuracy) / 2.0,
                        float(np.mean(truth_class == pred_class)),
                        fast_macro_f1(truth_class, pred_class),
                    )
                    if best is None or key > best[0]:
                        best = (
                            key,
                            {
                                "start_C": start_config[0],
                                "start_balanced": start_config[1],
                                "end_C": end_config[0],
                                "end_balanced": end_config[1],
                                "start_threshold": float(start_threshold),
                                "end_threshold": float(end_threshold),
                            },
                            pred_start,
                            pred_end,
                            start_oof[start_config].copy(),
                            end_oof[end_config].copy(),
                        )
    return best


def metric_block(frame: pd.DataFrame, pred_start: np.ndarray, pred_end: np.ndarray) -> dict[str, float]:
    truth_start = frame["start_time"].ne("").to_numpy()
    truth_end = frame["end_time"].ne("").to_numpy()
    truth_class = frame["classification"].to_numpy()
    pred_class = presence.class_from_presence(pred_start, pred_end)
    return {
        "classification_accuracy": float(np.mean(truth_class == pred_class)),
        "classification_macro_f1": fast_macro_f1(truth_class, pred_class),
        "start_presence_accuracy": float(np.mean(truth_start == pred_start)),
        "end_presence_accuracy": float(np.mean(truth_end == pred_end)),
        "joint_presence_accuracy": float(np.mean((truth_start == pred_start) & (truth_end == pred_end))),
    }


def write_predictions(
    path: Path,
    frame: pd.DataFrame,
    pred_start: np.ndarray,
    pred_end: np.ndarray,
    start_probability: np.ndarray,
    end_probability: np.ndarray,
) -> None:
    output = frame[["reviewed_row_id", "source_row", "source_group_id", "classification", "start_time", "end_time"]].copy()
    output["predicted_has_start"] = pred_start
    output["predicted_has_end"] = pred_end
    output["start_probability"] = start_probability
    output["end_probability"] = end_probability
    output["predicted_classification"] = presence.class_from_presence(pred_start, pred_end)
    output["correct_classification"] = output["classification"].eq(output["predicted_classification"])
    output.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--presence-model", type=Path, default=DEFAULT_PRESENCE)
    parser.add_argument("--class-model", type=Path, default=DEFAULT_CLASS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data, dtype=str, keep_default_na=False)
    validation = data[(data["dataset_split"].eq("validation")) & (data["row_origin"].eq("reviewed_real"))].copy().reset_index(drop=True)
    test = data[(data["dataset_split"].eq("test")) & (data["row_origin"].eq("reviewed_real"))].copy().reset_index(drop=True)
    presence_bundle = joblib.load(args.presence_model)
    class_bundle = joblib.load(args.class_model)
    validation_features = engineered_features(validation, presence_bundle, class_bundle)
    test_features = engineered_features(test, presence_bundle, class_bundle)
    selected = choose_configuration(validation, validation_features)
    assert selected is not None
    _, config, validation_start, validation_end, validation_start_probability, validation_end_probability = selected
    start_model = make_model(config["start_C"], config["start_balanced"])
    end_model = make_model(config["end_C"], config["end_balanced"])
    start_model.fit(validation_features, validation["start_time"].ne(""))
    end_model.fit(validation_features, validation["end_time"].ne(""))
    test_start_probability = start_model.predict_proba(test_features)[:, 1]
    test_end_probability = end_model.predict_proba(test_features)[:, 1]
    test_start = test_start_probability >= config["start_threshold"]
    test_end = test_end_probability >= config["end_threshold"]
    result = {
        "architecture": "two endpoint logistic stackers over base-model scores and deterministic temporal-candidate features; class derived from endpoints",
        "selection_metric": "grouped validation out-of-fold joint endpoint presence, then minimum and mean endpoint accuracy; hard-topology class is diagnostic",
        "selection": config,
        "validation_grouped_oof": metric_block(validation, validation_start, validation_end),
        "test": metric_block(test, test_start, test_end),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    joblib.dump(
        {
            "presence_bundle": presence_bundle,
            "class_bundle": class_bundle,
            "start_model": start_model,
            "end_model": end_model,
            "config": config,
        },
        args.output_dir / "temporal_state_stacker.joblib",
    )
    write_predictions(
        args.output_dir / "validation_oof_predictions.csv",
        validation,
        validation_start,
        validation_end,
        validation_start_probability,
        validation_end_probability,
    )
    write_predictions(
        args.output_dir / "test_predictions.csv",
        test,
        test_start,
        test_end,
        test_start_probability,
        test_end_probability,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
