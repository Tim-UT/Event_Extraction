#!/usr/bin/env python3
"""Run the selected event-extraction pipeline on a CSV file.

The first five named fields are model inputs. Any additional source metadata is
passed through unchanged, and a deterministic candidate identifier is added
when the input does not already provide one. Temporal endpoint presence is
predicted independently, normalized endpoint values are extracted, and the
classification is derived from the resulting start/end topology.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

import constrained_temporal_extractor as temporal
import hybrid_classification
import train_advanced_event_extractor as span_training
import train_constrained_temporal_model as presence
import train_temporal_state_stacker as stacker


MODEL_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = MODEL_ROOT / "artifacts"
DEFAULT_SPAN_MODEL = MODEL_ROOT / "span_model"
DEFAULT_BASE_ENDPOINT_MODEL = ARTIFACT_ROOT / "constrained_temporal_presence.joblib"
DEFAULT_CLASSIFICATION_MODEL = ARTIFACT_ROOT / "classification_ensemble.joblib"
DEFAULT_TREE_MODEL = ARTIFACT_ROOT / "tree_temporal_stacker.joblib"
DEFAULT_LOGISTIC_MODEL = ARTIFACT_ROOT / "temporal_state_stacker.joblib"
DEFAULT_CALIBRATION = ARTIFACT_ROOT / "endpoint_ensemble_calibration.joblib"

INPUT_FIELDS = [
    "course_name",
    "announcement_title",
    "author",
    "posted_at",
    "body_text",
]
TARGET_FIELDS = [
    "classification",
    "location",
    "item_urls",
    "extracted_title",
    "start_time",
    "end_time",
]
CORE_FIELDS = INPUT_FIELDS + TARGET_FIELDS
CANDIDATE_ID_FIELD = "candidate_id"
URL_CANDIDATE_RE = re.compile(
    r"(?:https?|ftp)://.*?(?=(?:https?|ftp)://|[\s<>\"']|$)",
    re.I,
)
REDIRECT_QUERY_KEYS = (
    "url",
    "u",
    "q",
    "target",
    "dest",
    "destination",
    "redirect",
    "redirect_url",
)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def endpoint_probabilities(
    frame: pd.DataFrame,
    base_bundle: dict[str, Any],
    class_bundle: dict[str, Any],
    tree_bundle: dict[str, Any],
    logistic_bundle: dict[str, Any],
) -> dict[str, np.ndarray]:
    features = stacker.engineered_features(frame, base_bundle, class_bundle)
    base_matrix = base_bundle["vectorizer"].transform(presence.compose_text(frame))
    _, class_probabilities = hybrid_classification.predict_rows(
        class_bundle,
        frame.to_dict("records"),
    )
    positions = {
        label: index for index, label in enumerate(class_bundle["labels"])
    }
    required_labels = {"not related", "notice", "deadline", "event"}
    missing_labels = required_labels.difference(positions)
    if missing_labels:
        raise ValueError(
            "Classification artifact is missing labels: "
            + ", ".join(sorted(missing_labels))
        )
    return {
        "tree_start": tree_bundle["start_model"].predict_proba(features)[:, 1],
        "tree_end": tree_bundle["end_model"].predict_proba(features)[:, 1],
        "logistic_start": logistic_bundle["start_model"].predict_proba(features)[
            :, 1
        ],
        "logistic_end": logistic_bundle["end_model"].predict_proba(features)[
            :, 1
        ],
        "base_start": sigmoid(
            base_bundle["models"]["start"].decision_function(base_matrix)
        ),
        "base_end": sigmoid(
            base_bundle["models"]["end"].decision_function(base_matrix)
        ),
        "direct_start": (
            class_probabilities[:, positions["notice"]]
            + class_probabilities[:, positions["event"]]
        ),
        "direct_end": (
            class_probabilities[:, positions["deadline"]]
            + class_probabilities[:, positions["event"]]
        ),
        **{
            f"direct_class_{label.replace(' ', '_')}": class_probabilities[
                :, position
            ]
            for label, position in positions.items()
        },
    }


def calibrated_endpoint_state(
    values: dict[str, np.ndarray],
    calibration: dict[str, Any],
    endpoint: str,
) -> np.ndarray:
    sources = ["tree", "logistic", "base", "direct"]
    matrix = np.column_stack(
        [values[f"{source}_{endpoint}"] for source in sources]
    )
    selection = calibration[endpoint]
    weights = np.asarray(selection["weights"], dtype=float)
    if weights.shape != (len(sources),):
        raise ValueError(
            f"{endpoint} calibration must contain {len(sources)} weights"
        )
    scores = matrix @ weights
    return scores >= float(selection["threshold"])


def constrained_classifications(
    values: dict[str, np.ndarray],
    calibration: dict[str, Any],
    has_start: np.ndarray,
    has_end: np.ndarray,
) -> list[str]:
    prediction = presence.class_from_presence(has_start, has_end)
    thresholds = calibration.get("semantic_notice_overrides")
    if not isinstance(thresholds, dict):
        return prediction.tolist()
    patterns = {
        "neither": ((~has_start) & (~has_end), "not_related"),
        "both": (has_start & has_end, "event"),
        "end_only": ((~has_start) & has_end, "deadline"),
    }
    notice = values["direct_class_notice"]
    for pattern, (mask, base_key) in patterns.items():
        margin = notice - values[f"direct_class_{base_key}"]
        prediction[
            mask & (margin >= float(thresholds.get(pattern, 1.05)))
        ] = "notice"
    return prediction.tolist()


def span_predictions(
    rows: list[dict[str, str]],
    classifications: list[str],
    model_dir: Path,
    max_length: int,
    batch_size: int,
) -> list[dict[str, str]]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    dataset = span_training.EventDataset(rows, tokenizer, max_length)
    collator = span_training.Collator(tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    device = span_training.choose_device()
    model.to(device).eval()

    output: list[dict[str, str]] = []
    prediction_index = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits.cpu()
            predicted_ids = logits.argmax(dim=-1).tolist()
            for example, predicted in zip(
                batch["examples"],
                predicted_ids,
                strict=True,
            ):
                output.append(
                    span_training.structured_prediction(
                        example,
                        predicted[: len(example.input_ids)],
                        classification_override=classifications[
                            prediction_index
                        ],
                    )
                )
                prediction_index += 1
    if prediction_index != len(rows):
        raise AssertionError(
            f"Span model returned {prediction_index} predictions for "
            f"{len(rows)} rows"
        )
    return output


def prepare_inference_rows(
    source_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in source_rows:
        row = {field: source.get(field, "") or "" for field in INPUT_FIELDS}
        row.update({field: "" for field in TARGET_FIELDS})
        row["classification"] = "not related"
        rows.append(row)
    return rows


def unwrap_redirect(url: str) -> tuple[str, object]:
    """Resolve common tracking-link query parameters without a network call."""

    parsed = urlsplit(url)
    for _ in range(4):
        query = parse_qs(parsed.query)
        destination = ""
        for key in REDIRECT_QUERY_KEYS:
            values = query.get(key, [])
            if values and values[0].strip().lower().startswith(
                ("http://", "https://", "ftp://")
            ):
                destination = unquote(values[0].strip())
                break
        if not destination or destination == url:
            break
        url = destination
        parsed = urlsplit(url)
    return url, parsed


def actionable_urls(value: str) -> str:
    """Remove Canvas/source-page URLs and retain unique external links."""

    kept: list[str] = []
    raw_value = str(value or "")
    candidates = URL_CANDIDATE_RE.findall(raw_value)
    if not candidates and raw_value.strip():
        candidates = re.split(r"[\r\n|]+", raw_value)
    for raw in candidates:
        url = raw.strip().strip("<>()[]{}.,;:")
        if not url:
            continue
        try:
            url, parsed = unwrap_redirect(url)
        except ValueError:
            continue
        host = (parsed.hostname or "").casefold()
        if (
            host in {"q.utoronto.ca", "canvas.utoronto.ca"}
            or host.endswith(".instructure.com")
            or host.startswith("canvas.")
            or re.search(
                r"/(?:api/v1/|courses/[^/]+/(?:discussion_topics|pages|files|"
                r"assignments|modules|quizzes)(?:/|$))",
                parsed.path,
                re.I,
            )
        ):
            continue
        if parsed.scheme not in {"http", "https", "ftp"} or not host:
            continue
        if url not in kept:
            kept.append(url)
    return "\n".join(kept)


def apply_endpoint_gate(
    row: dict[str, str],
    span: dict[str, str],
    classification: str,
    start_time: str,
    end_time: str,
) -> dict[str, str]:
    """Apply endpoint-derived classification to all dependent output fields."""
    output = {
        field: str(span.get(field, "") or "") for field in TARGET_FIELDS
    }
    output["classification"] = classification
    output["start_time"] = start_time
    output["end_time"] = end_time
    output["item_urls"] = actionable_urls(output["item_urls"])

    # Reviewed topology contract: a semantic notice is anchored at its post
    # time and never carries an end, even when it overrides another endpoint
    # state. This also eliminates notice/endpoint conflicts in exported rows.
    if classification == "notice":
        output["start_time"] = temporal.posted_at_toronto(
            row.get("posted_at", "")
        ).strftime("%Y-%m-%d %H:%M")
        output["end_time"] = ""

    if classification == "not related":
        output["location"] = ""
        output["item_urls"] = ""
        output["extracted_title"] = ""
    else:
        if not output["extracted_title"]:
            output["extracted_title"] = (
                span_training.canonical_announcement_title(row)
            )
        if not output["location"]:
            output["location"] = span_training.fallback_location(
                span_training.clean_text(row.get("body_text"))
            )
    return output


def assert_prediction_topology(
    prediction: dict[str, str],
    row_number: int,
) -> None:
    start_time = prediction["start_time"]
    end_time = prediction["end_time"]
    expected = temporal.derive_classification(start_time, end_time)
    actual = prediction["classification"]
    semantic_notice_exception = actual == "notice" and expected in {
        "not related",
        "deadline",
        "event",
    }
    if actual != expected and not semantic_notice_exception:
        raise AssertionError(
            f"Row {row_number}: classification {actual!r} conflicts with "
            f"start/end topology {expected!r}"
        )

    expected_presence = {
        "not related": (False, False),
        "notice": (True, False),
        "deadline": (False, True),
        "event": (True, True),
    }
    actual_presence = (bool(start_time), bool(end_time))
    if actual_presence != expected_presence[actual] and not semantic_notice_exception:
        raise AssertionError(
            f"Row {row_number}: invalid endpoint presence for {actual!r}"
        )
    if actual == "not related" and (
        prediction["location"] or prediction["extracted_title"]
    ):
        raise AssertionError(
            f"Row {row_number}: non-event title/location was not blanked"
        )


def predict_rows(
    rows: list[dict[str, str]],
    *,
    span_model: Path,
    base_endpoint_model: Path,
    classification_model: Path,
    tree_model: Path,
    logistic_model: Path,
    calibration_path: Path,
    max_length: int,
    batch_size: int,
) -> list[dict[str, str]]:
    if not rows:
        return []

    base_bundle = joblib.load(base_endpoint_model)
    class_bundle = joblib.load(classification_model)
    tree_bundle = joblib.load(tree_model)
    logistic_bundle = joblib.load(logistic_model)
    calibration_bundle = joblib.load(calibration_path)
    calibration = calibration_bundle["selection"]

    frame = pd.DataFrame(rows)
    probabilities = endpoint_probabilities(
        frame,
        base_bundle,
        class_bundle,
        tree_bundle,
        logistic_bundle,
    )
    has_start = calibrated_endpoint_state(
        probabilities,
        calibration,
        "start",
    )
    has_end = calibrated_endpoint_state(
        probabilities,
        calibration,
        "end",
    )
    temporal_values = [
        temporal.extract_endpoints(row, bool(start_state), bool(end_state))
        for row, start_state, end_state in zip(
            rows,
            has_start,
            has_end,
            strict=True,
        )
    ]
    classifications = constrained_classifications(
        probabilities,
        calibration,
        has_start,
        has_end,
    )
    spans = span_predictions(
        rows,
        classifications,
        span_model,
        max_length,
        batch_size,
    )

    predictions: list[dict[str, str]] = []
    for row_number, (row, span, classification, endpoints) in enumerate(
        zip(rows, spans, classifications, temporal_values, strict=True),
        start=2,
    ):
        prediction = apply_endpoint_gate(
            row,
            span,
            classification,
            endpoints[0],
            endpoints[1],
        )
        assert_prediction_topology(prediction, row_number)
        predictions.append(prediction)
    if len(predictions) != len(rows):
        raise AssertionError(
            f"Produced {len(predictions)} predictions for {len(rows)} rows"
        )
    return predictions


def candidate_ids(
    source_rows: list[dict[str, str]],
    source_fields: list[str],
) -> list[str]:
    """Return unique, deterministic identifiers without changing row order."""
    identity_fields = [
        field
        for field in source_fields
        if field not in TARGET_FIELDS and field != CANDIDATE_ID_FIELD
    ]
    generated_counts: dict[str, int] = {}
    used: set[str] = set()
    identifiers: list[str] = []

    for row_number, row in enumerate(source_rows, start=2):
        existing = str(row.get(CANDIDATE_ID_FIELD, "") or "").strip()
        if existing:
            if existing in used:
                raise ValueError(
                    f"Duplicate candidate_id {existing!r} at CSV row "
                    f"{row_number}"
                )
            identifier = existing
        else:
            payload = [
                (field, str(row.get(field, "") or ""))
                for field in identity_fields
            ]
            digest = hashlib.sha256(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:20]
            occurrence = generated_counts.get(digest, 0) + 1
            generated_counts[digest] = occurrence
            suffix = "" if occurrence == 1 else f"_{occurrence}"
            identifier = f"candidate_{digest}{suffix}"
            while identifier in used:
                occurrence += 1
                generated_counts[digest] = occurrence
                identifier = f"candidate_{digest}_{occurrence}"
        used.add(identifier)
        identifiers.append(identifier)
    return identifiers


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        source_fields = list(reader.fieldnames or [])
        if len(source_fields) != len(set(source_fields)):
            raise ValueError("Input CSV contains duplicate column names")
        missing = [
            field for field in INPUT_FIELDS if field not in source_fields
        ]
        if missing:
            raise ValueError(
                "Input is missing required columns: " + ", ".join(missing)
            )
        return source_fields, list(reader)


def write_csv(
    path: Path,
    source_fields: list[str],
    source_rows: list[dict[str, str]],
    predictions: list[dict[str, str]],
) -> None:
    identifiers = candidate_ids(source_rows, source_fields)
    metadata_fields = [
        field
        for field in source_fields
        if field not in CORE_FIELDS and field != CANDIDATE_ID_FIELD
    ]
    output_fields = CORE_FIELDS + [CANDIDATE_ID_FIELD] + metadata_fields

    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=output_fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        for source, prediction, identifier in zip(
            source_rows,
            predictions,
            identifiers,
            strict=True,
        ):
            output = {
                field: source.get(field, "") or "" for field in INPUT_FIELDS
            }
            output.update(prediction)
            output[CANDIDATE_ID_FIELD] = identifier
            output.update(
                {
                    field: source.get(field, "") or ""
                    for field in metadata_fields
                }
            )
            writer.writerow(output)
    os.chmod(path, 0o600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract six calendar-event fields from five announcement input "
            "fields while preserving source metadata."
        )
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--span-model",
        type=Path,
        default=DEFAULT_SPAN_MODEL,
    )
    parser.add_argument(
        "--base-endpoint-model",
        type=Path,
        default=DEFAULT_BASE_ENDPOINT_MODEL,
    )
    parser.add_argument(
        "--classification-model",
        type=Path,
        default=DEFAULT_CLASSIFICATION_MODEL,
    )
    parser.add_argument(
        "--tree-model",
        type=Path,
        default=DEFAULT_TREE_MODEL,
    )
    parser.add_argument(
        "--logistic-model",
        type=Path,
        default=DEFAULT_LOGISTIC_MODEL,
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_fields, source_rows = read_csv(args.input_csv)
    inference_rows = prepare_inference_rows(source_rows)
    predictions = predict_rows(
        inference_rows,
        span_model=args.span_model,
        base_endpoint_model=args.base_endpoint_model,
        classification_model=args.classification_model,
        tree_model=args.tree_model,
        logistic_model=args.logistic_model,
        calibration_path=args.calibration,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    write_csv(
        args.output_csv,
        source_fields,
        source_rows,
        predictions,
    )
    print(f"Wrote {len(predictions)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
