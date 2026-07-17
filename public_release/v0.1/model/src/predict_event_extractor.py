#!/usr/bin/env python3
"""Predict the six target fields with the hybrid event-extraction model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

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
INPUT_FIELDS = ["course_name", "announcement_title", "author", "posted_at", "body_text"]
TARGET_FIELDS = ["classification", "location", "item_urls", "extracted_title", "start_time", "end_time"]


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def endpoint_probabilities(
    frame: pd.DataFrame,
    base_bundle: dict,
    class_bundle: dict,
    tree_bundle: dict,
    logistic_bundle: dict,
) -> dict[str, np.ndarray]:
    features = stacker.engineered_features(frame, base_bundle, class_bundle)
    base_matrix = base_bundle["vectorizer"].transform(presence.compose_text(frame))
    _, class_probabilities = hybrid_classification.predict_rows(class_bundle, frame.to_dict("records"))
    positions = {label: index for index, label in enumerate(class_bundle["labels"])}
    return {
        "tree_start": tree_bundle["start_model"].predict_proba(features)[:, 1],
        "tree_end": tree_bundle["end_model"].predict_proba(features)[:, 1],
        "logistic_start": logistic_bundle["start_model"].predict_proba(features)[:, 1],
        "logistic_end": logistic_bundle["end_model"].predict_proba(features)[:, 1],
        "base_start": sigmoid(base_bundle["models"]["start"].decision_function(base_matrix)),
        "base_end": sigmoid(base_bundle["models"]["end"].decision_function(base_matrix)),
        "direct_start": class_probabilities[:, positions["notice"]] + class_probabilities[:, positions["event"]],
        "direct_end": class_probabilities[:, positions["deadline"]] + class_probabilities[:, positions["event"]],
    }


def calibrated_endpoint_state(
    values: dict[str, np.ndarray],
    calibration: dict,
    endpoint: str,
) -> np.ndarray:
    sources = ["tree", "logistic", "base", "direct"]
    matrix = np.column_stack([values[f"{source}_{endpoint}"] for source in sources])
    selection = calibration[endpoint]
    scores = matrix @ np.asarray(selection["weights"], dtype=float)
    return scores >= float(selection["threshold"])


def span_predictions(
    rows: list[dict[str, str]],
    classifications: list[str],
    model_dir: Path,
    max_length: int,
    batch_size: int,
):
    if len(rows) != len(classifications):
        raise ValueError("Each row must have one endpoint-derived classification.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    dataset = span_training.EventDataset(rows, tokenizer, max_length)
    collator = span_training.Collator(tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    device = span_training.choose_device()
    model.to(device).eval()
    output = []
    row_index = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits.cpu()
            predicted_ids = logits.argmax(dim=-1).tolist()
            for example, predicted in zip(batch["examples"], predicted_ids, strict=True):
                output.append(
                    span_training.structured_prediction(
                        example,
                        predicted[: len(example.input_ids)],
                        classification_override=classifications[row_index],
                    )
                )
                row_index += 1
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--span-model",
        type=Path,
        default=MODEL_ROOT / "span_model",
    )
    parser.add_argument(
        "--base-endpoint-model",
        type=Path,
        default=ARTIFACT_ROOT / "constrained_temporal_presence.joblib",
    )
    parser.add_argument(
        "--classification-model",
        type=Path,
        default=ARTIFACT_ROOT / "classification_ensemble.joblib",
    )
    parser.add_argument(
        "--tree-model",
        type=Path,
        default=ARTIFACT_ROOT / "tree_temporal_stacker.joblib",
    )
    parser.add_argument(
        "--logistic-model",
        type=Path,
        default=ARTIFACT_ROOT / "temporal_state_stacker.joblib",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=ARTIFACT_ROOT / "endpoint_ensemble_calibration.joblib",
    )
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    with args.input_csv.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    missing = [field for field in INPUT_FIELDS if source_rows and field not in source_rows[0]]
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")
    rows = []
    for source in source_rows:
        row = {field: source.get(field, "") for field in INPUT_FIELDS}
        row.update({field: "" for field in TARGET_FIELDS})
        row["classification"] = "not related"
        rows.append(row)

    base_bundle = joblib.load(args.base_endpoint_model)
    class_bundle = joblib.load(args.classification_model)
    tree_bundle = joblib.load(args.tree_model)
    logistic_bundle = joblib.load(args.logistic_model)
    calibration = joblib.load(args.calibration)["selection"]
    frame = pd.DataFrame(rows)
    probabilities = endpoint_probabilities(
        frame,
        base_bundle,
        class_bundle,
        tree_bundle,
        logistic_bundle,
    )
    has_start = calibrated_endpoint_state(probabilities, calibration, "start")
    has_end = calibrated_endpoint_state(probabilities, calibration, "end")
    temporal_predictions = []
    for row, start_state, end_state in zip(rows, has_start, has_end, strict=True):
        start_time, end_time = temporal.extract_endpoints(row, bool(start_state), bool(end_state))
        temporal_predictions.append(
            {
                "classification": temporal.derive_classification(start_time, end_time),
                "start_time": start_time,
                "end_time": end_time,
            }
        )
    spans = span_predictions(
        rows,
        [prediction["classification"] for prediction in temporal_predictions],
        args.span_model,
        args.max_length,
        args.batch_size,
    )

    predictions = []
    for span, temporal_prediction in zip(spans, temporal_predictions, strict=True):
        predictions.append(
            {
                "classification": temporal_prediction["classification"],
                "location": span["location"],
                "item_urls": span["item_urls"],
                "extracted_title": span["extracted_title"],
                "start_time": temporal_prediction["start_time"],
                "end_time": temporal_prediction["end_time"],
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INPUT_FIELDS + TARGET_FIELDS)
        writer.writeheader()
        for source, prediction in zip(source_rows, predictions, strict=True):
            writer.writerow({**{field: source.get(field, "") for field in INPUT_FIELDS}, **prediction})


if __name__ == "__main__":
    main()
