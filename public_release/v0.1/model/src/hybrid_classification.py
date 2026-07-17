"""Utilities for the validation-selected sparse classification ensemble."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


DEFAULT_MODEL = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "classification_ensemble.joblib"
)


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def load_model(path: Path = DEFAULT_MODEL) -> dict[str, Any]:
    return joblib.load(path)


def predict_rows(bundle: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    frame = pd.DataFrame(rows)
    labels = list(bundle["labels"])
    probabilities = []
    for feature in ("word", "char", "combined"):
        component = bundle["models"][feature]
        matrix = component["vectorizer"].transform(frame[bundle["input_columns"]])
        classifier = component["classifier"]
        scores = classifier.decision_function(matrix)
        positions = [list(classifier.classes_).index(label) for label in labels]
        probabilities.append(_softmax(scores[:, positions]))
    combined = sum(
        weight * values
        for weight, values in zip(bundle["weights"], probabilities, strict=True)
    )
    predictions = np.array(labels)[combined.argmax(axis=1)].tolist()
    return predictions, combined
