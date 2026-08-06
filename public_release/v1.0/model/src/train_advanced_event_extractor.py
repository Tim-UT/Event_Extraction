#!/usr/bin/env python3
"""BIO span utilities used by the released DistilBERT inference pipeline.

The model uses one Transformer token-classification head.  The first token is
supervised with an auxiliary item-type label; all remaining supervised tokens
use BIO labels for title, location, URL, date, and time.  Post-processing turns
the tagged spans into the six spreadsheet target columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
import hybrid_classification


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "outputs" / "event_extraction_refined_20260710" / "augmented_dataset.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "event_extraction_refined_20260710" / "model"
DEFAULT_MODEL = "distilbert-base-uncased"
RANDOM_STATE = 20260710
TARGET_FIELDS = ["classification", "location", "item_urls", "extracted_title", "start_time", "end_time"]
CLASS_NAMES = ["not related", "notice", "deadline", "event"]
ENTITY_NAMES = ["TITLE", "LOCATION", "URL", "DATE", "TIME"]
LABELS = [
    "TYPE_NOT_RELATED",
    "TYPE_NOTICE",
    "TYPE_DEADLINE",
    "TYPE_EVENT",
    "O",
    *[label for entity in ENTITY_NAMES for label in (f"B-{entity}", f"I-{entity}")],
]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
TYPE_LABEL_BY_CLASS = {
    "not related": "TYPE_NOT_RELATED",
    "notice": "TYPE_NOTICE",
    "deadline": "TYPE_DEADLINE",
    "event": "TYPE_EVENT",
}
CLASS_BY_TYPE_LABEL = {value: key for key, value in TYPE_LABEL_BY_CLASS.items()}
SPECIAL_TOKENS = ["[COURSE]", "[ANNOUNCEMENT_TITLE]", "[AUTHOR_FIELD]", "[POSTED_AT]", "[BODY]", "[AUTHOR]", "[EMAIL]", "[PHONE]", "[URL]"]

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)")
DATE_RE = re.compile(
    r"\b(?:today|tomorrow|tonight)\b|"
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b|"
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+20\d{2})?\b|"
    r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
    re.I,
)
TIME_RE = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b|\b(?:[01]?\d|2[0-3]):[0-5]\d\b", re.I)
ROOM_RE = re.compile(
    r"\b(?:EX|BA|SF|MY|MB|GB|MP|MS|UC|NL|SU|WB|HA|MC|HS|PB|PR|BN|LM|OI|KP|SS|RW|RM|RS|FE|ES)\s*[A-Z]?\s*\d{2,4}[A-Z]?\b",
    re.I,
)
ACTIONABLE_URL_EXCLUDE = re.compile(r"(?:api/v1/courses|discussion_topics|/courses/\d+/pages/)", re.I)
STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "for", "from", "has", "have", "in", "is", "of", "on", "or", "the", "to", "will", "with", "your", "our", "this", "that",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[ \t]+", " ", str(value)).strip()


def normalize_for_match(value: Any) -> str:
    text = unicodedata.normalize("NFKC", clean_text(value)).lower()
    text = text.replace("a.m.", "am").replace("p.m.", "pm")
    text = re.sub(r"[^a-z0-9:/#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def excel_datetime(value: Any) -> datetime:
    try:
        return datetime(1899, 12, 30) + timedelta(days=float(value))
    except (TypeError, ValueError):
        parsed = clean_text(value)
        try:
            return datetime.fromisoformat(parsed.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(parsed, fmt)
            except ValueError:
                pass
    return datetime(2025, 1, 1)


def parse_normalized_datetime(value: Any) -> datetime | None:
    match = re.match(r"^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})", clean_text(value))
    if not match:
        return None
    return datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y-%m-%d %H:%M")


def replace_private_values(body: str, author: str) -> tuple[str, list[dict[str, Any]]]:
    """Mask direct identifiers and URLs while retaining reversible URL order."""
    if author:
        body = re.sub(re.escape(author), "[AUTHOR]", body, flags=re.I)
    body = EMAIL_RE.sub("[EMAIL]", body)
    body = PHONE_RE.sub("[PHONE]", body)
    records: list[dict[str, Any]] = []
    chunks: list[str] = []
    cursor = 0
    length = 0
    for match in URL_RE.finditer(body):
        prefix = body[cursor : match.start()]
        chunks.append(prefix)
        length += len(prefix)
        placeholder_start = length
        chunks.append("[URL]")
        length += len("[URL]")
        records.append(
            {
                "start": placeholder_start,
                "end": length,
                "url": match.group(0).rstrip(".,;:"),
            }
        )
        cursor = match.end()
    chunks.append(body[cursor:])
    return "".join(chunks), records


def compose_text(row: dict[str, Any]) -> tuple[str, list[dict[str, Any]], int]:
    body, url_records = replace_private_values(clean_text(row.get("body_text")), clean_text(row.get("author")))
    posted = excel_datetime(row.get("posted_at")).strftime("%Y-%m-%d %H:%M")
    prefix = (
        f"[COURSE] {clean_text(row.get('course_name'))}\n"
        f"[ANNOUNCEMENT_TITLE] {clean_text(row.get('announcement_title'))}\n"
        f"[AUTHOR_FIELD] [AUTHOR]\n"
        f"[POSTED_AT] {posted}\n"
        "[BODY] "
    )
    body_start = len(prefix)
    for record in url_records:
        record["start"] += body_start
        record["end"] += body_start
    return prefix + body, url_records, body_start


def target_dates(row: dict[str, Any]) -> list[datetime]:
    output: list[datetime] = []
    for field in ("start_time", "end_time"):
        parsed = parse_normalized_datetime(row.get(field))
        if parsed and all(parsed.date() != existing.date() for existing in output):
            output.append(parsed)
    return output


def date_span_matches(value: str, targets: list[datetime], posted: datetime) -> bool:
    lower = normalize_for_match(value)
    if not targets:
        return False
    if "tomorrow" in lower:
        return any(target.date() == (posted + timedelta(days=1)).date() for target in targets)
    if "today" in lower or "tonight" in lower:
        return any(target.date() == posted.date() for target in targets)
    weekdays = [
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    ]
    for index, weekday in enumerate(weekdays):
        if weekday in lower:
            return any(target.weekday() == index for target in targets)
    month_names = {
        name: month
        for month, names in {
            1: ("jan", "january"), 2: ("feb", "february"), 3: ("mar", "march"), 4: ("apr", "april"), 5: ("may",), 6: ("jun", "june"), 7: ("jul", "july"), 8: ("aug", "august"), 9: ("sep", "sept", "september"), 10: ("oct", "october"), 11: ("nov", "november"), 12: ("dec", "december"),
        }.items()
        for name in names
    }
    for name, month in month_names.items():
        match = re.search(rf"\b{name}\w*\s+(\d{{1,2}})", lower)
        if match:
            day = int(match.group(1))
            return any(target.month == month and target.day == day for target in targets)
    iso = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", lower)
    if iso:
        year, month, day = map(int, iso.groups())
        return any((target.year, target.month, target.day) == (year, month, day) for target in targets)
    numeric = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", lower)
    if numeric:
        month, day = int(numeric.group(1)), int(numeric.group(2))
        return any(target.month == month and target.day == day for target in targets)
    return False


def parse_time_minutes(value: str, fallback_ampm: str = "") -> int | None:
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?\b", value, re.I)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    ampm = (match.group(3) or fallback_ampm).lower().replace(".", "")
    if hour > 23 or minute > 59:
        return None
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    return hour * 60 + minute


def find_all_exact(text: str, value: str, start_at: int = 0) -> list[tuple[int, int]]:
    value = clean_text(value)
    if not value:
        return []
    return [
        (start_at + match.start(), start_at + match.end())
        for match in re.finditer(re.escape(value), text[start_at:], flags=re.I)
    ]


def fuzzy_word_span(text: str, target: str, start_at: int = 0, threshold: float = 0.52) -> tuple[int, int] | None:
    target_words = [word for word in re.findall(r"[A-Za-z0-9#]+", target) if word.lower() not in STOPWORDS]
    if not target_words:
        return None
    words = list(re.finditer(r"[A-Za-z0-9#]+", text[start_at:]))
    best: tuple[float, int, int] | None = None
    target_norm = normalize_for_match(target)
    expected = max(1, len(target_words))
    for width in range(max(1, expected - 2), expected + 4):
        for index in range(0, len(words) - width + 1):
            begin = words[index].start() + start_at
            end = words[index + width - 1].end() + start_at
            candidate = text[begin:end]
            ratio = SequenceMatcher(None, normalize_for_match(candidate), target_norm).ratio()
            overlap = len(set(normalize_for_match(candidate).split()) & set(target_norm.split())) / max(1, len(set(target_norm.split())))
            score = 0.55 * ratio + 0.45 * overlap
            if best is None or score > best[0]:
                best = (score, begin, end)
    if best and best[0] >= threshold:
        return best[1], best[2]
    return None


def build_gold_spans(
    row: dict[str, Any],
    text: str,
    url_records: list[dict[str, Any]],
    body_start: int,
) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    classification = clean_text(row.get("classification"))
    if classification == "not related":
        # URLs may still be useful targets on non-calendar source rows.
        targets = set(clean_text(row.get("item_urls")).splitlines())
        for record in url_records:
            if record["url"] in targets:
                spans.append((record["start"], record["end"], "URL"))
        return spans

    title = clean_text(row.get("extracted_title"))
    exact_title = find_all_exact(text, title, 0)
    if exact_title:
        spans.append((*exact_title[0], "TITLE"))

    locations = [part.strip() for part in re.split(r"\n|,\s*(?=[A-Za-z])", clean_text(row.get("location"))) if part.strip()]
    for location in locations:
        exact = find_all_exact(text, location, body_start)
        if exact:
            spans.append((*exact[0], "LOCATION"))
            continue
        compact = re.sub(r"\s+", "", location)
        if compact:
            compact_pattern = r"\s*".join(map(re.escape, compact))
            match = re.search(compact_pattern, text[body_start:], re.I)
            if match:
                spans.append((body_start + match.start(), body_start + match.end(), "LOCATION"))
                continue

    target_urls = set(filter(None, clean_text(row.get("item_urls")).splitlines()))
    for record in url_records:
        if record["url"] in target_urls:
            spans.append((record["start"], record["end"], "URL"))

    dates = target_dates(row)
    posted = excel_datetime(row.get("posted_at"))
    title_marker = "[ANNOUNCEMENT_TITLE] "
    title_start = text.find(title_marker) + len(title_marker)
    title_end = text.find("\n", title_start)
    temporal_ranges = [(title_start, title_end), (body_start, len(text))]
    for range_start, range_end in temporal_ranges:
        for match in DATE_RE.finditer(text[range_start:range_end]):
            if date_span_matches(match.group(0), dates, posted):
                spans.append((range_start + match.start(), range_start + match.end(), "DATE"))

    target_minutes = []
    for field in ("start_time", "end_time"):
        value = clean_text(row.get(field))
        parsed = parse_normalized_datetime(value)
        if parsed and "time inferred" not in value.lower():
            target_minutes.append(parsed.hour * 60 + parsed.minute)
    for range_start, range_end in temporal_ranges:
        temporal_text = text[range_start:range_end]
        matches = list(TIME_RE.finditer(temporal_text))
        for match_index, match in enumerate(matches):
            fallback_ampm = ""
            if not re.search(r"\b(?:a\.?m\.?|p\.?m\.?)\b", match.group(0), re.I):
                for later in matches[match_index + 1 :]:
                    if later.start() - match.end() > 24:
                        break
                    marker = re.search(r"\b(a\.?m\.?|p\.?m\.?)\b", later.group(0), re.I)
                    if marker:
                        fallback_ampm = marker.group(1)
                        break
            minute = parse_time_minutes(match.group(0), fallback_ampm)
            if minute in target_minutes:
                spans.append((range_start + match.start(), range_start + match.end(), "TIME"))

    # Higher-priority spans come first when annotations overlap.
    priority = {"URL": 0, "LOCATION": 1, "TITLE": 2, "DATE": 3, "TIME": 4}
    return sorted(spans, key=lambda item: (priority[item[2]], item[0], -(item[1] - item[0])))


@dataclass
class EncodedExample:
    row: dict[str, Any]
    text: str
    url_records: list[dict[str, Any]]
    body_start: int
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    offsets: list[tuple[int, int]]


def encode_row(row: dict[str, Any], tokenizer: Any, max_length: int) -> EncodedExample:
    text, url_records, body_start = compose_text(row)
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    offsets = [tuple(map(int, offset)) for offset in encoded.pop("offset_mapping")]
    spans = build_gold_spans(row, text, url_records, body_start)
    labels = [-100] * len(offsets)
    cls_position = next(index for index, (start, end) in enumerate(offsets) if start == end == 0)
    labels[cls_position] = LABEL2ID[TYPE_LABEL_BY_CLASS[clean_text(row["classification"])]]
    occupied = [False] * len(offsets)
    for span_start, span_end, entity in spans:
        token_positions = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > start and start < span_end and end > span_start and not occupied[index]
        ]
        if not token_positions:
            continue
        for position_index, token_index in enumerate(token_positions):
            prefix = "B" if position_index == 0 else "I"
            labels[token_index] = LABEL2ID[f"{prefix}-{entity}"]
            occupied[token_index] = True
    for index, (start, end) in enumerate(offsets):
        if end > start and labels[index] == -100:
            labels[index] = LABEL2ID["O"]
    return EncodedExample(
        row=row,
        text=text,
        url_records=url_records,
        body_start=body_start,
        input_ids=list(map(int, encoded["input_ids"])),
        attention_mask=list(map(int, encoded["attention_mask"])),
        labels=labels,
        offsets=offsets,
    )


class EventDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.examples = [encode_row(row, tokenizer, max_length) for row in rows]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedExample:
        return self.examples[index]


class Collator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, examples: list[EncodedExample]) -> dict[str, Any]:
        max_len = max(len(example.input_ids) for example in examples)
        input_ids, attention_mask, labels = [], [], []
        for example in examples:
            padding = max_len - len(example.input_ids)
            input_ids.append(example.input_ids + [self.tokenizer.pad_token_id] * padding)
            attention_mask.append(example.attention_mask + [0] * padding)
            labels.append(example.labels + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "examples": examples,
        }


def entity_token_metrics(true_ids: list[list[int]], pred_ids: list[list[int]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    all_true: list[str] = []
    all_pred: list[str] = []
    by_entity: dict[str, tuple[list[int], list[int]]] = {
        entity: ([], []) for entity in ENTITY_NAMES
    }
    for truths, predictions in zip(true_ids, pred_ids, strict=True):
        for true_id, pred_id in zip(truths, predictions, strict=True):
            if true_id < 0:
                continue
            true_label = ID2LABEL[true_id]
            pred_label = ID2LABEL[pred_id]
            if true_label.startswith("TYPE_"):
                continue
            all_true.append(true_label)
            all_pred.append(pred_label)
            for entity, (entity_true, entity_pred) in by_entity.items():
                entity_true.append(int(true_label.endswith(entity)))
                entity_pred.append(int(pred_label.endswith(entity)))
    entity_true_all = [label.split("-", 1)[1] if "-" in label and not label.startswith("TYPE_") else "O" for label in all_true]
    entity_pred_all = [label.split("-", 1)[1] if "-" in label and not label.startswith("TYPE_") else "O" for label in all_pred]
    metrics["entity_micro_f1"] = f1_score(
        entity_true_all,
        entity_pred_all,
        labels=ENTITY_NAMES,
        average="micro",
        zero_division=0,
    )
    metrics["token_accuracy"] = accuracy_score(all_true, all_pred)
    metrics["per_entity_f1"] = {
        entity.lower(): f1_score(y_true, y_pred, zero_division=0)
        for entity, (y_true, y_pred) in by_entity.items()
    }
    return metrics


def class_metrics(true_classes: list[str], pred_classes: list[str]) -> dict[str, Any]:
    report = classification_report(
        true_classes,
        pred_classes,
        labels=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(true_classes, pred_classes),
        "macro_f1": f1_score(true_classes, pred_classes, labels=CLASS_NAMES, average="macro", zero_division=0),
        "per_class_f1": {name: report[name]["f1-score"] for name in CLASS_NAMES},
        "report": report,
    }


def model_loss(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        weight=class_weights,
        ignore_index=-100,
        label_smoothing=0.03,
    )


def decode_type(prediction: list[int], offsets: list[tuple[int, int]]) -> str:
    cls_position = next(index for index, (start, end) in enumerate(offsets) if start == end == 0)
    label = ID2LABEL[prediction[cls_position]]
    return CLASS_BY_TYPE_LABEL.get(label, "not related")


def decode_entity_spans(example: EncodedExample, prediction: list[int]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    current: dict[str, Any] | None = None
    for token_index, label_id in enumerate(prediction[: len(example.offsets)]):
        start, end = example.offsets[token_index]
        label = ID2LABEL[label_id]
        if end <= start or label == "O" or label.startswith("TYPE_"):
            if current:
                output[current["entity"]].append(current)
                current = None
            continue
        prefix, entity = label.split("-", 1)
        if prefix == "B" or current is None or current["entity"] != entity or start > current["end"] + 1:
            if current:
                output[current["entity"]].append(current)
            current = {"entity": entity, "start": start, "end": end}
        else:
            current["end"] = end
    if current:
        output[current["entity"]].append(current)
    for spans in output.values():
        for span in spans:
            span["text"] = example.text[span["start"] : span["end"]].strip()
    return dict(output)


def parse_date_value(value: str, posted: datetime) -> datetime | None:
    lower = normalize_for_match(value)
    if "tomorrow" in lower:
        return posted + timedelta(days=1)
    if "today" in lower or "tonight" in lower:
        return posted
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for weekday_index, weekday in enumerate(weekdays):
        if weekday in lower:
            delta = (weekday_index - posted.weekday()) % 7
            return posted + timedelta(days=delta)
    month_lookup = {
        name: number
        for number, names in {
            1: ("jan", "january"), 2: ("feb", "february"), 3: ("mar", "march"), 4: ("apr", "april"), 5: ("may",), 6: ("jun", "june"), 7: ("jul", "july"), 8: ("aug", "august"), 9: ("sep", "sept", "september"), 10: ("oct", "october"), 11: ("nov", "november"), 12: ("dec", "december"),
        }.items()
        for name in names
    }
    for name, month in month_lookup.items():
        match = re.search(rf"\b{name}\w*\s+(\d{{1,2}})(?:\s+(20\d{{2}}))?", lower)
        if match:
            day = int(match.group(1))
            year = int(match.group(2) or posted.year)
            try:
                candidate = datetime(year, month, day)
            except ValueError:
                return None
            if not match.group(2):
                if (candidate - posted).days > 183:
                    candidate = candidate.replace(year=year - 1)
                elif (posted - candidate).days > 183:
                    candidate = candidate.replace(year=year + 1)
            return candidate
    iso = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", lower)
    if iso:
        try:
            return datetime(*map(int, iso.groups()))
        except ValueError:
            return None
    numeric = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", lower)
    if numeric:
        month, day = int(numeric.group(1)), int(numeric.group(2))
        year = int(numeric.group(3) or posted.year)
        if year < 100:
            year += 2000
        try:
            return datetime(year, month, day)
        except ValueError:
            return None
    return None


def fallback_dates(body: str) -> list[str]:
    return [match.group(0) for match in DATE_RE.finditer(body)]


def fallback_times(body: str) -> list[str]:
    return [match.group(0) for match in TIME_RE.finditer(body)]


def normalized_temporal_output(
    row: dict[str, Any],
    classification: str,
    date_spans: list[str],
    time_spans: list[str],
) -> tuple[str, str]:
    if classification == "not related":
        return "", ""
    posted = excel_datetime(row.get("posted_at"))
    body = clean_text(row.get("body_text"))
    dates = [parse_date_value(value, posted) for value in date_spans]
    dates = [value for value in dates if value]
    if not dates:
        dates = [parse_date_value(value, posted) for value in fallback_dates(body)]
        dates = [value for value in dates if value]
    if not dates:
        return "", ""
    def parse_sequence(values: list[str]) -> list[int]:
        parsed_values: list[int] = []
        for index, value in enumerate(values):
            fallback_ampm = ""
            if not re.search(r"\b(?:a\.?m\.?|p\.?m\.?)\b", value, re.I):
                for later in values[index + 1 :]:
                    marker = re.search(r"\b(a\.?m\.?|p\.?m\.?)\b", later, re.I)
                    if marker:
                        fallback_ampm = marker.group(1)
                        break
            parsed = parse_time_minutes(value, fallback_ampm)
            if parsed is not None:
                parsed_values.append(parsed)
        return parsed_values

    time_values = parse_sequence(time_spans)
    time_values = [value for value in time_values if value is not None]
    if not time_values:
        time_values = parse_sequence(fallback_times(body))

    def combine(date: datetime, minutes: int, inferred: bool = False) -> str:
        value = date.replace(hour=minutes // 60, minute=minutes % 60)
        return value.strftime("%Y-%m-%d %H:%M") + (" (time inferred)" if inferred else "")

    if classification == "notice":
        return combine(dates[0], time_values[0] if time_values else 0, inferred=not time_values), ""
    if classification == "deadline":
        return "", combine(dates[-1], time_values[-1] if time_values else 23 * 60 + 59, inferred=not time_values)
    start_minutes = time_values[0] if time_values else 0
    if len(time_values) >= 2:
        end_minutes = time_values[1]
    else:
        duration_match = re.search(
            r"\b(\d{1,3})\s*[- ]?minutes?\b|\b(\d+(?:\.\d+)?)\s*[- ]?hours?\b",
            body,
            re.I,
        )
        if duration_match:
            duration_minutes = (
                int(duration_match.group(1))
                if duration_match.group(1)
                else round(float(duration_match.group(2)) * 60)
            )
            end_minutes = start_minutes + duration_minutes
        else:
            end_minutes = 23 * 60 + 59 if len(dates) >= 2 else start_minutes
    start = combine(dates[0], start_minutes, inferred=not time_values)
    end_date = dates[1] if len(dates) >= 2 else dates[0]
    if end_minutes >= 24 * 60:
        end_date += timedelta(days=end_minutes // (24 * 60))
        end_minutes %= 24 * 60
    end = combine(end_date, end_minutes, inferred=not time_values)
    return start, end


def clean_predicted_title(value: str, row: dict[str, Any]) -> str:
    value = re.sub(r"^\[[A-Z_]+\]\s*", "", clean_text(value))
    value = re.sub(r"\s+(?:on|at|from)\s+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec).*$", "", value, flags=re.I)
    value = value.strip(" .,:;-–—")
    if len(value) < 3 or len(value) > 110:
        value = clean_text(row.get("announcement_title"))
    return value


def canonical_announcement_title(row: dict[str, Any]) -> str:
    value = clean_text(row.get("announcement_title"))
    value = re.sub(r"^(?:reminder|reminders|update|important|new)\s*[:+\-–—]*\s*", "", value, flags=re.I)
    value = re.sub(
        r"\s*(?:[-–—:&]+\s*)?(?:details?(?:\s*&\s*room assignments?)?|due date|general information|information|reminder)\s*$",
        "",
        value,
        flags=re.I,
    )
    value = re.sub(r"\s+", " ", value).strip(" .,:;-–—")
    course_match = re.search(r"\b([A-Z]{2,4}\s*\d{3})", clean_text(row.get("course_name")))
    if (
        course_match
        and not re.search(re.escape(course_match.group(1).replace(" ", "")), value.replace(" ", ""), re.I)
        and re.search(r"\b(session|lecture|tutorial|exam|test|midterm|office hours?)\b", value, re.I)
    ):
        value = f"{course_match.group(1).replace(' ', '')} {value}"
    return value[:100]


def refine_predicted_title(value: str, row: dict[str, Any]) -> str:
    predicted = clean_predicted_title(value, row)
    candidate = canonical_announcement_title(row)
    predicted_words = normalize_for_match(predicted).split()
    candidate_words = normalize_for_match(candidate).split()
    if candidate and len(candidate_words) <= 9:
        predicted_norm = normalize_for_match(predicted)
        candidate_norm = normalize_for_match(candidate)
        looks_fragmentary = len(predicted_words) < 2 or predicted.lower().startswith(("at ", "up ", "elling "))
        if looks_fragmentary or (predicted_norm and predicted_norm in candidate_norm and len(candidate_words) <= len(predicted_words) + 3):
            return candidate
    return predicted


def fallback_location(body: str) -> str:
    rooms = []
    for match in ROOM_RE.finditer(body):
        value = re.sub(r"\s+", " ", match.group(0)).strip()
        if value not in rooms:
            rooms.append(value)
    online = []
    for platform in ("Zoom", "Microsoft Teams"):
        if re.search(rf"\b{re.escape(platform)}\b", body, re.I):
            online.append(platform)
    return ", ".join(rooms + online)


def structured_prediction(
    example: EncodedExample,
    prediction: list[int],
    classification_override: str | None = None,
) -> dict[str, str]:
    neural_classification = decode_type(prediction, example.offsets)
    classification = classification_override or neural_classification
    span_gate_classification = classification
    spans = decode_entity_spans(example, prediction)
    title_marker = "[ANNOUNCEMENT_TITLE] "
    title_start = example.text.find(title_marker) + len(title_marker)
    title_end = example.text.find("\n", title_start)
    title_values = [
        span["text"]
        for span in spans.get("TITLE", [])
        if (title_start <= span["start"] < title_end) or span["start"] >= example.body_start
    ]
    location_values = [
        span["text"]
        for span in spans.get("LOCATION", [])
        if (title_start <= span["start"] < title_end) or span["start"] >= example.body_start
    ]
    date_values = [span["text"] for span in spans.get("DATE", [])]
    time_values = [span["text"] for span in spans.get("TIME", [])]

    url_values: list[str] = []
    for span in spans.get("URL", []):
        for record in example.url_records:
            if span["start"] < record["end"] and span["end"] > record["start"]:
                if record["url"] not in url_values and not ACTIONABLE_URL_EXCLUDE.search(record["url"]):
                    url_values.append(record["url"])
    title = refine_predicted_title(title_values[0], example.row) if title_values else ""
    if span_gate_classification != "not related" and not title:
        title = canonical_announcement_title(example.row)
    location = ", ".join(dict.fromkeys(clean_text(value) for value in location_values if clean_text(value)))
    start_time, end_time = normalized_temporal_output(example.row, classification, date_values, time_values)
    if span_gate_classification == "not related":
        title = ""
        location = ""
    return {
        "classification": classification,
        "location": location,
        "item_urls": "\n".join(url_values),
        "extracted_title": title,
        "start_time": start_time,
        "end_time": end_time,
    }


def token_set_f1(true_value: str, predicted_value: str) -> float:
    true_tokens = set(normalize_for_match(true_value).split())
    pred_tokens = set(normalize_for_match(predicted_value).split())
    if not true_tokens and not pred_tokens:
        return 1.0
    if not true_tokens or not pred_tokens:
        return 0.0
    overlap = len(true_tokens & pred_tokens)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(true_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def url_set_f1(true_value: str, predicted_value: str) -> float:
    truth = set(filter(None, clean_text(true_value).splitlines()))
    prediction = set(filter(None, clean_text(predicted_value).splitlines()))
    if not truth and not prediction:
        return 1.0
    if not truth or not prediction:
        return 0.0
    overlap = len(truth & prediction)
    precision = overlap / len(prediction)
    recall = overlap / len(truth)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def strict_equal(field: str, truth: str, prediction: str) -> bool:
    if field == "item_urls":
        return set(filter(None, clean_text(truth).splitlines())) == set(filter(None, clean_text(prediction).splitlines()))
    return normalize_for_match(truth) == normalize_for_match(prediction)


def structured_metrics(rows: list[dict[str, Any]], predictions: list[dict[str, str]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in TARGET_FIELDS:
        exact = [strict_equal(field, clean_text(row.get(field)), prediction[field]) for row, prediction in zip(rows, predictions, strict=True)]
        nonempty_positions = [index for index, row in enumerate(rows) if clean_text(row.get(field))]
        presence_truth = [bool(clean_text(row.get(field))) for row in rows]
        presence_pred = [bool(clean_text(prediction[field])) for prediction in predictions]
        field_metrics: dict[str, Any] = {
            "exact_match_accuracy": float(np.mean(exact)),
            "nonempty_exact_match_accuracy": float(np.mean([exact[index] for index in nonempty_positions])) if nonempty_positions else None,
            "presence_f1": f1_score(presence_truth, presence_pred, zero_division=0),
            "target_90_percent_met": float(np.mean(exact)) >= 0.90,
        }
        if field in {"location", "extracted_title"}:
            field_metrics["mean_token_f1"] = float(np.mean([token_set_f1(clean_text(row.get(field)), prediction[field]) for row, prediction in zip(rows, predictions, strict=True)]))
        if field == "item_urls":
            field_metrics["mean_set_f1"] = float(np.mean([url_set_f1(clean_text(row.get(field)), prediction[field]) for row, prediction in zip(rows, predictions, strict=True)]))
        output[field] = field_metrics
    output["full_row_exact_match"] = float(
        np.mean(
            [
                all(strict_equal(field, clean_text(row.get(field)), prediction[field]) for field in TARGET_FIELDS)
                for row, prediction in zip(rows, predictions, strict=True)
            ]
        )
    )
    return output


@torch.no_grad()
def evaluate_model(
    model: Any,
    dataset: EventDataset,
    collator: Collator,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, str]], list[list[int]]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    model.eval()
    true_ids: list[list[int]] = []
    pred_ids: list[list[int]] = []
    true_classes: list[str] = []
    pred_classes: list[str] = []
    structured: list[dict[str, str]] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.cpu()
        predictions = logits.argmax(dim=-1).tolist()
        labels = batch["labels"].tolist()
        for example, prediction, truth in zip(batch["examples"], predictions, labels, strict=True):
            length = len(example.input_ids)
            prediction = prediction[:length]
            truth = truth[:length]
            pred_ids.append(prediction)
            true_ids.append(truth)
            pred_class = decode_type(prediction, example.offsets)
            pred_classes.append(pred_class)
            true_classes.append(clean_text(example.row["classification"]))
            structured.append(structured_prediction(example, prediction))
    metrics = {
        "classification": class_metrics(true_classes, pred_classes),
        "token_extraction": entity_token_metrics(true_ids, pred_ids),
        "structured_fields": structured_metrics([example.row for example in dataset.examples], structured),
    }
    return metrics, structured, pred_ids


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_predictions(
    path: Path,
    dataset: EventDataset,
    structured: list[dict[str, str]],
) -> None:
    fields = ["reviewed_row_id", "source_row", "source_group_id", "dataset_split"]
    for target in TARGET_FIELDS:
        fields.extend([f"true_{target}", f"predicted_{target}", f"correct_{target}"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for example, prediction in zip(dataset.examples, structured, strict=True):
            row = example.row
            output = {field: row.get(field, "") for field in fields[:4]}
            for target in TARGET_FIELDS:
                truth = clean_text(row.get(target))
                predicted = prediction[target]
                output[f"true_{target}"] = truth
                output[f"predicted_{target}"] = predicted
                output[f"correct_{target}"] = strict_equal(target, truth, predicted)
            writer.writerow(output)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--classification-model",
        type=Path,
        default=None,
        help="Validation-selected sparse semantic classifier used for final span gating.",
    )
    args = parser.parse_args()

    seed_everything(RANDOM_STATE)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.data)
    assigned_train_rows = [row for row in rows if row["dataset_split"] == "train"]
    allowed_train_origins = {"reviewed_real", "synthetic_paraphrase"}
    train_rows = [
        row for row in assigned_train_rows if row["row_origin"] in allowed_train_origins
    ]
    validation_rows = [row for row in rows if row["dataset_split"] == "validation" and row["row_origin"] == "reviewed_real"]
    test_rows = [row for row in rows if row["dataset_split"] == "test" and row["row_origin"] == "reviewed_real"]
    if not train_rows or not validation_rows or not test_rows:
        raise RuntimeError(
            "Training, validation, and test must each contain human-reviewed rows"
        )
    if any(row["row_origin"] not in allowed_train_origins for row in train_rows):
        raise RuntimeError("An unsupported row origin leaked into supervised training")
    if any(
        row["row_origin"] != "reviewed_real"
        for row in validation_rows + test_rows
    ):
        raise RuntimeError("A non-reviewed row leaked into validation or test")
    train_groups = {row["source_group_id"] for row in train_rows}
    validation_groups = {row["source_group_id"] for row in validation_rows}
    test_groups = {row["source_group_id"] for row in test_rows}
    overlap = {
        "train_validation": len(train_groups & validation_groups),
        "train_test": len(train_groups & test_groups),
        "validation_test": len(validation_groups & test_groups),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Source-group leakage detected: {overlap}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    print("Encoding datasets...", flush=True)
    train_dataset = EventDataset(train_rows, tokenizer, args.max_length)
    validation_dataset = EventDataset(validation_rows, tokenizer, args.max_length)
    test_dataset = EventDataset(test_rows, tokenizer, args.max_length)
    collator = Collator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )

    device = choose_device()
    model.to(device)
    head_parameters = []
    encoder_parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith("classifier"):
            head_parameters.append(parameter)
        else:
            encoder_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.learning_rate},
            {"params": head_parameters, "lr": args.learning_rate * 5.0},
        ],
        weight_decay=args.weight_decay,
    )
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_steps = max(1, optimizer_steps_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.10)),
        num_training_steps=total_steps,
    )
    weights = torch.ones(len(LABELS), dtype=torch.float32, device=device)
    weights[LABEL2ID["O"]] = 0.02
    for label in TYPE_LABEL_BY_CLASS.values():
        weights[LABEL2ID[label]] = 10.0
    for label in ("B-URL", "I-URL"):
        weights[LABEL2ID[label]] = 5.0
    for label in ("B-LOCATION", "I-LOCATION"):
        weights[LABEL2ID[label]] = 3.0
    for label in ("B-TITLE", "I-TITLE"):
        weights[LABEL2ID[label]] = 1.5

    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_dir = args.output_dir / "best_checkpoint"
    patience = 0
    print(
        json.dumps(
            {
                "device": str(device),
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "test_rows": len(test_rows),
                "train_groups": len(train_groups),
                "excluded_nonreviewed_training_rows": len(assigned_train_rows)
                - len(train_rows),
                "model": args.model_name,
            },
            indent=2,
        ),
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for batch_index, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = model_loss(logits, labels, weights) / args.gradient_accumulation
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation
            should_step = batch_index % args.gradient_accumulation == 0 or batch_index == len(train_loader)
            if should_step:
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if batch_index % 100 == 0:
                print(f"epoch={epoch} batch={batch_index}/{len(train_loader)} loss={running_loss / batch_index:.4f}", flush=True)

        validation_metrics, _, _ = evaluate_model(
            model,
            validation_dataset,
            collator,
            device,
            args.batch_size,
        )
        composite = 0.55 * validation_metrics["classification"]["macro_f1"] + 0.45 * validation_metrics["token_extraction"]["entity_micro_f1"]
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(train_loader)),
            "validation_composite": composite,
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, indent=2), flush=True)
        if composite > best_score + 1e-4:
            best_score = composite
            patience = 0
            if best_dir.exists():
                shutil.rmtree(best_dir)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
        else:
            patience += 1
            if patience >= 2:
                print("Early stopping after two non-improving epochs.", flush=True)
                break

    model = AutoModelForTokenClassification.from_pretrained(best_dir).to(device)
    validation_metrics, validation_structured, validation_ids = evaluate_model(
        model, validation_dataset, collator, device, args.batch_size
    )
    test_metrics, test_structured, test_ids = evaluate_model(
        model, test_dataset, collator, device, args.batch_size
    )
    if args.classification_model is not None:
        classifier_bundle = hybrid_classification.load_model(args.classification_model)
        validation_classes, _ = hybrid_classification.predict_rows(
            classifier_bundle, validation_rows
        )
        test_classes, _ = hybrid_classification.predict_rows(classifier_bundle, test_rows)
        validation_structured = [
            structured_prediction(example, prediction, classification_override=label)
            for example, prediction, label in zip(
                validation_dataset.examples, validation_ids, validation_classes, strict=True
            )
        ]
        test_structured = [
            structured_prediction(example, prediction, classification_override=label)
            for example, prediction, label in zip(
                test_dataset.examples, test_ids, test_classes, strict=True
            )
        ]
        validation_metrics = {
            "classification": class_metrics(
                [row["classification"] for row in validation_rows], validation_classes
            ),
            "token_extraction": validation_metrics["token_extraction"],
            "structured_fields": structured_metrics(validation_rows, validation_structured),
        }
        test_metrics = {
            "classification": class_metrics(
                [row["classification"] for row in test_rows], test_classes
            ),
            "token_extraction": test_metrics["token_extraction"],
            "structured_fields": structured_metrics(test_rows, test_structured),
        }
    write_predictions(args.output_dir / "validation_predictions.csv", validation_dataset, validation_structured)
    write_predictions(args.output_dir / "test_predictions.csv", test_dataset, test_structured)

    metrics = {
        "model_name": args.model_name,
        "architecture": "single ModernBERT token-classification head with BIO spans and auxiliary type label on first token",
        "random_state": RANDOM_STATE,
        "device": str(device),
        "max_length": args.max_length,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "train_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "test_groups": len(test_groups),
        "excluded_nonreviewed_training_rows": len(assigned_train_rows)
        - len(train_rows),
        "synthetic_train_rows": sum(row["row_origin"] == "synthetic_paraphrase" for row in train_rows),
        "group_overlap": overlap,
        "group_overlap_count": sum(overlap.values()),
        "best_validation_composite": best_score,
        "validation": validation_metrics,
        "test": test_metrics,
        "target_definition": "strict normalized exact-match accuracy >= 0.90 for each spreadsheet target; proposal-standard token F1 is reported separately",
        "training_history": history,
        "classification_source": (
            str(args.classification_model) if args.classification_model else "ModernBERT auxiliary TYPE token"
        ),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "label_maps.json").write_text(
        json.dumps({"labels": LABELS, "label2id": LABEL2ID, "id2label": ID2LABEL}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"best_validation_composite": best_score, "test": test_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
