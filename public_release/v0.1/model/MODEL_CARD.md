# Event Extraction Model

This model converts five Canvas announcement fields into six calendar-item fields.

Inputs: `course_name`, `announcement_title`, `author`, `posted_at`, and `body_text`.

Outputs: `classification`, `location`, `item_urls`, `extracted_title`, `start_time`, and `end_time`.

## Architecture

The model combines a ModernBERT token tagger for semantic spans with sparse temporal-presence models and a calibrated endpoint ensemble. A deterministic constraint derives the classification from the normalized endpoints:

- start and end present: `event`
- start only: `notice`
- end only: `deadline`
- neither: `not related`

## Data

Training used `announcement_item_marks_public_v0.1.csv`, which contains 821 pseudonymized, human-reviewed rows. The group-aware split contains 587 training rows, 116 validation rows, and 118 test rows with no source-group overlap.

The release contains the pseudonymized dataset, model source, documentation, and deployable artifacts.

## Held-out test results

The complete hybrid pipeline was evaluated once on the 118-row held-out test split. Accuracy uses normalized exact matching; URL order is ignored.

| Output field | Exact-match accuracy |
|---|---:|
| Classification | 75.42% |
| Location | 49.15% |
| Item URLs | 97.46% |
| Extracted title | 28.81% |
| Start time | 75.42% |
| End time | 69.49% |

All predicted classifications follow the start/end constraint. See `hybrid_test_metrics.json` for non-empty accuracy, presence F1, token/set similarity, and full-row accuracy.

## Endpoint-presence results

The validation-selected endpoint ensemble achieved the following exact presence results:

| Split | Classification | Start-time presence | End-time presence |
|---|---:|---:|---:|
| Validation | 81.03% | 90.52% | 88.79% |
| Held-out test | 75.42% | 78.81% | 85.59% |

The project target is 90% for each field. Item URLs meet the target in this release; the other fields require more data and model improvement. See `metrics.json` for the endpoint results and validation protocol.

## Use

Use Python 3.10 or later, install the pinned dependencies, then run:

```bash
python -m pip install -r requirements.txt
python src/predict_event_extractor.py input.csv output.csv
```

The input CSV must contain all five input fields. The output preserves those fields and appends the six predicted fields.

## Upstream model and license

The semantic span checkpoint is fine-tuned from `answerdotai/ModernBERT-base`. ModernBERT is distributed under the Apache License 2.0. See `THIRD_PARTY_NOTICES.md` and `MODERNBERT_LICENSE.txt`.
