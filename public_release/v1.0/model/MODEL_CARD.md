# Event Extraction Model v1.0

## Purpose

The model converts five fields from a Canvas announcement into six calendar fields.

Inputs: `course_name`, `announcement_title`, `author`, `posted_at`, and `body_text`.

Outputs: `classification`, `location`, `item_urls`, `extracted_title`, `start_time`, and `end_time`.

## Model

The selected hybrid combines a DistilBERT BIO span tagger, word- and character-level TF-IDF classifiers, logistic and ExtraTrees endpoint models, a deterministic temporal parser, and class-time constraints. The class-time rules are:

- start and end present: `event`
- start only: `notice`
- end only: `deadline`
- neither: `not related`

The included artifacts are the frozen models used for the reported v1.0 evaluation. The later experimental final-fit retrain is not included.

## Evaluation

The final 443-row acceptance set contains 327 source groups. Selection used annotation evidence only; model predictions and correctness were not selection inputs. It has no source-group or exact-input overlap with development data.

| Metric | Result |
|---|---:|
| Classification accuracy | 87.81% |
| Classification macro-F1 | 81.82% |
| Start-time exact match | 88.26% |
| Start-time exact match, nonblank rows | 87.00% |
| End-time exact match | 92.33% |
| End-time exact match, nonblank rows | 89.61% |

These are held-out source groups from the same collected corpus, not announcements collected after model development.

## Limitations

Title and physical-location exact match remain weaker than classification and temporal extraction. Predictions should be shown with their source text and confirmed before calendar insertion.

## Privacy

The public dataset uses stable aliases and example domains. Raw Canvas exports and the private alias key are not included.
