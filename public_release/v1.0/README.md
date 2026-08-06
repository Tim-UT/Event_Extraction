# Event Extraction v1.0

This folder contains the finalized public artifacts for the APS360 project:

- `dataset/announcement_item_marks_public_v1.0.csv`: privacy-mosaiced, reviewed dataset with fixed train, validation, and test assignments.
- `model/`: selected frozen hybrid model, inference code, dependencies, and evaluation metrics.
- `report/Event_Extraction_Final_Report_v1.0.pdf`: final project report.

Raw Canvas exports, alias keys, review workbooks, intermediate predictions, experimental retrains, and development logs are intentionally excluded.

The public dataset contains 6,060 rows. Eleven restricted or biographical rows were removed from the local v1.0 corpus before publication; labels and split assignments for all retained rows are unchanged.

## Run inference

```bash
python -m pip install -r model/requirements.txt
python model/src/predict_event_extractor.py input.csv output.csv
```

The input CSV must contain `course_name`, `announcement_title`, `author`, `posted_at`, and `body_text`. The output adds `classification`, `location`, `item_urls`, `extracted_title`, `start_time`, and `end_time`.

The Transformer checkpoint is stored with Git LFS.
