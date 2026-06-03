# Event Extraction from Canvas Announcements

This project builds a simple deep learning pipeline to extract calendar-related information from Canvas course announcements and assignment notifications.

## Goal

The system takes an anonymized Canvas announcement and extracts calendar fields such as:

- event or deadline title
- date
- start time
- end time
- location
- contact person
- course code

The extracted information can then be converted into structured JSON, `.ics`, or calendar CSV output.

## Model Plan

The project uses a single Transformer sequence-labeling model. The model predicts token labels such as `DATE`, `TIME`, `LOCATION`, `CONTACT_NAME`, and `COURSE_CODE`. Post-processing then groups the predicted spans into calendar items and derives the announcement type.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── anonymize_announcements.py
├── output_data_structure.md
├── project_flow_map.md
├── simplified_main_flow.svg
├── sample announcement.rtf
├── blurried sample announcement.rtf
├── data/
│   ├── README.md
│   └── sample_announcements.csv
├── notebooks/
│   └── README.md
└── outputs/
    └── .gitkeep
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the anonymization demo on a plain text file:

```bash
python anonymize_announcements.py path/to/announcements.txt
```

## Current Demo Example

The sample announcement includes a rescheduled tutorial session. Sensitive values are anonymized, for example:

```text
PRA0106 -> AAI2617
Reza Rahmati -> Denver Denrin
MY430 -> PY512
```

The expected structured output includes:

```json
{
  "title": "Alternative TA session",
  "start": "Nov 27, 10:00 AM",
  "end": "Nov 27, 12:00 PM",
  "location": "PY512",
  "contact": "Denver Denrin"
}
```

## Notes

Raw Canvas announcements may contain private information. Before training, names, emails, Zoom links, room numbers, and course identifiers should be anonymized.

