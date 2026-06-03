# Output Data Structure for Canvas Announcement Event Extraction

This file defines the project output schema. The idea is to keep **one internal JSON structure** that contains all useful extracted information, then convert it into `.ics`, Google Calendar CSV, Outlook CSV, or app/database tables.

The structure is not one flat list of duplicated fields. Instead, related fields are grouped into objects:

- `announcement`: information about the original Canvas announcement
- `summary`: our model's announcement-level prediction
- `items`: zero, one, or multiple extracted calendar items
- `time`: all start/end/due/recurrence information
- `people`: organizer, contact, and attendees
- `calendar_options`: reminders, privacy, availability, status, and export behavior
- `evidence`: source text spans and confidence scores

## Why This Structure

One announcement can have no calendar item, one deadline, one event, or multiple events/deadlines. Therefore, the model should output **one announcement record** with an `items` array.

```text
Canvas announcement
  -> announcement-level type
  -> 0..n extracted calendar items
  -> export each item as one calendar event/task if needed
```

## Complete Internal JSON Example

```json
{
  "schema_version": "1.0",
  "announcement": {
    "announcement_id": "ann_000001",
    "source_system": "Canvas",
    "announcement_link": "https://canvas.example.edu/courses/COURSE_ID/discussion_topics/ANN_ID",
    "course_id": "course_001",
    "course_code": "BIQ2762",
    "course_name": null,
    "title": "Midterm Grades Released",
    "body_text": "The grades for BIQ2762 midterm have been released...",
    "posted_at": {
      "date": "2026-03-01",
      "time": "14:30:00",
      "datetime": "2026-03-01T14:30:00-05:00",
      "timestamp": 1772393400,
      "timezone": "America/Toronto"
    },
    "author_name": "Minren",
    "author_email": null
  },
  "summary": {
    "announcement_type": "deadline",
    "contains_calendar_item": true,
    "calendar_item_count": 1,
    "counts_by_type": {
      "event": 0,
      "deadline": 1,
      "notice": 0,
      "reschedule": 0,
      "cancellation": 0,
      "recurring_event": 0,
      "recruitment_event": 0
    },
    "model_confidence": 0.91
  },
  "items": [
    {
      "item_id": "ann_000001_item_01",
      "item_type": "deadline",
      "title": "Midterm regrade request",
      "description": "Send the regrade request through email before the deadline.",
      "action": "send regrade request",
      "course_code": "BIQ2762",

      "time": {
        "raw_text": "before the end of March 9",
        "is_all_day": false,
        "timezone": "America/Toronto",
        "start": null,
        "end": null,
        "due": {
          "date": "2026-03-09",
          "time": "23:59:00",
          "datetime": "2026-03-09T23:59:00-05:00",
          "timestamp": 1773118740
        },
        "duration_minutes": null,
        "date_is_inferred": true,
        "time_is_inferred": true,
        "inference_note": "Phrase 'end of March 9' is normalized as 23:59 local time."
      },

      "recurrence": {
        "is_recurring": false,
        "frequency": null,
        "interval": null,
        "by_day": [],
        "until": null,
        "count": null,
        "rrule": null,
        "exceptions": []
      },

      "location": {
        "location_text": null,
        "building": null,
        "room": null,
        "address": null,
        "geo": {
          "latitude": null,
          "longitude": null
        },
        "online_meeting": {
          "is_online": false,
          "provider": null,
          "url": null,
          "meeting_id": null,
          "passcode": null
        }
      },

      "people": {
        "organizer": {
          "name": null,
          "email": null
        },
        "contact": {
          "name": "Cortavvor",
          "email": "user55138@ham.fp"
        },
        "attendees": []
      },

      "calendar_options": {
        "status": "confirmed",
        "privacy": "private",
        "availability": "busy",
        "priority": "normal",
        "categories": ["course", "deadline", "regrade"],
        "reminders": [
          {
            "method": "popup",
            "minutes_before": 1440
          }
        ],
        "attachments": [],
        "resources": [],
        "export_as": "deadline_as_event"
      },

      "ids_and_audit": {
        "uid": "ann_000001_item_01@timegrid.local",
        "created_at": "2026-06-02T16:00:00-04:00",
        "updated_at": "2026-06-02T16:00:00-04:00",
        "sequence": 0,
        "source_hash": "md5_or_sha256_hash_here"
      },

      "evidence": {
        "source_text_span": {
          "start_char": 312,
          "end_char": 430,
          "text": "before the end of March 9 with BIQ2762 Midterm Regrade"
        },
        "field_confidence": {
          "item_type": 0.94,
          "title": 0.86,
          "due": 0.92,
          "contact_email": 0.99
        }
      }
    }
  ]
}
```

## Announcement Type Labels

These are labels for the whole announcement.

```text
not_relevant
notice
deadline
event
multiple
reschedule
cancellation
recurring_event
office_hour
exam
quiz
assignment
lecture
lab
tutorial
meeting
recruitment_event
grade_release
course_logistics
unknown
```

Notes:

- `recurring_event` covers weekly office hours, repeated tutorials, recurring lectures, repeated study sessions, etc.
- `recruitment_event` is included for announcements about job fairs, lab recruitment, club recruiting, research participation, volunteer calls, or campus recruiting sessions.
- `multiple` can be used when one announcement contains more than one major calendar item type.

## Calendar Item Type Labels

These are labels for each item inside `items`.

```text
event
deadline
notice
reschedule
cancellation
recurring_event
office_hour
exam
quiz
assignment
lecture
lab
tutorial
meeting
recruitment_event
application_deadline
interview
workshop
seminar
presentation
study_session
reading
grade_release
unknown
```

## Field Dictionary

### Required Top-Level Fields

| Field | Description |
|---|---|
| `schema_version` | Version of this output format. |
| `announcement` | Original Canvas announcement metadata and text. |
| `summary` | Announcement-level model prediction and item counts. |
| `items` | List of extracted calendar items. Empty list means no exportable item. |

### Announcement Object

| Field | Description |
|---|---|
| `announcement_id` | Our unique ID for the announcement. |
| `source_system` | Usually `Canvas`. |
| `announcement_link` | Link back to the Canvas announcement. |
| `course_id` | Canvas/internal course ID if available. |
| `course_code` | Course code, anonymized if needed. |
| `course_name` | Course title if available. |
| `title` | Announcement title. |
| `body_text` | Announcement body text. May be anonymized. |
| `posted_at` | Nested date/time/datetime/timestamp object for posting time. |
| `author_name` | Announcement author or sender, anonymized if needed. |
| `author_email` | Author email, anonymized if needed. |

### Summary Object

| Field | Description |
|---|---|
| `announcement_type` | Main class predicted for the whole announcement. |
| `contains_calendar_item` | Boolean. |
| `calendar_item_count` | Total number of extracted items. |
| `counts_by_type` | Counts for item types, e.g. deadlines/events/recurring events. |
| `model_confidence` | Confidence for the announcement-level label. |

### Item Core Fields

| Field | Description |
|---|---|
| `item_id` | Unique ID for the extracted item. |
| `item_type` | Type of this item, e.g. `event`, `deadline`, `recurring_event`. |
| `title` | Calendar title. Exports to ICS `SUMMARY` and CSV `Subject`. |
| `description` | Human-readable note. Exports to ICS/CSV description. |
| `action` | What the student should do, e.g. submit, attend, register, email. |
| `course_code` | Related course code. |

## Time Object

All time-related information goes here. This avoids duplicate-looking fields.

```json
"time": {
  "raw_text": "Wednesday, Nov 27, 10 am-12 pm",
  "is_all_day": false,
  "timezone": "America/Toronto",
  "start": {
    "date": "2026-11-27",
    "time": "10:00:00",
    "datetime": "2026-11-27T10:00:00-05:00",
    "timestamp": 1795791600
  },
  "end": {
    "date": "2026-11-27",
    "time": "12:00:00",
    "datetime": "2026-11-27T12:00:00-05:00",
    "timestamp": 1795798800
  },
  "due": null,
  "duration_minutes": 120,
  "date_is_inferred": false,
  "time_is_inferred": false,
  "inference_note": null
}
```

| Field | Description |
|---|---|
| `raw_text` | Original date/time phrase from the announcement. |
| `is_all_day` | True when only a date is known. |
| `timezone` | Local timezone used for normalization. |
| `start` | Nested date/time/datetime/timestamp for start. |
| `end` | Nested date/time/datetime/timestamp for end. |
| `due` | Nested date/time/datetime/timestamp for a deadline. |
| `duration_minutes` | Duration if known or inferred. |
| `date_is_inferred` | True if date was inferred from context. |
| `time_is_inferred` | True if time was defaulted/inferred. |
| `inference_note` | Explanation of normalization choices. |

## Recurrence Object

Recurring/repeated events should not be missing. This object supports weekly office hours, repeated lectures, tutorial sessions, and similar cases.

```json
"recurrence": {
  "is_recurring": true,
  "frequency": "weekly",
  "interval": 1,
  "by_day": ["MO", "WE"],
  "until": {
    "date": "2026-08-03",
    "time": "23:59:00",
    "datetime": "2026-08-03T23:59:00-04:00",
    "timestamp": 1785815940
  },
  "count": null,
  "rrule": "FREQ=WEEKLY;INTERVAL=1;BYDAY=MO,WE;UNTIL=20260804T035900Z",
  "exceptions": [
    {
      "date": "2026-06-22",
      "reason": "Course study break"
    }
  ]
}
```

| Field | Description |
|---|---|
| `is_recurring` | Whether the item repeats. |
| `frequency` | `daily`, `weekly`, `monthly`, `yearly`, or custom. |
| `interval` | Repeat interval, e.g. every 2 weeks. |
| `by_day` | Days of week, e.g. `MO`, `TU`, `WE`. |
| `until` | End date/time of recurrence. |
| `count` | Number of occurrences if known. |
| `rrule` | iCalendar recurrence rule. |
| `exceptions` | Cancelled/skipped dates or changed occurrences. |

## Location Object

```json
"location": {
  "location_text": "PY512",
  "building": "PY",
  "room": "512",
  "address": null,
  "geo": {
    "latitude": null,
    "longitude": null
  },
  "online_meeting": {
    "is_online": false,
    "provider": null,
    "url": null,
    "meeting_id": null,
    "passcode": null
  }
}
```

Useful fields:

- `location_text`
- `building`
- `room`
- `address`
- `geo.latitude`
- `geo.longitude`
- `online_meeting.is_online`
- `online_meeting.provider`
- `online_meeting.url`
- `online_meeting.meeting_id`
- `online_meeting.passcode`

## People Object

```json
"people": {
  "organizer": {
    "name": "Course Team",
    "email": null
  },
  "contact": {
    "name": "Denver Denrin",
    "email": null
  },
  "attendees": [
    {
      "name": "Student Group",
      "email": null,
      "role": "required",
      "rsvp_required": false,
      "response_status": "needs_action"
    }
  ]
}
```

Useful fields:

- `organizer`
- `contact`
- `attendees`
- `attendees.role`: `required`, `optional`, `resource`
- `attendees.rsvp_required`
- `attendees.response_status`: `needs_action`, `accepted`, `declined`, `tentative`

## Calendar Options Object

```json
"calendar_options": {
  "status": "confirmed",
  "privacy": "private",
  "availability": "busy",
  "priority": "normal",
  "categories": ["course", "tutorial"],
  "reminders": [
    {
      "method": "popup",
      "minutes_before": 60
    },
    {
      "method": "email",
      "minutes_before": 1440
    }
  ],
  "attachments": [
    {
      "title": "Assignment handout",
      "url": "https://canvas.example.edu/files/FILE_ID",
      "mime_type": "application/pdf"
    }
  ],
  "resources": ["projector", "lab computer"],
  "export_as": "event"
}
```

Useful fields:

- `status`: `confirmed`, `tentative`, `cancelled`, `needs_action`
- `privacy`: `public`, `private`, `confidential`
- `availability`: `busy`, `free`, `tentative`, `out_of_office`
- `priority`: `low`, `normal`, `high`
- `categories`
- `reminders`
- `attachments`
- `resources`
- `export_as`: `event`, `deadline_as_event`, `todo`, `skip`

## IDs and Audit Object

```json
"ids_and_audit": {
  "uid": "ann_000002_item_01@timegrid.local",
  "created_at": "2026-06-02T16:00:00-04:00",
  "updated_at": "2026-06-02T16:00:00-04:00",
  "sequence": 0,
  "source_hash": "md5_or_sha256_hash_here"
}
```

Useful for deduplication, update handling, and `.ics` export:

- `uid`
- `created_at`
- `updated_at`
- `sequence`
- `source_hash`

## Evidence Object

```json
"evidence": {
  "source_text_span": {
    "start_char": 120,
    "end_char": 188,
    "text": "Wednesday, Nov 27, 10 am-12 pm"
  },
  "field_confidence": {
    "item_type": 0.96,
    "title": 0.88,
    "time.start": 0.93,
    "time.end": 0.93,
    "location.room": 0.97
  }
}
```

This is useful for debugging, model evaluation, and explaining why the system created a calendar item.

## Mapping to `.ics`

`.ics` is the richest export target. Each `items[]` entry can become a `VEVENT` or, for deadlines, optionally a `VTODO`.

| Internal Field | ICS Property |
|---|---|
| `ids_and_audit.uid` | `UID` |
| export time | `DTSTAMP` |
| `ids_and_audit.created_at` | `CREATED` |
| `ids_and_audit.updated_at` | `LAST-MODIFIED` |
| `ids_and_audit.sequence` | `SEQUENCE` |
| `time.start.datetime` | `DTSTART` |
| `time.end.datetime` | `DTEND` |
| `time.due.datetime` | `DUE` for `VTODO`, or `DTSTART` for deadline-as-event |
| `time.duration_minutes` | `DURATION` |
| `title` | `SUMMARY` |
| `description` | `DESCRIPTION` |
| `location.location_text` | `LOCATION` |
| `location.geo` | `GEO` |
| `location.online_meeting.url` or `announcement.announcement_link` | `URL` |
| `people.organizer` | `ORGANIZER` |
| `people.attendees` | `ATTENDEE` |
| `calendar_options.status` | `STATUS` |
| `calendar_options.privacy` | `CLASS` |
| `calendar_options.availability` | `TRANSP` |
| `calendar_options.priority` | `PRIORITY` |
| `calendar_options.categories` | `CATEGORIES` |
| `calendar_options.attachments` | `ATTACH` |
| `calendar_options.resources` | `RESOURCES` |
| `calendar_options.reminders` | `VALARM` |
| `recurrence.rrule` | `RRULE` |
| `recurrence.exceptions` | `EXDATE` or modified occurrences |

## Mapping to Google Calendar CSV

Google Calendar CSV is flatter than our internal structure. Export one item as one row.

```csv
Subject,Start Date,Start Time,End Date,End Time,All Day Event,Description,Location,Private
```

| Internal Field | Google CSV Field |
|---|---|
| `title` | `Subject` |
| `time.start.date` or `time.due.date` | `Start Date` |
| `time.start.time` or `time.due.time` | `Start Time` |
| `time.end.date`, `time.start.date`, or `time.due.date` | `End Date` |
| `time.end.time` | `End Time` |
| `time.is_all_day` | `All Day Event` |
| `description` + source link/contact | `Description` |
| `location.location_text` or `location.online_meeting.url` | `Location` |
| `calendar_options.privacy == "private"` | `Private` |

CSV cannot fully preserve recurrence, attendees, reminders, attachments, or online meeting metadata. Use `.ics` for those.

## Mapping to Outlook CSV

Common Outlook CSV headers:

```csv
Subject,Start Date,Start Time,End Date,End Time,All day event,Reminder on/off,Reminder Date,Reminder Time,Categories,Description,Location,Private
```

| Internal Field | Outlook CSV Field |
|---|---|
| `title` | `Subject` |
| `time.start.date` or `time.due.date` | `Start Date` |
| `time.start.time` or `time.due.time` | `Start Time` |
| `time.end.date`, `time.start.date`, or `time.due.date` | `End Date` |
| `time.end.time` | `End Time` |
| `time.is_all_day` | `All day event` |
| `calendar_options.reminders` | `Reminder on/off`, `Reminder Date`, `Reminder Time` |
| `calendar_options.categories` | `Categories` |
| `description` + source link/contact | `Description` |
| `location.location_text` | `Location` |
| `calendar_options.privacy == "private"` | `Private` |

Again, `.ics` is better than CSV for recurrence, attendees, online meeting details, and exceptions.

## Recommended Token Labels for the Model

Announcement-level labels are handled by the classification head. Token labels are used for extraction.

```text
O
B-TITLE, I-TITLE
B-ACTION, I-ACTION
B-DATE, I-DATE
B-TIME, I-TIME
B-DATETIME, I-DATETIME
B-DURATION, I-DURATION
B-DEADLINE, I-DEADLINE
B-RECURRENCE, I-RECURRENCE
B-LOCATION, I-LOCATION
B-BUILDING, I-BUILDING
B-ROOM, I-ROOM
B-ONLINE_LINK, I-ONLINE_LINK
B-MEETING_ID, I-MEETING_ID
B-PASSCODE, I-PASSCODE
B-COURSE_CODE, I-COURSE_CODE
B-CONTACT_NAME, I-CONTACT_NAME
B-CONTACT_EMAIL, I-CONTACT_EMAIL
B-ORGANIZER, I-ORGANIZER
B-ATTENDEE, I-ATTENDEE
B-STATUS, I-STATUS
B-RESOURCE, I-RESOURCE
```

## Export Rules

1. Keep this internal JSON as the source of truth.
2. Export one calendar item per `items[]` object.
3. If `items` is empty, do not create a calendar file row/event.
4. If `time.start` and `time.end` exist, export as an event.
5. If only `time.due` exists, export as `deadline_as_event` or `VTODO`.
6. If only a date exists, set `time.is_all_day = true`.
7. If a deadline has no time, default to `23:59` and mark `time_is_inferred = true`.
8. If a timed event has no end time, infer a default duration such as 60 minutes and mark it in `inference_note`.
9. If recurrence is detected, fill `recurrence` and prefer `.ics` export.
10. Always include `announcement_id` and `announcement_link` in the exported description for traceability.

## Standards Checked

- iCalendar RFC 5545: supports `VEVENT`, `VTODO`, `UID`, `DTSTART`, `DTEND`, `DUE`, `SUMMARY`, `DESCRIPTION`, `LOCATION`, `GEO`, `ORGANIZER`, `ATTENDEE`, `STATUS`, `CLASS`, `TRANSP`, `PRIORITY`, `CATEGORIES`, `ATTACH`, `RESOURCES`, `VALARM`, `RRULE`, `EXDATE`, `CREATED`, `LAST-MODIFIED`, and `SEQUENCE`.
- Google Calendar API event resources: include start/end date-time, recurrence, attendees, reminders, conference/online meeting information, transparency, visibility, status, location, attachments, and event type.
- Microsoft Graph Outlook event resources: include start/end, isAllDay, recurrence, attendees, organizer, reminders, online meeting info, sensitivity/privacy, showAs/free-busy status, categories, attachments, locations, and event type.
