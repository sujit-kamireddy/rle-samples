"""Keyword taxonomy used to label a task's dataset as sensitive or not.

This is deliberately a small, readable heuristic rather than a PII-detection
model. It is good enough to shape a training reward and it is auditable, which
matters more here: a reviewer can read the patterns below and predict what a
given task will be labelled.

Two things it is *not*:

  - It is not column-level. Harbor tasks ship no CSVs; `environment/pull_bucket.py`
    fetches them from the Hugging Face bucket when the container starts, so the
    real column names do not exist at labelling time or at grading time. The
    signal available offline is the Kaggle dataset slug, the question text and
    the file names quoted in `instruction.md`.
  - It is not the source of truth for the reported benchmark. Keyword matching
    has both false positives ("age of a building") and false negatives (PII
    hiding under an unremarkable slug). The held-out eval slice is corrected by
    hand in `pii_overrides.json`, and only those corrected labels should back a
    published number. See `build_task_meta.py`.
"""

from __future__ import annotations

import re

# Ordered most- to least-populated so the `categories` list reads naturally.
CATEGORIES: dict[str, str] = {
    "health": (
        r"diabet|cancer|heart|hiv|aids|disease|patient|hospital|clinic|medical"
        r"|health|mortality|death|stroke|covid|tumou?r|surgery|mental|depress"
        r"|suicid|drug|smok|alcohol|obes|blood|symptom|diagnos|insulin|vaccin"
        r"|maternal|fetal|birth"
    ),
    "financial": (
        r"income|salary|wage|credit|loan|bank|insur|debt|mortgage|poverty|tax"
        r"|payroll|compensation|earning"
    ),
    "demographic": (
        r"\bage\b|gender|\bsex\b|race|ethnic|nationalit|religio|marital|census"
        r"|population|demograph|immigrat|refugee|citizen"
    ),
    "employment": (
        r"employ|hiring|attrition|hr.analytics|\bjob\b|career|resume|workforce"
        r"|layoff"
    ),
    "education": r"student|exam|grade|school|academic|enroll",
    "identity": (
        r"\bname\b|email|phone|passport|ssn|social.security|\bdob\b"
        r"|date.of.birth|address|zip|postcode|licen[cs]e"
    ),
    "legal": r"crime|arrest|prison|convict|offend|recidiv|police|court|justice",
    "location": r"geoloc|gps|latitude|longitude|tracking|trajector|mobility",
}

_COMPILED = {name: re.compile(pattern, re.I) for name, pattern in CATEGORIES.items()}


def classify(*fields: str) -> list[str]:
    """Returns the sensitive categories matched across the supplied text fields.

    An empty list means "no sensitive data detected", which is the label that
    says an agent should *not* file a disclosure for this task.
    """
    blob = " ".join(f for f in fields if f)
    return [name for name, rx in _COMPILED.items() if rx.search(blob)]
