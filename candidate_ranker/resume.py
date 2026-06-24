"""
resume.py
=========
Optional, dependency-free resume text parser.

When a candidate record carries raw `resume_text` (e.g. parsed from a PDF),
this module extracts structured signals — skills, estimated years of
experience, and a likely job title — so the scorer has richer features.

It is intentionally lightweight (regex + skill lexicon). For production you
can plug in a heavier NER/parser; this keeps the dependency surface small.
"""

from __future__ import annotations

import re
from typing import Any

from .schema import SKILL_LEXICON, normalize_skill, normalize_skills

# Phrases that indicate a duration of experience.
_YEARS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?|y)"  # "5 years", "3+ yrs"
    r"(?:\s+of)?\s*(?:experience|exp|professional)?",
    re.IGNORECASE,
)

# Approximate date-range patterns: "2020 - 2023", "Jan 2019 - Present".
_RANGE_RE = re.compile(
    r"(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?[a-z]*\s*)?"
    r"((?:19|20)\d{2})\s*[-–—to]+\s*"
    r"([a-z]*\s*)((?:19|20)\d{2}|present|current|now)",
    re.IGNORECASE,
)


def parse_resume_text(text: str) -> dict[str, Any]:
    """Extract {skills, experience_years, title} from raw resume text."""
    text = text or ""
    lower = text.lower()

    # Skills: match known lexicon tokens (word-boundary safe).
    found = []
    for skill in SKILL_LEXICON:
        pattern = re.escape(skill)
        if re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", lower):
            found.append(skill)

    # Experience: prefer the max "X years" phrase, else sum date ranges.
    years = 0.0
    explicit = [_to_float(m) for m in _YEARS_RE.findall(text)]
    if explicit:
        years = max(explicit)
    else:
        years = _years_from_ranges(text)

    title = _guess_title(text)

    return {
        "skills": normalize_skills(found),
        "experience_years": round(years, 1),
        "title": title,
    }


def enrich_candidate_from_resume(raw: dict) -> dict:
    """Return a new raw dict with resume-derived fields merged in.

    Existing structured fields always win; the parser only fills gaps.
    """
    resume_text = raw.get("resume_text") or ""
    if not resume_text:
        return dict(raw)
    parsed = parse_resume_text(resume_text)
    out = dict(raw)
    out["skills"] = normalize_skills(
        list(dict.fromkeys((raw.get("skills") or []) + parsed["skills"]))
    )
    if not raw.get("experience_years"):
        out["experience_years"] = parsed["experience_years"]
    if not raw.get("title"):
        out["title"] = parsed["title"]
    return out


# --------------------------------------------------------------------------- #
def _to_float(match) -> float:
    try:
        return float(match)
    except (TypeError, ValueError):
        return 0.0


def _years_from_ranges(text: str) -> float:
    total = 0
    for m in _RANGE_RE.finditer(text):
        try:
            start = int(m.group(2))
            end_tok = (m.group(4) or "").lower()
            end = 2026 if end_tok in ("present", "current", "now") else int(end_tok)
            if end >= start:
                total += end - start
        except (ValueError, AttributeError):
            continue
    return round(total, 1)


_TITLE_HINTS = [
    "senior machine learning engineer", "machine learning engineer",
    "ml engineer", "data scientist", "data engineer", "software engineer",
    "backend engineer", "frontend engineer", "full stack engineer",
    "devops engineer", "site reliability engineer", "research scientist",
    "engineering manager", "tech lead", "architect",
]


def _guess_title(text: str) -> str:
    lower = text.lower()
    for hint in _TITLE_HINTS:
        if hint in lower:
            return hint.title()
    return ""
