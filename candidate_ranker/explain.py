"""
explain.py
==========
Turns a `ScoreRecord` into a human-readable, 1-2 line explanation plus a
confidence score. This is the "why did this rank here?" layer the spec
emphasizes.

A good explanation names the *specific* evidence:
    "Strong match: covers all required skills (Python, PyTorch, RAG), 7 yrs
     experience (min 5), and active GitHub (2.3k stars)."

Confidence reflects how trustworthy the final score is, blending:
  - how many independent signals agree (semantic + skill + experience),
  - profile completeness / verifiability,
  - the gap to neighbours (a wide margin = more decisive ranking).
"""

from __future__ import annotations

import math
from typing import Sequence

from .scoring import ScoreRecord
from .schema import Candidate, JobQuery


def explain_record(rec: ScoreRecord, job: JobQuery) -> dict:
    """Return {explanation, confidence, reasons[]} for one candidate."""
    reasons = _reasons(rec, job)
    explanation = _sentence(reasons)
    confidence = _confidence(rec, job)
    return {
        "explanation": explanation,
        "confidence": round(float(confidence), 3),
        "reasons": reasons,
    }


# --------------------------------------------------------------------------- #
# Reason mining
# --------------------------------------------------------------------------- #
def _reasons(rec: ScoreRecord, job: JobQuery) -> list[str]:
    c = rec.components
    cand = rec.candidate
    out: list[str] = []

    # Semantic
    sem = c["semantic"]
    if sem >= 0.75:
        out.append(f"strong semantic match to the '{job.title or 'role'}' description")
    elif sem >= 0.55:
        out.append("relevant profile text for the role")

    # Skills
    if rec.matched_required:
        names = _fmt_list(rec.matched_required[:5])
        out.append(f"covers required skills ({names})")
    if rec.matched_preferred:
        names = _fmt_list(rec.matched_preferred[:3])
        out.append(f"also brings preferred skills ({names})")
    if rec.missing_required:
        names = _fmt_list(rec.missing_required[:3])
        out.append(f"missing {names}")

    # Experience
    yrs = cand.experience_years
    if job.min_experience and yrs >= job.min_experience:
        out.append(f"{yrs:g} yrs experience (>= {job.min_experience:g} required)")
    elif job.min_experience and yrs > 0:
        out.append(f"{yrs:g} yrs experience (below {job.min_experience:g} required)")
    elif yrs > 0:
        out.append(f"{yrs:g} yrs experience")

    # Activity / GitHub
    gh = cand.github
    if gh.stars or gh.recent_commits or gh.repos:
        bits = []
        if gh.stars:
            bits.append(f"{gh.stars} GitHub stars")
        if gh.recent_commits:
            bits.append(f"{gh.recent_commits} recent commits")
        if gh.repos and not (gh.stars or gh.recent_commits):
            bits.append(f"{gh.repos} public repos")
        out.append(f"active GitHub ({', '.join(bits)})")

    # Completeness / quality caveats
    if c["quality"] < 0.8:
        out.append("some profile-quality concerns")
    if c["completeness"] < 0.5:
        out.append("sparse profile")

    if not out:
        out.append("partial match on profile text and skills")
    return out


def _sentence(reasons: list[str]) -> str:
    """Glue reasons into at most two natural sentences."""
    if not reasons:
        return "Limited signal available for this candidate."
    positive, caveat = _split(reasons)
    s1 = _join(positive)
    if caveat:
        s2 = _join(caveat)
        return f"{s1}. {s2.capitalize()}."
    return f"{s1}."


def _split(reasons: list[str]) -> tuple[list[str], list[str]]:
    caveats = {"sparse profile", "some profile-quality concerns"}
    pos, cav = [], []
    for r in reasons:
        (cav if any(k in r for k in caveats) or r.startswith("missing") else pos).append(r)
    return (pos or reasons[:1]), cav


def _join(parts: list[str]) -> str:
    parts = [p[0].lower() + p[1:] if p else p for p in parts]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _fmt_list(items: Sequence[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + ", " + items[-1]


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #
def _confidence(rec: ScoreRecord, job: JobQuery) -> float:
    c = rec.components
    cand = rec.candidate

    # (1) Signal agreement: do the big three point the same way?
    big = [c["semantic"], c["skill"], c["experience"]]
    agreement = 1.0 - _stddev(big)  # low variance -> high agreement

    # (2) Magnitude: a strong final score is more confident than a middling one.
    magnitude = rec.final

    # (3) Verifiability: completeness + quality + presence of dates.
    has_dates = any(r.start or r.end for r in cand.work_history)
    verifiable = 0.6 * c["completeness"] + 0.3 * c["quality"] + 0.1 * float(has_dates)

    conf = 0.4 * magnitude + 0.35 * agreement + 0.25 * verifiable
    # Required-skill gaps erode confidence sharply.
    if job.required_skills and rec.missing_required:
        miss_frac = len(rec.missing_required) / len(job.required_skills)
        conf *= (1.0 - 0.5 * miss_frac)
    return float(max(0.0, min(1.0, conf)))


def _stddev(xs: list[float]) -> float:
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
