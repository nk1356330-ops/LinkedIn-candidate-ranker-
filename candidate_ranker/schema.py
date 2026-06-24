"""
schema.py
=========
Lenient dataclasses + normalization helpers for the *Redrob candidate schema*.

This module assumes candidate/job data has ALREADY been fetched from the
upstream APIs (Bright Data, PDL, GitHub, job boards) and normalized to JSON.
It only makes the JSON ergonomic and computes derived features used downstream
(embedding text, fingerprints, skill normalization).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Skill canonicalization. Maps common aliases / abbreviations to a single key
# so "ML" == "machine learning", "k8s" == "kubernetes", etc. Extend freely.
# --------------------------------------------------------------------------- #
SKILL_ALIASES: dict[str, str] = {
    "ml": "machine learning",
    "machinelearning": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "llm": "large language models",
    "llms": "large language models",
    "gpt": "large language models",
    "k8s": "kubernetes",
    "kub": "kubernetes",
    "tf": "tensorflow",
    "pytorch": "pytorch",
    "torch": "pytorch",
    "js": "javascript",
    "ts": "typescript",
    "node": "node.js",
    "nodejs": "node.js",
    "postgres": "postgresql",
    "pg": "postgresql",
    "go": "golang",
    "golang": "golang",
    "c++": "c++",
    "cpp": "c++",
    "c#": "c#",
    "csharp": "c#",
    "rn": "react native",
    "aws": "aws",
    "gcp": "gcp",
    "azure": "azure",
    "tfidf": "tf-idf",
    "rag": "retrieval augmented generation",
    "etl": "etl",
    "ci/cd": "ci/cd",
    "cicd": "ci/cd",
    "sql": "sql",
    "nosql": "nosql",
    "rest": "rest api",
    "graphql": "graphql",
    "docker": "docker",
}

# A lightweight lexicon used by the optional resume parser to spot skills.
SKILL_LEXICON: tuple[str, ...] = tuple(sorted(set(SKILL_ALIASES.values())))


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _as_int(x: Any, default: int = 0) -> int:
    return int(round(_as_float(x, default)))


def normalize_skill(raw: str) -> str:
    """Lowercase + alias-collapse a single skill token."""
    s = raw.strip().lower()
    s = re.sub(r"[.\-/]+", " ", s)      # node.js -> node js
    s = re.sub(r"\s+", " ", s).strip()
    key = s.replace(" ", "")
    return SKILL_ALIASES.get(key, s or raw.strip().lower())


def normalize_skills(skills: Any) -> list[str]:
    """Accept a list, comma-string, or dict-with-keys; return clean unique list."""
    if not skills:
        return []
    out: list[str] = []
    if isinstance(skills, str):
        tokens = re.split(r"[,/;|•]+|\band\b", skills)
    elif isinstance(skills, dict):
        tokens = list(skills.keys())
    elif isinstance(skills, (list, tuple, set)):
        tokens = list(skills)
    else:
        tokens = [str(skills)]
    for t in tokens:
        s = normalize_skill(_as_str(t))
        if s and s not in out:
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Candidate
# --------------------------------------------------------------------------- #
@dataclass
class WorkRole:
    company: str = ""
    title: str = ""
    start: str = ""
    end: str = ""          # "present" or ISO date or ""


@dataclass
class GithubSignals:
    repos: int = 0
    stars: int = 0
    recent_commits: int = 0
    followers: int = 0
    languages: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    # Identity
    id: str = ""
    name: str = ""
    title: str = ""
    summary: str = ""
    # Structured
    skills: list[str] = field(default_factory=list)
    experience_years: float = 0.0
    location: dict[str, Any] = field(default_factory=dict)
    work_history: list[WorkRole] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    # Provenance / quality
    source: str = ""
    updated_at: str = ""
    resume_text: str = ""

    # ---- derived helpers ------------------------------------------------ #
    @property
    def github(self) -> GithubSignals:
        gh = self.signals.get("github") or {}
        return GithubSignals(
            repos=_as_int(gh.get("repos")),
            stars=_as_int(gh.get("stars")),
            recent_commits=_as_int(gh.get("recent_commits")),
            followers=_as_int(gh.get("followers")),
            languages=normalize_skills(gh.get("languages")),
        )

    @property
    def current_company(self) -> str:
        for role in self.work_history:
            if not role.end or role.end.lower() == "present":
                return role.company
        return self.work_history[0].company if self.work_history else ""

    @property
    def location_country(self) -> str:
        loc = self.location or {}
        return _as_str(loc.get("country") or loc.get("region") or loc.get("city"))

    @property
    def is_remote(self) -> bool:
        return bool((self.location or {}).get("remote"))

    def fingerprint(self) -> str:
        """Stable identity used for de-duplication.

        Lowercased name + current company is robust across the small field
        variations that different APIs (PDL vs Bright Data) introduce.
        """
        name_key = re.sub(r"[^a-z0-9]", "", self.name.lower())
        company_key = re.sub(r"[^a-z0-9]", "", self.current_company.lower())
        h = hashlib.sha1(f"{name_key}|{company_key}".encode()).hexdigest()
        # Fallback to name-only if company is missing.
        return h if company_key else hashlib.sha1(name_key.encode()).hexdigest()

    def profile_text(self) -> str:
        """Single searchable document used for semantic embedding."""
        parts: list[str] = []
        if self.title:
            parts.append(self.title)
        if self.summary:
            parts.append(self.summary)
        if self.skills:
            parts.append("Skills: " + ", ".join(self.skills[:40]))
        role_titles = [r.title for r in self.work_history if r.title]
        if role_titles:
            parts.append("Experience as: " + ", ".join(role_titles[:10]))
        companies = [r.company for r in self.work_history if r.company]
        if companies:
            parts.append("Worked at: " + ", ".join(companies[:10]))
        return ". ".join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# JobQuery
# --------------------------------------------------------------------------- #
@dataclass
class JobQuery:
    title: str = ""
    description: str = ""
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    min_experience: Optional[float] = None
    location: dict[str, Any] = field(default_factory=dict)
    remote_ok: bool = False
    top_n: int = 10

    def job_text(self) -> str:
        """Document that candidate profiles are matched against semantically."""
        parts: list[str] = []
        if self.title:
            parts.append(self.title)
        if self.description:
            parts.append(self.description)
        if self.required_skills:
            parts.append("Required skills: " + ", ".join(self.required_skills))
        if self.preferred_skills:
            parts.append("Nice to have: " + ", ".join(self.preferred_skills))
        if self.location:
            parts.append("Location: " + ", ".join(
                f"{k}:{v}" for k, v in self.location.items() if v))
        return ". ".join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# Parsing from raw JSON (lenient: missing fields never raise)
# --------------------------------------------------------------------------- #
def parse_candidate(raw: dict[str, Any]) -> Candidate:
    wh_raw = raw.get("work_history") or raw.get("experience") or []
    work = []
    if isinstance(wh_raw, list):
        for r in wh_raw:
            if not isinstance(r, dict):
                continue
            work.append(WorkRole(
                company=_as_str(r.get("company")),
                title=_as_str(r.get("title")),
                start=_as_str(r.get("start")),
                end=_as_str(r.get("end")),
            ))

    exp = raw.get("experience_years")
    if exp is None:
        exp = raw.get("yoe") or raw.get("years_of_experience")
    # Some sources encode experience as a list of roles with dates.
    if (exp in (None, "", 0)) and work:
        exp = _estimate_years_from_history(work)

    loc = raw.get("location") or {}
    if isinstance(loc, str):
        loc = {"city": loc}

    cid = _as_str(raw.get("id")) or hashlib.sha1(
        (raw.get("name", "") + raw.get("source", "")).encode()
    ).hexdigest()[:16]

    return Candidate(
        id=cid,
        name=_as_str(raw.get("name")),
        title=_as_str(raw.get("title") or raw.get("headline")),
        summary=_as_str(raw.get("summary") or raw.get("bio") or raw.get("about")),
        skills=normalize_skills(raw.get("skills")),
        experience_years=_as_float(exp),
        location=loc,
        work_history=work,
        signals=raw.get("signals") or {},
        source=_as_str(raw.get("source")),
        updated_at=_as_str(raw.get("updated_at")),
        resume_text=_as_str(raw.get("resume_text")),
    )


def parse_job(raw: dict[str, Any]) -> JobQuery:
    loc = raw.get("location") or {}
    if isinstance(loc, str):
        loc = {"country": loc}
    min_exp = raw.get("min_experience")
    if min_exp is None:
        min_exp = raw.get("minimum_experience")
    return JobQuery(
        title=_as_str(raw.get("title")),
        description=_as_str(raw.get("description") or raw.get("jd")),
        required_skills=normalize_skills(raw.get("required_skills")),
        preferred_skills=normalize_skills(raw.get("preferred_skills")),
        min_experience=_as_float(min_exp) if min_exp is not None else None,
        location=loc,
        remote_ok=bool(raw.get("remote_ok", False)),
        top_n=_as_int(raw.get("top_n"), 10),
    )


def _estimate_years_from_history(work: list[WorkRole]) -> float:
    """Best-effort YOE from role start/end dates."""
    total_days = 0
    today = date.today()
    for r in work:
        s = _parse_date(r.start)
        e = _parse_date(r.end) or today
        if s and e and e >= s:
            total_days += (e - s).days
    return round(total_days / 365.25, 1)


def _parse_date(s: str) -> Optional[date]:
    s = (_as_str(s) or "").lower()
    if not s or s in ("present", "current", "now"):
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y/%m/%d", "%b %Y", "%B %Y", "%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    m = re.search(r"(\d{4})", s)
    if m:
        try:
            return date(int(m.group(1)), 1, 1)
        except ValueError:
            return None
    return None


def days_since(s: str) -> Optional[int]:
    """Days between an ISO-ish date string and today. None if unparseable."""
    d = _parse_date(s)
    if not d:
        return None
    return max(0, (date.today() - d).days)
