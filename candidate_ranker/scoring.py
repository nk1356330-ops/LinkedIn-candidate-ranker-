"""
scoring.py
==========
Implements the weighted ranking formula:

    Final Score = w1*Semantic + w2*Skill + w3*Experience + w4*Behavior

Every component score is normalized to [0, 1], and weights are auto-normalized
to sum to 1 (see `config.normalize_weights`), so the final score is also in
[0, 1] and directly comparable across candidates and queries.

Sub-scores are returned individually so `explain.py` can attribute *why* a
candidate ranked where it did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import RankerConfig, normalize_weights
from .embedding import SemanticEmbedder
from .schema import Candidate, JobQuery, days_since


# --------------------------------------------------------------------------- #
# Result record produced per candidate.
# --------------------------------------------------------------------------- #
@dataclass
class ScoreRecord:
    candidate: Candidate
    final: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    matched_required: list[str] = field(default_factory=list)
    matched_preferred: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        c = self.candidate
        return {
            "id": c.id,
            "name": c.name,
            "title": c.title,
            "score": round(float(self.final), 4),
            "skills": c.skills,
            "matched_required_skills": self.matched_required,
            "matched_preferred_skills": self.matched_preferred,
            "experience_years": c.experience_years,
            "current_company": c.current_company,
            "location": c.location,
            "source": c.source,
            "component_scores": {k: round(float(v), 4)
                                 for k, v in self.components.items()},
        }


class Scorer:
    """Computes per-candidate component scores and the weighted final score."""

    def __init__(self, config: RankerConfig | None = None,
                 embedder: SemanticEmbedder | None = None):
        self.cfg = config or RankerConfig()
        self.embedder = embedder or SemanticEmbedder(self.cfg)
        self.weights = normalize_weights(self.cfg.weights)

    # ------------------------------------------------------------------ #
    # Main entry: score a batch of (already filtered) candidates.
    # ------------------------------------------------------------------ #
    def score(self, candidates: list[Candidate], job: JobQuery) -> list[ScoreRecord]:
        if not candidates:
            return []

        # 1) Semantic scores in one vectorized pass (cached + batched).
        job_vec = self.embedder.embed_one(job.job_text())
        cand_texts = [c.profile_text() for c in candidates]
        cand_matrix = self.embedder.embed_many(cand_texts)
        sem = self.embedder.cosine(job_vec, cand_matrix)
        sem = _minmax_clip(sem)  # keep in [0,1]; cosine can be slightly negative

        # 2) Structured scores per candidate.
        records: list[ScoreRecord] = []
        for i, cand in enumerate(candidates):
            skill, mr, mp, miss = self._skill_score(cand, job)
            exp = self._experience_score(cand, job)
            beh = self._behavior_score(cand, job)

            comp = {
                "semantic": float(sem[i]),
                "skill": skill,
                "experience": exp,
                "behavior": beh,
                # diagnostic sub-scores (not weighted directly):
                "completeness": self._completeness(cand),
                "activity": self._activity(cand),
                "consistency": self._consistency(cand, job),
                "quality": self._quality(cand),
            }
            records.append(ScoreRecord(
                candidate=cand, components=comp,
                matched_required=mr, matched_preferred=mp, missing_required=miss,
            ))

        # 3) Weighted combination.
        w = self.weights
        for rec in records:
            c = rec.components
            rec.final = (
                w["semantic"] * c["semantic"]
                + w["skill"] * c["skill"]
                + w["experience"] * c["experience"]
                + w["behavior"] * c["behavior"]
            )
        return records

    # ------------------------------------------------------------------ #
    # Component: SKILL COVERAGE
    # ------------------------------------------------------------------ #
    def _skill_score(self, cand: Candidate, job: JobQuery):
        """Weighted coverage of required (heavy) + preferred (light) skills."""
        cand_skills = set(cand.skills)
        # Expand candidate skills with their github languages for completeness.
        cand_skills |= set(cand.github.languages)

        req = job.required_skills
        pref = job.preferred_skills

        matched_req = [s for s in req if s in cand_skills]
        matched_pref = [s for s in pref if s in cand_skills]
        missing_req = [s for s in req if s not in cand_skills]

        wr = self.cfg.required_skill_weight
        wp = self.cfg.preferred_skill_weight

        req_part = (len(matched_req) / len(req)) if req else 1.0
        pref_part = (len(matched_pref) / len(pref)) if pref else 0.0

        # When there are no preferred skills, give required skills full weight.
        if not pref:
            score = req_part
        else:
            denom = wr + wp
            score = (wr * req_part + wp * pref_part) / denom if denom else req_part

        return float(np.clip(score, 0.0, 1.0)), matched_req, matched_pref, missing_req

    # ------------------------------------------------------------------ #
    # Component: EXPERIENCE FIT
    # ------------------------------------------------------------------ #
    def _experience_score(self, cand: Candidate, job: JobQuery) -> float:
        """Map years of experience to [0,1] relative to the job's minimum.

        - Below the minimum -> steep penalty (but still > 0 so borderline
          candidates with exceptional other signals aren't zeroed).
        - At/above the minimum -> smooth diminishing returns so 20y isn't
          scored infinitely better than 7y.
        """
        yrs = max(0.0, float(cand.experience_years))
        target = float(job.min_experience) if job.min_experience else 3.0

        if yrs <= 0:
            return 0.0
        if yrs < target:
            # partial credit, scaled by how close to the minimum
            ratio = yrs / max(target, 1e-6)
            return float(np.clip(0.5 * ratio, 0.0, 0.5))

        # Diminishing returns above target: e.g. target=5 -> 5y=0.75, 10y~0.94
        excess = yrs - target
        bonus = 0.25 + 0.25 * (1 - math.exp(-excess / max(target, 1e-6)))
        return float(np.clip(0.5 + bonus, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # Component: BEHAVIOUR & QUALITY (aggregated)
    # ------------------------------------------------------------------ #
    def _behavior_score(self, cand: Candidate, job: JobQuery) -> float:
        """Blend completeness, activity and consistency, minus quality penalty."""
        completeness = self._completeness(cand)
        activity = self._activity(cand)
        consistency = self._consistency(cand, job)
        quality = self._quality(cand)  # 1 = clean, lower = suspicious

        blended = 0.4 * completeness + 0.35 * activity + 0.25 * consistency
        # quality acts as a multiplier so bad profiles can't score high.
        return float(np.clip(blended * quality, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # Behaviour sub-scores
    # ------------------------------------------------------------------ #
    def _completeness(self, cand: Candidate) -> float:
        """Fraction of important profile fields that are populated."""
        checks = [
            bool(cand.name),
            bool(cand.title),
            bool(cand.summary),
            bool(cand.skills),
            cand.experience_years > 0,
            bool(cand.location),
            len(cand.work_history) >= 1,
            bool(cand.signals),
        ]
        return float(sum(checks) / len(checks))

    def _activity(self, cand: Candidate) -> float:
        """Recent-update + GitHub activity, decaying over time."""
        recency = self.cfg.activity_recency_days

        # (a) Profile recency.
        prof = 0.0
        d = days_since(cand.updated_at)
        if d is not None:
            prof = math.exp(-d / recency)
        else:
            prof = 0.3  # unknown date -> small neutral floor

        # (b) GitHub signal (normalized, log-scaled).
        gh = cand.github
        gh_raw = (math.log1p(gh.recent_commits) / math.log1p(50)
                  + math.log1p(gh.stars) / math.log1p(500)
                  + math.log1p(gh.repos) / math.log1p(30)) / 3.0
        gh_score = min(1.0, gh_raw) if (gh.repos or gh.stars or gh.recent_commits) else 0.0

        # If no github data at all, rely mostly on profile recency.
        has_gh = bool(gh.repos or gh.stars or gh.recent_commits)
        return float(np.clip(0.6 * prof + 0.4 * gh_score, 0.0, 1.0)) if has_gh \
            else float(np.clip(prof, 0.0, 1.0))

    def _consistency(self, cand: Candidate, job: JobQuery) -> float:
        """Do the candidate's skills line up with their stated title/history?

        Inconsistencies (e.g. a "Senior Backend Engineer" listing only design
        tools, or far more skills than plausible for their tenure) lower trust.
        """
        score = 1.0
        # (a) skill count sanity vs years of experience
        n_skills = len(cand.skills)
        yrs = cand.experience_years
        if n_skills > self.cfg.suspicious_skill_count:
            score -= 0.4
        if yrs > 0 and n_skills > max(8, 4 * yrs):
            score -= 0.2
        # (b) title mentions a skill family that appears nowhere in skills
        title = cand.title.lower()
        families = {
            "ml": ["machine learning", "deep learning", "pytorch", "tensorflow"],
            "data": ["sql", "spark", "etl", "pandas"],
            "frontend": ["javascript", "react", "css"],
            "devops": ["docker", "kubernetes", "ci/cd", "aws"],
        }
        cand_skills = set(cand.skills)
        for kw, members in families.items():
            if kw in title and not (cand_skills & set(members)):
                score -= 0.15
        return float(np.clip(score, 0.0, 1.0))

    def _quality(self, cand: Candidate) -> float:
        """A [0,1] trust multiplier; suspicious profiles get penalized."""
        q = 1.0
        if len(cand.skills) > self.cfg.suspicious_skill_count:
            q -= 0.3
        # Resume claims many skills but zero work history -> suspicious.
        if len(cand.skills) > 15 and not cand.work_history:
            q -= 0.25
        # No dates anywhere and no updated_at -> low verifiability.
        has_dates = any(r.start or r.end for r in cand.work_history)
        if not has_dates and not cand.updated_at:
            q -= 0.15
        return float(np.clip(q, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _minmax_clip(values: np.ndarray, floor: float = 0.0, ceil: float = 1.0) -> np.ndarray:
    """Clip into [floor, ceil] and rescale cosine-ish values into [0,1]."""
    v = np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)
    # map [-1,1] -> [0,1]
    v = (v + 1.0) / 2.0
    return np.clip(v, floor, ceil).astype(np.float32)
