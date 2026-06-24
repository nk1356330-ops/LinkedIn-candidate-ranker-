"""
filters.py
==========
Pre-ranking pruning and post-ranking refinement:

* `apply_hard_filters`  -> drop candidates failing must-have criteria early
                           (saves embedding/scoring cost on the rest).
* `deduplicate`         -> collapse the same person fetched from multiple APIs.
* `diversify`           -> MMR re-rank to avoid same-company / same-source bias.
"""

from __future__ import annotations

from typing import Iterable

from .config import RankerConfig
from .schema import Candidate, JobQuery
from .scoring import ScoreRecord


# --------------------------------------------------------------------------- #
# Hard filters
# --------------------------------------------------------------------------- #
def apply_hard_filters(candidates: Iterable[Candidate], job: JobQuery,
                       config: RankerConfig) -> list[Candidate]:
    """Enforce must-have criteria BEFORE the expensive embedding step."""
    kept: list[Candidate] = []
    grace = config.experience_grace_years

    for cand in candidates:
        # (1) Minimum experience (with a small grace window).
        if job.min_experience is not None:
            if cand.experience_years + grace < job.min_experience:
                continue
        # (2) Required-skill coverage threshold.
        if job.required_skills and config.min_required_skill_fraction > 0:
            cand_skills = set(cand.skills) | set(cand.github.languages)
            covered = sum(1 for s in job.required_skills if s in cand_skills)
            frac = covered / len(job.required_skills)
            if frac < config.min_required_skill_fraction:
                continue
        # (3) Location / remote preference (best-effort, never over-filter).
        if not _location_ok(cand, job):
            continue
        kept.append(cand)
    return kept


def _location_ok(cand: Candidate, job: JobQuery) -> bool:
    loc = job.location or {}
    wanted = (loc.get("country") or loc.get("region") or loc.get("city") or "").lower()
    if not wanted:
        return True
    if job.remote_ok and cand.is_remote:
        return True
    cand_loc = (cand.location_country or "").lower()
    if not cand_loc:
        return True  # unknown location -> don't exclude on uncertainty
    return wanted in cand_loc or cand_loc in wanted


# --------------------------------------------------------------------------- #
# De-duplication
# --------------------------------------------------------------------------- #
def deduplicate(candidates: list[Candidate]) -> list[Candidate]:
    """Collapse duplicate profiles (same person, multiple sources).

    When two records share a fingerprint we keep the more complete one and
    merge their skills so no signal is lost.
    """
    by_fp: dict[str, Candidate] = {}
    for cand in candidates:
        fp = cand.fingerprint()
        if fp not in by_fp:
            by_fp[fp] = cand
            continue
        kept = by_fp[fp]
        if _completeness_score(cand) > _completeness_score(kept):
            # Merge skills from the weaker record before replacing.
            merged_skills = list(dict.fromkeys(kept.skills + cand.skills))
            cand.skills = merged_skills
            by_fp[fp] = cand
        else:
            kept.skills = list(dict.fromkeys(kept.skills + cand.skills))
    return list(by_fp.values())


def _completeness_score(cand: Candidate) -> int:
    """Cheap completeness proxy used only for de-dup tie-breaking."""
    return (int(bool(cand.name)) + int(bool(cand.title)) + int(bool(cand.summary))
            + int(bool(cand.skills)) + int(cand.experience_years > 0)
            + int(bool(cand.location)) + len(cand.work_history)
            + int(bool(cand.signals)))


# --------------------------------------------------------------------------- #
# Diversity (MMR) re-ranking
# --------------------------------------------------------------------------- #
def diversify(records: list[ScoreRecord], config: RankerConfig,
              top_n: int) -> list[ScoreRecord]:
    """Maximal Marginal Relevance re-ranking over the top pool.

    Balances relevance with diversity so the final list isn't dominated by
    candidates from one company or one data source. `diversity_lambda`
    interpolates: 0 = pure relevance, 1 = max diversity.
    """
    if config.diversity_lambda <= 0 or len(records) <= max(2, top_n):
        return records

    pool = records[:config.diversity_pool]
    lam = config.diversity_lambda

    selected: list[ScoreRecord] = [pool[0]]
    remaining = list(pool[1:])

    while remaining and len(selected) < top_n:
        best, best_idx, best_score = None, -1, -1.0
        for i, rec in enumerate(remaining):
            rel = rec.final
            # diversity = 1 if this candidate's company/source is NOT yet chosen
            company = rec.candidate.current_company.lower()
            source = rec.candidate.source.lower()
            already = any(
                r.candidate.current_company.lower() == company
                or r.candidate.source.lower() == source
                for r in selected
            )
            div = 0.0 if already else 1.0
            mmr = (1 - lam) * rel + lam * div
            if mmr > best_score:
                best, best_idx, best_score = rec, i, mmr
        if best is None:
            break
        selected.append(best)
        remaining.pop(best_idx)

    # Append any leftover pool items (relevance order) if top_n exceeds pool.
    if len(selected) < top_n:
        selected.extend(remaining[: top_n - len(selected)])
    return selected
