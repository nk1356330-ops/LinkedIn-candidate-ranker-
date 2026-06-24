"""
pipeline.py
===========
The single public entry point that slots into your architecture at Step 4
("Run rank.py"):

      fetch (existing) -> normalize (existing) -> rank_candidates() -> HR

`rank_candidates(candidates_json, job_json)` accepts the normalized Redrob
candidate JSON + an HR job query and returns ranked, explainable JSON. It does
NOT touch any data-fetching API.

Pipeline (hybrid = hard rule-based filter + semantic + structured scoring):

    parse -> hard-filter -> dedup -> embed(batched,cached)
          -> weighted-score -> sort(stable) -> diversity re-rank(MMR)
          -> explain + confidence -> slice top-N -> JSON
"""

from __future__ import annotations

import time
from typing import Any

from .config import RankerConfig, normalize_weights
from .embedding import SemanticEmbedder
from .explain import explain_record
from .filters import apply_hard_filters, deduplicate, diversify
from .resume import enrich_candidate_from_resume
from .schema import Candidate, JobQuery, parse_candidate, parse_job
from .scoring import Scorer, ScoreRecord  # noqa: F401  (re-export)


def rank_candidates(
    candidates: list[dict[str, Any]],
    job: dict[str, Any],
    *,
    top_n: int | None = None,
    weights: dict[str, float] | None = None,
    config: RankerConfig | None = None,
    return_components: bool = True,
) -> dict[str, Any]:
    """Rank normalized candidate JSON against an HR job query.

    Parameters
    ----------
    candidates : list[dict]
        Normalized Redrob candidate records (output of your fetch+normalize step).
    job : dict
        Job query: {title, description, required_skills, preferred_skills,
        min_experience, location, remote_ok, top_n}.
    top_n : int, optional
        Override how many ranked results to return (dynamic Top 5/10/50).
    weights : dict, optional
        Override {semantic, skill, experience, behavior} weights.
    config : RankerConfig, optional
        Full tuning override; otherwise defaults are used.
    return_components : bool
        Include the sub-score breakdown in each result (useful for debugging).

    Returns
    -------
    dict  ->  {query, meta, results} where each result matches the spec's
              output schema (name, score, skills, experience, explanation).
    """
    cfg = config or RankerConfig()
    if weights:
        cfg.weights = {**cfg.weights, **weights}
    # Re-normalize so the printed weights always sum to 1.
    effective_weights = normalize_weights(cfg.weights)

    t0 = time.perf_counter()
    n_input = len(candidates)

    # 1) Parse + resume enrichment (lenient).
    job_q = parse_job(job)
    parsed = []
    for raw in candidates:
        raw = enrich_candidate_from_resume(raw) if raw.get("resume_text") else raw
        parsed.append(parse_candidate(raw))

    # 2) Hard filters (prune before the expensive embedding step).
    kept = apply_hard_filters(parsed, job_q, cfg)
    n_after_filter = len(kept)

    # 3) De-duplicate (same person across multiple APIs).
    kept = deduplicate(kept)
    n_after_dedup = len(kept)

    # 4) Score (embeds in batches with caching, then weighted combination).
    embedder = SemanticEmbedder(cfg)
    scorer = Scorer(cfg, embedder)
    records: list[ScoreRecord] = scorer.score(kept, job_q)

    # 5) Stable sort by final score (tie-break by id for ranking stability).
    records.sort(key=lambda r: (-r.final, r.candidate.id))

    # 6) Diversity re-ranking (MMR) on the top pool to avoid same-company bias.
    k = top_n or job_q.top_n or cfg.default_top_n
    top = diversify(records, cfg, k)

    # 7) Explain + confidence, then slice to top-N.
    results = []
    for rec in top[:k]:
        cand = rec.candidate
        expl = explain_record(rec, job_q)
        item = {
            "name": cand.name,
            "score": round(float(rec.final), 4),
            "title": cand.title,
            "skills": cand.skills,
            "experience": f"{cand.experience_years:g} yrs",
            "experience_years": cand.experience_years,
            "matched_required_skills": rec.matched_required,
            "matched_preferred_skills": rec.matched_preferred,
            "missing_required_skills": rec.missing_required,
            "current_company": cand.current_company,
            "location": cand.location,
            "source": cand.source,
            "explanation": expl["explanation"],
            "confidence": expl["confidence"],
        }
        if return_components:
            item["component_scores"] = {
                key: round(val, 4) for key, val in rec.components.items()
            }
        results.append(item)

    # Persist the embedding cache for next query.
    embedder.cache.save()

    elapsed = round(time.perf_counter() - t0, 4)
    meta = {
        "total_input": n_input,
        "after_filter": n_after_filter,
        "after_dedup": n_after_dedup,
        "returned": len(results),
        "top_n": k,
        "embedding_backend": embedder.backend_name,
        "embedding_dim": embedder.dim,
        "cache_size": len(embedder.cache),
        "effective_weights": effective_weights,
        "elapsed_seconds": elapsed,
    }
    return {
        "query": {
            "title": job_q.title,
            "min_experience": job_q.min_experience,
            "required_skills": job_q.required_skills,
            "preferred_skills": job_q.preferred_skills,
            "location": job_q.location,
            "remote_ok": job_q.remote_ok,
        },
        "meta": meta,
        "results": results,
    }
