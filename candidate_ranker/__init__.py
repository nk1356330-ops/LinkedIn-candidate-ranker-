"""
candidate_ranker
================
A modular, explainable candidate ranking engine designed to drop into an
existing real-time sourcing pipeline at the "rank.py" step.

Public API
----------
    from candidate_ranker import rank_candidates

    result = rank_candidates(candidates_json, job_json, top_n=5)

It does NOT fetch data. Feed it the normalized candidate JSON your fetch +
normalize steps already produce.

Modules
-------
- config.py    : tunable weights + thresholds (single place to adjust behaviour)
- schema.py    : Redrob candidate/job schema + normalization helpers
- cache.py     : disk-backed vector cache (for 10K-100K scale)
- embedding.py : Sentence Transformers w/ automatic TF-IDF fallback, batched+cached
- scoring.py   : weighted Final = w1*Semantic + w2*Skill + w3*Exp + w4*Behavior
- explain.py   : 1-2 line explanations + confidence scores
- filters.py   : hard filters, de-duplication, MMR diversity re-rank
- resume.py    : optional dependency-free resume-text parser
- pipeline.py  : rank_candidates() orchestrator (Step 4 entry point)
"""

from .config import RankerConfig, DEFAULT_WEIGHTS, normalize_weights
from .pipeline import rank_candidates
from .schema import Candidate, JobQuery, parse_candidate, parse_job

__all__ = [
    "rank_candidates",
    "RankerConfig",
    "DEFAULT_WEIGHTS",
    "normalize_weights",
    "Candidate",
    "JobQuery",
    "parse_candidate",
    "parse_job",
]

__version__ = "1.0.0"
