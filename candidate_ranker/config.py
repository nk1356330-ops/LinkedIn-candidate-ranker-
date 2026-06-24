"""
config.py
=========
Central, human-tunable configuration for the candidate ranker.

All scoring weights, thresholds, and model choices live here so HR/ML teams
can tune behaviour WITHOUT touching the algorithm code. Weights are
auto-normalised at runtime, so you do not have to keep them summing to 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Default scoring weights for the formula:
#
#   Final = w1*Semantic + w2*Skill + w3*Experience + w4*Behavior
#
# Values are relative (auto-normalised to sum to 1.0). Tweak freely.
# --------------------------------------------------------------------------- #
DEFAULT_WEIGHTS: dict[str, float] = {
    "semantic":    0.35,   # BERT / TF-IDF similarity between JD and profile
    "skill":       0.30,   # required + preferred skill coverage
    "experience":  0.20,   # years-of-experience fit vs. the job's minimum
    "behavior":    0.15,   # completeness + activity + consistency - penalties
}


@dataclass
class RankerConfig:
    """Tunable knobs for the full ranking pipeline."""

    # --- Embedding backend ------------------------------------------------ #
    # "auto"      -> use Sentence Transformers if available, else TF-IDF.
    # "sbert"     -> force Sentence Transformers (raises if missing).
    # "tfidf"     -> force the lightweight HashingVectorizer fallback.
    embedding_backend: str = "auto"

    # HuggingFace model id for the semantic backend. all-MiniLM-L6-v2 is
    # small (~80 MB), fast, and strong for short profile/JD texts.
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Embedding batch size. Larger = faster on GPU, more RAM on CPU.
    embedding_batch_size: int = 64

    # CPU/CUDA. Leave "cpu" unless you provision a GPU.
    device: str = "cpu"

    # --- Scoring weights -------------------------------------------------- #
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    # --- Hard-filter thresholds ------------------------------------------ #
    # A candidate must satisfy AT LEAST this fraction of `required_skills`.
    # 1.0 = strict (every required skill present), 0.0 = disabled.
    min_required_skill_fraction: float = 1.0

    # If the job sets `min_experience`, candidates below it are dropped.
    # A small grace window (years) avoids being overly punitive.
    experience_grace_years: float = 0.5

    # --- Skill matching -------------------------------------------------- #
    # When matching skills, also accept a semantic/alias match. Tuning
    # aliases lives in schema.SKILL_ALIASES.
    skill_fuzzy_match: bool = True

    # Weight given to *required* vs *preferred* skills inside the skill score.
    required_skill_weight: float = 0.70
    preferred_skill_weight: float = 0.30

    # --- Behaviour / quality --------------------------------------------- #
    # Penalise profiles that look suspiciously over-stuffed with skills.
    suspicious_skill_count: int = 40

    # Decay window (days) for "recent activity". Updates/commits newer than
    # this count fully; older ones decay toward zero.
    activity_recency_days: int = 365

    # --- Diversity (avoid same-company bias) ----------------------------- #
    # 0.0 = pure relevance, 1.0 = max diversity. ~0.3 is a good default.
    diversity_lambda: float = 0.3

    # Apply MMR diversity re-ranking only on this many top candidates.
    diversity_pool: int = 50

    # --- Output ---------------------------------------------------------- #
    default_top_n: int = 10
    confidence_margin_threshold: float = 0.02  # min gap for "high confidence"

    # --- Caching --------------------------------------------------------- #
    # Directory for the on-disk embedding cache. None -> in-memory only.
    cache_dir: Optional[str] = ".embedding_cache"


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Return a copy of `weights` scaled so the values sum to 1.0.

    Unknown keys are kept (they simply won't be used by the scorer) but only
    the four canonical weights influence the final score.
    """
    canonical = ("semantic", "skill", "experience", "behavior")
    total = sum(max(0.0, float(weights.get(k, 0.0))) for k in canonical)
    if total <= 0:
        # Degenerate input -> fall back to the documented defaults.
        return dict(DEFAULT_WEIGHTS)
    return {k: max(0.0, float(weights.get(k, 0.0))) / total for k in canonical}
