"""
embedding.py
============
Semantic embedding layer with an automatic, zero-friction backend fallback.

Backends
--------
* **sbert** (preferred): Sentence Transformers (`all-MiniLM-L6-v2`) gives dense
  BERT-style vectors with strong semantic generalization (e.g. "ML" ~
  "machine learning", "RAG" ~ "retrieval augmented generation").
* **tfidf** (fallback): a stateless HashingVectorizer over word n-grams. It
  needs no fitting, so every vector is a pure function of its text -> perfectly
  cacheable and streaming-friendly. Lower quality than SBERT but adds ZERO
  heavy dependencies (numpy/sklearn only) and runs anywhere.

Design goals
------------
* **Cached + batched**: candidate vectors are computed once and reused across
  queries via `cache.VectorCache`. Embedding is done in `batch_size` chunks.
* **Deterministic**: same text -> same vector, enabling stable rankings.
* **Normalized output**: every vector is L2-normalized so cosine similarity is
  a plain dot product (fast even at 100K candidates).
"""

from __future__ import annotations

import numpy as np

try:  # sklearn is a required light dependency
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.preprocessing import normalize as _sk_normalize
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required by candidate_ranker.embedding. "
        "Install it with `pip install scikit-learn`."
    ) from _exc

from .cache import VectorCache
from .config import RankerConfig


class SemanticEmbedder:
    """Embeds text into L2-normalized vectors, with caching + batching."""

    def __init__(self, config: RankerConfig | None = None):
        self.cfg = config or RankerConfig()
        self._backend = self._resolve_backend(self.cfg.embedding_backend)
        self._sbert = None
        self._hasher = None
        self.cache = VectorCache(self.cfg.cache_dir, backend_tag=self.backend_name)

    # ------------------------------------------------------------------ #
    # Backend selection
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sbert_available() -> bool:
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def _resolve_backend(self, requested: str) -> str:
        if requested == "sbert":
            if not self._sbert_available():
                raise ImportError(
                    "backend='sbert' requested but `sentence-transformers` is not "
                    "installed. Install it or use backend='auto'/'tfidf'."
                )
            return "sbert"
        if requested == "tfidf":
            return "tfidf"
        # auto
        return "sbert" if self._sbert_available() else "tfidf"

    @property
    def backend_name(self) -> str:
        return self._backend

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        if self._backend == "sbert":
            # Support both the new (>=5.0) and legacy method names.
            fn = (getattr(self._sbert, "get_embedding_dimension", None)
                  or self._sbert.get_sentence_embedding_dimension)
            return int(fn())
        return 2 ** 14  # hashing feature space

    def _ensure_loaded(self) -> None:
        if self._backend == "sbert" and self._sbert is None:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(
                self.cfg.sbert_model, device=self.cfg.device
            )
        if self._backend == "tfidf" and self._hasher is None:
            # Stateless: lowercase, word 1-2 grams, 16384 features.
            self._hasher = HashingVectorizer(
                n_features=2 ** 14,
                ngram_range=(1, 2),
                alternate_sign=True,
                norm=None,
                lowercase=True,
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text, using the cache when possible."""
        text = (text or "").strip()
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        vec = self._encode_batch([text])[0]
        self.cache.put(text, vec)
        return vec

    def embed_many(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts (cached hits are skipped, misses are batched).

        Returns an (N, D) float32 matrix with rows L2-normalized.
        """
        texts = [(t or "").strip() for t in texts]
        n = len(texts)
        if n == 0:
            return np.zeros((0, self.dim if self._backend == "tfidf" else 0),
                            dtype=np.float32)

        # Resolve cache hits first.
        results: list[np.ndarray | None] = [None] * n
        miss_idx: list[int] = []
        for i, t in enumerate(texts):
            cached = self.cache.get(t)
            if cached is not None:
                results[i] = cached
            else:
                miss_idx.append(i)

        # Batch-encode the misses.
        for start in range(0, len(miss_idx), self.cfg.embedding_batch_size):
            chunk_pos = miss_idx[start:start + self.cfg.embedding_batch_size]
            chunk_texts = [texts[i] for i in chunk_pos]
            chunk_vecs = self._encode_batch(chunk_texts)
            for pos, vec in zip(chunk_pos, chunk_vecs):
                results[pos] = vec
                self.cache.put(texts[pos], vec)

        return np.stack(results).astype(np.float32)

    def cosine(self, query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Cosine similarity (both args assumed L2-normalized) -> 1-D array."""
        if matrix.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        q = query_vec.reshape(-1)
        return (matrix @ q).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Internal encoders
    # ------------------------------------------------------------------ #
    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        if self._backend == "sbert":
            vecs = self._sbert.encode(
                texts,
                batch_size=min(self.cfg.embedding_batch_size, max(1, len(texts))),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(vecs, dtype=np.float32)
        # tfidf fallback
        sparse = self._hasher.transform(texts)
        dense = _sk_normalize(sparse, norm="l2", axis=1).toarray()
        return np.asarray(dense, dtype=np.float32)
