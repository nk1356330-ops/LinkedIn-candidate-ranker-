"""
cache.py
========
A small, dependency-free, disk-backed vector cache for embeddings.

Why it exists
-------------
For 10K-100K candidates, re-running a transformer on every query is wasteful
because candidate profile text rarely changes between queries. We key each
vector by sha1(backend + text) and persist it to disk so subsequent queries
re-use the embeddings (the job-query embedding changes, but candidate
embeddings are stable). This is the "vector caching" the spec calls for.

Format: a single compressed .npz containing a stacked (N, D) float32 matrix
plus an object array of keys. On load we rebuild an in-memory dict. For very
large scale you would swap this implementation for FAISS / LMDB without
changing the public interface.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import numpy as np


class VectorCache:
    """Dict-like cache mapping text keys -> normalized float32 vectors."""

    def __init__(self, cache_dir: Optional[str] = None, backend_tag: str = "v1"):
        self.cache_dir = cache_dir
        self.backend_tag = backend_tag
        self._store: dict[str, np.ndarray] = {}
        self._dirty = False
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._load()

    # ----- path ---------------------------------------------------------- #
    @property
    def _path(self) -> Optional[str]:
        return os.path.join(self.cache_dir, "embeddings.npz") if self.cache_dir else None

    def _load(self) -> None:
        p = self._path
        if not p or not os.path.exists(p):
            return
        try:
            data = np.load(p, allow_pickle=True)
            keys = data["keys"]
            vecs = data["vectors"]
            for k, v in zip(keys, vecs):
                self._store[str(k)] = np.asarray(v, dtype=np.float32)
        except Exception as exc:  # corrupted cache -> start fresh
            print(f"[VectorCache] could not load cache ({exc}); starting fresh")

    def save(self) -> None:
        if not self._path or not self._dirty or not self._store:
            return
        keys = np.array(list(self._store.keys()), dtype=object)
        vecs = np.stack([self._store[k] for k in self._store.keys()]).astype(np.float32)
        # numpy appends '.npz' unless the name already ends with it, so make
        # sure the temp name ends in '.npz' for a clean atomic replace.
        tmp = self._path[:-4] + ".tmp.npz" if self._path.endswith(".npz") \
            else self._path + ".npz"
        np.savez_compressed(tmp, keys=keys, vectors=vecs)
        os.replace(tmp, self._path)
        self._dirty = False

    # ----- dict-like API ------------------------------------------------- #
    @staticmethod
    def make_key(text: str, backend_tag: str) -> str:
        digest = hashlib.sha1(f"{backend_tag}::{text}".encode("utf-8")).hexdigest()
        return digest

    def get(self, text: str) -> Optional[np.ndarray]:
        return self._store.get(self.make_key(text, self.backend_tag))

    def has(self, text: str) -> bool:
        return self.make_key(text, self.backend_tag) in self._store

    def put(self, text: str, vec: np.ndarray) -> None:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        self._store[self.make_key(text, self.backend_tag)] = v
        self._dirty = True

    def __len__(self) -> int:
        return len(self._store)

    def __enter__(self) -> "VectorCache":
        return self

    def __exit__(self, *exc) -> None:
        self.save()
