"""
benchmark.py
============
A PERFORMANCE-ONLY stress tool. It generates SYNTHETIC, non-real candidate
records at runtime purely to measure pipeline throughput (parse -> filter ->
dedup -> embed -> score -> diversify). No candidate data is shipped or stored.

    python benchmark.py 10000       # stress-test with 10k synthetic records
    python benchmark.py 100000      # stress-test with 100k synthetic records

The generated names/companies are generic placeholders ("Person_0", "Company_0")
so there is zero ambiguity that this is NOT real candidate data. In production,
replace this generator with your real fetch+normalize output.
"""

from __future__ import annotations

import random
import sys
import time

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))

from candidate_ranker import rank_candidates

random.seed(42)

SKILL_POOL = ["python", "pytorch", "machine learning",
              "retrieval augmented generation", "docker", "kubernetes", "aws",
              "tensorflow", "sql", "spark", "natural language processing",
              "computer vision", "golang", "javascript", "react", "java"]
COMPANIES = [f"Company_{i}" for i in range(12)]
TITLES = ["ML Engineer", "Senior ML Engineer", "Data Scientist",
          "Backend Engineer", "ML Research Engineer", "Staff ML Engineer"]


def make_synthetic(i: int) -> dict:
    """Return one SYNTHETIC placeholder record (NOT real data)."""
    skills = random.sample(SKILL_POOL, random.randint(2, 10))
    n_jobs = random.randint(0, 3)
    work = [{
        "company": random.choice(COMPANIES),
        "title": random.choice(TITLES),
        "start": str(random.randint(2012, 2022)),
        "end": "present" if j == 0 else str(random.randint(2016, 2024)),
    } for j in range(n_jobs)]
    return {
        "id": f"syn-{i:06d}",
        "name": f"Person_{i}",                      # placeholder, not real
        "title": random.choice(TITLES),
        "summary": f"Engineer experienced with {', '.join(skills[:3])}.",
        "skills": skills,
        "experience_years": round(random.uniform(0, 14), 1),
        "location": {"city": f"City_{random.randint(0, 6)}", "country": "India",
                     "remote": random.random() < 0.3},
        "work_history": work,
        "signals": {"github": {"repos": random.randint(0, 30),
                               "stars": random.randint(0, 4000),
                               "recent_commits": random.randint(0, 80)}},
        "source": random.choice(["brightdata", "pdl", "github", "apify"]),
        "updated_at": f"2026-{random.randint(1, 5):02d}-{random.randint(1, 28):02d}",
    }


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    print(f"Generating {n:,} SYNTHETIC placeholder records (not real data) ...")
    t0 = time.perf_counter()
    candidates = [make_synthetic(i) for i in range(n)]
    print(f"  generated in {time.perf_counter()-t0:.2f}s")

    job = {
        "title": "ML Engineer",
        "description": "Build RAG systems and production ML models with PyTorch.",
        "required_skills": ["python", "pytorch", "machine learning"],
        "preferred_skills": ["docker", "aws"],
        "min_experience": 5,
        "location": {"country": "India"},
        "remote_ok": True,
        "top_n": 10,
    }

    for run in (1, 2):
        t = time.perf_counter()
        result = rank_candidates(candidates, job, top_n=10)
        dt = time.perf_counter() - t
        m = result["meta"]
        print(f"\nRun {run}: ranked {n:,} -> {m['returned']} in {dt:.2f}s "
              f"(backend={m['embedding_backend']}, cache={m['cache_size']:,}, "
              f"after_filter={m['after_filter']:,})")
        for r in result["results"][:3]:
            print(f"   {r['score']:.3f}  {r['name']:<16} {r['explanation'][:60]}")


if __name__ == "__main__":
    main()
