import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# --------------------------------------------------------------------------- #
# API Keys
# --------------------------------------------------------------------------- #

# GitHub Personal Access Token (free) — raises rate limit 60→5000 req/hr
# Create at: https://github.com/settings/tokens
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")

# LinkFinder AI — LinkedIn profile lookup & AI lead search
# Sign up: https://linkfinderai.com/sign-up
LINKFINDERAI_KEY: str = os.getenv("LINKFINDERAI_KEY", "")


# --------------------------------------------------------------------------- #
# Data source selection
# --------------------------------------------------------------------------- #
DATA_SOURCE: str = os.getenv("DATA_SOURCE", "github")  # "github" | "linkfinderai"

VALID_SOURCES = {"github", "linkfinderai"}

if DATA_SOURCE not in VALID_SOURCES:
    DATA_SOURCE = "github"


# --------------------------------------------------------------------------- #
# Per-source search configuration
# --------------------------------------------------------------------------- #
SEARCH_CONFIG: dict = {
    "max_candidates": 100,
    "results_per_page": 30,
    "min_repos_for_activity": 3,
    "search_pages": 3,
    # LinkFinder AI: max leads to return per search
    "linkfinderai_fetch_count": 25,
}

# --------------------------------------------------------------------------- #
# Tunable scoring weights (applied on top of candidate_ranker's formula)
# --------------------------------------------------------------------------- #
WEIGHTS: dict[str, float] = {
    "semantic":     0.30,
    "skill":        0.35,
    "experience":   0.20,
    "behavior":     0.15,
}

# --------------------------------------------------------------------------- #
# Location preferences
# --------------------------------------------------------------------------- #
PREFERRED_COUNTRIES: list[str] = os.getenv(
    "PREFERRED_COUNTRIES", "India,United States,United Kingdom,Germany,Canada"
).split(",")

REMOTE_OK: bool = os.getenv("REMOTE_OK", "true").lower() == "true"
