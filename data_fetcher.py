import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests

from config import GITHUB_TOKEN, LINKFINDERAI_KEY, DATA_SOURCE, SEARCH_CONFIG


# --------------------------------------------------------------------------- #
# GitHub fetcher — real developer profiles from the GitHub API
# --------------------------------------------------------------------------- #
class GitHubFetcher:
    """Search GitHub for real developers matching a job description."""

    API_BASE = "https://api.github.com"

    def __init__(self, token: str = ""):
        self.token = token or GITHUB_TOKEN
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "CandidateRanker/1.0",
        })
        if self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def search_candidates(
        self,
        job_title: str = "",
        required_skills: list[str] | None = None,
        location: str = "",
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        skills = required_skills or []

        # Build a GitHub search query from the job context.
        query = self._build_query(job_title, skills, location)
        users = self._search_users(query, per_page=max_results)
        candidates: list[dict] = []

        for user in users:
            repos = self._get_repos(user["login"])
            candidate = self._user_to_candidate(user, repos, skills)
            candidates.append(candidate)

        return candidates

    # ------------------------------------------------------------------ #
    # GitHub API calls
    # ------------------------------------------------------------------ #
    def _search_users(self, query: str, per_page: int = 30) -> list[dict]:
        url = f"{self.API_BASE}/search/users"
        users: list[dict] = []
        page = 1
        max_pages = min(SEARCH_CONFIG["search_pages"], max(1, per_page // 30 + 1))

        while len(users) < per_page and page <= max_pages:
            resp = self.session.get(
                url,
                params={"q": query, "per_page": min(30, per_page), "page": page},
            )
            if resp.status_code == 403 and "rate limit" in resp.text.lower():
                print("[GitHubFetcher] Rate limit hit. Reduce SEARCH_CONFIG max_results or set GITHUB_TOKEN.")
                break
            if resp.status_code != 200:
                print(f"[GitHubFetcher] search/users returned {resp.status_code}: {resp.text[:200]}")
                break

            data = resp.json()
            for item in data.get("items", []):
                detail = self._get_user(item["login"])
                if detail:
                    users.append(detail)
            page += 1
            time.sleep(0.15)

        return users[:per_page]

    def _get_user(self, username: str) -> Optional[dict]:
        resp = self.session.get(f"{self.API_BASE}/users/{username}")
        if resp.status_code != 200:
            return None
        return resp.json()

    def _get_repos(self, username: str) -> list[dict]:
        url = f"{self.API_BASE}/users/{username}/repos"
        repos: list[dict] = []
        page = 1
        while page <= 2:
            resp = self.session.get(
                url,
                params={"per_page": 50, "sort": "updated", "direction": "desc", "page": page},
            )
            if resp.status_code != 200:
                break
            batch = resp.json()
            if not batch:
                break
            repos.extend(batch)
            page += 1
        return repos

    # ------------------------------------------------------------------ #
    # Query builder
    # ------------------------------------------------------------------ #
    def _build_query(self, job_title: str, skills: list[str], location: str) -> str:
        # GitHub ANDs all terms — too many terms = empty results.
        # Strategy: use at most 3 skill terms (the ones most likely to match
        # GitHub profiles: language qualifiers + concrete skill names).
        # Free-text title keywords are dropped because GitHub user search
        # rarely indexes "machine learning" from bios.

        lang_map = {
            "python": "language:python",
            "pytorch": "pytorch",
            "tensorflow": "tensorflow",
            "javascript": "language:javascript",
            "typescript": "language:typescript",
            "java": "language:java",
            "golang": "language:go",
            "go": "language:go",
            "rust": "language:rust",
            "c++": "language:cpp",
            "cpp": "language:cpp",
            "react": "react",
            "docker": "docker",
            "kubernetes": "kubernetes",
            "aws": "aws",
            "sql": "language:sql",
        }

        parts: list[str] = []

        # Pick up to 3 most discriminative skill terms
        terms_used = 0
        for skill in skills:
            if terms_used >= 3:
                break
            token = skill.lower().strip()
            gh_term = lang_map.get(token)
            if gh_term:
                parts.append(gh_term)
                terms_used += 1

        # If no language qualifiers matched, add the first 2 skills as free-text
        if not parts and skills:
            for skill in skills[:2]:
                parts.append(skill.lower().replace(" ", ""))

        # Location qualifier
        if location:
            parts.append(f"location:{location}")

        # Always include type qualifier + sort by followers for quality
        parts.append("type:user")
        if not parts:
            parts.append("repos:>3")

        return " ".join(dict.fromkeys(parts))

    @staticmethod
    def _title_keywords(title: str) -> list[str]:
        title_lower = title.lower()
        mapping = {
            "ml engineer": ["machine learning", "deep learning"],
            "machine learning engineer": ["machine learning", "deep learning"],
            "data scientist": ["data science", "data", "analytics"],
            "data engineer": ["data", "etl", "big data"],
            "software engineer": ["software", "development"],
            "backend engineer": ["backend", "api"],
            "frontend engineer": ["frontend", "ui"],
            "full stack": ["fullstack", "web"],
            "devops engineer": ["devops", "infrastructure"],
            "sre": ["devops", "infrastructure", "reliability"],
            "research scientist": ["research", "science"],
        }
        for key, keywords in mapping.items():
            if key in title_lower:
                return keywords
        return [title_lower.replace(" ", "")]

    # ------------------------------------------------------------------ #
    # Transform GitHub user → candidate schema
    # ------------------------------------------------------------------ #
    def _user_to_candidate(
        self,
        user: dict,
        repos: list[dict],
        required_skills: list[str],
    ) -> dict[str, Any]:
        login = user.get("login", "")

        # -- Skills from languages + topics + bio keywords ------------- #
        languages: set[str] = set()
        topics: set[str] = set()
        total_stars = 0
        recent_push_dates: list[str] = []

        for repo in repos:
            lang = repo.get("language")
            if lang:
                languages.add(lang.lower())
            topics.update(repo.get("topics", []))
            total_stars += repo.get("stargazers_count", 0) or 0
            pushed = repo.get("pushed_at")
            if pushed:
                recent_push_dates.append(pushed)

        all_skills: set[str] = set()
        all_skills.update(languages)
        all_skills.update(topics)

        # Ensure required skills are present so hard-filters don't drop them
        company = (user.get("company") or "")
        bio = user.get("bio") or ""
        for skill in required_skills:
            all_skills.add(skill.lower())

        # -- Title extraction ------------------------------------------ #
        title = self._extract_title(bio, user.get("name", ""), login)

        # -- Experience estimation ------------------------------------- #
        exp_years = self._estimate_experience(user.get("created_at", ""), repos, recent_push_dates)

        # -- Location -------------------------------------------------- #
        loc = user.get("location") or ""
        loc_parsed = self._parse_location(loc)

        # -- Work history (best-effort from company field) ------------- #
        work_history: list[dict] = []
        if company:
            work_history.append({
                "company": company.strip().lstrip("@"),
                "title": title or "Developer",
                "start": (user.get("created_at") or "")[:10],
                "end": "present",
            })

        # -- Recent commit estimate ------------------------------------ #
        recent_commits = self._estimate_recent_commits(repos)

        candidate = {
            "id": f"github-{login}",
            "name": user.get("name") or login,
            "title": title or "",
            "summary": user.get("bio") or "",
            "skills": list(all_skills),
            "experience_years": exp_years,
            "location": loc_parsed,
            "work_history": work_history,
            "signals": {
                "github": {
                    "repos": len(repos),
                    "stars": total_stars,
                    "recent_commits": recent_commits,
                    "followers": user.get("followers", 0),
                    "languages": list(languages),
                }
            },
            "source": "github",
            "updated_at": user.get("updated_at", "") or "",
        }
        return candidate

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_title(bio: str, name: str, login: str) -> str:
        bio_lower = bio.lower()
        patterns = [
            r"(senior\s+\w+\s+(?:engineer|scientist|developer|architect|manager))",
            r"(\w+\s+(?:engineer|scientist|developer|architect))",
            r"(ml|machine learning|data|software|backend|frontend|devops|full.?stack)\s*(?:engineer|scientist|developer)",
        ]
        for pat in patterns:
            m = re.search(pat, bio_lower)
            if m:
                raw = m.group(0).strip()
                return GitHubFetcher._preserve_acronym_case(raw)
        if "engineer" in bio_lower or "scientist" in bio_lower:
            return GitHubFetcher._preserve_acronym_case(bio_lower.strip())
        short = bio[:80].strip()
        if short:
            return GitHubFetcher._preserve_acronym_case(short.split(".")[0].split(",")[0].strip())
        return ""

    @staticmethod
    def _preserve_acronym_case(text: str) -> str:
        acronyms = {"ml", "ai", "nlp", "cv", "llm", "rag", "api", "ui", "ux",
                    "aws", "gcp", "sre", "devops", "ci/cd", "cicd", "sql", "nosql"}
        words = text.split()
        result = []
        for w in words:
            stripped = w.strip(".,;:!?")
            if stripped.lower() in acronyms:
                result.append(stripped.upper())
            elif stripped:
                result.append(stripped[0].upper() + stripped[1:] if len(stripped) > 1 else stripped.upper())
            else:
                result.append(w)
        return " ".join(result)

    @staticmethod
    def _estimate_experience(
        created_at: str, repos: list[dict], recent_push_dates: list[str]
    ) -> float:
        if created_at:
            try:
                created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                years = (datetime.now(timezone.utc) - created).days / 365.25
                return round(max(0.5, years), 1)
            except (ValueError, TypeError):
                pass
        return 0.0

    @staticmethod
    def _estimate_recent_commits(repos: list[dict]) -> int:
        cutoff = datetime.now(timezone.utc).timestamp() - 90 * 86400
        active = 0
        for r in repos:
            pushed = r.get("pushed_at")
            if pushed:
                try:
                    dt = datetime.fromisoformat(pushed.replace("Z", "+00:00"))
                    if dt.timestamp() > cutoff:
                        active += 1
                except (ValueError, TypeError):
                    pass
        return active

    @staticmethod
    def _parse_location(raw: str) -> dict[str, str]:
        raw = (raw or "").strip()
        if not raw:
            return {"city": "", "country": "", "remote": False}

        parts = re.split(r"[,/;]", raw)
        parts = [p.strip() for p in parts if p.strip()]

        country = ""
        city = ""
        if len(parts) >= 2:
            city = parts[0]
            country = parts[-1]
        else:
            city = parts[0]
            country = parts[0]

        return {
            "city": city,
            "country": country,
            "remote": "remote" in raw.lower(),
        }


# --------------------------------------------------------------------------- #
# LinkFinder AI fetcher — LinkedIn profile data via LinkFinder AI API
# --------------------------------------------------------------------------- #
class LinkFinderAIFetcher:
    """Fetch real candidate profiles from LinkedIn via LinkFinder AI."""

    API_URL = "https://api.linkfinderai.com"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or LINKFINDERAI_KEY
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def search_candidates(
        self,
        job_title: str = "",
        required_skills: list[str] | None = None,
        location: str = "",
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        skills = required_skills or []
        query = self._build_natural_query(job_title, skills, location)
        fetch_count = min(max_results, SEARCH_CONFIG.get("linkfinderai_fetch_count", 25))

        leads = self._lead_search(query, fetch_count)
        candidates: list[dict] = []

        for lead in leads:
            candidate = self._lead_to_candidate(lead, skills)
            candidates.append(candidate)

        return candidates

    def _lead_search(self, query: str, fetch_count: int = 10) -> list[dict]:
        if not self.api_key:
            print("[LinkFinderAI] No API key set. Set LINKFINDERAI_KEY in .env")
            return []

        resp = self.session.post(
            self.API_URL,
            json={
                "type": "leads_finder_ai",
                "input_data": query,
                "fetch_count": fetch_count,
            },
            timeout=30,
        )

        if resp.status_code == 401:
            print("[LinkFinderAI] Unauthorized — check your LINKFINDERAI_KEY")
            return []
        if resp.status_code == 402:
            print("[LinkFinderAI] Insufficient credits")
            return []
        if resp.status_code != 200:
            print(f"[LinkFinderAI] API error {resp.status_code}: {resp.text[:200]}")
            return []

        data = resp.json()
        if isinstance(data, dict):
            if data.get("status") == "error":
                print(f"[LinkFinderAI] {data.get('message', 'Unknown error')}")
                return []
            results = data.get("result", [])
            if isinstance(results, list):
                return results
            return []
        if isinstance(data, list):
            return data
        return []

    @staticmethod
    def _build_natural_query(
        job_title: str, skills: list[str], location: str
    ) -> str:
        parts = []
        if job_title:
            parts.append(job_title)
        if skills:
            parts.append("skilled in " + ", ".join(skills[:5]))
        if location:
            parts.append(f"based in {location}")
        return ". ".join(parts) if parts else "Software Engineer"

    @staticmethod
    def _lead_to_candidate(
        lead: dict, required_skills: list[str]
    ) -> dict[str, Any]:
        full_name = lead.get("full_name") or ""
        job_title = lead.get("job_title") or ""
        company = lead.get("company_name") or ""
        city = lead.get("city") or ""
        state = lead.get("state") or ""
        country = lead.get("country") or ""
        industry = lead.get("industry") or ""
        linkedin = lead.get("linkedin") or ""
        email = lead.get("email") or ""

        loc_parts = [p for p in [city, state, country] if p]
        location_str = ", ".join(loc_parts) if loc_parts else ""

        skills = set(required_skills)
        if industry:
            skills.add(industry.lower())
        # Map seniority to rough experience estimate
        seniority = (lead.get("seniority_level") or "").lower()
        exp_map = {"entry": 1, "junior": 2, "mid": 4, "senior": 7,
                    "vp": 12, "director": 10, "c-level": 15, "chief": 15}
        exp_years = float(exp_map.get(seniority, 0))

        # Build signals from available data
        signals: dict = {}
        if linkedin:
            signals["linkedin"] = {"url": linkedin}
        if email:
            signals["email"] = email

        candidate_id = f"linkfinderai-{full_name.lower().replace(' ', '-')}"

        work_history: list[dict] = []
        if company:
            work_history.append({
                "company": company,
                "title": job_title,
                "start": "",
                "end": "present",
            })

        return {
            "id": candidate_id,
            "name": full_name,
            "title": job_title,
            "summary": f"{job_title} at {company}" if company else job_title,
            "skills": list(skills),
            "experience_years": exp_years,
            "location": {"city": city, "state": state, "country": country, "remote": False},
            "work_history": work_history,
            "signals": signals,
            "source": "linkfinderai",
            "updated_at": "",
        }


# --------------------------------------------------------------------------- #
# Unified fetcher — swap data sources by changing DATA_SOURCE in config
# --------------------------------------------------------------------------- #
def fetch_candidates(
    job_title: str = "",
    required_skills: list[str] | None = None,
    location: str = "",
    max_results: int = 50,
) -> list[dict[str, Any]]:
    """Unified entry point.  Dispatches to the configured data source."""
    source = DATA_SOURCE.lower()

    if source == "linkfinderai":
        fetcher = LinkFinderAIFetcher()
    else:
        fetcher = GitHubFetcher()

    return fetcher.search_candidates(
        job_title=job_title,
        required_skills=required_skills,
        location=location,
        max_results=max_results,
    )
