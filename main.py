"""
main.py — FastAPI backend for the Candidate Ranking Engine.

Endpoints
---------
POST /search          Real-time candidate search + ranking
POST /upload-resume   Parse a resume PDF/txt and rank against a job
GET  /health          Health check
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from typing import Any, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from candidate_ranker import rank_candidates
from candidate_ranker.resume import parse_resume_text
from config import SEARCH_CONFIG, WEIGHTS
from data_fetcher import fetch_candidates

app = FastAPI(
    title="Candidate Ranking Engine",
    version="2.0.0",
    description="AI-powered candidate search, ranking, and skill analysis.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #
class SearchRequest(BaseModel):
    job_title: str = ""
    description: str = ""
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    min_experience: Optional[float] = None
    location: str = ""
    remote_ok: bool = True
    top_n: int = 10


class RankFromResumeRequest(BaseModel):
    job_title: str = ""
    description: str = ""
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    min_experience: Optional[float] = None
    top_n: int = 10


# --------------------------------------------------------------------------- #
# POST /search
# --------------------------------------------------------------------------- #
@app.post("/search")
def search_candidates(req: SearchRequest):
    """
    1. Fetch real candidate profiles from GitHub API
    2. Pass them through the existing ranking engine
    3. Return ranked results with explanations
    """
    try:
        fetcher_max = min(SEARCH_CONFIG["max_candidates"], max(req.top_n * 3, 30))
        raw_candidates = fetch_candidates(
            job_title=req.job_title,
            required_skills=req.required_skills,
            location=req.location,
            max_results=fetcher_max,
        )

        if not raw_candidates:
            return {
                "query": req.dict(),
                "meta": {"total_input": 0, "message": "No candidates found. Try broader search terms or check your GitHub token."},
                "results": [],
            }

        job_dict = {
            "title": req.job_title,
            "description": req.description or req.job_title,
            "required_skills": req.required_skills,
            "preferred_skills": req.preferred_skills,
            "min_experience": req.min_experience,
            "location": {"country": req.location} if req.location else {},
            "remote_ok": req.remote_ok,
            "top_n": req.top_n,
        }

        result = rank_candidates(
            raw_candidates,
            job_dict,
            top_n=req.top_n,
            weights=dict(WEIGHTS),
        )
        return result

    except Exception as exc:
        traceback.print_exc()
        return {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


# --------------------------------------------------------------------------- #
# POST /upload-resume
# --------------------------------------------------------------------------- #
@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    job_title: str = Form(""),
    description: str = Form(""),
    required_skills: str = Form(""),
    preferred_skills: str = Form(""),
    min_experience: Optional[float] = Form(None),
    top_n: int = Form(10),
):
    """
    Upload a resume (PDF or .txt), parse it, and rank the candidate
    against the provided job requirements.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()

    # Read file bytes
    raw_bytes = await file.read()

    # Extract text
    resume_text = ""
    if ext == ".pdf":
        resume_text = _extract_pdf_text(raw_bytes)
    else:
        resume_text = raw_bytes.decode("utf-8", errors="replace")

    if not resume_text.strip():
        return {"error": "Could not extract text from the uploaded file."}

    parsed = parse_resume_text(resume_text)

    req_skills = [s.strip() for s in required_skills.split(",") if s.strip()]
    pref_skills = [s.strip() for s in preferred_skills.split(",") if s.strip()]

    candidate = {
        "id": "resume-upload",
        "name": parsed.get("title", "Uploaded Candidate") or "Uploaded Candidate",
        "title": parsed.get("title", ""),
        "summary": resume_text[:500],
        "skills": parsed.get("skills", []),
        "experience_years": parsed.get("experience_years", 0.0),
        "location": {},
        "work_history": [],
        "signals": {},
        "source": "resume_upload",
        "updated_at": "",
        "resume_text": resume_text,
    }

    job_dict = {
        "title": job_title,
        "description": description or job_title,
        "required_skills": req_skills,
        "preferred_skills": pref_skills,
        "min_experience": min_experience,
        "location": {},
        "remote_ok": True,
        "top_n": top_n,
    }

    result = rank_candidates([candidate], job_dict, top_n=top_n)
    result["parsed_resume"] = {
        "title": parsed.get("title", ""),
        "experience_years": parsed.get("experience_years", 0.0),
        "skills_found": parsed.get("skills", []),
    }
    return result


# --------------------------------------------------------------------------- #
# GET /health
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Candidate Ranking Engine",
        "version": "2.0.0",
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _extract_pdf_text(raw: bytes) -> str:
    """Extract text from a PDF.  Tries PyPDF2 first, falls back to pdfminer.six."""
    import io

    # --- Attempt 1: PyPDF2 (lightweight, fast) ---
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(raw))
        pages = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt and txt.strip():
                pages.append(txt.strip())
        combined = "\n".join(pages)
        if combined.strip():
            return combined
    except Exception:
        pass

    # --- Attempt 2: pdfminer.six (handles more PDF variants) ---
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        text = pdfminer_extract(io.BytesIO(raw))
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    return ""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
