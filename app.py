"""
app.py — Streamlit frontend for the Candidate Ranking Engine.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import itertools
import json
import os
import re
import sys
import urllib.parse

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import WEIGHTS

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Candidate Ranker",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# --------------------------------------------------------------------------- #
# Sidebar — settings
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Settings")
api_url = st.sidebar.text_input("API URL", value=API_BASE)
top_n = st.sidebar.slider("Results to show", 5, 50, 10, 5)
st.sidebar.markdown("---")
st.sidebar.markdown("### Scoring Weights")
for k, v in WEIGHTS.items():
    st.sidebar.markdown(f"- **{k}**: {v}")

if not st.sidebar.button("🧹 Clear cache"):
    pass

st.sidebar.markdown("---")
st.sidebar.caption("Candidate Ranking Engine v2.0")

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def call_api(endpoint: str, payload: dict) -> dict | None:
    url = f"{api_url.rstrip('/')}{endpoint}"
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f" Cannot connect to backend at `{url}`. Is FastAPI running?")
        st.info("Start the backend: `uvicorn main:app --reload`")
        return None
    except requests.exceptions.Timeout:
        st.error(" Request timed out. Try fewer skills or a smaller result count.")
        return None
    except requests.exceptions.RequestException as exc:
        st.error(f" API error: {exc}")
        return None


def highlight_skills(text: str, skills: list[str]) -> str:
    if not skills:
        return text
    escaped = [re.escape(s) for s in skills if s]
    if not escaped:
        return text
    pattern = "|".join(escaped)
    return re.sub(
        f"({pattern})",
        r'<mark style="background:#d4fcbc;padding:1px 3px;border-radius:3px">\1</mark>',
        text,
        flags=re.IGNORECASE,
    )


def skill_gap_analysis(candidate: dict, required: list[str]) -> list[dict]:
    matched = set(candidate.get("matched_required_skills", []))
    missing = set(candidate.get("missing_required_skills", []))
    rows = []
    for skill in required:
        rows.append({
            "skill": skill,
            "status": "✅ Matched" if skill in matched else "❌ Missing",
        })
    return rows


# --------------------------------------------------------------------------- #
# Main UI
# --------------------------------------------------------------------------- #
st.title("🏆 Candidate Ranking Engine")
st.markdown(
    "Search real developer profiles from **GitHub**, rank them by AI-powered "
    "scoring, and get explainable hiring recommendations."
)

tab1, tab2 = st.tabs(["🔍 Search Candidates", "📄 Upload Resume"])

# =========================================================================== #
# TAB 1 — Search
# =========================================================================== #
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        job_title = st.text_input(
            "Job Title",
            placeholder="e.g. Senior ML Engineer, Data Scientist",
            help="Enter the role you're hiring for.",
        )

    with col2:
        min_exp = st.number_input(
            "Min Experience (years)",
            min_value=0.0, max_value=30.0, value=2.0, step=0.5,
        )

    col3, col4 = st.columns(2)
    with col3:
        skills_input = st.text_area(
            "Required Skills (comma-separated)",
            placeholder="python, pytorch, machine learning, kubernetes, aws",
            height=80,
        )
    with col4:
        location = st.text_input(
            "Location (optional)",
            placeholder="e.g. India, United States, remote",
            help="Filter candidates by location on GitHub.",
        )

    search_clicked = st.button("🔍 Search Candidates", type="primary", use_container_width=True)

    if search_clicked:
        if not job_title and not skills_input:
            st.warning("Enter a job title or at least one skill.")
            st.stop()

        skills = [s.strip() for s in skills_input.split(",") if s.strip()]

        with st.spinner("Fetching real candidates from GitHub + ranking..."):
            payload = {
                "job_title": job_title,
                "description": job_title,
                "required_skills": skills,
                "preferred_skills": [],
                "min_experience": min_exp,
                "location": location,
                "remote_ok": True,
                "top_n": top_n,
            }
            result = call_api("/search", payload)

        if result is None:
            st.stop()

        if not result.get("results"):
            st.warning("No matching candidates found. Broaden your search terms or check your GitHub token.")
            meta = result.get("meta", {})
            if meta.get("message"):
                st.info(meta["message"])
            st.stop()

        meta = result.get("meta", {})
        st.success(
            f"Found **{meta.get('total_input', 0)}** profiles "
            f"(filtered to **{meta.get('returned', 0)}** ranked results) "
            f"using {meta.get('embedding_backend', '?')} backend "
            f"in {meta.get('elapsed_seconds', 0):.2f}s"
        )

        # -- Results table -- #
        results = result["results"]

        # Summary metrics
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        avg_score = sum(r["score"] for r in results) / len(results) if results else 0
        avg_conf = sum(r.get("confidence", 0) for r in results) / len(results) if results else 0
        mcol1.metric("Ranked Candidates", len(results))
        mcol2.metric("Avg Score", f"{avg_score:.3f}")
        mcol3.metric("Avg Confidence", f"{avg_conf:.3f}")
        mcol4.metric("Backend", meta.get("embedding_backend", "?").upper())

        # Tabular view
        df_rows = []
        for i, r in enumerate(results, 1):
            df_rows.append({
                "Rank": i,
                "Name": r["name"],
                "Score": f"{r['score']:.3f}",
                "Confidence": f"{r.get('confidence', 0):.2f}",
                "Title": r.get("title", ""),
                "Experience": r.get("experience", ""),
                "Company": r.get("current_company", ""),
                "Matched Skills": ", ".join(r.get("matched_required_skills", [])[:5]),
                "Missing Skills": ", ".join(r.get("missing_required_skills", [])[:3]),
            })

        df = pd.DataFrame(df_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("📋 Detailed Results")

        # Detail expanders
        for i, r in enumerate(results, 1):
            with st.expander(f"#{i}  **{r['name']}**  —  Score: **{r['score']:.3f}**"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Score", f"{r['score']:.3f}")
                    st.metric("Confidence", f"{r.get('confidence', 0):.3f}")
                    st.metric("Experience", r.get("experience", ""))
                with col2:
                    st.markdown(f"**{r.get('title', 'N/A')}**  \n{r.get('current_company', '')}  \n{json.dumps(r.get('location', {}))}  \nSource: {r.get('source', 'N/A')}")

                st.markdown("---")
                st.markdown("### 🎯 Why This Candidate?")
                st.info(r.get('explanation', ''))

                st.markdown("#### 📊 Score Breakdown")
                comps = r.get("component_scores", {})
                main_comps = {k: v for k, v in comps.items() if k in ("semantic", "skill", "experience", "behavior")}
                if main_comps:
                    c1, c2, c3, c4 = st.columns(4)
                    labels = {"semantic": "Semantic Match", "skill": "Skill Match", "experience": "Experience Fit", "behavior": "Behavior"}
                    for col, key in zip([c1, c2, c3, c4], ["semantic", "skill", "experience", "behavior"]):
                        if key in main_comps:
                            col.metric(labels[key], f"{main_comps[key]:.2f}")

                st.markdown("#### 🔧 Skills")
                matched = r.get("matched_required_skills", [])
                missing = r.get("missing_required_skills", [])
                all_skills = r.get("skills", [])
                highlighted = highlight_skills(", ".join(all_skills), matched)
                st.markdown(
                    f'<div style="font-size:0.9em">{highlighted}</div>',
                    unsafe_allow_html=True,
                )

                if matched:
                    st.markdown("**✅ Matched Required:** " + ", ".join(
                        f'<span style="color:green">{s}</span>' for s in matched
                    ), unsafe_allow_html=True)
                if missing:
                    st.markdown("**❌ Missing Required:** " + ", ".join(
                        f'<span style="color:red">{s}</span>' for s in missing
                    ), unsafe_allow_html=True)

                if missing:
                    st.markdown("#### 📊 Skill Gap Analysis")
                    gap = skill_gap_analysis(r, result["query"].get("required_skills", []))
                    gap_df = pd.DataFrame(gap)
                    st.dataframe(gap_df, hide_index=True, use_container_width=True)

        # JSON download
        st.markdown("---")
        st.download_button(
            "📥 Download Full Results (JSON)",
            data=json.dumps(result, indent=2, default=str),
            file_name="ranked_candidates.json",
            mime="application/json",
        )

# =========================================================================== #
# TAB 2 — Upload Resume
# =========================================================================== #
with tab2:
    st.markdown("Upload a candidate's resume (PDF or .txt) to rank it against a job description.")

    col_a, col_b = st.columns(2)
    with col_a:
        r_job_title = st.text_input("Job Title", key="r_title", placeholder="e.g. ML Engineer")
    with col_b:
        r_min_exp = st.number_input("Min Experience (years)", key="r_exp", min_value=0.0, value=1.0, step=0.5)

    r_skills = st.text_area(
        "Required Skills (comma-separated)",
        key="r_skills",
        placeholder="python, pytorch, machine learning",
    )

    uploaded_file = st.file_uploader(
        "Choose a resume file", type=["pdf", "txt"],
        help="Upload a PDF or plain-text resume.",
    )

    if uploaded_file and st.button("🚀 Rank Resume", type="primary", use_container_width=True):
        r_skills_list = [s.strip() for s in r_skills.split(",") if s.strip()]

        with st.spinner("Parsing resume and ranking..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            data = {
                "job_title": r_job_title,
                "required_skills": r_skills_list,
                "min_experience": r_min_exp,
                "top_n": top_n,
            }
            try:
                url = f"{api_url.rstrip('/')}/upload-resume"
                resp = requests.post(url, files=files, data=data, timeout=60)
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.RequestException as exc:
                st.error(f" API error: {exc}")
                st.stop()

        if "error" in result:
            st.error(result["error"])
            st.stop()

        parsed = result.get("parsed_resume", {})
        st.success("Resume parsed successfully!")

        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("Title", parsed.get("title", "N/A"))
        col_p2.metric("Experience", f"{parsed.get('experience_years', 0):.1f} yrs")
        col_p3.metric("Skills Found", len(parsed.get("skills_found", [])))

        if parsed.get("skills_found"):
            st.markdown("**Extracted Skills:** " + ", ".join(parsed["skills_found"]))

        ranked = result.get("results", [])
        if ranked:
            r = ranked[0]
            st.markdown("---")
            st.subheader("📊 Ranking Result")
            col_r1, col_r2 = st.columns(2)
            col_r1.metric("Overall Score", f"{r['score']:.3f}")
            col_r2.metric("Confidence", f"{r.get('confidence', 0):.3f}")

            st.markdown(f"**Explanation:** {r.get('explanation', '')}")

            matched = r.get("matched_required_skills", [])
            missing = r.get("missing_required_skills", [])
            if matched:
                st.markdown("**✅ Matched Skills:** " + ", ".join(matched))
            if missing:
                st.markdown("**❌ Missing Skills:** " + ", ".join(missing))
        else:
            st.warning("The resume did not score highly against the specified job requirements.")

st.markdown("---")
st.caption(
    "Powered by GitHub API + Candidate Ranking Engine | "
    "[Set GITHUB_TOKEN env var for higher API rate limits]"
)
