📌 Candidate Ranking Engine (AI-Based Recruitment System)
📖 Overview

The Candidate Ranking Engine is an AI-powered backend system designed to search, analyze, and rank job candidates based on job requirements. It automates the recruitment process by evaluating resumes, skills, experience, and relevance using intelligent scoring techniques.

This system helps recruiters quickly identify the best candidates for a given role with minimal manual effort.

🚀 Features
🔍 Real-time Candidate Search
📄 Resume Parsing (PDF/Text)
🧠 AI-Based Candidate Ranking
🎯 Skill Matching (Required + Preferred)
📊 Experience Filtering
🧾 Explainable Results (Why a candidate is ranked)
⚡ FastAPI Backend (High Performance APIs)
💾 Embedding Cache for Faster Processing
🏗️ Project Structure
ranker/
│
├── app.py                  # Application entry (optional runner)
├── main.py                 # FastAPI backend
├── config.py               # Global configuration
├── data_fetcher.py         # Candidate data fetching logic
├── benchmark.py            # Performance testing
├── requirements.txt        # Dependencies
│
├── candidate_ranker/
│   ├── pipeline.py         # Ranking pipeline
│   ├── scoring.py          # Candidate scoring logic
│   ├── embedding.py        # Text embeddings
│   ├── filters.py          # Candidate filtering
│   ├── resume.py           # Resume parsing
│   ├── explain.py          # Ranking explanation
│   ├── schema.py           # Data models
│   ├── cache.py            # Caching system
│   └── config.py           # Module config
│
├── uploads/                # Uploaded files / docs
└── .embedding_cache/       # Cached embeddings
⚙️ Installation
1. Clone the Repository
git clone <your-repo-link>
cd ranker
2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
3. Install Dependencies
pip install -r requirements.txt
▶️ Running the Application
uvicorn main:app --reload

Server will run at:

http://127.0.0.1:8000

Swagger API Docs:

http://127.0.0.1:8000/docs
📡 API Endpoints
1️⃣ Search Candidates

POST /search

Request Body
{
  "job_title": "Data Analyst",
  "description": "Analyze data and build dashboards",
  "required_skills": ["Python", "SQL"],
  "preferred_skills": ["Power BI"],
  "min_experience": 2,
  "location": "Remote",
  "remote_ok": true,
  "top_n": 5
}
Response
Ranked list of candidates with scores and explanations
2️⃣ Upload Resume & Rank

POST /upload-resume

Upload a resume file (PDF/TXT)
System parses and ranks candidate against job role
3️⃣ Health Check

GET /health

🧠 How It Works
📥 Fetch Candidates
From dataset / external API
🧾 Parse Resume
Extract skills, experience, keywords
🔤 Text Embedding
Convert job + candidate data into vectors
📊 Scoring System
Skill match
Experience match
Semantic similarity
🏆 Ranking
Candidates sorted by final score
📢 Explainability
Shows why candidate got that rank
🛠️ Technologies Used
Python
FastAPI
Pydantic
NLP / Embeddings
Machine Learning Concepts
📌 Use Cases
🧑‍💼 HR Recruitment Automation
📊 Resume Screening System
🤖 AI Hiring Assistant
🏢 Company Hiring Platforms
🔮 Future Enhancements
🌐 LinkedIn API Integration
🧠 Advanced ML Models (Deep Learning)
📊 Dashboard UI (React / Streamlit)
📍 Geo-based candidate filtering
🗣️ Interview chatbot integration

👨‍💻 Author
Naveen Kumar S
