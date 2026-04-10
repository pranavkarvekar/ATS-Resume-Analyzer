import re
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(
    title="ATS Resume Analyzer API",
    description="AI-powered resume evaluation with RAG indexing & Llama 3.3",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LLM Setup ──────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set")

llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0.5)
parser = StrOutputParser()

# ── RAG Setup (optional — graceful fallback) ───────────────
CACHE_DIR = "data"
INDEX_PATH = os.path.join(CACHE_DIR, "faiss.index")

_faiss_index = None
_embedder = None
_df_resumes = None

try:
    import faiss
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    dataset_path = os.path.join("Resume", "Resume.csv")
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)[["ID", "Resume_str", "Category"]]
        df.dropna(subset=["Resume_str"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        _df_resumes = df

    if os.path.exists(INDEX_PATH):
        _faiss_index = faiss.read_index(INDEX_PATH)
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("RAG pipeline loaded successfully")
except Exception as e:
    print(f"RAG pipeline not available: {e}")

# ── Prompts ────────────────────────────────────────────────
review_prompt = PromptTemplate(
    input_variables=["job_desc", "resume_text"],
    template="""You are a senior HR Manager with 15+ years of experience.
Carefully review the resume below against the job description.

Job Description:
{job_desc}

Resume:
{resume_text}

Provide a detailed review covering:
- Overall impression
- Key strengths (be specific)
- Critical weaknesses
- Alignment with the job role
Format your response with clear sections and bullet points.""",
)

optimize_prompt = PromptTemplate(
    input_variables=["job_desc", "resume_text"],
    template="""You are an expert career coach and resume strategist.
Provide concrete, actionable improvements to the resume to better match this job description.

Job Description:
{job_desc}

Resume:
{resume_text}

Structure your response as:
1. High-Priority Changes (do these first)
2. Skills & Keywords to Add
3. Formatting & Structure Improvements
4. Suggested Resume Summary / Objective
Use bullet points for clarity.""",
)

rag_score_prompt = PromptTemplate(
    input_variables=["examples_text", "job_desc", "resume_text"],
    template="""You are an ATS (Applicant Tracking System) engine with access to benchmark resumes.

Reference Resumes (from similar roles):
{examples_text}

Job Description:
{job_desc}

Candidate Resume:
{resume_text}

Using the reference resumes as context, score this candidate's resume:
1. ATS Score: X/100
2. Keyword Match Rate: X%
3. Experience Relevance: X/10
4. Skills Alignment: X/10
5. Strengths (3-5 bullet points)
6. Gaps & Missing Elements (3-5 bullet points)
7. Final Verdict: [Strong Match / Moderate Match / Weak Match]""",
)

score_prompt = PromptTemplate(
    input_variables=["job_desc", "resume_text"],
    template="""You are an ATS (Applicant Tracking System).

Job Description:
{job_desc}

Resume:
{resume_text}

Score this resume:
1. ATS Score: X/100
2. Keyword Match Rate: X%
3. Strengths (3-5 bullet points)
4. Gaps (3-5 bullet points)
5. Final Verdict: [Strong Match / Moderate Match / Weak Match]""",
)

fit_prompt = PromptTemplate(
    input_variables=["job_desc", "resume_text"],
    template="""You are a career advisor and talent acquisition specialist.

Job Description:
{job_desc}

Resume:
{resume_text}

Calculate and explain the Job Fit Score:
1. Job Fit Score: X/100
2. Cultural Fit Estimate: X/10
3. Experience Fit: X/10
4. Skills Fit: X/10
5. Why this candidate fits (or doesn't)
6. Recommended next steps for the candidate""",
)

design_prompt = PromptTemplate(
    input_variables=["resume_text"],
    template="""You are a professional resume designer and ATS optimization expert.

Resume:
{resume_text}

Provide:
1. ATS Friendliness Score: X/10 (with reasoning)
2. Top 3 Recommended Resume Template Styles for this candidate
3. Section Order Recommendation
4. Font & Layout Tips
5. Keywords to highlight or add
6. Quick wins (changes to make in under 30 minutes)""",
)

translate_prompt = PromptTemplate(
    input_variables=["resume_text", "language"],
    template="""Translate the following resume into {language}.
Ensure it remains professional, culturally appropriate, and ATS-friendly.
Maintain the same structure, bullet points, and format as the original.

Resume:
{resume_text}""",
)

# ── Helpers ────────────────────────────────────────────────
STOPWORDS = {
    "the", "and", "for", "are", "with", "that", "this", "from", "have", "has",
    "will", "your", "been", "also", "more", "into", "than", "their", "they",
    "were", "which",
}


def keyword_coverage(job_desc: str, resume_text: str):
    jd_keywords = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\+\#\.]+\b", job_desc.lower())
    jd_keywords = [w for w in jd_keywords if len(w) > 3 and w not in STOPWORDS]
    jd_keywords = list(set(jd_keywords))
    resume_lower = resume_text.lower()
    matched = [kw for kw in jd_keywords if kw in resume_lower]
    missing = [kw for kw in jd_keywords if kw not in resume_lower]
    return matched, missing


def retrieve_similar_resumes(resume_text: str, top_k: int = 3):
    if _faiss_index is None or _df_resumes is None or _embedder is None:
        return None
    import numpy as np
    query_emb = _embedder.encode([resume_text]).astype("float32")
    distances, indices = _faiss_index.search(query_emb, top_k)
    return _df_resumes.iloc[indices[0]]


def build_examples_text(similar_df) -> str:
    parts = []
    for i, (_, row) in enumerate(similar_df.iterrows(), 1):
        snippet = str(row["Resume_str"])[:600]
        parts.append(f"[Example {i} — Category: {row['Category']}]\n{snippet}...")
    return "\n\n".join(parts)


def get_model_response(mode: str, job_desc: str = "", resume_text: str = "", extra: str = None) -> str:
    if mode == "review":
        return (review_prompt | llm | parser).invoke({"job_desc": job_desc, "resume_text": resume_text})
    elif mode == "optimize":
        return (optimize_prompt | llm | parser).invoke({"job_desc": job_desc, "resume_text": resume_text})
    elif mode == "score":
        similar = retrieve_similar_resumes(resume_text)
        if similar is not None:
            examples_text = build_examples_text(similar)
            return (rag_score_prompt | llm | parser).invoke({
                "examples_text": examples_text,
                "job_desc": job_desc,
                "resume_text": resume_text,
            })
        return (score_prompt | llm | parser).invoke({"job_desc": job_desc, "resume_text": resume_text})
    elif mode == "fit":
        return (fit_prompt | llm | parser).invoke({"job_desc": job_desc, "resume_text": resume_text})
    elif mode == "design":
        return (design_prompt | llm | parser).invoke({"resume_text": resume_text})
    elif mode == "translate":
        return (translate_prompt | llm | parser).invoke({"resume_text": resume_text, "language": extra or "French"})
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ── Pydantic Models ────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    mode: str
    resume_text: str
    job_desc: Optional[str] = ""
    language: Optional[str] = "French"


class KeywordsRequest(BaseModel):
    job_desc: str
    resume_text: str


# ── Endpoints ──────────────────────────────────────────────
@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "rag_available": _faiss_index is not None,
        "model": "llama-3.3-70b-versatile",
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    if not req.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text is required")
    if req.mode in ["review", "optimize", "score", "fit"] and not (req.job_desc or "").strip():
        raise HTTPException(status_code=400, detail="job_desc is required for this mode")
    try:
        result = get_model_response(
            mode=req.mode,
            job_desc=req.job_desc or "",
            resume_text=req.resume_text,
            extra=req.language,
        )
        return {"result": result, "rag_used": req.mode == "score" and _faiss_index is not None}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/keywords")
def keywords(req: KeywordsRequest):
    if not req.job_desc.strip() or not req.resume_text.strip():
        raise HTTPException(status_code=400, detail="Both job_desc and resume_text are required")
    matched, missing = keyword_coverage(req.job_desc, req.resume_text)
    total = len(matched) + len(missing)
    coverage_pct = round(len(matched) / total * 100) if total else 0
    return {
        "matched": sorted(matched),
        "missing": sorted(missing),
        "coverage_pct": coverage_pct,
        "total": total,
    }
