import sys
import os
import re
import warnings

# Removed Streamlit-related imports and configurations
import numpy as np
import pandas as pd
import faiss
import pdfplumber
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Removed all Streamlit (`st`) references and configurations

# ──────────────────────────────────────────────────────────
#  LOAD ENV + LLM
# ──────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("🚨 **GROQ_API_KEY** not found in `.env` file!")

llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0.5)
parser = StrOutputParser()

# ──────────────────────────────────────────────────────────
#  PATHS
# ──────────────────────────────────────────────────────────
CACHE_DIR  = "data"
INDEX_PATH = os.path.join(CACHE_DIR, "faiss.index")
EMB_PATH   = os.path.join(CACHE_DIR, "embeddings.npy")

# ──────────────────────────────────────────────────────────
#  CACHED LOADERS
# ──────────────────────────────────────────────────────────
def load_resume_dataset():
    dataset_path = os.path.join("Resume", "Resume.csv")
    if not os.path.exists(dataset_path):
        return None
    df = pd.read_csv(dataset_path)[["ID", "Resume_str", "Category"]]
    df.dropna(subset=["Resume_str"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_faiss_index():
    if not os.path.exists(INDEX_PATH):
        return None, None
    index    = faiss.read_index(INDEX_PATH)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return index, embedder

# ──────────────────────────────────────────────────────────
#  RAG RETRIEVAL
# ──────────────────────────────────────────────────────────
def retrieve_similar_resumes(resume_text, df, index, embedder, top_k=3):
    query_emb = embedder.encode([resume_text]).astype("float32")
    distances, indices = index.search(query_emb, top_k)
    return df.iloc[indices[0]]

def build_examples_text(similar_df):
    parts = []
    for i, (_, row) in enumerate(similar_df.iterrows(), 1):
        snippet = str(row["Resume_str"])[:600]
        parts.append(f"[Example {i} — Category: {row['Category']}]\n{snippet}...")
    return "\n\n".join(parts)

# ──────────────────────────────────────────────────────────
#  PDF EXTRACTION
# ──────────────────────────────────────────────────────────
def extract_pdf_text(uploaded_file):
    try:
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text.strip()
    except Exception:
        return None

# ──────────────────────────────────────────────────────────
#  PROMPTS
# ──────────────────────────────────────────────────────────
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
Format your response with clear sections and bullet points."""
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
Use bullet points for clarity."""
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
5. Strengths (3–5 bullet points)
6. Gaps & Missing Elements (3–5 bullet points)
7. Final Verdict: [Strong Match / Moderate Match / Weak Match]"""
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
3. Strengths (3–5 bullet points)
4. Gaps (3–5 bullet points)
5. Final Verdict: [Strong Match / Moderate Match / Weak Match]"""
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
6. Recommended next steps for the candidate"""
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
6. Quick wins (changes to make in under 30 minutes)"""
)

translate_prompt = PromptTemplate(
    input_variables=["resume_text", "language"],
    template="""Translate the following resume into {language}.
Ensure it remains professional, culturally appropriate, and ATS-friendly.
Maintain the same structure, bullet points, and format as the original.

Resume:
{resume_text}"""
)

# ──────────────────────────────────────────────────────────
#  MODEL CALL
# ──────────────────────────────────────────────────────────
def get_model_response(mode, job_desc="", resume_text="", extra=None,
                        df=None, index=None, embedder=None):
    try:
        if mode == "review":
            chain = review_prompt | llm | parser
            return chain.invoke({"job_desc": job_desc, "resume_text": resume_text})

        elif mode == "optimize":
            chain = optimize_prompt | llm | parser
            return chain.invoke({"job_desc": job_desc, "resume_text": resume_text})

        elif mode == "score":
            # Use RAG if index is available
            if index is not None and df is not None and embedder is not None:
                similar = retrieve_similar_resumes(resume_text, df, index, embedder)
                examples_text = build_examples_text(similar)
                chain = rag_score_prompt | llm | parser
                return chain.invoke({
                    "examples_text": examples_text,
                    "job_desc": job_desc,
                    "resume_text": resume_text
                })
            else:
                chain = score_prompt | llm | parser
                return chain.invoke({"job_desc": job_desc, "resume_text": resume_text})

        elif mode == "fit":
            chain = fit_prompt | llm | parser
            return chain.invoke({"job_desc": job_desc, "resume_text": resume_text})

        elif mode == "design":
            chain = design_prompt | llm | parser
            return chain.invoke({"resume_text": resume_text})

        elif mode == "translate":
            chain = translate_prompt | llm | parser
            return chain.invoke({"resume_text": resume_text, "language": extra})

        return "Invalid mode selected."
    except Exception as e:
        return f"⚠️ Error calling model: {e}"

# ──────────────────────────────────────────────────────────
#  KEYWORD COVERAGE
# ──────────────────────────────────────────────────────────
STOPWORDS = {
    "the","and","for","are","with","that","this","from","have","has","will",
    "your","been","also","more","into","than","their","they","were","which"
}

def keyword_coverage(job_desc, resume_text):
    jd_keywords = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\+\#\.]+\b", job_desc.lower())
    jd_keywords = [w for w in jd_keywords if len(w) > 3 and w not in STOPWORDS]
    jd_keywords = list(set(jd_keywords))

    resume_lower = resume_text.lower()
    matched = [kw for kw in jd_keywords if kw in resume_lower]
    missing = [kw for kw in jd_keywords if kw not in resume_lower]
    return matched, missing
