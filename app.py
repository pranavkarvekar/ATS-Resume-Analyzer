import re
import os
import sys
import numpy as np
import pandas as pd
import faiss
import pdfplumber
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ──────────────────────────────────────────────────────────
#  STREAMLIT GUARD - Prevent running with `python app.py`
# ──────────────────────────────────────────────────────────
if not hasattr(st, '_is_running_with_streamlit'):
    # Check if running under Streamlit by looking for Streamlit's internal marker
    if 'streamlit' not in sys.modules or not hasattr(st, 'session_state'):
        print("\n" + "="*70)
        print("❌ ERROR: This app must be run with Streamlit, not directly with Python!")
        print("="*70)
        print("\n🚀 CORRECT WAY TO RUN:\n")
        print("   streamlit run app.py\n")
        print("="*70 + "\n")
        sys.exit(1)

# ──────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be the very first Streamlit call)
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ATS Resume Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
#  PREMIUM DARK THEME  CSS
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d0f1a 0%, #111827 60%, #0f172a 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: rgba(15,18,35,0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.2);
    backdrop-filter: blur(20px);
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Hero Header ── */
.hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.1) 50%, rgba(59,130,246,0.1) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 24px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 50% 50%, rgba(99,102,241,0.05) 0%, transparent 70%);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 0.5; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.05); }
}
.hero h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1.1rem;
    margin: 0;
    font-weight: 300;
}

/* ── Cards ── */
.card {
    background: rgba(17,24,39,0.8);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    border-color: rgba(99,102,241,0.5);
    box-shadow: 0 8px 32px rgba(99,102,241,0.15);
}
.card-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.8rem;
}

/* ── Section Labels ── */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6366f1;
    margin-bottom: 0.5rem;
}

/* ── Result Box ── */
.result-box {
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(99,102,241,0.25);
    border-left: 4px solid #6366f1;
    border-radius: 12px;
    padding: 1.5rem;
    color: #cbd5e1;
    line-height: 1.8;
    font-size: 0.95rem;
    white-space: pre-wrap;
}

/* ── Score Badge ── */
.score-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 110px;
    height: 110px;
    border-radius: 50%;
    background: conic-gradient(#6366f1 var(--pct), rgba(99,102,241,0.15) 0);
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0 auto 1rem;
    box-shadow: 0 0 30px rgba(99,102,241,0.3);
}

/* ── Keyword Chips ── */
.chip-matched {
    display: inline-block;
    background: rgba(34,197,94,0.15);
    border: 1px solid rgba(34,197,94,0.4);
    color: #4ade80;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    margin: 3px;
    font-weight: 500;
}
.chip-missing {
    display: inline-block;
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.3);
    color: #f87171;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    margin: 3px;
    font-weight: 500;
}

/* ── Stat Metric ── */
.metric-card {
    background: rgba(17,24,39,0.9);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
}
.metric-value {
    font-family: 'Outfit', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #818cf8;
}
.metric-label {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 0.3rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Streamlit overrides ── */
.stTextArea textarea {
    background: rgba(17,24,39,0.9) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.8rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.35) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(17,24,39,0.6) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}
.stFileUploader {
    background: rgba(17,24,39,0.6) !important;
    border: 2px dashed rgba(99,102,241,0.3) !important;
    border-radius: 14px !important;
}
.stSelectbox > div > div {
    background: rgba(17,24,39,0.9) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
div[data-testid="stAlert"] {
    border-radius: 12px !important;
}
.stSpinner > div {
    border-color: #6366f1 !important;
}
.stSidebar .stMarkdown h2 {
    color: #818cf8 !important;
    font-family: 'Outfit', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  LOAD ENV + LLM
# ──────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 **GROQ_API_KEY** not found in `.env` file!")
    st.stop()

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
@st.cache_resource(show_spinner=False)
def load_resume_dataset():
    dataset_path = os.path.join("Resume", "Resume.csv")
    if not os.path.exists(dataset_path):
        return None
    df = pd.read_csv(dataset_path)[["ID", "Resume_str", "Category"]]
    df.dropna(subset=["Resume_str"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_resource(show_spinner=False)
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

# ──────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:2.5rem;'>📊</div>
        <div style='font-family: Outfit, sans-serif; font-size:1.2rem; font-weight:700;
                    background: linear-gradient(135deg,#818cf8,#a78bfa);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            ATS Analyzer
        </div>
        <div style='color:#475569; font-size:0.75rem; margin-top:0.3rem;'>
            Powered by Llama 3.3 + RAG
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📋 How to Use")
    st.markdown("""
    <div style='color:#94a3b8; font-size:0.85rem; line-height:1.8;'>
    1. Paste your <b style='color:#818cf8;'>Job Description</b><br>
    2. Upload your <b style='color:#818cf8;'>Resume (PDF)</b><br>
    3. Choose an analysis tab<br>
    4. Click the action button<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🔧 Features")
    features = [
        ("🔍", "Resume Review"),
        ("🚀", "Optimization Tips"),
        ("📈", "ATS Score (RAG)"),
        ("🤝", "Job Fit Score"),
        ("📊", "Keyword Coverage"),
        ("🎨", "Design Suggestions"),
        ("🌍", "Multilingual Resume"),
    ]
    for icon, name in features:
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:0.6rem; padding:0.4rem 0;
                    color:#94a3b8; font-size:0.85rem;'>
            <span style='font-size:1rem;'>{icon}</span> {name}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # RAG status indicator
    idx, emb = load_faiss_index()
    rag_ok = idx is not None
    rag_color = "#4ade80" if rag_ok else "#f87171"
    rag_text  = "Active ✓" if rag_ok else "Index not found"
    st.markdown(f"""
    <div style='background:rgba(17,24,39,0.8); border:1px solid rgba(99,102,241,0.2);
                border-radius:10px; padding:0.8rem; text-align:center;'>
        <div style='font-size:0.7rem; color:#475569; text-transform:uppercase;
                    letter-spacing:0.1em; margin-bottom:0.3rem;'>RAG Pipeline</div>
        <div style='color:{rag_color}; font-weight:600; font-size:0.9rem;'>{rag_text}</div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  MAIN CONTENT
# ──────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
    <h1>📄 ATS Resume Analyzer</h1>
    <p>AI-powered resume evaluation with RAG indexing &amp; Llama 3.3 · Get scored, optimized &amp; hire-ready</p>
</div>
""", unsafe_allow_html=True)

# Load resources
df_resumes = load_resume_dataset()
faiss_index, embedder = load_faiss_index()

# ── Inputs ──────────────────────────────────────────────
col_jd, col_up = st.columns([1.1, 0.9], gap="large")

with col_jd:
    st.markdown('<div class="section-label">📝 Job Description</div>', unsafe_allow_html=True)
    job_desc = st.text_area(
        label="Job Description",
        label_visibility="collapsed",
        placeholder="Paste the full job description here…",
        height=200,
    )

with col_up:
    st.markdown('<div class="section-label">📁 Upload Resume</div>', unsafe_allow_html=True)
    upload_file = st.file_uploader(
        label="Upload Resume (PDF only)",
        label_visibility="collapsed",
        type=["pdf"],
    )
    if upload_file:
        st.markdown("""
        <div style='background:rgba(34,197,94,0.1); border:1px solid rgba(34,197,94,0.3);
                    border-radius:10px; padding:0.8rem; margin-top:0.8rem;
                    color:#4ade80; font-size:0.9rem; text-align:center;'>
            ✅ Resume uploaded successfully
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Analysis Tabs ────────────────────────────────────────
if upload_file:
    resume_text = extract_pdf_text(upload_file)
    if not resume_text:
        st.error("❌ Could not extract text. Is this a scanned/image-based PDF?")
        st.stop()

    tabs = st.tabs([
        "🔍 Review",
        "🚀 Optimize",
        "📈 ATS Score",
        "🤝 Job Fit",
        "📊 Keywords",
        "🎨 Design",
        "🌍 Translate",
    ])

    # ── Tab 1: Review ──
    with tabs[0]:
        st.markdown("### 🔍 Resume Review")
        st.markdown("*Get a thorough HR-perspective review of your resume against the job description.*")
        if not job_desc.strip():
            st.warning("⚠️ Please paste a job description first.")
        elif st.button("Run Resume Review", key="btn_review"):
            with st.spinner("Analyzing your resume…"):
                result = get_model_response("review", job_desc=job_desc, resume_text=resume_text)
            st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

    # ── Tab 2: Optimize ──
    with tabs[1]:
        st.markdown("### 🚀 Optimization Tips")
        st.markdown("*Actionable suggestions to make your resume stronger for this role.*")
        if not job_desc.strip():
            st.warning("⚠️ Please paste a job description first.")
        elif st.button("Generate Optimization Tips", key="btn_optimize"):
            with st.spinner("Crafting improvement strategies…"):
                result = get_model_response("optimize", job_desc=job_desc, resume_text=resume_text)
            st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

    # ── Tab 3: ATS Score ──
    with tabs[2]:
        st.markdown("### 📈 ATS Score")
        rag_badge = "🟢 RAG Active" if faiss_index is not None else "🟡 Standard Mode"
        st.markdown(f"*Scored against the job description. {rag_badge} — reference resumes used for calibration.*")
        if not job_desc.strip():
            st.warning("⚠️ Please paste a job description first.")
        elif st.button("Calculate ATS Score", key="btn_score"):
            with st.spinner("Scoring your resume with RAG pipeline…"):
                result = get_model_response(
                    "score",
                    job_desc=job_desc,
                    resume_text=resume_text,
                    df=df_resumes,
                    index=faiss_index,
                    embedder=embedder,
                )
            if faiss_index is not None:
                st.markdown("""
                <div style='background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.3);
                            border-radius:10px; padding:0.6rem 1rem; margin-bottom:1rem;
                            font-size:0.82rem; color:#a5b4fc;'>
                    ✨ Score calibrated using <b>Top-3 similar real-world resumes</b> from the FAISS index.
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

    # ── Tab 4: Job Fit ──
    with tabs[3]:
        st.markdown("### 🤝 Job Fit Score")
        st.markdown("*How well does your overall profile match this position?*")
        if not job_desc.strip():
            st.warning("⚠️ Please paste a job description first.")
        elif st.button("Calculate Job Fit Score", key="btn_fit"):
            with st.spinner("Evaluating job alignment…"):
                result = get_model_response("fit", job_desc=job_desc, resume_text=resume_text)
            st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

    # ── Tab 5: Keywords ──
    with tabs[4]:
        st.markdown("### 📊 Keyword Coverage Analysis")
        st.markdown("*See exactly which JD keywords appear in your resume and which are missing.*")
        if not job_desc.strip():
            st.warning("⚠️ Please paste a job description first.")
        elif st.button("Analyze Keywords", key="btn_keywords"):
            matched, missing = keyword_coverage(job_desc, resume_text)
            total = len(matched) + len(missing)
            coverage_pct = round(len(matched) / total * 100) if total else 0

            # Metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#4ade80;">{len(matched)}</div>
                    <div class="metric-label">Matched</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#f87171;">{len(missing)}</div>
                    <div class="metric-label">Missing</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                color = "#4ade80" if coverage_pct >= 60 else "#fbbf24" if coverage_pct >= 40 else "#f87171"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color};">{coverage_pct}%</div>
                    <div class="metric-label">Coverage</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_k1, col_k2 = st.columns(2)
            with col_k1:
                st.markdown("**✅ Matched Keywords**")
                chips = " ".join([f'<span class="chip-matched">{kw}</span>' for kw in sorted(matched)])
                st.markdown(f'<div style="margin-top:0.5rem;">{chips}</div>', unsafe_allow_html=True)

            with col_k2:
                st.markdown("**❌ Missing Keywords**")
                chips = " ".join([f'<span class="chip-missing">{kw}</span>' for kw in sorted(missing)])
                st.markdown(f'<div style="margin-top:0.5rem;">{chips}</div>', unsafe_allow_html=True)

            # Pie chart with dark theme
            st.markdown("<br>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5), facecolor="#111827")
            ax.set_facecolor("#111827")
            wedge_colors = ["#6366f1", "#1e293b"]
            explode     = (0.05, 0)
            wedges, texts, autotexts = ax.pie(
                [len(matched), len(missing)],
                labels=["Matched", "Missing"],
                autopct="%1.1f%%",
                startangle=90,
                colors=wedge_colors,
                explode=explode,
                textprops={"color": "#e2e8f0", "fontsize": 12},
                wedgeprops={"edgecolor": "#0d0f1a", "linewidth": 2},
            )
            for at in autotexts:
                at.set_color("#ffffff")
                at.set_fontweight("bold")
            ax.axis("equal")
            st.pyplot(fig)
            plt.close(fig)

    # ── Tab 6: Design ──
    with tabs[5]:
        st.markdown("### 🎨 Resume Design Suggestions")
        st.markdown("*ATS-friendliness score, template recommendations, and formatting tips.*")
        if st.button("Analyze Resume Design", key="btn_design"):
            with st.spinner("Evaluating design and structure…"):
                result = get_model_response("design", resume_text=resume_text)
            st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

    # ── Tab 7: Translate ──
    with tabs[6]:
        st.markdown("### 🌍 Translate Resume")
        st.markdown("*Get a professional, ATS-friendly translation of your resume.*")
        lang_options = ["French", "Spanish", "German", "Hindi", "Japanese", "Marathi",
                        "Portuguese", "Italian", "Arabic", "Chinese (Simplified)"]
        lang = st.selectbox("Select Target Language", lang_options, key="lang_select")
        if st.button(f"Translate to {lang}", key="btn_translate"):
            with st.spinner(f"Translating resume to {lang}…"):
                result = get_model_response("translate", resume_text=resume_text, extra=lang)
            st.markdown(f"""
            <div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2);
                        border-radius:10px; padding:0.6rem 1rem; margin-bottom:1rem;
                        font-size:0.82rem; color:#a5b4fc;'>
                🌍 Resume translated to <b>{lang}</b>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

else:
    # ── Empty State ──
    st.markdown("""
    <div style='text-align:center; padding: 4rem 2rem;'>
        <div style='font-size:5rem; margin-bottom:1rem;'>📄</div>
        <div style='font-family:Outfit,sans-serif; font-size:1.5rem; font-weight:600;
                    color:#475569; margin-bottom:0.8rem;'>
            No Resume Uploaded Yet
        </div>
        <div style='color:#334155; font-size:0.95rem; max-width:420px; margin:0 auto;'>
            Paste a job description and upload your PDF resume above to start the analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)
