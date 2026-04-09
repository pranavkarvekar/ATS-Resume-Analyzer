import streamlit as st
import os
import numpy as np
import pandas as pd
import faiss
import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -------------------- Load API Key --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 GROQ_API_KEY not found in .env file!")
    st.stop()

# -------------------- Initialize LLM --------------------
model = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)
parser = StrOutputParser()

# -------------------- Pre-built Index Paths --------------------
CACHE_DIR  = "data"
INDEX_PATH = os.path.join(CACHE_DIR, "faiss.index")
EMB_PATH   = os.path.join(CACHE_DIR, "embeddings.npy")


# -------------------- Load Resume Dataset --------------------
@st.cache_resource
def load_resume_dataset():
    try:
        dataset_path = os.path.join("Resume", "Resume.csv")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        df = pd.read_csv(dataset_path)[["ID", "Resume_str", "Category"]]
        df.dropna(subset=["Resume_str"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


# -------------------- Load Pre-built FAISS Index --------------------
@st.cache_resource
def load_faiss_index():
    """
    Loads the pre-built FAISS index from disk.
    Run  `python build_index.py`  once before starting the app.
    """
    if not os.path.exists(INDEX_PATH):
        return None, None

    index    = faiss.read_index(INDEX_PATH)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return index, embedder


# -------------------- Retrieve Similar Resumes --------------------
def retrieve_similar_resumes(resume_text, df, index, embedder, top_k=3):
    query_emb = embedder.encode([resume_text]).astype("float32")
    distances, indices = index.search(query_emb, top_k)
    return df.iloc[indices[0]]


# -------------------- PDF Text Extraction --------------------
def extract_pdf_text(uploaded_file):
    try:
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception:
        return None


# -------------------- Prompt Template --------------------
score_prompt = PromptTemplate(
    input_variables=["examples_text", "job_desc", "resume_text"],
    template="""
    You are an ATS (Applicant Tracking System) evaluator.
    Use the below examples as references to understand good resumes.

    Example Resumes:
    {examples_text}

    Now analyze the following resume against the given job description.
    Provide:
    1. ATS Score (1–100)
    2. Key strengths
    3. Weaknesses
    4. Reasoning for score.

    Job Description:
    {job_desc}

    Resume:
    {resume_text}
    """
)


# -------------------- RAG Chain --------------------
def get_rag_based_response(job_desc, resume_text, df, index, embedder):
    examples = retrieve_similar_resumes(resume_text, df, index, embedder, top_k=3)
    examples_text = "\n\n".join([
        f"Example Resume ({row.Category}): {row.Resume_str[:500]}..."
        for _, row in examples.iterrows()
    ])
    chain = score_prompt | model | parser
    return chain.invoke({
        "examples_text": examples_text,
        "job_desc": job_desc,
        "resume_text": resume_text
    })


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="📄 ATS Resume Analyzer with RAG", layout="wide")
st.title("📄 AI-Powered ATS Resume Scanner")

# Load dataset (fast — just reads CSV)
df = load_resume_dataset()

# Load pre-built FAISS index (instant — just reads files from disk)
faiss_index, embedder = load_faiss_index()

if faiss_index is None:
    st.error(
        "🚨 FAISS index not found!  Run the build script first:\n\n"
        "```\npython build_index.py\n```\n\n"
        "This is a one-time step (~2 min). After that the app starts instantly."
    )
    st.stop()

st.divider()

job_desc    = st.text_area("📝 Paste Job Description Here", height=150)
upload_file = st.file_uploader("📁 Upload Resume (PDF only)", type="pdf")

if upload_file:
    st.success("✅ Resume uploaded successfully!")

    resume_text = extract_pdf_text(upload_file)
    if not resume_text:
        st.error("❌ Could not extract text from PDF. It may be image-based or empty.")
        st.stop()

    (tab1,) = st.tabs(["📈 RAG-Based ATS Scoring"])

    with tab1:
        if df is None:
            st.warning("RAG scoring is unavailable because the resume dataset failed to load.")
        else:
            if st.button("🔍 Evaluate Resume (RAG-powered)", use_container_width=True):
                if not job_desc.strip():
                    st.warning("⚠️ Please paste a job description before evaluating.")
                else:
                    with st.spinner("🤖 Analyzing resume with dataset-enhanced reasoning…"):
                        response = get_rag_based_response(
                            job_desc, resume_text, df, faiss_index, embedder
                        )
                    st.subheader("📊 ATS Evaluation Result")
                    st.markdown(response)
else:
    st.info("📥 Please upload a PDF resume to begin analysis.")
