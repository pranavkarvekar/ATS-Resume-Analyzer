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
    except Exception as e:
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
    """
)

# -------------------- Prompts --------------------
review_prompt = PromptTemplate(
    input_variables=["job_desc", "resume_text"],
    template="""
    You are an experienced HR Manager. 
    Review the following resume against the job description. 
    Highlight strengths and weaknesses clearly.
    """
)

optimize_prompt = PromptTemplate(
    input_variables=["job_desc", "resume_text"],
    template="""
    You are a career coach. 
    Suggest concrete improvements to the resume so that it better matches the job description. 
    Provide actionable bullet points.

    Job Description:
    {job_desc}

    Resume:
    {resume_text}
    """
)

score_prompt = PromptTemplate(
    input_variables=["job_desc", "resume_text"],
    template="""
    You are an ATS (Applicant Tracking System). 
    Score the resume on a scale of 1-100 against the job description. 
    Also provide reasoning.

    Job Description:
    {job_desc}

    Resume:
    {resume_text}
    """
)

fit_prompt = PromptTemplate(
    input_variables=["job_desc", "resume_text"],
    template="""
    You are a career advisor. 
    Calculate the overall Job Fit Score (0-100) based on skills, experience, and job alignment.
    Also explain why.

    Job Description:
    {job_desc}

    Resume:
    {resume_text}
    """
)

design_prompt = PromptTemplate(
    input_variables=["resume_text"],
    template="""
    You are a professional resume designer. 
    Suggest 3 modern ATS-friendly resume template styles and formatting tips 
    (fonts, layout, sections, keywords). 
    Keep suggestions concise and practical.

    Resume:
    {resume_text}
    """
)

translate_prompt = PromptTemplate(
    input_variables=["resume_text", "language"],
    template="""
    Translate or rewrite this resume into {language}, 
    ensuring it remains professional and ATS-friendly.

    Resume:
    {resume_text}
    """
)

# -------------------- Model Call --------------------
def get_model_response(job_desc, resume_text, mode, extra=None):
    if mode == "review":
        chain = review_prompt | model | parser
    elif mode == "optimize":
        chain = optimize_prompt | model | parser
    elif mode == "score":
        chain = score_prompt | model | parser
    elif mode == "fit":
        chain = fit_prompt | model | parser
    elif mode == "design":
        chain = design_prompt | model | parser
    elif mode == "translate":
        chain = translate_prompt | model | parser
        return chain.invoke({
            "resume_text": resume_text,
            "language": extra
        })
    else:
        return "Invalid mode selected!"

    return chain.invoke({
        "job_desc": job_desc,
        "resume_text": resume_text
    })

# -------------------- Keyword Coverage --------------------
def keyword_coverage(job_desc, resume_text):
    jd_keywords = re.findall(r"\b\w+\b", job_desc.lower())
    jd_keywords = [word for word in jd_keywords if len(word) > 3]  # remove short words
    jd_keywords = list(set(jd_keywords))  # unique

    resume_words = resume_text.lower().split()
    matched = [kw for kw in jd_keywords if kw in resume_words]
    missing = [kw for kw in jd_keywords if kw not in resume_words]

    return matched, missing

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="ATS Resume Analyzer", layout="wide")
st.title("📄 ATS Resume Analyzer")

job_desc = st.text_area("📝 Paste Job Description Here")
upload_file = st.file_uploader("📁 Upload Resume (PDF only)", type='pdf')

if upload_file:
    st.success("✅ Resume uploaded successfully!")

    resume_text = extract_pdf_text(upload_file)
    if not resume_text:
        st.error("❌ Could not extract text from PDF. Is it scanned or image-based?")
        st.stop()

    # Tabs for different features
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Review", "🚀 Optimize", "📈 ATS Score",
        "🤝 Job Fit Score", "📊 Keyword Coverage", "🖼 Design & 🌍 Translate"
    ])

    with tab1:
        if st.button("Run Review"):
            result = get_model_response(job_desc, resume_text, "review")
            st.subheader("📊 Resume Review")
            st.write(result)

    with tab2:
        if st.button("Run Optimization"):
            result = get_model_response(job_desc, resume_text, "optimize")
            st.subheader("💡 Optimization Tips")
            st.write(result)

    with tab3:
        if st.button("Get ATS Score"):
            result = get_model_response(job_desc, resume_text, "score")
            st.subheader("📌 ATS Score")
            st.write(result)

    with tab4:
        if st.button("Get Job Fit Score"):
            result = get_model_response(job_desc, resume_text, "fit")
            st.subheader("🤝 Job Fit Score")
            st.write(result)

    with tab5:
        if st.button("Analyze Keyword Coverage"):
            matched, missing = keyword_coverage(job_desc, resume_text)

            st.subheader("📊 Keyword Coverage")
            st.write(f"✅ Matched Keywords: {', '.join(matched)}")
            st.write(f"❌ Missing Keywords: {', '.join(missing)}")

            # Pie chart visualization
            labels = ["Matched", "Missing"]
            values = [len(matched), len(missing)]

            fig, ax = plt.subplots()
            ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

    with tab6:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Get Resume Design Suggestions"):
                result = get_model_response(None, resume_text, "design")
                st.subheader("🖼 Resume Design Suggestions")
                st.write(result)

        with col2:
            lang = st.selectbox("🌍 Translate Resume To:", ["French", "Spanish", "German", "Hindi", "Japanese","Marathi"])
            if st.button("Translate Resume"):
                result = get_model_response(None, resume_text, "translate", extra=lang)
                st.subheader(f"🌍 Resume in {lang}")
                st.write(result)
else:
    st.info("📥 Please upload a PDF resume to begin analysis.")
