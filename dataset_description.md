# Resume Dataset Overview

The foundation of the ATS Resume Analyzer's intelligence relies heavily on its reference dataset. Below is a detailed breakdown of the dataset, its utility, and how it is technically applied within the application's architecture.

## 📊 1. Dataset Description

The system utilizes an offline, locally stored dataset composed of real-world candidate resumes.

- **File Source**: `Resume/Resume.csv`
- **Total Records**: **2,484** unique resumes.
- **Data Attributes**: Each record natively contains an `ID`, raw extracted text (`Resume_str`), HTML formatting (`Resume_html`), and a professional `Category`.
- **Diversity**: The dataset spans **24 distinct industries**, ensuring broad coverage. The top categories include:
  - *Information Technology, Business Development, Engineering, Finance, Accounting, Sales, Healthcare, HR*, and *Aviation*.

> [!NOTE]
> Unlike purely generative AI tools that guess what a good resume looks like based on generic internet training data, this dataset provides concrete, real-world benchmarks specific to professional fields.

---

## 🎯 2. How It Is Helpful for This Project

When an AI grades a resume, it needs a baseline to compare against. The dataset solves three major problems:

1. **Combating Hallucination**: AI models can hallucinate or give arbitrarily high scores if they don't have a grounded standard for comparison.
2. **Industry Specificity**: A good `Chef` resume is formatted completely differently and uses totally different action verbs than a good `Information Technology` resume. The dataset provides the AI with exact blueprints of what standard resumes in those specific fields look like.
3. **Contextual Scoring (RAG)**: By treating the dataset as an "answer key", the AI can objectively say, *"Top candidates in my database highlight X and Y skills. Your resume is missing them, therefore your ATS score is lower."*

---

## ⚙️ 3. How It Is Used and Applied Here

The application applies this dataset using a **Retrieval-Augmented Generation (RAG)** pipeline powered by a Vector Database. Here is the step-by-step technical implementation:

### Phase 1: Offline Indexing (`build_index.py`)
Because it would be too slow to read 2,484 resumes every time a user uploads their PDF, the system pre-processes them once:
1. It uses `SentenceTransformer` (`all-MiniLM-L6-v2`) to read all 2,484 resumes and turn their semantic meaning into numbers (dense vector embeddings).
2. It saves these numbers into a highly optimized Facebook AI Similarity Search database (`data/faiss.index`).

### Phase 2: User Input
When a user uploads their PDF, the application grabs their raw text and mathematically converts it into a vector using the exact same `SentenceTransformer` model.

### Phase 3: Fast Retrieval (`api/index.py`)
1. The app queries the FAISS database with the user's vector.
2. The database instantly calculates the nearest neighbors and returns the **top 3 most similar resumes** from the original dataset.

### Phase 4: Prompt Augmentation
The backend injects these 3 reference resumes directly into the hidden instructions sent to the Llama 3 LLM.

```python
# The hidden prompt structure sent to the LLM
rag_score_prompt = """
You are an ATS (Applicant Tracking System) engine with access to benchmark resumes.

Reference Resumes (from similar roles):
[Here we insert the 3 resumes fetched from the FAISS database]

Candidate Resume:
[The user's uploaded PDF text]

Using the reference resumes as context, score this candidate's resume...
"""
```

Ultimately, the dataset serves as the "ground truth" that empowers the LLM to give highly accurate, comparative ATS scores rather than generic feedback!
