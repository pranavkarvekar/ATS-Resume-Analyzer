# 📄 AI-Powered ATS Resume Analyzer (RAG)

An intelligent Applicant Tracking System (ATS) resume scanner that uses **RAG (Retrieval-Augmented Generation)** with FAISS vector search and Groq LLM to evaluate resumes against job descriptions.

## Features

- **RAG-powered evaluation** — retrieves similar resumes from a 2,400+ resume dataset to provide context-aware scoring
- **FAISS vector search** — fast semantic similarity search using SentenceTransformer embeddings
- **Groq LLM (Llama 3.3 70B)** — generates detailed ATS scores with strengths, weaknesses, and reasoning
- **PDF resume parsing** — extracts text from uploaded PDF resumes
- **Pre-built index** — FAISS index is built offline once, so the app starts instantly

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| LLM | Groq (Llama 3.3 70B Versatile) |
| Embeddings | SentenceTransformer (all-MiniLM-L6-v2) |
| Vector DB | FAISS |
| Orchestration | LangChain |
| PDF Parsing | pdfplumber |

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/pranavkarvekar/ATS-Resume-Analyzer.git
cd ATS-Resume-Analyzer
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Add your API key

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Add the resume dataset

Download `Resume.csv` and place it in:

```
Resume/Resume.csv
```

### 4. Build the FAISS index (one-time, ~2 min)

```bash
python build_index.py
```

### 5. Launch the app

```bash
streamlit run app.py
```

## Project Structure

```
├── app.py              # Streamlit app (loads pre-built index)
├── build_index.py      # One-time script to build FAISS index
├── requirements.txt    # Python dependencies
├── .env                # API key (not committed)
├── Resume/
│   └── Resume.csv      # Resume dataset (not committed)
└── data/
    ├── faiss.index      # Pre-built FAISS index (generated)
    └── embeddings.npy   # Pre-computed embeddings (generated)
```

## License

MIT
