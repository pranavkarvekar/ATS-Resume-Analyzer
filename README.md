# 📄 AI-Powered ATS Resume Analyzer (RAG)

An intelligent Applicant Tracking System (ATS) resume scanner that uses **RAG (Retrieval-Augmented Generation)** with FAISS vector search and Groq LLM to evaluate resumes against job descriptions.

## Features

- **RAG-powered evaluation** — retrieves similar resumes from a 2,400+ resume dataset to provide context-aware scoring
- **FAISS vector search** — fast semantic similarity search using SentenceTransformer embeddings
- **Groq LLM (Llama 3.3 70B)** — generates detailed ATS scores with strengths, weaknesses, and reasoning
- **PDF resume parsing** — extracts text from uploaded PDF resumes
- **Modern UI** — Available in both Streamlit and HTML/CSS/JS versions
- **Pre-built index** — FAISS index is built offline once, so the app starts instantly
- **Multi-language support** — Translate resumes to 10+ languages
- **Vercel-ready** — Deploy frontend + backend to Vercel with zero configuration

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit OR HTML/CSS/JS (PDF.js) |
| **Backend API** | FastAPI + LangChain |
| **LLM** | Groq (Llama 3.3 70B Versatile) |
| **Embeddings** | SentenceTransformer (all-MiniLM-L6-v2) |
| **Vector DB** | FAISS |
| **PDF Parsing** | pdfplumber |
| **Deployment** | Vercel (serverless) or Render |

## 🚀 Quick Start

### Option A: Run Locally with Streamlit

```bash
# 1. Clone repository
git clone https://github.com/pranavkarvekar/ATS-Resume-Analyzer.git
cd ATS-Resume-Analyzer

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
# Create .env file with:
# GROQ_API_KEY=your_groq_api_key_here

# 5. Build FAISS index (one-time, ~2 min)
python build_index.py

# 6. Run Streamlit app
streamlit run app.py
```

Then open http://localhost:8501

### Option B: Run HTML/CSS/JS Frontend Locally

```bash
# 1. Start the FastAPI backend
python -m uvicorn api.main:app --reload

# 2. Open in browser
# Open index.html in your browser (or use Live Server extension)
# Frontend will auto-connect to http://localhost:8000
```

### Option C: Deploy to Vercel ⭐

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete Vercel deployment steps. TL;DR:

```bash
# 1. Push to GitHub
git push origin main

# 2. Connect to Vercel
# Go to vercel.com → Import Project → Select your repo

# 3. Add GROQ_API_KEY environment variable
# In Vercel dashboard → Settings → Environment Variables

# 4. Deploy!
# Vercel auto-deploys on every push
```

Your app will be live at: `https://your-project.vercel.app`

## Project Structure

```
├── index.html              # HTML frontend (for Vercel)
├── app.js                  # Frontend logic (PDF extraction, API calls)
├── styles.css              # Premium dark theme
├── app.py                  # Streamlit app
├── build_index.py          # Build FAISS index (one-time)
├── requirements.txt        # Python (Streamlit) dependencies
├── DEPLOYMENT.md           # Complete Vercel deployment guide
├── .env.example            # Environment variables template
├── .vercelignore           # Vercel deployment exclusions
├── vercel.json             # Vercel frontend config
├── api/
│   ├── main.py             # FastAPI backend
│   ├── requirements.txt    # API dependencies
│   └── vercel.json         # Vercel serverless config
├── Resume/
│   └── Resume.csv          # Resume dataset
└── data/
    ├── faiss.index         # Pre-built FAISS index
    └── embeddings.npy      # Pre-computed embeddings
```

## UI Comparison

| Feature | Streamlit | HTML/CSS/JS |
|---------|-----------|------------|
| **Best for** | Local development | Production / Vercel |
| **Performance** | Fast (interactive) | Very fast (pure JS) |
| **Deployment** | Streamlit Cloud | Vercel (free tier) |
| **Customization** | Limited | Full control |
| **Mobile friendly** | Good | Excellent |

## Troubleshooting

### ⚠️ ScriptRunContext warnings (Streamlit)

**Problem:** If you see warnings like:
```
Thread 'MainThread': missing ScriptRunContext! This warning can be ignored...
```

**Solution:** Always run with Streamlit, not Python:
```bash
streamlit run app.py      # ✅ Correct
python app.py             # ❌ Wrong
```

### CORS errors in browser console

**Check**: The API is configured with CORS enabled. Make sure `api/main.py` includes:
```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```

### API endpoint not responding

**Debug**:
```javascript
// In browser console (F12):
fetch('http://localhost:8000/api/health').then(r => r.json()).then(console.log)
```

### Large data files

If `data/` folder is larger than 50MB, Vercel will reject it. Either:
- Remove before deployment (RAG will still work but without context)
- Use cloud storage (S3) and download at runtime

## License

MIT
