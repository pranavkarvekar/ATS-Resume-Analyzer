"""
build_index.py  --  Run this ONCE before launching the app.

    python build_index.py

It encodes all resumes with SentenceTransformer, builds a FAISS index,
and saves both to the `data/` folder.  After this the Streamlit app
starts instantly every time.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Fix Windows console encoding for emoji / unicode
sys.stdout.reconfigure(encoding="utf-8")

# -- Paths -----------------------------------------------------------------
DATASET_PATH = os.path.join("Resume", "Resume.csv")
CACHE_DIR    = "data"
INDEX_PATH   = os.path.join(CACHE_DIR, "faiss.index")
EMB_PATH     = os.path.join(CACHE_DIR, "embeddings.npy")

os.makedirs(CACHE_DIR, exist_ok=True)

# -- Load dataset ----------------------------------------------------------
print("[1/4] Loading resume dataset ...")
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)[["ID", "Resume_str", "Category"]]
df.dropna(subset=["Resume_str"], inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"       Loaded {len(df)} resumes.")

# -- Encode ----------------------------------------------------------------
print("[2/4] Encoding resumes with SentenceTransformer (all-MiniLM-L6-v2) ...")
t0 = time.time()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(
    df["Resume_str"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
)
print(f"       Encoded in {time.time() - t0:.1f}s  shape={embeddings.shape}")

# -- Build FAISS index -----------------------------------------------------
print("[3/4] Building FAISS IndexFlatL2 ...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings, dtype="float32"))
print(f"       Index contains {index.ntotal} vectors.")

# -- Save to disk ----------------------------------------------------------
print("[4/4] Saving to disk ...")
faiss.write_index(index, INDEX_PATH)
np.save(EMB_PATH, embeddings)
print(f"       Saved:  {INDEX_PATH}")
print(f"       Saved:  {EMB_PATH}")
print("\nDone!  You can now launch the app:  streamlit run app.py")
