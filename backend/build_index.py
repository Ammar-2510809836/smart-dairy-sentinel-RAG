import os
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# This points correctly to ".../RAG system/"
DATA_DIR = PROJECT_ROOT 
OUTPUT_DIR = Path(__file__).resolve().parent / "rag_index"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800     # characters per chunk
CHUNK_OVERLAP = 200  # character overlap between chunks


def pdf_to_text(pdf_path: Path) -> str:
    """Extracts text from a PDF file."""
    reader = PdfReader(str(pdf_path))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts)


def chunk_text(text: str, source: str) -> List[Dict]:
    """Splits text into overlapping chunks and handles the loop safely."""
    text = text.replace("\r", "\n")
    # collapse multiple newlines
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + CHUNK_SIZE, n)
        chunk = text[start:end].strip()
        
        if chunk:
            chunks.append(
                {
                    "id": len(chunks),
                    "source": source,
                    "text": chunk,
                }
            )
        
        # --- CRITICAL FIX: Stop if we reached the end of the text ---
        if end == n:
            break
            
        # Calculate next start position
        new_start = end - CHUNK_OVERLAP
        
        # Safety: Ensure we always move forward to avoid infinite loops
        # (If overlap is too big, this forces progress)
        if new_start <= start:
            start = end
        else:
            start = new_start

    return chunks


def build_corpus() -> List[Dict]:
    """Scans folders and builds the chunk database."""
    all_chunks: List[Dict] = []

    # Ensure these folder names match exactly what is on your disk
    for sub in ["health_basics", "iot_sensors"]:
        folder = DATA_DIR / sub
        if not folder.exists():
            print(f"[WARN] Folder not found: {folder}")
            continue

        for pdf_file in folder.glob("*.pdf"):
            print(f"[INFO] Reading {pdf_file}")
            raw_text = pdf_to_text(pdf_file)
            
            if not raw_text.strip():
                print(f"[WARN] No text extracted from {pdf_file}")
                continue

            source_label = f"{sub}/{pdf_file.name}"
            chunks = chunk_text(raw_text, source_label)
            print(f"[INFO] -> {len(chunks)} chunks from {pdf_file.name}")
            all_chunks.extend(chunks)

    return all_chunks


def main():
    print("[STEP 1] Building corpus from PDFs...")
    chunks = build_corpus()
    print(f"[INFO] Total chunks: {len(chunks)}")

    if not chunks:
        print("[ERROR] No chunks created. Check your PDFs and folder paths.")
        return

    print("[STEP 2] Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("[STEP 3] Embedding chunks...")
    texts = [c["text"] for c in chunks]
    
    # Generate embeddings
    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True)

    print("[STEP 4] Saving index...")
    # Save JSON metadata
    with open(OUTPUT_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # Save embeddings
    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)

    print(f"[DONE] Saved {len(chunks)} chunks and embeddings to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()