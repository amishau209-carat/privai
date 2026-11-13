"""PrivAI - RAG part 1: read PDFs -> chunks -> embeddings -> Chroma.

Usage:
    python rag.py ingest              # index every PDF inside the docs/ folder
    python rag.py search "your question"   # test retrieval
"""
import os
import sys
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"   # no data leaves the machine

import chromadb
from chromadb.config import Settings
import ollama
from pypdf import PdfReader

EMBED_MODEL = "nomic-embed-text"   # small, runs locally via Ollama
DOCS_DIR = Path("docs")
DB_DIR = "db"                      # Chroma saves its data here
CHUNK_SIZE = 800                   # characters per chunk
OVERLAP = 150                      # characters shared between chunks

client = chromadb.PersistentClient(
    path=DB_DIR, settings=Settings(anonymized_telemetry=False)
)
collection = client.get_or_create_collection(
    "docs", metadata={"hnsw:space": "cosine"}
)


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """Cut text into overlapping pieces so ideas are not split badly."""
    text = " ".join(text.split())            # clean extra spaces/newlines
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def embed(texts):
    """Turn a list of texts into a list of vectors (numbers)."""
    vectors = []
    for i in range(0, len(texts), 32):       # send 32 at a time
        res = ollama.embed(model=EMBED_MODEL, input=texts[i:i + 32])
        vectors.extend(res.embeddings)
    return vectors


def ingest():
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found. Put some inside the 'docs' folder first.")
        return
    for pdf in pdfs:
        ids, texts, metas = [], [], []
        for page_no, page in enumerate(PdfReader(pdf).pages, start=1):
            for j, chunk in enumerate(chunk_text(page.extract_text() or "")):
                if len(chunk) < 50:          # skip tiny/empty chunks
                    continue
                ids.append(f"{pdf.name}-p{page_no}-c{j}")
                texts.append(chunk)
                metas.append({"source": pdf.name, "page": page_no})
        if not texts:
            print(f"{pdf.name}: no text found (maybe a scanned PDF - OCR comes later)")
            continue
        collection.upsert(ids=ids, documents=texts,
                          embeddings=embed(texts), metadatas=metas)
        print(f"{pdf.name}: stored {len(texts)} chunks")
    print("Total chunks in database:", collection.count())


def search(question, k=3):
    """Find the k chunks most similar to the question."""
    q_vec = embed([question])
    res = collection.query(query_embeddings=q_vec, n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "ingest":
        ingest()
    elif len(sys.argv) >= 3 and sys.argv[1] == "search":
        for doc, meta, dist in search(" ".join(sys.argv[2:])):
            print(f"\n[{meta['source']} | page {meta['page']} | distance {dist:.3f}]")
            print(doc[:400], "...")
    else:
        print(__doc__)