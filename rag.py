"""PrivAI - RAG part 1: read PDFs -> chunks -> embeddings -> Chroma.

Usage:
    python rag.py ingest              # index every PDF / image inside the docs/ folder
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
DOCS_DIR.mkdir(exist_ok=True)
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


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
MIN_TEXT_CHARS = 30      # a PDF page with less text than this is treated as scanned


def extract_pages(path, progress=None):
    """Yield (page_number, text, used_ocr) for a PDF or an image file."""
    path = Path(path)
    if path.suffix.lower() in IMAGE_EXTS:
        from ocr import ocr_image
        yield 1, ocr_image(path), True
        return
    reader = PdfReader(path)
    total = len(reader.pages)
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        used_ocr = False
        if len(text) < MIN_TEXT_CHARS:           # looks scanned -> read it with OCR
            from ocr import ocr_pdf_page
            text = ocr_pdf_page(path, page_no - 1)
            used_ocr = True
        if progress:
            progress(page_no, total)
        yield page_no, text, used_ocr


def index_file(path, progress=None):
    """Read one file, chunk it, embed it, store it. Returns number of chunks."""
    path = Path(path)
    ids, texts, metas = [], [], []
    for page_no, text, used_ocr in extract_pages(path, progress):
        for j, chunk in enumerate(chunk_text(text)):
            if len(chunk) < 20:                  # skip tiny/empty chunks
                continue
            ids.append(f"{path.name}-p{page_no}-c{j}")
            texts.append(chunk)
            metas.append({"source": path.name, "page": page_no, "ocr": used_ocr})
    if not texts:
        print(f"{path.name}: no text found")
        return 0
    collection.delete(where={"source": path.name})   # replace old version, no duplicates
    collection.upsert(ids=ids, documents=texts,
                      embeddings=embed(texts), metadatas=metas)
    print(f"{path.name}: stored {len(texts)} chunks")
    return len(texts)


def ingest():
    files = sorted(p for p in DOCS_DIR.iterdir()
                   if p.suffix.lower() == ".pdf" or p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print("No PDFs or images found. Put some inside the 'docs' folder first.")
        return
    for f in files:
        index_file(f)
    print("Total chunks in database:", collection.count())


def list_sources():
    """Return {file_name: {"chunks": n, "ocr": True/False}} for the sidebar."""
    data = collection.get(include=["metadatas"])
    out = {}
    for m in data["metadatas"]:
        info = out.setdefault(m["source"], {"chunks": 0, "ocr": False})
        info["chunks"] += 1
        info["ocr"] = info["ocr"] or bool(m.get("ocr"))
    return out


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
            tag = " | OCR" if meta.get("ocr") else ""
            print(f"\n[{meta['source']} | page {meta['page']}{tag} | distance {dist:.3f}]")
            print(doc[:400], "...")
    else:
        print(__doc__)