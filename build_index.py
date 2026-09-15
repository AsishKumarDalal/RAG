"""Build FAISS index from Source Document.pdf (run once, fast reuse).
Usage:
    ..\\.venv\\Scripts\\python.exe build_index.py
Reads .env for GOOGLE_API_KEY / GEMINI_MODEL / GEMINI_EMBED_MODEL
"""
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# langchain-google-genai expects GOOGLE_API_KEY, allow GEMINI_API_KEY alias
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# langchain-community is sunset but still hosts FAISS/PyMuPDFLoader (no official
# standalone packages yet). Silence only that sunset notice.
warnings.filterwarnings(
    "ignore",
    message=r".*langchain-community.*sunset.*",
    category=DeprecationWarning,
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_community.vectorstores import FAISS

BASE = Path(__file__).parent
SOURCE_PDF = BASE / "data" / "pdfs" / "Source Document.pdf"
INDEX_DIR = BASE / "index" / "faiss_source"
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001")

# token-saver: small chunks -> top-3 keeps context ~900-1200 tokens
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 120


def main():
    assert SOURCE_PDF.exists(), f"Missing {SOURCE_PDF}. Run download_pdfs.py first."
    assert os.getenv("GOOGLE_API_KEY", "").startswith("AIza") or len(os.getenv("GOOGLE_API_KEY", "")) > 10, \
        "Put your Gemini key in .env as GOOGLE_API_KEY="

    print(f"[1/3] Loading {SOURCE_PDF.name}")
    docs = PyMuPDFLoader(str(SOURCE_PDF)).load()
    print(f"  pages: {len(docs)}")
    # add chunk-agnostic metadata
    for d in docs:
        d.metadata["source"] = "Source Document.pdf"

    print(f"[2/3] Splitting (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"c{i}"
        # PyMuPDFLoader page is 0-based -> make 1-based for citations
        if "page" in c.metadata:
            c.metadata["page"] = int(c.metadata["page"]) + 1
    print(f"  chunks: {len(chunks)}")

    print(f"[3/3] Embedding with {EMBED_MODEL} -> FAISS")
    emb = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    db = FAISS.from_documents(chunks, emb)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    db.save_local(str(INDEX_DIR))
    print(f"Saved to {INDEX_DIR} | chunks={len(chunks)}")
    print("Done. Now ask: ..\\.venv\\Scripts\\python.exe harness.py --ask \"What is the case about?\"")


if __name__ == "__main__":
    main()
