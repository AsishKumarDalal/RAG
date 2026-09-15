"""Small RAG harness: 1 LLM call per turn + bench + highlight on Final Document.
Usage:
    ..\\.venv\\Scripts\\python.exe harness.py --ask "When was the case decided?"
    ..\\.venv\\Scripts\\python.exe harness.py   (interactive loop)

.env keys: GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_EMBED_MODEL, TOP_K, MAX_OUTPUT_TOKENS
Output per turn: answer + cites + bench {input, output, thinking, total, latency_s} -> bench.csv
Highlight: output/highlighted_<ts>.pdf (Final Document with retrieved spans highlighted)
"""
import argparse
import csv
import logging
import os
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# langchain-community is sunset but still hosts FAISS (no official standalone
# `langchain-faiss` package exists yet). Silence only that sunset notice so the
# existing FAISS index keeps loading without spam.
warnings.filterwarnings(
    "ignore",
    message=r".*langchain-community.*sunset.*",
    category=DeprecationWarning,
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.vectorstores import FAISS

# google-genai logs "Direct use of AFC in Models.generate_content ..." when AFC
# is left enabled. We disable AFC per-call (no tools used here), this filter is
# just belt-and-braces in case a version ignores the flag.
class _AfcWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "automatic function calling (afc)" not in record.getMessage().lower()


logging.getLogger("google_genai.models").addFilter(_AfcWarningFilter())

BASE = Path(__file__).parent
INDEX_DIR = BASE / "index" / "faiss_source"
FINAL_PDF = BASE / "data" / "pdfs" / "Final Document.pdf"
OUT_DIR = BASE / "output"
BENCH_CSV = BASE / "bench.csv"

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001")
TOP_K = int(os.getenv("TOP_K", "3"))
MAX_OUT = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
# Gemini counts thinking/reasoning inside max_output_tokens. Default thinking
# (~800-950 tokens) + MAX_OUT=1000 left ~40 tokens for the answer -> 1 cut-off
# sentence with finish_reason=MAX_TOKENS. Keep thinking small so the answer fits.
THINKING_BUDGET = int(os.getenv("THINKING_BUDGET", "200"))

SYSTEM = (
    "You answer ONLY from CONTEXT. If the answer is not in CONTEXT, reply exactly: NOT FOUND IN SOURCE.\n"
    "Give a full answer: 4-10 sentences in 2-3 short paragraphs, not one sentence. "
    "Cover the key facts, dates, and outcome/reasoning.\n"
    "Every factual sentence must end with a citation like [c2 p.4].\n"
    "Do not use outside knowledge. Do not guess.\n"
)


def load_retriever():
    emb = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    db = FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": TOP_K})


def build_context(docs):
    parts = []
    for d in docs:
        cid = d.metadata.get("chunk_id", "?")
        pg = d.metadata.get("page", "?")
        txt = d.page_content.strip().replace("\n", " ")
        txt = txt[:1200]  # hard cap per chunk -> token saver
        parts.append(f"[{cid} p.{pg}] {txt}")
    return "\n\n".join(parts)


def get_text(resp) -> str:
    """Extract plain answer text. Newer langchain-google-genai returns a list
    of content blocks (with thought signatures) instead of str."""
    c = getattr(resp, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        texts = []
        for b in c:
            if isinstance(b, str):
                texts.append(b)
            elif isinstance(b, dict):
                t = b.get("text")
                if isinstance(t, str) and t and b.get("type", "text") == "text":
                    texts.append(t)
        if texts:
            return "\n".join(texts)
    return str(c)


def extract_usage(resp):
    """Return (in, out, thinking, total) robust across langchain-google-genai versions."""
    inp = out = think = total = 0
    try:
        # AIMessage.usage_metadata (newer)
        um = getattr(resp, "usage_metadata", None)
        if um:
            inp = um.get("input_tokens", 0) or 0
            out = um.get("output_tokens", 0) or 0
            total = um.get("total_tokens", inp + out) or (inp + out)
            # thinking sometimes in output_token_details
            det = um.get("output_token_details", {}) or {}
            think = det.get("reasoning", 0) or det.get("thinking", 0) or 0
        # fallback: response_metadata usage_metadata (google style)
        rm = getattr(resp, "response_metadata", {}) or {}
        gum = rm.get("usage_metadata", {}) or {}
        if gum and inp == 0:
            inp = gum.get("prompt_token_count", 0) or 0
            out = gum.get("candidates_token_count", 0) or 0
            think = gum.get("thoughts_token_count", 0) or think
            total = gum.get("total_token_count", inp + out) or (inp + out + think)
    except Exception:
        pass
    return inp, out, think, total


def highlight_final(docs, save_path: Path):
    """Highlight retrieved chunk spans on Final Document.pdf -> save_path. Returns mapped pages."""
    try:
        import pymupdf
    except ImportError:
        return []
    if not FINAL_PDF.exists():
        print(f"WARN: {FINAL_PDF} missing, skip highlight.")
        return []
    src = pymupdf.open(FINAL_PDF)
    dst = pymupdf.open(FINAL_PDF)  # copy to annotate
    hit_pages = []
    for d in docs:
        # use a 60-char anchor, cleaned
        anchor = re.sub(r"\s+", " ", d.page_content.strip())[:80]
        # take a mid-slice to avoid header junk
        probe = anchor[10:70] if len(anchor) > 70 else anchor
        if len(probe) < 20:
            continue
        for pno, page in enumerate(dst):
            try:
                rects = page.search_for(probe[:50])
            except Exception:
                rects = []
            for r in rects[:3]:
                try:
                    page.add_highlight_annot(r).update()
                    if (pno + 1) not in hit_pages:
                        hit_pages.append(pno + 1)
                except Exception:
                    pass
            if rects:
                break  # found, don't spray all pages
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst.save(save_path, garbage=3, deflate=True)
    dst.close(); src.close()
    return sorted(hit_pages)


def log_bench(row: dict):
    new = not BENCH_CSV.exists()
    with open(BENCH_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ts", "model", "question", "input_tokens", "output_tokens", "thinking_tokens", "total_tokens", "latency_s", "top_k", "highlight_pdf", "hit_pages"])
        if new:
            w.writeheader()
        w.writerow(row)


def ask_once(retriever, llm, question: str, do_highlight=True):
    docs = retriever.invoke(question)
    context = build_context(docs)
    prompt = f"{SYSTEM}\nCONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:"

    t0 = time.perf_counter()  # turn = 1 LLM call
    # No tools are used, so disable automatic function calling (AFC).
    # Otherwise google-genai logs: "Direct use of AFC in
    # Models.generate_content is not recommended..."
    try:
        resp = llm.invoke(prompt, automatic_function_calling={"disable": True})
    except TypeError:
        # Older langchain-google-genai that doesn't forward the kwarg
        resp = llm.invoke(prompt)
    latency = time.perf_counter() - t0

    inp, out, think, total = extract_usage(resp)
    answer = get_text(resp)
    finish = ""
    try:
        finish = (getattr(resp, "response_metadata", {}) or {}).get("finish_reason", "")
    except Exception:
        pass
    if finish == "MAX_TOKENS":
        answer += "\n[WARN: answer cut off (MAX_TOKENS). Raise MAX_OUTPUT_TOKENS or lower THINKING_BUDGET.]"

    hl_path, hit_pages = "", []
    if do_highlight:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        hl_path = str(OUT_DIR / f"highlighted_{ts}.pdf")
        hit_pages = highlight_final(docs, Path(hl_path))

    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL,
        "question": question[:300],
        "input_tokens": inp, "output_tokens": out,
        "thinking_tokens": think, "total_tokens": total,
        "latency_s": round(latency, 2),
        "top_k": TOP_K,
        "highlight_pdf": hl_path, "hit_pages": str(hit_pages),
    }
    log_bench(row)
    cites = [(d.metadata.get("chunk_id"), d.metadata.get("page")) for d in docs]
    return answer, cites, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", default=None)
    ap.add_argument("--no-highlight", action="store_true")
    args = ap.parse_args()

    assert INDEX_DIR.exists(), f"Missing {INDEX_DIR}. Run build_index.py first."
    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0, max_output_tokens=MAX_OUT, thinking_budget=THINKING_BUDGET)
    retriever = load_retriever()
    print(f"Model={MODEL} top_k={TOP_K} max_out={MAX_OUT} think_budget={THINKING_BUDGET} | turn = 1 LLM call")

    def run(q):
        ans, cites, bench = ask_once(retriever, llm, q, do_highlight=not args.no_highlight)
        print("\n--- ANSWER ---\n" + ans)
        print(f"\n cites(source): {cites} | highlight: {bench['highlight_pdf']} pages={bench['hit_pages']}")
        print(f" bench: in={bench['input_tokens']} out={bench['output_tokens']} think={bench['thinking_tokens']} total={bench['total_tokens']} time={bench['latency_s']}s")

    if args.ask:
        run(args.ask)
        return
    print("Interactive (empty line quits).")
    while True:
        try:
            q = input("\nQ> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        run(q)


if __name__ == "__main__":
    main()
