# Architecture, Flow & Logic (in detail)

Small grounded RAG: exactly **1 LLM call per question**, answers come **only from retrieved PDF chunks**, every run is benchmarked and visually grounded with highlights.

## 1. System architecture

```mermaid
flowchart TB
    subgraph SOURCES["Sources"]
        DRIVE["Google Drive folder"]
        SRC_PDF["data/pdfs/Source Document.pdf<br/>(indexed)"]
        FINAL_PDF["data/pdfs/Final Document.pdf<br/>(highlighted, never indexed)"]
    end

    subgraph SETUP["One-time setup"]
        DL["download_pdfs.py<br/>gdown download + PDF filter"]
        BI["build_index.py<br/>load - split - embed - FAISS save"]
        IDX[("index/faiss_source/<br/>FAISS + metadata")]
    end

    subgraph QUERY["Per-question (harness.py)"]
        RET["FAISS retriever<br/>top_k similarity"]
        CTX["build_context()<br/>1200 chars/chunk<br/>[cid p.page] prefix"]
        LLM["ChatGoogleGenerativeAI<br/>temp=0, AFC disabled<br/>1 x invoke()"]
        HL["highlight_final()<br/>pymupdf search + annotate"]
        BENCH[("bench.csv<br/>tokens + latency + hit pages")]
        OUTPDF["output/highlighted_TS.pdf"]
    end

    DRIVE --> DL --> SRC_PDF
    DRIVE --> DL --> FINAL_PDF
    SRC_PDF --> BI --> IDX
    IDX --> RET --> CTX --> LLM
    RET --> HL --> OUTPDF
    LLM --> BENCH
    RET --> BENCH
```

Key separation: `Source Document.pdf` is the **knowledge base** (embedded). `Final Document.pdf` is the **evidence surface** (highlighted, never retrieved from). The FAISS index stores vectors + `chunk_id` (`c0, c1, ...`) + 1-based `page` for citations like `[c10 p.4]`.

Repo map:

| Path | Role |
|---|---|
| `download_pdfs.py` | Drive -> `data/pdfs/`, temp `data/_raw_drive/` (deleted unless `--keep-raw`) |
| `build_index.py` | `Source Document.pdf` -> `index/faiss_source/` (run once, re-run on source/model change) |
| `harness.py` | interactive / `--ask` loop, 1 LLM call + bench + highlight |
| `list_models.py` | lists Gemini models + input/output token limits |
| `.env` | `GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_EMBED_MODEL, TOP_K, MAX_OUTPUT_TOKENS, THINKING_BUDGET` |
| `bench.csv` | append-only benchmark log, one row per turn |
| `output/` | `highlighted_<YYYYMMDD_HHMMSS>.pdf` per question (unless `--no-highlight`) |

Config defaults (`harness.py:55-66`, `.env` overrides): `MODEL=gemini-2.5-flash`, `TOP_K=3`, `MAX_OUT=2048`, `THINKING_BUDGET=200`. Thinking tokens share the `max_output_tokens` budget — a big thinking budget + small max-out truncates the answer (`finish_reason=MAX_TOKENS`).

## 2. Indexing flow (`build_index.py`)

```mermaid
sequenceDiagram
    participant U as User
    participant B as build_index.py
    participant L as PyMuPDFLoader
    participant S as RecursiveCharacterTextSplitter<br/>(1000/120)
    participant E as GoogleGenerativeAIEmbeddings
    participant F as FAISS

    U->>B: python build_index.py
    B->>L: load("Source Document.pdf")
    L-->>B: pages[] (0-based page)
    B->>B: metadata.source = "Source Document.pdf"
    B->>S: split_documents(pages)
    S-->>B: chunks[] (~1000 chars, 120 overlap)
    B->>B: chunk_id = c{i}, page += 1
    B->>E: embed(chunks)
    E-->>B: vectors[]
    B->>F: FAISS.from_documents(chunks, emb)
    F-->>B: save_local("index/faiss_source/")
```

Logic notes:

* Splitter separators: `["\n\n", "\n", ". ", " ", ""]` — keeps paragraphs/sentences intact (`build_index.py:54-57`).
* Overlap 120 chars preserves cross-boundary facts (dates, names).
* `chunk_id` is the citation key; page is made 1-based to match PDF readers.
* Embedding model must match at query time (`EMBED_MODEL` in both files reads the same `.env` key). Changing it requires rebuilding the index.

## 3. Query flow (`harness.py ask_once`, 1 LLM call)

```mermaid
sequenceDiagram
    participant U as User
    participant H as harness.py
    participant R as FAISS retriever (k=TOP_K)
    participant G as Gemini chat model
    participant P as pymupdf (Final Doc)
    participant C as bench.csv

    U->>H: question ("--ask" or "Q>")
    H->>R: invoke(question)
    R-->>H: docs[0..k-1] (content + chunk_id + page)
    H->>H: build_context() (truncate 1200/chunk)
    H->>G: invoke(SYSTEM + CONTEXT + QUESTION)<br/>automatic_function_calling.disable=true
    G-->>H: response (text blocks + usage_metadata)
    H->>H: get_text() + extract_usage()
    H->>P: search chunk probes, add_highlight_annot()
    P-->>H: highlighted_TS.pdf + hit_pages[]
    H->>C: append row (tokens, latency, top_k, hit pages)
    H-->>U: answer + cites + bench line
```

Prompt contract (`harness.py:60-66,187`):

```text
{SYSTEM}
CONTEXT:
[c10 p.4] <chunk text up to 1200 chars>
[c22 p.8] <...>

QUESTION: <user question>
ANSWER:
```

`SYSTEM` rules: answer **only** from context (else exactly `NOT FOUND IN SOURCE`), 4–10 sentences in 2–3 paragraphs, **every factual sentence ends with `[cid p.page]`**, no outside knowledge.

Generation details:

* `temperature=0` — deterministic, grounded answers.
* `automatic_function_calling={"disable": true}` — no tools are used; this silences the `google-genai` warning `Direct use of AFC in Models.generate_content...` (`harness.py:193-197`). A logging filter on `google_genai.models` is belt-and-braces.
* `get_text()` (`harness.py`) joins only `type=text` blocks — newer `langchain-google-genai` returns a block list with thought signatures, not a plain string. Without this the answer prints as a one-line Python repr.
* Truncation guard: if `finish_reason == MAX_TOKENS`, a `[WARN: answer cut off...]` hint is appended — raise `MAX_OUTPUT_TOKENS` or lower `THINKING_BUDGET`.

## 4. Highlight logic (`highlight_final`)

```mermaid
flowchart TD
    A["docs[0..k-1] from retriever"] --> B["anchor = normalizeWhitespace(content)[:80]"]
    B --> C["probe = anchor[10:70]<br/>(skip header junk, need len>=20)"]
    C --> D["for each page in Final Document.pdf:<br/>rects = page.search_for(probe[:50])"]
    D -->|found| E["add_highlight_annot(rect) x max 3<br/>record 1-based page, stop scanning"]
    D -->|not found| F["next page"]
    E --> G["save output/highlighted_TS.pdf<br/>return sorted hit_pages"]
```

* A copy of `Final Document.pdf` is annotated (`pymupdf.open`, never `fitz` — the `fitz` alias is deprecated).
* Only the first matching page per chunk is highlighted to avoid spraying.
* `hit_pages=[]` in `bench.csv` means the probe text wasn't found verbatim (paraphrase/OCR drift) — citations in the answer are still the source of truth.

## 5. Benchmarking (`log_bench` -> `bench.csv`)

One row per turn, columns: `ts, model, question, input_tokens, output_tokens, thinking_tokens, total_tokens, latency_s, top_k, highlight_pdf, hit_pages`.

* `latency_s` = `perf_counter()` around the single `invoke()` — pure model time, retrieval/highlight excluded.
* `output_tokens` **includes** thinking on Gemini — answer budget ≈ `output - thinking`. If `output ≈ MAX_OUTPUT_TOKENS`, the answer was truncated.
* How to compare: fix the question set, vary one `.env` knob (`TOP_K`, `MAX_OUTPUT_TOKENS`, `THINKING_BUDGET`, `GEMINI_MODEL`), re-run, compare `latency_s` / `total_tokens` (cost) / `hit_pages` (grounding). Example observed: `TOP_K 3->5` raised `input 751->1276` but `hit_pages [1,9]->[1,6,9,20]`.
