# GEP June 2026 - structure-aware RAG (200 pages)

Free tools only: local `BAAI/bge-small-en-v1.5`, BM25, local MiniLM cross-encoder, Gemini free tier.

## Pipeline
1. **Ingest** (`src/ingest.py`): PyMuPDF, fonts classify every line (h1 / h2 / box / table / body / note). PDF bookmarks give chapter ranges. Repairs the dropped "Th" ligature ("Te" -> "The") and ligature glyphs; drops chart labels; rebuilds TABLE rows from word positions.
2. **Structure** (`src/structure.py`): section tree with chapter, h1, h2, box, region and a heading path.
3. **Chunk** (`src/chunking.py`): ~280-token sentence children inside ~900-token parents. Search children, answer from parents. References are not indexed.
4. **Index** (`src/embeddings.py`): dense + BM25 over "heading path + passage", so "SAR" text is found by "South Asia".
5. **Query plan** (`src/query.py`): rules for explicit region / chapter / executive summary; one cached Gemini call for country-named questions (Pakistan -> Middle East, North Africa, Afghanistan, and Pakistan). Chapter is never guessed from topic.
6. **Retrieve** (`src/retrieval.py`): metadata filter, RRF (k=60), cross-encoder rerank + small heading-title bonus, collapse to parents (3 passages, ~2600 token budget).
7. **Generate** (`src/generation.py`): grounded Gemini prompt, passages labelled with chapter > section, pages; exact refusal when absent.

## Run
    ./venv/bin/python main.py -q "What is the growth outlook for South Asia in this report?"
    ./venv/bin/python evaluate.py                    # retrieval + generation
    ./venv/bin/python evaluate.py --retrieval-only --ablate
    ./venv/bin/python main.py --reindex              # rebuild index

Notes: statistical-annex and TABLE chunks are indexed but searched only when the question mentions a table / annex / statistics.
