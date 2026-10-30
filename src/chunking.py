"""
Hierarchical (parent/child) chunking over the section tree.

- Parent: a section (or a ~900-token slice of a long one). This is what the
  model reads, so it keeps the surrounding argument.
- Child: ~280-token whole-sentence windows inside one parent. This is what is
  embedded and searched, so matches are precise.

Every chunk inherits the section metadata (chapter, region, box, pages) and its
heading path, which is prepended for embedding / BM25 but not to the stored text.
"""

import re

from src.structure import heading_path

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "no", "vs", "inc", "co", "u.s", "e.g", "i.e", "etc",
    "dept", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov",
    "dec", "sec", "vol", "nos", "al", "fig", "eds", "ed", "approx",
}

PARENT_TOKENS = 900
CHILD_TOKENS = 280
CHILD_OVERLAP = 40
TABLE_ROWS = 22


def approx_tokens(text):
    return int(len(text.split()) * 1.3)


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    candidates = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“(\[])", text)
    sentences, buf = [], []
    for cand in candidates:
        buf.append(cand)
        words = cand.split()
        last = words[-1].rstrip(".!?\"'").lower() if words else ""
        if last not in ABBREVIATIONS and len(last) > 1:
            sentences.append(" ".join(buf))
            buf = []
    if buf:
        if sentences:
            sentences[-1] += " " + " ".join(buf)
        else:
            sentences.append(" ".join(buf))
    return [s.strip() for s in sentences if s.strip()]


def _meta(sec):
    return {
        "chapter_key": sec["chapter_key"],
        "chapter_no": sec["chapter_no"],
        "chapter": sec["chapter"],
        "h1": sec["h1"],
        "h2": sec["h2"],
        "box": sec["box"],
        "region": sec["region"],
        "path": heading_path(sec),
    }


def _split_parent_paragraphs(paragraphs):
    """Group (text, page) paragraphs into parents of ~PARENT_TOKENS, never splitting a paragraph."""
    groups, cur, size = [], [], 0
    for text, page in paragraphs:
        t = approx_tokens(text)
        if cur and size + t > PARENT_TOKENS:
            groups.append(cur)
            cur, size = [], 0
        cur.append((text, page))
        size += t
    if cur:
        groups.append(cur)
    # Fold a tiny trailing slice into the previous parent
    if len(groups) > 1 and sum(approx_tokens(t) for t, _ in groups[-1]) < PARENT_TOKENS * 0.25:
        groups[-2].extend(groups.pop())
    return groups


def _children_of(paragraphs):
    """Sentence-aware sliding windows across the paragraphs of one parent."""
    sentences = []
    for text, page in paragraphs:
        for s in split_sentences(text):
            sentences.append((s, page))
    children, cur, size = [], [], 0

    def emit():
        nonlocal cur, size
        if not cur:
            return
        children.append({
            "text": " ".join(s for s, _ in cur),
            "page_start": cur[0][1],
            "page_end": cur[-1][1],
        })
        keep, kept = [], 0
        for s, pg in reversed(cur):
            t = approx_tokens(s)
            if keep and kept + t > CHILD_OVERLAP:
                break
            keep.insert(0, (s, pg))
            kept += t
        cur, size = keep, kept

    for s, pg in sentences:
        t = approx_tokens(s)
        if size + t > CHILD_TOKENS and size > CHILD_OVERLAP:
            emit()
        cur.append((s, pg))
        size += t
    if cur and (not children or size > CHILD_OVERLAP + 5 or len(cur) == len(sentences)):
        emit()
    elif cur:
        # only overlap carried over: drop it, already covered by the previous child
        pass
    return children


def _table_chunks(sec, item, parents, children, base_meta):
    """One parent per table slice: caption + header row repeated on every slice."""
    table, page = item[1], item[2]
    rows = table["rows"]
    caption = table["text"] or "Statistical table"
    header = rows[:1]
    body = rows[1:] if header else rows
    for i in range(0, max(len(body), 1), TABLE_ROWS):
        slice_rows = header + body[i:i + TABLE_ROWS]
        text = caption + "\n" + "\n".join(slice_rows)
        pid = len(parents)
        parents.append({"id": pid, "text": text, "page_start": page, "page_end": page,
                        "type": "table", **base_meta})
        children.append({"id": len(children), "parent_id": pid, "text": text, "page_start": page,
                         "page_end": page, "type": "table", **base_meta})


def build_chunks(sections):
    parents, children = [], []
    for sec in sections:
        if sec["h1"] == "References":      # bibliographies match everything and answer nothing
            continue
        meta = _meta(sec)
        paragraphs = [(t, pg) for kind, t, pg in sec["items"] if kind in ("body", "note")]
        for item in sec["items"]:
            if item[0] == "table":
                _table_chunks(sec, item, parents, children, meta)
        if not paragraphs:
            continue
        for group in _split_parent_paragraphs(paragraphs):
            pid = len(parents)
            text = "\n\n".join(t for t, _ in group)
            parents.append({
                "id": pid, "text": text, "type": "text",
                "page_start": group[0][1], "page_end": group[-1][1], **meta,
            })
            for ch in _children_of(group):
                children.append({"id": len(children), "parent_id": pid, "type": "text", **ch, **meta})
    return parents, children


def embed_text(child):
    """What the embedder and BM25 see: heading path first, then the passage."""
    return f"{child['path']}\n{child['text']}"


if __name__ == "__main__":
    from src.ingest import ingest_pdf
    from src.structure import build_sections
    parents, children = build_chunks(build_sections(ingest_pdf("data/GEP-Jun-2026.pdf")))
    print(len(parents), "parents,", len(children), "children")
    import statistics
    print("child tokens median", statistics.median(approx_tokens(c["text"]) for c in children if c["type"] == "text"))
    print("parent tokens median", statistics.median(approx_tokens(p["text"]) for p in parents if p["type"] == "text"))
    sa = [c for c in children if c["region"] == "South Asia"]
    for c in sa:
        print(f"\n[{c['id']}] p{c['page_start']}-{c['page_end']} {c['path']}\n{c['text'][:200]}")
