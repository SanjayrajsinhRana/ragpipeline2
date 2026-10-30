"""
Layout- and font-aware ingestion for the World Bank Global Economic Prospects PDF.

Turns every page into an ordered list of typed elements (h1, h2, box caption,
paragraph, note, table) by reading the font of each line. The document's own
outline (PDF bookmarks) gives the chapter boundaries; fonts give the sections:

    Helvetica Bold  >= 15pt        -> section heading (h1)
    Helvetica Roman 11pt, blue     -> sub-section heading (h1 > h2)
    Helvetica Bold  ~10pt  "BOX"   -> boxed article caption
    Helvetica Bold  ~10pt  "TABLE"  -> table caption (rows rebuilt from positions)
    Garamond        >= 10pt        -> body text
    Garamond        < 10pt         -> footnote / source note
    Helvetica       < 9pt          -> chart labels and table cells (dropped, or
                                      rebuilt as rows under a TABLE caption)
"""

import re
import unicodedata
from collections import Counter

import pymupdf


HEADER_Y = 50.0          # running header sits above this
FOOTER_Y = 735.0         # running footer / page number sits below this
H2_COLOR = 0x4475A1

LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
}


# --------------------------------------------------------------------------
# Text repair
# --------------------------------------------------------------------------

def expand_ligatures(text):
    for lig, rep in LIGATURES.items():
        text = text.replace(lig, rep)
    return unicodedata.normalize("NFKC", text)


def build_ligature_fixes(raw_text):
    """
    This PDF's font drops the 'Th' ligature glyph: "The" is extracted as "Te",
    "This" as "Tis". To a keyword index those are different words. Learn the
    fixes from the document itself: a capitalised word T+x is a damaged word
    when it is not a word on its own and "Th"+x is attested elsewhere.
    """
    counts = Counter(re.findall(r"[A-Za-z]+", raw_text))
    fixes = {}
    for word in counts:
        if word[0] == "T" and not word.startswith("Th") and ("Th" + word[1:]) in counts:
            fixes[word] = "Th" + word[1:]
    return fixes


def repair_text(text, fixes):
    text = expand_ligatures(text)
    # Hyphenation at line ends: "invest- ment" -> "investment"
    text = re.sub(r"(?<=[a-z])-\s+(?=[a-z])", "", text)
    if fixes:
        pattern = r"\b(" + "|".join(map(re.escape, fixes)) + r")\b"
        text = re.sub(pattern, lambda m: fixes[m.group(1)], text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# Line classification
# --------------------------------------------------------------------------

def classify_line(spans, text):
    """Return one of: h1 h2 box table_caption figure_caption body note
    chart skip, judged from the first span's font."""
    s = spans[0]
    font, size, color = s["font"], s["size"], s["color"]
    helv = "Helvetica" in font or font == "Andes"
    bold = "Bold" in font

    if font.startswith("Helvetica") or "HelveticaLT" in font:
        if bold and size >= 15:
            return "h1"
        if not bold and abs(size - 11) < 0.6 and color == H2_COLOR:
            return "h2"
        if bold and 9.5 <= size <= 10.5:
            if text.startswith("BOX"):
                return "box"
            if text.startswith(("TABLE", "ANNEX")):
                return "table_caption" if text.startswith("TABLE") else "h1"
            if text.startswith("FIGURE"):
                return "figure_caption"
        if size < 9:
            return "chart"
        if "Cond" in font or size > 17:
            return "skip"           # chapter / summary page titles
        return "body"
    if font.startswith("AGaramond") or "Garamond" in font:
        if size >= 20:
            return "skip"           # chapter opener
        return "body" if size >= 10 else "note"
    if helv:
        return "skip"
    return "body" if size >= 10 else "note"


def line_text(line):
    return "".join(sp["text"] for sp in line["spans"]).strip()


# --------------------------------------------------------------------------
# Page extraction
# --------------------------------------------------------------------------

def _block_order(blocks, pw):
    """Reading order: wide blocks split the page into bands; inside a band the
    left column is read before the right column."""
    mid = pw / 2.0
    for b in blocks:
        x0, x1 = b["bbox"][0], b["bbox"][2]
        b["wide"] = (x1 - x0 > 0.58 * pw) or (x0 < mid - 40 and x1 > mid + 40)
        b["left"] = (x0 + x1) / 2.0 < mid
    ordered, band = [], []

    def flush():
        ordered.extend(sorted([b for b in band if b["left"]], key=lambda b: b["bbox"][1]))
        ordered.extend(sorted([b for b in band if not b["left"]], key=lambda b: b["bbox"][1]))
        band.clear()

    for b in sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0])):
        if b["wide"]:
            flush()
            ordered.append(b)
        else:
            band.append(b)
    flush()
    return ordered


def _leading_bold_run(block):
    """Executive-summary paragraphs open with a bold run-in title:
    'A Rising Challenge: Sovereign Debt ... EMDEs. Rising government ...'.
    Returns (title, rest) or None."""
    spans = [sp for ln in block["lines"] for sp in ln["spans"]]
    if not spans or "Bold" not in spans[0]["font"] or "Garamond" not in spans[0]["font"]:
        return None
    title, i = "", 0
    while i < len(spans) and "Bold" in spans[i]["font"] and "Garamond" in spans[i]["font"]:
        title += spans[i]["text"]
        i += 1
    rest = "".join(sp["text"] for sp in spans[i:])
    if not rest.strip() or len(title) > 160:
        return None
    return title.strip().rstrip(".").strip(), rest


def _rebuild_table(page, caption_y, stop_y=None):
    """Group small Helvetica spans below a TABLE caption into rows."""
    rows = {}
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            y = ln["bbox"][1]
            if y <= caption_y or y > FOOTER_Y or (stop_y and y >= stop_y):
                continue
            for sp in ln["spans"]:
                if sp["size"] < 9 and sp["text"].strip():
                    rows.setdefault(round(sp["bbox"][1] / 3), []).append(sp)
    out = []
    for key in sorted(rows):
        cells = sorted(rows[key], key=lambda s: s["bbox"][0])
        out.append(" | ".join(c["text"].strip() for c in cells))
    return out


def extract_page(page, page_num, in_annex=False):
    pw = page.rect.width
    raw = [b for b in page.get_text("dict")["blocks"] if b["type"] == 0]
    elements = []

    table_caption_y = None
    for b in _block_order(raw, pw):
        kept = []
        for ln in b["lines"]:
            text = line_text(ln)
            if not text or not ln["spans"]:
                continue
            y0, y1 = ln["bbox"][1], ln["bbox"][3]
            if y0 < HEADER_Y or y1 > FOOTER_Y:
                continue
            kept.append((classify_line(ln["spans"], text), text, ln, y0))
        if not kept:
            continue

        # Group consecutive lines of the same class into one element
        group_cls, group = None, []

        def flush_group(block=b):
            nonlocal group_cls, group
            if not group:
                return
            text = " ".join(t for t, _ in group)
            if group_cls in ("h1", "h2", "box", "table_caption"):
                elements.append({"page": page_num, "kind": group_cls, "text": text, "y": group[0][1]})
            elif group_cls in ("body", "note"):
                run = None
                if group_cls == "body" and group[0][1] is not None:
                    run = _leading_bold_run(block) if len(group) == len(kept) else None
                if run:
                    elements.append({"page": page_num, "kind": "h1", "text": run[0], "y": group[0][1], "runin": True})
                    text = run[1]
                elements.append({"page": page_num, "kind": group_cls, "text": text, "y": group[0][1]})
            group_cls, group = None, []

        for cls, text, ln, y0 in kept:
            if cls in ("chart", "skip", "figure_caption"):
                flush_group()
                continue
            if cls != group_cls:
                flush_group()
                group_cls = cls
            group.append((text, y0))
        flush_group()

    # Tables: TABLE captions on narrative pages, and every statistical-annex page
    for e in list(elements):
        if e["kind"] == "table_caption":
            table_caption_y = e["y"]
            rows = _rebuild_table(page, table_caption_y)
            elements.append({"page": page_num, "kind": "table", "text": e["text"], "rows": rows, "y": table_caption_y})
    if in_annex and not any(e["kind"] == "table" for e in elements):
        rows = _rebuild_table(page, HEADER_Y)
        if rows:
            elements.append({"page": page_num, "kind": "table", "text": "", "rows": rows, "y": HEADER_Y})
    return elements


# --------------------------------------------------------------------------
# Whole document
# --------------------------------------------------------------------------

def chapter_map(doc):
    """Chapter segments from the PDF bookmarks: [{key,title,start,end}]."""
    toc = [(t.replace("\r", " ").replace("\n", " ").strip(), p) for lvl, t, p in doc.get_toc() if lvl == 1]
    segments = []
    for title, page in toc:
        title = re.sub(r"\s+", " ", title)
        m = re.match(r"Ch\. (\d) (.*)", title)
        if m:
            segments.append({"key": f"ch{m.group(1)}", "no": int(m.group(1)), "title": m.group(2).strip(), "start": page})
        elif title == "Executive Summary":
            segments.append({"key": "exec", "no": 0, "title": "Executive Summary", "start": page})
        elif title == "Statistical Annex":
            segments.append({"key": "annex", "no": 5, "title": "Statistical Annex", "start": page})
    segments.sort(key=lambda s: s["start"])
    for i, s in enumerate(segments):
        s["end"] = segments[i + 1]["start"] - 1 if i + 1 < len(segments) else len(doc)
    return segments


def ingest_pdf(path):
    doc = pymupdf.open(path)
    segments = chapter_map(doc)
    first_content = segments[0]["start"]
    annex = next((s for s in segments if s["key"] == "annex"), None)

    fixes = build_ligature_fixes(" ".join(doc[i].get_text() for i in range(len(doc))))
    pages = []
    for i in range(first_content - 1, len(doc)):
        n = i + 1
        in_annex = bool(annex and annex["start"] <= n <= annex["end"])
        pages.append({"page": n, "elements": extract_page(doc[i], n, in_annex)})
    doc.close()

    for p in pages:
        for e in p["elements"]:
            if e["kind"] == "table":
                e["rows"] = [repair_text(r, fixes) for r in e["rows"]]
            e["text"] = repair_text(e["text"], fixes)
    return {"segments": segments, "pages": pages, "ligature_fixes": fixes}


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/GEP-Jun-2026.pdf"
    out = ingest_pdf(path)
    print("segments:", [(s["key"], s["start"], s["end"]) for s in out["segments"]])
    print("ligature fixes:", out["ligature_fixes"])
    kinds = Counter(e["kind"] for p in out["pages"] for e in p["elements"])
    print(kinds)
    for p in out["pages"]:
        if p["page"] in (19, 96):
            print(f"\n--- page {p['page']} ---")
            for e in p["elements"]:
                print(f"[{e['kind']}] {e['text'][:110]}")
