"""
Turns the typed element stream from ingest.py into a section tree.

Every section carries the metadata that later makes retrieval location-aware:
chapter, h1 section, h2 sub-section, boxed article, region, and a heading path
such as "Ch. 2 Regional Outlooks > South Asia > Outlook".
"""

import re

REGIONS = {
    "East Asia and Pacific": ["east asia and pacific", "eap"],
    "Europe and Central Asia": ["europe and central asia", "eca"],
    "Latin America and the Caribbean": ["latin america and the caribbean", "latin america", "lac"],
    "Middle East, North Africa, Afghanistan, and Pakistan": [
        "middle east, north africa, afghanistan, and pakistan", "middle east and north africa",
        "mena", "menap", "mnaap", "mna",
    ],
    "South Asia": ["south asia", "sar"],
    "Sub-Saharan Africa": ["sub-saharan africa", "ssa"],
}


def _norm(s):
    return re.sub(r"[^a-z]+", " ", s.lower()).strip()


def region_of_heading(h1):
    """Map a chapter-2 h1 heading to one of the six regions, or None."""
    n = _norm(h1)
    for region in REGIONS:
        if _norm(region) == n:
            return region
    return None


def chapter_for_page(segments, page):
    for s in segments:
        if s["start"] <= page <= s["end"]:
            return s
    return segments[-1]


def chapter_label(seg):
    if seg["key"].startswith("ch"):
        return f"Ch. {seg['no']} {seg['title']}"
    return seg["title"]


def _merge_heading_lines(elements):
    """Headings wrap over several lines ('Latin America and' / 'the Caribbean'):
    merge consecutive same-kind headings on the same page into one."""
    merged = []
    for e in elements:
        prev = merged[-1] if merged else None
        if (prev and e["kind"] in ("h1", "h2") and prev["kind"] == e["kind"]
                and prev["page"] == e["page"] and not e.get("runin") and not prev.get("runin")
                and 0 < e["y"] - prev["y"] < 30):
            prev["text"] = f"{prev['text']} {e['text']}"
            prev["y"] = e["y"]
            continue
        merged.append(dict(e))
    return merged


def build_sections(ingested):
    """Walk elements in order and emit sections: one per (chapter, h1, h2, box)."""
    segments = ingested["segments"]
    sections = []
    state = {"h1": None, "h2": None, "box": None, "box_page": None}
    current = None
    last_chapter = None

    def open_section(seg, page):
        nonlocal current
        region = region_of_heading(state["h1"]) if seg["key"] == "ch2" and state["h1"] else None
        current = {
            "id": len(sections),
            "chapter_key": seg["key"],
            "chapter_no": seg["no"],
            "chapter": chapter_label(seg),
            "h1": state["h1"],
            "h2": state["h2"],
            "box": state["box"],
            "region": region,
            "page_start": page,
            "page_end": page,
            "items": [],            # (kind, text, page)
        }
        sections.append(current)

    for page in ingested["pages"]:
        pn = page["page"]
        seg = chapter_for_page(segments, pn)
        if seg["key"] != last_chapter:
            state.update(h1=None, h2=None, box=None, box_page=None)
            current = None
            last_chapter = seg["key"]
        # A boxed article spans whole pages: it ends on the first page with no box caption
        has_box_caption = any(e["kind"] == "box" for e in page["elements"])
        if state["box"] and not has_box_caption and state["box_page"] != pn:
            state["box"] = None
            current = None

        for e in _merge_heading_lines(page["elements"]):
            kind = e["kind"]
            if kind == "box":
                title = re.sub(r"\s*\(continued\)\s*$", "", e["text"])
                if title != state["box"]:
                    state.update(box=title, box_page=pn, h2=None)
                    current = None
                continue
            if kind == "h1":
                state.update(h1=e["text"], h2=None, box=None)
                current = None
                continue
            if kind == "h2":
                state["h2"] = e["text"]
                current = None
                continue
            if kind in ("body", "note", "table"):
                if current is None:
                    open_section(seg, pn)
                current["page_end"] = pn
                if kind == "table":
                    current["items"].append(("table", e, pn))
                else:
                    current["items"].append((kind, e["text"], pn))
    return [s for s in sections if s["items"]]


def heading_path(sec):
    parts = [sec["chapter"]]
    # A boxed article is its own unit: it does not inherit the section it sits in
    keys = ("box", "h2") if sec.get("box") else ("h1", "h2")
    for key in keys:
        if sec.get(key):
            parts.append(sec[key])
    return " > ".join(parts)


if __name__ == "__main__":
    from src.ingest import ingest_pdf
    doc = ingest_pdf("data/GEP-Jun-2026.pdf")
    secs = build_sections(doc)
    print(len(secs), "sections")
    for s in secs:
        n = sum(len(i[1].split()) if isinstance(i[1], str) else 0 for i in s["items"])
        print(f"p{s['page_start']}-{s['page_end']} [{s['region'] or '-'}] {heading_path(s)} ({n}w)")
