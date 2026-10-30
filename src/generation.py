"""
Grounded answer generation with Google Gemini (free tier).

Each passage is labelled with where it lives (chapter > section, pages) so the
model can see it is reading, say, the South Asia outlook and not the East Asia one.
Without an API key it falls back to a generic extractive answer: the sentences of
the top passage that overlap most with the question (no question-specific rules).
"""

import json
import os
import re
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.chunking import split_sentences
from src.embeddings import tokenize

REFUSAL = "I could not find this in the document."

SYSTEM_PROMPT = f"""You answer questions about the World Bank report "Global Economic Prospects, June 2026" using ONLY the numbered passages.
- Use only facts stated in the passages. Never add outside knowledge.
- Each passage header shows its chapter, section and pages. Answer from the passage whose location matches what the question asks about; do not mix figures from different regions, chapters or sections.
- Quote numbers, names and dates exactly as written, and keep their period (for example "in 2026", "fiscal year 2026/27").
- Cite passages with square brackets only, like [1] or [2][3].
- If the passages do not contain the answer, reply exactly: {REFUSAL}
- Be brief: two to five sentences, or a short list for "what are the main..." questions."""


def build_context(hits):
    blocks = []
    for i, h in enumerate(hits, start=1):
        p = h["parent"]
        pages = f"p. {p['page_start']}" if p["page_start"] == p["page_end"] else f"pp. {p['page_start']}-{p['page_end']}"
        blocks.append(f"[{i}] {p['path']} ({pages})\n{p['text']}")
    return "\n\n".join(blocks)


def _client():
    from google import genai
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None
    return genai.Client(api_key=key)


def model_names():
    names = [os.environ.get("GEMINI_MODEL"), "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    seen, out = set(), []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _call(prompt, system=None, json_mode=False, retries=3):
    client = _client()
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not set")
    config = {"temperature": 0.0}
    if system:
        config["system_instruction"] = system
    if json_mode:
        config["response_mime_type"] = "application/json"
    last = None
    for model in model_names():
        for attempt in range(retries):
            try:
                return client.models.generate_content(model=model, contents=prompt, config=config).text.strip()
            except Exception as e:
                last = e
                msg = str(e)
                if "404" in msg or "NOT_FOUND" in msg:
                    break                      # model name not available: try the next one
                time.sleep(2 ** attempt * 2)   # 429 / transient: back off
    raise RuntimeError(f"Gemini call failed: {last}")


def generate_json(prompt):
    return _call(prompt, json_mode=True)


def extractive_answer(question, hits):
    if not hits:
        return REFUSAL
    q = set(tokenize(question)) - {"the", "a", "of", "in", "for", "is", "what", "does", "this", "and", "to", "are"}
    sents = split_sentences(hits[0]["parent"]["text"])
    scored = sorted(sents, key=lambda s: len(q & set(tokenize(s))), reverse=True)[:2]
    return " ".join(scored) + " [1]"


def answer_question(question, hits):
    if not hits:
        return REFUSAL
    if _client() is None:
        return extractive_answer(question, hits)
    prompt = f"Passages:\n\n{build_context(hits)}\n\nQuestion: {question}"
    return _call(prompt, system=SYSTEM_PROMPT)
