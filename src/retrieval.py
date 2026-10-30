"""
Filtered hybrid retrieval with parent expansion.

1. Metadata filter from the query plan (region / chapter; tables only on request).
2. Dense + BM25 over child chunks, merged with Reciprocal Rank Fusion (k=60).
3. Cross-encoder rerank of the fused shortlist (heading path + passage).
4. Search small, answer from the parent: children collapse to their parent
   section, best child decides the rank, and parents are added until the
   context budget is spent.
"""

import numpy as np
from sentence_transformers import CrossEncoder

from src.chunking import approx_tokens, embed_text
from src.embeddings import tokenize

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder = None


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)
    return _cross_encoder


def candidate_mask(index, plan):
    """Boolean mask over children allowed by the plan."""
    mask = np.ones(len(index.children), dtype=bool)
    for i, c in enumerate(index.children):
        if c["type"] == "table" and not plan.get("tables"):
            mask[i] = False
        elif plan.get("chapter") and c["chapter_key"] != plan["chapter"]:
            mask[i] = False
        elif plan.get("region") and c["region"] != plan["region"]:
            mask[i] = False
    return mask


def _ranked(scores, mask, k):
    s = np.where(mask, scores, -np.inf)
    order = np.argsort(s)[::-1][:k]
    return [int(i) for i in order if np.isfinite(s[i])]


def hybrid_search(index, question, mask, pool=30, rrf_k=60):
    dense = _ranked(index.embeddings @ index.encode_query(question), mask, pool)
    sparse = _ranked(index.bm25.get_scores(tokenize(question)), mask, pool)
    fused = {}
    for ranking in (dense, sparse):
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(fused, key=fused.get, reverse=True)


HEADING_STOPWORDS = {"the", "and", "of", "in", "to", "for", "a", "an", "on", "ch", "box", "annex"}
HEADING_BONUS = 1.5


def heading_bonus(child, question_tokens):
    """Section-aware nudge: a question that says "outlook" or "risks" is asking for
    the section called Outlook / Risks. Counts the share of the sub-section title's
    words (h2, or the box title) that the question repeats."""
    title = child.get("h2") or child.get("box") or ""
    words = [w for w in tokenize(title) if w not in HEADING_STOPWORDS and not w.isdigit()]
    if not words:
        return 0.0
    return HEADING_BONUS * sum(w in question_tokens for w in words) / len(words)


def rerank(index, question, candidates, top_n=20):
    if not candidates:
        return []
    pairs = [(question, embed_text(index.children[i])) for i in candidates]
    scores = get_cross_encoder().predict(pairs)
    qtok = set(tokenize(question))
    boosted = [(i, float(s) + heading_bonus(index.children[i], qtok)) for i, s in zip(candidates, scores)]
    return sorted(boosted, key=lambda t: t[1], reverse=True)[:top_n]


def retrieve(index, question, plan, max_parents=3, context_tokens=2600, use_reranker=True):
    """Return parent sections ordered by relevance, each with the child that matched."""
    mask = candidate_mask(index, plan)
    if not mask.any():                     # filter matched nothing: search the whole text
        mask = candidate_mask(index, {"tables": plan.get("tables")})
        plan = {**plan, "filter_dropped": True}
    fused = hybrid_search(index, question, mask)
    if use_reranker:
        ranked = rerank(index, question, fused)
    else:
        ranked = [(i, 1.0 / (r + 1)) for r, i in enumerate(fused[:20])]

    hits, seen, used = [], set(), 0
    for child_idx, score in ranked:
        child = index.children[child_idx]
        pid = child["parent_id"]
        if pid in seen:
            continue
        parent = index.parents[pid]
        cost = approx_tokens(parent["text"])
        if hits and used + cost > context_tokens:
            continue
        seen.add(pid)
        used += cost
        hits.append({"parent": parent, "child": child, "score": score, "plan": plan})
        if len(hits) >= max_parents:
            break
    return hits
