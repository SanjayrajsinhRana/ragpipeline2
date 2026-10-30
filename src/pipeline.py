"""
End-to-end orchestration: ingest -> sections -> parent/child chunks -> index,
then per question: plan (filters) -> filtered hybrid retrieval + rerank -> grounded answer.
"""

import os
import time

from src.chunking import build_chunks
from src.embeddings import DocumentIndex
from src.generation import answer_question, build_context, generate_json, _client
from src.ingest import ingest_pdf
from src.query import make_plan, rule_plan
from src.retrieval import retrieve
from src.structure import build_sections


class GEPPipeline:
    def __init__(self, pdf_path="data/GEP-Jun-2026.pdf", index_cache="data/index_cache.pkl",
                 use_reranker=True, use_llm_routing=True, reindex=False):
        self.pdf_path = pdf_path
        self.index_cache = index_cache
        self.use_reranker = use_reranker
        self.use_llm_routing = use_llm_routing
        self.index = self._load_or_build(reindex)

    def _load_or_build(self, reindex):
        if os.path.exists(self.index_cache) and not reindex:
            try:
                print(f"[Pipeline] Loading index from {self.index_cache}")
                return DocumentIndex.load(self.index_cache)
            except ValueError:
                print("[Pipeline] Cached index is stale, rebuilding")
        print(f"[Pipeline] Ingesting {self.pdf_path}")
        sections = build_sections(ingest_pdf(self.pdf_path))
        parents, children = build_chunks(sections)
        print(f"[Pipeline] {len(sections)} sections -> {len(parents)} parents, {len(children)} children")
        index = DocumentIndex.build(parents, children)
        index.save(self.index_cache)
        return index

    def plan(self, question):
        if self.use_llm_routing and _client() is not None:
            return make_plan(question, generate_json)
        return rule_plan(question)

    def retrieve(self, question):
        plan = self.plan(question)
        return plan, retrieve(self.index, question, plan, use_reranker=self.use_reranker)

    def query(self, question):
        t0 = time.time()
        plan, hits = self.retrieve(question)
        t1 = time.time()
        answer = answer_question(question, hits)
        t2 = time.time()
        return {"question": question, "plan": plan, "hits": hits, "answer": answer,
                "context": build_context(hits), "retrieval_time": t1 - t0, "generation_time": t2 - t1}
