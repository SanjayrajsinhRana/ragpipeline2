"""
Index over the child chunks:
1. Dense vectors from BAAI/bge-small-en-v1.5 (local, free) of "heading path + passage".
2. BM25 over the same contextualised text, so a passage that says "SAR" is still
   found by a query that says "South Asia" (the heading path spells it out).
3. Parent sections, looked up by id once a child matches.
Everything is cached to one pickle; delete it (or pass --reindex) to rebuild.
"""

import os
import pickle
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.chunking import embed_text

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
INDEX_VERSION = 2


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_embedder(name=EMBEDDING_MODEL_NAME):
    return SentenceTransformer(name)


class DocumentIndex:
    def __init__(self, parents, children, embeddings, model_name=EMBEDDING_MODEL_NAME):
        self.parents = parents
        self.children = children
        self.embeddings = embeddings
        self.model_name = model_name
        self.embedder = load_embedder(model_name)
        self.tokenized = [tokenize(embed_text(c)) for c in children]
        self.bm25 = BM25Okapi(self.tokenized)

    @classmethod
    def build(cls, parents, children, model_name=EMBEDDING_MODEL_NAME):
        embedder = load_embedder(model_name)
        print(f"Embedding {len(children)} child chunks with {model_name}...")
        vecs = embedder.encode([embed_text(c) for c in children], batch_size=32,
                               show_progress_bar=False, normalize_embeddings=True)
        return cls(parents, children, vecs, model_name)

    def encode_query(self, question):
        return self.embedder.encode([QUERY_PREFIX + question], normalize_embeddings=True)[0]

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"version": INDEX_VERSION, "parents": self.parents, "children": self.children,
                         "embeddings": self.embeddings, "model_name": self.model_name}, f)
        print(f"Index saved to {path}")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("version") != INDEX_VERSION:
            raise ValueError("stale index version")
        return cls(data["parents"], data["children"], data["embeddings"], data["model_name"])
