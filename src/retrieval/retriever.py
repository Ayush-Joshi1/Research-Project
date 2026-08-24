"""Retriever: builds FAISS IndexFlatIP over corpus embeddings and performs queries.

Behavior:
- Loads corpus from `data/processed/corpus.jsonl`.
- Builds embeddings via the embedder and L2-normalizes them.
- Uses FAISS IndexFlatIP so inner product on normalized vectors equals cosine similarity.
- Saves index and id mapping to `data/index/` to avoid recomputation.
"""
import json
import os
from typing import List, Dict, Tuple

import faiss
import numpy as np

from .embedder import encode_texts


DATA_INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "index")
os.makedirs(DATA_INDEX_DIR, exist_ok=True)
INDEX_PATH = os.path.join(DATA_INDEX_DIR, "faiss_index.ivf")
IDMAP_PATH = os.path.join(DATA_INDEX_DIR, "id_map.json")
CORPUS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "processed", "corpus.jsonl")


class Retriever:
    def __init__(self, rebuild: bool = False):
        self.corpus: List[Dict] = []
        self.id_map: List[str] = []  # position -> corpus_id
        self.index = None
        self.dim = None
        if not rebuild and os.path.exists(INDEX_PATH) and os.path.exists(IDMAP_PATH):
            try:
                self.index = faiss.read_index(INDEX_PATH)
                with open(IDMAP_PATH, "r", encoding="utf-8") as fh:
                    self.id_map = json.load(fh)
                self.corpus = self._load_corpus()
                self.dim = self.index.d
            except Exception:
                # fallback to rebuild
                self._build_index()
        else:
            self._build_index()

    def _load_corpus(self) -> List[Dict]:
        docs = []
        if not os.path.exists(CORPUS_PATH):
            raise FileNotFoundError(f"Corpus file not found: {CORPUS_PATH}")
        with open(CORPUS_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                docs.append(json.loads(line))
        return docs

    def _build_index(self):
        # Load corpus and compute embeddings
        self.corpus = self._load_corpus()
        texts = [d.get("text", "") for d in self.corpus]
        titles = [d.get("title", "") for d in self.corpus]
        corpus_ids = [d.get("corpus_id") for d in self.corpus]

        # Compute embeddings (normalized inside encode_texts)
        embeddings = encode_texts(texts)
        n, dim = embeddings.shape
        self.dim = dim

        # Build FAISS IndexFlatIP for inner-product search on normalized vectors
        # IndexFlatIP returns exact nearest neighbors via inner product.
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Save index and id map
        faiss.write_index(index, INDEX_PATH)
        with open(IDMAP_PATH, "w", encoding="utf-8") as fh:
            json.dump(corpus_ids, fh)

        self.index = index
        self.id_map = corpus_ids

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top_k docs for the query.

        Returns list of dicts with keys: corpus_id, title, text, similarity_score
        """
        # Encode and normalize query using same embedder
        q_emb = encode_texts([query])  # shape (1, dim)
        if self.index is None:
            raise RuntimeError("Index not initialized")
        D, I = self.index.search(q_emb, top_k)
        scores = D[0].tolist()
        idxs = I[0].tolist()
        results = []
        for score, idx in zip(scores, idxs):
            if idx < 0 or idx >= len(self.id_map):
                continue
            cid = self.id_map[idx]
            doc = next((d for d in self.corpus if d.get("corpus_id") == cid), None)
            if doc is None:
                continue
            results.append({
                "corpus_id": cid,
                "title": doc.get("title"),
                "text": doc.get("text"),
                "similarity_score": float(score),
            })
        return results
