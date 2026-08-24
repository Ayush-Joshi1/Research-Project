"""Simple embedder using sentence-transformers/all-MiniLM-L6-v2.

Provides `encode_texts(texts: List[str]) -> np.ndarray` which returns L2-normalized
embeddings so cosine similarity can be computed as inner product in FAISS IndexFlatIP.

Normalization rationale:
- FAISS IndexFlatIP performs inner-product search. If vectors are L2-normalized,
  inner product equals cosine similarity. Normalizing ensures scores are in [-1,1]
  and comparable across corpus and queries.
"""
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


# Load model once at import time to reuse across calls.
_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def encode_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Encode a list of texts into L2-normalized numpy embeddings.

    Args:
        texts: list of input strings.
        batch_size: batch size for model encoding.

    Returns:
        np.ndarray of shape (len(texts), dim) with dtype float32.
    """
    # Use the model to get embeddings (returns numpy array)
    embeddings = _MODEL.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    # ensure float32
    embeddings = embeddings.astype("float32")
    # L2-normalize each vector so that inner product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # avoid division by zero
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    return embeddings
