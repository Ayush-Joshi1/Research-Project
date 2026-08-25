"""Extract claim-relevant evidence sentences from retrieved documents."""
import re
from typing import Dict, List

import numpy as np

from src.retrieval.embedder import encode_texts


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])(?=[A-Z])")


def split_sentences(text: str) -> List[str]:
    """Split document text at common sentence-ending punctuation."""
    return [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(text.strip()) if sentence.strip()]


def extract_evidence(
    claim: str,
    documents: List[Dict],
    max_sentences_per_document: int = 2,
    max_total_sentences: int = 6,
) -> List[Dict]:
    """Return the highest-scoring evidence sentences for a claim.

    Embeddings are normalized by the shared project embedder. Their inner product
    is therefore cosine similarity between the claim and each sentence.
    """
    if max_sentences_per_document <= 0 or max_total_sentences <= 0:
        return []

    candidates = []
    for document_rank, document in enumerate(documents, start=1):
        # Keep the original text untouched; only create separate sentence strings.
        for sentence in split_sentences(document.get("text", "")):
            candidates.append({
                "corpus_id": document["corpus_id"],
                "sentence": sentence,
                "document_rank": document_rank,
            })

    if not candidates:
        return []

    # Encode the claim and all candidate sentences with the existing model.
    embeddings = encode_texts([claim] + [item["sentence"] for item in candidates])
    claim_embedding = embeddings[0]
    sentence_embeddings = embeddings[1:]
    # Normalized vectors make inner product equal cosine similarity.
    similarities = sentence_embeddings @ claim_embedding
    for item, similarity in zip(candidates, similarities):
        item["similarity"] = float(similarity)

    # Rank within each document, then keep the configured number from each one.
    selected = []
    for document_rank in range(1, len(documents) + 1):
        document_candidates = [item for item in candidates if item["document_rank"] == document_rank]
        document_candidates.sort(key=lambda item: item["similarity"], reverse=True)
        selected.extend(document_candidates[:max_sentences_per_document])

    # Combine document selections globally and return the strongest evidence first.
    selected.sort(key=lambda item: item["similarity"], reverse=True)
    return selected[:max_total_sentences]
