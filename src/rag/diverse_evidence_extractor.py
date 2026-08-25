"""Diversity-aware local evidence extraction."""
from typing import Dict, List

import numpy as np

from src.rag.evidence_extractor import split_sentences
from src.retrieval.embedder import encode_texts


def extract_diverse_evidence(
    claim: str,
    documents: List[Dict],
    max_sentences_per_document: int = 2,
    max_total_sentences: int = 6,
) -> List[Dict]:
    """Select claim-similar sentences while encouraging document diversity."""
    if max_sentences_per_document <= 0 or max_total_sentences <= 0:
        return []

    candidates = []
    for document_rank, document in enumerate(documents, start=1):
        for sentence in split_sentences(document.get("text", "")):
            candidates.append({
                "corpus_id": document["corpus_id"],
                "sentence": sentence,
                "document_rank": document_rank,
            })

    if not candidates:
        return []

    # Shared normalized embeddings make inner products cosine similarities.
    embeddings = encode_texts([claim] + [item["sentence"] for item in candidates])
    similarities = embeddings[1:] @ embeddings[0]
    for item, similarity in zip(candidates, similarities):
        item["similarity"] = float(similarity)

    selected = []
    represented_documents = set()
    while candidates and len(selected) < max_total_sentences:
        available = [
            item for item in candidates
            if sum(selected_item["document_rank"] == item["document_rank"] for selected_item in selected)
            < max_sentences_per_document
        ]
        if not available:
            break

        # Diversity applies only after the first sentence, which is pure similarity.
        for item in available:
            diversity = 0 if item["document_rank"] in represented_documents else 1
            item["selection_score"] = (
                item["similarity"] if not selected
                else 0.7 * item["similarity"] + 0.3 * diversity
            )
        best = max(available, key=lambda item: item["selection_score"])
        selected.append(best)
        represented_documents.add(best["document_rank"])
        candidates.remove(best)

    selected.sort(key=lambda item: item["selection_score"], reverse=True)
    return [
        {
            "corpus_id": item["corpus_id"],
            "sentence": item["sentence"],
            "similarity": item["similarity"],
            "selection_score": item["selection_score"],
            "document_rank": item["document_rank"],
        }
        for item in selected
    ]
