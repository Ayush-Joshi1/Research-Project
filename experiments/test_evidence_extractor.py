"""Run a local sanity check for evidence extraction."""
import json
import os
import sys

from sentence_transformers import CrossEncoder

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.rag.evidence_extractor import extract_evidence
from src.retrieval import Retriever


DEV_PATH = os.path.join(ROOT, "data", "processed", "dev_queries.jsonl")
QUERY_IDS = {"118", "1019", "1320", "1370", "1185"}
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def main():
    queries = [query for query in load_jsonl(DEV_PATH) if query["query_id"] in QUERY_IDS]
    retriever = Retriever(rebuild=False)
    reranker = CrossEncoder(RERANKER_NAME)

    for query in queries:
        claim = query["claim_text"]
        retrieved = retriever.retrieve(claim, top_k=10)
        pairs = [(claim, document["text"]) for document in retrieved]
        scores = reranker.predict(pairs)
        reranked = [
            {**document, "reranker_score": float(score), "original_faiss_rank": rank}
            for rank, (document, score) in enumerate(zip(retrieved, scores), start=1)
        ]
        reranked.sort(key=lambda document: document["reranker_score"], reverse=True)
        evidence = extract_evidence(claim, reranked[:5])

        print(f"QUERY ID: {query['query_id']}")
        print(f"CLAIM: {claim}")
        for item in evidence:
            print(f"Document ID: {item['corpus_id']}")
            print(f"Document rank: {item['document_rank']}")
            print(f"Similarity: {item['similarity']:.4f}")
            print(f"Sentence: {item['sentence']}")
        print(f"Extracted evidence sentences: {len(evidence)}")
        print()


if __name__ == "__main__":
    main()
