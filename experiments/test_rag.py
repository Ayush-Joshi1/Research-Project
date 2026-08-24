"""Run generation for the first development claim at four retrieval depths."""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.rag import generate_verification
from src.retrieval import Retriever

DEV_PATH = os.path.join(ROOT, "data", "processed", "dev_queries.jsonl")
K_VALUES = (1, 3, 5, 10)


def load_first_query():
    with open(DEV_PATH, "r", encoding="utf-8") as file_handle:
        return json.loads(next(line for line in file_handle if line.strip()))


def main():
    query = load_first_query()
    retriever = Retriever(rebuild=False)
    print("GEMINI_API_KEY detected: yes")
    for k_value in K_VALUES:
        documents = retriever.retrieve(query["claim_text"], top_k=k_value)
        result = generate_verification(query["claim_text"], documents)
        document_ids = [document["corpus_id"] for document in documents]
        print(f"\nquery_id: {query['query_id']}")
        print(f"K: {k_value}")
        print(f"claim: {query['claim_text']}")
        print(f"retrieved_document_ids: {document_ids}")
        print(f"model_verdict: {result['verdict']}")
        print(f"model_explanation: {result['explanation']}")
        print(f"ground_truth_label: {query.get('ground_truth_label')}")


if __name__ == "__main__":
    main()
