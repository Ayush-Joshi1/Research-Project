"""Run the improved prompt on the first five development claims."""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.rag.improved_generator import generate_verification_improved
from src.retrieval import Retriever

DEV_PATH = os.path.join(ROOT, "data", "processed", "dev_queries.jsonl")
BASELINE_PATH = os.path.join(ROOT, "results", "rag_results_clean.jsonl")
OUTPUT_PATH = os.path.join(ROOT, "results", "improved_prompt_sanity.jsonl")
K_VALUES = (1, 3, 5, 10)


def load_first_queries(count=5):
    with open(DEV_PATH, "r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()][:count]


def load_baseline():
    with open(BASELINE_PATH, "r", encoding="utf-8") as file_handle:
        return {
            (row["query_id"], int(row["k"])): row
            for line in file_handle
            if line.strip()
            for row in [json.loads(line)]
        }


def main():
    queries = load_first_queries()
    baseline = load_baseline()
    retriever = Retriever(rebuild=False)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    records = []

    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        for query in queries:
            relevant_ids = set(query.get("relevant_corpus_ids", []))
            for k_value in K_VALUES:
                documents = retriever.retrieve(query["claim_text"], top_k=k_value)
                result = generate_verification_improved(query["claim_text"], documents)
                retrieved_ids = [document["corpus_id"] for document in documents]
                record = {
                    "query_id": query["query_id"],
                    "k": k_value,
                    "ground_truth_label": query.get("ground_truth_label"),
                    "prediction": result["verdict"],
                    "retrieval_hit": any(doc_id in relevant_ids for doc_id in retrieved_ids),
                    "explanation": result["explanation"],
                    "retrieved_corpus_ids": retrieved_ids,
                    "baseline_prediction": baseline[(query["query_id"], k_value)]["predicted_verdict"],
                }
                output_file.write(json.dumps(record) + "\n")
                output_file.flush()
                records.append(record)
                print(json.dumps(record))

    print(f"API calls made: {len(records)}")
    print(f"Output records: {len(records)}")


if __name__ == "__main__":
    main()
