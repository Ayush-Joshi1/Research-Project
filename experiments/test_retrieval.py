"""Run full development-set retrieval at K=1, 3, 5, and 10."""
import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.retrieval import Retriever


DEV_PATH = os.path.join(ROOT, "data", "processed", "dev_queries.jsonl")
RESULTS_DIR = os.path.join(ROOT, "results")
RAW_PATH = os.path.join(RESULTS_DIR, "retrieval_results.jsonl")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "retrieval_summary.csv")
K_VALUES = (1, 3, 5, 10)


def load_dev_queries():
    with open(DEV_PATH, "r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def main():
    queries = load_dev_queries()
    retriever = Retriever(rebuild=False)
    raw_rows = []
    summary_rows = []

    for query in queries:
        query_id = query["query_id"]
        claim_text = query["claim_text"]
        label = query.get("ground_truth_label")
        relevant_ids = set(query.get("relevant_corpus_ids", []))
        top_results = retriever.retrieve(claim_text, top_k=10)
        hits = {}

        for k_value in K_VALUES:
            ranked_results = top_results[:k_value]
            hits[k_value] = int(
                any(result["corpus_id"] in relevant_ids for result in ranked_results)
            )
            for rank, result in enumerate(ranked_results, start=1):
                raw_rows.append(
                    {
                        "query_id": query_id,
                        "K": k_value,
                        "claim_text": claim_text,
                        "retrieved_rank": rank,
                        "corpus_id": result["corpus_id"],
                        "similarity_score": result["similarity_score"],
                        "is_relevant": int(result["corpus_id"] in relevant_ids),
                        "ground_truth_label": label,
                    }
                )

        summary_rows.append(
            {
                "query_id": query_id,
                "ground_truth_label": label,
                "hit@1": hits[1],
                "hit@3": hits[3],
                "hit@5": hits[5],
                "hit@10": hits[10],
            }
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RAW_PATH, "w", encoding="utf-8") as file_handle:
        for row in raw_rows:
            file_handle.write(json.dumps(row) + "\n")

    summary_fields = [
        "query_id",
        "ground_truth_label",
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
    ]
    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Queries evaluated: {len(queries)}")
    print(f"Raw rows written: {len(raw_rows)}")
    print(f"Summary rows written: {len(summary_rows)}")
    print(f"Index documents: {retriever.index.ntotal}")
    print(f"Index dimension: {retriever.index.d}")
    for label in (None, "SUPPORT", "CONTRADICT"):
        label_rows = [row for row in summary_rows if row["ground_truth_label"] == label]
        if not label_rows:
            continue
        name = label or "UNLABELED"
        aggregate = {
            k: sum(row[f"hit@{k}"] for row in label_rows) / len(label_rows)
            for k in K_VALUES
        }
        print(f"{name} ({len(label_rows)}): {aggregate}")
    overall = {
        k: sum(row[f"hit@{k}"] for row in summary_rows) / len(summary_rows)
        for k in K_VALUES
    }
    print(f"Overall: {overall}")
    print(f"Raw output: {RAW_PATH}")
    print(f"Summary output: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
