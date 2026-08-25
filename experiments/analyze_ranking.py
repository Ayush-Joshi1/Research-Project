"""Analyze saved retrieval ranks without rerunning retrieval or generation."""
import csv
import json
import os
import statistics


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEV_PATH = os.path.join(ROOT, "data", "processed", "dev_queries.jsonl")
RETRIEVAL_PATH = os.path.join(ROOT, "results", "retrieval_results.jsonl")
RAG_PATH = os.path.join(ROOT, "results", "rag_results_clean.jsonl")
JSON_PATH = os.path.join(ROOT, "results", "ranking_analysis.json")
CSV_PATH = os.path.join(ROOT, "results", "ranking_analysis.csv")
K_VALUES = (1, 3, 5, 10)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def main():
    dev = {row["query_id"]: row for row in load_jsonl(DEV_PATH)}
    retrieval_rows = load_jsonl(RETRIEVAL_PATH)
    rag_rows = load_jsonl(RAG_PATH)

    rankings = {}
    for row in retrieval_rows:
        if int(row["K"]) == 10:
            rankings.setdefault(row["query_id"], []).append(row)
    predictions = {
        (row["query_id"], int(row["k"])): row["predicted_verdict"]
        for row in rag_rows
    }

    query_rows = []
    for query_id, query in dev.items():
        relevant_ids = set(query.get("relevant_corpus_ids", []))
        ranked = sorted(rankings[query_id], key=lambda row: row["retrieved_rank"])
        relevant_ranks = [
            int(row["retrieved_rank"])
            for row in ranked
            if row["corpus_id"] in relevant_ids
        ]
        first_rank = min(relevant_ranks) if relevant_ranks else None
        query_rows.append(
            {
                "query_id": query_id,
                "first_relevant_rank": first_rank,
                "predictions": {
                    str(k): predictions[(query_id, k)] for k in K_VALUES
                },
                "ground_truth": query["ground_truth_label"],
            }
        )

    rank_distribution = {
        str(rank): sum(row["first_relevant_rank"] == rank for row in query_rows)
        for rank in range(1, 11)
    }
    rank_distribution["none"] = sum(
        row["first_relevant_rank"] is None for row in query_rows
    )
    recall = {
        str(k): sum(
            row["first_relevant_rank"] is not None
            and row["first_relevant_rank"] <= k
            for row in query_rows
        ) / len(query_rows)
        for k in K_VALUES
    }
    observed_ranks = [
        row["first_relevant_rank"] for row in query_rows
        if row["first_relevant_rank"] is not None
    ]

    report = {
        "records_analyzed": len(retrieval_rows) + len(rag_rows),
        "queries_analyzed": len(query_rows),
        "retrieval_records_used_for_rank": sum(len(rows) for rows in rankings.values()),
        "api_calls_made": 0,
        "retrieval_calls_made": 0,
        "rank_distribution": rank_distribution,
        "recall_at_k": recall,
        "first_relevant_rank_mean": statistics.mean(observed_ranks),
        "first_relevant_rank_median": statistics.median(observed_ranks),
        "queries_with_no_relevant_document_in_top_10": [
            row["query_id"] for row in query_rows if row["first_relevant_rank"] is None
        ],
        "queries_first_relevant_rank_1": [
            row["query_id"] for row in query_rows if row["first_relevant_rank"] == 1
        ],
        "queries_first_relevant_rank_2_to_5": [
            row["query_id"] for row in query_rows
            if row["first_relevant_rank"] is not None
            and 2 <= row["first_relevant_rank"] <= 5
        ],
        "queries_first_relevant_rank_6_to_10": [
            row["query_id"] for row in query_rows
            if row["first_relevant_rank"] is not None
            and 6 <= row["first_relevant_rank"] <= 10
        ],
        "query_comparisons": query_rows,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2)

    fields = [
        "query_id", "first_relevant_rank", "K=1_prediction", "K=3_prediction",
        "K=5_prediction", "K=10_prediction", "ground_truth",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        writer.writeheader()
        for row in query_rows:
            writer.writerow(
                {
                    "query_id": row["query_id"],
                    "first_relevant_rank": row["first_relevant_rank"],
                    "K=1_prediction": row["predictions"]["1"],
                    "K=3_prediction": row["predictions"]["3"],
                    "K=5_prediction": row["predictions"]["5"],
                    "K=10_prediction": row["predictions"]["10"],
                    "ground_truth": row["ground_truth"],
                }
            )

    print(json.dumps({
        "queries_analyzed": len(query_rows),
        "rank_distribution": rank_distribution,
        "recall_at_k": recall,
        "mean_first_relevant_rank": report["first_relevant_rank_mean"],
        "median_first_relevant_rank": report["first_relevant_rank_median"],
        "no_relevant_in_top_10": len(report["queries_with_no_relevant_document_in_top_10"]),
        "api_calls_made": 0,
        "retrieval_calls_made": 0,
    }, indent=2))
    print(f"Created: {JSON_PATH}")
    print(f"Created: {CSV_PATH}")


if __name__ == "__main__":
    main()
