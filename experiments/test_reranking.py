"""Evaluate cross-encoder reranking of saved top-10 FAISS candidates."""
import json
import os
import sys

from sentence_transformers import CrossEncoder

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CORPUS_PATH = os.path.join(ROOT, "data", "processed", "corpus.jsonl")
DEV_PATH = os.path.join(ROOT, "data", "processed", "dev_queries.jsonl")
RETRIEVAL_PATH = os.path.join(ROOT, "results", "retrieval_results.jsonl")
JSON_PATH = os.path.join(ROOT, "results", "reranking_sanity.json")
CSV_PATH = os.path.join(ROOT, "results", "reranking_sanity.csv")
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
K_VALUES = (1, 3, 5, 10)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def first_relevant_rank(documents, relevant_ids):
    for rank, document in enumerate(documents, start=1):
        if document["corpus_id"] in relevant_ids:
            return rank
    return None


def main():
    corpus = {
        document["corpus_id"]: document
        for document in load_jsonl(CORPUS_PATH)
    }
    dev_queries = load_jsonl(DEV_PATH)
    retrieval_rows = load_jsonl(RETRIEVAL_PATH)
    rankings = {}
    for row in retrieval_rows:
        if int(row["K"]) == 10:
            rankings.setdefault(row["query_id"], []).append(
                {
                    "corpus_id": row["corpus_id"],
                    "faiss_rank": int(row["retrieved_rank"]),
                    "faiss_score": row["similarity_score"],
                }
            )

    model = CrossEncoder(MODEL_NAME)
    analyses = []
    for query in dev_queries:
        query_id = query["query_id"]
        claim = query["claim_text"]
        relevant_ids = set(query.get("relevant_corpus_ids", []))
        original = sorted(rankings[query_id], key=lambda row: row["faiss_rank"])
        pairs = [(claim, corpus[row["corpus_id"]]["text"]) for row in original]
        scores = model.predict(pairs)
        reranked = [
            {**row, "cross_encoder_score": float(score)}
            for row, score in zip(original, scores)
        ]
        reranked.sort(key=lambda row: row["cross_encoder_score"], reverse=True)
        original_rank = first_relevant_rank(original, relevant_ids)
        reranked_rank = first_relevant_rank(reranked, relevant_ids)
        analyses.append(
            {
                "query_id": query_id,
                "ground_truth": query["ground_truth_label"],
                "relevant_corpus_ids": sorted(relevant_ids),
                "original_first_relevant_rank": original_rank,
                "reranked_first_relevant_rank": reranked_rank,
                "original_top_10": [row["corpus_id"] for row in original],
                "reranked_top_10": [row["corpus_id"] for row in reranked],
            }
        )

    def recall(analysis_rows, field, k):
        return sum(
            row[field] is not None and row[field] <= k
            for row in analysis_rows
        ) / len(analysis_rows)

    rank_changes = [
        row for row in analyses
        if row["original_first_relevant_rank"] is not None
        and row["reranked_first_relevant_rank"] is not None
    ]
    improved = [
        row for row in rank_changes
        if row["reranked_first_relevant_rank"] < row["original_first_relevant_rank"]
    ]
    worsened = [
        row for row in rank_changes
        if row["reranked_first_relevant_rank"] > row["original_first_relevant_rank"]
    ]
    unchanged = [
        row for row in rank_changes
        if row["reranked_first_relevant_rank"] == row["original_first_relevant_rank"]
    ]
    top_improvements = sorted(
        improved,
        key=lambda row: (
            row["original_first_relevant_rank"] - row["reranked_first_relevant_rank"],
            row["query_id"],
        ),
        reverse=True,
    )[:10]
    top_improvements = [
        {
            "query_id": row["query_id"],
            "original_rank": row["original_first_relevant_rank"],
            "reranked_rank": row["reranked_first_relevant_rank"],
            "original_corpus_id": next(
                corpus_id for corpus_id in row["original_top_10"]
                if corpus_id in row["relevant_corpus_ids"]
            ),
            "relevant_corpus_ids": row["relevant_corpus_ids"],
        }
        for row in top_improvements
    ]
    query_1320 = next(row for row in analyses if row["query_id"] == "1320")

    report = {
        "model": MODEL_NAME,
        "queries_analyzed": len(analyses),
        "documents_scored": len(analyses) * 10,
        "api_calls_made": 0,
        "retrieval_calls_made": 0,
        "recall_before": {str(k): recall(analyses, "original_first_relevant_rank", k) for k in K_VALUES},
        "recall_after": {str(k): recall(analyses, "reranked_first_relevant_rank", k) for k in K_VALUES},
        "rank_changes": {
            "improved": len(improved),
            "worsened": len(worsened),
            "unchanged": len(unchanged),
            "not_comparable_due_to_no_relevant_document_in_top_10": len(analyses) - len(rank_changes),
        },
        "top_10_improvements": top_improvements,
        "query_1320": query_1320,
        "query_results": analyses,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2)

    fields = [
        "query_id", "original_first_relevant_rank", "reranked_first_relevant_rank",
        "ground_truth", "relevant_corpus_ids", "original_top_10", "reranked_top_10",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file_handle:
        import csv
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        writer.writeheader()
        for row in analyses:
            writer.writerow({
                "query_id": row["query_id"],
                "original_first_relevant_rank": row["original_first_relevant_rank"],
                "reranked_first_relevant_rank": row["reranked_first_relevant_rank"],
                "ground_truth": row["ground_truth"],
                "relevant_corpus_ids": json.dumps(row["relevant_corpus_ids"]),
                "original_top_10": json.dumps(row["original_top_10"]),
                "reranked_top_10": json.dumps(row["reranked_top_10"]),
            })

    print(json.dumps({
        "queries_analyzed": len(analyses),
        "documents_scored": len(analyses) * 10,
        "recall_before": report["recall_before"],
        "recall_after": report["recall_after"],
        "rank_changes": report["rank_changes"],
        "top_10_improvements": top_improvements,
        "query_1320": query_1320,
        "api_calls_made": 0,
        "retrieval_calls_made": 0,
    }, indent=2))
    print(f"Created: {JSON_PATH}")
    print(f"Created: {CSV_PATH}")


if __name__ == "__main__":
    main()
