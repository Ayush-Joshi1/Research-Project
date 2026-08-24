"""Analyze Day 2 error patterns from the existing clean RAG results."""
import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEV_PATH = os.path.join(ROOT, "data", "processed", "dev_queries.jsonl")
RAW_PATH = os.path.join(ROOT, "results", "rag_results_clean.jsonl")
ERRORS_PATH = os.path.join(ROOT, "results", "error_analysis.jsonl")
SUMMARY_PATH = os.path.join(ROOT, "results", "error_analysis_summary.csv")
METRICS_PATH = os.path.join(ROOT, "results", "error_analysis_metrics.json")
CASES_PATH = os.path.join(ROOT, "results", "interesting_cases.json")
K_VALUES = (1, 3, 5, 10)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def classify(row, relevant_ids):
    prediction = row["predicted_verdict"]
    retrieval_hit = any(
        corpus_id in relevant_ids for corpus_id in row["retrieved_corpus_ids"]
    )
    if prediction == row["ground_truth_label"]:
        category = "CORRECT"
    elif prediction == "INSUFFICIENT_EVIDENCE":
        category = "INSUFFICIENT_EVIDENCE"
    elif not retrieval_hit:
        category = "RETRIEVAL_MISS"
    else:
        category = "RETRIEVAL_HIT_WRONG_VERDICT"
    return retrieval_hit, category


def main():
    dev_queries = {
        query["query_id"]: query for query in load_jsonl(DEV_PATH)
    }
    raw_rows = load_jsonl(RAW_PATH)
    analyzed = []
    for row in raw_rows:
        query = dev_queries[row["query_id"]]
        relevant_ids = set(query.get("relevant_corpus_ids", []))
        retrieval_hit, category = classify(row, relevant_ids)
        analyzed.append(
            {
                "query_id": row["query_id"],
                "k": row["k"],
                "ground_truth_label": row["ground_truth_label"],
                "predicted_verdict": row["predicted_verdict"],
                "retrieval_hit": retrieval_hit,
                "category": category,
                "retrieved_corpus_ids": row["retrieved_corpus_ids"],
                "relevant_corpus_ids": sorted(relevant_ids),
                "explanation": row["explanation"],
            }
        )

    with open(ERRORS_PATH, "w", encoding="utf-8") as file_handle:
        for row in analyzed:
            file_handle.write(json.dumps(row) + "\n")

    summary_fields = [
        "query_id", "k", "ground_truth_label", "predicted_verdict",
        "retrieval_hit", "category",
    ]
    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in summary_fields} for row in analyzed)

    by_query = {}
    for row in analyzed:
        by_query.setdefault(row["query_id"], {})[int(row["k"])] = row

    aggregate = {}
    for k_value in K_VALUES:
        rows = [row for row in analyzed if int(row["k"]) == k_value]
        aggregate[str(k_value)] = {
            "K": k_value,
            "correct": sum(row["category"] == "CORRECT" for row in rows),
            "retrieval_miss": sum(row["category"] == "RETRIEVAL_MISS" for row in rows),
            "retrieval_hit_wrong_verdict": sum(
                row["category"] == "RETRIEVAL_HIT_WRONG_VERDICT" for row in rows
            ),
            "insufficient_evidence": sum(
                row["category"] == "INSUFFICIENT_EVIDENCE" for row in rows
            ),
            "retrieval_hit": sum(row["retrieval_hit"] for row in rows),
            "retrieval_miss_any_prediction": sum(
                not row["retrieval_hit"] for row in rows
            ),
        }

    correctness = {
        query_id: [by_query[query_id][k]["category"] == "CORRECT" for k in K_VALUES]
        for query_id in by_query
    }
    transitions = {
        "incorrect_to_correct": sum(
            any(not values[index] and values[index + 1] for index in range(3))
            for values in correctness.values()
        ),
        "correct_to_incorrect": sum(
            any(values[index] and not values[index + 1] for index in range(3))
            for values in correctness.values()
        ),
        "always_incorrect": sum(not any(values) for values in correctness.values()),
        "always_correct": sum(all(values) for values in correctness.values()),
    }
    retrieval_hit_wrong = [
        row for row in analyzed
        if row["retrieval_hit"] and row["category"] == "RETRIEVAL_HIT_WRONG_VERDICT"
    ]
    retrieval_miss_correct = [
        row for row in analyzed if not row["retrieval_hit"] and row["category"] == "CORRECT"
    ]

    interesting = select_interesting_cases(analyzed)
    with open(CASES_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(interesting, file_handle, indent=2)

    metrics = {
        "records_analyzed": len(analyzed),
        "queries_analyzed": len(by_query),
        "k_values": list(K_VALUES),
        "aggregate_by_k": aggregate,
        "transitions": transitions,
        "retrieval_hit_but_wrong_verdict": {
            "query_k_records": len(retrieval_hit_wrong),
            "unique_queries": len({row["query_id"] for row in retrieval_hit_wrong}),
        },
        "retrieval_miss_any_prediction": {
            "query_k_records": sum(not row["retrieval_hit"] for row in analyzed),
            "unique_queries": len({row["query_id"] for row in analyzed if not row["retrieval_hit"]}),
        },
        "retrieval_miss_but_correct_verdict": {
            "query_k_records": len(retrieval_miss_correct),
            "unique_queries": len({row["query_id"] for row in retrieval_miss_correct}),
        },
        "interesting_case_count": len(interesting),
        "api_calls_made": 0,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Created: {ERRORS_PATH}")
    print(f"Created: {SUMMARY_PATH}")
    print(f"Created: {METRICS_PATH}")
    print(f"Created: {CASES_PATH}")


def select_interesting_cases(rows):
    by_query = {}
    for row in rows:
        by_query.setdefault(row["query_id"], {})[int(row["k"])] = row

    def priority(row):
        query_rows = by_query[row["query_id"]]
        k_value = int(row["k"])
        later_hit = any(
            query_rows[later_k]["retrieval_hit"]
            for later_k in K_VALUES
            if later_k > k_value
        )
        previous_incorrect = (
            k_value > K_VALUES[0]
            and not query_rows[K_VALUES[K_VALUES.index(k_value) - 1]]["category"] == "CORRECT"
            and row["category"] == "CORRECT"
        )
        return (
            0 if not row["retrieval_hit"] and later_hit else
            1 if row["category"] == "RETRIEVAL_HIT_WRONG_VERDICT" else
            2 if previous_incorrect else
            3 if row["category"] == "INSUFFICIENT_EVIDENCE" else 4,
            k_value,
        )

    selected = []
    selected_by_k = {k_value: 0 for k_value in K_VALUES}
    for row in sorted(rows, key=priority):
        k_value = int(row["k"])
        if selected_by_k[k_value] >= 3:
            continue
        if any(
            selected_case["query_id"] == row["query_id"]
            and selected_case["K"] == row["k"]
            for selected_case in selected
        ):
            continue
        selected.append(
            {
                "query_id": row["query_id"],
                "claim": next(
                    query["claim_text"]
                    for query in load_jsonl(DEV_PATH)
                    if query["query_id"] == row["query_id"]
                ),
                "ground_truth": row["ground_truth_label"],
                "K": row["k"],
                "prediction": row["predicted_verdict"],
                "retrieval_hit": row["retrieval_hit"],
                "category": row["category"],
                "retrieved_document_ids": row["retrieved_corpus_ids"],
                "short_explanation": row["explanation"],
            }
        )
        selected_by_k[k_value] += 1
        if len(selected) == 10:
            break
    return selected


if __name__ == "__main__":
    main()
