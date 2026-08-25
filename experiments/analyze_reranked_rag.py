"""Analyze completed reranked RAG results without making model or API calls."""
import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
RAW_PATH = os.path.join(RESULTS_DIR, "reranked_rag_results.jsonl")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "reranked_rag_summary.csv")
METRICS_PATH = os.path.join(RESULTS_DIR, "reranked_rag_metrics.json")
DEV_PATH = os.path.join(ROOT, "data", "processed", "dev_queries.jsonl")
ANALYSIS_JSON_PATH = os.path.join(RESULTS_DIR, "reranked_rag_analysis.json")
ANALYSIS_CSV_PATH = os.path.join(RESULTS_DIR, "reranked_rag_analysis.csv")
CASE_STUDIES_PATH = os.path.join(RESULTS_DIR, "reranked_rag_case_studies.json")
CONDITIONS = ("baseline", "reranked")
K_VALUES = (1, 3, 5, 10)
CASE_QUERY_IDS = ("1320", "1019", "1370", "1185", "314")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row):
    return row["condition"], str(row["query_id"]), int(row["k"])


def mean(values):
    return sum(values) / len(values) if values else None


def accuracy(rows, label=None):
    successful = [row for row in rows if row["api_success"]]
    if label is not None:
        successful = [row for row in successful if row["ground_truth"] == label]
    return mean([int(row["predicted_verdict"] == row["ground_truth"]) for row in successful])


def correct(row):
    return row["predicted_verdict"] == row["ground_truth"] if row["api_success"] else None


def pair_classification(baseline, reranked):
    baseline_correct = correct(baseline)
    reranked_correct = correct(reranked)
    if baseline_correct and reranked_correct:
        return "both_correct"
    if baseline_correct is False and reranked_correct is True:
        return "baseline_wrong_reranked_correct"
    if baseline_correct is True and reranked_correct is False:
        return "baseline_correct_reranked_wrong"
    if baseline_correct is False and reranked_correct is False:
        return "both_wrong"
    return "not_both_successful"


def pair_status(baseline, reranked):
    if not (baseline["api_success"] and reranked["api_success"]):
        return None
    baseline_correct = correct(baseline)
    reranked_correct = correct(reranked)
    retrieval_improved = reranked["retrieval_hit"] and not baseline["retrieval_hit"]
    retrieval_unchanged = reranked["retrieval_hit"] == baseline["retrieval_hit"]
    verdict_improved = reranked_correct and not baseline_correct
    verdict_unchanged = reranked_correct == baseline_correct
    verdict_worsened = baseline_correct and not reranked_correct
    if retrieval_improved and verdict_improved:
        return "A_retrieval_improved_and_verdict_improved"
    if retrieval_improved and verdict_unchanged:
        return "B_retrieval_improved_but_verdict_unchanged"
    if retrieval_improved and verdict_worsened:
        return "C_retrieval_improved_but_verdict_worsened"
    if retrieval_unchanged and verdict_improved:
        return "D_retrieval_unchanged_and_verdict_improved"
    if retrieval_unchanged and verdict_unchanged:
        return "E_retrieval_unchanged_and_verdict_unchanged"
    if retrieval_unchanged and verdict_worsened:
        return "F_retrieval_unchanged_and_verdict_worsened"
    return "retrieval_status_changed_without_improvement_category"


def main():
    queries = load_jsonl(DEV_PATH)
    rows = load_jsonl(RAW_PATH)
    # Read companion inputs to ensure the analysis is based on the completed run artifacts.
    with open(SUMMARY_PATH, "r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    with open(METRICS_PATH, "r", encoding="utf-8") as handle:
        source_metrics = json.load(handle)

    expected = [
        (query["query_id"], condition, k_value)
        for query in queries
        for condition in CONDITIONS
        for k_value in K_VALUES
    ]
    expected_keys = {(condition, str(query_id), k_value) for query_id, condition, k_value in expected}
    counts = Counter(key(row) for row in rows)
    unique_keys = set(counts)
    first_by_key = {}
    for row in rows:
        first_by_key.setdefault(key(row), row)
    canonical_rows = list(first_by_key.values())
    missing_keys = [item for item in expected if (item[1], str(item[0]), item[2]) not in unique_keys]

    by_condition_k = {}
    for condition in CONDITIONS:
        for k_value in K_VALUES:
            selected = [row for row in canonical_rows if row["condition"] == condition and int(row["k"]) == k_value]
            successful = [row for row in selected if row["api_success"]]
            by_condition_k[f"{condition}_k{k_value}"] = {
                "condition": condition,
                "K": k_value,
                "retrieval_hit_at_k": mean([int(row["retrieval_hit"]) for row in selected]),
                "verdict_accuracy_among_successful_api_calls": accuracy(selected),
                "SUPPORT_accuracy_among_successful_api_calls": accuracy(selected, "SUPPORT"),
                "CONTRADICT_accuracy_among_successful_api_calls": accuracy(selected, "CONTRADICT"),
                "INSUFFICIENT_EVIDENCE_count": sum(row["predicted_verdict"] == "INSUFFICIENT_EVIDENCE" for row in successful),
                "successful_api_calls": len(successful),
                "failed_api_calls": len(selected) - len(successful),
            }

    comparison = []
    for k_value in K_VALUES:
        baseline = by_condition_k[f"baseline_k{k_value}"]
        reranked = by_condition_k[f"reranked_k{k_value}"]
        comparison.append({
            "K": k_value,
            "Baseline Hit@K": baseline["retrieval_hit_at_k"],
            "Reranked Hit@K": reranked["retrieval_hit_at_k"],
            "Retrieval Delta": reranked["retrieval_hit_at_k"] - baseline["retrieval_hit_at_k"],
            "Baseline Verdict Accuracy": baseline["verdict_accuracy_among_successful_api_calls"],
            "Reranked Verdict Accuracy": reranked["verdict_accuracy_among_successful_api_calls"],
            "Verdict Delta": reranked["verdict_accuracy_among_successful_api_calls"] - baseline["verdict_accuracy_among_successful_api_calls"],
        })

    paired_by_k = defaultdict(list)
    compact_rows = []
    for query in queries:
        query_id = str(query["query_id"])
        ground_truth = query["ground_truth_label"]
        for k_value in K_VALUES:
            baseline = first_by_key.get(("baseline", query_id, k_value))
            reranked = first_by_key.get(("reranked", query_id, k_value))
            if baseline is None or reranked is None:
                continue
            classification = pair_classification(baseline, reranked)
            if classification != "not_both_successful":
                paired_by_k[k_value].append((baseline, reranked, classification))
            compact_rows.append({
                "query_id": query_id,
                "K": k_value,
                "ground_truth": ground_truth,
                "baseline_prediction": baseline["predicted_verdict"],
                "reranked_prediction": reranked["predicted_verdict"],
                "baseline_retrieval_hit": baseline["retrieval_hit"],
                "reranked_retrieval_hit": reranked["retrieval_hit"],
                "baseline_correct": correct(baseline),
                "reranked_correct": correct(reranked),
                "verdict_changed": baseline["predicted_verdict"] != reranked["predicted_verdict"] if baseline["api_success"] and reranked["api_success"] else None,
                "verdict_improved": correct(reranked) and not correct(baseline) if baseline["api_success"] and reranked["api_success"] else None,
                "verdict_worsened": correct(baseline) and not correct(reranked) if baseline["api_success"] and reranked["api_success"] else None,
                "retrieval_changed": baseline["retrieval_hit"] != reranked["retrieval_hit"],
            })

    transitions = {}
    retrieval_verdict_relationship = {}
    paired_changes = []
    for k_value in K_VALUES:
        pairs = paired_by_k[k_value]
        transition_counts = Counter(item[2] for item in pairs)
        transitions[str(k_value)] = {
            "paired_successful_queries": len(pairs),
            "baseline_wrong_reranked_correct": transition_counts["baseline_wrong_reranked_correct"],
            "baseline_correct_reranked_wrong": transition_counts["baseline_correct_reranked_wrong"],
            "both_correct": transition_counts["both_correct"],
            "both_wrong": transition_counts["both_wrong"],
            "same_prediction": sum(a["predicted_verdict"] == b["predicted_verdict"] for a, b, _ in pairs),
            "prediction_changed": sum(a["predicted_verdict"] != b["predicted_verdict"] for a, b, _ in pairs),
        }
        relationship_counts = Counter(pair_status(a, b) for a, b, _ in pairs)
        retrieval_verdict_relationship[str(k_value)] = {
            category: relationship_counts[category]
            for category in (
                "A_retrieval_improved_and_verdict_improved",
                "B_retrieval_improved_but_verdict_unchanged",
                "C_retrieval_improved_but_verdict_worsened",
                "D_retrieval_unchanged_and_verdict_improved",
                "E_retrieval_unchanged_and_verdict_unchanged",
                "F_retrieval_unchanged_and_verdict_worsened",
            )
        }
        for baseline, reranked, _ in pairs:
            paired_changes.append({
                "query_id": str(baseline["query_id"]),
                "K": k_value,
                "delta": int(correct(reranked)) - int(correct(baseline)),
            })

    largest_positive = [item for item in paired_changes if item["delta"] == 1]
    largest_negative = [item for item in paired_changes if item["delta"] == -1]
    k10 = comparison[-1]

    case_studies = {}
    for query_id in CASE_QUERY_IDS:
        entries = []
        for k_value in K_VALUES:
            baseline = first_by_key.get(("baseline", query_id, k_value))
            reranked = first_by_key.get(("reranked", query_id, k_value))
            if baseline is None or reranked is None:
                entries.append({"query_id": query_id, "K": k_value, "missing": True})
                continue
            baseline_correct = correct(baseline)
            reranked_correct = correct(reranked)
            entries.append({
                "query_id": query_id,
                "K": k_value,
                "ground_truth": baseline["ground_truth"],
                "baseline_prediction": baseline["predicted_verdict"],
                "reranked_prediction": reranked["predicted_verdict"],
                "baseline_retrieval_hit": baseline["retrieval_hit"],
                "reranked_retrieval_hit": reranked["retrieval_hit"],
                "baseline_retrieved_corpus_ids": baseline["retrieved_corpus_ids"],
                "reranked_retrieved_corpus_ids": reranked["retrieved_corpus_ids"],
                "reranker_scores": reranked["reranker_scores"],
                "original_faiss_ranks": reranked["original_faiss_ranks"],
                "verdict_status": "improved" if reranked_correct and not baseline_correct else "worsened" if baseline_correct and not reranked_correct else "unchanged",
                "retrieval_status": "changed" if baseline["retrieval_hit"] != reranked["retrieval_hit"] else "unchanged",
            })
        case_studies[query_id] = entries

    analysis = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_files": [RAW_PATH, SUMMARY_PATH, METRICS_PATH, DEV_PATH],
        "validation": {
            "total_records": len(rows),
            "unique_condition_query_K_combinations": len(unique_keys),
            "duplicate_combinations": sum(count - 1 for count in counts.values() if count > 1),
            "duplicate_keys": [list(item) for item, count in counts.items() if count > 1],
            "missing_combinations": [list(item) for item in missing_keys],
            "successful_api_calls": sum(row["api_success"] for row in rows),
            "failed_api_calls": sum(not row["api_success"] for row in rows),
            "unique_combination_successful_api_calls": sum(row["api_success"] for row in canonical_rows),
            "unique_combination_failed_api_calls": sum(not row["api_success"] for row in canonical_rows),
            "summary_csv_rows_read": len(summary_rows),
            "source_metrics_intended_api_calls": source_metrics.get("intended_api_calls"),
        },
        "metrics_by_condition_and_K": by_condition_k,
        "main_comparison": comparison,
        "verdict_transitions_by_K": transitions,
        "retrieval_verdict_relationship_by_K": retrieval_verdict_relationship,
        "K10_descriptive_observation": {
            "baseline_hit_at_10": k10["Baseline Hit@K"],
            "reranked_hit_at_10": k10["Reranked Hit@K"],
            "verdict_accuracy_difference": k10["Verdict Delta"],
            "observation": "At K=10 the baseline and reranked conditions use the same FAISS top-10 candidate set, only reordered; this is a descriptive result and does not establish causality.",
        },
        "largest_positive_verdict_changes": largest_positive,
        "largest_negative_verdict_changes": largest_negative,
        "paired_successful_count_by_K": {str(k): len(paired_by_k[k]) for k in K_VALUES},
    }

    with open(ANALYSIS_JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
    with open(CASE_STUDIES_PATH, "w", encoding="utf-8") as handle:
        json.dump(case_studies, handle, indent=2)
    fields = list(compact_rows[0]) if compact_rows else []
    with open(ANALYSIS_CSV_PATH, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(compact_rows)

    print(json.dumps({"validation": analysis["validation"], "main_comparison": comparison, "verdict_transitions_by_K": transitions, "largest_positive": largest_positive, "largest_negative": largest_negative}, indent=2))


if __name__ == "__main__":
    main()
