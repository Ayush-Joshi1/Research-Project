"""Offline consolidation of the completed Day 1, Day 2, and Day 3 analyses."""
import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
INPUTS = {
    "day1_results": os.path.join(RESULTS, "rag_results_clean.jsonl"),
    "day1_metrics": os.path.join(RESULTS, "rag_metrics_clean.json"),
    "day2_results": os.path.join(RESULTS, "reranked_rag_results.jsonl"),
    "day2_metrics": os.path.join(RESULTS, "reranked_rag_metrics.json"),
    "day2_analysis": os.path.join(RESULTS, "reranked_rag_analysis.json"),
    "ranking_analysis": os.path.join(RESULTS, "ranking_analysis.json"),
    "case_studies": os.path.join(RESULTS, "final_case_study_analysis.json"),
    "day3_results": os.path.join(RESULTS, "evidence_comparison.jsonl"),
    "day3_metrics": os.path.join(RESULTS, "evidence_comparison_metrics.json"),
    "day3_summary": os.path.join(RESULTS, "evidence_comparison_summary.txt"),
}
OUTPUTS = {
    "json": os.path.join(RESULTS, "final_analysis.json"),
    "csv": os.path.join(RESULTS, "final_analysis.csv"),
    "cases": os.path.join(RESULTS, "final_case_studies.json"),
    "comparison": os.path.join(RESULTS, "final_comparison_table.csv"),
    "summary": os.path.join(RESULTS, "final_analysis_summary.txt"),
}
K_VALUES = (1, 3, 5, 10)
DAY3_K_VALUES = (3, 5)
CONDITIONS = ("baseline", "reranked")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def row_key(row):
    return str(row["query_id"]), int(row["k"]), row["condition"]


def mean(values):
    return sum(values) / len(values) if values else None


def accuracy(rows):
    successful = [row for row in rows if row["api_success"]]
    return mean([int(row["predicted_verdict"] == row["ground_truth"]) for row in successful])


def validate_day1(rows, metrics):
    keys = {(str(row["query_id"]), int(row["k"])) for row in rows}
    expected = {(str(row["query_id"]), k) for row in rows for k in K_VALUES}
    by_k = {}
    for k_value in K_VALUES:
        selected = [row for row in rows if int(row["k"]) == k_value]
        by_k[str(k_value)] = {
            "records": len(selected),
            "successful": sum(row.get("api_success", True) for row in selected),
            "failed": sum(not row.get("api_success", True) for row in selected),
            "retrieval_hit": mean([int(row["retrieval_hit"]) for row in selected]),
            "verdict_accuracy": accuracy(selected),
        }
    consistency = []
    for k_value in K_VALUES:
        source = metrics["by_k"][str(k_value)]
        observed = by_k[str(k_value)]
        consistency.append({
            "K": k_value,
            "retrieval_hit_matches": observed["retrieval_hit"] == source["retrieval_hit_at_k"],
            "verdict_accuracy_matches": observed["verdict_accuracy"] == source["verdict_accuracy_among_successful_api_calls"],
            "counts_match": observed["successful"] == source["successful_calls"] and observed["failed"] == source["failed_calls"],
        })
    return {
        "records": len(rows),
        "unique_combinations": len(keys),
        "duplicate_combinations": len(rows) - len(keys),
        "missing_combinations": len(expected - keys),
        "queries": len({str(row["query_id"]) for row in rows}),
        "K_values": sorted({int(row["k"]) for row in rows}),
        "successful_api_calls": sum(row["api_success"] for row in rows),
        "failed_api_calls": sum(not row["api_success"] for row in rows),
        "metric_consistency": consistency,
    }


def validate_day2(rows, metrics, analysis):
    counts = Counter(row_key(row) for row in rows)
    canonical = {}
    for row in rows:
        canonical.setdefault(row_key(row), row)
    validation = analysis["validation"]
    return {
        "records": len(rows),
        "unique_combinations": len(canonical),
        "duplicate_combinations": sum(count - 1 for count in counts.values() if count > 1),
        "duplicate_keys": [list(item) for item, count in counts.items() if count > 1],
        "missing_combinations": len(analysis["validation"]["missing_combinations"]),
        "queries": len({str(row["query_id"]) for row in canonical.values()}),
        "K_values": sorted({int(row["k"]) for row in canonical.values()}),
        "successful_api_calls": sum(row["api_success"] for row in canonical.values()),
        "failed_api_calls": sum(not row["api_success"] for row in canonical.values()),
        "metric_consistency": {
            "analysis_unique_count_matches": len(canonical) == validation["unique_condition_query_K_combinations"],
            "analysis_failure_count_matches": sum(not row["api_success"] for row in canonical.values()) == validation["unique_combination_failed_api_calls"],
            "metrics_intended_calls_match": metrics["intended_api_calls"] == 400,
        },
    }


def validate_day3(rows, metrics):
    keys = {(str(row["query_id"]), int(row["k"]), row["condition"]) for row in rows}
    expected = {(str(row["query_id"]), k, condition) for row in rows for k in DAY3_K_VALUES for condition in ("full_documents", "original_evidence", "diverse_evidence")}
    return {
        "records": len(rows),
        "unique_combinations": len(keys),
        "duplicate_combinations": len(rows) - len(keys),
        "missing_combinations": len(expected - keys),
        "queries": len({str(row["query_id"]) for row in rows}),
        "K_values": sorted({int(row["k"]) for row in rows}),
        "conditions": sorted({row["condition"] for row in rows}),
        "successful_api_calls": sum(row["api_success"] for row in rows),
        "failed_api_calls": sum(not row["api_success"] for row in rows),
        "metric_consistency": {
            "metrics_total_matches": metrics["total_records"] == len(rows),
            "metrics_success_matches": metrics["successful_calls"] == sum(row["api_success"] for row in rows),
            "metrics_intended_calls_match": metrics["intended_calls"] == 30,
        },
    }


def day1_table(metrics):
    return [{
        "day": "Day 1",
        "condition": "baseline",
        "representation": "full_documents",
        "K": k,
        "retrieval_hit": metrics["by_k"][str(k)]["retrieval_hit_at_k"],
        "verdict_accuracy": metrics["by_k"][str(k)]["verdict_accuracy_among_successful_api_calls"],
        "retrieval_delta_vs_day1": 0.0,
        "verdict_delta_vs_day1": 0.0,
    } for k in K_VALUES]


def day2_table(analysis):
    rows = []
    for item in analysis["main_comparison"]:
        rows.extend([
            {"day": "Day 2", "condition": "baseline", "representation": "full_documents", "K": item["K"], "retrieval_hit": item["Baseline Hit@K"], "verdict_accuracy": item["Baseline Verdict Accuracy"], "retrieval_delta_vs_day1": 0.0, "verdict_delta_vs_day1": 0.0},
            {"day": "Day 2", "condition": "reranked", "representation": "full_documents_reranked", "K": item["K"], "retrieval_hit": item["Reranked Hit@K"], "verdict_accuracy": item["Reranked Verdict Accuracy"], "retrieval_delta_vs_day1": item["Reranked Hit@K"] - item["Baseline Hit@K"], "verdict_delta_vs_day1": item["Reranked Verdict Accuracy"] - item["Baseline Verdict Accuracy"]},
        ])
    return rows


def day3_table(metrics):
    rows = []
    for condition, label in (("full_documents", "full_documents"), ("original_evidence", "original_evidence"), ("diverse_evidence", "diverse_evidence")):
        for k in DAY3_K_VALUES:
            item = metrics["by_condition_and_K"][condition][str(k)]
            rows.append({"day": "Day 3", "condition": condition, "representation": label, "K": k, "retrieval_hit": None, "verdict_accuracy": item["verdict_accuracy"], "retrieval_delta_vs_day1": None, "verdict_delta_vs_day1": item["verdict_accuracy"] - next(row["verdict_accuracy"] for row in day1_table(load_json(INPUTS["day1_metrics"])) if row["K"] == k)})
    return rows


def retrieval_verdict_relationship(day2_analysis):
    return day2_analysis["retrieval_verdict_relationship_by_K"]


def increasing_k_wrong_cases(day1_rows):
    grouped = defaultdict(dict)
    for row in day1_rows:
        grouped[str(row["query_id"])][int(row["k"])] = row
    cases = []
    for query_id, rows in grouped.items():
        for previous_k, current_k in zip(K_VALUES, K_VALUES[1:]):
            previous = rows.get(previous_k)
            current = rows.get(current_k)
            if previous and current and current["retrieval_hit"] and not previous["retrieval_hit"] and current["api_success"] and current["predicted_verdict"] != current["ground_truth"]:
                cases.append({"query_id": query_id, "from_K": previous_k, "to_K": current_k, "retrieval": f"{previous['retrieval_hit']} -> {current['retrieval_hit']}", "prediction_at_higher_K": current["predicted_verdict"], "ground_truth": current["ground_truth"]})
    return cases


def main():
    missing = [path for path in INPUTS.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing authoritative inputs: " + ", ".join(missing))
    day1_rows = load_jsonl(INPUTS["day1_results"])
    dev_queries = {
        str(query["query_id"]): query
        for query in load_jsonl(os.path.join(ROOT, "data", "processed", "dev_queries.jsonl"))
    }
    day1_rows = [
        {
            **row,
            "ground_truth": row.get("ground_truth", row["ground_truth_label"]),
            "api_success": row.get("api_success", True),
            "retrieval_hit": row.get(
                "retrieval_hit",
                bool(set(row["retrieved_corpus_ids"]).intersection(
                    set(dev_queries[str(row["query_id"])].get("relevant_corpus_ids", []))
                )),
            ),
        }
        for row in day1_rows
    ]
    day1_metrics = load_json(INPUTS["day1_metrics"])
    day2_rows = load_jsonl(INPUTS["day2_results"])
    day2_metrics = load_json(INPUTS["day2_metrics"])
    day2_analysis = load_json(INPUTS["day2_analysis"])
    ranking = load_json(INPUTS["ranking_analysis"])
    cases = load_json(INPUTS["case_studies"])
    day3_rows = load_jsonl(INPUTS["day3_results"])
    day3_metrics = load_json(INPUTS["day3_metrics"])
    day1_validation = validate_day1(day1_rows, day1_metrics)
    day2_validation = validate_day2(day2_rows, day2_metrics, day2_analysis)
    day3_validation = validate_day3(day3_rows, day3_metrics)

    comparison_rows = day1_table(day1_metrics) + day2_table(day2_analysis) + day3_table(day3_metrics)
    day2_relationship = retrieval_verdict_relationship(day2_analysis)
    strongest = {
        "positive_verdict_changes": day2_analysis["largest_positive_verdict_changes"],
        "negative_verdict_changes": day2_analysis["largest_negative_verdict_changes"],
    }
    relationships = {
        "day2_reranking_by_K": day2_relationship,
        "cases_retrieval_improved_and_verdict_improved": {k: v["A_retrieval_improved_and_verdict_improved"] for k, v in day2_relationship.items()},
        "cases_retrieval_improved_but_verdict_did_not_improve": {k: v["B_retrieval_improved_but_verdict_unchanged"] + v["C_retrieval_improved_but_verdict_worsened"] for k, v in day2_relationship.items()},
        "cases_retrieval_unchanged_but_verdict_changed": {k: v["D_retrieval_unchanged_and_verdict_improved"] + v["F_retrieval_unchanged_and_verdict_worsened"] for k, v in day2_relationship.items()},
        "increasing_K_retrieval_improved_generation_incorrect": increasing_k_wrong_cases(day1_rows),
    }
    validation = {"day1": day1_validation, "day2": day2_validation, "day3": day3_validation}
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "authoritative_sources": INPUTS,
        "validation": validation,
        "day1_baseline": day1_metrics["by_k"],
        "day2_reranking": day2_analysis["main_comparison"],
        "day3_evidence_generation": day3_metrics["by_condition_and_K"],
        "ranking_analysis": {"FAISS_recall_at_K": ranking["recall_at_k"], "cross_encoder_recall_at_K": {str(k): day2_analysis["metrics_by_condition_and_K"][f"reranked_k{k}"]["retrieval_hit_at_k"] for k in K_VALUES}},
        "comparison_rows": comparison_rows,
        "retrieval_verdict_relationship": relationships,
        "strongest_changes": strongest,
        "case_studies": cases,
        "interpretation_framework": {"observation": "Directly recorded metric or prediction pattern.", "interpretation": "Cautious descriptive reading of an observed pattern.", "unsupported_causal_claim": "No claim that reranking or evidence representation caused a verdict change."},
        "api_calls_made_by_final_analysis": 0,
        "retrieval_calls_made_by_final_analysis": 0,
        "reranking_calls_made_by_final_analysis": 0,
    }
    with open(OUTPUTS["json"], "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    with open(OUTPUTS["cases"], "w", encoding="utf-8") as handle:
        json.dump(cases, handle, indent=2)

    fields = ["day", "condition", "representation", "K", "retrieval_hit", "verdict_accuracy", "retrieval_delta_vs_day1", "verdict_delta_vs_day1"]
    with open(OUTPUTS["csv"], "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison_rows)
    comparison_fields = ["K", "day1_baseline_hit", "day2_baseline_hit", "day2_reranked_hit", "day2_retrieval_delta", "day1_verdict_accuracy", "day2_baseline_verdict_accuracy", "day2_reranked_verdict_accuracy", "day2_verdict_delta"]
    with open(OUTPUTS["comparison"], "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        for item in day2_analysis["main_comparison"]:
            day1 = day1_metrics["by_k"][str(item["K"])]
            writer.writerow({"K": item["K"], "day1_baseline_hit": day1["retrieval_hit_at_k"], "day2_baseline_hit": item["Baseline Hit@K"], "day2_reranked_hit": item["Reranked Hit@K"], "day2_retrieval_delta": item["Retrieval Delta"], "day1_verdict_accuracy": day1["verdict_accuracy_among_successful_api_calls"], "day2_baseline_verdict_accuracy": item["Baseline Verdict Accuracy"], "day2_reranked_verdict_accuracy": item["Reranked Verdict Accuracy"], "day2_verdict_delta": item["Verdict Delta"]})

    lines = [
        "FINAL OFFLINE RAG ANALYSIS",
        "",
        "AUTHORITATIVE DATA SOURCES",
        "Day 1: results/rag_results_clean.jsonl and results/rag_metrics_clean.json",
        "Day 2: results/reranked_rag_results.jsonl, results/reranked_rag_metrics.json, results/reranked_rag_analysis.json, results/ranking_analysis.json, results/final_case_study_analysis.json",
        "Day 3: results/evidence_comparison.jsonl and results/evidence_comparison_metrics.json",
        "",
        "VALIDATION",
        json.dumps(validation, indent=2),
        "",
        "DAY 1 RESULTS",
    ]
    for row in day1_table(day1_metrics):
        lines.append(f"K={row['K']}: Hit@K={row['retrieval_hit']:.3f}, verdict accuracy={row['verdict_accuracy']:.3f}")
    lines.extend(["", "DAY 2 RESULTS"])
    for row in day2_analysis["main_comparison"]:
        lines.append(f"K={row['K']}: baseline Hit={row['Baseline Hit@K']:.3f}, reranked Hit={row['Reranked Hit@K']:.3f} ({row['Retrieval Delta']:+.3f}); baseline verdict={row['Baseline Verdict Accuracy']:.3f}, reranked verdict={row['Reranked Verdict Accuracy']:.3f} ({row['Verdict Delta']:+.3f})")
    lines.extend(["", "DAY 3 RESULTS", "K=3 and K=5 were tested with full documents, original evidence, and diverse evidence. All three representations produced identical predictions in the 10 paired query/K cases."])
    for condition in ("full_documents", "original_evidence", "diverse_evidence"):
        lines.append(f"{condition}: K=3 accuracy={day3_metrics['by_condition_and_K'][condition]['3']['verdict_accuracy']:.3f}; K=5 accuracy={day3_metrics['by_condition_and_K'][condition]['5']['verdict_accuracy']:.3f}")
    lines.extend(["", "RETRIEVAL VS VERDICT ANALYSIS", "Day 2 relationship counts are descriptive and come from paired successful calls.", json.dumps(relationships, indent=2), "", "STRONGEST IMPROVEMENTS", json.dumps(strongest["positive_verdict_changes"], indent=2), "", "STRONGEST REGRESSIONS", json.dumps(strongest["negative_verdict_changes"], indent=2), "", "CASE STUDIES"])
    for query_id in ("1320", "1019", "1370", "1185", "314"):
        lines.append(json.dumps(cases["case_studies"][query_id], indent=2))
    lines.extend([
        "",
        "LIMITATIONS",
        "This consolidation inherits the limitations of the source analyses: limited development samples, quota-related failed calls in Day 2, successful-call-only verdict accuracy, one duplicate raw Day 2 record, single datasets/models/configurations, and no statistical significance testing or confidence intervals.",
        "",
        "CAUTION",
        "Observation is separated from interpretation in the JSON output. The results do not establish causality, statistical significance, or universal superiority of reranking or evidence extraction.",
        "",
        "CONCLUSION",
        "Across the stored experiments, reranking improved observed retrieval ranking at lower K values and was associated with higher observed verdict accuracy at K=1, 3, and 5, while the K=10 verdict difference was slightly negative. In the small Day 3 comparison, changing evidence representation did not change Gemini predictions. These are descriptive findings, not causal conclusions.",
    ])
    with open(OUTPUTS["summary"], "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"validation": validation, "outputs": OUTPUTS, "api_calls": 0, "retrieval_calls": 0, "reranking_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
