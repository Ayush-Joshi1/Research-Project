"""Compare full-document and two evidence representations on five queries."""
import json
import os
import sys
import time
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.rag.diverse_evidence_extractor import extract_diverse_evidence
from src.rag.evidence_extractor import extract_evidence
from src.rag.evidence_generator import generate_evidence_verification
from src.rag.generator import generate_verification
from src.retrieval import Retriever

DEV_PATH = os.path.join(ROOT, "data", "processed", "dev_queries.jsonl")
OUTPUT_PATH = os.path.join(ROOT, "results", "evidence_comparison.jsonl")
METRICS_PATH = os.path.join(ROOT, "results", "evidence_comparison_metrics.json")
SUMMARY_PATH = os.path.join(ROOT, "results", "evidence_comparison_summary.txt")
QUERY_IDS = {"118", "1019", "1320", "1370", "1185"}
K_VALUES = (3, 5)
CONDITIONS = ("full_documents", "original_evidence", "diverse_evidence")
SUCCESS_DELAY_SECONDS = 5
RETRY_WAIT_SECONDS = 60


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_rate_limit_error(error):
    message = str(error).lower()
    return "429" in message or "rate limit" in message or "resource exhausted" in message


def generate_with_one_retry(generator, claim, value):
    try:
        return generator(claim, value), 1, 0, None
    except Exception as error:
        if not is_rate_limit_error(error):
            return None, 1, 0, str(error)
        time.sleep(RETRY_WAIT_SECONDS)
        try:
            return generator(claim, value), 2, 1, None
        except Exception as retry_error:
            return None, 2, 1, str(retry_error)


def make_record(query, k_value, condition, documents, evidence, selection_scores, generation, error):
    record = {
        "query_id": query["query_id"],
        "k": k_value,
        "condition": condition,
        "ground_truth": query["ground_truth_label"],
        "predicted_verdict": generation["verdict"] if generation else None,
        "explanation": generation["explanation"] if generation else f"API failure: {error}",
        "retrieved_corpus_ids": [document["corpus_id"] for document in documents],
        "api_success": generation is not None,
    }
    if condition != "full_documents":
        record["evidence_sentences"] = evidence
    if condition == "diverse_evidence":
        record["selection_scores"] = selection_scores
    return record


def metric_for(records, condition, label=None):
    selected = [record for record in records if record["condition"] == condition and record["api_success"]]
    if label is not None:
        selected = [record for record in selected if record["ground_truth"] == label]
    return sum(record["predicted_verdict"] == record["ground_truth"] for record in selected) / len(selected) if selected else None


def build_metrics(records):
    metrics = {"by_condition_and_K": {}, "prediction_comparisons": {}}
    for condition in CONDITIONS:
        metrics["by_condition_and_K"][condition] = {}
        for k_value in K_VALUES:
            selected = [record for record in records if record["condition"] == condition and record["k"] == k_value]
            successful = [record for record in selected if record["api_success"]]
            metrics["by_condition_and_K"][condition][str(k_value)] = {
                "successful_calls": len(successful),
                "failed_calls": len(selected) - len(successful),
                "verdict_accuracy": metric_for([record for record in records if record["k"] == k_value], condition),
                "SUPPORT_accuracy": metric_for([record for record in records if record["k"] == k_value], condition, "SUPPORT"),
                "CONTRADICT_accuracy": metric_for([record for record in records if record["k"] == k_value], condition, "CONTRADICT"),
                "INSUFFICIENT_EVIDENCE_count": sum(record["predicted_verdict"] == "INSUFFICIENT_EVIDENCE" for record in successful),
            }

    for k_value in K_VALUES:
        for query_id in QUERY_IDS:
            by_condition = {
                condition: next((record for record in records if record["query_id"] == query_id and record["k"] == k_value and record["condition"] == condition), None)
                for condition in CONDITIONS
            }
            full = by_condition["full_documents"]
            if not full or not full["api_success"]:
                continue
            for condition in ("original_evidence", "diverse_evidence"):
                other = by_condition[condition]
                if not other or not other["api_success"]:
                    continue
                metrics["prediction_comparisons"].setdefault(condition, {"changed_relative_to_full": 0, "matching_full": 0, "correct_other_incorrect_full": 0, "incorrect_other_correct_full": 0})
                if other["predicted_verdict"] == full["predicted_verdict"]:
                    metrics["prediction_comparisons"][condition]["matching_full"] += 1
                else:
                    metrics["prediction_comparisons"][condition]["changed_relative_to_full"] += 1
                other_correct = other["predicted_verdict"] == other["ground_truth"]
                full_correct = full["predicted_verdict"] == full["ground_truth"]
                if other_correct and not full_correct:
                    metrics["prediction_comparisons"][condition]["correct_other_incorrect_full"] += 1
                if full_correct and not other_correct:
                    metrics["prediction_comparisons"][condition]["incorrect_other_correct_full"] += 1
    return metrics


def write_summary(records, metrics):
    lines = [
        "EVIDENCE REPRESENTATION COMPARISON",
        "Exploratory five-query comparison. Accuracy is calculated among successful API calls only.",
        "",
        "ACCURACY BY K",
        "| K | Full Document Accuracy | Original Evidence Accuracy | Diverse Evidence Accuracy |",
        "|---:|---:|---:|---:|",
    ]
    for k_value in K_VALUES:
        values = [metrics["by_condition_and_K"][condition][str(k_value)]["verdict_accuracy"] for condition in CONDITIONS]
        lines.append(f"| {k_value} | {values[0] if values[0] is not None else 'NA'} | {values[1] if values[1] is not None else 'NA'} | {values[2] if values[2] is not None else 'NA'} |")
    lines.extend([
        "",
        "QUERY-LEVEL COMPARISON",
        "| Query | K | Ground Truth | Full Document | Original Evidence | Diverse Evidence |",
        "|---:|---:|---|---|---|---|",
    ])
    for k_value in K_VALUES:
        for query_id in sorted(QUERY_IDS):
            values = []
            for condition in CONDITIONS:
                record = next(record for record in records if record["query_id"] == query_id and record["k"] == k_value and record["condition"] == condition)
                values.append(record["predicted_verdict"] or "FAILED")
            ground_truth = next(record["ground_truth"] for record in records if record["query_id"] == query_id and record["k"] == k_value)
            lines.append(f"| {query_id} | {k_value} | {ground_truth} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend(["", "PREDICTION COMPARISONS TO FULL DOCUMENTS"])
    lines.append(json.dumps(metrics["prediction_comparisons"], indent=2))
    lines.extend([
        "",
        "INTERPRETATION",
        "This is an exploratory five-query comparison. It does not establish statistical significance, causality, general superiority, or universal improvement from evidence extraction.",
    ])
    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    queries = [query for query in load_jsonl(DEV_PATH) if query["query_id"] in QUERY_IDS]
    if os.path.exists(OUTPUT_PATH):
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_PATH}")
    retriever = Retriever(rebuild=False)
    records = []
    api_calls = 0
    failures = 0
    with open(OUTPUT_PATH, "x", encoding="utf-8") as output_file:
        for query in queries:
            for k_value in K_VALUES:
                documents = retriever.retrieve(query["claim_text"], top_k=k_value)
                original_evidence = extract_evidence(query["claim_text"], documents)
                diverse_evidence = extract_diverse_evidence(query["claim_text"], documents)
                inputs = (
                    ("full_documents", generate_verification, documents, None),
                    ("original_evidence", generate_evidence_verification, original_evidence, original_evidence),
                    ("diverse_evidence", generate_evidence_verification, diverse_evidence, diverse_evidence),
                )
                for condition, generator, value, evidence in inputs:
                    selection_scores = [item["selection_score"] for item in evidence] if condition == "diverse_evidence" else None
                    generation, calls, retries, error = generate_with_one_retry(generator, query["claim_text"], value)
                    api_calls += calls
                    record = make_record(query, k_value, condition, documents, evidence, selection_scores, generation, error)
                    records.append(record)
                    failures += not record["api_success"]
                    output_file.write(json.dumps(record) + "\n")
                    output_file.flush()
                    print(f"Query {query['query_id']} K={k_value} {condition}: {record['predicted_verdict'] or 'FAILED'}")
                    if generation is not None:
                        time.sleep(SUCCESS_DELAY_SECONDS)

    metrics = build_metrics(records)
    metrics.update({
        "intended_calls": 30,
        "total_records": len(records),
        "successful_calls": sum(record["api_success"] for record in records),
        "failed_calls": failures,
        "api_calls_including_retries": api_calls,
        "retry_wait_seconds": RETRY_WAIT_SECONDS,
        "success_delay_seconds": SUCCESS_DELAY_SECONDS,
    })
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    write_summary(records, metrics)
    print(json.dumps({"records": len(records), "successful": metrics["successful_calls"], "failed": failures, "api_calls": api_calls}, indent=2))


if __name__ == "__main__":
    main()
