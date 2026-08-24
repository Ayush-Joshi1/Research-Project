"""Run the full sequential RAG experiment on the fixed 50-query subset."""
import csv
import json
import os
import time
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.rag import generate_verification
from src.retrieval import Retriever
from src.rag.generator import MODEL_NAME

DEV_PATH = os.path.join(ROOT, "data", "processed", "dev_queries.jsonl")
RESULTS_DIR = os.path.join(ROOT, "results")
RAW_PATH = os.path.join(RESULTS_DIR, "rag_results_clean.jsonl")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "rag_summary_clean.csv")
METRICS_PATH = os.path.join(RESULTS_DIR, "rag_metrics_clean.json")
K_VALUES = (1, 3, 5, 10)
MAX_RETRIES = 3
SUCCESS_DELAY_SECONDS = 5


def load_queries():
    with open(DEV_PATH, "r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def is_transient_error(error):
    message = str(error).lower()
    return any(
        marker in message
        for marker in ("429", "rate limit", "resource exhausted", "temporarily", "timeout", "503", "500")
    )


def generate_with_retries(claim, documents):
    retries = 0
    while True:
        try:
            return generate_verification(claim, documents), retries, None
        except Exception as error:
            if not is_transient_error(error) or retries >= MAX_RETRIES:
                return None, retries, str(error)
            retries += 1
            time.sleep(5 * (2 ** (retries - 1)))


def write_jsonl_row(file_handle, row):
    file_handle.write(json.dumps(row) + "\n")
    file_handle.flush()


def main():
    start_time = time.monotonic()
    queries = load_queries()
    retriever = Retriever(rebuild=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_fields = [
        "query_id", "k", "ground_truth_label", "predicted_verdict", "correct", "explanation"
    ]
    summary_rows = []
    raw_rows = []
    retry_count = 0
    failed_count = 0

    with open(RAW_PATH, "w", encoding="utf-8") as raw_file:
        with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as summary_file:
            summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
            summary_writer.writeheader()
            summary_file.flush()
            for query in queries:
                claim = query["claim_text"]
                label = query.get("ground_truth_label")
                relevant_ids = set(query.get("relevant_corpus_ids", []))
                for k_value in K_VALUES:
                    documents = retriever.retrieve(claim, top_k=k_value)
                    generation, retries, error_message = generate_with_retries(claim, documents)
                    retry_count += retries
                    retrieved_ids = [document["corpus_id"] for document in documents]
                    scores = [document["similarity_score"] for document in documents]
                    retrieval_hit = any(doc_id in relevant_ids for doc_id in retrieved_ids)
                    if generation is None:
                        failed_count += 1
                        verdict = None
                        explanation = f"API failure: {error_message}"
                        correct = False
                    else:
                        verdict = generation["verdict"]
                        explanation = generation["explanation"]
                        correct = verdict == label

                    raw_row = {
                        "query_id": query["query_id"],
                        "k": k_value,
                        "claim_text": claim,
                        "ground_truth_label": label,
                        "predicted_verdict": verdict,
                        "correct": correct,
                        "explanation": explanation,
                        "retrieved_corpus_ids": retrieved_ids,
                        "retrieved_similarity_scores": scores,
                    }
                    write_jsonl_row(raw_file, raw_row)
                    raw_rows.append(raw_row)
                    summary_row = {
                        "query_id": query["query_id"],
                        "k": k_value,
                        "ground_truth_label": label,
                        "predicted_verdict": verdict,
                        "correct": correct,
                        "explanation": explanation,
                    }
                    summary_writer.writerow(summary_row)
                    summary_file.flush()
                    summary_rows.append(summary_row)
                    if generation is not None:
                        time.sleep(SUCCESS_DELAY_SECONDS)

    metrics = build_metrics(summary_rows, raw_rows, retry_count, failed_count, MODEL_NAME)
    metrics["runtime_seconds"] = time.monotonic() - start_time
    metrics["success_delay_seconds"] = SUCCESS_DELAY_SECONDS
    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"Claims: {len(queries)}")
    print(f"K values: {list(K_VALUES)}")
    print(f"Intended API calls: {len(queries) * len(K_VALUES)}")
    print(f"Successful calls: {len(raw_rows) - failed_count}")
    print(f"Failed calls: {failed_count}")
    print(f"Retries: {retry_count}")
    print(f"Runtime seconds: {metrics['runtime_seconds']:.2f}")
    print(f"Model: {MODEL_NAME}")
    for k_value in K_VALUES:
        result = metrics["by_k"][str(k_value)]
        print(
            f"K={k_value}: retrieval_hit={result['retrieval_hit_at_k']}, "
            f"verdict_accuracy_among_successful_api_calls="
            f"{result['verdict_accuracy_among_successful_api_calls']}"
        )
    print(f"JSONL records: {len(raw_rows)}")
    print(f"CSV rows: {len(summary_rows)}")


def mean(values):
    return sum(values) / len(values) if values else None


def build_metrics(summary_rows, raw_rows, retries, failures, model):
    by_k = {}
    for k_value in K_VALUES:
        k_rows = [row for row in summary_rows if int(row["k"]) == k_value]
        raw_k_rows = [row for row in raw_rows if row["k"] == k_value]
        by_label = {}
        for label in ("SUPPORT", "CONTRADICT"):
            label_rows = [
                row for row in k_rows
                if row["ground_truth_label"] == label and row["predicted_verdict"] is not None
            ]
            by_label[label] = {
                "verdict_accuracy": mean([int(row["correct"]) for row in label_rows]),
                "successful_calls": len(label_rows),
            }
        successful_rows = [row for row in k_rows if row["predicted_verdict"] is not None]
        by_k[str(k_value)] = {
            "retrieval_hit_at_k": retrieval_hit_for_rows(raw_k_rows),
            "verdict_accuracy_among_successful_api_calls": mean(
                [int(row["correct"]) for row in successful_rows]
            ),
            "intended_calls": len(k_rows),
            "successful_calls": len(successful_rows),
            "failed_calls": len(k_rows) - len(successful_rows),
            "failure_rate": (len(k_rows) - len(successful_rows)) / len(k_rows),
            "support_accuracy": by_label["SUPPORT"]["verdict_accuracy"],
            "contradict_accuracy": by_label["CONTRADICT"]["verdict_accuracy"],
            "insufficient_evidence_predictions": sum(
                row["predicted_verdict"] == "INSUFFICIENT_EVIDENCE" for row in successful_rows
            ),
        }

    prediction_counts = {}
    for verdict in ("SUPPORT", "CONTRADICT", "INSUFFICIENT_EVIDENCE"):
        prediction_counts[verdict] = sum(
            row["predicted_verdict"] == verdict for row in raw_rows
        )
    return {
        "model": model,
        "claims": len({row["query_id"] for row in summary_rows}),
        "k_values": list(K_VALUES),
        "intended_api_calls": len(raw_rows),
        "successful_calls": len(raw_rows) - failures,
        "failed_calls": failures,
        "retries": retries,
        "prediction_counts": prediction_counts,
        "by_k": by_k,
    }


def retrieval_hit_for_rows(rows):
    # The raw rows contain relevance flags only indirectly through document ids;
    # use the fixed development records to calculate retrieval Hit@K.
    queries = {query["query_id"]: query for query in load_queries()}
    hits = []
    for query_id in {row["query_id"] for row in rows}:
        query_rows = [row for row in rows if row["query_id"] == query_id]
        relevant = set(queries[query_id].get("relevant_corpus_ids", []))
        hits.append(
            int(any(doc_id in relevant for doc_id in query_rows[0]["retrieved_corpus_ids"]))
        )
    return mean(hits)


if __name__ == "__main__":
    main()
