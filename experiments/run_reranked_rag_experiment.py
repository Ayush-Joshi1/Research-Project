"""Run baseline and cross-encoder-reranked RAG conditions."""
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEV_PATH = os.path.join(ROOT, "data", "processed", "dev_queries.jsonl")
CORPUS_PATH = os.path.join(ROOT, "data", "processed", "corpus.jsonl")
RESULTS_DIR = os.path.join(ROOT, "results")
RAW_PATH = os.path.join(RESULTS_DIR, "reranked_rag_results.jsonl")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "reranked_rag_summary.csv")
METRICS_PATH = os.path.join(RESULTS_DIR, "reranked_rag_metrics.json")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "reranked_rag_checkpoint.json")
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
K_VALUES = (1, 3, 5, 10)
RETRY_WAIT_SECONDS = 60
SUCCESS_DELAY_SECONDS = 5
MAX_TEMPORARY_RATE_LIMIT_RETRIES = 5


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def is_quota_error(error):
    message = str(error).lower()
    return "429" in message or "rate limit" in message or "resource exhausted" in message


def is_daily_quota_error(error):
    message = str(error).lower()
    return (
        "per day" in message
        or "daily" in message
        or "quotaid" in message and "day" in message
        or "quota_exceeded" in message and "day" in message
    )


def retry_delay_seconds(error):
    message = str(error).lower()
    matches = re.findall(r"(?:retrydelay|retry delay)[^0-9]*(\d+)", message)
    return max(RETRY_WAIT_SECONDS, int(matches[0])) if matches else RETRY_WAIT_SECONDS


def generate_with_resilience(generate_verification, claim, documents):
    temporary_retries = 0
    api_calls = 0
    while True:
        try:
            generation = generate_verification(claim, documents)
            return generation, api_calls + 1, 0, None, None
        except Exception as error:
            api_calls += 1
            if not is_quota_error(error):
                return None, api_calls, 0, str(error), "permanent_failure"
            if is_daily_quota_error(error):
                return None, api_calls, 0, str(error), "daily_quota"
            if temporary_retries >= MAX_TEMPORARY_RATE_LIMIT_RETRIES:
                return None, api_calls, temporary_retries, str(error), "temporary_limit"
            temporary_retries += 1
            delay = retry_delay_seconds(error)
            print(
                f"Temporary Gemini rate limit; retry {temporary_retries}/"
                f"{MAX_TEMPORARY_RATE_LIMIT_RETRIES} in {delay} seconds."
            )
            time.sleep(delay)


def mean(values):
    return sum(values) / len(values) if values else None


def hit_for_row(row, relevant_ids):
    return any(doc_id in relevant_ids for doc_id in row["retrieved_corpus_ids"])


def record_key(row):
    return row["query_id"], row["condition"], row["k"]


def build_expected_combinations(queries):
    return [
        (query["query_id"], condition, k_value)
        for query in queries
        for condition in ("baseline", "reranked")
        for k_value in K_VALUES
    ]


def write_checkpoint(rows, expected_combinations, current=None):
    attempted_keys = {record_key(row) for row in rows}
    next_combination = next(
        (combination for combination in expected_combinations if combination not in attempted_keys),
        None,
    )
    checkpoint = {
        "total_combinations": len(expected_combinations),
        "attempted_combinations": len(attempted_keys),
        "successful_combinations": sum(row["api_success"] for row in rows),
        "failed_combinations": sum(not row["api_success"] for row in rows),
        "next_unattempted_combination": next_combination,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "current_condition_query_k": current,
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(checkpoint, file_handle, indent=2)
    return next_combination


def run_preflight():
    queries = load_jsonl(DEV_PATH)
    expected_combinations = build_expected_combinations(queries)
    existing_rows = load_jsonl(RAW_PATH) if os.path.exists(RAW_PATH) else []
    keys = [record_key(row) for row in existing_rows]
    unique_keys = set(keys)
    next_combination = next(
        (combination for combination in expected_combinations if combination not in unique_keys),
        None,
    )
    print("LOCAL PREFLIGHT")
    print(f"Existing JSONL records: {len(existing_rows)}")
    print(f"Expected total combinations: {len(expected_combinations)}")
    print(f"Successful records: {sum(row['api_success'] for row in existing_rows)}")
    print(f"Failed records: {sum(not row['api_success'] for row in existing_rows)}")
    print(f"Duplicates: {len(keys) - len(unique_keys)}")
    if next_combination is None:
        print("Next combination: none")
    else:
        query_id, condition, k_value = next_combination
        print(f"Next combination: {condition} / {query_id} / K={k_value}")


def calculate_metrics(rows):
    by_k = {}
    for k_value in K_VALUES:
        k_rows = [row for row in rows if row["k"] == k_value]
        by_condition = {}
        for condition in ("baseline", "reranked"):
            condition_rows = [row for row in k_rows if row["condition"] == condition]
            successful = [row for row in condition_rows if row["api_success"]]
            label_metrics = {}
            for label in ("SUPPORT", "CONTRADICT"):
                label_rows = [row for row in successful if row["ground_truth"] == label]
                label_metrics[label] = mean([
                    int(row["predicted_verdict"] == row["ground_truth"])
                    for row in label_rows
                ])
            by_condition[condition] = {
                "retrieval_hit_at_k": mean([int(row["retrieval_hit"]) for row in condition_rows]),
                "verdict_accuracy_among_successful_api_calls": mean([
                    int(row["predicted_verdict"] == row["ground_truth"])
                    for row in successful
                ]),
                "support_accuracy_among_successful_api_calls": label_metrics["SUPPORT"],
                "contradict_accuracy_among_successful_api_calls": label_metrics["CONTRADICT"],
                "insufficient_evidence_count": sum(
                    row["predicted_verdict"] == "INSUFFICIENT_EVIDENCE" for row in successful
                ),
                "successful_api_calls": len(successful),
                "failed_api_calls": len(condition_rows) - len(successful),
            }
        by_condition["comparison"] = {
            "verdict_accuracy_improvement_reranked_minus_baseline": (
                by_condition["reranked"]["verdict_accuracy_among_successful_api_calls"]
                - by_condition["baseline"]["verdict_accuracy_among_successful_api_calls"]
            ),
            "retrieval_improvement_reranked_minus_baseline": (
                by_condition["reranked"]["retrieval_hit_at_k"]
                - by_condition["baseline"]["retrieval_hit_at_k"]
            ),
        }
        by_k[str(k_value)] = by_condition
    return by_k


def main():
    from sentence_transformers import CrossEncoder
    from src.rag.generator import MODEL_NAME, generate_verification
    from src.retrieval import Retriever

    start = time.monotonic()
    queries = load_jsonl(DEV_PATH)
    corpus = {row["corpus_id"]: row for row in load_jsonl(CORPUS_PATH)}
    existing_rows = load_jsonl(RAW_PATH) if os.path.exists(RAW_PATH) else []
    completed_keys = {record_key(row) for row in existing_rows}
    expected_combinations = build_expected_combinations(queries)
    next_combination = next(
        (combination for combination in expected_combinations if combination not in completed_keys),
        None,
    )
    print("RESUME CHECK")
    print(f"Existing records: {len(existing_rows)}")
    print(f"Successful: {sum(row['api_success'] for row in existing_rows)}")
    print(f"Failed: {sum(not row['api_success'] for row in existing_rows)}")
    print(f"Duplicates: {len(existing_rows) - len(completed_keys)}")
    if next_combination is None:
        print("Next combination: none")
    else:
        condition, query_id, k_value = next_combination
        print(f"Next combination: {condition} / {query_id} / K={k_value}")
    retriever = Retriever(rebuild=False)
    reranker = CrossEncoder(RERANKER_NAME)
    rows = list(existing_rows)
    retries = 0
    failures = sum(not row["api_success"] for row in existing_rows)
    resume_api_calls = 0
    resume_failures = 0
    first_resumed_record = None
    last_attempted = None
    permanently_stopped = False
    case_query_ids = {"1320", "1019", "1370", "1185", "314"}

    with open(RAW_PATH, "a", encoding="utf-8") as raw_file:
        for query in queries:
            query_id = query["query_id"]
            claim = query["claim_text"]
            relevant_ids = set(query.get("relevant_corpus_ids", []))
            faiss_top10 = retriever.retrieve(claim, top_k=10)
            rerank_pairs = [(claim, corpus[doc["corpus_id"]]["text"]) for doc in faiss_top10]
            rerank_scores = reranker.predict(rerank_pairs)
            reranked_top10 = [
                {**doc, "reranker_score": float(score), "original_faiss_rank": rank}
                for rank, (doc, score) in enumerate(zip(faiss_top10, rerank_scores), start=1)
            ]
            reranked_top10.sort(key=lambda doc: doc["reranker_score"], reverse=True)

            for condition in ("baseline", "reranked"):
                for k_value in K_VALUES:
                    key = (query_id, condition, k_value)
                    if key in completed_keys:
                        continue
                    if first_resumed_record is None:
                        first_resumed_record = key
                    print("=" * 50)
                    print("RERANKED RAG EXPERIMENT")
                    print(f"Progress: {len(rows)} / {len(expected_combinations)}")
                    print(f"Successful: {sum(row['api_success'] for row in rows)}")
                    print(f"Failed: {sum(not row['api_success'] for row in rows)}")
                    print(f"Remaining: {len(expected_combinations) - len(rows)}")
                    print("Current:")
                    print(f"condition = {condition}")
                    print(f"query_id = {query_id}")
                    print(f"K = {k_value}")
                    print("=" * 50)
                    if condition == "baseline":
                        documents = retriever.retrieve(claim, top_k=k_value)
                    else:
                        documents = reranked_top10[:k_value]
                    generation, request_calls, request_retries, error_message, failure_type = generate_with_resilience(
                        generate_verification, claim, documents
                    )
                    retries += request_retries
                    resume_api_calls += request_calls
                    if failure_type in ("daily_quota", "temporary_limit"):
                        print("GEMINI DAILY QUOTA EXHAUSTED." if failure_type == "daily_quota" else "GEMINI RATE LIMIT RETRIES EXHAUSTED.")
                        print(f"Experiment paused at: condition={condition}, query_id={query_id}, K={k_value}")
                        write_checkpoint(rows, expected_combinations, key)
                        permanently_stopped = True
                        break
                    api_success = generation is not None
                    if not api_success:
                        resume_failures += 1
                        failures += 1
                    retrieved_ids = [doc["corpus_id"] for doc in documents]
                    record = {
                        "condition": condition,
                        "query_id": query_id,
                        "k": k_value,
                        "ground_truth": query["ground_truth_label"],
                        "predicted_verdict": generation["verdict"] if generation else None,
                        "explanation": generation["explanation"] if generation else f"API failure: {error_message}",
                        "retrieved_corpus_ids": retrieved_ids,
                        "retrieval_hit": any(doc_id in relevant_ids for doc_id in retrieved_ids),
                        "api_success": api_success,
                    }
                    if condition == "reranked":
                        record["reranker_scores"] = [doc["reranker_score"] for doc in documents]
                        record["original_faiss_ranks"] = [doc["original_faiss_rank"] for doc in documents]
                    else:
                        record["reranker_scores"] = None
                        record["original_faiss_ranks"] = list(range(1, len(documents) + 1))
                    raw_file.write(json.dumps(record) + "\n")
                    raw_file.flush()
                    rows.append(record)
                    completed_keys.add(key)
                    last_attempted = key
                    time.sleep(SUCCESS_DELAY_SECONDS) if api_success else None
                    if api_success:
                        print(f"Completed: {len(rows)} / {len(expected_combinations)}")
                        print(f"Remaining: {len(expected_combinations) - len(rows)}")
                if permanently_stopped:
                    break
            if permanently_stopped:
                break

    if not permanently_stopped:
        write_checkpoint(rows, expected_combinations)

    metrics_by_k = calculate_metrics(rows)
    metrics = {
        "model": MODEL_NAME,
        "reranker": RERANKER_NAME,
        "claims": len(queries),
        "k_values": list(K_VALUES),
        "conditions": ["baseline", "reranked"],
        "intended_api_calls": len(queries) * len(K_VALUES) * 2,
        "successful_api_calls": len(rows) - failures,
        "failed_api_calls": failures,
        "retries": retries,
        "runtime_seconds": time.monotonic() - start,
        "success_delay_seconds": SUCCESS_DELAY_SECONDS,
        "retry_wait_seconds": RETRY_WAIT_SECONDS,
        "by_k": metrics_by_k,
        "case_studies": {
            query_id: [row for row in rows if row["query_id"] == query_id]
            for query_id in sorted(case_query_ids)
        },
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    fields = [
        "condition", "query_id", "k", "ground_truth", "predicted_verdict",
        "explanation", "retrieved_corpus_ids", "retrieval_hit", "api_success",
        "reranker_scores", "original_faiss_ranks",
    ]
    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row[field]) if isinstance(row[field], list) else row[field] for field in fields})

    print(f"Claims: {len(queries)}")
    print("K values: [1, 3, 5, 10]")
    print("Conditions: baseline, reranked")
    print(f"Intended API calls: {len(queries) * len(K_VALUES) * 2}")
    print(f"Successful calls: {len(rows) - failures}")
    print(f"Failed calls: {failures}")
    print(f"Retries: {retries}")
    print(f"Runtime seconds: {metrics['runtime_seconds']:.2f}")
    for k_value in K_VALUES:
        result = metrics_by_k[str(k_value)]
        print(
            f"K={k_value}: baseline_hit={result['baseline']['retrieval_hit_at_k']}, "
            f"reranked_hit={result['reranked']['retrieval_hit_at_k']}, "
            f"baseline_accuracy={result['baseline']['verdict_accuracy_among_successful_api_calls']}, "
            f"reranked_accuracy={result['reranked']['verdict_accuracy_among_successful_api_calls']}"
        )
    print(f"JSONL records: {len(rows)}")
    print(f"CSV rows: {len(rows)}")
    print("RESUME STATUS:")
    print(f"Existing records before resume: {len(existing_rows)}")
    print(f"New records generated: {len(rows) - len(existing_rows)}")
    print(f"Final total records: {len(rows)}")
    print(f"First resumed record: {first_resumed_record}")
    print(f"Last completed record: {len(rows)}")
    print(f"API calls made during this resume: {resume_api_calls}")
    print(f"Failed calls: {resume_failures}")
    print(f"Full experiment complete: {len(rows) == len(queries) * len(K_VALUES) * 2}")


if __name__ == "__main__":
    if "--preflight" in sys.argv[1:]:
        run_preflight()
    else:
        main()
