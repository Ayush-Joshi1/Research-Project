"""Run a five-query, FAISS-only evidence-aware generation sanity test."""
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.rag.evidence_extractor import extract_evidence
from src.rag.evidence_generator import generate_evidence_verification
from src.rag.generator import generate_verification
from src.retrieval import Retriever

DEV_PATH = os.path.join(ROOT, "data", "processed", "dev_queries.jsonl")
OUTPUT_PATH = os.path.join(ROOT, "results", "evidence_rag_sanity.jsonl")
QUERY_IDS = {"118", "1019", "1320", "1370", "1185"}
K_VALUES = (3, 5)
SUCCESS_DELAY_SECONDS = 5
RETRY_WAIT_SECONDS = 60


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def is_rate_limit_error(error):
    message = str(error).lower()
    return "429" in message or "rate limit" in message or "resource exhausted" in message


def generate_with_one_retry(generator, claim, documents_or_evidence):
    try:
        return generator(claim, documents_or_evidence), 0
    except Exception as error:
        if not is_rate_limit_error(error):
            return None, 0
        time.sleep(RETRY_WAIT_SECONDS)
        try:
            return generator(claim, documents_or_evidence), 1
        except Exception:
            return None, 1


def make_record(query, k_value, condition, documents, evidence, generation):
    return {
        "query_id": query["query_id"],
        "k": k_value,
        "ground_truth": query["ground_truth_label"],
        "condition": condition,
        "predicted_verdict": generation["verdict"] if generation else None,
        "explanation": generation["explanation"] if generation else "API failure",
        "retrieved_corpus_ids": [document["corpus_id"] for document in documents],
        "evidence_sentences": evidence,
        "api_success": generation is not None,
    }


def main():
    queries = [query for query in load_jsonl(DEV_PATH) if query["query_id"] in QUERY_IDS]
    retriever = Retriever(rebuild=False)
    api_calls = 0
    retries = 0
    records = []

    with open(OUTPUT_PATH, "x", encoding="utf-8") as output_file:
        for query in queries:
            claim = query["claim_text"]
            for k_value in K_VALUES:
                documents = retriever.retrieve(claim, top_k=k_value)
                evidence = extract_evidence(claim, documents)
                for condition, generator, input_value in (
                    ("baseline", generate_verification, documents),
                    ("evidence_aware", generate_evidence_verification, evidence),
                ):
                    generation, request_retries = generate_with_one_retry(generator, claim, input_value)
                    api_calls += 1 + request_retries
                    retries += request_retries
                    record = make_record(query, k_value, condition, documents, evidence, generation)
                    records.append(record)
                    output_file.write(json.dumps(record) + "\n")
                    output_file.flush()
                    if condition == "baseline":
                        print(f"Query ID: {query['query_id']}")
                        print(f"K: {k_value}")
                        print(f"Ground truth: {query['ground_truth_label']}")
                        print(f"Baseline prediction: {record['predicted_verdict']}")
                        print(f"Baseline explanation: {record['explanation']}")
                    if condition == "evidence_aware":
                        print(f"Evidence-aware prediction: {record['predicted_verdict']}")
                        print(f"Evidence-aware explanation: {record['explanation']}")
                    print(f"Number of documents: {len(documents)}")
                    print(f"Number of extracted evidence sentences: {len(evidence)}")
                    print()
                    if generation is not None:
                        time.sleep(SUCCESS_DELAY_SECONDS)

    print(f"Records written: {len(records)}")
    print(f"API calls: {api_calls}")
    print(f"Retries: {retries}")
    print(f"Successful calls: {sum(record['api_success'] for record in records)}")
    print(f"Failed calls: {sum(not record['api_success'] for record in records)}")


if __name__ == "__main__":
    main()
