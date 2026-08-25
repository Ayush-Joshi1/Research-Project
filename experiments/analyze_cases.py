"""Create a read-only case study from the existing clean RAG outputs."""
import json
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEV_PATH = os.path.join(ROOT, "data", "processed", "dev_queries.jsonl")
ERRORS_PATH = os.path.join(ROOT, "results", "error_analysis.jsonl")
JSON_PATH = os.path.join(ROOT, "results", "case_study_analysis.json")
TXT_PATH = os.path.join(ROOT, "results", "case_study_analysis.txt")
K_VALUES = (1, 3, 5, 10)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def endpoint_groups(rows):
    by_query = {}
    for row in rows:
        by_query.setdefault(row["query_id"], {})[int(row["k"])] = row
    groups = {"A": [], "B": [], "C": [], "D": []}
    for query_id, query_rows in by_query.items():
        start_correct = query_rows[1]["category"] == "CORRECT"
        end_correct = query_rows[10]["category"] == "CORRECT"
        group = "D" if start_correct and end_correct else "A" if end_correct else "B" if start_correct else "C"
        groups[group].append(query_id)
    return by_query, groups


def classify_change(query_rows):
    first = query_rows[1]
    last = query_rows[10]
    if first["category"] != "CORRECT" and last["category"] == "CORRECT":
        if not first["retrieval_hit"] and last["retrieval_hit"]:
            return "RETRIEVAL_IMPROVEMENT", "A relevant document appears at K=10 but not at K=1."
        if first["predicted_verdict"] == "INSUFFICIENT_EVIDENCE":
            return "EVIDENCE_ACCUMULATION", "The prediction changes from insufficient evidence to the correct label as K increases."
        if first["retrieval_hit"]:
            return "EVIDENCE_ACCUMULATION", "Relevant evidence is already retrieved at K=1 and the later correct verdict coincides with additional context."
        return "OTHER", "The endpoint verdict changes, but neither the retrieval-hit nor prediction pattern matches the preferred categories."
    if first["category"] == "CORRECT" and last["category"] != "CORRECT":
        return "CONTEXT_REGRESSION", "The prediction changes from correct at K=1 to incorrect at K=10."
    if first["category"] != "CORRECT" and last["category"] != "CORRECT":
        return "PERSISTENT_FAILURE", "The endpoint verdict remains incorrect at K=1 and K=10."
    return "OTHER", "The endpoint verdict remains correct; this case is included as a stable comparison."


def case_priority(query_rows):
    first = query_rows[1]
    last = query_rows[10]
    if not first["retrieval_hit"] and last["retrieval_hit"]:
        return 0
    if first["predicted_verdict"] == "INSUFFICIENT_EVIDENCE" and last["category"] == "CORRECT":
        return 1
    if first["retrieval_hit"] and last["category"] == "CORRECT":
        return 2
    return 3


def make_case(query_rows, claims):
    first = query_rows[1]
    classification, why = classify_change(query_rows)
    return {
        "query_id": first["query_id"],
        "claim": claims[first["query_id"]],
        "ground_truth_label": first["ground_truth_label"],
        "classification": classification,
        "why_classification_was_assigned": why,
        "results_by_k": {
            str(k): {
                "prediction": query_rows[k]["predicted_verdict"],
                "retrieval_hit": query_rows[k]["retrieval_hit"],
                "retrieved_corpus_ids": query_rows[k]["retrieved_corpus_ids"],
                "relevant_corpus_ids": query_rows[k]["relevant_corpus_ids"] if k == 1 else None,
                "explanation": query_rows[k]["explanation"],
            }
            for k in K_VALUES
        },
    }


def write_text(report, groups, counts):
    lines = [
        "DAY 2 STEP 3: CASE STUDY ANALYSIS",
        "",
        "This report describes existing retrieval and model outputs. It does not establish causality.",
        "",
        "ENDPOINT COUNTS",
        f"K=1 incorrect -> K=10 correct: {counts['incorrect_to_correct']}",
        f"K=1 correct -> K=10 incorrect: {counts['correct_to_incorrect']}",
        f"Always incorrect: {counts['always_incorrect']}",
        f"Always correct: {counts['always_correct']}",
        "",
        "CASE STUDIES",
    ]
    for index, case in enumerate(report["selected_cases"], start=1):
        lines.extend([
            "",
            f"Case {index}: query {case['query_id']}",
            f"Claim: {case['claim']}",
            f"Ground truth: {case['ground_truth_label']}",
            f"Classification: {case['classification']}",
            f"Why: {case['why_classification_was_assigned']}",
        ])
        for k in K_VALUES:
            result = case["results_by_k"][str(k)]
            lines.extend([
                f"K={k}: prediction={result['prediction']}; retrieval_hit={result['retrieval_hit']}",
                f"Retrieved IDs: {result['retrieved_corpus_ids']}",
                f"Explanation: {result['explanation']}",
            ])
    lines.extend(["", "SUMMARY BY CLASSIFICATION"])
    for name, value in report["summary_by_classification"].items():
        lines.append(f"{name}: {value}")
    with open(TXT_PATH, "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(lines) + "\n")


def main():
    claims = {row["query_id"]: row["claim_text"] for row in load_jsonl(DEV_PATH)}
    rows = load_jsonl(ERRORS_PATH)
    by_query, groups = endpoint_groups(rows)
    counts = {
        "incorrect_to_correct": len(groups["A"]),
        "correct_to_incorrect": len(groups["B"]),
        "always_incorrect": len(groups["C"]),
        "always_correct": len(groups["D"]),
    }
    selected_ids = sorted(groups["A"], key=lambda query_id: (case_priority(by_query[query_id]), query_id))[:5]
    comparison_ids = sorted(
        groups["B"] + groups["D"],
        key=lambda query_id: (case_priority(by_query[query_id]), query_id),
    )[:3]
    selected = [
        make_case(by_query[query_id], claims)
        for query_id in selected_ids + comparison_ids
    ]
    summary = {name: sum(case["classification"] == name for case in selected) for name in (
        "RETRIEVAL_IMPROVEMENT", "EVIDENCE_ACCUMULATION", "PERSISTENT_FAILURE", "CONTEXT_REGRESSION", "OTHER"
    )}
    report = {
        "records_inspected": len(rows),
        "queries_inspected": len(by_query),
        "api_calls": 0,
        "endpoint_counts": counts,
        "selected_cases": selected,
        "summary_by_classification": summary,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2)
    write_text(report, groups, counts)
    print(json.dumps({"records_inspected": len(rows), "endpoint_counts": counts, "selected_cases": len(selected), "summary_by_classification": summary, "api_calls": 0}, indent=2))
    print(f"Created: {JSON_PATH}")
    print(f"Created: {TXT_PATH}")


if __name__ == "__main__":
    main()
