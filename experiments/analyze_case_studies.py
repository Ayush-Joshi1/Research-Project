"""Read-only evidence-level analysis for selected reranked RAG case studies."""
import json
import os
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
RAW_PATH = os.path.join(RESULTS_DIR, "reranked_rag_results.jsonl")
CASE_SOURCE_PATH = os.path.join(RESULTS_DIR, "reranked_rag_case_studies.json")
ANALYSIS_SOURCE_PATH = os.path.join(RESULTS_DIR, "reranked_rag_analysis.json")
DEV_PATH = os.path.join(ROOT, "data", "processed", "dev_queries.jsonl")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "final_case_study_analysis.json")
OUTPUT_TXT = os.path.join(RESULTS_DIR, "final_case_study_analysis.txt")
QUERY_IDS = ("1320", "1019", "1370", "1185", "314")
K_VALUES = (1, 3, 5, 10)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row):
    return row["condition"], str(row["query_id"]), int(row["k"])


def correct(row):
    return row["api_success"] and row["predicted_verdict"] == row["ground_truth"]


def verdict_status(baseline, reranked):
    if not baseline["api_success"] or not reranked["api_success"]:
        return "not_comparable_failed_api"
    baseline_correct = correct(baseline)
    reranked_correct = correct(reranked)
    if reranked_correct and not baseline_correct:
        return "verdict_improved"
    if baseline_correct and not reranked_correct:
        return "verdict_worsened"
    return "verdict_unchanged"


def comparison_category(baseline, reranked):
    verdict = verdict_status(baseline, reranked)
    retrieval_improved = reranked["retrieval_hit"] and not baseline["retrieval_hit"]
    retrieval_unchanged = reranked["retrieval_hit"] == baseline["retrieval_hit"]
    if verdict == "not_comparable_failed_api":
        return "not_comparable_failed_api"
    if retrieval_improved:
        return {
            "verdict_improved": "retrieval_improved_and_verdict_improved",
            "verdict_unchanged": "retrieval_improved_and_verdict_unchanged",
            "verdict_worsened": "retrieval_improved_and_verdict_worsened",
        }[verdict]
    if retrieval_unchanged:
        return {
            "verdict_improved": "retrieval_unchanged_and_verdict_improved",
            "verdict_unchanged": "retrieval_unchanged_and_verdict_unchanged",
            "verdict_worsened": "retrieval_unchanged_and_verdict_worsened",
        }[verdict]
    return "retrieval_status_changed_other_direction"


def rank_map(row):
    if row["condition"] == "reranked":
        return dict(zip(row["retrieved_corpus_ids"], row["original_faiss_ranks"]))
    return dict(zip(row["retrieved_corpus_ids"], range(1, len(row["retrieved_corpus_ids"]) + 1)))


def evidence_comparison(baseline, reranked):
    baseline_ids = baseline["retrieved_corpus_ids"]
    reranked_ids = reranked["retrieved_corpus_ids"]
    baseline_set = set(baseline_ids)
    reranked_set = set(reranked_ids)
    baseline_ranks = rank_map(baseline)
    reranked_ranks = rank_map(reranked)
    moved_earlier = [
        {"corpus_id": corpus_id, "baseline_rank": baseline_ranks[corpus_id], "reranked_rank": reranked_ranks[corpus_id]}
        for corpus_id in reranked_ids
        if corpus_id in baseline_ranks and reranked_ranks[corpus_id] < baseline_ranks[corpus_id]
    ]
    return {
        "baseline_only_documents": [corpus_id for corpus_id in baseline_ids if corpus_id not in reranked_set],
        "reranked_only_documents": [corpus_id for corpus_id in reranked_ids if corpus_id not in baseline_set],
        "shared_documents": [corpus_id for corpus_id in reranked_ids if corpus_id in baseline_set],
        "documents_appearing_earlier_after_reranking": moved_earlier,
        "baseline_document_set": sorted(baseline_set),
        "reranked_document_set": sorted(reranked_set),
        "document_sets_equal": baseline_set == reranked_set,
    }


def observation_text(query_id, claim, ground_truth, entries):
    successful_changes = [entry for entry in entries if entry["verdict_status"] in ("verdict_improved", "verdict_worsened")]
    failed_ks = [entry["K"] for entry in entries if entry["verdict_status"] == "not_comparable_failed_api"]
    sentences = [f"Query {query_id} has ground truth {ground_truth}: {claim}"]
    if successful_changes:
        details = "; ".join(
            f"K={entry['K']} {entry['verdict_status'].replace('_', ' ')}"
            for entry in successful_changes
        )
        sentences.append(f"The stored successful paired outputs show {details}.")
    else:
        sentences.append("The stored successful paired outputs show no verdict change.")
    if failed_ks:
        sentences.append(f"Verdict comparison is unavailable at K={','.join(map(str, failed_ks))} because at least one API call failed; failed calls are not treated as incorrect verdicts.")
    changed_evidence = [entry for entry in entries if entry["verdict_changed"]]
    if changed_evidence:
        moved = sorted({item["corpus_id"] for entry in changed_evidence for item in entry["evidence_comparison"]["documents_appearing_earlier_after_reranking"]})
        if moved:
            sentences.append(f"For changed verdicts, the stored rank evidence shows these documents appearing earlier after reranking: {', '.join(moved)}.")
        sentences.append("Cause of verdict change cannot be established from the stored outputs.")
    return " ".join(sentences)


def main():
    rows = load_jsonl(RAW_PATH)
    with open(CASE_SOURCE_PATH, "r", encoding="utf-8") as handle:
        case_source = json.load(handle)
    with open(ANALYSIS_SOURCE_PATH, "r", encoding="utf-8") as handle:
        analysis_source = json.load(handle)
    queries = {str(row["query_id"]): row for row in load_jsonl(DEV_PATH)}
    first_by_key = {}
    for row in rows:
        first_by_key.setdefault(key(row), row)

    case_studies = {}
    findings = {}
    for query_id in QUERY_IDS:
        query = queries[query_id]
        entries = []
        for k_value in K_VALUES:
            baseline = first_by_key[("baseline", query_id, k_value)]
            reranked = first_by_key[("reranked", query_id, k_value)]
            evidence = evidence_comparison(baseline, reranked)
            changed = baseline["api_success"] and reranked["api_success"] and baseline["predicted_verdict"] != reranked["predicted_verdict"]
            entries.append({
                "query_id": query_id,
                "K": k_value,
                "ground_truth": baseline["ground_truth"],
                "baseline": {
                    "api_success": baseline["api_success"],
                    "prediction": baseline["predicted_verdict"],
                    "retrieval_hit": baseline["retrieval_hit"],
                    "retrieved_corpus_ids": baseline["retrieved_corpus_ids"],
                    "explanation": baseline["explanation"],
                },
                "reranked": {
                    "api_success": reranked["api_success"],
                    "prediction": reranked["predicted_verdict"],
                    "retrieval_hit": reranked["retrieval_hit"],
                    "retrieved_corpus_ids": reranked["retrieved_corpus_ids"],
                    "reranker_scores": reranked["reranker_scores"],
                    "original_faiss_ranks": reranked["original_faiss_ranks"],
                    "explanation": reranked["explanation"],
                },
                "verdict_changed": changed,
                "verdict_status": verdict_status(baseline, reranked),
                "comparison_category": comparison_category(baseline, reranked),
                "retrieval_status": "changed" if baseline["retrieval_hit"] != reranked["retrieval_hit"] else "unchanged",
                "exact_verdict_change_K": k_value if changed else None,
                "evidence_comparison": evidence,
                "cause_statement": "Cause of verdict change cannot be established from the stored outputs." if changed else None,
            })
        case_studies[query_id] = {"claim_text": query["claim_text"], "ground_truth": query["ground_truth_label"], "K": entries}
        findings[query_id] = observation_text(query_id, query["claim_text"], query["ground_truth_label"], entries)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_files": [RAW_PATH, CASE_SOURCE_PATH, ANALYSIS_SOURCE_PATH, DEV_PATH],
        "analysis_scope": "Stored outputs only; no API, retrieval, or reranking calls.",
        "source_validation": {
            "raw_records_read": len(rows),
            "unique_raw_combinations": len({key(row) for row in rows}),
            "existing_analysis_validation": analysis_source.get("validation"),
            "case_source_query_ids": sorted(case_source),
        },
        "case_studies": case_studies,
        "case_study_findings": findings,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    lines = [
        "FINAL CASE STUDY ANALYSIS",
        "",
        "Source: stored experiment outputs only. No API, retrieval, or reranking calls were made.",
        "",
    ]
    for query_id in QUERY_IDS:
        study = case_studies[query_id]
        lines.extend([f"QUERY {query_id}", f"Claim: {study['claim_text']}", f"Ground truth: {study['ground_truth']}", ""])
        for entry in study["K"]:
            lines.extend([
                f"K={entry['K']}",
                "BASELINE:",
                f"  API success: {entry['baseline']['api_success']}",
                f"  Prediction: {entry['baseline']['prediction']}",
                f"  Retrieval hit: {entry['baseline']['retrieval_hit']}",
                f"  Retrieved corpus IDs: {json.dumps(entry['baseline']['retrieved_corpus_ids'])}",
                f"  Explanation: {entry['baseline']['explanation']}",
                "RERANKED:",
                f"  API success: {entry['reranked']['api_success']}",
                f"  Prediction: {entry['reranked']['prediction']}",
                f"  Retrieval hit: {entry['reranked']['retrieval_hit']}",
                f"  Retrieved corpus IDs: {json.dumps(entry['reranked']['retrieved_corpus_ids'])}",
                f"  Reranker scores: {json.dumps(entry['reranked']['reranker_scores'])}",
                f"  Original FAISS ranks: {json.dumps(entry['reranked']['original_faiss_ranks'])}",
                f"  Explanation: {entry['reranked']['explanation']}",
                f"Comparison category: {entry['comparison_category']}",
                f"Verdict status: {entry['verdict_status']}",
                f"Retrieval status: {entry['retrieval_status']}",
                f"Evidence comparison: {json.dumps(entry['evidence_comparison'])}",
                f"Exact verdict change K: {entry['exact_verdict_change_K']}",
                f"Cause statement: {entry['cause_statement']}",
                "",
            ])
        lines.extend(["CASE STUDY FINDINGS", findings[query_id], "", "=" * 72, ""])
    with open(OUTPUT_TXT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
