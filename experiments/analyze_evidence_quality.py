"""Offline diagnostic analysis of evidence-aware sanity outputs."""
import json
import os
from collections import Counter
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SANITY_PATH = os.path.join(ROOT, "results", "evidence_rag_sanity.jsonl")
DEV_PATH = os.path.join(ROOT, "data", "processed", "dev_queries.jsonl")
JSON_PATH = os.path.join(ROOT, "results", "evidence_quality_analysis.json")
TXT_PATH = os.path.join(ROOT, "results", "evidence_quality_analysis.txt")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def analyze_record(record, relevant_ids):
    evidence = record.get("evidence_sentences", [])
    similarities = [float(item["similarity"]) for item in evidence]
    evidence_doc_counts = Counter(item["corpus_id"] for item in evidence)
    selected_ids = set(evidence_doc_counts)
    retrieved_ids = set(record.get("retrieved_corpus_ids", []))
    relevant_retrieved = retrieved_ids.intersection(relevant_ids)
    represented_relevant = selected_ids.intersection(relevant_ids)
    relevant_scores = [item["similarity"] for item in evidence if item["corpus_id"] in relevant_ids]
    non_relevant_scores = [item["similarity"] for item in evidence if item["corpus_id"] not in relevant_ids]
    non_relevant_outranks_relevant = bool(relevant_scores and non_relevant_scores and max(non_relevant_scores) > max(relevant_scores))
    return {
        "query_id": str(record["query_id"]),
        "k": int(record["k"]),
        "ground_truth": record["ground_truth"],
        "retrieved_corpus_ids": record.get("retrieved_corpus_ids", []),
        "number_of_retrieved_documents": len(record.get("retrieved_corpus_ids", [])),
        "number_of_extracted_evidence_sentences": len(evidence),
        "average_sentence_similarity": sum(similarities) / len(similarities) if similarities else None,
        "maximum_sentence_similarity": max(similarities) if similarities else None,
        "minimum_sentence_similarity": min(similarities) if similarities else None,
        "number_of_unique_documents_represented": len(selected_ids),
        "evidence_sentences_per_document": dict(evidence_doc_counts),
        "relevant_corpus_ids": sorted(relevant_ids),
        "relevant_document_retrieved": bool(relevant_retrieved),
        "evidence_contains_relevant_doc": bool(represented_relevant),
        "relevant_documents_retrieved_but_no_sentence_selected": bool(relevant_retrieved) and not bool(represented_relevant),
        "relevant_document_represented": bool(represented_relevant),
        "highest_non_relevant_sentence_outranks_all_relevant_sentences": non_relevant_outranks_relevant,
        "evidence_from_only_one_document": len(selected_ids) == 1,
        "selected_evidence": evidence,
    }


def aggregate(records):
    similarities = [record["maximum_sentence_similarity"] for record in records if record["maximum_sentence_similarity"] is not None]
    evidence_counts = [record["number_of_extracted_evidence_sentences"] for record in records]
    represented_count = sum(record["evidence_contains_relevant_doc"] for record in records)
    return {
        "record_count": len(records),
        "average_evidence_sentences_per_record": sum(evidence_counts) / len(evidence_counts) if evidence_counts else None,
        "average_maximum_similarity": sum(similarities) / len(similarities) if similarities else None,
        "records_where_evidence_contains_known_relevant_document": represented_count,
        "proportion_where_evidence_contains_known_relevant_document": represented_count / len(records) if records else None,
    }


def format_evidence(evidence):
    lines = []
    for item in evidence:
        lines.extend([
            f"    Document ID: {item['corpus_id']}",
            f"    Document rank: {item['document_rank']}",
            f"    Similarity: {float(item['similarity']):.4f}",
            f"    Sentence: {item['sentence']}",
        ])
    return lines


def main():
    records = [row for row in load_jsonl(SANITY_PATH) if row.get("condition") == "evidence_aware"]
    queries = {str(row["query_id"]): row for row in load_jsonl(DEV_PATH)}
    analyzed = []
    for record in records:
        query = queries[str(record["query_id"])]
        analyzed.append(analyze_record(record, set(query.get("relevant_corpus_ids", []))))

    by_k = {}
    for k_value in (3, 5):
        by_k[str(k_value)] = aggregate([record for record in analyzed if record["k"] == k_value])

    interesting = {
        "A_relevant_retrieved_but_no_sentence_selected": [record for record in analyzed if record["relevant_documents_retrieved_but_no_sentence_selected"]],
        "B_relevant_document_represented": [record for record in analyzed if record["relevant_document_represented"]],
        "C_non_relevant_sentence_outranks_relevant_sentences": [record for record in analyzed if record["highest_non_relevant_sentence_outranks_all_relevant_sentences"]],
        "D_evidence_from_only_one_document": [record for record in analyzed if record["evidence_from_only_one_document"]],
    }
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_files": [SANITY_PATH, DEV_PATH],
        "interpretation_note": "evidence_contains_relevant_doc checks document-level representation only; it does not establish that any selected sentence is correct evidence.",
        "aggregate": aggregate(analyzed),
        "by_K": by_k,
        "records": analyzed,
        "interesting_cases": interesting,
        "api_calls_made": 0,
        "retrieval_calls_made": 0,
        "reranking_calls_made": 0,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    lines = [
        "EVIDENCE QUALITY ANALYSIS",
        "Offline analysis of stored evidence-aware records only.",
        "No API, retrieval, or reranking calls were made.",
        "",
        "Interpretation: evidence_contains_relevant_doc means at least one selected sentence came from a known relevant document. It does not mean that the sentence itself is relevant or correct evidence.",
        "",
        "AGGREGATE STATISTICS",
    ]
    for label, stats in (("ALL", output["aggregate"]), ("K=3", by_k["3"]), ("K=5", by_k["5"])):
        lines.extend([
            label,
            f"Total evidence-aware records: {stats['record_count']}",
            f"Average evidence sentences per record: {stats['average_evidence_sentences_per_record']:.4f}",
            f"Average maximum similarity: {stats['average_maximum_similarity']:.4f}",
            f"Evidence contains known relevant document: {stats['records_where_evidence_contains_known_relevant_document']}",
            f"Proportion containing known relevant document: {stats['proportion_where_evidence_contains_known_relevant_document']:.4f}",
            "",
        ])
    lines.append("PER-RECORD EVIDENCE")
    for record in analyzed:
        lines.extend([
            "",
            f"QUERY ID: {record['query_id']}",
            f"K: {record['k']}",
            f"GROUND TRUTH: {record['ground_truth']}",
            f"Retrieved documents: {record['number_of_retrieved_documents']}",
            f"Extracted evidence sentences: {record['number_of_extracted_evidence_sentences']}",
            f"Average similarity: {record['average_sentence_similarity']:.4f}",
            f"Maximum similarity: {record['maximum_sentence_similarity']:.4f}",
            f"Minimum similarity: {record['minimum_sentence_similarity']:.4f}",
            f"Unique evidence documents: {record['number_of_unique_documents_represented']}",
            f"Evidence sentences per document: {json.dumps(record['evidence_sentences_per_document'])}",
            f"Evidence contains relevant document: {record['evidence_contains_relevant_doc']}",
            "Evidence:",
        ])
        lines.extend(format_evidence(record["selected_evidence"]))

    lines.extend(["", "INTERESTING CASES"])
    labels = {
        "A_relevant_retrieved_but_no_sentence_selected": "A. Relevant document retrieved but no sentence selected",
        "B_relevant_document_represented": "B. Relevant document represented in selected evidence",
        "C_non_relevant_sentence_outranks_relevant_sentences": "C. Highest-similarity non-relevant sentence outranks relevant sentences",
        "D_evidence_from_only_one_document": "D. Evidence selected from only one document",
    }
    for category, label in labels.items():
        cases = interesting[category]
        lines.extend(["", label, f"Count: {len(cases)}"])
        for case in cases:
            lines.extend([
                f"  query_id={case['query_id']}, K={case['k']}, ground_truth={case['ground_truth']}",
                f"  relevant corpus IDs: {json.dumps(case['relevant_corpus_ids'])}",
                f"  similarity scores: {json.dumps([item['similarity'] for item in case['selected_evidence']])}",
                "  selected evidence:",
            ])
            lines.extend([f"    {item['corpus_id']} ({float(item['similarity']):.4f}): {item['sentence']}" for item in case["selected_evidence"]])

    with open(TXT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"aggregate": output["aggregate"], "by_K": by_k, "interesting_case_counts": {key: len(value) for key, value in interesting.items()}, "api_calls_made": 0, "retrieval_calls_made": 0, "reranking_calls_made": 0}, indent=2))


if __name__ == "__main__":
    main()
