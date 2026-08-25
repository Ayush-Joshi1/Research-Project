"""Offline comparison of original and diverse evidence selection."""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.rag.diverse_evidence_extractor import extract_diverse_evidence
from src.rag.evidence_extractor import extract_evidence
from src.retrieval import Retriever

DEV_PATH = os.path.join(ROOT, "data", "processed", "dev_queries.jsonl")
JSON_PATH = os.path.join(ROOT, "results", "evidence_extractor_comparison.json")
TXT_PATH = os.path.join(ROOT, "results", "evidence_extractor_comparison.txt")
QUERY_IDS = {"118", "1019", "1320", "1370", "1185"}
K_VALUES = (3, 5)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(evidence, documents, relevant_ids):
    similarities = [item["similarity"] for item in evidence]
    return {
        "number_of_evidence_sentences": len(evidence),
        "number_of_unique_documents_represented": len({item["corpus_id"] for item in evidence}),
        "evidence_contains_relevant_doc": bool({item["corpus_id"] for item in evidence}.intersection(relevant_ids)),
        "maximum_similarity": max(similarities) if similarities else None,
        "average_similarity": sum(similarities) / len(similarities) if similarities else None,
        "selected_corpus_ids": [item["corpus_id"] for item in evidence],
        "selected_evidence": evidence,
        "number_of_retrieved_documents": len(documents),
    }


def main():
    queries = [query for query in load_jsonl(DEV_PATH) if query["query_id"] in QUERY_IDS]
    query_by_id = {query["query_id"]: query for query in queries}
    retriever = Retriever(rebuild=False)
    records = []
    for query in queries:
        relevant_ids = set(query.get("relevant_corpus_ids", []))
        for k_value in K_VALUES:
            documents = retriever.retrieve(query["claim_text"], top_k=k_value)
            original = extract_evidence(query["claim_text"], documents)
            diverse = extract_diverse_evidence(query["claim_text"], documents)
            records.append({
                "query_id": query["query_id"],
                "k": k_value,
                "ground_truth": query["ground_truth_label"],
                "retrieved_corpus_ids": [document["corpus_id"] for document in documents],
                "original": summarize(original, documents, relevant_ids),
                "diverse": summarize(diverse, documents, relevant_ids),
            })

    aggregate = {}
    for name in ("original", "diverse"):
        aggregate[name] = {}
        for k_value in K_VALUES:
            selected = [record[name] for record in records if record["k"] == k_value]
            represented = sum(item["evidence_contains_relevant_doc"] for item in selected)
            aggregate[name][str(k_value)] = {
                "records": len(selected),
                "relevant_document_representation_count": represented,
                "representation_rate": represented / len(selected) if selected else None,
                "average_evidence_sentences": sum(item["number_of_evidence_sentences"] for item in selected) / len(selected),
                "average_unique_documents_represented": sum(item["number_of_unique_documents_represented"] for item in selected) / len(selected),
            }

    output = {
        "scope": "Offline FAISS retrieval and local evidence-selection comparison only.",
        "queries": sorted(query_by_id),
        "K_values": list(K_VALUES),
        "aggregate_by_extractor_and_K": aggregate,
        "records": records,
        "api_calls": 0,
        "retrieval_calls": len(records),
        "reranking_calls": 0,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    lines = [
        "EVIDENCE EXTRACTOR COMPARISON",
        "Offline comparison only. No Gemini calls, external API calls, or CrossEncoder reranking.",
        "",
        "AGGREGATE RESULTS",
    ]
    for k_value in K_VALUES:
        lines.append(f"K={k_value}")
        for name in ("original", "diverse"):
            stats = aggregate[name][str(k_value)]
            lines.append(
                f"  {name}: relevant-document representation {stats['relevant_document_representation_count']}/{stats['records']} "
                f"({stats['representation_rate']:.4f}); average evidence sentences {stats['average_evidence_sentences']:.2f}; "
                f"average unique documents {stats['average_unique_documents_represented']:.2f}"
            )
        lines.append("")
    lines.append("QUERY/K COMPARISONS")
    for record in records:
        lines.extend([
            "",
            f"QUERY ID: {record['query_id']}",
            f"K: {record['k']}",
            f"GROUND TRUTH: {record['ground_truth']}",
            f"Retrieved corpus IDs: {json.dumps(record['retrieved_corpus_ids'])}",
        ])
        for name in ("original", "diverse"):
            item = record[name]
            lines.extend([
                f"{name.upper()}:",
                f"  Evidence sentences: {item['number_of_evidence_sentences']}",
                f"  Unique documents: {item['number_of_unique_documents_represented']}",
                f"  Contains relevant document: {item['evidence_contains_relevant_doc']}",
                f"  Maximum similarity: {item['maximum_similarity']:.4f}",
                f"  Average similarity: {item['average_similarity']:.4f}",
                f"  Selected corpus IDs: {json.dumps(item['selected_corpus_ids'])}",
            ])
            for evidence in item["selected_evidence"]:
                lines.append(
                    f"    {evidence['corpus_id']} rank={evidence['document_rank']} "
                    f"similarity={evidence['similarity']:.4f}: {evidence['sentence']}"
                )

    lines.extend([
        "",
        "INTERPRETATION",
        "This is an offline evidence-selection diagnostic. Representation of a known relevant document does not establish that the selected sentence is correct evidence, and these results do not claim that the diverse extractor improves RAG.",
    ])
    with open(TXT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"aggregate": aggregate, "records": len(records), "api_calls": 0, "retrieval_calls": len(records), "reranking_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
