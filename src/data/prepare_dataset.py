"""Prepare BEIR SciFact dataset aligned with original SciFact annotations.

This script:
- Loads BEIR SciFact `corpus`, `queries`, and the `test` qrels.
- Loads original SciFact claim annotations from the Hugging Face `scifact` dataset.
- Aligns BEIR test queries with original SciFact claims by id or exact claim text.
- Builds a dev subset of exactly 50 queries (seed=42) preferring queries with at
  least one relevant document and known SUPPORT/CONTRADICT label.
- Writes JSONL outputs to `data/processed/` and a metadata JSON file.

Constraints: no embeddings, FAISS, LLM calls, or API keys.
"""
from __future__ import annotations

import json
import os
import random
import itertools
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from datasets import DatasetDict, load_dataset


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)


def save_jsonl(path: str, records: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_text(s: str) -> str:
    return " ".join(s.strip().split()).lower()


def main(*, seed: int = 42, dev_k: int = 50):
    random.seed(seed)

    # 1-3. Load BEIR SciFact corpus, queries, and qrels
    print("Loading BEIR SciFact corpus and queries...")
    beir_corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    beir_queries = load_dataset("BeIR/scifact", "queries", split="queries")

    print("Loading BEIR SciFact qrels (all splits)...")
    beir_qrels_all = load_dataset("BeIR/scifact-qrels")

    # Prefer the 'test' split when present
    qrels_test = None
    if isinstance(beir_qrels_all, DatasetDict) and "test" in beir_qrels_all:
        qrels_test = beir_qrels_all["test"]
    else:
        if isinstance(beir_qrels_all, DatasetDict):
            parts = []
            for k in beir_qrels_all.keys():
                parts.extend(list(beir_qrels_all[k]))
            qrels_test = parts
        else:
            qrels_test = list(beir_qrels_all)

    # 4. Load original SciFact annotations from Hugging Face. Try several ids.
    def try_load_original_scifact() -> List[Dict]:
        candidates = ["scifact", "ai2/scifact", "allenai/scifact", "ai2_scifact"]
        last_exc = None
        for cid in candidates:
            try:
                print(f"Trying original SciFact dataset id: {cid}")
                ds = load_dataset(cid)
                print(f"Loaded original SciFact candidate: {cid}")
                # Normalize to a list of examples (concatenate splits if needed)
                exs = []
                if isinstance(ds, DatasetDict):
                    for k in ds.keys():
                        exs.extend(list(ds[k]))
                else:
                    exs = list(ds)
                return exs
            except Exception as e:
                print(f"Could not load {cid}: {e}")
                last_exc = e
        print("ERROR: Could not load original SciFact annotations from known candidates.")
        raise RuntimeError("Original SciFact dataset unavailable: " + str(last_exc))

    print("Loading original SciFact annotations (trying known dataset ids)...")
    try:
        scifact_examples = try_load_original_scifact()
    except Exception:
        # Fallback: download official release tarball from scifact S3 and extract claim files
        print("Falling back to downloading official SciFact release tarball...")
        import urllib.request
        import tarfile
        raw_dir = os.path.join(OUT_DIR, "raw_scifact")
        os.makedirs(raw_dir, exist_ok=True)
        tar_url = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
        tar_path = os.path.join(raw_dir, "data.tar.gz")
        try:
            print(f"Downloading {tar_url} ...")
            urllib.request.urlretrieve(tar_url, tar_path)
            print("Download complete, extracting...")
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(path=raw_dir)
            # Look for claims_train.jsonl, claims_dev.jsonl, claims_test.jsonl
            scifact_examples = []
            for fname in ("claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl"):
                fp = os.path.join(raw_dir, "data", fname)
                if os.path.exists(fp):
                    with open(fp, "r", encoding="utf-8") as fh:
                        for line in fh:
                            try:
                                scifact_examples.append(json.loads(line))
                            except Exception:
                                continue
            if not scifact_examples:
                raise RuntimeError("Downloaded SciFact archive but no claims files found")
        except Exception as e:
            raise RuntimeError("Failed to obtain original SciFact annotations: " + str(e))

    # Build mapping for original scifact claims by id and normalized claim text
    # Extract claim-level ground truth from nested `evidence` structure per official schema.
    scifact_by_id: Dict[str, Dict] = {}
    scifact_by_text: Dict[str, Dict] = {}
    scifact_count = 0
    support_count = 0
    contradict_count = 0
    ambiguous_count = 0
    unlabeled_count = 0
    for ex in scifact_examples:
        scifact_count += 1
        ex_id = ex.get("id") or ex.get("_id")
        claim = ex.get("claim") or ex.get("text")
        if claim is None:
            continue

        # evidence is a mapping: {doc_id: [ {"label":..., "sentences": [...]}, ... ]}
        evidence = ex.get("evidence") or {}
        gold_doc_ids = []
        labels_found = set()
        if isinstance(evidence, dict):
            for dock, evid_list in evidence.items():
                # dock may be string or number; ensure string
                gold_doc_ids.append(str(dock))
                if isinstance(evid_list, list):
                    for item in evid_list:
                        lab = item.get("label") if isinstance(item, dict) else None
                        if lab is None:
                            continue
                        lab_u = str(lab).strip().upper()
                        if lab_u.startswith("SUP"):
                            labels_found.add("SUPPORT")
                        elif lab_u.startswith("CON") or lab_u.startswith("REF"):
                            labels_found.add("CONTRADICT")
                        else:
                            labels_found.add(lab_u)

        # Determine claim-level label
        if not labels_found:
            claim_label = None
            unlabeled_count += 1
        elif labels_found == {"SUPPORT"}:
            claim_label = "SUPPORT"
            support_count += 1
        elif labels_found == {"CONTRADICT"}:
            claim_label = "CONTRADICT"
            contradict_count += 1
        else:
            # conflicting labels
            claim_label = "AMBIGUOUS"
            ambiguous_count += 1

        entry = {
            "claim": claim,
            "label": claim_label,
            "evidence_doc_ids": gold_doc_ids,
        }
        if ex_id is not None:
            scifact_by_id[str(ex_id)] = entry
        scifact_by_text[normalize_text(claim)] = entry

    # Build qrels mapping: query-id -> list of corpus-ids
    qrels_map: Dict[str, List[str]] = defaultdict(list)
    qrels_count = 0
    for r in qrels_test:
        qid = r.get("query-id") or r.get("query_id") or r.get("qid")
        cid = r.get("corpus-id") or r.get("corpus_id") or r.get("docid") or r.get("doc-id")
        if qid is None or cid is None:
            continue
        qid_s = str(qid)
        cid_s = str(cid)
        qrels_map[qid_s].append(cid_s)
        qrels_count += 1

    # Corpus lookup (store all docs as requested)
    corpus_map: Dict[str, Dict] = {}
    for d in beir_corpus:
        cid = str(d.get("_id"))
        corpus_map[cid] = {"corpus_id": cid, "title": d.get("title"), "text": d.get("text")}

    # Align BEIR test queries with original scifact claims
    beir_queries_list = list(beir_queries)
    test_qids = set(qrels_map.keys())

    aligned = []
    failed_align = []
    for q in beir_queries_list:
        qid = str(q.get("_id"))
        if qid not in test_qids:
            continue
        claim_text = q.get("text") or q.get("title") or ""
        sc_ex = scifact_by_id.get(qid)
        if sc_ex is None:
            sc_ex = scifact_by_text.get(normalize_text(claim_text))
        if sc_ex is None:
            failed_align.append({"query_id": qid, "claim_text": claim_text})
            continue
        # sc_ex contains 'label' (SUPPORT/CONTRADICT/AMBIGUOUS/None) and evidence_doc_ids
        label_norm = sc_ex.get("label")
        rel_docs = qrels_map.get(qid, [])
        # Preserve qrels as retrieval relevance; do not conflate with evidence_doc_ids
        aligned.append({
            "query_id": qid,
            "claim_text": sc_ex.get("claim", claim_text),
            "ground_truth_label": label_norm,
            "evidence_doc_ids": sc_ex.get("evidence_doc_ids", []),
            "relevant_corpus_ids": [str(x) for x in rel_docs],
            "relevant_document_count": len(set(rel_docs)),
        })

    # Select development subset of exactly dev_k (seeded), preferring balanced SUPPORT/CONTRADICT
    support_candidates = [a for a in aligned if a.get("ground_truth_label") == "SUPPORT" and a["relevant_document_count"] > 0]
    contradict_candidates = [a for a in aligned if a.get("ground_truth_label") == "CONTRADICT" and a["relevant_document_count"] > 0]
    random.shuffle(support_candidates)
    random.shuffle(contradict_candidates)

    target_each = dev_k // 2
    available_support = len(support_candidates)
    available_contradict = len(contradict_candidates)

    if available_support >= target_each and available_contradict >= target_each:
        sel_support = support_candidates[:target_each]
        sel_contradict = contradict_candidates[:target_each]
        dev_selected = sel_support + sel_contradict
    else:
        # Maximum balanced subset available
        max_each = min(available_support, available_contradict)
        sel_support = support_candidates[:max_each]
        sel_contradict = contradict_candidates[:max_each]
        dev_selected = sel_support + sel_contradict

    # If we still don't have any (edge-case), leave dev_selected empty

    # Save outputs
    dev_queries_path = os.path.join(OUT_DIR, "dev_queries.jsonl")
    corpus_path = os.path.join(OUT_DIR, "corpus.jsonl")
    metadata_path = os.path.join(OUT_DIR, "dataset_metadata.json")

    # Save dev queries
    save_jsonl(dev_queries_path, dev_selected)
    # Save full corpus
    save_jsonl(corpus_path, list(corpus_map.values()))

    metadata = {
        "dataset_name": "BEIR SciFact (aligned to scifact)",
        "source": "Hugging Face: BeIR/scifact, BeIR/scifact-qrels, scifact",
        "random_seed": seed,
        "selected_queries": len(dev_selected),
        "total_corpus_documents": len(corpus_map),
        "total_beir_queries": len(beir_queries_list),
        "total_qrels_used": qrels_count,
        "total_original_scifact_claims": scifact_count,
        "num_aligned": len(aligned),
        "num_failed_align": len(failed_align),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "filtering": {
            "require_label_support_contradict": True,
            "require_relevant_doc": True,
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Reporting (concise)
    print("\nWHAT WE BUILT:")
    print(f" - dev queries: {dev_queries_path}")
    print(f" - corpus: {corpus_path}")
    print(f" - metadata: {metadata_path}")

    print("\nDATASET:")
    print(f" - Total corpus documents loaded: {len(corpus_map)}")
    print(f" - Total BEIR queries: {len(beir_queries_list)}")
    print(f" - Total test qrels records: {qrels_count}")
    print(f" - Total original SciFact labeled claims found: {scifact_count}")
    print(f" - Number successfully aligned: {len(aligned)}")
    print(f" - Number failed to align: {len(failed_align)}")

    print("\nDEVELOPMENT SUBSET:")
    support_count = sum(1 for x in dev_selected if x.get("ground_truth_label") == "SUPPORT")
    contradict_count = sum(1 for x in dev_selected if x.get("ground_truth_label") == "CONTRADICT")
    print(f" - Number of selected claims: {len(dev_selected)}")
    print(f" - Random seed: {seed}")
    print(f" - SUPPORT claims: {support_count}")
    print(f" - CONTRADICT claims: {contradict_count}")
    print(f" - Every selected claim has >=1 relevant document: {all(x['relevant_document_count']>0 for x in dev_selected)}")
    print(" - 3 example records:")
    for ex in dev_selected[:3]:
        print(json.dumps(ex, ensure_ascii=False))

    print("\nGROUND TRUTH:")
    print(" - SUPPORT/CONTRADICT labels came from the Hugging Face 'scifact' dataset (original SciFact annotations).")
    print(" - Relevant document ids were obtained from the BeIR scifact-qrels dataset (field 'corpus-id').")

    print("\nFILES GENERATED:")
    print(f" - {dev_queries_path}")
    print(f" - {corpus_path}")
    print(f" - {metadata_path}")

    return {
        "dev_queries_path": dev_queries_path,
        "corpus_path": corpus_path,
        "metadata_path": metadata_path,
        "stats": metadata,
        "examples": dev_selected[:3],
        "failed_align": failed_align,
    }


if __name__ == "__main__":
    result = main()
    print(json.dumps(result["stats"], indent=2, ensure_ascii=False))
