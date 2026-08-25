# Day 2 — Cross-Encoder Reranking for RAG

## 1. Objective

Day 2 tests whether cross-encoder reranking of the FAISS top-10 candidate documents changes:

1. retrieval quality;
2. final Gemini RAG verdict accuracy.

This is a descriptive experiment. It does not establish causality or statistical significance.

## 2. Experimental Setup

- **Dataset:** SciFact, 50 development claims
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Initial retriever:** existing FAISS `IndexFlatIP`
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Generator:** `gemini-3.5-flash-lite`
- **K values:** 1, 3, 5, 10

Both conditions use the same baseline Gemini prompt.

**Baseline**

```text
Claim -> FAISS -> top K -> Gemini
```

**Reranked**

```text
Claim -> FAISS top 10 -> CrossEncoder -> top K -> Gemini
```

## 3. Experimental Design

The experiment contains 50 claims x 4 K values x 2 conditions = 400 intended combinations.

For the reranked condition, FAISS first retrieves 10 candidates. The cross-encoder scores those 10 documents, the documents are sorted by cross-encoder score, and the prefix of size K is passed to Gemini.

At K=10, both conditions contain the same FAISS top-10 candidate set; only the ordering differs.

## 4. Retrieval Results

| K | Baseline Hit@K | Reranked Hit@K | Retrieval Delta |
|---|---:|---:|---:|
| 1 | 0.540 | 0.620 | +0.080 |
| 3 | 0.700 | 0.840 | +0.140 |
| 5 | 0.760 | 0.880 | +0.120 |
| 10 | 0.900 | 0.900 | +0.000 |

Reranking produced higher observed retrieval Hit@K at K=1, K=3, and K=5. The largest observed retrieval delta was at K=3. At K=10, the hit rate was unchanged because reranking changes order within the same candidate set.

## 5. RAG Verdict Results

| K | Baseline Accuracy | Reranked Accuracy | Verdict Delta |
|---|---:|---:|---:|
| 1 | 0.619 | 0.707 | +0.088 |
| 3 | 0.738 | 0.780 | +0.042 |
| 5 | 0.732 | 0.756 | +0.024 |
| 10 | 0.805 | 0.800 | -0.005 |

Accuracy is calculated only among successful API calls. Failed API calls are not counted as incorrect verdicts.

## 6. API Failures and Data Integrity

The intended experiment contains 400 unique combinations. The raw JSONL contains 401 records because one combination, `reranked / query_id 1041 / K=10`, appears twice.

- Intended combinations: 400
- Raw JSONL records: 401
- Duplicate combinations: 1
- Unique combinations: 400
- Successful unique combinations: 329
- Failed unique combinations: 71

The failures came from Gemini quota and rate-limit issues encountered during the long-running experiment. They are reported rather than hidden. Verdict accuracy excludes failed API calls. Aggregate metrics use one row per unique condition/query/K combination and do not double-count the duplicate raw record.

## 7. Ranking Analysis

The saved ranking analyses report the following recall values:

| K | FAISS Recall@K | Cross-Encoder Recall@K |
|---|---:|---:|
| 1 | 0.54 | 0.62 |
| 3 | 0.70 | 0.84 |
| 5 | 0.76 | 0.88 |
| 10 | 0.90 | 0.90 |

Reranking substantially improved the observed ordering of relevant documents within the FAISS top-10 candidate set at K=1, K=3, and K=5. At K=10, recall was unchanged because the candidate set was unchanged.

## 8. Case Studies

### Query 1320

The relevant document moved from FAISS rank 2 to reranked position 1. The stored verdicts remained unchanged. This is an observed rank movement and verdict outcome, not evidence of causality.

### Query 1019

The relevant document originally appeared at FAISS rank 6 and moved to reranked position 1. The stored verdicts remained unchanged.

### Query 1370

Reranking was associated with recorded changes from `INSUFFICIENT_EVIDENCE` to the correct `CONTRADICT` verdict at K=3 and K=5. These observations do not prove that reranking caused the verdict changes.

### Query 1185

The relevant document moved substantially, from FAISS rank 4 to reranked position 1, and retrieval improved at smaller K values. API failures prevent a complete verdict comparison for this query.

### Query 314

The baseline generation calls succeeded at K=1 and K=3 while the corresponding reranked generation calls failed; both conditions also had failed generation calls at K=5 and K=10. These failures prevent a complete verdict comparison.

## 9. Discussion

1. Reranking produced the largest observed retrieval gains at K=3 and K=5.
2. Reranking was associated with higher observed verdict accuracy at K=1, K=3, and K=5.
3. The observed verdict improvement decreased as K increased.
4. At K=10, retrieval performance was identical and reranked verdict accuracy was slightly lower.
5. Better retrieval ranking did not guarantee a better final RAG verdict.
6. The relationship between retrieval quality and generation quality was not perfectly monotonic in this experiment.

These are descriptive observations only; they do not establish causality.

## 10. Limitations

- Only 50 development claims were evaluated.
- Gemini free-tier quota failures affected the recorded results.
- Verdict accuracy uses successful API calls only.
- The raw JSONL contains one duplicate record.
- No statistical significance testing was performed.
- No multiple-seed evaluation was performed.
- No confidence intervals were calculated.
- Only one embedding model was used.
- Only one cross-encoder was used.
- Only one Gemini model was used.
- Only one dataset was used.

## 11. Conclusion

In this controlled experiment, cross-encoder reranking substantially improved the observed retrieval ranking at K=1, K=3, and K=5. It was also associated with improved observed verdict accuracy at those K values. At K=10, retrieval performance was identical and the observed verdict accuracy was slightly lower for reranking.

These results do not show that cross-encoder reranking always improves RAG performance, and they do not establish causality. They describe the behavior observed under this dataset, model configuration, prompt, and set of successful API calls.

## 12. Reproducibility

Important experiment and analysis scripts include:

- `experiments/run_reranked_rag_experiment.py`
- `experiments/analyze_reranked_rag.py`
- `experiments/analyze_case_studies.py`
- `experiments/analyze_ranking.py`
- `experiments/test_reranking.py`

Generated analysis outputs include:

- `results/reranked_rag_analysis.json`
- `results/reranked_rag_analysis.csv`
- `results/final_case_study_analysis.json`
- `results/day2_final_metrics_table.csv`
- `results/day2_final_summary.txt`

The baseline experiment outputs were preserved. No Gemini API calls, retrieval calls, or reranking calls were made while documenting this report. No dataset, FAISS index, or existing experiment result file was modified.
