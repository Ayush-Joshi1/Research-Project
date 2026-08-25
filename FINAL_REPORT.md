# Retrieval Depth, Cross-Encoder Reranking, and Evidence-Aware RAG: A Controlled Multi-Day Study

## 2. Abstract

This project studied how retrieval depth, cross-encoder reranking, and sentence-level evidence representation affect scientific claim verification with retrieval-augmented generation (RAG). Day 1 evaluated a 50-claim baseline across K=1, 3, 5, and 10. Day 2 compared FAISS ordering with cross-encoder reranking of the same top-10 candidates. Day 3 compared full documents with two locally extracted evidence representations on five queries.

Observed results showed that increasing K improved baseline retrieval Hit@K and verdict accuracy. Reranking improved observed retrieval ranking at K=1, 3, and 5 and was associated with higher observed verdict accuracy at those K values, while K=10 retrieval was unchanged and reranked verdict accuracy was slightly lower. In the small Day 3 comparison, all three evidence representations produced identical predictions. The findings are limited by the 50-query evaluation, Day 2 quota-related failures, the small Day 3 sample, one duplicate raw Day 2 record, and the absence of statistical significance testing.

## 3. Research Objective

The project investigated:

1. The effect of retrieval depth K on retrieval recall and final RAG performance.
2. Whether cross-encoder reranking improves retrieval ranking.
3. Whether improved retrieval ranking corresponds to improved final RAG verdict accuracy.
4. Whether sentence-level evidence extraction changes evidence-grounded verification.

## 4. Research Questions

**RQ1:** How does retrieval depth affect retrieval recall and final verdict accuracy?

**RQ2:** Does cross-encoder reranking improve retrieval ranking?

**RQ3:** Does improved retrieval ranking correspond to improved final RAG verdict accuracy?

**RQ4:** Does sentence-level evidence extraction improve or change evidence-grounded verification?

## 5. Dataset and Ground Truth

The experiments used the SciFact dataset and a controlled 50-query development subset. Claims were evaluated with `SUPPORT` and `CONTRADICT` ground-truth labels and associated relevant document IDs. The selected development queries were used consistently across the controlled evaluations where specified.

The validated reports do not record a separate corpus-size figure, so no corpus-size number is introduced here. The Day 1 and Day 2 controlled evaluations each contain 50 queries; Day 3 uses the specified five-query subset. The relevant document IDs and labels come from the processed development-query metadata and were not changed during the experiments.

## 6. Experimental Setup

The baseline pipeline was:

```text
Claim -> MiniLM embedding -> FAISS IndexFlatIP -> top-K retrieval -> Gemini -> verdict
```

Day 2 used:

```text
Claim -> FAISS top-10 -> CrossEncoder reranking -> top-K -> Gemini -> verdict
```

Day 3 used:

```text
Claim -> FAISS top-K -> sentence-level evidence extraction -> Gemini -> verdict
```

The embedding model was `sentence-transformers/all-MiniLM-L6-v2`. The Day 2 reranker was `cross-encoder/ms-marco-MiniLM-L-6-v2`. The generator model was `gemini-3.5-flash-lite` with the established baseline prompt for the baseline and reranking evaluations.

## 7. Day 1 — Baseline RAG

| K | Retrieval Hit@K | Verdict Accuracy | SUPPORT Accuracy | CONTRADICT Accuracy | Successful Calls | Failed Calls |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.540 | 0.620 | 0.640 | 0.600 | 50 | 0 |
| 3 | 0.700 | 0.720 | 0.720 | 0.720 | 50 | 0 |
| 5 | 0.760 | 0.740 | 0.840 | 0.640 | 50 | 0 |
| 10 | 0.900 | 0.800 | 0.840 | 0.760 | 50 | 0 |

In the Day 1 baseline, both retrieval Hit@K and verdict accuracy increased overall as K increased, although verdict accuracy rose by different amounts between adjacent K values. All 200 intended Day 1 calls succeeded.

## 8. Day 2 — Cross-Encoder Reranking

| K | Baseline Hit@K | Reranked Hit@K | Retrieval Delta | Baseline Accuracy | Reranked Accuracy | Verdict Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.540 | 0.620 | +0.080 | 0.619 | 0.707 | +0.088 |
| 3 | 0.700 | 0.840 | +0.140 | 0.738 | 0.780 | +0.042 |
| 5 | 0.760 | 0.880 | +0.120 | 0.732 | 0.756 | +0.024 |
| 10 | 0.900 | 0.900 | +0.000 | 0.805 | 0.800 | -0.005 |

Reranking improved observed retrieval at K=1, 3, and 5. At K=10, retrieval remained unchanged because reranking reordered the same FAISS top-10 candidate set. Verdict accuracy was higher for reranking at K=1, 3, and 5 and slightly lower at K=10. These are observed associations and do not establish causality.

## 9. Day 2 Ranking Analysis

The first relevant-document rank distribution in the FAISS top-10 results was:

| First relevant rank | Number of queries |
|---:|---:|
| 1 | 27 |
| 2 | 5 |
| 3 | 3 |
| 4 | 3 |
| 5 | 0 |
| 6 | 2 |
| 7 | 3 |
| 8 | 0 |
| 9 | 2 |
| 10 | 0 |
| None in top 10 | 5 |

Recall values were:

| K | FAISS Recall@K | Cross-Encoder Recall@K |
|---:|---:|---:|
| 1 | 0.54 | 0.62 |
| 3 | 0.70 | 0.84 |
| 5 | 0.76 | 0.88 |
| 10 | 0.90 | 0.90 |

The mean first relevant rank was 2.4222 and the median was 1. Five queries had no relevant document in the FAISS top 10. Ranking matters for downstream evidence access because a relevant document can be available within the candidate set but excluded from a smaller top-K prefix when it appears later in the ordering.

## 10. Day 2 Error Analysis

The paired successful verdict transitions were:

| K | Paired Successful | Baseline Wrong -> Reranked Correct | Baseline Correct -> Reranked Wrong | Both Correct | Both Wrong | Prediction Changed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41 | 4 | 0 | 25 | 12 | 4 |
| 3 | 41 | 3 | 1 | 29 | 8 | 4 |
| 5 | 41 | 2 | 1 | 29 | 9 | 3 |
| 10 | 40 | 1 | 1 | 31 | 7 | 2 |

The retrieval/verdict relationship counts were:

| K | Retrieval Improved + Verdict Improved | Retrieval Improved + Verdict Unchanged | Retrieval Improved + Verdict Worsened | Retrieval Unchanged + Verdict Improved | Retrieval Unchanged + Verdict Unchanged | Retrieval Unchanged + Verdict Worsened |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 3 | 0 | 1 | 32 | 0 |
| 3 | 3 | 3 | 0 | 0 | 33 | 1 |
| 5 | 1 | 3 | 0 | 1 | 35 | 1 |
| 10 | 0 | 0 | 0 | 1 | 38 | 1 |

A retrieval miss is distinct from a generation/verdict error: a relevant document may be absent from the retrieved prefix, while a verdict may still be wrong even when retrieval hits. The Day 2 analysis recorded strongest positive paired changes including `880/K=1`, `1180/K=1`, and `674/K=1`; recorded regressions included `1303/K=3` and `1163/K=5` and `K=10`. These changes are differences in stored predictions, not proof of a mechanism.

## 11. Day 2 Case Studies

### Query 1320

The claim concerns transplanted human glial progenitor cells and has ground truth `CONTRADICT`. The relevant document moved from FAISS rank 2 to reranked position 1. Stored verdicts remained unchanged across the evaluated K values in the case-study analysis. The observed rank improvement did not by itself produce a verdict change.

### Query 1019

The claim concerns rapid phosphotransfer rates and fidelity in two-component systems, with ground truth `SUPPORT`. The relevant document began at FAISS rank 6 and moved to reranked position 1. Stored verdicts remained unchanged. This shows that improved relevant-document rank and a changed final verdict did not always occur together.

### Query 1370

The claim states that vitamin D deficiency is unrelated to birth weight, with ground truth `CONTRADICT`. Reranking was associated with changes from `INSUFFICIENT_EVIDENCE` to the correct `CONTRADICT` verdict at K=3 and K=5. This is an observed paired outcome and does not prove causality.

### Query 1185

The claim concerns possible health-care savings from an optimized national kidney paired-donation program, with ground truth `SUPPORT`. The relevant document moved from FAISS rank 4 to reranked position 1, improving its position in smaller prefixes. API failures prevent a complete verdict comparison for this query.

### Query 314

The claim concerns cytidine deamination and G-to-A mutations, with ground truth `SUPPORT`. Baseline calls succeeded at K=1 and K=3 while corresponding reranked calls failed; both conditions also had failed calls at K=5 and K=10. These failures prevent a complete verdict comparison.

## 12. Day 3 — Evidence-Aware RAG

Day 3 tested two local evidence representations. The original extractor embeds the claim and individual document sentences with normalized `all-MiniLM-L6-v2` vectors and ranks sentences by cosine similarity, keeping at most two sentences per document and six total sentences.

The diverse extractor selects the first sentence by semantic similarity and scores subsequent candidates using:

```text
0.7 x semantic similarity + 0.3 x document diversity
```

It also enforces a maximum of two sentences per document and six total sentences.

The evidence-quality diagnostic found:

- 10 evidence-aware records
- Average 6.0 evidence sentences per record
- Average maximum similarity 0.7172
- Known relevant document represented in 5/10 records, rate 0.50
- K=3: 2/5, rate 0.40
- K=5: 3/5, rate 0.60
- Relevant document retrieved but no sentence selected: 0
- Relevant document represented: 5
- Non-relevant sentence outranking relevant sentences: 2
- Evidence selected from only one document: 0

Representation of a known relevant document does not prove that a selected sentence is correct evidence.

The original versus diverse diagnostic produced:

| K | Original Relevant Representation | Diverse Relevant Representation |
|---:|---:|---:|
| 3 | 2/5 (0.40) | 2/5 (0.40) |
| 5 | 3/5 (0.60) | 3/5 (0.60) |

Average unique documents represented was 3.0 for both extractors at K=3. At K=5 it was 4.2 for the original extractor and 5.0 for the diverse extractor. The diverse strategy increased document diversity at K=5 but did not increase relevant-document representation in this diagnostic.

The final 30-call comparison used five queries, K=3 and K=5, and three conditions: full documents, original evidence, and diverse evidence.

| K | Full Document Accuracy | Original Evidence Accuracy | Diverse Evidence Accuracy |
|---:|---:|---:|---:|
| 3 | 0.200 | 0.200 | 0.200 |
| 5 | 0.400 | 0.400 | 0.400 |

All three conditions produced identical predictions in all 10 query/K cases. Original evidence matched full documents 10/10, diverse evidence matched full documents 10/10, and there were zero prediction changes, improvements, or regressions relative to full documents.

## 13. Cross-Day Comparison

**Day 1 observation:** Increasing K improved both observed retrieval Hit@K and verdict accuracy in the baseline evaluation.

**Day 2 observation:** Cross-encoder reranking improved observed retrieval ranking at lower K values and was associated with higher observed verdict accuracy at K=1, 3, and 5. At K=10, retrieval was unchanged and reranked verdict accuracy was slightly lower.

**Day 3 observation:** Evidence extraction did not change predictions in the tested five-query sample. The diverse extractor increased document diversity at K=5 in the offline diagnostic, but this did not translate into different predictions in the 30-call comparison.

These statements describe recorded results. They do not establish causality or general superiority.

## 14. Key Findings

- Baseline retrieval Hit@K rose from 0.540 at K=1 to 0.900 at K=10.
- Baseline verdict accuracy rose from 0.620 to 0.800 across the same K values.
- Reranking produced its largest observed retrieval delta at K=3: +0.140.
- Reranking improved observed retrieval Hit@K at K=1, 3, and 5.
- Reranking was associated with higher observed verdict accuracy at K=1, 3, and 5.
- At K=10, retrieval was identical and reranked verdict accuracy was slightly lower.
- Day 3 full-document, original-evidence, and diverse-evidence predictions were identical on the tested sample.
- Better retrieval ranking did not guarantee a better final RAG verdict.

## 15. Limitations

- The main controlled evaluation used 50 queries.
- Day 3 used only 5 queries.
- Day 2 included free-tier Gemini quota/rate-limit failures.
- Day 2 contains 401 raw records versus 400 unique combinations.
- One duplicate raw Day 2 record was present.
- The evaluation is descriptive.
- No statistical significance testing was performed.
- No causal inference is supported.
- Results may depend on the selected models and prompt.
- Evidence extraction was tested on a small sample.

## 16. Threats to Validity

- **Sample size:** Fifty queries for the main evaluation and five for Day 3 limit how broadly the observed patterns can be interpreted.
- **API failure handling:** Day 2 failed calls reduce the successful-call sample and can affect which paired verdicts are available.
- **Model dependence:** Results may change with another embedding model, reranker, or Gemini model.
- **Retrieval-corpus dependence:** Candidate availability and relevant-document ranks depend on the selected corpus and FAISS index.
- **Prompt dependence:** The generator prompt is part of the measured system and may influence verdict behavior.
- **Reranker dependence:** Day 2 results are specific to the tested cross-encoder.
- **Evaluation-label dependence:** Accuracy depends on the supplied SUPPORT/CONTRADICT labels and relevant document IDs.
- **Generative randomness:** Even with the same inputs, generative model behavior may vary across calls or settings.

## 17. Reproducibility

Major project scripts include:

- `src/data/prepare_dataset.py`
- `experiments/test_retrieval.py`
- `experiments/run_rag_experiment.py`
- `experiments/analyze_ranking.py`
- `experiments/run_reranked_rag_experiment.py`
- `experiments/test_reranking.py`
- `src/rag/evidence_extractor.py`
- `src/rag/diverse_evidence_extractor.py`
- `src/rag/evidence_generator.py`
- `experiments/test_evidence_extractor.py`
- `experiments/test_evidence_rag.py`
- `experiments/analyze_evidence_quality.py`
- `experiments/compare_evidence_extractors.py`
- `experiments/run_evidence_comparison.py`
- `experiments/final_analysis.py`

Authoritative and generated analysis files include:

- `results/rag_metrics_clean.json`
- `results/reranked_rag_analysis.json`
- `results/ranking_analysis.json`
- `results/final_case_study_analysis.json`
- `results/evidence_quality_analysis.json`
- `results/evidence_extractor_comparison.json`
- `results/evidence_comparison_metrics.json`
- `results/final_analysis.json`
- `results/final_comparison_table.csv`
- `results/final_analysis_summary.txt`
- `docs/DAY2_REPORT.md`
- `docs/DAY3_REPORT.md`

Baseline experiment outputs were preserved. API keys and secrets were not committed.

## 18. Conclusion

In this controlled evaluation, increasing retrieval depth improved observed baseline retrieval and verdict performance. Cross-encoder reranking substantially improved observed retrieval ranking at lower K values and was associated with higher observed verdict accuracy at K=1, 3, and 5, while the effect disappeared for retrieval at K=10 and the reranked verdict accuracy was slightly lower there.

The Day 3 evidence pipeline technically selected focused sentence-level evidence. However, the small controlled comparison produced identical Gemini predictions for full documents, original extracted evidence, and diverse extracted evidence. The results are consistent with evidence representation being sufficient to preserve these particular predictions, but the experiment does not establish that evidence extraction is ineffective generally or that any method caused a verdict change.

## 19. Future Work

Future work could evaluate:

- A larger evaluation set
- Repeated runs and multiple seeds
- Statistical testing and confidence intervals
- Stronger or alternative rerankers
- Passage-level retrieval
- Improved evidence extraction strategies
- Alternative LLMs
- Latency and cost trade-offs

These extensions are suggestions only and were not implemented in this study.

## 20. Appendix

### A. Final Metric Tables

**Day 1 baseline**

| K | Hit@K | Verdict Accuracy |
|---:|---:|---:|
| 1 | 0.540 | 0.620 |
| 3 | 0.700 | 0.720 |
| 5 | 0.760 | 0.740 |
| 10 | 0.900 | 0.800 |

**Day 2 reranking**

| K | Baseline Hit@K | Reranked Hit@K | Retrieval Delta | Baseline Accuracy | Reranked Accuracy | Verdict Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.540 | 0.620 | +0.080 | 0.619 | 0.707 | +0.088 |
| 3 | 0.700 | 0.840 | +0.140 | 0.738 | 0.780 | +0.042 |
| 5 | 0.760 | 0.880 | +0.120 | 0.732 | 0.756 | +0.024 |
| 10 | 0.900 | 0.900 | +0.000 | 0.805 | 0.800 | -0.005 |

**Day 3 evidence representation**

| K | Full Documents | Original Evidence | Diverse Evidence |
|---:|---:|---:|---:|
| 3 | 0.200 | 0.200 | 0.200 |
| 5 | 0.400 | 0.400 | 0.400 |

### B. Experiment Configurations

- Day 1: 50 claims x K values 1, 3, 5, 10 x baseline condition.
- Day 2: 50 claims x K values 1, 3, 5, 10 x baseline and reranked conditions.
- Day 3: 5 claims x K values 3 and 5 x full-document, original-evidence, and diverse-evidence conditions.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
- Initial retrieval: FAISS `IndexFlatIP`.
- Day 2 reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Generator: `gemini-3.5-flash-lite`.

### C. Important Files and Outputs

- Day 1 clean baseline results and metrics: `results/rag_results_clean.jsonl`, `results/rag_metrics_clean.json`.
- Day 2 reranked results and analyses: `results/reranked_rag_results.jsonl`, `results/reranked_rag_analysis.json`, `results/ranking_analysis.json`.
- Day 3 evidence comparisons: `results/evidence_rag_sanity.jsonl`, `results/evidence_comparison.jsonl`, `results/evidence_comparison_metrics.json`.
- Day 4 consolidated outputs: `results/final_analysis.json`, `results/final_analysis.csv`, `results/final_case_studies.json`, `results/final_comparison_table.csv`, `results/final_analysis_summary.txt`.
- Day reports: `docs/DAY2_REPORT.md`, `docs/DAY3_REPORT.md`.
