# Day 3 — Evidence-Aware RAG Analysis

## 1. Objective

Day 3 investigated whether providing Gemini with focused evidence sentences instead of complete retrieved documents changes RAG verdict behavior.

Two evidence extraction approaches were tested:

1. Original similarity-based extraction
2. Diverse extraction using semantic similarity plus document diversity

These experiments were exploratory and small-scale. They do not establish statistical significance or causality.

## 2. Baseline Evidence Pipeline

The existing baseline approach is:

```text
Claim -> FAISS top-K -> retrieved documents -> Gemini
```

The evidence-aware approach is:

```text
Claim -> FAISS top-K -> sentence splitting -> sentence embeddings -> cosine similarity -> selected evidence sentences -> Gemini
```

## 3. Original Evidence Extractor

The original extractor uses `sentence-transformers/all-MiniLM-L6-v2` to encode the claim and individual document sentences. The embeddings are normalized, so their inner product represents cosine similarity.

Sentences are selected by semantic similarity to the claim. The extractor keeps a maximum of 2 sentences per document and a maximum of 6 sentences in total.

## 4. Evidence Quality Diagnostic

The evidence-quality diagnostic analyzed 10 evidence-aware records.

- Evidence-aware records: 10
- Average evidence sentences per record: 6.0
- Average maximum similarity: 0.7172
- Relevant document represented: 5/10
- Representation rate: 0.50

By K:

- K=3: 2/5 records represented a known relevant document, or 0.40.
- K=5: 3/5 records represented a known relevant document, or 0.60.

Additional diagnostic cases:

- Relevant document retrieved but no sentence selected: 0
- Relevant document represented: 5
- Non-relevant sentence outranking relevant sentences: 2
- Evidence selected from only one document: 0

“Relevant document represented” means that at least one selected sentence came from a document labeled relevant in the dataset. It does not mean that the selected sentence itself was proven to be valid evidence.

## 5. Diverse Evidence Extractor

The diverse extractor selects the first sentence by highest semantic similarity.

For each subsequent sentence, it uses:

```text
selection score = 0.7 x semantic similarity + 0.3 x document diversity
```

Document diversity is 1 when the sentence comes from a document not yet represented and 0 otherwise.

The extractor enforces a maximum of 2 sentences per document and a maximum of 6 total sentences.

## 6. Original vs Diverse Extractor

| K | Original Relevant Representation | Diverse Relevant Representation |
|---|---:|---:|
| 3 | 2/5 (0.40) | 2/5 (0.40) |
| 5 | 3/5 (0.60) | 3/5 (0.60) |

Average unique documents represented:

- K=3: original 3.0; diverse 3.0
- K=5: original 4.2; diverse 5.0

The diverse extractor increased document diversity at K=5 but did not increase relevant-document representation in this diagnostic. This does not show that the diverse strategy improves RAG.

## 7. Evidence-Aware Generation Sanity Test

The generation comparison used these five queries:

- 118
- 1019
- 1320
- 1370
- 1185

K values were 3 and 5. The three conditions were:

- Baseline full documents
- Original evidence
- Diverse evidence

There were 30 intended Gemini calls. All 30 calls succeeded and 0 failed. The model was `gemini-3.5-flash-lite`.

## 8. Final Day 3 Generation Results

| K | Full Document Accuracy | Original Evidence Accuracy | Diverse Evidence Accuracy |
|---|---:|---:|---:|
| 3 | 0.200 | 0.200 | 0.200 |
| 5 | 0.400 | 0.400 | 0.400 |

All three conditions produced identical predictions for all 10 query/K comparisons.

- Original evidence matched full documents: 10/10
- Diverse evidence matched full documents: 10/10
- Prediction changes: 0
- Improvements relative to full documents: 0
- Regressions relative to full documents: 0

## 9. Query-Level Findings

- **Query 118:** `SUPPORT` for both K values in all conditions.
- **Query 1019:** `INSUFFICIENT_EVIDENCE` for both K values in all conditions.
- **Query 1320:** `INSUFFICIENT_EVIDENCE` for both K values in all conditions.
- **Query 1370:** `INSUFFICIENT_EVIDENCE` for both K values in all conditions.
- **Query 1185:** `INSUFFICIENT_EVIDENCE` at K=3 and `SUPPORT` at K=5 for all conditions.

These are recorded outcomes only. No additional explanation for the verdicts is inferred here.

## 10. Interpretation

On this small exploratory sample, changing the evidence representation from full documents to extracted evidence sentences did not alter Gemini’s verdicts.

This result does not establish that evidence extraction is ineffective generally. One hypothesis is that the selected sentence sets preserved enough of the information used by the model for these particular claims and K values, but this hypothesis was not tested by the available experiment and should not be treated as a conclusion. Another hypothesis is that the five-query sample was too small to expose differences; this also requires further evaluation.

No causal claim is made.

## 11. Limitations

- Only 5 queries were evaluated.
- Only K=3 and K=5 were tested.
- The comparison used 30 total Gemini calls.
- Only one embedding model was used.
- Only one dataset was used.
- Only one Gemini model was used.
- No statistical significance testing was performed.
- No confidence intervals were calculated.
- Sentence similarity is not equivalent to evidence correctness.
- Relevant-document representation does not guarantee correct sentence selection.

## 12. Day 3 Conclusion

Day 3 showed that the proposed evidence extraction pipeline can technically select focused sentence-level evidence, but the small controlled generation comparison did not produce different Gemini verdicts relative to full-document input. The diverse selection strategy increased document diversity at K=5 but did not increase relevant-document representation in the diagnostic sample.

The findings are exploratory and do not show that the method fails universally or that evidence extraction improves RAG generally.

## 13. Reproducibility

Source components:

- `src/rag/evidence_extractor.py`
- `src/rag/diverse_evidence_extractor.py`
- `src/rag/evidence_generator.py`

Experiment scripts:

- `experiments/test_evidence_extractor.py`
- `experiments/test_evidence_rag.py`
- `experiments/analyze_evidence_quality.py`
- `experiments/compare_evidence_extractors.py`
- `experiments/run_evidence_comparison.py`

Day 3 outputs:

- `results/evidence_rag_sanity.jsonl`
- `results/evidence_quality_analysis.json`
- `results/evidence_quality_analysis.txt`
- `results/evidence_extractor_comparison.json`
- `results/evidence_extractor_comparison.txt`
- `results/evidence_comparison.jsonl`
- `results/evidence_comparison_metrics.json`
- `results/evidence_comparison_summary.txt`

During final documentation, API calls: 0. The dataset and FAISS index were unchanged. Day 1 and Day 2 outputs were unchanged, and the evidence experiment outputs were not modified.
