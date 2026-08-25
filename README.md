# Retrieval Depth, Reranking, and Evidence-Aware RAG

A controlled research project studying how retrieval depth, cross-encoder reranking, and sentence-level evidence representation affect scientific claim verification with retrieval-augmented generation (RAG).

The detailed final report is [FINAL_REPORT.md](FINAL_REPORT.md). It contains the complete methods, results, case studies, limitations, and threats to validity.

## Overview

RAG systems depend on which documents reach the language model. This project varies retrieval depth `K`, tests whether a cross-encoder improves the ordering of FAISS candidates, and examines whether replacing full documents with extracted evidence sentences changes Gemini verdicts. The goal is to separate retrieval behavior from downstream generation behavior using stored, reproducible experiment outputs.

## Research Questions

- **RQ1:** How does retrieval depth affect retrieval recall and final verdict accuracy?
- **RQ2:** Does cross-encoder reranking improve retrieval ranking?
- **RQ3:** Does improved retrieval ranking correspond to improved final RAG verdict accuracy?
- **RQ4:** Does sentence-level evidence extraction improve or change evidence-grounded verification?

## System Architecture

### Baseline RAG

```text
Claim
	↓
all-MiniLM-L6-v2
	↓
FAISS IndexFlatIP
	↓
Top-K documents
	↓
Gemini
	↓
SUPPORT / CONTRADICT / INSUFFICIENT_EVIDENCE
```

### Cross-Encoder Reranking

```text
Claim
	↓
FAISS Top-10
	↓
CrossEncoder reranking
	↓
Top-K
	↓
Gemini
	↓
Verdict
```

### Evidence-Aware RAG

```text
Claim
	↓
FAISS Top-K
	↓
Sentence-level evidence extraction
	↓
Gemini
	↓
Verdict
```

## Dataset

The project uses the SciFact dataset with a controlled 50-query development evaluation. Ground truth consists of `SUPPORT` and `CONTRADICT` labels with associated relevant document IDs. Day 3 is a smaller exploratory evaluation using five queries and K values 3 and 5.

## Models and Technologies

- Python
- Hugging Face Datasets
- Sentence Transformers
- `sentence-transformers/all-MiniLM-L6-v2`
- FAISS `IndexFlatIP`
- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Google Gemini API
- `gemini-3.5-flash-lite`
- NumPy, Pandas, and Matplotlib where applicable

## Experiments

### Baseline RAG

Day 1 evaluated retrieval depths `K = 1, 3, 5, 10` using FAISS retrieval and the baseline Gemini generator.

### Cross-Encoder Reranking

Day 2 retrieved FAISS top-10 candidates, reranked those candidates with the cross-encoder, and passed the top K documents to Gemini.

### Evidence-Aware RAG

Day 3 compared full-document input with original similarity-based evidence extraction and a diversity-aware extractor.

### Final Analysis

Day 4 consolidated the stored Day 1, Day 2, and Day 3 outputs offline. No new model calls are required for the final analysis.

## Main Results

### Day 1 Baseline

| K | Retrieval Hit@K | Verdict Accuracy | SUPPORT Accuracy | CONTRADICT Accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.540 | 0.620 | 0.640 | 0.600 |
| 3 | 0.700 | 0.720 | 0.720 | 0.720 |
| 5 | 0.760 | 0.740 | 0.840 | 0.640 |
| 10 | 0.900 | 0.800 | 0.840 | 0.760 |

### Day 2 Reranking

| K | Baseline Hit@K | Reranked Hit@K | Retrieval Delta | Baseline Accuracy | Reranked Accuracy | Verdict Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.540 | 0.620 | +0.080 | 0.619 | 0.707 | +0.088 |
| 3 | 0.700 | 0.840 | +0.140 | 0.738 | 0.780 | +0.042 |
| 5 | 0.760 | 0.880 | +0.120 | 0.732 | 0.756 | +0.024 |
| 10 | 0.900 | 0.900 | +0.000 | 0.805 | 0.800 | -0.005 |

Reranking improved observed retrieval at K=1, 3, and 5 and was associated with higher observed verdict accuracy at those K values. At K=10, retrieval was unchanged and reranked verdict accuracy was slightly lower. These are descriptive results, not causal claims.

### Day 3 Evidence Representation

| K | Full Document Accuracy | Original Evidence Accuracy | Diverse Evidence Accuracy |
|---:|---:|---:|---:|
| 3 | 0.200 | 0.200 | 0.200 |
| 5 | 0.400 | 0.400 | 0.400 |

All three representations produced identical predictions in the tested five-query comparison. This small exploratory result does not establish general superiority or ineffectiveness.

## Key Findings

- Increasing K improved observed baseline retrieval Hit@K and verdict accuracy in the 50-query evaluation.
- Cross-encoder reranking produced the largest observed retrieval gain at K=3: `+0.140`.
- Reranking was associated with higher observed verdict accuracy at K=1, 3, and 5.
- At K=10, retrieval was identical for baseline and reranked conditions.
- Better retrieval ranking did not guarantee a better final verdict.
- Day 3 evidence representations produced no prediction changes in the tested sample.

## Case Studies

The final analysis includes detailed case studies for queries `1320`, `1019`, `1370`, `1185`, and `314`. See [FINAL_REPORT.md](FINAL_REPORT.md), [final_case_studies.json](results/final_case_studies.json), and [final_case_study_analysis.json](results/final_case_study_analysis.json).

## Repository Structure

```text
data/
	processed/                 Prepared SciFact corpus and development queries
	index/                     Local FAISS index and ID map
experiments/
	run_rag_experiment.py      Day 1 baseline RAG
	test_retrieval.py          Retrieval checks
	analyze_ranking.py         Saved ranking analysis
	run_reranked_rag_experiment.py
															Day 2 reranked RAG
	test_reranking.py          Reranking checks
	test_evidence_extractor.py Day 3 extractor sanity test
	compare_evidence_extractors.py
															Original/diverse extractor comparison
	run_evidence_comparison.py
															Three-condition evidence comparison
	final_analysis.py          Offline cross-day consolidation
src/
	data/                      Dataset preparation
	retrieval/                 Embedding and FAISS retrieval
	rag/                       Generators and evidence extractors
results/                     Ignored generated outputs and reports
notebooks/                   Notebook workspace
docs/
	DAY2_REPORT.md
	DAY3_REPORT.md
```

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

Gemini generation requires the environment variable `GEMINI_API_KEY`. Never place the actual key in source code or commit it. Keep `.env` untracked.

## Reproducing the Research

The recommended high-level order is:

1. Prepare the dataset with `src/data/prepare_dataset.py`.
2. Inspect retrieval with `experiments/test_retrieval.py`.
3. Run the baseline retrieval/RAG experiment with `experiments/run_rag_experiment.py`.
4. Analyze saved ranks with `experiments/analyze_ranking.py`.
5. Run cross-encoder reranking with `experiments/run_reranked_rag_experiment.py`.
6. Run local evidence extraction and comparison with the Day 3 evidence scripts.
7. Run `experiments/final_analysis.py` for offline consolidation.

The RAG and reranking scripts consume Gemini API quota. Review their resume behavior, configuration, and expected call count before running them. The final reports and offline analyses do not require API calls.

## Results and Reports

- [FINAL_REPORT.md](FINAL_REPORT.md): complete project report
- [DAY2_REPORT.md](docs/DAY2_REPORT.md): reranking report
- [DAY3_REPORT.md](docs/DAY3_REPORT.md): evidence-aware report
- `results/`: detailed generated metrics, tables, case studies, and plots

## Limitations

The main evaluation is a controlled 50-query study. Day 3 uses only five queries. Day 2 contains Gemini free-tier quota failures and 401 raw records versus 400 unique combinations because of one duplicate record. The evaluation is descriptive, with no statistical significance testing or causal inference. Results depend on the selected models, reranker, prompt, and retrieval corpus.

## Reproducibility and Safety

- API keys and other secrets are not included.
- `.env` should remain untracked.
- Virtual environments and model caches should remain untracked.
- Generated experiment outputs should not be accidentally overwritten.
- Baseline, reranked, and evidence experiment outputs should be preserved before reruns.

## Research Conclusion

In this controlled evaluation, increasing retrieval depth improved observed baseline retrieval and verdict performance. Reranking was associated with stronger observed retrieval ranking and higher observed verdict accuracy at K=1, 3, and 5, while the K=10 retrieval result was unchanged. The small Day 3 comparison found identical predictions across full-document and extracted-evidence representations. These findings are descriptive and do not prove causality or universal improvement.

## Future Work

Potential extensions include larger evaluation sets, repeated runs, statistical testing, stronger rerankers, passage-level retrieval, improved evidence extraction, alternative LLMs, and latency/cost analysis. These were not implemented in this project.