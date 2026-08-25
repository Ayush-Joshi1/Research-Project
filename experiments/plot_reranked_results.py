"""Plot and summarize completed reranked RAG analysis outputs."""
import csv
import json
import os

import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
ANALYSIS_PATH = os.path.join(RESULTS_DIR, "reranked_rag_analysis.json")
CASE_STUDY_PATH = os.path.join(RESULTS_DIR, "final_case_study_analysis.json")
PLOT_PATHS = {
    "retrieval": os.path.join(RESULTS_DIR, "reranked_retrieval_hit_at_k.png"),
    "verdict": os.path.join(RESULTS_DIR, "reranked_verdict_accuracy_at_k.png"),
    "comparison": os.path.join(RESULTS_DIR, "reranked_comparison.png"),
    "improvement": os.path.join(RESULTS_DIR, "reranked_improvement.png"),
}
TABLE_PATH = os.path.join(RESULTS_DIR, "day2_final_metrics_table.csv")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "day2_final_summary.txt")
K_VALUES = (1, 3, 5, 10)


def load_inputs():
    with open(ANALYSIS_PATH, "r", encoding="utf-8") as handle:
        analysis = json.load(handle)
    with open(CASE_STUDY_PATH, "r", encoding="utf-8") as handle:
        case_studies = json.load(handle)
    return analysis, case_studies


def values(analysis, field):
    return {
        "baseline": [analysis["metrics_by_condition_and_K"][f"baseline_k{k}"][field] for k in K_VALUES],
        "reranked": [analysis["metrics_by_condition_and_K"][f"reranked_k{k}"][field] for k in K_VALUES],
    }


def annotate(ax, x_values, y_values, precision=2):
    for x_value, y_value in zip(x_values, y_values):
        if y_value is not None:
            ax.annotate(f"{y_value:.{precision}f}", (x_value, y_value), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)


def style_axes(ax, title, ylabel):
    ax.set_title(title)
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.set_xticks(K_VALUES)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.figure.tight_layout()


def save_line_plot(path, title, ylabel, series):
    figure, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_VALUES, series["baseline"], marker="o", linewidth=2, label="Baseline")
    ax.plot(K_VALUES, series["reranked"], marker="o", linewidth=2, label="Reranked")
    annotate(ax, K_VALUES, series["baseline"])
    annotate(ax, K_VALUES, series["reranked"])
    style_axes(ax, title, ylabel)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_metrics_table(analysis):
    rows = []
    for k_value in K_VALUES:
        baseline = analysis["metrics_by_condition_and_K"][f"baseline_k{k_value}"]
        reranked = analysis["metrics_by_condition_and_K"][f"reranked_k{k_value}"]
        rows.append({
            "K": k_value,
            "baseline_hit": baseline["retrieval_hit_at_k"],
            "reranked_hit": reranked["retrieval_hit_at_k"],
            "retrieval_delta": reranked["retrieval_hit_at_k"] - baseline["retrieval_hit_at_k"],
            "baseline_verdict_accuracy": baseline["verdict_accuracy_among_successful_api_calls"],
            "reranked_verdict_accuracy": reranked["verdict_accuracy_among_successful_api_calls"],
            "verdict_delta": reranked["verdict_accuracy_among_successful_api_calls"] - baseline["verdict_accuracy_among_successful_api_calls"],
        })
    fields = list(rows[0])
    with open(TABLE_PATH, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_summary(analysis, case_studies, table_rows):
    validation = analysis["validation"]
    lines = [
        "DAY 2 FINAL BASELINE VS RERANKED RAG SUMMARY",
        "",
        "This is a descriptive summary of the stored analysis outputs. It does not make statistical significance or causal claims.",
        "",
        "MAIN RETRIEVAL FINDINGS",
        "Reranking was associated with higher observed retrieval Hit@K at K=1, K=3, and K=5: deltas were +0.080, +0.140, and +0.120, respectively.",
        "At K=10, retrieval Hit@K was identical for baseline and reranked conditions at 0.900 because both use the same FAISS top-10 candidate set, only reordered.",
        "",
        "MAIN VERDICT FINDINGS",
        "Among successful API calls, reranked verdict accuracy was higher at K=1, K=3, and K=5, with observed deltas of +0.088, +0.042, and +0.024.",
        "At K=10, reranked verdict accuracy was 0.800 versus 0.805 for baseline, an observed delta of -0.005.",
        "Verdict accuracy excludes failed API calls. The experiment contains 71 API failures.",
        "Aggregate metrics use one row per unique condition/query/K combination: the raw experiment contains 401 records but 400 unique combinations because of one duplicate.",
        "",
        "K=10 OBSERVATION",
        "At K=10, baseline and reranked have the same retrieval Hit@10 (0.900). The candidate documents are the same FAISS top-10 documents and differ only in order. The observed verdict accuracy difference is -0.005 for reranked minus baseline; this is descriptive and does not establish causality.",
        "",
        "CASE-STUDY SUMMARY",
    ]
    for query_id in ("1320", "1019", "1370", "1185", "314"):
        lines.append(case_studies["case_study_findings"][query_id])
    lines.extend([
        "",
        "DATA QUALITY",
        f"Raw records: {validation['total_records']}; unique combinations: {validation['unique_condition_query_K_combinations']}; duplicate combinations: {validation['duplicate_combinations']}; missing combinations: {len(validation['missing_combinations'])}.",
        "Query-level verdict conclusions for 1185 and 314 are limited because they do not have sufficient successful paired API results for a complete verdict comparison.",
        "",
        "METRICS TABLE",
    ])
    for row in table_rows:
        lines.append(
            f"K={row['K']}: retrieval {row['baseline_hit']:.3f} -> {row['reranked_hit']:.3f} ({row['retrieval_delta']:+.3f}); "
            f"verdict {row['baseline_verdict_accuracy']:.3f} -> {row['reranked_verdict_accuracy']:.3f} ({row['verdict_delta']:+.3f})"
        )
    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    analysis, case_studies = load_inputs()
    retrieval = values(analysis, "retrieval_hit_at_k")
    verdict = values(analysis, "verdict_accuracy_among_successful_api_calls")
    comparison = {
        "baseline": retrieval["baseline"],
        "reranked": retrieval["reranked"],
    }
    comparison["baseline_verdict"] = verdict["baseline"]
    comparison["reranked_verdict"] = verdict["reranked"]
    improvements = {
        "retrieval": [
            analysis["metrics_by_condition_and_K"][f"reranked_k{k}"]["retrieval_hit_at_k"]
            - analysis["metrics_by_condition_and_K"][f"baseline_k{k}"]["retrieval_hit_at_k"]
            for k in K_VALUES
        ],
        "verdict": [
            analysis["metrics_by_condition_and_K"][f"reranked_k{k}"]["verdict_accuracy_among_successful_api_calls"]
            - analysis["metrics_by_condition_and_K"][f"baseline_k{k}"]["verdict_accuracy_among_successful_api_calls"]
            for k in K_VALUES
        ],
    }

    save_line_plot(PLOT_PATHS["retrieval"], "Retrieval Hit@K", "Proportion", retrieval)
    save_line_plot(PLOT_PATHS["verdict"], "Verdict Accuracy Among Successful API Calls", "Accuracy", verdict)

    figure, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_VALUES, comparison["baseline"], marker="o", linewidth=2, label="Baseline Retrieval Hit@K")
    ax.plot(K_VALUES, comparison["reranked"], marker="o", linewidth=2, label="Reranked Retrieval Hit@K")
    ax.plot(K_VALUES, comparison["baseline_verdict"], marker="s", linestyle="--", label="Baseline Verdict Accuracy")
    ax.plot(K_VALUES, comparison["reranked_verdict"], marker="s", linestyle="--", label="Reranked Verdict Accuracy")
    style_axes(ax, "Baseline vs Reranked Comparison", "Proportion / Accuracy")
    figure.savefig(PLOT_PATHS["comparison"], dpi=160)
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.plot(K_VALUES, improvements["retrieval"], marker="o", linewidth=2, label="Retrieval improvement")
    ax.plot(K_VALUES, improvements["verdict"], marker="s", linewidth=2, label="Verdict improvement")
    annotate(ax, K_VALUES, improvements["retrieval"], 3)
    annotate(ax, K_VALUES, improvements["verdict"], 3)
    ax.set_title("Reranked Minus Baseline Improvement")
    ax.set_xlabel("K")
    ax.set_ylabel("Delta (proportion / accuracy)")
    ax.set_xticks(K_VALUES)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    figure.tight_layout()
    figure.savefig(PLOT_PATHS["improvement"], dpi=160)
    plt.close(figure)

    table_rows = write_metrics_table(analysis)
    write_summary(analysis, case_studies, table_rows)
    for path in (*PLOT_PATHS.values(), TABLE_PATH, SUMMARY_PATH):
        print(path)


if __name__ == "__main__":
    main()
