"""Generate research plots from the existing clean RAG metrics."""
import json
import os

import matplotlib.pyplot as plt


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS_PATH = os.path.join(ROOT, "results", "rag_metrics_clean.json")
PLOTS_DIR = os.path.join(ROOT, "results", "plots")
K_VALUES = [1, 3, 5, 10]
EXPECTED_HITS = [0.54, 0.70, 0.76, 0.90]
EXPECTED_ACCURACY = [0.62, 0.72, 0.74, 0.80]


def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as file_handle:
        metrics = json.load(file_handle)
    hits = [metrics["by_k"][str(k)]["retrieval_hit_at_k"] for k in K_VALUES]
    accuracy = [
        metrics["by_k"][str(k)]["verdict_accuracy_among_successful_api_calls"]
        for k in K_VALUES
    ]
    if hits != EXPECTED_HITS or accuracy != EXPECTED_ACCURACY:
        raise ValueError("Clean metrics do not match the verified baseline values")
    return hits, accuracy


def annotate_values(axis, values):
    for k_value, value in zip(K_VALUES, values):
        axis.annotate(f"{value:.2f}", (k_value, value), textcoords="offset points", xytext=(0, 7), ha="center")


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
    plt.close()


def main():
    hits, accuracy = load_metrics()
    os.makedirs(PLOTS_DIR, exist_ok=True)

    figure, axis = plt.subplots()
    axis.plot(K_VALUES, hits, marker="o")
    annotate_values(axis, hits)
    axis.set_xlabel("Retrieval depth (K)")
    axis.set_ylabel("Retrieval Hit@K")
    axis.set_title("Retrieval Performance vs Retrieval Depth")
    axis.set_xticks(K_VALUES)
    axis.set_ylim(0, 1)
    save_plot("retrieval_hit_at_k.png")

    figure, axis = plt.subplots()
    axis.plot(K_VALUES, accuracy, marker="o")
    annotate_values(axis, accuracy)
    axis.set_xlabel("Retrieval depth (K)")
    axis.set_ylabel("Verdict Accuracy")
    axis.set_title("RAG Verdict Accuracy vs Retrieval Depth")
    axis.set_xticks(K_VALUES)
    axis.set_ylim(0, 1)
    save_plot("verdict_accuracy_at_k.png")

    figure, axis = plt.subplots()
    axis.plot(K_VALUES, hits, marker="o", label="Retrieval Hit@K")
    axis.plot(K_VALUES, accuracy, marker="o", label="Verdict Accuracy")
    annotate_values(axis, hits)
    annotate_values(axis, accuracy)
    axis.set_xlabel("Retrieval depth (K)")
    axis.set_ylabel("Score")
    axis.set_title("Retrieval and Verdict Performance vs K")
    axis.set_xticks(K_VALUES)
    axis.set_ylim(0, 1)
    axis.legend()
    save_plot("retrieval_vs_verdict.png")

    print(f"Created plots in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
