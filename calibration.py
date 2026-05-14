"""
Stretch Tuesday — Calibration Analysis.

Reliability diagram + Expected Calibration Error (ECE).
"""

import numpy as np


def reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Bin predictions by max predicted probability; compute empirical accuracy per bin.

    Returns (bucket_centers, bucket_accuracies, bucket_counts), all length n_bins.
    """

    # Max predicted probability for each sample
    confidences = np.max(probs, axis=1)

    # Predicted class index
    predictions = np.argmax(probs, axis=1)

    # Whether prediction is correct
    correct = predictions == y_true

    # Bin edges
    edges = np.linspace(0, 1, n_bins + 1)

    # Bucket centers (midpoints)
    bucket_centers = (edges[:-1] + edges[1:]) / 2

    bucket_accuracies = []
    bucket_counts = []

    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]

        # Last bin includes right edge
        if i == n_bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)

        bucket_count = np.sum(mask)

        if bucket_count > 0:
            bucket_accuracy = np.mean(correct[mask])
        else:
            bucket_accuracy = 0.0

        bucket_accuracies.append(bucket_accuracy)
        bucket_counts.append(bucket_count)

    return (
        np.array(bucket_centers),
        np.array(bucket_accuracies),
        np.array(bucket_counts)
    )


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE = sum over bins of (bucket_count / N) * |bucket_accuracy - bucket_confidence|.

    A perfectly calibrated model has ECE = 0.
    """

    confidences = np.max(probs, axis=1)

    predictions = np.argmax(probs, axis=1)

    correct = predictions == y_true

    edges = np.linspace(0, 1, n_bins + 1)

    N = len(y_true)

    ece = 0.0

    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]

        # Last bin inclusive on right edge
        if i == n_bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)

        bucket_count = np.sum(mask)

        if bucket_count > 0:
            bucket_accuracy = np.mean(correct[mask])

            bucket_confidence = np.mean(confidences[mask])

            ece += (
                (bucket_count / N)
                * abs(bucket_accuracy - bucket_confidence)
            )

    return float(ece)


def plot_reliability(centers: np.ndarray, accs: np.ndarray, counts: np.ndarray, output_path: str) -> None:
    """Save a reliability diagram. Provided helper — do not modify."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 1.0 / max(len(centers), 1)
    ax.bar(centers, accs, width=width * 0.9, edgecolor="black", alpha=0.8, label="Empirical accuracy")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (bucket center)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)



if __name__ == "__main__":

    import os
    import pandas as pd

    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification
    )

    from manual_eval import (
        manual_predict,
        compute_classification_report_from_arrays
    )

    MODEL_PATH = "./model"

    # Load model + tokenizer
    print("Loading model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    # Load dataset
    print("Loading dataset...")

    df = pd.read_csv("data/app_reviews_eval.csv")

    # Change column names if needed
    texts = df["text"].tolist()

    y_true = df["label"].to_numpy()

    # Manual inference
    print("Running manual prediction...")

    preds, probs = manual_predict(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        batch_size=8
    )

    # Classification metrics
    print("Computing classification metrics...")

    report = compute_classification_report_from_arrays(
        y_true=y_true,
        y_pred=preds
    )

    print("\nClassification Report:")
    print(report)

    # Reliability diagram
    print("\nGenerating reliability diagram...")

    centers, accs, counts = reliability_diagram(
        probs=probs,
        y_true=y_true,
        n_bins=10
    )

    # Expected Calibration Error
    print("Computing Expected Calibration Error (ECE)...")

    ece = expected_calibration_error(
        probs=probs,
        y_true=y_true,
        n_bins=10
    )

    print(f"\nECE: {ece:.4f}")

    # Save reliability plot
    os.makedirs("figures", exist_ok=True)

    output_path = "figures/reliability-diagram.png"

    plot_reliability(
        centers=centers,
        accs=accs,
        counts=counts,
        output_path=output_path
    )

    print(f"\nSaved reliability diagram to: {output_path}")