from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import accuracy_score

DATA_DIR = Path("data/raw")
MODEL_PATH = Path("models/testsense_best.joblib")
REPORT_DIR = Path("reports")
FIGURES_DIR = REPORT_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_chart(filename):
    """Save the current Matplotlib chart."""
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved chart: {FIGURES_DIR / filename}")


def main():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_DIR / "ood.csv")

    X = df["error_message"].astype(str)
    y_true = df["category"].astype(str)

    probabilities = model.predict_proba(X)
    predicted_indices = probabilities.argmax(axis=1)

    predictions = model.classes_[predicted_indices]
    confidence = probabilities.max(axis=1)

    known_categories = set(model.classes_)
    known_mask = y_true.isin(known_categories)

    results = pd.DataFrame({
        "error_message": X,
        "actual_category": y_true,
        "predicted_category": predictions,
        "confidence_score": confidence,
        "actual_category_seen_in_training": known_mask,
    })

    # This is a review heuristic, not a validated unknown detector.
    results["needs_review"] = (
        (results["confidence_score"] < 0.60)
        | (~results["actual_category_seen_in_training"])
    )

    results.to_csv(
        REPORT_DIR / "ood_confidence_analysis.csv",
        index=False,
    )

    print("\nOOD CONFIDENCE ANALYSIS")
    print("-" * 45)
    print(f"Total examples: {len(results)}")
    print(f"Mean confidence: {confidence.mean():.4f}")
    print(f"Median confidence: {results['confidence_score'].median():.4f}")
    print(
        f"Flagged for review: {results['needs_review'].sum()} "
        f"/ {len(results)}"
    )

    # Chart 1: Distribution of confidence scores.
    plt.figure(figsize=(9, 5))
    plt.hist(confidence, bins=20, edgecolor="black")
    plt.title("OOD Prediction Confidence Distribution")
    plt.xlabel("Maximum predicted class probability")
    plt.ylabel("Number of messages")
    save_chart("confidence_distribution.png")

    # Chart 2: Actual category frequencies.
    plt.figure(figsize=(11, 7))
    results["actual_category"].value_counts().sort_values().plot(
        kind="barh"
    )
    plt.title("Actual OOD Category Distribution")
    plt.xlabel("Number of messages")
    plt.ylabel("Actual category")
    save_chart("ood_category_distribution.png")

    # Chart 3: Predicted category frequencies.
    plt.figure(figsize=(11, 7))
    results["predicted_category"].value_counts().sort_values().plot(
        kind="barh"
    )
    plt.title("Predicted OOD Category Distribution")
    plt.xlabel("Number of messages")
    plt.ylabel("Predicted category")
    save_chart("ood_prediction_distribution.png")

    # Chart 4: Accuracy by confidence band, known categories only.
    known_results = results[known_mask].copy()

    bins = [0, 0.40, 0.60, 0.80, 1.000001]
    labels = ["0–0.40", "0.40–0.60", "0.60–0.80", "0.80–1.00"]

    known_results["confidence_band"] = pd.cut(
        known_results["confidence_score"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    band_accuracy = {}
    band_counts = {}

    for band in labels:
        group = known_results[
            known_results["confidence_band"] == band
        ]

        if len(group):
            band_accuracy[band] = accuracy_score(
                group["actual_category"],
                group["predicted_category"],
            )
            band_counts[band] = len(group)

    print("\nAccuracy by confidence band (known categories):")
    for band in labels:
        if band in band_accuracy:
            print(
                f"{band}: accuracy={band_accuracy[band]:.4f}, "
                f"count={band_counts[band]}"
            )

    if band_accuracy:
        plt.figure(figsize=(9, 5))
        plt.bar(
            list(band_accuracy.keys()),
            list(band_accuracy.values()),
            edgecolor="black",
        )
        plt.ylim(0, 1)
        plt.title("Accuracy by Confidence Band — Known OOD Categories")
        plt.xlabel("Maximum predicted class probability")
        plt.ylabel("Accuracy")
        plt.savefig(
            FIGURES_DIR / "accuracy_by_confidence.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        print(
            "Saved chart:",
            FIGURES_DIR / "accuracy_by_confidence.png",
        )

    print("\nSaved predictions to reports/ood_confidence_analysis.csv")
    print("All charts saved in reports/figures/")


if __name__ == "__main__":
    main()