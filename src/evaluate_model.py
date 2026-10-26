from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

DATA_DIR = Path("data/raw")
MODEL_PATH = Path("models/testsense_best.joblib")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading selected model and test dataset...")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_DIR / "test.csv")

    X_test = df["error_message"].astype(str)
    y_test = df["category"].astype(str)

    # The test set is used only for final evaluation.
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    print("\nFINAL TEST RESULTS")
    print("-" * 45)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test, predictions, zero_division=0
        )
    )

    # Save the metrics.
    with (REPORT_DIR / "test_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "classification_report": report,
            },
            file,
            indent=2,
        )

    # Save individual predictions.
    results = pd.DataFrame({
        "error_message": X_test,
        "actual_category": y_test,
        "predicted_category": predictions,
    })

    results.to_csv(
        REPORT_DIR / "test_predictions.csv", index=False
    )

    # Save confusion matrix.
    labels = sorted(set(y_test) | set(predictions))
    matrix = confusion_matrix(
        y_test, predictions, labels=labels
    )

    pd.DataFrame(
        matrix, index=labels, columns=labels
    ).to_csv(REPORT_DIR / "test_confusion_matrix.csv")

    print("\nReports saved:")
    print(" - reports/test_report.json")
    print(" - reports/test_predictions.csv")
    print(" - reports/test_confusion_matrix.csv")


if __name__ == "__main__":
    main()