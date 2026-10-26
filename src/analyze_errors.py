from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

DATA_DIR = Path("data/raw")
MODEL_PATH = Path("models/testsense_baseline.joblib")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading model and validation dataset...")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_DIR / "val.csv")

    X = df["error_message"].astype(str)
    y_true = df["category"].astype(str)
    y_pred = model.predict(X)

    # Save every incorrect prediction for manual investigation.
    errors = pd.DataFrame({
        "error_message": X,
        "actual_category": y_true,
        "predicted_category": y_pred,
    })

    errors = errors[errors["actual_category"] != errors["predicted_category"]]
    errors.to_csv(REPORT_DIR / "misclassified_examples.csv", index=False)

    print(f"\nIncorrect predictions: {len(errors)} / {len(df)}")
    print(f"Saved examples to: {REPORT_DIR / 'misclassified_examples.csv'}")

    print("\nMost common classification mistakes:")
    if not errors.empty:
        mistakes = (
            errors.groupby(["actual_category", "predicted_category"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        print(mistakes.head(20).to_string(index=False))

    # Display categories with the most missed examples.
    print("\nCategories with the most missed examples:")
    missed = errors["actual_category"].value_counts()
    print(missed.head(10).to_string())

    # Generate a confusion matrix.
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    ).to_csv(REPORT_DIR / "error_analysis_confusion_matrix.csv")

    print("\nSaved confusion matrix to reports/error_analysis_confusion_matrix.csv")


if __name__ == "__main__":
    main()