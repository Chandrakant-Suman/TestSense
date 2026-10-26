from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

DATA_DIR = Path("data/raw")
MODEL_PATH = Path("models/testsense_best.joblib")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading model and OOD dataset...")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_DIR / "ood.csv")

    required = {"error_message", "category"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"OOD dataset missing columns: {missing}")

    df = df.dropna(subset=["error_message", "category"]).copy()

    X = df["error_message"].astype(str)
    y_true = df["category"].astype(str)

    # These are the categories learned during training.
    known_categories = set(model.classes_)

    # Predict each OOD message.
    predictions = model.predict(X)

    results = df.copy()
    results["predicted_category"] = predictions
    results["category_seen_during_training"] = (
        y_true.isin(known_categories)
    )
    results["prediction_matches_actual"] = (
        y_true == results["predicted_category"]
    )

    # Save all predictions, including the source metadata.
    results.to_csv(
        REPORT_DIR / "ood_predictions.csv",
        index=False,
    )

    known_mask = results["category_seen_during_training"]
    unknown_mask = ~known_mask

    print(f"\nTotal OOD examples: {len(results)}")
    print(f"Known-category examples: {known_mask.sum()}")
    print(f"Unseen-category examples: {unknown_mask.sum()}")

    print("\nActual category distribution:")
    print(results["category"].value_counts().to_string())

    print("\nPredicted category distribution:")
    print(
        results["predicted_category"]
        .value_counts()
        .to_string()
    )

    # Evaluate accuracy only where the actual category is known.
    if known_mask.any():
        y_known = results.loc[known_mask, "category"]
        p_known = results.loc[known_mask, "predicted_category"]

        print("\nKNOWN-CATEGORY OOD RESULTS")
        print("-" * 45)
        print(f"Accuracy: {accuracy_score(y_known, p_known):.4f}")
        print(
            "Macro F1:",
            f"{f1_score(y_known, p_known, average='macro', zero_division=0):.4f}",
        )

        print("\nClassification report:")
        print(
            classification_report(
                y_known,
                p_known,
                labels=sorted(known_categories),
                zero_division=0,
            )
        )

    # For unseen labels, ordinary closed-set classification cannot
    # predict the correct label because the model never learned it.
    if unknown_mask.any():
        unseen_results = results.loc[unknown_mask]

        print("\nUNSEEN-CATEGORY EXAMPLES")
        print("-" * 45)
        print(
            "The model cannot predict these actual categories as labels "
            "because they were absent from training."
        )

        print("\nUnseen actual categories:")
        print(
            unseen_results["category"]
            .value_counts()
            .to_string()
        )

        print("\nWhere these examples were classified:")
        print(
            unseen_results["predicted_category"]
            .value_counts()
            .to_string()
        )

    print("\nSaved predictions to reports/ood_predictions.csv")


if __name__ == "__main__":
    main()