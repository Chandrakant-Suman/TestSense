from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


# Project paths
DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required_columns = {"error_message", "category"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{filename} is missing columns: {missing_columns}"
        )

    if df["error_message"].isna().any():
        raise ValueError(f"{filename} contains missing error messages.")

    if df["category"].isna().any():
        raise ValueError(f"{filename} contains missing categories.")

    return df


def main():
    print("Loading datasets...")

    train_df = load_dataset("train.csv")
    val_df = load_dataset("val.csv")
    test_df = load_dataset("test.csv")

    # Check for exact message overlap across splits.
    def normalize_messages(series):
        return set(series.astype(str).str.strip().str.lower())

    train_messages = normalize_messages(train_df["error_message"])
    val_messages = normalize_messages(val_df["error_message"])
    test_messages = normalize_messages(test_df["error_message"])

    print("\nExact message overlap:")
    print("Train vs validation:", len(train_messages & val_messages))
    print("Train vs test:", len(train_messages & test_messages))
    print("Validation vs test:", len(val_messages & test_messages))

    # Keep only the message as input; category is the target.
    X_train = train_df["error_message"].astype(str)
    y_train = train_df["category"].astype(str)

    X_val = val_df["error_message"].astype(str)
    y_val = val_df["category"].astype(str)

    # Establish a baseline NLP classifier.
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=50000,
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    print("\nTraining TF-IDF + Logistic Regression...")
    model.fit(X_train, y_train)

    # Evaluate on validation data only.
    print("\nEvaluating on validation data...")
    val_predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, val_predictions)

    report = classification_report(
        y_val,
        val_predictions,
        output_dict=True,
        zero_division=0,
    )

    print(f"\nValidation accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            val_predictions,
            zero_division=0,
        )
    )

    # Save model and evaluation reports.
    model_path = MODEL_DIR / "testsense_baseline.joblib"
    joblib.dump(model, model_path)

    report_path = REPORT_DIR / "validation_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "validation_accuracy": accuracy,
                "classification_report": report,
            },
            file,
            indent=2,
        )

    matrix = confusion_matrix(
        y_val,
        val_predictions,
        labels=model.classes_,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=model.classes_,
        columns=model.classes_,
    )
    matrix_df.to_csv(REPORT_DIR / "validation_confusion_matrix.csv")

    # Show a few predictions for manual inspection.
    examples = pd.DataFrame({
        "error_message": X_val.iloc[:10].values,
        "actual_category": y_val.iloc[:10].values,
        "predicted_category": val_predictions[:10],
    })
    examples.to_csv(
        REPORT_DIR / "sample_predictions.csv",
        index=False,
    )

    print("\nSaved model:", model_path)
    print("Saved validation report:", report_path)
    print("Saved confusion matrix:", REPORT_DIR / "validation_confusion_matrix.csv")
    print("Saved sample predictions:", REPORT_DIR / "sample_predictions.csv")
    print("\nBaseline training complete!")


if __name__ == "__main__":
    main()
