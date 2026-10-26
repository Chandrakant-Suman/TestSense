from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline


DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(filename):
    df = pd.read_csv(DATA_DIR / filename)

    if df["error_message"].isna().any():
        raise ValueError(f"{filename} contains missing messages.")

    if df["category"].isna().any():
        raise ValueError(f"{filename} contains missing categories.")

    return df


def build_baseline():
    return Pipeline([
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


def build_combined_model():
    features = FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                analyzer="word",
                lowercase=True,
                ngram_range=(1, 2),
                max_features=50000,
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                lowercase=True,
                ngram_range=(3, 5),
                max_features=50000,
                min_df=2,
                sublinear_tf=True,
            ),
        ),
    ])

    return Pipeline([
        ("features", features),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    print(f"\n{'=' * 60}")
    print(name)
    print("=" * 60)

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_val,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            predictions,
            zero_division=0,
        )
    )

    return {
        "name": name,
        "model": model,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def main():
    train_df = load_data("train.csv")
    val_df = load_data("val.csv")

    X_train = train_df["error_message"].astype(str)
    y_train = train_df["category"].astype(str)

    X_val = val_df["error_message"].astype(str)
    y_val = val_df["category"].astype(str)

    results = []

    results.append(
        evaluate_model(
            "Baseline: Word TF-IDF",
            build_baseline(),
            X_train,
            y_train,
            X_val,
            y_val,
        )
    )

    results.append(
        evaluate_model(
            "Candidate: Word + Character TF-IDF",
            build_combined_model(),
            X_train,
            y_train,
            X_val,
            y_val,
        )
    )

    comparison = pd.DataFrame([
        {
            "model": result["name"],
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "weighted_f1": result["weighted_f1"],
        }
        for result in results
    ])

    comparison.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False))

    # Select the candidate using validation macro F1.
    best = max(results, key=lambda result: result["macro_f1"])

    joblib.dump(
        best["model"],
        MODEL_DIR / "testsense_best.joblib",
    )

    print(f"\nBest validation model: {best['name']}")
    print(f"Best validation macro F1: {best['macro_f1']:.4f}")
    print("Saved to models/testsense_best.joblib")
    print("Saved comparison to reports/model_comparison.csv")


if __name__ == "__main__":
    main()