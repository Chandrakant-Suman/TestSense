from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw")

FILES = {
    "Train": DATA_DIR / "train.csv",
    "Validation": DATA_DIR / "val.csv",
    "Test": DATA_DIR / "test.csv",
    "Out-of-distribution": DATA_DIR / "ood.csv",
}


def inspect_dataset(name, path):
    print("\n" + "=" * 65)
    print(f"{name.upper()} DATASET")
    print("=" * 65)

    if not path.exists():
        print(f"File not found: {path}")
        return

    try:
        df = pd.read_csv(path)

        print(f"\nFile: {path}")
        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")

        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nMissing values:")
        print(df.isna().sum().to_string())

        print("\nDuplicate rows:", df.duplicated().sum())

        for column in ["error_message", "category"]:
            if column in df.columns:
                print(f"\nUnique {column} values: {df[column].nunique()}")

        if "category" in df.columns:
            print("\nCategory distribution:")
            print(df["category"].value_counts(dropna=False).to_string())

        if "error_message" in df.columns:
            print("\nSample error messages:")
            print(
                df["error_message"]
                .dropna()
                .head(3)
                .to_string(index=False)
            )

    except Exception as exc:
        print(f"Could not inspect {path}: {exc}")


if __name__ == "__main__":
    for dataset_name, dataset_path in FILES.items():
        inspect_dataset(dataset_name, dataset_path)
