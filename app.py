import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "testsense_best.joblib"
REPORT_PATH = ROOT / "reports" / "test_report.json"
FIGURES_DIR = ROOT / "reports" / "figures"

st.set_page_config(
    page_title="TestSense AI",
    page_icon="🧪",
    layout="wide",
)

# --------------------------------------------------
# Load existing model and evaluation report
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_report():
    with open(REPORT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    model = load_model()
    report = load_report()
except Exception as exc:
    st.error(f"Could not load the model or report: {exc}")
    st.stop()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🧪 TestSense")
st.subheader("Intelligent CI/CD Error Classification")

st.write(
    "Classify CI/CD error messages, inspect model predictions, "
    "and identify cases that may require human review."
)

st.info(
    "This is a classification assistant, not an autonomous repair system. "
    "Review suggested categories before taking corrective action."
)

# --------------------------------------------------
# Sidebar settings
# --------------------------------------------------

st.sidebar.header("Settings")

review_threshold = st.sidebar.slider(
    "Manual-review threshold",
    min_value=0.30,
    max_value=0.95,
    value=0.60,
    step=0.05,
    help=(
        "Flag predictions below this maximum class-probability score. "
        "This is a heuristic, not a calibrated probability of correctness."
    ),
)

st.sidebar.caption(
    "A low score does not necessarily mean the prediction is wrong. "
    "A high score does not guarantee correctness."
)

# --------------------------------------------------
# Error classification
# --------------------------------------------------

st.header("1. Classify an Error")

error_message = st.text_area(
    "Paste a CI/CD error message",
    placeholder=(
        "Example: Connection timed out while connecting "
        "to the database server"
    ),
    height=140,
)

if st.button("Analyze Error", type="primary"):
    if not error_message.strip():
        st.warning("Please enter an error message first.")
    else:
        try:
            probabilities = model.predict_proba([error_message])[0]
            classes = model.classes_

            ranking = (
                pd.DataFrame(
                    {
                        "Category": classes,
                        "Score": probabilities,
                    }
                )
                .sort_values("Score", ascending=False)
                .reset_index(drop=True)
            )

            predicted_category = ranking.iloc[0]["Category"]
            top_score = float(ranking.iloc[0]["Score"])

            st.session_state["analysis"] = {
                "message": error_message,
                "category": predicted_category,
                "score": top_score,
                "ranking": ranking,
            }

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# Display the latest prediction
analysis = st.session_state.get("analysis")

if analysis:
    st.divider()
    st.header("2. Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Category", analysis["category"])

    with col2:
        st.metric("Highest Class Score", f'{analysis["score"]:.1%}')

    if analysis["score"] < review_threshold:
        st.warning(
            "Manual review recommended: the highest class score "
            "is below your selected threshold."
        )
    else:
        st.success(
            "The highest class score meets your selected threshold. "
            "Verify the prediction before acting on it."
        )

    st.caption(
        "Scores are model outputs and have not been established as "
        "calibrated probabilities of correctness."
    )

    st.subheader("Top 5 Candidate Categories")

    top_five = analysis["ranking"].head(5).copy()
    top_five["Score"] = top_five["Score"].map(
        lambda value: f"{value:.2%}"
    )
    st.dataframe(top_five, use_container_width=True, hide_index=True)

    chart_data = analysis["ranking"].head(5).sort_values("Score")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(chart_data["Category"], chart_data["Score"])
    ax.set_xlabel("Model score")
    ax.set_title("Top 5 Predicted Categories")
    ax.set_xlim(0, 1)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Download Prediction as CSV",
        data=analysis["ranking"].to_csv(index=False).encode("utf-8"),
        file_name="testsense_prediction.csv",
        mime="text/csv",
    )

# --------------------------------------------------
# Model performance
# --------------------------------------------------

st.divider()
st.header("3. Test-Set Performance")

# Supports common report formats without assuming a particular schema.
metrics = report.get("metrics", report)

accuracy = metrics.get("accuracy")
macro_f1 = metrics.get("macro_f1")
weighted_f1 = metrics.get("weighted_f1")

metric_columns = st.columns(3)

for column, label, value in zip(
    metric_columns,
    ["Accuracy", "Macro F1", "Weighted F1"],
    [accuracy, macro_f1, weighted_f1],
):
    with column:
        if isinstance(value, (int, float)):
            st.metric(label, f"{value:.2%}")
        else:
            st.metric(label, "See report")

with st.expander("View complete test report"):
    st.json(report)

st.caption(
    "These metrics describe performance on the provided test split. "
    "They do not guarantee performance on production data."
)

# --------------------------------------------------
# Visualizations
# --------------------------------------------------

st.divider()
st.header("4. Model Analytics")

figure_files = {
    "Confidence Distribution": "confidence_distribution.png",
    "Accuracy by Confidence Band": "accuracy_by_confidence.png",
    "OOD Category Distribution": "ood_category_distribution.png",
    "OOD Prediction Distribution": "ood_prediction_distribution.png",
}

for title, filename in figure_files.items():
    image_path = FIGURES_DIR / filename

    if image_path.exists():
        st.subheader(title)
        st.image(str(image_path), use_container_width=True)
    else:
        st.warning(f"Visualization not found: {filename}")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()
st.caption(
    "TestSense AI | Experimental CI/CD error classification "
    "and decision support"
)