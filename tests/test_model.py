
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "testsense_best.joblib"


def test_model_file_exists():
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"


def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_predicts_category():
    model = joblib.load(MODEL_PATH)

    message = "ModuleNotFoundError: No module named 'requests'"
    prediction = model.predict([message])

    assert len(prediction) == 1
    assert prediction[0] in model.classes_


def test_model_predict_proba():
    model = joblib.load(MODEL_PATH)

    message = "ModuleNotFoundError: No module named 'requests'"
    probabilities = model.predict_proba([message])

    assert probabilities.shape == (1, len(model.classes_))
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()
    assert abs(probabilities.sum() - 1.0) < 1e-6
