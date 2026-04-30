import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


MODEL_PATH = "models/ldha_model.pkl"
FEATURE_PATH = "models/feature_names.pkl"


def train_demo_model():
    """
    Train a demo model so the app works immediately.
    Replace this later with your real LightGBM model trained on ChEMBL, BindingDB, and COCONUT data.
    """

    np.random.seed(42)

    feature_names = [
        "Molecular Weight",
        "LogP",
        "H-Bond Donors",
        "H-Bond Acceptors",
        "TPSA",
        "Rotatable Bonds",
        "QED",
    ]

    n = 300

    X = pd.DataFrame({
        "Molecular Weight": np.random.normal(350, 80, n),
        "LogP": np.random.normal(3, 1.2, n),
        "H-Bond Donors": np.random.randint(0, 6, n),
        "H-Bond Acceptors": np.random.randint(2, 12, n),
        "TPSA": np.random.normal(80, 30, n),
        "Rotatable Bonds": np.random.randint(0, 12, n),
        "QED": np.random.uniform(0.2, 0.95, n),
    })

    y = (
        (X["QED"] > 0.55)
        & (X["LogP"] > 1.2)
        & (X["LogP"] < 4.8)
        & (X["Molecular Weight"] < 520)
        & (X["TPSA"] < 130)
    ).astype(int)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X, y)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_names, FEATURE_PATH)

    return model, feature_names


def load_or_train_model():
    """Load existing model or train a demo model."""
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_PATH):
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURE_PATH)
    else:
        model, feature_names = train_demo_model()

    return model, feature_names


def build_feature_frame(props: dict, feature_names: list):
    """Convert calculated molecular properties into model-ready feature table."""
    row = {}

    for feature in feature_names:
        row[feature] = props.get(feature, 0)

    return pd.DataFrame([row])


def predict_ldha_probability(model, X):
    """Predict LDHA inhibition probability."""
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0, 1])

    prediction = model.predict(X)[0]
    return float(prediction)
