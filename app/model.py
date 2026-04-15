import os
import joblib
import numpy as np
import shap
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient


_model:     object = None
_scaler:    object = None
_explainer: object = None

FEATURE_NAMES = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
    "mean_smoothness", "mean_compactness", "mean_concavity",
    "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area",
    "worst_smoothness", "worst_compactness", "worst_concavity",
    "worst_concave_points", "worst_symmetry", "worst_fractal_dimension",
]


def load_model() -> None:
    global _model, _scaler, _explainer

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:

        import mlflow.sklearn
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)

        _model = mlflow.sklearn.load_model("models:/cancerguard-svm/Production")

        client = MlflowClient()
        versions = client.get_latest_versions("cancerguard-svm", stages=["Production"])
        run_id = versions[0].run_id
        local_scaler_path = client.download_artifacts(run_id, "scaler.joblib")
        _scaler = joblib.load(local_scaler_path)

    else:

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path  = os.path.join(base_dir, "model.joblib")
        scaler_path = os.path.join(base_dir, "scaler.joblib")

        _model  = joblib.load(model_path)
        _scaler = joblib.load(scaler_path)

    background = np.zeros((1, len(FEATURE_NAMES)))
    _explainer = shap.KernelExplainer(_model.predict_proba, background)


def _get_risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "high"
    elif prob >= 0.4:
        return "medium"
    return "low"


def predict(input_data: dict) -> dict:
    X = np.array([[input_data[f] for f in FEATURE_NAMES]])

    X_scaled = _scaler.transform(X)


    proba = _model.predict_proba(X_scaled)[0]
    prob_malignant = float(proba[0])
    prediction = "malignant" if prob_malignant >= 0.5 else "benign"

    shap_values = _explainer.shap_values(X_scaled)
    importances = list(zip(FEATURE_NAMES, shap_values[0][0]))

    importances.sort(key=lambda x: abs(x[1]), reverse=True)

    top_features = [
        {"feature": f, "shap_value": round(float(v), 4)}
        for f, v in importances[:3]
    ]

    return {
        "prediction": prediction,
        "probability_malignant": round(prob_malignant, 4),
        "risk_level": _get_risk_level(prob_malignant),
        "top_features": top_features,
    }
