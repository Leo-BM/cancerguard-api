import pytest
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="session", autouse=True)
def mock_model_globals():
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])

    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 30))

    rng = np.random.default_rng(seed=42)
    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = [rng.standard_normal((1, 30))]

    mock_logger = MagicMock()

    with (
        patch("app.model._model", mock_model),
        patch("app.model._scaler", mock_scaler),
        patch("app.model._explainer", mock_explainer),
        patch("app.model.load_model"),
        patch("app.main.prediction_logger", mock_logger),
    ):
        yield
