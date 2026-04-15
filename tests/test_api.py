import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "mean_radius": 14.0,
    "mean_texture": 19.0,
    "mean_perimeter": 91.0,
    "mean_area": 654.0,
    "mean_smoothness": 0.095,
    "mean_compactness": 0.109,
    "mean_concavity": 0.112,
    "mean_concave_points": 0.074,
    "mean_symmetry": 0.181,
    "mean_fractal_dimension": 0.057,
    "radius_se": 0.37,
    "texture_se": 0.87,
    "perimeter_se": 2.64,
    "area_se": 28.0,
    "smoothness_se": 0.005,
    "compactness_se": 0.021,
    "concavity_se": 0.032,
    "concave_points_se": 0.011,
    "symmetry_se": 0.019,
    "fractal_dimension_se": 0.003,
    "worst_radius": 16.0,
    "worst_texture": 25.0,
    "worst_perimeter": 105.0,
    "worst_area": 819.0,
    "worst_smoothness": 0.132,
    "worst_compactness": 0.240,
    "worst_concavity": 0.290,
    "worst_concave_points": 0.140,
    "worst_symmetry": 0.290,
    "worst_fractal_dimension": 0.082,
}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_response_has_status_healthy(self):
        response = client.get("/health")
        assert response.json()["status"] == "healthy"

    def test_response_has_model_field(self):
        response = client.get("/health")
        assert "model" in response.json()

    def test_response_has_version_field(self):
        response = client.get("/health")
        assert "version" in response.json()


class TestPredictValidation:
    def test_missing_field_returns_422(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "mean_radius"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_wrong_type_returns_422(self):
        payload = {**VALID_PAYLOAD, "mean_radius": "not_a_number"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_negative_value_returns_422(self):
        payload = {**VALID_PAYLOAD, "mean_radius": -5.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_zero_value_returns_422(self):
        payload = {**VALID_PAYLOAD, "mean_radius": 0.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_smoothness_above_1_returns_422(self):
        payload = {**VALID_PAYLOAD, "mean_smoothness": 1.5}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_empty_body_returns_422(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422


class TestPredictValidInput:
    def test_valid_payload_returns_200(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_response_has_prediction_field(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert "prediction" in response.json()

    def test_response_has_probability_field(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert "probability_malignant" in response.json()

    def test_response_has_risk_level_field(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert "risk_level" in response.json()

    def test_response_has_top_features_field(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert "top_features" in response.json()

    def test_prediction_is_valid_class(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.json()["prediction"] in ("malignant", "benign")

    def test_probability_is_between_0_and_1(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        prob = response.json()["probability_malignant"]
        assert 0.0 <= prob <= 1.0

    def test_risk_level_is_valid_category(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.json()["risk_level"] in ("high", "medium", "low")

    def test_top_features_is_a_list(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert isinstance(response.json()["top_features"], list)

    def test_top_features_has_feature_and_shap_value(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        for item in response.json()["top_features"]:
            assert "feature" in item
            assert "shap_value" in item

    def test_logger_is_called_on_valid_predict(self):
        with patch("app.main.prediction_logger") as mock_logger:
            client.post("/predict", json=VALID_PAYLOAD)
            mock_logger.log.assert_called_once()
