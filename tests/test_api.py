"""
test_api.py — Testes de integração dos endpoints HTTP do CancerGuard.

O TestClient do FastAPI simula requisições HTTP na memória, sem inicializar
um servidor real. Ele executa o lifespan completo da aplicação (startup →
yield → shutdown), por isso o conftest.py mocka o load_model() antes disso.

O que estes testes validam:
  - Que os endpoints respondem com os status HTTP corretos
  - Que o Pydantic rejeita inputs inválidos com HTTP 422
  - Que a estrutura e os tipos do response body estão corretos
  - Que o SQLite logger é chamado a cada predição bem-sucedida

O que estes testes NÃO validam (e não devem):
  - A acurácia ou lógica do modelo (responsabilidade do test_predict.py)
  - O comportamento do MLflow (mockado pelo conftest.py)
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

# ---------------------------------------------------------------------------
# Cliente de testes — inicializado uma vez por módulo.
# TestClient dispara o lifespan da FastAPI, incluindo o startup que chama
# load_model() e instancia o PredictionLogger. O conftest.py garante que
# load_model() é um no-op durante os testes.
# ---------------------------------------------------------------------------
client = TestClient(app)

# ---------------------------------------------------------------------------
# Payload de referência — 30 features com valores reais do dataset Wisconsin.
# Usado como base para os testes, modificado pontualmente quando necessário.
# ---------------------------------------------------------------------------
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
    """Testa o endpoint de liveness check."""

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


# ---------------------------------------------------------------------------
# POST /predict — inputs inválidos
# Esses testes verificam que o Pydantic rejeita dados ruins ANTES de tocar
# no modelo. O HTTP 422 (Unprocessable Entity) é retornado automaticamente.
# ---------------------------------------------------------------------------

class TestPredictValidation:
    """Valida que inputs inválidos são rejeitados com HTTP 422."""

    def test_missing_field_returns_422(self):
        # Remove mean_radius — campo obrigatório
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "mean_radius"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_wrong_type_returns_422(self):
        # mean_radius deve ser float, não string
        payload = {**VALID_PAYLOAD, "mean_radius": "not_a_number"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_negative_value_returns_422(self):
        # Viola Field(..., gt=0)
        payload = {**VALID_PAYLOAD, "mean_radius": -5.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_zero_value_returns_422(self):
        # gt=0 exclui o próprio zero
        payload = {**VALID_PAYLOAD, "mean_radius": 0.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_smoothness_above_1_returns_422(self):
        # mean_smoothness tem Field(..., gt=0, lt=1) — proporção
        payload = {**VALID_PAYLOAD, "mean_smoothness": 1.5}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_empty_body_returns_422(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /predict — inputs válidos
# ---------------------------------------------------------------------------

class TestPredictValidInput:
    """Valida o comportamento com inputs corretos."""

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
        """
        Verifica que o PredictionLogger persiste a predição.
        Mockamos o logger para não escrever em disco durante os testes.
        """
        with patch("app.main.prediction_logger") as mock_logger:
            client.post("/predict", json=VALID_PAYLOAD)
            mock_logger.log.assert_called_once()
