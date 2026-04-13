# Estratégia de Testes — CancerGuard API

← [Voltar ao índice](./README.md)

---

## 1. Pirâmide de Testes

```
            ┌──────────────────┐
            │   E2E (manual)   │  ← curl / browser na URL de produção
            ├──────────────────┤
            │   Integração     │  ← test_api.py (TestClient FastAPI)
            ├──────────────────┤
            │   Unitários      │  ← test_predict.py (função predict)
            └──────────────────┘
```

---

## 2. Testes Unitários — `tests/test_predict.py`

Testam as funções `predict()` e `_get_risk_level()` de `app/model.py` diretamente, isoladas da camada HTTP.

Organizados em **2 classes**, **15 testes** no total:

**`TestPredictOutput`** (7 testes) — valida a estrutura e valores do dicionário retornado por `predict()`:

| Teste | Asserção principal |
|---|---|
| `test_predict_returns_dict_with_all_keys` | `prediction`, `probability_malignant`, `risk_level`, `top_features` presentes |
| `test_prediction_is_malignant_when_prob_above_threshold` | mock retorna prob=0.85 → `prediction == "malignant"` |
| `test_probability_is_float_between_0_and_1` | `0.0 <= prob <= 1.0` |
| `test_probability_is_rounded_to_4_decimals` | até 4 casas decimais |
| `test_top_features_is_non_empty_list` | lista não vazia |
| `test_top_features_has_at_most_3_items` | `len <= 3` |
| `test_top_features_each_has_feature_and_shap_value` | cada item tem `feature` (str) e `shap_value` (float) |

**`TestRiskLevel`** (8 testes) — valida a lógica pura de `_get_risk_level(prob)`:

| Teste | Entrada | Saída esperada |
|---|---|---|
| `test_risk_level_high_at_0_7` | `0.70` | `"high"` |
| `test_risk_level_high_above_0_7` | `0.95` | `"high"` |
| `test_risk_level_medium_at_0_4` | `0.40` | `"medium"` |
| `test_risk_level_medium_below_0_7` | `0.69` | `"medium"` |
| `test_risk_level_low_below_0_4` | `0.39` | `"low"` |
| `test_risk_level_low_at_zero` | `0.0` | `"low"` |
| `test_risk_level_boundary_exactly_0_5` | `0.50` | `"medium"` |
| `test_risk_level_boundary_exactly_1_0` | `1.0` | `"high"` |

```python
# tests/test_predict.py — estrutura real implementada
import pytest
from app.model import predict, _get_risk_level

# Casos reais do dataset Wisconsin (amostra índices 10 e 0)
BENIGN_CASE = {
    "mean_radius": 11.42, "mean_texture": 20.38, "mean_perimeter": 77.58,
    "mean_area": 386.1, "mean_smoothness": 0.1425,
    # ... 25 features restantes
}

MALIGNANT_CASE = {
    "mean_radius": 20.57, "mean_texture": 17.77, "mean_perimeter": 132.9,
    "mean_area": 1326.0, "mean_smoothness": 0.08474,
    # ... 25 features restantes
}

class TestPredictOutput:
    def test_prediction_is_malignant_when_prob_above_threshold(self):
        # O mock (conftest.py) retorna predict_proba = [[0.85, 0.15]]
        # prob_malignant = proba[0] = 0.85 → acima de 0.5 → "malignant"
        result = predict(MALIGNANT_CASE)
        assert result["prediction"] == "malignant"

class TestRiskLevel:
    def test_risk_level_high_at_0_7(self):
        assert _get_risk_level(0.7) == "high"

    def test_risk_level_medium_at_0_4(self):
        assert _get_risk_level(0.4) == "medium"

    def test_risk_level_low_below_0_4(self):
        assert _get_risk_level(0.39) == "low"
```

---

## 3. Testes de Integração — `tests/test_api.py`

Testam os endpoints HTTP usando o `TestClient` do FastAPI (sem inicializar servidor real).
Organizados em **3 classes**, **21 testes** no total:

**`TestHealthEndpoint`** (4 testes):

| Teste | Endpoint | O que verifica |
|---|---|---|
| `test_health_returns_200` | `GET /health` | Status code 200 |
| `test_health_has_status_field` | `GET /health` | Campo `status` presente |
| `test_health_status_is_healthy` | `GET /health` | `status == "healthy"` |
| `test_health_has_model_field` | `GET /health` | Campo `model` presente |

**`TestPredictValidation`** (6 testes) — HTTP 422 em entradas inválidas:

| Teste | Entrada inválida | Asserção |
|---|---|---|
| `test_missing_field` | Sem `mean_radius` | Status 422 |
| `test_invalid_type` | `mean_radius = "texto"` | Status 422 |
| `test_negative_value` | `mean_radius = -1.0` | Status 422 (viola `gt=0`) |
| `test_zero_value` | `mean_radius = 0.0` | Status 422 (viola `gt=0`) |
| `test_smoothness_above_1` | `mean_smoothness = 1.5` | Status 422 (viola `lt=1`) |
| `test_empty_body` | `{}` | Status 422 |

**`TestPredictValidInput`** (11 testes) — resposta completa com entrada válida:

| Teste | O que verifica |
|---|---|
| `test_predict_returns_200` | Status code 200 |
| `test_prediction_field_present` | Campo `prediction` presente |
| `test_prediction_is_string` | `prediction` é string |
| `test_prediction_valid_value` | `prediction` é `"malignant"` ou `"benign"` |
| `test_probability_malignant_present` | Campo `probability_malignant` presente |
| `test_probability_malignant_is_float` | `probability_malignant` é float |
| `test_probability_malignant_range` | `0.0 <= prob <= 1.0` |
| `test_risk_level_present` | Campo `risk_level` presente |
| `test_risk_level_valid_value` | `risk_level` em `{"high", "medium", "low"}` |
| `test_top_features_present` | Campo `top_features` presente |
| `test_top_features_non_empty` | `len(top_features) >= 1` |

```python
# tests/test_api.py — estrutura real implementada
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_is_healthy(self):
        response = client.get("/health")
        assert response.json()["status"] == "healthy"

class TestPredictValidation:
    def test_missing_field(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "mean_radius"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

class TestPredictValidInput:
    def test_predict_returns_200(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_prediction_valid_value(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.json()["prediction"] in {"malignant", "benign"}

    def test_risk_level_valid_value(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.json()["risk_level"] in {"high", "medium", "low"}
```

---

## 4. Fixtures e Mocks — `tests/conftest.py`

O `conftest.py` usa `pytest.fixture(scope="session", autouse=True)` para interceptar o carregamento do modelo **antes de qualquer teste rodar**. Isso é necessário porque:

- O ambiente de CI **não tem MLflow rodando** — `load_model()` falharia
- O `TestClient` **não ativa o lifespan** do FastAPI — `prediction_logger` ficaria `None`

```python
# tests/conftest.py — mocks injetados automaticamente
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture(scope="session", autouse=True)
def mock_model_globals():
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])
    # proba[0] = 0.85 → predict() interpreta como prob_malignant

    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 30))

    mock_explainer = MagicMock()
    rng = np.random.default_rng(42)
    mock_explainer.shap_values.return_value = [rng.standard_normal((1, 30))]

    mock_logger = MagicMock()

    with (
        patch("app.model._model", mock_model),
        patch("app.model._scaler", mock_scaler),
        patch("app.model._explainer", mock_explainer),
        patch("app.model.load_model"),           # neutraliza chamada ao MLflow
        patch("app.main.prediction_logger", mock_logger),  # evita NoneType no log
    ):
        yield
```

**Por que `proba[0]`?**
O sklearn Wisconsin dataset codifica as classes como: `0 = malignant`, `1 = benign`. Portanto `predict_proba` retorna `[prob_malignant, prob_benign]` e `proba[0]` é a probabilidade de malignidade.

---

## 5. Resultados Reais (Fase 6)

```
36 passed in ~0.8s

Coverage report:
  app/schemas.py         100%
  app/model.py            74%
  app/main.py             72%
  app/logging_config.py   53%
  ─────────────────────────
  TOTAL                   80%
```

---

## 6. Executando os Testes

```bash
# Todos os testes
pytest tests/ -v

# Apenas unitários
pytest tests/test_predict.py -v

# Apenas integração
pytest tests/test_api.py -v

# Com cobertura
pytest tests/ --cov=app --cov-report=term-missing
```

---

## 5. Testes no CI/CD

Os testes são executados automaticamente pelo GitHub Actions em cada push e PR:

```yaml
- name: Run tests
  run: pytest tests/ -v --tb=short
```

O job de deploy só é acionado após os testes passarem.
