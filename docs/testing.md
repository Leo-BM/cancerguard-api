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

Testam a função `predict()` de `app/model.py` diretamente, isolada da camada HTTP.

| Teste | Caso de entrada | Asserção principal |
|---|---|---|
| `test_benign_prediction` | Features típicas de tumor benigno | `prediction == "benign"` e `probability_malignant < 0.5` |
| `test_malignant_prediction` | Features típicas de tumor maligno | `prediction == "malignant"` e `probability_malignant > 0.5` |
| `test_recall_is_prioritized` | Caso maligno com threshold conservador | `probability_malignant > 0.3` |
| `test_risk_level_high` | `probability_malignant = 0.75` | `risk_level == "high"` |
| `test_risk_level_medium` | `probability_malignant = 0.55` | `risk_level == "medium"` |
| `test_risk_level_low` | `probability_malignant = 0.20` | `risk_level == "low"` |
| `test_top_features_present` | Qualquer entrada válida | `len(top_features) > 0` |

```python
# tests/test_predict.py — estrutura de referência
import pytest
from app.model import predict

BENIGN_CASE = {
    "mean_radius": 12.0, "mean_texture": 15.0, "mean_perimeter": 77.0,
    "mean_area": 440.0, "mean_smoothness": 0.09,
    # ... demais 25 features com valores típicos de tumor benigno
}

MALIGNANT_CASE = {
    "mean_radius": 20.0, "mean_texture": 25.0, "mean_perimeter": 135.0,
    "mean_area": 1300.0, "mean_smoothness": 0.14,
    # ... demais 25 features com valores típicos de tumor maligno
}

def test_benign_prediction():
    result = predict(BENIGN_CASE)
    assert result["prediction"] == "benign"
    assert result["probability_malignant"] < 0.5

def test_malignant_prediction():
    result = predict(MALIGNANT_CASE)
    assert result["prediction"] == "malignant"
    assert result["probability_malignant"] > 0.5

def test_recall_is_prioritized():
    """O modelo deve errar para o lado seguro: preferir falso positivo a falso negativo."""
    result = predict(MALIGNANT_CASE)
    assert result["probability_malignant"] > 0.3
```

---

## 3. Testes de Integração — `tests/test_api.py`

Testam os endpoints HTTP usando o `TestClient` do FastAPI (sem inicializar servidor real).

| Teste | Endpoint | O que verifica |
|---|---|---|
| `test_health_endpoint` | `GET /health` | Status 200 + campo `status == "healthy"` |
| `test_predict_valid_input` | `POST /predict` | Status 200 com dados válidos |
| `test_predict_invalid_type` | `POST /predict` | Status 422 — campo com tipo errado |
| `test_predict_missing_field` | `POST /predict` | Status 422 — campo obrigatório ausente |
| `test_predict_negative_value` | `POST /predict` | Status 422 — valor negativo (viola `gt=0`) |
| `test_response_has_prediction` | `POST /predict` | Campo `prediction` presente no response |
| `test_response_has_probability` | `POST /predict` | Campo `probability_malignant` entre 0 e 1 |
| `test_response_has_risk_level` | `POST /predict` | `risk_level` é "high", "medium" ou "low" |
| `test_response_has_top_features` | `POST /predict` | `top_features` é lista não vazia |

```python
# tests/test_api.py — estrutura de referência
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_valid_input():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

def test_predict_invalid_type():
    payload = {**VALID_PAYLOAD, "mean_radius": "not_a_number"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_missing_field():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "mean_radius"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
```

---

## 4. Executando os Testes

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
