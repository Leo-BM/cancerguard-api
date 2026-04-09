# Roteiro de Implementação — CancerGuard API

← [Voltar ao índice](./README.md)

> **Para quem é este documento:** Estudante de Data Science implementando o projeto pela primeira vez. Cada etapa explica **o que** fazer, **por que** aquela decisão existe e **como verificar** que funcionou antes de avançar.

---

## Visão Geral das Fases

```
Fase 1 → Ambiente e Estrutura de Pastas
Fase 2 → Treinamento do Modelo com MLflow
Fase 3 → Registro do Modelo no MLflow Registry
Fase 4 → API FastAPI (Core)
Fase 5 → Explicabilidade (SHAP) e Logging (SQLite)
Fase 6 → Testes Automatizados
Fase 7 → Interface Visual (Streamlit)
Fase 8 → Containerização (Docker + Compose)
Fase 9 → CI/CD (GitHub Actions)
Fase 10 → Deploy na Nuvem (Render)
```

Cada fase gera algo **funcional e testável** antes de passar para a próxima. Nunca avance sem verificar a etapa atual.

---

## Fase 1 — Ambiente e Estrutura de Pastas

### Conceito
Antes de escrever qualquer código de ML, é necessário organizar o projeto de forma reproduzível. O `venv` isola as dependências do projeto das do sistema operacional. O `requirements.txt` com versões fixas garante que outra pessoa (ou o servidor de CI) consiga reproduzir exatamente o mesmo ambiente.

### Pré-requisitos
- Python 3.11 instalado
- Git inicializado no repositório

### Passos

**1.1 — Criar e ativar o ambiente virtual**
```bash
python3.11 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

**1.2 — Criar a estrutura de pastas**
```bash
mkdir -p app training tests .github/workflows streamlit_app
touch app/__init__.py training/__init__.py tests/__init__.py
```

A estrutura final deve ser:
```
cancerguard-api/
├── app/
│   ├── __init__.py
│   ├── main.py          ← FastAPI app
│   ├── model.py         ← carregamento e predição
│   ├── schemas.py       ← Pydantic models (input/output)
│   └── logging_config.py ← SQLite logger
├── training/
│   ├── __init__.py
│   ├── train.py         ← script de treinamento
│   └── register_model.py ← promoção no MLflow Registry
├── streamlit_app/
│   └── app.py
├── tests/
│   ├── __init__.py
│   ├── test_predict.py
│   └── test_api.py
├── .github/workflows/
│   └── ci.yml
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

**1.3 — Criar o `requirements.txt`**
```
fastapi==0.115.0
uvicorn==0.30.0
pydantic==2.8.0
scikit-learn==1.6.1
mlflow==2.19.0
shap==0.46.0
streamlit==1.41.0
pytest==8.3.0
httpx==0.27.0
python-dotenv==1.0.1
pytest-cov==5.0.0
```

**1.4 — Instalar as dependências**
```bash
pip install -r requirements.txt
```

**1.5 — Criar o `.gitignore`**
```
.env
.venv/
predictions.db
mlruns/
*.joblib
__pycache__/
.pytest_cache/
*.pyc
```

**1.6 — Criar o `.env.example`**
```
MLFLOW_TRACKING_URI=http://localhost:5000
PORT=8000
API_BASE_URL=http://localhost:8000
LOG_DB_PATH=predictions.db
```

### Verificação ✅
```bash
python --version        # deve retornar Python 3.11.x
pip list | grep fastapi # deve retornar fastapi 0.110.0
```

---

## Fase 2 — Treinamento do Modelo com MLflow

### Conceito
O MLflow é uma ferramenta de **rastreamento de experimentos** (experiment tracking). A cada treinamento, ele salva automaticamente: os hiperparâmetros usados, as métricas obtidas e os artefatos (modelo serializado, scaler). Isso responde à pergunta "qual combinação de hiperparâmetros gerou este modelo?" sem depender de anotações manuais.

O **StandardScaler** é salvo junto ao modelo porque ele foi ajustado (`fit`) nos dados de treino. Na predição, é obrigatório usar o mesmo scaler — caso contrário, as features chegarão ao SVM em escalas diferentes do que ele aprendeu.

O **SVM com kernel RBF** foi escolhido porque obteve o melhor Recall sobre a classe maligna (96,8%), que é a métrica prioritária neste domínio (ver [data-model.md](./data-model.md)).

### Arquivo: `training/train.py`

```python
import mlflow
import mlflow.sklearn
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score

def train():
    # 1. Carregar o dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # 2. Divisão treino/teste com estratificação
    #    stratify=y garante proporção de classes igual em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Normalização (ajuste SOMENTE no treino, transformação em ambos)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 4. Treinamento do SVM
    model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
    model.fit(X_train_scaled, y_train)

    # 5. Avaliação (y_pred sobre o conjunto de teste)
    y_pred = model.predict(X_test_scaled)
    metrics = {
        "recall":    recall_score(y_test, y_pred),
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
    }
    print("Métricas:", metrics)

    # 6. Registrar no MLflow
    mlflow.set_experiment("cancerguard")

    with mlflow.start_run(run_name="svm-rbf-baseline"):
        mlflow.log_params({"kernel": "rbf", "C": 1.0, "gamma": "scale"})
        mlflow.log_metrics(metrics)

        # Salvar o scaler como artefato separado
        joblib.dump(scaler, "scaler.joblib")
        mlflow.log_artifact("scaler.joblib")

        # Salvar o modelo
        mlflow.sklearn.log_model(model, "svm_model")
        mlflow.set_tags({"author": "Leo-BM", "dataset_version": "breast_cancer_v1"})

        print(f"Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    train()
```

### Passos

**2.1 — Subir a interface do MLflow localmente**
```bash
mlflow ui
# Abre http://localhost:5000 no navegador
```

**2.2 — Executar o treinamento em outro terminal**
```bash
source .venv/bin/activate
python training/train.py
```

### O que acontece internamente
```
train.py executa
      │
      ├── load_breast_cancer()   → 569 amostras, 30 features, 2 classes
      ├── train_test_split()     → 455 treino / 114 teste
      ├── StandardScaler.fit()   → aprende μ e σ do treino
      ├── SVC.fit()              → treina o modelo
      ├── predict() + métricas   → avalia no conjunto de teste
      └── mlflow.start_run()     → persiste tudo em mlruns/
```

### Verificação ✅
- O terminal imprimiu as métricas (recall deve ser ~0.968)
- Em `http://localhost:5000` aparece o experimento `cancerguard` com 1 run
- Clicando no run, é possível ver os parâmetros, métricas e artefatos (`svm_model/`, `scaler.joblib`)

---

## Fase 3 — Registro no MLflow Model Registry

### Conceito
O **Model Registry** é um repositório central de modelos versionados. Um modelo passa por estágios (`None → Staging → Production → Archived`). A API carrega **sempre o modelo em stage `Production`**, o que significa que:
- Trocar de modelo = mudar o stage no Registry, sem precisar rebuildar o container Docker
- Rollback = reverter o stage para a versão anterior

### Arquivo: `training/register_model.py`

```python
import mlflow

def register_and_promote(run_id: str):
    client = mlflow.MlflowClient()

    # 1. Registrar o modelo do run no Registry
    model_uri = f"runs:/{run_id}/svm_model"
    mv = mlflow.register_model(model_uri, "cancerguard-svm")
    print(f"Modelo registrado: versão {mv.version}")

    # 2. Promover para Staging
    client.transition_model_version_stage(
        name="cancerguard-svm",
        version=mv.version,
        stage="Staging",
    )
    print("Promovido para Staging")

    # 3. Promover para Production
    client.transition_model_version_stage(
        name="cancerguard-svm",
        version=mv.version,
        stage="Production",
    )
    print("Promovido para Production — API pode carregar este modelo")

if __name__ == "__main__":
    # Cole aqui o Run ID impresso pelo train.py
    run_id = input("Cole o Run ID do MLflow: ")
    register_and_promote(run_id)
```

### Passos

**3.1 — Executar o registro**
```bash
python training/register_model.py
# Cole o Run ID que apareceu no terminal do train.py
```

### Verificação ✅
- Em `http://localhost:5000` → aba **Models** → `cancerguard-svm` aparece com Version 1 em stage `Production`

---

## Fase 4 — API FastAPI (Core)

### Conceito
A API é dividida em três módulos com responsabilidades claras:

| Arquivo | Responsabilidade |
|---|---|
| `schemas.py` | Define o contrato de dados (input e output) com validação automática via Pydantic |
| `model.py` | Carrega o modelo do MLflow e executa a predição |
| `main.py` | Conecta tudo: recebe a requisição HTTP, chama o modelo, retorna o response |

O Pydantic valida **antes** de qualquer código de negócio ser executado. Se o input for inválido, a FastAPI retorna HTTP 422 automaticamente — o modelo nunca é atingido por dados ruins.

### Arquivo: `app/schemas.py`

```python
from pydantic import BaseModel, Field
from typing import List

class PredictionInput(BaseModel):
    mean_radius:            float = Field(..., gt=0)
    mean_texture:           float = Field(..., gt=0)
    mean_perimeter:         float = Field(..., gt=0)
    mean_area:              float = Field(..., gt=0)
    mean_smoothness:        float = Field(..., gt=0, lt=1)
    mean_compactness:       float = Field(..., gt=0)
    mean_concavity:         float = Field(..., gt=0)
    mean_concave_points:    float = Field(..., gt=0)
    mean_symmetry:          float = Field(..., gt=0)
    mean_fractal_dimension: float = Field(..., gt=0)
    radius_se:              float = Field(..., gt=0)
    texture_se:             float = Field(..., gt=0)
    perimeter_se:           float = Field(..., gt=0)
    area_se:                float = Field(..., gt=0)
    smoothness_se:          float = Field(..., gt=0)
    compactness_se:         float = Field(..., gt=0)
    concavity_se:           float = Field(..., gt=0)
    concave_points_se:      float = Field(..., gt=0)
    symmetry_se:            float = Field(..., gt=0)
    fractal_dimension_se:   float = Field(..., gt=0)
    worst_radius:           float = Field(..., gt=0)
    worst_texture:          float = Field(..., gt=0)
    worst_perimeter:        float = Field(..., gt=0)
    worst_area:             float = Field(..., gt=0)
    worst_smoothness:       float = Field(..., gt=0)
    worst_compactness:      float = Field(..., gt=0)
    worst_concavity:        float = Field(..., gt=0)
    worst_concave_points:   float = Field(..., gt=0)
    worst_symmetry:         float = Field(..., gt=0)
    worst_fractal_dimension: float = Field(..., gt=0)

class FeatureImportance(BaseModel):
    feature: str
    shap_value: float

class PredictionOutput(BaseModel):
    prediction:             str
    probability_malignant:  float
    risk_level:             str
    top_features:           List[FeatureImportance]
```

### Arquivo: `app/model.py`

```python
import mlflow.sklearn
import numpy as np
import shap
import os

_model = None
_scaler = None
_explainer = None

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

def load_model():
    """Carrega o modelo em Production do MLflow Registry (singleton)."""
    global _model, _scaler, _explainer

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    _model  = mlflow.sklearn.load_model("models:/cancerguard-svm/Production")
    _scaler = mlflow.artifacts.load_artifacts("models:/cancerguard-svm/Production", "scaler.joblib")
    
    # Inicializa o explainer SHAP com o modelo carregado
    _explainer = shap.Explainer(_model.predict_proba, shap.maskers.Independent(
        np.zeros((1, 30))
    ))

def _get_risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "high"
    elif prob >= 0.4:
        return "medium"
    return "low"

def predict(input_data: dict) -> dict:
    """Recebe um dict com as 30 features e retorna o resultado da predição."""
    X = np.array([[input_data[f] for f in FEATURE_NAMES]])
    X_scaled = _scaler.transform(X)

    proba = _model.predict_proba(X_scaled)[0]
    # scikit-learn: índice 0 = malignant, índice 1 = benign
    prob_malignant = float(proba[0])
    prediction = "malignant" if prob_malignant >= 0.5 else "benign"

    # SHAP para explicabilidade
    shap_values = _explainer(X_scaled)
    # valores SHAP para a classe maligna (índice 0)
    importances = list(zip(FEATURE_NAMES, shap_values.values[0, :, 0]))
    importances.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features = [{"feature": f, "shap_value": round(v, 4)} for f, v in importances[:3]]

    return {
        "prediction": prediction,
        "probability_malignant": round(prob_malignant, 4),
        "risk_level": _get_risk_level(prob_malignant),
        "top_features": top_features,
    }
```

### Arquivo: `app/main.py`

```python
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.schemas import PredictionInput, PredictionOutput
from app.model import load_model, predict
from app.logging_config import PredictionLogger
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cancerguard")

prediction_logger: PredictionLogger = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Executado no startup: carrega modelo e inicializa o logger."""
    global prediction_logger
    logger.info("Iniciando CancerGuard API...")
    load_model()
    prediction_logger = PredictionLogger()
    logger.info("Modelo carregado e logger inicializado.")
    yield

app = FastAPI(title="CancerGuard API", version="1.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "healthy", "model": "svm-rbf", "version": "1.0"}

@app.post("/predict", response_model=PredictionOutput)
def predict_endpoint(payload: PredictionInput):
    try:
        result = predict(payload.model_dump())
        prediction_logger.log(payload.model_dump(), result["prediction"], result["probability_malignant"])
        return result
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### Passos

**4.1 — Implementar os três arquivos acima**

**4.2 — Testar a API localmente (MLflow já deve estar rodando)**
```bash
uvicorn app.main:app --reload --port 8000
```

**4.3 — Testar manualmente via Swagger UI**

Abra `http://localhost:8000/docs` e execute:
- `GET /health` → deve retornar `{"status": "healthy", ...}`
- `POST /predict` com o payload de exemplo do [api-spec.md](./api-spec.md)

### Verificação ✅
- `/health` retorna HTTP 200
- `/predict` com dados válidos retorna HTTP 200 com `prediction`, `probability_malignant`, `risk_level` e `top_features`
- `/predict` com um campo faltando retorna HTTP 422

---

## Fase 5 — Logging de Predições (SQLite)

### Conceito
O `PredictionLogger` persiste cada predição em um banco SQLite local. Isso cria um **log de auditoria** que permite responder: "quais inputs chegaram na API e quais foram os resultados?". Em produção, esse log é a base para detectar data drift (quando a distribuição dos inputs muda ao longo do tempo).

O logger é instanciado como **singleton no startup** da FastAPI para evitar múltiplas conexões concorrentes ao SQLite.

### Arquivo: `app/logging_config.py`

```python
import sqlite3
import json
import os
from datetime import datetime

class PredictionLogger:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.getenv("LOG_DB_PATH", "predictions.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                timestamp   TEXT,
                input_data  TEXT,
                prediction  TEXT,
                probability REAL
            )
        """)
        self.conn.commit()

    def log(self, input_data: dict, prediction: str, probability: float):
        self.conn.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?)",
            (
                datetime.now().isoformat(),
                json.dumps(input_data),
                prediction,
                probability,
            ),
        )
        self.conn.commit()
```

### Verificação ✅
```bash
# Após algumas requisições ao /predict:
sqlite3 predictions.db "SELECT timestamp, prediction, probability FROM predictions LIMIT 5;"
```

---

## Fase 6 — Testes Automatizados

### Conceito
A **pirâmide de testes** define três camadas:

```
         E2E        ← poucos, lentos, testam o sistema inteiro
      Integração    ← testam a API HTTP (TestClient, sem servidor real)
      Unitários     ← testam funções isoladas, rápidos
```

Os testes unitários validam a lógica de `predict()` diretamente. Os de integração testam os endpoints HTTP usando o `TestClient` do FastAPI (é uma simulação de HTTP — não precisa do servidor rodando). Ver estratégia completa em [testing.md](./testing.md).

### Arquivo: `tests/test_predict.py`

```python
import pytest
from app.model import predict, load_model

# Features típicas de tumor benigno (valores baixos)
BENIGN_CASE = {
    "mean_radius": 12.0, "mean_texture": 15.0, "mean_perimeter": 77.0,
    "mean_area": 440.0, "mean_smoothness": 0.09, "mean_compactness": 0.07,
    "mean_concavity": 0.04, "mean_concave_points": 0.02, "mean_symmetry": 0.17,
    "mean_fractal_dimension": 0.06, "radius_se": 0.28, "texture_se": 0.80,
    "perimeter_se": 1.90, "area_se": 20.0, "smoothness_se": 0.005,
    "compactness_se": 0.015, "concavity_se": 0.020, "concave_points_se": 0.008,
    "symmetry_se": 0.015, "fractal_dimension_se": 0.002, "worst_radius": 14.0,
    "worst_texture": 20.0, "worst_perimeter": 90.0, "worst_area": 580.0,
    "worst_smoothness": 0.12, "worst_compactness": 0.14, "worst_concavity": 0.11,
    "worst_concave_points": 0.06, "worst_symmetry": 0.24, "worst_fractal_dimension": 0.07,
}

# Features típicas de tumor maligno (valores altos)
MALIGNANT_CASE = {
    "mean_radius": 20.0, "mean_texture": 25.0, "mean_perimeter": 135.0,
    "mean_area": 1300.0, "mean_smoothness": 0.14, "mean_compactness": 0.25,
    "mean_concavity": 0.30, "mean_concave_points": 0.15, "mean_symmetry": 0.22,
    "mean_fractal_dimension": 0.07, "radius_se": 0.80, "texture_se": 1.50,
    "perimeter_se": 5.50, "area_se": 90.0, "smoothness_se": 0.009,
    "compactness_se": 0.045, "concavity_se": 0.060, "concave_points_se": 0.020,
    "symmetry_se": 0.030, "fractal_dimension_se": 0.006, "worst_radius": 26.0,
    "worst_texture": 35.0, "worst_perimeter": 175.0, "worst_area": 2100.0,
    "worst_smoothness": 0.19, "worst_compactness": 0.55, "worst_concavity": 0.65,
    "worst_concave_points": 0.28, "worst_symmetry": 0.38, "worst_fractal_dimension": 0.10,
}

@pytest.fixture(scope="module", autouse=True)
def setup_model():
    load_model()

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

def test_risk_level_high():
    result = predict(MALIGNANT_CASE)
    assert result["risk_level"] == "high"

def test_top_features_present():
    result = predict(BENIGN_CASE)
    assert len(result["top_features"]) > 0
```

### Arquivo: `tests/test_api.py`

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "mean_radius": 14.0, "mean_texture": 19.0, "mean_perimeter": 91.0,
    "mean_area": 654.0, "mean_smoothness": 0.095, "mean_compactness": 0.109,
    "mean_concavity": 0.112, "mean_concave_points": 0.074, "mean_symmetry": 0.181,
    "mean_fractal_dimension": 0.057, "radius_se": 0.37, "texture_se": 0.87,
    "perimeter_se": 2.64, "area_se": 28.0, "smoothness_se": 0.005,
    "compactness_se": 0.021, "concavity_se": 0.032, "concave_points_se": 0.011,
    "symmetry_se": 0.019, "fractal_dimension_se": 0.003, "worst_radius": 16.0,
    "worst_texture": 25.0, "worst_perimeter": 105.0, "worst_area": 819.0,
    "worst_smoothness": 0.132, "worst_compactness": 0.240, "worst_concavity": 0.290,
    "worst_concave_points": 0.140, "worst_symmetry": 0.290, "worst_fractal_dimension": 0.082,
}

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

def test_predict_negative_value():
    payload = {**VALID_PAYLOAD, "mean_radius": -1.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_response_has_all_fields():
    response = client.post("/predict", json=VALID_PAYLOAD)
    body = response.json()
    assert "prediction" in body
    assert "probability_malignant" in body
    assert "risk_level" in body
    assert "top_features" in body
    assert body["risk_level"] in ("high", "medium", "low")
    assert 0.0 <= body["probability_malignant"] <= 1.0
```

### Passos

**6.1 — Executar todos os testes**
```bash
pytest tests/ -v
```

**6.2 — Verificar cobertura**
```bash
pytest tests/ --cov=app --cov-report=term-missing
```

### Verificação ✅
- Todos os testes passam (verde)
- Cobertura da pasta `app/` acima de 80%

---

## Fase 7 — Interface Visual (Streamlit)

### Conceito
O Streamlit permite criar interfaces web interativas em Python puro, sem HTML/CSS/JavaScript. Aqui ele serve como **cliente da API** — o usuário preenche os valores das features em sliders/inputs e a Streamlit envia um `POST /predict` para a FastAPI.

A Streamlit é **desacoplada** da API: cada uma roda em um container separado e se comunicam via HTTP. Isso permite desenvolver e deployar cada serviço de forma independente.

### Arquivo: `streamlit_app/app.py`

```python
import streamlit as st
import requests
import os

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.title("🔬 CancerGuard — Diagnóstico Assistido por IA")
st.markdown("Preencha as medidas do exame para obter a classificação do tumor.")

with st.form("prediction_form"):
    st.subheader("Valores Médios")
    col1, col2 = st.columns(2)
    with col1:
        mean_radius    = st.number_input("Raio Médio",    min_value=0.01, value=14.0)
        mean_texture   = st.number_input("Textura Média", min_value=0.01, value=19.0)
        mean_perimeter = st.number_input("Perímetro Med.",min_value=0.01, value=91.0)
        mean_area      = st.number_input("Área Média",    min_value=0.01, value=654.0)
        mean_smoothness = st.number_input("Suavidade Méd.", min_value=0.001, max_value=0.999, value=0.095)
    with col2:
        mean_compactness       = st.number_input("Compacidade Méd.", min_value=0.001, value=0.109)
        mean_concavity         = st.number_input("Concavidade Méd.", min_value=0.001, value=0.112)
        mean_concave_points    = st.number_input("Pts Côncavos Méd.", min_value=0.001, value=0.074)
        mean_symmetry          = st.number_input("Simetria Méd.",    min_value=0.001, value=0.181)
        mean_fractal_dimension = st.number_input("Dim. Fractal Méd.",min_value=0.001, value=0.057)

    # (adicionar campos SE e Worst de forma similar)
    # Omitido aqui para brevidade — preencher todos os 30 campos

    submitted = st.form_submit_button("Classificar Tumor")

if submitted:
    payload = {
        "mean_radius": mean_radius,
        "mean_texture": mean_texture,
        # ... demais campos
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Diagnóstico: **{result['prediction'].upper()}**")
            st.metric("Probabilidade Maligno", f"{result['probability_malignant']:.1%}")
            st.metric("Nível de Risco", result["risk_level"].upper())

            st.subheader("Top Features (SHAP)")
            for feat in result["top_features"]:
                st.write(f"• `{feat['feature']}`: {feat['shap_value']:.4f}")
        else:
            st.error(f"Erro da API: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Não foi possível conectar à API. Verifique se ela está rodando.")
```

### Passos

**7.1 — Rodar a Streamlit (com a API já em execução)**
```bash
streamlit run streamlit_app/app.py --server.port 8501
```

### Verificação ✅
- `http://localhost:8501` carrega o formulário
- Ao submeter, a predição aparece na tela

---

## Fase 8 — Containerização (Docker)

### Conceito
O Docker empacota a aplicação e todas as suas dependências em uma **imagem portável**. O Docker Compose orquestra múltiplos containers (API + MLflow) com uma única configuração. O resultado: qualquer pessoa com Docker instalado consegue rodar o projeto inteiro com um único comando.

**Importante: ordem das layers no Dockerfile.** O Docker aproveita cache — se uma layer não mudou, não rebuilda. Copiar `requirements.txt` antes do código garante que as dependências (layer lenta, ~2 min) só sejam reinstaladas quando as deps mudarem, não a cada mudança de código.

### Arquivo: `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Layer pesada — só rebuilda se requirements.txt mudar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layer de código — rebuilda a cada mudança
COPY app/ ./app/
COPY training/ ./training/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Arquivo: `docker-compose.yml`

```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0
    volumes:
      - mlflow_data:/mlflow

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow

  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://api:8000
    depends_on:
      - api

volumes:
  mlflow_data:
```

### Passos

**8.1 — Criar o `Dockerfile.streamlit`**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY streamlit_app/ ./streamlit_app/
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

**8.2 — Build e execução**

> **Atenção:** antes de subir via Docker, é necessário ter um modelo em `Production` no MLflow. Execute as Fases 2 e 3 localmente primeiro para popular o volume `mlflow_data`, ou configure o MLflow para usar um banco de dados persistente.

```bash
# Subir tudo
docker compose up --build

# Verificar containers rodando
docker compose ps

# Ver logs
docker compose logs api -f
```

### Verificação ✅
- `http://localhost:8000/health` responde com status `healthy`
- `http://localhost:5000` mostra a UI do MLflow com os experimentos
- `http://localhost:8501` carrega a interface Streamlit

---

## Fase 9 — CI/CD com GitHub Actions

### Conceito
O **GitHub Actions** é um serviço de automação que executa pipelines diretamente no GitHub. O pipeline definido aqui executa:
1. Os testes automatizados a cada push ou PR
2. O deploy na nuvem apenas quando os testes passam na branch `main`

O benefício: erros são detectados antes de chegarem ao servidor de produção.

### Arquivo: `.github/workflows/ci.yml`

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Instalar dependências
        run: pip install -r requirements.txt

      - name: Executar testes
        run: pytest tests/ -v --tb=short

  deploy:
    needs: test                  # só executa se "test" passar
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy no Render
        run: |
          curl -X POST "${{ secrets.RENDER_DEPLOY_HOOK }}"
```

### Arquivo: `.github/dependabot.yml`

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
```

### Passos

**9.1 — Fazer o push do projeto para o GitHub**
```bash
git add .
git commit -m "feat: projeto CancerGuard completo"
git push origin main
```

**9.2 — Na aba Actions do GitHub, verificar se o pipeline passou**

### Verificação ✅
- Aba **Actions** no GitHub mostra o job `test` com check verde ✅

---

## Fase 10 — Deploy na Nuvem (Render)

### Conceito
O Render é uma plataforma de cloud hosting que pode construir e servir containers Docker. A integração com o GitHub permite **deploy automático via webhook** — sempre que o GitHub Actions trigger o deploy (após testes passarem), o Render rebuilda e reinicia o container.

### Passos

**10.1 — Criar conta em [render.com](https://render.com)**

**10.2 — Criar um novo Web Service**
- Conectar o repositório GitHub
- Runtime: **Docker**
- Branch: `main`
- Porta: `8000`

**10.3 — Configurar variáveis de ambiente no painel do Render**
```
MLFLOW_TRACKING_URI = <URI do MLflow em produção>
LOG_DB_PATH         = predictions.db
```

**10.4 — Copiar o Deploy Hook URL e salvar como secret no GitHub**
- GitHub → Settings → Secrets → `RENDER_DEPLOY_HOOK`

**10.5 — Testar a URL de produção**
```bash
curl https://cancerguard-api.onrender.com/health
```

### Verificação ✅
- `https://cancerguard-api.onrender.com/health` retorna `{"status": "healthy", ...}`
- `https://cancerguard-api.onrender.com/docs` abre o Swagger UI

---

## Checklist Geral de Progresso

| Fase | Objetivo | Verificação |
|---|---|---|
| ✅ / ⬜ 1 | Ambiente e estrutura de pastas | `pip list` + estrutura de pastas criada |
| ✅ / ⬜ 2 | Treinamento + MLflow tracking | Run aparece em `http://localhost:5000` |
| ✅ / ⬜ 3 | Model Registry + stage Production | Modelo em `Production` no Registry |
| ✅ / ⬜ 4 | FastAPI core (schemas, model, main) | `/health` e `/predict` funcionando |
| ✅ / ⬜ 5 | Logger SQLite | Registros em `predictions.db` |
| ✅ / ⬜ 6 | Testes automatizados | `pytest tests/ -v` 100% verde |
| ✅ / ⬜ 7 | Streamlit UI | Formulário + predição em `localhost:8501` |
| ✅ / ⬜ 8 | Docker + Compose | Stack completa via `docker compose up` |
| ✅ / ⬜ 9 | GitHub Actions CI/CD | Pipeline verde no GitHub |
| ✅ / ⬜ 10 | Deploy no Render | API respondendo na URL de produção |

---

## Dicas para o Processo de Aprendizado

- **Não copie o código sem ler.** Cada trecho tem um "porquê" — escreva uma linha comentando o que ele faz antes de executar.
- **Quebre intencionalmente.** Após cada fase, tente causar um erro (ex: remova um campo do payload, passe um valor negativo) e observe o que acontece.
- **Leia os logs.** O `uvicorn` e o MLflow geram logs detalhados — eles são a primeira fonte de diagnóstico.
- **Use o Swagger.** `http://localhost:8000/docs` é um cliente HTTP integrado. Teste todos os casos de borda antes de escrever os testes automatizados.
- **Commite após cada fase.** Ter cada etapa funcionando em um commit separado facilita o rollback e ajuda a visualizar o progresso.

---

*CancerGuard API · Roteiro de Implementação · Atualizado em abril de 2026*
