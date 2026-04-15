# CancerGuard API

API de classificação de tumores mamários com rastreamento de experimentos, explicabilidade e interface visual — projeto de portfólio construído com práticas de MLOps.

---

## O Problema

Modelos com alta acurácia ainda podem falhar de forma crítica em diagnósticos oncológicos. Um classificador que erra sistematicamente tumores malignos tem consequências reais para pacientes. Por isso, a **métrica prioritária deste projeto é o Recall da classe maligna**: preferimos um falso positivo (dizer que é maligno quando é benigno) a um falso negativo (deixar passar um maligno).

---

## O Projeto

O CancerGuard API recebe medidas de núcleos celulares extraídas de imagens de biópsia (técnica FNA) e classifica o tumor como **maligno** ou **benigno**, junto com a probabilidade e as features mais relevantes para aquela predição — calculadas via SHAP.

**Dataset:** Breast Cancer Wisconsin (Diagnostic) — 569 amostras, 30 features numéricas contínuas, 2 classes (212 maligno / 357 benigno). Disponível via `sklearn.datasets`.

---

## Arquitetura

```
[Streamlit UI] ──HTTP──► [FastAPI] ──► [model.py] ──► [MLflow Registry]
    porta 8501               porta 8000      │               │
                                             │          SVM RBF Model
                                             │          StandardScaler
                                             │
                                        ┌────┴─────────────────────┐
                                        ▼                          ▼
                                 [SHAP Explainer]         [Logger / SQLite]
                                  (explicabilidade)         (auditoria)
```

- **FastAPI** — camada de API com validação automática via Pydantic
- **MLflow Registry** — versionamento e ciclo de vida do modelo (`Staging → Production`)
- **SHAP** — explicabilidade: quais features mais influenciaram cada predição
- **Streamlit** — interface visual para usuários não técnicos
- **SQLite** — log de auditoria de todas as predições recebidas
- **Docker + Compose** — containerização de todos os serviços
- **GitHub Actions** — CI/CD: testes automatizados a cada push

---

## Stack

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.13 |
| API | FastAPI + Uvicorn |
| Validação | Pydantic v2 |
| Modelo | scikit-learn (SVM RBF) |
| Experiment Tracking | MLflow |
| Explicabilidade | SHAP |
| Interface | Streamlit |
| Testes | pytest + httpx + pytest-cov |
| Containerização | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Deploy | Render |

---

## Resultados do Modelo

Treinado no dataset Wisconsin Breast Cancer com SVM kernel RBF (`C=1.0`, `gamma="scale"`):

| Métrica | Valor |
|---|---|
| Recall | **98.61%** |
| Accuracy | **98.24%** |
| Precision | **98.61%** |
| F1-Score | **98.61%** |

O `StandardScaler` é salvo como artefato junto ao modelo no MLflow — na predição, o mesmo scaler é obrigatório para evitar que as features cheguem ao SVM em escalas diferentes das aprendidas no treino.

---

## Endpoints da API

### `GET /health`
```json
{
  "status": "healthy",
  "model": "svm-rbf",
  "version": "1.0"
}
```

### `POST /predict`
Recebe as 30 features do exame e retorna:
```json
{
  "prediction": "malignant",
  "probability_malignant": 0.87,
  "risk_level": "high",
  "top_features": [
    { "feature": "worst_concave_points", "shap_value": 0.312 },
    { "feature": "worst_radius", "shap_value": 0.284 },
    { "feature": "mean_concave_points", "shap_value": 0.201 }
  ]
}
```

Documentação interativa disponível em `/docs` (Swagger UI) quando a API estiver rodando.

---

## Como Rodar Localmente

**Pré-requisitos:** Python 3.13, Git

```bash
# 1. Clonar e entrar no projeto
git clone https://github.com/Leo-BM/Mlops_Cancer_Breast.git
cd Mlops_Cancer_Breast

# 2. Criar e ativar o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Subir o MLflow
# Atenção: no macOS, a porta 5000 é ocupada pelo AirPlay Receiver.
# Use a porta 5001:
mlflow ui --port 5001
# Acesse: http://localhost:5001

# 5. Treinar o modelo (em outro terminal com .venv ativo)
python training/train.py
python training/register_model.py

# 6. Subir a API (passe a URI correta do MLflow)
MLFLOW_TRACKING_URI=http://localhost:5001 uvicorn app.main:app --reload --port 8000
# Acesse os docs: http://localhost:8000/docs

# 7. Subir a interface Streamlit (em outro terminal com .venv ativo)
streamlit run streamlit_app/app.py --server.port 8501
# Acesse: http://localhost:8501
```

## Como Rodar com Docker

**Pré-requisito:** Docker Desktop instalado e em execução.

```bash
# 1. Clonar e entrar no projeto
git clone https://github.com/Leo-BM/Mlops_Cancer_Breast.git
cd Mlops_Cancer_Breast

# 2. Garantir que o arquivo de log existe (necessário para o bind mount)
touch predictions.db

# 3. Build das imagens (primeira vez ~5 min — compila extensão C++ do SHAP)
docker compose build

# 4. Subir os 3 serviços
docker compose up
# MLflow:    http://localhost:5001
# API:       http://localhost:8000
# Streamlit: http://localhost:8501

# 5. Parar tudo
docker compose down
```

> **Atenção macOS:** o MLflow roda na porta `5001` do host (o AirPlay Receiver ocupa a `5000`). Dentro da rede Docker, os containers se comunicam via `mlflow:5000`.

## Estrutura do Projeto

```
Mlops_Cancer_Breast/
├── app/
│   ├── main.py              ← FastAPI: endpoints e lifespan
│   ├── model.py             ← carregamento do modelo e predição
│   ├── schemas.py           ← contratos de input/output (Pydantic)
│   └── logging_config.py   ← logger de predições (SQLite)
├── training/
│   ├── train.py             ← treinamento + rastreamento MLflow ✅
│   ├── register_model.py   ← promoção para Production no Registry
│   └── export_model.py     ← exporta modelo para deploy sem MLflow
├── streamlit_app/
│   └── app.py               ← interface visual
├── tests/
│   ├── test_predict.py      ← testes unitários do modelo
│   └── test_api.py          ← testes de integração da API
├── .github/workflows/
│   └── ci.yml               ← pipeline CI/CD (test + deploy)
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Status de Implementação

| Fase | Descrição | Status |
|---|---|---|
| 1 | Ambiente, estrutura e Git | ✅ Completo |
| 2 | Treinamento + MLflow tracking | ✅ Completo |
| 3 | MLflow Model Registry | ✅ Completo |
| 4 | FastAPI core | ✅ Completo |
| 5 | Logging SQLite | ✅ Completo |
| 6 | Testes automatizados | ✅ Completo — 36 testes, 80% cobertura |
| 7 | Interface Streamlit | ✅ Completo |
| 8 | Docker + Compose | ✅ Completo |
| 9 | GitHub Actions CI/CD | ✅ Completo |
| 10 | Deploy no Render | ⬜ Pendente |

---

## Variáveis de Ambiente

Copie `.env.example` para `.env` e ajuste conforme o ambiente:

```
MLFLOW_TRACKING_URI=http://localhost:5000
PORT=8000
API_BASE_URL=http://localhost:8000
LOG_DB_PATH=predictions.db
```

---

*Projeto de portfólio — Leonardo BM · 2026*
