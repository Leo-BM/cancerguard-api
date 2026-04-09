# MLflow — Rastreamento de Experimentos

← [Voltar ao índice](./README.md)

---

## 1. Conceitos Utilizados

| Conceito | Papel no projeto |
|---|---|
| **Experiment** | Agrupamento lógico dos runs — `"cancerguard"` |
| **Run** | Execução individual de treinamento com parâmetros, métricas e artefatos |
| **Artifact** | Arquivos salvos por run: modelo serializado, `scaler.joblib` |
| **Model Registry** | Repositório central de modelos versionados com controle de estágio |
| **Stage** | Estado do modelo: `None → Staging → Production → Archived` |

---

## 2. O que é Registrado por Run

```python
with mlflow.start_run(run_name="svm-rbf-baseline"):

    # Parâmetros (hiperparâmetros do modelo)
    mlflow.log_params({"kernel": "rbf", "C": 1.0, "gamma": "scale"})

    # Métricas de avaliação
    mlflow.log_metrics({
        "recall":    0.968,
        "accuracy":  0.982,
        "precision": 0.975,
        "f1":        0.971
    })

    # Artefatos
    mlflow.sklearn.log_model(model, "svm_model")
    mlflow.log_artifact("scaler.joblib")

    # Tags de rastreabilidade
    mlflow.set_tags({"author": "Leo-BM", "dataset_version": "breast_cancer_v1"})
```

---

## 3. Hierarquia de Experimentos

```
MLflow Experiment: "cancerguard"
    │
    ├── Run: "svm-linear-baseline"
    │       params:    {kernel: linear, C: 1.0}
    │       metrics:   {recall: 0.879, accuracy: 0.951}
    │       artifacts: [model/, scaler.joblib]
    │
    ├── Run: "svm-poly-v1"
    │       params:    {kernel: poly, C: 1.0, degree: 3}
    │       metrics:   {recall: 0.948, accuracy: 0.979}
    │       artifacts: [model/, scaler.joblib]
    │
    └── Run: "svm-rbf-baseline"  ← PROMOVIDO PARA PRODUCTION
            params:    {kernel: rbf, C: 1.0, gamma: scale}
            metrics:   {recall: 0.968, accuracy: 0.982}
            artifacts: [model/, scaler.joblib]
```

---

## 4. Model Registry — Ciclo de Vida

```
         Treinamento
              │
              ▼
        [Run registrado]
              │
    register_model.py
              │
              ▼
          [None]
              │
     validação manual
              │
              ▼
         [Staging]
              │
    aprovação / testes
              │
              ▼
        [Production]  ←── API carrega daqui
              │
    modelo substituído
              │
              ▼
         [Archived]
```

### Regra principal

A API (`app/model.py`) carrega **exclusivamente** o modelo com `stage="Production"`. Isso garante:
- Mudanças de modelo são controladas e auditáveis
- Rollback é feito revertendo o estágio no MLflow Registry, sem rebuild do container

---

## 5. Visualizando os Experimentos

```bash
# Localmente
mlflow ui

# Acessa em: http://localhost:5000
```

Ou via Docker Compose (já configurado):
```bash
docker compose up
# MLflow UI disponível em: http://localhost:5000
```

---

## 6. Promovendo um Modelo para Production

```python
# training/register_model.py
import mlflow

client = mlflow.MlflowClient()

# Registrar o modelo a partir de um run
model_uri = f"runs:/{run_id}/svm_model"
mv = mlflow.register_model(model_uri, "cancerguard-svm")

# Promover para Staging
client.transition_model_version_stage(
    name="cancerguard-svm",
    version=mv.version,
    stage="Staging"
)

# Promover para Production após validação
client.transition_model_version_stage(
    name="cancerguard-svm",
    version=mv.version,
    stage="Production"
)
```

---

## 7. Carregando o Modelo na API

```python
# app/model.py
import mlflow.sklearn

model = mlflow.sklearn.load_model("models:/cancerguard-svm/Production")
```

O modelo em "Production" é carregado uma única vez no startup e mantido em memória (padrão singleton).
