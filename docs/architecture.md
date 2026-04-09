# Arquitetura do Sistema — CancerGuard API

← [Voltar ao índice](./README.md)

---

## 1. Estilo Arquitetural

O sistema adota uma arquitetura **em camadas (layered)** com os seguintes princípios:

- **Separação de responsabilidades** — API, modelo, schemas e logging são módulos independentes
- **Injeção de dependência via MLflow** — a API não embute o modelo; carrega do registry em tempo de execução
- **Containerização como unidade de deploy** — cada serviço é um container isolado

---

## 2. Diagrama de Contexto

```
┌──────────────────────────────────────────────────────────────────┐
│                         USUÁRIOS                                  │
│                                                                    │
│  [Usuário Técnico]              [Usuário Não-Técnico]             │
│   POST /predict (JSON)           Interface Streamlit              │
└──────────┬──────────────────────────────┬────────────────────────-┘
           │                              │
           ▼                              ▼
┌──────────────────┐         ┌──────────────────────┐
│   FastAPI        │◄────────│   Streamlit App       │
│   (porta 8000)   │         │   (porta 8501)        │
└────────┬─────────┘         └──────────────────────-┘
         │
         ▼
┌──────────────────┐         ┌──────────────────────┐
│  Model Layer     │────────►│   MLflow Registry     │
│  (SVM RBF)       │         │   (Model Store)       │
└────────┬─────────┘         └──────────────────────-┘
         │
         ├──────────────────────────────────────────────┐
         ▼                                              ▼
┌──────────────────┐                       ┌───────────────────────┐
│  SHAP Explainer  │                       │  Logger (SQLite)      │
│  (Explicabilidade│                       │  (Auditoria)          │
└──────────────────┘                       └───────────────────────┘
```

---

## 3. Diagrama de Arquitetura em Camadas

```
┌─────────────────────────────────────────────────────────┐
│                 CAMADA DE APRESENTAÇÃO                   │
│                                                          │
│   ┌──────────────────────┐  ┌────────────────────────┐  │
│   │   Streamlit UI       │  │   FastAPI /docs        │  │
│   │   (porta 8501)       │  │   (Swagger auto-gen)   │  │
│   └──────────┬───────────┘  └────────────────────────┘  │
└──────────────┼──────────────────────────────────────────-┘
               │ HTTP POST /predict
┌──────────────┼──────────────────────────────────────────-┐
│              ▼           CAMADA DE API                   │
│   ┌──────────────────────────────────────┐               │
│   │          FastAPI (main.py)            │               │
│   │  ┌──────────────┐ ┌───────────────┐  │               │
│   │  │  /predict    │ │   /health     │  │               │
│   │  └──────┬───────┘ └───────────────┘  │               │
│   │         │ validate (Pydantic)         │               │
│   │  ┌──────▼───────┐                    │               │
│   │  │  schemas.py  │                    │               │
│   │  └──────┬───────┘                    │               │
│   └─────────┼────────────────────────────┘               │
└─────────────┼───────────────────────────────────────────-┘
              │
┌─────────────┼───────────────────────────────────────────-┐
│             ▼          CAMADA DE MODELO                  │
│   ┌──────────────────────────────────────┐               │
│   │              model.py                 │               │
│   │  ┌──────────────┐  ┌──────────────┐  │               │
│   │  │ load_model() │  │  predict()   │  │               │
│   │  └──────┬───────┘  └──────┬───────┘  │               │
│   └─────────┼─────────────────┼──────────┘               │
└─────────────┼─────────────────┼───────────────────────────┘
              │                 │
              ▼                 ▼
┌─────────────────────┐  ┌──────────────────────────────┐
│   MLflow Registry   │  │  SHAP Explainer              │
│   (Model Store)     │  │  (Feature Importance)        │
└─────────────────────┘  └──────────────────────────────┘
                                          │
                               ┌──────────▼──────────┐
                               │   Logger / SQLite    │
                               │   (Auditoria)        │
                               └─────────────────────-┘
```

---

## 4. Fluxo Completo de uma Requisição

```
1.  Cliente → POST /predict (JSON com 30 features)
2.  FastAPI recebe a requisição
3.  Pydantic valida tipos, ranges e campos obrigatórios
     → Inválido: HTTP 422 (sem atingir o modelo)
     → Válido: passa para model.py
4.  model.py verifica cache em memória
     → Cold start: carrega modelo do MLflow Registry
     → Warm: usa modelo já em memória
5.  StandardScaler normaliza as 30 features
6.  SVM RBF executa predict_proba → probabilidade por classe
7.  Lógica de risco classifica em "high" / "medium" / "low"
8.  SHAP calcula top features da predição individual
9.  PredictionOutput é montado com todos os campos
10. Logger persiste a predição no SQLite
11. FastAPI retorna HTTP 200 com JSON de resposta
```

---

## 5. Decisões de Arquitetura

| Decisão | Alternativa descartada | Justificativa |
|---|---|---|
| Modelo carregado em memória (singleton) | Carregar por requisição | Evita latência de I/O em cada chamada |
| Pydantic na borda da API | Validação manual | Erros descritivos, zero boilerplate |
| MLflow Registry como fonte do modelo | Modelo embarcado no container | Permite troca de modelo sem rebuild da imagem |
| Logger isolado em módulo próprio | Logging direto no endpoint | Facilita troca futura para PostgreSQL ou serviço externo |
| Streamlit desacoplada da API | Monolito | As duas apps são deployáveis de forma independente |
