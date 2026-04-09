# CancerGuard API — Documentação Técnica

> Índice central da documentação do projeto. Cada arquivo cobre um domínio específico da arquitetura.

---

## Documentos

| Arquivo | Conteúdo | Audiência |
|---|---|---|
| [SDD.md](../CancerGuard_SDD.md) | Software Design Document completo (visão consolidada) | Todos |
| [architecture.md](./architecture.md) | Diagrama de contexto, camadas e fluxo de requisição | Engenheiros |
| [api-spec.md](./api-spec.md) | Contratos REST: endpoints, schemas, exemplos de request/response | Desenvolvedores, consumidores da API |
| [data-model.md](./data-model.md) | Dataset, features, pré-processamento e serialização do modelo | Data Scientists, MLEs |
| [mlflow-experiments.md](./mlflow-experiments.md) | Rastreamento de experimentos, Model Registry e ciclo de vida do modelo | Data Scientists, MLEs |
| [deployment.md](./deployment.md) | Docker, Docker Compose, ambientes e pipeline CI/CD | DevOps, Engenheiros |
| [security.md](./security.md) | Validação de input, segredos, dependências e exposição de dados | Todos |
| [testing.md](./testing.md) | Pirâmide de testes, testes unitários e de integração | Engenheiros |
| [monitoring.md](./monitoring.md) | Logging estruturado, auditoria de predições e métricas futuras | MLEs, DevOps |

---

## Estrutura do Repositório

```
cancerguard-api/
├── docs/                       ← você está aqui
│   ├── README.md
│   ├── architecture.md
│   ├── api-spec.md
│   ├── data-model.md
│   ├── mlflow-experiments.md
│   ├── deployment.md
│   ├── security.md
│   ├── testing.md
│   └── monitoring.md
│
├── app/                        ← FastAPI
├── streamlit_app/              ← Interface visual
├── training/                   ← Treinamento + MLflow
├── tests/                      ← Testes unitários e de integração
├── .github/workflows/          ← CI/CD
├── Dockerfile
├── docker-compose.yml
└── CancerGuard_SDD.md          ← SDD consolidado
```

---

## Visão Rápida do Sistema

```
[Streamlit UI] ──HTTP──► [FastAPI] ──► [model.py] ──► [MLflow Registry]
                                  │                         │
                                  │                    SVM RBF Model
                                  │                    StandardScaler
                                  │
                                  ├──► [SHAP Explainer]
                                  └──► [Logger / SQLite]
```

---

*CancerGuard API · github.com/Leo-BM*
