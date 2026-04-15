# Deployment — CancerGuard API

← [Voltar ao índice](./README.md)

---

## 1. Ambientes

| Ambiente | Como subir | URL |
|---|---|---|
| Local | `docker compose up` | `http://localhost:8000` |
| Staging | Branch de feature + Pull Request | — |
| Produção | Push na `main` → CI/CD automático | `https://cancerguard-api.onrender.com` |

---

## 2. Dockerfile

O build segue a estratégia de **dependências antes do código** para maximizar o cache de layers:

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# shap==0.47.0 pode precisar compilar extensão C++ (_cext.cc) no Linux se não houver wheel.
# g++ é instalado antes do pip e o cache do apt removido para não inflar a imagem final.
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

# Layer pesada — só rebuilda se requirements.txt mudar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layer de código — rebuilda a cada mudança nos módulos
COPY app/ ./app/
COPY training/ ./training/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Ordem das layers (cache Docker)

```
Layer 1: python:3.13-slim           ← base, nunca muda
Layer 2: apt-get install g++        ← compilador C++ para SHAP
Layer 3: COPY requirements.txt      ← invalida só se as deps mudarem
Layer 4: RUN pip install            ← cache pesado (~5 min na 1ª vez, compila SHAP)
Layer 5: COPY app/ training/        ← invalidado a cada push de código
Layer 6: CMD uvicorn                ← ponto de entrada
```

---

## 3. Docker Compose

```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5001:5000"  # host:container (5000 bloqueada pelo macOS AirPlay)
    command: >
      mlflow server
      --host 0.0.0.0
      --backend-store-uri file:///mlflow/mlruns
      --default-artifact-root file:///mlflow/mlruns
    volumes:
      - ./mlruns:/mlflow/mlruns   # bind mount — usa modelo já treinado localmente
    healthcheck:
      test: ["CMD-SHELL", "wget -qO- http://localhost:5000/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 8
      start_period: 15s

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000  # rede interna Docker usa 5000
    depends_on:
      mlflow:
        condition: service_healthy   # aguarda healthcheck do mlflow passar
    volumes:
      - ./predictions.db:/app/predictions.db   # persiste log de predições no host

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
```

```
docker compose up
         │
         ├── mlflow (porta host 5001)  ← sobe primeiro
         │       └── bind mount: ./mlruns (modelo já treinado localmente)
         │
         ├── api (porta 8000)     ← sobe após mlflow
         │       └── conecta em http://mlflow:5000 (rede interna)
         │
         └── streamlit (porta 8501)  ← sobe após api
                 └── conecta em http://api:8000
```

### Comandos úteis

```bash
# Subir tudo
docker compose up

# Subir em background
docker compose up -d

# Rebuild após mudanças
docker compose up --build

# Ver logs da API
docker compose logs api -f

# Parar tudo
docker compose down
```

---

## 4. Variáveis de Ambiente

Configuradas via `.env` localmente ou via painel do Render em produção.

| Variável | Descrição | Default local |
|---|---|---|
| `MLFLOW_TRACKING_URI` | URI do servidor MLflow | `http://mlflow:5000` |
| `PORT` | Porta da FastAPI | `8000` |
| `API_BASE_URL` | URL base consumida pela Streamlit | `http://localhost:8000` |
| `LOG_DB_PATH` | Caminho do SQLite de logs | `predictions.db` |

**`.env.example`** (commitado no repositório — sem valores reais):
```
MLFLOW_TRACKING_URI=http://localhost:5000
PORT=8000
API_BASE_URL=http://localhost:8000
LOG_DB_PATH=predictions.db
```

---

## 5. Pipeline CI/CD (GitHub Actions)

```yaml
# .github/workflows/ci.yml
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
          python-version: "3.13"
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --tb=short

  deploy:
    needs: test                                  # só roda se test passou
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'          # só na main
    env:
      DEPLOY_HOOK: ${{ secrets.RENDER_DEPLOY_HOOK }}
    steps:
      - name: Deploy no Render
        run: |
          if [ -z "$DEPLOY_HOOK" ]; then
            echo "RENDER_DEPLOY_HOOK não configurado — deploy ignorado (configure na Fase 10)"
            exit 0
          fi
          curl -X POST "$DEPLOY_HOOK"
```

### Regras do Pipeline

- `test` roda em todo push e PR
- `deploy` roda **somente** quando `test` passou **e** o push foi na `main`
- Pull Requests não disparam deploy
- `RENDER_DEPLOY_HOOK` é um secret no GitHub Actions — nunca exposto em código

### Fluxo visual

```
push na main
     │
     ▼
GitHub Actions
     │
     ├── Job: test
     │    ├── setup Python 3.13
     │    ├── pip install
     │    └── pytest ──► PASS
     │                     │
     └── Job: deploy        │
          └── curl Render ◄─┘
                  │
                  ▼
            Render.com
       (pull imagem + restart)
```

---

## 6. Deploy no Render.com

### Por que Render e não AWS/Azure/GCP?
O Render oferece free tier real (sem cartão de crédito, sem risco de cobrança acidental), integração nativa com GitHub e deploy via Docker com 4 cliques. Para um portfólio de Data Science/MLOps, o objetivo é demonstrar a API funcionando na nuvem — não configurar infra cloud enterprise.

### Solução para o MLflow em produção

O Render não tem acesso ao MLflow local. O modelo é exportado como arquivo `.joblib` e commitado no repositório:

```bash
# 1. Exportar (com MLflow local rodando)
MLFLOW_TRACKING_URI=http://localhost:5001 python training/export_model.py

# 2. Commitar os artefatos (deve ser feito antes de qualquer docker build)
git add model.joblib scaler.joblib
git commit -m "feat(phase-10): artefatos do modelo para deploy"
git push origin main
```

O `app/model.py` detecta automaticamente o ambiente:
- `MLFLOW_TRACKING_URI` definida → carrega do MLflow (local/Docker)
- `MLFLOW_TRACKING_URI` ausente → carrega dos arquivos `.joblib` (Render)

### Passos no painel do Render

1. Criar conta em [render.com](https://render.com)
2. **New → Web Service → Connect GitHub repo**
3. Runtime: **Docker**, Branch: `main`, Port: `8000`
4. Variável de ambiente: `LOG_DB_PATH=predictions.db`
   - **Não** configurar `MLFLOW_TRACKING_URI` (ausência ativa o carregamento via `.joblib`)
5. Copiar o **Deploy Hook URL** → GitHub → Settings → Secrets → `RENDER_DEPLOY_HOOK`
6. A partir daí: cada push na `main` que passe nos testes faz deploy automático

> **Cold start:** o free tier hiberna após ~15 min de inatividade. Primeira requisição após hibernação demora ~30s. Aceitável para portfólio.

### Deploy da Interface (Streamlit Cloud)

1. Acessar [share.streamlit.io](https://share.streamlit.io)
2. Conectar repositório GitHub
3. Selecionar `streamlit_app/app.py` como entry point
4. Configurar `API_BASE_URL` apontando para a URL do Render
