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
FROM python:3.11-slim

WORKDIR /app

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
Layer 1: python:3.11-slim           ← base, nunca muda
Layer 2: COPY requirements.txt      ← invalida só se as deps mudarem
Layer 3: RUN pip install            ← cache pesado (~2 min)
Layer 4: COPY app/ training/        ← invalidado a cada push de código
Layer 5: CMD uvicorn                ← ponto de entrada
```

---

## 3. Docker Compose

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0
    volumes:
      - mlflow_data:/mlflow

volumes:
  mlflow_data:
```

```
docker compose up
         │
         ├── mlflow (porta 5000)  ← sobe primeiro (depends_on)
         │       └── volume: mlflow_data (persistência dos runs)
         │
         └── api (porta 8000)     ← sobe após mlflow estar healthy
                 └── conecta em http://mlflow:5000
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
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --tb=short

  deploy:
    needs: test                                  # só roda se test passou
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'          # só na main
    steps:
      - name: Deploy to Render
        run: curl -X POST ${{ secrets.RENDER_DEPLOY_HOOK_URL }}
```

### Regras do Pipeline

- `test` roda em todo push e PR
- `deploy` roda **somente** quando `test` passou **e** o push foi na `main`
- Pull Requests não disparam deploy
- `RENDER_DEPLOY_HOOK_URL` é um secret no GitHub Actions — nunca exposto em código

### Fluxo visual

```
push na main
     │
     ▼
GitHub Actions
     │
     ├── Job: test
     │    ├── setup Python 3.11
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

1. Criar conta em [render.com](https://render.com)
2. **New → Web Service → Connect GitHub repo**
3. Selecionar **Docker** como runtime
4. Configurar variáveis de ambiente no painel
5. Adicionar o Deploy Hook URL como secret no GitHub Actions (`RENDER_DEPLOY_HOOK_URL`)
6. A partir daí: cada push na `main` que passe nos testes faz deploy automático

### Deploy da Interface (Streamlit Cloud)

1. Acessar [share.streamlit.io](https://share.streamlit.io)
2. Conectar repositório GitHub
3. Selecionar `streamlit_app/app.py` como entry point
4. Configurar `API_BASE_URL` apontando para a URL do Render
