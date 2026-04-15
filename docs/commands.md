# Comandos de Terminal — Referência do Dia a Dia

← [Voltar ao índice](./README.md)

> Referência dos comandos usados durante a implementação do CancerGuard API. Atualizado conforme o projeto avança.

---

## Ambiente Virtual

```bash
# Criar o ambiente virtual
python3 -m venv .venv

# Ativar (macOS / Linux)
source .venv/bin/activate

# Confirmar que o ambiente está ativo (deve apontar para .venv)
which python

# Instalar dependências
pip install -r requirements.txt

# Listar pacotes instalados
pip list

# Verificar versão de um pacote específico
pip list | grep mlflow
pip list | grep fastapi
```

---

## Git

```bash
# Verificar estado atual do repositório
git status

# Ver o que será commitado (antes de git commit)
git diff --cached --stat

# Adicionar arquivos específicos
git add training/train.py
git add requirements.txt
git add README.md

# Adicionar todos os arquivos não ignorados
git add .

# Commitar com mensagem estruturada
git commit -m "feat(training): descrição do que foi feito"

# Enviar para o GitHub
git push origin main

# Ver histórico de commits
git log --oneline

# Remover arquivo do tracking do Git (sem deletar do disco)
git rm -r --cached nome-da-pasta/
```

**Prefixos de commit (Conventional Commits):**
| Prefixo | Uso |
|---|---|
| `feat` | Nova funcionalidade |
| `fix` | Correção de bug |
| `docs` | Documentação |
| `refactor` | Refatoração sem mudança de comportamento |
| `test` | Adição ou correção de testes |
| `chore` | Tarefas de manutenção (dependências, configs) |

---

## MLflow

```bash
# ⚠️ Atenção macOS: a porta 5000 é bloqueada pelo AirPlay Receiver (ControlCenter).
# Use sempre a porta 5001:
mlflow ui --port 5001
# Acesse: http://localhost:5001

# Verificar se a porta 5000 está ocupada (diagnóstico)
lsof -i :5000 | head -5

# Executar o treinamento (em outro terminal, com .venv ativo)
python training/train.py

# Registrar e promover o modelo no Registry
python training/register_model.py

# Exportar modelo para deploy no Render (Fase 10)
# MLflow deve estar rodando antes de executar
MLFLOW_TRACKING_URI=http://localhost:5001 python training/export_model.py
# Resultado: model.joblib e scaler.joblib criados na raiz do projeto
```

---

## Projeto — Estrutura

```bash
# Criar estrutura de pastas
mkdir -p app training tests .github/workflows streamlit_app

# Criar arquivos __init__.py
touch app/__init__.py training/__init__.py tests/__init__.py

# Verificar estrutura atual
find . -not -path './.venv/*' -not -path './.git/*' -not -path './mlruns/*' | sort
```

---

## Verificações Rápidas

```bash
# Versão do Python
python --version

# Verificar se um arquivo existe
ls -la training/train.py

# Ver conteúdo do .gitignore
cat .gitignore

# Ver o que o Git está rastreando
git ls-files
```

---

## FastAPI (Fase 4)

```bash
# Subir a API em modo de desenvolvimento (reload automático)
# Passe a URI do MLflow — use 5001 no macOS
MLFLOW_TRACKING_URI=http://localhost:5001 uvicorn app.main:app --reload --port 8000

# Acessar documentação interativa
# http://localhost:8000/docs   (Swagger UI)
# http://localhost:8000/redoc  (ReDoc)

# Testar o endpoint de health manualmente
curl http://localhost:8000/health

# Testar predição via curl
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"mean_radius": 20.57, "mean_texture": 17.77, "mean_perimeter": 132.9, "mean_area": 1326.0, "mean_smoothness": 0.08474, "mean_compactness": 0.07864, "mean_concavity": 0.0869, "mean_concave_points": 0.07017, "mean_symmetry": 0.1812, "mean_fractal_dimension": 0.05667, "radius_se": 0.5435, "texture_se": 0.7339, "perimeter_se": 3.398, "area_se": 74.08, "smoothness_se": 0.005225, "compactness_se": 0.01308, "concavity_se": 0.0186, "concave_points_se": 0.0134, "symmetry_se": 0.01389, "fractal_dimension_se": 0.003532, "worst_radius": 24.99, "worst_texture": 23.41, "worst_perimeter": 158.8, "worst_area": 1956.0, "worst_smoothness": 0.1238, "worst_compactness": 0.1866, "worst_concavity": 0.2416, "worst_concave_points": 0.186, "worst_symmetry": 0.275, "worst_fractal_dimension": 0.08902}'
```

---

## Testes Automatizados (Fase 6)

```bash
# Rodar todos os testes
pytest tests/ -v

# Apenas testes unitários
pytest tests/test_predict.py -v

# Apenas testes de integração
pytest tests/test_api.py -v

# Com relatório de cobertura
pytest tests/ --cov=app --cov-report=term-missing

# Resultado esperado: 36 passed, cobertura ~80%
```

---

## Streamlit UI (Fase 7)

```bash
# Subir a interface visual (a API deve estar rodando em :8000)
streamlit run streamlit_app/app.py --server.port 8501
# Acesse: http://localhost:8501

# Subir apontando para uma API em URL diferente
API_BASE_URL=http://outro-host:8000 streamlit run streamlit_app/app.py

# Ordem correta para subir tudo localmente (3 terminais):
# Terminal 1: mlflow ui --port 5001
# Terminal 2: MLFLOW_TRACKING_URI=http://localhost:5001 uvicorn app.main:app --reload --port 8000
# Terminal 3: streamlit run streamlit_app/app.py --server.port 8501
```

---

## Docker + Compose (Fase 8)

> **Pré-requisito:** Docker Desktop instalado e em execução.
> Verifique: `docker --version` e `docker compose version`

### Verificação rápida antes de subir

```bash
# Verificar que o Docker está rodando
docker info

# Verificar que o docker compose está disponível
docker compose version

# Confirmar que mlruns/ existe (o modelo precisa estar treinado localmente)
ls mlruns/

# Confirmar que o arquivo predictions.db existe (para o bind mount da API)
# Se não existir, crie um arquivo vazio ANTES de subir os containers:
touch predictions.db
```

### Build das imagens

```bash
# Build de todas as imagens definidas no docker-compose.yml
# --no-cache força reconstrução completa (útil quando há dúvidas de cache)
docker compose build

# Build só da imagem da API
docker compose build api

# Build só da imagem do Streamlit
docker compose build streamlit

# Build com output detalhado (útil para depurar erros de build)
docker compose build --progress=plain
```

### Subir os containers

```bash
# Subir todos os serviços em foreground (logs visíveis no terminal)
docker compose up

# Subir em background (detached mode)
docker compose up -d

# Subir com build forçado antes de iniciar (equivale a build + up)
docker compose up --build

# Subir apenas um serviço específico (e suas dependências)
docker compose up mlflow
docker compose up api
```

### Verificar status e logs

```bash
# Ver status de todos os containers
docker compose ps

# Ver logs de todos os serviços (útil para depurar)
docker compose logs

# Ver logs de um serviço específico e seguir em tempo real
docker compose logs -f api
docker compose logs -f mlflow
docker compose logs -f streamlit

# Ver apenas as últimas N linhas de log
docker compose logs --tail=50 api
```

### Testar os serviços no ar

```bash
# API: health check
curl http://localhost:8000/health

# API: predição de exemplo
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"mean_radius": 20.57, "mean_texture": 17.77, "mean_perimeter": 132.9, "mean_area": 1326.0, "mean_smoothness": 0.08474, "mean_compactness": 0.07864, "mean_concavity": 0.0869, "mean_concave_points": 0.07017, "mean_symmetry": 0.1812, "mean_fractal_dimension": 0.05667, "radius_se": 0.5435, "texture_se": 0.7339, "perimeter_se": 3.398, "area_se": 74.08, "smoothness_se": 0.005225, "compactness_se": 0.01308, "concavity_se": 0.0186, "concave_points_se": 0.0134, "symmetry_se": 0.01389, "fractal_dimension_se": 0.003532, "worst_radius": 24.99, "worst_texture": 23.41, "worst_perimeter": 158.8, "worst_area": 1956.0, "worst_smoothness": 0.1238, "worst_compactness": 0.1866, "worst_concavity": 0.2416, "worst_concave_points": 0.186, "worst_symmetry": 0.275, "worst_fractal_dimension": 0.08902}'

# MLflow UI: http://localhost:5001
# Streamlit UI: http://localhost:8501
```

### Parar e limpar

```bash
# Parar todos os containers (mantém as imagens)
docker compose down

# Parar e remover volumes anônimos também
docker compose down -v

# Remover imagens buildadas localmente (libera espaço)
docker compose down --rmi local

# Ver todas as imagens Docker no sistema
docker images

# Remover imagens não utilizadas (prune)
docker image prune

# Limpeza geral: containers parados + imagens sem uso + cache de build
docker system prune
```

### Inspecionar containers em execução

```bash
# Entrar no shell de um container em execução (para depurar)
docker compose exec api bash
docker compose exec streamlit bash

# Ver variáveis de ambiente de um container
docker compose exec api env

# Ver o IP dos containers na rede Docker
docker compose exec api cat /etc/hosts
```

### Rede interna Docker (referência)

| De → Para | URL correta a usar | Por quê |
|---|---|---|
| Host → API | `http://localhost:8000` | Porta mapeada para o host |
| Host → MLflow | `http://localhost:5001` | Porta mapeada (5001 no macOS) |
| API → MLflow | `http://mlflow:5000` | Nome do serviço na rede interna |
| Streamlit → API | `http://api:8000` | Nome do serviço na rede interna |

> **Atenção:** dentro dos containers, use sempre o nome do serviço (ex: `mlflow`, `api`).
> `localhost` dentro de um container refere-se ao próprio container, não ao host nem a outro container.

---

## GitHub Actions CI/CD (Fase 9)

> O pipeline roda automaticamente no GitHub a cada push. Os comandos abaixo são para consultar e depurar localmente.

### Verificar o pipeline no GitHub

```bash
# Ver o status do último pipeline no terminal (requer GitHub CLI instalado)
# Instalar: brew install gh
gh run list --limit 5

# Ver o log detalhado de um run específico
gh run view <run-id> --log

# Abrir a aba Actions do repositório no browser
gh repo view --web
# Em seguida: aba "Actions"
```

### Simular o job de testes localmente (antes de fazer push)

```bash
# Rodar exatamente o que o GitHub Actions vai rodar
# (garante que não há dependência faltando no requirements.txt)
pytest tests/ -v --tb=short

# Com cobertura (relatório completo)
pytest tests/ --cov=app --cov-report=term-missing -v
```

### Ver o arquivo de workflow

```bash
cat .github/workflows/ci.yml
```

### Secrets do GitHub (configuração via CLI)

```bash
# Listar secrets configurados no repositório
gh secret list

# Adicionar o Deploy Hook do Render como secret (Fase 10)
gh secret set RENDER_DEPLOY_HOOK
# Cole a URL quando solicitado (não aparece no terminal)
```

---

## Deploy no Render (Fase 10)

### Verificar se a API está no ar

```bash
# Health check da URL de produção
curl https://cancerguard-api.onrender.com/health

# Predição de exemplo na URL de produção
curl -X POST https://cancerguard-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d @- << 'EOF'
{
  "mean_radius": 14.0, "mean_texture": 19.0, "mean_perimeter": 91.0,
  "mean_area": 654.0, "mean_smoothness": 0.095, "mean_compactness": 0.109,
  "mean_concavity": 0.112, "mean_concave_points": 0.074, "mean_symmetry": 0.181,
  "mean_fractal_dimension": 0.057, "radius_se": 0.37, "texture_se": 0.87,
  "perimeter_se": 2.64, "area_se": 28.0, "smoothness_se": 0.005,
  "compactness_se": 0.021, "concavity_se": 0.032, "concave_points_se": 0.011,
  "symmetry_se": 0.019, "fractal_dimension_se": 0.003, "worst_radius": 16.0,
  "worst_texture": 25.0, "worst_perimeter": 105.0, "worst_area": 819.0,
  "worst_smoothness": 0.132, "worst_compactness": 0.240, "worst_concavity": 0.290,
  "worst_concave_points": 0.140, "worst_symmetry": 0.290, "worst_fractal_dimension": 0.082
}
EOF
```

### Disparar deploy manual via Deploy Hook

```bash
# Substitua pela URL real do seu Render Deploy Hook
curl -X POST "https://api.render.com/deploy/srv-XXXXXX?key=YYYYYY"
```

### Verificar logs de produção (via GitHub CLI)

```bash
# Ver o último run do pipeline (que disparou o deploy)
gh run list --limit 3
gh run view --log
```

---

*Atualizado na Fase 10 · CancerGuard API*
