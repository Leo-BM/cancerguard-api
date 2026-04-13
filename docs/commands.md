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

*Atualizado na Fase 7 · CancerGuard API*
