import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import PredictionInput, PredictionOutput
from app.model import load_model, predict
from app.logging_config import PredictionLogger

# ---------------------------------------------------------------------------
# Logging padrão do Python — separado do logger de predições (SQLite).
# Este captura eventos da aplicação: startup, erros, etc.
# O formato inclui timestamp, nível e nome do logger para facilitar o debug.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cancerguard")

# Declarado como global para ser acessível dentro dos endpoints.
# O tipo é anotado para ajudar o editor, mas a atribuição real ocorre
# no lifespan (antes de qualquer requisição chegar).
prediction_logger: PredictionLogger = None


# ---------------------------------------------------------------------------
# Lifespan — substitui os eventos on_event("startup") / on_event("shutdown")
# deprecados no FastAPI moderno. Tudo antes do `yield` é startup; tudo
# depois seria shutdown (cleanup de recursos, se necessário).
#
# Por que carregar o modelo aqui e não no módulo?
# Importar model.py não deve ter efeitos colaterais (carregar modelo é lento).
# O lifespan garante que o carregamento ocorre apenas quando a API sobe,
# não quando o módulo é importado (ex: durante os testes com TestClient).
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global prediction_logger
    logger.info("Iniciando CancerGuard API...")
    load_model()                          # conecta ao MLflow e carrega modelo + scaler
    prediction_logger = PredictionLogger()  # abre conexão SQLite
    logger.info("Modelo carregado e logger inicializado.")
    yield
    # Aqui entraria cleanup (ex: fechar conexões) — não necessário agora


app = FastAPI(
    title="CancerGuard API",
    version="1.0",
    description="Classificação de tumores mamários com SVM RBF e explicabilidade SHAP.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# GET /health
# Endpoint simples para liveness check — usado pelo Docker e pelo Render
# para saber se o container está operacional.
# Não verifica o modelo ativamente (isso seria um readiness check).
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "healthy", "model": "svm-rbf", "version": "1.0"}


# ---------------------------------------------------------------------------
# POST /predict
# Endpoint principal. O FastAPI:
#   1. Deserializa o JSON e valida com PredictionInput (Pydantic)
#   2. Passa o objeto validado para predict_endpoint
#   3. Valida o retorno contra PredictionOutput antes de serializar
#
# response_model=PredictionOutput garante que campos extras não vazem
# e que o schema do Swagger esteja sempre correto.
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionOutput)
def predict_endpoint(payload: PredictionInput):
    try:
        result = predict(payload.model_dump())

        # Persiste a predição no SQLite para auditoria
        prediction_logger.log(
            payload.model_dump(),
            result["prediction"],
            result["probability_malignant"],
        )
        return result

    except Exception as e:
        # Loga o erro completo internamente mas retorna mensagem genérica
        # ao cliente — evita expor detalhes de implementação (OWASP A05)
        logger.error(f"Erro na predição: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
