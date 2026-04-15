import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import PredictionInput, PredictionOutput
from app.model import load_model, predict
from app.logging_config import PredictionLogger


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cancerguard")

prediction_logger: PredictionLogger = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global prediction_logger
    logger.info("Iniciando CancerGuard API...")
    load_model()                      
    prediction_logger = PredictionLogger()  
    logger.info("Modelo carregado e logger inicializado.")
    yield


app = FastAPI(
    title="CancerGuard API",
    version="1.0",
    description="Classificação de tumores mamários com SVM RBF e explicabilidade SHAP.",
    lifespan=lifespan,
)

@app.get("/health")
def health():
    return {"status": "healthy", "model": "svm-rbf", "version": "1.0"}


@app.post("/predict", response_model=PredictionOutput)
def predict_endpoint(payload: PredictionInput):
    try:
        result = predict(payload.model_dump())
        prediction_logger.log(
            payload.model_dump(),
            result["prediction"],
            result["probability_malignant"],
        )
        return result

    except Exception as e:
        logger.error(f"Erro na predição: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
