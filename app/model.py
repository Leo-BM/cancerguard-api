import os
import joblib
import numpy as np
import shap
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

# ---------------------------------------------------------------------------
# Singletons — carregados uma única vez no startup da API (via load_model()).
# Usar variáveis globais aqui é intencional: o modelo SVM e o scaler são
# objetos grandes e imutáveis durante a vida da aplicação. Carregar a cada
# requisição seria proibitivo em latência.
# ---------------------------------------------------------------------------
_model:     object = None
_scaler:    object = None
_explainer: object = None

# A ordem desta lista deve ser IDÊNTICA à ordem das colunas usada no
# treinamento (load_breast_cancer() retorna as features nessa sequência).
# Qualquer divergência produziria predições silenciosamente erradas.
FEATURE_NAMES = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
    "mean_smoothness", "mean_compactness", "mean_concavity",
    "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area",
    "worst_smoothness", "worst_compactness", "worst_concavity",
    "worst_concave_points", "worst_symmetry", "worst_fractal_dimension",
]


def load_model() -> None:
    """
    Carrega o modelo SVM, o scaler e inicializa o explainer SHAP.
    Chamado uma única vez no startup da FastAPI (lifespan).

    Por que separar modelo e scaler?
    O scaler.joblib foi salvo como artefato do *run* de treinamento via
    mlflow.log_artifact(), não embutido dentro do modelo registrado no
    Registry. Por isso, mlflow.artifacts.load_artifacts("models:/...") não
    o encontraria. A solução é:
      1. Descobrir o run_id da versão em Production via MlflowClient
      2. Baixar o artefato diretamente daquele run com download_artifacts()
    """
    global _model, _scaler, _explainer

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    # 1. Carrega o modelo SVM do Registry (stage Production)
    _model = mlflow.sklearn.load_model("models:/cancerguard-svm/Production")

    # 2. Recupera o run_id da versão atualmente em Production
    #    get_latest_versions() retorna uma lista; pegamos o primeiro (e único)
    client = MlflowClient()
    versions = client.get_latest_versions("cancerguard-svm", stages=["Production"])
    run_id = versions[0].run_id

    # 3. Baixa o scaler.joblib do run original e desserializa com joblib
    #    download_artifacts() retorna o caminho local do arquivo baixado
    local_scaler_path = client.download_artifacts(run_id, "scaler.joblib")
    _scaler = joblib.load(local_scaler_path)

    # 4. Inicializa o KernelExplainer do SHAP
    #    - background: tensor de zeros usado como referência (baseline)
    #    - KernelExplainer funciona com qualquer modelo que tenha predict_proba
    #    - É mais lento que TreeExplainer, mas agnóstico ao tipo de modelo
    background = np.zeros((1, len(FEATURE_NAMES)))
    _explainer = shap.KernelExplainer(_model.predict_proba, background)


def _get_risk_level(prob: float) -> str:
    """
    Converte probabilidade de malignidade em categoria de risco.
    Thresholds definidos com base na documentação clínica do projeto
    (ver docs/data-model.md).
    """
    if prob >= 0.7:
        return "high"
    elif prob >= 0.4:
        return "medium"
    return "low"


def predict(input_data: dict) -> dict:
    """
    Recebe um dicionário com as 30 features e retorna o resultado completo.

    Fluxo:
      1. Monta array numpy na ordem correta de features
      2. Aplica o mesmo StandardScaler usado no treinamento
      3. Gera predição e probabilidades com o SVM
      4. Calcula valores SHAP para explicar a predição
      5. Retorna as 3 features com maior influência absoluta
    """
    # Monta a matriz de entrada respeitando a ordem de FEATURE_NAMES
    # shape: (1, 30) — uma amostra, 30 features
    X = np.array([[input_data[f] for f in FEATURE_NAMES]])

    # Aplica normalização (OBRIGATÓRIO: usa o mesmo scaler do treino)
    X_scaled = _scaler.transform(X)

    # predict_proba retorna [[prob_class_0, prob_class_1]]
    # No dataset Wisconsin com sklearn: índice 0 = malignant, índice 1 = benign
    proba = _model.predict_proba(X_scaled)[0]
    prob_malignant = float(proba[0])
    prediction = "malignant" if prob_malignant >= 0.5 else "benign"

    # SHAP: shap_values é uma lista com um array por classe
    # shap_values[0] → valores para classe malignant (índice 0)
    # shap_values[0][0] → valores da primeira (e única) amostra
    shap_values = _explainer.shap_values(X_scaled)
    importances = list(zip(FEATURE_NAMES, shap_values[0][0]))

    # Ordena por valor absoluto: features com maior impacto primeiro,
    # independente do sinal (positivo = empurra para maligno; negativo = benigno)
    importances.sort(key=lambda x: abs(x[1]), reverse=True)

    top_features = [
        {"feature": f, "shap_value": round(float(v), 4)}
        for f, v in importances[:3]
    ]

    return {
        "prediction": prediction,
        "probability_malignant": round(prob_malignant, 4),
        "risk_level": _get_risk_level(prob_malignant),
        "top_features": top_features,
    }
