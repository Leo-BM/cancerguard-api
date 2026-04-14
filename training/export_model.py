"""
Exportação do modelo para deploy sem servidor MLflow (Fase 10 — Render).

Por que este script existe?
    O Render não tem acesso ao servidor MLflow local. Em vez de subir um
    MLflow em produção (custo + complexidade extra), exportamos os artefatos
    já aprovados (modelo SVM + scaler) para arquivos .joblib.

    Esses arquivos são commitados no repositório e copiados para a imagem
    Docker pelo Dockerfile, permitindo que a API carregue o modelo sem
    depender de nenhum servidor externo — ideal para portfólio.

Resultado:
    model.joblib   ← SVM treinado (salvo na raiz do projeto)
    scaler.joblib  ← StandardScaler ajustado nos dados de treino (raiz)

Uso:
    source .venv/bin/activate
    MLFLOW_TRACKING_URI=http://localhost:5001 python training/export_model.py
"""

import os
import sys
import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

# ── Configuração ──────────────────────────────────────────────────────────────
# A URI pode vir da variável de ambiente ou usar o default do macOS (5001).
# No Docker Compose local, use http://localhost:5001.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME   = "cancerguard-svm"
STAGE        = "Production"

# Raiz do projeto: dois níveis acima de training/export_model.py
OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def export() -> None:
    print(f"[export] Conectando ao MLflow: {TRACKING_URI}")
    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient()

    # 1. Localiza a versão em Production no Registry
    versions = client.get_latest_versions(MODEL_NAME, stages=[STAGE])
    if not versions:
        print(f"[export] ERRO: nenhum modelo '{MODEL_NAME}' em stage '{STAGE}'.")
        print("         Execute as Fases 2 e 3 para treinar e promover o modelo.")
        sys.exit(1)

    version = versions[0]
    print(f"[export] Modelo: {MODEL_NAME} v{version.version}  run_id={version.run_id}")

    # 2. Carrega o modelo SVM do Registry e serializa com joblib
    #    mlflow.sklearn.load_model retorna o objeto scikit-learn original
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{STAGE}")
    model_path = os.path.join(OUTPUT_DIR, "model.joblib")
    joblib.dump(model, model_path)
    print(f"[export] model.joblib salvo em: {model_path}")

    # 3. Baixa o scaler.joblib do run original e re-serializa
    #    O scaler foi salvo como artefato do run — não está embutido no modelo.
    #    download_artifacts() retorna o caminho local do arquivo baixado.
    local_scaler_path = client.download_artifacts(version.run_id, "scaler.joblib")
    scaler = joblib.load(local_scaler_path)
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"[export] scaler.joblib salvo em: {scaler_path}")

    print("\n[export] Exportação concluída com sucesso.")
    print("\nPróximos passos:")
    print("  git add model.joblib scaler.joblib")
    print("  git commit -m 'feat(phase-10): artefatos do modelo para deploy no Render'")
    print("  git push origin main")


if __name__ == "__main__":
    export()
