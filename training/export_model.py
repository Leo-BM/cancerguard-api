import os
import sys
import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME   = "cancerguard-svm"
STAGE        = "Production"
OUTPUT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def export() -> None:
    print(f"[export] Conectando ao MLflow: {TRACKING_URI}")
    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient()

    versions = client.get_latest_versions(MODEL_NAME, stages=[STAGE])
    if not versions:
        print(f"[export] ERRO: nenhum modelo '{MODEL_NAME}' em stage '{STAGE}'.")
        print("         Execute as Fases 2 e 3 para treinar e promover o modelo.")
        sys.exit(1)

    version = versions[0]
    print(f"[export] Modelo: {MODEL_NAME} v{version.version}  run_id={version.run_id}")

    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{STAGE}")
    model_path = os.path.join(OUTPUT_DIR, "model.joblib")
    joblib.dump(model, model_path)
    print(f"[export] model.joblib salvo em: {model_path}")

    # scaler salvo como artefato do run, não embutido no modelo registrado
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
