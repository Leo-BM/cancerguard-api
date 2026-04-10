import mlflow
from mlflow import MlflowClient

def register_and_promote(run_id: str):
    client = MlflowClient()


    model_uri = f"runs:/{run_id}/svm_model"
    mv = mlflow.register_model(model_uri, "cancerguard-svm")
    print(f"Modelo registrado: versão {mv.version}")

    client.transition_model_version_stage(
        name="cancerguard-svm",
        version=mv.version,
        stage="Staging",
    )
    print("Promovido para Staging")

    client.transition_model_version_stage(
        name="cancerguard-svm",
        version=mv.version,
        stage="Production",
    )
    print("Promovido para Production")

if __name__ == "__main__":
    run_id = input("Cole o Run ID do MLflow: ")
    register_and_promote(run_id)