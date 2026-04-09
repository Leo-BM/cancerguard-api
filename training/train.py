import numpy as np
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score

def train():
    # 1. Carregar o dataset direto do scikit-learn (sem baixar nada)
    data = load_breast_cancer()
    X, y = data.data, data.target

    # 2. Divisão treino/teste — stratify=y mantém proporção de classes em ambos os splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Normalização — fit SOMENTE no treino, transform em ambos
    #    Evita data leakage: o scaler não pode "ver" os dados de teste
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 4. Treinamento do SVM RBF — probability=True habilita predict_proba (necessário para SHAP)
    model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
    model.fit(X_train_scaled, y_train)

    # 5. Avaliação no conjunto de teste
    y_pred = model.predict(X_test_scaled)
    metrics = {
        "recall":    recall_score(y_test, y_pred),
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
    }
    print("Métricas:", metrics)

    # 6. Registrar tudo no MLflow
    mlflow.set_experiment("cancerguard")

    with mlflow.start_run(run_name="svm-rbf-baseline"):
        mlflow.log_params({"kernel": "rbf", "C": 1.0, "gamma": "scale"})
        mlflow.log_metrics(metrics)

        # Salvar o scaler como artefato — será necessário na predição
        joblib.dump(scaler, "scaler.joblib")
        mlflow.log_artifact("scaler.joblib")

        signature = infer_signature(X_train_scaled, model.predict_proba(X_train_scaled))
        input_example = X_train_scaled[:1]
        mlflow.sklearn.log_model(model, "svm_model", signature=signature, input_example=input_example)
        mlflow.set_tags({"author": "Leo-BM", "dataset_version": "breast_cancer_v1"})

        print(f"\nRun ID: {mlflow.active_run().info.run_id}")
        print("Guarde este Run ID — será usado na Fase 3!")

if __name__ == "__main__":
    train()