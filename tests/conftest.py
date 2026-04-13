"""
conftest.py — Configuração global de testes para o CancerGuard.

Por que este arquivo existe?
-----------------------------
Quando o TestClient do FastAPI importa `app.main`, o lifespan (startup) é
executado, o que chama `load_model()`. Essa função tenta se conectar ao
MLflow em http://localhost:5000 e baixar artefatos — infraestrutura que não
existe no CI/CD (GitHub Actions) e que não deveria ser necessária para validar
a lógica da aplicação.

A solução é usar `unittest.mock.patch` para substituir os objetos reais por
mocks (objetos falsos que se comportam da forma que precisamos) antes de
qualquer teste rodar.

Fluxo do mock:
    1. O conftest define um fixture de sessão com `autouse=True`
    2. O pytest aplica esse fixture automaticamente a TODOS os testes
    3. O patch intercepta as variáveis globais `_model`, `_scaler`, `_explainer`
       em `app.model` e as substitui por MagicMock antes dos testes
    4. `load_model` em si também é mockado para não tentar conectar ao MLflow
    5. Após todos os testes, o patch é revertido automaticamente (o `with`
       garante isso via context manager)
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="session", autouse=True)
def mock_model_globals():
    """
    Substitui os singletons do app.model por objetos fake para toda a sessão.

    scope="session"  → criado uma vez, reutilizado por TODOS os testes.
                       Usar "function" recriaria o mock a cada teste (lento).
    autouse=True     → aplicado automaticamente. Os arquivos de teste não
                       precisam declarar esse fixture explicitamente.

    Por que mockamos cada variável separadamente?
    O `load_model()` atribui valores às variáveis globais _model, _scaler e
    _explainer. Ao mockamos as variáveis diretamente, não precisamos que
    load_model() rode de verdade — e nenhuma requisição de rede é feita.
    """

    # --- Mock do modelo SVM ---
    # predict_proba() retorna array shape (n_amostras, n_classes)
    # índice 0 = malignant, índice 1 = benign (comportamento do sklearn)
    # Valor fixo de 0.85 → predição sempre "malignant" + risk_level "high"
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])

    # --- Mock do scaler ---
    # transform() deve retornar array com shape (1, 30) — mesma forma da entrada
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 30))

    # --- Mock do SHAP explainer ---
    # shap_values() retorna lista de arrays por classe.
    # [0] → array shape (1, 30) para a classe maligna
    # Usamos valores aleatórios com seed fixo para reprodutibilidade
    rng = np.random.default_rng(seed=42)
    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = [rng.standard_normal((1, 30))]

    # Mock do PredictionLogger (SQLite)
    # O TestClient do Starlette moderno não dispara o lifespan quando usado
    # fora de um context manager, então prediction_logger fica como None no
    # módulo. Mockamos diretamente para que prediction_logger.log() funcione.
    mock_logger = MagicMock()

    # patch() intercepta os atributos nos módulos e os substitui durante o
    # bloco `with`. Ao sair do bloco, os originais são restaurados.
    with (
        patch("app.model._model", mock_model),
        patch("app.model._scaler", mock_scaler),
        patch("app.model._explainer", mock_explainer),
        patch("app.model.load_model"),         # impede conexão ao MLflow
        patch("app.main.prediction_logger", mock_logger),  # impede acesso ao SQLite
    ):
        yield  # os testes rodam aqui, com os mocks ativos
