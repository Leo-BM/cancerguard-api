# API Spec — CancerGuard API

← [Voltar ao índice](./README.md)

---

## Visão Geral

| Campo | Valor |
|---|---|
| Base URL (local) | `http://localhost:8000` |
| Base URL (produção) | `https://cancerguard-api.onrender.com` |
| Documentação interativa | `{base_url}/docs` (Swagger UI) |
| Formato | JSON |
| Autenticação | Nenhuma (escopo de portfólio) |

---

## Endpoints

### `GET /health`

Verifica se a API está operacional e qual modelo está carregado.

**Response 200:**
```json
{
  "status": "healthy",
  "model": "svm-rbf",
  "version": "1.0"
}
```

---

### `POST /predict`

Recebe os dados de um exame e retorna a classificação do tumor.

**Request Body** — `application/json`:

```json
{
  "mean_radius": 14.0,
  "mean_texture": 19.0,
  "mean_perimeter": 91.0,
  "mean_area": 654.0,
  "mean_smoothness": 0.095,
  "mean_compactness": 0.109,
  "mean_concavity": 0.112,
  "mean_concave_points": 0.074,
  "mean_symmetry": 0.181,
  "mean_fractal_dimension": 0.057,
  "radius_se": 0.37,
  "texture_se": 0.87,
  "perimeter_se": 2.64,
  "area_se": 28.0,
  "smoothness_se": 0.005,
  "compactness_se": 0.021,
  "concavity_se": 0.032,
  "concave_points_se": 0.011,
  "symmetry_se": 0.019,
  "fractal_dimension_se": 0.003,
  "worst_radius": 16.0,
  "worst_texture": 25.0,
  "worst_perimeter": 105.0,
  "worst_area": 819.0,
  "worst_smoothness": 0.132,
  "worst_compactness": 0.240,
  "worst_concavity": 0.290,
  "worst_concave_points": 0.140,
  "worst_symmetry": 0.290,
  "worst_fractal_dimension": 0.082
}
```

**Response 200:**
```json
{
  "prediction": "malignant",
  "probability_malignant": 0.83,
  "risk_level": "high",
  "top_features": [
    {"feature": "worst_concave_points", "shap_value": 0.61},
    {"feature": "mean_concave_points",  "shap_value": 0.48},
    {"feature": "worst_perimeter",      "shap_value": 0.39}
  ]
}
```

**Response 422** — Validação falhou (campo inválido ou fora do range):
```json
{
  "detail": [
    {
      "loc": ["body", "mean_radius"],
      "msg": "value is not a valid float",
      "type": "type_error.float"
    }
  ]
}
```

**Response 500** — Erro interno:
```json
{
  "detail": "Internal server error"
}
```

---

## Schema de Entrada — `PredictionInput`

Todos os 30 campos são obrigatórios e do tipo `float`.

| Campo | Constraint | Descrição |
|---|---|---|
| `mean_radius` | `> 0` | Raio médio do tumor |
| `mean_texture` | `> 0` | Textura média |
| `mean_perimeter` | `> 0` | Perímetro médio |
| `mean_area` | `> 0` | Área média |
| `mean_smoothness` | `> 0` e `< 1` | Suavidade média (proporção) |
| `mean_compactness` | `> 0` | Compacidade média |
| `mean_concavity` | `> 0` | Concavidade média |
| `mean_concave_points` | `> 0` | Pontos côncavos médios |
| `mean_symmetry` | `> 0` | Simetria média |
| `mean_fractal_dimension` | `> 0` | Dimensão fractal média |
| `radius_se` … `fractal_dimension_se` | `> 0` | Erro padrão das 10 features |
| `worst_radius` … `worst_fractal_dimension` | `> 0` | Pior valor das 10 features |

---

## Schema de Saída — `PredictionOutput`

| Campo | Tipo | Valores possíveis |
|---|---|---|
| `prediction` | `string` | `"malignant"` ou `"benign"` |
| `probability_malignant` | `float` | `0.0` – `1.0` |
| `risk_level` | `string` | `"high"` · `"medium"` · `"low"` |
| `top_features` | `array` | Lista de `{feature: string, shap_value: float}` |

### Lógica de `risk_level`

```
probability_malignant >= 0.7  →  "high"
probability_malignant >= 0.4  →  "medium"
probability_malignant <  0.4  →  "low"
```

Thresholds conservadores: preferir falso positivo (alto risco desnecessário) a falso negativo (câncer não detectado).

---

## Exemplos de Uso

**curl:**
```bash
curl -X POST https://cancerguard-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"mean_radius": 14.0, "mean_texture": 19.0, ...}'
```

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"mean_radius": 14.0, "mean_texture": 19.0, ...}
)
result = response.json()
print(result["prediction"], result["probability_malignant"])
```
