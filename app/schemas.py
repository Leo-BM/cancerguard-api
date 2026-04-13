from pydantic import BaseModel, Field
from typing import List


# ---------------------------------------------------------------------------
# PredictionInput
# Representa os dados de entrada da API: as 30 features do dataset Wisconsin.
#
# Field(..., gt=0) significa:
#   - "..." → campo obrigatório (sem valor padrão)
#   - gt=0  → greater than 0, valor deve ser positivo
#
# O Pydantic valida esses constraints ANTES de qualquer código de negócio
# ser executado. Se um campo vier negativo ou ausente, a FastAPI retorna
# HTTP 422 automaticamente — o modelo nunca chega a ser acionado.
#
# mean_smoothness tem também lt=1 (less than 1) pois é uma proporção [0, 1].
# ---------------------------------------------------------------------------

class PredictionInput(BaseModel):
    # Grupo 1 — Mean (média dos núcleos celulares)
    mean_radius:             float = Field(..., gt=0)
    mean_texture:            float = Field(..., gt=0)
    mean_perimeter:          float = Field(..., gt=0)
    mean_area:               float = Field(..., gt=0)
    mean_smoothness:         float = Field(..., gt=0, lt=1)
    mean_compactness:        float = Field(..., gt=0)
    mean_concavity:          float = Field(..., gt=0)
    mean_concave_points:     float = Field(..., gt=0)
    mean_symmetry:           float = Field(..., gt=0)
    mean_fractal_dimension:  float = Field(..., gt=0)

    # Grupo 2 — SE (erro padrão de cada medida)
    radius_se:               float = Field(..., gt=0)
    texture_se:              float = Field(..., gt=0)
    perimeter_se:            float = Field(..., gt=0)
    area_se:                 float = Field(..., gt=0)
    smoothness_se:           float = Field(..., gt=0)
    compactness_se:          float = Field(..., gt=0)
    concavity_se:            float = Field(..., gt=0)
    concave_points_se:       float = Field(..., gt=0)
    symmetry_se:             float = Field(..., gt=0)
    fractal_dimension_se:    float = Field(..., gt=0)

    # Grupo 3 — Worst (pior valor observado nos núcleos)
    worst_radius:            float = Field(..., gt=0)
    worst_texture:           float = Field(..., gt=0)
    worst_perimeter:         float = Field(..., gt=0)
    worst_area:              float = Field(..., gt=0)
    worst_smoothness:        float = Field(..., gt=0)
    worst_compactness:       float = Field(..., gt=0)
    worst_concavity:         float = Field(..., gt=0)
    worst_concave_points:    float = Field(..., gt=0)
    worst_symmetry:          float = Field(..., gt=0)
    worst_fractal_dimension: float = Field(..., gt=0)


# ---------------------------------------------------------------------------
# FeatureImportance
# Representa uma única feature e seu valor SHAP para aquela predição.
# shap_value > 0 → empurrou para "malignant"; < 0 → empurrou para "benign"
# ---------------------------------------------------------------------------

class FeatureImportance(BaseModel):
    feature:    str
    shap_value: float


# ---------------------------------------------------------------------------
# PredictionOutput
# Contrato de saída da API. O response_model no endpoint garante que
# a FastAPI valida e serializa exatamente este formato.
#
# risk_level: "high" (≥0.7), "medium" (≥0.4), "low" (<0.4)
# top_features: as 3 features com maior impacto absoluto no SHAP
# ---------------------------------------------------------------------------

class PredictionOutput(BaseModel):
    prediction:            str
    probability_malignant: float
    risk_level:            str
    top_features:          List[FeatureImportance]
