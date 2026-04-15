from pydantic import BaseModel, Field
from typing import List



class PredictionInput(BaseModel):
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

class FeatureImportance(BaseModel):
    feature:    str
    shap_value: float



class PredictionOutput(BaseModel):
    prediction:            str
    probability_malignant: float
    risk_level:            str
    top_features:          List[FeatureImportance]
