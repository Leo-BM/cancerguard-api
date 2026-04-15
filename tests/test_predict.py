import pytest
from app.model import predict, _get_risk_level

BENIGN_CASE = {
    "mean_radius": 11.42,
    "mean_texture": 20.38,
    "mean_perimeter": 77.58,
    "mean_area": 386.1,
    "mean_smoothness": 0.1425,
    "mean_compactness": 0.2839,
    "mean_concavity": 0.2414,
    "mean_concave_points": 0.1052,
    "mean_symmetry": 0.2597,
    "mean_fractal_dimension": 0.09744,
    "radius_se": 0.4956,
    "texture_se": 1.156,
    "perimeter_se": 3.445,
    "area_se": 27.23,
    "smoothness_se": 0.00911,
    "compactness_se": 0.07458,
    "concavity_se": 0.05661,
    "concave_points_se": 0.01867,
    "symmetry_se": 0.05963,
    "fractal_dimension_se": 0.009208,
    "worst_radius": 12.26,
    "worst_texture": 24.75,
    "worst_perimeter": 79.89,
    "worst_area": 471.4,
    "worst_smoothness": 0.1369,
    "worst_compactness": 0.1482,
    "worst_concavity": 0.1075,
    "worst_concave_points": 0.07431,
    "worst_symmetry": 0.2998,
    "worst_fractal_dimension": 0.07881,
}

MALIGNANT_CASE = {
    "mean_radius": 20.57,
    "mean_texture": 17.77,
    "mean_perimeter": 132.9,
    "mean_area": 1326.0,
    "mean_smoothness": 0.08474,
    "mean_compactness": 0.07864,
    "mean_concavity": 0.0869,
    "mean_concave_points": 0.07017,
    "mean_symmetry": 0.1812,
    "mean_fractal_dimension": 0.05667,
    "radius_se": 0.5435,
    "texture_se": 0.7339,
    "perimeter_se": 3.398,
    "area_se": 74.08,
    "smoothness_se": 0.005225,
    "compactness_se": 0.01308,
    "concavity_se": 0.0186,
    "concave_points_se": 0.0134,
    "symmetry_se": 0.01389,
    "fractal_dimension_se": 0.003532,
    "worst_radius": 24.99,
    "worst_texture": 23.41,
    "worst_perimeter": 158.8,
    "worst_area": 1956.0,
    "worst_smoothness": 0.1238,
    "worst_compactness": 0.1866,
    "worst_concavity": 0.2416,
    "worst_concave_points": 0.186,
    "worst_symmetry": 0.275,
    "worst_fractal_dimension": 0.08902,
}


class TestPredictOutput:

    def test_predict_returns_dict_with_all_keys(self):
        result = predict(MALIGNANT_CASE)
        assert "prediction" in result
        assert "probability_malignant" in result
        assert "risk_level" in result
        assert "top_features" in result

    def test_prediction_is_malignant_when_prob_above_threshold(self):
        result = predict(MALIGNANT_CASE)
        assert result["prediction"] == "malignant"

    def test_probability_is_float_between_0_and_1(self):
        result = predict(MALIGNANT_CASE)
        prob = result["probability_malignant"]
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_probability_is_rounded_to_4_decimals(self):
        result = predict(MALIGNANT_CASE)
        # round(0.85, 4) = 0.85 → str tem no máximo 4 casas decimais
        prob_str = str(result["probability_malignant"]).split(".")
        if len(prob_str) > 1:
            assert len(prob_str[1]) <= 4

    def test_top_features_is_non_empty_list(self):
        result = predict(BENIGN_CASE)
        assert isinstance(result["top_features"], list)
        assert len(result["top_features"]) > 0

    def test_top_features_has_at_most_3_items(self):
        result = predict(MALIGNANT_CASE)
        assert len(result["top_features"]) <= 3

    def test_top_features_each_has_feature_and_shap_value(self):
        result = predict(BENIGN_CASE)
        for item in result["top_features"]:
            assert "feature" in item
            assert "shap_value" in item
            assert isinstance(item["feature"], str)
            assert isinstance(item["shap_value"], float)


class TestRiskLevel:

    def test_risk_level_high_at_0_7(self):
        assert _get_risk_level(0.7) == "high"

    def test_risk_level_high_above_threshold(self):
        assert _get_risk_level(0.95) == "high"

    def test_risk_level_medium_at_0_4(self):
        assert _get_risk_level(0.4) == "medium"

    def test_risk_level_medium_between_thresholds(self):
        assert _get_risk_level(0.55) == "medium"

    def test_risk_level_low_below_0_4(self):
        assert _get_risk_level(0.39) == "low"

    def test_risk_level_low_near_zero(self):
        assert _get_risk_level(0.05) == "low"

    def test_risk_level_from_predict_is_valid_category(self):
        result = predict(MALIGNANT_CASE)
        assert result["risk_level"] in ("high", "medium", "low")

    def test_recall_priority(self):
        result = predict(MALIGNANT_CASE)
        assert result["risk_level"] == "high"
        assert result["probability_malignant"] > 0.3
