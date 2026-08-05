import warnings

import numpy as np
import pandas as pd
import pytest

from core.data_registry import load_dataset
from core.konu04_questions import KONU04_QUESTION_TYPES, generate_konu04_question
from core.model_utils import (
    MAX_UNIT_SCALE, MIN_UNIT_SCALE, fit_simple_ols, format_number, observation_decomposition,
    r_squared_from_error_sum, r_squared_from_model_sum, rescaled_coefficients, sum_of_squares,
    transform_functional_form, validate_sum_of_squares, validate_unit_scale,
)


def _result():
    return fit_simple_ols(pd.DataFrame({"y": [1.0, 3.0, 5.0, 7.0], "x": [0.0, 1.0, 2.0, 3.0]}), "y", "x")


def test_decomposition_sums_and_r_squared_agree_with_ols() -> None:
    result = _result()
    decomposition = observation_decomposition(result, 2)
    sums = sum_of_squares(result)
    assert decomposition.total_deviation == pytest.approx(decomposition.model_deviation + decomposition.residual_deviation)
    assert sums.total == pytest.approx(20.0)
    assert sums.model == pytest.approx(20.0)
    assert sums.error == pytest.approx(0.0)
    assert validate_sum_of_squares(sums)
    assert r_squared_from_model_sum(sums) == pytest.approx(result.r_squared)
    assert r_squared_from_error_sum(sums) == pytest.approx(result.r_squared)


def test_sum_of_squares_rejects_zero_total_variation_and_invalid_data() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="Bağımlı değişkende örneklem değişimi"):
            fit_simple_ols(pd.DataFrame({"y": [2.0, 2.0, 2.0], "x": [1.0, 2.0, 3.0]}), "y", "x")
    assert not any(item.category is RuntimeWarning for item in caught)
    result = _result()
    object.__setattr__(result, "residuals", pd.Series([0.0, np.inf, 0.0, 0.0]))
    with pytest.raises(ValueError, match="sonlu"):
        sum_of_squares(result)


def test_unit_conversion_follows_algebra_and_preserves_r_squared() -> None:
    result = _result()
    intercept, slope = rescaled_coefficients(result, dependent_scale=1000.0, explanatory_scale=0.01)
    assert intercept == pytest.approx(1000.0)
    assert slope == pytest.approx(200000.0)
    scaled = fit_simple_ols(pd.DataFrame({"y": [1000.0, 3000.0, 5000.0, 7000.0], "x": [0.0, 0.01, 0.02, 0.03]}), "y", "x")
    assert scaled.intercept == pytest.approx(intercept)
    assert scaled.slope == pytest.approx(slope)
    assert scaled.r_squared == pytest.approx(result.r_squared)


@pytest.mark.parametrize("value", (MIN_UNIT_SCALE, MAX_UNIT_SCALE, 0.125, 250.0))
def test_unit_scale_accepts_positive_bounds_and_custom_values(value: float) -> None:
    assert validate_unit_scale(value) == value


@pytest.mark.parametrize("value", (0.0, -1.0, float("nan"), float("inf"), float("-inf"), 1e-7, 1e7))
def test_unit_scale_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError):
        validate_unit_scale(value)


def test_small_and_zero_numbers_are_formatted_distinctly() -> None:
    assert format_number(0.000005414) == "5.4140e-06"
    assert format_number(0.0) == "0"


@pytest.mark.parametrize("form", ("düzey-düzey", "log-düzey", "düzey-log", "log-log"))
def test_four_functional_forms_transform_positive_data(form: str) -> None:
    data = pd.DataFrame({"y": [1.0, 2.0, 4.0], "x": [2.0, 4.0, 8.0]})
    transformed = transform_functional_form(data, "y", "x", form)  # type: ignore[arg-type]
    assert len(transformed.data) == len(data)
    assert np.isfinite(transformed.data.to_numpy()).all()


def test_log_transform_reports_invalid_observations_without_dropping() -> None:
    data = pd.DataFrame({"y": [1.0, 0.0, 4.0], "x": [2.0, 4.0, 8.0]})
    with pytest.raises(ValueError, match="sıfır veya negatiftir"):
        transform_functional_form(data, "y", "x", "log-düzey")


def test_konu04_questions_are_deterministic_and_cover_every_type() -> None:
    result = fit_simple_ols(load_dataset("wage1"), "wage", "educ")
    sums, decomposition = sum_of_squares(result), observation_decomposition(result, 0)
    generated = {
        generate_konu04_question("konu04:wage1:düzey-düzey:0", index, result, sums, decomposition, "düzey-düzey", "Saatlik ücret", "Eğitim").question_type
        for index in range(len(KONU04_QUESTION_TYPES))
    }
    assert generated == set(KONU04_QUESTION_TYPES)
