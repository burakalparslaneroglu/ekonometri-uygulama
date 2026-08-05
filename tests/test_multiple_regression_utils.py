import numpy as np
import pandas as pd
import pytest

from core.data_registry import load_dataset
from core.multiple_regression_utils import (
    add_konu05_derived_columns, adjusted_r_squared_from_r_squared, fit_multiple_ols,
    partial_regression_data, predict_multiple_value, profile_prediction_difference,
)


def _wage():
    return fit_multiple_ols(load_dataset("wage1"), "wage", ("educ", "exper", "tenure"))


def test_wage_models_and_fwl_match_notebook_values() -> None:
    result = _wage()
    assert result.coefficients["const"] == pytest.approx(-2.8727, abs=1e-3)
    assert result.coefficients["educ"] == pytest.approx(0.5990, abs=1e-3)
    assert result.coefficients["exper"] == pytest.approx(0.0223, abs=1e-3)
    assert result.coefficients["tenure"] == pytest.approx(0.1693, abs=1e-3)
    assert result.r_squared == pytest.approx(0.3064, abs=1e-4)
    assert result.adjusted_r_squared == pytest.approx(0.3024, abs=1e-4)
    partial = partial_regression_data(result, "educ")
    assert partial.slopes_match
    assert partial.partial_slope == pytest.approx(result.coefficients["educ"], abs=1e-10)


def test_log_wage_hprice_scales_and_profile_contributions() -> None:
    wage = fit_multiple_ols(load_dataset("wage1"), "lwage", ("educ", "exper", "tenure"))
    assert wage.coefficients["educ"] == pytest.approx(0.0920, abs=1e-3)
    raw = load_dataset("hprice1"); house = add_konu05_derived_columns(raw, "hprice1")
    assert house["lotsize1000"].equals(raw["lotsize"] / 1000)
    assert house["sqrft100"].equals(raw["sqrft"] / 100)
    result = fit_multiple_ols(house, "price", ("lotsize1000", "sqrft100", "bdrms"))
    assert result.coefficients["sqrft100"] == pytest.approx(12.2778, abs=1e-3)
    assert result.coefficients["lotsize1000"] == pytest.approx(2.0677, abs=1e-3)
    profiles = {"lotsize1000": 7.0, "sqrft100": 20.0, "bdrms": 3.0}, {"lotsize1000": 8.0, "sqrft100": 22.0, "bdrms": 4.0}
    comparison = profile_prediction_difference(result, *profiles)
    assert comparison["difference"] == pytest.approx(sum(comparison["contributions"].values()))
    assert predict_multiple_value(result, profiles[0]) == pytest.approx(comparison["prediction_a"])


def test_validation_common_sample_and_clear_errors() -> None:
    frame = pd.DataFrame({"y": [1, 2, 3, 4, 5, np.nan], "x1": [1, 2, 3, 4, 5, 6], "x2": [1, 2, np.inf, 5, 7, 9]})
    original = frame.copy(deep=True)
    result = fit_multiple_ols(frame, "y", ("x1", "x2"))
    assert result.nobs == 4
    assert frame.equals(original)
    assert result.adjusted_r_squared == pytest.approx(adjusted_r_squared_from_r_squared(result.r_squared, result.nobs, 2))
    with pytest.raises(ValueError, match="benzersiz"):
        fit_multiple_ols(frame, "y", ("x1", "x1"))
    with pytest.raises(ValueError, match="ayrı ayrı tahmin"):
        fit_multiple_ols(pd.DataFrame({"y": [1, 2, 3, 4], "x1": [1, 2, 3, 4], "x2": [2, 4, 6, 8]}), "y", ("x1", "x2"))
    with pytest.raises(ValueError, match="sonlu"):
        predict_multiple_value(_wage(), {"educ": np.inf, "exper": 2, "tenure": 1})
    with pytest.raises(ValueError, match="serbestlik"):
        adjusted_r_squared_from_r_squared(0.5, 3, 2)
