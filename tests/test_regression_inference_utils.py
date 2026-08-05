"""Konu 07 ortak çıkarım katmanının sayısal doğrulamaları."""

import numpy as np
import pandas as pd
import pytest

from core.data_registry import load_dataset
from core.regression_inference_utils import (
    ci_test_equivalence, coefficient_confidence_interval, coefficient_test,
    critical_t_value, fit_ols_inference, format_p_value, scale_coefficient_inference,
    significance_stars, simulate_confidence_coverage, simulate_standard_errors,
)


@pytest.fixture(scope="module")
def wage_result():
    return fit_ols_inference(load_dataset("wage1"), "wage", ("educ", "exper", "tenure"))


def test_wage1_matches_documented_conventional_inference(wage_result) -> None:
    """WAGE1 sonuçları ders notundaki geleneksel EKK değerleriyle uyumludur."""
    assert (wage_result.nobs, wage_result.df_resid, wage_result.covariance_type) == (526, 522, "nonrobust")
    assert wage_result.coefficients["educ"] == pytest.approx(0.5990, abs=0.0002)
    assert wage_result.standard_errors["educ"] == pytest.approx(0.0513, abs=0.0002)
    assert wage_result.t_values_zero["educ"] == pytest.approx(11.679, abs=0.01)
    assert wage_result.p_values_two_sided_zero["exper"] == pytest.approx(0.064, abs=0.002)
    assert tuple(wage_result.confidence_intervals_95.loc["educ"]) == pytest.approx((0.498, 0.700), abs=0.002)


def test_hprice1_uses_raw_scale() -> None:
    """HPRICE1 Konu 07 modelinin ham ölçekli katsayıları doğrulanır."""
    result = fit_ols_inference(load_dataset("hprice1"), "price", ("lotsize", "sqrft", "bdrms"))
    assert (result.nobs, result.df_resid) == (88, 84)
    assert result.coefficients["sqrft"] == pytest.approx(0.1228, abs=0.0002)
    assert result.standard_errors["bdrms"] == pytest.approx(9.0101, abs=0.001)


def test_coefficient_tests_and_ci_equivalence(wage_result) -> None:
    """Sıfır dışı null, kuyruk seçimi ve GA eşdeğerliği denetlenir."""
    zero = coefficient_test(wage_result, "educ")
    half = coefficient_test(wage_result, "educ", null_value=0.5)
    high = coefficient_test(wage_result, "educ", null_value=0.75)
    experience = coefficient_test(wage_result, "exper", alternative="greater")
    assert zero.t_statistic == pytest.approx(11.68, abs=0.02)
    assert half.t_statistic == pytest.approx(1.93, abs=0.02) and not half.reject_null
    assert high.reject_null
    assert experience.p_value == pytest.approx(0.032, abs=0.002)
    assert ci_test_equivalence(half).consistent
    assert ci_test_equivalence(experience).applicable is False


def test_validation_and_presentation_rules(wage_result) -> None:
    """Geçersiz girdiler, katı eşikler ve öğrenci biçimlendirmesi denetlenir."""
    with pytest.raises(ValueError):
        fit_ols_inference(pd.DataFrame({"y": [1, 2, 3, 4], "x": [1, 1, 1, 1]}), "y", ("x",))
    with pytest.raises(ValueError):
        coefficient_test(wage_result, "educ", alpha=1.0)
    assert critical_t_value(10, 0.05, "two-sided") == pytest.approx(2.228, abs=0.001)
    assert format_p_value(0.0001) == "< 0.001"
    assert significance_stars(0.009) == "***" and significance_stars(0.10) == ""


def test_scale_and_vectorized_simulations_are_deterministic() -> None:
    """Ölçekleme ile iki benzetimin seed'li temel özellikleri doğrulanır."""
    scaled = scale_coefficient_inference(0.1228, 0.0965, 0.1491, 100)
    assert (scaled.scaled_estimate, scaled.scaled_lower, scaled.scaled_upper) == pytest.approx((12.28, 9.65, 14.91), abs=0.01)
    first = simulate_standard_errors(nobs=50, repetitions=500, seed=19)
    second = simulate_standard_errors(nobs=50, repetitions=500, seed=19)
    assert np.array_equal(first.slope_estimates, second.slope_estimates)
    assert first.empirical_slope_std == pytest.approx(first.mean_reported_standard_error, rel=0.15)
    coverage = simulate_confidence_coverage(nobs=50, repetitions=3000, seed=19)
    assert coverage.coverage_rate == pytest.approx(0.95, abs=0.03)
    lower, upper = coefficient_confidence_interval(fit_ols_inference(load_dataset("wage1"), "wage", ("educ", "exper", "tenure")), "educ")
    assert lower < 0.5 < upper
