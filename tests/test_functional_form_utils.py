"""Konu 09 fonksiyonel biçim hesaplarının temel sayısal denetimleri."""

import numpy as np
import pytest

from core.data_registry import load_dataset
from core.functional_form_utils import (add_wage1_quadratic_columns, center_quadratic_model, log_level_percent_change,
    quadratic_discrete_change, quadratic_marginal_effect, quadratic_turning_point, standardized_regression, wage1_model_comparison)


@pytest.fixture(scope="module")
def wage_data():
    return load_dataset("wage1")


def test_standardization_log_and_quadratic_helpers(wage_data) -> None:
    """Temel dönüşümler ders notu büyüklükleriyle uyumludur."""
    data = add_wage1_quadratic_columns(wage_data)
    assert np.allclose(data.expersq, data.exper ** 2) and np.allclose(data.tenursq, data.tenure ** 2)
    standardized = standardized_regression(data, "lwage", ("educ", "exper", "tenure"))
    assert tuple(standardized.standardized_coefficients) == pytest.approx((0.479, 0.105, 0.300), abs=0.002)
    assert log_level_percent_change(0.06) == pytest.approx(6.0)
    assert log_level_percent_change(0.06, exact=True) == pytest.approx(6.1837, abs=0.001)
    assert quadratic_marginal_effect(4, -0.2, 5) == pytest.approx(2)
    assert quadratic_discrete_change(4, -0.2, 10) == pytest.approx(-0.2)
    turn = quadratic_turning_point(4, -0.1, np.arange(30))
    assert turn.value == pytest.approx(20)
    assert (turn.kind, turn.inside_data_range) == ("tepe", True)


def test_wage1_m4_and_centering_reuse_joint_f(wage_data) -> None:
    """M4 ve ortak kare testi ile merkezleme invariance'ı doğrulanır."""
    data = add_wage1_quadratic_columns(wage_data); comparison = wage1_model_comparison(data)
    m4 = comparison.models["M4"]
    assert (m4.nobs, m4.ssr, comparison.quadratic_joint_test.f_from_ssr) == pytest.approx((526, 93.911, 20.887), abs=0.01)
    assert comparison.quadratic_joint_test.formulas_match
    centered = center_quadratic_model(data, "lwage", "exper", "expersq", ("educ", "tenure", "tenursq"), center=10)
    assert centered.centered_result.coefficients[centered.centered_column] == pytest.approx(0.01746, abs=0.0001)
    assert centered.fitted_max_difference < 1e-10 and centered.residual_max_difference < 1e-10
