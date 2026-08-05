"""Konu 08 ortak F hesaplarının temel sayısal denetimleri."""

import numpy as np
import pytest

from core.data_registry import load_dataset
from core.joint_inference_utils import (LinearRestriction, f_distribution_plot_data, joint_f_test,
    classify_restriction_system, nested_exclusion_f_test, overall_f_test, simulate_large_sample_joint_test,
    single_restriction_equivalence, validate_restriction_system)
from core.regression_inference_utils import coefficient_test, fit_ols_inference


@pytest.fixture(scope="module")
def wage_result():
    return fit_ols_inference(load_dataset("wage1"), "wage", ("educ", "exper", "tenure"))


def test_wage1_and_hprice1_joint_tests_match_documented_values(wage_result) -> None:
    """SSR/R²/matrix F hesapları gerçek Wooldridge değerleriyle uyumludur."""
    wage = nested_exclusion_f_test(load_dataset("wage1"), "wage", ("educ", "exper", "tenure"), ("educ",))
    home = nested_exclusion_f_test(load_dataset("hprice1"), "price", ("lotsize", "sqrft", "bdrms"), ("sqrft",))
    assert (wage.q, wage.df_denom, wage.f_from_ssr, wage.p_value) == pytest.approx((2, 522, 53.3099, 8.56e-22), rel=2e-3, abs=1e-12)
    assert wage.formulas_match and wage.ssr_unrestricted <= wage.ssr_restricted
    assert (home.q, home.df_denom, home.f_from_ssr, home.p_value) == pytest.approx((2, 84, 6.6102, 0.002157), rel=2e-3)
    assert overall_f_test(wage_result).f_statistic == pytest.approx(76.873, abs=0.01)


def test_general_restrictions_and_single_t_equivalence(wage_result) -> None:
    """Eşitlik/toplam kısıtları ve F=t² bağlantısı güvenli biçimde çalışır."""
    equal = joint_f_test(wage_result, (LinearRestriction({"exper": 1, "tenure": -1}, 0, "exper = tenure"),))
    total = joint_f_test(wage_result, (LinearRestriction({"exper": 1, "tenure": 1}, 1, "exper + tenure = 1"),))
    education = joint_f_test(wage_result, (LinearRestriction({"educ": 1}, 0, "educ = 0"),))
    t_test = coefficient_test(wage_result, "educ")
    assert equal.f_statistic >= 0 and total.f_statistic >= 0
    assert education.f_statistic == pytest.approx(136.410, abs=0.01)
    assert single_restriction_equivalence(t_test, education)
    with pytest.raises(ValueError):
        joint_f_test(wage_result, (LinearRestriction({"exper": 1}, 0, "a"), LinearRestriction({"exper": 2}, 0, "b")))


def test_f_plot_and_large_sample_simulation_are_finite_and_deterministic() -> None:
    """Grafik verisi ve batch benzetimi değişmez girdiyle tekrarlanabilir."""
    plot = f_distribution_plot_data(1000, 2, 522, 0.05)
    assert np.isfinite(plot["x"]).all() and np.isfinite(plot["density"]).all()
    first = simulate_large_sample_joint_test(sample_sizes=(25, 100), repetitions=300, seed=41)
    second = simulate_large_sample_joint_test(sample_sizes=(25, 100), repetitions=300, seed=41)
    assert first.rejection_rates.equals(second.rejection_rates)
    assert np.isfinite(first.standardized_slope_stds.to_numpy()).all()


def test_custom_restriction_validation_classification_and_f_values(wage_result) -> None:
    """Rütbe, tutarlılık ve custom matrix-F politikası açıkça doğrulanır."""
    exclusion = (LinearRestriction({"exper": 1}, 0, "exper=0"), LinearRestriction({"tenure": 1}, 0, "tenure=0"))
    equality = (LinearRestriction({"exper": 1, "tenure": -1}, 0, "exper=tenure"),)
    nonzero = (LinearRestriction({"educ": 1}, 0.5, "educ=.5"),)
    dependent = (LinearRestriction({"exper": 1}, 0, "a"), LinearRestriction({"exper": 2}, 0, "b"))
    inconsistent = (LinearRestriction({"exper": 1}, 0, "a"), LinearRestriction({"exper": 1}, 1, "b"))
    zero = (LinearRestriction({"exper": 0}, 0, "zero"),)
    assert validate_restriction_system(wage_result, exclusion).q == 2
    assert joint_f_test(wage_result, exclusion).f_statistic == pytest.approx(53.3099, abs=0.001)
    assert classify_restriction_system(wage_result, exclusion).nested_exclusion_applicable
    assert classify_restriction_system(wage_result, equality).restriction_type == "katsayı eşitliği"
    assert not classify_restriction_system(wage_result, equality).nested_exclusion_applicable
    assert joint_f_test(wage_result, equality).f_statistic >= 0 and joint_f_test(wage_result, nonzero).f_statistic >= 0
    assert not validate_restriction_system(wage_result, dependent).is_valid
    invalid = validate_restriction_system(wage_result, inconsistent)
    assert not invalid.is_valid and invalid.rank_r < invalid.rank_augmented
    assert not validate_restriction_system(wage_result, zero).is_valid
