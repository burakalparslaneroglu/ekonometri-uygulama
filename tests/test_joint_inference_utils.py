"""Genel doğrusal kısıt F testi çekirdeğinin (Konu 9–12 sayfaları) sayısal denetimleri."""

import pytest
import statsmodels.formula.api as smf

from core.data_registry import load_dataset
from core.joint_inference_utils import LinearRestriction, joint_f_test, nested_exclusion_f_test, validate_restriction_system
from core.regression_inference_utils import fit_ols_inference


@pytest.fixture(scope="module")
def wage_result():
    return fit_ols_inference(load_dataset("wage1"), "wage", ("educ", "exper", "tenure"))


def test_wage1_and_hprice1_nested_tests_match_documented_values() -> None:
    """SSR, R² ve matris F hesapları notlardaki değerleri verir (§8.6: 53,31; §8.9: 6,61)."""
    wage = nested_exclusion_f_test(load_dataset("wage1"), "wage", ("educ", "exper", "tenure"), ("educ",))
    home = nested_exclusion_f_test(load_dataset("hprice1"), "price", ("lotsize", "sqrft", "bdrms"), ("sqrft",))
    assert (wage.q, wage.df_denom, wage.f_from_ssr, wage.p_value) == pytest.approx((2, 522, 53.3099, 8.56e-22), rel=2e-3, abs=1e-12)
    assert wage.formulas_match and wage.ssr_unrestricted <= wage.ssr_restricted
    assert (home.q, home.df_denom, home.f_from_ssr, home.p_value) == pytest.approx((2, 84, 6.6102, 0.002157), rel=2e-3)


def test_general_restrictions_match_statsmodels_and_single_t(wage_result) -> None:
    """Eşitlik ve toplam kısıtları statsmodels ``f_test`` ile aynıdır; tek kısıtta F = t²."""
    reference = smf.ols("wage ~ educ + exper + tenure", data=load_dataset("wage1")).fit()
    equal = joint_f_test(wage_result, (LinearRestriction({"exper": 1, "tenure": -1}, 0, "exper = tenure"),))
    total = joint_f_test(wage_result, (LinearRestriction({"exper": 1, "tenure": 1}, 1, "exper + tenure = 1"),))
    overall = joint_f_test(wage_result, tuple(LinearRestriction({name: 1}, 0, f"{name} = 0") for name in ("educ", "exper", "tenure")))
    assert equal.f_statistic == pytest.approx(float(reference.f_test("exper = tenure").fvalue), rel=1e-10)
    assert total.f_statistic == pytest.approx(float(reference.f_test("exper + tenure = 1").fvalue), rel=1e-10)
    assert overall.f_statistic == pytest.approx(reference.fvalue, rel=1e-10) and overall.df_num == 3
    education = joint_f_test(wage_result, (LinearRestriction({"educ": 1}, 0, "educ = 0"),))
    assert education.f_statistic == pytest.approx(wage_result.t_values_zero["educ"] ** 2, rel=1e-10)
    assert education.p_value == pytest.approx(wage_result.p_values_two_sided_zero["educ"], rel=1e-8)


def test_restriction_validation_rejects_invalid_systems(wage_result) -> None:
    """Rütbe, tutarlılık ve bilgi taşımayan satır denetimleri açıkça çalışır."""
    exclusion = (LinearRestriction({"exper": 1}, 0, "exper=0"), LinearRestriction({"tenure": 1}, 0, "tenure=0"))
    dependent = (LinearRestriction({"exper": 1}, 0, "a"), LinearRestriction({"exper": 2}, 0, "b"))
    inconsistent = (LinearRestriction({"exper": 1}, 0, "a"), LinearRestriction({"exper": 1}, 1, "b"))
    zero = (LinearRestriction({"exper": 0}, 0, "zero"),)
    assert validate_restriction_system(wage_result, exclusion).q == 2
    assert joint_f_test(wage_result, exclusion).f_statistic == pytest.approx(53.3099, abs=0.001)
    assert not validate_restriction_system(wage_result, dependent).is_valid
    invalid = validate_restriction_system(wage_result, inconsistent)
    assert not invalid.is_valid and invalid.rank_r < invalid.rank_augmented
    assert not validate_restriction_system(wage_result, zero).is_valid
    with pytest.raises(ValueError):
        joint_f_test(wage_result, dependent)
    with pytest.raises(ValueError):
        LinearRestriction({}, 0, "boş")
