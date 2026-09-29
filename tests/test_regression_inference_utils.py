"""Ortak EKK çıkarım çekirdeğinin (Konu 9–12 sayfaları) sayısal doğrulamaları."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from core.data_registry import load_dataset
from core.regression_inference_utils import fit_ols_inference, format_p_value, significance_stars


@pytest.fixture(scope="module")
def wage_result():
    return fit_ols_inference(load_dataset("wage1"), "wage", ("educ", "exper", "tenure"))


def test_wage1_matches_documented_conventional_inference(wage_result) -> None:
    """WAGE1 sonuçları ders notundaki geleneksel EKK değerleriyle uyumludur (Tablo 7.3)."""
    assert (wage_result.nobs, wage_result.df_resid, wage_result.covariance_type) == (526, 522, "nonrobust")
    assert wage_result.coefficients["educ"] == pytest.approx(0.5990, abs=0.0002)
    assert wage_result.standard_errors["educ"] == pytest.approx(0.0513, abs=0.0002)
    assert wage_result.t_values_zero["educ"] == pytest.approx(11.679, abs=0.01)
    assert wage_result.p_values_two_sided_zero["exper"] == pytest.approx(0.064, abs=0.002)
    assert tuple(wage_result.confidence_intervals_95.loc["educ"]) == pytest.approx((0.498, 0.700), abs=0.002)


def test_matches_statsmodels_formula_fit(wage_result) -> None:
    """Katsayı, standart hata, SSR, R² ve artık standart sapması statsmodels ile birebir aynıdır."""
    reference = smf.ols("wage ~ educ + exper + tenure", data=load_dataset("wage1")).fit()
    names = ("educ", "exper", "tenure")
    assert np.allclose(wage_result.coefficients[list(names)], reference.params[list(names)], rtol=0, atol=1e-12)
    assert np.allclose(wage_result.standard_errors[list(names)], reference.bse[list(names)], rtol=0, atol=1e-12)
    assert wage_result.ssr == pytest.approx(reference.ssr, rel=1e-12)
    assert wage_result.r_squared == pytest.approx(reference.rsquared, rel=1e-12)
    assert wage_result.residual_standard_deviation == pytest.approx(np.sqrt(reference.scale), rel=1e-12)


def test_hprice1_uses_raw_scale() -> None:
    """HPRICE1 ham ölçekli katsayıları doğrulanır."""
    result = fit_ols_inference(load_dataset("hprice1"), "price", ("lotsize", "sqrft", "bdrms"))
    assert (result.nobs, result.df_resid) == (88, 84)
    assert result.coefficients["sqrft"] == pytest.approx(0.1228, abs=0.0002)
    assert result.standard_errors["bdrms"] == pytest.approx(9.0101, abs=0.001)


def test_validation_and_presentation_rules() -> None:
    """Geçersiz girdiler, katı yıldız eşikleri ve p-değeri biçimi denetlenir."""
    with pytest.raises(ValueError):
        fit_ols_inference(pd.DataFrame({"y": [1, 2, 3, 4], "x": [1, 1, 1, 1]}), "y", ("x",))
    with pytest.raises(ValueError):
        fit_ols_inference(load_dataset("wage1"), "wage", ("educ",), covariance_type="HC1")
    with pytest.raises(ValueError):
        fit_ols_inference(load_dataset("wage1"), "wage", ("educ", "educ"))
    assert format_p_value(0.0001) == "< 0.001" and format_p_value(0.0641) == "0.064"
    assert significance_stars(0.009) == "***" and significance_stars(0.03) == "**"
    assert significance_stars(0.07) == "*" and significance_stars(0.10) == ""
    with pytest.raises(ValueError):
        format_p_value(1.2)
