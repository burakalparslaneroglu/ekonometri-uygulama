"""Konu 12 HC, tanı ve ortak test denetimleri."""
import numpy as np
import pytest
from core.data_registry import load_dataset
from core.joint_inference_utils import LinearRestriction
from core.robust_inference_utils import breusch_pagan_auxiliary_details,fit_robust_inference,heteroskedastic_pattern_simulation,heteroskedasticity_tests,robust_joint_test,simulate_heteroskedastic_coverage,white_test_details

@pytest.fixture(scope="module")
def result():
    h=load_dataset("hprice1").assign(lotsize1000=lambda d:d.lotsize/1000,sqrft100=lambda d:d.sqrft/100)
    return fit_robust_inference(h,"price",("lotsize1000","sqrft100","bdrms"),covariance_type="HC1")
def test_hc_keeps_ols_coefficients_and_matches_known_diagnostics(result) -> None:
    assert np.allclose(result.ols.coefficients,result.ols.coefficients)
    bp,white=heteroskedasticity_tests(result.ols)
    assert (bp.lm_statistic,bp.p_value,white.lm_statistic)==pytest.approx((14.092,.0028,33.732),abs=.01)
    joint=robust_joint_test(result,(LinearRestriction({"lotsize1000":1},0,"l"),LinearRestriction({"bdrms":1},0,"b")))
    assert (joint.q,joint.df_denom,joint.f_statistic,joint.p_value)==pytest.approx((2,84,2.365,.1002),abs=.002)
def test_simulation_is_deterministic_and_plausible() -> None:
    first=simulate_heteroskedastic_coverage(repetitions=500,seed=31); second=simulate_heteroskedastic_coverage(repetitions=500,seed=31)
    assert first==second and abs(first.slope_mean-2)<.15 and all(0<=x<=1 for x in (first.nonrobust_coverage,first.hc1_coverage,first.hc3_coverage))

def test_bp_white_manual_lm_and_variance_patterns(result) -> None:
    bp=breusch_pagan_auxiliary_details(result.ols); white=white_test_details(result.ols)
    assert bp.lm_manual==pytest.approx(bp.lm_statsmodels,abs=1e-8) and bp.df==3
    assert white.lm_manual==pytest.approx(white.lm_statsmodels,abs=1e-8) and white.matrix_rank==white.column_count
    constant=heteroskedastic_pattern_simulation(pattern="sabit"); increasing=heteroskedastic_pattern_simulation(pattern="artan")
    assert constant.koşullu_ortalama.equals(increasing.koşullu_ortalama)
    assert increasing.hata_sd.iloc[-1]>increasing.hata_sd.iloc[0]
