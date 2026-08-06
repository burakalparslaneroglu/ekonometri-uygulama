"""Konu 12 HC, tanı ve ortak test denetimleri."""
import numpy as np
import pytest
from core.data_registry import load_dataset
from core.joint_inference_utils import LinearRestriction
from core.robust_inference_utils import breusch_pagan_auxiliary_details,fit_robust_inference,heteroskedastic_pattern_simulation,heteroskedasticity_tests,residual_plot_data,robust_joint_test,simulate_heteroskedastic_coverage,white_test_details

@pytest.fixture(scope="module")
def result():
    h=load_dataset("hprice1").assign(lotsize1000=lambda d:d.lotsize/1000,sqrft100=lambda d:d.sqrft/100)
    return fit_robust_inference(h,"price",("lotsize1000","sqrft100","bdrms"),covariance_type="HC1")
def test_hc_keeps_ols_coefficients_and_matches_known_diagnostics(result) -> None:
    assert result.p_values_two_sided_zero["lotsize1000"] == pytest.approx(.1022, abs=.002)
    assert np.isfinite(result.confidence_intervals_95.to_numpy()).all()
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


def test_scale_location_uses_finite_studentized_residuals(result) -> None:
    plot_data = residual_plot_data(result.ols)
    assert {"studentize_artık", "scale_location"}.issubset(plot_data.columns)
    assert np.isfinite(plot_data[["studentize_artık", "scale_location"]].to_numpy()).all()
    assert (plot_data["scale_location"] >= 0).all()
