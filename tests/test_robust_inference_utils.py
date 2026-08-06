"""Konu 12 HC, tanı ve ortak test denetimleri."""

import numpy as np
import pytest
import statsmodels.api as sm

from core.data_registry import load_dataset
from core.joint_inference_utils import LinearRestriction
from core.robust_inference_utils import (
    breusch_pagan_auxiliary_details,
    fit_robust_inference,
    heteroskedastic_pattern_simulation,
    heteroskedasticity_tests,
    residual_plot_data,
    robust_joint_test,
    simulate_heteroskedastic_coverage,
    white_test_details,
)


@pytest.fixture(scope="module")
def hprice1_frame():
    return load_dataset("hprice1").assign(
        lotsize1000=lambda data: data.lotsize / 1000,
        sqrft100=lambda data: data.sqrft / 100,
    )


@pytest.fixture(scope="module")
def result(hprice1_frame):
    return fit_robust_inference(
        hprice1_frame,
        "price",
        ("lotsize1000", "sqrft100", "bdrms"),
        covariance_type="HC1",
    )


def test_hc_keeps_ols_coefficients_and_matches_statsmodels_use_t(result) -> None:
    design = sm.add_constant(result.ols.design_data, has_constant="add")
    expected = sm.OLS(result.ols.observed_values, design).fit().get_robustcov_results(
        cov_type="HC1",
        use_t=True,
    )

    assert np.allclose(result.coefficients.to_numpy(), result.ols.coefficients.to_numpy())
    assert np.allclose(result.coefficients.to_numpy(), expected.params)
    assert np.allclose(result.standard_errors.to_numpy(), expected.bse)
    assert np.allclose(result.p_values_two_sided_zero.to_numpy(), expected.pvalues)
    assert np.allclose(result.confidence_intervals_95.to_numpy(), expected.conf_int())

    assert result.standard_errors["lotsize1000"] == pytest.approx(1.2514, abs=2e-4)
    assert result.p_values_two_sided_zero["lotsize1000"] == pytest.approx(.1022, abs=2e-4)


def test_known_heteroskedasticity_and_joint_diagnostics(result) -> None:
    bp, white = heteroskedasticity_tests(result.ols)
    assert (bp.lm_statistic, bp.p_value, white.lm_statistic) == pytest.approx(
        (14.092, .0028, 33.732),
        abs=.01,
    )
    joint = robust_joint_test(
        result,
        (
            LinearRestriction({"lotsize1000": 1}, 0, "l"),
            LinearRestriction({"bdrms": 1}, 0, "b"),
        ),
    )
    assert (joint.q, joint.df_denom, joint.f_statistic, joint.p_value) == pytest.approx(
        (2, 84, 2.365, .1002),
        abs=.002,
    )


def test_simulation_is_deterministic_and_plausible() -> None:
    first = simulate_heteroskedastic_coverage(repetitions=500, seed=31)
    second = simulate_heteroskedastic_coverage(repetitions=500, seed=31)
    assert first == second
    assert abs(first.slope_mean - 2) < .15
    assert all(
        0 <= value <= 1
        for value in (
            first.nonrobust_coverage,
            first.hc1_coverage,
            first.hc3_coverage,
        )
    )


def test_bp_white_manual_lm_and_variance_patterns(result) -> None:
    bp = breusch_pagan_auxiliary_details(result.ols)
    white = white_test_details(result.ols)
    assert bp.lm_manual == pytest.approx(bp.lm_statsmodels, abs=1e-8)
    assert bp.df == 3
    assert white.lm_manual == pytest.approx(white.lm_statsmodels, abs=1e-8)
    assert white.matrix_rank == white.column_count

    constant = heteroskedastic_pattern_simulation(pattern="sabit")
    increasing = heteroskedastic_pattern_simulation(pattern="artan")
    assert constant.koşullu_ortalama.equals(increasing.koşullu_ortalama)
    assert increasing.hata_sd.iloc[-1] > increasing.hata_sd.iloc[0]


def test_scale_location_uses_finite_studentized_residuals(result) -> None:
    plot_data = residual_plot_data(result.ols)
    assert {"studentize_artık", "scale_location"}.issubset(plot_data.columns)
    assert np.isfinite(
        plot_data[["studentize_artık", "scale_location"]].to_numpy()
    ).all()
    assert (plot_data["scale_location"] >= 0).all()
