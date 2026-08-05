import numpy as np
import pandas as pd
import pytest

from core.assumption_diagnostics_utils import (
    NEAR_COLLINEARITY_SCENARIOS,
    calculate_vif,
    coefficient_sensitivity,
    design_matrix_rank,
    exact_collinearity_demo,
    generate_near_collinearity_data,
    generate_synthetic_ovb_data,
    ovb_bias,
    simulate_collinearity_variability,
    simulate_repeated_ols,
    synthetic_ovb_decomposition,
    wage1_ovb_decomposition,
)
from core.data_registry import load_dataset
from core.model_utils import format_numerical_difference, format_student_number


def test_repeated_sampling_is_deterministic_and_has_expected_centers() -> None:
    first = simulate_repeated_ols(seed=17, repetitions=1000)
    assert np.array_equal(first.estimates, simulate_repeated_ols(seed=17, repetitions=1000).estimates)
    assert not np.array_equal(first.estimates, simulate_repeated_ols(seed=18, repetitions=1000).estimates)
    assert first.mean_estimate == pytest.approx(1.5, abs=0.02)
    assert first.target_center == 1.5
    biased = simulate_repeated_ols(seed=17, repetitions=1000, conditional_mean_loading=0.8)
    assert biased.mean_estimate == pytest.approx(2.3, abs=0.02)
    assert biased.bias == pytest.approx(0.8, abs=0.02)
    assert biased.target_center == 2.3
    assert biased.estimate_std > 0 and np.isfinite(biased.estimates).all()
    with pytest.raises(ValueError):
        simulate_repeated_ols(nobs=2)
    with pytest.raises(ValueError):
        simulate_repeated_ols(true_slope=np.nan)


@pytest.mark.parametrize(("beta2", "delta", "direction"), [(2, 0.3, "yukarı yönlü"), (2, -0.3, "aşağı yönlü"), (-2, 0.3, "aşağı yönlü"), (-2, -0.3, "yukarı yönlü"), (0, 0.3, "yanlılık yok / yaklaşık sıfır")])
def test_ovb_formula_and_direction(beta2: float, delta: float, direction: str) -> None:
    result = ovb_bias(1.2, beta2, delta)
    assert result.bias == pytest.approx(beta2 * delta)
    assert result.direction == direction
    example = ovb_bias(1.2, -2, 0.4)
    assert example.bias == pytest.approx(-0.8)
    assert example.expected_short_coefficient == pytest.approx(0.4)


def test_synthetic_and_wage_decompositions_use_common_samples() -> None:
    frame = generate_synthetic_ovb_data(seed=123)
    original = frame.copy(deep=True)
    synthetic = synthetic_ovb_decomposition(frame)
    assert frame.equals(original)
    assert synthetic.auxiliary_slope == pytest.approx(0.7, abs=0.06)
    assert synthetic.short_model.slope == pytest.approx(4.1, abs=0.15)
    assert synthetic.long_model.coefficients["x"] == pytest.approx(2, abs=0.08)
    assert synthetic.long_model.coefficients["z"] == pytest.approx(3, abs=0.08)
    assert synthetic.decomposition_error == pytest.approx(0, abs=1e-10)
    wage = wage1_ovb_decomposition(load_dataset("wage1"))
    assert wage.nobs == wage.short_model.nobs == wage.middle_model.nobs == wage.full_model.nobs == 526
    assert wage.short_model.slope == pytest.approx(0.5414, abs=1e-3)
    assert wage.middle_model.coefficients["educ"] == pytest.approx(0.6443, abs=1e-3)
    assert wage.full_model.coefficients["educ"] == pytest.approx(0.5990, abs=1e-3)
    assert wage.auxiliary_model.slope == pytest.approx(-1.4682, abs=1e-3)
    assert wage.decomposition_error == pytest.approx(0, abs=1e-10)


def test_rank_vif_collinearity_and_sensitivity() -> None:
    exact = exact_collinearity_demo("monthly_annual_income")
    assert not exact.rank_result.full_rank
    high_not_exact = exact_collinearity_demo("high_but_not_exact")
    assert high_not_exact.rank_result.full_rank
    near = generate_near_collinearity_data("very_high", seed=50)
    original = near.data.copy(deep=True)
    assert near.rank_result.full_rank
    vif = calculate_vif(near.data, ("x1", "x2"))
    assert (vif["vif"] > 10).all() and not vif["exact_collinearity"].any()
    assert near.data.equals(original)
    exact_vif = calculate_vif(exact.data, exact.explanatory)
    assert exact_vif["exact_collinearity"].all() and np.isinf(exact_vif["vif"]).all()
    sensitivity = coefficient_sensitivity(near.data)
    assert sensitivity.original_rank.full_rank and sensitivity.perturbed_rank.full_rank
    assert sensitivity.original_coefficients != sensitivity.perturbed_coefficients
    independent = pd.DataFrame({"x": np.arange(20, dtype=float)})
    assert calculate_vif(independent, ("x",)).loc[0, "vif"] == 1.0
    scaled = pd.DataFrame({"x1": np.arange(10, dtype=float), "x2": np.arange(10, dtype=float) * 1e8})
    assert not design_matrix_rank(scaled, ("x1", "x2")).full_rank


def test_collinearity_variability_increases_with_connection() -> None:
    medium = simulate_collinearity_variability("high", repetitions=400, seed=8)
    very_high = simulate_collinearity_variability("very_high", repetitions=400, seed=8)
    near_exact = simulate_collinearity_variability("near_exact", repetitions=400, seed=8)
    assert very_high.mean_vif > medium.mean_vif
    assert very_high.beta1_std > medium.beta1_std
    assert very_high.beta_sum_std < very_high.beta1_std
    assert very_high.beta1_mean == pytest.approx(2, abs=0.15)
    assert near_exact.mean_correlation < 1
    assert set(NEAR_COLLINEARITY_SCENARIOS) == {"low_medium", "high", "very_high", "near_exact"}


def test_student_number_formatting_keeps_calculations_and_hides_machine_precision() -> None:
    original = 1.2552e-5
    assert format_student_number(original) == "0.000013"
    assert "e-" not in format_student_number(original)
    assert format_numerical_difference(4.44089210e-16) == "sayısal tolerans içinde 0"
    assert format_student_number(-4.44089210e-16) == "0"
    assert original == 1.2552e-5
    assert format_student_number(0.5413592546651751, decimals=4) == "0.5414"


def test_default_sensitivity_makes_relative_stability_visible() -> None:
    source = generate_near_collinearity_data("near_exact").data
    original = source.copy(deep=True)
    result = coefficient_sensitivity(source)
    delta_one = result.perturbed_coefficients["x1"] - result.original_coefficients["x1"]
    delta_two = result.perturbed_coefficients["x2"] - result.original_coefficients["x2"]
    delta_sum = sum(result.perturbed_coefficients.values()) - sum(result.original_coefficients.values())
    assert source.equals(original)
    assert result.original_rank.full_rank and result.perturbed_rank.full_rank
    assert abs(delta_one) > abs(delta_sum) and abs(delta_two) > abs(delta_sum)
    assert max(abs(delta_one), abs(delta_two)) >= 3 * abs(delta_sum)
    assert abs(result.perturbed_r_squared - result.original_r_squared) < 0.01
