"""Konu 10 kategorik hesaplarının temel sayısal denetimleri."""
import pytest

from core.categorical_regression_utils import (add_categorical_columns, binary_group_summary, compare_raw_and_controlled_dummy,
    build_reference_dummies, diagnose_dummy_trap, dummy_design_preview, dummy_log_ci_to_exact_percent, dummy_log_exact_percent,numeric_vs_dummy_coding_comparison)
from core.data_registry import load_dataset

@pytest.fixture(scope="module")
def wage():
    return add_categorical_columns(load_dataset("wage1"), "wage1")

def test_wage1_binary_gap_and_simple_model(wage) -> None:
    summary = binary_group_summary(wage, "wage", "female")
    result = compare_raw_and_controlled_dummy(wage)
    assert (summary.reference_n, summary.comparison_n, summary.reference_mean, summary.comparison_mean, summary.raw_difference) == pytest.approx((274, 252, 7.0995, 4.5877, -2.5118), abs=2e-4)
    assert result.raw_model.coefficients["const"] == pytest.approx(summary.reference_mean)
    assert result.raw_model.coefficients["female"] == pytest.approx(summary.raw_difference)
    assert result.raw_model.standard_errors["female"] == pytest.approx(.3034, abs=2e-4)
    assert result.raw_model.r_squared == pytest.approx(.116, abs=.002)

def test_log_conversion_and_dummy_trap(wage) -> None:
    result = compare_raw_and_controlled_dummy(wage)
    assert result.controlled_log_difference == pytest.approx(-.3011, abs=.001)
    assert result.exact_log_percent == pytest.approx(-26.00, abs=.05)
    assert dummy_log_exact_percent(0) == 0 and dummy_log_ci_to_exact_percent(-.1, .1)[0] < 0
    all_dummies = diagnose_dummy_trap(wage["female"], includes_intercept=True)
    m_minus_1 = diagnose_dummy_trap(wage["female"], includes_intercept=True, include_all_dummies=False)
    no_intercept = diagnose_dummy_trap(wage["female"], includes_intercept=False)
    assert not all_dummies.full_rank and m_minus_1.full_rank and no_intercept.full_rank

def test_numeric_code_imposes_equal_spacing_and_dummy_reproduces_means() -> None:
    import pandas as pd
    categories=pd.Series(["A"]*3+["B"]*3+["C"]*3); outcome=pd.Series([1,1,1,2,2,2,6,6,6],dtype=float)
    comparison=numeric_vs_dummy_coding_comparison(categories,outcome)
    assert comparison.equal_spacing_imposed and comparison.dummy_ssr < comparison.numeric_ssr
    preview,diagnostic=dummy_design_preview(categories,includes_intercept=True,include_all_dummies=True)
    assert preview.shape[1]==4 and not diagnostic.full_rank


def test_missing_category_is_not_silently_treated_as_reference() -> None:
    import pandas as pd
    categories = pd.Series(["A", None, "B"], index=[10, 11, 12])
    coding = build_reference_dummies(categories, reference_category="A", prefix="group", category_order=("A", "B"))
    assert coding.dummies.loc[10, "group_B"] == 0
    assert pd.isna(coding.dummies.loc[11, "group_B"])
    assert coding.dummies.loc[12, "group_B"] == 1
