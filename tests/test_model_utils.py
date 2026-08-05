import pandas as pd
import pytest

from core.model_utils import (
    descriptive_statistics,
    fitted_value,
    fit_simple_ols,
    observation_result,
    predict_value,
    residual_value,
)


def test_fit_simple_ols_exact_line() -> None:
    frame = pd.DataFrame({"y": [1.0, 3.0, 5.0, 7.0], "x": [0.0, 1.0, 2.0, 3.0]})
    result = fit_simple_ols(frame, "y", "x")
    assert result.intercept == pytest.approx(1.0, abs=1e-12)
    assert result.slope == pytest.approx(2.0, abs=1e-12)
    assert result.r_squared == pytest.approx(1.0, abs=1e-12)
    assert predict_value(result, 2.5) == pytest.approx(6.0, abs=1e-12)
    item = observation_result(result, 2)
    assert item["predicted"] == pytest.approx(5.0, abs=1e-12)
    assert item["residual"] == pytest.approx(0.0, abs=1e-12)


def test_descriptive_statistics_labels() -> None:
    frame = pd.DataFrame({"y": [1.0, 3.0, 5.0], "x": [0.0, 1.0, 2.0]})
    summary = descriptive_statistics(frame, ("y", "x"))
    assert list(summary.columns) == ["Gözlem", "Ortalama", "Std. sapma", "En küçük", "En büyük"]
    assert summary.loc["y", "Ortalama"] == pytest.approx(3.0, abs=1e-12)


def test_fitted_value_and_residual_value_cover_signs_and_near_zero() -> None:
    assert fitted_value(1.0, 2.0, 3.0) == pytest.approx(7.0, abs=1e-12)
    assert residual_value(9.0, 7.0) == pytest.approx(2.0, abs=1e-12)
    assert residual_value(5.0, 7.0) == pytest.approx(-2.0, abs=1e-12)
    assert residual_value(7.0 + 1e-12, 7.0) == pytest.approx(1e-12, abs=1e-15)


@pytest.mark.parametrize("value", [None, "geçersiz", float("nan"), float("inf")])
def test_fitted_and_residual_values_reject_invalid_inputs(value: object) -> None:
    with pytest.raises(ValueError):
        fitted_value(1.0, 2.0, value)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        residual_value(value, 1.0)  # type: ignore[arg-type]


def test_observation_result_rejects_invalid_position() -> None:
    result = fit_simple_ols(pd.DataFrame({"y": [1.0, 3.0, 5.0], "x": [0.0, 1.0, 2.0]}), "y", "x")
    with pytest.raises(IndexError):
        observation_result(result, 3)
    with pytest.raises(ValueError):
        observation_result(result, "1")  # type: ignore[arg-type]


def test_predict_value_requires_model_coefficients() -> None:
    with pytest.raises(ValueError, match="sabit terim ve eğim"):
        predict_value(object(), 1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("dataset_key", "dependent", "explanatory", "position"),
    [("wage1", "wage", "educ", 0), ("hprice1", "price", "sqrft", 0)],
)
def test_wooldridge_observation_result_matches_fitted_equation(
    dataset_key: str, dependent: str, explanatory: str, position: int
) -> None:
    from core.data_registry import load_dataset

    result = fit_simple_ols(load_dataset(dataset_key), dependent, explanatory)
    item = observation_result(result, position)
    assert item["predicted"] == pytest.approx(
        result.intercept + result.slope * item["x"], abs=1e-12
    )
    assert item["residual"] == pytest.approx(item["observed"] - item["predicted"], abs=1e-12)
