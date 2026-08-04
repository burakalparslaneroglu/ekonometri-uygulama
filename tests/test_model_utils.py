import pandas as pd
import pytest

from core.model_utils import descriptive_statistics, fit_simple_ols, observation_result, predict_value


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
