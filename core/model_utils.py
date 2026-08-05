"""Basit EKK hesapları için Streamlit'ten bağımsız yardımcılar."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class SimpleOLSResult:
    """Basit EKK tahmininin öğretim için gerekli çıktıları."""

    dependent: str
    explanatory: str
    intercept: float
    slope: float
    fitted_values: pd.Series
    residuals: pd.Series
    observed_values: pd.Series
    explanatory_values: pd.Series
    r_squared: float
    nobs: int

    @property
    def equation(self) -> str:
        """Tahmin denklemini açık bir metin olarak döndürür."""
        sign = "+" if self.slope >= 0 else "−"
        return f"ŷ = {self.intercept:.4f} {sign} {abs(self.slope):.4f} × {self.explanatory}"


def prepare_model_data(frame: pd.DataFrame, dependent: str, explanatory: str) -> pd.DataFrame:
    """İki değişkenli modeli eksik ve geçersiz gözlemlerden arındırır."""
    if dependent == explanatory:
        raise ValueError("Bağımlı ve açıklayıcı değişkenler farklı olmalıdır.")
    missing = [column for column in (dependent, explanatory) if column not in frame.columns]
    if missing:
        raise ValueError(f"Veri setinde bulunamayan değişken: {', '.join(missing)}")
    prepared = frame.loc[:, [dependent, explanatory]].apply(pd.to_numeric, errors="coerce").dropna()
    prepared = prepared[np.isfinite(prepared[dependent]) & np.isfinite(prepared[explanatory])].copy()
    if len(prepared) < 3:
        raise ValueError("Basit doğrusal regresyon için en az üç geçerli gözlem gerekir.")
    if prepared[explanatory].nunique() < 2:
        raise ValueError("Açıklayıcı değişken en az iki farklı değer içermelidir.")
    return prepared


def fit_simple_ols(frame: pd.DataFrame, dependent: str, explanatory: str) -> SimpleOLSResult:
    """Sabit terimli basit EKK modelini statsmodels ile tahmin eder."""
    prepared = prepare_model_data(frame, dependent, explanatory)
    design = sm.add_constant(prepared[explanatory], has_constant="add")
    fitted_model = sm.OLS(prepared[dependent], design).fit()
    fitted_values = pd.Series(fitted_model.fittedvalues, index=prepared.index, name="tahmin")
    residuals = pd.Series(fitted_model.resid, index=prepared.index, name="artık")
    return SimpleOLSResult(
        dependent=dependent,
        explanatory=explanatory,
        intercept=float(fitted_model.params["const"]),
        slope=float(fitted_model.params[explanatory]),
        fitted_values=fitted_values,
        residuals=residuals,
        observed_values=prepared[dependent].rename(dependent),
        explanatory_values=prepared[explanatory].rename(explanatory),
        r_squared=float(fitted_model.rsquared),
        nobs=int(fitted_model.nobs),
    )


def descriptive_statistics(frame: pd.DataFrame, variables: tuple[str, ...]) -> pd.DataFrame:
    """Seçili değişkenler için temel tanımlayıcı istatistikleri hesaplar."""
    missing = [column for column in variables if column not in frame.columns]
    if missing:
        raise ValueError(f"Veri setinde bulunamayan değişken: {', '.join(missing)}")
    summary = frame.loc[:, list(variables)].apply(pd.to_numeric, errors="coerce").describe().T
    return summary.loc[:, ["count", "mean", "std", "min", "max"]].rename(
        columns={"count": "Gözlem", "mean": "Ortalama", "std": "Std. sapma", "min": "En küçük", "max": "En büyük"}
    )


def predict_value(result: SimpleOLSResult, x_value: float) -> float:
    """Geçerli bir X değeri için tahmin edilen Y değerini hesaplar."""
    intercept, slope = _model_coefficients(result)
    return fitted_value(intercept, slope, x_value)


def _model_coefficients(result: SimpleOLSResult) -> tuple[float, float]:
    """Tahmin için gerekli model katsayılarını doğrular."""
    try:
        return result.intercept, result.slope
    except AttributeError as error:
        raise ValueError("Model tahmin için sabit terim ve eğim katsayısı içermelidir.") from error


def fitted_value(intercept: float, slope: float, x_value: float) -> float:
    """Sabit, eğim ve X değeriyle tahmin edilen değeri hesaplar."""
    values = {"Sabit terim": intercept, "Eğim katsayısı": slope, "X değeri": x_value}
    converted: dict[str, float] = {}
    for label, value in values.items():
        try:
            converted[label] = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} sayısal olmalıdır.") from error
        if not np.isfinite(converted[label]):
            raise ValueError(f"{label} sonlu bir sayı olmalıdır.")
    return converted["Sabit terim"] + converted["Eğim katsayısı"] * converted["X değeri"]


def residual_value(observed_value: float, predicted_value: float) -> float:
    """Gerçekleşen ve tahmin edilen değerlerden artığı hesaplar."""
    values = {"Y değeri": observed_value, "Tahmin edilen Y değeri": predicted_value}
    converted: dict[str, float] = {}
    for label, value in values.items():
        try:
            converted[label] = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} sayısal olmalıdır.") from error
        if not np.isfinite(converted[label]):
            raise ValueError(f"{label} sonlu bir sayı olmalıdır.")
    return converted["Y değeri"] - converted["Tahmin edilen Y değeri"]


def observation_result(result: SimpleOLSResult, position: int) -> dict[str, float | int]:
    """Sıfırdan başlayan sıra numarasıyla seçilen gözlemin tahmin ve artığını döndürür."""
    if not isinstance(position, (int, np.integer)):
        raise ValueError("Gözlem sırası tam sayı olmalıdır.")
    if position < 0 or position >= result.nobs:
        raise IndexError(f"Gözlem sırası 0 ile {result.nobs - 1} arasında olmalıdır.")
    x_value = result.explanatory_values.iloc[position]
    observed_value = result.observed_values.iloc[position]
    if pd.isna(x_value) or pd.isna(observed_value):
        raise ValueError("Seçili gözlemde X ve Y değerleri eksik olamaz.")
    intercept, slope = _model_coefficients(result)
    predicted_value = fitted_value(intercept, slope, x_value)
    residual = residual_value(observed_value, predicted_value)
    index = result.observed_values.index[position]
    return {
        "index": int(index),
        "x": float(x_value),
        "observed": float(observed_value),
        "predicted": predicted_value,
        "residual": residual,
    }
