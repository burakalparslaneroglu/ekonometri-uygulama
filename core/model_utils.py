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
    if not np.isfinite(x_value):
        raise ValueError("X değeri sonlu bir sayı olmalıdır.")
    return result.intercept + result.slope * float(x_value)


def observation_result(result: SimpleOLSResult, position: int) -> dict[str, float | int]:
    """Sıfırdan başlayan sıra numarasıyla seçilen gözlemin tahmin ve artığını döndürür."""
    if position < 0 or position >= result.nobs:
        raise IndexError(f"Gözlem sırası 0 ile {result.nobs - 1} arasında olmalıdır.")
    index = result.observed_values.index[position]
    return {
        "index": int(index),
        "x": float(result.explanatory_values.iloc[position]),
        "observed": float(result.observed_values.iloc[position]),
        "predicted": float(result.fitted_values.iloc[position]),
        "residual": float(result.residuals.iloc[position]),
    }
