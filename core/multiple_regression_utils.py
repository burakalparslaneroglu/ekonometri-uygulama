"""Çoklu EKK hesapları için Streamlit'ten bağımsız yardımcılar."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import pandas as pd
import statsmodels.api as sm


def add_konu05_derived_columns(frame: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """HPRICE1'in ders notundaki açık ölçek dönüşümlerini ekler."""
    if dataset_key != "hprice1":
        return frame.copy()
    if not {"lotsize", "sqrft"}.issubset(frame.columns):
        raise ValueError("HPRICE1 için lotsize ve sqrft sütunları gereklidir.")
    transformed = frame.copy()
    transformed["lotsize1000"] = pd.to_numeric(transformed["lotsize"], errors="coerce") / 1000
    transformed["sqrft100"] = pd.to_numeric(transformed["sqrft"], errors="coerce") / 100
    return transformed


@dataclass(frozen=True)
class MultipleOLSResult:
    """Öğretim için gerekli çoklu EKK sonuçlarını taşır."""

    dependent: str
    explanatory: tuple[str, ...]
    coefficients: pd.Series
    fitted_values: pd.Series
    residuals: pd.Series
    observed_values: pd.Series
    design_data: pd.DataFrame
    r_squared: float
    adjusted_r_squared: float
    nobs: int


@dataclass(frozen=True)
class PartialRegressionData:
    """Kontrollerin doğrusal katkısı ayrıldıktan sonraki ilişkiyi taşır."""

    x_residuals: pd.Series
    y_residuals: pd.Series
    partial_slope: float
    full_model_slope: float
    slopes_match: bool


def prepare_multiple_model_data(frame: pd.DataFrame, dependent: str, explanatory: tuple[str, ...]) -> pd.DataFrame:
    """Model değişkenlerini ortak complete-case örneklemine hazırlar."""
    if len(explanatory) < 2:
        raise ValueError("Çoklu regresyon için en az iki açıklayıcı değişken gerekir.")
    if len(set(explanatory)) != len(explanatory):
        raise ValueError("Açıklayıcı değişken adları benzersiz olmalıdır.")
    if dependent in explanatory:
        raise ValueError("Bağımlı değişken açıklayıcı değişkenler arasında bulunamaz.")
    columns = (dependent, *explanatory)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Veri setinde bulunamayan değişken: {', '.join(missing)}")
    prepared = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    prepared = prepared.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(prepared) <= len(explanatory) + 1:
        raise ValueError("Sabit terim ve eğimlerin tahmini için yeterli geçerli gözlem yoktur.")
    constant = [column for column in explanatory if prepared[column].nunique() < 2]
    if constant:
        raise ValueError(f"Açıklayıcı değişken en az iki farklı değer içermelidir: {', '.join(constant)}")
    design = sm.add_constant(prepared.loc[:, list(explanatory)], has_constant="add")
    if np.linalg.matrix_rank(design.to_numpy(dtype=float)) < design.shape[1]:
        raise ValueError("Tam doğrusal bağlantı nedeniyle model katsayıları ayrı ayrı tahmin edilemiyor.")
    return prepared


def adjusted_r_squared_from_r_squared(r_squared: float, nobs: int, n_explanatory: int) -> float:
    """Düzeltilmiş R-kareyi açık formülle hesaplar."""
    if nobs - n_explanatory - 1 <= 0:
        raise ValueError("Düzeltilmiş R-kare için geçerli artık serbestlik derecesi yoktur.")
    value = 1.0 - (1.0 - float(r_squared)) * (nobs - 1) / (nobs - n_explanatory - 1)
    if not np.isfinite(value):
        raise ValueError("Düzeltilmiş R-kare sonlu değil.")
    return float(value)


def fit_multiple_ols(frame: pd.DataFrame, dependent: str, explanatory: tuple[str, ...]) -> MultipleOLSResult:
    """Sabit terimli çoklu EKK modelini tahmin eder."""
    prepared = prepare_multiple_model_data(frame, dependent, explanatory)
    observed = prepared[dependent].to_numpy(dtype=float)
    if np.isclose(np.square(observed - observed.mean()).sum(), 0.0, atol=1e-12):
        raise ValueError("Bağımlı değişkende örneklem değişimi olmadığı için R-kare hesaplanamaz.")
    design = sm.add_constant(prepared.loc[:, list(explanatory)], has_constant="add")
    fitted = sm.OLS(prepared[dependent], design).fit()
    arrays = (fitted.params.to_numpy(), fitted.fittedvalues.to_numpy(), fitted.resid.to_numpy())
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("Modelin sayısal çıktıları sonlu olmalıdır.")
    nobs = int(fitted.nobs)
    adjusted = adjusted_r_squared_from_r_squared(float(fitted.rsquared), nobs, len(explanatory))
    return MultipleOLSResult(dependent, explanatory, fitted.params.copy(), pd.Series(fitted.fittedvalues, index=prepared.index, name="tahmin"), pd.Series(fitted.resid, index=prepared.index, name="artık"), prepared[dependent].rename(dependent), prepared.loc[:, list(explanatory)].copy(), float(fitted.rsquared), adjusted, nobs)


def _profile(result: MultipleOLSResult, profile: Mapping[str, object]) -> dict[str, float]:
    """Profil alanlarını modelle uyumlu ve sonlu sayılara dönüştürür."""
    expected, supplied = set(result.explanatory), set(profile)
    if expected != supplied:
        pieces = []
        if expected - supplied: pieces.append(f"eksik alan: {', '.join(sorted(expected - supplied))}")
        if supplied - expected: pieces.append(f"fazladan alan: {', '.join(sorted(supplied - expected))}")
        raise ValueError("Profil modelle eşleşmiyor (" + "; ".join(pieces) + ").")
    converted: dict[str, float] = {}
    for key in result.explanatory:
        try: value = float(profile[key])
        except (TypeError, ValueError) as error: raise ValueError(f"{key} sayısal olmalıdır.") from error
        if not np.isfinite(value): raise ValueError(f"{key} sonlu bir sayı olmalıdır.")
        converted[key] = value
    return converted


def predict_multiple_value(result: MultipleOLSResult, profile: Mapping[str, object]) -> float:
    """Verilen profil için tahmin edilen bağımlı değişken değerini hesaplar."""
    values = _profile(result, profile)
    prediction = float(result.coefficients["const"] + sum(result.coefficients[key] * values[key] for key in result.explanatory))
    if not np.isfinite(prediction): raise ValueError("Tahmin edilen değer sonlu değil.")
    return prediction


def multiple_observation_result(result: MultipleOLSResult, position: int) -> dict[str, object]:
    """Seçili gözlemin değeri, tahmini, artığı ve profilini döndürür."""
    if not isinstance(position, (int, np.integer)) or not 0 <= position < result.nobs:
        raise IndexError(f"Gözlem sırası 0 ile {result.nobs - 1} arasında tam sayı olmalıdır.")
    profile = {key: float(result.design_data.iloc[position][key]) for key in result.explanatory}
    predicted = predict_multiple_value(result, profile)
    observed = float(result.observed_values.iloc[position])
    return {"index": int(result.observed_values.index[position]), "observed": observed, "predicted": predicted, "residual": observed - predicted, "explanatory": profile}


def profile_prediction_difference(result: MultipleOLSResult, profile_a: Mapping[str, object], profile_b: Mapping[str, object]) -> dict[str, object]:
    """İki profilin tahmin farkını katsayı katkılarına ayırır."""
    a, b = _profile(result, profile_a), _profile(result, profile_b)
    contributions = {key: float(result.coefficients[key] * (b[key] - a[key])) for key in result.explanatory}
    difference = predict_multiple_value(result, b) - predict_multiple_value(result, a)
    if not np.isclose(difference, sum(contributions.values()), rtol=1e-10, atol=1e-10):
        raise ValueError("Profil farkı katsayı katkılarıyla doğrulanamadı.")
    return {"prediction_a": predict_multiple_value(result, a), "prediction_b": predict_multiple_value(result, b), "difference": difference, "contributions": contributions}


def partial_regression_data(result: MultipleOLSResult, focal_explanatory: str) -> PartialRegressionData:
    """FWL sezgisi için kontrol artıklarını ve eğim eşitliğini hesaplar."""
    if focal_explanatory not in result.explanatory: raise ValueError("Temel açıklayıcı değişken modelde bulunmalıdır.")
    controls = tuple(item for item in result.explanatory if item != focal_explanatory)
    control_design = sm.add_constant(result.design_data.loc[:, list(controls)], has_constant="add")
    y_resid = sm.OLS(result.observed_values, control_design).fit().resid
    x_resid = sm.OLS(result.design_data[focal_explanatory], control_design).fit().resid
    partial = sm.OLS(y_resid, x_resid).fit().params.iloc[0]
    full = float(result.coefficients[focal_explanatory])
    return PartialRegressionData(pd.Series(x_resid, index=result.design_data.index), pd.Series(y_resid, index=result.design_data.index), float(partial), full, bool(np.isclose(partial, full, rtol=1e-10, atol=1e-10)))
