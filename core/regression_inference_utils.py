"""EKK çıkarımının ortak çekirdeği: complete-case örneklem, geleneksel standart hatalar ve p-değeri biçimi.

Yalnız homoskedastisiteye dayanan geleneksel (``nonrobust``) standart hataları destekler; dayanıklı standart hatalar
Konu 12'nin kapsamıdır. Konu 9–12'nin sayfaları ve yardımcı modülleri kullanır; Konu 7–8'in çıkarımı
``core.labs`` tanımlarından üretilir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _finite(value: object, label: str) -> float:
    """Sonlu bir sayıyı doğrular."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} sayısal olmalıdır.") from error
    if not np.isfinite(number):
        raise ValueError(f"{label} sonlu bir sayı olmalıdır.")
    return number


@dataclass(frozen=True)
class OLSInferenceResult:
    """Geleneksel EKK çıkarımı için gerekli model çıktıları."""

    dependent: str
    explanatory: tuple[str, ...]
    coefficients: pd.Series
    standard_errors: pd.Series
    t_values_zero: pd.Series
    p_values_two_sided_zero: pd.Series
    confidence_intervals_95: pd.DataFrame
    fitted_values: pd.Series
    residuals: pd.Series
    observed_values: pd.Series
    design_data: pd.DataFrame
    r_squared: float
    adjusted_r_squared: float
    nobs: int
    df_resid: int
    n_explanatory: int
    covariance_type: str
    residual_standard_deviation: float
    covariance_matrix: pd.DataFrame
    ssr: float

    def __post_init__(self) -> None:
        """Değiştirilebilir pandas nesnelerini sonuçtan yalıtır."""
        series_fields = (
            "coefficients", "standard_errors", "t_values_zero", "p_values_two_sided_zero",
            "fitted_values", "residuals", "observed_values",
        )
        for field in series_fields:
            series = getattr(self, field).copy(deep=True)
            if not np.isfinite(series.to_numpy(dtype=float)).all():
                raise ValueError(f"{field} sonlu değerler içermelidir.")
            object.__setattr__(self, field, series)
        intervals = self.confidence_intervals_95.copy(deep=True)
        if intervals.shape[1] != 2 or not np.isfinite(intervals.to_numpy(dtype=float)).all():
            raise ValueError("Güven aralıkları iki sonlu sütundan oluşmalıdır.")
        object.__setattr__(self, "confidence_intervals_95", intervals)
        object.__setattr__(self, "design_data", self.design_data.copy(deep=True))
        covariance = self.covariance_matrix.copy(deep=True)
        if covariance.shape != (len(self.coefficients), len(self.coefficients)) or not np.isfinite(covariance.to_numpy(dtype=float)).all():
            raise ValueError("Kovaryans matrisi katsayılarla uyumlu sonlu bir kare matris olmalıdır.")
        if tuple(covariance.index) != tuple(self.coefficients.index) or tuple(covariance.columns) != tuple(self.coefficients.index):
            raise ValueError("Kovaryans matrisi katsayı adlarıyla uyumlu olmalıdır.")
        object.__setattr__(self, "covariance_matrix", covariance)
        if not np.isfinite(float(self.ssr)) or float(self.ssr) < 0:
            raise ValueError("Artık kareleri toplamı sonlu ve negatif olmayan bir sayı olmalıdır.")
        if self.covariance_type != "nonrobust":
            raise ValueError("Bu konuda yalnızca nonrobust kovaryans türü desteklenir.")
        if self.df_resid <= 0 or self.nobs <= self.n_explanatory + 1:
            raise ValueError("Artık serbestlik derecesi pozitif olmalıdır.")


def _prepare_data(frame: pd.DataFrame, dependent: str, explanatory: tuple[str, ...]) -> pd.DataFrame:
    """Ortak complete-case örneklemini oluşturur ve tasarımın rank'ını denetler."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Girdi bir pandas DataFrame olmalıdır.")
    if len(set(explanatory)) != len(explanatory):
        raise ValueError("Açıklayıcı değişken adları benzersiz olmalıdır.")
    if dependent in explanatory:
        raise ValueError("Bağımlı değişken açıklayıcılar arasında bulunamaz.")
    columns = (dependent, *explanatory)
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"Veri setinde bulunamayan değişken: {', '.join(missing)}")
    prepared = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    prepared = prepared.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(prepared) <= len(explanatory) + 1:
        raise ValueError("Pozitif artık serbestlik derecesi için yeterli geçerli gözlem yoktur.")
    if prepared[dependent].nunique() < 2:
        raise ValueError("Bağımlı değişkende örneklem değişimi olmadığı için çıkarım yapılamaz.")
    design = sm.add_constant(prepared.loc[:, list(explanatory)], has_constant="add")
    if np.linalg.matrix_rank(design.to_numpy(dtype=float)) < design.shape[1]:
        raise ValueError("Tam doğrusal bağlantı nedeniyle model katsayıları ayrı ayrı tahmin edilemiyor.")
    return prepared


def fit_ols_inference(
    frame: pd.DataFrame,
    dependent: str,
    explanatory: tuple[str, ...],
    *,
    covariance_type: str = "nonrobust",
) -> OLSInferenceResult:
    """Sabit terimli EKK modelini geleneksel çıkarım çıktılarıyla tahmin eder."""
    if covariance_type != "nonrobust":
        raise ValueError("Bu konuda yalnızca covariance_type='nonrobust' kabul edilir.")
    prepared = _prepare_data(frame, dependent, explanatory)
    design = sm.add_constant(prepared.loc[:, list(explanatory)], has_constant="add")
    fitted = sm.OLS(prepared[dependent], design).fit()
    df_resid = int(fitted.df_resid)
    if df_resid <= 0:
        raise ValueError("Artık serbestlik derecesi pozitif olmalıdır.")
    values = (fitted.params, fitted.bse, fitted.tvalues, fitted.pvalues, fitted.conf_int(), fitted.fittedvalues, fitted.resid)
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in values):
        raise ValueError("Modelin sayısal çıktıları sonlu olmalıdır.")
    residual_sd = float(np.sqrt(float(fitted.ssr) / df_resid))
    if not np.isfinite(residual_sd):
        raise ValueError("Artık standart sapması sonlu olmalıdır.")
    intervals = fitted.conf_int().copy()
    intervals.columns = ["lower", "upper"]
    return OLSInferenceResult(
        dependent=dependent, explanatory=tuple(explanatory), coefficients=fitted.params.copy(),
        standard_errors=fitted.bse.copy(), t_values_zero=fitted.tvalues.copy(),
        p_values_two_sided_zero=fitted.pvalues.copy(), confidence_intervals_95=intervals,
        fitted_values=pd.Series(fitted.fittedvalues, index=prepared.index, name="tahmin"),
        residuals=pd.Series(fitted.resid, index=prepared.index, name="artık"),
        observed_values=prepared[dependent].copy(), design_data=prepared.loc[:, list(explanatory)].copy(),
        r_squared=float(fitted.rsquared), adjusted_r_squared=float(fitted.rsquared_adj),
        nobs=int(fitted.nobs), df_resid=df_resid, n_explanatory=len(explanatory),
        covariance_type="nonrobust", residual_standard_deviation=residual_sd,
        covariance_matrix=fitted.cov_params().copy(), ssr=float(fitted.ssr),
    )


def format_p_value(p_value: float) -> str:
    """p-değerini sıfır ya da bilimsel gösterime düşmeden biçimlendirir."""
    value = _finite(p_value, "p-değeri")
    if not 0.0 <= value <= 1.0:
        raise ValueError("p-değeri 0 ile 1 arasında olmalıdır.")
    return "< 0.001" if value < 0.001 else f"{value:.3f}"
