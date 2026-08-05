"""Tek katsayılı EKK çıkarımı için Streamlit'ten bağımsız araçlar.

Bu modül yalnızca homoskedastisiteye dayanan geleneksel (``nonrobust``)
standart hataları destekler. Dayanıklı standart hatalar Konu 12'nin kapsamıdır.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t


Alternative = Literal["two-sided", "greater", "less"]
_ALTERNATIVES: tuple[str, ...] = ("two-sided", "greater", "less")


def _finite(value: object, label: str) -> float:
    """Sonlu bir sayıyı doğrular."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} sayısal olmalıdır.") from error
    if not np.isfinite(number):
        raise ValueError(f"{label} sonlu bir sayı olmalıdır.")
    return number


def _positive_int(value: object, label: str, *, minimum: int = 1) -> int:
    """Tam sayı boyut ya da serbestlik derecesini doğrular."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} tam sayı olmalıdır.")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{label} en az {minimum} olmalıdır.")
    return number


def _validate_alpha(alpha: object) -> float:
    """Anlamlılık düzeyini açık aralıkta doğrular."""
    number = _finite(alpha, "Anlamlılık düzeyi")
    if not 0.0 < number < 1.0:
        raise ValueError("Anlamlılık düzeyi 0 ile 1 arasında olmalıdır.")
    return number


def _validate_alternative(alternative: object) -> Alternative:
    """Desteklenen alternatif hipotez yönünü doğrular."""
    if alternative not in _ALTERNATIVES:
        raise ValueError("Alternatif yalnızca 'two-sided', 'greater' veya 'less' olabilir.")
    return alternative  # type: ignore[return-value]


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


@dataclass(frozen=True)
class CoefficientInference:
    """Tek katsayı için test ve isteğe bağlı güven aralığı sonucu."""

    coefficient_name: str
    estimate: float
    standard_error: float
    null_value: float
    alternative: Alternative
    alpha: float
    df_resid: int
    t_statistic: float
    p_value: float
    critical_value: float
    reject_null: bool
    confidence_level: float | None
    confidence_lower: float | None
    confidence_upper: float | None


@dataclass(frozen=True)
class ScaledInference:
    """Anlamlı bir değişim miktarına ölçeklenmiş katsayı ve aralığı."""

    scaled_estimate: float
    scaled_lower: float
    scaled_upper: float
    change: float


@dataclass(frozen=True)
class CITestEquivalence:
    """İki taraflı test ile karşılık gelen güven aralığının karşılaştırması."""

    applicable: bool
    ci_contains_null: bool | None
    reject_from_ci: bool | None
    reject_from_p_value: bool
    consistent: bool | None


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


def critical_t_value(df_resid: int, alpha: float, alternative: str) -> float:
    """Seçilen test için pozitif Student-t kritik değerini döndürür."""
    df = _positive_int(df_resid, "Serbestlik derecesi")
    level = _validate_alpha(alpha)
    direction = _validate_alternative(alternative)
    probability = 1.0 - level / 2.0 if direction == "two-sided" else 1.0 - level
    value = float(student_t.ppf(probability, df))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Kritik t değeri hesaplanamadı.")
    return value


def coefficient_confidence_interval(
    result: OLSInferenceResult,
    coefficient_name: str,
    *,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Bir katsayı için iki taraflı Student-t güven aralığını kurar."""
    level = _finite(confidence_level, "Güven düzeyi")
    if not 0.0 < level < 1.0:
        raise ValueError("Güven düzeyi 0 ile 1 arasında olmalıdır.")
    if coefficient_name not in result.coefficients.index:
        raise ValueError(f"Modelde bulunmayan katsayı: {coefficient_name}")
    estimate, standard_error = float(result.coefficients[coefficient_name]), float(result.standard_errors[coefficient_name])
    if standard_error <= 0 or not np.isfinite(standard_error):
        raise ValueError("Katsayının standart hatası pozitif ve sonlu olmalıdır.")
    critical = float(student_t.ppf(1.0 - (1.0 - level) / 2.0, result.df_resid))
    margin = critical * standard_error
    lower, upper = estimate - margin, estimate + margin
    if not np.isfinite((lower, upper)).all() or lower > upper:
        raise ValueError("Güven aralığı hesaplanamadı.")
    return float(lower), float(upper)


def coefficient_test(
    result: OLSInferenceResult,
    coefficient_name: str,
    *,
    null_value: float = 0.0,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
) -> CoefficientInference:
    """Tek katsayı için Student-t testi ve iki taraflı güven aralığını hesaplar."""
    if coefficient_name not in result.coefficients.index:
        raise ValueError(f"Modelde bulunmayan katsayı: {coefficient_name}")
    null = _finite(null_value, "Null değeri")
    direction = _validate_alternative(alternative)
    level = _validate_alpha(alpha)
    estimate = float(result.coefficients[coefficient_name])
    standard_error = float(result.standard_errors[coefficient_name])
    if standard_error <= 0.0 or not np.isfinite(standard_error):
        raise ValueError("Katsayının standart hatası pozitif ve sonlu olmalıdır.")
    statistic = (estimate - null) / standard_error
    if direction == "two-sided":
        p_value = 2.0 * float(student_t.sf(abs(statistic), result.df_resid))
        confidence_level: float | None = 1.0 - level
        lower, upper = coefficient_confidence_interval(result, coefficient_name, confidence_level=confidence_level)
    elif direction == "greater":
        p_value, confidence_level, lower, upper = float(student_t.sf(statistic, result.df_resid)), None, None, None
    else:
        p_value, confidence_level, lower, upper = float(student_t.cdf(statistic, result.df_resid)), None, None, None
    if not 0.0 <= p_value <= 1.0 or not np.isfinite(p_value):
        raise ValueError("p-değeri hesaplanamadı.")
    return CoefficientInference(
        coefficient_name=coefficient_name, estimate=estimate, standard_error=standard_error,
        null_value=null, alternative=direction, alpha=level, df_resid=result.df_resid,
        t_statistic=float(statistic), p_value=float(p_value),
        critical_value=critical_t_value(result.df_resid, level, direction),
        reject_null=bool(p_value < level), confidence_level=confidence_level,
        confidence_lower=lower, confidence_upper=upper,
    )


def ci_contains_value(lower: float, upper: float, value: float, *, tolerance: float = 1e-10) -> bool:
    """Bir değerin kapalı güven aralığında olup olmadığını toleransla belirler."""
    low, high, tested, tol = _finite(lower, "Alt sınır"), _finite(upper, "Üst sınır"), _finite(value, "Kontrol değeri"), _finite(tolerance, "Tolerans")
    if tol < 0 or low > high:
        raise ValueError("Aralık sınırları ve tolerans geçerli olmalıdır.")
    return bool(low - tol <= tested <= high + tol)


def ci_test_equivalence(test: CoefficientInference, *, tolerance: float = 1e-10) -> CITestEquivalence:
    """İki taraflı testte p-kararı ile güven aralığı kararının uyumunu denetler."""
    if test.alternative != "two-sided" or test.confidence_lower is None or test.confidence_upper is None:
        return CITestEquivalence(False, None, None, test.reject_null, None)
    contains = ci_contains_value(test.confidence_lower, test.confidence_upper, test.null_value, tolerance=tolerance)
    reject_from_ci = not contains
    return CITestEquivalence(True, contains, reject_from_ci, test.reject_null, reject_from_ci == test.reject_null)


def format_p_value(p_value: float) -> str:
    """p-değerini sıfır ya da bilimsel gösterime düşmeden biçimlendirir."""
    value = _finite(p_value, "p-değeri")
    if not 0.0 <= value <= 1.0:
        raise ValueError("p-değeri 0 ile 1 arasında olmalıdır.")
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def significance_stars(
    p_value: float,
    *,
    thresholds: tuple[tuple[float, str], ...] = ((0.01, "***"), (0.05, "**"), (0.10, "*")),
) -> str:
    """Tablo notundaki katı eşiklere göre anlamlılık yıldızı üretir."""
    value = _finite(p_value, "p-değeri")
    if not 0.0 <= value <= 1.0:
        raise ValueError("p-değeri 0 ile 1 arasında olmalıdır.")
    if not thresholds:
        raise ValueError("En az bir yıldız eşiği gereklidir.")
    checked: list[tuple[float, str]] = []
    for threshold, label in thresholds:
        limit = _finite(threshold, "Yıldız eşiği")
        if not 0.0 < limit < 1.0 or not isinstance(label, str) or not label:
            raise ValueError("Yıldız eşikleri 0 ile 1 arasında, etiketleri boş olmayan metin olmalıdır.")
        checked.append((limit, label))
    if len({threshold for threshold, _ in checked}) != len(checked):
        raise ValueError("Yıldız eşikleri benzersiz olmalıdır.")
    for threshold, label in sorted(checked, key=lambda item: item[0]):
        if value < threshold:
            return label
    return ""


def scale_coefficient_inference(estimate: float, lower: float, upper: float, change: float) -> ScaledInference:
    """Katsayı ve aralığını verilen değişim miktarıyla çarpar."""
    beta, low, high, delta = (_finite(estimate, "Katsayı"), _finite(lower, "Alt sınır"), _finite(upper, "Üst sınır"), _finite(change, "Değişim"))
    if low > high:
        raise ValueError("Güven aralığının alt sınırı üst sınırını aşamaz.")
    bounds = (low * delta, high * delta)
    return ScaledInference(beta * delta, float(min(bounds)), float(max(bounds)), delta)


@dataclass(frozen=True)
class StandardErrorSimulationResult:
    """Tekrarlanan basit EKK benzetiminde eğim ve SH özetleri."""

    true_intercept: float
    true_slope: float
    nobs: int
    repetitions: int
    df_resid: int
    slope_estimates: np.ndarray
    reported_standard_errors: np.ndarray
    mean_slope: float
    empirical_slope_std: float
    mean_reported_standard_error: float
    seed: int

    def __post_init__(self) -> None:
        """Dizilerin geçerli ve dışarıdan değiştirilemez olmasını sağlar."""
        for name in ("slope_estimates", "reported_standard_errors"):
            values = np.asarray(getattr(self, name), dtype=float).copy()
            if values.ndim != 1 or len(values) != self.repetitions or not np.isfinite(values).all() or np.any(values <= 0 if name == "reported_standard_errors" else False):
                raise ValueError(f"{name} sonlu ve tekrar sayısıyla uyumlu olmalıdır.")
            values.setflags(write=False)
            object.__setattr__(self, name, values)


def simulate_standard_errors(
    *, nobs: int = 50, repetitions: int = 5000, seed: int = 202507,
    true_intercept: float = 1.0, true_slope: float = 0.5,
) -> StandardErrorSimulationResult:
    """``Y=1+0.5X+u`` DGP'sinde vektörize eğim ve geleneksel SH üretir."""
    n, reps = _positive_int(nobs, "Örneklem büyüklüğü", minimum=3), _positive_int(repetitions, "Tekrar sayısı", minimum=2)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("Seed tam sayı olmalıdır.")
    intercept, slope = _finite(true_intercept, "Gerçek sabit"), _finite(true_slope, "Gerçek eğim")
    rng = np.random.default_rng(int(seed))
    x, errors = rng.normal(size=(reps, n)), rng.normal(size=(reps, n))
    y = intercept + slope * x + errors
    centered_x, centered_y = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
    sxx = np.square(centered_x).sum(axis=1)
    if np.any(sxx <= np.finfo(float).eps):
        raise RuntimeError("Benzetimde yeterli X değişkenliği oluşmadı.")
    slopes = (centered_x * centered_y).sum(axis=1) / sxx
    fitted = y.mean(axis=1, keepdims=True) + slopes[:, None] * centered_x
    residuals = y - fitted
    reported = np.sqrt(np.square(residuals).sum(axis=1) / (n - 2) / sxx)
    return StandardErrorSimulationResult(intercept, slope, n, reps, n - 2, slopes, reported, float(slopes.mean()), float(slopes.std(ddof=1)), float(reported.mean()), int(seed))


@dataclass(frozen=True)
class ConfidenceCoverageResult:
    """Tekrarlı güven aralığı kapsama benzetiminin çıktısı."""

    confidence_level: float
    critical_value: float
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    contains_truth: np.ndarray
    coverage_rate: float
    true_slope: float
    nobs: int
    repetitions: int
    seed: int

    def __post_init__(self) -> None:
        """Kapsama dizilerinin boyut ve sonluluğunu denetler."""
        lower, upper = np.asarray(self.lower_bounds, dtype=float).copy(), np.asarray(self.upper_bounds, dtype=float).copy()
        contains = np.asarray(self.contains_truth, dtype=bool).copy()
        if any(values.ndim != 1 or len(values) != self.repetitions for values in (lower, upper, contains)) or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
            raise ValueError("Kapsama dizileri geçerli aralıklardan oluşmalıdır.")
        for name, values in (("lower_bounds", lower), ("upper_bounds", upper), ("contains_truth", contains)):
            values.setflags(write=False)
            object.__setattr__(self, name, values)


def simulate_confidence_coverage(
    *, nobs: int = 50, repetitions: int = 5000, confidence_level: float = 0.95,
    seed: int = 202507, true_intercept: float = 1.0, true_slope: float = 0.5,
) -> ConfidenceCoverageResult:
    """Aynı DGP altında Student-t aralıklarının uzun dönem kapsamasını üretir."""
    level = _finite(confidence_level, "Güven düzeyi")
    if not 0.0 < level < 1.0:
        raise ValueError("Güven düzeyi 0 ile 1 arasında olmalıdır.")
    simulation = simulate_standard_errors(nobs=nobs, repetitions=repetitions, seed=seed, true_intercept=true_intercept, true_slope=true_slope)
    critical = float(student_t.ppf(1.0 - (1.0 - level) / 2.0, simulation.df_resid))
    margin = critical * simulation.reported_standard_errors
    lower, upper = simulation.slope_estimates - margin, simulation.slope_estimates + margin
    contains = (lower <= simulation.true_slope) & (simulation.true_slope <= upper)
    return ConfidenceCoverageResult(level, critical, lower, upper, contains, float(contains.mean()), simulation.true_slope, simulation.nobs, simulation.repetitions, simulation.seed)


def coverage_plot_data(result: ConfidenceCoverageResult, *, limit: int = 25) -> pd.DataFrame:
    """İlk güven aralıklarını grafik için etiketli tabloya dönüştürür."""
    count = _positive_int(limit, "Grafik aralık sayısı")
    count = min(count, result.repetitions)
    return pd.DataFrame({"interval": np.arange(1, count + 1), "lower": result.lower_bounds[:count], "upper": result.upper_bounds[:count], "contains_truth": result.contains_truth[:count], "status": np.where(result.contains_truth[:count], "Kapsıyor", "Kaçırıyor")})


def t_distribution_plot_data(df_resid: int, t_statistic: float, *, points: int = 601) -> pd.DataFrame:
    """Adaptif eksende Student-t yoğunluk grafiği için veri üretir."""
    df, observed, count = _positive_int(df_resid, "Serbestlik derecesi"), _finite(t_statistic, "t istatistiği"), _positive_int(points, "Grafik noktası", minimum=51)
    extent = max(4.5, abs(observed) + 1.0)
    x = np.linspace(-extent, extent, count)
    return pd.DataFrame({"t": x, "density": student_t.pdf(x, df)})
