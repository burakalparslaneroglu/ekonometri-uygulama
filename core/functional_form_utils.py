"""Konu 09 için ölçekleme, log-yüzde ve karesel model araçları."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.joint_inference_utils import NestedModelFResult, nested_exclusion_f_test
from core.regression_inference_utils import OLSInferenceResult, fit_ols_inference


def _finite(value: object, label: str) -> float:
    """Sonlu bir sayıyı doğrular."""
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} sayısal olmalıdır.") from error
    if not np.isfinite(value):
        raise ValueError(f"{label} sonlu olmalıdır.")
    return value


def add_wage1_quadratic_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """WAGE1 için deneyim ve kıdem karelerini girdiyi değiştirmeden ekler."""
    if not isinstance(frame, pd.DataFrame) or not {"exper", "tenure"}.issubset(frame.columns):
        raise ValueError("WAGE1 için exper ve tenure sütunları gereklidir.")
    output = frame.copy(deep=True)
    for original, derived in (("exper", "expersq"), ("tenure", "tenursq")):
        calculated = pd.to_numeric(output[original], errors="coerce") ** 2
        if derived in output and not np.allclose(pd.to_numeric(output[derived], errors="coerce"), calculated, equal_nan=True):
            raise ValueError(f"{derived} hazır sütunu {original}² ile uyuşmuyor.")
        output[derived] = calculated
    return output


def is_linear_in_parameters(expression_id: str) -> bool:
    """Öğretim örneklerinin parametrelerde doğrusal olup olmadığını sınıflandırır."""
    choices = {"quadratic": True, "log_level": True, "log_log": True, "level_level": True, "beta_squared": False}
    if expression_id not in choices:
        raise ValueError("Bilinmeyen model sınıflandırması.")
    return choices[expression_id]


@dataclass(frozen=True)
class RescalingResult:
    """İki aynı modelin ölçek değişimi altındaki çıkarım karşılaştırması."""

    original: OLSInferenceResult
    rescaled: OLSInferenceResult
    coefficient_factor: float
    standard_error_factor: float
    t_invariant: bool
    p_invariant: bool
    r_squared_invariant: bool
    overall_f_invariant: bool


def rescale_and_refit(frame: pd.DataFrame, dependent: str, explanatory: tuple[str, ...], *, dependent_factor: float = 1.0, explanatory_factor: float = 1.0, focal_explanatory: str | None = None) -> RescalingResult:
    """Y'=aY, seçilen X'=bX altında aynı complete-case modelini yeniden tahmin eder."""
    a, b = _finite(dependent_factor, "Y ölçek çarpanı"), _finite(explanatory_factor, "X ölçek çarpanı")
    if a <= 0 or b <= 0:
        raise ValueError("Ölçek çarpanları pozitif olmalıdır.")
    focal = focal_explanatory or explanatory[0]
    if focal not in explanatory:
        raise ValueError("Odak açıklayıcı modelde bulunmalıdır.")
    original = fit_ols_inference(frame, dependent, explanatory)
    transformed = frame.copy(deep=True)
    transformed[dependent] = pd.to_numeric(transformed[dependent], errors="coerce") * a
    transformed[focal] = pd.to_numeric(transformed[focal], errors="coerce") * b
    rescaled = fit_ols_inference(transformed, dependent, explanatory)
    factor = a / b if focal != "const" else a
    return RescalingResult(original, rescaled, factor, factor,
                           bool(np.isclose(original.t_values_zero[focal], rescaled.t_values_zero[focal])),
                           bool(np.isclose(original.p_values_two_sided_zero[focal], rescaled.p_values_two_sided_zero[focal])),
                           bool(np.isclose(original.r_squared, rescaled.r_squared)),
                           bool(np.isclose(_overall_f(original), _overall_f(rescaled))))


def _overall_f(result: OLSInferenceResult) -> float:
    """Sabit dışındaki eğimlerin sıfır olduğu model F'sini açık hesaplar."""
    return float((result.r_squared / result.n_explanatory) / ((1.0 - result.r_squared) / result.df_resid))


@dataclass(frozen=True)
class StandardizedRegressionResult:
    """Z-skorlarıyla tahmin edilen model ve standart katsayılar."""

    result: OLSInferenceResult
    standardized_coefficients: pd.Series
    means: pd.Series
    standard_deviations: pd.Series


def standardized_regression(frame: pd.DataFrame, dependent: str, explanatory: tuple[str, ...]) -> StandardizedRegressionResult:
    """Y ve açıklayıcıları örneklem standart sapmasıyla standartlaştırıp EKK tahmin eder."""
    columns = (dependent, *explanatory)
    prepared = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().copy()
    means, stds = prepared.mean(), prepared.std(ddof=1)
    if (stds <= 0).any() or not np.isfinite(stds.to_numpy()).all():
        raise ValueError("Standartlaştırma için tüm değişkenlerin pozitif standart sapması olmalıdır.")
    standardized = (prepared - means) / stds
    result = fit_ols_inference(standardized, dependent, explanatory)
    raw = fit_ols_inference(prepared, dependent, explanatory)
    betas = pd.Series({name: float(raw.coefficients[name] * stds[name] / stds[dependent]) for name in explanatory})
    if not np.allclose(betas.to_numpy(), result.coefficients.loc[list(explanatory)].to_numpy(), rtol=1e-10, atol=1e-10):
        raise RuntimeError("Standart katsayı doğrulaması başarısız oldu.")
    return StandardizedRegressionResult(result, betas, means.copy(), stds.copy())


def log_level_percent_change(beta: float, delta_x: float = 1.0, *, exact: bool = False) -> float:
    """Log-düzey modelde yaklaşık veya tam yüzde değişimi hesaplar."""
    change = _finite(beta, "Katsayı") * _finite(delta_x, "X değişimi")
    if exact:
        if change > 700:
            raise ValueError("Tam yüzde hesabı taşma riski nedeniyle yapılamaz.")
        return float(100.0 * np.expm1(change))
    return float(100.0 * change)


@dataclass(frozen=True)
class QuadraticEffect:
    """Karesel modelin belirli düzeydeki eğimi ve tam birim farkı."""

    x_value: float
    marginal_effect: float
    discrete_change: float


def quadratic_marginal_effect(beta_linear: float, beta_quadratic: float, x_value: float) -> float:
    """``beta1 + 2 beta2 X`` koşullu tahmin eğimini döndürür."""
    return float(_finite(beta_linear, "Düzey katsayısı") + 2.0 * _finite(beta_quadratic, "Kare katsayısı") * _finite(x_value, "X değeri"))


def quadratic_discrete_change(beta_linear: float, beta_quadratic: float, x_value: float, delta_x: float = 1.0) -> float:
    """Karesel modelde X'ten X+delta'ya tam tahmin farkını döndürür."""
    b1, b2, x, delta = (_finite(beta_linear, "Düzey katsayısı"), _finite(beta_quadratic, "Kare katsayısı"), _finite(x_value, "X değeri"), _finite(delta_x, "X değişimi"))
    return float(b1 * delta + b2 * (2.0 * x * delta + delta ** 2))


@dataclass(frozen=True)
class TurningPoint:
    """Karesel modelin dönüm noktası ve veri aralığı kontrolü."""

    value: float | None
    kind: str | None
    inside_data_range: bool | None
    data_min: float
    data_max: float


def quadratic_turning_point(beta_linear: float, beta_quadratic: float, values: pd.Series | np.ndarray) -> TurningPoint:
    """Dönüm noktası ile gözlenen X aralığındaki yerini belirler."""
    b1, b2 = _finite(beta_linear, "Düzey katsayısı"), _finite(beta_quadratic, "Kare katsayısı")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Veri aralığı sonlu ve boş olmayan bir dizi olmalıdır.")
    low, high = float(array.min()), float(array.max())
    if np.isclose(b2, 0.0):
        return TurningPoint(None, None, None, low, high)
    point = float(-b1 / (2.0 * b2))
    return TurningPoint(point, "tepe" if b2 < 0 else "dip", bool(low <= point <= high), low, high)


@dataclass(frozen=True)
class Wage1ModelComparison:
    """WAGE1 M1–M4 modelleri ve kareli terimlerin ortak F sonucu."""

    models: dict[str, OLSInferenceResult]
    metrics: pd.DataFrame
    quadratic_joint_test: NestedModelFResult


def wage1_model_comparison(frame: pd.DataFrame) -> Wage1ModelComparison:
    """ln(wage) için ders notundaki M1–M4 hiyerarşik modellerini kurar."""
    data = add_wage1_quadratic_columns(frame)
    specs = {
        "M1": ("educ", "exper", "tenure"),
        "M2": ("educ", "exper", "expersq", "tenure"),
        "M3": ("educ", "exper", "tenure", "tenursq"),
        "M4": ("educ", "exper", "expersq", "tenure", "tenursq"),
    }
    models = {name: fit_ols_inference(data, "lwage", explanatory) for name, explanatory in specs.items()}
    metrics = pd.DataFrame([{"Model": name, "n": model.nobs, "R²": model.r_squared, "Düzeltilmiş R²": model.adjusted_r_squared, "SSR": model.ssr} for name, model in models.items()])
    joint = nested_exclusion_f_test(data, "lwage", specs["M4"], specs["M1"])
    return Wage1ModelComparison(models, metrics, joint)


@dataclass(frozen=True)
class CenteringResult:
    """Ham ve merkezlenmiş karesel modelin tahmin-eşdeğer karşılaştırması."""

    center: float
    raw_result: OLSInferenceResult
    centered_result: OLSInferenceResult
    centered_column: str
    centered_square_column: str
    marginal_effect_at_center: float
    fitted_max_difference: float
    residual_max_difference: float
    correlation_before: float
    correlation_after: float


def center_quadratic_model(frame: pd.DataFrame, dependent: str, linear_name: str, square_name: str, controls: tuple[str, ...], *, center: float) -> CenteringResult:
    """X-c ve (X-c)² ile eşdeğer karesel modeli, referans eğimiyle kurar."""
    c = _finite(center, "Merkez noktası")
    data = frame.copy(deep=True)
    if linear_name not in data or square_name not in data:
        raise ValueError("Ham doğrusal ve kare sütunları veri setinde bulunmalıdır.")
    raw_explanatory = (linear_name, square_name, *controls)
    raw = fit_ols_inference(data, dependent, raw_explanatory)
    centered_name, centered_square = f"{linear_name}_centered", f"{linear_name}_centered_sq"
    data[centered_name] = pd.to_numeric(data[linear_name], errors="coerce") - c
    data[centered_square] = data[centered_name] ** 2
    centered = fit_ols_inference(data, dependent, (centered_name, centered_square, *controls))
    fitted_difference = float(np.max(np.abs(raw.fitted_values.to_numpy() - centered.fitted_values.to_numpy())))
    residual_difference = float(np.max(np.abs(raw.residuals.to_numpy() - centered.residuals.to_numpy())))
    return CenteringResult(c, raw, centered, centered_name, centered_square,
                           quadratic_marginal_effect(raw.coefficients[linear_name], raw.coefficients[square_name], c),
                           fitted_difference, residual_difference,
                           float(pd.Series(data[linear_name]).corr(pd.Series(data[square_name]))),
                           float(pd.Series(data[centered_name]).corr(pd.Series(data[centered_square]))))


def wrong_functional_form_data(seed: int = 202509) -> pd.DataFrame:
    """Yanlış doğrusal biçimin eğri ortalamayı kaçırdığını gösteren deterministik veri üretir."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 20.0, 100)
    truth = 1.0 + 0.6 * x - 0.03 * x ** 2
    return pd.DataFrame({"x": x, "gercek_egri": truth, "y": truth + rng.normal(scale=0.35, size=len(x)), "resmi_tani_testi": False})
