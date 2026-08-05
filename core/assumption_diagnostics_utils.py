"""Konu 06 varsayım, yanlılık ve bağlantı hesapları.

Bu modül Streamlit içermez; tüm rastgele süreçler açık seed ile yeniden
üretilebilir ve öğrenci arayüzü yalnız bu saf sonuçları biçimlendirir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from core.model_utils import SimpleOLSResult, fit_simple_ols
from core.multiple_regression_utils import MultipleOLSResult, fit_multiple_ols


DEFAULT_SIMULATION_SEED = 202506
_TOLERANCE = 1e-10


def _finite(value: object, label: str) -> float:
    """Sonlu bir sayıyı doğrular ve float olarak döndürür."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} sayısal olmalıdır.") from error
    if not np.isfinite(number):
        raise ValueError(f"{label} sonlu bir sayı olmalıdır.")
    return number


def _positive_int(value: object, label: str, *, minimum: int) -> int:
    """Benzetim boyutunu doğrular."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} tam sayı olmalıdır.")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{label} en az {minimum} olmalıdır.")
    return number


@dataclass(frozen=True)
class RepeatedSamplingResult:
    """Tekrarlı örnekleme ile elde edilen eğim tahminlerini taşır."""

    scenario_id: str
    true_intercept: float
    true_slope: float
    conditional_mean_loading: float
    nobs: int
    repetitions: int
    estimates: np.ndarray
    mean_estimate: float
    bias: float
    estimate_std: float
    target_center: float

    def __post_init__(self) -> None:
        estimates = np.asarray(self.estimates, dtype=float).copy()
        if estimates.ndim != 1 or len(estimates) != self.repetitions or not np.isfinite(estimates).all():
            raise ValueError("Tahmin dizisi sonlu ve tekrar sayısıyla uyumlu olmalıdır.")
        estimates.setflags(write=False)
        object.__setattr__(self, "estimates", estimates)


def bias_from_estimates(estimates: np.ndarray, target: float) -> float:
    """Tahmin ortalamasından hedef parametreyi çıkarır."""
    values = np.asarray(estimates, dtype=float)
    target_value = _finite(target, "Hedef parametre")
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Tahminler sonlu, boş olmayan tek boyutlu bir dizi olmalıdır.")
    return float(values.mean() - target_value)


def simulate_repeated_ols(
    *,
    scenario_id: str = "sifir_kosullu_ortalama",
    nobs: int = 80,
    repetitions: int = 3000,
    seed: int = DEFAULT_SIMULATION_SEED,
    true_intercept: float = 2.0,
    true_slope: float = 1.5,
    conditional_mean_loading: float = 0.0,
) -> RepeatedSamplingResult:
    """``u=gamma X+epsilon`` DGP'sinde vektörize EKK eğimlerini üretir."""
    n = _positive_int(nobs, "Örneklem büyüklüğü", minimum=3)
    reps = _positive_int(repetitions, "Tekrar sayısı", minimum=1)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("Seed tam sayı olmalıdır.")
    intercept = _finite(true_intercept, "Gerçek sabit")
    slope = _finite(true_slope, "Gerçek eğim")
    gamma = _finite(conditional_mean_loading, "Koşullu ortalama yükü")
    rng = np.random.default_rng(int(seed))
    x = rng.normal(size=(reps, n))
    epsilon = rng.normal(size=(reps, n))
    y = intercept + slope * x + gamma * x + epsilon
    centered_x = x - x.mean(axis=1, keepdims=True)
    centered_y = y - y.mean(axis=1, keepdims=True)
    denominator = np.square(centered_x).sum(axis=1)
    if np.any(denominator <= np.finfo(float).eps):
        raise RuntimeError("Benzetimde eğim hesaplamak için yeterli X değişimi oluşmadı.")
    estimates = (centered_x * centered_y).sum(axis=1) / denominator
    target = slope + gamma
    return RepeatedSamplingResult(
        scenario_id=scenario_id, true_intercept=intercept, true_slope=slope,
        conditional_mean_loading=gamma, nobs=n, repetitions=reps, estimates=estimates,
        mean_estimate=float(estimates.mean()), bias=bias_from_estimates(estimates, slope),
        estimate_std=float(estimates.std(ddof=1)) if reps > 1 else 0.0, target_center=target,
    )


@dataclass(frozen=True)
class OVBResult:
    """Tek dışlanan değişken için yön ve merkez hesabı."""

    bias: float
    expected_short_coefficient: float
    direction: str


def ovb_direction(omitted_outcome_coefficient: float, auxiliary_slope: float) -> str:
    """``beta_2 delta_1`` işaretine göre yanlılık yönünü sınıflandırır."""
    product = _finite(omitted_outcome_coefficient, "Dışlanan değişken katsayısı") * _finite(auxiliary_slope, "Yardımcı eğim")
    if np.isclose(product, 0.0, atol=_TOLERANCE):
        return "yanlılık yok / yaklaşık sıfır"
    return "yukarı yönlü" if product > 0 else "aşağı yönlü"


def ovb_bias(target_coefficient: float, omitted_outcome_coefficient: float, auxiliary_slope: float) -> OVBResult:
    """Eksik değişken yanlılığı formülünü uygular."""
    target = _finite(target_coefficient, "Hedef katsayı")
    omitted = _finite(omitted_outcome_coefficient, "Dışlanan değişken katsayısı")
    delta = _finite(auxiliary_slope, "Yardımcı eğim")
    bias = omitted * delta
    return OVBResult(bias=float(bias), expected_short_coefficient=float(target + bias), direction=ovb_direction(omitted, delta))


@dataclass(frozen=True)
class SyntheticOVBDecomposition:
    """Sentetik kısa–uzun–yardımcı model ayrıştırması."""

    short_model: SimpleOLSResult
    long_model: MultipleOLSResult
    auxiliary_model: SimpleOLSResult
    auxiliary_slope: float
    predicted_bias: float
    expected_short_slope: float
    observed_short_minus_long: float
    decomposition_error: float


def generate_synthetic_ovb_data(
    *, nobs: int = 5000, seed: int = DEFAULT_SIMULATION_SEED,
    z_loading: float = 0.7, x_coefficient: float = 2.0, z_coefficient: float = 3.0,
) -> pd.DataFrame:
    """Ders notundaki sentetik eksik değişken DGP'sini üretir."""
    n = _positive_int(nobs, "Örneklem büyüklüğü", minimum=4)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("Seed tam sayı olmalıdır.")
    loading = _finite(z_loading, "Z yükü")
    x_beta = _finite(x_coefficient, "X katsayısı")
    z_beta = _finite(z_coefficient, "Z katsayısı")
    rng = np.random.default_rng(int(seed))
    x = rng.normal(size=n)
    z = loading * x + rng.normal(size=n)
    y = 1.0 + x_beta * x + z_beta * z + rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": z, "y": y})


def synthetic_ovb_decomposition(
    frame: pd.DataFrame, *, x_coefficient: float = 2.0, z_coefficient: float = 3.0
) -> SyntheticOVBDecomposition:
    """Verilen sentetik veride kısa–uzun–yardımcı regresyon kimliğini kurar."""
    original = frame.copy(deep=True)
    short = fit_simple_ols(frame, "y", "x")
    long = fit_multiple_ols(frame, "y", ("x", "z"))
    auxiliary = fit_simple_ols(frame, "z", "x")
    if not frame.equals(original):
        raise RuntimeError("Girdi veri çerçevesi değiştirilmemelidir.")
    delta = auxiliary.slope
    predicted = _finite(z_coefficient, "Z katsayısı") * delta
    observed = short.slope - float(long.coefficients["x"])
    error = observed - float(long.coefficients["z"]) * delta
    return SyntheticOVBDecomposition(short, long, auxiliary, float(delta), float(predicted), float(_finite(x_coefficient, "X katsayısı") + predicted), float(observed), float(error))


@dataclass(frozen=True)
class Wage1OVBDecomposition:
    """WAGE1 ortak örneklemindeki eğitim–deneyim ayrıştırması."""

    short_model: SimpleOLSResult
    middle_model: MultipleOLSResult
    full_model: MultipleOLSResult
    auxiliary_model: SimpleOLSResult
    nobs: int
    decomposition_error: float


def wage1_ovb_decomposition(frame: pd.DataFrame) -> Wage1OVBDecomposition:
    """WAGE1 modellerini aynı complete-case örneklemde tahmin eder."""
    required = ["wage", "educ", "exper", "tenure"]
    missing = [item for item in required if item not in frame.columns]
    if missing:
        raise ValueError(f"WAGE1 için eksik sütunlar: {', '.join(missing)}")
    common = frame.loc[:, required].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(common) < 5:
        raise ValueError("WAGE1 ortak örneklemi yetersizdir.")
    short = fit_simple_ols(common, "wage", "educ")
    middle = fit_multiple_ols(common, "wage", ("educ", "exper"))
    full = fit_multiple_ols(common, "wage", ("educ", "exper", "tenure"))
    auxiliary = fit_simple_ols(common, "exper", "educ")
    error = short.slope - (float(middle.coefficients["educ"]) + float(middle.coefficients["exper"]) * auxiliary.slope)
    return Wage1OVBDecomposition(short, middle, full, auxiliary, len(common), float(error))


@dataclass(frozen=True)
class DesignMatrixRank:
    """Sabit terimli tasarım matrisinin rank tanısı."""

    rank: int
    columns: int
    full_rank: bool


def design_matrix_rank(frame: pd.DataFrame, explanatory: tuple[str, ...]) -> DesignMatrixRank:
    """Sabit terim içeren tasarım matrisinin rank'ını döndürür."""
    if not explanatory or len(set(explanatory)) != len(explanatory):
        raise ValueError("En az bir, benzersiz açıklayıcı değişken gereklidir.")
    missing = [item for item in explanatory if item not in frame.columns]
    if missing:
        raise ValueError(f"Veri setinde bulunamayan değişken: {', '.join(missing)}")
    data = frame.loc[:, list(explanatory)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2:
        raise ValueError("Rank hesabı için yeterli sonlu gözlem yoktur.")
    matrix = np.column_stack((np.ones(len(data)), data.to_numpy(dtype=float)))
    rank = int(np.linalg.matrix_rank(matrix))
    return DesignMatrixRank(rank=rank, columns=matrix.shape[1], full_rank=rank == matrix.shape[1])


@dataclass(frozen=True)
class CollinearityDemo:
    """Tasarlanmış tam/yüksek bağlantı senaryosu."""

    scenario_id: str
    data: pd.DataFrame
    explanatory: tuple[str, ...]
    expected_exact_collinearity: bool
    rank_result: DesignMatrixRank
    explanation: str


def exact_collinearity_demo(scenario_id: str = "monthly_annual_income", *, seed: int = DEFAULT_SIMULATION_SEED) -> CollinearityDemo:
    """Dört sabit bağlantı senaryosundan birini hazırlar."""
    rng = np.random.default_rng(int(seed))
    base = rng.normal(100, 15, size=40)
    if scenario_id == "monthly_annual_income":
        data = pd.DataFrame({"monthly_income": base, "annual_income": 12 * base})
        explanatory, exact, text = tuple(data.columns), True, "Yıllık gelir her gözlemde aylık gelirin 12 katıdır; ayrı katsayılar benzersiz değildir."
    elif scenario_id == "total_and_components":
        food = np.abs(base); other = np.abs(rng.normal(60, 10, size=40))
        data = pd.DataFrame({"total_spending": food + other, "food": food, "non_food": other})
        explanatory, exact, text = tuple(data.columns), True, "Toplam harcama iki bileşenin tam toplamıdır."
    elif scenario_id == "exact_linear_combination":
        x1, x2 = rng.normal(size=(2, 40))
        data = pd.DataFrame({"x1": x1, "x2": x2, "x3": 2 + 4 * x1 - x2})
        explanatory, exact, text = tuple(data.columns), True, "x3 bütün gözlemlerde x1 ve x2'nin sabitli doğrusal birleşimidir."
    elif scenario_id == "high_but_not_exact":
        data = generate_near_collinearity_data("very_high", nobs=40, seed=seed).data.loc[:, ["x1", "x2"]]
        explanatory, exact, text = tuple(data.columns), False, "Değişkenler çok yakın hareket eder; ancak tam doğrusal kimlik yoktur."
    else:
        raise ValueError("Desteklenmeyen bağlantı senaryosu.")
    rank = design_matrix_rank(data, explanatory)
    return CollinearityDemo(scenario_id, data, explanatory, exact, rank, text)


def calculate_vif(frame: pd.DataFrame, explanatory: tuple[str, ...]) -> pd.DataFrame:
    """Her açıklayıcı için yardımcı regresyon R-kare ve VIF hesaplar."""
    if not explanatory or len(set(explanatory)) != len(explanatory):
        raise ValueError("VIF için en az bir, benzersiz açıklayıcı değişken gerekir.")
    original = frame.copy(deep=True)
    data = frame.loc[:, list(explanatory)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < len(explanatory) + 1:
        raise ValueError("VIF için yeterli ortak gözlem yoktur.")
    rows: list[dict[str, object]] = []
    for variable in explanatory:
        others = tuple(item for item in explanatory if item != variable)
        if not others:
            rows.append({"variable": variable, "auxiliary_r_squared": 0.0, "vif": 1.0, "exact_collinearity": False})
            continue
        rank = design_matrix_rank(data, others)
        complete = design_matrix_rank(data, explanatory)
        if not complete.full_rank:
            rows.append({"variable": variable, "auxiliary_r_squared": 1.0, "vif": np.inf, "exact_collinearity": True})
            continue
        renamed = data.rename(columns={variable: "_target"})
        auxiliary_r_squared = (
            fit_simple_ols(renamed, "_target", others[0]).r_squared
            if len(others) == 1
            else fit_multiple_ols(renamed, "_target", others).r_squared
        )
        r_squared = min(1.0, max(0.0, auxiliary_r_squared))
        exact = bool(np.isclose(r_squared, 1.0, atol=1e-12) or not rank.full_rank)
        rows.append({"variable": variable, "auxiliary_r_squared": r_squared, "vif": np.inf if exact else 1.0 / (1.0 - r_squared), "exact_collinearity": exact})
    if not frame.equals(original):
        raise RuntimeError("VIF hesabı girdi veri çerçevesini değiştirmemelidir.")
    return pd.DataFrame(rows)


NEAR_COLLINEARITY_SCENARIOS: dict[str, tuple[str, float]] = {
    "low_medium": ("Düşük/orta bağlantı", 1.0),
    "high": ("Yüksek bağlantı", 0.5),
    "very_high": ("Çok yüksek bağlantı", 0.25),
    "near_exact": ("Neredeyse tam fakat tam olmayan bağlantı", 0.12),
}


@dataclass(frozen=True)
class NearCollinearityData:
    """Yüksek fakat tam olmayan bağlantı DGP'si ve metadata'sı."""

    scenario_id: str
    scenario_label: str
    sigma: float
    theoretical_correlation: float
    theoretical_vif: float
    data: pd.DataFrame
    rank_result: DesignMatrixRank


def generate_near_collinearity_data(scenario_id: str = "high", *, nobs: int = 80, seed: int = DEFAULT_SIMULATION_SEED) -> NearCollinearityData:
    """``X2=X1+sigma V`` DGP'sini tam rank doğrulamasıyla üretir."""
    try:
        label, sigma = NEAR_COLLINEARITY_SCENARIOS[scenario_id]
    except KeyError as error:
        raise ValueError("Desteklenmeyen yüksek bağlantı senaryosu.") from error
    n = _positive_int(nobs, "Örneklem büyüklüğü", minimum=5)
    rng = np.random.default_rng(int(seed))
    x1, v, error = rng.normal(size=(3, n))
    x2 = x1 + sigma * v
    y = 1.0 + 2.0 * x1 + 2.0 * x2 + error
    data = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    rank = design_matrix_rank(data, ("x1", "x2"))
    if not rank.full_rank:
        raise RuntimeError("Tam olmayan bağlantı senaryosu rank eksik üretilmemelidir.")
    rho = 1.0 / np.sqrt(1.0 + sigma ** 2)
    return NearCollinearityData(scenario_id, label, sigma, rho, 1.0 / (1.0 - rho ** 2), data, rank)


@dataclass(frozen=True)
class CollinearityVariabilityResult:
    """Tekrarlı bağlantı benzetiminin özet istatistikleri."""

    mean_correlation: float
    mean_vif: float
    beta1_mean: float
    beta2_mean: float
    beta_sum_mean: float
    beta1_std: float
    beta2_std: float
    beta_sum_std: float
    repetitions: int


def simulate_collinearity_variability(scenario_id: str = "high", *, nobs: int = 80, repetitions: int = 1000, seed: int = DEFAULT_SIMULATION_SEED) -> CollinearityVariabilityResult:
    """Bağlantı altında katsayıların tekrarlı örnek değişkenliğini hesaplar."""
    try:
        _, sigma = NEAR_COLLINEARITY_SCENARIOS[scenario_id]
    except KeyError as error:
        raise ValueError("Desteklenmeyen yüksek bağlantı senaryosu.") from error
    n, reps = _positive_int(nobs, "Örneklem büyüklüğü", minimum=5), _positive_int(repetitions, "Tekrar sayısı", minimum=2)
    rng = np.random.default_rng(int(seed))
    x1, v, error = rng.normal(size=(3, reps, n))
    x2 = x1 + sigma * v
    y = 1.0 + 2.0 * x1 + 2.0 * x2 + error
    x1c, x2c, yc = (value - value.mean(axis=1, keepdims=True) for value in (x1, x2, y))
    s11, s22, s12 = (x1c * x1c).sum(axis=1), (x2c * x2c).sum(axis=1), (x1c * x2c).sum(axis=1)
    determinant = s11 * s22 - s12 ** 2
    if np.any(determinant <= np.finfo(float).eps):
        raise RuntimeError("Benzetimde tekil tasarım matrisi oluştu.")
    beta1 = (s22 * (x1c * yc).sum(axis=1) - s12 * (x2c * yc).sum(axis=1)) / determinant
    beta2 = (s11 * (x2c * yc).sum(axis=1) - s12 * (x1c * yc).sum(axis=1)) / determinant
    correlations = s12 / np.sqrt(s11 * s22)
    vifs = 1.0 / np.maximum(1.0 - correlations ** 2, np.finfo(float).eps)
    total = beta1 + beta2
    return CollinearityVariabilityResult(float(correlations.mean()), float(vifs.mean()), float(beta1.mean()), float(beta2.mean()), float(total.mean()), float(beta1.std(ddof=1)), float(beta2.std(ddof=1)), float(total.std(ddof=1)), reps)


@dataclass(frozen=True)
class CoefficientSensitivityResult:
    """Tek gözlem pertürbasyonu öncesi ve sonrası model karşılaştırması."""

    original_coefficients: dict[str, float]
    perturbed_coefficients: dict[str, float]
    original_r_squared: float
    perturbed_r_squared: float
    original_vif: pd.DataFrame
    perturbed_vif: pd.DataFrame
    original_rank: DesignMatrixRank
    perturbed_rank: DesignMatrixRank


def coefficient_sensitivity(frame: pd.DataFrame, *, observation_position: int = 0, adjustment: float = 0.4) -> CoefficientSensitivityResult:
    """Bir X2 gözlemini değiştirip katsayı duyarlılığını karşılaştırır."""
    if not isinstance(observation_position, (int, np.integer)):
        raise ValueError("Gözlem sırası tam sayı olmalıdır.")
    change = _finite(adjustment, "Değişiklik miktarı")
    required = {"x1", "x2", "y"}
    if not required.issubset(frame.columns):
        raise ValueError("Duyarlılık hesabı x1, x2 ve y sütunlarını gerektirir.")
    original = frame.copy(deep=True)
    data = frame.loc[:, ["x1", "x2", "y"]].copy()
    if not 0 <= int(observation_position) < len(data):
        raise IndexError("Gözlem sırası veri aralığında olmalıdır.")
    perturbed = data.copy()
    perturbed.iloc[int(observation_position), perturbed.columns.get_loc("x2")] += change
    original_rank, perturbed_rank = design_matrix_rank(data, ("x1", "x2")), design_matrix_rank(perturbed, ("x1", "x2"))
    if not (original_rank.full_rank and perturbed_rank.full_rank):
        raise ValueError("Pertürbasyon tam doğrusal bağlantı oluşturdu; katsayılar karşılaştırılamaz.")
    first, second = fit_multiple_ols(data, "y", ("x1", "x2")), fit_multiple_ols(perturbed, "y", ("x1", "x2"))
    if not frame.equals(original):
        raise RuntimeError("Duyarlılık hesabı girdi veri çerçevesini değiştirmemelidir.")
    return CoefficientSensitivityResult({key: float(first.coefficients[key]) for key in ("x1", "x2")}, {key: float(second.coefficients[key]) for key in ("x1", "x2")}, first.r_squared, second.r_squared, calculate_vif(data, ("x1", "x2")), calculate_vif(perturbed, ("x1", "x2")), original_rank, perturbed_rank)
