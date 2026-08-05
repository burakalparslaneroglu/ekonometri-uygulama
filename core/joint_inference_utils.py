"""Konu 08 için geleneksel ortak F testleri ve büyük örneklem araçları."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution

from core.regression_inference_utils import CoefficientInference, OLSInferenceResult, fit_ols_inference


def _finite(value: object, label: str) -> float:
    """Sonlu bir sayıyı doğrular."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} sayısal olmalıdır.") from error
    if not np.isfinite(number):
        raise ValueError(f"{label} sonlu bir sayı olmalıdır.")
    return number


def _alpha(value: object) -> float:
    """Açık aralıktaki anlamlılık düzeyini doğrular."""
    alpha = _finite(value, "Anlamlılık düzeyi")
    if not 0.0 < alpha < 1.0:
        raise ValueError("Anlamlılık düzeyi 0 ile 1 arasında olmalıdır.")
    return alpha


@dataclass(frozen=True)
class LinearRestriction:
    """``R beta = r`` sistemindeki tek bir doğrusal kısıt."""

    weights: Mapping[str, float]
    rhs: float
    label: str

    def __post_init__(self) -> None:
        """Kısıt tanımını sonlu, değişmez ve boş olmayan hale getirir."""
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("Kısıt etiketi boş olamaz.")
        if not isinstance(self.weights, Mapping) or not self.weights:
            raise ValueError("Kısıt en az bir katsayı ağırlığı içermelidir.")
        checked: dict[str, float] = {}
        for name, weight in self.weights.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Katsayı adı boş olamaz.")
            if name in checked:
                raise ValueError("Aynı kısıtta yinelenen katsayı adı bulunamaz.")
            checked[name] = _finite(weight, f"{name} ağırlığı")
        object.__setattr__(self, "weights", dict(checked))
        object.__setattr__(self, "rhs", _finite(self.rhs, "Kısıtın sağ tarafı"))


@dataclass(frozen=True)
class JointFTestResult:
    """Genel doğrusal kısıt için geleneksel ortak F testi."""

    restrictions: tuple[LinearRestriction, ...]
    restriction_labels: tuple[str, ...]
    q: int
    f_statistic: float
    p_value: float
    df_num: int
    df_denom: int
    critical_value: float
    alpha: float
    reject_null: bool
    covariance_type: str


@dataclass(frozen=True)
class NestedModelFResult:
    """Sıfıra eşit dışlama kısıtları için iki eşdeğer F hesabı."""

    restricted_result: OLSInferenceResult
    unrestricted_result: OLSInferenceResult
    q: int
    ssr_restricted: float
    ssr_unrestricted: float
    r_squared_restricted: float
    r_squared_unrestricted: float
    f_from_ssr: float
    f_from_r_squared: float
    f_matrix: float | None
    p_value: float
    df_num: int
    df_denom: int
    formulas_match: bool


@dataclass(frozen=True)
class RestrictionValidationResult:
    """Kullanıcı kısıt sisteminin teknik geçerlilik denetimi."""

    is_valid: bool
    row_count: int
    rank_r: int
    rank_augmented: int
    q: int
    has_zero_information_row: bool
    is_consistent: bool
    has_dependent_rows: bool
    messages: tuple[str, ...]


@dataclass(frozen=True)
class RestrictionClassification:
    """Kısıt sisteminin öğretim amaçlı tür ve yöntem sınıflaması."""

    restriction_type: str
    nested_exclusion_applicable: bool
    explanation: str
    excluded_coefficients: tuple[str, ...]


def _raw_restriction_matrix(result: OLSInferenceResult, restrictions: tuple[LinearRestriction, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Sıfır ve bağımlı satırlar dahil, kullanıcı girişi için R ve r kurar."""
    if not restrictions:
        raise ValueError("En az bir doğrusal kısıt gereklidir.")
    names = tuple(result.coefficients.index)
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for restriction in restrictions:
        unknown = set(restriction.weights).difference(names)
        if unknown:
            raise ValueError(f"Modelde bulunmayan katsayı: {', '.join(sorted(unknown))}")
        rows.append(np.array([restriction.weights.get(name, 0.0) for name in names], dtype=float))
        rhs.append(float(restriction.rhs))
    return np.vstack(rows), np.asarray(rhs, dtype=float)


def validate_restriction_system(result: OLSInferenceResult, restrictions: tuple[LinearRestriction, ...]) -> RestrictionValidationResult:
    """Rütbe ve augmented-matrix ile custom kısıtların teknik geçerliliğini denetler."""
    matrix, rhs = _raw_restriction_matrix(result, restrictions)
    tolerance = np.finfo(float).eps * max(matrix.shape) * max(1.0, float(np.linalg.norm(matrix, ord=2)))
    rank_r = int(np.linalg.matrix_rank(matrix, tol=tolerance))
    augmented = np.column_stack((matrix, rhs))
    rank_augmented = int(np.linalg.matrix_rank(augmented, tol=tolerance))
    zero_rows = np.all(np.isclose(matrix, 0.0, atol=tolerance), axis=1)
    has_zero = bool(np.any(zero_rows))
    consistent = rank_r == rank_augmented
    dependent = rank_r < len(restrictions)
    messages: list[str] = []
    if has_zero:
        for index in np.flatnonzero(zero_rows):
            if np.isclose(rhs[index], 0.0, atol=tolerance):
                messages.append(f"{index + 1}. satır bilgi taşımıyor: 0 = 0.")
            else:
                messages.append(f"{index + 1}. satır tutarsız: 0 = {rhs[index]:g}.")
    if not consistent:
        messages.append("Kısıt sistemi tutarsız: rank(R) < rank([R|r]).")
    if dependent:
        messages.append("Girilen satırlar bağımsız değildir; ikinci satır yeni bağımsız bilgi eklemiyor olabilir.")
    if rank_r > len(result.coefficients):
        messages.append("Bağımsız kısıt sayısı modeldeki parametre boyutunu aşıyor.")
    if not messages:
        messages.append("Kısıt sistemi teknik olarak geçerli; girilen satırlar bağımsız ve tutarlıdır.")
    valid = bool(rank_r > 0 and consistent and not dependent and rank_r <= len(result.coefficients))
    return RestrictionValidationResult(valid, len(restrictions), rank_r, rank_augmented, rank_r, has_zero, consistent, dependent, tuple(messages))


def classify_restriction_system(result: OLSInferenceResult, restrictions: tuple[LinearRestriction, ...]) -> RestrictionClassification:
    """Kısıtları dışlama, eşitlik veya genel matrix-F türü olarak sınıflandırır."""
    validation = validate_restriction_system(result, restrictions)
    if not validation.is_valid:
        return RestrictionClassification("genel/karma doğrusal kısıt", False, "Teknik olarak geçersiz sistem için test yöntemi seçilmez.", ())
    slopes = set(result.explanatory)
    nonzero = [
        {name: weight for name, weight in restriction.weights.items() if not np.isclose(weight, 0.0)}
        for restriction in restrictions
    ]
    exclusion = all(len(weights) == 1 and next(iter(weights)) in slopes and np.isclose(next(iter(weights.values())), 1.0) and np.isclose(restriction.rhs, 0.0) for weights, restriction in zip(nonzero, restrictions))
    if exclusion:
        excluded = tuple(next(iter(weights)) for weights in nonzero)
        if set(excluded) == slopes:
            return RestrictionClassification("genel anlamlılık", True, "Bütün eğimler sıfıra eşitlenir; kısıtlı model yalnız sabit içerir.", excluded)
        return RestrictionClassification("değişken dışlama", True, "Her kısıt ayrı bir eğimi sıfıra eşitler; değişken silerek nested karşılaştırma kurulabilir.", excluded)
    if len(restrictions) == 1:
        weights, restriction = nonzero[0], restrictions[0]
        if len(weights) == 2 and np.isclose(restriction.rhs, 0.0) and set(np.round(list(weights.values()), 12)) == {-1.0, 1.0}:
            return RestrictionClassification("katsayı eşitliği", False, "Eşitlik kısıtında değişken silinmez; genel Rβ=r matrix F kullanılır.", ())
        if len(weights) == 1 and not np.isclose(restriction.rhs, 0.0):
            return RestrictionClassification("sıfır dışı tek katsayı", False, "Katsayı sıfır dışı hedefe eşitlenir; genel Rβ=r matrix F kullanılır.", ())
        if len(weights) > 1:
            return RestrictionClassification("doğrusal birleşim", False, "Birden çok katsayının doğrusal birleşimi sınanır; genel Rβ=r matrix F kullanılır.", ())
    return RestrictionClassification("genel/karma doğrusal kısıt", False, "Bu sistem güvenle basit dışlama olarak sınıflandırılamaz; genel Rβ=r matrix F kullanılır.", ())


def format_linear_restriction(restriction: LinearRestriction, *, labels: Mapping[str, str] | None = None) -> str:
    """Bir kısıtı düzgün işaretlerle okunabilir beta gösterimine dönüştürür."""
    pieces: list[str] = []
    for name, weight in restriction.weights.items():
        if np.isclose(weight, 0.0):
            continue
        label = (labels or {}).get(name, name)
        symbol = f"β_{label}" if name != "const" else "β_sabit"
        magnitude = abs(float(weight))
        term = symbol if np.isclose(magnitude, 1.0) else f"{magnitude:g}·{symbol}"
        if not pieces:
            pieces.append(term if weight > 0 else f"−{term}")
        else:
            pieces.append((" + " if weight > 0 else " − ") + term)
    lhs = "".join(pieces) if pieces else "0"
    return f"{lhs} = {restriction.rhs:g}"


def _restriction_matrix(result: OLSInferenceResult, restrictions: tuple[LinearRestriction, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Kısıt nesnelerini model katsayı düzenindeki R ve r matrisine çevirir."""
    validation = validate_restriction_system(result, restrictions)
    if not validation.is_valid:
        raise ValueError("Geçersiz kısıt sistemi: " + " ".join(validation.messages))
    return _raw_restriction_matrix(result, restrictions)


def joint_f_test(result: OLSInferenceResult, restrictions: tuple[LinearRestriction, ...], *, alpha: float = 0.05) -> JointFTestResult:
    """``H0: R beta=r`` için yalnız geleneksel kovaryansla F testi kurar."""
    level = _alpha(alpha)
    if result.covariance_type != "nonrobust":
        raise ValueError("Ortak F testi yalnız nonrobust kovaryans türüyle kullanılabilir.")
    matrix, rhs = _restriction_matrix(result, restrictions)
    covariance = result.covariance_matrix.loc[result.coefficients.index, result.coefficients.index].to_numpy(dtype=float)
    difference = matrix @ result.coefficients.to_numpy(dtype=float) - rhs
    middle = matrix @ covariance @ matrix.T
    if np.linalg.matrix_rank(middle) != len(restrictions):
        raise ValueError("Kısıt kovaryans matrisi tekildir; F testi tanımlı değildir.")
    statistic = float(difference.T @ np.linalg.solve(middle, difference) / len(restrictions))
    if statistic < -1e-10 or not np.isfinite(statistic):
        raise ValueError("F istatistiği hesaplanamadı.")
    statistic = max(0.0, statistic)
    p_value = float(f_distribution.sf(statistic, len(restrictions), result.df_resid))
    critical = float(f_distribution.ppf(1.0 - level, len(restrictions), result.df_resid))
    return JointFTestResult(restrictions, tuple(item.label for item in restrictions), len(restrictions), statistic, p_value,
                            len(restrictions), result.df_resid, critical, level, bool(p_value < level), result.covariance_type)


def nested_exclusion_f_test(frame: pd.DataFrame, dependent: str, unrestricted_explanatory: tuple[str, ...], restricted_explanatory: tuple[str, ...], *, alpha: float = 0.05) -> NestedModelFResult:
    """Aynı complete-case örnekleminde dışlama kısıtlarının SSR/R² F testini yapar."""
    level = _alpha(alpha)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Girdi bir pandas DataFrame olmalıdır.")
    if len(set(unrestricted_explanatory)) != len(unrestricted_explanatory) or len(set(restricted_explanatory)) != len(restricted_explanatory):
        raise ValueError("Açıklayıcı değişken adları benzersiz olmalıdır.")
    if not set(restricted_explanatory) < set(unrestricted_explanatory):
        raise ValueError("Kısıtlı açıklayıcılar kısıtsız modelin gerçek alt kümesi olmalıdır.")
    columns = (dependent, *unrestricted_explanatory)
    if any(name not in frame.columns for name in columns):
        raise ValueError("Bağımlı değişken ve tüm açıklayıcılar veri setinde bulunmalıdır.")
    prepared = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().copy()
    restricted = fit_ols_inference(prepared, dependent, restricted_explanatory)
    unrestricted = fit_ols_inference(prepared, dependent, unrestricted_explanatory)
    if restricted.nobs != unrestricted.nobs:
        raise ValueError("Kısıtlı ve kısıtsız model aynı complete-case örneklemi kullanmalıdır.")
    q = len(unrestricted_explanatory) - len(restricted_explanatory)
    ssr_r, ssr_ur = float(restricted.ssr), float(unrestricted.ssr)
    if ssr_ur > ssr_r + 1e-8 * max(1.0, ssr_r):
        raise ValueError("Kısıtsız modelin SSR değeri kısıtlı modelden büyük olamaz.")
    f_ssr = max(0.0, ((ssr_r - ssr_ur) / q) / (ssr_ur / unrestricted.df_resid))
    r2_r, r2_ur = restricted.r_squared, unrestricted.r_squared
    denominator = (1.0 - r2_ur) / unrestricted.df_resid
    if denominator <= 0:
        raise ValueError("R-kare F hesabının paydası pozitif olmalıdır.")
    f_r2 = max(0.0, ((r2_ur - r2_r) / q) / denominator)
    excluded = tuple(name for name in unrestricted_explanatory if name not in restricted_explanatory)
    matrix = joint_f_test(unrestricted, tuple(LinearRestriction({name: 1.0}, 0.0, f"{name} = 0") for name in excluded), alpha=level)
    return NestedModelFResult(restricted, unrestricted, q, ssr_r, ssr_ur, r2_r, r2_ur, f_ssr, f_r2,
                              matrix.f_statistic, matrix.p_value, q, unrestricted.df_resid,
                              bool(np.isclose(f_ssr, f_r2, rtol=1e-8, atol=1e-8) and np.isclose(f_ssr, matrix.f_statistic, rtol=1e-8, atol=1e-8)))


def overall_f_test(result: OLSInferenceResult, *, alpha: float = 0.05) -> JointFTestResult:
    """Sabit hariç tüm eğimlerin birlikte sıfır olduğu genel F testini yapar."""
    restrictions = tuple(LinearRestriction({name: 1.0}, 0.0, f"{name} = 0") for name in result.explanatory)
    return joint_f_test(result, restrictions, alpha=alpha)


def single_restriction_equivalence(coefficient_test_result: CoefficientInference, joint_test_result: JointFTestResult, *, tolerance: float = 1e-8) -> bool:
    """Uygun iki taraflı tek kısıtta ``F=t²`` eşitliğini doğrular."""
    if coefficient_test_result.alternative != "two-sided" or joint_test_result.q != 1:
        raise ValueError("F=t² eşdeğerliği yalnız iki taraflı tek kısıtta uygulanabilir.")
    if coefficient_test_result.df_resid != joint_test_result.df_denom:
        raise ValueError("t ve F testlerinin payda serbestlik dereceleri aynı olmalıdır.")
    if not np.isclose(coefficient_test_result.null_value, joint_test_result.restrictions[0].rhs):
        raise ValueError("t ve F testlerinin null değerleri aynı olmalıdır.")
    if coefficient_test_result.coefficient_name not in joint_test_result.restrictions[0].weights:
        raise ValueError("t testi ortak F kısıtındaki katsayıyla eşleşmelidir.")
    return bool(np.isclose(joint_test_result.f_statistic, coefficient_test_result.t_statistic ** 2, rtol=tolerance, atol=tolerance))


def f_distribution_plot_data(f_observed: float, df_num: int, df_denom: int, alpha: float = 0.05) -> dict[str, object]:
    """Üst kuyruk F grafiği için adaptif, Streamlit'ten bağımsız veri üretir."""
    observed, level = _finite(f_observed, "Gözlenen F"), _alpha(alpha)
    if observed < 0 or df_num < 1 or df_denom < 1:
        raise ValueError("F ve serbestlik dereceleri geçerli olmalıdır.")
    critical = float(f_distribution.ppf(1.0 - level, df_num, df_denom))
    baseline = float(f_distribution.ppf(0.999, df_num, df_denom))
    limit = max(baseline * 1.12, critical * 1.20, min(observed * 1.10, baseline * 5.0))
    x = np.linspace(0.0, limit, 600)
    density = f_distribution.pdf(x, df_num, df_denom)
    return {"x": x, "density": density, "critical_value": critical, "f_observed": observed,
            "critical_tail_mask": x >= critical, "p_value_tail_mask": x >= observed,
            "x_limit": limit, "observed_outside": bool(observed > limit)}


@dataclass(frozen=True)
class LargeSampleJointSimulationResult:
    """Sağa çarpık hata altında vektörize ortak F benzetim özeti."""

    sample_sizes: tuple[int, ...]
    repetitions: int
    rejection_rates: pd.Series
    standardized_slope_means: pd.Series
    standardized_slope_stds: pd.Series
    standardized_slope_skewness: pd.Series
    seed: int


def simulate_large_sample_joint_test(*, sample_sizes: tuple[int, ...] = (25, 100, 500), repetitions: int = 4000, seed: int = 202508, alpha: float = 0.05) -> LargeSampleJointSimulationResult:
    """Normal olmayan hata altında sıfır iki eğim için batch F benzetimi yapar."""
    level = _alpha(alpha)
    if not sample_sizes or any(not isinstance(n, (int, np.integer)) or n < 5 for n in sample_sizes):
        raise ValueError("Örneklem büyüklükleri en az 5 olan tam sayılar olmalıdır.")
    if not isinstance(repetitions, (int, np.integer)) or repetitions < 2:
        raise ValueError("Tekrar sayısı en az 2 olan tam sayı olmalıdır.")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("Seed tam sayı olmalıdır.")
    rng = np.random.default_rng(int(seed))
    rates: dict[int, float] = {}; means: dict[int, float] = {}; stds: dict[int, float] = {}; skews: dict[int, float] = {}
    for n in sample_sizes:
        x1 = rng.normal(size=(int(repetitions), int(n)))
        x2 = 0.55 * x1 + np.sqrt(1.0 - 0.55 ** 2) * rng.normal(size=(int(repetitions), int(n)))
        errors = rng.exponential(scale=1.0, size=(int(repetitions), int(n))) - 1.0
        design = np.stack((np.ones_like(x1), x1, x2), axis=-1)
        xtx = np.einsum("rni,rnj->rij", design, design)
        xty = np.einsum("rni,rn->ri", design, 2.0 + errors)
        beta = np.linalg.solve(xtx, xty[..., None])[..., 0]
        residuals = 2.0 + errors - np.einsum("rni,ri->rn", design, beta)
        sigma2 = np.square(residuals).sum(axis=1) / (int(n) - 3)
        inv_xtx = np.linalg.inv(xtx)
        se = np.sqrt(np.maximum(sigma2[:, None] * np.diagonal(inv_xtx, axis1=1, axis2=2)[:, 1:], np.finfo(float).eps))
        standardized = beta[:, 1:] / se
        f_values = np.square(standardized).sum(axis=1) / 2.0
        # Correlated slopes require the exact quadratic form, not the sum of t squares.
        covariance_slopes = sigma2[:, None, None] * inv_xtx[:, 1:, 1:]
        f_values = np.einsum("ri,rij,rj->r", beta[:, 1:], np.linalg.inv(covariance_slopes), beta[:, 1:]) / 2.0
        rates[int(n)] = float(np.mean(f_values > f_distribution.ppf(1.0 - level, 2, int(n) - 3)))
        flattened = standardized.reshape(-1)
        means[int(n)], stds[int(n)] = float(flattened.mean()), float(flattened.std(ddof=1))
        skews[int(n)] = float(np.mean(((flattened - flattened.mean()) / flattened.std(ddof=0)) ** 3))
    index = pd.Index(tuple(int(n) for n in sample_sizes), name="n")
    return LargeSampleJointSimulationResult(tuple(int(n) for n in sample_sizes), int(repetitions), pd.Series(rates, index=index), pd.Series(means, index=index), pd.Series(stds, index=index), pd.Series(skews, index=index), int(seed))


def classify_large_sample_scenario(scenario_id: str) -> dict[str, str | bool]:
    """Büyük n'nin hangi sorunu çözebileceğini güvenli dille sınıflandırır."""
    scenarios = {
        "omitted_ability": (False, "Örnekleme belirsizliği azalabilir.", "Gözlenmeyen yetenek kaynaklı eksik değişken yanlılığı sürer."),
        "voluntary_online_sample": (False, "Örneklem içindeki tahmin daha hassas olabilir.", "Gönüllü katılımdan doğan seçilim ve genellenebilirlik sorunu sürer."),
        "wrong_quadratic_form": (False, "Yanlış biçim daha görünür hale gelebilir.", "Yanlış fonksiyonel biçim kendiliğinden düzelmez."),
        "repeated_firms_dependence": (False, "Daha çok satır bilgi artırabilir.", "Bağımlı gözlemler uygun bağımlılık/standart hata yaklaşımı gerektirir."),
        "correct_design_more_n": (True, "Örnekleme belirsizliği ve standart hatalar genellikle azalır.", "İktisadi önem veya nedensellik otomatik olarak artmaz."),
        "heteroskedastic_nonrobust": (False, "Nokta tahmini daha hassas olabilir.", "Heteroskedastisite altında nonrobust standart hata ve F geçerli olmayabilir."),
    }
    if scenario_id not in scenarios:
        raise ValueError("Bilinmeyen büyük örneklem senaryosu.")
    solves, improves, not_solved = scenarios[scenario_id]
    return {"sorunu_cozer_mi": solves, "ne_iyilesebilir": improves, "ne_cozulmez": not_solved,
            "guvenli_aciklama": "Büyük örneklem yanlış merkezi, tasarımı veya bağımlılığı otomatik olarak düzeltmez."}
