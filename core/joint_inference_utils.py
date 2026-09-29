"""Genel doğrusal kısıtlar (``R beta = r``) için geleneksel ortak F testi ve iç içe model karşılaştırması.

Konu 9–12'nin sayfaları ve yardımcı modülleri kullanır; Konu 8'in F testleri ``core.labs`` tanımlarından üretilir.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution

from core.regression_inference_utils import OLSInferenceResult, fit_ols_inference


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
