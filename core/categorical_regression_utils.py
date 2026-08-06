"""Konu 10 için kategorik açıklayıcı değişken ve kukla araçları."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from core.joint_inference_utils import LinearRestriction, joint_f_test
from core.regression_inference_utils import OLSInferenceResult, fit_ols_inference


@dataclass(frozen=True)
class BinaryGroupSummary:
    """İki 0--1 grubunun complete-case ortalama özetidir."""
    group_variable: str; reference_value: int; comparison_value: int
    reference_label: str; comparison_label: str; reference_n: int; comparison_n: int
    reference_mean: float; comparison_mean: float; raw_difference: float


@dataclass(frozen=True)
class ReferenceCodingComparison:
    """İki referans kodlamasının tahmin değişmezliğini kaydeder."""
    first_reference: str; second_reference: str; coefficients_first: pd.Series; coefficients_second: pd.Series
    max_fitted_difference: float; max_residual_difference: float; r_squared_difference: float; ssr_difference: float


@dataclass(frozen=True)
class DummyTrapDiagnostic:
    """Kukla tuzağının rank denetimi sonucu."""
    n_categories: int; includes_intercept: bool; dummy_count: int; design_columns: tuple[str, ...]
    matrix_rank: int; column_count: int; full_rank: bool; exact_relation: str; explanation: str


@dataclass(frozen=True)
class CategoryJointTestSummary:
    """Bir kategori kukla kümesinin geleneksel ortak F testi."""
    category_name: str; reference_category: str; tested_coefficients: tuple[str, ...]
    q: int; f_statistic: float; p_value: float; reject_null: bool


@dataclass(frozen=True)
class DummyCoding:
    """Referans kategoriyle oluşturulmuş kukla tasarımı."""
    categories: tuple[str, ...]; reference_category: str; dummies: pd.DataFrame; coding_table: pd.DataFrame


@dataclass(frozen=True)
class RawControlledComparison:
    """Ham, kontrollü düzey ve log-kukla sonuçlarını birlikte sunar."""
    raw_model: OLSInferenceResult; controlled_level_model: OLSInferenceResult; controlled_log_model: OLSInferenceResult
    raw_difference: float; controlled_level_difference: float; controlled_log_difference: float
    exact_log_percent: float; exact_log_ci_percent: tuple[float, float]; controls: tuple[str, ...]; same_sample: bool


@dataclass(frozen=True)
class CodingComparison:
    """Tek sayısal kod ile referans kuklalarının grup ortalaması uyumu."""
    group_means: pd.DataFrame; numeric_fitted: pd.Series; dummy_fitted: pd.Series
    numeric_ssr: float; dummy_ssr: float; equal_spacing_imposed: bool


@dataclass(frozen=True)
class CategoryContrastInference:
    """İki kategori için model kovaryansından hesaplanan karşıtlık."""
    first_category: str; second_category: str; estimate: float; standard_error: float
    t_statistic: float; p_value: float; confidence_interval_95: tuple[float, float]


def _binary_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Mevcut gösterge sütunlarının gerçekten 0/1 olmasını doğrular."""
    for name in columns:
        if name not in frame:
            raise ValueError(f"Gerekli sütun bulunamadı: {name}")
        values = pd.to_numeric(frame[name], errors="coerce").dropna().unique()
        if not set(values).issubset({0, 1}):
            raise ValueError(f"{name} sütunu 0/1 gösterge değişkeni olmalıdır.")


def add_categorical_columns(frame: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """WAGE1 için gerekli kategorik sütunları doğrulayarak kopya döndürür."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Girdi bir pandas DataFrame olmalıdır.")
    data = frame.copy(deep=True)
    if dataset_key != "wage1":
        raise ValueError("Bu yardımcı şu anda yalnızca WAGE1 için tanımlıdır.")
    required = ("female", "married", "nonwhite", "northcen", "south", "west", "construc", "ndurman", "trcommpu", "trade", "services", "profserv")
    _binary_columns(data, required)
    for base, squared in (("exper", "expersq"), ("tenure", "tenursq")):
        if squared not in data:
            if base not in data:
                raise ValueError(f"Gerekli sütun bulunamadı: {base}")
            data[squared] = pd.to_numeric(data[base], errors="coerce") ** 2
    return data


def binary_group_summary(frame: pd.DataFrame, outcome: str, group: str, *, reference_value: int = 0, comparison_value: int = 1, reference_label: str = "Referans grup", comparison_label: str = "Karşılaştırma grubu") -> BinaryGroupSummary:
    """0/1 grup için karşılaştırma eksi referans ortalama farkını hesaplar."""
    if {reference_value, comparison_value} != {0, 1} or reference_value == comparison_value:
        raise ValueError("Referans ve karşılaştırma değerleri farklı 0 ve 1 olmalıdır.")
    if outcome not in frame or group not in frame:
        raise ValueError("Sonuç ve grup sütunları veri setinde bulunmalıdır.")
    data = frame.loc[:, [outcome, group]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    values = set(data[group].unique())
    if not values.issubset({0, 1}) or not {reference_value, comparison_value}.issubset(values):
        raise ValueError("Grup değişkeni 0/1 olmalı ve iki grup da gözlenmelidir.")
    ref, comp = data.loc[data[group] == reference_value, outcome], data.loc[data[group] == comparison_value, outcome]
    return BinaryGroupSummary(group, reference_value, comparison_value, reference_label, comparison_label, len(ref), len(comp), float(ref.mean()), float(comp.mean()), float(comp.mean() - ref.mean()))


def fit_binary_dummy_model(frame: pd.DataFrame, outcome: str, group: str) -> OLSInferenceResult:
    """Sabit ve tek 0/1 kukla ile geleneksel EKK modelini kurar."""
    binary_group_summary(frame, outcome, group)
    return fit_ols_inference(frame, outcome, (group,))


def dummy_log_exact_percent(delta: float) -> float:
    """Log bağımlı değişkende kukla katsayısını tam yüzde farka dönüştürür."""
    value = float(delta)
    if not math.isfinite(value) or value > 700:
        raise ValueError("Log-kukla katsayısı sonlu olmalı ve taşma yaratmamalıdır.")
    return float(100.0 * np.expm1(value))


def dummy_log_ci_to_exact_percent(lower: float, upper: float) -> tuple[float, float]:
    """Log katsayı güven aralığını sıra korunarak yüzde aralığına çevirir."""
    lo, hi = float(lower), float(upper)
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
        raise ValueError("Güven aralığı sonlu ve sıralı olmalıdır.")
    return dummy_log_exact_percent(lo), dummy_log_exact_percent(hi)


def compare_raw_and_controlled_dummy(frame: pd.DataFrame, group: str = "female") -> RawControlledComparison:
    """WAGE1 için ham ve eğitim/deneyim/kıdem kontrollü grup farklarını kurar."""
    raw = fit_binary_dummy_model(frame, "wage", group)
    controls = ("educ", "exper", "tenure")
    level = fit_ols_inference(frame, "wage", (group, *controls))
    log = fit_ols_inference(frame, "lwage", (group, *controls))
    ci = log.confidence_intervals_95.loc[group]
    raw_index = raw.observed_values.index
    return RawControlledComparison(raw, level, log, float(raw.coefficients[group]), float(level.coefficients[group]), float(log.coefficients[group]), dummy_log_exact_percent(log.coefficients[group]), dummy_log_ci_to_exact_percent(float(ci["lower"]), float(ci["upper"])), controls, bool(raw_index.equals(level.observed_values.index) and raw_index.equals(log.observed_values.index)))


def build_reference_dummies(category_series: pd.Series, *, reference_category: str, prefix: str, category_order: tuple[str, ...] | None = None) -> DummyCoding:
    """Belirtilen referans kategori dışında m-1 kukla ve kodlama tablosu üretir."""
    values = category_series.dropna().astype(str)
    categories = tuple(category_order or pd.unique(values))
    if len(categories) < 2 or reference_category not in categories or set(values.unique()).difference(categories):
        raise ValueError("Referans ve kategori sırası gözlenen kategorilerle uyumlu olmalıdır.")
    valid = category_series.notna()
    dummy_columns: dict[str, pd.Series] = {}
    for category in categories:
        if category == reference_category:
            continue
        dummy = pd.Series(np.nan, index=category_series.index, dtype=float)
        dummy.loc[valid] = (category_series.loc[valid].astype(str) == category).astype(float)
        dummy_columns[f"{prefix}_{category}"] = dummy
    dummies = pd.DataFrame(dummy_columns, index=category_series.index)
    table = pd.DataFrame({"kategori": categories, "referans": [cat == reference_category for cat in categories], "kod": ["0 (tüm kuklalar)" if cat == reference_category else f"{prefix}_{cat}=1" for cat in categories]})
    return DummyCoding(categories, reference_category, dummies, table)


def compare_reference_coding(frame: pd.DataFrame, dependent: str, controls: tuple[str, ...], category_series: pd.Series, *, first_reference: str, second_reference: str, prefix: str, category_order: tuple[str, ...] | None = None) -> ReferenceCodingComparison:
    """Aynı kategorik modelde iki referans seçiminin tahmin değişmezliğini denetler."""
    def model(reference: str) -> OLSInferenceResult:
        coding = build_reference_dummies(category_series, reference_category=reference, prefix=prefix, category_order=category_order)
        data = frame.copy(deep=True).join(coding.dummies)
        return fit_ols_inference(data, dependent, (*controls, *coding.dummies.columns))
    first, second = model(first_reference), model(second_reference)
    return ReferenceCodingComparison(first_reference, second_reference, first.coefficients, second.coefficients, float(np.max(np.abs(first.fitted_values.to_numpy() - second.fitted_values.to_numpy()))), float(np.max(np.abs(first.residuals.to_numpy() - second.residuals.to_numpy()))), abs(first.r_squared - second.r_squared), abs(first.ssr - second.ssr))


def diagnose_dummy_trap(category_series: pd.Series, *, includes_intercept: bool, include_all_dummies: bool = True, prefix: str = "D") -> DummyTrapDiagnostic:
    """Sabit ve kukla tasarımının rank'ını açıkça hesaplar."""
    values = category_series.dropna().astype(str)
    categories = tuple(pd.unique(values))
    if len(categories) < 2:
        raise ValueError("Rank tanısı için en az iki kategori gerekir.")
    use = categories if include_all_dummies else categories[:-1]
    design = pd.DataFrame({f"{prefix}_{cat}": (values == cat).astype(float).to_numpy() for cat in use})
    if includes_intercept:
        design.insert(0, "const", 1.0)
    rank = int(np.linalg.matrix_rank(design.to_numpy()))
    full = rank == design.shape[1]
    return DummyTrapDiagnostic(len(categories), includes_intercept, len(use), tuple(design.columns), rank, design.shape[1], full, "1 = " + " + ".join(f"{prefix}_{cat}" for cat in categories), "Tam doğrusal bağlantı vardır." if not full else "Tasarım matrisi tam ranklıdır.")


def category_joint_test(result: OLSInferenceResult, coefficients: tuple[str, ...], *, category_name: str, reference_category: str, alpha: float = .05) -> CategoryJointTestSummary:
    """Kategori kuklalarının hep birlikte sıfır olduğu geleneksel F testini kurar."""
    if not coefficients or any(name not in result.coefficients.index for name in coefficients):
        raise ValueError("Sınanacak kategori katsayıları modelde bulunmalıdır.")
    tested = joint_f_test(result, tuple(LinearRestriction({name: 1.0}, 0.0, f"{name} = 0") for name in coefficients), alpha=alpha)
    return CategoryJointTestSummary(category_name, reference_category, coefficients, tested.q, tested.f_statistic, tested.p_value, tested.reject_null)


def numeric_vs_dummy_coding_comparison(category: pd.Series, outcome: pd.Series) -> CodingComparison:
    """1–2–3 kodunun eşit aralık kısıtını serbest kategori ortalamalarıyla karşılaştırır."""
    data = pd.DataFrame({"category": category, "outcome": outcome}).dropna().copy()
    if data["category"].nunique() < 3:
        raise ValueError("Karşılaştırma için en az üç kategori gerekir.")
    order = tuple(pd.unique(data["category"].astype(str)))
    code_map = {name: index + 1 for index, name in enumerate(order)}
    data["numeric_code"] = data["category"].astype(str).map(code_map)
    numeric = fit_ols_inference(data, "outcome", ("numeric_code",))
    coding = build_reference_dummies(data["category"].astype(str), reference_category=order[0], prefix="cat", category_order=order)
    dummy_data = data.join(coding.dummies)
    dummy = fit_ols_inference(dummy_data, "outcome", tuple(coding.dummies.columns))
    means = data.groupby("category", sort=False)["outcome"].agg(["size", "mean"]).reset_index()
    return CodingComparison(means, numeric.fitted_values, dummy.fitted_values, numeric.ssr, dummy.ssr, True)


def category_contrast_inference(
    result: OLSInferenceResult,
    *,
    first_category: str,
    second_category: str,
    reference_category: str,
    coefficient_map: dict[str, str],
) -> CategoryContrastInference:
    """Kategori A−B karşıtlığını katsayı kovaryansıyla sınar."""
    names = list(result.coefficients.index)
    weights = np.zeros(len(names), dtype=float)
    for category, sign in ((first_category, 1.0), (second_category, -1.0)):
        if category == reference_category:
            continue
        coefficient = coefficient_map.get(category)
        if coefficient not in names:
            raise ValueError(f"{category} kategorisinin katsayısı modelde bulunamadı.")
        weights[names.index(coefficient)] += sign
    estimate = float(weights @ result.coefficients.to_numpy(float))
    variance = float(weights @ result.covariance_matrix.to_numpy(float) @ weights)
    if variance <= 0:
        raise ValueError("Kategori karşıtlığının varyansı pozitif olmalıdır.")
    from scipy.stats import t as student_t
    se = float(np.sqrt(variance)); statistic = estimate / se
    p_value = 2.0 * float(student_t.sf(abs(statistic), result.df_resid))
    critical = float(student_t.ppf(.975, result.df_resid))
    return CategoryContrastInference(first_category, second_category, estimate, se, statistic, p_value, (estimate-critical*se, estimate+critical*se))


def dummy_design_preview(category_series: pd.Series, *, includes_intercept: bool, include_all_dummies: bool, prefix: str = "D") -> tuple[pd.DataFrame, DummyTrapDiagnostic]:
    """Rank laboratuvarı için küçük tasarım matrisi ve tanıyı birlikte üretir."""
    values = category_series.dropna().astype(str)
    categories = tuple(pd.unique(values)); use = categories if include_all_dummies else categories[:-1]
    design = pd.DataFrame({f"{prefix}_{cat}": (values == cat).astype(int).to_numpy() for cat in use})
    if includes_intercept:
        design.insert(0, "const", 1)
    diagnostic = diagnose_dummy_trap(values, includes_intercept=includes_intercept, include_all_dummies=include_all_dummies, prefix=prefix)
    return design, diagnostic
