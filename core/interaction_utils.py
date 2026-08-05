"""Konu 11 için iki grup doğrusu ve etkileşim çıkarımı araçları."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from core.joint_inference_utils import LinearRestriction, joint_f_test
from core.regression_inference_utils import OLSInferenceResult, fit_ols_inference

@dataclass(frozen=True)
class GroupLines:
    """D=0 ve D=1 doğrularının sabit/eğim parametreleri."""
    intercept_zero: float; slope_zero: float; intercept_one: float; slope_one: float
    intercept_difference: float; slope_difference: float

@dataclass(frozen=True)
class ConditionalDifference:
    """Seçili X'te D=1 eksi D=0 farkı ve tekli t çıkarımı."""
    x_value: float; estimate: float; standard_error: float; t_statistic: float; p_value: float
    confidence_interval_95: tuple[float, float]

@dataclass(frozen=True)
class InteractionModelSummary:
    """Etkileşim modeli, grup doğruları ve ortak F testidir."""
    result: OLSInferenceResult; lines: GroupLines; joint_f: float; joint_p_value: float

@dataclass(frozen=True)
class CenteredInteractionComparison:
    """Merkezleme öncesi/sonrası eşdeğer etkileşim modelini karşılaştırır."""
    center: float; raw_result: OLSInferenceResult; centered_result: OLSInferenceResult
    fitted_max_difference: float; residual_max_difference: float; r_squared_difference: float; ssr_difference: float

def add_interaction_columns(frame: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """Konu 11 ders modellerinin türetilmiş sütunlarını kopyada oluşturur."""
    data = frame.copy(deep=True)
    if dataset_key == "wage1":
        required = ("educ", "female", "exper", "tenure", "lwage")
        if any(name not in data for name in required): raise ValueError("WAGE1 gerekli sütunları içermiyor.")
        if not set(pd.to_numeric(data["female"], errors="coerce").dropna().unique()).issubset({0, 1}): raise ValueError("female 0/1 olmalıdır.")
        data["educ12"] = pd.to_numeric(data["educ"], errors="coerce") - 12.0
        data["female_educ12"] = pd.to_numeric(data["female"], errors="coerce") * data["educ12"]
    elif dataset_key == "hprice1":
        required = ("colonial", "lotsize", "sqrft", "price", "bdrms")
        if any(name not in data for name in required): raise ValueError("HPRICE1 gerekli sütunları içermiyor.")
        # Ad ders notundaki merkezi (10 bin kare fit) belirtir; birim 1.000 kare fittir.
        data["lotsize10k"] = (pd.to_numeric(data["lotsize"], errors="coerce") - 10000.0) / 1000.0
        data["sqrft100"] = pd.to_numeric(data["sqrft"], errors="coerce") / 100.0
        data["colonial_lotsize10k"] = pd.to_numeric(data["colonial"], errors="coerce") * data["lotsize10k"]
    else: raise ValueError("Yalnızca WAGE1 ve HPRICE1 desteklenir.")
    return data

def group_lines(intercept: float, slope: float, dummy_effect: float, interaction_effect: float) -> GroupLines:
    """y=b0+b1X+g0D+g1DX modelinden iki grup doğrusunu çıkarır."""
    numbers = tuple(float(v) for v in (intercept, slope, dummy_effect, interaction_effect))
    if not all(math.isfinite(v) for v in numbers): raise ValueError("Model katsayıları sonlu olmalıdır.")
    b0,b1,g0,g1 = numbers
    return GroupLines(b0,b1,b0+g0,b1+g1,g0,g1)

def conditional_group_difference(result: OLSInferenceResult, dummy_name: str, interaction_name: str, x_value: float, *, alpha: float=.05) -> ConditionalDifference:
    """g0+g1x lineer bileşiminin tam kovaryansla t çıkarımını hesaplar."""
    if dummy_name not in result.coefficients or interaction_name not in result.coefficients: raise ValueError("Kukla ve etkileşim katsayıları modelde bulunmalıdır.")
    x=float(x_value)
    if not math.isfinite(x) or not 0 < alpha < 1: raise ValueError("X sonlu, alpha 0 ile 1 arasında olmalıdır.")
    weights=np.zeros(len(result.coefficients)); names=list(result.coefficients.index); weights[names.index(dummy_name)]=1; weights[names.index(interaction_name)]=x
    estimate=float(weights @ result.coefficients.to_numpy(float)); variance=float(weights @ result.covariance_matrix.to_numpy(float) @ weights)
    if variance < -1e-12: raise ValueError("Lineer bileşim varyansı negatif hesaplandı.")
    se=float(np.sqrt(max(variance, 0.0)))
    if se <= 0: raise ValueError("Lineer bileşim standart hatası pozitif olmalıdır.")
    t=estimate/se; p=2*float(student_t.sf(abs(t), result.df_resid)); critical=float(student_t.ppf(1-alpha/2,result.df_resid))
    return ConditionalDifference(x,estimate,se,t,p,(estimate-critical*se,estimate+critical*se))

def crossing_point(lines: GroupLines, *, data_min: float, data_max: float) -> tuple[float|None, bool|None]:
    """İki grubun kesişimini ve gözlenen X aralığında olup olmadığını verir."""
    if np.isclose(lines.slope_difference,0): return None,None
    point=float(-lines.intercept_difference/lines.slope_difference)
    return point, bool(data_min <= point <= data_max)

def fit_interaction_model(frame: pd.DataFrame, dependent: str, x_name: str, dummy_name: str, interaction_name: str, controls: tuple[str,...]=()) -> InteractionModelSummary:
    """Ana etkileri ve etkileşimi içeren modeli, grup doğrularını ve ortak F'yi kurar."""
    result=fit_ols_inference(frame,dependent,(dummy_name,x_name,interaction_name,*controls))
    lines=group_lines(result.coefficients["const"],result.coefficients[x_name],result.coefficients[dummy_name],result.coefficients[interaction_name])
    joint=joint_f_test(result,(LinearRestriction({dummy_name:1},0,f"{dummy_name}=0"),LinearRestriction({interaction_name:1},0,f"{interaction_name}=0")))
    return InteractionModelSummary(result,lines,joint.f_statistic,joint.p_value)

def classify_interaction_structure(dummy_effect: float, interaction_effect: float) -> str:
    """Dört temel grup regresyon yapısını etiketler."""
    if np.isclose(dummy_effect,0) and np.isclose(interaction_effect,0): return "aynı doğru"
    if not np.isclose(dummy_effect,0) and np.isclose(interaction_effect,0): return "paralel doğrular"
    if np.isclose(dummy_effect,0) and not np.isclose(interaction_effect,0): return "aynı sabit, farklı eğimler"
    return "farklı sabitler ve eğimler"

def center_interaction_model(frame: pd.DataFrame, dependent: str, x_name: str, dummy_name: str, controls: tuple[str,...], *, center: float) -> CenteredInteractionComparison:
    """X'i merkezleyerek eşdeğer etkileşim modelinin tahmin değişmezliğini doğrular."""
    c=float(center)
    if not math.isfinite(c): raise ValueError("Merkez sonlu olmalıdır.")
    raw_name=f"{dummy_name}_{x_name}"; data=frame.copy(deep=True)
    data[raw_name]=pd.to_numeric(data[dummy_name],errors="coerce")*pd.to_numeric(data[x_name],errors="coerce")
    raw=fit_ols_inference(data,dependent,(dummy_name,x_name,raw_name,*controls))
    centered_name=f"{x_name}_centered"; interaction_name=f"{dummy_name}_{centered_name}"
    data[centered_name]=pd.to_numeric(data[x_name],errors="coerce")-c; data[interaction_name]=pd.to_numeric(data[dummy_name],errors="coerce")*data[centered_name]
    centered=fit_ols_inference(data,dependent,(dummy_name,centered_name,interaction_name,*controls))
    return CenteredInteractionComparison(c,raw,centered,float(np.max(np.abs(raw.fitted_values.to_numpy()-centered.fitted_values.to_numpy()))),float(np.max(np.abs(raw.residuals.to_numpy()-centered.residuals.to_numpy()))),abs(raw.r_squared-centered.r_squared),abs(raw.ssr-centered.ssr))


def interaction_curve_data(lines: GroupLines, x_values: np.ndarray) -> pd.DataFrame:
    """İki grup doğrusu için çizime hazır tahmin ızgarası üretir."""
    x = np.asarray(x_values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("X ızgarası sonlu ve tek boyutlu olmalıdır.")
    return pd.DataFrame({"x": x, "D=0": lines.intercept_zero + lines.slope_zero*x, "D=1": lines.intercept_one + lines.slope_one*x})


def conditional_difference_grid(result: OLSInferenceResult, dummy_name: str, interaction_name: str, x_values: np.ndarray, *, data_min: float, data_max: float) -> pd.DataFrame:
    """X boyunca koşullu fark, SH ve güven bandını hesaplar."""
    rows=[]
    for value in np.asarray(x_values,dtype=float):
        inference=conditional_group_difference(result,dummy_name,interaction_name,float(value))
        rows.append({"x":float(value),"estimate":inference.estimate,"standard_error":inference.standard_error,"lower":inference.confidence_interval_95[0],"upper":inference.confidence_interval_95[1],"in_sample_range":bool(data_min<=value<=data_max)})
    return pd.DataFrame(rows)
