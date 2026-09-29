"""En küçük kareler: tahmin, katsayı ve model nicelikleri, makale tipi tablo.

Uygulama ve üretilen Python kodu aynı kütüphaneyi (statsmodels formül arayüzü) aynı biçimde çağırır; sayılar
bit düzeyinde aynıdır. R'de ``lm()`` aynı tahmin örneklemini (eksik değerli satırlar dışarıda) ve aynı klasik
standart hataları verir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.labs.spec import COEF_QUANTITIES, INTERCEPT, MODEL_QUANTITIES, OLS, RegressionTable

STAR_LEVELS = ((0.01, "***"), (0.05, "**"), (0.10, "*"))
"""Makale tablosunda yıldız eşikleri: iki yönlü p-değeri 0,01, 0,05 ve 0,10'dan küçükse ***, **, *."""


def fit_ols(op: OLS, frame: pd.DataFrame):
    import statsmodels.formula.api as smf

    used = [op.outcome, *op.regressors]
    missing = sorted(set(used) - set(frame.columns))
    if missing:
        raise ValueError(f"Veride olmayan değişken: {', '.join(missing)}")
    data = frame[used].dropna()
    if len(data) <= len(op.regressors) + 1:
        raise ValueError("Tahmin için yeterli gözlem yok (gözlem sayısı katsayı sayısından büyük olmalıdır).")
    design = np.column_stack([np.ones(len(data)), data[list(op.regressors)].to_numpy(dtype=float)])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError(
            "Açıklayıcı değişkenler arasında tam doğrusal bağlantı var (biri diğerlerinin doğrusal birleşimi); "
            "katsayılar tek biçimde tahmin edilemez."
        )
    return smf.ols(op.formula, data=frame).fit()


def coefficient(result, term: str, quantity: str) -> float:
    if quantity not in COEF_QUANTITIES:
        raise ValueError(f"Desteklenmeyen katsayı niceliği: {quantity}")
    if term not in result.params.index:
        raise KeyError(f"Modelde böyle bir terim yok: {term}")
    if quantity == "coef":
        return float(result.params[term])
    if quantity == "se":
        return float(result.bse[term])
    if quantity == "t":
        return float(result.tvalues[term])
    if quantity == "p":
        return float(result.pvalues[term])
    interval = result.conf_int()
    return float(interval.loc[term, 0 if quantity == "ci_low" else 1])


def model_quantity(result, quantity: str) -> float:
    if quantity not in MODEL_QUANTITIES:
        raise ValueError(f"Desteklenmeyen model niceliği: {quantity}")
    # Yalnız istenen nicelik hesaplanır: ör. tam uyumda F istatistiği tanımsızdır, R² ise hesaplanabilir.
    getters = {
        "r2": lambda: result.rsquared, "adj_r2": lambda: result.rsquared_adj, "nobs": lambda: result.nobs,
        "f": lambda: result.fvalue, "f_p": lambda: result.f_pvalue, "ssr": lambda: result.ssr,
        "df_resid": lambda: result.df_resid,
    }
    return float(getters[quantity]())


def coefficient_table(result) -> pd.DataFrame:
    """Katsayı tablosu: satırlar terimler (modeldeki sırayla), sütunlar ``COEF_QUANTITIES``."""

    return pd.DataFrame(
        {quantity: [coefficient(result, term, quantity) for term in result.params.index]
         for quantity in COEF_QUANTITIES},
        index=pd.Index(result.params.index, name="terim"),
    )


def stars(p_value: float) -> str:
    for level, mark in STAR_LEVELS:
        if p_value < level:
            return mark
    return ""


def regression_table(op: RegressionTable, models: dict) -> pd.DataFrame:
    """Makale tipi tablonun sayıları: ``terim`` ve (``standard_errors`` ise) ``terim_sh`` satırları, ``n``, (``r2``
    ise) ``r2`` ve (``adj_r2`` ise) ``adj_r2``; sütunlar modeller."""

    columns = {}
    for heading, name in op.models:
        result = models[name]
        cells: list[float] = []
        for term in op.terms:
            present = term in result.params.index
            cells.append(float(result.params[term]) if present else np.nan)
            if op.standard_errors:
                cells.append(float(result.bse[term]) if present else np.nan)
        cells.append(float(result.nobs))
        if op.r2:
            cells.append(float(result.rsquared))
        if op.adj_r2:
            cells.append(float(result.rsquared_adj))
        columns[heading] = cells
    rows_per_term = (lambda term: (term, f"{term}_sh")) if op.standard_errors else (lambda term: (term,))
    index = ([label for term in op.terms for label in rows_per_term(term)] + ["n"] + (["r2"] if op.r2 else [])
             + (["adj_r2"] if op.adj_r2 else []))
    return pd.DataFrame(columns, index=pd.Index(index, name="satir"))


def table_stars(op: RegressionTable, models: dict) -> dict[tuple[str, str], str]:
    """Her (terim, sütun başlığı) için yıldız; modelde olmayan terimde boş."""

    found = {}
    for heading, name in op.models:
        result = models[name]
        for term in op.terms:
            found[(term, heading)] = stars(float(result.pvalues[term])) if term in result.params.index else ""
    return found


def term_label(term: str, label) -> str:
    return "Sabit terim" if term == INTERCEPT else label(term)
