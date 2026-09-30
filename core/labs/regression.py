"""En küçük kareler: tahmin, katsayı ve model nicelikleri, makale tipi tablo.

Uygulama ve üretilen Python kodu aynı kütüphaneyi (statsmodels formül arayüzü) aynı biçimde çağırır; sayılar
bit düzeyinde aynıdır. R'de ``lm()`` aynı tahmin örneklemini (eksik değerli satırlar dışarıda) ve aynı klasik
standart hataları verir. Dayanıklı (HC0–HC3) kovaryans statsmodels'te ``fit(cov_type=..., use_t=True)`` ya da
``model.HC1_se`` ile, R'de sandviç formülüyle açıkça hesaplanır (Konu 12).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from core.labs.spec import (
    COEF_QUANTITIES,
    INTERCEPT,
    MODEL_QUANTITIES,
    OLS,
    CoefficientTable,
    RegressionTable,
)

STAR_LEVELS = ((0.01, "***"), (0.05, "**"), (0.10, "*"))
"""Makale tablosunda yıldız eşikleri: iki yönlü p-değeri 0,01, 0,05 ve 0,10'dan küçükse ***, **, *."""
CONFIDENCE_Q = 1 - 0.05 / 2
"""Yüzde 95 güven aralığının t kantili: statsmodels ``conf_int(alpha=0.05)`` hesabıyla aynı sayı (0,975)."""


def design_columns(op: OLS, data: pd.DataFrame) -> list[np.ndarray]:
    """Terimlerin sütunları (sabit terim hariç): değişken ya da etkileşimde iki değişkenin çarpımı."""

    columns = []
    for term in op.regressors:
        parts = term.split(":")
        values = data[parts[0]].to_numpy(dtype=float)
        if len(parts) == 2:
            values = values * data[parts[1]].to_numpy(dtype=float)
        columns.append(values)
    return columns


def fit_ols(op: OLS, frame: pd.DataFrame):
    import statsmodels.formula.api as smf

    used = list(op.variables)
    missing = sorted(set(used) - set(frame.columns))
    if missing:
        raise ValueError(f"Veride olmayan değişken: {', '.join(missing)}")
    data = frame[used].dropna()
    if len(data) <= len(op.regressors) + 1:
        raise ValueError("Tahmin için yeterli gözlem yok (gözlem sayısı katsayı sayısından büyük olmalıdır).")
    design = np.column_stack([np.ones(len(data)), *design_columns(op, data)])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError(
            "Açıklayıcı değişkenler arasında tam doğrusal bağlantı var (biri diğerlerinin doğrusal birleşimi); "
            "katsayılar tek biçimde tahmin edilemez."
        )
    model = smf.ols(op.formula, data=frame)
    if op.cov_type == "nonrobust":
        return model.fit()
    # use_t=True: t ve F dağılımı (n − k − 1 serbestlik derecesi); yalnız cov_type verilirse statsmodels normal dağılım
    # kullanır ve p-değerleri notlardaki (ve R'deki) sayılardan farklı olur.
    return model.fit(cov_type=op.cov_type, use_t=True)


def covariance_label(result) -> str:
    """Modelin katsayı kovaryansı: ``nonrobust`` ya da ``HC0``–``HC3``."""

    return str(getattr(result, "cov_type", "nonrobust"))


def robust_quantity(result, term: str, quantity: str, cov_type: str) -> float:
    """Aynı EKK tahmininden dayanıklı kovaryansla standart hata, t, p ya da yüzde 95 güven sınırı; üretilen Python
    koduyla aynı işlem sırası (``m.params[t] / m.HC1_se[t]`` gibi)."""

    coef = float(result.params[term])
    bse = float(getattr(result, f"{cov_type}_se")[term])
    if quantity == "se":
        return bse
    if quantity == "t":
        return coef / bse
    if quantity == "p":
        return float(stats.t.sf(abs(coef / bse), result.df_resid) * 2)
    critical = stats.t.ppf(CONFIDENCE_Q, result.df_resid)
    return float(coef - critical * bse) if quantity == "ci_low" else float(coef + critical * bse)


def coefficient(result, term: str, quantity: str, cov_type: str | None = None) -> float:
    if quantity not in COEF_QUANTITIES:
        raise ValueError(f"Desteklenmeyen katsayı niceliği: {quantity}")
    if term not in result.params.index:
        raise KeyError(f"Modelde böyle bir terim yok: {term}")
    if cov_type is not None and quantity != "coef":
        return robust_quantity(result, term, quantity, cov_type)
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


def confidence_interval(result, term: str, level: float) -> tuple[float, float]:
    """Yüzde ``100·level`` güven aralığı β̂ ± t_{α/2; n−k−1}·se(β̂), α = 1 − level (statsmodels ``conf_int``)."""

    interval = result.conf_int(alpha=round(1 - level, 10))
    return float(interval.loc[term, 0]), float(interval.loc[term, 1])


def inference_table(op: CoefficientTable, result) -> pd.DataFrame:
    """Tek katsayı çıkarım tablosu: satırlar terimler, sütunlar katsayı, standart hata, t, p ve güven aralığı."""

    missing = [term for term in op.terms if term not in result.params.index]
    if missing:
        raise KeyError(f"Modelde böyle bir terim yok: {', '.join(missing)}")
    interval = result.conf_int(alpha=round(1 - op.level, 10))
    terms = list(op.terms)
    return pd.DataFrame(
        {
            "katsayi": result.params[terms].to_numpy(dtype=float),
            "sh": result.bse[terms].to_numpy(dtype=float),
            "t": result.tvalues[terms].to_numpy(dtype=float),
            "p": result.pvalues[terms].to_numpy(dtype=float),
            "alt": interval.loc[terms, 0].to_numpy(dtype=float),
            "ust": interval.loc[terms, 1].to_numpy(dtype=float),
        },
        index=pd.Index(terms, name="terim"),
    )


def restriction_text(terms: tuple[str, ...]) -> str:
    """statsmodels ``f_test`` kısıt yazımı: "exper = 0, tenure = 0"."""

    return ", ".join(f"{term} = 0" for term in terms)


def joint_test(result, terms: tuple[str, ...]) -> tuple[float, float]:
    """Dışlama kısıtlarının F testi (statsmodels ``f_test``): F istatistiği ve p-değeri."""

    missing = [term for term in terms if term not in result.params.index]
    if missing:
        raise KeyError(f"Modelde böyle bir terim yok: {', '.join(missing)}")
    test = result.f_test(restriction_text(terms))
    return float(np.asarray(test.fvalue).squeeze()), float(np.asarray(test.pvalue).squeeze())


WHITE_RCOND = 1e-10
"""White yardımcı tasarımında bağımsız sütunları ayıran göreli tekil değer eşiği (sütunlar birim uzunlukta). Aynı
sütunun ikinci kopyasının (0/1 kuklanın karesi) tekil değeri ~1e-15 kalır; gerçek terimlerinki ≥ 1e-4'tür (WAGE1 ve
HPRICE1'in bütün açıklayıcı seçimleri). Eşik ikisinin arasındadır; her birinden en az beş büyüklük mertebesi uzaktır."""


def white_design(exog) -> np.ndarray:
    """White yardımcı regresyonunun tasarımı: sabit, açıklayıcıların düzeyleri, kareleri ve çapraz çarpımları.

    Yalnız sıfırdan oluşan sütunlar (ör. birbirini dışlayan iki kuklanın çarpımı) atılır; sütunlar birim uzunluğa
    çekilir, böylece rank kararı sütun ölçeğinden (ör. arsa büyüklüğünün karesi) etkilenmez."""

    x = np.asarray(exog, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or not np.all(x[:, 0] == 1):
        raise ValueError("White testi için modelde sabit terim (ilk sütun) ve en az bir açıklayıcı değişken olmalıdır.")
    level = x[:, 1:]
    first, second = np.triu_indices(level.shape[1])
    design = np.column_stack([np.ones(len(level)), level, level[:, first] * level[:, second]])
    design = design[:, design.any(axis=0)]
    return design / np.linalg.norm(design, axis=0)


def white_test(resid, exog) -> tuple[float, float]:
    """White (1980) LM testi: kareli artıklar, açıklayıcıların düzeyleri, kareleri ve çapraz çarpımları üzerine regrese
    edilir; LM = n·R², p = P(χ²_q > LM), q = yardımcı regresyondaki bağımsız terim sayısı (sabit hariç).

    0/1 kuklanın karesi kendisidir: yinelenen sütun rankla ayıklanır (``np.linalg.lstsq``, eşik ``WHITE_RCOND``) ve q'ya
    bir kez girer. Üretilen Python kodundaki ``white_testi`` ve R kodu aynı hesabı yapar. statsmodels ``het_white`` aynı
    sayıyı verir, ancak yardımcı tasarımın rankını iki ayrı sayısal eşikle karşılaştıran bir ``assert`` içerir
    (statsmodels 0.14.6, ``diagnostic.py``): kukla varken en küçük tekil değerin gürültüsü iki eşiğin arasına düşerse
    makineye göre AssertionError verir. Bu yüzden kullanılmaz."""

    design = white_design(exog)
    squared = np.asarray(resid, dtype=float) ** 2
    coefficients, _, rank, _ = np.linalg.lstsq(design, squared, rcond=WHITE_RCOND)
    r2 = 1 - np.sum((squared - design @ coefficients) ** 2) / np.sum((squared - squared.mean()) ** 2)
    lm = len(squared) * r2
    return float(lm), float(stats.chi2.sf(lm, rank - 1))


def hetero_test(result, kind: str) -> tuple[float, float]:
    """Heteroskedastisite testinin LM istatistiği ve p-değeri; yardımcı regresyon modelin artıkları ve açıklayıcılarıyla
    (``model.exog``) kurulur. ``bp``: statsmodels ``het_breuschpagan`` (Koenker'in n·R² biçimi); ``white``:
    :func:`white_test` (açık yardımcı regresyon)."""

    if kind == "white":
        return white_test(result.resid, result.model.exog)
    from statsmodels.stats.diagnostic import het_breuschpagan

    values = het_breuschpagan(result.resid, result.model.exog)
    return float(values[0]), float(values[1])


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


def regression_table(op: RegressionTable, models: dict, scalars: dict | None = None) -> pd.DataFrame:
    """Makale tipi tablonun sayıları: ``terim`` ve (``standard_errors`` ise) ``terim_sh`` satırları, ek satırlar
    (``extra``; skalerlerden), ``n``, (``r2`` ise) ``r2`` ve (``adj_r2`` ise) ``adj_r2``; sütunlar modeller."""

    columns = {}
    for position, (heading, name) in enumerate(op.models):
        result = models[name]
        cells: list[float] = []
        for term in op.terms:
            present = term in result.params.index
            cells.append(float(result.params[term]) if present else np.nan)
            if op.standard_errors:
                cells.append(float(result.bse[term]) if present else np.nan)
        # Boş skaler adı: o sütunda bu satırın değeri yok (ör. ortak test yalnız karesel modelde).
        cells += [float((scalars or {})[names[position]]) if names[position] else np.nan for _, _, names in op.extra]
        cells.append(float(result.nobs))
        if op.r2:
            cells.append(float(result.rsquared))
        if op.adj_r2:
            cells.append(float(result.rsquared_adj))
        columns[heading] = cells
    rows_per_term = (lambda term: (term, f"{term}_sh")) if op.standard_errors else (lambda term: (term,))
    index = ([label for term in op.terms for label in rows_per_term(term)] + [key for key, _, _ in op.extra] + ["n"]
             + (["r2"] if op.r2 else []) + (["adj_r2"] if op.adj_r2 else []))
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
