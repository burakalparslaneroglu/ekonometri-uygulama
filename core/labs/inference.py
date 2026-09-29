"""Konu 7–8 çıkarım grafiklerinin ortak hesabı: test dağılımının ekseni, kritik değerler, reddetme bölgesi ve
p-değeri alanı (``HypothesisPlot``), katsayı grafiği (``CoefficientPlot``) ve tekrarlı örneklemedeki güven
aralıkları (``IntervalPlot``).

Uygulama ve üretilen Python ile R kodu aynı kuralları kullanır: t testinde yatay eksen [−h, h],
h = max(4, min(|t| + 1, 6)); F testinde [0, üst], üst = max(2,4·kritik, min(1,15·F, 6·kritik)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from core.labs.spec import CoefficientPlot, HypothesisPlot, IntervalPlot


def critical_values(distribution: str, df: float, df2: float | None, alpha: float, alternative: str) -> tuple[float, ...]:
    """Kritik değer(ler): t'de iki taraflı ±t_{α/2}, sağ kuyrukta t_α, sol kuyrukta −t_α; F'de F_α (üst kuyruk)."""

    if distribution == "f":
        return (float(stats.f.ppf(1 - alpha, df, df2)),)
    if alternative == "iki":
        value = float(stats.t.ppf(1 - alpha / 2, df))
        return (-value, value)
    if alternative == "sag":
        return (float(stats.t.ppf(1 - alpha, df)),)
    return (float(stats.t.ppf(alpha, df)),)


def axis_range(distribution: str, statistic: float, critical: tuple[float, ...]) -> tuple[float, float]:
    """Yatay eksen: gözlenen istatistik makul uzaklıktaysa eksene girer; çok uzaksa eksen sınırlı kalır."""

    if distribution == "f":
        upper = max(2.4 * critical[0], min(1.15 * statistic, 6 * critical[0]))
        return 0.0, float(upper)
    half = max(4.0, min(abs(statistic) + 1.0, 6.0))
    return -half, half


def density(distribution: str, x: np.ndarray, df: float, df2: float | None) -> np.ndarray:
    return stats.f.pdf(x, df, df2) if distribution == "f" else stats.t.pdf(x, df)


def p_value(distribution: str, statistic: float, df: float, df2: float | None, alternative: str) -> float:
    """Gözlenen istatistiğin p-değeri: F'de ve sağ kuyrukta üst kuyruk, solda alt kuyruk, iki taraflıda 2·P(T > |t|)."""

    if distribution == "f":
        return float(stats.f.sf(statistic, df, df2))
    if alternative == "iki":
        return float(2 * stats.t.sf(abs(statistic), df))
    if alternative == "sag":
        return float(stats.t.sf(statistic, df))
    return float(stats.t.cdf(statistic, df))


def rejection_regions(distribution: str, alternative: str, critical: tuple[float, ...],
                      bounds: tuple[float, float]) -> list[tuple[float, float]]:
    low, high = bounds
    if distribution == "f" or alternative == "sag":
        return [(critical[-1], high)]
    if alternative == "sol":
        return [(low, critical[0])]
    return [(low, critical[0]), (critical[1], high)]


def p_regions(distribution: str, alternative: str, statistic: float,
              bounds: tuple[float, float]) -> list[tuple[float, float]]:
    """p-değeri alanı: gözlenen istatistiğin ötesindeki kuyruk(lar), eksenin içinde kalan kısmıyla."""

    low, high = bounds
    if distribution == "f" or alternative == "sag":
        start = max(statistic, low)
        return [(start, high)] if start < high else []
    if alternative == "sol":
        end = min(statistic, high)
        return [(low, end)] if end > low else []
    size = abs(statistic)
    return [(low, -size), (size, high)] if size < high else []


def hypothesis_layout(op: HypothesisPlot, statistic: float, df: float, df2: float | None) -> dict:
    """Grafiğin bütün parçaları: eğri, kritik değerler, reddetme ve p alanları, eksen, p-değeri."""

    critical = critical_values(op.distribution, df, df2, op.alpha, op.alternative)
    bounds = axis_range(op.distribution, statistic, critical)
    x = np.linspace(bounds[0], bounds[1], 801)
    return {
        "x": x,
        "f": density(op.distribution, x, df, df2),
        "kritik": critical,
        "reddetme": rejection_regions(op.distribution, op.alternative, critical, bounds),
        "p_alani": p_regions(op.distribution, op.alternative, statistic, bounds),
        "sinirlar": bounds,
        "gozlenen": float(statistic),
        "disarida": not bounds[0] <= statistic <= bounds[1],
        "p": p_value(op.distribution, statistic, df, df2, op.alternative),
        "sd": (df, df2),
    }


def coefficient_intervals(op: CoefficientPlot, result) -> pd.DataFrame:
    """Katsayı grafiğinin verisi: terim, tahmin ve güven aralığının sınırları (``terms`` sırasıyla)."""

    interval = result.conf_int(alpha=round(1 - op.level, 10))
    terms = list(op.terms)
    return pd.DataFrame({
        "terim": terms,
        "tahmin": result.params[terms].to_numpy(dtype=float),
        "alt": interval.loc[terms, 0].to_numpy(dtype=float),
        "ust": interval.loc[terms, 1].to_numpy(dtype=float),
    })


def first_intervals(op: IntervalPlot, table: pd.DataFrame, truth: float) -> pd.DataFrame:
    """İlk ``rows`` tekrarın aralığı ve gerçek değeri kapsayıp kapsamadığı (alt ≤ gerçek ≤ üst)."""

    first = table.head(op.rows)
    low = first[op.low].to_numpy(dtype=float)
    high = first[op.high].to_numpy(dtype=float)
    return pd.DataFrame({
        "tekrar": np.arange(1, len(first) + 1),
        "tahmin": first[op.estimate].to_numpy(dtype=float),
        "alt": low,
        "ust": high,
        "kapsiyor": (low <= truth) & (truth <= high),
    })
