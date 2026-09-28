"""Betimsel tabloların ve kategorik çekilişlerin hesabı (Streamlit'ten bağımsız).

Buradaki her fonksiyon, üretilen Python kodundaki satırlarla aynı işlem sırasını izler:
uygulamanın sayısı ile öğrencinin çalıştıracağı kodun sayısı bit düzeyinde aynıdır.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations, permutations, product

import numpy as np
import pandas as pd
from scipy import stats

from core.labs.spec import BOX_ROWS, CLASS_COLUMNS, COUNT_DISTRIBUTIONS, DENSITIES, TOTAL, TOTALLED_CLASS_COLUMNS


def inline_frame(columns: Sequence[str], rows: Sequence[Sequence[object]]) -> pd.DataFrame:
    """Satır demetlerinden veri çerçevesi; her sütun kendi türünde (metin veya sayı)."""

    widths = {len(row) for row in rows}
    if widths != {len(columns)}:
        raise ValueError("Her satırda sütun sayısı kadar değer olmalıdır.")
    return pd.DataFrame({name: [row[position] for row in rows] for position, name in enumerate(columns)})


def from_counts(columns: Sequence[str], rows: Sequence[Sequence[object]]) -> pd.DataFrame:
    """Sayım tablosunun her satırını ``sayi`` kez tekrarlayarak gözlem düzeyinde veri kurar."""

    counts = pd.DataFrame([tuple(row) for row in rows], columns=[*columns, "sayi"])
    if (counts["sayi"] < 0).any():
        raise ValueError("Sayılar negatif olamaz.")
    return counts.loc[counts.index.repeat(counts["sayi"]), list(columns)].reset_index(drop=True)


def outcomes(stages: Sequence[tuple[str, Sequence[object]]]) -> pd.DataFrame:
    """Aşamaların seçeneklerinin bütün bileşimleri; ilk aşama en yavaş değişir (``itertools.product``)."""

    names = [name for name, _ in stages]
    return pd.DataFrame(list(product(*[list(values) for _, values in stages])), columns=names)


def selections(items: Sequence[str], k: int, ordered: bool, columns: Sequence[str]) -> pd.DataFrame:
    """``items`` içinden ``k`` öğenin kombinasyonları (sıra önemsiz) ya da permütasyonları, sözlük sırasıyla."""

    if len(columns) != k:
        raise ValueError("Her seçim sırası için bir sütun adı gerekir.")
    chosen = permutations(items, k) if ordered else combinations(items, k)
    return pd.DataFrame(list(chosen), columns=list(columns))


def frequency_table(values: pd.Series, order: Sequence[object], *, relative: bool = True,
                    totals: bool = False) -> pd.DataFrame:
    """Frekans dağılımı: f, r = f/n ve p = 100 r; kategoriler ``order`` sırasıyla.

    ``order`` sayılardan oluşuyorsa (ör. iki zarın toplamı 2, 3, …, 12) değerler sayı olarak eşleştirilir.
    """

    numeric = all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in order)
    present = set(values.dropna()) if numeric else set(values.dropna().astype(str))
    unknown = sorted(str(value) for value in present - set(order))
    if unknown:
        raise ValueError("Sırada olmayan kategori: " + ", ".join(unknown))
    table = values.value_counts().reindex(list(order), fill_value=0).to_frame("frekans")
    table.index.name = values.name
    if relative:
        table["goreli"] = table["frekans"] / table["frekans"].sum()
        table["yuzde"] = 100 * table["goreli"]
    if totals:
        table.loc[TOTAL] = table.sum()
    return table


def crosstab(frame: pd.DataFrame, row: str, column: str, row_order: Sequence[str], column_order: Sequence[str], *,
             percent: str | None = None, margins: bool = False, weights: str | None = None) -> pd.DataFrame:
    """Çapraz tablo: sayılar, satır yüzdeleri (payda satır toplamı) veya sütun yüzdeleri (payda sütun toplamı).

    ``weights`` verilirse hücreler o sütunun toplamıdır (ör. örnek noktaların olasılıkları).
    """

    if weights is None:
        counts = pd.crosstab(frame[row], frame[column])
    else:
        if percent is not None:
            raise ValueError("Ağırlıklı çapraz tablo yalnız toplamları verir (percent=None).")
        counts = pd.crosstab(frame[row], frame[column], values=frame[weights], aggfunc="sum").fillna(0)
    counts = counts.reindex(index=list(row_order), columns=list(column_order), fill_value=0)
    if percent is None:
        table = counts
        if margins:
            table.loc[TOTAL] = table.sum()
            table[TOTAL] = table.sum(axis=1)
    elif percent == "satir":
        table = counts.div(counts.sum(axis=1), axis=0) * 100
        if margins:
            table[TOTAL] = table.sum(axis=1)
    elif percent == "sutun":
        table = counts.div(counts.sum(axis=0), axis=1) * 100
        if margins:
            table.loc[TOTAL] = table.sum()
    else:
        raise ValueError(f"Desteklenmeyen yüzde türü: {percent}")
    table.index.name = row
    table.columns.name = column
    return table


def boundary_label(value: float) -> str:
    """Sınıf sınırının yazımı: en çok 10 anlamlı basamak, ondalık virgül. R'de ``trimws(formatC(v, format = "fg",
    digits = 10, decimal.mark = ","))`` aynı metni verir."""

    return f"{value:.10g}".replace(".", ",")


def class_edges(values: pd.Series, width: float, lower: float | None = None,
                classes: int | None = None) -> np.ndarray:
    """Eşit genişlikli sınıf sınırları: a, a + h, ..., a + k·h.

    ``lower`` verilmezse a, en küçük değeri içeren h katıdır (⌊min/h⌋·h) ve k en büyük değeri kapsayan
    en küçük sınıf sayısıdır (⌊(max − a)/h⌋ + 1); böylece her gözlem bir sınıfa düşer.
    """

    if width <= 0:
        raise ValueError("Sınıf genişliği pozitif olmalıdır.")
    if lower is None:
        x = values.to_numpy(dtype=float)
        lower = np.floor(x.min() / width) * width
        classes = int(np.floor((x.max() - lower) / width)) + 1
    elif classes is None or classes < 1:
        raise ValueError("Alt sınır verildiğinde sınıf sayısı da verilmelidir.")
    return lower + width * np.arange(classes + 1)


def class_table(values: pd.Series, edges: np.ndarray, columns: Sequence[str], *, totals: bool = False,
                row_labels: str = "sinif") -> pd.DataFrame:
    """Sınıflara göre frekans dağılımı; sınıflar [alt, üst): alt sınır dahil, üst sınır hariç.

    Göreli frekans ve yüzde, sınıflara düşen gözlem sayısına (n) bölünür. Hesap sırası üretilen Python
    koduyla aynıdır (``pd.cut(..., right=False)``).
    """

    unknown = sorted(set(columns) - set(CLASS_COLUMNS))
    if unknown:
        raise ValueError("Tanınmayan sınıf tablosu sütunu: " + ", ".join(unknown))
    edges = np.asarray(edges, dtype=float)
    lower, upper = edges[:-1], edges[1:]
    counts = pd.cut(values, bins=edges, right=False).value_counts(sort=False).to_numpy()
    if row_labels == "sinif":
        labels = [f"{boundary_label(a)} ≤ x < {boundary_label(b)}" for a, b in zip(lower, upper)]
    elif row_labels == "ust":
        labels = [f"x < {boundary_label(b)}" for b in upper]
    else:
        raise ValueError(f"Desteklenmeyen satır adı türü: {row_labels}")
    table = pd.DataFrame({"alt": lower, "ust": upper}, index=pd.Index(labels, name="sinif"))
    n = counts.sum()
    computed = {"frekans": counts}
    computed["orta_nokta"] = (lower + upper) / 2
    computed["goreli"] = counts / n
    computed["yuzde"] = 100 * computed["goreli"]
    computed["kumulatif_frekans"] = np.cumsum(counts)
    computed["kumulatif_goreli"] = computed["kumulatif_frekans"] / n
    computed["kumulatif_yuzde"] = 100 * computed["kumulatif_goreli"]
    for column in CLASS_COLUMNS:
        if column in columns:
            table[column] = computed[column]
    if totals:
        summed = [column for column in TOTALLED_CLASS_COLUMNS if column in table.columns]
        table.loc[TOTAL] = table[summed].sum()
    return table


def stem_leaf(values: pd.Series) -> pd.DataFrame:
    """Gövde (onlar basamağı) ve yapraklar (birler basamağı); boş gövdeler de satır olarak yer alır."""

    x = np.sort(values.to_numpy(dtype=float))
    if np.any(x < 0) or np.any(x != np.round(x)):
        raise ValueError("Gövde–yaprak gösterimi için negatif olmayan tam sayılar gerekir.")
    whole = x.astype(int)
    stems, leaves = whole // 10, whole % 10
    rows = range(int(stems.min()), int(stems.max()) + 1)
    return pd.DataFrame(
        {
            "yapraklar": [" ".join(str(leaf) for leaf in leaves[stems == stem]) for stem in rows],
            "yaprak_sayisi": [int((stems == stem).sum()) for stem in rows],
        },
        index=pd.Index([str(stem) for stem in rows], name="govde"),
    )


def percentile_location(n: int, p: float, method: str = "ders") -> float:
    """Yüzdelik konumu: ders kuralı L_p = (p/100)(n + 1); yazılım varsayılanı 1 + (p/100)(n − 1)."""

    if method == "ders":
        return p / 100 * (n + 1)
    if method == "yazilim":
        return 1 + p / 100 * (n - 1)
    raise ValueError(f"Desteklenmeyen yüzdelik yöntemi: {method}")


def percentile(values, p: float, method: str = "ders") -> float:
    """p. yüzdelik. ``ders``: L_p konumunda doğrusal ara değer, uçlarda en küçük/en büyük gözlem
    (Hyndman–Fan tip 6). ``yazilim``: ``np.percentile`` varsayılanı (tip 7)."""

    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n == 0:
        raise ValueError("Veri seti boş olamaz.")
    if method == "yazilim":
        return float(np.percentile(x, p))
    location = percentile_location(n, p, method)
    if location <= 1:
        return float(x[0])
    if location >= n:
        return float(x[-1])
    k = int(np.floor(location))
    return float(x[k - 1] + (location - k) * (x[k] - x[k - 1]))


def box_summary(values) -> pd.Series:
    """Kutu grafiği özeti (``BOX_ROWS``): beş sayı özeti ders kuralıyla, IQR, 1,5·IQR sınırları, bıyık uçları.

    Bıyıklar sınırların içindeki en küçük ve en büyük gözleme uzanır; sınırların dışındaki gözlemler aykırı
    değer adaylarıdır. Üretilen koddaki ``kutu_ozeti`` fonksiyonuyla aynı işlem sırası.
    """

    x = np.sort(np.asarray(values, dtype=float))
    q1, medyan, q3 = percentile(x, 25), percentile(x, 50), percentile(x, 75)
    iqr = q3 - q1
    alt, ust = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    icerde = x[(x >= alt) & (x <= ust)]
    values_by_row = {
        "en_kucuk": x[0], "q1": q1, "medyan": medyan, "q3": q3, "en_buyuk": x[-1], "iqr": iqr,
        "alt_sinir": alt, "ust_sinir": ust, "alt_biyik": icerde.min(), "ust_biyik": icerde.max(),
        "aykiri_sayisi": float(((x < alt) | (x > ust)).sum()),
    }
    return pd.Series([float(values_by_row[row]) for row in BOX_ROWS], index=list(BOX_ROWS))


def outliers(values, summary: pd.Series) -> np.ndarray:
    """Q₁ − 1,5·IQR ve Q₃ + 1,5·IQR sınırlarının dışındaki gözlemler (veri sırasıyla)."""

    x = np.asarray(values, dtype=float)
    return x[(x < summary["alt_sinir"]) | (x > summary["ust_sinir"])]


def map_codes(values: pd.Series, mapping: Sequence[tuple[str, float]]) -> pd.Series:
    codes = values.map(dict(mapping))
    if codes.isna().any():
        missing = sorted(set(values[codes.isna()].astype(str)))
        raise ValueError("Kodu tanımlanmamış kategori: " + ", ".join(missing))
    return codes.astype(float)


def thresholds(probabilities: Sequence[float]) -> np.ndarray:
    """Birikimli olasılıklar; sonuncusu yuvarlama hatasına karşı tam 1 yapılır."""

    values = np.asarray(probabilities, dtype=float)
    if np.any(values < 0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("Olasılıklar negatif olmamalı ve toplamı 1 olmalıdır.")
    edges = np.cumsum(values)
    edges[-1] = 1.0
    return edges


def draw_categories(frame: pd.DataFrame, u: np.ndarray, categories: Sequence[str],
                    probabilities, by: Sequence[str]) -> np.ndarray:
    """u ~ Tek-düze(0, 1) değerlerinden kategoriler: birikimli olasılığı u'yu ilk aşan kategori."""

    labels = np.asarray(categories, dtype=object)
    result = np.full(len(frame), None, dtype=object)
    for condition, probs in probabilities:
        if len(probs) != len(categories):
            raise ValueError("Her kategori için bir olasılık gerekir.")
        selected = np.ones(len(frame), dtype=bool)
        for column, value in zip(by, condition):
            selected &= (frame[column] == value).to_numpy()
        result[selected] = labels[np.searchsorted(thresholds(probs), u[selected], side="right")]
    if any(value is None for value in result):
        raise ValueError("Bazı gözlemler için olasılık tanımlanmamış.")
    return result


def draw_discrete(u: np.ndarray, values: Sequence[float], probabilities: Sequence[float]) -> np.ndarray:
    """Ters dağılım fonksiyonu: X, birikimli olasılığı F(x) u'yu ilk aşan değerdir (``np.searchsorted``, sağ)."""

    if len(values) != len(probabilities):
        raise ValueError("Her değer için bir olasılık gerekir.")
    return np.asarray(values, dtype=float)[np.searchsorted(thresholds(probabilities), u, side="right")]


def tree_layout(frame: pd.DataFrame, first: str, second: str, first_p: str, second_p: str) -> pd.DataFrame:
    """İki aşamalı olasılık ağacının yolları ve çizim konumları (soldan sağa: kök 0, ilk aşama 1, yollar 2).

    Her satır bir tam yoldur; ilk yol en üstte (``y_yol = n − 1, …, 0``). İlk aşamadaki bir dal, kendi yollarının
    ortasında durur (``y_ilk``). ``ortak`` yolun ortak olasılığıdır: ilk dalın olasılığı × ikinci dalın koşullu
    olasılığı. Üretilen koddaki ağaç çizimiyle aynı kural.
    """

    paths = pd.DataFrame({
        "ilk": frame[first].astype(str).to_numpy(),
        "ikinci": frame[second].astype(str).to_numpy(),
        "p_ilk": frame[first_p].to_numpy(dtype=float),
        "p_ikinci": frame[second_p].to_numpy(dtype=float),
    })
    for name, group in paths.groupby("ilk", sort=False):
        if group["p_ilk"].nunique() != 1:
            raise ValueError(f"'{name}' dalının olasılığı her yolda aynı olmalıdır.")
    paths["ortak"] = paths["p_ilk"] * paths["p_ikinci"]
    paths["y_yol"] = np.arange(len(paths))[::-1].astype(float)
    paths["y_ilk"] = paths.groupby("ilk", sort=False)["y_yol"].transform("mean")
    return paths


def support(lower: int, upper: int) -> np.ndarray:
    """Kesikli değişkenin olası değerleri lower, …, upper (``np.arange(lower, upper + 1)``)."""

    if int(lower) != lower or int(upper) != upper or upper < lower:
        raise ValueError("Olası değerler lower ≤ upper olan tam sayılardır.")
    return np.arange(int(lower), int(upper) + 1)


def rectangle_midpoints(lower: float, width: float, count: int) -> np.ndarray:
    """Dikdörtgenlerin orta noktaları lower + width·(i − 0,5), i = 1, …, count (üretilen kodla aynı işlem sırası)."""

    return lower + width * (np.arange(1, count + 1) - 0.5)


def density(distribution: str, first: float, second: float, x) -> np.ndarray:
    """Yoğunluk f(x): normal (μ = first, σ = second), tek-düze U(a = first, b = second), üstel (ortalama süre
    μ = first, σ = second = μ) ya da gamma (biçim k = first, oran r = second)."""

    if distribution not in DENSITIES:
        raise ValueError(f"Desteklenmeyen yoğunluk: {distribution}")
    if distribution == "normal":
        if second <= 0:
            raise ValueError("Standart sapma pozitif olmalıdır.")
        return stats.norm.pdf(x, first, second)
    if distribution == "exponential":
        if first <= 0 or second != first:
            raise ValueError("Üstel dağılımda ortalama süre μ pozitiftir ve σ = μ'dür.")
        return stats.expon.pdf(x, scale=first)
    if distribution == "gamma":
        if first <= 0 or second <= 0:
            raise ValueError("Gamma dağılımında biçim ve oran pozitif olmalıdır.")
        return stats.gamma.pdf(x, first, scale=1 / second)
    if second <= first:
        raise ValueError("Tek-düze dağılımda b > a olmalıdır.")
    return stats.uniform.pdf(x, first, second - first)


def density_grid(distribution: str, first: float, second: float, x_range: tuple[float, float],
                 points: int = 401) -> pd.DataFrame:
    """Yoğunluk grafiğinin eğrisi: yatay eksende eşit aralıklı ``points`` nokta ve f(x)."""

    low, high = x_range
    if not low < high:
        raise ValueError("Yatay eksenin alt sınırı üst sınırından küçük olmalıdır.")
    x = np.linspace(low, high, points)
    return pd.DataFrame({"x": x, "f": density(distribution, first, second, x)})


def draw_count(rng: np.random.Generator, distribution: str, parameters: Sequence[float], size: int) -> np.ndarray:
    """Sayım çekilişi, üretilen Python koduyla aynı çağrı: ``rng.binomial``, ``rng.poisson`` ya da
    ``rng.hypergeometric``."""

    if COUNT_DISTRIBUTIONS.get(distribution) != len(parameters):
        raise ValueError(f"{distribution}: parametre sayısı uygun değil.")
    if distribution == "binomial":
        n, p = parameters
        return rng.binomial(int(n), p, size=size).astype(float)
    if distribution == "poisson":
        return rng.poisson(parameters[0], size=size).astype(float)
    population, successes, draws = (int(value) for value in parameters)
    if not 0 <= successes <= population or not 0 <= draws <= population:
        raise ValueError("Hipergeometrik: 0 ≤ r ≤ N ve 0 ≤ n ≤ N olmalıdır.")
    return rng.hypergeometric(successes, population - successes, draws, size=size).astype(float)
