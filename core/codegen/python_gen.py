"""Uygulama tanımından Python (pandas + matplotlib) kodu üretir.

Üretilen kod, uygulamanın hesabıyla aynı işlem sırasını izler; sayılar bit düzeyinde aynıdır.
"""

from __future__ import annotations

import re

from core.codegen.base import (
    HEAT_LOW,
    PALETTE,
    REFERENCE_COLORS,
    STAT_NAMES,
    Generator,
    flatten,
    functions_used,
    listing,
    text,
    uses_charts,
    wrapped,
)
from core.labs import expr as E
from core.labs.spec import (
    TOTAL,
    BarChart,
    BoxPlot,
    BoxSummary,
    OLS,
    CellTarget,
    Check,
    CoefTarget,
    Describe,
    GroupStats,
    LoadWooldridge,
    ModelTarget,
    ModelValue,
    PanelSummary,
    RegressionTable,
    Residuals,
    ShowModel,
    SortRows,
    ClassHistogram,
    ClassTable,
    CompareBarChart,
    CopyFrame,
    Count,
    CrossTab,
    DensityCompare,
    DensityPlot,
    Derive,
    DotPlot,
    Draw,
    DrawCategory,
    DrawCount,
    DrawDiscrete,
    Event,
    FrequencyTable,
    FromCounts,
    GroupedBarChart,
    Groups,
    GroupSummary,
    HeatMap,
    Histogram,
    InlineData,
    JoinColumns,
    LineChart,
    MapCodes,
    MonteCarlo,
    MosaicChart,
    NewSample,
    Operation,
    Outcomes,
    PairStatistic,
    Percentile,
    PieChart,
    PmfWithDensity,
    Rectangles,
    RowSum,
    Scalar,
    ScalarTable,
    ScalarTarget,
    ScatterPlot,
    Selections,
    Shape,
    ShowFrame,
    Statistic,
    StatTarget,
    StemLeaf,
    SummaryTable,
    Support,
    TableTarget,
    TreeDiagram,
    VariableTypes,
)
from core.labs.tables import class_edges

_STAT = {
    "count": "count()", "sum": "sum()", "mean": "mean()", "median": "median()", "mode": "mode().item()",
    "mode_freq": "value_counts().max()", "prod": "prod()", "min": "min()", "max": "max()", "var": "var()",
    "std": "std()", "nunique": "nunique()", "value": "item()",
}
"""İstatistiğin pandas karşılığı; ``mode().item()`` birden fazla mod varsa hata verir (tek mod beklenir).
``var()`` ve ``std()`` pandas'ta örneklem ölçüleridir (payda n − 1)."""
_SCIPY_FUNCTIONS = {"normcdf", "normpdf", "norminv", *E.DISTRIBUTION_FUNCTIONS}
_MATH_FUNCTIONS = set(E.COUNTING_FUNCTIONS)
_FUNCTIONS = {
    "log": "np.log", "exp": "np.exp", "sqrt": "np.sqrt", "abs": "np.abs", "maximum": "np.maximum",
    "minimum": "np.minimum", "round": "np.rint", "roundto": "np.round", "floor": "np.floor", "cumprod": "np.cumprod",
    "cummean": "np.cumsum({0}) / np.arange(1, len({0}) + 1)", "seq": "np.arange(1, len({0}) + 1)",
    "factorial": "math.factorial(int({0}))", "comb": "math.comb(int({0}), int({1}))",
    "perm": "math.perm(int({0}), int({1}))",
    "normcdf": "stats.norm.cdf", "normpdf": "stats.norm.pdf", "norminv": "stats.norm.ppf",
    "dbinom": "stats.binom.pmf", "pbinom": "stats.binom.cdf", "dpois": "stats.poisson.pmf",
    "ppois": "stats.poisson.cdf", "dhyper": "stats.hypergeom.pmf", "phyper": "stats.hypergeom.cdf",
    "dnorm": "stats.norm.pdf",
    **{name: f"np.where({{0}} {symbol} {{1}}, 1.0, 0.0)" for name, symbol in E.COMPARISONS.items()},
}

_NUMBER_TEXT = [
    "def sayi_metni(deger, basamak=0, yuzde=False):",
    '    """Grafik etiketleri için Türkçe sayı: ondalık virgül, yüzde işareti sayıdan önce."""',
    '    metin = f"{deger:.{basamak}f}".replace(".", ",")',
    '    return "%" + metin if yuzde else metin',
]
_BOUNDARY_TEXT = [
    "def sinir_metni(deger):",
    '    """Sınıf sınırının yazımı: ondalık virgül (ör. 12,5)."""',
    '    return f"{deger:.10g}".replace(".", ",")',
]
_PERCENTILE = [
    "def yuzdelik(degerler, p):",
    '    """Ders kuralı: L_p = (p/100)(n + 1). L_p tam sayı değilse komşu iki gözlem arasında doğrusal ara',
    "    değer; L_p ≤ 1 ise en küçük, L_p ≥ n ise en büyük gözlem. np.percentile(degerler, p,",
    '    method="weibull") aynı sonucu verir; np.percentile varsayılanı farklı bir kural kullanır."""',
    "    x = np.sort(np.asarray(degerler, dtype=float))",
    "    n = len(x)",
    "    konum = p / 100 * (n + 1)",
    "    if konum <= 1:",
    "        return x[0]",
    "    if konum >= n:",
    "        return x[-1]",
    "    k = int(np.floor(konum))",
    "    return x[k - 1] + (konum - k) * (x[k] - x[k - 1])",
]


_BOX_SUMMARY = [
    "def kutu_ozeti(degerler):",
    '    """Kutu grafiği özeti: çeyrekler ders kuralıyla (yuzdelik); bıyıklar Q1 − 1,5·IQR ile Q3 + 1,5·IQR',
    '    sınırlarının içindeki en uç gözlemlere uzanır; sınırların dışındakiler aykırı değer adayıdır."""',
    "    x = np.sort(np.asarray(degerler, dtype=float))",
    "    q1, medyan, q3 = yuzdelik(x, 25), yuzdelik(x, 50), yuzdelik(x, 75)",
    "    iqr = q3 - q1",
    "    alt, ust = q1 - 1.5 * iqr, q3 + 1.5 * iqr",
    "    icerde = x[(x >= alt) & (x <= ust)]",
    "    return pd.Series({",
    '        "en_kucuk": x[0], "q1": q1, "medyan": medyan, "q3": q3, "en_buyuk": x[-1], "iqr": iqr,',
    '        "alt_sinir": alt, "ust_sinir": ust, "alt_biyik": icerde.min(), "ust_biyik": icerde.max(),',
    '        "aykiri_sayisi": float(((x < alt) | (x > ust)).sum()),',
    "    })",
]


def _render(expression: E.Expr, dialect: E.Dialect) -> str:
    """İfadenin Python yazımı; tam sayı değişmezinin etrafındaki gereksiz ``int()`` atılır (``math.comb(5, 2)``)."""

    return re.sub(r"\bint\((\d+)\)", r"\1", E.render(expression, dialect))


def _list(values) -> str:
    return "[" + ", ".join(text(value) for value in values) + "]"


def _quote(value: str) -> str:
    """Çift tırnaklı dizge içinde düz metin."""

    return value.replace("\\", "\\\\").replace('"', '\\"')


def _fstring(value: str) -> str:
    """f-dizgesi içinde düz metin: tırnak, ters bölü ve süslü parantez kaçırılır."""

    return _quote(value).replace("{", "{{").replace("}", "}}")


def _needs_numpy(operations) -> bool:
    """numpy yalnız rastgele çekiliş, grup etiketi, histogram, yüzdelik, veriden sınıf sınırı veya ifade
    fonksiyonu varsa gerekir."""

    numpy_ops = (Groups, NewSample, Draw, DrawCategory, DrawDiscrete, DrawCount, Histogram, MonteCarlo, Percentile,
                 BoxSummary, BoxPlot, TreeDiagram, Support, Rectangles, DensityPlot, DensityCompare, PmfWithDensity)
    for op in flatten(operations):
        if isinstance(op, numpy_ops) or (isinstance(op, ClassTable) and op.lower is None):
            return True
        if isinstance(op, ScatterPlot) and (op.fit_line or op.lines):
            return True
    return bool(functions_used(operations) - _SCIPY_FUNCTIONS - _MATH_FUNCTIONS)


def _needs_percentile(operations) -> bool:
    """Ders kuralıyla yüzdelik fonksiyonu: yüzdelik işlemi ya da kutu grafiği özeti varsa."""

    for op in flatten(operations):
        if (isinstance(op, Percentile) and op.method == "ders") or isinstance(op, (BoxSummary, BoxPlot)):
            return True
    return False


def _stat_call(source: str, stat: str) -> str:
    return f"{source}.{_STAT[stat]}"


def _labelled_charts(operations) -> bool:
    """Değer etiketi yazan grafikler (Türkçe sayı yardımcısı gerekir)."""

    for op in flatten(operations):
        if isinstance(op, (BarChart, CompareBarChart, MosaicChart, TreeDiagram, HeatMap)):
            return True
        if isinstance(op, GroupedBarChart) and op.labels:
            return True
        if isinstance(op, ClassHistogram) and op.labels:
            return True
    return False


def _uses_density(op: Operation) -> bool:
    """Yoğunluk eğrisi çizen işlemler scipy.stats ister."""

    return isinstance(op, (DensityPlot, DensityCompare, PmfWithDensity)) or (isinstance(op, Histogram) and op.curves)


def _parameter(value) -> str:
    """Grafik parametresi: sayı ya da önceden hesaplanmış skalerin (aynı adlı değişken) adı."""

    return value if isinstance(value, str) else E.format_number(value)


def density_expression(distribution: str, first, second, x: str) -> str:
    """Yoğunluk f(x)'in scipy yazımı; ``first`` ve ``second`` sayı ya da değişken adıdır."""

    a, b = _parameter(first), _parameter(second)
    if distribution == "normal":
        return f"stats.norm.pdf({x}, {a}, {b})"
    if distribution == "exponential":
        return f"stats.expon.pdf({x}, scale={a})"
    if distribution == "gamma":
        return f"stats.gamma.pdf({x}, {a}, scale=1 / {b})"
    if isinstance(first, str) or isinstance(second, str):
        return f"stats.uniform.pdf({x}, {a}, {b} - {a})"
    return f"stats.uniform.pdf({x}, {a}, {E.format_number(second - first)})"


_COEF = {
    "coef": '{m}.params["{t}"]', "se": '{m}.bse["{t}"]', "t": '{m}.tvalues["{t}"]', "p": '{m}.pvalues["{t}"]',
    "ci_low": '{m}.conf_int().loc["{t}", 0]', "ci_high": '{m}.conf_int().loc["{t}", 1]',
}
"""Katsayı niceliklerinin statsmodels yazımı (t ve p: n − k serbestlik dereceli t dağılımı; GA: %95)."""
_MODEL = {
    "r2": "{m}.rsquared", "adj_r2": "{m}.rsquared_adj", "nobs": "{m}.nobs", "f": "{m}.fvalue",
    "f_p": "{m}.f_pvalue", "ssr": "{m}.ssr", "df_resid": "{m}.df_resid",
}
_REGRESSION_TABLE = [
    "def makale_tablosu(modeller, terimler):",
    '    """Makale tipi tablo: her terim için katsayı ve standart hata (terim_sh), sonra n ve R²."""',
    "    sutunlar = {}",
    "    for baslik, m in modeller.items():",
    "        hucreler = []",
    "        for terim in terimler:",
    "            hucreler += [m.params.get(terim, np.nan), m.bse.get(terim, np.nan)]",
    "        sutunlar[baslik] = hucreler + [m.nobs, m.rsquared]",
    '    satirlar = [ad for terim in terimler for ad in (terim, terim + "_sh")] + ["n", "r2"]',
    "    return pd.DataFrame(sutunlar, index=satirlar, dtype=float)",
]
def _read_fields_note(op: ShowModel, topic: int, stats: tuple[str, ...]) -> list[str]:
    """Çıkarım henüz işlenmediyse (Konu 7'den önce ya da ``columns == ("coef",)``) tam çıktının hangi alanlarının
    okunduğu; ``stats`` adımda okunan model bilgisidir."""

    if "se" in op.columns and topic >= 7:
        return []
    read = listing(("katsayılar (coef)", *(STAT_NAMES[stat] for stat in stats)))
    return [f"# Tam çıktı yazdırılır; bu adımda yalnız {read} okunur.",
            "# std err, t, P>|t| ve güven aralığı Konu 7'de, F istatistiği Konu 8'de yorumlanır."]


def _number(name: str, decimals: int) -> str:
    """f-string içinde sayının biçimi. Çok basamaklı gösterimde (≥ 10) yuvarlanınca sıfır olan küçük negatif sayı
    "-0.000…" yazılmasın diye önce yuvarlanır ve 0.0 eklenir (−0 + 0 = 0)."""

    if decimals >= 10:
        return f"{{round({name}, {decimals}) + 0.0:.{decimals}f}}"
    return f"{{{name}:.{decimals}f}}"


_REGRESSION_TABLE_OPTIONS = [
    "def makale_tablosu(modeller, terimler, sh=True, r2=True):",
    '    """Makale tipi tablo: her terim için katsayı ve (sh=True ise) standart hata (terim_sh), sonra n ve',
    '    (r2=True ise) R²."""',
    "    sutunlar = {}",
    "    for baslik, m in modeller.items():",
    "        hucreler = []",
    "        for terim in terimler:",
    "            hucreler.append(m.params.get(terim, np.nan))",
    "            if sh:",
    "                hucreler.append(m.bse.get(terim, np.nan))",
    "        hucreler.append(m.nobs)",
    "        if r2:",
    "            hucreler.append(m.rsquared)",
    "        sutunlar[baslik] = hucreler",
    '    satirlar = [ad for terim in terimler for ad in ((terim, terim + "_sh") if sh else (terim,))]',
    '    satirlar += ["n", "r2"] if r2 else ["n"]',
    "    return pd.DataFrame(sutunlar, index=satirlar, dtype=float)",
]
_REGRESSION_TABLE_ADJUSTED = [
    "def makale_tablosu(modeller, terimler, sh=True, r2=True, r2_duz=False):",
    '    """Makale tipi tablo: her terim için katsayı ve (sh=True ise) standart hata (terim_sh), sonra n,',
    '    (r2=True ise) R² ve (r2_duz=True ise) düzeltilmiş R² (adj_r2)."""',
    "    sutunlar = {}",
    "    for baslik, m in modeller.items():",
    "        hucreler = []",
    "        for terim in terimler:",
    "            hucreler.append(m.params.get(terim, np.nan))",
    "            if sh:",
    "                hucreler.append(m.bse.get(terim, np.nan))",
    "        hucreler.append(m.nobs)",
    "        if r2:",
    "            hucreler.append(m.rsquared)",
    "        if r2_duz:",
    "            hucreler.append(m.rsquared_adj)",
    "        sutunlar[baslik] = hucreler",
    '    satirlar = [ad for terim in terimler for ad in ((terim, terim + "_sh") if sh else (terim,))]',
    '    satirlar += ["n", "r2"] if r2 else ["n"]',
    '    satirlar += ["adj_r2"] if r2_duz else []',
    "    return pd.DataFrame(sutunlar, index=satirlar, dtype=float)",
]
_STARS = [
    "def yildiz(p):",
    '    """Katsayının yanındaki yıldız: *** p < 0,01; ** p < 0,05; * p < 0,10."""',
    '    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""',
]


def coef_expression(model: str, term: str, quantity: str) -> str:
    return _COEF[quantity].format(m=model, t=term)


def model_expression(model: str, quantity: str) -> str:
    return _MODEL[quantity].format(m=model)


def _where(frame: str, where) -> str:
    column, value = where
    return f'{frame}["{column}"] == {text(value)}'


class PythonGenerator(Generator):
    language = "Python"
    comment = "#"

    def dialect(self, frame: str) -> E.Dialect:
        return E.Dialect(variable=lambda name: f'{frame}["{name}"]', functions=_FUNCTIONS, power="**")

    # --- Başlık ve yardımcılar ------------------------------------------
    def imports(self, operations: tuple[Operation, ...], *, script: bool = False) -> list[str]:
        flat = flatten(operations)
        standard = ["import sys"] if script else []
        if functions_used(operations) & _MATH_FUNCTIONS:
            standard.append("import math")
        tools = {"product" for op in flat if isinstance(op, Outcomes) and len(op.stages) > 1}
        tools |= {"permutations" if op.ordered else "combinations" for op in flat if isinstance(op, Selections)}
        tools = sorted(tools)
        if tools:
            standard.append(f"from itertools import {', '.join(tools)}")
        lines = standard + [""] if standard else []
        if uses_charts(operations):
            lines.append("import matplotlib.pyplot as plt")
        if any(isinstance(op, HeatMap) for op in flat):
            lines.append("from matplotlib.colors import LinearSegmentedColormap")
        if _needs_numpy(operations) or any(isinstance(op, RegressionTable) for op in flat):
            lines.append("import numpy as np")
        lines.append("import pandas as pd")
        if any(isinstance(op, OLS) for op in flat):
            lines.append("import statsmodels.formula.api as smf")
        if any(isinstance(op, LoadWooldridge) for op in flat):
            lines.append("import wooldridge as wd  # Wooldridge (2020) veri setleri: pip install wooldridge")
        if functions_used(operations) & _SCIPY_FUNCTIONS or any(_uses_density(op) for op in flat):
            lines.append("from scipy import stats")
        lines.append("")
        return lines

    def output_setup(self) -> list[str]:
        return [
            "# Windows'ta çıktı bir dosyaya ya da başka bir programa yönlendirildiğinde Python yerel kod",
            "# sayfasını (ör. cp1254) kullanır ve bazı karakterleri yazamaz; çıktı UTF-8 olsun.",
            'if hasattr(sys.stdout, "reconfigure"):',
            '    sys.stdout.reconfigure(encoding="utf-8")',
            "",
        ]

    def helpers(self, operations: tuple[Operation, ...], *, with_checks: bool) -> list[str]:
        lines: list[str] = [""]  # üst düzey fonksiyonlardan önce iki boş satır (PEP 8)
        flat = flatten(operations)
        if _labelled_charts(operations):
            lines += _NUMBER_TEXT + ["", ""]
        if any(isinstance(op, ClassTable) for op in flat):
            lines += _BOUNDARY_TEXT + ["", ""]
        if _needs_percentile(operations):
            lines += _PERCENTILE + ["", ""]
        if any(isinstance(op, (BoxSummary, BoxPlot)) for op in flat):
            lines += _BOX_SUMMARY + ["", ""]
        if any(isinstance(op, RegressionTable) for op in flat):
            tables = [op for op in flat if isinstance(op, RegressionTable)]
            if any(op.adj_r2 for op in tables):
                helper = _REGRESSION_TABLE_ADJUSTED
            elif any(not (op.standard_errors and op.r2) for op in tables):
                helper = _REGRESSION_TABLE_OPTIONS
            else:
                helper = _REGRESSION_TABLE
            lines += helper + ["", ""]
            if any(isinstance(op, RegressionTable) and op.stars for op in flat):
                lines += _STARS + ["", ""]
        if with_checks:
            lines += [
                "def kontrol_et(etiket, deger, beklenen, ondalik=4):",
                '    """Hesaplanan değeri ders notlarındaki basılı değerle karşılaştırır."""',
                "    tolerans = 0.5 * 10 ** (-ondalik) + 1e-12",
                '    durum = "OK  " if abs(deger - beklenen) <= tolerans else "HATA"',
                '    print(f"  {durum} {etiket}: {deger:.{ondalik}f}  (notlar: {beklenen:.{ondalik}f})")',
                '    assert abs(deger - beklenen) <= tolerans, f"{etiket} notlarla uyuşmuyor."',
                "",
                "",
            ]
        return lines if len(lines) > 1 else []

    # --- İşlemler --------------------------------------------------------
    def operation(self, op: Operation) -> list[str]:
        lines = self._operation(op)
        if self.quiet:
            lines = [line for line in lines if not line.lstrip().startswith("print(")]
        return lines

    def _operation(self, op: Operation) -> list[str]:  # noqa: C901 - tek dağıtıcı, işlem türü başına bir blok
        if isinstance(op, LoadWooldridge):
            lines = [f"# {op.comment}", f'{op.frame} = wd.data("{op.dataset}")']
            if op.columns:
                lines.append(f"{op.frame} = {op.frame}[{_list(op.columns)}]")
            return lines
        if isinstance(op, SortRows):
            return [
                f"# {op.comment}",
                f'{op.frame} = {op.frame}.sort_values("{op.by}", kind="stable").reset_index(drop=True)',
            ]
        if isinstance(op, Describe):
            return [
                f"# {op.comment}",
                f"{op.result} = {op.frame}[{_list(op.variables)}].describe().T[{_list(op.stats)}]",
                f"print({op.result}.round({op.decimals}))",
            ]
        if isinstance(op, GroupStats):
            by = f'"{op.by[0]}"' if len(op.by) == 1 else _list(op.by)
            return [
                f"# {op.comment}",
                f'{op.result} = {op.frame}.groupby({by})["{op.variable}"].agg({_list(op.stats)})',
                f"print({op.result}.round({op.decimals}))",
            ]
        if isinstance(op, PanelSummary):
            periods = f"{op.result}_donem"
            return [
                f"# {op.comment}",
                f"# Her birimin ({op.unit}) kaç farklı dönemde ({op.time}) gözlendiği",
                f'{periods} = {op.frame}.groupby("{op.unit}")["{op.time}"].nunique()',
                f"print({periods}.describe())",
                f"{op.result} = pd.DataFrame({{\"deger\": {{",
                f'    "gozlem": len({op.frame}),',
                f'    "birim": {op.frame}["{op.unit}"].nunique(),',
                f'    "ilk_donem": {op.frame}["{op.time}"].min(),',
                f'    "son_donem": {op.frame}["{op.time}"].max(),',
                f'    "en_az_donem": {periods}.min(),',
                f'    "en_cok_donem": {periods}.max(),',
                "}}).astype(float)",
                f"print({op.result})",
            ]
        if isinstance(op, OLS):
            return [f"# {op.comment}", f'{op.name} = smf.ols("{op.formula}", data={op.frame}).fit()']
        if isinstance(op, Residuals):
            return [f"# {op.comment}", f'{op.frame}["{op.name}"] = {op.model}.resid']
        if isinstance(op, ShowModel):
            topic = int(self.spec.topic_key[-2:])
            return [f"# {op.comment}", *_read_fields_note(op, topic, self.fields_read(op)),
                    f"print({op.model}.summary())"]
        if isinstance(op, ModelValue):
            value = (coef_expression(op.model, op.term, op.quantity) if op.term is not None
                     else model_expression(op.model, op.quantity))
            return [
                f"# {op.comment}",
                f"{op.name} = {value}",
                f'print(f"{_fstring(op.comment)}: {_number(op.name, op.decimals)}")',
            ]
        if isinstance(op, RegressionTable):
            models = ", ".join(f"{text(heading)}: {name}" for heading, name in op.models)
            flags = (("" if op.standard_errors else ", sh=False") + ("" if op.r2 else ", r2=False")
                     + (", r2_duz=True" if op.adj_r2 else ""))
            lines = [
                f"# {op.comment}",
                f"{op.result} = makale_tablosu({{{models}}}, {_list(op.terms)}{flags})",
                f"print({op.result}.round({op.decimals}))",
            ]
            for digits in sorted({digits for _, digits in op.term_decimals}):
                terms = [term for term, value in op.term_decimals if value == digits]
                lines.append(f"print({op.result}.loc[{_list(terms)}].round({digits}))  # notlardaki gibi {digits} basamak")
            if op.stars:
                for heading, name in op.models:
                    lines.append(
                        f'print({text(heading)}, {{terim: yildiz({name}.pvalues[terim]) for terim in {_list(op.terms)} '
                        f"if terim in {name}.pvalues}})"
                    )
            return lines
        if isinstance(op, InlineData):
            return self._inline(op)
        if isinstance(op, FromCounts):
            return self._from_counts(op)
        if isinstance(op, Outcomes):
            return self._outcomes(op)
        if isinstance(op, Selections):
            return self._selections(op)
        if isinstance(op, VariableTypes):
            rows = [f'    ({text(v)}, {text(k)}, {text(d)}),' for v, k, d in op.rows]
            return [
                "# Yazılımın saklama türü: sayı (int64/float64) ya da metin (object)",
                f"print({op.frame}.dtypes)",
                "# İstatistiksel tür yazılımdan değil, değişkenin anlamından gelir",
                f"{op.result} = pd.DataFrame(",
                "    [",
                *[f"    {row}" for row in rows],
                "    ],",
                '    columns=["degisken", "tur", "ayrinti"],',
                ').set_index("degisken")',
                f"print({op.result})",
            ]
        if isinstance(op, Event):
            members = ", ".join(str(value) for value in op.values)
            return [
                f"# {op.comment}: {op.column} ∈ {{{members}}} olan satırlar 1, diğerleri 0",
                f'{op.frame}["{op.name}"] = {op.frame}["{op.column}"].isin({_list(op.values)}).astype(int)',
            ]
        if isinstance(op, ShowFrame):
            columns = _list(op.columns)
            if op.head:
                return [f"# {op.comment}", f"print({op.frame}[{columns}].head({op.head}))"]
            if op.rows:
                positions = ", ".join(str(row - 1) for row in op.rows)
                shown = ", ".join(str(row) for row in op.rows)
                return [
                    f"# {op.comment}",
                    f"# Gözlem numaraları {shown}; Python satırları 0'dan saydığı için sıra numarası bir eksiktir",
                    f"print({op.frame}.iloc[[{positions}]][{columns}])",
                ]
            if op.where is not None:
                return [f"# {op.comment}", f"print({op.frame}.loc[{_where(op.frame, op.where)}, {columns}])"]
            if op.decimals is not None:
                return [f"# {op.comment}", f"print({op.frame}[{columns}].round({op.decimals}))"]
            return [f"# {op.comment}", f"print({op.frame}[{columns}])"]
        if isinstance(op, MapCodes):
            pairs = ", ".join(f"{text(label)}: {text(code)}" for label, code in op.mapping)
            return [
                f"# {op.comment}",
                f'{op.frame}["{op.name}"] = {op.frame}["{op.source}"].map({{{pairs}}})',
                f'print({op.frame}[["{op.source}", "{op.name}"]].head(8))',
            ]
        if isinstance(op, Groups):
            return [
                f"# {op.comment}",
                f'{op.frame}["{op.name}"] = np.repeat({_list(op.labels)}, {_list(op.sizes)})',
            ]
        if isinstance(op, Support):
            lower, upper = E.format_number(op.lower), E.format_number(op.upper)
            return [f"# {op.comment}", f'{op.frame} = pd.DataFrame({{"{op.name}": np.arange({lower}, {upper} + 1)}})']
        if isinstance(op, RowSum):
            return [f"# {op.comment}", f'{op.frame}["{op.name}"] = {op.frame}[{_list(op.columns)}].sum(axis=1)']
        if isinstance(op, Rectangles):
            lower, width = E.format_number(op.lower), E.format_number(op.width)
            return [
                f"# {op.comment}",
                f"# {op.count} dikdörtgen; {op.name}: dikdörtgenlerin orta noktaları, alt sınır + genişlik × (i − 0,5)",
                f'{op.frame} = pd.DataFrame({{"{op.name}": {lower} + {width} * (np.arange(1, {op.count} + 1) - 0.5)}})',
            ]
        if isinstance(op, Derive):
            rhs = _render(op.expr, self.dialect(op.frame))
            return [f"# {op.comment}", f'{op.frame}["{op.name}"] = {rhs}']
        if isinstance(op, CopyFrame):
            return [f"# {op.comment}", f"{op.frame} = {op.source}.copy()  # bağımsız kopya: {op.source} değişmez"]
        if isinstance(op, NewSample):
            frame = f'{op.frame} = pd.DataFrame({{"id": np.arange(1, {op.nobs} + 1)}})'
            if op.seed is None:
                return [frame]
            return [
                "# Sabit tohum: betik her çalıştırmada aynı veriyi üretir",
                f"rng = np.random.default_rng({op.seed})",
                frame,
            ]
        if isinstance(op, Draw):
            a, b = E.format_number(op.first), E.format_number(op.second)
            if op.distribution == "exponential":  # ortalama süre μ; numpy'de ölçek parametresi
                call = f"rng.exponential({a}, size=len({op.frame}))"
            else:
                method = {"normal": "normal", "uniform": "uniform", "beta": "beta", "gamma": "gamma"}[op.distribution]
                call = f"rng.{method}({a}, {b}, size=len({op.frame}))"
            return [f"# {op.comment}", f'{op.frame}["{op.name}"] = {call}']
        if isinstance(op, DrawCount):
            return self._draw_count(op)
        if isinstance(op, DrawCategory):
            return self._draw_category(op)
        if isinstance(op, DrawDiscrete):
            return [
                f"# {op.comment}",
                "# tekduze ~ Tek-düze(0, 1); X, birikimli olasılığı F(x) bu sayıyı ilk aşan değerdir "
                "(ters dağılım fonksiyonu)",
                f"tekduze = rng.random(len({op.frame}))",
                f"{op.name}_degerler = np.array({_list(op.values)}, dtype=float)",
                f"esik = np.cumsum({_list(op.probabilities)})",
                "esik[-1] = 1.0  # yuvarlama hatasına karşı son eşik tam 1",
                f'{op.frame}["{op.name}"] = {op.name}_degerler[np.searchsorted(esik, tekduze, side="right")]',
            ]
        if isinstance(op, Shape):
            columns = f".drop(columns={_list(op.exclude)})" if op.exclude else ""
            note = "  # kimlik sütunu değişken sayılmaz" if op.exclude else ""
            return [
                f"{op.observations} = len({op.frame})  # gözlem sayısı",
                f"{op.variables} = {op.frame}{columns}.shape[1]{note}",
                f'print("Gözlem sayısı:", {op.observations}, "| Değişken sayısı:", {op.variables})',
            ]
        if isinstance(op, Count):
            return [
                f"# {op.comment}",
                f'{op.name} = int(({op.frame}["{op.column}"] == {text(op.value)}).sum())',
                f'print("{_quote(op.comment)}:", {op.name})',
            ]
        if isinstance(op, Statistic):
            source = (f'{op.frame}["{op.variable}"]' if op.where is None
                      else f'{op.frame}.loc[{_where(op.frame, op.where)}, "{op.variable}"]')
            return [
                f"# {op.comment}",
                f"{op.name} = {_stat_call(source, op.stat)}",
                f'print(f"{_fstring(op.comment)}: {_number(op.name, op.decimals)}")',
            ]
        if isinstance(op, PairStatistic):
            method = {"cov": "cov", "corr": "corr"}[op.stat]
            note = "  # payda n − 1" if op.stat == "cov" else "  # Pearson korelasyonu"
            return [
                f"# {op.comment}",
                f'{op.name} = {op.frame}["{op.x}"].{method}({op.frame}["{op.y}"]){note}',
                f'print(f"{_fstring(op.comment)}: {{{op.name}:.{op.decimals}f}}")',
            ]
        if isinstance(op, Scalar):
            rhs = _render(op.expr, self.dialect(""))
            shown = f"%{_number(op.name, op.decimals)}" if op.percent else _number(op.name, op.decimals)
            return [f"# {op.comment}", f"{op.name} = {rhs}", f'print(f"{_fstring(op.comment)}: {shown}")']
        if isinstance(op, ScalarTable):
            dialect = self.dialect("")
            rows = [f"    {text(label)}: {_render(expression, dialect)}," for label, expression in op.rows]
            return [
                f"{op.result} = pd.DataFrame({{\"deger\": {{", *rows, "}})",
                f"print({op.result}.round({op.decimals}))",
            ]
        if isinstance(op, GroupSummary):
            lines = [f'{op.result} = {op.frame}.groupby("{op.by}").agg(']
            for name, variable, stat in op.columns:
                lines.append(f'    {name}=("{variable}", "{stat}"),')
            lines += [f").reindex({_list(op.order)})", f"print({op.result}.round(4))"]
            return lines
        if isinstance(op, FrequencyTable):
            return self._frequency(op)
        if isinstance(op, CrossTab):
            return self._crosstab(op)
        if isinstance(op, JoinColumns):
            items = [f'    {text(name)}: {table}["{column}"].to_numpy(),' for name, table, column in op.columns]
            return [
                f"{op.result} = pd.DataFrame({{",
                *items,
                f"}}, index={op.columns[0][1]}.index)",
                f"print({op.result}.round({op.decimals}))",
            ]
        if isinstance(op, BoxSummary):
            items = [f'    {text(label)}: kutu_ozeti({frame}["{variable}"]),' for frame, variable, label in op.series]
            return [
                "# Beş sayı özeti, IQR, aykırı değer sınırları ve bıyık uçları",
                f"{op.result} = pd.DataFrame({{",
                *items,
                "})",
                f"print({op.result}.round(3))",
            ]
        if isinstance(op, ClassTable):
            return self._class_table(op)
        if isinstance(op, StemLeaf):
            return self._stem_leaf(op)
        if isinstance(op, Percentile):
            return self._percentile(op)
        if isinstance(op, ClassHistogram):
            return self._class_histogram(op)
        if isinstance(op, DotPlot):
            return self._dot_plot(op)
        if isinstance(op, BarChart):
            return self._bar(op)
        if isinstance(op, GroupedBarChart):
            return self._grouped(op)
        if isinstance(op, CompareBarChart):
            return self._compare(op)
        if isinstance(op, PieChart):
            return self._pie(op)
        if isinstance(op, LineChart):
            return self._line(op)
        if isinstance(op, ScatterPlot):
            return self._scatter(op)
        if isinstance(op, BoxPlot):
            return self._box_plot(op)
        if isinstance(op, Histogram):
            return self._histogram(op)
        if isinstance(op, MosaicChart):
            return self._mosaic(op)
        if isinstance(op, TreeDiagram):
            return self._tree(op)
        if isinstance(op, HeatMap):
            return self._heatmap(op)
        if isinstance(op, DensityPlot):
            return self._density(op)
        if isinstance(op, DensityCompare):
            return self._density_compare(op)
        if isinstance(op, PmfWithDensity):
            return self._pmf_density(op)
        if isinstance(op, MonteCarlo):
            return self._monte_carlo(op)
        if isinstance(op, SummaryTable):
            return self._summary_table(op)
        raise TypeError(f"Python üreticisi bu işlemi tanımıyor: {type(op).__name__}")

    @staticmethod
    def _axes(x_label: str, y_label: str, title: str, legend: bool | str = False) -> list[str]:
        lines = [
            f'ax.set_xlabel("{_quote(x_label)}")',
            f'ax.set_ylabel("{_quote(y_label)}")',
            f'ax.set_title("{_quote(title)}")',
        ]
        if legend == "disarida":
            lines.append('ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))  # açıklama sütunların dışında')
        elif legend:
            lines.append("ax.legend()")
        return lines + ["plt.tight_layout()", "plt.show()"]

    def _inline(self, op: InlineData) -> list[str]:
        lines = [f"# {op.comment}"]
        if op.layout and len(op.columns) == 1:
            name = f"{op.frame}_ham"
            items = [text(row[0]) for row in op.rows]
            lines += wrapped(f"{name} = [", items, "]", per_line=op.layout)
            lines += [f'{op.frame} = pd.DataFrame({{"{op.columns[0]}": {name}}})']
        else:
            lines.append(f"{op.frame} = pd.DataFrame({{")
            for position, column in enumerate(op.columns):
                values = [text(row[position]) for row in op.rows]
                lines += wrapped(f'    "{column}": [', values, "],")
            lines.append("})")
        lines.append(f"print({op.frame})" if len(op.rows) <= 12 else f"print({op.frame}.head())  # ilk beş gözlem")
        return lines

    def _from_counts(self, op: FromCounts) -> list[str]:
        name = f"{op.frame}_sayim"
        rows = ["        (" + ", ".join(text(value) for value in row) + ")," for row in op.rows]
        columns = _list((*op.columns, "sayi"))
        return [
            f"# {op.comment}",
            f"{name} = pd.DataFrame(",
            "    [",
            *rows,
            "    ],",
            f"    columns={columns},",
            ")",
            '# Her satır "sayi" kez tekrarlanır: bir satır = bir gözlem',
            f'satirlar = {name}.index.repeat({name}["sayi"])',
            f"{op.frame} = {name}.loc[satirlar, {_list(op.columns)}].reset_index(drop=True)",
            f"print(len({op.frame}))  # gözlem sayısı",
        ]

    def _outcomes(self, op: Outcomes) -> list[str]:
        shown = f"print({op.frame})" if self._row_count(op) <= 12 else f"print({op.frame}.head())  # ilk beş sonuç"
        if len(op.stages) == 1:
            name, values = op.stages[0]
            return [
                f"# {op.comment}",
                f"{op.frame} = pd.DataFrame({{{text(name)}: {_list(values)}}})",
                shown,
                f'print("Sonuç sayısı:", len({op.frame}))',
            ]
        stages = [f"    {text(name)}: {_list(values)}," for name, values in op.stages]
        return [
            f"# {op.comment}",
            f"{op.frame}_asamalar = {{",
            *stages,
            "}",
            "# Bütün bileşimler (çarpım kuralı): ilk aşama en yavaş, son aşama en hızlı değişir",
            f"{op.frame} = pd.DataFrame(list(product(*{op.frame}_asamalar.values())),",
            f"{' ' * len(op.frame)}                columns=list({op.frame}_asamalar))",
            shown,
            f'print("Sonuç sayısı:", len({op.frame}))',
        ]

    @staticmethod
    def _row_count(op: Outcomes) -> int:
        count = 1
        for _, values in op.stages:
            count *= len(values)
        return count

    def _selections(self, op: Selections) -> list[str]:
        method = "permutations" if op.ordered else "combinations"
        note = "sıra önemli: permütasyonlar" if op.ordered else "sıra önemsiz: kombinasyonlar"
        return [
            f"# {op.comment} ({note})",
            f"{op.frame}_ogeler = {_list(op.items)}",
            f"{op.frame} = pd.DataFrame(list({method}({op.frame}_ogeler, {op.k})), columns={_list(op.columns)})",
            f"print({op.frame}.head(10))",
            f'print("Seçim sayısı:", len({op.frame}))',
        ]

    @staticmethod
    def _draw_count(op: DrawCount) -> list[str]:
        values = [E.format_number(value) for value in op.parameters]
        size = f"size=len({op.frame})"
        if op.distribution == "binomial":
            call = f"rng.binomial(n={values[0]}, p={values[1]}, {size})"
            note = "n bağımsız Bernoulli denemesindeki başarı sayısı"
        elif op.distribution == "poisson":
            call = f"rng.poisson(lam={values[0]}, {size})"
            note = "aralıktaki olay sayısı"
        else:
            population, successes, draws = (int(value) for value in op.parameters)
            call = f"rng.hypergeometric(ngood={successes}, nbad={population - successes}, nsample={draws}, {size})"
            note = "N birimden yerine koymadan seçilen n birimdeki başarı sayısı"
        return [f"# {op.comment} ({note})", f'{op.frame}["{op.name}"] = {call}']

    def _draw_category(self, op: DrawCategory) -> list[str]:
        categories = f"{op.name}_kategoriler"
        lines = [
            f"# {op.comment}",
            "# u ~ Tek-düze(0, 1); kategori, birikimli olasılığı u'yu ilk aşan kategoridir",
            f"u = rng.random(len({op.frame}))",
            f"{categories} = np.array({_list(op.categories)})",
        ]
        if not op.by:
            probs = op.probabilities[0][1]
            return lines + [
                f"esik = np.cumsum({_list(probs)})",
                "esik[-1] = 1.0  # yuvarlama hatasına karşı son eşik tam 1",
                f'{op.frame}["{op.name}"] = {categories}[np.searchsorted(esik, u, side="right")]',
            ]
        if len(op.by) == 1:
            entries = [f"    {text(condition[0])}: {_list(probs)}," for condition, probs in op.probabilities]
            selection = f'({op.frame}["{op.by[0]}"] == kosul).to_numpy()'
        else:
            entries = [
                f"    ({', '.join(text(value) for value in condition)}): {_list(probs)},"
                for condition, probs in op.probabilities
            ]
            selection = " & ".join(
                f'({op.frame}["{column}"] == kosul[{index}]).to_numpy()' for index, column in enumerate(op.by)
            )
        return lines + [
            f"{op.name}_olasilik = {{",
            *entries,
            "}",
            f'{op.frame}["{op.name}"] = ""',
            f"for kosul, olasilik in {op.name}_olasilik.items():",
            f"    secili = {selection}",
            "    esik = np.cumsum(olasilik)",
            "    esik[-1] = 1.0",
            f'    {op.frame}.loc[secili, "{op.name}"] = {categories}[np.searchsorted(esik, u[secili], side="right")]',
        ]

    def _frequency(self, op: FrequencyTable) -> list[str]:
        order = f"{op.result}_sira"
        lines = [
            f"{order} = {_list(op.order)}",
            f'sayilar = {op.frame}["{op.variable}"].value_counts()  # her kategorideki gözlem sayısı',
            f'{op.result} = sayilar.reindex({order}, fill_value=0).to_frame("frekans")',
        ]
        if op.relative:
            lines += [
                f'{op.result}["goreli"] = {op.result}["frekans"] / {op.result}["frekans"].sum()  # r = f / n',
                f'{op.result}["yuzde"] = 100 * {op.result}["goreli"]  # p = 100 r',
            ]
        if op.totals:
            lines.append(f'{op.result}.loc["{TOTAL}"] = {op.result}.sum()')
        lines.append(f"print({op.result}.round(3))")
        return lines

    def _crosstab(self, op: CrossTab) -> list[str]:
        lines: list[str] = []
        data = op.frame
        if op.where is not None:
            data = f"{op.result}_veri"
            lines += [
                f"# Yalnız {op.where[0]} = {op.where[1]} olan gözlemler",
                f"{data} = {op.frame}[{_where(op.frame, op.where)}]",
            ]
        counts = op.result if op.percent is None else f"{op.result}_sayi"
        if op.weights is None:
            lines.append(f'{counts} = pd.crosstab({data}["{op.row}"], {data}["{op.column}"]).reindex(')
        else:
            lines += [
                f'# Hücreler gözlem sayısı değil, "{op.weights}" sütununun toplamıdır',
                f'{counts} = pd.crosstab({data}["{op.row}"], {data}["{op.column}"], values={data}["{op.weights}"],',
                f'{" " * (len(counts) + 15)}aggfunc="sum").fillna(0).reindex(',
            ]
        lines += [
            f"    index={_list(op.row_order)},",
            f"    columns={_list(op.column_order)},",
            "    fill_value=0,",
            ")",
        ]
        if op.percent is None:
            if op.margins:
                lines += [
                    f'{op.result}.loc["{TOTAL}"] = {op.result}.sum()  # sütun toplamları',
                    f'{op.result}["{TOTAL}"] = {op.result}.sum(axis=1)  # satır toplamları',
                ]
            return lines + [f"print({op.result})"]
        if op.percent == "satir":
            lines += [
                "# Satır yüzdesi: payda satır toplamıdır",
                f"{op.result} = {counts}.div({counts}.sum(axis=1), axis=0) * 100",
            ]
            if op.margins:
                lines.append(f'{op.result}["{TOTAL}"] = {op.result}.sum(axis=1)')
        else:
            lines += [
                "# Sütun yüzdesi: payda sütun toplamıdır",
                f"{op.result} = {counts}.div({counts}.sum(axis=0), axis=1) * 100",
            ]
            if op.margins:
                lines.append(f'{op.result}.loc["{TOTAL}"] = {op.result}.sum()')
        return lines + [f"print({op.result}.round({op.decimals}))"]

    def _class_table(self, op: ClassTable) -> list[str]:
        source = f'{op.frame}["{op.variable}"]'
        r = op.result
        if op.lower is None:
            lines = [
                f"h = {E.format_number(op.width)}  # sınıf genişliği",
                f"alt_sinir = np.floor({source}.min() / h) * h  # en küçük değeri içeren h katı",
                f"k = int(np.floor(({source}.max() - alt_sinir) / h)) + 1  # en büyük değeri de kapsayan sınıf sayısı",
                "kenarlar = alt_sinir + h * np.arange(k + 1)",
            ]
        else:
            edges = class_edges(None, op.width, op.lower, op.classes)
            lines = wrapped("kenarlar = [", [E.format_number(value) for value in edges], "]  # sınıf sınırları")
        if op.row_labels == "ust":
            labels = 'etiketler = [f"x < {sinir_metni(b)}" for b in kenarlar[1:]]'
        else:
            labels = ('etiketler = [f"{sinir_metni(a)} ≤ x < {sinir_metni(b)}" '
                      'for a, b in zip(kenarlar[:-1], kenarlar[1:])]')
        lines += [
            "# Sınıflar [alt, üst): alt sınır dahil, üst sınır hariç",
            f"siniflar = pd.cut({source}, bins=kenarlar, right=False)",
            "frekans = siniflar.value_counts(sort=False).to_numpy()  # her sınıftaki gözlem sayısı",
            "n_sinif = frekans.sum()",
            labels,
            f'{r} = pd.DataFrame({{"alt": kenarlar[:-1], "ust": kenarlar[1:]}}, index=etiketler)',
        ]
        steps = {
            "orta_nokta": f'{r}["orta_nokta"] = ({r}["alt"] + {r}["ust"]) / 2  # m = (alt + üst) / 2',
            "frekans": f'{r}["frekans"] = frekans',
            "goreli": f'{r}["goreli"] = frekans / n_sinif  # r = f / n',
            "yuzde": f'{r}["yuzde"] = 100 * (frekans / n_sinif)  # p = 100 r',
            "kumulatif_frekans": f'{r}["kumulatif_frekans"] = frekans.cumsum()  # F = f₁ + ... + fⱼ',
            "kumulatif_goreli": f'{r}["kumulatif_goreli"] = frekans.cumsum() / n_sinif',
            "kumulatif_yuzde": f'{r}["kumulatif_yuzde"] = 100 * (frekans.cumsum() / n_sinif)',
        }
        lines += [line for column, line in steps.items() if column in op.columns]
        if op.totals:
            summed = [column for column in ("frekans", "goreli", "yuzde") if column in op.columns]
            lines.append(f'{r}.loc["{TOTAL}"] = {r}[{_list(summed)}].sum()  # alt ve üst sınır toplanmaz')
        return lines + [f"print({r}.round(3))"]

    def _stem_leaf(self, op: StemLeaf) -> list[str]:
        r = op.result
        return [
            f'sirali = {op.frame}["{op.variable}"].sort_values().astype(int)',
            "govde, yaprak = sirali // 10, sirali % 10  # gövde: onlar basamağı, yaprak: birler basamağı",
            "govdeler = range(govde.min(), govde.max() + 1)",
            f"{r} = pd.DataFrame({{",
            '    "yapraklar": [" ".join(str(v) for v in yaprak[govde == g]) for g in govdeler],',
            '    "yaprak_sayisi": [int((govde == g).sum()) for g in govdeler],',
            "}, index=[str(g) for g in govdeler])",
            f"for g, satir in {r}.iterrows():",
            '    print(f"{g} | {satir[\'yapraklar\']}")',
        ]

    def _percentile(self, op: Percentile) -> list[str]:
        p = E.format_number(op.p)
        source = f'{op.frame}["{op.variable}"]'
        if op.method == "ders":
            lines = [f"# {op.comment}: ders kuralı L_p = (p/100)(n + 1)"]
            location = f"{p} / 100 * (len({source}) + 1)"
            value = f"yuzdelik({source}, {p})"
        else:
            lines = [f"# {op.comment}: numpy varsayılanı, konum 1 + (p/100)(n − 1)"]
            location = f"1 + {p} / 100 * (len({source}) - 1)"
            value = f"np.percentile({source}, {p})"
        if op.location is not None:
            lines.append(f"{op.location} = {location}")
        lines.append(f"{op.name} = {value}")
        shown = f"{_fstring(op.comment)}: {{{op.name}:.{op.decimals}f}}"
        if op.location is not None:
            shown += f" (konum {{{op.location}:.2f}})"
        return lines + [f'print(f"{shown}")']

    def _class_histogram(self, op: ClassHistogram) -> list[str]:
        lines = [
            f"cizim = {self._chart_table(op.table)}",
            "fig, ax = plt.subplots(figsize=(8, 5))",
            "# Bitişik dikdörtgenler: genişlik sınıf genişliği, yükseklik sınıfın değeri",
            f'cubuklar = ax.bar(cizim["alt"], cizim["{op.y}"], width=cizim["ust"] - cizim["alt"], align="edge",',
            f'                  color="{PALETTE[0]}", edgecolor="white")',
            'ax.set_xticks(list(cizim["alt"]) + [cizim["ust"].iloc[-1]])  # sınıf sınırları',
        ]
        if op.labels:
            percent = ", yuzde=True" if op.percent else ""
            lines.append(
                f'ax.bar_label(cubuklar, labels=[sayi_metni(v, {op.decimals}{percent}) for v in cizim["{op.y}"]], '
                "padding=2)"
            )
            lines.append("ax.margins(y=0.1)  # en yüksek dikdörtgenin etiketi için üstte boşluk")
        return lines + self._axes(op.x_label, op.y_label, op.title)

    def _dot_plot(self, op: DotPlot) -> list[str]:
        source = f'{op.frame}["{op.variable}"]'
        lines = [
            f'yigin = {op.frame}.groupby("{op.variable}").cumcount() + 1  # aynı değerdeki gözlemler üst üste',
            "fig, ax = plt.subplots(figsize=(8, 3.5))",
            f'ax.scatter({source}, yigin, color="{PALETTE[0]}", s=45, zorder=3)',
        ]
        for index, (name, label) in enumerate(op.references):
            color = REFERENCE_COLORS[index % len(REFERENCE_COLORS)]
            style = ("--", ":", "-.")[index % 3]
            lines.append(f'ax.axvline({name}, color="{color}", linestyle="{style}", linewidth=2, label={text(label)})')
        lines += [
            "ax.set_yticks(range(1, int(yigin.max()) + 1))",
            "ax.set_ylim(0.3, yigin.max() + 0.7)",
        ]
        if op.x_range is not None:
            low, high = (E.format_number(value) for value in op.x_range)
            lines.append(f"ax.set_xlim({low}, {high})  # karşılaştırılan grafiklerde aynı yatay eksen")
        return lines + self._axes(op.x_label, op.y_label, op.title, legend=bool(op.references))

    def _line(self, op: LineChart) -> list[str]:
        marker = ', marker="o"' if op.markers else ""
        if op.series:
            lines = [
                "fig, ax = plt.subplots(figsize=(8, 5))",
                f'ax.plot({op.frame}["{op.x}"], {op.frame}["{op.y}"]{marker}, color="{PALETTE[0]}", '
                f"label={text(op.legend or op.y_label)})",
            ]
            for index, (column, label) in enumerate(op.series, start=1):
                lines.append(f'ax.plot({op.frame}["{op.x}"], {op.frame}["{column}"]{marker}, '
                             f'color="{PALETTE[index % len(PALETTE)]}", label={text(label)})')
            return lines + ["ax.grid(alpha=0.3)", *self._axes(op.x_label, op.y_label, op.title, legend=True)]
        plot = f'ax.plot({op.frame}["{op.x}"], {op.frame}["{op.y}"]{marker}, color="{PALETTE[0]}"'
        lines = ["fig, ax = plt.subplots(figsize=(8, 5))"]
        legend = bool(op.references or op.bands)
        # Başvuru çizgisi ya da bant varsa açıklama (legend) gerekir; seri de adıyla açıklamada yer alır.
        lines += [plot + ",", f"        label={text(op.y_label)})"] if legend else [plot + ")"]
        for index, (name, label) in enumerate(op.references):
            color = REFERENCE_COLORS[index % len(REFERENCE_COLORS)]
            style = ("--", ":", "-.")[index % 3]
            lines.append(f'ax.axhline({name}, color="{color}", linestyle="{style}", linewidth=2, label={text(label)})')
        for column, label in op.bands:  # boş etiket açıklamada gösterilmez
            shown = text(label) if label else '"_nolegend_"'
            lines.append(f'ax.plot({op.frame}["{op.x}"], {op.frame}["{column}"], color="{PALETTE[2]}", '
                         f'linestyle="--", linewidth=1.5, label={shown})')
        return lines + self._axes(op.x_label, op.y_label, op.title, legend=legend)

    def _scatter(self, op: ScatterPlot) -> list[str]:
        size = E.format_number(4 * op.size)  # matplotlib noktanın alanını (pt²) alır; Plotly çapını
        alpha = "" if op.opacity >= 1 else f", alpha={E.format_number(op.opacity)}"
        lines = ["fig, ax = plt.subplots(figsize=(7, 5))"]
        if not (op.fit_line or op.lines or op.means):
            return lines + [
                f'ax.scatter({op.frame}["{op.x}"], {op.frame}["{op.y}"], color="{PALETTE[0]}", s={size}{alpha}, '
                f"zorder=3)",
                "ax.grid(alpha=0.3)",
                *self._axes(op.x_label, op.y_label, op.title),
            ]
        lines += [
            f'nokta = {op.frame}[["{op.x}", "{op.y}"]].dropna()',
            f'ax.scatter(nokta["{op.x}"], nokta["{op.y}"], color="{PALETTE[0]}", s={size}{alpha}, zorder=3)',
        ]
        if op.means:
            lines += [
                f"# {op.means}: aynı {op.x} değerindeki gözlemlerin {op.y} ortalaması",
                f'ortalama = nokta.groupby("{op.x}")["{op.y}"].mean()',
                f'ax.scatter(ortalama.index, ortalama.values, color="{PALETTE[3]}", s=64, edgecolor="white", '
                f"zorder=4, label={text(op.means)})",
            ]
        if op.fit_line or op.lines:
            lines.append(f'x_dogru = np.linspace(nokta["{op.x}"].min(), nokta["{op.x}"].max(), 100)')
        if op.fit_line:
            lines += [
                "# Tahmin edilen doğru: eğim ve sabit terim veriden (birinci dereceden polinom)",
                f'egim, sabit = np.polyfit(nokta["{op.x}"], nokta["{op.y}"], 1)',
                f'ax.plot(x_dogru, sabit + egim * x_dogru, color="{PALETTE[1]}", linewidth=2.5, '
                'label="Tahmin edilen doğru")',
            ]
        for index, (first, second, label) in enumerate(op.lines):
            color = REFERENCE_COLORS[index % len(REFERENCE_COLORS)]
            if isinstance(second, str) or second >= 0:
                line = f"{_parameter(first)} + {_parameter(second)} * x_dogru"
            else:  # "+ -0.5 * x" yazılmasın
                line = f"{_parameter(first)} - {_parameter(-second)} * x_dogru"
            lines += [
                f"# {label}: y = sabit + eğim · x (kesikli çizgi)",
                f"ax.plot(x_dogru, {line}, color=\"{color}\", "
                f'linestyle="--", linewidth=2.5, label={text(label)})',
            ]
        return lines + ["ax.grid(alpha=0.3)", *self._axes(op.x_label, op.y_label, op.title, legend=True)]

    def _box_plot(self, op: BoxPlot) -> list[str]:
        series = [f'    ({text(label)}, {frame}["{variable}"]),' for frame, variable, label in op.series]
        return [
            "# Kutu: Q1'den Q3'e; çizgi: medyan; bıyıklar: sınırların içindeki en uç gözlemler; noktalar: aykırı",
            "seriler = [",
            *series,
            "]",
            "fig, ax = plt.subplots(figsize=(8, 1.6 + 1.1 * len(seriler)))",
            "for y, (etiket, degerler) in enumerate(seriler, start=1):",
            "    k = kutu_ozeti(degerler)",
            f'    ax.add_patch(plt.Rectangle((k["q1"], y - 0.25), k["iqr"], 0.5, facecolor="{PALETTE[0]}33",',
            f'                               edgecolor="{PALETTE[0]}", linewidth=2))',
            f'    ax.plot([k["medyan"], k["medyan"]], [y - 0.25, y + 0.25], color="{PALETTE[1]}", linewidth=3)',
            '    ax.plot([k["alt_biyik"], k["q1"]], [y, y], color="black")  # sol bıyık',
            '    ax.plot([k["q3"], k["ust_biyik"]], [y, y], color="black")  # sağ bıyık',
            '    for uc in (k["alt_biyik"], k["ust_biyik"]):',
            '        ax.plot([uc, uc], [y - 0.12, y + 0.12], color="black")',
            "    x = np.asarray(degerler, dtype=float)",
            '    aykiri = x[(x < k["alt_sinir"]) | (x > k["ust_sinir"])]',
            f'    ax.scatter(aykiri, [y] * len(aykiri), color="{PALETTE[1]}", s=45, zorder=3)',
            "ax.set_yticks(range(1, len(seriler) + 1), [etiket for etiket, _ in seriler])",
            "ax.set_ylim(0.4, len(seriler) + 0.6)",
            "ax.autoscale(axis=\"x\")",
            *self._axes(op.x_label, op.y_label, op.title),
        ]

    def _chart_table(self, table: str) -> str:
        rows, columns = self.totals.get(table, (False, False))
        drops = []
        if rows:
            drops.append(f'index="{TOTAL}"')
        if columns:
            drops.append(f'columns="{TOTAL}"')
        return f"{table}.drop({', '.join(drops)})" if drops else table

    def _bar(self, op: BarChart) -> list[str]:
        if op.x is None:
            lines = [f'cizim = {self._chart_table(op.source)}["{op.y}"]']
            if op.source in self.numeric_tables:
                lines.append("cizim.index = cizim.index.astype(str)  # sayısal değerler kategori etiketi olarak")
        elif (op.source, op.x) in self.numeric_columns:
            lines = [
                "# Sayısal değerler kategori etiketi olarak: her değer bir sütun, eksende yalnız bu değerler yazılır",
                f'cizim = pd.Series({op.source}["{op.y}"].to_numpy(), index={op.source}["{op.x}"].astype(str))',
            ]
        else:
            lines = [f'cizim = pd.Series({op.source}["{op.y}"].to_numpy(), index={op.source}["{op.x}"])']
        if op.sort == "azalan":
            lines.append('cizim = cizim.sort_values(ascending=False, kind="stable")  # yüksekten düşüğe')
        labels = f"[sayi_metni(v, {op.decimals}{', yuzde=True' if op.percent else ''}) for v in cizim]"
        lines.append("fig, ax = plt.subplots(figsize=(8, 5))")
        if op.horizontal:
            lines += [
                f'cubuklar = ax.barh(cizim.index, cizim.values, color="{PALETTE[0]}")',
                "ax.invert_yaxis()  # ilk kategori en üstte",
                f"ax.bar_label(cubuklar, labels={labels}, padding=3)",
            ]
            if op.y_range is not None:
                lines.append(f"ax.set_xlim({E.format_number(op.y_range[0])}, {E.format_number(op.y_range[1])})")
            else:
                lines.append("ax.margins(x=0.1)  # en uzun sütunun etiketi için sağda boşluk")
            return lines + self._axes(op.y_label, op.x_label, op.title)
        lines += [
            f'cubuklar = ax.bar(cizim.index, cizim.values, color="{PALETTE[0]}")',
            f"ax.bar_label(cubuklar, labels={labels}, padding=3)",
        ]
        if op.y_range is not None:
            lines.append(f"ax.set_ylim({E.format_number(op.y_range[0])}, {E.format_number(op.y_range[1])})")
        else:
            lines.append("ax.margins(y=0.1)  # en yüksek sütunun etiketi için üstte boşluk")
        return lines + self._axes(op.x_label, op.y_label, op.title)

    def _grouped(self, op: GroupedBarChart) -> list[str]:
        table = self._chart_table(op.table)
        frame = f"{table}.T" if op.series == "satir" else table
        stacked = ", stacked=True" if op.stacked else ""
        position = ', label_type="center", color="white"' if op.stacked else ", padding=2"
        labels = [
            "for kap in ax.containers:",
            f"    ax.bar_label(kap, labels=[sayi_metni(v, {op.decimals}) for v in kap.datavalues]{position})",
        ] if op.labels else []
        if op.labels and not op.stacked:
            labels.append("ax.margins(y=0.1)  # en yüksek sütunun etiketi için üstte boşluk")
        return [
            f"cizim = {frame}  # satırlar yatay eksende, sütunlar seriler",
            f"renkler = {_list(PALETTE)}[: cizim.shape[1]]",
            "fig, ax = plt.subplots(figsize=(8, 5))",
            f"cizim.plot.bar(ax=ax, rot=0{stacked}, color=renkler)",
            *labels,
            *self._axes(op.x_label, op.y_label, op.title, legend="disarida" if op.stacked else True),
        ]

    def _compare(self, op: CompareBarChart) -> list[str]:
        items = [f'    {text(label)}: {name}["{op.column}"],' for label, name in op.tables]
        return [
            "# Her tablonun aynı sütunu yan yana; satırlar yatay eksende, sütunlar seriler",
            "cizim = pd.DataFrame({",
            *items,
            "}).T",
            f"renkler = {_list(PALETTE)}[: cizim.shape[1]]",
            "fig, ax = plt.subplots(figsize=(8, 5))",
            "cizim.plot.bar(ax=ax, rot=0, color=renkler)",
            "for kap in ax.containers:",
            f"    ax.bar_label(kap, labels=[sayi_metni(v, {op.decimals}) for v in kap.datavalues], padding=2)",
            "ax.margins(y=0.1)  # en yüksek sütunun etiketi için üstte boşluk",
            *self._axes(op.x_label, op.y_label, op.title, legend=True),
        ]

    def _pie(self, op: PieChart) -> list[str]:
        order = f"{op.result}_sira"
        return [
            f"{order} = {_list(op.order)}  # dilimlerin sırası",
            f'{op.result} = {op.table}.loc[{order}, ["{op.column}"]]',
            f'{op.result}["aci"] = 360 * {op.result}["{op.column}"]  # θ = 360° × r',
            f"print({op.result}.round(3))",
            "fig, ax = plt.subplots(figsize=(6, 6))",
            f'ax.pie({op.result}["{op.column}"], labels={op.result}.index, startangle=0, counterclock=True,',
            f"       colors={_list(PALETTE[:len(op.order)])},",
            '       autopct=lambda p: "%" + f"{p:.1f}".replace(".", ","))',
            f'ax.set_title("{_quote(op.title)}")',
            "plt.show()",
        ]

    def _histogram(self, op: Histogram) -> list[str]:
        lower, upper = E.format_number(op.lower), E.format_number(op.upper)
        lines = [
            f"# [{lower}, {upper}] aralığında {op.bins} eşit genişlikte kutu",
            f"kutular = np.linspace({lower}, {upper}, {op.bins} + 1)",
            "fig, ax = plt.subplots(figsize=(8, 5))",
        ]
        for index, (column, label) in enumerate(op.columns):
            color = PALETTE[index % len(PALETTE)]
            lines.append(
                f'ax.hist({op.table}["{column}"], bins=kutular, alpha=0.55, color="{color}", label={text(label)})'
            )
        for index, (value, label) in enumerate(op.references):
            color = REFERENCE_COLORS[index % len(REFERENCE_COLORS)]
            position = value if isinstance(value, str) else E.format_number(value)
            lines.append(
                f'ax.axvline({position}, color="{color}", linestyle="--", linewidth=2, label={text(label)})'
            )
        if op.curves:
            first = op.columns[0][0]
            lines += [
                "# Beklenen sayı eğrisi: gözlem sayısı × kutu genişliği × f(x)",
                f"eksen = np.linspace({lower}, {upper}, 401)",
                f'beklenen = len({op.table}["{first}"]) * (kutular[1] - kutular[0])',
            ]
            for index, (distribution, a, b, label) in enumerate(op.curves, start=len(op.columns)):
                color = PALETTE[index % len(PALETTE)]
                curve = density_expression(distribution, a, b, "eksen")
                lines.append(f'ax.plot(eksen, beklenen * {curve}, color="{color}", linewidth=2, label={text(label)})')
        return lines + self._axes(op.x_label, op.y_label, op.title, legend=True)

    def _mosaic(self, op: MosaicChart) -> list[str]:
        return [
            f"cizim = {self._chart_table(op.table)}  # hücreler: sayılar ya da ortak olasılıklar",
            "genislik = cizim.sum(axis=1) / cizim.to_numpy().sum()  # sütun genişliği: satırın marjinal payı",
            "pay = cizim.div(cizim.sum(axis=1), axis=0)  # sütun içindeki pay: satır verildiğinde koşullu olasılık",
            "sol = genislik.cumsum() - genislik  # sütunların sol kenarı",
            f"renkler = {_list(PALETTE)}[: cizim.shape[1]]",
            "fig, ax = plt.subplots(figsize=(8, 5))",
            "taban = pd.Series(0.0, index=cizim.index)",
            "for renk, sutun in reversed(list(zip(renkler, cizim.columns))):  # ilk sütun en üstte",
            '    ax.bar(sol, pay[sutun], width=genislik, bottom=taban, align="edge", color=renk,',
            '           edgecolor="white", linewidth=2)',
            "    for satir in cizim.index:  # parçanın alanı = ortak olasılık",
            "        alan = genislik[satir] * pay.loc[satir, sutun]",
            "        ax.text(sol[satir] + genislik[satir] / 2, taban[satir] + pay.loc[satir, sutun] / 2,",
            f'                f"{{sutun}}\\n{{sayi_metni(alan, {op.decimals})}}", ha="center", va="center",',
            '                color="white")',
            "    taban = taban + pay[sutun]",
            "ax.set_xticks(sol + genislik / 2, cizim.index)",
            "ax.set_xlim(0, 1)",
            "ax.set_ylim(0, 1)",
            *self._axes(op.x_label, op.y_label, op.title),
        ]

    def _tree(self, op: TreeDiagram) -> list[str]:
        columns = _list((op.first, op.second, op.first_p, op.second_p))
        return [
            "# Olasılık ağacı (soldan sağa): her satır bir yol, ilk yol en üstte",
            f"yollar = {op.frame}[{columns}]",
            "y_yol = np.arange(len(yollar))[::-1]",
            f'ilk_dallar = list(dict.fromkeys(yollar["{op.first}"]))',
            f'y_ilk = {{dal: y_yol[(yollar["{op.first}"] == dal).to_numpy()].mean() for dal in ilk_dallar}}',
            "y_kok = np.mean(list(y_ilk.values()))  # kök, ilk dalların ortasında",
            f'kutu = {{"boxstyle": "round", "facecolor": "white", "edgecolor": "{PALETTE[0]}"}}',
            'etiket = {"textcoords": "offset points", "xytext": (0, 5), "ha": "center",',
            f'          "color": "{REFERENCE_COLORS[0]}"}}',
            "fig, ax = plt.subplots(figsize=(8, 1.4 + 0.8 * len(yollar)))",
            "for dal in ilk_dallar:",
            f'    p = yollar.loc[yollar["{op.first}"] == dal, "{op.first_p}"].iloc[0]',
            f'    ax.plot([0, 1], [y_kok, y_ilk[dal]], color="{PALETTE[0]}")',
            "    ax.annotate(sayi_metni(p, 2), (0.5, (y_kok + y_ilk[dal]) / 2), **etiket)",
            '    ax.text(1, y_ilk[dal], dal, ha="center", va="center", bbox=kutu)',
            "for i, (dal, sonuc, p_ilk, p_ikinci) in enumerate(yollar.itertuples(index=False)):",
            f'    ax.plot([1, 2], [y_ilk[dal], y_yol[i]], color="{PALETTE[0]}")',
            "    ax.annotate(sayi_metni(p_ikinci, 2), (1.5, (y_ilk[dal] + y_yol[i]) / 2), **etiket)",
            '    ax.text(2, y_yol[i], sonuc, ha="center", va="center", bbox=kutu)',
            f'    ax.text(2.3, y_yol[i], f"ortak {{sayi_metni(p_ilk * p_ikinci, {op.decimals})}}", va="center",',
            f'            color="{PALETTE[1]}")  # yolun ortak olasılığı: dal olasılıklarının çarpımı',
            f'ax.text(0, y_kok, "{_quote(op.root)}", ha="center", va="center", bbox=kutu)',
            "ax.set_xlim(-0.3, 3.0)",
            "ax.set_ylim(-0.6, len(yollar) - 0.4)",
            'ax.axis("off")  # ağaçta eksen yoktur',
            f'ax.set_title("{_quote(op.title)}")',
            "plt.tight_layout()",
            "plt.show()",
        ]

    def _heatmap(self, op: HeatMap) -> list[str]:
        return [
            f"cizim = {self._chart_table(op.table)}",
            "degerler = cizim.to_numpy(dtype=float)",
            "# Sıfır açık tonda, en büyük değer ana renkte; sıfır hücreler de zeminden ayrılır",
            f'renk_skalasi = LinearSegmentedColormap.from_list("isi", ["{HEAT_LOW}", "{PALETTE[0]}"])',
            "fig, ax = plt.subplots(figsize=(7, 5))",
            "ax.imshow(degerler, cmap=renk_skalasi, vmin=0, vmax=degerler.max())  # ilk satır en üstte",
            "ax.set_xticks([j - 0.5 for j in range(degerler.shape[1] + 1)], minor=True)",
            "ax.set_yticks([i - 0.5 for i in range(degerler.shape[0] + 1)], minor=True)",
            'ax.grid(which="minor", color="white", linewidth=3)  # hücre sınırları',
            'ax.tick_params(which="both", length=0)',
            "for kenar in ax.spines.values():",
            "    kenar.set_visible(False)",
            "for i in range(degerler.shape[0]):",
            "    for j in range(degerler.shape[1]):",
            '        renk = "white" if degerler[i, j] > 0.6 * degerler.max() else "black"',
            f'        ax.text(j, i, sayi_metni(degerler[i, j], {op.decimals}), ha="center", va="center", color=renk)',
            "ax.set_xticks(range(degerler.shape[1]), cizim.columns)",
            "ax.set_yticks(range(degerler.shape[0]), cizim.index)",
            "ax.xaxis.tick_top()",
            'ax.xaxis.set_label_position("top")',
            *self._axes(op.x_label, op.y_label, op.title),
        ]

    @staticmethod
    def _density_call(op: DensityPlot, x: str) -> str:
        return density_expression(op.distribution, op.first, op.second, x)

    def _density(self, op: DensityPlot) -> list[str]:
        low, high = (E.format_number(value) for value in op.x_range)
        if op.distribution == "normal":
            note = f"N(μ, σ²) yoğunluğu: μ = {E.format_number(op.first)}, σ = {E.format_number(op.second)}"
        elif op.distribution == "exponential":
            note = (f"Üstel yoğunluk f(x) = (1/μ)e^(−x/μ): ortalama süre μ = {E.format_number(op.first)}; "
                    "scipy'de ölçek μ")
        elif op.distribution == "gamma":
            note = (f"Gamma yoğunluğu: biçim {E.format_number(op.first)}, oran {E.format_number(op.second)}; "
                    "scipy'de ölçek 1/oran")
        else:
            note = (f"U(a, b) yoğunluğu: a = {E.format_number(op.first)}, b = {E.format_number(op.second)}; "
                    "scipy'de konum a, ölçek b − a")
        lines = [
            f"# {note}",
            f"eksen = np.linspace({low}, {high}, 401)",
            "fig, ax = plt.subplots(figsize=(8, 5))",
            f'ax.plot(eksen, {self._density_call(op, "eksen")}, color="{PALETTE[0]}", linewidth=2)',
        ]
        if op.shade:
            pairs = ", ".join(f"({E.format_number(a)}, {E.format_number(b)})" for a, b in op.shade)
            lines += [
                f"for alt, ust in [{pairs}]:  # olasılık = eğri altındaki alan",
                "    xa = np.linspace(alt, ust, 200)",
                f'    ax.fill_between(xa, {self._density_call(op, "xa")}, color="{PALETTE[0]}", alpha=0.3)',
            ]
        for index, (value, label) in enumerate(op.references):
            color = REFERENCE_COLORS[index % len(REFERENCE_COLORS)]
            style = ("--", ":", "-.")[index % 3]
            lines.append(f'ax.axvline({E.format_number(value)}, color="{color}", linestyle="{style}", linewidth=2, '
                         f"label={text(label)})")
        limit = "bottom=0" if op.y_max is None else f"0, {E.format_number(op.y_max)}"
        lines += [f"ax.set_xlim({low}, {high})", f"ax.set_ylim({limit})"]
        return lines + self._axes(op.x_label, op.y_label, op.title, legend=bool(op.references))

    def _density_compare(self, op: DensityCompare) -> list[str]:
        low, high = (E.format_number(value) for value in op.x_range)
        lines = [
            "# Yoğunluk eğrileri aynı eksende",
            f"eksen = np.linspace({low}, {high}, 401)",
            "fig, ax = plt.subplots(figsize=(8, 5))",
        ]
        for index, (distribution, a, b, label) in enumerate(op.curves):
            curve = density_expression(distribution, a, b, "eksen")
            lines.append(f'ax.plot(eksen, {curve}, color="{PALETTE[index % len(PALETTE)]}", linewidth=2, '
                         f"label={text(label)})")
        limit = "bottom=0" if op.y_max is None else f"0, {E.format_number(op.y_max)}"
        lines += [f"ax.set_xlim({low}, {high})", f"ax.set_ylim({limit})"]
        return lines + self._axes(op.x_label, op.y_label, op.title, legend=True)

    def _pmf_density(self, op: PmfWithDensity) -> list[str]:
        frame, x = op.frame, f'{op.frame}["{op.x}"]'
        lines = [
            "fig, ax = plt.subplots(figsize=(8, 5))",
            f'ax.bar({x}, {frame}["{op.y}"], width=0.6, color="{PALETTE[0]}", alpha=0.8, label={text(op.bar_label)})',
            f"eksen = np.linspace({x}.min() - 0.5, {x}.max() + 0.5, 401)",
            f'ax.plot(eksen, {density_expression(op.distribution, op.first, op.second, "eksen")}, '
            f'color="{PALETTE[1]}", linewidth=2,',
            f"        label={text(op.curve_label)})",
        ]
        if op.shade:
            pairs = ", ".join(f"({E.format_number(a)}, {E.format_number(b)})" for a, b in op.shade)
            curve = density_expression(op.distribution, op.first, op.second, "xa")
            lines += [
                f"for alt, ust in [{pairs}]:  # sürekli yaklaşımda tam sayı değerinin alanı",
                "    xa = np.linspace(alt, ust, 200)",
                f'    ax.fill_between(xa, {curve}, color="{PALETTE[1]}", alpha=0.3)',
            ]
        lines.append("ax.set_ylim(bottom=0)")
        return lines + self._axes(op.x_label, op.y_label, op.title, legend=True)

    def _monte_carlo(self, op: MonteCarlo) -> list[str]:
        lines = [
            f"# {op.comment}",
            f"# {op.reps} tekrar; rastgele sayı üreteci döngüden önce bir kez tohumlanır",
            *(["# Her tekrarda bir EKK tahmini yapılır; döngü bilgisayara göre birkaç saniye ile bir dakika sürebilir."]
              if any(isinstance(inner, OLS) for inner in op.body) else []),
            f"rng = np.random.default_rng({op.seed})",
            "sonuclar = []",
            f"for tekrar in range({op.reps}):",
        ]
        self.quiet = True
        try:
            for inner in op.body:
                lines += [f"    {line}" if line else "" for line in self.operation(inner)]
        finally:
            self.quiet = False
        dialect = self.dialect("")
        lines.append("    sonuclar.append({")
        for name, expression in op.collect:
            # float(): np.where gibi işlemler tek bir sayı için 0 boyutlu dizi döndürür; describe() onu atlar.
            lines.append(f'        "{name}": float({_render(expression, dialect)}),')
        return lines + ["    })", f"{op.result} = pd.DataFrame(sonuclar)", f"print({op.result}.describe().round(3))"]

    def _summary_table(self, op: SummaryTable) -> list[str]:
        sources = f"{op.result}_kaynak"
        lines = [f"# {op.comment}", f"{sources} = {{  # satır etiketi: kaynak tablo"]
        lines += [f"    {text(label)}: {table}," for label, table in op.rows]
        lines += ["}", f"{op.result} = pd.DataFrame({{", "    etiket: {"]
        for name, source, stat in op.columns:
            note = "  # payda n − 1" if stat in ("std", "var") else ""
            lines.append(f'        "{name}": {_stat_call(f"tablo[{text(source)}]", stat)},{note}')
        digits = dict(op.column_decimals)
        rounding = (f"{{{', '.join(f'{text(name)}: {digits.get(name, op.decimals)}' for name, _, _ in op.columns)}}}"
                    if digits else str(op.decimals))
        return lines + [
            "    }",
            f"    for etiket, tablo in {sources}.items()",
            "}).T  # her kaynak tablo bir satır",
            f"print({op.result}.round({rounding}))",
        ]

    # --- Notlarla karşılaştırma -----------------------------------------
    def target(self, target) -> str:
        if isinstance(target, StatTarget):
            source = (f'{target.frame}["{target.variable}"]' if target.where is None
                      else f'{target.frame}.loc[{_where(target.frame, target.where)}, "{target.variable}"]')
            return _stat_call(source, target.stat)
        if isinstance(target, ScalarTarget):
            return target.name
        if isinstance(target, TableTarget):
            return f"{target.table}.loc[{text(target.row)}, {text(target.column)}]"
        if isinstance(target, CellTarget):
            return f'{target.frame}["{target.column}"].iloc[{target.row - 1}]'
        if isinstance(target, CoefTarget):
            return coef_expression(target.model, target.term, target.quantity)
        if isinstance(target, ModelTarget):
            return model_expression(target.model, target.quantity)
        raise TypeError(f"Tanınmayan hedef: {type(target).__name__}")

    def check_lines(self, checks: tuple[Check, ...]) -> list[str]:
        lines = ['print("Notlarla karşılaştırma:")']
        for check in checks:
            expected = f"{check.expected:.{check.decimals}f}"
            lines.append(
                f'kontrol_et("{_quote(check.label)}", {self.target(check.target)}, {expected}, {check.decimals})'
            )
        return lines

    def closing(self) -> list[str]:
        return [f'print("\\n{_quote(self.closing_message())}")']
