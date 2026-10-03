"""Konu 0 genel uygulaması: veri tablosu, merkez ve yayılım, birlikte hareket, yüzde ve logaritma, koşullu ortalama.

Notlardaki dokuz adım (§0.1, §0.3–§0.9) aynı numaralarla, verisi değiştirilebilir biçimde yazılır. Alternatif örnek
WAGE2'dir (935 erkek çalışan, 1980; ücret burada aylık kazançtır, ABD doları); Adım 4'ün negatif örneği ve Adım 6'nın
oranları OKUN'dan (ABD, 1959–2005) gelir, Tablo 0.2'nin karşılığı küçük bir sayı örneğidir. "Kendi verini yükle"
seçeneğinde aynı adımlar öğrencinin dosyasıyla kurulur. Notlardaki uygulama (``core.labs.konu00``) değişmez.

Etkileşim notlardaki gibidir: göstergenin kodlaması (Adım 1), özetlenen değişken (Adım 2–3), ölçek ve kayma (Adım 3),
örüntü (Adım 4), ikinci değişken (Adım 5), başlangıç ve yeni değer (Adım 6), koşul (Adım 7), açıklayıcı değişken
(Adım 9). Seçeneklerin değerleri veriden gelir; varsayılan seçim örneğin temel spesifikasyonudur.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd

from core import wooldridge_data as W
from core.labs import expr as E
from core.labs import kendi_veri as K
from core.labs.kurgusal_veri import PROGRAM_COLUMNS, PROGRAM_ROWS
from core.labs.ornek import (
    EXACT_FIT_NOTE,
    Case,
    CustomLab,
    Role,
    TopicVariants,
    exact_fit,
    free_name,
    kisa,
    md,
    sayi,
    sayim,
    stable_checks,
    usable_pair,
    with_app_values,
)
from core.labs.spec import (
    INTERCEPT,
    OLS,
    BarChart,
    CellTarget,
    Check,
    Choice,
    CoefTarget,
    CompleteCases,
    Derive,
    DotPlot,
    GroupStats,
    InlineData,
    JoinColumns,
    LabSpec,
    LabStep,
    LineChart,
    LoadWooldridge,
    MapCodes,
    ModelTarget,
    ModelValue,
    NoteRef,
    NumberChoice,
    Operation,
    PairStatistic,
    RegressionTable,
    Scalar,
    ScalarTable,
    ScalarTarget,
    ScatterPlot,
    Shape,
    ShowFrame,
    ShowModel,
    Statistic,
    Support,
    TableTarget,
    TakeRows,
    VariableTypes,
    interactive_step,
)
from core.labs.wording import at_zero, directions, signed_difference, tr_lower

TOPIC = "konu00"
TITLE = "Uygulama: Veri, Notasyon ve Temel İstatistik"
SONUC, ACIKLAYICI, GOSTERGE, KOSUL = "sonuc", "aciklayici", "gosterge", "kosul"
MAX_LEVELS = 25
"""Koşullu ortalama alınan değişkenin en çok farklı değer sayısı (grafikte her değer bir sütun)."""
EXCERPT = "ornek"
"""Adım 1–3'ün beş gözlemlik veri tablosu."""

ALT_DATA, OKUN = "wage2", "okun"
ALT_ROWS = (1, 188, 375, 562, 749)
"""WAGE2'den sistematik örnek: 1. gözlemden başlayarak her 187. çalışan (935 = 5 × 187)."""
ALT_REGRESSORS = ("educ", "exper", "tenure", "IQ")
TWO_SETS = (("gozlem", "A", "B"), tuple((i + 1, a, b) for i, (a, b) in enumerate(zip((20, 22, 24, 26, 28),
                                                                                      (12, 18, 24, 30, 36)))))
"""Tablo 0.2'nin karşılığı: ortalaması aynı (24), yayılımı farklı iki küçük veri seti (kavramsal, veri değil)."""
QUADRATIC_X = tuple((float(x),) for x in range(9))
"""Karesel örnek: x = 0, 1, …, 8 ve y = 8 − 0,5·(x − 4)²; ilişki tam ama doğrusal değil (simetri: r = 0)."""
PERCENT_BASE = 100
"""Tam yüzde değişim ile log farkını karşılaştıran eğride başlangıç değeri (x₀ = 100, notlardaki gibi)."""

_ALT_PHRASES = {
    "educ": "eğitimi bir yıl daha uzun olan",
    "exper": "iş deneyimi bir yıl daha fazla olan",
    "tenure": "mevcut işverendeki kıdemi bir yıl daha fazla olan",
    "IQ": "IQ puanı bir puan daha yüksek olan",
}
_ALT_UNITS = {"wage": "dolar", "educ": "yıl", "exper": "yıl", "tenure": "yıl", "IQ": "puan"}
"""Metinlerdeki kısa birimler (ör. "x̄ = 13,20 yıl")."""
_ALT_PHRASES_IN_TEXT = {"wage": "aylık kazanç", "educ": "eğitim", "exper": "iş deneyimi",
                        "tenure": "mevcut işverendeki kıdem", "IQ": "IQ puanı"}
"""Cümle içindeki adlar (küçük harf; "IQ" kısaltması korunur)."""
_ALT_POSSESSIVE = {"wage": "aylık kazancı"}
"""Bağımlı değişkenin iyelik ekli adı ("tahmin edilen aylık kazancı")."""


# --- Veri ve adlar -------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Pattern:
    """Adım 4'teki bir örüntü: bir saçılım grafiği ve korelasyonu."""

    key: str
    label: str
    frame: str
    x: str
    y: str
    x_label: str
    y_label: str
    title: str
    setup: tuple[Operation, ...] = ()
    kind: str = ""
    """Alternatif örnekte örüntünün türü ("pozitif", "negatif", "zayif", "karesel"); kendi verinde boş."""
    points: int = 0
    """Grafikteki nokta sayısı: çok noktalı grafikte noktalar küçük ve saydam çizilir."""


def _display(case: Case, column: str) -> str:
    """Tablo, eksen ve seçeneklerdeki ad: etiket ve (varsa) birim."""

    unit = case.units.get(column, "")
    return f"{case.name(column)} ({unit})" if unit else case.name(column)


def _phrase(case: Case, column: str) -> str:
    """Cümle içindeki ad: alternatif örnekte küçük harfle ("aylık kazanç"), kendi verinde tırnak içinde."""

    if case.own:
        return f"“{md(case.name(column))}”"
    return dict(case.extra.get("phrases", {})).get(column) or tr_lower(case.name(column))


def _capital(text: str) -> str:
    """Cümle başı: yalnız ilk harf büyür ("aylık kazanç" → "Aylık kazanç"; "IQ puanı" olduğu gibi kalır)."""

    if not text:
        return text
    first = {"i": "İ", "ı": "I"}.get(text[0], text[0].upper())
    return first + text[1:]


def _unit(case: Case, column: str) -> str:
    unit = dict(case.extra.get("short_units", {})).get(column, "")
    return f" {unit}" if unit else ""


def _scalar(name: str, label: str, decimals: int = 2) -> Check:
    """Kontrol: beklenen değer uygulamanın hesabıyla doldurulur (``with_app_values``)."""

    return Check(label, ScalarTarget(name), 0.0, decimals)


def _signed(value: float, decimals: int = 0) -> str:
    if value == 0:
        return ""
    return f" {'+' if value > 0 else '−'} {sayi(abs(value), decimals)}"


def _complete(case: Case, frame: str, columns: tuple[str, ...], name: str, comment: str):
    """``columns`` sütunlarından birinde eksik değer varsa tam gözlemlerin çerçevesi: (çerçeve adı, işlemler)."""

    if case.data[list(columns)].isna().any().any():
        return name, (CompleteCases(name, frame, columns, comment),)
    return frame, ()


def _numeric_options(case: Case, columns) -> tuple[tuple[str, str], ...]:
    return tuple((column, _display(case, column)) for column in columns)


def _levels(case: Case) -> tuple[float, ...]:
    """Koşul değişkeninin değerleri (küçükten büyüğe); koşul yoksa boş."""

    if not case.has(KOSUL):
        return ()
    values = case.data[case.roles[KOSUL]].dropna().unique()
    return tuple(sorted(float(value) for value in values))


def _level_text(value: float) -> str:
    return kisa(value, 6)


def _level_value(case: Case, value: float):
    """Koşulun koddaki değeri: tam sayı sütunda tam sayı (12), kesirli sütunda sayı (2.5)."""

    column = case.roles[KOSUL]
    if pd.api.types.is_integer_dtype(case.data[column]):
        return int(value)
    return int(value) if float(value).is_integer() else float(value)


# --- Adım 1: veri tablosunun anatomisi ----------------------------------------------------------------------

def _excerpt_columns(case: Case) -> tuple[str, ...]:
    columns = [case.roles[SONUC], case.roles[ACIKLAYICI]]
    columns += [column for column in case.extra.get("excerpt_extras", case.extras) if column not in columns]
    if case.has(GOSTERGE):
        columns.append(case.roles[GOSTERGE])
    return tuple(dict.fromkeys(columns))


def _indicator(case: Case) -> dict:
    return dict(case.extra["indicator"])


def _code_column(case: Case) -> str | None:
    """Gösterge değişkeninin 0/1 sütunu: alternatif örnekte verideki sütun, kendi verinde türetilen kod sütunu."""

    if not case.has(GOSTERGE):
        return None
    if _indicator(case).get("numeric"):
        return case.roles[GOSTERGE]
    return free_name(f"{case.roles[GOSTERGE]}_kod", set(case.data.columns) | {"gozlem"})


def _indicator_choice(case: Case) -> Choice:
    info = _indicator(case)
    return Choice("adim1_gosterge", "Gösterge değişkeninin kodlaması",
                  (("bir", f"{info['one']} = 1"), ("diger", f"{info['zero']} = 1")), "bir",
                  help=f"Varsayılan: {info['one']} = 1, {info['zero']} = 0.")


def _type_rows(case: Case, code: str | None) -> tuple[tuple[str, str, str], ...]:
    origin = "WAGE2'deki gözlem numarası" if not case.own else "Temizlenmiş verideki sıra"
    rows = [("gozlem", "Kimlik", f"{origin}; analitik değişken sayılmaz")]
    for column in _excerpt_columns(case):
        if case.has(GOSTERGE) and column == case.roles[GOSTERGE] and not _indicator(case).get("numeric"):
            rows.append((column, "Kategorik", "İki kategori"))
        elif case.has(GOSTERGE) and column == case.roles[GOSTERGE]:
            info = _indicator(case)
            rows.append((column, "Gösterge (0/1)", f"{info['one']} = 1, {info['zero']} = 0"))
        else:
            rows.append((column, "Nicel", _display(case, column)))
    if code is not None and code != case.roles.get(GOSTERGE):
        info = _indicator(case)
        rows.append((code, "Gösterge (0/1)", f"{info['one']} = 1, {info['zero']} = 0"))
    return tuple(rows)


def _step1(case: Case) -> LabStep:
    columns = _excerpt_columns(case)
    rows = tuple(case.extra["excerpt_rows"])
    code = _code_column(case)

    def build(choices) -> tuple:
        operations: list = [
            *case.load,
            TakeRows(EXCERPT, case.frame, str(case.extra["excerpt_comment"]), rows=rows, columns=columns,
                     number="gozlem"),
            Shape(EXCERPT, "n", "k", exclude=("gozlem",)),
        ]
        if case.has(GOSTERGE) and code != case.roles[GOSTERGE]:
            info = _indicator(case)
            operations.append(MapCodes(EXCERPT, case.roles[GOSTERGE], code, ((info["zero"], 0), (info["one"], 1)),
                                       f"Gösterge değişkeni: {info['zero']} = 0, {info['one']} = 1"))
        operations += [
            VariableTypes(EXCERPT, _type_rows(case, code), "turler"),
            ShowFrame(EXCERPT, ("gozlem", *columns, *([code] if code and code not in columns else [])),
                      "Her satır bir gözlem, her sütun bir değişken"),
        ]
        if case.has(GOSTERGE):
            info = _indicator(case)
            operations.append(Statistic(EXCERPT, code, "mean", "gosterge_payi",
                                        f"{info['one']} payı: gösterge ortalaması", decimals=2))
            if choices.get("adim1_gosterge") == "diger":
                operations.append(Scalar("diger_payi", E.sub(1, E.ref("gosterge_payi")),
                                         f"{info['zero']} payı: 1 − {tr_lower(info['one'])} payı", decimals=2))
        return tuple(operations)

    def note(state, choices) -> str:
        k = int(state.scalars["k"])
        text = (f"Beş gözlem ve {k} analitik değişken vardır; gözlem numarası gözlemi tanımlar, ölçüm değildir "
                "(§0.1).")
        if not case.has(GOSTERGE):
            return text + (" Dosyanızda iki kategorili bir sütun (ör. evet/hayır) varsa “İki kategorili gösterge” "
                           "rolüne seçin: gösterge değişkeninin ortalamasının bir pay olduğunu da görürsünüz.")
        info = _indicator(case)
        one, zero = md(info["one"]), md(info["zero"])
        share = state.scalars["gosterge_payi"]
        count = int(round(5 * share))
        if choices.get("adim1_gosterge") != "diger":
            return text + (f" Gösterge değişkeninin ortalaması 1 ile kodlanan grubun örneklemdeki payıdır: "
                           f"{sayi(share, 2)}. Beş gözlemden {count} tanesinde gösterge 1 değerini alır ({one}). "
                           f"Kodlamayı çevirip ({zero} = 1) payın nasıl değiştiğine bakın.")
        other = state.scalars["diger_payi"]
        return text + (f" Ters kodlamada gösterge 1 − ilk göstergedir: {one} olan gözlemde 0, {zero} olan gözlemde 1 "
                       f"olur. Ortalaması {zero} grubunun payıdır ({sayi(other, 2)}) ve öteki grubun payıyla "
                       "toplamı 1 eder. Payın yorumu hangi grubun 1 ile kodlandığına bağlıdır; bu yüzden kodlama "
                       "her zaman açıkça yazılır.")

    checks = [_scalar("n", "Gözlem sayısı n", 0), _scalar("k", "Analitik değişken sayısı", 0)]
    if case.has(GOSTERGE):
        checks.append(_scalar("gosterge_payi", "Gösterge ortalaması (1 ile kodlanan grubun payı)", 2))
    common = dict(number=1, title="Veri tablosunun anatomisi", note=NoteRef("0.1", objects=("Tablo 0.1",)),
                  explanation=str(case.extra["excerpt_text"]), checks=tuple(checks),
                  note_for=lambda state, choices: note(state, choices))
    if case.has(GOSTERGE):
        return interactive_step(controls=(_indicator_choice(case),), build=build, **common)
    return LabStep(operations=build({}), **common)


# --- Adım 2 ve 3: toplam, ortalama, sapma ve varyans -------------------------------------------------------------

def _summary_columns(case: Case) -> tuple[str, ...]:
    """Beş gözlemde boş hücresi olmayan sayısal sütunlar (sonuç, açıklayıcı, ek değişkenler)."""

    excerpt = case.data.iloc[[row - 1 for row in case.extra["excerpt_rows"]]]
    candidates = [column for column in _excerpt_columns(case) if column != case.roles.get(GOSTERGE)]
    return tuple(column for column in candidates if excerpt[column].notna().all())


def _variable_choice(case: Case) -> Choice:
    return Choice("adim2_degisken", "Özetlenen değişken (beş gözlem)",
                  _numeric_options(case, _summary_columns(case)), case.roles[ACIKLAYICI],
                  help="Varsayılan: açıklayıcı değişken.")


def _excerpt_values(case: Case, column: str) -> list[float]:
    return [float(case.data[column].iloc[row - 1]) for row in case.extra["excerpt_rows"]]


def _step2(case: Case) -> LabStep:
    def build(choices) -> tuple:
        variable = choices["adim2_degisken"]
        return (
            ShowFrame(EXCERPT, ("gozlem", variable), "xᵢ: i numaralı gözlemin değeri"),
            Statistic(EXCERPT, variable, "sum", "toplam", "Toplam Σxᵢ", decimals=2),
            Statistic(EXCERPT, variable, "count", "n_gozlem", "Gözlem sayısı n", decimals=0),
            Scalar("ortalama", E.div(E.ref("toplam"), E.ref("n_gozlem")), "Ortalama x̄ = Σxᵢ / n", decimals=2),
            Statistic(EXCERPT, variable, "mean", "ortalama_yazilim", "Yazılımın ortalama fonksiyonu", decimals=2),
        )

    def note(state, choices) -> str:
        variable = choices["adim2_degisken"]
        values = _excerpt_values(case, variable)
        written = " + ".join(kisa(value, 4) for value in values)
        total, mean = state.scalars["toplam"], state.scalars["ortalama"]
        text = (f"Σxᵢ = {written} = {kisa(total, 4)}; x̄ = {kisa(total, 4)} / 5 = {sayi(mean, 2)}"
                f"{_unit(case, variable)}.")
        if not any(abs(value - mean) < 1e-9 for value in values):
            text += (" Ortalama veride gözlenen bir değer olmak zorunda değildir: hiçbir gözlemin değeri tam olarak "
                     "ortalamaya eşit değildir.")
        return text + " Ortalama merkezi özetler, yayılımı göstermez; bunu bir sonraki adımdaki varyans tamamlar (§0.3)."

    return interactive_step(
        number=2,
        title="Toplam sembolü ve aritmetik ortalama",
        note=NoteRef("0.3"),
        explanation=(
            "$x_i$, $x$ değişkeninin $i$ numaralı gözlemdeki değeridir. Toplam sembolü "
            "$\\sum_{i=1}^{n} x_i = x_1 + x_2 + \\cdots + x_n$ bütün gözlemlerin toplamını, örneklem ortalaması "
            "$\\bar{x} = \\frac{1}{n}\\sum_{i=1}^{n} x_i$ bu toplamın gözlem sayısına bölümünü verir. Hesap Adım 1'deki "
            "beş gözlemle yapılır."
        ),
        controls=(_variable_choice(case),),
        build=build,
        checks=(
            _scalar("toplam", "Toplam Σxᵢ", 2),
            _scalar("ortalama", "Ortalama x̄", 2),
            _scalar("ortalama_yazilim", "Yazılımla ortalama", 2),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


SCALE_CHOICE = NumberChoice("adim3_a", "Ölçek a: B'nin değerleri a ile çarpılır", -3, 3, 1, 0.5, decimals=1,
                            help="yᵢ = a·Bᵢ + c; Bᵢ, iki veri setinden B'nin değerleridir. Varsayılan: a = 1, c = 0 "
                                 "(B'nin kendisi).")
SHIFT_CHOICE = NumberChoice("adim3_c", "Kayma c: B'nin değerlerine c eklenir", -10, 20, 0, 1, integer=True,
                            decimals=0, help="Kayma ortalamayı değiştirir, yayılımı değiştirmez.")


def _affine_expr(a: float, c: int) -> E.Expr:
    scaled = E.mul(a, E.var("B"))
    if c > 0:
        return E.add(scaled, c)
    return E.sub(scaled, -c) if c < 0 else scaled


def _affine(a: float, c: int) -> str:
    return f"y = {sayi(a, 1)}·B{_signed(c)}"


def _dot_range(a: float, c: int) -> tuple[float, float]:
    """İki nokta grafiğinin ortak ekseni: A, B ve y = a·B + c görünür; uçlarda en az bir birim pay."""

    values = [value for _, first, second in TWO_SETS[1] for value in (first, second)]
    low_value, high_value = min(values), max(values)
    ends = (a * low_value + c, a * high_value + c)
    low = math.floor((min(low_value, *ends) - 1) / 5) * 5
    high = math.ceil((max(high_value, *ends) + 1) / 5) * 5
    return low, high


def _two_set_mean() -> float:
    return float(np.mean([b for _, _, b in TWO_SETS[1]]))


def _step3(case: Case) -> LabStep:
    variable_choice = _variable_choice(case)

    def build(choices) -> tuple:
        variable = choices["adim2_degisken"]
        a, c = round(float(choices["adim3_a"]), 2), int(choices["adim3_c"])
        return (
            Derive(EXCERPT, "sapma", E.sub(E.var(variable), E.ref("ortalama")), "Sapma xᵢ − x̄"),
            Derive(EXCERPT, "sapma_kare", E.power(E.var("sapma"), 2), "Kareli sapma (xᵢ − x̄)²"),
            ShowFrame(EXCERPT, ("gozlem", variable, "sapma", "sapma_kare"), "Sapmalar ve kareli sapmalar"),
            Statistic(EXCERPT, "sapma", "sum", "sapma_toplami", "Sapmaların toplamı", decimals=2),
            Statistic(EXCERPT, "sapma_kare", "sum", "kare_toplami", "Kareli sapmaların toplamı", decimals=2),
            Scalar("varyans", E.div(E.ref("kare_toplami"), E.sub(E.ref("n_gozlem"), 1)),
                   "Varyans s² = toplam / (n − 1)", decimals=2),
            Scalar("std_sapma", E.sqrt(E.ref("varyans")), "Standart sapma s = √s²", decimals=2),
            Statistic(EXCERPT, variable, "var", "varyans_yazilim", "Yazılımla varyans (payda n − 1)", decimals=2),
            InlineData("iki_veri", TWO_SETS[0], TWO_SETS[1],
                       "Tablo 0.2'nin karşılığı: aynı ortalamaya sahip iki küçük veri seti (sayı örneği)"),
            Statistic("iki_veri", "A", "mean", "ort_A", "A: ortalama", decimals=2),
            Statistic("iki_veri", "B", "mean", "ort_B", "B: ortalama", decimals=2),
            Statistic("iki_veri", "A", "var", "var_A", "A: varyans s²", decimals=2),
            Statistic("iki_veri", "B", "var", "var_B", "B: varyans s²", decimals=2),
            Statistic("iki_veri", "A", "std", "std_A", "A: standart sapma s", decimals=2),
            Statistic("iki_veri", "B", "std", "std_B", "B: standart sapma s", decimals=2),
            Derive("iki_veri", "y", _affine_expr(a, c), f"Dönüştürülmüş B: {_affine(a, c)}"),
            Statistic("iki_veri", "y", "mean", "ort_y", "y: ortalama", decimals=2),
            Statistic("iki_veri", "y", "std", "std_y", "y: standart sapma", decimals=2),
            ScalarTable(
                (
                    ("A: ortalama", E.ref("ort_A")),
                    ("A: varyans s²", E.ref("var_A")),
                    ("A: standart sapma s", E.ref("std_A")),
                    ("B: ortalama", E.ref("ort_B")),
                    ("B: varyans s²", E.ref("var_B")),
                    ("B: standart sapma s", E.ref("std_B")),
                    ("y = a·B + c: ortalama", E.ref("ort_y")),
                    ("y = a·B + c: standart sapma", E.ref("std_y")),
                ),
                "yayilim",
                decimals=2,
            ),
            DotPlot("iki_veri", "A", "Değer", "Veri seti A", references=(("ort_A", "Ortalama"),),
                    x_range=_dot_range(a, c)),
            DotPlot("iki_veri", "y", "Değer", "Veri seti B" if (a, c) == (1, 0) else f"Dönüştürülmüş B: {_affine(a, c)}",
                    references=(("ort_y", "Ortalama"),), x_range=_dot_range(a, c)),
        )

    def note(state, choices) -> str:
        s = state.scalars
        a, c = round(float(choices["adim3_a"]), 2), int(choices["adim3_c"])
        variable = choices["adim2_degisken"]
        unit = _unit(case, variable).strip()
        squared = f"({unit})²" if unit else "değişkenin biriminin karesi"
        text = (f"Sapmaların toplamı sıfırdır ({sayi(s['sapma_toplami'], 2)}); kareli sapmaların toplamı "
                f"{kisa(s['kare_toplami'], 4)}, varyans {sayi(s['varyans'], 2)} ve standart sapma "
                f"{sayi(s['std_sapma'], 2)}. Varyansın birimi {squared}, standart sapmanın birimi değişkenin kendi "
                "birimidir. ")
        text += (f"İki veri setinde A ve B'nin ortalaması aynıdır ({sayi(s['ort_A'], 0)}); standart sapmalar "
                 f"{sayi(s['std_A'], 2)} ve {sayi(s['std_B'], 2)}: B daha geniş yayılır. ")
        if a == 1 and c == 0:
            return text + ("Ölçek a ve kayma c ile yᵢ = a·Bᵢ + c dönüşümünde ȳ = a·B̄ + c ve s_y = |a|·s_B kurallarını "
                           "deneyin (§0.4).")
        spread = "0" if a == 0 else f"{sayi(abs(a), 1)}·{sayi(s['std_B'], 4)} ≈ {sayi(s['std_y'], 2)}"
        text += (f"{_affine(a, c)}: ortalama ȳ = {sayi(a, 1)}·{sayi(_two_set_mean(), 0)}{_signed(c)} = "
                 f"{sayi(s['ort_y'], 2)}; standart sapma s_y = |a|·s_B = {spread}. ")
        if a == 0:
            text += f"a = 0 iken bütün değerler {sayi(c, 0)} olur: yayılım kalmaz, standart sapma sıfırdır."
        elif abs(a) > 1:
            text += "|a| > 1 olduğu için ölçek yayılımı |a| katına çıkarır."
        elif abs(a) < 1:
            text += "|a| < 1 olduğu için ölçek yayılımı küçültür: standart sapma |a| ile çarpılır."
        else:
            text += "|a| = 1 olduğu için yayılım değişmez."
        if a < 0:
            text += " a < 0 iken değerlerin sırası tersine döner; standart sapma yine negatif olmaz, bu yüzden |a| yazılır."
        if c != 0:
            text += " Kayma c ortalamayı kaydırır, yayılımı değiştirmez."
        return text + " Genel kural: ȳ = a·B̄ + c ve s_y = |a|·s_B (§0.4)."

    a_values = ", ".join(str(first) for _, first, _ in TWO_SETS[1])
    b_values = ", ".join(str(second) for _, _, second in TWO_SETS[1])
    return interactive_step(
        number=3,
        title="Sapma, varyans ve standart sapma",
        note=NoteRef("0.4", objects=("Tablo 0.2",)),
        explanation=(
            "Sapma $x_i - \\bar{x}$; sapmaların toplamı her zaman sıfırdır. Örneklem varyansı "
            "$s_x^2 = \\frac{1}{n-1}\\sum_{i=1}^{n}(x_i - \\bar{x})^2$, standart sapma $s_x = \\sqrt{s_x^2}$; "
            "sapma ve varyans Adım 2'de seçilen değişkenle hesaplanır. Tablo 0.2'nin karşılığında iki küçük veri seti "
            f"vardır: A = {a_values} ve B = {b_values}. Ortalamaları aynı, yayılımları farklıdır. Ölçek $a$ ve kayma "
            "$c$, B'nin değerlerini $y_i = a\\,B_i + c$ ile dönüştürür."
        ),
        uses=(variable_choice,),
        controls=(SCALE_CHOICE, SHIFT_CHOICE),
        build=build,
        checks=(
            *(Check(f"{i}. gözlemin sapması", CellTarget(EXCERPT, "sapma", i), 0.0, 2) for i in range(1, 6)),
            *(Check(f"{i}. gözlemin kareli sapması", CellTarget(EXCERPT, "sapma_kare", i), 0.0, 2)
              for i in range(1, 6)),
            _scalar("sapma_toplami", "Sapmaların toplamı", 0),
            _scalar("kare_toplami", "Kareli sapmaların toplamı", 2),
            _scalar("varyans", "Varyans s²", 2),
            _scalar("std_sapma", "Standart sapma s", 2),
            _scalar("ort_A", "A: ortalama", 2),
            _scalar("ort_B", "B: ortalama", 2),
            _scalar("var_A", "A: varyans", 2),
            _scalar("var_B", "B: varyans", 2),
            _scalar("std_A", "A: standart sapma", 2),
            _scalar("std_B", "B: standart sapma", 2),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 4: korelasyonun yönü ve doğrusallık ------------------------------------------------------------------

def _quadratic() -> Pattern:
    setup = (
        InlineData("karesel", ("x",), QUADRATIC_X, "Karesel örnek: x = 0, 1, …, 8", layout=9),
        Derive("karesel", "y", E.sub(8, E.mul(0.5, E.power(E.sub(E.var("x"), 4), 2))), "y = 8 − 0,5·(x − 4)²"),
    )
    return Pattern("karesel", "Karesel ilişki: y = 8 − 0,5·(x − 4)²", "karesel", "x", "y", "x", "y = 8 − 0,5·(x − 4)²",
                   "Kusursuz ama doğrusal olmayan ilişki: r = 0", setup=setup, kind="karesel", points=len(QUADRATIC_X))


def _own_patterns(case: Case) -> tuple[Pattern, ...]:
    """Kendi verinde sütun çiftleri: sonuç ile her sayısal değişken ve açıklayıcı ile her ek değişken; korelasyonu
    tanımlı olmayan (sabit sütunlu) çiftler alınmaz."""

    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    pairs = [(x, y), *((extra, y) for extra in case.extras), *((x, extra) for extra in case.extras)]
    patterns = []
    for index, (first, second) in enumerate(pairs, start=1):
        if not usable_pair(case.data, first, second):
            continue
        complete = case.data[[first, second]].dropna()
        frame, setup = _complete(case, case.frame, (first, second), f"cift{index}",
                                 f"Tam gözlemler: {case.name(first)} ve {case.name(second)}")
        patterns.append(Pattern(
            f"cift{index}", f"{case.name(first)} ve {case.name(second)}", frame, first, second,
            _display(case, first), _display(case, second), f"{case.name(first)} ve {case.name(second)}", setup=setup,
            points=len(complete),
        ))
    return tuple(patterns) + (_quadratic(),)


def _patterns(case: Case) -> tuple[Pattern, ...]:
    return tuple(case.extra["patterns"]) if "patterns" in case.extra else _own_patterns(case)


def _pattern_names(case: Case, pattern: Pattern) -> tuple[str, str]:
    """Örüntünün iki değişkeninin cümle içindeki adları (alternatif örnekte OKUN değişkenleri de)."""

    if case.own:
        return _phrase(case, pattern.x), _phrase(case, pattern.y)
    names = dict(case.extra.get("pattern_phrases", {}))
    return (names.get(pattern.x) or _phrase(case, pattern.x)), (names.get(pattern.y) or _phrase(case, pattern.y))


def _strength(r: float) -> str:
    size = abs(r)
    if size < 0.1:
        return "çok zayıf"
    if size < 0.3:
        return "zayıf"
    if size < 0.7:
        return "orta güçte"
    return "güçlü"


def _step4(case: Case) -> LabStep:
    patterns = _patterns(case)
    by_key = {pattern.key: pattern for pattern in patterns}
    choice = Choice("adim4_desen", "Gösterilen örüntü", tuple((pattern.key, pattern.label) for pattern in patterns),
                    patterns[0].key, help="Seçenekler gerçek veri çiftleri ve karesel bir sayı örneğidir.")

    def build(choices) -> tuple:
        chosen = by_key[choices["adim4_desen"]]
        operations: list = []
        rows = []
        for pattern in patterns:
            operations += [op for op in pattern.setup if op not in operations]
            operations.append(PairStatistic(pattern.frame, pattern.x, pattern.y, "corr", f"r_{pattern.key}",
                                            f"{pattern.label}: r", decimals=3))
            rows.append((pattern.label, E.ref(f"r_{pattern.key}")))
        operations.append(ScalarTable(tuple(rows), "korelasyonlar", decimals=3, heading="Örüntü", value="r"))
        large = chosen.points > 100
        operations.append(ScatterPlot(chosen.frame, chosen.x, chosen.y, chosen.x_label, chosen.y_label, chosen.title,
                                      size=7 if large else 11, opacity=0.35 if large else 1.0))
        return tuple(operations)

    def note(state, choices) -> str:
        chosen = by_key[choices["adim4_desen"]]
        r = state.scalars[f"r_{chosen.key}"]
        if chosen.kind == "karesel":
            return (f"r = {sayi(r, 3)}: y, x ile tam olarak belirlenir, ama ilişki doğrusal değildir (önce artar, "
                    "sonra azalır). Korelasyon yalnız doğrusal ilişkiyi ölçer; r = 0 “ilişki yok” demek değildir "
                    "(§0.5). Bu yüzden korelasyonu yorumlamadan önce saçılım grafiğine bakmak yararlıdır.")
        strength = _strength(r)
        if strength == "çok zayıf":
            return (f"r = {sayi(r, 3)}: noktalar belirgin bir doğru çevresinde toplanmıyor; doğrusal ilişki çok zayıf. "
                    "İşaret ilişkinin yönünü, mutlak değer doğrusal ilişkinin gücünü gösterir (§0.5). Doğrusal "
                    "ilişkinin zayıf olması değişkenlerin hiç ilişkili olmadığı anlamına gelmez.")
        first, second = _pattern_names(case, chosen)
        direction = f"{first} arttıkça {second} çoğunlukla {'artar' if r > 0 else 'azalır'}"
        return (f"r = {sayi(r, 3)}: {direction}; doğrusal ilişki {strength}. Korelasyon ilişkinin yönünü ve doğrusal "
                "gücünü özetler; nedenselliği göstermez (§0.5).")

    if case.own:
        intro = ("Seçenekler dosyanızdaki sayısal sütun çiftleri ve karesel bir sayı örneğidir; tabloda her çiftin "
                 "korelasyonu vardır. Eksik değerli çiftlerde yalnız iki değeri de olan gözlemler kullanılır.")
    else:
        intro = str(case.extra["patterns_text"])
    return interactive_step(
        number=4,
        title="Korelasyonun yönü ve doğrusallık",
        note=NoteRef("0.5", objects=("Şekil 0.2",)),
        explanation=(
            "Kovaryans $s_{xy} = \\frac{1}{n-1}\\sum_{i=1}^{n}(x_i - \\bar{x})(y_i - \\bar{y})$ iki değişkenin "
            "birlikte hareketinin yönünü gösterir; korelasyon $r_{xy} = s_{xy}/(s_x s_y)$ onu standartlaştırır ve $-1$ "
            f"ile $1$ arasındadır. {intro}"
        ),
        controls=(choice,),
        build=build,
        checks=tuple(_scalar(f"r_{pattern.key}", f"{pattern.label}: r", 3) for pattern in patterns),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 5: kovaryans, korelasyon ve ölçü birimi ----------------------------------------------------------------

def _second_options(case: Case) -> tuple[str, ...]:
    return tuple(dict.fromkeys((case.roles[ACIKLAYICI], *case.extras)))


def _pair_options(case: Case) -> tuple[str, ...]:
    """Sonuçla birlikte kullanılabilen ikinci değişkenler: iki değeri de olan gözlemlerde ikisi de değişmeli (sabit
    bir değişkenle korelasyon ve eğim tanımsızdır)."""

    y = case.roles[SONUC]
    return tuple(column for column in _second_options(case) if usable_pair(case.data, y, column))


def _scaled(case: Case) -> str:
    return str(case.extra.get("scaled") or free_name(f"{case.roles[SONUC]}_100", case.data.columns))


def _logged(case: Case) -> str | None:
    """Sonucun logaritması: alternatif örnekte verideki sütun, kendi verinde (sonuç pozitifse) türetilen sütun."""

    if "logged" in case.extra:
        return str(case.extra["logged"])
    if (case.data[case.roles[SONUC]] > 0).all():
        return free_name(f"ln_{case.roles[SONUC]}", set(case.data.columns) | {_scaled(case)})
    return None


def _step5(case: Case) -> LabStep:
    y = case.roles[SONUC]
    scaled, logged = _scaled(case), _logged(case)
    choice = Choice("adim5_x", f"İkinci değişken (birinci değişken: {case.name(y)})",
                    _numeric_options(case, _pair_options(case)), case.roles[ACIKLAYICI],
                    help="Varsayılan: açıklayıcı değişken.")
    derived: list = [Derive(case.frame, scaled, E.mul(100, E.var(y)), str(case.extra.get("scaled_comment") or
                                                                         f"{case.name(y)} × 100"))]
    if logged is not None and "logged" not in case.extra:
        derived.append(Derive(case.frame, logged, E.log(E.var(y)), f"{case.name(y)}: doğal logaritma"))

    def build(choices) -> tuple:
        x = choices["adim5_x"]
        frame, complete = _complete(case, case.frame, (y, x), "ikili",
                                    f"Tam gözlemler: {case.name(y)} ve {case.name(x)}")
        operations = [
            *derived,
            *complete,
            PairStatistic(frame, y, x, "cov", "kovaryans", "Kovaryans s_xy", decimals=4),
            Statistic(frame, y, "std", "std_y", f"{case.name(y)}: standart sapma", decimals=4),
            Statistic(frame, x, "std", "std_x", f"{case.name(x)}: standart sapma", decimals=4),
            Scalar("korelasyon", E.div(E.ref("kovaryans"), E.mul(E.ref("std_y"), E.ref("std_x"))),
                   "Korelasyon r = s_xy / (s_x·s_y)", decimals=3),
            PairStatistic(frame, y, x, "corr", "korelasyon_yazilim", "Yazılımla korelasyon", decimals=3),
            PairStatistic(frame, scaled, x, "cov", "kovaryans_100", f"Kovaryans ({str(case.extra.get('scaled_name') or 'sonuç × 100')})",
                          decimals=2),
            PairStatistic(frame, scaled, x, "corr", "korelasyon_100", f"Korelasyon ({str(case.extra.get('scaled_name') or 'sonuç × 100')})",
                          decimals=3),
        ]
        rows_cov = [("Kovaryans, özgün birim", E.ref("kovaryans")),
                    (f"Kovaryans, {str(case.extra.get('scaled_name') or 'sonuç × 100')}", E.ref("kovaryans_100"))]
        rows_corr = [("Korelasyon, özgün birim", E.ref("korelasyon")),
                     (f"Korelasyon, {str(case.extra.get('scaled_name') or 'sonuç × 100')}", E.ref("korelasyon_100"))]
        if logged is not None:
            operations.append(PairStatistic(frame, logged, x, "corr", "korelasyon_log", "Korelasyon, logaritma",
                                            decimals=3))
            rows_corr.append(("Korelasyon, logaritma", E.ref("korelasyon_log")))
        operations += [
            ScalarTable(tuple(rows_cov), "birim_kovaryans", decimals=4),
            ScalarTable(tuple(rows_corr), "birim_korelasyon", decimals=3),
            ScatterPlot(frame, x, y, _display(case, x), _display(case, y),
                        f"{case.name(x)} ve {case.name(y) if case.own else _phrase(case, y)}",
                        size=7 if len(case.data) > 100 else 11, opacity=0.35 if len(case.data) > 100 else 1.0),
        ]
        return tuple(operations)

    def note(state, choices) -> str:
        s = state.scalars
        x = choices["adim5_x"]
        text = (f"{_capital(_phrase(case, y))} ile {_phrase(case, x)} "
                f"arasındaki kovaryans {sayi(s['kovaryans'], 4)}; korelasyon {sayi(s['kovaryans'], 4)} / "
                f"({sayi(s['std_y'], 4)} × {sayi(s['std_x'], 4)}) = {sayi(s['korelasyon'], 3)}. "
                f"{str(case.extra.get('scaled_sentence') or 'Sonuç 100 ile çarpılınca (ör. TL yerine kuruş)')} "
                f"kovaryans 100 katına çıkar ({sayi(s['kovaryans_100'], 2)}), korelasyon değişmez "
                f"({sayi(s['korelasyon_100'], 3)}). ")
        if "korelasyon_log" in s:
            text += (f"Logaritma doğrusal bir dönüşüm değildir: logaritmayla korelasyon {sayi(s['korelasyon_log'], 3)}. ")
        else:
            text += "Sonucun sıfır ya da negatif değerleri olduğu için logaritması alınmadı. "
        complete = "ikili" in state.frames and len(state.frames["ikili"]) < len(case.data)
        if complete:
            text += f"Hesap iki değeri de olan {sayim(len(state.frames['ikili']))} gözlemle yapıldı. "
        return text + ("Kovaryansın işareti ilişkinin yönünü gösterir; büyüklüğü ölçü birimine bağlı olduğu için "
                       "ilişkinin gücü korelasyonla karşılaştırılır (§0.5).")

    checks = [
        _scalar("kovaryans", "Kovaryans s_xy", 4),
        _scalar("std_y", "Sonucun standart sapması", 4),
        _scalar("std_x", "İkinci değişkenin standart sapması", 4),
        _scalar("korelasyon", "Korelasyon r", 3),
        _scalar("korelasyon_yazilim", "Yazılımla korelasyon", 3),
        _scalar("kovaryans_100", "Kovaryans, sonuç × 100", 2),
        _scalar("korelasyon_100", "Korelasyon, sonuç × 100", 3),
    ]
    if logged is not None:
        checks.append(_scalar("korelasyon_log", "Korelasyon, logaritma", 3))
    return interactive_step(
        number=5,
        title=f"Kovaryans, korelasyon ve ölçü birimi{str(case.extra.get('step5_suffix', ''))}",
        note=NoteRef("0.5"),
        explanation=str(case.extra.get("step5_text") or (
            f"Dosyanızdaki {sayim(len(case.data))} gözlem: {_phrase(case, y)} ile ikinci değişken arasındaki kovaryans "
            "ve korelasyon. Sonuç 100 ile çarpılınca kovaryans 100 katına çıkar, korelasyon değişmez; logaritma gibi "
            "doğrusal olmayan bir dönüşüm ise korelasyonu genellikle değiştirir.")),
        controls=(choice,),
        build=build,
        checks=tuple(checks),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 6: yüzde değişim, yüzde puan ve log farkı ---------------------------------------------------------------

def _percent_change(start, end) -> E.Expr:
    return E.mul(100, E.div(E.sub(end, start), start))


def _log_change(start, end) -> E.Expr:
    return E.mul(100, E.sub(E.log(end), E.log(start)))


def _group_means(case: Case) -> dict[float, float]:
    column, y = case.roles[KOSUL], case.roles[SONUC]
    means = case.data.groupby(column)[y].mean()
    return {float(level): float(value) for level, value in means.items()}


def _percent_levels(case: Case) -> tuple[float, ...]:
    """Adım 6'da seçilebilen koşul değerleri: grup ortalaması pozitif olanlar (log farkı tanımlı olsun)."""

    means = _group_means(case)
    return tuple(level for level in _levels(case) if means.get(level, 0) > 0)


def _rate_setup(case: Case) -> dict | None:
    return dict(case.extra["rates"]) if "rates" in case.extra else None


def _table03(case: Case) -> tuple[float, tuple[float, ...]]:
    if "table03" in case.extra:
        base, targets = case.extra["table03"]
        return float(base), tuple(float(value) for value in targets)
    levels = _percent_levels(case)
    targets = sorted({levels[1], levels[len(levels) // 2], levels[-1]} - {levels[0]})
    return levels[0], tuple(targets)


def _percent_controls(case: Case) -> tuple:
    levels = _percent_levels(case)
    options = tuple((_level_text(level), f"{_level_text(level)}{_unit(case, case.roles[KOSUL])}") for level in levels)
    first, second = case.extra.get("percent_defaults", (levels[0], levels[-1]))
    name = case.name(case.roles[KOSUL])
    controls = [
        Choice("adim6_d0", f"Başlangıç: {name}", options, _level_text(first),
               help="x₀: bu değerdeki gözlemlerin ortalama sonucu."),
        Choice("adim6_d1", f"Yeni değer: {name}", options, _level_text(second),
               help="x₁: bu değerdeki gözlemlerin ortalama sonucu."),
    ]
    rates = _rate_setup(case)
    if rates is not None:
        years = tuple((str(year), str(year)) for year in rates["options"])
        controls += [
            Choice("adim6_y0", f"Başlangıç yılı ({rates['label']})", years, str(rates["defaults"][0]),
                   help=f"{rates['label']}: başlangıç oranı."),
            Choice("adim6_y1", f"Yeni yıl ({rates['label']})", years, str(rates["defaults"][1]),
                   help=f"{rates['label']}: yeni oran."),
        ]
    else:
        controls += [
            NumberChoice("adim6_oran0", "Başlangıç oranı (%)", 1, 99, 30, 1, integer=True, decimals=0,
                         help="Sayı örneği: %30'dan %35'e."),
            NumberChoice("adim6_oran1", "Yeni oran (%)", 1, 99, 35, 1, integer=True, decimals=0,
                         help="Sayı örneği: %30'dan %35'e."),
        ]
    return tuple(controls)


def _step6(case: Case) -> LabStep:
    title, note_ref = "Yüzde değişim, yüzde puan ve log farkı", NoteRef("0.6", objects=("Tablo 0.3",))
    formula = ("Yüzde değişim $\\%\\Delta x = 100\\,(x_1 - x_0)/x_0$; küçük değişimlerde $100\\,[\\ln(x_1) - "
               "\\ln(x_0)]$ ona yaklaşır.")
    if not case.has(KOSUL) or len(_percent_levels(case)) < 2:
        need = ("açıklayıcı değişkenin en çok 25 farklı değeri olmalı ya da ayrı bir koşul sütunu seçilmeli"
                if not case.has(KOSUL) else "koşul değişkeninin ortalama sonucu pozitif olan en az iki değeri olmalı")
        return LabStep(number=6, title=title, note=note_ref,
                       explanation=f"{formula} Bu adımda başlangıç ve yeni değer verinizdeki iki grubun ortalama "
                                   f"sonucudur; bunun için {need}.")
    column, y = case.roles[KOSUL], case.roles[SONUC]
    rates = _rate_setup(case)
    base, targets = _table03(case)
    values = {_level_text(level): _level_value(case, level) for level in _levels(case)}
    unit = _unit(case, column)

    def rate_operations(choices) -> list:
        if rates is not None:
            y0, y1 = int(choices["adim6_y0"]), int(choices["adim6_y1"])
            return [
                Statistic(rates["frame"], rates["column"], "value", "oran0", f"{rates['label']}, {y0}",
                          where=(rates["by"], y0), decimals=1),
                Statistic(rates["frame"], rates["column"], "value", "oran1", f"{rates['label']}, {y1}",
                          where=(rates["by"], y1), decimals=1),
            ]
        r0, r1 = int(choices["adim6_oran0"]), int(choices["adim6_oran1"])
        return [Scalar("oran0", E.const(float(r0)), "Başlangıç oranı (%)", decimals=0),
                Scalar("oran1", E.const(float(r1)), "Yeni oran (%)", decimals=0)]

    def rate_words(choices) -> tuple[str, str]:
        if rates is not None:
            return str(choices["adim6_y0"]), str(choices["adim6_y1"])
        return f"%{choices['adim6_oran0']}", f"%{choices['adim6_oran1']}"

    def build(choices) -> tuple:
        d0, d1 = choices["adim6_d0"], choices["adim6_d1"]
        first, second = rate_words(choices)
        forward, back = directions(d0, d1)
        operations = [
            Statistic(case.frame, y, "mean", "x0", f"x₀: {case.name(column)} = {d0} olanların ortalaması",
                      where=(column, values[d0]), decimals=2),
            Statistic(case.frame, y, "mean", "x1", f"x₁: {case.name(column)} = {d1} olanların ortalaması",
                      where=(column, values[d1]), decimals=2),
            Scalar("yuzde_degisim", _percent_change(E.ref("x0"), E.ref("x1")), "Yüzde değişim: x₀ → x₁", decimals=2,
                   percent=True),
            Scalar("geri_donus", _percent_change(E.ref("x1"), E.ref("x0")), "Geri dönüş: x₁ → x₀", decimals=2,
                   percent=True),
            Scalar("log_farki", _log_change(E.ref("x0"), E.ref("x1")), "100·[ln(x₁) − ln(x₀)]", decimals=2),
            Scalar("log_geri", _log_change(E.ref("x1"), E.ref("x0")), "100·[ln(x₀) − ln(x₁)]", decimals=2),
            Scalar("yaklasim_farki", E.sub(E.ref("yuzde_degisim"), E.ref("log_farki")),
                   "Tam yüzde değişim − log farkı", decimals=2),
            *rate_operations(choices),
            Scalar("yuzde_puan", E.sub(E.ref("oran1"), E.ref("oran0")), f"Yüzde puan farkı: {first} → {second}",
                   decimals=1),
            Scalar("goreli_degisim", _percent_change(E.ref("oran0"), E.ref("oran1")),
                   f"Göreli yüzde değişim: {first} → {second}", decimals=2, percent=True),
            ScalarTable(
                (
                    (f"Yüzde değişim, {forward}", E.ref("yuzde_degisim")),
                    (f"Log farkı ×100, {forward}", E.ref("log_farki")),
                    (f"Yüzde değişim, {back}", E.ref("geri_donus")),
                    (f"Log farkı ×100, {back}", E.ref("log_geri")),
                    (f"Yüzde puan, {first} → {second}", E.ref("yuzde_puan")),
                    (f"Göreli yüzde değişim, {first} → {second}", E.ref("goreli_degisim")),
                ),
                "yuzdeler",
                decimals=2,
            ),
        ]
        base_text = _level_text(base)
        operations.append(Statistic(case.frame, y, "mean", "tablo_x0", f"{case.name(column)} = {base_text}: ortalama",
                                    where=(column, values[base_text]), decimals=2))
        tables = {"tam": [], "log": [], "fark": []}
        for index, target in enumerate(targets, start=1):
            text = _level_text(target)
            operations.append(Statistic(case.frame, y, "mean", f"tablo_x{index}",
                                        f"{case.name(column)} = {text}: ortalama", where=(column, values[text]),
                                        decimals=2))
            label = f"{base_text} → {text}{unit}"
            start, end = E.ref("tablo_x0"), E.ref(f"tablo_x{index}")
            tables["tam"].append((label, _percent_change(start, end)))
            tables["log"].append((label, _log_change(start, end)))
            tables["fark"].append((label, E.sub(_percent_change(start, end), _log_change(start, end))))
        operations += [
            ScalarTable(tuple(tables["tam"]), "tablo03_tam", decimals=2),
            ScalarTable(tuple(tables["log"]), "tablo03_log", decimals=2),
            ScalarTable(tuple(tables["fark"]), "tablo03_fark", decimals=2),
            JoinColumns("tablo03", (("tam", "tablo03_tam", "deger"), ("log_fark", "tablo03_log", "deger"),
                                    ("fark", "tablo03_fark", "deger")), decimals=2,
                        heading=f"{case.name(column)} değişimi",
                        title=f"Tablo 0.3'ün karşılığı: ortalama {case.name(y) if case.own else _phrase(case, y)}, "
                              "tam yüzde değişim ve log farkı"),
            Support("egri", "x1", 50, 200, "x₀ = 100 iken x₁ = 50, 51, …, 200"),
            Derive("egri", "tam", _percent_change(PERCENT_BASE, E.var("x1")), "Tam yüzde değişim"),
            Derive("egri", "log_fark", _log_change(PERCENT_BASE, E.var("x1")), "100·[ln(x₁) − ln(100)]"),
            LineChart("egri", "x1", "tam", "Yeni değer x₁ (x₀ = 100)", "Değişim (%)",
                      "Tam yüzde değişim ve log farkı: küçük değişimlerde yakın, büyük değişimlerde ayrışır",
                      markers=False, series=(("log_fark", "100·[ln(x₁) − ln(x₀)]"),), legend="Tam yüzde değişim"),
        ]
        return tuple(operations)

    def note(state, choices) -> str:
        s = state.scalars
        d0, d1 = choices["adim6_d0"], choices["adim6_d1"]
        first, second = rate_words(choices)
        name = _phrase(case, column)
        if d0 == d1:
            change = f"Başlangıç ve yeni değer aynı gruptan ({d0}): değişim yoktur; tam yüzde değişim de log farkı da sıfırdır."
        else:
            forward, back = s["yuzde_degisim"], s["geri_donus"]
            digits = 4 if sayi(abs(forward), 2) == sayi(abs(back), 2) else 2
            change = (f"{_capital(name)} = {d0} olanların ortalaması x₀ = "
                      f"{sayi(s['x0'], 2)}, {d1} olanlarınki x₁ = {sayi(s['x1'], 2)}: tam yüzde değişim "
                      f"%{sayi(forward, digits)}, geri dönüş (x₁ → x₀) ise %{sayi(back, digits)}; yüzde değişim "
                      "simetrik değildir, çünkü payda (başlangıç değeri) değişir. Log farkı simetriktir: "
                      f"{sayi(s['log_farki'], 2)} ve {sayi(s['log_geri'], 2)}. Tam değişim ile log farkı arasındaki "
                      f"fark {sayi(s['yaklasim_farki'], 2)}; değişim büyüdükçe artar (Tablo 0.3'ün karşılığı).")
        r0, r1 = s["oran0"], s["oran1"]
        if abs(r1 - r0) < 1e-9:
            rate = f" Oran değişmezse ({first} → {second}) fark 0 yüzde puandır, göreli değişim de sıfırdır."
        else:
            verb = "artar" if r1 > r0 else "düşer"
            subject = f"{rates['label']} %{sayi(r0, 1)} iken %{sayi(r1, 1)} olunca" if rates is not None else \
                f"Oran %{sayi(r0, 0)} iken %{sayi(r1, 0)} olursa"
            when = f" ({first} → {second})" if rates is not None else ""
            rate = (f" {subject}{when} oran {sayi(abs(r1 - r0), 1)} yüzde puan {verb}; göreli yüzde değişim "
                    f"%{sayi(s['goreli_degisim'], 2)}.")
        return change + rate + " Yüzde ile yüzde puan karıştırılmamalıdır (§0.6)."

    checks = [
        _scalar("x0", "Başlangıç değeri x₀", 2),
        _scalar("x1", "Yeni değer x₁", 2),
        _scalar("yuzde_degisim", "Yüzde değişim x₀ → x₁", 2),
        _scalar("geri_donus", "Yüzde değişim x₁ → x₀", 2),
        _scalar("log_farki", "Log farkı x₀ → x₁", 2),
        _scalar("log_geri", "Log farkı x₁ → x₀", 2),
        _scalar("yuzde_puan", "Yüzde puan farkı", 1),
        _scalar("goreli_degisim", "Göreli yüzde değişim", 2),
    ]
    for index, target in enumerate(targets, start=1):
        label = f"{_level_text(base)} → {_level_text(target)}"
        checks += [Check(f"Tablo 0.3'ün karşılığı {label}: {name}", TableTarget("tablo03", f"{label}{unit}", column_name),
                         0.0, 2)
                   for column_name, name in (("tam", "tam yüzde değişim"), ("log_fark", "log farkı"), ("fark", "fark"))]
    return interactive_step(
        number=6,
        title=title,
        note=note_ref,
        explanation=f"{formula} {case.extra.get('percent_text') or _own_percent_text(case)}",
        controls=_percent_controls(case),
        build=build,
        checks=tuple(checks),
        note_for=lambda state, choices: note(state, choices),
    )


def _own_percent_text(case: Case) -> str:
    return (f"Başlangıç ve yeni değer, {_phrase(case, case.roles[KOSUL])} değişkeninin iki değerindeki ortalama "
            f"{_phrase(case, case.roles[SONUC])} değeridir. Oran örneği veriden bağımsız bir sayı örneğidir: bir oran "
            "%30'dan %35'e çıkarsa artış 5 yüzde puandır; göreli yüzde değişim ise $100\\,(35 - 30)/30$.")


# --- Adım 7: beklenen değer ve koşullu ortalama -----------------------------------------------------------------

def _step7(case: Case) -> LabStep:
    title, note_ref = "Beklenen değer ve koşullu ortalama", NoteRef("0.7")
    definition = ("$\\mathbb{E}(Y)$ anakütlenin teorik ortalamasıdır; $\\mathbb{E}(Y \\mid X = x)$, $X = x$ koşulunda "
                  "$Y$'nin beklenen değeridir. Örneklemdeki karşılıkları ortalama ve koşula uyan gözlemlerin "
                  "ortalamasıdır.")
    if not case.has(KOSUL):
        return LabStep(number=7, title=title, note=note_ref,
                       explanation=f"{definition} Koşullu ortalamalar için açıklayıcı değişkenin en çok 25 farklı "
                                   "değeri olmalı ya da en çok 25 farklı değeri olan ayrı bir koşul sütunu "
                                   "seçilmelidir.")
    column, y = case.roles[KOSUL], case.roles[SONUC]
    levels = _levels(case)
    values = {_level_text(level): _level_value(case, level) for level in levels}
    counts = case.data[column].value_counts()
    default = case.extra.get("condition_default")
    if default is None:
        default = max(levels, key=lambda level: (int(counts[_level_value(case, level)]), -level))
    unit = _unit(case, column)
    choice = Choice("adim7_kosul", f"Koşul: {case.name(column)}",
                    tuple((text, f"{text}{unit}") for text in values), _level_text(default),
                    help="Koşula uyan gözlemlerin ortalaması.")
    y_name = case.name(y) if case.own else _phrase(case, y)

    def build(choices) -> tuple:
        level = choices["adim7_kosul"]
        return (
            Statistic(case.frame, y, "mean", "ortalama_sonuc", "Koşulsuz örneklem ortalaması ȳ", decimals=2),
            GroupStats(case.frame, (column,), y, ("count", "mean"), "kosullu",
                       f"{case.name(column)} değerine göre gözlem sayısı ve ortalama {y_name}", decimals=2),
            Statistic(case.frame, y, "mean", "kosullu_ortalama", f"Koşullu ortalama: {case.name(column)} = {level}",
                      where=(column, values[level]), decimals=2),
            Statistic(case.frame, y, "count", "kosul_n", f"Gözlem sayısı: {case.name(column)} = {level}",
                      where=(column, values[level]), decimals=0),
            BarChart("kosullu", "mean", _display(case, column), f"Ortalama · {_display(case, y)}",
                     f"{case.name(column)} değerine göre ortalama {y_name} (koşullu örneklem ortalamaları)",
                     decimals=0 if case.data[y].abs().mean() >= 100 else 1),
        )

    def note(state, choices) -> str:
        s = state.scalars
        level = choices["adim7_kosul"]
        count = int(s["kosul_n"])
        text = (f"Örneklemdeki {sayim(len(case.data))} {case.unit} için ortalama {y_name if not case.own else '“' + md(y_name) + '”'} "
                f"(koşulsuz örneklem ortalaması ȳ) {sayi(s['ortalama_sonuc'], 2)}; "
                f"{_phrase(case, column)} = {level} olan {sayim(count)} {case.unit} için ortalama "
                f"{sayi(s['kosullu_ortalama'], 2)}. ")
        if count < 15:
            text += (f"Bu grup küçüktür ({sayim(count)} gözlem); örneklem koşullu ortalaması az sayıda gözleme dayanır "
                     "ve örneklemden örnekleme çok değişebilir. ")
        return text + ("Bu sayılar örneklemden hesaplanır; anakütledeki koşullu beklenen değer bilinmez. Grafik, "
                       "örneklemdeki koşullu ortalamaların koşul değişkeniyle nasıl değiştiğini gösterir; regresyon, "
                       "koşullu ortalamanın açıklayıcı değişkenle nasıl değiştiğini inceler (§0.7).")

    shown = [levels[0], default, levels[-1]]
    checks = [_scalar("ortalama_sonuc", "Koşulsuz ortalama", 2)]
    checks += [Check(f"{case.name(column)} = {_level_text(level)}: ortalama",
                     TableTarget("kosullu", _level_value(case, level), "mean"), 0.0, 2)
               for level in dict.fromkeys(shown)]
    checks.append(_scalar("kosullu_ortalama", "Koşullu ortalama", 2))
    example = case.extra.get("condition_text") or (
        f"Örnek: {_phrase(case, column)} = {_level_text(default)} olan gözlemlerin ortalama {_phrase(case, y)} değeri.")
    return interactive_step(
        number=7,
        title=title,
        note=note_ref,
        explanation=f"{definition} {example}",
        controls=(choice,),
        build=build,
        checks=tuple(checks),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 8: parametre, tahmin edici ve tahmin ---------------------------------------------------------------------

def _subsample_names(case: Case) -> tuple[str, str, str, str]:
    """Adım 8'in türetilen sütunları (sıra, tek, çift, her beşinci); verideki adlarla çakışmaz."""

    taken = set(case.data.columns) | {_scaled(case), _logged(case) or ""}
    names: list[str] = []
    for base in ("sira", "tek", "cift", "besinci"):
        names.append(free_name(base, taken | set(names)))
    return names[0], names[1], names[2], names[3]


def _step8(case: Case) -> LabStep:
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    n = len(case.data)
    odd, even, fifth = (n + 1) // 2, n // 2, (n + 4) // 5
    sira, tek, cift, besinci = _subsample_names(case)
    y_name = _phrase(case, y)
    return LabStep(
        number=8,
        title="Parametre, tahmin edici ve tahmin",
        note=NoteRef("0.8", objects=("Tablo 0.4",)),
        explanation=(
            "**Parametre** anakütleye ait bilinmeyen sayıdır (ör. anakütledeki ortalama $\\mu$). **Tahmin edici** "
            "örneklemi sayıya dönüştüren kuraldır ($\\bar{X} = n^{-1}\\sum X_i$). **Tahmin** bu kuralın belirli bir "
            "örneklemde verdiği sayıdır. Aynı kural farklı örneklemlerde farklı tahminler verebilir."
        ),
        operations=(
            TakeRows("tablo04", case.frame, f"Tablo 0.4'ün örneği: Adım 1'deki beş gözlemin {case.name(x)} değerleri",
                     rows=tuple(case.extra["excerpt_rows"]), columns=(x,)),
            Statistic("tablo04", x, "mean", "tahmin_tablo", "Tahmin x̄ = Σxᵢ / n", decimals=2),
            Derive(case.frame, sira, E.seq(E.var(y)), f"Sıra numarası 1, 2, …, {sayim(n)}"),
            Derive(case.frame, tek, E.compare("eq", E.sub(E.var(sira), E.mul(2, E.floor(E.div(E.var(sira), 2)))), 1),
                   "Tek sıra numaralı gözlem (1)"),
            Derive(case.frame, cift, E.sub(1, E.var(tek)), "Çift sıra numaralı gözlem (1)"),
            Derive(case.frame, besinci, E.compare("eq", E.sub(E.var(sira), E.mul(5, E.floor(E.div(
                E.sub(E.var(sira), 1), 5)))), 1), "Her beşinci gözlem: 1., 6., 11., … (1)"),
            Statistic(case.frame, y, "mean", "tahmin_tum", f"{sayim(n)} gözlem", decimals=2),
            Statistic(case.frame, y, "mean", "tahmin_tek", f"Tek sıradakiler ({sayim(odd)})", where=(tek, 1),
                      decimals=2),
            Statistic(case.frame, y, "mean", "tahmin_cift", f"Çift sıradakiler ({sayim(even)})", where=(cift, 1),
                      decimals=2),
            Statistic(case.frame, y, "mean", "tahmin_besinci", f"Her beşinci gözlem ({sayim(fifth)})",
                      where=(besinci, 1), decimals=2),
            ScalarTable(
                (
                    (f"{sayim(n)} gözlem (bütün örneklem)", E.ref("tahmin_tum")),
                    (f"Tek sıra numaralı {sayim(odd)} gözlem", E.ref("tahmin_tek")),
                    (f"Çift sıra numaralı {sayim(even)} gözlem", E.ref("tahmin_cift")),
                    (f"Her beşinci gözlem ({sayim(fifth)} gözlem)", E.ref("tahmin_besinci")),
                ),
                "tahminler",
                decimals=2,
                heading="Örneklem",
                value=f"Ortalama {case.name(y) if case.own else _phrase(case, y)}",
            ),
        ),
        checks=(
            _scalar("tahmin_tablo", "Tablo 0.4'ün örneği: tahmin x̄", 2),
            _scalar("tahmin_tum", "Bütün örneklemde ortalama", 2),
            _scalar("tahmin_tek", "Tek sıradakilerin ortalaması", 2),
            _scalar("tahmin_cift", "Çift sıradakilerin ortalaması", 2),
            _scalar("tahmin_besinci", "Her beşinci gözlemin ortalaması", 2),
        ),
        takeaway=(
            f"Dört satırın hepsi aynı tahmin edicinin (örneklem ortalaması) sonucudur; kullanılan gözlemler değişince "
            f"{y_name} için tahmin de değişebilir. Buradaki alt örneklemler aynı veriden alınır; anakütleden her "
            "seferinde yeni bir örneklem çekmenin sonucunu Sezgi sekmesindeki Deney 1 gösterir. Anakütledeki ortalama "
            f"μ tek bir sayıdır ve bilinmez. Örneklemden örnekleme değişim istatistiksel belirsizliğin kaynağıdır; "
            "ölçülmesi (standart hata) Konu 7'nin konusudur (§0.8)."
        ),
    )


# --- Adım 9: ilk regresyon çıktısı ---------------------------------------------------------------------------

def _slope_phrase(case: Case, x: str) -> str:
    phrases = dict(case.extra.get("slope_phrases", {}))
    if x in phrases:
        return phrases[x]
    return f"“{md(case.name(x))}” değeri bir birim daha yüksek olan"


def _step9(case: Case) -> LabStep:
    y, default = case.roles[SONUC], case.roles[ACIKLAYICI]
    choice = Choice("adim9_x", f"Açıklayıcı değişken (bağımlı değişken: {case.name(y)})",
                    _numeric_options(case, _pair_options(case)), default,
                    help="Konu 0'da yalnız çıktının parçaları okunur; basit regresyon Konu 3'te kurulur.")

    def build(choices) -> tuple:
        x = choices["adim9_x"]
        frame, complete = _complete(case, case.frame, (y, x), "model_veri",
                                    f"Tam gözlemler: {case.name(y)} ve {case.name(x)}")
        main = x == default
        operations = [
            *complete,
            OLS("model", frame, y, (x,), f"Basit regresyon: {y} ~ {x}" if main else
                f"Seçtiğiniz değişkenle: {y} ~ {x}"),
            ShowModel("model", "Yazılım çıktısının temel bölümü (Kod 0.2'deki gibi)" if main else
                      "Seçtiğiniz modelin çıktısı (Kod 0.2'deki gibi)", stats=("nobs", "r2"), stars=False),
            PairStatistic(frame, y, x, "cov", "kov_yx", "Kovaryans s_xy", decimals=4),
            Statistic(frame, x, "var", "var_x", "Varyans s_x²", decimals=4),
            Scalar("egim_formul", E.div(E.ref("kov_yx"), E.ref("var_x")), "Eğim = kovaryans / varyans", decimals=4),
            PairStatistic(frame, y, x, "corr", "r_yx", "Korelasyon r", decimals=4),
            Scalar("r_kare", E.power(E.ref("r_yx"), 2), "Korelasyonun karesi r²", decimals=3),
            ModelValue("b1", "model", "coef", "Çıktıdaki eğim β̂₁", term=x),
            ModelValue("r2", "model", "r2", "Çıktıdaki R²", decimals=3),
        ]
        if not main:
            operations += [
                OLS("model_varsayilan", case.frame, y, (default,), f"Karşılaştırma için varsayılan model: {y} ~ {default}"),
                RegressionTable((("(1) Varsayılan", "model_varsayilan"), ("(2) Seçiminiz", "model")),
                                (default, x, INTERCEPT), "karsilastirma",
                                f"Varsayılan model ile seçtiğiniz model, bağımlı değişken: {case.name(y)}", stars=False),
            ]
        return tuple(operations)

    def note(state, choices) -> str:
        s = state.scalars
        x = choices["adim9_x"]
        b1 = s["b1"]
        unit = _unit(case, y).strip()
        digits = 2 if abs(b1) >= 0.005 else 4
        amount = f"{sayi(abs(b1), digits)} {unit}".strip() if unit else f"{sayi(abs(b1), digits)} birim"
        text = (f"Çıktıdaki eğim {sayi(b1, 4)}; aynı sayı kovaryansın varyansa oranıdır: {sayi(s['kov_yx'], 4)} / "
                f"{sayi(s['var_x'], 4)} = {sayi(s['egim_formul'], 4)}. R² = {sayi(s['r2'], 3)}, korelasyonun "
                f"karesiyle aynıdır ({sayi(s['r_kare'], 3)}); bu eşitlik yalnız sabit terimli basit regresyonda "
                "geçerlidir. ")
        subject = case.extra.get("plural_unit", "gözlemlerin")
        y_name = dict(case.extra.get("possessive", {})).get(y) or f"{_phrase(case, y)} değeri"
        if at_zero(b1, 4):
            text += (f"Mekanik yorum: eğim dört basamakta sıfırdır; örneklemde {_slope_phrase(case, x)} {subject} "
                     f"tahmin edilen {y_name} ortalama olarak aynıdır.")
        else:
            text += (f"Mekanik yorum: örneklemde {_slope_phrase(case, x)} {subject} tahmin edilen {y_name} ortalama "
                     f"{amount} daha {signed_difference(b1)}.")
        text += (" Bu, nedensel bir etki iddiası değildir. Standart hata (`std err`), t, p-değeri (`P>|t|`) ve %95 "
                 "güven aralığı sütunları Konu 7'de, F istatistiği (`F-statistic`) Konu 8'de açıklanır (§0.9).")
        if case.own and exact_fit(case.data, y, x):
            text += EXACT_FIT_NOTE
        if x != default:
            text += " Karşılaştırma tablosunda sütun (1) varsayılan model, sütun (2) seçtiğiniz modeldir."
        return text

    def coef(term: str, quantity: str, decimals: int, label: str) -> Check:
        return Check(label, CoefTarget("model", term, quantity), 0.0, decimals)

    return interactive_step(
        number=9,
        title="İlk Python regresyon çıktısını tanıma",
        note=NoteRef("0.9", objects=("Kod 0.1", "Kod 0.2")),
        explanation=(
            f"`{y} ~ {default}` yazımı {_phrase(case, y)} değişkenini bağımlı, {_phrase(case, default)} değişkenini "
            "açıklayıcı değişken yapar. Kodun ezberlenmesi beklenmez; çıktıda bağımlı değişken (`Dep. Variable`), "
            "gözlem sayısı (`No. Observations`), katsayılar (`coef`) ve $R^2$ (`R-squared`) okunur."
        ),
        controls=(choice,),
        build=build,
        checks=stable_checks((
            coef(INTERCEPT, "coef", 4, "Sabit terim"),
            coef(default, "coef", 4, "Eğim katsayısı"),
            coef(INTERCEPT, "se", 3, "Sabit terimin standart hatası"),
            coef(default, "se", 3, "Eğimin standart hatası"),
            coef(INTERCEPT, "t", 3, "Sabit terimin t değeri"),
            coef(default, "t", 3, "Eğimin t değeri"),
            coef(INTERCEPT, "p", 3, "Sabit terimin p-değeri"),
            coef(default, "p", 3, "Eğimin p-değeri"),
            coef(INTERCEPT, "ci_low", 3, "Sabit terim, %95 GA alt sınır"),
            coef(INTERCEPT, "ci_high", 3, "Sabit terim, %95 GA üst sınır"),
            coef(default, "ci_low", 3, "Eğim, %95 GA alt sınır"),
            coef(default, "ci_high", 3, "Eğim, %95 GA üst sınır"),
            Check("R²", ModelTarget("model", "r2"), 0.0, 3),
            Check("F istatistiği", ModelTarget("model", "f"), 0.0, 1),
            Check("Gözlem sayısı", ModelTarget("model", "nobs"), 0.0, 0),
            _scalar("var_x", "Açıklayıcının varyansı", 4),
            _scalar("kov_yx", "Kovaryans", 4),
            _scalar("egim_formul", "Eğim = kovaryans / varyans", 4),
            _scalar("r_kare", "R² = r²", 3),
        ), case.own and exact_fit(case.data, y, default)),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Uygulama -------------------------------------------------------------------------------------------------

def _labels(case: Case) -> tuple[tuple[str, str], ...]:
    labels = {column: _display(case, column) for column in case.labels}
    labels.update({
        "gozlem": "Gözlem (sıra)" if case.own else "Gözlem (WAGE2'deki sıra)",
        "sapma": "Sapma xᵢ − x̄", "sapma_kare": "Kareli sapma", "A": "Veri seti A", "B": "Veri seti B", "y": "y",
        "x": "x", "x1": "Yeni değer x₁", "tam": "Tam yüzde değişim", "log_fark": "Log farkı ×100", "fark": "Fark",
        "count": "Gözlem sayısı", "mean": "Ortalama", "deger": "Değer", INTERCEPT: "Sabit terim", "n": "Gözlem sayısı",
        "k": "Analitik değişken sayısı",
    })
    code = _code_column(case)
    if code is not None and code not in labels:
        labels[code] = f"{case.name(case.roles[GOSTERGE])} kodu (1/0)"
    labels[_scaled(case)] = str(case.extra.get("scaled_label") or f"{case.name(case.roles[SONUC])} × 100")
    logged = _logged(case)
    if logged is not None and logged not in labels:
        labels[logged] = f"ln({case.name(case.roles[SONUC])})"
    for name, text in zip(_subsample_names(case), ("Sıra numarası", "Tek sıra (1/0)", "Çift sıra (1/0)",
                                                     "Her beşinci (1/0)")):
        labels.setdefault(name, text)
    labels.update(dict(case.extra.get("labels", {})))
    return tuple(labels.items())


def build(case: Case) -> LabSpec:
    """Konu 0 uygulamasını verilen veriyle kurar; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    spec = LabSpec(
        topic_key=TOPIC,
        title=TITLE,
        note_section="0",
        steps=(_step1(case), _step2(case), _step3(case), _step4(case), _step5(case), _step6(case), _step7(case),
               _step8(case), _step9(case)),
        labels=_labels(case),
        source=case.source,
    )
    return with_app_values(spec)


# --- Alternatif örnek: WAGE2 ve OKUN ---------------------------------------------------------------------------

def alternative_case() -> Case:
    data = W.load(ALT_DATA)
    okun = (LoadWooldridge(OKUN, "OKUN veri seti (Wooldridge, 2020): ABD, 1959–2005"),
            CompleteCases("okun_tam", OKUN, ("pcrgdp", "cunem"),
                          "İşsizlik değişimi tanımlı yıllar (1960–2005; 1959'un bir önceki yılı veride yok)"))
    patterns = (
        Pattern("pozitif", "Pozitif ilişki: eğitim ve IQ puanı (WAGE2)", ALT_DATA, "educ", "IQ", "Eğitim (yıl)",
                "IQ puanı (puan)", "WAGE2: eğitim ve IQ puanı", kind="pozitif", points=len(data)),
        Pattern("negatif", "Negatif ilişki: büyüme ve işsizlik değişimi (OKUN)", "okun_tam", "pcrgdp", "cunem",
                "Reel GSYH büyümesi (%)", "İşsizlik oranındaki değişim (yüzde puan)",
                "OKUN: büyüme ve işsizlik oranındaki değişim", setup=okun, kind="negatif",
                points=int(W.load(OKUN)[["pcrgdp", "cunem"]].dropna().shape[0])),
        Pattern("zayif", "Zayıf doğrusal ilişki: iş deneyimi ve aylık kazanç (WAGE2)", ALT_DATA, "exper", "wage",
                "İş deneyimi (yıl)", "Aylık kazanç (ABD doları/ay)", "WAGE2: iş deneyimi ve aylık kazanç",
                kind="zayif", points=len(data)),
        _quadratic(),
    )
    catalog = W.DATASETS[ALT_DATA].variables
    return Case(
        source="alternatif",
        load=(LoadWooldridge(ALT_DATA, "WAGE2 veri seti (Wooldridge, 2020): 935 erkek çalışan, 1980"),),
        frame=ALT_DATA,
        data=data,
        roles={SONUC: "wage", ACIKLAYICI: "educ", GOSTERGE: "married", KOSUL: "educ"},
        labels={name: item.label for name, item in catalog.items()},
        extras=("exper", "tenure", "IQ"),
        units={name: item.unit for name, item in catalog.items()},
        unit="çalışan",
        extra={
            "excerpt_rows": ALT_ROWS,
            "excerpt_extras": ("exper",),
            "excerpt_comment": "WAGE2'den sistematik örnek: 1. gözlemden başlayarak her 187. çalışan",
            "excerpt_text": (
                "WAGE2'den beş çalışan sistematik olarak seçilir: 1. gözlemden başlayarak her 187. çalışan (1., 188., "
                "375., 562. ve 749. gözlem). Her **satır** bir gözlem (bir çalışan), her **sütun** bir değişkendir. "
                "“Gözlem” sütunu çalışanın WAGE2'deki sırasıdır; gözlemi tanımlar, analitik değişken sayılmaz. "
                "“Evli” iki kategorili bir gösterge değişkenidir: evli çalışanlar için 1, diğerleri için 0."
            ),
            "indicator": {"one": "Evli", "zero": "Bekâr", "numeric": True},
            "short_units": _ALT_UNITS,
            "patterns": patterns,
            "patterns_text": (
                "Seçenekler gerçek veri çiftleridir: WAGE2'de eğitim ve IQ puanı, OKUN'da reel GSYH büyümesi ve "
                "işsizlik oranındaki değişim, WAGE2'de iş deneyimi ve aylık kazanç. Karesel örnek kusursuz ama "
                "doğrusal olmayan bir sayı örneğidir."
            ),
            "scaled": "wage_sent",
            "scaled_label": "Aylık kazanç (sent)",
            "scaled_name": "kazanç sent",
            "scaled_comment": "Aylık kazanç, sent (1 dolar = 100 sent)",
            "scaled_sentence": "Kazanç sent olarak ölçülünce",
            "logged": "lwage",
            "step5_suffix": " (WAGE2)",
            "step5_text": (
                "WAGE2'deki 935 çalışan: aylık kazanç (dolar) ile ikinci değişken arasındaki kovaryans ve korelasyon. "
                "Kazanç sent olarak ölçülünce kovaryans 100 katına çıkar, korelasyon değişmez; logaritma gibi doğrusal "
                "olmayan bir dönüşüm ise korelasyonu genellikle değiştirir."
            ),
            "percent_defaults": (12, 16),
            "rates": {"frame": OKUN, "column": "unem", "by": "year", "label": "İşsizlik oranı",
                      "options": tuple(range(1959, 2006)), "defaults": (1979, 1982)},
            "table03": (12, (13, 16, 18)),
            "percent_text": (
                "Başlangıç ve yeni değer WAGE2'de iki eğitim düzeyinin ortalama aylık kazancıdır (varsayılan: 12 ve 16 "
                "yıl). Oran örneği OKUN'dan: ABD'de işsizlik oranı 1979'da %5,8 iken 1982'de %9,7 oldu; artış "
                "yüzde puanla ve göreli yüzde değişimle ayrı ayrı yazılır."
            ),
            "condition_default": 16.0,
            "condition_text": "Örnek: 16 yıl eğitimli çalışanların ortalama aylık kazancı.",
            "slope_phrases": _ALT_PHRASES,
            "phrases": _ALT_PHRASES_IN_TEXT,
            "pattern_phrases": {"pcrgdp": "reel GSYH büyümesi", "cunem": "işsizlik oranındaki değişim"},
            "possessive": _ALT_POSSESSIVE,
            "plural_unit": "çalışanların",
            "labels": dict(W.labels(OKUN)) | {"wage_sent": "Aylık kazanç (sent)"},
        },
    )


@cache
def alternative() -> LabSpec:
    return build(alternative_case())


STORY = (
    "Alternatif örnek WAGE2'dir: 935 erkek çalışanın 1980 verisi; ücret burada aylık kazançtır (ABD doları). Adım 4'ün "
    "negatif örneği ve Adım 6'nın oranları OKUN'dan (ABD, 1959–2005) gelir."
)


# --- Kendi verin ---------------------------------------------------------------------------------------------

def _own_extra(case: Case) -> dict:
    """Kendi verinin Konu 0 ayarları: beş gözlemlik tablo, gösterge kodlaması."""

    extra: dict = {
        "excerpt_rows": (1, 2, 3, 4, 5),
        "excerpt_comment": "Verinin ilk beş gözlemi",
        "excerpt_text": (
            "Dosyanızın ilk beş gözlemi küçük bir veri tablosu olarak alınır (Tablo 0.1 gibi). Her **satır** bir gözlem, "
            "her **sütun** bir değişkendir. “Gözlem” sütunu satırın temizlenmiş verideki sırasıdır; gözlemi tanımlar, "
            "analitik değişken sayılmaz."
        ),
    }
    if case.has(GOSTERGE):
        column = case.roles[GOSTERGE]
        one = case.levels[GOSTERGE]
        zero = next(item for item in case.orders[column] if item != one)
        extra["indicator"] = {"one": one, "zero": zero, "numeric": False}
        extra["excerpt_text"] += (f" “{md(case.name(column))}” iki kategorili bir değişkendir; gösterge değişkeni "
                                  f"{md(one)} için 1, {md(zero)} için 0 değerini alır.")
    return extra


def custom_build(case: Case) -> LabSpec:
    """Kendi veri: koşul rolü seçilmediyse açıklayıcı değişken en çok 25 farklı değer alıyorsa koşul olur."""

    roles = dict(case.roles)
    if KOSUL not in roles and case.data[roles[ACIKLAYICI]].nunique() <= MAX_LEVELS:
        roles[KOSUL] = roles[ACIKLAYICI]
    prepared = Case(source=case.source, load=case.load, frame=case.frame, data=case.data, roles=roles,
                    labels=case.labels, extras=case.extras, units=case.units, levels=case.levels, orders=case.orders,
                    unit=case.unit, extra={**dict(case.extra), **_own_extra(case)})
    return build(prepared)


def validate(case: Case) -> None:
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    if y == x:
        raise K.UploadError("Sonuç ve açıklayıcı değişken için farklı sütunlar seçin.")
    for column in (y, x, *case.extras):
        values = case.data[column].dropna()
        if values.nunique() < 2:
            raise K.UploadError(f"“{case.name(column)}” sütununda en az iki farklı değer olmalı.")
        if len(values) < 3:
            raise K.UploadError(f"“{case.name(column)}” sütununda en az üç dolu hücre olmalı.")
    if case.has(KOSUL):
        if case.roles[KOSUL] == y:
            raise K.UploadError("Koşul değişkeni sonuç değişkeninden farklı olmalı.")
        levels = case.data[case.roles[KOSUL]].nunique()
        if levels > MAX_LEVELS:
            raise K.UploadError(f"“{case.label(KOSUL)}” sütununda {levels} farklı değer var; koşul değişkeninin en çok "
                                f"{MAX_LEVELS} farklı değeri olmalı.")
        if levels < 2:
            raise K.UploadError(f"“{case.label(KOSUL)}” sütununda en az iki farklı değer olmalı.")


def sample() -> pd.DataFrame:
    """Örnek dosya: kurgusal iş arama programı verisinden çalışanlar (bu modülün verisi; öğrenci verisi değil)."""

    frame = pd.DataFrame(list(PROGRAM_ROWS), columns=list(PROGRAM_COLUMNS))
    frame = frame[frame["issiz"] == 0].reset_index(drop=True)
    return pd.DataFrame({
        "Yıllık kazanç (bin TL)": frame["kazanc"],
        "Eğitim yılı": frame["egitim"],
        "Yaş": frame["yas"],
        "Cinsiyet": np.where(frame["kadin"] == 1, "Kadın", "Erkek"),
        "Önceki yıllık kazanç (bin TL)": frame["onceki_kazanc"],
    })


ROLES = (
    Role(SONUC, "Sonuç değişkeni", "sayisal", True, (1, 2, 3, 5, 6, 7, 8, 9),
         "Açıklanan sayısal değişken (ör. kazanç, sınav puanı): ortalama, yayılım, koşullu ortalama ve regresyonun "
         "bağımlı değişkeni."),
    Role(ACIKLAYICI, "Açıklayıcı değişken", "sayisal", True, (1, 2, 3, 4, 5, 7, 9),
         "Sonucu açıklamak için kullanılan sayısal değişken (ör. eğitim yılı). En çok 25 farklı değeri varsa koşullu "
         "ortalamalar da bu değişkenle alınır."),
    Role(GOSTERGE, "İki kategorili gösterge", "kategorik", False, (1,),
         "Tam iki kategorili sütun (ör. evet/hayır, kadın/erkek, 0/1): bir kategori 1, diğeri 0 kodlanır ve ortalaması "
         "bir paydır. Her gözlemde dolu olmalı.", levels=(2, 2), pick="1 ile kodlanan kategori", suggest=True,
         complete=True),
    Role(KOSUL, "Koşul değişkeni", "sayisal", False, (6, 7),
         "Koşullu ortalamalar için en çok 25 farklı değeri olan sayısal sütun (ör. eğitim yılı, çocuk sayısı). "
         "Açıklayıcı değişken en çok 25 farklı değer alıyorsa gerekmez. Her gözlemde dolu olmalı.", complete=True),
)

CUSTOM = CustomLab(
    roles=ROLES,
    build=custom_build,
    sample=sample,
    intro=(
        "Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sonuç ve açıklayıcı değişken zorunludur ve sayısal olmalıdır; "
        "bu iki sütunda boş hücresi olan satırlar analizden çıkarılır. Gösterge, koşul ve ek sayısal değişkenler "
        "isteğe bağlıdır. Adım 1–3 verinin ilk beş gözlemiyle yapılır."
    ),
    order_roles=(GOSTERGE,),
    min_rows=5,
    extra_columns=True,
    extra_use="sayisal",
    extra_label="Ek sayısal değişkenler (isteğe bağlı, en çok 4)",
    extra_help="Adım 2, 4, 5 ve 9'daki seçeneklere eklenir. Boş hücre içerebilir; o adımlarda yalnız değeri olan "
               "gözlemler kullanılır.",
    max_extra=4,
    validate=validate,
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
