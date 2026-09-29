"""Ders notu uygulamalarının ortak "Uygulama" sekmesi.

Her uygulama ``core.labs`` altındaki tek bir tanımdan beslenir: adım metni, uygulamanın hesabı,
Python ve R kodu ve notlarla karşılaştırma aynı kaynaktan gelir. Veriler Wooldridge (2020) veri
setleridir (``wooldridge`` paketi) ya da notlardaki küçük örneklerdir.

Etkileşimli adımlarda öğrenci spesifikasyonu değiştirebilir (ör. açıklayıcı değişken). Varsayılan seçimler
notlardaki spesifikasyondur; başka seçimde hesap, grafik ve kod seçime göre yeniden üretilir, notlarla
karşılaştırma yalnız notlardaki spesifikasyonda yapılır.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st

from core import wooldridge_data as W
from core.charts import CHART_TYPES, figure_for, p_text, show_figure, tr_number
from core.codegen.base import LANGUAGE_INFO, LANGUAGES, render_script, render_step, script_filename
from core.labs import regression as RG
from core.labs.registry import get_lab
from core.labs.runner import LabRun, LabState, run_lab, run_operations, shown_frame
from core.labs.spec import (
    REPRO_DESCRIPTIONS,
    TOTAL,
    Choice,
    CoefficientTable,
    Control,
    Describe,
    GroupStats,
    LoadWooldridge,
    ModelValue,
    MultiChoice,
    NumberChoice,
    PanelSummary,
    RegressionTable,
    ShowModel,
    SortRows,
    BoxSummary,
    ClassTable,
    Count,
    CrossTab,
    FrequencyTable,
    FromCounts,
    GroupSummary,
    InlineData,
    JoinColumns,
    JointTest,
    LabSpec,
    LabStep,
    MapCodes,
    Outcomes,
    PairStatistic,
    Percentile,
    PieChart,
    Scalar,
    ScalarTable,
    Selections,
    Shape,
    ShowFrame,
    Statistic,
    StemLeaf,
    SummaryTable,
    VariableTypes,
)

CODE_LANGUAGE_KEY = "code_language"
_COLUMN_LABELS = {
    "frekans": "Frekans",
    "goreli": "Göreli frekans",
    "yuzde": "Yüzde frekans",
    "aci": "Dilim açısı (°)",
    "sayi": "Sayı",
    "alt": "Alt sınır",
    "ust": "Üst sınır",
    "orta_nokta": "Orta nokta",
    "kumulatif_frekans": "Kümülatif frekans",
    "kumulatif_goreli": "Kümülatif göreli frekans",
    "kumulatif_yuzde": "Kümülatif yüzde",
    "yapraklar": "Yapraklar",
    "yaprak_sayisi": "Yaprak sayısı",
    "nicelik": "Büyüklük",
    TOTAL: TOTAL,
}
_DESCRIBE_LABELS = {"count": "Gözlem sayısı", "mean": "Ortalama", "std": "Standart sapma", "min": "En küçük",
                   "max": "En büyük", "sum": "Toplam"}
_PANEL_LABELS = {"gozlem": "Gözlem sayısı (satır)", "birim": "Birim sayısı", "ilk_donem": "İlk dönem",
                 "son_donem": "Son dönem", "en_az_donem": "Birim başına en az dönem",
                 "en_cok_donem": "Birim başına en çok dönem"}
_COEF_LABELS = {"coef": "Katsayı", "se": "Standart hata", "t": "t", "p": "p-değeri", "ci_low": "%95 GA alt",
                "ci_high": "%95 GA üst"}
_COEF_DECIMALS = {"coef": 4, "se": 3, "t": 3, "p": 3, "ci_low": 3, "ci_high": 3}
"""Katsayı tablosunun gösterim basamağı: yazılım çıktısındaki (statsmodels) gibi."""
_MODEL_LABELS = {"nobs": "Gözlem sayısı n", "r2": "R²", "adj_r2": "Düzeltilmiş R²", "f": "F istatistiği",
                 "f_p": "F testinin p-değeri", "ssr": "Artık kareler toplamı", "df_resid": "Artık serbestlik derecesi"}
_MODEL_DECIMALS = {"nobs": 0, "r2": 3, "adj_r2": 3, "f": 2, "f_p": 3, "ssr": 2, "df_resid": 0}
_BOX_LABELS = {
    "en_kucuk": "En küçük değer",
    "q1": "Q₁ (birinci çeyrek)",
    "medyan": "Medyan",
    "q3": "Q₃ (üçüncü çeyrek)",
    "en_buyuk": "En büyük değer",
    "iqr": "IQR = Q₃ − Q₁",
    "alt_sinir": "Alt sınır Q₁ − 1,5·IQR",
    "ust_sinir": "Üst sınır Q₃ + 1,5·IQR",
    "alt_biyik": "Sol bıyık ucu",
    "ust_biyik": "Sağ bıyık ucu",
    "aykiri_sayisi": "Aykırı değer sayısı",
}


# --- Tablo biçimleme ---------------------------------------------------------------

def _count(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", ".")


def _integer(value: float) -> str:
    """Ondalıksız özet değeri: binlik ayırıcı nokta, tipografik eksi (2.000; −1.250)."""

    return _count(value).replace("-", "−")


def _blank_if_missing(formatter: Callable[[float], str]) -> Callable[[float], str]:
    """Toplam satırında toplanmayan hücreler (ör. sınıf sınırları) boş gösterilir."""

    return lambda value: "" if pd.isna(value) else formatter(value)


def _boundary(value: float) -> str:
    return tr_number(value, 0 if float(value).is_integer() else 2)


def _index_text(item) -> str:
    """Satır adı: ondalık sayılar Türkçe yazımla (1,5; −1); ondalıksız değerde ",0" yazılmaz."""

    if isinstance(item, (float, np.floating)) and np.isfinite(item):
        return f"{float(item):.10g}".replace(".", ",").replace("-", "−")
    return str(item)


def _formatted(table: pd.DataFrame, formats: dict[str, Callable[[float], str]], index_label: str,
               label: Callable[[str], str]) -> pd.DataFrame:
    """Sayıları Türkçe biçimde metne çevirir; satır adları ilk sütun olur."""

    shown = pd.DataFrame(index=table.index)
    for column in table.columns:
        formatter = formats.get(str(column))
        values = table[column]
        shown[column] = [formatter(value) for value in values] if formatter else values
    shown = shown.rename(columns=lambda name: _COLUMN_LABELS.get(str(name), label(str(name))))
    shown.index = [_index_text(item) for item in shown.index]
    return shown.rename_axis(index_label).reset_index()


def _join_cells(op: JoinColumns, table: pd.DataFrame) -> pd.DataFrame:
    """Birleşik tablonun hücre metinleri: p-değeri sütunları p-değeri biçimiyle; satır basamağı sütun basamağından
    önce gelir; ondalıksız sayılar binlik ayırıcıyla (526)."""

    columns, rows = dict(op.column_decimals), dict(op.row_decimals)

    def cell(row, column: str, value: float) -> str:
        if pd.isna(value):
            return "—"
        if column in op.p_columns:
            return p_text(value, 3)
        digits = rows.get(str(row), columns.get(column, op.decimals))
        return _integer(value) if digits == 0 else tr_number(value, digits, op.percent)

    return pd.DataFrame({column: [cell(row, column, value) for row, value in table[column].items()]
                         for column in table.columns}, index=table.index)


def display_table(op, table: pd.DataFrame, label: Callable[[str], str] = lambda name: name) -> pd.DataFrame:
    """Bir sonuç tablosunun ekranda gösterilecek biçimi."""

    if isinstance(op, FrequencyTable):
        formats = {"frekans": _count, "goreli": lambda v: tr_number(v, 3), "yuzde": lambda v: tr_number(v, 1, True)}
        return _formatted(table, formats, label(op.variable), label)
    if isinstance(op, CrossTab):
        if op.weights is not None:
            def formatter(value: float) -> str:
                return tr_number(value, op.decimals)
        elif op.percent is None:
            formatter = _count
        else:
            def formatter(value: float) -> str:
                return tr_number(value, op.decimals, True)
        formats = {str(column): formatter for column in table.columns}
        return _formatted(table, formats, f"{label(op.row)} \\ {label(op.column)}", label)
    if isinstance(op, PieChart):
        formats = {op.column: lambda v: tr_number(v, 3), "aci": lambda v: tr_number(v, 1)}
        return _formatted(table, formats, "Kategori", label)
    if isinstance(op, ClassTable):
        formats = {
            "orta_nokta": _boundary, "frekans": _count, "goreli": lambda v: tr_number(v, 3),
            "yuzde": lambda v: tr_number(v, 1, True), "kumulatif_frekans": _count,
            "kumulatif_goreli": lambda v: tr_number(v, 3), "kumulatif_yuzde": lambda v: tr_number(v, 1, True),
        }
        shown = table.drop(columns=["alt", "ust"])  # sınıf sınırları satır adında yazılı
        heading = "Sınıf" if op.row_labels == "sinif" else "Sınır"
        return _formatted(shown, {k: _blank_if_missing(f) for k, f in formats.items()}, heading, label)
    if isinstance(op, StemLeaf):
        return _formatted(table, {"yaprak_sayisi": _count}, "Gövde", label)
    if isinstance(op, ScalarTable):
        return pd.DataFrame({op.heading: table.index, op.value: [tr_number(v, op.decimals) for v in table["deger"]]})
    if isinstance(op, GroupSummary):
        formats = {name: _count if stat == "count" else (lambda v: tr_number(v, op.decimals))
                   for name, _, stat in op.columns}
        return _formatted(table, formats, op.heading or label(op.by), label)
    if isinstance(op, JoinColumns):
        index = str(table.index.name or "")
        heading = op.heading or _COLUMN_LABELS.get(index, label(index)) or "Kategori"
        if not (op.column_decimals or op.row_decimals or op.p_columns):
            formats = {name: (lambda v: tr_number(v, op.decimals, op.percent)) for name, _, _ in op.columns}
            return _formatted(table, formats, heading, label)
        return _formatted(_join_cells(op, table), {}, heading, label)
    if isinstance(op, BoxSummary):
        shown = table.rename(index=_BOX_LABELS)
        formats = {str(column): _boundary for column in shown.columns}
        return _formatted(shown, formats, "Özet", label)
    if isinstance(op, VariableTypes):
        shown = table.reset_index()
        shown.insert(0, "Değişken", [label(name) for name in shown["degisken"]])
        return shown.rename(columns={"degisken": "Koddaki adı", "saklama": "Yazılımda saklama",
                                     "tur": "İstatistiksel tür", "ayrinti": "Ayrıntı"})
    if isinstance(op, (Describe, GroupStats)):
        formats = {stat: (_count if stat == "count" else (lambda v: tr_number(v, op.decimals))) for stat in table.columns}
        shown = pd.DataFrame(index=table.index)
        for column in table.columns:
            shown[_DESCRIBE_LABELS.get(column, column)] = [formats[column](value) for value in table[column]]
        if isinstance(op, Describe):
            shown.index = [f"{label(name)} · {name}" for name in shown.index]
            return shown.rename_axis("Değişken").reset_index()
        heading = " × ".join(label(name) for name in op.by)
        shown.index = [" · ".join(_index_text(part) for part in item) if isinstance(item, tuple) else _index_text(item)
                       for item in shown.index]
        return shown.rename_axis(heading).reset_index()
    if isinstance(op, SummaryTable):
        digits = dict(op.column_decimals)
        formats = {name: _count if stat == "count" else _integer if digits.get(name, op.decimals) == 0
                   else (lambda v, d=digits.get(name, op.decimals): tr_number(v, d))
                   for name, _, stat in op.columns}
        return _formatted(table, formats, op.heading, label)
    if isinstance(op, CoefficientTable):
        level = tr_number(100 * op.level, 0 if float(round(100 * op.level, 8)).is_integer() else 1)
        return pd.DataFrame({
            "Değişken": [RG.term_label(term, label) for term in table.index],
            "Katsayı": [coefficient_number(value, op.decimals) for value in table["katsayi"]],
            "SH": [tr_number(value, op.decimals) for value in table["sh"]],
            "t": [tr_number(value, op.t_decimals) for value in table["t"]],
            "p": [p_text(value, op.p_decimals) for value in table["p"]],
            f"Yüzde {level} GA": [f"[{tr_number(low, op.decimals)}; {tr_number(high, op.decimals)}]"
                                  for low, high in zip(table["alt"], table["ust"])],
        })
    if isinstance(op, PanelSummary):
        # Dönemler (ör. yıl 1980) binlik ayırıcısız, sayımlar (4.360 satır) binlik ayırıcıyla yazılır.
        return pd.DataFrame({"Büyüklük": [_PANEL_LABELS[name] for name in table.index],
                             "Değer": [tr_number(value, 0) if name in ("ilk_donem", "son_donem") else _count(value)
                                       for name, value in table["deger"].items()]})
    raise TypeError(f"Tablo türü tanınmıyor: {type(op).__name__}")


def _model_stat_text(stat: str, value: float) -> str:
    """Model bilgisinin ekran biçimi; F testinin p-değeri p-değeri biçimiyle (çok küçükse "< 0,001")."""

    return p_text(value, _MODEL_DECIMALS[stat]) if stat == "f_p" else tr_number(value, _MODEL_DECIMALS[stat])


def coefficient_number(value: float, decimals: int) -> str:
    """Katsayının ekran biçimi: çok küçük katsayıda (ör. 0,000402) en az üç anlamlı basamak gösterilir; aksi hâlde
    ``decimals`` basamak (4 basamakta 0,0004 yazılıp bilgi kaybolmasın)."""

    if value != 0 and np.isfinite(value) and abs(value) < 10 ** (2 - decimals):
        decimals = max(decimals, 2 - int(np.floor(np.log10(abs(value)))))
    return tr_number(value, decimals)


def coefficient_display(op: ShowModel, result, label: Callable[[str], str]) -> pd.DataFrame:
    """Yazılım çıktısının katsayı tablosu, Türkçe sayılarla; satırlar terimler (sabit terim ilk)."""

    table = RG.coefficient_table(result)
    shown = pd.DataFrame({"Terim": [f"{RG.term_label(term, label)} · {term}" for term in table.index]})
    digits = {**_COEF_DECIMALS, **dict(op.decimals)}
    for column in op.columns:
        if column == "coef" and not op.exact:
            shown[_COEF_LABELS[column]] = [coefficient_number(value, digits[column]) for value in table[column]]
        else:
            shown[_COEF_LABELS[column]] = [tr_number(value, digits[column]) for value in table[column]]
    return shown


def regression_display(op: RegressionTable, state: LabState, label: Callable[[str], str]) -> pd.DataFrame:
    """Makale tipi tablo: katsayı (yıldızla) ve altında parantez içinde standart hata (``standard_errors`` ise)."""

    table = state.tables[op.result]
    marks = RG.table_stars(op, state.models) if op.stars else {}
    digits = dict(op.term_decimals)
    rows = []
    for term in op.terms:
        coefficients, errors = {"": RG.term_label(term, label)}, {"": ""}
        for heading, _ in op.models:
            value = table.loc[term, heading]
            number = tr_number if op.exact else coefficient_number
            coefficients[heading] = ("" if pd.isna(value) else number(value, digits.get(term, op.decimals))
                                     + marks.get((term, heading), ""))
            if op.standard_errors:  # standart hata katsayıyla aynı basamakta (ör. karesel terimde 6)
                error = table.loc[f"{term}_sh", heading]
                errors[heading] = ("" if pd.isna(error)
                                   else f"({tr_number(error, digits.get(term, op.decimals))})")
        rows += [coefficients, errors] if op.standard_errors else [coefficients]
    for key, row_label, _ in op.extra:  # ör. ortak F ve p-değeri (Konu 8); boş hücre "—"
        formatter = p_text if key.endswith("_p") else (lambda value: tr_number(value, op.extra_decimals))
        rows.append({"": row_label, **{heading: "—" if pd.isna(table.loc[key, heading])
                                       else formatter(table.loc[key, heading]) for heading, _ in op.models}})
    rows.append({"": "Gözlem sayısı", **{heading: _count(table.loc["n", heading]) for heading, _ in op.models}})
    fit = op.decimals if op.r2_decimals is None else op.r2_decimals
    if op.r2:
        rows.append({"": "R²", **{heading: tr_number(table.loc["r2", heading], fit) for heading, _ in op.models}})
    if op.adj_r2:
        rows.append({"": "Düzeltilmiş R²", **{heading: tr_number(table.loc["adj_r2", heading], fit)
                                              for heading, _ in op.models}})
    return pd.DataFrame(rows).rename(columns={"": "Değişken"})


def crosstab_caption(op: CrossTab, label: Callable[[str], str] = lambda name: name) -> str:
    """Çapraz tablonun ne gösterdiği: sayılar mı, hangi paydayla yüzdeler mi, hangi alt grupta mı."""

    if op.weights is not None:
        kind = (f"**Çapraz tablo — toplanan sütun: {label(op.weights)}** (her hücre, o hücreye düşen satırlardaki "
                "değerlerin toplamıdır)")
    else:
        kind = {
            None: "**Çapraz tablo: sayılar**",
            "satir": "**Satır yüzdeleri** (payda: satır toplamı)",
            "sutun": "**Sütun yüzdeleri** (payda: sütun toplamı)",
        }[op.percent]
    if op.where is not None:
        column, value = op.where
        kind += f" · yalnız {label(column).lower()}: {value}"
    return kind


def show_table(shown) -> None:
    """Tabloyu gösterir; ``shown`` bir DataFrame ya da Türkçe sayı biçimli Styler'dır (bkz. ``_frame``)."""

    rows = len(shown) if isinstance(shown, pd.DataFrame) else len(shown.data)
    # 16 satıra kadar (ör. 5 dakikalık 14 sınıf ve Toplam) tablo kaydırmadan görünür; daha uzun ham veri kayar.
    height = min(35 * (rows + 1) + 3, 598)
    st.dataframe(shown, hide_index=True, width="stretch", height=height)


def _decimals(values: pd.Series) -> int:
    """Kesirli bir sütunun gösterim basamağı: en az 2 (notlardaki 0,25 ve 17,50 gibi), en çok 4."""

    for decimals in (2, 3):
        if np.allclose(values, np.round(values, decimals), rtol=0, atol=1e-9):
            return decimals
    return 4


def frame_display(frame: pd.DataFrame, label: Callable[[str], str], decimals: int | None = None):
    """Veri çerçevesinin ekran biçimi: tam sayı değerli sütunlar tam sayı, kesirli sütunlar ondalık virgülle.

    Kesirli sütunlar Styler ile biçimlenir; sütun sayısal kaldığı için sağa hizalı görünür. ``decimals`` verilirse
    bütün kesirli sütunlar o basamakla yazılır.
    """

    shown = frame.copy()
    formats: dict[str, Callable[[float], str]] = {}
    for column in shown.columns:
        if shown[column].dtype.kind != "f":
            continue
        if np.allclose(shown[column], np.round(shown[column]), rtol=0, atol=1e-9):
            shown[column] = shown[column].round().astype(int)
            if (shown[column] < 0).any():  # tipografik eksi: −10
                formats[column] = lambda value: tr_number(value, 0)
        else:
            digits = decimals if decimals is not None else _decimals(shown[column])
            formats[column] = lambda value, digits=digits: tr_number(value, digits)
    rename = {column: _COLUMN_LABELS.get(str(column), label(str(column))) for column in shown.columns}
    shown = shown.rename(columns=rename)
    if not formats:
        return shown
    return shown.style.format({rename[column]: formatter for column, formatter in formats.items()})


# --- Adım gezinimi -----------------------------------------------------------------

def _step_key(spec: LabSpec) -> str:
    return f"{spec.topic_key}_lab_step"


def _shift(spec: LabSpec, delta: int) -> None:
    key = _step_key(spec)
    numbers = [step.number for step in spec.steps]
    current = st.session_state.get(key) or numbers[0]
    index = min(max(numbers.index(current) + delta, 0), len(numbers) - 1)
    st.session_state[key] = numbers[index]


def _render_navigation(spec: LabSpec) -> LabStep:
    key = _step_key(spec)
    numbers = [step.number for step in spec.steps]
    if st.session_state.get(key) not in numbers:
        st.session_state[key] = numbers[0]
    # Adım düğmeleri satırın tamamını kullanır; on adımlı bir konu da geniş ekranda tek satıra sığar.
    st.segmented_control(
        "Adım", options=numbers, format_func=lambda number: f"Adım {number}", key=key,
        label_visibility="collapsed", width="stretch",
    )
    current = st.session_state.get(key) or numbers[0]
    left, _, right = st.columns([1, 4, 1])
    left.button("‹ Önceki", key=f"{spec.topic_key}_lab_prev", on_click=_shift, args=(spec, -1), width="stretch",
                disabled=current == numbers[0])
    right.button("Sonraki ›", key=f"{spec.topic_key}_lab_next", on_click=_shift, args=(spec, 1), width="stretch",
                 disabled=current == numbers[-1])
    return spec.step(current)


# --- Sonuçlar ----------------------------------------------------------------------

_METRICS = (Shape, Count, Statistic, PairStatistic, Scalar, Percentile, ModelValue, JointTest)
_SUBSCRIPTS = str.maketrans("0123456789,", "₀₁₂₃₄₅₆₇₈₉,")


def _metrics(op, state: LabState) -> list[tuple[str, str]]:
    if isinstance(op, Percentile):
        items = []
        if op.location is not None:
            index = tr_number(op.p, 0 if float(op.p).is_integer() else 1).translate(_SUBSCRIPTS)
            location = tr_number(state.scalars[op.location], 2).rstrip("0").rstrip(",")  # 7,80 → 7,8
            items.append((f"Konum L{index}", location))
        return items + [(op.comment, tr_number(state.scalars[op.name], op.decimals))]
    if isinstance(op, Shape):
        return [("Gözlem sayısı n", _count(state.scalars[op.observations])),
                ("Değişken sayısı", _count(state.scalars[op.variables]))]
    if isinstance(op, Count):
        return [(op.comment, _count(state.scalars[op.name]))]
    if isinstance(op, Scalar) and not op.shown:
        return []
    if isinstance(op, JointTest):
        return [(op.comment, tr_number(state.scalars[op.name], op.decimals)),
                ("Ortak p-değeri", p_text(state.scalars[op.p_value]))]
    if isinstance(op, ModelValue) and op.quantity in ("p", "f_p"):
        return [(op.comment, p_text(state.scalars[op.name], op.decimals))]
    if isinstance(op, (Statistic, PairStatistic, ModelValue)):
        return [(op.comment, tr_number(state.scalars[op.name], op.decimals))]
    if op.p_value:
        return [(op.comment, p_text(state.scalars[op.name], op.decimals))]
    return [(op.comment, tr_number(state.scalars[op.name], op.decimals, op.percent))]


def _show_metrics(items: list[tuple[str, str]]) -> None:
    for start in range(0, len(items), 4):
        chunk = items[start:start + 4]
        for column, (title, value) in zip(st.columns(len(chunk)), chunk):
            column.metric(title, value)


def _input_only(operations, index: int, state: LabState) -> bool:
    """Satır içi veri, aynı adımda aynı çerçevenin bütün sütunlarını gösteren bir ``ShowFrame`` ile sonuçlanıyorsa
    yalnız girdi sütunlarıyla gösterilir: aynı tablo iki kez görünmez."""

    op = operations[index]
    columns = set(state.frames[op.frame].columns)
    return any(isinstance(later, ShowFrame) and later.frame == op.frame and columns <= set(later.columns)
               for later in operations[index + 1:])


def coefficient_caption(topic_key: str) -> str:
    """Standart hatasız regresyon tablosunun altyazısı; standart hatalar Konu 7'de işlenir."""

    later = topic_key.startswith("konu") and int(topic_key[-2:]) < 7
    return "Yalnız katsayılar; standart hatalar tabloya Konu 7'de eklenir." if later else "Yalnız katsayılar."


def render_operations(operations, state: LabState, label: Callable[[str], str], key_prefix: str,
                      topic_key: str = "") -> None:
    """İşlemlerin sonuçlarını sırayla gösterir; art arda gelen tek sayılar tek satırda toplanır."""

    pending: list[tuple[str, str]] = []
    # Yan yana birleştirilen (JoinColumns) ara skaler tabloları ayrıca gösterilmez; birleşik tablo gösterilir.
    joined = {table for op in operations if isinstance(op, JoinColumns) for _, table, _ in op.columns}
    for index, op in enumerate(operations):
        if isinstance(op, ScalarTable) and op.result in joined:
            continue
        if isinstance(op, _METRICS):
            pending.extend(_metrics(op, state))
            continue
        if pending:
            _show_metrics(pending)
            pending = []
        if isinstance(op, LoadWooldridge):
            info = W.DATASETS[op.dataset]
            frame = state.frames[op.frame]
            # Yüklenen sütunlar: adımın sonraki işlemlerinde türetilen sütunlar (ör. wage = exp(lwage)) sayılmaz.
            loaded = len(op.columns) if op.columns else W.load(op.dataset).shape[1]
            st.caption(
                f"**{op.dataset.upper()}** · {info.title} · {_count(len(frame))} gözlem, {loaded} değişken · "
                f"gözlem birimi: {info.unit} · {info.structure}, {info.generation} veri · kaynak: Wooldridge (2020), "
                "`wooldridge` paketi"
            )
            continue
        if isinstance(op, SortRows):
            st.caption(f"{op.comment} (sıralama değişkeni: {label(op.by)}).")
            continue
        if isinstance(op, ShowModel):
            result = state.models[op.model]
            st.markdown(f"**{op.comment}** · bağımlı değişken: {label(result.model.endog_names)}")
            show_table(coefficient_display(op, result, label))
            _show_metrics([(_MODEL_LABELS[stat], _model_stat_text(stat, RG.model_quantity(result, stat)))
                           for stat in op.stats])
            continue
        if isinstance(op, RegressionTable):
            st.markdown(f"**{op.comment}**")
            show_table(regression_display(op, state, label))
            if op.stars:
                st.caption("Parantez içinde klasik (geleneksel) standart hatalar. *** p < 0,01; ** p < 0,05; "
                           "* p < 0,10.")
            elif op.standard_errors:
                st.caption("Parantez içinde standart hatalar.")
            else:
                st.caption(coefficient_caption(topic_key))
            continue
        if isinstance(op, InlineData):
            frame = state.frames[op.frame]
            if _input_only(operations, index, state):
                frame = frame[list(op.columns)]
            st.markdown(f"**{op.comment}**")
            if op.layout and len(op.columns) == 1 and len(frame) % op.layout == 0:
                # Notlardaki gibi satır başına ``layout`` değer; sütun başlıkları satır içindeki sıradır.
                values = frame[op.columns[0]].to_numpy()
                grid = pd.DataFrame(values.reshape(-1, op.layout), columns=[str(i) for i in range(1, op.layout + 1)])
                show_table(frame_display(grid, str))
            else:
                show_table(frame_display(frame, label))
        elif isinstance(op, FromCounts):
            counts = pd.DataFrame([tuple(row) for row in op.rows], columns=[*op.columns, "sayi"])
            st.markdown(f"**{op.comment}**")
            show_table(frame_display(counts, label))
            st.caption(f"Her satır sayısı kadar tekrarlanır: toplam {_count(len(state.frames[op.frame]))} gözlem.")
        elif isinstance(op, (Outcomes, Selections)):
            frame = state.frames[op.frame]
            st.markdown(f"**{op.comment}**")
            show_table(frame_display(frame, label))
            noun = "Sonuç" if isinstance(op, Outcomes) else "Seçim"
            st.caption(f"{noun} sayısı: {_count(len(frame))}.")
        elif isinstance(op, ShowFrame):
            st.markdown(f"**{op.comment}**")
            shown = shown_frame(op, state)
            if op.rows or op.head:  # notlardaki gibi gözlem numarası 1'den başlar
                shown = shown.copy()
                shown.insert(0, "Gözlem", [position + 1 for position in
                                           (range(op.head) if op.head else [row - 1 for row in op.rows])][:len(shown)])
            show_table(frame_display(shown, label, op.decimals))
        elif isinstance(op, MapCodes):
            frame = state.frames[op.frame][[op.source, op.name]].head(8)
            st.markdown(f"**{op.comment}**")
            show_table(frame_display(frame, label))
        elif isinstance(op, CrossTab):
            st.markdown(crosstab_caption(op, label))
            show_table(display_table(op, state.tables[op.result], label))
        elif isinstance(op, (VariableTypes, FrequencyTable, ScalarTable, GroupSummary, ClassTable, StemLeaf,
                             JoinColumns, BoxSummary, Describe, GroupStats, PanelSummary, SummaryTable,
                             CoefficientTable)):
            if isinstance(op, (Describe, GroupStats, PanelSummary, SummaryTable, CoefficientTable)):
                st.markdown(f"**{op.comment}**")
            elif getattr(op, "title", ""):  # ScalarTable, GroupSummary, JoinColumns: notlardaki tablo başlığı
                st.markdown(f"**{op.title}**")
            show_table(display_table(op, state.tables[op.result], label))
        if isinstance(op, PieChart):
            show_figure(figure_for(op, state, label), key=f"{key_prefix}_grafik_{index}")
            show_table(display_table(op, state.tables[op.result], label))
        elif isinstance(op, CHART_TYPES):
            show_figure(figure_for(op, state, label), key=f"{key_prefix}_grafik_{index}")
    if pending:
        _show_metrics(pending)


def _render_checks(step: LabStep, run: LabRun, variant: bool) -> None:
    if variant:
        st.caption(":material/info: Notlarla karşılaştırma yalnız notlardaki spesifikasyonda yapılır; bu adımda "
                   "seçtiğiniz spesifikasyonun sonuçları gösteriliyor.")
        return
    results = run.step_checks(step.number)
    if not results:
        return
    passed = sum(item.passed for item in results)
    title = f"Notlarla karşılaştırma: {passed}/{len(results)} değer aynı"
    with st.expander(title, icon=":material/fact_check:" if passed == len(results) else ":material/error:"):
        rows = [
            {
                "Değer": item.check.label,
                "Uygulama": tr_number(item.value, item.check.decimals),
                "Notlar": tr_number(item.check.expected, item.check.decimals),
                "Durum": "✓" if item.passed else "✗",
            }
            for item in results
        ]
        show_table(pd.DataFrame(rows))


def _render_code(spec: LabSpec, step: LabStep, variant: bool) -> None:
    if not step.operations:
        return
    language = st.session_state.get(CODE_LANGUAGE_KEY) or LANGUAGES[0]
    info = LANGUAGE_INFO[language]
    title, description = REPRO_DESCRIPTIONS[step.reproducibility]
    st.markdown(f"**{language} kodu**" + (" (seçtiğiniz spesifikasyonla)" if variant else ""))
    st.caption(f"İki dilde sonuç: **{title}**. {description}")
    st.code(render_step(spec, step.number, language), language=info.highlight, line_numbers=True)
    if step.code_note:
        st.caption(step.code_note)


def _download_row(spec: LabSpec, suffix: str) -> None:
    for column, language in zip(st.columns(len(LANGUAGES)), LANGUAGES):
        info = LANGUAGE_INFO[language]
        column.download_button(
            f"{language} (.{info.extension})",
            data=render_script(spec, language),
            file_name=script_filename(spec, language),
            mime=info.mime,
            key=f"{spec.topic_key}_lab_download{suffix}_{language}",
            icon=":material/download:",
            width="stretch",
        )


def _render_downloads(notes: LabSpec, chosen: LabSpec) -> None:
    st.markdown("**Bütün uygulamayı indirin**")
    st.caption(
        "Her dosya bütün adımları çalıştırır ve sonunda sonuçları ders notlarındaki sayılarla karşılaştırır. "
        "Bir sayı tutmazsa hangisinin tutmadığını söyleyerek durur."
    )
    _download_row(notes, "")
    if chosen.variant:
        steps = ", ".join(str(number) for number in chosen.variant)
        which = "bu adımlar" if len(chosen.variant) > 1 else "bu adım"
        st.caption(f"Seçtiğiniz spesifikasyonla (Adım {steps} notlardan farklı; {which} notlarla karşılaştırılmaz):")
        _download_row(chosen, "_secim")


# --- Etkileşimli spesifikasyon -----------------------------------------------------

def _choice_key(spec: LabSpec, control: Control) -> str:
    return f"{spec.topic_key}_secim_{control.key}"


def _stored(control: Control, value):
    """Denetimin oturumdaki değeri: çoklu seçim liste, sayı sayı, tek seçim dizge."""

    return list(value) if isinstance(control, MultiChoice) else value


def _current_choices(spec: LabSpec) -> tuple[dict[str, object], list[str]]:
    """Oturumdaki seçimler; geçersiz seçim (ör. boş değişken listesi) varsayılanla değiştirilir ve bildirilir."""

    chosen: dict[str, object] = {}
    problems: list[str] = []
    for control in spec.controls:
        key = _choice_key(spec, control)
        if key not in st.session_state:
            st.session_state[key] = _stored(control, control.default)
        try:
            chosen[control.key] = control.normalize(st.session_state[key])
        except ValueError as error:
            problems.append(f"{error} Notlardaki seçim kullanılıyor.")
            chosen[control.key] = control.normalize(control.default)
    return chosen, problems


def _token(choices: dict[str, object]) -> tuple:
    return tuple(sorted(choices.items()))


def _reset(spec: LabSpec, controls: tuple[Control, ...]) -> None:
    for control in controls:
        st.session_state[_choice_key(spec, control)] = _stored(control, control.default)


def _render_controls(spec: LabSpec, step: LabStep, problems: list[str]) -> None:
    with st.container(border=True):
        st.markdown("**Spesifikasyon**")
        st.caption("Varsayılan seçimler notlardaki spesifikasyondur. Seçimi değiştirdiğinizde sonuçlar, grafikler ve "
                   "kod seçiminize göre yeniden üretilir.")
        for control in step.controls:
            key = _choice_key(spec, control)
            if isinstance(control, Choice):
                if len(control.options) <= 3:
                    st.segmented_control(control.label, options=[value for value, _ in control.options],
                                         format_func=control.option_label, key=key, help=control.help or None,
                                         required=True)
                else:
                    st.selectbox(control.label, options=[value for value, _ in control.options],
                                 format_func=control.option_label, key=key, help=control.help or None)
            elif isinstance(control, MultiChoice):
                st.multiselect(control.label, options=[value for value, _ in control.options],
                               format_func=control.option_label, key=key, help=control.help or None,
                               max_selections=control.maximum)
            elif isinstance(control, NumberChoice):
                if control.integer:
                    st.slider(control.label, min_value=int(control.minimum), max_value=int(control.maximum),
                              step=int(control.step), key=key, help=control.help or None)
                else:
                    decimal_slider(st, control.label, minimum=float(control.minimum), maximum=float(control.maximum),
                                   step=float(control.step), decimals=control.decimals, key=key,
                                   help=control.help or None)
        for problem in problems:
            st.warning(problem, icon=":material/warning:")


def decimal_slider(container, label: str, *, minimum: float, maximum: float, step: float, decimals: int, key: str,
                   help: str | None = None) -> None:
    """Kesirli kaydırıcı; değerler ondalık virgülle yazılır (``st.slider`` biçimi yalnız ondalık noktayı bilir)."""

    options = [round(minimum + index * step, decimals) for index in range(int(round((maximum - minimum) / step)) + 1)]
    current = st.session_state.get(key)
    if current is not None and current not in options:
        st.session_state[key] = min(options, key=lambda option: abs(option - float(current)))
    container.select_slider(label, options=options, key=key, help=help,
                            format_func=lambda value: tr_number(value, decimals))


def upstream_steps(spec: LabSpec, step: LabStep, choices: dict[str, object]) -> list[LabStep]:
    """Seçimi notlardan farklı olan ve bu adımın sonucunu gerçekten değiştiren önceki adımlar.

    Her önceki adımın seçimi tek başına uygulanır; ``LabSpec.resolve`` bu adımı notlardan farklı sayıyorsa o adım
    listelenir (ör. Adım 3 yalnız Adım 2'nin seçimine bağlıysa Adım 1'deki seçim sayılmaz).
    """

    found = []
    for item in spec.steps:
        if item.number >= step.number or not item.controls:
            continue
        own = {control.key: choices[control.key] for control in item.controls
               if choices[control.key] != control.normalize(control.default)}
        if own and step.number in spec.resolve(own).variant:
            found.append(item)
    if found:
        return found
    # Adım yalnız seçimlerin birlikte etkisiyle farklıysa seçimi değişen bütün önceki adımlar listelenir.
    return [item for item in spec.steps if item.number < step.number and item.controls and any(
        choices[control.key] != control.normalize(control.default) for control in item.controls)]


def _render_variant_notice(spec: LabSpec, step: LabStep, choices: dict[str, object]) -> None:
    own_changed = any(choices[control.key] != control.normalize(control.default) for control in step.controls)
    if own_changed:
        left, right = st.columns([4, 1], vertical_alignment="center")
        left.info("Bu adımda notlardan farklı bir spesifikasyon seçili.", icon=":material/tune:")
        right.button("Notlara dön", key=f"{spec.topic_key}_notlara_don_{step.number}", on_click=_reset,
                     args=(spec, step.controls), width="stretch")
        return
    upstream = upstream_steps(spec, step, choices)
    numbers = ", ".join(str(item.number) for item in upstream)
    left, right = st.columns([4, 1], vertical_alignment="center")
    left.info(f"Bu adım, önceki bir adımdaki seçiminize göre hesaplandı (Adım {numbers}); notlardaki sayılardan "
              "farklı olabilir.", icon=":material/tune:")
    right.button("Notlara dön", key=f"{spec.topic_key}_notlara_don_{step.number}", on_click=_reset,
                 args=(spec, tuple(control for item in upstream for control in item.controls)), width="stretch")


# --- Hesap önbelleği -----------------------------------------------------------------

@st.cache_resource(show_spinner=False, max_entries=64)
def _resolved(topic_key: str, token: tuple) -> LabSpec:
    spec = get_lab(topic_key)
    return spec.resolve(dict(token)) if spec.controls else spec


@st.cache_resource(show_spinner=False, max_entries=64)
def _run(topic_key: str, token: tuple) -> LabRun:
    return run_lab(_resolved(topic_key, token))


@st.cache_resource(show_spinner=False, max_entries=256)
def _state_through(topic_key: str, token: tuple, number: int) -> LabState:
    """Adımın sonundaki durum: sonraki adımların eklediği sütunlar bu adımda görünmez."""

    return run_operations(_resolved(topic_key, token).operations_through(number))


def widget_keys(spec: LabSpec) -> set[str]:
    """Adım seçimi ve spesifikasyon denetimlerinin oturum anahtarları (``topics.shared.keep_widget_state``)."""

    return {_step_key(spec), *(_choice_key(spec, control) for control in spec.controls)}


def render_lab(spec: LabSpec) -> None:
    st.markdown(
        f"Bu sekme ders notlarındaki çözümlü örnekleri (**{spec.title}**) adım adım yeniden üretir. "
        "Tablolar notlardaki sayıların aynısını verir; kod dilini kenar çubuğundan seçin."
    )
    choices, problems = _current_choices(spec)
    token = _token(choices)
    step = _render_navigation(spec)
    st.subheader(f"Adım {step.number}: {step.title}")
    st.caption(step.note.label())
    st.markdown(step.explanation)
    try:
        chosen = _resolved(spec.topic_key, token)
        run = _run(spec.topic_key, token)
    except ValueError as error:  # ör. tam doğrusal bağlantı: katsayılar tek biçimde tahmin edilemez
        if step.controls:
            _render_controls(spec, step, problems)
        st.error(f"Bu spesifikasyon tahmin edilemiyor: {error}", icon=":material/error:")
        return
    variant = step.number in chosen.variant
    current = chosen.step(step.number)
    if step.controls:
        _render_controls(spec, step, problems)
    if variant:
        _render_variant_notice(spec, step, choices)
    state = None
    if current.operations:
        state = _state_through(spec.topic_key, token, step.number)
        render_operations(current.operations, state, spec.label, f"{spec.topic_key}_adim{step.number}",
                          spec.topic_key)
        _render_checks(current, run, variant)
    note = step.note_for(state, choices) if (step.note_for is not None and state is not None) else step.takeaway
    if note:
        st.info(note, icon=":material/lightbulb:")
    _render_code(chosen, current, variant)
    if step.number == spec.steps[-1].number:
        _render_downloads(spec, chosen)
