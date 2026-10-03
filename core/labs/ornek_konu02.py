"""Konu 2 genel uygulaması: ekonomik veri yapıları ve rastgele atamalı deney.

Notlardaki altı adım (§2.2, §2.3, §2.4, §2.5, §2.11, §2.12) aynı numaralarla yazılır. Alternatif örnekte her veri
yapısı başka bir Wooldridge (2020) veri setiyle gösterilir: WAGE2 (yatay kesit, 935 erkek çalışan, 1980), OKUN
(zaman serisi, ABD 1959–2005), KIELMC (havuzlanmış yatay kesit: 1978 ve 1981'de satılan konutlar), CRIME4 (panel:
Kuzey Karolina'da 90 ilçe, 1981–1987). Rastgele atamalı deney adımı kurgusal bir iş arama programıdır
(``core.labs.kurgusal_veri``; 200 kişi, kurayla 100 program ve 100 kontrol).

"Kendi verini yükle" seçeneğinde adımlar dosyanın yapısına göre kurulur: sayısal sütunlar zorunludur; dönem, birim
kimliği ve deney (0/1 atama ve sonuç) sütunları isteğe bağlıdır. Rolü seçilmeyen adım neye ihtiyacı olduğunu yazar; bir
panel dosyası (birim ve dönem) yüklenirse 1–4. adımların hepsi aynı dosyayla çalışır. Notlardaki uygulama
(``core.labs.konu02``) değişmez.

Etkileşim notlardaki gibidir: satır sırası (Adım 1–2), grafikteki seriler (Adım 2), özetlenen değişken ve gruplama
(Adım 3), gösterilen birim (Adım 4) ve gruplar arasında karşılaştırılan değişken (Adım 5).
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd

from core import wooldridge_data as W
from core.labs import expr as E
from core.labs import kendi_veri as K
from core.labs.kurgusal_veri import (
    PROGRAM_COLUMNS,
    PROGRAM_EFFECT,
    PROGRAM_LABELS,
    PROGRAM_OUTCOMES,
    PROGRAM_PRE,
    PROGRAM_ROWS,
    PROGRAM_UNEMPLOYMENT,
    program_frame,
)
from core.labs.ornek import Case, CustomLab, Role, TopicVariants, free_name, kisa, liste, md, sayi, sayim, \
    with_app_values
from core.labs.spec import (
    BarChart,
    CellTarget,
    Check,
    Choice,
    CompleteCases,
    CopyFrame,
    Derive,
    GroupStats,
    InlineData,
    LabSpec,
    LabStep,
    LineChart,
    LoadWooldridge,
    MapCodes,
    MultiChoice,
    NoteRef,
    PanelSummary,
    Scalar,
    ScalarTarget,
    Shape,
    ShowFrame,
    SortRows,
    Statistic,
    TableTarget,
    TakeRows,
    interactive_step,
)
from core.labs.wording import tr_lower

TOPIC = "konu02"
TITLE = "Uygulama: Veri Yapıları ve Rastgele Atamalı Deney"
WAGE2, OKUN, KIELMC, CRIME4, PROGRAM = "wage2", "okun", "kielmc", "crime4", "program"
INDICATOR = "0/1 gösterge"
SAYISAL, DONEM, BIRIM, ATAMA, DENEY_SONUC = "sayisal", "donem", "birim", "atama", "deney_sonuc"
SHOWN_UNITS = 8
"""Panel ve zaman serisi adımlarında seçilebilen birim sayısı (dosyadaki ilk birimler)."""

_NOTES = {
    1: NoteRef("2.2", 0, ("Tablo 2.1",)),
    2: NoteRef("2.3", 0, ("Kod 2.1", "Kod 2.2", "Şekil 2.2")),
    3: NoteRef("2.4", 0, ("Kod 2.3", "Kod 2.4")),
    4: NoteRef("2.5", 0, ("Kod 2.5",)),
    5: NoteRef("2.11", 0, ("Kod 2.6", "Kod 2.7", "Şekil 2.4")),
    6: NoteRef("2.12", 0, ("Tablo 2.4",)),
}
_TITLES = {
    1: "Yatay kesit verisi",
    2: "Zaman serisi verisi",
    3: "Havuzlanmış yatay kesit verisi",
    4: "Panel verisi",
    5: "Rastgele atamalı deney",
    6: "Gözlemsel ve deneysel bulguları aynı dille yazmamak",
}


def _check(label: str, target, decimals: int) -> Check:
    """Kontrol: beklenen değer uygulamanın hesabıyla doldurulur (``with_app_values``)."""

    return Check(label, target, 0.0, decimals)


def _key(value) -> str:
    """Seçenek değeri (denetim anahtarı): tam sayı "12", kesirli sayı "2.5", metin olduğu gibi."""

    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        number = float(value)
        return str(int(number)) if number.is_integer() else repr(number)
    return str(value)


def _shown(value) -> str:
    """Metindeki yazım: sayılar Türkçe (2,5), metin olduğu gibi."""

    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return kisa(float(value), 6)
    return md(value)


# --- Ortak adım kalıpları ---------------------------------------------------------------------------------------

def _cross_section_note(order: str, sort_words: dict[str, str], last: int) -> str:
    if order == "dosya":
        return ("Her satır bir çalışandır ve her çalışan veri setinde bir kez görünür: veri yatay kesittir, gözlem birimi "
                f"çalışandır. Tabloda yalnız 1, 2, 3 ve {sayim(last)} numaralı satırlar gösterilir; aradaki satırlar aynı "
                "yapıdadır. Eğitim yılı yüksek çalışanların kazancı da yüksekse bu farkın tamamı hemen eğitime "
                "bağlanamaz: çalışanlar deneyim, yetenek ve bölge bakımından da farklıdır.")
    return (f"Satırlar {sort_words[order]} göre sıralandı: 1, 2, 3 ve {sayim(last)} numaralı satırlarda artık başka "
            "çalışanlar görünür. Her satır yine bir çalışandır ve aynı çalışanlar zaman içinde tekrar gözlenmez; "
            "satırların yeri değişti, veri yapısı değişmedi. Zaman serisinde satır sırası ekonomik bilgi taşır (Adım 2).")


# --- Alternatif örnek ---------------------------------------------------------------------------------------------

WAGE2_COLUMNS = ("wage", "educ", "exper", "tenure")
WAGE2_ORDERS = {"wage": "aylık kazanca", "educ": "eğitime"}


def _alt_step1() -> LabStep:
    rows = (1, 2, 3, 935)

    def build(choices) -> tuple:
        order = choices["adim1_sira"]
        operations: list = [LoadWooldridge(WAGE2, "WAGE2 veri seti (Wooldridge, 2020): 935 erkek çalışan, 1980")]
        if order != "dosya":
            operations.append(SortRows(WAGE2, order, f"Satırlar {WAGE2_ORDERS[order]} göre küçükten büyüğe sıralanır"))
        operations += [
            Shape(WAGE2, "n", "k"),
            ShowFrame(WAGE2, WAGE2_COLUMNS, "İlk üç ve son gözlem" if order == "dosya"
                      else "Sıralamadan sonra 1, 2, 3 ve 935 numaralı satırlar", rows=rows),
        ]
        return tuple(operations)

    names = {"wage": "aylık kazanç", "educ": "eğitim", "exper": "iş deneyimi", "tenure": "kıdem"}
    return interactive_step(
        number=1,
        title=f"{_TITLES[1]}: WAGE2",
        note=_NOTES[1],
        explanation=(
            "Yatay kesit verisi çok sayıda birimin belirli bir zamanda gözlenmesiyle oluşur. WAGE2'de 1980'de gözlenen "
            "935 erkek çalışanın aylık kazancı (ABD doları), eğitim yılı, iş deneyimi, mevcut işverendeki kıdemi ve "
            "başka özellikleri vardır. Yatay kesitte satırların çoğu zaman ekonomik bir zaman sırası yoktur; satır "
            "sırasını değiştirerek tabloda hangi çalışanların göründüğüne bakabilirsiniz."
        ),
        controls=(
            Choice("adim1_sira", "Satır sırası",
                   (("dosya", "Veri dosyasındaki sıra"), ("wage", "Aylık kazanca göre"), ("educ", "Eğitime göre")),
                   "dosya", help="Varsayılan: veri dosyasındaki sıra."),
        ),
        build=build,
        checks=(_check("Çalışan sayısı", ScalarTarget("n"), 0),) + tuple(
            _check(f"{row}. gözlem, {names[column]}", CellTarget(WAGE2, column, row), 0)
            for row in rows for column in WAGE2_COLUMNS
        ),
        note_for=lambda state, choices: _cross_section_note(choices["adim1_sira"], WAGE2_ORDERS, 935),
    )


OKUN_COLUMNS = ("year", "pcrgdp", "unem")
OKUN_SERIES = {"pcrgdp": "Reel GSYH büyümesi", "unem": "İşsizlik oranı"}


def _series_note(state, frame: str, time: str, order: str, series, names: dict[str, str], first_period: str,
                 last_period: str, row_word: str) -> str:
    """``names``: değişkenlerin cümle içindeki adları; ``row_word``: "Her satır bir yıldır" gibi cümle başı."""

    if order == time:
        if len(series) > 1:
            chart = ("Grafik serilerin zaman içindeki hareketini bu sırayla gösterir. Serilerin bazı dönemlerde birlikte "
                     "ya da ters yönde hareket etmesi tek başına nedensellik göstermez.")
        else:
            chart = f"Grafik {names[series[0]]} serisinin zaman içindeki hareketini gösterir."
        return (f"{row_word} ve satırlar zaman sırasındadır ({first_period}–{last_period}): her dönem bir öncekinden "
                f"sonra gelir. {chart}")
    shown = ", ".join(_shown(value) for value in state.frames[frame][time].head(6))
    rest = "; diğer serilerin zaman içindeki hareketi kaybolur" if len(series) > 1 or order not in series else ""
    return (f"Satırlar {names[order]} değerine göre sıralandığında ilk altı satırın dönemleri {shown} olur; dönemler "
            "artık ardışık değildir. Grafiğin yatay ekseni dönem değil satır sırasıdır"
            + (f": {names[order]} serisi azalmayan, basamaklı bir çizgiye dönüşür" if order in series else "")
            + f"{rest}. Zaman sırası bozulduğunda veri setinin önemli bir özelliği kaybolur; yatay kesitte (Adım 1) "
            "sıralama veri yapısını değiştirmez.")


def _alt_step2() -> LabStep:
    def build(choices) -> tuple:
        order, series = choices["adim2_sira"], choices["adim2_seriler"]
        operations: list = [
            LoadWooldridge(OKUN, "OKUN veri seti (Wooldridge, 2020): ABD'de yıllık büyüme ve işsizlik",
                           columns=OKUN_COLUMNS),
        ]
        x, x_label = "year", "Yıl"
        if order != "year":
            operations += [
                SortRows(OKUN, order, f"Satırlar {tr_lower(OKUN_SERIES[order])} değerine göre küçükten büyüğe sıralanır"),
                Derive(OKUN, "sira", E.seq(E.var("year")), "Sıralamadan sonraki satır numarası 1, 2, …, n"),
            ]
            x, x_label = "sira", f"Satır sırası ({tr_lower(OKUN_SERIES[order])} değerine göre sıralı)"
        operations.append(ShowFrame(OKUN, OKUN_COLUMNS, "İlk altı gözlem" if order == "year"
                                    else "Sıralamadan sonra ilk altı satır", head=6))
        first, *others = series
        if others:
            y_label, title = "Yüzde (%)", ("OKUN: ABD'de reel GSYH büyümesi ve işsizlik oranı" if order == "year"
                                           else f"OKUN: satırlar {tr_lower(OKUN_SERIES[order])} değerine göre sıralı")
        else:
            y_label = f"{OKUN_SERIES[first]} (%)"
            name = tr_lower(OKUN_SERIES[first])
            title = (f"OKUN: yıllık {name}" if order == "year"
                     else f"OKUN: {name}, satırlar {tr_lower(OKUN_SERIES[order])} değerine göre sıralı")
        operations += [
            LineChart(OKUN, x, first, x_label, y_label, title, markers=order != "year",
                      series=tuple((column, OKUN_SERIES[column]) for column in others),
                      legend=OKUN_SERIES[first] if others else ""),
            Statistic(OKUN, "year", "min", "ilk_yil", "İlk yıl", decimals=0),
            Statistic(OKUN, "year", "max", "son_yil", "Son yıl", decimals=0),
        ]
        return tuple(operations)

    def note(state, choices) -> str:
        first, last = sayi(state.scalars["ilk_yil"], 0), sayi(state.scalars["son_yil"], 0)
        return _series_note(state, OKUN, "year", choices["adim2_sira"], choices["adim2_seriler"],
                            {key: tr_lower(value) for key, value in OKUN_SERIES.items()}, first, last,
                            "Her satır bir yıldır")

    names = {"year": "yıl", "pcrgdp": "reel GSYH büyümesi", "unem": "işsizlik oranı"}
    return interactive_step(
        number=2,
        title=f"{_TITLES[2]}: OKUN",
        note=_NOTES[2],
        explanation=(
            "Zaman serisi bir veya daha fazla değişkenin ardışık dönemlerde gözlenmesiyle oluşur. OKUN, 1959–2005 "
            "dönemi için ABD'nin yıllık reel GSYH büyümesini (`pcrgdp`) ve işsizlik oranını (`unem`) yüzde olarak "
            "içerir. Her satır bir yıldır ve sıralama ekonomik bilgi taşır."
        ),
        controls=(
            MultiChoice("adim2_seriler", "Grafikteki seriler", tuple(OKUN_SERIES.items()), ("pcrgdp", "unem"),
                        help="Varsayılan: iki seri birlikte."),
            Choice("adim2_sira", "Satır sırası",
                   (("year", "Zaman sırası (yıl)"), ("unem", "İşsizlik oranına göre sıralı")), "year",
                   help="Varsayılan: veri zaman sırasındadır."),
        ),
        build=build,
        checks=tuple(
            _check(f"{row}. gözlem, {names[column]}", CellTarget(OKUN, column, row), 0 if column == "year" else 1)
            for row in range(1, 7) for column in OKUN_COLUMNS
        ) + (_check("İlk yıl", ScalarTarget("ilk_yil"), 0), _check("Son yıl", ScalarTarget("son_yil"), 0)),
        note_for=lambda state, choices: note(state, choices),
    )


KIELMC_VARIABLES = ("price", "rprice", "age", "rooms", "area", "baths", "dist")
KIELMC_DECIMALS = {"price": 0, "rprice": 0, "area": 0, "dist": 0}
KIELMC_GROUPINGS = {"year": ("year",), "year_nearinc": ("year", "nearinc")}
_KIELMC_UNITS = {"price": " dolar", "rprice": " dolar", "age": " yıl", "area": " fit²", "dist": " fit"}


def _alt_step3() -> LabStep:
    def build(choices) -> tuple:
        variable, grouping = choices["adim3_degisken"], choices["adim3_gruplama"]
        groups = "yıllara" if grouping == "year" else "yıl ve tesise yakınlığa"
        return (
            LoadWooldridge(KIELMC, "KIELMC veri seti (Wooldridge, 2020): 1978 ve 1981'de satılan 321 konut"),
            GroupStats(KIELMC, KIELMC_GROUPINGS[grouping], variable, ("count", "mean"), "donem_ozeti",
                       f"{groups.capitalize()} göre gözlem sayısı ve ortalama: {W.variable(KIELMC, variable).label}",
                       decimals=KIELMC_DECIMALS.get(variable, 2)),
        )

    def note(state, choices) -> str:
        variable, grouping = choices["adim3_degisken"], choices["adim3_gruplama"]
        table = state.tables["donem_ozeti"]
        if grouping == "year":
            counts = f"1978 örnekleminde {sayim(table.loc[1978, 'count'])}, 1981 örnekleminde {sayim(table.loc[1981, 'count'])} konut vardır"
            digits = KIELMC_DECIMALS.get(variable, 2)
            unit = _KIELMC_UNITS.get(variable, "")
            means = (f"Ortalama {tr_lower(W.variable(KIELMC, variable).label)} 1978 örnekleminde "
                     f"{sayi(table.loc[1978, 'mean'], digits)}{unit}, 1981 örnekleminde "
                     f"{sayi(table.loc[1981, 'mean'], digits)}{unit}.")
        else:
            counts = "Her satır bir yıldaki bir yakınlık grubudur (satır adındaki ikinci sayı: 1 tesise 3 mil içinde, 0 uzak)"
            means = ""
        text = (f"{counts}; iki yılın satışları ayrı ayrı gözlenmiştir ve aynı konutlar izlenmez: yıllar arasındaki fark "
                f"aynı konutların değer değişimi değil, iki ayrı örneklemin farkıdır. {means}").strip()
        if variable == "price":
            text += (" Fiyatlar cari fiyatlarladır: 1978 ile 1981 arasındaki farkın bir kısmı genel fiyat düzeyindeki "
                     "artıştır. 1978 fiyatlarıyla ölçülen satış fiyatını (`rprice`) seçerek karşılaştırın.")
        elif variable == "rprice":
            text += " 1978 fiyatlarıyla ölçülen fiyat, genel fiyat artışını ayıklar; kalan fark yine iki ayrı satış örnekleminin farkıdır."
        else:
            text += " Konut özelliklerindeki fark, iki yılda satılan konutların bileşiminin de değiştiğini gösterebilir."
        return text

    return interactive_step(
        number=3,
        title=f"{_TITLES[3]}: KIELMC",
        note=_NOTES[3],
        explanation=(
            "KIELMC, Massachusetts'in North Andover kasabasında 1978 ve 1981 yıllarında satılan konutların iki ayrı "
            "örneklemini tek veri setinde birleştirir; 1978'de satılan bir konut 1981'de aynı konut olarak izlenmez. "
            "`year` satış yılıdır; `nearinc` konutun çöp yakma tesisine 3 mil ya da daha yakın olduğunu gösterir. "
            "Satış fiyatı `price` cari fiyatlarla, `rprice` 1978 fiyatlarıyla ölçülür (ABD doları)."
        ),
        controls=(
            Choice("adim3_degisken", "Özetlenen değişken", W.options(KIELMC, KIELMC_VARIABLES), "price",
                   help="Varsayılan: satış fiyatı (cari fiyatlarla)."),
            Choice("adim3_gruplama", "Gruplama", (("year", "Yıl"), ("year_nearinc", "Yıl × tesise yakınlık")), "year",
                   help="Varsayılan: yalnız yıl."),
        ),
        build=build,
        checks=(
            _check("1978 gözlem sayısı", TableTarget("donem_ozeti", 1978, "count"), 0),
            _check("1981 gözlem sayısı", TableTarget("donem_ozeti", 1981, "count"), 0),
            _check("1978 ortalama satış fiyatı", TableTarget("donem_ozeti", 1978, "mean"), 2),
            _check("1981 ortalama satış fiyatı", TableTarget("donem_ozeti", 1981, "mean"), 2),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


CRIME4_COLUMNS = ("county", "year", "crmrte", "polpc")
COUNTIES = (1, 3, 5, 7, 9, 11, 13, 15)
"""Veri setindeki ilk sekiz ilçenin kimliği (``county``)."""


def _panel_note(state, frame: str, unit: str, time: str, shown: str, value, words: tuple[str, str],
                first_rows: str) -> str:
    """Panel adımının yorumu. ``words``: birimlerin tamlayan adı ve dönem sözcüğü (ör. "ilçenin", "yıl");
    ``first_rows``: ilk on satırın düzenini anlatan cümle; ``value``: gösterilen birimin koddaki değeri."""

    unit_word, time_word = words
    table = state.tables["panel"]["deger"]
    data = state.frames[frame]
    units, rows, periods = int(table["birim"]), int(table["gozlem"]), int(data[time].nunique())
    if rows == units * periods:
        structure = (f"{sayim(units)} {unit_word} her biri {sayim(periods)} {time_word} gözlenir ({sayim(rows)} satır): "
                     "her birim her dönemde gözlendiği için panel dengelidir.")
    else:
        structure = (f"Birimler {sayim(periods)} dönemin hepsinde gözlenmez (birim başına en az "
                     f"{sayim(table['en_az_donem'])}, en çok {sayim(table['en_cok_donem'])} dönem; {sayim(units)} birim, "
                     f"{sayim(rows)} satır): panel dengesizdir.")
    if shown == "ilk10":
        return f"{first_rows} Gözlem birimi birim–dönem birleşimidir. {structure}"
    count = int((data[unit] == value).sum())
    if count < 2:
        return (f"Bu birim yalnız bir dönemde gözlenir; dönemden döneme değişimi okunamaz. {structure} Panel veri "
                "kullanmak tek başına bütün karşılaştırma sorunlarını çözmez.")
    return (f"Bu birim {sayim(count)} {time_word} gözlenir: değişkenlerin dönemden döneme değişimi bu satırlardan "
            "okunur. Havuzlanmış yatay kesitte (Adım 3) aynı birim izlenmediği için bu değişim gözlenemez. "
            f"{structure} Panel veri kullanmak tek başına bütün karşılaştırma sorunlarını çözmez.")


def _alt_step4() -> LabStep:
    def build(choices) -> tuple:
        shown = choices["adim4_gosterim"]
        frame = (ShowFrame(CRIME4, CRIME4_COLUMNS, "İlk on satır", head=10, decimals=4) if shown == "ilk10" else
                 ShowFrame(CRIME4, CRIME4_COLUMNS, f"county = {shown} numaralı ilçenin bütün gözlemleri",
                           where=("county", int(shown)), decimals=4))
        return (
            LoadWooldridge(CRIME4, "CRIME4 veri seti (Wooldridge, 2020): Kuzey Karolina'da 90 ilçe, 1981–1987"),
            frame,
            PanelSummary(CRIME4, "county", "year", "panel", "Panel yapısı: her ilçenin kaç farklı yılda gözlendiği"),
        )

    return interactive_step(
        number=4,
        title=f"{_TITLES[4]}: CRIME4",
        note=_NOTES[4],
        explanation=(
            "CRIME4'te aynı ilçeler 1981–1987 yılları arasında her yıl yeniden gözlenir; her satır bir ilçe–yıl "
            "birleşimidir. İlçeyi `county`, yılı `year` belirler (`year` = 81, …, 87); ikisi birlikte bir gözlemi "
            "tanımlar. `crmrte` kişi başına suç sayısı, `polpc` kişi başına polis sayısıdır."
        ),
        controls=(
            Choice("adim4_gosterim", "Gösterilen satırlar",
                   (("ilk10", "İlk on satır"), *((str(county), f"İlçe county = {county}") for county in COUNTIES)),
                   "ilk10", help="Varsayılan: ilk on satır."),
        ),
        build=build,
        checks=(
            _check("İlk yıl", TableTarget("panel", "ilk_donem", "deger"), 0),
            _check("Son yıl", TableTarget("panel", "son_donem", "deger"), 0),
            _check("İlçe sayısı", TableTarget("panel", "birim", "deger"), 0),
            _check("Gözlem sayısı (satır)", TableTarget("panel", "gozlem", "deger"), 0),
        ),
        note_for=lambda state, choices: _panel_note(
            state, CRIME4, "county", "year", choices["adim4_gosterim"],
            None if choices["adim4_gosterim"] == "ilk10" else int(choices["adim4_gosterim"]), ("ilçenin", "yıl"),
            "İlk satırlar aynı ilçeye (county = 1) aittir: her yıl bir satır."),
    )


_PROGRAM_SHARES = {"issiz": "program sonrası yıl boyunca işsiz kalanların payı", "kadin": "kadınların payı",
                   "evli": "evlilerin payı"}
_PROGRAM_EDUCATION_GAP = float(program_frame().groupby("program")["egitim"].mean().diff().iloc[-1])
"""Bu kurada program grubunun ortalama eğitim süresi eksi kontrol grubununki (yıl)."""
_PROGRAM_MEANS = {
    "kazanc": ("ortalama program sonrası yıllık kazancı", "bin TL"),
    "yas": ("ortalama yaşı", "yıl"),
    "egitim": ("ortalama eğitim süresi", "yıl"),
    "onceki_kazanc": ("ortalama program öncesi yıllık kazancı", "bin TL"),
}


def _experiment_operations(frame: str, group: str, variable: str, title: str, y_label: str, comment: str,
                           setup: tuple = ()) -> tuple:
    return (
        *setup,
        GroupStats(frame, (group,), variable, ("count", "mean"), "grup_ozeti", comment, decimals=4),
        Statistic(frame, variable, "mean", "ort_kontrol", "Kontrol grubu (atama = 0)", where=(group, 0), decimals=3),
        Statistic(frame, variable, "mean", "ort_program", "Program grubu (atama = 1)", where=(group, 1), decimals=3),
        Scalar("fark", E.sub(E.ref("ort_program"), E.ref("ort_kontrol")), "Gözlenen fark (program − kontrol)",
               decimals=3),
        BarChart("grup_ozeti", "mean", "Atama grubu (0 = kontrol, 1 = program)", y_label, title, decimals=3),
    )


def _alt_step5() -> LabStep:
    data = InlineData(PROGRAM, PROGRAM_COLUMNS, PROGRAM_ROWS,
                      "Kurgusal iş arama programı: 200 kişi, kurayla 100 program ve 100 kontrol")

    def build(choices) -> tuple:
        variable = choices["adim5_degisken"]
        label = PROGRAM_LABELS[variable]
        if variable in _PROGRAM_SHARES:
            title, y_label = f"Atama gruplarında {_PROGRAM_SHARES[variable]}", f"{label}: pay (0–1)"
        else:
            title, y_label = f"Atama gruplarının {_PROGRAM_MEANS[variable][0]}", f"Ortalama · {label}"
        comment = ("Grup ortalamaları: program sonrası yıllık kazanç" if variable == "kazanc"
                   else f"Gruplara göre gözlem sayısı ve ortalama: {label}")
        return _experiment_operations(PROGRAM, "program", variable, title, y_label, comment, setup=(data,))

    def note(state, choices) -> str:
        variable = choices["adim5_degisken"]
        control, treated, difference = (state.scalars[name] for name in ("ort_kontrol", "ort_program", "fark"))
        if variable == "kazanc":
            gap = _PROGRAM_EDUCATION_GAP
            return (f"Kontrol grubunun ortalama program sonrası yıllık kazancı yaklaşık {sayi(control, 3)} bin TL, program "
                    f"grubununki {sayi(treated, 3)} bin TL; gözlenen ortalama fark {sayi(difference, 3)} bin TL. Atama "
                    "kurayla yapıldığı için grup ortalamalarının farkı (tahmin edici), programa atanmanın ortalama "
                    "kazanca etkisini (parametre) tahmin eder ve gözlemsel olarak seçilmiş iki grubun farkından daha "
                    f"güçlü bir nedensel yorum temeli sunar; {sayi(difference, 3)} bin TL bu kuradaki tahmindir. Veri "
                    "kurgusal olduğu için parametre veri üretim sürecinden bilinir: gerçek ortalama etki yaklaşık "
                    f"{sayi(PROGRAM_EFFECT, 1)} bin TL. Tahminin bundan uzak olması şanstandır: kura grupları ortalamada "
                    "benzer kılar, fakat tek bir kurada gruplar şans eseri farklılaşabilir (bu kurada program grubunun "
                    f"ortalama eğitim süresi {sayi(abs(gap), 2)} yıl daha {'uzundur' if gap > 0 else 'kısadır'}; "
                    "karşılaştırılan değişkeni değiştirerek bakın) ve benzer kişilerin kazançları da şans eseri "
                    "farklıdır. Başka bir kura başka bir tahmin verirdi; tahminin kuradan kuraya ne kadar değiştiği "
                    "Konu 7'de ele alınır. Gerçek bir deneyde ayrıca atamanın uygulanıp uygulanmadığı, sonuç verisinin "
                    "eksik olup olmadığı ve deney örnekleminin kimleri temsil ettiği değerlendirilmelidir.")
        if variable in _PROGRAM_SHARES:
            text = (f"Kontrol grubunda {_PROGRAM_SHARES[variable]}: %{sayi(100 * control, 1)}; program grubunda: "
                    f"%{sayi(100 * treated, 1)}. Fark (program − kontrol): {sayi(100 * difference, 1)} yüzde puan.")
        else:
            phrase, unit = _PROGRAM_MEANS[variable]
            text = (f"Kontrol grubunun {phrase}: {sayi(control, 3)} {unit}; program grubununki: {sayi(treated, 3)} "
                    f"{unit}. Fark (program − kontrol): {sayi(difference, 3)} {unit}.")
        if variable == "issiz":
            return text + (" Bu değişken de programdan sonra ölçülen bir sonuçtur. Veri üretim sürecinde program, yıl boyunca "
                           f"işsiz kalma olasılığını %{sayi(100 * PROGRAM_UNEMPLOYMENT[0], 0)} düzeyinden "
                           f"%{sayi(100 * PROGRAM_UNEMPLOYMENT[1], 0)} düzeyine, yani "
                           f"{sayi(100 * (PROGRAM_UNEMPLOYMENT[0] - PROGRAM_UNEMPLOYMENT[1]), 0)} yüzde puan düşürür; "
                           "bu kuradaki fark onun tahminidir.")
        if variable in PROGRAM_OUTCOMES:
            return text + " Bu değişken de programdan sonra ölçülen bir sonuçtur."
        return text + (
            " Bu özellik kuradan önce ölçülmüştür; program onu değiştiremez. Rastgele atama grupların sistematik olarak "
            "farklı olmamasını hedefler, fakat örneklemde ortalamaların tam eşit olmasını garanti etmez. Böyle bir "
            "farkın şansla açıklanıp açıklanamayacağı hipotez testleriyle (Konu 7) değerlendirilir.")

    options = tuple((name, PROGRAM_LABELS[name]) for name in (*PROGRAM_OUTCOMES, *PROGRAM_PRE))
    return interactive_step(
        number=5,
        title=f"{_TITLES[5]}: kurgusal iş arama programı",
        note=_NOTES[5],
        explanation=(
            "Kurgusal bir deney: iş arayan 200 kişiden 100'ü kurayla bir iş arama programına atanır (`program` = 1), "
            "diğer 100 kişi kontrol grubudur (`program` = 0). Sonuç değişkeni `kazanc`, program sonrası yıllık kazançtır "
            "(bin TL; yıl boyunca işsiz kalanlarda 0). Veri bu uygulamada üretilmiştir ve kodun içinde yazılıdır; "
            "gerçek bir programın sonucu değildir; veri üretim süreci bilindiği için tahmin, programın gerçek ortalama "
            "etkisiyle karşılaştırılabilir. Bu adımda regresyon kurulmaz: iki grubun ortalamaları okunur ve farkları "
            "hesaplanır. Karşılaştırılan değişkeni değiştirerek kuradan önce ölçülen özelliklerde (yaş, eğitim, önceki "
            "kazanç, …) grupların ne kadar benzer olduğuna da bakabilirsiniz."
        ),
        controls=(
            Choice("adim5_degisken", "Gruplar arasında karşılaştırılan değişken", options, "kazanc",
                   help="İlk iki değişken programdan sonra, diğerleri kuradan önce ölçülür."),
        ),
        build=build,
        checks=(
            _check("Kontrol grubu gözlem sayısı", TableTarget("grup_ozeti", 0, "count"), 0),
            _check("Program grubu gözlem sayısı", TableTarget("grup_ozeti", 1, "count"), 0),
            _check("Kontrol grubu ortalaması", TableTarget("grup_ozeti", 0, "mean"), 4),
            _check("Program grubu ortalaması", TableTarget("grup_ozeti", 1, "mean"), 4),
            _check("Kontrol grubu ortalaması, yaklaşık", ScalarTarget("ort_kontrol"), 3),
            _check("Program grubu ortalaması, yaklaşık", ScalarTarget("ort_program"), 3),
            _check("Gözlenen ortalama fark", ScalarTarget("fark"), 3),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


def _alt_step6() -> LabStep:
    return LabStep(
        number=6,
        title=_TITLES[6],
        note=_NOTES[6],
        explanation=(
            "İki bulgu yüzeyde benzer görünür: WAGE2'de eğitim yılı yüksek çalışanların ortalama aylık kazancı daha "
            "yüksektir; kurgusal programda programa atanan grubun ortalama kazancı kontrol grubundan daha yüksektir. "
            "İlk veri gözlemseldir: çalışanlar eğitim yıllarına rastgele atanmadığı için güvenli ifade bir ilişki "
            "cümlesidir. İkinci örnekte atama kurayla yapıldığı için grup farkı daha güçlü bir nedensel yorum "
            "taşıyabilir.\n\n"
            "| Özellik | WAGE2: eğitim ve kazanç | Kurgusal program: program ve kazanç |\n"
            "|---|---|---|\n"
            "| Veri üretimi | Gözlemsel | Rastgele atamalı deney (kura) |\n"
            "| Temel karşılaştırma | Farklı eğitim düzeyindeki çalışanlar | Program ve kontrol grupları |\n"
            "| Başlıca kaygı | Grupların başka özelliklerde (ör. yetenek) farklı olması | Atamanın uygulanması ve "
            "genellenebilirlik |\n"
            "| Güvenli başlangıç dili | \"İlişkilidir\", \"ortalama olarak farklıdır\" | Tasarım uygunsa \"program ataması "
            "ortalama sonucu değiştirmiştir\" |\n\n"
            "Konu 3'te iki grubun ortalama farkının, 0/1 atama değişkeninin açıklayıcı olduğu basit regresyonun eğim "
            "katsayısına eşit olduğu gösterilir (notlarda JTRAIN2 ile)."
        ),
    )


def alternative_build() -> LabSpec:
    labels = dict(W.labels(WAGE2, OKUN, KIELMC, CRIME4))
    labels.update(PROGRAM_LABELS)
    labels.update({"year": "Yıl", "sira": "Satır sırası", "n": "Gözlem sayısı", "k": "Değişken sayısı",
                   "exper": "İş deneyimi (yıl)", "educ": "Eğitim (yıl)", "age": "Yaş (yıl)"})
    spec = LabSpec(
        topic_key=TOPIC,
        title=TITLE,
        note_section="2",
        steps=(_alt_step1(), _alt_step2(), _alt_step3(), _alt_step4(), _alt_step5(), _alt_step6()),
        labels=tuple(labels.items()),
        source="alternatif",
    )
    return with_app_values(spec)


@cache
def alternative() -> LabSpec:
    return alternative_build()


STORY = ("Her veri yapısı başka bir Wooldridge (2020) veri setiyle gösterilir: WAGE2 (yatay kesit), OKUN (zaman serisi), "
         "KIELMC (havuzlanmış yatay kesit) ve CRIME4 (panel). Deney adımı kurgusal bir iş arama programıdır (200 kişi, "
         "kurayla 100 program ve 100 kontrol).")


# --- Kendi verin ---------------------------------------------------------------------------------------------

def _numeric(case: Case) -> tuple[str, ...]:
    return tuple(dict.fromkeys((case.roles[SAYISAL], *case.extras)))


def _repeated_units(case: Case) -> bool:
    """Aynı birim birden çok dönemde gözlenir mi (``validate`` her birim–dönem birleşiminin tek satır olmasını sağlar)."""

    return case.has(BIRIM) and case.has(DONEM) and bool(case.data[case.roles[BIRIM]].duplicated().any())


def _structure(case: Case) -> str:
    """Verinin yapısı: aynı birimler birden çok dönemde gözleniyorsa panel; dönem sütunu her satırda farklıysa zaman
    serisi, değilse havuzlanmış yatay kesit; dönem sütunu yoksa yatay kesit. Birim kimliği seçilip hiçbir birim
    tekrar etmiyorsa veri panel sayılmaz."""

    if _repeated_units(case):
        return "panel"
    if case.has(DONEM):
        return "zaman" if case.data[case.roles[DONEM]].is_unique else "havuz"
    return "kesit"


def _single_units(case: Case) -> str:
    """Birim kimliği seçildi, ama hiçbir birim tekrar etmiyor: açıklama cümlesi (yoksa boş)."""

    if not case.has(BIRIM) or _repeated_units(case):
        return ""
    return (f" Seçtiğiniz birim kimliğinde ({_quoted(case, case.roles[BIRIM])}) her birim yalnız bir kez görünür: aynı "
            "birimler yeniden gözlenmediği için veri panel değildir.")


def _balanced(case: Case) -> bool:
    data = case.data
    unit, time = case.roles[BIRIM], case.roles[DONEM]
    return len(data) == data[unit].nunique() * data[time].nunique()


def _first_rows(case: Case) -> str:
    """Panelin ilk on satırının düzeni, dosyadaki sırayla (birime ya da döneme göre dizilmiş olabilir)."""

    unit, time = case.roles[BIRIM], case.roles[DONEM]
    head = case.data.head(10)
    if head[unit].iloc[0] == head[unit].iloc[1]:
        return (f"İlk satırlar aynı birime ({md(case.name(unit))} = {_shown(head[unit].iloc[0])}) aittir: her dönem bir "
                "satır.")
    if head[time].iloc[0] == head[time].iloc[1]:
        return (f"Dosyanız dönemlere göre dizilmiş: ilk satırlar aynı dönemin ({md(case.name(time))} = "
                f"{_shown(head[time].iloc[0])}) farklı birimleridir.")
    return "Dosyadaki satırlar birime ya da döneme göre dizilmemiş."


def _value(case: Case, column: str, value):
    """Koşul değeri koddaki türüyle: sayısal sütunda sayı, metin sütununda metin."""

    if pd.api.types.is_numeric_dtype(case.data[column]):
        number = float(value)
        return int(number) if number.is_integer() else number
    return str(value)


def _units(case: Case) -> tuple:
    column = case.roles[BIRIM]
    return tuple(pd.unique(case.data[column].dropna()))[:SHOWN_UNITS]


def _display(case: Case, column: str) -> str:
    return case.name(column)


def _quoted(case: Case, column: str) -> str:
    return f"“{md(case.name(column))}”"


def _own_step1(case: Case) -> LabStep:
    structure = _structure(case)
    numeric = _numeric(case)
    shown = tuple(dict.fromkeys((*(case.roles[role] for role in (BIRIM, DONEM) if case.has(role)), *numeric[:4])))
    orders = {column: f"“{case.name(column)}”" for column in numeric}
    if structure in ("panel", "havuz"):
        period = case.data[case.roles[DONEM]].min()
        period_value = _value(case, case.roles[DONEM], period)
        rows_in_frame = int((case.data[case.roles[DONEM]] == period).sum())
        setup = (TakeRows("kesit", case.frame, f"Tek bir dönemin gözlemleri: {case.name(case.roles[DONEM])} = "
                          f"{kisa(float(period), 6)}", where=(case.roles[DONEM], period_value)),)
        what = ("Dosyanızda aynı birimler birden çok dönemde gözlenir (panel veri)" if structure == "panel" else
                "Dosyanızda birden çok dönemin gözlemleri bir aradadır")
        lead = (f"{what}; tek bir dönemin satırları bir yatay kesittir. Bu adımda ilk dönemin "
                f"({md(case.name(case.roles[DONEM]))} = {_shown(period)}) gözlemleri kullanılır.{_single_units(case)}")
    else:
        rows_in_frame = len(case.data)
        setup = (CopyFrame("kesit", case.frame, "Verinin kopyası: sıralama özgün veriyi değiştirmesin"),)
        lead = ("Dosyanızın her satırı bir dönemdir: veri bir zaman serisidir, yatay kesit değildir. Satırları aşağıda "
                "yine de sıralayabilirsiniz; zaman serisinde sıralamanın etkisini Adım 2 gösterir."
                if structure == "zaman" else
                "Yatay kesit verisi çok sayıda birimin belirli bir zamanda gözlenmesiyle oluşur; dosyanızda dönem "
                "sütunu seçilmediği için her satır ayrı bir gözlem birimi sayılır.")
    rows = tuple(sorted({1, 2, 3, rows_in_frame} & set(range(1, rows_in_frame + 1))))

    def build(choices) -> tuple:
        order = choices["adim1_sira"]
        operations: list = [*case.load, *setup]
        if order != "dosya":
            operations.append(SortRows("kesit", order, f"Satırlar {orders[order]} değerine göre küçükten büyüğe "
                                                       "sıralanır"))
        operations += [
            Shape("kesit", "n", "k"),
            ShowFrame("kesit", shown, "İlk üç ve son gözlem" if order == "dosya"
                      else f"Sıralamadan sonra {liste([str(row) for row in rows])} numaralı satırlar", rows=rows),
        ]
        return tuple(operations)

    def note(state, choices) -> str:
        order = choices["adim1_sira"]
        n = int(state.scalars["n"])
        if order == "dosya":
            return (f"Tabloda {sayim(n)} gözlemin yalnız {liste([str(row) for row in rows])} numaralı satırları gösterilir. "
                    "Yatay kesitte satırların sırası çoğu zaman ekonomik bilgi taşımaz; sıralamayı değiştirerek "
                    "tabloda hangi gözlemlerin göründüğüne bakabilirsiniz.")
        return (f"Satırlar {md(orders[order])} değerine göre sıralandı: aynı satır numaralarında artık başka gözlemler "
                "görünür. Satırların yeri değişti, veri yapısı değişmedi. Zaman serisinde satır sırası ekonomik bilgi "
                "taşır (Adım 2).")

    options = (("dosya", "Veri dosyasındaki sıra"), *((column, f"{case.name(column)} değerine göre") for column in numeric))
    return interactive_step(
        number=1,
        title=_TITLES[1],
        note=_NOTES[1],
        explanation=lead,
        controls=(Choice("adim1_sira", "Satır sırası", options, "dosya", help="Varsayılan: dosyadaki sıra."),),
        build=build,
        checks=(_check("Gözlem sayısı", ScalarTarget("n"), 0),) + tuple(
            _check(f"{row}. gözlem, {case.name(column)}", CellTarget("kesit", column, row), 4)
            for row in rows for column in numeric[:2]
        ),
        note_for=lambda state, choices: note(state, choices),
    )


def _own_step2(case: Case) -> LabStep:
    structure = _structure(case)
    if structure not in ("zaman", "panel"):
        need = ("her dönemde tek gözlem olan bir dönem sütunu (zaman serisi) ya da dönem ve birim kimliği (panel)"
                if structure == "havuz" else "bir dönem (zaman) sütunu")
        return LabStep(number=2, title=_TITLES[2], note=_NOTES[2],
                       explanation=("Zaman serisi bir veya daha fazla değişkenin ardışık dönemlerde gözlenmesiyle "
                                    f"oluşur. Bu adım için dosyanızda {need} seçilmelidir.{_single_units(case)}"))
    time = case.roles[DONEM]
    numeric = _numeric(case)
    labels = {column: case.name(column) for column in numeric}
    controls: list = [
        MultiChoice("adim2_seriler", "Grafikteki seriler", tuple(labels.items()), numeric[:2],
                    help="Varsayılan: ilk iki sayısal değişken.", maximum=4),
        Choice("adim2_sira", "Satır sırası", ((time, f"Zaman sırası ({case.name(time)})"),
                                              *((column, f"{case.name(column)} değerine göre sıralı") for column in numeric)),
               time, help="Varsayılan: zaman sırası."),
    ]
    units = _units(case) if structure == "panel" else ()
    if structure == "panel":
        unit = case.roles[BIRIM]
        controls.insert(0, Choice("adim2_birim", f"Gösterilen birim ({case.name(unit)})",
                                  tuple((_key(value), str(value) if isinstance(value, str) else kisa(float(value), 6))
                                        for value in units), _key(units[0]),
                                  help="Panelde her birimin kendi zaman serisi vardır."))

    def build(choices) -> tuple:
        order, series = choices["adim2_sira"], tuple(choices["adim2_seriler"])
        if structure == "panel":
            value = _value(case, case.roles[BIRIM], choices["adim2_birim"])
            setup = TakeRows("seri", case.frame, f"Bir birimin bütün dönemleri: {case.name(case.roles[BIRIM])} = "
                                                 f"{value}", where=(case.roles[BIRIM], value))
        else:
            setup = CopyFrame("seri", case.frame, "Verinin kopyası: sıralama özgün veriyi değiştirmesin")
        operations: list = [setup, SortRows("seri", time, f"Satırlar zaman sırasına ({case.name(time)}) dizilir")]
        x, x_label = time, case.name(time)
        if order != time:
            sira = free_name("sira", case.data.columns)
            operations += [
                SortRows("seri", order, f"Satırlar “{case.name(order)}” değerine göre küçükten büyüğe sıralanır"),
                Derive("seri", sira, E.seq(E.var(time)), "Sıralamadan sonraki satır numarası 1, 2, …, n"),
            ]
            x, x_label = sira, f"Satır sırası ({case.name(order)} değerine göre sıralı)"
        operations.append(ShowFrame("seri", (time, *numeric[:4]), "İlk altı gözlem" if order == time
                                    else "Sıralamadan sonra ilk altı satır", head=6))
        first, *others = series
        operations += [
            LineChart("seri", x, first, x_label, "Değer" if others else case.name(first),
                      "Zaman serisi" + ("" if order == time else f": satırlar {case.name(order)} değerine göre sıralı"),
                      markers=order != time, series=tuple((column, labels[column]) for column in others),
                      legend=labels[first] if others else ""),
            Statistic("seri", time, "min", "ilk_donem", "İlk dönem", decimals=0),
            Statistic("seri", time, "max", "son_donem", "Son dönem", decimals=0),
        ]
        return tuple(operations)

    def note(state, choices) -> str:
        first, last = _shown(state.scalars["ilk_donem"]), _shown(state.scalars["son_donem"])
        names = {column: _quoted(case, column) for column in (*numeric, time)}
        return _series_note(state, "seri", time, choices["adim2_sira"], tuple(choices["adim2_seriler"]), names, first,
                            last, "Her satır bir dönemdir")

    lead = ("Zaman serisi bir veya daha fazla değişkenin ardışık dönemlerde gözlenmesiyle oluşur. "
            + ("Panel veride her birimin kendi zaman serisi vardır; aşağıda seçilen birimin dönemleri kullanılır."
               if structure == "panel" else "Dosyanızın her satırı bir dönemdir ve sıralama ekonomik bilgi taşır."))
    return interactive_step(
        number=2,
        title=_TITLES[2],
        note=_NOTES[2],
        explanation=lead,
        controls=tuple(controls),
        build=build,
        checks=(_check("İlk dönem", ScalarTarget("ilk_donem"), 0), _check("Son dönem", ScalarTarget("son_donem"), 0)),
        note_for=lambda state, choices: note(state, choices),
    )


def _own_step3(case: Case) -> LabStep:
    structure = _structure(case)
    if structure not in ("havuz", "panel"):
        need = ("her dönemde birden çok gözlem" if structure == "zaman" else "bir dönem (zaman) sütunu")
        return LabStep(number=3, title=_TITLES[3], note=_NOTES[3],
                       explanation=("Havuzlanmış yatay kesit, farklı dönemlerde ayrı ayrı seçilmiş örneklemleri tek "
                                    f"veri setinde birleştirir. Bu adım için dosyanızda {need} olmalıdır."))
    time = case.roles[DONEM]
    numeric = _numeric(case)
    periods = sorted(case.data[time].unique())

    def build(choices) -> tuple:
        variable = choices["adim3_degisken"]
        return (GroupStats(case.frame, (time,), variable, ("count", "mean"), "donem_ozeti",
                           f"Dönemlere göre gözlem sayısı ve ortalama: {case.name(variable)}", decimals=2),)

    def note(state, choices) -> str:
        table = state.tables["donem_ozeti"]
        counts = liste([f"{_shown(period)}: {sayim(table.loc[period, 'count'])}" for period in table.index[:6]])
        variable = choices["adim3_degisken"]
        if structure == "panel":
            if _balanced(case) and case.data[variable].notna().all():
                same = ("Dosyanız dengeli bir paneldir: her dönemde aynı birimler gözlenir. Bu yüzden iki dönemin "
                        "ortalamaları arasındaki fark, aynı birimlerin o iki dönem arasındaki ortalama değişimine eşittir.")
            else:
                reason = ("birimlerin hepsi her dönemde gözlenmediği için" if not _balanced(case) else
                          f"{_quoted(case, variable)} sütununda boş hücreler olduğu için")
                same = (f"Dosyanız bir paneldir, fakat {reason} dönem ortalamalarının farkı aynı birimlerin ortalama "
                        "değişimine eşit olmayabilir: dönemden döneme ortalamaya giren birimler de değişir.")
            return (f"Dönemlere göre gözlem sayıları {counts}. {same} Havuzlanmış yatay kesitte ise her dönemin örneklemi "
                    "ayrı seçilir ve aynı birimler izlenmez; panel yapısı Adım 4'te incelenir.")
        return (f"Dönemlere göre gözlem sayıları {counts}. Her dönemin örneklemi ayrı seçildiyse aynı birimler izlenmez: "
                "dönemler arasındaki fark aynı birimlerin değişimi değil, ayrı örneklemlerin farkıdır. Fark, "
                "örneklemlerin bileşimi ve genel koşullar (ör. fiyat düzeyi) değiştiği için de ortaya çıkabilir."
                + _single_units(case))

    return interactive_step(
        number=3,
        title=_TITLES[3],
        note=_NOTES[3],
        explanation=(f"Gözlemler {_quoted(case, time)} sütununun {len(periods)} değerine göre gruplanır; her grup bir "
                     "dönemin gözlemleridir."),
        controls=(Choice("adim3_degisken", "Özetlenen değişken", tuple((c, case.name(c)) for c in numeric), numeric[0],
                         help="Varsayılan: sayısal değişken."),),
        build=build,
        checks=tuple(_check(f"{kisa(float(period), 6)}: gözlem sayısı",
                            TableTarget("donem_ozeti", _value(case, time, period), "count"), 0) for period in periods[:2])
        + tuple(_check(f"{kisa(float(period), 6)}: ortalama", TableTarget("donem_ozeti", _value(case, time, period), "mean"),
                       4) for period in periods[:2]),
        note_for=lambda state, choices: note(state, choices),
    )


def _own_step4(case: Case) -> LabStep:
    if _structure(case) != "panel":
        need = (_single_units(case).strip() + " Bu adım için aynı birimlerin birden çok dönemde gözlendiği bir dosya "
                "gerekir." if case.has(BIRIM) and case.has(DONEM) else
                "Bu adım için dosyanızda bir birim kimliği (ör. il, firma, kişi numarası) ve bir dönem sütunu "
                "seçilmelidir.")
        return LabStep(number=4, title=_TITLES[4], note=_NOTES[4],
                       explanation=("Panel veride aynı birimler birden çok dönemde yeniden gözlenir; her satır bir "
                                    f"birim–dönem birleşimidir. {need}"))
    unit, time = case.roles[BIRIM], case.roles[DONEM]
    columns = (unit, time, *_numeric(case)[:3])
    units = _units(case)

    def build(choices) -> tuple:
        shown = choices["adim4_gosterim"]
        frame = (ShowFrame(case.frame, columns, "İlk on satır", head=10) if shown == "ilk10" else
                 ShowFrame(case.frame, columns, f"{case.name(unit)} = {shown} biriminin bütün gözlemleri",
                           where=(unit, _value(case, unit, shown))))
        return (frame, PanelSummary(case.frame, unit, time, "panel",
                                    "Panel yapısı: her birimin kaç farklı dönemde gözlendiği"))

    return interactive_step(
        number=4,
        title=_TITLES[4],
        note=_NOTES[4],
        explanation=(f"Dosyanızda birimi {_quoted(case, unit)}, dönemi {_quoted(case, time)} belirler; ikisi birlikte bir "
                     "gözlemi tanımlar."),
        controls=(Choice("adim4_gosterim", "Gösterilen satırlar",
                         (("ilk10", "İlk on satır"),
                          *((_key(value), f"{case.name(unit)} = {value if isinstance(value, str) else kisa(float(value), 6)}")
                            for value in units)),
                         "ilk10", help="Varsayılan: ilk on satır."),),
        build=build,
        checks=(
            _check("İlk dönem", TableTarget("panel", "ilk_donem", "deger"), 0),
            _check("Son dönem", TableTarget("panel", "son_donem", "deger"), 0),
            _check("Birim sayısı", TableTarget("panel", "birim", "deger"), 0),
            _check("Gözlem sayısı (satır)", TableTarget("panel", "gozlem", "deger"), 0),
        ),
        note_for=lambda state, choices: _panel_note(
            state, case.frame, unit, time, choices["adim4_gosterim"],
            None if choices["adim4_gosterim"] == "ilk10" else _value(case, unit, choices["adim4_gosterim"]),
            ("birimin", "dönem"), _first_rows(case)),
    )


def _own_step5(case: Case) -> LabStep:
    if not case.has(ATAMA):
        return LabStep(number=5, title=_TITLES[5], note=_NOTES[5],
                       explanation=("Rastgele atamalı deneyde birimler kurayla iki gruba ayrılır ve grupların ortalama "
                                    "sonucu karşılaştırılır. Bu adım için dosyanızda iki kategorili bir atama sütunu "
                                    "(ör. program/kontrol, 1/0) ve bir sonuç sütunu seçilmelidir."))
    assigned = case.roles[ATAMA]
    treated = case.levels[ATAMA]
    other = next(item for item in case.orders[assigned] if item != treated)
    group = free_name("atama01", case.data.columns)
    outcome = case.roles[DENEY_SONUC]
    variables = tuple(dict.fromkeys((outcome, *_numeric(case))))
    coding = MapCodes(case.frame, assigned, group, ((other, 0), (treated, 1)),
                      f"Atama göstergesi: {other} = 0 (kontrol), {treated} = 1 (program)")

    def build(choices) -> tuple:
        variable = choices["adim5_degisken"]
        setup: tuple = (coding,)
        frame = case.frame
        if case.data[variable].isna().any():
            frame = "deney"
            setup += (CompleteCases("deney", case.frame, (group, variable), f"Tam gözlemler: {case.name(variable)}"),)
        return _experiment_operations(frame, group, variable, f"Atama gruplarına göre ortalama: {case.name(variable)}",
                                      f"Ortalama · {case.name(variable)}", f"Gruplara göre gözlem sayısı ve ortalama: "
                                      f"{case.name(variable)}", setup=setup)

    def note(state, choices) -> str:
        variable = choices["adim5_degisken"]
        control, program, difference = (state.scalars[name] for name in ("ort_kontrol", "ort_program", "fark"))
        text = (f"Kontrol grubunda ({md(other)}) ortalama {_quoted(case, variable)}: {sayi(control, 3)}; program grubunda "
                f"({md(treated)}): {sayi(program, 3)}. Fark (program − kontrol): {sayi(difference, 3)}. ")
        if variable == outcome:
            return text + ("Atama gerçekten kurayla yapıldıysa bu fark programın ortalama sonuca etkisinin bir tahminidir; "
                           "atama rastgele değilse fark yalnız bir ilişki cümlesiyle yazılır.")
        return text + ("Bu değişken atamadan önce ölçüldüyse rastgele atama grupların bu özellikte sistematik olarak "
                       "farklı olmamasını hedefler; örneklemde tam eşitlik beklenmez. Farkın şansla açıklanıp "
                       "açıklanamayacağı hipotez testleriyle (Konu 7) değerlendirilir.")

    return interactive_step(
        number=5,
        title=_TITLES[5],
        note=_NOTES[5],
        explanation=(f"Atama sütunu {_quoted(case, assigned)}: {md(treated)} = 1 (program), {md(other)} = 0 (kontrol). Sonuç "
                     f"değişkeni {_quoted(case, outcome)}. Bu adımda regresyon kurulmaz: iki grubun ortalamaları okunur ve "
                     "farkları hesaplanır. Diğer sayısal değişkenleri seçerek grupların başka özelliklerde ne kadar "
                     "benzer olduğuna da bakabilirsiniz."),
        controls=(Choice("adim5_degisken", "Gruplar arasında karşılaştırılan değişken",
                         tuple((column, case.name(column)) for column in variables), outcome,
                         help="Varsayılan: deneyin sonuç değişkeni."),),
        build=build,
        checks=(
            _check("Kontrol grubu gözlem sayısı", TableTarget("grup_ozeti", 0, "count"), 0),
            _check("Program grubu gözlem sayısı", TableTarget("grup_ozeti", 1, "count"), 0),
            _check("Kontrol grubu ortalaması", ScalarTarget("ort_kontrol"), 4),
            _check("Program grubu ortalaması", ScalarTarget("ort_program"), 4),
            _check("Gözlenen fark", ScalarTarget("fark"), 4),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


def _own_step6(case: Case) -> LabStep:
    experiment = ("Dosyanızdaki atama gerçekten kurayla yapıldıysa grup farkı daha güçlü bir nedensel yorum taşıyabilir; "
                  "atama kişilerin kendi seçimiyse ya da başka bir ölçüte göre yapıldıysa grup farkı da bir ilişki "
                  "cümlesiyle yazılır." if case.has(ATAMA) else
                  "Dosyanızda bir atama sütunu yoksa verinin nasıl üretildiğini düşünün: gözlemsel veride güvenli ifade "
                  "bir ilişki cümlesidir.")
    return LabStep(
        number=6,
        title=_TITLES[6],
        note=_NOTES[6],
        explanation=(
            "Bir bulgunun dili verinin nasıl üretildiğine bağlıdır. Gözlemsel veride birimler karşılaştırılan "
            "özelliğe rastgele atanmadığı için güvenli ifade bir ilişki cümlesidir (\"ilişkilidir\", \"ortalama olarak "
            f"farklıdır\"). {experiment}\n\n"
            "| Özellik | Gözlemsel veri | Rastgele atamalı deney |\n"
            "|---|---|---|\n"
            "| Temel karşılaştırma | Farklı özellikteki birimler | Program ve kontrol grupları |\n"
            "| Başlıca kaygı | Grupların başka özelliklerde farklı olması | Atamanın uygulanması ve genellenebilirlik |\n"
            "| Güvenli başlangıç dili | \"İlişkilidir\" | Tasarım uygunsa \"program ataması ortalama sonucu "
            "değiştirmiştir\" |"
        ),
    )


def custom_build(case: Case) -> LabSpec:
    labels = {column: case.name(column) for column in case.labels}
    labels.update({"n": "Gözlem sayısı", "k": "Değişken sayısı", "sira": "Satır sırası"})
    if case.has(ATAMA):
        labels[free_name("atama01", case.data.columns)] = f"{case.name(case.roles[ATAMA])}: atama (1/0)"
    spec = LabSpec(
        topic_key=TOPIC,
        title=TITLE,
        note_section="2",
        steps=(_own_step1(case), _own_step2(case), _own_step3(case), _own_step4(case), _own_step5(case),
               _own_step6(case)),
        labels=tuple(labels.items()),
        source="kendi",
    )
    return with_app_values(spec)


def validate(case: Case) -> None:
    if case.has(DONEM) and case.roles[DONEM] in _numeric(case):
        raise K.UploadError("Dönem sütunu sayısal değişkenlerden farklı bir sütun olmalı; dönem sütunu tablolarda ve "
                            "grafiklerde zaten kullanılır.")
    if case.has(BIRIM) and not case.has(DONEM):
        raise K.UploadError("Birim kimliği dönem sütunuyla birlikte (panel veri) kullanılır; dönem sütununu da seçin "
                            "ya da birim kimliğini kaldırın.")
    if case.has(BIRIM) and case.data.duplicated(subset=[case.roles[BIRIM], case.roles[DONEM]]).any():
        raise K.UploadError("Aynı birim aynı dönemde birden çok kez görünüyor; panelde her birim–dönem birleşimi tek "
                            "satır olmalı.")
    for column in _numeric(case):
        if case.data[column].dropna().nunique() < 2:
            raise K.UploadError(f"“{case.name(column)}” sütununda en az iki farklı değer olmalı.")
    if case.has(DONEM) and case.data[case.roles[DONEM]].nunique() < 2:
        raise K.UploadError(f"“{case.label(DONEM)}” sütununda en az iki farklı dönem olmalı.")
    if case.has(ATAMA):
        assigned = case.data[case.roles[ATAMA]]
        if assigned.value_counts().min() < 2:
            raise K.UploadError("Atama sütununun iki grubunda da en az iki gözlem olmalı.")


_ASSIGNMENT_WORDS = frozenset(("grup", "group", "atama", "program", "deney", "tedavi", "treatment", "treat", "kura"))
_ASSIGNMENT_LEVELS = frozenset(("program", "kontrol", "tedavi", "deney", "control", "treatment", "0", "1"))
_OUTCOME_WORDS = frozenset(("kazanç", "kazanc", "gelir", "sonuç", "sonuc", "outcome", "puan", "skor", "ücret", "ucret",
                            "wage", "earnings"))


def suggest(table) -> dict[str, str]:
    """Dosyanın yapısından rol önerileri: adı dönem gibi olan sütun (Yıl, Dönem, …), onunla birlikte her satırı tek
    tanımlayan birim sütunu (panel), iki kategorili bir atama sütunu ve sonuç. Öğrenci önerileri değiştirebilir."""

    frame = table.frame
    numeric = [column for column in table.columns
               if pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column])]
    found: dict[str, str] = {}
    time = next((column for column in numeric if K.time_like(column) and frame[column].notna().all()
                 and K.usable(table, column, "sayisal")), None)
    if time is not None:
        found[DONEM] = time
        if not frame[time].is_unique:
            for column in table.columns:
                series = frame[column]
                if column == time or series.isna().any() or series.nunique() < 2 or series.nunique() == len(series):
                    continue
                if column in numeric and not (series == series.round()).all():
                    continue
                if not frame.duplicated(subset=[column, time]).any() and K.usable(table, column, "serbest"):
                    found[BIRIM] = column
                    break
    taken = set(found.values())
    for column in table.columns:
        if column in taken:
            continue
        levels = {K.fold(K.clean_text(value)) for value in frame[column].dropna()}
        named = bool(set(K.name_words(column)) & _ASSIGNMENT_WORDS)
        if len(levels) == 2 and (named or levels <= _ASSIGNMENT_LEVELS) and K.usable(table, column, "kategorik") \
                and frame[column].notna().all():
            outcomes = [other for other in numeric if other not in taken and other != column
                        and not K.id_like(table, other) and K.usable(table, other, "sayisal")]
            preferred = [other for other in outcomes if set(K.name_words(other)) & _OUTCOME_WORDS]
            if outcomes:
                found[ATAMA] = column
                found[DENEY_SONUC] = (preferred or outcomes)[0]
            break
    reserved = {found.get(role) for role in (DONEM, BIRIM, ATAMA)}
    measure = next((column for column in numeric if column not in reserved and not K.id_like(table, column)
                    and not K.time_like(column) and K.usable(table, column, "sayisal")), None)
    if measure is not None:
        found[SAYISAL] = measure
    return found


def _sample_panel() -> pd.DataFrame:
    """Örnek panel: sekiz kurgusal il, 2018–2024 (deterministik formül; gerçek veri değil)."""

    rows = []
    for index, name in enumerate(("İl A", "İl B", "İl C", "İl D", "İl E", "İl F", "İl G", "İl H")):
        base_unemployment = 7.0 + 1.1 * index - 0.12 * index * index
        for year in range(2018, 2025):
            t = year - 2018
            cycle = (1.8 if year == 2020 else 0.0) - 0.25 * t
            unemployment = round(base_unemployment + cycle + 0.3 * ((index + t) % 3 - 1), 1)
            growth = round(4.5 - 0.15 * index - (6.0 if year == 2020 else 0.0) + 2.5 * (year == 2021)
                           + 0.4 * ((2 * index + t) % 3 - 1), 1)
            income = round(90 + 12 * index + 9 * t + 3 * ((index * t) % 4), 1)
            rows.append((name, year, unemployment, growth, income))
    return pd.DataFrame(rows, columns=["İl", "Yıl", "İşsizlik oranı (%)", "Büyüme (%)", "Kişi başı gelir (bin TL)"])


def _sample_experiment() -> pd.DataFrame:
    frame = pd.DataFrame(list(PROGRAM_ROWS), columns=list(PROGRAM_COLUMNS))
    return pd.DataFrame({
        "Kişi": frame["kisi"],
        "Grup": np.where(frame["program"] == 1, "Program", "Kontrol"),
        "Yıllık kazanç (bin TL)": frame["kazanc"],
        "Yaş": frame["yas"],
        "Eğitim yılı": frame["egitim"],
        "Önceki yıllık kazanç (bin TL)": frame["onceki_kazanc"],
    })


def sample() -> dict[str, pd.DataFrame]:
    """Örnek dosya: iki sayfa (kurgusal panel ve kurgusal deney); öğrenci verisi değil."""

    return {"Panel": _sample_panel(), "Deney": _sample_experiment()}


ROLES = (
    Role(SAYISAL, "Sayısal değişken", "sayisal", True, (1, 2, 3, 4),
         "Tablolarda ve grafiklerde gösterilen sayısal değişken (ör. işsizlik oranı, kazanç)."),
    Role(DONEM, "Dönem (zaman) sütunu", "sayisal", False, (1, 2, 3, 4),
         "Yıl ya da dönem numarası. Her dönemde tek gözlem varsa veri zaman serisidir; birim kimliğiyle birlikte panel "
         "veridir. Her gözlemde dolu olmalı.", complete=True),
    Role(BIRIM, "Birim kimliği", "serbest", False, (2, 4),
         "Paneldeki birimin adı ya da numarası (ör. il, firma, kişi). Dönem sütunuyla birlikte seçilir; her gözlemde dolu "
         "olmalı.", complete=True),
    Role(ATAMA, "Atama (iki kategori)", "kategorik", False, (5,),
         "Rastgele atamalı deneyde grup sütunu (ör. Program/Kontrol, 1/0). Sonuç sütunuyla birlikte seçilir; her gözlemde "
         "dolu olmalı.", levels=(2, 2), pick="Programa atanan grup (kod 1)", group="deney", complete=True),
    Role(DENEY_SONUC, "Deney sonucu", "sayisal", False, (5,),
         "Gruplar arasında karşılaştırılan sonuç (ör. program sonrası kazanç). Atama sütunuyla birlikte seçilir.",
         group="deney"),
)

CUSTOM = CustomLab(
    roles=ROLES,
    build=custom_build,
    sample=sample,
    intro=(
        "Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sayısal değişken zorunludur; dönem, birim kimliği ve deney "
        "(atama ve sonuç) sütunları isteğe bağlıdır. Rolü seçilmeyen adım neye ihtiyacı olduğunu yazar. Bir panel "
        "dosyası (birim ve dönem) yüklerseniz 1–4. adımların hepsi aynı dosyayla çalışır; örnek dosyanın iki sayfası "
        "vardır: bir panel ve bir deney."
    ),
    min_rows=5,
    extra_columns=True,
    extra_use="sayisal",
    extra_label="Ek sayısal değişkenler (isteğe bağlı, en çok 6)",
    extra_help="Tablolara, grafiklere ve Adım 5'in karşılaştırma seçeneklerine eklenir.",
    max_extra=6,
    validate=validate,
    suggest=suggest,
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
