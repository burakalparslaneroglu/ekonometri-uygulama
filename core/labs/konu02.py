"""Konu 2 uygulaması: ekonomik veri yapıları ve rastgele atamalı deney.

Notlarda ayrı bir laboratuvar bölümü yoktur; çözümlü örnekler bölüm sırasıyla adım olur: §2.2 WAGE1 (yatay
kesit, Tablo 2.1), §2.3 PHILLIPS (zaman serisi, Kod 2.1–2.2, Şekil 2.2), §2.4 CPS78_85 (havuzlanmış yatay
kesit, Kod 2.3–2.4), §2.5 WAGEPAN (panel, Kod 2.5), §2.11 JTRAIN2 (rastgele atamalı deney, Kod 2.6–2.7,
Şekil 2.4) ve §2.12 (gözlemsel ve deneysel bulguların dili, Tablo 2.4; işlemsiz son adım). Her ``Check``
notlarda basılı bir sayıdır; değer notlardan kopyalanmıştır, hesaplanmamıştır.

Etkileşim: satır sırası (Adım 1–2: yatay kesitte sıralama veri yapısını değiştirmez, zaman serisinde zaman
sırasını bozar), grafikteki seriler (Adım 2), özetlenen değişken ve gruplama (Adım 3), gösterilen çalışan
(Adım 4) ve gruplar arasında karşılaştırılan değişken (Adım 5: atama sonrası sonuçlar ve atama öncesi
özellikler). Konu 2'de regresyon kurulmaz: JTRAIN2 grup farkının basit regresyon eğimine eşit olduğu Konu 3'te,
farkların şansla açıklanıp açıklanamayacağı hipotez testleriyle Konu 7'de işlenir.
"""

from __future__ import annotations

from core import wooldridge_data as W
from core.labs import expr as E
from core.labs.sezgi import plain
from core.labs.wording import tr_lower
from core.labs.spec import (
    BarChart,
    CellTarget,
    Check,
    Choice,
    Derive,
    GroupStats,
    LabSpec,
    LabStep,
    LineChart,
    LoadWooldridge,
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
    interactive_step,
)

WAGE1, PHILLIPS, CPS, WAGEPAN, JTRAIN = "wage1", "phillips", "cps78_85", "wagepan", "jtrain2"
INDICATOR = "0/1 gösterge"


# --- Adım 1: yatay kesit (WAGE1) ---------------------------------------------------------

WAGE1_COLUMNS = ("wage", "educ", "exper", "tenure")
WAGE1_ROWS = (1, 2, 3, 526)
WAGE1_ORDERS = {"wage": "saatlik ücrete", "educ": "eğitime"}
"""Satır sırası seçenekleri: sıralama değişkeni ve cümle içindeki adı ("... göre sıralı")."""


def _cross_section(choices) -> tuple:
    order = choices["adim1_sira"]
    operations: list = [LoadWooldridge(WAGE1, "WAGE1 veri seti (Wooldridge, 2020): 526 çalışan")]
    if order != "dosya":
        operations.append(SortRows(WAGE1, order, f"Satırlar {WAGE1_ORDERS[order]} göre küçükten büyüğe sıralanır"))
    operations += [
        Shape(WAGE1, "n", "k"),
        ShowFrame(WAGE1, WAGE1_COLUMNS, "Tablo 2.1: ilk üç ve son gözlem" if order == "dosya"
                  else "Sıralamadan sonra 1, 2, 3 ve 526 numaralı satırlar", rows=WAGE1_ROWS),
    ]
    return tuple(operations)


def _cross_section_note(state, choices) -> str:
    order = choices["adim1_sira"]
    if order == "dosya":
        return (
            "Her satır bir çalışandır ve her çalışan veri setinde bir kez görünür: veri yatay kesittir, gözlem birimi "
            "çalışandır. Tablo 2.1'deki ⋮ satırı 4 ile 525 arasındaki gözlemlerin gösterilmediğini belirtir. Eğitim "
            "yılı yüksek çalışanların ücreti de yüksekse bu farkın tamamı hemen eğitime bağlanamaz: çalışanlar deneyim, "
            "meslek, sektör ve bölge bakımından da farklıdır."
        )
    return (
        f"Satırlar {WAGE1_ORDERS[order]} göre sıralandı: 1, 2, 3 ve 526 numaralı satırlarda artık başka çalışanlar "
        "görünür. Her satır yine bir çalışandır ve aynı çalışanlar zaman içinde tekrar gözlenmez; satırların yeri "
        "değişti, veri yapısı değişmedi. Zaman serisinde satır sırası ekonomik bilgi taşır (Adım 2)."
    )


def _wage1_cell(row: int, column: str, expected: float) -> Check:
    names = {"wage": "saatlik ücret", "educ": "eğitim", "exper": "deneyim", "tenure": "kıdem"}
    return Check(f"Tablo 2.1: {row}. gözlem, {names[column]}", CellTarget(WAGE1, column, row), expected,
                 2 if column == "wage" else 0)


_TABLE_2_1 = {1: (3.10, 11, 2, 0), 2: (3.24, 12, 22, 2), 3: (3.00, 11, 2, 0), 526: (3.50, 14, 5, 4)}
"""Tablo 2.1: gözlem numarası → saatlik ücret, eğitim, deneyim, kıdem."""


# --- Adım 2: zaman serisi (PHILLIPS) ------------------------------------------------------

PHILLIPS_COLUMNS = ("year", "unem", "inf")
SERIES_LABELS = {"inf": "Enflasyon oranı", "unem": "İşsizlik oranı"}


def _time_series(choices) -> tuple:
    order, series = choices["adim2_sira"], choices["adim2_seriler"]
    operations: list = [
        LoadWooldridge(PHILLIPS, "PHILLIPS veri seti (Wooldridge, 2020): yıllık enflasyon ve işsizlik (Kod 2.1)",
                       columns=PHILLIPS_COLUMNS),
    ]
    x, x_label = "year", "Yıl"
    if order != "year":
        operations += [
            SortRows(PHILLIPS, order, "Satırlar işsizlik oranına göre küçükten büyüğe sıralanır"),
            Derive(PHILLIPS, "sira", E.seq(E.var("year")), "Sıralamadan sonraki satır numarası 1, 2, …, n"),
        ]
        x, x_label = "sira", "Satır sırası (işsizlik oranına göre sıralı)"
    operations.append(ShowFrame(PHILLIPS, PHILLIPS_COLUMNS, "Kod 2.2: ilk altı gözlem" if order == "year"
                                else "Sıralamadan sonra ilk altı satır", head=6))
    first, *others = series
    if others:
        y_label = "Oran (%)"
        title = ("PHILLIPS veri setinde yıllık enflasyon ve işsizlik oranları" if order == "year"
                 else "PHILLIPS: satırlar işsizlik oranına göre sıralı")
    else:
        y_label = f"{SERIES_LABELS[first]} (%)"
        name = tr_lower(SERIES_LABELS[first])
        title = (f"PHILLIPS: yıllık {name}" if order == "year"
                 else f"PHILLIPS: {name}, satırlar işsizlik oranına göre sıralı")
    operations += [
        LineChart(PHILLIPS, x, first, x_label, y_label, title, markers=order != "year",
                  series=tuple((column, SERIES_LABELS[column]) for column in others),
                  legend=SERIES_LABELS[first] if others else ""),
        Statistic(PHILLIPS, "year", "min", "ilk_yil", "İlk yıl", decimals=0),
        Statistic(PHILLIPS, "year", "max", "son_yil", "Son yıl", decimals=0),
    ]
    return tuple(operations)


def _time_series_note(state, choices) -> str:
    first, last = plain(state.scalars["ilk_yil"], 0), plain(state.scalars["son_yil"], 0)
    series = choices["adim2_seriler"]
    if choices["adim2_sira"] == "year":
        if len(series) > 1:
            chart = ("Grafik iki serinin zaman içindeki hareketini bu sırayla gösterir. Serilerin bazı dönemlerde "
                     "birlikte ya da ters yönde hareket etmesi tek başına nedensellik göstermez.")
        else:
            chart = f"Grafik {tr_lower(SERIES_LABELS[series[0]])} serisinin zaman içindeki hareketini gösterir."
        return (
            f"Her satır bir yıldır ve satırlar zaman sırasındadır ({first}–{last}): 1970 yılı 1969'dan sonra ve "
            f"1971'den önce gelir. {chart}"
        )
    shown = ", ".join(plain(value, 0) for value in state.frames[PHILLIPS]["year"].head(6))
    effects = []
    if "unem" in series:
        effects.append("işsizlik serisi azalmayan, basamaklı bir çizgiye dönüşür")
    if "inf" in series:
        effects.append("enflasyonun zaman içindeki hareketi kaybolur")
    return (
        f"Satırlar işsizlik oranına göre sıralandığında ilk altı satırın yılları {shown} olur; yıllar artık ardışık "
        f"değildir. Grafiğin yatay ekseni yıl değil satır sırasıdır: {', '.join(effects)}. Zaman sırası bozulduğunda "
        "veri setinin önemli bir özelliği kaybolur; yatay kesitte (Adım 1) sıralama veri yapısını değiştirmez."
    )


_CODE_2_2 = (
    (1948, 3.8, 8.1), (1949, 5.9, -1.2), (1950, 5.3, 1.3), (1951, 3.3, 7.9), (1952, 3.0, 1.9), (1953, 2.9, 0.8),
)
"""Kod 2.2: PHILLIPS veri setinin ilk altı gözlemi (year, unem, inf)."""


def _phillips_cell(row: int, column: str, expected: float) -> Check:
    names = {"year": "yıl", "unem": "işsizlik oranı", "inf": "enflasyon oranı"}
    return Check(f"Kod 2.2: {row}. gözlem, {names[column]}", CellTarget(PHILLIPS, column, row), expected,
                 0 if column == "year" else 1)


# --- Adım 3: havuzlanmış yatay kesit (CPS78_85) ---------------------------------------------

CPS_VARIABLES = ("wage", "lwage", "educ", "exper", "age", "union", "nonwhite", "south", "married")
GROUPINGS = {"year": ("year",), "year_female": ("year", "female")}


def _pooled(choices) -> tuple:
    variable, grouping = choices["adim3_degisken"], choices["adim3_gruplama"]
    if variable == "wage" and grouping == "year":
        comment = "Kod 2.4: dönemlere göre gözlem sayısı ve ortalama saatlik ücret"
    else:
        groups = "dönemlere" if grouping == "year" else "dönem ve cinsiyete"
        comment = f"{groups.capitalize()} göre gözlem sayısı ve ortalama: {W.variable(CPS, variable).label}"
    return (
        LoadWooldridge(CPS, "CPS78_85 veri seti (Wooldridge, 2020): 1978 ve 1985 çalışan örneklemleri"),
        Derive(CPS, "wage", E.exp(E.var("lwage")), "Saatlik ücret: log ücretin üsteli, wage = exp(lwage) (Kod 2.3)"),
        GroupStats(CPS, GROUPINGS[grouping], variable, ("count", "mean"), "donem_ozeti", comment, decimals=2),
    )


def _pooled_note(state, choices) -> str:
    variable, grouping = choices["adim3_degisken"], choices["adim3_gruplama"]
    table = state.tables["donem_ozeti"]
    if variable == "wage" and grouping == "year":
        return (
            f"1978 örnekleminde {plain(table.loc[78, 'count'], 0)}, 1985 örnekleminde "
            f"{plain(table.loc[85, 'count'], 0)} çalışan vardır; iki örneklem ayrı ayrı seçilmiştir ve aynı kişiler "
            "izlenmez. Ortalama "
            f"saatlik ücret 1978'de yaklaşık {plain(table.loc[78, 'mean'], 2)}, 1985'te "
            f"{plain(table.loc[85, 'mean'], 2)} ABD dolarıdır (cari fiyatlarla). Bu fark aynı kişilerin ücret "
            "değişimi değildir ve tamamı bir politika değişikliğine bağlanamaz: fiyat düzeyi, çalışan bileşimi ve "
            "ekonomik koşullar da değişmiş olabilir."
        )
    rows = ("Her satır bir dönemin örneklemidir." if grouping == "year"
            else "Her satır bir dönemdeki bir cinsiyet grubudur.")
    text = (f"{rows} İki dönemin örneklemleri ayrı ayrı seçilmiştir ve aynı kişiler izlenmez: dönemler arasındaki "
            "fark aynı kişilerin değişimi değil, iki ayrı örneklemin farkıdır.")
    if W.variable(CPS, variable).unit == INDICATOR:
        text += " 0/1 gösterge değişkeninin ortalaması, 1 değerini alan çalışanların payıdır."
    if variable == "lwage":
        text += (" Log ücretin ortalaması, ortalama ücretin logaritmasına eşit değildir: exp(ortalama log ücret) "
                 "ortalama ücretten küçüktür.")
    if grouping == "year_female":
        text += " Satır adındaki ikinci sayı kadın göstergesidir: 0 erkek, 1 kadın."
    return text


# --- Adım 4: panel (WAGEPAN) ----------------------------------------------------------------

PANEL_COLUMNS = ("nr", "year", "lwage", "exper")
PERSONS = (13, 17, 18, 45, 110, 120, 126, 150)
"""Veri setindeki ilk sekiz çalışanın kimliği (``nr``)."""


def _panel(choices) -> tuple:
    shown = choices["adim4_gosterim"]
    if shown == "ilk10":
        frame = ShowFrame(WAGEPAN, PANEL_COLUMNS, "Kod 2.5: ilk on satır", head=10)
    else:
        frame = ShowFrame(WAGEPAN, PANEL_COLUMNS, f"nr = {shown} numaralı çalışanın bütün gözlemleri",
                          where=("nr", int(shown)))
    return (
        LoadWooldridge(WAGEPAN, "WAGEPAN veri seti (Wooldridge, 2020): 1980–1987 çalışan paneli"),
        frame,
        PanelSummary(WAGEPAN, "nr", "year", "panel", "Panel yapısı: her çalışanın kaç farklı yılda gözlendiği (Kod 2.5)"),
    )


def _panel_note(state, choices) -> str:
    table = state.tables["panel"]["deger"]
    units, rows = plain(table["birim"], 0), f"{int(table['gozlem']):,}".replace(",", ".")
    balanced = table["en_az_donem"] == table["en_cok_donem"]
    periods = plain(table["en_cok_donem"], 0)
    structure = (f"{units} çalışanın her biri {periods} yıl gözlenir ({rows} satır): her birim aynı sayıda dönemde "
                 "gözlendiği için panel dengelidir." if balanced else
                 f"{units} çalışan farklı sayıda dönemde gözlenir: panel dengesizdir.")
    if choices["adim4_gosterim"] == "ilk10":
        return (
            "İlk sekiz satır aynı çalışana (nr = 13) aittir: 1980'den 1987'ye her yıl bir satır. Dokuzuncu satırda "
            f"ikinci çalışan (nr = 17) başlar. Gözlem birimi kişi–yıl birleşimidir. {structure}"
        )
    return (
        f"Aynı çalışan her yıl yeniden gözlenir: log ücretin ve deneyimin yıldan yıla değişimi bu satırlardan okunur. "
        f"Havuzlanmış yatay kesitte (Adım 3) aynı kişi izlenmediği için bu değişim gözlenemez. {structure} Panel "
        "veri kullanmak tek başına bütün karşılaştırma sorunlarını çözmez."
    )


# --- Adım 5: rastgele atamalı deney (JTRAIN2) --------------------------------------------------

OUTCOMES = ("re78", "unem78")
PRE_TREATMENT = ("re74", "re75", "age", "educ", "black", "hisp", "married", "nodegree")
SHARES = {
    "unem78": "1978'in tamamında işsiz kalanların payı",
    "black": "siyahların payı",
    "hisp": "Hispaniklerin payı",
    "married": "evlilerin payı",
    "nodegree": "lise diploması olmayanların payı",
}
MEANS = {
    "re78": ("ortalama 1978 reel kazancı", "bin ABD doları"),
    "re74": ("ortalama 1974 reel kazancı", "bin ABD doları"),
    "re75": ("ortalama 1975 reel kazancı", "bin ABD doları"),
    "age": ("1977'deki ortalama yaş", "yıl"),
    "educ": ("ortalama öğrenim süresi", "yıl"),
}


def _experiment(choices) -> tuple:
    variable = choices["adim5_degisken"]
    item = W.variable(JTRAIN, variable)
    notes = variable == "re78"
    if notes:
        title, y_label = "JTRAIN2: atama gruplarının ortalama kazancı", "1978 ortalama reel kazancı (bin ABD doları)"
    elif item.unit == INDICATOR:
        title, y_label = f"JTRAIN2: atama gruplarında {SHARES[variable]}", f"{item.label}: pay (0–1)"
    else:
        title, y_label = f"JTRAIN2: atama gruplarında {MEANS[variable][0]}", f"Ortalama · {item.text}"
    return (
        LoadWooldridge(JTRAIN, "JTRAIN2 veri seti (Wooldridge, 2020): 445 kişi, rastgele atamalı iş eğitimi"),
        GroupStats(JTRAIN, ("train",), variable, ("count", "mean"), "grup_ozeti",
                   "Kod 2.7: grup ortalamaları" if notes else f"Gruplara göre gözlem sayısı ve ortalama: {item.label}",
                   decimals=4),
        Statistic(JTRAIN, variable, "mean", "ort_kontrol", "Kontrol grubu (train = 0)", where=("train", 0),
                  decimals=3),
        Statistic(JTRAIN, variable, "mean", "ort_egitim", "Eğitim grubu (train = 1)", where=("train", 1),
                  decimals=3),
        Scalar("fark", E.sub(E.ref("ort_egitim"), E.ref("ort_kontrol")), "Gözlenen fark (eğitim − kontrol)",
               decimals=3),
        BarChart("grup_ozeti", "mean", "Atama grubu (train: 0 = kontrol, 1 = eğitim)", y_label, title, decimals=3),
    )


def _experiment_note(state, choices) -> str:
    variable = choices["adim5_degisken"]
    control, treated, difference = (state.scalars[name] for name in ("ort_kontrol", "ort_egitim", "fark"))
    if variable == "re78":
        return (
            f"Kontrol grubunun ortalama kazancı yaklaşık {plain(control, 3)} bin ABD doları, eğitim grubununki "
            f"{plain(treated, 3)} bin ABD dolarıdır; gözlenen ortalama fark {plain(difference, 3)} bin ABD dolarıdır. "
            "Rastgele atama bilgisi doğruysa bu fark, gözlemsel olarak seçilmiş iki grubun farkından daha güçlü bir "
            "nedensel yorum temeli sunar. Yine de atamanın gerçekten uygulanıp uygulanmadığı, sonuç verisinin eksik "
            "olup olmadığı ve deney örnekleminin kimleri temsil ettiği değerlendirilmelidir."
        )
    if variable in SHARES:
        text = (f"Kontrol grubunda {SHARES[variable]}: %{plain(100 * control, 1)}; eğitim grubunda: "
                f"%{plain(100 * treated, 1)}. Fark (eğitim − kontrol): {plain(100 * difference, 1)} yüzde puan.")
    else:
        phrase, unit = MEANS[variable]
        text = (f"Kontrol grubunda {phrase}: {plain(control, 3)} {unit}; eğitim grubunda: {plain(treated, 3)} {unit}. "
                f"Fark (eğitim − kontrol): {plain(difference, 3)} {unit}.")
    if variable in OUTCOMES:
        return text + " Bu değişken de programdan sonra ölçülen bir sonuçtur."
    return text + (
        " Bu özellik atamadan önce ölçülmüştür; program onu değiştiremez. Rastgele atama grupların sistematik olarak "
        "farklı olmamasını hedefler, fakat örneklemde ortalamaların tam eşit olmasını garanti etmez. Böyle bir farkın "
        "şansla açıklanıp açıklanamayacağı hipotez testleriyle (Konu 7) değerlendirilir."
    )


# --- Adımlar --------------------------------------------------------------------------------

STEPS = (
    interactive_step(
        number=1,
        title="Yatay kesit verisi: WAGE1",
        note=NoteRef("2.2", 0, ("Tablo 2.1",)),
        explanation=(
            "Yatay kesit verisi çok sayıda birimin belirli bir zamanda gözlenmesiyle oluşur. WAGE1'de 526 çalışanın "
            "saatlik ücreti, eğitim yılı, potansiyel deneyimi, mevcut işverendeki kıdemi ve başka özellikleri vardır. "
            "Yatay kesitte satırların çoğu zaman ekonomik bir zaman sırası yoktur; satır sırasını değiştirerek "
            "tabloda hangi çalışanların göründüğüne bakabilirsiniz."
        ),
        controls=(
            Choice("adim1_sira", "Satır sırası",
                   (("dosya", "Veri dosyasındaki sıra"), ("wage", "Saatlik ücrete göre"), ("educ", "Eğitime göre")),
                   "dosya", help="Notlardaki Tablo 2.1 veri dosyasındaki sırayı kullanır."),
        ),
        build=_cross_section,
        checks=(Check("§2.2: çalışan sayısı", ScalarTarget("n"), 526, 0),) + tuple(
            _wage1_cell(row, column, value)
            for row, values in _TABLE_2_1.items() for column, value in zip(WAGE1_COLUMNS, values)
        ),
        note_for=lambda state, choices: _cross_section_note(state, choices),
    ),
    interactive_step(
        number=2,
        title="Zaman serisi verisi: PHILLIPS",
        note=NoteRef("2.3", 0, ("Kod 2.1", "Kod 2.2", "Şekil 2.2")),
        explanation=(
            "Zaman serisi bir veya daha fazla değişkenin ardışık dönemlerde gözlenmesiyle oluşur. PHILLIPS, "
            "1948–2003 dönemi için ABD'nin yıllık işsizlik (`unem`) ve enflasyon (`inf`) oranlarını yüzde olarak "
            "içerir. Her satır bir yıldır ve sıralama ekonomik bilgi taşır."
        ),
        controls=(
            MultiChoice("adim2_seriler", "Grafikteki seriler", tuple(SERIES_LABELS.items()), ("inf", "unem"),
                        help="Notlardaki Şekil 2.2 iki seriyi birlikte çizer."),
            Choice("adim2_sira", "Satır sırası",
                   (("year", "Zaman sırası (yıl)"), ("unem", "İşsizlik oranına göre sıralı")), "year",
                   help="Notlardaki veri zaman sırasındadır."),
        ),
        build=_time_series,
        checks=tuple(
            _phillips_cell(row, column, value)
            for row, values in enumerate(_CODE_2_2, start=1) for column, value in zip(PHILLIPS_COLUMNS, values)
        ) + (
            Check("§2.3: ilk yıl", ScalarTarget("ilk_yil"), 1948, 0),
            Check("§2.3: son yıl", ScalarTarget("son_yil"), 2003, 0),
        ),
        note_for=lambda state, choices: _time_series_note(state, choices),
    ),
    interactive_step(
        number=3,
        title="Havuzlanmış yatay kesit verisi: CPS78_85",
        note=NoteRef("2.4", 0, ("Kod 2.3", "Kod 2.4")),
        explanation=(
            "CPS78_85, 1978 ve 1985 yıllarına ait iki farklı çalışan örneklemini tek veri setinde birleştirir; "
            "1978'deki bir çalışan 1985'te aynı kişi olarak izlenmez. `year` = 78 değeri 1978'i, `year` = 85 değeri 1985'i gösterir. "
            "Veri setinde saatlik ücretin logaritması (`lwage`) vardır; saatlik ücret `wage` = exp(`lwage`) ile "
            "türetilir (ABD doları, cari fiyatlarla)."
        ),
        controls=(
            Choice("adim3_degisken", "Özetlenen değişken", W.options(CPS, CPS_VARIABLES), "wage",
                   help="Notlardaki Kod 2.4: saatlik ücret."),
            Choice("adim3_gruplama", "Gruplama", (("year", "Yıl"), ("year_female", "Yıl × cinsiyet")), "year",
                   help="Notlardaki Kod 2.4: yalnız yıl."),
        ),
        build=_pooled,
        checks=(
            Check("Kod 2.4: 1978 gözlem sayısı", TableTarget("donem_ozeti", 78, "count"), 550, 0),
            Check("Kod 2.4: 1985 gözlem sayısı", TableTarget("donem_ozeti", 85, "count"), 534, 0),
            Check("Kod 2.4: 1978 ortalama saatlik ücret", TableTarget("donem_ozeti", 78, "mean"), 6.06, 2),
            Check("Kod 2.4: 1985 ortalama saatlik ücret", TableTarget("donem_ozeti", 85, "mean"), 9.02, 2),
        ),
        note_for=lambda state, choices: _pooled_note(state, choices),
    ),
    interactive_step(
        number=4,
        title="Panel verisi: WAGEPAN",
        note=NoteRef("2.5", 0, ("Kod 2.5",)),
        explanation=(
            "WAGEPAN'da aynı çalışanlar 1980–1987 yılları arasında tekrar gözlenir; her satır bir kişi–yıl "
            "birleşimidir. Çalışanı `nr`, yılı `year` belirler; ikisi birlikte bir gözlemi tanımlar. Kod 2.5 ilk on "
            "satırı ve her çalışanın kaç farklı yılda gözlendiğinin özetini yazdırır."
        ),
        controls=(
            Choice("adim4_gosterim", "Gösterilen satırlar",
                   (("ilk10", "İlk on satır (Kod 2.5)"), *((str(person), f"Çalışan nr = {person}") for person in PERSONS)),
                   "ilk10", help="Notlardaki Kod 2.5 ilk on satırı yazdırır."),
        ),
        build=_panel,
        checks=(
            Check("§2.5: ilk yıl", TableTarget("panel", "ilk_donem", "deger"), 1980, 0),
            Check("§2.5: son yıl", TableTarget("panel", "son_donem", "deger"), 1987, 0),
        ),
        note_for=lambda state, choices: _panel_note(state, choices),
    ),
    interactive_step(
        number=5,
        title="Rastgele atamalı iş eğitimi programı: JTRAIN2",
        note=NoteRef("2.11", 0, ("Kod 2.6", "Kod 2.7", "Şekil 2.4")),
        explanation=(
            "JTRAIN2 deneysel veridir: `train` = 1 eğitim programına atananları, `train` = 0 kontrol grubuna "
            "atananları gösterir. Sonuç değişkeni `re78`, 1978 reel kazancıdır (bin ABD doları). Bu adımda regresyon "
            "kurulmaz; iki grubun ortalamaları okunur ve farkları hesaplanır. Karşılaştırılan değişkeni değiştirerek "
            "atama öncesi özelliklerde (1974–1975 kazancı, yaş, öğrenim süresi, …) grupların ne kadar benzer olduğuna da "
            "bakabilirsiniz."
        ),
        controls=(
            Choice("adim5_degisken", "Gruplar arasında karşılaştırılan değişken",
                   W.options(JTRAIN, OUTCOMES + PRE_TREATMENT), "re78",
                   help="Notlardaki Kod 2.7: 1978 reel kazancı. İlk iki değişken programdan sonra, diğerleri "
                        "atamadan önce ölçülür."),
        ),
        build=_experiment,
        checks=(
            Check("Kod 2.7: kontrol grubu gözlem sayısı", TableTarget("grup_ozeti", 0, "count"), 260, 0),
            Check("Kod 2.7: eğitim grubu gözlem sayısı", TableTarget("grup_ozeti", 1, "count"), 185, 0),
            Check("Kod 2.7: kontrol grubu ortalaması", TableTarget("grup_ozeti", 0, "mean"), 4.5548, 4),
            Check("Kod 2.7: eğitim grubu ortalaması", TableTarget("grup_ozeti", 1, "mean"), 6.3491, 4),
            Check("§2.11: kontrol grubu ortalaması, yaklaşık", ScalarTarget("ort_kontrol"), 4.555, 3),
            Check("§2.11: eğitim grubu ortalaması, yaklaşık", ScalarTarget("ort_egitim"), 6.349, 3),
            Check("§2.11: gözlenen ortalama fark", ScalarTarget("fark"), 1.794, 3),
        ),
        note_for=lambda state, choices: _experiment_note(state, choices),
    ),
    LabStep(
        number=6,
        title="Gözlemsel ve deneysel bulguları aynı dille yazmamak",
        note=NoteRef("2.12", 0, ("Tablo 2.4",)),
        explanation=(
            "İki bulgu yüzeyde benzer görünür: WAGE1'de eğitim yılı yüksek çalışanların ortalama ücreti daha "
            "yüksektir; JTRAIN2'de eğitim programına atanan grubun ortalama kazancı kontrol grubundan daha yüksektir. "
            "İlk veri gözlemseldir: çalışanlar eğitim yıllarına rastgele atanmadığı için güvenli ifade bir ilişki "
            "cümlesidir. İkinci örnekte program ataması rastgele yapıldığı için grup farkı daha güçlü bir nedensel "
            "yorum taşıyabilir.\n\n"
            "| Özellik | WAGE1: eğitim ve ücret | JTRAIN2: program ve kazanç |\n"
            "|---|---|---|\n"
            "| Veri üretimi | Gözlemsel | Rastgele atamalı deney |\n"
            "| Temel karşılaştırma | Farklı eğitim düzeyindeki çalışanlar | Eğitim ve kontrol grupları |\n"
            "| Başlıca kaygı | Grupların başka özelliklerde farklı olması | Atamanın uygulanması ve genellenebilirlik |\n"
            "| Güvenli başlangıç dili | \"İlişkilidir\", \"ortalama olarak farklıdır\" | Tasarım uygunsa \"program "
            "ataması ortalama sonucu değiştirmiştir\" |\n\n"
            "Bir sonraki konuda JTRAIN2'deki grup farkı basit regresyonun eğim katsayısı olarak yeniden yazılacaktır."
        ),
    ),
)


KONU02_LAB = LabSpec(
    topic_key="konu02",
    title="Uygulama: Veri Yapıları ve Rastgele Atamalı Deney",
    note_section="2",
    steps=STEPS,
    labels=(
        *W.labels(WAGE1, PHILLIPS, CPS, WAGEPAN, JTRAIN),
        # Birden çok veri setinde geçen adlar: veri setine özgü ayrıntı adım metninde ve grafiklerde yazılır.
        ("wage", "Saatlik ücret (ABD doları/saat)"),
        ("exper", "Deneyim (yıl)"),
        ("year", "Yıl"),
        ("sira", "Satır sırası"),
        ("n", "Gözlem sayısı"),
        ("k", "Değişken sayısı"),
    ),
    consistency_notes=(
        "Veriler notlardaki gibi Wooldridge veri setleridir; notlardaki kod, uygulama ve üretilen kod onları "
        "wooldridge paketinden okur (kitabın 7. baskısının verisi).",
        "PHILLIPS 1948–2003 dönemini (56 yıl) kapsar; notlar buna göre yazılıdır (önceki sürümde 1948–1996).",
        "CPS78_85: veri setinde saatlik ücret yoktur; wage = exp(lwage) türetilir ve cari fiyatlarladır; year 78 ve 85 "
        "değerlerini alır. Notlardaki Kod 2.4 bu veriyle 6,06 ve 9,02 verir (önceki sürümde 0/1 yıl kodu ve 5,65 / "
        "8,20 yazılıydı).",
        "Tablo 2.1'in son satırı 526. gözlemdir: 3,50 · 14 · 5 · 4 (önceki sürümde 525. gözlemin değerleri "
        "yazılıydı).",
    ),
)
