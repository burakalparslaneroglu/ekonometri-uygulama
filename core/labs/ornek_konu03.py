"""Konu 3 genel uygulaması: basit doğrusal regresyon modeli.

Notlardaki sekiz adım (``core.labs.konu03``) aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde
yazılır: düzey ortalamaları ve EKK doğrusu, koşullu ortalama, beş gözlemle adım adım EKK hesabı, basit regresyon,
eğim ve sabit terimin yorumu, sıfır–bir açıklayıcı değişken, tahmin edilen değer ile artık ve okuma kontrol listesi.

Alternatif örnek WAGE2'dir (935 erkek çalışan, 1980; ücret aylık kazançtır, ABD doları). Küçük EKK hesabının beş
çalışanı, eğitime göre sıralanmış verinin en küçük, alt çeyrek, medyan, üst çeyrek ve en büyük konumundaki
gözlemlerdir (sıralamada eşitler dosya sırasını korur). Sıfır–bir değişken adımı Konu 2'nin kurgusal iş arama
programıdır (kurayla atama; veri üretim süreci ``core.labs.kurgusal_veri`` belgesinde, gerçek etki bilinir). "Kendi
verini yükle" seçeneğinde aynı adımlar öğrencinin dosyasıyla kurulur; iki kategorili değişken seçilmezse Adım 6 neye
ihtiyacı olduğunu yazar.

Etkileşim notlardaki gibidir: grafiğin açıklayıcı değişkeni (Adım 1), koşullu ortalamanın düzeyi (Adım 2), beşinci
gözlemin sonuç değeri (Adım 3), modelin açıklayıcı değişkeni (Adım 4; Adım 5 ve 7 aynı modeli kullanır), sıfır–bir
değişken (Adım 6, alternatif örnekte) ve tahmin noktası ile gözlenen değer (Adım 7). Konu 3'te çıktıdan yalnız gözlem
sayısı ve katsayılar okunur.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from core.labs import expr as E
from core.labs import kendi_veri as K
from core.labs.kurgusal_veri import PROGRAM_EFFECT
from core.labs.ornek import EXACT_FIT_NOTE, Case, Role, TopicVariants, exact_fit, free_name, md, sayi, sayim, \
    with_app_values
from core.labs.ornek_regresyon import (
    ACIKLAYICI,
    GOSTERGE,
    MAX_LEVELS,
    ROW_RULE,
    SONUC,
    amount,
    candidates,
    capital,
    change,
    custom_lab,
    digits_for,
    discrete,
    display,
    labels_of,
    level_text,
    level_value,
    levels,
    negligible,
    number_control,
    options,
    outcome_words,
    phrase,
    plural,
    program_case,
    roles,
    sample_people,
    short_unit,
    step_words,
    validate,
    wage2_case,
)
from core.labs.spec import (
    INTERCEPT,
    OLS,
    CellTarget,
    Check,
    Choice,
    CoefTarget,
    Count,
    Derive,
    GroupStats,
    InlineData,
    LabSpec,
    LabStep,
    MapCodes,
    ModelTarget,
    ModelValue,
    NoteRef,
    NumberChoice,
    RegressionTable,
    Scalar,
    ScalarTarget,
    ScatterPlot,
    ShowFrame,
    ShowModel,
    Statistic,
    TableTarget,
    interactive_step,
)
from core.labs.wording import at_zero

TOPIC = "konu03"
TITLE = "Uygulama: Basit Doğrusal Regresyon Modeli"
SMALL = "kucuk_ornek"
X_KEY = "adim4_x"
ALT_REGRESSORS = ("educ", "exper", "tenure")
"""Basit regresyonda seçilebilen açıklayıcı değişkenler (notlardaki gibi hepsi yıl biriminde)."""
BINARY_WORDS = {
    "program": ("programa atananlar", "kontrol grubu"),
    "kadin": ("kadınlar", "erkekler"),
    "evli": ("evliler", "evli olmayanlar"),
}
"""Kurgusal programda sıfır–bir değişkenin 1 ve 0 değerini alan grubu."""
SHAPES = {
    "educ": ("Eğitim yükseldikçe ortalama aylık kazanç genel olarak artar (10 yılda küçük bir düşüşle); doğru bu artışı "
             "izler ama ortalamaların hepsinden geçmez."),
    "exper": ("İş deneyimi düzeylerinde ortalamalar belirgin bir eğilim göstermeden dalgalanır; doğru neredeyse "
              "yataydır. Bu, deneyimin kazançla ilişkisiz olduğunu göstermez: WAGE2'de deneyimi fazla olanların "
              "eğitimi ortalamada daha azdır; çoklu regresyon Konu 5'te."),
    "tenure": ("Kıdem arttıkça ortalamalar genel olarak yükselir; doğru bu eğilimi özetler ama ortalamaların hepsinden "
               "geçmez."),
}
"""WAGE2'de düzey ortalamalarının biçimi (veriden: eğitimde artan, deneyimde yatay, kıdemde artan)."""
_SMALL_NAMES = {"x_sapma": "Xᵢ − X̄", "y_sapma": "Yᵢ − Ȳ", "carpim": "çarpım", "x_kare": "(Xᵢ − X̄)²",
                "tahmin": "Ŷᵢ", "artik": "ûᵢ", "artik_kare": "ûᵢ²"}


def _check(label: str, target, decimals: int) -> Check:
    """Kontrol: beklenen değer uygulamanın hesabıyla doldurulur (``with_app_values``)."""

    return Check(label, target, 0.0, decimals)


def _scalar(name: str, label: str, decimals: int) -> Check:
    return _check(label, ScalarTarget(name), decimals)


def _name(case: Case, column: str) -> str:
    """Başlık ve kod yorumundaki ad (Markdown değil): alternatif örnekte küçük harfle, kendi verinde dosyadaki ad."""

    return case.name(column) if case.own else phrase(case, column)


def _slope(case: Case, x: str) -> tuple[float, float]:
    """Sonucun ``x`` üzerine basit regresyonunun sabiti ve eğimi (metinler için)."""

    data = case.data[[case.roles[SONUC], x]].astype(float)
    slope, intercept = np.polyfit(data[x], data[case.roles[SONUC]], 1)
    return float(intercept), float(slope)


def _data_name(case: Case) -> str:
    return str(case.extra.get("data_name", "Verileriniz"))


# --- Adım 1: düzey ortalamaları ve EKK doğrusu -------------------------------------------------------------

def _step1(case: Case) -> LabStep:
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    many = len(case.data) > 100
    control = Choice("adim1_x", "Yatay eksendeki değişken", options(case, candidates(case)), x,
                     help=f"Varsayılan: {case.name(x)}.")

    def build(choices) -> tuple:
        column = choices["adim1_x"]
        grouped = discrete(case, column)
        operations: list = [*case.load]
        if grouped:
            operations.append(GroupStats(case.frame, (column,), y, ("count", "mean"), "ortalamalar",
                                         f"{capital(_name(case, column))} düzeylerine göre gözlem sayısı ve ortalama "
                                         f"{_name(case, y)}", decimals=2))
        operations.append(ScatterPlot(
            case.frame, column, y, display(case, column), display(case, y),
            f"{_data_name(case)}: tek tek gözlemler, " + ("düzey ortalamaları " if grouped else "") + "ve EKK doğrusu",
            fit_line=True, size=6 if many else 9, opacity=0.25 if many else 0.8,
            means=f"{capital(_name(case, column))} düzeyi ortalaması" if grouped else ""))
        return tuple(operations)

    def note(state, choices) -> str:
        column = choices["adim1_x"]
        if not discrete(case, column):
            return (f"{capital(phrase(case, column))} çok sayıda farklı değer alıyor; düzey ortalamaları en çok "
                    f"{MAX_LEVELS} farklı değerde gösterilir. Doğru bütün örneklemi tek bir doğrusal ilişkiyle özetler. "
                    "Aynı değerdeki gözlemlerin dikey dağılımı, doğrunun tek tek gözlemleri açıklamadığını gösterir.")
        table = state.tables["ortalamalar"]
        count, sparse = len(table), int((table["count"] < 5).sum())
        mean = str(case.extra.get("mean_object", f"ortalama {phrase(case, y)} değerini"))
        text = (f"Turuncu noktalar aynı {phrase(case, column)} düzeyindeki {plural(case)} {mean} gösterir: koşullu "
                "ortalama E(Y | X)'in örneklemdeki karşılığı. Doğru bütün örneklemi tek bir doğrusal ilişkiyle "
                "özetler.")
        shape = dict(case.extra.get("shapes", {})).get(column)
        if shape:
            text += f" {shape}"
        text += (" Aynı düzeydeki noktaların dikey dağılımı, doğrunun tek tek gözlemleri açıklamadığını gösterir. "
                 f"Düzey sayısı: {count}; 5'ten az gözlemi olan düzey sayısı: {sparse}.")
        if sparse:
            text += " Az gözlemli düzeylerin ortalamaları tek tek gözlemlere bağlı olduğu için dalgalıdır."
        return text

    return interactive_step(
        number=1,
        title="Regresyon doğrusuna neden ihtiyaç duyarız?",
        note=NoteRef("3.1", 0, ("Şekil 3.1",)),
        explanation=str(case.extra.get("step1_text") or (
            f"Aynı {phrase(case, x)} değerindeki gözlemlerin {phrase(case, y)} değerleri farklıdır; yine de "
            f"{phrase(case, x)} yükseldikçe **ortalama** sonucun nasıl değiştiğini özetlemek isteriz. "
            + ("Grafikte soluk noktalar tek tek gözlemleri, turuncu noktalar aynı düzeydeki gözlemlerin ortalamasını, "
               "doğru ise bütün örneklemin doğrusal özetini gösterir."
               if all(discrete(case, column) for column in candidates(case)) else
               "Grafikte soluk noktalar tek tek gözlemleri, doğru bütün örneklemin doğrusal özetini gösterir; seçilen "
               f"değişkenin en çok {MAX_LEVELS} farklı değeri varsa turuncu noktalar aynı düzeydeki gözlemlerin "
               "ortalamasıdır."))),
        controls=(control,),
        build=build,
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 2: koşullu ortalama ve eğim -----------------------------------------------------------------------

def _level_pairs(case: Case) -> list[tuple[float, float]]:
    found = levels(case, case.roles[ACIKLAYICI])
    return list(zip(found[:-1], found[1:]))


def _step2(case: Case) -> LabStep:
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    title = "Koşullu ortalama ve eğim"
    note_ref = NoteRef("3.3", 0, ("Denklem 3.1", "Denklem 3.2"))
    lead = ("$\\mathbb{E}(Y \\mid X = x)$, açıklayıcı değişkeni $x$ olan gözlemlerin anakütledeki ortalama sonucudur; "
            "örneklemde bu grubun ortalamasıyla tahmin edilir. Doğrusal anakütle regresyon fonksiyonunda "
            "$\\mathbb{E}(Y \\mid X = x + 1) - \\mathbb{E}(Y \\mid X = x) = \\beta_1$ her $x$ için aynıdır.")
    if case.own and not discrete(case, x):
        return LabStep(number=2, title=title, note=note_ref,
                       explanation=(f"{lead} Bu adım için temel açıklayıcı değişkenin en çok {MAX_LEVELS} farklı değeri "
                                    f"olmalı; {phrase(case, x)} {case.data[x].nunique()} farklı değer alıyor."))
    if case.own:
        pairs = _level_pairs(case)
        counts = case.data[x].value_counts()
        start = max(pairs, key=lambda pair: (counts[pair[0]], -pair[0]))[0]
        control = Choice("adim2_x", f"Düzey x ({case.name(x)}): x ile bir sonraki düzey karşılaştırılır",
                         tuple((repr(low), level_text(low)) for low, _ in pairs), repr(start),
                         help="Varsayılan: en çok gözlemi olan düzey.")
    else:
        control = NumberChoice("adim2_x", "Eğitim düzeyi x (yıl)", 9, 17, 12, 1, integer=True,
                               help="x ile x + 1 yıl karşılaştırılır. WAGE2'de 9–18 yıl arasındaki her düzeyde en az "
                                    "10 çalışan var.")

    def bounds(choices) -> tuple[float, float]:
        if case.own:
            low = float(choices["adim2_x"])
            return low, dict(_level_pairs(case))[low]
        low = float(choices["adim2_x"])
        return low, low + 1

    def build(choices) -> tuple:
        low, high = bounds(choices)
        first, second = level_value(case, x, low), level_value(case, x, high)
        a, b = level_text(low), level_text(high)
        return (
            Statistic(case.frame, y, "count", "n_x", f"{case.name(x)} = {a} olan gözlem sayısı", where=(x, first),
                      decimals=0),
            Statistic(case.frame, y, "mean", "ort_x", f"{case.name(x)} = {a} olanların ortalaması", where=(x, first),
                      decimals=2),
            Statistic(case.frame, y, "count", "n_x1", f"{case.name(x)} = {b} olan gözlem sayısı", where=(x, second),
                      decimals=0),
            Statistic(case.frame, y, "mean", "ort_x1", f"{case.name(x)} = {b} olanların ortalaması",
                      where=(x, second), decimals=2),
            Scalar("fark", E.sub(E.ref("ort_x1"), E.ref("ort_x")), f"Koşullu ortalamalar arasındaki fark ({b} − {a})",
                   decimals=2),
        )

    def note(state, choices) -> str:
        low, high = bounds(choices)
        s = state.scalars
        _, slope = _slope(case, x)
        gap = high - low
        subject = dict(case.extra.get("subjects", {})).get(x, f"{phrase(case, x)} değeri")
        x_unit = f" {short_unit(case, x)}" if short_unit(case, x) else ""
        y_unit = f" {short_unit(case, y)}" if short_unit(case, y) else ""
        owner = str(case.extra.get("unit_genitive", "gözlemin"))
        text = (f"Örneklemde {subject} {level_text(low)}{x_unit} olan {int(s['n_x'])} {owner} ortalama "
                f"{outcome_words(case)} {sayi(s['ort_x'], 2)}{y_unit}, {level_text(high)}{x_unit} olan {int(s['n_x1'])} "
                f"{owner}ki {sayi(s['ort_x1'], 2)}{y_unit}; fark {sayi(s['fark'], 2)}{y_unit}. Bu ortalamalar "
                "anakütledeki E(Y | X = x) koşullu ortalamalarının örneklem karşılığıdır. ")
        shown = sayi(slope, digits_for(slope, 4))
        if gap == 1:
            text += ("Doğrusal anakütle regresyon fonksiyonunda komşu iki düzeyin farkı her x için aynıdır ve β₁'e eşittir "
                     "(Denklem 3.2); örneklemdeki farklar ise düzeyden düzeye değişir. Adım 4'teki EKK eğimi "
                     f"({shown}) bu farkları tek bir sayıyla özetler.")
        else:
            text += ("Doğrusal anakütle regresyon fonksiyonunda bir birimlik farkın karşılığı her x için aynıdır (β₁, "
                     f"Denklem 3.2); bu iki düzey arasındaki uzaklık {level_text(gap)} birim olduğu için anakütledeki "
                     f"fark {level_text(gap)} · β₁ olur. Örneklemdeki farklar ise düzeyden düzeye değişir. Adım 4'teki EKK "
                     f"eğimi ({shown}) bu farkları tek bir sayıyla özetler: {level_text(gap)} birimlik uzaklıkta "
                     f"{sayi(gap * slope, digits_for(gap * slope, 2))}{y_unit}.")
        return text + " Grup küçüldükçe ortalama tek tek gözlemlere daha çok bağlıdır."

    return interactive_step(
        number=2,
        title=title,
        note=note_ref,
        explanation=f"{lead} Bir düzey seçin; o düzeyin ve bir sonraki düzeyin ortalamasını karşılaştırın.",
        controls=(control,),
        build=build,
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 3: beş gözlemle adım adım EKK ---------------------------------------------------------------------

_OWN_SMALL = ("Verinizden beş gözlem alınır (açıklayıcı değişkene göre sıralanmış verinin en küçük, alt çeyrek, medyan, "
              "üst çeyrek ve en büyük konumu; sıralamada eşitler dosya sırasını korur)")


def small_rows(case: Case) -> tuple[tuple[int, float, float], ...]:
    """Küçük hesabın beş gözlemi: açıklayıcı değişkene göre sıralanmış verinin (eşitlerde dosya sırası) en küçük, alt
    çeyrek, medyan, üst çeyrek ve en büyük konumu; (gözlem numarası, x, y)."""

    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    data = case.data[[x, y]].copy()
    data["gozlem"] = np.arange(1, len(data) + 1)
    ordered = data.sort_values(x, kind="stable").reset_index(drop=True)
    n = len(ordered)
    positions = [(i * (n - 1)) // 4 for i in range(5)]
    rows = []
    for position in positions:
        row = ordered.iloc[position]
        rows.append((int(row["gozlem"]), _plain(row[x]), _plain(row[y])))
    return tuple(rows)


def _plain(value) -> float | int:
    number = float(value)
    return int(number) if number.is_integer() else number


def _digits(values) -> int:
    """Hesap tablosunun basamağı: tam sayı veride sapmalar için 2, değilse 4."""

    return 2 if all(float(value).is_integer() for value in values) else 4


def _step3(case: Case) -> LabStep:
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    rows = small_rows(case)
    observed = rows[-1][2]
    xs, ys = [row[1] for row in rows], [row[2] for row in rows]
    values = case.data[y]
    span = float(values.max() - values.min())
    low = max(0.0, float(values.min()) - span / 4) if float(values.min()) >= 0 else float(values.min()) - span / 4
    control = (case.extra["adim3"] if "adim3" in case.extra else
               number_control("adim3_y5", f"Beşinci gözlemin değeri Y₅ ({case.name(y)})", observed, low,
                              float(values.max()) + span / 4, help=f"Verideki değer: {level_text(observed)}."))
    digits = _digits((*xs, *ys))

    def fifth(choices) -> float | int:
        """Beşinci gözlemin değeri: varsayılan seçimde verideki değer (kaydırıcı onu adımına yuvarlamış olabilir)."""

        chosen = choices["adim3_y5"]
        return observed if float(chosen) == float(control.default) else chosen

    def build(choices) -> tuple:
        y5 = fifth(choices)
        data_rows = tuple((number, xv, yv) for number, xv, yv in rows[:4]) + ((rows[4][0], rows[4][1], y5),)
        same = float(y5) == float(observed)
        title = (str(case.extra.get("small_title", "Beş gözlem: açıklayıcı değişkene göre sıralanmış verinin en küçük, "
                                                    "alt çeyrek, medyan, üst çeyrek ve en büyük konumu"))
                 + ("" if same else f"; beşinci gözlemin değeri {level_text(y5)}"))
        return (
            InlineData(SMALL, ("gozlem", x, y), data_rows, title),
            Statistic(SMALL, x, "sum", "toplam_x", "Toplam ΣXᵢ", decimals=digits),
            Statistic(SMALL, y, "sum", "toplam_y", "Toplam ΣYᵢ", decimals=digits),
            Statistic(SMALL, x, "mean", "xbar", "Ortalama X̄", decimals=digits),
            Statistic(SMALL, y, "mean", "ybar", "Ortalama Ȳ", decimals=digits),
            Derive(SMALL, "x_sapma", E.sub(E.var(x), E.ref("xbar")), "Sapma Xᵢ − X̄"),
            Derive(SMALL, "y_sapma", E.sub(E.var(y), E.ref("ybar")), "Sapma Yᵢ − Ȳ"),
            Derive(SMALL, "carpim", E.mul(E.var("x_sapma"), E.var("y_sapma")), "Çarpım (Xᵢ − X̄)(Yᵢ − Ȳ)"),
            Derive(SMALL, "x_kare", E.power(E.var("x_sapma"), 2), "Kareli sapma (Xᵢ − X̄)²"),
            Statistic(SMALL, "x_sapma", "sum", "toplam_x_sapma", "Toplam Σ(Xᵢ − X̄)", decimals=digits),
            Statistic(SMALL, "y_sapma", "sum", "toplam_y_sapma", "Toplam Σ(Yᵢ − Ȳ)", decimals=digits),
            Statistic(SMALL, "carpim", "sum", "pay", "Pay Σ(Xᵢ − X̄)(Yᵢ − Ȳ)", decimals=digits),
            Statistic(SMALL, "x_kare", "sum", "payda", "Payda Σ(Xᵢ − X̄)²", decimals=digits),
            Scalar("b1_kucuk", E.div(E.ref("pay"), E.ref("payda")), "Eğim β̂₁ = pay / payda (Denklem 3.8)", decimals=4),
            Scalar("b0_kucuk", E.sub(E.ref("ybar"), E.mul(E.ref("b1_kucuk"), E.ref("xbar"))),
                   "Sabit β̂₀ = Ȳ − β̂₁X̄ (Denklem 3.9)", decimals=4),
            Derive(SMALL, "tahmin", E.add(E.ref("b0_kucuk"), E.mul(E.ref("b1_kucuk"), E.var(x))),
                   "Tahmin edilen değer Ŷᵢ = β̂₀ + β̂₁Xᵢ"),
            Derive(SMALL, "artik", E.sub(E.var(y), E.var("tahmin")), "Artık ûᵢ = Yᵢ − Ŷᵢ"),
            Derive(SMALL, "artik_kare", E.power(E.var("artik"), 2), "Kareli artık ûᵢ²"),
            ShowFrame(SMALL, ("gozlem", x, y, "x_sapma", "y_sapma", "carpim", "x_kare", "tahmin", "artik", "artik_kare"),
                      "Hesap tablosu" if same else f"Hesap tablosu (beşinci gözlemin değeri {level_text(y5)})",
                      decimals=2),
            Statistic(SMALL, "artik", "sum", "artik_toplami", "Artıkların toplamı Σûᵢ", decimals=4),
            Statistic(SMALL, "artik_kare", "sum", "artik_kare_toplami", "Kareli artıklar toplamı Σûᵢ²", decimals=2),
            OLS("model_kucuk", SMALL, y, (x,), f"Aynı doğru yazılımla: {y} ~ {x}"),
            ModelValue("b1_yazilim", "model_kucuk", "coef", "Yazılımla eğim", term=x),
            ModelValue("b0_yazilim", "model_kucuk", "coef", "Yazılımla sabit", term=INTERCEPT),
        )

    def note(state, choices) -> str:
        y5 = float(fifth(choices))
        s = state.scalars
        b0, b1 = s["b0_kucuk"], s["b1_kucuk"]
        digits = digits_for(b1, 4)
        sign = "−" if b1 < 0 and not at_zero(b1, digits) else "+"
        text = (f"Tahmin edilen doğru: ŷ = {sayi(b0, 4)} {sign} {sayi(abs(b1), digits)} · x (x: {_coded(case, x)}, "
                f"y: {_coded(case, y)}). Formüllerle bulunan eğim ve sabit, yazılımın EKK tahminiyle aynıdır. Artıkların "
                f"toplamı {sayi(s['artik_toplami'], 4)}; kareli artıklar toplamı {sayi(s['artik_kare_toplami'], 2)}. Sabit "
                f"terimli EKK doğrusunda artıkların toplamı sıfırdır ve doğru (X̄, Ȳ) = ({sayi(s['xbar'], 2)}; "
                f"{sayi(s['ybar'], 2)}) noktasından geçer.")
        if y5 == float(observed):
            _, slope = _slope(case, x)
            return text + (f" Bu beş gözlemlik örneklemde eğim {sayi(b1, 4)}; Adım 4'te bütün örneklemle "
                           f"({sayim(len(case.data))} gözlem) tahmin edilen eğim {sayi(slope, 4)}. Küçük bir örneklemin "
                           "eğimi bütün örneklemin eğiminden çok farklı olabilir: beş gözlem tek başına ilişki "
                           "hakkında az bilgi taşır.")
        x5, xbar = float(rows[-1][1]), float(np.mean(xs))
        sxx = float(np.sum((np.asarray(xs, dtype=float) - xbar) ** 2))
        pivot = xbar - sxx / (len(rows) * (x5 - xbar))
        fixed = b0 + b1 * pivot
        text += (f" Beşinci gözlemin değeri {level_text(y5)} (verideki değer {level_text(observed)}). Yalnız bir gözlem "
                 f"değişti; eğim ve sabit değişti. Doğru x = {sayi(pivot, 2)} noktası çevresinde döner: o noktadaki "
                 f"tahmin ({sayi(fixed, 2)}) değişmez, öteki tahminler ve artıklar değişir.")
        on_pivot = [number for number, (_, xv, _) in enumerate(rows, start=1) if abs(float(xv) - pivot) < 1e-9]
        if on_pivot:
            text += f" Bu noktadaki {on_pivot[0]}. gözlemin tahmini ve artığı değişmez."
        return text + (" Artıkların toplamı yine sıfırdır ve doğru yine (X̄, Ȳ) noktasından geçer. Eğimdeki değişim "
                       f"(X₅ − X̄)(Y₅ − {level_text(observed)})/Σ(Xᵢ − X̄)² kadardır; burada X₅ − X̄ = "
                       f"{sayi(x5 - xbar, 2)} ve Σ(Xᵢ − X̄)² = {sayi(sxx, 2)}: X̄'dan uzak bir gözlem eğimi daha çok "
                       "etkiler.")

    checks = [
        _scalar("toplam_x", "ΣXᵢ", digits), _scalar("toplam_y", "ΣYᵢ", digits),
        _scalar("xbar", "X̄", digits), _scalar("ybar", "Ȳ", digits),
    ]
    for column in ("x_sapma", "y_sapma", "carpim", "x_kare"):
        checks += [_check(f"{_SMALL_NAMES[column]}, {row}. gözlem", CellTarget(SMALL, column, row), digits)
                   for row in range(1, 6)]
    checks += [
        _scalar("toplam_x_sapma", "Σ(Xᵢ − X̄)", digits), _scalar("toplam_y_sapma", "Σ(Yᵢ − Ȳ)", digits),
        _scalar("pay", "Σ(Xᵢ − X̄)(Yᵢ − Ȳ)", digits), _scalar("payda", "Σ(Xᵢ − X̄)²", digits),
        _scalar("b1_kucuk", "Eğim pay / payda", 4), _scalar("b0_kucuk", "Sabit Ȳ − β̂₁X̄", 4),
    ]
    for column in ("tahmin", "artik", "artik_kare"):
        checks += [_check(f"{_SMALL_NAMES[column]}, {row}. gözlem", CellTarget(SMALL, column, row), 2)
                   for row in range(1, 6)]
    checks += [
        _scalar("artik_toplami", "Σûᵢ", 4), _scalar("artik_kare_toplami", "Σûᵢ²", 2),
        _scalar("b1_yazilim", "Eğim (yazılımla)", 4), _scalar("b0_yazilim", "Sabit (yazılımla)", 4),
    ]
    return interactive_step(
        number=3,
        title="Adım adım küçük bir EKK hesabı",
        note=NoteRef("3.9", 0, ("Tablo 3.2", "Tablo 3.3", "Denklem 3.10")),
        explanation=(
            f"{case.extra.get('small_text', _OWN_SMALL)}: X {_coded(case, x)}, Y {_coded(case, y)}. "
            "Eğim ve sabit formülleri adım adım uygulanır: $\\hat\\beta_1 = \\sum (X_i - \\bar X)(Y_i - \\bar Y) / "
            "\\sum (X_i - \\bar X)^2$ (Denklem 3.8) ve $\\hat\\beta_0 = \\bar Y - \\hat\\beta_1 \\bar X$ (Denklem 3.9). "
            "Son satırlar aynı doğrunun yazılımla tahminidir. Beşinci gözlemin Y değerini değiştirerek tek bir gözlemin "
            "doğruyu nasıl etkilediğini görün."
        ),
        controls=(control,),
        build=build,
        checks=tuple(checks),
        note_for=lambda state, choices: note(state, choices),
    )


def _coded(case: Case, column: str) -> str:
    """Sütunun koddaki adı; kendi verinde dosyadaki adıyla birlikte: “Eğitim yılı” (`egitim_yili`)."""

    return f"{phrase(case, column)} (`{column}`)" if case.own else f"`{column}`"


# --- Adım 4: basit regresyon --------------------------------------------------------------------------------

def x_choice(case: Case) -> Choice:
    return Choice(X_KEY, f"Açıklayıcı değişken (bağımlı değişken: {case.name(case.roles[SONUC])})",
                  options(case, candidates(case)), case.roles[ACIKLAYICI],
                  help=f"Varsayılan: {case.name(case.roles[ACIKLAYICI])}. Adım 5 ve 7 bu modeli kullanır.")


def _step4(case: Case, choice: Choice) -> LabStep:
    y, default = case.roles[SONUC], case.roles[ACIKLAYICI]

    def build(choices) -> tuple:
        x = choices[X_KEY]
        operations = (
            OLS("model", case.frame, y, (x,), f"Basit regresyon: {y} ~ {x}"),
            ShowModel("model", "Bu bölümde okunan alanlar (Kod 3.2'deki gibi)" if x == default
                      else "Seçtiğiniz modelin çıktısı", columns=("coef",), stats=("nobs",), stars=False),
        )
        if x == default:
            return operations
        return operations + (
            OLS("model_varsayilan", case.frame, y, (default,), f"Karşılaştırma için varsayılan model: {y} ~ {default}"),
            RegressionTable((("(1) Varsayılan", "model_varsayilan"), ("(2) Seçiminiz", "model")),
                            (default, x, INTERCEPT), "karsilastirma",
                            f"Varsayılan model ile seçtiğiniz model, bağımlı değişken: {case.name(y)}", stars=False,
                            decimals=4, standard_errors=False, r2=False),
        )

    def note(state, choices) -> str:
        x = choices[X_KEY]
        exact = EXACT_FIT_NOTE if case.own and exact_fit(case.data, y, x) else ""
        if x != default:
            return (f"Sütun (1) varsayılan model ({phrase(case, default)}), sütun (2) {phrase(case, x)} modelidir. İki "
                    "basit regresyon farklı soruları cevaplar: her eğim yalnız kendi değişkeniyle sonuç arasındaki "
                    "örneklem ilişkisini özetler. Adım 5 ve 7 seçtiğiniz modeli kullanır." + exact)
        model = state.models["model"]
        b0, b1 = float(model.params[INTERCEPT]), float(model.params[x])
        digits = digits_for(b1, 4)
        sign = "−" if b1 < 0 and not at_zero(b1, digits) else "+"
        return (f"Örneklem regresyon doğrusu ŷ = {sayi(b0, 4)} {sign} {sayi(abs(b1), digits)} · {x} (Denklem 3.11'in "
                f"karşılığı); tahmin {sayim(model.nobs)} gözleme dayanır. `Intercept` satırı sabit tahmini β̂₀, `{x}` "
                "satırı eğim tahmini β̂₁'dir. Adımın altındaki kodun yazdırdığı tam yazılım çıktısında görünen standart "
                "hata, t, p-değeri ve güven aralığı bu bölümde yorumlanmaz; R² Konu 4'te, ötekiler Konu 7'de işlenir."
                + exact)

    return interactive_step(
        number=4,
        title=str(case.extra.get("step4_title", "Python ile basit regresyon")),
        note=NoteRef("3.10", 0, ("Kod 3.1", "Kod 3.2", "Denklem 3.11")),
        explanation=(f"`{y} ~ {default}` yazımı `{y}` değişkenini bağımlı, `{default}` değişkenini açıklayıcı değişken "
                     "yapar. Kod 3.2'deki gibi bu bölümde okunan alanlar: bağımlı değişken, gözlem sayısı, sabit ve "
                     "eğim. Kodun ezberlenmesi beklenmez."),
        controls=(choice,),
        build=build,
        checks=(
            _check("Gözlem sayısı", ModelTarget("model", "nobs"), 0),
            _check("Sabit terim", CoefTarget("model", INTERCEPT), 4),
            _check("Eğim katsayısı", CoefTarget("model", default), 4),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 5: eğim ve sabit terimin yorumu --------------------------------------------------------------------

def _step5(case: Case, choice: Choice) -> LabStep:
    y = case.roles[SONUC]

    def build(choices) -> tuple:
        x = choices[X_KEY]
        return (
            ModelValue("b0", "model", "coef", "Sabit terim β̂₀", term=INTERCEPT),
            ModelValue("b1", "model", "coef", f"Eğim β̂₁ ({x})", term=x),
            Count(case.frame, "x_sifir", x, 0, f"{case.name(x)} değeri sıfır olan gözlem sayısı"),
        )

    def note(state, choices) -> str:
        x = choices[X_KEY]
        b0, b1, zeros = state.scalars["b0"], state.scalars["b1"], int(state.scalars["x_sifir"])
        if negligible(case, b1, y, x):
            slope = (f"Eğim hesap hassasiyetinde sıfırdır: örneklemde {step_words(case, x)} {plural(case)} tahmin edilen "
                     f"{outcome_words(case)} ortalama olarak aynıdır.")
        else:
            slope = (f"Eğim: örneklemde {step_words(case, x)} {plural(case)} tahmin edilen {outcome_words(case)} "
                     f"ortalama {change(b1, amount(case, y, abs(b1), digits_for(b1, 3)), False)}.")
        units = case.extra.get("units_text")
        slope += (f" Yorum sonucun örnekleme ait olduğunu, değişimin birimini ve sonucun birimini ({units}) söyler."
                  if units else " Yorum sonucun örnekleme ait olduğunu, açıklayıcı değişkendeki değişimin birimini ve "
                                "sonucun birimini söyler.")
        values = case.data[x]
        subject = dict(case.extra.get("subjects", {})).get(x, f"{phrase(case, x)} değeri")
        if zeros:
            where = f"örneklemde {subject} sıfır olan {zeros} {case.unit} var"
        else:
            where = (f"örneklemde {subject} sıfır olan {case.unit} yok (en küçük değer "
                     f"{level_text(float(values.min()))}): sıfır noktası verinin dışındadır")
        constant = (f" Sabit terim {sayi(b0, 4)}: {phrase(case, x)} sıfır olduğunda doğrunun verdiği değerdir; {where}. "
                    "Sabit, doğrunun konumunu belirler; ekonomik yorumu verinin aralığına bağlıdır.")
        if b0 < 0 <= float(case.data[y].min()):
            constant += " Negatif bir tahmin, doğrunun veri aralığının dışına taşınmasından doğar."
        if case.own:
            causal = (f" Nedensel dil sınırı: {phrase(case, x)} gözlemlere rastgele atanmadıysa bu eğim, bir gözlemin "
                      f"{phrase(case, x)} değeri bir birim daha yüksek olsaydı sonucun ne kadar farklı olacağını "
                      "söylemez.")
        else:
            one = dict(case.extra.get("one_unit", {})).get(x, "bir birim")
            causal = (f" Nedensel dil sınırı: {phrase(case, x)} {case.extra.get('unit_plural', 'gözlemlere')} rastgele "
                      f"atanmadığı için bu eğim, bir {case.unit} için {phrase(case, x)} {one} daha fazla olsaydı sonucun "
                      "kesin olarak ne kadar farklı olacağını söylemez.")
        return slope + constant + causal

    return interactive_step(
        number=5,
        title="Eğim ve sabit terimi doğru yorumlamak",
        note=NoteRef("3.11", 0),
        explanation=(
            "Doğru bir eğim yorumu dört unsuru birlikte verir: sonucun örnekleme ait olduğu, $X$'in değişim birimi, "
            "$Y$'nin değişim birimi ve nedensel olmayan dil. Sabit terim $X = 0$ noktasındaki tahmindir; anlamı sıfır "
            "değerinin veride ne kadar gözlendiğine bağlıdır. Bu adım Adım 4'te kurulan modeli yorumlar."
        ),
        uses=(choice,),
        build=build,
        checks=(
            _scalar("b1", "Eğim", 4),
            _scalar("b0", "Sabit terim", 4),
            _scalar("x_sifir", "Açıklayıcı değişkeni sıfır olan gözlem", 0),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 6: sıfır–bir açıklayıcı değişken ------------------------------------------------------------------

_BINARY_LEAD = (
    "Açıklayıcı değişken yalnız 0 ve 1 değerlerini alırsa $\\mathbb{E}(Y \\mid D = 0) = \\beta_0$ ve "
    "$\\mathbb{E}(Y \\mid D = 1) = \\beta_0 + \\beta_1$ olur: sabit $D = 0$ grubunun, sabit ile eğimin toplamı $D = 1$ "
    "grubunun ortalamasıdır; eğim iki grubun ortalama farkıdır. EKK tahmininde aynı eşitlik örneklem ortalamaları için "
    "geçerlidir."
)


def _binary_operations(frame: str, y: str, d: str, label: str, setup: tuple = (), decimals: int = 4) -> tuple:
    return (
        *setup,
        GroupStats(frame, (d,), y, ("count", "mean"), "grup_ozeti", f"{label}: gruplara göre gözlem sayısı ve ortalama",
                   decimals=decimals),
        Statistic(frame, y, "mean", "ort_0", f"Ortalama: {d} = 0", where=(d, 0)),
        Statistic(frame, y, "mean", "ort_1", f"Ortalama: {d} = 1", where=(d, 1)),
        Scalar("grup_farki", E.sub(E.ref("ort_1"), E.ref("ort_0")), "Grup ortalamaları arasındaki fark"),
        OLS("model_ikili", frame, y, (d,), f"Basit regresyon: {y} ~ {d}"),
        ShowModel("model_ikili", "Sıfır–bir açıklayıcı değişkenli modelin temel katsayıları (Kod 3.4'teki gibi)",
                  columns=("coef",), stats=("nobs",), stars=False),
        ModelValue("sabit_j", "model_ikili", "coef", "Sabit β̂₀", term=INTERCEPT),
        ModelValue("egim_j", "model_ikili", "coef", f"Eğim β̂₁ ({d})", term=d),
        Scalar("bir_grubu", E.add(E.ref("sabit_j"), E.ref("egim_j")), f"β̂₀ + β̂₁: {d} = 1 için tahmin"),
    )


_BINARY_CHECKS = (
    _check("D = 0 grubunun gözlem sayısı", TableTarget("grup_ozeti", 0, "count"), 0),
    _check("D = 1 grubunun gözlem sayısı", TableTarget("grup_ozeti", 1, "count"), 0),
    _check("D = 0 grubunun ortalaması", TableTarget("grup_ozeti", 0, "mean"), 4),
    _check("D = 1 grubunun ortalaması", TableTarget("grup_ozeti", 1, "mean"), 4),
    _check("Gözlem sayısı", ModelTarget("model_ikili", "nobs"), 0),
    _check("Sabit terim", CoefTarget("model_ikili", INTERCEPT), 4),
    _scalar("bir_grubu", "Sabit + eğim", 4),
    _scalar("grup_farki", "Eğim = grup farkı", 4),
)


def _alt_step6(binary: Case) -> LabStep:
    y = binary.roles[SONUC]

    def build(choices) -> tuple:
        d = choices["adim6_d"]
        return _binary_operations(binary.frame, y, d, binary.name(d), setup=binary.load)

    def note(state, choices) -> str:
        d = choices["adim6_d"]
        s = state.scalars
        ones, zeros = BINARY_WORDS[d]
        identity = (f"Sabit {sayi(s['sabit_j'], 4)}, `{d}` = 0 grubunun ({zeros}) ortalamasına; sabit + eğim = "
                    f"{sayi(s['bir_grubu'], 4)}, `{d}` = 1 grubunun ({ones}) ortalamasına eşittir. Eğim "
                    f"{sayi(s['egim_j'], 4)}, iki grup ortalamasının farkıdır (bin TL).")
        if d == "program":
            return identity + (
                " Program kurayla atandığı için programa atanma, kazancı etkileyen öteki faktörlerden bağımsızdır; sıfır "
                "koşullu ortalama koşulu (Denklem 3.5) tasarım gereği savunulabilir. Bu yüzden eğim (tahmin), programa "
                "atanmanın ortalama kazanç etkisinin (parametre) tahminidir (Denklem 3.12'nin karşılığı). Veri kurgusal "
                f"olduğu için parametre veri üretim sürecinden bilinir: yaklaşık {sayi(PROGRAM_EFFECT, 1)} bin TL. Bu "
                f"kuradaki tahmin ({sayi(s['egim_j'], 3)} bin TL) ondan şans eseri uzaktır: tek bir kurada gruplar "
                "başka özelliklerde de farklılaşabilir (Konu 2, Adım 5). Tahminin kuradan kuraya değişkenliği Konu 6'da "
                "(varyans), standart hatası Konu 7'de ele alınır."
            )
        return identity + (
            " Bu özellik rastgele atanmadı: iki grup eğitim, yaş ve başka özelliklerde farklı olabilir. Eğim yalnız grup "
            "ortalamalarının farkını özetler; nedensel etki olarak okunamaz. Nedensel dil yalnız kurayla atanan "
            "`program` değişkeninde kullanılır."
        )

    return interactive_step(
        number=6,
        title="Sıfır–bir açıklayıcı değişken: kurgusal iş arama programı",
        note=NoteRef("3.11", 0, ("Kod 3.3", "Kod 3.4", "Denklem 3.12")),
        explanation=(
            f"{_BINARY_LEAD} Veri Konu 2'deki kurgusal iş arama programıdır: 200 kişiden 100'ü kurayla programa atanır "
            "(`program` = 1), diğerleri kontrol grubudur. Sonuç `kazanc`, program sonrası yıllık kazançtır (bin TL; yıl "
            "boyunca işsiz kalanlarda 0). Veri bu uygulamada üretilmiştir; gerçek bir programın sonucu değildir."
        ),
        controls=(Choice("adim6_d", "Sıfır–bir açıklayıcı değişken (bağımlı değişken: program sonrası yıllık kazanç)",
                         options(binary, ("program", "kadin", "evli")), "program",
                         help="Varsayılan: `program`. Yalnız `program` kurayla atanmıştır; ötekiler kişisel "
                              "özelliklerdir."),),
        build=build,
        checks=(*_BINARY_CHECKS, _check("Eğim (program)", CoefTarget("model_ikili", "program"), 4)),
        note_for=lambda state, choices: note(state, choices),
    )


def _code_column(case: Case) -> str:
    return free_name(f"{case.roles[GOSTERGE]}_01", set(case.data.columns))


def _own_step6(case: Case) -> LabStep:
    title = "Sıfır–bir açıklayıcı değişken"
    note_ref = NoteRef("3.11", 0, ("Kod 3.3", "Kod 3.4", "Denklem 3.12"))
    if not case.has(GOSTERGE):
        return LabStep(number=6, title=title, note=note_ref,
                       explanation=(f"{_BINARY_LEAD} Bu adım için dosyanızda iki kategorili bir sütun (ör. "
                                    "program/kontrol, evet/hayır, 1/0) seçilmelidir."))
    y, column = case.roles[SONUC], case.roles[GOSTERGE]
    one = case.levels[GOSTERGE]
    zero = next(item for item in case.orders[column] if item != one)
    code = _code_column(case)
    coding = MapCodes(case.frame, column, code, ((zero, 0), (one, 1)),
                      f"Sıfır–bir değişken: {zero} = 0, {one} = 1")
    operations = _binary_operations(case.frame, y, code, case.name(column), setup=(coding,))

    def note(state) -> str:
        s = state.scalars
        return (f"Sabit {sayi(s['sabit_j'], 4)}, 0 grubunun ({md(zero)}) ortalamasına; sabit + eğim = "
                f"{sayi(s['bir_grubu'], 4)}, 1 grubunun ({md(one)}) ortalamasına eşittir. Eğim {sayi(s['egim_j'], 4)}, "
                "iki grup ortalamasının farkıdır. Gruplar rastgele atandıysa (ör. kurayla) eğim, 1 grubunda olmanın "
                "ortalama sonuca etkisinin bir tahminidir; atama rastgele değilse eğim yalnız grup farkını özetler ve "
                "nedensel etki olarak okunamaz.")

    return LabStep(
        number=6,
        title=title,
        note=note_ref,
        explanation=(f"{_BINARY_LEAD} Dosyanızda sıfır–bir değişken {phrase(case, column)} sütunundan kurulur: "
                     f"{md(one)} = 1, {md(zero)} = 0. Sonuç {phrase(case, y)}."),
        operations=operations,
        checks=(*_BINARY_CHECKS, _check("Eğim", CoefTarget("model_ikili", code), 4)),
        note_for=lambda state, choices: note(state),
    )


# --- Adım 7: gözlenen değer, tahmin edilen değer ve artık --------------------------------------------------

def _x0_control(case: Case) -> NumberChoice:
    """Kendi verinde tahmin noktası x₀: temel açıklayıcının aralığı, iki yandan aralığın dörtte biri kadar genişletilmiş
    (dışa doğru tahmin için; veri negatif değilse sıfırın altına inmez). Seçenekler aynı ölçekteyse (aralıkların
    birleşimi temel açıklayıcının aralığının en çok on katı) aralık hepsini kapsar; böylece Adım 4'te başka bir değişken
    seçildiğinde de x₀ o değişkenin aralığında seçilebilir. Bütün değerler tam sayıysa kaydırıcı tam sayıdır."""

    x = case.roles[ACIKLAYICI]
    columns = candidates(case)
    low, high = float(case.data[x].min()), float(case.data[x].max())
    union_low = min(float(case.data[column].min()) for column in columns)
    union_high = max(float(case.data[column].max()) for column in columns)
    together = union_high - union_low <= 10 * max(high - low, np.finfo(float).tiny)
    if together:
        low, high = union_low, union_high
    used = columns if together else (x,)
    pad = 0.25 * (high - low)
    start = low - pad if low < 0 else max(0.0, low - pad)
    integral = all(bool(np.all(case.data[column] == np.round(case.data[column]))) for column in used)
    label = ("Tahmin noktası x₀ (seçilen açıklayıcı değişkenin biriminde)" if together else
             f"Tahmin noktası x₀ (seçilen açıklayıcının biriminde; kaydırıcı {case.name(x)} aralığına göre)")
    return number_control("adim7_x0", label, float(case.data[x].median()), start, high + pad,
                          help="Varsayılan: temel açıklayıcının medyanı. Verideki aralığın dışı dışa doğru tahmindir.",
                          step=1.0 if integral and high + pad - start <= 2000 else None)


def _step7(case: Case, choice: Choice) -> LabStep:
    y = case.roles[SONUC]
    rounded = bool(case.extra.get("rounded_coefficients"))
    if "adim7" in case.extra:
        controls = tuple(case.extra["adim7"])
    else:
        values = case.data[y]
        controls = (_x0_control(case), number_control(
            "adim7_y0", f"Gözlenen değer y₀ ({case.name(y)})", float(values.median()), float(values.min()),
            float(values.max()), help="Varsayılan: sonucun medyanı."))
    rows = min(6, len(case.data))

    def build(choices) -> tuple:
        x, x0, y0 = choices[X_KEY], choices["adim7_x0"], choices["adim7_y0"]
        b0, b1 = (E.roundto(E.ref("b0"), 4), E.roundto(E.ref("b1"), 4)) if rounded else (E.ref("b0"), E.ref("b1"))
        return (
            Derive(case.frame, "tahmin", E.add(E.ref("b0"), E.mul(E.ref("b1"), E.var(x))),
                   "Tahmin edilen değer Ŷᵢ = β̂₀ + β̂₁Xᵢ"),
            Derive(case.frame, "artik", E.sub(E.var(y), E.var("tahmin")), "Artık ûᵢ = Yᵢ − Ŷᵢ"),
            ShowFrame(case.frame, (y, x, "tahmin", "artik"), f"İlk {rows} gözlem: tahmin edilen değer ve artık",
                      head=rows, decimals=2),
            Scalar("tahmin_x0", E.add(b0, E.mul(b1, x0)),
                   f"x₀ = {level_text(x0)} için tahmin" + (" (dört basamaklı katsayılarla)" if rounded else ""),
                   decimals=3),
            Scalar("artik_y0", E.sub(y0, E.ref("tahmin_x0")), f"y₀ = {level_text(y0)} için artık", decimals=3),
        )

    def note(state, choices) -> str:
        x, x0 = choices[X_KEY], float(choices["adim7_x0"])
        s = state.scalars
        residual = s["artik_y0"]
        if at_zero(residual, 3):
            position = "tam doğrunun üzerindedir (artık sıfır)"
        elif residual > 0:
            position = "doğrunun üzerindedir: gözlenen değer tahminden yüksektir"
        else:
            position = "doğrunun altındadır: gözlenen değer tahminden düşüktür"
        unit = short_unit(case, y)
        text = (f"Tabloda ilk {rows} gözlemin tahmin edilen değeri ve artığı görünür; artık pozitifse gözlem doğrunun "
                f"üzerindedir. Seçtiğiniz noktada tahmin {sayi(s['tahmin_x0'], 3)}{' ' + unit if unit else ''}, artık "
                f"{sayi(residual, 3)}{' ' + unit if unit else ''}; gözlem {position}. Büyük bir artık tek başına ölçüm "
                "hatası anlamına gelmez.")
        low, high = float(case.data[x].min()), float(case.data[x].max())
        if not low <= x0 <= high:
            text += (f" x₀ verideki {phrase(case, x)} aralığının ({level_text(low)}–{level_text(high)}) dışında: bu "
                     "dışa doğru bir tahmindir ve doğrusal ilişkinin orada da geçerli olduğu varsayımına dayanır.")
        return text

    explanation = (
        "Tahmin edilen değer $\\hat Y_i = \\hat\\beta_0 + \\hat\\beta_1 X_i$, artık $\\hat u_i = Y_i - \\hat Y_i$'dir. "
        f"Tablo ilk {rows} gözlem içindir (yuvarlanmamış katsayılarla, iki ondalık). "
        + str(case.extra.get("step7_text", "Altta seçtiğiniz bir nokta için tahmin ve artık hesaplanır."))
        + " Bu adım Adım 4'te kurulan modeli kullanır."
    )
    return interactive_step(
        number=7,
        title="Gözlenen değer, tahmin edilen değer ve artık",
        note=NoteRef("3.12", 0, ("Şekil 3.4", "Tablo 3.4")),
        explanation=explanation,
        uses=(choice,),
        controls=controls,
        build=build,
        checks=(
            *(_check(f"{_SMALL_NAMES[column]}, {row}. gözlem", CellTarget(case.frame, column, row), 2)
              for column in ("tahmin", "artik") for row in range(1, rows + 1)),
            _scalar("tahmin_x0", "x₀ için tahmin", 3),
            _scalar("artik_y0", "y₀ için artık", 3),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 8: kontrol listesi --------------------------------------------------------------------------------

_CHECKLIST = (
    "Bir basit regresyon denklemi, Python çıktısı veya makale tablosu gördüğünüzde:\n\n1. Araştırma sorusu nedir?\n2. "
    "Gözlem birimi ve veri yapısı nedir?\n3. $Y$ ve $X$ hangi değişkenlerdir?\n4. Değişkenlerin ölçü birimleri nedir?\n"
    "5. Anakütle modeli ile örneklem tahmini ayrılmış mı?\n6. Eğim katsayısının işareti ve büyüklüğü nedir?\n7. Sabit "
    "terimin ekonomik yorumu veri aralığında anlamlı mı?\n8. Tahmin edilen değer ve artık nasıl hesaplanır?\n9. Yorum "
    "ilişki düzeyinde mi, nedensel düzeyde mi?\n10. Henüz öğrenilmemiş çıktı alanları için hangi sonraki bölüme "
    "bakılmalıdır?\n\nBir çıktıda çok sayıda sayı bulunması, bütün sayıların aynı anda yorumlanması gerektiği anlamına "
    "gelmez: R² Konu 4'te, standart hata, t, p ve güven aralığı Konu 7'de işlenir."
)


def _step8() -> LabStep:
    return LabStep(number=8, title="Basit regresyon için okuma kontrol listesi", note=NoteRef("3.13", 0),
                   explanation=_CHECKLIST)


# --- Tanım ------------------------------------------------------------------------------------------------------

_LABELS = {
    INTERCEPT: "Sabit terim", "gozlem": "Gözlem", "x_sapma": "Xᵢ − X̄", "y_sapma": "Yᵢ − Ȳ", "carpim": "Çarpım",
    "x_kare": "(Xᵢ − X̄)²", "tahmin": "Tahmin edilen değer Ŷᵢ", "artik": "Artık ûᵢ", "artik_kare": "Kareli artık ûᵢ²",
}


def build(case: Case) -> LabSpec:
    """Konu 3 uygulamasını verilen veriyle kurar; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    choice = x_choice(case)
    labels = labels_of(case)
    binary = case.extra.get("binary")
    if binary is not None:
        labels.update(labels_of(binary))
        sixth = _alt_step6(binary)
    else:
        sixth = _own_step6(case)
        if case.has(GOSTERGE):
            labels[_code_column(case)] = f"{case.name(case.roles[GOSTERGE])} (1/0)"
    labels.update(_LABELS)
    spec = LabSpec(
        topic_key=TOPIC,
        title=str(case.extra.get("title", TITLE)),
        note_section="3",
        steps=(_step1(case), _step2(case), _step3(case), _step4(case, choice), _step5(case, choice), sixth,
               _step7(case, choice), _step8()),
        labels=tuple(labels.items()),
        source=case.source,
    )
    return with_app_values(spec)


# --- Alternatif örnek ----------------------------------------------------------------------------------------

def alternative_case() -> Case:
    return wage2_case(
        ALT_REGRESSORS,
        title="Uygulama: Basit Doğrusal Regresyon (WAGE2)",
        shapes=SHAPES,
        binary=program_case(),
        unit_plural="çalışanlara",
        unit_genitive="çalışanın",
        subjects={"educ": "eğitimi", "exper": "iş deneyimi", "tenure": "kıdemi"},
        mean_object="ortalama aylık kazancını",
        one_unit={"educ": "bir yıl", "exper": "bir yıl", "tenure": "bir yıl"},
        units_text="bir yıl; ayda dolar",
        small_text=("WAGE2'den beş çalışan alınır (eğitime göre sıralanmış verinin en küçük, alt çeyrek, medyan, üst "
                    "çeyrek ve en büyük konumu; sıralamada eşitler dosya sırasını korur)"),
        small_title=("WAGE2'den beş çalışan: eğitime göre sıralanmış verinin en küçük, alt çeyrek, medyan, üst çeyrek ve "
                     "en büyük konumu"),
        adim3=NumberChoice("adim3_y5", "Beşinci çalışanın aylık kazancı Y₅ (dolar)", 100, 3100, 1573, 1, integer=True,
                           help="Verideki değer: 1573 dolar (eğitimi 18 yıl)."),
        adim7=(
            NumberChoice("adim7_x0", "Tahmin noktası x₀ (yıl)", 0, 25, 12, 1, integer=True,
                         help="Verideki aralık: eğitim 9–18, iş deneyimi 1–23, kıdem 0–22 yıl; aralığın dışı dışa "
                              "doğru tahmindir."),
            NumberChoice("adim7_y0", "Gözlenen aylık kazanç y₀ (dolar)", 0, 3100, 900, 10, integer=True,
                         help="Varsayılan örnek: eğitimi 12 yıl, aylık kazancı 900 dolar olan bir çalışan."),
        ),
        step7_text=("Altta bir örnek: eğitimi 12 yıl, aylık kazancı 900 dolar olan bir çalışan için dört basamaklı "
                    "katsayılarla tahmin ve artık."),
        step4_title="Python ile WAGE2 basit regresyonu",
        step1_text=("Aynı eğitim düzeyindeki çalışanların aylık kazançları farklıdır; yine de eğitim yükseldikçe "
                    "**ortalama** kazancın nasıl değiştiğini özetlemek isteriz. Grafikte soluk noktalar tek tek "
                    "çalışanları, turuncu noktalar aynı düzeydeki çalışanların ortalama kazancını, doğru ise bütün "
                    "örneklemin doğrusal özetini gösterir."),
    )


@cache
def alternative() -> LabSpec:
    return build(alternative_case())


STORY = ("Alternatif örnek WAGE2'dir: 935 erkek çalışanın 1980 verisi; ücret burada aylık kazançtır (ABD doları). "
         "Sıfır–bir değişken adımı Konu 2'deki kurgusal iş arama programıdır (kurayla atama; gerçek etki veri üretim "
         "sürecinden bilinir).")


# --- Kendi verin ---------------------------------------------------------------------------------------------

def own_validate(case: Case) -> None:
    validate(case)
    if case.has(GOSTERGE) and case.data[case.roles[GOSTERGE]].value_counts().min() < 2:
        raise K.UploadError("İki kategorili değişkenin iki grubunda da en az iki gözlem olmalı.")


ROLES = roles((1, 2, 3, 4, 5, 6, 7), (
    Role(GOSTERGE, "İki kategorili değişken", "kategorik", False, (6,),
         "Tam iki kategorili sütun (ör. program/kontrol, evet/hayır, 1/0): bir kategori 1, diğeri 0 kodlanır. Her "
         "gözlemde dolu olmalı.", levels=(2, 2), pick="1 ile kodlanan grup", suggest=True, complete=True),
))

CUSTOM = custom_lab(
    build,
    sample_people,
    ("Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sonuç ve temel açıklayıcı değişken zorunludur ve sayısal olmalıdır. "
     "Ek sayısal değişkenler Adım 1 ve 4'ün seçeneklerine eklenir; iki kategorili değişken (ör. program/kontrol) Adım "
     f"6 içindir. {ROW_RULE}"),
    ROLES,
    "Adım 1 ve 4'teki açıklayıcı değişken seçeneklerine eklenir.",
    order_roles=(GOSTERGE,),
    validate=own_validate,
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
