"""Konu 1 genel uygulaması: bir araştırma sorusundan basit regresyon çıktısına.

Notlardaki §1.6'nın altı adımı aynı numaralarla, verisi değiştirilebilir biçimde yazılır: araştırma sorusu, veriyi
tanıma, görsel inceleme, ekonometrik model ve tahmin, katsayının doğru yorumu ve sonucun sınırı. Alternatif örnek
WAGE2'dir (935 erkek çalışan, 1980; ücret burada aylık kazançtır, ABD doları). "Kendi verini yükle" seçeneğinde aynı
adımlar öğrencinin dosyasıyla kurulur. Notlardaki uygulama (``core.labs.konu01``) değişmez.

Etkileşim notlardaki gibidir: betimsel özetin değişkenleri (Adım 2), saçılım grafiğinin ve modelin açıklayıcı değişkeni
(Adım 3–4). Konu 1'de yalnız basit regresyon kurulur.
"""

from __future__ import annotations

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
    liste,
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
    Check,
    Choice,
    CoefTarget,
    CompleteCases,
    Describe,
    LabSpec,
    LabStep,
    LoadWooldridge,
    ModelTarget,
    ModelValue,
    MultiChoice,
    NoteRef,
    RegressionTable,
    Scalar,
    ScalarTarget,
    ScatterPlot,
    Shape,
    ShowFrame,
    ShowModel,
    TableTarget,
    interactive_step,
)
from core.labs.wording import at_zero, signed_difference, tr_lower

TOPIC = "konu01"
TITLE = "Uygulama: Bir Araştırma Sorusundan Basit Regresyona"
SONUC, ACIKLAYICI = "sonuc", "aciklayici"
ALT_DATA = "wage2"
ALT_DESCRIBED = ("wage", "educ", "exper", "tenure", "IQ", "hours", "age", "married", "black", "south", "urban", "sibs")
ALT_DEFAULT_DESCRIBED = ("wage", "educ", "exper", "tenure")
ALT_REGRESSORS = ("educ", "exper", "tenure", "IQ")
ALT_PHRASES = {
    "educ": ("eğitim süresi bir yıl daha uzun olan", "eğitim"),
    "exper": ("iş deneyimi bir yıl daha fazla olan", "iş deneyimi"),
    "tenure": ("mevcut işverendeki kıdemi bir yıl daha fazla olan", "kıdem"),
    "IQ": ("IQ puanı bir puan daha yüksek olan", "IQ puanı"),
}
"""Açıklayıcı değişkenin cümle içindeki adı: (… bir yıl daha fazla olan, kısa ad)."""
STATS = ("count", "mean", "std", "min", "max")
_STAT_NAMES = {"count": "gözlem sayısı", "mean": "ortalama", "std": "standart sapma", "min": "en küçük",
               "max": "en büyük"}


# --- Adlar -------------------------------------------------------------------------------------------------

def _display(case: Case, column: str) -> str:
    unit = case.units.get(column, "")
    return f"{case.name(column)} ({unit})" if unit else case.name(column)


def _phrase(case: Case, column: str) -> str:
    """Cümle içindeki ad: alternatif örnekte küçük harfle, kendi verinde tırnak içinde."""

    if case.own:
        return f"“{md(case.name(column))}”"
    return dict(case.extra.get("phrases", {})).get(column) or tr_lower(case.name(column))


def _capital(text: str) -> str:
    if not text:
        return text
    return {"i": "İ", "ı": "I"}.get(text[0], text[0].upper()) + text[1:]


def _unit(case: Case, column: str) -> str:
    return dict(case.extra.get("short_units", {})).get(column, "")


def _regressors(case: Case) -> tuple[str, ...]:
    """Açıklayıcı değişken seçenekleri. Kendi verinde yalnız sonuçla birlikte kullanılabilen sütunlar: iki değeri de
    olan en az üç gözlemde ikisi de değişmeli (sabit bir değişkenle eğim ve R² tanımsızdır)."""

    if "regressors" in case.extra:
        return tuple(case.extra["regressors"])
    y = case.roles[SONUC]
    return tuple(column for column in dict.fromkeys((case.roles[ACIKLAYICI], *case.extras))
                 if usable_pair(case.data, y, column))


def _described(case: Case) -> tuple[str, ...]:
    return tuple(case.extra.get("described") or dict.fromkeys((case.roles[SONUC], case.roles[ACIKLAYICI],
                                                               *case.extras)))


def _indicators(case: Case, columns) -> list[str]:
    """0/1 gösterge değişkenleri: bütün değerleri 0 ya da 1 olan sütunlar."""

    found = []
    for column in columns:
        values = case.data[column].dropna()
        if len(values) and set(np.unique(values)) <= {0, 1}:
            found.append(column)
    return found


def _complete(case: Case, columns: tuple[str, ...], name: str):
    if case.data[list(columns)].isna().any().any():
        names = " ve ".join(case.name(column) for column in columns)
        return name, (CompleteCases(name, case.frame, columns, f"Tam gözlemler: {names}"),)
    return case.frame, ()


def _options(case: Case, columns) -> tuple[tuple[str, str], ...]:
    return tuple((column, _display(case, column)) for column in columns)


def _x_choice(case: Case, key: str, label: str, help_text: str) -> Choice:
    return Choice(key, label, _options(case, _regressors(case)), case.roles[ACIKLAYICI], help=help_text)


# --- Adımlar ---------------------------------------------------------------------------------------------------

def _step1(case: Case) -> LabStep:
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    question = case.extra.get("question") or (
        f"İlk soru bilinçli olarak ilişki düzeyindedir: **Örneklemde {_phrase(case, x)} değeri daha yüksek "
        f"gözlemlerin {_phrase(case, y)} değeri ortalama olarak ne kadar farklıdır?** Bağımlı değişken `{y}` "
        f"({md(case.name(y))}), temel açıklayıcı değişken `{x}` ({md(case.name(x))}). Katsayı, açıklayıcı "
        "değişkendeki bir birimlik farka karşılık sonuçtaki ortalama fark olarak yorumlanacaktır.")
    return LabStep(
        number=1,
        title="Araştırma sorusu",
        note=NoteRef("1.6", 1),
        explanation=str(question),
        operations=(
            *case.load,
            Shape(case.frame, "n", "k"),
            ShowFrame(case.frame, (y, x), "Bağımlı ve temel açıklayıcı değişken: ilk beş gözlem", head=5),
        ),
        checks=(Check("Gözlem sayısı n", ScalarTarget("n"), 0.0, 0), Check("Değişken sayısı", ScalarTarget("k"), 0.0, 0)),
        takeaway=str(case.extra.get("step1_takeaway") or (
            "Her satır bir gözlemdir. Soru örneklemdeki ortalama ilişkiyi sorar; açıklayıcı değişkendeki farkın "
            "sonuçtaki farkı oluşturup oluşturmadığı daha güçlü bir sorudur ve ek varsayım gerektirir (Konu 2).")),
    )


def _step2(case: Case) -> LabStep:
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    described = _described(case)
    default = tuple(case.extra.get("default_described") or described)
    control = MultiChoice("adim2_degiskenler", "Özetlenecek değişkenler", _options(case, described), default,
                          help="Varsayılan: " + liste([case.name(column) for column in default]) + ".")

    def build(choices) -> tuple:
        return (Describe(case.frame, tuple(choices["adim2_degiskenler"]), "ozet", "Betimsel özet (Kod 1.2'deki gibi)"),)

    def note(state, choices) -> str:
        variables = tuple(choices["adim2_degiskenler"])
        table = state.tables["ozet"]
        parts = []
        counts = {int(table.loc[name, "count"]) for name in variables}
        if len(counts) == 1:
            parts.append(f"Her değişken için {sayim(counts.pop())} gözlem kullanılır.")
        else:
            parts.append("Boş hücresi olan değişkenlerde gözlem sayısı daha küçüktür: yazılım her değişkeni kendi dolu "
                         "hücreleriyle özetler.")
        for column in (y, x):
            if column in variables:
                unit = _unit(case, column)
                parts.append(f"Ortalama {_phrase(case, column)} yaklaşık {sayi(table.loc[column, 'mean'], 2)}"
                             f"{' ' + unit if unit else ''}.")
        if x in variables:
            low, high = table.loc[x, "min"], table.loc[x, "max"]
            parts.append(f"{_capital(_phrase(case, x))} {sayi(low, 2 if low % 1 else 0)} ile "
                         f"{sayi(high, 2 if high % 1 else 0)} arasında değişir; bu aralık sabit terimi yorumlarken "
                         "önem taşır.")
        indicators = _indicators(case, variables)
        if indicators:
            names = liste([_phrase(case, column) for column in indicators])
            parts.append(f"0/1 gösterge değişkenlerinde ({names}) ortalama, 1 değerini alan gözlemlerin payıdır.")
        parts.append("Ortalama ve standart sapma değişkenin kendi biriminde okunur.")
        return " ".join(parts)

    checks = tuple(
        Check(f"{case.name(column)}: {_STAT_NAMES[stat]}", TableTarget("ozet", column, stat), 0.0,
              0 if stat == "count" else 3)
        for column in default for stat in STATS
    )
    return interactive_step(
        number=2,
        title="Veriyi tanıma",
        note=NoteRef("1.6", 2, ("Kod 1.1", "Kod 1.2")),
        explanation=("Seçilen değişkenler için gözlem sayısı, ortalama, standart sapma, en küçük ve en büyük değer. "
                     "Varsayılan özet bağımlı değişkeni, temel açıklayıcı değişkeni ve diğer sayısal değişkenleri "
                     "içerir."),
        controls=(control,),
        build=build,
        checks=checks,
        note_for=lambda state, choices: note(state, choices),
    )


def _step3(case: Case) -> LabStep:
    y = case.roles[SONUC]
    choice = _x_choice(case, "adim3_x", "Yatay eksendeki değişken",
                       f"Varsayılan: {case.name(case.roles[ACIKLAYICI])}.")
    many = len(case.data) > 100

    def build(choices) -> tuple:
        x = choices["adim3_x"]
        frame, complete = _complete(case, (y, x), "grafik_veri")
        title = f"{case.extra.get('data_name', 'Veri')}: {case.name(x)} ve {tr_lower(case.name(y)) if not case.own else case.name(y)}"
        return (*complete, ScatterPlot(frame, x, y, _display(case, x), _display(case, y),
                                       f"{title} arasındaki örneklem ilişkisi", fit_line=True,
                                       size=7 if many else 11, opacity=0.35 if many else 1.0))

    def note(state, choices) -> str:
        x = choices["adim3_x"]
        text = (f"Doğru, {_phrase(case, x)} düzeylerine göre tahmin edilen ortalama {_phrase(case, y)} değerinin "
                "doğrusal özetidir. Noktalar aynı doğru üzerinde değildir: aynı düzeyde farklı sonuçlar gözlenir. Bu, "
                "ekonometrik modeldeki hata teriminin neden gerekli olduğunu gösterir. Doğrunun eğimi bir örneklem "
                "ilişkisidir; farklı açıklayıcı değişkenlerin doğruları farklı soruları cevaplar.")
        if "grafik_veri" in state.frames:
            text += f" Grafikte iki değeri de olan {sayim(len(state.frames['grafik_veri']))} gözlem var."
        return text

    return interactive_step(
        number=3,
        title="Görsel inceleme",
        note=NoteRef("1.6", 3, ("Şekil 1.2",)),
        explanation=("Her nokta bir gözlemdir. Doğru, açıklayıcı değişkenin düzeylerine göre tahmin edilen ortalama "
                     "sonucun doğrusal özetidir."),
        controls=(choice,),
        build=build,
        note_for=lambda state, choices: note(state, choices),
    )


X_KEY = "adim4_x"


def _step4(case: Case, choice: Choice) -> LabStep:
    y, default = case.roles[SONUC], case.roles[ACIKLAYICI]

    def build(choices) -> tuple:
        x = choices[X_KEY]
        frame, complete = _complete(case, (y, x), "model_veri")
        operations = (
            *complete,
            OLS("model", frame, y, (x,), f"Basit regresyon: {y} ~ {x} (Kod 1.3'teki gibi)"),
            ShowModel("model", "Yazılım çıktısının temel bölümü (Kod 1.4'teki gibi)"),
        )
        if x == default:
            return operations + (
                RegressionTable((("(1)", "model"),), (default, INTERCEPT), "makale",
                                f"Makale tipi sunum (Tablo 1.3'teki gibi), bağımlı değişken: {case.name(y)}"),
            )
        return operations + (
            OLS("model_varsayilan", case.frame, y, (default,), f"Karşılaştırma için varsayılan model: {y} ~ {default}"),
            RegressionTable((("(1) Varsayılan", "model_varsayilan"), ("(2) Seçiminiz", "model")),
                            (default, x, INTERCEPT), "makale",
                            f"Varsayılan model ile seçtiğiniz model, bağımlı değişken: {case.name(y)}"),
        )

    def note(state, choices) -> str:
        x = choices[X_KEY]
        exact = EXACT_FIT_NOTE if case.own and exact_fit(case.data, y, x) else ""
        if x == default:
            return ("Program çıktısı ile makale tablosu aynı temel sonuçları verir: yazılım çıktısındaki `coef` (ekranda "
                    "Katsayı) makale tablosundaki katsayıdır; `std err` (Standart hata) parantez içindeki standart "
                    "hatadır; `No. Observations` (Gözlem sayısı n) gözlem sayısı, `R-squared` (R²) R² satırıdır. "
                    "Yıldızlar p-değerlerinden türetilir; eşikleri tablo notunda yazılmalıdır." + exact)
        return (f"Sütun (1) varsayılan modeldir ({_phrase(case, default)}), sütun (2) {_phrase(case, x)} modelidir. İki "
                "basit regresyon farklı soruları cevaplar: her eğim yalnız kendi değişkeniyle sonuç arasındaki örneklem "
                "ilişkisini özetler. R²'si daha yüksek olan değişkenin sonucu daha çok etkilediği sonucu çıkmaz; R² "
                "nedensellik ölçüsü değildir." + exact)

    def coef(term: str, quantity: str, decimals: int, label: str) -> Check:
        return Check(label, CoefTarget("model", term, quantity), 0.0, decimals)

    return interactive_step(
        number=4,
        title="Ekonometrik model ve tahmin",
        note=NoteRef("1.6", 4, ("Kod 1.3", "Kod 1.4", "Tablo 1.3")),
        explanation=(
            f"Model $y_i = \\beta_0 + \\beta_1\\,x_i + u_i$; burada $y$ `{y}`, $x$ `{default}` değişkenidir. "
            f"`{y} ~ {default}` yazımında `{y}` sonuç, `{default}` açıklayıcı değişkendir. Kodun ezberlenmesi "
            "beklenmez; çıktının hangi bilgiyi nerede verdiğini okuyacağız."
        ) if case.own else str(case.extra["model_text"]),
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
            Check("Makale tablosu: eğim", TableTarget("makale", default, "(1)"), 0.0, 3),
            Check("Makale tablosu: eğimin standart hatası", TableTarget("makale", f"{default}_sh", "(1)"), 0.0, 3),
            Check("Makale tablosu: sabit terim", TableTarget("makale", INTERCEPT, "(1)"), 0.0, 3),
            Check("Makale tablosu: gözlem sayısı", TableTarget("makale", "n", "(1)"), 0.0, 0),
            Check("Makale tablosu: R²", TableTarget("makale", "r2", "(1)"), 0.0, 3),
        ), case.own and exact_fit(case.data, y, default), table="makale"),
        note_for=lambda state, choices: note(state, choices),
    )


def _slope_words(case: Case, x: str) -> str:
    phrases = dict(case.extra.get("slope_phrases", {}))
    if x in phrases:
        return phrases[x][0]
    return f"“{md(case.name(x))}” değeri bir birim daha yüksek olan"


def _step5(case: Case, choice: Choice) -> LabStep:
    y = case.roles[SONUC]

    def build(choices) -> tuple:
        x = choices[X_KEY]
        return (
            ModelValue("b0", "model", "coef", "Sabit terim β̂₀", term=INTERCEPT),
            ModelValue("b1", "model", "coef", f"Eğim β̂₁ ({x})", term=x),
            ModelValue("r2", "model", "r2", "R²", decimals=3),
            Scalar("r2_yuzde", E.mul(E.ref("r2"), 100), "R², yüzde olarak", decimals=1, percent=True),
        )

    def note(state, choices) -> str:
        x = choices[X_KEY]
        b0, b1, r2 = state.scalars["b0"], state.scalars["b1"], state.scalars["r2"]
        sign = "−" if b1 < 0 and not at_zero(b1, 4) else "+"
        equation = f"Tahmin edilen denklem: ŷ = {sayi(b0, 4)} {sign} {sayi(abs(b1), 4)} · {x} (y: `{y}`)."
        unit = _unit(case, y)
        digits = 2 if abs(b1) >= 0.005 else 4
        amount = f"{sayi(abs(b1), digits)} {unit}" if unit else f"{sayi(abs(b1), digits)} birim"
        subject = case.extra.get("plural_unit", "gözlemlerin")
        y_name = dict(case.extra.get("possessive", {})).get(y) or f"{_phrase(case, y)} değeri"
        if at_zero(b1, 4):
            slope = (f"Eğim dört basamakta sıfırdır: örneklemde {_slope_words(case, x)} {subject} tahmin edilen "
                     f"{y_name} ortalama olarak aynıdır.")
        else:
            slope = (f"Örneklemde {_slope_words(case, x)} {subject} tahmin edilen {y_name} ortalama olarak yaklaşık "
                     f"{amount} daha {signed_difference(b1)}.")
        fit = (f"R² = {sayi(r2, 3)}: örneklemdeki {_phrase(case, y)} değişkenliğinin yaklaşık %{sayi(100 * r2, 1)} "
               "kadarı bu doğrusal modelde açıklanır. Bu, modelin nedensel olarak doğru olduğunu kanıtlamaz.")
        values = case.data[x].dropna()
        constant = ""
        if values.min() > 0 or values.max() < 0:
            constant = (f" Sabit terim, {_phrase(case, x)} sıfırken tahmin edilen değerdir; verideki değerler "
                        f"{sayi(values.min(), 2 if values.min() % 1 else 0)} ile "
                        f"{sayi(values.max(), 2 if values.max() % 1 else 0)} arasında olduğu için bu nokta verinin "
                        "dışındadır ve ekonomik olarak yorumlanmaz.")
        return (f"{equation} {slope} Yorum sonucun örnekleme ait olduğunu, değişimin birimini ve sonucun birimini "
                f"söyler; nedensel fiil kullanmaz.{constant} {fit}")

    return interactive_step(
        number=5,
        title="Katsayının doğru yorumu",
        note=NoteRef("1.6", 5, ("Denklem 1.4",)),
        explanation=("Doğru bir yorum dört unsuru birlikte verir: sonucun örnekleme ait olduğu, değişimin birimi, "
                     "sonucun birimi ve nedensel olmayan dil. Bu adım, Adım 4'te kurulan modeli yorumlar; Adım 4'te "
                     "başka bir değişken seçildiyse aşağıdaki denklem o modelindir."),
        uses=(choice,),
        build=build,
        checks=(
            Check("Sabit terim β̂₀", ScalarTarget("b0"), 0.0, 4),
            Check("Eğim β̂₁", ScalarTarget("b1"), 0.0, 4),
            Check("R²", ScalarTarget("r2"), 0.0, 3),
            Check("R², yüzde", ScalarTarget("r2_yuzde"), 0.0, 1),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


def _step6(case: Case) -> LabStep:
    if not case.own:
        text = str(case.extra["limits_text"])
    else:
        y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
        complete = case.data[[y, x]].dropna()
        r = float(complete[x].corr(complete[y]))
        if abs(r) < 1e-9:  # yuvarlama hatası düzeyinde: eğim sıfırdır
            finding = (f"Örneklemde {_phrase(case, x)} ile {_phrase(case, y)} arasında doğrusal bir ortalama ilişki "
                       "görülmez (eğim sıfırdır); bu, değişkenlerin hiç ilişkili olmadığı anlamına gelmez")
        else:
            finding = (f"Örneklemde {_phrase(case, x)} ile {_phrase(case, y)} arasında "
                       f"{'pozitif' if r > 0 else 'negatif'} bir ortalama ilişki vardır")
        text = (
            f"Basit model {_phrase(case, y)} değerini etkileyebilecek başka etkenleri açıkça içermez. Bu etkenler hem "
            f"{_phrase(case, x)} değişkeniyle hem sonuçla ilişkiliyse eğim katsayısı {_phrase(case, x)} "
            "değişkeninin saf etkisi olarak okunamaz (atlanan değişken yanlılığı, Konu 6).\n\n"
            f"**Güvenli sonuç:** {finding}; ancak bu basit regresyon tek başına nedensel bir etkiyi belirlemez. "
            "Nedensellik ve *ceteris paribus* Konu 2'de, çoklu regresyon Konu 5'te ele alınır."
        )
    return LabStep(number=6, title="Bu sonuç neyi söylemez?", note=NoteRef("1.6", 6), explanation=text)


def build(case: Case) -> LabSpec:
    """Konu 1 uygulamasını verilen veriyle kurar; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    choice = _x_choice(case, X_KEY, f"Açıklayıcı değişken (bağımlı değişken: {case.name(case.roles[SONUC])})",
                       "Konu 1'de yalnız basit regresyon kurulur; birden çok açıklayıcı değişken Konu 5'te.")
    labels = {column: _display(case, column) for column in case.labels}
    labels.update({INTERCEPT: "Sabit terim", "n": "Gözlem sayısı", "k": "Değişken sayısı"})
    spec = LabSpec(
        topic_key=TOPIC,
        title=str(case.extra.get("title", TITLE)),
        note_section="1",
        steps=(_step1(case), _step2(case), _step3(case), _step4(case, choice), _step5(case, choice), _step6(case)),
        labels=tuple(labels.items()),
        source=case.source,
    )
    return with_app_values(spec)


# --- Alternatif örnek: WAGE2 -----------------------------------------------------------------------------

def _alternative_limits(data: pd.DataFrame) -> str:
    slope = np.polyfit(data["educ"].astype(float), data["wage"].astype(float), 1)[0]
    r = data["educ"].corr(data["IQ"])
    return (
        "Basit model iş deneyimi, kıdem, meslek, sektör, yetenek ve bölge gibi faktörleri açıkça içermez. WAGE2'de "
        "yeteneğin bir ölçüsü olarak IQ puanı vardır: eğitim süresi uzun çalışanların IQ puanı da ortalamada daha "
        f"yüksektir (korelasyon katsayısı r = {sayi(r, 2)}). Bu unsurlar hem eğitimle hem kazançla ilişkiliyse "
        f"{sayi(slope, 2)} katsayısı eğitimin saf etkisi olarak okunamaz (atlanan değişken yanlılığı, Konu 6).\n\n"
        "**Güvenli sonuç:** WAGE2 örnekleminde eğitim ile aylık kazanç arasında pozitif ve belirgin bir ortalama "
        "ilişki vardır; ancak bu basit regresyon tek başına ek bir eğitim yılının nedensel kazanç etkisini belirlemez. "
        "Nedensellik ve *ceteris paribus* Konu 2'de, çoklu regresyon Konu 5'te ele alınır."
    )


def alternative_case() -> Case:
    data = W.load(ALT_DATA)
    catalog = W.DATASETS[ALT_DATA].variables
    return Case(
        source="alternatif",
        load=(LoadWooldridge(ALT_DATA, "WAGE2 veri seti (Wooldridge, 2020): 935 erkek çalışan, 1980"),),
        frame=ALT_DATA,
        data=data,
        roles={SONUC: "wage", ACIKLAYICI: "educ"},
        labels={name: item.label for name, item in catalog.items()},
        extras=tuple(name for name in ALT_DESCRIBED if name not in ("wage", "educ")),
        units={name: item.unit for name, item in catalog.items()},
        unit="çalışan",
        extra={
            "described": ALT_DESCRIBED,
            "default_described": ALT_DEFAULT_DESCRIBED,
            "regressors": ALT_REGRESSORS,
            "data_name": "WAGE2",
            "title": "Uygulama: Eğitim ve Aylık Kazanç (WAGE2)",
            "phrases": {"wage": "aylık kazanç", "educ": "eğitim", "exper": "iş deneyimi",
                        "tenure": "mevcut işverendeki kıdem", "IQ": "IQ puanı", "hours": "haftalık çalışma saati",
                        "age": "yaş", "married": "evli", "black": "siyah", "south": "güneyde yaşıyor",
                        "urban": "büyükşehir alanında yaşıyor", "sibs": "kardeş sayısı"},
            "short_units": {"wage": "dolar", "educ": "yıl", "exper": "yıl", "tenure": "yıl", "IQ": "puan"},
            "slope_phrases": ALT_PHRASES,
            "possessive": {"wage": "aylık kazancı"},
            "plural_unit": "çalışanların",
            "question": (
                "İlk soru bilinçli olarak ilişki düzeyindedir: **Örneklemde eğitim yılı daha yüksek çalışanların aylık "
                "kazancı ortalama olarak ne kadar farklıdır?** Bağımlı değişken `wage` (aylık kazanç, ABD doları), "
                "temel açıklayıcı değişken `educ` (tamamlanan eğitim yılı). Katsayı, bir eğitim yılı farkına karşılık "
                "dolar cinsinden aylık kazanç farkı olarak yorumlanacaktır. Veri WAGE2'dir: 1980'de gözlenen 935 erkek "
                "çalışan."
            ),
            "step1_takeaway": (
                "Her satır bir çalışandır (yatay kesit). Soru örneklemdeki ortalama ilişkiyi sorar; eğitimdeki farkın "
                "kazanç farkını oluşturup oluşturmadığı daha güçlü bir sorudur ve ek varsayım gerektirir (Konu 2)."
            ),
            "model_text": (
                "Model $\\text{wage}_i = \\beta_0 + \\beta_1\\,\\text{educ}_i + u_i$ (Denklem 1.3'ün karşılığı). "
                "`wage ~ educ` yazımı `wage`'i sonuç, `educ`'u açıklayıcı değişken yapar. Kodun ezberlenmesi beklenmez; "
                "çıktının hangi bilgiyi nerede verdiğini okuyacağız."
            ),
            "limits_text": _alternative_limits(data),
        },
    )


@cache
def alternative() -> LabSpec:
    return build(alternative_case())


STORY = ("Alternatif örnek WAGE2'dir: 935 erkek çalışanın 1980 verisi; ücret burada aylık kazançtır (ABD doları). "
         "Soru notlardaki gibidir: eğitim ile kazanç arasındaki örneklem ilişkisi.")


# --- Kendi verin ---------------------------------------------------------------------------------------------

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


def sample() -> pd.DataFrame:
    """Örnek dosya: kurgusal iş arama programı verisinden çalışanlar (öğrenci verisi değil)."""

    frame = pd.DataFrame(list(PROGRAM_ROWS), columns=list(PROGRAM_COLUMNS))
    frame = frame[frame["issiz"] == 0].reset_index(drop=True)
    return pd.DataFrame({
        "Yıllık kazanç (bin TL)": frame["kazanc"],
        "Eğitim yılı": frame["egitim"],
        "Yaş": frame["yas"],
        "Kadın (1/0)": frame["kadin"],
        "Evli (1/0)": frame["evli"],
        "Önceki yıllık kazanç (bin TL)": frame["onceki_kazanc"],
    })


ROLES = (
    Role(SONUC, "Sonuç (bağımlı) değişken", "sayisal", True, (1, 2, 3, 4, 5, 6),
         "Açıklanan sayısal değişken (ör. kazanç, sınav puanı)."),
    Role(ACIKLAYICI, "Temel açıklayıcı değişken", "sayisal", True, (1, 2, 3, 4, 5, 6),
         "Araştırma sorusundaki açıklayıcı değişken (ör. eğitim yılı)."),
)

CUSTOM = CustomLab(
    roles=ROLES,
    build=build,
    sample=sample,
    intro=(
        "Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sonuç ve temel açıklayıcı değişken zorunludur ve sayısal "
        "olmalıdır; bu iki sütunda boş hücresi olan satırlar analizden çıkarılır. Ek sayısal değişkenler betimsel "
        "özete ve Adım 3–4'teki seçeneklere eklenir."
    ),
    min_rows=5,
    extra_columns=True,
    extra_use="sayisal",
    extra_label="Ek sayısal değişkenler (isteğe bağlı, en çok 8)",
    extra_help="Adım 2'nin özetine ve Adım 3–4'ün seçeneklerine eklenir. Boş hücre içerebilir; o değişkeni kullanan "
               "adımlarda yalnız değeri olan gözlemler kullanılır.",
    max_extra=8,
    validate=validate,
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
