"""Konu 1 uygulaması: eğitim ve saatlik ücret (WAGE1).

Ders notlarının uygulama bölümü §1.6'nın altı adımı: araştırma sorusu, veriyi tanıma (Kod 1.2),
görsel inceleme (Şekil 1.2), ekonometrik model ve tahmin (Kod 1.4, Tablo 1.3), katsayının doğru
yorumu (Denklem 1.4) ve sonucun sınırı. Her ``Check`` notlarda basılı bir sayıdır; değer notlardan
kopyalanmıştır, hesaplanmamıştır.

Etkileşim: betimsel özetin değişkenleri (Adım 2), saçılım grafiğinin ve modelin açıklayıcı değişkeni
(Adım 3–4; bağımlı değişken ücrettir). Konu 1'de yalnız basit regresyon kurulur; çoklu regresyon Konu 5'in
konusudur. Standart hata, t, p ve güven aralığı notlardaki gibi yazılım çıktısında görünür; yorumları Konu 7'dedir.
"""

from __future__ import annotations

from core import wooldridge_data as W
from core.labs import expr as E
from core.labs.spec import (
    INTERCEPT,
    OLS,
    Check,
    Choice,
    CoefTarget,
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
from core.labs.sezgi import plain
from core.labs.wording import signed_difference, tr_lower

DATA = "wage1"
OUTCOME = "wage"
REGRESSORS = ("educ", "exper", "tenure")
"""Basit regresyonda seçilebilen açıklayıcı değişkenler (hepsi yıl biriminde)."""
DESCRIBED = ("wage", "educ", "exper", "tenure", "female", "married", "nonwhite", "numdep", "smsa")
NOTES_DESCRIBED = ("wage", "educ", "exper", "tenure")
IN_WORDS = {
    "educ": ("eğitim süresi", "eğitim"),
    "exper": ("potansiyel deneyimi", "potansiyel deneyim"),
    "tenure": ("mevcut işverendeki kıdemi", "kıdem"),
}
"""Açıklayıcı değişkenin cümle içindeki adı: (çalışanın ... bir yıl daha fazla olan, kısa ad)."""


X_CHOICE = Choice("adim4_x", "Açıklayıcı değişken (bağımlı değişken: saatlik ücret)", W.options(DATA, REGRESSORS),
                  "educ", help="Konu 1'de yalnız basit regresyon kurulur; birden çok açıklayıcı değişken Konu 5'te.")


def _load() -> LoadWooldridge:
    return LoadWooldridge(DATA, "WAGE1 veri seti (Wooldridge, 2020): 526 çalışan")


# --- Adım 2: veriyi tanıma ------------------------------------------------------------

def _describe(choices) -> tuple:
    variables = choices["adim2_degiskenler"]
    return (Describe(DATA, variables, "ozet", "Betimsel özet (Kod 1.2)"),)


def _describe_note(state, choices) -> str:
    variables = choices["adim2_degiskenler"]
    if tuple(variables) == NOTES_DESCRIBED:
        return (
            "Her değişken için 526 gözlem kullanılır. Ortalama saatlik ücret yaklaşık 5,90 dolar, ortalama "
            "eğitim süresi yaklaşık 12,56 yıldır. Eğitim 0 ile 18 yıl arasında değişir; bu aralık sabit terimi "
            "yorumlarken önem taşır. Deneyim ve kıdem de ücretle ilişkili olabileceği için geniş dağılımları, yalnız "
            "eğitimi kullanan ilk modelin sınırlılığını düşündürür."
        )
    indicators = [name for name in variables if W.variable(DATA, name).unit == "0/1 gösterge"]
    text = ("Seçtiğiniz değişkenlerin özeti. WAGE1'de eksik değer olmadığı için her değişkenin gözlem sayısı 526'dır; "
            "ortalama ve standart sapma değişkenin kendi biriminde okunur.")
    if indicators:
        first = W.variable(DATA, indicators[0]).label
        names = ", ".join(tr_lower(W.variable(DATA, name).label) for name in indicators)
        text += (f" 0/1 gösterge değişkenlerinde ({names}) ortalama, 1 değerini alan gözlemlerin payıdır: ör. "
                 f"“{first}” değişkeninin ortalaması örneklemde bu özelliği taşıyan çalışanların oranıdır.")
    return text


# --- Adım 3: görsel inceleme ----------------------------------------------------------

def _scatter(choices) -> tuple:
    x = choices["adim3_x"]
    label = W.variable(DATA, x)
    return (
        ScatterPlot(DATA, x, OUTCOME, label.text, "Saatlik ücret (ABD doları/saat)",
                    f"WAGE1: {IN_WORDS[x][1].capitalize()} ve saatlik ücret arasındaki örneklem ilişkisi",
                    fit_line=True, size=7, opacity=0.35),
    )


def _scatter_note(state, choices) -> str:
    x = choices["adim3_x"]
    if x == "educ":
        return (
            "Noktalar aynı doğru üzerinde değildir: aynı eğitim düzeyinde farklı ücretler gözlenir. Bu, ekonometrik "
            "modeldeki hata teriminin neden gerekli olduğunu gösterir. Yukarı eğimli doğru örneklemde eğitim ile "
            "ücret arasında pozitif bir ilişkiyi; noktaların doğru çevresindeki geniş dağılımı ise ücret "
            "farklılıklarının önemli kısmının yalnız eğitimle açıklanmadığını gösterir."
        )
    words = IN_WORDS[x][1]
    return (
        f"Doğru, {words} düzeylerine göre tahmin edilen ortalama ücretin doğrusal özetidir. Noktaların doğru "
        "çevresinde dağılması, aynı değerdeki çalışanların farklı ücret aldığını gösterir. Doğrunun eğimi bir "
        "örneklem ilişkisidir; farklı açıklayıcı değişkenlerin doğruları farklı soruları cevaplar."
    )


# --- Adım 4: model ve tahmin ----------------------------------------------------------

def _model(choices) -> tuple:
    x = choices["adim4_x"]
    operations = (
        OLS("model", DATA, OUTCOME, (x,), f"Basit regresyon: {OUTCOME} ~ {x} (Kod 1.3)"),
        ShowModel("model", "Yazılım çıktısının temel bölümü (Kod 1.4)"),
    )
    if x == "educ":
        return operations + (
            RegressionTable((("(1)", "model"),), ("educ", INTERCEPT), "makale",
                            "Makale tipi sunum (Tablo 1.3), bağımlı değişken: saatlik ücret"),
        )
    return operations + (
        OLS("model_notlar", DATA, OUTCOME, ("educ",), "Karşılaştırma için notlardaki model: wage ~ educ"),
        RegressionTable((("(1) Notlar", "model_notlar"), ("(2) Seçiminiz", "model")), ("educ", x, INTERCEPT),
                        "makale", "Notlardaki model ile seçtiğiniz model, bağımlı değişken: saatlik ücret"),
    )


def _model_note(state, choices) -> str:
    x = choices["adim4_x"]
    if x == "educ":
        return (
            "Program çıktısı ile makale tablosu aynı temel sonuçları verir: yazılım çıktısındaki `coef` (ekranda "
            "Katsayı) makale tablosundaki katsayıdır; `std err` (Standart hata) parantez içindeki standart hatadır; "
            "`No. Observations` (Gözlem sayısı n) gözlem sayısı, `R-squared` (R²) R² satırıdır. Yıldızlar "
            "p-değerlerinden türetilir; eşikleri tablo notunda yazılmalıdır."
        )
    words = IN_WORDS[x][1]
    return (
        f"Sütun (1) notlardaki eğitim modeli, sütun (2) {words} modelidir. İki basit regresyon farklı soruları "
        "cevaplar: her eğim yalnız kendi değişkeniyle ücret arasındaki örneklem ilişkisini özetler. R²'si daha "
        "yüksek olan değişkenin ücreti daha çok etkilediği sonucu çıkmaz; R² nedensellik ölçüsü değildir."
    )


# --- Adım 5: katsayının doğru yorumu ----------------------------------------------------

def _interpretation(choices) -> tuple:
    x = choices["adim4_x"]
    return (
        ModelValue("b0", "model", "coef", "Sabit terim β̂₀", term=INTERCEPT),
        ModelValue("b1", "model", "coef", f"Eğim β̂₁ ({x})", term=x),
        ModelValue("r2", "model", "r2", "R²", decimals=3),
        Scalar("r2_yuzde", E.mul(E.ref("r2"), 100), "R², yüzde olarak", decimals=1, percent=True),
    )


def _interpretation_note(state, choices) -> str:
    x = choices["adim4_x"]
    b0, b1, r2 = state.scalars["b0"], state.scalars["b1"], state.scalars["r2"]
    phrase = IN_WORDS[x][0]
    equation = f"Tahmin edilen denklem: ŵage = {plain(b0, 4)} + {plain(b1, 4)} · {x}."
    slope = (f"Örneklemde {phrase} bir yıl daha yüksek olan çalışanların tahmin edilen saatlik ücreti ortalama "
             f"olarak yaklaşık {plain(abs(b1), 2)} dolar daha {signed_difference(b1)}.")
    fit = (f"R² = {plain(r2, 3)}: örneklemdeki ücret değişkenliğinin yaklaşık %{plain(100 * r2, 1)} kadarı bu "
           "doğrusal modelde açıklanır. Bu, modelin nedensel olarak doğru olduğunu kanıtlamaz.")
    constant = ""
    if x == "educ":
        constant = (" Sabit terim negatiftir: eğitim yılı sıfır olduğunda tahmin edilen ücret ekonomik olarak "
                    "anlamlı değildir; bu değer verinin merkezinden uzaktır.")
    return f"{equation} {slope} Yorum sonucun örnekleme ait olduğunu, değişimin birimini (bir yıl) ve sonucun " \
           f"birimini (dolar) söyler; nedensel fiil kullanmaz.{constant} {fit}"


def _coef(term: str, quantity: str, expected: float, decimals: int, label: str) -> Check:
    return Check(label, CoefTarget("model", term, quantity), expected, decimals)


def _described(variable: str, stat: str, expected: float, decimals: int) -> Check:
    names = {"count": "gözlem sayısı", "mean": "ortalama", "std": "standart sapma", "min": "en küçük",
             "max": "en büyük"}
    return Check(f"Kod 1.2: {variable} {names[stat]}", TableTarget("ozet", variable, stat), expected, decimals)


_DESCRIPTIVE = {
    "wage": (526, 5.896, 3.693, 0.53, 24.98),
    "educ": (526, 12.563, 2.769, 0.00, 18.00),
    "exper": (526, 17.017, 13.572, 1.00, 51.00),
    "tenure": (526, 5.105, 7.224, 0.00, 44.00),
}
"""Kod 1.2'deki betimsel çıktı: count, mean, std, min, max."""

STEPS = (
    LabStep(
        number=1,
        title="Araştırma sorusu",
        note=NoteRef("1.6", 1),
        explanation=(
            "İlk soru bilinçli olarak ilişki düzeyindedir: **Örneklemde eğitim yılı daha yüksek çalışanların saatlik "
            "ücreti ortalama olarak ne kadar farklıdır?** Bağımlı değişken `wage` (ortalama saatlik kazanç), temel "
            "açıklayıcı değişken `educ` (tamamlanan eğitim yılı). Katsayı, bir eğitim yılı farkına karşılık dolar "
            "cinsinden saatlik ücret farkı olarak yorumlanacaktır."
        ),
        operations=(
            _load(),
            Shape(DATA, "n", "k"),
            ShowFrame(DATA, (OUTCOME, "educ"), "Bağımlı ve temel açıklayıcı değişken: ilk beş çalışan", head=5),
        ),
        checks=(Check("§1.6: çalışan sayısı", ScalarTarget("n"), 526, 0),),
        takeaway=(
            "Her satır bir çalışandır (yatay kesit). Soru örneklemdeki ortalama ilişkiyi sorar; eğitimdeki farkın "
            "ücret farkını oluşturup oluşturmadığı daha güçlü bir sorudur ve ek varsayım gerektirir (Konu 2)."
        ),
    ),
    interactive_step(
        number=2,
        title="Veriyi tanıma",
        note=NoteRef("1.6", 2, ("Kod 1.1", "Kod 1.2")),
        explanation=(
            "Seçilen değişkenler için gözlem sayısı, ortalama, standart sapma, en küçük ve en büyük değer. Notlardaki "
            "çıktı ücret, eğitim, deneyim ve kıdem içindir."
        ),
        controls=(
            MultiChoice("adim2_degiskenler", "Özetlenecek değişkenler", W.options(DATA, DESCRIBED), NOTES_DESCRIBED,
                        help="Notlardaki çıktı: ücret, eğitim, deneyim, kıdem."),
        ),
        build=_describe,
        checks=tuple(
            _described(variable, stat, value, 0 if stat == "count" else (3 if stat in ("mean", "std") else 2))
            for variable, values in _DESCRIPTIVE.items()
            for stat, value in zip(("count", "mean", "std", "min", "max"), values)
        ) + (
            Check("Yorum kutusu: ortalama ücret yaklaşık", TableTarget("ozet", "wage", "mean"), 5.90, 2),
            Check("Yorum kutusu: ortalama eğitim yaklaşık", TableTarget("ozet", "educ", "mean"), 12.56, 2),
        ),
        note_for=lambda state, choices: _describe_note(state, choices),
    ),
    interactive_step(
        number=3,
        title="Görsel inceleme",
        note=NoteRef("1.6", 3, ("Şekil 1.2",)),
        explanation=(
            "Her nokta bir çalışandır. Doğru, açıklayıcı değişkenin düzeylerine göre tahmin edilen ortalama ücretin "
            "doğrusal özetidir."
        ),
        controls=(
            Choice("adim3_x", "Yatay eksendeki değişken", W.options(DATA, REGRESSORS), "educ",
                   help="Notlardaki Şekil 1.2: eğitim."),
        ),
        build=_scatter,
        note_for=lambda state, choices: _scatter_note(state, choices),
    ),
    interactive_step(
        number=4,
        title="Ekonometrik model ve tahmin",
        note=NoteRef("1.6", 4, ("Kod 1.3", "Kod 1.4", "Tablo 1.3")),
        explanation=(
            "İlk model $\\text{wage}_i = \\beta_0 + \\beta_1\\,\\text{educ}_i + u_i$ (Denklem 1.3). `wage ~ educ` "
            "yazımı `wage`'i sonuç, `educ`'u açıklayıcı değişken yapar. Kodun ezberlenmesi beklenmez; çıktının hangi "
            "bilgiyi nerede verdiğini okuyacağız."
        ),
        controls=(X_CHOICE,),
        build=_model,
        checks=(
            _coef(INTERCEPT, "coef", -0.9049, 4, "Kod 1.4: sabit terim"),
            _coef("educ", "coef", 0.5414, 4, "Kod 1.4: eğitim katsayısı"),
            _coef(INTERCEPT, "se", 0.685, 3, "Kod 1.4: sabit terimin standart hatası"),
            _coef("educ", "se", 0.053, 3, "Kod 1.4: eğitim katsayısının standart hatası"),
            _coef(INTERCEPT, "t", -1.321, 3, "Kod 1.4: sabit terimin t değeri"),
            _coef("educ", "t", 10.167, 3, "Kod 1.4: eğitim katsayısının t değeri"),
            _coef(INTERCEPT, "p", 0.187, 3, "Kod 1.4: sabit terimin p-değeri"),
            _coef("educ", "p", 0.000, 3, "Kod 1.4: eğitim katsayısının p-değeri"),
            _coef(INTERCEPT, "ci_low", -2.250, 3, "Kod 1.4: sabit terim, %95 GA alt sınır"),
            _coef(INTERCEPT, "ci_high", 0.441, 3, "Kod 1.4: sabit terim, %95 GA üst sınır"),
            _coef("educ", "ci_low", 0.437, 3, "Kod 1.4: eğitim, %95 GA alt sınır"),
            _coef("educ", "ci_high", 0.646, 3, "Kod 1.4: eğitim, %95 GA üst sınır"),
            Check("Kod 1.4: R²", ModelTarget("model", "r2"), 0.165, 3),
            Check("Kod 1.4: F istatistiği", ModelTarget("model", "f"), 103.4, 1),
            Check("Kod 1.4: gözlem sayısı", ModelTarget("model", "nobs"), 526, 0),
            Check("Tablo 1.3: eğitim katsayısı", TableTarget("makale", "educ", "(1)"), 0.541, 3),
            Check("Tablo 1.3: eğitim standart hatası", TableTarget("makale", "educ_sh", "(1)"), 0.053, 3),
            Check("Tablo 1.3: sabit terim", TableTarget("makale", INTERCEPT, "(1)"), -0.905, 3),
            Check("Tablo 1.3: sabit terimin standart hatası", TableTarget("makale", f"{INTERCEPT}_sh", "(1)"),
                  0.685, 3),
            Check("Tablo 1.3: gözlem sayısı", TableTarget("makale", "n", "(1)"), 526, 0),
            Check("Tablo 1.3: R²", TableTarget("makale", "r2", "(1)"), 0.165, 3),
        ),
        note_for=lambda state, choices: _model_note(state, choices),
    ),
    interactive_step(
        number=5,
        title="Katsayının doğru yorumu",
        note=NoteRef("1.6", 5, ("Denklem 1.4",)),
        explanation=(
            "Notlarda tahmin edilen denklem $\\widehat{\\text{wage}} = -0{,}9049 + 0{,}5414\\,\\text{educ}$ "
            "(Denklem 1.4); Adım 4'te başka bir değişken seçildiyse aşağıdaki denklem o modelindir. "
            "Doğru bir yorum dört unsuru birlikte verir: sonucun örnekleme ait olduğu, değişimin birimi, sonucun "
            "birimi ve nedensel olmayan dil. Bu adım, Adım 4'te kurulan modeli yorumlar."
        ),
        uses=(X_CHOICE,),
        build=_interpretation,
        checks=(
            Check("Denklem 1.4: sabit terim", ScalarTarget("b0"), -0.9049, 4),
            Check("Denklem 1.4: eğim", ScalarTarget("b1"), 0.5414, 4),
            Check("Mekanik yorum: yaklaşık eğim", ScalarTarget("b1"), 0.54, 2),
            Check("§1.6: R²", ScalarTarget("r2"), 0.165, 3),
            Check("§1.6: R², yüzde", ScalarTarget("r2_yuzde"), 16.5, 1),
        ),
        note_for=lambda state, choices: _interpretation_note(state, choices),
    ),
    LabStep(
        number=6,
        title="Bu sonuç neyi söylemez?",
        note=NoteRef("1.6", 6),
        explanation=(
            "Basit model deneyim, kıdem, meslek, sektör, yetenek ve bölge gibi faktörleri açıkça içermez. Daha uzun "
            "eğitim alan bireyler farklı mesleklere girebilir ya da gözlenmeyen yetenekleri farklı olabilir. Bu "
            "unsurlar hem eğitimle hem ücretle ilişkiliyse 0,5414 katsayısı eğitimin saf etkisi olarak okunamaz.\n\n"
            "**Güvenli sonuç:** WAGE1 örnekleminde eğitim ile saatlik ücret arasında pozitif ve belirgin bir ortalama "
            "ilişki vardır; ancak bu basit regresyon tek başına ek bir eğitim yılının nedensel ücret etkisini "
            "belirlemez. Nedensellik ve *ceteris paribus* Konu 2'de, çoklu regresyon Konu 5'te ele alınır."
        ),
    ),
)


KONU01_LAB = LabSpec(
    topic_key="konu01",
    title="Uygulama: Eğitim ve Saatlik Ücret",
    note_section="1",
    steps=STEPS,
    labels=(*W.labels(DATA), (INTERCEPT, "Sabit terim"), ("n", "Gözlem sayısı"), ("k", "Değişken sayısı")),
    consistency_notes=(
        "Veri notlardaki gibi WAGE1'dir; notlardaki kod, uygulama ve üretilen kod onu wooldridge paketinden okur.",
    ),
)
