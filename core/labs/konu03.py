"""Konu 3 uygulaması: basit doğrusal regresyon modeli (WAGE1, küçük örnek, JTRAIN2).

Bölüm 3'ün laboratuvar bölümü yoktur; adımlar bölümün çözümlü örnekleridir ve bölüm sırasıyla alt bölümlere
bağlanır: eğitim düzeylerine göre ortalama ücret ve EKK doğrusu (§3.1, Şekil 3.1), koşullu ortalama ve eğim
(§3.3, Denklem 3.2), adım adım küçük EKK hesabı (§3.9, Tablo 3.2–3.3), WAGE1 basit regresyonu (§3.10, Kod 3.1–3.2),
eğim ve sabit terimin yorumu (§3.11), sıfır–bir açıklayıcı değişken: JTRAIN2 (§3.11, Kod 3.3–3.4), gözlenen değer,
tahmin edilen değer ve artık (§3.12, Tablo 3.4) ve okuma kontrol listesi (§3.13). Her ``Check`` notlarda basılı bir
sayıdır; değer notlardan kopyalanmıştır, hesaplanmamıştır.

Etkileşim: grafiğin açıklayıcı değişkeni (Adım 1), eğitim düzeyi (Adım 2), küçük örnekte beşinci öğrencinin notu
(Adım 3), modelin açıklayıcı değişkeni (Adım 4; Adım 5 ve 7 aynı modeli kullanır), sıfır–bir açıklayıcı değişken
(Adım 6), tahmin noktası ve gözlenen değer (Adım 7). Konu 3'te yazılım çıktısından yalnız gözlem sayısı ve
katsayılar okunur (Kod 3.2): R² Konu 4'te, standart hata, t, p ve güven aralığı Konu 7'de işlenir. Üretilen kod
notlardaki ``print(model.summary())`` açıklaması gibi yazılımın tam çıktısını yazdırır.
"""

from __future__ import annotations

from core import wooldridge_data as W
from core.labs import expr as E
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
    LoadWooldridge,
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
from core.labs.sezgi import plain
from core.labs.wording import signed_difference

DATA = "wage1"
JTRAIN = "jtrain2"
SMALL = "kucuk_ornek"
OUTCOME = "wage"
REGRESSORS = ("educ", "exper", "tenure")
"""Basit regresyonda seçilebilen açıklayıcı değişkenler (hepsi yıl biriminde)."""
IN_WORDS = {
    "educ": ("eğitim süresi", "eğitim"),
    "exper": ("potansiyel deneyimi", "potansiyel deneyim"),
    "tenure": ("mevcut işverendeki kıdemi", "kıdem"),
}
"""Açıklayıcı değişkenin cümle içindeki adı: (çalışanın ... bir yıl daha fazla olan, kısa ad)."""
BINARY = ("train", "black", "hisp", "married", "nodegree")
"""JTRAIN2'deki sıfır–bir değişkenler: yalnız ``train`` rastgele atanmıştır; ötekiler kişisel özelliklerdir."""
BINARY_WORDS = {
    "train": ("eğitim programına atananlar", "kontrol grubu"),
    "black": ("siyah olanlar", "siyah olmayanlar"),
    "hisp": ("Hispanik olanlar", "Hispanik olmayanlar"),
    "married": ("evliler", "evli olmayanlar"),
    "nodegree": ("lise diploması olmayanlar", "lise diploması olanlar"),
}
"""Sıfır–bir değişkenin 1 ve 0 değerini alan grubu."""
HOURS = (2, 4, 6, 8, 10)
SCORES = (55, 60, 65, 72, 78)
"""Tablo 3.2: beş öğrencinin haftalık çalışma saati X ve sınav notu Y."""

X_CHOICE = Choice("adim4_x", "Açıklayıcı değişken (bağımlı değişken: saatlik ücret)", W.options(DATA, REGRESSORS),
                  "educ", help="Notlardaki Kod 3.1: eğitim. Adım 5 ve 7 bu modeli kullanır.")


def _load() -> LoadWooldridge:
    return LoadWooldridge(DATA, "WAGE1 veri seti (Wooldridge, 2020): 526 çalışan")


def _scalar(name: str, expected: float, label: str, decimals: int) -> Check:
    return Check(label, ScalarTarget(name), expected, decimals)


# --- Adım 1: eğitim düzeyi ortalamaları ve EKK doğrusu ---------------------------------

_SHAPES = {
    "educ": "Eğitim yükseldikçe ortalamalar artar; doğru bu artışı izler ama ortalamaların hepsinden geçmez.",
    "exper": ("Deneyimde ortalamalar önce artar (yaklaşık 20–25 yıla kadar), sonra azalır; düz doğru bu biçimi "
              "yakalayamaz ve neredeyse yatay kalır. Bu tür eğrisel ilişkiler ilerleyen bölümlerde ele alınır."),
    "tenure": ("Kıdem arttıkça ortalamalar genel olarak yükselir; doğru bu eğilimi özetler ama ortalamaların "
               "hepsinden geçmez."),
}
"""WAGE1'de düzey ortalamalarının biçimi (veriden: eğitimde artan, deneyimde önce artan sonra azalan, kıdemde artan)."""

def _means(choices) -> tuple:
    x = choices["adim1_x"]
    words = IN_WORDS[x][1]
    return (
        _load(),
        GroupStats(DATA, (x,), OUTCOME, ("count", "mean"), "ortalamalar",
                   f"{words.capitalize()} düzeylerine göre çalışan sayısı ve ortalama ücret", decimals=2),
        ScatterPlot(DATA, x, OUTCOME, W.variable(DATA, x).text, "Saatlik ücret (ABD doları/saat)",
                    f"WAGE1: bireysel ücretler, {words} düzeyi ortalamaları ve EKK doğrusu",
                    fit_line=True, size=6, opacity=0.25, means=f"{words.capitalize()} düzeyi ortalaması"),
    )


def _means_note(state, choices) -> str:
    x = choices["adim1_x"]
    table = state.tables["ortalamalar"]
    levels, sparse = len(table), int((table["count"] < 5).sum())
    words = IN_WORDS[x][1]
    text = (
        f"Turuncu noktalar aynı {words} düzeyindeki çalışanların ortalama ücretidir: koşullu ortalama "
        f"E(ücret | {words})'in örneklemdeki karşılığı. Doğru bütün örneklemi tek bir doğrusal ilişkiyle özetler. "
        f"{_SHAPES[x]} Aynı düzeydeki noktaların dikey dağılımı, doğrunun tek tek gözlemleri açıklamadığını gösterir."
    )
    count = f" Düzey sayısı: {levels}; 5'ten az çalışanı olan düzey sayısı: {sparse}."
    if x == "educ":
        return text + count + (" Az gözlemli düzeylerde (0–7 yıl) ortalamalar "
                               "dalgalıdır; en kalabalık grup 12 yıllık eğitimdir (lise).")
    return text + count + " Az gözlemli düzeylerin ortalamaları tek tek çalışanlara bağlı olduğu için dalgalıdır."


# --- Adım 2: koşullu ortalama ve eğim ---------------------------------------------------

def _conditional(choices) -> tuple:
    x = int(choices["adim2_x"])
    return (
        Statistic(DATA, OUTCOME, "count", "n_x", f"Eğitimi {x} yıl olan çalışan sayısı", where=("educ", x),
                  decimals=0),
        Statistic(DATA, OUTCOME, "mean", "ort_x", f"Eğitimi {x} yıl olanların ortalama ücreti", where=("educ", x),
                  decimals=2),
        Statistic(DATA, OUTCOME, "count", "n_x1", f"Eğitimi {x + 1} yıl olan çalışan sayısı",
                  where=("educ", x + 1), decimals=0),
        Statistic(DATA, OUTCOME, "mean", "ort_x1", f"Eğitimi {x + 1} yıl olanların ortalama ücreti",
                  where=("educ", x + 1), decimals=2),
        Scalar("fark", E.sub(E.ref("ort_x1"), E.ref("ort_x")),
               f"Koşullu ortalamalar arasındaki fark ({x + 1} − {x} yıl)", decimals=2),
    )


def _conditional_note(state, choices) -> str:
    x = int(choices["adim2_x"])
    s = state.scalars
    difference = s["fark"]
    return (
        f"Örneklemde eğitimi {x} yıl olan {int(s['n_x'])} çalışanın ortalama ücreti {plain(s['ort_x'], 2)} dolar, "
        f"{x + 1} yıl olan {int(s['n_x1'])} çalışanınki {plain(s['ort_x1'], 2)} dolardır; fark "
        f"{plain(difference, 2)} dolar. Bu ortalamalar anakütledeki E(ücret | eğitim = x) koşullu ortalamalarının "
        "örneklem karşılığıdır. Doğrusal anakütle regresyon fonksiyonunda komşu iki düzeyin farkı her x için aynıdır "
        "ve β₁'e eşittir (Denklem 3.2); örneklemdeki farklar ise düzeyden düzeye değişir. Denklem 3.11'deki EKK eğimi "
        "(0,5414) bu farkları tek bir sayıyla özetler. Grup küçüldükçe ortalama tek tek çalışanlara daha çok bağlıdır."
    )


# --- Adım 3: küçük örnekte adım adım EKK ------------------------------------------------

_SMALL_COLUMNS = ("ogrenci", "saat", "puan", "x_sapma", "y_sapma", "carpim", "x_kare", "tahmin", "artik",
                  "artik_kare")


def _small(choices) -> tuple:
    y5 = int(choices["adim3_y5"])
    scores = (*SCORES[:4], y5)
    rows = tuple((i, x, y) for i, (x, y) in enumerate(zip(HOURS, scores), start=1))
    notes = y5 == SCORES[-1]
    data_title = ("Tablo 3.2: çalışma saati X ve sınav notu Y" if notes else
                  f"Tablo 3.2'nin verisi, beşinci öğrencinin notu {y5}")
    return (
        InlineData(SMALL, ("ogrenci", "saat", "puan"), rows, data_title),
        Statistic(SMALL, "saat", "sum", "toplam_x", "Toplam ΣXᵢ", decimals=0),
        Statistic(SMALL, "puan", "sum", "toplam_y", "Toplam ΣYᵢ", decimals=0),
        Statistic(SMALL, "saat", "mean", "xbar", "Ortalama X̄", decimals=1),
        Statistic(SMALL, "puan", "mean", "ybar", "Ortalama Ȳ", decimals=1),
        Derive(SMALL, "x_sapma", E.sub(E.var("saat"), E.ref("xbar")), "Sapma Xᵢ − X̄"),
        Derive(SMALL, "y_sapma", E.sub(E.var("puan"), E.ref("ybar")), "Sapma Yᵢ − Ȳ"),
        Derive(SMALL, "carpim", E.mul(E.var("x_sapma"), E.var("y_sapma")), "Çarpım (Xᵢ − X̄)(Yᵢ − Ȳ)"),
        Derive(SMALL, "x_kare", E.power(E.var("x_sapma"), 2), "Kareli sapma (Xᵢ − X̄)²"),
        Statistic(SMALL, "x_sapma", "sum", "toplam_x_sapma", "Toplam Σ(Xᵢ − X̄)", decimals=0),
        Statistic(SMALL, "y_sapma", "sum", "toplam_y_sapma", "Toplam Σ(Yᵢ − Ȳ)", decimals=0),
        Statistic(SMALL, "carpim", "sum", "pay", "Pay Σ(Xᵢ − X̄)(Yᵢ − Ȳ)", decimals=0),
        Statistic(SMALL, "x_kare", "sum", "payda", "Payda Σ(Xᵢ − X̄)²", decimals=0),
        Scalar("b1_kucuk", E.div(E.ref("pay"), E.ref("payda")), "Eğim β̂₁ = pay / payda (Denklem 3.8)", decimals=1),
        Scalar("b0_kucuk", E.sub(E.ref("ybar"), E.mul(E.ref("b1_kucuk"), E.ref("xbar"))),
               "Sabit β̂₀ = Ȳ − β̂₁X̄ (Denklem 3.9)", decimals=1),
        Derive(SMALL, "tahmin", E.add(E.ref("b0_kucuk"), E.mul(E.ref("b1_kucuk"), E.var("saat"))),
               "Tahmin edilen değer Ŷᵢ = β̂₀ + β̂₁Xᵢ"),
        Derive(SMALL, "artik", E.sub(E.var("puan"), E.var("tahmin")), "Artık ûᵢ = Yᵢ − Ŷᵢ"),
        Derive(SMALL, "artik_kare", E.power(E.var("artik"), 2), "Kareli artık ûᵢ²"),
        ShowFrame(SMALL, _SMALL_COLUMNS, "Tablo 3.2–3.3: hesap tablosu" if notes else
                  f"Hesap tablosu (beşinci öğrencinin notu {y5})"),
        Statistic(SMALL, "artik", "sum", "artik_toplami", "Artıkların toplamı Σûᵢ", decimals=1),
        Statistic(SMALL, "artik_kare", "sum", "artik_kare_toplami", "Kareli artıklar toplamı Σûᵢ²", decimals=2),
        OLS("model_kucuk", SMALL, "puan", ("saat",), "Aynı doğru yazılımla: puan ~ saat"),
        ModelValue("b1_yazilim", "model_kucuk", "coef", "Yazılımla eğim", term="saat", decimals=1),
        ModelValue("b0_yazilim", "model_kucuk", "coef", "Yazılımla sabit", term=INTERCEPT, decimals=1),
    )


def _small_note(state, choices) -> str:
    y5 = int(choices["adim3_y5"])
    s = state.scalars
    b1 = s["b1_kucuk"]
    sign = "+" if b1 >= 0 else "−"
    line = f"not̂ = {plain(s['b0_kucuk'], 1)} {sign} {plain(abs(b1), 1)} · saat"
    text = (f"Tahmin edilen doğru: {line}. Formüllerle bulunan eğim ve sabit, yazılımın EKK tahminiyle aynıdır. "
            f"Artıkların toplamı {plain(s['artik_toplami'], 1)}; kareli artıklar toplamı "
            f"{plain(s['artik_kare_toplami'], 2)}.")
    if y5 == SCORES[-1]:
        return text + (" Notlardaki değerlerle Denklem 3.10 elde edilir: bu beş öğrencilik örneklemde bir saat daha "
                       "fazla çalışan öğrencinin tahmin edilen notu ortalama 2,9 puan daha yüksektir. Sabit terimli "
                       "EKK doğrusunda artıkların toplamı sıfırdır ve doğru (X̄, Ȳ) = (6, 66) noktasından geçer.")
    return text + (
        f" Beşinci öğrencinin notu: {y5} (notlarda 78). Yalnız bir gözlem değişti; eğim ve sabit değişti. Doğru "
        "X = 4 noktası çevresinde döner: ikinci öğrencinin tahmini 60,2'de, artığı −0,2'de kalır, öteki tahminler ve "
        "artıklar değişir. Artıkların toplamı yine sıfırdır ve doğru yine (X̄, Ȳ) noktasından geçer. Eğimdeki değişim "
        "(X₅ − X̄)(Y₅ − 78)/Σ(Xᵢ − X̄)² = 4(Y₅ − 78)/40 kadardır: X̄'dan uzak bir gözlem eğimi daha çok etkiler."
    )


# --- Adım 4: WAGE1 basit regresyonu -----------------------------------------------------

def _model(choices) -> tuple:
    x = choices["adim4_x"]
    notes = x == "educ"
    operations = (
        OLS("model", DATA, OUTCOME, (x,), f"Basit regresyon: {OUTCOME} ~ {x}" + (" (Kod 3.1)" if notes else "")),
        ShowModel("model", "Kod 3.2: bu bölümde okunan alanlar" if notes else "Seçtiğiniz modelin çıktısı",
                  columns=("coef",), stats=("nobs",), stars=False),
    )
    if notes:
        return operations
    return operations + (
        OLS("model_notlar", DATA, OUTCOME, ("educ",), "Karşılaştırma için notlardaki model: wage ~ educ"),
        RegressionTable((("(1) Notlar", "model_notlar"), ("(2) Seçiminiz", "model")), ("educ", x, INTERCEPT),
                        "karsilastirma", "Notlardaki model ile seçtiğiniz model, bağımlı değişken: saatlik ücret",
                        stars=False, decimals=4, standard_errors=False, r2=False),
    )


def _model_note(state, choices) -> str:
    x = choices["adim4_x"]
    if x == "educ":
        return (
            "Örneklem regresyon doğrusu ŵage = −0,9049 + 0,5414 · educ (Denklem 3.11); tahmin 526 çalışana dayanır. "
            "`Intercept` satırı sabit tahmini β̂₀, `educ` satırı eğim tahmini β̂₁'dir. Adımın altındaki kodun "
            "yazdırdığı tam yazılım çıktısında görünen standart hata, t, p-değeri ve güven aralığı bu bölümde "
            "yorumlanmaz; R² Konu 4'te, ötekiler Konu 7'de işlenir."
        )
    words = IN_WORDS[x][1]
    return (
        f"Sütun (1) notlardaki eğitim modeli, sütun (2) {words} modelidir. İki basit regresyon farklı soruları "
        "cevaplar: her eğim yalnız kendi değişkeniyle ücret arasındaki örneklem ilişkisini özetler. Adım 5 ve 7 "
        "seçtiğiniz modeli kullanır."
    )


# --- Adım 5: eğim ve sabit terimin yorumu -----------------------------------------------

_ZERO_LABELS = {
    "educ": "Eğitimi sıfır olan çalışan sayısı",
    "exper": "Potansiyel deneyimi sıfır olan çalışan sayısı",
    "tenure": "Kıdemi sıfır olan çalışan sayısı",
}

def _interpretation(choices) -> tuple:
    x = choices["adim4_x"]
    return (
        ModelValue("b0", "model", "coef", "Sabit terim β̂₀", term=INTERCEPT),
        ModelValue("b1", "model", "coef", f"Eğim β̂₁ ({x})", term=x),
        Count(DATA, "x_sifir", x, 0, _ZERO_LABELS[x]),
    )


def _interpretation_note(state, choices) -> str:
    x = choices["adim4_x"]
    b0, b1, zeros = state.scalars["b0"], state.scalars["b1"], int(state.scalars["x_sifir"])
    phrase, words = IN_WORDS[x]
    slope = (f"Eğim: örneklemde {phrase} bir yıl daha yüksek olan çalışanların tahmin edilen saatlik ücreti "
             f"ortalama yaklaşık {plain(abs(b1), 3)} dolar daha {signed_difference(b1)}. Yorum sonucun örnekleme ait "
             "olduğunu, değişimin birimini (bir yıl) ve sonucun birimini (saat başına dolar) söyler.")
    where = (f"örneklemde {words} değeri sıfır olan {zeros} çalışan var" if zeros
             else f"örneklemde {words} değeri sıfır olan çalışan yok")
    constant = (f" Sabit terim {plain(b0, 4)}: {words} sıfır olduğunda doğrunun verdiği ücrettir; {where}. "
                "Sabit, doğrunun konumunu belirler; ekonomik yorumu verinin aralığına bağlıdır.")
    if x == "educ":
        constant += " Negatif bir ücret tahmini, doğrunun veri aralığının kenarına taşınmasından doğar."
    causal = (f" Nedensel dil sınırı: {words} bireylere rastgele atanmadığı için bu eğim, bir çalışanın {words} "
              "süresi bir yıl daha fazla olsaydı ücretinin kesin olarak ne kadar farklı olacağını söylemez.")
    return slope + constant + causal


# --- Adım 6: sıfır–bir açıklayıcı değişken (JTRAIN2) ------------------------------------

def _binary(choices) -> tuple:
    d = choices["adim6_d"]
    label = W.variable(JTRAIN, d).label
    notes = d == "train"
    ones, zeros = BINARY_WORDS[d]
    return (
        LoadWooldridge(JTRAIN, "JTRAIN2 veri seti (Wooldridge, 2020): 445 kişi, rastgele atamalı iş eğitimi"),
        GroupStats(JTRAIN, (d,), "re78", ("count", "mean"), "grup_ozeti",
                   f"{label}: gruplara göre kişi sayısı ve 1978 ortalama reel kazancı" + (" (§2.11)" if notes else ""),
                   decimals=4),
        Statistic(JTRAIN, "re78", "mean", "ort_0", f"Ortalama kazanç: {d} = 0 ({zeros})", where=(d, 0)),
        Statistic(JTRAIN, "re78", "mean", "ort_1", f"Ortalama kazanç: {d} = 1 ({ones})", where=(d, 1)),
        Scalar("grup_farki", E.sub(E.ref("ort_1"), E.ref("ort_0")), "Grup ortalamaları arasındaki fark"),
        OLS("model_jtrain", JTRAIN, "re78", (d,), f"Basit regresyon: re78 ~ {d}" + (" (Kod 3.3)" if notes else "")),
        ShowModel("model_jtrain", "Kod 3.4: temel katsayılar" if notes else "Seçtiğiniz modelin çıktısı",
                  columns=("coef",), stats=("nobs",), stars=False),
        ModelValue("sabit_j", "model_jtrain", "coef", "Sabit β̂₀", term=INTERCEPT),
        ModelValue("egim_j", "model_jtrain", "coef", f"Eğim β̂₁ ({d})", term=d),
        Scalar("bir_grubu", E.add(E.ref("sabit_j"), E.ref("egim_j")), f"β̂₀ + β̂₁: {d} = 1 ({ones}) için tahmin"),
    )


def _binary_note(state, choices) -> str:
    d = choices["adim6_d"]
    s = state.scalars
    ones, zeros = BINARY_WORDS[d]
    identity = (f"Sabit {plain(s['sabit_j'], 4)}, `{d}` = 0 grubunun ({zeros}) ortalamasına; sabit + eğim = "
                f"{plain(s['bir_grubu'], 4)}, `{d}` = 1 grubunun ({ones}) ortalamasına eşittir. Eğim "
                f"{plain(s['egim_j'], 4)}, iki grup ortalamasının farkıdır (bin dolar).")
    if d == "train":
        return identity + (
            " Program rastgele atandığı için programa katılım, kazancı etkileyen öteki faktörlerden bağımsızdır; sıfır "
            "koşullu ortalama koşulu (Denklem 3.5) tasarım gereği savunulabilir. Bu nedenle 1,794 bin dolarlık fark, "
            "programın ortalama kazanç etkisinin tahmini olarak okunabilir (Denklem 3.12)."
        )
    return identity + (
        " Bu özellik rastgele atanmadı: iki grup eğitim, deneyim ve başka özelliklerde farklı olabilir. Eğim yalnız "
        "grup ortalamalarının farkını özetler; nedensel etki olarak okunamaz. Nedensel dil yalnız rastgele atanan "
        "`train` değişkeninde kullanılır."
    )


# --- Adım 7: gözlenen değer, tahmin edilen değer ve artık -------------------------------

def _fitted(choices) -> tuple:
    x, x0, y0 = choices["adim4_x"], int(choices["adim7_x0"]), float(choices["adim7_y0"])
    return (
        Derive(DATA, "tahmin", E.add(E.ref("b0"), E.mul(E.ref("b1"), E.var(x))),
               "Tahmin edilen değer Ŷᵢ = β̂₀ + β̂₁Xᵢ"),
        Derive(DATA, "artik", E.sub(E.var(OUTCOME), E.var("tahmin")), "Artık ûᵢ = Yᵢ − Ŷᵢ"),
        ShowFrame(DATA, (OUTCOME, x, "tahmin", "artik"),
                  "Tablo 3.4: ilk altı gözlem" if x == "educ" else
                  "İlk altı gözlem: seçtiğiniz modelle tahmin ve artık",
                  head=6, decimals=2),
        Scalar("tahmin_x0", E.add(E.roundto(E.ref("b0"), 4), E.mul(E.roundto(E.ref("b1"), 4), x0)),
               f"x₀ = {x0} için tahmin (dört basamaklı katsayılarla)", decimals=3),
        Scalar("artik_y0", E.sub(y0, E.ref("tahmin_x0")), f"y₀ = {plain(y0, 2)} için artık", decimals=3),
    )


def _fitted_note(state, choices) -> str:
    x, x0 = choices["adim4_x"], int(choices["adim7_x0"])
    s = state.scalars
    words = IN_WORDS[x][1]
    residual = s["artik_y0"]
    position = ("doğrunun üzerindedir: gözlenen ücret tahminden yüksektir" if residual > 0 else
                "doğrunun altındadır: gözlenen ücret tahminden düşüktür" if residual < 0 else
                "tam doğrunun üzerindedir (artık sıfır)")
    low, high = {"educ": (0, 18), "exper": (1, 51), "tenure": (0, 44)}[x]
    text = (f"Tabloda ilk altı çalışanın tahmin edilen değeri ve artığı görünür; artık pozitifse gözlem doğrunun "
            f"üzerindedir. Seçtiğiniz noktada tahmin {plain(s['tahmin_x0'], 3)} dolar, artık "
            f"{plain(residual, 3)} dolar; gözlem {position}. Büyük bir artık tek başına ölçüm hatası anlamına gelmez.")
    if not low <= x0 <= high:
        text += (f" x₀ verideki {words} aralığının ({low}–{high} yıl) dışında: bu dışa doğru bir tahmindir ve doğrusal "
                 "ilişkinin orada da geçerli olduğu varsayımına dayanır.")
    return text


# --- Tanım ---------------------------------------------------------------------------

_SMALL_DEVIATIONS = {
    "x_sapma": (-4, -2, 0, 2, 4),
    "y_sapma": (-11, -6, -1, 6, 12),
    "carpim": (44, 12, 0, 12, 48),
    "x_kare": (16, 4, 0, 4, 16),
}
"""Tablo 3.2'nin sapma, çarpım ve kare sütunları."""
_SMALL_FITTED = {
    "tahmin": ((54.4, 60.2, 66.0, 71.8, 77.6), 1),
    "artik": ((0.6, -0.2, -1.0, 0.2, 0.4), 1),
    "artik_kare": ((0.36, 0.04, 1.00, 0.04, 0.16), 2),
}
"""Tablo 3.3: tahmin edilen değerler, artıklar ve kareli artıklar (basılı basamakla)."""
_FIRST_SIX = {
    "tahmin": (5.05, 5.59, 5.05, 3.43, 5.59, 7.76),
    "artik": (-1.95, -2.35, -2.05, 2.57, -0.29, 0.99),
}
"""Tablo 3.4: WAGE1'in ilk altı gözleminde tahmin edilen değer ve artık."""
_SMALL_NAMES = {"x_sapma": "Xᵢ − X̄", "y_sapma": "Yᵢ − Ȳ", "carpim": "çarpım", "x_kare": "(Xᵢ − X̄)²",
                "tahmin": "Ŷᵢ", "artik": "ûᵢ", "artik_kare": "ûᵢ²"}

STEPS = (
    interactive_step(
        number=1,
        title="Regresyon doğrusuna neden ihtiyaç duyarız?",
        note=NoteRef("3.1", 0, ("Şekil 3.1",)),
        explanation=(
            "Aynı eğitim düzeyindeki çalışanların ücretleri farklıdır; yine de eğitim yükseldikçe **ortalama** ücretin "
            "nasıl değiştiğini özetlemek isteriz. Grafikte soluk noktalar tek tek çalışanları, turuncu noktalar aynı "
            "düzeydeki çalışanların ortalama ücretini, doğru ise bütün örneklemin doğrusal özetini gösterir."
        ),
        controls=(
            Choice("adim1_x", "Yatay eksendeki değişken", W.options(DATA, REGRESSORS), "educ",
                   help="Notlardaki Şekil 3.1: eğitim."),
        ),
        build=_means,
        note_for=lambda state, choices: _means_note(state, choices),
    ),
    interactive_step(
        number=2,
        title="Koşullu ortalama ve eğim",
        note=NoteRef("3.3", 0, ("Denklem 3.1", "Denklem 3.2")),
        explanation=(
            "$\\mathbb{E}(\\text{ücret} \\mid \\text{eğitim} = x)$, eğitimi $x$ yıl olan çalışanların anakütledeki "
            "ortalama ücretidir; örneklemde bu grubun ortalamasıyla tahmin edilir. Doğrusal anakütle regresyon "
            "fonksiyonunda $\\mathbb{E}(Y \\mid X = x + 1) - \\mathbb{E}(Y \\mid X = x) = \\beta_1$ her $x$ için "
            "aynıdır. Bir eğitim düzeyi seçin; o düzeyin ve bir yıl üstünün ortalamasını karşılaştırın."
        ),
        controls=(
            NumberChoice("adim2_x", "Eğitim düzeyi x (yıl)", 8, 17, 12, 1, integer=True,
                         help="x ile x + 1 yıl karşılaştırılır. 8–18 yıl arasındaki her düzeyde en az 12 çalışan var."),
        ),
        build=_conditional,
        note_for=lambda state, choices: _conditional_note(state, choices),
    ),
    interactive_step(
        number=3,
        title="Adım adım küçük bir EKK hesabı",
        note=NoteRef("3.9", 0, ("Tablo 3.2", "Tablo 3.3", "Denklem 3.10")),
        explanation=(
            "Beş öğrencinin haftalık çalışma saati $X$ ve sınav notu $Y$ ile eğim ve sabit formülleri adım adım "
            "uygulanır: $\\hat\\beta_1 = \\sum (X_i - \\bar X)(Y_i - \\bar Y) / \\sum (X_i - \\bar X)^2$ (Denklem 3.8) "
            "ve $\\hat\\beta_0 = \\bar Y - \\hat\\beta_1 \\bar X$ (Denklem 3.9). Son satırlar aynı doğrunun yazılımla "
            "tahminidir. Beşinci öğrencinin notunu değiştirerek tek bir gözlemin doğruyu nasıl etkilediğini görün."
        ),
        controls=(
            NumberChoice("adim3_y5", "Beşinci öğrencinin sınav notu Y₅", 40, 100, 78, 1, integer=True,
                         help="Notlardaki Tablo 3.2: 78."),
        ),
        build=_small,
        checks=(
            _scalar("toplam_x", 30, "Tablo 3.2: ΣXᵢ", 0),
            _scalar("toplam_y", 330, "Tablo 3.2: ΣYᵢ", 0),
            _scalar("xbar", 6, "§3.9: X̄", 0),
            _scalar("ybar", 66, "§3.9: Ȳ", 0),
            *(Check(f"Tablo 3.2: {_SMALL_NAMES[column]}, {row}. öğrenci", CellTarget(SMALL, column, row), value, 0)
              for column, values in _SMALL_DEVIATIONS.items() for row, value in enumerate(values, start=1)),
            _scalar("toplam_x_sapma", 0, "Tablo 3.2: Σ(Xᵢ − X̄)", 0),
            _scalar("toplam_y_sapma", 0, "Tablo 3.2: Σ(Yᵢ − Ȳ)", 0),
            _scalar("pay", 116, "Tablo 3.2: Σ(Xᵢ − X̄)(Yᵢ − Ȳ)", 0),
            _scalar("payda", 40, "Tablo 3.2: Σ(Xᵢ − X̄)²", 0),
            _scalar("b1_kucuk", 2.9, "§3.9: eğim 116/40", 1),
            _scalar("b0_kucuk", 48.6, "§3.9: sabit 66 − 2,9 · 6", 1),
            *(Check(f"Tablo 3.3: {_SMALL_NAMES[column]}, {row}. öğrenci", CellTarget(SMALL, column, row), value,
                    decimals)
              for column, (values, decimals) in _SMALL_FITTED.items() for row, value in enumerate(values, start=1)),
            _scalar("artik_toplami", 0.0, "Tablo 3.3: Σûᵢ", 1),
            _scalar("artik_kare_toplami", 1.60, "Tablo 3.3: Σûᵢ²", 2),
            _scalar("b1_yazilim", 2.9, "Denklem 3.10: eğim (yazılımla)", 1),
            _scalar("b0_yazilim", 48.6, "Denklem 3.10: sabit (yazılımla)", 1),
        ),
        note_for=lambda state, choices: _small_note(state, choices),
    ),
    interactive_step(
        number=4,
        title="Python ile WAGE1 basit regresyonu",
        note=NoteRef("3.10", 0, ("Kod 3.1", "Kod 3.2", "Denklem 3.11")),
        explanation=(
            "`wage ~ educ` yazımı `wage`'i bağımlı, `educ`'u açıklayıcı değişken yapar. Kod 3.2 bu bölümde okunan "
            "alanları verir: bağımlı değişken, gözlem sayısı, sabit ve eğim. Kodun ezberlenmesi beklenmez."
        ),
        controls=(X_CHOICE,),
        build=_model,
        checks=(
            Check("Kod 3.2: gözlem sayısı", ModelTarget("model", "nobs"), 526, 0),
            Check("Kod 3.2: sabit terim", CoefTarget("model", INTERCEPT), -0.9049, 4),
            Check("Kod 3.2: eğitim katsayısı", CoefTarget("model", "educ"), 0.5414, 4),
        ),
        note_for=lambda state, choices: _model_note(state, choices),
    ),
    interactive_step(
        number=5,
        title="Eğim ve sabit terimi doğru yorumlamak",
        note=NoteRef("3.11", 0),
        explanation=(
            "Doğru bir eğim yorumu dört unsuru birlikte verir: sonucun örnekleme ait olduğu, $X$'in değişim birimi, "
            "$Y$'nin değişim birimi ve nedensel olmayan dil. Sabit terim $X = 0$ noktasındaki tahmindir; anlamı sıfır "
            "değerinin veride ne kadar gözlendiğine bağlıdır. Bu adım Adım 4'te kurulan modeli yorumlar."
        ),
        uses=(X_CHOICE,),
        build=_interpretation,
        checks=(
            _scalar("b1", 0.5414, "§3.11: eğitim katsayısı", 4),
            _scalar("b1", 0.541, "§3.11: yorumdaki yaklaşık eğim", 3),
            _scalar("b0", -0.9049, "§3.11: sabit terim", 4),
            _scalar("x_sifir", 2, "§3.11: eğitimi sıfır olan çalışan", 0),
        ),
        note_for=lambda state, choices: _interpretation_note(state, choices),
    ),
    interactive_step(
        number=6,
        title="Sıfır–bir açıklayıcı değişken: JTRAIN2",
        note=NoteRef("3.11", 0, ("Kod 3.3", "Kod 3.4", "Denklem 3.12")),
        explanation=(
            "Açıklayıcı değişken yalnız 0 ve 1 değerlerini alırsa $\\mathbb{E}(Y \\mid D = 0) = \\beta_0$ ve "
            "$\\mathbb{E}(Y \\mid D = 1) = \\beta_0 + \\beta_1$ olur: sabit $D = 0$ grubunun, sabit ile eğimin toplamı "
            "$D = 1$ grubunun ortalamasıdır; eğim iki grubun ortalama farkıdır. EKK tahmininde aynı eşitlik örneklem "
            "ortalamaları için geçerlidir. Notlarda $D$, eğitim programına atanmayı gösteren `train` değişkenidir."
        ),
        controls=(
            Choice("adim6_d", "Sıfır–bir açıklayıcı değişken (bağımlı değişken: 1978 reel kazancı)",
                   W.options(JTRAIN, BINARY), "train",
                   help="Notlardaki Kod 3.3: `train`. Yalnız `train` rastgele "
                        "atanmıştır; ötekiler kişisel özelliklerdir."),
        ),
        build=_binary,
        checks=(
            Check("§2.11: kontrol grubu kişi sayısı", TableTarget("grup_ozeti", 0, "count"), 260, 0),
            Check("§2.11: eğitim grubu kişi sayısı", TableTarget("grup_ozeti", 1, "count"), 185, 0),
            Check("§3.11: kontrol grubu ortalaması", TableTarget("grup_ozeti", 0, "mean"), 4.5548, 4),
            Check("§3.11: eğitim grubu ortalaması", TableTarget("grup_ozeti", 1, "mean"), 6.3491, 4),
            Check("Kod 3.4: gözlem sayısı", ModelTarget("model_jtrain", "nobs"), 445, 0),
            Check("Kod 3.4: sabit terim", CoefTarget("model_jtrain", INTERCEPT), 4.5548, 4),
            Check("Kod 3.4: train katsayısı", CoefTarget("model_jtrain", "train"), 1.7943, 4),
            _scalar("bir_grubu", 6.3491, "§3.11: 4,5548 + 1,7943", 4),
            _scalar("grup_farki", 1.7943, "§3.11: eğim = grup farkı", 4),
            _scalar("grup_farki", 1.794, "§3.11: 1,794 bin dolarlık fark", 3),
        ),
        note_for=lambda state, choices: _binary_note(state, choices),
    ),
    interactive_step(
        number=7,
        title="Gözlenen değer, tahmin edilen değer ve artık",
        note=NoteRef("3.12", 0, ("Şekil 3.4", "Tablo 3.4")),
        explanation=(
            "Tahmin edilen değer $\\hat Y_i = \\hat\\beta_0 + \\hat\\beta_1 X_i$, artık $\\hat u_i = Y_i - \\hat "
            "Y_i$'dir. Tablo 3.4 WAGE1'in ilk altı çalışanı içindir (yuvarlanmamış katsayılarla, iki ondalık). Altta "
            "notlardaki örnek: eğitimi 12 yıl, ücreti 7,00 dolar olan bir çalışan için dört basamaklı katsayılarla "
            "tahmin ve artık. Bu adım Adım 4'te kurulan modeli kullanır."
        ),
        uses=(X_CHOICE,),
        controls=(
            NumberChoice("adim7_x0", "Tahmin noktası x₀ (yıl)", 0, 51, 12, 1, integer=True,
                         help="Notlardaki örnek: eğitim 12 yıl. Verideki aralık: eğitim 0–18, deneyim 1–51, "
                              "kıdem 0–44 yıl; aralığın dışı dışa doğru tahmindir."),
            NumberChoice("adim7_y0", "Gözlenen ücret y₀ (dolar/saat)", 0.0, 30.0, 7.0, 0.25,
                         help="Notlardaki örnek: 7,00 dolar."),
        ),
        build=_fitted,
        checks=(
            *(Check(f"Tablo 3.4: {_SMALL_NAMES[column]}, {row}. gözlem", CellTarget(DATA, column, row), value, 2)
              for column, values in _FIRST_SIX.items() for row, value in enumerate(values, start=1)),
            _scalar("tahmin_x0", 5.592, "§3.12: −0,9049 + 0,5414(12)", 3),
            _scalar("artik_y0", 1.408, "§3.12: 7,00 − 5,592", 3),
        ),
        note_for=lambda state, choices: _fitted_note(state, choices),
    ),
    LabStep(
        number=8,
        title="Basit regresyon için okuma kontrol listesi",
        note=NoteRef("3.13", 0),
        explanation=(
            "Bir basit regresyon denklemi, Python çıktısı veya makale tablosu gördüğünüzde:\n\n1. Araştırma sorusu "
            "nedir?\n2. Gözlem birimi ve veri yapısı nedir?\n3. $Y$ ve $X$ hangi değişkenlerdir?\n4. Değişkenlerin "
            "ölçü birimleri nedir?\n5. Anakütle modeli ile örneklem tahmini ayrılmış mı?\n6. Eğim katsayısının işareti "
            "ve büyüklüğü nedir?\n7. Sabit terimin ekonomik yorumu veri aralığında anlamlı mı?\n8. Tahmin edilen değer "
            "ve artık nasıl hesaplanır?\n9. Yorum ilişki düzeyinde mi, nedensel düzeyde mi?\n10. Henüz öğrenilmemiş "
            "çıktı alanları için hangi sonraki bölüme bakılmalıdır?\n\nBir çıktıda çok sayıda sayı bulunması, bütün "
            "sayıların aynı anda yorumlanması gerektiği anlamına gelmez: R² Konu 4'te, standart hata, t, p ve güven "
            "aralığı Konu 7'de işlenir."
        ),
    ),
)


KONU03_LAB = LabSpec(
    topic_key="konu03",
    title="Uygulama: Basit Doğrusal Regresyon Modeli",
    note_section="3",
    steps=STEPS,
    labels=(
        *W.labels(JTRAIN, DATA),
        (INTERCEPT, "Sabit terim"),
        ("ogrenci", "Öğrenci i"),
        ("saat", "Çalışma saati X"),
        ("puan", "Sınav notu Y"),
        ("x_sapma", "Xᵢ − X̄"),
        ("y_sapma", "Yᵢ − Ȳ"),
        ("carpim", "Çarpım"),
        ("x_kare", "(Xᵢ − X̄)²"),
        ("tahmin", "Tahmin edilen değer Ŷᵢ"),
        ("artik", "Artık ûᵢ"),
        ("artik_kare", "Kareli artık ûᵢ²"),
    ),
    consistency_notes=(
        "Veri notlardaki gibi WAGE1 ve JTRAIN2'dir; notlardaki kod, betikler, uygulama ve üretilen kod onları "
        "wooldridge paketinden okur.",
        "Bölüm 2'nin sözü (JTRAIN2 grup farkı basit regresyonun eğimidir) §3.11'deki JTRAIN2 alt başlığında gösterilir "
        "(Kod 3.3–3.4, Denklem 3.12).",
        "Tablo 3.4 (ilk altı gözlemde tahmin ve artık) önceden yalnız sunumdaydı; notlara eklendi.",
        "Kod 3.2 yalnız bu bölümde okunan alanları yazdırır; üretilen kod yazılımın tam çıktısını da yazdırır.",
    ),
)
