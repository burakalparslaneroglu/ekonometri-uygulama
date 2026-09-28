"""Konu 1 Sezgi deneyleri: veri üretim süreci bilinen kontrollü simülasyonlar.

Deney 1  Kahve zinciri: sistematik bölüm ve hata terimi          (Notlar §1.3)
Deney 2  Aynı doğru, farklı sorular: tahmin ve nedensel etki     (Notlar §1.5)
Deney 3  Sabit terim ve verinin aralığı                           (Notlar §1.6, Adım 5)

Deney 1, Konu 1 sunumundaki kahve zinciri senaryosunun simülasyonudur: son 60 günde reklam harcaması, hafta sonu,
hava ve diğer etkilerle günlük satış (tohum 305). Sunumun hata terimi slaydı bu deneye yönlendirir. Deney 2 aynı
zincirde yöneticinin yoğun gün beklediğinde daha çok reklam verdiği durumu, Deney 3 WAGE1 uygulamasındaki negatif
sabit terimi (Denklem 1.4) kontrollü veriyle inceler. Konu 1'de standart hata ve test yoktur; deneyler katsayıyı,
R²'yi ve bilinen gerçeği karşılaştırır. Beklenen değer notlardaki gibi E(· | ·) ile yazılır (LaTeX'te 𝔼).
"""

from __future__ import annotations

import math

from core.labs import expr as E
from core.labs.runner import LabState
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, number, plain
from core.labs.spec import (
    INTERCEPT,
    OLS,
    Derive,
    Draw,
    DrawCount,
    LineChart,
    ModelValue,
    NewSample,
    NoteRef,
    Scalar,
    ScatterPlot,
    Statistic,
    Support,
)

SEED = 305
TOPIC = "konu01"


def _rounded(parameters: Parameters, key: str, decimals: int) -> float:
    """Kaydırıcı değeri, kodda 0,30000000000000004 gibi yazımlar olmasın diye yuvarlanır."""

    return round(float(parameters[key]), decimals)


def _tex(value: float, decimals: int = 2) -> str:
    """LaTeX içinde sondaki sıfırları atılmış sayı (13{,}5; 3100)."""

    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return text.replace(".", "{,}")


def _short(value: float, decimals: int = 2) -> str:
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return text.replace(".", ",").replace("-", "−")


# --- Deney 1: kahve zinciri, sistematik bölüm ve hata terimi ----------------------------------------

BASE_SALES = 3100
WEEKEND, WEATHER, OTHER = 560, 380, 500
"""Hata teriminin bileşenleri: hafta sonu etkisi (TL), bir standart sapmalık hava etkisi (TL), diğer etkilerin standart
sapması (TL)."""
WEEKEND_SHARE = 2 / 7
"""Bir gün hafta sonu olma olasılığı: haftanın 7 gününden 2'si."""
WEEKEND_MEAN = 160
"""Hata teriminin k = 1'deki ortalaması: 560 · 2/7 = 160 TL."""


def _coffee_settings(parameters: Parameters) -> tuple[float, float, int]:
    return _rounded(parameters, "olcek", 2), _rounded(parameters, "beta1", 1), int(parameters["n"])


def _build_coffee(parameters: Parameters) -> tuple:
    scale, slope, n = _coffee_settings(parameters)
    error = E.mul(scale, E.add(E.add(E.mul(WEEKEND, E.var("hafta_sonu")), E.mul(WEATHER, E.var("hava"))),
                               E.var("e")))
    # k = 0 ve β₁ = 0 iken satış sabittir: R² tanımsızdır ve ne uygulamada ne üretilen kodda hesaplanır.
    r_squared = () if scale == 0 and slope == 0 else (ModelValue("r2", "model", "r2", "R²", decimals=3),)
    return (
        NewSample("gunler", n, SEED),
        Draw("gunler", "reklam", "uniform", 20, 220, "Günlük dijital reklam harcaması (TL)"),
        DrawCount("gunler", "hafta_sonu", "binomial", (1, WEEKEND_SHARE),
                  "Hafta sonu mu? (1: evet; olasılık 2/7, haftanın 7 gününden 2'si)"),
        Draw("gunler", "hava", "normal", 0, 1, "Hava koşulları endeksi (ortalama 0)"),
        Draw("gunler", "e", "normal", 0, OTHER, "Diğer gözlenmeyen etkiler (TL)"),
        Derive("gunler", "u", error, "Hata terimi: u = k·(560·hafta sonu + 380·hava + e)"),
        Derive("gunler", "satis", E.add(E.add(BASE_SALES, E.mul(slope, E.var("reklam"))), E.var("u")),
               "Günlük satış (TL)"),
        OLS("model", "gunler", "satis", ("reklam",), "Basit regresyon: satış ~ reklam"),
        ModelValue("b0", "model", "coef", "Tahmin edilen sabit terim β̂₀", term=INTERCEPT, decimals=2),
        ModelValue("b1", "model", "coef", "Tahmin edilen eğim β̂₁", term="reklam", decimals=2),
        *r_squared,
        Scalar("sabit_gercek", E.add(BASE_SALES, E.mul(scale, WEEKEND_MEAN)),
               "Gerçek ortalama ilişkinin sabiti: 3100 + 160·k", decimals=2),
        ScatterPlot("gunler", "reklam", "satis", "Günlük reklam harcaması (TL)", "Günlük satış (TL)",
                    "Kahve zinciri: reklam harcaması ve günlük satış", fit_line=True, size=8, opacity=0.7,
                    lines=(("sabit_gercek", slope, "Gerçek ortalama ilişki E(satış | reklam)"),)),
    )


def _coffee_dgp(parameters: Parameters) -> tuple[str, ...]:
    scale, slope, n = _coffee_settings(parameters)
    return (
        rf"\text{{satış}}_i = 3100 + \beta_1\,\text{{reklam}}_i + u_i, \qquad \beta_1 = {_tex(slope, 1)}, "
        rf"\qquad i = 1, \dots, n = {n}",
        rf"u_i = k\,(560\,\text{{hafta sonu}}_i + 380\,\text{{hava}}_i + e_i), \qquad k = {_tex(scale)}",
        r"\text{reklam}_i \sim U(20,\ 220), \quad \text{hafta sonu}_i \sim \text{Bernoulli}(2/7), \quad "
        r"\text{hava}_i \sim N(0,\ 1), \quad e_i \sim N(0,\ 500^2)",
        rf"\text{{Gerçek ortalama ilişki: }} \mathbb{{E}}(\text{{satış}} \mid \text{{reklam}}) = "
        rf"{_tex(BASE_SALES + scale * WEEKEND_MEAN)} + {_tex(slope, 1)}\,\text{{reklam}}",
    )


def _coffee_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    scale, slope, _ = _coffee_settings(parameters)
    s = state.scalars
    return (
        SimMetric("Gerçek eğim β₁", plain(slope, 2), "Veri üretim sürecindeki reklam katsayısı: TL satış / TL reklam."),
        SimMetric("Tahmin β̂₁", plain(s["b1"], 2), "Bu örneklemden tahmin edilen eğim."),
        SimMetric("Tahmin β̂₀", plain(s["b0"], 0),
                  f"Gerçek ortalama ilişkinin sabiti 3100 + 160·k = {plain(s['sabit_gercek'], 0)} TL."),
        SimMetric("R²", plain(s["r2"], 3) if math.isfinite(s.get("r2", math.nan)) else "tanımsız",
                  "Satıştaki değişkenliğin reklamla doğrusal olarak açıklanan payı."),
    )


def _coffee_takeaway(state: LabState, parameters: Parameters) -> str:
    scale, slope, _ = _coffee_settings(parameters)
    s = state.scalars
    if scale == 0 and slope == 0:
        return (
            "k = 0 ve β₁ = 0: satış her gün 3100 TL'dir. Açıklanacak değişkenlik olmadığı için R² tanımsızdır ve "
            "tahmin edilen eğim sıfırdır. Gerçek verilerde satış günden güne değişir: bu değişimin bir kısmını reklam, "
            "bir kısmını hata terimindeki etkiler açıklar (§1.3)."
        )
    if scale == 0:
        return (
            "k = 0: hata terimi yok. Bütün noktalar doğru üzerindedir; tahmin edilen doğru gerçek ilişkiyle aynıdır ve "
            "R² = 1. Gerçek verilerde bu durum yoktur: satışı reklam dışında hafta sonu, hava, rakip kampanyaları gibi "
            "birçok faktör etkiler. Ekonometrik model bu etkileri uᵢ hata teriminde toplar (§1.3)."
        )
    if slope == 0:
        return (
            "β₁ = 0: reklamın satışla gerçek bir ilişkisi yok; satıştaki bütün değişkenlik hata teriminden gelir. "
            f"Tahmin edilen eğim ({plain(s['b1'], 2)}) sıfırdan yalnız bu örneklemin çekilişi yüzünden farklıdır ve "
            f"R² ({plain(s['r2'], 3)}) sıfıra yakındır. k burada yalnız noktaların dağılımının ölçeğini değiştirir; R² "
            "k'dan etkilenmez. Hata terimi yanlış hesap değildir; modelde açıkça yer almayan etkilerin toplamıdır (§1.3)."
        )
    return (
        f"Noktalar doğrunun çevresinde dağılır: aynı reklam harcamasında farklı satışlar gözlenir. Bu farkları hata "
        f"terimi u taşır (hafta sonu, hava ve diğer etkiler). Tahmin edilen eğim {plain(s['b1'], 2)}, gerçek eğim "
        f"{plain(slope, 2)} (TL satış / TL reklam): aradaki fark bu örneklemin çekilişinden gelir. Gözlenmeyen "
        f"etkilerin ölçeği k büyüdükçe noktalar doğrudan uzaklaşır ve R² genellikle düşer (şu an "
        f"{plain(s['r2'], 3)}); tahmin edilen eğim de gerçek eğimden daha fazla sapabilir. Hata terimi yanlış hesap "
        "değildir; modelde açıkça yer almayan etkilerin toplamıdır (§1.3)."
    )


COFFEE = SimExperiment(
    topic_key=TOPIC,
    number=1,
    title="Kahve zinciri: sistematik bölüm ve hata terimi",
    question="Günlük satışı reklam harcamasıyla açıklayan modelde hata terimi neyi temsil eder ve büyüdükçe veri ile "
             "tahmin edilen doğru nasıl değişir?",
    note=NoteRef("1.3"),
    parameters=(
        SimParameter("olcek", "Gözlenmeyen etkilerin ölçeği k", 0, 3, 1, 0.25,
                     "k = 0: hata terimi yok. k = 1: kahve zinciri senaryosunun temel ölçeği.", decimals=2),
        SimParameter("beta1", "Gerçek eğim β₁ (TL / TL)", 0, 30, 13.5, 0.5,
                     "Reklam harcaması 1 TL yüksek günlerde ortalama satış farkı.", decimals=1),
        SimParameter("n", "Gün sayısı n", 20, 365, 60, 5, "Sunumdaki senaryo: son 60 gün.", integer=True,
                     decimals=0),
    ),
    dgp=_coffee_dgp,
    dgp_note=(
        "Hafta sonu, hava ve diğer etkiler araştırmacının basit modelinde yoktur; hepsi hata terimindedir. Reklam bu "
        "etkilerden bağımsız çekilir. Hafta sonu (haftanın 7 gününden 2'si) satışa 560 TL eklediği için u'nun "
        "ortalaması 560 · 2/7 · k = 160·k TL'dir; gerçek ortalama ilişkinin sabiti bu yüzden 3100 + 160·k'dır."
    ),
    look_at=(
        "**Saçılım grafiği** — noktaların doğru çevresindeki dağılımı hata terimidir.",
        "**İki doğru** — kesikli çizgi bilinen gerçek ortalama ilişki, düz çizgi veriden tahmin edilen doğru.",
        "**R²** — k büyüdükçe satıştaki değişkenliğin reklamla açıklanan payı.",
    ),
    build=_build_coffee,
    metrics=_coffee_metrics,
    takeaway=_coffee_takeaway,
    labels=(("reklam", "Reklam harcaması (TL)"), ("satis", "Günlük satış (TL)"), ("hafta_sonu", "Hafta sonu"),
            ("hava", "Hava endeksi"), ("u", "Hata terimi")),
)


# --- Deney 2: tahmin sorusu ve nedensel soru ----------------------------------------------------------

BUSY_EFFECT = 600
"""Beklenen yoğunluk bir standart sapma yüksek olan günde satış farkı (TL)."""
AD_MEAN, AD_SPREAD = 120, 40
"""Reklam harcamasının ortalaması ve standart sapması (TL)."""
PREDICTION_POINT = 160
"""Tahmin sorusundaki reklam düzeyi: ortalamadan bir standart sapma yüksek (TL)."""
EXTRA_AD = 10


def _questions_settings(parameters: Parameters) -> tuple[float, float, int]:
    return _rounded(parameters, "rho", 1), _rounded(parameters, "beta1", 1), int(parameters["n"])


def _conditional_line(rho: float, slope: float) -> tuple[float, float]:
    """Gerçek koşullu ortalama E(satış | reklam = x) = sabit + eğim·x (yoğunluk ile reklam birlikte normal)."""

    shift = BUSY_EFFECT * rho / AD_SPREAD
    # Kodda 10,499999999999998 gibi yazımlar olmasın: doğrular altı basamakla yeterince kesindir.
    return round(BASE_SALES - shift * AD_MEAN, 6), round(slope + shift, 6)


def _build_questions(parameters: Parameters) -> tuple:
    rho, slope, n = _questions_settings(parameters)
    intercept, conditional_slope = _conditional_line(rho, slope)
    advertising = E.add(AD_MEAN, E.mul(AD_SPREAD, E.add(E.mul(rho, E.var("yogunluk")),
                                                        E.mul(E.sqrt(E.sub(1, E.power(rho, 2))), E.var("v")))))
    sales = E.add(E.add(E.add(BASE_SALES, E.mul(slope, E.var("reklam"))), E.mul(BUSY_EFFECT, E.var("yogunluk"))),
                  E.var("e"))
    return (
        NewSample("gunler", n, SEED),
        Draw("gunler", "yogunluk", "normal", 0, 1, "Beklenen yoğunluk (yönetici bilir, araştırmacı gözlemez)"),
        Draw("gunler", "v", "normal", 0, 1, "Reklam kararındaki diğer etkenler"),
        Draw("gunler", "e", "normal", 0, 400, "Diğer gözlenmeyen etkiler (TL)"),
        Derive("gunler", "reklam", advertising, "Reklam harcaması: 120 + 40·(ρ·yoğunluk + √(1 − ρ²)·v)"),
        Derive("gunler", "satis", sales, "Günlük satış: 3100 + β₁·reklam + 600·yoğunluk + e"),
        OLS("model", "gunler", "satis", ("reklam",), "Basit regresyon: satış ~ reklam"),
        ModelValue("b0", "model", "coef", "Tahmin edilen sabit terim β̂₀", term=INTERCEPT, decimals=2),
        ModelValue("b1", "model", "coef", "Tahmin edilen eğim β̂₁", term="reklam", decimals=2),
        Scalar("tahmin", E.add(E.ref("b0"), E.mul(E.ref("b1"), PREDICTION_POINT)),
               "Tahmin: reklam = 160 TL olan günde ortalama satış", decimals=0),
        Scalar("kosullu_ortalama", E.add(intercept, E.mul(conditional_slope, PREDICTION_POINT)),
               "Gerçek koşullu ortalama E(satış | reklam = 160)", decimals=0),
        Scalar("regresyon_cevabi", E.mul(E.ref("b1"), EXTRA_AD), "Regresyonun cevabı: 10 TL ek reklam", decimals=0),
        Scalar("nedensel_cevap", E.mul(slope, EXTRA_AD), "Gerçek etki: 10 TL ek reklam", decimals=0),
        ScatterPlot("gunler", "reklam", "satis", "Günlük reklam harcaması (TL)", "Günlük satış (TL)",
                    "Reklam ve satış: tahmin edilen doğru ve iki bilinen doğru", fit_line=True, size=7, opacity=0.6,
                    lines=((intercept, conditional_slope, "Gerçek koşullu ortalama E(satış | reklam)"),
                           (BASE_SALES, slope, "Yoğunluk sabitken reklamın etkisi (β₁)"))),
    )


def _questions_dgp(parameters: Parameters) -> tuple[str, ...]:
    rho, slope, n = _questions_settings(parameters)
    intercept, conditional_slope = _conditional_line(rho, slope)
    return (
        rf"\text{{satış}}_i = 3100 + \beta_1\,\text{{reklam}}_i + 600\,\text{{yoğunluk}}_i + e_i, \qquad "
        rf"\beta_1 = {_tex(slope, 1)}, \qquad i = 1, \dots, n = {n}",
        rf"\text{{reklam}}_i = 120 + 40\left(\rho\,\text{{yoğunluk}}_i + \sqrt{{1-\rho^2}}\,v_i\right), \qquad "
        rf"\rho = {_tex(rho, 1)}",
        r"\text{yoğunluk}_i \sim N(0,\ 1), \quad v_i \sim N(0,\ 1), \quad e_i \sim N(0,\ 400^2)",
        rf"\mathbb{{E}}(\text{{satış}} \mid \text{{reklam}}) = {_tex(intercept)} + "
        rf"{_tex(conditional_slope)}\,\text{{reklam}}",
    )


def _questions_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    s = state.scalars
    return (
        SimMetric("Nedensel etki β₁", plain(_questions_settings(parameters)[1], 2),
                  "Beklenen yoğunluk sabitken 1 TL ek reklamın satış etkisi (veri üretim süreci)."),
        SimMetric("Veriden eğim β̂₁", plain(s["b1"], 2),
                  "Reklamı 1 TL yüksek günlerin ortalama satış farkı: örneklem ilişkisi."),
        SimMetric("Tahmin, reklam 160", f"{plain(s['tahmin'], 0)} TL",
                  "Regresyon doğrusunun reklam = 160 TL'deki değeri."),
        SimMetric("Gerçek ortalama", f"{plain(s['kosullu_ortalama'], 0)} TL",
                  "Reklamın 160 TL olduğu günlerde gerçek ortalama satış E(satış | reklam = 160)."),
    )


def expected_shift(rho: float) -> float:
    """Yoğunluk sabit tutulmadığında regresyon eğiminin ortalamada kayması (TL/TL): 600 · 40ρ / 40² = 15ρ."""

    return BUSY_EFFECT * rho / AD_SPREAD


def sampling_dominates(rho: float, slope: float, b1: float) -> bool:
    """Bu örneklemde eğim farkının işareti beklenen kaymanın tersiyse ya da fark beklenen kaymadan yarısından çok
    ayrılıyorsa örneklem çekilişi baskındır."""

    shift, gap = expected_shift(rho), b1 - slope
    return gap * shift <= 0 or abs(gap - shift) > 0.5 * shift


def _questions_takeaway(state: LabState, parameters: Parameters) -> str:
    rho, slope, _ = _questions_settings(parameters)
    s = state.scalars
    prediction = (f"Tahmin sorusu: reklamın 160 TL olduğu günlerde ortalama satış için doğru "
                  f"{plain(s['tahmin'], 0)} TL verir; gerçek koşullu ortalama {plain(s['kosullu_ortalama'], 0)} TL. "
                  "Doğru bu koşullu ortalamayı hedefler; aradaki fark örneklem çekilişinden gelir ve n büyüdükçe "
                  "küçülür.")
    if rho == 0:
        return (
            "ρ = 0: reklam kararı beklenen yoğunluktan bağımsız; iki bilinen doğru aynı eğimdedir. Veriden eğim "
            f"({plain(s['b1'], 2)}) ile nedensel etki ({plain(slope, 2)}) arasındaki fark yalnız bu örneklemin "
            f"çekilişinden gelir. {prediction} Bu durumda aynı doğru hem tahmin hem nedensel soru için kullanılabilir."
        )
    shift = expected_shift(rho)
    gap = s["regresyon_cevabi"] - s["nedensel_cevap"]
    text = (
        f"Yönetici yoğun gün beklediğinde daha çok reklam veriyor (ρ = {_short(rho, 1)}): yüksek reklamlı günler "
        f"ortalamada daha yoğundur. Veriden eğim {plain(s['b1'], 2)}, reklamın gerçek etkisi {plain(slope, 2)}. "
        f"{prediction} Nedensel soru: reklamı 10 TL artırmanın etkisi regresyona göre "
        f"{plain(s['regresyon_cevabi'], 0)} TL, gerçekte {plain(s['nedensel_cevap'], 0)} TL. Yoğunluk sabit "
        f"tutulmadığı için regresyon eğimi ortalamada 15·ρ = {plain(shift, 2)} TL/TL yukarı kayar (10 TL için "
        f"{plain(EXTRA_AD * shift, 0)} TL); bu örneklemdeki fark ({plain(gap, 0)} TL) bu kaymayı ve örneklem "
        "çekilişini birlikte içerir."
    )
    if sampling_dominates(rho, slope, s["b1"]):
        text += (" Bu ayarda örneklem çekilişi baskındır: fark beklenen kaymadan belirgin biçimde ayrılıyor; n'yi "
                 "artırın.")
    return text + " Aynı çıktı dört soru türüne kendiliğinden cevap vermez (§1.5)."


QUESTIONS = SimExperiment(
    topic_key=TOPIC,
    number=2,
    title="Aynı doğru, farklı sorular: tahmin ve nedensel etki",
    question="Yönetici yoğun gün beklediğinde daha çok reklam veriyorsa, reklam–satış doğrusu hangi soruya doğru cevap "
             "verir: belirli bir reklam düzeyinde ortalama satışı tahmin etmeye mi, reklamı artırmanın etkisine mi?",
    note=NoteRef("1.5"),
    parameters=(
        SimParameter("rho", "Reklam ile beklenen yoğunluğun korelasyonu ρ", 0, 0.9, 0.5, 0.1,
                     "ρ = 0: reklam kararı yoğunluktan bağımsız.", decimals=1),
        SimParameter("beta1", "Reklamın gerçek etkisi β₁ (TL / TL)", 0, 30, 13.5, 0.5,
                     "Beklenen yoğunluk sabitken 1 TL ek reklamın satış etkisi.", decimals=1),
        SimParameter("n", "Gün sayısı n", 30, 730, 120, 10, "Gözlenen gün sayısı.", integer=True, decimals=0),
    ),
    dgp=_questions_dgp,
    dgp_note=(
        "Beklenen yoğunluğu yönetici bilir ve reklam kararında kullanır; araştırmacının verisinde yoktur. Yoğunluk "
        "satışı doğrudan da artırır. Son satır gerçek koşullu ortalamadır: reklamın belirli bir değer aldığı günlerde "
        "ortalama satış."
    ),
    look_at=(
        "**Düz çizgi ve birinci kesikli çizgi** — tahmin edilen doğru gerçek koşullu ortalamayı hedefler; n büyüdükçe "
        "ona yaklaşır: tahmin sorusu.",
        "**İkinci kesikli çizgi** — yoğunluk sabitken reklamın etkisi: nedensel soru. ρ > 0 iken daha yatıktır.",
        "**Metrikler** — veriden eğim ile nedensel etki, tahmin ile gerçek ortalama.",
    ),
    build=_build_questions,
    metrics=_questions_metrics,
    takeaway=_questions_takeaway,
    labels=(("reklam", "Reklam harcaması (TL)"), ("satis", "Günlük satış (TL)"), ("yogunluk", "Beklenen yoğunluk")),
)


# --- Deney 3: sabit terim ve verinin aralığı ------------------------------------------------------------

LOG_INTERCEPT, LOG_SLOPE, LOG_SPREAD = 0.4, 0.1, 0.45
"""ücret = exp(0,4 + 0,1·eğitim + e), e ~ N(0, 0,45²): WAGE1'e benzer düzeyler ve dışbükey ortalama ilişki."""
MAX_EDUCATION = 18


def _intercept_settings(parameters: Parameters) -> tuple[int, int]:
    return int(parameters["alt"]), int(parameters["n"])


def _true_mean(education: float) -> float:
    return math.exp(LOG_INTERCEPT + LOG_SLOPE * education + LOG_SPREAD ** 2 / 2)


def _build_intercept(parameters: Parameters) -> tuple:
    lowest, n = _intercept_settings(parameters)
    mean_curve = E.exp(E.add(E.add(LOG_INTERCEPT, E.mul(LOG_SLOPE, E.var("educ"))), E.div(E.power(LOG_SPREAD, 2), 2)))
    return (
        NewSample("orneklem", n, SEED),
        Draw("orneklem", "u", "uniform", lowest, MAX_EDUCATION + 1, "Eğitimi çekmek için tek-düze sayı"),
        Derive("orneklem", "educ", E.floor(E.var("u")), f"Eğitim (yıl): {lowest}, …, 18 arasında eşit olasılıklı"),
        Draw("orneklem", "e", "normal", 0, LOG_SPREAD, "Gözlenmeyen etkiler"),
        Derive("orneklem", "wage", E.exp(E.add(E.add(LOG_INTERCEPT, E.mul(LOG_SLOPE, E.var("educ"))), E.var("e"))),
               "Saatlik ücret: exp(0,4 + 0,1·eğitim + e)"),
        OLS("model", "orneklem", "wage", ("educ",), "Basit regresyon: wage ~ educ"),
        ModelValue("b0", "model", "coef", "Tahmin edilen sabit terim β̂₀", term=INTERCEPT, decimals=3),
        ModelValue("b1", "model", "coef", "Tahmin edilen eğim β̂₁", term="educ", decimals=3),
        Statistic("orneklem", "educ", "min", "en_dusuk", "Örneklemdeki en düşük eğitim", decimals=0),
        Scalar("gercek_sifir", E.exp(E.add(LOG_INTERCEPT, E.div(E.power(LOG_SPREAD, 2), 2))),
               "Gerçek ortalama ücret, eğitim = 0", decimals=3),
        ScatterPlot("orneklem", "educ", "wage", "Eğitim (yıl)", "Saatlik ücret", "Örneklem ve tahmin edilen doğru",
                    fit_line=True, size=6, opacity=0.35),
        Support("izgara", "educ", 0, MAX_EDUCATION, "Eğitim ızgarası: 0, 1, …, 18"),
        Derive("izgara", "gercek", mean_curve, "Gerçek ortalama ücret E(ücret | eğitim)"),
        Derive("izgara", "dogru", E.add(E.ref("b0"), E.mul(E.ref("b1"), E.var("educ"))),
               "Tahmin edilen doğru, eğitim = 0'a uzatılmış"),
        LineChart("izgara", "educ", "gercek", "Eğitim (yıl)", "Saatlik ücret",
                  "Sabit terim: doğrunun eğitim = 0'a uzatılması", markers=False,
                  series=(("dogru", "Tahmin edilen doğru (β̂₀ + β̂₁·eğitim)"),),
                  legend="Gerçek ortalama ücret E(ücret | eğitim)"),
    )


def _intercept_dgp(parameters: Parameters) -> tuple[str, ...]:
    lowest, n = _intercept_settings(parameters)
    return (
        rf"\text{{ücret}}_i = \exp(0{{,}}4 + 0{{,}}1\,\text{{eğitim}}_i + e_i), \qquad e_i \sim N(0,\ 0{{,}}45^2), "
        rf"\qquad i = 1, \dots, n = {n}",
        rf"\text{{eğitim}}_i \in \{{{lowest}, {lowest + 1}, \dots, 18\}}\ \text{{eşit olasılıkla}}",
        rf"\mathbb{{E}}(\text{{ücret}} \mid \text{{eğitim}}) = \exp(0{{,}}4 + 0{{,}}1\,\text{{eğitim}} + "
        rf"0{{,}}45^2/2), \qquad \mathbb{{E}}(\text{{ücret}} \mid \text{{eğitim}} = 0) = {number(_true_mean(0), 3)}",
    )


def _intercept_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    s = state.scalars
    return (
        SimMetric("Sabit terim β̂₀", plain(s["b0"], 3), "Doğrunun eğitim = 0'daki değeri."),
        SimMetric("Gerçek E(ücret | 0)", plain(s["gercek_sifir"], 3), "Eğitimi 0 olanların gerçek ortalama ücreti."),
        SimMetric("Eğim β̂₁", plain(s["b1"], 3), "Bir yıl fazla eğitimli çalışanların tahmin edilen ortalama ücret farkı."),
        SimMetric("En düşük eğitim", plain(s["en_dusuk"], 0), "Örneklemdeki en düşük eğitim yılı."),
    )


def _intercept_takeaway(state: LabState, parameters: Parameters) -> str:
    s = state.scalars
    lowest = int(round(s["en_dusuk"]))
    sign = "negatif" if s["b0"] < 0 else "pozitif"
    fit = (f"Sabit terim β̂₀ = {plain(s['b0'], 3)}: doğrunun eğitim = 0'daki değeri ({sign}). Gerçek ortalama ücret "
           f"eğitim 0'da {plain(s['gercek_sifir'], 3)}.")
    if lowest > 0:
        return (
            f"Örneklemde eğitimi {lowest} yıldan az kimse yok. {fit} Eğitim = 0 bu örneklemin dışında olduğu için sabit "
            "terim doğrunun veri aralığının dışına uzatılmasıyla bulunur. Doğrusal özet verinin aralığında ortalama "
            "ilişkiyi yaklaşık olarak özetler; aralığın dışında yanıltabilir. WAGE1'de durum biraz farklıdır: eğitimi 0 "
            "olan yalnız iki çalışan vardır ve bu değer verinin merkezinden (ortalama 12,56 yıl) uzaktır; negatif ücret "
            "de ekonomik olarak anlamlı değildir. Bu yüzden WAGE1'deki negatif sabit terim (−0,9049) ekonomik olarak "
            "yorumlanmaz (§1.6, Adım 5)."
        )
    return (
        f"Örneklemde eğitimi 0 olan çalışanlar var. {fit} Doğru bu kez veri aralığının içindedir; yine de gerçek "
        "ilişki eğri olduğu için doğrusal özet uçlarda sapar. Sabit terimi yorumlamadan önce eğitim = 0'ın veride "
        "bulunup bulunmadığına ve ilişkinin o bölgede doğrusal olup olmadığına bakılır (§1.6, Adım 5)."
    )


INTERCEPT_RANGE = SimExperiment(
    topic_key=TOPIC,
    number=3,
    title="Sabit terim ve verinin aralığı",
    question="Eğitimi 0 olan kimse örneklemde yoksa sabit terim neyi söyler? Tahmin edilen doğru verinin dışına "
             "uzatıldığında gerçek ortalama ilişkiden ne kadar ayrılır?",
    note=NoteRef("1.6", 5, ("Denklem 1.4",)),
    parameters=(
        SimParameter("alt", "Örneklemdeki en düşük eğitim (yıl)", 0, 12, 8, 1,
                     "Eğitim bu değer ile 18 arasında eşit olasılıkla çekilir.", integer=True, decimals=0),
        SimParameter("n", "Çalışan sayısı n", 100, 2000, 526, 1, "WAGE1'de 526 çalışan.", integer=True, decimals=0),
    ),
    dgp=_intercept_dgp,
    dgp_note=(
        "Ortalama ücret eğitimle dışbükey (hızlanarak) artar ve her eğitim düzeyinde pozitiftir. Araştırmacı bu eğri "
        "yerine bir doğru tahmin eder; sabit terim doğrunun eğitim = 0'daki değeridir."
    ),
    look_at=(
        "**Saçılım grafiği** — örneklemin eğitim aralığı ve tahmin edilen doğru.",
        "**Çizgi grafiği** — doğru eğitim = 0'a uzatıldığında gerçek ortalama ücretten ne kadar ayrılıyor?",
        "**Metrikler** — sabit terim ile eğitim = 0'daki gerçek ortalama ücret.",
    ),
    build=_build_intercept,
    metrics=_intercept_metrics,
    takeaway=_intercept_takeaway,
    labels=(("educ", "Eğitim (yıl)"), ("wage", "Saatlik ücret"), ("gercek", "Gerçek ortalama ücret"),
            ("dogru", "Tahmin edilen doğru")),
)


KONU01_EXPERIMENTS = (COFFEE, QUESTIONS, INTERCEPT_RANGE)
