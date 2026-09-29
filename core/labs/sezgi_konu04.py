"""Konu 4 Sezgi deneyleri: veri üretim süreci bilinen kontrollü simülasyonlar.

Deney 1  R² ve gürültü: aynı eğim, farklı uyum                        (Notlar §4.5–4.6)
Deney 2  X'in yayılımı ve R²                                          (Notlar §4.5)
Deney 3  Fonksiyonel biçim: log–log ilişkide iki model                (Notlar §4.6, §4.9)

Bölüm 4'te simülasyonla üretilmiş bir tablo ya da şekil yoktur; deneyler bölümün kavramlarını bilinen bir anakütle
modeliyle gösterir. Anakütle R²'si β₁²·Var(X) / (β₁²·Var(X) + σ²) formülüyle hesaplanır ve örneklem R²'siyle yan yana
yazılır (MKT = β̂₁²·Σ(Xᵢ − X̄)², §4.5). Yüzde yorumları yaklaşıktır; tam yüzde dönüşümü Konu 9'dadır. Tohum 305.
"""

from __future__ import annotations

import math

from core.labs import expr as E
from core.labs.runner import LabState
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, plain
from core.labs.wording import signed_difference
from core.labs.spec import (
    OLS,
    Derive,
    Draw,
    ModelValue,
    NewSample,
    NoteRef,
    ScatterPlot,
    Statistic,
)

SEED = 305
TOPIC = "konu04"


def _rounded(parameters: Parameters, key: str, decimals: int) -> float:
    """Kaydırıcı değeri, kodda 0,30000000000000004 gibi yazımlar olmasın diye yuvarlanır."""

    return round(float(parameters[key]), decimals)


def _tex(value: float, decimals: int = 2) -> str:
    """LaTeX içinde sondaki sıfırları atılmış sayı (0{,}5; 3)."""

    text = f"{value:.{decimals}f}"
    text = text.rstrip("0").rstrip(".") if "." in text else text  # 20 → "2" olmasın
    return "0" if text == "-0" else text.replace(".", "{,}")


def population_r2(slope: float, var_x: float, sigma: float) -> float:
    """Anakütle R²'si: β₁²·Var(X) / (β₁²·Var(X) + σ²); σ = 0 ve β₁ ≠ 0 ise 1."""

    signal = slope ** 2 * var_x
    return signal / (signal + sigma ** 2) if signal + sigma ** 2 > 0 else math.nan


# --- Deney 1: R² ve gürültü ------------------------------------------------------------------------

BETA_1 = (1.0, 0.5)
"""Deney 1 ve 2'nin anakütle doğrusu: E(Y | X) = 1 + 0,5·X."""
VAR_UNIFORM_10 = 100 / 12
"""X ~ U(0, 10) için Var(X) = 10²/12."""


def _noise_settings(parameters: Parameters) -> tuple[int, float]:
    return int(parameters["n"]), _rounded(parameters, "sigma", 1)


def _build_noise(parameters: Parameters) -> tuple:
    n, sigma = _noise_settings(parameters)
    intercept, slope = BETA_1
    return (
        NewSample("orneklem", n, SEED),
        Draw("orneklem", "x", "uniform", 0, 10, "Açıklayıcı değişken X ~ U(0, 10)"),
        Draw("orneklem", "u", "normal", 0, sigma, f"Hata terimi u ~ N(0, σ²), σ = {plain(sigma, 1)}"),
        Derive("orneklem", "y", E.add(E.add(intercept, E.mul(slope, E.var("x"))), E.var("u")), "Y = 1 + 0,5·X + u"),
        OLS("model", "orneklem", "y", ("x",), "Basit regresyon: y ~ x"),
        ModelValue("b1", "model", "coef", "Tahmin edilen eğim β̂₁", term="x", decimals=3),
        ModelValue("r2", "model", "r2", "R²", decimals=3),
        ScatterPlot("orneklem", "x", "y", "X", "Y", f"Aynı doğru, σ = {plain(sigma, 1)}: tahmin ve anakütle doğrusu",
                    fit_line=True, size=7, opacity=0.6,
                    lines=((intercept, slope, "Anakütle doğrusu E(Y | X) = 1 + 0,5·X"),)),
    )


def _noise_dgp(parameters: Parameters) -> tuple[str, ...]:
    n, sigma = _noise_settings(parameters)
    return (
        rf"Y_i = 1 + 0{{,}}5\,X_i + u_i, \qquad X_i \sim U(0,\ 10), \qquad u_i \sim N(0,\ \sigma^2), \qquad "
        rf"\sigma = {_tex(sigma, 1)}, \quad n = {n}",
        r"\text{Anakütle } R^2 = \frac{\beta_1^2\,\operatorname{Var}(X)}{\beta_1^2\,\operatorname{Var}(X) + \sigma^2}, "
        r"\qquad \operatorname{Var}(X) = 10^2/12",
    )


def _noise_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    _, sigma = _noise_settings(parameters)
    s = state.scalars
    return (
        SimMetric("Anakütle eğimi β₁", plain(BETA_1[1], 3), "Veri üretim sürecindeki eğim."),
        SimMetric("Tahmin β̂₁", plain(s["b1"], 3), "Bu örneklemin EKK eğimi."),
        SimMetric("Örneklem R²", plain(s["r2"], 3), "Bu örneklemde MKT / TKT."),
        SimMetric("Anakütle R²", plain(population_r2(BETA_1[1], VAR_UNIFORM_10, sigma), 3),
                  "β₁²·Var(X) / (β₁²·Var(X) + σ²): çok büyük bir örneklemde R²'nin yaklaştığı değer."),
    )


def _noise_takeaway(state: LabState, parameters: Parameters) -> str:
    _, sigma = _noise_settings(parameters)
    s = state.scalars
    if sigma == 0:
        return (
            "σ = 0: hata terimi yok; bütün noktalar doğrunun üzerindedir, R² = 1 ve tahmin edilen eğim 0,5'tir. Gerçek "
            "verilerde modelde yer almayan etkiler her zaman vardır (§4.6)."
        )
    return (
        f"Anakütle doğrusu her ayarda aynıdır (eğim 0,5); σ hata teriminin büyüklüğünü, n gözlem sayısını değiştirir. "
        f"Bu örneklemde β̂₁ = {plain(s['b1'], 3)} ve R² = {plain(s['r2'], 3)}; anakütle R²'si "
        f"{plain(population_r2(BETA_1[1], VAR_UNIFORM_10, sigma), 3)}. σ büyüdükçe anakütle R²'si düşer: aynı "
        "ilişkinin çevresinde daha büyük sapmalar vardır ve değişkenliğin daha küçük bir payı doğru boyunca hareket "
        "eder. Örneklem R²'si bu değerin çevresinde dalgalanır; küçük örneklemlerde dalgalanma büyüktür. Deney aynı "
        "çekilişleri σ ile ölçeklediği için bu örneklemde eğim tahmininin 0,5'ten sapması da σ ile orantılı büyür: "
        "gürültü arttıkça tek bir örneklemin tahmini gerçek eğimden daha uzağa düşebilir (başka bir örneklemde sapma "
        "öbür yöne olabilir). Bu belirsizlik Konu 7'de standart hatayla ölçülür. Düşük R² eğimin yanlış ya da önemsiz "
        "olduğu anlamına gelmez; yüksek R² de modelin doğru ya da nedensel olduğunu göstermez (§4.6)."
    )


NOISE = SimExperiment(
    topic_key=TOPIC,
    number=1,
    title="R² ve gürültü: aynı eğim, farklı uyum",
    question="İki araştırmacı aynı ilişkiyi inceliyor; birinin verisinde gözlenmeyen etkiler daha büyük. Eğim "
             "tahminleri ve R² değerleri nasıl farklılaşır? Düşük R² eğimin yanlış olduğunu gösterir mi?",
    note=NoteRef("4.5", objects=("§4.6",)),
    parameters=(
        SimParameter("sigma", "Hata teriminin standart sapması σ", 0, 10, 2, 0.5, "σ = 0: hata terimi yok.",
                     decimals=1),
        SimParameter("n", "Örneklem büyüklüğü n", 20, 500, 100, 10, "Gözlem sayısı.", integer=True, decimals=0),
    ),
    dgp=_noise_dgp,
    dgp_note=(
        "Eğim ve X'in dağılımı her ayarda aynıdır; σ kaydırıcısı hata teriminin büyüklüğünü, n kaydırıcısı gözlem "
        "sayısını değiştirir. Aynı tohumla aynı standart normal çekilişler z kullanılır ve u = σ·z olur: σ değişince "
        "yalnız gürültünün ölçeği değişir. Anakütle R²'si, MKT = "
        "β̂₁²·Σ(Xᵢ − X̄)² ilişkisinin (§4.5) anakütledeki karşılığıdır."
    ),
    look_at=(
        "**Grafik** — noktaların doğru çevresindeki dağılımı σ ile nasıl değişiyor?",
        "**R²** — örneklem R²'si ile anakütle R²'si.",
        "**n** — aynı σ'da örneklem R²'si anakütle R²'sine ne kadar yakın; n küçülünce ne oluyor?",
    ),
    build=_build_noise,
    metrics=_noise_metrics,
    takeaway=_noise_takeaway,
    labels=(("x", "X"), ("y", "Y"), ("u", "Hata terimi u")),
)


# --- Deney 2: X'in yayılımı ve R² ----------------------------------------------------------------------

N_2, SIGMA_2, CENTER_2 = 100, 2.0, 5.0
"""Deney 2: n = 100, σ = 2 ve X'in merkezi 5 sabittir; yalnız X'in aralığının genişliği değişir."""


def _spread_settings(parameters: Parameters) -> float:
    return _rounded(parameters, "genislik", 1)


def _build_spread(parameters: Parameters) -> tuple:
    width = _spread_settings(parameters)
    intercept, slope = BETA_1
    low, high = round(CENTER_2 - width / 2, 2), round(CENTER_2 + width / 2, 2)
    return (
        NewSample("orneklem", N_2, SEED),
        Draw("orneklem", "x", "uniform", low, high, f"Açıklayıcı değişken X ~ U({plain(low, 2)}; {plain(high, 2)})"),
        Draw("orneklem", "u", "normal", 0, SIGMA_2, "Hata terimi u ~ N(0; 2²)"),
        Derive("orneklem", "y", E.add(E.add(intercept, E.mul(slope, E.var("x"))), E.var("u")), "Y = 1 + 0,5·X + u"),
        Statistic("orneklem", "x", "std", "s_x", "X'in örneklem standart sapması", decimals=3),
        OLS("model", "orneklem", "y", ("x",), "Basit regresyon: y ~ x"),
        ModelValue("b1", "model", "coef", "Tahmin edilen eğim β̂₁", term="x", decimals=3),
        ModelValue("r2", "model", "r2", "R²", decimals=3),
        ScatterPlot("orneklem", "x", "y", "X", "Y", f"X'in aralığı {plain(width, 1)} birim: tahmin ve anakütle doğrusu",
                    fit_line=True, size=7, opacity=0.6,
                    lines=((intercept, slope, "Anakütle doğrusu E(Y | X) = 1 + 0,5·X"),)),
    )


def _spread_dgp(parameters: Parameters) -> tuple[str, ...]:
    width = _spread_settings(parameters)
    low, high = CENTER_2 - width / 2, CENTER_2 + width / 2
    return (
        rf"Y_i = 1 + 0{{,}}5\,X_i + u_i, \qquad u_i \sim N(0,\ 2^2), \qquad n = {N_2}",
        rf"X_i \sim U({_tex(low)};\ {_tex(high)}), \qquad "
        rf"\operatorname{{Var}}(X) = w^2/12, \qquad w = {_tex(width, 1)}",
    )


def _spread_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    width = _spread_settings(parameters)
    s = state.scalars
    return (
        SimMetric("X'in std. sapması", plain(s["s_x"], 3), "Örneklemde X'in standart sapması."),
        SimMetric("Tahmin β̂₁", plain(s["b1"], 3), "Anakütle eğimi 0,5."),
        SimMetric("Örneklem R²", plain(s["r2"], 3), "Bu örneklemde MKT / TKT."),
        SimMetric("Anakütle R²", plain(population_r2(BETA_1[1], width ** 2 / 12, SIGMA_2), 3),
                  "β₁²·Var(X) / (β₁²·Var(X) + σ²), Var(X) = w²/12."),
    )


def _spread_takeaway(state: LabState, parameters: Parameters) -> str:
    width = _spread_settings(parameters)
    s = state.scalars
    return (
        f"İlişki her ayarda aynıdır: eğim 0,5, hata teriminin standart sapması 2. Değişen yalnız X'in örneklemdeki "
        f"yayılımıdır (aralık {plain(width, 1)} birim). MKT = β̂₁²·Σ(Xᵢ − X̄)² ilişkisinin (§4.5) anakütledeki "
        "karşılığı β₁²·Var(X) = 0,25·w²/12'dir. X daha geniş bir aralıkta değiştikçe bu pay büyür ve anakütle R²'si "
        f"artar; şu an anakütle R²'si {plain(population_r2(BETA_1[1], width ** 2 / 12, SIGMA_2), 3)}, bu örneklemde "
        f"R² = {plain(s['r2'], 3)}. "
        "Çok dar aralıkta anakütle R²'si sıfıra yakındır ve örneklem R²'si onun çevresinde küçük dalgalanmalar "
        "gösterir. R² bu yüzden bir ilişkinin “gücünün” tek başına ölçüsü değildir; aynı eğim farklı örneklemlerde "
        "farklı R² verebilir. Dar aralıkta eğim tahmini de daha çok "
        "dalgalanır; bu belirsizlik Konu 7'de standart hatayla ölçülür."
    )


SPREAD = SimExperiment(
    topic_key=TOPIC,
    number=2,
    title="X'in yayılımı ve R²",
    question="Aynı ilişki, aynı gürültü; yalnız açıklayıcı değişkenin örneklemdeki aralığı farklı. R² değişir mi? R² "
             "bir ilişkinin gücünü mü ölçer, yoksa örneklemdeki değişkenliğin açıklanan payını mı?",
    note=NoteRef("4.5"),
    parameters=(
        SimParameter("genislik", "X'in aralığının genişliği w", 0.5, 20, 10, 0.5,
                     "X ~ U(5 − w/2, 5 + w/2); w = 10: Deney 1'deki U(0, 10).", decimals=1),
    ),
    dgp=_spread_dgp,
    dgp_note=(
        "Anakütle doğrusu ve hata terimi sabittir; kaydırıcı yalnız X'in çekildiği aralığı daraltır ya da genişletir. "
        "Çekilişler aynı tohumla yapılır: w değişince noktalar aynı sırayla ölçeklenir."
    ),
    look_at=(
        "**Grafik** — X'in aralığı daraldığında noktalar doğrunun çevresinde nasıl görünüyor?",
        "**R²** — w büyüdükçe örneklem ve anakütle R²'si.",
        "**Eğim** — w değişince tahmin edilen eğim.",
    ),
    build=_build_spread,
    metrics=_spread_metrics,
    takeaway=_spread_takeaway,
    labels=(("x", "X"), ("y", "Y"), ("u", "Hata terimi u")),
)


# --- Deney 3: fonksiyonel biçim --------------------------------------------------------------------

X_CENTER_3, Y_CENTER_3 = 2000, 300
"""Deney 3: konut büyüklüğü (kare fit) ve fiyat (bin dolar) ölçeğinde; ln X ~ N(ln 2000, 0,3²)."""
SIGMA_3 = 0.2


def _forms_settings(parameters: Parameters) -> tuple[float, int]:
    return _rounded(parameters, "esneklik", 2), int(parameters["n"])


def _forms_intercept(elasticity: float) -> float:
    """ln Y = β₀ + β₁·ln X; β₀, X = 2000 kare fitte ortalama log fiyat ln 300 olacak biçimde seçilir."""

    return round(math.log(Y_CENTER_3) - elasticity * math.log(X_CENTER_3), 6)


def _build_forms(parameters: Parameters) -> tuple:
    elasticity, n = _forms_settings(parameters)
    intercept = _forms_intercept(elasticity)
    return (
        NewSample("konutlar", n, SEED),
        Draw("konutlar", "ln_buyukluk", "normal", round(math.log(X_CENTER_3), 6), 0.3,
             "ln(büyüklük) ~ N(ln 2000; 0,3²)"),
        Draw("konutlar", "u", "normal", 0, SIGMA_3, "Hata terimi u ~ N(0; 0,2²)"),
        Derive("konutlar", "ln_fiyat", E.add(E.add(intercept, E.mul(elasticity, E.var("ln_buyukluk"))), E.var("u")),
               "ln(fiyat) = β₀ + β₁·ln(büyüklük) + u"),
        Derive("konutlar", "buyukluk", E.exp(E.var("ln_buyukluk")), "Büyüklük (kare fit)"),
        Derive("konutlar", "fiyat", E.exp(E.var("ln_fiyat")), "Fiyat (bin dolar)"),
        OLS("duzey", "konutlar", "fiyat", ("buyukluk",), "Düzey–düzey model: fiyat ~ büyüklük"),
        OLS("loglog", "konutlar", "ln_fiyat", ("ln_buyukluk",), "Log–log model: ln(fiyat) ~ ln(büyüklük)"),
        ModelValue("b_duzey", "duzey", "coef", "Düzey–düzey eğim (bin dolar / kare fit)", term="buyukluk", decimals=4),
        ModelValue("b_log", "loglog", "coef", "Log–log eğim (esneklik)", term="ln_buyukluk", decimals=3),
        ModelValue("r2_duzey", "duzey", "r2", "Düzey–düzey R² (fiyatın değişkenliği)", decimals=3),
        ModelValue("r2_log", "loglog", "r2", "Log–log R² (log fiyatın değişkenliği)", decimals=3),
        ScatterPlot("konutlar", "buyukluk", "fiyat", "Konut büyüklüğü (kare fit)", "Konut fiyatı (bin dolar)",
                    "Düzeyler: düzey–düzey doğrusu", fit_line=True, size=7, opacity=0.6),
        ScatterPlot("konutlar", "ln_buyukluk", "ln_fiyat", "ln(büyüklük)", "ln(fiyat)",
                    "Logaritmalar: log–log doğrusu ve anakütle doğrusu", fit_line=True, size=7, opacity=0.6,
                    lines=((intercept, elasticity, "Anakütle doğrusu: β₀ + β₁·ln(büyüklük)"),)),
    )


def _forms_dgp(parameters: Parameters) -> tuple[str, ...]:
    elasticity, n = _forms_settings(parameters)
    intercept = _forms_intercept(elasticity)
    return (
        r"\ln(\text{fiyat}_i) = \beta_0 + \beta_1 \ln(\text{büyüklük}_i) + u_i, \qquad \beta_1 = "
        rf"{_tex(elasticity)}, \quad \beta_0 = {_tex(intercept, 3)}, \qquad n = {n}",
        r"\ln(\text{büyüklük}_i) \sim N(\ln 2000;\ 0{,}3^2), \qquad u_i \sim N(0;\ 0{,}2^2)",
    )


def _forms_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    elasticity, _ = _forms_settings(parameters)
    s = state.scalars
    return (
        SimMetric("Gerçek esneklik β₁", plain(elasticity, 2), "Veri üretim sürecindeki log–log eğim."),
        SimMetric("Log–log β̂₁", plain(s["b_log"], 3), "Log–log modelin eğimi: esneklik tahmini."),
        SimMetric("Düzey–düzey eğim", plain(s["b_duzey"], 4),
                  "Bir kare fitlik farka karşılık gelen tahmini fiyat farkı (bin dolar)."),
        SimMetric("R²: düzey / log", f"{plain(s['r2_duzey'], 2)}/{plain(s['r2_log'], 2)}",
                  "Farklı bağımlı değişkenlerin R²'leri: doğrudan karşılaştırılmaz (§4.6)."),
    )


def _forms_takeaway(state: LabState, parameters: Parameters) -> str:
    elasticity, _ = _forms_settings(parameters)
    s = state.scalars
    shape = ("DGP'de β₁ < 1: ortalama fiyat büyüklükle birlikte artar ama giderek daha yavaş; bir kare fitlik farka "
             "karşılık gelen fiyat farkı büyük konutlarda daha küçüktür. Düz doğru bu değişen farkı tek bir sayıyla "
             "özetler." if elasticity < 1 else
             "DGP'de β₁ > 1: ortalama fiyat büyüklükten daha hızlı artar; bir kare fitlik farka karşılık gelen fiyat "
             "farkı büyük konutlarda daha büyüktür. Düz doğru bu değişen farkı tek bir sayıyla özetler."
             if elasticity > 1 else
             "DGP'de β₁ = 1: ortalama fiyat büyüklükle orantılıdır; düzeydeki ilişki de ortalamada bir doğrudur ve bir "
             "kare fitlik farka karşılık gelen fiyat farkı her büyüklükte aynıdır.")
    b_log = s["b_log"]
    return (
        f"Log–log modelin eğimi esnekliği tahmin eder: β̂₁ = {plain(b_log, 3)}, DGP'deki esneklik "
        f"{plain(elasticity, 2)}. Bu örneklemde büyüklüğü yüzde 1 daha yüksek konutların tahmin edilen fiyatı "
        f"yaklaşık yüzde {plain(abs(b_log), 3)} daha {signed_difference(b_log)} (§4.9). Düzey–düzey eğim "
        f"({plain(s['b_duzey'], 4)} bin dolar / kare fit) başka bir soruya cevap verir: bir kare fitlik farka "
        f"karşılık gelen ortalama tahmini fiyat farkı. {shape} R²'ler ({plain(s['r2_duzey'], 3)} ve "
        f"{plain(s['r2_log'], 3)}) farklı bağımlı değişkenlerin değişkenliğini ölçtüğü için hangisinin “daha iyi” "
        "olduğunu göstermez (§4.6). Biçim seçimi araştırma sorusuna ve katsayının yorumuna göre yapılır."
    )


FORMS = SimExperiment(
    topic_key=TOPIC,
    number=3,
    title="Fonksiyonel biçim: log–log ilişkide iki model",
    question="Gerçek ilişki log–log (sabit esneklik) ise düzey–düzey ve log–log modeller neyi tahmin eder? R²'si daha "
             "yüksek olan model daha mı doğrudur?",
    note=NoteRef("4.9", objects=("§4.6",)),
    parameters=(
        SimParameter("esneklik", "Gerçek esneklik β₁", 0.2, 1.6, 0.8, 0.1,
                     "HPRICE1'de log–log eğim 0,8727 (Denklem 4.9).", decimals=2),
        SimParameter("n", "Konut sayısı n", 30, 500, 88, 2, "HPRICE1'de 88 konut.", integer=True, decimals=0),
    ),
    dgp=_forms_dgp,
    dgp_note=(
        "Büyüklük ve fiyat HPRICE1'e benzer ölçektedir (kare fit, bin dolar). β₀, 2000 kare fitlik bir konutun "
        "ortalama log fiyatı ln 300 olacak biçimde esnekliğe göre seçilir. İki model aynı veriye kurulur."
    ),
    look_at=(
        "**Düzey grafiği** — düz doğru eğrisel ilişkiyi nasıl özetliyor?",
        "**Log grafiği** — log–log doğrusu anakütle doğrusuna ne kadar yakın?",
        "**Metrikler** — esneklik tahmini ve iki R².",
    ),
    build=_build_forms,
    metrics=_forms_metrics,
    takeaway=_forms_takeaway,
    labels=(("buyukluk", "Konut büyüklüğü (kare fit)"), ("fiyat", "Konut fiyatı (bin dolar)"),
            ("ln_buyukluk", "ln(büyüklük)"), ("ln_fiyat", "ln(fiyat)"), ("u", "Hata terimi u")),
)


KONU04_EXPERIMENTS = (NOISE, SPREAD, FORMS)
