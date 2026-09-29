"""Konu 10 Sezgi deneyleri: veri üretim süreci bilinen kontrollü simülasyonlar.

Deney 1  Kategorileri 1, 2, 3 diye kodlamak ile kukla kullanmak           (Notlar §10.1)
Deney 2  Ham fark ve kontrollü fark                                       (Notlar §10.3)
Deney 3  Log modelinde kukla: yaklaşık ve tam yüzde fark                  (Notlar §10.7)

Bölüm 10'da benzetim yoktur; deneylerin notlarda sayısal karşılığı yoktur. Varsayılan ayarlar bölümün kavramlarını
göstermek için seçilmiştir; her deney tohum 305 ile başlar.
"""

from __future__ import annotations

import math

from core.labs import expr as E
from core.labs.runner import LabState
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, plain
from core.labs.spec import (
    OLS,
    BarChart,
    Derive,
    Draw,
    DrawDiscrete,
    Event,
    GroupSummary,
    Histogram,
    JoinColumns,
    LineChart,
    ModelValue,
    MonteCarlo,
    NewSample,
    NoteRef,
    Scalar,
    ScalarTable,
    Statistic,
    Support,
)

SEED = 305
TOPIC = "konu10"


def _rounded(parameters: Parameters, key: str, decimals: int) -> float:
    return round(float(parameters[key]), decimals)


def _tex(value: float, decimals: int = 2) -> str:
    text = f"{value:.{decimals}f}"
    text = text.rstrip("0").rstrip(".") if "." in text else text  # 20 → "2" olmasın
    return "0" if text in ("-0", "") else text.replace(".", "{,}")


def _short(value: float, decimals: int = 2) -> str:
    text = f"{value:.{decimals}f}"
    text = text.rstrip("0").rstrip(".") if "." in text else text  # 20 → "2" olmasın
    return "0" if text in ("-0", "0", "") else text.replace(".", ",").replace("-", "−")


def _thousands(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _plus(value: float, term: str) -> str:
    """Düz metinde işaretli terim: "+ 2·D", "− 3·D" (katsayı negatifse "+ −3" yazılmaz)."""

    return f"{'−' if value < 0 else '+'} {_short(abs(value))}·{term}"


def _tex_plus(value: float, term: str) -> str:
    """LaTeX'te işaretli terim: "+ 2\\,D", "- 3\\,D"."""

    return f"{'-' if value < 0 else '+'} {_tex(abs(value))}\\,{term}"


# --- Deney 1: 1, 2, 3 kodlaması ve kuklalar -------------------------------------------------------------------------

BASE_1 = 10.0
SHARES_1 = (0.3, 0.4, 0.3)
"""Üç kategorinin payları; Y = 10 + a·D₂ + b·D₃ + u."""


def _coding_settings(parameters: Parameters) -> tuple[float, float, int]:
    return _rounded(parameters, "a", 2), _rounded(parameters, "b", 2), int(parameters["n"])


def _build_coding(parameters: Parameters) -> tuple:
    a, b, n = _coding_settings(parameters)
    mean = E.add(E.add(BASE_1, E.mul(a, E.var("d2"))), E.mul(b, E.var("d3")))
    return (
        NewSample("veri", n, SEED),
        DrawDiscrete("veri", "kod", (1, 2, 3), SHARES_1, "Kategori kodu: 1, 2, 3 (payları 0,3; 0,4; 0,3)"),
        Event("veri", "d2", "kod", (2,), "Kategori 2 kuklası"),
        Event("veri", "d3", "kod", (3,), "Kategori 3 kuklası"),
        Draw("veri", "u", "normal", 0, 1, "u ~ N(0, 1)"),
        Derive("veri", "y", E.add(mean, E.var("u")), f"Y = 10 {_plus(a, 'D₂')} {_plus(b, 'D₃')} + u"),
        GroupSummary("veri", "kod", (("n", "y", "count"), ("ortalama", "y", "mean")), "gruplar", (1, 2, 3),
                     decimals=3, labels=((1, "Kategori 1"), (2, "Kategori 2"), (3, "Kategori 3")),
                     heading="Kategori"),
        OLS("sayisal", "veri", "y", ("kod",), "Kodu sayı gibi kullanmak: y ~ kod"),
        OLS("kukla", "veri", "y", ("d2", "d3"), "Kuklalar (kategori 1 referans): y ~ d2 + d3"),
        ModelValue("egim", "sayisal", "coef", "Sayısal kod modelinin eğimi", term="kod", decimals=3),
        ModelValue("k2", "kukla", "coef", "Kukla modeli: kategori 2 − kategori 1", term="d2", decimals=3),
        ModelValue("k3", "kukla", "coef", "Kukla modeli: kategori 3 − kategori 1", term="d3", decimals=3),
        ModelValue("r2_sayisal", "sayisal", "r2", "Sayısal kod modeli: R²", decimals=3),
        ModelValue("r2_kukla", "kukla", "r2", "Kukla modeli: R²", decimals=3),
        Scalar("sayisal_31", E.mul(2, E.ref("egim")),
               "Sayısal kod modelinin ima ettiği kategori 3 − kategori 1 farkı: 2 × eğim", decimals=3),
        ScalarTable((("Kategori 2 − kategori 1", E.const(a)), ("Kategori 3 − kategori 1", E.const(b))), "gercek",
                    decimals=3, heading="Fark"),
        ScalarTable((("Kategori 2 − kategori 1", E.ref("egim")), ("Kategori 3 − kategori 1", E.ref("sayisal_31"))),
                    "sayisal_fark", decimals=3, heading="Fark"),
        ScalarTable((("Kategori 2 − kategori 1", E.ref("k2")), ("Kategori 3 − kategori 1", E.ref("k3"))),
                    "kukla_fark", decimals=3, heading="Fark"),
        JoinColumns("farklar", (("Gerçek fark", "gercek", "deger"), ("1, 2, 3 kodlu model", "sayisal_fark", "deger"),
                                ("Kukla modeli", "kukla_fark", "deger")), decimals=3, heading="Fark"),
        BarChart("gruplar", "ortalama", "Kategori", "Y'nin ortalaması", "Kategorilere göre Y'nin ortalaması",
                 decimals=2),
    )


def _coding_dgp(parameters: Parameters) -> tuple[str, ...]:
    a, b, n = _coding_settings(parameters)
    return (
        rf"Y_i = 10 {_tex_plus(a, 'D_{2i}')} {_tex_plus(b, 'D_{3i}')} + u_i, \qquad u_i \sim N(0,\ 1), \qquad n = {n}",
        r"\text{kategori} \in \{1,\ 2,\ 3\}, \quad \text{payları } 0{,}3;\ 0{,}4;\ 0{,}3",
        r"D_{2i} = 1\{\text{kategori} = 2\}, \quad D_{3i} = 1\{\text{kategori} = 3\}",
        r"\text{Sayısal kod: } Y_i = \alpha_0 + \alpha_1\,\text{kod}_i + v_i \qquad \text{Kukla: } "
        r"Y_i = \delta_0 + \delta_2 D_{2i} + \delta_3 D_{3i} + u_i",
    )


def _coding_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    s = state.scalars
    return (
        SimMetric("Sayısal kod eğimi", plain(s["egim"], 3), "Kod bir artınca Y'nin tahmin edilen değişimi; her adım "
                  "aynı kabul edilir."),
        SimMetric("δ̂₂ (kukla)", plain(s["k2"], 3), "Kategori 2 ile kategori 1 arasındaki tahmin edilen fark."),
        SimMetric("δ̂₃ (kukla)", plain(s["k3"], 3), "Kategori 3 ile kategori 1 arasındaki tahmin edilen fark."),
        SimMetric("R²: kod / kukla", f"{plain(s['r2_sayisal'], 3)} / {plain(s['r2_kukla'], 3)}",
                  "Sayısal kod modeli ile kukla modelinin R²'si."),
    )


def _coding_takeaway(state: LabState, parameters: Parameters) -> str:
    a, b, _ = _coding_settings(parameters)
    s = state.scalars
    text = (f"Sayısal kod modeli kategoriler arasında eşit adımlar varsayar: kategori 2 − kategori 1 farkı "
            f"{plain(s['egim'], 3)} ise kategori 3 − kategori 1 farkı bunun iki katıdır ({plain(s['sayisal_31'], 3)}). "
            f"Kukla modeli iki farkı ayrı ayrı tahmin eder: {plain(s['k2'], 3)} ve {plain(s['k3'], 3)} (gerçek "
            f"değerler {_short(a)} ve {_short(b)}). ")
    if abs(b - 2 * a) < 1e-9:
        text += ("Bu ayarda gerçek farklar eşit adımlıdır (b = 2a); iki model yakın sonuç verir. Bu bir rastlantıdır; "
                 "kategori kodlarının sırası ve aralarındaki uzaklık genellikle bir anlam taşımaz. ")
    else:
        text += ("Gerçek farklar eşit adımlı olmadığı için sayısal kod modeli farkları bozar; R²'si de kukla modelininkinden "
                 "düşüktür. ")
    return text + ("1, 2, 3 yalnızca etiket görevi görür; kodu tek bir sayısal açıklayıcı değişken gibi kullanmak "
                   "kategoriler arasında yapay bir sıralama ve eşit uzaklık varsayımı yaratır. Her kategori için bir "
                   "kukla tanımlanır ve bir kategori referans bırakılır (§10.1).")


CODING = SimExperiment(
    topic_key=TOPIC,
    number=1,
    title="1, 2, 3 kodlaması ile kukla değişkenler",
    question="Üç kategorili bir değişkeni 1, 2, 3 diye kodlayıp tek bir sayısal değişken gibi kullanmak ile her "
             "kategori için kukla kullanmak aynı sonucu verir mi?",
    note=NoteRef("10.1"),
    parameters=(
        SimParameter("a", "Kategori 2 ile kategori 1 farkı a", -3.0, 3.0, 2.0, 0.25, "Gerçek fark δ₂ = a.",
                     decimals=2),
        SimParameter("b", "Kategori 3 ile kategori 1 farkı b", -3.0, 3.0, 1.0, 0.25, "Gerçek fark δ₃ = b.",
                     decimals=2),
        SimParameter("n", "Örneklem büyüklüğü n", 150, 1500, 600, 150, "Gözlem sayısı.", integer=True, decimals=0),
    ),
    dgp=_coding_dgp,
    dgp_note=(
        "Notlarda bu deneyin sayısal karşılığı yoktur. Tek örneklem, tohum 305; önce kategori, sonra u çekilir. "
        "Kategori 1 referanstır. Varsayılan ayarda ortadaki kategori en yüksek ortalamaya sahiptir: kodların ima ettiği "
        "sıralama ve eşit uzaklık varsayımı açıkça yanlıştır."
    ),
    look_at=(
        "**Tablo** — sayısal kod modelinin ima ettiği farklar gerçek farklara yakın mı? Kukla modeli farkları yakalıyor mu?",
        "**Grafik** — kategori ortalamaları doğrusal bir sıra izliyor mu?",
        "**a ve b** — b = 2a yapın: iki model ne zaman aynı sonucu verir?",
    ),
    build=_build_coding,
    metrics=_coding_metrics,
    takeaway=_coding_takeaway,
    tables=(("farklar", "Gerçek farklar ve iki modelin tahminleri"), ("gruplar", "Kategori ortalamaları")),
    labels=(("kod", "Kategori kodu"), ("d2", "D₂"), ("d3", "D₃"), ("u", "u"), ("y", "Y"), ("n", "Gözlem sayısı"),
            ("ortalama", "Y'nin ortalaması")),
)


# --- Deney 2: ham fark ve kontrollü fark ------------------------------------------------------------------------------

REPS_2 = 1000
SLOPE_2 = 0.5
"""Y = 5 + δD + 0,5X + u; X = 12 + Δ·D + e: iki grup X bakımından Δ kadar farklıdır."""


def _raw_settings(parameters: Parameters) -> tuple[float, float, int]:
    return _rounded(parameters, "delta", 2), _rounded(parameters, "gap", 2), int(parameters["n"])


def _build_raw(parameters: Parameters) -> tuple:
    delta, gap, n = _raw_settings(parameters)
    body = (
        NewSample("orneklem", n, None),
        Draw("orneklem", "v", "uniform", 0, 1, "v ~ U(0, 1)"),
        Derive("orneklem", "d", E.compare("lt", E.var("v"), 0.5), "D = 1{v < 0,5}: gruplar yarı yarıya"),
        Draw("orneklem", "e", "normal", 0, 2, "e ~ N(0, 2²)"),
        Derive("orneklem", "x", E.add(E.add(12, E.mul(gap, E.var("d"))), E.var("e")),
               f"X = 12 {_plus(gap, 'D')} + e: D = 1 grubunun X ortalaması Δ = {_short(gap)} kadar farklı"),
        Draw("orneklem", "u", "normal", 0, 1, "u ~ N(0, 1)"),
        Derive("orneklem", "y", E.add(E.add(E.add(5, E.mul(delta, E.var("d"))), E.mul(SLOPE_2, E.var("x"))), E.var("u")),
               f"Y = 5 {_plus(delta, 'D')} + 0,5X + u"),
        OLS("ham", "orneklem", "y", ("d",), "Ham fark: y ~ d"),
        OLS("kontrollu", "orneklem", "y", ("d", "x"), "Kontrollü fark: y ~ d + x"),
        ModelValue("d_ham", "ham", "coef", "Ham farkın tahmini", term="d"),
        ModelValue("d_kontrol", "kontrollu", "coef", "Kontrollü farkın tahmini", term="d"),
    )
    theory = round(delta + SLOPE_2 * gap, 10)
    return (
        MonteCarlo("tekrarlar", REPS_2, SEED, body, (("ham", E.ref("d_ham")), ("kontrollu", E.ref("d_kontrol"))),
                   f"{_thousands(REPS_2)} örneklem, her birinde {n} gözlem: ham ve kontrollü kukla katsayısı"),
        Statistic("tekrarlar", "ham", "mean", "ort_ham", "Ham farkın tekrarlar arasında ortalaması", decimals=3),
        Statistic("tekrarlar", "kontrollu", "mean", "ort_kontrol", "Kontrollü farkın tekrarlar arasında ortalaması",
                  decimals=3),
        Statistic("tekrarlar", "ham", "std", "ss_ham", "Ham farkın standart sapması", decimals=3),
        Statistic("tekrarlar", "kontrollu", "std", "ss_kontrol", "Kontrollü farkın standart sapması", decimals=3),
        Scalar("kuram_ham", E.const(theory), "Ham farkın beklenen değeri: δ + 0,5·Δ", decimals=3),
        Histogram("tekrarlar", (("ham", "Ham fark (y ~ d)"), ("kontrollu", "Kontrollü fark (y ~ d + x)")), 50,
                  min(delta, theory) - 1.5, max(delta, theory) + 1.5,
                  f"Kukla katsayısının dağılımı (δ = {_short(delta)}, Δ = {_short(gap)})", "Kukla katsayısının tahmini",
                  references=((delta, f"Gerçek δ = {_short(delta)}"), (theory, f"δ + 0,5·Δ = {_short(theory)}"))),
    )


def _raw_dgp(parameters: Parameters) -> tuple[str, ...]:
    delta, gap, n = _raw_settings(parameters)
    return (
        rf"Y_i = 5 + \delta D_i + 0{{,}}5\,X_i + u_i, \qquad \delta = {_tex(delta)}, \qquad u_i \sim N(0,\ 1)",
        rf"X_i = 12 + \Delta D_i + e_i, \qquad \Delta = {_tex(gap)}, \qquad e_i \sim N(0,\ 2^2), \qquad "
        rf"P(D_i = 1) = 0{{,}}5, \qquad n = {n}",
        r"\text{Koşullu fark (X sabit): } \delta = \mathbb{E}(Y \mid D = 1, X) - \mathbb{E}(Y \mid D = 0, X)",
        r"\text{Ham fark: } \mathbb{E}(Y \mid D = 1) - \mathbb{E}(Y \mid D = 0) = \delta + 0{,}5\,\Delta",
    )


def _raw_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    delta, _, _ = _raw_settings(parameters)
    s = state.scalars
    return (
        SimMetric("Gerçek δ", _short(delta), "X sabitken iki grup arasındaki gerçek fark."),
        SimMetric("Ham fark (ort.)", plain(s["ort_ham"], 3), "y ~ d modelinde kukla katsayısının ortalaması."),
        SimMetric("δ + 0,5Δ", plain(s["kuram_ham"], 3), "Ham farkın beklenen değeri."),
        SimMetric("Kontrollü (ort.)", plain(s["ort_kontrol"], 3), "y ~ d + x modelinde kukla katsayısının ortalaması."),
    )


def _raw_takeaway(state: LabState, parameters: Parameters) -> str:
    delta, gap, _ = _raw_settings(parameters)
    s = state.scalars
    ham, kontrol = plain(s["ort_ham"], 3), plain(s["ort_kontrol"], 3)
    if gap == 0:
        text = (f"Δ = 0 iken iki grubun X ortalaması aynıdır: ham farkın tekrarlar boyunca ortalaması {ham}, kontrollü "
                f"farkınki {kontrol}; ikisi de gerçek δ = {_short(delta)} çevresindedir. Gruplar kontrol değişkeni "
                "bakımından farklı değilse kontrol eklemek kukla katsayısının beklenen değerini değiştirmez. ")
    else:
        side = "yüksektir" if gap > 0 else "düşüktür"
        text = (f"Ham farkın tekrarlar boyunca ortalaması {ham}: gerçek δ = {_short(delta)} değil, δ + 0,5Δ = "
                f"{plain(s['kuram_ham'], 3)} çevresindedir. Çünkü D = 1 grubunun X ortalaması öteki gruptan "
                f"{_short(abs(gap))} birim {side} ve X de Y'yi etkiler (eğim 0,5). X kontrol edilince kukla katsayısının "
                f"ortalaması {kontrol}: gerçek δ'ya yakındır. ")
        theory = delta + SLOPE_2 * gap
        if abs(theory) < 1e-9:
            text += ("Bu ayarda ham farkın beklenen değeri sıfırdır: X sabitken var olan fark, gruplar arasındaki X "
                     "farkıyla tam olarak dengelenir. ")
        elif delta == 0:
            text += "Bu ayarda X sabitken grup farkı yoktur; ham farkın tamamı X'teki farktan gelir. "
        elif theory * delta < 0:
            text += "Bu ayarda ham fark, X sabitken var olan farkın işaretini bile tersine çevirir. "
    return text + ("Kontrol eklemek yalnız modele alınan değişkenlerin farkını ayırır; modelde olmayan bir etmen "
                   "gruplar arasında farklıysa kontrollü katsayı da nedensel etki olarak okunamaz (§10.3).")


RAW = SimExperiment(
    topic_key=TOPIC,
    number=2,
    title="Ham fark ve kontrollü fark",
    question="İki grup, Y'yi etkileyen başka bir değişken bakımından da farklıysa ham grup farkı neyi ölçer? Kontrol "
             "değişkeni eklenince kukla katsayısı neye yaklaşır?",
    note=NoteRef("10.3"),
    parameters=(
        SimParameter("delta", "Gerçek koşullu fark δ", -2.0, 2.0, -1.0, 0.25, "X sabitken D = 1 grubunun farkı.",
                     decimals=2),
        SimParameter("gap", "Gruplar arasındaki X farkı Δ", -4.0, 4.0, 2.0, 0.5,
                     "D = 1 grubunun X ortalaması Δ kadar farklıdır.", decimals=2),
        SimParameter("n", "Örneklem büyüklüğü n", 100, 1000, 300, 100, "Her örneklemdeki gözlem sayısı.", integer=True,
                     decimals=0),
    ),
    dgp=_raw_dgp,
    dgp_note=(
        "Notlarda bu deneyin sayısal karşılığı yoktur. δ koşullu farktır: X sabitken D = 1 ve D = 0 gruplarının Y "
        "ortalamaları arasındaki fark. Tohum 305; her tekrarda sırasıyla v, e ve u çekilir. D = 1{v < 0,5}. Varsayılan "
        "ayarda δ = −1 ve Δ = 2: ham farkın beklenen değeri sıfırdır, oysa X sabitken gerçek fark −1'dir."
    ),
    look_at=(
        "**Histogram** — ham fark ve kontrollü fark hangi değerlerin çevresinde toplanıyor?",
        "**Δ** — gruplar arasındaki X farkını sıfırlayın: ham fark ile kontrollü fark birbirine yaklaşıyor mu?",
        "**δ** — gerçek fark değişince iki tahmin nasıl kayıyor?",
    ),
    build=_build_raw,
    metrics=_raw_metrics,
    takeaway=_raw_takeaway,
    labels=(("v", "v"), ("d", "D"), ("e", "e"), ("x", "X"), ("u", "u"), ("y", "Y"), ("ham", "Ham fark"),
            ("kontrollu", "Kontrollü fark")),
)


# --- Deney 3: log modelinde kukla, yaklaşık ve tam yüzde ---------------------------------------------------------

def _log_settings(parameters: Parameters) -> tuple[float, float, int]:
    return _rounded(parameters, "delta", 2), _rounded(parameters, "sigma", 2), int(parameters["n"])


def _build_log(parameters: Parameters) -> tuple:
    delta, sigma, n = _log_settings(parameters)
    return (
        NewSample("veri", n, SEED),
        Draw("veri", "v", "uniform", 0, 1, "v ~ U(0, 1)"),
        Derive("veri", "d", E.compare("lt", E.var("v"), 0.5), "D = 1{v < 0,5}"),
        Draw("veri", "u", "normal", 0, sigma, f"u ~ N(0, {_short(sigma)}²)"),
        Derive("veri", "lny", E.add(E.add(2, E.mul(delta, E.var("d"))), E.var("u")), f"ln(Y) = 2 {_plus(delta, 'D')} + u"),
        Derive("veri", "y", E.exp(E.var("lny")), "Y = exp(ln(Y))"),
        OLS("model", "veri", "lny", ("d",), "Log model: lny ~ d"),
        ModelValue("d_hat", "model", "coef", "Kukla katsayısının tahmini δ̂", term="d", decimals=4),
        Scalar("yaklasik", E.mul(100, E.ref("d_hat")), "Yaklaşık yüzde fark 100·δ̂", decimals=2, percent=True),
        Scalar("tam", E.mul(100, E.sub(E.exp(E.ref("d_hat")), 1)), "Tam yüzde fark 100·(exp(δ̂) − 1)", decimals=2,
               percent=True),
        Statistic("veri", "y", "mean", "ort_1", "D = 1 grubunda Y'nin ortalaması", where=("d", 1), decimals=3),
        Statistic("veri", "y", "mean", "ort_0", "D = 0 grubunda Y'nin ortalaması", where=("d", 0), decimals=3),
        Scalar("ortalama_fark", E.mul(100, E.sub(E.div(E.ref("ort_1"), E.ref("ort_0")), 1)),
               "Y ortalamalarının yüzde farkı 100·(ort₁/ort₀ − 1)", decimals=2, percent=True),
        Scalar("gercek_tam", E.const(round(100 * math.expm1(delta), 10)), "Gerçek tam yüzde fark 100·(exp(δ) − 1)",
               decimals=2, percent=True),
        Support("izgara", "k", -100, 100, "Izgara: k = −100, …, 100"),
        Derive("izgara", "delta", E.div(E.var("k"), 100), "δ = k/100: −1'den 1'e"),
        Derive("izgara", "tam", E.mul(100, E.sub(E.exp(E.var("delta")), 1)), "Tam yüzde: 100·(exp(δ) − 1)"),
        Derive("izgara", "yaklasik", E.mul(100, E.var("delta")), "Yaklaşık yüzde: 100·δ"),
        LineChart("izgara", "delta", "tam", "Kukla katsayısı δ", "Yüzde fark",
                  "Yaklaşık ve tam yüzde fark: katsayı büyüdükçe ayrışır", markers=False,
                  series=(("yaklasik", "Yaklaşık: 100·δ"),), legend="Tam: 100·(exp(δ) − 1)"),
    )


def _log_dgp(parameters: Parameters) -> tuple[str, ...]:
    delta, sigma, n = _log_settings(parameters)
    return (
        rf"\ln(Y_i) = 2 + \delta D_i + u_i, \qquad \delta = {_tex(delta)}, \qquad u_i \sim N(0,\ {_tex(sigma)}^2), "
        rf"\qquad P(D_i = 1) = 0{{,}}5, \qquad n = {n}",
        r"\frac{\mathbb{E}(Y \mid D = 1)}{\mathbb{E}(Y \mid D = 0)} - 1 = e^{\delta} - 1 \quad "
        r"\text{(iki grupta hata dağılımı aynı)}",
    )


def _log_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    s = state.scalars
    return (
        SimMetric("Yaklaşık: 100·δ̂", "%" + plain(s["yaklasik"], 2), "Küçük katsayılar için yaklaşık yüzde fark."),
        SimMetric("Tam: 100(e^δ̂ − 1)", "%" + plain(s["tam"], 2), "Kuklanın 0'dan 1'e değişiminin tam yüzde karşılığı."),
        SimMetric("Ortalamalar farkı", "%" + plain(s["ortalama_fark"], 2),
                  "Veride iki grubun Y ortalamaları arasındaki yüzde fark."),
        SimMetric("Gerçek tam fark", "%" + plain(s["gercek_tam"], 2), "100·(exp(δ) − 1)."),
    )


def _log_takeaway(state: LabState, parameters: Parameters) -> str:
    delta, _, _ = _log_settings(parameters)
    s = state.scalars
    if delta == 0:
        return ("δ = 0 iken iki grubun Y dağılımı aynıdır: iki formül de sıfır verir; tahminlerin sıfırdan farkı "
                f"yalnız örnekleme gürültüsüdür (δ̂ = {plain(s['d_hat'], 4)}). Katsayı sıfırdan uzaklaştıkça 100·δ "
                "yaklaşımı ile 100·(exp(δ) − 1) arasındaki fark artar (§10.7).")
    gap = 100 * (math.expm1(delta) - delta)
    text = (f"δ̂ = {plain(s['d_hat'], 4)}: yaklaşık yorum %{plain(s['yaklasik'], 2)}, tam yorum %{plain(s['tam'], 2)}. "
            f"Verideki iki grubun Y ortalamaları arasındaki yüzde fark %{plain(s['ortalama_fark'], 2)}; DGP'deki "
            f"gerçek tam fark %{plain(s['gercek_tam'], 2)}. Tam yorum ile ortalamaların yüzde farkı aynı büyüklüğü, "
            "Y'nin iki grup arasındaki gerçek yüzde farkını tahmin eder; 100·δ̂ ise bir yaklaşımdır. ")
    if abs(delta) <= 0.1:
        text += (f"Bu ayarda δ küçük olduğundan iki formül arasındaki kuramsal fark yalnız {plain(abs(gap), 2)} yüzde "
                 "puandır; yaklaşık yorum yeterlidir. ")
    else:
        direction = "abartır" if delta < 0 else "küçük gösterir"
        text += (f"Bu ayarda iki formül arasındaki kuramsal fark {plain(abs(gap), 2)} yüzde puandır: 100·δ yaklaşımı "
                 f"farkı {direction}. ")
    return text + ("Kukla 0'dan 1'e tam bir birim değiştiği için katsayının mutlak değeri büyüdükçe iki formül "
                   "arasındaki fark artar (§10.7).")


LOG = SimExperiment(
    topic_key=TOPIC,
    number=3,
    title="Log modelinde kukla: yaklaşık ve tam yüzde",
    question="Log bağımlı değişkenli modelde kukla katsayısı δ ise iki grup arasındaki yüzde fark 100·δ mı, "
             "100·(exp(δ) − 1) mi? Verideki gerçek yüzde fark hangisine yakındır?",
    note=NoteRef("10.7"),
    parameters=(
        SimParameter("delta", "Kukla katsayısı δ", -1.0, 1.0, -0.3, 0.05, "Notlardaki kadın katsayısı −0,30 civarındadır.",
                     decimals=2),
        SimParameter("sigma", "Hata standart sapması σ", 0.2, 1.0, 0.5, 0.1, "ln(Y)'deki gürültü.", decimals=2),
        SimParameter("n", "Örneklem büyüklüğü n", 500, 10000, 4000, 500, "Gözlem sayısı.", integer=True, decimals=0),
    ),
    dgp=_log_dgp,
    dgp_note=(
        "Notlarda bu deneyin sayısal karşılığı yoktur. Tek örneklem, tohum 305; sırasıyla v ve u çekilir. İki grupta "
        "hata dağılımı aynı olduğu için Y ortalamalarının oranı exp(δ)'dır; bu yüzden tam yüzde fark ortalamalar "
        "arasındaki yüzde farktır. Grafik iki formülü δ'nın −1 ile 1 arasındaki değerleri için karşılaştırır."
    ),
    look_at=(
        "**Ölçüler** — verideki ortalamalar farkı hangi yoruma yakın?",
        "**Grafik** — iki eğri δ = 0 çevresinde neredeyse aynı; δ büyüdükçe ne oluyor?",
        "**δ** — katsayıyı −0,8 ve 0,8 yapın: yaklaşık yorumun hatası hangi yönde?",
    ),
    build=_build_log,
    metrics=_log_metrics,
    takeaway=_log_takeaway,
    labels=(("v", "v"), ("d", "D"), ("u", "u"), ("lny", "ln(Y)"), ("y", "Y"), ("k", "k"), ("delta", "δ"),
            ("tam", "Tam yüzde"), ("yaklasik", "Yaklaşık yüzde")),
)


KONU10_EXPERIMENTS = (CODING, RAW, LOG)
