"""Konu 3 Sezgi deneyleri: veri üretim süreci bilinen kontrollü simülasyonlar.

Deney 1  Örneklemden örneklemeye değişen doğru                       (Notlar §3.7)
Deney 2  EKK doğrusunu elle aramak: kareli artıklar toplamı         (Notlar §3.8)
Deney 3  Hata terimi ve artık aynı şey değildir                      (Notlar §3.5, §3.7)

Bölüm 3'te simülasyonla üretilmiş bir tablo ya da şekil yoktur; deneyler bölümün kavramlarını bilinen bir anakütle
doğrusuyla gösterir. Deney 1 ve 3'teki anakütle doğrusu E(Y | X) = −1 + 0,5·X, WAGE1'deki eğitim–ücret doğrusuna
(Denklem 3.11) yakın ölçektedir. Standart hata, yansızlık ve varyans Konu 6–7'nin konusudur; deneyler yalnız tahminin
örneklemden örneklemeye değiştiğini, EKK ölçütünü ve hata terimi–artık ayrımını gösterir. Tohum 305.
"""

from __future__ import annotations

from core.labs import expr as E
from core.labs.runner import LabState
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, percent, plain
from core.labs.spec import (
    INTERCEPT,
    OLS,
    Derive,
    Draw,
    Histogram,
    ModelValue,
    MonteCarlo,
    NewSample,
    NoteRef,
    PairStatistic,
    Scalar,
    ScatterPlot,
    ShowFrame,
    Statistic,
)

SEED = 305
TOPIC = "konu03"
BETA0, BETA1 = -1.0, 0.5
"""Deney 1 ve 3'teki anakütle doğrusu: E(Y | X) = −1 + 0,5·X."""
X_LOW, X_HIGH = 8, 18
"""Açıklayıcı değişkenin aralığı: X ~ U(8, 18) (eğitim yılına benzer)."""


def _rounded(parameters: Parameters, key: str, decimals: int) -> float:
    """Kaydırıcı değeri, kodda 0,30000000000000004 gibi yazımlar olmasın diye yuvarlanır."""

    return round(float(parameters[key]), decimals)


def _tex(value: float, decimals: int = 2) -> str:
    """LaTeX içinde sondaki sıfırları atılmış sayı (0{,}5; 3)."""

    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text.replace(".", "{,}")


def _short(value: float, decimals: int = 2) -> str:
    """Sondaki sıfırları atılmış düz metin sayı (0,5; 1,5; 3); yuvarlanınca sıfır olan değer "0" yazılır."""

    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "0", "") else text.replace(".", ",").replace("-", "−")


def _signed(value: float, decimals: int = 2) -> str:
    """Toplamdaki terim: "+ 0,5" ya da "− 0,5" (“+ −0,5” yazılmaz)."""

    return f"{'−' if value < 0 else '+'} {_short(abs(value), decimals)}"


def _population_draws(frame: str, sigma: float, *, x_comment: str = "Açıklayıcı değişken X ~ U(8, 18)") -> tuple:
    """Anakütle modelinden bir örneklem: X, hata terimi u ve Y = −1 + 0,5·X + u (çekiliş sırası sabit)."""

    return (
        Draw(frame, "x", "uniform", X_LOW, X_HIGH, x_comment),
        Draw(frame, "u", "normal", 0, sigma, f"Hata terimi u ~ N(0, σ²), σ = {_short(sigma, 1)}"),
        Derive(frame, "y", E.add(E.add(BETA0, E.mul(BETA1, E.var("x"))), E.var("u")), "Y = −1 + 0,5·X + u"),
    )


# --- Deney 1: örneklemden örneklemeye değişen doğru ------------------------------------------------

REPS_1 = 500
BINS_1 = 40
CLOSE_1 = 0.1
"""Tekrar ve histogram kutu sayısı; "yakın" eşiği: |β̂₁ − β₁| < 0,1."""
SD_X = 10 / 12 ** 0.5
"""U(8, 18) dağılımının standart sapması: 10/√12."""


def lines_axis(sigma: float) -> tuple[float, float]:
    """Histogram ekseni σ'ya göre seçilir, n'ye göre değişmez (n büyüdükçe daralma görünsün): en küçük n = 10'da
    eğim tahminlerinin yaklaşık standart sapması σ / (√10 · s_X)'in beş katı; en az ±0,5. (Dört kat, n = 10'da
    500 tahminden birini eksen dışında bırakıyordu.)"""

    half = max(0.5, 5 * sigma / (10 ** 0.5 * SD_X))
    return round(BETA1 - half, 1), round(BETA1 + half, 1)


def _lines_settings(parameters: Parameters) -> tuple[int, float]:
    return int(parameters["n"]), _rounded(parameters, "sigma", 1)


def _build_lines(parameters: Parameters) -> tuple:
    n, sigma = _lines_settings(parameters)
    slope = E.div(E.ref("kov_xy"), E.ref("var_x"))
    return (
        NewSample("orneklem", n, SEED),
        *_population_draws("orneklem", sigma),
        OLS("model", "orneklem", "y", ("x",), "Bu örneklemin EKK doğrusu: y ~ x"),
        ModelValue("b0", "model", "coef", "Tahmin edilen sabit β̂₀", term=INTERCEPT, decimals=3),
        ModelValue("b1", "model", "coef", "Tahmin edilen eğim β̂₁", term="x", decimals=3),
        ScatterPlot("orneklem", "x", "y", "X", "Y", f"Tek bir örneklem (n = {n}): tahmin edilen ve anakütle doğrusu",
                    fit_line=True, size=8, opacity=0.7,
                    lines=((BETA0, BETA1, "Anakütle doğrusu E(Y | X) = −1 + 0,5·X"),)),
        MonteCarlo(
            "tekrarlar",
            REPS_1,
            SEED,
            (
                NewSample("tekrar_orneklem", n, None),
                *_population_draws("tekrar_orneklem", sigma),
                PairStatistic("tekrar_orneklem", "x", "y", "cov", "kov_xy", "Kovaryans s_xy"),
                Statistic("tekrar_orneklem", "x", "var", "var_x", "Varyans s_x²"),
            ),
            (
                ("b1", slope),
                ("yakin", E.compare("lt", E.absolute(E.sub(slope, BETA1)), CLOSE_1)),
            ),
            f"Aynı anakütleden {REPS_1} örneklem; her birinde eğim β̂₁ = s_xy / s_x² (Denklem 3.8)",
        ),
        Histogram("tekrarlar", (("b1", "Eğim tahminleri β̂₁"),), BINS_1, *lines_axis(sigma),
                  f"{REPS_1} örneklemin eğim tahmini (n = {n})", "Eğim tahmini β̂₁",
                  references=((BETA1, "β₁ = 0,5 (anakütle eğimi)"),), y_label="Örneklem sayısı"),
    )


def _lines_dgp(parameters: Parameters) -> tuple[str, ...]:
    n, sigma = _lines_settings(parameters)
    return (
        r"Y_i = \beta_0 + \beta_1 X_i + u_i, \qquad \beta_0 = -1, \quad \beta_1 = 0{,}5, "
        rf"\qquad i = 1, \dots, n = {n}",
        rf"X_i \sim U(8,\ 18), \qquad u_i \sim N(0,\ \sigma^2), \qquad \sigma = {_tex(sigma, 1)}",
        rf"\text{{Her örneklemde: }} \hat\beta_1 = s_{{xy}} / s_x^2 \quad ({REPS_1}\ \text{{örneklem}})",
    )


def _lines_summary(state: LabState) -> tuple[float, float, float]:
    table = state.tables["tekrarlar"]
    return float(table["b1"].min()), float(table["b1"].max()), float(table["yakin"].mean())


def _lines_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    low, high, close = _lines_summary(state)
    return (
        SimMetric("Anakütle eğimi β₁", plain(BETA1, 3), "Veri üretim sürecindeki eğim; gerçek hayatta bilinmez."),
        SimMetric("Bu örneklemde β̂₁", plain(state.scalars["b1"], 3), "Grafikteki örneklemin EKK eğimi."),
        SimMetric("En küçük β̂₁", plain(low, 3), f"{REPS_1} örneklemin eğimlerinin en küçüğü."),
        SimMetric("β₁'e 0,1'den yakın", percent(100 * close, 1),
                  f"|β̂₁ − 0,5| < 0,1 olan örneklemlerin payı; en büyük β̂₁ = {plain(high, 3)}."),
    )


def _lines_takeaway(state: LabState, parameters: Parameters) -> str:
    n, sigma = _lines_settings(parameters)
    low, high, close = _lines_summary(state)
    b1 = state.scalars["b1"]
    if sigma == 0:
        return (
            "σ = 0: hata terimi yok, bütün noktalar anakütle doğrusunun üzerindedir. Her örneklemde β̂₁ = 0,5 ve β̂₀ = "
            "−1 bulunur: tahmin edilen doğru anakütle doğrusuyla çakışır ve histogram tek bir çizgiye iner. "
            "Örneklemden örneklemeye değişim hata teriminden gelir (§3.7)."
        )
    return (
        f"Grafikteki örneklemin eğimi {plain(b1, 3)}; anakütle eğimi 0,5. Aynı anakütleden çekilen {REPS_1} "
        f"örneklemin eğimleri {plain(low, 3)} ile {plain(high, 3)} arasında değişir (grafikteki örneklem bunların "
        f"ilkidir). β̂₁ tek bir örneklemden hesaplanan sayıdır; başka bir örneklem başka bir değer verir, bu yüzden "
        f"tahmin ile anakütle parametresi aynı kavram değildir (§3.7). Örneklemlerin {percent(100 * close, 1)} "
        "kadarında β̂₁, 0,5'e 0,1'den yakındır. n'yi ve σ'yı değiştirip histogramın nasıl daralıp genişlediğine bakın; "
        "bu dağılımın özellikleri (yansızlık, varyans) Konu 6'da, standart hata Konu 7'de işlenir."
    )


LINES = SimExperiment(
    topic_key=TOPIC,
    number=1,
    title="Örneklemden örneklemeye değişen doğru",
    question="Anakütle doğrusu bilinmez; onu bir örneklemden tahmin ederiz. Aynı anakütleden başka bir örneklem "
             "çekseydik aynı doğruyu mu bulurduk? Örneklem büyüklüğü "
             "ve hata teriminin büyüklüğü bu değişimi nasıl etkiler?",
    note=NoteRef("3.7"),
    parameters=(
        SimParameter("n", "Örneklem büyüklüğü n", 10, 400, 30, 10, "Her örneklemdeki gözlem sayısı.", integer=True,
                     decimals=0),
        SimParameter("sigma", "Hata teriminin standart sapması σ", 0, 6, 3, 0.5,
                     "σ = 0: hata terimi yok. σ = 3, WAGE1'deki eğitim–ücret doğrusunun artıklarına yakın ölçektir.",
                     decimals=1),
    ),
    dgp=_lines_dgp,
    dgp_note=(
        "Anakütle doğrusu bilindiği için her örneklemin tahmini gerçek değerle karşılaştırılabilir. Grafikteki "
        "örneklem Monte Carlo döngüsünün ilk örneklemidir (aynı tohum, aynı çekiliş sırası). Döngüde eğim EKK "
        "formülüyle hesaplanır: β̂₁ = s_xy / s_x², yazılımın EKK eğimiyle aynıdır."
    ),
    look_at=(
        "**Saçılım grafiği** — düz çizgi bu örneklemin EKK doğrusu, kesikli çizgi anakütle doğrusu.",
        f"**Histogram** — {REPS_1} örneklemin eğim tahminleri ve anakütle eğimi 0,5.",
        "**n ve σ** — n büyüdükçe ya da σ küçüldükçe histogram nasıl değişiyor?",
    ),
    build=_build_lines,
    metrics=_lines_metrics,
    takeaway=_lines_takeaway,
    labels=(("x", "X"), ("y", "Y"), ("u", "Hata terimi u"), ("b1", "Eğim tahmini β̂₁")),
)


# --- Deney 2: EKK doğrusunu elle aramak ----------------------------------------------------------------

N_2 = 20
TRUE_2 = (2.0, 0.8)
SIGMA_2 = 1.5
"""Deney 2'nin anakütle doğrusu Y = 2 + 0,8·X + u, X ~ U(0, 10), u ~ N(0, 1,5²), n = 20."""


def _search_settings(parameters: Parameters) -> tuple[float, float]:
    return _rounded(parameters, "egim", 2), _rounded(parameters, "kayma", 1)


def _build_search(parameters: Parameters) -> tuple:
    slope, shift = _search_settings(parameters)
    tilt = E.mul(abs(slope), E.sub(E.var("x"), E.ref("xbar")))
    line = E.sub(E.ref("ybar"), tilt) if slope < 0 else E.add(E.ref("ybar"), tilt)
    candidate = E.sub(line, -shift) if shift < 0 else E.add(line, shift)  # kodda "+ -3" yazılmasın
    return (
        NewSample("orneklem", N_2, SEED),
        Draw("orneklem", "x", "uniform", 0, 10, "Açıklayıcı değişken X ~ U(0, 10)"),
        Draw("orneklem", "u", "normal", 0, SIGMA_2, "Hata terimi u ~ N(0; 1,5²)"),
        Derive("orneklem", "y", E.add(E.add(TRUE_2[0], E.mul(TRUE_2[1], E.var("x"))), E.var("u")), "Y = 2 + 0,8·X + u"),
        Statistic("orneklem", "x", "mean", "xbar", "X̄", decimals=3),
        Statistic("orneklem", "y", "mean", "ybar", "Ȳ", decimals=3),
        Derive("orneklem", "aday", candidate, f"Aday doğru: Ȳ {_signed(slope)}·(X − X̄) {_signed(shift, 1)}"),
        Derive("orneklem", "aday_artik", E.sub(E.var("y"), E.var("aday")), "Aday doğrunun artığı"),
        Derive("orneklem", "aday_kare", E.power(E.var("aday_artik"), 2), "Aday doğrunun kareli artığı"),
        Statistic("orneklem", "aday_artik", "sum", "aday_toplam", "Aday doğru: artıkların toplamı", decimals=3),
        Statistic("orneklem", "aday_kare", "sum", "aday_hkt", "Aday doğru: kareli artıklar toplamı", decimals=3),
        OLS("model", "orneklem", "y", ("x",), "EKK doğrusu: y ~ x"),
        ModelValue("b0", "model", "coef", "EKK sabiti β̂₀", term=INTERCEPT, decimals=3),
        ModelValue("b1", "model", "coef", "EKK eğimi β̂₁", term="x", decimals=3),
        ModelValue("ekk_hkt", "model", "ssr", "EKK doğrusu: kareli artıklar toplamı", decimals=3),
        Scalar("aday_sabit", E.sub(E.ref("ybar"), E.mul(E.ref("xbar"), slope)) if slope >= 0 else
               E.add(E.ref("ybar"), E.mul(E.ref("xbar"), -slope)), "Aday doğrunun sabiti (kayma öncesi)", decimals=3),
        Scalar("aday_sabit_d", E.sub(E.ref("aday_sabit"), -shift) if shift < 0 else E.add(E.ref("aday_sabit"), shift),
               "Aday doğrunun sabiti", decimals=3),
        ScatterPlot("orneklem", "x", "y", "X", "Y", "Aday doğru ve EKK doğrusu", fit_line=True, size=10,
                    opacity=0.8, lines=(("aday_sabit_d", slope, "Aday doğru"),)),
    )


def _search_dgp(parameters: Parameters) -> tuple[str, ...]:
    slope, shift = _search_settings(parameters)
    return (
        r"Y_i = 2 + 0{,}8\,X_i + u_i, \qquad X_i \sim U(0,\ 10), \qquad u_i \sim N(0;\ 1{,}5^2), \qquad n = 20",
        rf"\text{{Aday doğru: }} \tilde Y_i = \bar Y + b\,(X_i - \bar X) + d, \qquad b = {_tex(slope)}, "
        rf"\quad d = {_tex(shift, 1)}",
        r"\text{EKK ölçütü: } \min_{b_0,\,b_1} \sum_{i=1}^{n} (Y_i - b_0 - b_1 X_i)^2",
    )


def _search_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    s = state.scalars
    return (
        SimMetric("Aday: Σ(artık)²", plain(s["aday_hkt"], 2), "Aday doğrunun kareli artıklar toplamı."),
        SimMetric("EKK: Σ(artık)²", plain(s["ekk_hkt"], 2), "EKK doğrusunun kareli artıklar toplamı: en küçük değer."),
        SimMetric("Aday: Σ(artık)", plain(s["aday_toplam"], 2), "Aday doğrunun artıklarının toplamı."),
        SimMetric("EKK eğimi β̂₁", plain(s["b1"], 3), f"EKK sabiti β̂₀ = {plain(s['b0'], 3)}."),
    )


def _search_takeaway(state: LabState, parameters: Parameters) -> str:
    slope, shift = _search_settings(parameters)
    s = state.scalars
    gap = s["aday_hkt"] - s["ekk_hkt"]
    if gap < 0.01 * s["ekk_hkt"]:
        best = round(round(s["b1"] / 0.05) * 0.05, 2)  # kaydırıcı ızgarasında EKK eğimine en yakın eğim
        text = (
            f"Aday doğru EKK doğrusuna çok yakın: kareli artıklar toplamı {plain(s['aday_hkt'], 2)}, en küçük değer "
            f"{plain(s['ekk_hkt'], 2)}. EKK doğrusu (eğim {plain(s['b1'], 3)}, (X̄, Ȳ)'den geçer) Denklem 3.7'deki "
            "ölçütü en küçük yapan tek doğrudur; ondan hangi yöne ayrılırsanız ayrılın toplam artar (§3.8)."
        )
        if shift != 0 or abs(slope - best) > 1e-9:
            text += (" Kaydırıcılarla ulaşılabilecek en küçük toplam, "
                     f"eğim {_short(best)} ve kayma 0 iken elde edilir.")
        return text
    text = (f"Aday doğrunun kareli artıklar toplamı {plain(s['aday_hkt'], 2)}, EKK doğrusununki "
            f"{plain(s['ekk_hkt'], 2)}; fark {plain(gap, 2)}.")
    if shift == 0:
        text += (" Kayma d = 0 olduğu için aday doğru (X̄, Ȳ) noktasından geçer ve artıklarının toplamı sıfırdır; "
                 "yine de kareli toplam EKK'ninkinden büyüktür. Artıkların toplamının sıfır olması doğruyu belirlemez: "
                 "(X̄, Ȳ)'den geçen her doğrunun artık toplamı sıfırdır. "
                 "EKK, kareli toplamı en küçük yapan eğimi seçer.")
    else:
        text += (f" Kayma d = {_short(shift, 1)}: doğru (X̄, Ȳ) noktasından geçmez; artıkların toplamı "
                 f"{plain(s['aday_toplam'], 2)} = −n·d olur ve kareli toplam büyür.")
    if slope == 0 and shift == 0:
        text += " b = 0 ve d = 0: aday doğru yatay Ȳ çizgisidir; yalnız ortalamayı kullanan tahmin (Konu 4'te ölçüt)."
    return text + " Eğimi EKK eğimine yaklaştırıp kaymayı sıfırlayın (§3.8)."


SEARCH = SimExperiment(
    topic_key=TOPIC,
    number=2,
    title="EKK doğrusunu elle aramak: kareli artıklar toplamı",
    question="Aynı noktalara birçok doğru çizilebilir. Hangi doğru ‘en iyisi’dir? Artıkların toplamını sıfır yapmak "
             "yeter mi, yoksa kareli artıklar toplamına mı bakmalıyız?",
    note=NoteRef("3.8"),
    parameters=(
        SimParameter("egim", "Aday doğrunun eğimi b", -0.5, 2, 0, 0.05,
                     "b = 0: yatay çizgi. EKK eğimini metriklerde görebilirsiniz.", decimals=2),
        SimParameter("kayma", "Dikey kayma d", -3, 3, 0, 0.1,
                     "d = 0: aday doğru (X̄, Ȳ) noktasından geçer.", decimals=1),
    ),
    dgp=_search_dgp,
    dgp_note=(
        "Aday doğru Ȳ + b·(X − X̄) + d biçimindedir: d = 0 iken (X̄, Ȳ) noktasından geçer. Veri tek bir örneklemdir (n "
        "= 20, tohum 305); kaydırıcılar veriyi değil, aday doğruyu değiştirir."
    ),
    look_at=(
        "**Grafik** — kesikli çizgi aday doğru, düz çizgi EKK doğrusu.",
        "**Kareli artıklar toplamı** — aday doğru EKK doğrusuna yaklaştıkça nasıl değişiyor?",
        "**Artıkların toplamı** — d = 0 iken eğim ne olursa olsun sıfır mı?",
    ),
    build=_build_search,
    metrics=_search_metrics,
    takeaway=_search_takeaway,
    labels=(("x", "X"), ("y", "Y"), ("aday", "Aday doğrunun değeri"), ("aday_artik", "Aday artık"),
            ("aday_kare", "Aday kareli artık")),
)


# --- Deney 3: hata terimi ve artık ----------------------------------------------------------------------

def _errors_settings(parameters: Parameters) -> tuple[int, float]:
    return int(parameters["n"]), _rounded(parameters, "sigma", 1)


def _build_errors(parameters: Parameters) -> tuple:
    n, sigma = _errors_settings(parameters)
    return (
        NewSample("orneklem", n, SEED),
        *_population_draws("orneklem", sigma),
        OLS("model", "orneklem", "y", ("x",), "EKK doğrusu: y ~ x"),
        ModelValue("b0", "model", "coef", "Tahmin edilen sabit β̂₀", term=INTERCEPT, decimals=3),
        ModelValue("b1", "model", "coef", "Tahmin edilen eğim β̂₁", term="x", decimals=3),
        Derive("orneklem", "tahmin", E.add(E.ref("b0"), E.mul(E.ref("b1"), E.var("x"))), "Tahmin Ŷ = β̂₀ + β̂₁X"),
        Derive("orneklem", "artik", E.sub(E.var("y"), E.var("tahmin")), "Artık û = Y − Ŷ"),
        Derive("orneklem", "fark", E.sub(E.var("u"), E.var("artik")), "Hata terimi − artık: u − û"),
        Derive("orneklem", "fark_mutlak", E.absolute(E.var("fark")), "|u − û|"),
        Statistic("orneklem", "u", "sum", "toplam_u", "Hata terimlerinin toplamı Σu", decimals=3),
        Statistic("orneklem", "artik", "sum", "toplam_artik", "Artıkların toplamı Σû", decimals=3),
        PairStatistic("orneklem", "u", "artik", "corr", "r_u_artik", "Hata terimi ile artığın korelasyonu", decimals=3),
        Statistic("orneklem", "fark", "max", "en_buyuk_fark", "En büyük u − û", decimals=3),
        Statistic("orneklem", "fark", "min", "en_kucuk_fark", "En küçük u − û", decimals=3),
        Statistic("orneklem", "fark_mutlak", "max", "en_buyuk_mutlak", "En büyük |u − û|", decimals=3),
        ShowFrame("orneklem", ("x", "y", "u", "tahmin", "artik"), f"İlk {min(n, 8)} gözlem: hata terimi u ve artık û",
                  head=8, decimals=3),
        ScatterPlot("orneklem", "u", "artik", "Hata terimi u (simülasyonda bilinir)", "Artık û (veriden hesaplanır)",
                    "Hata terimi ve artık", size=9, opacity=0.8, lines=((0, 1, "û = u doğrusu"),)),
    )


def _errors_dgp(parameters: Parameters) -> tuple[str, ...]:
    n, sigma = _errors_settings(parameters)
    return (
        rf"Y_i = -1 + 0{{,}}5\,X_i + u_i, \qquad X_i \sim U(8,\ 18), \qquad u_i \sim N(0,\ \sigma^2), \qquad "
        rf"\sigma = {_tex(sigma, 1)}, \quad n = {n}",
        r"\hat u_i = Y_i - \hat\beta_0 - \hat\beta_1 X_i = u_i - "
        r"(\hat\beta_0 - \beta_0) - (\hat\beta_1 - \beta_1)\,X_i",
    )


def _errors_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    s = state.scalars
    return (
        SimMetric("Σu", plain(s["toplam_u"], 3), "Hata terimlerinin bu örneklemdeki toplamı (simülasyonda bilinir)."),
        SimMetric("Σû", plain(s["toplam_artik"], 3), "Artıkların toplamı: sabit terimli EKK'de her zaman sıfır."),
        SimMetric("r(u, û)", plain(s["r_u_artik"], 3), "Hata terimi ile artığın korelasyonu."),
        SimMetric("En büyük |u − û|", plain(s["en_buyuk_mutlak"], 3),
                  "Hata terimi ile artık arasındaki en büyük mutlak fark."),
    )


def _errors_takeaway(state: LabState, parameters: Parameters) -> str:
    n, _ = _errors_settings(parameters)
    s = state.scalars
    return (
        f"Artıkların toplamı {plain(s['toplam_artik'], 3)}: EKK'nin cebirsel özelliğidir (§3.9). Hata terimlerinin "
        f"toplamı ise {plain(s['toplam_u'], 3)}: u'nun anakütle ortalaması sıfırdır ama belirli bir örneklemdeki "
        "toplamı sıfır olmak zorunda değildir. Artık, hata teriminin veriden hesaplanan karşılığıdır: "
        "û = u − (β̂₀ − β₀) − (β̂₁ − β₁)·X. Tahminler anakütle parametrelerinden ne kadar ayrılırsa artık da hata "
        f"teriminden o kadar ayrılır; bu örneklemde u − û, {plain(s['en_kucuk_fark'], 2)} ile "
        f"{plain(s['en_buyuk_fark'], 2)} arasındadır. Gerçek "
        "verilerde u hiç gözlenmez; yalnız û hesaplanır (§3.5, §3.7)."
    )


ERRORS = SimExperiment(
    topic_key=TOPIC,
    number=3,
    title="Hata terimi ve artık aynı şey değildir",
    question="Hata terimi u anakütle modeline aittir ve gözlenmez; artık û örneklemden hesaplanır. Simülasyonda "
             "ikisini de bildiğimize göre: ne kadar benzerler, nerede ayrılırlar?",
    note=NoteRef("3.5", objects=("§3.7",)),
    parameters=(
        SimParameter("n", "Örneklem büyüklüğü n", 5, 300, 20, 5, "Gözlem sayısı.", integer=True, decimals=0),
        SimParameter("sigma", "Hata teriminin standart sapması σ", 0.5, 6, 3, 0.5, "Hata teriminin ölçeği.",
                     decimals=1),
    ),
    dgp=_errors_dgp,
    dgp_note=(
        "Simülasyonda hata terimi u çekildiği için bilinir. Artık û ise EKK doğrusu tahmin edildikten sonra "
        "hesaplanır. İkinci satır iki büyüklüğün cebirsel ilişkisidir."
    ),
    look_at=(
        "**Tablo** — tablonun satırlarında (en çok sekiz gözlem) u ve û yan yana.",
        "**Grafik** — noktalar û = u doğrusunun ne kadar yakınında?",
        "**Toplamlar** — Σû her zaman sıfır; Σu ya?",
    ),
    build=_build_errors,
    metrics=_errors_metrics,
    takeaway=_errors_takeaway,
    labels=(("x", "X"), ("y", "Y"), ("u", "Hata terimi u"), ("tahmin", "Tahmin Ŷ"), ("artik", "Artık û"),
            ("fark", "u − û"), ("fark_mutlak", "|u − û|")),
)


KONU03_EXPERIMENTS = (LINES, SEARCH, ERRORS)
