"""Konu 6 Sezgi deneyleri: veri üretim süreci bilinen kontrollü simülasyonlar.

Deney 1  Yansızlık: tekrarlı örneklemde EKK eğimi                     (Notlar §6.4, Tablo 6.3, Şekil 6.1–6.2)
Deney 2  Eksik değişken formülünü görmek                               (Notlar §6.7, Tablo 6.5)
Deney 3  Çoklu doğrusal bağlantı ve katsayı kararsızlığı              (Notlar §6.10–6.11, Tablo 6.7–6.8, Şekil 6.4–6.5)

Bölüm 6'nın benzetimleri bu deneylerin varsayılan ayarlarıdır (tohum 305); notlardaki tablolar ve şekiller
(``scripts/bolum6_uygulama.py``) aynı çekiliş sırasıyla üretilir ve testle karşılaştırılır. Deney 1'in iki senaryosu ve
Deney 3'ün altı bağlantı düzeyi aynı tohumla başlar (ortak rastgele sayılar): senaryolar ve düzeyler arasında yalnız
parametre değişir. Standart hata Konu 7'nin konusudur; burada tahminlerin dağılımı tekrarlardan doğrudan hesaplanır.
"""

from __future__ import annotations

from core.labs import expr as E
from core.labs.runner import LabState
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, plain
from core.labs.spec import (
    INTERCEPT,
    OLS,
    BarChart,
    CopyFrame,
    Derive,
    Draw,
    Histogram,
    LineChart,
    ModelValue,
    MonteCarlo,
    NewSample,
    NoteRef,
    PairStatistic,
    RegressionTable,
    Scalar,
    ScalarTable,
    ScatterPlot,
    Statistic,
    SummaryTable,
)

SEED = 305
N_NOTES_1, N_MAX_1 = 80, 400
"""Deney 1'de notlardaki gözlem sayısı ve kaydırıcının üst ucu."""
N_NOTES_3, SIGMA_MIN_3, DELTA_MAX_3 = 120, 0.01, 0.5
"""Deney 3'te notlardaki gözlem sayısı ve kaydırıcı uçları (özet metindeki öneriler bunlara göre verilir)."""
VIF_LOW_3 = 10.0
"""Deney 3 özetinin iç eşiği: altında "yeterli bağımsız hareket" cümlesi yazılır. Notlar evrensel bir VIF eşiği vermez;
bu yalnız metnin hangi durumu anlattığını seçer."""
TOPIC = "konu06"


def _rounded(parameters: Parameters, key: str, decimals: int) -> float:
    """Kaydırıcı değeri, kodda 0,30000000000000004 gibi yazımlar olmasın diye yuvarlanır."""

    return round(float(parameters[key]), decimals)


def _tex(value: float, decimals: int = 2) -> str:
    """LaTeX içinde sondaki sıfırları atılmış sayı (0{,}5; 3)."""

    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text.replace(".", "{,}")


def _thousands(value: int) -> str:
    """Binlik ayırıcılı tam sayı (3.000; 5.000)."""

    return f"{value:,}".replace(",", ".")


def _short(value: float, decimals: int = 2) -> str:
    """Sondaki sıfırları atılmış düz metin sayı (0,5; 3); yuvarlanınca sıfır olan değer "0" yazılır."""

    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "0", "") else text.replace(".", ",").replace("-", "−")


def _factor(text: str) -> str:
    """Çarpımda negatif sayı parantez içinde yazılır: "(−4)·(−1,552)" (“−4·−1,552” yazılmaz)."""

    return f"({text})" if text.startswith("−") else text


def _gamma_x(gamma: float) -> str:
    """γX'in yazımı: 0,8X; X (γ = 1); −X (γ = −1); 0 (γ = 0)."""

    if gamma == 0:
        return "0"
    if abs(gamma) == 1:
        return "X" if gamma > 0 else "−X"
    return f"{_short(gamma, 1)}X"


def _correlation(value: float) -> str:
    """Korelasyon: üç basamak; yuvarlanınca 1,000 görünecek kadar yüksekse (tam 1 değil) dört ya da beş basamak."""

    for decimals in (3, 4, 5):
        if abs(round(value, decimals)) < 1:
            return plain(value, decimals)
    return plain(value, 6)


# --- Deney 1: yansızlık ------------------------------------------------------------------------------

REPS_1 = 3000
BETA_1 = 1.5
BINS_1 = 45
"""Tablo 6.3: Y = 2 + 1,5·X + u; 3.000 örneklem; histogramlarda 45 kutu (Şekil 6.1–6.2)."""


def _unbiased_settings(parameters: Parameters) -> tuple[int, float]:
    return int(parameters["n"]), _rounded(parameters, "gamma", 1)


def unbiased_axis(n: int, gamma: float) -> tuple[float, float]:
    """İki histogramın ortak ekseni: iki merkezin (1,5 ve 1,5 + γ) çevresinde tahminlerin yaklaşık standart sapmasının
    (1/√n) beş katı."""

    half = 5 / n ** 0.5
    return round(min(BETA_1, BETA_1 + gamma) - half, 1), round(max(BETA_1, BETA_1 + gamma) + half, 1)


def _scenario(n: int, gamma: float | None) -> tuple:
    """Bir senaryonun tekrar gövdesi: X ve ε çekilir; Y = 2 + 1,5X (+ γX) + ε; eğim EKK ile tahmin edilir."""

    systematic = E.add(2, E.mul(BETA_1, E.var("x")))
    if gamma is None:
        outcome, comment = E.add(systematic, E.var("e")), "Y = 2 + 1,5X + u, u = ε"
    else:
        outcome = E.add(E.add(systematic, E.mul(gamma, E.var("x"))), E.var("e"))
        comment = (f"Y = 2 + 1,5X + u, u = {_gamma_x(gamma)} + ε" if gamma != 0
                   else "Y = 2 + 1,5X + u, u = ε (γ = 0: senaryo 1 ile aynı)")
    return (
        NewSample("orneklem", n, None),
        Draw("orneklem", "x", "normal", 0, 1, "X ~ N(0, 1)"),
        Draw("orneklem", "e", "normal", 0, 1, "ε ~ N(0, 1)"),
        Derive("orneklem", "y", outcome, comment),
        OLS("model", "orneklem", "y", ("x",), "EKK: y ~ x"),
        ModelValue("egim", "model", "coef", "Eğim tahmini β̂₁", term="x"),
    )


def _build_unbiased(parameters: Parameters) -> tuple:
    n, gamma = _unbiased_settings(parameters)
    low, high = unbiased_axis(n, gamma)
    collect = (("b1", E.ref("egim")), ("sapma", E.sub(E.ref("egim"), BETA_1)))
    label = f"E(u | X) = {_gamma_x(gamma)}" if gamma != 0 else "Senaryo 2 (γ = 0)"
    size = "" if n == N_NOTES_1 else f", n = {n}"
    figure = (f"Şekil 6.2: sıfır koşullu ortalama bozulduğunda ({label}{size})" if gamma != 0
              else f"Senaryo 2: γ = 0, hata terimi X ile ilişkisiz (senaryo 1 ile aynı{size})")
    return (
        MonteCarlo("gecerli", REPS_1, SEED, _scenario(n, None), collect,
                   f"Senaryo 1, E(u | X) = 0: {REPS_1} örneklem, her birinde {n} gözlem"),
        MonteCarlo("ihlal", REPS_1, SEED, _scenario(n, gamma), collect,
                   f"Senaryo 2, {label}: aynı tohum, aynı X ve ε çekilişleri"),
        SummaryTable((("E(u | X) = 0", "gecerli"), (label, "ihlal")),
                     (("ortalama", "b1", "mean"), ("yanlilik", "sapma", "mean"), ("std_sapma", "b1", "std")),
                     "tablo63", "Tablo 6.3: iki senaryonun özeti (gerçek β₁ = 1,5)"
                     + ("" if (n, gamma) == (N_NOTES_1, 0.8) else ", seçtiğiniz ayarlarla"), decimals=3,
                     heading="Senaryo"),
        Statistic("gecerli", "b1", "mean", "ort_gecerli", "Senaryo 1: tahminlerin ortalaması", decimals=3),
        Statistic("ihlal", "b1", "mean", "ort_ihlal", "Senaryo 2: tahminlerin ortalaması", decimals=3),
        Statistic("gecerli", "b1", "std", "sd_gecerli", "Senaryo 1: tahminlerin standart sapması", decimals=3),
        Statistic("ihlal", "b1", "std", "sd_ihlal", "Senaryo 2: tahminlerin standart sapması", decimals=3),
        Histogram("gecerli", (("b1", "Eğim tahminleri β̂₁"),), BINS_1, low, high,
                  "Şekil 6.1: sıfır koşullu ortalama sağlandığında" + (f" (n = {n})" if size else ""),
                  "Tahmin edilen eğim β̂₁",
                  references=((BETA_1, "Gerçek β₁ = 1,5"), ("ort_gecerli", "Ortalama tahmin"))),
        Histogram("ihlal", (("b1", "Eğim tahminleri β̂₁"),), BINS_1, low, high, figure, "Tahmin edilen eğim β̂₁",
                  references=((BETA_1, "Gerçek β₁ = 1,5"), ("ort_ihlal", "Ortalama tahmin"))),
    )


def _unbiased_dgp(parameters: Parameters) -> tuple[str, ...]:
    n, gamma = _unbiased_settings(parameters)
    return (
        rf"Y_i = 2 + 1{{,}}5\,X_i + u_i, \qquad X_i \sim N(0,\ 1), \qquad \varepsilon_i \sim N(0,\ 1), \qquad n = {n}",
        rf"\text{{Senaryo 1: }} u_i = \varepsilon_i, \qquad \text{{Senaryo 2: }} u_i = \gamma X_i + \varepsilon_i, "
        rf"\quad \gamma = {_tex(gamma, 1)}",
        rf"\text{{Her senaryoda {_thousands(REPS_1)} örneklem; iki senaryo aynı }} X_i \text{{ ve }} \varepsilon_i "
        r"\text{ çekilişlerini kullanır}",
    )


def _unbiased_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    _, gamma = _unbiased_settings(parameters)
    s = state.scalars
    return (
        SimMetric("Ortalama β̂₁: E(u | X) = 0", plain(s["ort_gecerli"], 3),
                  f"{_thousands(REPS_1)} tahminin ortalaması; gerçek eğim 1,5."),
        SimMetric(f"Ortalama β̂₁: E(u | X) = {_gamma_x(gamma)}" if gamma != 0 else "Ortalama β̂₁: senaryo 2",
                  plain(s["ort_ihlal"], 3),
                  "Sıfır koşullu ortalama bozulduğunda tahminlerin ortalaması." if gamma != 0
                  else "γ = 0: senaryo 2, senaryo 1 ile aynıdır."),
        SimMetric("Senaryo 2'de yanlılık", plain(s["ort_ihlal"] - BETA_1, 3),
                  "Tahminlerin ortalaması − gerçek eğim; beklenen değer γ'dır."),
        SimMetric("Tahminlerin std. sapması", plain(s["sd_gecerli"], 3),
                  f"Senaryo 1'de; senaryo 2'de {plain(s['sd_ihlal'], 3)}: "
                  + ("yayılım aynı, merkez farklı." if gamma != 0 else "yayılım da merkez de aynı.")),
    )


def _unbiased_takeaway(state: LabState, parameters: Parameters) -> str:
    n, gamma = _unbiased_settings(parameters)
    s = state.scalars
    text = (f"Birinci senaryoda {_thousands(REPS_1)} tahminin ortalaması {plain(s['ort_gecerli'], 3)}: gerçek eğim 1,5'in çevresinde "
            "toplanır. Tek tek tahminler gerçek değerden sapar; fakat sapmalar sistematik olarak tek yönde değildir. "
            "Yansızlık tek bir tahminin değil, tahmin edicinin tekrarlı örneklem davranışıdır (§6.3, §6.4).")
    if gamma == 0:
        return text + " γ = 0 iken iki senaryo aynıdır; γ'yı değiştirerek sıfır koşullu ortalamayı bozun."
    return text + (
        f" İkinci senaryoda hata terimi X ile sistematik ilişkilidir: E(u | X) = {_gamma_x(gamma)}. Tahminler "
        f"{plain(s['ort_ihlal'], 3)} çevresinde toplanır; yanlılık yaklaşık {plain(s['ort_ihlal'] - BETA_1, 3)}. İki "
        f"senaryoda tahminlerin standart sapması aynıdır ({plain(s['sd_gecerli'], 3)}): yöntem ve örneklem büyüklüğü "
        "aynı, yalnız merkez kaymıştır. "
        + ("n'yi büyütün: histogram daralır ama ikinci senaryonun merkezi 1,5'e dönmez; " if n < N_MAX_1 else
           f"n = {N_MAX_1} ile histogram en dar hâlindedir ama ikinci senaryonun merkezi 1,5'e dönmez; ")
        + "daha fazla tekrar ya da gözlem sistematik yanlılığı ortadan kaldırmaz (§6.4).")


UNBIASED = SimExperiment(
    topic_key=TOPIC,
    number=1,
    title="Yansızlık: tekrarlı örneklemde EKK eğimi",
    question="Aynı veri üretim sürecinden binlerce örneklem çekip her birinde EKK eğimini tahmin etseydik tahminler "
             "nerede toplanırdı? Hata terimi X ile sistematik ilişkili olduğunda ne değişir?",
    note=NoteRef("6.4", objects=("Tablo 6.3", "Şekil 6.1–6.2")),
    parameters=(
        SimParameter("n", "Örneklem büyüklüğü n", 10, N_MAX_1, N_NOTES_1, 10,
                     "Her örneklemdeki gözlem sayısı; notlarda 80.", integer=True, decimals=0),
        SimParameter("gamma", "Senaryo 2'de E(u | X) = γX: γ", -1.5, 1.5, 0.8, 0.1,
                     "γ = 0: iki senaryo aynı. Notlarda 0,8.", decimals=1),
    ),
    dgp=_unbiased_dgp,
    dgp_note=(
        "Varsayılan ayarlar notlardaki benzetimdir (Tablo 6.3, Kod 6.1; tohum 305). İki senaryo aynı tohumla başladığı "
        "için aynı X ve ε çekilişlerini kullanır; yalnız hata teriminin X ile ilişkisi farklıdır. Uygulama "
        "tekrarları vektörel hesaplar; üretilen kod notlardaki gibi döngüyle aynı sayıları verir."
    ),
    look_at=(
        "**İki histogram** — tahminler hangi değerin çevresinde toplanıyor? Kesikli çizgi gerçek eğim 1,5.",
        "**Tablo** — ortalama, yanlılık ve tahminlerin standart sapması.",
        "**n ve γ** — n büyüyünce histogram daralıyor mu? İkinci senaryonun merkezi 1,5'e dönüyor mu?",
    ),
    build=_build_unbiased,
    metrics=_unbiased_metrics,
    takeaway=_unbiased_takeaway,
    tables=(("tablo63", lambda parameters: "Tablo 6.3: sıfır koşullu ortalama sağlandığında ve bozulduğunda"
             if _unbiased_settings(parameters)[1] != 0 else "Tablo 6.3: γ = 0 iken iki senaryo aynıdır",
             ("n", "gamma")),),
    labels=(("x", "X"), ("e", "ε"), ("y", "Y"), ("b1", "Eğim tahmini β̂₁"), ("sapma", "β̂₁ − β₁"),
            ("ortalama", "Ortalama tahmin"), ("yanlilik", "Yanlılık"), ("std_sapma", "Tahminlerin std. sapması")),
)


# --- Deney 2: eksik değişken formülü -----------------------------------------------------------------

BETA_X = 2.0
"""Deney 2'nin gerçek modeli: Y = 1 + 2X + β_Z·Z + u, Z = δX + v (notlarda β_Z = 3, δ = 0,7, n = 5.000)."""


def _omitted_settings(parameters: Parameters) -> tuple[int, float, float]:
    return int(parameters["n"]), _rounded(parameters, "delta", 1), _rounded(parameters, "beta_z", 1)


def _build_omitted(parameters: Parameters) -> tuple:
    n, delta, beta_z = _omitted_settings(parameters)
    return (
        NewSample("veri", n, SEED),
        Draw("veri", "x", "normal", 0, 1, "X ~ N(0, 1)"),
        Draw("veri", "v", "normal", 0, 1, "v ~ N(0, 1)"),
        Derive("veri", "z", E.add(E.mul(delta, E.var("x")), E.var("v")), f"Z = {_short(delta, 1)}X + v"),
        Draw("veri", "u", "normal", 0, 1, "u ~ N(0, 1)"),
        Derive("veri", "y", E.add(E.add(E.add(1, E.mul(BETA_X, E.var("x"))), E.mul(beta_z, E.var("z"))), E.var("u")),
               f"Y = 1 + 2X + {_short(beta_z, 1)}Z + u"),
        OLS("kisa", "veri", "y", ("x",), "Kısa model (Z dışarıda): y ~ x"),
        OLS("uzun", "veri", "y", ("x", "z"), "Uzun model: y ~ x + z"),
        OLS("yardimci", "veri", "z", ("x",), "Yardımcı regresyon: z ~ x"),
        ModelValue("kisa_x", "kisa", "coef", "Kısa model: X eğimi", term="x", decimals=3),
        ModelValue("uzun_x", "uzun", "coef", "Uzun model: X eğimi", term="x", decimals=3),
        ModelValue("uzun_z", "uzun", "coef", "Uzun model: Z eğimi", term="z", decimals=3),
        ModelValue("yardimci_egim", "yardimci", "coef", "Z'nin X üzerindeki yardımcı eğimi δ̂", term="x", decimals=3),
        Scalar("beklenen_katki", E.mul(beta_z, E.ref("yardimci_egim")), "Beklenen eksik değişken katkısı β_Z·δ̂",
               decimals=3),
        Scalar("beklenen_kisa", E.add(BETA_X, E.ref("beklenen_katki")), "Beklenen kısa model eğimi β_X + β_Z·δ̂",
               decimals=3),
        ScalarTable((
            ("Gerçek X katsayısı", E.const(BETA_X)),
            ("Gerçek Z katsayısı", E.const(beta_z)),
            ("Z'nin X üzerindeki yardımcı eğimi", E.ref("yardimci_egim")),
            ("Beklenen eksik değişken katkısı", E.ref("beklenen_katki")),
            ("Beklenen kısa model X eğimi", E.ref("beklenen_kisa")),
            ("Tahmin edilen kısa model X eğimi", E.ref("kisa_x")),
            ("Tahmin edilen uzun model X eğimi", E.ref("uzun_x")),
            ("Tahmin edilen uzun model Z eğimi", E.ref("uzun_z")),
        ), "tablo65", decimals=3),
        ScalarTable((("Gerçek β_X", E.const(BETA_X)), ("Kısa model", E.ref("kisa_x")), ("Uzun model", E.ref("uzun_x"))),
                    "egimler", decimals=3),
        BarChart("egimler", "deger", "Model", "X katsayısı", "Gerçek X katsayısı, kısa ve uzun model tahmini",
                 decimals=3),
    )


def _omitted_dgp(parameters: Parameters) -> tuple[str, ...]:
    n, delta, beta_z = _omitted_settings(parameters)
    return (
        rf"Z_i = \delta X_i + v_i, \qquad Y_i = 1 + \beta_X X_i + \beta_Z Z_i + u_i, \qquad \beta_X = 2, "
        rf"\quad \delta = {_tex(delta, 1)}, \quad \beta_Z = {_tex(beta_z, 1)}",
        rf"X_i,\ v_i,\ u_i \sim N(0,\ 1), \qquad n = {_thousands(n)}",
        r"\text{Kısa model: } Y \sim X \qquad \text{Uzun model: } Y \sim X + Z \qquad "
        r"\mathbb{E}(\widehat{\widetilde\beta}_1) = \beta_X + \beta_Z\,\delta",
    )


def _omitted_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    s = state.scalars
    return (
        SimMetric("Kısa model: X eğimi", plain(s["kisa_x"], 3), "Z dışarıda."),
        SimMetric("Beklenen kısa eğim", plain(s["beklenen_kisa"], 3), "β_X + β_Z·δ̂ (Denklem 6.8)."),
        SimMetric("Uzun model: X eğimi", plain(s["uzun_x"], 3), "Z modelde; gerçek değer 2."),
        SimMetric("Yardımcı eğim δ̂", plain(s["yardimci_egim"], 3), "Z'nin X üzerindeki regresyon eğimi."),
    )


def _omitted_takeaway(state: LabState, parameters: Parameters) -> str:
    _, delta, beta_z = _omitted_settings(parameters)
    s = state.scalars
    if delta == 0 or beta_z == 0:
        if delta == 0 and beta_z == 0:
            missing = "Z, X ile ilişkisiz (δ = 0) ve Y'yi etkilemiyor (β_Z = 0): iki koşul da eksik"
        elif delta == 0:
            missing = "Z, X ile ilişkisiz (δ = 0): iki koşuldan biri eksik"
        else:
            missing = "Z, Y'yi etkilemiyor (β_Z = 0): iki koşuldan biri eksik"
        return (f"{missing}. Kısa model eğimi {plain(s['kisa_x'], 3)}, uzun model eğimi "
                f"{plain(s['uzun_x'], 3)}; ikisi de gerçek değer 2'ye yakındır. Eksik değişken yanlılığı için Z hem Y "
                "ile hem X ile ilişkili olmalıdır (§6.6).")
    direction = "yukarı" if beta_z * delta > 0 else "aşağı"
    return (f"Kısa model X katsayısını {plain(s['kisa_x'], 3)} bulur; formül β_X + β_Z·δ̂ = 2 + "
            f"{_factor(_short(beta_z, 1))}·{_factor(plain(s['yardimci_egim'], 3))} = {plain(s['beklenen_kisa'], 3)} "
            "bekler. Uzun model Z'yi "
            f"eklediğinde katsayılar gerçek değerlere yaklaşır ({plain(s['uzun_x'], 3)} ve {plain(s['uzun_z'], 3)}). "
            f"β_Z ve δ'nın işaretleri {'aynı' if beta_z * delta > 0 else 'farklı'}: {direction} yönlü yanlılık (Tablo "
            "6.4). Yanlılık iki bağlantının çarpımıdır; büyük örneklem kısa modelin yanlılığını ortadan kaldırmaz, "
            "yalnız yanlış hedefi daha kesin tahmin eder (§6.7).")


OMITTED = SimExperiment(
    topic_key=TOPIC,
    number=2,
    title="Eksik değişken formülünü görmek",
    question="Gerçek modelde Z de var, ama biz yalnız X'i kullanıyoruz. X'in katsayısı ne kadar ve hangi yönde "
             "sapar? Sapma, Z'nin Y ile ve X ile ilişkisine nasıl bağlıdır?",
    note=NoteRef("6.7", objects=("Tablo 6.5",)),
    parameters=(
        SimParameter("delta", "Z'nin X ile ilişkisi δ", -1.5, 1.5, 0.7, 0.1, "δ = 0: Z, X ile ilişkisiz. Notlarda 0,7.",
                     decimals=1),
        SimParameter("beta_z", "Z'nin Y üzerindeki katsayısı β_Z", -4, 4, 3, 0.5,
                     "β_Z = 0: Z, Y'yi etkilemez. Notlarda 3.", decimals=1),
        SimParameter("n", "Örneklem büyüklüğü n", 500, 20000, 5000, 500, "Notlarda 5.000.", integer=True, decimals=0),
    ),
    dgp=_omitted_dgp,
    dgp_note=(
        "Varsayılan ayarlar notlardaki sentetik uygulamadır (Tablo 6.5, Kod 6.2; tohum 305). Beklenen kısa model eğimi "
        "yardımcı regresyonun tahmin edilen eğimiyle hesaplanır (notlardaki gibi)."
    ),
    look_at=(
        "**Tablo** — beklenen kısa model eğimi ile tahmin edilen kısa model eğimi ne kadar yakın?",
        "**Grafik** — gerçek X katsayısı, kısa ve uzun model tahmini.",
        "**δ ve β_Z** — biri sıfır olunca yanlılık ne oluyor? İşaretler değişince yön?",
    ),
    build=_build_omitted,
    metrics=_omitted_metrics,
    takeaway=_omitted_takeaway,
    tables=(("tablo65", "Tablo 6.5: sentetik veride eksik değişken yanlılığı", ("delta", "beta_z", "n")),),
    labels=(("x", "X"), ("v", "v"), ("z", "Z"), ("u", "u"), ("y", "Y")),
)


# --- Deney 3: çoklu doğrusal bağlantı -------------------------------------------------------------------

REPS_3 = 700
LEVELS = (1.0, 0.5, 0.25, 0.12, 0.06, 0.03)
"""Tablo 6.7'nin gürültü düzeyleri σ (X₂ = X₁ + σ·gürültü); her düzeyde 700 örneklem."""


def _collinear_settings(parameters: Parameters) -> tuple[int, float, float]:
    return int(parameters["n"]), _rounded(parameters, "sigma", 2), _rounded(parameters, "delta", 2)


def _collinear_draws(frame: str, sigma: float) -> tuple:
    """X₁, gürültü, X₂ = X₁ + gürültü, u ve Y = 1 + 2X₁ + 2X₂ + u (çekiliş sırası sabit)."""

    return (
        Draw(frame, "x1", "normal", 0, 1, "X₁ ~ N(0, 1)"),
        Draw(frame, "gurultu", "normal", 0, sigma, f"Gürültü σ·g ~ N(0, σ²), σ = {_short(sigma)}"),
        Derive(frame, "x2", E.add(E.var("x1"), E.var("gurultu")), "X₂ = X₁ + gürültü"),
        Draw(frame, "u", "normal", 0, 1, "u ~ N(0, 1)"),
        Derive(frame, "y", E.add(E.add(E.add(1, E.mul(2, E.var("x1"))), E.mul(2, E.var("x2"))), E.var("u")),
               "Y = 1 + 2X₁ + 2X₂ + u"),
    )


def _vif(r: E.Expr) -> E.Expr:
    """İki açıklayıcı değişkenli modelde VIF = 1 / (1 − r²). r² çarpım olarak yazılır (r·r): üs alma (``**``) ile
    çarpım son basamakta farklı olabilir ve yüksek korelasyonda 1 − r² bu farkı büyütür."""

    return E.div(1, E.sub(1, E.mul(r, r)))


def _level(index: int, sigma: float, n: int) -> MonteCarlo:
    body = (
        NewSample("ornek", n, None),
        *_collinear_draws("ornek", sigma),
        OLS("model", "ornek", "y", ("x1", "x2"), "EKK: y ~ x1 + x2"),
        ModelValue("b1_t", "model", "coef", "β̂₁", term="x1"),
        ModelValue("b2_t", "model", "coef", "β̂₂", term="x2"),
        PairStatistic("ornek", "x1", "x2", "corr", "r_t", "X₁ ile X₂ korelasyonu"),
    )
    return MonteCarlo(
        f"duzey_{index}", REPS_3, SEED, body,
        (("b1", E.ref("b1_t")), ("b2", E.ref("b2_t")), ("toplam", E.add(E.ref("b1_t"), E.ref("b2_t"))),
         ("r", E.ref("r_t")), ("vif", _vif(E.ref("r_t")))),
        f"σ = {_short(sigma)}: {REPS_3} örneklem (her düzey aynı tohumla başlar)",
    )


def _scatter_title(sigma: float, n: int) -> str:
    """Şekil 6.4 başlığı: notlardaki σ = 0,05'te şekil numarasıyla; başka σ'da seçilen düzeyle; n notlardakinden
    farklıysa n de yazılır."""

    setting = f"σ = {_short(sigma)}" + ("" if n == N_NOTES_3 else f", n = {n}")
    if sigma == 0.05:
        return f"Şekil 6.4: tam olmayan fakat çok yüksek doğrusal bağlantı ({setting})"
    return f"X₁ ile X₂ ({setting})" + (": çok yüksek doğrusal bağlantı" if sigma <= 0.1 else "")


def _build_collinear(parameters: Parameters) -> tuple:
    n, sigma, delta = _collinear_settings(parameters)
    shift = E.mul(delta, E.compare("eq", E.var("id"), 1))
    single = (
        NewSample("orneklem", n, SEED),
        *_collinear_draws("orneklem", sigma),
        CopyFrame("degisik", "orneklem", "Özgün verinin kopyası; yalnız ilk gözlemin X₂ değeri değişecek"
                  if delta != 0 else "Özgün verinin kopyası (δ = 0: değişiklik yapılmaz, iki veri aynı)"),
        *((Derive("degisik", "x2", E.add(E.var("x2"), shift), f"İlk gözlemin X₂ değeri {_short(delta)} artırılır"),)
          if delta != 0 else ()),
        OLS("m_ozgun", "orneklem", "y", ("x1", "x2"), "Özgün veri: y ~ x1 + x2"),
        OLS("m_degisik", "degisik", "y", ("x1", "x2"), "Tek gözlemi değişmiş veri: y ~ x1 + x2" if delta != 0
            else "Kopya veri (δ = 0, değişiklik yok): y ~ x1 + x2"),
        ModelValue("b1_ozgun", "m_ozgun", "coef", "Özgün: β̂₁", term="x1", decimals=3),
        ModelValue("b2_ozgun", "m_ozgun", "coef", "Özgün: β̂₂", term="x2", decimals=3),
        ModelValue("b1_degisik", "m_degisik", "coef", "Değişik: β̂₁", term="x1", decimals=3),
        ModelValue("b2_degisik", "m_degisik", "coef", "Değişik: β̂₂", term="x2", decimals=3),
        Scalar("toplam_ozgun", E.add(E.ref("b1_ozgun"), E.ref("b2_ozgun")), "Özgün: β̂₁ + β̂₂", decimals=3),
        Scalar("toplam_degisik", E.add(E.ref("b1_degisik"), E.ref("b2_degisik")), "Değişik: β̂₁ + β̂₂", decimals=3),
        PairStatistic("orneklem", "x1", "x2", "corr", "r_ozgun", "Özgün: X₁–X₂ korelasyonu", decimals=3),
        PairStatistic("degisik", "x1", "x2", "corr", "r_degisik", "Değişik: X₁–X₂ korelasyonu", decimals=3),
        Scalar("vif_ozgun", _vif(E.ref("r_ozgun")), "Özgün: VIF", decimals=1),
        Scalar("vif_degisik", _vif(E.ref("r_degisik")), "Değişik: VIF", decimals=1),
        RegressionTable((("Özgün veri", "m_ozgun"),
                         (f"İlk X₂ gözlemi + {_short(delta)}" if delta != 0 else "Değişiklik yok (δ = 0)", "m_degisik")),
                        ("x1", "x2"), "tablo68", "Tablo 6.8: küçük veri değişikliğine duyarlılık"
                        + ("" if (n, sigma, delta) == (N_NOTES_3, 0.05, 0.08) else ", seçtiğiniz ayarlarla"),
                        stars=False,
                        decimals=4, standard_errors=False, term_decimals=(("x1", 3), ("x2", 3))),
        ScatterPlot("orneklem", "x1", "x2", "X₁", "X₂", _scatter_title(sigma, n), size=7, opacity=0.7,
                    lines=((0, 1, "X₂ = X₁"),)),
    )
    return (
        *(_level(index, level, n) for index, level in enumerate(LEVELS, start=1)),
        SummaryTable(
            tuple((f"σ = {_short(level)}", f"duzey_{index}") for index, level in enumerate(LEVELS, start=1)),
            (("ort_korelasyon", "r", "mean"), ("b1_std", "b1", "std"), ("b2_std", "b2", "std"),
             ("toplam_std", "toplam", "std"), ("ort_vif", "vif", "mean")),
            "tablo67", "Tablo 6.7: bağlantı arttıkça katsayıların değişkenliği"
            + ("" if n == N_NOTES_3 else ", seçtiğiniz ayarlarla"), decimals=3, heading="Gürültü σ",
            column_decimals=(("ort_vif", 1),),
        ),
        LineChart("tablo67", "ort_korelasyon", "b1_std", "X₁ ile X₂ arasındaki ortalama korelasyon",
                  "Tahminlerin standart sapması",
                  "Şekil 6.5: bağlantı yükseldikçe katsayıların değişkenliği" + ("" if n == N_NOTES_3 else f" (n = {n})"),
                  legend="β̂₁", series=(("b2_std", "β̂₂"), ("toplam_std", "β̂₁ + β̂₂"))),
        *single,
    )


def _collinear_dgp(parameters: Parameters) -> tuple[str, ...]:
    n, sigma, delta = _collinear_settings(parameters)
    levels = r";\ ".join(_tex(level) for level in LEVELS)
    return (
        r"Y_i = 1 + 2X_{1i} + 2X_{2i} + u_i, \qquad X_{2i} = X_{1i} + \sigma g_i, \qquad X_{1i},\ g_i,\ u_i \sim N(0,\ 1)",
        rf"\text{{Tekrarlı örneklem: }} \sigma \in \{{{levels}\}}, \quad \text{{her düzeyde {REPS_3} örneklem}}, "
        rf"\quad n = {n}",
        (rf"\text{{Tek örneklem: }} \sigma = {_tex(sigma)}, \quad \text{{ilk gözlemin }} X_2 \text{{ değeri }} "
         rf"{_tex(delta)} \text{{ artırılır}}" if delta != 0 else
         rf"\text{{Tek örneklem: }} \sigma = {_tex(sigma)}, \quad \delta = 0\text{{: iki sürüm aynı}}"),
    )


def _collinear_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    s = state.scalars
    _, _, delta = _collinear_settings(parameters)
    return (
        SimMetric("Özgün: β̂₁ + β̂₂", plain(s["toplam_ozgun"], 3), "Gerçek toplam 4."),
        SimMetric("Değişik: β̂₁ + β̂₂", plain(s["toplam_degisik"], 3),
                  "Tek gözlem değişince toplam." if delta != 0 else "δ = 0: özgün veriyle aynı."),
        SimMetric("Özgün: VIF", plain(s["vif_ozgun"], 1), f"X₁–X₂ korelasyonu {_correlation(s['r_ozgun'])}."),
        SimMetric("Değişik: VIF", plain(s["vif_degisik"], 1), f"X₁–X₂ korelasyonu {_correlation(s['r_degisik'])}."),
    )


def _collinear_takeaway(state: LabState, parameters: Parameters) -> str:
    _, sigma, delta = _collinear_settings(parameters)
    s = state.scalars
    table = state.tables["tablo67"]
    first, last = table.iloc[0], table.iloc[-1]
    text = (f"Ortalama korelasyon {plain(first['ort_korelasyon'], 3)} iken β̂₁'in tekrarlar arasındaki standart sapması "
            f"{plain(first['b1_std'], 3)}, korelasyon {plain(last['ort_korelasyon'], 4)} iken {plain(last['b1_std'], 3)}. "
            f"Toplamın (β̂₁ + β̂₂) standart sapması bütün düzeylerde {plain(first['toplam_std'], 3)}: veri ortak katkıyı "
            "bağlantıdan etkilenmeden belirler, iki değişken arasındaki paylaşımı ise giderek zayıf belirler. Yüksek "
            "bağlantı katsayıları yanlı yapmaz; ayrı katsayıları oynak yapar (§6.10).")
    if delta == 0:
        return text + " δ = 0 iken iki veri seti aynıdır; δ'yı artırıp Tablo 6.8'e bakın."
    change_1 = s["b1_degisik"] - s["b1_ozgun"]
    change_2 = s["b2_degisik"] - s["b2_ozgun"]
    small = max(abs(change_1), abs(change_2)) < 0.05
    digits = 4 if small else 3  # küçük değişimler üç basamakta 0,000 görünmesin
    vif = s["vif_ozgun"]
    text += (f" Tek örneklemde (σ = {_short(sigma)}, VIF {plain(vif, 1)}) ilk gözlemin X₂ değeri "
             f"{_short(delta)} artırılınca β̂₁ {plain(change_1, digits)}, β̂₂ {plain(change_2, digits)} değişir; toplam "
             f"{plain(s['toplam_ozgun'], 3)} ve {plain(s['toplam_degisik'], 3)}.")
    if not small:
        return text + (" Gerçek katsayıların ikisi de 2'dir: ayrı katsayıların büyüklüğüne ya da sıralamasına tek "
                       "örneklemde güçlü anlam yüklemek yanlış olur (§6.11).")
    moves = [move for move, possible in (("σ'yı küçültün", sigma > SIGMA_MIN_3), ("δ'yı artırın", delta < DELTA_MAX_3))
             if possible]
    advice = f" {' ya da '.join(moves)} (§6.11)." if moves else " (§6.11)."  # sembol büyük harfe çevrilmez
    if vif < VIF_LOW_3:
        return text + (" Bu σ düzeyinde X₁ ile X₂ arasında yeterli bağımsız hareket vardır: tek gözlemdeki küçük değişiklik "
                       "ayrı katsayıları yalnız biraz değiştirir." + advice)
    return text + (" Bağlantı yüksek, ama bu değişiklik ayrı katsayıları henüz az oynattı: değişim δ ile ve bağlantı "
                   "düzeyiyle birlikte büyür, gözlem sayısı arttıkça tek bir gözlemin payı küçülür." + advice)


COLLINEAR = SimExperiment(
    topic_key=TOPIC,
    number=3,
    title="Çoklu doğrusal bağlantı ve katsayı kararsızlığı",
    question="İki açıklayıcı değişken neredeyse aynı bilgiyi taşıdığında EKK ayrı katsayıları ne kadar iyi ayırır? "
             "Katsayıların toplamı ve tek bir gözlemdeki küçük değişiklik ne söyler?",
    note=NoteRef("6.10", objects=("§6.11", "Tablo 6.7", "Tablo 6.8")),
    parameters=(
        SimParameter("n", "Örneklem büyüklüğü n", 40, 400, N_NOTES_3, 20,
                     "Her örneklemdeki gözlem sayısı; notlarda 120.", integer=True, decimals=0),
        SimParameter("sigma", "Tek örneklemde gürültü σ", SIGMA_MIN_3, 1.0, 0.05, 0.01,
                     "σ küçüldükçe X₂, X₁'e yaklaşır. Notlarda 0,05.", decimals=2),
        SimParameter("delta", "İlk gözlemin X₂ değerine eklenen δ", 0.0, DELTA_MAX_3, 0.08, 0.01, "Notlarda 0,08.",
                     decimals=2),
    ),
    dgp=_collinear_dgp,
    dgp_note=(
        "Varsayılan ayarlar notlardaki benzetimdir (Tablo 6.7 ve 6.8; tohum 305). Altı düzey aynı tohumla başlar: "
        "düzeyler arasında yalnız σ değişir. Tek örneklemin ikinci sürümü özgün verinin kopyasıdır; δ > 0 ise yalnız "
        "ilk gözlemin X₂ değeri farklıdır. VIF iki açıklayıcı değişkenli modelde 1 / (1 − r²)'dir."
    ),
    look_at=(
        "**Tablo 6.7 ve Şekil 6.5** — korelasyon yükseldikçe β̂₁ ve β̂₂'nin değişkenliği; toplamınki?",
        "**Tablo 6.8** — tek gözlem değişince ayrı katsayılar ve toplam.",
        "**Şekil 6.4** — X₂ ile X₁ neredeyse aynı doğru üzerinde; σ'yı büyütüp küçültün.",
    ),
    build=_build_collinear,
    metrics=_collinear_metrics,
    takeaway=_collinear_takeaway,
    tables=(("tablo67", "Tablo 6.7: tekrarlı örneklerde bağlantı ve katsayı değişkenliği", ("n",)),
            ("tablo68", "Tablo 6.8: küçük veri değişikliğine duyarlılık", ("n", "sigma", "delta"))),
    labels=(("x1", "X₁"), ("x2", "X₂"), ("gurultu", "Gürültü σ·g"), ("u", "u"), ("y", "Y"), (INTERCEPT, "Sabit terim"),
            ("ort_korelasyon", "Ort. korelasyon"), ("b1_std", "β̂₁ std. sapma"), ("b2_std", "β̂₂ std. sapma"),
            ("toplam_std", "(β̂₁ + β̂₂) std. sapma"), ("ort_vif", "Ort. VIF")),
)


KONU06_EXPERIMENTS = (UNBIASED, OMITTED, COLLINEAR)
