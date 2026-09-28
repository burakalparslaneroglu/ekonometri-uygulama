"""Konu 0 Sezgi deneyleri: veri üretim süreci bilinen kontrollü simülasyonlar.

Deney 1  Örneklem ortalaması: örnekleme değişimi ve seçim yanlılığı   (Notlar §0.2, §0.8)
Deney 2  Korelasyon yalnız doğrusal ilişkiyi ölçer                     (Notlar §0.5)
Deney 3  Koşullu ortalama: E(Y | X) ve örneklemdeki grup ortalamaları  (Notlar §0.7)

Simülasyonda normalde bilinmeyen anakütle nicelikleri (μ, ρ, E(Y | X)) bilinir; deneyler örneklemden hesaplananı bu
bilinen değerle karşılaştırır. Konu 0'da regresyon ve standart hata yoktur. Tohum 305.
"""

from __future__ import annotations

import math

from core.labs import expr as E
from core.labs.runner import LabState
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, number, percent, plain
from core.labs.spec import (
    Derive,
    Draw,
    DrawDiscrete,
    Histogram,
    InlineData,
    LineChart,
    MonteCarlo,
    NewSample,
    NoteRef,
    PairStatistic,
    ScatterPlot,
    ShowFrame,
    Statistic,
)

SEED = 305
TOPIC = "konu00"


def _rounded(parameters: Parameters, key: str, decimals: int) -> float:
    """Kaydırıcı değeri, kodda 0,30000000000000004 gibi yazımlar olmasın diye yuvarlanır."""

    return round(float(parameters[key]), decimals)


def _tex(value: float, decimals: int = 2) -> str:
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return text.replace(".", "{,}")


def _short(value: float, decimals: int = 2) -> str:
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return text.replace(".", ",").replace("-", "−")


# --- Deney 1: örnekleme değişimi ve seçim yanlılığı -----------------------------------------------------------

SHAPE = 4.0
SCALE_A, SCALE_B = 1.125, 1.875
MEAN_A, MEAN_B = SHAPE * SCALE_A, SHAPE * SCALE_B
MU = (MEAN_A + MEAN_B) / 2
"""Anakütle: eşit büyüklükte iki grup; A'da ücret ~ Gamma(4; 1,125) (ortalama 4,5), B'de Gamma(4; 1,875) (ortalama
7,5). Anakütle ortalaması μ = 6; WAGE1'deki ücrete benzer biçimde sağa çarpık."""
AXIS_1 = (2.0, 14.0)
BINS_1 = 120
"""Yatay eksen: kaydırıcıların bütün değerlerinde (tohum 305) bütün x̄'ler bu aralıktadır (en büyük 13,001)."""
CLOSE = 0.5


def sampled_share(strength: float) -> float:
    """Seçimli örneklemede bir gözlemin B grubundan gelme olasılığı p = (1 + s)/2; s = 0 basit rastgele örnekleme."""

    return (1 + strength) / 2


def expected_mean(strength: float) -> float:
    """E(x̄) = (1 − p)·4,5 + p·7,5 = 6 + 1,5·s."""

    p = sampled_share(strength)
    return (1 - p) * MEAN_A + p * MEAN_B


def sampled_sd(strength: float) -> float:
    """Örneklenen tek bir gözlemin standart sapması σ_p: grup içi varyansların ortalaması artı grup ortalamalarının
    varyansı."""

    p = sampled_share(strength)
    within = (1 - p) * SHAPE * SCALE_A ** 2 + p * SHAPE * SCALE_B ** 2
    between = p * (1 - p) * (MEAN_B - MEAN_A) ** 2
    return math.sqrt(within + between)


def _sampling_settings(parameters: Parameters) -> tuple[int, float, int]:
    return int(parameters["n"]), _rounded(parameters, "secim", 1), int(parameters["tekrar"])


def _build_sampling(parameters: Parameters) -> tuple:
    n, strength, reps = _sampling_settings(parameters)
    p = sampled_share(strength)
    xbar = E.ref("xbar")
    references = [(MU, "μ = 6 (anakütle ortalaması)")]
    if strength > 0:
        references.append((round(expected_mean(strength), 2), "E(X̄): seçimli örneklemede beklenen değer"))
    return (
        MonteCarlo(
            "tekrarlar",
            reps,
            SEED,
            (
                NewSample("orneklem", n, None),
                Draw("orneklem", "u", "uniform", 0, 1, "Grup için tek-düze sayı"),
                Draw("orneklem", "ucret_a", "gamma", SHAPE, SCALE_A, "A grubunda ücret: Gamma(4; 1,125)"),
                Draw("orneklem", "ucret_b", "gamma", SHAPE, SCALE_B, "B grubunda ücret: Gamma(4; 1,875)"),
                Derive("orneklem", "b_grubu", E.compare("lt", E.var("u"), p),
                       f"B grubundan mı? (1: evet); olasılık p = {plain(p, 2)}"),
                Derive("orneklem", "ucret", E.add(E.mul(E.var("b_grubu"), E.var("ucret_b")),
                                                  E.mul(E.sub(1, E.var("b_grubu")), E.var("ucret_a"))),
                       "Gözlenen ücret: grubunun dağılımından"),
                Statistic("orneklem", "ucret", "mean", "xbar", "Örneklem ortalaması x̄"),
            ),
            (
                ("xbar", xbar),
                ("yakin", E.compare("lt", E.absolute(E.sub(xbar, MU)), CLOSE)),
            ),
            "Her tekrarda anakütleden n gözlemlik yeni bir örneklem çekilir ve ortalaması alınır",
        ),
        Histogram("tekrarlar", (("xbar", "Örneklem ortalamaları x̄"),), BINS_1, AXIS_1[0], AXIS_1[1],
                  f"{reps} örneklemin ortalaması (n = {n})", "Örneklem ortalaması x̄ (ücret)",
                  references=tuple(references), y_label="Örneklem sayısı"),
    )


def _sampling_dgp(parameters: Parameters) -> tuple[str, ...]:
    n, strength, reps = _sampling_settings(parameters)
    p = sampled_share(strength)
    return (
        r"\text{Anakütle: iki eşit grup; A'da } X \sim \operatorname{Gamma}(4;\ 1{,}125),\ \mu_A = 4{,}5; \quad "
        r"\text{B'de } X \sim \operatorname{Gamma}(4;\ 1{,}875),\ \mu_B = 7{,}5; \quad \mu = 6",
        rf"P(\text{{gözlem B'den}}) = p = \frac{{1 + s}}{{2}} = {number(p, 2)}, \qquad s = {_tex(strength, 1)}, "
        rf"\qquad n = {n}, \qquad {reps} \text{{ örneklem}}",
        rf"\mathbb{{E}}(\bar X) = (1 - p)\,4{{,}}5 + p\,7{{,}}5 = {number(expected_mean(strength), 2)}, \qquad "
        rf"\text{{yanlılık}}\ \mathbb{{E}}(\bar X) - \mu = 1{{,}}5\,s = {number(expected_mean(strength) - MU, 2)}",
    )


def _sampling_summary(state: LabState) -> tuple[float, float, float]:
    table = state.tables["tekrarlar"]
    return float(table["xbar"].mean()), float(table["xbar"].std()), float(table["yakin"].mean())


def _sampling_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    _, strength, _ = _sampling_settings(parameters)
    mean, sd, close = _sampling_summary(state)
    return (
        SimMetric("x̄'lerin ortalaması", plain(mean, 3),
                  f"Kuramsal değer E(X̄) = {plain(expected_mean(strength), 2)}; anakütle ortalaması μ = 6."),
        SimMetric("Yanlılık E(X̄) − μ", plain(expected_mean(strength) - MU, 2),
                  f"Kuramsal yanlılık 1,5·s. Simülasyonda x̄'lerin ortalaması − μ = {plain(mean - MU, 3)}."),
        SimMetric("x̄'lerin standart sapması", plain(sd, 3),
                  "Aynı büyüklükteki örneklemlerin ortalamaları ne kadar yayılıyor? n dört katına çıkınca yaklaşık "
                  "yarıya iner."),
        SimMetric("μ'ye 0,5'ten yakın", percent(100 * close, 1),
                  "|x̄ − μ| < 0,5 olan örneklemlerin payı: tek bir örneklemin tahmini ne sıklıkla μ'ye yakın?"),
    )


def _sampling_takeaway(state: LabState, parameters: Parameters) -> str:
    _, strength, reps = _sampling_settings(parameters)
    mean, sd, close = _sampling_summary(state)
    spread = (f"{reps} örneklemin her biri aynı kuralla (X̄) farklı bir tahmin verir: bu örneklemden örnekleme "
              f"değişimdir. x̄'lerin standart sapması {plain(sd, 3)}; n dört katına çıkınca (ör. 50'den 200'e) "
              "yaklaşık yarıya iner.")
    if close == 0:
        closeness = "Hiçbir örneklemin ortalaması μ'ye 0,5'ten yakın değildir."
    elif close == 1:
        closeness = "Bütün örneklemlerin ortalaması μ'ye 0,5'ten yakındır."
    else:
        closeness = f"Örneklemlerin {percent(100 * close, 1)} kadarının ortalaması μ'ye 0,5'ten yakındır."
    if strength == 0:
        return (
            f"Basit rastgele örneklemede her gözlem iki gruptan eşit olasılıkla gelir: x̄'ler μ = 6 çevresinde "
            f"toplanır (ortalamaları {plain(mean, 3)}). {spread} {closeness} Seçim gücü s'yi artırarak ne "
            "değiştiğine bakın (§0.2, §0.8)."
        )
    group = ("Yalnız B grubu örnekleme girer (p = 1)" if strength == 1 else
             f"Seçimli örneklemede B grubu fazla temsil edilir (p = {plain(sampled_share(strength), 2)})")
    return (
        f"{group}: x̄'ler μ = 6 yerine E(X̄) = {plain(expected_mean(strength), 2)} çevresinde toplanır; yanlılık "
        f"E(X̄) − μ = {plain(expected_mean(strength) - MU, 2)} (simülasyonda {plain(mean - MU, 3)}). {spread} n'yi "
        f"büyütmek yayılımı azaltır ama yanlılığı gidermez: histogram daralır, merkezi kaymış kalır. {closeness} "
        "Örneklemin hangi anakütleyi temsil ettiği, büyüklüğünden önce gelir (§0.2)."
    )


SAMPLING = SimExperiment(
    topic_key=TOPIC,
    number=1,
    title="Örneklem ortalaması: örnekleme değişimi ve seçim yanlılığı",
    question="Anakütledeki ortalama ücret μ bilinmez; örneklem ortalaması x̄ onu tahmin eder. Aynı kuralı farklı "
             "örneklemlere uygulayınca tahminler ne kadar değişir; örneklem büyüdükçe ne olur; örneklem sistematik "
             "biçimde seçilirse büyük bir örneklem yeterli olur mu?",
    note=NoteRef("0.2", objects=("§0.8",)),
    parameters=(
        SimParameter("n", "Örneklem büyüklüğü n", 10, 500, 50, 10, "Her örneklemdeki gözlem sayısı.", integer=True,
                     decimals=0),
        SimParameter("secim", "Seçim gücü s", 0, 1, 0, 0.1,
                     "s = 0: basit rastgele örnekleme. s büyüdükçe B grubu (yüksek ücretliler) fazla temsil edilir; "
                     "s = 1: yalnız B grubu.", decimals=1),
        SimParameter("tekrar", "Örneklem sayısı", 500, 2000, 1000, 500, "Her tekrarda yeni bir örneklem çekilir.",
                     integer=True, decimals=0),
    ),
    dgp=_sampling_dgp,
    dgp_note=(
        "Anakütle eşit büyüklükte iki gruptan oluşur; ücretler gamma dağılımıyla üretilir (pozitif ve sağa çarpık, "
        "WAGE1'deki ücrete benzer). Anakütle ortalaması μ = 6 simülasyonda bilinir; gerçek araştırmada bilinmez. "
        "Seçim gücü s, örnekleme giren kişilerin B grubundan gelme olasılığını artırır (ör. anketin yalnız yüksek "
        "ücretli bölgelerde yapılması). Tohum 305."
    ),
    look_at=(
        "**Histogram** — her çubuk, aynı büyüklükteki örneklemlerden kaçının o x̄ değerini verdiğini gösterir; kesikli "
        "çizgiler μ = 6 ve (s > 0 iken) E(X̄).",
        "**Metrikler** — x̄'lerin ortalaması μ'ye eşit mi (yanlılık), yayılımı n ile nasıl değişiyor?",
        "**Deneyin** — önce n'yi 10'dan 500'e çıkarın, sonra s'yi artırıp n'yi yeniden büyütün.",
    ),
    build=_build_sampling,
    metrics=_sampling_metrics,
    takeaway=_sampling_takeaway,
    labels=(("xbar", "Örneklem ortalaması x̄"), ("yakin", "μ'ye 0,5'ten yakın (1/0)"), ("u", "Tek-düze sayı"),
            ("ucret_a", "A grubunda ücret"), ("ucret_b", "B grubunda ücret"), ("b_grubu", "B grubu (1/0)"),
            ("ucret", "Ücret")),
)


# --- Deney 2: korelasyon ve doğrusallık -------------------------------------------------------------------------

VAR_X = 25 / 12
"""x ~ U(0, 5) için Var(x)."""
VAR_SQUARE = 125 / 36
"""Var((x − 2,5)²), x ~ U(0, 5); Cov(x, (x − 2,5)²) = 0 (simetri)."""


def population_rho(beta: float, gamma: float, sigma: float) -> float | None:
    """Anakütle korelasyonu ρ(x, y); y sabitse (β = γ = σ = 0) tanımsız (``None``)."""

    total = beta ** 2 * VAR_X + gamma ** 2 * VAR_SQUARE + sigma ** 2
    return None if total == 0 else beta * math.sqrt(VAR_X) / math.sqrt(total)


def _linearity_settings(parameters: Parameters) -> tuple[float, float, float, int, int]:
    return (_rounded(parameters, "beta", 2), _rounded(parameters, "gamma", 2), _rounded(parameters, "sigma", 2),
            int(parameters["olcek"]), int(parameters["n"]))


def _build_linearity(parameters: Parameters) -> tuple:
    beta, gamma, sigma, scale, n = _linearity_settings(parameters)
    signal = E.add(E.mul(beta, E.var("x")), E.mul(gamma, E.power(E.sub(E.var("x"), 2.5), 2)))
    operations = [
        NewSample("veri", n, SEED),
        Draw("veri", "x", "uniform", 0, 5, "x ~ U(0, 5)"),
        Draw("veri", "u", "normal", 0, sigma, f"Gürültü u ~ N(0; {plain(sigma, 2)}²)"),
        Derive("veri", "y", E.mul(scale, E.add(signal, E.var("u"))),
               f"y = {scale}·[β·x + γ·(x − 2,5)² + u]"),
        PairStatistic("veri", "x", "y", "cov", "kovaryans", "Örneklem kovaryansı s_xy", decimals=4),
    ]
    if population_rho(beta, gamma, sigma) is not None:
        operations.append(PairStatistic("veri", "x", "y", "corr", "r", "Örneklem korelasyonu r", decimals=4))
    operations.append(ScatterPlot("veri", "x", "y", "x", f"y (ölçek a = {scale})",
                                  "Aynı veri, iki ölçü: kovaryans ve korelasyon", size=8, opacity=0.7))
    return tuple(operations)


def _linearity_dgp(parameters: Parameters) -> tuple[str, ...]:
    beta, gamma, sigma, scale, n = _linearity_settings(parameters)
    rho = population_rho(beta, gamma, sigma)
    rho_text = r"\text{tanımsız (y sabit)}" if rho is None else number(rho, 3)
    return (
        rf"y_i = a\,[\beta\,x_i + \gamma\,(x_i - 2{{,}}5)^2 + u_i], \qquad x_i \sim U(0,\ 5), \qquad "
        rf"u_i \sim N(0,\ \sigma^2), \qquad i = 1, \dots, n = {n}",
        rf"\beta = {_tex(beta)}, \quad \gamma = {_tex(gamma)}, \quad \sigma = {_tex(sigma)}, \quad a = {scale}",
        r"\rho(x, y) = \frac{\beta\,\sqrt{\operatorname{Var}(x)}}{\sqrt{\beta^2 \operatorname{Var}(x) + \gamma^2 "
        rf"\operatorname{{Var}}((x - 2{{,}}5)^2) + \sigma^2}}}} = {rho_text}",
    )


def _linearity_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    beta, gamma, sigma, scale, _ = _linearity_settings(parameters)
    rho = population_rho(beta, gamma, sigma)
    s = state.scalars
    r_text = "tanımsız" if rho is None else plain(s["r"], 3)
    return (
        SimMetric("Örneklem r", r_text, "Örneklem korelasyonu r = s_xy / (s_x·s_y)."),
        SimMetric("Anakütle ρ", "tanımsız" if rho is None else plain(rho, 3),
                  "Veri üretim sürecinden hesaplanan korelasyon. Payı yalnız β'ya bağlıdır; γ ve σ paydayı büyütür."),
        SimMetric("Kovaryans", plain(s["kovaryans"], 3),
                  "Örneklem kovaryansı. a = 1: y özgün biriminde; a değişince kovaryans a ile çarpılır, korelasyon "
                  "değişmez." if scale == 1 else
                  f"Örneklem kovaryansı; y'nin bütün değerleri a = {scale} ile çarpıldığı için kovaryans da a ile "
                  "çarpılır."),
        SimMetric("Ölçek a", str(scale), "y'nin bütün değerleri a ile çarpılır (ör. dolar yerine sent: a = 100)."),
    )


def _sample_r(r: float, rho: float) -> str:
    """Örneklem r'sinin anakütle ρ'suna göre durumu; karşılaştırma ekranda görünen (üç basamaklı) değerlerle yapılır."""

    shown, target = round(r * 1000), round(rho * 1000)
    text = f" Bu örneklemde r = {plain(r, 3)}"
    if abs(shown - target) < 100:
        return text + ", ρ'ya yakın."
    if shown == 0:
        return text + (": örneklemde doğrusal ilişki neredeyse yok; örneklem r'si anakütle ρ'sundan şans eseri sapar, "
                       "n büyüdükçe sapma küçülür.")
    if shown * target < 0:
        return text + (": işareti bile ρ'nunkinden farklıdır. Örneklem r'si anakütle ρ'sundan şans eseri sapar; küçük "
                       "örneklemde ya da ρ sıfıra yakınken işaret de değişebilir. n büyüdükçe sapma küçülür.")
    return text + ": örneklem r'si anakütle ρ'sundan şans eseri sapar; n büyüdükçe sapma küçülür."


def _linearity_takeaway(state: LabState, parameters: Parameters) -> str:
    beta, gamma, sigma, scale, _ = _linearity_settings(parameters)
    rho = population_rho(beta, gamma, sigma)
    s = state.scalars
    if rho is None:
        return ("β = γ = σ = 0: y her gözlemde aynı değeri alır. Değişkenliği olmayan bir değişkenin korelasyonu "
                "tanımsızdır (payda sıfır), kovaryansı sıfırdır. Bir kaydırıcıyı değiştirin (§0.5).")
    unit = (f" Ölçek a = {scale}: kovaryans a ile çarpılır ({plain(s['kovaryans'], 3)}), korelasyon değişmez; "
            "korelasyon ölçü biriminden bağımsızdır." if scale != 1 else
            " Ölçeği (a) değiştirip kovaryansın ve korelasyonun nasıl davrandığına bakın.")
    sample = _sample_r(s["r"], rho)
    if beta == 0 and gamma > 0:
        curve = gamma ** 2 * VAR_SQUARE
        share = percent(100 * curve / (curve + sigma ** 2), 1)
        return (f"β = 0, γ = {_short(gamma)}: y ile x arasında U biçimli (doğrusal olmayan) bir ilişki vardır; y'nin "
                f"anakütle varyansında bu bileşenin payı {share}. Eğrilik x ile doğrusal olarak ilişkisiz olduğu için "
                f"anakütlede ρ = 0.{sample} ρ = 0 “doğrusal ilişki yok” demektir, “ilişki yok” demek değildir "
                f"(§0.5).{unit}")
    if beta == 0:
        return f"β = γ = 0: y yalnız gürültüdür ve x'ten bağımsızdır; anakütlede ρ = 0 (§0.5).{sample}{unit}"
    if gamma > 0:
        linear = population_rho(beta, 0, sigma)
        return (f"Doğrusal kısım (β = {_short(beta)}) ile eğrilik (γ = {_short(gamma)}) birlikte. Anakütlede eğrilik "
                f"kovaryansa katkı yapmaz, yalnız y'nin varyansını büyütür: ρ = {plain(rho, 3)} (γ = 0 olsaydı "
                f"{plain(linear, 3)}).{sample} Örneklemde x'ler 2,5 çevresinde tam simetrik dağılmadığı için eğrilik "
                f"r'yi iki yöne de itebilir. Saçılım grafiğine bakmadan korelasyon yorumlanmaz (§0.5).{unit}")
    return (f"Doğrusal ilişki (β = {_short(beta)}): anakütlede ρ = {plain(rho, 3)}.{sample} Gürültü σ büyüdükçe "
            f"noktalar doğrunun çevresinde daha geniş dağılır ve |ρ| küçülür; işaret yönü, mutlak değer doğrusal "
            f"ilişkinin gücünü gösterir (§0.5).{unit}")


LINEARITY = SimExperiment(
    topic_key=TOPIC,
    number=2,
    title="Korelasyon yalnız doğrusal ilişkiyi ölçer",
    question="İki değişken arasında güçlü bir ilişki varken korelasyon sıfıra yakın çıkabilir mi? Ölçü birimi "
             "değişince kovaryans ve korelasyon nasıl davranır?",
    note=NoteRef("0.5", objects=("Şekil 0.2",)),
    parameters=(
        SimParameter("beta", "Doğrusal eğim β", -2, 2, 1, 0.25, "β = 0: doğrusal bileşen yok.", decimals=2),
        SimParameter("gamma", "Eğrilik γ", 0, 2, 0, 0.25, "γ > 0: U biçimli (karesel) bileşen.", decimals=2),
        SimParameter("sigma", "Gürültü σ", 0, 3, 0.5, 0.25, "Noktaların ortalama ilişki çevresindeki dağılımı.",
                     decimals=2),
        SimParameter("olcek", "Ölçek a (y'nin birimi)", 1, 100, 1, 1, "a = 100: ör. dolar yerine sent.",
                     integer=True, decimals=0),
        SimParameter("n", "Gözlem sayısı n", 20, 500, 100, 10, "Örneklemdeki gözlem sayısı.", integer=True,
                     decimals=0),
    ),
    dgp=_linearity_dgp,
    dgp_note=(
        "x, 0 ile 5 arasında tek-düze dağılır; y doğrusal bir bileşen (β·x), x = 2,5 çevresinde simetrik bir karesel "
        "bileşen (γ·(x − 2,5)²) ve gürültüden (u) oluşur. Karesel bileşen x ile doğrusal olarak ilişkisizdir: ρ'nun "
        "payı yalnız β'ya bağlıdır (β = 0 ise ρ = 0); γ ve σ yalnız paydayı büyütür, bu yüzden |ρ| küçülür. Tohum "
        "305."
    ),
    look_at=(
        "**Saçılım grafiği** — noktalar bir doğru çevresinde mi, bir eğri çevresinde mi toplanıyor?",
        "**Metrikler** — örneklem r ile anakütle ρ; ölçek a değişince kovaryans ve r.",
        "**Deneyin** — σ = 0,5 iken β = 0 ve γ = 1 yapın: güçlü ama doğrusal olmayan ilişki. Sonra a'yı 100 yapın.",
    ),
    build=_build_linearity,
    metrics=_linearity_metrics,
    takeaway=_linearity_takeaway,
    labels=(("x", "x"), ("y", "y"), ("u", "Gürültü u")),
)


# --- Deney 3: koşullu ortalama ----------------------------------------------------------------------------------

LEVELS = (8, 10, 12, 14, 16, 18)
"""X: eğitim yılı; altı düzey eşit olasılıklı."""
INTERCEPT_3, SLOPE_3, CENTER_3 = 1.0, 0.5, 13
"""E(Y | X) = 1 + 0,5·X + γ·(X − 13)²: WAGE1'deki koşullu ortalamalara benzer düzeyler (dolar)."""


def conditional_mean(x: float, gamma: float) -> float:
    return INTERCEPT_3 + SLOPE_3 * x + gamma * (x - CENTER_3) ** 2


def unconditional_mean(gamma: float) -> float:
    """E(Y) = altı düzeydeki E(Y | X) değerlerinin ortalaması (düzeyler eşit olasılıklı)."""

    return sum(conditional_mean(level, gamma) for level in LEVELS) / len(LEVELS)


def _conditional_settings(parameters: Parameters) -> tuple[int, float, float]:
    return int(parameters["n"]), _rounded(parameters, "sigma", 1), _rounded(parameters, "gamma", 2)


def _true_mean_expr(gamma: float) -> E.Expr:
    line = E.add(INTERCEPT_3, E.mul(SLOPE_3, E.var("x")))
    if gamma == 0:  # kodda "0 * (x - 13) ** 2" yazılmasın
        return line
    return E.add(line, E.mul(gamma, E.power(E.sub(E.var("x"), CENTER_3), 2)))


def _by_level(prefix: str) -> E.Expr:
    """Düzey tablosunda her satıra kendi düzeyinin skalerini yazan ifade: Σ 1(x = düzey)·skaler."""

    total: E.Expr | None = None
    for level in LEVELS:
        term = E.mul(E.compare("eq", E.var("x"), level), E.ref(f"{prefix}_{level}"))
        total = term if total is None else E.add(total, term)
    return total


def _build_conditional(parameters: Parameters) -> tuple:
    n, sigma, gamma = _conditional_settings(parameters)
    per_level = []
    for level in LEVELS:
        per_level += [
            Statistic("orneklem", "y", "mean", f"ort_{level}", f"X = {level}: örneklem ortalaması", where=("x", level),
                      decimals=3),
            Statistic("orneklem", "y", "count", f"n_{level}", f"X = {level}: gözlem sayısı", where=("x", level),
                      decimals=0),
        ]
    return (
        NewSample("orneklem", n, SEED),
        DrawDiscrete("orneklem", "x", LEVELS, tuple(1 / len(LEVELS) for _ in LEVELS),
                     "Eğitim yılı X: 8, 10, …, 18 eşit olasılıkla"),
        Draw("orneklem", "u", "normal", 0, sigma, f"Diğer etkiler u ~ N(0; {plain(sigma, 1)}²)"),
        Derive("orneklem", "y", E.add(_true_mean_expr(gamma), E.var("u")),
               "Ücret Y = 1 + 0,5·X + u" if gamma == 0 else f"Ücret Y = 1 + 0,5·X + {plain(gamma, 2)}·(X − 13)² + u"),
        Statistic("orneklem", "y", "mean", "y_ortalama", "Koşulsuz örneklem ortalaması ȳ", decimals=3),
        *per_level,
        InlineData("duzeyler", ("x",), tuple((level,) for level in LEVELS), "Eğitim düzeyleri", layout=len(LEVELS)),
        Derive("duzeyler", "gozlem", _by_level("n"), "Düzeydeki gözlem sayısı"),
        Derive("duzeyler", "orneklem_ort", _by_level("ort"), "Örneklem koşullu ortalaması"),
        Derive("duzeyler", "gercek", _true_mean_expr(gamma), "Gerçek E(Y | X)"),
        Derive("duzeyler", "fark", E.sub(E.var("orneklem_ort"), E.var("gercek")), "Örneklem − gerçek"),
        ShowFrame("duzeyler", ("x", "gozlem", "orneklem_ort", "gercek", "fark"),
                  "Her eğitim düzeyinde örneklem ortalaması ve gerçek E(Y | X)", decimals=3),
        LineChart("duzeyler", "x", "orneklem_ort", "Eğitim yılı X", "Ücret (dolar)",
                  "Koşullu ortalamalar: örneklem ve gerçek E(Y | X)",
                  series=(("gercek", "Gerçek E(Y | X)"),), legend="Örneklem koşullu ortalaması"),
        ScatterPlot("orneklem", "x", "y", "Eğitim yılı X", "Ücret Y (dolar)",
                    "Gözlemler: aynı X'te farklı Y değerleri", size=6, opacity=0.3),
    )


def _conditional_dgp(parameters: Parameters) -> tuple[str, ...]:
    n, sigma, gamma = _conditional_settings(parameters)
    return (
        rf"Y_i = 1 + 0{{,}}5\,X_i + \gamma\,(X_i - 13)^2 + u_i, \qquad \gamma = {_tex(gamma)}, \qquad "
        rf"u_i \sim N(0,\ \sigma^2),\ \sigma = {_tex(sigma, 1)}",
        rf"X_i \in \{{8, 10, 12, 14, 16, 18\}} \text{{ eşit olasılıkla}}, \qquad i = 1, \dots, n = {n}",
        rf"\mathbb{{E}}(Y \mid X = x) = 1 + 0{{,}}5\,x + \gamma\,(x - 13)^2, \qquad \mathbb{{E}}(Y) = "
        rf"{number(unconditional_mean(gamma), 3)}",
    )


def _conditional_summary(state: LabState) -> tuple[float, int, float]:
    frame = state.frames["duzeyler"]
    worst = float(frame["fark"].abs().max())
    smallest = int(frame["gozlem"].min())
    return worst, smallest, float(state.scalars["y_ortalama"])


def _conditional_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    _, _, gamma = _conditional_settings(parameters)
    worst, smallest, ybar = _conditional_summary(state)
    return (
        SimMetric("ȳ (koşulsuz)", plain(ybar, 3), f"Gerçek E(Y) = {plain(unconditional_mean(gamma), 3)}."),
        SimMetric("E(Y | X = 16)", plain(conditional_mean(16, gamma), 3),
                  f"Örneklemde X = 16 olanların ortalaması: {plain(state.scalars['ort_16'], 3)}."),
        SimMetric("En büyük fark", plain(worst, 3),
                  "Altı düzeyde |örneklem koşullu ortalaması − gerçek E(Y | X)| değerlerinin en büyüğü."),
        SimMetric("En küçük grup", str(smallest), "En az gözlemi olan eğitim düzeyindeki gözlem sayısı."),
    )


def _conditional_takeaway(state: LabState, parameters: Parameters) -> str:
    n, sigma, gamma = _conditional_settings(parameters)
    worst, smallest, ybar = _conditional_summary(state)
    text = (f"Her eğitim düzeyindeki gözlemlerin ortalaması, o düzeydeki gerçek E(Y | X) değerini tahmin eder; en "
            f"büyük fark {plain(worst, 3)} dolar. Düzey başına yaklaşık {round(n / len(LEVELS))} gözlem düşer (en "
            f"küçük grup {smallest}); n büyüdükçe ve σ küçüldükçe örneklem çizgisi gerçek çizgiye yaklaşır. Koşulsuz "
            f"ortalama ȳ = {plain(ybar, 3)} bütün düzeylerin tek bir özetidir; koşullu ortalama ise X ile değişir. ")
    if gamma == 0:
        return text + ("Bu ayarda E(Y | X) bir doğrudur. γ'yı artırarak koşullu ortalamanın eğri olduğu durumu "
                       "deneyin: basit doğrusal regresyon E(Y | X)'in bir doğruyla ne ölçüde temsil edilebildiğini "
                       "araştırır (§0.7).")
    return text + (f"γ = {_short(gamma)}: gerçek E(Y | X) bir doğru değil, eğridir. Bir doğru bu ilişkiyi ancak "
                   "yaklaşık olarak özetleyebilir; basit doğrusal regresyonun sorusu, E(Y | X)'in bir doğruyla ne "
                   "ölçüde temsil edilebildiğidir (§0.7).")


CONDITIONAL = SimExperiment(
    topic_key=TOPIC,
    number=3,
    title="Koşullu ortalama: E(Y | X) ve örneklemdeki grup ortalamaları",
    question="Anakütlede her eğitim düzeyinin ortalama ücreti E(Y | X) bilinseydi, örneklemdeki grup ortalamaları "
             "ona ne kadar yakın olurdu? Koşulsuz ortalama ile koşullu ortalama arasındaki fark nedir?",
    note=NoteRef("0.7"),
    parameters=(
        SimParameter("n", "Gözlem sayısı n", 60, 2000, 300, 20, "Altı eğitim düzeyine yaklaşık eşit dağılır.",
                     integer=True, decimals=0),
        SimParameter("sigma", "Diğer etkilerin standart sapması σ", 0.5, 4, 2, 0.5,
                     "Aynı eğitim düzeyindeki ücretlerin yayılımı.", decimals=1),
        SimParameter("gamma", "Eğrilik γ", 0, 0.2, 0, 0.02, "γ = 0: E(Y | X) bir doğrudur.", decimals=2),
    ),
    dgp=_conditional_dgp,
    dgp_note=(
        "Eğitim yılı X altı düzeyden birini eşit olasılıkla alır; ücret Y, düzeyin gerçek koşullu ortalamasına diğer "
        "etkilerin (u) eklenmesiyle oluşur. γ = 0 iken koşullu ortalamalar WAGE1'dekilerle aynı büyüklüktedir (8 "
        "yılda 5, 18 yılda 10 dolar); WAGE1'deki grup ortalamaları ise bir doğru üzerinde değildir. Basitlik için u "
        "normal dağılımla üretilir; bu yüzden az sayıda gözlemde ücret negatif çıkabilir. Deneyin konusu "
        "ortalamalardır; bu basitleştirme sonucu değiştirmez. Tohum 305."
    ),
    look_at=(
        "**Tablo** — her eğitim düzeyinde gözlem sayısı, örneklem ortalaması, gerçek E(Y | X) ve fark.",
        "**Çizgi grafiği** — iki seri: örneklem koşullu ortalamaları ve gerçek E(Y | X) (lejanttaki renklerle).",
        "**Deneyin** — n'yi küçültün ve σ'yı artırın; sonra γ'yı artırıp E(Y | X)'in eğrileştiğini izleyin.",
    ),
    build=_build_conditional,
    metrics=_conditional_metrics,
    takeaway=_conditional_takeaway,
    labels=(("x", "Eğitim yılı X"), ("y", "Ücret Y (dolar)"), ("u", "Diğer etkiler u"),
            ("gozlem", "Gözlem sayısı"), ("orneklem_ort", "Örneklem ortalaması"), ("gercek", "Gerçek E(Y | X)"),
            ("fark", "Fark")),
)


KONU00_EXPERIMENTS = (SAMPLING, LINEARITY, CONDITIONAL)
