"""Konu 9 Sezgi deneyleri: veri üretim süreci bilinen kontrollü simülasyonlar.

Deney 1  Standartlaştırılmış katsayı örneklem değişkenliğine bağlıdır      (Notlar §9.2)
Deney 2  Dönüm noktası veri aralığının dışında kalırsa                     (Notlar §9.4–9.5)
Deney 3  Yanlış fonksiyonel biçim: eğri ilişkiye düz çizgi                 (Notlar §9.9)

Bölüm 9'da benzetim yoktur; deneylerin notlarda sayısal karşılığı yoktur. Varsayılan ayarlar bölümün kavramlarını
göstermek için seçilmiştir; her deney ve senaryo tohum 305 ile başlar.
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
    Histogram,
    JoinColumns,
    ModelValue,
    MonteCarlo,
    NewSample,
    NoteRef,
    Scalar,
    ScalarTable,
    ScatterPlot,
    Statistic,
    SummaryTable,
    Support,
)

SEED = 305
TOPIC = "konu09"


def _rounded(parameters: Parameters, key: str, decimals: int) -> float:
    """Kaydırıcı değeri, kodda 0,30000000000000004 gibi yazımlar olmasın diye yuvarlanır."""

    return round(float(parameters[key]), decimals)


def _tex(value: float, decimals: int = 2) -> str:
    """LaTeX içinde sondaki sıfırları atılmış sayı (0{,}5; 3)."""

    text = f"{value:.{decimals}f}"
    text = text.rstrip("0").rstrip(".") if "." in text else text  # 20 → "2" olmasın
    return "0" if text in ("-0", "") else text.replace(".", "{,}")


def _short(value: float, decimals: int = 2) -> str:
    """Sondaki sıfırları atılmış düz metin sayı (0,5; 3); yuvarlanınca sıfır olan değer "0" yazılır."""

    text = f"{value:.{decimals}f}"
    text = text.rstrip("0").rstrip(".") if "." in text else text  # 20 → "2" olmasın
    return "0" if text in ("-0", "0", "") else text.replace(".", ",").replace("-", "−")


def _thousands(value: int) -> str:
    return f"{value:,}".replace(",", ".")


# --- Deney 1: standartlaştırılmış katsayı ve örneklem değişkenliği ---------------------------------------------

SLOPE_1 = 0.5
"""Y = 1 + 0,5X₁ + 0,5X₂ + u: iki değişkenin gerçek eğimi aynıdır."""


def _std_settings(parameters: Parameters) -> tuple[float, float, int]:
    return _rounded(parameters, "s1", 2), _rounded(parameters, "s2", 2), int(parameters["n"])


def population_standardized(s1: float, s2: float) -> tuple[float, float, float]:
    """Anakütle standart sapması σ_Y = √(0,25σ₁² + 0,25σ₂² + 1) ve iki standartlaştırılmış katsayı 0,5·σⱼ/σ_Y."""

    sd_y = math.sqrt(SLOPE_1 ** 2 * s1 ** 2 + SLOPE_1 ** 2 * s2 ** 2 + 1)
    return sd_y, SLOPE_1 * s1 / sd_y, SLOPE_1 * s2 / sd_y


def _build_standardized(parameters: Parameters) -> tuple:
    s1, s2, n = _std_settings(parameters)
    sd_y, g1, g2 = population_standardized(s1, s2)
    return (
        NewSample("veri", n, SEED),
        Draw("veri", "x1", "normal", 0, s1, f"X₁ ~ N(0, {_short(s1)}²)"),
        Draw("veri", "x2", "normal", 0, s2, f"X₂ ~ N(0, {_short(s2)}²)"),
        Draw("veri", "u", "normal", 0, 1, "u ~ N(0, 1)"),
        Derive("veri", "y", E.add(E.add(E.add(1, E.mul(SLOPE_1, E.var("x1"))), E.mul(SLOPE_1, E.var("x2"))), E.var("u")),
               "Y = 1 + 0,5X₁ + 0,5X₂ + u: iki eğim de 0,5"),
        OLS("model", "veri", "y", ("x1", "x2"), "EKK: y ~ x1 + x2"),
        ModelValue("b1", "model", "coef", "X₁'in ham eğimi β̂₁", term="x1", decimals=3),
        ModelValue("b2", "model", "coef", "X₂'nin ham eğimi β̂₂", term="x2", decimals=3),
        Statistic("veri", "y", "std", "s_y", "Y'nin örneklem standart sapması", decimals=3),
        Statistic("veri", "x1", "std", "s_x1", "X₁'in örneklem standart sapması", decimals=3),
        Statistic("veri", "x2", "std", "s_x2", "X₂'nin örneklem standart sapması", decimals=3),
        Scalar("z1", E.div(E.mul(E.ref("b1"), E.ref("s_x1")), E.ref("s_y")),
               "X₁'in standartlaştırılmış katsayısı β̂₁·s_X₁/s_Y (standartlaştırılmış değişkenlerle EKK ile aynı)",
               decimals=3),
        Scalar("z2", E.div(E.mul(E.ref("b2"), E.ref("s_x2")), E.ref("s_y")),
               "X₂'nin standartlaştırılmış katsayısı β̂₂·s_X₂/s_Y", decimals=3),
        Scalar("g1", E.const(round(g1, 12)), "Anakütlede X₁'in standartlaştırılmış katsayısı 0,5·σ₁/σ_Y", decimals=3),
        Scalar("g2", E.const(round(g2, 12)), "Anakütlede X₂'nin standartlaştırılmış katsayısı 0,5·σ₂/σ_Y", decimals=3),
        ScalarTable((("X₁", E.const(SLOPE_1)), ("X₂", E.const(SLOPE_1))), "gercek", decimals=3, heading="Değişken"),
        ScalarTable((("X₁", E.ref("b1")), ("X₂", E.ref("b2"))), "ham", decimals=3, heading="Değişken"),
        ScalarTable((("X₁", E.ref("z1")), ("X₂", E.ref("z2"))), "standart", decimals=3, heading="Değişken",
                    value="Standartlaştırılmış katsayı"),
        ScalarTable((("X₁", E.ref("g1")), ("X₂", E.ref("g2"))), "anakutle", decimals=3, heading="Değişken"),
        JoinColumns("katsayilar", (("Gerçek eğim", "gercek", "deger"), ("Ham eğim tahmini", "ham", "deger"),
                                   ("Standartlaştırılmış (anakütle)", "anakutle", "deger"),
                                   ("Standartlaştırılmış (örneklem)", "standart", "deger")), decimals=3,
                    heading="Değişken"),
        BarChart("standart", "deger", "Değişken", "Standartlaştırılmış katsayı",
                 f"Aynı gerçek eğim (0,5), aynı yayılım: σ₁ = σ₂ = {_short(s1)}" if s1 == s2
                 else f"Aynı gerçek eğim (0,5), farklı yayılım: σ₁ = {_short(s1)}, σ₂ = {_short(s2)}", decimals=3),
    )


def _std_dgp(parameters: Parameters) -> tuple[str, ...]:
    s1, s2, n = _std_settings(parameters)
    return (
        r"Y_i = 1 + 0{,}5\,X_{1i} + 0{,}5\,X_{2i} + u_i, \qquad u_i \sim N(0,\ 1), \qquad " + rf"n = {n}",
        rf"X_{{1i}} \sim N(0,\ {_tex(s1)}^2), \qquad X_{{2i}} \sim N(0,\ {_tex(s2)}^2)",
        r"\beta_j^* = \beta_j\,\frac{\sigma_j}{\sigma_Y}, \qquad \sigma_Y = \sqrt{0{,}25\,\sigma_1^2 + 0{,}25\,\sigma_2^2 + 1}",
    )


def _std_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    s = state.scalars
    return (
        SimMetric("β̂₁ (ham)", plain(s["b1"], 3), "X₁'in ham eğimi; gerçek değer 0,5."),
        SimMetric("β̂₂ (ham)", plain(s["b2"], 3), "X₂'nin ham eğimi; gerçek değer 0,5."),
        SimMetric("β̂₁* (standart)", plain(s["z1"], 3), "X₁'de bir standart sapmalık artışın Y'de kaç standart sapmalık "
                  "değişimle ilişkili olduğu."),
        SimMetric("β̂₂* (standart)", plain(s["z2"], 3), "X₂'de bir standart sapmalık artışın Y'de kaç standart sapmalık "
                  "değişimle ilişkili olduğu."),
    )


def _std_takeaway(state: LabState, parameters: Parameters) -> str:
    s1, s2, _ = _std_settings(parameters)
    s = state.scalars
    close = abs(s["b1"] - SLOPE_1) < 0.1 and abs(s["b2"] - SLOPE_1) < 0.1
    text = (f"İki değişkenin gerçek eğimi aynıdır (0,5); ham tahminler {plain(s['b1'], 3)} ve {plain(s['b2'], 3)}"
            + (". " if close else ": yayılımı dar bir değişkenin eğimi az bilgiyle tahmin edilir ve gerçek değerden (0,5) sapabilir. "))
    text += f"Standartlaştırılmış katsayılar {plain(s['z1'], 3)} ve {plain(s['z2'], 3)}; anakütle değerleri "
    text += f"{plain(s['g1'], 3)} ve {plain(s['g2'], 3)}. "
    if s1 == s2:
        text += ("İki değişkenin yayılımı aynı olduğu için anakütlede standartlaştırılmış katsayılar eşittir; örneklemdeki "
                 "fark tahmin belirsizliğinden gelir. σ₁ ile σ₂'yi farklılaştırın: daha çok değişen değişken daha "
                 "‘önemli’ görünür. ")
    else:
        larger = "X₂" if s2 > s1 else "X₁"
        text += (f"Anakütlede daha çok değişen {larger} daha büyük standartlaştırılmış katsayı alır ve daha ‘önemli’ "
                 "görünür, oysa bir birimlik etkileri aynıdır: standartlaştırılmış katsayı eğimi değişkenin yayılımıyla "
                 "çarpar. ")
        text += ("Bu örneklemde de sıralama aynıdır. " if (s["z2"] > s["z1"]) == (s2 > s1)
                 else "Bu örneklemde tahmin belirsizliği sıralamayı tersine çevirmiştir. ")
        text += "σ₁ ile σ₂'yi yer değiştirin: anakütledeki sıralama da yer değiştirir. "
    return text + ("Standartlaştırılmış katsayılar örneklem standart sapmalarına bağlıdır; başka bir örneklemde sıralama "
                   "değişebilir ve bu sıralama nedensel önem sıralaması değildir (§9.2).")


STANDARDIZED = SimExperiment(
    topic_key=TOPIC,
    number=1,
    title="Standartlaştırılmış katsayı yayılıma bağlıdır",
    question="İki değişkenin Y üzerindeki gerçek etkisi aynıysa, standartlaştırılmış katsayıları da aynı mı olur? "
             "Değişkenlerden biri örneklemde daha çok değişirse ‘en önemli değişken’ hangisi görünür?",
    note=NoteRef("9.2"),
    parameters=(
        SimParameter("s1", "X₁'in standart sapması σ₁", 0.25, 3.0, 1.0, 0.25, "X₁'in örneklemdeki yayılımı.",
                     decimals=2),
        SimParameter("s2", "X₂'nin standart sapması σ₂", 0.25, 3.0, 2.0, 0.25, "X₂'nin örneklemdeki yayılımı.",
                     decimals=2),
        SimParameter("n", "Örneklem büyüklüğü n", 100, 2000, 500, 100, "Gözlem sayısı.", integer=True, decimals=0),
    ),
    dgp=_std_dgp,
    dgp_note=(
        "Notlarda bu deneyin sayısal karşılığı yoktur. Standartlaştırılmış katsayı, Uygulama Adım 2'deki gibi bütün "
        "değişkenler standartlaştırılarak EKK ile ya da aynı sonucu veren β̂ⱼ·s_Xⱼ/s_Y formülüyle hesaplanır. Tek "
        "örneklem, tohum 305; sırasıyla X₁, X₂ ve u çekilir."
    ),
    look_at=(
        "**Tablo** — ham eğim tahminleri birbirine yakınken standartlaştırılmış katsayılar neden farklı?",
        "**σ₁ ve σ₂** — iki değeri eşitleyin, sonra yer değiştirin: ‘en önemli değişken’ değişiyor mu?",
        "**n** — örneklem büyüdükçe örneklem değerleri anakütle değerlerine yaklaşıyor mu?",
    ),
    build=_build_standardized,
    metrics=_std_metrics,
    takeaway=_std_takeaway,
    tables=(("katsayilar", "Ham ve standartlaştırılmış katsayılar"),),
    labels=(("x1", "X₁"), ("x2", "X₂"), ("u", "u"), ("y", "Y")),
)


# --- Deney 2: dönüm noktası ve veri aralığı -----------------------------------------------------------------------

REPS_2 = 1000
CURVATURE_2 = -0.05
INSIDE_2 = 5.0
"""Y = 1 + β₁X + β₂X² + u, X ~ U(0, 10), β₂ = −0,05, β₁ = −2β₂x* = 0,1x*: gerçek dönüm noktası x*. A senaryosunda
x* = 5 (veri aralığının ortası), B senaryosunda x* kaydırıcıyla seçilir. İki senaryo aynı tohumla başlar: X ve u aynı
çekilişlerdir, yalnız x* farklıdır."""
LOW_2, HIGH_2 = -10.0, 50.0
"""Histogramın gösterdiği aralık; dışında kalan tahminlerin payı tabloda ayrıca verilir."""


def _turning_settings(parameters: Parameters) -> tuple[float, int, float]:
    return _rounded(parameters, "xstar", 0), int(parameters["n"]), _rounded(parameters, "sigma", 2)


def _turning_scenario(result: str, xstar: float, n: int, sigma: float, label: str) -> MonteCarlo:
    beta1 = round(-2 * CURVATURE_2 * xstar, 10)
    mean = E.add(E.add(1, E.mul(beta1, E.var("x"))), E.mul(CURVATURE_2, E.var("xsq")))
    body = (
        NewSample("orneklem", n, None),
        Draw("orneklem", "x", "uniform", 0, 10, "X ~ U(0, 10): veri aralığı 0–10"),
        Draw("orneklem", "u", "normal", 0, sigma, f"u ~ N(0, {_short(sigma)}²)"),
        Derive("orneklem", "xsq", E.power(E.var("x"), 2), "X²"),
        Derive("orneklem", "y", E.add(mean, E.var("u")), f"Y = 1 + {_short(beta1, 3)}X − 0,05X² + u"),
        OLS("model", "orneklem", "y", ("x", "xsq"), "EKK: y ~ x + xsq"),
        ModelValue("b1", "model", "coef", "β̂₁", term="x"),
        ModelValue("b2", "model", "coef", "β̂₂", term="xsq"),
        Scalar("donum", E.div(E.neg(E.ref("b1")), E.mul(2, E.ref("b2"))), "Tahmin edilen dönüm noktası −β̂₁/(2β̂₂)"),
    )
    collect = (
        ("donum", E.ref("donum")),
        ("yakin", E.compare("lt", E.absolute(E.sub(E.ref("donum"), xstar)), 1)),
        ("aralikta", E.mul(E.compare("ge", E.ref("donum"), 0), E.compare("le", E.ref("donum"), 10))),
        ("disarida", E.sub(1, E.mul(E.compare("ge", E.ref("donum"), LOW_2), E.compare("le", E.ref("donum"), HIGH_2)))),
    )
    return MonteCarlo(result, REPS_2, SEED, body, collect,
                      f"{label}: {_thousands(REPS_2)} örneklem, her birinde {n} gözlem; tahmin edilen dönüm noktası")


def _build_turning(parameters: Parameters) -> tuple:
    xstar, n, sigma = _turning_settings(parameters)
    return (
        _turning_scenario("icerde", INSIDE_2, n, sigma, "A: x* = 5 (veri aralığının ortası)"),
        _turning_scenario("secilen", xstar, n, sigma, f"B: x* = {_short(xstar)}"),
        SummaryTable((("A: x* = 5", "icerde"), (f"B: x* = {_short(xstar)}", "secilen")),
                     (("medyan", "donum", "median"), ("yakin", "yakin", "mean"), ("aralikta", "aralikta", "mean"),
                      ("disarida", "disarida", "mean")),
                     "donum_ozet", "Tahmin edilen dönüm noktası: medyan ve isabet oranları", decimals=3,
                     heading="Senaryo", column_decimals=(("medyan", 2),)),
        Statistic("icerde", "donum", "median", "med_a", "A: tahminlerin medyanı", decimals=2),
        Statistic("secilen", "donum", "median", "med_b", "B: tahminlerin medyanı", decimals=2),
        Statistic("icerde", "yakin", "mean", "yakin_a", "A: gerçek değere 1 yıldan yakın tahminlerin payı", decimals=3),
        Statistic("secilen", "yakin", "mean", "yakin_b", "B: gerçek değere 1 yıldan yakın tahminlerin payı",
                  decimals=3),
        Statistic("secilen", "aralikta", "mean", "aralik_b", "B: tahmini 0–10 aralığında kalanların payı", decimals=3),
        JoinColumns("donumler", (("icerde", "icerde", "donum"), ("secilen", "secilen", "donum")), decimals=2),
        Histogram("donumler", (("icerde", "A: x* = 5"), ("secilen", f"B: x* = {_short(xstar)}")), 60, LOW_2, HIGH_2,
                  "Tahmin edilen dönüm noktaları (−10 ile 50 arası gösterilir)", "Tahmin edilen dönüm noktası −β̂₁/(2β̂₂)",
                  references=((INSIDE_2, "A: gerçek x* = 5"), (xstar, f"B: gerçek x* = {_short(xstar)}"),
                              (10.0, "Veri aralığının üst sınırı"))),
    )


def _turning_dgp(parameters: Parameters) -> tuple[str, ...]:
    xstar, n, sigma = _turning_settings(parameters)
    return (
        r"Y_i = 1 + \beta_1 X_i + \beta_2 X_i^2 + u_i, \qquad X_i \sim U(0,\ 10), \qquad "
        rf"u_i \sim N(0,\ {_tex(sigma)}^2), \qquad n = {n}",
        r"\beta_2 = -0{,}05, \qquad \beta_1 = -2\beta_2 x^* = 0{,}1\,x^*, \qquad x^* = -\frac{\beta_1}{2\beta_2}",
        rf"\text{{A: }} x^* = 5 \qquad \text{{B: }} x^* = {_tex(xstar, 0)}; \qquad "
        rf"\text{{her senaryoda {_thousands(REPS_2)} örneklem}}",
    )


def _turning_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    s = state.scalars
    return (
        SimMetric("A: medyan tahmin", plain(s["med_a"], 2), "x* = 5 senaryosunda tahmin edilen dönüm noktalarının "
                  "medyanı."),
        SimMetric("B: medyan tahmin", plain(s["med_b"], 2), "Seçilen x* senaryosunda tahminlerin medyanı."),
        SimMetric("B: 1 yıldan yakın", "%" + plain(100 * s["yakin_b"], 1),
                  "Tahminin gerçek x*'a 1 yıldan yakın olduğu örneklemlerin payı (A'da "
                  + "%" + plain(100 * s["yakin_a"], 1) + ")."),
        SimMetric("B: tahmin 0–10'da", "%" + plain(100 * s["aralik_b"], 1),
                  "Tahmin edilen dönüm noktasının veri aralığında kaldığı örneklemlerin payı."),
    )


def _turning_takeaway(state: LabState, parameters: Parameters) -> str:
    xstar, _, _ = _turning_settings(parameters)
    s = state.scalars
    text = (f"x* = 5 iken tahminlerin %{plain(100 * s['yakin_a'], 1)} kadarı gerçek dönüm noktasına 1 yıldan yakındır "
            f"(medyan {plain(s['med_a'], 2)})")
    text += (": tepe verinin ortasında olduğu için veri tepeyi iyi belirler. " if s["yakin_a"] >= 0.8
             else ": tepe verinin içinde olsa da bu gürültü düzeyinde ve bu örneklem büyüklüğünde tahmin belirsizdir. ")
    if xstar == INSIDE_2:
        text += "B senaryosu bu ayarda A ile aynıdır. "
    elif xstar <= 10:
        text += (f"B senaryosunda da gerçek dönüm noktası veri aralığının içindedir (x* = {_short(xstar)}); gerçek değere "
                 f"1 yıldan yakın tahminlerin payı %{plain(100 * s['yakin_b'], 1)}. Tepe kenara yaklaştıkça bir yanında "
                 "daha az gözlem kalır ve belirsizlik büyür. ")
    else:
        text += (f"B senaryosunda gerçek dönüm noktası veri aralığının dışındadır (x* = {_short(xstar)}): tahminlerin "
                 f"yalnız %{plain(100 * s['yakin_b'], 1)} kadarı 1 yıldan yakındır, medyan {plain(s['med_b'], 2)}. ")
    text += ("İki senaryoda X ve u aynı çekilişler olduğu için eğrilik tahmini β̂₂ her tekrarda iki senaryoda aynıdır; "
             "değişen, bu tahmin hatasının dönüm noktasına yansımasıdır. Tahmin −β̂₁/(2β̂₂) = x*·β₂/β̂₂ − "
             "(β̂₁ − β₁)/(2β̂₂) biçiminde yazılır: β̂₂'nin gerçek değerden oransal sapması x* ile çarpılır. β̂₂ sıfıra "
             "yaklaştığında tahmin çok büyür; β̂₂ pozitifse tahmin genellikle negatif çıkar. ")
    return text + ("Dönüm noktası iki tahminin oranıdır; veri aralığının dışındaki ya da az gözleme dayanan bir dönüm "
                   "noktası güçlü biçimde yorumlanmaz (§9.4–9.5).")


TURNING = SimExperiment(
    topic_key=TOPIC,
    number=2,
    title="Dönüm noktası ve veri aralığı",
    question="Karesel modelde dönüm noktası −β̂₁/(2β̂₂) ile hesaplanır. Gerçek dönüm noktası verinin içindeyken ve "
             "dışındayken bu tahmin ne kadar güvenilirdir?",
    note=NoteRef("9.4"),
    parameters=(
        SimParameter("xstar", "B senaryosunda gerçek dönüm noktası x*", 2, 40, 20, 1,
                     "Veri 0–10 aralığındadır; 10'un üzerindeki x* veri aralığının dışındadır.", integer=True,
                     decimals=0),
        SimParameter("n", "Örneklem büyüklüğü n", 50, 500, 100, 50, "Her örneklemdeki gözlem sayısı.", integer=True,
                     decimals=0),
        SimParameter("sigma", "Hata standart sapması σ", 0.5, 3.0, 1.0, 0.25, "Gürültü büyüdükçe β̂₂ daha belirsizdir.",
                     decimals=2),
    ),
    dgp=_turning_dgp,
    dgp_note=(
        "Notlarda bu deneyin sayısal karşılığı yoktur. Eğrilik iki senaryoda aynıdır (β₂ = −0,05); yalnız tepe noktası "
        "kayar. İki senaryo tohum 305 ile başlar; her tekrarda sırasıyla X ve u çekilir, bu yüzden iki senaryonun "
        "verisi yalnız x* bakımından farklıdır. Histogram −10 ile 50 arasındaki tahminleri gösterir; bu aralığın "
        "dışında kalan tahminlerin payı tablodaki ‘Grafik dışında’ sütunundadır. ‘1 yıldan yakın’, tahminin gerçek "
        "x*'a 1 yıldan yakın olduğu örneklemlerin payıdır."
    ),
    look_at=(
        "**Tablo** — A'da tahminlerin çoğu 5'in çevresinde mi? B'de gerçek değere yakın tahminlerin payı kaç?",
        "**Histogram** — x* veri aralığının dışına çıktıkça tahminlerin dağılımı nasıl yayılıyor?",
        "**n ve σ** — örneklem büyüyünce ya da gürültü azalınca B'deki belirsizlik küçülüyor mu?",
    ),
    build=_build_turning,
    metrics=_turning_metrics,
    takeaway=_turning_takeaway,
    tables=(("donum_ozet", "Tahmin edilen dönüm noktası"),),
    labels=(("x", "X"), ("xsq", "X²"), ("u", "u"), ("y", "Y"), ("medyan", "Medyan tahmin"), ("yakin", "1 yıldan yakın (pay)"), ("aralikta", "0–10 aralığında (pay)"),
            ("disarida", "Grafik dışında (pay)"), ("icerde", "A: x* = 5"), ("secilen", "B: seçilen x*")),
)


# --- Deney 3: yanlış fonksiyonel biçim -----------------------------------------------------------------------------

def _form_settings(parameters: Parameters) -> tuple[float, int]:
    return _rounded(parameters, "gamma", 2), int(parameters["n"])


REGIONS_3 = (("dusuk", "Düşük X (0–3,3)", 0), ("orta", "Orta X (3,3–6,7)", 1), ("yuksek", "Yüksek X (6,7–10)", 2))
"""X'in üç eşit bölgesi: (ad, etiket, bölge kodu)."""


def _build_form(parameters: Parameters) -> tuple:
    gamma, n = _form_settings(parameters)
    truth = E.sub(E.add(1, E.mul(2, E.var("x"))), E.mul(gamma, E.power(E.var("x"), 2)))
    line = E.add(E.ref("a0"), E.mul(E.ref("a1"), E.var("x")))
    fitted = E.add(E.add(E.ref("c0"), E.mul(E.ref("c1"), E.var("x"))), E.mul(E.ref("c2"), E.power(E.var("x"), 2)))
    third = round(10 / 3, 10)
    deviations = []
    for column, prefix, model in (("sapma_d", "sap", "Düz çizgi"), ("sapma_k", "sapk", "Karesel model")):
        deviations += [Statistic("veri", column, "mean", f"{prefix}_{name}",
                                 f"{model}: {label} bölgesinde ortalama sapma (tahmin − gerçek)",
                                 where=("bolge", code), decimals=3) for name, label, code in REGIONS_3]
    return (
        NewSample("veri", n, SEED),
        Draw("veri", "x", "uniform", 0, 10, "X ~ U(0, 10)"),
        Draw("veri", "u", "normal", 0, 1, "u ~ N(0, 1)"),
        Derive("veri", "xsq", E.power(E.var("x"), 2), "X²"),
        Derive("veri", "y", E.add(E.sub(E.add(1, E.mul(2, E.var("x"))), E.mul(gamma, E.var("xsq"))), E.var("u")),
               f"Y = 1 + 2X − {_short(gamma)}X² + u"),
        OLS("dogrusal", "veri", "y", ("x",), "Düz çizgi: y ~ x"),
        OLS("karesel", "veri", "y", ("x", "xsq"), "Karesel model: y ~ x + xsq"),
        ModelValue("r2_d", "dogrusal", "r2", "Düz çizgi: R²", decimals=3),
        ModelValue("r2_k", "karesel", "r2", "Karesel model: R²", decimals=3),
        ModelValue("a0", "dogrusal", "coef", "Düz çizgi: sabit", term="Intercept", decimals=4),
        ModelValue("a1", "dogrusal", "coef", "Düz çizgi: X katsayısı", term="x", decimals=4),
        ModelValue("c0", "karesel", "coef", "Karesel model: sabit", term="Intercept", decimals=4),
        ModelValue("c1", "karesel", "coef", "Karesel model: X katsayısı", term="x", decimals=4),
        ModelValue("c2", "karesel", "coef", "Karesel model: X² katsayısı", term="xsq", decimals=4),
        ModelValue("p_kare", "karesel", "p", "X² katsayısının p-değeri (H₀: katsayı = 0)", term="xsq", decimals=4),
        Derive("veri", "gercek", truth, "Gerçek ortalama ilişki E(Y | X)"),
        Derive("veri", "sapma_d", E.sub(line, E.var("gercek")), "Düz çizginin tahmini − gerçek E(Y | X)"),
        Derive("veri", "sapma_k", E.sub(fitted, E.var("gercek")), "Karesel modelin tahmini − gerçek E(Y | X)"),
        Derive("veri", "bolge", E.add(E.compare("gt", E.var("x"), third), E.compare("gt", E.var("x"), 2 * third)),
               "X bölgesi: 0 = düşük (0–3,3), 1 = orta, 2 = yüksek (6,7–10)"),
        *deviations,
        ScalarTable(tuple((label, E.ref(f"sap_{name}")) for name, label, _ in REGIONS_3), "sapma_dogrusal", decimals=3,
                    heading="X bölgesi"),
        ScalarTable(tuple((label, E.ref(f"sapk_{name}")) for name, label, _ in REGIONS_3), "sapma_karesel", decimals=3,
                    heading="X bölgesi"),
        JoinColumns("sapma_ozet", (("Düz çizgi − gerçek", "sapma_dogrusal", "deger"),
                                   ("Karesel model − gerçek", "sapma_karesel", "deger")), decimals=3,
                    heading="X bölgesi"),
        Support("izgara", "k", 0, 100, "Izgara: k = 0, 1, …, 100"),
        Derive("izgara", "x", E.div(E.var("k"), 10), "X = k/10: 0, 0,1, …, 10"),
        Derive("izgara", "gercek", truth, "Gerçek ortalama ilişki E(Y | X)"),
        Derive("izgara", "karesel_tahmin", fitted, "Karesel modelin tahmin edilen eğrisi"),
        ScatterPlot("veri", "x", "y", "X", "Y", f"Gerçek ilişki, düz çizgi ve karesel model (γ = {_short(gamma)}, n = {n})",
                    fit_line=True, size=6, opacity=0.4,
                    curves=(("izgara", "x", "gercek", "Gerçek ortalama ilişki"),
                            ("izgara", "x", "karesel_tahmin", "Karesel model"))),
    )


def _form_dgp(parameters: Parameters) -> tuple[str, ...]:
    gamma, n = _form_settings(parameters)
    return (
        rf"Y_i = 1 + 2X_i - \gamma X_i^2 + u_i, \qquad \gamma = {_tex(gamma)}, \qquad X_i \sim U(0,\ 10), \qquad "
        rf"u_i \sim N(0,\ 1), \qquad n = {n}",
        r"\text{Düz çizgi: } Y_i = \alpha_0 + \alpha_1 X_i + v_i \qquad \text{Karesel model: } "
        r"Y_i = \beta_0 + \beta_1 X_i + \beta_2 X_i^2 + u_i \quad (\beta_2 = -\gamma)",
        r"\text{Sapma: } \hat Y_i - \mathbb{E}(Y \mid X_i), \quad \text{X'in üç eşit bölgesinde ortalanır}",
    )


def _form_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    s = state.scalars
    p = s["p_kare"]
    return (
        SimMetric("R², düz çizgi", plain(s["r2_d"], 3), "Düz çizginin (y ~ x) R²'si."),
        SimMetric("R², karesel", plain(s["r2_k"], 3), "Karesel modelin (y ~ x + x²) R²'si."),
        SimMetric("Orta X'te sapma", plain(s["sap_orta"], 3),
                  "Düz çizginin orta X bölgesinde tahmin − gerçek E(Y | X) ortalaması; gerçek ilişki düz çizgiyse "
                  "sıfıra yakındır."),
        SimMetric("X²'nin p-değeri", "< 0,001" if p < 0.0005 else plain(p, 3),
                  "Karesel modelde H₀: β₂ = 0 testinin p-değeri."),
    )


def _form_takeaway(state: LabState, parameters: Parameters) -> str:
    gamma, _ = _form_settings(parameters)
    s = state.scalars
    p = s["p_kare"]
    p_text = "p < 0,001" if p < 0.0005 else f"p = {plain(p, 3)}"
    line = f"{plain(s['sap_dusuk'], 3)}, {plain(s['sap_orta'], 3)} ve {plain(s['sap_yuksek'], 3)}"
    if gamma == 0:
        text = ("γ = 0 iken gerçek ilişki düz çizgidir ve düz çizgi doğru biçimdir: tahmininin gerçek ortalamadan "
                f"sapması düşük, orta ve yüksek X'te {line}; yalnız örneklem hatasıdır. ")
        if p >= 0.05:
            return text + (f"X² terimi anlamlı değildir ({p_text}); gereksiz terim modele anlamlı katkı sağlamaz "
                           "(§9.9).")
        return text + (f"X² terimi bu örneklemde %5 düzeyinde anlamlı çıkmıştır ({p_text}); gerçek katsayı sıfır "
                       "olduğundan bu bir birinci tür hatadır (§9.9).")
    text = f"Gerçek ilişki eğridir (γ = {_short(gamma)}). "
    if s["sap_orta"] < 0 < min(s["sap_dusuk"], s["sap_yuksek"]):
        text += (f"Düz çizgi uçlarda fazla, ortada az tahmin eder: tahminin gerçek ortalamadan sapması düşük, orta ve "
                 f"yüksek X'te {line}. ")
    else:
        text += (f"Eğrilik bu örneklemde küçük kalır: düz çizginin sapması düşük, orta ve yüksek X'te {line}; örneklem "
                 "hatasından açıkça ayrışmaz. ")
    text += ("Düz çizgi modelinde hata terimi dışarıda bırakılan eğriliği taşır ve X ile sistematik ilişkilidir; sıfır "
             "koşullu ortalama varsayımı bozulur. ")
    line_gap = max(abs(s[f"sap_{name}"]) for name, _, _ in REGIONS_3)
    quad_gap = max(abs(s[f"sapk_{name}"]) for name, _, _ in REGIONS_3)
    if quad_gap < line_gap:
        text += (f"Karesel modelin sapması daha küçüktür: en büyük mutlak sapma {plain(quad_gap, 3)}, düz çizgide "
                 f"{plain(line_gap, 3)}; R² {plain(s['r2_d'], 3)} → {plain(s['r2_k'], 3)}. ")
    else:
        text += (f"Bu örneklemde karesel modelin tahmin hatası da düz çizginin sapması kadar büyüktür (en büyük mutlak "
                 f"sapma {plain(quad_gap, 3)} ve {plain(line_gap, 3)}): eğrilik küçükken ek terim az bilgi ekler, "
                 "tahmin belirsizliğini ise artırır. ")
    if p < 0.05:
        return text + (f"X² terimi anlamlıdır ({p_text}): karesel biçimi destekler; ancak bütün olası biçimleri dışlamaz "
                       "(§9.9).")
    return text + (f"X² terimi bu örneklemde %5 düzeyinde anlamlı değildir ({p_text}): eğrilik küçükken ya da örneklem "
                   "küçükken test eğriliği ayırt edemeyebilir; anlamsızlık doğrusal biçimi kanıtlamaz (§9.9).")


FORM = SimExperiment(
    topic_key=TOPIC,
    number=3,
    title="Yanlış fonksiyonel biçim",
    question="Gerçek ortalama ilişki eğriyken düz çizgi tahmin edilirse ne olur? Düz çizgi gerçek ortalamadan nerede ve "
             "hangi yönde sapar?",
    note=NoteRef("9.9"),
    parameters=(
        SimParameter("gamma", "Eğrilik γ", 0.0, 0.3, 0.15, 0.01, "γ = 0: gerçek ilişki düz çizgi.", decimals=2),
        SimParameter("n", "Örneklem büyüklüğü n", 50, 1000, 200, 50, "Gözlem sayısı.", integer=True, decimals=0),
    ),
    dgp=_form_dgp,
    dgp_note=(
        "Notlarda bu deneyin sayısal karşılığı yoktur. Tek örneklem, tohum 305; sırasıyla X ve u çekilir. Gerçek "
        "ortalama ilişki bilindiği için iki modelin tahmin edilen değerleri gerçek E(Y | X) ile karşılaştırılır; "
        "farklar X'in üç eşit bölgesinde ortalanır. Eğriler 0–10 aralığındaki bir ızgarada çizilir."
    ),
    look_at=(
        "**Saçılım** — düz çizgi gerçek eğriden nerede ayrılıyor? Karesel model eğriyi yakalıyor mu?",
        "**Tablo** — düz çizginin sapması X bölgelerine göre sistematik mi? Karesel modelin sapması nasıl?",
        "**γ** — eğriliği sıfıra indirin: sapmalar ve X² teriminin p-değeri nasıl değişiyor?",
    ),
    build=_build_form,
    metrics=_form_metrics,
    takeaway=_form_takeaway,
    tables=(("sapma_ozet", "Tahminin gerçek ortalamadan sapması: X bölgelerine göre ortalama (tahmin − gerçek)"),),
    labels=(("x", "X"), ("xsq", "X²"), ("u", "u"), ("y", "Y"), ("gercek", "Gerçek ortalama ilişki"),
            ("sapma_d", "Düz çizgi − gerçek E(Y | X)"), ("sapma_k", "Karesel model − gerçek E(Y | X)"),
            ("bolge", "X bölgesi"), ("karesel_tahmin", "Karesel model"), ("k", "k")),
)


KONU09_EXPERIMENTS = (STANDARDIZED, TURNING, FORM)
