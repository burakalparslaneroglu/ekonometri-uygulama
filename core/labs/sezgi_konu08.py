"""Konu 8 Sezgi deneyleri: veri üretim süreci bilinen kontrollü simülasyonlar.

Deney 1  Normal olmayan hatalarla F testi: büyük örneklem mantığı      (Notlar §8.10, Tablo 8.3, Şekil 8.2–8.3)
Deney 2  Ayrı t testleri ve ortak F testi                              (Notlar §8.1)
Deney 3  Büyük örneklem neyi çözmez?                                   (Notlar §8.11)

Bölüm 8'in benzetimi Deney 1'in varsayılan ayarlarıdır (tohum 305); notlardaki tablo ve şekiller
(``scripts/bolum8_uygulama.py``) aynı çekiliş sırasıyla üretilir ve testle karşılaştırılır. Üç örneklem büyüklüğünün
her biri tohum 305 ile başlar. Deney 2 ve 3'ün notlarda sayısal karşılığı yoktur; varsayılan ayarları bölümün
kavramlarını göstermek için seçilmiştir. Her senaryo ve örneklem büyüklüğü tohum 305 ile başlar.
"""

from __future__ import annotations

import numpy as np

from core.labs import expr as E
from core.labs.runner import LabState
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, plain
from core.labs.spec import (
    OLS,
    Derive,
    Draw,
    Histogram,
    JoinColumns,
    LineChart,
    ModelValue,
    MonteCarlo,
    NewSample,
    NoteRef,
    Scalar,
    ScatterPlot,
    Statistic,
    SummaryTable,
)

SEED = 305
TOPIC = "konu08"


def _rounded(parameters: Parameters, key: str, decimals: int) -> float:
    """Kaydırıcı değeri, kodda 0,30000000000000004 gibi yazımlar olmasın diye yuvarlanır."""

    return round(float(parameters[key]), decimals)


def _tex(value: float, decimals: int = 2) -> str:
    """LaTeX içinde sondaki sıfırları atılmış sayı (0{,}5; 3)."""

    text = f"{value:.{decimals}f}"
    text = text.rstrip("0").rstrip(".") if "." in text else text  # 20 → "2" olmasın
    return "0" if text in ("-0", "") else text.replace(".", "{,}")


def _thousands(value: int) -> str:
    """Binlik ayırıcılı tam sayı (4.000)."""

    return f"{value:,}".replace(",", ".")


def _short(value: float, decimals: int = 2) -> str:
    """Sondaki sıfırları atılmış düz metin sayı (0,5; 3); yuvarlanınca sıfır olan değer "0" yazılır."""

    text = f"{value:.{decimals}f}"
    text = text.rstrip("0").rstrip(".") if "." in text else text  # 20 → "2" olmasın
    return "0" if text in ("-0", "0", "") else text.replace(".", ",").replace("-", "−")


# --- Deney 1: normal olmayan hatalarla F testi ---------------------------------------------------------------

REPS_1 = 4000
SIZES_1 = (25, 100, 500)
RHO_1 = 0.45
ALPHA_1 = 0.05
SHAPE_NOTES_1 = 1.0
"""Tablo 8.3: gerçek eğimler sıfır; x₂ = 0,45x₁ + √(1 − 0,45²)e; hata u = (g − κ)/√κ, g ~ Gamma(κ, 1) (notlarda κ = 1:
üstel dağılım, çarpıklık 2); her n için 4.000 veri seti, yüzde 5 ortak F testi. Biçim parametresi κ'dır; k Bölüm 8'de
eğim sayısıdır."""


def skewness(shape: float) -> float:
    """Gamma(κ, 1) dağılımının (ve u = (g − κ)/√κ hatasının) çarpıklığı 2/√κ."""

    return 2 / shape ** 0.5


def _large_sample_settings(parameters: Parameters) -> float:
    return _rounded(parameters, "k", 2)


def _sample_size(n: int, shape: float) -> MonteCarlo:
    error = E.div(E.sub(E.var("g"), shape), E.sqrt(shape))
    body = (
        NewSample("orneklem", n, None),
        Draw("orneklem", "x1", "normal", 0, 1, "x₁ ~ N(0, 1)"),
        Draw("orneklem", "e", "normal", 0, 1, "e ~ N(0, 1)"),
        Derive("orneklem", "x2", E.add(E.mul(RHO_1, E.var("x1")), E.mul(E.sqrt(1 - RHO_1 ** 2), E.var("e"))),
               "x₂ = 0,45x₁ + √(1 − 0,45²)e: x₁ ile korelasyonu 0,45"),
        Draw("orneklem", "g", "gamma", shape, 1, f"g ~ Gamma({_short(shape)}, 1)"),
        Derive("orneklem", "u", error, "u = (g − κ)/√κ: ortalaması 0, varyansı 1, sağa çarpık"),
        Derive("orneklem", "y", E.add(2, E.var("u")), "y = 2 + u: gerçek eğimler sıfır (H₀ doğru)"),
        OLS("model", "orneklem", "y", ("x1", "x2"), "EKK: y ~ x1 + x2"),
        ModelValue("p_F", "model", "f_p", "Ortak F testinin p-değeri (H₀: iki eğim de sıfır)"),
        ModelValue("t_x1", "model", "t", "x₁ eğiminin t istatistiği", term="x1"),
    )
    collect = (("reddet", E.compare("lt", E.ref("p_F"), ALPHA_1)), ("t_x1", E.ref("t_x1")))
    return MonteCarlo(f"orneklem_{n}", REPS_1, SEED, body, collect,
                      f"n = {n}: {_thousands(REPS_1)} veri seti, her birinde yüzde 5 ortak F testi")


def _build_large_sample(parameters: Parameters) -> tuple:
    shape = _large_sample_settings(parameters)
    changed = "" if shape == SHAPE_NOTES_1 else f" (κ = {_short(shape)})"
    rows = tuple((f"n = {n}", f"orneklem_{n}") for n in SIZES_1)
    return (
        *(_sample_size(n, shape) for n in SIZES_1),
        SummaryTable(rows, (("tekrar", "reddet", "count"), ("ret_orani", "reddet", "mean"),
                            ("t_sd", "t_x1", "std"), ("carpiklik", "t_x1", "skew")),
                     "tablo83", "Tablo 8.3: normal olmayan hatalar altında ortak F testi", decimals=3,
                     heading="Örneklem büyüklüğü"),
        *(Statistic(f"orneklem_{n}", "reddet", "mean", f"ret_{n}", f"n = {n}: ret oranı", decimals=4)
          for n in SIZES_1),
        JoinColumns("t_istatistikleri", tuple((f"n{n}", f"orneklem_{n}", "t_x1") for n in SIZES_1), decimals=3),
        Histogram("t_istatistikleri", tuple((f"n{n}", f"n = {n}") for n in SIZES_1), 40, -4, 4,
                  f"Şekil 8.2: x₁ eğiminin t istatistiği ve standart normal{changed}", "x₁ eğiminin t istatistiği",
                  curves=(("normal", 0, 1, "Standart normal (beklenen sayı)"),)),
    )


def _large_sample_dgp(parameters: Parameters) -> tuple[str, ...]:
    shape = _large_sample_settings(parameters)
    return (
        r"y_i = 2 + 0\cdot x_{1i} + 0\cdot x_{2i} + u_i, \quad x_{1i},\ e_i \sim N(0,\ 1), \quad "
        r"x_{2i} = 0{,}45\,x_{1i} + \sqrt{1 - 0{,}45^2}\;e_i",
        rf"u_i = \frac{{g_i - \kappa}}{{\sqrt{{\kappa}}}}, \quad g_i \sim \text{{Gamma}}(\kappa,\ 1), \quad "
        rf"\kappa = {_tex(shape)} \qquad \text{{çarpıklık}} = 2/\sqrt{{\kappa}} = {_tex(skewness(shape), 2)}",
        rf"n \in \{{25,\ 100,\ 500\}}, \quad \text{{her }} n \text{{ için {_thousands(REPS_1)} veri seti}}; \quad "
        r"H_0: \beta_1 = \beta_2 = 0 \text{ yüzde 5 düzeyinde}",
    )


def _large_sample_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    shape = _large_sample_settings(parameters)
    s = state.scalars
    metrics = tuple(
        SimMetric(f"Ret oranı, n = {n}", "%" + plain(100 * s[f"ret_{n}"], 1),
                  "H₀ doğruyken yüzde 5 ortak F testinin reddetme oranı; hedef %5.")
        for n in SIZES_1
    )
    return metrics + (SimMetric("Hatanın çarpıklığı", plain(skewness(shape), 2),
                                "2/√κ; normal dağılımda 0. Notlarda κ = 1 (üstel dağılım): 2."),)


def _large_sample_takeaway(state: LabState, parameters: Parameters) -> str:
    shape = _large_sample_settings(parameters)
    s = state.scalars
    table = state.tables["tablo83"]
    rates = ", ".join(f"%{plain(100 * s[f'ret_{n}'], 1)}" for n in SIZES_1)
    spreads = ", ".join(plain(value, 3) for value in table["t_sd"])
    text = (f"Hatalar normal değil (çarpıklık {plain(skewness(shape), 2)}); yine de sıfır hipotezi doğruyken yüzde 5 "
            f"ortak F testinin ret oranları n = 25, 100 ve 500 için sırasıyla {rates}: hedef %5'in çevresinde. "
            f"x₁ eğiminin t istatistiğinin standart sapması {spreads}; standart normalde 1'dir (n = 25'te hatalar "
            "normal olsaydı da t(22) dağılımının standart sapması √(22/20) ≈ 1,05 olurdu). ")
    text += ("Büyük örneklem teorisi, belirli koşullar altında standartlaştırılmış istatistiğin dağılımının standart "
             "normale yaklaştığını söyler; bu yüzden t ve F testleri yaklaşık geçerlidir (§8.10). Bu tasarımda ret "
             "oranı küçük n'de bile %5'e yakın kalır; bu sonuç her dağılım ve tasarım için garanti değildir ve "
             "evrensel bir “büyük örneklem” eşiği yoktur.")
    return text


LARGE_SAMPLE = SimExperiment(
    topic_key=TOPIC,
    number=1,
    title="Normal olmayan hatalarla F testi",
    question="Hata terimi normal değil, belirgin biçimde sağa çarpıksa geleneksel F testi hâlâ güvenilir mi? Sıfır "
             "hipotezi doğruyken testin reddetme oranı ve t istatistiğinin dağılımı örneklem büyüdükçe nasıl değişir?",
    note=NoteRef("8.10", objects=("Tablo 8.3", "Şekil 8.2")),
    parameters=(
        SimParameter("k", "Hata dağılımının biçimi κ", 0.25, 5.0, SHAPE_NOTES_1, 0.25,
                     "Gamma(κ, 1): κ küçüldükçe hata daha çarpık (çarpıklık 2/√κ). Notlarda κ = 1 (üstel dağılım).",
                     decimals=2),
    ),
    dgp=_large_sample_dgp,
    dgp_note=(
        "Varsayılan ayarlar notlardaki benzetimdir (Tablo 8.3, Şekil 8.2–8.3; tohum 305). κ = 1'de g üstel dağılımlıdır "
        "ve u = g − 1'dir. Her n için üreteç tohum 305 ile başlar; her tekrarda sırasıyla x₁, e ve g çekilir. Şekil "
        "8.3'teki ret oranları tablonun ret oranı sütunudur. Uygulama tekrarları vektörel hesaplar; üretilen kod "
        "notlardaki gibi döngüyle aynı sayıları verir."
    ),
    look_at=(
        "**Tablo** — H₀ doğruyken ret oranı %5'e ne kadar yakın? t istatistiğinin standart sapması 1'e yaklaşıyor mu?",
        "**Histogram** — n büyüdükçe t istatistiğinin dağılımı standart normal eğriye yaklaşıyor mu?",
        "**κ** — hatayı daha çarpık yapın: küçük örneklemde ret oranı ve t'nin dağılımı ne kadar değişiyor?",
    ),
    build=_build_large_sample,
    metrics=_large_sample_metrics,
    takeaway=_large_sample_takeaway,
    tables=(("tablo83", "Tablo 8.3: normal olmayan hatalar altında büyük örneklem benzetimi", ("k",)),),
    labels=(("x1", "x₁"), ("x2", "x₂"), ("e", "e"), ("g", "g"), ("u", "u"), ("y", "y"),
            ("reddet", "H₀ reddedildi (p < 0,05)"), ("t_x1", "x₁ eğiminin t istatistiği"),
            ("tekrar", "Tekrar"), ("ret_orani", "H₀ ret oranı"), ("t_sd", "t'nin std. sapması"),
            ("carpiklik", "t'nin çarpıklığı")),
)



# --- Deney 2: ayrı t testleri ve ortak F testi -----------------------------------------------------------

REPS_2 = 2000
ALPHA_2 = 0.05
"""Y = 1 + bX₁ + bX₂ + u; X₂ = ρX₁ + √(1 − ρ²)e; iki senaryo: H₀ doğru (b = 0) ve H₀ yanlış (b > 0); her senaryoda
2.000 örneklem (aynı tohumla başlar: iki senaryoda X₁, e ve u aynıdır, yalnız b farklıdır)."""


def _joint_settings(parameters: Parameters) -> tuple[float, float, int]:
    return _rounded(parameters, "rho", 2), _rounded(parameters, "b", 2), int(parameters["n"])


def _rejects(name: str) -> E.Expr:
    return E.compare("lt", E.ref(name), ALPHA_2)


def _joint_scenario(result: str, rho: float, b: float, n: int, label: str) -> MonteCarlo:
    mean = E.add(E.add(1, E.mul(b, E.var("x1"))), E.mul(b, E.var("x2")))
    body = (
        NewSample("orneklem", n, None),
        Draw("orneklem", "x1", "normal", 0, 1, "X₁ ~ N(0, 1)"),
        Draw("orneklem", "e", "normal", 0, 1, "e ~ N(0, 1)"),
        Derive("orneklem", "x2", E.add(E.mul(rho, E.var("x1")), E.mul(E.sqrt(1 - rho ** 2), E.var("e"))),
               f"X₂ = {_short(rho)}X₁ + √(1 − {_short(rho)}²)e: X₁ ile korelasyonu {_short(rho)}"),
        Draw("orneklem", "u", "normal", 0, 1, "u ~ N(0, 1)"),
        Derive("orneklem", "y", E.add(mean, E.var("u")), f"Y = 1 + {_short(b)}X₁ + {_short(b)}X₂ + u"),
        OLS("model", "orneklem", "y", ("x1", "x2"), "EKK: y ~ x1 + x2"),
        ModelValue("p_F", "model", "f_p", "Ortak F testinin p-değeri (H₀: β₁ = β₂ = 0)"),
        ModelValue("p1", "model", "p", "X₁ katsayısının ayrı t testi p-değeri", term="x1"),
        ModelValue("p2", "model", "p", "X₂ katsayısının ayrı t testi p-değeri", term="x2"),
        ModelValue("egim1", "model", "coef", "β̂₁", term="x1"),
        ModelValue("egim2", "model", "coef", "β̂₂", term="x2"),
    )
    any_t = E.maximum(_rejects("p1"), _rejects("p2"))
    collect = (
        ("F_ret", _rejects("p_F")),
        ("t_herhangi", any_t),
        ("yalniz_F", E.mul(_rejects("p_F"), E.sub(1, any_t))),
        ("b1", E.ref("egim1")),
        ("b2", E.ref("egim2")),
    )
    return MonteCarlo(result, REPS_2, SEED, body, collect,
                      f"{label}: {_thousands(REPS_2)} örneklem, her birinde {n} gözlem; ayrı t testleri ve ortak F testi")


def _build_joint(parameters: Parameters) -> tuple:
    rho, b, n = _joint_settings(parameters)
    return (
        _joint_scenario("h0_dogru", rho, 0.0, n, "H₀ doğru (β₁ = β₂ = 0)"),
        _joint_scenario("h0_yanlis", rho, b, n, f"H₀ yanlış (β₁ = β₂ = {_short(b)})"),
        SummaryTable((("H₀ doğru: β₁ = β₂ = 0", "h0_dogru"), (f"H₀ yanlış: β₁ = β₂ = {_short(b)}", "h0_yanlis")),
                     (("tekrar", "F_ret", "count"), ("F_ret", "F_ret", "mean"), ("t_herhangi", "t_herhangi", "mean"),
                      ("yalniz_F", "yalniz_F", "mean")),
                     "ortak", "Ayrı t testleri ve ortak F testi: yüzde 5 düzeyinde ret oranları", decimals=3,
                     heading="Senaryo"),
        Statistic("h0_dogru", "F_ret", "mean", "F_dogru", "H₀ doğru: F ret oranı", decimals=3),
        Statistic("h0_dogru", "t_herhangi", "mean", "t_dogru", "H₀ doğru: en az bir t testi reddeder", decimals=3),
        Statistic("h0_yanlis", "F_ret", "mean", "F_yanlis", "H₀ yanlış: F ret oranı", decimals=3),
        Statistic("h0_yanlis", "yalniz_F", "mean", "yalniz_yanlis",
                  "H₀ yanlış: F reddeder, iki t testi de reddetmez", decimals=3),
        ScatterPlot("h0_yanlis", "b1", "b2", "β̂₁ (X₁ katsayısı tahmini)", "β̂₂ (X₂ katsayısı tahmini)",
                    f"H₀ yanlış senaryosunda iki tahmin birlikte (ρ = {_short(rho)}, n = {n})", size=5, opacity=0.35,
                    lines=((2 * b, -1, f"β̂₁ + β̂₂ = {_short(2 * b)} (gerçek toplam)"),)),
    )


def _joint_dgp(parameters: Parameters) -> tuple[str, ...]:
    rho, b, n = _joint_settings(parameters)
    return (
        r"Y_i = 1 + \beta_1 X_{1i} + \beta_2 X_{2i} + u_i, \qquad X_{2i} = \rho X_{1i} + \sqrt{1 - \rho^2}\,e_i, "
        rf"\qquad \rho = {_tex(rho)}, \quad n = {n}",
        rf"X_{{1i}},\ e_i,\ u_i \sim N(0,\ 1); \qquad \text{{H}}_0\text{{ doğru: }} \beta_1 = \beta_2 = 0; \qquad "
        rf"\text{{H}}_0\text{{ yanlış: }} \beta_1 = \beta_2 = {_tex(b)}",
        rf"\text{{her senaryoda {_thousands(REPS_2)} örneklem, yüzde 5 düzeyi}}",
    )


def _joint_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    s = state.scalars
    return (
        SimMetric("H₀ doğru: F ret oranı", "%" + plain(100 * s["F_dogru"], 1), "Ortak testin I. tür hatası; hedef %5."),
        SimMetric("H₀ doğru: en az bir t reddeder", "%" + plain(100 * s["t_dogru"], 1),
                  "İki ayrı yüzde 5 testinden en az birinin yanlışlıkla reddetme oranı."),
        SimMetric("H₀ yanlış: F ret oranı", "%" + plain(100 * s["F_yanlis"], 1), "Ortak testin gücü."),
        SimMetric("H₀ yanlış: yalnız F reddeder", "%" + plain(100 * s["yalniz_yanlis"], 1),
                  "Ortak test reddederken iki ayrı t testinin de reddetmediği örneklemler."),
    )


def estimate_correlation(state: LabState) -> float:
    """H₀ yanlış senaryosunda iki katsayı tahmininin tekrarlar arasındaki korelasyonu (saçılım grafiğinin özeti)."""

    table = state.tables["h0_yanlis"]
    return float(np.corrcoef(table["b1"], table["b2"])[0, 1])


def _joint_takeaway(state: LabState, parameters: Parameters) -> str:
    rho, b, _ = _joint_settings(parameters)
    s = state.scalars
    text = (f"H₀ doğruyken ortak F testi örneklemlerin %{plain(100 * s['F_dogru'], 1)} kadarında reddeder: hedef %5. "
            f"İki ayrı yüzde 5 t testinden en az biri %{plain(100 * s['t_dogru'], 1)} kadarında reddeder. ")
    if s["t_dogru"] > max(s["F_dogru"], ALPHA_2) + 0.01:
        text += "Ayrı testler ortak yanlış karar olasılığını %5'te tutmaz (§8.1). "
    else:
        text += ("Bu ayarda iki t istatistiği güçlü biçimde birlikte hareket ettiği için bu oran %5'e yakındır; ρ = 0 "
                 "iken yaklaşık 1 − 0,95² = %9,75 olur. Ayrı testlerin birleşik yanlış ret oranı tasarıma bağlıdır; "
                 "ortak F testi her ρ'da %5 hedefindedir (§8.1). ")
    text += (f"β₁ = β₂ = {_short(b)} iken F testi örneklemlerin %{plain(100 * s['F_yanlis'], 1)} kadarında reddeder; "
             f"%{plain(100 * s['yalniz_yanlis'], 1)} kadarında iki ayrı t testi de reddetmediği hâlde ortak test "
             "reddeder. ")
    correlation = estimate_correlation(state)
    if rho >= 0.7:
        text += (f"X₁ ile X₂ yüksek korelasyonlu: iki tahminin tekrarlar arasındaki korelasyonu "
                 f"{plain(correlation, 2)}, yani ters yönde birlikte hareket ederler (saçılım). Veri katsayıların "
                 "toplamını iyi, ayrı katkılarını zayıf belirler; tek tek anlamsız görünen katsayılar birlikte güçlü "
                 "açıklayıcılık taşıyabilir.")
    else:
        text += (f"İki tahminin tekrarlar arasındaki korelasyonu {plain(correlation, 2)}. ρ'yu yükseltin: tahminler "
                 "ters yönde birlikte hareket etmeye başlar; etki küçükken ayrı t testleri zayıflar, ortak test gücünü "
                 "korur.")
    return text


JOINT = SimExperiment(
    topic_key=TOPIC,
    number=2,
    title="Ayrı t testleri ve ortak F testi",
    question="İki katsayının birlikte sıfır olduğu hipotezini iki ayrı t testiyle sınamak ortak F testiyle aynı şey "
             "midir? Açıklayıcı değişkenler yüksek korelasyonluysa ayrı testler ne gösterir, ortak test ne gösterir?",
    note=NoteRef("8.1"),
    parameters=(
        SimParameter("rho", "X₁ ile X₂ arasındaki korelasyon ρ", 0.0, 0.95, 0.9, 0.05,
                     "Yüksek ρ: katsayı tahminleri ters yönde birlikte hareket eder.", decimals=2),
        SimParameter("b", "H₀ yanlış senaryosunda ortak eğim b", 0.05, 0.5, 0.15, 0.05,
                     "β₁ = β₂ = b; H₀ doğru senaryosunda iki eğim de sıfırdır.", decimals=2),
        SimParameter("n", "Örneklem büyüklüğü n", 30, 300, 100, 10, "Her örneklemdeki gözlem sayısı.", integer=True,
                     decimals=0),
    ),
    dgp=_joint_dgp,
    dgp_note=(
        "Notlarda bu deneyin sayısal karşılığı yoktur; §8.1'deki iki gerekçeyi (ortak yanlış karar olasılığı ve birlikte "
        "hareket eden tahminler) tekrarlı örneklemede gösterir. İki senaryo tohum 305 ile başlar; her tekrarda "
        "sırasıyla X₁, e ve u çekilir. Ortak F testi modeldeki iki eğimin birlikte sıfır olduğu hipotezini sınar "
        "(genel anlamlılık testi)."
    ),
    look_at=(
        "**Tablo** — H₀ doğruyken F testinin ret oranı %5'e yakın mı? En az bir t testinin reddetme oranı kaç?",
        "**Saçılım** — ρ yüksekken β̂₁ ve β̂₂ nasıl birlikte hareket ediyor? Toplamları gerçek değerin çevresinde mi?",
        "**ρ** — korelasyonu sıfıra indirin: yalnız F'nin reddettiği örneklemler azalıyor mu?",
    ),
    build=_build_joint,
    metrics=_joint_metrics,
    takeaway=_joint_takeaway,
    tables=(("ortak", "Ayrı t testleri ve ortak F testi: ret oranları"),),
    labels=(("x1", "X₁"), ("x2", "X₂"), ("e", "e"), ("u", "u"), ("y", "Y"), ("b1", "β̂₁"), ("b2", "β̂₂"),
            ("tekrar", "Tekrar sayısı"), ("F_ret", "F testi reddeder"), ("t_herhangi", "En az bir t testi reddeder"),
            ("yalniz_F", "Yalnız F testi reddeder")),
)


# --- Deney 3: büyük örneklem neyi çözmez? -----------------------------------------------------------------

REPS_3 = 500
SIZES_3 = (50, 200, 800, 2000)
BETA_3 = 0.5
"""Y = 1 + 0,5X + γZ + u, Z = ρX + √(1 − ρ²)e; kısa model Z'yi dışarıda bırakır. Her n için 500 örneklem (tohum 305);
her örneklemde kısa ve uzun model ile X katsayısının yüzde 95 güven aralığı."""


def _omitted_settings(parameters: Parameters) -> tuple[float, float]:
    return _rounded(parameters, "gamma", 2), _rounded(parameters, "rho", 2)


def _covers(low: str, high: str) -> E.Expr:
    return E.mul(E.compare("le", E.ref(low), BETA_3), E.compare("le", BETA_3, E.ref(high)))


def _omitted_size(n: int, gamma: float, rho: float) -> MonteCarlo:
    mean = E.add(E.add(1, E.mul(BETA_3, E.var("x"))), E.mul(gamma, E.var("z")))
    body = (
        NewSample("orneklem", n, None),
        Draw("orneklem", "x", "normal", 0, 1, "X ~ N(0, 1)"),
        Draw("orneklem", "e", "normal", 0, 1, "e ~ N(0, 1)"),
        Derive("orneklem", "z", E.add(E.mul(rho, E.var("x")), E.mul(E.sqrt(1 - rho ** 2), E.var("e"))),
               f"Z = {_short(rho)}X + √(1 − {_short(rho)}²)e"),
        Draw("orneklem", "u", "normal", 0, 1, "u ~ N(0, 1)"),
        Derive("orneklem", "y", E.add(mean, E.var("u")), f"Y = 1 + 0,5X + {_short(gamma)}Z + u"),
        OLS("kisa", "orneklem", "y", ("x",), "Kısa model (Z dışarıda): y ~ x"),
        OLS("uzun", "orneklem", "y", ("x", "z"), "Uzun model: y ~ x + z"),
        ModelValue("b_kisa", "kisa", "coef", "Kısa model: X katsayısı", term="x"),
        ModelValue("alt_kisa", "kisa", "ci_low", "Kısa model: yüzde 95 GA alt sınırı", term="x"),
        ModelValue("ust_kisa", "kisa", "ci_high", "Kısa model: yüzde 95 GA üst sınırı", term="x"),
        ModelValue("b_uzun", "uzun", "coef", "Uzun model: X katsayısı", term="x"),
        ModelValue("alt_uzun", "uzun", "ci_low", "Uzun model: yüzde 95 GA alt sınırı", term="x"),
        ModelValue("ust_uzun", "uzun", "ci_high", "Uzun model: yüzde 95 GA üst sınırı", term="x"),
    )
    collect = (
        ("n", E.const(n)),
        ("kisa", E.ref("b_kisa")),
        ("kisa_kapsar", _covers("alt_kisa", "ust_kisa")),
        ("uzun", E.ref("b_uzun")),
        ("uzun_kapsar", _covers("alt_uzun", "ust_uzun")),
    )
    return MonteCarlo(f"orneklem_{n}", REPS_3, SEED, body, collect,
                      f"n = {_thousands(n)}: {REPS_3} örneklem; kısa ve uzun modelde X katsayısı ve yüzde 95 güven "
                      "aralığı")


def _build_omitted(parameters: Parameters) -> tuple:
    gamma, rho = _omitted_settings(parameters)
    return (
        *(_omitted_size(n, gamma, rho) for n in SIZES_3),
        SummaryTable(tuple((f"n = {_thousands(n)}", f"orneklem_{n}") for n in SIZES_3),
                     (("n", "n", "mean"), ("kisa_ort", "kisa", "mean"), ("kisa_sd", "kisa", "std"),
                      ("kisa_kapsama", "kisa_kapsar", "mean"), ("uzun_ort", "uzun", "mean"),
                      ("uzun_kapsama", "uzun_kapsar", "mean")),
                     "buyuk_n", "Örneklem büyüdükçe kısa ve uzun modelde X katsayısı", decimals=3,
                     heading="Örneklem", column_decimals=(("n", 0),)),
        Scalar("hedef", E.const(0.95), "Güven düzeyi (hedef kapsama oranı)", decimals=2),
        LineChart("buyuk_n", "n", "kisa_kapsama", "Örneklem büyüklüğü n",
                  "Gerçek katsayıyı (0,5) kapsayan aralıkların payı",
                  f"Kapsama oranı ve örneklem büyüklüğü (γ = {_short(gamma)}, ρ = {_short(rho)})",
                  references=(("hedef", "Güven düzeyi %95"),), legend="Kısa model (Z dışarıda)",
                  series=(("uzun_kapsama", "Uzun model (Z modelde)"),)),
    )


def _omitted_dgp(parameters: Parameters) -> tuple[str, ...]:
    gamma, rho = _omitted_settings(parameters)
    sizes = r",\ ".join(_thousands(n) for n in SIZES_3)
    return (
        rf"Y_i = 1 + 0{{,}}5\,X_i + \gamma Z_i + u_i, \qquad Z_i = \rho X_i + \sqrt{{1 - \rho^2}}\;e_i, \qquad "
        rf"\gamma = {_tex(gamma)},\ \rho = {_tex(rho)}",
        rf"X_i,\ e_i,\ u_i \sim N(0,\ 1); \qquad n \in \{{{sizes}\}}, \quad \text{{her }} n \text{{ için {REPS_3} "
        r"örneklem}",
        r"\text{kısa model: } Y \text{ üzerine yalnız } X; \qquad \text{uzun model: } Y \text{ üzerine } X \text{ ve } Z",
    )


def _omitted_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    gamma, rho = _omitted_settings(parameters)
    table = state.tables["buyuk_n"]
    first, last = table.iloc[0], table.iloc[-1]
    largest = SIZES_3[-1]
    return (
        SimMetric("Kısa modelin yanlılığı γρ", plain(gamma * rho, 3),
                  f"Kısa modelin tahminleri 0,5 yerine 0,5 + γρ = {plain(BETA_3 + gamma * rho, 3)} çevresinde toplanır."),
        SimMetric(f"Kısa model kapsama, n = {SIZES_3[0]}", "%" + plain(100 * first["kisa_kapsama"], 1),
                  "Gerçek katsayıyı kapsayan yüzde 95 aralıkların payı."),
        SimMetric(f"Kısa model kapsama, n = {_thousands(largest)}", "%" + plain(100 * last["kisa_kapsama"], 1),
                  "Yanlı tahminin çevresinde daralan aralıklar gerçek değeri kaçırır." if gamma * rho != 0
                  else "γρ = 0: kısa model yanlı değildir; kapsama hedef düzeyin çevresindedir."),
        SimMetric(f"Uzun model kapsama, n = {_thousands(largest)}", "%" + plain(100 * last["uzun_kapsama"], 1),
                  "Z modeldeyken aralıklar hedef düzeyin çevresinde kalır."),
    )


def _omitted_takeaway(state: LabState, parameters: Parameters) -> str:
    gamma, rho = _omitted_settings(parameters)
    table = state.tables["buyuk_n"]
    first, last = table.iloc[0], table.iloc[-1]
    if gamma * rho == 0:
        return ("γρ = 0: dışarıda kalan Z ya Y'yi etkilemiyor ya da X ile ilişkisiz; kısa model yanlı değildir ve iki "
                "modelin kapsama oranı da %95 çevresindedir. γ ve ρ'yu sıfırdan farklı yapın.")
    text = (f"Kısa modelin tahminleri 0,5 yerine yaklaşık {plain(BETA_3 + gamma * rho, 3)} çevresinde toplanır "
            f"(en büyük örneklemde, n = {_thousands(SIZES_3[-1])}, ortalama {plain(last['kisa_ort'], 3)}). Örneklem "
            f"büyüdükçe tahminlerin standart sapması {plain(first['kisa_sd'], 3)} değerinden "
            f"{plain(last['kisa_sd'], 3)} değerine düşer; aralıklar daralır ama yanlış değerin çevresinde. ")
    if first["kisa_kapsama"] - last["kisa_kapsama"] > 0.01:
        text += ("Bu yüzden kısa modelde gerçek katsayıyı kapsayan aralıkların payı "
                 f"%{plain(100 * first['kisa_kapsama'], 1)} değerinden %{plain(100 * last['kisa_kapsama'], 1)} "
                 "değerine iner")
    else:
        text += (f"Kısa modelde kapsama oranı en küçük örneklemde %{plain(100 * first['kisa_kapsama'], 1)}, en büyük "
                 f"örneklemde %{plain(100 * last['kisa_kapsama'], 1)}; yanlılık en küçük örneklemde bile aralıkların "
                 "gerçek değeri kaçırmasına yeter")
    return text + (f"; uzun modelde %{plain(100 * last['uzun_kapsama'], 1)}. Büyük örneklem örnekleme belirsizliğini "
                   "azaltır; eksik değişken yanlılığını, içselliği ya da yanlış fonksiyonel biçimi düzeltmez (§8.11).")


OMITTED = SimExperiment(
    topic_key=TOPIC,
    number=3,
    title="Büyük örneklem neyi çözmez?",
    question="Önemli bir değişken modelden dışarıda kalmışsa örneklemi büyütmek sorunu çözer mi? Güven aralıkları "
             "daralırken gerçek katsayıyı kapsamaya devam eder mi?",
    note=NoteRef("8.11"),
    parameters=(
        SimParameter("gamma", "Dışarıda kalan Z'nin etkisi γ", 0.0, 1.0, 0.5, 0.1, "Z'nin Y üzerindeki etkisi.",
                     decimals=1),
        SimParameter("rho", "X ile Z arasındaki korelasyon ρ", 0.0, 0.9, 0.5, 0.1,
                     "Kısa modelin yanlılığı γρ'dur.", decimals=1),
    ),
    dgp=_omitted_dgp,
    dgp_note=(
        "Notlarda bu deneyin sayısal karşılığı yoktur; §8.11'deki uyarıyı (büyük örneklem eksik değişken yanlılığını "
        "düzeltmez) tekrarlı örneklemede gösterir. Her n için üreteç tohum 305 ile başlar; her tekrarda sırasıyla X, e "
        "ve u çekilir. Eksik değişken yanlılığı Konu 6'da işlendi; burada soru büyük örneklemin ne yapıp ne "
        "yapamadığıdır."
    ),
    look_at=(
        "**Tablo** — n büyüdükçe kısa modelin tahminleri 0,5'e mi, başka bir değere mi yaklaşıyor?",
        "**Grafik** — kısa modelde kapsama oranı n ile nasıl değişiyor? Uzun modelde?",
        "**γ ve ρ** — birini sıfır yapın: yanlılık ve kapsama sorunu ortadan kalkıyor mu?",
    ),
    build=_build_omitted,
    metrics=_omitted_metrics,
    takeaway=_omitted_takeaway,
    tables=(("buyuk_n", "Örneklem büyüklüğüne göre kısa ve uzun model"),),
    labels=(("x", "X"), ("e", "e"), ("z", "Z"), ("u", "u"), ("y", "Y"), ("n", "n"),
            ("kisa", "Kısa model: β̂"), ("uzun", "Uzun model: β̂"), ("kisa_kapsar", "Kısa model GA 0,5'i kapsar"),
            ("uzun_kapsar", "Uzun model GA 0,5'i kapsar"), ("kisa_ort", "Kısa: ortalama β̂"),
            ("kisa_sd", "Kısa: β̂'nin std. sapması"), ("kisa_kapsama", "Kısa: kapsama"),
            ("uzun_ort", "Uzun: ortalama β̂"), ("uzun_kapsama", "Uzun: kapsama")),
)


KONU08_EXPERIMENTS = (LARGE_SAMPLE, JOINT, OMITTED)
