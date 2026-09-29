"""Konu 7 Sezgi deneyleri: veri üretim süreci bilinen kontrollü simülasyonlar.

Deney 1  Standart hata ve güven aralığının kapsaması               (Notlar §7.2 ve §7.6, Tablo 7.1, Şekil 7.1 ve 7.3)
Deney 2  Testin hataları: boyut, güç ve p-değerlerinin dağılımı      (Notlar §7.3, Tablo 7.2; §7.5)
Deney 3  İstatistiksel anlamlılık ile iktisadi önem                   (Notlar §7.11)

Bölüm 7'nin benzetimi Deney 1'in varsayılan ayarlarıdır (tohum 305); notlardaki tablo ve şekiller
(``scripts/bolum7_uygulama.py``) aynı çekiliş sırasıyla üretilir ve testle karşılaştırılır. Deney 2 ve 3'ün notlarda
sayısal karşılığı yoktur; varsayılan ayarları bölümün kavramlarını göstermek için seçilmiştir. Deney 3'ün iki
senaryosu aynı tohumla başlar.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from core.labs import expr as E
from core.labs.runner import LabState
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, plain
from core.labs.spec import (
    OLS,
    Derive,
    Draw,
    Histogram,
    IntervalPlot,
    ModelValue,
    MonteCarlo,
    NewSample,
    NoteRef,
    Scalar,
    Statistic,
    SummaryTable,
)

SEED = 305
TOPIC = "konu07"


def _rounded(parameters: Parameters, key: str, decimals: int) -> float:
    """Kaydırıcı değeri, kodda 0,30000000000000004 gibi yazımlar olmasın diye yuvarlanır."""

    return round(float(parameters[key]), decimals)


def _tex(value: float, decimals: int = 2) -> str:
    """LaTeX içinde sondaki sıfırları atılmış sayı (0{,}5; 3)."""

    text = f"{value:.{decimals}f}"
    text = text.rstrip("0").rstrip(".") if "." in text else text  # 20 → "2" olmasın
    return "0" if text in ("-0", "") else text.replace(".", "{,}")


def _thousands(value: int) -> str:
    """Binlik ayırıcılı tam sayı (5.000)."""

    return f"{value:,}".replace(",", ".")


def _short(value: float, decimals: int = 2) -> str:
    """Sondaki sıfırları atılmış düz metin sayı (0,5; 3); yuvarlanınca sıfır olan değer "0" yazılır."""

    text = f"{value:.{decimals}f}"
    text = text.rstrip("0").rstrip(".") if "." in text else text  # 20 → "2" olmasın
    return "0" if text in ("-0", "0", "") else text.replace(".", ",").replace("-", "−")


# --- Deney 1: standart hata ve güven aralığının kapsaması -------------------------------------------------

REPS_1 = 5000
BETA_1 = 0.5
N_NOTES_1, N_MAX_1 = 50, 400
LEVEL_NOTES_1 = 95
ROWS_1 = 25
"""Tablo 7.1: Y = 1 + 0,5·X + u; 5.000 örneklem, her birinde 50 gözlem; yüzde 95 güven aralığı; Şekil 7.3 ilk 25
aralık."""


def _coverage_settings(parameters: Parameters) -> tuple[int, float, int]:
    return int(parameters["n"]), _rounded(parameters, "sigma", 1), int(parameters["duzey"])


def coverage_axis(n: int, sigma: float) -> tuple[float, float]:
    """Histogram ekseni: gerçek eğimin çevresinde tahminlerin yaklaşık standart sapmasının (σ/√n) beş katı."""

    half = 5 * sigma / n ** 0.5
    return round(BETA_1 - half, 2), round(BETA_1 + half, 2)


def critical_level(level: int) -> float:
    """İki taraflı kritik değerin olasılığı 1 − α/2 = (1 + düzey)/2 (yüzde 95 için 0,975)."""

    return round((100 + level) / 200, 6)


def _build_coverage(parameters: Parameters) -> tuple:
    n, sigma, level = _coverage_settings(parameters)
    low, high = coverage_axis(n, sigma)
    body = (
        NewSample("orneklem", n, None),
        Draw("orneklem", "x", "normal", 0, 1, "X ~ N(0, 1)"),
        Draw("orneklem", "u", "normal", 0, sigma, f"u ~ N(0, σ²), σ = {_short(sigma, 1)}"),
        Derive("orneklem", "y", E.add(E.add(1, E.mul(BETA_1, E.var("x"))), E.var("u")), "Y = 1 + 0,5X + u"),
        OLS("model", "orneklem", "y", ("x",), "EKK: y ~ x"),
        ModelValue("egim", "model", "coef", "Eğim tahmini β̂₁", term="x"),
        ModelValue("sh", "model", "se", "Raporlanan standart hata se(β̂₁)", term="x"),
        ModelValue("sd_artik", "model", "df_resid", "Artık serbestlik derecesi n − 2", decimals=0),
        Scalar("kritik", E.tinv(critical_level(level), E.ref("sd_artik")),
               f"Kritik değer t(n − 2), yüzde {level} iki taraflı"),
        Scalar("alt", E.sub(E.ref("egim"), E.mul(E.ref("kritik"), E.ref("sh"))), "Güven aralığının alt sınırı"),
        Scalar("ust", E.add(E.ref("egim"), E.mul(E.ref("kritik"), E.ref("sh"))), "Güven aralığının üst sınırı"),
    )
    collect = (
        ("b1", E.ref("egim")),
        ("sh", E.ref("sh")),
        ("alt", E.ref("alt")),
        ("ust", E.ref("ust")),
        ("kapsar", E.mul(E.compare("le", E.ref("alt"), BETA_1), E.compare("le", BETA_1, E.ref("ust")))),
    )
    note = "" if (n, sigma) == (N_NOTES_1, 1.0) else f" (n = {n}, σ = {_short(sigma, 1)})"
    return (
        MonteCarlo("tekrarlar", REPS_1, SEED, body, collect,
                   f"{_thousands(REPS_1)} örneklem, her birinde {n} gözlem: eğim, standart hata ve yüzde {level} "
                   "güven aralığı"),
        SummaryTable((("Tekrarlı örnekleme", "tekrarlar"),),
                     (("tekrar", "b1", "count"), ("ortalama", "b1", "mean"), ("std_sapma", "b1", "std"),
                      ("ort_sh", "sh", "mean"), ("kapsama", "kapsar", "mean")),
                     "tablo71", "Tablo 7.1 ve §7.6: standart hata ve kapsama oranı", decimals=3, heading="Benzetim",
                     column_decimals=(("kapsama", 4),)),
        Statistic("tekrarlar", "b1", "mean", "ort_b1", "Tahminlerin ortalaması", decimals=3),
        Statistic("tekrarlar", "b1", "std", "sd_b1", "Tahminlerin gerçek standart sapması", decimals=3),
        Statistic("tekrarlar", "sh", "mean", "ort_sh", "Ortalama raporlanan standart hata", decimals=3),
        Statistic("tekrarlar", "kapsar", "mean", "kapsama", "Kapsama oranı", decimals=4),
        Histogram("tekrarlar", (("b1", "Eğim tahminleri β̂₁"),), 45, low, high,
                  f"Şekil 7.1: eğim tahminlerinin örnekleme dağılımı{note}", "Tahmin edilen eğim β̂₁",
                  references=((BETA_1, "Gerçek eğim β₁ = 0,5"), ("ort_b1", "Tahminlerin ortalaması"))),
        IntervalPlot("tekrarlar", "b1", "alt", "ust", BETA_1, _interval_title(parameters),
                     f"Eğim için yüzde {level} güven aralığı", rows=ROWS_1),
    )


def _coverage_dgp(parameters: Parameters) -> tuple[str, ...]:
    n, sigma, level = _coverage_settings(parameters)
    return (
        rf"Y_i = 1 + 0{{,}}5\,X_i + u_i, \qquad X_i \sim N(0,\ 1), \qquad u_i \sim N(0,\ \sigma^2), \quad "
        rf"\sigma = {_tex(sigma, 1)}, \qquad n = {n}",
        rf"\text{{{_thousands(REPS_1)} örneklem; her birinde }} \widehat\beta_1,\ \operatorname{{se}}(\widehat\beta_1)"
        rf"\text{{ ve yüzde {level} güven aralığı }} \widehat\beta_1 \pm t_{{\alpha/2;\,n-2}}\,"
        r"\operatorname{se}(\widehat\beta_1)",
    )


def _coverage_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    _, _, level = _coverage_settings(parameters)
    s = state.scalars
    covered = int(state.plots["IntervalPlot:" + _interval_title(parameters)]["kapsiyor"].sum())
    return (
        SimMetric("Tahminlerin std. sapması", plain(s["sd_b1"], 3),
                  f"{_thousands(REPS_1)} tahminin gerçek yayılımı: örnekleme dağılımının standart sapması."),
        SimMetric("Ortalama standart hata", plain(s["ort_sh"], 3),
                  "Her örneklemde yazılımın raporladığı se(β̂₁) değerlerinin ortalaması."),
        SimMetric("Kapsama oranı", "%" + plain(100 * s["kapsama"], 2),
                  f"Gerçek eğimi kapsayan yüzde {level} güven aralıklarının payı."),
        SimMetric(f"İlk {ROWS_1} aralıkta kapsayan", str(covered), "Şekil 7.3'teki aralıklardan gerçek eğimi içerenler."),
    )


def _interval_title(parameters: Parameters) -> str:
    n, sigma, _ = _coverage_settings(parameters)
    note = "" if (n, sigma) == (N_NOTES_1, 1.0) else f" (n = {n}, σ = {_short(sigma, 1)})"
    return f"Şekil 7.3: aynı yöntemle kurulan ilk {ROWS_1} güven aralığı{note}"


def _coverage_takeaway(state: LabState, parameters: Parameters) -> str:
    n, _, level = _coverage_settings(parameters)
    s = state.scalars
    gap = abs(s["sd_b1"] - s["ort_sh"])
    closeness = "birbirine yakındır" if gap < 0.1 * s["sd_b1"] else "farklıdır"
    text = (f"Tahminlerin {_thousands(REPS_1)} tekrardaki gerçek standart sapması {plain(s['sd_b1'], 3)}, tek tek "
            f"örneklemlerde raporlanan standart hataların ortalaması {plain(s['ort_sh'], 3)}; iki sayı {closeness}. "
            "Standart hata, göremediğimiz örnekleme dağılımının yayılımını tek bir örneklemden tahmin eder (§7.2). ")
    covered = int(state.plots["IntervalPlot:" + _interval_title(parameters)]["kapsiyor"].sum())
    text += (f"Gerçek eğimi (β₁ = 0,5) kapsayan yüzde {level} güven aralıklarının payı %{plain(100 * s['kapsama'], 2)}. "
             "Bu, uzun dönem kapsama oranıdır; tek bir aralığın parametreyi içerme olasılığı değildir. ")
    if covered < ROWS_1:
        text += (f"Şekil 7.3'teki ilk {ROWS_1} aralıktan {ROWS_1 - covered} tanesi gerçek değeri kaçırır; yöntem her "
                 "tekil aralığın doğru olacağını garanti etmez (§7.6).")
    else:
        text += (f"Bu ayarda ilk {ROWS_1} aralığın hepsi gerçek değeri kapsıyor; yine de "
                 f"{_thousands(REPS_1)} aralığın %{plain(100 * (1 - s['kapsama']), 2)} kadarı kaçırır: yöntem her "
                 "tekil aralığın doğru olacağını garanti etmez (§7.6).")
    if n < N_MAX_1:
        text += " n'yi büyütün: aralıklar daralır, kapsama oranı güven düzeyinin çevresinde kalır."
    return text


COVERAGE = SimExperiment(
    topic_key=TOPIC,
    number=1,
    title="Standart hata ve güven aralığının kapsaması",
    question="Aynı veri üretim sürecinden binlerce örneklem çekseydik eğim tahminleri ne kadar yayılırdı? Yazılımın "
             "raporladığı standart hata bu yayılımı ne kadar iyi tahmin eder? Yüzde 95 güven aralıklarının kaçı gerçek "
             "eğimi kapsar?",
    note=NoteRef("7.2", objects=("Tablo 7.1", "Şekil 7.1", "§7.6", "Şekil 7.3")),
    parameters=(
        SimParameter("n", "Örneklem büyüklüğü n", 10, N_MAX_1, N_NOTES_1, 10,
                     "Her örneklemdeki gözlem sayısı; notlarda 50.", integer=True, decimals=0),
        SimParameter("sigma", "Hata teriminin standart sapması σ", 0.5, 3.0, 1.0, 0.5, "Notlarda 1.", decimals=1),
        SimParameter("duzey", "Güven düzeyi (%)", 80, 99, LEVEL_NOTES_1, 1, "Notlarda yüzde 95.", integer=True,
                     decimals=0),
    ),
    dgp=_coverage_dgp,
    dgp_note=(
        "Varsayılan ayarlar notlardaki benzetimdir (Tablo 7.1, Şekil 7.1 ve 7.3; tohum 305). Her tekrarda önce X, "
        "sonra u çekilir. Uygulama tekrarları vektörel hesaplar; üretilen kod notlardaki gibi döngüyle aynı sayıları "
        "verir."
    ),
    look_at=(
        "**Histogram** — tahminler gerçek eğim 0,5'in çevresinde ne kadar yayılıyor?",
        "**Tablo** — tahminlerin gerçek standart sapması ile ortalama raporlanan standart hata yakın mı?",
        "**Aralıklar** — hangi aralıklar gerçek değeri kaçırıyor? Kapsama oranı güven düzeyine yakın mı?",
    ),
    build=_build_coverage,
    metrics=_coverage_metrics,
    takeaway=_coverage_takeaway,
    tables=(("tablo71", "Tablo 7.1 ve §7.6: tekrarlı örneklemede standart hata ve kapsama oranı",
             ("n", "sigma", "duzey")),),
    labels=(("x", "X"), ("u", "u"), ("y", "Y"), ("b1", "Eğim tahmini β̂₁"), ("sh", "Standart hata"),
            ("tekrar", "Tekrar sayısı"), ("ortalama", "Tahminlerin ortalaması"),
            ("std_sapma", "Tahminlerin gerçek std. sapması"), ("ort_sh", "Ortalama raporlanan SH"),
            ("kapsama", "Kapsama oranı")),
)


# --- Deney 2: testin hataları ---------------------------------------------------------------------------

REPS_2 = 4000
BINS_2 = 20
"""Y = 1 + β₁X + u, X, u ~ N(0, 1); her örneklemde H₀: β₁ = 0 iki taraflı t testi. β₁ = 0'da H₀ doğrudur."""


def _errors_settings(parameters: Parameters) -> tuple[float, int, int]:
    return _rounded(parameters, "beta", 2), int(parameters["n"]), int(parameters["alfa"])


POWER_GRID = 200
"""Kuramsal gücün hesabında χ²(n − 1) dağılımının eşit olasılıklı nokta sayısı."""


def theoretical_power(beta: float, n: int, alpha: float) -> float:
    """İki taraflı t testinin kuramsal gücü (Var(X) = Var(u) = 1). X verilmişken t istatistiği serbestlik derecesi
    n − 2 ve merkezî olmama parametresi β₁·√(Σ(Xᵢ − X̄)²) olan merkezî olmayan t dağılımındadır; Σ(Xᵢ − X̄)² ~ χ²(n − 1)
    olduğu için güç, bu dağılımın ``POWER_GRID`` eşit olasılıklı noktası üzerinden ortalanır. β₁ = 0'da α'dır."""

    critical = stats.t.ppf(1 - alpha / 2, n - 2)
    spread = stats.chi2.ppf((np.arange(POWER_GRID) + 0.5) / POWER_GRID, n - 1)
    shift = abs(beta) * np.sqrt(spread)  # simetri: β₁ ile −β₁ aynı gücü verir
    # Uzak kuyruk: büyük kaymada nct.sf(c, ν, −δ) sayısal olarak NaN dönebilir; değer fiilen sıfırdır.
    far = np.nan_to_num(stats.nct.sf(critical, n - 2, -shift), nan=0.0)
    return float(np.mean(stats.nct.sf(critical, n - 2, shift) + far))


def _build_errors(parameters: Parameters) -> tuple:
    beta, n, level = _errors_settings(parameters)
    alpha = level / 100
    body = (
        NewSample("orneklem", n, None),
        Draw("orneklem", "x", "normal", 0, 1, "X ~ N(0, 1)"),
        Draw("orneklem", "u", "normal", 0, 1, "u ~ N(0, 1)"),
        Derive("orneklem", "y", E.add(E.add(1, E.mul(beta, E.var("x"))), E.var("u")), f"Y = 1 + {_short(beta)}X + u"),
        OLS("model", "orneklem", "y", ("x",), "EKK: y ~ x"),
        ModelValue("p_x", "model", "p", "H₀: β₁ = 0 için iki taraflı p-değeri", term="x"),
        ModelValue("t_x", "model", "t", "t istatistiği", term="x"),
    )
    collect = (("p", E.ref("p_x")), ("t", E.ref("t_x")), ("reddet", E.compare("lt", E.ref("p_x"), alpha)))
    truth = "H₀ doğru" if beta == 0 else "H₀ yanlış"
    return (
        MonteCarlo("tekrarlar", REPS_2, SEED, body, collect,
                   f"{_thousands(REPS_2)} örneklem, her birinde {n} gözlem: H₀: β₁ = 0 için yüzde {level} t testi"),
        SummaryTable(((f"β₁ = {_short(beta)} ({truth})", "tekrarlar"),),
                     (("tekrar", "reddet", "count"), ("ret_orani", "reddet", "mean"), ("ort_p", "p", "mean"),
                      ("medyan_p", "p", "median")),
                     "testler", "Tekrarlı örneklemede H₀: β₁ = 0 testinin kararları", decimals=3,
                     heading="Gerçek eğim", column_decimals=(("ret_orani", 4),)),
        Statistic("tekrarlar", "reddet", "mean", "ret", "H₀ ret oranı", decimals=4),
        Histogram("tekrarlar", (("p", "p-değerleri"),), BINS_2, 0, 1,
                  f"p-değerlerinin dağılımı (β₁ = {_short(beta)}, n = {n})", "H₀: β₁ = 0 testinin p-değeri",
                  references=((alpha, f"α = {_short(alpha)}"),),
                  curves=(("uniform", 0, 1, "H₀ doğruyken beklenen sayı (düzgün dağılım)"),)),
    )


def _errors_dgp(parameters: Parameters) -> tuple[str, ...]:
    beta, n, level = _errors_settings(parameters)
    return (
        rf"Y_i = 1 + \beta_1 X_i + u_i, \qquad X_i,\ u_i \sim N(0,\ 1), \qquad \beta_1 = {_tex(beta)}, \qquad n = {n}",
        rf"H_0: \beta_1 = 0 \text{{ karşısında }} H_1: \beta_1 \neq 0, \quad \alpha = {_tex(level / 100)}; \quad "
        rf"\text{{{_thousands(REPS_2)} örneklem}}",
    )


def _errors_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    beta, n, level = _errors_settings(parameters)
    rate = state.scalars["ret"]
    median = SimMetric("Medyan p-değeri", plain(state.tables["testler"]["medyan_p"].iloc[0], 3),
                       "H₀ doğruyken yaklaşık 0,5; H₀ yanlışken güç arttıkça sıfıra yaklaşır.")
    if beta == 0:
        return (
            SimMetric("I. tür hata oranı", "%" + plain(100 * rate, 1),
                      "H₀ doğruyken reddetme oranı (testin boyutu); hedef α."),
            SimMetric("Kuramsal boyut α", "%" + plain(level, 0), "H₀ doğruyken beklenen ret oranı."),
            SimMetric("Reddedilmeyen tekrar", "%" + plain(100 * (1 - rate), 1), "Doğru karar: H₀ reddedilmez."),
            median,
        )
    expected = theoretical_power(beta, n, level / 100)
    return (
        SimMetric("Güç (ret oranı)", "%" + plain(100 * rate, 1),
                  "H₀ yanlışken reddetme oranı: testin gücünün benzetim karşılığı."),
        SimMetric("Kuramsal güç", "%" + plain(100 * expected, 1),
                  "X verilmişken merkezî olmayan t dağılımından, X'in dağılımı üzerinden ortalanarak hesaplanır."),
        SimMetric("II. tür hata oranı", "%" + plain(100 * (1 - rate), 1),
                  "H₀ yanlışken reddedilmeyen tekrarların payı."),
        median,
    )


def _errors_takeaway(state: LabState, parameters: Parameters) -> str:
    beta, _, level = _errors_settings(parameters)
    rate = state.scalars["ret"]
    if beta == 0:
        return (f"H₀ doğru: {_thousands(REPS_2)} örneklemin %{plain(100 * rate, 1)} kadarında H₀ reddedildi. Bu "
                f"reddetmelerin hepsi I. tür hatadır ve oran seçilen anlamlılık düzeyinin (%{level}) çevresindedir. "
                "p-değerleri 0 ile 1 arasında yaklaşık düzgün dağılır: H₀ doğruyken p < α olma olasılığı α'dır (§7.3, "
                "§7.5). β₁'i sıfırdan uzaklaştırın: ret oranı artık testin gücüdür.")
    median = state.tables["testler"]["medyan_p"].iloc[0]
    if median < 0.05:
        shape = f"p-değerleri sıfırın yakınında yığılır (medyan p = {plain(median, 3)}). "
    elif median < 0.45:
        shape = (f"p-değerleri sıfıra doğru kayar ama 0 ile 1 arasına yayılmayı sürdürür (medyan p = "
                 f"{plain(median, 3)}). ")
    else:
        shape = (f"Güç çok düşük: p-değerlerinin dağılımı H₀ doğruyken olduğundan neredeyse ayırt edilemez (medyan "
                 f"p = {plain(median, 3)}). ")
    return (f"H₀ yanlış (β₁ = {_short(beta)}): örneklemlerin %{plain(100 * rate, 1)} kadarında H₀ reddedildi; bu "
            f"testin gücüdür. Kalan %{plain(100 * (1 - rate), 1)} kadarında H₀ reddedilemedi: II. tür hata. " + shape
            + "Güç, |β₁| büyüdükçe ve örneklem büyüdükçe artar; α küçüldükçe azalır. Reddedememek H₀'ın doğru "
            "olduğunu göstermez (§7.3, §7.5).")


ERRORS = SimExperiment(
    topic_key=TOPIC,
    number=2,
    title="Testin hataları: boyut, güç ve p-değerleri",
    question="Sıfır hipotezi doğruyken yüzde 5 testi onu ne sıklıkla yanlışlıkla reddeder? Hipotez yanlışken doğru "
             "kararı ne sıklıkla verir? p-değerleri iki durumda nasıl dağılır?",
    note=NoteRef("7.3", objects=("Tablo 7.2", "§7.5")),
    parameters=(
        SimParameter("beta", "Gerçek eğim β₁", -1.0, 1.0, 0.0, 0.05, "0: sıfır hipotezi doğru.", decimals=2),
        SimParameter("n", "Örneklem büyüklüğü n", 10, 200, 30, 10, "Her örneklemdeki gözlem sayısı.", integer=True,
                     decimals=0),
        SimParameter("alfa", "Anlamlılık düzeyi α (%)", 1, 10, 5, 1, "İki taraflı testin anlamlılık düzeyi.",
                     integer=True, decimals=0),
    ),
    dgp=_errors_dgp,
    dgp_note=(
        "Notlarda bu deneyin sayısal karşılığı yoktur; Tablo 7.2'deki iki hata türünü tekrarlı örneklemede gösterir. "
        "Her tekrarda önce X, sonra u çekilir (tohum 305). Uygulama tekrarları vektörel hesaplar; üretilen kod döngüyle "
        "aynı sayıları verir."
    ),
    look_at=(
        "**Ret oranı** — β₁ = 0 iken α'ya yakın mı? β₁ ≠ 0 iken (güç) n ile nasıl değişiyor?",
        "**Histogram** — H₀ doğruyken p-değerleri 0 ile 1 arasında yayılıyor mu? H₀ yanlışken sıfıra mı yığılıyor?",
        "**α** — anlamlılık düzeyini küçültün: I. tür hata azalırken güç ne oluyor?",
    ),
    build=_build_errors,
    metrics=_errors_metrics,
    takeaway=_errors_takeaway,
    tables=(("testler", "Tekrarlı örneklemede testin kararları"),),
    labels=(("x", "X"), ("u", "u"), ("y", "Y"), ("p", "p-değeri"), ("t", "t istatistiği"),
            ("reddet", "H₀ reddedildi (p < α)"), ("tekrar", "Tekrar sayısı"), ("ret_orani", "H₀ ret oranı"),
            ("ort_p", "Ortalama p-değeri"), ("medyan_p", "Medyan p-değeri")),
)


# --- Deney 3: istatistiksel anlamlılık ile iktisadi önem ------------------------------------------------

REPS_3 = 500
ROWS_3 = 20
"""İki senaryo: A küçük etki ve büyük örneklem, B büyük etki ve küçük örneklem; Y = 1 + β₁X + u, X, u ~ N(0, 1);
her senaryoda 500 örneklem (aynı tohumla başlar)."""


def _importance_settings(parameters: Parameters) -> tuple[float, int, float, int]:
    return (_rounded(parameters, "beta_a", 2), int(parameters["n_a"]), _rounded(parameters, "beta_b", 1),
            int(parameters["n_b"]))


def _scenario(name: str, beta: float, n: int, label: str) -> MonteCarlo:
    body = (
        NewSample("orneklem", n, None),
        Draw("orneklem", "x", "normal", 0, 1, "X ~ N(0, 1)"),
        Draw("orneklem", "u", "normal", 0, 1, "u ~ N(0, 1)"),
        Derive("orneklem", "y", E.add(E.add(1, E.mul(beta, E.var("x"))), E.var("u")), f"Y = 1 + {_short(beta)}X + u"),
        OLS("model", "orneklem", "y", ("x",), "EKK: y ~ x"),
        ModelValue("egim", "model", "coef", "Eğim tahmini β̂₁", term="x"),
        ModelValue("sh", "model", "se", "Standart hata", term="x"),
        ModelValue("p_x", "model", "p", "H₀: β₁ = 0 için p-değeri", term="x"),
        ModelValue("ga_alt", "model", "ci_low", "Yüzde 95 GA alt sınırı", term="x"),
        ModelValue("ga_ust", "model", "ci_high", "Yüzde 95 GA üst sınırı", term="x"),
    )
    collect = (("b1", E.ref("egim")), ("sh", E.ref("sh")), ("p", E.ref("p_x")),
               ("reddet", E.compare("lt", E.ref("p_x"), 0.05)), ("alt", E.ref("ga_alt")), ("ust", E.ref("ga_ust")),
               ("genislik", E.sub(E.ref("ga_ust"), E.ref("ga_alt"))))
    return MonteCarlo(name, REPS_3, SEED, body, collect,
                      f"{label}: {REPS_3} örneklem, her birinde {_thousands(n)} gözlem")


def _label_a(beta: float, n: int) -> str:
    return f"A: küçük etki, büyük örneklem (β₁ = {_short(beta)}, n = {_thousands(n)})"


def _label_b(beta: float, n: int) -> str:
    return f"B: büyük etki, küçük örneklem (β₁ = {_short(beta)}, n = {_thousands(n)})"


def _build_importance(parameters: Parameters) -> tuple:
    beta_a, n_a, beta_b, n_b = _importance_settings(parameters)
    zero = (0.0, "Sıfır: aralık sıfırı kesiyorsa H₀ reddedilemez")
    return (
        _scenario("senaryo_a", beta_a, n_a, "A"),
        _scenario("senaryo_b", beta_b, n_b, "B"),
        SummaryTable((("A: küçük etki, büyük örneklem", "senaryo_a"), ("B: büyük etki, küçük örneklem", "senaryo_b")),
                     (("ort_tahmin", "b1", "mean"), ("ort_sh", "sh", "mean"), ("ret_orani", "reddet", "mean"),
                      ("ort_genislik", "genislik", "mean")),
                     "onem", "Anlamlılık ve büyüklük: iki senaryo", decimals=3, heading="Senaryo"),
        Statistic("senaryo_a", "reddet", "mean", "ret_a", "A: H₀ ret oranı", decimals=3),
        Statistic("senaryo_b", "reddet", "mean", "ret_b", "B: H₀ ret oranı", decimals=3),
        IntervalPlot("senaryo_a", "b1", "alt", "ust", beta_a, _label_a(beta_a, n_a),
                     "Eğim için yüzde 95 güven aralığı", rows=ROWS_3, reference=zero),
        IntervalPlot("senaryo_b", "b1", "alt", "ust", beta_b, _label_b(beta_b, n_b),
                     "Eğim için yüzde 95 güven aralığı", rows=ROWS_3, reference=zero),
    )


def _importance_dgp(parameters: Parameters) -> tuple[str, ...]:
    beta_a, n_a, beta_b, n_b = _importance_settings(parameters)
    return (
        r"Y_i = 1 + \beta_1 X_i + u_i, \qquad X_i,\ u_i \sim N(0,\ 1)",
        rf"\text{{A: }} \beta_1 = {_tex(beta_a)},\ n = {_thousands(n_a)}; \qquad \text{{B: }} \beta_1 = {_tex(beta_b)},\ "
        rf"n = {n_b}",
        rf"\text{{her senaryoda {REPS_3} örneklem; }} H_0: \beta_1 = 0 \text{{ yüzde 5 düzeyinde}}",
    )


def _importance_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    beta_a, _, beta_b, _ = _importance_settings(parameters)
    s = state.scalars
    table = state.tables["onem"]
    return (
        SimMetric("A: H₀ ret oranı", "%" + plain(100 * s["ret_a"], 1), f"Gerçek eğim {_short(beta_a)}."),
        SimMetric("B: H₀ ret oranı", "%" + plain(100 * s["ret_b"], 1), f"Gerçek eğim {_short(beta_b)}."),
        SimMetric("A: ortalama GA genişliği", plain(table["ort_genislik"].iloc[0], 3),
                  "Büyük örneklemde aralık dardır: kesin tahmin."),
        SimMetric("B: ortalama GA genişliği", plain(table["ort_genislik"].iloc[1], 3),
                  "A'dakiyle karşılaştırın: aralık genişledikçe tahmin belirsizleşir."),
    )


def _importance_takeaway(state: LabState, parameters: Parameters) -> str:
    beta_a, n_a, beta_b, n_b = _importance_settings(parameters)
    s = state.scalars
    text = (f"A'da gerçek eğim {_short(beta_a)}: X'teki bir standart sapmalık fark, Y'de hatanın standart sapmasının "
            f"yalnız {_short(beta_a)} katı kadar bir farkla ilişkilidir. ")
    if s["ret_a"] >= 0.5:
        text += (f"Yine de n = {_thousands(n_a)} olduğu için örneklemlerin %{plain(100 * s['ret_a'], 1)} kadarında "
                 "H₀: β₁ = 0 reddedilir. ")
    elif s["ret_a"] >= 0.1:
        text += (f"n = {_thousands(n_a)} iken örneklemlerin yalnız %{plain(100 * s['ret_a'], 1)} kadarında reddedilir: "
                 "bu örneklem bu kadar küçük bir etkiyi çoğu zaman sıfırdan ayırt edemez. ")
    else:
        text += (f"n = {_thousands(n_a)} iken ret oranı %{plain(100 * s['ret_a'], 1)}; α'ya yakın: bu örneklem bu kadar "
                 "küçük bir etkiyi sıfırdan ayırt edemez. ")
    if s["ret_b"] >= 0.8:
        text += (f"B'de gerçek eğim {_short(beta_b)}; n = {n_b} bu büyüklükteki bir etkiyi neredeyse her örneklemde "
                 f"sıfırdan ayırt eder (%{plain(100 * s['ret_b'], 1)}). ")
    else:
        text += (f"B'de gerçek eğim {_short(beta_b)}; n = {n_b} olduğu için örneklemlerin yalnız "
                 f"%{plain(100 * s['ret_b'], 1)} kadarında reddedilir. ")
    return text + ("Küçük p-değeri etkinin büyük olduğunu, reddedememek etkinin olmadığını göstermez. p-değeri "
                   "örneklemdeki kanıtın gücüne bağlıdır; iktisadi önem katsayının büyüklüğü, güven aralığı ve "
                   "bağlamla değerlendirilir (§7.11).")


IMPORTANCE = SimExperiment(
    topic_key=TOPIC,
    number=3,
    title="İstatistiksel anlamlılık ile iktisadi önem",
    question="Çok büyük bir örneklemde iktisadi olarak küçük bir etki istatistiksel olarak anlamlı çıkabilir mi? Küçük "
             "bir örneklemde büyük bir etki anlamsız görünebilir mi?",
    note=NoteRef("7.11"),
    parameters=(
        SimParameter("beta_a", "A: küçük gerçek eğim β₁", 0.01, 0.1, 0.05, 0.01, "Senaryo A'nın gerçek eğimi.",
                     decimals=2),
        SimParameter("n_a", "A: büyük örneklem n", 500, 5000, 4000, 500, "Senaryo A'nın gözlem sayısı.", integer=True,
                     decimals=0),
        SimParameter("beta_b", "B: büyük gerçek eğim β₁", 0.3, 1.0, 0.5, 0.1, "Senaryo B'nin gerçek eğimi.",
                     decimals=1),
        SimParameter("n_b", "B: küçük örneklem n", 10, 60, 15, 5, "Senaryo B'nin gözlem sayısı.", integer=True,
                     decimals=0),
    ),
    dgp=_importance_dgp,
    dgp_note=(
        "Notlarda bu deneyin sayısal karşılığı yoktur; §7.11'deki iki uyarıyı tekrarlı örneklemede gösterir. Her "
        "senaryo tohum 305 ile başlar; her tekrarda önce X, sonra u çekilir."
    ),
    look_at=(
        "**Tablo** — hangi senaryoda gerçek etki büyük, hangisinde H₀ daha sık reddediliyor?",
        "**Aralıklar** — A'nın dar aralıkları sıfırın yakınında mı? B'nin geniş aralıkları sıfırı kesiyor mu?",
        "**n** — A'da n'yi küçültün, B'de büyütün: ret oranları nasıl değişiyor?",
    ),
    build=_build_importance,
    metrics=_importance_metrics,
    takeaway=_importance_takeaway,
    tables=(("onem", "İki senaryoda tahmin, belirsizlik ve H₀ kararları"),),
    labels=(("x", "X"), ("u", "u"), ("y", "Y"), ("b1", "Eğim tahmini β̂₁"), ("sh", "Standart hata"),
            ("p", "p-değeri"), ("reddet", "H₀ reddedildi (p < 0,05)"), ("alt", "GA alt sınırı"),
            ("ust", "GA üst sınırı"), ("genislik", "GA genişliği"), ("ort_tahmin", "Ortalama tahmin"),
            ("ort_sh", "Ortalama SH"), ("ret_orani", "H₀ ret oranı"), ("ort_genislik", "Ortalama GA genişliği")),
)


KONU07_EXPERIMENTS = (COVERAGE, ERRORS, IMPORTANCE)
