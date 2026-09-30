"""Konu 12 Sezgi deneyleri: veri üretim süreci bilinen kontrollü simülasyonlar.

Deney 1  Katsayı doğru merkezde, aralık yanlış genişlikte   (Notlar §12.1 ve §12.9; Şekil 12.1–12.2, Tablo 12.6,
                                                             Şekil 12.7–12.8)
Deney 2  Breusch–Pagan ve White testleri neyi yakalar?      (Notlar §12.5)
Deney 3  Küçük örneklemde HC0, HC1, HC2 ve HC3              (Notlar §12.6, Tablo 12.3)

Bölüm 12'nin benzetimi (§12.9) ve §12.1'deki iki örnek şekil Deney 1'in varsayılan ayarlarıdır (tohum 305); notlardaki
tablo ve şekiller (``scripts/bolum12_uygulama.py``) aynı çekiliş sırasıyla üretilir ve testle karşılaştırılır. Deney 2 ve
3'ün notlarda sayısal karşılığı yoktur; varsayılan ayarları bölümün kavramlarını göstermek için seçilmiştir.
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
    HeteroskedasticityTest,
    Histogram,
    JoinColumns,
    ModelValue,
    MonteCarlo,
    NewSample,
    NoteRef,
    Residuals,
    Scalar,
    ScalarTable,
    ScatterPlot,
    Statistic,
)

SEED = 305
TOPIC = "konu12"
LEVEL = 95
"""Güven aralıklarının düzeyi (yüzde); testlerin anlamlılık düzeyi yüzde 5."""


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


def _covers(low: str, high: str, truth: float) -> E.Expr:
    """Güven aralığı gerçek değeri kapsıyorsa 1: alt ≤ gerçek ≤ üst."""

    return E.mul(E.compare("le", E.ref(low), truth), E.compare("le", truth, E.ref(high)))


def _percent(value: float, decimals: int = 2) -> str:
    return "%" + plain(value, decimals)


def _inference(covariances: tuple, truth: float) -> tuple[list, list]:
    """Tekrarın ``model``'i için eğim, kritik değer ve her kovaryans türüyle standart hata ve yüzde 95 güven aralığı
    (statsmodels ``conf_int`` gibi: β̂ ± t(0,975; n − 2)·SH) ile toplanan sütunlar: eğim, SH'ler ve kapsama
    göstergeleri."""

    body = [ModelValue("egim", "model", "coef", "Eğim tahmini β̂₁", term="x"),
            ModelValue("sd_artik", "model", "df_resid", "Artık serbestlik derecesi n − 2", decimals=0),
            Scalar("kritik", E.tinv(0.975, E.ref("sd_artik")), "Kritik değer t(0,975; n − 2)")]
    collect = [("b1", E.ref("egim"))]
    for suffix, label, cov in covariances:
        margin = E.mul(E.ref("kritik"), E.ref(f"sh_{suffix}"))
        body += [
            ModelValue(f"sh_{suffix}", "model", "se", f"{label} standart hata", term="x", cov_type=cov),
            Scalar(f"alt_{suffix}", E.sub(E.ref("egim"), margin), f"{label} yüzde 95 GA alt sınırı"),
            Scalar(f"ust_{suffix}", E.add(E.ref("egim"), margin), f"{label} yüzde 95 GA üst sınırı"),
        ]
        collect += [(f"sh_{suffix}", E.ref(f"sh_{suffix}")),
                    (f"kapsar_{suffix}", _covers(f"alt_{suffix}", f"ust_{suffix}", truth))]
    return body, collect


def _lower_first(label: str) -> str:
    """Cümle içindeki etiket: "Geleneksel" → "geleneksel"; "HC1" olduğu gibi."""

    return label[:1].lower() + label[1:] if label[1:2].islower() else label


def _p_value(value: float) -> str:
    """Ölçü kutusunda p-değeri: üç basamakta sıfıra yuvarlanıyorsa "< 0,001"."""

    return "< 0,001" if value < 0.0005 else plain(value, 3)


# --- Deney 1: katsayı doğru merkezde, aralık yanlış genişlikte ------------------------------------------------

REPS_1 = 4000
A_1 = 0.3
BETA_1 = 2.0
X_HIGH_1 = 4
N_NOTES_1, C_NOTES_1 = 60, 0.7
"""§12.9: Y = 1 + 2X + u, u = (0,3 + 0,7X²)·Z, X ~ U(0, 4), Z ~ N(0, 1); 4.000 örneklem, her birinde 60 gözlem."""
X2_MEAN, X4_MEAN = 16 / 3, 256 / 5
"""X ~ U(0, 4) için E[X²] = 16/3 ve E[X⁴] = 256/5."""
COVARIANCES_1 = (("gel", "Geleneksel", None), ("hc1", "HC1", "HC1"), ("hc3", "HC3", "HC3"))
"""(ad eki, etiket, kovaryans türü): Tablo 12.6'daki üç standart hata."""


def _hetero_settings(parameters: Parameters) -> tuple[int, float]:
    return int(parameters["n"]), _rounded(parameters, "c", 2)


def average_sd(c: float) -> float:
    """Hatanın koşulsuz standart sapması σ̄ = √E[(0,3 + cX²)²] = √(0,3² + 2·0,3·c·E[X²] + c²·E[X⁴]). Şekil 12.1'in
    homoskedastik süreci bu sabit standart sapmayı kullanır: iki süreçte hatanın koşulsuz varyansı aynıdır."""

    return math.sqrt(A_1**2 + 2 * A_1 * c * X2_MEAN + c**2 * X4_MEAN)


def slope_sd(n: int, c: float) -> float:
    """Eğim tahmininin büyük örneklem standart sapması: Var(β̂₁) ≈ E[(X − 2)²σ²(X)] / (n·Var(X)²), σ(X) = 0,3 + cX²,
    Var(X) = 4/3. Beklenen değer U(0, 4) üzerinde polinomun kesin integralidir; histogram ekseni için kullanılır."""

    # (x − 2)²(a + cx²)² polinomunun katsayıları (sabitten başlayarak), sonra ∫₀⁴ … dx / 4
    square = (4.0, -4.0, 1.0)
    variance = (A_1**2, 0.0, 2 * A_1 * c, 0.0, c**2)
    product = [0.0] * (len(square) + len(variance) - 1)
    for i, left in enumerate(square):
        for j, right in enumerate(variance):
            product[i + j] += left * right
    moment = sum(coefficient * X_HIGH_1 ** (power + 1) / (power + 1) for power, coefficient in enumerate(product))
    moment /= X_HIGH_1
    return math.sqrt(moment / (n * (X_HIGH_1**2 / 12) ** 2))


def hetero_axis(n: int, c: float) -> tuple[float, float]:
    """Histogram ekseni: gerçek eğimin çevresinde büyük örneklem standart sapmasının beş katı."""

    half = 5 * slope_sd(n, c)
    return round(BETA_1 - half, 1), round(BETA_1 + half, 1)


def _settings_note_1(n: int, c: float) -> str:
    return "" if (n, c) == (N_NOTES_1, C_NOTES_1) else f" (n = {n}, c = {_short(c)})"


def _sd_expression(c: float) -> E.Expr:
    """Hatanın koşullu standart sapması 0,3 + cX²; c = 0'da sıfır terim yazılmaz (sayılar aynı)."""

    return E.const(A_1) if c == 0 else E.add(A_1, E.mul(c, E.power(E.var("x"), 2)))


def _outcome(c: float) -> E.Expr:
    """Y = 1 + 2X + (0,3 + cX²)·Z (notlardaki betikle aynı işlem sırası)."""

    return E.add(E.add(1, E.mul(BETA_1, E.var("x"))), E.mul(_sd_expression(c), E.var("z")))


def _outcome_text(c: float) -> str:
    return "Y = 1 + 2X + 0,3·Z" if c == 0 else f"Y = 1 + 2X + (0,3 + {_short(c)}X²)·Z"


def _draws(frame: str) -> tuple:
    return (
        Draw(frame, "x", "uniform", 0, X_HIGH_1, "X ~ U(0, 4)"),
        Draw(frame, "z", "normal", 0, 1, "Z ~ N(0, 1)"),
    )


def _build_hetero(parameters: Parameters) -> tuple:
    n, c = _hetero_settings(parameters)
    note = _settings_note_1(n, c)
    low, high = hetero_axis(n, c)
    sigma_bar = (E.const(A_1) if c == 0 else  # σ̄ = √E[(0,3 + cX²)²]; c = 0'da 0,3
                 E.sqrt(E.add(E.add(E.power(A_1, 2), E.mul(E.mul(E.mul(2, A_1), c), E.div(16, 3))),
                              E.mul(E.power(c, 2), E.div(256, 5)))))
    inference, collect = _inference(COVARIANCES_1, BETA_1)
    body = [NewSample("tekrar_orneklem", n, None), *_draws("tekrar_orneklem"),
            Derive("tekrar_orneklem", "y", _outcome(c), _outcome_text(c)),
            OLS("model", "tekrar_orneklem", "y", ("x",), "EKK: y ~ x"), *inference]
    homo_title = f"Şekil 12.1: homoskedastik süreç, yayılım X boyunca sabit{note}"
    hetero_title = (f"Şekil 12.2: heteroskedastik süreç, yayılım X ile büyüyor{note}" if c > 0
                    else f"Seçilen süreç (c = 0): hata varyansı sabit, σ = 0,3 (n = {n})")
    truth = ((1.0, BETA_1, "Gerçek koşullu ortalama E(Y | X) = 1 + 2X"),)
    return (
        NewSample("orneklem", n, SEED),
        *_draws("orneklem"),
        Scalar("sigma_ort", sigma_bar, "Hatanın koşulsuz std. sapması σ̄ = √E[(0,3 + cX²)²]; E[X²] = 16/3, E[X⁴] = 256/5",
               decimals=4),
        Derive("orneklem", "y_homo", E.add(E.add(1, E.mul(BETA_1, E.var("x"))), E.mul(E.ref("sigma_ort"), E.var("z"))),
               "Homoskedastik karşılaştırma (aynı X ve Z): Y = 1 + 2X + σ̄·Z"),
        Derive("orneklem", "y", _outcome(c), _outcome_text(c)),
        ScatterPlot("orneklem", "x", "y_homo", "X", "Y", homo_title, size=9, opacity=0.8, lines=truth),
        ScatterPlot("orneklem", "x", "y", "X", "Y", hetero_title, size=9, opacity=0.8, lines=truth),
        MonteCarlo("tekrarlar", REPS_1, SEED, tuple(body), tuple(collect),
                   f"{_thousands(REPS_1)} örneklem, her birinde {n} gözlem: eğim, geleneksel, HC1 ve HC3 standart hata ve "
                   "yüzde 95 güven aralıkları"),
        Statistic("tekrarlar", "b1", "mean", "ort_b1", "Eğim tahminlerinin ortalaması", decimals=3),
        Statistic("tekrarlar", "b1", "std", "sd_b1", "Eğim tahminlerinin ampirik standart sapması", decimals=3),
        *(Statistic("tekrarlar", f"sh_{suffix}", "mean", f"ort_sh_{suffix}", f"Ortalama {_lower_first(label)} standart hata",
                    decimals=3) for suffix, label, _ in COVARIANCES_1),
        *(Statistic("tekrarlar", f"kapsar_{suffix}", "mean", f"kap_{suffix}", f"{label} aralıkların kapsama oranı",
                    decimals=4) for suffix, label, _ in COVARIANCES_1),
        ScalarTable((
            ("Ortalama eğim tahmini", E.ref("ort_b1")),
            ("Eğim tahminlerinin ampirik standart sapması", E.ref("sd_b1")),
            *((f"Ortalama {label if cov else 'geleneksel'} standart hata", E.ref(f"ort_sh_{suffix}"))
              for suffix, label, cov in COVARIANCES_1),
            *((f"{label} yüzde 95 güven aralığı kapsaması", E.mul(100, E.ref(f"kap_{suffix}")))
              for suffix, label, _ in COVARIANCES_1),
        ), "tablo126", decimals=3, heading="Ölçü", value="Değer", title=_table_title_1(parameters),
            row_decimals=tuple((f"{label} yüzde 95 güven aralığı kapsaması", 2) for _, label, _ in COVARIANCES_1),
            percent_rows=tuple(f"{label} yüzde 95 güven aralığı kapsaması" for _, label, _ in COVARIANCES_1)),
        ScalarTable(tuple((label, E.mul(100, E.ref(f"kap_{suffix}"))) for suffix, label, _ in COVARIANCES_1),
                    "kapsama", decimals=2, heading="Standart hata", value="Kapsama (%)"),
        Histogram("tekrarlar", (("b1", "Eğim tahminleri β̂₁"),), 45, low, high,
                  (f"Şekil 12.7: heteroskedastisite altında EKK eğim tahminlerinin dağılımı{note}" if c > 0 else
                   f"Seçilen süreç (c = 0, homoskedastik): EKK eğim tahminlerinin dağılımı (n = {n})"),
                  "Eğim tahmini β̂₁",
                  references=((BETA_1, "Gerçek eğim β₁ = 2"), ("ort_b1", "Tahminlerin ortalaması"))),
        BarChart("kapsama", "deger", "Standart hata türü", "Gerçek eğimi kapsayan aralıkların oranı (%)",
                 (f"Şekil 12.8: yüzde 95 güven aralıklarının kapsama oranları{note}" if c > 0 else
                  f"Seçilen süreç (c = 0, homoskedastik): yüzde 95 güven aralıklarının kapsama oranları (n = {n})"),
                 percent=True, decimals=2, references=((float(LEVEL), "Nominal düzey"),)),
    )


def _table_title_1(parameters: Parameters) -> str:
    _, c = _hetero_settings(parameters)
    return ("Tablo 12.6: heteroskedastik benzetimde standart hata ve kapsama sonuçları" if c > 0 else
            "Homoskedastik süreçte (c = 0) standart hata ve kapsama sonuçları")


def _hetero_dgp(parameters: Parameters) -> tuple[str, ...]:
    n, c = _hetero_settings(parameters)
    return (
        rf"Y_i = 1 + 2X_i + u_i, \qquad u_i = (0{{,}}3 + c\,X_i^2)\,Z_i, \qquad c = {_tex(c)}",
        rf"X_i \sim U(0,\ 4), \qquad Z_i \sim N(0,\ 1), \qquad n = {n}",
        r"\operatorname{Var}(u_i \mid X_i) = (0{,}3 + c\,X_i^2)^2; \qquad \text{Şekil 12.1: } u_i = \bar\sigma Z_i, \quad "
        rf"\bar\sigma = \sqrt{{\mathbb{{E}}[(0{{,}}3 + cX^2)^2]}} = {_tex(average_sd(c))}",
        rf"\text{{{_thousands(REPS_1)} örneklem; her birinde }} \widehat\beta_1 \text{{ ile geleneksel, HC1 ve HC3 "
        r"standart hata}",
        r"\text{ve yüzde 95 güven aralığı (kritik değer }t_{0{,}025;\,n-2}\text{)}",
    )


def _hetero_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    s = state.scalars
    return (
        SimMetric("Tahminlerin std. sapması", plain(s["sd_b1"], 3),
                  f"{_thousands(REPS_1)} eğim tahmininin ampirik standart sapması (tablonun ikinci satırı)."),
        SimMetric("Ortalama geleneksel SH", plain(s["ort_sh_gel"], 3),
                  "Homoskedastisite varsayan formülün her örneklemde verdiği standart hataların ortalaması."),
        SimMetric("Geleneksel kapsama", _percent(100 * s["kap_gel"]),
                  "Gerçek eğimi kapsayan geleneksel yüzde 95 güven aralıklarının payı."),
        SimMetric("HC1 kapsama", _percent(100 * s["kap_hc1"]),
                  "Gerçek eğimi kapsayan HC1 yüzde 95 güven aralıklarının payı; HC3 tabloda."),
    )


def _hetero_takeaway(state: LabState, parameters: Parameters) -> str:
    n, c = _hetero_settings(parameters)
    s = state.scalars
    covered = (f"Geleneksel yüzde 95 aralıkların {_percent(100 * s['kap_gel'])} kadarı, HC1 aralıklarının "
               f"{_percent(100 * s['kap_hc1'])} kadarı ve HC3 aralıklarının {_percent(100 * s['kap_hc3'])} kadarı gerçek "
               "eğimi kapsar. ")
    if c == 0:
        return ("c = 0: hata varyansı her X'te aynıdır (σ = 0,3); homoskedastisite geçerlidir. Tahminlerin ampirik "
                f"standart sapması {plain(s['sd_b1'], 3)}, geleneksel standart hataların ortalaması "
                f"{plain(s['ort_sh_gel'], 3)}: geleneksel formül doğru ölçektedir. " + covered
                + "Homoskedastisite altında dayanıklı standart hata da geçerlidir; küçük örneklemde kapsaması nominal "
                "düzeyin biraz altında kalabilir. c'yi büyütün: geleneksel aralıklar gerçek belirsizliğe göre dar kalır "
                "ve kapsama düşer.")
    direction = "küçümser" if s["ort_sh_gel"] < s["sd_b1"] else "abartır"
    text = (f"Eğim tahminlerinin ortalaması {plain(s['ort_b1'], 3)}; gerçek eğim 2. Sıfır koşullu ortalama "
            "geçerliyken heteroskedastisite EKK eğimini yanlı yapmaz (§12.3). Tahminlerin ampirik standart sapması "
            f"{plain(s['sd_b1'], 3)}, geleneksel standart hataların ortalaması {plain(s['ort_sh_gel'], 3)}: geleneksel "
            f"formül belirsizliği {direction}. HC1 ve HC3 standart hatalarının ortalaması {plain(s['ort_sh_hc1'], 3)} ve "
            f"{plain(s['ort_sh_hc3'], 3)}. " + covered)
    if n < 200:
        text += ("Dayanıklı standart hata büyük örneklem mantığına dayanır: n'yi büyütün, HC1 ve HC3 kapsaması yüzde "
                 "95'e yaklaşır; geleneksel aralıkların kapsaması yüzde 95'e yaklaşmaz (§12.9).")
    else:
        text += ("Büyük örneklemde HC1 ve HC3 birbirine ve yüzde 95'e yaklaşır; geleneksel aralıkların kapsaması yüzde "
                 "95'e yaklaşmaz, çünkü geleneksel formül bu süreçte yanlış varyansı hedefler (§12.9).")
    return text


HETERO = SimExperiment(
    topic_key=TOPIC,
    number=1,
    title="Katsayı doğru merkezde, aralık yanlış genişlikte",
    question="Hata varyansı X ile büyüdüğünde EKK eğimi yine gerçek değerin çevresinde mi dağılır? Geleneksel standart "
             "hata bu dağılımın genişliğini doğru ölçer mi? Geleneksel, HC1 ve HC3 yüzde 95 güven aralıklarının kaçı "
             "gerçek eğimi kapsar?",
    note=NoteRef("12.9", objects=("§12.1", "Şekil 12.1", "Şekil 12.2", "Tablo 12.6", "Şekil 12.7", "Şekil 12.8")),
    parameters=(
        SimParameter("n", "Örneklem büyüklüğü n", 20, 400, N_NOTES_1, 10, "Her örneklemdeki gözlem sayısı; notlarda 60.",
                     integer=True, decimals=0),
        SimParameter("c", "Heteroskedastisitenin gücü c", 0.0, 1.5, C_NOTES_1, 0.1,
                     "Hatanın standart sapması 0,3 + cX²; c = 0 homoskedastisitedir. Notlarda 0,7.", decimals=1),
    ),
    dgp=_hetero_dgp,
    dgp_note=(
        "Varsayılan ayarlar notlardaki benzetimdir (§12.9: Tablo 12.6, Şekil 12.7–12.8) ve §12.1'deki iki örnek "
        "şeklin örneklemidir (Şekil 12.1–12.2; tohum 305). Her tekrarda önce X, sonra Z çekilir; şekillerdeki örneklem "
        "benzetimin ilk tekrarıdır. Şekil 12.1 aynı X ve Z çekilişleriyle sabit standart sapmalı hatayı kullanır: "
        "hatanın koşulsuz varyansı iki süreçte aynıdır, değişen yalnız koşullu varyanstır. Uygulama tekrarları "
        "vektörel hesaplar; üretilen kod döngüyle aynı sayıları verir."
    ),
    look_at=(
        "**Şekil 12.1–12.2** — aynı çekilişler, aynı koşulsuz hata varyansı: yayılım X boyunca sabit mi, X ile mi "
        "büyüyor?",
        "**Histogram** — eğim tahminleri gerçek değer 2'nin çevresinde mi?",
        "**Kapsama** — hangi aralıklar nominal yüzde 95'e yakın? c = 0 ve büyük n neyi değiştiriyor?",
        "**Tablo 12.6** — tahminlerin ampirik standart sapması ile üç standart hatanın ortalaması yakın mı?",
    ),
    build=_build_hetero,
    metrics=_hetero_metrics,
    takeaway=_hetero_takeaway,
    tables=(("tablo126", _table_title_1, ("n", "c")),),
    labels=(("x", "X"), ("z", "Z"), ("y", "Y"), ("y_homo", "Y (homoskedastik)"), ("b1", "Eğim tahmini β̂₁"),
            ("sh_gel", "Geleneksel SH"), ("sh_hc1", "HC1 SH"), ("sh_hc3", "HC3 SH"),
            ("kapsar_gel", "Geleneksel GA kapsar"), ("kapsar_hc1", "HC1 GA kapsar"), ("kapsar_hc3", "HC3 GA kapsar")),
)


# --- Deney 2: Breusch–Pagan ve White testleri --------------------------------------------------------------------

REPS_2 = 2000
ALPHA_2 = 0.05
"""Y = 1 + 2X + u, u = [1 + γ(X − m)²]·Z, X ~ U(0, 4); her örneklemde BP (û² ~ X) ve White (û² ~ X + X²) testleri."""


def _tests_settings(parameters: Parameters) -> tuple[int, float, float]:
    return int(parameters["n"]), _rounded(parameters, "gamma", 2), _rounded(parameters, "m", 1)


def _test_outcome(gamma: float, center: float) -> E.Expr:
    """Y = 1 + 2X + [1 + γ(X − m)²]·Z; γ = 0 ya da m = 0'da sıfır terimler yazılmaz (sayılar aynı)."""

    mean = E.add(1, E.mul(BETA_1, E.var("x")))
    if gamma == 0:
        return E.add(mean, E.var("z"))
    deviation = E.var("x") if center == 0 else E.sub(E.var("x"), center)
    spread = E.add(1, E.mul(gamma, E.power(deviation, 2)))
    return E.add(mean, E.mul(spread, E.var("z")))


def _test_outcome_text(gamma: float, center: float) -> str:
    return "Y = 1 + 2X + Z" if gamma == 0 else f"Y = 1 + 2X + [1 + {_short(gamma)}(X − {_short(center, 1)})²]·Z"


def _build_tests(parameters: Parameters) -> tuple:
    n, gamma, center = _tests_settings(parameters)
    body = (
        NewSample("tekrar_orneklem", n, None),
        *_draws("tekrar_orneklem"),
        Derive("tekrar_orneklem", "y", _test_outcome(gamma, center), _test_outcome_text(gamma, center)),
        OLS("model", "tekrar_orneklem", "y", ("x",), "EKK: y ~ x"),
        Residuals("tekrar_orneklem", "artik", "model", "Artıklar û"),
        Derive("tekrar_orneklem", "artik2", E.power(E.var("artik"), 2), "Kareli artık û²"),
        Derive("tekrar_orneklem", "x2", E.power(E.var("x"), 2), "X²"),
        OLS("yardimci_bp", "tekrar_orneklem", "artik2", ("x",), "Breusch–Pagan yardımcı regresyonu: û² ~ X"),
        OLS("yardimci_white", "tekrar_orneklem", "artik2", ("x", "x2"), "White yardımcı regresyonu: û² ~ X + X²"),
        ModelValue("r2_bp", "yardimci_bp", "r2", "BP yardımcı regresyonunun R²'si"),
        ModelValue("r2_white", "yardimci_white", "r2", "White yardımcı regresyonunun R²'si"),
        Scalar("lm_bp", E.mul(n, E.ref("r2_bp")), "BP: LM = n·R²"),
        Scalar("lm_white", E.mul(n, E.ref("r2_white")), "White: LM = n·R²"),
        Scalar("p_bp", E.chi2sf(E.ref("lm_bp"), 1), "BP p-değeri P(χ²₁ > LM)", p_value=True),
        Scalar("p_white", E.chi2sf(E.ref("lm_white"), 2), "White p-değeri P(χ²₂ > LM)", p_value=True),
    )
    collect = (("lm_bp", E.ref("lm_bp")), ("p_bp", E.ref("p_bp")), ("reddet_bp", E.compare("lt", E.ref("p_bp"), ALPHA_2)),
               ("lm_white", E.ref("lm_white")), ("p_white", E.ref("p_white")),
               ("reddet_white", E.compare("lt", E.ref("p_white"), ALPHA_2)))
    rows = (("Breusch–Pagan (q = 1)", "bp"), ("White (q = 2)", "white"))
    return (
        NewSample("orneklem", n, SEED),
        *_draws("orneklem"),
        Derive("orneklem", "y", _test_outcome(gamma, center), _test_outcome_text(gamma, center)),
        OLS("ilk_model", "orneklem", "y", ("x",), "İlk örneklemde EKK: y ~ x"),
        Residuals("orneklem", "artik", "ilk_model", "İlk örneklemin artıkları û"),
        HeteroskedasticityTest("lm_bp_ilk", "p_bp_ilk", "ilk_model", "bp", "İlk örneklem: Breusch–Pagan testi"),
        HeteroskedasticityTest("lm_white_ilk", "p_white_ilk", "ilk_model", "white", "İlk örneklem: White testi"),
        ScatterPlot("orneklem", "x", "artik", "X", "Artık û",
                    f"İlk örneklemde artıklar ve X (γ = {_short(gamma)}, m = {_short(center, 1)}, n = {n})",
                    size=9, opacity=0.8, lines=((0.0, 0.0, "Sıfır çizgisi"),)),
        MonteCarlo("tekrarlar", REPS_2, SEED, body, collect,
                   f"{_thousands(REPS_2)} örneklem, her birinde {n} gözlem: Breusch–Pagan ve White testleri"),
        *(Statistic("tekrarlar", f"reddet_{key}", "mean", f"ret_{key}", f"{label}: H₀ ret oranı", decimals=4)
          for label, key in rows),
        *(Statistic("tekrarlar", f"p_{key}", "median", f"medyan_p_{key}", f"{label}: medyan p-değeri", decimals=3)
          for label, key in rows),
        ScalarTable(tuple((label, E.mul(100, E.ref(f"ret_{key}"))) for label, key in rows), "ret_oranlari", decimals=1,
                    heading="Test", value="H₀ ret oranı (%)"),
        ScalarTable(tuple((label, E.ref(f"medyan_p_{key}")) for label, key in rows), "medyan_p", decimals=3,
                    heading="Test", value="Medyan p-değeri"),
        JoinColumns("testlerin_gucu", (("H₀ ret oranı (%)", "ret_oranlari", "deger"),
                                       ("Medyan p-değeri", "medyan_p", "deger")),
                    decimals=3, heading="Test", column_decimals=(("H₀ ret oranı (%)", 1),),
                    p_columns=("Medyan p-değeri",)),
        BarChart("ret_oranlari", "deger", "Test", "H₀ ret oranı (%)",
                 f"{_thousands(REPS_2)} örneklemde homoskedastisite hipotezinin ret oranı (γ = {_short(gamma)}, "
                 f"m = {_short(center, 1)}, n = {n})", percent=True, decimals=1,
                 references=((100 * ALPHA_2, "Anlamlılık düzeyi α"),)),
    )


def _tests_dgp(parameters: Parameters) -> tuple[str, ...]:
    n, gamma, center = _tests_settings(parameters)
    return (
        r"Y_i = 1 + 2X_i + u_i, \qquad u_i = \left[1 + \gamma\,(X_i - m)^2\right] Z_i, \qquad X_i \sim U(0,\ 4), \qquad "
        r"Z_i \sim N(0,\ 1)",
        rf"\gamma = {_tex(gamma)}, \qquad m = {_tex(center, 1)}, \qquad n = {n}; \qquad H_0: "
        r"\operatorname{Var}(u_i \mid X_i) = \sigma^2 \text{ (} \gamma = 0 \text{ iken doğru)}",
        r"\text{BP: } \hat u_i^2 = \delta_0 + \delta_1 X_i + v_i; \qquad \text{White: } \hat u_i^2 = \delta_0 + "
        r"\delta_1 X_i + \delta_2 X_i^2 + v_i",
        rf"LM = nR^2_{{aux}}; \qquad \text{{{_thousands(REPS_2)} örneklem}}",
    )


def _tests_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    s = state.scalars
    return (
        SimMetric("BP ret oranı", _percent(100 * s["ret_bp"], 1),
                  "Breusch–Pagan testinin homoskedastisiteyi yüzde 5 düzeyinde reddettiği örneklemlerin payı."),
        SimMetric("White ret oranı", _percent(100 * s["ret_white"], 1),
                  "White testinin homoskedastisiteyi yüzde 5 düzeyinde reddettiği örneklemlerin payı."),
        SimMetric("İlk örneklemde BP p", _p_value(s["p_bp_ilk"]),
                  "Grafikteki örneklemde statsmodels `het_breuschpagan` p-değeri."),
        SimMetric("İlk örneklemde White p", _p_value(s["p_white_ilk"]),
                  "Grafikteki örneklemde White testinin p-değeri (yardımcı regresyon: û² ~ X + X²)."),
    )


def _tests_takeaway(state: LabState, parameters: Parameters) -> str:
    _, gamma, center = _tests_settings(parameters)
    s = state.scalars
    bp, white = _percent(100 * s["ret_bp"], 1), _percent(100 * s["ret_white"], 1)
    end = ("Reddedememek homoskedastisitenin kanıtlandığı anlamına gelmez; test sonucu dayanıklı standart hata "
           "kullanmanın ön koşulu da değildir (§12.5).")
    if gamma == 0:
        return (f"γ = 0: hata varyansı sabittir, H₀ doğrudur. BP testi örneklemlerin {bp} kadarında, White testi {white} "
                "kadarında H₀'ı reddeder; bu reddetmeler I. tür hatadır ve oranlar α = %5 çevresindedir. γ'yı büyütün: "
                "ret oranı artık testin gücüdür. " + end)
    text = (f"Hata varyansı X ile değişir (γ = {_short(gamma)}): H₀ yanlıştır. BP testi örneklemlerin {bp} kadarında, "
            f"White testi {white} kadarında homoskedastisiteyi reddeder. ")
    if center in (0.0, 4.0):
        text += ("Varyans X aralığında tek yönde değişir (m aralığın ucunda): kareli artıklarda doğrusal bir eğilim "
                 "oluşur; γ ve n yeterince büyükse iki test de bu örüntüyü yakalar. White daha çok serbestlik derecesi "
                 "kullandığı için küçük örneklemde biraz daha az güçlü olabilir. m'yi X aralığının ortasına (2) taşıyın. ")
    elif center == 2.0:
        text += ("Varyans X aralığının ortasında (m = 2) en küçük, iki ucunda büyük: simetrik, U biçimli bir örüntü. "
                 "BP'nin yardımcı regresyonu yalnız X'i içerir; bu örüntüde kareli artıkların X ile doğrusal eğilimi "
                 "zayıftır" + (" ve BP çoğu örneklemde reddedemez. " if s["ret_bp"] < 0.5 else ". ")
                 + ("White X²'yi de içerdiği için bu örüntüyü yakalar. " if s["ret_white"] >= 0.5 else
                    "White X²'yi de içerdiği için bu örüntüyü yakalayabilir; ama bu ayarda γ ya da n küçük olduğu "
                    "için White da çoğu örneklemde reddedemez. γ'yı ya da n'yi büyütün. "))
    else:
        text += (f"Varyans m = {_short(center, 1)} noktasında en küçük; X aralığının bir ucunda diğerinden daha büyük olduğu "
                 "için kareli artıklarda doğrusal bir eğilim de oluşur ve BP de yakalayabilir. m'yi 2'ye taşıyın: "
                 "örüntü simetrikleşir ve BP'nin gücü düşer. ")
    return text + end


TESTS = SimExperiment(
    topic_key=TOPIC,
    number=2,
    title="Breusch–Pagan ve White testleri neyi yakalar?",
    question="Hata varyansı X ile değişirken Breusch–Pagan ve White testleri homoskedastisite hipotezini ne sıklıkla "
             "reddeder? Varyansın X'e bağlılığının biçimi testlerin gücünü nasıl etkiler?",
    note=NoteRef("12.5"),
    parameters=(
        SimParameter("gamma", "Heteroskedastisitenin gücü γ", 0.0, 1.0, 0.5, 0.05, "γ = 0: homoskedastisite (H₀ doğru).",
                     decimals=2),
        SimParameter("m", "Varyansın en küçük olduğu X değeri m", 0.0, 4.0, 2.0, 0.5,
                     "m = 2: U biçimli varyans (X aralığının ortası); m = 0 ya da 4: X ile tek yönde değişen varyans.",
                     decimals=1),
        SimParameter("n", "Örneklem büyüklüğü n", 20, 400, 100, 20, "Her örneklemdeki gözlem sayısı.", integer=True,
                     decimals=0),
    ),
    dgp=_tests_dgp,
    dgp_note=(
        "Notlarda bu deneyin sayısal karşılığı yoktur; §12.5'teki iki testin gücünü tekrarlı örneklemede gösterir. Her "
        "tekrarda önce X, sonra Z çekilir (tohum 305); grafikteki örneklem benzetimin ilk tekrarıdır. Tekrarlarda "
        "yardımcı regresyon açıkça kurulur (LM = n·R²; hazır işlevlerle aynı sayı); ilk örneklemin p-değerleri "
        "Breusch–Pagan için statsmodels `het_breuschpagan`, White için üretilen koddaki `white_testi` işleviyle "
        "hesaplanır."
    ),
    look_at=(
        "**Artık grafiği** — artıkların yayılımı X ile nasıl değişiyor: tek yönde mi, U biçiminde mi?",
        "**Ret oranları** — γ = 0 iken α'ya yakın mı? γ > 0 iken hangi test daha sık reddediyor?",
        "**m** — varyansın en küçük olduğu noktayı X aralığının ortasına ve ucuna taşıyın.",
    ),
    build=_build_tests,
    metrics=_tests_metrics,
    takeaway=_tests_takeaway,
    tables=(("testlerin_gucu", "Tekrarlı örneklemede iki testin kararları"),),
    labels=(("x", "X"), ("z", "Z"), ("y", "Y"), ("artik", "Artık û"), ("artik2", "Kareli artık û²"), ("x2", "X²"),
            ("lm_bp", "BP LM"), ("p_bp", "BP p-değeri"), ("reddet_bp", "BP H₀'ı reddetti"),
            ("lm_white", "White LM"), ("p_white", "White p-değeri"), ("reddet_white", "White H₀'ı reddetti")),
)


# --- Deney 3: küçük örneklemde HC0–HC3 -----------------------------------------------------------------------------

REPS_3 = 4000
COVARIANCES_3 = (("gel", "Geleneksel", None), ("hc0", "HC0", "HC0"), ("hc1", "HC1", "HC1"), ("hc2", "HC2", "HC2"),
                 ("hc3", "HC3", "HC3"))
"""Y = 1 + 2X + X^δ·Z, X = exp(s·W), W, Z ~ N(0, 1): s büyüdükçe X sağa çarpık, birkaç gözlem yüksek kaldıraçlı."""


def _small_settings(parameters: Parameters) -> tuple[int, float, float]:
    return int(parameters["n"]), _rounded(parameters, "s", 2), _rounded(parameters, "delta", 2)


def _small_draws(frame: str, s: float, delta: float) -> tuple:
    if delta == 0:
        error, text = E.var("z"), "Z"
    elif delta == 1:
        error, text = E.mul(E.var("x"), E.var("z")), "X·Z"
    else:
        error, text = E.mul(E.power(E.var("x"), delta), E.var("z")), f"X^{_short(delta)}·Z"
    return (
        Draw(frame, "w", "normal", 0, 1, "W ~ N(0, 1)"),
        Draw(frame, "z", "normal", 0, 1, "Z ~ N(0, 1)"),
        Derive(frame, "x", E.exp(E.mul(s, E.var("w"))), f"X = exp({_short(s)}·W): sağa çarpık açıklayıcı"),
        Derive(frame, "y", E.add(E.add(1, E.mul(BETA_1, E.var("x"))), error), f"Y = 1 + 2X + u, u = {text}"),
    )


def _build_small(parameters: Parameters) -> tuple:
    n, s, delta = _small_settings(parameters)
    inference, collect = _inference(COVARIANCES_3, BETA_1)
    body = [NewSample("tekrar_orneklem", n, None), *_small_draws("tekrar_orneklem", s, delta),
            OLS("model", "tekrar_orneklem", "y", ("x",), "EKK: y ~ x"), *inference]
    return (
        NewSample("orneklem", n, SEED),
        *_small_draws("orneklem", s, delta),
        ScatterPlot("orneklem", "x", "y", "X", "Y", f"İlk örneklem: X sağa çarpık (s = {_short(s)}, δ = {_short(delta)}, "
                    f"n = {n})", fit_line=True, size=9, opacity=0.8,
                    lines=((1.0, BETA_1, "Gerçek koşullu ortalama E(Y | X) = 1 + 2X"),)),
        MonteCarlo("tekrarlar", REPS_3, SEED, tuple(body), tuple(collect),
                   f"{_thousands(REPS_3)} örneklem, her birinde {n} gözlem: geleneksel ve HC0–HC3 standart hata ve yüzde "
                   "95 güven aralıkları"),
        Statistic("tekrarlar", "b1", "std", "sd_b1", "Eğim tahminlerinin ampirik standart sapması", decimals=3),
        *(Statistic("tekrarlar", f"sh_{suffix}", "mean", f"ort_sh_{suffix}", f"Ortalama {_lower_first(label)} standart hata",
                    decimals=3) for suffix, label, _ in COVARIANCES_3),
        *(Statistic("tekrarlar", f"kapsar_{suffix}", "mean", f"kap_{suffix}", f"{label} aralıkların kapsama oranı",
                    decimals=4) for suffix, label, _ in COVARIANCES_3),
        ScalarTable(tuple((label, E.mul(100, E.ref(f"kap_{suffix}"))) for suffix, label, _ in COVARIANCES_3),
                    "kapsama3", decimals=1, heading="Standart hata", value="Kapsama (%)"),
        ScalarTable(tuple((label, E.ref(f"ort_sh_{suffix}")) for suffix, label, _ in COVARIANCES_3), "ort_sh3",
                    decimals=3, heading="Standart hata", value="Ortalama SH"),
        JoinColumns("hc_ozet", (("Ortalama SH", "ort_sh3", "deger"), ("Kapsama (%)", "kapsama3", "deger")), decimals=3,
                    heading="Standart hata", column_decimals=(("Kapsama (%)", 1),)),
        BarChart("kapsama3", "deger", "Standart hata türü", "Gerçek eğimi kapsayan aralıkların oranı (%)",
                 f"Yüzde 95 güven aralıklarının kapsama oranları (s = {_short(s)}, δ = {_short(delta)}, n = {n})",
                 percent=True, decimals=1, references=((float(LEVEL), "Nominal düzey"),)),
    )


def _small_dgp(parameters: Parameters) -> tuple[str, ...]:
    n, s, delta = _small_settings(parameters)
    return (
        r"Y_i = 1 + 2X_i + u_i, \qquad u_i = X_i^{\delta} Z_i, \qquad X_i = e^{s W_i}",
        rf"W_i,\ Z_i \sim N(0,\ 1), \qquad s = {_tex(s)}, \quad \delta = {_tex(delta)}, \quad n = {n}",
        r"\widehat{\operatorname{Var}}(\widehat\beta_1) = \frac{\sum_i (X_i - \bar X)^2\,\omega_i}"
        r"{\left[\sum_i (X_i - \bar X)^2\right]^2}, \qquad h_i = \frac{1}{n} + "
        r"\frac{(X_i - \bar X)^2}{\sum_j (X_j - \bar X)^2} \text{ (kaldıraç)}",
        r"\omega_i: \quad \text{HC0: } \hat u_i^2, \quad \text{HC1: } \tfrac{n}{n-2}\,\hat u_i^2, \quad "
        r"\text{HC2: } \tfrac{\hat u_i^2}{1-h_i}, \quad \text{HC3: } \tfrac{\hat u_i^2}{(1-h_i)^2}",
        rf"\text{{{_thousands(REPS_3)} örneklem; her birinde geleneksel ve HC0–HC3}}",
        r"\text{yüzde 95 güven aralığı (kritik değer }t_{0{,}025;\,n-2}\text{)}",
    )


def _small_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    s = state.scalars
    return (
        SimMetric("Tahminlerin std. sapması", plain(s["sd_b1"], 3),
                  f"{_thousands(REPS_3)} eğim tahmininin ampirik standart sapması: standart hataların hedefi."),
        SimMetric("Geleneksel kapsama", _percent(100 * s["kap_gel"], 1), "Homoskedastisite varsayan aralıkların kapsaması."),
        SimMetric("HC1 kapsama", _percent(100 * s["kap_hc1"], 1),
                  "Kareli artıklar n/(n − 2) ile çarpılır; HC0 tabloda."),
        SimMetric("HC3 kapsama", _percent(100 * s["kap_hc3"], 1),
                  "Kareli artıklar (1 − h)²'ye bölünerek büyütülür; HC2 tabloda."),
    )


def _small_takeaway(state: LabState, parameters: Parameters) -> str:
    n, s_value, delta = _small_settings(parameters)
    s = state.scalars
    rates = {suffix: _percent(100 * s[f"kap_{suffix}"], 1) for suffix, _, _ in COVARIANCES_3}
    text = (f"Kapsama oranları: geleneksel {rates['gel']}, HC0 {rates['hc0']}, HC1 {rates['hc1']}, HC2 {rates['hc2']}, "
            f"HC3 {rates['hc3']}. ")
    if delta == 0:
        text += ("δ = 0: hata varyansı sabittir; geleneksel standart hata doğru formüldür. Dayanıklı hesaplar da büyük "
                 "örneklemde geçerlidir; küçük örneklemde HC0 ve HC1 kapsamayı düşürebilir, HC3 daha koruyucudur. ")
    elif s_value <= 0.25:
        text += ("X hafif çarpık: kaldıraç değerleri birbirine yakındır ve dayanıklı hesaplar arasındaki fark küçülür. "
                 "Yine de küçük örneklemde kareli artıklar hata varyansını küçümser; HC0 ve HC1 kapsamayı düşürür, HC3 "
                 "en koruyucudur. ")
    else:
        text += ("Heteroskedastisite X ile büyüyor ve sağa çarpık X birkaç yüksek kaldıraçlı gözlem üretiyor. Bu "
                 "gözlemlerin artıkları küçük kalır (doğru onlara doğru çekilir); HC0 ve HC1 bu yüzden belirsizliği "
                 "küçümser. HC2 ve HC3 kaldıraçlı gözlemlerin artıklarını büyütür; HC3 en koruyucudur. ")
    if delta > 0:
        robust = min(s[f"kap_{suffix}"] for suffix, _, _ in COVARIANCES_3 if suffix != "gel")
        text += ("Geleneksel aralıklar heteroskedastisiteyi hesaba katmadığı için en düşük kapsamayı verir; n büyüse "
                 "de düzelmez. " if s["kap_gel"] < robust else
                 "Bu ayarda küçük örneklem yanlılığı baskın: geleneksel aralıklar bazı dayanıklı aralıklardan daha iyi "
                 "kapsıyor; n'yi büyütün. ")
    if n <= 40:
        text += ("Örneklem küçükken hiçbir düzeltme yüzde 95'i garanti etmez (§12.6). n'yi büyütün: HC0–HC3 birbirine "
                 "yaklaşır.")
    else:
        text += ("n büyüdükçe HC0–HC3 birbirine yaklaşır; X ne kadar çarpıksa yakınsama o kadar yavaştır. Makale tablosunda "
                 "hangi kovaryansın kullanıldığı yazılmalıdır; küçük örneklemde HC3 duyarlılık kontrolüdür (§12.6).")
    if s_value >= 1 and n <= 40:
        text += " s'yi küçültün: kaldıraç azaldıkça dört hesap birbirine yaklaşır."
    return text


SMALL = SimExperiment(
    topic_key=TOPIC,
    number=3,
    title="Küçük örneklemde HC0, HC1, HC2 ve HC3",
    question="Örneklem küçükken ve açıklayıcı değişkende birkaç uç (yüksek kaldıraçlı) gözlem varken dayanıklı güven "
             "aralıkları gerçek eğimi ne sıklıkla kapsar? HC0, HC1, HC2 ve HC3 arasındaki fark nereden gelir?",
    note=NoteRef("12.6", objects=("Tablo 12.3", "Denklem 12.6 (§12.3)")),
    parameters=(
        SimParameter("n", "Örneklem büyüklüğü n", 10, 200, 20, 10, "Her örneklemdeki gözlem sayısı.", integer=True,
                     decimals=0),
        SimParameter("s", "X'in çarpıklığı s", 0.25, 1.25, 0.5, 0.25,
                     "X = exp(s·W): s büyüdükçe birkaç gözlem çok büyük X değeri alır (yüksek kaldıraç).", decimals=2),
        SimParameter("delta", "Heteroskedastisite üssü δ", 0.0, 1.5, 1.0, 0.25,
                     "Hatanın standart sapması X^δ; δ = 0 homoskedastisitedir.", decimals=2),
    ),
    dgp=_small_dgp,
    dgp_note=(
        "Notlarda bu deneyin sayısal karşılığı yoktur; Tablo 12.3'teki dört dayanıklı kovaryans seçeneğini küçük "
        "örneklemde karşılaştırır. Her tekrarda önce W, sonra Z çekilir (tohum 305); grafikteki örneklem benzetimin ilk "
        "tekrarıdır. Kaldıraç hᵢ, gözlemin X değerinin ortalamadan uzaklığıyla büyür (1/n ile 1 arasında). Uygulama "
        "tekrarları vektörel hesaplar; üretilen kod döngüyle aynı sayıları verir."
    ),
    look_at=(
        "**Grafik** — birkaç gözlem X ekseninde diğerlerinden çok uzakta mı? Tahmin edilen doğru onlara doğru mu "
        "çekiliyor?",
        "**Kapsama** — dört dayanıklı hesap sıralanıyor mu? Hangisi nominal yüzde 95'e en yakın?",
        "**n ve s** — örneklem büyüdükçe ve X'in çarpıklığı azaldıkça farklar nasıl değişiyor?",
    ),
    build=_build_small,
    metrics=_small_metrics,
    takeaway=_small_takeaway,
    tables=(("hc_ozet", "Tekrarlı örneklemede standart hata ve kapsama"),),
    labels=(("w", "W"), ("z", "Z"), ("x", "X"), ("y", "Y"), ("b1", "Eğim tahmini β̂₁"),
            *((f"sh_{suffix}", f"{label} SH") for suffix, label, _ in COVARIANCES_3),
            *((f"kapsar_{suffix}", f"{label} GA kapsar") for suffix, label, _ in COVARIANCES_3)),
)


KONU12_EXPERIMENTS = (HETERO, TESTS, SMALL)
