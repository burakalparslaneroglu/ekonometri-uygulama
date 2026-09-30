"""Konu 11 Sezgi deneyleri: veri üretim süreci bilinen kontrollü simülasyonlar.

Deney 1  Grup farkı hangi X'te? Merkezleme ve kesinlik                (Notlar §11.4 ve §11.5)
Deney 2  Etkileşimi atlamak: additif model hangi farkı tahmin eder?   (Notlar §11.1 ve §11.4)
Deney 3  Tekli t testleri ile ortak F testi                            (Notlar §11.8)

Bölüm 11'de benzetim yoktur; deneylerin notlarda sayısal karşılığı yoktur. Varsayılan ayarlar bölümün kavramlarını
göstermek için seçilmiştir; her deney tohum 305 ile başlar ve grafikteki tek örneklem benzetimin ilk tekrarıdır. Üç deney
aynı veri üretim sürecini kullanır: X ~ U(8, 18) (ör. eğitim yılı), D grup kuklası, gruplar arasındaki fark X = 13'te
(X'in ortası) −0,45 ve X ile γ₁ hızında değişir.
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
    HypothesisPlot,
    JoinColumns,
    JointTest,
    LineChart,
    ModelValue,
    MonteCarlo,
    NewSample,
    NoteRef,
    RegressionTable,
    Scalar,
    ScalarTable,
    Statistic,
    Support,
)

SEED = 305
TOPIC = "konu11"
X_LOW, X_HIGH, X_MID = 8, 18, 13
CLOSE_SD = 1.15
"""Deney 1: merkez c'deki ve 13'teki standart sapmaların oranı bundan küçükse ikisi neredeyse aynı sayılır."""
"""X ~ U(8, 18): ortalaması 13, varyansı 100/12."""
BETA_0, BETA_1 = 1.0, 0.1
"""Referans grubun (D = 0) sabiti ve eğimi."""
GAP_MID = -0.45
"""X = 13'teki gerçek grup farkı (Deney 1 ve 2)."""
SIGMA = 0.4
ALPHA = 0.05
VAR_X = (X_HIGH - X_LOW) ** 2 / 12


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


def _percent(value: float, decimals: int = 1) -> str:
    return "%" + plain(value, decimals)


def _signed(value: float, decimals: int = 2) -> str:
    """Toplamdaki terim: "+ 0,05" ya da "− 0,05" (“+ −0,05” yazılmaz)."""

    return f"{'−' if value < 0 else '+'} {_short(abs(value), decimals)}"


def gap(slope_gap: float, x: float, level: float = GAP_MID) -> float:
    """Gerçek grup farkı g(x) = E(Y | X = x, D = 1) − E(Y | X = x, D = 0) = g₁₃ + γ₁(x − 13)."""

    return level + slope_gap * (x - X_MID)


def _plus(expression: E.Expr, coefficient: float, term: E.Expr) -> E.Expr:
    """ifade + katsayı·terim; negatif katsayı çıkarma olarak yazılır ("+ -0.05" yazılmaz), sıfır katsayılı terim
    yazılmaz."""

    if coefficient == 0:
        return expression
    if coefficient < 0:
        return E.sub(expression, E.mul(-coefficient, term))
    return E.add(expression, E.mul(coefficient, term))


def _centered(c: float) -> E.Expr:
    """X − c; c = 0 iken yalnız X (kodda "x - 0" yazılmaz)."""

    return E.var("x") if c == 0 else E.sub(E.var("x"), c)


def _draws(frame: str, share: float = 0.5) -> tuple:
    """X ~ U(8, 18), V ~ U(0, 1), u ~ N(0, 0,4²) bu sırayla; D = 1{V < p}."""

    return (
        Draw(frame, "x", "uniform", X_LOW, X_HIGH, "X ~ U(8, 18) (ör. eğitim yılı)"),
        Draw(frame, "v", "uniform", 0, 1, "V ~ U(0, 1): grup ataması için"),
        Draw(frame, "u", "normal", 0, SIGMA, "u ~ N(0, 0,4²)"),
        Derive(frame, "d", E.compare("lt", E.var("v"), share), f"D = 1{{V < {_short(share, 1)}}}: grup kuklası"),
    )


def _outcome(frame: str, slope_gap: float, level: float = GAP_MID) -> Derive:
    """Y = 1 + 0,1X + D·[g₁₃ + γ₁(X − 13)] + u."""

    line = E.add(BETA_0, E.mul(BETA_1, E.var("x")))
    if slope_gap != 0:
        group = _plus(E.const(level), slope_gap, E.sub(E.var("x"), X_MID))
        line = E.add(line, E.mul(E.var("d"), group))
    else:
        line = _plus(line, level, E.var("d"))
    expression = E.add(line, E.var("u"))
    if slope_gap != 0:
        text = f"Y = 1 + 0,1X + D·[{_short(level)} {_signed(slope_gap)}(X − 13)] + u"
    else:
        text = "Y = 1 + 0,1X + u" if level == 0 else f"Y = 1 + 0,1X {_signed(level)}·D + u"
    return Derive(frame, "y", expression, text)


# --- Deney 1: grup farkı hangi X'te? --------------------------------------------------------------------------

REPS_1 = 2000


def _centering_settings(parameters: Parameters) -> tuple[float, float, int]:
    return _rounded(parameters, "c", 0), _rounded(parameters, "g1", 2), int(parameters["n"])


def gap_sd(c: float, n: int, share: float = 0.5) -> float:
    """c'deki fark tahmininin yaklaşık standart sapması: iki grubun c'deki doğru yüksekliklerinin varyanslarının toplamı,
    σ²/(n·p)·[1 + (c − 13)²/Var(X)] + σ²/(n(1 − p))·[…]."""

    spread = 1 + (c - X_MID) ** 2 / VAR_X
    return (SIGMA**2 * spread * (1 / (n * share) + 1 / (n * (1 - share)))) ** 0.5


def centering_axis(c: float, slope_gap: float, n: int) -> tuple[float, float]:
    """Histogram ekseni: iki gerçek farkın çevresinde c'deki tahminin yaklaşık standart sapmasının dört katı."""

    half = 4 * gap_sd(c, n)
    values = (gap(slope_gap, c), gap(slope_gap, X_MID))
    return round(min(values) - half, 2), round(max(values) + half, 2)


def _centering_labels(c: float) -> tuple[str, str]:
    return f"Merkez c: X = {_short(c, 0)} noktasında fark", "Merkez 13: verinin ortasında fark"


def _build_centering(parameters: Parameters) -> tuple:
    c, slope_gap, n = _centering_settings(parameters)
    truth_c, truth_mid = gap(slope_gap, c), gap(slope_gap, X_MID)
    label_c, label_mid = _centering_labels(c)
    body = (
        NewSample("tekrar_orneklem", n, None),
        *_draws("tekrar_orneklem"),
        _outcome("tekrar_orneklem", slope_gap),
        Derive("tekrar_orneklem", "xc", _centered(c), f"X − c: c = {_short(c, 0)} etrafında merkezlenmiş X"),
        Derive("tekrar_orneklem", "x13", E.sub(E.var("x"), X_MID), "X − 13: verinin ortası etrafında merkezlenmiş X"),
        OLS("model_c", "tekrar_orneklem", "y", ("d", "xc", "d:xc"), "Etkileşimli model, merkez c: y ~ d * xc"),
        OLS("model_13", "tekrar_orneklem", "y", ("d", "x13", "d:x13"), "Etkileşimli model, merkez 13: y ~ d * x13"),
        *(item for suffix, term in (("c", "d"), ("13", "d")) for item in (
            ModelValue(f"fark_{suffix}", f"model_{suffix}", "coef", f"D katsayısı (merkez {suffix})", term=term),
            ModelValue(f"sh_{suffix}", f"model_{suffix}", "se", f"D katsayısının standart hatası (merkez {suffix})",
                       term=term),
            ModelValue(f"p_{suffix}", f"model_{suffix}", "p", f"H₀: fark = 0 için p-değeri (merkez {suffix})", term=term),
            ModelValue(f"alt_{suffix}", f"model_{suffix}", "ci_low", f"Yüzde 95 GA alt sınırı (merkez {suffix})",
                       term=term),
            ModelValue(f"ust_{suffix}", f"model_{suffix}", "ci_high", f"Yüzde 95 GA üst sınırı (merkez {suffix})",
                       term=term),
        )),
    )
    collect = []
    for suffix, truth in (("c", truth_c), ("13", truth_mid)):
        collect += [(f"fark_{suffix}", E.ref(f"fark_{suffix}")), (f"sh_{suffix}", E.ref(f"sh_{suffix}")),
                    (f"kapsar_{suffix}", E.mul(E.compare("le", E.ref(f"alt_{suffix}"), truth),
                                               E.compare("le", truth, E.ref(f"ust_{suffix}")))),
                    (f"reddet_{suffix}", E.compare("lt", E.ref(f"p_{suffix}"), ALPHA))]
    rows = ((label_c, "c", truth_c), (label_mid, "13", truth_mid))
    summaries = [Statistic("tekrarlar", f"{column}_{suffix}", stat, f"{name}_{suffix}", f"{label}: {words}",
                           decimals=4)
                 for name, column, stat, words in (("ort", "fark", "mean", "ortalama tahmin"),
                                                   ("sd", "fark", "std", "tahminlerin std. sapması"),
                                                   ("ortsh", "sh", "mean", "ortalama standart hata"),
                                                   ("kap", "kapsar", "mean", "kapsama oranı"),
                                                   ("ret", "reddet", "mean", "H₀: fark = 0 ret oranı"))
                 for label, suffix, _ in rows]
    tables = [ScalarTable(tuple((label, E.const(truth)) for label, _, truth in rows), "gercek_fark", decimals=3,
                          heading="Karşılaştırma noktası")]
    for name, scale in (("ort", 1), ("sd", 1), ("ortsh", 1), ("kap", 100), ("ret", 100)):
        tables.append(ScalarTable(tuple((label, E.mul(scale, E.ref(f"{name}_{suffix}")) if scale != 1
                                         else E.ref(f"{name}_{suffix}")) for label, suffix, _ in rows),
                                  f"ozet_{name}", decimals=3, heading="Karşılaştırma noktası"))
    low, high = centering_axis(c, slope_gap, n)
    return (
        NewSample("orneklem", n, SEED),
        *_draws("orneklem"),
        _outcome("orneklem", slope_gap),
        CopyFrame("orneklem_13", "orneklem", "Aynı örneklemin kopyası: X'i 13 etrafında merkezlemek için"),
        Derive("orneklem", "xm", _centered(c), f"Merkezlenmiş X: X − {_short(c, 0)}"),
        Derive("orneklem_13", "xm", E.sub(E.var("x"), X_MID), "Merkezlenmiş X: X − 13"),
        OLS("ilk_c", "orneklem", "y", ("d", "xm", "d:xm"), f"İlk örneklem, merkez c = {_short(c, 0)}: y ~ d * xm"),
        OLS("ilk_13", "orneklem_13", "y", ("d", "xm", "d:xm"), "İlk örneklem, merkez 13: y ~ d * xm"),
        RegressionTable(((f"(1) Merkez c = {_short(c, 0)}", "ilk_c"), ("(2) Merkez 13", "ilk_13")),
                        ("d", "xm", "d:xm", INTERCEPT), "merkezleme",
                        "İlk örneklem: aynı model, iki merkezleme noktası", decimals=3),
        ModelValue("g13_ilk", "ilk_13", "coef", "İlk örneklem: X = 13'te fark tahmini", term="d"),
        ModelValue("g1_ilk", "ilk_13", "coef", "İlk örneklem: eğim farkı tahmini γ̂₁", term="d:xm"),
        ModelValue("gc_ilk", "ilk_c", "coef", f"İlk örneklem: X = {_short(c, 0)} noktasında fark tahmini", term="d"),
        Scalar("c_nokta", E.const(c), "Merkezleme noktası c", decimals=0, shown=False),
        Statistic("orneklem", "x", "min", "x_min", "Örneklemde en küçük X", decimals=2),
        Statistic("orneklem", "x", "max", "x_max", "Örneklemde en büyük X", decimals=2),
        Support("izgara", "k", 0, 40, "Izgara: k = 0, 1, …, 40"),
        Derive("izgara", "x", E.div(E.var("k"), 2), "X = k/2: 0, 0,5, …, 20"),
        Derive("izgara", "gercek_fark", _plus(E.const(GAP_MID), slope_gap, E.sub(E.var("x"), X_MID)),
               "Gerçek fark g(X) = −0,45 + γ₁(X − 13)"),
        Derive("izgara", "tahmin_fark", E.add(E.ref("g13_ilk"), E.mul(E.ref("g1_ilk"), E.sub(E.var("x"), X_MID))),
               "Tahmin edilen fark ĝ(X) = ĝ₁₃ + γ̂₁(X − 13)"),
        LineChart("izgara", "x", "tahmin_fark", "X", "Grup farkı (D = 1 eksi D = 0)",
                  f"Grup farkı X'e göre: ilk örneklemin tahmini ve gerçek fark (n = {n})", markers=False,
                  legend="Tahmin edilen fark (ilk örneklem)", bands=(("gercek_fark", "Gerçek fark"),),
                  vlines=(("c_nokta", "Merkezleme noktası c"), ("x_min", "Örneklemde en küçük X"),
                          ("x_max", "Örneklemde en büyük X"))),
        MonteCarlo("tekrarlar", REPS_1, SEED, body, tuple(collect),
                   f"{_thousands(REPS_1)} örneklem, her birinde {n} gözlem: aynı model, iki merkezleme noktası"),
        *summaries,
        *tables,
        JoinColumns("fark_ozet", (("Gerçek fark", "gercek_fark", "deger"), ("Ortalama tahmin", "ozet_ort", "deger"),
                                  ("Std. sapma", "ozet_sd", "deger"), ("Ortalama SH", "ozet_ortsh", "deger"),
                                  ("Kapsama (%)", "ozet_kap", "deger"), ("H₀: fark = 0 ret (%)", "ozet_ret", "deger")),
                    decimals=3, heading="Karşılaştırma noktası",
                    column_decimals=(("Kapsama (%)", 1), ("H₀: fark = 0 ret (%)", 1))),
        Histogram("tekrarlar", (("fark_c", label_c), ("fark_13", label_mid)), 45, low, high,
                  f"{_thousands(REPS_1)} örneklemde D katsayısı: iki merkezleme noktası (n = {n})",
                  "D katsayısı (merkez noktasındaki grup farkı)",
                  references=((truth_c, f"Gerçek fark, X = {_short(c, 0)}"), (truth_mid, "Gerçek fark, X = 13"))),
    )


def _centering_dgp(parameters: Parameters) -> tuple[str, ...]:
    c, slope_gap, n = _centering_settings(parameters)
    return (
        r"Y_i = 1 + 0{,}1\,X_i + D_i\,g(X_i) + u_i, \qquad g(X) = -0{,}45 + \gamma_1 (X - 13), \qquad "
        rf"\gamma_1 = {_tex(slope_gap)}",
        r"X_i \sim U(8,\ 18), \qquad D_i = \mathbf{1}\{V_i < 0{,}5\},\ V_i \sim U(0,\ 1), \qquad "
        rf"u_i \sim N(0,\ 0{{,}}4^2), \qquad n = {n}",
        rf"\text{{Model: }} Y_i = \alpha_0 + \beta_1 (X_i - c) + \gamma_0 D_i + \gamma_1 D_i (X_i - c) + u_i, \quad "
        rf"c = {_tex(c, 0)}; \quad \gamma_0 = g(c) = {_tex(gap(slope_gap, c), 3)}",
    )


def _centering_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    c, slope_gap, _ = _centering_settings(parameters)
    s = state.scalars
    return (
        SimMetric("Merkez c'de gerçek fark", plain(gap(slope_gap, c), 3),
                  "Merkez c'de iki grubun koşullu ortalamaları arasındaki gerçek fark: D katsayısının hedefi."),
        SimMetric("Std. sapma, merkez c", plain(s["sd_c"], 3),
                  f"{_thousands(REPS_1)} örneklemde D katsayısının (c'deki fark) ampirik standart sapması."),
        SimMetric("Std. sapma, merkez 13", plain(s["sd_13"], 3),
                  "X'in ortasındaki (13) fark tahmininin ampirik standart sapması."),
        SimMetric("c'de fark anlamlı", _percent(100 * s["ret_c"]),
                  "D katsayısının yüzde 5 düzeyinde sıfırdan farklı bulunduğu örneklemlerin payı."),
    )


def _centering_takeaway(state: LabState, parameters: Parameters) -> str:
    c, slope_gap, _ = _centering_settings(parameters)
    s = state.scalars
    truth = gap(slope_gap, c)
    same = ("Merkezleme tahmin edilen değerleri, R²'yi ve iki grubun eğimlerini (tablodaki Merkezlenmiş X ve D × "
            "Merkezlenmiş X satırları) değiştirmez; yalnız sabit terimin ve D katsayısının referans noktasını "
            "değiştirir (§11.5). ")
    if c == X_MID:
        text = ("Merkez c = 13: iki model aynıdır ve D katsayısı verinin ortasındaki farkı tahmin eder; bu noktada "
                f"gerçek fark {plain(truth, 3)}, tahminlerin ampirik standart sapması {plain(s['sd_c'], 3)}. Veri "
                "ortasında fark en kesin biçimde tahmin edilir. ")
        return text + "c'yi verinin dışına taşıyın (ör. 0): aynı model o noktadaki farkı çok daha belirsiz tahmin eder."
    if slope_gap == 0:
        return ("γ₁ = 0: gerçek fark her X'te −0,45'tir ve doğrular paraleldir. D katsayısı her merkezde aynı farkı "
                f"hedefler; yine de tahminin yayılımı c'ye bağlıdır (merkez c: {plain(s['sd_c'], 3)}, merkez 13: "
                f"{plain(s['sd_13'], 3)}), çünkü model eğim farkını da tahmin eder. " + same)
    text = (f"Merkez c = {_short(c, 0)}: D katsayısı X = {_short(c, 0)} noktasındaki farkı tahmin eder; bu noktada "
            f"gerçek fark {plain(truth, 3)}. ")
    zero = abs(truth) < 1e-9
    if zero:
        text += ("Bu noktada iki grup arasında fark yoktur: ‘c'de fark anlamlı’ oranı I. tür hata oranıdır ve yüzde 5 "
                 "çevresinde beklenir. ")
    if c < X_LOW or c > X_HIGH:
        text += ("Bu nokta örneklemdeki X aralığının (8–18) dışındadır: fark, iki grup doğrusunun veri olmayan bir "
                 "bölgeye uzatılmasıyla tahmin edilir. ")
        if (not zero and (truth > 0) != (gap(slope_gap, X_LOW) > 0)
                and (gap(slope_gap, X_LOW) > 0) == (gap(slope_gap, X_HIGH) > 0)):
            text += "Veri aralığında fark her yerde aynı işaretliyken bu noktadaki farkın işareti terstir. "
    ratio = s["sd_c"] / s["sd_13"]
    text += (f"Tahminlerin ampirik standart sapması merkez c'de {plain(s['sd_c'], 3)}, X'in ortasında (13) "
             f"{plain(s['sd_13'], 3)}: {plain(ratio, 1)} kat. "
             + ("Merkez verinin ortasına yakın olduğu için iki standart sapma neredeyse aynıdır; c 13'ten "
                "uzaklaştıkça fark büyür. " if ratio < CLOSE_SD else
                "Merkez noktası verinin ortasından uzaklaştıkça fark tahmini belirsizleşir. ") + same)
    return text + "c'yi 13'e taşıyın: D katsayısı verinin ortasındaki farkı en kesin biçimde verir."


CENTERING = SimExperiment(
    topic_key=TOPIC,
    number=1,
    title="Grup farkı hangi X'te? Merkezleme ve kesinlik",
    question="Etkileşimli modelde kukla katsayısı hangi X değerindeki grup farkını gösterir? Merkezleme noktası verinin "
             "dışına çıktığında bu farkın tahmini ne kadar belirsizleşir?",
    note=NoteRef("11.5", objects=("§11.4",)),
    parameters=(
        SimParameter("c", "Merkezleme noktası c", 0, 20, 0, 1,
                     "D katsayısı X = c'deki farktır; c = 0 merkezlenmemiş modeldir.", integer=True, decimals=0),
        SimParameter("g1", "Eğim farkı γ₁", -0.1, 0.1, -0.05, 0.01,
                     "D = 1 grubunun eğimi 0,1 + γ₁; γ₁ = 0 paralel doğrulardır.", decimals=2),
        SimParameter("n", "Örneklem büyüklüğü n", 50, 500, 300, 50, "Her örneklemdeki gözlem sayısı.", integer=True,
                     decimals=0),
    ),
    dgp=_centering_dgp,
    dgp_note=(
        "Notlarda bu deneyin sayısal karşılığı yoktur; §11.4–11.5'teki koşullu fark ve merkezlemeyi tekrarlı örneklemede "
        "gösterir. Her tekrarda önce X, sonra V ve u çekilir (tohum 305); grafik ve tablodaki örneklem benzetimin ilk "
        "tekrarıdır. Aynı örneklem iki merkezle tahmin edilir: seçilen c ve verinin ortası 13."
    ),
    look_at=(
        "**Çizgi grafiği** — merkez c örneklemdeki X aralığının içinde mi? D katsayısı fark doğrusunun hangi "
        "noktasıdır?",
        "**Histogram** — c'deki ve 13'teki fark tahminleri ne kadar yayılıyor?",
        "**Tablolar** — iki sütunda hangi katsayılar aynı, hangileri farklı? Tekrarlarda hangi noktadaki fark daha "
        "kesin?",
    ),
    build=_build_centering,
    metrics=_centering_metrics,
    takeaway=_centering_takeaway,
    tables=(("merkezleme", "İlk örneklem: aynı model, iki merkezleme noktası"),
            ("fark_ozet", "Tekrarlı örneklemede iki noktadaki fark tahmini")),
    labels=(("x", "X"), ("v", "V"), ("u", "u"), ("d", "D (grup kuklası)"), ("y", "Y"), ("xc", "X − c"),
            ("x13", "X − 13"), ("xm", "Merkezlenmiş X"), (INTERCEPT, "Sabit terim"),
            ("fark_c", "D katsayısı (merkez c)"), ("fark_13", "D katsayısı (merkez 13)"), ("sh_c", "SH (merkez c)"),
            ("sh_13", "SH (merkez 13)"), ("kapsar_c", "GA kapsar (merkez c)"), ("kapsar_13", "GA kapsar (merkez 13)"),
            ("reddet_c", "H₀ reddedildi (merkez c)"), ("reddet_13", "H₀ reddedildi (merkez 13)"),
            ("gercek_fark", "Gerçek fark"), ("tahmin_fark", "Tahmin edilen fark")),
)


# --- Deney 2: etkileşimi atlamak ------------------------------------------------------------------------------

REPS_2 = 2000
POINTS_2 = (X_LOW, X_MID, X_HIGH)


def _omitted_settings(parameters: Parameters) -> tuple[float, float, int]:
    return _rounded(parameters, "g1", 2), _rounded(parameters, "p", 1), int(parameters["n"])


def _build_omitted(parameters: Parameters) -> tuple:
    slope_gap, share, n = _omitted_settings(parameters)
    body = (
        NewSample("tekrar_orneklem", n, None),
        *_draws("tekrar_orneklem", share),
        _outcome("tekrar_orneklem", slope_gap),
        Derive("tekrar_orneklem", "x13", E.sub(E.var("x"), X_MID), "X − 13"),
        OLS("additif", "tekrar_orneklem", "y", ("d", "x"), "Additif model (etkileşimsiz): y ~ d + x"),
        OLS("etkilesimli", "tekrar_orneklem", "y", ("d", "x13", "d:x13"), "Etkileşimli model: y ~ d * x13"),
        ModelValue("add_d", "additif", "coef", "Additif model: D katsayısı", term="d"),
        ModelValue("add_x", "additif", "coef", "Additif model: ortak X eğimi", term="x"),
        ModelValue("int_d", "etkilesimli", "coef", "Etkileşimli model: X = 13'te fark", term="d"),
        ModelValue("int_g1", "etkilesimli", "coef", "Etkileşimli model: eğim farkı γ̂₁", term="d:x13"),
        ModelValue("int_x", "etkilesimli", "coef", "Etkileşimli model: D = 0 grubunun eğimi", term="x13"),
    )
    collect = (("add_d", E.ref("add_d")), ("add_x", E.ref("add_x")), ("int_d", E.ref("int_d")),
               ("int_g1", E.ref("int_g1")), ("int_x", E.ref("int_x")))
    points = tuple((f"X = {point}", point) for point in POINTS_2)
    return (
        NewSample("orneklem", n, SEED),
        *_draws("orneklem", share),
        _outcome("orneklem", slope_gap),
        Derive("orneklem", "x13", E.sub(E.var("x"), X_MID), "X − 13"),
        OLS("ilk_additif", "orneklem", "y", ("d", "x"), "İlk örneklem, additif model: y ~ d + x"),
        OLS("ilk_etkilesimli", "orneklem", "y", ("d", "x13", "d:x13"), "İlk örneklem, etkileşimli model: y ~ d * x13"),
        ModelValue("ilk_add_d", "ilk_additif", "coef", "İlk örneklem: additif D katsayısı", term="d"),
        ModelValue("ilk_int_d", "ilk_etkilesimli", "coef", "İlk örneklem: X = 13'te fark", term="d"),
        ModelValue("ilk_int_g1", "ilk_etkilesimli", "coef", "İlk örneklem: eğim farkı", term="d:x13"),
        Support("izgara", "k", 0, 20, "Izgara: k = 0, 1, …, 20"),
        Derive("izgara", "x", E.add(X_LOW, E.div(E.var("k"), 2)), "X = 8 + k/2: 8, 8,5, …, 18"),
        Derive("izgara", "gercek_fark", _plus(E.const(GAP_MID), slope_gap, E.sub(E.var("x"), X_MID)),
               "Gerçek fark g(X) = −0,45 + γ₁(X − 13)"),
        Derive("izgara", "additif_fark", E.ref("ilk_add_d"), "Additif modelin farkı: her X'te aynı (D katsayısı)"),
        Derive("izgara", "etkilesimli_fark",
               E.add(E.ref("ilk_int_d"), E.mul(E.ref("ilk_int_g1"), E.sub(E.var("x"), X_MID))),
               "Etkileşimli modelin farkı: ĝ₁₃ + γ̂₁(X − 13)"),
        LineChart("izgara", "x", "additif_fark", "X", "Grup farkı (D = 1 eksi D = 0)",
                  f"Grup farkı: gerçek, additif ve etkileşimli model (ilk örneklem, n = {_thousands(n)})",
                  markers=False, legend="Additif model", series=(("etkilesimli_fark", "Etkileşimli model"),),
                  bands=(("gercek_fark", "Gerçek fark"),)),
        MonteCarlo("tekrarlar", REPS_2, SEED, body, collect,
                   f"{_thousands(REPS_2)} örneklem, her birinde {_thousands(n)} gözlem: additif ve etkileşimli model"),
        *(Statistic("tekrarlar", column, "mean", f"ort_{column}", f"{words}: tekrarların ortalaması", decimals=4)
          for column, words in (("add_d", "Additif model, D katsayısı"), ("add_x", "Additif model, ortak X eğimi"),
                                ("int_d", "Etkileşimli model, X = 13'teki fark"),
                                ("int_g1", "Etkileşimli model, eğim farkı"),
                                ("int_x", "Etkileşimli model, D = 0 grubunun eğimi"))),
        *(Scalar(f"int_fark_{point}", E.ref("ort_int_d") if point == X_MID else
                 _plus(E.ref("ort_int_d"), point - X_MID, E.ref("ort_int_g1")),
                 f"Etkileşimli model: X = {point} noktasında ortalama fark tahmini", decimals=3) for point in POINTS_2),
        ScalarTable(tuple((label, E.const(gap(slope_gap, point))) for label, point in points), "fark_gercek",
                    decimals=3, heading="X"),
        ScalarTable(tuple((label, E.ref("ort_add_d")) for label, _ in points), "fark_additif", decimals=3, heading="X"),
        ScalarTable(tuple((label, E.ref(f"int_fark_{point}")) for label, point in points), "fark_etkilesimli",
                    decimals=3, heading="X"),
        JoinColumns("farklar", (("Gerçek fark", "fark_gercek", "deger"), ("Additif model", "fark_additif", "deger"),
                                ("Etkileşimli model", "fark_etkilesimli", "deger")), decimals=3, heading="X"),
        Scalar("egim_ortalamasi", E.add(BETA_1, E.mul(share, slope_gap)),
               "Grup eğimlerinin p ağırlıklı ortalaması: 0,1 + p·γ₁", decimals=4),
        ScalarTable((("D = 0 grubu", E.const(BETA_1)), ("D = 1 grubu", E.const(BETA_1 + slope_gap)),
                     ("Additif ortak eğim", E.ref("egim_ortalamasi"))), "egim_gercek", decimals=3, heading="Eğim"),
        ScalarTable((("D = 0 grubu", E.ref("ort_int_x")), ("D = 1 grubu", E.add(E.ref("ort_int_x"), E.ref("ort_int_g1"))),
                     ("Additif ortak eğim", E.ref("ort_add_x"))), "egim_tahmin", decimals=3, heading="Eğim"),
        JoinColumns("egimler", (("Gerçek (ya da hedef)", "egim_gercek", "deger"),
                                ("Ortalama tahmin", "egim_tahmin", "deger")), decimals=3, heading="Eğim"),
    )


def _omitted_dgp(parameters: Parameters) -> tuple[str, ...]:
    slope_gap, share, n = _omitted_settings(parameters)
    return (
        r"Y_i = 1 + 0{,}1\,X_i + D_i\,g(X_i) + u_i, \qquad g(X) = -0{,}45 + \gamma_1 (X - 13), \qquad "
        rf"\gamma_1 = {_tex(slope_gap)}",
        rf"X_i \sim U(8,\ 18), \qquad D_i = \mathbf{{1}}\{{V_i < p\}},\ V_i \sim U(0,\ 1),\ p = {_tex(share, 1)}, \qquad "
        rf"u_i \sim N(0,\ 0{{,}}4^2), \qquad n = {_thousands(n)}",
        r"\text{Additif: } Y_i = \beta_0 + \beta_1 X_i + \gamma_0 D_i + u_i",
        r"\text{Etkileşimli: } Y_i = \alpha_0 + \beta_1 (X_i - 13) + \gamma_0 D_i + \gamma_1 D_i (X_i - 13) + u_i",
    )


def _omitted_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    slope_gap, share, _ = _omitted_settings(parameters)
    s = state.scalars
    return (
        SimMetric("Additif D katsayısı", plain(s["ort_add_d"], 3),
                  f"{_thousands(REPS_2)} örneklemde additif modelin D katsayısının ortalaması."),
        SimMetric("X = 13'te gerçek fark", plain(GAP_MID, 3), "X'in ortalamasında (13) iki grup arasındaki gerçek fark."),
        SimMetric("Additif ortak eğim", plain(s["ort_add_x"], 3),
                  "Additif modelin X eğiminin tekrarlardaki ortalaması."),
        SimMetric("0,1 + p·γ₁", plain(BETA_1 + share * slope_gap, 3),
                  "Grup eğimlerinin grup paylarıyla ağırlıklı ortalaması."),
    )


def _omitted_takeaway(state: LabState, parameters: Parameters) -> str:
    slope_gap, share, _ = _omitted_settings(parameters)
    s = state.scalars
    if slope_gap == 0:
        return ("γ₁ = 0: iki grubun doğruları paraleldir; additif model doğru modeldir. Additif D katsayısının "
                f"ortalaması {plain(s['ort_add_d'], 3)}; gerçek fark her X'te −0,45. Etkileşimli model de her X'teki "
                "farkı gerçek değerin çevresinde tahmin eder; ek parametresi (γ₁) bu ayarda gereksizdir. γ₁'i sıfırdan "
                "uzaklaştırın.")
    low, high = gap(slope_gap, X_LOW), gap(slope_gap, X_HIGH)
    return (f"Gerçek fark X = 8'de {plain(low, 3)}, X = 13'te −0,450, X = 18'de {plain(high, 3)}. Additif modelin D "
            f"katsayısının ortalaması {plain(s['ort_add_d'], 3)}: D ile X bağımsızken additif model X'in ortalamasındaki "
            "(13) farkı tahmin eder ve her X'e aynı farkı yayar; X = 8 ve X = 18'deki farkları sistematik olarak yanlış "
            f"verir. Ortak X eğimi ({plain(s['ort_add_x'], 3)}) iki grup eğiminin grup paylarıyla ağırlıklı "
            f"ortalamasına ({plain(BETA_1 + share * slope_gap, 3)}) yakındır; p'yi değiştirin. Etkileşimli model her "
            "X'teki farkı gerçek değerin çevresinde tahmin eder. Paralel doğru varsayımı veriyle ayrıca "
            "değerlendirilmelidir (§11.1, §11.4).")


OMITTED = SimExperiment(
    topic_key=TOPIC,
    number=2,
    title="Etkileşimi atlamak: additif model hangi farkı tahmin eder?",
    question="Grup farkı X'e göre değişirken etkileşimsiz (additif) model kurulursa kukla katsayısı neyi tahmin eder? "
             "Ortak X eğimi iki grubun eğimleriyle nasıl ilişkilidir?",
    note=NoteRef("11.1", objects=("§11.4",)),
    parameters=(
        SimParameter("g1", "Eğim farkı γ₁", -0.1, 0.1, -0.05, 0.01,
                     "D = 1 grubunun eğimi 0,1 + γ₁; γ₁ = 0 paralel doğrulardır.", decimals=2),
        SimParameter("p", "D = 1 grubunun payı p", 0.1, 0.9, 0.5, 0.1, "Grup kuklası D = 1{V < p}.", decimals=1),
        SimParameter("n", "Örneklem büyüklüğü n", 100, 1000, 300, 50,
                     "Her örneklemdeki gözlem sayısı; en az 100 (p = 0,1 iken küçük grup yeterince gözlem içersin).",
                     integer=True, decimals=0),
    ),
    dgp=_omitted_dgp,
    dgp_note=(
        "Notlarda bu deneyin sayısal karşılığı yoktur; §11.1'deki paralel doğru varsayımının bedelini gösterir. D ile X "
        "bağımsızdır. Her tekrarda önce X, sonra V ve u çekilir (tohum 305); grafikteki örneklem benzetimin ilk "
        "tekrarıdır."
    ),
    look_at=(
        "**Çizgi grafiği** — additif modelin sabit farkı gerçek fark doğrusunu (kesikli) nerede kesiyor?",
        "**Farklar tablosu** — X = 8, 13 ve 18'de additif ve etkileşimli model gerçek farka ne kadar yakın?",
        "**Eğimler tablosu** — additif ortak eğim iki grubun eğimlerinin arasında mı? p ile nasıl değişiyor?",
    ),
    build=_build_omitted,
    metrics=_omitted_metrics,
    takeaway=_omitted_takeaway,
    tables=(("farklar", "Tekrarlı örneklemede grup farkı: ortalama tahminler"),
            ("egimler", "Tekrarlı örneklemede X eğimleri: ortalama tahminler")),
    labels=(("x", "X"), ("v", "V"), ("u", "u"), ("d", "D (grup kuklası)"), ("y", "Y"), ("x13", "X − 13"),
            ("add_d", "Additif D katsayısı"), ("add_x", "Additif X eğimi"), ("int_d", "Etkileşimli: 13'te fark"),
            ("int_g1", "Etkileşimli: eğim farkı"), ("int_x", "Etkileşimli: D = 0 eğimi"),
            ("gercek_fark", "Gerçek fark"), ("additif_fark", "Additif model"), ("etkilesimli_fark", "Etkileşimli model")),
)


# --- Deney 3: tekli t testleri ile ortak F testi --------------------------------------------------------------------

REPS_3 = 2000
TESTS_3 = (("t: H₀: γ₀ = 0 (c'de fark)", "reddet_d"), ("t: H₀: γ₁ = 0 (eğim farkı)", "reddet_dx"),
           ("F: H₀: γ₀ = γ₁ = 0", "reddet_f"), ("İki t reddetmez, F reddeder", "yalniz_f"))


def _joint_settings(parameters: Parameters) -> tuple[float, float, int]:
    return _rounded(parameters, "g13", 2), _rounded(parameters, "c", 0), int(parameters["n"])


def _build_joint(parameters: Parameters) -> tuple:
    level, c, n = _joint_settings(parameters)
    df = n - 4
    body = (
        NewSample("tekrar_orneklem", n, None),
        *_draws("tekrar_orneklem"),
        _outcome("tekrar_orneklem", 0.0, level),
        Derive("tekrar_orneklem", "xc", _centered(c), f"X − c: c = {_short(c, 0)}"),
        OLS("model", "tekrar_orneklem", "y", ("d", "xc", "d:xc"), "Kısıtsız model: y ~ d * xc"),
        OLS("kisitli", "tekrar_orneklem", "y", ("xc",), "Kısıtlı model (γ₀ = γ₁ = 0): y ~ xc"),
        ModelValue("p_d", "model", "p", "t testi, H₀: γ₀ = 0 (c'deki fark)", term="d"),
        ModelValue("p_dx", "model", "p", "t testi, H₀: γ₁ = 0 (eğim farkı)", term="d:xc"),
        ModelValue("ssr_ur", "model", "ssr", "Kısıtsız modelin SSR'si"),
        ModelValue("ssr_r", "kisitli", "ssr", "Kısıtlı modelin SSR'si"),
        Scalar("F_ist", E.div(E.div(E.sub(E.ref("ssr_r"), E.ref("ssr_ur")), 2), E.div(E.ref("ssr_ur"), df)),
               f"F = [(SSR_R − SSR_UR)/2] / [SSR_UR/{df}]"),
        Scalar("p_F", E.fsf(E.ref("F_ist"), 2, df), f"F testinin p-değeri P(F(2, {df}) > F)", p_value=True),
    )
    rejects = {name: E.compare("lt", E.ref(source), ALPHA) for name, source in
               (("reddet_d", "p_d"), ("reddet_dx", "p_dx"), ("reddet_f", "p_F"))}
    collect = (("p_d", E.ref("p_d")), ("p_dx", E.ref("p_dx")), ("F_ist", E.ref("F_ist")), ("p_F", E.ref("p_F")),
               *rejects.items(),
               ("yalniz_f", E.mul(E.mul(E.sub(1, rejects["reddet_d"]), E.sub(1, rejects["reddet_dx"])),
                                  rejects["reddet_f"])))
    return (
        NewSample("orneklem", n, SEED),
        *_draws("orneklem"),
        _outcome("orneklem", 0.0, level),
        Derive("orneklem", "xm", _centered(c), f"Merkezlenmiş X: X − {_short(c, 0)}"),
        OLS("ilk_model", "orneklem", "y", ("d", "xm", "d:xm"), "İlk örneklem: y ~ d * xm"),
        JointTest("F_ilk", "p_F_ilk", "ilk_model", ("d", "d:xm"), "İlk örneklem: ortak F testi, H₀: γ₀ = γ₁ = 0",
                  decimals=3),
        ModelValue("sd_ilk", "ilk_model", "df_resid", "Payda serbestlik derecesi n − 4", decimals=0),
        RegressionTable(((f"Merkez c = {_short(c, 0)}", "ilk_model"),), ("d", "xm", "d:xm", INTERCEPT), "ilk_tablo",
                        "İlk örneklem: etkileşimli model ve ortak F testi", decimals=3,
                        extra=(("F", "Ortak F (γ₀ = γ₁ = 0)", ("F_ilk",)), ("F_p", "Ortak F p-değeri", ("p_F_ilk",))),
                        extra_decimals=3),
        HypothesisPlot("f", "F_ilk", 2, f"İlk örneklem: ortak F testi, F(2, {df})", "F değeri", alternative="sag",
                       df2="sd_ilk"),
        MonteCarlo("tekrarlar", REPS_3, SEED, body, collect,
                   f"{_thousands(REPS_3)} örneklem, her birinde {n} gözlem: iki t testi ve ortak F testi"),
        *(Statistic("tekrarlar", name, "mean", f"ret_{name}", f"{label}: ret oranı", decimals=4)
          for label, name in TESTS_3),
        ScalarTable(tuple((label, E.mul(100, E.ref(f"ret_{name}"))) for label, name in TESTS_3), "ret_oranlari3",
                    decimals=1, heading="Test", value="H₀ ret oranı (%)"),
        BarChart("ret_oranlari3", "deger", "Test", "Ret oranı (%)",
                 f"{_thousands(REPS_3)} örneklemde yüzde 5 düzeyinde ret oranları (merkez c = {_short(c, 0)}, n = {n})",
                 percent=True, decimals=1, references=((100 * ALPHA, "Anlamlılık düzeyi α"),)),
    )


def _joint_dgp(parameters: Parameters) -> tuple[str, ...]:
    level, c, n = _joint_settings(parameters)
    return (
        rf"Y_i = 1 + 0{{,}}1\,X_i + g_{{13}} D_i + u_i, \qquad g_{{13}} = {_tex(level)} \quad "
        r"\text{(paralel doğrular: eğim farkı yok)}",
        rf"X_i \sim U(8,\ 18), \qquad D_i = \mathbf{{1}}\{{V_i < 0{{,}}5\}},\ V_i \sim U(0,\ 1), \qquad "
        rf"u_i \sim N(0,\ 0{{,}}4^2), "
        rf"\qquad n = {n}",
        r"\text{Model: } Y_i = \alpha_0 + \beta_1 (X_i - c) + \gamma_0 D_i + \gamma_1 D_i (X_i - c) + u_i",
        rf"c = {_tex(c, 0)}; \qquad H_0: \gamma_0 = \gamma_1 = 0 \text{{ (iki doğru aynı)}}",
    )


def _joint_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    s = state.scalars
    return (
        SimMetric("γ₀ t testi ret", _percent(100 * s["ret_reddet_d"]),
                  "D katsayısının (c'deki fark) yüzde 5 düzeyinde anlamlı bulunduğu örneklemlerin payı."),
        SimMetric("γ₁ t testi ret", _percent(100 * s["ret_reddet_dx"]),
                  "Etkileşim katsayısının (eğim farkı) anlamlı bulunduğu örneklemlerin payı; gerçekte γ₁ = 0."),
        SimMetric("Ortak F ret", _percent(100 * s["ret_reddet_f"]),
                  "İki grubun doğrularının aynı olduğu hipotezinin reddedildiği örneklemlerin payı."),
        SimMetric("Yalnız F reddeder", _percent(100 * s["ret_yalniz_f"]),
                  "İki t testi de reddetmezken ortak F testinin reddettiği örneklemlerin payı."),
    )


def _joint_takeaway(state: LabState, parameters: Parameters) -> str:
    level, c, _ = _joint_settings(parameters)
    s = state.scalars
    rates = {name: _percent(100 * s[f"ret_{name}"]) for _, name in TESTS_3}
    text = (f"γ₀'ın t testi örneklemlerin {rates['reddet_d']} kadarında, γ₁'in t testi {rates['reddet_dx']} kadarında, "
            f"ortak F testi {rates['reddet_f']} kadarında reddeder. ")
    if level == 0:
        text += ("g₁₃ = 0: iki grubun doğruları aynıdır; bütün reddetmeler I. tür hatadır ve üç testin ret oranları "
                 "α = %5 çevresindedir. ")
    else:
        text += (f"Gerçek eğim farkı sıfırdır; γ₁'in ret oranı I. tür hata oranıdır. Gruplar arasında her X'te {plain(level, 2)} "
                 "fark vardır: ortak hipotez yanlıştır. ")
        if c < X_LOW or c > X_HIGH:
            text += (f"Merkez c = {_short(c, 0)} veri aralığının dışında: γ₀ bu noktadaki farktır ve çok belirsiz tahmin "
                     "edilir; γ₀ ile γ₁ tahminleri güçlü biçimde ilişkilidir. Örneklemlerin "
                     f"{rates['yalniz_f']} kadarında iki t testi de reddetmezken F testi iki doğrunun aynı olduğu "
                     "hipotezini reddeder. ")
        elif c == X_MID:
            text += ("Merkez c = 13 verinin ortasında: γ₀'ın t testi bu noktadaki farkı en güçlü biçimde sınar. ")
        else:
            text += (f"Merkez c = {_short(c, 0)} veri aralığında: γ₀'ın t testi bu noktadaki farkı doğrudan sınar; "
                     "merkez verinin ortasına (13) yaklaştıkça bu testin gücü artar. ")
    return text + ("F testi merkezlemeden etkilenmez: c'yi değiştirin, F'nin ret oranı değişmez (aynı çekilişler). "
                   "Test seçimi sonuçlara bakılarak değil, araştırma sorusuna göre yapılır (§11.8).")


JOINT = SimExperiment(
    topic_key=TOPIC,
    number=3,
    title="Tekli t testleri ile ortak F testi",
    question="Kukla katsayısının ve etkileşim katsayısının t testleri ayrı ayrı reddetmezken iki grubun doğrularının aynı "
             "olduğu hipotezi reddedilebilir mi? Merkezleme hangi testi değiştirir, hangisini değiştirmez?",
    note=NoteRef("11.8"),
    parameters=(
        SimParameter("g13", "Gruplar arası fark g₁₃", -0.3, 0.3, -0.2, 0.05,
                     "Her X'te aynı gerçek fark (paralel doğrular); 0: iki doğru aynı.", decimals=2),
        SimParameter("c", "Merkezleme noktası c", 0, 20, 0, 1, "γ₀ X = c'deki farktır; c = 0 merkezlenmemiş modeldir.",
                     integer=True, decimals=0),
        SimParameter("n", "Örneklem büyüklüğü n", 50, 500, 350, 50, "Her örneklemdeki gözlem sayısı.", integer=True,
                     decimals=0),
    ),
    dgp=_joint_dgp,
    dgp_note=(
        "Notlarda bu deneyin sayısal karşılığı yoktur; §11.8'deki üç hipotezi tekrarlı örneklemede karşılaştırır. Her "
        "tekrarda önce X, sonra V ve u çekilir (tohum 305); tablo ve F grafiği benzetimin ilk tekrarıdır. Tekrarlarda F "
        "istatistiği kısıtlı ve kısıtsız modelin SSR'lerinden hesaplanır; ilk örneklemde aynı test tek çağrıyla yapılır."
    ),
    look_at=(
        "**F grafiği** — ilk örneklemin F istatistiği reddetme bölgesinde mi?",
        "**Ret oranları** — hangi test hangi hipotezi sınıyor? Yalnız F'nin reddettiği örneklemler ne kadar?",
        "**Tablo** — ilk örneklemde iki katsayının t testleri ile ortak F testi aynı sonucu mu veriyor?",
        "**c** — merkezi 13'e taşıyın: hangi ret oranı değişiyor, hangisi aynı kalıyor?",
    ),
    build=_build_joint,
    metrics=_joint_metrics,
    takeaway=_joint_takeaway,
    tables=(("ilk_tablo", "İlk örneklem: etkileşimli model ve ortak F testi"),),
    labels=(("x", "X"), ("v", "V"), ("u", "u"), ("d", "D (grup kuklası)"), ("y", "Y"), ("xc", "X − c"),
            ("xm", "Merkezlenmiş X"), (INTERCEPT, "Sabit terim"), ("p_d", "γ₀ t testi p"), ("p_dx", "γ₁ t testi p"),
            ("F_ist", "Ortak F"), ("p_F", "Ortak F p"), ("reddet_d", "γ₀ reddedildi"), ("reddet_dx", "γ₁ reddedildi"),
            ("reddet_f", "F reddedildi"), ("yalniz_f", "Yalnız F reddetti")),
)


KONU11_EXPERIMENTS = (CENTERING, OMITTED, JOINT)
