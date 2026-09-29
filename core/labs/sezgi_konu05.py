"""Konu 5 Sezgi deneyleri: veri üretim süreci bilinen kontrollü simülasyonlar.

Deney 1  İlişkili açıklayıcı değişkenler: basit ve çoklu katsayı      (Notlar §5.7, §5.3)
Deney 2  Kontrolün doğrusal katkısını ayırmak                        (Notlar §5.5)
Deney 3  R² ve düzeltilmiş R²: gereksiz değişken eklemek              (Notlar §5.9)

Bölüm 5'te simülasyonla üretilmiş bir tablo ya da şekil yoktur; deneyler bölümün kavramlarını bilinen bir anakütle
modeliyle gösterir. Dışarıda kalan değişkenin katsayıyı hangi yönde ve ne kadar etkilediğinin formülü Konu 6'nın
konusudur; Deney 1 yalnız iki katsayının ne zaman ayrıldığını gösterir. Standart hata Konu 7'dedir. Tohum 305.
"""

from __future__ import annotations

from core.labs import expr as E
from core.labs.runner import LabState
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, plain
from core.labs.spec import (
    INTERCEPT,
    OLS,
    BarChart,
    Derive,
    Draw,
    GroupedBarChart,
    JoinColumns,
    ModelValue,
    NewSample,
    NoteRef,
    PairStatistic,
    RegressionTable,
    Residuals,
    Scalar,
    ScalarTable,
    ScatterPlot,
)

SEED = 305
TOPIC = "konu05"
BETA_1 = 1.0
"""Deney 1 ve 2'de X₁'in gerçek katsayısı."""


def _rounded(parameters: Parameters, key: str, decimals: int) -> float:
    """Kaydırıcı değeri, kodda 0,30000000000000004 gibi yazımlar olmasın diye yuvarlanır."""

    return round(float(parameters[key]), decimals)


def _tex(value: float, decimals: int = 2) -> str:
    """LaTeX içinde sondaki sıfırları atılmış sayı (0{,}5; 3)."""

    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text.replace(".", "{,}")


def _short(value: float, decimals: int = 2) -> str:
    """Sondaki sıfırları atılmış düz metin sayı (0,5; 3); yuvarlanınca sıfır olan değer "0" yazılır."""

    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "0", "") else text.replace(".", ",").replace("-", "−")


def _signed(value: float, decimals: int = 1) -> str:
    """Toplamdaki terim: "+ 0,5" ya da "− 0,5" (“+ −0,5” yazılmaz)."""

    return f"{'−' if value < 0 else '+'} {_short(abs(value), decimals)}"


def _related_draws(frame: str, a: float, beta_2: float) -> tuple:
    """X₁, v, X₂ = a·X₁ + v, u ve Y = 1 + 1·X₁ + β₂·X₂ + u (çekiliş sırası sabit)."""

    return (
        Draw(frame, "x1", "normal", 0, 1, "X₁ ~ N(0, 1)"),
        Draw(frame, "v", "normal", 0, 1, "v ~ N(0, 1)"),
        Derive(frame, "x2", E.add(E.mul(a, E.var("x1")), E.var("v")), f"X₂ = {_short(a, 1)}·X₁ + v"),
        Draw(frame, "u", "normal", 0, 1, "u ~ N(0, 1)"),
        Derive(frame, "y", E.add(E.add(E.add(1, E.mul(BETA_1, E.var("x1"))), E.mul(beta_2, E.var("x2"))), E.var("u")),
               f"Y = 1 + X₁ {_signed(beta_2)}·X₂ + u"),
    )


# --- Deney 1: basit ve çoklu katsayı -------------------------------------------------------------------

def _compare_settings(parameters: Parameters) -> tuple[int, float, float]:
    return int(parameters["n"]), _rounded(parameters, "a", 1), _rounded(parameters, "beta_2", 1)


def _build_compare(parameters: Parameters) -> tuple:
    n, a, beta_2 = _compare_settings(parameters)
    return (
        NewSample("veri", n, SEED),
        *_related_draws("veri", a, beta_2),
        OLS("basit", "veri", "y", ("x1",), "Basit model: y ~ x1"),
        OLS("coklu", "veri", "y", ("x1", "x2"), "Çoklu model: y ~ x1 + x2"),
        ModelValue("basit_b1", "basit", "coef", "Basit model: X₁ katsayısı", term="x1", decimals=3),
        ModelValue("coklu_b1", "coklu", "coef", "Çoklu model: X₁ katsayısı", term="x1", decimals=3),
        ModelValue("coklu_b2", "coklu", "coef", "Çoklu model: X₂ katsayısı", term="x2", decimals=3),
        PairStatistic("veri", "x1", "x2", "corr", "r12", "X₁ ile X₂ korelasyonu", decimals=3),
        RegressionTable((("(1) Basit model", "basit"), ("(2) Çoklu model", "coklu")), ("x1", "x2", INTERCEPT),
                        "modeller", "Basit ve çoklu model", stars=False, decimals=3, standard_errors=False, r2=False),
        ScalarTable((("Gerçek β₁", E.const(BETA_1)), ("Basit model", E.ref("basit_b1")),
                     ("Çoklu model", E.ref("coklu_b1"))), "x1_katsayilari", decimals=3, heading="Model",
                    value="X₁ katsayısı"),
        BarChart("x1_katsayilari", "deger", "Model", "X₁ katsayısı", "X₁ katsayısı: gerçek değer, basit ve çoklu model",
                 decimals=3),
    )


def _compare_dgp(parameters: Parameters) -> tuple[str, ...]:
    n, a, beta_2 = _compare_settings(parameters)
    return (
        rf"X_{{2i}} = a\,X_{{1i}} + v_i, \qquad a = {_tex(a, 1)}",
        rf"Y_i = 1 + 1 \cdot X_{{1i}} + \beta_2 X_{{2i}} + u_i, \qquad \beta_2 = {_tex(beta_2, 1)}, \qquad n = {n}",
        r"X_{1i},\ v_i,\ u_i \sim N(0,\ 1) \qquad \text{Basit model: } Y \sim X_1 \qquad \text{Çoklu model: } "
        r"Y \sim X_1 + X_2",
    )


def _compare_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    s = state.scalars
    return (
        SimMetric("Basit model: X₁ katsayısı", plain(s["basit_b1"], 3), "X₂ modelde yok."),
        SimMetric("Çoklu model: X₁ katsayısı", plain(s["coklu_b1"], 3), "X₂ sabitken; gerçek değer 1."),
        SimMetric("Çoklu model: X₂ katsayısı", plain(s["coklu_b2"], 3), "X₁ sabitken."),
        SimMetric("X₁–X₂ korelasyonu", plain(s["r12"], 3), "İki açıklayıcı değişkenin birlikte hareketi."),
    )


def _compare_takeaway(state: LabState, parameters: Parameters) -> str:
    _, a, beta_2 = _compare_settings(parameters)
    s = state.scalars
    gap = s["basit_b1"] - s["coklu_b1"]
    text = (f"Basit model X₁ katsayısını {plain(s['basit_b1'], 3)}, çoklu model {plain(s['coklu_b1'], 3)} bulur; fark "
            f"{plain(gap, 3)}. Basit model X₁ farklı olan gözlemleri X₂ farklarını ayırmadan karşılaştırır; çoklu model "
            "X₂'yi aynı tutar.")
    if a == 0 or beta_2 == 0:
        reason = ("X₂, X₁ ile birlikte hareket etmiyor (a = 0)" if a == 0
                  else "X₁ sabitken X₂, Y ile ilişkili değil (β₂ = 0)")
        return text + (f" Bu ayarda {reason}: iki katsayı birbirine yakındır, fark örneklem dalgalanmasıdır. İki "
                       "katsayının belirgin biçimde ayrılması için X₂'nin hem X₁ ile birlikte hareket etmesi hem de X₁ "
                       "sabitken Y ile ilişkili olması gerekir.")
    text += (" X₂ hem X₁ ile birlikte hareket ediyor (a ≠ 0) hem de X₁ sabitken Y ile ilişkili (β₂ ≠ 0); bu yüzden "
             "iki katsayı ayrılır. a'yı ya da β₂'yi sıfıra çekin: fark kaybolur.")
    if abs(a * beta_2) < 0.3:
        text += (" Bu ayarda a ile β₂'nin çarpımı küçüktür: sistematik fark da küçüktür ve tek örneklemde örneklem "
                 "dalgalanmasıyla karışabilir.")
    return text + (" Katsayı değişimi tek başına hangi modelin doğru olduğunu söylemez (§5.7); dışarıda kalan "
                   "değişkenin katsayıyı hangi yönde ve ne kadar etkilediği Konu 6'da formülle işlenir.")


COMPARE = SimExperiment(
    topic_key=TOPIC,
    number=1,
    title="İlişkili açıklayıcı değişkenler: basit ve çoklu katsayı",
    question="Aynı değişkenin katsayısı basit modelde ve çoklu modelde ne zaman farklı çıkar? İkinci değişkenin "
             "birinciyle ve sonuçla ilişkisi bu farkı nasıl etkiler?",
    note=NoteRef("5.7", objects=("§5.3",)),
    parameters=(
        SimParameter("a", "X₂'nin X₁ ile birlikte hareketi a", -1.5, 1.5, 0.8, 0.1,
                     "a = 0: X₂, X₁ ile ilişkisiz.", decimals=1),
        SimParameter("beta_2", "X₂'nin Y üzerindeki katsayısı β₂", -3, 3, 2, 0.5,
                     "β₂ = 0: X₁ sabitken X₂, Y ile ilişkisiz.", decimals=1),
        SimParameter("n", "Örneklem büyüklüğü n", 50, 1000, 200, 50, "Gözlem sayısı.", integer=True, decimals=0),
    ),
    dgp=_compare_dgp,
    dgp_note=(
        "X₁'in gerçek katsayısı 1'dir. Veri tek bir örneklemdir (tohum 305); kaydırıcılar veri üretim sürecini "
        "değiştirir, çekilişler aynı kalır."
    ),
    look_at=(
        "**Grafik ve tablo** — X₁'in katsayısı basit ve çoklu modelde; gerçek değer 1.",
        "**a ve β₂** — hangisi sıfır olunca iki katsayı birbirine yaklaşıyor?",
        "**Korelasyon** — X₁ ile X₂ ne kadar birlikte hareket ediyor?",
    ),
    build=_build_compare,
    metrics=_compare_metrics,
    takeaway=_compare_takeaway,
    tables=(("modeller", "Basit ve çoklu model (bağımlı değişken: Y)"),),
    labels=(("x1", "X₁"), ("x2", "X₂"), ("v", "v"), ("u", "u"), ("y", "Y"), (INTERCEPT, "Sabit terim")),
)


# --- Deney 2: kontrolün doğrusal katkısını ayırmak ------------------------------------------------

def _partial_settings(parameters: Parameters) -> tuple[int, float]:
    return int(parameters["n"]), _rounded(parameters, "a", 1)


BETA_2_PARTIAL = 2.0
"""Deney 2'de X₂'nin gerçek katsayısı."""


def _build_partial(parameters: Parameters) -> tuple:
    n, a = _partial_settings(parameters)
    return (
        NewSample("veri", n, SEED),
        *_related_draws("veri", a, BETA_2_PARTIAL),
        OLS("coklu", "veri", "y", ("x1", "x2"), "Çoklu model: y ~ x1 + x2"),
        ModelValue("coklu_b1", "coklu", "coef", "Çoklu model: X₁ katsayısı", term="x1", decimals=4),
        OLS("yardimci_y", "veri", "y", ("x2",), "1. adım: Y'yi X₂'ye göre tahmin et"),
        Residuals("veri", "y_artik", "yardimci_y", "Y'nin X₂ ile açıklanamayan kısmı"),
        OLS("yardimci_x1", "veri", "x1", ("x2",), "2. adım: X₁'i X₂'ye göre tahmin et"),
        Residuals("veri", "x1_artik", "yardimci_x1", "X₁'in X₂ ile açıklanamayan kısmı"),
        OLS("kismi", "veri", "y_artik", ("x1_artik",), "3. adım: artıkların artıklara göre regresyonu"),
        ModelValue("kismi_egim", "kismi", "coef", "Artıklar regresyonunun eğimi", term="x1_artik", decimals=4),
        Scalar("fark", E.sub(E.ref("kismi_egim"), E.ref("coklu_b1")), "Aradaki fark", decimals=12),
        OLS("basit", "veri", "y", ("x1",), "Karşılaştırma: basit model y ~ x1"),
        ModelValue("basit_b1", "basit", "coef", "Basit model: X₁ katsayısı", term="x1", decimals=4),
        ScatterPlot("veri", "x1", "y", "X₁", "Y", "Ham veri: Y ile X₁ (eğim = basit model katsayısı)", fit_line=True,
                    size=7, opacity=0.5),
        ScatterPlot("veri", "x1_artik", "y_artik", "X₁'in X₂'den arındırılmış kısmı", "Y'nin X₂'den arındırılmış kısmı",
                    "Artıklar: eğim = çoklu modeldeki X₁ katsayısı", fit_line=True, size=7, opacity=0.5),
    )


def _partial_dgp(parameters: Parameters) -> tuple[str, ...]:
    n, a = _partial_settings(parameters)
    return (
        rf"X_{{2i}} = a\,X_{{1i}} + v_i, \qquad Y_i = 1 + X_{{1i}} + 2X_{{2i}} + u_i, \qquad a = {_tex(a, 1)}, "
        rf"\quad n = {n}",
        r"\text{1) } Y \text{'yi } X_2\text{'ye, 2) } X_1\text{'i } X_2\text{'ye göre tahmin et; 3) artıkları "
        r"birbirine göre regresyona tabi tut}",
    )


def _partial_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    s = state.scalars
    return (
        SimMetric("Çoklu model: X₁ katsayısı", plain(s["coklu_b1"], 4), "Y ~ X₁ + X₂."),
        SimMetric("Artıklar regresyonunun eğimi", plain(s["kismi_egim"], 4), "Üç adımlı hesap."),
        SimMetric("Aradaki fark (mutlak değer)", "< 10⁻¹⁰" if abs(s["fark"]) < 1e-10 else plain(abs(s["fark"]), 6),
                  "Hesap hassasiyeti içinde sıfır: iki yol aynı sayıyı verir."),
        SimMetric("Basit model: X₁ katsayısı", plain(s["basit_b1"], 4), "X₂ ayrılmadan."),
    )


def _partial_takeaway(state: LabState, parameters: Parameters) -> str:
    _, a = _partial_settings(parameters)
    s = state.scalars
    text = (f"Artıklar regresyonunun eğimi {plain(s['kismi_egim'], 4)}, çoklu modeldeki X₁ katsayısı "
            f"{plain(s['coklu_b1'], 4)}: aynı sayı. “X₂ sabitken” yorumu, X₂'nin doğrusal katkısının hem Y'den hem X₁'den "
            "ayrılması demektir; veri içinde birebir aynı X₂ değerli gözlemleri bulmak gerekmez (§5.5).")
    if a == 0:
        return text + (" a = 0 iken X₁ zaten X₂ ile ilişkisiz: ayırma X₁'i neredeyse değiştirmez ve basit model "
                       f"katsayısı ({plain(s['basit_b1'], 4)}) çoklu modelinkine yakındır.")
    return text + (f" Ham verideki eğim (basit model) {plain(s['basit_b1'], 4)}: X₂ farklarını da taşır. |a| "
                   "büyüdükçe iki grafiğin eğimleri birbirinden uzaklaşır; artıkların yatay yayılımı daralır, çünkü "
                   "X₁'in X₂'den bağımsız kalan hareketi azalır.")


PARTIAL = SimExperiment(
    topic_key=TOPIC,
    number=2,
    title="Kontrolün doğrusal katkısını ayırmak",
    question="Çoklu regresyonda “X₂ sabitken” X₁'in katsayısı nasıl hesaplanır? X₂'nin katkısını Y'den ve X₁'den "
             "ayırıp kalanları karşılaştırırsak ne buluruz?",
    note=NoteRef("5.5"),
    parameters=(
        SimParameter("a", "X₂'nin X₁ ile birlikte hareketi a", -1.5, 1.5, 0.8, 0.1,
                     "a = 0: X₁ ile X₂ ilişkisiz.", decimals=1),
        SimParameter("n", "Örneklem büyüklüğü n", 30, 500, 150, 10, "Gözlem sayısı.", integer=True, decimals=0),
    ),
    dgp=_partial_dgp,
    dgp_note=(
        "Üç adım notlardaki WAGE1 uygulamasıyla aynıdır (§5.5); burada veri üretim süreci bilinir. Eşitlik her "
        "örneklemde cebirsel olarak sağlanır (Frisch–Waugh–Lovell)."
    ),
    look_at=(
        "**İki grafik** — soldaki ham ilişki, sağdaki kontrolden arındırılmış ilişki.",
        "**Eğimler** — artıklar regresyonunun eğimi çoklu modeldeki katsayıyla aynı mı?",
        "**|a|** — büyüdükçe arındırılmış X₁'in yayılımı ne oluyor?",
    ),
    build=_build_partial,
    metrics=_partial_metrics,
    takeaway=_partial_takeaway,
    labels=(("x1", "X₁"), ("x2", "X₂"), ("v", "v"), ("u", "u"), ("y", "Y"), ("y_artik", "Arındırılmış Y"),
            ("x1_artik", "Arındırılmış X₁")),
)


# --- Deney 3: R² ve düzeltilmiş R² ------------------------------------------------------------------------

def _fit_settings(parameters: Parameters) -> tuple[int, int]:
    return int(parameters["n"]), int(parameters["k"])


def _build_fit(parameters: Parameters) -> tuple:
    n, k = _fit_settings(parameters)
    noise = [f"z{index}" for index in range(1, k + 1)]
    models = []
    for size in range(k + 1):
        regressors = ("x", *noise[:size])
        models += [
            OLS(f"m{size}", "veri", "y", regressors, f"{size} gereksiz değişkenli model: y ~ {' + '.join(regressors)}"),
            ModelValue(f"r2_{size}", f"m{size}", "r2", f"R² ({size} gereksiz değişken)", decimals=4),
            ModelValue(f"r2d_{size}", f"m{size}", "adj_r2", f"Düzeltilmiş R² ({size} gereksiz değişken)", decimals=4),
        ]
    return (
        NewSample("veri", n, SEED),
        Draw("veri", "x", "normal", 0, 1, "X ~ N(0, 1)"),
        Draw("veri", "u", "normal", 0, 1, "u ~ N(0, 1)"),
        Derive("veri", "y", E.add(E.add(1, E.mul(0.5, E.var("x"))), E.var("u")), "Y = 1 + 0,5·X + u"),
        *(Draw("veri", name, "normal", 0, 1, f"Gereksiz değişken {name}: Y ile ilişkisiz, N(0, 1)") for name in noise),
        *models,
        ScalarTable(tuple((str(size), E.ref(f"r2_{size}")) for size in range(k + 1)), "r2_tablo", decimals=4),
        ScalarTable(tuple((str(size), E.ref(f"r2d_{size}")) for size in range(k + 1)), "r2d_tablo", decimals=4),
        JoinColumns("uyum", (("R²", "r2_tablo", "deger"), ("Düzeltilmiş R²", "r2d_tablo", "deger")), decimals=4,
                    heading="Gereksiz değişken sayısı"),
        GroupedBarChart("uyum", "Modele eklenen gereksiz değişken sayısı", "Uyum ölçüsü",
                        "Gereksiz değişken ekledikçe R² ve düzeltilmiş R²", series="sutun", decimals=3, labels=False),
    )


def _fit_dgp(parameters: Parameters) -> tuple[str, ...]:
    n, m = _fit_settings(parameters)
    noise = {1: r"Z_{1i}", 2: r"Z_{1i},\ Z_{2i}"}.get(m, rf"Z_{{1i}}, \dots, Z_{{{m}i}}")
    return (
        rf"Y_i = 1 + 0{{,}}5\,X_i + u_i, \qquad X_i,\ u_i \sim N(0,\ 1), \qquad n = {n}",
        rf"{noise} \sim N(0,\ 1) \text{{ bağımsız: Y ile ilişkisiz}} \qquad (m = {m})",
        r"\text{Model } j: Y \sim X + Z_1 + \cdots + Z_j,\ k = j + 1, \qquad \bar R^2 = 1 - "
        r"\frac{\text{HKT}/(n-k-1)}{\text{TKT}/(n-1)}",
    )


def _fit_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    _, m = _fit_settings(parameters)
    s = state.scalars
    return (
        SimMetric("R²: yalnız X", plain(s["r2_0"], 4), "Gerçek modeldeki tek açıklayıcı değişken."),
        SimMetric(f"R²: X ve {m} gereksiz değişken", plain(s[f"r2_{m}"], 4), "Gereksiz değişkenlerle."),
        SimMetric("Düzeltilmiş R²: yalnız X", plain(s["r2d_0"], 4), "Katsayı sayısının maliyeti hesaba katılır."),
        SimMetric(f"Düzeltilmiş R²: {m} gereksiz değişkenle", plain(s[f"r2d_{m}"], 4), "Gereksiz değişkenlerle."),
    )


def _fit_takeaway(state: LabState, parameters: Parameters) -> str:
    n, m = _fit_settings(parameters)
    table = state.tables["uyum"]
    r2 = table["R²"].to_numpy()
    adjusted = table["Düzeltilmiş R²"].to_numpy()
    falls = int((adjusted[1:] < adjusted[:-1]).sum())
    text = (f"R² yalnız X'le {plain(r2[0], 4)}, {m} gereksiz değişkenle {plain(r2[-1], 4)}: hiçbir adımda azalmaz, "
            "çünkü EKK yeni katsayıyı sıfır seçerek eski modeli koruyabilir (§5.9). Düzeltilmiş R² her yeni katsayının "
            "maliyetini hesaba katar: ")
    if falls == 0:
        text += ("bu örneklemde hiçbir adımda düşmedi; gereksiz değişkenlerin rastlantısal uyumu her adımda bu "
                 f"maliyeti karşıladı. Son modelde {plain(adjusted[-1], 4)}.")
    else:
        text += f"{m} adımın {falls} tanesinde düşer; son modelde {plain(adjusted[-1], 4)}."
    if adjusted[-1] > adjusted[0]:
        text += (" Son modelin düzeltilmiş R²'si yalnız X'li modelinkinden yüksek: gereksiz değişkenlerin rastlantısal "
                 "uyumu bu örneklemde düzeltme payını aştı; düzeltilmiş R² de rastlantıdan etkilenir.")
    return text + (" Gereksiz değişkenler Y ile ilişkisizdir; R²'deki artış yalnız bu örneklemdeki rastlantısal "
                   "uyumdur. Ceza n'ye bağlıdır: gözlem sayısı küçükken iki ölçü arasındaki fark büyür. Düzeltilmiş R² "
                   "de tek başına model seçimi kuralı değildir.")


FIT = SimExperiment(
    topic_key=TOPIC,
    number=3,
    title="R² ve düzeltilmiş R²: gereksiz değişken eklemek",
    question="Y ile hiç ilişkisi olmayan değişkenleri modele eklersek R² ne olur? Düzeltilmiş R² bu artışa nasıl "
             "tepki verir?",
    note=NoteRef("5.9", objects=("Denklem 5.8",)),
    parameters=(
        SimParameter("k", "Eklenen gereksiz değişken sayısı m", 1, 15, 10, 1,
                     "Her biri Y ile ilişkisiz. Formüldeki k, X ile birlikte açıklayıcı değişken sayısıdır (k = j + 1).",
                     integer=True, decimals=0),
        SimParameter("n", "Örneklem büyüklüğü n", 20, 200, 40, 5, "Gözlem sayısı.", integer=True, decimals=0),
    ),
    dgp=_fit_dgp,
    dgp_note=(
        "Gerçek modelde yalnız X vardır. Gereksiz değişkenler birer birer eklenir; her model aynı örneklemde tahmin "
        "edilir (tohum 305)."
    ),
    look_at=(
        "**Grafik** — her gereksiz değişken eklendiğinde R² ve düzeltilmiş R².",
        "**R²** — hiç düşüyor mu?",
        "**n** — örneklem büyüdükçe iki ölçünün farkı ne oluyor?",
    ),
    build=_build_fit,
    metrics=_fit_metrics,
    takeaway=_fit_takeaway,
    labels=(("x", "X"), ("u", "u"), ("y", "Y")),
)


KONU05_EXPERIMENTS = (COMPARE, PARTIAL, FIT)
