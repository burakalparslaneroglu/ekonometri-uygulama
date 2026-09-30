"""Konu 11 uygulaması: etkileşim terimleri ve gruplar arasında sabit ile eğim farkları (gerçek veri).

Bölüm 11'in çözümlü örnekleri bölüm sırasıyla: additif kukla modelinin paralel doğruları (§11.1), etkileşim teriminin
kurulması ve grup denklemleri (§11.2), dört olası yapı (§11.3), koşullu grup farkı ve merkezleme (§11.4–11.5),
WAGE1 uygulaması (§11.6, Kod 11.1–11.2, Tablo 11.3–11.4, Şekil 11.3–11.4), HPRICE1 uygulaması (§11.7, Kod
11.3–11.4, Tablo 11.5, Şekil 11.5–11.6), etkileşim hipotez testleri (§11.8) ve formül yazımı ile makale tablosu
(§11.9, Tablo 11.6). Her ``Check`` notlarda basılı bir sayıdır; değer notlardan kopyalanmıştır.

Etkileşim terimi iki değişkenin çarpımıdır ve ``female:educ12`` yazılır; ana etkiler ve etkileşim birlikte
``female * educ12`` formülüyle kurulur (notlardaki gibi). Etkileşim: additif modeldeki kukla (Adım 1), etkileşimdeki
nicel değişken (Adım 2), grafikteki yapı (Adım 3), merkezleme noktası (Adım 4), eğitimle etkileşen kukla (Adım 5–6),
koşullu farkın hesaplandığı eğitim düzeyi (Adım 6), HPRICE1'de kolonyal tarzla etkileşen değişken (Adım 7–8), gösterilen
test (Adım 9) ve formül yazımı (Adım 10). Standart hatalar klasik EKK standart hatalarıdır (dayanıklı çıkarım Konu 12).
"""

from __future__ import annotations

from core import wooldridge_data as W
from core.labs import expr as E
from core.labs.sezgi import plain
from core.labs.spec import (
    INTERCEPT,
    OLS,
    Check,
    Choice,
    CoefficientTable,
    Count,
    CoefTarget,
    CopyFrame,
    Derive,
    HypothesisPlot,
    JoinColumns,
    JointTest,
    LabSpec,
    LineChart,
    LoadWooldridge,
    ModelTarget,
    ModelValue,
    NoteRef,
    NumberChoice,
    RegressionTable,
    Scalar,
    ScalarTable,
    ScalarTarget,
    ShowFrame,
    ShowModel,
    Statistic,
    Support,
    TableTarget,
    interactive_step,
)

WAGE = "wage1"
HOUSE = "hprice1"
CONTROLS = ("exper", "tenure")
"""WAGE1 modellerinin kontrol değişkenleri: deneyim ve kıdem."""
DUMMIES = ("female", "married", "nonwhite")
"""Eğitimle etkileşebilecek kuklalar: kadın, evli, beyaz olmayan."""
DUMMY_WORDS = {"female": ("kadın", "erkek"), "married": ("evli", "evli olmayan"),
               "nonwhite": ("beyaz olmayan", "beyaz")}
"""Kuklanın 1 ve 0 grupları; 0 grubu referanstır."""
DUMMY_TITLES = {"female": "Kadın", "married": "Evli", "nonwhite": "Beyaz olmayan", "colonial": "Kolonyal"}
X_OPTIONS = ("educ12", "exper", "tenure")
"""WAGE1'de kukla ile etkileşen nicel değişkenler (Adım 2)."""
X_WORDS = {"educ12": "eğitim", "exper": "deneyim", "tenure": "kıdem", "lotsize10k": "arsa büyüklüğü",
           "sqrft100": "konut büyüklüğü", "bdrms": "yatak odası sayısı"}
X_UNITS = {"educ12": "yıl", "exper": "yıl", "tenure": "yıl", "lotsize10k": "1.000 fit²", "sqrft100": "100 fit²",
           "bdrms": "oda"}
HOUSE_X = ("lotsize10k", "sqrft100", "bdrms")
"""HPRICE1 modelinin nicel açıklayıcıları; biri kolonyal tarzla etkileşir (Adım 7–8)."""
AVERAGE_LABELS = {"educ12": "Eğitim − 12 (yıl)", "exper": "Deneyim (yıl)", "tenure": "Kıdem (yıl)",
                  "lotsize10k": "(Arsa − 10.000)/1.000 (bin fit²)", "sqrft100": "Konut büyüklüğü/100 (yüz fit²)",
                  "bdrms": "Yatak odası sayısı"}
"""Doğruların çiziminde ortalamada tutulan değişkenlerin modeldeki ölçeğiyle etiketi (ör. educ12'nin ortalaması 0,56)."""
HOUSE_RANGES = {"lotsize10k": "1.000–92.681 fit²", "sqrft100": "1.171–3.880 fit²", "bdrms": "2–7 oda"}
"""HPRICE1'de nicel açıklayıcıların özgün ölçekteki gözlenen aralığı (88 konut; testle doğrulanır)."""
CAUSAL_WORDS = {"female": "cinsiyete dayalı ayrımcılığın", "married": "evliliğin", "nonwhite": "ayrımcılığın"}
"""Adım 5: gözlemsel veriyle ölçülemeyen nedensel etki, seçilen kuklaya göre."""
FEW_OBSERVATIONS = 10
"""Adım 4: merkezleme noktasında bundan az gözlem varsa farkın büyük ölçüde doğrusal varsayıma dayandığı yazılır."""


def _scalar(name: str, expected: float, label: str, decimals: int) -> Check:
    return Check(label, ScalarTarget(name), expected, decimals)


def _p(value: float, decimals: int = 3) -> str:
    if value < 0.5 * 10 ** -decimals:
        return "p < " + plain(10 ** -decimals, decimals)
    return f"p = {plain(value, decimals)}"


def _signed(value: float, decimals: int) -> str:
    """Toplamın ikinci terimi: negatifse parantez içinde (1,5480 + (−0,2973))."""

    return plain(value, decimals) if value >= 0 else f"({plain(value, decimals)})"


def _slope_sum(first: float, second: float, exact: float, source: str = "") -> str:
    """İki katsayının toplamı ekrandaki (dört basamaklı) terimlerle: 0,0903 + (−0,0072) = 0,0831. Yuvarlanmamış
    katsayılarla toplam farklı yuvarlanıyorsa parantezde verilir; ``source`` notlardaki tablo (ör. "Tablo 11.3")."""

    rounded = float(f"{first:.4f}") + float(f"{second:.4f}")  # ekranda yazılan terimlerin toplamı
    text = f"{plain(first, 4)} + {_signed(second, 4)} = {plain(rounded, 4)}"
    notes = []
    if plain(rounded, 4) != plain(exact, 4):
        notes.append(f"yuvarlanmamış katsayılarla {plain(exact, 4)}")
    if source:
        notes.append(source)
    return text + (f" ({'; '.join(notes)})" if notes else "")


def _load_wage() -> LoadWooldridge:
    return LoadWooldridge(WAGE, "WAGE1 veri seti (Wooldridge, 2020): 526 çalışan")


def _center_education() -> Derive:
    return Derive(WAGE, "educ12", E.sub(E.var("educ"), 12), "Eğitim 12 yıl etrafında merkezlenir: educ12 = educ − 12")


def _load_house() -> tuple:
    return (
        LoadWooldridge(HOUSE, "HPRICE1 veri seti (Wooldridge, 2020): 88 konut"),
        Derive(HOUSE, "lotsize10k", E.div(E.sub(E.var("lotsize"), 10000), 1000),
               "Arsa büyüklüğü 10.000 fit² etrafında merkezlenir, 1.000 fit² biriminde: (lotsize − 10000)/1000"),
        Derive(HOUSE, "sqrft100", E.div(E.var("sqrft"), 100), "Konut büyüklüğü 100 fit² biriminde: sqrft/100"),
    )


def _formula(outcome: str, terms: tuple[str, ...]) -> str:
    return OLS("gecici", "gecici", outcome, terms, "").formula


def _others(chosen: str, options: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(item for item in options if item != chosen)


def _wage_terms(dummy: str, x: str = "educ12") -> tuple[str, ...]:
    """WAGE1 etkileşim modelinin terimleri: kukla, nicel değişken, etkileşim ve diğer kontroller."""

    return (dummy, x, f"{dummy}:{x}", *_others(x, ("educ12", *CONTROLS)))


def _group_lines(frame: str, axis: str, grid: tuple[int, int], prefix: str, dummy: str, slope_term: str,
                 center: float, others: tuple[str, ...], averages: str, log: bool = False) -> tuple:
    """İki grubun tahmin edilen doğruları bir ızgarada: diğer açıklayıcılar örneklem ortalamasında.

    ``prefix`` katsayı skalerlerinin önekidir (``{prefix}_b0``, ``{prefix}_g0``, ``{prefix}_b1``, ``{prefix}_g1`` ve
    ``{prefix}_{değişken}``); ``averages`` ortalama skalerlerinin önekidir. ``center``: ızgaradaki değer ile modeldeki
    değişken arasındaki fark (educ12 = educ − 12 gibi); ``log`` ise doğrular exp() ile saatlik ücrete çevrilir."""

    lower, upper = grid
    x = E.sub(E.var(axis), center) if center else E.var(axis)
    line = E.add(E.ref(f"{prefix}_b0"), E.mul(E.ref(f"{prefix}_b1"), x))
    for term in others:
        line = E.add(line, E.mul(E.ref(f"{prefix}_{term}"), E.ref(f"{averages}_{term}")))
    shifted = E.add(E.add(E.var("grup0"), E.ref(f"{prefix}_g0")), E.mul(E.ref(f"{prefix}_g1"), x))
    one, zero = DUMMY_WORDS.get(dummy, ("kolonyal", "diğer"))
    operations: list = [
        Support(frame, axis, lower, upper, f"Izgara: {lower}, …, {upper}"),
        Derive(frame, "grup0", line, f"{zero.capitalize()} ({dummy} = 0): β̂₀ + β̂₁·x; diğer açıklayıcılar ortalamada"),
        Derive(frame, "grup1", shifted, f"{one.capitalize()} ({dummy} = 1): önceki doğru + γ̂₀ + γ̂₁·x"),
    ]
    if log:
        operations += [
            Derive(frame, "ucret0", E.exp(E.var("grup0")), f"{zero.capitalize()}: exp(tahmin edilen ln(ücret))"),
            Derive(frame, "ucret1", E.exp(E.var("grup1")), f"{one.capitalize()}: exp(tahmin edilen ln(ücret))"),
        ]
    return tuple(operations)


def _coefficients(model: str, prefix: str, dummy: str, slope_term: str, others: tuple[str, ...],
                  interaction: bool = True, shown: bool = True) -> tuple:
    """Katsayı skalerleri: sabit, kukla, eğim, (etkileşim) ve diğer açıklayıcılar. ``shown=False``: sayılar aynı
    adımda bir çıktıda ya da tabloda görünüyor; ölçü kutusu olarak tekrar edilmez."""

    values = [
        ModelValue(f"{prefix}_b0", model, "coef", "Sabit terim β̂₀", term=INTERCEPT, decimals=4, shown=shown),
        ModelValue(f"{prefix}_g0", model, "coef", "Kukla katsayısı γ̂₀", term=dummy, decimals=4, shown=shown),
        ModelValue(f"{prefix}_b1", model, "coef", f"{X_WORDS.get(slope_term, slope_term).capitalize()} eğimi β̂₁",
                   term=slope_term, decimals=4, shown=shown),
    ]
    if interaction:
        values.append(ModelValue(f"{prefix}_g1", model, "coef", "Etkileşim katsayısı γ̂₁",
                                 term=f"{dummy}:{slope_term}", decimals=4, shown=shown))
    values += [ModelValue(f"{prefix}_{term}", model, "coef", f"{X_WORDS.get(term, term).capitalize()} katsayısı",
                          term=term, decimals=4, shown=shown) for term in others]
    return tuple(values)


def _averages(frame: str, prefix: str, terms: tuple[str, ...]) -> tuple:
    return tuple(Statistic(frame, term, "mean", f"{prefix}_{term}",
                           f"{AVERAGE_LABELS.get(term, term)}: örneklem ortalaması", decimals=4)
                 for term in terms)


# --- Adım 1: additif kukla modeli ve paralel doğrular ------------------------------------------------------------

DUMMY_1 = Choice("adim1_kukla", "Kukla değişken D", W.options(WAGE, DUMMIES), "female",
                 help="Notlarda kadın kuklası; erkek çalışanlar referans gruptur (Tablo 11.6, Sütun 1).")


def _additive(choices) -> tuple:
    dummy = choices["adim1_kukla"]
    one, zero = DUMMY_WORDS[dummy]
    notes = dummy == "female"
    terms = (dummy, "educ12", *CONTROLS)
    side: tuple = () if notes else (  # notlardaki kadın kuklası seçilen kuklayla yan yana
        OLS("m_add_n", WAGE, "lwage", ("female", "educ12", *CONTROLS), "Notlardaki additif model: lwage ~ female + "
            "educ12 + exper + tenure"),
        RegressionTable((("Notlar: kadın", "m_add_n"), (f"Seçiminiz: {one}", "m_add")),
                        (INTERCEPT, *dict.fromkeys(("female", dummy)), "educ12", *CONTROLS), "yan111",
                        "Notlardaki additif model ve seçtiğiniz kukla", decimals=4, exact=True, adj_r2=True),
    )
    return (
        _load_wage(),
        _center_education(),
        OLS("m_add", WAGE, "lwage", terms, f"Additif kukla modeli: {_formula('lwage', terms)}"),
        ShowModel("m_add", "Additif kukla modelinin Python çıktısı" + ("" if notes else " (seçiminiz)"),
                  columns=("coef", "se", "t", "p"), stats=("nobs", "r2", "adj_r2"), decimals=(("se", 4), ("p", 4)),
                  exact=True),
        *side,
        *_coefficients("m_add", "a1", dummy, "educ12", CONTROLS, interaction=False, shown=False),  # çıktıda görünür
        *_averages(WAGE, "ort1", CONTROLS),
        Scalar("a1_g1", E.const(0), "Additif modelde eğim farkı yoktur (γ₁ = 0)", decimals=0, shown=False),
        *_group_lines("dogru1", "educ", (0, 18), "a1", dummy, "educ12", 12, CONTROLS, "ort1"),
        LineChart("dogru1", "educ", "grup0", "Eğitim (yıl)", "Tahmin edilen ln(saatlik ücret)",
                  f"WAGE1, additif kukla modeli: {one} ve {zero} çalışanların doğruları paraleldir (deneyim ve kıdem "
                  "ortalamada)", markers=False, series=(("grup1", one.capitalize()),), legend=zero.capitalize()),
        Derive("dogru1", "fark", E.sub(E.var("grup1"), E.var("grup0")), f"Dikey fark: {one} − {zero}"),
        Statistic("dogru1", "fark", "min", "fark1_min", "Dikey fark: en küçük (her eğitim düzeyinde)", decimals=4),
        Statistic("dogru1", "fark", "max", "fark1_max", "Dikey fark: en büyük", decimals=4),
    )


def _additive_note(state, choices) -> str:
    dummy = choices["adim1_kukla"]
    one, zero = DUMMY_WORDS[dummy]
    s = state.scalars
    return (f"İki doğrunun eğimi aynıdır (eğitim katsayısı {plain(s['a1_b1'], 4)}); dikey fark her eğitim düzeyinde "
            f"kukla katsayısıdır: en küçük ve en büyük fark {plain(s['fark1_min'], 4)} ve {plain(s['fark1_max'], 4)}. "
            f"Additif model {one} ve {zero} çalışanlara farklı sabit verir, farklı eğim vermez; grup farkı eğitimden "
            "bağımsız varsayılır. Bu varsayım veriyle ayrıca değerlendirilir: eğimin gruba göre değişmesine izin veren "
            "terim etkileşim terimidir (Adım 2, §11.1).")


# --- Adım 2: etkileşim terimi ve grup denklemleri ----------------------------------------------------------------

X_2 = Choice("adim2_x", "Kadın kuklasıyla etkileşen nicel değişken X",
             (("educ12", "Eğitim − 12 (yıl)"), ("exper", "Potansiyel deneyim (yıl)"),
              ("tenure", "Mevcut işverendeki kıdem (yıl)")), "educ12",
             help="Notlarda 12 yıl etrafında merkezlenmiş eğitim (§11.6).")


GROUP_ROWS = (("Erkek (female = 0): sabit β̂₀", "b0"), ("Kadın (female = 1): sabit β̂₀ + γ̂₀", "sabit1"),
              ("Erkek: eğim β̂₁", "b1"), ("Kadın: eğim β̂₁ + γ̂₁", "egim1"), ("Sabit farkı γ̂₀", "g0"),
              ("Eğim farkı γ̂₁", "g1"))
"""Grup denklemleri tablosunun satırları (Tablo 11.1'deki roller) ve skaler sonekleri."""


def _group_equations(model: str, prefix: str, x: str, shown: bool = True) -> tuple:
    """Etkileşimli modelin grup denklemleri: iki grubun sabiti ve eğimi, sabit ve eğim farkı."""

    word = X_WORDS[x]
    return (
        *_coefficients(model, prefix, "female", x, _others(x, ("educ12", *CONTROLS)), shown=shown),
        Scalar(f"{prefix}_sabit1", E.add(E.ref(f"{prefix}_b0"), E.ref(f"{prefix}_g0")), "Kadın grubunun sabiti β̂₀ + γ̂₀",
               decimals=4, shown=shown),
        Scalar(f"{prefix}_egim1", E.add(E.ref(f"{prefix}_b1"), E.ref(f"{prefix}_g1")),
               f"Kadın grubunun {word} eğimi β̂₁ + γ̂₁", decimals=4, shown=shown),
    )


def _interaction(choices) -> tuple:
    x = choices["adim2_x"]
    word = X_WORDS[x]
    notes = x == "educ12"
    terms = _wage_terms("female", x)
    others = _others(x, ("educ12", *CONTROLS))
    grids = {"educ12": ("educ", (0, 18), 12), "exper": ("exper", (1, 51), 0), "tenure": ("tenure", (0, 44), 0)}
    axis, grid, center = grids[x]
    axis_label = {"educ": "Eğitim (yıl)", "exper": "Potansiyel deneyim (yıl)", "tenure": "Kıdem (yıl)"}[axis]
    if notes:
        table: tuple = (
            ScalarTable(tuple((label.replace("eğim", "eğitim eğimi"), E.ref(f"e2_{suffix}"))
                              for label, suffix in GROUP_ROWS),
                        "grup_denklemleri", decimals=4, heading="Katsayı", value="Tahmin",
                        title="Grup denklemleri: sabitler ve eğimler (Tablo 11.1'deki roller)"),
        )
    else:  # notlardaki eğitim etkileşimi seçilen değişkenle yan yana
        table = (
            OLS("m_etk_n", WAGE, "lwage", _wage_terms("female"), "Notlardaki model: lwage ~ female * educ12 + exper "
                "+ tenure"),
            *_group_equations("m_etk_n", "e2n", "educ12", shown=False),
            *(ScalarTable(tuple((label, E.ref(f"{prefix}_{suffix}")) for label, suffix in GROUP_ROWS), result,
                          decimals=4, heading="Katsayı") for prefix, result in (("e2n", "gd_notlar"), ("e2", "gd_secim"))),
            JoinColumns("grup_denklemleri", (("Notlar: eğitim", "gd_notlar", "deger"),
                                             (f"Seçiminiz: {word}", "gd_secim", "deger")),
                        decimals=4, heading="Katsayı",
                        title="Grup denklemleri: notlardaki model ve seçtiğiniz değişken (Tablo 11.1'deki roller)"),
        )
    return (
        CopyFrame("carpim", WAGE, "Etkileşim sütunu için veri setinin kopyası (özgün veri değişmez)"),
        Derive("carpim", "kadin_x", E.mul(E.var("female"), E.var(x)),
               f"Etkileşim terimi: female × {x} (erkekte 0, kadında {word})"),
        ShowFrame("carpim", ("female", x, "kadin_x"), "İlk sekiz çalışan: çarpım erkekte 0, kadında X'in kendisidir",
                  rows=(1, 2, 3, 4, 5, 6, 7, 8)),
        OLS("m_etk", WAGE, "lwage", terms, f"Etkileşimli model: {_formula('lwage', terms)}"),
        *_group_equations("m_etk", "e2", x, shown=False),
        Scalar("e2_egim1_yuvarlak", E.add(E.roundto(E.ref("e2_b1"), 4), E.roundto(E.ref("e2_g1"), 4)),
               "Dört basamaklı katsayılarla (Tablo 11.3 notu): β̂₁ + γ̂₁", decimals=4, shown=False),
        *table,
        *_averages(WAGE, "ort2", others),
        *_group_lines("dogru2", axis, grid, "e2", "female", x, center, others, "ort2"),
        LineChart("dogru2", axis, "grup0", axis_label, "Tahmin edilen ln(saatlik ücret)",
                  f"WAGE1, etkileşimli model: kadın ve erkek çalışanlarda farklı {word} eğimi (diğer değişkenler "
                  "ortalamada)",
                  markers=False, series=(("grup1", "Kadın"),), legend="Erkek"),
    )


def _interaction_note(state, choices) -> str:
    x = choices["adim2_x"]
    word = X_WORDS[x]
    s = state.scalars
    total = _slope_sum(s["e2_b1"], s["e2_g1"], s["e2_egim1"], "Tablo 11.3" if x == "educ12" else "")
    return (f"Erkek çalışanların {word} eğimi β̂₁ = {plain(s['e2_b1'], 4)}; kadın çalışanlarınki β̂₁ + γ̂₁ = "
            f"{total}. Etkileşim katsayısı "
            f"γ̂₁ = {plain(s['e2_g1'], 4)} kadın grubunun eğimi değil, iki eğim arasındaki farktır; kukla katsayısı "
            f"γ̂₀ = {plain(s['e2_g0'], 4)} da kadın grubunun sabiti değil, X = 0 noktasındaki farktır. Çarpım sütunu "
            "erkeklerde sıfır olduğu için etkileşim yalnız kadın grubunun eğimini değiştirir (§11.2).")


# --- Adım 3: dört olası regresyon yapısı ---------------------------------------------------------------------------

STRUCTURES = {
    "1": ("Aynı sabit, aynı eğim", ("educ12", *CONTROLS)),
    "2": ("Farklı sabit, aynı eğim", ("female", "educ12", *CONTROLS)),
    "3": ("Aynı sabit, farklı eğim", ("female:educ12", "educ12", *CONTROLS)),
    "4": ("Farklı sabit, farklı eğim", ("female", "educ12", "female:educ12", *CONTROLS)),
}
"""Tablo 11.2'nin dört yapısı, WAGE1'de: (1) 1, X; (2) 1, X, D; (3) 1, X, DX; (4) 1, X, D, DX (deneyim ve kıdem her
modelde). Yapı 3'te etkileşim formülde eğitimden önce yazılır: R terimi ``female:educ12`` olarak adlandırsın."""
STRUCTURE_3 = Choice("adim3_yapi", "Grafikte gösterilen yapı",
                     tuple((key, f"({key}) {title}") for key, (title, _) in STRUCTURES.items()), "4",
                     help="Dört model her seçimde tahmin edilir; seçim yalnız grafikteki doğruları değiştirir.")


def _structures(choices) -> tuple:
    chosen = choices["adim3_yapi"]
    title, terms = STRUCTURES[chosen]
    has_dummy, has_interaction = "female" in terms, "female:educ12" in terms
    models = tuple(OLS(f"m_yapi{key}", WAGE, "lwage", model_terms, f"Yapı ({key}), {name.lower()}: "
                       f"{_formula('lwage', model_terms)}") for key, (name, model_terms) in STRUCTURES.items())
    values: list = [  # sayılar tablonun sütununda görünür; ölçü kutusu olarak tekrar edilmez
        ModelValue("y3_b0", f"m_yapi{chosen}", "coef", "Sabit terim", term=INTERCEPT, decimals=4, shown=False),
        ModelValue("y3_b1", f"m_yapi{chosen}", "coef", "Eğitim eğimi β̂₁ (erkek; yapı 1–2'de iki grupta ortak)",
                   term="educ12", decimals=4, shown=False),
        *(ModelValue(f"y3_{term}", f"m_yapi{chosen}", "coef", f"{X_WORDS[term].capitalize()} katsayısı", term=term,
                     decimals=4, shown=False) for term in CONTROLS),
    ]
    values.append(ModelValue("y3_g0", f"m_yapi{chosen}", "coef", "Kukla katsayısı", term="female", decimals=4,
                             shown=False)
                  if has_dummy else Scalar("y3_g0", E.const(0), "Sabit farkı yok (γ₀ = 0)", decimals=0, shown=False))
    values.append(ModelValue("y3_g1", f"m_yapi{chosen}", "coef", "Etkileşim katsayısı", term="female:educ12",
                             decimals=4, shown=False)
                  if has_interaction else Scalar("y3_g1", E.const(0), "Eğim farkı yok (γ₁ = 0)", decimals=0,
                                                 shown=False))
    return (
        *models,
        RegressionTable(tuple((f"({key}) {name}", f"m_yapi{key}") for key, (name, _) in STRUCTURES.items()),
                        (INTERCEPT, "female", "educ12", "female:educ12", *CONTROLS), "tablo_yapi",
                        "Dört yapı WAGE1'de: bağımlı değişken ln(saatlik ücret), eğitim 12 yıl merkezli",
                        decimals=4, exact=True, adj_r2=True, r2_decimals=5),
        *values,
        *_averages(WAGE, "ort3", CONTROLS),
        *_group_lines("dogru3", "educ", (0, 18), "y3", "female", "educ12", 12, CONTROLS, "ort3"),
        LineChart("dogru3", "educ", "grup0", "Eğitim (yıl)", "Tahmin edilen ln(saatlik ücret)",
                  f"WAGE1, yapı ({chosen}): {title.lower()}", markers=False, series=(("grup1", "Kadın"),),
                  legend="Erkek"),
    )


def _structures_note(state, choices) -> str:
    chosen = choices["adim3_yapi"]
    table = state.tables["tablo_yapi"]
    headings = [f"({key}) {name}" for key, (name, _) in STRUCTURES.items()]
    best = max(headings, key=lambda heading: table.loc["adj_r2", heading])
    text = {
        "1": "Grup bilgisi modelde yoktur: iki grubun doğrusu aynıdır. ",
        "2": "Additif model: doğrular paraleldir, aralarındaki uzaklık kukla katsayısıdır. ",
        "3": ("Kukla ana etkisi yoktur: iki doğru eğitim 12 yılda (`educ12` = 0) aynı noktadan geçer. Hiyerarşi ilkesine "
              "aykırı bu model, 12 yıllık eğitimde ücret farkı olmadığını varsayar; bu güçlü ve veriyle çelişen bir "
              "kısıttır. "),
        "4": "Genel etkileşim modeli: hem başlangıç düzeyi hem eğim gruba göre değişebilir. ",
    }[chosen]
    return (text + f"Düzeltilmiş R² en yüksek olan yapı: {best} ({plain(table.loc['adj_r2', best], 5)}). Uyum ölçüsü "
            "tek başına yapı seçmez; etkileşimin ekonomik gerekçesi ve belirsizliği birlikte değerlendirilir (§11.3).")


# --- Adım 4: koşullu grup farkı ve merkezleme ----------------------------------------------------------------------

CENTER_4 = NumberChoice("adim4_c", "Merkezleme noktası c (eğitim yılı)", 0, 18, 12, 1, integer=True,
                        help="Notlarda c = 12. c = 0 merkezlenmemiş modeldir: kukla katsayısı sıfır yıllık eğitimdeki "
                             "farktır.")


def _centering(choices) -> tuple:
    center = int(choices["adim4_c"])
    terms = ("female", "educ_c", "female:educ_c", *CONTROLS)
    notes = center == 12
    side: tuple = () if notes else (  # notlardaki merkezleme (c = 12) seçilen noktayla yan yana
        OLS("m_merk12", WAGE, "lwage", _wage_terms("female"), "Notlardaki model, c = 12: lwage ~ female * educ12 + "
            "exper + tenure"),
        RegressionTable((("Notlar: c = 12", "m_merk12"), (f"Seçiminiz: c = {center}", "m_merk")),
                        (INTERCEPT, "female", "educ12", "educ_c", "female:educ12", "female:educ_c", *CONTROLS),
                        "yan114", "Aynı model, iki merkezleme noktası: yalnız sabit ve kukla katsayısı değişir",
                        decimals=4, exact=True, r2_decimals=6),
    )
    return (
        CopyFrame("merkez", WAGE, "Merkezleme için veri setinin kopyası (özgün veri değişmez)"),
        Derive("merkez", "educ_c", E.sub(E.var("educ"), center), f"Merkezlenmiş eğitim: educ − {center}"),
        OLS("m_merk", "merkez", "lwage", terms, f"Etkileşimli model, c = {center}: lwage ~ female * educ_c + exper + "
            "tenure"),
        CoefficientTable("m_merk", ("female", "female:educ_c"), "fark_c",
                         f"Kukla katsayısı: eğitim {center} yılda kadın–erkek log ücret farkı; etkileşim: eğim farkı",
                         decimals=4, t_decimals=3, p_decimals=4, exact=True),
        Count(WAGE, "n_c", "educ", center, f"Eğitimi {center} yıl olan çalışan sayısı"),
        ModelValue("g0_c", "m_merk", "coef", f"Eğitim {center} yılda log ücret farkı γ̂₀", term="female", decimals=4),
        ModelValue("g1_c", "m_merk", "coef", "Eğim farkı γ̂₁ (merkezlemeden bağımsız)", term="female:educ_c",
                   decimals=4),
        ModelValue("b1_c", "m_merk", "coef", "Erkek eğitim eğimi (merkezlemeden bağımsız)", term="educ_c", decimals=4),
        Scalar("tam_c", E.mul(100, E.sub(E.exp(E.ref("g0_c")), 1)), f"Eğitim {center} yılda tam yüzde fark",
               decimals=2, percent=True),
        ModelValue("r2_c", "m_merk", "r2", "R² (merkezlemeden bağımsız)", decimals=6),
        *side,
    )


def _centering_note(state, choices) -> str:
    center = int(choices["adim4_c"])
    s = state.scalars
    table = state.tables["fark_c"]
    p = float(table.loc["female", "p"])
    gap = (f"Eğitim {center} yılda kadın ve erkek çalışanlar arasındaki log ücret farkı {plain(s['g0_c'], 4)}; tam "
           f"yüzde fark: %{plain(s['tam_c'], 2)} ({_p(p, 4)}). ")
    count = int(s["n_c"])
    if count == 0:
        gap += (f"Örneklemde eğitimi {center} yıl olan çalışan yoktur: bu noktadaki fark tümüyle doğrusal model "
                "varsayımına dayanır. ")
    elif count < FEW_OBSERVATIONS:
        gap += (f"Örneklemde eğitimi {center} yıl olan yalnız {count} çalışan vardır: bu noktadaki fark büyük ölçüde "
                "doğrusal model varsayımına dayanır ve belirsizliği büyüktür. ")
    return (gap + f"Eğim farkı ({plain(s['g1_c'], 4)}), erkek eğitim eğimi ({plain(s['b1_c'], 4)}) ve R² "
            f"({plain(s['r2_c'], 6)}) c ne olursa olsun aynıdır: merkezleme tahmin edilen değerleri değiştirmez, yalnız "
            "sabitin ve kukla katsayısının hangi eğitim düzeyinde okunduğunu değiştirir. Kukla katsayısının standart "
            "hatası o noktadaki farkın standart hatasıdır (§11.4–11.5).")


# --- Adım 5: WAGE1 uygulaması ------------------------------------------------------------------------------------

DUMMY_5 = Choice("adim5_kukla", "Eğitimle etkileşen kukla", W.options(WAGE, DUMMIES), "female",
                 help="Notlarda kadın kuklası (Kod 11.2). Adım 6 aynı kuklayı kullanır.")


def _wage_rows(model: str, dummy: str, prefix: str) -> tuple:
    """Tablo 11.3: iki grubun eğitim eğimi, yaklaşık ve tam yüzde; ``prefix`` notlardaki model için "n"."""

    one, zero = DUMMY_WORDS[dummy]
    return (  # sayılar Tablo 11.3'te ve Kod 11.2 çıktısında görünür; ölçü kutusu olarak tekrar edilmez
        ModelValue(f"w{prefix}_b1", model, "coef", f"{zero.capitalize()}: eğitim eğimi β̂₁", term="educ12", decimals=4,
                   shown=False),
        ModelValue(f"w{prefix}_g1", model, "coef", "Eğim farkı γ̂₁", term=f"{dummy}:educ12", decimals=4, shown=False),
        Scalar(f"w{prefix}_egim1", E.add(E.ref(f"w{prefix}_b1"), E.ref(f"w{prefix}_g1")),
               f"{one.capitalize()}: eğitim eğimi β̂₁ + γ̂₁", decimals=4, shown=False),
        *(Scalar(f"w{prefix}_yak{group}", E.mul(100, E.ref(name)), f"{label}: yaklaşık yüzde 100·eğim", decimals=2,
                 shown=False)
          for group, name, label in ((0, f"w{prefix}_b1", zero.capitalize()), (1, f"w{prefix}_egim1", one.capitalize()))),
        *(Scalar(f"w{prefix}_tam{group}", E.mul(100, E.sub(E.exp(E.ref(name)), 1)),
                 f"{label}: tam yüzde 100·(exp(eğim) − 1)", decimals=2, shown=False)
          for group, name, label in ((0, f"w{prefix}_b1", zero.capitalize()), (1, f"w{prefix}_egim1", one.capitalize()))),
    )


def _wage_table(dummy: str, prefix: str, title: str) -> tuple:
    one, zero = DUMMY_WORDS[dummy]
    rows = ((zero.capitalize(), 0), (one.capitalize(), 1))
    return (
        ScalarTable(tuple((label, E.ref(f"w{prefix}_b1" if group == 0 else f"w{prefix}_egim1")) for label, group in rows),
                    f"egim113{prefix}", decimals=4, heading="Grup"),
        ScalarTable(tuple((label, E.ref(f"w{prefix}_yak{group}")) for label, group in rows), f"yak113{prefix}",
                    decimals=2, heading="Grup"),
        ScalarTable(tuple((label, E.ref(f"w{prefix}_tam{group}")) for label, group in rows), f"tam113{prefix}",
                    decimals=2, heading="Grup"),
        JoinColumns(f"tablo113{prefix}", (("Log eğim", f"egim113{prefix}", "deger"),
                                          ("Yaklaşık yüzde", f"yak113{prefix}", "deger"),
                                          ("Tam yüzde", f"tam113{prefix}", "deger")),
                    decimals=2, heading="Grup", column_decimals=(("Log eğim", 4),), title=title),
    )


def _wage_application(choices) -> tuple:
    dummy = choices["adim5_kukla"]
    one, zero = DUMMY_WORDS[dummy]
    notes = dummy == "female"
    terms = _wage_terms(dummy)
    side: tuple = () if notes else (  # notlardaki kadın–eğitim etkileşimi seçilen kuklayla yan yana
        OLS("m_w_n", WAGE, "lwage", _wage_terms("female"), "Notlardaki model (Kod 11.1): lwage ~ female * educ12 + "
            "exper + tenure"),
        *_wage_rows("m_w_n", "female", "n"),
        *_wage_table("female", "n", "Notlar — Tablo 11.3: cinsiyete göre eğitim eğimleri"),
    )
    return (
        OLS("m_w", WAGE, "lwage", terms, ("Kod 11.1: " if notes else "Etkileşimli model: ") + _formula("lwage", terms)),
        ShowModel("m_w", "Kod 11.2: WAGE1 etkileşim modelinin seçilmiş Python çıktısı" if notes
                  else "Etkileşimli modelin Python çıktısı (seçiminiz)", columns=("coef", "se", "t", "p"),
                  stats=("nobs", "r2", "adj_r2"), decimals=(("se", 4), ("p", 4)), exact=True),
        *_wage_rows("m_w", dummy, ""),
        *_wage_table(dummy, "", "Tablo 11.3: WAGE1 etkileşim modelinde cinsiyete göre eğitim eğimleri" if notes
                     else f"Seçiminiz: {one} ve {zero} çalışanların eğitim eğimleri"),
        *side,
        *_coefficients("m_w", "w5", dummy, "educ12", CONTROLS, shown=False),
        *_averages(WAGE, "ort5", CONTROLS),
        *_group_lines("cizgi5", "educ", (0, 18), "w5", dummy, "educ12", 12, CONTROLS, "ort5", log=True),
        LineChart("cizgi5", "educ", "ucret0", "Eğitim (yıl)", "Tahmin edilen saatlik ücret (dolar)",
                  ("Şekil 11.3: WAGE1 etkileşim modelinde cinsiyete göre tahmin edilen eğitim–ücret ilişkisi" if notes
                   else f"Etkileşimli model: {one} ve {zero} çalışanların tahmin edilen ücreti"),
                  markers=False, series=(("ucret1", one.capitalize()),), legend=zero.capitalize()),
    )


def _wage_note(state, choices) -> str:
    dummy = choices["adim5_kukla"]
    one, zero = DUMMY_WORDS[dummy]
    s = state.scalars
    p = float(state.models["m_w"].pvalues[f"{dummy}:educ12"])
    rejected = p < 0.05

    def change(value: float) -> str:
        return f"yüzde {plain(abs(value), 2)} daha {'yüksek' if value >= 0 else 'düşük'}"

    return (f"Deneyim ve kıdem sabitken bir yıl daha fazla eğitim, {zero} çalışanlarda yaklaşık {change(s['w_tam0'])}, "
            f"{one} çalışanlarda yaklaşık {change(s['w_tam1'])} tahmin edilen ücretle ilişkilidir (tam dönüşüm). "
            f"Etkileşim katsayısının p-değeri {plain(p, 4)}: yüzde 5 düzeyinde eğimlerin eşit olduğu hipotezi "
            + ("reddedilir. " if rejected else "reddedilemez; eğimler sayısal olarak farklı görünse de fark örnekleme "
               "belirsizliğine göre küçüktür. ")
            + "Şekildeki eğriler exp(tahmin edilen ln(ücret)) değerleridir; deneyim ve kıdem örneklem ortalamasındadır. "
            f"Gözlemsel veriyle bu katsayılar {CAUSAL_WORDS[dummy]} nedensel etkisini ölçmez (§11.6).")


# --- Adım 6: eğitim düzeyine göre grup farkı ---------------------------------------------------------------------

LEVEL_6 = NumberChoice("adim6_egitim", "Farkın hesaplandığı eğitim düzeyi (yıl)", 0, 18, 12, 1, integer=True,
                       help="Tablo 11.4 notlardaki dört düzeyi gösterir; ölçüler seçtiğiniz düzeydeki farktır. Notlarda "
                            "12 yıl (−25,72).")
GAP_LEVELS = (8, 12, 16, 18)
"""Tablo 11.4'ün eğitim düzeyleri."""


def _gap_scalars(level: int, prefix: str, source: str = "w6", shown: bool = False) -> tuple:
    """Eğitim ``level`` yılda log, yaklaşık ve tam yüzde fark; katsayılar ``{source}_g0`` ve ``{source}_g1``
    (yuvarlanmamış). Tablo 11.4'teki düzeyler tabloda görünür; ölçü kutusu yalnız seçilen düzey içindir."""

    log = E.add(E.ref(f"{source}_g0"), E.mul(E.ref(f"{source}_g1"), level - 12))
    return (
        Scalar(f"{prefix}_log", log, f"Eğitim {level} yıl: log fark γ̂₀ + γ̂₁·({level} − 12)", decimals=4, shown=shown),
        Scalar(f"{prefix}_yak", E.mul(100, E.ref(f"{prefix}_log")), f"Eğitim {level} yıl: yaklaşık yüzde", decimals=2,
               shown=shown),
        Scalar(f"{prefix}_tam", E.mul(100, E.sub(E.exp(E.ref(f"{prefix}_log")), 1)), f"Eğitim {level} yıl: tam yüzde",
               decimals=2, shown=shown),
    )


def _gap_table(source: str, suffix: str, title: str) -> tuple:
    """Tablo 11.4: dört eğitim düzeyinde log, yaklaşık ve tam yüzde fark; ``suffix`` notlardaki model için "n"."""

    rows = tuple(f"{value}" for value in GAP_LEVELS)
    return (
        *(item for value in GAP_LEVELS for item in _gap_scalars(value, f"fark{value}{suffix}", source)),
        *(ScalarTable(tuple((row, E.ref(f"fark{row}{suffix}_{part}")) for row in rows), f"{part}114{suffix}",
                      decimals=2, heading="Eğitim yılı") for part in ("log", "yak", "tam")),
        JoinColumns(f"tablo114{suffix}", (("Log fark", f"log114{suffix}", "deger"),
                                          ("Yaklaşık yüzde", f"yak114{suffix}", "deger"),
                                          ("Tam yüzde", f"tam114{suffix}", "deger")), decimals=2,
                    heading="Eğitim yılı", column_decimals=(("Log fark", 4),), title=title),
    )


def _gaps(choices) -> tuple:
    dummy, level = choices["adim5_kukla"], int(choices["adim6_egitim"])
    one, zero = DUMMY_WORDS[dummy]
    notes = dummy == "female"
    terms_add = (dummy, "educ12", *CONTROLS)
    side: tuple = () if notes else (  # notlardaki Tablo 11.4 (kadın kuklası, Adım 5'teki notlar modeli) yan yana
        ModelValue("w6n_g0", "m_w_n", "coef", "Notlar: kadın katsayısı γ̂₀", term="female", decimals=4, shown=False),
        ModelValue("w6n_g1", "m_w_n", "coef", "Notlar: etkileşim katsayısı γ̂₁", term="female:educ12", decimals=4,
                   shown=False),
        *_gap_table("w6n", "n", "Notlar — Tablo 11.4: WAGE1 modelinde eğitim düzeyine göre tahmin edilen kadın–erkek "
                    "ücret farkı"),
    )
    return (
        ModelValue("w6_g0", "m_w", "coef", "Kukla katsayısı γ̂₀ (12 yıllık eğitimde fark)", term=dummy, decimals=4),
        ModelValue("w6_g1", "m_w", "coef", "Etkileşim katsayısı γ̂₁", term=f"{dummy}:educ12", decimals=4),
        *_gap_table("w6", "", "Tablo 11.4: WAGE1 modelinde eğitim düzeyine göre tahmin edilen kadın–erkek ücret farkı"
                    if notes else f"Seçiminiz: {one} − {zero} ücret farkı, eğitim düzeyine göre"),
        *side,
        *_gap_scalars(level, "secilen", shown=True),
        Scalar("fark8_yuvarlak", E.add(E.roundto(E.ref("w6_g0"), 4), E.mul(E.roundto(E.ref("w6_g1"), 4), -4)),
               "Dört basamaklı katsayılarla 8 yılda log fark (Tablo 11.4 notu)", decimals=4, shown=False),
        Support("fark6", "educ", 0, 18, "Eğitim ızgarası: 0, 1, …, 18 yıl"),
        Derive("fark6", "log_fark", E.add(E.ref("w6_g0"), E.mul(E.ref("w6_g1"), E.sub(E.var("educ"), 12))),
               "Log fark: γ̂₀ + γ̂₁·(educ − 12)"),
        Derive("fark6", "tam_fark", E.mul(100, E.sub(E.exp(E.var("log_fark")), 1)), "Tam yüzde fark"),
        Scalar("sifir6", E.const(0), "Fark yok", decimals=0, shown=False),
        Scalar("merkez6", E.const(12), "Merkezleme noktası (12 yıl)", decimals=0, shown=False),
        LineChart("fark6", "educ", "tam_fark", "Eğitim (yıl)",
                  f"{one.capitalize()} − {zero} tahmin edilen ücret farkı (%)",
                  ("Şekil 11.4: WAGE1 modelinde kadın–erkek ücret farkının eğitim düzeyiyle değişimi" if notes
                   else f"{one.capitalize()} − {zero} ücret farkının eğitim düzeyiyle değişimi"),
                  references=(("sifir6", "Fark yok"),), markers=False, vlines=(("merkez6", "Merkezleme noktası (12 yıl)"),)),
        OLS("m_add6", WAGE, "lwage", terms_add, f"Additif model: {_formula('lwage', terms_add)}"),
        *(ModelValue(name, model, stat, label, decimals=5, shown=False)  # sayılar açıklama metninde
          for name, model, stat, label in (("r2_add6", "m_add6", "r2", "Additif model: R²"),
                                           ("r2_int6", "m_w", "r2", "Etkileşimli model: R²"),
                                           ("adj_add6", "m_add6", "adj_r2", "Additif model: düzeltilmiş R²"),
                                           ("adj_int6", "m_w", "adj_r2", "Etkileşimli model: düzeltilmiş R²"))),
    )


def _gap_direction(g0: float, g1: float) -> str:
    """Log farkın eğitimle (0–18 yıl) değişimi: işaret değiştirdiği nokta aralıktaysa o nokta yazılır."""

    rising = "artar" if g1 > 0 else "azalır"
    if g1 == 0:
        return "γ̂₁ = 0 olduğundan fark eğitimle değişmez. "
    zero = 12 - g0 / g1
    if 0 <= zero <= 18:
        return (f"γ̂₁ = {plain(g1, 4)} olduğundan log fark eğitim arttıkça {rising} ve yaklaşık {plain(zero, 1)} yılda "
                "işaret değiştirir: bu noktanın iki yanında farkın yönü terstir. ")
    grows = (g0 < 0) == (g1 < 0)  # 0–18 aralığında farkın işareti sabit; büyüklüğü aynı yönde değişir
    return (f"γ̂₁ = {plain(g1, 4)} olduğundan log fark eğitim arttıkça {rising}; 0–18 yıl aralığında işareti değişmez ve "
            f"büyüklüğü {'büyür' if grows else 'küçülür'}. ")


def _gaps_note(state, choices) -> str:
    dummy, level = choices["adim5_kukla"], int(choices["adim6_egitim"])
    one, zero = DUMMY_WORDS[dummy]
    s = state.scalars
    return (f"Eğitim {level} yılda log fark {plain(s['secilen_log'], 4)}; yaklaşık yüzde {plain(s['secilen_yak'], 2)}, "
            f"tam yüzde {plain(s['secilen_tam'], 2)}. Etkileşimli modelde {one}–{zero} log farkı eğitimle doğrusal "
            "değişir; tam yüzde fark üstel dönüşüm nedeniyle doğrusal değişmez. " + _gap_direction(s["w6_g0"], s["w6_g1"])
            + "Tablo 11.4 ve ölçüler yuvarlanmamış katsayılarla hesaplanır; dört basamaklı katsayılarla son basamak "
            f"farklı olabilir (notlardaki tablo notu). Etkileşim eklenince R²: {plain(s['r2_add6'], 5)} → "
            f"{plain(s['r2_int6'], 5)}; düzeltilmiş R²: {plain(s['adj_add6'], 5)} → {plain(s['adj_int6'], 5)}. Etkileşim "
            "istatistiksel olarak anlamlı değilse düzeylere göre değişen farkları kesin bir örüntü gibi sunmak doğru "
            "olmaz (§11.6).")


# --- Adım 7: HPRICE1 uygulaması --------------------------------------------------------------------------------------

X_7 = Choice("adim7_x", "Kolonyal tarzla etkileşen değişken",
             (("lotsize10k", "Arsa büyüklüğü (10.000 fit² merkezli, 1.000 fit²)"),
              ("sqrft100", "Konut büyüklüğü (100 fit²)"), ("bdrms", "Yatak odası sayısı")), "lotsize10k",
             help="Notlarda arsa büyüklüğü (Kod 11.4). Adım 8 aynı seçimi kullanır.")


def _house_terms(x: str) -> tuple[str, ...]:
    return ("colonial", x, f"colonial:{x}", *_others(x, HOUSE_X))


def _house(choices) -> tuple:
    x = choices["adim7_x"]
    notes = x == "lotsize10k"
    terms = _house_terms(x)
    word = X_WORDS[x]
    side: tuple = () if notes else (  # notlardaki kolonyal–arsa etkileşimi seçilen değişkenle yan yana
        OLS("m_h_n", HOUSE, "price", _house_terms("lotsize10k"), "Notlardaki model (Kod 11.3): price ~ colonial * "
            "lotsize10k + sqrft100 + bdrms"),
        RegressionTable((("Notlar: kolonyal × arsa", "m_h_n"), (f"Seçiminiz: kolonyal × {word}", "m_h")),
                        (INTERCEPT, "colonial", *HOUSE_X, "colonial:lotsize10k", f"colonial:{x}"), "yan117",
                        "Notlardaki model ve seçtiğiniz etkileşim · bağımlı değişken: fiyat (bin dolar)", decimals=4,
                        exact=True, adj_r2=True),
    )
    return (
        *_load_house(),
        OLS("m_h", HOUSE, "price", terms, ("Kod 11.3: " if notes else "Etkileşimli model: ") + _formula("price", terms)),
        ShowModel("m_h", "Kod 11.4: HPRICE1 etkileşim modelinin seçilmiş Python çıktısı" if notes
                  else "Etkileşimli modelin Python çıktısı (seçiminiz)", columns=("coef", "se", "t", "p"),
                  stats=("nobs", "r2", "adj_r2"), decimals=(("se", 4), ("p", 4)), exact=True),
        *side,
        ModelValue("h_b1", "m_h", "coef", f"Diğer tarz: {word} eğimi β̂₁", term=x, decimals=4),
        ModelValue("h_g1", "m_h", "coef", "Eğim farkı γ̂₁", term=f"colonial:{x}", decimals=4),
        ModelValue("h_g0", "m_h", "coef", "Kolonyal katsayısı γ̂₀", term="colonial", decimals=4),
        Scalar("h_egim1", E.add(E.ref("h_b1"), E.ref("h_g1")), f"Kolonyal: {word} eğimi β̂₁ + γ̂₁", decimals=4),
        Scalar("h_egim1_yuvarlak", E.add(E.roundto(E.ref("h_b1"), 4), E.roundto(E.ref("h_g1"), 4)),
               "Dört basamaklı katsayılarla (notlardaki gibi): β̂₁ + γ̂₁", decimals=4),
    )


def _house_note(state, choices) -> str:
    x = choices["adim7_x"]
    word, unit = X_WORDS[x], X_UNITS[x]
    s = state.scalars
    p = float(state.models["m_h"].pvalues[f"colonial:{x}"])
    change = "yüksektir" if s["h_b1"] > 0 else "düşüktür"
    text = (f"Kolonyal olmayan konutlarda {word} eğimi {plain(s['h_b1'], 4)}: diğer özellikler sabitken {word} bir "
            f"birim ({unit}) fazla olan konutun tahmin edilen fiyatı yaklaşık {plain(abs(s['h_b1']), 2)} bin dolar daha "
            f"{change}. Kolonyal konutların eğimi {_slope_sum(s['h_b1'], s['h_g1'], s['h_egim1'])}. Etkileşimin "
            f"p-değeri {plain(p, 4)}: yüzde 5 düzeyinde eğimlerin eşit olduğu hipotezi "
            + ("reddedilir. " if p < 0.05 else "reddedilemez. "))
    if x == "lotsize10k":
        text += ("Bu sonuç tek bir gözleme çok duyarlıdır: örneklemdeki en büyük arsa (92.681 fit²) kolonyal bir "
                 "konuta aittir; bu konut çıkarılıp model yeniden tahmin edilince etkileşim katsayısının işareti "
                 "değişir ve kolonyal konutların arsa eğimi daha yüksek çıkar. "
                 f"Kolonyal katsayısı ({plain(s['h_g0'], 4)}) yalnız 10.000 fit² arsada, diğer değişkenler sabitken iki "
                 "tarz arasındaki farktır; anlamsız olması etkileşimin de anlamsız olduğunu göstermez (§11.7).")
    else:
        text += (f"Kolonyal katsayısı ({plain(s['h_g0'], 4)}) {word} sıfırken iki tarz arasındaki farktır; örneklemde "
                 f"{word} {HOUSE_RANGES[x]} aralığındadır, bu nokta veri aralığının dışındadır ve katsayı tek başına "
                 "yorumlanmaz: ekstrapolasyondur. Anlamlı bir karşılaştırma noktası için değişken merkezlenir (§11.5, "
                 "§11.7).")
    return text


# --- Adım 8: arsa büyüklüğüne göre tarz farkı ---------------------------------------------------------------------

HOUSE_POINTS = {
    "lotsize10k": ((5000, -5), (7500, -2.5), (10000, 0), (12500, 2.5), (15000, 5), (20000, 10)),
    "sqrft100": ((1500, 15), (2000, 20), (2500, 25), (3000, 30), (3500, 35)),
    "bdrms": ((2, 2), (3, 3), (4, 4), (5, 5)),
}
"""Farkın hesaplandığı noktalar: (özgün ölçekte değer, modeldeki değer). Arsa için Tablo 11.5'in düzeyleri."""
HOUSE_GRIDS = {"lotsize10k": ("arsa_bin", (3, 20), 10, "Arsa büyüklüğü (bin fit²)"),
               "sqrft100": ("konut_yuz", (12, 38), 0, "Konut büyüklüğü (yüz fit²)"),
               "bdrms": ("oda", (2, 7), 0, "Yatak odası sayısı")}
"""Grafik ızgarası: (sütun, aralık, merkez, eksen adı). Arsada ızgara bin fit²; modeldeki değer ızgara − 10. Arsa
ızgarası notlardaki Şekil 11.5–11.6 gibi 3.000–20.000 fit²'dir: 88 konuttan 82'si bu aralıktadır (Tablo 11.5'in
bütün düzeyleri içinde)."""


def _point_label(x: str, value: float) -> str:
    return f"{value:,.0f}".replace(",", ".") if x != "bdrms" else f"{value:.0f}"


POINT_HEADINGS = {"lotsize10k": "Arsa büyüklüğü (fit²)", "sqrft100": "Konut büyüklüğü (fit²)",
                  "bdrms": "Yatak odası sayısı"}


def _gap_points(x: str, model: str, prefix: str, result: str, title: str) -> tuple:
    """Tablo 11.5: seçilen noktalarda kolonyal − diğer farkı γ̂₀ + γ̂₁·x (``model``in katsayılarıyla; ölçü kutusu yok,
    sayılar tabloda)."""

    word = X_WORDS[x]
    return (
        ModelValue(f"{prefix}_g0", model, "coef", "Kolonyal katsayısı γ̂₀", term="colonial", decimals=4, shown=False),
        ModelValue(f"{prefix}_g1", model, "coef", "Eğim farkı γ̂₁", term=f"colonial:{x}", decimals=4, shown=False),
        *(Scalar(f"{prefix}_{index}", E.add(E.ref(f"{prefix}_g0"), E.mul(E.ref(f"{prefix}_g1"), model_value)),
                 f"{word.capitalize()} {_point_label(x, value)}: γ̂₀ + γ̂₁·x", decimals=2, shown=False)
          for index, (value, model_value) in enumerate(HOUSE_POINTS[x])),
        ScalarTable(tuple((_point_label(x, value), E.ref(f"{prefix}_{index}"))
                          for index, (value, _) in enumerate(HOUSE_POINTS[x])),
                    result, decimals=2, heading=POINT_HEADINGS[x], value="Kolonyal − diğer farkı (bin dolar)",
                    title=title),
    )


def _house_gaps(choices) -> tuple:
    x = choices["adim7_x"]
    notes = x == "lotsize10k"
    word = X_WORDS[x]
    column, grid, center, axis = HOUSE_GRIDS[x]
    others = _others(x, HOUSE_X)
    terms_add = ("colonial", *HOUSE_X)
    line_values = tuple(  # doğruların katsayıları Kod 11.4 çıktısında görünür; ölçü kutusu olarak tekrar edilmez
        ModelValue(name, "m_h", "coef", label, term=term, decimals=4, shown=False)
        for name, label, term in (("h8_b0", "Sabit terim", INTERCEPT), ("h8_b1", f"{word.capitalize()} eğimi", x),
                                  ("h8_g0", "Kolonyal katsayısı", "colonial"), ("h8_g1", "Eğim farkı", f"colonial:{x}"),
                                  *((f"h8_{term}", f"{X_WORDS[term].capitalize()} katsayısı", term) for term in others)))
    counts: tuple = () if not notes else (  # §11.7: grafik aralığındaki ve üstündeki konut sayıları
        CopyFrame("arsa_sayim", HOUSE, "Arsa aralıklarındaki konutları saymak için veri setinin kopyası (özgün veri "
                  "değişmez)"),
        Derive("arsa_sayim", "arsa_3_20", E.mul(E.compare("ge", E.var("lotsize"), 3000),
                                                E.compare("le", E.var("lotsize"), 20000)),
               "Gösterge: arsa 3.000–20.000 fit² aralığında (1) ya da değil (0)"),
        Derive("arsa_sayim", "arsa_20_ustu", E.compare("gt", E.var("lotsize"), 20000),
               "Gösterge: arsa 20.000 fit²'den büyük (1) ya da değil (0)"),
        Count("arsa_sayim", "n_arsa_3_20", "arsa_3_20", 1, "Arsası 3.000–20.000 fit² olan konut"),
        Count("arsa_sayim", "n_arsa_20", "arsa_20_ustu", 1, "Arsası 20.000 fit²'den büyük konut"),
    )
    side: tuple = () if notes else _gap_points(  # notlardaki Tablo 11.5 (kolonyal × arsa, Adım 7'deki notlar modeli)
        "lotsize10k", "m_h_n", "hfarkn", "tablo115n",
        "Notlar — Tablo 11.5: HPRICE1 modelinde arsa büyüklüğüne göre kolonyal tarz fiyat farkı")
    return (
        *_gap_points(x, "m_h", "hfark", "tablo115",
                     "Tablo 11.5: HPRICE1 modelinde arsa büyüklüğüne göre kolonyal tarz fiyat farkı" if notes
                     else f"Seçiminiz: kolonyal tarz fiyat farkı, {word} düzeyine göre"),
        *side,
        *counts,
        *line_values,
        *_averages(HOUSE, "ort8", others),
        *_group_lines("cizgi8", column, grid, "h8", "colonial", x, center, others, "ort8"),
        LineChart("cizgi8", column, "grup0", axis, "Tahmin edilen konut fiyatı (bin dolar)",
                  ("Şekil 11.5: HPRICE1 modelinde mimari tarza göre arsa büyüklüğü eğimleri" if notes
                   else f"Mimari tarza göre {word} eğimleri"),
                  markers=False, series=(("grup1", "Kolonyal"),), legend="Kolonyal değil"),
        Derive("cizgi8", "tarz_farki", E.sub(E.var("grup1"), E.var("grup0")), "Kolonyal − diğer farkı"),
        Scalar("sifir8", E.const(0), "Fark yok", decimals=0, shown=False),
        Scalar("merkez8", E.const(center if center else grid[0]), "Merkezleme noktası", decimals=0, shown=False),
        LineChart("cizgi8", column, "tarz_farki", axis, "Kolonyal − diğer fiyat farkı (bin dolar)",
                  ("Şekil 11.6: HPRICE1 modelinde mimari tarz farkının arsa büyüklüğüyle değişimi" if notes
                   else f"Mimari tarz farkının {word} düzeyiyle değişimi"),
                  references=(("sifir8", "Fark yok"),), markers=False,
                  vlines=(("merkez8", "Merkezleme noktası (10.000 fit²)"),) if notes else ()),
        OLS("m_h_add", HOUSE, "price", terms_add, f"Additif model: {_formula('price', terms_add)}"),
        *(ModelValue(name, model, stat, label, decimals=4, shown=False)  # sayılar açıklama metninde
          for name, model, stat, label in (("r2_hadd", "m_h_add", "r2", "Additif model: R²"),
                                           ("r2_hint", "m_h", "r2", "Etkileşimli model: R²"),
                                           ("adj_hadd", "m_h_add", "adj_r2", "Additif model: düzeltilmiş R²"),
                                           ("adj_hint", "m_h", "adj_r2", "Etkileşimli model: düzeltilmiş R²"))),
    )


def _house_gaps_note(state, choices) -> str:
    x = choices["adim7_x"]
    s = state.scalars
    text = (f"Tarz farkı seçilen değişkenle doğrusal değişir: γ̂₀ + γ̂₁·x; tablodaki noktalar örneklemdeki aralığın "
            f"({HOUSE_RANGES[x]}) içindedir. "
            if x != "lotsize10k" else
            f"10.000 fit² arsada fark {plain(s['hfark_2'], 2)} bin dolar; arsa büyüdükçe fark küçülür ve işaret "
            f"değiştirdiği arsa büyüklüğü yaklaşık {_point_label(x, 10000 - 1000 * s['h_g0'] / s['h_g1'])} fit². "
            f"Grafikler 3.000–20.000 fit² aralığındadır: {plain(state.models['m_h'].nobs, 0)} konutun "
            f"{plain(s['n_arsa_3_20'], 0)} tanesi bu aralıktadır, 20.000 fit²'nin üstünde yalnız "
            f"{plain(s['n_arsa_20'], 0)} konut vardır. ")
    return (text + f"Etkileşim eklenince R²: {plain(s['r2_hadd'], 4)} → {plain(s['r2_hint'], 4)}; düzeltilmiş R²: "
            f"{plain(s['adj_hadd'], 4)} → {plain(s['adj_hint'], 4)}. Tablodaki farklar modelin tahminleridir; "
            "her noktadaki farkın standart hatası hesaplanmadan ‘anlamlı fark vardır’ denmez ve veri aralığının dışındaki "
            "noktalar ekstrapolasyon riski taşır (§11.7).")


# --- Adım 9: etkileşimli modellerde hipotez testleri ---------------------------------------------------------------

DATA_9 = Choice("adim9_veri", "Grafikteki uygulama", (("wage1", "WAGE1: kadın × eğitim"),
                                                     ("hprice1", "HPRICE1: kolonyal × arsa")), "wage1",
                help="Tablo her iki uygulamanın testlerini gösterir; seçim yalnız grafiği değiştirir.")
TEST_9 = Choice("adim9_test", "Grafikteki hipotez", (("egim", "Eğimler eşit: γ₁ = 0"),
                                                    ("ortak", "İki doğru aynı: γ₀ = γ₁ = 0")), "ortak",
                help="Tek kısıt t testiyle (F = t²), iki kısıt ortak F testiyle sınanır.")


def _tests(choices) -> tuple:
    data, test = choices["adim9_veri"], choices["adim9_test"]
    wage_terms, house_terms = _wage_terms("female"), _house_terms("lotsize10k")
    specs = (("w", "m_t_w", ("female", "female:educ12"), "WAGE1"),
             ("h", "m_t_h", ("colonial", "colonial:lotsize10k"), "HPRICE1"))
    operations: list = [
        OLS("m_t_w", WAGE, "lwage", wage_terms, f"WAGE1: {_formula('lwage', wage_terms)}"),
        OLS("m_t_h", HOUSE, "price", house_terms, f"HPRICE1: {_formula('price', house_terms)}"),
    ]
    for key, model, (dummy, interaction), name in specs:
        operations += [  # F ve p tabloda; ölçü kutuları t, serbestlik derecesi
            JointTest(f"F_egim_{key}", f"p_egim_{key}", model, (interaction,),
                      f"{name}: eğimler eşit, H₀: γ₁ = 0 (tek kısıt)", decimals=3, p_decimals=4, shown=False),
            ModelValue(f"t_egim_{key}", model, "t", f"{name}: etkileşimin t istatistiği", term=interaction, decimals=3),
            Scalar(f"t2_egim_{key}", E.power(E.ref(f"t_egim_{key}"), 2), f"{name}: t² (= F)", decimals=3, shown=False),
            JointTest(f"F_ortak_{key}", f"p_ortak_{key}", model, (dummy, interaction),
                      f"{name}: iki doğru aynı, H₀: γ₀ = 0, γ₁ = 0", decimals=3, p_decimals=4, shown=False),
            ModelValue(f"sd_{key}", model, "df_resid", f"{name}: payda serbestlik derecesi n − k − 1", decimals=0),
        ]
    rows = (("WAGE1 · H₀: γ₁ = 0", "egim_w"), ("WAGE1 · H₀: γ₀ = γ₁ = 0", "ortak_w"),
            ("HPRICE1 · H₀: γ₁ = 0", "egim_h"), ("HPRICE1 · H₀: γ₀ = γ₁ = 0", "ortak_h"))
    operations += [
        ScalarTable(tuple((label, E.ref(f"F_{name}")) for label, name in rows), "f118", decimals=3, heading="Test"),
        ScalarTable(tuple((label, E.ref(f"p_{name}")) for label, name in rows), "p118", decimals=4, heading="Test"),
        ScalarTable(tuple((label, E.const(1 if name.startswith("egim") else 2)) for label, name in rows), "q118",
                    decimals=0, heading="Test"),
        JoinColumns("tablo118", (("F", "f118", "deger"), ("p", "p118", "deger"), ("Kısıt sayısı q", "q118", "deger")),
                    decimals=3, heading="Test", column_decimals=(("Kısıt sayısı q", 0), ("p", 4)), p_columns=("p",),
                    title="Etkileşim testleri: eğimlerin eşitliği ve iki doğrunun bütünüyle aynı olması"),
    ]
    key = "w" if data == "wage1" else "h"
    name = "WAGE1" if key == "w" else "HPRICE1"
    if test == "egim":  # tek kısıt: t testi (F = t²)
        operations.append(HypothesisPlot("t", f"t_egim_{key}", f"sd_{key}", f"{name}: H₀: γ₁ = 0, t testi",
                                         "t değeri"))
    else:
        operations += [
            Scalar("q9", E.const(2), "Kısıt sayısı q", decimals=0, shown=False),
            HypothesisPlot("f", f"F_ortak_{key}", "q9", f"{name}: H₀: γ₀ = γ₁ = 0, ortak F testi", "F değeri",
                           alternative="sag", df2=f"sd_{key}"),
        ]
    return tuple(operations)


def _tests_note(state, choices) -> str:
    s = state.scalars
    return (f"WAGE1'de eğim eşitliği reddedilmez (F = {plain(s['F_egim_w'], 3)}, {_p(s['p_egim_w'])}), fakat iki "
            f"doğrunun bütünüyle aynı olduğu hipotezi güçlü biçimde reddedilir (F = {plain(s['F_ortak_w'], 3)}, "
            f"{_p(s['p_ortak_w'])}): 12 yıllık eğitimdeki düzey farkı büyüktür. HPRICE1'de tersi görülür: eğim farkı "
            f"yüzde 5 düzeyinde anlamlıdır (F = {plain(s['F_egim_h'], 3)}, {_p(s['p_egim_h'], 4)}), ortak test ise yüzde "
            f"5'te reddetmez, yüzde 10'da reddeder (F = {plain(s['F_ortak_h'], 3)}, {_p(s['p_ortak_h'], 4)}). Çelişki "
            "yoktur: tekli test yalnız γ₁'i, ortak test γ₀ ve γ₁'i birlikte sınar. Tek kısıtta F = t² "
            f"(WAGE1: ({plain(s['t_egim_w'], 3)})² = {plain(s['t2_egim_w'], 3)}). Test araştırma sorusuna göre "
            "seçilir (§11.8).")


# --- Adım 10: formül yazımı ve makale tablosu ------------------------------------------------------------------------

FORMULA_10 = Choice(
    "adim10_formul", "Üçüncü sütundaki formül",
    (("yok", "Üçüncü sütun yok (notlardaki tablo)"),
     ("acik", "female + educ12 + exper + tenure + female:educ12"),
     ("iki_nokta", "female:educ12 + exper + tenure (ana etkiler yok)")), "yok",
    help="female * educ12, ana etkileri ve etkileşimi birlikte ekler; female:educ12 yalnız çarpımı ekler.")
FORMULAS = {"acik": ("female", "educ12", *CONTROLS, "female:educ12"), "iki_nokta": ("female:educ12", *CONTROLS)}


def _paper(choices) -> tuple:
    variant = choices["adim10_formul"]
    models: tuple = (("(1) Additif model", "m_s1"), ("(2) Etkileşimli model", "m_s2"))
    extra: tuple = ()
    if variant != "yok":
        terms = FORMULAS[variant]
        extra = (OLS("m_s3", WAGE, "lwage", terms, f"Sütun (3): {_formula('lwage', terms)}"),)
        models += (("(3) Seçiminiz", "m_s3"),)
    return (
        OLS("m_s1", WAGE, "lwage", ("female", "educ12", *CONTROLS), "Sütun (1): lwage ~ female + educ12 + exper + "
            "tenure"),
        OLS("m_s2", WAGE, "lwage", _wage_terms("female"), "Sütun (2): lwage ~ female * educ12 + exper + tenure"),
        *extra,
        RegressionTable(models, ("female", "educ12", "female:educ12", *CONTROLS), "tablo116",
                        ("Tablo 11.6: etkileşim sonuçlarının makale tipi tabloda gösterimi" if variant == "yok"
                         else "Tablo 11.6 ve seçtiğiniz formül") + " · bağımlı değişken: ln(saatlik ücret)",
                        decimals=4, exact=True, adj_r2=True),
    )


def _paper_note(state, choices) -> str:
    variant = choices["adim10_formul"]
    table = state.tables["tablo116"]
    text = ("Tabloyu okuma sırası: referans grup erkek çalışanlardır ve eğitim 12 yıl etrafında merkezlidir; Sütun (2)'de "
            f"eğitim katsayısı ({plain(table.loc['educ12', '(2) Etkileşimli model'], 4)}) erkeklerin eğitim eğimi, "
            f"etkileşim ({plain(table.loc['female:educ12', '(2) Etkileşimli model'], 4)}) eğim farkıdır; kadınların "
            "eğimi ikisinin toplamıdır. Kadın katsayısı yalnız 12 yıllık eğitimdeki farktır. ")
    if variant == "acik":
        text += ("Sütun (3) terimleri açıkça yazar ve Sütun (2) ile aynı modeldir: `female * educ12`, `female + educ12 + "
                 "female:educ12` demektir. ")
    elif variant == "iki_nokta":
        text += ("Sütun (3)'te yalnız çarpım vardır: kadın ve eğitim ana etkileri modelden çıkar. Eğitim eğimi "
                 "erkeklerde sıfır, kadınlarda etkileşim katsayısı olmaya zorlanır ve iki grubun 12 yıllık eğitimdeki "
                 "farkı sıfır sayılır; hiyerarşi ilkesine aykırı bu kısıtlar katsayıların anlamını değiştirir. Sütun "
                 "(3)'teki etkileşim katsayısı artık eğim farkı değil, kadınların eğitim eğimidir; yıldızı bu yüzden "
                 "Sütun (2)'dekiyle karşılaştırılamaz. ")
    p = float(state.models["m_s2"].pvalues["female:educ12"])
    stars = ("yıldız olmaması eğim farkının yüzde 10 düzeyinde bile anlamlı olmadığını gösterir" if p >= 0.10
             else "yıldız eğim farkının anlamlı olduğunu gösterir")
    return text + f"Sütun (2)'de etkileşim satırında {stars} (§11.9)."


# --- Tanım ---------------------------------------------------------------------------------------------------------

_KOD112 = (
    (INTERCEPT, "Intercept", (1.5480, 0.0374, 41.346, 0.0)),
    ("female", "female", (-0.2973, 0.0380, -7.833, 0.0)),
    ("educ12", "educ12", (0.0903, 0.0087, 10.359, 0.0)),
    ("female:educ12", "female:educ12", (-0.0072, 0.0136, -0.534, 0.5935)),
    ("exper", "exper", (0.0046, 0.0016, 2.850, 0.0045)),
    ("tenure", "tenure", (0.0174, 0.0030, 5.849, 0.0)),
)
_KOD114 = (
    (INTERCEPT, "Intercept", (8.3369, 29.8451, 0.279, 0.7807)),
    ("colonial", "colonial", (8.2437, 14.4859, 0.569, 0.5709)),
    ("lotsize10k", "lotsize10k", (6.0025, 1.8396, 3.263, 0.0016)),
    ("colonial:lotsize10k", "colonial:lotsize10k", (-4.4064, 1.9407, -2.271, 0.0258)),
    ("sqrft100", "sqrft100", (11.8121, 1.3292, 8.886, 0.0)),
    ("bdrms", "bdrms", (12.5592, 9.3109, 1.349, 0.1811)),
)
_QUANTITIES = (("coef", "coef", 4), ("se", "std err", 4), ("t", "t", 3), ("p", "P>|t|", 4))
_TABLE113 = (("Erkek", (0.0903, 9.03, 9.45)), ("Kadın", (0.0830, 8.30, 8.66)))
_TABLE114 = (("8", (-0.2683, -26.83, -23.53)), ("12", (-0.2973, -29.73, -25.72)), ("16", (-0.3263, -32.63, -27.84)),
             ("18", (-0.3408, -34.08, -28.88)))
_TABLE115 = (("5.000", 30.28), ("7.500", 19.26), ("10.000", 8.24), ("12.500", -2.77), ("15.000", -13.79),
             ("20.000", -35.82))
TABLE116_WORDS = {"female": "kadın", "female_sh": "kadın (SH)", "educ12": "eğitim", "educ12_sh": "eğitim (SH)",
                  "female:educ12": "kadın × eğitim", "female:educ12_sh": "kadın × eğitim (SH)", "exper": "deneyim",
                  "exper_sh": "deneyim (SH)", "tenure": "kıdem", "tenure_sh": "kıdem (SH)", "n": "gözlem sayısı",
                  "r2": "R²", "adj_r2": "düzeltilmiş R²"}
"""Tablo 11.6 kontrol etiketleri: satır anahtarlarının Türkçe adları."""
_TABLE116 = (
    ("(1) Additif model", (("female", -0.3011), ("female_sh", 0.0372), ("educ12", 0.0875), ("educ12_sh", 0.0069),
                           ("exper", 0.0046), ("exper_sh", 0.0016), ("tenure", 0.0174), ("tenure_sh", 0.0030),
                           ("n", 526), ("r2", 0.3923), ("adj_r2", 0.3876))),
    ("(2) Etkileşimli model", (("female", -0.2973), ("female_sh", 0.0380), ("educ12", 0.0903), ("educ12_sh", 0.0087),
                               ("female:educ12", -0.0072), ("female:educ12_sh", 0.0136), ("exper", 0.0046),
                               ("exper_sh", 0.0016), ("tenure", 0.0174), ("tenure_sh", 0.0030), ("n", 526),
                               ("r2", 0.3926), ("adj_r2", 0.3868))),
)


def _output_checks(model: str, rows, source: str) -> tuple[Check, ...]:
    return tuple(Check(f"{source}: {name}, {label}", CoefTarget(model, term, quantity), value, 3 if quantity == "t" else 4)
                 for term, name, values in rows for (quantity, label, _), value in zip(_QUANTITIES, values))


STEPS = (
    interactive_step(
        number=1,
        title="Additif kukla modeli: paralel doğrular",
        note=NoteRef("11.1", 0, ("Şekil 11.1", "Tablo 11.6 (§11.9)")),
        explanation=(
            "Additif modelde $Y_i = \\beta_0 + \\beta_1 X_i + \\gamma_0 D_i + u_i$: iki grubun sabiti farklı olabilir, "
            "eğimi aynıdır. $\\mathbb{E}(Y \\mid X, D = 1) - \\mathbb{E}(Y \\mid X, D = 0) = \\gamma_0$ her $X$ "
            "düzeyinde aynıdır ve tahmin doğruları paraleldir. WAGE1'de ln(ücret), 12 yıl etrafında merkezlenmiş eğitim, "
            "deneyim ve kıdemle açıklanır. Kukla değişkeni değiştirin."
        ),
        controls=(DUMMY_1,),
        build=_additive,
        checks=(
            *(Check(f"Tablo 11.6 (1): {label}", CoefTarget("m_add", term, quantity), value, 4)
              for term, label, quantity, value in (
                  ("female", "kadın", "coef", -0.3011), ("female", "kadın (SH)", "se", 0.0372),
                  ("educ12", "eğitim", "coef", 0.0875), ("educ12", "eğitim (SH)", "se", 0.0069),
                  ("exper", "deneyim", "coef", 0.0046), ("exper", "deneyim (SH)", "se", 0.0016),
                  ("tenure", "kıdem", "coef", 0.0174), ("tenure", "kıdem (SH)", "se", 0.0030))),
            Check("Tablo 11.6 (1): gözlem sayısı", ModelTarget("m_add", "nobs"), 526, 0),
            Check("§11.6: additif modelin R²'si", ModelTarget("m_add", "r2"), 0.39227, 5),
            Check("§11.6: additif modelin düzeltilmiş R²'si", ModelTarget("m_add", "adj_r2"), 0.38760, 5),
        ),
        note_for=lambda state, choices: _additive_note(state, choices),
    ),
    interactive_step(
        number=2,
        title="Etkileşim terimi ve grup denklemleri",
        note=NoteRef("11.2", 0, ("Tablo 11.1",)),
        explanation=(
            "Kukla ile nicel değişkenin çarpımı modele eklenir: $Y_i = \\beta_0 + \\beta_1 X_i + \\gamma_0 D_i + "
            "\\gamma_1 (D_i X_i) + u_i$. $D_i X_i$ erkekte (D = 0) sıfır, kadında (D = 1) $X_i$'dir. Böylece $D = 0$ "
            "grubunun denklemi $\\beta_0 + \\beta_1 X$, $D = 1$ grubununki $(\\beta_0 + \\gamma_0) + (\\beta_1 + "
            "\\gamma_1) X$ olur: $\\gamma_0$ sabit farkı, $\\gamma_1$ eğim farkıdır. Etkileşimdeki nicel değişkeni "
            "değiştirin."
        ),
        controls=(X_2,),
        build=_interaction,
        checks=(
            _scalar("e2_b0", 1.5480, "Kod 11.2: erkek grubunun sabiti β̂₀", 4),
            _scalar("e2_g0", -0.2973, "Kod 11.2: sabit farkı γ̂₀", 4),
            _scalar("e2_b1", 0.0903, "Tablo 11.3: erkek eğitim eğimi β̂₁", 4),
            _scalar("e2_g1", -0.0072, "Kod 11.2: eğim farkı γ̂₁", 4),
            _scalar("e2_egim1", 0.0830, "Tablo 11.3: kadın eğitim eğimi β̂₁ + γ̂₁", 4),
            _scalar("e2_egim1_yuvarlak", 0.0831, "Tablo 11.3 notu: 0,0903 − 0,0072", 4),
        ),
        note_for=lambda state, choices: _interaction_note(state, choices),
    ),
    interactive_step(
        number=3,
        title="Gruplar arasında dört olası regresyon yapısı",
        note=NoteRef("11.3", 0, ("Tablo 11.2", "Şekil 11.2")),
        explanation=(
            "İki grup ve bir nicel değişkenle dört yapı kurulabilir: (1) aynı sabit, aynı eğim: $1, X$; (2) farklı sabit, "
            "aynı eğim: $1, X, D$; (3) aynı sabit, farklı eğim: $1, X, DX$; (4) farklı sabit, farklı eğim: "
            "$1, X, D, DX$. Hiyerarşi ilkesi: $DX$ modeldeyse $D$ ve $X$ ana etkileri de tutulur; (3) $D$'yi "
            "çıkararak iki grubun $X = 0$ noktasında aynı sabite sahip olduğunu dayatır. Grafikteki yapıyı değiştirin."
        ),
        controls=(STRUCTURE_3,),
        build=_structures,
        checks=(
            Check("§11.6: additif modelin R²'si (yapı 2)", TableTarget("tablo_yapi", "r2", "(2) Farklı sabit, aynı eğim"),
                  0.39227, 5),
            Check("§11.6: etkileşimli modelin R²'si (yapı 4)",
                  TableTarget("tablo_yapi", "r2", "(4) Farklı sabit, farklı eğim"), 0.39260, 5),
            Check("§11.6: additif modelin düzeltilmiş R²'si", TableTarget("tablo_yapi", "adj_r2",
                                                                         "(2) Farklı sabit, aynı eğim"), 0.38760, 5),
            Check("§11.6: etkileşimli modelin düzeltilmiş R²'si",
                  TableTarget("tablo_yapi", "adj_r2", "(4) Farklı sabit, farklı eğim"), 0.38676, 5),
        ),
        note_for=lambda state, choices: _structures_note(state, choices),
    ),
    interactive_step(
        number=4,
        title="Koşullu grup farkı ve merkezleme",
        note=NoteRef("11.4", 0, ("§11.5",)),
        explanation=(
            "Etkileşimli modelde grup farkı $X$'e bağlıdır: $\\mathbb{E}(Y \\mid X, D = 1) - \\mathbb{E}(Y \\mid X, "
            "D = 0) = \\gamma_0 + \\gamma_1 X$ (§11.4). $\\gamma_0$ yalnız $X = 0$ noktasındaki farktır. $X$ anlamlı bir "
            "$c$ değeri etrafında merkezlenirse ($X^c = X - c$) kukla katsayısı farkı doğrudan $X = c$ noktasında verir; "
            "tahmin edilen değerler, R² ve eğimler değişmez. Merkezleme noktasını değiştirin."
        ),
        controls=(CENTER_4,),
        build=_centering,
        checks=(
            _scalar("g0_c", -0.2973, "Kod 11.2: 12 yıllık eğitimde log ücret farkı", 4),
            _scalar("tam_c", -25.72, "§11.6: 100(e^(−0,2973) − 1)", 2),
            _scalar("g1_c", -0.0072, "Kod 11.2: eğim farkı", 4),
            _scalar("r2_c", 0.393, "Kod 11.2: R²", 3),
        ),
        note_for=lambda state, choices: _centering_note(state, choices),
    ),
    interactive_step(
        number=5,
        title="WAGE1: eğitim eğimi gruplar arasında değişiyor mu?",
        note=NoteRef("11.6", 0, ("Kod 11.1", "Kod 11.2", "Tablo 11.3", "Şekil 11.3")),
        explanation=(
            "Notlardaki model: $\\ln(\\text{ücret}_i) = \\beta_0 + \\gamma_0 \\text{female}_i + \\beta_1 \\text{educ12}_i + \\gamma_1 "
            "(\\text{female}_i \\times \\text{educ12}_i) + \\beta_2 \\text{exper}_i + \\beta_3 \\text{tenure}_i + u_i$. "
            "Erkek çalışanlar referans gruptur. Erkeklerin eğitim eğimi $\\beta_1$, kadınlarınki $\\beta_1 + "
            "\\gamma_1$; log eğim tam yüzdeye $100(e^{b} - 1)$ ile çevrilir. Eğitimle etkileşen kuklayı değiştirin: "
            "notlardaki model yan yana gösterilir."
        ),
        controls=(DUMMY_5,),
        build=_wage_application,
        checks=(
            *_output_checks("m_w", _KOD112, "Kod 11.2"),
            Check("Kod 11.2: No. Observations", ModelTarget("m_w", "nobs"), 526, 0),
            Check("Kod 11.2: R-squared", ModelTarget("m_w", "r2"), 0.393, 3),
            Check("Kod 11.2: Adj. R-squared", ModelTarget("m_w", "adj_r2"), 0.387, 3),
            *(Check(f"Tablo 11.3: {group}, {column}", TableTarget("tablo113", group, column), value,
                    4 if column == "Log eğim" else 2)
              for group, values in _TABLE113
              for column, value in zip(("Log eğim", "Yaklaşık yüzde", "Tam yüzde"), values)),
        ),
        note_for=lambda state, choices: _wage_note(state, choices),
    ),
    interactive_step(
        number=6,
        title="Grup farkı eğitim düzeyine bağlıdır",
        note=NoteRef("11.6", 0, ("Tablo 11.4", "Şekil 11.4")),
        explanation=(
            "Notlardaki modelde kadın–erkek log ücret farkı $\\hat\\gamma_0 + \\hat\\gamma_1 \\cdot \\text{educ12}$'dir; tam yüzde fark "
            "$100[\\exp(\\hat\\gamma_0 + \\hat\\gamma_1 \\cdot \\text{educ12}) - 1]$ (§11.4). Eğitim 12 yılda "
            "$\\text{educ12} = 0$ olduğundan kadın katsayısı doğrudan bu noktadaki farktır. Etkileşimin uyuma katkısı "
            "additif modelle karşılaştırılır. Adım 5'teki kukla seçimi bu adımı da belirler. Farkın hesaplandığı eğitim "
            "düzeyini değiştirin."
        ),
        controls=(LEVEL_6,),
        uses=(DUMMY_5,),
        build=_gaps,
        checks=(
            *(Check(f"Tablo 11.4: {row} yıl, {column}", TableTarget("tablo114", row, column), value,
                    4 if column == "Log fark" else 2)
              for row, values in _TABLE114 for column, value in zip(("Log fark", "Yaklaşık yüzde", "Tam yüzde"), values)),
            _scalar("secilen_tam", -25.72, "§11.6: 12 yıllık eğitimde tam yüzde fark", 2),
            _scalar("fark8_yuvarlak", -0.2685, "Tablo 11.4 notu: 8 yılda −0,2973 − 0,0072·(−4)", 4),
            _scalar("r2_add6", 0.39227, "§11.6: additif modelin R²'si", 5),
            _scalar("r2_int6", 0.39260, "§11.6: etkileşimli modelin R²'si", 5),
            _scalar("adj_add6", 0.38760, "§11.6: additif modelin düzeltilmiş R²'si", 5),
            _scalar("adj_int6", 0.38676, "§11.6: etkileşimli modelin düzeltilmiş R²'si", 5),
        ),
        note_for=lambda state, choices: _gaps_note(state, choices),
    ),
    interactive_step(
        number=7,
        title="HPRICE1: eğim mimari tarza göre değişiyor mu?",
        note=NoteRef("11.7", 0, ("Kod 11.3", "Kod 11.4")),
        explanation=(
            "Notlardaki model: $\\text{price}_i = \\beta_0 + \\gamma_0 \\text{colonial}_i + \\beta_1 \\text{lotsize10k}_i + \\gamma_1 "
            "(\\text{colonial}_i \\times \\text{lotsize10k}_i) + \\beta_2 \\text{sqrft100}_i + \\beta_3 \\text{bdrms}_i "
            "+ u_i$. Fiyat bin dolar; arsa büyüklüğü 10.000 fit² etrafında merkezli ve 1.000 fit² biriminde, konut "
            "büyüklüğü 100 fit² biriminde. Kolonyal olmayan konutlar referans gruptur. Kolonyal tarzla etkileşen "
            "değişkeni değiştirin: notlardaki model yan yana gösterilir."
        ),
        controls=(X_7,),
        build=_house,
        checks=(
            *_output_checks("m_h", _KOD114, "Kod 11.4"),
            Check("Kod 11.4: No. Observations", ModelTarget("m_h", "nobs"), 88, 0),
            Check("Kod 11.4: R-squared", ModelTarget("m_h", "r2"), 0.695, 3),
            Check("Kod 11.4: Adj. R-squared", ModelTarget("m_h", "adj_r2"), 0.676, 3),
            _scalar("h_egim1_yuvarlak", 1.5961, "§11.7: kolonyal eğim 6,0025 − 4,4064", 4),
            _scalar("h_egim1", 1.5960, "§11.7: kolonyal eğim, yuvarlanmamış katsayılarla", 4),
        ),
        note_for=lambda state, choices: _house_note(state, choices),
    ),
    interactive_step(
        number=8,
        title="Mimari tarz farkı etkileşen değişkene bağlıdır",
        note=NoteRef("11.7", 0, ("Tablo 11.5", "Şekil 11.5", "Şekil 11.6")),
        explanation=(
            "Notlardaki modelde kolonyal ve diğer konutlar arasındaki fark $\\hat\\gamma_0 + \\hat\\gamma_1 \\cdot "
            "\\text{lotsize10k}$ ile arsa büyüklüğüne göre değişir; kolonyal katsayısı yalnız 10.000 fit²'deki farktır. "
            "Farklar modelin tahminleridir: her noktadaki farkın standart hatası ayrıca hesaplanmadan ‘anlamlı fark "
            "vardır’ denmez; veri aralığının dışı ekstrapolasyondur. Adım 7'deki seçim bu adımı da belirler; notlardaki "
            "Tablo 11.5 yan yana gösterilir."
        ),
        uses=(X_7,),
        build=_house_gaps,
        checks=(
            *(Check(f"Tablo 11.5: {row} fit²", TableTarget("tablo115", row, "deger"), value, 2)
              for row, value in _TABLE115),
            _scalar("n_arsa_3_20", 82, "§11.7: arsası 3.000–20.000 fit² olan konut sayısı", 0),
            _scalar("n_arsa_20", 4, "§11.7: arsası 20.000 fit²'den büyük konut sayısı", 0),
            _scalar("r2_hadd", 0.6758, "§11.7: additif modelin R²'si", 4),
            _scalar("r2_hint", 0.6950, "§11.7: etkileşimli modelin R²'si", 4),
            _scalar("adj_hadd", 0.6602, "§11.7: additif modelin düzeltilmiş R²'si", 4),
            _scalar("adj_hint", 0.6764, "§11.7: etkileşimli modelin düzeltilmiş R²'si", 4),
        ),
        note_for=lambda state, choices: _house_gaps_note(state, choices),
    ),
    interactive_step(
        number=9,
        title="Etkileşimli modellerde hipotez testleri",
        note=NoteRef("11.8", 0),
        explanation=(
            "Araştırma sorusuna göre üç hipotez ayrılır: eğimler eşit mi ($H_0: \\gamma_1 = 0$; tek kısıt, $t$ ya da "
            "$F = t^2$), seçilen referans noktasında fark var mı ($H_0: \\gamma_0 = 0$), iki grubun regresyon doğrusu "
            "bütünüyle aynı mı ($H_0: \\gamma_0 = 0,\\ \\gamma_1 = 0$; iki kısıt, ortak $F$ testi). Grafikteki "
            "uygulamayı ve testi değiştirin."
        ),
        controls=(DATA_9, TEST_9),
        build=_tests,
        checks=(
            _scalar("F_egim_w", 0.285, "§11.8: WAGE1 eğim testi F", 3),
            _scalar("p_egim_w", 0.593, "§11.8: WAGE1 eğim testi p", 3),
            _scalar("F_egim_h", 5.155, "§11.8: HPRICE1 eğim testi F", 3),
            _scalar("p_egim_h", 0.0258, "§11.8: HPRICE1 eğim testi p", 4),
            _scalar("F_ortak_w", 32.785, "§11.8: WAGE1 ortak test F", 3),
            _scalar("p_ortak_w", 0.0, "§11.8: WAGE1 ortak test p < 0,001", 3),
            _scalar("F_ortak_h", 3.039, "§11.8: HPRICE1 ortak test F", 3),
            _scalar("p_ortak_h", 0.0533, "§11.8: HPRICE1 ortak test p", 4),
        ),
        note_for=lambda state, choices: _tests_note(state, choices),
    ),
    interactive_step(
        number=10,
        title="Formül yazımı ve makale tablosunda etkileşim",
        note=NoteRef("11.9", 0, ("Tablo 11.6",)),
        explanation=(
            "statsmodels formülünde `D * X` terimleri $1 + D + X + D{:}X$ olarak açar; `D:X` yalnız çarpımı, `D + X` "
            "etkileşimsiz additif modeli kurar. Makale tablosu okunurken önce referans grup ve merkezleme noktası "
            "belirlenir; ana $X$ katsayısı referans grubun eğimi, etkileşim eğim farkı, kukla katsayısı merkezleme "
            "noktasındaki farktır. Üçüncü sütunun formülünü değiştirin."
        ),
        controls=(FORMULA_10,),
        build=_paper,
        checks=tuple(
            Check(f"Tablo 11.6: {heading}, {TABLE116_WORDS[row]}", TableTarget("tablo116", row, heading), value,
                  0 if row == "n" else 4)
            for heading, values in _TABLE116 for row, value in values
        ),
        note_for=lambda state, choices: _paper_note(state, choices),
    ),
)


def _interaction_labels() -> tuple[tuple[str, str], ...]:
    """Etkileşim terimlerinin kısa adları (seçimlerdeki bütün bileşimler)."""

    short = {"female": "Kadın", "married": "Evli", "nonwhite": "Beyaz olmayan", "colonial": "Kolonyal"}
    words = {"educ12": "eğitim", "educ_c": "merkezlenmiş eğitim", "exper": "deneyim", "tenure": "kıdem",
             "lotsize10k": "arsa", "sqrft100": "konut büyüklüğü", "bdrms": "yatak odası"}
    pairs = [(dummy, x) for dummy in DUMMIES for x in ("educ12", "educ_c", "exper", "tenure")]
    pairs += [("colonial", x) for x in HOUSE_X]
    return tuple((f"{dummy}:{x}", f"{short[dummy]} × {words[x]}") for dummy, x in pairs)


KONU11_LAB = LabSpec(
    topic_key="konu11",
    title="Uygulama: Etkileşim Terimleri ve Gruplar Arasında Sabit ile Eğim Farklılıkları",
    note_section="11",
    steps=STEPS,
    labels=(
        *W.labels(WAGE, HOUSE),
        (INTERCEPT, "Sabit terim"),
        ("educ12", "Eğitim − 12 (yıl)"),
        ("educ_c", "Eğitim − c (yıl)"),
        ("lotsize10k", "(Arsa − 10.000)/1.000 (bin fit²)"),
        ("sqrft100", "Konut büyüklüğü/100 (yüz fit²)"),
        ("arsa_3_20", "Arsa 3.000–20.000 fit² (gösterge)"),
        ("arsa_20_ustu", "Arsa > 20.000 fit² (gösterge)"),
        ("kadin_x", "Kadın × X (çarpım)"),
        *_interaction_labels(),
        ("grup0", "Referans grup: tahmin"),
        ("grup1", "Karşılaştırma grubu: tahmin"),
        ("arsa_bin", "Arsa büyüklüğü (bin fit²)"),
        ("konut_yuz", "Konut büyüklüğü (yüz fit²)"),
        ("oda", "Yatak odası sayısı"),
    ),
    consistency_notes=(
        "Veri notlardaki gibi WAGE1 ve HPRICE1'dir. Notlardaki Kod 11.1 ve Kod 11.3, bölüm betiği, uygulama ve üretilen "
        "kod veriyi wooldridge paketinden okur. Önceki notlarda Kod 11.1 veriyi data/wage1.csv kopyasından okuyordu, "
        "Kod 11.3'te veri okuma satırı yoktu (house tablosu hazır varsayılıyordu); Bölüm 11 betiği de data/ altındaki "
        "CSV kopyalarını okuyordu. Basılı sayıların hiçbiri değişmedi.",
        "§11.7'de kolonyal eğim 6,0025 − 4,4064 = 1,5961 yuvarlanmış katsayılarla hesaplanır; yuvarlanmamış katsayılarla "
        "1,5960'tır. Notlara bu açıklama eklendi; uygulama iki sayıyı da gösterir.",
        "Tablo 11.3 ve 11.4 yuvarlanmamış katsayılarla hesaplanmıştır; metindeki dört basamaklı katsayılarla son basamak "
        "farklı olabilir (ör. kadınların eğitim eğimi 0,0903 − 0,0072 = 0,0831, tabloda 0,0830). Notlara iki tablo için "
        "bu açıklamayı veren tablo notu eklendi (Konu 9'daki ve §11.7'deki örnekle aynı); uygulama toplamları ekrandaki "
        "terimlerle yazar, yuvarlanmamış değeri parantezde verir.",
        "Şekil 11.5 ve 11.6'nın arsa ızgarası notlarda ve uygulamada 3.000–20.000 fit²'dir (88 konuttan 82'si bu "
        "aralıkta; Tablo 11.5'in bütün düzeyleri içinde). Notlardaki iki şekil önceden farklı aralıklar kullanıyordu "
        "(yaklaşık 3.300–17.800 ve 3.000–30.000 fit²); basılı sayıların hiçbiri değişmedi.",
        "Etkileşim terimi statsmodels ve R'de aynı adı taşır (female:educ12). Ana etkiler ve etkileşim notlardaki gibi "
        "female * educ12 formülüyle kurulur; R etkileşimleri çıktıda ana etkilerden sonra listeler.",
        "Standart hatalar klasik EKK standart hatalarıdır; heteroskedastisiteye dayanıklı çıkarım Konu 12'de.",
    ),
)
