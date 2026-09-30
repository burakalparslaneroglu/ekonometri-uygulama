"""Konu 12 uygulaması: heteroskedastisite ve heteroskedastisiteye dayanıklı çıkarım (gerçek veri).

Bölüm 12'nin çözümlü örnekleri bölüm sırasıyla: HPRICE1 düzey ve log modellerinin artık grafikleri (§12.4, Şekil
12.3–12.5), Breusch–Pagan ve White testleri (§12.5, Tablo 12.2), HC0–HC3 dayanıklı kovaryans seçenekleri (§12.6, Tablo 12.3),
geleneksel ve HC1 çıkarımı: düzey modeli, güven aralıkları, log modeli ve makale tablosu (§12.7, Şekil 12.6, Tablo 12.4),
dayanıklı ortak test (§12.8, Tablo 12.5) ve WAGE1 düzey–log karşılaştırması (§12.10). Heteroskedastik benzetim
(§12.9, Tablo 12.6, Şekil 12.1–12.2 ve 12.7–12.8) Sezgi sekmesinin Deney 1'idir. Her ``Check`` notlarda basılı bir
sayıdır; değer notlardan kopyalanmıştır.

Dayanıklı kovaryans statsmodels'te ``fit(cov_type="HC1", use_t=True)`` ile (t ve F dağılımı, n − k − 1 serbestlik
derecesi, k sabit dışındaki açıklayıcı sayısı; notlardaki ``get_robustcov_results`` ile aynı sayılar), R'de sandviç
formülüyle açıkça hesaplanır. Notlardan farklı bir seçimde notlardaki model (HC1, notlardaki açıklayıcılar ve sınanan
katsayılar) yan yana gösterilir.
Etkileşim: incelenen model (Adım 1–3), yardımcı regresyon (Adım 2), dayanıklı kovaryans türü (Adım 4; Adım 5–7 aynı
türü kullanır), ortak testte sınanan katsayılar (Adım 7) ve WAGE1 modellerinin açıklayıcıları (Adım 8).
"""

from __future__ import annotations

import math

from core import wooldridge_data as W
from core.labs import expr as E
from core.labs.regression import stars
from core.labs.sezgi import plain
from core.labs.spec import (
    HETERO_TESTS,
    INTERCEPT,
    OLS,
    Check,
    Choice,
    CoefficientPlot,
    CoefficientTable,
    CopyFrame,
    CoefTarget,
    Derive,
    GroupSummary,
    HeteroskedasticityTest,
    HypothesisPlot,
    JoinColumns,
    JointTest,
    LabSpec,
    LoadWooldridge,
    ModelTarget,
    ModelValue,
    MultiChoice,
    NoteRef,
    Percentile,
    RegressionTable,
    Residuals,
    Scalar,
    ScalarTable,
    ScalarTarget,
    ScatterPlot,
    ShowModel,
    TableTarget,
    interactive_step,
)

HOUSE = "hprice1"
WAGE = "wage1"
LEVEL_X = ("lotsize1000", "sqrft100", "bdrms")
LOG_X = ("llotsize", "lsqrft", "bdrms")
MODELS = {
    "duzey": ("m_duzey", "price", LEVEL_X, "Fiyat düzeyi", "Tahmin edilen konut fiyatı (bin dolar)"),
    "log": ("m_log", "lprice", LOG_X, "Log fiyat", "Tahmin edilen log konut fiyatı"),
}
"""Bölüm 12'nin iki HPRICE1 modeli: (model adı, bağımlı değişken, açıklayıcılar, kısa ad, tahmin ekseninin adı)."""
MODEL_OPTIONS = (("duzey", "Fiyat düzeyi: price ~ lotsize1000 + sqrft100 + bdrms (notlar)"),
                 ("log", "Log fiyat: lprice ~ llotsize + lsqrft + bdrms"))
HC_TYPES = ("HC0", "HC1", "HC2", "HC3")
TERM_WORDS = {"lotsize1000": "arsa büyüklüğü", "sqrft100": "konut büyüklüğü", "bdrms": "yatak odası sayısı",
              "llotsize": "log arsa büyüklüğü", "lsqrft": "log konut büyüklüğü", INTERCEPT: "sabit terim"}
PLOT_LABELS = (("lotsize1000", "Arsa / 1.000"), ("sqrft100", "Konut / 100"), ("bdrms", "Yatak odası"),
               ("llotsize", "log(arsa)"), ("lsqrft", "log(konut)"))
RESIDUAL_UNITS = {"duzey": " (bin dolar)", "log": " (log birimi)"}
"""Artık eksenlerinin birimi: düzey modelinde fiyat bin dolar, log modelde log fiyat."""
LEVERAGE = {"duzey": 0.843, "log": 0.319}
"""Arsası 92.681 fit² olan konutun kaldıracı (iki modelde de en büyük kaldıraç; ortalama kaldıraç 4/88 ≈ 0,045);
testle doğrulanır."""
NOTES_JOINT = ("lotsize1000", "bdrms")
"""Tablo 12.5'te birlikte sınanan katsayılar."""
WAGE_X = ("educ", "exper", "tenure", "female", "married", "nonwhite")
WAGE_DEFAULT = ("educ", "exper", "tenure", "female")
WAGE_DUMMIES = ("female", "married", "nonwhite")
"""0/1 kuklalar: White testinde kareleri kendileridir (tek başına seçilince White = BP)."""
"""§12.10'un WAGE1 modelleri: ücret (düzey) ve log ücret, eğitim, deneyim, kıdem ve kadın kuklasıyla."""


def _scalar(name: str, expected: float, label: str, decimals: int) -> Check:
    return Check(label, ScalarTarget(name), expected, decimals)


def _p(value: float, decimals: int = 4) -> str:
    if value < 0.5 * 10 ** -decimals:
        return "p < " + plain(10 ** -decimals, decimals)
    return f"p = {plain(value, decimals)}"


def _formula(outcome: str, terms: tuple[str, ...]) -> str:
    return f"{outcome} ~ " + " + ".join(terms)


def _load_house() -> tuple:
    return (
        LoadWooldridge(HOUSE, "HPRICE1 veri seti (Wooldridge, 2020): 88 konut"),
        Derive(HOUSE, "lotsize1000", E.div(E.var("lotsize"), 1000), "Arsa büyüklüğü 1.000 fit² biriminde: lotsize/1000"),
        Derive(HOUSE, "sqrft100", E.div(E.var("sqrft"), 100), "Konut büyüklüğü 100 fit² biriminde: sqrft/100"),
    )


def _models() -> tuple:
    return tuple(OLS(name, HOUSE, outcome, terms, f"{title} modeli: {_formula(outcome, terms)}")
                 for name, outcome, terms, title, _ in MODELS.values())


# --- Adım 1: artık grafikleri ------------------------------------------------------------------------------------

MODEL_1 = Choice("adim1_model", "İncelenen model", MODEL_OPTIONS, "duzey",
                 help="Notlarda önce düzey modeli (Şekil 12.3–12.4), sonra log model (Şekil 12.5).")
QUARTER_LABELS = ((1, "1. çeyrek (en düşük tahmin)"), (2, "2. çeyrek"), (3, "3. çeyrek"), (4, "4. çeyrek (en yüksek)"))


def _residual_plots(choices) -> tuple:
    kind = choices["adim1_model"]
    name, outcome, terms, title, axis = MODELS[kind]
    level = kind == "duzey"
    return (
        *_load_house(),
        *_models(),
        ShowModel(name, f"{title} modelinin katsayıları", columns=("coef",), stats=("nobs", "r2"), exact=True),
        CopyFrame("tani", HOUSE, "Tanı için veri setinin kopyası (özgün veri değişmez)"),
        Residuals("tani", "artik", name, f"{title} modelinin artıkları û"),
        Derive("tani", "tahmin", E.sub(E.var(outcome), E.var("artik")), "Tahmin edilen değer: Ŷ = Y − û"),
        Derive("tani", "mutlak", E.absolute(E.var("artik")), "Mutlak artık |û|"),
        ScatterPlot("tani", "tahmin", "artik", axis, "Artık" + RESIDUAL_UNITS[kind],
                    ("Şekil 12.3: HPRICE1 düzey modelinde artıklar ve tahmin edilen fiyatlar" if level
                     else "Şekil 12.5: HPRICE1 log modelinde artıklar ve tahmin edilen log fiyatlar"),
                    lines=((0.0, 0.0, "Sıfır çizgisi"),), opacity=0.8),
        ScatterPlot("tani", "tahmin", "mutlak", axis, "Mutlak artık" + RESIDUAL_UNITS[kind],
                    ("Şekil 12.4: HPRICE1 düzey modelinde mutlak artıklar ve tahmin edilen fiyatlar" if level
                     else "Log modelde mutlak artıklar ve tahmin edilen log fiyatlar"), opacity=0.8),
        *(Percentile("tani", "tahmin", p, f"q{p}", f"Tahmin edilen değerin {p}. yüzdeliği", decimals=3)
          for p in (25, 50, 75)),
        Derive("tani", "ceyrek", E.add(E.add(E.add(1, E.compare("ge", E.var("tahmin"), E.ref("q25"))),
                                              E.compare("ge", E.var("tahmin"), E.ref("q50"))),
                                        E.compare("ge", E.var("tahmin"), E.ref("q75"))),
               "Tahmin edilen değerin çeyreği: 1 (en düşük) … 4 (en yüksek)"),
        GroupSummary("tani", "ceyrek", (("n", "artik", "count"), ("ss", "artik", "std"), ("ort_mutlak", "mutlak", "mean")),
                     "yayilim", (1, 2, 3, 4), decimals=3 if level else 4, labels=QUARTER_LABELS,
                     heading="Tahmin edilen değer",
                     title="Artıkların yayılımı tahmin edilen değerin çeyreklerine göre (standart sapma ve ortalama "
                           "mutlak artık)"),
    )


def _residual_note(state, choices) -> str:
    kind = choices["adim1_model"]
    table = state.tables["yayilim"]
    mean = table["ort_mutlak"].to_numpy(dtype=float)
    spread = table["ss"].to_numpy(dtype=float)
    decimals = 2 if kind == "duzey" else 4
    text = (f"Ortalama mutlak artık ilk üç çeyrekte {plain(mean[:3].min(), decimals)} ile {plain(mean[:3].max(), decimals)} "
            f"arasında, en yüksek tahmin çeyreğinde {plain(mean[3], decimals)}; artıkların standart sapması en yüksek "
            f"çeyrekte {plain(spread[3], decimals)}, diğer çeyreklerde {plain(spread[:3].min(), decimals)}–"
            f"{plain(spread[:3].max(), decimals)}. ")
    if kind == "duzey":
        text += ("Düşük ve orta tahmin düzeylerinde artıklar dar bir alanda toplanırken yüksek tahmin edilen fiyatlarda "
                 "çok daha büyük pozitif ve negatif artıklar görülür: huni biçimi heteroskedastisite şüphesini güçlendirir. ")
    else:
        text += ("Log modelde yayılım düzey modeline göre daha dengelidir; log dönüşümü ölçek etkisini azaltabilir ama "
                 "katsayıların yorumunu da değiştirir (esneklik). ")
    return (text + "Grafik kesin bir test değildir: az sayıdaki uç gözlem ya da doğrusal olmayan bir koşullu ortalama da "
            "benzer iz bırakabilir (§12.4).")


# --- Adım 2: Breusch–Pagan ve White testleri -----------------------------------------------------------------------

TEST_2 = Choice("adim2_test", "Yardımcı regresyon",
                (("bp", "Breusch–Pagan: açıklayıcıların düzeyleri (notlar)"),
                 ("white", "White: düzeyler, kareler ve çapraz çarpımlar")), "bp",
                help="Yardımcı regresyonun bağımlı değişkeni artıkların karesidir; LM = n·R².")
MODEL_2 = Choice("adim2_model", "Model", MODEL_OPTIONS, "duzey", help="Notlarda düzey modeli (§12.5).")


def _white_terms(frame: str, terms: tuple[str, ...]) -> tuple[tuple, tuple[str, ...]]:
    """White yardımcı regresyonunun kare ve çapraz çarpım sütunları ile bütün terimleri."""

    derived, names = [], list(terms)
    for index, first in enumerate(terms):
        for second in terms[index:]:
            name = f"{first}_kare" if first == second else f"{first}_x_{second}"
            derived.append(Derive(frame, name, E.mul(E.var(first), E.var(second)),
                                  f"{first}²" if first == second else f"{first} × {second}"))
            names.append(name)
    return tuple(derived), tuple(names)


def _tests(choices) -> tuple:
    test, kind = choices["adim2_test"], choices["adim2_model"]
    name, _, terms, title, _ = MODELS[kind]
    white, aux_terms = _white_terms("yardimci", terms) if test == "white" else ((), terms)
    library: list = []
    for model_kind in ("duzey", "log"):
        model = MODELS[model_kind][0]
        for test_kind in ("bp", "white"):
            library.append(HeteroskedasticityTest(f"lm_{test_kind}_{model_kind}", f"p_{test_kind}_{model_kind}", model,
                                                  test_kind, f"{MODELS[model_kind][3]}: {HETERO_TESTS[test_kind]} testi",
                                                  shown=False))  # sayılar Tablo 12.2'de
    rows = tuple((MODELS[model_kind][3], model_kind) for model_kind in ("duzey", "log"))
    return (
        CopyFrame("yardimci", HOUSE, "Yardımcı regresyon için veri setinin kopyası (özgün veri değişmez)"),
        Residuals("yardimci", "artik", name, f"{title} modelinin artıkları û"),
        Derive("yardimci", "artik2", E.power(E.var("artik"), 2), "Kareli artık û²: hata varyansının gözlenebilir ölçüsü"),
        *white,
        OLS("m_yardimci", "yardimci", "artik2", aux_terms,
            f"{HETERO_TESTS[test]} yardımcı regresyonu: {_formula('artik2', aux_terms)}"),
        ModelValue("r2_yardimci", "m_yardimci", "r2", "Yardımcı regresyonun R²'si", decimals=6),
        ModelValue("n_yardimci", "m_yardimci", "nobs", "Gözlem sayısı n", decimals=0),
        Scalar("lm_elle", E.mul(E.ref("n_yardimci"), E.ref("r2_yardimci")), "LM = n·R² (elle)", decimals=3),
        Scalar("q_yardimci", E.const(len(aux_terms)), "Serbestlik derecesi q (sabit dışındaki terim sayısı)", decimals=0),
        Scalar("p_elle", E.chi2sf(E.ref("lm_elle"), E.ref("q_yardimci")), "p = P(χ²_q > LM)", decimals=4,
               p_value=True),
        HeteroskedasticityTest("lm_kutuphane", "p_kutuphane", name, test,
                               f"Tek çağrıyla (`{'het_breuschpagan' if test == 'bp' else 'white_testi'}`)"),
        *library,
        *(ScalarTable(tuple((label, E.ref(f"{quantity}_{test_kind}_{model_kind}")) for label, model_kind in rows),
                      f"{quantity}_{test_kind}_122", decimals=3, heading="Model")
          for quantity in ("lm", "p") for test_kind in ("bp", "white")),
        JoinColumns("tablo122", (("BP LM", "lm_bp_122", "deger"), ("BP p", "p_bp_122", "deger"),
                                 ("White LM", "lm_white_122", "deger"), ("White p", "p_white_122", "deger")),
                    decimals=3, heading="Model", p_columns=("BP p", "White p"),
                    column_decimals=(("BP p", 4), ("White p", 4)),
                    title="Tablo 12.2: HPRICE1 düzey ve log modellerinde heteroskedastisite testleri"),
    )


def _tests_note(state, choices) -> str:
    test, kind = choices["adim2_test"], choices["adim2_model"]
    s = state.scalars
    rejected = s["p_elle"] < 0.05
    product = s["n_yardimci"] * float(f"{s['r2_yardimci']:.6f}")  # ekrandaki R² ile çarpım
    sign = "=" if plain(product, 3) == plain(s["lm_elle"], 3) else "≈"
    return (f"{HETERO_TESTS[test]} ({MODELS[kind][3].lower()} modeli): yardımcı regresyonun R²'si "
            f"{plain(s['r2_yardimci'], 6)}, LM = n·R² = "
            f"{plain(s['n_yardimci'], 0)} × {plain(s['r2_yardimci'], 6)} {sign} {plain(s['lm_elle'], 3)}, "
            f"q = {plain(s['q_yardimci'], 0)} (yardımcı regresyondaki sabit dışındaki terim sayısı), "
            f"{_p(s['p_elle'])}; tek çağrı ({'statsmodels `het_breuschpagan`' if test == 'bp' else '`white_testi`'}) "
            f"aynı sayıyı verir ({plain(s['lm_kutuphane'], 3)}). Yüzde 5 düzeyinde homoskedastisite hipotezi "
            + ("reddedilir. " if rejected else "reddedilemez; reddedememek homoskedastisitenin kanıtlandığı anlamına "
               "gelmez, testin gücü sınırlı olabilir. ")
            + "White testi yanlış fonksiyonel biçime de duyarlıdır ve daha çok serbestlik derecesi tüketir. Test sonucu "
            "dayanıklı standart hata kullanmanın ön koşulu değildir (§12.5).")


# --- Adım 3: HC0–HC3 ---------------------------------------------------------------------------------------------

MODEL_3 = Choice("adim3_model", "Model", MODEL_OPTIONS, "duzey",
                 help="Beş sütunda aynı EKK katsayıları, farklı kovaryans hesabı.")


def _hc_table(choices) -> tuple:
    kind = choices["adim3_model"]
    name, outcome, terms, title, _ = MODELS[kind]
    robust = tuple(OLS(f"m3_{cov.lower()}", HOUSE, outcome, terms, f"{title}, {cov} kovaryansıyla", cov_type=cov)
                   for cov in HC_TYPES)
    return (
        *robust,
        RegressionTable((("(1) Geleneksel", name), *((f"({index}) {cov}", f"m3_{cov.lower()}")
                                                     for index, cov in enumerate(HC_TYPES, start=2))),
                        (*terms, INTERCEPT), "tablo_hc",
                        f"{title} modeli: aynı katsayılar, beş kovaryans hesabı", decimals=4, exact=True),
        *(ModelValue(f"sh3_{cov.lower()}", f"m3_{cov.lower()}", "se", f"{cov} SH ({TERM_WORDS[terms[0]]})",
                     term=terms[0], decimals=4, shown=False) for cov in HC_TYPES),  # sayılar tabloda
        ModelValue("sh3_gel", name, "se", f"Geleneksel SH ({TERM_WORDS[terms[0]]})", term=terms[0], decimals=4,
                   shown=False),
    )


def _hc_note(state, choices) -> str:
    kind = choices["adim3_model"]
    name, _, terms, _, _ = MODELS[kind]
    word = TERM_WORDS[terms[0]]
    s = state.scalars
    model = state.models[name]
    n, df = int(model.nobs), int(model.df_resid)
    scaled = float(f"{s['sh3_hc0']:.5f}") * math.sqrt(n / df)  # ekrandaki HC0 ile
    sign = "=" if plain(scaled, 4) == plain(s["sh3_hc1"], 4) else "≈"
    text = (f"Katsayılar beş sütunda aynıdır; yalnız parantez içindeki standart hatalar ve yıldızlar değişir. "
            f"{word.capitalize()} katsayısının standart hatası: geleneksel {plain(s['sh3_gel'], 4)}, HC0 "
            f"{plain(s['sh3_hc0'], 4)}, HC1 {plain(s['sh3_hc1'], 4)}, HC2 {plain(s['sh3_hc2'], 4)}, HC3 "
            f"{plain(s['sh3_hc3'], 4)}. HC1, HC0 varyansını n/(n − k − 1) = {n}/{df} ile çarpar (k: sabit dışındaki "
            f"açıklayıcı sayısı); standart hata bunun kareköküyle büyür: {plain(s['sh3_hc0'], 5)} × √({n}/{df}) {sign} "
            f"{plain(s['sh3_hc1'], 4)}. ")
    if kind == "duzey":
        text += ("HC2 ve HC3'teki büyük artış yüksek kaldıraçlı bir gözlemden gelir: arsası 92.681 fit² olan konutun "
                 f"kaldıracı yaklaşık {plain(LEVERAGE['duzey'], 2)} (ortalama kaldıraç (k + 1)/n = 4/88 ≈ 0,045). HC3 bu "
                 "gözlemin kareli artığını (1 − h)² ≈ 0,025 ile bölerek yaklaşık 40 kat büyütür; arsa katsayısının HC3 "
                 "varyansının yaklaşık %99'u bu tek gözlemden gelir. ")
    else:
        text += (f"Log modelde aynı konutun kaldıracı yaklaşık {plain(LEVERAGE['log'], 2)}: log dönüşümü uç arsanın "
                 "öteki gözlemlerden uzaklığını azaltır ve dört dayanıklı standart hata birbirine daha yakın kalır. ")
    return (text + "HC3 en koruyucudur. Notlar HC1 kullanır; küçük örneklemde ya da yüksek kaldıraçta HC3 duyarlılık "
            "kontrolüdür. Tabloda hangi kovaryansın kullanıldığı mutlaka yazılır (§12.6).")


# --- Adım 4: HPRICE1 düzey modeli, geleneksel ve dayanıklı ---------------------------------------------------------

HC_4 = Choice("adim4_hc", "Dayanıklı kovaryans türü", tuple((cov, f"{cov} (notlar)" if cov == "HC1" else cov)
                                                         for cov in HC_TYPES), "HC1",
              help="Notlarda HC1. Adım 5–7 aynı türü kullanır.")


def _comparison(model: str, robust: str, terms: tuple[str, ...], cov: str, result: str, title: str,
                notes_robust: str | None = None) -> tuple:
    """Notlardaki çıktı düzeni: katsayı, geleneksel SH ve p, dayanıklı SH ve p. ``notes_robust``: HC1 dışında bir tür
    seçildiyse notlardaki HC1 modeli; HC1 sütunları seçilen türün solunda yan yana gösterilir."""

    order = (INTERCEPT, *terms)
    tables = [CoefficientTable(model, order, f"{result}_gel", "Geleneksel standart hatayla çıkarım", decimals=4,
                               t_decimals=3, p_decimals=4, exact=True)]
    columns = [("coef", f"{result}_gel", "katsayi"), ("OLS se", f"{result}_gel", "sh"), ("OLS p", f"{result}_gel", "p")]
    p_columns = ["OLS p"]
    if notes_robust is not None:
        tables.append(CoefficientTable(notes_robust, order, f"{result}_n", "Notlardaki HC1 dayanıklı standart hatayla "
                                       "çıkarım", decimals=4, t_decimals=3, p_decimals=4, exact=True))
        columns += [("HC1 se (notlar)", f"{result}_n", "sh"), ("HC1 p (notlar)", f"{result}_n", "p")]
        p_columns.append("HC1 p (notlar)")
    tables.append(CoefficientTable(robust, order, f"{result}_rob", f"{cov} dayanıklı standart hatayla çıkarım",
                                   decimals=4, t_decimals=3, p_decimals=4, exact=True))
    columns += [(f"{cov} se", f"{result}_rob", "sh"), (f"{cov} p", f"{result}_rob", "p")]
    p_columns.append(f"{cov} p")
    return (
        *tables,
        JoinColumns(result, tuple(columns), decimals=4, heading="Terim", p_columns=tuple(p_columns),
                    column_decimals=tuple((name, 4) for name in p_columns), title=title, term_rows=True),
    )


def _notes_robust(cov: str, outcome: str, terms: tuple[str, ...], name: str, title: str) -> tuple:
    """HC1 dışında bir tür seçildiyse notlardaki HC1 modeli (aynı EKK tahmini) yan yana gösterim için."""

    return () if cov == "HC1" else (OLS(name, HOUSE, outcome, terms, f"{title}, notlardaki HC1 kovaryansıyla",
                                        cov_type="HC1"),)


def _level(choices) -> tuple:
    cov = choices["adim4_hc"]
    notes = cov == "HC1"
    values = (("sh4_gel", "m_duzey", "se", "Arsa: geleneksel SH", "lotsize1000"),
              ("sh4_rob", "m_rob", "se", f"Arsa: {cov} SH", "lotsize1000"),
              ("p4_gel", "m_duzey", "p", "Arsa: geleneksel p", "lotsize1000"),
              ("p4_rob", "m_rob", "p", f"Arsa: {cov} p", "lotsize1000"),
              ("alt4_rob", "m_rob", "ci_low", f"Arsa: {cov} GA alt sınırı", "lotsize1000"),
              ("p4_konut_rob", "m_rob", "p", f"Konut büyüklüğü: {cov} p", "sqrft100"),
              ("sh4_oda_gel", "m_duzey", "se", "Yatak odası: geleneksel SH", "bdrms"),
              ("sh4_oda_rob", "m_rob", "se", f"Yatak odası: {cov} SH", "bdrms"),
              *(() if notes else (("sh4_n", "m_rob_n", "se", "Arsa: HC1 SH (notlar)", "lotsize1000"),
                                  ("p4_n", "m_rob_n", "p", "Arsa: HC1 p (notlar)", "lotsize1000"))))
    return (
        OLS("m_rob", HOUSE, "price", LEVEL_X, f"Aynı düzey modeli, {cov} dayanıklı kovaryansla", cov_type=cov),
        *_notes_robust(cov, "price", LEVEL_X, "m_rob_n", "Aynı düzey modeli"),
        *_comparison("m_duzey", "m_rob", LEVEL_X, cov, "kod123",
                     "Kod 12.3: HPRICE1 düzey modeli, geleneksel ve HC1 sonuçlarının seçilmiş kısmı" if notes
                     else f"HPRICE1 düzey modeli: geleneksel, notlardaki HC1 ve seçtiğiniz {cov}",
                     None if notes else "m_rob_n"),
        CoefficientPlot("m_duzey", LEVEL_X, "Şekil 12.6: HPRICE1 düzey modelinde geleneksel ve HC1 yüzde 95 güven "
                        "aralıkları" if notes else f"Geleneksel, HC1 (notlar) ve {cov} yüzde 95 güven aralıkları",
                        "Katsayı (bin dolar) ve yüzde 95 güven aralığı", y_label="Değişken", labels=PLOT_LABELS,
                        compare=((f"{cov} dayanıklı", "m_rob"),) + (() if notes else (("HC1 dayanıklı (notlar)",
                                                                                         "m_rob_n"),)),
                        legend="Geleneksel"),
        *(ModelValue(name, model, quantity, label, term=term, decimals=4, shown=False)  # sayılar tabloda
          for name, model, quantity, label, term in values),
    )


def _level_note(state, choices) -> str:
    cov = choices["adim4_hc"]
    s = state.scalars
    covered = s["alt4_rob"] < 0
    rooms = "düşer" if s["sh4_oda_rob"] < s["sh4_oda_gel"] else "yükselir"
    house = ("yüzde 1 düzeyinde anlamlı kalır" if s["p4_konut_rob"] < 0.01 else
             f"anlamlılığını kısmen yitirir ({_p(s['p4_konut_rob'])})")
    every = "iki hesapta da" if cov == "HC1" else "bütün hesaplarda"
    text = (f"Arsa büyüklüğü katsayısı {every} {plain(state.models['m_duzey'].params['lotsize1000'], 4)}: diğer "
            "değişkenler sabitken arsadaki 1.000 fit² artış tahmin edilen fiyatta yaklaşık 2,07 bin dolarlık artışla "
            f"ilişkilidir. Değişen standart hatadır: geleneksel hesapla {plain(s['sh4_gel'], 4)} ({_p(s['p4_gel'])}), "
            f"{cov} hesabıyla {plain(s['sh4_rob'], 4)} ({_p(s['p4_rob'])})"
            + ("" if cov == "HC1" else f"; notlardaki HC1 hesabıyla {plain(s['sh4_n'], 4)} ({_p(s['p4_n'])})") + ". "
            + (f"{cov} güven aralığı sıfırı içerir, geleneksel aralık içermez. " if covered else "")
            + f"Katsayı ve ekonomik büyüklük değişmez; değişen belirsizlik ölçüsüdür. Konut büyüklüğü {house}; yatak "
            f"odası katsayısında standart hata {rooms} ({plain(s['sh4_oda_gel'], 4)} → {plain(s['sh4_oda_rob'], 4)}): "
            "heteroskedastisite geleneksel standart hatayı her katsayıda aynı yönde etkilemez (§12.7).")
    return text


# --- Adım 5: log model ----------------------------------------------------------------------------------------------

def _log(choices) -> tuple:
    cov = choices["adim4_hc"]
    notes = cov == "HC1"
    return (
        OLS("m_log_rob", HOUSE, "lprice", LOG_X, f"Aynı log modeli, {cov} dayanıklı kovaryansla", cov_type=cov),
        *_notes_robust(cov, "lprice", LOG_X, "m_log_rob_n", "Aynı log modeli"),
        *_comparison("m_log", "m_log_rob", LOG_X, cov, "kod124",
                     "Kod 12.4: HPRICE1 log modeli, geleneksel ve HC1 sonuçlarının seçilmiş kısmı" if notes
                     else f"HPRICE1 log modeli: geleneksel, notlardaki HC1 ve seçtiğiniz {cov}",
                     None if notes else "m_log_rob_n"),
        ModelValue("sh5_gel", "m_log", "se", "Log arsa: geleneksel SH", term="llotsize", decimals=4, shown=False),
        ModelValue("sh5_rob", "m_log_rob", "se", f"Log arsa: {cov} SH", term="llotsize", decimals=4, shown=False),
        ModelValue("e5", "m_log", "coef", "Arsa büyüklüğü esnekliği", term="llotsize", decimals=4, shown=False),
    )


CLOSE_RATIO = 1.15
"""Dayanıklı/geleneksel SH oranı bundan küçükse iki standart hata "birbirine yakın" sayılır (notlarda log modelde
HC1 ile 1,08)."""


def _log_note(state, choices) -> str:
    cov = choices["adim4_hc"]
    s = state.scalars
    ratio = s["sh5_rob"] / s["sh5_gel"]
    level = s["sh4_rob"] / s["sh4_gel"]
    if ratio < CLOSE_RATIO:
        closeness = "birbirine yakındır"
    elif ratio < level:
        closeness = "düzey modeline göre birbirine çok daha yakındır"
    else:
        closeness = "birbirinden uzaktır"
    return (f"Arsa büyüklüğü katsayısı esnekliktir: arsadaki yüzde 1 artış, diğer değişkenler sabitken fiyatta yaklaşık "
            f"yüzde {plain(s['e5'], 3)} artışla ilişkilidir. {cov} ve geleneksel standart hatalar {closeness} (arsa "
            f"katsayısında {cov}/geleneksel SH oranı {plain(ratio, 2)}; düzey modelinde {plain(level, 2)}). Bu, log modelde testlerin reddetmemesiyle "
            "uyumludur; reddetmemek homoskedastisiteyi kanıtlamaz. Düzey ve log modeli yalnız heteroskedastisite "
            "testine göre seçilmez: bağımlı değişken, ekonomik soru ve yorum belirleyicidir (§12.7).")


# --- Adım 6: makale tablosu ----------------------------------------------------------------------------------------

def _paper(choices) -> tuple:
    cov = choices["adim4_hc"]
    notes = cov == "HC1"
    columns = (("(1) Geleneksel SH", "m_duzey"), (f"(2) {cov} SH", "m_rob"))
    if not notes:  # notlardaki HC1 sütunu yan yana
        columns += (("(3) HC1 SH (notlar)", "m_rob_n"),)
    return (
        RegressionTable(columns, (*LEVEL_X, INTERCEPT), "tablo124",
                        ("Tablo 12.4: HPRICE1 düzey modelinin makale tipi sunumu" if notes
                         else f"Makale tablosu: geleneksel, seçtiğiniz {cov} ve notlardaki HC1 standart hatalar")
                        + " · bağımlı değişken: fiyat (bin dolar)", decimals=3, exact=True),
    )


def _stars_word(p_value: float) -> str:
    return stars(p_value) or "yıldızsız"


def _paper_note(state, choices) -> str:
    cov = choices["adim4_hc"]
    conventional, robust = state.models["m_duzey"], state.models["m_rob"]
    changed = [f"{TERM_WORDS[term]} (Sütun 1'de {_stars_word(conventional.pvalues[term])}, Sütun 2'de "
               f"{_stars_word(robust.pvalues[term])})" for term in LEVEL_X
               if stars(conventional.pvalues[term]) != stars(robust.pvalues[term])]
    text = "Katsayılar sütunlarda aynıdır; yıldızlar standart hata türüne bağlıdır. "
    text += (f"Sütun (2)'de ({cov}) yıldızı değişen katsayılar: {'; '.join(changed)}. " if changed else
             f"Bu seçimde ({cov}) yıldızlar iki sütunda aynıdır. ")
    return (text + "Bu yüzden makale tablosunda yıldızlardan önce tablo notu okunur: hangi standart hata (geleneksel, "
            "HC1, HC3), parantez içinde standart hata mı t istatistiği mi, yıldız eşikleri ve kontroller (§12.7).")


# --- Adım 7: dayanıklı ortak test ------------------------------------------------------------------------------------

TERMS_7 = MultiChoice("adim7_terimler", "Sınanan katsayılar", tuple((term, dict(PLOT_LABELS)[term]) for term in LEVEL_X),
                      ("lotsize1000", "bdrms"), help="Notlarda arsa büyüklüğü ve yatak odası (Tablo 12.5).")


def _hypothesis(terms: tuple[str, ...]) -> str:
    """Sıfır hipotezinin Türkçe yazımı: "arsa büyüklüğü ve yatak odası sayısı katsayıları sıfır"."""

    words = [TERM_WORDS[term] for term in terms]
    if len(words) == 1:
        return f"{words[0]} katsayısı sıfır"
    return f"{', '.join(words[:-1])} ve {words[-1]} katsayıları sıfır"


def _joint(choices) -> tuple:
    cov, terms = choices["adim4_hc"], tuple(choices["adim7_terimler"])
    notes_terms = terms == NOTES_JOINT
    notes = cov == "HC1" and notes_terms
    hypothesis = _hypothesis(terms)
    q = len(terms)
    test = "Tek kısıtlı test" if q == 1 else "Ortak test"
    plot = (HypothesisPlot("t", "t7_rob", "sd7", f"{cov} dayanıklı t testi, H₀: {hypothesis}", "t değeri")
            if q == 1 else
            HypothesisPlot("f", "F_rob", "q7", f"{cov} dayanıklı ortak test, H₀: {hypothesis}: F({q}, 84)", "F değeri",
                           alternative="sag", df2="sd7"))
    rows = [("Geleneksel", "gel", q), (f"{cov} dayanıklı", "rob", q)]
    tests = [JointTest("F_gel", "p_gel", "m_duzey", terms, f"Geleneksel F testi, H₀: {hypothesis}", decimals=3,
                       p_decimals=4, shown=False),
             JointTest("F_rob", "p_rob", "m_rob", terms, f"{cov} dayanıklı Wald testi (F biçimi)", decimals=3,
                       p_decimals=4, shown=False)]  # F ve p tabloda
    if cov != "HC1":  # aynı hipotez, notlardaki HC1 kovaryansıyla
        tests.append(JointTest("F_rob_n", "p_rob_n", "m_rob_n", terms, "HC1 dayanıklı Wald testi (notlardaki tür)",
                               decimals=3, p_decimals=4, shown=False))
        rows.append(("HC1 dayanıklı (notlardaki tür)", "rob_n", q))
    if not notes_terms:  # notlardaki hipotez: arsa büyüklüğü ve yatak odası sayısı
        tests += [JointTest("F_gel_nn", "p_gel_nn", "m_duzey", NOTES_JOINT, "Notlar: geleneksel F testi", decimals=3,
                            p_decimals=4, shown=False),
                  JointTest("F_rob_nn", "p_rob_nn", "m_rob" if cov == "HC1" else "m_rob_n", NOTES_JOINT,
                            "Notlar: HC1 dayanıklı Wald testi", decimals=3, p_decimals=4, shown=False)]
        rows += [("Notlar (arsa ve yatak odası): geleneksel", "gel_nn", 2),
                 ("Notlar (arsa ve yatak odası): HC1", "rob_nn", 2)]
    columns = (((f"F({q}, 84)", "f125", "deger"), ("p-değeri", "p125", "deger")) if notes_terms else
               (("F", "f125", "deger"), ("p-değeri", "p125", "deger"), ("Kısıt sayısı q", "q125", "deger")))
    return (
        *tests,
        ModelValue("sd7", "m_duzey", "df_resid", "Payda serbestlik derecesi n − k − 1", decimals=0),
        Scalar("q7", E.const(q), "Kısıt sayısı q", decimals=0),
        *((ModelValue("t7_rob", "m_rob", "t", f"{cov} dayanıklı t istatistiği", term=terms[0], decimals=3),
           Scalar("t7_kare", E.power(E.ref("t7_rob"), 2), "t² (= F)", decimals=3, shown=False)) if q == 1 else ()),
        ScalarTable(tuple((label, E.ref(f"F_{key}")) for label, key, _ in rows), "f125", decimals=3,
                    heading="Kovaryans hesabı"),
        ScalarTable(tuple((label, E.ref(f"p_{key}")) for label, key, _ in rows), "p125", decimals=4,
                    heading="Kovaryans hesabı"),
        *(() if notes_terms else (ScalarTable(tuple((label, E.const(count)) for label, _, count in rows), "q125",
                                              decimals=0, heading="Kovaryans hesabı"),)),
        JoinColumns("tablo125", columns, decimals=3, heading="Kovaryans hesabı", p_columns=("p-değeri",),
                    column_decimals=(("p-değeri", 4),) + (() if notes_terms else (("Kısıt sayısı q", 0),)),
                    title="Tablo 12.5: arsa büyüklüğü ve yatak odası katsayılarının ortak testi" if notes
                    else f"{test}, H₀: {hypothesis}" + ("" if notes_terms else "; notlardaki test yanda")),
        plot,
    )


def _joint_note(state, choices) -> str:
    cov, terms = choices["adim4_hc"], tuple(choices["adim7_terimler"])
    s = state.scalars
    same = (s["p_gel"] < 0.05) == (s["p_rob"] < 0.05)
    text = (f"Aynı hipotez, aynı katsayılar: geleneksel F = {plain(s['F_gel'], 3)} ({_p(s['p_gel'])}), {cov} dayanıklı "
            f"F = {plain(s['F_rob'], 3)} ({_p(s['p_rob'])}). "
            + ("Yüzde 5 düzeyinde iki hesap aynı kararı verir. " if same else
               "Yüzde 5 düzeyinde iki hesap farklı karar verir: kovaryans türü ortak testin sonucunu da değiştirebilir. "))
    if len(terms) == 1:
        shown_t = float(f"{s['t7_rob']:.3f}")
        square = plain(shown_t ** 2, 3)
        exact = "" if square == plain(s["t7_kare"], 3) else f" (yuvarlanmamış t ile {plain(s['t7_kare'], 3)} = F)"
        text += (f"Tek kısıtta F = t²: {cov} dayanıklı t = {plain(s['t7_rob'], 3)}, t² = {square}{exact}; F testi "
                 "dayanıklı t testiyle aynı kararı verir. ")
    if cov != "HC1":
        text += f"Notlardaki HC1 kovaryansıyla F = {plain(s['F_rob_n'], 3)} ({_p(s['p_rob_n'])}). "
    return (text + "Tekli testler dayanıklı hesapla yapılıyorsa ortak test de aynı kovaryansla yapılır; tabloda testin "
            "hangi kovaryansla hesaplandığı yazılır (§12.8).")


# --- Adım 8: WAGE1 düzey ve log ücret modelleri ---------------------------------------------------------------------

X_8 = MultiChoice("adim8_x", "Açıklayıcı değişkenler", W.options(WAGE, WAGE_X), WAGE_DEFAULT,
                  help="Notlarda eğitim, deneyim, kıdem ve kadın kuklası (§12.10).")


WAGE_MODELS = (("wd", "wage", "Ücret düzeyi"), ("wl", "lwage", "Log ücret"))
"""§12.10'un iki WAGE1 modeli: (önek, bağımlı değişken, başlık)."""
EDUC_SE = (("wd", "", "Ücret: eğitimin geleneksel SH'si", 3), ("wd", "_hc1", "Ücret: eğitimin HC1 SH'si", 3),
           ("wl", "", "Log ücret: eğitimin geleneksel SH'si", 4), ("wl", "_hc1", "Log ücret: eğitimin HC1 SH'si", 4))
"""Eğitim katsayısının dört standart hatası: (model öneki, kovaryans soneki, etiket, basamak)."""


def _wage_models(terms: tuple[str, ...], suffix: str, note: str) -> tuple:
    """İki model (geleneksel ve HC1) ve heteroskedastisite testleri; ``suffix`` notlardaki model için "_n"."""

    operations: list = []
    for key, outcome, title in WAGE_MODELS:
        name = f"{key}{suffix}"
        operations += [
            OLS(f"m_{name}", WAGE, outcome, terms, f"{note}{title}: {_formula(outcome, terms)}"),
            OLS(f"m_{name}_hc1", WAGE, outcome, terms, f"{note}{title}, HC1 dayanıklı kovaryansla", cov_type="HC1"),
            HeteroskedasticityTest(f"bp_{name}", f"p_bp_{name}", f"m_{name}", "bp", f"{note}{title}: Breusch–Pagan",
                                   shown=False),  # sayılar tabloda
            HeteroskedasticityTest(f"wh_{name}", f"p_wh_{name}", f"m_{name}", "white", f"{note}{title}: White",
                                   shown=False),
        ]
    if "educ" in terms:
        operations += [ModelValue(f"sh_educ_{key}{cov}{suffix}", f"m_{key}{suffix}{cov}", "se", note + label,
                                  term="educ", decimals=digits, shown=False) for key, cov, label, digits in EDUC_SE]
    return tuple(operations)


def _wage(choices) -> tuple:
    terms = tuple(choices["adim8_x"])
    notes = terms == WAGE_DEFAULT
    rows = [(title, key) for key, _, title in WAGE_MODELS]
    side: tuple = ()
    if not notes:  # notlardaki model (eğitim, deneyim, kıdem, kadın) yan yana
        side = _wage_models(WAGE_DEFAULT, "_n", "Notlar — ")
        rows = [(f"Notlar: {title.lower()}", f"{key}_n") for key, _, title in WAGE_MODELS] + [
            (f"Seçiminiz: {title.lower()}", key) for key, _, title in WAGE_MODELS]
    educ_rows = tuple((label.replace("Ücret: eğitimin ", "Ücret, ").replace("Log ücret: eğitimin ", "Log ücret, ")
                       .replace(" SH'si", ""), key + cov) for key, cov, label, _ in EDUC_SE)
    educ: tuple = ()
    if not notes:  # eğitim katsayısının standart hataları: notlardaki model ve seçiminiz
        educ = (ScalarTable(tuple((label, E.ref(f"sh_educ_{key}_n")) for label, key in educ_rows),
                            "egitim_notlar", decimals=4, heading="Model ve standart hata", value="Notlar",
                            title="" if "educ" in terms else "Notlardaki model: eğitim katsayısının standart hataları"),)
        if "educ" in terms:
            educ += (ScalarTable(tuple((label, E.ref(f"sh_educ_{key}")) for label, key in educ_rows), "egitim_secim",
                                 decimals=4, heading="Model ve standart hata"),
                     JoinColumns("egitim_sh", (("Notlar", "egitim_notlar", "deger"), ("Seçiminiz", "egitim_secim", "deger")),
                                 decimals=4, heading="Model ve standart hata",
                                 title="Eğitim katsayısının standart hatası: notlardaki model ve seçiminiz"))
    return (
        LoadWooldridge(WAGE, "WAGE1 veri seti (Wooldridge, 2020): 526 çalışan"),
        *_wage_models(terms, "", ""),
        *side,
        *(ScalarTable(tuple((title, E.ref(f"{prefix}_{key}")) for title, key in rows), f"{prefix}_1210", decimals=3,
                      heading="Model") for prefix in ("bp", "p_bp", "wh", "p_wh")),
        JoinColumns("testler1210", (("BP LM", "bp_1210", "deger"), ("BP p", "p_bp_1210", "deger"),
                                    ("White LM", "wh_1210", "deger"), ("White p", "p_wh_1210", "deger")),
                    decimals=3, heading="Model", p_columns=("BP p", "White p"),
                    title="WAGE1: ücret düzeyi ve log ücret modellerinde heteroskedastisite testleri" + (
                        "" if notes else " (notlardaki model ve seçtiğiniz açıklayıcılar)")),
        RegressionTable((("(1) Ücret, geleneksel", "m_wd"), ("(2) Ücret, HC1", "m_wd_hc1"),
                         ("(3) Log ücret, geleneksel", "m_wl"), ("(4) Log ücret, HC1", "m_wl_hc1")),
                        (*terms, INTERCEPT), "tablo1210", "WAGE1: aynı katsayılar, geleneksel ve HC1 standart hatalar"
                        + ("" if notes else " (seçtiğiniz açıklayıcılar)"), decimals=4, exact=True),
        *educ,
    )


def _wage_note(state, choices) -> str:
    notes = tuple(choices["adim8_x"]) == WAGE_DEFAULT
    s = state.scalars
    text = (f"Ücret düzeyi modelinde BP {_p(s['p_bp_wd'], 3)}, White {_p(s['p_wh_wd'], 3)}; log ücret modelinde BP "
            f"{_p(s['p_bp_wl'], 3)}, White {_p(s['p_wh_wl'], 3)}. ")
    chosen = tuple(choices["adim8_x"])
    if len(chosen) == 1 and chosen[0] in WAGE_DUMMIES:
        text += ("Tek bir 0/1 kukla açıklayıcıda White testi Breusch–Pagan testiyle aynıdır: kuklanın karesi "
                 "kendisidir. ")
    if s["p_bp_wl"] >= 0.05 > s["p_wh_wl"]:
        text += "Log dönüşümü örüntüyü azaltmış görünse de iki test aynı sonucu vermez. "
    if "sh_educ_wd" in s:
        text += (f"Eğitim katsayısının standart hatası düzey modelinde {plain(s['sh_educ_wd'], 3)} (geleneksel) ve "
                 f"{plain(s['sh_educ_wd_hc1'], 3)} (HC1), log modelde {plain(s['sh_educ_wl'], 4)} ve "
                 f"{plain(s['sh_educ_wl_hc1'], 4)}. ")
    if not notes:
        text += (f"Notlardaki modelde (eğitim, deneyim, kıdem, kadın) log ücret için BP {_p(s['p_bp_wl_n'], 3)}, White "
                 f"{_p(s['p_wh_wl_n'], 3)}; eğitimin standart hatası log modelde {plain(s['sh_educ_wl_n'], 4)} "
                 f"(geleneksel) ve {plain(s['sh_educ_wl_hc1_n'], 4)} (HC1). ")
    return (text + "Yatay kesitte heteroskedastisite makul bir olasılıktır: log modelde de dayanıklı standart hata "
            "raporlamak makuldür. Dayanıklı standart hata eksik değişken ya da nedensellik sorununu çözmez (§12.10).")


# --- Tanım ---------------------------------------------------------------------------------------------------------

_LEVEL = (
    (INTERCEPT, "Intercept", (-21.7703, 29.4750, 0.4622, 37.1382, 0.5593)),
    ("lotsize1000", "lotsize1000", (2.0677, 0.6421, 0.0018, 1.2514, 0.1022)),
    ("sqrft100", "sqrft100", (12.2778, 1.3237, 0.0, 1.7725, 0.0)),
    ("bdrms", "bdrms", (13.8525, 9.0101, 0.1279, 8.4786, 0.1060)),
)
_LOG = (
    (INTERCEPT, "Intercept", (-1.2970, 0.6513, 0.0497, 0.7813, 0.1006)),
    ("llotsize", "llotsize", (0.1680, 0.0383, 0.0, 0.0415, 0.0001)),
    ("lsqrft", "lsqrft", (0.7002, 0.0929, 0.0, 0.1038, 0.0)),
    ("bdrms", "bdrms", (0.0370, 0.0275, 0.1831, 0.0306, 0.2305)),
)
_COLUMNS = ("coef", "OLS se", "OLS p", "HC1 se", "HC1 p")
_TABLE122 = (("Fiyat düzeyi", (14.092, 0.0028, 33.732, 0.0)), ("Log fiyat", (4.223, 0.2383, 9.549, 0.3882)))
TABLE124_WORDS = {"lotsize1000": "arsa", "lotsize1000_sh": "arsa (SH)", "sqrft100": "konut büyüklüğü",
                  "sqrft100_sh": "konut büyüklüğü (SH)", "bdrms": "yatak odası", "bdrms_sh": "yatak odası (SH)",
                  INTERCEPT: "sabit", f"{INTERCEPT}_sh": "sabit (SH)", "n": "gözlem sayısı", "r2": "R²"}
"""Tablo 12.4 kontrol etiketleri: satır anahtarlarının Türkçe adları."""
_TABLE124 = (
    ("(1) Geleneksel SH", (("lotsize1000", 2.068), ("lotsize1000_sh", 0.642), ("sqrft100", 12.278),
                           ("sqrft100_sh", 1.324), ("bdrms", 13.853), ("bdrms_sh", 9.010), (INTERCEPT, -21.770),
                           (f"{INTERCEPT}_sh", 29.475), ("n", 88), ("r2", 0.672))),
    ("(2) HC1 SH", (("lotsize1000", 2.068), ("lotsize1000_sh", 1.251), ("sqrft100", 12.278), ("sqrft100_sh", 1.773),
                    ("bdrms", 13.853), ("bdrms_sh", 8.479), (INTERCEPT, -21.770), (f"{INTERCEPT}_sh", 37.138),
                    ("n", 88), ("r2", 0.672))),
)


def _output_checks(table: str, rows, source: str) -> tuple[Check, ...]:
    checks = []
    for term, name, values in rows:
        for column, value in zip(_COLUMNS, values):
            label = f"{source}: {name}, {column}" + (" (< 0,0001)" if column.endswith("p") and value == 0 else "")
            checks.append(Check(label, TableTarget(table, term, column), value, 4))
    return tuple(checks)


STEPS = (
    interactive_step(
        number=1,
        title="Artık grafikleriyle ilk tanı",
        note=NoteRef("12.4", 0, ("Şekil 12.3", "Şekil 12.4", "Şekil 12.5")),
        explanation=(
            "Hata terimleri gözlenmez; EKK artıkları $\\hat u_i$ kullanılır. Artık–tahmin grafiğinde yatay eksende "
            "$\\hat Y_i$, dikey eksende $\\hat u_i$ vardır. Homoskedastik modelde sıfır çizgisi çevresindeki dikey "
            "yayılım yaklaşık sabittir; huni ya da yelpaze biçimi heteroskedastisite şüphesi doğurur. Mutlak artıklar "
            "hata büyüklüğünü işaretinden bağımsız gösterir. İncelenen modeli değiştirin."
        ),
        controls=(MODEL_1,),
        build=_residual_plots,
        checks=(
            *(Check(f"Kod 12.3: {term}, coef", CoefTarget("m_duzey", term, "coef"), value, 4)
              for term, value in ((INTERCEPT, -21.7703), ("lotsize1000", 2.0677), ("sqrft100", 12.2778),
                                  ("bdrms", 13.8525))),
            Check("Kod 12.3: R-squared", ModelTarget("m_duzey", "r2"), 0.672, 3),
            *(Check(f"Kod 12.4: {term}, coef", CoefTarget("m_log", term, "coef"), value, 4)
              for term, value in ((INTERCEPT, -1.2970), ("llotsize", 0.1680), ("lsqrft", 0.7002), ("bdrms", 0.0370))),
            Check("Kod 12.4: R-squared", ModelTarget("m_log", "r2"), 0.643, 3),
        ),
        note_for=lambda state, choices: _residual_note(state, choices),
    ),
    interactive_step(
        number=2,
        title="Breusch–Pagan ve White testleri",
        note=NoteRef("12.5", 0, ("Tablo 12.2", "Kod 12.1")),
        explanation=(
            "İki testin sıfır hipotezi homoskedastisitedir: $H_0: \\operatorname{Var}(u_i \\mid X_i) = \\sigma^2$. "
            "Kareli artıklar $\\hat u_i^2$ yardımcı regresyonla açıklanır; $H_0$ altında ve büyük örneklemde "
            "$LM = nR^2_{aux}$ yaklaşık $\\chi^2_q$ dağılır ($q$: yardımcı regresyondaki sabit dışındaki terim sayısı). "
            "Breusch–Pagan yardımcı regresyonda modelin açıklayıcılarını, White ayrıca karelerini ve çapraz çarpımlarını "
            "kullanır. Küçük p-değeri homoskedastisitenin reddedilmesine yol açar. Testi ve modeli değiştirin."
        ),
        controls=(TEST_2, MODEL_2),
        build=_tests,
        checks=(
            _scalar("lm_elle", 14.092, "§12.5: BP LM = n·R² (düzey modeli)", 3),
            _scalar("p_elle", 0.0028, "§12.5: BP p (düzey modeli)", 4),
            *(Check(f"Tablo 12.2: {row}, {column}" + (" < 0,001" if value == 0 else ""),
                    TableTarget("tablo122", row, column), value, 3 if value == 0 or "LM" in column else 4)
              for row, values in _TABLE122 for column, value in zip(("BP LM", "BP p", "White LM", "White p"), values)),
        ),
        note_for=lambda state, choices: _tests_note(state, choices),
    ),
    interactive_step(
        number=3,
        title="HC0, HC1, HC2 ve HC3: aynı katsayı, farklı standart hata",
        note=NoteRef("12.6", 0, ("Tablo 12.3",)),
        explanation=(
            "Dayanıklı standart hata her gözlemin hata büyüklüğünün aynı olduğunu varsaymaz. Basit regresyonda "
            "§12.3'teki gerçek varyans formülünde $\\sigma_i^2$ yerine gözleme özgü bir tahmin $\\omega_i$ konur: "
            "$\\widehat{\\operatorname{Var}}(\\hat\\beta_1) = \\sum_i (X_i - \\bar X)^2\\omega_i / [\\sum_i (X_i - "
            "\\bar X)^2]^2$; çoklu regresyonda aynı fikir bütün katsayılara uygulanır. HC0'da $\\omega_i = \\hat u_i^2$; "
            "HC1 bunu $n/(n - k - 1)$ ile çarpar ($k$: sabit dışındaki açıklayıcı sayısı); HC2 ve HC3 gözlemin "
            "kaldıracı $h_i$ ile $\\hat u_i^2/(1 - h_i)$ ve $\\hat u_i^2/(1 - h_i)^2$ kullanır. Kaldıraç, gözlemin "
            "açıklayıcı değerlerinin örneklemin merkezinden uzaklığını ölçer ($0 < h_i < 1$; basit regresyonda "
            "$h_i = 1/n + (X_i - \\bar X)^2 / \\sum_j (X_j - \\bar X)^2$). Katsayılar, tahmin edilen değerler ve R² "
            "değişmez. Modeli değiştirin."
        ),
        controls=(MODEL_3,),
        build=_hc_table,
        checks=tuple(
            Check(f"Kod 12.3: {term}, HC1 se", TableTarget("tablo_hc", f"{term}_sh", "(3) HC1"), value, 4)
            for term, value in ((INTERCEPT, 37.1382), ("lotsize1000", 1.2514), ("sqrft100", 1.7725), ("bdrms", 8.4786))
        ),
        note_for=lambda state, choices: _hc_note(state, choices),
    ),
    interactive_step(
        number=4,
        title="HPRICE1: aynı katsayı, farklı çıkarım",
        note=NoteRef("12.7", 0, ("Kod 12.2 (§12.6)", "Kod 12.3", "Şekil 12.6")),
        explanation=(
            "Düzey fiyat modelinin geleneksel ve dayanıklı sonuçları aynı tabloda: katsayılar aynıdır, standart hata, "
            "p-değeri ve güven aralıkları değişir. statsmodels'te `fit(cov_type=\"HC1\", use_t=True)` (Kod 12.2) t ve F "
            "dağılımını ($n - k - 1$ serbestlik derecesi; $k$ sabit dışındaki açıklayıcı sayısı) kullanır; `use_t` "
            "yazılmazsa normal dağılım kullanılır ve p-değerleri notlardakinden farklı olur. Dayanıklı kovaryans türünü "
            "değiştirin: notlardaki HC1 sütunu yan yana gösterilir."
        ),
        controls=(HC_4,),
        build=_level,
        checks=(
            *_output_checks("kod123", _LEVEL, "Kod 12.3"),
            Check("Kod 12.3: No. Observations", ModelTarget("m_duzey", "nobs"), 88, 0),
            Check("Kod 12.3: R-squared", ModelTarget("m_duzey", "r2"), 0.672, 3),
        ),
        note_for=lambda state, choices: _level_note(state, choices),
    ),
    interactive_step(
        number=5,
        title="Log modelle karşılaştırma",
        note=NoteRef("12.7", 0, ("Kod 12.4",)),
        explanation=(
            "$\\ln(\\text{price}_i) = \\beta_0 + \\beta_1 \\ln(\\text{lotsize}_i) + \\beta_2 \\ln(\\text{sqrft}_i) + "
            "\\beta_3 \\text{bdrms}_i + u_i$: arsa ve konut büyüklüğü katsayıları esnekliktir. Notlardaki HC1 hesabıyla "
            "log modelde geleneksel ve dayanıklı standart hatalar birbirine yakındır; bu, artık grafiği ve testlerle "
            "uyumludur. Adım 4'teki kovaryans seçimi bu adımı da belirler."
        ),
        uses=(HC_4,),
        build=_log,
        checks=(
            *_output_checks("kod124", _LOG, "Kod 12.4"),
            Check("Kod 12.4: No. Observations", ModelTarget("m_log", "nobs"), 88, 0),
            Check("Kod 12.4: R-squared", ModelTarget("m_log", "r2"), 0.643, 3),
        ),
        note_for=lambda state, choices: _log_note(state, choices),
    ),
    interactive_step(
        number=6,
        title="Makale tablosunda standart hata türü",
        note=NoteRef("12.7", 0, ("Tablo 12.4",)),
        explanation=(
            "Aynı katsayılar standart hata türüne göre farklı yıldızlar alabilir. Makale tablosunda tablo notu önce "
            "okunur: kullanılan standart hata ya da kovaryans türü (HC0–HC3 hangisi), parantez içinde ne olduğu, "
            "yıldız eşikleri, gözlem sayısı ve kontroller. Adım 4'teki kovaryans seçimi bu adımı da belirler."
        ),
        uses=(HC_4,),
        build=_paper,
        checks=tuple(
            Check(f"Tablo 12.4: {heading}, {TABLE124_WORDS[row]}", TableTarget("tablo124", row, heading), value,
                  0 if row == "n" else 3)
            for heading, values in _TABLE124 for row, value in values
        ),
        note_for=lambda state, choices: _paper_note(state, choices),
    ),
    interactive_step(
        number=7,
        title="Dayanıklı ortak test",
        note=NoteRef("12.8", 0, ("Tablo 12.5", "Kod 12.5")),
        explanation=(
            "Klasik $F$ istatistiği homoskedastik kovaryansa dayanır. Dayanıklı ortak test (Wald testi) sınanan "
            "katsayıların tahminlerini onların dayanıklı varyans ve kovaryanslarıyla ölçekler; yazılım sonucu kısıt "
            "sayısı $q$'ya bölerek $F$ biçiminde raporlar ve p-değerini yaklaşık olarak $F(q, n - k - 1)$ dağılımından "
            "okur. Hipotez aynıdır; değişen, katsayıların ortak belirsizliğinin ölçüsüdür. Sınanan katsayıları "
            "değiştirin: notlardaki test yan yana gösterilir."
        ),
        controls=(TERMS_7,),
        uses=(HC_4,),
        build=_joint,
        checks=(
            _scalar("F_gel", 6.610, "Tablo 12.5: geleneksel F(2, 84)", 3),
            _scalar("p_gel", 0.0022, "Tablo 12.5: geleneksel p", 4),
            _scalar("F_rob", 2.365, "Tablo 12.5: HC1 dayanıklı F(2, 84)", 3),
            _scalar("p_rob", 0.1002, "Tablo 12.5: HC1 dayanıklı p", 4),
        ),
        note_for=lambda state, choices: _joint_note(state, choices),
    ),
    interactive_step(
        number=8,
        title="WAGE1: düzey ve log ücret modellerinde tanı ve dayanıklı standart hata",
        note=NoteRef("12.10", 0),
        explanation=(
            "Heteroskedastisiteyle karşılaşıldığında tek bir mekanik çözüm yoktur: yatay kesit çıkarımında dayanıklı "
            "standart hata temel seçenektir; fonksiyonel biçim ekonomik gerekçeyle yeniden düşünülür; ağırlıklı en "
            "küçük kareler yalnız güvenilir varyans bilgisi varsa düşünülür. WAGE1'de ücret düzeyi ve log ücret "
            "modelleri karşılaştırılır. Açıklayıcı değişkenleri değiştirin."
        ),
        controls=(X_8,),
        build=_wage,
        checks=(
            _scalar("p_bp_wd", 0.0, "§12.10: ücret düzeyi, BP p-değeri çok küçük (< 0,001)", 3),
            _scalar("p_wh_wd", 0.0, "§12.10: ücret düzeyi, White p-değeri çok küçük (< 0,001)", 3),
            _scalar("p_bp_wl", 0.125, "§12.10: log ücret, BP p", 3),
            _scalar("p_wh_wl", 0.041, "§12.10: log ücret, White p", 3),
            _scalar("sh_educ_wd", 0.049, "§12.10: ücret düzeyi, eğitimin geleneksel SH'si", 3),
            _scalar("sh_educ_wd_hc1", 0.061, "§12.10: ücret düzeyi, eğitimin HC1 SH'si", 3),
            _scalar("sh_educ_wl", 0.0069, "§12.10: log ücret, eğitimin geleneksel SH'si", 4),
            _scalar("sh_educ_wl_hc1", 0.0080, "§12.10: log ücret, eğitimin HC1 SH'si", 4),
        ),
        note_for=lambda state, choices: _wage_note(state, choices),
    ),
)


def _derived_labels() -> tuple[tuple[str, str], ...]:
    """White yardımcı regresyonunun kare ve çarpım sütunlarının adları."""

    short = {"lotsize1000": "arsa/1.000", "sqrft100": "konut/100", "bdrms": "yatak odası", "llotsize": "log arsa",
             "lsqrft": "log konut"}
    found = []
    for terms in (LEVEL_X, LOG_X):
        for index, first in enumerate(terms):
            for second in terms[index:]:
                name = f"{first}_kare" if first == second else f"{first}_x_{second}"
                found.append((name, f"({short[first]})²" if first == second else f"{short[first]} × {short[second]}"))
    return tuple(dict.fromkeys(found))


KONU12_LAB = LabSpec(
    topic_key="konu12",
    title="Uygulama: Heteroskedastisite ve Heteroskedastisiteye Dayanıklı Çıkarım",
    note_section="12",
    steps=STEPS,
    labels=(
        *W.labels(HOUSE, WAGE),
        (INTERCEPT, "Sabit terim"),
        ("lotsize1000", "Arsa büyüklüğü/1.000 (bin fit²)"),
        ("sqrft100", "Konut büyüklüğü/100 (yüz fit²)"),
        ("artik", "Artık û"),
        ("artik2", "Kareli artık û²"),
        ("tahmin", "Tahmin edilen değer Ŷ"),
        ("mutlak", "Mutlak artık |û|"),
        ("ceyrek", "Tahmin edilen değerin çeyreği"),
        ("n", "Gözlem sayısı"),
        ("ss", "Artıkların standart sapması"),
        ("ort_mutlak", "Ortalama mutlak artık"),
        *_derived_labels(),
    ),
    consistency_notes=(
        "Veri notlardaki gibi HPRICE1 ve WAGE1'dir. Notlardaki Kod 12.1 veriyi wooldridge paketinden okur, "
        "değişkenleri türetir ve modeli kurar; Kod 12.2 ve 12.5 aynı house tablosunu kullanır. Önceki notlarda Kod 12.1'de "
        "bu satırlar yoktu (house ve model hazır varsayılıyordu). Bölüm betiği, uygulama ve üretilen kod da veriyi "
        "paketten okur (Bölüm 12 betiği önceden data/ CSV kopyasını okuyordu; basılı sayıların hiçbiri değişmedi).",
        "Notlardaki Kod 12.2'nin ikinci yazımı fit(cov_type=\"HC1\") t yerine normal dağılım kullanıyordu (arsa "
        "katsayısının p-değeri 0,1022 yerine 0,0985); use_t=True eklendi. get_robustcov_results(cov_type=\"HC1\") "
        "t dağılımını kullanır ve notlardaki sayıları verir.",
        "Breusch–Pagan testi Koenker'in n·R² biçimidir (statsmodels het_breuschpagan varsayılanı, R'de lmtest::bptest "
        "varsayılanı); R kodu yardımcı regresyonu lm() ile açıkça kurar.",
        "White testi uygulamada ve üretilen Python kodunda yardımcı regresyonla açıkça kurulur (`white_testi`; R'de "
        "lm()). Notlardaki Kod 12.5 het_white çağırır ve HPRICE1'de aynı sayıyı verir (33,732). §12.10'da 0/1 kukla "
        "açıklayıcı seçilince het_white kullanılmadı: kuklanın karesi kendisi olduğundan yardımcı tasarım tam ranklı "
        "değildir ve statsmodels 0.14.6'daki rank denetimi (diagnostic.py, assert) sayısal gürültüyle bazı makinelerde "
        "AssertionError verir. Açık hesap kuklanın karesini bir kez sayar ve het_white'la aynı LM ve p'yi verir.",
        "§12.10'a WAGE1 modellerinin açıklayıcıları (eğitim, deneyim, kıdem, kadın) ve eğitim katsayısının geleneksel ve "
        "HC1 standart hataları eklendi (0,049 ve 0,061; log modelde 0,0069 ve 0,0080).",
        "Heteroskedastik benzetim (§12.9, Tablo 12.6, Şekil 12.7–12.8) ve §12.1'deki Şekil 12.1–12.2 Sezgi sekmesindeki "
        "Deney 1'in varsayılan ayarlarıyla (tohum 305) yeniden üretildi; notlara veri üretim süreci ve tohum eklendi. "
        "Tablo 12.6'nın sayıları bu yüzden değişti (ör. ortalama eğim 2,009 yerine 1,986, geleneksel kapsama %88,3 "
        "yerine %88,10).",
    ),
)
