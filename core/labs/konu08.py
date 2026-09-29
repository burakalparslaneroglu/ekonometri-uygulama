"""Konu 8 uygulaması: çoklu doğrusal kısıtlar ve F testi (gerçek veri).

Bölüm 8'in çözümlü örnekleri bölüm sırasıyla: F testinin karar kuralı ve ortak p-değeri (§8.5, Şekil 8.1), WAGE1'de
yazılım çıktısındaki iki farklı F (§8.6, Kod 8.1–8.2), kısıtlı ve kısıtsız model karşılaştırması ile F'nin iki formülü
(§8.6, Tablo 8.1), genel anlamlılık testi (§8.7), tek kısıtta F = t² (§8.8), HPRICE1 ortak testi (§8.9, Tablo 8.2) ve
makale tablosunda ortak testler (§8.12, Tablo 8.6). Bölümün benzetimi (Tablo 8.3, Şekil 8.2–8.3, Kod 8.3) Sezgi
sekmesindeki Konu 8 Deney 1'dir. Her ``Check`` notlarda basılı bir sayıdır; değer notlardan kopyalanmıştır,
hesaplanmamıştır.

Etkileşim: birlikte sınanan katsayılar ve anlamlılık düzeyi (Adım 1), kısıtlı modelden çıkarılan değişkenler (Adım 3
ve 6), genel testteki model (Adım 4) ve F = t² bağlantısındaki katsayı (Adım 5). Ortak testler klasik (homoskedastik)
EKK varsayımlarına dayanır; heteroskedastisiteye dayanıklı ortak test Konu 12'nin konusudur.
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
    CoefTarget,
    HypothesisPlot,
    JointTest,
    LabSpec,
    LabStep,
    LoadWooldridge,
    ModelTarget,
    ModelValue,
    MultiChoice,
    NoteRef,
    RegressionTable,
    Scalar,
    ScalarTarget,
    ShowModel,
    TableTarget,
    interactive_step,
)

DATA = "wage1"
HPRICE = "hprice1"
REGRESSORS = ("educ", "exper", "tenure")
"""Bölüm 8'in ücret modeli: wage ~ educ + exper + tenure (n = 526)."""
TESTED = ("exper", "tenure")
"""Notlardaki ortak hipotez: H₀: β_deneyim = 0, β_kıdem = 0."""
HOUSE_TERMS = ("lotsize", "sqrft", "bdrms")
"""HPRICE1 konut fiyatı modeli: price ~ lotsize + sqrft + bdrms (n = 88)."""
HOUSE_TESTED = ("lotsize", "bdrms")
"""Notlardaki HPRICE1 ortak hipotezi: H₀: β_arsa = 0, β_yatak = 0 (konut büyüklüğü modelde kalır)."""
MODEL_OPTIONS = ("educ", "exper", "tenure", "numdep")
"""Genel testte modele alınabilen açıklayıcı değişkenler (kukla olmayan sayısal değişkenler)."""
TERM_OPTIONS = W.options(DATA, REGRESSORS)
WORDS = {"educ": "eğitim", "exper": "deneyim", "tenure": "kıdem", "numdep": "bakmakla yükümlü kişi sayısı",
         "lotsize": "arsa büyüklüğü", "sqrft": "konut büyüklüğü", "bdrms": "yatak odası sayısı"}
GENITIVE = {"educ": "eğitimin", "exper": "deneyimin", "tenure": "kıdemin", "lotsize": "arsa büyüklüğünün",
            "sqrft": "konut büyüklüğünün", "bdrms": "yatak odası sayısının"}
SYMBOLS = {"educ": "β_eğitim", "exper": "β_deneyim", "tenure": "β_kıdem", "lotsize": "β_arsa",
           "sqrft": "β_konut", "bdrms": "β_yatak"}
ALPHAS = (("0.1", "%10"), ("0.05", "%5"), ("0.01", "%1"))


def _scalar(name: str, expected: float, label: str, decimals: int) -> Check:
    return Check(label, ScalarTarget(name), expected, decimals)


def _listing(items) -> str:
    """Türkçe sıralama: "a", "a ve b", "a, b ve c"."""

    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " ve " + items[-1]


def _jointly(terms: tuple[str, ...]) -> str:
    """Tamlama: "deneyim ve kıdemin" (ek yalnız son öğeye)."""

    return _listing([*(WORDS[term] for term in terms[:-1]), GENITIVE[terms[-1]]])


def _hypothesis(terms: tuple[str, ...]) -> str:
    """Sıfır hipotezi: "H₀: β_deneyim = 0, β_kıdem = 0"."""

    return "H₀: " + ", ".join(f"{SYMBOLS[term]} = 0" for term in terms)


def _percent(value: float) -> str:
    return "%" + plain(100 * value, 0)


def _p(value: float, decimals: int = 3) -> str:
    """p-değerinin eşitlikli yazımı: "p = 0,064"; basamakta sıfıra yuvarlanıyorsa "p < 0,001"."""

    if value < 0.5 * 10 ** -decimals:
        return "p < " + plain(10 ** -decimals, decimals)
    return f"p = {plain(value, decimals)}"


def _load() -> tuple:
    return (LoadWooldridge(DATA, "WAGE1 veri seti (Wooldridge, 2020): 526 çalışan"),
            OLS("m", DATA, "wage", REGRESSORS, "Kısıtsız model: wage ~ educ + exper + tenure"))


# --- Adım 1: karar kuralı ve p-değeri -------------------------------------------------------------------

TESTED_1 = MultiChoice("adim1_x", "Birlikte sınanan katsayılar", TERM_OPTIONS, TESTED,
                       help="Notlarda deneyim ve kıdem (Şekil 8.1). Üçü birden seçilirse genel anlamlılık testi olur.")
ALPHA_1 = Choice("adim1_alfa", "Anlamlılık düzeyi α", ALPHAS, "0.05", help="Şekil 8.1 yüzde 5 düzeyindedir.")


def _decision(choices) -> tuple:
    terms, alpha = tuple(choices["adim1_x"]), float(choices["adim1_alfa"])
    q = len(terms)
    notes = terms == TESTED and alpha == 0.05
    subject = (f"{_jointly(terms)} birlikte" if q > 1 else f"{WORDS[terms[0]]} katsayısının")
    title = ("Şekil 8.1: Deneyim ve kıdemin birlikte sınanması: F(2, 522)" if notes
             else f"{subject[0].upper() + subject[1:]} sınanması: F({q}, 522), yüzde {plain(100 * alpha, 0)} düzeyi")
    return (
        *_load(),
        JointTest("F_ortak", "p_ortak", "m", terms, f"Ortak F testi, {_hypothesis(terms)}"),
        ModelValue("sd_artik", "m", "df_resid", "Payda serbestlik derecesi n − k − 1", decimals=0),
        Scalar("pay_sd", E.const(q), "Pay serbestlik derecesi q (kısıt sayısı)", decimals=0),
        Scalar("kritik_F", E.finv(round(1 - alpha, 6), E.ref("pay_sd"), E.ref("sd_artik")),
               f"Kritik değer: F(q, n − k − 1) dağılımının üst yüzde {plain(100 * alpha, 0)} noktası", decimals=2),
        HypothesisPlot("f", "F_ortak", "pay_sd", title, "F değeri", alpha=alpha, alternative="sag", df2="sd_artik"),
    )


def _decision_note(state, choices) -> str:
    terms, alpha = tuple(choices["adim1_x"]), float(choices["adim1_alfa"])
    s = state.scalars
    f, critical, p = s["F_ortak"], s["kritik_F"], s["p_ortak"]
    rejected = f > critical
    text = (f"Gözlenen F = {plain(f, 2)}; {_percent(alpha)} düzeyinde kritik değer {plain(critical, 2)}. "
            f"F {'>' if rejected else '≤'} kritik değer olduğundan H₀ {'reddedilir' if rejected else 'reddedilemez'}; "
            f"{'ortak ' if len(terms) > 1 else ''}{_p(p)} ile karar aynıdır (p {'<' if p < alpha else '≥'} α). F "
            "negatif olamaz ve reddetme bölgesi "
            "yalnız üst kuyruktadır. ")
    if len(terms) == 1:
        return text + "Tek kısıtta F testi iki taraflı t testiyle aynı kararı verir: F = t² (§8.8)."
    if len(terms) == len(REGRESSORS):
        text += "Bütün eğimlerin birlikte sınanması genel anlamlılık testidir; çıktıdaki F-statistic budur (§8.7). "
    if rejected:
        return text + ("Ortak testin reddedilmesi sınanan katsayıların hepsinin ayrı ayrı sıfırdan farklı olduğunu "
                       "göstermez; kısıtlardan en az birinin veriyle uyumsuz olduğunu gösterir (§8.5).")
    return text + "H₀'ın reddedilememesi katsayıların birlikte sıfır olduğunu kanıtlamaz (§8.5)."


# --- Adım 2: yazılım çıktısında iki farklı F -----------------------------------------------------------

def _output() -> tuple:
    return (
        ShowModel("m", "WAGE1 modelinin temel Python çıktısı", columns=("coef", "se", "t", "p"),
                  stats=("nobs", "df_resid", "r2", "f", "f_p")),
        ModelValue("n_m", "m", "nobs", "No. Observations: gözlem sayısı n", decimals=0),
        ModelValue("sd_m", "m", "df_resid", "Df Residuals: n − k − 1", decimals=0),
        Scalar("sd_model", E.sub(E.sub(E.ref("n_m"), E.ref("sd_m")), 1), "Df Model: eğim sayısı k", decimals=0),
        ModelValue("genel_F", "m", "f", "F-statistic: genel anlamlılık testinin F istatistiği", decimals=2),
        ModelValue("genel_p", "m", "f_p", "Prob (F-statistic): genel testin p-değeri", decimals=3),
        Scalar("genel_p_olcek", E.mul(E.ref("genel_p"), 1e41), "Prob (F-statistic) × 10⁴¹ (çıktıda 3.41e-41)",
               decimals=2),
        JointTest("F_cikti", "p_cikti", "m", TESTED, 'model.f_test("exper = 0, tenure = 0"): deneyim ve kıdem birlikte',
                  decimals=4),
        Scalar("p_cikti_olcek", E.mul(E.ref("p_cikti"), 1e22), "Ortak p-değeri × 10²² (çıktıda 8.56e-22)",
               decimals=2),
    )


# --- Adım 3: kısıtlı ve kısıtsız model --------------------------------------------------------------------

DROPPED_3 = MultiChoice("adim3_x", "Kısıtlı modelden çıkarılan (birlikte sınanan) değişkenler", TERM_OPTIONS, TESTED,
                        help="Notlarda deneyim ve kıdem; kısıtlı model yalnız eğitimi içerir. Kısıtlı modelde en az "
                             "bir değişken kalır (yalnız sabitli model genel testtir: Adım 4).",
                        maximum=2)


def _heading(terms: tuple[str, ...], restricted: bool) -> str:
    kind = "Kısıtlı model" if restricted else "Kısıtsız model"
    return f"{kind}: {_listing([WORDS[term] for term in terms])}"


def _restricted(choices) -> tuple:
    dropped = tuple(choices["adim3_x"])
    kept = tuple(term for term in REGRESSORS if term not in dropped)
    q = len(dropped)
    ssr = E.div(E.div(E.sub(E.ref("ssr_k"), E.ref("ssr_s")), E.ref("q3")), E.div(E.ref("ssr_s"), E.ref("sd_s")))
    r2 = E.div(E.div(E.sub(E.ref("r2_s"), E.ref("r2_k")), E.ref("q3")),
               E.div(E.sub(1, E.ref("r2_s")), E.ref("sd_s")))
    notes = dropped == TESTED
    return (
        OLS("mk", DATA, "wage", kept, f"Kısıtlı model: wage ~ {' + '.join(kept)}"),
        ModelValue("ssr_k", "mk", "ssr", "SSR_R: kısıtlı modelin artık kareleri toplamı", decimals=3),
        ModelValue("ssr_s", "m", "ssr", "SSR_UR: kısıtsız modelin artık kareleri toplamı", decimals=3),
        ModelValue("r2_k", "mk", "r2", "R²_R: kısıtlı modelin R²'si", decimals=5),
        ModelValue("r2_s", "m", "r2", "R²_UR: kısıtsız modelin R²'si", decimals=5),
        ModelValue("sd_s", "m", "df_resid", "Kısıtsız modelin artık serbestlik derecesi n − k − 1", decimals=0),
        Scalar("q3", E.const(q), "Kısıt sayısı q", decimals=0),
        Scalar("F_ssr", ssr, "F, SSR formülü: [(SSR_R − SSR_UR)/q] / [SSR_UR/(n − k − 1)]", decimals=2),
        Scalar("F_r2", r2, "F, R² formülü: [(R²_UR − R²_R)/q] / [(1 − R²_UR)/(n − k − 1)]", decimals=2),
        RegressionTable(((_heading(kept, True), "mk"), (_heading(REGRESSORS, False), "m")), (), "tablo81",
                        "Tablo 8.1: WAGE1 kısıtlı ve kısıtsız model karşılaştırması" if notes
                        else "WAGE1 kısıtlı ve kısıtsız model karşılaştırması",
                        stars=False, standard_errors=False, r2=False,
                        extra=(("ssr", "SSR", ("ssr_k", "ssr_s")), ("r2", "R²", ("r2_k", "r2_s"))), extra_decimals=3),
    )


def _restricted_note(state, choices) -> str:
    dropped = tuple(choices["adim3_x"])
    s = state.scalars
    q = len(dropped)
    return (f"SSR formülü: [({plain(s['ssr_k'], 3)} − {plain(s['ssr_s'], 3)})/{q}] / [{plain(s['ssr_s'], 3)}/"
            f"{plain(s['sd_s'], 0)}] = {plain(s['F_ssr'], 2)}. R² formülü: [({plain(s['r2_s'], 5)} − "
            f"{plain(s['r2_k'], 5)})/{q}] / [(1 − {plain(s['r2_s'], 5)})/{plain(s['sd_s'], 0)}] = "
            f"{plain(s['F_r2'], 2)}. İki formül aynı sonucu verir; çünkü iki model aynı bağımlı değişkeni ve aynı "
            "gözlemleri kullanır. Değişken çıkarmak uyumu artırmaz: kısıtlı modelin SSR'si büyük, R²'si küçüktür. "
            "Soru, bu uyum kaybının kısıtsız modeldeki artık değişkenliğine göre büyük olup olmadığıdır (§8.6). R² "
            "formülünde beş basamaklı R² kullanılır; üç basamağa yuvarlanmış R² ile hesap belirgin biçimde sapar.")


# --- Adım 4: genel anlamlılık testi ------------------------------------------------------------------------

MODEL_4 = MultiChoice("adim4_x", "Modeldeki açıklayıcı değişkenler", W.options(DATA, MODEL_OPTIONS), REGRESSORS,
                      help="Notlarda eğitim, deneyim ve kıdem.")


def _overall(choices) -> tuple:
    regressors = tuple(choices["adim4_x"])
    k = len(regressors)
    formula = E.div(E.div(E.ref("r2_genel"), E.ref("k_genel")),
                    E.div(E.sub(1, E.ref("r2_genel")), E.ref("sd_genel")))
    return (
        OLS("mg", DATA, "wage", regressors, f"Genel test modeli: wage ~ {' + '.join(regressors)}"),
        ModelValue("F_genel", "mg", "f", "Genel F istatistiği (F-statistic)", decimals=2),
        ModelValue("p_genel", "mg", "f_p", "Genel F testinin p-değeri (Prob (F-statistic))", decimals=3),
        ModelValue("r2_genel", "mg", "r2", "R²", decimals=3),
        ModelValue("n_genel", "mg", "nobs", "Gözlem sayısı", decimals=0),
        ModelValue("sd_genel", "mg", "df_resid", "Payda serbestlik derecesi n − k − 1", decimals=0),
        Scalar("k_genel", E.const(k), "Eğim sayısı k (genel testte kısıt sayısı q = k)", decimals=0),
        Scalar("F_genel_r2", formula, "F = (R²/k) / [(1 − R²)/(n − k − 1)]", decimals=2),
    )


def _overall_note(state, choices) -> str:
    regressors = tuple(choices["adim4_x"])
    s = state.scalars
    k, df = len(regressors), int(round(s["sd_genel"]))
    rejected = s["p_genel"] < 0.05
    names = _listing([WORDS[term] for term in regressors])
    text = (f"Genel test F({k}, {df}) = {plain(s['F_genel'], 2)}, {_p(s['p_genel'])}: {names} "
            f"{'katsayılarının tamamının birlikte' if k > 1 else 'katsayısının'} sıfır olduğu hipotezi yüzde 5 "
            f"düzeyinde {'reddedilir' if rejected else 'reddedilemez'}. Kısıtlı model yalnız sabit terimi içerir, "
            f"bu yüzden R²_R = 0 ve F = (R²/k)/[(1 − R²)/(n − k − 1)] = {plain(s['F_genel_r2'], 2)}. ")
    if k == 1:
        text += "Tek eğimli modelde genel F, eğim katsayısının t istatistiğinin karesidir (§8.8). "
    if rejected:
        return text + ("Genel testin reddedilmesi bütün eğimlerin ayrı ayrı anlamlı olduğunu ya da modelin doğru, "
                       "nedensel veya iyi belirlenmiş olduğunu göstermez (§8.7).")
    return text + ("Genel testin reddedilememesi eğimlerin sıfır olduğunu kanıtlamaz; veri, seçilen düzeyde yeterli "
                   "kanıt sunmamıştır (§8.7).")


# --- Adım 5: tek kısıtta F = t² -------------------------------------------------------------------------

TERM_5 = Choice("adim5_terim", "Sınanan katsayı", TERM_OPTIONS, "educ", help="Notlarda eğitim katsayısı (§8.8).")


def _single(choices) -> tuple:
    term = choices["adim5_terim"]
    return (
        ModelValue("t_tek", "m", "t", f"{WORDS[term].capitalize()} katsayısının t istatistiği", term=term, decimals=4),
        Scalar("t_kare", E.power(E.ref("t_tek"), 2), "t²", decimals=2),
        JointTest("F_tek", "p_tek", "m", (term,), f"Tek kısıtlı F testi, {_hypothesis((term,))}"),
        ModelValue("p_t_tek", "m", "p", "t testinin iki taraflı p-değeri", term=term, decimals=4),
    )


def _single_note(state, choices) -> str:
    term = choices["adim5_terim"]
    s = state.scalars
    return (f"t = {plain(s['t_tek'], 4)}, t² = {plain(s['t_kare'], 2)}; aynı hipotezin F testi F(1, 522) = "
            f"{plain(s['F_tek'], 2)}. p-değerleri de aynıdır: t testinde {_p(s['p_t_tek'], 4)}, F testinde "
            f"{_p(s['p_tek'], 4)}. Tek kısıtta iki taraflı t testi ile F testi aynı kararı verir. F işaret taşımaz; "
            f"{WORDS[term]} katsayısının yönünü t (ve katsayının kendisi) gösterir. Birden fazla kısıtta tek bir t "
            "istatistiği yoktur; ortak belirsizlik F testiyle değerlendirilir (§8.8).")


# --- Adım 6: HPRICE1 ortak testi ---------------------------------------------------------------------------

DROPPED_6 = MultiChoice("adim6_x", "Birlikte sınanan (kısıtlı modelden çıkarılan) değişkenler",
                        W.options(HPRICE, HOUSE_TERMS), HOUSE_TESTED,
                        help="Notlarda arsa büyüklüğü ve yatak odası sayısı; konut büyüklüğü modelde kalır.", maximum=2)


def _house(choices) -> tuple:
    dropped = tuple(choices["adim6_x"])
    kept = tuple(term for term in HOUSE_TERMS if term not in dropped)
    notes = dropped == HOUSE_TESTED
    return (
        LoadWooldridge(HPRICE, "HPRICE1 veri seti (Wooldridge, 2020): 88 konut"),
        OLS("h", HPRICE, "price", HOUSE_TERMS, "Kısıtsız model: price ~ lotsize + sqrft + bdrms"),
        OLS("hk", HPRICE, "price", kept, f"Kısıtlı model: price ~ {' + '.join(kept)}"),
        ModelValue("ssr_hk", "hk", "ssr", "SSR_R: kısıtlı model", decimals=3),
        ModelValue("ssr_h", "h", "ssr", "SSR_UR: kısıtsız model", decimals=3),
        ModelValue("r2_hk", "hk", "r2", "R²_R", decimals=3),
        ModelValue("r2_h", "h", "r2", "R²_UR", decimals=3),
        ModelValue("sd_h", "h", "df_resid", "Payda serbestlik derecesi n − k − 1", decimals=0),
        JointTest("F_h", "p_h", "h", dropped, f"Ortak F testi, {_hypothesis(dropped)}", decimals=2),
        ModelValue("p_yatak", "h", "p", "Yatak odası katsayısının ayrı p-değeri", term="bdrms", decimals=3),
        RegressionTable(((_heading(kept, True), "hk"), (_heading(HOUSE_TERMS, False), "h")), (), "tablo82",
                        "Tablo 8.2: HPRICE1 ortak test karşılaştırması" if notes
                        else "HPRICE1 kısıtlı ve kısıtsız model karşılaştırması",
                        stars=False, standard_errors=False, r2=False,
                        extra=(("ssr", "SSR", ("ssr_hk", "ssr_h")), ("r2", "R²", ("r2_hk", "r2_h"))),
                        extra_decimals=3),
    )


def _house_note(state, choices) -> str:
    dropped = tuple(choices["adim6_x"])
    kept = tuple(term for term in HOUSE_TERMS if term not in dropped)
    s = state.scalars
    q, df = len(dropped), int(round(s["sd_h"]))
    rejected = s["p_h"] < 0.05
    subject = _listing([WORDS[term] for term in dropped])
    text = (f"F({q}, {df}) = {plain(s['F_h'], 2)}, {_p(s['p_h'], 4)}: yüzde 5 düzeyinde H₀ "
            f"{'reddedilir' if rejected else 'reddedilemez'}. {_listing([WORDS[term] for term in kept]).capitalize()} "
            f"sabitken {subject} {'birlikte ' if q > 1 else ''}fiyat modeline "
            f"{'katkı sağlar' if rejected else 'istatistiksel olarak ayırt edilebilir katkı sağlamaz'}. ")
    if q == 1:
        return text + "Tek kısıtta bu test katsayının ayrı iki taraflı t testiyle aynı kararı verir: F = t² (§8.8)."
    if "bdrms" in dropped:
        text += (f"Yatak odası katsayısının ayrı {_p(s['p_yatak'])}; ortak test ile ayrı test farklı soruları "
                 "yanıtlar. ")
    if rejected:
        return text + ("Ortak ret, sınanan katsayıların hepsinin ayrı ayrı anlamlı olduğunu göstermez; nedensel etkiyi "
                       "de kanıtlamaz (§8.9).")
    return text + "Ortak testin reddedilememesi katsayıların birlikte sıfır olduğunu kanıtlamaz (§8.9)."


# --- Tanım ---------------------------------------------------------------------------------------------

_OUTPUT = (
    (INTERCEPT, "sabit terim", (-2.8727, 0.729, -3.941, 0.000)),
    ("educ", "educ", (0.5990, 0.051, 11.679, 0.000)),
    ("exper", "exper", (0.0223, 0.012, 1.853, 0.064)),
    ("tenure", "tenure", (0.1693, 0.022, 7.820, 0.000)),
)
_OUTPUT_QUANTITIES = (("coef", "coef", 4), ("se", "std err", 3), ("t", "t", 3), ("p", "P>|t|", 3))
_TABLE86 = (
    ("(1) Ücret", (("educ", 0.599), ("educ_sh", 0.051), ("exper", 0.022), ("exper_sh", 0.012), ("tenure", 0.169),
                   ("tenure_sh", 0.022), ("ortak_F", 53.31), ("genel_F", 76.87), ("n", 526), ("r2", 0.306))),
    ("(2) ln(Ücret)", (("educ", 0.092), ("educ_sh", 0.007), ("exper", 0.004), ("exper_sh", 0.002), ("tenure", 0.022),
                       ("tenure_sh", 0.003), ("ortak_F", 49.69), ("genel_F", 80.39), ("n", 526), ("r2", 0.316))),
)
_TABLE86_DECIMALS = {"ortak_F": 2, "genel_F": 2, "n": 0}
_HEADING_R = _heading(("educ",), True)
_HEADING_UR = _heading(REGRESSORS, False)
_HOUSE_R = _heading(("sqrft",), True)
_HOUSE_UR = _heading(HOUSE_TERMS, False)

STEPS = (
    interactive_step(
        number=1,
        title="Karar kuralı: kritik değer ve ortak p-değeri",
        note=NoteRef("8.5", 0, ("Şekil 8.1",)),
        explanation=(
            "F istatistiği kısıtların yarattığı uyum kaybını kısıtsız modeldeki artık değişkenliğine göre ölçer; büyük "
            "değerler $H_0$ aleyhine kanıttır ve reddetme bölgesi F dağılımının üst kuyruğundadır. Şekil 8.1'de WAGE1 "
            "ücret modelinde deneyim ve kıdem katsayılarının birlikte sıfır olduğu hipotezi sınanır: pay serbestlik "
            "derecesi $q = 2$, payda serbestlik derecesi $n - k - 1 = 522$. Birlikte sınanan katsayıları ve "
            "anlamlılık düzeyini değiştirin."
        ),
        controls=(TESTED_1, ALPHA_1),
        build=_decision,
        checks=(
            _scalar("pay_sd", 2, "§8.5: pay serbestlik derecesi", 0),
            _scalar("sd_artik", 522, "§8.5: payda serbestlik derecesi", 0),
            _scalar("kritik_F", 3.01, "Şekil 8.1: kritik değer", 2),
            _scalar("F_ortak", 53.31, "Şekil 8.1: gözlenen F", 2),
        ),
        note_for=lambda state, choices: _decision_note(state, choices),
    ),
    LabStep(
        number=2,
        title="Python çıktısında iki farklı F",
        note=NoteRef("8.6", 0, ("Kod 8.1", "Kod 8.2")),
        explanation=(
            "Statsmodels özetindeki `F-statistic` ve `Prob (F-statistic)` bütün eğimlerin birlikte sıfır olduğu genel "
            "anlamlılık testidir. Kullanıcının yazdığı kısıtlar ise `f_test` ile ayrıca sınanır: "
            "`model.f_test(\"exper = 0, tenure = 0\")` yalnız deneyim ve kıdem katsayılarının birlikte sıfır olduğu "
            "hipotezini sınar. Çok küçük p-değerleri çıktıda bilimsel gösterimle yazılır (3.41e-41 = 3,41 × 10⁻⁴¹)."
        ),
        operations=_output(),
        checks=(
            *(Check(f"Kod 8.2: {name}, {label}", CoefTarget("m", term, quantity), value, decimals)
              for term, name, values in _OUTPUT
              for (quantity, label, decimals), value in zip(_OUTPUT_QUANTITIES, values)),
            Check("Kod 8.2: No. Observations", ModelTarget("m", "nobs"), 526, 0),
            Check("Kod 8.2: Df Residuals", ModelTarget("m", "df_resid"), 522, 0),
            _scalar("sd_model", 3, "Kod 8.2: Df Model", 0),
            Check("Kod 8.2: R-squared", ModelTarget("m", "r2"), 0.306, 3),
            Check("Kod 8.2: F-statistic", ModelTarget("m", "f"), 76.87, 2),
            _scalar("genel_p_olcek", 3.41, "Kod 8.2: Prob (F-statistic) = 3.41e-41", 2),
            _scalar("F_cikti", 53.3099, "Kod 8.2: F test, F", 4),
            _scalar("p_cikti_olcek", 8.56, "Kod 8.2: F test, p = 8.56e-22", 2),
            _scalar("sd_m", 522, "Kod 8.2: F test, df_denom", 0),
            _scalar("F_cikti", 53.31, "§8.6: ortak F", 2),
        ),
        takeaway=(
            "Aynı çıktıda iki F vardır: 76,87 bütün eğimlerin (eğitim, deneyim, kıdem) birlikte sıfır olduğu genel "
            "testi, 53,31 yalnız deneyim ve kıdemin birlikte sıfır olduğu özel ortak testi sınar. İkisinin de p-değeri "
            "çok küçüktür (3,41 × 10⁻⁴¹ ve 8,56 × 10⁻²²); yüzde 5 düzeyinde iki hipotez de reddedilir. Deneyim "
            "katsayısının ayrı p-değeri 0,064 iken ortak testin reddedilmesi çelişki değildir: ortak test iki kısıtı "
            "bir bütün olarak değerlendirir (§8.6)."
        ),
    ),
    interactive_step(
        number=3,
        title="Kısıtlı ve kısıtsız model: iki formül aynı F'yi verir",
        note=NoteRef("8.6", 0, ("Tablo 8.1",)),
        explanation=(
            "Kısıtlı model sıfır hipotezini modele yükler: deneyim ve kıdem çıkarılır, model yalnız eğitimle tahmin "
            "edilir. F, SSR farkını kısıt sayısına ve kısıtsız modelin artık değişkenliğine göre ölçekler: "
            "$F = \\frac{(SSR_R - SSR_{UR})/q}{SSR_{UR}/(n-k-1)}$. Aynı bağımlı değişken ve aynı örneklemle R² "
            "biçimi de aynı değeri verir. Kısıtlı modelden çıkarılan değişkenleri değiştirin."
        ),
        controls=(DROPPED_3,),
        build=_restricted,
        checks=(
            Check("Tablo 8.1: kısıtlı model, SSR", TableTarget("tablo81", "ssr", _HEADING_R), 5980.682, 3),
            Check("Tablo 8.1: kısıtsız model, SSR", TableTarget("tablo81", "ssr", _HEADING_UR), 4966.303, 3),
            Check("Tablo 8.1: kısıtlı model, R²", TableTarget("tablo81", "r2", _HEADING_R), 0.165, 3),
            Check("Tablo 8.1: kısıtsız model, R²", TableTarget("tablo81", "r2", _HEADING_UR), 0.306, 3),
            Check("Tablo 8.1: kısıtlı model, gözlem sayısı", TableTarget("tablo81", "n", _HEADING_R), 526, 0),
            Check("Tablo 8.1: kısıtsız model, gözlem sayısı", TableTarget("tablo81", "n", _HEADING_UR), 526, 0),
            _scalar("r2_s", 0.30642, "§8.6: R²_UR (beş basamak)", 5),
            _scalar("r2_k", 0.16476, "§8.6: R²_R (beş basamak)", 5),
            _scalar("sd_s", 522, "§8.6: 526 − 3 − 1", 0),
            _scalar("q3", 2, "§8.6: kısıt sayısı q", 0),
            _scalar("F_ssr", 53.31, "§8.6: F, SSR formülü", 2),
            _scalar("F_r2", 53.31, "§8.6: F, R² formülü", 2),
        ),
        note_for=lambda state, choices: _restricted_note(state, choices),
    ),
    interactive_step(
        number=4,
        title="Genel anlamlılık testi",
        note=NoteRef("8.7", 0),
        explanation=(
            "Genel test bütün eğim katsayılarının birlikte sıfır olduğu hipotezini sınar: $H_0: \\beta_1 = \\beta_2 = "
            "\\cdots = \\beta_k = 0$. Kısıtlı model yalnız sabit terimi içerdiği için $R^2_R = 0$ ve "
            "$F = \\frac{R^2/k}{(1-R^2)/(n-k-1)}$. Modeldeki açıklayıcı değişkenleri değiştirin: aynı R² farklı k ile "
            "farklı kanıt üretir."
        ),
        controls=(MODEL_4,),
        build=_overall,
        checks=(
            _scalar("F_genel", 76.87, "§8.7: genel F", 2),
            _scalar("F_genel_r2", 76.87, "§8.7: R² biçimiyle genel F", 2),
            _scalar("sd_genel", 522, "§8.7: F(3, 522), payda serbestlik derecesi", 0),
            _scalar("k_genel", 3, "§8.7: F(3, 522), pay serbestlik derecesi", 0),
            _scalar("n_genel", 526, "§8.7: makale tablosunda gözlem sayısı", 0),
            _scalar("r2_genel", 0.306, "§8.7: makale tablosunda R²", 3),
        ),
        note_for=lambda state, choices: _overall_note(state, choices),
    ),
    interactive_step(
        number=5,
        title="Tek kısıtta F = t²",
        note=NoteRef("8.8", 0),
        explanation=(
            "Ortak hipotez tek kısıt içeriyorsa geleneksel EKK çıkarımında F testi ile iki taraflı t testi aynı kararı "
            "verir: $F(1, n-k-1) = t^2$. Notlarda eğitim katsayısı için $t = 11{,}6795$. Sınanan katsayıyı "
            "değiştirin: F her seçimde t'nin karesi mi?"
        ),
        controls=(TERM_5,),
        build=_single,
        checks=(
            _scalar("t_tek", 11.6795, "§8.8: t", 4),
            _scalar("t_kare", 136.41, "§8.8: t²", 2),
            _scalar("F_tek", 136.41, "§8.8: F(1, 522)", 2),
        ),
        note_for=lambda state, choices: _single_note(state, choices),
    ),
    interactive_step(
        number=6,
        title="HPRICE1: birlikte katkı ayrı testlerden farklıdır",
        note=NoteRef("8.9", 0, ("Tablo 8.2",)),
        explanation=(
            "Konut fiyatı modelinde konut büyüklüğü tutulurken arsa büyüklüğü ve yatak odası sayısının birlikte "
            "gereksiz olduğu hipotezi sınanır: $H_0: \\beta_{\\mathrm{arsa}} = 0,\\ \\beta_{\\mathrm{yatak}} = 0$. "
            "Kısıtlı model yalnız konut büyüklüğünü içerir. Yatak odası katsayısı ayrı testte anlamlı değildir; ortak "
            "test farklı bir soruyu yanıtlar. Birlikte sınanan değişkenleri değiştirin."
        ),
        controls=(DROPPED_6,),
        build=_house,
        checks=(
            Check("Tablo 8.2: kısıtlı model, SSR", TableTarget("tablo82", "ssr", _HOUSE_R), 348053.432, 3),
            Check("Tablo 8.2: kısıtsız model, SSR", TableTarget("tablo82", "ssr", _HOUSE_UR), 300723.805, 3),
            Check("Tablo 8.2: kısıtlı model, R²", TableTarget("tablo82", "r2", _HOUSE_R), 0.621, 3),
            Check("Tablo 8.2: kısıtsız model, R²", TableTarget("tablo82", "r2", _HOUSE_UR), 0.672, 3),
            Check("Tablo 8.2: kısıtlı model, gözlem sayısı", TableTarget("tablo82", "n", _HOUSE_R), 88, 0),
            Check("Tablo 8.2: kısıtsız model, gözlem sayısı", TableTarget("tablo82", "n", _HOUSE_UR), 88, 0),
            _scalar("sd_h", 84, "§8.9: F(2, 84), payda serbestlik derecesi", 0),
            _scalar("F_h", 6.61, "§8.9: F(2, 84)", 2),
            _scalar("p_h", 0.0022, "§8.9: ortak p-değeri", 4),
            _scalar("p_yatak", 0.128, "§8.9: yatak odası katsayısının ayrı p-değeri", 3),
        ),
        note_for=lambda state, choices: _house_note(state, choices),
    ),
    LabStep(
        number=7,
        title="Makale tablosunda ortak testler",
        note=NoteRef("8.12", 0, ("Tablo 8.6",)),
        explanation=(
            "Makale tablolarında ortak testler katsayıların altında ayrı satırlarda verilir: sınanan hipotez ve "
            "standart hata türü tablo notunda yazılır. Tablo 8.6'da Sütun (1) düzey–düzey, Sütun (2) log–düzey ücret "
            "modelidir; ortak test deneyim ve kıdem katsayılarının birlikte sıfır olduğunu, genel F bütün eğimleri "
            "sınar."
        ),
        operations=(
            OLS("ml", DATA, "lwage", REGRESSORS, "Log ücret modeli: lwage ~ educ + exper + tenure"),
            JointTest("ortak_F1", "ortak_p1", "m", TESTED, "Sütun (1): deneyim ve kıdem ortak F testi"),
            JointTest("ortak_F2", "ortak_p2", "ml", TESTED, "Sütun (2): deneyim ve kıdem ortak F testi"),
            ModelValue("genel_F1", "m", "f", "Sütun (1): genel F", decimals=2),
            ModelValue("genel_F2", "ml", "f", "Sütun (2): genel F", decimals=2),
            RegressionTable((("(1) Ücret", "m"), ("(2) ln(Ücret)", "ml")), REGRESSORS, "tablo86",
                            "Tablo 8.6: ücret modellerinde bireysel ve ortak çıkarım",
                            extra=(("ortak_F", "Deneyim ve kıdem ortak F", ("ortak_F1", "ortak_F2")),
                                   ("ortak_p", "Ortak test p-değeri", ("ortak_p1", "ortak_p2")),
                                   ("genel_F", "Genel F", ("genel_F1", "genel_F2")))),
        ),
        checks=tuple(Check(f"Tablo 8.6: {heading}, {row}", TableTarget("tablo86", row, heading), value,
                           _TABLE86_DECIMALS.get(row, 3))
                     for heading, values in _TABLE86 for row, value in values),
        takeaway=(
            "Sütun (1)'de deneyim yüzde 5 düzeyinde tek başına anlamlı değildir (tek yıldız: p < 0,10); buna karşılık "
            "deneyim ve kıdem birlikte güçlü biçimde anlamlıdır (F = 53,31; p < 0,001). Ortak test ayrı yıldızların "
            "basit toplamı değildir. Sütun (2)'de bağımlı değişken log ücret olduğu için katsayılar yaklaşık yüzde "
            "farklar olarak okunur; iki sütunun katsayı büyüklükleri doğrudan karşılaştırılamaz. Ortak testin "
            "reddedilmesi nedensellik için yeterli değildir (§8.12)."
        ),
    ),
)


KONU08_LAB = LabSpec(
    topic_key="konu08",
    title="Uygulama: Çoklu Kısıtlar ve F Testi",
    note_section="8",
    steps=STEPS,
    labels=(
        *W.labels(DATA, HPRICE),
        (INTERCEPT, "Sabit terim"),
    ),
    consistency_notes=(
        "Veri notlardaki gibi WAGE1 ve HPRICE1'dir; notlardaki kod, betikler, uygulama ve üretilen kod onları wooldridge "
        "paketinden okur. HPRICE1 modellerinin artık kareleri toplamı (348053,432 ve 300723,805) paket verisiyle "
        "hesaplanmıştır.",
        "Tablo 8.6'daki log ücret modelinde deneyim ve kıdemin ortak F değeri 49,69'dur (eski metin 29,44 yazıyordu); "
        "notlar ve sunum düzeltildi.",
        "§8.6'daki R² formülü beş basamaklı R² ile yazılır (0,30642 ve 0,16476); üç basamaklı değerlerle hesap 53,31'i "
        "vermez. §8.8'de t dört basamakla (11,6795) yazılır ve t² = F = 136,41.",
        "Bölümün benzetimi (Tablo 8.3, Şekil 8.2–8.3) tohum 305 ile yeniden üretildi ve Sezgi sekmesindeki Konu 8 "
        "Deney 1'in varsayılan ayarlarıdır.",
        "Ortak testler klasik EKK varsayımlarına dayanır; heteroskedastisiteye dayanıklı ortak test Konu 12'de.",
    ),
)
