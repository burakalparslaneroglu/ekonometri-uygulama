"""Konu 7 uygulaması: tek katsayı için hipotez testleri (standart hata, t istatistiği, p-değeri, güven aralığı).

Bölüm 7'nin çözümlü örnekleri bölüm sırasıyla: t istatistiği ve kritik değerler (§7.4, Tablo 7.3), WAGE1'de eğitim
katsayısının t testi (§7.4), deneyim katsayısının p-değeri (§7.5, Şekil 7.2), güven aralığının sayısal örneği (§7.6),
test ile güven aralığının eşdeğerliği (§7.7, Tablo 7.4, Şekil 7.4), tek taraflı test (§7.8), Python çıktısı (§7.9),
makale tablosu (§7.10, Tablo 7.5), istatistiksel anlamlılık ile iktisadi önem (§7.11, Tablo 7.6) ve raporlama
(§7.13). Bölümün benzetimi (Tablo 7.1, Şekil 7.1 ve 7.3) Sezgi sekmesindeki Konu 7 Deney 1'dir. Her ``Check`` notlarda
basılı bir sayıdır; değer notlardan kopyalanmıştır, hesaplanmamıştır.

Etkileşim: sınanan katsayı, sıfır hipotezindeki değer a, anlamlılık düzeyi, güven düzeyi, alternatif hipotezin yönü,
yazılım çıktısının modeli ve iktisadi önemde eğitim farkı. Standart hatalar klasik (homoskedastik) EKK standart
hatalarıdır; dayanıklı standart hatalar Konu 12'nin konusudur. F istatistiği Konu 8'de yorumlanır.
"""

from __future__ import annotations

from core import wooldridge_data as W
from core.labs import expr as E
from core.labs.sezgi import plain
from core.labs.spec import (
    COEF_QUANTITIES,
    INTERCEPT,
    OLS,
    Check,
    Choice,
    CoefficientPlot,
    CoefficientTable,
    CoefTarget,
    HypothesisPlot,
    LabSpec,
    LabStep,
    LoadWooldridge,
    ModelTarget,
    ModelValue,
    MultiChoice,
    NoteRef,
    NumberChoice,
    RegressionTable,
    Scalar,
    ScalarTable,
    ScalarTarget,
    ShowModel,
    TableTarget,
    interactive_step,
)

DATA = "wage1"
HPRICE = "hprice1"
REGRESSORS = ("educ", "exper", "tenure")
"""Bölüm 7'nin ücret modeli: wage ~ educ + exper + tenure (n = 526)."""
TERM_OPTIONS = W.options(DATA, REGRESSORS)
WORDS = {"educ": "eğitim", "exper": "deneyim", "tenure": "kıdem"}
TITLES = {"educ": "Eğitim", "exper": "Deneyim", "tenure": "Kıdem"}
GENITIVE = {"educ": "eğitimin", "exper": "deneyimin", "tenure": "kıdemin"}
OTHERS = {"educ": "deneyim ve kıdem", "exper": "eğitim ve kıdem", "tenure": "eğitim ve deneyim"}
ALPHAS = (("0.1", "%10"), ("0.05", "%5"), ("0.01", "%1"))
LEVELS = (("0.9", "%90"), ("0.95", "%95"), ("0.99", "%99"))
DIRECTIONS = (("sag", "Sağ kuyruk: H₁: β > 0"), ("sol", "Sol kuyruk: H₁: β < 0"), ("iki", "İki taraflı: H₁: β ≠ 0"))
CRITICAL_DF = (("10", 10), ("20", 20), ("30", 30), ("60", 60), ("120", 120))
"""Tablo 7.3'ün serbestlik dereceleri; son satır ("Çok büyük") standart normal kritik değeridir."""
OUTPUT_OPTIONS = ("educ", "exper", "tenure", "numdep")
"""Yazılım çıktısı adımında seçilebilen açıklayıcı değişkenler (kukla olmayan sayısal değişkenler)."""


def _scalar(name: str, expected: float, label: str, decimals: int) -> Check:
    return Check(label, ScalarTarget(name), expected, decimals)


def _value(value: float) -> str:
    """Sıfır hipotezindeki değerin yazımı: tam sayıysa ondalıksız (0), değilse iki basamak (0,50)."""

    return plain(value, 0) if float(value).is_integer() else plain(value, 2)


def _percent(value: float) -> str:
    """Anlamlılık ya da güven düzeyinin yazımı: %5, %2,5."""

    return "%" + plain(100 * value, 0 if float(round(100 * value, 8)).is_integer() else 1)


def _p(value: float) -> str:
    """p-değerinin eşitlikli yazımı: "p = 0,064"; üç basamakta 0 ya da 1 görünüyorsa "p < 0,001" ya da "p > 0,999"."""

    if value < 0.0005:
        return "p < 0,001"
    return "p > 0,999" if value > 0.9995 else f"p = {plain(value, 3)}"


def _load() -> tuple:
    return (LoadWooldridge(DATA, "WAGE1 veri seti (Wooldridge, 2020): 526 çalışan"),
            OLS("m", DATA, "wage", REGRESSORS, "Ücret modeli: wage ~ educ + exper + tenure"))


# --- Adım 1: t istatistiği ve kritik değerler ----------------------------------------------------------

B_CHOICE = NumberChoice("adim1_b", "Katsayı tahmini β̂", -2.0, 2.0, 0.30, 0.01, help="Notlarda 0,30.")
SE_CHOICE = NumberChoice("adim1_se", "Standart hata se(β̂)", 0.01, 1.0, 0.12, 0.01,
                         help="Notlarda 0,12; 0,30 seçilirse t = 1 olur.")
A_CHOICE = NumberChoice("adim1_a", "Sıfır hipotezindeki değer a", -1.0, 1.0, 0.0, 0.05, help="Notlarda H₀: β = 0.")
ALPHA_1 = Choice("adim1_alfa", "Anlamlılık düzeyi α (iki taraflı)", ALPHAS, "0.05",
                 help="Tablo 7.3 yüzde 5 iki taraflı test içindir: t₀,₀₂₅.")


def _t_statistic(choices) -> tuple:
    alpha = float(choices["adim1_alfa"])
    q = round(1 - alpha / 2, 6)
    return (
        Scalar("b_ornek", E.const(choices["adim1_b"]), "Katsayı tahmini β̂", decimals=2),
        Scalar("sh_ornek", E.const(choices["adim1_se"]), "Standart hata se(β̂)", decimals=2),
        Scalar("t_ornek", E.div(E.sub(E.ref("b_ornek"), choices["adim1_a"]), E.ref("sh_ornek")),
               f"t = (β̂ − a)/se(β̂), a = {_value(choices['adim1_a'])}", decimals=2),
        Scalar("kritik_buyuk", E.norminv(q), "Büyük örneklem kritik değeri", decimals=3),
        ScalarTable((*((label, E.tinv(q, df)) for label, df in CRITICAL_DF), ("Çok büyük", E.norminv(q))),
                    "kritik_degerler", decimals=3, heading="Serbestlik derecesi",
                    value=f"Kritik değer (yüzde {plain(100 * alpha, 0)} iki taraflı)"),
    )


def _t_note(state, choices) -> str:
    s = state.scalars
    t, critical = s["t_ornek"], s["kritik_buyuk"]
    a = _value(choices["adim1_a"])
    level = _percent(float(choices["adim1_alfa"]))
    decision = "reddedilir" if abs(t) > critical else "reddedilemez"
    text = (f"Tahmin, sıfır hipotezindeki değerden (a = {a}) |t| = {plain(abs(t), 2)} standart hata uzaktadır. Büyük "
            f"örneklemde {level} iki taraflı testin kritik değeri {plain(critical, 2)}: H₀ {decision} (Denklem 7.6). "
            "Serbestlik derecesi büyüdükçe kritik değer standart normal değere yaklaşır (Tablo 7.3).")
    if choices["adim1_se"] == 0.12 and choices["adim1_b"] == 0.30 and choices["adim1_a"] == 0:
        text += " Aynı tahminin standart hatası 0,30 olsaydı t = 1 olurdu: veri sıfırdan ayrışma konusunda daha zayıf kanıt sunardı."
    return text


# --- Adım 2: WAGE1, eğitim katsayısı için t testi --------------------------------------------------------

TERM_2 = Choice("adim2_terim", "Sınanan katsayı", TERM_OPTIONS, "educ", help="Notlarda eğitim katsayısı.")
A_2 = NumberChoice("adim2_a", "İkinci sıfır hipotezindeki değer a", -1.0, 1.0, 0.50, 0.01,
                   help="Notlarda H₀: β = 0,50 (eğitim).")


def _wage_t(choices) -> tuple:
    term, a = choices["adim2_terim"], choices["adim2_a"]
    return (
        *_load(),
        ModelValue("b2", "m", "coef", f"{TITLES[term]} katsayısı β̂", term=term, decimals=3),
        ModelValue("sh2", "m", "se", "Standart hata se(β̂)", term=term, decimals=4),
        ModelValue("sd_artik", "m", "df_resid", "Artık serbestlik derecesi n − k − 1", decimals=0),
        Scalar("t0_2", E.div(E.ref("b2"), E.ref("sh2")), "t, H₀: β = 0", decimals=2),
        Scalar("ta_2", E.div(E.sub(E.ref("b2"), a), E.ref("sh2")), f"t, H₀: β = {_value(a)}", decimals=2),
        Scalar("kritik_2", E.tinv(0.975, E.ref("sd_artik")), "Kritik değer t₀,₀₂₅ (sd = 522)", decimals=3),
    )


def _wage_t_note(state, choices) -> str:
    term, a = choices["adim2_terim"], choices["adim2_a"]
    s = state.scalars
    critical = s["kritik_2"]

    def decision(t: float) -> str:
        return "reddedilir" if abs(t) > critical else "reddedilemez"

    return (f"H₀: β = 0 için t = {plain(s['t0_2'], 2)}: yüzde 5 iki taraflı testte H₀ {decision(s['t0_2'])}. "
            f"H₀: β = {_value(a)} için t = {plain(s['ta_2'], 2)}: H₀ {decision(s['ta_2'])} (kritik değer "
            f"{plain(critical, 3)}). Hipotez testinin cevabı sınanan değere bağlıdır: katsayının sıfırdan farklı olması "
            f"ile belirli bir değerden farklı olması aynı sonuç değildir (§7.4). Sonuç {OTHERS[term]} sabitken "
            f"{GENITIVE[term]} ücretle doğrusal ilişkisi hakkındadır; gözlemsel veride test tek başına nedensel etkiyi "
            "kanıtlamaz.")


# --- Adım 3: p-değeri --------------------------------------------------------------------------------------

TERM_3 = Choice("adim3_terim", "Sınanan katsayı", TERM_OPTIONS, "exper", help="Notlarda deneyim katsayısı (§7.5).")
ALPHA_3 = Choice("adim3_alfa", "Anlamlılık düzeyi α", ALPHAS, "0.05", help="Şekil 7.2 yüzde 5 düzeyindedir.")


def _p_value(choices) -> tuple:
    term, alpha = choices["adim3_terim"], float(choices["adim3_alfa"])
    return (
        ModelValue("b3", "m", "coef", f"{TITLES[term]} katsayısı β̂", term=term, decimals=4),
        ModelValue("sh3", "m", "se", "Standart hata se(β̂)", term=term, decimals=4),
        ModelValue("t3", "m", "t", "t istatistiği (H₀: β = 0)", term=term, decimals=3),
        ModelValue("p3", "m", "p", "İki taraflı p-değeri", term=term, decimals=3),
        HypothesisPlot("t", "t3", "sd_artik",
                       f"Şekil 7.2: {WORDS[term]} katsayısı için iki taraflı kritik bölgeler ve p-değeri alanı"
                       if (term, alpha) == ("exper", 0.05)
                       else f"{TITLES[term]} katsayısı: yüzde {plain(100 * alpha, 0)} iki taraflı test",
                       "t değeri (serbestlik derecesi 522)", alpha=alpha),
    )


def _p_note(state, choices) -> str:
    term, alpha = choices["adim3_terim"], float(choices["adim3_alfa"])
    s = state.scalars
    p = s["p3"]
    levels = [(0.10, "%10"), (0.05, "%5"), (0.01, "%1")]
    rejected = [label for level, label in levels if p < level]
    kept = [label for level, label in levels if p >= level]
    parts = []
    if rejected:
        parts.append(f"{', '.join(rejected)} düzeyinde reddedilir")
    if kept:
        parts.append(f"{', '.join(kept)} düzeyinde reddedilemez")
    decision = "p < α olduğundan H₀: β = 0 reddedilir" if p < alpha else "p ≥ α olduğundan H₀: β = 0 reddedilemez"
    text = (f"{TITLES[term]} katsayısı {plain(s['b3'], 4)}, t = {plain(s['t3'], 3)}, iki taraflı {_p(p)}. Seçilen "
            f"{_percent(alpha)} düzeyinde {decision} (Denklem 7.7). Üç yaygın düzeyde H₀: " + "; ".join(parts) + ". ")
    if p >= alpha:
        text += (f"Reddedememek {GENITIVE[term]} ücretle hiçbir ilişkisi olmadığı anlamına gelmez; seçilen düzeyde "
                 "belirsizlik yeterince küçülmemiştir. ")
    return text + ("p-değeri H₀'ın doğru olma olasılığı değildir: H₀ ve model varsayımları doğruyken gözlenen kadar uç "
                   "bir t istatistiği elde etme olasılığıdır (§7.5).")


# --- Adım 4: güven aralığı, sayısal örnek ---------------------------------------------------------------

B_4 = NumberChoice("adim4_b", "Katsayı tahmini β̂", -2.0, 2.0, 0.30, 0.01, help="Notlarda 0,30.")
SE_4 = NumberChoice("adim4_se", "Standart hata se(β̂)", 0.01, 1.0, 0.12, 0.01, help="Notlarda 0,12.")
LEVEL_4 = Choice("adim4_duzey", "Güven düzeyi", LEVELS, "0.95", help="Notlarda yüzde 95 (kritik değer 1,96).")


def _interval_example(choices) -> tuple:
    level = float(choices["adim4_duzey"])
    q = round((1 + level) / 2, 6)
    return (
        Scalar("tahmin4", E.const(choices["adim4_b"]), "Katsayı tahmini β̂", decimals=2),
        Scalar("sh4", E.const(choices["adim4_se"]), "Standart hata se(β̂)", decimals=2),
        Scalar("kritik4", E.norminv(q), f"Büyük örneklem kritik değeri (yüzde {plain(100 * level, 0)})", decimals=3),
        Scalar("alt4", E.sub(E.ref("tahmin4"), E.mul(E.ref("kritik4"), E.ref("sh4"))), "Alt sınır", decimals=4),
        Scalar("ust4", E.add(E.ref("tahmin4"), E.mul(E.ref("kritik4"), E.ref("sh4"))), "Üst sınır", decimals=4),
    )


def _interval_note(state, choices) -> str:
    s = state.scalars
    level = _percent(float(choices["adim4_duzey"]))
    zero = s["alt4"] <= 0 <= s["ust4"]
    return (f"{level} güven aralığı [{plain(s['alt4'], 3)}; {plain(s['ust4'], 3)}]: yarı genişlik kritik değer × standart "
            f"hata = {plain(s['kritik4'] * s['sh4'], 4)}. Aralık sıfırı {'kapsar' if zero else 'kapsamaz'}. Güven "
            "düzeyi yükseldikçe kritik değer ve aralık büyür; standart hata küçüldükçe aralık daralır. Yorum tekrarlı "
            "örneklemeye dayanır: aynı yöntemle kurulan aralıkların uzun dönemde yaklaşık bu oranı gerçek parametreyi "
            "kapsar (§7.6).")


# --- Adım 5: güven aralığı ile test (WAGE1) ------------------------------------------------------------

LEVEL_5 = Choice("adim5_duzey", "Güven düzeyi", LEVELS, "0.95", help="Notlarda yüzde 95 (Tablo 7.4).")


def _equivalence(choices) -> tuple:
    level = float(choices["adim5_duzey"])
    notes = level == 0.95
    shown = plain(100 * level, 0)
    return (
        CoefficientTable("m", REGRESSORS, "tablo74",
                         "Tablo 7.4: WAGE1 ücret modelinde tek katsayı çıkarımı" if notes
                         else f"WAGE1 ücret modelinde tek katsayı çıkarımı, yüzde {shown} güven aralığı", level=level),
        CoefficientPlot("m", REGRESSORS, "Şekil 7.4: WAGE1 katsayı tahminleri ve yüzde 95 güven aralıkları" if notes
                        else f"WAGE1 katsayı tahminleri ve yüzde {shown} güven aralıkları",
                        f"Katsayı tahmini ve yüzde {shown} güven aralığı", level=level),
    )


def _equivalence_note(state, choices) -> str:
    level = float(choices["adim5_duzey"])
    table = state.tables["tablo74"]
    low, high = table.loc["educ", "alt"], table.loc["educ", "ust"]
    alpha = _percent(round(1 - level, 10))
    checks = []
    for value in (0.0, 0.50, 0.75):
        inside = low <= value <= high
        checks.append(f"{_value(value)} aralıkta {'olduğu için H₀ reddedilemez' if inside else 'olmadığı için H₀ reddedilir'}")
    crosses = [TITLES[term] for term in REGRESSORS if table.loc[term, "alt"] <= 0 <= table.loc[term, "ust"]]
    text = (f"Eğitim katsayısının {_percent(level)} güven aralığı [{plain(low, 3)}; {plain(high, 3)}]. {alpha} iki "
            f"taraflı H₀: β = a testinde: " + "; ".join(checks) + " (Denklem 7.9). ")
    if crosses:
        text += f"Sıfırı kapsayan aralık: {', '.join(crosses).lower()}; bu katsayı {alpha} düzeyinde sıfırdan ayrıştırılamaz. "
    else:
        text += "Hiçbir aralık sıfırı kapsamıyor. "
    return text + ("Katsayılar farklı birimlerle ölçüldüğü için grafikteki yatay konumlar büyüklük sıralaması olarak "
                   "okunmamalıdır (§7.7).")


# --- Adım 6: tek taraflı test -------------------------------------------------------------------------

TERM_6 = Choice("adim6_terim", "Sınanan katsayı", TERM_OPTIONS, "exper", help="Notlarda deneyim katsayısı (§7.8).")
DIRECTION_6 = Choice("adim6_yon", "Alternatif hipotez", DIRECTIONS, "sag", help="Notlarda H₁: β_deneyim > 0.")


def _one_sided(choices) -> tuple:
    term, direction = choices["adim6_terim"], choices["adim6_yon"]
    t, df = E.ref("t6"), E.ref("sd_artik")
    p_value = {"sag": E.tsf(t, df), "sol": E.tcdf(t, df), "iki": E.mul(2, E.tsf(E.absolute(t), df))}[direction]
    label = {"sag": "Tek taraflı p = P(T > t)", "sol": "Tek taraflı p = P(T < t)",
             "iki": "İki taraflı p = 2·P(T > |t|)"}[direction]
    return (
        ModelValue("t6", "m", "t", f"{TITLES[term]} katsayısının t istatistiği", term=term, decimals=3),
        Scalar("p_secilen", p_value, label, decimals=3, p_value=True),
        Scalar("p_iki6", E.mul(2, E.tsf(E.absolute(t), df)), "İki taraflı p = 2·P(T > |t|)", decimals=3, p_value=True),
        HypothesisPlot("t", "t6", "sd_artik",
                       "Deneyim katsayısı: yüzde 5 sağ kuyruk testi" if (term, direction) == ("exper", "sag")
                       else f"{TITLES[term]} katsayısı: yüzde 5 " + {"sag": "sağ kuyruk testi", "sol": "sol kuyruk testi",
                                                                        "iki": "iki taraflı test"}[direction],
                       "t değeri (serbestlik derecesi 522)", alternative=direction),
    )


def _one_sided_note(state, choices) -> str:
    term, direction = choices["adim6_terim"], choices["adim6_yon"]
    s = state.scalars
    chosen, both = s["p_secilen"], s["p_iki6"]
    if direction == "iki":
        return (f"{TITLES[term]} katsayısı için iki taraflı testte {_p(both)}. Sağ ya da sol kuyruk seçin: tek "
                "taraflı p-değeri, gözlenen t seçilen yöndeyse iki taraflı değerin yarısıdır, ters yöndeyse değildir "
                "(§7.8).")
    same = (s["t6"] > 0) == (direction == "sag")
    text = (f"{TITLES[term]} katsayısı için tek taraflı {_p(chosen)}, iki taraflı {_p(both)}. Yüzde 5 tek "
            "taraflı testte H₀ "
            f"{'reddedilir' if chosen < 0.05 else 'reddedilemez'}; iki taraflı testte "
            f"{'reddedilir' if both < 0.05 else 'reddedilemez'}. ")
    if not same:
        text += ("Gözlenen t alternatifin tersi yönde: tek taraflı p-değeri büyüktür; test ters yöndeki büyük bir "
                 "etkiyi alternatifin kanıtı saymaz. ")
    return text + ("Test yönü sonuç görüldükten sonra değil, teori veya önceden belirlenmiş analiz planıyla seçilir; "
                   "daha küçük p-değeri için yön değiştirmek uygun değildir (§7.8).")


# --- Adım 7: Python çıktısı ----------------------------------------------------------------------------

OUTCOME_7 = Choice("adim7_bagimli", "Bağımlı değişken", W.options(DATA, ("wage", "lwage")), "wage",
                   help="Notlarda saatlik ücret.")
REGRESSORS_7 = MultiChoice("adim7_x", "Açıklayıcı değişkenler", W.options(DATA, OUTPUT_OPTIONS), REGRESSORS,
                           help="Notlarda eğitim, deneyim ve kıdem.")


def _output(choices) -> tuple:
    outcome, regressors = choices["adim7_bagimli"], tuple(choices["adim7_x"])
    return (
        OLS("cikti", DATA, outcome, regressors, f"Python çıktısı: {outcome} ~ {' + '.join(regressors)}"),
        ShowModel("cikti", "Statsmodels çıktısından çıkarım için temel bölüm", columns=COEF_QUANTITIES,
                  stats=("nobs", "df_resid", "r2")),
    )


def _output_note(state, choices) -> str:
    result = state.models["cikti"]
    k = len(choices["adim7_x"])
    n, df = int(result.nobs), int(result.df_resid)
    significant = [W.variable(DATA, term).label.lower() for term in choices["adim7_x"] if result.pvalues[term] < 0.05]
    text = (f"Artık serbestlik derecesi n − k − 1 = {n} − {k} − 1 = {df}. `coef` katsayı tahmini, `std err` "
            "klasik EKK standart hatası, `t` ve `P>|t|` H₀: β = 0 için t istatistiği ve iki taraflı p-değeri, "
            "`[0.025 0.975]` yüzde 95 güven aralığıdır. ")
    text += (f"Yüzde 5 düzeyinde sıfırdan farklı eğimler: {', '.join(significant)}. " if significant
             else "Yüzde 5 düzeyinde sıfırdan farklı eğim yok. ")
    return text + ("`P>|t|` sütunundaki 0.000 gerçek sıfır değildir; gösterim hassasiyetinden küçüktür (p < 0,001). "
                   "\"Covariance Type: nonrobust\" standart hataların homoskedastisiteye dayandığını bildirir (§7.9).")


# --- Adım 8: makale tablosu ----------------------------------------------------------------------------

def _paper() -> tuple:
    return (
        OLS("ml", DATA, "lwage", REGRESSORS, "Log ücret modeli: lwage ~ educ + exper + tenure"),
        RegressionTable((("(1) Ücret", "m"), ("(2) ln(Ücret)", "ml")), (*REGRESSORS, INTERCEPT), "tablo75",
                        "Tablo 7.5: ücret modelleri için örnek makale tablosu"),
    )


# --- Adım 9: istatistiksel anlamlılık ve iktisadi önem ---------------------------------------------------

YEARS_9 = NumberChoice("adim9_fark", "Eğitim farkı (yıl)", 1, 8, 4, 1, help="Notlarda dört yıl.", integer=True,
                       decimals=0)
HOUSE_TERMS = ("lotsize", "sqrft", "bdrms")


def _magnitude(choices) -> tuple:
    years = choices["adim9_fark"]
    return (
        ModelValue("egitim_alt", "m", "ci_low", "Eğitim katsayısının yüzde 95 GA alt sınırı", term="educ"),
        ModelValue("egitim_ust", "m", "ci_high", "Eğitim katsayısının yüzde 95 GA üst sınırı", term="educ"),
        ModelValue("egitim9", "m", "coef", "Eğitim katsayısı", term="educ", decimals=3),
        Scalar("fark_tahmin", E.mul(years, E.ref("egitim9")), f"{years} yıllık farkın nokta tahmini (dolar)",
               decimals=3),
        Scalar("fark_alt", E.mul(years, E.ref("egitim_alt")), "Aralığın alt sınırı (dolar)", decimals=3),
        Scalar("fark_ust", E.mul(years, E.ref("egitim_ust")), "Aralığın üst sınırı (dolar)", decimals=3),
        LoadWooldridge(HPRICE, "HPRICE1 veri seti (Wooldridge, 2020): 88 konut"),
        OLS("h", HPRICE, "price", HOUSE_TERMS, "Konut fiyatı modeli: price ~ lotsize + sqrft + bdrms"),
        CoefficientTable("h", HOUSE_TERMS, "tablo76", "Tablo 7.6: HPRICE1 modelinde seçilmiş katsayılar", decimals=4),
    )


def _magnitude_note(state, choices) -> str:
    years = choices["adim9_fark"]
    s = state.scalars
    table = state.tables["tablo76"]
    beds = table.loc["bdrms"]
    return (f"{years} yıllık eğitim farkının nokta tahmini {plain(s['fark_tahmin'], 3)} dolar, yüzde 95 aralığı "
            f"[{plain(s['fark_alt'], 3)}; {plain(s['fark_ust'], 3)}] dolar: katsayının yalnız “anlamlı” olduğunu "
            "söylemekten daha açıklayıcıdır. HPRICE1'de yatak odası katsayısı "
            f"{plain(beds['katsayi'], 2)} bin dolar ({_p(beds['p'])}), aralık [{plain(beds['alt'], 2)}; "
            f"{plain(beds['ust'], 2)}]: hem küçük negatif hem büyük pozitif değerler veriyle uyumludur. Doğru sonuç "
            "“etki yok” değil, tahminin belirsiz olduğudur. Büyüklük, belirsizlik aralığı ve ekonomik bağlam birlikte "
            "raporlanır (§7.11).")


# --- Adım 10: raporlama -----------------------------------------------------------------------------------

TERM_10 = Choice("adim10_terim", "Raporlanan katsayı", TERM_OPTIONS, "educ", help="Notlarda eğitim katsayısı (§7.13).")


def _report(choices) -> tuple:
    term = choices["adim10_terim"]
    return (
        ModelValue("b10", "m", "coef", f"{TITLES[term]} katsayısı", term=term, decimals=3),
        ModelValue("sh10", "m", "se", "Standart hata", term=term, decimals=3),
        ModelValue("t10", "m", "t", "t istatistiği", term=term, decimals=2),
        ModelValue("p10", "m", "p", "p-değeri", term=term, decimals=3),
        ModelValue("alt10", "m", "ci_low", "Yüzde 95 GA alt sınırı", term=term, decimals=3),
        ModelValue("ust10", "m", "ci_high", "Yüzde 95 GA üst sınırı", term=term, decimals=3),
        Scalar("dort10", E.mul(4, E.ref("b10")), "Dört yıllık farkın nokta tahmini (dolar)", decimals=2),
    )


def _report_note(state, choices) -> str:
    term = choices["adim10_terim"]
    s = state.scalars
    kept = s["p10"] >= 0.05
    text = (f"Örnek raporlama paragrafı: “Eğitim, deneyim ve kıdemin saatlik ücretle ilişkisini inceleyen EKK modelinde "
            f"{WORDS[term]} katsayısı {plain(s['b10'], 3)} olarak tahmin edilmiştir (SH = {plain(s['sh10'], 3)}; "
            f"t = {plain(s['t10'], 2)}; {_p(s['p10'])}; yüzde 95 GA [{plain(s['alt10'], 3)}; "
            f"{plain(s['ust10'], 3)}]). Buna göre {OTHERS[term]} sabitken bir ek {WORDS[term]} yılı, örneklemde tahmin "
            f"edilen saatlik ücretin ortalama {plain(abs(s['b10']), 3)} dolar daha "
            f"{'yüksek' if s['b10'] > 0 else 'düşük'} olmasıyla ilişkilidir. ")
    if kept:
        text += "Katsayı yüzde 5 düzeyinde sıfırdan istatistiksel olarak ayrışmamaktadır. "
    return text + (f"Dört yıllık {WORDS[term]} farkının nokta tahmini yaklaşık {plain(abs(s['dort10']), 2)} dolardır. "
                   f"{TITLES[term]} gözlemsel veride rastgele atanmadığından bulgu nedensel etki olarak "
                   "yorumlanmamıştır.” Paragraf beş bileşeni birlikte taşır: katsayı büyüklüğü, standart hata, test "
                   "sonucu, güven aralığı ve yorum sınırı (§7.13).")


# --- Tanım ---------------------------------------------------------------------------------------------

_T3 = (("10", 2.228), ("20", 2.086), ("30", 2.042), ("60", 2.000), ("120", 1.980), ("Çok büyük", 1.960))
_TABLE74 = (
    ("educ", "Eğitim", (0.599, 0.051, 11.68, None, 0.498, 0.700)),
    ("exper", "Deneyim", (0.022, 0.012, 1.85, 0.064, -0.001, 0.046)),
    ("tenure", "Kıdem", (0.169, 0.022, 7.82, None, 0.127, 0.212)),
)
_COLUMN_LABELS = {"katsayi": "katsayı", "sh": "SH", "t": "t", "p": "p", "alt": "GA alt", "ust": "GA üst"}
_TABLE74_COLUMNS = (("katsayi", "katsayı", 3), ("sh", "SH", 3), ("t", "t", 2), ("p", "p", 3), ("alt", "GA alt", 3),
                    ("ust", "GA üst", 3))
_OUTPUT = (
    (INTERCEPT, "sabit terim", (-2.8727, 0.729, -3.941, 0.000, -4.305, -1.441)),
    ("educ", "educ", (0.5990, 0.051, 11.679, 0.000, 0.498, 0.700)),
    ("exper", "exper", (0.0223, 0.012, 1.853, 0.064, -0.001, 0.046)),
    ("tenure", "tenure", (0.1693, 0.022, 7.820, 0.000, 0.127, 0.212)),
)
_OUTPUT_QUANTITIES = (("coef", "coef", 4), ("se", "std err", 3), ("t", "t", 3), ("p", "P>|t|", 3),
                      ("ci_low", "GA alt", 3), ("ci_high", "GA üst", 3))
_TABLE75 = (
    ("(1) Ücret", (("educ", 0.599), ("educ_sh", 0.051), ("exper", 0.022), ("exper_sh", 0.012), ("tenure", 0.169),
                   ("tenure_sh", 0.022), (INTERCEPT, -2.873), (f"{INTERCEPT}_sh", 0.729), ("r2", 0.306))),
    ("(2) ln(Ücret)", (("educ", 0.092), ("educ_sh", 0.007), ("exper", 0.004), ("exper_sh", 0.002), ("tenure", 0.022),
                       ("tenure_sh", 0.003), (INTERCEPT, 0.284), (f"{INTERCEPT}_sh", 0.104), ("r2", 0.316))),
)
_TABLE76 = (
    ("lotsize", "Arsa büyüklüğü", (("katsayi", 0.0021, 4), ("sh", 0.0006, 4), ("p", 0.002, 3), ("alt", 0.0008, 4),
                                   ("ust", 0.0033, 4), ("t", 3.22, 2))),
    ("sqrft", "Konut büyüklüğü", (("katsayi", 0.1228, 4), ("sh", 0.0132, 4), ("alt", 0.0965, 4), ("ust", 0.1491, 4),
                                  ("t", 9.28, 2))),
    ("bdrms", "Yatak odası", (("katsayi", 13.8525, 4), ("sh", 9.0101, 4), ("p", 0.128, 3), ("alt", -4.0651, 4),
                              ("ust", 31.7702, 4), ("t", 1.54, 2))),
)

STEPS = (
    interactive_step(
        number=1,
        title="t istatistiği: tahmin kaç standart hata uzakta?",
        note=NoteRef("7.4", 0, ("Denklem 7.5", "Denklem 7.6", "Tablo 7.3")),
        explanation=(
            "Tek katsayı için test istatistiği $t = (\\widehat\\beta_j - a)/\\operatorname{se}(\\widehat\\beta_j)$'dir: "
            "tahmin, sıfır hipotezindeki değerden kaç standart hata uzaktadır? Notlardaki örnekte "
            "$\\widehat\\beta = 0{,}30$, $\\operatorname{se} = 0{,}12$ ve $H_0: \\beta = 0$. Değerleri değiştirin. "
            "Tablo 7.3 yüzde 5 iki taraflı testin kritik değerlerini verir; serbestlik derecesi büyüdükçe t dağılımı "
            "standart normale yaklaşır."
        ),
        controls=(B_CHOICE, SE_CHOICE, A_CHOICE, ALPHA_1),
        build=_t_statistic,
        checks=(
            _scalar("t_ornek", 2.50, "§7.4: t = 0,30/0,12", 2),
            *(Check(f"Tablo 7.3: sd = {label}", TableTarget("kritik_degerler", label, "deger"), value, 3)
              for label, value in _T3),
        ),
        note_for=lambda state, choices: _t_note(state, choices),
    ),
    interactive_step(
        number=2,
        title="WAGE1: eğitim katsayısı için t testi",
        note=NoteRef("7.4", 0, ("WAGE1 örneği",)),
        explanation=(
            "Ücret modeli eğitim, deneyim ve kıdemle tahmin edilir. Eğitim katsayısının standart hatası 0,0513'tür: "
            "$H_0: \\beta = 0$ için t ≈ 11,68, $H_0: \\beta = 0{,}50$ için t ≈ 1,93. Sınanan katsayıyı ve ikinci "
            "hipotezdeki değeri değiştirin: sonucun sınanan değere nasıl bağlı olduğunu görün."
        ),
        controls=(TERM_2, A_2),
        build=_wage_t,
        checks=(
            Check("§7.4: sabit terim", CoefTarget("m", INTERCEPT), -2.873, 3),
            Check("§7.4: eğitim katsayısı", CoefTarget("m", "educ"), 0.599, 3),
            Check("§7.4: deneyim katsayısı", CoefTarget("m", "exper"), 0.022, 3),
            Check("§7.4: kıdem katsayısı", CoefTarget("m", "tenure"), 0.169, 3),
            _scalar("sh2", 0.0513, "§7.4: eğitim katsayısının standart hatası", 4),
            _scalar("t0_2", 11.68, "§7.4: t, H₀: β = 0", 2),
            _scalar("ta_2", 1.93, "§7.4: t, H₀: β = 0,50", 2),
        ),
        note_for=lambda state, choices: _wage_t_note(state, choices),
    ),
    interactive_step(
        number=3,
        title="p-değeri: deneyim katsayısı",
        note=NoteRef("7.5", 0, ("Şekil 7.2",)),
        explanation=(
            "p-değeri, sıfır hipotezi ve model varsayımları doğruyken gözlenen kadar veya daha uç bir test istatistiği "
            "elde etme olasılığıdır. İki taraflı testte her iki kuyruktaki büyük |t| değerleri sayılır. Deneyim "
            "katsayısında p = 0,064: yüzde 5 düzeyinde reddedilemez, yüzde 10 düzeyinde reddedilir. Katsayıyı ve "
            "anlamlılık düzeyini değiştirin."
        ),
        controls=(TERM_3, ALPHA_3),
        build=_p_value,
        checks=(
            _scalar("b3", 0.0223, "§7.5: deneyim katsayısı", 4),
            _scalar("sh3", 0.0121, "§7.5: standart hata", 4),
            _scalar("t3", 1.853, "§7.5: t", 3),
            _scalar("t3", 1.85, "Şekil 7.2: gözlenen t", 2),
            _scalar("p3", 0.064, "§7.5: iki taraflı p-değeri", 3),
        ),
        note_for=lambda state, choices: _p_note(state, choices),
    ),
    interactive_step(
        number=4,
        title="Güven aralığı: tek sayı yerine uyumlu değerler",
        note=NoteRef("7.6", 0, ("Denklem 7.8",)),
        explanation=(
            "Yüzde $100(1-\\alpha)$ güven aralığı $\\widehat\\beta_j \\pm t_{\\alpha/2;\\,n-k-1}\\operatorname{se}"
            "(\\widehat\\beta_j)$'dir; büyük örneklemde yüzde 95 için kritik değer yaklaşık 1,96. Notlardaki örnekte "
            "$\\widehat\\beta = 0{,}30$ ve $\\operatorname{se} = 0{,}12$. Güven düzeyini ve standart hatayı "
            "değiştirin: aralık nasıl genişliyor ya da daralıyor?"
        ),
        controls=(B_4, SE_4, LEVEL_4),
        build=_interval_example,
        checks=(
            _scalar("alt4", 0.0648, "§7.6: alt sınır", 4),
            _scalar("ust4", 0.5352, "§7.6: üst sınır", 4),
            _scalar("alt4", 0.065, "§7.6: aralık [0,065; 0,535], alt", 3),
            _scalar("ust4", 0.535, "§7.6: aralık [0,065; 0,535], üst", 3),
        ),
        note_for=lambda state, choices: _interval_note(state, choices),
    ),
    interactive_step(
        number=5,
        title="Hipotez testi ile güven aralığı: WAGE1",
        note=NoteRef("7.7", 0, ("Tablo 7.4", "Şekil 7.4")),
        explanation=(
            "İki taraflı $H_0: \\beta_j = a$ testi ile güven aralığı aynı bilgiyi verir: $a$ aralığın dışındaysa $H_0$ "
            "reddedilir (Denklem 7.9). Tablo 7.4 üç katsayının çıkarımını, Şekil 7.4 aralıkları gösterir. Güven "
            "düzeyini değiştirin: hangi aralık sıfırı kapsıyor?"
        ),
        controls=(LEVEL_5,),
        build=_equivalence,
        checks=(
            *(Check(f"Tablo 7.4: {name}, {label}", TableTarget("tablo74", term, column), value, decimals)
              for term, name, values in _TABLE74
              for (column, label, decimals), value in zip(_TABLE74_COLUMNS, values) if value is not None),
            Check("§7.7: eğitim GA alt sınırı", TableTarget("tablo74", "educ", "alt"), 0.498, 3),
            Check("§7.7: eğitim GA üst sınırı", TableTarget("tablo74", "educ", "ust"), 0.700, 3),
        ),
        note_for=lambda state, choices: _equivalence_note(state, choices),
    ),
    interactive_step(
        number=6,
        title="Tek taraflı test: yön ne zaman kullanılabilir?",
        note=NoteRef("7.8", 0),
        explanation=(
            "Teori ve araştırma planı önceden deneyim katsayısının pozitif olmasını öngörüyorsa $H_0: \\beta \\leq 0$, "
            "$H_1: \\beta > 0$ kurulabilir. Tek taraflı p-değeri yalnız seçilen kuyruktaki alanı sayar. Yönü "
            "değiştirin: gözlenen t alternatifin tersi yönde olduğunda ne oluyor?"
        ),
        controls=(TERM_6, DIRECTION_6),
        build=_one_sided,
        checks=(
            _scalar("t6", 1.853, "§7.8: t", 3),
            _scalar("p_secilen", 0.032, "§7.8: tek taraflı p-değeri", 3),
            _scalar("p_iki6", 0.064, "§7.8: iki taraflı p-değeri", 3),
        ),
        note_for=lambda state, choices: _one_sided_note(state, choices),
    ),
    interactive_step(
        number=7,
        title="Python çıktısını satır satır okumak",
        note=NoteRef("7.9", 0, ("Kod 7.1", "Kod 7.2")),
        explanation=(
            "Statsmodels çıktısında `coef`, `std err`, `t`, `P>|t|` ve `[0.025 0.975]` sütunları aynı katsayıyı farklı "
            "açılardan özetler. Önce bağımlı değişkeni ve gözlem sayısını, sonra ilgilenilen katsayı satırını okuyun. "
            "Modeli değiştirin: artık serbestlik derecesi ve katsayı satırları nasıl değişiyor?"
        ),
        controls=(OUTCOME_7, REGRESSORS_7),
        build=_output,
        checks=(
            *(Check(f"Kod 7.2: {name}, {label}", CoefTarget("cikti", term, quantity), value, decimals)
              for term, name, values in _OUTPUT
              for (quantity, label, decimals), value in zip(_OUTPUT_QUANTITIES, values)),
            Check("Kod 7.2: gözlem sayısı", ModelTarget("cikti", "nobs"), 526, 0),
            Check("Kod 7.2: artık serbestlik derecesi", ModelTarget("cikti", "df_resid"), 522, 0),
            Check("Kod 7.2: R²", ModelTarget("cikti", "r2"), 0.306, 3),
        ),
        note_for=lambda state, choices: _output_note(state, choices),
    ),
    LabStep(
        number=8,
        title="Makale tablosu: parantezler, yıldızlar ve notlar",
        note=NoteRef("7.10", 0, ("Tablo 7.5",)),
        explanation=(
            "Makaleler tam çıktı yerine sıkıştırılmış tablo verir. Parantezde ne olduğu (standart hata, t ya da güven "
            "aralığı) ve yıldız eşikleri tablo notundan okunur. Sütun (1) düzey–düzey, Sütun (2) log–düzey modelidir; "
            "katsayıların birimleri farklıdır."
        ),
        operations=_paper(),
        checks=tuple(Check(f"Tablo 7.5: {heading}, {row}", TableTarget("tablo75", row, heading), value, 3)
                     for heading, values in _TABLE75 for row, value in values),
        takeaway=(
            "Sütun (2)'de eğitim katsayısı 0,092: diğer değişkenler sabitken bir ek eğitim yılı tahmin edilen ücretle "
            "yaklaşık yüzde 9,2 daha yüksek ilişkilidir (daha hassas dönüşüm Konu 9'da). Üç yıldız etkinin büyük "
            "olduğunu değil, p-değerinin 0,01'den küçük olduğunu gösterir. Deneyim Sütun (1)'de yalnız yüzde 10 "
            "düzeyinde, Sütun (2)'de yüzde 5 düzeyinde sıfırdan ayrışır (§7.10)."
        ),
    ),
    interactive_step(
        number=9,
        title="İstatistiksel anlamlılık ile iktisadi önem",
        note=NoteRef("7.11", 0, ("Tablo 7.6",)),
        explanation=(
            "Bir katsayının sıfırdan istatistiksel olarak farklı olması etkinin büyük olduğunu göstermez. WAGE1'de "
            "eğitim farkının ücret karşılığını aralığıyla birlikte hesaplayın; HPRICE1'de yatak odası katsayısı "
            "pozitif ama belirsizdir. Eğitim farkını değiştirin."
        ),
        controls=(YEARS_9,),
        build=_magnitude,
        checks=(
            _scalar("fark_tahmin", 2.396, "§7.11: 4(0,599)", 3),
            _scalar("fark_alt", 1.993, "§7.11: dört yıllık aralık, alt", 3),
            _scalar("fark_ust", 2.799, "§7.11: dört yıllık aralık, üst", 3),
            *(Check(f"Tablo 7.6: {name}, {_COLUMN_LABELS[column]}", TableTarget("tablo76", term, column), value,
                    decimals)
              for term, name, cells in _TABLE76 for column, value, decimals in cells if column != "t"),
            *(Check(f"§7.13 egzersiz tablosu: {name}, t", TableTarget("tablo76", term, "t"), value, decimals)
              for term, name, cells in _TABLE76 for column, value, decimals in cells if column == "t"),
            Check("§7.11: yatak odası katsayısı", TableTarget("tablo76", "bdrms", "katsayi"), 13.85, 2),
            Check("§7.11: yatak odası standart hatası", TableTarget("tablo76", "bdrms", "sh"), 9.01, 2),
            Check("§7.11: yatak odası GA alt", TableTarget("tablo76", "bdrms", "alt"), -4.07, 2),
            Check("§7.11: yatak odası GA üst", TableTarget("tablo76", "bdrms", "ust"), 31.77, 2),
        ),
        note_for=lambda state, choices: _magnitude_note(state, choices),
    ),
    interactive_step(
        number=10,
        title="Bütünleşik uygulama: bir sonuç nasıl raporlanır?",
        note=NoteRef("7.13", 0, ("Adım 1–5", "Örnek raporlama paragrafı")),
        explanation=(
            "İyi bir rapor beş bileşeni birlikte taşır: katsayı büyüklüğü, standart hata, test sonucu, güven aralığı ve "
            "yorum sınırı (§7.13, Adım 1–5). Raporlanan katsayıyı değiştirin; paragraf seçiminize göre yeniden yazılır."
        ),
        controls=(TERM_10,),
        build=_report,
        checks=(
            _scalar("b10", 0.599, "§7.13: katsayı", 3),
            _scalar("sh10", 0.051, "§7.13: standart hata", 3),
            _scalar("t10", 11.68, "§7.13: t", 2),
            _scalar("alt10", 0.498, "§7.13: yüzde 95 GA alt", 3),
            _scalar("ust10", 0.700, "§7.13: yüzde 95 GA üst", 3),
            _scalar("dort10", 2.40, "§7.13: dört yıllık fark", 2),
        ),
        note_for=lambda state, choices: _report_note(state, choices),
    ),
)


KONU07_LAB = LabSpec(
    topic_key="konu07",
    title="Uygulama: Tek Katsayı İçin Hipotez Testleri",
    note_section="7",
    steps=STEPS,
    labels=(
        *W.labels(DATA, HPRICE),
        (INTERCEPT, "Sabit terim"),
    ),
    consistency_notes=(
        "Veri notlardaki gibi WAGE1 ve HPRICE1'dir; notlardaki kod, betikler, uygulama ve üretilen kod onları wooldridge "
        "paketinden okur. Paket verisiyle Bölüm 7'deki bütün sayılar aynıdır.",
        "Bölümün benzetimi (Tablo 7.1, kapsama oranı, Şekil 7.1 ve 7.3) tohum 305 ile yeniden üretildi ve Sezgi "
        "sekmesindeki Konu 7 Deney 1'in varsayılan ayarlarıdır.",
        "§7.11'deki dört yıllık aralık yuvarlanmamış sınırlarla hesaplanır: [1,993; 2,799] (eski metin yuvarlanmış "
        "sınırlarla [1,992; 2,800] yazıyordu). WAGE1 ücreti “dolar” (eski metinde “ücret birimi”).",
        "Standart hatalar klasik EKK standart hatalarıdır; heteroskedastisiteye dayanıklı standart hatalar Konu 12'de.",
    ),
)
