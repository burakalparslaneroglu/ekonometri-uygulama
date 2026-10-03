"""Konu 7 genel uygulaması: tek katsayı için hipotez testleri (standart hata, t, p-değeri, güven aralığı).

Notlardaki on adım (``core.labs.konu07``) aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde
yazılır: t istatistiği ve kritik değerler, temel katsayının t testi, p-değeri, güven aralığının sayısal örneği, test
ile güven aralığının eşdeğerliği, tek taraflı test, yazılım çıktısı, makale tablosu, istatistiksel anlamlılık ile
iktisadi önem ve raporlama.

Alternatif örnekte ücret modeli WAGE2'dir (935 erkek çalışan, 1980; aylık kazanç ~ eğitim + deneyim + kıdem).
Sayısal örnekler (Adım 1 ve 4) aynı modelin p-değeri en büyük katsayısından kurulur (WAGE2'de kıdem); p-değeri ve tek
taraflı test adımlarının varsayılanı da odur. İktisadi önem adımının ikinci verisi KIELMC'nin 1978 satışlarıdır (fiyat
~ büyüklük + oda + banyo; oda sayısının katsayısı pozitif ama belirsizdir). "Kendi verini yükle" seçeneğinde model
sonucun temel açıklayıcı ve ek değişkenlere göre regresyonudur; sayısal örnekler aynı kuralla kurulur, iktisadi önem
adımının tablosu ana modelin katsayılarıdır. Uyum tamsa (R² ≈ 1) standart hataya bağlı kontroller çıkarılır, sayısal
örneklerin standart hatası katsayının yarısıdır ve metinler bunu bildirir.

Etkileşim notlardaki gibidir: sayısal örneğin değerleri, sınanan katsayı, sıfır hipotezindeki değer, anlamlılık ve
güven düzeyi, alternatif hipotezin yönü, yazılım çıktısının modeli ve iktisadi önemde açıklayıcı değişkendeki fark.
Standart hatalar klasik (homoskedastik) EKK standart hatalarıdır; dayanıklı standart hatalar Konu 12'nin konusudur.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import cache

import numpy as np
import statsmodels.formula.api as smf

from core.labs import expr as E
from core.labs.ornek import Case, TopicVariants, sayi, sayim, stable_checks, with_app_values
from core.labs.ornek_regresyon import (
    ACIKLAYICI,
    EXACT_MULTI_NOTE,
    ROW_RULE,
    SONUC,
    candidates,
    capital,
    change,
    custom_lab,
    digits_for,
    exact_multi,
    gap_control,
    house_case,
    labels_of,
    level_text,
    listing,
    log_column,
    log_label,
    log_operations,
    negligible,
    nice,
    number_control,
    options,
    outcome_words,
    phrase,
    plural,
    positive,
    reload,
    roles,
    rough,
    sample_employed,
    second,
    short_unit,
    step_words,
    wage2_case,
)
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

TOPIC = "konu07"
TITLE = "Uygulama: Tek Katsayı İçin Hipotez Testleri"
ALT_REGRESSORS = ("educ", "exper", "tenure")
ALT_OUTPUT = ("educ", "exper", "tenure", "IQ")
ALPHAS = (("0.1", "%10"), ("0.05", "%5"), ("0.01", "%1"))
LEVELS = (("0.9", "%90"), ("0.95", "%95"), ("0.99", "%99"))
DIRECTIONS = (("sag", "Sağ kuyruk: H₁: β > 0"), ("sol", "Sol kuyruk: H₁: β < 0"), ("iki", "İki taraflı: H₁: β ≠ 0"))
CRITICAL_DF = (("10", 10), ("20", 20), ("30", 30), ("60", 60), ("120", 120))


def _check(label: str, target, decimals: int) -> Check:
    return Check(label, target, 0.0, decimals)


def _scalar(name: str, label: str, decimals: int) -> Check:
    return _check(label, ScalarTarget(name), decimals)


def _name(case: Case, column: str) -> str:
    return case.name(column) if case.own else phrase(case, column)


def _p(value: float) -> str:
    """p-değerinin eşitlikli yazımı: "p = 0,064"; üç basamakta 0 ya da 1 görünüyorsa "p < 0,001" ya da "p > 0,999"."""

    if value < 0.0005:
        return "p < 0,001"
    return "p > 0,999" if value > 0.9995 else f"p = {sayi(value, 3)}"


def _t(value: float, decimals: int = 2) -> str:
    """"t = 2,31"; uyum tamken standart hata sıfıra çok yakın, t yuvarlama gürültüsüyle çok büyüktür: "|t| > 10⁶"."""

    if not math.isfinite(value) or abs(value) >= 1e6:
        return "|t| > 10⁶"
    return f"t = {sayi(value, decimals)}"


def _percent(value: float) -> str:
    return "%" + sayi(100 * value, 0 if float(round(100 * value, 8)).is_integer() else 1)


def _regressors(case: Case) -> tuple[str, ...]:
    """Ana model: alternatif örnekte notlardaki gibi eğitim, deneyim ve kıdem; kendi verinde bütün seçenekler."""

    return tuple(case.extra.get("default_regressors") or candidates(case))


def _fit(case: Case, outcome: str, regressors, data=None) -> object:
    """Metinler ve varsayılanlar için modelin kendisi (uygulamanın hesabı işlemlerle yapılır)."""

    return smf.ols(f"{outcome} ~ {' + '.join(regressors)}", data=case.data if data is None else data).fit()


def _exact(case: Case) -> bool:
    return case.own and exact_multi(case, case.roles[SONUC], _regressors(case))


def _exact_note(case: Case) -> str:
    return EXACT_MULTI_NOTE if _exact(case) else ""


def _exact_model(case: Case, outcome: str, regressors: tuple[str, ...]) -> bool:
    """Seçilen modelin uyumu tam mı (kendi verinde); log sonuç verideki sonuçtan hesaplanır."""

    if not case.own or not regressors:
        return False
    y = case.roles[SONUC]
    if outcome != y and outcome not in case.data.columns:
        data = case.data.assign(**{outcome: np.log(case.data[y].astype(float))})
        return exact_multi(replace(case, data=data), outcome, regressors)
    return exact_multi(case, outcome, regressors)


def _loosest(case: Case) -> str:
    """Ana modelde p-değeri en büyük eğim (sayısal örneklerin, p-değeri ve tek taraflı test adımlarının varsayılanı).
    Uyum tamsa p-değerleri yuvarlama hatasıdır; varsayılan temel açıklayıcıdır."""

    if _exact(case):
        return case.roles[ACIKLAYICI]
    p_values = _fit(case, case.roles[SONUC], _regressors(case)).pvalues.drop(INTERCEPT)
    return str(p_values.idxmax()) if p_values.notna().any() else case.roles[ACIKLAYICI]


def _loosest_words(case: Case, name: str) -> str:
    """Varsayılan katsayının tanımı (açıklama ve yardım metinleri için); ``name`` katsayının yazılmış adıdır."""

    if _exact(case):
        return f"temel açıklayıcının katsayısı ({name}; uyum tam olduğu için p-değerleri yuvarlama hatasıdır)"
    return f"p-değeri en büyük katsayı ({name})"


def _example_values(case: Case) -> tuple[str, float, float, bool]:
    """Sayısal örneğin terimi, katsayısı ve standart hatası; son değer standart hatanın sıfıra çok yakın (uyum tam ya da
    standart hata katsayının 10⁻⁹'undan küçük) olup olmadığıdır."""

    term = _loosest(case)
    model = _fit(case, case.roles[SONUC], _regressors(case))
    b, se = float(model.params[term]), float(model.bse[term])
    degenerate = _exact(case) or not (math.isfinite(se) and se > 1e-9 * max(abs(b), 1.0))
    return term, b, se, degenerate


# --- Adım 1: t istatistiği ---------------------------------------------------------------------------------

def _example_controls(case: Case, prefix: str) -> tuple[NumberChoice, NumberChoice]:
    """Sayısal örneğin katsayısı ve standart hatası: p-değeri en büyük katsayı, üç anlamlı basamağa yuvarlanmış. Standart
    hata sıfıra çok yakınsa (uyum tam) varsayılan standart hata katsayının yarısıdır (katsayı sıfırsa 1)."""

    if f"{prefix}_ornek" in case.extra:
        return tuple(case.extra[f"{prefix}_ornek"])
    term, b, se, degenerate = _example_values(case)
    if degenerate:
        se = abs(b) / 2 if abs(b) > 0 else 1.0
        se_help = (f"Uyum tam olduğu için {case.name(term)} katsayısının standart hatası sıfıra çok yakındır; varsayılan "
                   "katsayının yarısıdır.")
    else:
        se_help = f"Varsayılan: {case.name(term)} katsayısının standart hatası (yuvarlanmış)."
    scale = max(abs(b) + 4 * se, se)
    b_control = number_control(f"{prefix}_b", "Katsayı tahmini β̂", nice(b, 3), -scale, scale,
                               help=f"Varsayılan: {case.name(term)} katsayısı (yuvarlanmış).")
    se_control = number_control(f"{prefix}_se", "Standart hata se(β̂)", nice(se, 3), 0.0, 4 * se, help=se_help)
    if se_control.minimum <= 0:  # sıfır standart hata tanımsız t verir
        se_control = replace(se_control, minimum=se_control.step)
    return b_control, se_control


def _step1(case: Case) -> LabStep:
    b_control, se_control = _example_controls(case, "adim1")
    span = max(abs(b_control.maximum), abs(b_control.minimum))
    a_control = number_control("adim1_a", "Sıfır hipotezindeki değer a", 0.0, -span / 2, span / 2,
                               help="Varsayılan: H₀: β = 0.")
    alpha_control = Choice("adim1_alfa", "Anlamlılık düzeyi α (iki taraflı)", ALPHAS, "0.05",
                           help="Tablo 7.3 yüzde 5 iki taraflı test içindir: t₀,₀₂₅.")
    term = _loosest(case)
    if "adim1_ornek" not in case.extra and _example_values(case)[3]:
        example = (f"Adım 2'deki modelde {phrase(case, term)} katsayısı (yuvarlanmış); uyum tam olduğu için modelin "
                   "standart hatası sıfıra çok yakındır, örnekteki standart hata katsayının yarısıdır")
    else:
        example = f"Adım 2'deki modelde {phrase(case, term)} katsayısı ve standart hatası (yuvarlanmış)"

    def build(choices) -> tuple:
        alpha = float(choices["adim1_alfa"])
        q = round(1 - alpha / 2, 6)
        return (
            Scalar("b_ornek", E.const(choices["adim1_b"]), "Katsayı tahmini β̂", decimals=b_control.decimals),
            Scalar("sh_ornek", E.const(choices["adim1_se"]), "Standart hata se(β̂)", decimals=se_control.decimals),
            Scalar("t_ornek", E.div(E.sub(E.ref("b_ornek"), choices["adim1_a"]), E.ref("sh_ornek")),
                   f"t = (β̂ − a)/se(β̂), a = {level_text(choices['adim1_a'])}", decimals=2),
            Scalar("kritik_buyuk", E.norminv(q), "Büyük örneklem kritik değeri", decimals=3),
            ScalarTable((*((label, E.tinv(q, df)) for label, df in CRITICAL_DF), ("Çok büyük", E.norminv(q))),
                        "kritik_degerler", decimals=3, heading="Serbestlik derecesi",
                        value=f"Kritik değer (yüzde {sayi(100 * alpha, 0)} iki taraflı)"),
        )

    def note(state, choices) -> str:
        s = state.scalars
        t, critical = s["t_ornek"], s["kritik_buyuk"]
        a = level_text(choices["adim1_a"])
        level = _percent(float(choices["adim1_alfa"]))
        decision = "reddedilir" if abs(t) > critical else "reddedilemez"
        text = (f"Tahmin, sıfır hipotezindeki değerden (a = {a}) |t| = {sayi(abs(t), 2)} standart hata uzaktadır. Büyük "
                f"örneklemde {level} iki taraflı testin kritik değeri {sayi(critical, 2)}: H₀ {decision} (Denklem 7.6). "
                "Serbestlik derecesi büyüdükçe kritik değer standart normal değere yaklaşır (Tablo 7.3).")
        if (choices["adim1_b"], choices["adim1_se"], choices["adim1_a"]) == (b_control.default, se_control.default, 0):
            se2 = 2 * float(se_control.default)
            text += (f" Aynı tahminin standart hatası iki katı ({level_text(round(se2, se_control.decimals))}) olsaydı "
                     f"t = {sayi(t / 2, 2)} olurdu: veri sıfırdan ayrışma konusunda daha zayıf kanıt sunardı.")
        return text

    return interactive_step(
        number=1,
        title="t istatistiği: tahmin kaç standart hata uzakta?",
        note=NoteRef("7.4", 0, ("Denklem 7.5", "Denklem 7.6", "Tablo 7.3")),
        explanation=(
            "Tek katsayı için test istatistiği $t = (\\widehat\\beta_j - a)/\\operatorname{se}(\\widehat\\beta_j)$'dir: "
            f"tahmin, sıfır hipotezindeki değerden kaç standart hata uzaktadır? Sayısal örnek: {example}, "
            "$H_0: \\beta = 0$. Değerleri değiştirin. "
            "Tablo 7.3'teki gibi tablo yüzde 5 iki taraflı testin kritik değerlerini verir; serbestlik derecesi büyüdükçe "
            "t dağılımı standart normale yaklaşır."
        ),
        controls=(b_control, se_control, a_control, alpha_control),
        build=build,
        checks=(
            _scalar("t_ornek", "t = β̂ / se(β̂)", 2),
            *(_check(f"Kritik değer: sd = {label}", TableTarget("kritik_degerler", label, "deger"), 3)
              for label in (*(label for label, _ in CRITICAL_DF), "Çok büyük")),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 2: temel katsayının t testi ----------------------------------------------------------------------

def _second_value(case: Case) -> NumberChoice:
    if "adim2_a" in case.extra:
        return case.extra["adim2_a"]
    x = case.roles[ACIKLAYICI]
    model = _fit(case, case.roles[SONUC], _regressors(case))
    b, se = float(model.params[x]), float(model.bse[x])
    if _exact(case) or not (math.isfinite(se) and se > 1e-9 * max(abs(b), 1.0)):
        # Standart hata sıfıra çok yakın: b ± 10·se aralığı kaydırıcı için anlamsız derecede dardır.
        return number_control("adim2_a", "İkinci sıfır hipotezindeki değer a", nice(b / 2, 2), b - 2 * abs(b) - 1,
                              b + 2 * abs(b) + 1,
                              help="Uyum tam olduğu için standart hata sıfıra çok yakındır; varsayılan tahminin yarısıdır.")
    value = nice(b - 1.5 * se, 2)
    return number_control("adim2_a", "İkinci sıfır hipotezindeki değer a", value, b - 10 * se, b + 10 * se,
                          help="Varsayılan: tahminin yaklaşık 1,5 standart hata altındaki yuvarlak bir değer.")


def _load(case: Case) -> tuple:
    y = case.roles[SONUC]
    regressors = _regressors(case)
    return (*case.load, OLS("m", case.frame, y, regressors, f"Ana model: {y} ~ {' + '.join(regressors)}"))


def _step2(case: Case) -> LabStep:
    x = case.roles[ACIKLAYICI]
    regressors = _regressors(case)
    term_control = Choice("adim2_terim", "Sınanan katsayı", options(case, regressors), x,
                          help=f"Varsayılan: {case.name(x)} katsayısı.")
    a_control = _second_value(case)

    def build(choices) -> tuple:
        term, a = choices["adim2_terim"], choices["adim2_a"]
        return (
            *_load(case),
            ModelValue("b2", "m", "coef", f"{capital(_name(case, term))} katsayısı β̂", term=term, decimals=3),
            ModelValue("sh2", "m", "se", "Standart hata se(β̂)", term=term, decimals=4),
            ModelValue("sd_artik", "m", "df_resid", "Artık serbestlik derecesi n − k − 1", decimals=0),
            Scalar("t0_2", E.div(E.ref("b2"), E.ref("sh2")), "t, H₀: β = 0", decimals=2),
            Scalar("ta_2", E.div(E.sub(E.ref("b2"), a), E.ref("sh2")), f"t, H₀: β = {level_text(a)}", decimals=2),
            Scalar("kritik_2", E.tinv(0.975, E.ref("sd_artik")), "Kritik değer t₀,₀₂₅", decimals=3),
        )

    def note(state, choices) -> str:
        term, a = choices["adim2_terim"], choices["adim2_a"]
        s = state.scalars
        critical = s["kritik_2"]
        others = listing([phrase(case, name) for name in regressors if name != term])

        def decision(t: float) -> str:
            return "reddedilir" if abs(t) > critical else "reddedilemez"

        held = f"{others} sabitken " if others else ""
        return (f"H₀: β = 0 için {_t(s['t0_2'])}: yüzde 5 iki taraflı testte H₀ {decision(s['t0_2'])}. "
                f"H₀: β = {level_text(a)} için {_t(s['ta_2'])}: H₀ {decision(s['ta_2'])} (kritik değer "
                f"{sayi(critical, 3)}). Hipotez testinin cevabı sınanan değere bağlıdır: katsayının sıfırdan farklı olması "
                f"ile belirli bir değerden farklı olması aynı sonuç değildir (§7.4). Test, {held}{phrase(case, term)} ile "
                f"{phrase(case, case.roles[SONUC])} arasındaki doğrusal ilişki hakkındadır; gözlemsel veride tek başına "
                "nedensel etkiyi kanıtlamaz." + _exact_note(case))

    exact = _exact(case)
    return interactive_step(
        number=2,
        title=str(case.extra.get("step2_title", "Temel katsayı için t testi")),
        note=NoteRef("7.4", 0, ("WAGE1 örneği",)),
        explanation=(
            f"Model `{case.roles[SONUC]} ~ {' + '.join(regressors)}`. Temel katsayının standart hatasıyla $H_0: \\beta = 0$ "
            f"ve $H_0: \\beta = a$ sınanır (varsayılan a = {level_text(a_control.default)}). Sınanan katsayıyı ve ikinci "
            "hipotezdeki değeri değiştirin: sonucun sınanan değere nasıl bağlı olduğunu görün."
        ),
        controls=(term_control, a_control),
        build=build,
        checks=(
            _check("Sabit terim", CoefTarget("m", INTERCEPT), 3),
            *(_check(f"{case.name(name)} katsayısı", CoefTarget("m", name), 3) for name in regressors),
            *(() if exact else (_scalar("sh2", "Temel katsayının standart hatası", 4), _scalar("t0_2", "t, H₀: β = 0", 2),
                                _scalar("ta_2", "t, H₀: β = a", 2))),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 3: p-değeri ---------------------------------------------------------------------------------------

def _step3(case: Case) -> LabStep:
    regressors = _regressors(case)
    term = _loosest(case)
    term_control = Choice("adim3_terim", "Sınanan katsayı", options(case, regressors), term,
                          help=f"Varsayılan: {_loosest_words(case, case.name(term))}.")
    alpha_control = Choice("adim3_alfa", "Anlamlılık düzeyi α", ALPHAS, "0.05", help="Varsayılan: yüzde 5.")

    def build(choices) -> tuple:
        chosen, alpha = choices["adim3_terim"], float(choices["adim3_alfa"])
        return (
            ModelValue("b3", "m", "coef", f"{capital(_name(case, chosen))} katsayısı β̂", term=chosen, decimals=4),
            ModelValue("sh3", "m", "se", "Standart hata se(β̂)", term=chosen, decimals=4),
            ModelValue("t3", "m", "t", "t istatistiği (H₀: β = 0)", term=chosen, decimals=3),
            ModelValue("p3", "m", "p", "İki taraflı p-değeri", term=chosen, decimals=3),
            HypothesisPlot("t", "t3", "sd_artik",
                           f"{capital(_name(case, chosen))} katsayısı: yüzde {sayi(100 * alpha, 0)} iki taraflı test",
                           "t değeri", alpha=alpha),
        )

    def note(state, choices) -> str:
        chosen, alpha = choices["adim3_terim"], float(choices["adim3_alfa"])
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
        coefficient = sayi(s["b3"], digits_for(s["b3"], 4))
        text = (f"{capital(phrase(case, chosen))} katsayısı {coefficient}, {_t(s['t3'], 3)}, iki taraflı {_p(p)}. "
                f"Seçilen {_percent(alpha)} düzeyinde {decision} (Denklem 7.7). Üç yaygın düzeyde H₀: "
                + "; ".join(parts) + ". ")
        if p >= alpha:
            text += (f"Reddedememek {phrase(case, chosen)} ile {phrase(case, case.roles[SONUC])} arasında hiçbir ilişki "
                     "olmadığı anlamına gelmez; seçilen düzeyde belirsizlik yeterince küçülmemiştir. ")
        return text + ("p-değeri H₀'ın doğru olma olasılığı değildir: H₀ ve model varsayımları doğruyken gözlenen kadar uç "
                       "bir t istatistiği elde etme olasılığıdır (§7.5)." + _exact_note(case))

    return interactive_step(
        number=3,
        title="p-değeri",
        note=NoteRef("7.5", 0, ("Şekil 7.2",)),
        explanation=(
            "p-değeri, sıfır hipotezi ve model varsayımları doğruyken gözlenen kadar veya daha uç bir test istatistiği "
            "elde etme olasılığıdır. İki taraflı testte her iki kuyruktaki büyük |t| değerleri sayılır. Varsayılan: "
            f"modelde {_loosest_words(case, phrase(case, term))}. Katsayıyı ve anlamlılık düzeyini değiştirin."
        ),
        controls=(term_control, alpha_control),
        build=build,
        checks=(_scalar("b3", "Katsayı", 4),) if _exact(case) else (
            _scalar("b3", "Katsayı", 4), _scalar("sh3", "Standart hata", 4), _scalar("t3", "t", 3),
            _scalar("p3", "İki taraflı p-değeri", 3),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 4: güven aralığı, sayısal örnek ------------------------------------------------------------------

def _step4(case: Case) -> LabStep:
    b_control, se_control = _example_controls(case, "adim4")
    level_control = Choice("adim4_duzey", "Güven düzeyi", LEVELS, "0.95", help="Varsayılan: yüzde 95 (kritik değer 1,96).")
    term = _loosest(case)

    def build(choices) -> tuple:
        level = float(choices["adim4_duzey"])
        q = round((1 + level) / 2, 6)
        return (
            Scalar("tahmin4", E.const(choices["adim4_b"]), "Katsayı tahmini β̂", decimals=b_control.decimals),
            Scalar("sh4", E.const(choices["adim4_se"]), "Standart hata se(β̂)", decimals=se_control.decimals),
            Scalar("kritik4", E.norminv(q), f"Büyük örneklem kritik değeri (yüzde {sayi(100 * level, 0)})", decimals=3),
            Scalar("alt4", E.sub(E.ref("tahmin4"), E.mul(E.ref("kritik4"), E.ref("sh4"))), "Alt sınır", decimals=4),
            Scalar("ust4", E.add(E.ref("tahmin4"), E.mul(E.ref("kritik4"), E.ref("sh4"))), "Üst sınır", decimals=4),
        )

    def note(state, choices) -> str:
        s = state.scalars
        level = _percent(float(choices["adim4_duzey"]))
        zero = s["alt4"] <= 0 <= s["ust4"]
        return (f"{level} güven aralığı [{sayi(s['alt4'], 3)}; {sayi(s['ust4'], 3)}]: yarı genişlik kritik değer × "
                f"standart hata = {sayi(s['kritik4'] * s['sh4'], 4)}. Aralık sıfırı {'kapsar' if zero else 'kapsamaz'}. "
                "Güven düzeyi yükseldikçe kritik değer ve aralık büyür; standart hata küçüldükçe aralık daralır. Yorum "
                "tekrarlı örneklemeye dayanır: aynı yöntemle kurulan aralıkların uzun dönemde yaklaşık bu oranı gerçek "
                "parametreyi kapsar (§7.6).")

    return interactive_step(
        number=4,
        title="Güven aralığı: tek sayı yerine uyumlu değerler",
        note=NoteRef("7.6", 0, ("Denklem 7.8",)),
        explanation=(
            "Yüzde $100(1-\\alpha)$ güven aralığı $\\widehat\\beta_j \\pm t_{\\alpha/2;\\,n-k-1}\\operatorname{se}"
            "(\\widehat\\beta_j)$'dir; büyük örneklemde yüzde 95 için kritik değer yaklaşık 1,96. Sayısal örnek Adım "
            f"1'deki gibi {phrase(case, term)} katsayısıdır. Güven düzeyini ve standart hatayı değiştirin: aralık nasıl "
            "genişliyor ya da daralıyor?"
        ),
        controls=(b_control, se_control, level_control),
        build=build,
        checks=(_scalar("alt4", "Alt sınır", 4), _scalar("ust4", "Üst sınır", 4)),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 5: güven aralığı ile test ----------------------------------------------------------------------

def _step5(case: Case) -> LabStep:
    x = case.roles[ACIKLAYICI]
    regressors = _regressors(case)
    model = _fit(case, case.roles[SONUC], regressors)
    a_value = float(_second_value(case).default)
    far = nice(float(model.params[x] + 3 * model.bse[x]), 2)
    values = tuple(dict.fromkeys((0.0, a_value, far)))

    def build(choices) -> tuple:
        level = float(choices["adim5_duzey"])
        shown = sayi(100 * level, 0)
        same = level == 0.95
        return (
            CoefficientTable("m", regressors, "tablo74", "Ana modelde tek katsayı çıkarımı (Tablo 7.4'teki gibi)" if same
                             else f"Ana modelde tek katsayı çıkarımı, yüzde {shown} güven aralığı", level=level),
            CoefficientPlot("m", regressors, "Katsayı tahminleri ve yüzde 95 güven aralıkları (Şekil 7.4'teki gibi)" if same
                            else f"Katsayı tahminleri ve yüzde {shown} güven aralıkları",
                            f"Katsayı tahmini ve yüzde {shown} güven aralığı", level=level),
        )

    def note(state, choices) -> str:
        level = float(choices["adim5_duzey"])
        table = state.tables["tablo74"]
        low, high = table.loc[x, "alt"], table.loc[x, "ust"]
        alpha = _percent(round(1 - level, 10))
        d = digits_for(max(abs(low), abs(high)), 3)
        interval = (f"{capital(phrase(case, x))} katsayısının {_percent(level)} güven aralığı [{sayi(low, d)}; "
                    f"{sayi(high, d)}]")
        if exact:
            return (f"{interval}: uyum tam olduğu için aralıklar neredeyse tek bir noktadır; hangi değerlerin aralıkta "
                    "kaldığı yuvarlama hatasına bağlıdır, bu yüzden test sonuçları yazılmaz. Katsayılar farklı birimlerle "
                    "ölçüldüğü için grafikteki yatay konumlar büyüklük sıralaması olarak okunmamalıdır (§7.7)."
                    + _exact_note(case))
        checks = [f"{level_text(value)} aralıkta "
                  f"{'olduğu için H₀ reddedilemez' if low <= value <= high else 'olmadığı için H₀ reddedilir'}"
                  for value in values]
        crosses = [phrase(case, term) for term in regressors if table.loc[term, "alt"] <= 0 <= table.loc[term, "ust"]]
        text = f"{interval}. {alpha} iki taraflı H₀: β = a testinde: " + "; ".join(checks) + " (Denklem 7.9). "
        if len(crosses) > 1:
            text += (f"Sıfırı kapsayan aralıklar: {listing(crosses)}; bu katsayılar {alpha} düzeyinde sıfırdan "
                     "ayrıştırılamaz. ")
        elif crosses:
            text += f"Sıfırı kapsayan aralık: {crosses[0]}; bu katsayı {alpha} düzeyinde sıfırdan ayrıştırılamaz. "
        else:
            text += "Hiçbir aralık sıfırı kapsamıyor. "
        return text + ("Katsayılar farklı birimlerle ölçüldüğü için grafikteki yatay konumlar büyüklük sıralaması olarak "
                       "okunmamalıdır (§7.7)." + _exact_note(case))

    columns = (("katsayi", "katsayı", 3), ("sh", "SH", 3), ("t", "t", 2), ("p", "p", 3), ("alt", "GA alt", 3),
               ("ust", "GA üst", 3))
    exact = _exact(case)
    return interactive_step(
        number=5,
        title="Hipotez testi ile güven aralığı",
        note=NoteRef("7.7", 0, ("Tablo 7.4", "Şekil 7.4")),
        explanation=(
            "İki taraflı $H_0: \\beta_j = a$ testi ile güven aralığı aynı bilgiyi verir: $a$ aralığın dışındaysa $H_0$ "
            "reddedilir (Denklem 7.9). Tablo ana modelin katsayılarının çıkarımını, grafik aralıkları gösterir (Tablo 7.4 "
            "ve Şekil 7.4'teki gibi). Güven düzeyini değiştirin: hangi aralık sıfırı kapsıyor?"
        ),
        controls=(Choice("adim5_duzey", "Güven düzeyi", LEVELS, "0.95", help="Varsayılan: yüzde 95."),),
        build=build,
        checks=tuple(
            _check(f"Tablo: {case.name(term)}, {label}", TableTarget("tablo74", term, column), decimals)
            for term in regressors for column, label, decimals in columns if not exact or column == "katsayi"),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 6: tek taraflı test ------------------------------------------------------------------------------

def _step6(case: Case) -> LabStep:
    regressors = _regressors(case)
    term = _loosest(case)

    def build(choices) -> tuple:
        chosen, direction = choices["adim6_terim"], choices["adim6_yon"]
        t, df = E.ref("t6"), E.ref("sd_artik")
        p_value = {"sag": E.tsf(t, df), "sol": E.tcdf(t, df), "iki": E.mul(2, E.tsf(E.absolute(t), df))}[direction]
        label = {"sag": "Tek taraflı p = P(T > t)", "sol": "Tek taraflı p = P(T < t)",
                 "iki": "İki taraflı p = 2·P(T > |t|)"}[direction]
        return (
            ModelValue("t6", "m", "t", f"{capital(_name(case, chosen))} katsayısının t istatistiği", term=chosen,
                       decimals=3),
            Scalar("p_secilen", p_value, label, decimals=3, p_value=True),
            Scalar("p_iki6", E.mul(2, E.tsf(E.absolute(t), df)), "İki taraflı p = 2·P(T > |t|)", decimals=3,
                   p_value=True),
            HypothesisPlot("t", "t6", "sd_artik", f"{capital(_name(case, chosen))} katsayısı: yüzde 5 " + {
                "sag": "sağ kuyruk testi", "sol": "sol kuyruk testi", "iki": "iki taraflı test"}[direction],
                "t değeri", alternative=direction),
        )

    def note(state, choices) -> str:
        chosen, direction = choices["adim6_terim"], choices["adim6_yon"]
        s = state.scalars
        picked, both = s["p_secilen"], s["p_iki6"]
        if direction == "iki":
            return (f"{capital(phrase(case, chosen))} katsayısı için iki taraflı testte {_p(both)}. Sağ ya da sol kuyruk "
                    "seçin: tek taraflı p-değeri, gözlenen t seçilen yöndeyse iki taraflı değerin yarısıdır, ters yöndeyse "
                    "değildir (§7.8)." + _exact_note(case))
        same = (s["t6"] > 0) == (direction == "sag")
        text = (f"{capital(phrase(case, chosen))} katsayısı için tek taraflı {_p(picked)}, iki taraflı {_p(both)}. Yüzde 5 "
                f"tek taraflı testte H₀ {'reddedilir' if picked < 0.05 else 'reddedilemez'}; iki taraflı testte "
                f"{'reddedilir' if both < 0.05 else 'reddedilemez'}. ")
        if not same:
            text += ("Gözlenen t alternatifin tersi yönde: tek taraflı p-değeri büyüktür; test ters yöndeki büyük bir etkiyi "
                     "alternatifin kanıtı saymaz. ")
        return text + ("Test yönü sonuç görüldükten sonra değil, teori veya önceden belirlenmiş analiz planıyla seçilir; "
                       "daha küçük p-değeri için yön değiştirmek uygun değildir (§7.8)." + _exact_note(case))

    return interactive_step(
        number=6,
        title="Tek taraflı test: yön ne zaman kullanılabilir?",
        note=NoteRef("7.8", 0),
        explanation=(
            "Teori ve araştırma planı önceden katsayının pozitif olmasını öngörüyorsa $H_0: \\beta \\leq 0$, $H_1: \\beta > "
            "0$ kurulabilir. Tek taraflı p-değeri yalnız seçilen kuyruktaki alanı sayar. Yönü değiştirin: gözlenen t "
            "alternatifin tersi yönde olduğunda ne oluyor?"
        ),
        controls=(
            Choice("adim6_terim", "Sınanan katsayı", options(case, regressors), term,
                   help=f"Varsayılan: {_loosest_words(case, case.name(term))}."),
            Choice("adim6_yon", "Alternatif hipotez", DIRECTIONS, "sag", help="Varsayılan: H₁: β > 0."),
        ),
        build=build,
        checks=() if _exact(case) else (_scalar("t6", "t", 3), _scalar("p_secilen", "Tek taraflı p-değeri", 3),
                                        _scalar("p_iki6", "İki taraflı p-değeri", 3)),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 7: yazılım çıktısı -------------------------------------------------------------------------------

def _step7(case: Case) -> LabStep:
    y = case.roles[SONUC]
    regressors = _regressors(case)
    outputs = tuple(case.extra.get("output_options") or candidates(case))
    outcomes = [(y, case.name(y))]
    if positive(case, y):
        outcomes.append((log_column(case, y), log_label(case, y)))

    def build(choices) -> tuple:
        outcome, chosen = choices["adim7_bagimli"], tuple(choices["adim7_x"])
        logs = log_operations(case, case.frame, (y,)) if outcome != y else ()
        return (
            *(reload(case) if logs else ()),
            *logs,
            OLS("cikti", case.frame, outcome, chosen, f"Python çıktısı: {outcome} ~ {' + '.join(chosen)}"),
            ShowModel("cikti", "Statsmodels çıktısından çıkarım için temel bölüm", columns=COEF_QUANTITIES,
                      stats=("nobs", "df_resid", "r2")),
        )

    def note(state, choices) -> str:
        result = state.models["cikti"]
        chosen = choices["adim7_x"]
        k = len(chosen)
        n, df = int(result.nobs), int(result.df_resid)
        significant = [phrase(case, term) for term in chosen if result.pvalues[term] < 0.05]
        text = (f"Artık serbestlik derecesi n − k − 1 = {sayim(n)} − {k} − 1 = {sayim(df)}. `coef` katsayı tahmini, "
                "`std err` klasik EKK standart hatası, `t` ve `P>|t|` H₀: β = 0 için t istatistiği ve iki taraflı "
                "p-değeri, `[0.025 0.975]` yüzde 95 güven aralığıdır. ")
        text += (f"Yüzde 5 düzeyinde sıfırdan farklı eğimler: {listing(significant)}. " if significant
                 else "Yüzde 5 düzeyinde sıfırdan farklı eğim yok. ")
        exact = _exact_model(case, choices["adim7_bagimli"], tuple(chosen))
        if (result.pvalues < 0.0005).any():
            text += ("`P>|t|` sütunundaki 0.000 gerçek sıfır değildir; gösterim hassasiyetinden küçüktür (p < 0,001). ")
        return text + ("\"Covariance Type: nonrobust\" standart hataların homoskedastisiteye dayandığını bildirir (§7.9)."
                       + (EXACT_MULTI_NOTE if exact else ""))

    quantities = (("coef", "katsayı", 4), ("se", "std err", 3), ("t", "t", 3), ("p", "P>|t|", 3),
                  ("ci_low", "GA alt", 3), ("ci_high", "GA üst", 3))
    return interactive_step(
        number=7,
        title="Python çıktısını satır satır okumak",
        note=NoteRef("7.9", 0, ("Kod 7.1", "Kod 7.2")),
        explanation=(
            "Statsmodels çıktısında `coef`, `std err`, `t`, `P>|t|` ve `[0.025 0.975]` sütunları aynı katsayıyı farklı "
            "açılardan özetler. Önce bağımlı değişkeni ve gözlem sayısını, sonra ilgilenilen katsayı satırını okuyun. "
            "Modeli değiştirin: artık serbestlik derecesi ve katsayı satırları nasıl değişiyor?"
        ),
        controls=(
            Choice("adim7_bagimli", "Bağımlı değişken", tuple(outcomes), y, help=f"Varsayılan: {case.name(y)}."),
            MultiChoice("adim7_x", "Açıklayıcı değişkenler", options(case, outputs), regressors,
                        help="Varsayılan: ana modelin değişkenleri."),
        ),
        build=build,
        checks=stable_checks((
            *(_check(f"{case.name(term) if term != INTERCEPT else 'Sabit terim'}: {label}",
                     CoefTarget("cikti", term, quantity), decimals)
              for term in (INTERCEPT, *regressors) for quantity, label, decimals in quantities),
            _check("Gözlem sayısı", ModelTarget("cikti", "nobs"), 0),
            _check("Artık serbestlik derecesi", ModelTarget("cikti", "df_resid"), 0),
            _check("R²", ModelTarget("cikti", "r2"), 3),
        ), _exact(case)),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 8: makale tablosu ---------------------------------------------------------------------------------

def _step8(case: Case) -> LabStep:
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    regressors = _regressors(case)
    logs = positive(case, y)
    level, logged = (case.name(y) if case.own else capital(phrase(case, y))), log_label(case, y)
    columns = [(f"(1) {level}", "m")]
    operations: list = []
    if logs:
        ly = log_column(case, y)
        operations += [*(reload(case) if case.own else ()), *log_operations(case, case.frame, (y,)),
                       OLS("ml", case.frame, ly, regressors, f"Log sonuç modeli: {ly} ~ {' + '.join(regressors)}")]
        columns.append((f"(2) {logged}", "ml"))
    operations.append(RegressionTable(tuple(columns), (*regressors, INTERCEPT), "tablo75",
                                      "Makale tablosu (Tablo 7.5'teki gibi)"))
    exact = _exact(case)
    checks = []
    for heading, _ in columns:
        for name in (*regressors, INTERCEPT):
            checks.append(_check(f"{heading}: {case.name(name) if name != INTERCEPT else 'sabit'}",
                                 TableTarget("tablo75", name, heading), 3))
            checks.append(_check(f"{heading}: {case.name(name) if name != INTERCEPT else 'sabit'} (SH)",
                                 TableTarget("tablo75", f"{name}_sh", heading), 3))
        checks.append(_check(f"{heading}: R²", TableTarget("tablo75", "r2", heading), 3))
    if logs:
        logged_data = case.data.assign(**{log_column(case, y): np.log(case.data[y].astype(float))})
        model = _fit(case, log_column(case, y), regressors, data=logged_data)
        b = float(model.params[x])
        p = float(model.pvalues[x])
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "yıldızsız"
        held = "diğer değişkenler aynıyken " if len(regressors) > 1 else ""
        zero = negligible(replace(case, data=logged_data), b, log_column(case, y), x)
        percent = sayi(abs(100 * b), digits_for(100 * b, 1))
        caveat = rough(b) or " (daha hassas dönüşüm Konu 9'da)."
        takeaway = (f"Sütun (2)'de {phrase(case, x)} katsayısı {sayi(b, digits_for(b, 3))}: {held}{step_words(case, x)} "
                    f"{plural(case)} tahmin edilen {outcome_words(case)} {change(b, '%' + percent, zero)}"
                    + (caveat if caveat.startswith(" (") else "." + caveat)
                    + f" Bu katsayı {stars}: yıldız etkinin büyük olduğunu değil, p-değerinin eşiklerden küçük olduğunu "
                    "gösterir (*** p < 0,01, ** p < 0,05, * p < 0,10). Sütun (1) düzey–düzey, sütun (2) log–düzey "
                    "modelidir; katsayıların birimleri farklıdır (§7.10).")
    else:
        takeaway = (f"{capital(phrase(case, y))} değişkeninde sıfır ya da negatif değer olduğu için log sonuç sütunu kurulmaz. "
                    "Yıldız etkinin büyük olduğunu değil, p-değerinin eşiklerden küçük olduğunu gösterir (*** p < 0,01, "
                    "** p < 0,05, * p < 0,10; §7.10).")
    return LabStep(
        number=8,
        title="Makale tablosu: parantezler, yıldızlar ve notlar",
        note=NoteRef("7.10", 0, ("Tablo 7.5",)),
        explanation=(
            "Makaleler tam çıktı yerine sıkıştırılmış tablo verir. Parantezde ne olduğu (standart hata, t ya da güven "
            "aralığı) ve yıldız eşikleri tablo notundan okunur. "
            + ("Sütun (1) düzey–düzey, sütun (2) log–düzey modelidir; katsayıların birimleri farklıdır." if logs else
               "Tabloda ana model vardır.")
        ),
        operations=tuple(operations),
        checks=stable_checks(tuple(checks), exact, table="tablo75"),
        takeaway=takeaway + (" Uyum tam olduğu için standart hatalar ve yıldızlar yuvarlama hatasına duyarlıdır." if exact
                             else ""),
    )


# --- Adım 9: istatistiksel anlamlılık ve iktisadi önem --------------------------------------------------------

def _gap_control(case: Case) -> NumberChoice:
    if "adim9" in case.extra:
        return case.extra["adim9"]
    return gap_control(case, "adim9_fark", f"{case.name(case.roles[ACIKLAYICI])} farkı",
                       "Varsayılan: açıklayıcının yaklaşık bir standart sapması.")


def _house_terms(house: Case) -> tuple[str, ...]:
    return tuple(house.extra.get("default_regressors") or candidates(house))


def _step9(case: Case, house: Case) -> LabStep:
    """Alternatif örnekte ikinci tablo KIELMC modelidir; kendi verinde aynı dosyanın ana modeli (``m``) kullanılır,
    dosya yeniden okunmaz."""

    x = case.roles[ACIKLAYICI]
    control = _gap_control(case)
    unit = f" {short_unit(case, case.roles[SONUC])}" if short_unit(case, case.roles[SONUC]) else ""
    gap_unit = str(case.extra.get("gap_unit", "birimlik"))
    if case.own:
        table_case, model_name, terms, loose = case, "m", _regressors(case), _loosest(case)
        second_model: tuple = ()
        where, table_title = "Ana modelde", "Ana modelde seçilmiş katsayılar (Tablo 7.6'daki gibi)"
        context = "Tablo ana modelin katsayılarını yüzde 95 güven aralıklarıyla gösterir."
    else:
        y_house = house.roles[SONUC]
        table_case, model_name, terms = house, "h", _house_terms(house)
        loose = str(_fit(house, y_house, terms).pvalues.drop(INTERCEPT).idxmax())
        second_model = (*house.load, OLS("h", house.frame, y_house, terms,
                                         f"İkinci model: {y_house} ~ {' + '.join(terms)}"))
        where, table_title = str(house.extra["data_name"]), "İkinci modelde seçilmiş katsayılar (Tablo 7.6'daki gibi)"
        context = str(house.extra["significance_text"])
    table_unit = short_unit(table_case, table_case.roles[SONUC])
    table_unit = f" {table_unit}" if table_unit else ""

    def build(choices) -> tuple:
        gap = choices["adim9_fark"]
        return (
            ModelValue("x_alt", "m", "ci_low", "Temel katsayının yüzde 95 GA alt sınırı", term=x),
            ModelValue("x_ust", "m", "ci_high", "Temel katsayının yüzde 95 GA üst sınırı", term=x),
            ModelValue("x9", "m", "coef", "Temel katsayı", term=x, decimals=3),
            Scalar("fark_tahmin", E.mul(gap, E.ref("x9")), f"{level_text(gap)} {gap_unit} farkın nokta tahmini",
                   decimals=3),
            Scalar("fark_alt", E.mul(gap, E.ref("x_alt")), "Aralığın alt sınırı", decimals=3),
            Scalar("fark_ust", E.mul(gap, E.ref("x_ust")), "Aralığın üst sınırı", decimals=3),
            *second_model,
            CoefficientTable(model_name, terms, "tablo76", table_title, decimals=4),
        )

    def note(state, choices) -> str:
        gap = choices["adim9_fark"]
        s = state.scalars
        table = state.tables["tablo76"]
        row = table.loc[loose]
        d = digits_for(max(abs(s["fark_alt"]), abs(s["fark_ust"])), 3)
        text = (f"{level_text(gap)} {gap_unit} farkın nokta tahmini {sayi(s['fark_tahmin'], d)}{unit}, yüzde 95 aralığı "
                f"[{sayi(s['fark_alt'], d)}; {sayi(s['fark_ust'], d)}]{unit}: katsayının yalnız “anlamlı” olduğunu "
                "söylemekten daha açıklayıcıdır. ")
        if exact:
            return text + ("Uyum tam olduğu için aralıklar neredeyse tek bir noktadır; hangi katsayıların sıfırdan "
                           "ayrıştığı yuvarlama hatasına bağlıdır, bu yüzden yazılmaz. Büyüklük, belirsizlik aralığı ve "
                           "ekonomik bağlam birlikte raporlanır (§7.11)." + _exact_note(case))
        low, high = float(row["alt"]), float(row["ust"])
        r = digits_for(max(abs(low), abs(high)), 2)
        interval = f"[{sayi(low, r)}; {sayi(high, r)}]"
        if row["p"] >= 0.05 and low < 0 < high:
            sizes = ("hem küçük negatif hem büyük pozitif değerler" if high >= -2 * low else
                     "hem büyük negatif hem küçük pozitif değerler" if -low >= 2 * high else
                     "hem negatif hem pozitif değerler")
            text += (f"{where} {phrase(table_case, loose)} katsayısı {sayi(row['katsayi'], r)}{table_unit} "
                     f"({_p(row['p'])}), aralık {interval}: {sizes} veriyle uyumludur. Doğru sonuç “etki yok” değil, "
                     "tahminin belirsiz olduğudur. ")
        elif row["p"] >= 0.05:
            text += (f"{where} {phrase(table_case, loose)} katsayısı {sayi(row['katsayi'], r)}{table_unit} "
                     f"({_p(row['p'])}), aralık {interval}: tahmin belirsizdir. ")
        else:
            text += (f"{where} bütün katsayılar yüzde 5 düzeyinde sıfırdan ayrışır; p-değeri en büyük olan "
                     f"{phrase(table_case, loose)} katsayısının aralığı {interval}: aralığın genişliği büyüklük hakkındaki "
                     "belirsizliği gösterir. ")
        return text + "Büyüklük, belirsizlik aralığı ve ekonomik bağlam birlikte raporlanır (§7.11)."

    columns = (("katsayi", "katsayı", 4), ("sh", "SH", 4), ("p", "p", 3), ("alt", "GA alt", 4), ("ust", "GA üst", 4),
               ("t", "t", 2))
    exact = _exact(case)
    checks = [_scalar("fark_tahmin", "Farkın nokta tahmini", 3)]
    if not exact:
        checks += [_scalar("fark_alt", "Aralığın alt sınırı", 3), _scalar("fark_ust", "Aralığın üst sınırı", 3)]
    checks += [_check(f"{'Ana' if case.own else 'İkinci'} model: {table_case.name(term)}, {label}",
                      TableTarget("tablo76", term, column), decimals)
               for term in terms for column, label, decimals in columns if not exact or column == "katsayi"]
    return interactive_step(
        number=9,
        title="İstatistiksel anlamlılık ile iktisadi önem",
        note=NoteRef("7.11", 0, ("Tablo 7.6",)),
        explanation=(
            "Bir katsayının sıfırdan istatistiksel olarak farklı olması etkinin büyük olduğunu göstermez. Temel "
            f"açıklayıcıdaki bir farkın sonuçtaki karşılığını aralığıyla birlikte hesaplayın. {context} Farkı değiştirin."
        ),
        controls=(control,),
        build=build,
        checks=tuple(checks),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 10: raporlama --------------------------------------------------------------------------------------

def _step10(case: Case) -> LabStep:
    x = case.roles[ACIKLAYICI]
    regressors = _regressors(case)
    gap = float(_gap_control(case).default)
    unit = f" {short_unit(case, case.roles[SONUC])}" if short_unit(case, case.roles[SONUC]) else ""
    gap_unit = str(case.extra.get("gap_unit", "birimlik"))

    def build(choices) -> tuple:
        term = choices["adim10_terim"]
        return (
            ModelValue("b10", "m", "coef", f"{capital(_name(case, term))} katsayısı", term=term, decimals=3),
            ModelValue("sh10", "m", "se", "Standart hata", term=term, decimals=3),
            ModelValue("t10", "m", "t", "t istatistiği", term=term, decimals=2),
            ModelValue("p10", "m", "p", "p-değeri", term=term, decimals=3),
            ModelValue("alt10", "m", "ci_low", "Yüzde 95 GA alt sınırı", term=term, decimals=3),
            ModelValue("ust10", "m", "ci_high", "Yüzde 95 GA üst sınırı", term=term, decimals=3),
            Scalar("dort10", E.mul(gap, E.ref("b10")), f"{level_text(gap)} {gap_unit} farkın nokta tahmini", decimals=2),
        )

    def note(state, choices) -> str:
        term = choices["adim10_terim"]
        s = state.scalars
        others = listing([phrase(case, name) for name in regressors if name != term])
        held = f"{others} sabitken " if others else ""
        question = str(case.extra.get("report_question", f"sonuç değişkeni {phrase(case, case.roles[SONUC])} ile "
                                                          "açıklayıcı değişkenler arasındaki ilişkiyi"))
        b = s["b10"]
        d = digits_for(b, 3)
        zero = negligible(case, b, case.roles[SONUC], term)
        limits = digits_for(max(abs(s["alt10"]), abs(s["ust10"])), 3)
        se = s["sh10"]
        se_text = ("SH ≈ 0" if se <= 1e-9 * max(abs(b), np.finfo(float).tiny) else
                   f"SH = {sayi(se, digits_for(se, 3))}")
        paragraph = (f"{capital(question)} inceleyen EKK modelinde {phrase(case, term)} katsayısı {sayi(b, d)} olarak "
                     f"tahmin edilmiştir ({se_text}; {_t(s['t10'])}; "
                     f"{_p(s['p10'])}; yüzde 95 GA [{sayi(s['alt10'], limits)}; {sayi(s['ust10'], limits)}]). Buna göre "
                     f"{held}{step_words(case, term)} {plural(case)} tahmin edilen {outcome_words(case)} örneklemde "
                     f"ortalama {change(b, sayi(abs(b), d) + unit, zero)}. ")
        if s["p10"] >= 0.05:
            paragraph += "Katsayı yüzde 5 düzeyinde sıfırdan istatistiksel olarak ayrışmamaktadır. "
        if term == x:
            paragraph += (f"{level_text(gap)} {gap_unit} farkın nokta tahmini yaklaşık "
                          f"{sayi(s['dort10'], digits_for(s['dort10'], 2))}{unit}. ")
        if case.own:
            paragraph += (f"{capital(phrase(case, term))} rastgele atanmadıysa bulgu nedensel etki olarak "
                          "yorumlanmamalıdır.")
            text = f"Örnek raporlama paragrafı:\n\n> {paragraph}\n\n"
        else:
            paragraph += (f"{capital(phrase(case, term))} gözlemsel veride rastgele atanmadığından bulgu nedensel etki "
                          "olarak yorumlanmamıştır.")
            text = f"Örnek raporlama paragrafı: “{paragraph}” "
        return text + ("Paragraf beş bileşeni birlikte taşır: katsayı büyüklüğü, standart hata, test sonucu, güven "
                       "aralığı ve yorum sınırı (§7.13)." + _exact_note(case))

    exact = _exact(case)
    return interactive_step(
        number=10,
        title="Bütünleşik uygulama: bir sonuç nasıl raporlanır?",
        note=NoteRef("7.13", 0, ("Adım 1–5", "Örnek raporlama paragrafı")),
        explanation=(
            "İyi bir rapor beş bileşeni birlikte taşır: katsayı büyüklüğü, standart hata, test sonucu, güven aralığı ve "
            "yorum sınırı (§7.13, Adım 1–5). Raporlanan katsayıyı değiştirin; paragraf seçiminize göre yeniden yazılır."
        ),
        controls=(Choice("adim10_terim", "Raporlanan katsayı", options(case, regressors), x,
                         help=f"Varsayılan: {case.name(x)} katsayısı."),),
        build=build,
        checks=(_scalar("b10", "Katsayı", 3), _scalar("dort10", "Farkın nokta tahmini", 2)) if exact else (
            _scalar("b10", "Katsayı", 3), _scalar("sh10", "Standart hata", 3), _scalar("t10", "t", 2),
            _scalar("alt10", "Yüzde 95 GA alt", 3), _scalar("ust10", "Yüzde 95 GA üst", 3),
            _scalar("dort10", "Farkın nokta tahmini", 2),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Tanım ------------------------------------------------------------------------------------------------------

def build(case: Case) -> LabSpec:
    """Konu 7 uygulamasını verilen veriyle kurar; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    house = second(case)
    labels = labels_of(case)
    labels.update(labels_of(house))
    if positive(case, case.roles[SONUC]):
        labels.setdefault(log_column(case, case.roles[SONUC]), log_label(case, case.roles[SONUC]))
    labels[INTERCEPT] = "Sabit terim"
    spec = LabSpec(
        topic_key=TOPIC,
        title=str(case.extra.get("title", TITLE)),
        note_section="7",
        steps=(_step1(case), _step2(case), _step3(case), _step4(case), _step5(case), _step6(case), _step7(case),
               _step8(case), _step9(case, house), _step10(case)),
        labels=tuple(labels.items()),
        source=case.source,
    )
    return with_app_values(spec)


# --- Alternatif örnek ----------------------------------------------------------------------------------------

def _house() -> Case:
    return house_case(
        regressors=("area", "rooms", "baths"),
        default_regressors=("area", "rooms", "baths"),
        data_name="KIELMC'de (1978)",
        significance_text=("KIELMC'nin 1978 satışlarında (fiyat ~ büyüklük + oda + banyo) oda sayısının katsayısı pozitif "
                           "ama belirsizdir."),
    )


def alternative_case() -> Case:
    return wage2_case(
        ALT_OUTPUT,
        title="Uygulama: Tek Katsayı İçin Hipotez Testleri (WAGE2, KIELMC)",
        default_regressors=ALT_REGRESSORS,
        house=_house(),
        step2_title="WAGE2: eğitim katsayısı için t testi",
        adim1_ornek=(
            NumberChoice("adim1_b", "Katsayı tahmini β̂", -20.0, 20.0, 8.26, 0.02,
                         help="Varsayılan: kıdem katsayısı (8,2568, yuvarlanmış)."),
            NumberChoice("adim1_se", "Standart hata se(β̂)", 0.1, 10.0, 2.5, 0.1,
                         help="Varsayılan: kıdem katsayısının standart hatası (2,498, yuvarlanmış).", decimals=1),
        ),
        adim4_ornek=(
            NumberChoice("adim4_b", "Katsayı tahmini β̂", -20.0, 20.0, 8.26, 0.02,
                         help="Varsayılan: kıdem katsayısı (8,2568, yuvarlanmış)."),
            NumberChoice("adim4_se", "Standart hata se(β̂)", 0.1, 10.0, 2.5, 0.1,
                         help="Varsayılan: kıdem katsayısının standart hatası (2,498, yuvarlanmış).", decimals=1),
        ),
        adim2_a=NumberChoice("adim2_a", "İkinci sıfır hipotezindeki değer a", 0.0, 150.0, 65.0, 0.5,
                             help="Varsayılan: H₀: β = 65 (eğitim; tahminin yaklaşık 1,5 standart hata altı).",
                             decimals=1),
        adim9=NumberChoice("adim9_fark", "Eğitim farkı (yıl)", 1, 8, 4, 1, help="Varsayılan: dört yıl.", integer=True,
                           decimals=0),
        gap_unit="yıllık",
        report_question="eğitim, iş deneyimi ve kıdemin aylık kazançla ilişkisini",
    )


@cache
def alternative() -> LabSpec:
    return build(alternative_case())


STORY = ("Ücret modeli WAGE2'dir (935 erkek çalışan, 1980; aylık kazanç ~ eğitim + deneyim + kıdem). Sayısal örnekler "
         "ve p-değeri adımları kıdem katsayısını kullanır; iktisadi önem adımının ikinci verisi KIELMC'nin 1978 "
         "satışlarıdır (oda sayısının katsayısı pozitif ama belirsiz).")


# --- Kendi verin ---------------------------------------------------------------------------------------------

CUSTOM = custom_lab(
    build,
    sample_employed,
    ("Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sonuç ve temel açıklayıcı değişken zorunludur ve sayısal olmalıdır. "
     "Ana model sonucun temel açıklayıcı ve ek değişkenlere (en çok 3) göre regresyonudur; sayısal örnekler bu modelin "
     "p-değeri en büyük katsayısından kurulur. Log sonuçlu sütunlar için sonucun bütün değerleri pozitif olmalı. "
     f"{ROW_RULE}"),
    roles((1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),
    "Ana modelin değişkenleridir.",
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
