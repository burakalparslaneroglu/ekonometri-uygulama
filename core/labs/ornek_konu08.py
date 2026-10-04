"""Konu 8 genel uygulaması: çoklu doğrusal kısıtlar ve F testi.

Notlardaki yedi adım (``core.labs.konu08``) aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde
yazılır: F testinin karar kuralı ve ortak p-değeri, yazılım çıktısındaki iki farklı F, kısıtlı ve kısıtsız model
karşılaştırması ile F'nin iki formülü, genel anlamlılık testi, tek kısıtta F = t², ikinci bir veride ortak test ile ayrı
testlerin farkı ve makale tablosunda ortak testler.

Alternatif örnekte ücret modeli WAGE2'dir (935 erkek çalışan, 1980; aylık kazanç ~ eğitim + deneyim + kıdem; notlardaki
gibi deneyim ve kıdem birlikte sınanır). Genel testin seçeneklerinde notlardaki bakmakla yükümlü kişi sayısının yerinde
IQ puanı vardır. İkinci veri KIELMC'nin 1978 satışlarıdır (fiyat bin dolar ~ büyüklük + oda + banyo): büyüklük modelde
kalırken oda ve banyo sayısı birlikte sınanır; oda sayısının katsayısı tek başına anlamlı değildir, ortak test
anlamlıdır (notlardaki HPRICE1 örneğindeki yatak odası gibi). "Kendi verini yükle" seçeneğinde model sonucun temel
açıklayıcı ve ek değişkenlere göre regresyonudur; ortak testin varsayılanı ek değişkenlerdir (temel açıklayıcı modelde
kalır). İkinci veri adımı aynı dosyanın ayrı bir çerçevesidir. Uyum tamsa (R² ≈ 1) F, t ve p yuvarlama hatasıdır: bu
kontroller çıkarılır, metinler nedenini söyler.

Etkileşim notlardaki gibidir: birlikte sınanan katsayılar ve anlamlılık düzeyi (Adım 1), kısıtlı modelden çıkarılan
değişkenler (Adım 3 ve 6), genel testteki model (Adım 4) ve F = t² bağlantısındaki katsayı (Adım 5). Ortak testler
klasik (homoskedastik) EKK varsayımlarına dayanır; dayanıklı ortak test Konu 12'nin konusudur.
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
    custom_lab,
    digits_for,
    exact_multi,
    house_case,
    labels_of,
    listing,
    log_column,
    log_label,
    log_operations,
    options,
    p_text,
    phrase,
    positive,
    roles,
    sample_employed,
    second,
    wage2_case,
)
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

TOPIC = "konu08"
TITLE = "Uygulama: Çoklu Kısıtlar ve F Testi"
ALPHAS = (("0.1", "%10"), ("0.05", "%5"), ("0.01", "%1"))
SUPERSCRIPT = str.maketrans("0123456789−", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
TINY = 1e-3
"""Bu değerden küçük p-değerleri çıktıda bilimsel gösterimle yazılır; uygulama onları 10ᵏ ile ölçekleyerek gösterir."""
FRAGILE = frozenset(("F_ortak", "F_cikti", "p_cikti_olcek", "genel_p_olcek", "F_ssr", "F_r2", "F_genel", "F_genel_r2",
                     "t_tek", "t_kare", "F_tek", "F_h", "p_h", "p_ayri"))
"""Uyum tamken yuvarlama hatasına duyarlı skalerler (F, t, p)."""


def _check(label: str, target, decimals: int) -> Check:
    return Check(label, target, 0.0, decimals)


def _scalar(name: str, label: str, decimals: int) -> Check:
    return _check(label, ScalarTarget(name), decimals)


def _name(case: Case, column: str) -> str:
    return case.name(column) if case.own else phrase(case, column)


def _regressors(case: Case) -> tuple[str, ...]:
    """Kısıtsız model: alternatif örnekte notlardaki gibi eğitim, deneyim ve kıdem; kendi verinde bütün seçenekler."""

    return tuple(case.extra.get("main") or candidates(case))


def _tested(case: Case) -> tuple[str, ...]:
    """Ortak testin varsayılanı: alternatif örnekte deneyim ve kıdem; kendi verinde ek değişkenler (temel açıklayıcı
    modelde kalır), ek değişken yoksa temel açıklayıcının kendisi."""

    if "tested" in case.extra:
        return tuple(case.extra["tested"])
    regressors = _regressors(case)
    return regressors[1:] if len(regressors) > 1 else regressors


def _fit(case: Case, outcome: str, regressors, data=None):
    """Metinler ve varsayılanlar için modelin kendisi (uygulamanın hesabı işlemlerle yapılır)."""

    return smf.ols(f"{outcome} ~ {' + '.join(regressors)}", data=case.data if data is None else data).fit()


def _exact(case: Case) -> bool:
    return case.own and exact_multi(case, case.roles[SONUC], _regressors(case))


def _robust_checks(checks, exact: bool, table: str | None = None) -> tuple[Check, ...]:
    """Uyum tamsa standart hataya, t'ye, p'ye ve F'ye bağlı kontroller çıkarılır."""

    kept = stable_checks(tuple(checks), exact, table)
    if not exact:
        return kept
    return tuple(check for check in kept if not (isinstance(check.target, ScalarTarget) and check.target.name in FRAGILE)
                 and not (isinstance(check.target, TableTarget) and check.target.row in ("ortak_F", "ortak_p", "genel_F")))


def _power(exponent: int) -> str:
    """10'un kuvveti üst simgeyle: 10⁻³¹."""

    return "10" + str(exponent).replace("-", "−").translate(SUPERSCRIPT)


def _sci(value: float, decimals: int = 3) -> str:
    """Çok küçük bir p-değerinin Türkçe bilimsel yazımı (3,41 × 10⁻⁴¹); diğerleri ``decimals`` basamakla."""

    if value <= 0 or not math.isfinite(value):
        return sayi(0.0, decimals)
    if value >= TINY:
        return sayi(value, decimals)
    exponent = math.floor(math.log10(value))
    mantissa = value / 10 ** exponent
    if round(mantissa, 2) >= 10:
        mantissa, exponent = mantissa / 10, exponent + 1
    return f"{sayi(mantissa, 2)} × {_power(exponent)}"


def _f_text(value: float) -> str:
    """F'nin yazımı; uyum tamken F yuvarlama gürültüsüyle çok büyüktür: "F > 10⁶"."""

    if not math.isfinite(value) or abs(value) >= 1e6:
        return "F > 10⁶"
    return f"F = {sayi(value, 2)}"


def _percent(value: float) -> str:
    return "%" + sayi(100 * value, 0)


def _names(case: Case, terms: tuple[str, ...], raw: bool) -> str:
    """Değişkenlerin sıralı adı: ``raw`` ise açıklama ve etiketler için dosyadaki ad (arayüz kaçırır), değilse Markdown
    metni için ``phrase`` (kendi verinde tırnaklı ve kaçırılmış)."""

    return listing([_name(case, term) if raw else phrase(case, term) for term in terms])


def _hypothesis(case: Case, terms: tuple[str, ...], raw: bool = True) -> str:
    """Sıfır hipotezi: alternatif örnekte notlardaki gibi "H₀: β_deneyim = 0, β_kıdem = 0"; kendi verinde adlarla."""

    symbols = dict(case.extra.get("symbols", {}))
    if all(term in symbols for term in terms):
        return "H₀: " + ", ".join(f"{symbols[term]} = 0" for term in terms)
    return f"H₀: {_names(case, terms, raw)} {'katsayısı' if len(terms) == 1 else 'katsayılarının hepsi'} sıfır"


def _subject(case: Case, terms: tuple[str, ...], raw: bool = False) -> str:
    """Cümle öznesi: "iş deneyimi ve kıdem katsayıları" ya da "eğitim katsayısı"."""

    names = _names(case, terms, raw)
    return f"{names} katsayıları" if len(terms) > 1 else f"{names} katsayısı"


def _genitive(case: Case, terms: tuple[str, ...]) -> str:
    """Tamlayan: "iş deneyimi ve kıdem katsayılarının birlikte" ya da "eğitim katsayısının" (… sıfır olduğu)."""

    names = _names(case, terms, False)
    return f"{names} katsayılarının birlikte" if len(terms) > 1 else f"{names} katsayısının"


def _together(terms: tuple[str, ...]) -> str:
    return " birlikte" if len(terms) > 1 else ""


def _heading(case: Case, terms: tuple[str, ...], restricted: bool) -> str:
    kind = "Kısıtlı model" if restricted else "Kısıtsız model"
    return f"{kind}: {listing([_name(case, term) for term in terms])}"


def _load(case: Case) -> tuple:
    y = case.roles[SONUC]
    regressors = _regressors(case)
    return (*case.load, OLS("m", case.frame, y, regressors, f"Kısıtsız model: {y} ~ {' + '.join(regressors)}"))


def _df(case: Case) -> int:
    return len(case.data) - len(_regressors(case)) - 1


# --- Adım 1: karar kuralı ve p-değeri -------------------------------------------------------------------

def _step1(case: Case) -> LabStep:
    regressors, tested = _regressors(case), _tested(case)
    df = _df(case)
    exact = _exact(case)
    terms_control = MultiChoice(
        "adim1_x", "Birlikte sınanan katsayılar", options(case, regressors), tested,
        help=f"Varsayılan: {listing([case.name(term) for term in tested])}. Bütün katsayılar seçilirse genel anlamlılık "
             "testi olur.")
    alpha_control = Choice("adim1_alfa", "Anlamlılık düzeyi α", ALPHAS, "0.05", help="Varsayılan: yüzde 5.")

    def build(choices) -> tuple:
        terms, alpha = tuple(choices["adim1_x"]), float(choices["adim1_alfa"])
        q = len(terms)
        same = terms == tested and alpha == 0.05
        title = (f"Ortak test (Şekil 8.1'deki gibi): {_subject(case, terms, True)}{_together(terms)} sınanır, "
                 f"F({q}, {df})" if same else f"{capital(_subject(case, terms, True))}{_together(terms)} sınanır: "
                                              f"F({q}, {df}), yüzde {sayi(100 * alpha, 0)} düzeyi")
        return (
            *_load(case),
            JointTest("F_ortak", "p_ortak", "m", terms, f"Ortak F testi, {_hypothesis(case, terms)}"),
            ModelValue("sd_artik", "m", "df_resid", "Payda serbestlik derecesi n − k − 1", decimals=0),
            Scalar("pay_sd", E.const(q), "Pay serbestlik derecesi q (kısıt sayısı)", decimals=0),
            Scalar("kritik_F", E.finv(round(1 - alpha, 6), E.ref("pay_sd"), E.ref("sd_artik")),
                   f"Kritik değer: F(q, n − k − 1) dağılımının üst yüzde {sayi(100 * alpha, 0)} noktası", decimals=2),
            HypothesisPlot("f", "F_ortak", "pay_sd", title, "F değeri", alpha=alpha, alternative="sag", df2="sd_artik"),
        )

    def note(state, choices) -> str:
        terms, alpha = tuple(choices["adim1_x"]), float(choices["adim1_alfa"])
        s = state.scalars
        f, critical, p = s["F_ortak"], s["kritik_F"], s["p_ortak"]
        if exact:
            return (f"{_percent(alpha)} düzeyinde kritik değer {sayi(critical, 2)}. Uyum tam olduğu için kısıtsız modelin "
                    "artık değişkenliği sıfıra çok yakındır; F çok büyük ve yuvarlama gürültüsüdür, karar yazılmaz."
                    + EXACT_MULTI_NOTE)
        rejected = f > critical
        text = (f"Gözlenen {_f_text(f)}; {_percent(alpha)} düzeyinde kritik değer {sayi(critical, 2)}. "
                f"F {'>' if rejected else '≤'} kritik değer olduğundan H₀ {'reddedilir' if rejected else 'reddedilemez'}; "
                f"{'ortak ' if len(terms) > 1 else ''}{p_text(p)} ile karar aynıdır (p {'<' if p < alpha else '≥'} α). "
                "F negatif olamaz ve reddetme bölgesi yalnız üst kuyruktadır. ")
        if len(terms) == 1:
            return text + "Tek kısıtta F testi iki taraflı t testiyle aynı kararı verir: F = t² (§8.8)."
        if len(terms) == len(regressors):
            text += "Bütün eğimlerin birlikte sınanması genel anlamlılık testidir; çıktıdaki F-statistic budur (§8.7). "
        if rejected:
            return text + ("Ortak testin reddedilmesi sınanan katsayıların hepsinin ayrı ayrı sıfırdan farklı olduğunu "
                           "göstermez; kısıtlardan en az birinin veriyle uyumsuz olduğunu gösterir (§8.5).")
        return text + "H₀'ın reddedilememesi katsayıların birlikte sıfır olduğunu kanıtlamaz (§8.5)."

    q = len(tested)
    return interactive_step(
        number=1,
        title="Karar kuralı: kritik değer ve ortak p-değeri",
        note=NoteRef("8.5", 0, ("Şekil 8.1",)),
        explanation=(
            "F istatistiği kısıtların yarattığı uyum kaybını kısıtsız modeldeki artık değişkenliğine göre ölçer; büyük "
            "değerler $H_0$ aleyhine kanıttır ve reddetme bölgesi F dağılımının üst kuyruğundadır. "
            f"Model `{case.roles[SONUC]} ~ {' + '.join(regressors)}`; varsayılan hipotezde {_subject(case, tested)}"
            f"{_together(tested)} sıfırdır: pay serbestlik derecesi $q = {q}$, payda serbestlik derecesi $n - k - 1 = "
            f"{df}$. Birlikte "
            "sınanan katsayıları ve anlamlılık düzeyini değiştirin."
        ),
        controls=(terms_control, alpha_control),
        build=build,
        checks=_robust_checks((
            _scalar("pay_sd", "Pay serbestlik derecesi", 0),
            _scalar("sd_artik", "Payda serbestlik derecesi", 0),
            _scalar("kritik_F", "Kritik değer", 2),
            _scalar("F_ortak", "Gözlenen F", 2),
        ), exact),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 2: yazılım çıktısında iki farklı F -----------------------------------------------------------

def _scaled(name: str, source: str, value: float, label: str) -> tuple[Scalar, ...]:
    """Çok küçük bir p-değeri ekranda 10ᵏ ile ölçeklenmiş olarak (çıktıdaki bilimsel gösterimin anlamlı basamakları);
    değer küçük değilse ya da sıfıra yuvarlanıyorsa ölçeklenmez."""

    if not (0 < value < TINY) or not math.isfinite(value):
        return ()
    exponent = -math.floor(math.log10(value))
    if round(value * 10 ** exponent, 2) >= 10:
        exponent -= 1
    return (Scalar(name, E.mul(E.ref(source), 10.0 ** exponent),
                   f"{label} × {_power(exponent)} (çıktıda {value:.2e})", decimals=2),)


def _step2(case: Case) -> LabStep:
    y = case.roles[SONUC]
    regressors, tested = _regressors(case), _tested(case)
    exact = _exact(case)
    model = _fit(case, y, regressors)
    overall_p = float(model.f_pvalue)
    joint = model.f_test(", ".join(f"{term} = 0" for term in tested))
    joint_p = float(np.asarray(joint.pvalue).squeeze())
    restriction = ", ".join(f"{term} = 0" for term in tested)
    operations = (
        ShowModel("m", "Kısıtsız modelin temel Python çıktısı (Kod 8.2'deki gibi)", columns=("coef", "se", "t", "p"),
                  stats=("nobs", "df_resid", "r2", "f", "f_p")),
        ModelValue("n_m", "m", "nobs", "No. Observations: gözlem sayısı n", decimals=0),
        ModelValue("sd_m", "m", "df_resid", "Df Residuals: n − k − 1", decimals=0),
        Scalar("sd_model", E.sub(E.sub(E.ref("n_m"), E.ref("sd_m")), 1), "Df Model: eğim sayısı k", decimals=0),
        ModelValue("genel_F", "m", "f", "F-statistic: genel anlamlılık testinin F istatistiği", decimals=2),
        ModelValue("genel_p", "m", "f_p", "Prob (F-statistic): genel testin p-değeri", decimals=3),
        *(() if exact else _scaled("genel_p_olcek", "genel_p", overall_p, "Prob (F-statistic)")),
        JointTest("F_cikti", "p_cikti", "m", tested,
                  f'model.f_test("{restriction}"): {_subject(case, tested, True)}{_together(tested)}', decimals=4),
        *(() if exact else _scaled("p_cikti_olcek", "p_cikti", joint_p, "Ortak p-değeri")),
    )
    scaled = {op.name for op in operations if isinstance(op, Scalar)}

    def note(state) -> str:
        s = state.scalars
        names = listing([phrase(case, term) for term in regressors])
        if exact:
            return ("Çıktıda iki F vardır: `F-statistic` bütün eğimlerin birlikte sıfır olduğu genel testi, `f_test` "
                    f"yalnız {_genitive(case, tested)} sıfır olduğu hipotezini sınar. Uyum tam olduğu için iki F de "
                    "yuvarlama gürültüsüyle çok büyüktür; p-değerleri ve kararlar yazılmaz." + EXACT_MULTI_NOTE)
        if tuple(tested) == tuple(regressors):
            text = (f"Bu modelde sınanan katsayılar bütün eğimlerdir ({names}): `f_test` ile genel test aynı hipotezi "
                    f"sınar, iki F de {sayi(s['genel_F'], 2)} ve p-değeri {_sci(s['genel_p'])}. ")
        else:
            first, second = s["genel_p"] < 0.05, s["p_cikti"] < 0.05
            decision = ("iki hipotez de reddedilir" if first and second else
                        "iki hipotez de reddedilemez" if not (first or second) else
                        f"genel test {'reddedilir' if first else 'reddedilemez'}, özel ortak test "
                        f"{'reddedilir' if second else 'reddedilemez'}")
            text = (f"Aynı çıktıda iki F vardır: {sayi(s['genel_F'], 2)} bütün eğimlerin ({names}) birlikte sıfır olduğu "
                    f"genel testi, {sayi(s['F_cikti'], 2)} yalnız {_genitive(case, tested)} sıfır olduğu özel ortak testi "
                    f"sınar. p-değerleri {_sci(s['genel_p'])} ve {_sci(s['p_cikti'])}; yüzde 5 düzeyinde {decision}. ")
        if scaled:
            text += ("Çıktıdaki bilimsel gösterim 10'un kuvvetini yazar; ölçü kutusu aynı p-değerini 10'un kuvvetiyle "
                     "çarparak anlamlı basamaklarıyla gösterir. ")
        result = state.models["m"]
        loose = [term for term in tested if result.pvalues[term] >= 0.05]
        if len(tested) > 1 and loose and s["p_cikti"] < 0.05:
            term = loose[0]
            text += (f"{capital(phrase(case, term))} katsayısının ayrı {p_text(float(result.pvalues[term]))} iken ortak "
                     "testin reddedilmesi çelişki değildir: ortak test kısıtları bir bütün olarak değerlendirir (§8.6).")
        elif len(tested) > 1:
            text += "Ortak test kısıtları bir bütün olarak değerlendirir; ayrı t testlerinin toplamı değildir (§8.6)."
        else:
            text += "Tek kısıtta ortak test t testinin karesidir: F = t² (§8.8)."
        return text

    checks = [
        *(_check(f"Kod 8.2 karşılığı: {case.name(term) if term != INTERCEPT else 'sabit terim'}, {label}",
                 CoefTarget("m", term, quantity), decimals)
          for term in (INTERCEPT, *regressors)
          for quantity, label, decimals in (("coef", "coef", 4), ("se", "std err", 3), ("t", "t", 3), ("p", "P>|t|", 3))),
        _check("No. Observations", ModelTarget("m", "nobs"), 0),
        _check("Df Residuals", ModelTarget("m", "df_resid"), 0),
        _scalar("sd_model", "Df Model", 0),
        _check("R-squared", ModelTarget("m", "r2"), 3),
        _check("F-statistic", ModelTarget("m", "f"), 2),
        *(_scalar(name, label, 2) for name, label in (("genel_p_olcek", "Prob (F-statistic), ölçekli"),
                                                      ("p_cikti_olcek", "F test, p, ölçekli")) if name in scaled),
        _scalar("F_cikti", "F test, F", 4),
        _scalar("sd_m", "F test, df_denom", 0),
    ]
    return LabStep(
        number=2,
        title="Python çıktısında iki farklı F",
        note=NoteRef("8.6", 0, ("Kod 8.1", "Kod 8.2")),
        explanation=(
            "Statsmodels özetindeki `F-statistic` ve `Prob (F-statistic)` bütün eğimlerin birlikte sıfır olduğu genel "
            "anlamlılık testidir. Kullanıcının yazdığı kısıtlar ise `f_test` ile ayrıca sınanır: "
            f"`model.f_test(\"{restriction}\")` yalnız {_genitive(case, tested)} sıfır olduğu hipotezini sınar. Çok küçük "
            "p-değerleri çıktıda bilimsel gösterimle yazılır"
            + (f" (bu modelde genel testin p-değeri {overall_p:.2e} = {_sci(overall_p)})."
               if 0 < overall_p < TINY and not exact else " (ör. 3.41e-41 = 3,41 × 10⁻⁴¹).")
        ),
        operations=operations,
        checks=_robust_checks(tuple(checks), exact),
        note_for=lambda state, choices: note(state),
    )


# --- Adım 3: kısıtlı ve kısıtsız model --------------------------------------------------------------------

def _need(number: int, title: str, note: NoteRef, lead: str) -> LabStep:
    return LabStep(number=number, title=title, note=note,
                   explanation=f"{lead} Bu adım için en az iki açıklayıcı değişken gerekir: en az bir ek sayısal değişken "
                               "seçin (kısıtlı modelde en az bir değişken kalır).")


def _ssr_decimals(case: Case, outcome: str, regressors: tuple[str, ...], kept: tuple[str, ...], exact: bool) -> int:
    """Artık kareleri toplamının basamağı: en az üç, küçük değerlerde en az üç anlamlı basamak. Uyum tamsa kısıtsız
    modelin SSR'si sıfıra çok yakındır; basamak kısıtlı modelden alınır."""

    ssr = _fit(case, outcome, kept if exact else regressors).ssr
    return digits_for(float(ssr), 3)


def _restricted_note(s: dict, q: int, d: int, exact: bool) -> str:
    if exact:
        return (f"Kısıtlı modelin SSR'si {sayi(s['ssr_k'], d)}, kısıtsız modelinki sıfıra çok yakındır: uyum tam olduğu "
                "için F'nin paydası yuvarlama gürültüsüdür ve iki formülün sonucu yazılmaz. Değişken çıkarmak uyumu "
                "artırmaz: kısıtlı modelin SSR'si büyük, R²'si küçüktür (§8.6)." + EXACT_MULTI_NOTE)
    shown = [round(s[name], 3) for name in ("r2_s", "r2_k")]
    rough = ((shown[0] - shown[1]) / q) / ((1 - shown[0]) / s["sd_s"]) if shown[0] < 1 else math.nan
    text = (f"SSR formülü: [({sayi(s['ssr_k'], d)} − {sayi(s['ssr_s'], d)})/{q}] / [{sayi(s['ssr_s'], d)}/"
            f"{sayim(s['sd_s'])}] = {sayi(s['F_ssr'], 2)}. R² formülü: [({sayi(s['r2_s'], 5)} − {sayi(s['r2_k'], 5)})/{q}] / "
            f"[(1 − {sayi(s['r2_s'], 5)})/{sayim(s['sd_s'])}] = {sayi(s['F_r2'], 2)}. İki formül aynı sonucu verir; çünkü "
            "iki model aynı bağımlı değişkeni ve aynı gözlemleri kullanır. Değişken çıkarmak uyumu artırmaz: kısıtlı "
            "modelin SSR'si büyük, R²'si küçüktür. Soru, bu uyum kaybının kısıtsız modeldeki artık değişkenliğine göre "
            "büyük olup olmadığıdır (§8.6). ")
    if math.isfinite(rough) and sayi(rough, 2) != sayi(s["F_r2"], 2):
        return text + (f"R² formülünde beş basamaklı R² kullanılır; üç basamağa yuvarlanmış R² ile hesap "
                       f"{sayi(rough, 2)} verirdi.")
    return text + "R² formülünde beş basamaklı R² kullanılır; bu modelde üç basamaklı R² de aynı sonucu verir."


def _step3(case: Case) -> LabStep:
    y = case.roles[SONUC]
    regressors, tested = _regressors(case), _tested(case)
    title = "Kısıtlı ve kısıtsız model: iki formül aynı F'yi verir"
    note_ref = NoteRef("8.6", 0, ("Tablo 8.1",))
    lead = ("Kısıtlı model sıfır hipotezini modele yükler: sınanan değişkenler çıkarılır, model kalan değişkenlerle "
            "tahmin edilir.")
    if len(regressors) < 2:
        return _need(3, title, note_ref, lead)
    exact = _exact(case)
    kept_default = tuple(term for term in regressors if term not in tested)
    d = _ssr_decimals(case, y, regressors, kept_default, exact)
    dropped = MultiChoice("adim3_x", "Kısıtlı modelden çıkarılan (birlikte sınanan) değişkenler",
                          options(case, regressors), tested,
                          help=f"Varsayılan: {listing([case.name(term) for term in tested])}; kısıtlı model "
                               f"{listing([case.name(term) for term in kept_default])} içerir. Kısıtlı modelde en az "
                               "bir değişken kalır (yalnız sabitli model genel testtir: Adım 4).",
                          maximum=len(regressors) - 1)

    def build(choices) -> tuple:
        chosen = tuple(choices["adim3_x"])
        kept = tuple(term for term in regressors if term not in chosen)
        ssr = E.div(E.div(E.sub(E.ref("ssr_k"), E.ref("ssr_s")), E.ref("q3")), E.div(E.ref("ssr_s"), E.ref("sd_s")))
        r2 = E.div(E.div(E.sub(E.ref("r2_s"), E.ref("r2_k")), E.ref("q3")),
                   E.div(E.sub(1, E.ref("r2_s")), E.ref("sd_s")))
        same = chosen == tested
        return (
            OLS("mk", case.frame, y, kept, f"Kısıtlı model: {y} ~ {' + '.join(kept)}"),
            ModelValue("ssr_k", "mk", "ssr", "SSR_R: kısıtlı modelin artık kareleri toplamı", decimals=d),
            ModelValue("ssr_s", "m", "ssr", "SSR_UR: kısıtsız modelin artık kareleri toplamı", decimals=d),
            ModelValue("r2_k", "mk", "r2", "R²_R: kısıtlı modelin R²'si", decimals=5),
            ModelValue("r2_s", "m", "r2", "R²_UR: kısıtsız modelin R²'si", decimals=5),
            ModelValue("sd_s", "m", "df_resid", "Kısıtsız modelin artık serbestlik derecesi n − k − 1", decimals=0),
            Scalar("q3", E.const(len(chosen)), "Kısıt sayısı q", decimals=0),
            *(() if exact else (  # uyum tamken SSR_UR ≈ 0: iki formülün paydası sıfırdır
                Scalar("F_ssr", ssr, "F, SSR formülü: [(SSR_R − SSR_UR)/q] / [SSR_UR/(n − k − 1)]", decimals=2),
                Scalar("F_r2", r2, "F, R² formülü: [(R²_UR − R²_R)/q] / [(1 − R²_UR)/(n − k − 1)]", decimals=2),
            )),
            RegressionTable(((_heading(case, kept, True), "mk"), (_heading(case, regressors, False), "m")), (), "tablo81",
                            "Kısıtlı ve kısıtsız model karşılaştırması (Tablo 8.1'deki gibi)" if same
                            else "Seçtiğiniz kısıtlı model ile kısıtsız model",
                            stars=False, standard_errors=False, r2=False,
                            extra=(("ssr", "SSR", ("ssr_k", "ssr_s")), ("r2", "R²", ("r2_k", "r2_s"))), extra_decimals=d),
        )

    def note(state, choices) -> str:
        return _restricted_note(state.scalars, len(choices["adim3_x"]), d, exact)

    restricted, unrestricted = _heading(case, kept_default, True), _heading(case, regressors, False)
    checks = (
        _check("Tablo: kısıtlı model, SSR", TableTarget("tablo81", "ssr", restricted), d),
        *(() if exact else (_check("Tablo: kısıtsız model, SSR", TableTarget("tablo81", "ssr", unrestricted), d),)),
        _check("Tablo: kısıtlı model, R²", TableTarget("tablo81", "r2", restricted), 3),
        _check("Tablo: kısıtsız model, R²", TableTarget("tablo81", "r2", unrestricted), 3),
        _check("Tablo: kısıtlı model, gözlem sayısı", TableTarget("tablo81", "n", restricted), 0),
        _check("Tablo: kısıtsız model, gözlem sayısı", TableTarget("tablo81", "n", unrestricted), 0),
        _scalar("r2_s", "R²_UR (beş basamak)", 5),
        _scalar("r2_k", "R²_R (beş basamak)", 5),
        _scalar("sd_s", "n − k − 1", 0),
        _scalar("q3", "Kısıt sayısı q", 0),
        _scalar("F_ssr", "F, SSR formülü", 2),
        _scalar("F_r2", "F, R² formülü", 2),
    )
    return interactive_step(
        number=3,
        title=title,
        note=note_ref,
        explanation=(
            f"{lead} F, SSR farkını kısıt sayısına ve kısıtsız modelin artık değişkenliğine göre ölçekler: "
            "$F = \\frac{(SSR_R - SSR_{UR})/q}{SSR_{UR}/(n-k-1)}$. Aynı bağımlı değişken ve aynı örneklemle R² biçimi de "
            "aynı değeri verir. Kısıtlı modelden çıkarılan değişkenleri değiştirin."
        ),
        controls=(dropped,),
        build=build,
        checks=_robust_checks(checks, exact),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 4: genel anlamlılık testi ------------------------------------------------------------------------

def _step4(case: Case) -> LabStep:
    y = case.roles[SONUC]
    regressors = _regressors(case)
    choices_all = tuple(case.extra.get("model_options") or regressors)
    exact = _exact(case)

    def tight(chosen: tuple[str, ...]) -> bool:
        return case.own and exact_multi(case, y, chosen)

    def build(choices) -> tuple:
        chosen = tuple(choices["adim4_x"])
        formula = E.div(E.div(E.ref("r2_genel"), E.ref("k_genel")),
                        E.div(E.sub(1, E.ref("r2_genel")), E.ref("sd_genel")))
        formula_value: tuple = () if tight(chosen) else (  # uyum tamken 1 − R² ≈ 0
            Scalar("F_genel_r2", formula, "F = (R²/k) / [(1 − R²)/(n − k − 1)]", decimals=2),)
        return (
            OLS("mg", case.frame, y, chosen, f"Genel test modeli: {y} ~ {' + '.join(chosen)}"),
            ModelValue("F_genel", "mg", "f", "Genel F istatistiği (F-statistic)", decimals=2),
            ModelValue("p_genel", "mg", "f_p", "Genel F testinin p-değeri (Prob (F-statistic))", decimals=3),
            ModelValue("r2_genel", "mg", "r2", "R²", decimals=3),
            ModelValue("n_genel", "mg", "nobs", "Gözlem sayısı", decimals=0),
            ModelValue("sd_genel", "mg", "df_resid", "Payda serbestlik derecesi n − k − 1", decimals=0),
            Scalar("k_genel", E.const(len(chosen)), "Eğim sayısı k (genel testte kısıt sayısı q = k)", decimals=0),
            *formula_value,
        )

    def note(state, choices) -> str:
        chosen = tuple(choices["adim4_x"])
        s = state.scalars
        k, df = len(chosen), int(round(s["sd_genel"]))
        names = listing([phrase(case, term) for term in chosen])
        if tight(chosen):
            return (f"Model {names} ile veriye neredeyse tam uyar (R² = {sayi(s['r2_genel'], 3)}): genel F yuvarlama "
                    "gürültüsüyle çok büyüktür ve karar yazılmaz. Kısıtlı model yalnız sabit terimi içerir, bu yüzden "
                    "R²_R = 0 (§8.7)." + EXACT_MULTI_NOTE)
        rejected = s["p_genel"] < 0.05
        text = (f"Genel test F({k}, {sayim(df)}): {_f_text(s['F_genel'])}, {p_text(s['p_genel'])}: {names} "
                f"{'katsayılarının tamamının birlikte' if k > 1 else 'katsayısının'} sıfır olduğu hipotezi yüzde 5 "
                f"düzeyinde {'reddedilir' if rejected else 'reddedilemez'}. Kısıtlı model yalnız sabit terimi içerir, bu "
                f"yüzden R²_R = 0 ve F = (R²/k)/[(1 − R²)/(n − k − 1)] = {sayi(s['F_genel_r2'], 2)}. ")
        if k == 1:
            text += "Tek eğimli modelde genel F, eğim katsayısının t istatistiğinin karesidir (§8.8). "
        if rejected:
            return text + ("Genel testin reddedilmesi bütün eğimlerin ayrı ayrı anlamlı olduğunu ya da modelin doğru, "
                           "nedensel veya iyi belirlenmiş olduğunu göstermez (§8.7).")
        return text + ("Genel testin reddedilememesi eğimlerin sıfır olduğunu kanıtlamaz; veri, seçilen düzeyde yeterli "
                       "kanıt sunmamıştır (§8.7).")

    return interactive_step(
        number=4,
        title="Genel anlamlılık testi",
        note=NoteRef("8.7", 0),
        explanation=(
            "Genel test bütün eğim katsayılarının birlikte sıfır olduğu hipotezini sınar: $H_0: \\beta_1 = \\beta_2 = "
            "\\cdots = \\beta_k = 0$. Kısıtlı model yalnız sabit terimi içerdiği için $R^2_R = 0$ ve "
            "$F = \\frac{R^2/k}{(1-R^2)/(n-k-1)}$. Modeldeki açıklayıcı değişkenleri değiştirin: aynı R² farklı k ile "
            "farklı kanıt üretir."
        ),
        controls=(MultiChoice("adim4_x", "Modeldeki açıklayıcı değişkenler", options(case, choices_all), regressors,
                              help="Varsayılan: Adım 1'deki kısıtsız modelin değişkenleri."),),
        build=build,
        checks=_robust_checks((
            _scalar("F_genel", "Genel F", 2),
            _scalar("F_genel_r2", "R² biçimiyle genel F", 2),
            _scalar("sd_genel", "Payda serbestlik derecesi", 0),
            _scalar("k_genel", "Pay serbestlik derecesi", 0),
            _scalar("n_genel", "Gözlem sayısı", 0),
            _scalar("r2_genel", "R²", 3),
        ), exact),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 5: tek kısıtta F = t² -------------------------------------------------------------------------

def _step5(case: Case) -> LabStep:
    regressors = _regressors(case)
    x = case.roles[ACIKLAYICI]
    exact = _exact(case)
    df = _df(case)
    model = _fit(case, case.roles[SONUC], regressors)
    t_value = float(model.tvalues[x])

    def build(choices) -> tuple:
        term = choices["adim5_terim"]
        return (
            ModelValue("t_tek", "m", "t", f"{capital(_name(case, term))} katsayısının t istatistiği", term=term,
                       decimals=4),
            Scalar("t_kare", E.power(E.ref("t_tek"), 2), "t²", decimals=2),
            JointTest("F_tek", "p_tek", "m", (term,), f"Tek kısıtlı F testi, {_hypothesis(case, (term,))}"),
            ModelValue("p_t_tek", "m", "p", "t testinin iki taraflı p-değeri", term=term, decimals=4),
        )

    def note(state, choices) -> str:
        term = choices["adim5_terim"]
        s = state.scalars
        if exact:
            return ("Uyum tam olduğu için standart hata sıfıra çok yakındır: t ve F yuvarlama gürültüsüyle çok büyüktür. "
                    "Yine de tanım gereği F = t²'dir: tek kısıtta iki taraflı t testi ile F testi aynı hipotezi sınar "
                    "(§8.8)." + EXACT_MULTI_NOTE)
        return (f"t = {sayi(s['t_tek'], 4)}, t² = {sayi(s['t_kare'], 2)}; aynı hipotezin F testi F(1, {sayim(df)}) = "
                f"{sayi(s['F_tek'], 2)}. p-değerleri de aynıdır: t testinde {p_text(s['p_t_tek'], 4)}, F testinde "
                f"{p_text(s['p_tek'], 4)}. Tek kısıtta iki taraflı t testi ile F testi aynı kararı verir. F işaret taşımaz; "
                f"{phrase(case, term)} katsayısının yönünü t (ve katsayının kendisi) gösterir. Birden fazla kısıtta tek "
                "bir t istatistiği yoktur; ortak belirsizlik F testiyle değerlendirilir (§8.8).")

    t_text = "" if exact else f" Varsayılan modelde {phrase(case, x)} katsayısı için $t = {sayi(t_value, 4)}$."
    return interactive_step(
        number=5,
        title="Tek kısıtta F = t²",
        note=NoteRef("8.8", 0),
        explanation=(
            "Ortak hipotez tek kısıt içeriyorsa geleneksel EKK çıkarımında F testi ile iki taraflı t testi aynı kararı "
            f"verir: $F(1, n-k-1) = t^2$.{t_text} Sınanan katsayıyı değiştirin: F her seçimde t'nin karesi mi?"
        ),
        controls=(Choice("adim5_terim", "Sınanan katsayı", options(case, regressors), x,
                         help=f"Varsayılan: {case.name(x)} katsayısı."),),
        build=build,
        checks=_robust_checks((_scalar("t_tek", "t", 4), _scalar("t_kare", "t²", 2), _scalar("F_tek", "F(1, n − k − 1)", 2)),
                              exact),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 6: ikinci veride ortak test ----------------------------------------------------------------------

def _step6(house: Case) -> LabStep:
    title = str(house.extra.get("step6_title", "Birlikte katkı ayrı testlerden farklı olabilir"))
    note_ref = NoteRef("8.9", 0, ("Tablo 8.2",))
    lead = ("İkinci bir modelde bazı değişkenler tutulurken ötekilerin birlikte gereksiz olduğu hipotezi sınanır; "
            "kısıtlı model yalnız tutulan değişkenleri içerir.")
    terms = _regressors(house)
    if len(terms) < 2:
        return _need(6, title, note_ref, lead)
    y = house.roles[SONUC]
    tested = _tested(house)
    exact = _exact(house)
    full = _fit(house, y, terms)
    kept_default = tuple(term for term in terms if term not in tested)
    d = _ssr_decimals(house, y, terms, kept_default, exact)

    def loosest(dropped: tuple[str, ...]) -> str:
        """Çıkarılan değişkenlerden ayrı p-değeri en büyük olan (notlarda yatak odası sayısı)."""

        return max(dropped, key=lambda term: float(full.pvalues[term]))

    def build(choices) -> tuple:
        dropped = tuple(choices["adim6_x"])
        kept = tuple(term for term in terms if term not in dropped)
        loose = loosest(dropped)
        same = dropped == tested
        return (
            *house.load,
            OLS("h", house.frame, y, terms, f"Kısıtsız model: {y} ~ {' + '.join(terms)}"),
            OLS("hk", house.frame, y, kept, f"Kısıtlı model: {y} ~ {' + '.join(kept)}"),
            ModelValue("ssr_hk", "hk", "ssr", "SSR_R: kısıtlı model", decimals=d),
            ModelValue("ssr_h", "h", "ssr", "SSR_UR: kısıtsız model", decimals=d),
            ModelValue("r2_hk", "hk", "r2", "R²_R", decimals=3),
            ModelValue("r2_h", "h", "r2", "R²_UR", decimals=3),
            ModelValue("sd_h", "h", "df_resid", "Payda serbestlik derecesi n − k − 1", decimals=0),
            JointTest("F_h", "p_h", "h", dropped, f"Ortak F testi, {_hypothesis(house, dropped)}", decimals=2,
                      p_decimals=4),
            ModelValue("p_ayri", "h", "p", f"{capital(_name(house, loose))} katsayısının ayrı p-değeri", term=loose,
                       decimals=3),
            RegressionTable(((_heading(house, kept, True), "hk"), (_heading(house, terms, False), "h")), (), "tablo82",
                            "Ortak test karşılaştırması (Tablo 8.2'deki gibi)" if same
                            else "Seçtiğiniz kısıtlı model ile kısıtsız model",
                            stars=False, standard_errors=False, r2=False,
                            extra=(("ssr", "SSR", ("ssr_hk", "ssr_h")), ("r2", "R²", ("r2_hk", "r2_h"))),
                            extra_decimals=d),
        )

    def note(state, choices) -> str:
        dropped = tuple(choices["adim6_x"])
        kept = tuple(term for term in terms if term not in dropped)
        s = state.scalars
        q, df = len(dropped), int(round(s["sd_h"]))
        subject = listing([phrase(house, term) for term in dropped])
        held = capital(listing([phrase(house, term) for term in kept]))
        if exact:
            return (f"{held} sabitken {subject} {'birlikte ' if q > 1 else ''}sınanır; uyum tam olduğu için F ve "
                    "p-değerleri yuvarlama gürültüsüdür, karar yazılmaz (§8.9)." + EXACT_MULTI_NOTE)
        rejected = s["p_h"] < 0.05
        text = (f"F({q}, {sayim(df)}) = {sayi(s['F_h'], 2)}, {p_text(s['p_h'], 4)}: yüzde 5 düzeyinde H₀ "
                f"{'reddedilir' if rejected else 'reddedilemez'}. {held} sabitken {subject} {'birlikte ' if q > 1 else ''}"
                f"modele {'katkı sağlar' if rejected else 'istatistiksel olarak ayırt edilebilir katkı sağlamaz'}. ")
        if q == 1:
            return text + "Tek kısıtta bu test katsayının ayrı iki taraflı t testiyle aynı kararı verir: F = t² (§8.9)."
        loose = loosest(dropped)
        separate = s["p_ayri"]
        if (separate < 0.05) != rejected:
            text += (f"{capital(phrase(house, loose))} katsayısının ayrı {p_text(separate)}: ayrı test ile ortak test "
                     "farklı sonuç verir, çünkü farklı soruları yanıtlar. ")
        else:
            text += (f"Ayrı p-değeri en büyük olan {phrase(house, loose)} katsayısı için {p_text(separate)}; ortak test "
                     "ile ayrı test farklı soruları yanıtlar. ")
        if rejected:
            return text + ("Ortak ret, sınanan katsayıların hepsinin ayrı ayrı anlamlı olduğunu göstermez; nedensel "
                           "etkiyi de kanıtlamaz (§8.9).")
        return text + "Ortak testin reddedilememesi katsayıların birlikte sıfır olduğunu kanıtlamaz (§8.9)."

    restricted, unrestricted = _heading(house, kept_default, True), _heading(house, terms, False)
    checks = (
        _check("Tablo: kısıtlı model, SSR", TableTarget("tablo82", "ssr", restricted), d),
        *(() if exact else (_check("Tablo: kısıtsız model, SSR", TableTarget("tablo82", "ssr", unrestricted), d),)),
        _check("Tablo: kısıtlı model, R²", TableTarget("tablo82", "r2", restricted), 3),
        _check("Tablo: kısıtsız model, R²", TableTarget("tablo82", "r2", unrestricted), 3),
        _check("Tablo: kısıtlı model, gözlem sayısı", TableTarget("tablo82", "n", restricted), 0),
        _check("Tablo: kısıtsız model, gözlem sayısı", TableTarget("tablo82", "n", unrestricted), 0),
        _scalar("sd_h", "Payda serbestlik derecesi", 0),
        _scalar("F_h", "Ortak F", 2),
        _scalar("p_h", "Ortak p-değeri", 4),
        _scalar("p_ayri", "Ayrı p-değeri", 3),
    )
    context = str(house.extra.get("step6_text", "Model aynı dosyanın sonucu ile temel açıklayıcı ve ek "
                                                "değişkenleridir (ayrı bir çerçevede)."))
    return interactive_step(
        number=6,
        title=title,
        note=note_ref,
        explanation=(
            f"{lead} {context} Varsayılan hipotezde {listing([phrase(house, term) for term in kept_default])} "
            f"tutulurken {_subject(house, tested)}{_together(tested)} sıfırdır: {_hypothesis(house, tested, False)}. Bir "
            "değişkenin ayrı testte anlamlı "
            "olmaması ortak testin sonucunu belirlemez. Birlikte sınanan değişkenleri değiştirin."
        ),
        controls=(MultiChoice("adim6_x", "Birlikte sınanan (kısıtlı modelden çıkarılan) değişkenler",
                              options(house, terms), tested,
                              help=f"Varsayılan: {listing([house.name(term) for term in tested])}; "
                                   f"{listing([house.name(term) for term in kept_default])} modelde kalır.",
                              maximum=len(terms) - 1),),
        build=build,
        checks=_robust_checks(checks, exact),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 7: makale tablosunda ortak testler --------------------------------------------------------------

def _step7(case: Case) -> LabStep:
    y = case.roles[SONUC]
    regressors, tested = _regressors(case), _tested(case)
    logs = positive(case, y)
    exact = _exact(case)
    level = case.name(y) if case.own else capital(phrase(case, y))
    columns = [(f"(1) {level}", "m")]
    operations: list = []
    if logs:
        ly = log_column(case, y)
        operations += [*log_operations(case, case.frame, (y,)),
                       OLS("ml", case.frame, ly, regressors, f"Log sonuç modeli: {ly} ~ {' + '.join(regressors)}")]
        columns.append((f"(2) {log_label(case, y)}", "ml"))
    models = [name for _, name in columns]
    for index, model in enumerate(models, start=1):
        operations += [
            JointTest(f"ortak_F{index}", f"ortak_p{index}", model, tested,
                      f"Sütun ({index}): {_subject(case, tested, True)} için ortak F testi"),
            ModelValue(f"genel_F{index}", model, "f", f"Sütun ({index}): genel F", decimals=2),
        ]
    label = f"{capital(_subject(case, tested, True))}{_together(tested)} sıfır: ortak F"
    operations.append(RegressionTable(
        tuple(columns), regressors, "tablo86", "Bireysel ve ortak çıkarım (Tablo 8.6'daki gibi)",
        extra=(("ortak_F", label, tuple(f"ortak_F{index}" for index in range(1, len(models) + 1))),
               ("ortak_p", "Ortak test p-değeri", tuple(f"ortak_p{index}" for index in range(1, len(models) + 1))),
               ("genel_F", "Genel F", tuple(f"genel_F{index}" for index in range(1, len(models) + 1))))))
    log_data = case.data.assign(**{log_column(case, y): np.log(case.data[y].astype(float))}) if logs else None
    log_exact = logs and case.own and exact_multi(replace(case, data=log_data), log_column(case, y), regressors)

    def note(state) -> str:
        result = state.models["m"]
        s = state.scalars
        heading = columns[0][0]
        if exact or log_exact:
            return ("Uyum tam olduğu için standart hatalar, yıldızlar ve F değerleri yuvarlama gürültüsüdür; tablodan "
                    "katsayılar okunur (§8.12)." + EXACT_MULTI_NOTE)
        loose = [term for term in tested if result.pvalues[term] >= 0.05]
        rejected = s["ortak_p1"] < 0.05
        text = f"Sütun {heading.split(' ')[0]}'de "
        if loose:
            text += (f"{listing([phrase(case, term) for term in loose])} yüzde 5 düzeyinde tek başına anlamlı değildir; "
                     f"buna karşılık {_genitive(case, tested)} sıfır olduğu hipotezi ortak testte "
                     f"{'reddedilir' if rejected else 'reddedilemez'} ({_f_text(s['ortak_F1'])}; {p_text(s['ortak_p1'])}). "
                     "Ortak test ayrı yıldızların basit toplamı değildir. ")
        else:
            text += (f"sınanan katsayılar tek başına da yüzde 5 düzeyinde anlamlıdır; ortak test {_f_text(s['ortak_F1'])} "
                     f"({p_text(s['ortak_p1'])}). Ortak test yine ayrı yıldızların basit toplamı değildir. ")
        if logs:
            text += ("Sütun (2)'de bağımlı değişken log sonuç olduğu için katsayılar yaklaşık yüzde farklar olarak "
                     "okunur; iki sütunun katsayı büyüklükleri doğrudan karşılaştırılamaz. ")
        else:
            text += "Sonuçta sıfır ya da negatif değer olduğu için log sonuç sütunu kurulmaz. "
        return text + "Ortak testin reddedilmesi nedensellik için yeterli değildir (§8.12)."

    checks = []
    for heading, _ in columns:
        for term in regressors:
            checks += [_check(f"{heading}: {case.name(term)}", TableTarget("tablo86", term, heading), 3),
                       _check(f"{heading}: {case.name(term)} (SH)", TableTarget("tablo86", f"{term}_sh", heading), 3)]
        checks += [_check(f"{heading}: ortak F", TableTarget("tablo86", "ortak_F", heading), 2),
                   _check(f"{heading}: genel F", TableTarget("tablo86", "genel_F", heading), 2),
                   _check(f"{heading}: gözlem sayısı", TableTarget("tablo86", "n", heading), 0),
                   _check(f"{heading}: R²", TableTarget("tablo86", "r2", heading), 3)]
    return LabStep(
        number=7,
        title="Makale tablosunda ortak testler",
        note=NoteRef("8.12", 0, ("Tablo 8.6",)),
        explanation=(
            "Makale tablolarında ortak testler katsayıların altında ayrı satırlarda verilir: sınanan hipotez ve standart "
            "hata türü tablo notunda yazılır. "
            + ("Sütun (1) düzey–düzey, Sütun (2) log–düzey modelidir; " if logs else "Tabloda kısıtsız model vardır; ")
            + f"ortak test {_genitive(case, tested)} sıfır olduğunu, genel F bütün eğimleri sınar."
        ),
        operations=tuple(operations),
        checks=_robust_checks(tuple(checks), exact or log_exact, table="tablo86"),
        note_for=lambda state, choices: note(state),
    )


# --- Tanım ------------------------------------------------------------------------------------------------------

def build(case: Case) -> LabSpec:
    """Konu 8 uygulamasını verilen veriyle kurar; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    house = second(case)
    labels = labels_of(case)
    labels.update(labels_of(house))
    y = case.roles[SONUC]
    if positive(case, y):
        labels.setdefault(log_column(case, y), log_label(case, y))
    labels[INTERCEPT] = "Sabit terim"
    spec = LabSpec(
        topic_key=TOPIC,
        title=str(case.extra.get("title", TITLE)),
        note_section="8",
        steps=(_step1(case), _step2(case), _step3(case), _step4(case), _step5(case), _step6(house),
               _step7(case)),
        labels=tuple(labels.items()),
        source=case.source,
    )
    return with_app_values(spec)


# --- Alternatif örnek ----------------------------------------------------------------------------------------

def _house() -> Case:
    return house_case(
        regressors=("area", "rooms", "baths"),
        outcome="price1000",
        derived=("price1000",),
        main=("area", "rooms", "baths"),
        tested=("rooms", "baths"),
        symbols={"area": "β_büyüklük", "rooms": "β_oda", "baths": "β_banyo"},
        step6_title="KIELMC (1978): birlikte katkı ayrı testlerden farklıdır",
        step6_text=("Veri KIELMC'nin 1978 satışlarıdır (179 konut; fiyat bin dolar ~ büyüklük + oda + banyo). Oda sayısının "
                    "katsayısı ayrı testte anlamlı değildir; ortak test farklı bir soruyu yanıtlar."),
    )


def alternative_case() -> Case:
    return wage2_case(
        ("educ", "exper", "tenure", "IQ"),
        title="Uygulama: Çoklu Kısıtlar ve F Testi (WAGE2, KIELMC)",
        main=("educ", "exper", "tenure"),
        tested=("exper", "tenure"),
        model_options=("educ", "exper", "tenure", "IQ"),
        symbols={"educ": "β_eğitim", "exper": "β_deneyim", "tenure": "β_kıdem", "IQ": "β_IQ"},
        house=_house(),
    )


@cache
def alternative() -> LabSpec:
    return build(alternative_case())


STORY = ("Ücret modeli WAGE2'dir (935 erkek çalışan, 1980; aylık kazanç ~ eğitim + deneyim + kıdem; deneyim ve kıdem "
         "birlikte sınanır). İkinci veri KIELMC'nin 1978 satışlarıdır (fiyat bin dolar ~ büyüklük + oda + banyo; oda ve "
         "banyo sayısı birlikte sınanır).")


# --- Kendi verin ---------------------------------------------------------------------------------------------

CUSTOM = custom_lab(
    build,
    sample_employed,
    ("Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sonuç ve temel açıklayıcı değişken zorunludur ve sayısal olmalıdır. "
     "Kısıtsız model sonucun temel açıklayıcı ve ek değişkenlere (en çok 3) göre regresyonudur; ortak testin "
     "varsayılanı ek değişkenlerdir (temel açıklayıcı modelde kalır). Kısıtlı model adımları (3 ve 6) için en az bir ek "
     "değişken gerekir; log sonuçlu sütun için sonucun bütün değerleri pozitif olmalı. "
     f"{ROW_RULE}"),
    roles((1, 2, 3, 4, 5, 6, 7)),
    "Kısıtsız modelin değişkenleridir; ortak test varsayılan olarak bunları sınar.",
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
