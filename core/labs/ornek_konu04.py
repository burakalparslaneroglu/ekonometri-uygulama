"""Konu 4 genel uygulaması: EKK tahminini değerlendirme, uyum, ölçü birimleri ve temel fonksiyonel biçimler.

Notlardaki on adım (``core.labs.konu04``) aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde
yazılır: EKK doğrusunun örneklem özellikleri, ortalama ölçütü, bir gözlemdeki sapmanın iki bileşeni, kareler
toplamları, R², ölçü birimleri, dört temel biçim, log–düzey model, makale tablosu ve kontrol listesi.

Alternatif örnekte ücret adımları WAGE2 (935 erkek çalışan, 1980; aylık kazanç, ABD doları), notlarda HPRICE1 ile
yapılan konut adımları (6, 7 ve 9) KIELMC'nin 1978 satışlarıdır (179 konut; fiyat dolar, büyüklük fit²). Adım 3'ün
sayıları WAGE2'nin ilk çalışanıdır. "Kendi verini yükle" seçeneğinde bütün adımlar öğrencinin dosyasıyla kurulur:
konut adımları aynı dosyanın sonuç ve açıklayıcı değişkenini kullanır; logaritmik biçimler bütün değerleri pozitif
olan değişkenlerde kurulur, değilse adım neye ihtiyacı olduğunu yazar.

Etkileşim notlardaki gibidir: modelin açıklayıcı değişkeni (Adım 1; Adım 2, 4 ve 5 aynı modeli kullanır), sapma
örneğinin üç değeri (Adım 3), sonuç ve açıklayıcı değişkenin birimi (Adım 6), incelenen biçim (Adım 7), log–düzey
modelin açıklayıcı değişkeni (Adım 8) ve makale tablosunun sütunları (Adım 9). Yüzde yorumları yaklaşıktır (100·β̂₁).
"""

from __future__ import annotations

from functools import cache

import numpy as np

from core.labs import expr as E
from core.labs.ornek import EXACT_FIT_NOTE, Case, TopicVariants, exact_fit, md, sayi, sayim, stable_checks, \
    with_app_values
from core.labs.ornek_regresyon import (
    ACIKLAYICI,
    ROW_RULE,
    SONUC,
    candidates,
    capital,
    change,
    custom_lab,
    digits_for,
    display,
    house_case,
    labels_of,
    level_text,
    log_column,
    log_label,
    log_operations,
    negligible,
    noise_decimals,
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
    BarChart,
    Check,
    Choice,
    CoefTarget,
    Derive,
    LabSpec,
    LabStep,
    ModelTarget,
    ModelValue,
    MultiChoice,
    NoteRef,
    NumberChoice,
    PairStatistic,
    RegressionTable,
    Scalar,
    ScalarTable,
    ScalarTarget,
    ScatterPlot,
    ShowModel,
    Statistic,
    TableTarget,
    interactive_step,
)
from core.labs.wording import at_zero

TOPIC = "konu04"
TITLE = "Uygulama: EKK Çıktısı, Uyum ve Temel Fonksiyonel Biçimler"
X_KEY = "adim1_x"
ALT_REGRESSORS = ("educ", "exper", "tenure")
FORM_NAMES = {"duzey_duzey": "Düzey–düzey", "log_duzey": "Log–düzey", "duzey_log": "Düzey–log", "log_log": "Log–log"}
MODELS = {"duzey_duzey": "m_dd", "log_duzey": "m_ld", "duzey_log": "m_dl", "log_log": "m_ll"}
ARTICLE_DEFAULT = ("duzey_duzey", "log_log")


def _check(label: str, target, decimals: int) -> Check:
    return Check(label, target, 0.0, decimals)


def _scalar(name: str, label: str, decimals: int) -> Check:
    return _check(label, ScalarTarget(name), decimals)


def _name(case: Case, column: str) -> str:
    """Başlık ve kod yorumundaki ad (Markdown değil)."""

    return case.name(column) if case.own else phrase(case, column)


def _title(case: Case, column: str) -> str:
    """Cümle başındaki ya da sütun başlığındaki ad: alternatif örnekte büyük harfle, kendi verinde dosyadaki ad."""

    return case.name(column) if case.own else capital(phrase(case, column))


def _accusative(case: Case, x: str) -> str:
    """Açıklayıcı değişkenin belirtme durumu: "eğitimi"; kendi verinde "“X” değişkenini"."""

    if case.own:
        return f"{phrase(case, x)} değişkenini"
    return dict(case.extra.get("accusative", {})).get(x, f"{phrase(case, x)} değişkenini")


# --- Adım 1: EKK doğrusunun örneklem özellikleri ----------------------------------------------------------

def x_choice(case: Case) -> Choice:
    return Choice(X_KEY, f"Açıklayıcı değişken (bağımlı değişken: {case.name(case.roles[SONUC])})",
                  options(case, candidates(case)), case.roles[ACIKLAYICI],
                  help=f"Varsayılan: {case.name(case.roles[ACIKLAYICI])}. Adım 2, 4 ve 5 bu modeli kullanır.")


def _step1(case: Case, choice: Choice) -> LabStep:
    y = case.roles[SONUC]
    noise = noise_decimals(case, regressors=(case.roles[ACIKLAYICI],))

    def build(choices) -> tuple:
        x = choices[X_KEY]
        shown = noise_decimals(case, regressors=(x,))
        return (
            *case.load,
            OLS("model", case.frame, y, (x,), f"Basit regresyon: {y} ~ {x}"),
            ModelValue("b0", "model", "coef", "Sabit terim β̂₀", term=INTERCEPT),
            ModelValue("b1", "model", "coef", f"Eğim β̂₁ ({x})", term=x),
            Derive(case.frame, "tahmin", E.add(E.ref("b0"), E.mul(E.ref("b1"), E.var(x))), "Tahmin edilen değer Ŷᵢ"),
            Derive(case.frame, "artik", E.sub(E.var(y), E.var("tahmin")), "Artık ûᵢ = Yᵢ − Ŷᵢ"),
            Statistic(case.frame, y, "count", "n", "Gözlem sayısı", decimals=0),
            Statistic(case.frame, y, "mean", "ort_y", "Sonucun ortalaması Ȳ", decimals=6),
            Statistic(case.frame, "artik", "sum", "artik_toplami", "Artıkların toplamı Σûᵢ", decimals=shown),
            Statistic(case.frame, "tahmin", "mean", "ort_tahmin", "Tahmin edilen değerlerin ortalaması", decimals=6),
            Statistic(case.frame, x, "mean", "ort_x", "Açıklayıcı değişkenin ortalaması X̄", decimals=4),
            Scalar("dogru_xbar", E.add(E.ref("b0"), E.mul(E.ref("b1"), E.ref("ort_x"))),
                   "Doğrunun X̄'daki değeri Ŷ(X̄)", decimals=6),
        )

    def note(state, choices) -> str:
        x = choices[X_KEY]
        s = state.scalars
        unit = f" {short_unit(case, y)}" if short_unit(case, y) else ""
        shown = noise_decimals(case, regressors=(x,))
        text = (f"Artıkların toplamı {sayi(s['artik_toplami'], shown)}: bilgisayarın hesap hassasiyeti içinde sıfırdır "
                "(Denklem 4.1); küçük işaret yazılıma göre değişebilir. Tahmin edilen değerlerin ortalaması "
                f"{sayi(s['ort_tahmin'], 6)} ve doğrunun X̄'daki değeri {sayi(s['dogru_xbar'], 6)}, sonucun ortalamasına "
                f"({sayi(s['ort_y'], 6)}{unit}) eşittir (Denklem 4.2–4.3). Bu eşitlikler sabit terimli her EKK "
                "doğrusunun cebirsel özelliğidir; modelin doğru olduğunu ya da her artığın küçük olduğunu göstermez.")
        if x != case.roles[ACIKLAYICI]:
            text += f" Açıklayıcı değişkeni {phrase(case, x)} yaptığınızda da aynı eşitlikler geçerlidir."
        return text

    return interactive_step(
        number=1,
        title="EKK doğrusunun temel örneklem özellikleri",
        note=NoteRef("4.1", 0, ("Kod 4.1", "Denklem 4.1–4.3")),
        explanation=(
            "Sabit terimli EKK tahmininde üç eşitlik her örneklemde sağlanır: artıkların toplamı sıfırdır ($\\sum "
            "\\hat u_i = 0$), tahmin edilen değerlerin ortalaması $\\bar Y$'ye eşittir ve doğru $(\\bar X, \\bar Y)$ "
            "noktasından geçer. Önce model tahmin edilir; tahmin edilen değer ve artık her gözlem için hesaplanır."
        ),
        controls=(choice,),
        build=build,
        checks=(
            _scalar("n", "Gözlem sayısı", 0),
            _scalar("ort_y", "Sonucun ortalaması", 6),
            _scalar("artik_toplami", "Artıkların toplamı", noise),
            _scalar("ort_tahmin", "Tahmin edilen değerlerin ortalaması", 6),
            _scalar("dogru_xbar", "Doğru (X̄, Ȳ) noktasından geçer", 6),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 2: ortalama ölçütü -------------------------------------------------------------------------------

def _step2(case: Case, choice: Choice) -> LabStep:
    y = case.roles[SONUC]
    many = len(case.data) > 100

    def build(choices) -> tuple:
        x = choices[X_KEY]
        return (
            Statistic(case.frame, y, "mean", "ybar", "Yalnız ortalamayı kullanan tahmin Ȳ", decimals=3),
            ScatterPlot(case.frame, x, y, display(case, x), display(case, y),
                        f"{case.extra.get('data_name', 'Verileriniz')}: yalnız ortalama ile {_name(case, x)} bilgisini "
                        "kullanan EKK doğrusu", fit_line=True, size=6 if many else 9, opacity=0.3 if many else 0.8,
                        lines=(("ybar", 0, "Yalnız ortalama: Ȳ"),)),
        )

    def note(state, choices) -> str:
        x = choices[X_KEY]
        unit = f" {short_unit(case, y)}" if short_unit(case, y) else ""
        return (f"Kesikli yatay çizgi her gözlem için aynı tahmini ({sayi(state.scalars['ybar'], 3)}{unit}) verir: "
                "açıklayıcı değişken kullanılmadığında kareli tahmin hatalarını en küçük yapan sabit değer örneklem "
                f"ortalamasıdır. Eğimli doğru {phrase(case, x)} bilgisini kullanır ve farklı düzeylere farklı tahminler "
                "üretir. Uyum sorusu şudur: doğru, yalnız ortalamayı kullanan tahmine göre kareli sapmaları ne kadar "
                "azaltıyor? Cevap Adım 4–5'teki kareler toplamları ve R²'dir.")

    return interactive_step(
        number=2,
        title="Ortalama ölçütü ile regresyon doğrusunu karşılaştırmak",
        note=NoteRef("4.2", 0, ("Şekil 4.1",)),
        explanation=(
            "Açıklayıcı değişken kullanmadan her gözlem için aynı değeri tahmin edersek, kareli tahmin hatalarını en "
            "küçük yapan sabit değer örneklem ortalamasıdır: $\\hat Y_i^{(0)} = \\bar Y$. EKK doğrusu bu doğal "
            "başlangıç ölçütüyle karşılaştırılır. Bu adım Adım 1'deki modeli kullanır."
        ),
        uses=(choice,),
        build=build,
        checks=(_scalar("ybar", "Sonucun ortalaması", 3),),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 3: bir gözlemdeki sapmanın iki bileşeni ----------------------------------------------------------

def _first_worker(case: Case) -> tuple[float, float, float]:
    """İlk gözlem için örneklem ortalaması, varsayılan modelin tahmini ve gözlenen değer."""

    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    data = case.data[[y, x]].astype(float)
    slope, intercept = np.polyfit(data[x], data[y], 1)
    return float(data[y].mean()), float(intercept + slope * data[x].iloc[0]), float(data[y].iloc[0])


def _decomposition_controls(case: Case) -> tuple[NumberChoice, NumberChoice, NumberChoice]:
    if "adim3" in case.extra:
        return tuple(case.extra["adim3"])
    mean, fitted, observed = _first_worker(case)
    values = case.data[case.roles[SONUC]]
    low, high = min(float(values.min()), fitted), max(float(values.max()), fitted)
    template = number_control("adim3_y", "Gözlenen değer Yᵢ", observed, low, high)
    common = dict(low=template.minimum, high=template.maximum, step=template.step)
    return (
        number_control("adim3_ybar", "Örneklem ortalaması Ȳ", mean, help="Varsayılan: verinizin ortalaması.",
                       **common),
        number_control("adim3_yhat", "Modelin tahmini Ŷᵢ", fitted,
                       help="Varsayılan: ilk gözlem için varsayılan modelin tahmini.", **common),
        number_control("adim3_y", "Gözlenen değer Yᵢ", observed, help="Varsayılan: ilk gözlemin değeri.", **common),
    )


def _step3(case: Case) -> LabStep:
    controls = _decomposition_controls(case)
    digits = controls[0].decimals

    def build(choices) -> tuple:
        ybar, yhat, y = (choices[key] for key in ("adim3_ybar", "adim3_yhat", "adim3_y"))
        text = {key: sayi(float(choices[key]), digits) for key in ("adim3_ybar", "adim3_yhat", "adim3_y")}
        return (
            Scalar("toplam_sapma", E.sub(y, ybar), f"Toplam sapma Yᵢ − Ȳ = {text['adim3_y']} − {text['adim3_ybar']}",
                   decimals=digits),
            Scalar("model_sapma", E.sub(yhat, ybar),
                   f"Model kaynaklı sapma Ŷᵢ − Ȳ = {text['adim3_yhat']} − {text['adim3_ybar']}", decimals=digits),
            Scalar("artik_sapma", E.sub(y, yhat), f"Artık ûᵢ = Yᵢ − Ŷᵢ = {text['adim3_y']} − {text['adim3_yhat']}",
                   decimals=digits),
            Scalar("iki_bilesen", E.add(E.ref("model_sapma"), E.ref("artik_sapma")), "Model kaynaklı sapma + artık",
                   decimals=digits),
        )

    def note(state, choices) -> str:
        s = state.scalars
        total, model, residual = s["toplam_sapma"], s["model_sapma"], s["artik_sapma"]
        text = (f"Toplam sapma {sayi(total, digits)} = model kaynaklı sapma {sayi(model, digits)} + artık "
                f"{sayi(residual, digits)} (Denklem 4.4). Eşitlik her gözlemde tanım gereği sağlanır.")
        if total * model < 0:
            text += (" Bu gözlemde model tahmini ortalamanın öbür tarafındadır: model kaynaklı sapma ile toplam sapmanın "
                     "işaretleri farklıdır; artık bu farkı kapatır.")
        elif abs(residual) > abs(total) and total != 0:
            text += " Artık toplam sapmadan büyüktür: bu gözlemde doğru ortalamadan daha kötü bir tahmin vermiştir."
        return text + (" “Model kaynaklı” sapma nedensel bir açıklama değildir; yalnız tahmin edilen değerin ortalamadan "
                       "ne kadar ayrıldığını anlatır.")

    return interactive_step(
        number=3,
        title="Bir gözlemdeki sapmanın iki bileşeni",
        note=NoteRef("4.3", 0, ("Denklem 4.4",)),
        explanation=(
            "Her gözlemde $Y_i - \\bar Y = (\\hat Y_i - \\bar Y) + \\hat u_i$ (Denklem 4.4): toplam sapma, model "
            "kaynaklı sapma ile artığın toplamıdır. "
            + str(case.extra.get("step3_text", "Varsayılan değerler verinizin ilk gözlemidir: örneklem ortalaması, "
                                               "varsayılan modelin bu gözlem için tahmini ve gözlenen değer "
                                               "(kaydırıcının adımına yuvarlanmış)."))
            + " Değerleri değiştirerek iki bileşenin toplam sapmayı nasıl paylaştığını görün."
        ),
        controls=controls,
        build=build,
        checks=(
            _scalar("toplam_sapma", "Toplam sapma", digits),
            _scalar("model_sapma", "Model kaynaklı sapma", digits),
            _scalar("artik_sapma", "Artık", digits),
            _scalar("iki_bilesen", "Model kaynaklı sapma + artık", digits),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 4: kareler toplamları ------------------------------------------------------------------------------

def _step4(case: Case, choice: Choice) -> LabStep:
    y = case.roles[SONUC]

    def build(choices) -> tuple:
        x = choices[X_KEY]
        return (
            Derive(case.frame, "toplam_kare", E.power(E.sub(E.var(y), E.ref("ort_y")), 2), "(Yᵢ − Ȳ)²"),
            Derive(case.frame, "model_kare", E.power(E.sub(E.var("tahmin"), E.ref("ort_y")), 2), "(Ŷᵢ − Ȳ)²"),
            Derive(case.frame, "artik_kare", E.power(E.var("artik"), 2), "ûᵢ²"),
            Statistic(case.frame, "toplam_kare", "sum", "tkt", "TKT = Σ(Yᵢ − Ȳ)²", decimals=3),
            Statistic(case.frame, "model_kare", "sum", "mkt", "MKT = Σ(Ŷᵢ − Ȳ)²", decimals=3),
            Statistic(case.frame, "artik_kare", "sum", "hkt", "HKT = Σûᵢ²", decimals=3),
            Scalar("mkt_hkt", E.add(E.ref("mkt"), E.ref("hkt")), "MKT + HKT", decimals=3),
            ScalarTable((("TKT", E.ref("tkt")), ("MKT", E.ref("mkt")), ("HKT", E.ref("hkt"))), "kareler", decimals=1),
            BarChart("kareler", "deger", "Kareler toplamı türü",
                     str(case.extra.get("square_unit", "Kareler toplamı (sonucun biriminin karesi)")),
                     f"{case.extra.get('data_name', 'Verileriniz')}: {_name(case, y)}–{_name(case, x)} modelinde toplam, "
                     "model ve artık kareler toplamları", decimals=1),
        )

    def note(state, choices) -> str:
        x = choices[X_KEY]
        s = state.scalars
        return (f"TKT = {sayi(s['tkt'], 3)} = MKT {sayi(s['mkt'], 3)} + HKT {sayi(s['hkt'], 3)} (Denklem 4.5). Toplam "
                f"değişkenliğin yaklaşık {sayi(s['mkt'], 1)} birimlik kısmı {phrase(case, x)} doğrusu boyunca hareket "
                f"ederken {sayi(s['hkt'], 1)} birimlik kısmı artıklarda kalır. TKT açıklayıcı değişkene bağlı değildir: "
                "yalnız sonucun ortalama çevresindeki değişkenliğidir. EKK'nin en küçük yaptığı ölçüt HKT'dir. "
                "Kısaltmalara dikkat: bazı yazılımlarda `SSR` artık kareler toplamı, bazılarında regresyon kareler "
                "toplamıdır.")

    return interactive_step(
        number=4,
        title="Kareler toplamlarının ayrıştırılması",
        note=NoteRef("4.4", 0, ("Kod 4.2", "Denklem 4.5", "Şekil 4.2")),
        explanation=(
            "Sapmaların işaretleri birbirini götürebildiği için kareleri toplanır: TKT $= \\sum (Y_i - \\bar Y)^2$, "
            "MKT $= \\sum (\\hat Y_i - \\bar Y)^2$, HKT $= \\sum \\hat u_i^2$. Sabit terimli EKK modelinde TKT = MKT + "
            "HKT (Denklem 4.5). Bu adım Adım 1'deki modeli kullanır."
        ),
        uses=(choice,),
        build=build,
        checks=(
            _scalar("tkt", "TKT", 3), _scalar("mkt", "MKT", 3), _scalar("hkt", "HKT", 3),
            _scalar("mkt_hkt", "MKT + HKT", 3),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 5: R² ve korelasyon --------------------------------------------------------------------------------

def _step5(case: Case, choice: Choice) -> LabStep:
    y = case.roles[SONUC]

    def build(choices) -> tuple:
        x = choices[X_KEY]
        return (
            Scalar("r2_mkt", E.div(E.ref("mkt"), E.ref("tkt")), "R² = MKT / TKT", decimals=4),
            Scalar("r2_hkt", E.sub(1, E.div(E.ref("hkt"), E.ref("tkt"))), "R² = 1 − HKT / TKT", decimals=4),
            ModelValue("r2_yazilim", "model", "r2", "Yazılımın R²'si", decimals=4),
            Scalar("r2_yuzde", E.mul(E.ref("r2_mkt"), 100), "R², yüzde olarak", decimals=1, percent=True),
            PairStatistic(case.frame, y, x, "corr", "r", f"{_title(case, y)} ile {_name(case, x)} arasındaki "
                          "korelasyon r", decimals=4),
            Scalar("r_kare", E.power(E.ref("r"), 2), "Korelasyonun karesi r²", decimals=4),
        )

    def note(state, choices) -> str:
        x = choices[X_KEY]
        s = state.scalars
        where = case.extra.get("data_name", "Verinizin")
        return (f"R² = {sayi(s['r2_mkt'], 4)}: {where} örnekleminde {phrase(case, y)} değişkenliğinin yaklaşık "
                f"%{sayi(s['r2_yuzde'], 1)} kadarı {_accusative(case, x)} kullanan doğrusal model tarafından örneklem "
                "içinde izlenir. Yorum örnekleme ve seçilen doğrusal modele aittir; nedensel etki oranı değildir. Sabit "
                f"terimli basit regresyonda R² korelasyonun karesidir: r = {sayi(s['r'], 4)}, r² = {sayi(s['r_kare'], 4)}. "
                "Yüksek R² nedenselliği ya da doğru modeli garanti etmez; düşük R² katsayıyı değersiz yapmaz. R²'ler "
                "yalnız aynı bağımlı değişkenli modeller arasında karşılaştırılır (§4.6).")

    return interactive_step(
        number=5,
        title="Belirleme katsayısı R²",
        note=NoteRef("4.5", 0, ("Denklem 4.6",)),
        explanation=(
            "$R^2 = \\text{MKT}/\\text{TKT} = 1 - \\text{HKT}/\\text{TKT}$ (Denklem 4.6): bağımlı değişkenin örneklem "
            "değişkenliğinin doğrusal model tarafından izlenen payı. Sabit terimli basit regresyonda $R^2 = r_{XY}^2$ "
            "(§4.5). Yorumun dört sınırı: örnekleme aittir, bağımlı değişkenin örneklem değişkenliğiyle ilgilidir, "
            "seçilen doğrusal model içindir, nedensel etki oranı değildir (§4.5–4.6). Bu adım Adım 1'deki modeli "
            "kullanır."
        ),
        uses=(choice,),
        build=build,
        checks=(
            _scalar("r2_mkt", "R² = MKT / TKT", 4), _scalar("r2_hkt", "R² = 1 − HKT / TKT", 4),
            _scalar("r2_yazilim", "R² (yazılım)", 4), _scalar("r2_yuzde", "R², yüzde", 1),
            _scalar("r", "Korelasyon r", 4), _scalar("r_kare", "r²", 4),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 6: ölçü birimleri ---------------------------------------------------------------------------------

def _units_setup(house: Case) -> dict:
    """Birim seçenekleri: (anahtar, etiket, sütun, çarpan, metindeki ad); sonuç ``çarpan × Y``, açıklayıcı
    ``X / bölen``. Özet skalerler: (ad, etiket, katsayı, basamak) ile β̂ · katsayı / (çarpan · bölen)."""

    if "units" in house.extra:
        return dict(house.extra["units"])
    y, x = house.roles[SONUC], house.roles[ACIKLAYICI]
    return {
        "y": (("dosya", "Dosyadaki birim", y, 1.0, "sonucun dosyadaki birimi"),
              ("bin_kat", "1000 ile çarpılmış (ör. bin TL → TL)", "y_olcekli", 1000.0, "sonucun 1000 katı")),
        "x": (("dosya", "Dosyadaki birim", x, 1.0, "açıklayıcının dosyadaki birimi"),
              ("yuzde_bir", "100'e bölünmüş", "x_olcekli", 100.0, "açıklayıcının 100'de biri")),
        "derived": ((("y_olcekli", E.mul(1000, E.var(y)), f"{house.name(y)} × 1000"),
                     ("x_olcekli", E.div(E.var(x), 100), f"{house.name(x)} / 100"))),
        "summaries": (("bir_birim", "Açıklayıcıda bir birimlik farkın tahmini sonuç farkı (dosyadaki birimlerle)", 1.0,
                       4),
                      ("yuz_birim", "Açıklayıcıda 100 birimlik farkın tahmini sonuç farkı (dosyadaki birimlerle)",
                       100.0, 2)),
        "labels": {"y_olcekli": f"{house.name(y)} × 1000", "x_olcekli": f"{house.name(x)} / 100"},
        "change_words": ("sonucu 1000 ile çarptınız", "açıklayıcıyı 100'e böldünüz"),
    }


def _step6(house: Case) -> LabStep:
    setup = _units_setup(house)
    y_options, x_options = setup["y"], setup["x"]

    def build(choices) -> tuple:
        y_key, x_key = choices["adim6_y"], choices["adim6_x"]
        _, _, y, a, y_name = next(item for item in y_options if item[0] == y_key)
        _, _, x, c, x_name = next(item for item in x_options if item[0] == x_key)
        default = (y_key, x_key) == (y_options[0][0], x_options[0][0])
        operations = (
            *house.load,
            *(Derive(house.frame, name, expression, comment) for name, expression, comment in setup["derived"]),
            OLS("model_birim", house.frame, y, (x,), f"Düzey–düzey model: {y} ~ {x}"),
            ShowModel("model_birim", "Verideki birimlerle model" if default else "Seçtiğiniz birimlerle model",
                      columns=("coef",), stats=("nobs", "r2"), stars=False),
            ModelValue("sabit_birim", "model_birim", "coef", f"Sabit terim ({y_name})", term=INTERCEPT, decimals=3),
            ModelValue("egim_birim", "model_birim", "coef", f"Eğim ({y_name} / {x_name})", term=x),
            ModelValue("r2_birim", "model_birim", "r2", "R²", decimals=4),
            *(Scalar(name, E.mul(E.ref("egim_birim"), factor / (a * c)), label, decimals=decimals)
              for name, label, factor, decimals in setup["summaries"]),
        )
        if default:
            return operations
        y0, x0 = y_options[0][2], x_options[0][2]
        return operations + (
            OLS("model_veri_birim", house.frame, y0, (x0,), f"Karşılaştırma için verideki birimler: {y0} ~ {x0}"),
            RegressionTable((("(1) Verideki birimler", "model_veri_birim"), ("(2) Seçiminiz", "model_birim")),
                            tuple(dict.fromkeys((x0, x, INTERCEPT))), "birim_karsilastirma",
                            "Verideki birimler ile seçtiğiniz birimler", stars=False, decimals=4,
                            standard_errors=False),
        )

    def note(state, choices) -> str:
        y_key, x_key = choices["adim6_y"], choices["adim6_x"]
        s = state.scalars
        y_name = next(item for item in y_options if item[0] == y_key)[4]
        x_name = next(item for item in x_options if item[0] == x_key)[4]
        (first, _, _, d1), (second, _, _, d2) = setup["summaries"]
        words = setup.get("summary_words", ("açıklayıcıda bir birimlik farkın tahmini sonuç farkı",
                                            "100 birimlik farkınki"))
        units = setup.get("summary_units", ("", ""))
        text = (f"Eğim {sayi(s['egim_birim'], 4)} ({y_name} / {x_name}); sabit {sayi(s['sabit_birim'], 3)} ({y_name}). "
                f"Ekonomik ilişki aynıdır: {words[0]} {sayi(s[first], d1)}{units[0]}, {words[1]} "
                f"{sayi(s[second], d2)}{units[1]} (seçilen birimlerden bağımsız); R² = {sayi(s['r2_birim'], 4)}.")
        if (y_key, x_key) == (y_options[0][0], x_options[0][0]):
            return text + (" Birim kuralı (Tablo 4.1): sonuç bir sayıyla çarpılırsa sabit ve eğim aynı sayıyla çarpılır; "
                           "açıklayıcı bir sayıya bölünürse eğim o sayıyla çarpılır, sabit değişmez. Tahmin edilen "
                           "değerler ekonomik olarak aynıdır; R² değişmez. Birimleri değiştirerek deneyin.")
        changes = []
        y_words, x_words = setup.get("change_words", (f"sonucu {y_name} yaptınız", f"açıklayıcıyı {x_name} yaptınız"))
        if y_key != y_options[0][0]:
            changes.append(f"{y_words}: sabit, eğim ve tahmin edilen değerler aynı çarpanla değişti")
        if x_key != x_options[0][0]:
            changes.append(f"{x_words}: eğim bölenle çarpıldı, sabit değişmedi")
        predictions = ("Tahmin edilen değerler ekonomik olarak aynıdır, yalnız yazıldıkları birim değişir"
                       if y_key != y_options[0][0] else "Tahmin edilen değerler hiç değişmez")
        return text + " Tabloda " + "; ".join(changes) + f". {predictions}; R² değişmez (Tablo 4.1)."

    y0, x0 = house.roles[SONUC], house.roles[ACIKLAYICI]
    return interactive_step(
        number=6,
        title="Ölçü birimleri ve katsayıların dönüşümü",
        note=NoteRef("4.7", 0, ("Tablo 4.1",)),
        explanation=(
            "Eğimin birimi $Y$'nin birimi bölü $X$'in birimidir. "
            + str(house.extra.get("units_text", f"Sonuç {phrase(house, y0)}, açıklayıcı {phrase(house, x0)}."))
            + " Birimleri değiştirin: katsayılar değişir; ekonomik ilişki ve R² değişmez. Sonucun birimi değişirse "
              "tahmin edilen değerler yeni birimde yazılır; yalnız açıklayıcının birimi değişirse tahminler hiç "
              "değişmez (Tablo 4.1)."
        ),
        controls=(
            Choice("adim6_y", "Sonucun birimi", tuple((key, label) for key, label, *_ in y_options), y_options[0][0]),
            Choice("adim6_x", "Açıklayıcının birimi", tuple((key, label) for key, label, *_ in x_options),
                   x_options[0][0]),
        ),
        build=build,
        checks=(
            _scalar("sabit_birim", "Sabit terim", 3),
            _scalar("egim_birim", "Eğim", 4),
            *(_scalar(name, label, decimals) for name, label, _, decimals in setup["summaries"]),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 7: dört temel biçim --------------------------------------------------------------------------------

def _forms(house: Case) -> dict[str, tuple[str, str, str, str]]:
    """Kurulabilen biçimler: anahtar → (ad, bağımlı değişken, açıklayıcı değişken, model adı). Logaritma yalnız
    bütün değerleri pozitif değişkenlerde alınır."""

    y, x = house.roles[SONUC], house.roles[ACIKLAYICI]
    ly = log_column(house, y) if positive(house, y) else None
    lx = log_column(house, x) if positive(house, x) else None
    found = {"duzey_duzey": (FORM_NAMES["duzey_duzey"], y, x, MODELS["duzey_duzey"])}
    if ly:
        found["log_duzey"] = (FORM_NAMES["log_duzey"], ly, x, MODELS["log_duzey"])
    if lx:
        found["duzey_log"] = (FORM_NAMES["duzey_log"], y, lx, MODELS["duzey_log"])
    if ly and lx:
        found["log_log"] = (FORM_NAMES["log_log"], ly, lx, MODELS["log_log"])
    return found


def _form_logs(house: Case) -> tuple:
    y, x = house.roles[SONUC], house.roles[ACIKLAYICI]
    return log_operations(house, house.frame, tuple(column for column in (y, x) if positive(house, column)))


def _need_logs(house: Case) -> str:
    y, x = house.roles[SONUC], house.roles[ACIKLAYICI]
    missing = [phrase(house, column) for column in (y, x) if not positive(house, column)]
    return (f"Logaritma yalnız pozitif değerlerde tanımlıdır; {' ve '.join(missing)} "
            f"{'değişkenlerinde' if len(missing) > 1 else 'değişkeninde'} sıfır ya da negatif değer olduğu için "
            "logaritmalı biçimler kurulamaz.")


def _step7(house: Case) -> LabStep:
    forms = _forms(house)
    y, x = house.roles[SONUC], house.roles[ACIKLAYICI]
    title = "Dört temel biçimin adım adım yorumu"
    note_ref = NoteRef("4.9", 0, ("Tablo 4.2", "Tablo 4.3", "Şekil 4.3", "Şekil 4.4"))
    lead = ("Aynı iki değişken düzeyleriyle ya da doğal logaritmalarıyla modele girebilir (§4.8, Tablo 4.2): düzey–düzey "
            "(birim → birim), log–düzey (birim → yaklaşık yüzde $100\\beta_1$), düzey–log (yüzde 1 → $\\beta_1/100$ "
            "birim) ve log–log (yüzde 1 → yaklaşık yüzde $\\beta_1$, esneklik).")
    if len(forms) < 4:
        return LabStep(number=7, title=title, note=note_ref,
                       explanation=f"{lead} {_need_logs(house)} Bu adım için iki değişkenin de bütün değerleri pozitif "
                                   "olmalı.")
    step = float(house.extra.get("x_step", 1))
    step_text = str(house.extra.get("x_step_text", "bir birimlik"))
    axis = {y: display(house, y), x: display(house, x), forms["log_log"][1]: log_label(house, y),
            forms["log_log"][2]: log_label(house, x)}
    terms = (x, forms["log_log"][2], INTERCEPT)

    def build(choices) -> tuple:
        key = choices["adim7_bicim"]
        name, outcome, regressor, model = forms[key]
        return (
            *_form_logs(house),
            *(OLS(m, house.frame, o, (r,), f"{n}: {o} ~ {r}") for n, o, r, m in forms.values()),
            RegressionTable(tuple((f"({i}) {n}", m) for i, (n, _, _, m) in enumerate(forms.values(), start=1)), terms,
                            "dort_bicim", "Dört temel biçim (Tablo 4.3'teki gibi); bağımlı değişken (1), (3) düzey, (2), "
                            "(4) logaritma", stars=False, decimals=4, standard_errors=False),
            ModelValue("egim_dd", "m_dd", "coef", "Düzey–düzey eğim", term=x),
            ModelValue("egim_ld", "m_ld", "coef", "Log–düzey eğim", term=x, decimals=6),
            ModelValue("egim_dl", "m_dl", "coef", "Düzey–log eğim", term=forms["log_log"][2], decimals=3),
            ModelValue("egim_ll", "m_ll", "coef", "Log–log eğim", term=forms["log_log"][2]),
            Scalar("dd_fark", E.mul(E.ref("egim_dd"), step), f"Düzey–düzey: {step_text} → sonuç birimi", decimals=2),
            Scalar("ld_fark", E.mul(E.ref("egim_ld"), 100 * step), f"Log–düzey: {step_text} → yaklaşık yüzde",
                   decimals=2, percent=True),
            Scalar("dl_1", E.div(E.ref("egim_dl"), 100), "Düzey–log: yüzde 1 → sonuç birimi (β̂₁ / 100)", decimals=3),
            Scalar("ll_1", E.ref("egim_ll"), "Log–log: yüzde 1 → yaklaşık yüzde (β̂₁)", decimals=3, percent=True),
            Scalar("ll_10", E.mul(E.ref("egim_ll"), 10), "Log–log: yüzde 10 → yaklaşık yüzde (10 β̂₁)", decimals=2,
                   percent=True),
            ShowModel(model, f"İncelenen biçim: {name} ({outcome} ~ {regressor})", columns=("coef",),
                      stats=("nobs", "r2"), stars=False),
            ScatterPlot(house.frame, regressor, outcome, axis[regressor], axis[outcome],
                        f"{house.extra.get('data_name', 'Verileriniz')}: {name.lower()} model", fit_line=True,
                        size=8, opacity=0.7),
        )

    unit = f" {short_unit(house, y)}" if short_unit(house, y) else ""
    formula = "100 · β̂₁" if step == 1 else f"100 · Δx · β̂₁, Δx = {level_text(step)}"

    def note(state, choices) -> str:
        key = choices["adim7_bicim"]
        s = state.scalars
        flat = at_zero(s["ll_1"], 3)
        elasticity = f"yüzde {sayi(abs(s['ll_1']), 3)}"
        texts = {
            "duzey_duzey": (f"İki değişken de düzeydedir: {step_text} fark tahmin edilen sonuçta ortalama "
                            f"{sayi(s['dd_fark'], digits_for(s['dd_fark'], 2))}{unit} farkla ilişkilidir."),
            "log_duzey": (f"Sonuç logaritmik, açıklayıcı düzeydedir: {step_text} fark sonuçta yaklaşık yüzde "
                          f"{sayi(s['ld_fark'], digits_for(s['ld_fark'], 2))} farkla ilişkilidir ({formula}; küçük "
                          "değişim yaklaşımı)."
                          + (" Eğim küçük görünür, çünkü açıklayıcının birimi (fit²) küçüktür." if not house.own else "")
                          + rough(step * s["egim_ld"])),
            "duzey_log": (f"Sonuç düzeyde, açıklayıcı logaritmiktir: açıklayıcıdaki yüzde 1'lik fark sonuçta yaklaşık "
                          f"β̂₁/100 = {sayi(s['dl_1'], digits_for(s['dl_1'], 3))}{unit} farkla ilişkilidir. Sonuç yüzde değil, sonucun "
                          "birimidir. Sabitin ekonomik anlamı yoktur: ln(X) = 0, yani X = 1 demektir."),
            "log_log": (f"İki değişken de logaritmiktir; eğim esnekliktir: açıklayıcı yüzde 1 daha yüksek olduğunda "
                        f"tahmin edilen sonuç {change(s['ll_1'], elasticity, flat)}"
                        + ("." if flat else f"; yüzde 10 için yaklaşık yüzde {sayi(abs(s['ll_10']), 2)}.")),
        }
        return texts[key] + (
            " Eğimlerin sayısal büyüklükleri çok farklıdır; çünkü her biri farklı bir değişim türünü ölçer (Tablo 4.2). "
            "Tablodaki R²'ler yalnız aynı bağımlı değişkene sahip sütunlar arasında karşılaştırılır: (1) ile (3) "
            "sonucun, (2) ile (4) log sonucun değişkenliğini özetler. Yüzde yorumları küçük değişimler için yaklaşıktır."
        )

    return interactive_step(
        number=7,
        title=title,
        note=note_ref,
        explanation=f"{lead} Tablo dört biçimi aynı veriyle yan yana gösterir; incelenecek biçimi seçin.",
        controls=(Choice("adim7_bicim", "İncelenen biçim", tuple((key, value[0]) for key, value in forms.items()),
                         "duzey_duzey", help="Tablo 4.3'teki gibi dört sütun."),),
        build=build,
        checks=(
            *(_check(f"({i}) {n}: eğim", TableTarget("dort_bicim", r, f"({i}) {n}"), 6 if key == "log_duzey" else 4)
              for i, (key, (n, _, r, _)) in enumerate(forms.items(), start=1)),
            *(_check(f"({i}) {n}: sabit", TableTarget("dort_bicim", INTERCEPT, f"({i}) {n}"), 4)
              for i, (n, *_) in enumerate(forms.values(), start=1)),
            *(_check(f"({i}) {n}: gözlem sayısı", TableTarget("dort_bicim", "n", f"({i}) {n}"), 0)
              for i, (n, *_) in enumerate(forms.values(), start=1)),
            *(_check(f"({i}) {n}: R²", TableTarget("dort_bicim", "r2", f"({i}) {n}"), 4)
              for i, (n, *_) in enumerate(forms.values(), start=1)),
            _scalar("dd_fark", "Düzey–düzey fark", 2), _scalar("ld_fark", "Log–düzey yüzde", 2),
            _scalar("dl_1", "Düzey–log: yüzde 1", 3), _scalar("ll_1", "Log–log: yüzde 1", 3),
            _scalar("ll_10", "Log–log: yüzde 10", 2),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 8: log–düzey model ---------------------------------------------------------------------------------

def _exact_log(case: Case, x: str) -> bool:
    data = case.data[[case.roles[SONUC], x]].astype(float)
    data = data.assign(**{"_ln": np.log(data[case.roles[SONUC]])})
    return exact_fit(data, "_ln", x)


def _step8(case: Case) -> LabStep:
    y = case.roles[SONUC]
    title = "Python çıktısında uyum ve logaritmik model"
    note_ref = NoteRef("4.10", 0, ("Kod 4.3", "Kod 4.4", "Denklem 4.8"))
    if not positive(case, y):
        return LabStep(number=8, title=title, note=note_ref,
                       explanation=("Log–düzey modelde bağımlı değişken sonucun doğal logaritmasıdır. Logaritma yalnız "
                                    f"pozitif değerlerde tanımlıdır; {phrase(case, y)} değişkeninde sıfır ya da negatif "
                                    "değer olduğu için bu adım kurulamaz. Bu adım için sonucun bütün değerleri pozitif "
                                    "olmalı."))
    ly = log_column(case, y)
    default = case.roles[ACIKLAYICI]

    def build(choices) -> tuple:
        x = choices["adim8_x"]
        return (
            *reload(case),
            *log_operations(case, case.frame, (y,)),
            OLS("model_log", case.frame, ly, (x,), f"Log–düzey model: {ly} ~ {x}"),
            ShowModel("model_log", "Statsmodels çıktısından temel alanlar (Kod 4.4'teki gibi)" if x == default
                      else "Seçtiğiniz modelin çıktısı", columns=COEF_QUANTITIES, stats=("nobs", "r2"), stars=False),
            ModelValue("b1_log", "model_log", "coef", f"Eğim β̂₁ ({x})", term=x),
            Scalar("yuzde_log", E.mul(E.ref("b1_log"), 100), "Yaklaşık yüzde fark 100 · β̂₁", decimals=2, percent=True),
            ModelValue("r2_log", "model_log", "r2", "R²", decimals=3),
            Scalar("r2_log_yuzde", E.mul(E.ref("r2_log"), 100), "R², yüzde olarak", decimals=1, percent=True),
        )

    def note(state, choices) -> str:
        x = choices["adim8_x"]
        s = state.scalars
        b1 = s["b1_log"]
        zero = negligible(case, b1, ly, x)
        percent = sayi(abs(100 * b1), digits_for(100 * b1, 2))
        text = (f"Bağımlı değişken `{ly}` ({md(log_label(case, y))}), açıklayıcı değişken düzeydedir: log–düzey model. "
                f"Örneklemde {step_words(case, x)} {plural(case)} {outcome_words(case)} "
                f"{change(b1, '%' + percent, zero)} (100 · β̂₁; küçük değişim yaklaşımı)." + rough(b1)
                + f" R² = {sayi(s['r2_log'], 3)}: log sonucun değişkenliğinin yaklaşık "
                f"%{sayi(s['r2_log_yuzde'], 1)} kadarı örneklemde izlenir; bu sayı düzey modelin R²'siyle "
                "karşılaştırılmaz (farklı bağımlı değişken). Standart hata, t, p-değeri ve güven aralığı sütunları "
                "Konu 7'de yorumlanır.")
        return text + (EXACT_FIT_NOTE if case.own and _exact_log(case, x) else "")

    def checks_for() -> tuple:
        coefficients = []
        for term, name in ((INTERCEPT, "sabit terim"), (default, "eğim")):
            for quantity, label, decimals in (("coef", "katsayı", 4), ("se", "standart hata", 3), ("t", "t", 3),
                                              ("p", "p-değeri", 3), ("ci_low", "%95 GA alt", 3),
                                              ("ci_high", "%95 GA üst", 3)):
                coefficients.append(_check(f"{capital(name)}: {label}", CoefTarget("model_log", term, quantity),
                                           decimals))
        return stable_checks((
            _check("Gözlem sayısı", ModelTarget("model_log", "nobs"), 0),
            _check("R²", ModelTarget("model_log", "r2"), 3),
            *coefficients,
            _scalar("yuzde_log", "Yaklaşık yüzde fark", 2),
            _scalar("r2_log_yuzde", "R², yüzde", 1),
        ), case.own and _exact_log(case, default))

    return interactive_step(
        number=8,
        title=title,
        note=note_ref,
        explanation=(f"`{ly} ~ {default}` log sonucu bağımlı, `{default}` değişkenini açıklayıcı değişken yapar (Denklem "
                     "4.8'in karşılığı). Bu aşamada çıktıda okunanlar: bağımlı değişken, gözlem sayısı, katsayılar ve "
                     "R². Standart hata, t, p-değeri ve güven aralığı sütunları görünür ama henüz yorumlanmaz."),
        controls=(Choice("adim8_x", f"Açıklayıcı değişken (bağımlı değişken: {log_label(case, y)})",
                         options(case, candidates(case)), default, help=f"Varsayılan: {case.name(default)}."),),
        build=build,
        checks=checks_for(),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 9: makale tablosu ----------------------------------------------------------------------------------

def _step9(house: Case) -> LabStep:
    forms = _forms(house)
    title = "Aynı sonucun makale tablosunda gösterimi"
    note_ref = NoteRef("4.11", 0, ("Tablo 4.4",))
    lead = ("Makale tablosu uzun yazılım çıktısını sıkıştırır: her sütun bir model, satırlar katsayılar, altta gözlem "
            "sayısı ve R².")
    if len(forms) < 4:
        return LabStep(number=9, title=title, note=note_ref,
                       explanation=f"{lead} {_need_logs(house)} Bu adım Adım 7'nin dört biçimini kullanır.")
    y = house.roles[SONUC]
    headings = {"duzey_duzey": _title(house, y), "duzey_log": _title(house, y),
                "log_duzey": log_label(house, y), "log_log": log_label(house, y)}
    step = float(house.extra.get("x_step", 1))
    step_text = str(house.extra.get("x_step_text", "bir birimlik"))
    unit = f" {short_unit(house, y)}" if short_unit(house, y) else ""

    def build(choices) -> tuple:
        selected = choices["adim9_sutunlar"]
        columns = tuple((f"({i}) {headings[key]}", forms[key][3]) for i, key in enumerate(selected, start=1))
        terms = tuple(dict.fromkeys((*(forms[key][2] for key in selected), INTERCEPT)))
        default = tuple(selected) == ARTICLE_DEFAULT
        operations: tuple = (RegressionTable(columns, terms, "makale",
                                             "Makale tipi tablo: düzey–düzey ve log–log (Tablo 4.4'teki gibi)" if default
                                             else "Seçtiğiniz modellerle makale tipi tablo", stars=False, decimals=4,
                                             standard_errors=False),)
        if "duzey_duzey" in selected:
            operations += (Scalar("makale_fark", E.mul(E.ref("egim_dd"), step),
                                  f"Düzey–düzey: {step_text} fark (sonucun biriminde)", decimals=2),)
        if "log_log" in selected:
            operations += (Scalar("makale_esneklik", E.ref("egim_ll"), "Log–log: yüzde 1'lik fark (yaklaşık yüzde)",
                                  decimals=3, percent=True),)
        return operations

    def note(state, choices) -> str:
        selected = choices["adim9_sutunlar"]
        s = state.scalars
        text = ("Tabloyu okuma sırası: her sütunun bağımlı değişkeni, katsayının satırı, değişkenlerin düzey mi logaritma "
                "mı olduğu, katsayının birim ya da yüzde diliyle yorumu, gözlem sayısı ve R².")
        if tuple(selected) == ARTICLE_DEFAULT:
            text += (f" Sütun (1)'de {step_text} fark tahmin edilen sonuçta yaklaşık {sayi(s['makale_fark'], 2)}{unit} "
                     f"farkla; sütun (2)'de yüzde 1'lik fark yaklaşık yüzde {sayi(s['makale_esneklik'], 3)} farkla "
                     "ilişkilidir.")
        groups: dict[str, list[int]] = {}
        for number, key in enumerate(selected, start=1):
            groups.setdefault(headings[key], []).append(number)
        if len(selected) > 1 and len(groups) == 1:
            text += (" Sütunların bağımlı değişkeni ve örneklemi aynıdır "
                     f"({sayim(len(house.data))} gözlem): R²'leri doğrudan karşılaştırılabilir.")
        elif len(groups) > 1:
            parts = ", ".join(f"{' ve '.join(f'({n})' for n in numbers)} {'log sonucun' if name.startswith('ln(') else 'sonucun'}"
                              for name, numbers in groups.items())
            text += (f" R²'ler yalnız aynı bağımlı değişkene sahip sütunlar arasında karşılaştırılır: {parts} "
                     "değişkenliğini özetler; farklı bağımlı değişkenli sütunların R²'leri mekanik biçimde sıralanmaz.")
        return text + (" Gerçek makale tablolarında katsayıların altında parantez içinde standart hatalar da bulunur; "
                       "onlar Konu 7'de eklenir.")

    first, second = (f"(1) {headings['duzey_duzey']}", f"(2) {headings['log_log']}")
    return interactive_step(
        number=9,
        title=title,
        note=note_ref,
        explanation=f"{lead} Tablonun sütunlarını Adım 7'deki dört biçimden siz seçin; varsayılan düzey–düzey ve log–log.",
        controls=(MultiChoice("adim9_sutunlar", "Tablodaki modeller", tuple((key, value[0]) for key, value in forms.items()),
                              ARTICLE_DEFAULT, help="Varsayılan: düzey–düzey ve log–log (Tablo 4.4'teki gibi).",
                              maximum=4),),
        build=build,
        checks=(
            _check("(1) eğim", TableTarget("makale", house.roles[ACIKLAYICI], first), 4),
            _check("(2) esneklik", TableTarget("makale", forms["log_log"][2], second), 4),
            _check("(1) sabit", TableTarget("makale", INTERCEPT, first), 4),
            _check("(2) sabit", TableTarget("makale", INTERCEPT, second), 4),
            _check("(1) gözlem sayısı", TableTarget("makale", "n", first), 0),
            _check("(2) gözlem sayısı", TableTarget("makale", "n", second), 0),
            _check("(1) R²", TableTarget("makale", "r2", first), 4),
            _check("(2) R²", TableTarget("makale", "r2", second), 4),
            _scalar("makale_fark", "Düzey–düzey fark", 2),
            _scalar("makale_esneklik", "Log–log: yüzde 1", 3),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


_CHECKLIST = (
    "Bir regresyon çıktısı veya makale tablosunda uyum ve fonksiyonel biçimi değerlendirirken:\n\n1. Araştırma sorusu, "
    "gözlem birimi ve veri yapısı nedir?\n2. Bağımlı ve açıklayıcı değişkenler hangileridir?\n3. Değişkenler düzeyde mi, "
    "logaritmik mi?\n4. Eğim katsayısının işareti, büyüklüğü ve ölçü birimi nedir?\n5. Katsayı birim, yüzde veya "
    "esneklik olarak mı yorumlanmalıdır?\n6. Tahmin edilen değer ve artık nasıl hesaplanır?\n7. R² hangi bağımlı "
    "değişkenin örneklem değişkenliğini özetlemektedir?\n8. R² yorumu nedensellik iddiasına dönüştürülmüş mü?\n9. Farklı "
    "modeller aynı bağımlı değişken ve aynı örneklem üzerinde mi karşılaştırılıyor?\n10. Henüz tanımlanmamış standart "
    "hata ve test sonuçları ileri bölümlere bırakılmış mı?"
)


# --- Tanım ------------------------------------------------------------------------------------------------------

_LABELS = {
    INTERCEPT: "Sabit terim", "tahmin": "Tahmin edilen değer Ŷᵢ", "artik": "Artık ûᵢ", "toplam_kare": "(Yᵢ − Ȳ)²",
    "model_kare": "(Ŷᵢ − Ȳ)²", "artik_kare": "ûᵢ²",
}


def build(case: Case) -> LabSpec:
    """Konu 4 uygulamasını verilen veriyle kurar; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    house = second(case)
    choice = x_choice(case)
    labels = labels_of(case)
    labels.update(labels_of(house))
    labels.update(dict(_units_setup(house).get("labels", {})))
    for owner in (case, house):
        for column in (owner.roles[SONUC], owner.roles[ACIKLAYICI]):
            if positive(owner, column):
                labels.setdefault(log_column(owner, column), log_label(owner, column))
    labels.update(_LABELS)
    spec = LabSpec(
        topic_key=TOPIC,
        title=str(case.extra.get("title", TITLE)),
        note_section="4",
        steps=(_step1(case, choice), _step2(case, choice), _step3(case), _step4(case, choice), _step5(case, choice),
               _step6(house), _step7(house), _step8(case), _step9(house),
               LabStep(number=10, title="Bölüm sonu okuma kontrol listesi", note=NoteRef("4.12", 0),
                       explanation=_CHECKLIST)),
        labels=tuple(labels.items()),
        source=case.source,
    )
    return with_app_values(spec)


# --- Alternatif örnek ----------------------------------------------------------------------------------------

def _house() -> Case:
    return house_case(
        regressors=("area",),
        x_step=100,
        x_step_text="100 fit²lik",
        units_text=("KIELMC'nin 1978 satışlarında fiyat dolar, konut büyüklüğü fit² cinsindendir. Fiyatı bin dolara ya da "
                    "büyüklüğü yüz fit²ye çevirin."),
        units={
            "y": (("dolar", "Dolar (verideki)", "price", 1.0, "dolar"),
                  ("bin", "Bin dolar", "fiyat_bin", 0.001, "bin dolar")),
            "x": (("fit", "Fit² (verideki)", "area", 1.0, "fit²"),
                  ("yuz", "Yüz fit²", "buyukluk_yuz", 100.0, "yüz fit²")),
            "derived": (("fiyat_bin", E.div(E.var("price"), 1000), "Fiyat, bin dolar (price / 1000)"),
                        ("buyukluk_yuz", E.div(E.var("area"), 100), "Büyüklük, yüz fit² (area / 100)")),
            "summaries": (("bir_fit_dolar", "Bir fit²lik farkın tahmini fiyat farkı (dolar)", 1.0, 2),
                          ("yuz_fit_bin", "100 fit²lik farkın tahmini fiyat farkı (bin dolar)", 0.1, 3)),
            "summary_words": ("bir fit²lik farkın tahmini fiyat farkı", "100 fit²lik farkınki"),
            "summary_units": (" dolar", " bin dolar"),
            "labels": {"fiyat_bin": "Satış fiyatı (bin ABD doları, 1978)", "buyukluk_yuz": "Konut büyüklüğü (yüz fit²)"},
        },
    )


def alternative_case() -> Case:
    mean, fitted, observed = 957.945455, 869.523839, 769.0
    return wage2_case(
        ALT_REGRESSORS,
        title="Uygulama: EKK Çıktısı, Uyum ve Fonksiyonel Biçimler (WAGE2, KIELMC)",
        house=_house(),
        accusative={"educ": "eğitimi", "exper": "iş deneyimini", "tenure": "kıdemi"},
        square_unit="Kareler toplamı ((dolar/ay)²)",
        step3_text=("Varsayılan değerler WAGE2'nin ilk çalışanıdır: örneklem ortalaması 957,95, eğitim modelinin bu "
                    "çalışan için tahmini 869,52 ve gözlenen aylık kazanç 769 dolar (kaydırıcılarda tam sayıya "
                    "yuvarlanmış)."),
        adim3=(
            NumberChoice("adim3_ybar", "Örneklem ortalaması Ȳ (dolar)", 0, 3100, round(mean), 1, integer=True, decimals=0,
                         help="WAGE2: 957,95 dolar."),
            NumberChoice("adim3_yhat", "Modelin tahmini Ŷᵢ (dolar)", 0, 3100, round(fitted), 1, integer=True, decimals=0,
                         help="WAGE2'nin ilk çalışanı için eğitim modelinin tahmini: 869,52 dolar."),
            NumberChoice("adim3_y", "Gözlenen değer Yᵢ (dolar)", 0, 3100, round(observed), 1, integer=True, decimals=0,
                         help="WAGE2'nin ilk çalışanının aylık kazancı: 769 dolar."),
        ),
    )


@cache
def alternative() -> LabSpec:
    return build(alternative_case())


STORY = ("Ücret adımları WAGE2'dir (935 erkek çalışan, 1980; aylık kazanç, ABD doları). Notlarda HPRICE1 ile yapılan "
         "konut adımları (6, 7 ve 9) KIELMC'nin 1978 satışlarıdır: 179 konut, fiyat dolar, büyüklük fit².")


# --- Kendi verin ---------------------------------------------------------------------------------------------

CUSTOM = custom_lab(
    build,
    sample_employed,
    ("Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sonuç ve temel açıklayıcı değişken zorunludur ve sayısal olmalıdır. "
     "Ek sayısal değişkenler Adım 1 ve 8'in seçeneklerine eklenir. Notlarda konut verisiyle yapılan adımlar (6, 7 ve 9) "
     "aynı dosyanın sonuç ve açıklayıcı değişkeniyle kurulur; logaritmalı biçimler için iki değişkenin de bütün "
     f"değerleri pozitif olmalı. {ROW_RULE}"),
    roles((1, 2, 3, 4, 5, 6, 7, 8, 9)),
    "Adım 1 ve 8'deki açıklayıcı değişken seçeneklerine eklenir.",
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
