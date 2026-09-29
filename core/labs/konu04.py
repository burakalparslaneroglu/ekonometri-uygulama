"""Konu 4 uygulaması: EKK tahminini değerlendirme, uyum, ölçü birimleri ve temel fonksiyonel biçimler.

Bölüm 4'ün laboratuvar bölümü yoktur; adımlar bölümün çözümlü örnekleridir ve bölüm sırasıyla alt bölümlere
bağlanır: EKK doğrusunun örneklem özellikleri (§4.1, Kod 4.1), ortalama ölçütü (§4.2, Şekil 4.1), bir gözlemdeki
sapmanın iki bileşeni (§4.3), kareler toplamları (§4.4, Kod 4.2, Şekil 4.2), R² ve R² = r² (§4.5), ölçü birimleri
(§4.7; HPRICE1 sayıları §4.9.1'den), dört temel biçim (§4.9, Tablo 4.3, Şekil 4.3–4.4), Python çıktısında log–düzey
model (§4.10, Kod 4.3–4.4), makale tablosu (§4.11, Tablo 4.4) ve kontrol listesi (§4.12). §4.6 ve §4.8'de hesap
yoktur; içerikleri Adım 5 ve 7'nin açıklamalarındadır. Her ``Check`` notlarda basılı bir sayıdır; değer notlardan
kopyalanmıştır, hesaplanmamıştır.

Etkileşim: modelin açıklayıcı değişkeni (Adım 1; Adım 2, 4 ve 5 aynı modeli kullanır), sapma örneğinin üç değeri
(Adım 3), fiyat ve büyüklük birimi (Adım 6), incelenen fonksiyonel biçim (Adım 7), log–düzey modelin açıklayıcı
değişkeni (Adım 8) ve makale tablosunun sütunları (Adım 9). Kod 4.4'teki standart hata, t, p ve güven aralığı
notlardaki gibi yazılım çıktısında görünür; yorumları Konu 7'dedir. Makale tablolarında yalnız katsayılar, gözlem
sayısı ve R² vardır. Yüzde yorumları yaklaşıktır (100·β̂₁); tam yüzde dönüşümü Konu 9'da işlenir.
"""

from __future__ import annotations

from core import wooldridge_data as W
from core.labs import expr as E
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
    LoadWooldridge,
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
from core.labs.sezgi import plain
from core.labs.wording import signed_difference

DATA = "wage1"
HPRICE = "hprice1"
OUTCOME = "wage"
REGRESSORS = ("educ", "exper", "tenure")
IN_WORDS = {
    "educ": ("eğitim süresi", "eğitim"),
    "exper": ("potansiyel deneyimi", "potansiyel deneyim"),
    "tenure": ("mevcut işverendeki kıdemi", "kıdem"),
}
"""Açıklayıcı değişkenin cümle içindeki adı: (çalışanın ... bir yıl daha fazla olan, kısa ad)."""

FORMS = {
    "duzey_duzey": ("Düzey–düzey", "price", "sqrft", "m_dd"),
    "log_duzey": ("Log–düzey", "lprice", "sqrft", "m_ld"),
    "duzey_log": ("Düzey–log", "price", "lsqrft", "m_dl"),
    "log_log": ("Log–log", "lprice", "lsqrft", "m_ll"),
}
"""HPRICE1'de dört temel biçim: (ad, bağımlı değişken, açıklayıcı değişken, model adı); Tablo 4.3'ün sütunları."""
FORM_OPTIONS = tuple((key, name) for key, (name, *_rest) in FORMS.items())
AXIS_LABELS = {
    "price": "Konut fiyatı (bin ABD doları)",
    "lprice": "ln(konut fiyatı)",
    "sqrft": "Konut büyüklüğü (fit²)",
    "lsqrft": "ln(konut büyüklüğü)",
}
TABLE_HEADINGS = {"duzey_duzey": "Fiyat", "log_duzey": "ln(Fiyat)", "duzey_log": "Fiyat", "log_log": "ln(Fiyat)"}
"""Makale tablosunda sütun başlığı: bağımlı değişken (notlardaki Tablo 4.4 gibi)."""
ARTICLE_DEFAULT = ("duzey_duzey", "log_log")
"""Notlardaki Tablo 4.4'ün sütunları."""

X_CHOICE = Choice("adim1_x", "Açıklayıcı değişken (bağımlı değişken: saatlik ücret)", W.options(DATA, REGRESSORS),
                  "educ", help="Notlardaki model: eğitim. Adım 2, 4 ve 5 bu modeli kullanır.")


def _scalar(name: str, expected: float, label: str, decimals: int) -> Check:
    return Check(label, ScalarTarget(name), expected, decimals)


# --- Adım 1: EKK doğrusunun örneklem özellikleri -----------------------------------------

def _properties(choices) -> tuple:
    x = choices["adim1_x"]
    return (
        LoadWooldridge(DATA, "WAGE1 veri seti (Wooldridge, 2020): 526 çalışan"),
        OLS("model", DATA, OUTCOME, (x,), f"Basit regresyon: {OUTCOME} ~ {x}"),
        ModelValue("b0", "model", "coef", "Sabit terim β̂₀", term=INTERCEPT),
        ModelValue("b1", "model", "coef", f"Eğim β̂₁ ({x})", term=x),
        Derive(DATA, "tahmin", E.add(E.ref("b0"), E.mul(E.ref("b1"), E.var(x))), "Tahmin edilen değer Ŷᵢ"),
        Derive(DATA, "artik", E.sub(E.var(OUTCOME), E.var("tahmin")), "Artık ûᵢ = Yᵢ − Ŷᵢ"),
        Statistic(DATA, OUTCOME, "count", "n", "Gözlem sayısı", decimals=0),
        Statistic(DATA, OUTCOME, "mean", "ort_ucret", "Ortalama ücret Ȳ", decimals=6),
        Statistic(DATA, "artik", "sum", "artik_toplami", "Artıkların toplamı Σûᵢ", decimals=10),
        Statistic(DATA, "tahmin", "mean", "ort_tahmin", "Tahmin edilen değerlerin ortalaması", decimals=6),
        Statistic(DATA, x, "mean", "ort_x", "Açıklayıcı değişkenin ortalaması X̄", decimals=4),
        Scalar("dogru_xbar", E.add(E.ref("b0"), E.mul(E.ref("b1"), E.ref("ort_x"))), "Doğrunun X̄'daki değeri Ŷ(X̄)",
               decimals=6),
    )


def _properties_note(state, choices) -> str:
    x = choices["adim1_x"]
    s = state.scalars
    words = IN_WORDS[x][1]
    text = (
        f"Artıkların toplamı {plain(s['artik_toplami'], 10)}: bilgisayarın hesap hassasiyeti içinde sıfırdır "
        "(Denklem 4.1); küçük işaret yazılıma göre değişebilir. Tahmin edilen değerlerin ortalaması "
        f"{plain(s['ort_tahmin'], 6)} ve doğrunun X̄'daki değeri {plain(s['dogru_xbar'], 6)}, ortalama ücrete "
        f"({plain(s['ort_ucret'], 6)} dolar) eşittir (Denklem 4.2–4.3). Bu eşitlikler sabit terimli her EKK "
        "doğrusunun cebirsel özelliğidir; modelin doğru olduğunu ya da her artığın küçük olduğunu göstermez."
    )
    if x != "educ":
        text += f" Açıklayıcı değişkeni {words} yaptığınızda da aynı eşitlikler geçerlidir."
    return text


# --- Adım 2: ortalama ölçütü ------------------------------------------------------------

def _benchmark(choices) -> tuple:
    x = choices["adim1_x"]
    words = IN_WORDS[x][1]
    return (
        Statistic(DATA, OUTCOME, "mean", "ybar", "Yalnız ortalamayı kullanan tahmin Ȳ", decimals=3),
        ScatterPlot(DATA, x, OUTCOME, W.variable(DATA, x).text, "Saatlik ücret (ABD doları/saat)",
                    f"WAGE1: yalnız ortalama ile {words} bilgisini kullanan EKK doğrusu", fit_line=True, size=6,
                    opacity=0.3, lines=(("ybar", 0, "Yalnız ortalama: Ȳ"),)),
    )


def _benchmark_note(state, choices) -> str:
    x = choices["adim1_x"]
    words = IN_WORDS[x][1]
    return (
        f"Kesikli yatay çizgi her çalışan için aynı tahmini ({plain(state.scalars['ybar'], 3)} dolar) verir: "
        "açıklayıcı değişken kullanılmadığında kareli tahmin hatalarını en küçük yapan sabit değer örneklem "
        f"ortalamasıdır. Eğimli doğru {words} bilgisini kullanır ve farklı düzeylere farklı tahminler üretir. Uyum "
        "sorusu şudur: doğru, yalnız ortalamayı kullanan tahmine göre kareli sapmaları ne kadar azaltıyor? Cevap Adım "
        "4–5'teki kareler toplamları ve R²'dir."
    )


# --- Adım 3: bir gözlemdeki sapmanın iki bileşeni ---------------------------------------

def _decomposition(choices) -> tuple:
    ybar, yhat, y = (float(choices[key]) for key in ("adim3_ybar", "adim3_yhat", "adim3_y"))
    return (
        Scalar("toplam_sapma", E.sub(y, ybar), f"Toplam sapma Yᵢ − Ȳ = {plain(y, 2)} − {plain(ybar, 2)}",
               decimals=2),
        Scalar("model_sapma", E.sub(yhat, ybar), f"Model kaynaklı sapma Ŷᵢ − Ȳ = {plain(yhat, 2)} − {plain(ybar, 2)}",
               decimals=2),
        Scalar("artik_sapma", E.sub(y, yhat), f"Artık ûᵢ = Yᵢ − Ŷᵢ = {plain(y, 2)} − {plain(yhat, 2)}", decimals=2),
        Scalar("iki_bilesen", E.add(E.ref("model_sapma"), E.ref("artik_sapma")), "Model kaynaklı sapma + artık",
               decimals=2),
    )


def _decomposition_note(state, choices) -> str:
    s = state.scalars
    total, model, residual = s["toplam_sapma"], s["model_sapma"], s["artik_sapma"]
    text = (f"Toplam sapma {plain(total, 2)} = model kaynaklı sapma {plain(model, 2)} + artık {plain(residual, 2)} "
            "(Denklem 4.4). Eşitlik her gözlemde tanım gereği sağlanır.")
    if total * model < 0:
        text += (" Bu gözlemde model tahmini ortalamanın öbür tarafındadır: model kaynaklı sapma ile toplam sapmanın "
                 "işaretleri farklıdır; artık bu farkı kapatır.")
    elif abs(residual) > abs(total) and total != 0:
        text += " Artık toplam sapmadan büyüktür: bu gözlemde doğru ortalamadan daha kötü bir tahmin vermiştir."
    return text + (" “Model kaynaklı” sapma nedensel bir açıklama değildir; yalnız tahmin edilen değerin ortalamadan "
                   "ne kadar ayrıldığını anlatır.")


# --- Adım 4: kareler toplamları ---------------------------------------------------------

def _sums_of_squares(choices) -> tuple:
    words = IN_WORDS[choices["adim1_x"]][1]
    return (
        Derive(DATA, "toplam_kare", E.power(E.sub(E.var(OUTCOME), E.ref("ort_ucret")), 2), "(Yᵢ − Ȳ)²"),
        Derive(DATA, "model_kare", E.power(E.sub(E.var("tahmin"), E.ref("ort_ucret")), 2), "(Ŷᵢ − Ȳ)²"),
        Derive(DATA, "artik_kare", E.power(E.var("artik"), 2), "ûᵢ²"),
        Statistic(DATA, "toplam_kare", "sum", "tkt", "TKT = Σ(Yᵢ − Ȳ)²", decimals=3),
        Statistic(DATA, "model_kare", "sum", "mkt", "MKT = Σ(Ŷᵢ − Ȳ)²", decimals=3),
        Statistic(DATA, "artik_kare", "sum", "hkt", "HKT = Σûᵢ²", decimals=3),
        Scalar("mkt_hkt", E.add(E.ref("mkt"), E.ref("hkt")), "MKT + HKT", decimals=3),
        ScalarTable((("TKT", E.ref("tkt")), ("MKT", E.ref("mkt")), ("HKT", E.ref("hkt"))), "kareler", decimals=1),
        BarChart("kareler", "deger", "Kareler toplamı türü", "Kareler toplamı ((dolar/saat)²)",
                 f"WAGE1, ücret–{words} modeli: toplam, model ve artık kareler toplamları", decimals=1),
    )


def _sums_note(state, choices) -> str:
    x = choices["adim1_x"]
    s = state.scalars
    words = IN_WORDS[x][1]
    return (
        f"TKT = {plain(s['tkt'], 3)} = MKT {plain(s['mkt'], 3)} + HKT {plain(s['hkt'], 3)} (Denklem 4.5). Toplam "
        f"değişkenliğin yaklaşık {plain(s['mkt'], 1)} birimlik kısmı {words}–ücret doğrusu boyunca hareket ederken "
        f"{plain(s['hkt'], 1)} birimlik kısmı artıklarda kalır. TKT açıklayıcı değişkene bağlı değildir: yalnız "
        "ücretin ortalama çevresindeki değişkenliğidir. EKK'nin en küçük yaptığı ölçüt HKT'dir. Kısaltmalara dikkat: "
        "bazı yazılımlarda `SSR` artık kareler toplamı, bazılarında regresyon kareler toplamıdır."
    )


# --- Adım 5: R² ve korelasyon -------------------------------------------------------------

def _r_squared(choices) -> tuple:
    x = choices["adim1_x"]
    return (
        Scalar("r2_mkt", E.div(E.ref("mkt"), E.ref("tkt")), "R² = MKT / TKT", decimals=4),
        Scalar("r2_hkt", E.sub(1, E.div(E.ref("hkt"), E.ref("tkt"))), "R² = 1 − HKT / TKT", decimals=4),
        ModelValue("r2_yazilim", "model", "r2", "Yazılımın R²'si", decimals=4),
        Scalar("r2_yuzde", E.mul(E.ref("r2_mkt"), 100), "R², yüzde olarak", decimals=1, percent=True),
        PairStatistic(DATA, OUTCOME, x, "corr", "r", f"Ücret ile {IN_WORDS[x][1]} arasındaki korelasyon r",
                      decimals=4),
        Scalar("r_kare", E.power(E.ref("r"), 2), "Korelasyonun karesi r²", decimals=4),
    )


def _r_squared_note(state, choices) -> str:
    x = choices["adim1_x"]
    s = state.scalars
    words = IN_WORDS[x][1]
    return (
        f"R² = {plain(s['r2_mkt'], 4)}: WAGE1 örnekleminde saatlik ücretin ortalama çevresindeki değişkenliğinin "
        f"yaklaşık %{plain(s['r2_yuzde'], 1)} kadarı {_accusative(words)} kullanan doğrusal model tarafından "
        "örneklem içinde izlenir. Yorum örnekleme ve seçilen doğrusal modele aittir; nedensel etki oranı değildir. "
        f"Sabit terimli basit regresyonda R² korelasyonun karesidir: r = {plain(s['r'], 4)}, r² = "
        f"{plain(s['r_kare'], 4)}. Yüksek R² nedenselliği ya da doğru modeli garanti etmez; düşük R² katsayıyı "
        "değersiz yapmaz. R²'ler yalnız aynı bağımlı değişkenli modeller arasında karşılaştırılır (§4.6)."
    )


def _accusative(words: str) -> str:
    """Açıklayıcı değişkenin adı, belirtme durumunda (sabit sözcükler; sayı değil)."""

    return {"eğitim": "eğitimi", "potansiyel deneyim": "potansiyel deneyimi", "kıdem": "kıdemi"}[words]


# --- Adım 6: ölçü birimleri (HPRICE1) -----------------------------------------------------

Y_UNITS = {"bin": ("price", "bin dolar", 1.0), "dolar": ("fiyat_dolar", "dolar", 1000.0)}
X_UNITS = {"fit": ("sqrft", "kare fit", 1.0), "yuz": ("buyukluk_yuz", "yüz kare fit", 100.0)}
"""Birim seçenekleri: (değişken, ad, notlardaki birime göre çarpan). Fiyat notlarda bin dolar, büyüklük kare fittir."""


def _units(choices) -> tuple:
    y_key, x_key = choices["adim6_y"], choices["adim6_x"]
    y, y_name, y_factor = Y_UNITS[y_key]
    x, x_name, x_size = X_UNITS[x_key]
    notes = (y_key, x_key) == ("bin", "fit")
    operations = (
        LoadWooldridge(HPRICE, "HPRICE1 veri seti (Wooldridge, 2020): 88 konut"),
        Derive(HPRICE, "fiyat_dolar", E.mul(1000, E.var("price")), "Fiyat, dolar (1 bin dolar = 1000 dolar)"),
        Derive(HPRICE, "buyukluk_yuz", E.div(E.var("sqrft"), 100), "Büyüklük, yüz kare fit (sqrft / 100)"),
        OLS("model_birim", HPRICE, y, (x,), f"Düzey–düzey model: {y} ~ {x}"),
        ShowModel("model_birim", "Notlardaki birimlerle model (§4.9.1)" if notes else "Seçtiğiniz birimlerle model",
                  columns=("coef",), stats=("nobs", "r2"), stars=False),
        ModelValue("sabit_birim", "model_birim", "coef", f"Sabit terim ({y_name})", term=INTERCEPT, decimals=3),
        ModelValue("egim_birim", "model_birim", "coef", f"Eğim ({y_name} / {x_name})", term=x),
        ModelValue("r2_birim", "model_birim", "r2", "R²", decimals=4),
        Scalar("bir_fit_dolar", E.mul(E.ref("egim_birim"), 1000 / (y_factor * x_size)),
               "Bir kare fitlik farkın tahmini fiyat farkı (dolar)", decimals=1),
        Scalar("yuz_fit_bin", E.mul(E.ref("egim_birim"), 100 / (y_factor * x_size)),
               "100 kare fitlik farkın tahmini fiyat farkı (bin dolar)", decimals=2),
    )
    if notes:
        return operations
    return operations + (
        OLS("model_notlar_birim", HPRICE, "price", ("sqrft",), "Karşılaştırma için notlardaki birimler: price ~ sqrft"),
        RegressionTable((("(1) Notlar", "model_notlar_birim"), ("(2) Seçiminiz", "model_birim")),
                        tuple(dict.fromkeys(("sqrft", x, INTERCEPT))), "birim_karsilastirma",
                        "Notlardaki birimler ile seçtiğiniz birimler", stars=False, decimals=4, standard_errors=False),
    )


def _units_note(state, choices) -> str:
    y_key, x_key = choices["adim6_y"], choices["adim6_x"]
    s = state.scalars
    y_name, x_name = Y_UNITS[y_key][1], X_UNITS[x_key][1]
    text = (f"Eğim {plain(s['egim_birim'], 4)} {y_name} / {x_name}; sabit {plain(s['sabit_birim'], 3)} {y_name}. "
            f"Ekonomik ilişki aynıdır: bir kare fitlik farkın tahmini fiyat farkı {plain(s['bir_fit_dolar'], 1)} "
            f"dolar, 100 kare fitlik farkınki {plain(s['yuz_fit_bin'], 2)} bin dolardır; R² = "
            f"{plain(s['r2_birim'], 4)}.")
    if (y_key, x_key) == ("bin", "fit"):
        return text + (" Birim kuralı (Tablo 4.1): fiyat 1000 ile çarpılırsa sabit ve eğim 1000 ile çarpılır; "
                       "büyüklük 100'e bölünürse eğim 100 ile çarpılır, sabit değişmez. Tahmin edilen fiyatlar "
                       "ekonomik olarak aynıdır (fiyat dolar yazılınca sayı olarak 1000 ile çarpılır); R² değişmez. "
                       "Birimleri değiştirerek deneyin.")
    changes = []
    if y_key == "dolar":
        changes.append("fiyatı dolar yaptınız: sabit, eğim ve tahmin edilen fiyatlar 1000 ile çarpıldı")
    if x_key == "yuz":
        changes.append("büyüklüğü yüz kare fit yaptınız: eğim 100 ile çarpıldı, sabit değişmedi")
    predictions = ("Tahmin edilen fiyatlar ekonomik olarak aynıdır, yalnız yazıldıkları birim değişir"
                   if y_key == "dolar" else "Tahmin edilen fiyatlar hiç değişmez")
    return text + " Tabloda " + "; ".join(changes) + f". {predictions}; R² değişmez (Tablo 4.1)."


# --- Adım 7: dört temel biçim (HPRICE1) ---------------------------------------------------

def _forms(choices) -> tuple:
    key = choices["adim7_bicim"]
    name, y, x, model = FORMS[key]
    shown = (
        ShowModel(model, f"İncelenen biçim: {name} ({y} ~ {x})", columns=("coef",), stats=("nobs", "r2"),
                  stars=False),
        ScatterPlot(HPRICE, x, y, AXIS_LABELS[x], AXIS_LABELS[y], f"HPRICE1: {name.lower()} model", fit_line=True,
                    size=8, opacity=0.7),
    )
    return (
        *(OLS(m, HPRICE, outcome, (regressor,), f"{form}: {outcome} ~ {regressor}")
          for form, outcome, regressor, m in FORMS.values()),
        RegressionTable(tuple((f"({i}) {form}", m) for i, (form, _, _, m) in enumerate(FORMS.values(), start=1)),
                        ("sqrft", "lsqrft", INTERCEPT), "dort_bicim",
                        "Tablo 4.3: dört temel biçim; bağımlı değişken (1), (3) fiyat, (2), (4) ln(fiyat)",
                        stars=False, decimals=4, standard_errors=False),
        ModelValue("egim_dd", "m_dd", "coef", "Düzey–düzey eğim", term="sqrft"),
        ModelValue("egim_ld", "m_ld", "coef", "Log–düzey eğim", term="sqrft", decimals=6),
        ModelValue("egim_dl", "m_dl", "coef", "Düzey–log eğim", term="lsqrft", decimals=3),
        ModelValue("egim_ll", "m_ll", "coef", "Log–log eğim", term="lsqrft"),
        Scalar("dd_100", E.mul(E.ref("egim_dd"), 100), "Düzey–düzey: 100 kare fit → bin dolar (100 β̂₁)",
               decimals=2),
        Scalar("ld_100", E.mul(E.ref("egim_ld"), 10000), "Log–düzey: 100 kare fit → yaklaşık yüzde (100 · 100 β̂₁)",
               decimals=2, percent=True),
        Scalar("dl_1", E.div(E.ref("egim_dl"), 100), "Düzey–log: yüzde 1 → bin dolar (β̂₁ / 100)", decimals=3),
        Scalar("ll_1", E.ref("egim_ll"), "Log–log: yüzde 1 → yaklaşık yüzde (β̂₁)", decimals=3, percent=True),
        Scalar("ll_10", E.mul(E.ref("egim_ll"), 10), "Log–log: yüzde 10 → yaklaşık yüzde (10 β̂₁)", decimals=2,
               percent=True),
        *shown,
    )


_FORM_NOTES = {
    "duzey_duzey": ("Hem fiyat hem büyüklük düzeydedir: bir kare fit daha büyük konutların tahmin edilen fiyatı "
                    "ortalama {b} bin dolar (yaklaşık 140,2 dolar) daha "
                    "yüksektir; 100 kare fitlik fark {dd} bin dolardır."),
    "log_duzey": ("Fiyat logaritmik, büyüklük düzeydedir: bir kare fitlik fark fiyatta yaklaşık yüzde 100·β̂₁ = "
                  "{ld1} farkla ilişkilidir; 100 kare fitlik fark için yaklaşık yüzde {ld}. Eğim küçük görünür, çünkü "
                  "bir kare fit küçük bir birimdir."),
    "duzey_log": ("Fiyat düzeyde, büyüklük logaritmiktir: büyüklükteki yüzde 1'lik fark fiyatta yaklaşık β̂₁/100 = "
                  "{dl} bin dolarlık farkla ilişkilidir. Sonuç yüzde değil, fiyat birimidir. Sabitin ekonomik anlamı "
                  "yoktur: ln(büyüklük) = 0, yani bir kare fitlik konut demektir."),
    "log_log": ("İki değişken de logaritmiktir; eğim esnekliktir: büyüklük yüzde 1 daha yüksek olduğunda tahmin "
                "edilen fiyat yaklaşık yüzde {ll} daha yüksektir; yüzde 10 için yaklaşık yüzde {ll10}."),
}


def _forms_note(state, choices) -> str:
    key = choices["adim7_bicim"]
    s = state.scalars
    text = _FORM_NOTES[key].format(
        b=plain(s["egim_dd"], 4), dd=plain(s["dd_100"], 2), ld1=plain(100 * s["egim_ld"], 4),
        ld=plain(s["ld_100"], 2), dl=plain(s["dl_1"], 3), ll=plain(s["ll_1"], 3), ll10=plain(s["ll_10"], 2),
    )
    return text + (
        " Eğimlerin sayısal büyüklükleri çok farklıdır; çünkü her biri farklı bir değişim türünü ölçer (Tablo 4.2). "
        "Tablo 4.3'teki R²'ler yalnız aynı bağımlı değişkene sahip sütunlar arasında karşılaştırılır: (1) ile (3) "
        "fiyatın, (2) ile (4) log fiyatın değişkenliğini özetler. Yüzde yorumları küçük değişimler için yaklaşıktır."
    )


# --- Adım 8: Python çıktısında log–düzey model --------------------------------------------

def _log_model(choices) -> tuple:
    x = choices["adim8_x"]
    notes = x == "educ"
    return (
        LoadWooldridge(DATA, "WAGE1 veri seti: önceki adımlarda türetilen sütunlar olmadan yeniden yüklenir"),
        OLS("model_log", DATA, "lwage", (x,), f"Log–düzey model: lwage ~ {x}" + (" (Kod 4.3)" if notes else "")),
        ShowModel("model_log", "Kod 4.4: statsmodels çıktısından temel alanlar" if notes else
                  "Seçtiğiniz modelin çıktısı", columns=COEF_QUANTITIES, stats=("nobs", "r2"), stars=False),
        ModelValue("b1_log", "model_log", "coef", f"Eğim β̂₁ ({x})", term=x),
        Scalar("yuzde_log", E.mul(E.ref("b1_log"), 100), "Yaklaşık yüzde ücret farkı 100 · β̂₁", decimals=2,
               percent=True),
        ModelValue("r2_log", "model_log", "r2", "R²", decimals=3),
        Scalar("r2_log_yuzde", E.mul(E.ref("r2_log"), 100), "R², yüzde olarak", decimals=1, percent=True),
    )


def _log_note(state, choices) -> str:
    x = choices["adim8_x"]
    s = state.scalars
    phrase = IN_WORDS[x][0]
    b1 = s["b1_log"]
    return (
        "Bağımlı değişken `lwage` (log saatlik ücret), açıklayıcı "
        f"değişken düzeydedir: log–düzey model. {phrase.capitalize()} "
        f"bir yıl daha yüksek olan çalışanların saatlik ücreti örneklemde yaklaşık %{plain(abs(100 * b1), 2)} daha "
        f"{signed_difference(b1)} (100 · β̂₁; küçük değişim yaklaşımı). R² = {plain(s['r2_log'], 3)}: log ücret "
        f"değişkenliğinin yaklaşık %{plain(s['r2_log_yuzde'], 1)} kadarı örneklemde izlenir; bu sayı düzey ücret "
        "modelinin R²'siyle karşılaştırılmaz (farklı bağımlı değişken). Standart hata, t, p-değeri ve güven aralığı "
        "sütunları Konu 7'de yorumlanır."
    )


# --- Adım 9: makale tablosu ---------------------------------------------------------------

def _article(choices) -> tuple:
    selected = choices["adim9_sutunlar"]
    columns = tuple((f"({i}) {TABLE_HEADINGS[key]}", FORMS[key][3]) for i, key in enumerate(selected, start=1))
    terms = tuple(dict.fromkeys((*(FORMS[key][2] for key in selected), INTERCEPT)))
    title = ("Tablo 4.4: konut büyüklüğü ile fiyat arasındaki iki basit regresyon" if tuple(selected) == ARTICLE_DEFAULT
             else "Seçtiğiniz modellerle makale tipi tablo: konut büyüklüğü ve fiyat")
    operations: tuple = (
        RegressionTable(columns, terms, "makale", title, stars=False, decimals=4, standard_errors=False),
    )
    if "duzey_duzey" in selected:
        operations += (Scalar("makale_100", E.mul(E.ref("egim_dd"), 100),
                              "Düzey–düzey: 100 kare fitlik fark (bin dolar)", decimals=2),)
    if "log_log" in selected:
        operations += (Scalar("makale_esneklik", E.ref("egim_ll"), "Log–log: yüzde 1'lik fark (yaklaşık yüzde)",
                              decimals=3, percent=True),)
    return operations


def _article_note(state, choices) -> str:
    selected = choices["adim9_sutunlar"]
    s = state.scalars
    text = ("Tabloyu okuma sırası: her sütunun bağımlı değişkeni, katsayının satırı, değişkenlerin düzey mi logaritma "
            "mı olduğu, katsayının birim ya da yüzde diliyle yorumu, gözlem sayısı ve R².")
    if tuple(selected) == ("duzey_duzey", "log_log"):
        text += (" Sütun (1)'de 100 kare fitlik büyüklük farkı tahmin edilen fiyatta yaklaşık "
                 f"{plain(s['makale_100'], 2)} bin dolarlık farkla; sütun (2)'de yüzde 1'lik fark yaklaşık yüzde "
                 f"{plain(s['makale_esneklik'], 3)} farkla ilişkilidir.")
    groups: dict[str, list[int]] = {}
    for number, key in enumerate(selected, start=1):
        groups.setdefault(TABLE_HEADINGS[key], []).append(number)
    if len(selected) > 1 and len(groups) == 1:
        text += (" Sütunların bağımlı değişkeni aynıdır ve örneklem "
                 "aynıdır (88 konut): R²'leri doğrudan karşılaştırılabilir.")
    elif len(groups) > 1:
        owners = {"Fiyat": "fiyatın", "ln(Fiyat)": "log fiyatın"}
        parts = ", ".join(f"{' ve '.join(f'({n})' for n in numbers)} {owners[name]}"
                          for name, numbers in groups.items())
        text += (f" R²'ler yalnız aynı bağımlı değişkene sahip sütunlar arasında karşılaştırılır: {parts} "
                 "değişkenliğini özetler; farklı bağımlı değişkenli sütunların R²'leri mekanik biçimde sıralanmaz.")
    return text + (" Gerçek makale tablolarında katsayıların altında parantez "
                   "içinde standart hatalar da bulunur; onlar Konu 7'de eklenir.")


# --- Tanım ---------------------------------------------------------------------------

_FORM_TABLE = (
    ("(1) Düzey–düzey", "sqrft", 0.1402, 4), ("(1) Düzey–düzey", INTERCEPT, 11.2041, 4),
    ("(2) Log–düzey", "sqrft", 0.000402, 6), ("(2) Log–düzey", INTERCEPT, 4.8240, 4),
    ("(3) Düzey–log", "lsqrft", 297.911, 3), ("(3) Düzey–log", INTERCEPT, -1962.420, 3),
    ("(4) Log–log", "lsqrft", 0.8727, 4), ("(4) Log–log", INTERCEPT, -0.9751, 4),
)
"""Tablo 4.3: dört biçimin eğim ve sabit tahminleri (basılı basamakla)."""
_FORM_R2 = (("(1) Düzey–düzey", 0.6208), ("(2) Log–düzey", 0.5837), ("(3) Düzey–log", 0.5630), ("(4) Log–log", 0.5530))
_TERM_NAMES = {"sqrft": "eğim", "lsqrft": "eğim", INTERCEPT: "sabit"}

STEPS = (
    interactive_step(
        number=1,
        title="EKK doğrusunun temel örneklem özellikleri",
        note=NoteRef("4.1", 0, ("Kod 4.1", "Denklem 4.1–4.3")),
        explanation=(
            "Sabit terimli EKK tahmininde üç eşitlik her örneklemde sağlanır: artıkların toplamı sıfırdır ($\\sum "
            "\\hat u_i = 0$), tahmin edilen değerlerin ortalaması $\\bar Y$'ye eşittir ve doğru $(\\bar X, \\bar Y)$ "
            "noktasından geçer. Önce model tahmin edilir; tahmin edilen değer ve artık her çalışan için hesaplanır."
        ),
        controls=(X_CHOICE,),
        build=_properties,
        checks=(
            _scalar("n", 526, "Kod 4.1: gözlem sayısı", 0),
            _scalar("ort_ucret", 5.896103, "Kod 4.1: ortalama ücret", 6),
            _scalar("artik_toplami", 0.0, "Kod 4.1: artıkların toplamı", 10),
            _scalar("ort_tahmin", 5.896103, "Kod 4.1: tahmin edilen değerlerin ortalaması", 6),
            _scalar("dogru_xbar", 5.896103, "Denklem 4.3: doğru (X̄, Ȳ) noktasından geçer", 6),
        ),
        note_for=lambda state, choices: _properties_note(state, choices),
    ),
    interactive_step(
        number=2,
        title="Ortalama ölçütü ile regresyon doğrusunu karşılaştırmak",
        note=NoteRef("4.2", 0, ("Şekil 4.1",)),
        explanation=(
            "Açıklayıcı değişken kullanmadan her çalışan için aynı değeri tahmin edersek, kareli tahmin hatalarını en "
            "küçük yapan sabit değer örneklem ortalamasıdır: $\\hat Y_i^{(0)} = \\bar Y$. EKK doğrusu bu doğal "
            "başlangıç ölçütüyle karşılaştırılır. Bu adım Adım 1'deki modeli kullanır."
        ),
        uses=(X_CHOICE,),
        build=_benchmark,
        checks=(_scalar("ybar", 5.896, "§4.2: ortalama saatlik ücret", 3),),
        note_for=lambda state, choices: _benchmark_note(state, choices),
    ),
    interactive_step(
        number=3,
        title="Bir gözlemdeki sapmanın iki bileşeni",
        note=NoteRef("4.3", 0, ("Denklem 4.4",)),
        explanation=(
            "Her gözlemde $Y_i - \\bar Y = (\\hat Y_i - \\bar Y) + \\hat u_i$ (Denklem 4.4): toplam sapma, model "
            "kaynaklı sapma ile artığın toplamıdır. Notlardaki örnekte örneklem ortalaması 6, modelin tahmini 8, "
            "gözlenen ücret 9,5'tir. Değerleri değiştirerek iki bileşenin toplam sapmayı nasıl paylaştığını görün."
        ),
        controls=(
            NumberChoice("adim3_ybar", "Örneklem ortalaması Ȳ", 0.0, 20.0, 6.0, 0.5, help="Notlarda 6."),
            NumberChoice("adim3_yhat", "Modelin tahmini Ŷᵢ", 0.0, 20.0, 8.0, 0.5, help="Notlarda 8."),
            NumberChoice("adim3_y", "Gözlenen değer Yᵢ", 0.0, 20.0, 9.5, 0.5, help="Notlarda 9,5."),
        ),
        build=_decomposition,
        checks=(
            _scalar("toplam_sapma", 3.5, "§4.3: toplam sapma 9,5 − 6", 1),
            _scalar("model_sapma", 2, "§4.3: model kaynaklı sapma 8 − 6", 0),
            _scalar("artik_sapma", 1.5, "§4.3: artık 9,5 − 8", 1),
            _scalar("iki_bilesen", 3.5, "§4.3: 2 + 1,5", 1),
        ),
        note_for=lambda state, choices: _decomposition_note(state, choices),
    ),
    interactive_step(
        number=4,
        title="Kareler toplamlarının ayrıştırılması",
        note=NoteRef("4.4", 0, ("Kod 4.2", "Denklem 4.5", "Şekil 4.2")),
        explanation=(
            "Sapmaların işaretleri birbirini götürebildiği için kareleri toplanır: TKT $= \\sum (Y_i - \\bar Y)^2$, "
            "MKT $= \\sum (\\hat Y_i - \\bar Y)^2$, HKT $= \\sum \\hat u_i^2$. Sabit terimli EKK modelinde TKT = MKT + "
            "HKT (Denklem 4.5). Bu adım Adım 1'deki modeli kullanır."
        ),
        uses=(X_CHOICE,),
        build=_sums_of_squares,
        checks=(
            _scalar("tkt", 7160.414, "Kod 4.2: TKT", 3),
            _scalar("mkt", 1179.732, "Kod 4.2: MKT", 3),
            _scalar("hkt", 5980.682, "Kod 4.2: HKT", 3),
            _scalar("mkt_hkt", 7160.414, "Kod 4.2: MKT + HKT", 3),
            _scalar("mkt", 1179.7, "§4.4: doğru boyunca hareket eden kısım", 1),
            _scalar("hkt", 5980.7, "§4.4: artıklarda kalan kısım", 1),
        ),
        note_for=lambda state, choices: _sums_note(state, choices),
    ),
    interactive_step(
        number=5,
        title="Belirleme katsayısı R²",
        note=NoteRef("4.5", 0, ("Denklem 4.6",)),
        explanation=(
            "$R^2 = \\text{MKT}/\\text{TKT} = 1 - \\text{HKT}/\\text{TKT}$ (Denklem 4.6): bağımlı değişkenin örneklem "
            "değişkenliğinin doğrusal model tarafından izlenen payı. Sabit terimli basit regresyonda $R^2 = r_{XY}^2$ "
            "(§4.5). Yorumun dört sınırı: örnekleme aittir, bağımlı değişkenin örneklem değişkenliğiyle ilgilidir, "
            "seçilen doğrusal model içindir, nedensel etki oranı "
            "değildir (§4.5–4.6). Bu adım Adım 1'deki modeli kullanır."
        ),
        uses=(X_CHOICE,),
        build=_r_squared,
        checks=(
            _scalar("r2_mkt", 0.1648, "§4.5: R² = 1179,732 / 7160,414", 4),
            _scalar("r2_hkt", 0.1648, "§4.5: R² = 1 − 5980,682 / 7160,414", 4),
            _scalar("r2_yazilim", 0.1648, "§4.5: R² (yazılım)", 4),
            _scalar("r2_yuzde", 16.5, "§4.5: yaklaşık yüzde 16,5", 1),
            _scalar("r", 0.4059, "§4.5: ücret–eğitim korelasyonu", 4),
            _scalar("r_kare", 0.1648, "§4.5: 0,4059²", 4),
        ),
        note_for=lambda state, choices: _r_squared_note(state, choices),
    ),
    interactive_step(
        number=6,
        title="Ölçü birimleri ve katsayıların dönüşümü",
        note=NoteRef("4.7", 0, ("Tablo 4.1",)),
        explanation=(
            "Eğimin birimi $Y$'nin birimi bölü $X$'in birimidir. HPRICE1'de fiyat bin dolar, konut büyüklüğü kare "
            "fittir (sayılar §4.9.1'de). Fiyatı dolara ya da büyüklüğü yüz kare fite çevirin: katsayılar değişir; "
            "ekonomik ilişki ve R² değişmez. Fiyat dolar yazılırsa tahmin edilen fiyatlar yeni birimde yazılır; yalnız "
            "büyüklüğün birimi değişirse tahminler hiç değişmez (Tablo 4.1)."
        ),
        controls=(
            Choice("adim6_y", "Fiyat birimi", (("bin", "Bin dolar (notlardaki)"), ("dolar", "Dolar")), "bin"),
            Choice("adim6_x", "Büyüklük birimi", (("fit", "Kare fit (notlardaki)"), ("yuz", "Yüz kare fit")), "fit"),
        ),
        build=_units,
        checks=(
            _scalar("sabit_birim", 11.204, "§4.9.1: sabit terim", 3),
            _scalar("egim_birim", 0.1402, "§4.9.1: eğim (bin dolar / kare fit)", 4),
            _scalar("bir_fit_dolar", 140.2, "§4.9.1: bir kare fit, yaklaşık 140,2 dolar", 1),
            _scalar("yuz_fit_bin", 14.02, "§4.9.1: 100 kare fit, 14,02 bin dolar", 2),
        ),
        note_for=lambda state, choices: _units_note(state, choices),
    ),
    interactive_step(
        number=7,
        title="Dört temel biçimin adım adım yorumu",
        note=NoteRef("4.9", 0, ("Tablo 4.2", "Tablo 4.3", "Şekil 4.3", "Şekil 4.4")),
        explanation=(
            "Aynı iki değişken düzeyleriyle ya da doğal logaritmalarıyla modele girebilir (§4.8, Tablo 4.2): "
            "düzey–düzey (birim → birim), log–düzey (birim → yaklaşık yüzde $100\\beta_1$), düzey–log (yüzde 1 → "
            "$\\beta_1/100$ birim) ve log–log (yüzde 1 → yaklaşık yüzde $\\beta_1$, esneklik). Tablo 4.3 dört biçimi "
            "HPRICE1'de yan yana gösterir; incelenecek biçimi seçin."
        ),
        controls=(
            Choice("adim7_bicim", "İncelenen biçim", FORM_OPTIONS, "duzey_duzey",
                   help="Tablo 4.3'ün dört sütunu. Notlarda §4.9.1 düzey–düzey, §4.9.4 log–log örneğidir."),
        ),
        build=_forms,
        checks=(
            *(Check(f"Tablo 4.3: {heading}, {_TERM_NAMES[term]}", TableTarget("dort_bicim", term, heading), value,
                    decimals)
              for heading, term, value, decimals in _FORM_TABLE),
            *(Check(f"Tablo 4.3: {heading}, gözlem sayısı", TableTarget("dort_bicim", "n", heading), 88, 0)
              for heading, _ in _FORM_R2),
            *(Check(f"Tablo 4.3: {heading}, R²", TableTarget("dort_bicim", "r2", heading), value, 4)
              for heading, value in _FORM_R2),
            _scalar("dd_100", 14.02, "§4.9.1 ve Tablo 4.3: 100 kare fit, 14,02 bin dolar", 2),
            _scalar("ld_100", 4.02, "Tablo 4.3: log–düzey, 100 kare fit, yüzde 4,02", 2),
            _scalar("dl_1", 2.979, "Tablo 4.3: düzey–log, yüzde 1, 2,979 bin dolar", 3),
            _scalar("ll_1", 0.873, "§4.9.4 ve Tablo 4.3: log–log, yüzde 0,873", 3),
            _scalar("ll_10", 8.73, "§4.9.4: yüzde 10, yaklaşık yüzde 8,73", 2),
            _scalar("egim_ll", 0.8727, "Denklem 4.9: log–log eğim", 4),
        ),
        note_for=lambda state, choices: _forms_note(state, choices),
    ),
    interactive_step(
        number=8,
        title="Python çıktısında uyum ve logaritmik model",
        note=NoteRef("4.10", 0, ("Kod 4.3", "Kod 4.4", "Denklem 4.8")),
        explanation=(
            "`lwage ~ educ` log saatlik ücreti bağımlı, eğitim yılını açıklayıcı değişken yapar (Denklem 4.8). Bu "
            "aşamada çıktıda okunanlar: bağımlı değişken, gözlem sayısı, katsayılar ve R². Standart hata, t, p-değeri "
            "ve güven aralığı sütunları görünür ama henüz yorumlanmaz."
        ),
        controls=(
            Choice("adim8_x", "Açıklayıcı değişken (bağımlı değişken: log saatlik ücret)", W.options(DATA, REGRESSORS),
                   "educ", help="Notlardaki Kod 4.3: eğitim."),
        ),
        build=_log_model,
        checks=(
            Check("Kod 4.4: gözlem sayısı", ModelTarget("model_log", "nobs"), 526, 0),
            Check("Kod 4.4: R²", ModelTarget("model_log", "r2"), 0.186, 3),
            Check("Kod 4.4: sabit terim", CoefTarget("model_log", INTERCEPT), 0.5838, 4),
            Check("Kod 4.4: eğitim katsayısı", CoefTarget("model_log", "educ"), 0.0827, 4),
            Check("Kod 4.4: sabit terimin standart hatası", CoefTarget("model_log", INTERCEPT, "se"), 0.097, 3),
            Check("Kod 4.4: eğitim katsayısının standart hatası", CoefTarget("model_log", "educ", "se"), 0.008, 3),
            Check("Kod 4.4: sabit terimin t değeri", CoefTarget("model_log", INTERCEPT, "t"), 5.998, 3),
            Check("Kod 4.4: eğitim katsayısının t değeri", CoefTarget("model_log", "educ", "t"), 10.935, 3),
            Check("Kod 4.4: sabit terimin p-değeri", CoefTarget("model_log", INTERCEPT, "p"), 0.000, 3),
            Check("Kod 4.4: eğitim katsayısının p-değeri", CoefTarget("model_log", "educ", "p"), 0.000, 3),
            Check("Kod 4.4: sabit terim, %95 GA alt sınır", CoefTarget("model_log", INTERCEPT, "ci_low"), 0.393, 3),
            Check("Kod 4.4: sabit terim, %95 GA üst sınır", CoefTarget("model_log", INTERCEPT, "ci_high"), 0.775, 3),
            Check("Kod 4.4: eğitim, %95 GA alt sınır", CoefTarget("model_log", "educ", "ci_low"), 0.068, 3),
            Check("Kod 4.4: eğitim, %95 GA üst sınır", CoefTarget("model_log", "educ", "ci_high"), 0.098, 3),
            _scalar("yuzde_log", 8.27, "§4.9.2 ve §4.10: yaklaşık yüzde 8,27", 2),
            _scalar("r2_log_yuzde", 18.6, "§4.10: yaklaşık yüzde 18,6", 1),
        ),
        note_for=lambda state, choices: _log_note(state, choices),
    ),
    interactive_step(
        number=9,
        title="Aynı sonucun makale tablosunda gösterimi",
        note=NoteRef("4.11", 0, ("Tablo 4.4",)),
        explanation=(
            "Makale tablosu uzun yazılım çıktısını sıkıştırır: her sütun bir model, satırlar katsayılar, altta gözlem "
            "sayısı ve R². Notlardaki Tablo 4.4 düzey–düzey ve log–log modelleri yan yana koyar. Tablonun sütunlarını "
            "Adım 7'deki dört biçimden siz seçin."
        ),
        controls=(
            MultiChoice("adim9_sutunlar", "Tablodaki modeller", FORM_OPTIONS, ARTICLE_DEFAULT,
                        help="Notlardaki Tablo 4.4: düzey–düzey ve log–log.", maximum=4),
        ),
        build=_article,
        checks=(
            Check("Tablo 4.4: (1) konut büyüklüğü", TableTarget("makale", "sqrft", "(1) Fiyat"), 0.1402, 4),
            Check("Tablo 4.4: (2) ln(konut büyüklüğü)", TableTarget("makale", "lsqrft", "(2) ln(Fiyat)"), 0.8727, 4),
            Check("Tablo 4.4: (1) sabit", TableTarget("makale", INTERCEPT, "(1) Fiyat"), 11.2041, 4),
            Check("Tablo 4.4: (2) sabit", TableTarget("makale", INTERCEPT, "(2) ln(Fiyat)"), -0.9751, 4),
            Check("Tablo 4.4: (1) gözlem sayısı", TableTarget("makale", "n", "(1) Fiyat"), 88, 0),
            Check("Tablo 4.4: (2) gözlem sayısı", TableTarget("makale", "n", "(2) ln(Fiyat)"), 88, 0),
            Check("Tablo 4.4: (1) R²", TableTarget("makale", "r2", "(1) Fiyat"), 0.6208, 4),
            Check("Tablo 4.4: (2) R²", TableTarget("makale", "r2", "(2) ln(Fiyat)"), 0.5530, 4),
            _scalar("makale_100", 14.02, "§4.11: 100 kare fit, 14,02 bin dolar", 2),
            _scalar("makale_esneklik", 0.873, "§4.11: yüzde 1, yaklaşık yüzde 0,873", 3),
        ),
        note_for=lambda state, choices: _article_note(state, choices),
    ),
    LabStep(
        number=10,
        title="Bölüm sonu okuma kontrol listesi",
        note=NoteRef("4.12", 0),
        explanation=(
            "Bir regresyon çıktısı veya makale tablosunda uyum ve fonksiyonel biçimi değerlendirirken:\n\n1. Araştırma "
            "sorusu, gözlem birimi ve veri yapısı nedir?\n2. Bağımlı ve açıklayıcı değişkenler hangileridir?\n3. "
            "Değişkenler düzeyde mi, logaritmik mi?\n4. Eğim katsayısının işareti, büyüklüğü ve ölçü birimi nedir?\n5. "
            "Katsayı birim, yüzde veya esneklik olarak mı yorumlanmalıdır?\n6. Tahmin edilen değer ve artık nasıl "
            "hesaplanır?\n7. R² hangi bağımlı değişkenin örneklem değişkenliğini özetlemektedir?\n8. R² yorumu "
            "nedensellik iddiasına dönüştürülmüş mü?\n9. Farklı modeller aynı bağımlı değişken ve aynı örneklem "
            "üzerinde mi karşılaştırılıyor?\n10. Henüz tanımlanmamış "
            "standart hata ve test sonuçları ileri bölümlere bırakılmış mı?"
        ),
    ),
)


KONU04_LAB = LabSpec(
    topic_key="konu04",
    title="Uygulama: EKK Çıktısı, Uyum ve Temel Fonksiyonel Biçimler",
    note_section="4",
    steps=STEPS,
    labels=(
        *W.labels(HPRICE, DATA),
        (INTERCEPT, "Sabit terim"),
        ("tahmin", "Tahmin edilen değer Ŷᵢ"),
        ("artik", "Artık ûᵢ"),
        ("toplam_kare", "(Yᵢ − Ȳ)²"),
        ("model_kare", "(Ŷᵢ − Ȳ)²"),
        ("artik_kare", "ûᵢ²"),
        ("fiyat_dolar", "Konut fiyatı (ABD doları)"),
        ("buyukluk_yuz", "Konut büyüklüğü (yüz fit²)"),
    ),
    consistency_notes=(
        "Veri notlardaki gibi WAGE1 ve HPRICE1'dir; notlardaki kod, betikler, uygulama ve üretilen kod onları "
        "wooldridge paketinden okur. Kod 4.2'deki kareler toplamları üç ondalıkla basılır (eski çıktının altı ondalığı "
        "notlar klasöründeki CSV kopyasından geliyordu; kopya paketten en çok 9,2 × 10⁻⁷ farklıydı).",
        "Tablo 4.3 (HPRICE1'de dört biçim) önceden yalnız sunumdaydı; notlara eklendi. Sunumdaki tam yüzde dönüşümü "
        "100[exp(β₁) − 1] Konu 9'un konusudur; bu konuda yaklaşık yorum (100·β₁) kullanılır.",
        "R² = r² eşitliğinin türetimi §4.5'e eklendi (Bölüm 0 §0.9 bu türetime gönderme yapar).",
    ),
)
