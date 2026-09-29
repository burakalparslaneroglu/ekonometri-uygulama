"""Konu 5 uygulaması: çoklu regresyon modeli ve ceteris paribus yorumu.

Bölüm 5'in çözümlü örnekleri bölüm sırasıyla: basit ve çoklu ücret modeli ile katsayı yorumları (§5.1, §5.3,
Denklem 5.5), birden fazla birimlik fark, iki çalışanın karşılaştırılması ve ceteris paribus çizgileri (§5.3,
Şekil 5.1), tahmin edilen değer ve artık (§5.4), kontrollerin doğrusal katkısını ayırma (§5.5, Şekil 5.2), Python
çıktısı ve Tablo 5.1 (§5.6, Kod 5.1–5.2), katsayı değişimi (§5.7, Şekil 5.3), HPRICE1 (§5.8, Tablo 5.2–5.3,
Şekil 5.4), R² ve düzeltilmiş R² (§5.9, Denklem 5.8), makale tablosu (§5.10, Tablo 5.4) ve kontrol değişkeni seçimi
(§5.11). §5.2'de hesap yoktur; §5.12'deki bütünleşik uygulama bir egzersizdir (cevapları eklerde). Her ``Check``
notlarda basılı bir sayıdır; değer notlardan kopyalanmıştır, hesaplanmamıştır.

Etkileşim: ücret modelinin açıklayıcı değişkenleri (Adım 1; Adım 2, 3, 5, 6 ve 8 aynı modeli kullanır; seçenekler
kukla olmayan sayısal değişkenlerdir: eğitim, deneyim, kıdem ve bakmakla yükümlü kişi sayısı, kukla değişkenler Konu
10'dadır),
eğitim farkı (Adım 2), tahmin edilen çalışanın özellikleri (Adım 3), kısmi ilişkisi incelenen değişken (Adım 4),
konut modelinin açıklayıcı değişkenleri (Adım 7), A, B ve C modellerinde gözlem sayısı (Adım 8) ve makale tablosunun
sütunları (Adım 9). Kod 5.2'deki standart hata, t, p ve güven aralığı notlardaki gibi yazılım çıktısında görünür;
yorumları Konu 7'dedir. Makale tablolarında katsayılar, gözlem sayısı, R² ve düzeltilmiş R² vardır.
"""

from __future__ import annotations

from core import wooldridge_data as W
from core.labs import expr as E
from core.labs.sezgi import plain
from core.labs.spec import (
    COEF_QUANTITIES,
    INTERCEPT,
    OLS,
    CellTarget,
    Check,
    Choice,
    CoefTarget,
    Derive,
    GroupedBarChart,
    InlineData,
    JoinColumns,
    LabSpec,
    LabStep,
    LineChart,
    LoadWooldridge,
    ModelTarget,
    ModelValue,
    MultiChoice,
    NoteRef,
    NumberChoice,
    RegressionTable,
    Residuals,
    Scalar,
    ScalarTable,
    ScalarTarget,
    ScatterPlot,
    ShowModel,
    Statistic,
    Support,
    TableTarget,
    interactive_step,
)
from core.labs.wording import signed_difference

DATA = "wage1"
HPRICE = "hprice1"
OUTCOME = "wage"
WAGE_OPTIONS = ("educ", "exper", "tenure", "numdep")
"""Ücret modelinin seçilebilen açıklayıcı değişkenleri: WAGE1'deki kukla olmayan sayısal değişkenler (kukla değişkenler
Konu 10)."""
NOTES_X = ("educ", "exper", "tenure")
"""Notlardaki çoklu ücret modeli (Denklem 5.5)."""
HOUSE_OPTIONS = ("sqrft", "bdrms", "lotsize")
PROFILE_NUMDEP = 1
"""Bakmakla yükümlü kişi sayısı modele eklenirse tahmin edilen çalışanlarda bu değer kullanılır (WAGE1'deki medyan)."""
WORDS = {
    "educ": ("eğitim", "eğitim yılı bir yıl daha yüksek olan"),
    "exper": ("deneyim", "potansiyel deneyimi bir yıl daha yüksek olan"),
    "tenure": ("kıdem", "mevcut işverendeki kıdemi bir yıl daha yüksek olan"),
    "numdep": ("bakmakla yükümlü kişi sayısı", "bakmakla yükümlü olduğu kişi sayısı bir kişi daha fazla olan"),
}
"""Açıklayıcı değişkenin cümle içindeki kısa adı ve bir birimlik farkın yazımı."""
TITLES = {"educ": "Eğitim", "exper": "Deneyim", "tenure": "Kıdem", "numdep": "Bakmakla yükümlü kişi sayısı"}
HOUSE_WORDS = {"sqrft": "konut büyüklüğü", "bdrms": "yatak odası sayısı", "lotsize": "arsa büyüklüğü"}
ARTICLE = {
    "basit": ("Ücret", "wage", ("educ",), "Ücret ~ eğitim"),
    "coklu": ("Ücret", "wage", NOTES_X, "Ücret ~ eğitim + deneyim + kıdem"),
    "log_coklu": ("ln(Ücret)", "lwage", NOTES_X, "ln(Ücret) ~ eğitim + deneyim + kıdem"),
    "deneyimli": ("Ücret", "wage", ("educ", "exper"), "Ücret ~ eğitim + deneyim"),
    "log_basit": ("ln(Ücret)", "lwage", ("educ",), "ln(Ücret) ~ eğitim"),
}
"""Makale tablosunun seçilebilen sütunları: (başlık, bağımlı değişken, açıklayıcı değişkenler, seçenek adı)."""
ARTICLE_DEFAULT = ("basit", "coklu", "log_coklu")
"""Notlardaki Tablo 5.4'ün sütunları."""

RELOAD = "WAGE1 veri seti: önceki adımlarda türetilen sütunlar olmadan yeniden yüklenir"
X_MULTI = MultiChoice(
    "adim1_x", "Çoklu modelin açıklayıcı değişkenleri (bağımlı değişken: saatlik ücret)",
    W.options(DATA, WAGE_OPTIONS), NOTES_X,
    help="Notlardaki model: eğitim, deneyim ve kıdem (Denklem 5.5). Adım 2, 3, 5, 6 ve 8 bu modeli kullanır.",
)


def _scalar(name: str, expected: float, label: str, decimals: int) -> Check:
    return Check(label, ScalarTarget(name), expected, decimals)


def _listing(items: list[str]) -> str:
    """Türkçe sıralama: "a", "a ve b", "a, b ve c"."""

    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " ve " + items[-1]


def _ordered(names, options=WAGE_OPTIONS) -> tuple[str, ...]:
    """Adlar seçeneklerin sırasıyla (tablolarda notlardaki satır sırası)."""

    chosen = set(names)
    return tuple(name for name in options if name in chosen)


def _predict(regressors, values: dict, prefix: str = "b_") -> E.Expr:
    """Tahmin edilen değer: sabit + Σ katsayı × değer; katsayılar ``prefix``le adlandırılmış skalerlerdir."""

    expression = E.ref(f"{prefix}sabit")
    for name in regressors:
        expression = E.add(expression, E.mul(E.ref(f"{prefix}{name}"), values[name]))
    return expression


def _notes_model(state, name: str = "coklu_notlar"):
    """Notlardan farklı bir seçimde yan yana gösterilen notlardaki model (Adım 1'de tahmin edilir)."""

    return state.models[name]


def _interpretation(name: str, value: float, others: list[str]) -> str:
    """Bir katsayının ceteris paribus cümlesi (notlardaki üç unsur: değişen değişken, tahmin edilen fark, sabitler)."""

    phrase = WORDS[name][1]
    if others:
        held = _listing([WORDS[other][0] for other in others])
        start = f"{held[0].upper()}{held[1:]} aynıyken, {phrase}"
    else:
        start = f"{phrase[0].upper()}{phrase[1:]}"
    return (f"{start} çalışanların tahmin edilen saatlik ücreti yaklaşık {plain(abs(value), 3)} dolar daha "
            f"{signed_difference(value)}.")


# --- Adım 1: basit modelden çoklu modele ---------------------------------------------------

def _models(choices) -> tuple:
    chosen = choices["adim1_x"]
    notes = tuple(chosen) == NOTES_X
    formula = " + ".join(chosen)
    operations: list = [
        LoadWooldridge(DATA, "WAGE1 veri seti (Wooldridge, 2020): 526 çalışan"),
        OLS("basit", DATA, OUTCOME, ("educ",), "Basit regresyon (§5.1): wage ~ educ"),
        ModelValue("basit_sabit", "basit", "coef", "Basit model: sabit terim", term=INTERCEPT),
        ModelValue("basit_egitim", "basit", "coef", "Basit model: eğitim katsayısı", term="educ"),
        OLS("coklu", DATA, OUTCOME, tuple(chosen),
            f"Çoklu regresyon: wage ~ {formula}" + (" (Denklem 5.5)" if notes else "")),
        ModelValue("b_sabit", "coklu", "coef", "Çoklu model: sabit terim", term=INTERCEPT),
        *(ModelValue(f"b_{name}", "coklu", "coef", f"Çoklu model: {WORDS[name][0]} katsayısı", term=name)
          for name in chosen),
    ]
    columns = [("(1) Basit model", "basit"), ("(2) Çoklu model" if notes else "(2) Seçtiğiniz model", "coklu")]
    used = {"educ", *chosen}
    if not notes:
        operations.append(OLS("coklu_notlar", DATA, OUTCOME, NOTES_X,
                              "Karşılaştırma için notlardaki model: wage ~ educ + exper + tenure"))
        columns.append(("(3) Notlardaki model", "coklu_notlar"))
        used |= set(NOTES_X)
    title = ("Basit ve çoklu ücret modeli (bağımlı değişken: saatlik ücret)" if notes
             else "Basit model, seçtiğiniz model ve notlardaki model (bağımlı değişken: saatlik ücret)")
    operations.append(RegressionTable(tuple(columns), (*_ordered(used), INTERCEPT), "modeller", title, stars=False,
                                      decimals=4, standard_errors=False, r2=False))
    return tuple(operations)


def _models_note(state, choices) -> str:
    chosen = list(choices["adim1_x"])
    s = state.scalars
    sentences = [_interpretation(name, s[f"b_{name}"], [other for other in chosen if other != name])
                 for name in chosen]
    text = " ".join(sentences)
    if len(chosen) == 1:
        text += " Modelde tek açıklayıcı değişken var; katsayı bir basit regresyon eğimidir."
    if "educ" not in chosen:
        text += " Eğitim modelde yok; eğitim farkları hata teriminde kalır."
    elif len(chosen) > 1:
        text += (f" Eğitim katsayısı basit modelde {plain(s['basit_egitim'], 4)}, çoklu modelde "
                 f"{plain(s['b_educ'], 4)}: iki katsayı farklı karşılaştırmalar yapar (§5.7).")
    if len(chosen) == 1:
        return text + (" Tek değişkenli modelde sabit tutulan başka bir değişken yoktur: “diğerleri aynıyken” unsuru "
                       "ancak çoklu modelde vardır. Yorum bir ilişkidir; nedensel etki iddiası değildir.")
    return text + (" Her yorumda üç unsur vardır: değişen değişken ve birimi, bağımlı değişkendeki tahmin edilen fark "
                   "ve birimi, sabit tutulan değişkenler. Ceteris paribus yorum yalnız modelde bulunan değişkenleri "
                   "sabit tutar; nedensel etki iddiası değildir.")


# --- Adım 2: birden fazla birimlik fark, iki çalışan ve ceteris paribus çizgileri ---------

EXPER_LEVELS = (5, 20, 35)
"""Şekil 5.1'deki deneyim düzeyleri (yıl)."""


def _comparison(choices) -> tuple:
    chosen = choices["adim1_x"]
    gap = int(choices["adim2_fark"])
    extra = ("numdep",) if "numdep" in chosen else ()
    columns = ("calisan", "educ", "exper", "tenure", *extra)
    rows = (("A", 12, 10, 5, *(PROFILE_NUMDEP for _ in extra)), ("B", 12 + gap, 10, 5, *(PROFILE_NUMDEP for _ in extra)))
    operations: list = []
    if "educ" in chosen:
        operations.append(Scalar("egitim_farki", E.mul(gap, E.ref("b_educ")),
                                 f"{gap} yıllık eğitim farkı: {gap} × β̂ (eğitim), dolar", decimals=3))
    operations += [
        InlineData("calisanlar", columns, rows, "İki çalışan: deneyim ve kıdem aynı, eğitim farklı"),
        Derive("calisanlar", "tahmin", _predict(chosen, {name: E.var(name) for name in chosen}),
               "Tahmin edilen saatlik ücret Ŷ"),
        Statistic("calisanlar", "tahmin", "value", "tahmin_A", "A çalışanının tahmini (dolar)", where=("calisan", "A"),
                  decimals=2),
        Statistic("calisanlar", "tahmin", "value", "tahmin_B", "B çalışanının tahmini (dolar)", where=("calisan", "B"),
                  decimals=2),
        Scalar("fark_BA", E.sub(E.ref("tahmin_B"), E.ref("tahmin_A")), "Tahmin farkı B − A (dolar)", decimals=3),
    ]
    if {"educ", "exper"} <= set(chosen):
        held = {"educ": E.var("educ"), "tenure": E.ref("kidem_medyan"), "numdep": E.ref("bakmakla_medyan")}
        fixed = [text for name, text in (("tenure", "kıdem medyan düzeyinde (2 yıl)"),
                                         ("numdep", f"bakmakla yükümlü kişi sayısı medyan düzeyinde ({PROFILE_NUMDEP})"))
                 if name in chosen]
        title = ("Şekil 5.1: " if tuple(chosen) == NOTES_X else "Seçtiğiniz modelle Şekil 5.1: ") + (
            f"{_listing(fixed)} sabitken üç" if fixed else "üç") + " deneyim düzeyinde tahmin çizgileri"
        operations += [
            Statistic(DATA, "tenure", "median", "kidem_medyan", "Kıdemin medyanı (yıl)", decimals=0),
            *((Statistic(DATA, "numdep", "median", "bakmakla_medyan", "Bakmakla yükümlü kişi sayısının medyanı",
                         decimals=0),) if extra else ()),
            Support("izgara", "educ", 0, 18, "Eğitim ızgarası: 0–18 yıl (WAGE1'deki aralık)"),
            *(Derive("izgara", f"tahmin_{level}", _predict(chosen, {**held, "exper": level}),
                     f"Deneyim {level} yıl: tahmin edilen ücret") for level in EXPER_LEVELS),
            LineChart("izgara", "educ", f"tahmin_{EXPER_LEVELS[0]}", "Eğitim (yıl)",
                      "Tahmin edilen saatlik ücret (ABD doları/saat)", title, markers=False, legend=f"Deneyim = {EXPER_LEVELS[0]} yıl",
                      series=tuple((f"tahmin_{level}", f"Deneyim = {level} yıl") for level in EXPER_LEVELS[1:])),
        ]
    return tuple(operations)


def _comparison_note(state, choices) -> str:
    chosen = choices["adim1_x"]
    gap = int(choices["adim2_fark"])
    s = state.scalars
    if "educ" not in chosen:
        notes = _notes_model(state).params["educ"]
        return ("Eğitim modelde olmadığı için iki çalışanın tahmini aynıdır: model eğitim farkını kullanmaz. Eğitimi "
                f"Adım 1'de modele ekleyin. Notlardaki modelde (eğitim, deneyim, kıdem) fark {gap} × {plain(notes, 4)} = "
                f"{plain(gap * notes, 3)} dolardır.")
    text = (f"B çalışanının eğitimi A'nınkinden {gap} yıl fazla; deneyim ve kıdem aynı. Tahminler "
            f"{plain(s['tahmin_A'], 2)} ve {plain(s['tahmin_B'], 2)} dolar; fark {plain(s['fark_BA'], 3)} dolar, yani "
            f"{gap} × β̂ (eğitim) = {plain(s['egitim_farki'], 3)}. "
            + ("Diğer değişkenler aynı tutulduğu için fark yalnız eğitim katkısıdır. " if len(chosen) > 1 else
               "Modelde eğitimden başka değişken olmadığı için fark yalnız eğitim katkısıdır. ")
            + "Birden fazla değişken aynı anda değişirse toplam fark, bütün katsayı katkılarının toplamıdır (§5.3).")
    if tuple(chosen) != NOTES_X:
        notes = _notes_model(state).params["educ"]
        text += (f" Notlardaki modelde (eğitim, deneyim, kıdem) aynı fark {gap} × {plain(notes, 4)} = "
                 f"{plain(gap * notes, 3)} dolardır.")
    if "numdep" in chosen:
        text += f" Bakmakla yükümlü kişi sayısı iki çalışanda da {PROFILE_NUMDEP} alındı (WAGE1'deki medyan)."
    if "exper" in chosen:
        text += (" Şekilde doğrular birbirine paraleldir: deneyim düzeyi değişince doğru yukarı ya da aşağı kayar, "
                 "eğitim eğimi değişmez; çünkü modelde etkileşim terimi yoktur (etkileşimler Konu 11).")
    else:
        text += " Deneyim modelde olmadığı için Şekil 5.1'deki deneyim düzeylerine göre çizgiler çizilmez."
    return text


# --- Adım 3: tahmin edilen değer ve artık -----------------------------------------------------

def _fitted(choices) -> tuple:
    chosen = choices["adim1_x"]
    profile = {"educ": float(choices["adim3_egitim"]), "exper": float(choices["adim3_deneyim"]),
               "tenure": float(choices["adim3_kidem"]), "numdep": float(PROFILE_NUMDEP)}
    observed = float(choices["adim3_ucret"])
    return (
        Scalar("birey_tahmin", _predict(chosen, profile), "Tahmin edilen ücret Ŷ (dolar)", decimals=3),
        Scalar("birey_artik", E.sub(observed, E.ref("birey_tahmin")), "Artık û = Y − Ŷ (dolar)", decimals=3),
        Derive(DATA, "tahmin", _predict(chosen, {name: E.var(name) for name in chosen}), "Tahmin edilen değer Ŷᵢ"),
        Derive(DATA, "artik", E.sub(E.var(OUTCOME), E.var("tahmin")), "Artık ûᵢ = Yᵢ − Ŷᵢ"),
        Statistic(DATA, "artik", "sum", "artik_toplami", "Örneklemde artıkların toplamı Σûᵢ", decimals=10),
        Statistic(DATA, "tahmin", "mean", "ort_tahmin", "Tahmin edilen değerlerin ortalaması", decimals=6),
        Statistic(DATA, OUTCOME, "mean", "ort_ucret", "Ortalama ücret Ȳ", decimals=6),
    )


def _fitted_note(state, choices) -> str:
    s = state.scalars
    observed = float(choices["adim3_ucret"])
    residual = s["birey_artik"]
    if residual > 0:
        sign = ("Pozitif artık: gözlenen ücret model tahmininden yüksektir. Fark modelde bulunmayan özelliklerden, "
                "doğrusal biçimin sınırlılığından veya ölçüm hatasından kaynaklanabilir.")
    elif residual < 0:
        sign = "Negatif artık: gözlenen ücret model tahmininden düşüktür."
    else:
        sign = "Artık sıfır: gözlenen ücret tahminle aynıdır."
    chosen = choices["adim1_x"]
    definitions = ("Tanımlar basit regresyondakiyle aynıdır; yalnız tahmin edilen değerde daha fazla açıklayıcı "
                   "değişken vardır." if len(chosen) > 1 else
                   "Modelde tek açıklayıcı değişken var; tanımlar basit regresyondakiyle aynıdır.")
    text = (f"Seçilen özelliklerdeki çalışan için tahmin {plain(s['birey_tahmin'], 3)} dolar; gözlenen ücret "
            f"{plain(observed, 1)} dolar ise artık {plain(residual, 3)} dolardır. {sign} {definitions}")
    unused = [WORDS[name][0] for name in ("educ", "exper", "tenure") if name not in chosen]
    if unused:
        text += (f" Modelde olmayan değişkenin ({unused[0]}) değeri tahmini etkilemez." if len(unused) == 1 else
                 f" Modelde olmayan değişkenlerin ({_listing(unused)}) değeri tahmini etkilemez.")
    if "numdep" in chosen:
        text += f" Bakmakla yükümlü kişi sayısı {PROFILE_NUMDEP} alındı (WAGE1'deki medyan)."
    if s["birey_tahmin"] < 0:
        text += (" Tahmin sıfırın altında: bu özellikler veride çok nadirdir ve doğrusal model verinin kenarında "
                 "anlamsız değerler üretebilir (§3.11).")
    if tuple(chosen) != NOTES_X:
        profile = {"educ": float(choices["adim3_egitim"]), "exper": float(choices["adim3_deneyim"]),
                   "tenure": float(choices["adim3_kidem"])}
        params = _notes_model(state).params
        notes = params[INTERCEPT] + sum(params[name] * value for name, value in profile.items())
        text += f" Notlardaki modelle (eğitim, deneyim, kıdem) aynı çalışanın tahmini {plain(notes, 3)} dolardır."
    return text + (f" Sabit terimli EKK'de artıkların toplamı sıfırdır ({plain(s['artik_toplami'], 10)}, hesap "
                   f"hassasiyeti içinde) ve tahmin edilen değerlerin ortalaması ({plain(s['ort_tahmin'], 6)}) "
                   "ortalama ücrete eşittir.")


# --- Adım 4: doğrusal katkıyı ayırmak (kısmi regresyon) ------------------------------------

PARTIAL_OPTIONS = tuple((name, W.variable(DATA, name).text) for name in NOTES_X)


def _partial(choices) -> tuple:
    focus = choices["adim4_x"]
    others = tuple(name for name in NOTES_X if name != focus)
    controls = " + ".join(others)
    short = WORDS[focus][0]
    return (
        LoadWooldridge(DATA, RELOAD),
        OLS("yardimci_ucret", DATA, OUTCOME, others, f"1. adım: ücreti kontrollere göre tahmin et (wage ~ {controls})"),
        Residuals(DATA, "ucret_artik", "yardimci_ucret", "Ücretin kontrollerle açıklanamayan kısmı (artık)"),
        OLS("yardimci_x", DATA, focus, others, f"2. adım: {short} değişkenini kontrollere göre tahmin et "
                                               f"({focus} ~ {controls})"),
        Residuals(DATA, "x_artik", "yardimci_x", f"{TITLES[focus]} değişkeninin kontrollerle açıklanamayan kısmı"),
        OLS("kismi", DATA, "ucret_artik", ("x_artik",), "3. adım: artıkların artıklara göre regresyonu"),
        ModelValue("kismi_egim", "kismi", "coef", "Artıklar regresyonunun eğimi", term="x_artik", decimals=6),
        OLS("tam", DATA, OUTCOME, NOTES_X, "Çoklu regresyon: wage ~ educ + exper + tenure"),
        ModelValue("tam_katsayi", "tam", "coef", f"Çoklu regresyonda {short} katsayısı", term=focus, decimals=6),
        Scalar("kismi_fark", E.sub(E.ref("kismi_egim"), E.ref("tam_katsayi")), "Aradaki sayısal fark", decimals=12),
        ScatterPlot(DATA, "x_artik", "ucret_artik", f"Kontrollerden arındırılmış {short} (yıl)",
                    "Arındırılmış ücret (ABD doları/saat)", f"{TITLES[focus]} değişkeninin kısmi ilişkisi: artıklar "
                    "üzerinden", fit_line=True, size=6, opacity=0.4),
    )


def _partial_note(state, choices) -> str:
    focus = choices["adim4_x"]
    s = state.scalars
    others = _listing([WORDS[name][0] for name in NOTES_X if name != focus])
    return (f"Artıklar regresyonunun eğimi {plain(s['kismi_egim'], 6)}, çoklu regresyondaki {WORDS[focus][0]} "
            f"katsayısı {plain(s['tam_katsayi'], 6)}: aynı sayıdır (fark {plain(s['kismi_fark'], 12)}). "
            f"Yatay eksen, {WORDS[focus][0]} değişkeninin {others} ile açıklanamayan kısmıdır; dikey eksen ücretin "
            "aynı kontrollerle açıklanamayan kısmıdır. “Sabit tutma” birebir eşleştirme değildir; kontrollerin "
            "doğrusal katkısı ayrılır. İleri ekonometride bu sonuç Frisch–Waugh–Lovell teoremi olarak bilinir. Grafik "
            "nedensellik kanıtı değildir; modelde olmayan faktörler iki artıkta da kalabilir (§5.5)."
            + ("" if focus == "educ" else " Notlardaki Şekil 5.2 eğitim için çizilmiştir: orada iki eğim de 0,599'dur "
               "(Denklem 5.5)."))


# --- Adım 5: Python çıktısı ve Tablo 5.1 --------------------------------------------------

def _output(choices) -> tuple:
    chosen = choices["adim1_x"]
    notes = tuple(chosen) == NOTES_X
    formula = " + ".join(chosen)
    operations: list = [
        LoadWooldridge(DATA, RELOAD),
        ShowModel("coklu", "Kod 5.2: çoklu ücret modelinin temel çıktısı" if notes
                  else "Seçtiğiniz modelin temel çıktısı", columns=COEF_QUANTITIES, stats=("nobs",), stars=False),
        ModelValue("r2_kod", "coklu", "r2", "R²", decimals=3),
        ModelValue("r2d_kod", "coklu", "adj_r2", "Düzeltilmiş R²", decimals=3),
        Scalar("r2_yuzde", E.mul(E.ref("r2_kod"), 100), "R², yüzde olarak", decimals=1, percent=True),
        OLS("log_coklu", DATA, "lwage", tuple(chosen), f"Log ücret modeli: lwage ~ {formula}"),
    ]
    if "educ" in chosen:
        operations += [
            ModelValue("log_egitim", "log_coklu", "coef", "Log modelde eğitim katsayısı", term="educ"),
            Scalar("log_yuzde", E.mul(E.ref("log_egitim"), 100), "Yaklaşık yüzde ücret farkı 100 · β̂ (eğitim)",
                   decimals=1, percent=True),
        ]
    columns = [("(1) Ücret", "basit"), ("(2) Ücret", "coklu"), ("(3) ln(Ücret)", "log_coklu")]
    used = {"educ", *chosen}
    if not notes:
        columns.append(("(4) Ücret, notlardaki model", "coklu_notlar"))
        used |= set(NOTES_X)
    title = ("Tablo 5.1: WAGE1 verisinde basit ve çoklu ücret modelleri" if notes
             else "Tablo 5.1'in seçtiğiniz modelle kurulmuş hâli; (4) notlardaki model")
    operations.append(RegressionTable(tuple(columns), (*_ordered(used), INTERCEPT), "tablo51", title, stars=False,
                                      decimals=4, standard_errors=False, adj_r2=True))
    return tuple(operations)


def _output_note(state, choices) -> str:
    chosen = choices["adim1_x"]
    s = state.scalars
    text = (f"Çıktı okuma sırası: bağımlı değişken `wage`, gözlem sayısı 526, katsayılar `coef` sütununda. R² = "
            f"{plain(s['r2_kod'], 3)}: saatlik ücretin örneklem ortalaması çevresindeki değişkenliğinin yaklaşık "
            f"%{plain(s['r2_yuzde'], 1)} kadarı model tarafından örneklem içinde izlenir; düzeltilmiş R² = "
            f"{plain(s['r2d_kod'], 3)} (Adım 8). Standart hata, t, p-değeri ve güven aralığı sütunları görünür ama "
            "henüz yorumlanmaz (Konu 7).")
    if "educ" in chosen:
        held = "diğer değişkenler aynıyken " if len(chosen) > 1 else ""
        text += (f" Sütun (3)'te bağımlı değişken log ücrettir: {held}bir ek eğitim yılı, tahmin edilen ücrette "
                 f"yaklaşık %{plain(s['log_yuzde'], 1)} farkla ilişkilidir.")
    return text + (" Sütun (2) ile (3)'ün R²'leri karşılaştırılmaz: bağımlı değişkenlerden biri ücret, diğeri log "
                   "ücrettir.")


# --- Adım 6: bir değişken eklendiğinde katsayı neden değişir? ------------------------------

def _change(choices) -> tuple:
    chosen = choices["adim1_x"]
    operations: list = []
    simple = {}
    for name in chosen:
        if name == "educ":
            simple[name] = "basit_egitim"
            continue
        operations += [
            OLS(f"basit_{name}", DATA, OUTCOME, (name,), f"Basit regresyon: wage ~ {name}"),
            ModelValue(f"basit_b_{name}", f"basit_{name}", "coef", f"Basit model: {WORDS[name][0]} katsayısı",
                       term=name),
        ]
        simple[name] = f"basit_b_{name}"
    operations += [
        ScalarTable(tuple((TITLES[name], E.ref(simple[name])) for name in chosen), "katsayi_basit", decimals=4),
        ScalarTable(tuple((TITLES[name], E.ref(f"b_{name}")) for name in chosen), "katsayi_coklu", decimals=4),
        JoinColumns("katsayilar", (("Basit model", "katsayi_basit", "deger"), ("Çoklu model", "katsayi_coklu", "deger")),
                    decimals=4, heading="Açıklayıcı değişken"),
        GroupedBarChart("katsayilar", "Açıklayıcı değişken", "Katsayı (dolar)",
                        "Basit ve çoklu modelde katsayılar (eğitim: Şekil 5.3; deneyim ve kıdem de eklendi)"
                        if tuple(chosen) == NOTES_X else "Seçtiğiniz modelle: basit ve çoklu modelde katsayılar",
                        series="sutun", decimals=3),
    ]
    return tuple(operations)


def _change_note(state, choices) -> str:
    chosen = choices["adim1_x"]
    table = state.tables["katsayilar"]
    if len(chosen) == 1:
        return (f"Modelde tek açıklayıcı değişken var ({WORDS[chosen[0]][0]}): çoklu model basit modelle aynıdır ve "
                "katsayı değişmez. Katsayıların neden değiştiğini görmek için Adım 1'de en az iki değişken seçin.")
    parts = [f"{WORDS[name][0]} {plain(table.loc[TITLES[name], 'Basit model'], 4)} → "
             f"{plain(table.loc[TITLES[name], 'Çoklu model'], 4)}" for name in chosen]
    text = (f"Basit modelden çoklu modele katsayılar: {'; '.join(parts)}. Katsayının değişmesi yazılım hatası değildir; "
            "basit model bir değişkenin farklı olduğu bütün çalışanları diğer farkları ayırmadan karşılaştırır, çoklu "
            "model diğer değişkenleri aynı tutar.")
    if tuple(chosen) != NOTES_X:
        text += " Notlardaki modelde eğitim katsayısı 0,5414 → 0,5990 (Şekil 5.3)."
    return text + (" Değişim tek başına hangi modelin doğru olduğunu söylemez: eklenen değişken önemli bir faktör "
                   "olabilir, araştırma sorusu için gereksiz olabilir ya da temel değişkenin bir sonucu olabilir. "
                   "Dışarıda kalan değişkenin katsayıyı hangi koşullarda ve hangi yönde etkilediği Konu 6'da işlenir.")


# --- Adım 7: HPRICE1 ----------------------------------------------------------------------

HOUSES = (("A", 2000, 3, 7000), ("B", 2000, 4, 7000), ("C", 2250, 4, 8000))
"""Tablo 5.3'teki konutlar: (konut, büyüklük, yatak odası, arsa)."""


def _houses(choices) -> tuple:
    chosen = choices["adim7_x"]
    notes = tuple(chosen) == HOUSE_OPTIONS
    formula = " + ".join(chosen)
    operations: list = [
        LoadWooldridge(HPRICE, "HPRICE1 veri seti (Wooldridge, 2020): 88 konut"),
        OLS("fiyat_basit", HPRICE, "price", ("sqrft",), "Basit model: price ~ sqrft"),
        OLS("fiyat_coklu", HPRICE, "price", tuple(chosen),
            f"Çoklu model: price ~ {formula}" + (" (Denklem 5.7)" if notes else "")),
        ModelValue("f_sabit", "fiyat_coklu", "coef", "Sabit terim (bin dolar)", term=INTERCEPT),
        *(ModelValue(f"f_{name}", "fiyat_coklu", "coef", f"{HOUSE_WORDS[name].capitalize()} katsayısı",
                     term=name, decimals=6 if name == "lotsize" else 4) for name in chosen),
    ]
    if "sqrft" in chosen:
        operations.append(Scalar("yuz_fit", E.mul(E.ref("f_sqrft"), 100),
                                 "100 kare fit daha büyük konut: fiyat farkı (bin dolar)", decimals=2))
    if "lotsize" in chosen:
        operations.append(Scalar("bin_fit_arsa", E.mul(E.ref("f_lotsize"), 1000),
                                 "1.000 kare fit daha büyük arsa: fiyat farkı (bin dolar)", decimals=2))
    columns = [("(1) Fiyat", "fiyat_basit"), ("(2) Fiyat", "fiyat_coklu")]
    used = {"sqrft", *chosen}
    if not notes:
        operations.append(OLS("fiyat_notlar", HPRICE, "price", HOUSE_OPTIONS,
                              "Karşılaştırma için notlardaki model: price ~ sqrft + bdrms + lotsize"))
        columns.append(("(3) Fiyat, notlardaki model", "fiyat_notlar"))
        used |= set(HOUSE_OPTIONS)
    terms = _ordered(used, HOUSE_OPTIONS)
    title = ("Tablo 5.2: HPRICE1 verisinde basit ve çoklu konut fiyatı modelleri" if notes
             else "Tablo 5.2'nin seçtiğiniz modelle kurulmuş hâli; (3) notlardaki model")
    operations += [
        RegressionTable(tuple(columns), (*terms, INTERCEPT), "tablo52", title, stars=False, decimals=4,
                        standard_errors=False, adj_r2=True,
                        term_decimals=(("lotsize", 6),) if "lotsize" in terms else ()),
        InlineData("konutlar", ("konut", *HOUSE_OPTIONS), HOUSES, "Tablo 5.3: tahmin edilecek üç konut"),
        Derive("konutlar", "tahmin", _predict(chosen, {name: E.var(name) for name in chosen}, "f_"),
               "Tahmin edilen fiyat (bin dolar)"),
        *(Statistic("konutlar", "tahmin", "value", f"konut_{house}", f"{house} konutunun tahmini fiyatı (bin dolar)",
                    where=("konut", house), decimals=2) for house, *_ in HOUSES),
        Scalar("konut_fark", E.sub(E.ref("konut_B"), E.ref("konut_A")), "Tahmin farkı B − A (bin dolar)", decimals=2),
        Derive(HPRICE, "fiyat_tahmin", _predict(chosen, {name: E.var(name) for name in chosen}, "f_"),
               "Tahmin edilen fiyat Ŷᵢ"),
        ScatterPlot(HPRICE, "price", "fiyat_tahmin", "Gözlenen fiyat (bin ABD doları)",
                    "Tahmin edilen fiyat (bin ABD doları)",
                    "Şekil 5.4: HPRICE1'de gözlenen ve tahmin edilen fiyat" if notes
                    else "Seçtiğiniz modelle: HPRICE1'de gözlenen ve tahmin edilen fiyat",
                    size=8, opacity=0.7, lines=((0, 1, "45 derece çizgisi"),)),
    ]
    return tuple(operations)


def _houses_note(state, choices) -> str:
    chosen = choices["adim7_x"]
    s = state.scalars
    parts = []
    if "sqrft" in chosen:
        held = _listing([HOUSE_WORDS[name] for name in chosen if name != "sqrft"]) if len(chosen) > 1 else ""
        parts.append((f"{held[0].upper()}{held[1:]} aynıyken, " if held else "") +
                     f"100 kare fit daha büyük konutun tahmin edilen fiyatı yaklaşık {plain(abs(s['yuz_fit']), 2)} bin "
                     f"dolar daha {signed_difference(s['yuz_fit'])}.")
    held = ", diğer değişkenler aynıyken," if len(chosen) > 1 else ""
    if "bdrms" in chosen:
        parts.append(f"Bir ek yatak odası{held} yaklaşık {plain(abs(s['f_bdrms']), 2)} bin "
                     f"dolar daha {'yüksek' if s['f_bdrms'] > 0 else 'düşük'} fiyatla ilişkilidir.")
    if "lotsize" in chosen:
        parts.append(f"1.000 kare fit daha büyük arsa{held} yaklaşık {plain(abs(s['bin_fit_arsa']), 2)} bin dolar daha "
                     f"{'yüksek' if s['bin_fit_arsa'] > 0 else 'düşük'} fiyatla ilişkilidir.")
    text = " ".join(parts)
    if {"sqrft", "bdrms"} <= set(chosen):
        text += (" Yatak odası katsayısı “daha büyük ev” karşılaştırması değildir: büyüklüğü aynı evler arasında oda "
                 "sayısı farkıdır.")
    if "bdrms" in chosen:
        text += (f" A ile B yalnız yatak odası sayısında farklıdır: tahmin farkı {plain(s['konut_fark'], 2)} bin dolar, "
                 "yani yatak odası katsayısıdır.")
    else:
        text += " A ile B yalnız yatak odası sayısında farklıdır; yatak odası modelde olmadığı için iki tahmin aynıdır."
    differing = [name for name in ("sqrft", "lotsize") if name in chosen]
    if len(differing) == 2:
        text += " B ile C arasında büyüklük ve arsa da değiştiği için fark birden fazla katsayının katkısıdır."
    elif differing:
        owner = {"sqrft": "konut büyüklüğünün", "lotsize": "arsa büyüklüğünün"}[differing[0]]
        other = {"sqrft": "arsa büyüklüğü", "lotsize": "konut büyüklüğü"}[differing[0]]
        text += (f" B ile C arasındaki tahmin farkı yalnız {owner} katkısıdır: {other} de iki konutta farklıdır ama "
                 "modelde olmadığı için tahmine girmez" + ("; yatak odası sayısı iki konutta aynıdır." if "bdrms" in chosen
                                                         else "."))
    else:
        text += " B ile C yalnız büyüklük ve arsada farklıdır; ikisi de modelde olmadığı için tahminleri aynıdır."
    return text + (" Şekilde noktalar 45 derece çizgisine yaklaştıkça tahmin gözlenen fiyata yaklaşır; çizgiden uzaklık "
                   "artıktır. Konum ve yapı kalitesi gibi modelde bulunmayan özellikler artıkta kalır.")


# --- Adım 8: çoklu R² ve düzeltilmiş R² ---------------------------------------------------

EXERCISE = (("Model A (k = 1)", 0.40, 1), ("Model B (k = 3)", 0.43, 3), ("Model C (k = 8)", 0.44, 8))
"""§5.9'daki egzersiz tablosu: aynı bağımlı değişken, R² ve açıklayıcı değişken sayısı k."""


def _adjusted(r2: float, n: int, k: int) -> E.Expr:
    """Denklem 5.8: 1 − (1 − R²)(n − 1)/(n − k − 1)."""

    return E.sub(1, E.mul(E.sub(1, r2), E.div(n - 1, n - k - 1)))


def _fit(choices) -> tuple:
    chosen = choices["adim1_x"]
    notes = tuple(chosen) == NOTES_X
    n = int(choices["adim8_n"])
    k = len(chosen)
    hkt_per_df = E.div(E.ref("hkt_coklu"), E.sub(E.ref("n_coklu"), k + 1))
    tkt_per_df = E.div(E.ref("tkt"), E.sub(E.ref("n_coklu"), 1))
    label = "Çoklu model" if notes else "Seçtiğiniz model"
    extra = () if notes else (
        ModelValue("r2_notlar", "coklu_notlar", "r2", "Notlardaki model: R²"),
        ModelValue("r2d_notlar", "coklu_notlar", "adj_r2", "Notlardaki model: düzeltilmiş R²"),
    )
    notes_r2 = () if notes else (("Notlardaki model", E.ref("r2_notlar")),)
    notes_r2d = () if notes else (("Notlardaki model", E.ref("r2d_notlar")),)
    return (
        ModelValue("r2_basit", "basit", "r2", "Basit model: R²"),
        ModelValue("r2d_basit", "basit", "adj_r2", "Basit model: düzeltilmiş R²"),
        ModelValue("r2_coklu", "coklu", "r2", f"{label}: R²"),
        ModelValue("r2d_coklu", "coklu", "adj_r2", f"{label}: düzeltilmiş R²"),
        *extra,
        ModelValue("hkt_coklu", "coklu", "ssr", f"{label}: HKT = Σûᵢ²", decimals=3),
        ModelValue("n_coklu", "coklu", "nobs", "Gözlem sayısı n", decimals=0),
        Statistic(DATA, OUTCOME, "mean", "ybar", "Ortalama ücret Ȳ", decimals=6),
        Derive(DATA, "sapma_kare", E.power(E.sub(E.var(OUTCOME), E.ref("ybar")), 2), "(Yᵢ − Ȳ)²"),
        Statistic(DATA, "sapma_kare", "sum", "tkt", "TKT = Σ(Yᵢ − Ȳ)²", decimals=3),
        Scalar("r2d_elle", E.sub(1, E.div(hkt_per_df, tkt_per_df)),
               f"Denklem 5.8 ile: 1 − [HKT/(n − {k} − 1)] / [TKT/(n − 1)]", decimals=4),
        ScalarTable((("Basit model", E.ref("r2_basit")), (label, E.ref("r2_coklu")), *notes_r2), "uyum_r2",
                    decimals=4),
        ScalarTable((("Basit model", E.ref("r2d_basit")), (label, E.ref("r2d_coklu")), *notes_r2d), "uyum_r2d",
                    decimals=4),
        JoinColumns("uyum", (("R²", "uyum_r2", "deger"), ("Düzeltilmiş R²", "uyum_r2d", "deger")), decimals=4,
                    heading="Model"),
        GroupedBarChart("uyum", "Model", "Uyum ölçüsü",
                        "WAGE1: basit ve çoklu modelde R² ve düzeltilmiş R²" if notes
                        else "WAGE1: basit model, seçtiğiniz model ve notlardaki model", series="sutun", decimals=4),
        ScalarTable(tuple((name, _adjusted(r2, n, size)) for name, r2, size in EXERCISE), "abc", decimals=3,
                    heading=f"§5.9 egzersizindeki model (n = {n})", value="Düzeltilmiş R²"),
    )


def _fit_note(state, choices) -> str:
    chosen = choices["adim1_x"]
    n = int(choices["adim8_n"])
    s = state.scalars
    table = state.tables["abc"]["deger"]
    best = table.idxmax()
    label = "çoklu model" if tuple(chosen) == NOTES_X else "seçtiğiniz model"
    text = (f"Basit model: R² = {plain(s['r2_basit'], 4)}, düzeltilmiş R² = {plain(s['r2d_basit'], 4)}; {label}: "
            f"{plain(s['r2_coklu'], 4)} ve {plain(s['r2d_coklu'], 4)}. Denklem 5.8 ile elde hesaplanan değer "
            f"{plain(s['r2d_elle'], 4)} yazılımınkiyle aynıdır.")
    if tuple(chosen) != NOTES_X:
        text += (f" Notlardaki model (eğitim, deneyim, kıdem): {plain(s['r2_notlar'], 4)} ve "
                 f"{plain(s['r2d_notlar'], 4)}.")
    if tuple(chosen) == ("educ",):
        text += " Seçtiğiniz model basit modelin kendisidir; iki model aynı sayıları verir."
    text += (" Yeni açıklayıcı değişken R²'yi azaltmaz (EKK yeni katsayıyı sıfır seçerek eski modeli koruyabilir); "
             "düzeltilmiş R² her yeni katsayının serbestlik derecesi maliyetini hesaba katar ve düşebilir.")
    if "educ" not in chosen:
        text += (" Seçtiğiniz modelde eğitim yok: iki model iç içe değildir (biri diğerine değişken eklenerek elde "
                 "edilmez), bu yüzden kural aralarında geçerli olmaz.")
    return text + (f" Son tabloda (n = {n}) düzeltilmiş R² en yüksek olan: {best}. n büyüdükçe ceza küçülür. İki ölçü de "
                   "yalnız aynı bağımlı değişken ve aynı örneklemde karşılaştırılır; yüksek değer doğru ekonomik model "
                   "veya nedensellik garantisi değildir.")


# --- Adım 9: makale tablosu -----------------------------------------------------------------

def _article(choices) -> tuple:
    selected = choices["adim9_sutunlar"]
    columns = []
    operations: list = []
    for number, key in enumerate(selected, start=1):
        heading, outcome, regressors, _ = ARTICLE[key]
        operations.append(OLS(f"makale_{key}", DATA, outcome, regressors,
                              f"Sütun ({number}): {outcome} ~ {' + '.join(regressors)}"))
        columns.append((f"({number}) {heading}", f"makale_{key}"))
    used = {name for key in selected for name in ARTICLE[key][2]}
    title = ("Tablo 5.4: ücret belirleyicilerine ilişkin makale tablosu" if tuple(selected) == ARTICLE_DEFAULT
             else "Seçtiğiniz modellerle makale tipi tablo")
    operations.append(RegressionTable(tuple(columns), (*_ordered(used), INTERCEPT), "makale", title, stars=False,
                                      decimals=3, standard_errors=False, adj_r2=True))
    return tuple(operations)


def _article_note(state, choices) -> str:
    selected = choices["adim9_sutunlar"]
    groups: dict[str, list[int]] = {}
    for number, key in enumerate(selected, start=1):
        groups.setdefault(ARTICLE[key][0], []).append(number)
    text = ("Okuma sırası: her sütunun bağımlı değişkeni, gözlem sayısı, sütundaki açıklayıcı değişkenler, katsayının "
            "birimi (düzey ücrette dolar, log ücrette yaklaşık yüzde) ve ceteris paribus koşulu; sonra R² ve düzeltilmiş "
            "R².")
    if len(groups) > 1:
        owners = {"Ücret": "ücretin", "ln(Ücret)": "log ücretin"}
        parts = ", ".join(f"{_listing([f'({n})' for n in numbers])} {owners[name]}" for name, numbers in groups.items())
        text += (f" R²'ler yalnız aynı bağımlı değişkenli sütunlar arasında karşılaştırılır: {parts} değişkenliğini "
                 "özetler.")
    elif len(selected) > 1:
        text += " Sütunların bağımlı değişkeni ve örneklemi (526 çalışan) aynıdır: R²'leri doğrudan karşılaştırılabilir."
    return text + (" Makalelerde kontroller bazen tek tek yazılmaz; “Kontroller: Evet” satırı hangi değişkenlerin "
                   "kontrol edildiğini söylemez, tablo notunda açıklanmalıdır (Tablo 5.4). Anlamlılık yıldızları ve "
                   "parantez içindeki standart hatalar Konu 7'de okunur.")


# --- Tanım ---------------------------------------------------------------------------

_KOD52 = (
    (INTERCEPT, (-2.8727, 0.729, -3.941, 0.000, -4.305, -1.441)),
    ("educ", (0.5990, 0.051, 11.679, 0.000, 0.498, 0.700)),
    ("exper", (0.0223, 0.012, 1.853, 0.064, -0.001, 0.046)),
    ("tenure", (0.1693, 0.022, 7.820, 0.000, 0.127, 0.212)),
)
"""Kod 5.2: katsayı, standart hata, t, p, %95 GA alt ve üst sınır (basılı basamakla)."""
_TERM_NAMES = {INTERCEPT: "sabit terim", "educ": "eğitim", "exper": "deneyim", "tenure": "kıdem"}
_QUANTITY_NAMES = {"coef": "katsayı", "se": "standart hata", "t": "t", "p": "p-değeri", "ci_low": "%95 GA alt",
                   "ci_high": "%95 GA üst"}
_TABLE51 = (
    ("(1) Ücret", {"educ": 0.5414, INTERCEPT: -0.9049, "n": 526, "r2": 0.1648, "adj_r2": 0.1632}),
    ("(2) Ücret", {"educ": 0.5990, "exper": 0.0223, "tenure": 0.1693, INTERCEPT: -2.8727, "n": 526, "r2": 0.3064,
                   "adj_r2": 0.3024}),
    ("(3) ln(Ücret)", {"educ": 0.0920, "exper": 0.0041, "tenure": 0.0221, INTERCEPT: 0.2844, "n": 526, "r2": 0.3160,
                       "adj_r2": 0.3121}),
)
_TABLE54 = (
    ("(1) Ücret", {"educ": 0.541, INTERCEPT: -0.905, "n": 526, "r2": 0.165, "adj_r2": 0.163}),
    ("(2) Ücret", {"educ": 0.599, "exper": 0.022, "tenure": 0.169, INTERCEPT: -2.873, "n": 526, "r2": 0.306,
                   "adj_r2": 0.302}),
    ("(3) ln(Ücret)", {"educ": 0.092, "exper": 0.004, "tenure": 0.022, INTERCEPT: 0.284, "n": 526, "r2": 0.316,
                       "adj_r2": 0.312}),
)
_TABLE52 = (
    ("(1) Fiyat", {"sqrft": 0.1402, INTERCEPT: 11.2041, "n": 88, "r2": 0.6208, "adj_r2": 0.6164}),
    ("(2) Fiyat", {"sqrft": 0.1228, "bdrms": 13.8525, "lotsize": 0.002068, INTERCEPT: -21.7703, "n": 88,
                   "r2": 0.6724, "adj_r2": 0.6607}),
)
_ROW_NAMES = {"n": "gözlem sayısı", "r2": "R²", "adj_r2": "düzeltilmiş R²", INTERCEPT: "sabit",
              "educ": "eğitim", "exper": "deneyim", "tenure": "kıdem", "sqrft": "konut büyüklüğü",
              "bdrms": "yatak odası", "lotsize": "arsa büyüklüğü"}


def _table_checks(table: str, label: str, cells, decimals: int) -> tuple[Check, ...]:
    """Bir makale tablosunun bütün hücreleri; basamak: gözlem sayısı 0, arsa katsayısı 6, diğerleri ``decimals``."""

    checks = []
    for heading, values in cells:
        for row, value in values.items():
            digits = 0 if row == "n" else 6 if row == "lotsize" else decimals
            checks.append(Check(f"{label}: {heading}, {_ROW_NAMES[row]}", TableTarget(table, row, heading), value,
                                digits))
    return tuple(checks)


STEPS = (
    interactive_step(
        number=1,
        title="Basit modelden çoklu modele: katsayıların ceteris paribus yorumu",
        note=NoteRef("5.3", 0, ("§5.1", "Denklem 5.5")),
        explanation=(
            "Basit regresyon ($\\widehat{\\text{ücret}} = -0{,}9049 + 0{,}5414\\,\\text{eğitim}$, §5.1) eğitim düzeyi "
            "farklı çalışanları, diğer farkları ayırmadan karşılaştırır. Çoklu regresyonda her eğim katsayısı, "
            "modeldeki diğer açıklayıcı değişkenler aynıyken yorumlanır (Denklem 5.5). Modelin açıklayıcı "
            "değişkenlerini seçin: tablo basit modeli ve seçtiğiniz çoklu modeli yan yana gösterir. Adım 2, 3, 5, 6 "
            "ve 8 bu modeli kullanır."
        ),
        controls=(X_MULTI,),
        build=_models,
        checks=(
            Check("§5.1: basit model, sabit terim", CoefTarget("basit", INTERCEPT), -0.9049, 4),
            Check("§5.1: basit model, eğitim katsayısı", CoefTarget("basit", "educ"), 0.5414, 4),
            _scalar("basit_egitim", 0.541, "§5.1: yaklaşık 0,541 dolar", 3),
            Check("Denklem 5.5: sabit terim", CoefTarget("coklu", INTERCEPT), -2.8727, 4),
            Check("Denklem 5.5: eğitim katsayısı", CoefTarget("coklu", "educ"), 0.5990, 4),
            Check("Denklem 5.5: deneyim katsayısı", CoefTarget("coklu", "exper"), 0.0223, 4),
            Check("Denklem 5.5: kıdem katsayısı", CoefTarget("coklu", "tenure"), 0.1693, 4),
            _scalar("b_educ", 0.599, "§5.3: eğitim yorumu, yaklaşık 0,599 dolar", 3),
            _scalar("b_exper", 0.022, "§5.3: deneyim yorumu, yaklaşık 0,022 dolar", 3),
            _scalar("b_tenure", 0.169, "§5.3: kıdem yorumu, yaklaşık 0,169 dolar", 3),
        ),
        note_for=lambda state, choices: _models_note(state, choices),
    ),
    interactive_step(
        number=2,
        title="Birden fazla birimlik fark ve iki çalışanın karşılaştırılması",
        note=NoteRef("5.3", 0, ("Şekil 5.1",)),
        explanation=(
            "Bir açıklayıcı değişken birden fazla birim farklıysa, diğer değişkenler aynıyken tahmin edilen fark "
            "katsayı ile farkın çarpımıdır: notlarda $4(0{,}5990) = 2{,}396$ dolar. A çalışanının eğitimi 12, "
            "deneyimi 10, kıdemi 5 yıl; B çalışanının eğitimi daha yüksek, deneyimi ve kıdemi aynıdır. Eğitim farkını "
            "değiştirin. Şekil 5.1 kıdem sabitken üç deneyim düzeyinde eğitim–ücret tahmin çizgilerini gösterir. Bu "
            "adım Adım 1'deki modeli kullanır."
        ),
        uses=(X_MULTI,),
        controls=(NumberChoice("adim2_fark", "Eğitim farkı (yıl): B'nin eğitimi A'nınkinden bu kadar fazla", 1, 6, 4,
                               1, help="Notlarda 4 yıl: A 12, B 16 yıl eğitimli.", integer=True, decimals=0),),
        build=_comparison,
        checks=(
            _scalar("egitim_farki", 2.396, "§5.3: 4(0,5990)", 3),
            _scalar("tahmin_A", 5.38, "§5.3: A çalışanının tahmini", 2),
            _scalar("tahmin_B", 7.78, "§5.3: B çalışanının tahmini", 2),
            _scalar("fark_BA", 2.396, "§5.3: B − A = 4β̂ (eğitim)", 3),
            _scalar("fark_BA", 2.40, "§5.3: yaklaşık 2,40 dolar", 2),
        ),
        note_for=lambda state, choices: _comparison_note(state, choices),
    ),
    interactive_step(
        number=3,
        title="Tahmin edilen değer ve artık",
        note=NoteRef("5.4", 0, ("Denklem 5.6",)),
        explanation=(
            "Bir çalışan için tahmin edilen değer $\\widehat Y = \\hat\\beta_0 + \\hat\\beta_1 X_1 + \\cdots + "
            "\\hat\\beta_k X_k$, artık $\\hat u = Y - \\widehat Y$'dir. Çoklu EKK, kareli artıklar toplamını en küçük "
            "yapan katsayıları birlikte seçer (Denklem 5.6). Notlardaki çalışanın eğitimi 16, deneyimi 20, kıdemi 10 "
            "yıl; gözlenen ücreti 10 dolardır. Değerleri değiştirin. Bu adım Adım 1'deki modeli kullanır."
        ),
        uses=(X_MULTI,),
        controls=(
            NumberChoice("adim3_egitim", "Eğitim (yıl)", 0, 18, 16, 1, help="Notlarda 16.", integer=True, decimals=0),
            NumberChoice("adim3_deneyim", "Potansiyel deneyim (yıl)", 0, 50, 20, 1, help="Notlarda 20.", integer=True,
                         decimals=0),
            NumberChoice("adim3_kidem", "Kıdem (yıl)", 0, 40, 10, 1, help="Notlarda 10.", integer=True, decimals=0),
            NumberChoice("adim3_ucret", "Gözlenen saatlik ücret (dolar)", 0.5, 25.0, 10.0, 0.5, help="Notlarda 10.",
                         decimals=1),
        ),
        build=_fitted,
        checks=(
            _scalar("birey_tahmin", 8.850, "§5.4: tahmin edilen ücret", 3),
            _scalar("birey_artik", 1.150, "§5.4: artık 10 − 8,850", 3),
        ),
        note_for=lambda state, choices: _fitted_note(state, choices),
    ),
    interactive_step(
        number=4,
        title="“Sabit tutma”: kontrollerin doğrusal katkısını ayırmak",
        note=NoteRef("5.5", 0, ("Şekil 5.2",)),
        explanation=(
            "Üç adım: (1) ücreti kontrollere göre tahmin edip artığını al; (2) incelenen değişkeni aynı kontrollere göre "
            "tahmin edip artığını al; (3) birinci artıkları ikinci artıklara göre regresyona tabi tut. Üçüncü adımdaki "
            "eğim çoklu regresyondaki katsayıyla aynıdır (§5.5). Notlarda incelenen değişken eğitim, kontroller deneyim "
            "ve kıdemdir; değişkeni değiştirin."
        ),
        controls=(Choice("adim4_x", "Kısmi ilişkisi incelenen değişken (diğer ikisi kontroldür)", PARTIAL_OPTIONS,
                         "educ", help="Notlarda eğitim."),),
        build=_partial,
        checks=(
            _scalar("kismi_egim", 0.598965, "§5.5: artıklar regresyonunun eğimi", 6),
            _scalar("tam_katsayi", 0.598965, "§5.5: çoklu regresyondaki eğitim katsayısı", 6),
            _scalar("kismi_egim", 0.599, "§5.5: doğrunun eğimi 0,599", 3),
        ),
        note_for=lambda state, choices: _partial_note(state, choices),
    ),
    interactive_step(
        number=5,
        title="Python çıktısı ve basit–çoklu model tablosu",
        note=NoteRef("5.6", 0, ("Kod 5.1", "Kod 5.2", "Tablo 5.1")),
        explanation=(
            "Kod 5.1 çoklu modeli tahmin eder: formüldeki `+` işaretleri açıklayıcı değişkenleri birlikte kullanır. "
            "Bu aşamada çıktıda okunanlar: bağımlı değişken, gözlem sayısı, katsayılar, R² ve düzeltilmiş R². "
            "Tablo 5.1 basit modeli, çoklu modeli ve bağımlı değişkeni log ücret olan çoklu modeli yan yana koyar. Bu "
            "adım Adım 1'deki modeli kullanır."
        ),
        uses=(X_MULTI,),
        build=_output,
        checks=(
            *(Check(f"Kod 5.2: {_TERM_NAMES[term]}, {_QUANTITY_NAMES[quantity]}", CoefTarget("coklu", term, quantity),
                    value, 4 if quantity == "coef" else 3)
              for term, values in _KOD52 for quantity, value in zip(COEF_QUANTITIES, values)),
            Check("Kod 5.2: gözlem sayısı", ModelTarget("coklu", "nobs"), 526, 0),
            _scalar("r2_kod", 0.306, "Kod 5.2: R²", 3),
            _scalar("r2d_kod", 0.302, "Kod 5.2: düzeltilmiş R²", 3),
            _scalar("r2_yuzde", 30.6, "§5.6: yaklaşık yüzde 30,6", 1),
            *_table_checks("tablo51", "Tablo 5.1", _TABLE51, 4),
            _scalar("log_yuzde", 9.2, "§5.6: log ücret sütunu, yaklaşık yüzde 9,2", 1),
        ),
        note_for=lambda state, choices: _output_note(state, choices),
    ),
    interactive_step(
        number=6,
        title="Bir değişken eklendiğinde katsayı neden değişir?",
        note=NoteRef("5.7", 0, ("Şekil 5.3",)),
        explanation=(
            "Aynı değişkenin katsayısı basit ve çoklu modelde farklı olabilir; bu yazılım hatası değildir. Basit "
            "katsayı, değişkenle birlikte hareket eden diğer farkları da taşır; çoklu katsayı, modeldeki diğer "
            "değişkenlerin doğrusal katkısı ayrıldıktan sonraki ilişkidir. Notlarda eğitim katsayısı 0,541'den "
            "0,599'a yükselir. Grafik, modeldeki her değişken için iki katsayıyı yan yana gösterir. Bu adım Adım 1'deki "
            "modeli kullanır."
        ),
        uses=(X_MULTI,),
        build=_change,
        checks=(
            Check("Şekil 5.3: basit model, eğitim katsayısı", TableTarget("katsayilar", "Eğitim", "Basit model"), 0.541,
                  3),
            Check("Şekil 5.3: çoklu model, eğitim katsayısı", TableTarget("katsayilar", "Eğitim", "Çoklu model"), 0.599,
                  3),
        ),
        note_for=lambda state, choices: _change_note(state, choices),
    ),
    interactive_step(
        number=7,
        title="HPRICE1: konut fiyatında birden fazla özellik",
        note=NoteRef("5.8", 0, ("Denklem 5.7", "Tablo 5.2", "Tablo 5.3", "Şekil 5.4")),
        explanation=(
            "Fiyat bin dolar, konut ve arsa büyüklüğü kare fit cinsindedir. Çoklu model konut büyüklüğü, yatak odası "
            "sayısı ve arsa büyüklüğünü birlikte kullanır (Denklem 5.7). Tablo 5.3'teki üç konut için tahmin üretilir; "
            "şekil gözlenen ve tahmin edilen fiyatları karşılaştırır. Modelin açıklayıcı değişkenlerini seçin."
        ),
        controls=(MultiChoice("adim7_x", "Konut fiyatı modelinin açıklayıcı değişkenleri (bağımlı değişken: fiyat)",
                              W.options(HPRICE, HOUSE_OPTIONS), HOUSE_OPTIONS,
                              help="Notlardaki model: üç değişken birlikte (Denklem 5.7)."),),
        build=_houses,
        checks=(
            Check("§5.8: basit model, sabit terim", CoefTarget("fiyat_basit", INTERCEPT), 11.2041, 4),
            Check("§5.8: basit model, konut büyüklüğü", CoefTarget("fiyat_basit", "sqrft"), 0.1402, 4),
            Check("Denklem 5.7: sabit terim", CoefTarget("fiyat_coklu", INTERCEPT), -21.7703, 4),
            Check("Denklem 5.7: konut büyüklüğü", CoefTarget("fiyat_coklu", "sqrft"), 0.1228, 4),
            Check("Denklem 5.7: yatak odası sayısı", CoefTarget("fiyat_coklu", "bdrms"), 13.8525, 4),
            Check("Denklem 5.7: arsa büyüklüğü", CoefTarget("fiyat_coklu", "lotsize"), 0.002068, 6),
            *_table_checks("tablo52", "Tablo 5.2", _TABLE52, 4),
            _scalar("yuz_fit", 12.28, "§5.8: 100(0,1228) = 12,28 bin dolar", 2),
            _scalar("f_bdrms", 13.85, "§5.8: bir ek yatak odası, 13,85 bin dolar", 2),
            _scalar("bin_fit_arsa", 2.07, "§5.8: 1000(0,002068) = 2,07 bin dolar", 2),
            Check("Tablo 5.3: A konutu", CellTarget("konutlar", "tahmin", 1), 279.82, 2),
            Check("Tablo 5.3: B konutu", CellTarget("konutlar", "tahmin", 2), 293.67, 2),
            Check("Tablo 5.3: C konutu", CellTarget("konutlar", "tahmin", 3), 326.43, 2),
            _scalar("konut_fark", 13.85, "§5.8: 293,67 − 279,82", 2),
        ),
        note_for=lambda state, choices: _houses_note(state, choices),
    ),
    interactive_step(
        number=8,
        title="Çoklu R² ve düzeltilmiş R²",
        note=NoteRef("5.9", 0, ("Denklem 5.8",)),
        explanation=(
            "$R^2 = 1 - \\text{HKT}/\\text{TKT}$ çoklu modelde de aynıdır. Yeni değişken eklemek R²'yi düşürmez; "
            "düzeltilmiş R² açıklayıcı değişken sayısını hesaba katar: $\\bar R^2 = 1 - "
            "\\frac{\\text{HKT}/(n-k-1)}{\\text{TKT}/(n-1)}$ (Denklem 5.8). Son tablo §5.9'daki üç modeli "
            "(R² 0,40; 0,43; 0,44; k = 1, 3, 8) seçtiğiniz gözlem sayısıyla hesaplar; notlarda n = 200. Bu adım Adım "
            "1'deki modeli kullanır."
        ),
        uses=(X_MULTI,),
        controls=(NumberChoice("adim8_n", "Gözlem sayısı n (A, B ve C modelleri)", 20, 2000, 200, 10,
                               help="Notlardaki egzersizde n = 200.", integer=True, decimals=0),),
        build=_fit,
        checks=(
            _scalar("r2_basit", 0.1648, "§5.9: basit model, R²", 4),
            _scalar("r2d_basit", 0.1632, "§5.9: basit model, düzeltilmiş R²", 4),
            _scalar("r2_coklu", 0.3064, "§5.9: çoklu model, R²", 4),
            _scalar("r2d_coklu", 0.3024, "§5.9: çoklu model, düzeltilmiş R²", 4),
            _scalar("r2d_elle", 0.3024, "Denklem 5.8 ile elde hesap", 4),
            *(Check(f"§5.9: {label}, düzeltilmiş R²", TableTarget("abc", label, "deger"), value, 3)
              for (label, _, _), value in zip(EXERCISE, (0.397, 0.421, 0.417))),
        ),
        note_for=lambda state, choices: _fit_note(state, choices),
    ),
    interactive_step(
        number=9,
        title="Python çıktısı ile makale tablosunu sistematik okumak",
        note=NoteRef("5.10", 0, ("Tablo 5.4",)),
        explanation=(
            "Makale tablosu yazılım çıktısını sıkıştırır: her sütun bir model, satırlar katsayılar; altta gözlem sayısı, "
            "R² ve düzeltilmiş R². Okuma sırası aynıdır: bağımlı değişken, gözlem sayısı, sütundaki değişkenler, "
            "katsayının birimi ve ceteris paribus koşulu. Notlardaki Tablo 5.4 üç sütunludur; sütunları siz seçin."
        ),
        controls=(MultiChoice("adim9_sutunlar", "Tablodaki modeller",
                              tuple((key, name) for key, (*_, name) in ARTICLE.items()), ARTICLE_DEFAULT,
                              help="Notlardaki Tablo 5.4: ilk üç model.", maximum=4),),
        build=_article,
        checks=_table_checks("makale", "Tablo 5.4", _TABLE54, 3),
        note_for=lambda state, choices: _article_note(state, choices),
    ),
    LabStep(
        number=10,
        title="Kontrol değişkeni seçimi: sorulacak altı soru",
        note=NoteRef("5.11"),
        explanation=(
            "Bir değişkeni kontrol olarak eklemeden önce sorulacaklar (§5.11):\n\n1. Değişken bağımlı değişkenle ekonomik "
            "olarak neden ilişkili olabilir?\n2. Temel açıklayıcı değişkenle de ilişkili olabilir mi?\n3. Temel "
            "açıklayıcı değişkenden önce mi, sonra mı belirlenmektedir?\n4. Doğru ölçülmüş müdür?\n5. Onu sabit tutmak "
            "cevaplamak istediğimiz karşılaştırmaya uygun mudur?\n6. Aynı değişkenin farklı tanımları veya gereksiz "
            "tekrarları modele eklenmiş midir?\n\nÖrneğin eğitimin ücretle ilişkisinde deneyim ve kıdem makul "
            "kontrollerdir; eğitimden sonra seçilen mesleği kontrol etmek ise sorunun bir bölümünü ortadan kaldırabilir. "
            "Tek bir evrensel değişken listesi yoktur: araştırma sorusu hangi karşılaştırmanın hedeflendiğini belirler. "
            "Çok sayıda ve birbirine çok yakın değişken katsayıların ayrı ayrı yorumunu zorlaştırır; bu, Konu 6'daki "
            "çoklu doğrusal bağlantı konusudur."
        ),
    ),
)


KONU05_LAB = LabSpec(
    topic_key="konu05",
    title="Uygulama: Çoklu Regresyon Modeli ve Ceteris Paribus Yorumu",
    note_section="5",
    steps=STEPS,
    labels=(
        *W.labels(DATA, HPRICE),
        (INTERCEPT, "Sabit terim"),
        ("calisan", "Çalışan"),
        ("konut", "Konut"),
        ("tahmin", "Tahmin edilen değer Ŷᵢ"),
        ("artik", "Artık ûᵢ"),
        ("ucret_artik", "Arındırılmış ücret"),
        ("x_artik", "Arındırılmış açıklayıcı değişken"),
        ("fiyat_tahmin", "Tahmin edilen fiyat (bin ABD doları)"),
        ("sapma_kare", "(Yᵢ − Ȳ)²"),
        *((f"tahmin_{level}", f"Deneyim = {level} yıl") for level in EXPER_LEVELS),
    ),
    consistency_notes=(
        "Veri notlardaki gibi WAGE1 ve HPRICE1'dir; notlardaki kod (Kod 5.1), betikler, uygulama ve üretilen kod onları "
        "wooldridge paketinden okur.",
        "§5.3'te iki çalışanın tahminleri iki basamakla yazılır (5,38 ve 7,78): yuvarlanmış katsayılarla elle hesap "
        "B için 7,781, tam duyarlık 7,780 verir; iki basamakta ikisi de 7,78'dir. Fark 4(0,5990) = 2,396'dır.",
        "Bütünleşik uygulamada (§5.12) düzeltilmiş R², n = 300, k = 3 ve R² = 0,58 ile 0,576'dır (önceki metin 0,575).",
        "Ücret yorumlarında birim notlardaki gibi dolardır (önceki sunumda 'birim').",
    ),
)
