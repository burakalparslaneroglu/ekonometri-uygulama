"""Konu 6 genel uygulaması: eksik değişken yanlılığı ve çoklu doğrusal bağlantı.

Notlardaki altı adım (``core.labs.konu06``) aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde
yazılır: kısa ve uzun model, yardımcı regresyon ve katsayının örneklem ayrıştırması, kontroller eklendikçe temel
katsayı ve yazılım çıktısı, tam çoklu doğrusal bağlantı, ikinci bir veride VIF ve bütünleşik okuma listesi.

Alternatif örnekte ücret adımları WAGE2'dir (935 erkek çalışan, 1980; aylık kazanç, ABD doları); dışarıda kalan
değişkenin varsayılanı IQ puanıdır (yeteneğin bir ölçüsü: eğitimin kazançla ilişkisindeki yetenek yanlılığı, Wooldridge
C3.6'nın düzey kazançtaki karşılığı). VIF adımı KIELMC'nin 1978 satışlarıdır (179 konut). "Kendi verini yükle"
seçeneğinde dışarıda kalan değişken ek sayısal değişkenlerden seçilir; ek değişken yoksa Adım 1–3 neye ihtiyacı
olduğunu yazar. VIF aynı dosyanın açıklayıcı değişkenleriyle hesaplanır.

Etkileşim notlardaki gibidir: kısa modelde dışarıda kalan değişken (Adım 1; Adım 2 aynı değişkeni kullanır), tablonun
sütunları (Adım 3) ve VIF hesabındaki açıklayıcı değişkenler (Adım 5). Yazılım çıktısındaki standart hata, t ve
p-değeri notlardaki gibi görünür; yorumları Konu 7'dedir.
"""

from __future__ import annotations

import itertools
from functools import cache

import numpy as np

from core.labs import expr as E
from core.labs.ornek import Case, TopicVariants, md, sayi, stable_checks, with_app_values
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
    house_case,
    labels_of,
    level_text,
    listing,
    negligible,
    nice,
    options,
    phrase,
    roles,
    sample_employed,
    second,
    short_unit,
    shown_decimals,
    wage2_case,
)
from core.labs.spec import (
    INTERCEPT,
    OLS,
    BarChart,
    CellTarget,
    Check,
    Choice,
    CoefTarget,
    Derive,
    InlineData,
    LabSpec,
    LabStep,
    ModelTarget,
    ModelValue,
    MultiChoice,
    NoteRef,
    PairStatistic,
    RegressionTable,
    Scalar,
    ScalarTable,
    ScalarTarget,
    ShowFrame,
    ShowModel,
    TableTarget,
    interactive_step,
)

TOPIC = "konu06"
TITLE = "Uygulama: EKK Varsayımları, Yansızlık ve Model Sorunları"
Z_KEY = "adim1_z"
ALT_OMITTED = ("IQ", "KWW", "exper", "tenure", "sibs")


def _check(label: str, target, decimals: int) -> Check:
    return Check(label, target, 0.0, decimals)


def _scalar(name: str, label: str, decimals: int) -> Check:
    return _check(label, ScalarTarget(name), decimals)


def _name(case: Case, column: str) -> str:
    return case.name(column) if case.own else phrase(case, column)


def _title(case: Case, column: str) -> str:
    return case.name(column) if case.own else capital(phrase(case, column))


def _omitted(case: Case) -> tuple[str, ...]:
    """Dışarıda kalan değişkenin seçenekleri: alternatif örnekte IQ, KWW, deneyim, kıdem, kardeş sayısı; kendi verinde
    ek sayısal değişkenler."""

    if "omitted" in case.extra:
        return tuple(case.extra["omitted"])
    x = case.roles[ACIKLAYICI]
    return tuple(name for name in candidates(case) if name != x)


def _unit_of(case: Case, column: str) -> str:
    return short_unit(case, column) or "birim"


# --- Adım 1: kısa ve uzun model -------------------------------------------------------------------------------

def z_choice(case: Case) -> Choice:
    names = _omitted(case)
    return Choice(Z_KEY, "Kısa modelde dışarıda kalan, uzun modele eklenen değişken", options(case, names), names[0],
                  help=f"Varsayılan: {case.name(names[0])}. Adım 2 aynı değişkeni kullanır.")


_LEAD1 = ("Sonuç yalnız temel açıklayıcı değişkene göre tahmin edildiğinde (kısa model) öteki değişkenler hata teriminde "
          "kalır. Bir değişken eklendiğinde (uzun model) temel katsayı değişir.")


def _digits(*values: float) -> int:
    """Katsayıların metindeki basamağı: dört, çok küçük katsayılarda en az üç anlamlı basamak."""

    return digits_for(max(abs(float(value)) for value in values), 4)


def _step1(case: Case, choice: Choice) -> LabStep:
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    default = choice.default

    def build(choices) -> tuple:
        z = choices[Z_KEY]
        same = z == default
        columns = [("(1) Kısa model", "kisa"), ("(2) Uzun model", "uzun")]
        extra: tuple = ()
        if not same:
            extra = (OLS("uzun_varsayilan", case.frame, y, (x, default),
                         f"Karşılaştırma için varsayılan uzun model: {y} ~ {x} + {default}"),)
            columns.append(("(3) Varsayılan uzun model", "uzun_varsayilan"))
        terms = (x, z, INTERCEPT) if same else (x, z, default, INTERCEPT)
        return (
            *case.load,
            OLS("kisa", case.frame, y, (x,), f"Kısa model: {y} ~ {x} (Denklem 6.10'un karşılığı)"),
            ModelValue("kisa_x", "kisa", "coef", f"Kısa model: {_name(case, x)} katsayısı", term=x),
            OLS("uzun", case.frame, y, (x, z), f"Uzun model: {y} ~ {x} + {z}"),
            ModelValue("uzun_x", "uzun", "coef", f"Uzun model: {_name(case, x)} katsayısı", term=x),
            ModelValue("uzun_z", "uzun", "coef", f"Uzun model: {_name(case, z)} katsayısı", term=z),
            *extra,
            RegressionTable(tuple(columns), terms, "kisa_uzun",
                            f"Kısa ve uzun model: {_name(case, z)} dışarıda / içeride (bağımlı değişken: {case.name(y)})"
                            if same else f"Kısa model, {_name(case, z)} eklenmiş uzun model ve varsayılan uzun model",
                            stars=False, decimals=4, standard_errors=False),
        )

    def note(state, choices) -> str:
        z = choices[Z_KEY]
        s = state.scalars
        gap = s["uzun_x"] - s["kisa_x"]
        d = _digits(s["kisa_x"], s["uzun_x"])
        text = (f"{capital(phrase(case, x))} katsayısı kısa modelde {sayi(s['kisa_x'], d)}, {phrase(case, z)} eklenince "
                f"{sayi(s['uzun_x'], d)}; fark {sayi(gap, d)}. Kısa modelde {phrase(case, z)} hata teriminde kalır. "
                "Eksik değişken yanlılığı için iki koşul birlikte gerekir: dışarıda kalan değişken sonuçla ilişkili "
                f"olmalı ve temel açıklayıcı değişkenle ilişkili olmalıdır (§6.6). Uzun modelde {phrase(case, z)} "
                f"katsayısı {sayi(s['uzun_z'], _digits(s['uzun_z']))}; temel değişkenle ilişkisini Adım 2'deki "
                "yardımcı regresyon verir. Katsayının değişmesi tek başına yanlılık kanıtı değildir.")
        if z != default:
            model = state.models["uzun_varsayilan"]
            text += (f" Varsayılan uzun modelde ({phrase(case, default)} eklenince) {phrase(case, x)} katsayısı "
                     f"{sayi(state.models['kisa'].params[x], 4)} → {sayi(model.params[x], 4)} (sütun 3).")
        return text

    return interactive_step(
        number=1,
        title="Kısa ve uzun model: dışarıda kalan değişken",
        note=NoteRef("6.8", 0, ("Denklem 6.10", "Denklem 6.11")),
        explanation=(f"{_LEAD1} {case.extra.get('omitted_text', '')}Kısa modelde dışarıda kalan değişkeni seçin; Adım 2 "
                     "değişimin örneklemdeki ayrıştırmasını gösterir."),
        controls=(choice,),
        build=build,
        checks=(
            _check("Kısa model: sabit terim", CoefTarget("kisa", INTERCEPT), 4),
            _check("Kısa model: temel katsayı", CoefTarget("kisa", x), 4),
            _check("Uzun model: sabit terim", CoefTarget("uzun", INTERCEPT), 4),
            _check("Uzun model: temel katsayı", CoefTarget("uzun", x), 4),
            _check(f"Uzun model: {case.name(default)} katsayısı", CoefTarget("uzun", default), 4),
            _scalar("kisa_x", "Kısa model, yorumdaki katsayı", 3),
            _scalar("uzun_x", "Uzun model, yorumdaki katsayı", 3),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 2: yardımcı regresyon ve ayrıştırma ------------------------------------------------------------------

def _step2(case: Case, choice: Choice) -> LabStep:
    x = case.roles[ACIKLAYICI]
    y_unit = short_unit(case, case.roles[SONUC])

    def build(choices) -> tuple:
        z = choices[Z_KEY]
        return (
            OLS("yardimci", case.frame, z, (x,), f"Yardımcı regresyon: {z} ~ {x} (Denklem 6.12'nin karşılığı)"),
            ModelValue("yardimci_sabit", "yardimci", "coef", "Yardımcı regresyon: sabit terim", term=INTERCEPT),
            ModelValue("yardimci_egim", "yardimci", "coef", f"Yardımcı eğim δ̂ ({_name(case, z)} ~ {_name(case, x)})",
                       term=x),
            Scalar("katki", E.mul(E.ref("uzun_z"), E.ref("yardimci_egim")),
                   f"Dışarıda kalan değişkenin katkısı: β̂ ({_name(case, z)}) × δ̂", decimals=4),
            Scalar("yeniden", E.add(E.ref("uzun_x"), E.ref("katki")), "Uzun modelin temel katsayısı + katkı",
                   decimals=4),
            ScalarTable((
                ("Uzun model: temel katsayı", E.ref("uzun_x")),
                (f"Katkı: β̂ ({_name(case, z)}) × δ̂", E.ref("katki")),
                ("Kısa model: temel katsayı", E.ref("kisa_x")),
            ), "ayristirma", decimals=4, heading="Bileşen", value="Katsayı"),
            BarChart("ayristirma", "deger", "Bileşen", f"Katsayı{' (' + y_unit + ')' if y_unit else ''}",
                     "Kısa model katsayısı = uzun model katsayısı + dışarıda kalan değişkenin katkısı", decimals=4),
        )

    def note(state, choices) -> str:
        z = choices[Z_KEY]
        s = state.scalars
        gamma, delta = s["uzun_z"], s["yardimci_egim"]
        flat_gamma = negligible(case, gamma, case.roles[SONUC], z)
        flat_delta = negligible(case, delta, z, x)
        same = gamma * delta > 0
        subject = str(case.extra.get("aux_subject", f"{capital(phrase(case, x))} değeri bir birim daha yüksek olan "
                                                    "gözlemlerde"))
        amount_text = f"{sayi(abs(delta), digits_for(delta, 3))} {_unit_of(case, z)}"
        d = _digits(s["kisa_x"], s["uzun_x"])
        text = f"{subject} {phrase(case, z)} örneklemde ortalama {change(delta, amount_text, flat_delta)}. "
        if flat_gamma or flat_delta:
            which = (f"uzun modelde {phrase(case, z)} katsayısı" if flat_gamma else "yardımcı eğim")
            text += (f"Katkı {sayi(s['katki'], d)}: {which} hesap hassasiyetinde sıfır olduğu için {phrase(case, z)} "
                     "değişkeninin dışarıda bırakılması temel katsayıyı değiştirmez (Tablo 6.4). ")
        else:
            text += (f"{capital(phrase(case, z))} katsayısı {'pozitif' if gamma > 0 else 'negatif'}, temel değişkenle "
                     f"ilişkisi {'pozitif' if delta > 0 else 'negatif'}: işaretler {'aynı' if same else 'farklı'}, katkı "
                     f"{sayi(s['katki'], d)}. İşaret tablosuna göre {phrase(case, z)} değişkeninin dışarıda bırakılması "
                     f"temel katsayıyı {'yukarı' if same else 'aşağı'} yönlü etkileyebilir (Tablo 6.4). ")
        return text + (f"Örneklemde ilişki tam olarak sağlanır: {sayi(s['kisa_x'], d)} = {sayi(s['uzun_x'], d)} + "
                       f"({sayi(s['katki'], d)}). Bu ayrıştırma katsayı değişiminin mekanizmasını gösterir; uzun modelin "
                       "nedensel model olduğunu kanıtlamaz: başka faktörler hâlâ hata teriminde olabilir.")

    return interactive_step(
        number=2,
        title="Yardımcı regresyon ve katsayının örneklem ayrıştırması",
        note=NoteRef("6.8", 0, ("Denklem 6.12", "Denklem 6.8")),
        explanation=(
            "Dışarıda kalan değişken temel açıklayıcı değişkene göre tahmin edilir (yardımcı regresyon, Denklem 6.12'nin "
            "karşılığı). Örneklem katsayıları arasında tam olarak kısa katsayı = uzun katsayı + (dışarıda kalanın "
            "katsayısı) × (yardımcı eğim) ilişkisi vardır; Denklem 6.8'deki $\\beta_1 + \\beta_2\\delta_1$ yapısının "
            "örneklem karşılığı. Bu adım Adım 1'deki değişkeni kullanır."
        ),
        uses=(choice,),
        build=build,
        checks=(
            _scalar("yardimci_sabit", "Yardımcı regresyon: sabit terim", 4),
            _scalar("yardimci_egim", "Yardımcı eğim", 4),
            _scalar("katki", "Katkı", 4),
            _scalar("yeniden", "Uzun katsayı + katkı = kısa katsayı", 4),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


def _need_extra(number: int, title: str, note: NoteRef, lead: str) -> LabStep:
    return LabStep(number=number, title=title, note=note,
                   explanation=f"{lead} Bu adım için en az bir ek sayısal değişken seçilmelidir (dışarıda kalan değişken "
                               "ek değişkenlerden seçilir).")


# --- Adım 3: kontroller eklendikçe temel katsayı -------------------------------------------------------------

def _columns(case: Case) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Tablonun seçilebilen sütunları: anahtar → (başlık, açıklayıcı değişkenler)."""

    if "columns" in case.extra:
        return dict(case.extra["columns"])
    x = case.roles[ACIKLAYICI]
    others = _omitted(case)
    found = {"yalniz": (f"Yalnız {case.name(x)}", (x,))}
    for index, name in enumerate(others, start=1):
        found[f"tek_{index}"] = (f"{case.name(x)} + {case.name(name)}", (x, name))
    if len(others) > 1:
        found["tam"] = ("Bütün değişkenler", (x, *others))
    return found


def _columns_default(case: Case, columns) -> tuple[str, ...]:
    if "columns_default" in case.extra:
        return tuple(case.extra["columns_default"])
    keys = list(columns)
    return tuple(keys[:2] + (["tam"] if "tam" in columns else []))


def _step3(case: Case, z_default: str) -> LabStep:
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    columns = _columns(case)
    default = _columns_default(case, columns)
    order = tuple(dict.fromkeys(name for _, names in columns.values() for name in names))
    exact = case.own and exact_multi(case, y, (x, z_default))

    def build(choices) -> tuple:
        selected = choices["adim3_modeller"]
        same = tuple(selected) == default
        used = {name for key in selected for name in columns[key][1]}
        terms = tuple(name for name in order if name in used)
        return (
            *(OLS(f"m_{key}", case.frame, y, columns[key][1], f"{columns[key][0]}: {y} ~ {' + '.join(columns[key][1])}")
              for key in selected),
            *(ModelValue(f"x_{key}", f"m_{key}", "coef", f"Temel katsayı: {columns[key][0]}", term=x) for key in selected),
            RegressionTable(tuple((columns[key][0], f"m_{key}") for key in selected), terms, "tablo66",
                            "Kontroller eklendikçe temel katsayı (Tablo 6.6'daki gibi)" if same
                            else "Seçtiğiniz modellerle temel katsayının değişimi", stars=False, decimals=4,
                            standard_errors=False),
            ScalarTable(tuple((columns[key][0], E.ref(f"x_{key}")) for key in selected), "x_katsayilari", decimals=3,
                        heading="Model", value="Temel katsayı"),
            BarChart("x_katsayilari", "deger", "Model", f"{capital(_name(case, x))} katsayısı",
                     "Farklı modellerde temel katsayı (Şekil 6.3'teki gibi)" if same
                     else "Seçtiğiniz modellerde temel katsayı", decimals=3),
            OLS("kod63", case.frame, y, (x, z_default), f"Kod 6.3'teki gibi: {y} ~ {x} + {z_default}"),
            ShowModel("kod63", "Uzun modelin temel çıktısı (Kod 6.3'teki gibi)", columns=("coef", "se", "t", "p"),
                      stats=("nobs", "r2"), stars=False),
        )

    def note(state, choices) -> str:
        selected = choices["adim3_modeller"]
        table = state.tables["x_katsayilari"]["deger"]
        d = digits_for(float(table.abs().max()), 3)
        path = " → ".join(sayi(value, d) for value in table)
        changes = []
        for before, after, old, new in zip(selected, selected[1:], table.iloc[:-1], table.iloc[1:]):
            added = [phrase(case, name) for name in columns[after][1] if name not in columns[before][1]]
            removed = [phrase(case, name) for name in columns[before][1] if name not in columns[after][1]]
            how = " ve ".join(([f"{listing(added)} eklenince"] if added else []) +
                              ([f"{listing(removed)} çıkarılınca"] if removed else []))
            shown_old, shown_new = round(float(old), d), round(float(new), d)  # yazılan basamakta karşılaştırılır
            verb = "yükselir" if shown_new > shown_old else "düşer" if shown_new < shown_old else "değişmez"
            changes.append(f"{how} {verb}")
        moves = "; ".join(change for change in changes if change.strip())
        start = (f"{capital(phrase(case, x))} katsayısı sütunlar boyunca: {path}." if len(selected) > 1
                 else f"{capital(phrase(case, x))} katsayısı: {path}.")
        return (start + (f" {capital(moves)}." if moves else "") +
                " Her sütun farklı bir karşılaştırma yapar; katsayı değişimi otomatik olarak “son model doğrudur” demek "
                "değildir. Hangi karşılaştırmanın uygun olduğu iktisadi gerekçeyle tartışılır. Kod 6.3'teki gibi çıktıda "
                "bu aşamada yalnız `coef`, bağımlı değişken, gözlem sayısı ve R² okunur; `std err`, `t` ve `P>|t|` Konu "
                "7'de işlenir." + ("" if len(selected) > 1 else " Karşılaştırma için en az iki model seçin.")
                + (EXACT_MULTI_NOTE if exact else ""))

    checks = []
    for key in default:
        heading, names = columns[key]
        checks += [_check(f"Tablo: {heading}, {case.name(name)}", TableTarget("tablo66", name, heading), 4)
                   for name in names]
        checks.append(_check(f"Tablo: {heading}, R²", TableTarget("tablo66", "r2", heading), 4))
        checks.append(_check(f"Grafik: {heading}", TableTarget("x_katsayilari", heading, "deger"), 3))
    checks += [
        _check(f"Kod 6.3 karşılığı: {case.name(term) if term != INTERCEPT else 'sabit terim'}, {label}",
               CoefTarget("kod63", term, quantity), decimals)
        for term in (INTERCEPT, x, z_default)
        for quantity, label, decimals in (("coef", "katsayı", 4), ("se", "standart hata", 3), ("t", "t", 3),
                                          ("p", "p-değeri", 3))
    ]
    checks += [_check("Kod 6.3 karşılığı: gözlem sayısı", ModelTarget("kod63", "nobs"), 0),
               _check("Kod 6.3 karşılığı: R²", ModelTarget("kod63", "r2"), 3)]
    return interactive_step(
        number=3,
        title="Kontroller eklendikçe temel katsayı ve Python çıktısı",
        note=NoteRef("6.8", 0, ("Tablo 6.6", "Şekil 6.3", "Kod 6.3")),
        explanation=(
            "Tablo temel açıklayıcı değişkenin katsayısını farklı kontrol kümeleriyle yan yana gösterir (Tablo 6.6'daki "
            "gibi). Tablonun sütunlarını seçin. Altta varsayılan uzun modelin temel çıktısı vardır (Kod 6.3'teki gibi); "
            "bu aşamada yalnız katsayılar, bağımlı değişken, gözlem sayısı ve R² okunur."
        ),
        controls=(MultiChoice("adim3_modeller", "Tablodaki modeller",
                              tuple((key, heading) for key, (heading, _) in columns.items()), default,
                              help="Varsayılan: " + ", ".join(columns[key][0] for key in default) + ".", maximum=4),),
        build=build,
        checks=stable_checks(tuple(checks), exact),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 4: tam çoklu doğrusal bağlantı ---------------------------------------------------------------------

def _combination(case: Case) -> float:
    """Çiftlerin birleşimi β₁ + 12β₂: sonucun temel açıklayıcıya göre basit regresyon eğimi, yuvarlanmış."""

    if "combination" in case.extra:
        return float(case.extra["combination"])
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    data = case.data[[y, x]].astype(float)
    slope = float(np.polyfit(data[x], data[y], 1)[0])
    return nice(slope, 2) or 1.0


def _step4(case: Case, load: bool = False) -> LabStep:
    """``load``: Adım 1–3 kurulmadıysa (kendi verinde ek değişken yok) veri bu adımda okunur."""

    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    total = _combination(case)
    pairs = ((total, 0.0), (0.0, total / 12), (total / 2, total / 24), (2 * total, -total / 12))
    digits = max(shown_decimals(total, 2), 0)
    data = case.data[[y, x]].astype(float)
    slope = float(np.polyfit(data[x], data[y], 1)[0])
    first, second = slope / 145, 12 * slope / 145
    months = str(case.extra.get("months_text", f"{capital(phrase(case, x))} ile onun 12 katı"))
    return LabStep(
        number=4,
        title="Tam çoklu doğrusal bağlantı: aynı bilgiyi iki kez vermek",
        note=NoteRef("6.9", 0, ("Denklem 6.13",)),
        explanation=(
            "Bir değişken her zaman diğerinin 12 katıysa (ör. yıllık ve aylık gelir) $\\beta_1 X + \\beta_2 (12X) = "
            "(\\beta_1 + 12\\beta_2) X$ olur: veri yalnız $\\beta_1 + 12\\beta_2$ birleşimini belirler. Tablodaki dört "
            f"çiftin hepsi aynı birleşimi ({level_text(total)}, temel modeldeki eğimin yuvarlanmışı) verir. {months} aynı "
            "bilgiyi taşır; korelasyonları 1'dir. İkisi birlikte modele konamaz: uygulama böyle bir modeli “tam doğrusal "
            "bağlantı” uyarısıyla reddeder; R bir katsayıyı NA raporlar. statsmodels ise hata vermeden bir ayrıştırma "
            f"raporlar ({sayi(first, 4)} ve {sayi(second, 4)}: {sayi(first, 4)} + 12 × {sayi(second, 4)} ≈ "
            f"{sayi(slope, 2)}) ve yalnız özetin sonunda tekil tasarım notu yazar: veri yalnız birleşimi belirler, "
            "ayrışım keyfîdir."
        ),
        operations=(
            *(case.load if load else ()),
            InlineData("ciftler", ("beta1", "beta2"), tuple((round(a, 6), round(b, 6)) for a, b in pairs),
                       "Aynı değişkenin iki ölçeğiyle kurulan modelde katsayı çiftleri (§6.9'daki gibi)"),
            Derive("ciftler", "birlesim", E.add(E.var("beta1"), E.mul(12, E.var("beta2"))), "β₁ + 12β₂"),
            ShowFrame("ciftler", ("beta1", "beta2", "birlesim"), "Dört çift, aynı birleşim"),
            Derive(case.frame, "x_ay", E.mul(12, E.var(x)), f"{case.name(x)} × 12"),
            PairStatistic(case.frame, x, "x_ay", "corr", "r_ay", f"{capital(_name(case, x))} ile 12 katının korelasyonu",
                          decimals=4),
        ),
        checks=tuple(_check(f"{i}. çift: β₁ + 12β₂", CellTarget("ciftler", "birlesim", i), digits)
                     for i in range(1, len(pairs) + 1)) + (_scalar("r_ay", "Korelasyon", 4),),
        takeaway=(
            "Tam bağlantıda katsayılar “yanlış” olmadan önce ayrı ayrı tanımlanamaz: aynı tahmini veren sonsuz sayıda "
            "katsayı çifti vardır. Sık kaynaklar: aynı değişkenin farklı birimleri, toplam ile bütün parçaları, sabit "
            "terimle birlikte bütün kategori göstergeleri (kukla değişken tuzağı, Konu 10). A3 yalnız tam tekrarı "
            "yasaklar; değişkenlerin ilişkili olması olağandır."
        ),
    )


# --- Adım 5: ikinci veride VIF -------------------------------------------------------------------------------

def _vif_options(house: Case) -> tuple[str, ...]:
    return tuple(house.extra.get("vif_options") or candidates(house))


def _vif_default(house: Case) -> tuple[str, ...]:
    return tuple(house.extra.get("vif_default") or _vif_options(house))


def _strongest_pair(house: Case, names: tuple[str, ...]) -> tuple[str, str]:
    """Kendi verinde korelasyonu (mutlak değerce) en büyük değişken çifti; metin bu çiftin ilişkisini anlatır."""

    def strength(pair: tuple[str, str]) -> float:
        value = float(np.corrcoef(house.data[pair[0]].astype(float), house.data[pair[1]].astype(float))[0, 1])
        return abs(value) if np.isfinite(value) else -1.0

    return max(itertools.combinations(names, 2), key=strength)


def _step5(house: Case) -> LabStep:
    title = "Gerçek veride VIF"
    note_ref = NoteRef("6.12", 0, ("Kod 6.4", "Tablo 6.9"))
    lead = ("Her açıklayıcı değişken diğerlerine göre regresyona alınır; yardımcı regresyonun $R_j^2$ değeri "
            "$\\text{VIF}_j = 1/(1 - R_j^2)$ formülüne yerleştirilir (Kod 6.4).")
    names = _vif_options(house)
    if len(names) < 2:
        return LabStep(number=5, title=title, note=note_ref,
                       explanation=f"{lead} Bu adım için en az iki açıklayıcı değişken gerekir: ek bir sayısal değişken "
                                   "seçin.")
    default = _vif_default(house)
    pair = tuple(house.extra.get("vif_pair") or _strongest_pair(house, default))
    twins = tuple(house.extra.get("vif_twins", ()))

    def build(choices) -> tuple:
        chosen = choices["adim5_x"]
        same = tuple(chosen) == default
        operations: list = list(house.load)
        for name in chosen:
            others = tuple(other for other in chosen if other != name)
            operations += [
                OLS(f"yardimci_{name}", house.frame, name, others,
                    f"{capital(_name(house, name))} diğer açıklayıcı değişkenlere göre: {name} ~ {' + '.join(others)}"),
                ModelValue(f"r2_{name}", f"yardimci_{name}", "r2", f"R²ⱼ: {_name(house, name)}", decimals=4),
            ]
        operations += [
            ScalarTable(tuple((_title(house, name), E.div(1, E.sub(1, E.ref(f"r2_{name}")))) for name in chosen), "vif",
                        decimals=3, heading="Açıklayıcı değişken", value="VIF"),
            BarChart("vif", "deger", "Açıklayıcı değişken", "VIF = 1 / (1 − R²ⱼ)",
                     f"{house.extra.get('data_name', 'Verileriniz')}: VIF (Tablo 6.9'daki gibi)" if same
                     else "Seçtiğiniz modelde VIF değerleri", decimals=3),
        ]
        if set(pair) <= set(chosen):
            operations.append(PairStatistic(house.frame, pair[0], pair[1], "corr", "r_cift",
                                            f"{capital(_name(house, pair[0]))} ile {_name(house, pair[1])} korelasyonu",
                                            decimals=2))
        return tuple(operations)

    def note(state, choices) -> str:
        chosen = choices["adim5_x"]
        table = state.tables["vif"]["deger"]
        if len(chosen) == 2:
            text = (f"İki değişkenli modelde iki VIF eşittir: 1/(1 − r²) = {sayi(table.iloc[0], 3)}. VIF, bir açıklayıcı "
                    "değişkenin diğerleri tarafından ne kadar iyi açıklandığını özetler: R²ⱼ bire yaklaştıkça VIF büyür ve "
                    "değişkenin diğerlerinden bağımsız kalan hareketi azalır.")
        else:
            largest = md(str(table.idxmax()))
            text = (f"En büyük VIF {sayi(table.max(), 3)} ({largest}). VIF, bir açıklayıcı değişkenin diğerleri "
                    "tarafından ne kadar iyi açıklandığını özetler: R²ⱼ bire yaklaştıkça VIF büyür ve değişkenin "
                    "diğerlerinden bağımsız kalan hareketi azalır.")
        if set(pair) <= set(chosen):
            r = state.scalars["r_cift"]
            lead = (f" Varsayılan değişkenler arasında en güçlü ikili doğrusal ilişki {phrase(house, pair[0])} ile "
                    f"{phrase(house, pair[1])} arasındadır" if house.own else
                    f" {capital(phrase(house, pair[0]))} ile {phrase(house, pair[1])} ilişkilidir")
            if abs(r) >= 0.3 or not house.own:
                text += (f"{lead} (korelasyon {sayi(r, 2)}); ilişkili olmaları olağandır, VIF bu ilişkinin katsayıların "
                         "ayrı ayrı belirlenmesini ne kadar zorlaştırdığını özetler.")
            else:
                text += (f"{lead} (korelasyon {sayi(r, 2)}); ikili ilişkiler zayıftır. VIF ise bir değişkenin bütün "
                         "diğerleriyle birlikte ilişkisini özetler.")
        for level, logged in twins:
            if level in chosen and logged in chosen:
                text += (f" {capital(phrase(house, level))} ile {phrase(house, logged)} birlikte olunca VIF çok büyür: iki "
                         "değişken neredeyse aynı bilgiyi taşır; yine de logaritma doğrusal bir dönüşüm olmadığı için tam "
                         "bağlantı yoktur.")
        if tuple(chosen) != default:
            text += " Varsayılan modelin VIF'leriyle karşılaştırmak için varsayılan seçime dönün."
        return text + " VIF bir uyarı ölçüsüdür, mekanik bir silme kuralı değildir."

    return interactive_step(
        number=5,
        title=str(house.extra.get("vif_title", title)),
        note=note_ref,
        explanation=f"{lead} " + str(house.extra.get("vif_text", "Değişkenleri değiştirin: yakın bilgi taşıyan bir "
                                                                    "değişken eklenince VIF'ler nasıl değişiyor?")),
        controls=(MultiChoice("adim5_x", "VIF hesaplanan açıklayıcı değişkenler (en az iki)", options(house, names),
                              default, help="Varsayılan: " + ", ".join(house.name(name) for name in default) + ".",
                              minimum=2),),
        build=build,
        checks=(
            *(_scalar(f"r2_{name}", f"{house.name(name)}: R²ⱼ", 4) for name in default),
            *(_check(f"{house.name(name)}: VIF", TableTarget("vif", _title(house, name), "deger"), 3)
              for name in default),
            _scalar("r_cift", "İki değişkenin korelasyonu", 2),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


_READING = (
    "Eksik değişken yanlılığı katsayının yanlış merkez çevresinde toplanmasıdır; yüksek çoklu doğrusal bağlantı sıfır "
    "koşullu ortalama sağlansa bile ayrı katsayıların hassas ayrıştırılmasını zorlaştırır (Tablo 6.10). Bir çıktıyı "
    "okurken sırasıyla sorun (§6.13):\n\n1. Araştırma sorusu ve hedef katsayı nedir?\n2. Modelde hangi temel kontroller "
    "var, hangileri dışarıda kalmış?\n3. Dışlanan bir faktörün hem sonuçla hem temel değişkenle ilişkili olması makul "
    "mü?\n4. Açıklayıcı değişkenler aynı bilgiyi tekrar ediyor olabilir mi?\n5. Katsayılar alternatif fakat ekonomik "
    "olarak savunulabilir modellerde nasıl değişiyor?\n6. Uyum yüksek olsa bile ayrı katsayıların yorumu destekleniyor "
    "mu?\n7. Sonuç ilişki diliyle mi, nedensel dille mi raporlanmalı?\n\nStandart hata, t ve p-değeri Konu 7'de "
    "sistematik olarak okunur."
)


# --- Tanım ------------------------------------------------------------------------------------------------------

def build(case: Case) -> LabSpec:
    """Konu 6 uygulamasını verilen veriyle kurar; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    house = second(case)
    labels = labels_of(case)
    labels.update(labels_of(house))
    labels.update({INTERCEPT: "Sabit terim", "beta1": "β₁", "beta2": "β₂", "birlesim": "β₁ + 12β₂",
                   "x_ay": str(case.extra.get("months_label", f"{case.name(case.roles[ACIKLAYICI])} × 12"))})
    if _omitted(case):
        choice = z_choice(case)
        first = (_step1(case, choice), _step2(case, choice), _step3(case, choice.default))
    else:
        first = (
            _need_extra(1, "Kısa ve uzun model: dışarıda kalan değişken", NoteRef("6.8", 0, ("Denklem 6.10", "Denklem 6.11")),
                        _LEAD1),
            _need_extra(2, "Yardımcı regresyon ve katsayının örneklem ayrıştırması",
                        NoteRef("6.8", 0, ("Denklem 6.12", "Denklem 6.8")),
                        "Dışarıda kalan değişken temel açıklayıcı değişkene göre tahmin edilir."),
            _need_extra(3, "Kontroller eklendikçe temel katsayı ve Python çıktısı",
                        NoteRef("6.8", 0, ("Tablo 6.6", "Şekil 6.3", "Kod 6.3")),
                        "Tablo temel açıklayıcı değişkenin katsayısını farklı kontrol kümeleriyle yan yana gösterir."),
        )
    spec = LabSpec(
        topic_key=TOPIC,
        title=str(case.extra.get("title", TITLE)),
        note_section="6",
        steps=(*first, _step4(case, load=not _omitted(case)), _step5(house),
               LabStep(number=6, title="Bütünleşik çıktı okuma: yanlılık ile belirsizliği ayırmak",
                       note=NoteRef("6.13", 0, ("Tablo 6.10",)), explanation=_READING)),
        labels=tuple(labels.items()),
        source=case.source,
    )
    return with_app_values(spec)


# --- Alternatif örnek ----------------------------------------------------------------------------------------

def _house() -> Case:
    return house_case(
        regressors=("area", "rooms", "baths", "land"),
        vif_options=("area", "rooms", "baths", "land", "larea"),
        vif_default=("area", "rooms", "baths"),
        vif_pair=("area", "baths"),
        vif_twins=(("area", "larea"),),
        vif_title="Gerçek veride VIF: KIELMC (1978)",
        vif_text=("Varsayılan model Konu 5'teki konut fiyatı modelinin üç değişkenidir (büyüklük, oda, banyo). "
                  "Değişkenleri değiştirin: arsa büyüklüğü ya da konut büyüklüğünün logaritması eklenince VIF'ler nasıl "
                  "değişiyor?"),
    )


def alternative_case() -> Case:
    return wage2_case(
        ("educ", *ALT_OMITTED),
        title="Uygulama: Eksik Değişken Yanlılığı ve Çoklu Bağlantı (WAGE2, KIELMC)",
        omitted=ALT_OMITTED,
        house=_house(),
        aux_subject="Eğitim süresi bir yıl daha uzun olan çalışanlarda",
        omitted_text=("WAGE2'de yeteneğin bir ölçüsü olarak IQ puanı vardır: eğitim süresi uzun çalışanların IQ puanı da "
                      "ortalamada yüksektir. "),
        columns={
            "yalniz": ("Yalnız eğitim", ("educ",)),
            "iq": ("Eğitim + IQ", ("educ", "IQ")),
            "deneyim": ("Eğitim + deneyim", ("educ", "exper")),
            "tam": ("Eğitim + deneyim + kıdem", ("educ", "exper", "tenure")),
            "genis": ("Eğitim + deneyim + kıdem + IQ", ("educ", "exper", "tenure", "IQ")),
        },
        columns_default=("yalniz", "iq", "genis"),
        combination=60.0,
        months_text="WAGE2'de eğitim yılı ile eğitim ayı (12 × yıl)",
        months_label="Eğitim (ay)",
    )


@cache
def alternative() -> LabSpec:
    return build(alternative_case())


STORY = ("Ücret adımları WAGE2'dir (935 erkek çalışan, 1980; aylık kazanç, ABD doları); dışarıda kalan değişkenin "
         "varsayılanı IQ puanıdır (yetenek). VIF adımı KIELMC'nin 1978 satışlarıdır (179 konut).")


# --- Kendi verin ---------------------------------------------------------------------------------------------

CUSTOM = custom_lab(
    build,
    sample_employed,
    ("Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sonuç ve temel açıklayıcı değişken zorunludur ve sayısal olmalıdır. "
     "Dışarıda kalan değişken ek sayısal değişkenlerden seçilir (en az bir, en çok 3); VIF temel açıklayıcı ve ek "
     f"değişkenlerle hesaplanır. {ROW_RULE}"),
    roles((1, 2, 3, 4, 5)),
    "Kısa modelde dışarıda kalan değişkenin seçenekleri ve VIF'in değişkenleridir.",
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
