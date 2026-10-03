"""Konu 0 uygulaması: veri tablosu, merkez ve yayılım, birlikte hareket, yüzde ve logaritma, koşullu ortalama.

Bölüm 0'ın laboratuvar bölümü yoktur; adımlar bölümün çözümlü örnekleridir ve bölüm sırasıyla alt bölümlere
bağlanır: Tablo 0.1 (§0.1), toplam ve ortalama (§0.3), sapma ve varyans ile Tablo 0.2 (§0.4), Şekil 0.2 ve WAGE1
kovaryansı (§0.5), yüzde değişim ile Tablo 0.3 (§0.6), koşullu ortalama (§0.7), tahmin (§0.8, Tablo 0.4) ve ilk
regresyon çıktısı (§0.9, Kod 0.1–0.2). §0.2'de hesap yoktur; örnekleme Sezgi sekmesinde (Deney 1) ele alınır. Her
``Check`` notlarda basılı bir sayıdır; değer notlardan kopyalanmıştır, hesaplanmamıştır.

Etkileşim: gösterge değişkeninin kodlaması (Adım 1), özetlenen değişken (Adım 2–3), ölçek ve kayma (Adım 3), örüntü
(Adım 4), ikinci değişken (Adım 5), başlangıç ve yeni değer (Adım 6), koşul (Adım 7) ve açıklayıcı değişken
(Adım 9). Standart hata, t, p ve güven aralığı notlardaki gibi yazılım çıktısında görünür; yorumları Konu 7'dedir.
Anlamlılık yıldızları gösterilmez: R çıktısı da yıldızsız yazdırılır (yıldızlar Konu 7'de, F testi Konu 8'de).
"""

from __future__ import annotations

import math

from core import wooldridge_data as W
from core.labs import expr as E
from core.labs.spec import (
    INTERCEPT,
    OLS,
    BarChart,
    CellTarget,
    Check,
    Choice,
    CoefTarget,
    Derive,
    DotPlot,
    GroupStats,
    InlineData,
    LabSpec,
    LabStep,
    LineChart,
    LoadWooldridge,
    ModelTarget,
    ModelValue,
    NoteRef,
    NumberChoice,
    PairStatistic,
    RegressionTable,
    Scalar,
    ScalarTable,
    ScalarTarget,
    ScatterPlot,
    Shape,
    ShowFrame,
    ShowModel,
    Statistic,
    Support,
    TableTarget,
    VariableTypes,
    interactive_step,
)
from core.labs.sezgi import plain
from core.labs.wording import directions, signed_difference, tr_lower

DATA = "wage1"
WORKERS = "calisanlar"
WORKER_COLUMNS = ("calisan", "ucret", "egitim", "deneyim", "kadin")
WORKER_ROWS = (
    (1, 8.50, 12, 4, 1),
    (2, 12.00, 16, 3, 0),
    (3, 9.75, 14, 6, 1),
    (4, 15.25, 18, 8, 0),
    (5, 10.50, 12, 10, 0),
)
"""Tablo 0.1: örnek bir yatay kesit veri tablosu (beş çalışan)."""

WORKER_LABELS = {"ucret": "Saatlik ücret", "egitim": "Eğitim yılı", "deneyim": "Deneyim yılı"}
WORKER_UNITS = {"ucret": "", "egitim": " yıl", "deneyim": " yıl"}

TWO_SETS = (("gozlem", "A", "B"), tuple((i + 1, a, b) for i, (a, b) in enumerate(zip((8, 9, 10, 11, 12),
                                                                                      (2, 6, 10, 14, 18)))))
"""Tablo 0.2: aynı ortalamaya sahip iki veri seti."""

PATTERN_ROWS = tuple(zip(range(1, 10),
                         (1.7, 2.2, 3.1, 3.7, 5.2, 5.8, 7.2, 7.6, 9.0),
                         (9.0, 8.2, 7.4, 6.2, 5.5, 4.8, 3.2, 2.4, 1.5),
                         (5.0, 2.0, 7.5, 4.5, 8.0, 1.8, 6.0, 3.8, 5.3)))
"""Şekil 0.2'nin noktaları: x = 1, …, 9 ve üç örüntünün y değerleri."""

PATTERNS = {
    "pozitif": "Pozitif ilişki",
    "negatif": "Negatif ilişki",
    "zayif": "Zayıf doğrusal ilişki",
    "karesel": "Doğrusal olmayan (karesel) ilişki",
}

REGRESSORS = ("educ", "exper", "tenure")
EDUCATION_LEVELS = (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
"""Koşul olarak seçilebilen eğitim yılları: WAGE1'de her iki cinsiyetten gözlem bulunan düzeyler."""


def _worker_data() -> InlineData:
    return InlineData(WORKERS, WORKER_COLUMNS, WORKER_ROWS, "Tablo 0.1: örnek bir yatay kesit veri tablosu")


def _load() -> LoadWooldridge:
    return LoadWooldridge(DATA, "WAGE1 veri seti (Wooldridge, 2020): 526 çalışan")


def _scalar(name: str, expected: float, label: str, decimals: int) -> Check:
    return Check(label, ScalarTarget(name), expected, decimals)


def _small(value: float, decimals: int = 2) -> str:
    """Sıfıra çok yakın ama sıfır olmayan bir fark "0,00" yerine ilk anlamlı basamağıyla yazılır (ör. 0,005).

    Kesirli sayılar uygulamanın geri kalanı ve notlardaki formüllerle aynı biçimde binlik ayırıcısız yazılır."""

    while decimals < 4 and value != 0 and abs(value) < 0.5 * 10 ** (-decimals):
        decimals += 1
    return plain(value, decimals)


def _signed(value: float, decimals: int = 0) -> str:
    """Bir toplamın ikinci terimi: " + 5", " − 10"; sıfırsa boş."""

    if value == 0:
        return ""
    return f" {'+' if value > 0 else '−'} {plain(abs(value), decimals)}"


def _affine_expr(a: float, c: int) -> E.Expr:
    """a·B + c; kod "+ -10" yerine "- 10" yazsın ve c = 0 terimi düşsün diye işarete göre kurulur."""

    scaled = E.mul(a, E.var("B"))
    if c > 0:
        return E.add(scaled, c)
    return E.sub(scaled, -c) if c < 0 else scaled


def _affine(a: float, c: int) -> str:
    """Dönüşümün yazımı: "y = 2,0·B + 5", "y = −3,0·B − 10", "y = 0,5·B"."""

    return f"y = {plain(a, 1)}·B{_signed(c)}"


# --- Adım 1: veri tablosunun anatomisi ---------------------------------------------------------------------

INDICATOR_CHOICE = Choice(
    "adim1_gosterge", "Gösterge değişkeninin kodlaması", (("kadin", "Kadın = 1"), ("erkek", "Erkek = 1")), "kadin",
    help="Notlardaki Tablo 0.1: kadın çalışanlar 1, diğer çalışanlar 0.",
)


def _table(choices) -> tuple:
    indicator = choices["adim1_gosterge"]
    operations = [
        _worker_data(),
        Shape(WORKERS, "n", "k", exclude=("calisan",)),
        VariableTypes(
            WORKERS,
            (
                ("calisan", "Kimlik", "Gözlemi tanımlar; analitik değişken sayılmaz"),
                ("ucret", "Nicel", "Saatlik ücret"),
                ("egitim", "Nicel", "Tamamlanan eğitim, yıl"),
                ("deneyim", "Nicel", "İş deneyimi, yıl"),
                ("kadin", "Gösterge (0/1)", "Kadın = 1, diğer = 0"),
            ),
            "turler",
        ),
    ]
    operations += [
        ShowFrame(WORKERS, WORKER_COLUMNS, "Tablo 0.1: her satır bir çalışan, her sütun bir değişken"),
        Statistic(WORKERS, "kadin", "mean", "kadin_payi", "Kadın payı: gösterge ortalaması",
                  decimals=2),
    ]
    if indicator == "erkek":
        operations.append(Scalar("erkek_payi", E.sub(1, E.ref("kadin_payi")),
                                 "Erkek payı: 1 − kadın payı", decimals=2))
    return tuple(operations)


def _table_note(state, choices) -> str:
    male = choices["adim1_gosterge"] == "erkek"
    share = state.scalars["erkek_payi" if male else "kadin_payi"]
    text = ("Beş gözlem ve dört analitik değişken vardır; kimlik numarası gözlemi tanımlar, ölçüm değildir. "
            f"Gösterge değişkeninin ortalaması 1 ile kodlanan grubun örneklemdeki payıdır: {plain(share, 2)} (§0.1).")
    if not male:
        return text + " Beş çalışandan ikisi kadındır. Kodlamayı çevirip (erkek = 1) payın nasıl değiştiğine bakın."
    return text + (" Erkek göstergesi 1 − kadın göstergesidir: kadın = 1 olan gözlemde erkek = 0, kadın = 0 olan "
                   "gözlemde erkek = 1 olur. Ortalaması erkeklerin payıdır ve kadınların payıyla toplamı 1 eder. "
                   "Payın yorumu hangi grubun 1 ile kodlandığına bağlıdır; bu yüzden kodlama her zaman açıkça "
                   "yazılır.")


# --- Adım 2 ve 3: toplam, ortalama, sapma ve varyans ----------------------------------------------------------

VARIABLE_CHOICE = Choice(
    "adim2_degisken", "Özetlenen değişken (Tablo 0.1)", tuple(WORKER_LABELS.items()), "egitim",
    help="Notlardaki örnek: eğitim yılları 12, 16, 14, 18 ve 12.",
)


def _sum_mean(choices) -> tuple:
    variable = choices["adim2_degisken"]
    return (
        ShowFrame(WORKERS, ("calisan", variable), "xᵢ: i numaralı gözlemin değeri"),
        Statistic(WORKERS, variable, "sum", "toplam", "Toplam Σxᵢ", decimals=2),
        Statistic(WORKERS, variable, "count", "n_gozlem", "Gözlem sayısı n", decimals=0),
        Scalar("ortalama", E.div(E.ref("toplam"), E.ref("n_gozlem")), "Ortalama x̄ = Σxᵢ / n", decimals=2),
        Statistic(WORKERS, variable, "mean", "ortalama_yazilim", "Yazılımın ortalama fonksiyonu", decimals=2),
    )


def _sum_mean_note(state, choices) -> str:
    variable = choices["adim2_degisken"]
    values = [row[WORKER_COLUMNS.index(variable)] for row in WORKER_ROWS]
    written = " + ".join(plain(value, 2 if variable == "ucret" else 0) for value in values)
    total, mean = state.scalars["toplam"], state.scalars["ortalama"]
    digits = 2 if variable == "ucret" else 0
    text = (f"Σxᵢ = {written} = {plain(total, digits)}; x̄ = {plain(total, digits)} / 5 = {plain(mean, 2)}"
            f"{WORKER_UNITS[variable]}.")
    if not any(abs(value - mean) < 1e-9 for value in values):
        text += " Ortalama veride gözlenen bir değer olmak zorunda değildir: hiçbir çalışanın değeri tam olarak " \
                "ortalamaya eşit değildir."
    return text + " Ortalama merkezi özetler, yayılımı göstermez; bunu bir sonraki adımdaki varyans tamamlar (§0.3)."


SCALE_CHOICE = NumberChoice("adim3_a", "Ölçek a: B'nin değerleri a ile çarpılır", -3, 3, 1, 0.5, decimals=1,
                            help="yᵢ = a·Bᵢ + c; Bᵢ, Tablo 0.2'deki B'nin değerleridir. Notlardaki tablo: a = 1, "
                                 "c = 0 (B'nin kendisi).")
SHIFT_CHOICE = NumberChoice("adim3_c", "Kayma c: B'nin değerlerine c eklenir", -10, 20, 0, 1, integer=True,
                            decimals=0, help="Kayma ortalamayı değiştirir, yayılımı değiştirmez.")


def _dot_range(a: float, c: int) -> tuple[float, float]:
    """İki nokta grafiğinin ortak ekseni: A, B ve y = a·B + c görünür (varsayılan ayarda 0–20, slayttaki gibi).

    Uç gözlemler eksenin sınırına düşmesin diye her iki yanda en az bir birim pay bırakılır.
    """

    ends = (a * 2 + c, a * 18 + c)
    low = math.floor((min(2.0, *ends) - 1) / 5) * 5
    high = math.ceil((max(18.0, *ends) + 1) / 5) * 5
    return low, high


def _variance(choices) -> tuple:
    variable = choices["adim2_degisken"]
    a, c = round(float(choices["adim3_a"]), 2), int(choices["adim3_c"])
    axis = _dot_range(a, c)
    return (
        Derive(WORKERS, "sapma", E.sub(E.var(variable), E.ref("ortalama")), "Sapma xᵢ − x̄"),
        Derive(WORKERS, "sapma_kare", E.power(E.var("sapma"), 2), "Kareli sapma (xᵢ − x̄)²"),
        ShowFrame(WORKERS, ("calisan", variable, "sapma", "sapma_kare"), "Sapmalar ve kareli sapmalar"),
        Statistic(WORKERS, "sapma", "sum", "sapma_toplami", "Sapmaların toplamı", decimals=2),
        Statistic(WORKERS, "sapma_kare", "sum", "kare_toplami", "Kareli sapmaların toplamı", decimals=2),
        Scalar("varyans", E.div(E.ref("kare_toplami"), E.sub(E.ref("n_gozlem"), 1)), "Varyans s² = toplam / (n − 1)",
               decimals=2),
        Scalar("std_sapma", E.sqrt(E.ref("varyans")), "Standart sapma s = √s²", decimals=2),
        Statistic(WORKERS, variable, "var", "varyans_yazilim", "Yazılımla varyans (payda n − 1)",
                  decimals=2),
        InlineData("iki_veri", TWO_SETS[0], TWO_SETS[1], "Tablo 0.2: aynı ortalamaya sahip iki veri seti"),
        Statistic("iki_veri", "A", "mean", "ort_A", "A: ortalama", decimals=2),
        Statistic("iki_veri", "B", "mean", "ort_B", "B: ortalama", decimals=2),
        Statistic("iki_veri", "A", "var", "var_A", "A: varyans s²", decimals=2),
        Statistic("iki_veri", "B", "var", "var_B", "B: varyans s²", decimals=2),
        Statistic("iki_veri", "A", "std", "std_A", "A: standart sapma s", decimals=2),
        Statistic("iki_veri", "B", "std", "std_B", "B: standart sapma s", decimals=2),
        Derive("iki_veri", "y", _affine_expr(a, c), f"Dönüştürülmüş B: {_affine(a, c)}"),
        Statistic("iki_veri", "y", "mean", "ort_y", "y: ortalama", decimals=2),
        Statistic("iki_veri", "y", "std", "std_y", "y: standart sapma", decimals=2),
        ScalarTable(
            (
                ("A: ortalama", E.ref("ort_A")),
                ("A: varyans s²", E.ref("var_A")),
                ("A: standart sapma s", E.ref("std_A")),
                ("B: ortalama", E.ref("ort_B")),
                ("B: varyans s²", E.ref("var_B")),
                ("B: standart sapma s", E.ref("std_B")),
                ("y = a·B + c: ortalama", E.ref("ort_y")),
                ("y = a·B + c: standart sapma", E.ref("std_y")),
            ),
            "yayilim",
            decimals=2,
        ),
        DotPlot("iki_veri", "A", "Değer", "Veri seti A (Tablo 0.2)", references=(("ort_A", "Ortalama"),),
                x_range=axis),
        DotPlot("iki_veri", "y", "Değer",
                "Veri seti B (Tablo 0.2)" if (a, c) == (1, 0) else f"Dönüştürülmüş B: {_affine(a, c)}",
                references=(("ort_y", "Ortalama"),), x_range=axis),
    )


def _variance_note(state, choices) -> str:
    s = state.scalars
    a, c = round(float(choices["adim3_a"]), 2), int(choices["adim3_c"])
    variable = choices["adim2_degisken"]
    unit = WORKER_UNITS[variable].strip()
    squared = f"({unit})²" if unit else "özgün birimin karesi"
    text = (f"Sapmaların toplamı sıfırdır ({plain(s['sapma_toplami'], 2)}); kareli sapmaların toplamı "
            f"{plain(s['kare_toplami'], 2)}, varyans {plain(s['varyans'], 2)} ve standart sapma "
            f"{plain(s['std_sapma'], 2)}. Varyansın birimi {squared}, standart sapmanın birimi değişkenin kendi "
            "birimidir. ")
    text += (f"Tablo 0.2'de A ve B'nin ortalaması aynıdır ({plain(s['ort_A'], 0)}); standart sapmalar "
             f"{plain(s['std_A'], 2)} ve {plain(s['std_B'], 2)}: B daha geniş yayılır. ")
    if a == 1 and c == 0:
        return text + ("Ölçek a ve kayma c ile yᵢ = a·Bᵢ + c dönüşümünde ȳ = a·B̄ + c ve s_y = |a|·s_B kurallarını "
                       "deneyin (§0.4).")
    spread = ("0" if a == 0 else
              f"{plain(abs(a), 1)}·{plain(s['std_B'], 4)} ≈ {plain(s['std_y'], 2)}")
    text += (f"{_affine(a, c)}: ortalama ȳ = {plain(a, 1)}·10{_signed(c)} = {plain(s['ort_y'], 2)}; standart sapma "
             f"s_y = |a|·s_B = {spread}. ")
    if a == 0:
        text += f"a = 0 iken bütün değerler {plain(c, 0)} olur: yayılım kalmaz, standart sapma sıfırdır."
    elif abs(a) > 1:
        text += "|a| > 1 olduğu için ölçek yayılımı |a| katına çıkarır."
    elif abs(a) < 1:
        text += "|a| < 1 olduğu için ölçek yayılımı küçültür: standart sapma |a| ile çarpılır."
    else:
        text += "|a| = 1 olduğu için yayılım değişmez."
    if a < 0:
        text += " a < 0 iken değerlerin sırası tersine döner; standart sapma yine negatif olmaz, bu yüzden |a| yazılır."
    if c != 0:
        text += " Kayma c ortalamayı kaydırır, yayılımı değiştirmez."
    return text + " Genel kural: ȳ = a·B̄ + c ve s_y = |a|·s_B (§0.4)."


# --- Adım 4: korelasyonun yönü ve doğrusallık ---------------------------------------------------------------

PATTERN_CHOICE = Choice("adim4_desen", "Gösterilen örüntü", tuple(PATTERNS.items()), "pozitif",
                        help="Notlardaki Şekil 0.2'nin üç örüntüsü ile §0.5'teki karesel örnek.")


def _patterns(choices) -> tuple:
    pattern = choices["adim4_desen"]
    if pattern == "karesel":
        plot = ScatterPlot("karesel", "x", "y", "x", "y = 0,6·(x − 2,5)²",
                           "Kusursuz ama doğrusal olmayan ilişki: r = 0")
    else:
        plot = ScatterPlot("desenler", "x", pattern, "x", "y", f"Şekil 0.2: {PATTERNS[pattern].lower()}")
    return (
        InlineData("desenler", ("x", "pozitif", "negatif", "zayif"), PATTERN_ROWS,
                   "Şekil 0.2: temsili örüntülerin noktaları"),
        PairStatistic("desenler", "x", "pozitif", "corr", "r_pozitif", "Pozitif ilişki: r", decimals=3),
        PairStatistic("desenler", "x", "negatif", "corr", "r_negatif", "Negatif ilişki: r", decimals=3),
        PairStatistic("desenler", "x", "zayif", "corr", "r_zayif", "Zayıf doğrusal ilişki: r", decimals=3),
        InlineData("karesel", ("x",), tuple((i / 2,) for i in range(11)), "§0.5: x = 0; 0,5; …; 5", layout=11),
        Derive("karesel", "y", E.mul(0.6, E.power(E.sub(E.var("x"), 2.5), 2)), "y = 0,6·(x − 2,5)²"),
        PairStatistic("karesel", "x", "y", "corr", "r_karesel", "Karesel ilişki: r", decimals=3),
        ScalarTable(
            (
                ("Pozitif ilişki", E.ref("r_pozitif")),
                ("Negatif ilişki", E.ref("r_negatif")),
                ("Zayıf doğrusal ilişki", E.ref("r_zayif")),
                ("Karesel ilişki", E.ref("r_karesel")),
            ),
            "korelasyonlar",
            decimals=3,
        ),
        plot,
    )


def _patterns_note(state, choices) -> str:
    pattern = choices["adim4_desen"]
    r = state.scalars[f"r_{pattern}"]
    if pattern == "karesel":
        return (f"r = {plain(r, 3)}: y, x ile tam olarak belirlenir, ama ilişki doğrusal değildir. Korelasyon yalnız "
                "doğrusal ilişkiyi ölçer; r = 0 “ilişki yok” demek değildir (§0.5). Bu yüzden korelasyonu "
                "yorumlamadan önce saçılım grafiğine bakmak yararlıdır.")
    if pattern == "zayif":
        return (f"r = {plain(r, 3)}: noktalar belirgin bir doğru çevresinde toplanmıyor; doğrusal ilişki çok zayıf. "
                "İşaret ilişkinin yönünü, mutlak değer doğrusal ilişkinin gücünü gösterir (§0.5).")
    direction = "x arttıkça y de çoğunlukla artar" if r > 0 else "x arttıkça y çoğunlukla azalır"
    return (f"r = {plain(r, 3)}: {direction}; noktalar bir doğru çevresinde sıkı toplanır. Korelasyon ilişkinin "
            "yönünü ve doğrusal gücünü özetler; nedenselliği göstermez (§0.5).")


# --- Adım 5: WAGE1'de kovaryans, korelasyon ve ölçü birimi ----------------------------------------------------

SECOND_CHOICE = Choice("adim5_x", "İkinci değişken (birinci değişken: saatlik ücret)", W.options(DATA, REGRESSORS),
                       "educ", help="Notlardaki örnek: saatlik ücret ve eğitim yılı.")


def _covariance(choices) -> tuple:
    x = choices["adim5_x"]
    label = W.variable(DATA, x)
    return (
        _load(),
        Derive(DATA, "wage_sent", E.mul(100, E.var("wage")), "Saatlik ücret, sent/saat (1 dolar = 100 sent)"),
        PairStatistic(DATA, "wage", x, "cov", "kovaryans", "Kovaryans s_xy (dolar/saat)", decimals=4),
        Statistic(DATA, "wage", "std", "std_wage", "Ücretin standart sapması", decimals=4),
        Statistic(DATA, x, "std", "std_x", f"{label.label}: standart sapma", decimals=4),
        Scalar("korelasyon", E.div(E.ref("kovaryans"), E.mul(E.ref("std_wage"), E.ref("std_x"))),
               "Korelasyon r = s_xy / (s_x·s_y)", decimals=3),
        PairStatistic(DATA, "wage", x, "corr", "korelasyon_yazilim", "Yazılımla korelasyon", decimals=3),
        PairStatistic(DATA, "wage_sent", x, "cov", "kovaryans_sent", "Kovaryans (ücret sent/saat)", decimals=2),
        PairStatistic(DATA, "wage_sent", x, "corr", "korelasyon_sent", "Korelasyon (ücret sent/saat)", decimals=3),
        PairStatistic(DATA, "lwage", x, "corr", "korelasyon_log", "Korelasyon, ln(ücret)", decimals=3),
        ScalarTable(
            (
                ("Kovaryans, ücret dolar/saat", E.ref("kovaryans")),
                ("Kovaryans, ücret sent/saat", E.ref("kovaryans_sent")),
            ),
            "birim_kovaryans",
            decimals=4,
        ),
        ScalarTable(
            (
                ("Korelasyon, ücret dolar/saat", E.ref("korelasyon")),
                ("Korelasyon, ücret sent/saat", E.ref("korelasyon_sent")),
                ("Korelasyon, ücretin logaritması", E.ref("korelasyon_log")),
            ),
            "birim_korelasyon",
            decimals=3,
        ),
        ScatterPlot(DATA, x, "wage", label.text, "Saatlik ücret (dolar/saat)",
                    f"WAGE1: {label.label.lower()} ve saatlik ücret", size=7, opacity=0.35),
    )


def _covariance_note(state, choices) -> str:
    s = state.scalars
    label = tr_lower(W.variable(DATA, choices["adim5_x"]).label)
    return (
        f"Ücret ile {label} arasındaki kovaryans {plain(s['kovaryans'], 4)}; korelasyon "
        f"{plain(s['kovaryans'], 4)} / ({plain(s['std_wage'], 4)} × {plain(s['std_x'], 4)}) = "
        f"{plain(s['korelasyon'], 3)}. Ücret sent olarak ölçülünce kovaryans 100 katına çıkar "
        f"({plain(s['kovaryans_sent'], 2)}), korelasyon değişmez ({plain(s['korelasyon_sent'], 3)}). Logaritma "
        f"doğrusal bir dönüşüm değildir: ücretin logaritmasıyla korelasyon {plain(s['korelasyon_log'], 3)}. "
        "Kovaryansın işareti ilişkinin yönünü gösterir; büyüklüğü ölçü birimine bağlı olduğu için ilişkinin gücü "
        "korelasyonla karşılaştırılır (§0.5)."
    )


# --- Adım 6: yüzde değişim, yüzde puan ve log farkı ----------------------------------------------------------

START_CHOICE = NumberChoice("adim6_x0", "Başlangıç değeri x₀", 10, 300, 100, 1, integer=True, decimals=0,
                            help="Notlardaki örnek: 100'den 120'ye.")
END_CHOICE = NumberChoice("adim6_x1", "Yeni değer x₁", 10, 300, 120, 1, integer=True, decimals=0,
                          help="Notlardaki örnek: 100'den 120'ye.")
RATE0_CHOICE = NumberChoice("adim6_oran0", "Başlangıç oranı (%)", 1, 99, 30, 1, integer=True, decimals=0,
                            help="Notlardaki örnek: %30'dan %35'e.")
RATE1_CHOICE = NumberChoice("adim6_oran1", "Yeni oran (%)", 1, 99, 35, 1, integer=True, decimals=0,
                            help="Notlardaki örnek: %30'dan %35'e.")


def _percent_change(start, end) -> E.Expr:
    return E.mul(100, E.div(E.sub(end, start), start))


def _log_change(start, end) -> E.Expr:
    return E.mul(100, E.sub(E.log(end), E.log(start)))


def _percent(choices) -> tuple:
    x0, x1 = int(choices["adim6_x0"]), int(choices["adim6_x1"])
    r0, r1 = int(choices["adim6_oran0"]), int(choices["adim6_oran1"])
    forward, back = directions(str(x0), str(x1))
    return (
        Scalar("yuzde_degisim", _percent_change(x0, x1), f"Yüzde değişim: {x0} → {x1}", decimals=2, percent=True),
        Scalar("geri_donus", _percent_change(x1, x0), f"Geri dönüş: {x1} → {x0}", decimals=2,
               percent=True),
        Scalar("log_farki", _log_change(x0, x1), f"100·[ln(x₁) − ln(x₀)]: {x0} → {x1}", decimals=2),
        Scalar("log_geri", _log_change(x1, x0), f"100·[ln(x₀) − ln(x₁)]: {x1} → {x0}", decimals=2),
        Scalar("yaklasim_farki", E.sub(E.ref("yuzde_degisim"), E.ref("log_farki")),
               "Tam yüzde değişim − log farkı", decimals=2),
        Scalar("yuzde_puan", E.sub(r1, r0), f"Yüzde puan farkı: %{r0} → %{r1}", decimals=0),
        Scalar("goreli_degisim", _percent_change(r0, r1), f"Göreli yüzde değişim: %{r0} → %{r1}", decimals=2,
               percent=True),
        ScalarTable(
            (
                (f"Yüzde değişim, {forward}", E.ref("yuzde_degisim")),
                (f"Log farkı ×100, {forward}", E.ref("log_farki")),
                (f"Yüzde değişim, {back}", E.ref("geri_donus")),
                (f"Log farkı ×100, {back}", E.ref("log_geri")),
                (f"Yüzde puan, %{r0} → %{r1}", E.ref("yuzde_puan")),
                (f"Göreli yüzde değişim, %{r0} → %{r1}", E.ref("goreli_degisim")),
            ),
            "yuzdeler",
            decimals=2,
        ),
        InlineData("log_tablo", ("x0", "x1"), ((100, 105), (100, 120), (100, 150)), "Tablo 0.3: üç değişim"),
        Derive("log_tablo", "tam", _percent_change(E.var("x0"), E.var("x1")), "Tam yüzde değişim"),
        Derive("log_tablo", "log_fark", _log_change(E.var("x0"), E.var("x1")), "100·[ln(x₁) − ln(x₀)]"),
        Derive("log_tablo", "fark", E.sub(E.var("tam"), E.var("log_fark")), "Fark"),
        ShowFrame("log_tablo", ("x0", "x1", "tam", "log_fark", "fark"), "Tablo 0.3: tam yüzde değişim ve log farkı"),
        Support("egri", "x1", 50, 200, "x₀ = 100 iken x₁ = 50, 51, …, 200"),
        Derive("egri", "tam", _percent_change(100, E.var("x1")), "Tam yüzde değişim"),
        Derive("egri", "log_fark", _log_change(100, E.var("x1")), "100·[ln(x₁) − ln(100)]"),
        LineChart("egri", "x1", "tam", "Yeni değer x₁ (x₀ = 100)", "Değişim (%)",
                  "Tam yüzde değişim ve log farkı: küçük değişimlerde yakın, büyük değişimlerde ayrışır",
                  markers=False, series=(("log_fark", "100·[ln(x₁) − ln(x₀)]"),), legend="Tam yüzde değişim"),
    )


def _percent_note(state, choices) -> str:
    s = state.scalars
    x0, x1 = int(choices["adim6_x0"]), int(choices["adim6_x1"])
    r0, r1 = int(choices["adim6_oran0"]), int(choices["adim6_oran1"])
    if x0 == x1:
        change = f"x₀ = x₁ = {x0}: değişim yoktur; tam yüzde değişim de log farkı da sıfırdır."
    else:
        forward, back = s["yuzde_degisim"], s["geri_donus"]
        # İki yöndeki değişim iki basamakta aynı görünüyorsa (ör. %0,50 ve %−0,50) asimetri dört basamakta görünür.
        digits = 4 if plain(abs(forward), 2) == plain(abs(back), 2) else 2
        change = (f"{x0} → {x1}: tam yüzde değişim %{plain(forward, digits)}, geri dönüş {x1} → {x0} ise "
                  f"%{plain(back, digits)}; yüzde değişim simetrik değildir, çünkü payda (başlangıç değeri) "
                  f"değişir. Log farkı simetriktir: {plain(s['log_farki'], 2)} ve {plain(s['log_geri'], 2)}. Tam "
                  f"değişim ile log farkı arasındaki fark {_small(s['yaklasim_farki'])}; değişim büyüdükçe artar "
                  "(Tablo 0.3).")
    if r0 == r1:
        rate = f" Oran değişmezse (%{r0} → %{r1}) fark 0 yüzde puandır, göreli değişim de sıfırdır."
    else:
        verb = "artar" if r1 > r0 else "düşer"
        rate = (f" Oran %{r0} iken %{r1} olursa oran {abs(r1 - r0)} yüzde puan {verb}; göreli yüzde değişim "
                f"%{plain(s['goreli_degisim'], 2)}.")
    return change + rate + " Yüzde ile yüzde puan karıştırılmamalıdır (§0.6)."


# --- Adım 7: beklenen değer ve koşullu ortalama ----------------------------------------------------------------

def _level_options() -> tuple[tuple[str, str], ...]:
    return tuple((str(level), f"{level} yıl") for level in EDUCATION_LEVELS)


LEVEL_CHOICE = Choice("adim7_egitim", "Koşul: eğitim yılı", _level_options(), "16",
                      help="Notlardaki örnek: E(ücret | eğitim = 16).")


def _conditional(choices) -> tuple:
    level = int(choices["adim7_egitim"])
    return (
        Statistic(DATA, "wage", "mean", "ortalama_ucret", "Koşulsuz örneklem ortalaması ȳ", decimals=2),
        GroupStats(DATA, ("educ",), "wage", ("count", "mean"), "egitime_gore",
                   "Eğitim yılına göre çalışan sayısı ve ortalama saatlik ücret", decimals=2),
        Statistic(DATA, "wage", "mean", "kosullu_ortalama", f"Koşullu ortalama: eğitim = {level}",
                  where=("educ", level), decimals=2),
        Statistic(DATA, "wage", "count", "kosul_n", f"Çalışan sayısı: eğitim = {level}",
                  where=("educ", level), decimals=0),
        BarChart("egitime_gore", "mean", "Eğitim yılı", "Ortalama saatlik ücret (dolar/saat)",
                 "WAGE1: eğitim yılına göre ortalama saatlik ücret (koşullu örneklem ortalamaları)", decimals=1),
    )


def _conditional_note(state, choices) -> str:
    s = state.scalars
    level = int(choices["adim7_egitim"])
    count = int(s["kosul_n"])
    text = (f"Örneklemdeki 526 çalışanın ortalama saatlik ücreti (koşulsuz örneklem ortalaması ȳ) "
            f"{plain(s['ortalama_ucret'], 2)} dolar; eğitimi {level} yıl olan {count} çalışanın ortalaması "
            f"{plain(s['kosullu_ortalama'], 2)} dolar. ")
    if count < 15:
        text += (f"Bu grup küçüktür ({count} gözlem); örneklem koşullu ortalaması az sayıda gözleme dayanır ve "
                 "örneklemden örnekleme çok değişebilir. ")
    return text + ("Bu sayılar örneklemden hesaplanır; anakütledeki E(ücret | eğitim) bilinmez. Koşul birden fazla "
                   "değişkenle de konabilir: ör. E(ücret | eğitim = 16, kadın = 1). Grafik, örneklemdeki koşullu "
                   "ortalamaların eğitimle nasıl değiştiğini gösterir; regresyon, koşullu ortalamanın açıklayıcı "
                   "değişkenle nasıl değiştiğini inceler (§0.7).")


# --- Adım 9: ilk regresyon çıktısı ---------------------------------------------------------------------------

X_CHOICE = Choice("adim9_x", "Açıklayıcı değişken (bağımlı değişken: saatlik ücret)", W.options(DATA, REGRESSORS),
                  "educ", help="Konu 0'da yalnız çıktının parçaları okunur; basit regresyon Konu 3'te kurulur.")


_SLOPE_PHRASES = {
    "educ": "eğitimi bir yıl daha uzun olan",
    "exper": "potansiyel deneyimi bir yıl daha fazla olan",
    "tenure": "mevcut işverendeki kıdemi bir yıl daha fazla olan",
}


def _regression(choices) -> tuple:
    x = choices["adim9_x"]
    notes = x == "educ"
    operations = [
        OLS("model", DATA, "wage", (x,), "Kod 0.1: wage ~ educ" if notes else
            f"Kod 0.1'deki model, seçtiğiniz değişkenle: wage ~ {x}"),
        ShowModel("model", "Kod 0.2: yazılım çıktısının temel bölümü" if notes else
                  "Seçtiğiniz modelin çıktısı (Kod 0.2'deki gibi)", stats=("nobs", "r2"), stars=False),
        PairStatistic(DATA, "wage", x, "cov", "kov_wx", "Kovaryans s_xy", decimals=4),
        Statistic(DATA, x, "var", "var_x", "Varyans s_x²", decimals=4),
        Scalar("egim_formul", E.div(E.ref("kov_wx"), E.ref("var_x")), "Eğim = kovaryans / varyans", decimals=4),
        PairStatistic(DATA, "wage", x, "corr", "r_wx", "Korelasyon r", decimals=4),
        Scalar("r_kare", E.power(E.ref("r_wx"), 2), "Korelasyonun karesi r²", decimals=3),
        ModelValue("b1", "model", "coef", "Çıktıdaki eğim β̂₁", term=x),
        ModelValue("r2", "model", "r2", "Çıktıdaki R²", decimals=3),
    ]
    if x != "educ":
        operations += [
            OLS("model_notlar", DATA, "wage", ("educ",), "Karşılaştırma için notlardaki model: wage ~ educ"),
            RegressionTable((("(1) Notlar", "model_notlar"), ("(2) Seçiminiz", "model")), ("educ", x, INTERCEPT),
                            "karsilastirma", "Notlardaki model ile seçtiğiniz model, bağımlı değişken: saatlik ücret",
                            stars=False),
        ]
    return tuple(operations)


def _regression_note(state, choices) -> str:
    s = state.scalars
    x = choices["adim9_x"]
    b1 = s["b1"]
    text = (f"Çıktıdaki eğim {plain(b1, 4)}; aynı sayı kovaryansın varyansa oranıdır: {plain(s['kov_wx'], 4)} / "
            f"{plain(s['var_x'], 4)} = {plain(s['egim_formul'], 4)}. R² = {plain(s['r2'], 3)}, korelasyonun karesiyle "
            f"aynıdır ({plain(s['r_kare'], 3)}); bu eşitlik yalnız sabit terimli basit regresyonda geçerlidir. ")
    text += (f"Mekanik yorum: örneklemde {_SLOPE_PHRASES[x]} çalışanların tahmin edilen saatlik ücreti ortalama "
             f"{plain(abs(b1), 2)} dolar daha {signed_difference(b1)}. Bu, nedensel bir etki iddiası değildir. "
             "Standart hata (`std err`), t, p-değeri (`P>|t|`) ve %95 güven aralığı sütunları Konu 7'de, F istatistiği "
             "(`F-statistic`) Konu 8'de açıklanır (§0.9).")
    if x != "educ":
        text += " Karşılaştırma tablosunda sütun (1) notlardaki eğitim modeli, sütun (2) seçtiğiniz modeldir."
    return text


def _coef(term: str, quantity: str, expected: float, decimals: int, label: str) -> Check:
    return Check(label, CoefTarget("model", term, quantity), expected, decimals)


# --- Adımlar ---------------------------------------------------------------------------------------------------

_LOG_TABLE = ((5.00, 4.88, 0.12), (20.00, 18.23, 1.77), (50.00, 40.55, 9.45))
"""Tablo 0.3: tam yüzde değişim, log farkı ×100 ve fark (100 → 105, 120, 150)."""

STEPS = (
    interactive_step(
        number=1,
        title="Veri tablosunun anatomisi",
        note=NoteRef("0.1", objects=("Tablo 0.1",)),
        explanation=(
            "Her **satır** bir gözlem (bir çalışan), her **sütun** bir değişkendir. “Çalışan” sütunu yalnız gözlemi "
            "tanımlayan kimlik numarasıdır; analitik değişken sayılmaz. “Kadın” iki kategorili bir gösterge "
            "değişkenidir: kadın çalışanlar için 1, diğerleri için 0."
        ),
        controls=(INDICATOR_CHOICE,),
        build=_table,
        checks=(
            _scalar("n", 5, "Tablo 0.1: gözlem sayısı", 0),
            _scalar("k", 4, "§0.1: analitik değişken sayısı", 0),
            _scalar("kadin_payi", 0.40, "§0.1: kadın çalışanların payı", 2),
        ),
        note_for=lambda state, choices: _table_note(state, choices),
    ),
    interactive_step(
        number=2,
        title="Toplam sembolü ve aritmetik ortalama",
        note=NoteRef("0.3"),
        explanation=(
            "$x_i$, $x$ değişkeninin $i$ numaralı gözlemdeki değeridir. Toplam sembolü "
            "$\\sum_{i=1}^{n} x_i = x_1 + x_2 + \\cdots + x_n$ bütün gözlemlerin toplamını, örneklem ortalaması "
            "$\\bar{x} = \\frac{1}{n}\\sum_{i=1}^{n} x_i$ bu toplamın gözlem sayısına bölümünü verir."
        ),
        controls=(VARIABLE_CHOICE,),
        build=_sum_mean,
        checks=(
            _scalar("toplam", 72, "§0.3.1: eğitim yıllarının toplamı", 0),
            _scalar("ortalama", 14.4, "§0.3.2: ortalama eğitim yılı", 1),
            _scalar("ortalama_yazilim", 14.4, "§0.3.2: yazılımla ortalama", 1),
        ),
        note_for=lambda state, choices: _sum_mean_note(state, choices),
    ),
    interactive_step(
        number=3,
        title="Sapma, varyans ve standart sapma",
        note=NoteRef("0.4", objects=("Tablo 0.2",)),
        explanation=(
            "Sapma $x_i - \\bar{x}$; sapmaların toplamı her zaman sıfırdır. Örneklem varyansı "
            "$s_x^2 = \\frac{1}{n-1}\\sum_{i=1}^{n}(x_i - \\bar{x})^2$, standart sapma $s_x = \\sqrt{s_x^2}$; "
            "sapma ve varyans Adım 2'de seçilen değişkenle hesaplanır. Tablo 0.2'nin iki veri setinin ortalaması aynı, "
            "yayılımı farklıdır. Ölçek $a$ ve kayma $c$, B'nin değerlerini $y_i = a\\,B_i + c$ ile dönüştürür."
        ),
        uses=(VARIABLE_CHOICE,),
        controls=(SCALE_CHOICE, SHIFT_CHOICE),
        build=_variance,
        checks=(
            *(Check(f"§0.4: {i}. gözlemin sapması", CellTarget(WORKERS, "sapma", i), value, 1)
              for i, value in enumerate((-2.4, 1.6, -0.4, 3.6, -2.4), start=1)),
            *(Check(f"§0.4: {i}. gözlemin kareli sapması", CellTarget(WORKERS, "sapma_kare", i), value, 2)
              for i, value in enumerate((5.76, 2.56, 0.16, 12.96, 5.76), start=1)),
            _scalar("sapma_toplami", 0, "§0.4: sapmaların toplamı", 0),
            _scalar("kare_toplami", 27.2, "§0.4: kareli sapmaların toplamı", 1),
            _scalar("varyans", 6.80, "§0.4: varyans s²", 2),
            _scalar("std_sapma", 2.61, "§0.4: standart sapma s", 2),
            _scalar("ort_A", 10, "Tablo 0.2: A ortalaması", 0),
            _scalar("ort_B", 10, "Tablo 0.2: B ortalaması", 0),
            _scalar("var_A", 2.50, "Tablo 0.2: A varyansı", 2),
            _scalar("var_B", 40.00, "Tablo 0.2: B varyansı", 2),
            _scalar("std_A", 1.58, "Tablo 0.2: A standart sapması", 2),
            _scalar("std_B", 6.32, "Tablo 0.2: B standart sapması", 2),
        ),
        note_for=lambda state, choices: _variance_note(state, choices),
    ),
    interactive_step(
        number=4,
        title="Korelasyonun yönü ve doğrusallık",
        note=NoteRef("0.5", objects=("Şekil 0.2",)),
        explanation=(
            "Kovaryans $s_{xy} = \\frac{1}{n-1}\\sum_{i=1}^{n}(x_i - \\bar{x})(y_i - \\bar{y})$ iki değişkenin "
            "birlikte "
            "hareketinin yönünü gösterir; korelasyon $r_{xy} = s_{xy}/(s_x s_y)$ onu standartlaştırır ve $-1$ ile $1$ "
            "arasındadır. Seçenekler Şekil 0.2'nin üç örüntüsü ve §0.5'teki karesel örnektir."
        ),
        controls=(PATTERN_CHOICE,),
        build=_patterns,
        checks=(
            _scalar("r_pozitif", 0.994, "Şekil 0.2: pozitif ilişki, r", 3),
            _scalar("r_negatif", -0.997, "Şekil 0.2: negatif ilişki, r", 3),
            _scalar("r_zayif", 0.019, "Şekil 0.2: zayıf doğrusal ilişki, r", 3),
            _scalar("r_karesel", 0, "§0.5: karesel ilişki, r", 3),
        ),
        note_for=lambda state, choices: _patterns_note(state, choices),
    ),
    interactive_step(
        number=5,
        title="Kovaryans, korelasyon ve ölçü birimi (WAGE1)",
        note=NoteRef("0.5"),
        explanation=(
            "WAGE1'deki 526 çalışan: saatlik ücret (dolar/saat) ile ikinci değişken arasındaki kovaryans ve "
            "korelasyon. Ücret sent olarak ölçülünce kovaryans 100 katına çıkar, korelasyon değişmez; logaritma gibi "
            "doğrusal olmayan bir dönüşüm ise korelasyonu genellikle değiştirir."
        ),
        controls=(SECOND_CHOICE,),
        build=_covariance,
        checks=(
            _scalar("kovaryans", 4.1509, "§0.5: ücret ile eğitimin kovaryansı", 4),
            _scalar("std_wage", 3.6931, "§0.5: ücretin standart sapması", 4),
            _scalar("std_x", 2.7690, "§0.5: eğitimin standart sapması", 4),
            _scalar("korelasyon", 0.406, "§0.5: korelasyon", 3),
            _scalar("korelasyon_yazilim", 0.406, "§0.5: yazılımla korelasyon", 3),
            _scalar("kovaryans_sent", 415.09, "§0.5: kovaryans, ücret sent", 2),
            _scalar("korelasyon_sent", 0.406, "§0.5: korelasyon, ücret sent", 3),
            _scalar("korelasyon_log", 0.431, "§0.5: ücretin logaritması ile korelasyon", 3),
        ),
        note_for=lambda state, choices: _covariance_note(state, choices),
    ),
    interactive_step(
        number=6,
        title="Yüzde değişim, yüzde puan ve log farkı",
        note=NoteRef("0.6", objects=("Tablo 0.3",)),
        explanation=(
            "Yüzde değişim $\\%\\Delta x = 100\\,(x_1 - x_0)/x_0$; küçük değişimlerde "
            "$100\\,[\\ln(x_1) - \\ln(x_0)]$ ona yaklaşır. Bir oran %30'dan %35'e çıkarsa artış 5 yüzde puandır; "
            "göreli yüzde değişim ise $100\\,(35 - 30)/30$."
        ),
        controls=(START_CHOICE, END_CHOICE, RATE0_CHOICE, RATE1_CHOICE),
        build=_percent,
        checks=(
            _scalar("yuzde_degisim", 20, "§0.6.1: 100 → 120 yüzde değişim", 0),
            _scalar("geri_donus", -16.67, "§0.6.1: 120 → 100 yüzde değişim", 2),
            _scalar("log_farki", 18.23, "§0.6.3: 100 → 120 log farkı", 2),
            _scalar("log_geri", -18.23, "§0.6.3: 120 → 100 log farkı", 2),
            _scalar("yuzde_puan", 5, "§0.6.2: yüzde puan", 0),
            _scalar("goreli_degisim", 16.67, "§0.6.2: göreli yüzde değişim", 2),
            *(Check(f"Tablo 0.3: 100 → {end}, {name}", CellTarget("log_tablo", column, row), value, 2)
              for row, (end, values) in enumerate(zip((105, 120, 150), _LOG_TABLE), start=1)
              for (column, name), value in zip((("tam", "tam yüzde değişim"), ("log_fark", "log farkı"),
                                                ("fark", "fark")), values)),
        ),
        note_for=lambda state, choices: _percent_note(state, choices),
    ),
    interactive_step(
        number=7,
        title="Beklenen değer ve koşullu ortalama",
        note=NoteRef("0.7"),
        explanation=(
            "$\\mathbb{E}(Y)$ anakütlenin teorik ortalamasıdır; $\\mathbb{E}(Y \\mid X = x)$, $X = x$ koşulunda "
            "$Y$'nin "
            "beklenen değeridir. Örneklemdeki karşılıkları ortalama ve koşula uyan gözlemlerin ortalamasıdır. "
            "Örnek: 16 yıl eğitimli çalışanların ortalama saatlik ücreti."
        ),
        controls=(LEVEL_CHOICE,),
        build=_conditional,
        checks=(
            _scalar("ortalama_ucret", 5.90, "§0.7: ortalama saatlik ücret", 2),
            Check("§0.7: eğitimi 8 yıl olanların ortalama ücreti", TableTarget("egitime_gore", 8, "mean"), 5.04, 2),
            Check("§0.7: eğitimi 12 yıl olanların ortalama ücreti", TableTarget("egitime_gore", 12, "mean"), 5.37, 2),
            Check("§0.7: eğitimi 16 yıl olanların ortalama ücreti", TableTarget("egitime_gore", 16, "mean"), 8.04, 2),
            _scalar("kosullu_ortalama", 8.04, "§0.7: E(ücret | eğitim = 16), örneklem karşılığı", 2),
        ),
        note_for=lambda state, choices: _conditional_note(state, choices),
    ),
    LabStep(
        number=8,
        title="Parametre, tahmin edici ve tahmin",
        note=NoteRef("0.8", objects=("Tablo 0.4",)),
        explanation=(
            "**Parametre** anakütleye ait bilinmeyen sayıdır (ör. anakütledeki ortalama ücret $\\mu$). **Tahmin "
            "edici** "
            "örneklemi sayıya dönüştüren kuraldır ($\\bar{X} = n^{-1}\\sum X_i$). **Tahmin** bu kuralın belirli bir "
            "örneklemde verdiği sayıdır. Aynı kural farklı örneklemlerde farklı tahminler verebilir."
        ),
        operations=(
            InlineData("tablo04", ("egitim",), tuple((row[2],) for row in WORKER_ROWS),
                       "Tablo 0.4'ün örneği: Tablo 0.1'deki beş çalışanın eğitim yılları", layout=5),
            Statistic("tablo04", "egitim", "mean", "tahmin_tablo", "Tahmin x̄ = Σxᵢ / n", decimals=1),
            Derive(DATA, "sira", E.seq(E.var("wage")), "Sıra numarası 1, 2, …, 526"),
            Derive(DATA, "tek", E.compare("eq", E.sub(E.var("sira"), E.mul(2, E.floor(E.div(E.var("sira"), 2)))), 1),
                   "Tek sıra numaralı çalışan (1)"),
            Derive(DATA, "cift", E.sub(1, E.var("tek")), "Çift sıra numaralı çalışan (1)"),
            Derive(DATA, "besinci", E.compare("eq", E.sub(E.var("sira"), E.mul(5, E.floor(E.div(
                E.sub(E.var("sira"), 1), 5)))), 1), "Her beşinci çalışan: 1., 6., 11., … (1)"),
            Statistic(DATA, "wage", "mean", "tahmin_tum", "526 çalışan", decimals=2),
            Statistic(DATA, "wage", "mean", "tahmin_tek", "Tek sıradakiler (263)", where=("tek", 1),
                      decimals=2),
            Statistic(DATA, "wage", "mean", "tahmin_cift", "Çift sıradakiler (263)", where=("cift", 1),
                      decimals=2),
            Statistic(DATA, "wage", "mean", "tahmin_besinci", "Her beşinci çalışan (106)",
                      where=("besinci", 1), decimals=2),
            ScalarTable(
                (
                    ("526 çalışan (bütün örneklem)", E.ref("tahmin_tum")),
                    ("Tek sıra numaralı 263 çalışan", E.ref("tahmin_tek")),
                    ("Çift sıra numaralı 263 çalışan", E.ref("tahmin_cift")),
                    ("Her beşinci çalışan (106 çalışan)", E.ref("tahmin_besinci")),
                ),
                "tahminler",
                decimals=2,
            ),
        ),
        checks=(
            _scalar("tahmin_tablo", 14.4, "Tablo 0.4: tahmin x̄", 1),
            _scalar("tahmin_tum", 5.90, "Egzersiz 0.8: 526 kişilik örneklemde ortalama ücret", 2),
        ),
        takeaway=(
            "Dört satırın hepsi aynı tahmin edicinin (örneklem ortalaması) sonucudur; kullanılan gözlemler değişince "
            "tahmin de değişebilir. Buradaki alt örneklemler aynı veriden alınır; anakütleden her seferinde yeni bir "
            "örneklem çekmenin sonucunu Sezgi sekmesindeki Deney 1 gösterir. Anakütledeki ortalama ücret μ tek bir "
            "sayıdır ve bilinmez. Örneklemden örnekleme değişim istatistiksel belirsizliğin kaynağıdır; ölçülmesi "
            "(standart hata) Konu 7'nin konusudur. $H_0: \\beta_1 = 0$ gibi hipotezler de parametreler hakkındadır "
            "(§0.8)."
        ),
    ),
    interactive_step(
        number=9,
        title="İlk Python regresyon çıktısını tanıma",
        note=NoteRef("0.9", objects=("Kod 0.1", "Kod 0.2")),
        explanation=(
            "`wage ~ educ` yazımı saatlik ücreti bağımlı, eğitim yılını açıklayıcı değişken yapar. Kodun ezberlenmesi "
            "beklenmez; çıktıda bağımlı değişken (`Dep. Variable`), gözlem sayısı (`No. Observations`), katsayılar "
            "(`coef`) ve $R^2$ (`R-squared`) okunur."
        ),
        controls=(X_CHOICE,),
        build=_regression,
        checks=(
            _coef(INTERCEPT, "coef", -0.9049, 4, "Kod 0.2: sabit terim"),
            _coef("educ", "coef", 0.5414, 4, "Kod 0.2: eğitim katsayısı"),
            _coef(INTERCEPT, "se", 0.685, 3, "Kod 0.2: sabit terimin standart hatası"),
            _coef("educ", "se", 0.053, 3, "Kod 0.2: eğitim katsayısının standart hatası"),
            _coef(INTERCEPT, "t", -1.321, 3, "Kod 0.2: sabit terimin t değeri"),
            _coef("educ", "t", 10.167, 3, "Kod 0.2: eğitim katsayısının t değeri"),
            _coef(INTERCEPT, "p", 0.187, 3, "Kod 0.2: sabit terimin p-değeri"),
            _coef("educ", "p", 0.000, 3, "Kod 0.2: eğitim katsayısının p-değeri"),
            _coef(INTERCEPT, "ci_low", -2.250, 3, "Kod 0.2: sabit terim, %95 GA alt sınır"),
            _coef(INTERCEPT, "ci_high", 0.441, 3, "Kod 0.2: sabit terim, %95 GA üst sınır"),
            _coef("educ", "ci_low", 0.437, 3, "Kod 0.2: eğitim, %95 GA alt sınır"),
            _coef("educ", "ci_high", 0.646, 3, "Kod 0.2: eğitim, %95 GA üst sınır"),
            Check("Kod 0.2: R²", ModelTarget("model", "r2"), 0.165, 3),
            Check("Kod 0.2: F istatistiği", ModelTarget("model", "f"), 103.4, 1),
            Check("Kod 0.2: gözlem sayısı", ModelTarget("model", "nobs"), 526, 0),
            _scalar("var_x", 7.6675, "§0.9: eğitimin varyansı", 4),
            _scalar("kov_wx", 4.1509, "§0.9: ücret ile eğitimin kovaryansı", 4),
            _scalar("egim_formul", 0.5414, "§0.9: eğim = kovaryans / varyans", 4),
            _scalar("r_kare", 0.165, "§0.9: R² = r²", 3),
        ),
        note_for=lambda state, choices: _regression_note(state, choices),
    ),
)


KONU00_LAB = LabSpec(
    topic_key="konu00",
    title="Uygulama: Veri, Notasyon ve Temel İstatistik",
    note_section="0",
    steps=STEPS,
    labels=(
        *W.labels(DATA),
        ("calisan", "Çalışan (kimlik)"), ("ucret", "Saatlik ücret"), ("egitim", "Eğitim yılı"),
        ("deneyim", "Deneyim yılı"), ("kadin", "Kadın (1/0)"),
        ("sapma", "Sapma xᵢ − x̄"), ("sapma_kare", "Kareli sapma"), ("gozlem", "Gözlem"),
        ("A", "Veri seti A"), ("B", "Veri seti B"), ("y", "y"), ("x", "x"),
        ("pozitif", "Pozitif ilişki"), ("negatif", "Negatif ilişki"), ("zayif", "Zayıf doğrusal ilişki"),
        ("x0", "Başlangıç x₀"), ("x1", "Yeni değer x₁"), ("tam", "Tam yüzde değişim"),
        ("log_fark", "Log farkı ×100"), ("fark", "Fark"), ("wage_sent", "Saatlik ücret (sent)"),
        ("sira", "Sıra numarası"), ("tek", "Tek sıra (1/0)"), ("cift", "Çift sıra (1/0)"),
        ("besinci", "Her beşinci (1/0)"), ("count", "Çalışan sayısı"), ("mean", "Ortalama saatlik ücret"),
        ("deger", "Değer"), (INTERCEPT, "Sabit terim"), ("n", "Gözlem sayısı"), ("k", "Analitik değişken sayısı"),
    ),
    consistency_notes=(
        "Bölüm 0'ın laboratuvar bölümü yoktur; adımlar bölümün çözümlü örnekleridir ve alt bölümlere bağlıdır.",
        "Notlar bu sürümde slaytlarla eşlendi: §0.1'e gösterge ortalaması (0,40), §0.3'e toplam sembolünün kuralları, "
        "§0.4'e eğitim örneğinin varyansı ve Tablo 0.2'ye s² ile s, §0.5'e WAGE1 kovaryansı ve korelasyonu ile Şekil "
        "0.2'nin r değerleri, §0.6'ya Tablo 0.3, §0.7'ye WAGE1 koşullu ortalamaları, §0.9'a eğim = kovaryans / varyans "
        "eklendi.",
        "Bölüm 0'da tablo ve şekiller önceki sürümde Tablo 1–3 ve Şekil 1–2 diye numaralanıyordu; artık Tablo 0.1–0.4 "
        "ve Şekil 0.1–0.2'dir. Kod 0.1 veriyi wooldridge paketinden okur.",
    ),
)
