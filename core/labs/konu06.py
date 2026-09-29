"""Konu 6 uygulaması: EKK varsayımları, eksik değişken yanlılığı ve çoklu doğrusal bağlantı (gerçek veri).

Bölüm 6'nın gerçek veriyle çözümlü örnekleri bölüm sırasıyla: WAGE1'de kısa ve uzun ücret modeli (§6.8, Denklem
6.10–6.11), yardımcı regresyon ve eğitim katsayısının örneklem ayrıştırması (§6.8, Denklem 6.12), kontroller
eklendikçe eğitim katsayısı ve Python çıktısı (§6.8, Tablo 6.6, Şekil 6.3, Kod 6.3), tam çoklu doğrusal bağlantı
(§6.9), HPRICE1'de VIF (§6.12, Kod 6.4, Tablo 6.9) ve bütünleşik okuma listesi (§6.13). Bölümün benzetimleri
(Tablo 6.3, 6.5, 6.7 ve 6.8) Sezgi sekmesindeki Konu 6 deneyleridir. Her ``Check`` notlarda basılı bir sayıdır;
değer notlardan kopyalanmıştır, hesaplanmamıştır.

Etkileşim: kısa modelde dışarıda kalan değişken (Adım 1; Adım 2 aynı değişkeni kullanır), Tablo 6.6'nın sütunları
(Adım 3) ve VIF hesabındaki açıklayıcı değişkenler (Adım 5). Kod 6.3'teki standart hata, t ve p-değeri notlardaki gibi
yazılım çıktısında görünür; yorumları Konu 7'dedir.
"""

from __future__ import annotations

from core import wooldridge_data as W
from core.labs import expr as E
from core.labs.sezgi import plain
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
    LoadWooldridge,
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

DATA = "wage1"
HPRICE = "hprice1"
OUTCOME = "wage"
OMITTED_OPTIONS = ("exper", "tenure", "numdep")
"""Kısa modelde dışarıda kalan, uzun modele eklenen değişkenin seçenekleri (kukla olmayan sayısal değişkenler)."""
WORDS = {"exper": "deneyim", "tenure": "kıdem", "numdep": "bakmakla yükümlü kişi sayısı"}
TITLES = {"exper": "Deneyim", "tenure": "Kıdem", "numdep": "Bakmakla yükümlü kişi sayısı"}
UNITS = {"exper": "yıl", "tenure": "yıl", "numdep": "kişi"}
COLUMNS = {
    "yalniz": ("Yalnız eğitim", ("educ",)),
    "deneyim": ("Eğitim + deneyim", ("educ", "exper")),
    "tam": ("Eğitim + deneyim + kıdem", ("educ", "exper", "tenure")),
    "kidem": ("Eğitim + kıdem", ("educ", "tenure")),
    "genis": ("Eğitim + deneyim + kıdem + bakmakla yükümlü kişi sayısı", ("educ", "exper", "tenure", "numdep")),
}
"""Tablo 6.6'nın seçilebilen sütunları: (başlık, açıklayıcı değişkenler)."""
COLUMNS_DEFAULT = ("yalniz", "deneyim", "tam")
"""Notlardaki Tablo 6.6."""
VIF_OPTIONS = ("sqrft", "bdrms", "lotsize", "assess", "lsqrft")
VIF_DEFAULT = ("sqrft", "bdrms", "lotsize")
"""Notlardaki VIF örneği (§6.12): Denklem 5.7'deki konut fiyatı modelinin açıklayıcı değişkenleri."""
VIF_NAMES = {"sqrft": "Konut büyüklüğü", "bdrms": "Yatak odası sayısı", "lotsize": "Arsa büyüklüğü",
             "assess": "Vergi değeri", "lsqrft": "Konut büyüklüğünün logaritması"}

Z_CHOICE = Choice("adim1_z", "Kısa modelde dışarıda kalan, uzun modele eklenen değişken", W.options(DATA, OMITTED_OPTIONS),
                  "exper", help="Notlarda deneyim (Denklem 6.11). Adım 2 aynı değişkeni kullanır.")


def _scalar(name: str, expected: float, label: str, decimals: int) -> Check:
    return Check(label, ScalarTarget(name), expected, decimals)


def _listing(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " ve " + items[-1]


# --- Adım 1: kısa ve uzun model --------------------------------------------------------------

def _short_long(choices) -> tuple:
    z = choices["adim1_z"]
    notes = z == "exper"
    columns = [("(1) Kısa model", "kisa"), ("(2) Uzun model", "uzun")]
    extra: tuple = ()
    if not notes:
        extra = (OLS("uzun_notlar", DATA, OUTCOME, ("educ", "exper"),
                     "Karşılaştırma için notlardaki uzun model: wage ~ educ + exper (Denklem 6.11)"),)
        columns.append(("(3) Notlardaki uzun model", "uzun_notlar"))
    terms = ("educ", z, INTERCEPT) if notes else ("educ", z, "exper", INTERCEPT)
    return (
        LoadWooldridge(DATA, "WAGE1 veri seti (Wooldridge, 2020): 526 çalışan"),
        OLS("kisa", DATA, OUTCOME, ("educ",), "Kısa model: wage ~ educ (Denklem 6.10)"),
        ModelValue("kisa_egitim", "kisa", "coef", "Kısa model: eğitim katsayısı", term="educ"),
        OLS("uzun", DATA, OUTCOME, ("educ", z), f"Uzun model: wage ~ educ + {z}" + (" (Denklem 6.11)" if notes else "")),
        ModelValue("uzun_egitim", "uzun", "coef", "Uzun model: eğitim katsayısı", term="educ"),
        ModelValue("uzun_z", "uzun", "coef", f"Uzun model: {WORDS[z]} katsayısı", term=z),
        *extra,
        RegressionTable(tuple(columns), terms, "kisa_uzun",
                        f"Kısa ve uzun ücret modeli: {WORDS[z]} dışarıda / içeride (bağımlı değişken: saatlik ücret)"
                        if notes else f"Kısa model, {WORDS[z]} eklenmiş uzun model ve notlardaki uzun model (bağımlı "
                        "değişken: saatlik ücret)",
                        stars=False, decimals=4, standard_errors=False),
    )


def _short_long_note(state, choices) -> str:
    z = choices["adim1_z"]
    s = state.scalars
    change = s["uzun_egitim"] - s["kisa_egitim"]
    return (f"Eğitim katsayısı kısa modelde {plain(s['kisa_egitim'], 4)}, {WORDS[z]} eklenince "
            f"{plain(s['uzun_egitim'], 4)}; fark {plain(change, 4)}. Kısa modelde {WORDS[z]} hata teriminde kalır. "
            f"Eksik değişken yanlılığı için iki koşul birlikte gerekir: dışarıda kalan değişken ücretle ilişkili olmalı ve "
            f"eğitimle ilişkili olmalıdır (§6.6). Uzun modelde {WORDS[z]} katsayısı {plain(s['uzun_z'], 4)}; eğitimle "
            "ilişkisini Adım 2'deki yardımcı regresyon verir. Katsayının değişmesi tek başına yanlılık kanıtı değildir."
            + ("" if z == "exper" else " Notlarda deneyim eklenince eğitim katsayısı 0,5414 → 0,6443 (sütun 3)."))


# --- Adım 2: yardımcı regresyon ve ayrıştırma ---------------------------------------------

def _decomposition(choices) -> tuple:
    z = choices["adim1_z"]
    notes = z == "exper"
    return (
        OLS("yardimci", DATA, z, ("educ",), f"Yardımcı regresyon: {z} ~ educ" + (" (Denklem 6.12)" if notes else "")),
        ModelValue("yardimci_sabit", "yardimci", "coef", "Yardımcı regresyon: sabit terim", term=INTERCEPT),
        ModelValue("yardimci_egim", "yardimci", "coef", f"Yardımcı eğim δ̂ ({WORDS[z]} ~ eğitim)", term="educ"),
        Scalar("katki", E.mul(E.ref("uzun_z"), E.ref("yardimci_egim")),
               f"Dışarıda kalan değişkenin katkısı: β̂ ({WORDS[z]}) × δ̂", decimals=4),
        Scalar("yeniden", E.add(E.ref("uzun_egitim"), E.ref("katki")),
               "Uzun modelin eğitim katsayısı + katkı", decimals=4),
        ScalarTable((
            ("Uzun model: eğitim katsayısı", E.ref("uzun_egitim")),
            (f"Katkı: β̂ ({WORDS[z]}) × δ̂", E.ref("katki")),
            ("Kısa model: eğitim katsayısı", E.ref("kisa_egitim")),
        ), "ayristirma", decimals=4, heading="Bileşen", value="Katsayı (dolar)"),
        BarChart("ayristirma", "deger", "Bileşen", "Katsayı (dolar)",
                 "Kısa model katsayısı = uzun model katsayısı + dışarıda kalan değişkenin katkısı", decimals=4),
    )


def _decomposition_note(state, choices) -> str:
    z = choices["adim1_z"]
    s = state.scalars
    gamma, delta = s["uzun_z"], s["yardimci_egim"]
    unit = UNITS[z]
    if unit == "kişi":
        relation = "fazladır" if delta > 0 else "azdır"
    else:
        relation = "yüksektir" if delta > 0 else "düşüktür"
    same = gamma * delta > 0
    return (f"Eğitim yılı bir yıl daha yüksek çalışanlarda {WORDS[z]} örneklemde ortalama yaklaşık "
            f"{plain(abs(delta), 3)} {unit} daha {relation}. {TITLES[z]} katsayısı "
            f"{'pozitif' if gamma > 0 else 'negatif'}, eğitimle ilişkisi {'pozitif' if delta > 0 else 'negatif'}: "
            f"işaretler {'aynı' if same else 'farklı'}, katkı {plain(s['katki'], 4)}. İşaret tablosuna göre "
            f"{WORDS[z]} değişkeninin dışarıda bırakılması eğitim katsayısını {'yukarı' if same else 'aşağı'} yönlü "
            f"etkileyebilir (Tablo 6.4). Örneklemde ilişki tam olarak sağlanır: {plain(s['kisa_egitim'], 4)} = "
            f"{plain(s['uzun_egitim'], 4)} + ({plain(s['katki'], 4)})."
            + ("" if z == "exper" else " Notlarda (deneyim dışarıda): 0,5414 = 0,6443 + (−0,1029).")
            + " Bu ayrıştırma katsayı değişiminin mekanizmasını gösterir; uzun modelin nedensel model olduğunu "
            "kanıtlamaz: yetenek, meslek veya eğitim kalitesi hâlâ hata teriminde olabilir.")


# --- Adım 3: kontroller eklendikçe eğitim katsayısı ----------------------------------------

def _controls(choices) -> tuple:
    selected = choices["adim3_modeller"]
    notes = tuple(selected) == COLUMNS_DEFAULT
    used = {name for key in selected for name in COLUMNS[key][1]}
    terms = tuple(name for name in ("educ", "exper", "tenure", "numdep") if name in used)
    return (
        *(OLS(f"m_{key}", DATA, OUTCOME, COLUMNS[key][1], f"{COLUMNS[key][0]}: wage ~ {' + '.join(COLUMNS[key][1])}")
          for key in selected),
        *(ModelValue(f"egitim_{key}", f"m_{key}", "coef", f"Eğitim katsayısı: {COLUMNS[key][0].lower()}",
                     term="educ") for key in selected),
        RegressionTable(tuple((COLUMNS[key][0], f"m_{key}") for key in selected), terms, "tablo66",
                        "Tablo 6.6: WAGE1 ücret modellerinde eğitim katsayısının değişimi" if notes
                        else "Seçtiğiniz modellerle eğitim katsayısının değişimi", stars=False, decimals=4,
                        standard_errors=False),
        ScalarTable(tuple((COLUMNS[key][0], E.ref(f"egitim_{key}")) for key in selected), "egitim_katsayilari",
                    decimals=3, heading="Model", value="Eğitim katsayısı"),
        BarChart("egitim_katsayilari", "deger", "Model", "Eğitim katsayısı (dolar)",
                 "Şekil 6.3: WAGE1'de farklı modellerin eğitim katsayıları" if notes
                 else "Seçtiğiniz modellerde eğitim katsayıları", decimals=3),
        OLS("kod63", DATA, OUTCOME, ("educ", "exper"), "Kod 6.3: deneyim kontrollü ücret modeli"),
        ShowModel("kod63", "Kod 6.3: deneyim kontrollü modelin temel çıktısı", columns=("coef", "se", "t", "p"),
                  stats=("nobs", "r2"), stars=False),
    )


def _controls_note(state, choices) -> str:
    selected = choices["adim3_modeller"]
    table = state.tables["egitim_katsayilari"]["deger"]
    path = " → ".join(plain(value, 3) for value in table)
    changes = []
    for before, after, old, new in zip(selected, selected[1:], table.iloc[:-1], table.iloc[1:]):
        added = [WORDS[name] for name in COLUMNS[after][1] if name not in COLUMNS[before][1]]
        removed = [WORDS[name] for name in COLUMNS[before][1] if name not in COLUMNS[after][1]]
        how = " ve ".join(([f"{_listing(added)} eklenince"] if added else []) +
                          ([f"{_listing(removed)} çıkarılınca"] if removed else []))
        changes.append(f"{how} {'yükselir' if new > old else 'düşer' if new < old else 'değişmez'}")
    moves = "; ".join(changes)
    start = f"Eğitim katsayısı sütunlar boyunca: {path}." if len(selected) > 1 else f"Eğitim katsayısı: {path}."
    return (start + (f" {moves[0].upper()}{moves[1:]}." if moves else "") +
            " Her sütun farklı bir karşılaştırma yapar; katsayı değişimi otomatik olarak “son model doğrudur” demek "
            "değildir. Hangi karşılaştırmanın uygun olduğu iktisadi gerekçeyle tartışılır. Kod 6.3'te bu aşamada "
            "yalnız `coef`, bağımlı değişken, gözlem sayısı ve R² okunur; `std err`, `t` ve `P>|t|` Konu 7'de işlenir."
            + ("" if len(selected) > 1 else " Karşılaştırma için en az iki model seçin."))


# --- Adım 4: tam çoklu doğrusal bağlantı ---------------------------------------------------

PAIRS = ((12, 0), (0, 1), (6, 0.5), (24, -1))
"""§6.9'daki katsayı çiftleri: hepsi β₁ + 12β₂ = 12."""


def _perfect() -> tuple:
    return (
        InlineData("ciftler", ("beta1", "beta2"), PAIRS, "Aylık ve yıllık gelir modelinde katsayı çiftleri (§6.9)"),
        Derive("ciftler", "birlesim", E.add(E.var("beta1"), E.mul(12, E.var("beta2"))), "β₁ + 12β₂"),
        ShowFrame("ciftler", ("beta1", "beta2", "birlesim"), "Dört çift, aynı birleşim"),
        Derive(DATA, "egitim_ay", E.mul(12, E.var("educ")), "Eğitim, ay cinsinden: 12 × eğitim yılı"),
        PairStatistic(DATA, "educ", "egitim_ay", "corr", "r_ay", "Eğitim yılı ile eğitim ayının korelasyonu",
                      decimals=4),
    )


# --- Adım 5: gerçek veride VIF (HPRICE1) ---------------------------------------------------

def _vif(choices) -> tuple:
    chosen = choices["adim5_x"]
    notes = tuple(chosen) == VIF_DEFAULT
    operations: list = [LoadWooldridge(HPRICE, "HPRICE1 veri seti (Wooldridge, 2020): 88 konut")]
    for name in chosen:
        others = tuple(other for other in chosen if other != name)
        operations += [
            OLS(f"yardimci_{name}", HPRICE, name, others,
                f"{VIF_NAMES[name]} diğer açıklayıcı değişkenlere göre: {name} ~ {' + '.join(others)}"),
            ModelValue(f"r2_{name}", f"yardimci_{name}", "r2", f"R²ⱼ: {VIF_NAMES[name].lower()}", decimals=4),
        ]
    operations += [
        ScalarTable(tuple((VIF_NAMES[name], E.div(1, E.sub(1, E.ref(f"r2_{name}")))) for name in chosen), "vif",
                    decimals=3, heading="Açıklayıcı değişken", value="VIF"),
        BarChart("vif", "deger", "Açıklayıcı değişken", "VIF = 1 / (1 − R²ⱼ)",
                 "HPRICE1 konut fiyatı modelinde VIF (Tablo 6.9'daki değerler)" if notes
                 else "Seçtiğiniz modelde VIF değerleri", decimals=3),
    ]
    if {"sqrft", "bdrms"} <= set(chosen):
        operations.append(PairStatistic(HPRICE, "sqrft", "bdrms", "corr", "r_buyukluk_oda",
                                        "Konut büyüklüğü ile yatak odası sayısının korelasyonu", decimals=2))
    if "assess" in chosen:
        size = _size_variable(chosen)
        if size is not None:
            owner = {"sqrft": "konut büyüklüğünün", "lsqrft": "konut büyüklüğünün logaritmasının"}[size]
            operations.append(PairStatistic(HPRICE, "assess", size, "corr", "r_vergi_buyukluk",
                                            f"Vergi değeri ile {owner} korelasyonu", decimals=2))
        operations.append(PairStatistic(HPRICE, "assess", "price", "corr", "r_vergi_fiyat",
                                        "Vergi değeri ile fiyatın korelasyonu", decimals=2))
    return tuple(operations)


def _size_variable(chosen) -> str | None:
    """Modeldeki büyüklük ölçüsü: konut büyüklüğü, yoksa logaritması."""

    return next((name for name in ("sqrft", "lsqrft") if name in chosen), None)


def _vif_note(state, choices) -> str:
    chosen = choices["adim5_x"]
    table = state.tables["vif"]["deger"]
    if len(chosen) == 2:
        # İki değişkende iki yardımcı regresyonun R²'si de r²'dir: iki VIF eşittir (seçim yuvarlama gürültüsüne kalmasın).
        text = (f"İki değişkenli modelde iki VIF eşittir: 1/(1 − r²) = {plain(table.iloc[0], 3)}. VIF, bir açıklayıcı "
                "değişkenin diğerleri tarafından ne kadar iyi açıklandığını özetler: R²ⱼ bire yaklaştıkça VIF büyür ve "
                "değişkenin diğerlerinden bağımsız kalan hareketi azalır.")
    else:
        largest = table.idxmax()
        text = (f"En büyük VIF {plain(table.max(), 3)} ({largest.lower()}). VIF, bir açıklayıcı değişkenin diğerleri "
                "tarafından ne kadar iyi açıklandığını özetler: R²ⱼ bire yaklaştıkça VIF büyür ve değişkenin "
                "diğerlerinden bağımsız kalan hareketi azalır.")
    if tuple(chosen) == VIF_DEFAULT:
        text += (f" Konut büyüklüğü ile yatak odası sayısı ilişkilidir (korelasyon {plain(state.scalars['r_buyukluk_oda'], 2)}) "
                 "ama aynı bilgiyi taşımaz; ayrı katsayılar için veride yeterince bağımsız hareket vardır. Tablo 6.7'deki "
                 "yüksek VIF'ler değişkenlerin neredeyse aynı bilgiyi taşıdığı durumda ortaya çıkar.")
    if tuple(chosen) != VIF_DEFAULT:
        text += " Notlardaki modelde (büyüklük, oda, arsa) VIF'ler 1,419; 1,397 ve 1,037'dir (Tablo 6.9)."
    s = state.scalars
    if "assess" in chosen and _size_variable(chosen) is not None:
        size = VIF_NAMES[_size_variable(chosen)].lower()
        text += (f" Vergi değeri (`assess`) ile {size} güçlü ilişkilidir (korelasyon "
                 f"{plain(s['r_vergi_buyukluk'], 2)}): ikisi birlikte olunca VIF'leri belirgin biçimde yükselir.")
    if "assess" in chosen:
        text += (f" Vergi değeri fiyatla da çok ilişkilidir (korelasyon {plain(s['r_vergi_fiyat'], 2)}); kontrol "
                 "olarak uygun olup olmadığı araştırma sorusuna bağlıdır.")
    if "lsqrft" in chosen and "sqrft" in chosen:
        text += (" Konut büyüklüğü ile logaritması birlikte olunca VIF çok büyür: iki değişken neredeyse aynı bilgiyi "
                 "taşır; yine de logaritma doğrusal bir dönüşüm olmadığı için tam bağlantı yoktur.")
    return text + " VIF bir uyarı ölçüsüdür, mekanik bir silme kuralı değildir."


# --- Tanım ---------------------------------------------------------------------------

_TABLE66 = (
    ("Yalnız eğitim", {"educ": 0.5414, "r2": 0.1648}),
    ("Eğitim + deneyim", {"educ": 0.6443, "exper": 0.0701, "r2": 0.2252}),
    ("Eğitim + deneyim + kıdem", {"educ": 0.5990, "exper": 0.0223, "tenure": 0.1693, "r2": 0.3064}),
)
_ROW_NAMES = {"educ": "eğitim katsayısı", "exper": "deneyim katsayısı", "tenure": "kıdem katsayısı", "r2": "R²"}
_KOD63 = (
    (INTERCEPT, "sabit terim", (-3.3905, 0.767, -4.423, 0.000)),
    ("educ", "eğitim", (0.6443, 0.054, 11.974, 0.000)),
    ("exper", "deneyim", (0.0701, 0.011, 6.385, 0.000)),
)
_QUANTITIES = (("coef", "katsayı", 4), ("se", "standart hata", 3), ("t", "t", 3), ("p", "p-değeri", 3))
_TABLE69 = (("Konut büyüklüğü", "sqrft", 0.2951, 1.419), ("Yatak odası sayısı", "bdrms", 0.2840, 1.397),
            ("Arsa büyüklüğü", "lotsize", 0.0359, 1.037))

STEPS = (
    interactive_step(
        number=1,
        title="Kısa ve uzun model: dışarıda kalan değişken",
        note=NoteRef("6.8", 0, ("Denklem 6.10", "Denklem 6.11")),
        explanation=(
            "Ücret yalnız eğitime göre tahmin edildiğinde (kısa model) deneyim hata teriminde kalır. Deneyim "
            "eklendiğinde (uzun model) eğitim katsayısı değişir. Kısa modelde dışarıda kalan değişkeni seçin; Adım 2 "
            "değişimin örneklemdeki ayrıştırmasını gösterir."
        ),
        controls=(Z_CHOICE,),
        build=_short_long,
        checks=(
            Check("Denklem 6.10: sabit terim", CoefTarget("kisa", INTERCEPT), -0.9049, 4),
            Check("Denklem 6.10: eğitim katsayısı", CoefTarget("kisa", "educ"), 0.5414, 4),
            Check("Denklem 6.11: sabit terim", CoefTarget("uzun", INTERCEPT), -3.3905, 4),
            Check("Denklem 6.11: eğitim katsayısı", CoefTarget("uzun", "educ"), 0.6443, 4),
            Check("Denklem 6.11: deneyim katsayısı", CoefTarget("uzun", "exper"), 0.0701, 4),
            _scalar("kisa_egitim", 0.541, "§6.8: kısa model, 0,541", 3),
            _scalar("uzun_egitim", 0.644, "§6.8: uzun model, 0,644", 3),
        ),
        note_for=lambda state, choices: _short_long_note(state, choices),
    ),
    interactive_step(
        number=2,
        title="Yardımcı regresyon ve katsayının örneklem ayrıştırması",
        note=NoteRef("6.8", 0, ("Denklem 6.12", "Denklem 6.8")),
        explanation=(
            "Dışarıda kalan değişken eğitime göre tahmin edilir (yardımcı regresyon, Denklem 6.12). Örneklem "
            "katsayıları arasında tam olarak kısa katsayı = uzun katsayı + (dışarıda kalanın katsayısı) × (yardımcı "
            "eğim) ilişkisi vardır; Denklem 6.8'deki $\\beta_1 + \\beta_2\\delta_1$ yapısının örneklem karşılığı. Bu "
            "adım Adım 1'deki değişkeni kullanır."
        ),
        uses=(Z_CHOICE,),
        build=_decomposition,
        checks=(
            _scalar("yardimci_sabit", 35.4615, "Denklem 6.12: sabit terim", 4),
            _scalar("yardimci_egim", -1.4682, "Denklem 6.12: yardımcı eğim", 4),
            _scalar("yardimci_egim", -1.468, "§6.8: yaklaşık 1,468 yıl daha düşük", 3),
            _scalar("katki", -0.1029, "§6.8: (0,0701)(−1,4682)", 4),
            _scalar("yeniden", 0.5414, "§6.8: 0,6443 − 0,1029", 4),
        ),
        note_for=lambda state, choices: _decomposition_note(state, choices),
    ),
    interactive_step(
        number=3,
        title="Kontroller eklendikçe eğitim katsayısı ve Python çıktısı",
        note=NoteRef("6.8", 0, ("Tablo 6.6", "Şekil 6.3", "Kod 6.3")),
        explanation=(
            "Tablo 6.6 eğitim katsayısını üç modelde yan yana gösterir: yalnız eğitim, eğitim ve deneyim, eğitim, "
            "deneyim ve kıdem. Tablonun sütunlarını seçin. Kod 6.3 deneyim kontrollü modelin temel çıktısıdır; bu "
            "aşamada yalnız katsayılar, bağımlı değişken, gözlem sayısı ve R² okunur."
        ),
        controls=(MultiChoice("adim3_modeller", "Tablodaki modeller",
                              tuple((key, heading) for key, (heading, _) in COLUMNS.items()), COLUMNS_DEFAULT,
                              help="Notlardaki Tablo 6.6: ilk üç model.", maximum=4),),
        build=_controls,
        checks=(
            *(Check(f"Tablo 6.6: {heading}, {_ROW_NAMES[row]}", TableTarget("tablo66", row, heading), value, 4)
              for heading, values in _TABLE66 for row, value in values.items()),
            *(Check(f"Şekil 6.3: {heading}", TableTarget("egitim_katsayilari", heading, "deger"), value["educ"], 3)
              for heading, value in _TABLE66),
            *(Check(f"Kod 6.3: {name}, {label}", CoefTarget("kod63", term, quantity), value, decimals)
              for term, name, values in _KOD63 for (quantity, label, decimals), value in zip(_QUANTITIES, values)),
            Check("Kod 6.3: gözlem sayısı", ModelTarget("kod63", "nobs"), 526, 0),
            Check("Kod 6.3: R²", ModelTarget("kod63", "r2"), 0.225, 3),
        ),
        note_for=lambda state, choices: _controls_note(state, choices),
    ),
    LabStep(
        number=4,
        title="Tam çoklu doğrusal bağlantı: aynı bilgiyi iki kez vermek",
        note=NoteRef("6.9", 0, ("Denklem 6.13",)),
        explanation=(
            "Yıllık gelir her zaman aylık gelirin 12 katıysa $\\beta_1\\,\\text{aylık} + \\beta_2(12\\,\\text{aylık}) "
            "= (\\beta_1 + 12\\beta_2)\\,\\text{aylık}$ olur: veri yalnız $\\beta_1 + 12\\beta_2$ birleşimini "
            "belirler. Notlardaki dört çift aynı tahmini üretir. WAGE1'de eğitim yılı ile eğitim ayı (12 × yıl) aynı "
            "bilgiyi taşır; korelasyonları 1'dir. İkisi birlikte modele konamaz: uygulama böyle bir modeli “tam "
            "doğrusal bağlantı” uyarısıyla reddeder; R bir katsayıyı NA raporlar. statsmodels ise hata vermeden bir "
            "ayrıştırma raporlar (eğitim 0,0037, eğitim ayı 0,0448: 0,0037 + 12 × 0,0448 ≈ 0,541) ve yalnız özetin "
            "sonunda tekil tasarım notu yazar: veri yalnız birleşimi belirler, ayrışım keyfîdir."
        ),
        operations=_perfect(),
        checks=tuple(Check(f"§6.9: {i}. çift, β₁ + 12β₂", CellTarget("ciftler", "birlesim", i), 12, 0)
                     for i in range(1, len(PAIRS) + 1)),
        takeaway=(
            "Tam bağlantıda katsayılar “yanlış” olmadan önce ayrı ayrı tanımlanamaz: aynı tahmini veren sonsuz sayıda "
            "katsayı çifti vardır. Sık kaynaklar: aynı değişkenin farklı birimleri, toplam ile bütün parçaları, sabit "
            "terimle birlikte bütün kategori göstergeleri (kukla değişken tuzağı, Konu 10). A3 yalnız tam tekrarı "
            "yasaklar; değişkenlerin ilişkili olması olağandır."
        ),
    ),
    interactive_step(
        number=5,
        title="Gerçek veride VIF: HPRICE1",
        note=NoteRef("6.12", 0, ("Kod 6.4", "Tablo 6.9")),
        explanation=(
            "Her açıklayıcı değişken diğerlerine göre regresyona alınır; yardımcı regresyonun $R_j^2$ değeri "
            "$\\text{VIF}_j = 1/(1 - R_j^2)$ formülüne yerleştirilir (Kod 6.4). Notlarda Denklem 5.7'deki konut fiyatı "
            "modelinin üç değişkeni kullanılır. Değişkenleri değiştirin: vergi değeri ya da konut büyüklüğünün "
            "logaritması eklenince VIF'ler nasıl değişiyor?"
        ),
        controls=(MultiChoice("adim5_x", "VIF hesaplanan açıklayıcı değişkenler (en az iki)", W.options(HPRICE, VIF_OPTIONS),
                              VIF_DEFAULT, help="Notlardaki model: konut büyüklüğü, yatak odası sayısı, arsa büyüklüğü.",
                              minimum=2),),
        build=_vif,
        checks=(
            *(_scalar(f"r2_{name}", r2, f"Tablo 6.9: {label}, R²ⱼ", 4) for label, name, r2, _ in _TABLE69),
            *(Check(f"Tablo 6.9: {label}, VIF", TableTarget("vif", label, "deger"), vif, 3)
              for label, _, _, vif in _TABLE69),
            _scalar("r_buyukluk_oda", 0.53, "§6.12: konut büyüklüğü–yatak odası korelasyonu", 2),
        ),
        note_for=lambda state, choices: _vif_note(state, choices),
    ),
    LabStep(
        number=6,
        title="Bütünleşik çıktı okuma: yanlılık ile belirsizliği ayırmak",
        note=NoteRef("6.13", 0, ("Tablo 6.10",)),
        explanation=(
            "Eksik değişken yanlılığı katsayının yanlış merkez çevresinde toplanmasıdır; yüksek çoklu doğrusal bağlantı "
            "sıfır koşullu ortalama sağlansa bile ayrı katsayıların hassas ayrıştırılmasını zorlaştırır (Tablo 6.10). "
            "Bir çıktıyı okurken sırasıyla sorun (§6.13):\n\n1. Araştırma sorusu ve hedef katsayı nedir?\n2. Modelde "
            "hangi temel kontroller var, hangileri dışarıda kalmış?\n3. Dışlanan bir faktörün hem sonuçla hem temel "
            "değişkenle ilişkili olması makul mü?\n4. Açıklayıcı değişkenler aynı bilgiyi tekrar ediyor olabilir mi?\n5. "
            "Katsayılar alternatif fakat ekonomik olarak savunulabilir modellerde nasıl değişiyor?\n6. Uyum yüksek olsa "
            "bile ayrı katsayıların yorumu destekleniyor mu?\n7. Sonuç ilişki diliyle mi, nedensel dille mi "
            "raporlanmalı?\n\nStandart hata, t ve p-değeri Konu 7'de sistematik olarak okunur."
        ),
    ),
)


KONU06_LAB = LabSpec(
    topic_key="konu06",
    title="Uygulama: EKK Varsayımları, Yansızlık ve Model Sorunları",
    note_section="6",
    steps=STEPS,
    labels=(
        *W.labels(DATA, HPRICE),
        (INTERCEPT, "Sabit terim"),
        ("beta1", "β₁ (aylık gelir)"),
        ("beta2", "β₂ (yıllık gelir)"),
        ("birlesim", "β₁ + 12β₂"),
        ("egitim_ay", "Eğitim (ay)"),
    ),
    consistency_notes=(
        "Veri notlardaki gibi WAGE1 ve HPRICE1'dir; notlardaki kod, betikler, uygulama ve üretilen kod onları wooldridge "
        "paketinden okur.",
        "Bölümün benzetimleri (Tablo 6.3, 6.5, 6.7, 6.8) tohum 305 ile yeniden üretildi ve Sezgi sekmesindeki Konu 6 "
        "deneylerinin varsayılan ayarlarıdır; çoklu bağlantı düzeyleri aynı tohumla başlar (ortak rastgele sayılar).",
        "§6.12'ye HPRICE1 VIF örneği (Kod 6.4, Tablo 6.9) eklendi; sunumdaki kukla tuzağı örneği notlardaki aylık/yıllık "
        "gelir örneğiyle değiştirildi.",
    ),
)
