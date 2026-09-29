"""Konu 10 uygulaması: kukla değişkenler ve kategorik açıklayıcı değişkenler (gerçek veri).

Bölüm 10'un çözümlü örnekleri bölüm sırasıyla: kadın ve erkek çalışanların ham ortalamaları ve basit kukla modeli
(§10.2, Tablo 10.1, Kod 10.1–10.2, Şekil 10.1), kontrollü kukla modeli (§10.3, Tablo 10.2), additif modelin paralel
doğruları (§10.3, Şekil 10.2), kukla katsayısı için çıkarım (§10.4), bölge kuklaları ve referans kategori (§10.5,
Tablo 10.3, Şekil 10.3), referans kategorinin değişmesi (§10.5), kukla değişken tuzağı (§10.6), log modelinde tam
yüzde fark (§10.7, Kod 10.3–10.4), bölge ve endüstri kuklalarının ortak testleri (§10.8, Tablo 10.4) ve makale
tablosu (§10.9, Tablo 10.5). Her ``Check`` notlarda basılı bir sayıdır; değer notlardan kopyalanmıştır.

Bölge kuklaları açıkça kurulur: WAGE1'deki üç bölge kuklasına (northcen, south, west) Kuzeydoğu göstergesi
northeast = 1 − northcen − south − west eklenir; referans kategori modele girmeyen kukladır. Notlardaki kod aynı modeli
``C(region, Treatment(reference="Kuzeydoğu"))`` yazımıyla kurar; katsayılar aynıdır.

Etkileşim: bağımlı değişken ve kukla kodlaması (Adım 1), kontrol değişkenleri (Adım 2 ve 11), paralel doğruların
yatay ekseni (Adım 3), incelenen kukla (Adım 4 ve 8), referans kategori (Adım 5, 6 ve 9). Standart hatalar klasik
EKK standart hatalarıdır.
"""

from __future__ import annotations

import math

from core import wooldridge_data as W
from core.labs import expr as E
from core.labs.sezgi import plain
from core.labs.spec import (
    INTERCEPT,
    OLS,
    BarChart,
    Check,
    Choice,
    CoefficientPlot,
    CoefficientTable,
    CoefTarget,
    CopyFrame,
    Derive,
    GroupSummary,
    HypothesisPlot,
    JoinColumns,
    JointTest,
    LabSpec,
    LabStep,
    LineChart,
    LoadWooldridge,
    ModelTarget,
    ModelValue,
    MultiChoice,
    NoteRef,
    RegressionTable,
    Residuals,
    Scalar,
    ScalarTable,
    ScalarTarget,
    ShowModel,
    Statistic,
    Support,
    TableTarget,
    interactive_step,
)

DATA = "wage1"
CONTROLS = ("educ", "exper", "tenure")
"""Bölüm 10'un kontrol değişkenleri: eğitim, deneyim ve kıdem."""
CONTROL_OPTIONS = ("educ", "exper", "tenure", "numdep")
"""Kontrol seçimleri: kukla olmayan sayısal değişkenler (Konu 5–8 ile aynı küme)."""
BASE = ("educ", "exper", "expersq", "tenure", "tenursq")
"""Bölge ve endüstri modellerinin ortak açıklayıcıları."""
REGIONS = ("northeast", "northcen", "south", "west")
REGION_NAMES = {"northeast": "Kuzeydoğu", "northcen": "Kuzey Merkez", "south": "Güney", "west": "Batı"}
INDUSTRY = ("construc", "ndurman", "trcommpu", "trade", "services", "profserv")
INDUSTRY_NAMES = {"construc": "İnşaat", "ndurman": "Dayanıksız imalat", "trcommpu": "Ulaştırma/iletişim/kamu hizmeti",
                  "trade": "Ticaret", "services": "Hizmetler", "profserv": "Profesyonel hizmetler"}
DUMMIES = ("female", "married", "nonwhite", "smsa")
"""Kukla seçimleri (Adım 4 ve 8): kadın, evli, beyaz olmayan, büyükşehirde yaşıyor."""
DUMMY_WORDS = {"female": ("kadın", "erkek"), "married": ("evli", "evli olmayan"),
               "nonwhite": ("beyaz olmayan", "beyaz"), "smsa": ("büyükşehirde yaşayan", "büyükşehir dışında yaşayan"),
               "male": ("erkek", "kadın")}
"""Kuklanın 1 ve 0 grupları."""
DUMMY_TITLES = {"female": "Kadın", "married": "Evli", "nonwhite": "Beyaz olmayan", "smsa": "Büyükşehir",
                "male": "Erkek"}
"""Başlıklarda kuklanın adı ("Büyükşehir kuklası")."""
REGION_ABBR = {"northeast": "KD", "northcen": "KM", "south": "G", "west": "B"}
"""Notlardaki kısaltmalar: δ_KM, δ_G, δ_B."""
WORDS = {"educ": "eğitim", "exper": "deneyim", "tenure": "kıdem", "numdep": "bakmakla yükümlü kişi sayısı"}
Y_WORDS = {"wage": ("saatlik ücret", "dolar"), "lwage": ("log saatlik ücret", "log birim"), "educ": ("eğitim", "yıl"),
           "exper": ("deneyim", "yıl"), "tenure": ("kıdem", "yıl")}


def _scalar(name: str, expected: float, label: str, decimals: int) -> Check:
    return Check(label, ScalarTarget(name), expected, decimals)


def _p(value: float, decimals: int = 3) -> str:
    if value < 0.5 * 10 ** -decimals:
        return "p < " + plain(10 ** -decimals, decimals)
    return f"p = {plain(value, decimals)}"


def _listing(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " ve " + items[-1]


def _load() -> LoadWooldridge:
    return LoadWooldridge(DATA, "WAGE1 veri seti (Wooldridge, 2020): 526 çalışan")


def _formula(outcome: str, terms: tuple[str, ...]) -> str:
    return f"{outcome} ~ " + " + ".join(terms)


# --- Adım 1: ham ortalamalar ve basit kukla modeli ------------------------------------------------------------------

Y_1 = Choice("adim1_y", "Bağımlı değişken", W.options(DATA, ("wage", "lwage", "educ", "exper", "tenure")), "wage",
             help="Notlarda saatlik ücret (Kod 10.1–10.2).")
CODE_1 = Choice("adim1_kod", "Kukla değişkenin kodlaması",
                (("female", "1 = kadın, 0 = erkek (notlar)"), ("male", "1 = erkek, 0 = kadın")), "female",
                help="Kodlama ters çevrilirse katsayının işareti değişir; iki grubun ortalamaları değişmez.")


def _simple(choices) -> tuple:
    outcome, coding = choices["adim1_y"], choices["adim1_kod"]
    frame, dummy = DATA, "female"
    recode: tuple = ()
    if coding == "male":
        frame, dummy = "kukla", "male"
        recode = (CopyFrame("kukla", DATA, "Ters kodlama için veri setinin kopyası (özgün veri değişmez)"),
                  Derive("kukla", "male", E.sub(1, E.var("female")), "Erkek göstergesi: male = 1 − female"))
    one, zero = DUMMY_WORDS[dummy]
    notes = outcome == "wage" and coding == "female"
    return (
        _load(),
        GroupSummary(DATA, "female", (("n", "wage", "count"), ("ucret", "wage", "mean"), ("egitim", "educ", "mean"),
                                      ("deneyim", "exper", "mean"), ("kidem", "tenure", "mean")), "tablo101", (0, 1),
                     decimals=2, labels=((0, "Erkek"), (1, "Kadın")), heading="Grup",
                     title="Tablo 10.1: WAGE1 veri setinde kadın ve erkek çalışanların ham ortalamaları"),
        BarChart("tablo101", "ucret", "Grup", "Ortalama saatlik ücret (dolar)",
                 "Şekil 10.1: WAGE1'de kadın ve erkek çalışanların ham ücret ortalamaları", decimals=2),
        *recode,
        OLS("m_basit", frame, outcome, (dummy,),
            "Kod 10.1: " + _formula(outcome, (dummy,)) if notes else "Basit kukla modeli: " + _formula(outcome, (dummy,))),
        ShowModel("m_basit", "Kod 10.2: basit kukla modelinin Python çıktısı" if notes
                  else "Basit kukla modelinin Python çıktısı", columns=("coef", "se", "t", "p"), stats=("nobs", "r2"),
                  decimals=(("se", 4), ("p", 4)), exact=True),
        *(() if notes else (  # notlardaki model seçilen modelle yan yana
            OLS("m_basit_n", DATA, "wage", ("female",), "Notlardaki model (Kod 10.1): wage ~ female"),
            RegressionTable((("Notlar: ücret ~ kadın", "m_basit_n"),
                             (f"Seçiminiz: {W.variable(DATA, outcome).text.split(' (')[0].lower()} ~ "
                              f"{DUMMY_WORDS[dummy][0]}", "m_basit")),
                            (INTERCEPT, *dict.fromkeys(("female", dummy))), "yan101",
                            "Notlardaki model ve seçtiğiniz model", decimals=4, exact=True),
        )),
        ModelValue("b0", "m_basit", "coef", f"Sabit terim: {zero} grubunun ortalaması", term=INTERCEPT, decimals=4),
        ModelValue("delta", "m_basit", "coef", f"Kukla katsayısı: {one} − {zero} farkı", term=dummy, decimals=4),
        Statistic(frame, outcome, "mean", "ort_0", f"{zero.capitalize()} grubunun örneklem ortalaması",
                  where=(dummy, 0), decimals=4),
        Statistic(frame, outcome, "mean", "ort_1", f"{one.capitalize()} grubunun örneklem ortalaması",
                  where=(dummy, 1), decimals=4),
        Scalar("grup1_tahmin", E.add(E.ref("b0"), E.ref("delta")), f"{one.capitalize()} grubu: β̂₀ + δ̂", decimals=4),
        Scalar("fark", E.sub(E.ref("ort_1"), E.ref("ort_0")), f"Ortalamalar farkı: {one} − {zero}", decimals=4),
    )


def _simple_note(state, choices) -> str:
    outcome, coding = choices["adim1_y"], choices["adim1_kod"]
    dummy = "female" if coding == "female" else "male"
    one, zero = DUMMY_WORDS[dummy]
    s = state.scalars
    word, unit = Y_WORDS[outcome]
    return (f"Sabit terim {plain(s['b0'], 4)}, {zero} çalışanların ortalama {word} değeridir ({unit}). Kukla katsayısı "
            f"{plain(s['delta'], 4)}, {one} ve {zero} ortalamaları arasındaki farktır: {plain(s['ort_1'], 4)} − "
            f"{plain(s['ort_0'], 4)} = {plain(s['fark'], 4)}. {one.capitalize()} grubunun tahmin edilen ortalaması "
            f"sabit ile katsayının toplamıdır: {plain(s['b0'], 4)} + "
            f"{plain(s['delta'], 4) if s['delta'] >= 0 else '(' + plain(s['delta'], 4) + ')'} = "
            f"{plain(s['grup1_tahmin'], 4)}. Kodlama ters çevrilirse sabit diğer grubun ortalaması olur ve katsayının "
            "işareti değişir; iki grup arasındaki fark aynıdır. Bu fark nedensel bir etki değildir (§10.2).")


# --- Adım 2: kontrollü kukla modeli --------------------------------------------------------------------------------

CONTROLS_2 = MultiChoice("adim2_x", "Kontrol değişkenleri", W.options(DATA, CONTROL_OPTIONS), CONTROLS,
                         help="Notlarda eğitim, deneyim ve kıdem (Tablo 10.2).")


def _controlled_models(terms: tuple[str, ...], suffix: str, prefix: str) -> tuple[list, list]:
    words = " + ".join(WORDS[term] for term in terms)
    rows = [(f"{prefix}Ücret ~ kadın + {words}", f"m_kontrol{suffix}", "wage"),
            (f"{prefix}Log ücret ~ kadın + {words}", f"m_logk{suffix}", "lwage")]
    operations = [OLS(name, DATA, outcome, ("female", *terms), f"{label}: {_formula(outcome, ('female', *terms))}")
                  for label, name, outcome in rows]
    return operations, rows


def _controlled(choices) -> tuple:
    chosen = tuple(choices["adim2_x"])
    notes = chosen == CONTROLS
    operations: list = [OLS("m_ham", DATA, "wage", ("female",), "Ham fark: wage ~ female")]
    rows = [("Ücret ~ kadın", "m_ham", "wage")]
    if notes:
        more, extra = _controlled_models(chosen, "", "")
        operations += more
        rows += extra
    else:  # notlardaki modeller seçilen modellerle yan yana
        noted, noted_rows = _controlled_models(CONTROLS, "_n", "Notlar: ")
        chose, chose_rows = _controlled_models(chosen, "", "Seçiminiz: ")
        operations += noted + chose
        rows += [noted_rows[0], chose_rows[0], noted_rows[1], chose_rows[1]]
    values: list = []
    for index, (label, model, _) in enumerate(rows, start=1):
        values += [
            ModelValue(f"k2_{index}", model, "coef", f"{label}: kadın katsayısı", term="female", decimals=3),
            ModelValue(f"s2_{index}", model, "se", f"{label}: standart hata", term="female", decimals=3),
            ModelValue(f"t2_{index}", model, "t", f"{label}: t", term="female", decimals=3),
            ModelValue(f"r2_{index}", model, "r2", f"{label}: R²", decimals=3),
        ]
    tables = [ScalarTable(tuple((label, E.ref(f"{prefix}_{i}")) for i, (label, _, _) in enumerate(rows, start=1)),
                          result, decimals=3, heading="Model")
              for prefix, result in (("k2", "k102"), ("s2", "s102"), ("t2", "t102"), ("r2", "r102"))]
    return (
        *operations,
        *values,
        *tables,
        JoinColumns("tablo102", (("Kadın katsayısı", "k102", "deger"), ("Standart hata", "s102", "deger"),
                                 ("t", "t102", "deger"), ("R²", "r102", "deger")), decimals=3, heading="Model",
                    title="Tablo 10.2: Kadın kuklasının farklı modellerdeki tahmini" if notes
                    else "Kadın kuklası: notlardaki ve seçtiğiniz modeller"),
    )


def _controlled_note(state, choices) -> str:
    chosen = tuple(choices["adim2_x"])
    s = state.scalars
    raw = s["k2_1"]
    level = s["k2_2"] if chosen == CONTROLS else s["k2_3"]
    log = s["k2_3"] if chosen == CONTROLS else s["k2_5"]
    if abs(level - raw) < 0.05:
        change = ("Katsayı neredeyse değişmez: seçilen değişkenler iki grupta benzer ya da ücretle zayıf ilişkilidir; "
                  "ham farkın bu değişkenlerle açıklanan bir bölümü yoktur. ")
    else:
        change = ("Katsayının değişmesi, ham farkın bir bölümünün bu değişkenlerdeki grup farklarıyla bağlantılı "
                  f"olduğunu gösterir; kontrollü fark yine de {'negatiftir' if level < 0 else 'pozitiftir'}. ")
    return (f"Ham fark {plain(raw, 3)} dolardır; {_listing([WORDS[t] for t in chosen])} sabit tutulunca kadın "
            f"katsayısı {plain(level, 3)} dolar olur. {change}Log ücret "
            f"modelinde katsayı {plain(log, 3)}: kontrol değişkenleri sabitken yaklaşık yüzde fark (tam dönüşüm "
            "Adım 8'de). Kontrol eklemek katsayıyı nedensel yapmaz; meslek, sektör, çalışma süresi gibi etmenler "
            "modelde yoktur (§10.3).")


# --- Adım 3: additif model ve paralel doğrular -------------------------------------------------------------------

AXIS_3 = Choice("adim3_x", "Yatay eksendeki değişken", W.options(DATA, CONTROLS), "educ",
                help="Notlarda eğitim; diğer iki değişken örneklem ortalamasında tutulur (Şekil 10.2).")
RANGES = {"educ": (0, 18), "exper": (1, 51), "tenure": (0, 44)}
"""Yatay eksen ızgarası: örneklemdeki en küçük ve en büyük değer."""


def _parallel(choices) -> tuple:
    axis = choices["adim3_x"]
    others = [term for term in CONTROLS if term != axis]
    line = E.add(E.ref("b0_log"), E.mul(E.ref(f"b_{axis}"), E.var(axis)))
    for term in others:
        line = E.add(line, E.mul(E.ref(f"b_{term}"), E.ref(f"ort_{term}")))
    lower, upper = RANGES[axis]
    notes = axis == "educ"
    return (
        OLS("m_log", DATA, "lwage", ("female", *CONTROLS), "Kontrollü log ücret modeli: lwage ~ female + educ + exper + "
            "tenure"),
        ModelValue("b0_log", "m_log", "coef", "Sabit terim", term=INTERCEPT, decimals=4),
        ModelValue("d_log", "m_log", "coef", "Kadın katsayısı", term="female", decimals=4),
        *(ModelValue(f"b_{term}", "m_log", "coef", f"{WORDS[term].capitalize()} katsayısı", term=term, decimals=4)
          for term in CONTROLS),
        *(Statistic(DATA, term, "mean", f"ort_{term}", f"{WORDS[term].capitalize()}: örneklem ortalaması", decimals=4)
          for term in others),
        Support("dogru", axis, lower, upper, f"{WORDS[axis].capitalize()} ızgarası: {lower}, …, {upper} (örneklemin "
                "aralığı)"),
        Derive("dogru", "erkek", line, f"Erkek (female = 0): sabit + {WORDS[axis]} eğimi; "
               f"{_listing([WORDS[t] for t in others])} ortalamada"),
        Derive("dogru", "kadin", E.add(E.var("erkek"), E.ref("d_log")), "Kadın (female = 1): erkek doğrusu + δ̂"),
        LineChart("dogru", axis, "erkek", W.variable(DATA, axis).text, "Tahmin edilen ln(saatlik ücret)",
                  "Şekil 10.2: additif kukla modelinde ortak eğim ve farklı düzey" if notes
                  else f"Additif kukla modeli: {WORDS[axis]} ekseninde paralel doğrular",
                  markers=False, series=(("kadin", "Kadın"),), legend="Erkek"),
    )


def _parallel_note(state, choices) -> str:
    axis = choices["adim3_x"]
    s = state.scalars
    return (f"İki doğrunun eğimi aynıdır ({WORDS[axis]} katsayısı {plain(s[f'b_{axis}'], 4)}); aralarındaki dikey "
            f"uzaklık her {WORDS[axis]} düzeyinde kadın katsayısıdır ({plain(s['d_log'], 4)}). Additif kukla modeli "
            "gruplara farklı sabit verir, farklı eğim vermez: doğrular hangi değişken yatay eksende olursa olsun "
            "paraleldir. Eğim farkı için etkileşim terimi gerekir (Konu 11). Log ölçekteki sabit fark, düzeyde sabit bir "
            "yüzde farka karşılık gelir (§10.3).")


# --- Adım 4: kukla katsayısı için çıkarım --------------------------------------------------------------------------

DUMMY_4 = Choice("adim4_kukla", "Sınanan kukla değişken", W.options(DATA, DUMMIES), "female",
                 help="Model: ln(ücret) ~ kukla + eğitim + deneyim + kıdem. Notlarda kadın kuklası (§10.4).")


def _inference(choices) -> tuple:
    dummy = choices["adim4_kukla"]
    title = DUMMY_TITLES[dummy]
    terms = (dummy, *CONTROLS)
    notes: tuple = () if dummy == "female" else (  # notlardaki kadın kuklası seçilen kuklayla yan yana
        OLS("m_cik_n", DATA, "lwage", ("female", *CONTROLS), "Notlardaki model: lwage ~ female + educ + exper + tenure"),
        CoefficientTable("m_cik_n", ("female",), "cikarim_n", "Notlar (§10.4): kadın kuklası", decimals=4,
                         t_decimals=3),
    )
    return (
        *notes,
        OLS("m_cik", DATA, "lwage", terms, f"Kontrollü log ücret modeli: {_formula('lwage', terms)}"),
        CoefficientTable("m_cik", (dummy,), "cikarim", f"{title} kuklası: katsayı, standart hata, t, p ve yüzde 95 güven "
                         "aralığı" + ("" if dummy == "female" else " (seçiminiz)"), decimals=4, t_decimals=3),
        ModelValue("t_cik", "m_cik", "t", f"{title} kuklasının t istatistiği", term=dummy, decimals=3),
        ModelValue("p_cik", "m_cik", "p", "İki taraflı p-değeri", term=dummy, decimals=3),
        ModelValue("sd_cik", "m_cik", "df_resid", "Serbestlik derecesi n − k − 1", decimals=0),
        HypothesisPlot("t", "t_cik", "sd_cik", f"{title} kuklası: H₀: δ = 0 için t testi", "t değeri"),
    )


def _inference_note(state, choices) -> str:
    dummy = choices["adim4_kukla"]
    one, zero = DUMMY_WORDS[dummy]
    s = state.scalars
    table = state.tables["cikarim"]
    delta = float(table.loc[dummy, "katsayi"])
    rejected = s["p_cik"] < 0.05
    text = (f"δ̂ = {plain(delta, 4)}, t = {plain(s['t_cik'], 3)}, {_p(s['p_cik'])}: yüzde 5 düzeyinde H₀: δ = 0 "
            f"{'reddedilir' if rejected else 'reddedilemez'}. Eğitim, deneyim ve kıdem sabitken {one} ve {zero} "
            "çalışanlar arasında anakütlede log ücret farkı olmadığı hipotezi ")
    if rejected:
        return text + ("veriyle uyumsuzdur. Küçük bir p-değeri farkın nedenini açıklamaz ve eksik değişken olmadığını "
                       "göstermez; farkın büyüklüğü ve iktisadi önemi ayrıca değerlendirilir (§10.4).")
    return text + ("reddedilemez. Reddedememek farkın sıfır olduğunu kanıtlamaz: veri bu büyüklükteki bir farkı sıfırdan "
                   "ayırt edecek kadar kesin değildir; güven aralığı hem sıfırı hem de iktisadi olarak önemli olabilecek "
                   "değerleri içerebilir (§10.4).")


# --- Adım 5: bölge kuklaları ve referans kategori ------------------------------------------------------------------

REF_5 = Choice("adim5_ref", "Referans kategori", tuple((region, REGION_NAMES[region]) for region in REGIONS),
               "northeast", help="Notlarda Kuzeydoğu (Tablo 10.3). Referans modele girmeyen kukladır.")


def _region_frame() -> tuple:
    return (
        CopyFrame("bolge", DATA, "Bölge modelleri için veri setinin kopyası (özgün veri değişmez)"),
        Derive("bolge", "northeast", E.sub(E.sub(E.sub(1, E.var("northcen")), E.var("south")), E.var("west")),
               "Kuzeydoğu göstergesi: 1 − northcen − south − west"),
    )


def _region_table(model: str, reference: str, prefix: str, title: str) -> tuple:
    """Bir referansla bölge modelinin log katsayıları, tam yüzde farkları, p-değerleri ve tam yüzde güven sınırları;
    ``prefix`` skaler ve tablo adlarının önekidir (seçilen model "", notlardaki model "n")."""

    others = tuple(region for region in REGIONS if region != reference)
    values: list = []
    for region in others:
        label = REGION_NAMES[region]
        values += [
            ModelValue(f"k{prefix}_{region}", model, "coef", f"{label}: log katsayı", term=region, decimals=3),
            ModelValue(f"p{prefix}_{region}", model, "p", f"{label}: p-değeri", term=region, decimals=3),
            ModelValue(f"alt{prefix}_{region}", model, "ci_low", f"{label}: yüzde 95 GA alt sınırı (log)", term=region,
                       decimals=4),
            ModelValue(f"ust{prefix}_{region}", model, "ci_high", f"{label}: yüzde 95 GA üst sınırı (log)",
                       term=region, decimals=4),
            Scalar(f"tam{prefix}_{region}", E.mul(100, E.sub(E.exp(E.ref(f"k{prefix}_{region}")), 1)),
                   f"{label}: tam yüzde fark 100·(exp(δ̂) − 1)", decimals=2, percent=True),
            Scalar(f"talt{prefix}_{region}", E.mul(100, E.sub(E.exp(E.ref(f"alt{prefix}_{region}")), 1)),
                   f"{label}: tam yüzde GA alt sınırı", decimals=2, percent=True),
            Scalar(f"tust{prefix}_{region}", E.mul(100, E.sub(E.exp(E.ref(f"ust{prefix}_{region}")), 1)),
                   f"{label}: tam yüzde GA üst sınırı", decimals=2, percent=True),
        ]
    tables = [ScalarTable(tuple((REGION_NAMES[region], E.ref(f"{name}{prefix}_{region}")) for region in others),
                          f"{result}{prefix}", decimals=3, heading="Bölge")
              for name, result in (("k", "k103"), ("tam", "tam103"), ("p", "p103"), ("talt", "talt103"),
                                   ("tust", "tust103"))]
    return (
        *values,
        *tables,
        JoinColumns(f"tablo103{prefix}", (("Log katsayı", f"k103{prefix}", "deger"),
                                          ("Tam yüzde fark", f"tam103{prefix}", "deger"),
                                          ("p", f"p103{prefix}", "deger"), ("GA alt (%)", f"talt103{prefix}", "deger"),
                                          ("GA üst (%)", f"tust103{prefix}", "deger")),
                    decimals=2, heading="Bölge", column_decimals=(("Log katsayı", 3),), p_columns=("p",), title=title),
    )


def _regions(choices) -> tuple:
    reference = choices["adim5_ref"]
    others = tuple(region for region in REGIONS if region != reference)
    terms = (*BASE, *others)
    notes = reference == "northeast"
    name = REGION_NAMES[reference]
    noted: tuple = () if notes else (  # notlardaki (Kuzeydoğu referanslı) tablo seçilen referansla yan yana
        OLS("m_bolge_n", "bolge", "lwage", (*BASE, "northcen", "south", "west"),
            "Notlardaki model, referans Kuzeydoğu: lwage ~ educ + exper + expersq + tenure + tenursq + northcen + south "
            "+ west"),
        *_region_table("m_bolge_n", "northeast", "n", "Notlar — Tablo 10.3: Kuzeydoğu referans olduğunda bölge "
                                                      "kuklaları"),
    )
    return (
        *_region_frame(),
        *noted,
        OLS("m_bolge", "bolge", "lwage", terms, f"Bölge modeli, referans {name}: {_formula('lwage', terms)}"),
        *_region_table("m_bolge", reference, "", "Tablo 10.3: Kuzeydoğu referans olduğunda bölge kuklaları" if notes
                       else f"Seçiminiz: {name} referans olduğunda bölge kuklaları"),
        CoefficientPlot("m_bolge", others, "Şekil 10.3: Kuzeydoğu'ya göre bölge farkları ve yüzde 95 güven aralıkları"
                        if notes else f"Referans {name}: bölge farkları ve yüzde 95 güven aralıkları",
                        f"Referansa ({name}) göre tam yüzde fark", y_label="Bölge", percent=True,
                        labels=tuple(REGION_NAMES.items())),
    )


def _regions_note(state, choices) -> str:
    reference = choices["adim5_ref"]
    others = [region for region in REGIONS if region != reference]
    s = state.scalars
    name = REGION_NAMES[reference]
    significant = [REGION_NAMES[region] for region in others if s[f"p_{region}"] < 0.05]
    text = (f"Sabit terimli modelde dört kategori için üç kukla kullanılır; referans kategori {name} modelde yer almaz "
            "ve her satır bu referansa göre koşullu farktır. ")
    largest = max(others, key=lambda region: abs(s[f"tam_{region}"]))
    percent = s[f"tam_{largest}"]
    text += (f"Örneğin diğer değişkenler sabitken {REGION_NAMES[largest]} bölgesindeki çalışanların tahmin edilen "
             f"ücreti {name} bölgesindekilere göre tam hesapla %{plain(abs(percent), 2)} "
             f"{'daha yüksektir' if percent > 0 else 'daha düşüktür'} ({_p(s[f'p_{largest}'])}). ")
    if significant:
        text += f"Yüzde 5 düzeyinde referanstan ayrışan bölge: {_listing(significant)}. "
    else:
        text += "Güven aralıklarının hepsi sıfırı içerir: hiçbir bölge yüzde 5 düzeyinde referanstan ayrışmaz. "
    return text + "Referans kategori ‘normal’ ya da ‘üstün’ kategori değildir; yalnız karşılaştırma tabanıdır (§10.5)."


# --- Adım 6: referans kategori değişince -----------------------------------------------------------------------------

REF_6 = Choice("adim6_ref", "İkinci modelin referans kategorisi",
               tuple((region, REGION_NAMES[region]) for region in REGIONS if region != "northeast"), "south",
               help="İlk model notlardaki gibi Kuzeydoğu referanslıdır; notlarda ikinci referans Güney (§10.5).")


def _reference(choices) -> tuple:
    reference = choices["adim6_ref"]
    others = tuple(region for region in REGIONS if region != reference)
    name = REGION_NAMES[reference]
    comparisons = tuple(ModelValue(f"k2_{region}", "m_ref", "coef", f"{REGION_NAMES[region]}: {name} referansına göre",
                                   term=region, decimals=4) for region in others)
    return (
        CopyFrame("bolge6", "bolge", "Referans karşılaştırması için bölge verisinin kopyası"),
        OLS("m_kd", "bolge6", "lwage", (*BASE, "northcen", "south", "west"),
            "Kuzeydoğu referanslı model: lwage ~ educ + exper + expersq + tenure + tenursq + northcen + south + west"),
        OLS("m_ref", "bolge6", "lwage", (*BASE, *others), f"{name} referanslı model: {_formula('lwage', (*BASE, *others))}"),
        RegressionTable((("Referans: Kuzeydoğu", "m_kd"), (f"Referans: {name}", "m_ref")),
                        (INTERCEPT, *REGIONS), "tablo_ref", "İki referansla bölge katsayıları (boş hücre: o modelin "
                        "referans kategorisi; bölge dışı katsayılar aynı)",
                        stars=False, standard_errors=False, decimals=4, r2_decimals=6),
        ModelValue("k1_south", "m_kd", "coef", "Kuzeydoğu referanslı modelde Güney katsayısı", term="south", decimals=4),
        ModelValue("k1_west", "m_kd", "coef", "Kuzeydoğu referanslı modelde Batı katsayısı", term="west", decimals=4),
        *comparisons,
        Scalar("bati_guney", E.sub(E.ref("k1_west"), E.ref("k1_south")),
               "Kuzeydoğu referanslı katsayılardan Batı − Güney farkı", decimals=4),
        ModelValue("r2_kd", "m_kd", "r2", "Kuzeydoğu referanslı model: R²", decimals=6),
        ModelValue("r2_ref", "m_ref", "r2", f"{name} referanslı model: R²", decimals=6),
        Residuals("bolge6", "u_kd", "m_kd", "Kuzeydoğu referanslı modelin artıkları"),
        Residuals("bolge6", "u_ref", "m_ref", f"{name} referanslı modelin artıkları"),
        Derive("bolge6", "u_fark", E.absolute(E.sub(E.var("u_kd"), E.var("u_ref"))), "İki modelin artık farkı"),
        Statistic("bolge6", "u_fark", "max", "azami_fark", "Tahmin edilen değerlerin (ve artıkların) azami farkı",
                  decimals=15),
    )


def _reference_note(state, choices) -> str:
    reference = choices["adim6_ref"]
    s = state.scalars
    name = REGION_NAMES[reference]
    model = state.models["m_kd"]
    other = next(region for region in ("west", "northcen", "south") if region != reference)
    first, base = float(model.params[other]), float(model.params[reference])
    return (f"{name} referans olunca Kuzeydoğu katsayısı {plain(s['k2_northeast'], 4)} olur: önceki modeldeki {name} "
            "katsayısının işaret değiştirmiş hâli. Diğer katsayılar da yeni referansa göre yazılır: Kuzeydoğu "
            f"referanslı modelde {REGION_NAMES[other]} − {name} farkı {plain(first, 4)} − "
            f"{plain(base, 4) if base >= 0 else '(' + plain(base, 4) + ')'} = {plain(first - base, 4)}; bu, {name} "
            f"referanslı modeldeki {REGION_NAMES[other]} katsayısıdır ({plain(s[f'k2_{other}'], 4)}). İki modelin R²'si "
            f"aynıdır ({plain(s['r2_kd'], 6)}) ve tahmin edilen "
            "değerleri arasındaki en büyük fark yalnız bilgisayar yuvarlaması düzeyindedir (10⁻¹²'den küçük). Referans "
            "değişikliği modeli değil, katsayıların hangi gruba göre yazıldığını değiştirir (§10.5).")


# --- Adım 7: kukla değişken tuzağı -----------------------------------------------------------------------------------

def _trap() -> tuple:
    return (
        Derive("bolge", "toplam", E.add(E.add(E.add(E.var("northeast"), E.var("northcen")), E.var("south")),
                                         E.var("west")), "Dört bölge kuklasının toplamı"),
        Statistic("bolge", "toplam", "min", "toplam_min", "Dört kuklanın toplamı: en küçük değer", decimals=0),
        Statistic("bolge", "toplam", "max", "toplam_max", "Dört kuklanın toplamı: en büyük değer", decimals=0),
        *(Statistic("bolge", region, "sum", f"sayi_{region}", f"{REGION_NAMES[region]}: çalışan sayısı", decimals=0)
          for region in REGIONS),
        ScalarTable(tuple((REGION_NAMES[region], E.ref(f"sayi_{region}")) for region in REGIONS), "bolge_sayilari",
                    decimals=0, heading="Bölge", value="Çalışan sayısı"),
    )


# --- Adım 8: log modelinde tam yüzde fark ----------------------------------------------------------------------------

DUMMY_8 = Choice("adim8_kukla", "İncelenen kukla değişken", W.options(DATA, DUMMIES), "female",
                 help="Model: ln(ücret) ~ kukla + eğitim + deneyim + kıdem. Notlarda kadın kuklası (Kod 10.4).")


def _percent_rows(model: str, dummy: str, prefix: str) -> tuple:
    """Kukla katsayısı, yaklaşık ve tam yüzde fark ve tam yüzde güven sınırları; ``prefix`` notlardaki model için "n"."""

    title = DUMMY_TITLES[dummy]
    return (
        ModelValue(f"d{prefix}_yuzde", model, "coef", f"{title} kuklasının katsayısı δ̂", term=dummy, decimals=4),
        ModelValue(f"d{prefix}_alt", model, "ci_low", "δ̂'nın yüzde 95 GA alt sınırı", term=dummy, decimals=4),
        ModelValue(f"d{prefix}_ust", model, "ci_high", "δ̂'nın yüzde 95 GA üst sınırı", term=dummy, decimals=4),
        Scalar(f"yaklasik{prefix}_d", E.mul(100, E.ref(f"d{prefix}_yuzde")), "Yaklaşık yüzde fark: 100·δ̂", decimals=2,
               percent=True),
        Scalar(f"tam{prefix}_d", E.mul(100, E.sub(E.exp(E.ref(f"d{prefix}_yuzde")), 1)),
               "Tam yüzde fark: 100·(exp(δ̂) − 1)", decimals=2, percent=True),
        Scalar(f"tam{prefix}_alt", E.mul(100, E.sub(E.exp(E.ref(f"d{prefix}_alt")), 1)), "Tam yüzde farkın GA alt sınırı",
               decimals=2, percent=True),
        Scalar(f"tam{prefix}_ust", E.mul(100, E.sub(E.exp(E.ref(f"d{prefix}_ust")), 1)), "Tam yüzde farkın GA üst sınırı",
               decimals=2, percent=True),
    )


def _percent(choices) -> tuple:
    dummy = choices["adim8_kukla"]
    terms = (dummy, *CONTROLS)
    notes = dummy == "female"
    side: tuple = () if notes else (  # notlardaki kadın kuklası seçilen kuklayla yan yana
        OLS("m_yuzde_n", DATA, "lwage", ("female", *CONTROLS), "Notlardaki model (Kod 10.4): lwage ~ female + educ + "
            "exper + tenure"),
        *_percent_rows("m_yuzde_n", "female", "n"),
        *(ScalarTable((("Notlar: kadın", E.ref(f"{name}n_{part}")), (f"Seçiminiz: {DUMMY_WORDS[dummy][0]}",
                                                                     E.ref(f"{name}_{part}"))),
                      f"yuzde_{column}", decimals=2, heading="Kukla")
          for column, name, part in (("d", "d", "yuzde"), ("yak", "yaklasik", "d"), ("tam", "tam", "d"),
                                     ("alt", "tam", "alt"), ("ust", "tam", "ust"))),
    )
    comparison: tuple = () if notes else (
        JoinColumns("yuzde_karsilastirma", (("δ̂", "yuzde_d", "deger"), ("Yaklaşık (%)", "yuzde_yak", "deger"),
                                            ("Tam (%)", "yuzde_tam", "deger"), ("GA alt (%)", "yuzde_alt", "deger"),
                                            ("GA üst (%)", "yuzde_ust", "deger")),
                    decimals=2, heading="Kukla", column_decimals=(("δ̂", 4),),
                    title="Notlardaki kadın kuklası ve seçtiğiniz kukla: yaklaşık ve tam yüzde fark"),
    )
    return (
        OLS("m_yuzde", DATA, "lwage", terms, f"Kontrollü log ücret modeli: {_formula('lwage', terms)}"),
        ShowModel("m_yuzde", "Kod 10.4: kontrollü log ücret modelinin Python çıktısı" if notes
                  else "Kontrollü log ücret modelinin Python çıktısı (seçiminiz)", columns=("coef", "se", "t", "p"),
                  stats=("nobs", "r2"), decimals=(("se", 4), ("p", 4)), exact=True),
        *_percent_rows("m_yuzde", dummy, ""),
        *side,
        *comparison,
        Scalar("ornek_30", E.mul(100, E.sub(E.exp(-0.30), 1)), "δ = −0,30 için tam yüzde fark", decimals=1,
               percent=True),
    )


def _percent_note(state, choices) -> str:
    dummy = choices["adim8_kukla"]
    one, zero = DUMMY_WORDS[dummy]
    s = state.scalars
    return (f"δ̂ = {plain(s['d_yuzde'], 4)}: yaklaşık yorum %{plain(s['yaklasik_d'], 2)}, tam yorum "
            f"%{plain(s['tam_d'], 2)}. Eğitim, deneyim ve kıdem sabitken {one} çalışanların tahmin edilen saatlik "
            f"ücreti {zero} çalışanlara göre tam hesapla %{plain(abs(s['tam_d']), 2)} "
            f"{'daha yüksektir' if s['tam_d'] > 0 else 'daha düşüktür'}. Yüzde 95 güven aralığının tam yüzde biçimi "
            f"[%{plain(s['tam_alt'], 2)}; %{plain(s['tam_ust'], 2)}]"
            + ("; aralık sıfırı içerir, bu fark yüzde 5 düzeyinde istatistiksel olarak anlamlı değildir. "
               if s["tam_alt"] < 0 < s["tam_ust"] else ". ")
            + "Kukla 0'dan 1'e tam bir birim değiştiği için katsayı büyükse 100·δ̂ yaklaşımı belirgin hata verir. "
            "‘Yüzde’ fark ‘yüzde puan’ farkı değildir (§10.7).")


# --- Adım 9: bölge kuklalarının ortak testi -------------------------------------------------------------------------

REF_9 = Choice("adim9_ref", "Referans kategori", tuple((region, REGION_NAMES[region]) for region in REGIONS),
               "northeast", help="Notlarda Kuzeydoğu. Ortak test hangi kategori referans olursa olsun aynıdır.")


def _joint_region(choices) -> tuple:
    reference = choices["adim9_ref"]
    others = tuple(region for region in REGIONS if region != reference)
    name = REGION_NAMES[reference]
    hypothesis = ", ".join(f"δ_{REGION_ABBR[region]} = 0" for region in others)
    return (
        OLS("m_ortak", "bolge", "lwage", (*BASE, *others), f"Bölge modeli, referans {name}"),
        JointTest("F_bolge", "p_bolge", "m_ortak", others, f"Üç bölge kuklasının ortak testi, H₀: {hypothesis}",
                  decimals=3),
        ModelValue("sd_bolge", "m_ortak", "df_resid", "Payda serbestlik derecesi n − k − 1", decimals=0),
        Scalar("q_bolge", E.const(len(others)), "Kısıt sayısı q", decimals=0),
        HypothesisPlot("f", "F_bolge", "q_bolge", "Bölge kuklalarının ortak testi: F(3, 517)", "F değeri",
                       alternative="sag", df2="sd_bolge"),
    )


def _joint_region_note(state, choices) -> str:
    reference = choices["adim9_ref"]
    s = state.scalars
    rejected = s["p_bolge"] < 0.05
    return (f"Referans {REGION_NAMES[reference]} iken F(3, {plain(s['sd_bolge'], 0)}) = {plain(s['F_bolge'], 3)}, "
            f"{_p(s['p_bolge'])}: yüzde 5 düzeyinde bölge kuklalarının birlikte sıfır olduğu hipotezi "
            f"{'reddedilir' if rejected else 'reddedilemez'}. Referansı değiştirin: F değişmez, çünkü ortak test "
            "kategorik değişkenin bütününü sınar ve referans değişikliği modelin uyumunu değiştirmez. Tek tek t testleri "
            "ise referansa bağlıdır (§10.8).")


# --- Adım 10: endüstri kuklaları -------------------------------------------------------------------------------------

def _industry() -> tuple:
    values: list = []
    for term in INDUSTRY:
        label = INDUSTRY_NAMES[term]
        values += [
            ModelValue(f"k_{term}", "m_end", "coef", f"{label}: log katsayı", term=term, decimals=3),
            ModelValue(f"p_{term}", "m_end", "p", f"{label}: p-değeri", term=term, decimals=3),
            Scalar(f"tam_{term}", E.mul(100, E.sub(E.exp(E.ref(f"k_{term}")), 1)),
                   f"{label}: tam yüzde fark 100·(exp(δ̂) − 1)", decimals=2, percent=True),
        ]
    tables = [ScalarTable(tuple((INDUSTRY_NAMES[term], E.ref(f"{prefix}_{term}")) for term in INDUSTRY), result,
                          decimals=3, heading="Endüstri")
              for prefix, result in (("k", "k104"), ("tam", "tam104"), ("p", "p104"))]
    return (
        OLS("m_end", DATA, "lwage", (*BASE, *INDUSTRY), f"Endüstri modeli: {_formula('lwage', (*BASE, *INDUSTRY))}"),
        JointTest("F_end", "p_end", "m_end", INDUSTRY, "Altı endüstri kuklasının ortak testi", decimals=3),
        ModelValue("sd_end", "m_end", "df_resid", "Payda serbestlik derecesi n − k − 1", decimals=0),
        *values,
        *tables,
        JoinColumns("tablo104", (("Log katsayı", "k104", "deger"), ("Tam yüzde fark", "tam104", "deger"),
                                 ("p", "p104", "deger")), decimals=2, heading="Endüstri",
                    column_decimals=(("Log katsayı", 3),), p_columns=("p",),
                    title="Tablo 10.4: WAGE1 modelinde seçilmiş endüstri kuklaları"),
        Scalar("q_end", E.const(len(INDUSTRY)), "Kısıt sayısı q (pay serbestlik derecesi)", decimals=0),
    )


# --- Adım 11: makale tablosu --------------------------------------------------------------------------------------

CONTROLS_11 = MultiChoice("adim11_x", "Sütun (2)'deki kontrol değişkenleri", W.options(DATA, CONTROL_OPTIONS), CONTROLS,
                          help="Notlarda eğitim, deneyim ve kıdem (Tablo 10.5).")


def _paper(choices) -> tuple:
    chosen = tuple(choices["adim11_x"])
    notes = chosen == CONTROLS
    if notes:
        models: tuple = (("(1)", "m_s1"), ("(2)", "m_s2"))
        extra: tuple = ()
        terms = ("female", *CONTROLS)
    else:  # notlardaki Sütun (2) seçilen modelle yan yana
        models = (("(1)", "m_s1"), ("(2) Notlar", "m_s2n"), ("(2) Seçiminiz", "m_s2"))
        extra = (OLS("m_s2n", DATA, "lwage", ("female", *CONTROLS), "Notlardaki Sütun (2): lwage ~ female + educ + "
                     "exper + tenure"),)
        terms = ("female", *(term for term in CONTROL_OPTIONS if term in chosen or term in CONTROLS))
    return (
        OLS("m_s1", DATA, "lwage", ("female",), "Sütun (1): lwage ~ female"),
        *extra,
        OLS("m_s2", DATA, "lwage", ("female", *chosen), f"Sütun (2): {_formula('lwage', ('female', *chosen))}"),
        RegressionTable(models, terms, "tablo105",
                        ("Tablo 10.5: kukla değişken içeren örnek makale tablosu" if notes
                         else "Kukla değişken içeren makale tablosu (notlardaki ve seçtiğiniz Sütun (2))")
                        + " · bağımlı değişken: ln(saatlik ücret)", decimals=3, exact=True),
    )


def _paper_note(state, choices) -> str:
    table = state.tables["tablo105"]
    heading = "(2)" if tuple(choices["adim11_x"]) == CONTROLS else "(2) Seçiminiz"
    first, second = table.loc["female", "(1)"], table.loc["female", heading]
    return (f"Sütun (1)'de kadın katsayısı {plain(first, 3)}: kontrol yokken ham log ücret farkı; tam yüzde "
            f"%{plain(100 * (math.exp(first) - 1), 1)}. Sütun (2)'de katsayı {plain(second, 3)}: "
            "kontrol değişkenleri sabitken koşullu fark. Katsayının sütunlar arasında değişmesi modelin cevapladığı "
            "sorunun değiştiğini gösterir. Referans grup erkek çalışanlardır (kadın kuklası 0); parantez içindeki "
            "sayılar standart hatalardır (§10.9).")


# --- Tanım ---------------------------------------------------------------------------------------------------------

_TABLE101 = (("Erkek", (274, 7.10, 12.79, 17.56, 6.47)), ("Kadın", (252, 4.59, 12.32, 16.43, 3.62)))
_COLUMNS101 = (("n", 0), ("ucret", 2), ("egitim", 2), ("deneyim", 2), ("kidem", 2))
_KOD102 = ((INTERCEPT, "Intercept", (7.0995, 0.2100, 33.806, 0.0)), ("female", "female", (-2.5118, 0.3034, -8.279, 0.0)))
_KOD104 = (
    (INTERCEPT, "Intercept", (0.5013, 0.1019, 4.920, 0.0)),
    ("female", "female", (-0.3011, 0.0372, -8.085, 0.0)),
    ("educ", "educ", (0.0875, 0.0069, 12.605, 0.0)),
    ("exper", "exper", (0.0046, 0.0016, 2.845, 0.0046)),
    ("tenure", "tenure", (0.0174, 0.0030, 5.835, 0.0)),
)
_QUANTITIES = (("coef", "coef", 4), ("se", "std err", 4), ("t", "t", 3), ("p", "P>|t|", 4))
_TABLE102 = (
    ("Ücret ~ kadın", (-2.512, 0.303, -8.279, 0.116)),
    ("Ücret ~ kadın + eğitim + deneyim + kıdem", (-1.811, 0.265, -6.838, 0.364)),
    ("Log ücret ~ kadın + eğitim + deneyim + kıdem", (-0.301, 0.037, -8.085, 0.392)),
)
_TABLE103 = (
    ("Kuzey Merkez", (-0.071, -6.89, 0.186, -16.24, 3.50)),
    ("Güney", (-0.083, -7.95, 0.101, -16.64, 1.64)),
    ("Batı", (0.033, 3.36, 0.580, -8.09, 16.24)),
)
_COLUMNS103 = (("Log katsayı", 3), ("Tam yüzde fark", 2), ("p", 3), ("GA alt (%)", 2), ("GA üst (%)", 2))
_TABLE104 = (
    ("İnşaat", (-0.035, -3.46, 0.714)),
    ("Dayanıksız imalat", (-0.163, -15.05, 0.021)),
    ("Ulaştırma/iletişim/kamu hizmeti", (-0.166, -15.26, 0.093)),
    ("Ticaret", (-0.320, -27.38, 0.0)),
    ("Hizmetler", (-0.418, -34.15, 0.0)),
    ("Profesyonel hizmetler", (-0.190, -17.27, 0.002)),
)
_TABLE105 = (
    ("(1)", (("female", -0.397), ("female_sh", 0.043), ("n", 526), ("r2", 0.140))),
    ("(2)", (("female", -0.301), ("female_sh", 0.037), ("educ", 0.087), ("educ_sh", 0.007), ("exper", 0.005),
             ("exper_sh", 0.002), ("tenure", 0.017), ("tenure_sh", 0.003), ("n", 526), ("r2", 0.392))),
)

STEPS = (
    interactive_step(
        number=1,
        title="İki grup ortalaması ve basit kukla modeli",
        note=NoteRef("10.2", 0, ("Tablo 10.1", "Kod 10.1", "Kod 10.2", "Şekil 10.1")),
        explanation=(
            "Kukla değişken iki kategoriyi 0 ve 1 ile ayırır: $\\text{female} = 1$ kadın, 0 erkek çalışan. "
            "$Y = \\beta_0 + \\delta D + u$ modelinde $\\mathbb{E}(Y \\mid D = 0) = \\beta_0$ ve "
            "$\\mathbb{E}(Y \\mid D = 1) = \\beta_0 + \\delta$: sabit terim referans grubun ortalaması, kukla katsayısı "
            "iki grup ortalamasının farkıdır. Bağımlı değişkeni ve kodlama yönünü değiştirin."
        ),
        controls=(Y_1, CODE_1),
        build=_simple,
        checks=(
            *(Check(f"Tablo 10.1: {group}, {column}", TableTarget("tablo101", group, column), value, decimals)
              for group, values in _TABLE101 for (column, decimals), value in zip(_COLUMNS101, values)),
            *(Check(f"Kod 10.2: {name}, {label}", CoefTarget("m_basit", term, quantity), value,
                    3 if quantity == "t" else 4)
              for term, name, values in _KOD102 for (quantity, label, _), value in zip(_QUANTITIES, values)),
            Check("Kod 10.2: No. Observations", ModelTarget("m_basit", "nobs"), 526, 0),
            Check("Kod 10.2: R-squared", ModelTarget("m_basit", "r2"), 0.116, 3),
            _scalar("ort_0", 7.0995, "§10.2: erkek çalışanların ortalama ücreti", 4),
            _scalar("ort_1", 4.5877, "§10.2: kadın çalışanların ortalama ücreti", 4),
            _scalar("fark", -2.5118, "§10.2: 4,5877 − 7,0995", 4),
            _scalar("grup1_tahmin", 4.5877, "§10.2: β̂₀ + δ̂ = 7,0995 − 2,5118", 4),
        ),
        note_for=lambda state, choices: _simple_note(state, choices),
    ),
    interactive_step(
        number=2,
        title="Kontrol değişkenleri eklenince kukla katsayısı",
        note=NoteRef("10.3", 0, ("Tablo 10.2",)),
        explanation=(
            "Ham grup farkı, grupların başka özellikler bakımından farklı olabileceğini dikkate almaz. Kontrollü "
            "modelde $\\text{ücret} = \\beta_0 + \\delta\\,\\text{female} + \\beta_1\\text{eğitim} + "
            "\\beta_2\\text{deneyim} + \\beta_3\\text{kıdem} + u$; $\\delta$ eğitim, deneyim ve kıdem aynı "
            "tutulduğunda kadın ve erkek çalışanlar arasındaki tahmin edilen ortalama ücret farkıdır. Kontrol "
            "değişkenlerini değiştirin."
        ),
        controls=(CONTROLS_2,),
        build=_controlled,
        checks=tuple(
            Check(f"Tablo 10.2: {row}, {column}", TableTarget("tablo102", row, column), value, 3)
            for row, values in _TABLE102
            for column, value in zip(("Kadın katsayısı", "Standart hata", "t", "R²"), values)
        ),
        note_for=lambda state, choices: _controlled_note(state, choices),
    ),
    interactive_step(
        number=3,
        title="Additif kukla modeli: paralel doğrular",
        note=NoteRef("10.3", 0, ("Şekil 10.2",)),
        explanation=(
            "Additif modelde kukla yalnız düzeyi kaydırır: bütün açıklayıcıların eğimi iki grupta aynıdır. Log ücret "
            "modeli: $\\widehat{\\ln(\\text{ücret})} = 0{,}5013 - 0{,}3011\\,\\text{female} + "
            "0{,}0875\\,\\text{eğitim} + 0{,}0046\\,\\text{deneyim} + 0{,}0174\\,\\text{kıdem}$. Diğer iki değişken "
            "örneklem ortalamasında tutulur. Yatay eksendeki değişkeni değiştirin."
        ),
        controls=(AXIS_3,),
        build=_parallel,
        checks=(
            _scalar("b0_log", 0.5013, "§10.3: sabit terim", 4),
            _scalar("d_log", -0.3011, "§10.3: kadın katsayısı", 4),
            _scalar("b_educ", 0.0875, "§10.3: eğitim katsayısı", 4),
            _scalar("b_exper", 0.0046, "§10.3: deneyim katsayısı", 4),
            _scalar("b_tenure", 0.0174, "§10.3: kıdem katsayısı", 4),
        ),
        note_for=lambda state, choices: _parallel_note(state, choices),
    ),
    interactive_step(
        number=4,
        title="Kukla katsayısı için istatistiksel çıkarım",
        note=NoteRef("10.4", 0),
        explanation=(
            "Kukla katsayısı da standart hata, t istatistiği, p-değeri ve güven aralığıyla değerlendirilir: "
            "$H_0: \\delta = 0$, $H_1: \\delta \\neq 0$. $H_0$ kontrol değişkenleri sabitken iki kategori arasında "
            "anakütlede ortalama fark bulunmadığını söyler. Model: ln(ücret) ~ kukla + eğitim + deneyim + kıdem. "
            "Sınanan kuklayı değiştirin."
        ),
        controls=(DUMMY_4,),
        build=_inference,
        checks=(
            Check("§10.4: kadın katsayısı", TableTarget("cikarim", "female", "katsayi"), -0.3011, 4),
            Check("§10.4: standart hata", TableTarget("cikarim", "female", "sh"), 0.0372, 4),
            Check("§10.4: t istatistiği", TableTarget("cikarim", "female", "t"), -8.085, 3),
            Check("§10.4: p < 0,001", TableTarget("cikarim", "female", "p"), 0.0, 3),
        ),
        note_for=lambda state, choices: _inference_note(state, choices),
    ),
    interactive_step(
        number=5,
        title="Çok kategorili değişken: bölge kuklaları",
        note=NoteRef("10.5", 0, ("Tablo 10.3", "Şekil 10.3", "Kod 10.5")),
        explanation=(
            "Kategorik bir değişken m kategori içeriyorsa sabit terimli modelde m − 1 kukla kullanılır; modele girmeyen "
            "kategori referanstır. WAGE1'de dört bölge vardır: Kuzeydoğu, Kuzey Merkez, Güney, Batı. Kuzeydoğu "
            "göstergesi diğer üç kukladan kurulur: $D_{KD} = 1 - D_{KM} - D_G - D_B$. Log katsayılar tam yüzde farka "
            "çevrilir: $100(e^{\\hat\\delta} - 1)$. Notlardaki Kod 10.5 aynı modeli metin değişkeniyle kurar: "
            "`C(region, Treatment(reference=\"Kuzeydoğu\"))` Kuzeydoğu'yu referans yapar ve çıktıdaki `[T.Güney]` "
            "satırı buradaki `south` kuklasının katsayısıdır. Referans kategoriyi değiştirin."
        ),
        controls=(REF_5,),
        build=_regions,
        checks=tuple(
            Check(f"Tablo 10.3: {row}, {column}", TableTarget("tablo103", row, column), value, decimals)
            for row, values in _TABLE103 for (column, decimals), value in zip(_COLUMNS103, values)
        ),
        note_for=lambda state, choices: _regions_note(state, choices),
    ),
    interactive_step(
        number=6,
        title="Referans kategori değişince ne olur?",
        note=NoteRef("10.5", 0),
        explanation=(
            "Referans kategori değişince sabit terim ve kukla katsayılarının sayısal değerleri ile karşılaştırılan grup "
            "değişir; tahmin edilen değerler, artıklar, R² ve modelin ortak uyumu değişmez. İlk model Kuzeydoğu "
            "referanslıdır. İkinci modelin referansını değiştirin."
        ),
        controls=(REF_6,),
        build=_reference,
        checks=(
            _scalar("k2_northeast", 0.0829, "§10.5: Güney referansken Kuzeydoğu katsayısı", 4),
            _scalar("k1_south", -0.0829, "§10.5: Kuzeydoğu referansken Güney katsayısı", 4),
            _scalar("k1_west", 0.0331, "§10.5: Kuzeydoğu referansken Batı katsayısı", 4),
            _scalar("bati_guney", 0.1160, "§10.5: 0,0331 − (−0,0829)", 4),
            _scalar("k2_west", 0.1160, "§10.5: Güney referansken Batı katsayısı", 4),
            _scalar("azami_fark", 0.0, "§10.5: iki referansla tahmin edilen değerlerin farkı (10⁻¹²'den küçük)", 12),
        ),
        note_for=lambda state, choices: _reference_note(state, choices),
    ),
    LabStep(
        number=7,
        title="Kukla değişken tuzağı",
        note=NoteRef("10.6", 0),
        explanation=(
            "Her çalışan tam olarak bir bölgededir: $D_{KD} + D_{KM} + D_G + D_B = 1$ her gözlemde geçerlidir. Sabit "
            "terim de her gözlemde 1 olan bir sütundur; sabit terim ile dört kuklanın tamamı birlikte kullanılırsa tam "
            "çoklu doğrusal bağlantı ortaya çıkar. Çözüm: sabit terimi koruyup bir kategoriyi referans bırakmak ya da "
            "sabit terimi kaldırıp bütün kuklaları kullanmak."
        ),
        operations=_trap(),
        checks=(
            _scalar("toplam_min", 1, "§10.6: dört kuklanın toplamı, en küçük değer", 0),
            _scalar("toplam_max", 1, "§10.6: dört kuklanın toplamı, en büyük değer", 0),
        ),
        takeaway=(
            "Dört kuklanın toplamı en küçük ve en büyük değerde 1'dir: toplam her gözlemde sabit terimin sütununa eşittir. "
            "Bu yüzden sabit terimli modelde en fazla üç bölge kuklası kullanılabilir; dört kukla ve sabit birlikte "
            "tam çoklu doğrusal bağlantı yaratır ve katsayılar tek biçimde tahmin edilemez. Yazılım bir kuklayı "
            "kendiliğinden düşürürse hangi kategorinin referans kaldığı kontrol edilir (§10.6)."
        ),
    ),
    interactive_step(
        number=8,
        title="Log bağımlı değişkende kukla katsayısı: tam yüzde fark",
        note=NoteRef("10.7", 0, ("Kod 10.3", "Kod 10.4")),
        explanation=(
            "Kukla değişkende değişim 0'dan 1'e tam bir birimdir; katsayı küçük değilse $100\\hat\\delta$ yaklaşımı "
            "önemli hata yaratabilir. Tam yüzde fark $100(e^{\\hat\\delta} - 1)$ ile hesaplanır; güven aralığının "
            "sınırları da aynı dönüşümle yüzdeye çevrilir. İncelenen kuklayı değiştirin."
        ),
        controls=(DUMMY_8,),
        build=_percent,
        checks=(
            *(Check(f"Kod 10.4: {name}, {label}", CoefTarget("m_yuzde", term, quantity), value,
                    3 if quantity == "t" else 4)
              for term, name, values in _KOD104 for (quantity, label, _), value in zip(_QUANTITIES, values)),
            Check("Kod 10.4: No. Observations", ModelTarget("m_yuzde", "nobs"), 526, 0),
            Check("Kod 10.4: R-squared", ModelTarget("m_yuzde", "r2"), 0.392, 3),
            _scalar("yaklasik_d", -30.11, "§10.7: yaklaşık yüzde −30,11", 2),
            _scalar("tam_d", -26.00, "§10.7: tam yüzde 100(exp(−0,3011) − 1)", 2),
            _scalar("tam_alt", -31.22, "§10.7: tam yüzde GA alt sınırı", 2),
            _scalar("tam_ust", -20.39, "§10.7: tam yüzde GA üst sınırı", 2),
            _scalar("ornek_30", -25.9, "§10.7: δ = −0,30 için tam yüzde fark", 1),
        ),
        note_for=lambda state, choices: _percent_note(state, choices),
    ),
    interactive_step(
        number=9,
        title="Bölge kuklalarının ortak testi",
        note=NoteRef("10.8", 0),
        explanation=(
            "Kategorik değişkenin bütünüyle modele katkı sağlayıp sağlamadığı ortak F testiyle sınanır: "
            "$H_0: \\delta_{KM} = 0,\\ \\delta_G = 0,\\ \\delta_B = 0$ (q = 3 kısıt). Tek tek t testleri kategorilerin "
            "referanstan farkını sınar; ortak test kategorik değişkenin bütününü. Referans kategoriyi değiştirin: F "
            "değişir mi?"
        ),
        controls=(REF_9,),
        build=_joint_region,
        checks=(
            _scalar("F_bolge", 2.095, "§10.8: F(3, 517)", 3),
            _scalar("p_bolge", 0.100, "§10.8: p-değeri", 3),
            _scalar("sd_bolge", 517, "§10.8: payda serbestlik derecesi", 0),
            _scalar("q_bolge", 3, "§10.8: kısıt sayısı", 0),
        ),
        note_for=lambda state, choices: _joint_region_note(state, choices),
    ),
    LabStep(
        number=10,
        title="Endüstri kuklaları: ortak test ve tam yüzde farklar",
        note=NoteRef("10.8", 0, ("Tablo 10.4",)),
        explanation=(
            "Altı endüstri kuklası birlikte sınanır; referans listelenmeyen diğer endüstrilerdir. Tek tek katsayılar "
            "referans endüstriye göre log farklardır ve tam yüzde farka çevrilir: $100(e^{\\hat\\delta} - 1)$."
        ),
        operations=_industry(),
        checks=(
            _scalar("F_end", 8.242, "§10.8: F(6, 514)", 3),
            _scalar("sd_end", 514, "§10.8: payda serbestlik derecesi", 0),
            _scalar("q_end", 6, "§10.8: F(6, 514), kısıt sayısı", 0),
            _scalar("p_end", 0.0, "§10.8: p < 0,001", 3),
            *(Check(f"Tablo 10.4: {row}, {'p < 0,001' if column == 'p' and value == 0 else column}",
                    TableTarget("tablo104", row, column), value, decimals)
              for row, values in _TABLE104
              for (column, decimals), value in zip((("Log katsayı", 3), ("Tam yüzde fark", 2), ("p", 3)), values)),
        ),
        takeaway=(
            "Endüstri kuklaları birlikte güçlü biçimde anlamlıdır: F(6, 514) = 8,242, p < 0,001. Buna karşılık bölge "
            "kuklaları yüzde 5 düzeyinde birlikte anlamlı değildir (Adım 9). Diğer değişkenler sabitken ticaret "
            "sektörünün referans endüstri grubuna göre tam yüzde farkı yaklaşık −27,38, hizmetlerin yaklaşık −34,15'tir. "
            "Bazı tekil katsayılar (inşaat, ulaştırma) yüzde 5 düzeyinde anlamlı olmasa da kategorik değişken bütünüyle "
            "anlamlı olabilir; tek tek katsayılar ile ortak test farklı soruları yanıtlar (§10.8)."
        ),
    ),
    interactive_step(
        number=11,
        title="Makale tablosunda kukla değişken",
        note=NoteRef("10.9", 0, ("Tablo 10.5",)),
        explanation=(
            "Makale tablosunu okurken önce tablonun sözlüğü okunur: bağımlı değişken düzey mi log mu, hangi kategori "
            "referans, parantez içindeki sayı standart hata mı t istatistiği mi, yıldızların eşikleri ne, hangi "
            "kontroller var. Sütun (1) ham log ücret farkı, Sütun (2) kontrollü farktır. Sütun (2)'nin kontrollerini "
            "değiştirin."
        ),
        controls=(CONTROLS_11,),
        build=_paper,
        checks=tuple(
            Check(f"Tablo 10.5: {heading}, {row}", TableTarget("tablo105", row, heading), value, 0 if row == "n" else 3)
            for heading, values in _TABLE105 for row, value in values
        ),
        note_for=lambda state, choices: _paper_note(state, choices),
    ),
)


KONU10_LAB = LabSpec(
    topic_key="konu10",
    title="Uygulama: Kukla Değişkenler ve Kategorik Açıklayıcı Değişkenler",
    note_section="10",
    steps=STEPS,
    labels=(
        *W.labels(DATA),
        (INTERCEPT, "Sabit terim"),
        ("male", "Erkek (0/1 gösterge)"),
        ("northeast", "Kuzeydoğu bölgesi (0/1 gösterge)"),
        ("northcen", "Kuzey Merkez bölgesi (0/1 gösterge)"),
        ("n", "Gözlem sayısı"),
        ("ucret", "Saatlik ücret (dolar)"),
        ("egitim", "Eğitim (yıl)"),
        ("deneyim", "Deneyim (yıl)"),
        ("kidem", "Kıdem (yıl)"),
        ("erkek", "Erkek: tahmin edilen ln(ücret)"),
        ("kadin", "Kadın: tahmin edilen ln(ücret)"),
        ("toplam", "Dört bölge kuklasının toplamı"),
    ),
    consistency_notes=(
        "Veri notlardaki gibi WAGE1'dir; notlardaki kod, bölüm betiği, uygulama ve üretilen kod veriyi wooldridge "
        "paketinden okur (Bölüm 10 betiği önceden data/ CSV kopyasını okuyordu; basılı sayıların hiçbiri değişmedi).",
        "Kod 10.2'de sabit terimin standart hatası 0.2100 ve t istatistiği 33.806'dır (eski metin 0.2117 ve 33.536 "
        "yazıyordu; iki veri kaynağında da 0.2100); notlar ve sunum düzeltildi.",
        "Tablo 10.1'e deneyim sütunu eklendi (metin eğitim, deneyim ve kıdemi karşılaştırır). Şekil 10.2 artık tahmin "
        "edilen log ücret doğrularını gösterir; önceki şekil exp(tahmin) çiziyordu ve doğrular paralel değildi.",
        "Bölge kuklaları uygulamada açıkça kurulur (northeast = 1 − northcen − south − west); notlardaki "
        "C(region, Treatment(...)) yazımıyla katsayılar aynıdır. İki referansla tahmin edilen değerlerin farkı "
        "makineye bağlı yuvarlama düzeyindedir (10⁻¹²'den küçük).",
        "Standart hatalar klasik EKK standart hatalarıdır; heteroskedastisiteye dayanıklı çıkarım Konu 12'de.",
    ),
)
