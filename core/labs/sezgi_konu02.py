"""Konu 2 Sezgi deneyleri: veri üretim süreci bilinen kontrollü simülasyonlar.

Deney 1  Rastgele atama ve gönüllü katılım                         (Notlar §2.7, §2.10, §2.11)
Deney 2  Karıştırıcı faktör: aynı havadaki günleri karşılaştırmak   (Notlar §2.9, §2.10)
Deney 3  Ortak eğilim: birlikte yükselen iki bağımsız seri          (Notlar §2.3, §2.8)

Deney 1'in düzeyleri JTRAIN2'ye benzer (445 kişi, yaklaşık %40 program grubu, bin ABD doları); programın gerçek
etkisi simülasyonda bilinir. Konu 2'de regresyon ve test yoktur: deneyler grup ortalamalarını, farkları ve
korelasyonu bilinen gerçekle karşılaştırır. Tohum 305.
"""

from __future__ import annotations

from core.labs import expr as E
from core.labs.runner import LabState
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, percent, plain
from core.labs.spec import (
    CompareBarChart,
    Derive,
    Draw,
    JoinColumns,
    LineChart,
    NewSample,
    NoteRef,
    PairStatistic,
    Scalar,
    ScalarTable,
    ScatterPlot,
    Statistic,
)

SEED = 305
TOPIC = "konu02"


def _rounded(parameters: Parameters, key: str, decimals: int) -> float:
    """Kaydırıcı değeri, kodda 0,30000000000000004 gibi yazımlar olmasın diye yuvarlanır."""

    return round(float(parameters[key]), decimals)


def _tex(value: float, decimals: int = 2) -> str:
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return text.replace(".", "{,}")


def _short(value: float, decimals: int = 2) -> str:
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return text.replace(".", ",").replace("-", "−")


def _difference(frame: str, outcome: str, group: str, suffix: str, label: str) -> tuple:
    """İki grubun ortalaması (grup = 1 ve grup = 0) ve farkı."""

    return (
        Statistic(frame, outcome, "mean", f"ort1_{suffix}", f"{label}: grup = 1 ortalaması", where=(group, 1),
                  decimals=3),
        Statistic(frame, outcome, "mean", f"ort0_{suffix}", f"{label}: grup = 0 ortalaması", where=(group, 0),
                  decimals=3),
        Scalar(f"fark_{suffix}", E.sub(E.ref(f"ort1_{suffix}"), E.ref(f"ort0_{suffix}")), f"{label}: fark",
               decimals=3),
    )


# --- Deney 1: rastgele atama ve gönüllü katılım ---------------------------------------------------------

BASE_EARNINGS, MOTIVATION_EFFECT, NOISE = 4.5, 2.0, 5.0
"""Programsız kazanç: 4,5 + 2·motivasyon + e, e ~ N(0, 5²) (bin ABD doları)."""
SHARE = 0.4
"""Program grubunun beklenen payı; JTRAIN2'de 185/445 ≈ 0,42."""


def _assignment_settings(parameters: Parameters) -> tuple[float, float, int]:
    return _rounded(parameters, "tau", 1), _rounded(parameters, "secilim", 1), int(parameters["n"])


def _build_assignment(parameters: Parameters) -> tuple:
    effect, strength, n = _assignment_settings(parameters)
    threshold = E.mul(E.norminv(1 - SHARE), E.sqrt(E.add(E.power(strength, 2), 1)))
    return (
        NewSample("kisiler", n, SEED),
        Draw("kisiler", "motivasyon", "normal", 0, 1, "Motivasyon (araştırmacı gözlemez)"),
        Draw("kisiler", "e", "normal", 0, NOISE, "Diğer etkiler (bin dolar)"),
        Draw("kisiler", "u", "uniform", 0, 1, "Kura için tek-düze sayı"),
        Draw("kisiler", "w", "normal", 0, 1, "Katılım kararındaki diğer etkenler"),
        Derive("kisiler", "kazanc0", E.add(E.add(BASE_EARNINGS, E.mul(MOTIVATION_EFFECT, E.var("motivasyon"))),
                                           E.var("e")), "Programa katılmazsa kazanç: 4,5 + 2·motivasyon + e"),
        Derive("kisiler", "d_rastgele", E.compare("lt", E.var("u"), SHARE), "Rastgele atama: kura ile program (1)"),
        Derive("kisiler", "d_gonullu",
               E.compare("gt", E.add(E.mul(strength, E.var("motivasyon")), E.var("w")), threshold),
               "Gönüllü katılım: s·motivasyon + w eşiği aşarsa program (1)"),
        Derive("kisiler", "kazanc_rastgele", E.add(E.var("kazanc0"), E.mul(effect, E.var("d_rastgele"))),
               "Rastgele atamada gözlenen kazanç"),
        Derive("kisiler", "kazanc_gonullu", E.add(E.var("kazanc0"), E.mul(effect, E.var("d_gonullu"))),
               "Gönüllü katılımda gözlenen kazanç"),
        *_difference("kisiler", "kazanc_rastgele", "d_rastgele", "r", "Rastgele atama, kazanç"),
        *_difference("kisiler", "kazanc_gonullu", "d_gonullu", "g", "Gönüllü katılım, kazanç"),
        *_difference("kisiler", "motivasyon", "d_rastgele", "mr", "Rastgele atama, motivasyon"),
        *_difference("kisiler", "motivasyon", "d_gonullu", "mg", "Gönüllü katılım, motivasyon"),
        Statistic("kisiler", "d_rastgele", "mean", "pay_r", "Rastgele atama: program grubunun payı", decimals=3),
        Statistic("kisiler", "d_gonullu", "mean", "pay_g", "Gönüllü katılım: program grubunun payı", decimals=3),
        ScalarTable((("Programsız grup", E.ref("ort0_r")), ("Program grubu", E.ref("ort1_r"))), "grafik_r", decimals=3),
        ScalarTable((("Programsız grup", E.ref("ort0_g")), ("Program grubu", E.ref("ort1_g"))), "grafik_g", decimals=3),
        ScalarTable((
            ("Programsız grubun ortalama kazancı (bin dolar)", E.ref("ort0_r")),
            ("Program grubunun ortalama kazancı (bin dolar)", E.ref("ort1_r")),
            ("Gözlenen fark (bin dolar)", E.ref("fark_r")),
            ("Program grubunun payı", E.ref("pay_r")),
            ("Motivasyon farkı (program − programsız)", E.ref("fark_mr")),
        ), "ozet_r", decimals=3),
        ScalarTable((
            ("Programsız grubun ortalama kazancı (bin dolar)", E.ref("ort0_g")),
            ("Program grubunun ortalama kazancı (bin dolar)", E.ref("ort1_g")),
            ("Gözlenen fark (bin dolar)", E.ref("fark_g")),
            ("Program grubunun payı", E.ref("pay_g")),
            ("Motivasyon farkı (program − programsız)", E.ref("fark_mg")),
        ), "ozet_g", decimals=3),
        JoinColumns("ozet", (("Rastgele atama", "ozet_r", "deger"), ("Gönüllü katılım", "ozet_g", "deger")),
                    decimals=3),
        CompareBarChart((("Rastgele atama", "grafik_r"), ("Gönüllü katılım", "grafik_g")), "deger",
                        "Tasarım", "Ortalama kazanç (bin dolar)", "Aynı kişiler, iki tasarım: grup ortalamaları",
                        decimals=2),
    )


def _assignment_dgp(parameters: Parameters) -> tuple[str, ...]:
    effect, strength, n = _assignment_settings(parameters)
    return (
        rf"\text{{kazanç}}_i = 4{{,}}5 + 2\,\text{{motivasyon}}_i + \tau\,D_i + e_i, \qquad \tau = {_tex(effect, 1)}, "
        rf"\qquad i = 1, \dots, n = {n}",
        r"\text{motivasyon}_i \sim N(0,\ 1), \qquad e_i \sim N(0,\ 5^2)",
        r"\text{Rastgele atama: } D_i = 1 \text{ eğer } u_i < 0{,}4, \qquad u_i \sim U(0,\ 1)",
        rf"\text{{Gönüllü katılım: }} D_i = 1 \text{{ eğer }} s\,\text{{motivasyon}}_i + w_i > c, \qquad "
        rf"s = {_tex(strength, 1)}, \quad w_i \sim N(0,\ 1)",
    )


def _assignment_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    effect = _assignment_settings(parameters)[0]
    s = state.scalars
    return (
        SimMetric("Gerçek etki τ", f"{plain(effect, 2)}", "Programın kazanca gerçek etkisi (bin dolar), veri üretim süreci."),
        SimMetric("Rastgele: fark", plain(s["fark_r"], 2),
                  "Kura ile atanan program grubu − programsız grup: ortalama kazanç farkı (bin dolar)."),
        SimMetric("Gönüllü: fark", plain(s["fark_g"], 2),
                  "Gönüllü katılanlar − katılmayanlar: ortalama kazanç farkı (bin dolar)."),
        SimMetric("Motivasyon farkı", plain(s["fark_mg"], 2),
                  "Gönüllü tasarımda katılanlar − katılmayanlar: ortalama motivasyon farkı (standart sapma birimi)."),
    )


def _assignment_takeaway(state: LabState, parameters: Parameters) -> str:
    effect, strength, _ = _assignment_settings(parameters)
    s = state.scalars
    random_part = (f"Rastgele atamada kimin programa gireceğini kura belirler: gruplar motivasyonda ancak şans eseri "
                   f"farklıdır (fark {plain(s['fark_mr'], 2)}). Gözlenen fark {plain(s['fark_r'], 2)} bin dolar, gerçek "
                   f"etki {plain(effect, 2)} bin dolar: aradaki fark örneklemden gelir.")
    if strength == 0:
        return (
            f"s = 0: katılım kararı motivasyondan bağımsız; gönüllü katılım da kura gibi işler. {random_part} Gönüllü "
            f"tasarımdaki fark ({plain(s['fark_g'], 2)} bin dolar) da gerçek etkiden yalnız örneklem nedeniyle ayrılır "
            "(§2.11)."
        )
    if s["fark_mg"] < 0.1:
        return (
            f"Seçilim bu ayarda zayıf (s = {_short(strength, 1)}): gönüllü tasarımda katılanlar ile katılmayanların "
            f"motivasyon farkı {plain(s['fark_mg'], 2)} standart sapma. Gönüllü tasarımdaki fark "
            f"{plain(s['fark_g'], 2)} bin dolar. {random_part} s'yi artırarak seçilimin etkisini izleyin (§2.10, §2.11)."
        )
    return (
        f"Gönüllü katılımda motivasyonu yüksek kişiler programa daha çok katılıyor: katılanların ortalama motivasyonu "
        f"katılmayanlardan {plain(s['fark_mg'], 2)} standart sapma yüksek. Bu kişiler programsız da daha çok "
        f"kazanırdı; bu yüzden gönüllü tasarımdaki fark ({plain(s['fark_g'], 2)} bin dolar) programın etkisi ile "
        f"başlangıç farkını birlikte içerir. {random_part} Seçilim, karşılaştırılan grupların nasıl oluştuğuyla "
        "ilgilidir (§2.10, §2.11)."
    )


ASSIGNMENT = SimExperiment(
    topic_key=TOPIC,
    number=1,
    title="Rastgele atama ve gönüllü katılım",
    question="Aynı kişiler için bir iş eğitimi programına katılım kurayla belirlenseydi ya da kişiler gönüllü olarak "
             "katılsaydı, iki grubun ortalama kazanç farkı programın gerçek etkisini ne kadar iyi gösterirdi?",
    note=NoteRef("2.11", objects=("§2.7", "§2.10")),
    parameters=(
        SimParameter("tau", "Programın gerçek etkisi τ (bin dolar)", 0, 4, 1.8, 0.1,
                     "JTRAIN2'de gözlenen fark 1,794 bin dolar.", decimals=1),
        SimParameter("secilim", "Seçilimin gücü s", 0, 2, 1, 0.1,
                     "s = 0: gönüllü katılım motivasyondan bağımsız. s büyüdükçe motivasyonu yüksek olanlar katılır.",
                     decimals=1),
        SimParameter("n", "Kişi sayısı n", 100, 3000, 445, 5, "JTRAIN2'de 445 kişi.", integer=True, decimals=0),
    ),
    dgp=_assignment_dgp,
    dgp_note=(
        "Motivasyon kazancı artırır ve araştırmacının verisinde yoktur. İki tasarımda kişiler ve programsız kazançları "
        "aynıdır; yalnız programa kimin katıldığı farklıdır. Eşik c, program grubunun beklenen payı iki tasarımda da "
        "%40 olacak biçimde seçilir: c = 0,2533·√(s² + 1); 0,2533, standart normal dağılımda değerlerin %60'ının "
        "altında kaldığı noktadır. Basitlik için kazanç normal dağılımla üretilir; bu yüzden bazı kişilerin kazancı "
        "negatif çıkabilir (gerçek kazanç negatif olamaz). Deneyin konusu grup ortalamalarının farkıdır; bu "
        "basitleştirme sonucu değiştirmez."
    ),
    look_at=(
        "**Sütun grafiği** — iki tasarımda programsız ve program gruplarının ortalama kazancı.",
        "**Özet tablo** — gözlenen fark ve motivasyon farkı: gruplar başlangıçta benzer mi?",
        "**Metrikler** — iki tasarımın farkını gerçek etki τ ile karşılaştırın; sonra s'yi değiştirin.",
    ),
    build=_build_assignment,
    metrics=_assignment_metrics,
    takeaway=_assignment_takeaway,
    tables=(("ozet", "İki tasarımın özeti"),),
    labels=(("motivasyon", "Motivasyon"), ("kazanc0", "Programsız kazanç (bin dolar)")),
)


# --- Deney 2: karıştırıcı faktör ve ceteris paribus karşılaştırması ---------------------------------------

BASE_SALES, HOT_EFFECT, SALES_NOISE = 1000, 400, 100
"""Günlük satış: 1000 + τ·kampanya + 400·sıcak + e, e ~ N(0, 100²) (TL)."""


def _confounder_settings(parameters: Parameters) -> tuple[float, float, int]:
    return _rounded(parameters, "bag", 1), _rounded(parameters, "tau", 0), int(parameters["n"])


def _build_confounder(parameters: Parameters) -> tuple:
    link, effect, n = _confounder_settings(parameters)
    campaign_probability = E.add(0.5, E.mul(link, E.sub(E.var("sicak"), 0.5)))
    groups = (("t_tum", "tum", "Bütün günler"), ("t_soguk", "soguk", "Soğuk günler"),
              ("t_sicak", "sicak", "Sıcak günler"))
    # (kampanyalı, kampanyasız) günlerin koşulu; grup = 2·sıcak + kampanya.
    conditions = {"tum": (("kampanya", 1), ("kampanya", 0)), "soguk": (("grup", 1), ("grup", 0)),
                  "sicak": (("grup", 3), ("grup", 2))}
    means = []
    for _, suffix, label in groups:
        with_campaign, without_campaign = conditions[suffix]
        means += [
            Statistic("gunler", "satis", "mean", f"ort1_{suffix}", f"{label}: kampanyalı günlerin ortalaması",
                      where=with_campaign, decimals=1),
            Statistic("gunler", "satis", "mean", f"ort0_{suffix}", f"{label}: kampanyasız günlerin ortalaması",
                      where=without_campaign, decimals=1),
            Scalar(f"fark_{suffix}", E.sub(E.ref(f"ort1_{suffix}"), E.ref(f"ort0_{suffix}")),
                   f"{label}: kampanyalı − kampanyasız", decimals=1),
        ]
    tables = tuple(
        ScalarTable((("Kampanyasız günler", E.ref(f"ort0_{suffix}")), ("Kampanyalı günler", E.ref(f"ort1_{suffix}"))),
                    name, decimals=1)
        for name, suffix, _ in groups
    )
    summaries = tuple(
        ScalarTable((
            ("Kampanyasız günlerin ortalama satışı (TL)", E.ref(f"ort0_{suffix}")),
            ("Kampanyalı günlerin ortalama satışı (TL)", E.ref(f"ort1_{suffix}")),
            ("Fark (TL)", E.ref(f"fark_{suffix}")),
        ), f"ozet_{suffix}", decimals=1)
        for _, suffix, _ in groups
    )
    return (
        NewSample("gunler", n, SEED),
        Draw("gunler", "u1", "uniform", 0, 1, "Hava için tek-düze sayı"),
        Draw("gunler", "u2", "uniform", 0, 1, "Kampanya kararı için tek-düze sayı"),
        Draw("gunler", "e", "normal", 0, SALES_NOISE, "Diğer etkiler (TL)"),
        Derive("gunler", "sicak", E.compare("lt", E.var("u1"), 0.5), "Sıcak gün mü? (1: evet)"),
        Derive("gunler", "kampanya", E.compare("lt", E.var("u2"), campaign_probability),
               "Kampanya günü mü? (1: evet); olasılık 0,5 + g·(sıcak − 0,5)"),
        Derive("gunler", "satis", E.add(E.add(E.add(BASE_SALES, E.mul(effect, E.var("kampanya"))),
                                              E.mul(HOT_EFFECT, E.var("sicak"))), E.var("e")),
               "Günlük satış: 1000 + τ·kampanya + 400·sıcak + e"),
        Derive("gunler", "grup", E.add(E.mul(2, E.var("sicak")), E.var("kampanya")),
               "Hava ve kampanya bileşimi: 0 soğuk–kampanyasız, 1 soğuk–kampanyalı, 2 sıcak–kampanyasız, "
               "3 sıcak–kampanyalı"),
        *means,
        Statistic("gunler", "sicak", "mean", "sicak_pay1", "Kampanyalı günlerde sıcak günlerin payı",
                  where=("kampanya", 1), decimals=3),
        Statistic("gunler", "sicak", "mean", "sicak_pay0", "Kampanyasız günlerde sıcak günlerin payı",
                  where=("kampanya", 0), decimals=3),
        *tables,
        *summaries,
        JoinColumns("ozet", tuple((label, f"ozet_{suffix}", "deger") for _, suffix, label in groups), decimals=1),
        CompareBarChart(tuple((label, name) for name, _, label in groups), "deger", "Karşılaştırılan günler",
                        "Ortalama satış (TL)", "Kampanyalı ve kampanyasız günler: bütün günler ve aynı havadaki günler",
                        decimals=0),
    )


def _confounder_dgp(parameters: Parameters) -> tuple[str, ...]:
    link, effect, n = _confounder_settings(parameters)
    return (
        rf"\text{{satış}}_i = 1000 + \tau\,\text{{kampanya}}_i + 400\,\text{{sıcak}}_i + e_i, \qquad "
        rf"\tau = {_tex(effect, 0)}, \qquad e_i \sim N(0,\ 100^2), \qquad i = 1, \dots, n = {n}",
        r"P(\text{sıcak}_i = 1) = 0{,}5",
        rf"P(\text{{kampanya}}_i = 1 \mid \text{{sıcak}}_i) = 0{{,}}5 + g\,(\text{{sıcak}}_i - 0{{,}}5), \qquad "
        rf"g = {_tex(link, 1)}",
    )


def _confounder_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    effect = _confounder_settings(parameters)[1]
    s = state.scalars
    return (
        SimMetric("Gerçek etki τ", plain(effect, 0), "Kampanyanın satışa gerçek etkisi (TL), veri üretim süreci."),
        SimMetric("Bütün günler", plain(s["fark_tum"], 0), "Kampanyalı − kampanyasız günlerin ortalama satış farkı."),
        SimMetric("Sıcak günlerde", plain(s["fark_sicak"], 0), "Yalnız sıcak günler: kampanyalı − kampanyasız."),
        SimMetric("Soğuk günlerde", plain(s["fark_soguk"], 0), "Yalnız soğuk günler: kampanyalı − kampanyasız."),
    )


def _confounder_takeaway(state: LabState, parameters: Parameters) -> str:
    link, effect, _ = _confounder_settings(parameters)
    s = state.scalars
    within = (f"Aynı havadaki günleri karşılaştırınca (sıcaklık sabit: ceteris paribus) fark sıcak günlerde "
              f"{plain(s['fark_sicak'], 0)}, soğuk günlerde {plain(s['fark_soguk'], 0)} TL. Bu karşılaştırmalar "
              f"sıcaklığın etkisini içermez; gerçek etkiden ({plain(effect, 0)} TL) yalnız örneklem nedeniyle ayrılır.")
    shares = (f"kampanyalı günlerin {percent(100 * s['sicak_pay1'], 0)} kadarı, kampanyasız günlerin "
              f"{percent(100 * s['sicak_pay0'], 0)} kadarı sıcak")
    if link == 0:
        return (
            f"g = 0: kampanya kararı havadan bağımsız ({shares}). Sıcak günler iki gruba benzer oranda dağıldığı için "
            f"bütün günlerin karşılaştırması ({plain(s['fark_tum'], 0)} TL) sıcaklığın etkisini sistematik olarak "
            f"içermez. {within}"
        )
    return (
        f"Yönetici kampanyayı daha çok sıcak günlerde yapıyor: {shares}. Sıcaklık satışı da artırdığı için karıştırıcı "
        f"faktördür: bütün günlerin karşılaştırması {plain(s['fark_tum'], 0)} TL fark verir, gerçek kampanya etkisi "
        f"{plain(effect, 0)} TL. {within} Diğer koşulları sabit tutmanın bir yolu birbirine benzer birimleri "
        "karşılaştırmaktır (§2.9, §2.10)."
    )


CONFOUNDER = SimExperiment(
    topic_key=TOPIC,
    number=2,
    title="Karıştırıcı faktör: aynı havadaki günleri karşılaştırmak",
    question="Bir dondurmacı kampanyayı daha çok sıcak günlerde yapıyorsa, kampanyalı ve kampanyasız günlerin satış "
             "farkı kampanyanın etkisini gösterir mi? Aynı havadaki günleri karşılaştırmak neyi değiştirir?",
    note=NoteRef("2.10", objects=("§2.9",)),
    parameters=(
        SimParameter("bag", "Kampanya kararının sıcaklığa bağlılığı g", 0, 0.8, 0.6, 0.1,
                     "g = 0: kampanya havadan bağımsız. g = 0,6: sıcak günlerde kampanya olasılığı 0,8, soğuk "
                     "günlerde 0,2.", decimals=1),
        SimParameter("tau", "Kampanyanın gerçek etkisi τ (TL)", 0, 300, 100, 10,
                     "Aynı havadaki bir günde kampanyanın ortalama satış etkisi.", integer=True, decimals=0),
        SimParameter("n", "Gün sayısı n", 200, 2000, 730, 10, "Gözlenen gün sayısı; 730 gün iki yıldır.", integer=True,
                     decimals=0),
    ),
    dgp=_confounder_dgp,
    dgp_note=(
        "Sıcaklık hem kampanya kararını hem satışı etkiler: karıştırıcı faktördür. Kampanyanın satışa etkisi her "
        "havada aynıdır (τ)."
    ),
    look_at=(
        "**Sütun grafiği** — kampanyalı ve kampanyasız günlerin ortalama satışı: bütün günler, soğuk günler, sıcak "
        "günler.",
        "**Özet tablo** — üç karşılaştırmanın farkı.",
        "**Metrikler** — bütün günlerin farkı ile aynı havadaki günlerin farkını gerçek etki τ ile karşılaştırın.",
    ),
    build=_build_confounder,
    metrics=_confounder_metrics,
    takeaway=_confounder_takeaway,
    tables=(("ozet", "Üç karşılaştırma"),),
    labels=(("satis", "Günlük satış (TL)"), ("sicak", "Sıcak gün"), ("kampanya", "Kampanya günü")),
)


# --- Deney 3: ortak eğilim ----------------------------------------------------------------------------

FIRST_YEAR = 1980
TREND_A, TREND_B, NOISE_A, NOISE_B = 1.0, 1.5, 4.0, 6.0
"""A_t = 100 + 1,0·k·t + 4·ε, B_t = 100 + 1,5·k·t + 6·η: iki seri yalnız zamanla birlikte artar."""


def _trend_settings(parameters: Parameters) -> tuple[float, int]:
    return _rounded(parameters, "egilim", 1), int(parameters["T"])


def _build_trend(parameters: Parameters) -> tuple:
    strength, years = _trend_settings(parameters)
    trend_a = E.add(100, E.mul(TREND_A * strength, E.var("id")))
    trend_b = E.add(100, E.mul(TREND_B * strength, E.var("id")))
    return (
        NewSample("yillar", years, SEED),
        Draw("yillar", "eps", "normal", 0, 1, "A serisinin dalgalanması"),
        Draw("yillar", "eta", "normal", 0, 1, "B serisinin dalgalanması (A'dan bağımsız)"),
        Derive("yillar", "yil", E.add(FIRST_YEAR, E.var("id")), "Yıl"),
        Derive("yillar", "A", E.add(trend_a, E.mul(NOISE_A, E.var("eps"))), "Seri A: kahve tüketimi endeksi"),
        Derive("yillar", "B", E.add(trend_b, E.mul(NOISE_B, E.var("eta"))), "Seri B: üniversite mezunu sayısı endeksi"),
        Derive("yillar", "A_sapma", E.sub(E.var("A"), trend_a), "A'nın bilinen eğilimden sapması"),
        Derive("yillar", "B_sapma", E.sub(E.var("B"), trend_b), "B'nin bilinen eğilimden sapması"),
        PairStatistic("yillar", "A", "B", "corr", "r", "Korelasyon r(A, B)", decimals=3),
        PairStatistic("yillar", "A_sapma", "B_sapma", "corr", "r_sapma", "Eğilimden sapmaların korelasyonu",
                      decimals=3),
        LineChart("yillar", "yil", "A", "Yıl", "Endeks", "İki seri zaman içinde", markers=False,
                  series=(("B", "Seri B: üniversite mezunu sayısı endeksi"),),
                  legend="Seri A: kahve tüketimi endeksi"),
        ScatterPlot("yillar", "A", "B", "Seri A: kahve tüketimi endeksi", "Seri B: üniversite mezunu sayısı endeksi",
                    "Aynı yılların A ve B değerleri", fit_line=True, size=8, opacity=0.8),
    )


def _trend_dgp(parameters: Parameters) -> tuple[str, ...]:
    strength, years = _trend_settings(parameters)
    return (
        rf"A_t = 100 + 1{{,}}0\,k\,t + 4\,\varepsilon_t, \qquad B_t = 100 + 1{{,}}5\,k\,t + 6\,\eta_t, \qquad "
        rf"k = {_tex(strength, 1)}, \qquad t = 1, \dots, T = {years}",
        r"\varepsilon_t \sim N(0,\ 1), \quad \eta_t \sim N(0,\ 1), \quad \varepsilon \text{ ve } \eta "
        r"\text{ bağımsız: A, B'yi etkilemez; B, A'yı etkilemez}",
    )


def _trend_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    s = state.scalars
    _, years = _trend_settings(parameters)
    return (
        SimMetric("r(A, B)", plain(s["r"], 3), "Aynı yılların A ve B değerleri arasındaki korelasyon."),
        SimMetric("Sapmaların r'si", plain(s["r_sapma"], 3),
                  "Bilinen eğilim çıkarıldıktan sonra kalan dalgalanmaların korelasyonu."),
        SimMetric("Gerçek etki", "yok", "Veri üretim sürecinde A ile B arasında hiçbir etki yoktur."),
        SimMetric("Yıl sayısı T", str(years), f"{FIRST_YEAR + 1}–{FIRST_YEAR + years}."),
    )


def _trend_takeaway(state: LabState, parameters: Parameters) -> str:
    strength, _ = _trend_settings(parameters)
    s = state.scalars
    if strength == 0:
        return (
            f"k = 0: ortak eğilim yok. İki seri birbirinden bağımsız dalgalanır; korelasyon r = {plain(s['r'], 3)} "
            "yalnız örneklem nedeniyle sıfırdan farklıdır. Eğilimi artırarak iki bağımsız serinin nasıl birlikte "
            "hareket etmeye başladığını izleyin."
        )
    if s["r"] >= 0.5:
        strength_text = "ortak eğilim güçlü bir birlikte hareket üretir"
    elif s["r"] > 0:
        strength_text = ("eğilim bu ayarda zayıf ya da seri kısa olduğu için birlikte hareket sınırlıdır; k'yı veya "
                         "T'yi artırın")
    else:
        strength_text = ("eğilim bu ayarda dalgalanmalara göre çok zayıf olduğu için korelasyonun işareti bile "
                         "şansa bağlıdır; k'yı artırın")
    return (
        f"A ve B birbirini etkilemiyor; ikisinin de ortalaması zamanla artıyor. Korelasyon r = {plain(s['r'], 3)}: "
        f"{strength_text}. "
        f"Bilinen eğilim çıkarıldığında kalan dalgalanmaların korelasyonu {plain(s['r_sapma'], 3)}. Zaman serisinde "
        "birlikte yükselme tek başına nedensellik göstermez; ortak eğilim veya üçüncü bir faktör olabilir "
        "(§2.3, §2.8)."
    )


TREND = SimExperiment(
    topic_key=TOPIC,
    number=3,
    title="Ortak eğilim: birlikte yükselen iki bağımsız seri",
    question="Birbirini hiç etkilemeyen yıllık iki seri, yalnız ikisi de zamanla arttığı için güçlü bir ilişki "
             "gösterebilir mi?",
    note=NoteRef("2.3", objects=("§2.8",)),
    parameters=(
        SimParameter("egilim", "Ortak eğilimin gücü k", 0, 2, 1, 0.1, "k = 0: eğilim yok.", decimals=1),
        SimParameter("T", "Yıl sayısı T", 10, 100, 40, 5, "Serilerin uzunluğu.", integer=True, decimals=0),
    ),
    dgp=_trend_dgp,
    dgp_note="İki serinin dalgalanmaları birbirinden bağımsız çekilir; ortak olan tek şey zamanla artan eğilimdir.",
    look_at=(
        "**Çizgi grafiği** — iki seri zaman içinde birlikte yükseliyor mu?",
        "**Saçılım grafiği** — aynı yılların A ve B değerleri: noktalar bir doğru çevresinde mi?",
        "**Metrikler** — r(A, B) ile eğilim çıkarıldıktan sonraki korelasyon.",
    ),
    build=_build_trend,
    metrics=_trend_metrics,
    takeaway=_trend_takeaway,
    labels=(("yil", "Yıl"), ("A", "Seri A"), ("B", "Seri B")),
)


KONU02_EXPERIMENTS = (ASSIGNMENT, CONFOUNDER, TREND)
