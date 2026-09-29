"""Konu 9 uygulaması: ölçekleme, logaritmik modeller, karesel terimler ve model seçimi (gerçek veri).

Bölüm 9'un çözümlü örnekleri bölüm sırasıyla: ölçü birimi değişikliği (§9.2, Tablo 9.2), standartlaştırılmış
katsayılar (§9.2, Tablo 9.3, Şekil 9.1), log–düzey modelinde yaklaşık ve tam yüzde (§9.3, Tablo 9.4), karesel ücret
modeli ve karesel terimlerin ortak testi (§9.5, Kod 9.1–9.2), deneyimin marjinal etkisi ve dönüm noktaları (§9.5,
Tablo 9.5, Şekil 9.2–9.3), merkezleme (§9.6, Tablo 9.6), model karşılaştırması (§9.7, Tablo 9.7, Şekil 9.4) ve makale
tablosu (§9.8, Tablo 9.8). Her ``Check`` notlarda basılı bir sayıdır; değer notlardan kopyalanmıştır, hesaplanmamıştır.

Etkileşim: ücretin ve eğitimin ölçü birimi (Adım 1), standartlaştırılan modelin açıklayıcıları (Adım 2), katsayı β ve
X'teki değişim ΔX (Adım 3), marjinal etkinin hesaplandığı deneyim düzeyi (Adım 5), merkez noktası (Adım 6) ve
karşılaştırmaya eklenen beşinci model (Adım 7). Standart hatalar klasik (homoskedastik) EKK standart hatalarıdır.
"""

from __future__ import annotations

from core import wooldridge_data as W
from core.labs import expr as E
from core.labs.sezgi import plain
from core.labs.spec import (
    INTERCEPT,
    OLS,
    BarChart,
    Check,
    Choice,
    CoefficientTable,
    CoefTarget,
    CopyFrame,
    Derive,
    InlineData,
    CellTarget,
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
    NumberChoice,
    RegressionTable,
    Residuals,
    Scalar,
    ScalarTable,
    ScalarTarget,
    ScatterPlot,
    ShowFrame,
    ShowModel,
    Statistic,
    Support,
    TableTarget,
    PairStatistic,
    interactive_step,
)

DATA = "wage1"
REGRESSORS = ("educ", "exper", "tenure")
"""Bölüm 9'un doğrusal ücret modeli (M₁): açıklayıcılar eğitim, deneyim ve kıdem (n = 526)."""
QUADRATIC = ("educ", "exper", "expersq", "tenure", "tenursq")
"""Karesel ücret modeli (M₄, Kod 9.1): ln(ücret) ~ eğitim + deneyim + deneyim² + kıdem + kıdem²."""
WORDS = {"educ": "eğitim", "exper": "deneyim", "tenure": "kıdem", "numdep": "bakmakla yükümlü kişi sayısı"}
TITLES = {"educ": "Eğitim", "exper": "Deneyim", "tenure": "Kıdem", "numdep": "Bakmakla yükümlü kişi sayısı"}


def _scalar(name: str, expected: float, label: str, decimals: int) -> Check:
    return Check(label, ScalarTarget(name), expected, decimals)


def _p(value: float, decimals: int = 3) -> str:
    """p-değerinin eşitlikli yazımı: "p = 0,017"; basamakta sıfıra yuvarlanıyorsa "p < 0,001"."""

    if value < 0.5 * 10 ** -decimals:
        return "p < " + plain(10 ** -decimals, decimals)
    return f"p = {plain(value, decimals)}"


def _load() -> LoadWooldridge:
    return LoadWooldridge(DATA, "WAGE1 veri seti (Wooldridge, 2020): 526 çalışan")


# --- Adım 1: ölçü birimi değişikliği ----------------------------------------------------------------------

Y_UNITS = {
    "sent": ("wage_sent", "sent", E.mul(100, E.var("wage")), "Ücret sent cinsinden: 100 × wage"),
    "bin": ("wage_bin", "bin dolar", E.div(E.var("wage"), 1000), "Ücret bin dolar cinsinden: wage / 1000"),
}
X_UNITS = {
    "on_yil": ("educ_on", "on yıl", E.div(E.var("educ"), 10), "Eğitim on yıllık birimle: educ / 10"),
    "ay": ("educ_ay", "ay", E.mul(12, E.var("educ")), "Eğitim ay cinsinden: 12 × educ"),
}
Y_UNIT_1 = Choice("adim1_y", "Ücretin yeni ölçü birimi", (("sent", "Sent (dolar × 100)"),
                                                          ("bin", "Bin dolar (dolar ÷ 1.000)")),
                  "sent", help="Notlarda ücret dolardan sente çevrilir (Tablo 9.2, ikinci satır).")
X_UNIT_1 = Choice("adim1_x", "Eğitimin yeni ölçü birimi", (("on_yil", "On yıl (yıl ÷ 10)"), ("ay", "Ay (yıl × 12)")),
                  "on_yil", help="Notlarda eğitim yıldan on yıllık birime çevrilir (Tablo 9.2, üçüncü satır).")


def _unit_decimals(y_unit: str, x_unit: str) -> tuple[int, int]:
    """Ücret bin dolara çevrilince katsayılar 1.000'e bölünür, eğitim aya çevrilince eğitim katsayısı 12'ye bölünür:
    küçülen sayıların bilgisi kaybolmasın diye ek ondalık (0,000599 "0,001" yazılmaz)."""

    return (3 if y_unit == "bin" else 0), (1 if x_unit == "ay" else 0)


def _row_labels(y_unit: str, x_unit: str) -> tuple[str, str, str]:
    return ("Ücret dolar, eğitim yıl", f"Ücret {Y_UNITS[y_unit][1]}, eğitim yıl",
            f"Ücret dolar, eğitim {X_UNITS[x_unit][1]}")


def _scaling(choices) -> tuple:
    y_unit, x_unit = choices["adim1_y"], choices["adim1_x"]
    y_column, y_name, y_expr, y_comment = Y_UNITS[y_unit]
    x_column, x_name, x_expr, x_comment = X_UNITS[x_unit]
    rows = _row_labels(y_unit, x_unit)
    extra_y, extra_x = _unit_decimals(y_unit, x_unit)
    notes = y_unit == "sent" and x_unit == "on_yil"
    # (satır, model, terim, ek basamak, skaler öneki): notlardan farklı birimde notlardaki satır seçilenin üstünde
    table_rows = [(rows[0], "m_dolar", "educ", 0, "1")]
    side: list = []
    if y_unit != "sent":
        column, _, expr, comment = Y_UNITS["sent"]
        side += [Derive("olcek", column, expr, comment),
                 OLS("m_y_n", "olcek", column, REGRESSORS, f"Notlar: ücret sent, eğitim yıl: {column} ~ educ + exper + "
                     "tenure")]
        table_rows.append((_row_labels("sent", "on_yil")[1] + " (notlar)", "m_y_n", "educ", 0, "n2"))
    table_rows.append((rows[1], "m_y", "educ", extra_y, "2"))
    if x_unit != "on_yil":
        column, _, expr, comment = X_UNITS["on_yil"]
        side += [Derive("olcek", column, expr, comment),
                 OLS("m_x_n", "olcek", "wage", (column, "exper", "tenure"),
                     f"Notlar: ücret dolar, eğitim on yıl: wage ~ {column} + exper + tenure")]
        table_rows.append((_row_labels("sent", "on_yil")[2] + " (notlar)", "m_x_n", column, 0, "n3"))
    table_rows.append((rows[2], "m_x", x_column, extra_x, "3"))
    values = []
    for label, model, term, extra, suffix in table_rows:
        values += [
            ModelValue(f"b_{suffix}", model, "coef", f"{label}: eğitim katsayısı", term=term, decimals=3 + extra),
            ModelValue(f"t_{suffix}", model, "t", f"{label}: eğitim katsayısının t istatistiği", term=term,
                       decimals=3),
            ModelValue(f"p_{suffix}", model, "p", f"{label}: eğitim katsayısının p-değeri", term=term, decimals=3),
            ModelValue(f"r2_{suffix}", model, "r2", f"{label}: R²", decimals=3),
        ]
    widest = 3 + max(extra_y, extra_x)
    return (
        _load(),
        OLS("m_dolar", DATA, "wage", REGRESSORS, "Ücret dolar, eğitim yıl: wage ~ educ + exper + tenure"),
        CopyFrame("olcek", DATA, "Birim dönüşümleri için veri setinin kopyası (özgün veri değişmez)"),
        Derive("olcek", y_column, y_expr, y_comment),
        Derive("olcek", x_column, x_expr, x_comment),
        OLS("m_y", "olcek", y_column, REGRESSORS, f"Ücret {y_name}, eğitim yıl: {y_column} ~ educ + exper + tenure"),
        OLS("m_x", "olcek", "wage", (x_column, "exper", "tenure"),
            f"Ücret dolar, eğitim {x_name}: wage ~ {x_column} + exper + tenure"),
        *side,
        ModelValue("a_1", "m_dolar", "coef", "Ücret dolar: sabit terim", term=INTERCEPT, decimals=3),
        ModelValue("a_2", "m_y", "coef", f"Ücret {y_name}: sabit terim", term=INTERCEPT, decimals=3 + extra_y),
        ModelValue("sh_1", "m_dolar", "se", "Ücret dolar: eğitim katsayısının standart hatası", term="educ", decimals=4),
        ModelValue("sh_2", "m_y", "se", f"Ücret {y_name}: eğitim katsayısının standart hatası", term="educ",
                   decimals=4 + extra_y),
        *values,
        *(ScalarTable(tuple((label, E.ref(f"{prefix}_{suffix}")) for label, _, _, _, suffix in table_rows), result,
                      decimals=digits)
          for prefix, result, digits in (("b", "katsayi92", widest), ("t", "t92", 3), ("p", "p92", 4),
                                         ("r2", "r292", 3))),
        JoinColumns("tablo92", (("Eğitim katsayısı", "katsayi92", "deger"), ("t", "t92", "deger"), ("p", "p92", "deger"),
                                ("R²", "r292", "deger")), decimals=3, heading="Model", p_columns=("p",),
                    column_decimals=(("Eğitim katsayısı", widest),),
                    title="Tablo 9.2: WAGE1 modelinde ölçü birimi değişikliklerinin etkisi" if notes
                    else "Ölçü birimi değişikliklerinin etkisi: notlardaki ve seçtiğiniz birimler"),
    )


def _scaling_note(state, choices) -> str:
    y_unit, x_unit = choices["adim1_y"], choices["adim1_x"]
    s = state.scalars
    y_name, x_name = Y_UNITS[y_unit][1], X_UNITS[x_unit][1]
    extra_y, extra_x = _unit_decimals(y_unit, x_unit)
    factor = "100 ile çarpılır" if y_unit == "sent" else "1.000'e bölünür"
    x_factor = "10'a bölünür" if x_unit == "on_yil" else "12 ile çarpılır"
    x_effect = ("bir birimlik artış artık on yıllık eğitim artışıdır" if x_unit == "on_yil"
                else "bir birimlik artış artık bir aylık eğitim artışıdır")
    return (f"Ücret {y_name} cinsinden yazılınca bağımlı değişken {factor}; sabit terim ({plain(s['a_1'], 3)} → "
            f"{plain(s['a_2'], 3 + extra_y)}), eğitim katsayısı ({plain(s['b_1'], 3)} → {plain(s['b_2'], 3 + extra_y)}) "
            f"ve standart hatası ({plain(s['sh_1'], 4)} → {plain(s['sh_2'], 4 + extra_y)}) aynı oranda değişir. Eğitim "
            f"{x_name} biriminde yazılınca açıklayıcı değişken {x_factor}; {x_effect} ve katsayı "
            f"{plain(s['b_3'], 3 + extra_x)} olur. Tablodaki bütün modellerde t = {plain(s['t_1'], 3)} ve "
            f"R² = {plain(s['r2_1'], 3)}: katsayı ile standart hata aynı ölçekte "
            "değiştiği için oranları değişmez. Birim dönüşümü ekonomik ilişkiyi ve istatistiksel kanıtı değiştirmez; "
            "yalnız katsayının hangi birimle okunacağını değiştirir (§9.2).")


# --- Adım 2: standartlaştırılmış katsayılar ------------------------------------------------------------------

STD_OPTIONS = ("educ", "exper", "tenure", "numdep")
STD_2 = MultiChoice("adim2_x", "Standartlaştırılan modelin açıklayıcıları", W.options(DATA, STD_OPTIONS), REGRESSORS,
                    help="Notlarda eğitim, deneyim ve kıdem; bağımlı değişken ln(ücret) (Tablo 9.3).")


def _standardized(choices) -> tuple:
    chosen = tuple(choices["adim2_x"])
    notes = chosen == REGRESSORS
    operations: list = [CopyFrame("std", DATA, "Standartlaştırma için veri setinin kopyası (özgün veri değişmez)")]
    # Notlardan farklı seçimde notlardaki modelin değişkenleri de standartlaştırılır (iki tablo yan yana).
    for name in ("lwage", *dict.fromkeys((*chosen, *(() if notes else REGRESSORS)))):
        label = "ln(ücret)" if name == "lwage" else WORDS[name]
        title = label if name == "lwage" else TITLES[name]  # "ln(ücret)" cümle başında da küçük harfle
        operations += [
            Statistic("std", name, "mean", f"ort_{name}", f"{title}: örneklem ortalaması"),
            Statistic("std", name, "std", f"ss_{name}", f"{title}: örneklem standart sapması (payda n − 1)"),
            Derive("std", f"z_{name}", E.div(E.sub(E.var(name), E.ref(f"ort_{name}")), E.ref(f"ss_{name}")),
                   f"Standartlaştırılmış {label}: (x − ortalama) / standart sapma"),
        ]
    terms = tuple(f"z_{name}" for name in chosen)
    if not notes:
        noted = tuple(f"z_{name}" for name in REGRESSORS)
        operations += [
            OLS("m_std_n", "std", "z_lwage", noted, "Notlardaki model: z_lwage ~ " + " + ".join(noted)),
            CoefficientTable("m_std_n", noted, "tablo93_n", "Notlar — Tablo 9.3: WAGE1 modelinde standartlaştırılmış "
                             "katsayılar", decimals=3, t_decimals=3),
        ]
    return (
        *operations,
        OLS("m_std", "std", "z_lwage", terms, "Standartlaştırılmış model: z_lwage ~ " + " + ".join(terms)),
        CoefficientTable("m_std", terms, "tablo93",
                         "Tablo 9.3: WAGE1 modelinde standartlaştırılmış katsayılar" if notes
                         else "Standartlaştırılmış katsayılar (seçtiğiniz model)", decimals=3, t_decimals=3),
        *(ModelValue(f"bz_{name}", "m_std", "coef", f"{TITLES[name]}: standartlaştırılmış katsayı", term=f"z_{name}",
                     decimals=3) for name in chosen),
        ScalarTable(tuple((TITLES[name], E.ref(f"bz_{name}")) for name in chosen), "std_katsayi", decimals=3,
                    heading="Değişken", value="Standartlaştırılmış katsayı"),
        BarChart("std_katsayi", "deger", "Açıklayıcı değişken", "Standartlaştırılmış katsayı",
                 "Şekil 9.1: WAGE1, ln(ücret) modeli: standartlaştırılmış katsayılar" if notes
                 else "Standartlaştırılmış katsayılar (seçtiğiniz model)", decimals=3),
    )


def _standardized_note(state, choices) -> str:
    chosen = tuple(choices["adim2_x"])
    s = state.scalars
    if len(chosen) == 1:
        (name,) = chosen
        value = s[f"bz_{name}"]
        return (f"Tek açıklayıcıyla standartlaştırılmış katsayı ({plain(value, 3)}), {WORDS[name]} ile ln(ücret) "
                f"arasındaki örneklem korelasyonuna eşittir: {WORDS[name]} bir standart sapma yüksekken ln(ücret) "
                f"ortalama {plain(abs(value), 3)} standart sapma {'yüksektir' if value > 0 else 'düşüktür'}. "
                "Standartlaştırma t ve p-değerlerini değiştirmez; yalnız yorum ölçeğini değiştirir. Katsayı örneklem "
                "standart sapmalarına bağlıdır ve nedensel önem ölçüsü değildir (§9.2).")
    ranked = sorted(chosen, key=lambda name: -abs(s[f"bz_{name}"]))
    largest = ranked[0]
    text = (f"Bu modelde mutlak değerce en büyük standartlaştırılmış katsayı {WORDS[largest]} değişkenindedir "
            f"({plain(s[f'bz_{largest}'], 3)}): {WORDS[largest]} bir standart sapma yüksekken, diğer değişkenler "
            f"sabitken ln(ücret) ortalama {plain(abs(s[f'bz_{largest}']), 3)} standart sapma "
            f"{'yüksektir' if s[f'bz_{largest}'] > 0 else 'düşüktür'}. ")
    return text + ("Standartlaştırma t ve p-değerlerini değiştirmez; yalnız yorum ölçeğini değiştirir. Katsayılar "
                   "örneklem standart sapmalarına bağlıdır: bu sıralama nedensel önem sıralaması değildir ve başka "
                   "bir örneklemde değişebilir (§9.2).")


# --- Adım 3: yaklaşık ve tam yüzde değişim ---------------------------------------------------------------------

TABLE_94 = ((0.02, 1), (0.08, 1), (0.20, 1), (0.08, 4))
"""Tablo 9.4: katsayı β ve X'teki değişim ΔX."""
BETA_3 = NumberChoice("adim3_beta", "Katsayı β", -0.5, 0.5, 0.08, 0.01,
                      help="Log–düzey modelinde X'in katsayısı. Varsayılan Tablo 9.4'ün son satırı (β = 0,08).")
DX_3 = NumberChoice("adim3_dx", "X'teki değişim ΔX", 1, 10, 4, 1, integer=True,
                    help="Varsayılan Tablo 9.4'ün son satırı (ΔX = 4).")


def _exact(choices) -> tuple:
    beta, dx = float(choices["adim3_beta"]), int(choices["adim3_dx"])
    product = E.mul(E.ref("beta_sec"), E.ref("dx_sec"))
    return (
        InlineData("tablo94", ("beta", "dx"), TABLE_94, "Tablo 9.4: katsayı β ve X'teki değişim ΔX"),
        Derive("tablo94", "yaklasik", E.mul(100, E.mul(E.var("beta"), E.var("dx"))), "Yaklaşık yüzde: 100·β·ΔX"),
        Derive("tablo94", "tam", E.mul(100, E.sub(E.exp(E.mul(E.var("beta"), E.var("dx"))), 1)),
               "Tam yüzde: 100·(exp(β·ΔX) − 1)"),
        ShowFrame("tablo94", ("beta", "dx", "yaklasik", "tam"),
                  "Tablo 9.4: log–düzey modelinde yaklaşık ve tam yüzde yorumları", decimals=2),
        Scalar("ornek_yaklasik", E.mul(100, 0.0845), "Eğitim katsayısı 0,0845: yaklaşık yüzde 100 × 0,0845",
               decimals=2, percent=True),
        Scalar("ornek_tam", E.mul(100, E.sub(E.exp(0.0845), 1)), "Eğitim katsayısı 0,0845: tam yüzde 100·(exp(0,0845) − 1)",
               decimals=2, percent=True),
        Scalar("beta_sec", E.const(beta), "Seçilen katsayı β", decimals=2),
        Scalar("dx_sec", E.const(dx), "Seçilen değişim ΔX", decimals=0),
        Scalar("yaklasik_sec", E.mul(100, product), "Seçilen β ve ΔX: yaklaşık yüzde 100·β·ΔX", decimals=2,
               percent=True),
        Scalar("tam_sec", E.mul(100, E.sub(E.exp(product), 1)), "Seçilen β ve ΔX: tam yüzde 100·(exp(β·ΔX) − 1)",
               decimals=2, percent=True),
    )


def _exact_note(state, choices) -> str:
    beta, dx = float(choices["adim3_beta"]), int(choices["adim3_dx"])
    s = state.scalars
    gap = s["tam_sec"] - s["yaklasik_sec"]
    if beta == 0:
        return "β = 0 iken iki yorum da sıfırdır: X'in değişimi ln(Y)'yi değiştirmez (§9.3)."
    return (f"β = {plain(beta, 2)} ve ΔX = {dx} için yaklaşık yorum %{plain(s['yaklasik_sec'], 2)}, tam yorum "
            f"%{plain(s['tam_sec'], 2)}; fark {plain(gap, 2)} yüzde puan. Tam yüzde her zaman yaklaşık yüzdeden "
            "büyüktür (β·ΔX pozitifken ondan büyük, negatifken mutlak değerce ondan küçük bir değişim). β·ΔX "
            "sıfırdan uzaklaştıkça fark büyür: katsayı büyükse ya da X birden fazla birim değişiyorsa tam formül "
            "kullanılır. Birkaç birimlik değişimi tek birimlik yaklaşık yüzdeyi çarparak hesaplamak bileşik "
            "değişimi göz ardı eder"
            + (". Burada yaklaşık yorum %−100'ün altına iniyor: Y en fazla %100 azalabileceği için bu sayı anlamsızdır; "
               "tam formül her zaman −100'den büyük bir değer verir (§9.3)." if beta * dx <= -1 else " (§9.3)."))


# --- Adım 4: karesel ücret modeli ve ortak test ------------------------------------------------------------------

_KOD92 = (
    (INTERCEPT, "sabit terim", (0.2016, 0.101, 1.987, 0.048)),
    ("educ", "educ", (0.0845, 0.007, 11.803, 0.000)),
    ("exper", "exper", (0.0293, 0.005, 5.540, 0.000)),
    ("expersq", "expersq", (-0.0006, 0.000, -5.189, 0.000)),
    ("tenure", "tenure", (0.0371, 0.007, 5.125, 0.000)),
    ("tenursq", "tenursq", (-0.0006, 0.000, -2.468, 0.014)),
)
_KOD92_QUANTITIES = (("coef", "coef", 4), ("se", "std err", 3), ("t", "t", 3), ("p", "P>|t|", 3))


def _quadratic() -> tuple:
    return (
        OLS("m4", DATA, "lwage", QUADRATIC, "Kod 9.1: lwage ~ educ + exper + expersq + tenure + tenursq"),
        ShowModel("m4", "Kod 9.2: WAGE1 karesel modelinin Python çıktısı", columns=("coef", "se", "t", "p"),
                  stats=("nobs", "r2", "adj_r2"), exact=True),
        JointTest("F_kare", "p_kare", "m4", ("expersq", "tenursq"),
                  "Karesel terimlerin ortak F testi, H₀: β_deneyim² = 0, β_kıdem² = 0", decimals=3),
        Scalar("q_kare", E.const(2), "Kısıt sayısı q (pay serbestlik derecesi)", decimals=0),
        ModelValue("sd_m4", "m4", "df_resid", "Payda serbestlik derecesi n − k − 1", decimals=0),
        ModelValue("b_egitim", "m4", "coef", "Eğitim katsayısı", term="educ", decimals=4),
        Scalar("egitim_yaklasik", E.mul(100, E.ref("b_egitim")), "Eğitim: yaklaşık yüzde 100·β̂", decimals=2,
               percent=True),
        Scalar("egitim_tam", E.mul(100, E.sub(E.exp(E.ref("b_egitim")), 1)), "Eğitim: tam yüzde 100·(exp(β̂) − 1)",
               decimals=2, percent=True),
    )


# --- Adım 5: marjinal etki ve dönüm noktası -----------------------------------------------------------------------

TABLE_95 = (0, 5, 10, 20, 25, 30, 40)
"""Tablo 9.5'in deneyim düzeyleri."""
X_5 = NumberChoice("adim5_x", "Deneyim düzeyi x (yıl)", 0, 50, 10, 1, integer=True,
                   help="Bir ek deneyim yılının etkisinin hesaplandığı düzey. Notlarda 10 yıl (§9.5–9.6).")


def _marginal(choices) -> tuple:
    x0 = int(choices["adim5_x"])
    b1, b2 = E.ref("b_d"), E.ref("b_dd")
    slope = E.add(b1, E.mul(E.mul(2, b2), E.ref("x0")))
    one_year = E.add(b1, E.mul(b2, E.add(E.mul(2, E.ref("x0")), 1)))
    curve = E.add(E.add(E.add(E.add(E.add(E.ref("b_0"), E.mul(E.ref("b_egitim"), E.ref("ort_egitim"))),
                                     E.mul(b1, E.var("exper"))), E.mul(b2, E.power(E.var("exper"), 2))),
                        E.mul(E.ref("b_k"), E.ref("ort_kidem"))), E.mul(E.ref("b_kk"), E.power(E.ref("ort_kidem"), 2)))
    return (
        ModelValue("b_0", "m4", "coef", "Sabit terim", term=INTERCEPT, decimals=4),
        ModelValue("b_d", "m4", "coef", "Deneyim katsayısı β̂₂", term="exper", decimals=6),
        ModelValue("b_dd", "m4", "coef", "Deneyim² katsayısı β̂₃", term="expersq", decimals=6),
        ModelValue("b_k", "m4", "coef", "Kıdem katsayısı β̂₄", term="tenure", decimals=6),
        ModelValue("b_kk", "m4", "coef", "Kıdem² katsayısı β̂₅", term="tenursq", decimals=6),
        Scalar("donum_d", E.div(E.neg(b1), E.mul(2, b2)), "Deneyim dönüm noktası −β̂₂ / (2β̂₃)", decimals=2),
        Scalar("donum_d_yuvarlak", E.div(0.0293, E.mul(2, 0.000592)),
               "Yuvarlanmış katsayılarla: 0,0293 / (2 × 0,000592)", decimals=2),
        Scalar("donum_k", E.div(E.neg(E.ref("b_k")), E.mul(2, E.ref("b_kk"))), "Kıdem dönüm noktası −β̂₄ / (2β̂₅)",
               decimals=2),
        Statistic(DATA, "exper", "min", "en_az_d", "Örneklemde en küçük deneyim", decimals=0),
        Statistic(DATA, "exper", "max", "en_cok_d", "Örneklemde en yüksek deneyim", decimals=0),
        Statistic(DATA, "tenure", "min", "en_az_k", "Örneklemde en küçük kıdem", decimals=0),
        Statistic(DATA, "tenure", "max", "en_cok_k", "Örneklemde en yüksek kıdem", decimals=0),
        CopyFrame("donum_otesi", DATA, "Dönüm noktasının ötesindeki gözlemleri saymak için veri setinin kopyası"),
        Derive("donum_otesi", "otesi_d", E.compare("gt", E.var("exper"), E.ref("donum_d")),
               "Deneyimi dönüm noktasından büyük mü (1 = evet)"),
        Derive("donum_otesi", "otesi_k", E.compare("gt", E.var("tenure"), E.ref("donum_k")),
               "Kıdemi dönüm noktasından büyük mü (1 = evet)"),
        Statistic("donum_otesi", "otesi_d", "sum", "sayi_d", "Deneyimi dönüm noktasından büyük çalışan sayısı",
                  decimals=0),
        Statistic("donum_otesi", "otesi_k", "sum", "sayi_k", "Kıdemi dönüm noktasından büyük çalışan sayısı",
                  decimals=0),
        InlineData("tablo95", ("deneyim",), tuple((value,) for value in TABLE_95), "Tablo 9.5'in deneyim düzeyleri"),
        Derive("tablo95", "yaklasik", E.mul(100, E.add(b1, E.mul(E.mul(2, b2), E.var("deneyim")))),
               "Yaklaşık yüzde etki: 100·(β̂₂ + 2β̂₃·deneyim)"),
        Derive("tablo95", "tam", E.mul(100, E.sub(E.exp(E.add(b1, E.mul(b2, E.add(E.mul(2, E.var("deneyim")), 1)))), 1)),
               "Bir yıllık tam yüzde değişim: 100·(exp(β̂₂ + β̂₃(2·deneyim + 1)) − 1)"),
        ShowFrame("tablo95", ("deneyim", "yaklasik", "tam"),
                  "Tablo 9.5: deneyim düzeyine göre bir ek yılın tahmin edilen ücret ilişkisi", decimals=2),
        Scalar("x0", E.const(x0), "Seçilen deneyim düzeyi x", decimals=0),
        Scalar("egim_x0", slope, "x yılda marjinal etki (log ücret): β̂₂ + 2β̂₃x", decimals=5),
        Scalar("yuzde_x0", E.mul(100, slope), "x yılda yaklaşık yüzde etki", decimals=2, percent=True),
        Scalar("tam_x0", E.mul(100, E.sub(E.exp(one_year), 1)), "x'ten x + 1'e tam yüzde değişim", decimals=2,
               percent=True),
        Statistic(DATA, "educ", "mean", "ort_egitim", "Eğitimin örneklem ortalaması", decimals=4),
        Statistic(DATA, "tenure", "mean", "ort_kidem", "Kıdemin örneklem ortalaması", decimals=4),
        Support("egri", "exper", 1, 51, "Deneyim ızgarası: 1, 2, …, 51 yıl (örneklemin aralığı)"),
        Derive("egri", "tahmin", curve,
               "Tahmin edilen ln(ücret): eğitim ve kıdem örneklem ortalamasında (kıdem² = ortalama kıdemin karesi)"),
        Derive("egri", "etki", E.mul(100, E.add(b1, E.mul(E.mul(2, b2), E.var("exper")))),
               "Bir ek yılın yaklaşık yüzde etkisi: 100·(β̂₂ + 2β̂₃·deneyim)"),
        Scalar("sifir", E.const(0), "Sıfır etki çizgisi", decimals=0, shown=False),
        ScatterPlot(DATA, "exper", "lwage", "Deneyim (yıl)", "ln(saatlik ücret)",
                    "Şekil 9.2: deneyim ile log ücret arasındaki karesel ilişki", size=6, opacity=0.35,
                    curves=(("egri", "exper", "tahmin", "Tahmin edilen eğri"),)),
        LineChart("egri", "exper", "etki", "Deneyim (yıl)", "Bir ek yılın yaklaşık yüzde etkisi",
                  "Şekil 9.3: deneyimin tahmin edilen marjinal etkisi", references=(("sifir", "Sıfır etki"),),
                  markers=False, vlines=(("donum_d", "Dönüm noktası (yıl)"), ("x0", "Seçilen deneyim (yıl)"))),
    )


def _marginal_note(state, choices) -> str:
    x0 = int(choices["adim5_x"])
    s = state.scalars
    slope = s["egim_x0"]
    if abs(slope) < 5e-6:
        direction = "Bu düzeyde marjinal etki yaklaşık sıfırdır: dönüm noktasına çok yakınsınız."
    elif slope > 0:
        direction = (f"Deneyim {x0} yıl iken bir ek yıl log ücrette yaklaşık {plain(slope, 5)} artışla, yani yaklaşık "
                     f"%{plain(s['yuzde_x0'], 2)} daha yüksek ücretle ilişkilidir; tam hesap %{plain(s['tam_x0'], 2)}.")
    else:
        direction = (f"Deneyim {x0} yıl iken tahmin edilen marjinal etki negatiftir ({plain(slope, 5)}; yaklaşık "
                     f"%{plain(s['yuzde_x0'], 2)}). Bu, ileri deneyimin ücreti nedensel olarak düşürdüğünü "
                     "kanıtlamaz; eğrinin bu bölgesi gözlemsel ilişkidir.")
    return (f"{direction} Etki deneyim düzeyine bağlıdır: β̂₂ + 2β̂₃x. Deneyim dönüm noktası "
            f"{plain(s['donum_d'], 2)} yıl, kıdem dönüm noktası {plain(s['donum_k'], 2)} yıl. Örneklemde deneyim "
            f"{plain(s['en_az_d'], 0)} ile {plain(s['en_cok_d'], 0)}, kıdem {plain(s['en_az_k'], 0)} ile "
            f"{plain(s['en_cok_k'], 0)} yıl arasındadır: iki dönüm noktası da veri aralığının içindedir. Ancak "
            f"kıdemi dönüm noktasından büyük yalnız {plain(s['sayi_k'], 0)} çalışan vardır; eğrinin o bölgesi çok az "
            "gözleme dayanır (§9.4–9.5).")


# --- Adım 6: merkezleme -------------------------------------------------------------------------------------------

CENTER_6 = NumberChoice("adim6_c", "Merkez noktası c (yıl)", 0, 40, 10, 1, integer=True,
                        help="Deneyim c yılında merkezlenir: deneyim − c. Notlarda 10 yıl (Tablo 9.6).")


def _centered_table(c: int, model: str, frame: str, suffix: str, title: str) -> tuple:
    """Ham model ile c yılda merkezlenmiş modelin R², düzeltilmiş R², HKT ve c yıldaki eğimi (Tablo 9.6)."""

    return (
        CopyFrame(frame, DATA, "Merkezleme için veri setinin kopyası (özgün veri değişmez)"),
        Derive(frame, "exper_c", E.sub(E.var("exper"), c), f"Merkezlenmiş deneyim: exper − {c}"),
        Derive(frame, "exper_c_sq", E.power(E.var("exper_c"), 2), "Merkezlenmiş deneyimin karesi"),
        OLS(model, frame, "lwage", ("educ", "exper_c", "exper_c_sq", "tenure", "tenursq"),
            f"{c} yılda merkezlenmiş model: lwage ~ educ + exper_c + exper_c_sq + tenure + tenursq"),
        Scalar(f"c_merkez{suffix}", E.const(c), "Merkez noktası c", decimals=0),
        Scalar(f"egim_ham{suffix}", E.add(E.ref("b_d"), E.mul(E.mul(2, E.ref("b_dd")), E.ref(f"c_merkez{suffix}"))),
               f"Ham modelde {c} yıldaki deneyim eğimi: β̂₂ + 2β̂₃c", decimals=6),
        ModelValue(f"r2_mer{suffix}", model, "r2", "Merkezlenmiş model: R²", decimals=6),
        ModelValue(f"r2d_mer{suffix}", model, "adj_r2", "Merkezlenmiş model: düzeltilmiş R²", decimals=6),
        ModelValue(f"hkt_mer{suffix}", model, "ssr", "Merkezlenmiş model: HKT", decimals=4),
        ModelValue(f"egim_mer{suffix}", model, "coef", "Merkezlenmiş modelde deneyim katsayısı", term="exper_c",
                   decimals=6),
        ScalarTable((("R²", E.ref("r2_ham")), ("Düzeltilmiş R²", E.ref("r2d_ham")), ("HKT", E.ref("hkt_ham")),
                     (f"{c} yıldaki deneyim eğimi", E.ref(f"egim_ham{suffix}"))), f"ham96{suffix}", decimals=6),
        ScalarTable((("R²", E.ref(f"r2_mer{suffix}")), ("Düzeltilmiş R²", E.ref(f"r2d_mer{suffix}")),
                     ("HKT", E.ref(f"hkt_mer{suffix}")), (f"{c} yıldaki deneyim eğimi", E.ref(f"egim_mer{suffix}"))),
                    f"merkez96{suffix}", decimals=6),
        JoinColumns(f"tablo96{suffix}", (("Ham model", f"ham96{suffix}", "deger"),
                                         ("Merkezlenmiş model", f"merkez96{suffix}", "deger")),
                    decimals=6, heading="Ölçüt", row_decimals=(("HKT", 4),), title=title),
    )


def _centering(choices) -> tuple:
    c = int(choices["adim6_c"])
    notes = c == 10
    noted = () if notes else _centered_table(  # notlardaki 10 yıllık merkezleme seçilenle yan yana
        10, "m_c_n", "merkez_n", "_n", "Notlar — Tablo 9.6: Ham ve 10 yılda merkezlenmiş karesel modelin karşılaştırılması")
    return (
        ModelValue("r2_ham", "m4", "r2", "Ham model: R²", decimals=6),
        ModelValue("r2d_ham", "m4", "adj_r2", "Ham model: düzeltilmiş R²", decimals=6),
        ModelValue("hkt_ham", "m4", "ssr", "Ham model: HKT (artık kareleri toplamı)", decimals=4),
        *noted,
        *_centered_table(c, "m_c", "merkez", "", "Tablo 9.6: Ham ve 10 yılda merkezlenmiş karesel modelin "
                         "karşılaştırılması" if notes else f"Seçiminiz: ham ve {c} yılda merkezlenmiş model"),
        Residuals("merkez", "u_ham", "m4", "Ham modelin artıkları"),
        Residuals("merkez", "u_mer", "m_c", "Merkezlenmiş modelin artıkları"),
        Derive("merkez", "u_fark", E.absolute(E.sub(E.var("u_ham"), E.var("u_mer"))), "İki modelin artık farkı"),
        Statistic("merkez", "u_fark", "max", "azami_fark", "Artıkların (ve tahmin edilen değerlerin) azami farkı",
                  decimals=12),
        PairStatistic("merkez", "exper", "expersq", "corr", "r_ham", "Korelasyon: deneyim ile deneyim²", decimals=3),
        PairStatistic("merkez", "exper_c", "exper_c_sq", "corr", "r_mer",
                      "Korelasyon: merkezlenmiş deneyim ile karesi", decimals=3),
    )


def _centering_note(state, choices) -> str:
    c = int(choices["adim6_c"])
    s = state.scalars
    if c == 0:
        return (f"c = 0 ham modelin kendisidir: merkezlenmiş deneyim deneyimle aynıdır ve katsayı "
                f"{plain(s['egim_mer'], 6)}, deneyim 0 yıl iken (veri aralığının dışında) eğimdir. Anlamlı bir nokta "
                "seçin, ör. notlardaki gibi 10 yıl: katsayı o noktadaki eğim olur. Merkezleme R²'yi, HKT'yi ve tahmin "
                "edilen değerleri değiştirmez; ekonomik ilişkiyi ya da içselliği değiştirmez (§9.6).")
    return (f"Merkezlenmiş modelde deneyim katsayısı {plain(s['egim_mer'], 6)}: ham modelde {c} yıldaki eğim "
            f"β̂₂ + 2β̂₃·{c} = {plain(s['egim_ham'], 6)} ile aynıdır. R², düzeltilmiş R² ve HKT değişmez; iki "
            f"modelin tahmin edilen değerleri arasındaki en büyük fark yalnız bilgisayar yuvarlaması düzeyindedir. "
            f"Merkezleme deneyim ile karesi arasındaki korelasyonu değiştirir ({plain(s['r_ham'], 3)} → "
            f"{plain(s['r_mer'], 3)}); ekonomik ilişkiyi, eksik değişken sorununu ya da içselliği değiştirmez (§9.6).")


# --- Adım 7: model karşılaştırması ---------------------------------------------------------------------------------

MODELS_7 = (
    ("m1", "M₁: Doğrusal", REGRESSORS),
    ("m2", "M₂: Deneyim karesel", ("educ", "exper", "expersq", "tenure")),
    ("m3", "M₃: Kıdem karesel", ("educ", "exper", "tenure", "tenursq")),
    ("m4", "M₄: İki karesel terim", QUADRATIC),
)
EXTRA_7 = Choice("adim7_ek", "Beşinci model: M₄'e eklenen terim",
                 (("yok", "Yok (notlardaki dört model)"), ("educsq", "Eğitimin karesi"),
                  ("numdep", "Bakmakla yükümlü kişi sayısı"), ("tenuresqrt", "Kıdemin karekökü")), "yok",
                 help="Ek bir terim R²'yi azaltmaz; düzeltilmiş R² ise azalabilir (§9.7).")
EXTRA_WORDS = {"educsq": "eğitimin karesi", "numdep": "bakmakla yükümlü kişi sayısı",
               "tenuresqrt": "kıdemin karekökü"}
EXTRA_TERMS = {"educsq": E.power(E.var("educ"), 2), "tenuresqrt": E.power(E.var("tenure"), 0.5)}
"""Türetilen ek terimler (veri setinde olmayanlar)."""


def _comparison(choices) -> tuple:
    extra = choices["adim7_ek"]
    models = list(MODELS_7)
    operations: list = [OLS(name, DATA, "lwage", terms, f"{label}: lwage ~ {' + '.join(terms)}")
                        for name, label, terms in MODELS_7 if name != "m4"]
    if extra != "yok":
        terms = (*QUADRATIC, extra)
        label = "M₅: M₄ + " + EXTRA_WORDS[extra]
        frame = DATA
        if extra in EXTRA_TERMS:
            frame = "model5"
            operations += [CopyFrame("model5", DATA, "Beşinci model için veri setinin kopyası (özgün veri değişmez)"),
                           Derive("model5", extra, EXTRA_TERMS[extra], EXTRA_WORDS[extra].capitalize())]
        operations.append(OLS("m5", frame, "lwage", terms, f"{label}: lwage ~ {' + '.join(terms)}"))
        models.append(("m5", label, terms))
    for name, label, terms in models:
        operations += [
            Scalar(f"k_{name}", E.const(len(terms)), f"{label}: eğim sayısı", decimals=0),
            ModelValue(f"r2_{name}", name, "r2", f"{label}: R²", decimals=4),
            ModelValue(f"r2d_{name}", name, "adj_r2", f"{label}: düzeltilmiş R²", decimals=4),
            ModelValue(f"hkt_{name}", name, "ssr", f"{label}: HKT", decimals=3),
        ]
    tables = []
    for prefix, result in (("k", "k97"), ("r2", "r297"), ("r2d", "r2d97"), ("hkt", "hkt97")):
        tables.append(ScalarTable(tuple((label, E.ref(f"{prefix}_{name}")) for name, label, _ in models), result,
                                  decimals=4, heading="Model"))
    notes = extra == "yok"
    return (
        *operations,
        *tables,
        JoinColumns("tablo97", (("Eğim sayısı", "k97", "deger"), ("R²", "r297", "deger"),
                                ("Düzeltilmiş R²", "r2d97", "deger"), ("HKT", "hkt97", "deger")),
                    decimals=4, heading="Model", column_decimals=(("Eğim sayısı", 0), ("HKT", 3)),
                    title="Tablo 9.7: WAGE1 modellerinin uyum karşılaştırması" if notes
                    else "WAGE1 modellerinin uyum karşılaştırması: notlardaki dört model ve seçtiğiniz beşinci model"),
        BarChart("r2d97", "deger", "Model", "Düzeltilmiş R²",
                 "Şekil 9.4: WAGE1 fonksiyonel biçimlerinde düzeltilmiş R²" if notes
                 else "Fonksiyonel biçimlerde düzeltilmiş R² (beşinci modelle)", decimals=4),
    )


def _comparison_note(state, choices) -> str:
    extra = choices["adim7_ek"]
    s = state.scalars
    text = (f"R² her ek terimle artar: M₁'de {plain(s['r2_m1'], 4)}, M₄'te {plain(s['r2_m4'], 4)}. Düzeltilmiş R² "
            f"ek terimleri cezalandırır; bu dört model içinde en yüksek değer M₄'tedir ({plain(s['r2d_m4'], 4)}). "
            "M₁ ile M₄ iç içedir: iki karesel kısıt ortak F testiyle reddedilir (Adım 4). ")
    if extra != "yok":
        rose = s["r2d_m5"] > s["r2d_m4"]
        text += (f"M₅'te {EXTRA_WORDS[extra]} eklenince R² {plain(s['r2_m4'], 4)} → {plain(s['r2_m5'], 4)} artar; "
                 f"düzeltilmiş R² {plain(s['r2d_m4'], 4)} → {plain(s['r2d_m5'], 4)} "
                 f"{'artar' if rose else 'azalır'}. ")
    return text + ("En yüksek uyum tek başına model seçimi değildir: teori, grafik, katsayıların yorumu, ortak test ve "
                   "basitlik birlikte değerlendirilir; uyum ölçüleri yalnız aynı bağımlı değişken ve aynı örneklemle "
                   "karşılaştırılır (§9.7).")


# --- Adım 8: makale tablosu ---------------------------------------------------------------------------------------

MODEL_8 = Choice("adim8_model", "Sütun (2)'deki karesel model",
                 (("m4", "M₄: deneyim² ve kıdem² (notlar)"), ("m2", "M₂: yalnız deneyim²"), ("m3", "M₃: yalnız kıdem²")),
                 "m4", help="Notlarda Sütun (2) iki karesel terimli modeldir (Tablo 9.8).")
SQUARES = {"m4": ("expersq", "tenursq"), "m2": ("expersq",), "m3": ("tenursq",)}
PAPER_TERMS = ("educ", "exper", "expersq", "tenure", "tenursq")
MARGINAL_8 = {
    "m4": "deneyim ve kıdem katsayıları tek başına yorumlanmaz: deneyimin marjinal etkisi β̂₂ + 2β̂₃x, kıdeminki "
          "β̂₄ + 2β̂₅x ile belirli bir düzeyde hesaplanır",
    "m2": "deneyim katsayısı tek başına yorumlanmaz: marjinal etki β̂₂ + 2β̂₃x ile belirli bir deneyim düzeyinde "
          "hesaplanır",
    "m3": "kıdem katsayısı tek başına yorumlanmaz: marjinal etki β̂₃ + 2β̂₄x ile belirli bir kıdem düzeyinde "
          "hesaplanır",
}
"""Seçilen Sütun (2) modelinde tek başına yorumlanmayan katsayılar (β numaraları modelin kendi sırasıyla)."""


def _paper(choices) -> tuple:
    model = choices["adim8_model"]
    notes = model == "m4"
    single = len(SQUARES[model]) == 1
    test = JointTest("F_98", "p_98", model, SQUARES[model],
                     "Sütun (2): karesel terimin F testi (tek kısıt)" if single
                     else "Sütun (2): karesel terimlerin ortak F testi", decimals=3)
    options = dict(decimals=4, term_decimals=(("expersq", 6), ("tenursq", 6)), adj_r2=True, r2_decimals=3,
                   extra_decimals=3, exact=True)
    if notes:
        return (
            test,
            RegressionTable((("(1) Doğrusal", "m1"), ("(2) Karesel", "m4")), PAPER_TERMS, "tablo98",
                            "Tablo 9.8: log ücretin fonksiyonel biçimi",
                            extra=(("ortak_F", "Karesel terimler ortak F", ("", "F_98")),
                                   ("ortak_p", "Ortak test p-değeri", ("", "p_98"))), **options),
        )
    name = model.replace("m", "M").translate(str.maketrans("23", "₂₃"))
    return (  # notlardaki Sütun (2) (M₄) seçilen modelle yan yana
        test,
        JointTest("F_98n", "p_98n", "m4", SQUARES["m4"], "Notlar — Sütun (2), M₄: karesel terimlerin ortak F testi",
                  decimals=3),
        RegressionTable((("(1) Doğrusal", "m1"), (f"(2) Seçiminiz: {name}", model), ("(2) Notlar: M₄", "m4")),
                        PAPER_TERMS, "tablo98",
                        f"Log ücretin fonksiyonel biçimi: Sütun (2) için seçtiğiniz {name} ve notlardaki M₄",
                        extra=(("ortak_F", "Karesel terim testi: F", ("", "F_98", "F_98n")),
                               ("ortak_p", "Karesel terim testi: p-değeri", ("", "p_98", "p_98n"))), **options),
    )


def _paper_note(state, choices) -> str:
    model = choices["adim8_model"]
    s = state.scalars
    squares = SQUARES[model]
    subject = "iki karesel terimin birlikte" if len(squares) == 2 else "karesel terimin"
    text = (f"Sütun (2)'de {subject} sıfır olduğu hipotezi için F = {plain(s['F_98'], 3)}, {_p(s['p_98'])}. ")
    if len(squares) == 1:
        text += "Tek kısıtta bu F, karesel terimin t istatistiğinin karesidir (§8.8). "
        text += (f"Notlardaki Sütun (2) iki karesel terimli M₄'tür: ortak F = {plain(s['F_98n'], 3)}, "
                 f"{_p(s['p_98n'])}. ")
    example = plain(state.models[model].params[squares[0]], 6)
    return text + (f"Karesel terimlerin katsayıları küçük görünür (ör. {example}); fakat X² ile çarpıldıkları için "
                   f"etkileri X büyüdükçe birikir. Sütun (2)'de {MARGINAL_8[model]}. Bu tablo nedensel deneyim "
                   "getirisini kanıtlamaz (§9.8).")


# --- Tanım ---------------------------------------------------------------------------------------------------------

_TABLE92 = (
    ("Eğitim katsayısı", (0.599, 59.897, 5.990), 3),
    ("t", (11.679, 11.679, 11.679), 3),
    ("R²", (0.306, 0.306, 0.306), 3),
)
_ROWS92 = _row_labels("sent", "on_yil")
_TABLE93 = (("z_educ", "Eğitim", (0.479, 12.555)), ("z_exper", "Deneyim", (0.105, 2.391)),
            ("z_tenure", "Kıdem", (0.300, 7.133)))
_TABLE95 = ((0, 2.93, 2.91), (5, 2.34, 2.31), (10, 1.75, 1.70), (20, 0.56, 0.50), (25, -0.03, -0.09),
            (30, -0.62, -0.68), (40, -1.80, -1.85))
_TABLE97 = (
    ("M₁: Doğrusal", 3, 0.3160, 0.3121, 101.456),
    ("M₂: Deneyim karesel", 4, 0.3595, 0.3545, 95.011),
    ("M₃: Kıdem karesel", 4, 0.3341, 0.3290, 98.774),
    ("M₄: İki karesel terim", 5, 0.3669, 0.3608, 93.911),
)
_TABLE98 = (
    ("(1) Doğrusal", (("educ", 0.0920, 4), ("educ_sh", 0.0073, 4), ("exper", 0.0041, 4), ("exper_sh", 0.0017, 4),
                      ("tenure", 0.0221, 4), ("tenure_sh", 0.0031, 4), ("n", 526, 0), ("r2", 0.316, 3),
                      ("adj_r2", 0.312, 3))),
    ("(2) Karesel", (("educ", 0.0845, 4), ("educ_sh", 0.0072, 4), ("exper", 0.0293, 4), ("exper_sh", 0.0053, 4),
                     ("expersq", -0.000592, 6), ("expersq_sh", 0.000114, 6), ("tenure", 0.0371, 4),
                     ("tenure_sh", 0.0072, 4), ("tenursq", -0.000616, 6), ("tenursq_sh", 0.000249, 6),
                     ("ortak_F", 20.887, 3), ("n", 526, 0), ("r2", 0.367, 3), ("adj_r2", 0.361, 3))),
)

STEPS = (
    interactive_step(
        number=1,
        title="Ölçü birimi değişikliği: katsayı değişir, kanıt değişmez",
        note=NoteRef("9.2", 0, ("Tablo 9.2",)),
        explanation=(
            "Bir değişkeni başka bir birimle yazmak ekonomik bilgiyi değiştirmez; yalnız katsayının sayısal ölçeğini "
            "değiştirir. Ücret dolar yerine sent cinsinden yazılırsa bağımlı değişken 100 ile çarpılır; sabit terim, "
            "eğim katsayıları ve standart hatalar da 100 ile çarpılır. Eğitim on yıllık birimle yazılırsa bir birimlik "
            "artış on yıllık artış olur. Model: $\\text{ücret} = \\beta_0 + \\beta_1\\text{eğitim} + "
            "\\beta_2\\text{deneyim} + \\beta_3\\text{kıdem} + u$. Ücretin ve eğitimin yeni birimini değiştirin."
        ),
        controls=(Y_UNIT_1, X_UNIT_1),
        build=_scaling,
        checks=(
            Check("§9.2: sabit terim", CoefTarget("m_dolar", INTERCEPT, "coef"), -2.873, 3),
            Check("§9.2: eğitim katsayısı", CoefTarget("m_dolar", "educ", "coef"), 0.599, 3),
            Check("§9.2: deneyim katsayısı", CoefTarget("m_dolar", "exper", "coef"), 0.022, 3),
            Check("§9.2: kıdem katsayısı", CoefTarget("m_dolar", "tenure", "coef"), 0.169, 3),
            *(Check(f"Tablo 9.2: {row}, {column}", TableTarget("tablo92", row, column), value, decimals)
              for column, values, decimals in _TABLE92 for row, value in zip(_ROWS92, values)),
            *(Check(f"Tablo 9.2: {row}, p < 0,001", TableTarget("tablo92", row, "p"), 0.0, 3) for row in _ROWS92),
        ),
        note_for=lambda state, choices: _scaling_note(state, choices),
    ),
    interactive_step(
        number=2,
        title="Standartlaştırılmış katsayılar",
        note=NoteRef("9.2", 0, ("Tablo 9.3", "Şekil 9.1")),
        explanation=(
            "Standartlaştırma bir değişkenin ortalamasını çıkarıp standart sapmasına böler: $Z_X = (X - \\bar X)/s_X$. "
            "Hem ln(ücret) hem açıklayıcılar standartlaştırılınca katsayı şöyle okunur: diğer değişkenler sabitken "
            "$X_j$'de bir standart sapmalık artış, ln(ücret)'te ortalama $\\hat\\beta_j^*$ standart sapmalık "
            "değişimle ilişkilidir. Modelin açıklayıcılarını değiştirin."
        ),
        controls=(STD_2,),
        build=_standardized,
        checks=(
            *(Check(f"Tablo 9.3: {label}, standartlaştırılmış katsayı", TableTarget("tablo93", term, "katsayi"),
                    values[0], 3) for term, label, values in _TABLE93),
            *(Check(f"Tablo 9.3: {label}, t", TableTarget("tablo93", term, "t"), values[1], 3)
              for term, label, values in _TABLE93),
            Check("Tablo 9.3: Eğitim, p < 0,001", TableTarget("tablo93", "z_educ", "p"), 0.0, 3),
            Check("Tablo 9.3: Deneyim, p", TableTarget("tablo93", "z_exper", "p"), 0.017, 3),
            Check("Tablo 9.3: Kıdem, p < 0,001", TableTarget("tablo93", "z_tenure", "p"), 0.0, 3),
        ),
        note_for=lambda state, choices: _standardized_note(state, choices),
    ),
    interactive_step(
        number=3,
        title="Log–düzey modelinde yaklaşık ve tam yüzde",
        note=NoteRef("9.3", 0, ("Tablo 9.4",)),
        explanation=(
            "$\\ln(Y) = \\beta_0 + \\beta_1 X + u$ modelinde X bir birim artınca Y yaklaşık yüzde $100\\beta_1$ "
            "değişir. Tam yüzde değişim $100(e^{\\beta_1}-1)$, X'teki değişim $\\Delta X$ birimse "
            "$100(e^{\\beta_1 \\Delta X}-1)$ formülüyle hesaplanır. Notlardaki örnekte eğitim katsayısı 0,0845'tir. "
            "Katsayıyı ve X'teki değişimi değiştirin: iki yorum ne zaman ayrışır?"
        ),
        controls=(BETA_3, DX_3),
        build=_exact,
        checks=(
            *(Check(f"Tablo 9.4: β = {plain(beta, 2)}, ΔX = {dx}, {name}", CellTarget("tablo94", column, row), value, 2)
              for row, ((beta, dx), (approx, exact)) in enumerate(
                  zip(TABLE_94, ((2.00, 2.02), (8.00, 8.33), (20.00, 22.14), (32.00, 37.71))), start=1)
              for column, name, value in (("yaklasik", "yaklaşık yüzde", approx), ("tam", "tam yüzde", exact))),
            _scalar("ornek_yaklasik", 8.45, "§9.3: yaklaşık yorum 100 × 0,0845", 2),
            _scalar("ornek_tam", 8.82, "§9.3: tam yorum 100(exp(0,0845) − 1)", 2),
        ),
        note_for=lambda state, choices: _exact_note(state, choices),
    ),
    LabStep(
        number=4,
        title="Karesel ücret modeli ve karesel terimlerin ortak testi",
        note=NoteRef("9.5", 0, ("Kod 9.1", "Kod 9.2")),
        explanation=(
            "Log saatlik ücret modeline deneyim ve kıdemin kareleri eklenir: $\\ln(\\text{ücret}) = \\beta_0 + "
            "\\beta_1\\text{eğitim} + \\beta_2\\text{deneyim} + \\beta_3\\text{deneyim}^2 + \\beta_4\\text{kıdem} + "
            "\\beta_5\\text{kıdem}^2 + u$. Model parametrelerde doğrusaldır; kareler ayrı açıklayıcı değişken gibi "
            "EKK ile tahmin edilir. Ortak test $H_0: \\beta_3 = 0,\\ \\beta_5 = 0$ hipotezini sınar: doğrusal model "
            "karesel modelin kısıtlı biçimidir."
        ),
        operations=_quadratic(),
        checks=(
            *(Check(f"Kod 9.2: {name}, {label}", CoefTarget("m4", term, quantity), value, decimals)
              for term, name, values in _KOD92
              for (quantity, label, decimals), value in zip(_KOD92_QUANTITIES, values)),
            Check("Kod 9.2: No. Observations", ModelTarget("m4", "nobs"), 526, 0),
            Check("Kod 9.2: R-squared", ModelTarget("m4", "r2"), 0.367, 3),
            Check("Kod 9.2: Adj. R-squared", ModelTarget("m4", "adj_r2"), 0.361, 3),
            _scalar("F_kare", 20.887, "Kod 9.2: ortak test F(2, 520)", 3),
            _scalar("sd_m4", 520, "Kod 9.2: F(2, 520), payda serbestlik derecesi", 0),
            _scalar("q_kare", 2, "Kod 9.2: F(2, 520), kısıt sayısı", 0),
            _scalar("p_kare", 0.0, "Kod 9.2: ortak test p < 0,001", 3),
            _scalar("b_egitim", 0.0845, "§9.5: eğitim katsayısı", 4),
            _scalar("egitim_yaklasik", 8.45, "§9.5: yaklaşık yüzde 8,45", 2),
            _scalar("egitim_tam", 8.82, "§9.5: tam yüzde 8,82", 2),
        ),
        takeaway=(
            "Eğitim katsayısı 0,0845: diğer değişkenler sabitken bir yıllık eğitim farkı yaklaşık %8,45, tam hesapla "
            "%8,82 daha yüksek ücretle ilişkilidir. Deneyim pozitif, deneyim² negatif: deneyimin etkisi deneyim "
            "arttıkça azalır. Karesel terimlerin ortak testinde F(2, 520) = 20,887 ve p < 0,001: iki karesel terimin "
            "birlikte sıfır olduğu hipotezi reddedilir; doğrusal model örneklemde gereksiz yere katı olabilir. Ortak "
            "ret karesel biçimin kesin doğru olduğunu kanıtlamaz; başka eğriler de veriye uyabilir (§9.5, §9.9)."
        ),
    ),
    interactive_step(
        number=5,
        title="Deneyimin marjinal etkisi ve dönüm noktası",
        note=NoteRef("9.5", 0, ("Tablo 9.5", "Şekil 9.2", "Şekil 9.3")),
        explanation=(
            "Karesel modelde deneyimin etkisi sabit değildir: marjinal etki $\\hat\\beta_2 + 2\\hat\\beta_3 x$, "
            "x'ten x + 1'e tam değişim $\\hat\\beta_2 + \\hat\\beta_3(2x + 1)$. Dönüm noktası "
            "$x^* = -\\hat\\beta_2/(2\\hat\\beta_3)$; $\\hat\\beta_3 < 0$ olduğu için tepe noktasıdır. Dönüm "
            "noktası veri aralığında ve yeterli gözlemin bulunduğu bölgede yorumlanır. Deneyim düzeyini değiştirin."
        ),
        controls=(X_5,),
        build=_marginal,
        checks=(
            *(Check(f"Tablo 9.5: deneyim {x}, {name}", CellTarget("tablo95", column, row), value, 2)
              for row, (x, approx, exact) in enumerate(_TABLE95, start=1)
              for column, name, value in (("yaklasik", "yaklaşık yüzde etki", approx),
                                          ("tam", "bir yıllık tam yüzde değişim", exact))),
            _scalar("b_d", 0.0293, "§9.5: deneyim katsayısı", 4),
            _scalar("b_dd", -0.000592, "§9.5: deneyim² katsayısı", 6),
            _scalar("donum_d_yuvarlak", 24.75, "§9.5: dönüm noktası, yuvarlanmış katsayılarla", 2),
            _scalar("donum_d", 24.76, "§9.5: dönüm noktası, yuvarlanmamış katsayılarla", 2),
            _scalar("donum_k", 30.15, "§9.5: kıdem dönüm noktası", 2),
            _scalar("en_az_d", 1, "§9.5: örneklemde en küçük deneyim", 0),
            _scalar("en_cok_d", 51, "§9.5: örneklemde en yüksek deneyim", 0),
            _scalar("en_az_k", 0, "§9.5: örneklemde en küçük kıdem", 0),
            _scalar("en_cok_k", 44, "§9.5: örneklemde en yüksek kıdem", 0),
            _scalar("sayi_d", 147, "§9.5: deneyimi dönüm noktasından büyük çalışan", 0),
            _scalar("sayi_k", 6, "§9.5: kıdemi dönüm noktasından büyük çalışan", 0),
        ),
        note_for=lambda state, choices: _marginal_note(state, choices),
    ),
    interactive_step(
        number=6,
        title="Merkezleme: katsayıyı anlamlı bir noktada okumak",
        note=NoteRef("9.6", 0, ("Tablo 9.6",)),
        explanation=(
            "Karesel modelde deneyim katsayısı deneyim = 0 noktasındaki eğimdir. Deneyim c yılında merkezlenirse "
            "($X_c = X - c$) doğrusal terim doğrudan c yıldaki eğimi verir. Model aynı tahmin edilen değerleri üretir; "
            "yalnız katsayıların referans noktası değişir. Merkez noktasını değiştirin."
        ),
        controls=(CENTER_6,),
        build=_centering,
        checks=(
            Check("Tablo 9.6: ham model, R²", TableTarget("tablo96", "R²", "Ham model"), 0.366875, 6),
            Check("Tablo 9.6: merkezlenmiş model, R²", TableTarget("tablo96", "R²", "Merkezlenmiş model"), 0.366875, 6),
            Check("Tablo 9.6: ham model, düzeltilmiş R²", TableTarget("tablo96", "Düzeltilmiş R²", "Ham model"),
                  0.360787, 6),
            Check("Tablo 9.6: merkezlenmiş model, düzeltilmiş R²",
                  TableTarget("tablo96", "Düzeltilmiş R²", "Merkezlenmiş model"), 0.360787, 6),
            Check("Tablo 9.6: ham model, HKT", TableTarget("tablo96", "HKT", "Ham model"), 93.9113, 4),
            Check("Tablo 9.6: merkezlenmiş model, HKT", TableTarget("tablo96", "HKT", "Merkezlenmiş model"), 93.9113, 4),
            Check("Tablo 9.6: ham model, 10 yıldaki eğim",
                  TableTarget("tablo96", "10 yıldaki deneyim eğimi", "Ham model"), 0.017465, 6),
            Check("Tablo 9.6: merkezlenmiş model, 10 yıldaki eğim",
                  TableTarget("tablo96", "10 yıldaki deneyim eğimi", "Merkezlenmiş model"), 0.017465, 6),
            _scalar("egim_mer", 0.01746, "§9.6: 10 yılda merkezlenmiş deneyim katsayısı", 5),
            _scalar("egim_ham", 0.01746, "§9.6: 0,0293 + 2(−0,000592)(10)", 5),
            _scalar("azami_fark", 0.0, "§9.6: iki model aynı tahmin edilen değerleri üretir", 10),
        ),
        note_for=lambda state, choices: _centering_note(state, choices),
    ),
    interactive_step(
        number=7,
        title="Model seçimi: yalnız R² yetmez",
        note=NoteRef("9.7", 0, ("Tablo 9.7", "Şekil 9.4")),
        explanation=(
            "Dört log ücret modeli karşılaştırılır: M₁ doğrusal, M₂ = M₁ + deneyim², M₃ = M₁ + kıdem², M₄ = M₁ + "
            "deneyim² + kıdem². Aynı bağımlı değişken ve aynı örneklemle R², düzeltilmiş R² ve HKT "
            "karşılaştırılabilir. R² her ek terimle artar; düzeltilmiş R² gereksiz terimleri cezalandırır. "
            "Karşılaştırmaya beşinci bir model ekleyin."
        ),
        controls=(EXTRA_7,),
        build=_comparison,
        checks=tuple(
            check for label, k, r2, adjusted, ssr in _TABLE97 for check in (
                Check(f"Tablo 9.7: {label}, eğim sayısı", TableTarget("tablo97", label, "Eğim sayısı"), k, 0),
                Check(f"Tablo 9.7: {label}, R²", TableTarget("tablo97", label, "R²"), r2, 4),
                Check(f"Tablo 9.7: {label}, düzeltilmiş R²", TableTarget("tablo97", label, "Düzeltilmiş R²"),
                      adjusted, 4),
                Check(f"Tablo 9.7: {label}, HKT", TableTarget("tablo97", label, "HKT"), ssr, 3),
            )
        ),
        note_for=lambda state, choices: _comparison_note(state, choices),
    ),
    interactive_step(
        number=8,
        title="Makale tablosunda fonksiyonel biçim",
        note=NoteRef("9.8", 0, ("Tablo 9.8",)),
        explanation=(
            "Makale tablosunda değişken adları modelin biçimini gösterir: bağımlı değişken log ücret, deneyim² ve "
            "kıdem² karesel terimlerdir. Okuma sırası: bağımlı değişken log mu, karesel terimle birlikte düzey terimi "
            "var mı, işaretler eğrinin yönü hakkında ne söylüyor, karesel terimler birlikte gerekli mi, düzeltilmiş "
            "R² aynı bağımlı değişkenle mi karşılaştırılıyor. Sütun (2)'deki modeli değiştirin."
        ),
        controls=(MODEL_8,),
        build=_paper,
        checks=(
            *(Check(f"Tablo 9.8: {heading}, {row}", TableTarget("tablo98", row, heading), value, decimals)
              for heading, values in _TABLE98 for row, value, decimals in values),
            Check("Tablo 9.8: (2) Karesel, ortak test p < 0,001", TableTarget("tablo98", "ortak_p", "(2) Karesel"),
                  0.0, 3),
        ),
        note_for=lambda state, choices: _paper_note(state, choices),
    ),
)


KONU09_LAB = LabSpec(
    topic_key="konu09",
    title="Uygulama: Ölçekleme, Log ve Karesel Terimler, Model Seçimi",
    note_section="9",
    steps=STEPS,
    labels=(
        *W.labels(DATA),
        (INTERCEPT, "Sabit terim"),
        ("wage_sent", "Saatlik ücret (sent/saat)"),
        ("wage_bin", "Saatlik ücret (bin ABD doları/saat)"),
        ("educ_on", "Eğitim (on yıl)"),
        ("educ_ay", "Eğitim (ay)"),
        ("educsq", "Eğitimin karesi (yıl²)"),
        ("z_lwage", "Standartlaştırılmış ln(ücret)"),
        ("z_educ", "Standartlaştırılmış eğitim"),
        ("z_exper", "Standartlaştırılmış deneyim"),
        ("z_tenure", "Standartlaştırılmış kıdem"),
        ("z_numdep", "Standartlaştırılmış bakmakla yükümlü kişi sayısı"),
        ("exper_c", "Merkezlenmiş deneyim (deneyim − c)"),
        ("exper_c_sq", "Merkezlenmiş deneyimin karesi"),
        ("beta", "Katsayı β"),
        ("dx", "X'teki değişim ΔX"),
        ("yaklasik", "Yaklaşık yüzde"),
        ("tam", "Tam yüzde"),
        ("deneyim", "Deneyim (yıl)"),
    ),
    consistency_notes=(
        "Veri notlardaki gibi WAGE1'dir; notlardaki kod, bölüm betiği, uygulama ve üretilen kod veriyi wooldridge "
        "paketinden okur (Bölüm 9 betiği önceden data/ CSV kopyasını okuyordu; basılı sayıların hiçbiri değişmedi).",
        "Tablo 9.8'de Sütun (1) eğitim katsayısının standart hatası 0,0073 (eski metin 0,0074), Sütun (2) kıdem² "
        "katsayısının standart hatası 0,000249'dur (eski metin 0,000250); notlar ve betik düzeltildi.",
        "§9.5'teki dönüm noktası yuvarlanmış katsayılarla (0,0293 ve 0,000592) 24,75, yuvarlanmamış katsayılarla "
        "24,76 yıldır; notlar ikisini de yazar.",
        "Standart hatalar klasik EKK standart hatalarıdır; heteroskedastisiteye dayanıklı çıkarım Konu 12'de.",
    ),
)
