"""Konu 9 genel uygulaması: ölçekleme, log ve karesel terimler, model seçimi.

Notlardaki sekiz adım (``core.labs.konu09``) aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde
yazılır: ölçü birimi değişikliği, standartlaştırılmış katsayılar, log–düzey modelinde yaklaşık ve tam yüzde, karesel
model ve karesel terimlerin ortak testi, marjinal etki ve dönüm noktası, merkezleme, model karşılaştırması ve makale
tablosu.

Alternatif örnekte veri WAGE2'dir (935 erkek çalışan, 1980; aylık kazanç, log aylık kazanç; eğitim, iş deneyimi ve
kıdem). Standartlaştırmada ve model karşılaştırmasında notlardaki bakmakla yükümlü kişi sayısının yerinde IQ puanı
vardır. Karesel terimler notlardaki gibi deneyim ve kıdemdedir; WAGE2'de iki karesel terim birlikte anlamsızdır ve
deneyimin dönüm noktası veri aralığının çok dışındadır: metin anlamsız bir karesel terimde dönüm noktasının
yorumlanmadığını söyler. "Kendi verini yükle" seçeneğinde modeller notlardaki gibi log sonuçla kurulur (sonucun bütün
değerleri pozitif olmalı); karesel terim ve merkezleme temel açıklayıcı ile (varsa) ilk ek değişken içindir; yüzde
yorumu doğrusal modelin (M₁) temel açıklayıcı katsayısından yapılır (karesel modelde temel açıklayıcının etkisi
düzeyine bağlıdır).

Etkileşim notlardaki gibidir: sonucun ve temel açıklayıcının ölçü birimi (Adım 1), standartlaştırılan modelin
açıklayıcıları (Adım 2), katsayı β ve X'teki değişim ΔX (Adım 3), marjinal etkinin hesaplandığı düzey (Adım 5), merkez
noktası (Adım 6), karşılaştırmaya eklenen beşinci model (Adım 7) ve makale tablosunun karesel modeli (Adım 8).
Standart hatalar klasik (homoskedastik) EKK standart hatalarıdır.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import cache

import numpy as np
import statsmodels.formula.api as smf

from core.labs import expr as E
from core.labs import kendi_veri as K
from core.labs.ornek import Case, TopicVariants, free_name, md, sayi, sayim, stable_checks, with_app_values
from core.labs.ornek_regresyon import (
    ACIKLAYICI,
    EXACT_MULTI_NOTE,
    POSITIVE_RULE,
    ROW_RULE,
    SONUC,
    candidates,
    capital,
    custom_lab,
    digits_for,
    exact_multi,
    labels_of,
    level_text,
    listing,
    log_column,
    log_label,
    log_operations,
    nice,
    noise_decimals,
    number_control,
    options,
    outcome_words,
    p_text,
    phrase,
    plural,
    roles,
    sample_employed,
    short_unit,
    step_words,
    validate,
    validate_positive,
    wage2_case,
)
from core.labs.wording import tr_lower
from core.labs.spec import (
    INTERCEPT,
    OLS,
    BarChart,
    CellTarget,
    Check,
    Choice,
    CoefficientTable,
    CoefTarget,
    CopyFrame,
    Derive,
    InlineData,
    JoinColumns,
    JointTest,
    LabSpec,
    LabStep,
    LineChart,
    ModelTarget,
    ModelValue,
    MultiChoice,
    NoteRef,
    NumberChoice,
    PairStatistic,
    RegressionTable,
    Residuals,
    Scalar,
    ScalarTable,
    ScalarTarget,
    ScatterPlot,
    ShowFrame,
    ShowModel,
    Statistic,
    TableTarget,
    interactive_step,
)

TOPIC = "konu09"
TITLE = "Uygulama: Ölçekleme, Log ve Karesel Terimler, Model Seçimi"
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
GRID = 50
"""Eğrinin ızgara nokta sayısı (sürekli değişkende); tam sayılı ve dar aralıklı değişkende her tam sayı."""
MAX_SQUARE = 1e4
"""Kendi verinde karesi alınan değişkenin en büyük mutlak değeri: karesi 10⁸'i aşan bir sütun tasarımı kötü koşullu
yapar ve iki yazılımın katsayıları son basamaklarda ayrışır; öğrenci değişkeni daha büyük bir birimle yazar."""


def _check(label: str, target, decimals: int) -> Check:
    return Check(label, target, 0.0, decimals)


def _scalar(name: str, label: str, decimals: int) -> Check:
    return _check(label, ScalarTarget(name), decimals)


def _name(case: Case, column: str) -> str:
    return case.name(column) if case.own else phrase(case, column)


def _title(case: Case, column: str) -> str:
    return case.name(column) if case.own else capital(phrase(case, column))


def _m1(case: Case) -> tuple[str, ...]:
    """Doğrusal model (M₁): alternatif örnekte eğitim, deneyim ve kıdem; kendi verinde bütün seçenekler."""

    return tuple(case.extra.get("main") or candidates(case))


def _squares(case: Case) -> tuple[str, ...]:
    """Karesi alınan değişkenler: alternatif örnekte deneyim ve kıdem; kendi verinde temel açıklayıcı ve (varsa) ilk ek
    değişken."""

    if "squares" in case.extra:
        return tuple(case.extra["squares"])
    m1 = _m1(case)
    return m1[:2]


def _taken(case: Case) -> set[str]:
    return set(case.data.columns) | {log_column(case, case.roles[SONUC])}


def _square_names(case: Case) -> dict[str, str]:
    """Karesel terimlerin sütun adları (alternatif örnekte notlardaki gibi ``expersq`` ve ``tenursq``)."""

    if "square_names" in case.extra:
        return dict(case.extra["square_names"])
    taken = _taken(case)
    names = {}
    for variable in _squares(case):
        names[variable] = free_name(f"{variable}_kare", taken)
        taken.add(names[variable])
    return names


def _m4(case: Case) -> tuple[str, ...]:
    """Karesel model (M₄): her karesi alınan değişkenin karesi hemen ardından gelir."""

    squares = _square_names(case)
    terms: list[str] = []
    for variable in _m1(case):
        terms.append(variable)
        if variable in squares:
            terms.append(squares[variable])
    return tuple(terms)


def _ly(case: Case) -> str:
    return log_column(case, case.roles[SONUC])


def _log_data(case: Case):
    """Metinler ve varsayılanlar için log sonuçlu ve kareli veri (uygulamanın hesabı işlemlerle yapılır)."""

    y = case.roles[SONUC]
    data = case.data.copy()
    data[_ly(case)] = np.log(data[y].astype(float))
    for variable, name in _square_names(case).items():
        data[name] = data[variable].astype(float) ** 2
    return data


def _fit(case: Case, outcome: str, regressors, data=None):
    return smf.ols(f"{outcome} ~ {' + '.join(regressors)}", data=_log_data(case) if data is None else data).fit()


def _exact(case: Case, outcome: str, regressors) -> bool:
    if not case.own:
        return False
    return exact_multi(replace(case, data=_log_data(case)), outcome, tuple(regressors))


def _percent_source(case: Case) -> tuple[str, str]:
    """Yüzde yorumunun modeli ve terimi: alternatif örnekte notlardaki gibi karesel modelde (M₄) doğrusal giren eğitim;
    kendi verinde doğrusal modelde (M₁) temel açıklayıcı."""

    if "linear" in case.extra:
        return "m4", str(case.extra["linear"])
    return "m1", case.roles[ACIKLAYICI]


def _model_title(index: int) -> str:
    return "M" + str(index).translate(_SUB)


_LOCATIVE = {1: "de", 2: "de", 3: "te", 4: "te", 5: "te", 6: "da", 7: "de", 8: "de", 9: "da"}
"""Model numarasının bulunma eki (M₁'de, M₃'te): sayının okunuşuna göre (bir, iki, üç, dört, beş, …)."""


def _at(label: str, copula: bool = False) -> str:
    """"M₃: …" etiketinin numarası bulunma ekiyle: "M₃'te" (``copula``: "M₃'tedir")."""

    head = label.split(":")[0]
    digit = int(head[1:].translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")))
    suffix = _LOCATIVE[digit % 10 or 10] if digit % 10 else "da"
    return f"{head}'{suffix}{'dir' if copula and suffix in ('de', 'te') else 'dır' if copula else ''}"


def _unit(case: Case, column: str) -> str:
    unit = short_unit(case, column)
    return f" {unit}" if unit else ""


# --- Adım 1: ölçü birimi değişikliği ----------------------------------------------------------------------

def _units(case: Case) -> tuple[dict, dict, str, str]:
    """Sonucun ve temel açıklayıcının yeni birimleri: anahtar → (sütun, birim adı, ifade, açıklama, çarpan, seçenek)."""

    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    taken = _taken(case)
    if case.own:
        ys = {"yuz": (free_name(f"{y}_yuz", taken), "× 100", E.mul(100, E.var(y)), f"{case.name(y)} × 100", 100.0,
                      "Sonuç × 100"),
              "bin": (free_name(f"{y}_bin", taken), "÷ 1.000", E.div(E.var(y), 1000), f"{case.name(y)} ÷ 1000", 0.001,
                      "Sonuç ÷ 1.000")}
        xs = {"on": (free_name(f"{x}_on", taken), "÷ 10", E.div(E.var(x), 10), f"{case.name(x)} ÷ 10", 0.1,
                     "Temel açıklayıcı ÷ 10"),
              "ay": (free_name(f"{x}_12", taken), "× 12", E.mul(12, E.var(x)), f"{case.name(x)} × 12", 12.0,
                     "Temel açıklayıcı × 12")}
        return ys, xs, "yuz", "on"
    ys = {"yillik": (f"{y}_yillik", "yıllık dolar", E.mul(12, E.var(y)), "Yıllık kazanç: 12 × wage", 12.0,
                     "Yıllık dolar (aylık × 12)"),
          "bin": (f"{y}_bin", "bin dolar", E.div(E.var(y), 1000), "Aylık kazanç bin dolar: wage / 1000", 0.001,
                  "Bin dolar (dolar ÷ 1.000)")}
    xs = {"on": (f"{x}_on", "on yıl", E.div(E.var(x), 10), "Eğitim on yıllık birimle: educ / 10", 0.1,
                 "On yıl (yıl ÷ 10)"),
          "ay": (f"{x}_ay", "ay", E.mul(12, E.var(x)), "Eğitim ay cinsinden: 12 × educ", 12.0, "Ay (yıl × 12)")}
    return ys, xs, "yillik", "on"


def _row_labels(case: Case, ys: dict, xs: dict, y_key: str, x_key: str) -> tuple[str, str, str]:
    if case.own:
        return "Verideki birimler", f"Sonuç {ys[y_key][1]}", f"Temel açıklayıcı {xs[x_key][1]}"
    return ("Kazanç dolar, eğitim yıl", f"Kazanç {ys[y_key][1]}, eğitim yıl", f"Kazanç dolar, eğitim {xs[x_key][1]}")


def _step1(case: Case) -> LabStep:
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    m1 = _m1(case)
    ys, xs, y_default, x_default = _units(case)
    level = smf.ols(f"{y} ~ {' + '.join(m1)}", data=case.data).fit()
    b, intercept, se = float(level.params[x]), float(level.params[INTERCEPT]), float(level.bse[x])
    factors = [factor for *_, factor, _ in ys.values()]
    widest = max(digits_for(value, 3) for value in [b, *(b * f for f in factors), *(b / f for *_, f, _ in xs.values())])
    exact = case.own and exact_multi(case, y, m1)
    column = "Eğitim katsayısı" if not case.own else "Temel açıklayıcının katsayısı"

    def build(choices) -> tuple:
        y_key, x_key = choices["adim1_y"], choices["adim1_x"]
        y_column, y_name, y_expr, y_comment, *_ = ys[y_key]
        x_column, x_name, x_expr, x_comment, *_ = xs[x_key]
        rows = _row_labels(case, ys, xs, y_key, x_key)
        default_rows = _row_labels(case, ys, xs, y_default, x_default)
        swapped = tuple(x_column if term == x else term for term in m1)
        table_rows = [(rows[0], "m_dolar", x, "1")]
        side: list = []
        if y_key != y_default:
            column_n, _, expr, comment, *_ = ys[y_default]
            side += [Derive("olcek", column_n, expr, comment),
                     OLS("m_y_n", "olcek", column_n, m1, f"Varsayılan birim: {column_n} ~ {' + '.join(m1)}")]
            table_rows.append((default_rows[1] + " (varsayılan)", "m_y_n", x, "n2"))
        table_rows.append((rows[1], "m_y", x, "2"))
        if x_key != x_default:
            column_n, _, expr, comment, *_ = xs[x_default]
            noted = tuple(column_n if term == x else term for term in m1)
            side += [Derive("olcek", column_n, expr, comment),
                     OLS("m_x_n", "olcek", y, noted, f"Varsayılan birim: {y} ~ {' + '.join(noted)}")]
            table_rows.append((default_rows[2] + " (varsayılan)", "m_x_n", column_n, "n3"))
        table_rows.append((rows[2], "m_x", x_column, "3"))
        values = []
        for label, model, term, suffix in table_rows:
            values += [
                ModelValue(f"b_{suffix}", model, "coef", f"{label}: temel açıklayıcının katsayısı", term=term,
                           decimals=widest),
                ModelValue(f"t_{suffix}", model, "t", f"{label}: katsayının t istatistiği", term=term, decimals=3),
                ModelValue(f"p_{suffix}", model, "p", f"{label}: katsayının p-değeri", term=term, decimals=3),
                ModelValue(f"r2_{suffix}", model, "r2", f"{label}: R²", decimals=3),
            ]
        same = (y_key, x_key) == (y_default, x_default)
        return (
            *case.load,
            OLS("m_dolar", case.frame, y, m1, f"{rows[0]}: {y} ~ {' + '.join(m1)}"),
            CopyFrame("olcek", case.frame, "Birim dönüşümleri için verinin kopyası (özgün veri değişmez)"),
            Derive("olcek", y_column, y_expr, y_comment),
            Derive("olcek", x_column, x_expr, x_comment),
            OLS("m_y", "olcek", y_column, m1, f"{rows[1]}: {y_column} ~ {' + '.join(m1)}"),
            OLS("m_x", "olcek", y, swapped, f"{rows[2]}: {y} ~ {' + '.join(swapped)}"),
            *side,
            ModelValue("a_1", "m_dolar", "coef", f"{rows[0]}: sabit terim", term=INTERCEPT,
                       decimals=digits_for(intercept, 3)),
            ModelValue("a_2", "m_y", "coef", f"{rows[1]}: sabit terim", term=INTERCEPT, decimals=widest),
            ModelValue("sh_1", "m_dolar", "se", f"{rows[0]}: katsayının standart hatası", term=x,
                       decimals=digits_for(se, 4)),
            ModelValue("sh_2", "m_y", "se", f"{rows[1]}: katsayının standart hatası", term=x,
                       decimals=max(digits_for(se * factor, 4) for factor in factors)),
            *values,
            *(ScalarTable(tuple((label, E.ref(f"{prefix}_{suffix}")) for label, _, _, suffix in table_rows), result,
                          decimals=digits)
              for prefix, result, digits in (("b", "katsayi92", widest), ("t", "t92", 3), ("p", "p92", 4),
                                             ("r2", "r292", 3))),
            JoinColumns("tablo92", ((column, "katsayi92", "deger"), ("t", "t92", "deger"), ("p", "p92", "deger"),
                                    ("R²", "r292", "deger")), decimals=3, heading="Model", p_columns=("p",),
                        column_decimals=((column, widest),),
                        title="Ölçü birimi değişikliklerinin etkisi (Tablo 9.2'deki gibi)" if same
                        else "Ölçü birimi değişikliklerinin etkisi: varsayılan ve seçtiğiniz birimler"),
        )

    def note(state, choices) -> str:
        y_key, x_key = choices["adim1_y"], choices["adim1_x"]
        s = state.scalars
        y_factor, x_factor = ys[y_key][4], xs[x_key][4]
        factor = (f"{level_text(y_factor)} ile çarpılır" if y_factor > 1 else f"{level_text(round(1 / y_factor))}'e bölünür")
        x_words = (f"{level_text(round(1 / x_factor))}'a bölünür" if x_factor < 1 else f"{level_text(x_factor)} ile çarpılır")
        if case.own:
            effect = (f"bir birimlik artış eski birimle {level_text(round(1 / x_factor))} birimlik artıştır" if x_factor < 1
                      else f"bir birimlik artış eski birimle 1/{level_text(x_factor)} birimlik artıştır")
        else:
            effect = ("bir birimlik artış artık on yıllık eğitim artışıdır" if x_key == "on"
                      else "bir birimlik artış artık bir aylık eğitim artışıdır")
        a2 = digits_for(s["a_2"], 3)
        text = (f"Sonuç {ys[y_key][1]} cinsinden yazılınca bağımlı değişken {factor}; sabit terim "
                f"({sayi(s['a_1'], digits_for(s['a_1'], 3))} → {sayi(s['a_2'], a2)}), {phrase(case, x)} katsayısı "
                f"({sayi(s['b_1'], digits_for(s['b_1'], 3))} → {sayi(s['b_2'], digits_for(s['b_2'], 3))}) ve standart "
                f"hatası ({sayi(s['sh_1'], digits_for(s['sh_1'], 4))} → {sayi(s['sh_2'], digits_for(s['sh_2'], 4))}) aynı "
                f"oranda değişir. {capital(phrase(case, x))} {xs[x_key][1]} biriminde yazılınca açıklayıcı değişken "
                f"{x_words}; {effect} ve katsayı {sayi(s['b_3'], digits_for(s['b_3'], 3))} olur. ")
        if exact:
            return text + ("Uyum tam olduğu için t ve p yuvarlama hatasıdır; birim dönüşümü yine katsayıyla standart "
                           "hatayı aynı oranda değiştirir (§9.2)." + EXACT_MULTI_NOTE)
        return text + (f"Tablodaki bütün modellerde t = {sayi(s['t_1'], 3)} ve R² = {sayi(s['r2_1'], 3)}: katsayı ile "
                       "standart hata aynı ölçekte değiştiği için oranları değişmez. Birim dönüşümü ekonomik ilişkiyi ve "
                       "istatistiksel kanıtı değiştirmez; yalnız katsayının hangi birimle okunacağını değiştirir (§9.2).")

    rows = _row_labels(case, ys, xs, y_default, x_default)
    checks = [
        _check("Sabit terim", CoefTarget("m_dolar", INTERCEPT, "coef"), digits_for(intercept, 3)),
        *(_check(f"{case.name(term)} katsayısı", CoefTarget("m_dolar", term, "coef"),
                 digits_for(float(level.params[term]), 3)) for term in m1),
        *(_check(f"Tablo: {row}, katsayı", TableTarget("tablo92", row, column), widest) for row in rows),
        *(_check(f"Tablo: {row}, R²", TableTarget("tablo92", row, "R²"), 3) for row in rows),
    ]
    if not exact:
        checks += [*(_check(f"Tablo: {row}, t", TableTarget("tablo92", row, "t"), 3) for row in rows),
                   *(_check(f"Tablo: {row}, p", TableTarget("tablo92", row, "p"), 3) for row in rows)]
    lead = ("Bir değişkeni başka bir birimle yazmak ekonomik bilgiyi değiştirmez; yalnız katsayının sayısal ölçeğini "
            "değiştirir. Sonuç bir sayıyla çarpılırsa sabit terim, eğim katsayıları ve standart hatalar da aynı sayıyla "
            "çarpılır; açıklayıcı bir sayıya bölünürse onun katsayısı aynı sayıyla çarpılır. ")
    model = f"Model `{y} ~ {' + '.join(m1)}`."
    return interactive_step(
        number=1,
        title="Ölçü birimi değişikliği: katsayı değişir, kanıt değişmez",
        note=NoteRef("9.2", 0, ("Tablo 9.2",)),
        explanation=f"{lead}{model} Sonucun ve temel açıklayıcının yeni birimini değiştirin.",
        controls=(
            Choice("adim1_y", "Sonucun yeni ölçü birimi", tuple((key, value[5]) for key, value in ys.items()), y_default,
                   help=f"Varsayılan: {ys[y_default][5]}."),
            Choice("adim1_x", "Temel açıklayıcının yeni ölçü birimi",
                   tuple((key, value[5]) for key, value in xs.items()), x_default,
                   help=f"Varsayılan: {xs[x_default][5]}."),
        ),
        build=build,
        checks=tuple(checks),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 2: standartlaştırılmış katsayılar ------------------------------------------------------------------

def _z_names(case: Case) -> dict[str, str]:
    taken = _taken(case)
    names = {}
    for name in (_ly(case), *tuple(case.extra.get("std_options") or candidates(case))):
        names[name] = free_name(f"z_{name}", taken)
        taken.add(names[name])
    return names


def _step2(case: Case) -> LabStep:
    y = case.roles[SONUC]
    ly = _ly(case)
    m1 = _m1(case)
    choices_all = tuple(case.extra.get("std_options") or candidates(case))
    z = _z_names(case)
    exact = _exact(case, ly, m1)
    log_title = log_label(case, y) if case.own else phrase(case, ly)  # açıklama ve etiketler (arayüz kaçırır)
    log_text = f"ln({phrase(case, y)})" if case.own else phrase(case, ly)  # Markdown metni

    def build(choices) -> tuple:
        chosen = tuple(choices["adim2_x"])
        same = chosen == m1
        operations: list = [*log_operations(case, case.frame, (y,)),
                            CopyFrame("std", case.frame, "Standartlaştırma için verinin kopyası (özgün veri değişmez)")]
        for name in (ly, *dict.fromkeys((*chosen, *(() if same else m1)))):
            label = log_title if name == ly else _name(case, name)
            title = log_title if name == ly else _title(case, name)
            operations += [
                Statistic("std", name, "mean", f"ort_{name}", f"{title}: örneklem ortalaması"),
                Statistic("std", name, "std", f"ss_{name}", f"{title}: örneklem standart sapması (payda n − 1)"),
                Derive("std", z[name], E.div(E.sub(E.var(name), E.ref(f"ort_{name}")), E.ref(f"ss_{name}")),
                       f"Standartlaştırılmış {label}: (x − ortalama) / standart sapma"),
            ]
        terms = tuple(z[name] for name in chosen)
        if not same:
            noted = tuple(z[name] for name in m1)
            operations += [
                OLS("m_std_n", "std", z[ly], noted, f"Varsayılan model: {z[ly]} ~ " + " + ".join(noted)),
                CoefficientTable("m_std_n", noted, "tablo93_n", "Varsayılan model: standartlaştırılmış katsayılar",
                                 decimals=3, t_decimals=3),
            ]
        return (
            *operations,
            OLS("m_std", "std", z[ly], terms, f"Standartlaştırılmış model: {z[ly]} ~ " + " + ".join(terms)),
            CoefficientTable("m_std", terms, "tablo93", "Standartlaştırılmış katsayılar (Tablo 9.3'teki gibi)" if same
                             else "Standartlaştırılmış katsayılar (seçtiğiniz model)", decimals=3, t_decimals=3),
            *(ModelValue(f"bz_{name}", "m_std", "coef", f"{_title(case, name)}: standartlaştırılmış katsayı",
                         term=z[name], decimals=3) for name in chosen),
            ScalarTable(tuple((_title(case, name), E.ref(f"bz_{name}")) for name in chosen), "std_katsayi", decimals=3,
                        heading="Değişken", value="Standartlaştırılmış katsayı"),
            BarChart("std_katsayi", "deger", "Açıklayıcı değişken", "Standartlaştırılmış katsayı",
                     f"{case.extra.get('data_name', 'Verileriniz')}, {log_title} modeli: standartlaştırılmış katsayılar"
                     if same else "Standartlaştırılmış katsayılar (seçtiğiniz model)", decimals=3),
        )

    def note(state, choices) -> str:
        chosen = tuple(choices["adim2_x"])
        s = state.scalars
        if len(chosen) == 1:
            (name,) = chosen
            value = s[f"bz_{name}"]
            return (f"Tek açıklayıcıyla standartlaştırılmış katsayı ({sayi(value, 3)}), {phrase(case, name)} ile "
                    f"{log_text} arasındaki örneklem korelasyonuna eşittir: {phrase(case, name)} bir standart sapma "
                    f"yüksekken {log_text} ortalama {sayi(abs(value), 3)} standart sapma "
                    f"{'yüksektir' if value > 0 else 'düşüktür'}. Standartlaştırma t ve p-değerlerini değiştirmez; yalnız "
                    "yorum ölçeğini değiştirir. Katsayı örneklem standart sapmalarına bağlıdır ve nedensel önem ölçüsü "
                    "değildir (§9.2).")
        largest = max(chosen, key=lambda name: abs(s[f"bz_{name}"]))
        value = s[f"bz_{largest}"]
        text = (f"Bu modelde mutlak değerce en büyük standartlaştırılmış katsayı {phrase(case, largest)} değişkenindedir "
                f"({sayi(value, 3)}): {phrase(case, largest)} bir standart sapma yüksekken, diğer değişkenler sabitken "
                f"{log_text} ortalama {sayi(abs(value), 3)} standart sapma {'yüksektir' if value > 0 else 'düşüktür'}. ")
        return text + ("Standartlaştırma t ve p-değerlerini değiştirmez; yalnız yorum ölçeğini değiştirir. Katsayılar "
                       "örneklem standart sapmalarına bağlıdır: bu sıralama nedensel önem sıralaması değildir ve başka "
                       "bir örneklemde değişebilir (§9.2)." + (EXACT_MULTI_NOTE if exact else ""))

    checks = [_check(f"Tablo: {_title(case, name)}, standartlaştırılmış katsayı", TableTarget("tablo93", z[name],
                                                                                              "katsayi"), 3)
              for name in m1]
    if not exact:
        checks += [_check(f"Tablo: {_title(case, name)}, {label}", TableTarget("tablo93", z[name], column), 3)
                   for name in m1 for column, label in (("t", "t"), ("p", "p"))]
    return interactive_step(
        number=2,
        title="Standartlaştırılmış katsayılar",
        note=NoteRef("9.2", 0, ("Tablo 9.3", "Şekil 9.1")),
        explanation=(
            "Standartlaştırma bir değişkenin ortalamasını çıkarıp standart sapmasına böler: $Z_X = (X - \\bar X)/s_X$. "
            f"Hem {log_text} hem açıklayıcılar standartlaştırılınca katsayı şöyle okunur: diğer değişkenler sabitken "
            f"$X_j$'de bir standart sapmalık artış, {log_text} değişkeninde ortalama $\\hat\\beta_j^*$ standart sapmalık "
            "değişimle ilişkilidir. Modelin açıklayıcılarını değiştirin."
        ),
        controls=(MultiChoice("adim2_x", "Standartlaştırılan modelin açıklayıcıları", options(case, choices_all), m1,
                              help="Varsayılan: doğrusal modelin (M₁) açıklayıcıları; bağımlı değişken log sonuçtur."),),
        build=build,
        checks=tuple(checks),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 3: yaklaşık ve tam yüzde değişim ---------------------------------------------------------------------

def _percent_value(case: Case) -> float:
    model, term = _percent_source(case)
    regressors = _m4(case) if model == "m4" else _m1(case)
    return float(_fit(case, _ly(case), regressors).params[term])


def _step3(case: Case) -> LabStep:
    b = _percent_value(case)
    _, term = _percent_source(case)
    shown = nice(b, 3)
    beta = round(b, 2)
    illustrative = beta == 0
    if illustrative:
        beta = 0.08  # katsayı iki basamakta sıfır: notlardaki gibi bir örnek değer
    reach = max(0.5, math.ceil(abs(2 * beta) * 10) / 10)
    rows = tuple(dict.fromkeys(((0.02, 1), (beta, 1), (0.2, 1), (beta, 4))))
    d = digits_for(100 * shown, 2)
    beta_control = NumberChoice("adim3_beta", "Katsayı β", -reach, reach, beta, 0.01,
                                help=(f"Log–düzey modelinde X'in katsayısı. Varsayılan: {_name(case, term)} katsayısı "
                                      f"(iki basamakla {sayi(beta, 2)})." if not illustrative else
                                      "Log–düzey modelinde X'in katsayısı. Varsayılan bir örnek değerdir (β = 0,08)."))
    dx_control = NumberChoice("adim3_dx", "X'teki değişim ΔX", 1, 10, 4, 1, integer=True,
                              help="Varsayılan: dört birimlik değişim (tablonun son satırı).")

    def build(choices) -> tuple:
        chosen, dx = float(choices["adim3_beta"]), int(choices["adim3_dx"])
        product = E.mul(E.ref("beta_sec"), E.ref("dx_sec"))
        return (
            InlineData("tablo94", ("beta", "dx"), rows, "Katsayı β ve X'teki değişim ΔX (Tablo 9.4'teki gibi)"),
            Derive("tablo94", "yaklasik", E.mul(100, E.mul(E.var("beta"), E.var("dx"))), "Yaklaşık yüzde: 100·β·ΔX"),
            Derive("tablo94", "tam", E.mul(100, E.sub(E.exp(E.mul(E.var("beta"), E.var("dx"))), 1)),
                   "Tam yüzde: 100·(exp(β·ΔX) − 1)"),
            ShowFrame("tablo94", ("beta", "dx", "yaklasik", "tam"),
                      "Log–düzey modelinde yaklaşık ve tam yüzde yorumları (Tablo 9.4'teki gibi)", decimals=2),
            Scalar("ornek_yaklasik", E.mul(100, shown),
                   f"{capital(_name(case, term))} katsayısı {sayi(shown, digits_for(shown, 4))}: yaklaşık yüzde 100·β̂",
                   decimals=d, percent=True),
            Scalar("ornek_tam", E.mul(100, E.sub(E.exp(shown), 1)),
                   f"{capital(_name(case, term))} katsayısı: tam yüzde 100·(exp(β̂) − 1)", decimals=d, percent=True),
            Scalar("beta_sec", E.const(chosen), "Seçilen katsayı β", decimals=2),
            Scalar("dx_sec", E.const(dx), "Seçilen değişim ΔX", decimals=0),
            Scalar("yaklasik_sec", E.mul(100, product), "Seçilen β ve ΔX: yaklaşık yüzde 100·β·ΔX", decimals=2,
                   percent=True),
            Scalar("tam_sec", E.mul(100, E.sub(E.exp(product), 1)), "Seçilen β ve ΔX: tam yüzde 100·(exp(β·ΔX) − 1)",
                   decimals=2, percent=True),
        )

    def note(state, choices) -> str:
        chosen, dx = float(choices["adim3_beta"]), int(choices["adim3_dx"])
        s = state.scalars
        gap = s["tam_sec"] - s["yaklasik_sec"]
        if chosen == 0:
            return "β = 0 iken iki yorum da sıfırdır: X'in değişimi ln(Y)'yi değiştirmez (§9.3)."
        return (f"β = {sayi(chosen, 2)} ve ΔX = {dx} için yaklaşık yorum %{sayi(s['yaklasik_sec'], 2)}, tam yorum "
                f"%{sayi(s['tam_sec'], 2)}; fark {sayi(gap, 2)} yüzde puan. Tam yüzde her zaman yaklaşık yüzdeden "
                "büyüktür (β·ΔX pozitifken ondan büyük, negatifken mutlak değerce ondan küçük bir değişim). β·ΔX "
                "sıfırdan uzaklaştıkça fark büyür: katsayı büyükse ya da X birden fazla birim değişiyorsa tam formül "
                "kullanılır. Birkaç birimlik değişimi tek birimlik yaklaşık yüzdeyi çarparak hesaplamak bileşik "
                "değişimi göz ardı eder"
                + (". Burada yaklaşık yorum %−100'ün altına iniyor: Y en fazla %100 azalabileceği için bu sayı "
                   "anlamsızdır; tam formül her zaman −100'den büyük bir değer verir (§9.3)." if chosen * dx <= -1
                   else " (§9.3)."))

    model, _ = _percent_source(case)
    where = ("karesel modelde (Adım 4)" if model == "m4" else "doğrusal log modelde (M₁)")
    example = (f"Bu veride {where} {phrase(case, term)} katsayısı {sayi(shown, digits_for(shown, 4))}"
               + ("; iki basamakta sıfır olduğu için kaydırıcının varsayılanı bir örnek değerdir (0,08)."
                  if illustrative else "."))
    return interactive_step(
        number=3,
        title="Log–düzey modelinde yaklaşık ve tam yüzde",
        note=NoteRef("9.3", 0, ("Tablo 9.4",)),
        explanation=(
            "$\\ln(Y) = \\beta_0 + \\beta_1 X + u$ modelinde X bir birim artınca Y yaklaşık yüzde $100\\beta_1$ "
            "değişir. Tam yüzde değişim $100(e^{\\beta_1}-1)$, X'teki değişim $\\Delta X$ birimse "
            f"$100(e^{{\\beta_1 \\Delta X}}-1)$ formülüyle hesaplanır. {example} Katsayıyı ve X'teki değişimi "
            "değiştirin: iki yorum ne zaman ayrışır?"
        ),
        controls=(beta_control, dx_control),
        build=build,
        checks=(
            *(_check(f"Tablo: β = {sayi(value, 2)}, ΔX = {change}, {name}", CellTarget("tablo94", column, row), 2)
              for row, (value, change) in enumerate(rows, start=1)
              for column, name in (("yaklasik", "yaklaşık yüzde"), ("tam", "tam yüzde"))),
            _scalar("ornek_yaklasik", "Örnek: yaklaşık yüzde", d),
            _scalar("ornek_tam", "Örnek: tam yüzde", d),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 4: karesel model ve ortak test ------------------------------------------------------------------

def _square_words(case: Case, variable: str, text: bool = True) -> str:
    """Karesel terimin adı: metinde ``phrase`` (kendi verinde tırnaklı ve kaçırılmış), açıklamada dosyadaki ad."""

    return f"{phrase(case, variable) if text else _name(case, variable)}²"


def _step4(case: Case) -> LabStep:
    ly = _ly(case)
    m1, m4 = _m1(case), _m4(case)
    squares = _square_names(case)
    exact = _exact(case, ly, m4)
    model_name, term = _percent_source(case)
    q = len(squares)
    operations: list = [Derive(case.frame, name, E.power(E.var(variable), 2), f"{case.name(variable)}²")
                        for variable, name in squares.items()]
    if model_name == "m1":
        operations.append(OLS("m1", case.frame, ly, m1, f"M₁, doğrusal model: {ly} ~ {' + '.join(m1)}"))
    hypothesis = ", ".join(f"β({_square_words(case, variable, text=False)}) = 0" for variable in squares)
    b = _percent_value(case)
    d = digits_for(b, 4)
    operations += [
        OLS("m4", case.frame, ly, m4, f"Karesel model: {ly} ~ {' + '.join(m4)}"),
        ShowModel("m4", "Karesel modelin Python çıktısı (Kod 9.2'deki gibi)", columns=("coef", "se", "t", "p"),
                  stats=("nobs", "r2", "adj_r2")),
        JointTest("F_kare", "p_kare", "m4", tuple(squares.values()),
                  f"Karesel {'terimlerin ortak' if q > 1 else 'terimin'} F testi, H₀: {hypothesis}", decimals=3),
        Scalar("q_kare", E.const(q), "Kısıt sayısı q (pay serbestlik derecesi)", decimals=0),
        ModelValue("sd_m4", "m4", "df_resid", "Payda serbestlik derecesi n − k − 1", decimals=0),
        ModelValue("b_egitim", model_name, "coef",
                   f"{capital(_name(case, term))} katsayısı" + (" (doğrusal model M₁)" if model_name == "m1" else ""),
                   term=term, decimals=d),
        Scalar("egitim_yaklasik", E.mul(100, E.ref("b_egitim")), "Yaklaşık yüzde 100·β̂", decimals=digits_for(100 * b, 2),
               percent=True),
        Scalar("egitim_tam", E.mul(100, E.sub(E.exp(E.ref("b_egitim")), 1)), "Tam yüzde 100·(exp(β̂) − 1)",
               decimals=digits_for(100 * b, 2), percent=True),
    ]
    fit = _fit(case, ly, m4)

    def note(state) -> str:
        s = state.scalars
        result = state.models["m4"]
        held = "diğer değişkenler sabitken " if len(m1) > 1 else ""
        text = (f"{capital(phrase(case, term))} katsayısı {sayi(s['b_egitim'], d)}"
                + (" (doğrusal modelde)" if model_name == "m1" else "")
                + f": {held}{step_words(case, term)} {plural(case)} tahmin edilen {outcome_words(case)} yaklaşık "
                f"%{sayi(abs(s['egitim_yaklasik']), digits_for(s['egitim_yaklasik'], 2))}, tam hesapla "
                f"%{sayi(abs(s['egitim_tam']), digits_for(s['egitim_tam'], 2))} "
                f"{'daha yüksektir' if s['b_egitim'] > 0 else 'daha düşüktür'}. ")
        data = _log_data(case)
        for variable, name in squares.items():
            linear, square = float(result.params[variable]), float(result.params[name])
            if _noise(square, data[name], data[ly]):  # tam uyumda doğrusal gerçek: kare katsayısı gürültüdür
                text += (f"{capital(_square_words(case, variable))} katsayısı hesap hassasiyetinde sıfırdır: "
                         f"{phrase(case, variable)} değişkeninin marjinal etkisi düzeye bağlı görünmez. ")
                continue
            text += (f"{capital(phrase(case, variable))} {'pozitif' if linear > 0 else 'negatif'}, "
                     f"{_square_words(case, variable)} {'negatif' if square < 0 else 'pozitif'} katsayılı: "
                     f"{phrase(case, variable)} değişkeninin marjinal etkisi düzey arttıkça "
                     f"{'azalır' if square < 0 else 'artar'}. ")
        if exact:
            return text + ("Uyum tam olduğu için ortak testin F'si yuvarlama hatasıdır ve karar yazılmaz (§9.5)."
                           + EXACT_MULTI_NOTE)
        rejected = s["p_kare"] < 0.05
        subject = "iki karesel terimin birlikte" if q > 1 else "karesel terimin"
        text += (f"Karesel {'terimlerin ortak testinde' if q > 1 else 'terimin testinde'} F({q}, {sayim(s['sd_m4'])}) = "
                 f"{sayi(s['F_kare'], 3)} ve {p_text(s['p_kare'])}: {subject} sıfır olduğu hipotezi yüzde 5 düzeyinde "
                 f"{'reddedilir' if rejected else 'reddedilemez'}; ")
        if rejected:
            return text + ("doğrusal model örneklemde gereksiz yere katı olabilir. Ortak ret karesel biçimin kesin doğru "
                           "olduğunu kanıtlamaz; başka eğriler de veriye uyabilir (§9.5, §9.9).")
        return text + ("veri doğrusal modelden ayrışacak kadar eğrilik göstermez. Reddedememek doğrusal biçimin doğru "
                       "olduğunu kanıtlamaz; karesel terimlerin katsayıları sıfıra yakın ve belirsizdir (§9.5, §9.9).")

    labels = {INTERCEPT: "sabit terim"}
    checks = [
        *(_check(f"Kod 9.2 karşılığı: {labels.get(name, case.name(name) if name in case.data.columns else name)}, "
                 f"{label}", CoefTarget("m4", name, quantity), decimals)
          for name in (INTERCEPT, *m4)
          for quantity, label, decimals in (("coef", "coef", max(4, digits_for(float(fit.params[name]), 4))),
                                            ("se", "std err", 3), ("t", "t", 3), ("p", "P>|t|", 3))),
        _check("No. Observations", ModelTarget("m4", "nobs"), 0),
        _check("R-squared", ModelTarget("m4", "r2"), 3),
        _check("Adj. R-squared", ModelTarget("m4", "adj_r2"), 3),
        _scalar("F_kare", "Karesel terimlerin F testi", 3),
        _scalar("sd_m4", "Payda serbestlik derecesi", 0),
        _scalar("q_kare", "Kısıt sayısı", 0),
        _scalar("p_kare", "Ortak test p-değeri", 3),
        _scalar("b_egitim", "Yüzde yorumundaki katsayı", d),
        _scalar("egitim_yaklasik", "Yaklaşık yüzde", digits_for(100 * b, 2)),
        _scalar("egitim_tam", "Tam yüzde", digits_for(100 * b, 2)),
    ]
    if exact:
        checks = [check for check in stable_checks(tuple(checks), True)
                  if not (isinstance(check.target, ScalarTarget) and check.target.name in ("F_kare", "p_kare"))]
    names = listing([phrase(case, variable) for variable in squares])
    return LabStep(
        number=4,
        title="Karesel model ve karesel terimlerin ortak testi",
        note=NoteRef("9.5", 0, ("Kod 9.1", "Kod 9.2")),
        explanation=(
            f"Log sonuç modeline {names} değişken{'lerinin' if q > 1 else 'inin'} karesi eklenir: "
            f"`{ly} ~ {' + '.join(m4)}`. Model parametrelerde doğrusaldır; kareler ayrı açıklayıcı değişken gibi EKK ile "
            f"tahmin edilir. {'Ortak test' if q > 1 else 'Test'} $H_0$: karesel "
            f"{'terimlerin katsayıları sıfır' if q > 1 else 'terimin katsayısı sıfır'} hipotezini sınar: doğrusal model "
            "karesel modelin kısıtlı biçimidir."
        ),
        operations=tuple(operations),
        checks=tuple(checks),
        note_for=lambda state, choices: note(state),
    )


# --- Adım 5: marjinal etki ve dönüm noktası -----------------------------------------------------------------

def _grid(case: Case, variable: str) -> tuple[float, ...]:
    """Eğrinin ızgarası: tam sayılı ve dar aralıklı değişkende her tam sayı, diğerlerinde ``GRID`` eşit aralıklı nokta."""

    values = case.data[variable].astype(float)
    low, high = float(values.min()), float(values.max())
    if bool(np.all(values == np.round(values))) and high - low <= 100:
        return tuple(float(value) for value in range(int(low), int(high) + 1))
    step = nice((high - low) / GRID, 2) or (high - low) / GRID
    points = np.unique(np.round(np.append(np.arange(low, high, step), high), 10))
    return tuple(float(value) for value in points)


def _levels(case: Case, variable: str) -> tuple[float, ...]:
    """Tablonun düzeyleri: verinin aralığında yuvarlak altı-yedi değer (Tablo 9.5'teki gibi)."""

    values = case.data[variable].astype(float)
    low, high = float(values.min()), float(values.max())
    span = high - low
    step = 10 ** math.floor(math.log10(span / 6)) if span > 0 else 1.0
    for factor in (1, 2, 5, 10):
        if span / (step * factor) <= 7:
            step *= factor
            break
    start = math.ceil(low / step) * step
    levels = np.arange(start, high + 1e-9 * max(1.0, abs(high)), step)
    digits = max(0, -math.floor(math.log10(step))) if step < 1 else 0
    return tuple(float(round(value, digits)) for value in levels)


def _noise(value: float, regressor, outcome) -> bool:
    """Katsayı hesap hassasiyetinde sıfır mı: |β̂| · s_X ≤ 10⁻⁹ · s_Y (ör. tam uyumda gerçek katsayısı sıfır olan kare
    terimi). Küçük ama gerçek bir katsayı sıfır sayılmaz (``ornek_regresyon.negligible`` ile aynı kural)."""

    spread_x, spread_y = float(np.std(regressor)), float(np.std(outcome))
    return value == 0 or abs(value) * spread_x <= 1e-9 * max(spread_y, np.finfo(float).tiny)


def _delta(values) -> float:
    """Ek artışın büyüklüğü Δ: aralığın beşte birinden küçük en büyük 10'un kuvveti (deneyim ya da eğitim yılında 1;
    TL cinsinden gelirde 1000; 0–0,01 arası bir oranda 0,001), tam sayılı değişkende en az 1. Bir birimlik artış dar
    aralıklı değişkende verinin dışına taşar, geniş aralıklıda (ör. 1 TL) anlamsız derecede küçük kalır."""

    values = np.asarray(values, dtype=float)
    span = float(values.max() - values.min())
    if span <= 0:
        return 1.0
    step = float(10.0 ** math.floor(math.log10(span / 5)))
    return max(1.0, step) if bool(np.all(values == np.round(values))) else step


def _turning(fit, variable: str, square: str) -> float:
    linear, curve = float(fit.params[variable]), float(fit.params[square])
    return -linear / (2 * curve) if curve != 0 else math.inf


def _step5(case: Case) -> LabStep:
    ly = _ly(case)
    m4 = _m4(case)
    squares = _square_names(case)
    first, *rest = squares
    second = rest[0] if rest else None
    data = case.data
    fit = _fit(case, ly, m4)
    values = data[first].astype(float)
    low, high = float(values.min()), float(values.max())
    integral = bool(np.all(values == np.round(values)))
    default = float(case.extra.get("adim5_default", nice(float(values.median()), 2)))
    control = number_control("adim5_x", f"{_title(case, first)} düzeyi x", default, low, high,
                             help="Bir birimlik ek artışın etkisinin hesaplandığı düzey.",
                             step=1.0 if integral and high - low <= 2000 else None)
    turning = _turning(fit, first, squares[first])
    grid = _grid(case, first)
    exact = _exact(case, ly, m4)  # uyum tamsa karesel katsayı yuvarlama gürültüsüdür: dönüm noktası tanımsız
    inside = grid[0] <= turning <= grid[-1] and not exact
    levels = _levels(case, first)
    others = [term for term in m4 if term not in (first, squares[first])]
    dd = digits_for(float(fit.params[squares[first]]), 6)
    unit = _unit(case, first)
    bounds_digits = 0 if integral else digits_for(max(abs(low), abs(high)), 2)
    delta = _delta(values)
    step_text = "bir birimlik" if delta == 1 else f"Δ = {level_text(delta)} birimlik"
    turning_digits = max(2, digits_for(turning, 2)) if math.isfinite(turning) else 2

    def change(x):
        """Δ'lık ek artışta log sonucun değişimi: Δ·(β̂₁ + β̂₂(2x + Δ)); Δ = 1'de β̂₁ + β̂₂(2x + 1)."""

        b1, b2 = E.ref("b_d"), E.ref("b_dd")
        if delta == 1:
            return E.add(b1, E.mul(b2, E.add(E.mul(2, x), 1)))
        return E.mul(delta, E.add(b1, E.mul(b2, E.add(E.mul(2, x), delta))))

    def approximate(x):
        """Yaklaşık yüzde etki: 100·Δ·(β̂₁ + 2β̂₂x)."""

        b1, b2 = E.ref("b_d"), E.ref("b_dd")
        inner = E.add(b1, E.mul(E.mul(2, b2), x))
        return E.mul(100, inner) if delta == 1 else E.mul(100 * delta, inner)

    def build(choices) -> tuple:
        x0 = float(choices["adim5_x"])
        b1, b2 = E.ref("b_d"), E.ref("b_dd")
        slope = E.add(b1, E.mul(E.mul(2, b2), E.ref("x0")))
        one_unit = change(E.ref("x0"))
        curve = E.add(E.ref("b_0"), E.add(E.mul(b1, E.var(first)), E.mul(b2, E.power(E.var(first), 2))))
        means: list = []
        for term in others:
            source = next((variable for variable, name in squares.items() if name == term), None)
            if source is not None:  # ikinci değişkenin karesi: ortalamanın karesi (notlardaki gibi)
                curve = E.add(curve, E.mul(E.ref(f"bo_{term}"), E.power(E.ref(f"ort5_{source}"), 2)))
            else:
                curve = E.add(curve, E.mul(E.ref(f"bo_{term}"), E.ref(f"ort5_{term}")))
                means.append(term)
        if second is not None and second not in means:
            means.append(second)
        operations: list = [
            ModelValue("b_0", "m4", "coef", "Sabit terim", term=INTERCEPT, decimals=4),
            ModelValue("b_d", "m4", "coef", f"{_title(case, first)} katsayısı β̂₁", term=first,
                       decimals=digits_for(float(fit.params[first]), 6)),
            ModelValue("b_dd", "m4", "coef", f"{_title(case, first)}² katsayısı β̂₂", term=squares[first], decimals=dd),
            *(ModelValue(f"bo_{term}", "m4", "coef", f"{case.name(term) if term in data.columns else term}: katsayı",
                         term=term, decimals=6, shown=False) for term in others),
            Scalar("donum_d", E.div(E.neg(b1), E.mul(2, b2)), f"{_title(case, first)} dönüm noktası −β̂₁ / (2β̂₂)",
                   decimals=turning_digits),
            Statistic(case.frame, first, "min", "en_az_d", f"Örneklemde en küçük {_name(case, first)}",
                      decimals=bounds_digits),
            Statistic(case.frame, first, "max", "en_cok_d", f"Örneklemde en büyük {_name(case, first)}",
                      decimals=bounds_digits),
            CopyFrame("donum_otesi", case.frame, "Dönüm noktasının ötesindeki gözlemleri saymak için verinin kopyası"),
            Derive("donum_otesi", "otesi_d", E.compare("gt", E.var(first), E.ref("donum_d")),
                   f"{_title(case, first)} dönüm noktasından büyük mü (1 = evet)"),
            Statistic("donum_otesi", "otesi_d", "sum", "sayi_d",
                      f"{_title(case, first)} dönüm noktasından büyük gözlem sayısı", decimals=0),
        ]
        if second is not None:
            square = squares[second]
            two = values_of(second)
            operations += [
                Scalar("donum_k", E.div(E.neg(E.ref(f"bo_{second}")), E.mul(2, E.ref(f"bo_{square}"))),
                       f"{_title(case, second)} dönüm noktası", decimals=2),
                Statistic(case.frame, second, "min", "en_az_k", f"Örneklemde en küçük {_name(case, second)}",
                          decimals=two),
                Statistic(case.frame, second, "max", "en_cok_k", f"Örneklemde en büyük {_name(case, second)}",
                          decimals=two),
                Derive("donum_otesi", "otesi_k", E.compare("gt", E.var(second), E.ref("donum_k")),
                       f"{_title(case, second)} dönüm noktasından büyük mü (1 = evet)"),
                Statistic("donum_otesi", "otesi_k", "sum", "sayi_k",
                          f"{_title(case, second)} dönüm noktasından büyük gözlem sayısı", decimals=0),
            ]
        operations += [
            InlineData("tablo95", (first,), tuple((value,) for value in levels),
                       f"{_title(case, first)} düzeyleri (Tablo 9.5'teki gibi)"),
            Derive("tablo95", "yaklasik", approximate(E.var(first)),
                   "Yaklaşık yüzde etki: 100·(β̂₁ + 2β̂₂·x)" if delta == 1 else
                   f"Yaklaşık yüzde etki: 100·Δ·(β̂₁ + 2β̂₂·x), Δ = {level_text(delta)}"),
            Derive("tablo95", "tam", E.mul(100, E.sub(E.exp(change(E.var(first))), 1)),
                   "Bir birimlik tam yüzde değişim: 100·(exp(β̂₁ + β̂₂(2x + 1)) − 1)" if delta == 1 else
                   f"Δ = {level_text(delta)} birimlik tam yüzde değişim: 100·(exp(Δ·(β̂₁ + β̂₂(2x + Δ))) − 1)"),
            ShowFrame("tablo95", (first, "yaklasik", "tam"),
                      f"{_title(case, first)} düzeyine göre {step_text} ek artışın tahmin edilen ilişkisi", decimals=2),
            Scalar("x0", E.const(x0), "Seçilen düzey x",
                   decimals=0 if float(x0).is_integer() else max(2, digits_for(x0, 2))),
            Scalar("egim_x0", slope, "x düzeyinde marjinal etki (log birim): β̂₁ + 2β̂₂x",
                   decimals=max(5, digits_for(float(fit.params[first]), 5))),
            Scalar("yuzde_x0", approximate(E.ref("x0")), f"x düzeyinde {step_text} ek artışın yaklaşık yüzde etkisi",
                   decimals=2, percent=True),
            Scalar("tam_x0", E.mul(100, E.sub(E.exp(one_unit), 1)),
                   "x'ten x + 1'e tam yüzde değişim" if delta == 1 else
                   f"x ile x + Δ arasında tam yüzde değişim (Δ = {level_text(delta)})", decimals=2, percent=True),
            *(Statistic(case.frame, term, "mean", f"ort5_{term}", f"{case.name(term)}: örneklem ortalaması", decimals=4)
              for term in means),
            InlineData("egri", (first,), tuple((value,) for value in grid),
                       f"{_title(case, first)} ızgarası: verideki aralık ({level_text(grid[0])}–{level_text(grid[-1])})"),
            Derive("egri", "tahmin", curve, "Tahmin edilen log sonuç: diğer açıklayıcılar örneklem ortalamasında"),
            Derive("egri", "etki", approximate(E.var(first)),
                   f"{capital(step_text)} ek artışın yaklaşık yüzde etkisi: 100·(β̂₁ + 2β̂₂·x)" if delta == 1 else
                   f"{capital(step_text)} ek artışın yaklaşık yüzde etkisi: 100·Δ·(β̂₁ + 2β̂₂·x)"),
            Scalar("sifir", E.const(0), "Sıfır etki çizgisi", decimals=0, shown=False),
            ScatterPlot(case.frame, first, ly, _axis(case, first), log_label(case, case.roles[SONUC]),
                        f"{_title(case, first)} ile log sonuç arasındaki karesel ilişki (Şekil 9.2'deki gibi)", size=6,
                        opacity=0.35, curves=(("egri", first, "tahmin", "Tahmin edilen eğri"),)),
            LineChart("egri", first, "etki", _axis(case, first), f"{capital(step_text)} ek artışın yaklaşık yüzde etkisi",
                      f"{_title(case, first)} için tahmin edilen marjinal etki (Şekil 9.3'teki gibi)",
                      references=(("sifir", "Sıfır etki"),), markers=False,
                      vlines=((("donum_d", "Dönüm noktası"),) if inside else ()) + (("x0", "Seçilen düzey"),)),
        ]
        return tuple(operations)

    def values_of(variable: str) -> int:
        column = data[variable].astype(float)
        return 0 if bool(np.all(column == np.round(column))) else digits_for(float(column.abs().max()), 2)

    def note(state, choices) -> str:
        x0 = float(choices["adim5_x"])
        s = state.scalars
        result = state.models["m4"]
        slope = s["egim_x0"]
        here = level_text(x0)
        near = (not exact and math.isfinite(s["donum_d"]) and high > low
                and abs(x0 - s["donum_d"]) <= max(0.01 * (high - low), delta))

        def word(value: float) -> str:
            return "daha yüksek" if value > 0 else "daha düşük"

        if near:
            direction = (f"{capital(phrase(case, first))} {here}{unit} dönüm noktasına çok yakındır: {step_text} bir artış "
                         f"dönüm noktasını aşabilir. Yaklaşık etki %{sayi(s['yuzde_x0'], 2)}, tam değişim "
                         f"%{sayi(s['tam_x0'], 2)}.")
        elif sayi(s["yuzde_x0"], 2) == sayi(0, 2):
            direction = f"{capital(phrase(case, first))} {here}{unit} iken marjinal etki yaklaşık sıfırdır."
        elif slope > 0:
            direction = (f"{capital(phrase(case, first))} {here}{unit} iken {step_text} ek artış, tahmin edilen log "
                         f"sonuçta yaklaşık {sayi(slope * delta, max(5, digits_for(slope * delta, 5)))} artışla "
                         f"ilişkilidir: yaklaşık %{sayi(abs(s['yuzde_x0']), 2)} {word(s['yuzde_x0'])}, tam hesapla "
                         f"%{sayi(abs(s['tam_x0']), 2)} {word(s['tam_x0'])} sonuç.")
        else:
            direction = (f"{capital(phrase(case, first))} {here}{unit} iken tahmin edilen marjinal etki negatiftir "
                         f"({sayi(slope, max(5, digits_for(slope, 5)))}; {step_text} ek artışta yaklaşık "
                         f"%{sayi(abs(s['yuzde_x0']), 2)} daha düşük sonuç). Bu, ileri düzeylerin sonucu nedensel olarak "
                         "düşürdüğünü kanıtlamaz; eğrinin bu bölgesi gözlemsel ilişkidir.")
        text = f"{direction} Etki düzeye bağlıdır: β̂₁ + 2β̂₂x. "
        if exact:
            return (text + "Uyum tam olduğu için karesel terimlerin katsayıları yuvarlama gürültüsüdür: dönüm noktası "
                    "tanımsızdır ve yorumlanmaz (§9.4–9.5)." + EXACT_MULTI_NOTE)
        text += _turning_text(case, first, squares[first], s["donum_d"], s["en_az_d"], s["en_cok_d"], s["sayi_d"],
                              float(result.params[squares[first]]), float(result.pvalues[squares[first]]))
        if second is not None:
            text += " " + _turning_text(case, second, squares[second], s["donum_k"], s["en_az_k"], s["en_cok_k"],
                                        s["sayi_k"], float(result.params[squares[second]]),
                                        float(result.pvalues[squares[second]]))
        return text.rstrip(".") + " (§9.4–9.5)."

    checks = [
        *(_check(f"Tablo: {_name(case, first)} {level_text(value)}, {name}", CellTarget("tablo95", column, row), 2)
          for row, value in enumerate(levels, start=1)
          for column, name in (("yaklasik", "yaklaşık yüzde etki"), ("tam", "bir birimlik tam yüzde değişim"))),
        _scalar("b_d", f"{_title(case, first)} katsayısı", digits_for(float(fit.params[first]), 4)),
        _scalar("b_dd", f"{_title(case, first)}² katsayısı", dd),
        _scalar("donum_d", "Dönüm noktası", turning_digits),
        _scalar("en_az_d", "En küçük değer", bounds_digits),
        _scalar("en_cok_d", "En büyük değer", bounds_digits),
        _scalar("sayi_d", "Dönüm noktasından büyük gözlem", 0),
    ]
    if second is not None:
        checks += [_scalar("donum_k", f"{_title(case, second)} dönüm noktası", 2),
                   _scalar("en_az_k", "İkinci değişken: en küçük değer", values_of(second)),
                   _scalar("en_cok_k", "İkinci değişken: en büyük değer", values_of(second)),
                   _scalar("sayi_k", "İkinci değişken: dönüm noktasından büyük gözlem", 0)]
    if exact:  # dönüm noktası ve ötesindeki gözlem sayısı yuvarlama gürültüsüne bağlıdır
        checks = [check for check in checks
                  if not (isinstance(check.target, ScalarTarget) and check.target.name.startswith(("donum_", "sayi_")))]
    return interactive_step(
        number=5,
        title="Marjinal etki ve dönüm noktası",
        note=NoteRef("9.5", 0, ("Tablo 9.5", "Şekil 9.2", "Şekil 9.3")),
        explanation=(
            f"Karesel modelde {phrase(case, first)} değişkeninin etkisi sabit değildir: marjinal etki "
            "$\\hat\\beta_1 + 2\\hat\\beta_2 x$, x'ten x + 1'e tam değişim $\\hat\\beta_1 + \\hat\\beta_2(2x + 1)$. "
            + ("" if delta == 1 else f"Bu değişkenin aralığı dar olduğu için ek artış bir birim yerine Δ = "
               f"{level_text(delta)} alınır: x'ten x + Δ'ya tam değişim $\\Delta[\\hat\\beta_1 + \\hat\\beta_2(2x + "
               "\\Delta)]$. ") +
            "Dönüm noktası $x^* = -\\hat\\beta_1/(2\\hat\\beta_2)$; $\\hat\\beta_2 < 0$ ise tepe, $\\hat\\beta_2 > 0$ "
            "ise dip noktasıdır. Dönüm noktası veri aralığında ve yeterli gözlemin bulunduğu bölgede yorumlanır; karesel "
            "terim anlamsızsa dönüm noktası da belirsizdir. Düzeyi değiştirin."
        ),
        controls=(control,),
        build=build,
        checks=tuple(checks),
        note_for=lambda state, choices: note(state, choices),
    )


def _axis(case: Case, column: str) -> str:
    unit = case.units.get(column, "")
    return f"{case.name(column)} ({unit})" if unit else case.name(column)


def _turning_text(case: Case, variable: str, square: str, turning: float, low: float, high: float, beyond: float,
                  curve: float, p: float) -> str:
    """Dönüm noktasının yorumu: tepe ya da dip, veri aralığında mı, kaç gözlem ötesinde, karesel terim anlamlı mı."""

    unit = _unit(case, variable)
    kind = "tepe" if curve < 0 else "dip"
    name = phrase(case, variable)
    span = f"{level_text(low)}–{level_text(high)}{unit}"
    if not math.isfinite(turning):
        return f"{capital(name)} karesinin katsayısı sıfırdır: dönüm noktası yoktur."
    text = (f"{capital(name)} için dönüm noktası {sayi(turning, max(2, digits_for(turning, 2)))}{unit} ({kind} noktası); "
            f"örneklemde {name} {span}. ")
    if not low <= turning <= high:
        trend = "artandır" if (curve < 0) == (turning > high) else "azalandır"
        text += (f"Dönüm noktası veri aralığının dışındadır: veri aralığında tahmin edilen eğri {trend} ve dönüm "
                 "noktası yorumlanmaz.")
    else:
        count = int(round(beyond))
        text += f"Dönüm noktası veri aralığının içindedir; dönüm noktasından büyük {sayim(count)} gözlem vardır."
        if count < 0.05 * len(case.data):
            text += " Eğrinin o bölgesi çok az gözleme dayanır."
    if p >= 0.05:
        text += (f" Karesinin katsayısı yüzde 5 düzeyinde anlamlı değildir ({p_text(p)}): eğriliğin kendisi belirsizdir, "
                 "bu yüzden dönüm noktası da belirsizdir.")
    return text


# --- Adım 6: merkezleme -------------------------------------------------------------------------------------------

def _step6(case: Case) -> LabStep:
    ly = _ly(case)
    m4 = _m4(case)
    squares = _square_names(case)
    first = next(iter(squares))
    values = case.data[first].astype(float)
    low, high = float(values.min()), float(values.max())
    integral = bool(np.all(values == np.round(values)))
    default = float(case.extra.get("adim6_default", nice(float(values.median()), 2)))
    control = number_control("adim6_c", f"Merkez noktası c ({_name(case, first)})", default, low, high,
                             help=f"{_title(case, first)} c düzeyinde merkezlenir: {_name(case, first)} − c.",
                             step=1.0 if integral and high - low <= 2000 else None)
    taken = _taken(case) | set(squares.values())
    centered = free_name(f"{first}_c", taken)
    centered_sq = free_name(f"{first}_c_kare", taken | {centered})
    terms = tuple(centered if term == first else centered_sq if term == squares[first] else term for term in m4)
    noise = max(noise_decimals(case, case.roles[SONUC]) if case.own else 10, 6)
    exact = _exact(case, ly, m4)

    def table(c: float, model: str, frame: str, suffix: str, title: str) -> tuple:
        shown = level_text(c)
        return (
            CopyFrame(frame, case.frame, "Merkezleme için verinin kopyası (özgün veri değişmez)"),
            Derive(frame, centered, E.sub(E.var(first), c), f"Merkezlenmiş {_name(case, first)}: {first} − {shown}"),
            Derive(frame, centered_sq, E.power(E.var(centered), 2), "Merkezlenmiş değişkenin karesi"),
            OLS(model, frame, ly, terms, f"{shown} düzeyinde merkezlenmiş model: {ly} ~ {' + '.join(terms)}"),
            Scalar(f"c_merkez{suffix}", E.const(c), "Merkez noktası c", decimals=0 if float(c).is_integer() else 2),
            Scalar(f"egim_ham{suffix}", E.add(E.ref("b_d"), E.mul(E.mul(2, E.ref("b_dd")), E.ref(f"c_merkez{suffix}"))),
                   f"Ham modelde {shown} düzeyindeki eğim: β̂₁ + 2β̂₂c", decimals=6),
            ModelValue(f"r2_mer{suffix}", model, "r2", "Merkezlenmiş model: R²", decimals=6),
            ModelValue(f"r2d_mer{suffix}", model, "adj_r2", "Merkezlenmiş model: düzeltilmiş R²", decimals=6),
            ModelValue(f"hkt_mer{suffix}", model, "ssr", "Merkezlenmiş model: HKT", decimals=4),
            ModelValue(f"egim_mer{suffix}", model, "coef", "Merkezlenmiş modelde doğrusal terimin katsayısı",
                       term=centered, decimals=6),
            ScalarTable((("R²", E.ref("r2_ham")), ("Düzeltilmiş R²", E.ref("r2d_ham")), ("HKT", E.ref("hkt_ham")),
                         (f"{shown} düzeyindeki eğim", E.ref(f"egim_ham{suffix}"))), f"ham96{suffix}", decimals=6),
            ScalarTable((("R²", E.ref(f"r2_mer{suffix}")), ("Düzeltilmiş R²", E.ref(f"r2d_mer{suffix}")),
                         ("HKT", E.ref(f"hkt_mer{suffix}")), (f"{shown} düzeyindeki eğim", E.ref(f"egim_mer{suffix}"))),
                        f"merkez96{suffix}", decimals=6),
            JoinColumns(f"tablo96{suffix}", (("Ham model", f"ham96{suffix}", "deger"),
                                             ("Merkezlenmiş model", f"merkez96{suffix}", "deger")),
                        decimals=6, heading="Ölçüt", row_decimals=(("HKT", 4),), title=title),
        )

    def build(choices) -> tuple:
        c = float(choices["adim6_c"])
        same = c == control.default
        noted = () if same else table(control.default, "m_c_n", "merkez_n", "_n",
                                      f"Varsayılan: ham ve {level_text(control.default)} düzeyinde merkezlenmiş model")
        return (
            ModelValue("r2_ham", "m4", "r2", "Ham model: R²", decimals=6),
            ModelValue("r2d_ham", "m4", "adj_r2", "Ham model: düzeltilmiş R²", decimals=6),
            ModelValue("hkt_ham", "m4", "ssr", "Ham model: HKT (artık kareleri toplamı)", decimals=4),
            *noted,
            *table(c, "m_c", "merkez", "", f"Ham ve {level_text(c)} düzeyinde merkezlenmiş karesel model (Tablo 9.6'daki "
                                           "gibi)" if same else f"Seçiminiz: ham ve {level_text(c)} düzeyinde merkezlenmiş "
                                                                "model"),
            Residuals("merkez", "u_ham", "m4", "Ham modelin artıkları"),
            Residuals("merkez", "u_mer", "m_c", "Merkezlenmiş modelin artıkları"),
            Derive("merkez", "u_fark", E.absolute(E.sub(E.var("u_ham"), E.var("u_mer"))), "İki modelin artık farkı"),
            Statistic("merkez", "u_fark", "max", "azami_fark", "Artıkların (ve tahmin edilen değerlerin) azami farkı",
                      decimals=noise),
            PairStatistic("merkez", first, squares[first], "corr", "r_ham",
                          f"Korelasyon: {_name(case, first)} ile karesi", decimals=3),
            PairStatistic("merkez", centered, centered_sq, "corr", "r_mer",
                          "Korelasyon: merkezlenmiş değişken ile karesi", decimals=3),
        )

    def note(state, choices) -> str:
        c = float(choices["adim6_c"])
        s = state.scalars
        shown = level_text(c)
        return (f"Merkezlenmiş modelde doğrusal terimin katsayısı {sayi(s['egim_mer'], 6)}: ham modelde {shown} "
                f"düzeyindeki eğim β̂₁ + 2β̂₂·{shown} = {sayi(s['egim_ham'], 6)} ile aynıdır. R², düzeltilmiş R² ve HKT "
                "değişmez; iki modelin tahmin edilen değerleri arasındaki en büyük fark yalnız bilgisayar yuvarlaması "
                f"düzeyindedir. Merkezleme {phrase(case, first)} ile karesi arasındaki korelasyonu değiştirir "
                f"({sayi(s['r_ham'], 3)} → {sayi(s['r_mer'], 3)}); ekonomik ilişkiyi, eksik değişken sorununu ya da "
                "içselliği değiştirmez (§9.6)." + (EXACT_MULTI_NOTE if exact else ""))

    return interactive_step(
        number=6,
        title="Merkezleme: katsayıyı anlamlı bir noktada okumak",
        note=NoteRef("9.6", 0, ("Tablo 9.6",)),
        explanation=(
            f"Karesel modelde {phrase(case, first)} katsayısı {phrase(case, first)} = 0 noktasındaki eğimdir. Değişken c "
            "düzeyinde merkezlenirse ($X_c = X - c$) doğrusal terim doğrudan c düzeyindeki eğimi verir. Model aynı "
            "tahmin edilen değerleri üretir; yalnız katsayıların referans noktası değişir. Merkez noktasını değiştirin."
        ),
        controls=(control,),
        uses=(),
        build=build,
        checks=(
            _check("Tablo: ham model, R²", TableTarget("tablo96", "R²", "Ham model"), 6),
            _check("Tablo: merkezlenmiş model, R²", TableTarget("tablo96", "R²", "Merkezlenmiş model"), 6),
            _check("Tablo: ham model, düzeltilmiş R²", TableTarget("tablo96", "Düzeltilmiş R²", "Ham model"), 6),
            _check("Tablo: merkezlenmiş model, düzeltilmiş R²",
                   TableTarget("tablo96", "Düzeltilmiş R²", "Merkezlenmiş model"), 6),
            _check("Tablo: ham model, HKT", TableTarget("tablo96", "HKT", "Ham model"), 4),
            _check("Tablo: merkezlenmiş model, HKT", TableTarget("tablo96", "HKT", "Merkezlenmiş model"), 4),
            _check("Tablo: ham model, eğim", TableTarget("tablo96", f"{level_text(control.default)} düzeyindeki eğim",
                                                         "Ham model"), 6),
            _check("Tablo: merkezlenmiş model, eğim",
                   TableTarget("tablo96", f"{level_text(control.default)} düzeyindeki eğim", "Merkezlenmiş model"), 6),
            _scalar("azami_fark", "İki model aynı tahmin edilen değerleri üretir", noise),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 7: model karşılaştırması ---------------------------------------------------------------------------------

def _models(case: Case) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Karşılaştırılan modeller: M₁ doğrusal, her karesel terim ayrı ayrı ve birlikte (notlardaki M₁–M₄)."""

    m1 = _m1(case)
    squares = _square_names(case)
    items = list(squares.items())
    found = [("m1", f"{_model_title(1)}: Doğrusal", m1)]
    if len(items) == 2:
        for name, (variable, square) in (("m2", items[0]), ("m3", items[1])):
            terms = tuple(term for value in m1 for term in ((value, square) if value == variable else (value,)))
            index = len(found) + 1
            found.append((name, f"{_model_title(index)}: {_title(case, variable)} karesel", terms))
        found.append(("m4", f"{_model_title(4)}: İki karesel terim", _m4(case)))
    else:
        found.append(("m4", f"{_model_title(2)}: {_title(case, items[0][0])} karesel", _m4(case)))
    return tuple(found)


def _extras(case: Case) -> dict[str, tuple[str, object]]:
    """Beşinci modelin ek terimleri: anahtar → (açıklama, ifade ya da veri sütunu)."""

    if "extra_terms" in case.extra:
        return dict(case.extra["extra_terms"])
    x = case.roles[ACIKLAYICI]
    found: dict[str, tuple[str, object]] = {}
    if case.data[x].nunique() < 4:  # üç farklı değerde x'in her fonksiyonu 1, x, x² ile tam ilişkilidir
        return found
    taken = _taken(case) | set(_square_names(case).values())
    if float(case.data[x].min()) > 0:
        found[free_name(f"{x}_log", taken)] = (f"{case.name(x)} değişkeninin logaritması", E.log(E.var(x)))
    found[free_name(f"{x}_kup", taken)] = (f"{case.name(x)} değişkeninin küpü", E.power(E.var(x), 3))
    return found


def _step7(case: Case) -> LabStep:
    ly = _ly(case)
    models = _models(case)
    extras = _extras(case)
    m4 = _m4(case)
    created = {"m4"} | ({"m1"} if _percent_source(case)[0] == "m1" else set())
    words = {key: words for key, (words, _) in extras.items()}

    def build(choices) -> tuple:
        extra = choices["adim7_ek"]
        chosen = list(models)
        operations: list = [OLS(name, case.frame, ly, terms, f"{label}: {ly} ~ {' + '.join(terms)}")
                            for name, label, terms in models if name not in created]
        if extra != "yok":
            terms = (*m4, extra)
            label = f"{_model_title(len(models) + 1)}: {models[-1][1].split(':')[0]} + {words[extra]}"
            frame = case.frame
            source = extras[extra][1]
            if not isinstance(source, str):  # türetilen ek terim
                frame = "model5"
                operations += [CopyFrame("model5", case.frame, "Beşinci model için verinin kopyası (özgün veri değişmez)"),
                               Derive("model5", extra, source, capital(words[extra]))]
            operations.append(OLS("m5", frame, ly, terms, f"{label}: {ly} ~ {' + '.join(terms)}"))
            chosen.append(("m5", label, terms))
        for name, label, terms in chosen:
            operations += [
                Scalar(f"k_{name}", E.const(len(terms)), f"{label}: eğim sayısı", decimals=0),
                ModelValue(f"r2_{name}", name, "r2", f"{label}: R²", decimals=4),
                ModelValue(f"r2d_{name}", name, "adj_r2", f"{label}: düzeltilmiş R²", decimals=4),
                ModelValue(f"hkt_{name}", name, "ssr", f"{label}: HKT", decimals=3),
            ]
        tables = [ScalarTable(tuple((label, E.ref(f"{prefix}_{name}")) for name, label, _ in chosen), result,
                              decimals=4, heading="Model")
                  for prefix, result in (("k", "k97"), ("r2", "r297"), ("r2d", "r2d97"), ("hkt", "hkt97"))]
        same = extra == "yok"
        return (
            *operations,
            *tables,
            JoinColumns("tablo97", (("Eğim sayısı", "k97", "deger"), ("R²", "r297", "deger"),
                                    ("Düzeltilmiş R²", "r2d97", "deger"), ("HKT", "hkt97", "deger")),
                        decimals=4, heading="Model", column_decimals=(("Eğim sayısı", 0), ("HKT", 3)),
                        title="Modellerin uyum karşılaştırması (Tablo 9.7'deki gibi)" if same
                        else "Modellerin uyum karşılaştırması ve seçtiğiniz beşinci model"),
            BarChart("r2d97", "deger", "Model", "Düzeltilmiş R²",
                     "Fonksiyonel biçimlerde düzeltilmiş R² (Şekil 9.4'teki gibi)" if same
                     else "Fonksiyonel biçimlerde düzeltilmiş R² (beşinci modelle)", decimals=4),
        )

    def note(state, choices) -> str:
        extra = choices["adim7_ek"]
        s = state.scalars
        first, last = models[0], models[-1]
        if exact_all:
            return ("Uyum tam olduğu için bütün modellerin R² ve düzeltilmiş R² değerleri 1'e çok yakındır: uyum ölçüleri "
                    "modelleri ayırt etmez; farklar yuvarlama gürültüsüdür (§9.7)." + EXACT_MULTI_NOTE)
        best = max(models, key=lambda item: s[f"r2d_{item[0]}"])
        text = (f"R² her ek terimle artar: {_at(first[1])} {sayi(s['r2_m1'], 4)}, {_at(last[1])} "
                f"{sayi(s['r2_m4'], 4)}. Düzeltilmiş R² ek terimleri cezalandırır; bu {len(models)} model içinde en "
                f"yüksek değer {_at(best[1], copula=True)} ({sayi(s[f'r2d_{best[0]}'], 4)}). ")
        if extra != "yok":
            rose = s["r2d_m5"] > s["r2d_m4"]
            text += (f"Beşinci modelde {words[extra]} eklenince R² {sayi(s['r2_m4'], 4)} → {sayi(s['r2_m5'], 4)} artar; "
                     f"düzeltilmiş R² {sayi(s['r2d_m4'], 4)} → {sayi(s['r2d_m5'], 4)} {'artar' if rose else 'azalır'}. ")
        return text + ("En yüksek uyum tek başına model seçimi değildir: teori, grafik, katsayıların yorumu, ortak test ve "
                       "basitlik birlikte değerlendirilir; uyum ölçüleri yalnız aynı bağımlı değişken ve aynı örneklemle "
                       "karşılaştırılır (§9.7).")

    exact_all = _exact(case, ly, _m1(case))  # en küçük model tam uyuyorsa bütün modeller tam uyar
    options_7 = (("yok", "Yok (varsayılan modeller)"), *((key, capital(value)) for key, value in words.items()))
    checks = tuple(check for name, label, terms in models for check in (
        _check(f"Tablo: {label}, eğim sayısı", TableTarget("tablo97", label, "Eğim sayısı"), 0),
        _check(f"Tablo: {label}, R²", TableTarget("tablo97", label, "R²"), 4),
        _check(f"Tablo: {label}, düzeltilmiş R²", TableTarget("tablo97", label, "Düzeltilmiş R²"), 4),
        _check(f"Tablo: {label}, HKT", TableTarget("tablo97", label, "HKT"), digits_for(float(_fit(case, ly, terms).ssr), 3)),
    ))
    names = ", ".join(md(label) if case.own else label for _, label, _ in models)  # Markdown metni
    return interactive_step(
        number=7,
        title="Model seçimi: yalnız R² yetmez",
        note=NoteRef("9.7", 0, ("Tablo 9.7", "Şekil 9.4")),
        explanation=(
            f"Log sonuç modelleri karşılaştırılır: {names}. Aynı bağımlı değişken ve aynı örneklemle R², düzeltilmiş R² "
            "ve HKT karşılaştırılabilir. R² her ek terimle artar; düzeltilmiş R² gereksiz terimleri cezalandırır. "
            + ("Karşılaştırmaya beşinci bir model ekleyin." if len(options_7) > 1 else
               "Temel açıklayıcının üçten az farklı değeri olduğu için beşinci model seçeneği yoktur.")
        ),
        controls=(Choice("adim7_ek", "Beşinci model: son modele eklenen terim", options_7, "yok",
                         help="Ek bir terim R²'yi azaltmaz; düzeltilmiş R² ise azalabilir (§9.7)."),),
        build=build,
        checks=checks,
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 8: makale tablosu ---------------------------------------------------------------------------------------

def _step8(case: Case) -> LabStep:
    models = _models(case)
    squares = _square_names(case)
    m4 = _m4(case)
    ly = _ly(case)
    fit = _fit(case, ly, m4)
    exact = _exact(case, ly, m4)
    labels = {name: label for name, label, _ in models}
    columns_of = {"m4": tuple(squares.values())}
    if len(models) == 4:
        columns_of.update({"m2": (squares[next(iter(squares))],), "m3": (list(squares.values())[1],)})
    term_decimals = tuple((name, max(6, digits_for(float(fit.params[name]), 4))) for name in squares.values())
    options_ = dict(decimals=4, term_decimals=term_decimals, adj_r2=True, r2_decimals=3, extra_decimals=3, exact=True)

    def build(choices) -> tuple:
        model = choices["adim8_model"] if len(models) == 4 else "m4"
        single = len(columns_of[model]) == 1
        test = JointTest("F_98", "p_98", model, columns_of[model],
                         "Sütun (2): karesel terimin F testi (tek kısıt)" if single
                         else "Sütun (2): karesel terimlerin ortak F testi", decimals=3)
        heading = labels[model].split(":")[0]
        if model == "m4":
            return (
                test,
                RegressionTable((("(1) Doğrusal", "m1"), ("(2) Karesel", "m4")), m4, "tablo98",
                                f"Log sonucun fonksiyonel biçimi (Tablo 9.8'deki gibi) · bağımlı değişken: "
                                f"{log_label(case, case.roles[SONUC])}",
                                extra=(("ortak_F", "Karesel terimler: F", ("", "F_98")),
                                       ("ortak_p", "Karesel terimler: p-değeri", ("", "p_98"))), **options_),
            )
        return (  # varsayılan Sütun (2) seçilen modelle yan yana
            test,
            JointTest("F_98n", "p_98n", "m4", columns_of["m4"], "Varsayılan Sütun (2): karesel terimlerin ortak F testi",
                      decimals=3),
            RegressionTable((("(1) Doğrusal", "m1"), (f"(2) Seçiminiz: {heading}", model), ("(2) Varsayılan", "m4")), m4,
                            "tablo98", f"Log sonucun fonksiyonel biçimi: seçtiğiniz {heading} ve varsayılan model",
                            extra=(("ortak_F", "Karesel terim testi: F", ("", "F_98", "F_98n")),
                                   ("ortak_p", "Karesel terim testi: p-değeri", ("", "p_98", "p_98n"))), **options_),
        )

    def note(state, choices) -> str:
        model = choices.get("adim8_model", "m4") if len(models) == 4 else "m4"
        s = state.scalars
        chosen = columns_of[model]
        subject = "iki karesel terimin birlikte" if len(chosen) == 2 else "karesel terimin"
        if exact:
            text = f"Sütun (2)'de {subject} sıfır olduğu hipotezinin F'si uyum tam olduğu için yuvarlama hatasıdır. "
        else:
            text = f"Sütun (2)'de {subject} sıfır olduğu hipotezi için F = {sayi(s['F_98'], 3)}, {p_text(s['p_98'])}. "
        if len(chosen) == 1 and len(columns_of["m4"]) == 2:
            text += "Tek kısıtta bu F, karesel terimin t istatistiğinin karesidir (§8.8). "
            if not exact:
                text += f"Varsayılan Sütun (2) iki karesel terimlidir: ortak F = {sayi(s['F_98n'], 3)}, {p_text(s['p_98n'])}. "
        value = float(state.models[model].params[chosen[0]])
        names = listing([phrase(case, variable) for variable, name in squares.items() if name in chosen])
        return text + (f"Karesel terimlerin katsayıları küçük görünür (ör. {sayi(value, max(6, digits_for(value, 4)))}); "
                       "fakat X² ile çarpıldıkları için etkileri X büyüdükçe birikir. Sütun (2)'de "
                       f"{names} {'katsayısı' if len(chosen) == 1 else 'katsayıları'} tek başına yorumlanmaz: marjinal "
                       "etki β̂₁ + 2β̂₂x ile belirli bir düzeyde "
                       "hesaplanır. Bu tablo nedensel bir etkiyi kanıtlamaz (§9.8)."
                       + (EXACT_MULTI_NOTE if exact else ""))

    checks = []
    for heading, terms in (("(1) Doğrusal", _m1(case)), ("(2) Karesel", m4)):
        for name in terms:
            digits = dict(term_decimals).get(name, 4)
            checks += [_check(f"Tablo: {heading}, {name}", TableTarget("tablo98", name, heading), digits),
                       _check(f"Tablo: {heading}, {name} (SH)", TableTarget("tablo98", f"{name}_sh", heading), digits)]
        checks += [_check(f"Tablo: {heading}, gözlem sayısı", TableTarget("tablo98", "n", heading), 0),
                   _check(f"Tablo: {heading}, R²", TableTarget("tablo98", "r2", heading), 3),
                   _check(f"Tablo: {heading}, düzeltilmiş R²", TableTarget("tablo98", "adj_r2", heading), 3)]
    if not exact:
        checks.append(_check("Tablo: (2) Karesel, karesel terimlerin F'si", TableTarget("tablo98", "ortak_F",
                                                                                         "(2) Karesel"), 3))
    lead = ("Makale tablosunda değişken adları modelin biçimini gösterir: bağımlı değişken log sonuç, kareli terimler "
            "karesel terimlerdir. Okuma sırası: bağımlı değişken log mu, karesel terimle birlikte düzey terimi var mı, "
            "işaretler eğrinin yönü hakkında ne söylüyor, karesel terimler birlikte gerekli mi, düzeltilmiş R² aynı "
            "bağımlı değişkenle mi karşılaştırılıyor.")
    common = dict(number=8, title="Makale tablosunda fonksiyonel biçim", note=NoteRef("9.8", 0, ("Tablo 9.8",)),
                  checks=stable_checks(tuple(checks), exact, table="tablo98"),
                  note_for=lambda state, choices: note(state, choices))
    if len(models) == 4:
        control = Choice("adim8_model", "Sütun (2)'deki karesel model",
                         tuple((name, labels[name]) for name in ("m4", "m2", "m3")), "m4",
                         help="Varsayılan: iki karesel terimli model.")
        return interactive_step(explanation=f"{lead} Sütun (2)'deki modeli değiştirin.", controls=(control,),
                                build=build, **common)
    return LabStep(explanation=lead, operations=build({}), **common)


# --- Tanım ------------------------------------------------------------------------------------------------------

def build(case: Case) -> LabSpec:
    """Konu 9 uygulamasını verilen veriyle kurar; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    y = case.roles[SONUC]
    labels = labels_of(case)
    labels[_ly(case)] = labels.get(_ly(case)) or log_label(case, y)
    labels[INTERCEPT] = "Sabit terim"
    ys, xs, _, _ = _units(case)
    for column, _, _, comment, _, _ in (*ys.values(), *xs.values()):
        labels.setdefault(column, comment.split(": ")[0] if not case.own else comment)
    for name, original in _z_names(case).items():
        shown = labels.get(name, case.name(name))
        labels.setdefault(original, f"Standartlaştırılmış {shown if case.own else tr_lower(shown[:1]) + shown[1:]}")
    for variable, name in _square_names(case).items():
        labels.setdefault(name, f"{case.name(variable)}²" if case.own else labels.get(name, name))
    first = next(iter(_square_names(case)))
    labels.update({"beta": "Katsayı β", "dx": "X'teki değişim ΔX", "yaklasik": "Yaklaşık yüzde", "tam": "Tam yüzde",
                   "tahmin": "Tahmin edilen log sonuç", "etki": "Bir birimlik ek artışın yaklaşık yüzde etkisi"})
    taken = _taken(case) | set(_square_names(case).values())
    centered = free_name(f"{first}_c", taken)
    labels[centered] = f"Merkezlenmiş {case.name(first)} ({case.name(first)} − c)"
    labels[free_name(f"{first}_c_kare", taken | {centered})] = f"Merkezlenmiş {case.name(first)} karesi"
    for key, (words, _) in _extras(case).items():
        labels.setdefault(key, capital(words))
    spec = LabSpec(
        topic_key=TOPIC,
        title=str(case.extra.get("title", TITLE)),
        note_section="9",
        steps=(_step1(case), _step2(case), _step3(case), _step4(case), _step5(case), _step6(case), _step7(case),
               _step8(case)),
        labels=tuple(labels.items()),
        source=case.source,
    )
    return with_app_values(spec)


# --- Alternatif örnek ----------------------------------------------------------------------------------------

def alternative_case() -> Case:
    return wage2_case(
        ("educ", "exper", "tenure", "IQ"),
        title="Uygulama: Ölçekleme, Log ve Karesel Terimler, Model Seçimi (WAGE2)",
        main=("educ", "exper", "tenure"),
        std_options=("educ", "exper", "tenure", "IQ"),
        squares=("exper", "tenure"),
        square_names={"exper": "expersq", "tenure": "tenursq"},
        linear="educ",
        adim5_default=10,
        adim6_default=10,
        extra_terms={"educsq": ("eğitimin karesi", E.power(E.var("educ"), 2)), "IQ": ("IQ puanı", "IQ"),
                     "tenuresqrt": ("kıdemin karekökü", E.power(E.var("tenure"), 0.5))},
    )


@cache
def alternative() -> LabSpec:
    return build(alternative_case())


STORY = ("Veri WAGE2'dir (935 erkek çalışan, 1980; log aylık kazanç ~ eğitim + deneyim + kıdem). Karesel terimler "
         "notlardaki gibi deneyim ve kıdemdedir; standartlaştırma ve model karşılaştırmasında notlardaki bakmakla "
         "yükümlü kişi sayısının yerinde IQ puanı vardır.")


# --- Kendi verin ---------------------------------------------------------------------------------------------

def own_validate(case: Case) -> None:
    """Notlardaki kurallar ve log sonuç; karesi alınan değişkenlerin en az üç farklı değeri (iki değerli bir değişkenin
    karesi kendisiyle ve sabitle tam ilişkilidir) ve karesel modelde tam doğrusal bağlantı olmaması."""

    validate(case)
    validate_positive(case)
    for variable in _squares(case):
        if case.data[variable].nunique() < 3:
            raise K.UploadError(f"“{case.name(variable)}” sütununun en az üç farklı değeri olmalı: karesel terim iki "
                                "farklı değerli bir değişkende değişkenin kendisiyle tam ilişkilidir.")
        if float(case.data[variable].abs().max()) > MAX_SQUARE:
            raise K.UploadError(f"“{case.name(variable)}” sütununda mutlak değeri 10.000'den büyük değerler var. Karesel "
                                "terim bu büyüklükte sayısal olarak kararsızdır; değişkeni dosyada daha büyük bir "
                                "birimle yazın (ör. TL yerine bin TL).")
    data = _log_data(case)
    columns = _m4(case)
    design = np.column_stack([np.ones(len(data)), data[list(columns)].to_numpy(dtype=float)])
    scaled = design / np.linalg.norm(design, axis=0)
    if np.linalg.matrix_rank(scaled, tol=1e-10) < design.shape[1]:
        raise K.UploadError("Karesel modelde açıklayıcılar arasında tam doğrusal bağlantı var (ör. bir değişken başka "
                            "bir değişkenin karesi). Bu sütunlardan birini ek değişkenlerden çıkarın.")


CUSTOM = custom_lab(
    build,
    sample_employed,
    ("Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sonuç ve temel açıklayıcı değişken zorunludur ve sayısal olmalıdır. "
     f"{POSITIVE_RULE} Doğrusal model sonucun logaritmasının temel açıklayıcı ve ek değişkenlere (en çok 3) göre "
     "regresyonudur; karesel terim ve merkezleme temel açıklayıcı ile ilk ek değişken içindir. "
     f"{ROW_RULE}"),
    roles((1, 2, 3, 4, 5, 6, 7, 8)),
    "Doğrusal modelin değişkenleridir; ilk ek değişkenin karesi de modele girer.",
    validate=own_validate,
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
