"""Konu 10 genel uygulaması: kukla değişkenler ve kategorik açıklayıcı değişkenler.

Notlardaki on bir adım (``core.labs.konu10``) aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde
yazılır: iki grup ortalaması ve basit kukla modeli, kontrollü kukla modeli, additif modelin paralel doğruları, kukla
katsayısı için çıkarım, çok kategorili değişken ve referans kategori, referansın değişmesi, kukla değişken tuzağı, log
modelinde tam yüzde fark, kategorilerin ortak testi, ikinci bir kategorik değişkenin kuklaları ve makale tablosu.

Alternatif örnekte veri BEAUTY'dir (Hamermesh ve Biddle, 1994; 1260 çalışan, saatlik ücret). Kukla değişken notlardaki
gibi kadın kuklasıdır; notlardaki dört bölgenin yerinde görünüşün üç kategorisi (ortalamanın altı: puan ≤ 2; ortalama;
ortalamanın üstü: puan ≥ 4; referans ortalama), altı endüstri kuklasının yerinde şehir büyüklüğünün iki kuklası (büyük
şehir, küçük şehir; referans diğer yerleşimler) vardır. Kontroller eğitim ve iş deneyimidir; kategorik değişkenlerin
modelleri ayrıca deneyimin karesini ve kadın kuklasını içerir. Görünüş bir görüşmecinin puanıdır ve rastgele
atanmamıştır: katsayılar koşullu farklardır, nedensel etki değildir.

"Kendi verini yükle" seçeneğinde iki kategorili değişken (bir kategori 1, diğeri 0) ve çok kategorili değişkenden (3–15
kategori) en az biri seçilir; notlardaki endüstri adımı için isteğe bağlı ikinci bir kategorik değişken seçilebilir.
Bir rolü seçilmeyen adımlar neye ihtiyaç duyduklarını yazar. Modeller notlardaki gibi log sonuçla kurulur (sonucun
bütün değerleri pozitif olmalı); kontroller temel açıklayıcı ve ek sayısal değişkenlerdir.

Etkileşim notlardaki gibidir: bağımlı değişken ve kodlama yönü (Adım 1), kontroller (Adım 2 ve 11), paralel doğruların
yatay ekseni (Adım 3), incelenen kukla (Adım 4 ve 8), referans kategori (Adım 5, 6 ve 9). Standart hatalar klasik EKK
standart hatalarıdır.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import cache

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from core.labs import expr as E
from core.labs import kendi_veri as K
from core.labs.ornek import Case, Role, TopicVariants, md, sayi, stable_checks, with_app_values
from core.labs.ornek_regresyon import (
    ACIKLAYICI,
    GOSTERGE,
    POSITIVE_RULE,
    ROW_RULE,
    SONUC,
    beauty_case,
    candidates,
    capital,
    custom_lab,
    digits_for,
    exact_multi,
    labels_of,
    listing,
    log_column,
    log_label,
    log_operations,
    p_text,
    phrase,
    roles,
    short_unit,
    validate,
    validate_positive,
)
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
    InlineData,
    JoinColumns,
    JointTest,
    LabSpec,
    LabStep,
    LineChart,
    MapCodes,
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

TOPIC = "konu10"
TITLE = "Uygulama: Kukla Değişkenler ve Kategorik Açıklayıcı Değişkenler"
KATEGORI, KATEGORI2 = "kategori", "kategori2"
GRID = 50


def _check(label: str, target, decimals: int) -> Check:
    return Check(label, target, 0.0, decimals)


def _scalar(name: str, label: str, decimals: int) -> Check:
    return _check(label, ScalarTarget(name), decimals)


def _cap(case: Case, text: str) -> str:
    """Etiketteki kategori adı: alternatif örnekte büyük harfle ("Kadın"); kendi verinde dosyadaki gibi (yalnız ilk
    harfi farklı iki kategori, ör. "evet" ve "Evet", aynı etikete inmesin)."""

    return text if case.own else capital(text)


def _name(case: Case, column: str) -> str:
    """Açıklama ve etiketlerde ad (arayüz kendi verinde kaçırır)."""

    return case.name(column) if case.own else phrase(case, column)


# --- Kukla ve kategoriler ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Dummy:
    """İki kategorili değişken: sütun (0/1), 1 ve 0 gruplarının adları (metin için kaçırılmış ``one``/``zero``, etiket
    için dosyadaki ``one_raw``/``zero_raw``) ve kuklanın başlığı."""

    column: str
    one: str
    zero: str
    one_raw: str
    zero_raw: str
    title: str


@dataclass(frozen=True)
class Groups:
    """Kategorik değişken: her kategori için (kukla sütunu, etiket, metindeki ad), kuklaları kuran işlemler (çerçeveye
    göre), varsayılan referans ve değişkenin adı."""

    items: tuple[tuple[str, str, str], ...]
    reference: str
    title: str
    text: str
    setup: object

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(code for code, _, _ in self.items)

    def label(self, code: str) -> str:
        return dict((item[0], item[1]) for item in self.items)[code]

    def words(self, code: str) -> str:
        return dict((item[0], item[2]) for item in self.items)[code]


def _code_column(case: Case) -> str:
    return K.code_name(f"{case.roles[GOSTERGE]}_01", set(case.data.columns))


def _own_dummy(case: Case) -> Dummy | None:
    if not case.own or not case.has(GOSTERGE):
        return None
    column = case.roles[GOSTERGE]
    one = case.levels[GOSTERGE]
    zero = next(item for item in case.orders[column] if item != one)
    return Dummy(_code_column(case), f"“{md(one)}”", f"“{md(zero)}”", one, zero, case.name(column))


def _dummies(case: Case) -> dict[str, Dummy]:
    """İncelenebilen kuklalar: alternatif örnekte kadın, evli, siyah ve sendika üyesi; kendi verinde iki kategorili
    değişken."""

    if "dummies" in case.extra:
        return {key: Dummy(key, one, zero, one, zero, title) for key, (one, zero, title) in case.extra["dummies"].items()}
    found = _own_dummy(case)
    return {found.column: found} if found is not None else {}


def _dummy(case: Case) -> Dummy | None:
    dummies = _dummies(case)
    return next(iter(dummies.values())) if dummies else None


def _dummy_setup(case: Case) -> tuple:
    """Kendi verinde iki kategorili sütunun 0/1 kodlaması (ana çerçevede, Adım 1'de bir kez)."""

    found = _own_dummy(case)
    if found is None:
        return ()
    column = case.roles[GOSTERGE]
    return (MapCodes(case.frame, column, found.column, ((found.zero_raw, 0), (found.one_raw, 1)),
                     f"Kukla değişken: {found.zero_raw} = 0, {found.one_raw} = 1"),)


def _groups(case: Case, which: str) -> Groups | None:
    """Kategorik değişkenin kuklaları: alternatif örnekte görünüş (Adım 5–9) ve şehir büyüklüğü (Adım 10); kendi
    verinde çok kategorili değişken ve isteğe bağlı ikinci kategorik değişken."""

    key = "groups" if which == KATEGORI else "groups2"
    if key in case.extra:
        items, reference, title, text, derived = case.extra[key]
        return Groups(tuple(items), reference, title, text, derived)
    if not case.has(which):
        return None
    column = case.roles[which]
    categories = case.orders[column]
    taken = set(case.data.columns) | ({_code_column(case)} if case.has(GOSTERGE) else set())
    items = []
    for category in categories:
        code = K.code_name(f"{column}_{category}", taken)
        taken.add(code)
        items.append((code, f"{case.name(column)}: {category}", f"“{md(category)}”"))
    return Groups(tuple(items), items[0][0], case.name(column), phrase(case, column), column)


def _group_setup(case: Case, groups: Groups, frame: str) -> tuple:
    """Kategorik değişkenin bütün kuklaları ``frame`` çerçevesinde: alternatif örnekte eksik kukla türetilir (notlardaki
    Kuzeydoğu gibi), kendi verinde her kategori 0/1 kodlanır."""

    if isinstance(groups.setup, tuple):  # (kukla, ifade, açıklama): türetilen referans kuklası
        code, expression, comment = groups.setup
        return (Derive(frame, code, expression, comment),)
    column = groups.setup
    categories = case.orders[column]
    return tuple(MapCodes(frame, column, code, tuple((category, 1 if category == own else 0) for category in categories),
                          f"Kukla: {case.name(column)} = {own}")
                 for (code, _, _), own in zip(groups.items, categories))


# --- Değişkenler ------------------------------------------------------------------------------------------------

def _controls(case: Case) -> tuple[str, ...]:
    """Kontrol değişkenleri: alternatif örnekte eğitim ve iş deneyimi; kendi verinde temel açıklayıcı ve ek değişkenler."""

    return tuple(case.extra.get("controls") or candidates(case))


def _control_options(case: Case) -> tuple[str, ...]:
    return tuple(case.extra.get("control_options") or candidates(case))


def _base(case: Case) -> tuple[str, ...]:
    """Kategorik değişken modellerinin ortak açıklayıcıları: alternatif örnekte eğitim, deneyim, deneyim² ve kadın;
    kendi verinde kontroller ve (varsa) kukla."""

    if "base" in case.extra:
        return tuple(case.extra["base"])
    dummy = _dummy(case)
    return (*_controls(case), *((dummy.column,) if dummy else ()))


def _ly(case: Case) -> str:
    return log_column(case, case.roles[SONUC])


def _log_phrase(case: Case) -> str:
    """Log sonucun Markdown metnindeki adı: "log saatlik ücret"; kendi verinde ln(“Y”)."""

    return f"ln({phrase(case, case.roles[SONUC])})" if case.own else phrase(case, _ly(case))


def _without(terms: tuple[str, ...], drop: str) -> tuple[str, ...]:
    return tuple(term for term in terms if term != drop)


def _data(case: Case) -> pd.DataFrame:
    """Metinler ve varsayılanlar için veri: log sonuç, kukla ve kategorilerin kuklaları (uygulama işlemlerle hesaplar)."""

    y = case.roles[SONUC]
    data = case.data.copy()
    data[_ly(case)] = np.log(data[y].astype(float))
    found = _own_dummy(case)
    if found is not None:
        data[found.column] = (data[case.roles[GOSTERGE]] == found.one_raw).astype(float)
    for which in (KATEGORI, KATEGORI2):
        groups = _groups(case, which)
        if groups is None:
            continue
        if isinstance(groups.setup, tuple):
            code, expression, _ = groups.setup
            data[code] = E.evaluate(expression, data)
        else:
            for (code, _, _), category in zip(groups.items, case.orders[groups.setup]):
                data[code] = (data[groups.setup] == category).astype(float)
    return data


def _fit(case: Case, outcome: str, terms):
    return smf.ols(f"{outcome} ~ {' + '.join(terms)}", data=_data(case)).fit()


def _exact(case: Case, outcome: str, terms) -> bool:
    return case.own and exact_multi(replace(case, data=_data(case)), outcome, tuple(terms))


def _unit(case: Case, column: str) -> str:
    unit = short_unit(case, column)
    return unit or "birim"


def _p(value: float) -> str:
    return p_text(value, 3)


def _percent(value: float) -> str:
    return "%" + sayi(value, 2)


def _formula(outcome: str, terms) -> str:
    return f"{outcome} ~ " + " + ".join(terms)


def _need(number: int, title: str, note: NoteRef, lead: str, what: str) -> LabStep:
    return LabStep(number=number, title=title, note=note, explanation=f"{lead} Bu adım için {what} seçilmelidir.")


_DUMMY_NEED = "dosyanızda iki kategorili bir sütun (ör. kadın/erkek, evet/hayır, 1/0)"
_GROUP_NEED = "dosyanızda çok kategorili bir sütun (3–15 kategori; ör. bölge, eğitim düzeyi)"
_SECOND_NEED = "ikinci bir kategorik sütun (ör. sektör, şehir büyüklüğü)"


def _stable(checks, exact: bool, fragile: tuple[str, ...] = ()) -> tuple[Check, ...]:
    kept = stable_checks(tuple(checks), exact)
    if not exact:
        return kept
    return tuple(check for check in kept if not (isinstance(check.target, ScalarTarget) and check.target.name in fragile)
                 and not (isinstance(check.target, TableTarget) and check.target.column in (
                     "sh", "t", "p", "alt", "ust", "Standart hata", "t", "p", "GA alt (%)", "GA üst (%)")))


# --- Adım 1: ham ortalamalar ve basit kukla modeli ------------------------------------------------------------------

def _step1(case: Case) -> LabStep:
    title = "İki grup ortalaması ve basit kukla modeli"
    note_ref = NoteRef("10.2", 0, ("Tablo 10.1", "Kod 10.1", "Kod 10.2", "Şekil 10.1"))
    lead = ("Kukla değişken iki kategoriyi 0 ve 1 ile ayırır. $Y = \\beta_0 + \\delta D + u$ modelinde "
            "$\\mathbb{E}(Y \\mid D = 0) = \\beta_0$ ve $\\mathbb{E}(Y \\mid D = 1) = \\beta_0 + \\delta$: sabit terim "
            "referans grubun ortalaması, kukla katsayısı iki grup ortalamasının farkıdır.")
    dummy = _dummy(case)
    if dummy is None:  # veri (ve log sonuç) yine bu adımda kurulur: sonraki adımlar onu kullanır
        return LabStep(number=1, title=title, note=note_ref,
                       operations=(*case.load, *log_operations(case, case.frame, (case.roles[SONUC],))),
                       explanation=f"{lead} Bu adım için {_DUMMY_NEED} seçilmelidir.")
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    ly = _ly(case)
    controls = _controls(case)
    outcomes = tuple(dict.fromkeys((y, ly, x, *_without(controls, x))))
    reverse = case.extra.get("reverse") or K.code_name(f"{dummy.column}_ters", set(case.data.columns) | {dummy.column})
    data = _data(case)
    summary = ((("n", y, "count"), (y, y, "mean")) + tuple((name, name, "mean") for name in controls))
    exact = _exact(case, y, (dummy.column,))
    outcome_control = Choice("adim1_y", "Bağımlı değişken", tuple((name, _axis(case, name)) for name in outcomes), y,
                             help=f"Varsayılan: {case.name(y)}.")
    coding = Choice("adim1_kod", "Kukla değişkenin kodlaması",
                    (("asil", f"1 = {dummy.one_raw}, 0 = {dummy.zero_raw}"),
                     ("ters", f"1 = {dummy.zero_raw}, 0 = {dummy.one_raw}")), "asil",
                    help="Kodlama ters çevrilirse katsayının işareti değişir; iki grubun ortalamaları değişmez.")

    def build(choices) -> tuple:
        outcome, code = choices["adim1_y"], choices["adim1_kod"]
        frame, column = case.frame, dummy.column
        one, zero = dummy.one_raw, dummy.zero_raw
        recode: tuple = ()
        if code == "ters":
            frame, column, one, zero = "kukla", reverse, dummy.zero_raw, dummy.one_raw
            recode = (CopyFrame("kukla", case.frame, "Ters kodlama için verinin kopyası (özgün veri değişmez)"),
                      Derive("kukla", reverse, E.sub(1, E.var(dummy.column)),
                             f"Ters kukla: {reverse} = 1 − {dummy.column}"))
        same = outcome == y and code == "asil"
        logs = log_operations(case, case.frame, (y,)) if case.own else ()
        return (
            *case.load,
            *_dummy_setup(case),
            *logs,
            GroupSummary(case.frame, dummy.column, summary, "tablo101", (0, 1), decimals=2,
                         labels=((0, _cap(case, dummy.zero_raw)), (1, _cap(case, dummy.one_raw))), heading="Grup",
                         title="İki grubun ham ortalamaları (Tablo 10.1'deki gibi)"),
            BarChart("tablo101", y, "Grup", f"Ortalama {case.name(y) if case.own else phrase(case, y)}",
                     f"İki grubun ham ortalamaları: {case.name(y) if case.own else phrase(case, y)} (Şekil 10.1'deki gibi)",
                     decimals=2),
            *recode,
            OLS("m_basit", frame, outcome, (column,), f"Basit kukla modeli: {_formula(outcome, (column,))}"),
            ShowModel("m_basit", "Basit kukla modelinin Python çıktısı (Kod 10.2'deki gibi)" if same
                      else "Basit kukla modelinin Python çıktısı (seçiminiz)", columns=("coef", "se", "t", "p"),
                      stats=("nobs", "r2"), decimals=(("se", 4), ("p", 4))),
            *(() if same else (
                OLS("m_basit_n", case.frame, y, (dummy.column,), f"Varsayılan model: {_formula(y, (dummy.column,))}"),
                RegressionTable(((f"Varsayılan: {_short(case, y)} ~ {dummy.one_raw}", "m_basit_n"),
                                 (f"Seçiminiz: {_short(case, outcome)} ~ {one}", "m_basit")),
                                (INTERCEPT, *dict.fromkeys((dummy.column, column))), "yan101",
                                "Varsayılan model ve seçtiğiniz model", decimals=4),
            )),
            ModelValue("b0", "m_basit", "coef", f"Sabit terim: {zero} grubunun ortalaması", term=INTERCEPT, decimals=4),
            ModelValue("delta", "m_basit", "coef", f"Kukla katsayısı: {one} − {zero} farkı", term=column, decimals=4),
            Statistic(frame, outcome, "mean", "ort_0", f"{_cap(case, zero)} grubunun örneklem ortalaması",
                      where=(column, 0), decimals=4),
            Statistic(frame, outcome, "mean", "ort_1", f"{_cap(case, one)} grubunun örneklem ortalaması",
                      where=(column, 1), decimals=4),
            Scalar("grup1_tahmin", E.add(E.ref("b0"), E.ref("delta")), f"{_cap(case, one)} grubu: β̂₀ + δ̂", decimals=4),
            Scalar("fark", E.sub(E.ref("ort_1"), E.ref("ort_0")), f"Ortalamalar farkı: {one} − {zero}", decimals=4),
        )

    def note(state, choices) -> str:
        outcome, code = choices["adim1_y"], choices["adim1_kod"]
        one, zero = (dummy.one, dummy.zero) if code == "asil" else (dummy.zero, dummy.one)
        s = state.scalars
        d = max(4, digits_for(max(abs(s["b0"]), abs(s["delta"])), 4))
        unit = f" ({short_unit(case, outcome)})" if outcome != ly and short_unit(case, outcome) else ""
        delta = sayi(s["delta"], d) if s["delta"] >= 0 else f"({sayi(s['delta'], d)})"
        return (f"Sabit terim {sayi(s['b0'], d)}, {zero} grubunun ortalama {phrase(case, outcome)} değeridir{unit}. Kukla "
                f"katsayısı {sayi(s['delta'], d)}, {one} ve {zero} ortalamaları arasındaki farktır: {sayi(s['ort_1'], d)} − "
                f"{sayi(s['ort_0'], d)} = {sayi(s['fark'], d)}. {capital(one)} grubunun tahmin edilen ortalaması sabit ile "
                f"katsayının toplamıdır: {sayi(s['b0'], d)} + {delta} = {sayi(s['grup1_tahmin'], d)}. Kodlama ters "
                "çevrilirse sabit diğer grubun ortalaması olur ve katsayının işareti değişir; iki grup arasındaki fark "
                "aynıdır. Bu fark nedensel bir etki değildir (§10.2).")

    checks = [
        _check(f"Tablo: {dummy.zero_raw}, gözlem sayısı", TableTarget("tablo101", _cap(case, dummy.zero_raw), "n"), 0),
        _check(f"Tablo: {dummy.one_raw}, gözlem sayısı", TableTarget("tablo101", _cap(case, dummy.one_raw), "n"), 0),
        *(_check(f"Tablo: {group}, {case.name(column)} ortalaması", TableTarget("tablo101", _cap(case, group), column), 2)
          for group in (dummy.zero_raw, dummy.one_raw) for _, column, _ in summary[1:]),
        *(_check(f"Model: {'sabit terim' if term == INTERCEPT else 'kukla'}, {label}", CoefTarget("m_basit", term, quantity),
                 3 if quantity == "t" else 4)
          for term in (INTERCEPT, dummy.column) for quantity, label in (("coef", "coef"), ("se", "std err"), ("t", "t"),
                                                                        ("p", "P>|t|"))),
        _check("Gözlem sayısı", ModelTarget("m_basit", "nobs"), 0),
        _check("R²", ModelTarget("m_basit", "r2"), 3),
        _scalar("ort_0", "0 grubunun ortalaması", 4),
        _scalar("ort_1", "1 grubunun ortalaması", 4),
        _scalar("fark", "Ortalamalar farkı", 4),
        _scalar("grup1_tahmin", "β̂₀ + δ̂", 4),
    ]
    del data
    return interactive_step(
        number=1,
        title=title,
        note=note_ref,
        explanation=(f"{lead} {case.extra.get('dummy_text', '')}Kukla: {dummy.one} = 1, {dummy.zero} = 0. Bağımlı değişkeni "
                     "ve kodlama yönünü değiştirin."),
        controls=(outcome_control, coding),
        build=build,
        checks=_stable(checks, exact),
        note_for=lambda state, choices: note(state, choices),
    )


def _short(case: Case, column: str) -> str:
    """Tablo başlığındaki ad: log sonuçta ln(…), diğerlerinde etiket (arayüz kendi verinde kaçırır)."""

    return log_label(case, case.roles[SONUC]) if column == _ly(case) and case.own else case.name(column)


def _axis(case: Case, column: str) -> str:
    if column == _ly(case):
        return log_label(case, case.roles[SONUC]) if case.own else case.name(column)
    unit = case.units.get(column, "")
    return f"{case.name(column)} ({unit})" if unit else case.name(column)


# --- Adım 2: kontrollü kukla modeli --------------------------------------------------------------------------------

def _step2(case: Case) -> LabStep:
    title = "Kontrol değişkenleri eklenince kukla katsayısı"
    note_ref = NoteRef("10.3", 0, ("Tablo 10.2",))
    lead = ("Ham grup farkı, grupların başka özellikler bakımından farklı olabileceğini dikkate almaz. Kontrollü modelde "
            "kukla katsayısı, kontrol değişkenleri aynı tutulduğunda iki grup arasındaki tahmin edilen ortalama farktır.")
    dummy = _dummy(case)
    if dummy is None:
        return _need(2, title, note_ref, lead, _DUMMY_NEED)
    y, ly = case.roles[SONUC], _ly(case)
    controls, choices_all = _controls(case), _control_options(case)
    column = dummy.column

    def models(terms: tuple[str, ...], suffix: str, prefix: str) -> tuple[list, list]:
        words = " + ".join(case.name(term) if case.own else phrase(case, term) for term in terms)
        level = case.name(y) if case.own else capital(phrase(case, y))
        rows = [(f"{prefix}{level} ~ {dummy.one_raw} + {words}", f"m_kontrol{suffix}", y),
                (f"{prefix}{log_label(case, y) if case.own else capital(phrase(case, ly))} ~ {dummy.one_raw} + {words}",
                 f"m_logk{suffix}", ly)]
        ops = [OLS(name, case.frame, outcome, (column, *terms), f"{label}: {_formula(outcome, (column, *terms))}")
               for label, name, outcome in rows]
        return ops, rows

    def build(choices) -> tuple:
        chosen = tuple(term for term in choices["adim2_x"] if term != column)
        same = chosen == controls
        operations: list = [OLS("m_ham", case.frame, y, (column,), f"Ham fark: {_formula(y, (column,))}")]
        rows = [(f"{case.name(y) if case.own else capital(phrase(case, y))} ~ {dummy.one_raw}", "m_ham", y)]
        if same:
            more, extra = models(chosen, "", "")
            operations += more
            rows += extra
        else:
            noted, noted_rows = models(controls, "_n", "Varsayılan: ")
            chose, chose_rows = models(chosen, "", "Seçiminiz: ")
            operations += noted + chose
            rows += [noted_rows[0], chose_rows[0], noted_rows[1], chose_rows[1]]
        values: list = []
        for index, (label, model, _) in enumerate(rows, start=1):
            values += [
                ModelValue(f"k2_{index}", model, "coef", f"{label}: kukla katsayısı", term=column, decimals=3),
                ModelValue(f"s2_{index}", model, "se", f"{label}: standart hata", term=column, decimals=3),
                ModelValue(f"t2_{index}", model, "t", f"{label}: t", term=column, decimals=3),
                ModelValue(f"r2_{index}", model, "r2", f"{label}: R²", decimals=3),
            ]
        tables = [ScalarTable(tuple((label, E.ref(f"{prefix}_{i}")) for i, (label, _, _) in enumerate(rows, start=1)),
                              result, decimals=3, heading="Model")
                  for prefix, result in (("k2", "k102"), ("s2", "s102"), ("t2", "t102"), ("r2", "r102"))]
        return (
            *operations,
            *values,
            *tables,
            JoinColumns("tablo102", (("Kukla katsayısı", "k102", "deger"), ("Standart hata", "s102", "deger"),
                                     ("t", "t102", "deger"), ("R²", "r102", "deger")), decimals=3, heading="Model",
                        title="Kukla katsayısı farklı modellerde (Tablo 10.2'deki gibi)" if same
                        else "Kukla katsayısı: varsayılan ve seçtiğiniz modeller"),
        )

    def note(state, choices) -> str:
        chosen = tuple(term for term in choices["adim2_x"] if term != column)
        s = state.scalars
        same = chosen == controls
        raw, level = s["k2_1"], s["k2_2"] if same else s["k2_3"]
        log = s["k2_3"] if same else s["k2_5"]
        unit = _unit(case, y)
        d = digits_for(max(abs(raw), abs(level)), 3)
        names = listing([phrase(case, term) for term in chosen]) if chosen else "hiçbir değişken"
        scale = float(_data(case)[y].std())
        if abs(level - raw) < 0.01 * scale:
            moved = ("Katsayı neredeyse değişmez: seçilen değişkenler iki grupta benzer ya da sonuçla zayıf ilişkilidir. ")
        else:
            moved = ("Katsayının değişmesi, ham farkın bir bölümünün bu değişkenlerdeki grup farklarıyla bağlantılı "
                     f"olduğunu gösterir; kontrollü fark {'negatiftir' if level < 0 else 'pozitiftir'}. ")
        return (f"Ham fark {sayi(raw, d)} {unit}; {names} sabit tutulunca kukla katsayısı {sayi(level, d)} {unit} olur. "
                f"{moved}Log sonuç modelinde katsayı {sayi(log, digits_for(log, 3))}: kontrol değişkenleri sabitken "
                "yaklaşık yüzde fark (tam dönüşüm Adım 8'de). Kontrol eklemek katsayıyı nedensel yapmaz; modelde olmayan "
                "etmenler farkın bir bölümünü açıklayabilir (§10.3)." + _exact_note(exact))

    labels = [f"{case.name(y) if case.own else capital(phrase(case, y))} ~ {dummy.one_raw}"]
    _, extra = models(controls, "", "")
    labels += [label for label, _, _ in extra]
    # Her satırın modeli ayrı: log sonuç modeli tam uyabilir, düzey modeli uymayabilir (ya da tersi)
    fragile_rows = {label for label, outcome, terms in ((labels[0], y, (column,)), (labels[1], y, (column, *controls)),
                                                        (labels[2], ly, (column, *controls)))
                    if _exact(case, outcome, terms)}
    exact = bool(fragile_rows)
    checks = [_check(f"Tablo: {label}, {column_name}", TableTarget("tablo102", label, column_name), 3)
              for label in labels for column_name in ("Kukla katsayısı", "Standart hata", "t", "R²")
              if not (label in fragile_rows and column_name in ("Standart hata", "t"))]
    return interactive_step(
        number=2,
        title=title,
        note=note_ref,
        explanation=f"{lead} Kontrol değişkenlerini değiştirin.",
        controls=(MultiChoice("adim2_x", "Kontrol değişkenleri", options_of(case, choices_all), controls,
                              help="Varsayılan: " + ", ".join(case.name(term) for term in controls) + "."),),
        build=build,
        checks=tuple(checks),
        note_for=lambda state, choices: note(state, choices),
    )


def options_of(case: Case, columns) -> tuple[tuple[str, str], ...]:
    return tuple((column, _axis(case, column)) for column in columns)


def _exact_note(exact: bool) -> str:
    return (" Model veriye neredeyse tam uyuyor (R² ≈ 1): standart hata, t ve p yuvarlama hatasına duyarlıdır ve "
            "indirilen kodda karşılaştırılmaz." if exact else "")


# --- Adım 3: additif model ve paralel doğrular -------------------------------------------------------------------

def _line_grid(case: Case, column: str) -> tuple:
    values = case.data[column].astype(float)
    low, high = float(values.min()), float(values.max())
    if bool(np.all(values == np.round(values))) and high - low <= 100:
        return ("support", int(low), int(high))
    points = np.linspace(low, high, GRID)
    return ("inline", tuple(float(round(value, 10)) for value in points))


def _step3(case: Case) -> LabStep:
    title = "Additif kukla modeli: paralel doğrular"
    note_ref = NoteRef("10.3", 0, ("Şekil 10.2",))
    lead = "Additif modelde kukla yalnız düzeyi kaydırır: bütün açıklayıcıların eğimi iki grupta aynıdır."
    dummy = _dummy(case)
    if dummy is None:
        return _need(3, title, note_ref, lead, _DUMMY_NEED)
    ly = _ly(case)
    controls = _controls(case)
    column = dummy.column
    fit = _fit(case, ly, (column, *controls))
    equation = sayi(float(fit.params[INTERCEPT]), 4)
    for term in (column, *controls):
        value = float(fit.params[term])
        name = f"`{term}`" if case.own else (dummy.title.lower() if term == column else phrase(case, term))
        equation += f" {'−' if value < 0 else '+'} {sayi(abs(value), max(4, digits_for(value, 4)))}·{name}"

    def build(choices) -> tuple:
        axis = choices["adim3_x"]
        others = [term for term in controls if term != axis]
        line = E.add(E.ref("b0_log"), E.mul(E.ref(f"b_{axis}"), E.var(axis)))
        for term in others:
            line = E.add(line, E.mul(E.ref(f"b_{term}"), E.ref(f"ort_{term}")))
        kind, *grid = _line_grid(case, axis)
        support = (Support("dogru", axis, grid[0], grid[1], f"{case.name(axis)} ızgarası: {grid[0]}, …, {grid[1]} "
                           "(örneklemin aralığı)") if kind == "support" else
                   InlineData("dogru", (axis,), tuple((value,) for value in grid[0]),
                              f"{case.name(axis)} ızgarası: örneklemin aralığında {GRID} nokta"))
        held = listing([_name(case, term) for term in others])
        return (
            OLS("m_log", case.frame, ly, (column, *controls), f"Kontrollü log sonuç modeli: {_formula(ly, (column, *controls))}"),
            ModelValue("b0_log", "m_log", "coef", "Sabit terim", term=INTERCEPT, decimals=4),
            ModelValue("d_log", "m_log", "coef", "Kukla katsayısı", term=column, decimals=4),
            *(ModelValue(f"b_{term}", "m_log", "coef", f"{capital(_name(case, term))} katsayısı", term=term,
                         decimals=max(4, digits_for(float(fit.params[term]), 4))) for term in controls),
            *(Statistic(case.frame, term, "mean", f"ort_{term}", f"{capital(_name(case, term))}: örneklem ortalaması",
                        decimals=4) for term in others),
            support,
            Derive("dogru", "grup0", line, f"{_cap(case, dummy.zero_raw)} ({column} = 0): sabit + eğim"
                   + (f"; {held} ortalamada" if others else "")),
            Derive("dogru", "grup1", E.add(E.var("grup0"), E.ref("d_log")),
                   f"{_cap(case, dummy.one_raw)} ({column} = 1): önceki doğru + δ̂"),
            LineChart("dogru", axis, "grup0", _axis(case, axis), "Tahmin edilen log sonuç",
                      f"Additif kukla modelinde ortak eğim ve farklı düzey: {case.name(axis)} ekseninde (Şekil 10.2'deki "
                      "gibi)", markers=False, series=(("grup1", _cap(case, dummy.one_raw)),),
                      legend=_cap(case, dummy.zero_raw)),
        )

    def note(state, choices) -> str:
        axis = choices["adim3_x"]
        s = state.scalars
        return (f"İki doğrunun eğimi aynıdır ({phrase(case, axis)} katsayısı "
                f"{sayi(s[f'b_{axis}'], max(4, digits_for(s[f'b_{axis}'], 4)))}); aralarındaki dikey uzaklık her düzeyde "
                f"kukla katsayısıdır ({sayi(s['d_log'], 4)}). Additif kukla modeli gruplara farklı sabit verir, farklı eğim "
                "vermez: doğrular hangi değişken yatay eksende olursa olsun paraleldir. Eğim farkı için etkileşim terimi "
                "gerekir (Konu 11). Log ölçekteki sabit fark, düzeyde sabit bir yüzde farka karşılık gelir (§10.3).")

    return interactive_step(
        number=3,
        title=title,
        note=note_ref,
        explanation=(f"{lead} Log sonuç modeli: ln(Y) = {equation} (katsayılar dört basamakla). Diğer kontroller "
                     "örneklem ortalamasında tutulur. Yatay eksendeki değişkeni değiştirin."),
        controls=(Choice("adim3_x", "Yatay eksendeki değişken", options_of(case, controls), controls[0],
                         help="Diğer kontroller örneklem ortalamasında tutulur (Şekil 10.2'deki gibi)."),),
        build=build,
        checks=(_scalar("b0_log", "Sabit terim", 4), _scalar("d_log", "Kukla katsayısı", 4),
                *(_scalar(f"b_{term}", f"{case.name(term)} katsayısı", max(4, digits_for(float(fit.params[term]), 4)))
                  for term in controls)),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 4: kukla katsayısı için çıkarım --------------------------------------------------------------------------

def _maybe_interactive(number: int, title: str, note: NoteRef, explanation: str, control: Choice, build, checks,
                       note_for) -> LabStep:
    """Tek seçenekli denetim (kendi verinde tek kukla) etkileşimli adım kurmaz; adım varsayılan işlemlerle kurulur."""

    if len(control.options) > 1:
        return interactive_step(number=number, title=title, note=note, explanation=explanation, controls=(control,),
                                build=build, checks=checks, note_for=note_for)
    return LabStep(number=number, title=title, note=note, explanation=explanation,
                   operations=tuple(build({control.key: control.default})), checks=checks,
                   note_for=lambda state, choices: note_for(state, {control.key: control.default}))


def _dummy_choice(case: Case, key: str, label: str) -> Choice:
    dummies = _dummies(case)
    default = next(iter(dummies))
    return Choice(key, label, tuple((code, item.title) for code, item in dummies.items()), default,
                  help=f"Varsayılan: {dummies[default].title}. Model: log sonuç ~ kukla + kontroller.")


def _step4(case: Case) -> LabStep:
    title = "Kukla katsayısı için istatistiksel çıkarım"
    note_ref = NoteRef("10.4", 0)
    lead = ("Kukla katsayısı da standart hata, t istatistiği, p-değeri ve güven aralığıyla değerlendirilir: "
            "$H_0: \\delta = 0$, $H_1: \\delta \\neq 0$. $H_0$ kontrol değişkenleri sabitken iki kategori arasında "
            "anakütlede ortalama fark bulunmadığını söyler.")
    dummies = _dummies(case)
    if not dummies:
        return _need(4, title, note_ref, lead, _DUMMY_NEED)
    ly = _ly(case)
    controls = _controls(case)
    control = _dummy_choice(case, "adim4_kukla", "Sınanan kukla değişken")
    default = control.default
    exact = _exact(case, ly, (default, *_without(controls, default)))

    def build(choices) -> tuple:
        chosen = choices["adim4_kukla"]
        terms = (chosen, *_without(controls, chosen))
        side: tuple = () if chosen == default else (
            OLS("m_cik_n", case.frame, ly, (default, *_without(controls, default)),
                f"Varsayılan model: {_formula(ly, (default, *_without(controls, default)))}"),
            CoefficientTable("m_cik_n", (default,), "cikarim_n", f"Varsayılan: {dummies[default].title} kuklası",
                             decimals=4, t_decimals=3),
        )
        return (
            *side,
            OLS("m_cik", case.frame, ly, terms, f"Kontrollü log sonuç modeli: {_formula(ly, terms)}"),
            CoefficientTable("m_cik", (chosen,), "cikarim", f"{dummies[chosen].title} kuklası: katsayı, standart hata, t, p "
                             "ve yüzde 95 güven aralığı" + ("" if chosen == default else " (seçiminiz)"), decimals=4,
                             t_decimals=3),
            ModelValue("t_cik", "m_cik", "t", f"{dummies[chosen].title} kuklasının t istatistiği", term=chosen,
                       decimals=3),
            ModelValue("p_cik", "m_cik", "p", "İki taraflı p-değeri", term=chosen, decimals=3),
            ModelValue("sd_cik", "m_cik", "df_resid", "Serbestlik derecesi n − k − 1", decimals=0),
            HypothesisPlot("t", "t_cik", "sd_cik", f"{dummies[chosen].title} kuklası: H₀: δ = 0 için t testi", "t değeri"),
        )

    def note(state, choices) -> str:
        chosen = choices["adim4_kukla"]
        item = dummies[chosen]
        s = state.scalars
        delta = float(state.tables["cikarim"].loc[chosen, "katsayi"])
        if exact:
            return (f"δ̂ = {sayi(delta, 4)}. Uyum tam olduğu için t ve p yuvarlama hatasıdır; karar yazılmaz (§10.4)."
                    + _exact_note(True))
        rejected = s["p_cik"] < 0.05
        held = listing([phrase(case, term) for term in _without(controls, chosen)])
        text = (f"δ̂ = {sayi(delta, 4)}, t = {sayi(s['t_cik'], 3)}, {_p(s['p_cik'])}: yüzde 5 düzeyinde H₀: δ = 0 "
                f"{'reddedilir' if rejected else 'reddedilemez'}. {capital(held)} sabitken {item.one} ve {item.zero} "
                f"grupları arasında anakütlede {_log_phrase(case)} farkı olmadığı hipotezi ")
        if rejected:
            return text + ("veriyle uyumsuzdur. Küçük bir p-değeri farkın nedenini açıklamaz ve eksik değişken olmadığını "
                           "göstermez; farkın büyüklüğü ve iktisadi önemi ayrıca değerlendirilir (§10.4).")
        return text + ("reddedilemez. Reddedememek farkın sıfır olduğunu kanıtlamaz: veri bu büyüklükteki bir farkı "
                       "sıfırdan ayırt edecek kadar kesin değildir; güven aralığı hem sıfırı hem de iktisadi olarak önemli "
                       "olabilecek değerleri içerebilir (§10.4).")

    checks = [_check(f"Tablo: kukla, {label}", TableTarget("cikarim", default, column), decimals)
              for column, label, decimals in (("katsayi", "katsayı", 4), ("sh", "standart hata", 4), ("t", "t", 3),
                                              ("p", "p", 3))]
    return _maybe_interactive(4, title, note_ref, f"{lead} Model: log sonuç ~ kukla + kontroller. Sınanan kuklayı "
                                                  "değiştirin." if len(dummies) > 1 else
                              f"{lead} Model: log sonuç ~ kukla + kontroller.", control, build,
                              _stable(checks, exact), note)


# --- Adım 5: kategorik değişken ve referans kategori ---------------------------------------------------------------

def _group_frame(case: Case, groups: Groups, frame: str) -> tuple:
    return (CopyFrame(frame, case.frame, f"{groups.title} kuklaları için verinin kopyası (özgün veri değişmez)"),
            *_group_setup(case, groups, frame))


def _group_table(groups: Groups, model: str, reference: str, prefix: str, title: str) -> tuple:
    """Bir referansla kategori modelinin log katsayıları, tam yüzde farkları, p-değerleri ve tam yüzde güven sınırları;
    ``prefix`` skaler ve tablo adlarının önekidir (seçilen model "", varsayılan model "n")."""

    others = tuple(code for code in groups.codes if code != reference)
    values: list = []
    for code in others:
        label = groups.label(code)
        values += [
            ModelValue(f"k{prefix}_{code}", model, "coef", f"{label}: log katsayı", term=code, decimals=3),
            ModelValue(f"p{prefix}_{code}", model, "p", f"{label}: p-değeri", term=code, decimals=3),
            ModelValue(f"alt{prefix}_{code}", model, "ci_low", f"{label}: yüzde 95 GA alt sınırı (log)", term=code,
                       decimals=4),
            ModelValue(f"ust{prefix}_{code}", model, "ci_high", f"{label}: yüzde 95 GA üst sınırı (log)", term=code,
                       decimals=4),
            Scalar(f"tam{prefix}_{code}", E.mul(100, E.sub(E.exp(E.ref(f"k{prefix}_{code}")), 1)),
                   f"{label}: tam yüzde fark 100·(exp(δ̂) − 1)", decimals=2, percent=True),
            Scalar(f"talt{prefix}_{code}", E.mul(100, E.sub(E.exp(E.ref(f"alt{prefix}_{code}")), 1)),
                   f"{label}: tam yüzde GA alt sınırı", decimals=2, percent=True),
            Scalar(f"tust{prefix}_{code}", E.mul(100, E.sub(E.exp(E.ref(f"ust{prefix}_{code}")), 1)),
                   f"{label}: tam yüzde GA üst sınırı", decimals=2, percent=True),
        ]
    tables = [ScalarTable(tuple((groups.label(code), E.ref(f"{name}{prefix}_{code}")) for code in others),
                          f"{result}{prefix}", decimals=3, heading="Kategori")
              for name, result in (("k", "k103"), ("tam", "tam103"), ("p", "p103"), ("talt", "talt103"),
                                   ("tust", "tust103"))]
    return (
        *values,
        *tables,
        JoinColumns(f"tablo103{prefix}", (("Log katsayı", f"k103{prefix}", "deger"),
                                          ("Tam yüzde fark", f"tam103{prefix}", "deger"),
                                          ("p", f"p103{prefix}", "deger"), ("GA alt (%)", f"talt103{prefix}", "deger"),
                                          ("GA üst (%)", f"tust103{prefix}", "deger")),
                    decimals=2, heading="Kategori", column_decimals=(("Log katsayı", 3),), p_columns=("p",), title=title),
    )


def _step5(case: Case) -> LabStep:
    title = "Çok kategorili değişken ve referans kategori"
    note_ref = NoteRef("10.5", 0, ("Tablo 10.3", "Şekil 10.3", "Kod 10.5"))
    lead = ("Kategorik bir değişken m kategori içeriyorsa sabit terimli modelde m − 1 kukla kullanılır; modele girmeyen "
            "kategori referanstır. Log katsayılar tam yüzde farka çevrilir: $100(e^{\\hat\\delta} - 1)$.")
    groups = _groups(case, KATEGORI)
    if groups is None:
        return _need(5, title, note_ref, lead, _GROUP_NEED)
    ly = _ly(case)
    base = _base(case)
    default = groups.reference
    exact = _exact(case, ly, (*base, *(code for code in groups.codes if code != default)))
    reference = Choice("adim5_ref", "Referans kategori", tuple((code, groups.label(code)) for code in groups.codes),
                       default, help=f"Varsayılan: {groups.label(default)}. Referans modele girmeyen kukladır.")

    def build(choices) -> tuple:
        chosen = choices["adim5_ref"]
        others = tuple(code for code in groups.codes if code != chosen)
        terms = (*base, *others)
        same = chosen == default
        noted: tuple = () if same else (
            OLS("m_bolge_n", "bolge", ly, (*base, *(code for code in groups.codes if code != default)),
                f"Varsayılan model, referans {groups.label(default)}"),
            *_group_table(groups, "m_bolge_n", default, "n", f"Varsayılan: referans {groups.label(default)}"),
        )
        return (
            *_group_frame(case, groups, "bolge"),
            *noted,
            OLS("m_bolge", "bolge", ly, terms, f"Kategori modeli, referans {groups.label(chosen)}: {_formula(ly, terms)}"),
            *_group_table(groups, "m_bolge", chosen, "", f"Referans {groups.label(chosen)}: kategori kuklaları (Tablo "
                          "10.3'teki gibi)" if same else f"Seçiminiz: referans {groups.label(chosen)}"),
            CoefficientPlot("m_bolge", others, f"Referansa ({groups.label(chosen)}) göre farklar ve yüzde 95 güven "
                            "aralıkları (Şekil 10.3'teki gibi)", f"Referansa ({groups.label(chosen)}) göre tam yüzde fark",
                            y_label="Kategori", percent=True,
                            labels=tuple((code, groups.label(code)) for code in groups.codes)),
        )

    def note(state, choices) -> str:
        chosen = choices["adim5_ref"]
        others = [code for code in groups.codes if code != chosen]
        s = state.scalars
        m = len(groups.codes)
        text = (f"Sabit terimli modelde {m} kategori için {m - 1} kukla kullanılır; referans kategori "
                f"{groups.words(chosen)} modelde yer almaz ve her satır bu referansa göre koşullu farktır. ")
        if exact:
            return text + "Uyum tam olduğu için p-değerleri ve güven aralıkları yuvarlama hatasıdır (§10.5)." + _exact_note(True)
        largest = max(others, key=lambda code: abs(s[f"tam_{code}"]))
        percent = s[f"tam_{largest}"]
        text += (f"Örneğin diğer değişkenler sabitken {groups.words(largest)} kategorisinde tahmin edilen "
                 f"{phrase(case, case.roles[SONUC])}, {groups.words(chosen)} kategorisine göre tam hesapla "
                 f"%{sayi(abs(percent), 2)} {'daha yüksektir' if percent > 0 else 'daha düşüktür'} "
                 f"({_p(s[f'p_{largest}'])}). ")
        significant = [groups.words(code) for code in others if s[f"p_{code}"] < 0.05]
        if significant:
            text += f"Yüzde 5 düzeyinde referanstan ayrışan kategori: {listing(significant)}. "
        else:
            text += "Güven aralıklarının hepsi sıfırı içerir: hiçbir kategori yüzde 5 düzeyinde referanstan ayrışmaz. "
        return text + ("Referans kategori ‘normal’ ya da ‘üstün’ kategori değildir; yalnız karşılaştırma tabanıdır. Kategori "
                       "katsayıları koşullu farklardır, nedensel etki değildir (§10.5).")

    others = [code for code in groups.codes if code != default]
    checks = [_check(f"Tablo: {groups.label(code)}, {column}", TableTarget("tablo103", groups.label(code), column),
                     decimals)
              for code in others for column, decimals in (("Log katsayı", 3), ("Tam yüzde fark", 2), ("p", 3),
                                                          ("GA alt (%)", 2), ("GA üst (%)", 2))]
    return interactive_step(
        number=5,
        title=title,
        note=note_ref,
        explanation=f"{lead} {case.extra.get('groups_text', '')}Referans kategoriyi değiştirin.",
        controls=(reference,),
        build=build,
        checks=_stable(checks, exact),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 6: referans kategori değişince ---------------------------------------------------------------------------

def _step6(case: Case) -> LabStep:
    title = "Referans kategori değişince ne olur?"
    note_ref = NoteRef("10.5", 0)
    lead = ("Referans kategori değişince sabit terim ve kukla katsayılarının sayısal değerleri ile karşılaştırılan grup "
            "değişir; tahmin edilen değerler, artıklar, R² ve modelin ortak uyumu değişmez.")
    groups = _groups(case, KATEGORI)
    if groups is None:
        return _need(6, title, note_ref, lead, _GROUP_NEED)
    ly = _ly(case)
    base = _base(case)
    default = groups.reference
    choices_6 = tuple(code for code in groups.codes if code != default)
    noise = 10 if not case.own else 8
    first_terms = (*base, *(code for code in groups.codes if code != default))

    def build(choices) -> tuple:
        chosen = choices["adim6_ref"]
        others = tuple(code for code in groups.codes if code != chosen)
        other = next(code for code in groups.codes if code not in (default, chosen))
        terms = (*base, *others)
        return (
            CopyFrame("bolge6", "bolge", "Referans karşılaştırması için kategori verisinin kopyası"),
            OLS("m_kd", "bolge6", ly, first_terms, f"Referans {groups.label(default)}: {_formula(ly, first_terms)}"),
            OLS("m_ref", "bolge6", ly, terms, f"Referans {groups.label(chosen)}: {_formula(ly, terms)}"),
            RegressionTable(((f"Referans: {groups.label(default)}", "m_kd"), (f"Referans: {groups.label(chosen)}", "m_ref")),
                            (INTERCEPT, *groups.codes), "tablo_ref", "İki referansla kategori katsayıları (boş hücre: o "
                            "modelin referans kategorisi; kategori dışı katsayılar aynı)",
                            stars=False, standard_errors=False, decimals=4, r2_decimals=6),
            ModelValue("k1_ref", "m_kd", "coef", f"Varsayılan modelde {groups.label(chosen)} katsayısı", term=chosen,
                       decimals=4),
            ModelValue("k1_diger", "m_kd", "coef", f"Varsayılan modelde {groups.label(other)} katsayısı", term=other,
                       decimals=4),
            ModelValue("k2_varsayilan", "m_ref", "coef", f"{groups.label(default)}: {groups.label(chosen)} referansına göre",
                       term=default, decimals=4),
            ModelValue("k2_diger", "m_ref", "coef", f"{groups.label(other)}: {groups.label(chosen)} referansına göre",
                       term=other, decimals=4),
            Scalar("fark6", E.sub(E.ref("k1_diger"), E.ref("k1_ref")),
                   f"Varsayılan katsayılardan {groups.label(other)} − {groups.label(chosen)} farkı", decimals=4),
            ModelValue("r2_kd", "m_kd", "r2", "Varsayılan referanslı model: R²", decimals=6),
            ModelValue("r2_ref", "m_ref", "r2", "Seçilen referanslı model: R²", decimals=6),
            Residuals("bolge6", "u_kd", "m_kd", "Varsayılan referanslı modelin artıkları"),
            Residuals("bolge6", "u_ref", "m_ref", "Seçilen referanslı modelin artıkları"),
            Derive("bolge6", "u_fark", E.absolute(E.sub(E.var("u_kd"), E.var("u_ref"))), "İki modelin artık farkı"),
            Statistic("bolge6", "u_fark", "max", "azami_fark", "Tahmin edilen değerlerin (ve artıkların) azami farkı",
                      decimals=noise),
        )

    def note(state, choices) -> str:
        chosen = choices["adim6_ref"]
        other = next(code for code in groups.codes if code not in (default, chosen))
        s = state.scalars
        base_value = s["k1_ref"]
        shown = sayi(base_value, 4) if base_value >= 0 else f"({sayi(base_value, 4)})"
        return (f"{capital(groups.words(chosen))} referans olunca {groups.words(default)} katsayısı "
                f"{sayi(s['k2_varsayilan'], 4)} olur: varsayılan modeldeki {groups.words(chosen)} katsayısının işaret "
                "değiştirmiş hâli. Diğer katsayılar da yeni referansa göre yazılır: varsayılan modelde "
                f"{groups.words(other)} − {groups.words(chosen)} farkı {sayi(s['k1_diger'], 4)} − {shown} = "
                f"{sayi(s['fark6'], 4)}; bu, yeni modeldeki {groups.words(other)} katsayısıdır "
                f"({sayi(s['k2_diger'], 4)}). İki modelin R²'si aynıdır ({sayi(s['r2_kd'], 6)}) ve tahmin edilen "
                "değerleri arasındaki en büyük fark yalnız bilgisayar yuvarlaması düzeyindedir. Referans değişikliği "
                "modeli değil, katsayıların hangi gruba göre yazıldığını değiştirir (§10.5).")

    return interactive_step(
        number=6,
        title=title,
        note=note_ref,
        explanation=(f"{lead} İlk model varsayılan referanslıdır ({groups.label(default)}). İkinci modelin referansını "
                     "değiştirin."),
        controls=(Choice("adim6_ref", "İkinci modelin referans kategorisi",
                         tuple((code, groups.label(code)) for code in choices_6), choices_6[0],
                         help=f"İlk model varsayılan referanslıdır ({groups.label(default)})."),),
        build=build,
        checks=(
            _scalar("k2_varsayilan", "Yeni referansa göre varsayılan kategorinin katsayısı", 4),
            _scalar("k1_ref", "Varsayılan modelde yeni referansın katsayısı", 4),
            _scalar("k1_diger", "Varsayılan modelde üçüncü kategorinin katsayısı", 4),
            _scalar("fark6", "Katsayı farkı", 4),
            _scalar("k2_diger", "Yeni referansa göre üçüncü kategorinin katsayısı", 4),
            _scalar("azami_fark", "Tahmin edilen değerlerin farkı (yuvarlama düzeyinde)", noise),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 7: kukla değişken tuzağı ---------------------------------------------------------------------------------

def _step7(case: Case) -> LabStep:
    title = "Kukla değişken tuzağı"
    note_ref = NoteRef("10.6", 0)
    lead = ("Her gözlem tam olarak bir kategoridedir: bütün kuklaların toplamı her gözlemde 1'dir. Sabit terim de her "
            "gözlemde 1 olan bir sütundur; sabit terim ile bütün kuklalar birlikte kullanılırsa tam çoklu doğrusal "
            "bağlantı ortaya çıkar. Çözüm: sabit terimi koruyup bir kategoriyi referans bırakmak ya da sabit terimi "
            "kaldırıp bütün kuklaları kullanmak.")
    groups = _groups(case, KATEGORI)
    if groups is None:
        return _need(7, title, note_ref, lead, _GROUP_NEED)
    total = E.var(groups.codes[0])
    for code in groups.codes[1:]:
        total = E.add(total, E.var(code))
    operations = (
        Derive("bolge", "toplam", total, "Bütün kategori kuklalarının toplamı"),
        Statistic("bolge", "toplam", "min", "toplam_min", "Kuklaların toplamı: en küçük değer", decimals=0),
        Statistic("bolge", "toplam", "max", "toplam_max", "Kuklaların toplamı: en büyük değer", decimals=0),
        *(Statistic("bolge", code, "sum", f"sayi_{code}", f"{groups.label(code)}: gözlem sayısı", decimals=0)
          for code in groups.codes),
        ScalarTable(tuple((groups.label(code), E.ref(f"sayi_{code}")) for code in groups.codes), "kategori_sayilari",
                    decimals=0, heading="Kategori", value="Gözlem sayısı"),
    )
    m = len(groups.codes)
    return LabStep(
        number=7,
        title=title,
        note=note_ref,
        explanation=lead,
        operations=operations,
        checks=(_scalar("toplam_min", "Kuklaların toplamı, en küçük değer", 0),
                _scalar("toplam_max", "Kuklaların toplamı, en büyük değer", 0),
                *(_scalar(f"sayi_{code}", f"{groups.label(code)}: gözlem sayısı", 0) for code in groups.codes)),
        takeaway=(f"{m} kuklanın toplamı en küçük ve en büyük değerde 1'dir: toplam her gözlemde sabit terimin sütununa "
                  f"eşittir. Bu yüzden sabit terimli modelde en fazla {m - 1} kategori kuklası kullanılabilir; {m} kukla "
                  "ve sabit birlikte tam çoklu doğrusal bağlantı yaratır ve katsayılar tek biçimde tahmin edilemez. "
                  "Yazılım bir kuklayı kendiliğinden düşürürse hangi kategorinin referans kaldığı kontrol edilir (§10.6)."),
    )


# --- Adım 8: log modelinde tam yüzde fark ----------------------------------------------------------------------------

def _percent_rows(model: str, dummy: str, prefix: str, title: str) -> tuple:
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


def _step8(case: Case) -> LabStep:
    title = "Log bağımlı değişkende kukla katsayısı: tam yüzde fark"
    note_ref = NoteRef("10.7", 0, ("Kod 10.3", "Kod 10.4"))
    lead = ("Kukla değişkende değişim 0'dan 1'e tam bir birimdir; katsayı küçük değilse $100\\hat\\delta$ yaklaşımı "
            "önemli hata yaratabilir. Tam yüzde fark $100(e^{\\hat\\delta} - 1)$ ile hesaplanır; güven aralığının "
            "sınırları da aynı dönüşümle yüzdeye çevrilir.")
    dummies = _dummies(case)
    if not dummies:
        return _need(8, title, note_ref, lead, _DUMMY_NEED)
    ly = _ly(case)
    controls = _controls(case)
    control = _dummy_choice(case, "adim8_kukla", "İncelenen kukla değişken")
    default = control.default
    exact = _exact(case, ly, (default, *_without(controls, default)))

    def build(choices) -> tuple:
        chosen = choices["adim8_kukla"]
        terms = (chosen, *_without(controls, chosen))
        same = chosen == default
        side: tuple = () if same else (
            OLS("m_yuzde_n", case.frame, ly, (default, *_without(controls, default)),
                f"Varsayılan model: {_formula(ly, (default, *_without(controls, default)))}"),
            *_percent_rows("m_yuzde_n", default, "n", dummies[default].title),
            *(ScalarTable(((f"Varsayılan: {dummies[default].title}", E.ref(f"{name}n_{part}")),
                           (f"Seçiminiz: {dummies[chosen].title}", E.ref(f"{name}_{part}"))),
                          f"yuzde_{column}", decimals=2, heading="Kukla")
              for column, name, part in (("d", "d", "yuzde"), ("yak", "yaklasik", "d"), ("tam", "tam", "d"),
                                         ("alt", "tam", "alt"), ("ust", "tam", "ust"))),
            JoinColumns("yuzde_karsilastirma", (("δ̂", "yuzde_d", "deger"), ("Yaklaşık (%)", "yuzde_yak", "deger"),
                                                ("Tam (%)", "yuzde_tam", "deger"), ("GA alt (%)", "yuzde_alt", "deger"),
                                                ("GA üst (%)", "yuzde_ust", "deger")),
                        decimals=2, heading="Kukla", column_decimals=(("δ̂", 4),),
                        title="Varsayılan ve seçtiğiniz kukla: yaklaşık ve tam yüzde fark"),
        )
        return (
            OLS("m_yuzde", case.frame, ly, terms, f"Kontrollü log sonuç modeli: {_formula(ly, terms)}"),
            ShowModel("m_yuzde", "Kontrollü log sonuç modelinin Python çıktısı (Kod 10.4'teki gibi)" if same
                      else "Kontrollü log sonuç modelinin Python çıktısı (seçiminiz)", columns=("coef", "se", "t", "p"),
                      stats=("nobs", "r2"), decimals=(("se", 4), ("p", 4))),
            *_percent_rows("m_yuzde", chosen, "", dummies[chosen].title),
            *side,
            Scalar("ornek_30", E.mul(100, E.sub(E.exp(-0.30), 1)), "δ = −0,30 için tam yüzde fark", decimals=1,
                   percent=True),
        )

    def note(state, choices) -> str:
        chosen = choices["adim8_kukla"]
        item = dummies[chosen]
        s = state.scalars
        held = listing([phrase(case, term) for term in _without(controls, chosen)])
        text = (f"δ̂ = {sayi(s['d_yuzde'], 4)}: yaklaşık yorum %{sayi(s['yaklasik_d'], 2)}, tam yorum %{sayi(s['tam_d'], 2)}. "
                f"{capital(held)} sabitken {item.one} grubunda tahmin edilen {phrase(case, case.roles[SONUC])}, "
                f"{item.zero} grubuna göre tam hesapla %{sayi(abs(s['tam_d']), 2)} "
                f"{'daha yüksektir' if s['tam_d'] > 0 else 'daha düşüktür'}. ")
        if not exact:
            text += (f"Yüzde 95 güven aralığının tam yüzde biçimi [%{sayi(s['tam_alt'], 2)}; %{sayi(s['tam_ust'], 2)}]"
                     + ("; aralık sıfırı içerir, bu fark yüzde 5 düzeyinde istatistiksel olarak anlamlı değildir. "
                        if s["tam_alt"] < 0 < s["tam_ust"] else ". "))
        return text + ("Kukla 0'dan 1'e tam bir birim değiştiği için katsayı büyükse 100·δ̂ yaklaşımı belirgin hata verir. "
                       "‘Yüzde’ fark ‘yüzde puan’ farkı değildir (§10.7)." + _exact_note(exact))

    terms = (default, *_without(controls, default))
    checks = [
        *(_check(f"Model: {'sabit terim' if term == INTERCEPT else case.name(term) if term in case.data.columns else term}, "
                 f"{label}", CoefTarget("m_yuzde", term, quantity), 3 if quantity == "t" else 4)
          for term in (INTERCEPT, *terms) for quantity, label in (("coef", "coef"), ("se", "std err"), ("t", "t"),
                                                                  ("p", "P>|t|"))),
        _check("Gözlem sayısı", ModelTarget("m_yuzde", "nobs"), 0),
        _check("R²", ModelTarget("m_yuzde", "r2"), 3),
        _scalar("yaklasik_d", "Yaklaşık yüzde", 2),
        _scalar("tam_d", "Tam yüzde", 2),
        _scalar("tam_alt", "Tam yüzde GA alt sınırı", 2),
        _scalar("tam_ust", "Tam yüzde GA üst sınırı", 2),
        _scalar("ornek_30", "δ = −0,30 için tam yüzde fark", 1),
    ]
    return _maybe_interactive(8, title, note_ref, f"{lead} İncelenen kuklayı değiştirin." if len(dummies) > 1 else lead,
                              control, build, _stable(checks, exact, ("tam_alt", "tam_ust")), note)


# --- Adım 9: kategorilerin ortak testi -------------------------------------------------------------------------------

def _step9(case: Case) -> LabStep:
    title = "Kategori kuklalarının ortak testi"
    note_ref = NoteRef("10.8", 0)
    lead = ("Kategorik değişkenin bütünüyle modele katkı sağlayıp sağlamadığı ortak F testiyle sınanır: referans dışındaki "
            "kategorilerin katsayıları birlikte sıfır mı? Tek tek t testleri kategorilerin referanstan farkını sınar; "
            "ortak test kategorik değişkenin bütününü.")
    groups = _groups(case, KATEGORI)
    if groups is None:
        return _need(9, title, note_ref, lead, _GROUP_NEED)
    ly = _ly(case)
    base = _base(case)
    default = groups.reference
    exact = _exact(case, ly, (*base, *(code for code in groups.codes if code != default)))
    q = len(groups.codes) - 1
    df = len(case.data) - len(base) - q - 1

    def build(choices) -> tuple:
        chosen = choices["adim9_ref"]
        others = tuple(code for code in groups.codes if code != chosen)
        return (
            OLS("m_ortak", "bolge", ly, (*base, *others), f"Kategori modeli, referans {groups.label(chosen)}"),
            JointTest("F_bolge", "p_bolge", "m_ortak", others, f"{q} kategori kuklasının ortak testi", decimals=3),
            ModelValue("sd_bolge", "m_ortak", "df_resid", "Payda serbestlik derecesi n − k − 1", decimals=0),
            Scalar("q_bolge", E.const(len(others)), "Kısıt sayısı q", decimals=0),
            HypothesisPlot("f", "F_bolge", "q_bolge", f"Kategori kuklalarının ortak testi: F({q}, {df})", "F değeri",
                           alternative="sag", df2="sd_bolge"),
        )

    def note(state, choices) -> str:
        chosen = choices["adim9_ref"]
        s = state.scalars
        if exact:
            return ("Uyum tam olduğu için F yuvarlama hatasıdır ve karar yazılmaz. Referansı değiştirin: ortak test "
                    "kategorik değişkenin bütününü sınar (§10.8)." + _exact_note(True))
        rejected = s["p_bolge"] < 0.05
        return (f"Referans {groups.words(chosen)} iken F({q}, {sayi(s['sd_bolge'], 0)}) = {sayi(s['F_bolge'], 3)}, "
                f"{_p(s['p_bolge'])}: yüzde 5 düzeyinde kategori kuklalarının birlikte sıfır olduğu hipotezi "
                f"{'reddedilir' if rejected else 'reddedilemez'}. Referansı değiştirin: F değişmez, çünkü ortak test "
                "kategorik değişkenin bütününü sınar ve referans değişikliği modelin uyumunu değiştirmez. Tek tek t "
                "testleri ise referansa bağlıdır (§10.8).")

    checks = (_scalar("F_bolge", "Ortak F", 3), _scalar("p_bolge", "Ortak p-değeri", 3),
              _scalar("sd_bolge", "Payda serbestlik derecesi", 0), _scalar("q_bolge", "Kısıt sayısı", 0))
    return interactive_step(
        number=9,
        title=title,
        note=note_ref,
        explanation=f"{lead} Referans kategoriyi değiştirin: F değişir mi?",
        controls=(Choice("adim9_ref", "Referans kategori", tuple((code, groups.label(code)) for code in groups.codes),
                         default, help="Ortak test hangi kategori referans olursa olsun aynıdır."),),
        build=build,
        checks=_stable(checks, exact, ("F_bolge", "p_bolge")),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 10: ikinci kategorik değişken ------------------------------------------------------------------------------

def _step10(case: Case) -> LabStep:
    title = "İkinci bir kategorik değişken: ortak test ve tam yüzde farklar"
    note_ref = NoteRef("10.8", 0, ("Tablo 10.4",))
    lead = ("İkinci bir kategorik değişkenin kuklaları birlikte sınanır; referans modele girmeyen kategoridir. Tek tek "
            "katsayılar referansa göre log farklardır ve tam yüzde farka çevrilir: $100(e^{\\hat\\delta} - 1)$.")
    groups = _groups(case, KATEGORI2)
    if groups is None:
        return _need(10, title, note_ref, lead, _SECOND_NEED)
    ly = _ly(case)
    base = _base(case)
    first = _groups(case, KATEGORI)
    others = tuple(code for code in groups.codes if code != groups.reference)
    exact = _exact(case, ly, (*base, *others))
    values: list = []
    for code in others:
        label = groups.label(code)
        values += [
            ModelValue(f"k_{code}", "m_end", "coef", f"{label}: log katsayı", term=code, decimals=3),
            ModelValue(f"p_{code}", "m_end", "p", f"{label}: p-değeri", term=code, decimals=3),
            Scalar(f"tam_{code}", E.mul(100, E.sub(E.exp(E.ref(f"k_{code}")), 1)),
                   f"{label}: tam yüzde fark 100·(exp(δ̂) − 1)", decimals=2, percent=True),
        ]
    tables = [ScalarTable(tuple((groups.label(code), E.ref(f"{prefix}_{code}")) for code in others), result,
                          decimals=3, heading="Kategori")
              for prefix, result in (("k", "k104"), ("tam", "tam104"), ("p", "p104"))]
    frame = "sehir"
    operations = (
        *_group_frame(case, groups, frame),
        OLS("m_end", frame, ly, (*base, *others), f"İkinci kategorik değişkenin modeli: {_formula(ly, (*base, *others))}"),
        JointTest("F_end", "p_end", "m_end", others, f"{len(others)} kuklanın ortak testi", decimals=3),
        ModelValue("sd_end", "m_end", "df_resid", "Payda serbestlik derecesi n − k − 1", decimals=0),
        *values,
        *tables,
        JoinColumns("tablo104", (("Log katsayı", "k104", "deger"), ("Tam yüzde fark", "tam104", "deger"),
                                 ("p", "p104", "deger")), decimals=2, heading="Kategori",
                    column_decimals=(("Log katsayı", 3),), p_columns=("p",),
                    title=f"{groups.title}: kuklalar referansa ({groups.label(groups.reference)}) göre (Tablo 10.4'teki "
                          "gibi)"),
        Scalar("q_end", E.const(len(others)), "Kısıt sayısı q (pay serbestlik derecesi)", decimals=0),
    )

    def note(state) -> str:
        s = state.scalars
        q = len(others)
        if exact:
            return ("Uyum tam olduğu için F ve p-değerleri yuvarlama hatasıdır; kategorilerin katsayıları okunur (§10.8)."
                    + _exact_note(True))
        rejected = s["p_end"] < 0.05
        together = "kuklaları birlikte" if q > 1 else "kuklası"
        text = (f"{capital(groups.text)} {together} F({q}, {sayi(s['sd_end'], 0)}) = {sayi(s['F_end'], 3)}, "
                f"{_p(s['p_end'])}: yüzde 5 düzeyinde{' birlikte' if q > 1 else ''} "
                f"{'anlamlıdır' if rejected else 'anlamlı değildir'}. ")
        if first is not None and "p_bolge" in s:
            other = s["p_bolge"] < 0.05
            text += (f"{capital(first.text)} kuklaları {'da' if other == rejected else 'ise'} yüzde 5 düzeyinde birlikte "
                     f"{'anlamlıdır' if other else 'anlamlı değildir'} (Adım 9). ")
        largest = max(others, key=lambda code: abs(s[f"tam_{code}"]))
        text += (f"Diğer değişkenler sabitken {groups.words(largest)} kategorisinin referansa "
                 f"({groups.words(groups.reference)}) göre tam yüzde farkı yaklaşık %{sayi(s[f'tam_{largest}'], 2)}. ")
        loose = [groups.words(code) for code in others if s[f"p_{code}"] >= 0.05]
        if loose and rejected:
            return text + (f"Bazı tekil katsayılar ({listing(loose)}) yüzde 5 düzeyinde anlamlı olmasa da kategorik "
                           "değişken bütünüyle anlamlı olabilir; tek tek katsayılar ile ortak test farklı soruları "
                           "yanıtlar (§10.8).")
        return text + "Tek tek katsayılar ile ortak test farklı soruları yanıtlar (§10.8)."

    checks = (
        _scalar("F_end", "Ortak F", 3),
        _scalar("sd_end", "Payda serbestlik derecesi", 0),
        _scalar("q_end", "Kısıt sayısı", 0),
        _scalar("p_end", "Ortak p-değeri", 3),
        *(_check(f"Tablo: {groups.label(code)}, {column}", TableTarget("tablo104", groups.label(code), column), decimals)
          for code in others for column, decimals in (("Log katsayı", 3), ("Tam yüzde fark", 2), ("p", 3))),
    )
    return LabStep(
        number=10,
        title=title,
        note=note_ref,
        explanation=f"{lead} {case.extra.get('groups2_text', '')}",
        operations=operations,
        checks=_stable(checks, exact, ("F_end", "p_end")),
        note_for=lambda state, choices: note(state),
    )


# --- Adım 11: makale tablosu --------------------------------------------------------------------------------------

def _step11(case: Case) -> LabStep:
    title = "Makale tablosunda kukla değişken"
    note_ref = NoteRef("10.9", 0, ("Tablo 10.5",))
    lead = ("Makale tablosunu okurken önce tablonun sözlüğü okunur: bağımlı değişken düzey mi log mu, hangi kategori "
            "referans, parantez içindeki sayı standart hata mı t istatistiği mi, yıldızların eşikleri ne, hangi kontroller "
            "var. Sütun (1) ham log sonuç farkı, Sütun (2) kontrollü farktır.")
    dummy = _dummy(case)
    if dummy is None:
        return _need(11, title, note_ref, lead, _DUMMY_NEED)
    ly = _ly(case)
    controls, choices_all = _controls(case), _control_options(case)
    column = dummy.column
    exact = _exact(case, ly, (column, *controls))

    def build(choices) -> tuple:
        chosen = tuple(term for term in choices["adim11_x"] if term != column)
        same = chosen == controls
        if same:
            models: tuple = (("(1)", "m_s1"), ("(2)", "m_s2"))
            extra: tuple = ()
            terms = (column, *controls)
        else:
            models = (("(1)", "m_s1"), ("(2) Varsayılan", "m_s2n"), ("(2) Seçiminiz", "m_s2"))
            extra = (OLS("m_s2n", case.frame, ly, (column, *controls),
                         f"Varsayılan Sütun (2): {_formula(ly, (column, *controls))}"),)
            terms = (column, *(term for term in choices_all if term in chosen or term in controls))
        return (
            OLS("m_s1", case.frame, ly, (column,), f"Sütun (1): {_formula(ly, (column,))}"),
            *extra,
            OLS("m_s2", case.frame, ly, (column, *chosen), f"Sütun (2): {_formula(ly, (column, *chosen))}"),
            RegressionTable(models, terms, "tablo105",
                            ("Kukla değişken içeren makale tablosu (Tablo 10.5'teki gibi)" if same
                             else "Kukla değişken içeren makale tablosu (varsayılan ve seçtiğiniz Sütun (2))")
                            + f" · bağımlı değişken: {log_label(case, case.roles[SONUC])}", decimals=3),
        )

    def note(state, choices) -> str:
        table = state.tables["tablo105"]
        chosen = tuple(term for term in choices["adim11_x"] if term != column)
        heading = "(2)" if chosen == controls else "(2) Seçiminiz"
        first, second = float(table.loc[column, "(1)"]), float(table.loc[column, heading])
        return (f"Sütun (1)'de kukla katsayısı {sayi(first, 3)}: kontrol yokken ham log sonuç farkı; tam yüzde "
                f"%{sayi(100 * (math.exp(first) - 1), 1)}. Sütun (2)'de katsayı {sayi(second, 3)}: kontrol değişkenleri "
                "sabitken koşullu fark. Katsayının sütunlar arasında değişmesi modelin cevapladığı sorunun değiştiğini "
                f"gösterir. Referans grup {dummy.zero} grubudur (kukla 0); parantez içindeki sayılar standart hatalardır "
                "(§10.9)." + _exact_note(exact))

    checks = []
    for heading, terms in (("(1)", (column,)), ("(2)", (column, *controls))):
        for term in terms:
            checks += [_check(f"Tablo: {heading}, {case.name(term) if term in case.data.columns else term}",
                              TableTarget("tablo105", term, heading), 3),
                       _check(f"Tablo: {heading}, {case.name(term) if term in case.data.columns else term} (SH)",
                              TableTarget("tablo105", f"{term}_sh", heading), 3)]
        checks += [_check(f"Tablo: {heading}, gözlem sayısı", TableTarget("tablo105", "n", heading), 0),
                   _check(f"Tablo: {heading}, R²", TableTarget("tablo105", "r2", heading), 3)]
    return interactive_step(
        number=11,
        title=title,
        note=note_ref,
        explanation=f"{lead} Sütun (2)'nin kontrollerini değiştirin.",
        controls=(MultiChoice("adim11_x", "Sütun (2)'deki kontrol değişkenleri", options_of(case, choices_all), controls,
                              help="Varsayılan: " + ", ".join(case.name(term) for term in controls) + "."),),
        build=build,
        checks=stable_checks(tuple(checks), exact, table="tablo105"),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Tanım ------------------------------------------------------------------------------------------------------

def build(case: Case) -> LabSpec:
    """Konu 10 uygulamasını verilen veriyle kurar; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    y = case.roles[SONUC]
    labels = labels_of(case)
    labels[_ly(case)] = labels.get(_ly(case)) or log_label(case, y)
    labels[INTERCEPT] = "Sabit terim"
    dummy = _dummy(case)
    if dummy is not None:
        labels.setdefault(dummy.column, f"{dummy.title} (1 = {dummy.one_raw}, 0 = {dummy.zero_raw})")
        reverse = case.extra.get("reverse") or K.code_name(f"{dummy.column}_ters", set(case.data.columns) | {dummy.column})
        labels.setdefault(reverse, f"{_cap(case, dummy.zero_raw)} (1/0 gösterge)")
        labels.update({"grup0": f"{_cap(case, dummy.zero_raw)}: tahmin edilen log sonuç",
                       "grup1": f"{_cap(case, dummy.one_raw)}: tahmin edilen log sonuç"})
    for which in (KATEGORI, KATEGORI2):
        groups = _groups(case, which)
        if groups is not None:
            for code, label, _ in groups.items:
                labels.setdefault(code, label)
    labels.update({"n": "Gözlem sayısı", "toplam": "Kategori kuklalarının toplamı"})
    spec = LabSpec(
        topic_key=TOPIC,
        title=str(case.extra.get("title", TITLE)),
        note_section="10",
        steps=(_step1(case), _step2(case), _step3(case), _step4(case), _step5(case), _step6(case), _step7(case),
               _step8(case), _step9(case), _step10(case), _step11(case)),
        labels=tuple(labels.items()),
        source=case.source,
    )
    return with_app_values(spec)


# --- Alternatif örnek ----------------------------------------------------------------------------------------

LOOKS = (("belavg", "Ortalamanın altında", "ortalamanın altında"), ("avglooks", "Ortalama", "ortalama"),
         ("abvavg", "Ortalamanın üstünde", "ortalamanın üstünde"))
CITY = (("bigcity", "Büyük şehir", "büyük şehir"), ("smllcity", "Küçük şehir", "küçük şehir"),
        ("othcity", "Diğer yerleşim", "diğer yerleşim"))


def alternative_case() -> Case:
    return beauty_case(
        extras=("exper", "expersq", "goodhlth"),
        title="Uygulama: Kukla Değişkenler ve Kategorik Açıklayıcı Değişkenler (BEAUTY)",
        controls=("educ", "exper"),
        control_options=("educ", "exper", "expersq", "goodhlth"),
        base=("educ", "exper", "expersq", "female"),
        dummies={"female": ("kadın", "erkek", "Kadın"), "married": ("evli", "evli olmayan", "Evli"),
                 "black": ("siyah", "siyah olmayan", "Siyah"), "union": ("sendika üyesi", "sendika üyesi olmayan",
                                                                         "Sendika üyesi")},
        reverse="male",
        groups=(LOOKS, "avglooks", "Görünüş", "görünüş",
                ("avglooks", E.sub(E.sub(1, E.var("belavg")), E.var("abvavg")),
                 "Görünüşü ortalama (puan 3): 1 − belavg − abvavg")),
        groups2=(CITY, "othcity", "Şehir büyüklüğü", "şehir büyüklüğü",
                 ("othcity", E.sub(E.sub(1, E.var("bigcity")), E.var("smllcity")),
                  "Büyük ya da küçük şehirde yaşamıyor: 1 − bigcity − smllcity")),
        dummy_text="Veri BEAUTY'dir (1260 çalışan, saatlik ücret, dolar). ",
        groups_text=("Görünüş bir görüşmecinin 1–5 puanıdır; üç kategori ortalamanın altı (puan ≤ 2), ortalama (puan 3) ve "
                     "ortalamanın üstüdür (puan ≥ 4). Ortalama kategorisinin göstergesi diğer iki kukladan kurulur. Model "
                     "eğitim, deneyim, deneyim² ve kadın kuklasını içerir. "),
        groups2_text=("Şehir büyüklüğünün iki kuklası büyük ve küçük şehirdir; referans diğer yerleşimlerdir (göstergesi iki "
                      "kukladan kurulur)."),
    )


@cache
def alternative() -> LabSpec:
    return build(alternative_case())


STORY = ("Veri BEAUTY'dir (Hamermesh ve Biddle, 1994; 1260 çalışan, saatlik ücret). Kukla değişken kadın kuklasıdır; "
         "çok kategorili değişken görünüştür (ortalamanın altı, ortalama, üstü), ikinci kategorik değişken şehir "
         "büyüklüğüdür. Görünüş rastgele atanmamıştır: katsayılar koşullu farklardır.")


# --- Kendi verin ---------------------------------------------------------------------------------------------

def own_validate(case: Case) -> None:
    """Notlardaki kurallar, log sonuç; iki kategorili ve çok kategorili değişkenden en az biri; her kategoride en az iki
    gözlem; kategori modellerinde tam doğrusal bağlantı olmaması."""

    validate(case)
    validate_positive(case)
    if not (case.has(GOSTERGE) or case.has(KATEGORI)):
        raise K.UploadError("İki kategorili değişken ve çok kategorili değişkenden en az birini seçin.")
    for role in (GOSTERGE, KATEGORI, KATEGORI2):
        if case.has(role) and case.data[case.roles[role]].value_counts().min() < 2:
            raise K.UploadError(f"“{case.name(case.roles[role])}” sütununun her kategorisinde en az iki gözlem olmalı.")
    data = _data(case)
    base = _base(case)
    for which in (KATEGORI, KATEGORI2):
        groups = _groups(case, which)
        if groups is None:
            continue
        terms = (*base, *(code for code in groups.codes if code != groups.reference))
        design = np.column_stack([np.ones(len(data)), data[list(terms)].to_numpy(dtype=float)])
        scaled = design / np.linalg.norm(design, axis=0)
        if len(data) < design.shape[1] + 1 or np.linalg.matrix_rank(scaled, tol=1e-10) < design.shape[1]:
            raise K.UploadError(f"“{groups.title}” kategorileriyle kurulan modelde tam doğrusal bağlantı var ya da gözlem "
                                "yetersiz (ör. iki kategorili değişken bu kategorilerin birleşimi). Başka bir sütun seçin.")


ROLES = roles((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), (
    Role(GOSTERGE, "İki kategorili değişken", "kategorik", False, (1, 2, 3, 4, 8, 11),
         "Tam iki kategorili sütun (ör. kadın/erkek, evet/hayır): bir kategori 1, diğeri 0 kodlanır. Her gözlemde dolu "
         "olmalı.", levels=(2, 2), pick="1 ile kodlanan grup", suggest=True, complete=True),
    Role(KATEGORI, "Çok kategorili değişken", "kategorik", False, (5, 6, 7, 9),
         "3–15 kategorili sütun (ör. bölge, eğitim düzeyi): her kategori için bir kukla kurulur, ilki referanstır. Her "
         "gözlemde dolu olmalı.", levels=(3, 15), suggest=True, complete=True),
    Role(KATEGORI2, "İkinci kategorik değişken", "kategorik", False, (10,),
         "2–15 kategorili ikinci bir sütun (ör. sektör): kuklaları birlikte sınanır. Her gözlemde dolu olmalı.",
         levels=(2, 15), complete=True),
))


def sample_groups() -> pd.DataFrame:
    """Örnek dosya: kurgusal iş arama programının program sonrası çalışan kişileri; kategoriler aynı verinin sütunlarından
    kurulur (cinsiyet, medeni durum, yaş grubu, program grubu). Öğrenci verisi değil."""

    from core.labs.ornek_regresyon import program_frame

    frame = program_frame()
    frame = frame[frame["issiz"] == 0].reset_index(drop=True)
    age = np.where(frame["yas"] < 30, "20–29", np.where(frame["yas"] < 40, "30–39", "40 ve üstü"))
    return pd.DataFrame({
        "Yıllık kazanç (bin TL)": frame["kazanc"],
        "Eğitim yılı": frame["egitim"],
        "Önceki yıllık kazanç (bin TL)": frame["onceki_kazanc"],
        "Cinsiyet": np.where(frame["kadin"] == 1, "Kadın", "Erkek"),
        "Medeni durum": np.where(frame["evli"] == 1, "Evli", "Bekâr"),
        "Yaş grubu": age,
        "Grup": np.where(frame["program"] == 1, "Program", "Kontrol"),
    })


CUSTOM = custom_lab(
    build,
    sample_groups,
    ("Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sonuç ve temel açıklayıcı değişken zorunludur ve sayısal olmalıdır. "
     f"{POSITIVE_RULE} İki kategorili değişken (kukla) ile çok kategorili değişkenden en az biri seçilmelidir; ikinci "
     "kategorik değişken isteğe bağlıdır (Adım 10). Kontroller temel açıklayıcı ve ek sayısal değişkenlerdir (en çok 3). "
     f"{ROW_RULE}"),
    ROLES,
    "Kontrol değişkenleridir (Adım 2, 3 ve 11).",
    order_roles=(GOSTERGE, KATEGORI, KATEGORI2),
    validate=own_validate,
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
