"""Konu 11 genel uygulaması: etkileşim terimleri ve gruplar arasında sabit ile eğim farkları.

Notlardaki on adım (``core.labs.konu11``) aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde yazılır:
additif kukla modelinin paralel doğruları, etkileşim terimi ve grup denklemleri, dört olası yapı, koşullu grup farkı ve
merkezleme, ana uygulama (log sonuç), grup farkının açıklayıcının düzeyine göre değişimi, ikinci uygulama (düzey sonuç),
ikinci uygulamada farkın değişimi, etkileşim hipotez testleri ve formül yazımı ile makale tablosu.

Alternatif örnekte ana veri BEAUTY'dir (Hamermesh ve Biddle, 1994; 1260 çalışan): log saatlik ücret, 12 yıl etrafında
merkezlenmiş eğitim (``educ12``) ve iş deneyimi; kukla notlardaki gibi kadın kuklasıdır (evli, siyah ve sendika üyesi
kuklaları da seçilebilir). Notlarda HPRICE1 ile yapılan adımlar KIELMC'nin 1978 satışlarıyla yapılır: fiyat bin dolar,
konut büyüklüğü 2.000 fit² etrafında merkezli ve 100 fit² biriminde (``area2k``), oda ve banyo sayısı. Kukla konutun
sonradan kurulan çöp yakma tesisine 3 mil ya da daha yakın olmasıdır (``nearinc``; 1978'de tesis yoktu, kukla konumu
gösterir).

"Kendi verini yükle" seçeneğinde iki kategorili değişken zorunludur (bir kategori 1, diğeri 0). Ana modeller notlardaki
gibi log sonuçla kurulur (sonucun bütün değerleri pozitif olmalı); temel açıklayıcı ortancasına yakın yuvarlak bir değer
etrafında merkezlenir (notlardaki ``educ12`` gibi). Notlarda ikinci veriyle yapılan adımlar (7–8) aynı dosyanın düzey
sonucuyla, ayrı bir çerçevede kurulur. Etkileşim temel açıklayıcıyla ya da bir ek değişkenle kurulabilir.

Etkileşim katsayısının işareti ya da yüzde 5 düzeyindeki anlamlılığı tek bir gözleme bağlıysa (Cook uzaklığı en büyük
gözlem çıkarılınca sonuç değişiyorsa) Adım 7 o gözlem olmadan da modeli kurar ve sonucu yazar; notlardaki HPRICE1
örneğinde en büyük arsanın etkisi gibi. Gözlemin seçimi uygulamada hesaplanır (statsmodels ``get_influence``); model ve
sayıları üretilen kodda da vardır.

Etkileşim notlardaki gibidir: additif modeldeki kukla (Adım 1), etkileşimdeki nicel değişken (Adım 2), grafikteki yapı
(Adım 3), merkezleme noktası (Adım 4), ana uygulamadaki kukla (Adım 5–6), farkın hesaplandığı düzey (Adım 6), ikinci
uygulamada kuklayla etkileşen değişken (Adım 7–8), gösterilen test (Adım 9) ve formül yazımı (Adım 10). Kendi verinde tek
seçenekli denetimler (tek kukla, ek değişken yoksa tek açıklayıcı) gösterilmez. Standart hatalar klasik EKK standart
hatalarıdır (dayanıklı çıkarım Konu 12).
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
from core.labs.ornek import EXACT_FIT, Case, Role, TopicVariants, free_name, md, sayi, sayim, stable_checks, \
    with_app_values
from core.labs.ornek_regresyon import (
    ACIKLAYICI,
    BEAUTY_PHRASES,
    BEAUTY_UNITS,
    GOSTERGE,
    HOUSE_COLUMNS,
    POSITIVE_RULE,
    ROW_RULE,
    SONUC,
    beauty_case,
    candidates,
    capital,
    custom_lab,
    digits_for,
    display,
    house_case,
    labels_of,
    level_text,
    level_value,
    listing,
    log_label,
    nice,
    number_control,
    p_text,
    phrase,
    program_frame,
    roles,
    second,
    short_unit,
    validate,
    validate_positive,
)
from core.labs.spec import (
    INTERCEPT,
    OLS,
    Check,
    Choice,
    CoefficientTable,
    CoefTarget,
    CopyFrame,
    Count,
    Derive,
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
    NoteRef,
    NumberChoice,
    RegressionTable,
    Scalar,
    ScalarTable,
    ScalarTarget,
    ShowFrame,
    ShowModel,
    Statistic,
    Support,
    TableTarget,
    TakeRows,
    interactive_step,
)

TOPIC = "konu11"
TITLE = "Uygulama: Etkileşim Terimleri ve Gruplar Arasında Sabit ile Eğim Farklılıkları"
GRID = 50
FEW_OBSERVATIONS = 10
"""Adım 4: merkezleme noktasında bundan az gözlem varsa farkın büyük ölçüde doğrusal varsayıma dayandığı yazılır."""
FRAGILE = frozenset(("F_egim_w", "p_egim_w", "t_egim_w", "t2_egim_w", "F_ortak_w", "p_ortak_w", "F_egim_h", "p_egim_h",
                     "t_egim_h", "t2_egim_h", "F_ortak_h", "p_ortak_h"))
"""Uyum tamsa (R² ≈ 1) yuvarlama hatasına duyarlı Adım 9 skalerleri."""


def _check(label: str, target, decimals: int) -> Check:
    return Check(label, target, 0.0, decimals)


def _scalar(name: str, label: str, decimals: int) -> Check:
    return _check(label, ScalarTarget(name), decimals)


def _d(value: float, minimum: int = 4) -> int:
    """Metindeki ve kontroldeki basamak: en az ``minimum``; küçük değerde en az üç anlamlı basamak."""

    return max(minimum, digits_for(float(value), minimum))


def _formula(outcome: str, terms) -> str:
    return OLS("gecici", "gecici", outcome, tuple(terms), "").formula


def _code(value: float) -> str:
    """Koddaki sayı yazımı (açıklamada ters tırnak içinde): 12, 2.5."""

    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


# --- Adlar ve roller ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Names:
    """Ana verinin türetilen sütunları: kukla (kendi verinde 0/1 kodu), log sonuç, merkezlenmiş temel açıklayıcı ve Adım
    4'ün merkezlenmiş değişkeni."""

    dummy: str
    ly: str
    xc: str
    xcc: str


def _spread_round(value: float, values) -> float:
    """Değerin verinin yayılımına göre yuvarlanmışı: aralığın onda birinden küçük en büyük 10'un kuvvetine (ör.
    1990–2020 yıllarında 1, 20.000–90.000 TL'de 1.000); verinin aralığının dışına taşmaz."""

    values = np.asarray(values, dtype=float)
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return float(value)
    step = 10.0 ** math.floor(math.log10((high - low) / 10))
    digits = max(0, -int(math.floor(math.log10(step))))
    return float(min(max(round(round(value / step) * step, digits), low), high))


def _number(case: Case, key: str, label: str, help: str = "") -> NumberChoice:
    """Temel açıklayıcı için sayı denetimi (Adım 4 ve 6): aralık verinin aralığı ve sıfır, varsayılan ortancanın verinin
    yayılımına göre yuvarlanmışı (denetimin adımına yuvarlanır; merkezleme noktası da budur)."""

    values = case.data[case.roles[ACIKLAYICI]].astype(float)
    low, high = min(float(values.min()), 0.0), max(float(values.max()), 0.0)
    integral = bool(np.all(values == np.round(values)))
    return number_control(key, label, _spread_round(float(values.median()), values), low, high, help=help,
                          step=1.0 if integral and high - low <= 2000 else None)


def _center(case: Case) -> float:
    """Temel açıklayıcının merkezleme noktası: alternatif örnekte 12 yıl; kendi verinde ortancanın verinin yayılımına
    göre yuvarlanmışı (notlardaki educ12 gibi yuvarlak bir değer; Adım 4'ün varsayılanıyla aynı)."""

    if "center" in case.extra:
        return float(case.extra["center"])
    return float(_number(case, "adim4_c", "c").default)


def _names(case: Case) -> Names:
    if "names" in case.extra:
        return Names(**case.extra["names"])
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    taken = set(case.data.columns)
    dummy = K.code_name(f"{case.roles[GOSTERGE]}_01", taken)
    taken.add(dummy)
    ly = free_name(f"ln_{y}", taken)
    taken.add(ly)
    xc = x if _center(case) == 0 else free_name(f"{x}_m", taken)
    taken.add(xc)
    return Names(dummy, ly, xc, free_name(f"{x}_c", taken))


@dataclass(frozen=True)
class Dummy:
    """İki kategorili değişken: 0/1 sütunu, 1 ve 0 gruplarının metindeki (kaçırılmış) ve etiketteki (dosyadaki) adları ve
    kuklanın başlığı."""

    column: str
    one: str
    zero: str
    one_raw: str
    zero_raw: str
    title: str


def _own_dummy(case: Case) -> Dummy:
    column = case.roles[GOSTERGE]
    one = case.levels[GOSTERGE]
    zero = next(item for item in case.orders[column] if item != one)
    return Dummy(_names(case).dummy, f"“{md(one)}”", f"“{md(zero)}”", one, zero, case.name(column))


def _dummies(case: Case) -> dict[str, Dummy]:
    """Ana uygulamanın kuklaları: alternatif örnekte kadın, evli, siyah ve sendika üyesi; kendi verinde iki kategorili
    değişken."""

    if "dummies" in case.extra:
        return {key: Dummy(key, one, zero, one, zero, title) for key, (one, zero, title) in case.extra["dummies"].items()}
    found = _own_dummy(case)
    return {found.column: found}


def _dummy(case: Case) -> Dummy:
    return next(iter(_dummies(case).values()))


def _house(case: Case) -> Case:
    """Notlarda ikinci veriyle yapılan adımların (7–8) verisi: alternatif örnekte KIELMC (1978), kendi verinde aynı dosya
    ayrı bir çerçevede (düzey sonuç)."""

    return second(case)


def _house_dummy(case: Case) -> Dummy:
    house = _house(case)
    if "dummy" in house.extra:
        key, one, zero, title = house.extra["dummy"]
        return Dummy(key, one, zero, one, zero, title)
    return _own_dummy(case)


def _house_outcome(case: Case) -> str:
    return _house(case).roles[SONUC]


def _controls(case: Case) -> tuple[str, ...]:
    """Ana modellerin kontrolleri: alternatif örnekte iş deneyimi; kendi verinde kullanılabilen ek değişkenler."""

    if "controls" in case.extra:
        return tuple(case.extra["controls"])
    x = case.roles[ACIKLAYICI]
    return tuple(column for column in candidates(case) if column != x)


def _options(case: Case) -> tuple[str, ...]:
    """Kuklayla etkileşebilen nicel değişkenler: merkezlenmiş temel açıklayıcı ve kontroller."""

    return (_names(case).xc, *_controls(case))


def _house_options(case: Case) -> tuple[str, ...]:
    house = _house(case)
    return tuple(candidates(house)) if not case.own else _options(case)


def _others(term: str, options: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(item for item in options if item != term)


def _terms(dummy: str, x: str, others: tuple[str, ...]) -> tuple[str, ...]:
    return (dummy, x, f"{dummy}:{x}", *others)


# --- Metindeki adlar --------------------------------------------------------------------------------------------

def _base(case: Case, term: str) -> bool:
    """Terim kendi verinde merkezlenmiş temel açıklayıcı mı."""

    return case.own and term == _names(case).xc and term != case.roles[ACIKLAYICI]


def _word(case: Case, term: str) -> str:
    """Markdown metnindeki ad: "eğitim"; kendi verinde tırnak içinde dosyadaki ad (merkezlenmiş değişkende temel
    açıklayıcınınki)."""

    return phrase(case, case.roles[ACIKLAYICI] if _base(case, term) else term)


def _word_raw(case: Case, term: str) -> str:
    """Etiket ve başlıklardaki ad (arayüz kendi verinde kaçırır)."""

    if case.own:
        return case.name(case.roles[ACIKLAYICI] if _base(case, term) else term)
    return phrase(case, term)


def _label(case: Case, term: str) -> str:
    """Başlıkta büyük harfle başlayan ad: "Eğitim"; kendi verinde dosyadaki ad."""

    return _word_raw(case, term) if case.own else capital(_word_raw(case, term))


def _title(case: Case, term: str) -> str:
    """Seçeneklerdeki ve ölçülerdeki ad, birimiyle: "Eğitim − 12 (yıl)"; kendi verinde merkezlenmiş değişken "X − c"."""

    if _base(case, term):
        return _minus(case.name(case.roles[ACIKLAYICI]), _center(case))
    if case.own and term == _names(case).dummy:
        return case.name(case.roles[GOSTERGE])
    return display(case, term)


def _cap(case: Case, text: str) -> str:
    return text if case.own else capital(text)


def _forms(case: Case) -> dict[str, str]:
    """Grupların cümledeki biçimi: "kadın çalışanların", "tesise yakın konutlarda"; kendi verinde "“Kadın” grubunun"."""

    members = case.extra.get("members")
    if members:
        plural = f"{members}lar"
        return {"nom": plural, "gen": f"{plural}ın", "loc": f"{plural}da", "nom_pl": plural, "gen_pl": f"{plural}ın",
                "dat_pl": f"{plural}a", "loc_pl": f"{plural}da"}
    return {"nom": "grubu", "gen": "grubunun", "loc": "grubunda", "nom_pl": "grupları", "gen_pl": "gruplarının",
            "dat_pl": "gruplarına", "loc_pl": "gruplarında"}


def _unit(case: Case, term: str) -> str:
    unit = short_unit(case, term)
    return unit or "birim"


def _log_phrase(case: Case) -> str:
    """Log sonucun metindeki adı: "log saatlik ücret"; kendi verinde ln(“Y”)."""

    if case.own:
        return f"ln({phrase(case, case.roles[SONUC])})"
    return phrase(case, _names(case).ly)


def _log_title(case: Case) -> str:
    return log_label(case, case.roles[SONUC])


def _held(case: Case, others: tuple[str, ...]) -> str:
    """Grafik başlığında ortalamada tutulanlar: " (iş deneyimi ortalamada)"."""

    if not others:
        return ""
    return f" ({listing([_word_raw(case, term) for term in others])} ortalamada)"


def _data_prefix(case: Case) -> str:
    name = case.extra.get("data_name")
    return f"{name}, " if name and not case.own else ""


def _zero_point(case: Case, term: str) -> str | None:
    """X = 0 noktasının metindeki yazımı (değişken merkezliyse): "eğitim 12 yıl iken"; merkezli değilse ``None``."""

    known = dict(case.extra.get("zero_points", {}))
    if term in known:
        return known[term]
    if _base(case, term):
        return f"{phrase(case, case.roles[ACIKLAYICI])} = {level_text(_center(case))} iken"
    return None


def _raw(case: Case, term: str) -> tuple[str, float, float]:
    """Modeldeki değerin özgün ölçeğe çevrilmesi: (özgün sütun, çarpan, öteleme); özgün değer = çarpan·değer + öteleme
    (ör. area2k → area = 100·area2k + 2000)."""

    known = dict(case.extra.get("raw", {}))
    if term in known:
        return known[term]
    if _base(case, term):
        return case.roles[ACIKLAYICI], 1.0, _center(case)
    return term, 1.0, 0.0


def _point_label(value: float) -> str:
    """Tablo satırındaki değer: tam sayıda binlik ayırıcı (1.500), kesirlide ondalık virgül."""

    number = float(value)
    return sayim(number) if number.is_integer() else level_text(number)


def _number_text(value: float) -> str:
    """Metindeki bir hesap sonucu (ör. farkın işaret değiştirdiği nokta): büyük değerde binlik ayırıcıyla tam sayı
    (2.244), küçük değerde üç anlamlı basamak (0,342; 0,00499)."""

    number = float(value)
    if abs(number) >= 1000:
        return sayim(round(number))
    return level_text(nice(number, 3))


def _minus(name: str, center: float) -> str:
    """"X − c" yazımı; c negatifse "X + |c|"."""

    return f"{name} − {level_text(center)}" if center >= 0 else f"{name} + {level_text(-center)}"


# --- Veri (metinler, basamaklar ve doğrulama için; uygulamanın hesabı işlemlerle yapılır) -------------------------

def _data(case: Case) -> pd.DataFrame:
    names = _names(case)
    data = case.data.copy()
    if case.own:
        data[names.ly] = np.log(data[case.roles[SONUC]].astype(float))
        dummy = _own_dummy(case)
        data[names.dummy] = (data[case.roles[GOSTERGE]] == dummy.one_raw).astype(float)
    x = case.roles[ACIKLAYICI]
    if names.xc != x:
        data[names.xc] = data[x].astype(float) - _center(case)
    return data


def _house_data(case: Case) -> pd.DataFrame:
    return _data(case) if case.own else _house(case).data


def _matrix(data: pd.DataFrame, terms) -> np.ndarray:
    columns = [np.ones(len(data))]
    for term in terms:
        value = np.ones(len(data))
        for part in term.split(":"):
            value = value * data[part].to_numpy(dtype=float)
        columns.append(value)
    return np.column_stack(columns)


def _full_rank(data: pd.DataFrame, terms) -> bool:
    """Model kurulabilir mi: tam doğrusal bağlantı yok ve artık serbestlik derecesi en az 1."""

    design = _matrix(data, terms)
    norms = np.linalg.norm(design, axis=0)
    if len(data) < design.shape[1] + 1 or np.any(norms == 0):
        return False
    return int(np.linalg.matrix_rank(design / norms, tol=1e-10)) == design.shape[1]


def _exact(case: Case, data: pd.DataFrame, outcome: str, terms) -> bool:
    """Uyum (neredeyse) tam mı (R² ≈ 1): standart hata, t, p ve F yuvarlama hatasına duyarlıdır."""

    if not case.own:
        return False
    design = _matrix(data, terms)
    values = data[outcome].to_numpy(dtype=float)
    estimates, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ estimates
    total = float(((values - values.mean()) ** 2).sum())
    return float((residual ** 2).sum()) <= EXACT_FIT * total


def _fit(data: pd.DataFrame, outcome: str, terms):
    return smf.ols(f"{outcome} ~ {' + '.join(terms)}", data=data).fit()


def _decimals(fit, terms, minimum: int = 4) -> int:
    return max((_d(float(fit.params[term]), minimum) for term in terms if term in fit.params.index), default=minimum)


def _stable(checks, exact: bool, table: str | None = None) -> tuple[Check, ...]:
    kept = stable_checks(tuple(checks), exact, table)
    if not exact:
        return kept
    return tuple(check for check in kept if not (isinstance(check.target, ScalarTarget) and check.target.name in FRAGILE))


EXACT_NOTE = (" Model veriye neredeyse tam uyuyor (R² ≈ 1): standart hata, t, p ve F yuvarlama hatasına duyarlıdır ve "
              "indirilen kodda karşılaştırılmaz.")


# --- Ortak işlemler -------------------------------------------------------------------------------------------

def _live(control) -> bool:
    options = getattr(control, "options", None)
    return options is None or len(options) > 1


def _step(*, number: int, title: str, note: NoteRef, explanation: str, build, checks, note_for, controls=(),
          uses=()) -> LabStep:
    """Adım: tek seçenekli denetimler (kendi verinde tek kukla ya da tek açıklayıcı) gösterilmez, varsayılanıyla
    kurulur; hiç denetim kalmazsa adım etkileşimli değildir."""

    shown = tuple(control for control in controls if _live(control))
    used = tuple(control for control in uses if _live(control))
    fixed = {control.key: control.default for control in (*controls, *uses) if not _live(control)}

    def run(choices) -> tuple:
        return tuple(build({**fixed, **choices}))

    def text(state, choices) -> str:
        return note_for(state, {**fixed, **choices})

    if not shown and not used:
        return LabStep(number=number, title=title, note=note, explanation=explanation, operations=run({}), checks=checks,
                       note_for=lambda state, choices: text(state, {}))
    return interactive_step(number=number, title=title, note=note, explanation=explanation, controls=shown, uses=used,
                            build=run, checks=checks, note_for=text)


def _setup(case: Case, frame: str, log: bool = True) -> tuple:
    """Kendi verinde kuklanın 0/1 kodu ve log sonuç; merkezlenmiş temel açıklayıcı (alternatif örnekte educ12)."""

    names = _names(case)
    x, y = case.roles[ACIKLAYICI], case.roles[SONUC]
    operations: list = []
    if case.own:
        dummy = _own_dummy(case)
        operations.append(MapCodes(frame, case.roles[GOSTERGE], dummy.column, ((dummy.zero_raw, 0), (dummy.one_raw, 1)),
                                   f"Kukla değişken: {dummy.zero_raw} = 0, {dummy.one_raw} = 1"))
        if log:
            operations.append(Derive(frame, names.ly, E.log(E.var(y)), f"ln({case.name(y)})"))
    if names.xc != x:
        center = _center(case)
        text = level_text(center)
        operations.append(Derive(frame, names.xc, E.sub(E.var(x), level_value(case, x, center)),
                                 case.extra.get("center_comment")
                                 or f"{case.name(x)} {text} etrafında merkezlenir: {names.xc} = {_minus(x, center)}"))
    return tuple(operations)


def _line_grid(values: pd.Series) -> tuple:
    values = values.astype(float)
    low, high = float(values.min()), float(values.max())
    if bool(np.all(values == np.round(values))) and high - low <= 100:
        return ("support", int(low), int(high))
    return ("inline", tuple(float(round(value, 10)) for value in np.linspace(low, high, GRID)))


def _grid_info(case: Case, term: str) -> tuple[str, tuple, float, str]:
    """Grafiğin ızgarası: (ızgara sütunu, ızgara, merkez, eksen adı). Doğrudaki değer ızgara − merkezdir (educ12 =
    educ − 12)."""

    known = dict(case.extra.get("grids", {}))
    if term in known:
        return known[term]
    column, _, offset = _raw(case, term)
    return column, _line_grid(case.data[column]), offset, case.name(column)


def _grid_op(frame: str, column: str, grid: tuple, label: str):
    kind, *rest = grid
    if kind == "support":
        lower, upper = rest
        return Support(frame, column, lower, upper, f"Izgara: {lower}, …, {upper}")
    return InlineData(frame, (column,), tuple((value,) for value in rest[0]),
                      f"{label} ızgarası: örneklemin aralığında {GRID} nokta")


def _grid_bounds(grid: tuple) -> tuple[float, float]:
    kind, *rest = grid
    return (float(rest[0]), float(rest[1])) if kind == "support" else (float(rest[0][0]), float(rest[0][-1]))


def _shift(column: str, center: float):
    if not center:
        return E.var(column)
    return E.sub(E.var(column), int(center) if float(center).is_integer() else center)


def _group_lines(frame: str, info: tuple, prefix: str, dummy: Dummy, others: tuple[str, ...], averages: str,
                 log: bool = False) -> tuple:
    """İki grubun tahmin edilen doğruları bir ızgarada: diğer açıklayıcılar örneklem ortalamasında.

    Katsayı skalerleri ``{prefix}_b0``, ``{prefix}_g0``, ``{prefix}_b1``, ``{prefix}_g1`` ve ``{prefix}_k_{terim}``;
    ortalamalar ``{averages}_{terim}``. ``log``: doğrular exp() ile sonucun düzeyine çevrilir."""

    column, grid, center, label = info
    x = _shift(column, center)
    line = E.add(E.ref(f"{prefix}_b0"), E.mul(E.ref(f"{prefix}_b1"), x))
    for term in others:
        line = E.add(line, E.mul(E.ref(f"{prefix}_k_{term}"), E.ref(f"{averages}_{term}")))
    shifted = E.add(E.add(E.var("grup0"), E.ref(f"{prefix}_g0")), E.mul(E.ref(f"{prefix}_g1"), x))
    held = "; diğer açıklayıcılar ortalamada" if others else ""
    operations: list = [
        _grid_op(frame, column, grid, label),
        Derive(frame, "grup0", line, f"{dummy.zero_raw} ({dummy.column} = 0): β̂₀ + β̂₁·x{held}"),
        Derive(frame, "grup1", shifted, f"{dummy.one_raw} ({dummy.column} = 1): önceki doğru + γ̂₀ + γ̂₁·x"),
    ]
    if log:
        operations += [Derive(frame, "duzey0", E.exp(E.var("grup0")), f"{dummy.zero_raw}: exp(tahmin edilen log sonuç)"),
                       Derive(frame, "duzey1", E.exp(E.var("grup1")), f"{dummy.one_raw}: exp(tahmin edilen log sonuç)")]
    return tuple(operations)


def _coefficients(case: Case, model: str, prefix: str, dummy: str, slope: str, others: tuple[str, ...],
                  interaction: bool = True, shown: bool = True, decimals: int = 4) -> tuple:
    """Katsayı skalerleri: sabit, kukla, eğim, (etkileşim) ve diğer açıklayıcılar. ``shown=False``: sayılar aynı adımda
    bir çıktıda ya da tabloda görünüyor."""

    values = [
        ModelValue(f"{prefix}_b0", model, "coef", "Sabit terim β̂₀", term=INTERCEPT, decimals=decimals, shown=shown),
        ModelValue(f"{prefix}_g0", model, "coef", "Kukla katsayısı γ̂₀", term=dummy, decimals=decimals, shown=shown),
        ModelValue(f"{prefix}_b1", model, "coef", f"{_label(case, slope)} eğimi β̂₁", term=slope, decimals=decimals,
                   shown=shown),
    ]
    if interaction:
        values.append(ModelValue(f"{prefix}_g1", model, "coef", "Etkileşim katsayısı γ̂₁", term=f"{dummy}:{slope}",
                                 decimals=decimals, shown=shown))
    values += [ModelValue(f"{prefix}_k_{term}", model, "coef", f"{_title(case, term)} katsayısı", term=term,
                          decimals=decimals, shown=False) for term in others]
    return tuple(values)


def _averages(case: Case, frame: str, prefix: str, terms: tuple[str, ...]) -> tuple:
    return tuple(Statistic(frame, term, "mean", f"{prefix}_{term}", f"{_title(case, term)}: örneklem ortalaması",
                           decimals=4) for term in terms)


def _signed(value: float, decimals: int) -> str:
    return sayi(value, decimals) if value >= 0 else f"({sayi(value, decimals)})"


def _slope_sum(first: float, second: float, exact: float, decimals: int) -> str:
    """İki katsayının toplamı ekrandaki terimlerle: 0,0663 + 0,0192 = 0,0855; yuvarlanmamış katsayılarla toplam farklı
    yuvarlanıyorsa parantezde verilir."""

    rounded = float(f"{first:.{decimals}f}") + float(f"{second:.{decimals}f}")
    text = f"{sayi(first, decimals)} + {_signed(second, decimals)} = {sayi(rounded, decimals)}"
    if sayi(rounded, decimals) != sayi(exact, decimals):
        text += f" (yuvarlanmamış katsayılarla {sayi(exact, decimals)})"
    return text


def _example_rows(data: pd.DataFrame, column: str) -> tuple[int, ...]:
    """Her gruptan ilk dört gözlemin sıra numarası (1'den)."""

    values = data[column].to_numpy(dtype=float)
    zero = [index + 1 for index in np.flatnonzero(values == 0)[:4]]
    one = [index + 1 for index in np.flatnonzero(values == 1)[:4]]
    return tuple(sorted(int(row) for row in (*zero, *one)))


# --- Adım 1: additif kukla modeli ve paralel doğrular ------------------------------------------------------------

def _dummy_control(case: Case, key: str, label: str, help: str) -> Choice:
    dummies = _dummies(case)
    default = next(iter(dummies))
    return Choice(key, label, tuple((code, item.title) for code, item in dummies.items()), default, help=help)


def _model_text(case: Case, terms: tuple[str, ...], outcome: str) -> str:
    return f"`{_formula(outcome, terms)}`"


def _centering_text(case: Case) -> str:
    """Merkezlemenin açıklaması: "eğitim 12 yıl etrafında merkezlidir (`educ12 = educ − 12`)"."""

    names = _names(case)
    x = case.roles[ACIKLAYICI]
    if names.xc == x:
        return ""
    center = _center(case)
    code = f"{x} - {_code(center)}" if center >= 0 else f"{x} + {_code(-center)}"
    return f" {capital(phrase(case, x))} {level_text(center)} etrafında merkezlidir: `{names.xc} = {code}`."


def _step1(case: Case) -> LabStep:
    names = _names(case)
    dummies = _dummies(case)
    control = _dummy_control(case, "adim1_kukla", "Kukla değişken D",
                             "Varsayılan: " + _dummy(case).title.lower() + " kuklası; diğer kuklalar aynı modelde "
                             "karşılaştırılır.")
    default = control.default
    controls = _controls(case)
    info = _grid_info(case, names.xc)
    data = _data(case)
    base_terms = (default, names.xc, *controls)
    fit = _fit(data, names.ly, base_terms)
    exact = _exact(case, data, names.ly, base_terms)
    d = _decimals(fit, base_terms)

    def build(choices) -> tuple:
        code = choices["adim1_kukla"]
        dummy = dummies[code]
        terms = (code, names.xc, *controls)
        same = code == default
        side: tuple = () if same else (
            OLS("m_add_n", case.frame, names.ly, base_terms, f"Varsayılan additif model: {_formula(names.ly, base_terms)}"),
            RegressionTable(((f"Varsayılan: {dummies[default].title}", "m_add_n"), (f"Seçiminiz: {dummy.title}", "m_add")),
                            (INTERCEPT, *dict.fromkeys((default, code)), names.xc, *controls), "yan111",
                            "Varsayılan additif model ve seçtiğiniz kukla", decimals=4, exact=not case.own,
                            adj_r2=True),
        )
        dd = _decimals(_fit(data, names.ly, terms), terms)
        return (
            *case.load,
            *_setup(case, case.frame),
            OLS("m_add", case.frame, names.ly, terms, f"Additif kukla modeli: {_formula(names.ly, terms)}"),
            ShowModel("m_add", "Additif kukla modelinin Python çıktısı" + ("" if same else " (seçiminiz)"),
                      columns=("coef", "se", "t", "p"), stats=("nobs", "r2", "adj_r2"), decimals=(("se", 4), ("p", 4)),
                      exact=not case.own),
            *side,
            *_coefficients(case, "m_add", "a1", code, names.xc, controls, interaction=False, shown=False, decimals=dd),
            *_averages(case, case.frame, "ort1", controls),
            Scalar("a1_g1", E.const(0), "Additif modelde eğim farkı yoktur (γ₁ = 0)", decimals=0, shown=False),
            *_group_lines("dogru1", info, "a1", dummy, controls, "ort1"),
            LineChart("dogru1", info[0], "grup0", info[3], f"Tahmin edilen {_log_title(case)}",
                      f"{_data_prefix(case)}additif kukla modeli: {dummy.one_raw} ve {dummy.zero_raw} "
                      f"{_forms(case)['gen_pl']} doğruları paraleldir{_held(case, controls)}",
                      markers=False, series=(("grup1", _cap(case, dummy.one_raw)),), legend=_cap(case, dummy.zero_raw)),
            Derive("dogru1", "fark", E.sub(E.var("grup1"), E.var("grup0")),
                   f"Dikey fark: {dummy.one_raw} − {dummy.zero_raw}"),
            Statistic("dogru1", "fark", "min", "fark1_min", "Dikey fark: en küçük (her düzeyde)", decimals=dd),
            Statistic("dogru1", "fark", "max", "fark1_max", "Dikey fark: en büyük", decimals=dd),
        )

    def note(state, choices) -> str:
        dummy = dummies[choices["adim1_kukla"]]
        s = state.scalars
        forms = _forms(case)
        slope = _d(s["a1_b1"])
        gap = _d(s["fark1_min"])
        word = _word(case, names.xc)
        return (f"İki doğrunun eğimi aynıdır ({word} katsayısı {sayi(s['a1_b1'], slope)}); dikey fark her {word} "
                f"düzeyinde kukla katsayısıdır: en küçük ve en büyük fark {sayi(s['fark1_min'], gap)} ve "
                f"{sayi(s['fark1_max'], gap)}. Additif model {dummy.one} ve {dummy.zero} {forms['dat_pl']} farklı sabit "
                f"verir, farklı eğim vermez; grup farkı {word} düzeyinden bağımsız varsayılır. Bu varsayım veriyle ayrıca "
                "değerlendirilir: eğimin gruba göre değişmesine izin veren terim etkileşim terimidir (Adım 2, §11.1).")

    checks = [
        *(_check(f"Additif model: {_title(case, term)}, {label}", CoefTarget("m_add", term, quantity),
                 _d(float(getattr(fit, attribute)[term])))
          for term in base_terms for quantity, label, attribute in (("coef", "katsayı", "params"),
                                                                    ("se", "standart hata", "bse"))),
        _check("Additif model: gözlem sayısı", ModelTarget("m_add", "nobs"), 0),
        _check("Additif model: R²", ModelTarget("m_add", "r2"), 5),
        _check("Additif model: düzeltilmiş R²", ModelTarget("m_add", "adj_r2"), 5),
        _scalar("fark1_min", "Dikey fark: en küçük", d),
        _scalar("fark1_max", "Dikey fark: en büyük", d),
    ]
    lead = ("Additif modelde $Y_i = \\beta_0 + \\beta_1 X_i + \\gamma_0 D_i + u_i$: iki grubun sabiti farklı olabilir, "
            "eğimi aynıdır. $\\mathbb{E}(Y \\mid X, D = 1) - \\mathbb{E}(Y \\mid X, D = 0) = \\gamma_0$ her $X$ düzeyinde "
            "aynıdır ve tahmin doğruları paraleldir.")
    return _step(
        number=1,
        title="Additif kukla modeli: paralel doğrular",
        note=NoteRef("11.1", 0, ("Şekil 11.1", "Tablo 11.6 (§11.9)")),
        explanation=(f"{lead} {case.extra.get('model_text', 'Model:')} "
                     f"{_model_text(case, base_terms, names.ly)}.{_centering_text(case) if case.own else ''}"
                     + (" Kukla değişkeni değiştirin." if _live(control) else "")),
        controls=(control,),
        build=build,
        checks=_stable(checks, exact),
        note_for=note,
    )


# --- Adım 2: etkileşim terimi ve grup denklemleri ----------------------------------------------------------------

def _group_rows(case: Case, dummy: Dummy, slope: str | None) -> tuple[tuple[str, str], ...]:
    """Grup denklemleri tablosunun satırları (Tablo 11.1'deki roller) ve skaler sonekleri; ``slope`` verilirse eğim
    satırlarında değişkenin adı yazılır."""

    zero, one = _cap(case, dummy.zero_raw), _cap(case, dummy.one_raw)
    word = f"{_word_raw(case, slope)} eğimi" if slope else "eğim"
    return ((f"{zero} ({dummy.column} = 0): sabit β̂₀", "b0"), (f"{one} ({dummy.column} = 1): sabit β̂₀ + γ̂₀", "sabit1"),
            (f"{zero}: {word} β̂₁", "b1"), (f"{one}: {word} β̂₁ + γ̂₁", "egim1"), ("Sabit farkı γ̂₀", "g0"),
            ("Eğim farkı γ̂₁", "g1"))


def _group_equations(case: Case, model: str, prefix: str, dummy: Dummy, slope: str, others: tuple[str, ...],
                     decimals: int, shown: bool = True) -> tuple:
    return (
        *_coefficients(case, model, prefix, dummy.column, slope, others, shown=shown, decimals=decimals),
        Scalar(f"{prefix}_sabit1", E.add(E.ref(f"{prefix}_b0"), E.ref(f"{prefix}_g0")),
               f"{_cap(case, dummy.one_raw)} grubunun sabiti β̂₀ + γ̂₀", decimals=decimals, shown=shown),
        Scalar(f"{prefix}_egim1", E.add(E.ref(f"{prefix}_b1"), E.ref(f"{prefix}_g1")),
               f"{_cap(case, dummy.one_raw)} grubunun {_word_raw(case, slope)} eğimi β̂₁ + γ̂₁", decimals=decimals,
               shown=shown),
    )


def _step2(case: Case) -> LabStep:
    names = _names(case)
    dummy = _dummy(case)
    options = _options(case)
    data = _data(case)
    control = Choice("adim2_x", f"{dummy.title} kuklasıyla etkileşen nicel değişken X",
                     tuple((term, _title(case, term)) for term in options), names.xc,
                     help=f"Varsayılan: {_title(case, names.xc)}.")
    default_terms = _terms(dummy.column, names.xc, _others(names.xc, options))
    exact = _exact(case, data, names.ly, default_terms)
    fit = _fit(data, names.ly, default_terms)
    d = _decimals(fit, default_terms)
    rows = _example_rows(data, dummy.column)

    def build(choices) -> tuple:
        x = choices["adim2_x"]
        others = _others(x, options)
        terms = _terms(dummy.column, x, others)
        info = _grid_info(case, x)
        same = x == names.xc
        dd = _decimals(_fit(data, names.ly, terms), terms)
        if same:
            table: tuple = (
                ScalarTable(tuple((label, E.ref(f"e2_{suffix}")) for label, suffix in _group_rows(case, dummy, x)),
                            "grup_denklemleri", decimals=dd, heading="Katsayı", value="Tahmin",
                            title="Grup denklemleri: sabitler ve eğimler (Tablo 11.1'deki roller)"),
            )
        else:  # varsayılan etkileşim seçilen değişkenle yan yana
            dn = max(dd, d)
            table = (
                OLS("m_etk_n", case.frame, names.ly, default_terms,
                    f"Varsayılan model: {_formula(names.ly, default_terms)}"),
                *_group_equations(case, "m_etk_n", "e2n", dummy, names.xc, _others(names.xc, options), dn, shown=False),
                *(ScalarTable(tuple((label, E.ref(f"{prefix}_{suffix}")) for label, suffix in _group_rows(case, dummy, None)),
                              result, decimals=dn, heading="Katsayı")
                  for prefix, result in (("e2n", "gd_varsayilan"), ("e2", "gd_secim"))),
                JoinColumns("grup_denklemleri", ((f"Varsayılan: {_word_raw(case, names.xc)}", "gd_varsayilan", "deger"),
                                                 (f"Seçiminiz: {_word_raw(case, x)}", "gd_secim", "deger")),
                            decimals=dn, heading="Katsayı",
                            title="Grup denklemleri: varsayılan model ve seçtiğiniz değişken (Tablo 11.1'deki roller)"),
            )
        return (
            CopyFrame("carpim", case.frame, "Etkileşim sütunu için verinin kopyası (özgün veri değişmez)"),
            Derive("carpim", "kukla_x", E.mul(E.var(dummy.column), E.var(x)),
                   f"Etkileşim terimi: {dummy.column} × {x} ({dummy.zero_raw} grubunda 0, {dummy.one_raw} grubunda X)"),
            ShowFrame("carpim", (dummy.column, x, "kukla_x"),
                      "Her gruptan ilk dört gözlem: çarpım 0 grubunda 0, 1 grubunda X'in kendisidir", rows=rows),
            OLS("m_etk", case.frame, names.ly, terms, f"Etkileşimli model: {_formula(names.ly, terms)}"),
            *_group_equations(case, "m_etk", "e2", dummy, x, others, dd, shown=False),
            *table,
            *_averages(case, case.frame, "ort2", others),
            *_group_lines("dogru2", info, "e2", dummy, others, "ort2"),
            LineChart("dogru2", info[0], "grup0", info[3], f"Tahmin edilen {_log_title(case)}",
                      f"{_data_prefix(case)}etkileşimli model: {dummy.one_raw} ve {dummy.zero_raw} "
                      f"{_forms(case)['loc_pl']} farklı {_word_raw(case, x)} eğimi{_held(case, others)}",
                      markers=False, series=(("grup1", _cap(case, dummy.one_raw)),), legend=_cap(case, dummy.zero_raw)),
        )

    def note(state, choices) -> str:
        x = choices["adim2_x"]
        s = state.scalars
        forms = _forms(case)
        dd = max(_d(s["e2_b1"]), _d(s["e2_g1"]))
        total = _slope_sum(s["e2_b1"], s["e2_g1"], s["e2_egim1"], dd)
        point = _zero_point(case, x)
        where = f"X = 0 noktasındaki ({point}) farktır" if point else "X = 0 noktasındaki farktır"
        return (f"{capital(dummy.zero)} {forms['gen']} {_word(case, x)} eğimi β̂₁ = {sayi(s['e2_b1'], dd)}; "
                f"{dummy.one} {forms['gen']}ki β̂₁ + γ̂₁ = {total}. Etkileşim katsayısı γ̂₁ = {sayi(s['e2_g1'], dd)} "
                f"{dummy.one} grubunun eğimi değil, iki eğim arasındaki farktır; kukla katsayısı γ̂₀ = "
                f"{sayi(s['e2_g0'], _d(s['e2_g0']))} da {dummy.one} grubunun sabiti değil, {where}. Çarpım sütunu "
                f"{dummy.zero} {forms['loc']} sıfır olduğu için etkileşim yalnız {dummy.one} grubunun eğimini değiştirir "
                "(§11.2).")

    checks = [_scalar(f"e2_{suffix}", label, d) for label, suffix in _group_rows(case, dummy, names.xc)]
    return _step(
        number=2,
        title="Etkileşim terimi ve grup denklemleri",
        note=NoteRef("11.2", 0, ("Tablo 11.1",)),
        explanation=(
            "Kukla ile nicel değişkenin çarpımı modele eklenir: $Y_i = \\beta_0 + \\beta_1 X_i + \\gamma_0 D_i + "
            "\\gamma_1 (D_i X_i) + u_i$. $D_i X_i$ referans grupta (D = 0) sıfır, diğer grupta (D = 1) $X_i$'dir. Böylece "
            "$D = 0$ grubunun denklemi $\\beta_0 + \\beta_1 X$, $D = 1$ grubununki $(\\beta_0 + \\gamma_0) + (\\beta_1 + "
            f"\\gamma_1) X$ olur: $\\gamma_0$ sabit farkı, $\\gamma_1$ eğim farkıdır. Kukla: {dummy.one} = 1, {dummy.zero} = "
            f"0. Model: {_model_text(case, default_terms, names.ly)}."
            + (" Etkileşimdeki nicel değişkeni değiştirin." if _live(control) else "")
        ),
        controls=(control,),
        build=build,
        checks=_stable(checks, exact),
        note_for=note,
    )


# --- Adım 3: dört olası regresyon yapısı ---------------------------------------------------------------------------

def _structures(case: Case) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Tablo 11.2'nin dört yapısı: (1) 1, X; (2) 1, X, D; (3) 1, X, DX; (4) 1, X, D, DX (kontroller her modelde). Yapı
    3'te etkileşim formülde X'ten önce yazılır: R terimi ``D:X`` olarak adlandırsın."""

    names = _names(case)
    dummy = _dummy(case).column
    controls = _controls(case)
    product = f"{dummy}:{names.xc}"
    return {
        "1": ("Aynı sabit, aynı eğim", (names.xc, *controls)),
        "2": ("Farklı sabit, aynı eğim", (dummy, names.xc, *controls)),
        "3": ("Aynı sabit, farklı eğim", (product, names.xc, *controls)),
        "4": ("Farklı sabit, farklı eğim", (dummy, names.xc, product, *controls)),
    }


def _step3(case: Case) -> LabStep:
    names = _names(case)
    dummy = _dummy(case)
    controls = _controls(case)
    structures = _structures(case)
    data = _data(case)
    product = f"{dummy.column}:{names.xc}"
    exact = _exact(case, data, names.ly, structures["4"][1])
    full = _fit(data, names.ly, structures["4"][1])
    d = _decimals(full, structures["4"][1])
    control = Choice("adim3_yapi", "Grafikte gösterilen yapı",
                     tuple((key, f"({key}) {title}") for key, (title, _) in structures.items()), "4",
                     help="Dört model her seçimde tahmin edilir; seçim yalnız grafikteki doğruları değiştirir.")
    heading = f"Dört yapı: bağımlı değişken {_log_title(case)}"
    if names.xc != case.roles[ACIKLAYICI]:
        heading += f"; {_word_raw(case, names.xc)} {level_text(_center(case))} etrafında merkezli"

    def build(choices) -> tuple:
        chosen = choices["adim3_yapi"]
        title, terms = structures[chosen]
        model = f"m_yapi{chosen}"
        values: list = [  # sayılar tablonun sütununda görünür; ölçü kutusu olarak tekrar edilmez
            ModelValue("y3_b0", model, "coef", "Sabit terim", term=INTERCEPT, decimals=d, shown=False),
            ModelValue("y3_b1", model, "coef", f"{_label(case, names.xc)} eğimi β̂₁ (referans grup; yapı 1–2'de iki "
                       "grupta ortak)", term=names.xc, decimals=d, shown=False),
            *(ModelValue(f"y3_k_{term}", model, "coef", f"{_title(case, term)} katsayısı", term=term, decimals=d,
                         shown=False) for term in controls),
        ]
        values.append(ModelValue("y3_g0", model, "coef", "Kukla katsayısı", term=dummy.column, decimals=d, shown=False)
                      if dummy.column in terms else
                      Scalar("y3_g0", E.const(0), "Sabit farkı yok (γ₀ = 0)", decimals=0, shown=False))
        values.append(ModelValue("y3_g1", model, "coef", "Etkileşim katsayısı", term=product, decimals=d, shown=False)
                      if product in terms else
                      Scalar("y3_g1", E.const(0), "Eğim farkı yok (γ₁ = 0)", decimals=0, shown=False))
        info = _grid_info(case, names.xc)
        return (
            *(OLS(f"m_yapi{key}", case.frame, names.ly, model_terms,
                  f"Yapı ({key}), {name.lower()}: {_formula(names.ly, model_terms)}")
              for key, (name, model_terms) in structures.items()),
            RegressionTable(tuple((f"({key}) {name}", f"m_yapi{key}") for key, (name, _) in structures.items()),
                            (INTERCEPT, dummy.column, names.xc, product, *controls), "tablo_yapi", heading,
                            decimals=4, exact=not case.own, adj_r2=True, r2_decimals=5),
            *values,
            *_averages(case, case.frame, "ort3", controls),
            *_group_lines("dogru3", info, "y3", dummy, controls, "ort3"),
            LineChart("dogru3", info[0], "grup0", info[3], f"Tahmin edilen {_log_title(case)}",
                      f"{_data_prefix(case)}yapı ({chosen}): {title.lower()}", markers=False,
                      series=(("grup1", _cap(case, dummy.one_raw)),), legend=_cap(case, dummy.zero_raw)),
        )

    def note(state, choices) -> str:
        chosen = choices["adim3_yapi"]
        table = state.tables["tablo_yapi"]
        headings = [f"({key}) {name}" for key, (name, _) in structures.items()]
        top = max(float(table.loc["adj_r2", heading]) for heading in headings)
        ties = [heading for heading in headings if top - float(table.loc["adj_r2", heading]) <= 1e-9]
        best = ties[0]
        point = _zero_point(case, names.xc) or f"{_word(case, names.xc)} sıfırken"
        p = float(state.models["m_yapi4"].pvalues[dummy.column])
        strong = "güçlü ve veriyle çelişen bir kısıttır" if not exact and p < 0.05 else "güçlü bir kısıttır"
        text = {
            "1": "Grup bilgisi modelde yoktur: iki grubun doğrusu aynıdır. ",
            "2": "Additif model: doğrular paraleldir, aralarındaki uzaklık kukla katsayısıdır. ",
            "3": (f"Kukla ana etkisi yoktur: iki doğru {point} aynı noktadan geçer. Hiyerarşi ilkesine aykırı bu model, bu "
                  f"noktada {_log_phrase(case)} farkı olmadığını varsayar; bu {strong}. "),
            "4": "Genel etkileşim modeli: hem başlangıç düzeyi hem eğim gruba göre değişebilir. ",
        }[chosen]
        if len(ties) > 1:
            text += (f"Düzeltilmiş R² en yüksek olan yapılar birbirinden ayırt edilemez ({listing(ties)}; "
                     f"{sayi(top, 5)}): aralarındaki farklar yuvarlama gürültüsüdür. ")
        else:
            text += f"Düzeltilmiş R² en yüksek olan yapı: {best} ({sayi(top, 5)}). "
        return (text + "Uyum ölçüsü tek başına yapı seçmez; etkileşimin iktisadi gerekçesi ve belirsizliği birlikte "
                "değerlendirilir (§11.3).")

    checks = [_check(f"Tablo: {heading}, {label}", TableTarget("tablo_yapi", row, heading), 5)
              for heading in (f"({key}) {name}" for key, (name, _) in structures.items())
              for row, label in (("r2", "R²"), ("adj_r2", "düzeltilmiş R²"))]
    return _step(
        number=3,
        title="Gruplar arasında dört olası regresyon yapısı",
        note=NoteRef("11.3", 0, ("Tablo 11.2", "Şekil 11.2")),
        explanation=(
            "İki grup ve bir nicel değişkenle dört yapı kurulabilir: (1) aynı sabit, aynı eğim: $1, X$; (2) farklı sabit, "
            "aynı eğim: $1, X, D$; (3) aynı sabit, farklı eğim: $1, X, DX$; (4) farklı sabit, farklı eğim: "
            "$1, X, D, DX$. Hiyerarşi ilkesi: $DX$ modeldeyse $D$ ve $X$ ana etkileri de tutulur; (3) $D$'yi "
            "çıkararak iki grubun $X = 0$ noktasında aynı sabite sahip olduğunu dayatır. Grafikteki yapıyı değiştirin."
        ),
        controls=(control,),
        build=build,
        checks=_stable(checks, exact, table="tablo_yapi"),
        note_for=note,
    )


# --- Adım 4: koşullu grup farkı ve merkezleme ----------------------------------------------------------------------

def _discrete(values: pd.Series) -> bool:
    values = values.astype(float)
    return bool(np.all(values == np.round(values))) and values.nunique() <= 25


def _center_control(case: Case) -> NumberChoice:
    if "center_control" in case.extra:
        return case.extra["center_control"]
    x = case.roles[ACIKLAYICI]
    return _number(case, "adim4_c", f"Merkezleme noktası c ({case.name(x)})",
                   help=f"Varsayılan c = {level_text(_center(case))}. c = 0 merkezlenmemiş modeldir: kukla katsayısı "
                        "temel açıklayıcı sıfırken farktır.")


def _step4(case: Case) -> LabStep:
    names = _names(case)
    dummy = _dummy(case)
    controls = _controls(case)
    x = case.roles[ACIKLAYICI]
    control = _center_control(case)
    default = float(control.default)
    data = _data(case)
    main_terms = _terms(dummy.column, names.xc, controls)
    exact = _exact(case, data, names.ly, main_terms)
    fit = _fit(data, names.ly, main_terms)
    d = _decimals(fit, (dummy.column, f"{dummy.column}:{names.xc}", names.xc))
    values = case.data[x].astype(float)
    discrete = _discrete(values)
    low, high = float(values.min()), float(values.max())

    def build(choices) -> tuple:
        center = float(choices["adim4_c"])
        shown = level_text(center)
        terms = _terms(dummy.column, names.xcc, controls)
        side: tuple = () if center == default else (  # varsayılan merkezleme seçilen noktayla yan yana
            OLS("m_merk_n", case.frame, names.ly, main_terms,
                f"Varsayılan model, c = {level_text(default)}: {_formula(names.ly, main_terms)}"),
            RegressionTable(((f"Varsayılan: c = {level_text(default)}", "m_merk_n"), (f"Seçiminiz: c = {shown}", "m_merk")),
                            tuple(dict.fromkeys((INTERCEPT, dummy.column, names.xc, names.xcc, f"{dummy.column}:{names.xc}",
                                                 f"{dummy.column}:{names.xcc}", *controls))),
                            "yan114", "Aynı model, iki merkezleme noktası: yalnız sabit ve kukla katsayısı değişir",
                            decimals=4, exact=not case.own, r2_decimals=6),
        )
        count: tuple = (Count(case.frame, "n_c", x, level_value(case, x, center),
                              f"{case.name(x)} değeri {shown} olan gözlem sayısı"),) if discrete else ()
        return (
            CopyFrame("merkez", case.frame, "Merkezleme için verinin kopyası (özgün veri değişmez)"),
            Derive("merkez", names.xcc, _shift(x, center) if center else E.var(x),
                   f"Merkezlenmiş {case.name(x)}: {_minus(x, center)}"),
            OLS("m_merk", "merkez", names.ly, terms, f"Etkileşimli model, c = {shown}: {_formula(names.ly, terms)}"),
            CoefficientTable("m_merk", (dummy.column, f"{dummy.column}:{names.xcc}"), "fark_c",
                             f"Kukla katsayısı: {case.name(x)} = {shown} noktasında log sonuç farkı; etkileşim: eğim farkı",
                             decimals=4, t_decimals=3, p_decimals=4, exact=not case.own),
            *count,
            ModelValue("g0_c", "m_merk", "coef", f"{case.name(x)} = {shown} noktasında log sonuç farkı γ̂₀",
                       term=dummy.column, decimals=d),
            ModelValue("g1_c", "m_merk", "coef", "Eğim farkı γ̂₁ (merkezlemeden bağımsız)",
                       term=f"{dummy.column}:{names.xcc}", decimals=d),
            ModelValue("b1_c", "m_merk", "coef", f"{_cap(case, dummy.zero_raw)} grubunun eğimi (merkezlemeden bağımsız)",
                       term=names.xcc, decimals=d),
            Scalar("tam_c", E.mul(100, E.sub(E.exp(E.ref("g0_c")), 1)), f"{case.name(x)} = {shown} noktasında tam yüzde "
                   "fark", decimals=2, percent=True),
            ModelValue("r2_c", "m_merk", "r2", "R² (merkezlemeden bağımsız)", decimals=6),
            *side,
        )

    def note(state, choices) -> str:
        center = float(choices["adim4_c"])
        shown = level_text(center)
        s = state.scalars
        word = phrase(case, x)
        unit = short_unit(case, x)
        text = (f"{capital(word)} {shown}{f' {unit}' if unit else ''} iken {dummy.one} ve {dummy.zero} "
                f"{_forms(case)['nom_pl']} arasındaki {_log_phrase(case)} farkı {sayi(s['g0_c'], _d(s['g0_c']))}; tam "
                f"yüzde fark: %{sayi(s['tam_c'], 2)}")
        if exact:
            text += ". "
        else:
            p = float(state.tables["fark_c"].loc[dummy.column, "p"])
            text += f" ({p_text(p, 4)}). "
        if discrete:
            count = int(s["n_c"])
            if count == 0:
                text += (f"Örneklemde {word} değeri {shown} olan gözlem yoktur: bu noktadaki fark tümüyle doğrusal model "
                         "varsayımına dayanır. ")
            elif count < FEW_OBSERVATIONS:
                text += (f"Örneklemde {word} değeri {shown} olan yalnız {count} gözlem vardır: bu noktadaki fark büyük "
                         "ölçüde doğrusal model varsayımına dayanır ve belirsizliği büyüktür. ")
        elif not low <= center <= high:
            text += (f"Bu nokta verideki aralığın ({level_text(low)}–{level_text(high)}) dışındadır: fark tümüyle doğrusal "
                     "model varsayımına dayanır (ekstrapolasyon). ")
        return (text + f"Eğim farkı ({sayi(s['g1_c'], _d(s['g1_c']))}), {dummy.zero} {_forms(case)['gen']} eğimi "
                f"({sayi(s['b1_c'], _d(s['b1_c']))}) ve R² ({sayi(s['r2_c'], 6)}) c ne olursa olsun aynıdır: merkezleme "
                "tahmin edilen değerleri değiştirmez, yalnız sabitin ve kukla katsayısının hangi düzeyde okunduğunu "
                "değiştirir. Kukla katsayısının standart hatası o noktadaki farkın standart hatasıdır (§11.4–11.5)."
                + (EXACT_NOTE if exact else ""))

    checks = (_scalar("g0_c", "Merkezleme noktasında log sonuç farkı", d), _scalar("tam_c", "Tam yüzde fark", 2),
              _scalar("g1_c", "Eğim farkı", d), _scalar("b1_c", "Referans grubun eğimi", d), _scalar("r2_c", "R²", 6))
    return _step(
        number=4,
        title="Koşullu grup farkı ve merkezleme",
        note=NoteRef("11.4", 0, ("§11.5",)),
        explanation=(
            "Etkileşimli modelde grup farkı $X$'e bağlıdır: $\\mathbb{E}(Y \\mid X, D = 1) - \\mathbb{E}(Y \\mid X, "
            "D = 0) = \\gamma_0 + \\gamma_1 X$ (§11.4). $\\gamma_0$ yalnız $X = 0$ noktasındaki farktır. $X$ anlamlı bir "
            "$c$ değeri etrafında merkezlenirse ($X^c = X - c$) kukla katsayısı farkı doğrudan $X = c$ noktasında verir; "
            f"tahmin edilen değerler, R² ve eğimler değişmez. Burada $X$: {phrase(case, x)}. Merkezleme noktasını "
            "değiştirin."
        ),
        controls=(control,),
        build=build,
        checks=checks,
        note_for=note,
    )


# --- Adım 5: ana uygulama -------------------------------------------------------------------------------------

def _dummy5(case: Case) -> Choice:
    return _dummy_control(case, "adim5_kukla", f"{_label(case, _names(case).xc)} ile etkileşen kukla",
                          f"Varsayılan: {_dummy(case).title.lower()} kuklası. Adım 6 aynı kuklayı kullanır.")


def _slope_rows(model: str, dummy: Dummy, prefix: str, case: Case, slope: str, decimals: tuple[int, int]) -> tuple:
    """Tablo 11.3: iki grubun eğimi, yaklaşık ve tam yüzde; ``prefix`` varsayılan model için "n"."""

    log_digits, percent_digits = decimals
    zero, one = _cap(case, dummy.zero_raw), _cap(case, dummy.one_raw)
    return (  # sayılar tabloda ve çıktıda görünür; ölçü kutusu olarak tekrar edilmez
        ModelValue(f"w{prefix}_b1", model, "coef", f"{zero}: eğim β̂₁", term=slope, decimals=log_digits, shown=False),
        ModelValue(f"w{prefix}_g1", model, "coef", "Eğim farkı γ̂₁", term=f"{dummy.column}:{slope}",
                   decimals=log_digits, shown=False),
        Scalar(f"w{prefix}_egim1", E.add(E.ref(f"w{prefix}_b1"), E.ref(f"w{prefix}_g1")), f"{one}: eğim β̂₁ + γ̂₁",
               decimals=log_digits, shown=False),
        *(Scalar(f"w{prefix}_yak{group}", E.mul(100, E.ref(name)), f"{label}: yaklaşık yüzde 100·eğim",
                 decimals=percent_digits, shown=False)
          for group, name, label in ((0, f"w{prefix}_b1", zero), (1, f"w{prefix}_egim1", one))),
        *(Scalar(f"w{prefix}_tam{group}", E.mul(100, E.sub(E.exp(E.ref(name)), 1)),
                 f"{label}: tam yüzde 100·(exp(eğim) − 1)", decimals=percent_digits, shown=False)
          for group, name, label in ((0, f"w{prefix}_b1", zero), (1, f"w{prefix}_egim1", one))),
    )


def _slope_table(case: Case, dummy: Dummy, prefix: str, title: str, decimals: tuple[int, int]) -> tuple:
    log_digits, percent_digits = decimals
    rows = ((_cap(case, dummy.zero_raw), 0), (_cap(case, dummy.one_raw), 1))
    return (
        ScalarTable(tuple((label, E.ref(f"w{prefix}_b1" if group == 0 else f"w{prefix}_egim1")) for label, group in rows),
                    f"egim113{prefix}", decimals=log_digits, heading="Grup"),
        ScalarTable(tuple((label, E.ref(f"w{prefix}_yak{group}")) for label, group in rows), f"yak113{prefix}",
                    decimals=percent_digits, heading="Grup"),
        ScalarTable(tuple((label, E.ref(f"w{prefix}_tam{group}")) for label, group in rows), f"tam113{prefix}",
                    decimals=percent_digits, heading="Grup"),
        JoinColumns(f"tablo113{prefix}", (("Log eğim", f"egim113{prefix}", "deger"),
                                          ("Yaklaşık yüzde", f"yak113{prefix}", "deger"),
                                          ("Tam yüzde", f"tam113{prefix}", "deger")),
                    decimals=percent_digits, heading="Grup", column_decimals=(("Log eğim", log_digits),), title=title),
    )


def _slope_digits(fit, dummy: str, slope: str) -> tuple[int, int]:
    first = float(fit.params[slope])
    second = first + float(fit.params[f"{dummy}:{slope}"])
    log_digits = max(_d(first), _d(second))
    percent_digits = max(2, digits_for(100 * first, 2), digits_for(100 * second, 2))
    return log_digits, percent_digits


def _step5(case: Case) -> LabStep:
    names = _names(case)
    dummies = _dummies(case)
    controls = _controls(case)
    control = _dummy5(case)
    default = control.default
    data = _data(case)
    info = _grid_info(case, names.xc)
    main_terms = _terms(default, names.xc, controls)
    exact = _exact(case, data, names.ly, main_terms)
    fit = _fit(data, names.ly, main_terms)
    digits = _slope_digits(fit, default, names.xc)
    d = _decimals(fit, main_terms)
    y = case.roles[SONUC]

    def build(choices) -> tuple:
        code = choices["adim5_kukla"]
        dummy = dummies[code]
        terms = _terms(code, names.xc, controls)
        same = code == default
        chosen = _slope_digits(_fit(data, names.ly, terms), code, names.xc)
        side: tuple = () if same else (  # varsayılan model seçilen kuklayla yan yana
            OLS("m_w_n", case.frame, names.ly, main_terms, f"Varsayılan model: {_formula(names.ly, main_terms)}"),
            *_slope_rows("m_w_n", dummies[default], "n", case, names.xc, digits),
            *_slope_table(case, dummies[default], "n", f"Varsayılan: {dummies[default].title.lower()} kuklası, grupların "
                          "eğimleri (Tablo 11.3'teki gibi)", digits),
        )
        return (
            OLS("m_w", case.frame, names.ly, terms, f"Etkileşimli model: {_formula(names.ly, terms)}"),
            ShowModel("m_w", "Etkileşimli modelin Python çıktısı (Kod 11.2'deki gibi)" if same
                      else "Etkileşimli modelin Python çıktısı (seçiminiz)", columns=("coef", "se", "t", "p"),
                      stats=("nobs", "r2", "adj_r2"), decimals=(("se", 4), ("p", 4)), exact=not case.own),
            *_slope_rows("m_w", dummy, "", case, names.xc, chosen),
            *_slope_table(case, dummy, "", (f"Grupların {_word_raw(case, names.xc)} eğimleri (Tablo 11.3'teki gibi)" if same
                                            else f"Seçiminiz: {dummy.title.lower()} kuklası, grupların eğimleri"), chosen),
            *side,
            *_coefficients(case, "m_w", "w5", code, names.xc, controls, shown=False, decimals=d),
            *_averages(case, case.frame, "ort5", controls),
            *_group_lines("cizgi5", info, "w5", dummy, controls, "ort5", log=True),
            LineChart("cizgi5", info[0], "duzey0", info[3], f"Tahmin edilen {case.name(y)}",
                      f"{_data_prefix(case)}etkileşimli model: {dummy.one_raw} ve {dummy.zero_raw} {_forms(case)['gen_pl']} "
                      f"tahmin edilen {_word_raw(case, y)} değeri{_held(case, controls)} (Şekil 11.3'teki gibi)",
                      markers=False, series=(("duzey1", _cap(case, dummy.one_raw)),), legend=_cap(case, dummy.zero_raw)),
        )

    def note(state, choices) -> str:
        dummy = dummies[choices["adim5_kukla"]]
        s = state.scalars
        forms = _forms(case)
        percent = max(2, digits_for(s["w_tam0"], 2), digits_for(s["w_tam1"], 2))

        def change(value: float) -> str:
            return f"yüzde {sayi(abs(value), percent)} daha {'yüksek' if value >= 0 else 'düşük'}"

        held = listing([_word(case, term) for term in controls])
        more = dict(case.extra.get("more", {})).get(names.xc) or f"{_word(case, names.xc)} değerinde bir birimlik artış"
        outcome = case.extra.get("outcome_with") or f"{phrase(case, y)} değeriyle"
        text = (f"{f'{capital(held)} sabitken ' if held else ''}{more}, {dummy.zero} {forms['loc']} yaklaşık "
                f"{change(s['w_tam0'])}, {dummy.one} {forms['loc']} yaklaşık {change(s['w_tam1'])} tahmin edilen {outcome} "
                "ilişkilidir (tam dönüşüm). ")
        if not text[0].isupper() and not text.startswith("“"):
            text = capital(text)
        if exact:
            text += "Uyum tam olduğu için etkileşimin p-değeri yuvarlama hatasıdır; karar yazılmaz. "
        else:
            p = float(state.models["m_w"].pvalues[f"{dummy.column}:{names.xc}"])
            text += (f"Etkileşim katsayısının p-değeri {sayi(p, 4) if p >= 0.00005 else '< 0,0001'}: yüzde 5 düzeyinde "
                     "eğimlerin eşit olduğu hipotezi "
                     + ("reddedilir. " if p < 0.05 else "reddedilemez; eğimler sayısal olarak farklı görünse de fark "
                        "örnekleme belirsizliğine göre küçüktür. "))
        causal = dict(case.extra.get("causal", {})).get(dummy.column, "grup üyeliğinin")
        return (text + "Şekildeki eğriler exp(tahmin edilen log sonuç) değerleridir"
                + (f"; {held} örneklem ortalamasındadır" if held else "")
                + f". Gözlemsel veriyle bu katsayılar {causal} nedensel etkisini ölçmez (§11.6).")

    log_digits, percent_digits = digits
    checks = [
        *(_check(f"Çıktı: {'sabit terim' if term == INTERCEPT else _title(case, term) if ':' not in term else 'etkileşim'}, "
                 f"{label}", CoefTarget("m_w", term, quantity), 3 if quantity == "t" else
                 _d(float(getattr(fit, attribute)[term])) if quantity != "p" else 4)
          for term in (INTERCEPT, *main_terms)
          for quantity, label, attribute in (("coef", "coef", "params"), ("se", "std err", "bse"), ("t", "t", "tvalues"),
                                             ("p", "P>|t|", "pvalues"))),
        _check("Çıktı: gözlem sayısı", ModelTarget("m_w", "nobs"), 0),
        _check("Çıktı: R²", ModelTarget("m_w", "r2"), 4),
        _check("Çıktı: düzeltilmiş R²", ModelTarget("m_w", "adj_r2"), 4),
        *(_check(f"Tablo: {group}, {column}", TableTarget("tablo113", group, column),
                 log_digits if column == "Log eğim" else percent_digits)
          for group in (_cap(case, dummies[default].zero_raw), _cap(case, dummies[default].one_raw))
          for column in ("Log eğim", "Yaklaşık yüzde", "Tam yüzde")),
    ]
    model = _model_text(case, main_terms, names.ly)
    return _step(
        number=5,
        title=case.extra.get("step5_title", "Ana uygulama: eğim gruplar arasında değişiyor mu?"),
        note=NoteRef("11.6", 0, ("Kod 11.1", "Kod 11.2", "Tablo 11.3", "Şekil 11.3")),
        explanation=(
            f"Model: {model}. {capital(_dummy(case).zero)} {_forms(case)['nom']} referans gruptur. Referans grubun "
            f"{_word(case, names.xc)} eğimi $\\beta_1$, diğer grubunki $\\beta_1 + \\gamma_1$; log eğim tam yüzdeye "
            "$100(e^{b} - 1)$ ile çevrilir."
            + (" Etkileşen kuklayı değiştirin: varsayılan model yan yana gösterilir." if _live(control) else "")
        ),
        controls=(control,),
        build=build,
        checks=_stable(checks, exact),
        note_for=note,
    )


# --- Adım 6: grup farkı açıklayıcının düzeyine bağlıdır -----------------------------------------------------------

def _gap_levels(case: Case) -> tuple[float, ...]:
    """Farklar tablosunun düzeyleri: alternatif örnekte 8, 12, 16, 17 yıl; kendi verinde yüzde 10'luk dilim, merkez,
    yüzde 90'lık dilim ve en büyük değer (yuvarlanmış)."""

    if "gap_levels" in case.extra:
        return tuple(float(value) for value in case.extra["gap_levels"])
    values = case.data[case.roles[ACIKLAYICI]].astype(float)
    candidates_ = (_spread_round(float(values.quantile(0.1)), values), _center(case),
                   _spread_round(float(values.quantile(0.9)), values), _spread_round(float(values.max()), values))
    return tuple(sorted({float(round(value, 10)) for value in candidates_}))


def _level_control(case: Case) -> NumberChoice:
    if "level_control" in case.extra:
        return case.extra["level_control"]
    x = case.roles[ACIKLAYICI]
    return _number(case, "adim6_duzey", f"Farkın hesaplandığı {case.name(x)} düzeyi",
                   help=f"Tablo birkaç düzeyi gösterir; ölçüler seçtiğiniz düzeydeki farktır. Varsayılan: merkezleme "
                        f"noktası ({level_text(_center(case))}).")


def _gap_scalars(level: float, prefix: str, source: str, center: float, decimals: int, shown: bool = False,
                 label: str = "") -> tuple:
    """``level`` düzeyinde log, yaklaşık ve tam yüzde fark; katsayılar ``{source}_g0`` ve ``{source}_g1``."""

    distance = round(level - center, 10)
    log = E.add(E.ref(f"{source}_g0"), E.mul(E.ref(f"{source}_g1"), int(distance) if float(distance).is_integer()
                                             else distance))
    shown_level = level_text(level)
    return (
        Scalar(f"{prefix}_log", log, f"{label} {shown_level}: log fark γ̂₀ + γ̂₁·({_minus(shown_level, center)})",
               decimals=decimals, shown=shown),
        Scalar(f"{prefix}_yak", E.mul(100, E.ref(f"{prefix}_log")), f"{label} {shown_level}: yaklaşık yüzde", decimals=2,
               shown=shown),
        Scalar(f"{prefix}_tam", E.mul(100, E.sub(E.exp(E.ref(f"{prefix}_log")), 1)), f"{label} {shown_level}: tam yüzde",
               decimals=2, shown=shown),
    )


def _gap_table(case: Case, source: str, suffix: str, title: str, decimals: int) -> tuple:
    """Tablo 11.4: düzeylerde log, yaklaşık ve tam yüzde fark; ``suffix`` varsayılan model için "n"."""

    levels = _gap_levels(case)
    center = _center(case)
    label = _label(case, case.roles[ACIKLAYICI])
    rows = tuple((level_text(level), f"fark{index}{suffix}") for index, level in enumerate(levels))
    return (
        *(item for index, level in enumerate(levels)
          for item in _gap_scalars(level, f"fark{index}{suffix}", source, center, decimals, label=label)),
        *(ScalarTable(tuple((row, E.ref(f"{name}_{part}")) for row, name in rows), f"{part}114{suffix}", decimals=2,
                      heading=_title(case, case.roles[ACIKLAYICI])) for part in ("log", "yak", "tam")),
        JoinColumns(f"tablo114{suffix}", (("Log fark", f"log114{suffix}", "deger"),
                                          ("Yaklaşık yüzde", f"yak114{suffix}", "deger"),
                                          ("Tam yüzde", f"tam114{suffix}", "deger")), decimals=2,
                    heading=_title(case, case.roles[ACIKLAYICI]), column_decimals=(("Log fark", decimals),), title=title),
    )


def _noise(value: float, regressor, outcome) -> bool:
    """Katsayı hesap hassasiyetinde sıfır mı: |β̂| · s_X ≤ 10⁻⁹ · s_Y (ör. tam uyumda gerçek değeri sıfır olan eğim
    farkı). Küçük ama gerçek bir katsayı sıfır sayılmaz."""

    spread_x, spread_y = float(np.std(regressor)), float(np.std(outcome))
    return value == 0 or abs(value) * spread_x <= 1e-9 * max(spread_y, np.finfo(float).tiny)


def _gap_direction(case: Case, g0: float, g1: float, low: float, high: float) -> str:
    """Log farkın açıklayıcıyla değişimi: işaret değiştirdiği nokta aralıktaysa o nokta yazılır."""

    word = phrase(case, case.roles[ACIKLAYICI])
    unit = short_unit(case, case.roles[ACIKLAYICI])
    unit = f" {unit}" if unit else ""
    data = _data(case)
    if _noise(g1, data[_names(case).xc], data[_names(case).ly]):
        return "γ̂₁ hesap hassasiyetinde sıfırdır: log fark düzeyle değişmez. "
    rising = "artar" if g1 > 0 else "azalır"
    zero = _center(case) - g0 / g1
    if low <= zero <= high:
        return (f"γ̂₁ = {sayi(g1, _d(g1))} olduğundan log fark {word} arttıkça {rising} ve {word} yaklaşık "
                f"{_number_text(zero)}{unit} iken işaret değiştirir: bu noktanın iki yanında farkın yönü terstir. ")
    grows = (g0 < 0) == (g1 < 0)
    return (f"γ̂₁ = {sayi(g1, _d(g1))} olduğundan log fark {word} arttıkça {rising}; grafiğin aralığında "
            f"({level_text(low)}–{level_text(high)}{unit}) işareti değişmez ve büyüklüğü "
            f"{'büyür' if grows else 'küçülür'}. ")


def _step6(case: Case) -> LabStep:
    names = _names(case)
    dummies = _dummies(case)
    controls = _controls(case)
    dummy_control = _dummy5(case)
    default = dummy_control.default
    control = _level_control(case)
    x = case.roles[ACIKLAYICI]
    center = _center(case)
    data = _data(case)
    main_terms = _terms(default, names.xc, controls)
    fit = _fit(data, names.ly, main_terms)
    d = max(_d(float(fit.params[default])), _d(float(fit.params[f"{default}:{names.xc}"])))
    _, grid, grid_center, axis = _grid_info(case, names.xc)
    column = _grid_info(case, names.xc)[0]
    low, high = _grid_bounds(grid)
    exact = _exact(case, data, names.ly, main_terms)

    def build(choices) -> tuple:
        code, level = choices["adim5_kukla"], float(choices["adim6_duzey"])
        dummy = dummies[code]
        terms_add = (code, names.xc, *controls)
        side: tuple = () if code == default else (  # varsayılan kuklanın tablosu yan yana (Adım 5'teki model)
            ModelValue("w6n_g0", "m_w_n", "coef", "Varsayılan: kukla katsayısı γ̂₀", term=default, decimals=d,
                       shown=False),
            ModelValue("w6n_g1", "m_w_n", "coef", "Varsayılan: etkileşim katsayısı γ̂₁", term=f"{default}:{names.xc}",
                       decimals=d, shown=False),
            *_gap_table(case, "w6n", "n", f"Varsayılan: {dummies[default].title.lower()} kuklası, düzeye göre fark", d),
        )
        unit = short_unit(case, x)
        vline = ((("merkez6", f"Merkezleme noktası ({level_text(center)}{f' {unit}' if unit else ''})"),)
                 if low <= center <= high else ())
        return (
            ModelValue("w6_g0", "m_w", "coef", f"Kukla katsayısı γ̂₀ ({case.name(x)} = {level_text(center)} noktasında "
                       "fark)", term=code, decimals=d),
            ModelValue("w6_g1", "m_w", "coef", "Etkileşim katsayısı γ̂₁", term=f"{code}:{names.xc}", decimals=d),
            *_gap_table(case, "w6", "", (f"{dummy.one_raw} − {dummy.zero_raw} log sonuç farkı, {_word_raw(case, x)} "
                                         "düzeyine göre (Tablo 11.4'teki gibi)") if code == default else
                        f"Seçiminiz: {dummy.one_raw} − {dummy.zero_raw} farkı, {_word_raw(case, x)} düzeyine göre", d),
            *side,
            *_gap_scalars(level, "secilen", "w6", center, d, shown=True, label=_label(case, x)),
            _grid_op("fark6", column, grid, axis),
            Derive("fark6", "log_fark", E.add(E.ref("w6_g0"), E.mul(E.ref("w6_g1"), _shift(column, grid_center))),
                   f"Log fark: γ̂₀ + γ̂₁·({column} − {level_text(grid_center)})" if grid_center else
                   "Log fark: γ̂₀ + γ̂₁·x"),
            Derive("fark6", "tam_fark", E.mul(100, E.sub(E.exp(E.var("log_fark")), 1)), "Tam yüzde fark"),
            Scalar("sifir6", E.const(0), "Fark yok", decimals=0, shown=False),
            Scalar("merkez6", E.const(int(center) if center.is_integer() else center), "Merkezleme noktası",
                   decimals=0 if center.is_integer() else 2, shown=False),
            LineChart("fark6", column, "tam_fark", axis, f"{_cap(case, dummy.one_raw)} − {dummy.zero_raw} tahmin edilen "
                      "fark (%)", f"{_data_prefix(case)}{dummy.one_raw} − {dummy.zero_raw} farkının "
                      f"{_word_raw(case, x)} düzeyiyle değişimi (Şekil 11.4'teki gibi)",
                      references=(("sifir6", "Fark yok"),), markers=False, vlines=vline),
            OLS("m_add6", case.frame, names.ly, terms_add, f"Additif model: {_formula(names.ly, terms_add)}"),
            *(ModelValue(name, model, stat, label, decimals=5, shown=False)  # sayılar açıklama metninde
              for name, model, stat, label in (("r2_add6", "m_add6", "r2", "Additif model: R²"),
                                               ("r2_int6", "m_w", "r2", "Etkileşimli model: R²"),
                                               ("adj_add6", "m_add6", "adj_r2", "Additif model: düzeltilmiş R²"),
                                               ("adj_int6", "m_w", "adj_r2", "Etkileşimli model: düzeltilmiş R²"))),
        )

    def note(state, choices) -> str:
        code, level = choices["adim5_kukla"], float(choices["adim6_duzey"])
        dummy = dummies[code]
        s = state.scalars
        unit = short_unit(case, x)
        return (f"{capital(phrase(case, x))} {level_text(level)}{f' {unit}' if unit else ''} iken log fark "
                f"{sayi(s['secilen_log'], _d(s['secilen_log']))}; yaklaşık "
                f"yüzde {sayi(s['secilen_yak'], 2)}, tam yüzde {sayi(s['secilen_tam'], 2)}. Etkileşimli modelde "
                f"{dummy.one}–{dummy.zero} log farkı {phrase(case, x)} ile doğrusal değişir; tam yüzde fark üstel dönüşüm "
                "nedeniyle doğrusal değişmez. " + _gap_direction(case, s["w6_g0"], s["w6_g1"], low, high)
                + f"Etkileşim eklenince R²: {sayi(s['r2_add6'], 5)} → {sayi(s['r2_int6'], 5)}; düzeltilmiş R²: "
                f"{sayi(s['adj_add6'], 5)} → {sayi(s['adj_int6'], 5)}. Etkileşim istatistiksel olarak anlamlı değilse "
                "düzeylere göre değişen farkları kesin bir örüntü gibi sunmak doğru olmaz (§11.6)."
                + (EXACT_NOTE if exact else ""))

    checks = [
        *(_check(f"Tablo: {level_text(level)}, {column_name}", TableTarget("tablo114", level_text(level), column_name),
                 d if column_name == "Log fark" else 2)
          for level in _gap_levels(case) for column_name in ("Log fark", "Yaklaşık yüzde", "Tam yüzde")),
        _scalar("secilen_tam", "Seçilen düzeyde tam yüzde fark", 2),
        _scalar("r2_add6", "Additif modelin R²'si", 5),
        _scalar("r2_int6", "Etkileşimli modelin R²'si", 5),
        _scalar("adj_add6", "Additif modelin düzeltilmiş R²'si", 5),
        _scalar("adj_int6", "Etkileşimli modelin düzeltilmiş R²'si", 5),
    ]
    dummy = _dummy(case)
    point = _zero_point(case, names.xc)
    return _step(
        number=6,
        title=f"Grup farkı {_word_raw(case, x)} düzeyine bağlıdır" if not case.own else "Grup farkı açıklayıcının "
                                                                                     "düzeyine bağlıdır",
        note=NoteRef("11.6", 0, ("Tablo 11.4", "Şekil 11.4")),
        explanation=(
            f"Etkileşimli modelde {dummy.one}–{dummy.zero} log farkı $\\hat\\gamma_0 + \\hat\\gamma_1 X^c$'dir ($X^c$: "
            f"merkezlenmiş {phrase(case, x)}); tam yüzde fark $100[\\exp(\\hat\\gamma_0 + \\hat\\gamma_1 X^c) - 1]$ "
            "(§11.4)."
            + (f" {capital(point)} $X^c = 0$ olduğundan kukla katsayısı doğrudan bu noktadaki farktır." if point else "")
            + " Etkileşimin uyuma katkısı additif modelle karşılaştırılır."
            + (" Adım 5'teki kukla seçimi bu adımı da belirler." if _live(dummy_control) else "")
            + " Farkın hesaplandığı düzeyi değiştirin."
        ),
        controls=(control,),
        uses=(dummy_control,),
        build=build,
        checks=_stable(checks, exact),
        note_for=note,
    )


# --- Adım 7: ikinci uygulama (düzey sonuç) ------------------------------------------------------------------------

def _house_setup(case: Case) -> tuple:
    """İkinci verinin yüklenmesi: alternatif örnekte KIELMC (1978) ve türetilen sütunlar; kendi verinde aynı dosya ayrı
    bir çerçevede, kukla kodu ve merkezlenmiş temel açıklayıcıyla (log sonuç gerekmez)."""

    house = _house(case)
    return (*house.load, *(_setup(case, house.frame, log=False) if case.own else ()))


def _influence(data: pd.DataFrame, outcome: str, terms: tuple[str, ...]) -> tuple | None:
    """Cook uzaklığı en büyük gözlem ve o gözlem olmadan etkileşim katsayısı: (sıra no, Cook uzaklığı, γ̂₁, p, γ̂₁
    gözlemsiz, p gözlemsiz). Gözlem çıkarılınca model kurulamıyorsa ``None``."""

    fit = _fit(data, outcome, terms)
    with np.errstate(divide="ignore", invalid="ignore"):
        cooks = np.asarray(fit.get_influence().cooks_distance[0], dtype=float)
    if not np.isfinite(cooks).any():
        return None
    index = int(np.nanargmax(np.where(np.isfinite(cooks), cooks, -np.inf)))
    rest = data.drop(index=data.index[index])
    if not _full_rank(rest, terms):
        return None
    product = terms[2]
    without = _fit(rest, outcome, terms)
    return (index + 1, float(cooks[index]), float(fit.params[product]), float(fit.pvalues[product]),
            float(without.params[product]), float(without.pvalues[product]))


def _fragile(found: tuple | None) -> bool:
    """Sonuç tek gözleme bağlı mı: gözlem çıkarılınca etkileşimin işareti ya da yüzde 5'teki anlamlılığı değişiyor."""

    if found is None:
        return False
    _, _, g1, p, without, p_without = found
    return (g1 > 0) != (without > 0) or (p < 0.05) != (p_without < 0.05)


def _step7_control(case: Case) -> Choice:
    house = _house(case)
    options = _house_options(case)
    dummy = _house_dummy(case)
    return Choice("adim7_x", f"{dummy.title} kuklasıyla etkileşen değişken",
                  tuple((term, _title(house, term)) for term in options), options[0],
                  help=f"Varsayılan: {_title(house, options[0])}. Adım 8 aynı seçimi kullanır.")


def _step7(case: Case) -> LabStep:
    house = _house(case)
    hy = _house_outcome(case)
    dummy = _house_dummy(case)
    options = _house_options(case)
    control = _step7_control(case)
    default = control.default
    data = _house_data(case)
    default_terms = _terms(dummy.column, default, _others(default, options))
    exact = _exact(case, data, hy, default_terms)
    fit = _fit(data, hy, default_terms)
    unit_y = short_unit(house, hy)
    found_all = {term: (None if exact else _influence(data, hy, _terms(dummy.column, term, _others(term, options))))
                 for term in options}

    def build(choices) -> tuple:
        x = choices["adim7_x"]
        terms = _terms(dummy.column, x, _others(x, options))
        same = x == default
        chosen = _fit(data, hy, terms)
        d = _decimals(chosen, terms)
        side: tuple = () if same else (  # varsayılan etkileşim seçilen değişkenle yan yana
            OLS("m_h_n", house.frame, hy, default_terms, f"Varsayılan model: {_formula(hy, default_terms)}"),
            RegressionTable(((f"Varsayılan: {dummy.title} × {_word_raw(house, default)}", "m_h_n"),
                             (f"Seçiminiz: {dummy.title} × {_word_raw(house, x)}", "m_h")),
                            (INTERCEPT, dummy.column, *options, f"{dummy.column}:{default}", f"{dummy.column}:{x}"),
                            "yan117", f"Varsayılan model ve seçtiğiniz etkileşim · bağımlı değişken: {display(house, hy)}",
                            decimals=4, exact=not case.own, adj_r2=True),
        )
        found = found_all[x]
        loo: tuple = ()
        if _fragile(found):
            row, cook, *_ = found
            product = f"{dummy.column}:{x}"
            loo = (
                CopyFrame("etkili", house.frame, "Etkili gözlemi çıkarmak için verinin kopyası (özgün veri değişmez)"),
                Derive("etkili", "sira_no", E.seq(E.var(hy)), "Gözlem sıra numarası: 1, 2, …, n"),
                Derive("etkili", "dahil", E.compare("ne", E.var("sira_no"), row), f"{row}. gözlem dışındakiler: 1"),
                TakeRows("etkisiz", "etkili", f"{row}. gözlem çıkarılır: tek başına katsayıları en çok değiştiren gözlem "
                         f"(Cook uzaklığı {sayi(cook, 2)})", where=("dahil", 1)),
                OLS("m_h_loo", "etkisiz", hy, terms, f"Aynı model, {row}. gözlem olmadan: {_formula(hy, terms)}"),
                ModelValue("loo_g1", "m_h_loo", "coef", f"{row}. gözlem olmadan: eğim farkı γ̂₁", term=product,
                           decimals=d),
                ModelValue("loo_p", "m_h_loo", "p", f"{row}. gözlem olmadan: γ̂₁'in p-değeri", term=product, decimals=4),
            )
        return (
            *_house_setup(case),
            OLS("m_h", house.frame, hy, terms, f"Etkileşimli model: {_formula(hy, terms)}"),
            ShowModel("m_h", "Etkileşimli modelin Python çıktısı (Kod 11.4'teki gibi)" if same
                      else "Etkileşimli modelin Python çıktısı (seçiminiz)", columns=("coef", "se", "t", "p"),
                      stats=("nobs", "r2", "adj_r2"), decimals=(("se", 4), ("p", 4)), exact=not case.own),
            *side,
            ModelValue("h_b1", "m_h", "coef", f"{_cap(house, dummy.zero_raw)}: {_word_raw(house, x)} eğimi β̂₁", term=x,
                       decimals=d),
            ModelValue("h_g1", "m_h", "coef", "Eğim farkı γ̂₁", term=f"{dummy.column}:{x}", decimals=d),
            ModelValue("h_g0", "m_h", "coef", f"{dummy.title} katsayısı γ̂₀", term=dummy.column, decimals=d),
            Scalar("h_egim1", E.add(E.ref("h_b1"), E.ref("h_g1")),
                   f"{_cap(house, dummy.one_raw)}: {_word_raw(house, x)} eğimi β̂₁ + γ̂₁", decimals=d),
            *loo,
        )

    def describe(row: int, x: str) -> str:
        column, _, _ = _raw(house, x)
        values = data.iloc[row - 1]
        parts = [f"{phrase(house, hy)} {_number_text(float(values[hy]))}{f' {unit_y}' if unit_y else ''}"]
        if column != hy:
            unit = short_unit(house, column)
            parts.append(f"{phrase(house, column)} {_number_text(float(values[column]))}{f' {unit}' if unit else ''}")
        parts.append(dummy.one if float(values[dummy.column]) == 1 else dummy.zero)
        return ", ".join(parts)

    def note(state, choices) -> str:
        x = choices["adim7_x"]
        s = state.scalars
        forms = _forms(house)
        word = _word(house, x)
        d = max(_d(s["h_b1"]), _d(s["h_g1"]))
        unit = short_unit(house, x)
        member = house.extra.get("member_gen", "gözlemin")
        outcome = dict(house.extra.get("possessive", {})).get(hy) if not case.own else None
        outcome = outcome or f"{phrase(house, hy)} değeri"
        amount = f"{sayi(abs(s['h_b1']), max(2, digits_for(s['h_b1'], 2)))}{f' {unit_y}' if unit_y else ' birim'}"
        text = (f"{capital(dummy.zero)} {forms['gen']} {word} eğimi {sayi(s['h_b1'], d)}: diğer değişkenler sabitken {word} "
                f"bir birim{f' ({unit})' if unit else ''} fazla olan {member} tahmin edilen {outcome} yaklaşık {amount} daha "
                f"{'yüksektir' if s['h_b1'] > 0 else 'düşüktür'}. {capital(dummy.one)} {forms['gen']} eğimi "
                f"{_slope_sum(s['h_b1'], s['h_g1'], s['h_egim1'], d)}. ")
        model = state.models["m_h"]
        if exact:
            text += "Uyum tam olduğu için p-değerleri yuvarlama hatasıdır; karar yazılmaz. "
        else:
            p = float(model.pvalues[f"{dummy.column}:{x}"])
            text += (f"Etkileşimin p-değeri {sayi(p, 4) if p >= 0.00005 else '< 0,0001'}: yüzde 5 düzeyinde eğimlerin eşit "
                     "olduğu hipotezi " + ("reddedilir. " if p < 0.05 else "reddedilemez. "))
        found = found_all[x]
        if "loo_g1" in s and _fragile(found):
            row = found[0]
            g1, without, p_without = s["h_g1"], s["loo_g1"], s["loo_p"]
            changes = []
            if (g1 > 0) != (without > 0):
                changes.append("işareti değişir")
            if (found[3] < 0.05) != (p_without < 0.05):
                changes.append("yüzde 5 düzeyinde anlamlı olur" if p_without < 0.05 else
                               "yüzde 5 düzeyinde anlamlı olmaz")
            text += (f"Bu sonuç tek bir gözleme çok duyarlıdır: tek başına çıkarıldığında tahminleri en çok değiştiren "
                     f"gözlem ({row}. gözlem: {describe(row, x)}) çıkarılıp model yeniden tahmin edilince etkileşim "
                     f"katsayısı {sayi(g1, d)} yerine {sayi(without, d)} olur ({p_text(p_without, 3)}): "
                     f"{listing(changes)}. ")
        point = _zero_point(house, x)
        g0 = sayi(s["h_g0"], _d(s["h_g0"]))
        p0 = float(model.pvalues[dummy.column])
        if point:
            text += (f"Kukla katsayısı ({g0}) yalnız {point}, diğer değişkenler sabitken iki grup arasındaki farktır"
                     + ("; anlamsız olması etkileşimin de anlamsız olduğunu göstermez" if not exact and p0 >= 0.05 else "")
                     + " (§11.7).")
        else:
            column, _, _ = _raw(house, x)
            values = data[column].astype(float)
            low, high = float(values.min()), float(values.max())
            unit = short_unit(house, column)
            span = f"{_point_label(low)}–{_point_label(high)}{f' {unit}' if unit else ''}"
            text += f"Kukla katsayısı ({g0}) {word} sıfırken iki grup arasındaki farktır; örneklemde {word} {span} aralığındadır"
            if low <= 0 <= high:
                text += ": sıfır bu aralığın içindedir (§11.5, §11.7)."
            else:
                text += (", bu nokta veri aralığının dışındadır ve katsayı tek başına yorumlanmaz: ekstrapolasyondur. Anlamlı "
                         "bir karşılaştırma noktası için değişken merkezlenir (§11.5, §11.7).")
        return text + (EXACT_NOTE if exact else "")

    checks = [
        *(_check(f"Çıktı: {'sabit terim' if term == INTERCEPT else _title(house, term) if ':' not in term else 'etkileşim'}, "
                 f"{label}", CoefTarget("m_h", term, quantity), 3 if quantity == "t" else 4 if quantity == "p"
                 else _d(float(getattr(fit, attribute)[term])))
          for term in (INTERCEPT, *default_terms)
          for quantity, label, attribute in (("coef", "coef", "params"), ("se", "std err", "bse"), ("t", "t", "tvalues"),
                                             ("p", "P>|t|", "pvalues"))),
        _check("Çıktı: gözlem sayısı", ModelTarget("m_h", "nobs"), 0),
        _check("Çıktı: R²", ModelTarget("m_h", "r2"), 4),
        _check("Çıktı: düzeltilmiş R²", ModelTarget("m_h", "adj_r2"), 4),
        _scalar("h_egim1", "Diğer grubun eğimi β̂₁ + γ̂₁", _decimals(fit, default_terms)),
    ]
    if _fragile(found_all[default]):
        checks += [_scalar("loo_g1", "Etkili gözlem olmadan eğim farkı", _decimals(fit, default_terms)),
                   _scalar("loo_p", "Etkili gözlem olmadan p-değeri", 4)]
    lead = house.extra.get("house_text") or (
        "Notlarda ikinci bir veriyle yapılan bu adım aynı dosyanın düzey sonucuyla kurulur (log alınmaz).")
    return _step(
        number=7,
        title=house.extra.get("step7_title", "İkinci uygulama: eğim gruplar arasında değişiyor mu?"),
        note=NoteRef("11.7", 0, ("Kod 11.3", "Kod 11.4")),
        explanation=(
            f"{lead} Model: {_model_text(house, default_terms, hy)}. {capital(dummy.zero)} {_forms(house)['nom']} "
            "referans gruptur. Etkileşimin işareti ya da anlamlılığı tek bir gözleme bağlıysa o gözlem olmadan da model "
            "kurulur (notlardaki en büyük arsa gibi)."
            + (" Kukla ile etkileşen değişkeni değiştirin: varsayılan model yan yana gösterilir." if _live(control) else "")
        ),
        controls=(control,),
        build=build,
        checks=_stable(checks, exact),
        note_for=note,
    )


# --- Adım 8: ikinci uygulamada farkın değişimi ---------------------------------------------------------------------

def _house_points(case: Case, x: str) -> tuple[tuple[float, float], ...]:
    """Farkın hesaplandığı noktalar: (özgün ölçekte değer, modeldeki değer)."""

    house = _house(case)
    known = dict(house.extra.get("points", {}))
    if x in known:
        return tuple(known[x])
    column, scale, offset = _raw(house, x)
    if column == case.roles[ACIKLAYICI] and offset == _center(case):
        levels = _gap_levels(case)
    else:
        values = _house_data(case)[column].astype(float)
        levels = tuple(sorted({float(round(_spread_round(float(values.quantile(share)), values), 10))
                               for share in (0.1, 0.5, 0.9)}))
    return tuple((level, round((level - offset) / scale, 10)) for level in levels)


def _point_heading(case: Case, x: str) -> str:
    house = _house(case)
    known = dict(house.extra.get("point_headings", {}))
    if x in known:
        return known[x]
    column, _, _ = _raw(house, x)
    return display(house, column)


def _gap_points(case: Case, x: str, model: str, prefix: str, result: str, title: str, decimals: int) -> tuple:
    """Tablo 11.5: seçilen noktalarda fark γ̂₀ + γ̂₁·x (``model``in katsayılarıyla; sayılar tabloda)."""

    house = _house(case)
    dummy = _house_dummy(case)
    points = _house_points(case, x)
    unit = short_unit(house, _house_outcome(case))

    def value(number: float):
        return int(number) if float(number).is_integer() else number

    return (
        ModelValue(f"{prefix}_g0", model, "coef", f"{dummy.title} katsayısı γ̂₀", term=dummy.column, decimals=4,
                   shown=False),
        ModelValue(f"{prefix}_g1", model, "coef", "Eğim farkı γ̂₁", term=f"{dummy.column}:{x}", decimals=4, shown=False),
        *(Scalar(f"{prefix}_{index}", E.add(E.ref(f"{prefix}_g0"), E.mul(E.ref(f"{prefix}_g1"), value(model_value))),
                 f"{_label(house, x)} {_point_label(level)}: γ̂₀ + γ̂₁·x", decimals=decimals, shown=False)
          for index, (level, model_value) in enumerate(points)),
        ScalarTable(tuple((_point_label(level), E.ref(f"{prefix}_{index}")) for index, (level, _) in enumerate(points)),
                    result, decimals=decimals, heading=_point_heading(case, x),
                    value=f"{_cap(house, dummy.one_raw)} − {dummy.zero_raw} farkı" + (f" ({unit})" if unit else ""),
                    title=title),
    )


def _point_digits(case: Case, x: str) -> int:
    house = _house(case)
    dummy = _house_dummy(case)
    options = _house_options(case)
    fit = _fit(_house_data(case), _house_outcome(case), _terms(dummy.column, x, _others(x, options)))
    g0, g1 = float(fit.params[dummy.column]), float(fit.params[f"{dummy.column}:{x}"])
    values = [g0 + g1 * model_value for _, model_value in _house_points(case, x)]
    del house
    return max([2] + [digits_for(value, 2) for value in values if value != 0])


def _step8(case: Case) -> LabStep:
    house = _house(case)
    hy = _house_outcome(case)
    dummy = _house_dummy(case)
    options = _house_options(case)
    control = _step7_control(case)
    default = control.default
    data = _house_data(case)
    exact = _exact(case, data, hy, _terms(dummy.column, default, _others(default, options)))
    unit_y = short_unit(house, hy)
    counts_spec = house.extra.get("counts")

    def build(choices) -> tuple:
        x = choices["adim7_x"]
        others = _others(x, options)
        info = _grid_info(house, x)
        column, grid, center, axis = info
        terms_add = (dummy.column, *options)
        d = _point_digits(case, x)
        line_values = tuple(  # doğruların katsayıları Adım 7'deki çıktıda görünür; ölçü kutusu olarak tekrar edilmez
            ModelValue(name, "m_h", "coef", label, term=term, decimals=4, shown=False)
            for name, label, term in (("h8_b0", "Sabit terim", INTERCEPT), ("h8_b1", f"{_label(house, x)} eğimi", x),
                                      ("h8_g0", f"{dummy.title} katsayısı", dummy.column),
                                      ("h8_g1", "Eğim farkı", f"{dummy.column}:{x}"),
                                      *((f"h8_k_{term}", f"{_title(house, term)} katsayısı", term) for term in others)))
        counts: tuple = ()
        if counts_spec and counts_spec[0] == x:
            _, raw_column, low, high, unit, _ = counts_spec
            counts = (
                CopyFrame("aralik_sayim", house.frame, "Aralıktaki gözlemleri saymak için verinin kopyası (özgün veri "
                          "değişmez)"),
                Derive("aralik_sayim", "aralik_ici", E.mul(E.compare("ge", E.var(raw_column), low),
                                                           E.compare("le", E.var(raw_column), high)),
                       f"Gösterge: {raw_column} {sayim(low)}–{sayim(high)} {unit} aralığında (1) ya da değil (0)"),
                Derive("aralik_sayim", "aralik_alti", E.compare("lt", E.var(raw_column), low),
                       f"Gösterge: {raw_column} {sayim(low)} {unit}'den küçük (1) ya da değil (0)"),
                Derive("aralik_sayim", "aralik_ustu", E.compare("gt", E.var(raw_column), high),
                       f"Gösterge: {raw_column} {sayim(high)} {unit}'den büyük (1) ya da değil (0)"),
                Count("aralik_sayim", "n_aralik", "aralik_ici", 1, f"{sayim(low)}–{sayim(high)} {unit} aralığındaki gözlem"),
                Count("aralik_sayim", "n_alti", "aralik_alti", 1, f"{sayim(low)} {unit}'den küçük gözlem"),
                Count("aralik_sayim", "n_ustu", "aralik_ustu", 1, f"{sayim(high)} {unit}'den büyük gözlem"),
            )
        side: tuple = () if x == default else _gap_points(  # varsayılan etkileşimin tablosu (Adım 7'deki model)
            case, default, "m_h_n", "hfarkn", "tablo115n", f"Varsayılan: {dummy.title} × {_word_raw(house, default)}, "
            "noktalara göre fark", _point_digits(case, default))
        low, high = _grid_bounds(grid)
        raw_column, _, offset = _raw(house, x)
        raw_unit = short_unit(house, raw_column)
        vline = ((("merkez8", f"Merkezleme noktası ({_point_label(offset)}{f' {raw_unit}' if raw_unit else ''})"),)
                 if _zero_point(house, x) and center and low <= center <= high else ())
        return (
            *_gap_points(case, x, "m_h", "hfark", "tablo115",
                         (f"{_cap(house, dummy.one_raw)} − {dummy.zero_raw} farkı, {_word_raw(house, x)} düzeyine göre "
                          "(Tablo 11.5'teki gibi)") if x == default else
                         f"Seçiminiz: {dummy.one_raw} − {dummy.zero_raw} farkı, {_word_raw(house, x)} düzeyine göre", d),
            *side,
            *counts,
            *line_values,
            *_averages(house, house.frame, "ort8", others),
            *_group_lines("cizgi8", info, "h8", dummy, others, "ort8"),
            LineChart("cizgi8", column, "grup0", axis, f"Tahmin edilen {display(house, hy)}",
                      f"{_data_prefix(house)}gruplara göre {_word_raw(house, x)} eğimleri (Şekil 11.5'teki gibi)",
                      markers=False, series=(("grup1", _cap(house, dummy.one_raw)),), legend=_cap(house, dummy.zero_raw)),
            Derive("cizgi8", "grup_farki", E.sub(E.var("grup1"), E.var("grup0")),
                   f"{dummy.one_raw} − {dummy.zero_raw} farkı"),
            Scalar("sifir8", E.const(0), "Fark yok", decimals=0, shown=False),
            Scalar("merkez8", E.const(int(center) if float(center).is_integer() else center), "Merkezleme noktası",
                   decimals=0 if float(center).is_integer() else 2, shown=False),
            LineChart("cizgi8", column, "grup_farki", axis,
                      f"{_cap(house, dummy.one_raw)} − {dummy.zero_raw} farkı" + (f" ({unit_y})" if unit_y else ""),
                      f"{_data_prefix(house)}grup farkının {_word_raw(house, x)} düzeyiyle değişimi (Şekil 11.6'daki gibi)",
                      references=(("sifir8", "Fark yok"),), markers=False, vlines=vline),
            OLS("m_h_add", house.frame, hy, terms_add, f"Additif model: {_formula(hy, terms_add)}"),
            *(ModelValue(name, model, stat, label, decimals=4, shown=False)  # sayılar açıklama metninde
              for name, model, stat, label in (("r2_hadd", "m_h_add", "r2", "Additif model: R²"),
                                               ("r2_hint", "m_h", "r2", "Etkileşimli model: R²"),
                                               ("adj_hadd", "m_h_add", "adj_r2", "Additif model: düzeltilmiş R²"),
                                               ("adj_hint", "m_h", "adj_r2", "Etkileşimli model: düzeltilmiş R²"))),
        )

    def note(state, choices) -> str:
        x = choices["adim7_x"]
        s = state.scalars
        word = _word(house, x)
        point = _zero_point(house, x)
        column, scale, offset = _raw(house, x)
        values = data[column].astype(float)
        low, high = float(values.min()), float(values.max())
        unit = short_unit(house, column)
        g0, g1 = s["h8_g0"], s["h8_g1"]
        amount = f"{f' {unit_y}' if unit_y else ''}"
        flat = _noise(g1, data[x], data[hy])
        if point:
            text = f"{capital(point)} fark {sayi(g0, max(2, digits_for(g0, 2)))}{amount}; "
            if flat:
                text += "γ̂₁ hesap hassasiyetinde sıfırdır: fark düzeyle değişmez"
            else:
                text += f"{word} arttıkça fark {'artar' if g1 > 0 else 'azalır'}"
                crossing = scale * (-g0 / g1) + offset
                if low <= crossing <= high:
                    text += (f" ve {word} yaklaşık {_number_text(crossing)}{f' {unit}' if unit else ''} düzeyinde "
                             "işaret değiştirir")
                else:
                    text += "; veri aralığında işareti değişmez"
            text += ". "
        else:
            span = f"{_point_label(low)}–{_point_label(high)}{f' {unit}' if unit else ''}"
            text = (f"Fark seçilen değişkenle doğrusal değişir: γ̂₀ + γ̂₁·x; tablodaki noktalar örneklemdeki aralığın "
                    f"({span}) içindedir. ")
        if counts_spec and counts_spec[0] == x and "n_aralik" in s:
            _, _, lo, hi, unit_text, member = counts_spec
            text += (f"Grafikler {sayim(lo)}–{sayim(hi)} {unit_text} aralığındadır: "
                     f"{sayim(state.models['m_h'].nobs)} {member} {sayim(s['n_aralik'])} tanesi bu aralıktadır; "
                     f"{sayim(lo)} {unit_text} altında {sayim(s['n_alti'])}, {sayim(hi)} {unit_text} üstünde "
                     f"{sayim(s['n_ustu'])} {house.extra.get('unit_word', 'gözlem')} vardır. ")
        return (text + f"Etkileşim eklenince R²: {sayi(s['r2_hadd'], 4)} → {sayi(s['r2_hint'], 4)}; düzeltilmiş R²: "
                f"{sayi(s['adj_hadd'], 4)} → {sayi(s['adj_hint'], 4)}. Tablodaki farklar modelin tahminleridir; her "
                "noktadaki farkın standart hatası hesaplanmadan ‘anlamlı fark vardır’ denmez ve veri aralığının dışındaki "
                "noktalar ekstrapolasyon riski taşır (§11.7)." + (EXACT_NOTE if exact else ""))

    d = _point_digits(case, default)
    checks = [
        *(_check(f"Tablo: {_point_label(level)}", TableTarget("tablo115", _point_label(level), "deger"), d)
          for level, _ in _house_points(case, default)),
        _scalar("r2_hadd", "Additif modelin R²'si", 4),
        _scalar("r2_hint", "Etkileşimli modelin R²'si", 4),
        _scalar("adj_hadd", "Additif modelin düzeltilmiş R²'si", 4),
        _scalar("adj_hint", "Etkileşimli modelin düzeltilmiş R²'si", 4),
    ]
    if counts_spec and counts_spec[0] == default:
        checks += [_scalar("n_aralik", "Grafiğin aralığındaki gözlem", 0), _scalar("n_alti", "Aralığın altındaki gözlem", 0),
                   _scalar("n_ustu", "Aralığın üstündeki gözlem", 0)]
    return _step(
        number=8,
        title=house.extra.get("step8_title", "Grup farkı etkileşen değişkene bağlıdır"),
        note=NoteRef("11.7", 0, ("Tablo 11.5", "Şekil 11.5", "Şekil 11.6")),
        explanation=(
            "Etkileşimli modelde iki grup arasındaki fark $\\hat\\gamma_0 + \\hat\\gamma_1 X$ ile $X$'e göre değişir; kukla "
            "katsayısı yalnız $X = 0$ noktasındaki farktır. Farklar modelin tahminleridir: her noktadaki farkın standart "
            "hatası ayrıca hesaplanmadan ‘anlamlı fark vardır’ denmez; veri aralığının dışı ekstrapolasyondur."
            + (" Adım 7'deki seçim bu adımı da belirler." if _live(control) else "")
        ),
        uses=(control,),
        build=build,
        checks=_stable(checks, exact),
        note_for=note,
    )


# --- Adım 9: etkileşimli modellerde hipotez testleri ---------------------------------------------------------------

def _test_names(case: Case) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    """İki uygulamanın (seçenek etiketi, tablo satırı öneki, metindeki yeri)."""

    if "test_names" in case.extra:
        first, second_ = case.extra["test_names"]
        return tuple(first), tuple(second_)
    return (("Ana model (log sonuç)", "Ana model", "Ana modelde (log sonuç)"),
            ("İkinci model (düzey sonuç)", "İkinci model", "İkinci modelde (düzey sonuç)"))


def _decision(p: float) -> str:
    if p < 0.05:
        return "reddedilir"
    if p < 0.10:
        return "yüzde 5'te reddedilmez, yüzde 10'da reddedilir"
    return "reddedilmez"


def _f(value: float) -> str:
    if not math.isfinite(value) or abs(value) >= 1e6:
        return "F > 10⁶"
    return f"F = {sayi(value, 3)}"


def _step9(case: Case) -> LabStep:
    names = _names(case)
    house = _house(case)
    hy = _house_outcome(case)
    dummy, house_dummy = _dummy(case), _house_dummy(case)
    controls = _controls(case)
    options = _house_options(case)
    hx = options[0]
    main_terms = _terms(dummy.column, names.xc, controls)
    house_terms = _terms(house_dummy.column, hx, _others(hx, options))
    (label_w, row_w, where_w), (label_h, row_h, where_h) = _test_names(case)
    exact_w = _exact(case, _data(case), names.ly, main_terms)
    exact_h = _exact(case, _house_data(case), hy, house_terms)
    data_control = Choice("adim9_veri", "Grafikteki uygulama", (("ana", label_w), ("ikinci", label_h)), "ana",
                          help="Tablo iki uygulamanın testlerini gösterir; seçim yalnız grafiği değiştirir.")
    test_control = Choice("adim9_test", "Grafikteki hipotez", (("egim", "Eğimler eşit: γ₁ = 0"),
                                                             ("ortak", "İki doğru aynı: γ₀ = γ₁ = 0")), "ortak",
                          help="Tek kısıt t testiyle (F = t²), iki kısıt ortak F testiyle sınanır.")

    def build(choices) -> tuple:
        data, test = choices["adim9_veri"], choices["adim9_test"]
        specs = (("w", "m_t_w", (dummy.column, f"{dummy.column}:{names.xc}"), row_w),
                 ("h", "m_t_h", (house_dummy.column, f"{house_dummy.column}:{hx}"), row_h))
        operations: list = [
            OLS("m_t_w", case.frame, names.ly, main_terms, f"{row_w}: {_formula(names.ly, main_terms)}"),
            OLS("m_t_h", house.frame, hy, house_terms, f"{row_h}: {_formula(hy, house_terms)}"),
        ]
        for key, model, (main, product), name in specs:
            operations += [  # F ve p tabloda; ölçü kutuları t ve serbestlik derecesi
                JointTest(f"F_egim_{key}", f"p_egim_{key}", model, (product,),
                          f"{name}: eğimler eşit, H₀: γ₁ = 0 (tek kısıt)", decimals=3, p_decimals=4, shown=False),
                ModelValue(f"t_egim_{key}", model, "t", f"{name}: etkileşimin t istatistiği", term=product, decimals=3),
                Scalar(f"t2_egim_{key}", E.power(E.ref(f"t_egim_{key}"), 2), f"{name}: t² (= F)", decimals=3,
                       shown=False),
                JointTest(f"F_ortak_{key}", f"p_ortak_{key}", model, (main, product),
                          f"{name}: iki doğru aynı, H₀: γ₀ = 0, γ₁ = 0", decimals=3, p_decimals=4, shown=False),
                ModelValue(f"sd_{key}", model, "df_resid", f"{name}: payda serbestlik derecesi n − k − 1", decimals=0),
            ]
        rows = ((f"{row_w} · H₀: γ₁ = 0", "egim_w"), (f"{row_w} · H₀: γ₀ = γ₁ = 0", "ortak_w"),
                (f"{row_h} · H₀: γ₁ = 0", "egim_h"), (f"{row_h} · H₀: γ₀ = γ₁ = 0", "ortak_h"))
        operations += [
            ScalarTable(tuple((label, E.ref(f"F_{name}")) for label, name in rows), "f118", decimals=3, heading="Test"),
            ScalarTable(tuple((label, E.ref(f"p_{name}")) for label, name in rows), "p118", decimals=4, heading="Test"),
            ScalarTable(tuple((label, E.const(1 if name.startswith("egim") else 2)) for label, name in rows), "q118",
                        decimals=0, heading="Test"),
            JoinColumns("tablo118", (("F", "f118", "deger"), ("p", "p118", "deger"), ("Kısıt sayısı q", "q118", "deger")),
                        decimals=3, heading="Test", column_decimals=(("Kısıt sayısı q", 0), ("p", 4)), p_columns=("p",),
                        title="Etkileşim testleri: eğimlerin eşitliği ve iki doğrunun bütünüyle aynı olması"),
        ]
        key = "w" if data == "ana" else "h"
        name = row_w if key == "w" else row_h
        if test == "egim":  # tek kısıt: t testi (F = t²)
            operations.append(HypothesisPlot("t", f"t_egim_{key}", f"sd_{key}", f"{name}: H₀: γ₁ = 0, t testi",
                                             "t değeri"))
        else:
            operations += [
                Scalar("q9", E.const(2), "Kısıt sayısı q", decimals=0, shown=False),
                HypothesisPlot("f", f"F_ortak_{key}", "q9", f"{name}: H₀: γ₀ = γ₁ = 0, ortak F testi", "F değeri",
                               alternative="sag", df2=f"sd_{key}"),
            ]
        return tuple(operations)

    def note(state, choices) -> str:
        s = state.scalars
        text = ""
        for key, where, exact in (("w", where_w, exact_w), ("h", where_h, exact_h)):
            if exact:
                text += f"{where} uyum tam olduğu için F ve p yuvarlama hatasıdır; karar yazılmaz. "
                continue
            text += (f"{where} eğimlerin eşit olduğu hipotezi {_decision(s[f'p_egim_{key}'])} ({_f(s[f'F_egim_{key}'])}, "
                     f"{p_text(s[f'p_egim_{key}'], 4)}); iki doğrunun bütünüyle aynı olduğu hipotezi "
                     f"{_decision(s[f'p_ortak_{key}'])} ({_f(s[f'F_ortak_{key}'])}, {p_text(s[f'p_ortak_{key}'], 4)}). ")
        text += ("Tekli test yalnız γ₁'i, ortak test γ₀ ve γ₁'i birlikte sınar: iki testin farklı sonuç vermesi çelişki "
                 "değildir. ")
        if not exact_w:
            text += f"Tek kısıtta F = t² ({row_w}: ({sayi(s['t_egim_w'], 3)})² = {sayi(s['t2_egim_w'], 3)}). "
        return text + "Test araştırma sorusuna göre seçilir (§11.8)." + (EXACT_NOTE if exact_w or exact_h else "")

    checks = [_scalar(name, label, decimals) for name, label, decimals in (
        ("F_egim_w", f"{row_w}: eğim testi F", 3), ("p_egim_w", f"{row_w}: eğim testi p", 4),
        ("F_ortak_w", f"{row_w}: ortak test F", 3), ("p_ortak_w", f"{row_w}: ortak test p", 4),
        ("F_egim_h", f"{row_h}: eğim testi F", 3), ("p_egim_h", f"{row_h}: eğim testi p", 4),
        ("F_ortak_h", f"{row_h}: ortak test F", 3), ("p_ortak_h", f"{row_h}: ortak test p", 4),
        ("t2_egim_w", f"{row_w}: t²", 3))]
    return _step(
        number=9,
        title="Etkileşimli modellerde hipotez testleri",
        note=NoteRef("11.8", 0),
        explanation=(
            "Araştırma sorusuna göre üç hipotez ayrılır: eğimler eşit mi ($H_0: \\gamma_1 = 0$; tek kısıt, $t$ ya da "
            "$F = t^2$), seçilen referans noktasında fark var mı ($H_0: \\gamma_0 = 0$), iki grubun regresyon doğrusu "
            "bütünüyle aynı mı ($H_0: \\gamma_0 = 0,\\ \\gamma_1 = 0$; iki kısıt, ortak $F$ testi). Grafikteki "
            "uygulamayı ve testi değiştirin."
        ),
        controls=(data_control, test_control),
        build=build,
        checks=_stable(checks, exact_w or exact_h),
        note_for=note,
    )


# --- Adım 10: formül yazımı ve makale tablosu ------------------------------------------------------------------------

def _step10(case: Case) -> LabStep:
    names = _names(case)
    dummy = _dummy(case)
    controls = _controls(case)
    product = f"{dummy.column}:{names.xc}"
    formulas = {"acik": (dummy.column, names.xc, *controls, product), "iki_nokta": (product, *controls)}
    control = Choice(
        "adim10_formul", "Üçüncü sütundaki formül",
        (("yok", "Üçüncü sütun yok (Tablo 11.6'daki gibi)"),
         ("acik", " + ".join(formulas["acik"])),
         ("iki_nokta", " + ".join(formulas["iki_nokta"]) + " (ana etkiler yok)")), "yok",
        help=f"{dummy.column} * {names.xc}, ana etkileri ve etkileşimi birlikte ekler; {product} yalnız çarpımı ekler.")
    main_terms = _terms(dummy.column, names.xc, controls)
    data = _data(case)
    exact = _exact(case, data, names.ly, main_terms)
    terms_add = (dummy.column, names.xc, *controls)

    def build(choices) -> tuple:
        variant = choices["adim10_formul"]
        models: tuple = (("(1) Additif model", "m_s1"), ("(2) Etkileşimli model", "m_s2"))
        extra: tuple = ()
        if variant != "yok":
            terms = formulas[variant]
            extra = (OLS("m_s3", case.frame, names.ly, terms, f"Sütun (3): {_formula(names.ly, terms)}"),)
            models += (("(3) Seçiminiz", "m_s3"),)
        return (
            OLS("m_s1", case.frame, names.ly, terms_add, f"Sütun (1): {_formula(names.ly, terms_add)}"),
            OLS("m_s2", case.frame, names.ly, main_terms, f"Sütun (2): {_formula(names.ly, main_terms)}"),
            *extra,
            RegressionTable(models, (dummy.column, names.xc, product, *controls), "tablo116",
                            ("Etkileşim sonuçlarının makale tipi tabloda gösterimi (Tablo 11.6'daki gibi)" if variant == "yok"
                             else "Makale tablosu ve seçtiğiniz formül") + f" · bağımlı değişken: {_log_title(case)}",
                            decimals=4, exact=not case.own, adj_r2=True),
        )

    def note(state, choices) -> str:
        variant = choices["adim10_formul"]
        table = state.tables["tablo116"]
        slope = float(table.loc[names.xc, "(2) Etkileşimli model"])
        inter = float(table.loc[product, "(2) Etkileşimli model"])
        point = _zero_point(case, names.xc) or f"{_word(case, names.xc)} sıfırken"
        members = case.extra.get("members")
        unit = short_unit(case, case.roles[ACIKLAYICI])
        text = (f"Tabloyu okuma sırası: referans grup {dummy.zero} " + (f"{members}lardır" if members else "grubudur")
                + (f" ve {_word(case, names.xc)} {level_text(_center(case))}{f' {unit}' if unit else ''} etrafında "
                   "merkezlidir" if names.xc != case.roles[ACIKLAYICI] else "")
                + f"; Sütun (2)'de {_word(case, names.xc)} katsayısı ({sayi(slope, _d(slope))}) referans grubun eğimi, "
                f"etkileşim ({sayi(inter, _d(inter))}) eğim farkıdır; diğer grubun eğimi ikisinin toplamıdır. Kukla "
                f"katsayısı yalnız {point} farktır. ")
        if variant == "acik":
            text += (f"Sütun (3) terimleri açıkça yazar ve Sütun (2) ile aynı modeldir: `{dummy.column} * {names.xc}`, "
                     f"`{dummy.column} + {names.xc} + {product}` demektir. ")
        elif variant == "iki_nokta":
            text += ("Sütun (3)'te yalnız çarpım vardır: kukla ve açıklayıcının ana etkileri modelden çıkar. Referans "
                     "grubun eğimi sıfır, diğer grubunki etkileşim katsayısı olmaya zorlanır ve iki grubun merkezleme "
                     "noktasındaki farkı sıfır sayılır; hiyerarşi ilkesine aykırı bu kısıtlar katsayıların anlamını "
                     "değiştirir. Sütun (3)'teki etkileşim katsayısı artık eğim farkı değil, diğer grubun eğimidir; yıldızı "
                     "bu yüzden Sütun (2)'dekiyle karşılaştırılamaz. ")
        if exact:
            return text + "Uyum tam olduğu için yıldızlar yuvarlama hatasıdır (§11.9)." + EXACT_NOTE
        p = float(state.models["m_s2"].pvalues[product])
        if p < 0.01:
            stars = "üç yıldız eğim farkının yüzde 1 düzeyinde anlamlı olduğunu gösterir"
        elif p < 0.05:
            stars = "iki yıldız eğim farkının yüzde 5 düzeyinde anlamlı olduğunu gösterir"
        elif p < 0.10:
            stars = "tek yıldız eğim farkının yalnız yüzde 10 düzeyinde anlamlı olduğunu gösterir"
        else:
            stars = "yıldız olmaması eğim farkının yüzde 10 düzeyinde bile anlamlı olmadığını gösterir"
        return text + f"Sütun (2)'de etkileşim satırında {stars} (§11.9)."

    checks = []
    for heading, terms in (("(1) Additif model", terms_add), ("(2) Etkileşimli model", main_terms)):
        for term in terms:
            name = "etkileşim" if ":" in term else _title(case, term)
            checks += [_check(f"Tablo: {heading}, {name}", TableTarget("tablo116", term, heading), 4),
                       _check(f"Tablo: {heading}, {name} (SH)", TableTarget("tablo116", f"{term}_sh", heading), 4)]
        checks += [_check(f"Tablo: {heading}, gözlem sayısı", TableTarget("tablo116", "n", heading), 0),
                   _check(f"Tablo: {heading}, R²", TableTarget("tablo116", "r2", heading), 4),
                   _check(f"Tablo: {heading}, düzeltilmiş R²", TableTarget("tablo116", "adj_r2", heading), 4)]
    return _step(
        number=10,
        title="Formül yazımı ve makale tablosunda etkileşim",
        note=NoteRef("11.9", 0, ("Tablo 11.6",)),
        explanation=(
            "statsmodels formülünde `D * X` terimleri $1 + D + X + D{:}X$ olarak açar; `D:X` yalnız çarpımı, `D + X` "
            "etkileşimsiz additif modeli kurar. Makale tablosu okunurken önce referans grup ve merkezleme noktası "
            "belirlenir; ana $X$ katsayısı referans grubun eğimi, etkileşim eğim farkı, kukla katsayısı merkezleme "
            "noktasındaki farktır. Üçüncü sütunun formülünü değiştirin."
        ),
        controls=(control,),
        build=build,
        checks=stable_checks(tuple(checks), exact, table="tablo116"),
        note_for=note,
    )


# --- Tanım ------------------------------------------------------------------------------------------------------

def build(case: Case) -> LabSpec:
    """Konu 11 uygulamasını verilen veriyle kurar; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    names = _names(case)
    house = _house(case)
    labels = labels_of(case)
    for column, label in labels_of(house).items():
        labels.setdefault(column, label)
    labels[INTERCEPT] = "Sabit terim"
    x, y = case.roles[ACIKLAYICI], case.roles[SONUC]
    labels.setdefault(names.ly, _log_title(case))
    if names.xc != x:
        labels.setdefault(names.xc, _minus(case.name(x), _center(case)))
    labels.setdefault(names.xcc, f"{case.name(x)} − c")
    for code, dummy in _dummies(case).items():
        labels.setdefault(code, f"{dummy.title} (1 = {dummy.one_raw}, 0 = {dummy.zero_raw})")
    house_dummy = _house_dummy(case)
    labels.setdefault(house_dummy.column, f"{house_dummy.title} (1 = {house_dummy.one_raw}, 0 = {house_dummy.zero_raw})")
    labels.update({
        "kukla_x": "Kukla × X (çarpım)",
        "grup0": "Referans grup: tahmin",
        "grup1": "Diğer grup: tahmin",
        "duzey0": f"Referans grup: tahmin edilen {case.name(y)}",
        "duzey1": f"Diğer grup: tahmin edilen {case.name(y)}",
        "grup_farki": "Gruplar arasındaki fark",
        "log_fark": "Log fark",
        "tam_fark": "Tam yüzde fark",
        "fark": "Dikey fark",
        "sira_no": "Gözlem sıra numarası",
        "dahil": "Modelde (1 = evet)",
        "n": "Gözlem sayısı",
    })
    for column, label in dict(case.extra.get("grid_labels", {})).items():
        labels.setdefault(column, label)
    for column, label in dict(house.extra.get("grid_labels", {})).items():
        labels.setdefault(column, label)
    spec = LabSpec(
        topic_key=TOPIC,
        title=str(case.extra.get("title", TITLE)),
        note_section="11",
        steps=(_step1(case), _step2(case), _step3(case), _step4(case), _step5(case), _step6(case), _step7(case),
               _step8(case), _step9(case), _step10(case)),
        labels=tuple(labels.items()),
        source=case.source,
    )
    return with_app_values(spec)


# --- Alternatif örnek ----------------------------------------------------------------------------------------

def _house_case() -> Case:
    """KIELMC'nin 1978 satışları: fiyat bin dolar ~ tesise yakınlık × konut büyüklüğü (2.000 fit² merkezli, 100 fit²)
    + oda + banyo."""

    return house_case(
        ("area2k", "rooms", "baths"), outcome="price1000", base="area2k", columns=(*HOUSE_COLUMNS, "nearinc"),
        derived=("price1000", "area2k"),
        dummy=("nearinc", "tesise yakın", "tesise uzak", "Tesise yakın"),
        members="konut", member_gen="konutun", unit_word="konut",
        grids={"area2k": ("konut_yuz", ("support", 10, 40), 20, "Konut büyüklüğü (yüz fit²)"),
               "rooms": ("oda", ("support", 4, 10), 0, "Oda sayısı"),
               "baths": ("banyo", ("support", 1, 4), 0, "Banyo sayısı")},
        grid_labels={"konut_yuz": "Konut büyüklüğü (yüz fit²)", "oda": "Oda sayısı", "banyo": "Banyo sayısı",
                     "aralik_ici": "Aralıkta (gösterge)", "aralik_alti": "Aralığın altında (gösterge)",
                     "aralik_ustu": "Aralığın üstünde (gösterge)"},
        zero_points={"area2k": "konut büyüklüğü 2.000 fit² iken"},
        raw={"area2k": ("area", 100.0, 2000.0)},
        points={"area2k": ((1000, -10), (1500, -5), (2000, 0), (2500, 5), (3000, 10), (3500, 15)),
                "rooms": ((5, 5), (6, 6), (7, 7), (8, 8), (9, 9)),
                "baths": ((1, 1), (2, 2), (3, 3), (4, 4))},
        point_headings={"area2k": "Konut büyüklüğü (fit²)", "rooms": "Oda sayısı", "baths": "Banyo sayısı"},
        counts=("area2k", "area", 1000, 4000, "fit²", "konutun"),
        house_text=("Notlarda HPRICE1 ile yapılan adım KIELMC'nin 1978 satışlarıyla yapılır: fiyat bin dolar, konut "
                    "büyüklüğü 2.000 fit² etrafında merkezli ve 100 fit² biriminde (`area2k`). Kukla konutun sonradan "
                    "kurulan çöp yakma tesisine 3 mil ya da daha yakın olmasıdır (`nearinc`; 1978'de tesis yoktu, kukla "
                    "konumu gösterir)."),
        step7_title="KIELMC (1978): eğim konuma göre değişiyor mu?",
        step8_title="Konum farkı etkileşen değişkene bağlıdır",
    )


def alternative_case() -> Case:
    base = beauty_case(
        extras=("exper",),
        title="Uygulama: Etkileşim Terimleri ve Gruplar Arasında Sabit ile Eğim Farklılıkları (BEAUTY ve KIELMC)",
        names={"dummy": "female", "ly": "lwage", "xc": "educ12", "xcc": "educ_c"},
        dummies={"female": ("kadın", "erkek", "Kadın"), "married": ("evli", "evli olmayan", "Evli"),
                 "black": ("siyah", "siyah olmayan", "Siyah"),
                 "union": ("sendika üyesi", "sendika üyesi olmayan", "Sendika üyesi")},
        controls=("exper",),
        center=12,
        center_comment="Eğitim 12 yıl etrafında merkezlenir: educ12 = educ − 12",
        center_control=NumberChoice("adim4_c", "Merkezleme noktası c (eğitim yılı)", 0, 17, 12, 1, integer=True,
                                    help="Varsayılan c = 12. c = 0 merkezlenmemiş modeldir: kukla katsayısı sıfır yıllık "
                                         "eğitimdeki farktır (örneklemde en az 5 yıl)."),
        level_control=NumberChoice("adim6_duzey", "Farkın hesaplandığı eğitim düzeyi (yıl)", 5, 17, 12, 1, integer=True,
                                   help="Tablo dört düzeyi gösterir; ölçüler seçtiğiniz düzeydeki farktır. Varsayılan 12 "
                                        "yıl."),
        gap_levels=(8, 12, 16, 17),
        grids={"educ12": ("educ", ("support", 5, 17), 12, "Eğitim (yıl)"),
               "exper": ("exper", ("support", 0, 48), 0, "İş deneyimi (yıl)")},
        zero_points={"educ12": "eğitim 12 yıl iken"},
        raw={"educ12": ("educ", 1.0, 12.0)},
        members="çalışan",
        phrases={**BEAUTY_PHRASES, "educ12": "eğitim", "educ_c": "eğitim"},
        short_units={**BEAUTY_UNITS, "educ12": "yıl", "educ_c": "yıl"},
        more={"educ12": "bir yıl daha fazla eğitim"},
        outcome_with="saatlik ücretle",
        causal={"female": "cinsiyete dayalı ayrımcılığın", "married": "evliliğin", "black": "ırka dayalı ayrımcılığın",
                "union": "sendika üyeliğinin"},
        model_text="BEAUTY'de (1260 çalışan) log saatlik ücret, 12 yıl etrafında merkezlenmiş eğitim ve iş deneyimiyle "
                   "açıklanır:",
        test_names=(("BEAUTY: kadın × eğitim", "BEAUTY", "BEAUTY verisinde"),
                    ("KIELMC (1978): tesise yakınlık × konut büyüklüğü", "KIELMC (1978)", "KIELMC (1978) verisinde")),
        step5_title="BEAUTY: eğitim eğimi gruplar arasında değişiyor mu?",
        house=_house_case(),
    )
    labels = {**base.labels, "educ12": "Eğitim − 12", "educ_c": "Eğitim − c"}
    units = {**base.units, "educ12": "yıl", "educ_c": "yıl"}
    return replace(base, labels=labels, units=units)


@cache
def alternative() -> LabSpec:
    return build(alternative_case())


STORY = ("Ana veri BEAUTY'dir (Hamermesh ve Biddle, 1994; 1260 çalışan): log saatlik ücret ~ kadın × eğitim (12 yıl "
         "merkezli) + iş deneyimi. Notlarda HPRICE1 ile yapılan adımlar KIELMC'nin 1978 satışlarıyla yapılır: fiyat ~ "
         "tesise yakınlık × konut büyüklüğü + oda + banyo. Katsayılar koşullu farklardır, nedensel etki değildir.")


# --- Kendi verin ---------------------------------------------------------------------------------------------

def own_validate(case: Case) -> None:
    """Notlardaki kurallar, log sonuç; iki kategorili değişkenin her grubunda en az iki gözlem; etkileşimli modeller her
    açıklayıcıyla kurulabilir (bir grupta açıklayıcı sabitse eğim farkı tanımsızdır)."""

    validate(case)
    validate_positive(case)
    column = case.roles[GOSTERGE]
    if case.data[column].value_counts().min() < 2:
        raise K.UploadError(f"“{case.name(column)}” sütununun her kategorisinde en az iki gözlem olmalı.")
    data = _data(case)
    dummy = _own_dummy(case).column
    options = _options(case)
    for term in options:
        if not _full_rank(data, _terms(dummy, term, _others(term, options))):
            name = case.name(case.roles[ACIKLAYICI] if _base(case, term) else term)
            raise K.UploadError(f"“{name}” ile kurulan etkileşimli model kurulamıyor: iki kategorili değişkenin bir "
                                "grubunda bu sütun sabit, açıklayıcılar arasında tam doğrusal bağlantı var ya da gözlem "
                                "yetersiz. Başka bir sütun seçin ya da daha az ek değişken kullanın.")


ROLES = roles((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), (
    Role(GOSTERGE, "İki kategorili değişken", "kategorik", True, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
         "Tam iki kategorili sütun (ör. kadın/erkek, evet/hayır): bir kategori 1, diğeri 0 kodlanır; 0 grubu referanstır. "
         "Boş hücresi olan satırlar çıkarılır.", levels=(2, 2), pick="1 ile kodlanan grup"),
))


def sample_groups() -> pd.DataFrame:
    """Örnek dosya: kurgusal iş arama programının program sonrası çalışan kişileri; cinsiyet ve medeni durum aynı verinin
    sütunlarından. Öğrenci verisi değil."""

    frame = program_frame()
    frame = frame[frame["issiz"] == 0].reset_index(drop=True)
    return pd.DataFrame({
        "Yıllık kazanç (bin TL)": frame["kazanc"],
        "Eğitim yılı": frame["egitim"],
        "Yaş": frame["yas"],
        "Önceki yıllık kazanç (bin TL)": frame["onceki_kazanc"],
        "Cinsiyet": np.where(frame["kadin"] == 1, "Kadın", "Erkek"),
        "Medeni durum": np.where(frame["evli"] == 1, "Evli", "Bekâr"),
    })


CUSTOM = custom_lab(
    build,
    sample_groups,
    ("Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sonuç ve temel açıklayıcı değişken zorunludur ve sayısal olmalıdır; "
     "iki kategorili değişken (kukla) de zorunludur. "
     f"{POSITIVE_RULE} Temel açıklayıcı ortancasına yakın yuvarlak bir değer etrafında merkezlenir. Kontroller ek sayısal "
     "değişkenlerdir (en çok 3); kukla temel açıklayıcıyla ya da bir ek değişkenle etkileşebilir. Adım 7–8 aynı dosyanın "
     f"düzey sonucuyla kurulur. {ROW_RULE}"),
    ROLES,
    "Kontrol değişkenleridir; kuklayla etkileşebilirler (Adım 2 ve 7).",
    order_roles=(GOSTERGE,),
    validate=own_validate,
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
