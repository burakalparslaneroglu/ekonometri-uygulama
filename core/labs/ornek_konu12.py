"""Konu 12 genel uygulaması: heteroskedastisite ve heteroskedastisiteye dayanıklı çıkarım.

Notlardaki sekiz adım (``core.labs.konu12``) aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde
yazılır: artık grafikleri, Breusch–Pagan ve White testleri, HC0–HC3 dayanıklı kovaryans seçenekleri, düzey modelinde
geleneksel ve dayanıklı çıkarım, log modelle karşılaştırma, makale tablosunda standart hata türü, dayanıklı ortak test ve
ikinci bir veride düzey ve log sonuç modellerinin karşılaştırılması.

Alternatif örnekte ana veri KIELMC'nin 1978 satışlarıdır (179 konut; notlardaki HPRICE1 yerine): düzey modeli fiyat (bin
dolar) ~ arsa büyüklüğü (bin fit²) + konut büyüklüğü (yüz fit²) + oda + banyo, log modeli log fiyat ~ log arsa + log konut
büyüklüğü + oda + banyo. Düzey modelinde testler homoskedastisiteyi güçlü biçimde reddeder ve arsası 544.500 fit² olan tek
bir konutun kaldıracı yüksektir (notlardaki en büyük arsa gibi); log modelde testler reddetmez. İkinci veri WAGE2'dir
(935 erkek çalışan, 1980; notlardaki WAGE1 yerine): aylık kazanç ve log aylık kazanç modelleri.

"Kendi verini yükle" seçeneğinde düzey modeli her zaman kurulur: sonuç ~ temel açıklayıcı + ek değişkenler. Log modeli
için sonucun ve temel açıklayıcının bütün değerleri pozitif olmalı: ln(sonuç) ~ ln(temel açıklayıcı) + ek değişkenler
(notlardaki log fiyat modeli gibi; sayım değişkenleri düzeyde kalır). Log modeli kurulamıyorsa Adım 5 neye ihtiyaç
duyduğunu yazar; Adım 8'de (aynı dosya, ayrı çerçeve) sonucun pozitif olması yeter. Kaldıraç ve HC3 payı metinleri
veriden hesaplanır (notlarda basılı sayılardır). White testi için gözlem sayısı yardımcı regresyonun terim sayısını
aşmalıdır; aşmıyorsa dosya daha az ek değişkenle istenir.

Etkileşim notlardaki gibidir: incelenen model (Adım 1–3), yardımcı regresyon (Adım 2), dayanıklı kovaryans türü (Adım 4;
Adım 5–7 aynı türü kullanır), ortak testte sınanan katsayılar (Adım 7) ve ikinci verinin açıklayıcıları (Adım 8).
Varsayılandan farklı bir seçimde varsayılan (HC1, varsayılan açıklayıcılar ve sınanan katsayılar) yan yana gösterilir.
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
from core.labs import tables as T
from core.labs.ornek import EXACT_FIT, Case, TopicVariants, free_name, sayi, sayim, with_app_values
from core.labs.ornek_regresyon import (
    ACIKLAYICI,
    HOUSE_COLUMNS,
    ROW_RULE,
    SONUC,
    candidates,
    capital,
    custom_lab,
    digits_for,
    display,
    house_case,
    labels_of,
    level_text,
    listing,
    p_text,
    phrase,
    positive,
    roles,
    sample_employed,
    second,
    short_unit,
    validate,
    wage2_case,
)
from core.labs.regression import stars
from core.labs.spec import (
    HETERO_TESTS,
    INTERCEPT,
    OLS,
    Check,
    Choice,
    CoefficientPlot,
    CoefficientTable,
    CoefTarget,
    CopyFrame,
    Derive,
    GroupSummary,
    HeteroskedasticityTest,
    HypothesisPlot,
    JoinColumns,
    JointTest,
    LabSpec,
    LabStep,
    ModelTarget,
    ModelValue,
    MultiChoice,
    NoteRef,
    Percentile,
    RegressionTable,
    Residuals,
    Scalar,
    ScalarTable,
    ScalarTarget,
    ScatterPlot,
    ShowModel,
    TableTarget,
    interactive_step,
)

TOPIC = "konu12"
TITLE = "Uygulama: Heteroskedastisite ve Heteroskedastisiteye Dayanıklı Çıkarım"
HC_TYPES = ("HC0", "HC1", "HC2", "HC3")
QUARTER_LABELS = {1: "1. çeyrek (en düşük tahmin)", 2: "2. çeyrek", 3: "3. çeyrek", 4: "4. çeyrek (en yüksek)"}
CLOSE_RATIO = 1.15
"""Dayanıklı/geleneksel SH oranı bundan küçükse iki standart hata "birbirine yakın" sayılır (notlardaki gibi)."""
SPREAD_RATIO = 1.5
"""En yüksek tahmin çeyreğindeki artık standart sapması diğerlerinin en büyüğünün bu katından büyükse huni biçimi."""
HIGH_LEVERAGE = 3
"""Kaldıraç ortalama kaldıracın ((k + 1)/n) bu katından büyükse yüksek sayılır (Belsley, Kuh ve Welsch, 1980'in 2(k + 1)/n
kuralından daha temkinli)."""
FRAGILE = ("se", "t", "p", "ci_low", "ci_high", "sh", "alt", "ust")


def _check(label: str, target, decimals: int) -> Check:
    return Check(label, target, 0.0, decimals)


def _scalar(name: str, label: str, decimals: int) -> Check:
    return _check(label, ScalarTarget(name), decimals)


def _d(value: float, minimum: int = 4) -> int:
    return max(minimum, digits_for(float(value), minimum))


def _formula(outcome: str, terms) -> str:
    return f"{outcome} ~ " + " + ".join(terms)


def _p(value: float, decimals: int = 4) -> str:
    return p_text(value, decimals)


# --- Modeller --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Model:
    """Ana verinin modeli: (anahtar, model adı, bağımlı değişken, açıklayıcılar, kısa ad, tahmin ekseninin adı, artık
    ekseninin birimi)."""

    key: str
    name: str
    outcome: str
    terms: tuple[str, ...]
    title: str
    axis: str
    unit: str


@dataclass(frozen=True)
class Names:
    ly: str
    lx: str


def _names(case: Case) -> Names:
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    taken = set(case.data.columns)
    ly = free_name(f"ln_{y}", taken)
    taken.add(ly)
    return Names(ly, free_name(f"ln_{x}", taken))


def _extras(case: Case) -> tuple[str, ...]:
    x = case.roles[ACIKLAYICI]
    return tuple(column for column in candidates(case) if column != x)


def _matrix(data: pd.DataFrame, terms) -> np.ndarray:
    return np.column_stack([np.ones(len(data)), *(data[term].to_numpy(dtype=float) for term in terms)])


def _rank(design: np.ndarray) -> int:
    norms = np.linalg.norm(design, axis=0)
    if np.any(norms == 0):
        return int(np.linalg.matrix_rank(design[:, norms > 0] / norms[norms > 0], tol=1e-10))
    return int(np.linalg.matrix_rank(design / norms, tol=1e-10))


def _leverage(data: pd.DataFrame, terms) -> np.ndarray:
    design = _matrix(data, terms)
    inverse = np.linalg.pinv(design.T @ design)
    return np.einsum("ij,jk,ik->i", design, inverse, design)


def _usable(data: pd.DataFrame, terms) -> bool:
    """Model kurulabilir mi ve HC2–HC3 tanımlı mı: tam doğrusal bağlantı yok, artık serbestlik derecesi en az 1 ve hiçbir
    gözlemin kaldıracı 1 değil."""

    design = _matrix(data, terms)
    if len(data) < design.shape[1] + 1 or _rank(design) < design.shape[1]:
        return False
    return float(_leverage(data, terms).max()) < 1 - 1e-8


def _log_ok(case: Case) -> bool:
    """Kendi verinde log modeli: sonuç ve temel açıklayıcı pozitif, model kurulabilir."""

    if not case.own:
        return "log" in case.extra.get("models", {})
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    if not (positive(case, y) and positive(case, x)):
        return False
    names = _names(case)
    return _usable(_data(case, logs=True), (names.lx, *_extras(case)))


def _models(case: Case) -> dict[str, Model]:
    if "models" in case.extra:
        return {key: Model(key, *values) for key, values in case.extra["models"].items()}
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    found = {"duzey": Model("duzey", "m_duzey", y, (x, *_extras(case)), "Düzey modeli", f"Tahmin edilen {case.name(y)}",
                            "")}
    if _log_ok(case):
        names = _names(case)
        found["log"] = Model("log", "m_log", names.ly, (names.lx, *_extras(case)), "Log modeli",
                             f"Tahmin edilen ln({case.name(y)})", " (log birimi)")
    return found


def _data(case: Case, logs: bool | None = None) -> pd.DataFrame:
    """Metinler, basamaklar ve doğrulama için veri: kendi verinde log sütunları eklenir (uygulama işlemlerle hesaplar)."""

    data = case.data.copy()
    if case.own and (logs if logs is not None else _log_ok(case)):
        names = _names(case)
        data[names.ly] = np.log(data[case.roles[SONUC]].astype(float))
        data[names.lx] = np.log(data[case.roles[ACIKLAYICI]].astype(float))
    return data


def _setup(case: Case) -> tuple:
    if not case.own or not _log_ok(case):
        return ()
    names = _names(case)
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    return (Derive(case.frame, names.ly, E.log(E.var(y)), f"ln({case.name(y)})"),
            Derive(case.frame, names.lx, E.log(E.var(x)), f"ln({case.name(x)})"))


def _fits(case: Case) -> tuple:
    return tuple(OLS(model.name, case.frame, model.outcome, model.terms,
                     f"{model.title}: {_formula(model.outcome, model.terms)}") for model in _models(case).values())


def _fit(data: pd.DataFrame, outcome: str, terms, cov: str = "nonrobust"):
    model = smf.ols(f"{outcome} ~ {' + '.join(terms)}", data=data)
    return model.fit() if cov == "nonrobust" else model.fit(cov_type=cov, use_t=True)


def _exact(case: Case, data: pd.DataFrame, outcome: str, terms) -> bool:
    if not case.own:
        return False
    design = _matrix(data, terms)
    values = data[outcome].to_numpy(dtype=float)
    estimates, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ estimates
    total = float(((values - values.mean()) ** 2).sum())
    return float((residual ** 2).sum()) <= EXACT_FIT * total


def _model_options(case: Case) -> tuple[tuple[str, str], ...]:
    return tuple((key, f"{model.title}: {_formula(model.outcome, model.terms)}") for key, model in _models(case).items())


def _live(control) -> bool:
    options = getattr(control, "options", None)
    return options is None or len(options) > 1


def _step(*, number: int, title: str, note: NoteRef, explanation: str, build, checks, note_for, controls=(),
          uses=()) -> LabStep:
    """Tek seçenekli denetimler (kendi verinde log modeli yoksa model seçimi) gösterilmez, varsayılanıyla kurulur."""

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


def _exact_flags(case: Case) -> dict[str, bool]:
    data = _data(case)
    return {key: _exact(case, data, model.outcome, model.terms) for key, model in _models(case).items()}


def _stable(checks, exact: bool) -> tuple[Check, ...]:
    """Uyum tamsa (R² ≈ 1) artıklar yuvarlama gürültüsüdür: standart hata, t, p, güven aralığı, LM ve F kontrolleri
    çıkarılır."""

    if not exact:
        return tuple(checks)
    kept = []
    for check in checks:
        target = check.target
        if isinstance(target, CoefTarget) and target.quantity in FRAGILE:
            continue
        if isinstance(target, ModelTarget) and target.quantity in ("f", "f_p"):
            continue
        if isinstance(target, TableTarget) and (str(target.row).endswith("_sh") or any(
                word in str(target.column) for word in (" se", " p", "LM", "p-değeri")) or str(target.column)[:1] == "F"):
            continue
        if isinstance(target, ScalarTarget) and target.name.startswith(("lm_", "p_", "sh", "F_", "t7", "p4", "p5", "alt",
                                                                         "ust", "bp_", "wh_")):
            continue
        kept.append(check)
    return tuple(kept)


EXACT_NOTE = (" Model veriye neredeyse tam uyuyor (R² ≈ 1): artıklar yuvarlama gürültüsüdür; standart hatalar, testler ve "
              "p-değerleri yorumlanmaz ve indirilen kodda karşılaştırılmaz.")


# --- Metindeki adlar --------------------------------------------------------------------------------------------

def _word(case: Case, term: str) -> str:
    """Markdown metnindeki ad: "arsa büyüklüğü"; kendi verinde tırnak içinde dosyadaki ad (log sütununda ln(“X”))."""

    if case.own:
        names = _names(case)
        if term == names.lx:
            return f"ln({phrase(case, case.roles[ACIKLAYICI])})"
        if term == names.ly:
            return f"ln({phrase(case, case.roles[SONUC])})"
    return phrase(case, term)


def _word_raw(case: Case, term: str) -> str:
    if case.own:
        names = _names(case)
        if term == names.lx:
            return f"ln({case.name(case.roles[ACIKLAYICI])})"
        if term == names.ly:
            return f"ln({case.name(case.roles[SONUC])})"
        return case.name(term)
    return phrase(case, term)


def _label(case: Case, term: str) -> str:
    return _word_raw(case, term) if case.own else capital(_word_raw(case, term))


def _plot_labels(case: Case) -> tuple[tuple[str, str], ...]:
    if "plot_labels" in case.extra:
        return tuple(case.extra["plot_labels"])
    return tuple((term, _word_raw(case, term)) for model in _models(case).values() for term in model.terms)


def _prefix(case: Case) -> str:
    name = case.extra.get("data_name")
    return f"{name}, " if name and not case.own else ""


def _raw(case: Case, term: str) -> str:
    """Kaldıraç metnindeki gözlem tanımında kullanılan özgün sütun (ör. land1000 ve lland için land)."""

    known = dict(case.extra.get("raw", {}))
    if term in known:
        return known[term]
    if case.own and term == _names(case).lx:
        return case.roles[ACIKLAYICI]
    return term


def _value_text(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return sayim(number)
    return level_text(round(number, 4))


# --- Adım 1: artık grafikleri ------------------------------------------------------------------------------------

def _quartiles(case: Case, model: Model) -> tuple[int, ...]:
    """Tahmin edilen değerin çeyrekleri (uygulamadaki kuralla): en az iki gözlemi olan çeyrekler (standart sapma
    tanımlı olsun)."""

    data = _data(case)
    fit = _fit(data, model.outcome, model.terms)
    fitted = data[model.outcome].to_numpy(dtype=float) - fit.resid.to_numpy()
    cuts = [T.percentile(fitted, p) for p in (25, 50, 75)]
    groups = 1 + (fitted >= cuts[0]).astype(int) + (fitted >= cuts[1]).astype(int) + (fitted >= cuts[2]).astype(int)
    return tuple(group for group in (1, 2, 3, 4) if int((groups == group).sum()) >= 2)


def _spread_digits(case: Case, model: Model) -> int:
    if model.key == "log":
        return 4
    data = _data(case)
    fit = _fit(data, model.outcome, model.terms)
    return max(2, digits_for(float(np.abs(fit.resid).mean()), 2))


def _step1(case: Case) -> LabStep:
    models = _models(case)
    control = Choice("adim1_model", "İncelenen model", _model_options(case), "duzey",
                     help="Notlarda önce düzey modeli (Şekil 12.3–12.4), sonra log model (Şekil 12.5).")
    exact = _exact_flags(case)

    def build(choices) -> tuple:
        model = models[choices["adim1_model"]]
        level = model.key == "duzey"
        groups = _quartiles(case, model)
        summary: tuple = ()
        if len(groups) >= 2:
            summary = (GroupSummary("tani", "ceyrek", (("n", "artik", "count"), ("ss", "artik", "std"),
                                                       ("ort_mutlak", "mutlak", "mean")),
                                    "yayilim", groups, decimals=_spread_digits(case, model),
                                    labels=tuple((group, QUARTER_LABELS[group]) for group in groups),
                                    heading="Tahmin edilen değer",
                                    title="Artıkların yayılımı tahmin edilen değerin çeyreklerine göre (standart sapma ve "
                                          "ortalama mutlak artık)"),)
        return (
            *case.load,
            *_setup(case),
            *_fits(case),
            ShowModel(model.name, f"{model.title}: katsayılar", columns=("coef",), stats=("nobs", "r2"),
                      exact=not case.own),
            CopyFrame("tani", case.frame, "Tanı için verinin kopyası (özgün veri değişmez)"),
            Residuals("tani", "artik", model.name, f"{model.title}: artıklar û"),
            Derive("tani", "tahmin", E.sub(E.var(model.outcome), E.var("artik")), "Tahmin edilen değer: Ŷ = Y − û"),
            Derive("tani", "mutlak", E.absolute(E.var("artik")), "Mutlak artık |û|"),
            ScatterPlot("tani", "tahmin", "artik", model.axis, "Artık" + model.unit,
                        f"{_prefix(case)}{model.title.lower()}: artıklar ve tahmin edilen değerler (Şekil "
                        f"{'12.3' if level else '12.5'}'teki gibi)", lines=((0.0, 0.0, "Sıfır çizgisi"),), opacity=0.8),
            ScatterPlot("tani", "tahmin", "mutlak", model.axis, "Mutlak artık" + model.unit,
                        f"{_prefix(case)}{model.title.lower()}: mutlak artıklar ve tahmin edilen değerler (Şekil 12.4'teki "
                        "gibi)", opacity=0.8),
            *(Percentile("tani", "tahmin", p, f"q{p}", f"Tahmin edilen değerin {p}. yüzdeliği", decimals=3)
              for p in (25, 50, 75)),
            Derive("tani", "ceyrek", E.add(E.add(E.add(1, E.compare("ge", E.var("tahmin"), E.ref("q25"))),
                                                  E.compare("ge", E.var("tahmin"), E.ref("q50"))),
                                            E.compare("ge", E.var("tahmin"), E.ref("q75"))),
                   "Tahmin edilen değerin çeyreği: 1 (en düşük) … 4 (en yüksek)"),
            *summary,
        )

    def note(state, choices) -> str:
        kind = choices["adim1_model"]
        model = models[kind]
        if exact[kind]:
            return ("Uyum tam olduğu için artıklar yuvarlama gürültüsüdür: grafikteki yayılım heteroskedastisite hakkında "
                    "bilgi vermez (§12.4)." + EXACT_NOTE)
        text = ""
        table = state.tables.get("yayilim")
        if table is not None and len(table) >= 2:
            decimals = _spread_digits(case, model)
            mean = table["ort_mutlak"].to_numpy(dtype=float)
            spread = table["ss"].to_numpy(dtype=float)
            top = int(_quartiles(case, model)[-1])
            label = "en yüksek tahmin çeyreğinde" if top == 4 else f"en yüksek dolu çeyrekte ({top}. çeyrek)"
            lower = "diğer çeyreklerde" if len(table) > 2 else "diğer çeyrekte"
            low_mean = (sayi(mean[:-1].min(), decimals) if len(mean) == 2 else
                        f"{sayi(mean[:-1].min(), decimals)} ile {sayi(mean[:-1].max(), decimals)} arasında")
            low_spread = (sayi(spread[:-1].min(), decimals) if len(spread) == 2 else
                          f"{sayi(spread[:-1].min(), decimals)}–{sayi(spread[:-1].max(), decimals)}")
            text = (f"Ortalama mutlak artık {lower} {low_mean}, {label} {sayi(mean[-1], decimals)}; artıkların standart "
                    f"sapması {label} {sayi(spread[-1], decimals)}, {lower} {low_spread}. ")
            if spread[-1] >= spread.max() and spread[-1] >= SPREAD_RATIO * spread[0]:
                text += ("Düşük tahmin düzeylerinde artıklar daha dar bir alanda toplanırken yüksek tahmin edilen değerlerde "
                         "çok daha büyük pozitif ve negatif artıklar görülür: huni biçimi heteroskedastisite şüphesini "
                         "güçlendirir. ")
            elif spread[0] >= spread.max() and spread[0] >= SPREAD_RATIO * spread[-1]:
                text += ("Yayılım tahmin edilen değer büyüdükçe daralır: daralan bir yayılım da heteroskedastisite "
                         "izidir. ")
            elif spread.max() >= SPREAD_RATIO * spread.min():
                text += ("Yayılım çeyreklere göre değişir ama tahmin edilen değerle düzenli biçimde artmaz ya da azalmaz: "
                         "grafik açık bir huni biçimi göstermez. ")
            else:
                text += "Yayılım çeyrekler arasında birbirine yakındır: grafik belirgin bir huni biçimi göstermez. "
        if kind == "log":
            text += "Log dönüşümü ölçek etkisini azaltabilir ama katsayıların yorumunu da değiştirir (esneklik). "
        return (text + "Grafik kesin bir test değildir: az sayıdaki uç gözlem ya da doğrusal olmayan bir koşullu ortalama "
                "da benzer iz bırakabilir (§12.4).")

    data = _data(case)
    checks = []
    for model in models.values():
        fit = _fit(data, model.outcome, model.terms)
        checks += [_check(f"{model.title}: {'sabit terim' if term == INTERCEPT else _word_raw(case, term)}, katsayı",
                          CoefTarget(model.name, term, "coef"), _d(float(fit.params[term])))
                   for term in (INTERCEPT, *model.terms)]
        checks.append(_check(f"{model.title}: R²", ModelTarget(model.name, "r2"), 4))
    return _step(
        number=1,
        title="Artık grafikleriyle ilk tanı",
        note=NoteRef("12.4", 0, ("Şekil 12.3", "Şekil 12.4", "Şekil 12.5")),
        explanation=(
            "Hata terimleri gözlenmez; EKK artıkları $\\hat u_i$ kullanılır. Artık–tahmin grafiğinde yatay eksende "
            "$\\hat Y_i$, dikey eksende $\\hat u_i$ vardır. Homoskedastik modelde sıfır çizgisi çevresindeki dikey "
            "yayılım yaklaşık sabittir; huni ya da yelpaze biçimi heteroskedastisite şüphesi doğurur. Mutlak artıklar "
            "hata büyüklüğünü işaretinden bağımsız gösterir. "
            + case.extra.get("model_text", "")
            + ("İncelenen modeli değiştirin." if _live(control) else
               "Log modeli için sonucun ve temel açıklayıcının bütün değerleri pozitif olmalı; bu dosyada yalnız düzey "
               "modeli kurulur.")
        ),
        controls=(control,),
        build=build,
        checks=tuple(checks),
        note_for=note,
    )


# --- Adım 2: Breusch–Pagan ve White testleri -----------------------------------------------------------------------

def _white_pairs(case: Case, terms: tuple[str, ...]) -> tuple[tuple[str, str, str], ...]:
    """White yardımcı regresyonunun kare ve çapraz çarpım sütunları: (ad, birinci, ikinci). Önceki terimlerin doğrusal
    birleşimi olan sütun (ör. 0/1 kuklanın karesi kendisidir) eklenmez: testin serbestlik derecesi bağımsız terim
    sayısıdır. Adlar verideki sütunlarla karışmaz."""

    data = _data(case)
    taken = set(data.columns) | set(terms)
    design = [np.ones(len(data)), *(data[term].to_numpy(dtype=float) for term in terms)]
    rank = _rank(np.column_stack(design))
    found = []
    for index, first in enumerate(terms):
        for second_ in terms[index:]:
            column = data[first].to_numpy(dtype=float) * data[second_].to_numpy(dtype=float)
            if _rank(np.column_stack([*design, column])) <= rank:
                continue
            name = free_name(f"{first}_kare" if first == second_ else f"{first}_x_{second_}", taken)
            taken.add(name)
            found.append((name, first, second_))
            design.append(column)
            rank += 1
    return tuple(found)


def _white_terms(case: Case, frame: str, terms: tuple[str, ...]) -> tuple[tuple, tuple[str, ...]]:
    """White yardımcı regresyonunun türetilen sütunları ve bütün terimleri (düzeyler, kareler, çapraz çarpımlar)."""

    pairs = _white_pairs(case, terms)
    derived = tuple(Derive(frame, name, E.mul(E.var(first), E.var(second_)),
                           f"{first}²" if first == second_ else f"{first} × {second_}") for name, first, second_ in pairs)
    return derived, (*terms, *(name for name, _, _ in pairs))


def _white_count(data: pd.DataFrame, terms: tuple[str, ...]) -> int:
    """White yardımcı regresyonundaki terim sayısı (sabit hariç): düzeyler, kareler ve çapraz çarpımlar; iki değerli bir
    sütunun karesi (ör. 0/1 kukla) sütunun ve sabitin doğrusal birleşimidir, sayılmaz. Gözlem sayısıyla sınırlanmaz."""

    count = len(terms)
    two_valued = sum(1 for term in terms if data[term].nunique() <= 2)
    return count + count * (count + 1) // 2 - two_valued


def _step2(case: Case) -> LabStep:
    models = _models(case)
    exact = _exact_flags(case)
    test_control = Choice("adim2_test", "Yardımcı regresyon",
                          (("bp", "Breusch–Pagan: açıklayıcıların düzeyleri (notlar)"),
                           ("white", "White: düzeyler, kareler ve çapraz çarpımlar")), "bp",
                          help="Yardımcı regresyonun bağımlı değişkeni artıkların karesidir; LM = n·R².")
    model_control = Choice("adim2_model", "Model", _model_options(case), "duzey", help="Notlarda düzey modeli (§12.5).")

    def build(choices) -> tuple:
        test, kind = choices["adim2_test"], choices["adim2_model"]
        model = models[kind]
        white, aux_terms = _white_terms(case, "yardimci", model.terms) if test == "white" else ((), model.terms)
        library: list = []
        for other in models.values():
            for test_kind in ("bp", "white"):
                library.append(HeteroskedasticityTest(f"lm_{test_kind}_{other.key}", f"p_{test_kind}_{other.key}",
                                                      other.name, test_kind,
                                                      f"{other.title}: {HETERO_TESTS[test_kind]} testi", shown=False))
        rows = tuple((other.title, other.key) for other in models.values())
        return (
            CopyFrame("yardimci", case.frame, "Yardımcı regresyon için verinin kopyası (özgün veri değişmez)"),
            Residuals("yardimci", "artik", model.name, f"{model.title}: artıklar û"),
            Derive("yardimci", "artik2", E.power(E.var("artik"), 2), "Kareli artık û²: hata varyansının gözlenebilir "
                   "ölçüsü"),
            *white,
            OLS("m_yardimci", "yardimci", "artik2", aux_terms,
                f"{HETERO_TESTS[test]} yardımcı regresyonu: {_formula('artik2', aux_terms)}"),
            ModelValue("r2_yardimci", "m_yardimci", "r2", "Yardımcı regresyonun R²'si", decimals=6),
            ModelValue("n_yardimci", "m_yardimci", "nobs", "Gözlem sayısı n", decimals=0),
            Scalar("lm_elle", E.mul(E.ref("n_yardimci"), E.ref("r2_yardimci")), "LM = n·R² (elle)", decimals=3),
            Scalar("q_yardimci", E.const(len(aux_terms)), "Serbestlik derecesi q (sabit dışındaki terim sayısı)",
                   decimals=0),
            Scalar("p_elle", E.chi2sf(E.ref("lm_elle"), E.ref("q_yardimci")), "p = P(χ²_q > LM)", decimals=4,
                   p_value=True),
            HeteroskedasticityTest("lm_kutuphane", "p_kutuphane", model.name, test,
                                   f"Tek çağrıyla (`{'het_breuschpagan' if test == 'bp' else 'white_testi'}`)"),
            *library,
            *(ScalarTable(tuple((label, E.ref(f"{quantity}_{test_kind}_{key}")) for label, key in rows),
                          f"{quantity}_{test_kind}_122", decimals=3, heading="Model")
              for quantity in ("lm", "p") for test_kind in ("bp", "white")),
            JoinColumns("tablo122", (("BP LM", "lm_bp_122", "deger"), ("BP p", "p_bp_122", "deger"),
                                     ("White LM", "lm_white_122", "deger"), ("White p", "p_white_122", "deger")),
                        decimals=3, heading="Model", p_columns=("BP p", "White p"),
                        column_decimals=(("BP p", 4), ("White p", 4)),
                        title=f"{_prefix(case)}heteroskedastisite testleri (Tablo 12.2'deki gibi)"),
        )

    def note(state, choices) -> str:
        test, kind = choices["adim2_test"], choices["adim2_model"]
        model = models[kind]
        s = state.scalars
        if exact[kind]:
            return (f"{HETERO_TESTS[test]} ({model.title.lower()}): uyum tam olduğu için kareli artıklar yuvarlama "
                    "gürültüsüdür; LM ve p yorumlanmaz (§12.5)." + EXACT_NOTE)
        rejected = s["p_elle"] < 0.05
        product = s["n_yardimci"] * float(f"{s['r2_yardimci']:.6f}")  # ekrandaki R² ile çarpım
        sign = "=" if sayi(product, 3) == sayi(s["lm_elle"], 3) else "≈"
        return (f"{HETERO_TESTS[test]} ({model.title.lower()}): yardımcı regresyonun R²'si {sayi(s['r2_yardimci'], 6)}, "
                f"LM = n·R² = {sayi(s['n_yardimci'], 0)} × {sayi(s['r2_yardimci'], 6)} {sign} {sayi(s['lm_elle'], 3)}, "
                f"q = {sayi(s['q_yardimci'], 0)} (yardımcı regresyondaki sabit dışındaki terim sayısı), "
                f"{_p(s['p_elle'])}; tek çağrı ({'statsmodels `het_breuschpagan`' if test == 'bp' else '`white_testi`'}) "
                f"aynı sayıyı verir ({sayi(s['lm_kutuphane'], 3)}). Yüzde 5 düzeyinde homoskedastisite hipotezi "
                + ("reddedilir. " if rejected else "reddedilemez; reddedememek homoskedastisitenin kanıtlandığı anlamına "
                   "gelmez, testin gücü sınırlı olabilir. ")
                + "White testi yanlış fonksiyonel biçime de duyarlıdır ve daha çok serbestlik derecesi tüketir. Test "
                "sonucu dayanıklı standart hata kullanmanın ön koşulu değildir (§12.5).")

    checks = [_scalar("lm_elle", "BP LM = n·R² (düzey modeli)", 3), _scalar("p_elle", "BP p (düzey modeli)", 4)]
    for model in models.values():
        checks += [_check(f"Tablo: {model.title}, {column}", TableTarget("tablo122", model.title, column),
                          3 if "LM" in column else 4) for column in ("BP LM", "BP p", "White LM", "White p")]
    level_exact = exact["duzey"]
    checks = [check for check in checks
              if not (isinstance(check.target, TableTarget) and exact[next(key for key, model in models.items()
                                                                          if model.title == check.target.row)])
              and not (isinstance(check.target, ScalarTarget) and level_exact)]
    return _step(
        number=2,
        title="Breusch–Pagan ve White testleri",
        note=NoteRef("12.5", 0, ("Tablo 12.2", "Kod 12.1")),
        explanation=(
            "İki testin sıfır hipotezi homoskedastisitedir: $H_0: \\operatorname{Var}(u_i \\mid X_i) = \\sigma^2$. "
            "Kareli artıklar $\\hat u_i^2$ yardımcı regresyonla açıklanır; $H_0$ altında ve büyük örneklemde "
            "$LM = nR^2_{aux}$ yaklaşık $\\chi^2_q$ dağılır ($q$: yardımcı regresyondaki sabit dışındaki terim sayısı). "
            "Breusch–Pagan yardımcı regresyonda modelin açıklayıcılarını, White ayrıca karelerini ve çapraz çarpımlarını "
            "kullanır. Küçük p-değeri homoskedastisitenin reddedilmesine yol açar. "
            + ("Testi ve modeli değiştirin." if _live(model_control) else "Testi değiştirin.")
        ),
        controls=(test_control, model_control),
        build=build,
        checks=tuple(checks),
        note_for=note,
    )


# --- Adım 3: HC0–HC3 ---------------------------------------------------------------------------------------------

def _influence(case: Case, model: Model) -> dict[str, float]:
    """En büyük kaldıraçlı gözlem ve ilk katsayının HC3 varyansındaki payı (metin için, veriden)."""

    data = _data(case)
    design = _matrix(data, model.terms)
    inverse = np.linalg.pinv(design.T @ design)
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design)
    values = data[model.outcome].to_numpy(dtype=float)
    residual = values - design @ (inverse @ design.T @ values)
    weights = (inverse @ design.T)[1] ** 2 * residual ** 2
    hc3 = weights / (1 - leverage) ** 2
    index = int(np.argmax(leverage))
    return {"index": index, "h": float(leverage[index]), "mean": float(leverage.mean()),
            "share": float(hc3[index] / hc3.sum()) if hc3.sum() > 0 else 0.0,
            "ratio": float(math.sqrt(hc3.sum() / weights.sum())) if weights.sum() > 0 else 1.0,
            "leverage": leverage}


def _describe(case: Case, model: Model, index: int) -> str:
    column = _raw(case, model.terms[0])
    value = float(case.data[column].iloc[index])
    unit = short_unit(case, column)
    return f"{index + 1}. gözlemin ({phrase(case, column)} {_value_text(value)}{f' {unit}' if unit else ''})"


def _leverage_text(case: Case, model: Model) -> str:
    found = _influence(case, model)
    k = len(model.terms) + 1
    n = len(case.data)
    word = _word(case, model.terms[0])
    mean = f"(ortalama kaldıraç (k + 1)/n = {k}/{n} ≈ {sayi(found['mean'], 3)})"
    h = found["h"]
    if h >= HIGH_LEVERAGE * found["mean"] and found["share"] >= 0.2:
        lead = ("HC2 ve HC3'teki artış yüksek kaldıraçlı bir gözlemden gelir: " if found["share"] >= 0.5 else
                "En büyük kaldıraç: ")
        text = (f"{lead}{_describe(case, model, found['index'])} kaldıracı yaklaşık {sayi(h, 2)} {mean}. HC3 bu gözlemin "
                f"kareli artığını (1 − h)² ≈ {sayi((1 - h) ** 2, 3)} ile bölerek yaklaşık {sayi(1 / (1 - h) ** 2, 1)} kat "
                f"büyütür; {word} katsayısının HC3 varyansının yaklaşık yüzde {sayi(100 * found['share'], 0)} kadarı bu "
                "tek gözlemden gelir. ")
    elif h >= HIGH_LEVERAGE * found["mean"]:
        text = (f"En büyük kaldıraç: {_describe(case, model, found['index'])} kaldıracı yaklaşık {sayi(h, 2)} {mean}; "
                f"ortalamanın birkaç katıdır ama bu gözlemin {word} katsayısının HC3 varyansındaki payı küçüktür"
                + ("; dört dayanıklı standart hata birbirine yakın kalır. " if found["ratio"] < CLOSE_RATIO else ". "))
    else:
        text = (f"En büyük kaldıraç yaklaşık {sayi(found['h'], 2)} {mean}: yüksek kaldıraçlı gözlem yoktur"
                + ("; dört dayanıklı standart hata birbirine yakın kalır. " if found["ratio"] < CLOSE_RATIO else ". "))
    models = _models(case)
    if model.key == "log" and "duzey" in models:
        level = _influence(case, models["duzey"])
        if level["h"] >= HIGH_LEVERAGE * level["mean"]:
            same = float(found["leverage"][level["index"]])
            if same < level["h"]:
                text += (f"Düzey modelinde kaldıracı {sayi(level['h'], 2)} olan gözlemin bu modeldeki kaldıracı yaklaşık "
                         f"{sayi(same, 2)}: log dönüşümü uç değerin öteki gözlemlerden uzaklığını azaltır. ")
    return text


def _step3(case: Case) -> LabStep:
    models = _models(case)
    exact = _exact_flags(case)
    control = Choice("adim3_model", "Model", _model_options(case), "duzey",
                     help="Beş sütunda aynı EKK katsayıları, farklı kovaryans hesabı.")

    def build(choices) -> tuple:
        model = models[choices["adim3_model"]]
        first = model.terms[0]
        d = _d(float(_fit(_data(case), model.outcome, model.terms).bse[first]))
        return (
            *(OLS(f"m3_{cov.lower()}", case.frame, model.outcome, model.terms, f"{model.title}, {cov} kovaryansıyla",
                  cov_type=cov) for cov in HC_TYPES),
            RegressionTable((("(1) Geleneksel", model.name), *((f"({index}) {cov}", f"m3_{cov.lower()}")
                                                               for index, cov in enumerate(HC_TYPES, start=2))),
                            (*model.terms, INTERCEPT), "tablo_hc",
                            f"{model.title}: aynı katsayılar, beş kovaryans hesabı", decimals=4, exact=not case.own),
            *(ModelValue(f"sh3_{cov.lower()}", f"m3_{cov.lower()}", "se", f"{cov} SH ({_word_raw(case, first)})",
                         term=first, decimals=d, shown=False) for cov in HC_TYPES),  # sayılar tabloda
            ModelValue("sh3_gel", model.name, "se", f"Geleneksel SH ({_word_raw(case, first)})", term=first, decimals=d,
                       shown=False),
        )

    def note(state, choices) -> str:
        kind = choices["adim3_model"]
        model = models[kind]
        if exact[kind]:
            return ("Katsayılar beş sütunda aynıdır. Uyum tam olduğu için bütün standart hatalar yuvarlama gürültüsüdür "
                    "(§12.6)." + EXACT_NOTE)
        s = state.scalars
        word = _word(case, model.terms[0])
        fitted = state.models[model.name]
        n, df = int(fitted.nobs), int(fitted.df_resid)
        d = _d(s["sh3_gel"])
        scaled = float(f"{s['sh3_hc0']:.{d + 1}f}") * math.sqrt(n / df)  # ekrandaki HC0 ile
        sign = "=" if sayi(scaled, d) == sayi(s["sh3_hc1"], d) else "≈"
        text = (f"Katsayılar beş sütunda aynıdır; yalnız parantez içindeki standart hatalar ve yıldızlar değişir. "
                f"{word if word.startswith('ln(') else capital(word)} katsayısının standart hatası: geleneksel "
                f"{sayi(s['sh3_gel'], d)}, HC0 "
                f"{sayi(s['sh3_hc0'], d)}, HC1 {sayi(s['sh3_hc1'], d)}, HC2 {sayi(s['sh3_hc2'], d)}, HC3 "
                f"{sayi(s['sh3_hc3'], d)}. HC1, HC0 varyansını n/(n − k − 1) = {n}/{df} ile çarpar (k: sabit dışındaki "
                f"açıklayıcı sayısı); standart hata bunun kareköküyle büyür: {sayi(s['sh3_hc0'], d + 1)} × √({n}/{df}) "
                f"{sign} {sayi(s['sh3_hc1'], d)}. ")
        return (text + _leverage_text(case, model) + "HC3 en koruyucudur. Varsayılan tür HC1'dir (notlardaki gibi); küçük "
                "örneklemde ya da yüksek kaldıraçta HC3 duyarlılık kontrolüdür. Tabloda hangi kovaryansın kullanıldığı "
                "mutlaka yazılır (§12.6).")

    level = models["duzey"]
    fit = _fit(_data(case), level.outcome, level.terms)
    checks = tuple(_check(f"Tablo: {'sabit terim' if term == INTERCEPT else _word_raw(case, term)}, HC1 SH",
                          TableTarget("tablo_hc", f"{term}_sh", "(3) HC1"), _d(float(fit.bse[term])))
                   for term in (INTERCEPT, *level.terms))
    return _step(
        number=3,
        title="HC0, HC1, HC2 ve HC3: aynı katsayı, farklı standart hata",
        note=NoteRef("12.6", 0, ("Tablo 12.3",)),
        explanation=(
            "Dayanıklı standart hata her gözlemin hata büyüklüğünün aynı olduğunu varsaymaz. Basit regresyonda "
            "§12.3'teki gerçek varyans formülünde $\\sigma_i^2$ yerine gözleme özgü bir tahmin $\\omega_i$ konur: "
            "$\\widehat{\\operatorname{Var}}(\\hat\\beta_1) = \\sum_i (X_i - \\bar X)^2\\omega_i / [\\sum_i (X_i - "
            "\\bar X)^2]^2$; çoklu regresyonda aynı fikir bütün katsayılara uygulanır. HC0'da $\\omega_i = \\hat u_i^2$; "
            "HC1 bunu $n/(n - k - 1)$ ile çarpar ($k$: sabit dışındaki açıklayıcı sayısı); HC2 ve HC3 gözlemin "
            "kaldıracı $h_i$ ile $\\hat u_i^2/(1 - h_i)$ ve $\\hat u_i^2/(1 - h_i)^2$ kullanır. Kaldıraç, gözlemin "
            "açıklayıcı değerlerinin örneklemin merkezinden uzaklığını ölçer ($0 < h_i < 1$). Katsayılar, tahmin edilen "
            "değerler ve R² değişmez." + (" Modeli değiştirin." if _live(control) else "")
        ),
        controls=(control,),
        build=build,
        checks=_stable(checks, exact["duzey"]),
        note_for=note,
    )


# --- Adım 4: düzey modeli, geleneksel ve dayanıklı ---------------------------------------------------------------------

HC_4 = Choice("adim4_hc", "Dayanıklı kovaryans türü", tuple((cov, f"{cov} (varsayılan)" if cov == "HC1" else cov)
                                                         for cov in HC_TYPES), "HC1",
              help="Notlarda HC1. Adım 5–7 aynı türü kullanır.")


def _comparison(case: Case, model: str, robust: str, terms: tuple[str, ...], cov: str, result: str, title: str,
                default_robust: str | None = None, decimals: int = 4) -> tuple:
    """Notlardaki çıktı düzeni: katsayı, geleneksel SH ve p, dayanıklı SH ve p. ``default_robust``: HC1 dışında bir tür
    seçildiyse varsayılan HC1 modeli; HC1 sütunları seçilen türün solunda yan yana gösterilir."""

    order = (INTERCEPT, *terms)
    exact = not case.own
    tables = [CoefficientTable(model, order, f"{result}_gel", "Geleneksel standart hatayla çıkarım", decimals=decimals,
                               t_decimals=3, p_decimals=4, exact=exact)]
    columns = [("coef", f"{result}_gel", "katsayi"), ("OLS se", f"{result}_gel", "sh"), ("OLS p", f"{result}_gel", "p")]
    p_columns = ["OLS p"]
    if default_robust is not None:
        tables.append(CoefficientTable(default_robust, order, f"{result}_n", "Varsayılan HC1 dayanıklı standart hatayla "
                                       "çıkarım", decimals=decimals, t_decimals=3, p_decimals=4, exact=exact))
        columns += [("HC1 se (varsayılan)", f"{result}_n", "sh"), ("HC1 p (varsayılan)", f"{result}_n", "p")]
        p_columns.append("HC1 p (varsayılan)")
    tables.append(CoefficientTable(robust, order, f"{result}_rob", f"{cov} dayanıklı standart hatayla çıkarım",
                                   decimals=decimals, t_decimals=3, p_decimals=4, exact=exact))
    columns += [(f"{cov} se", f"{result}_rob", "sh"), (f"{cov} p", f"{result}_rob", "p")]
    p_columns.append(f"{cov} p")
    return (
        *tables,
        JoinColumns(result, tuple(columns), decimals=decimals, heading="Terim", p_columns=tuple(p_columns),
                    column_decimals=tuple((name, 4) for name in p_columns), title=title, term_rows=True),
    )


def _default_robust(case: Case, cov: str, model: Model, name: str) -> tuple:
    return () if cov == "HC1" else (OLS(name, case.frame, model.outcome, model.terms,
                                        f"{model.title}, varsayılan HC1 kovaryansıyla", cov_type="HC1"),)


def _table_decimals(case: Case, model: Model) -> int:
    fit = _fit(_data(case), model.outcome, model.terms)
    return max(4, *(digits_for(float(value), 4) for value in (*fit.params, *fit.bse)))


def _output_checks(case: Case, table: str, model: Model, decimals: int) -> list[Check]:
    columns = ("coef", "OLS se", "OLS p", "HC1 se", "HC1 p")
    return [_check(f"Tablo: {'sabit terim' if term == INTERCEPT else _word_raw(case, term)}, {column}",
                   TableTarget(table, term, column), 4 if column.endswith("p") else decimals)
            for term in (INTERCEPT, *model.terms) for column in columns]


def _step4(case: Case) -> LabStep:
    models = _models(case)
    level = models["duzey"]
    exact = _exact_flags(case)["duzey"]
    first, others = level.terms[0], level.terms[1:]
    decimals = _table_decimals(case, level)
    data = _data(case)
    fit = _fit(data, level.outcome, level.terms)
    y = level.outcome

    def build(choices) -> tuple:
        cov = choices["adim4_hc"]
        same = cov == "HC1"
        values = [("sh4_gel", level.name, "se", f"{_label(case, first)}: geleneksel SH", first),
                  ("sh4_rob", "m_rob", "se", f"{_label(case, first)}: {cov} SH", first),
                  ("p4_gel", level.name, "p", f"{_label(case, first)}: geleneksel p", first),
                  ("p4_rob", "m_rob", "p", f"{_label(case, first)}: {cov} p", first),
                  ("alt4_gel", level.name, "ci_low", f"{_label(case, first)}: geleneksel GA alt sınırı", first),
                  ("ust4_gel", level.name, "ci_high", f"{_label(case, first)}: geleneksel GA üst sınırı", first),
                  ("alt4_rob", "m_rob", "ci_low", f"{_label(case, first)}: {cov} GA alt sınırı", first),
                  ("ust4_rob", "m_rob", "ci_high", f"{_label(case, first)}: {cov} GA üst sınırı", first)]
        for index, term in enumerate(others, start=1):
            values += [(f"sh4_{index}_gel", level.name, "se", f"{_label(case, term)}: geleneksel SH", term),
                       (f"sh4_{index}_rob", "m_rob", "se", f"{_label(case, term)}: {cov} SH", term),
                       (f"p4_{index}_gel", level.name, "p", f"{_label(case, term)}: geleneksel p", term),
                       (f"p4_{index}_rob", "m_rob", "p", f"{_label(case, term)}: {cov} p", term)]
        if not same:
            values += [("sh4_n", "m_rob_n", "se", f"{_label(case, first)}: HC1 SH (varsayılan)", first),
                       ("p4_n", "m_rob_n", "p", f"{_label(case, first)}: HC1 p (varsayılan)", first)]
        return (
            OLS("m_rob", case.frame, y, level.terms, f"Aynı düzey modeli, {cov} dayanıklı kovaryansla", cov_type=cov),
            *_default_robust(case, cov, level, "m_rob_n"),
            *_comparison(case, level.name, "m_rob", level.terms, cov, "kod123",
                         f"{_prefix(case)}düzey modeli: geleneksel ve HC1 sonuçları (Kod 12.3'teki gibi)" if same
                         else f"{_prefix(case)}düzey modeli: geleneksel, varsayılan HC1 ve seçtiğiniz {cov}",
                         None if same else "m_rob_n", decimals),
            CoefficientPlot(level.name, level.terms, (f"{_prefix(case)}düzey modelinde geleneksel ve HC1 yüzde 95 güven "
                                                      "aralıkları (Şekil 12.6'daki gibi)") if same else
                            f"Geleneksel, varsayılan HC1 ve {cov} yüzde 95 güven aralıkları",
                            "Katsayı ve yüzde 95 güven aralığı" + (f" ({short_unit(case, y)})" if short_unit(case, y)
                                                                     else ""),
                            y_label="Değişken", labels=_plot_labels(case),
                            compare=((f"{cov} dayanıklı", "m_rob"),) + (() if same else (("HC1 dayanıklı (varsayılan)",
                                                                                       "m_rob_n"),)),
                            legend="Geleneksel"),
            *(ModelValue(name, model, quantity, label, term=term, decimals=4 if quantity == "p" else decimals,
                         shown=False) for name, model, quantity, label, term in values),  # sayılar tabloda
        )

    def note(state, choices) -> str:
        cov = choices["adim4_hc"]
        s = state.scalars
        b = float(state.models[level.name].params[first])
        every = "iki hesapta da" if cov == "HC1" else "bütün hesaplarda"
        unit_x, unit_y = short_unit(case, first), short_unit(case, y)
        member = case.extra.get("member_gen", "gözlemin")
        outcome = dict(case.extra.get("possessive", {})).get(y) if not case.own else None
        outcome = outcome or f"{phrase(case, y)} değeri"
        db = _d(b)
        text = (f"{capital(_word(case, first))} katsayısı {every} {sayi(b, db)}: diğer değişkenler sabitken "
                f"{_word(case, first)} bir birim{f' ({unit_x})' if unit_x else ''} fazla olan {member} tahmin edilen "
                f"{outcome} yaklaşık {sayi(abs(b), max(2, digits_for(b, 2)))}{f' {unit_y}' if unit_y else ' birim'} daha "
                f"{'yüksektir' if b > 0 else 'düşüktür'}. ")
        if exact:
            return text + "Uyum tam olduğu için standart hatalar yuvarlama gürültüsüdür (§12.7)." + EXACT_NOTE
        d = _d(s["sh4_gel"])
        text += (f"Değişen standart hatadır: geleneksel hesapla {sayi(s['sh4_gel'], d)} ({_p(s['p4_gel'])}), {cov} "
                 f"hesabıyla {sayi(s['sh4_rob'], d)} ({_p(s['p4_rob'])})"
                 + ("" if cov == "HC1" else f"; varsayılan HC1 hesabıyla {sayi(s['sh4_n'], d)} ({_p(s['p4_n'])})") + ". ")
        covered_rob = s["alt4_rob"] <= 0 <= s["ust4_rob"]
        covered_gel = s["alt4_gel"] <= 0 <= s["ust4_gel"]
        if covered_rob and not covered_gel:
            text += f"{cov} güven aralığı sıfırı içerir, geleneksel aralık içermez. "
        elif covered_gel and not covered_rob:
            text += f"Geleneksel güven aralığı sıfırı içerir, {cov} aralığı içermez. "
        text += "Katsayı ve iktisadi büyüklük değişmez; değişen belirsizlik ölçüsüdür. "
        rises = [_word(case, term) for index, term in enumerate(others, start=1)
                 if s[f"sh4_{index}_rob"] > s[f"sh4_{index}_gel"]]
        falls = [_word(case, term) for index, term in enumerate(others, start=1)
                 if s[f"sh4_{index}_rob"] < s[f"sh4_{index}_gel"]]
        flips = [_word(case, term) for index, term in enumerate(others, start=1)
                 if (s[f"p4_{index}_gel"] < 0.05) != (s[f"p4_{index}_rob"] < 0.05)]
        if rises and falls:
            text += (f"Diğer katsayılarda standart hata {listing(rises)} için yükselir, {listing(falls)} için düşer: "
                     "heteroskedastisite geleneksel standart hatayı her katsayıda aynı yönde etkilemez. ")
        elif len(others) == 1 and (rises or falls):
            text += f"Diğer katsayıda ({listing(rises or falls)}) standart hata {'yükselir' if rises else 'düşer'}. "
        elif rises or falls:
            text += f"Diğer katsayıların hepsinde standart hata {'yükselir' if rises else 'düşer'}. "
        if flips:
            text += f"Yüzde 5 düzeyindeki anlamlılığı değişen diğer katsayı: {listing(flips)}. "
        return text.rstrip().rstrip(".") + " (§12.7)."

    checks = [*_output_checks(case, "kod123", level, decimals),
              _check("Gözlem sayısı", ModelTarget(level.name, "nobs"), 0),
              _check("R²", ModelTarget(level.name, "r2"), 4)]
    del fit
    return _step(
        number=4,
        title=case.extra.get("step4_title", "Düzey modeli: aynı katsayı, farklı çıkarım"),
        note=NoteRef("12.7", 0, ("Kod 12.2 (§12.6)", "Kod 12.3", "Şekil 12.6")),
        explanation=(
            "Düzey modelinin geleneksel ve dayanıklı sonuçları aynı tabloda: katsayılar aynıdır, standart hata, "
            "p-değeri ve güven aralıkları değişir. statsmodels'te `fit(cov_type=\"HC1\", use_t=True)` (Kod 12.2) t ve F "
            "dağılımını ($n - k - 1$ serbestlik derecesi; $k$ sabit dışındaki açıklayıcı sayısı) kullanır; `use_t` "
            "yazılmazsa normal dağılım kullanılır ve p-değerleri farklı olur. Dayanıklı kovaryans türünü değiştirin: "
            "varsayılan HC1 sütunu yan yana gösterilir."
        ),
        controls=(HC_4,),
        build=build,
        checks=_stable(checks, exact),
        note_for=note,
    )


# --- Adım 5: log model ----------------------------------------------------------------------------------------------

def _step5(case: Case) -> LabStep:
    models = _models(case)
    title = "Log modelle karşılaştırma"
    note_ref = NoteRef("12.7", 0, ("Kod 12.4",))
    lead = ("Log modelde boyut değişkenlerinin katsayıları esnekliktir; log dönüşümü ölçek etkisini azaltabilir ve "
            "geleneksel ile dayanıklı standart hatalar birbirine yaklaşabilir.")
    if "log" not in models:
        return LabStep(number=5, title=title, note=note_ref,
                       explanation=(f"{lead} Bu adım için sonucun ve temel açıklayıcının bütün değerleri pozitif olmalı "
                                    "(log modeli: ln(sonuç) ~ ln(temel açıklayıcı) + ek değişkenler)."))
    log = models["log"]
    level = models["duzey"]
    exact = _exact_flags(case)["log"]
    decimals = _table_decimals(case, log)
    first = log.terms[0]
    x = case.roles[ACIKLAYICI]

    def build(choices) -> tuple:
        cov = choices["adim4_hc"]
        same = cov == "HC1"
        return (
            OLS("m_log_rob", case.frame, log.outcome, log.terms, f"Aynı log modeli, {cov} dayanıklı kovaryansla",
                cov_type=cov),
            *_default_robust(case, cov, log, "m_log_rob_n"),
            *_comparison(case, log.name, "m_log_rob", log.terms, cov, "kod124",
                         f"{_prefix(case)}log modeli: geleneksel ve HC1 sonuçları (Kod 12.4'teki gibi)" if same
                         else f"{_prefix(case)}log modeli: geleneksel, varsayılan HC1 ve seçtiğiniz {cov}",
                         None if same else "m_log_rob_n", decimals),
            ModelValue("sh5_gel", log.name, "se", f"{_label(case, first)}: geleneksel SH", term=first, decimals=decimals,
                       shown=False),
            ModelValue("sh5_rob", "m_log_rob", "se", f"{_label(case, first)}: {cov} SH", term=first, decimals=decimals,
                       shown=False),
            ModelValue("e5", log.name, "coef", f"{_label(case, first)}: esneklik", term=first, decimals=4, shown=False),
        )

    def note(state, choices) -> str:
        cov = choices["adim4_hc"]
        s = state.scalars
        member = case.extra.get("member_gen", "gözlemin")
        outcome = dict(case.extra.get("possessive", {})).get(level.outcome) if not case.own else None
        outcome = outcome or f"{phrase(case, case.roles[SONUC])} değeri"
        word = _word(case, first)
        text = (f"{word if word.startswith('ln(') else capital(word)} katsayısı esnekliktir: {phrase(case, x)} yüzde 1 daha "
                f"büyük olan {member} "
                f"tahmin edilen {outcome}, diğer değişkenler sabitken yaklaşık yüzde {sayi(abs(s['e5']), 3)} daha "
                f"{'yüksektir' if s['e5'] > 0 else 'düşüktür'}. ")
        if exact:
            return text + "Uyum tam olduğu için standart hatalar yuvarlama gürültüsüdür (§12.7)." + EXACT_NOTE
        if _exact_flags(case)["duzey"]:
            return (text + "Düzey modelinin uyumu tam olduğu için iki modelin standart hatalarını karşılaştırmak anlamlı "
                    "değildir (§12.7)." + EXACT_NOTE)
        ratio = s["sh5_rob"] / s["sh5_gel"]
        level_ratio = s["sh4_rob"] / s["sh4_gel"]
        distance, level_distance = abs(math.log(ratio)), abs(math.log(level_ratio))  # oranın 1'den uzaklığı (iki yön)
        close = distance < math.log(CLOSE_RATIO)
        if close and level_distance >= math.log(CLOSE_RATIO):
            closeness = "birbirine yakındır"
        elif close:
            closeness = "birbirine yakındır; düzey modelinde de yakındı"
        elif distance < 0.5 * level_distance:
            closeness = "düzey modeline göre birbirine çok daha yakındır"
        elif distance < level_distance:
            closeness = "düzey modeline göre birbirine daha yakındır"
        else:
            closeness = "birbirinden uzaktır"
        nearer = close or distance < level_distance
        text += (f"{cov} ve geleneksel standart hatalar {closeness} ({word} katsayısında {cov}/geleneksel SH oranı "
                 f"{sayi(ratio, 2)}; düzey modelinde ilk katsayıda {sayi(level_ratio, 2)}). ")
        bp, white = s.get("p_bp_log"), s.get("p_white_log")
        if bp is not None and white is not None:
            if bp >= 0.05 and white >= 0.05 and nearer:
                text += ("Bu, log modelde testlerin reddetmemesiyle uyumludur; reddetmemek homoskedastisiteyi kanıtlamaz. ")
            elif bp >= 0.05 and white >= 0.05:
                text += ("Log modelde testler reddetmese de iki standart hata farklıdır: reddetmemek homoskedastisiteyi "
                         "kanıtlamaz; dayanıklı standart hata raporlamak temkinli seçimdir. ")
            elif bp < 0.05 and white < 0.05:
                text += "Log modelde de iki test homoskedastisiteyi reddeder: dayanıklı standart hata burada da gerekir. "
            else:
                text += "Log modelde iki test farklı sonuç verir: dayanıklı standart hata raporlamak temkinli seçimdir. "
        return (text + "Düzey ve log modeli yalnız heteroskedastisite testine göre seçilmez: bağımlı değişken, iktisadi soru "
                "ve yorum belirleyicidir (§12.7).")

    checks = [*_output_checks(case, "kod124", log, decimals),
              _check("Gözlem sayısı", ModelTarget(log.name, "nobs"), 0),
              _check("R²", ModelTarget(log.name, "r2"), 4)]
    return interactive_step(
        number=5,
        title=title,
        note=note_ref,
        explanation=(f"{lead} Model: `{_formula(log.outcome, log.terms)}`. Adım 4'teki kovaryans seçimi bu adımı da "
                     "belirler."),
        uses=(HC_4,),
        build=build,
        checks=_stable(checks, exact),
        note_for=note,
    )


# --- Adım 6: makale tablosu ----------------------------------------------------------------------------------------

def _stars_word(p_value: float) -> str:
    """Yıldızlar Markdown metninde kaçırılır (vurgu işareti sanılmasın)."""

    return stars(p_value).replace("*", "\\*") or "yıldızsız"


def _step6(case: Case) -> LabStep:
    level = _models(case)["duzey"]
    exact = _exact_flags(case)["duzey"]
    y = level.outcome

    def build(choices) -> tuple:
        cov = choices["adim4_hc"]
        same = cov == "HC1"
        columns = (("(1) Geleneksel SH", level.name), (f"(2) {cov} SH", "m_rob"))
        if not same:  # varsayılan HC1 sütunu yan yana
            columns += (("(3) HC1 SH (varsayılan)", "m_rob_n"),)
        return (
            RegressionTable(columns, (*level.terms, INTERCEPT), "tablo124",
                            (f"{_prefix(case)}düzey modelinin makale tipi sunumu (Tablo 12.4'teki gibi)" if same
                             else f"Makale tablosu: geleneksel, seçtiğiniz {cov} ve varsayılan HC1 standart hatalar")
                            + f" · bağımlı değişken: {display(case, y)}", decimals=3, exact=not case.own),
        )

    def note(state, choices) -> str:
        cov = choices["adim4_hc"]
        if exact:
            return "Uyum tam olduğu için yıldızlar yuvarlama gürültüsüdür (§12.7)." + EXACT_NOTE
        conventional, robust = state.models[level.name], state.models["m_rob"]
        changed = [f"{_word(case, term)} (Sütun 1'de {_stars_word(conventional.pvalues[term])}, Sütun 2'de "
                   f"{_stars_word(robust.pvalues[term])})" for term in level.terms
                   if stars(conventional.pvalues[term]) != stars(robust.pvalues[term])]
        text = "Katsayılar sütunlarda aynıdır; yıldızlar standart hata türüne bağlıdır. "
        text += (f"Sütun (2)'de ({cov}) yıldızı değişen katsayılar: {'; '.join(changed)}. " if changed else
                 f"Bu seçimde ({cov}) yıldızlar iki sütunda aynıdır. ")
        return (text + "Bu yüzden makale tablosunda yıldızlardan önce tablo notu okunur: hangi standart hata (geleneksel, "
                "HC1, HC3), parantez içinde standart hata mı t istatistiği mi, yıldız eşikleri ve kontroller (§12.7).")

    fit = _fit(_data(case), level.outcome, level.terms)
    checks = []
    for heading in ("(1) Geleneksel SH", "(2) HC1 SH"):
        for term in (*level.terms, INTERCEPT):
            name = "sabit" if term == INTERCEPT else _word_raw(case, term)
            checks += [_check(f"Tablo: {heading}, {name}", TableTarget("tablo124", term, heading),
                              max(3, digits_for(float(fit.params[term]), 3))),
                       _check(f"Tablo: {heading}, {name} (SH)", TableTarget("tablo124", f"{term}_sh", heading),
                              max(3, digits_for(float(fit.bse[term]), 3)))]
        checks += [_check(f"Tablo: {heading}, gözlem sayısı", TableTarget("tablo124", "n", heading), 0),
                   _check(f"Tablo: {heading}, R²", TableTarget("tablo124", "r2", heading), 3)]
    return interactive_step(
        number=6,
        title="Makale tablosunda standart hata türü",
        note=NoteRef("12.7", 0, ("Tablo 12.4",)),
        explanation=(
            "Aynı katsayılar standart hata türüne göre farklı yıldızlar alabilir. Makale tablosunda tablo notu önce "
            "okunur: kullanılan standart hata ya da kovaryans türü (HC0–HC3 hangisi), parantez içinde ne olduğu, "
            "yıldız eşikleri, gözlem sayısı ve kontroller. Adım 4'teki kovaryans seçimi bu adımı da belirler."
        ),
        uses=(HC_4,),
        build=build,
        checks=_stable(checks, exact),
        note_for=note,
    )


# --- Adım 7: dayanıklı ortak test ------------------------------------------------------------------------------------

def _joint_default(case: Case) -> tuple[str, ...]:
    if "joint" in case.extra:
        return tuple(case.extra["joint"])
    terms = _models(case)["duzey"].terms
    return terms[:2] if len(terms) >= 2 else terms[:1]


def _hypothesis(case: Case, terms: tuple[str, ...]) -> str:
    """Sıfır hipotezinin yazımı: "arsa büyüklüğü ve oda sayısı katsayıları sıfır"."""

    words = [_word(case, term) for term in terms]
    if len(words) == 1:
        return f"{words[0]} katsayısı sıfır"
    return f"{listing(words)} katsayıları sıfır"


def _hypothesis_raw(case: Case, terms: tuple[str, ...]) -> str:
    words = [_word_raw(case, term) for term in terms]
    if len(words) == 1:
        return f"{words[0]} katsayısı sıfır"
    return f"{listing(words)} katsayıları sıfır"


def _step7(case: Case) -> LabStep:
    level = _models(case)["duzey"]
    exact = _exact_flags(case)["duzey"]
    default = _joint_default(case)
    df = len(case.data) - len(level.terms) - 1
    control = MultiChoice("adim7_terimler", "Sınanan katsayılar", tuple((term, _label(case, term)) for term in level.terms),
                          default, help="Varsayılan: " + ", ".join(_label(case, term) for term in default)
                          + " (Tablo 12.5'teki gibi iki katsayı).")

    def build(choices) -> tuple:
        cov, terms = choices["adim4_hc"], tuple(choices["adim7_terimler"])
        same_terms = terms == default
        same = cov == "HC1" and same_terms
        hypothesis = _hypothesis_raw(case, terms)
        q = len(terms)
        test = "Tek kısıtlı test" if q == 1 else "Ortak test"
        plot = (HypothesisPlot("t", "t7_rob", "sd7", f"{cov} dayanıklı t testi, H₀: {hypothesis}", "t değeri")
                if q == 1 else
                HypothesisPlot("f", "F_rob", "q7", f"{cov} dayanıklı ortak test, H₀: {hypothesis}: F({q}, {df})",
                               "F değeri", alternative="sag", df2="sd7"))
        rows = [("Geleneksel", "gel", q), (f"{cov} dayanıklı", "rob", q)]
        tests = [JointTest("F_gel", "p_gel", level.name, terms, f"Geleneksel F testi, H₀: {hypothesis}", decimals=3,
                           p_decimals=4, shown=False),
                 JointTest("F_rob", "p_rob", "m_rob", terms, f"{cov} dayanıklı Wald testi (F biçimi)", decimals=3,
                           p_decimals=4, shown=False)]  # F ve p tabloda
        if cov != "HC1":  # aynı hipotez, varsayılan HC1 kovaryansıyla
            tests.append(JointTest("F_rob_n", "p_rob_n", "m_rob_n", terms, "HC1 dayanıklı Wald testi (varsayılan tür)",
                                   decimals=3, p_decimals=4, shown=False))
            rows.append(("HC1 dayanıklı (varsayılan tür)", "rob_n", q))
        if not same_terms:  # varsayılan hipotez
            words = listing([_word_raw(case, term) for term in default])
            tests += [JointTest("F_gel_nn", "p_gel_nn", level.name, default, "Varsayılan hipotez: geleneksel F testi",
                                decimals=3, p_decimals=4, shown=False),
                      JointTest("F_rob_nn", "p_rob_nn", "m_rob" if cov == "HC1" else "m_rob_n", default,
                                "Varsayılan hipotez: HC1 dayanıklı Wald testi", decimals=3, p_decimals=4, shown=False)]
            rows += [(f"Varsayılan ({words}): geleneksel", "gel_nn", len(default)),
                     (f"Varsayılan ({words}): HC1", "rob_nn", len(default))]
        columns = (((f"F({q}, {df})", "f125", "deger"), ("p-değeri", "p125", "deger")) if same_terms else
                   (("F", "f125", "deger"), ("p-değeri", "p125", "deger"), ("Kısıt sayısı q", "q125", "deger")))
        return (
            *tests,
            ModelValue("sd7", level.name, "df_resid", "Payda serbestlik derecesi n − k − 1", decimals=0),
            Scalar("q7", E.const(q), "Kısıt sayısı q", decimals=0),
            *((ModelValue("t7_rob", "m_rob", "t", f"{cov} dayanıklı t istatistiği", term=terms[0], decimals=3),
               Scalar("t7_kare", E.power(E.ref("t7_rob"), 2), "t² (= F)", decimals=3, shown=False)) if q == 1 else ()),
            ScalarTable(tuple((label, E.ref(f"F_{key}")) for label, key, _ in rows), "f125", decimals=3,
                        heading="Kovaryans hesabı"),
            ScalarTable(tuple((label, E.ref(f"p_{key}")) for label, key, _ in rows), "p125", decimals=4,
                        heading="Kovaryans hesabı"),
            *(() if same_terms else (ScalarTable(tuple((label, E.const(count)) for label, _, count in rows), "q125",
                                                 decimals=0, heading="Kovaryans hesabı"),)),
            JoinColumns("tablo125", columns, decimals=3, heading="Kovaryans hesabı", p_columns=("p-değeri",),
                        column_decimals=(("p-değeri", 4),) + (() if same_terms else (("Kısıt sayısı q", 0),)),
                        title=(f"{_prefix(case)}{_hypothesis_raw(case, default)}: ortak test (Tablo 12.5'teki gibi)"
                               if same else f"{test}, H₀: {hypothesis}" + ("" if same_terms else
                                                                         "; varsayılan test yanda"))),
            plot,
        )

    def note(state, choices) -> str:
        cov, terms = choices["adim4_hc"], tuple(choices["adim7_terimler"])
        if exact:
            return "Uyum tam olduğu için F ve p yuvarlama gürültüsüdür; karar yazılmaz (§12.8)." + EXACT_NOTE
        s = state.scalars
        same = (s["p_gel"] < 0.05) == (s["p_rob"] < 0.05)
        text = (f"Aynı hipotez ({_hypothesis(case, terms)}), aynı katsayılar: geleneksel F = {sayi(s['F_gel'], 3)} "
                f"({_p(s['p_gel'])}), {cov} dayanıklı F = {sayi(s['F_rob'], 3)} ({_p(s['p_rob'])}). "
                + ("Yüzde 5 düzeyinde iki hesap aynı kararı verir. " if same else
                   "Yüzde 5 düzeyinde iki hesap farklı karar verir: kovaryans türü ortak testin sonucunu da "
                   "değiştirebilir. "))
        if len(terms) == 1:
            shown_t = float(f"{s['t7_rob']:.3f}")
            square = sayi(shown_t ** 2, 3)
            exact_square = "" if square == sayi(s["t7_kare"], 3) else f" (yuvarlanmamış t ile {sayi(s['t7_kare'], 3)} = F)"
            text += (f"Tek kısıtta F = t²: {cov} dayanıklı t = {sayi(s['t7_rob'], 3)}, t² = {square}{exact_square}; F "
                     "testi dayanıklı t testiyle aynı kararı verir. ")
        if cov != "HC1":
            text += f"Varsayılan HC1 kovaryansıyla F = {sayi(s['F_rob_n'], 3)} ({_p(s['p_rob_n'])}). "
        return (text + "Tekli testler dayanıklı hesapla yapılıyorsa ortak test de aynı kovaryansla yapılır; tabloda testin "
                "hangi kovaryansla hesaplandığı yazılır (§12.8).")

    checks = (_scalar("F_gel", "Geleneksel F", 3), _scalar("p_gel", "Geleneksel p", 4),
              _scalar("F_rob", "HC1 dayanıklı F", 3), _scalar("p_rob", "HC1 dayanıklı p", 4))
    return _step(
        number=7,
        title="Dayanıklı ortak test",
        note=NoteRef("12.8", 0, ("Tablo 12.5", "Kod 12.5")),
        explanation=(
            "Klasik $F$ istatistiği homoskedastik kovaryansa dayanır. Dayanıklı ortak test (Wald testi) sınanan "
            "katsayıların tahminlerini onların dayanıklı varyans ve kovaryanslarıyla ölçekler; yazılım sonucu kısıt "
            "sayısı $q$'ya bölerek $F$ biçiminde raporlar ve p-değerini yaklaşık olarak $F(q, n - k - 1)$ dağılımından "
            "okur. Hipotez aynıdır; değişen, katsayıların ortak belirsizliğinin ölçüsüdür."
            + (" Sınanan katsayıları değiştirin: varsayılan test yan yana gösterilir." if _live(control) else "")
        ),
        controls=(control,),
        uses=(HC_4,),
        build=build,
        checks=_stable(checks, exact),
        note_for=note,
    )


# --- Adım 8: ikinci veride düzey ve log sonuç ------------------------------------------------------------------------

def _second(case: Case) -> Case:
    return second(case)


def _second_outcomes(case: Case) -> tuple[str, str]:
    """İkinci verinin (düzey, log) sonuçları: alternatif örnekte wage ve lwage; kendi verinde sonuç ve ln(sonuç)."""

    other = _second(case)
    if not case.own:
        return other.roles[SONUC], str(other.extra["log_outcome"])
    return case.roles[SONUC], _names(case).ly


def _second_options(case: Case) -> tuple[str, ...]:
    other = _second(case)
    return tuple(other.extra.get("second_x") or candidates(other))


def _second_default(case: Case) -> tuple[str, ...]:
    other = _second(case)
    return tuple(other.extra.get("second_default") or _second_options(case))


def _binary(data: pd.DataFrame, column: str) -> bool:
    values = set(np.unique(data[column].to_numpy(dtype=float)))
    return values <= {0.0, 1.0}


def _step8(case: Case) -> LabStep:
    title = case.extra.get("step8_title", "İkinci veri: düzey ve log sonuç modellerinde tanı ve dayanıklı standart hata")
    note_ref = NoteRef("12.10", 0)
    lead = ("Heteroskedastisiteyle karşılaşıldığında tek bir mekanik çözüm yoktur: yatay kesit çıkarımında dayanıklı "
            "standart hata temel seçenektir; fonksiyonel biçim iktisadi gerekçeyle yeniden düşünülür; ağırlıklı en küçük "
            "kareler yalnız güvenilir varyans bilgisi varsa düşünülür.")
    if case.own and not positive(case, case.roles[SONUC]):
        return LabStep(number=8, title=title, note=note_ref,
                       explanation=f"{lead} Bu adım için sonucun bütün değerleri pozitif olmalı (log sonuç modeli).")
    other = _second(case)
    level, log = _second_outcomes(case)
    options = _second_options(case)
    default = _second_default(case)
    base = other.roles[ACIKLAYICI]
    data = other.data.copy() if not case.own else _data(case, logs=False)
    if case.own:
        data[log] = np.log(data[case.roles[SONUC]].astype(float))
    control = MultiChoice("adim8_x", "Açıklayıcı değişkenler", tuple((term, display(other, term)) for term in options),
                          default, help="Varsayılan: " + ", ".join(other.name(term) for term in default) + ".")
    models = (("wd", level, "Düzey sonuç"), ("wl", log, "Log sonuç"))
    exact = {key: _exact(case, data, outcome, default) for key, outcome, _ in models}
    se_digits = {key: _d(float(_fit(data, outcome, default).bse[base]), 3 if key == "wd" else 4)
                 for key, outcome, _ in models} if base in default else {"wd": 3, "wl": 4}
    setup = (*other.load, *((Derive(other.frame, log, E.log(E.var(level)), f"ln({case.name(level)})"),)
                            if case.own else ()))

    def fits(terms: tuple[str, ...], suffix: str, note: str) -> tuple:
        operations: list = []
        for key, outcome, heading in models:
            name = f"{key}{suffix}"
            operations += [
                OLS(f"m_{name}", other.frame, outcome, terms, f"{note}{heading}: {_formula(outcome, terms)}"),
                OLS(f"m_{name}_hc1", other.frame, outcome, terms, f"{note}{heading}, HC1 dayanıklı kovaryansla",
                    cov_type="HC1"),
                HeteroskedasticityTest(f"bp_{name}", f"p_bp_{name}", f"m_{name}", "bp", f"{note}{heading}: Breusch–Pagan",
                                       shown=False),  # sayılar tabloda
                HeteroskedasticityTest(f"wh_{name}", f"p_wh_{name}", f"m_{name}", "white", f"{note}{heading}: White",
                                       shown=False),
            ]
        if base in terms:
            operations += [ModelValue(f"sh_x_{key}{cov}{suffix}", f"m_{key}{suffix}{cov}", "se",
                                      f"{note}{heading}: {other.name(base)} katsayısının {label} SH'si", term=base,
                                      decimals=se_digits[key], shown=False)
                           for key, _, heading in models for cov, label in (("", "geleneksel"), ("_hc1", "HC1"))]
        return tuple(operations)

    def build(choices) -> tuple:
        terms = tuple(choices["adim8_x"])
        same = terms == default
        rows = [(heading, key) for key, _, heading in models]
        side: tuple = ()
        if not same:  # varsayılan model yan yana
            side = fits(default, "_n", "Varsayılan — ")
            rows = [(f"Varsayılan: {heading.lower()}", f"{key}_n") for key, _, heading in models] + [
                (f"Seçiminiz: {heading.lower()}", key) for key, _, heading in models]
        se_rows = tuple((f"{heading}, {label}", f"{key}{cov}") for key, _, heading in models
                        for cov, label in (("", "geleneksel"), ("_hc1", "HC1")))
        se_tables: tuple = ()
        if not same:
            se_tables = (ScalarTable(tuple((label, E.ref(f"sh_x_{key}_n")) for label, key in se_rows), "x_varsayilan",
                                     decimals=4, heading="Model ve standart hata", value="Varsayılan",
                                     title="" if base in terms else
                                     f"Varsayılan model: {other.name(base)} katsayısının standart hataları"),)
            if base in terms:
                se_tables += (ScalarTable(tuple((label, E.ref(f"sh_x_{key}")) for label, key in se_rows), "x_secim",
                                          decimals=4, heading="Model ve standart hata"),
                              JoinColumns("x_sh", (("Varsayılan", "x_varsayilan", "deger"),
                                                   ("Seçiminiz", "x_secim", "deger")),
                                          decimals=4, heading="Model ve standart hata",
                                          title=f"{other.name(base)} katsayısının standart hatası: varsayılan model ve "
                                                "seçiminiz"))
        return (
            *setup,
            *fits(terms, "", ""),
            *side,
            *(ScalarTable(tuple((heading, E.ref(f"{prefix}_{key}")) for heading, key in rows), f"{prefix}_1210",
                          decimals=3, heading="Model") for prefix in ("bp", "p_bp", "wh", "p_wh")),
            JoinColumns("testler1210", (("BP LM", "bp_1210", "deger"), ("BP p", "p_bp_1210", "deger"),
                                        ("White LM", "wh_1210", "deger"), ("White p", "p_wh_1210", "deger")),
                        decimals=3, heading="Model", p_columns=("BP p", "White p"),
                        title=f"{_prefix(other)}düzey ve log sonuç modellerinde heteroskedastisite testleri" + (
                            "" if same else " (varsayılan model ve seçtiğiniz açıklayıcılar)")),
            RegressionTable((("(1) Düzey, geleneksel", "m_wd"), ("(2) Düzey, HC1", "m_wd_hc1"),
                             ("(3) Log, geleneksel", "m_wl"), ("(4) Log, HC1", "m_wl_hc1")),
                            (*terms, INTERCEPT), "tablo1210", f"{_prefix(other)}aynı katsayılar, geleneksel ve HC1 "
                            "standart hatalar" + ("" if same else " (seçtiğiniz açıklayıcılar)"), decimals=4,
                            exact=not case.own),
            *se_tables,
        )

    def note(state, choices) -> str:
        terms = tuple(choices["adim8_x"])
        same = terms == default
        s = state.scalars
        if any(_exact(case, data, outcome, terms) for _, outcome, _ in models):
            return ("Uyum tam olduğu için artıklar yuvarlama gürültüsüdür; testler ve standart hatalar yorumlanmaz "
                    "(§12.10)." + EXACT_NOTE)
        text = (f"Düzey sonuç modelinde BP {_p(s['p_bp_wd'], 3)}, White {_p(s['p_wh_wd'], 3)}; log sonuç modelinde BP "
                f"{_p(s['p_bp_wl'], 3)}, White {_p(s['p_wh_wl'], 3)}. ")
        if len(terms) == 1 and _binary(data, terms[0]):
            text += ("Tek bir 0/1 kukla açıklayıcıda White testi Breusch–Pagan testiyle aynıdır: kuklanın karesi "
                     "kendisidir. ")
        if s["p_bp_wl"] >= 0.05 > s["p_wh_wl"] or s["p_wh_wl"] >= 0.05 > s["p_bp_wl"]:
            text += "Log modelde iki test aynı sonucu vermez. "
        if "sh_x_wd" in s:
            text += (f"{capital(phrase(other, base))} katsayısının standart hatası düzey modelinde "
                     f"{sayi(s['sh_x_wd'], se_digits['wd'])} (geleneksel) ve {sayi(s['sh_x_wd_hc1'], se_digits['wd'])} "
                     f"(HC1), log modelde {sayi(s['sh_x_wl'], se_digits['wl'])} ve "
                     f"{sayi(s['sh_x_wl_hc1'], se_digits['wl'])}. ")
        if not same:
            text += (f"Varsayılan modelde log sonuç için BP {_p(s['p_bp_wl_n'], 3)}, White {_p(s['p_wh_wl_n'], 3)}"
                     + (f"; {phrase(other, base)} katsayısının standart hatası log modelde "
                        f"{sayi(s['sh_x_wl_n'], se_digits['wl'])} (geleneksel) ve "
                        f"{sayi(s['sh_x_wl_hc1_n'], se_digits['wl'])} (HC1)" if base in default else "") + ". ")
        rejects = s["p_bp_wl"] < 0.05 or s["p_wh_wl"] < 0.05
        return (text + ("Log dönüşümü sonrasında da testlerden en az biri homoskedastisiteyi reddeder: log modelde de "
                        "dayanıklı standart hata raporlamak gerekir. " if rejects else
                        "Yatay kesitte heteroskedastisite makul bir olasılıktır: testler reddetmese de log modelde "
                        "dayanıklı standart hata raporlamak makuldür. ")
                + "Dayanıklı standart hata eksik değişken ya da nedensellik sorununu çözmez (§12.10).")

    checks = [_scalar(f"p_{test}_{key}", f"{heading}: {name} p", 3)
              for key, _, heading in models for test, name in (("bp", "BP"), ("wh", "White"))]
    if base in default:
        checks += [_scalar(f"sh_x_{key}{cov}", f"{heading}: {other.name(base)} katsayısının {label} SH'si",
                           se_digits[key])
                   for key, _, heading in models for cov, label in (("", "geleneksel"), ("_hc1", "HC1"))]
    checks = [check for check in checks
              if not (exact["wd"] or exact["wl"]) or not isinstance(check.target, ScalarTarget)]
    return _step(
        number=8,
        title=title,
        note=note_ref,
        explanation=(f"{lead} {other.extra.get('second_text', 'Aynı dosyanın düzey ve log sonuç modelleri ayrı bir çerçevede karşılaştırılır.')}"
                     + (" Açıklayıcı değişkenleri değiştirin." if _live(control) else "")),
        controls=(control,),
        build=build,
        checks=tuple(checks),
        note_for=note,
    )


# --- Tanım ------------------------------------------------------------------------------------------------------

def build(case: Case) -> LabSpec:
    """Konu 12 uygulamasını verilen veriyle kurar; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    labels = labels_of(case)
    other = _second(case)
    for column, label in labels_of(other).items():
        labels.setdefault(column, label)
    labels[INTERCEPT] = "Sabit terim"
    if case.own:
        names = _names(case)
        labels.setdefault(names.ly, f"ln({case.name(case.roles[SONUC])})")
        labels.setdefault(names.lx, f"ln({case.name(case.roles[ACIKLAYICI])})")
    labels.update({"artik": "Artık û", "artik2": "Kareli artık û²", "tahmin": "Tahmin edilen değer Ŷ",
                   "mutlak": "Mutlak artık |û|", "ceyrek": "Tahmin edilen değerin çeyreği", "n": "Gözlem sayısı",
                   "ss": "Artıkların standart sapması", "ort_mutlak": "Ortalama mutlak artık"})
    short = dict(_plot_labels(case))
    for model in _models(case).values():  # White yardımcı regresyonunun kare ve çarpım sütunları
        for name, first, second_ in _white_pairs(case, model.terms):
            one, two = short.get(first, labels.get(first, first)), short.get(second_, labels.get(second_, second_))
            labels.setdefault(name, f"({one})²" if first == second_ else f"{one} × {two}")
    spec = LabSpec(
        topic_key=TOPIC,
        title=str(case.extra.get("title", TITLE)),
        note_section="12",
        steps=(_step1(case), _step2(case), _step3(case), _step4(case), _step5(case), _step6(case), _step7(case),
               _step8(case)),
        labels=tuple(labels.items()),
        source=case.source,
    )
    return with_app_values(spec)


# --- Alternatif örnek ----------------------------------------------------------------------------------------

def _second_case() -> Case:
    """WAGE2 (notlardaki WAGE1 yerine): aylık kazanç ve log aylık kazanç modelleri."""

    base = wage2_case(("educ", "exper", "tenure", "married", "black", "south", "urban"),
                      second_x=("educ", "exper", "tenure", "married", "black", "south", "urban"),
                      second_default=("educ", "exper", "tenure", "married"), log_outcome="lwage",
                      second_text=("WAGE2'de (935 erkek çalışan, 1980) aylık kazanç (düzey) ve log aylık kazanç modelleri "
                                   "karşılaştırılır."))
    extra_labels = {"married": ("Evli", "0/1 gösterge"), "black": ("Siyah", "0/1 gösterge"),
                    "south": ("Güneyde yaşıyor", "0/1 gösterge"), "urban": ("Büyükşehir alanında yaşıyor", "0/1 gösterge")}
    labels = {**base.labels, **{name: label for name, (label, _) in extra_labels.items()}}
    units = {**base.units, **{name: unit for name, (_, unit) in extra_labels.items()}}
    return replace(base, labels=labels, units=units)


def alternative_case() -> Case:
    level_terms = ("land1000", "area100", "rooms", "baths")
    log_terms = ("lland", "larea", "rooms", "baths")
    return house_case(
        level_terms, outcome="price1000", base="land1000", columns=(*HOUSE_COLUMNS, "lland"),
        derived=("price1000", "area100", "land1000"),
        title="Uygulama: Heteroskedastisite ve Heteroskedastisiteye Dayanıklı Çıkarım (KIELMC ve WAGE2)",
        models={"duzey": ("m_duzey", "price1000", level_terms, "Fiyat düzeyi", "Tahmin edilen konut fiyatı (bin dolar)",
                          " (bin dolar)"),
                "log": ("m_log", "lprice", log_terms, "Log fiyat", "Tahmin edilen log konut fiyatı", " (log birimi)")},
        plot_labels=(("land1000", "Arsa / 1.000"), ("area100", "Konut / 100"), ("rooms", "Oda"), ("baths", "Banyo"),
                     ("lland", "log(arsa)"), ("larea", "log(konut)")),
        raw={"land1000": "land", "lland": "land"},
        joint=("land1000", "rooms"),
        member_gen="konutun",
        model_text=("KIELMC'nin 1978 satışları (179 konut): düzey modeli `price1000 ~ land1000 + area100 + rooms + baths` "
                    "(fiyat bin dolar, arsa bin fit², konut büyüklüğü yüz fit²), log modeli `lprice ~ lland + larea + "
                    "rooms + baths`. "),
        step4_title="KIELMC (1978): aynı katsayı, farklı çıkarım",
        step8_title="WAGE2: düzey ve log kazanç modellerinde tanı ve dayanıklı standart hata",
        second=_second_case(),
    )


@cache
def alternative() -> LabSpec:
    return build(alternative_case())


STORY = ("Ana veri KIELMC'nin 1978 satışlarıdır (179 konut): fiyat ~ arsa + konut büyüklüğü + oda + banyo, düzey ve log "
         "modelleri. Düzey modelinde testler homoskedastisiteyi reddeder ve en büyük arsalı konutun kaldıracı yüksektir; "
         "log modelde testler reddetmez. İkinci veri WAGE2'dir (aylık kazanç ve log aylık kazanç).")


# --- Kendi verin ---------------------------------------------------------------------------------------------

def own_validate(case: Case) -> None:
    """Notlardaki kurallar; hiçbir gözlemin kaldıracı 1 değil (HC2 ve HC3 tanımlı); White yardımcı regresyonu için
    yeterli gözlem."""

    validate(case)
    data = _data(case, logs=False)
    terms = candidates(case)
    if float(_leverage(data, terms).max()) >= 1 - 1e-8:
        raise K.UploadError("Bir gözlemin kaldıracı 1: HC2 ve HC3 standart hataları hesaplanamaz (ör. yalnız bir gözlemde "
                            "1 olan bir kukla). Bu değişkeni ek değişkenlerden çıkarın.")
    count = _white_count(data, terms)
    if len(data) < count + 2:
        raise K.UploadError(f"White testinin yardımcı regresyonunda {count} terim var; analizde {len(data)} gözlem var. "
                            f"White testi için en az {count + 2} gözlem gerekir: daha az ek değişken seçin.")


CUSTOM = custom_lab(
    build,
    sample_employed,
    ("Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sonuç ve temel açıklayıcı değişken zorunludur ve sayısal olmalıdır. "
     "Düzey modeli sonuç ~ temel açıklayıcı + ek değişkenlerdir. Log modeli (ln(sonuç) ~ ln(temel açıklayıcı) + ek "
     "değişkenler) için sonucun ve temel açıklayıcının bütün değerleri pozitif olmalı; Adım 8'in log sonuç modeli için "
     f"sonucun pozitif olması yeter. Ek sayısal değişkenler en çok 3 tanedir. {ROW_RULE}"),
    roles((1, 2, 3, 4, 5, 6, 7, 8)),
    "Modelin ek açıklayıcılarıdır (log modelde düzeyde kalırlar).",
    validate=own_validate,
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
