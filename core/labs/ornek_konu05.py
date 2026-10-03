"""Konu 5 genel uygulaması: çoklu regresyon modeli ve ceteris paribus yorumu.

Notlardaki on adım (``core.labs.konu05``) aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde
yazılır: basit ve çoklu model, birden fazla birimlik fark ve iki gözlemin karşılaştırılması, tahmin edilen değer ve
artık, kontrollerin doğrusal katkısını ayırma (kısmi regresyon), yazılım çıktısı ve basit–çoklu model tablosu,
katsayı değişimi, ikinci bir veriyle çoklu model (konut fiyatı), R² ve düzeltilmiş R², makale tablosu ve kontrol
değişkeni seçimi.

Alternatif örnekte ücret adımları WAGE2 (935 erkek çalışan, 1980; aylık kazanç, ABD doları), konut adımı KIELMC'nin
1978 satışlarıdır (179 konut). Adım 8'in alıştırma tablosu notlardaki varsayımsal R² değerleri yerine bu uygulamanın
üç modelini (basit model, seçilen çoklu model, bütün seçenekleri içeren model) seçilen gözlem sayısıyla karşılaştırır.
"Kendi verini yükle" seçeneğinde bütün adımlar öğrencinin dosyasıyla kurulur: konut adımı aynı dosyanın sonuç ve
açıklayıcı değişkenlerini kullanır; çoklu model için en az bir ek sayısal değişken gerekir.

Etkileşim notlardaki gibidir: modelin açıklayıcı değişkenleri (Adım 1; Adım 2, 3, 5, 6 ve 8 aynı modeli kullanır),
açıklayıcı değişkendeki fark (Adım 2), tahmin edilen gözlemin özellikleri (Adım 3), kısmi ilişkisi incelenen değişken
(Adım 4), ikinci modelin açıklayıcı değişkenleri (Adım 7), alıştırmadaki gözlem sayısı (Adım 8) ve makale tablosunun
sütunları (Adım 9).
"""

from __future__ import annotations

from functools import cache

import numpy as np

from core.labs import expr as E
from core.labs.ornek import Case, TopicVariants, sayi, sayim, stable_checks, with_app_values
from core.labs.ornek_regresyon import (
    ACIKLAYICI,
    EXACT_MULTI_NOTE,
    ROW_RULE,
    SONUC,
    amount,
    candidates,
    capital,
    change,
    coefficients,
    custom_lab,
    digits_for,
    display,
    exact_multi,
    gap_control,
    house_case,
    labels_of,
    level_text,
    listing,
    log_column,
    log_label,
    log_operations,
    negligible,
    nice,
    noise_decimals,
    number_control,
    options,
    outcome_words,
    phrase,
    plural,
    positive,
    reload,
    roles,
    rough,
    sample_employed,
    second,
    short_unit,
    step_words,
    wage2_case,
)
from core.labs.spec import (
    COEF_QUANTITIES,
    INTERCEPT,
    OLS,
    CellTarget,
    Check,
    Choice,
    CoefTarget,
    Derive,
    GroupedBarChart,
    InlineData,
    JoinColumns,
    LabSpec,
    LabStep,
    LineChart,
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
    ShowModel,
    Statistic,
    TableTarget,
    interactive_step,
)

TOPIC = "konu05"
TITLE = "Uygulama: Çoklu Regresyon Modeli ve Ceteris Paribus Yorumu"
X_KEY = "adim1_x"
ALT_OPTIONS = ("educ", "exper", "tenure", "IQ")
ALT_DEFAULT = ("educ", "exper", "tenure")


def _check(label: str, target, decimals: int) -> Check:
    return Check(label, target, 0.0, decimals)


def _scalar(name: str, label: str, decimals: int) -> Check:
    return _check(label, ScalarTarget(name), decimals)


def _name(case: Case, column: str) -> str:
    """Başlık ve kod yorumundaki ad (Markdown değil)."""

    return case.name(column) if case.own else phrase(case, column)


def _title(case: Case, column: str) -> str:
    """Tablo satırındaki ad: alternatif örnekte büyük harfle, kendi verinde dosyadaki ad."""

    return case.name(column) if case.own else capital(phrase(case, column))


def _default(case: Case) -> tuple[str, ...]:
    """Varsayılan çoklu model: alternatif örnekte notlardaki gibi üç değişken, kendi verinde bütün seçenekler."""

    return tuple(case.extra.get("default_regressors") or candidates(case))


def _ordered(case: Case, names) -> tuple[str, ...]:
    chosen = set(names)
    return tuple(name for name in candidates(case) if name in chosen)


def _predict(regressors, values: dict, prefix: str = "b_") -> E.Expr:
    """Tahmin edilen değer: sabit + Σ katsayı × değer; katsayılar ``prefix``le adlandırılmış skalerlerdir."""

    expression = E.ref(f"{prefix}sabit")
    for name in regressors:
        expression = E.add(expression, E.mul(E.ref(f"{prefix}{name}"), values[name]))
    return expression


def _interpretation(case: Case, name: str, value: float, others: list[str]) -> str:
    """Bir katsayının ceteris paribus cümlesi: değişen değişken, tahmin edilen fark ve sabit tutulanlar."""

    phrase_ = step_words(case, name)
    if others:
        held = listing([phrase(case, other) for other in others])
        start = f"{capital(held)} aynıyken, {phrase_}"
    else:
        start = capital(phrase_)
    y = case.roles[SONUC]
    zero = negligible(case, value, y, name)
    digits = digits_for(value, int(case.extra.get("coefficient_digits", 3)))
    return (f"{start} {plural(case)} tahmin edilen {outcome_words(case)} "
            f"{change(value, amount(case, y, abs(value), digits), zero)}.")


def _median(case: Case, column: str) -> float:
    """Profil değeri: medyan, iki anlamlı basamağa yuvarlanmış (tam sayı veride tam sayı)."""

    values = case.data[column]
    median = float(values.median())
    if np.all(values == np.round(values)):
        return float(round(median))
    return nice(median, 2)


def x_multi(case: Case) -> MultiChoice:
    y = case.roles[SONUC]
    default = _default(case)
    return MultiChoice(
        X_KEY, f"Çoklu modelin açıklayıcı değişkenleri (bağımlı değişken: {case.name(y)})",
        options(case, candidates(case)), default,
        help=("Varsayılan: " + ", ".join(case.name(name) for name in default) + ". Adım 2, 3, 5, 6 ve 8 bu modeli "
              "kullanır."),
    )


# --- Adım 1: basit modelden çoklu modele --------------------------------------------------------------------

def _step1(case: Case, choice: MultiChoice) -> LabStep:
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    default = _default(case)

    def build(choices) -> tuple:
        chosen = choices[X_KEY]
        formula = " + ".join(chosen)
        same = tuple(chosen) == default
        operations: list = [
            *case.load,
            OLS("basit", case.frame, y, (x,), f"Basit regresyon: {y} ~ {x}"),
            ModelValue("basit_sabit", "basit", "coef", "Basit model: sabit terim", term=INTERCEPT),
            ModelValue("basit_x", "basit", "coef", f"Basit model: {_name(case, x)} katsayısı", term=x),
            OLS("coklu", case.frame, y, tuple(chosen), f"Çoklu regresyon: {y} ~ {formula}"),
            ModelValue("b_sabit", "coklu", "coef", "Çoklu model: sabit terim", term=INTERCEPT),
            *(ModelValue(f"b_{name}", "coklu", "coef", f"Çoklu model: {_name(case, name)} katsayısı", term=name)
              for name in chosen),
        ]
        columns = [("(1) Basit model", "basit"), ("(2) Çoklu model" if same else "(2) Seçtiğiniz model", "coklu")]
        used = {x, *chosen}
        if not same:
            operations.append(OLS("coklu_varsayilan", case.frame, y, default,
                                  f"Karşılaştırma için varsayılan model: {y} ~ {' + '.join(default)}"))
            columns.append(("(3) Varsayılan model", "coklu_varsayilan"))
            used |= set(default)
        title = (f"Basit ve çoklu model (bağımlı değişken: {case.name(y)})" if same else
                 f"Basit model, seçtiğiniz model ve varsayılan model (bağımlı değişken: {case.name(y)})")
        operations.append(RegressionTable(tuple(columns), (*_ordered(case, used), INTERCEPT), "modeller", title,
                                          stars=False, decimals=4, standard_errors=False, r2=False))
        return tuple(operations)

    def note(state, choices) -> str:
        chosen = list(choices[X_KEY])
        s = state.scalars
        text = " ".join(_interpretation(case, name, s[f"b_{name}"], [other for other in chosen if other != name])
                        for name in chosen)
        if len(chosen) == 1:
            text += " Modelde tek açıklayıcı değişken var; katsayı bir basit regresyon eğimidir."
        if x not in chosen:
            text += f" {capital(phrase(case, x))} modelde yok; onun farkları hata teriminde kalır."
        elif len(chosen) > 1:
            text += (f" {capital(phrase(case, x))} katsayısı basit modelde {sayi(s['basit_x'], 4)}, çoklu modelde "
                     f"{sayi(s[f'b_{x}'], 4)}: iki katsayı farklı karşılaştırmalar yapar (§5.7).")
        if len(chosen) == 1:
            return text + (" Tek değişkenli modelde sabit tutulan başka bir değişken yoktur: “diğerleri aynıyken” "
                           "unsuru ancak çoklu modelde vardır. Yorum bir ilişkidir; nedensel etki iddiası değildir.")
        return text + (" Her yorumda üç unsur vardır: değişen değişken ve birimi, bağımlı değişkendeki tahmin edilen fark "
                       "ve birimi, sabit tutulan değişkenler. Ceteris paribus yorum yalnız modelde bulunan değişkenleri "
                       "sabit tutar; nedensel etki iddiası değildir.")

    return interactive_step(
        number=1,
        title="Basit modelden çoklu modele: katsayıların ceteris paribus yorumu",
        note=NoteRef("5.3", 0, ("§5.1", "Denklem 5.5")),
        explanation=(
            f"Basit regresyon (`{y} ~ {x}`) "
            f"{dict(case.extra.get('subjects', {})).get(x, phrase(case, x) + ' değeri')} farklı "
            f"{case.extra.get('unit_accusative', 'gözlemleri')}, diğer farkları ayırmadan karşılaştırır. Çoklu regresyonda her eğim katsayısı, modeldeki diğer açıklayıcı değişkenler aynıyken "
            "yorumlanır (Denklem 5.5'in karşılığı). Modelin açıklayıcı değişkenlerini seçin: tablo basit modeli ve "
            "seçtiğiniz çoklu modeli yan yana gösterir. Adım 2, 3, 5, 6 ve 8 bu modeli kullanır."
        ),
        controls=(choice,),
        build=build,
        checks=(
            _check("Basit model: sabit terim", CoefTarget("basit", INTERCEPT), 4),
            _check("Basit model: eğim", CoefTarget("basit", x), 4),
            _check("Çoklu model: sabit terim", CoefTarget("coklu", INTERCEPT), 4),
            *(_check(f"Çoklu model: {case.name(name)} katsayısı", CoefTarget("coklu", name), 4) for name in default),
            *(_scalar(f"b_{name}", f"Yorumdaki {case.name(name)} katsayısı", 3) for name in default),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 2: birden fazla birimlik fark ve iki gözlem -------------------------------------------------------

def _line_setup(case: Case) -> tuple[str | None, tuple[float, ...]]:
    """Çizgilerin değişkeni ve düzeyleri: alternatif örnekte deneyim 5/12/20, kendi verinde varsayılan modelin ikinci
    değişkeninin %10, %50 ve %90 değerleri."""

    if "line" in case.extra:
        return case.extra["line"]
    x = case.roles[ACIKLAYICI]
    others = [name for name in _default(case) if name != x]
    if not others:
        return None, ()
    column = others[0]
    values = case.data[column]
    found = []
    for q in (0.1, 0.5, 0.9):
        value = float(values.quantile(q))
        value = float(round(value)) if np.all(values == np.round(values)) else nice(value, 2)
        if value not in found:
            found.append(value)
    return column, tuple(found)


def _grid(case: Case) -> tuple[float, ...]:
    """Çizgilerin yatay ekseni: alternatif örnekte eğitimin aralığı (9–18), kendi verinde 21 eşit aralıklı nokta."""

    if "grid" in case.extra:
        return tuple(case.extra["grid"])
    values = case.data[case.roles[ACIKLAYICI]]
    low, high = float(values.min()), float(values.max())
    return tuple(float(nice(value, 4)) for value in np.linspace(low, high, 21))


def _profile(case: Case) -> dict[str, float]:
    """A gözleminin özellikleri: alternatif örnekte eğitim 12, deneyim 10, kıdem 5, IQ 100; kendi verinde medyanlar."""

    if "profile" in case.extra:
        return dict(case.extra["profile"])
    return {name: _median(case, name) for name in candidates(case)}


def _gap_control(case: Case) -> NumberChoice:
    if "adim2" in case.extra:
        return case.extra["adim2"]
    return gap_control(case, "adim2_fark", f"Fark: B'nin {case.name(case.roles[ACIKLAYICI])} değeri A'nınkinden bu "
                       "kadar fazla", "Varsayılan: açıklayıcının yaklaşık bir standart sapması.")


def _step2(case: Case, choice: MultiChoice) -> LabStep:
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    control = _gap_control(case)
    line, line_levels = _line_setup(case)
    grid = _grid(case)
    profile = _profile(case)
    names = tuple(candidates(case))
    default = _default(case)
    gap_word = str(case.extra.get("gap_word", "birim"))
    subject = dict(case.extra.get("subjects", {})).get(x, f"{phrase(case, x)} değeri")

    def build(choices) -> tuple:
        chosen = choices[X_KEY]
        gap = choices["adim2_fark"]
        rows = (("A", *(profile[name] for name in names)),
                ("B", *(profile[name] + (gap if name == x else 0) for name in names)))
        operations: list = []
        if x in chosen:
            operations.append(Scalar("x_farki", E.mul(gap, E.ref(f"b_{x}")),
                                     f"{level_text(gap)} birimlik fark: {level_text(gap)} × β̂ ({_name(case, x)})",
                                     decimals=3))
        operations += [
            InlineData("calisanlar", ("calisan", *names), rows,
                       f"İki gözlem: {_name(case, x)} dışındaki özellikler aynı"),
            Derive("calisanlar", "tahmin", _predict(chosen, {name: E.var(name) for name in chosen}),
                   "Tahmin edilen değer Ŷ"),
            Statistic("calisanlar", "tahmin", "value", "tahmin_A", "A'nın tahmini", where=("calisan", "A"), decimals=2),
            Statistic("calisanlar", "tahmin", "value", "tahmin_B", "B'nin tahmini", where=("calisan", "B"), decimals=2),
            Scalar("fark_BA", E.sub(E.ref("tahmin_B"), E.ref("tahmin_A")), "Tahmin farkı B − A", decimals=3),
        ]
        if line is not None and {x, line} <= set(chosen):
            held_names = [name for name in chosen if name not in (x, line)]
            held = {x: E.var(x), **{name: E.ref(f"medyan_{index}") for index, name in enumerate(held_names, start=1)}}
            fixed = [f"{_name(case, name)} medyanında" for name in held_names]
            title = (f"{capital(listing(fixed))} sabitken üç" if fixed else "Üç") + \
                f" {_name(case, line)} düzeyinde tahmin çizgileri (Şekil 5.1'deki gibi)"
            operations += [
                *(Statistic(case.frame, name, "median", f"medyan_{index}", f"{case.name(name)} medyanı", decimals=2)
                  for index, name in enumerate(held_names, start=1)),
                InlineData("izgara", (x,), tuple((value,) for value in grid),
                           f"{case.name(x)} ızgarası: verideki aralık"),
                *(Derive("izgara", f"tahmin_d{index}", _predict(chosen, {**held, line: level}),
                         f"{case.name(line)} = {level_text(level)}: tahmin edilen değer")
                  for index, level in enumerate(line_levels, start=1)),
                LineChart("izgara", x, "tahmin_d1", display(case, x), f"Tahmin edilen {_name(case, y)}", title,
                          markers=False, legend=f"{case.name(line)} = {level_text(line_levels[0])}",
                          series=tuple((f"tahmin_d{index}", f"{case.name(line)} = {level_text(level)}")
                                       for index, level in enumerate(line_levels[1:], start=2))),
            ]
        return tuple(operations)

    def note(state, choices) -> str:
        chosen = choices[X_KEY]
        gap = choices["adim2_fark"]
        s = state.scalars
        unit = f" {short_unit(case, y)}" if short_unit(case, y) else ""
        if x not in chosen:
            notes = state.models["coklu_varsayilan"].params[x] if "coklu_varsayilan" in state.models else None
            text = (f"{capital(phrase(case, x))} modelde olmadığı için iki gözlemin tahmini aynıdır: model bu farkı "
                    f"kullanmaz. {capital(phrase(case, x))} değişkenini Adım 1'de modele ekleyin.")
            if notes is not None:
                text += (f" Varsayılan modelde fark {level_text(gap)} × {sayi(notes, 4)} = {sayi(gap * notes, 3)}"
                         f"{unit}.")
            return text
        text = (f"B'nin {subject} A'nınkinden {level_text(gap)} {gap_word} fazla; diğer özellikleri aynı. "
                f"Tahminler {sayi(s['tahmin_A'], 2)} ve {sayi(s['tahmin_B'], 2)}{unit}; fark {sayi(s['fark_BA'], 3)}"
                f"{unit}, yani {level_text(gap)} × β̂ = {sayi(s['x_farki'], 3)}. "
                + ("Diğer değişkenler aynı tutulduğu için fark yalnız bu değişkenin katkısıdır. " if len(chosen) > 1 else
                   "Modelde başka değişken olmadığı için fark yalnız bu değişkenin katkısıdır. ")
                + "Birden fazla değişken aynı anda değişirse toplam fark, bütün katsayı katkılarının toplamıdır (§5.3).")
        if tuple(chosen) != default and "coklu_varsayilan" in state.models:
            notes = state.models["coklu_varsayilan"].params[x]
            text += f" Varsayılan modelde aynı fark {level_text(gap)} × {sayi(notes, 4)} = {sayi(gap * notes, 3)}{unit}."
        if line is not None and {x, line} <= set(chosen):
            text += (f" Şekilde doğrular birbirine paraleldir: {phrase(case, line)} düzeyi değişince doğru yukarı ya da "
                     f"aşağı kayar, {phrase(case, x)} eğimi değişmez; çünkü modelde etkileşim terimi yoktur (etkileşimler "
                     "Konu 11).")
        elif line is not None:
            text += (f" {capital(phrase(case, line))} modelde olmadığı için düzeylerine göre çizgiler çizilmez.")
        return text

    profile_text = case.extra.get("profile_text") or (
        "A gözleminin özellikleri değişkenlerin medyanlarıdır: "
        + ", ".join(f"{phrase(case, name)} {level_text(profile[name])}" for name in names)
        + "; B'nin yalnız temel açıklayıcı değişkeni farklıdır.")
    return interactive_step(
        number=2,
        title="Birden fazla birimlik fark ve iki gözlemin karşılaştırılması",
        note=NoteRef("5.3", 0, ("Şekil 5.1",)),
        explanation=(
            "Bir açıklayıcı değişken birden fazla birim farklıysa, diğer değişkenler aynıyken tahmin edilen fark "
            f"katsayı ile farkın çarpımıdır. {profile_text} Farkı değiştirin. Grafik, bir değişken sabitken üç düzeyde "
            "tahmin çizgilerini gösterir (Şekil 5.1'deki gibi). Bu adım Adım 1'deki modeli kullanır."
        ),
        uses=(choice,),
        controls=(control,),
        build=build,
        checks=(
            _scalar("x_farki", "Fark × katsayı", 3),
            _scalar("tahmin_A", "A'nın tahmini", 2),
            _scalar("tahmin_B", "B'nin tahmini", 2),
            _scalar("fark_BA", "B − A", 3),
        ) if x in default else (_scalar("tahmin_A", "A'nın tahmini", 2), _scalar("tahmin_B", "B'nin tahmini", 2)),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 3: tahmin edilen değer ve artık -------------------------------------------------------------------

def _profile_controls(case: Case) -> tuple[NumberChoice, ...]:
    if "adim3" in case.extra:
        return tuple(case.extra["adim3"])
    y = case.roles[SONUC]
    controls = []
    for index, name in enumerate(candidates(case), start=1):
        values = case.data[name]
        integral = bool(np.all(values == np.round(values)))
        controls.append(number_control(f"adim3_d{index}", case.name(name), _median(case, name), float(values.min()),
                                       float(values.max()), help="Varsayılan: medyan.",
                                       step=1.0 if integral and float(values.max() - values.min()) <= 1000 else None))
    values = case.data[y]
    controls.append(number_control("adim3_y", f"Gözlenen değer ({case.name(y)})", _median(case, y),
                                   float(values.min()), float(values.max()), help="Varsayılan: sonucun medyanı."))
    return tuple(controls)


def _step3(case: Case, choice: MultiChoice) -> LabStep:
    y = case.roles[SONUC]
    controls = _profile_controls(case)
    names = tuple(candidates(case))
    keys = dict(zip(names, (control.key for control in controls[:-1])))
    observed_key = controls[-1].key
    unit = f" {short_unit(case, y)}" if short_unit(case, y) else ""

    def build(choices) -> tuple:
        chosen = choices[X_KEY]
        profile = {name: float(choices[keys[name]]) for name in names}
        observed = float(choices[observed_key])
        return (
            Scalar("birey_tahmin", _predict(chosen, profile), "Tahmin edilen değer Ŷ", decimals=3),
            Scalar("birey_artik", E.sub(observed, E.ref("birey_tahmin")), "Artık û = Y − Ŷ", decimals=3),
            Derive(case.frame, "tahmin", _predict(chosen, {name: E.var(name) for name in chosen}),
                   "Tahmin edilen değer Ŷᵢ"),
            Derive(case.frame, "artik", E.sub(E.var(y), E.var("tahmin")), "Artık ûᵢ = Yᵢ − Ŷᵢ"),
            Statistic(case.frame, "artik", "sum", "artik_toplami", "Örneklemde artıkların toplamı Σûᵢ",
                      decimals=noise_decimals(case, regressors=chosen)),
            Statistic(case.frame, "tahmin", "mean", "ort_tahmin", "Tahmin edilen değerlerin ortalaması", decimals=6),
            Statistic(case.frame, y, "mean", "ort_y", "Sonucun ortalaması Ȳ", decimals=6),
        )

    def note(state, choices) -> str:
        chosen = choices[X_KEY]
        s = state.scalars
        observed = float(choices[observed_key])
        residual = s["birey_artik"]
        if residual > 0:
            sign = ("Pozitif artık: gözlenen değer model tahmininden yüksektir. Fark modelde bulunmayan özelliklerden, "
                    "doğrusal biçimin sınırlılığından veya ölçüm hatasından kaynaklanabilir.")
        elif residual < 0:
            sign = "Negatif artık: gözlenen değer model tahmininden düşüktür."
        else:
            sign = "Artık sıfır: gözlenen değer tahminle aynıdır."
        definitions = ("Tanımlar basit regresyondakiyle aynıdır; yalnız tahmin edilen değerde daha fazla açıklayıcı "
                       "değişken vardır." if len(chosen) > 1 else
                       "Modelde tek açıklayıcı değişken var; tanımlar basit regresyondakiyle aynıdır.")
        text = (f"Seçilen özelliklerdeki gözlem için tahmin {sayi(s['birey_tahmin'], 3)}{unit}; gözlenen değer "
                f"{level_text(observed)}{unit} ise artık {sayi(residual, 3)}{unit}. {sign} {definitions}")
        unused = [phrase(case, name) for name in names if name not in chosen]
        if unused:
            text += (f" Modelde olmayan değişkenin ({unused[0]}) değeri tahmini etkilemez." if len(unused) == 1 else
                     f" Modelde olmayan değişkenlerin ({listing(unused)}) değeri tahmini etkilemez.")
        if s["birey_tahmin"] < 0 <= float(case.data[y].min()):
            text += (" Tahmin sıfırın altında: bu özellikler veride çok nadirdir ve doğrusal model verinin kenarında "
                     "anlamsız değerler üretebilir (§3.11).")
        noise = noise_decimals(case, regressors=chosen)
        return text + (f" Sabit terimli EKK'de artıkların toplamı sıfırdır ({sayi(s['artik_toplami'], noise)}, hesap "
                       f"hassasiyeti içinde) ve tahmin edilen değerlerin ortalaması ({sayi(s['ort_tahmin'], 6)}) "
                       "sonucun ortalamasına eşittir.")

    return interactive_step(
        number=3,
        title="Tahmin edilen değer ve artık",
        note=NoteRef("5.4", 0, ("Denklem 5.6",)),
        explanation=(
            "Bir gözlem için tahmin edilen değer $\\widehat Y = \\hat\\beta_0 + \\hat\\beta_1 X_1 + \\cdots + "
            "\\hat\\beta_k X_k$, artık $\\hat u = Y - \\widehat Y$'dir. Çoklu EKK, kareli artıklar toplamını en küçük "
            "yapan katsayıları birlikte seçer (Denklem 5.6). "
            + str(case.extra.get("profile3_text", "Varsayılan özellikler değişkenlerin medyanlarıdır."))
            + " Değerleri değiştirin. Bu adım Adım 1'deki modeli kullanır."
        ),
        uses=(choice,),
        controls=controls,
        build=build,
        checks=(_scalar("birey_tahmin", "Tahmin edilen değer", 3), _scalar("birey_artik", "Artık", 3)),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 4: kontrollerin doğrusal katkısını ayırmak --------------------------------------------------------

def _step4(case: Case) -> LabStep:
    y = case.roles[SONUC]
    default = _default(case)
    title = "“Sabit tutma”: kontrollerin doğrusal katkısını ayırmak"
    note_ref = NoteRef("5.5", 0, ("Şekil 5.2",))
    lead = ("Üç adım: (1) sonucu kontrollere göre tahmin edip artığını al; (2) incelenen değişkeni aynı kontrollere göre "
            "tahmin edip artığını al; (3) birinci artıkları ikinci artıklara göre regresyona tabi tut. Üçüncü adımdaki "
            "eğim çoklu regresyondaki katsayıyla aynıdır (§5.5).")
    if len(default) < 2:
        return LabStep(number=4, title=title, note=note_ref,
                       explanation=f"{lead} Bu adım için en az iki açıklayıcı değişken gerekir: ek bir sayısal değişken "
                                   "seçin.")
    x = case.roles[ACIKLAYICI]

    def build(choices) -> tuple:
        focus = choices["adim4_x"]
        others = tuple(name for name in default if name != focus)
        controls_text = " + ".join(others)
        digits = digits_for(coefficients(case, y, default)[focus], 6)
        return (
            *reload(case),
            OLS("yardimci_y", case.frame, y, others, f"1. adım: sonucu kontrollere göre tahmin et ({y} ~ {controls_text})"),
            Residuals(case.frame, "y_artik", "yardimci_y", "Sonucun kontrollerle açıklanamayan kısmı (artık)"),
            OLS("yardimci_x", case.frame, focus, others,
                f"2. adım: {_name(case, focus)} değişkenini kontrollere göre tahmin et ({focus} ~ {controls_text})"),
            Residuals(case.frame, "x_artik", "yardimci_x",
                      f"{capital(_name(case, focus))} değişkeninin kontrollerle açıklanamayan kısmı"),
            OLS("kismi", case.frame, "y_artik", ("x_artik",), "3. adım: artıkların artıklara göre regresyonu"),
            ModelValue("kismi_egim", "kismi", "coef", "Artıklar regresyonunun eğimi", term="x_artik", decimals=digits),
            OLS("tam", case.frame, y, default, f"Çoklu regresyon: {y} ~ {' + '.join(default)}"),
            ModelValue("tam_katsayi", "tam", "coef", f"Çoklu regresyonda {_name(case, focus)} katsayısı", term=focus,
                       decimals=digits),
            Scalar("kismi_fark", E.sub(E.ref("kismi_egim"), E.ref("tam_katsayi")), "Aradaki sayısal fark", decimals=12),
            ScatterPlot(case.frame, "x_artik", "y_artik", f"Kontrollerden arındırılmış {_name(case, focus)}",
                        f"Arındırılmış {_name(case, y)}", f"{capital(_name(case, focus))} değişkeninin kısmi ilişkisi: "
                        "artıklar üzerinden", fit_line=True, size=6, opacity=0.4),
        )

    def note(state, choices) -> str:
        focus = choices["adim4_x"]
        s = state.scalars
        others = listing([phrase(case, name) for name in default if name != focus])
        digits = digits_for(s["tam_katsayi"], 6)
        gap = s["kismi_fark"]
        tiny = abs(gap) <= 1e-9 * max(abs(s["tam_katsayi"]), abs(s["kismi_egim"]), np.finfo(float).tiny)
        gap_text = "fark hesap hassasiyetinde sıfırdır" if tiny else f"fark {sayi(gap, digits_for(gap, 12))}"
        return (f"Artıklar regresyonunun eğimi {sayi(s['kismi_egim'], digits)}, çoklu regresyondaki {phrase(case, focus)} "
                f"katsayısı {sayi(s['tam_katsayi'], digits)}: aynı sayıdır ({gap_text}). Yatay eksen, "
                f"{phrase(case, focus)} değişkeninin {others} ile açıklanamayan kısmıdır; dikey eksen sonucun aynı "
                "kontrollerle açıklanamayan kısmıdır. “Sabit tutma” birebir eşleştirme değildir; kontrollerin doğrusal "
                "katkısı ayrılır. İleri ekonometride bu sonuç Frisch–Waugh–Lovell teoremi olarak bilinir. Grafik "
                "nedensellik kanıtı değildir; modelde olmayan faktörler iki artıkta da kalabilir (§5.5).")

    return interactive_step(
        number=4,
        title=title,
        note=note_ref,
        explanation=(f"{lead} Model varsayılan çoklu modeldir (`{' + '.join(default)}`); kısmi ilişkisi incelenen "
                     "değişkeni seçin, diğerleri kontroldür."),
        controls=(Choice("adim4_x", "Kısmi ilişkisi incelenen değişken (diğerleri kontroldür)", options(case, default),
                         x if x in default else default[0], help=f"Varsayılan: {case.name(x)}."),),
        build=build,
        checks=(_scalar("kismi_egim", "Artıklar regresyonunun eğimi", 6),
                _scalar("tam_katsayi", "Çoklu regresyondaki katsayı", 6)),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 5: yazılım çıktısı ve basit–çoklu model tablosu ---------------------------------------------------

def _step5(case: Case, choice: MultiChoice) -> LabStep:
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    default = _default(case)
    logs = positive(case, y)
    ly = log_column(case, y)
    exact = case.own and exact_multi(case, case.roles[SONUC], default)

    def build(choices) -> tuple:
        chosen = choices[X_KEY]
        same = tuple(chosen) == default
        formula = " + ".join(chosen)
        operations: list = [
            *reload(case),
            ShowModel("coklu", "Çoklu modelin temel çıktısı (Kod 5.2'deki gibi)" if same else
                      "Seçtiğiniz modelin temel çıktısı", columns=COEF_QUANTITIES, stats=("nobs",), stars=False),
            ModelValue("r2_kod", "coklu", "r2", "R²", decimals=3),
            ModelValue("r2d_kod", "coklu", "adj_r2", "Düzeltilmiş R²", decimals=3),
            Scalar("r2_yuzde", E.mul(E.ref("r2_kod"), 100), "R², yüzde olarak", decimals=1, percent=True),
        ]
        columns = [(f"(1) {_title(case, y)}", "basit"), (f"(2) {_title(case, y)}", "coklu")]
        if logs:
            operations += [
                *log_operations(case, case.frame, (y,)),
                OLS("log_coklu", case.frame, ly, tuple(chosen), f"Log sonuç modeli: {ly} ~ {formula}"),
            ]
            if x in chosen:
                operations += [
                    ModelValue("log_x", "log_coklu", "coef", f"Log modelde {_name(case, x)} katsayısı", term=x),
                    Scalar("log_yuzde", E.mul(E.ref("log_x"), 100), "Yaklaşık yüzde fark 100 · β̂", decimals=1,
                           percent=True),
                ]
            columns.append((f"(3) {log_label(case, y)}", "log_coklu"))
        used = {x, *chosen}
        if not same:
            columns.append((f"({len(columns) + 1}) {_title(case, y)}, varsayılan model", "coklu_varsayilan"))
            used |= set(default)
        operations.append(RegressionTable(tuple(columns), (*_ordered(case, used), INTERCEPT), "tablo51",
                                          "Basit ve çoklu modeller (Tablo 5.1'deki gibi)" if same else
                                          "Tablo 5.1'in seçtiğiniz modelle kurulmuş hâli", stars=False, decimals=4,
                                          standard_errors=False, adj_r2=True))
        return tuple(operations)

    def note(state, choices) -> str:
        chosen = choices[X_KEY]
        s = state.scalars
        text = (f"Çıktı okuma sırası: bağımlı değişken `{y}`, gözlem sayısı {sayim(len(case.data))}, katsayılar `coef` "
                f"sütununda. R² = {sayi(s['r2_kod'], 3)}: sonucun örneklem ortalaması çevresindeki değişkenliğinin "
                f"yaklaşık %{sayi(s['r2_yuzde'], 1)} kadarı model tarafından örneklem içinde izlenir; düzeltilmiş R² = "
                f"{sayi(s['r2d_kod'], 3)} (Adım 8). Standart hata, t, p-değeri ve güven aralığı sütunları görünür ama "
                "henüz yorumlanmaz (Konu 7).")
        if logs and x in chosen:
            held = "diğer değişkenler aynıyken " if len(chosen) > 1 else ""
            percent = s["log_yuzde"]
            zero = negligible(case, s["log_x"], ly, x)
            text += (f" Sütun (3)'te bağımlı değişken log sonuçtur: {held}{step_words(case, x)} {plural(case)} tahmin "
                     f"edilen {outcome_words(case)} "
                     f"{change(percent, '%' + sayi(abs(percent), digits_for(percent, 1)), zero)}."
                     + rough(s["log_x"]))
        if logs:
            text += (" Sütun (2) ile (3)'ün R²'leri karşılaştırılmaz: bağımlı değişkenlerden biri sonuç, diğeri log "
                     "sonuçtur.")
        else:
            text += (f" {capital(phrase(case, y))} değişkeninde sıfır ya da negatif değer olduğu için log sonuç sütunu "
                     "kurulmaz (logaritma yalnız pozitif değerlerde tanımlıdır).")
        return text + (EXACT_MULTI_NOTE if case.own and exact_multi(case, case.roles[SONUC], tuple(chosen)) else "")

    coefficient_checks = [
        _check(f"{case.name(term) if term != INTERCEPT else 'Sabit terim'}: {label}", CoefTarget("coklu", term, quantity),
               4 if quantity == "coef" else 3)
        for term in (INTERCEPT, *default)
        for quantity, label in (("coef", "katsayı"), ("se", "standart hata"), ("t", "t"), ("p", "p-değeri"),
                                ("ci_low", "%95 GA alt"), ("ci_high", "%95 GA üst"))
    ]
    heading1, heading2 = f"(1) {_title(case, y)}", f"(2) {_title(case, y)}"
    table = [
        *(_check(f"Tablo (1): {case.name(x)}", TableTarget("tablo51", x, heading1), 4),),
        _check("Tablo (1): sabit", TableTarget("tablo51", INTERCEPT, heading1), 4),
        *(_check(f"Tablo (2): {case.name(name)}", TableTarget("tablo51", name, heading2), 4) for name in default),
        _check("Tablo (2): sabit", TableTarget("tablo51", INTERCEPT, heading2), 4),
        *(_check(f"Tablo ({i}): {label}", TableTarget("tablo51", row, heading), digits)
          for i, heading in ((1, heading1), (2, heading2))
          for row, label, digits in (("n", "gözlem sayısı", 0), ("r2", "R²", 4), ("adj_r2", "düzeltilmiş R²", 4))),
    ]
    if logs:
        heading3 = f"(3) {log_label(case, y)}"
        table += [
            *(_check(f"Tablo (3): {case.name(name)}", TableTarget("tablo51", name, heading3), 4) for name in default),
            _check("Tablo (3): R²", TableTarget("tablo51", "r2", heading3), 4),
        ]
        if x in default:
            table.append(_scalar("log_yuzde", "Log modelde yaklaşık yüzde fark", 1))
    return interactive_step(
        number=5,
        title="Python çıktısı ve basit–çoklu model tablosu",
        note=NoteRef("5.6", 0, ("Kod 5.1", "Kod 5.2", "Tablo 5.1")),
        explanation=(
            "Kod 5.1'deki gibi formüldeki `+` işaretleri açıklayıcı değişkenleri birlikte kullanır. Bu aşamada çıktıda "
            "okunanlar: bağımlı değişken, gözlem sayısı, katsayılar, R² ve düzeltilmiş R². Tablo basit modeli, çoklu "
            "modeli ve (sonucun bütün değerleri pozitifse) bağımlı değişkeni log sonuç olan çoklu modeli yan yana "
            "koyar. Bu adım Adım 1'deki modeli kullanır."
        ),
        uses=(choice,),
        build=build,
        checks=stable_checks((
            *coefficient_checks,
            _check("Gözlem sayısı", ModelTarget("coklu", "nobs"), 0),
            _scalar("r2_kod", "R²", 3), _scalar("r2d_kod", "Düzeltilmiş R²", 3), _scalar("r2_yuzde", "R², yüzde", 1),
            *table,
        ), exact),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 6: katsayı değişimi -------------------------------------------------------------------------------

def _step6(case: Case, choice: MultiChoice) -> LabStep:
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    default = _default(case)

    def build(choices) -> tuple:
        chosen = choices[X_KEY]
        operations: list = []
        simple = {}
        for name in chosen:
            if name == x:
                simple[name] = "basit_x"
                continue
            operations += [
                OLS(f"basit_{name}", case.frame, y, (name,), f"Basit regresyon: {y} ~ {name}"),
                ModelValue(f"basit_b_{name}", f"basit_{name}", "coef", f"Basit model: {_name(case, name)} katsayısı",
                           term=name),
            ]
            simple[name] = f"basit_b_{name}"
        operations += [
            ScalarTable(tuple((_title(case, name), E.ref(simple[name])) for name in chosen), "katsayi_basit", decimals=4),
            ScalarTable(tuple((_title(case, name), E.ref(f"b_{name}")) for name in chosen), "katsayi_coklu", decimals=4),
            JoinColumns("katsayilar", (("Basit model", "katsayi_basit", "deger"), ("Çoklu model", "katsayi_coklu", "deger")),
                        decimals=4, heading="Açıklayıcı değişken"),
            GroupedBarChart("katsayilar", "Açıklayıcı değişken", "Katsayı",
                            "Basit ve çoklu modelde katsayılar (Şekil 5.3'teki gibi)" if tuple(chosen) == default
                            else "Seçtiğiniz modelle: basit ve çoklu modelde katsayılar", series="sutun", decimals=3),
        ]
        return tuple(operations)

    def note(state, choices) -> str:
        chosen = choices[X_KEY]
        table = state.tables["katsayilar"]
        if len(chosen) == 1:
            return (f"Modelde tek açıklayıcı değişken var ({phrase(case, chosen[0])}): çoklu model basit modelle aynıdır "
                    "ve katsayı değişmez. Katsayıların neden değiştiğini görmek için Adım 1'de en az iki değişken seçin.")
        parts = [f"{phrase(case, name)} {sayi(table.loc[_title(case, name), 'Basit model'], 4)} → "
                 f"{sayi(table.loc[_title(case, name), 'Çoklu model'], 4)}" for name in chosen]
        return (f"Basit modelden çoklu modele katsayılar: {'; '.join(parts)}. Katsayının değişmesi yazılım hatası değildir; "
                "basit model bir değişkenin farklı olduğu bütün gözlemleri diğer farkları ayırmadan karşılaştırır, çoklu "
                "model diğer değişkenleri aynı tutar. Değişim tek başına hangi modelin doğru olduğunu söylemez: eklenen "
                "değişken önemli bir faktör olabilir, araştırma sorusu için gereksiz olabilir ya da temel değişkenin bir "
                "sonucu olabilir. Dışarıda kalan değişkenin katsayıyı hangi koşullarda ve hangi yönde etkilediği Konu "
                "6'da işlenir.")

    return interactive_step(
        number=6,
        title="Bir değişken eklendiğinde katsayı neden değişir?",
        note=NoteRef("5.7", 0, ("Şekil 5.3",)),
        explanation=(
            "Aynı değişkenin katsayısı basit ve çoklu modelde farklı olabilir; bu yazılım hatası değildir. Basit katsayı, "
            "değişkenle birlikte hareket eden diğer farkları da taşır; çoklu katsayı, modeldeki diğer değişkenlerin "
            "doğrusal katkısı ayrıldıktan sonraki ilişkidir. Grafik, modeldeki her değişken için iki katsayıyı yan yana "
            "gösterir. Bu adım Adım 1'deki modeli kullanır."
        ),
        uses=(choice,),
        build=build,
        checks=(
            _check(f"Basit model: {case.name(x)}", TableTarget("katsayilar", _title(case, x), "Basit model"), 3),
            _check(f"Çoklu model: {case.name(x)}", TableTarget("katsayilar", _title(case, x), "Çoklu model"), 3),
        ) if x in default else (),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 7: ikinci veriyle çoklu model (konut fiyatı) ------------------------------------------------------

def _houses(house: Case) -> tuple[tuple[str, ...], tuple[tuple, ...]]:
    """Tahmin edilecek üç gözlem: (sütunlar, satırlar). A ile B yalnız ikinci değişkende, B ile C birden fazla
    değişkende farklıdır."""

    if "houses" in house.extra:
        return house.extra["houses"]
    names = tuple(candidates(house))
    base = {name: _median(house, name) for name in names}
    steps = {}
    for name in names:
        values = house.data[name]
        step = nice(float(values.std()), 1) or 1.0
        steps[name] = max(1.0, float(round(step))) if np.all(values == np.round(values)) else step
    second = names[1] if len(names) > 1 else names[0]
    b = dict(base, **{second: base[second] + steps[second]})
    c = dict(b, **{names[0]: b[names[0]] + steps[names[0]]})
    if len(names) > 2:
        c[names[2]] = c[names[2]] + steps[names[2]]
    return names, tuple((label, *(values[name] for name in names)) for label, values in (("A", base), ("B", b), ("C", c)))


def _step7(house: Case) -> LabStep:
    y, x = house.roles[SONUC], house.roles[ACIKLAYICI]
    regressors = tuple(candidates(house))
    default = tuple(house.extra.get("default_regressors") or regressors)
    effects = {name: spec for name, spec in dict(house.extra.get("effects", {})).items()}
    columns, rows = _houses(house)
    title = str(house.extra.get("house_title", "İkinci model: aynı veride birden fazla özellik"))

    def build(choices) -> tuple:
        chosen = choices["adim7_x"]
        same = tuple(chosen) == default
        operations: list = [
            *house.load,
            OLS("fiyat_basit", house.frame, y, (x,), f"Basit model: {y} ~ {x}"),
            OLS("fiyat_coklu", house.frame, y, tuple(chosen), f"Çoklu model: {y} ~ {' + '.join(chosen)}"),
            ModelValue("f_sabit", "fiyat_coklu", "coef", "Sabit terim", term=INTERCEPT),
            *(ModelValue(f"f_{name}", "fiyat_coklu", "coef", f"{capital(_name(house, name))} katsayısı", term=name,
                         decimals=6 if name in house.extra.get("fine", ()) else 4) for name in chosen),
        ]
        for name in chosen:
            if name in effects:
                scalar, label, factor, decimals = effects[name]
                operations.append(Scalar(scalar, E.mul(E.ref(f"f_{name}"), factor), label, decimals=decimals))
        table_columns = [(f"(1) {_title(house, y)}", "fiyat_basit"), (f"(2) {_title(house, y)}", "fiyat_coklu")]
        used = {x, *chosen}
        if not same:
            operations.append(OLS("fiyat_varsayilan", house.frame, y, default,
                                  f"Karşılaştırma için varsayılan model: {y} ~ {' + '.join(default)}"))
            table_columns.append((f"(3) {_title(house, y)}, varsayılan model", "fiyat_varsayilan"))
            used |= set(default)
        terms = _ordered(house, used)
        operations += [
            RegressionTable(tuple(table_columns), (*terms, INTERCEPT), "tablo52",
                            "Basit ve çoklu model (Tablo 5.2'deki gibi)" if same else
                            "Tablo 5.2'nin seçtiğiniz modelle kurulmuş hâli", stars=False, decimals=4,
                            standard_errors=False, adj_r2=True,
                            term_decimals=tuple((name, 6) for name in terms if name in house.extra.get("fine", ()))),
            InlineData("konutlar", ("konut", *columns), rows, "Tahmin edilecek üç gözlem (Tablo 5.3'teki gibi)"),
            Derive("konutlar", "tahmin", _predict(chosen, {name: E.var(name) for name in chosen}, "f_"),
                   "Tahmin edilen değer"),
            *(Statistic("konutlar", "tahmin", "value", f"konut_{label}", f"{label} gözleminin tahmini",
                        where=("konut", label), decimals=2) for label, *_ in rows),
            Scalar("konut_fark", E.sub(E.ref("konut_B"), E.ref("konut_A")), "Tahmin farkı B − A", decimals=2),
            Derive(house.frame, "fiyat_tahmin", _predict(chosen, {name: E.var(name) for name in chosen}, "f_"),
                   "Tahmin edilen değer Ŷᵢ"),
            ScatterPlot(house.frame, y, "fiyat_tahmin", f"Gözlenen {_name(house, y)}", f"Tahmin edilen {_name(house, y)}",
                        f"{house.extra.get('data_name', 'Verileriniz')}: gözlenen ve tahmin edilen değer (Şekil 5.4'teki "
                        "gibi)" if same else "Seçtiğiniz modelle: gözlenen ve tahmin edilen değer", size=8, opacity=0.7,
                        lines=((0, 1, "45 derece çizgisi"),)),
        ]
        return tuple(operations)

    unit = f" {short_unit(house, y)}" if short_unit(house, y) else ""
    second = columns[1] if len(columns) > 1 else columns[0]

    def note(state, choices) -> str:
        chosen = list(choices["adim7_x"])
        s = state.scalars
        parts = []
        for name in chosen:
            others = [other for other in chosen if other != name]
            if name in effects:
                scalar, label, factor, decimals = effects[name]
                words = house.extra["effect_words"][name]
                start = (f"{capital(listing([phrase(house, other) for other in others]))} aynıyken, {words}" if others
                         else capital(words))
                zero = negligible(house, s[f"f_{name}"], y, name)
                parts.append(f"{start} tahmin edilen {outcome_words(house)} "
                             f"{change(s[scalar], sayi(abs(s[scalar]), decimals) + unit, zero)}.")
            else:
                parts.append(_interpretation(house, name, s[f"f_{name}"], others))
        text = " ".join(parts)
        a_b = [name for name in columns if rows[0][1 + columns.index(name)] != rows[1][1 + columns.index(name)]]
        changed = a_b[0] if a_b else second
        if changed in chosen:
            gap = rows[1][1 + columns.index(changed)] - rows[0][1 + columns.index(changed)]
            times = f"{phrase(house, changed)} katsayısıdır" if gap == 1 else \
                f"{level_text(gap)} × {phrase(house, changed)} katsayısıdır"
            text += (f" A ile B yalnız {phrase(house, changed)} değişkeninde farklıdır: tahmin farkı "
                     f"{sayi(s['konut_fark'], 2)}{unit}, yani {times}.")
        else:
            text += (f" A ile B yalnız {phrase(house, changed)} değişkeninde farklıdır; bu değişken modelde olmadığı için "
                     "iki tahmin aynıdır.")
        differing = [name for name in columns if rows[1][1 + columns.index(name)] != rows[2][1 + columns.index(name)]]
        in_model = [name for name in differing if name in chosen]
        if len(in_model) > 1:
            text += " B ile C arasında birden fazla özellik değiştiği için fark birden fazla katsayının katkısıdır."
        elif in_model:
            text += (f" B ile C arasındaki tahmin farkı yalnız {phrase(house, in_model[0])} katkısıdır; farklı olan öteki "
                     "özellikler modelde olmadığı için tahmine girmez.")
        else:
            text += " B ile C'nin farklı olduğu özellikler modelde olmadığı için tahminleri aynıdır."
        return text + (" Şekilde noktalar 45 derece çizgisine yaklaştıkça tahmin gözlenen değere yaklaşır; çizgiden "
                       "uzaklık artıktır. Modelde bulunmayan özellikler artıkta kalır.")

    checks = [
        _check("Basit model: sabit terim", CoefTarget("fiyat_basit", INTERCEPT), 4),
        _check("Basit model: eğim", CoefTarget("fiyat_basit", x), 4),
        _check("Çoklu model: sabit terim", CoefTarget("fiyat_coklu", INTERCEPT), 4),
        *(_check(f"Çoklu model: {house.name(name)}", CoefTarget("fiyat_coklu", name),
                 6 if name in house.extra.get("fine", ()) else 4) for name in default),
    ]
    for heading in (f"(1) {_title(house, y)}", f"(2) {_title(house, y)}"):
        checks += [_check(f"Tablo {heading}: {label}", TableTarget("tablo52", row, heading), digits)
                   for row, label, digits in (("n", "gözlem sayısı", 0), ("r2", "R²", 4), ("adj_r2", "düzeltilmiş R²", 4))]
    checks += [_scalar(effects[name][0], effects[name][1], effects[name][3]) for name in default if name in effects]
    checks += [_check(f"{label} gözleminin tahmini", CellTarget("konutlar", "tahmin", row), 2)
               for row, (label, *_) in enumerate(rows, start=1)]
    checks.append(_scalar("konut_fark", "Tahmin farkı B − A", 2))
    return interactive_step(
        number=7,
        title=title,
        note=NoteRef("5.8", 0, ("Denklem 5.7", "Tablo 5.2", "Tablo 5.3", "Şekil 5.4")),
        explanation=str(house.extra.get("house_text") or (
            f"Aynı veride sonuç ({phrase(house, y)}) birden fazla açıklayıcı değişkenle tahmin edilir. Tablo 5.3'teki gibi "
            "üç gözlem için tahmin üretilir: A değişkenlerin medyanlarıdır, B yalnız bir değişkende A'dan farklıdır, C "
            "birden fazla değişkende B'den farklıdır. Şekil gözlenen ve tahmin edilen değerleri karşılaştırır. Modelin "
            "açıklayıcı değişkenlerini seçin.")),
        controls=(MultiChoice("adim7_x", f"Modelin açıklayıcı değişkenleri (bağımlı değişken: {house.name(y)})",
                              options(house, regressors), default,
                              help="Varsayılan: " + ", ".join(house.name(name) for name in default) + "."),),
        build=build,
        checks=tuple(checks),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 8: R² ve düzeltilmiş R² ---------------------------------------------------------------------------

def _adjusted(r2, n: int, k: int) -> E.Expr:
    """Denklem 5.8: 1 − (1 − R²)(n − 1)/(n − k − 1)."""

    return E.sub(1, E.mul(E.sub(1, r2), E.div(n - 1, n - k - 1)))


def _n_control(case: Case) -> NumberChoice:
    if "adim8" in case.extra:
        return case.extra["adim8"]
    n = len(case.data)
    k = len(candidates(case))
    low = max(k + 2, 10)
    return number_control("adim8_n", "Gözlem sayısı n (A, B ve C modelleri)", n, low, max(2000, n),
                          help=f"Varsayılan: verinizin gözlem sayısı ({n}).", step=1.0)


def _step8(case: Case, choice: MultiChoice) -> LabStep:
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    default = _default(case)
    every = tuple(candidates(case))
    control = _n_control(case)

    def build(choices) -> tuple:
        chosen = choices[X_KEY]
        same = tuple(chosen) == default
        n = int(choices["adim8_n"])
        k = len(chosen)
        hkt_per_df = E.div(E.ref("hkt_coklu"), E.sub(E.ref("n_coklu"), k + 1))
        tkt_per_df = E.div(E.ref("tkt"), E.sub(E.ref("n_coklu"), 1))
        label = "Çoklu model" if same else "Seçtiğiniz model"
        extra = () if same else (
            ModelValue("r2_varsayilan", "coklu_varsayilan", "r2", "Varsayılan model: R²"),
            ModelValue("r2d_varsayilan", "coklu_varsayilan", "adj_r2", "Varsayılan model: düzeltilmiş R²"),
        )
        default_r2 = () if same else (("Varsayılan model", E.ref("r2_varsayilan")),)
        default_r2d = () if same else (("Varsayılan model", E.ref("r2d_varsayilan")),)
        models = (("Model A", "basit", 1, "r2_basit"), ("Model B", "coklu", k, "r2_coklu"),
                  ("Model C", "genis", len(every), "r2_genis"))
        return (
            ModelValue("r2_basit", "basit", "r2", "Basit model: R²"),
            ModelValue("r2d_basit", "basit", "adj_r2", "Basit model: düzeltilmiş R²"),
            ModelValue("r2_coklu", "coklu", "r2", f"{label}: R²"),
            ModelValue("r2d_coklu", "coklu", "adj_r2", f"{label}: düzeltilmiş R²"),
            *extra,
            ModelValue("hkt_coklu", "coklu", "ssr", f"{label}: HKT = Σûᵢ²", decimals=3),
            ModelValue("n_coklu", "coklu", "nobs", "Gözlem sayısı n", decimals=0),
            Statistic(case.frame, y, "mean", "ybar", "Sonucun ortalaması Ȳ", decimals=6),
            Derive(case.frame, "sapma_kare", E.power(E.sub(E.var(y), E.ref("ybar")), 2), "(Yᵢ − Ȳ)²"),
            Statistic(case.frame, "sapma_kare", "sum", "tkt", "TKT = Σ(Yᵢ − Ȳ)²", decimals=3),
            Scalar("r2d_elle", E.sub(1, E.div(hkt_per_df, tkt_per_df)),
                   f"Denklem 5.8 ile: 1 − [HKT/(n − {k} − 1)] / [TKT/(n − 1)]", decimals=4),
            ScalarTable((("Basit model", E.ref("r2_basit")), (label, E.ref("r2_coklu")), *default_r2), "uyum_r2",
                        decimals=4),
            ScalarTable((("Basit model", E.ref("r2d_basit")), (label, E.ref("r2d_coklu")), *default_r2d), "uyum_r2d",
                        decimals=4),
            JoinColumns("uyum", (("R²", "uyum_r2", "deger"), ("Düzeltilmiş R²", "uyum_r2d", "deger")), decimals=4,
                        heading="Model"),
            GroupedBarChart("uyum", "Model", "Uyum ölçüsü",
                            f"{case.extra.get('data_name', 'Verileriniz')}: basit ve çoklu modelde R² ve düzeltilmiş R²"
                            if same else "Basit model, seçtiğiniz model ve varsayılan model", series="sutun",
                            decimals=4),
            OLS("genis", case.frame, y, every, f"Model C: bütün seçenekler ({y} ~ {' + '.join(every)})"),
            ModelValue("r2_genis", "genis", "r2", "Model C: R²"),
            ScalarTable(tuple((f"{name} (k = {size})", _adjusted(E.ref(r2), n, size)) for name, _, size, r2 in models),
                        "abc", decimals=3, heading=f"Model (n = {n})", value="Düzeltilmiş R²"),
        )

    def note(state, choices) -> str:
        chosen = choices[X_KEY]
        n = int(choices["adim8_n"])
        s = state.scalars
        table = state.tables["abc"]["deger"]
        best = table.idxmax()
        label = "çoklu model" if tuple(chosen) == default else "seçtiğiniz model"
        text = (f"Basit model: R² = {sayi(s['r2_basit'], 4)}, düzeltilmiş R² = {sayi(s['r2d_basit'], 4)}; {label}: "
                f"{sayi(s['r2_coklu'], 4)} ve {sayi(s['r2d_coklu'], 4)}. Denklem 5.8 ile elde hesaplanan değer "
                f"{sayi(s['r2d_elle'], 4)} yazılımınkiyle aynıdır.")
        if tuple(chosen) != default:
            text += f" Varsayılan model: {sayi(s['r2_varsayilan'], 4)} ve {sayi(s['r2d_varsayilan'], 4)}."
        if tuple(chosen) == (x,):
            text += " Seçtiğiniz model basit modelin kendisidir; iki model aynı sayıları verir."
        text += (" Yeni açıklayıcı değişken R²'yi azaltmaz (EKK yeni katsayıyı sıfır seçerek eski modeli koruyabilir); "
                 "düzeltilmiş R² her yeni katsayının serbestlik derecesi maliyetini hesaba katar ve düşebilir.")
        if x not in chosen:
            text += (f" Seçtiğiniz modelde {phrase(case, x)} yok: iki model iç içe değildir (biri diğerine değişken "
                     "eklenerek elde edilmez), bu yüzden kural aralarında geçerli olmaz.")
        return text + (f" Son tablo bu üç modelin R²'sini seçilen gözlem sayısıyla (n = {n}) düzeltir; A basit model, B "
                       f"seçtiğiniz model, C bütün seçenekleri içeren model. Düzeltilmiş R² en yüksek olan: {best}. n "
                       "küçüldükçe ek değişkenlerin cezası büyür. İki ölçü de yalnız aynı bağımlı değişken ve aynı "
                       "örneklemde karşılaştırılır; yüksek değer doğru ekonomik model veya nedensellik garantisi değildir.")

    sizes = (1, len(default), len(every))
    return interactive_step(
        number=8,
        title="Çoklu R² ve düzeltilmiş R²",
        note=NoteRef("5.9", 0, ("Denklem 5.8",)),
        explanation=(
            "$R^2 = 1 - \\text{HKT}/\\text{TKT}$ çoklu modelde de aynıdır. Yeni değişken eklemek R²'yi düşürmez; "
            "düzeltilmiş R² açıklayıcı değişken sayısını hesaba katar: $\\bar R^2 = 1 - "
            "\\frac{\\text{HKT}/(n-k-1)}{\\text{TKT}/(n-1)}$ (Denklem 5.8). Son tablo notlardaki alıştırmanın karşılığıdır: "
            "bu uygulamanın üç modelinin (basit model, seçtiğiniz çoklu model, bütün seçenekleri içeren model) R²'si "
            "seçtiğiniz gözlem sayısıyla düzeltilir. Bu adım Adım 1'deki modeli kullanır."
        ),
        uses=(choice,),
        controls=(control,),
        build=build,
        checks=(
            _scalar("r2_basit", "Basit model: R²", 4), _scalar("r2d_basit", "Basit model: düzeltilmiş R²", 4),
            _scalar("r2_coklu", "Çoklu model: R²", 4), _scalar("r2d_coklu", "Çoklu model: düzeltilmiş R²", 4),
            _scalar("r2d_elle", "Denklem 5.8 ile elde hesap", 4),
            *(_check(f"Alıştırma: {name} (k = {size})", TableTarget("abc", f"{name} (k = {size})", "deger"), 3)
              for name, size in zip(("Model A", "Model B", "Model C"), sizes)),
        ),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 9: makale tablosu ----------------------------------------------------------------------------------

def _article(case: Case) -> dict[str, tuple[str, str, tuple[str, ...], str]]:
    """Makale tablosunun sütunları: anahtar → (başlık, bağımlı değişken, açıklayıcılar, seçenek adı)."""

    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    default = _default(case)
    ly = log_column(case, y)
    level, logged = _title(case, y), log_label(case, y)
    found = {"basit": (level, y, (x,), f"{_title(case, y)} ~ {_name(case, x)}")}
    if len(default) > 1:
        found["coklu"] = (level, y, default, f"{_title(case, y)} ~ varsayılan çoklu model")
        if positive(case, y):
            found["log_coklu"] = (logged, ly, default, f"{logged} ~ varsayılan çoklu model")
    if len(default) > 2:
        found["deneyimli"] = (level, y, default[:2], f"{_title(case, y)} ~ {_name(case, default[0])} + "
                                                      f"{_name(case, default[1])}")
    if positive(case, y):
        found["log_basit"] = (logged, ly, (x,), f"{logged} ~ {_name(case, x)}")
    return found


def _step9(case: Case) -> LabStep:
    article = _article(case)
    default = tuple(key for key in ("basit", "coklu", "log_coklu") if key in article)
    logs = positive(case, case.roles[SONUC])

    def build(choices) -> tuple:
        selected = choices["adim9_sutunlar"]
        columns = []
        operations: list = list(log_operations(case, case.frame, (case.roles[SONUC],))) if logs and any(
            article[key][1] != case.roles[SONUC] for key in selected) else []
        for number, key in enumerate(selected, start=1):
            heading, outcome, regressors, _ = article[key]
            operations.append(OLS(f"makale_{key}", case.frame, outcome, regressors,
                                  f"Sütun ({number}): {outcome} ~ {' + '.join(regressors)}"))
            columns.append((f"({number}) {heading}", f"makale_{key}"))
        used = {name for key in selected for name in article[key][2]}
        operations.append(RegressionTable(tuple(columns), (*_ordered(case, used), INTERCEPT), "makale",
                                          "Makale tablosu (Tablo 5.4'teki gibi)" if tuple(selected) == default
                                          else "Seçtiğiniz modellerle makale tipi tablo", stars=False, decimals=3,
                                          standard_errors=False, adj_r2=True))
        return tuple(operations)

    def note(state, choices) -> str:
        selected = choices["adim9_sutunlar"]
        groups: dict[str, list[int]] = {}
        for number, key in enumerate(selected, start=1):
            groups.setdefault(article[key][0], []).append(number)
        text = ("Okuma sırası: her sütunun bağımlı değişkeni, gözlem sayısı, sütundaki açıklayıcı değişkenler, katsayının "
                "birimi (düzey sonuçta sonucun birimi, log sonuçta yaklaşık yüzde) ve ceteris paribus koşulu; sonra R² "
                "ve düzeltilmiş R².")
        if len(groups) > 1:
            parts = ", ".join(f"{listing([f'({n})' for n in numbers])} {'log sonucun' if name.startswith('ln(') else 'sonucun'}"
                              for name, numbers in groups.items())
            text += (f" R²'ler yalnız aynı bağımlı değişkenli sütunlar arasında karşılaştırılır: {parts} değişkenliğini "
                     "özetler.")
        elif len(selected) > 1:
            text += (" Sütunların bağımlı değişkeni ve örneklemi "
                     f"({sayim(len(case.data))} gözlem) aynıdır: R²'leri doğrudan karşılaştırılabilir.")
        return text + (" Makalelerde kontroller bazen tek tek yazılmaz; “Kontroller: Evet” satırı hangi değişkenlerin "
                       "kontrol edildiğini söylemez, tablo notunda açıklanmalıdır (Tablo 5.4). Anlamlılık yıldızları ve "
                       "parantez içindeki standart hatalar Konu 7'de okunur.")

    checks = []
    for number, key in enumerate(default, start=1):
        heading, _, regressors, _ = article[key]
        column = f"({number}) {heading}"
        checks += [_check(f"Tablo ({number}): {case.name(name)}", TableTarget("makale", name, column), 3)
                   for name in regressors]
        checks += [_check(f"Tablo ({number}): {label}", TableTarget("makale", row, column), digits)
                   for row, label, digits in ((INTERCEPT, "sabit", 3), ("n", "gözlem sayısı", 0), ("r2", "R²", 3),
                                              ("adj_r2", "düzeltilmiş R²", 3))]
    return interactive_step(
        number=9,
        title="Python çıktısı ile makale tablosunu sistematik okumak",
        note=NoteRef("5.10", 0, ("Tablo 5.4",)),
        explanation=(
            "Makale tablosu yazılım çıktısını sıkıştırır: her sütun bir model, satırlar katsayılar; altta gözlem sayısı, "
            "R² ve düzeltilmiş R². Okuma sırası aynıdır: bağımlı değişken, gözlem sayısı, sütundaki değişkenler, "
            "katsayının birimi ve ceteris paribus koşulu. Sütunları siz seçin."
        ),
        controls=(MultiChoice("adim9_sutunlar", "Tablodaki modeller",
                              tuple((key, name) for key, (*_, name) in article.items()), default,
                              help="Varsayılan: basit model, çoklu model ve (varsa) log sonuçlu çoklu model.",
                              maximum=4),),
        build=build,
        checks=tuple(checks),
        note_for=lambda state, choices: note(state, choices),
    )


# --- Adım 10: kontrol değişkeni seçimi -----------------------------------------------------------------------

_CONTROLS = (
    "Bir değişkeni kontrol olarak eklemeden önce sorulacaklar (§5.11):\n\n1. Değişken bağımlı değişkenle ekonomik olarak "
    "neden ilişkili olabilir?\n2. Temel açıklayıcı değişkenle de ilişkili olabilir mi?\n3. Temel açıklayıcı değişkenden "
    "önce mi, sonra mı belirlenmektedir?\n4. Doğru ölçülmüş müdür?\n5. Onu sabit tutmak cevaplamak istediğimiz "
    "karşılaştırmaya uygun mudur?\n6. Aynı değişkenin farklı tanımları veya gereksiz tekrarları modele eklenmiş "
    "midir?\n\n{example}Tek bir evrensel değişken listesi yoktur: araştırma sorusu hangi karşılaştırmanın hedeflendiğini "
    "belirler. Çok sayıda ve birbirine çok yakın değişken katsayıların ayrı ayrı yorumunu zorlaştırır; bu, Konu 6'daki "
    "çoklu doğrusal bağlantı konusudur."
)


def _step10(case: Case) -> LabStep:
    example = str(case.extra.get("controls_example", ""))
    return LabStep(number=10, title="Kontrol değişkeni seçimi: sorulacak altı soru", note=NoteRef("5.11"),
                   explanation=_CONTROLS.format(example=f"{example} " if example else ""))


# --- Tanım ------------------------------------------------------------------------------------------------------

_LABELS = {
    INTERCEPT: "Sabit terim", "calisan": "Gözlem", "konut": "Gözlem", "tahmin": "Tahmin edilen değer Ŷᵢ",
    "artik": "Artık ûᵢ", "y_artik": "Arındırılmış sonuç", "x_artik": "Arındırılmış açıklayıcı değişken",
    "fiyat_tahmin": "Tahmin edilen değer Ŷᵢ", "sapma_kare": "(Yᵢ − Ȳ)²",
}


def build(case: Case) -> LabSpec:
    """Konu 5 uygulamasını verilen veriyle kurar; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    house = second(case)
    choice = x_multi(case)
    labels = labels_of(case)
    labels.update(labels_of(house))
    if positive(case, case.roles[SONUC]):
        labels.setdefault(log_column(case, case.roles[SONUC]), log_label(case, case.roles[SONUC]))
    line, line_levels = _line_setup(case)
    if line is not None:
        labels.update({f"tahmin_d{index}": f"{case.name(line)} = {level_text(level)}"
                       for index, level in enumerate(line_levels, start=1)})
    labels.update(_LABELS)
    spec = LabSpec(
        topic_key=TOPIC,
        title=str(case.extra.get("title", TITLE)),
        note_section="5",
        steps=(_step1(case, choice), _step2(case, choice), _step3(case, choice), _step4(case), _step5(case, choice),
               _step6(case, choice), _step7(house), _step8(case, choice), _step9(case), _step10(case)),
        labels=tuple(labels.items()),
        source=case.source,
    )
    return with_app_values(spec)


# --- Alternatif örnek ----------------------------------------------------------------------------------------

def _house() -> Case:
    return house_case(
        regressors=("area", "rooms", "baths", "land"),
        default_regressors=("area", "rooms", "baths"),
        effects={"area": ("yuz_fit", "100 fit² daha büyük konut: fiyat farkı (dolar)", 100, 2),
                 "land": ("bin_fit_arsa", "1.000 fit² daha büyük arsa: fiyat farkı (dolar)", 1000, 2)},
        effect_words={"area": "büyüklüğü 100 fit² daha fazla olan konutların",
                      "land": "arsası 1.000 fit² daha büyük olan konutların"},
        coefficient_digits=2,
        fine=("land",),
        houses=(("area", "rooms", "baths", "land"),
                (("A", 2000, 7, 2, 40000), ("B", 2000, 8, 2, 40000), ("C", 2250, 8, 3, 45000))),
        house_title="KIELMC (1978): konut fiyatında birden fazla özellik",
        house_text=("Fiyat dolar, konut ve arsa büyüklüğü fit² cinsindendir. Çoklu model konut büyüklüğü, oda sayısı ve "
                    "banyo sayısını birlikte kullanır (Denklem 5.7'nin karşılığı). Tablo 5.3'teki gibi üç konut için "
                    "tahmin üretilir: A ve B yalnız oda sayısında, B ve C büyüklük, banyo ve arsada farklıdır. Şekil "
                    "gözlenen ve tahmin edilen fiyatları karşılaştırır. Modelin açıklayıcı değişkenlerini seçin."),
    )


def alternative_case() -> Case:
    return wage2_case(
        ALT_OPTIONS,
        title="Uygulama: Çoklu Regresyon ve Ceteris Paribus Yorumu (WAGE2, KIELMC)",
        default_regressors=ALT_DEFAULT,
        house=_house(),
        line=("exper", (5.0, 12.0, 20.0)),
        grid=tuple(float(value) for value in range(9, 19)),
        profile={"educ": 12.0, "exper": 10.0, "tenure": 5.0, "IQ": 100.0},
        profile_text=("A çalışanının eğitimi 12, iş deneyimi 10, kıdemi 5 yıl, IQ puanı 100; B'nin eğitimi daha yüksek, "
                      "diğer özellikleri aynıdır."),
        gap_word="yıl",
        subjects={"educ": "eğitimi", "exper": "iş deneyimi", "tenure": "kıdemi", "IQ": "IQ puanı"},
        unit_accusative="çalışanları",
        adim2=NumberChoice("adim2_fark", "Eğitim farkı (yıl): B'nin eğitimi A'nınkinden bu kadar fazla", 1, 6, 4, 1,
                           help="Varsayılan: 4 yıl (A 12, B 16 yıl eğitimli).", integer=True, decimals=0),
        adim3=(
            NumberChoice("adim3_d1", "Eğitim (yıl)", 0, 20, 16, 1, help="Varsayılan: 16.", integer=True, decimals=0),
            NumberChoice("adim3_d2", "İş deneyimi (yıl)", 0, 30, 10, 1, help="Varsayılan: 10.", integer=True,
                         decimals=0),
            NumberChoice("adim3_d3", "Kıdem (yıl)", 0, 25, 5, 1, help="Varsayılan: 5.", integer=True, decimals=0),
            NumberChoice("adim3_d4", "IQ puanı", 50, 150, 100, 1, help="Varsayılan: 100 (modelde IQ varsa).",
                         integer=True, decimals=0),
            NumberChoice("adim3_y", "Gözlenen aylık kazanç (dolar)", 0, 3100, 1200, 10, help="Varsayılan: 1200 dolar.",
                         integer=True, decimals=0),
        ),
        profile3_text=("Varsayılan çalışanın eğitimi 16, iş deneyimi 10, kıdemi 5 yıl, IQ puanı 100; gözlenen aylık "
                       "kazancı 1200 dolardır."),
        adim8=NumberChoice("adim8_n", "Gözlem sayısı n (A, B ve C modelleri)", 20, 2000, 935, 5,
                           help="Varsayılan: WAGE2'nin gözlem sayısı (935); küçük n'de cezayı görün.", integer=True,
                           decimals=0),
        controls_example=("Örneğin eğitimin kazançla ilişkisinde iş deneyimi, kıdem ve (yeteneğin bir ölçüsü olarak) IQ "
                          "makul kontrollerdir; eğitimden sonra seçilen mesleği kontrol etmek ise sorunun bir bölümünü "
                          "ortadan kaldırabilir."),
    )


@cache
def alternative() -> LabSpec:
    return build(alternative_case())


STORY = ("Ücret adımları WAGE2'dir (935 erkek çalışan, 1980; aylık kazanç, ABD doları); seçeneklere IQ puanı eklendi. "
         "Konut adımı KIELMC'nin 1978 satışlarıdır: fiyat ~ büyüklük + oda + banyo (179 konut).")


# --- Kendi verin ---------------------------------------------------------------------------------------------

CUSTOM = custom_lab(
    build,
    sample_employed,
    ("Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sonuç ve temel açıklayıcı değişken zorunludur ve sayısal olmalıdır. "
     "Çoklu model için en az bir ek sayısal değişken seçin (en çok 3); varsayılan model bütün seçilen değişkenleri "
     "içerir. Notlarda konut verisiyle yapılan Adım 7 aynı dosyayla kurulur. Log sonuçlu sütunlar için sonucun bütün "
     f"değerleri pozitif olmalı. {ROW_RULE}"),
    roles((1, 2, 3, 4, 5, 6, 7, 8, 9)),
    "Çoklu modelin değişkenleridir (Adım 1–9).",
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
