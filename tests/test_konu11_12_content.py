"""Konu 11–12 bloğu: etkileşim terimleri ve grup farkları; heteroskedastisite ve dayanıklı çıkarım. Etkileşimli
adımlar, motorun etkileşim ve dayanıklı kovaryans işlemleri, Sezgi deneyleri ve sorular.

Genel sözleşmeler (notlardaki her sayının uygulamada, üretilen Python'da ve R'de tutması; deneylerin üretilen Python
koduyla birebir aynı sayıları vermesi) ``test_all_labs.py``'dedir; bu dosya bloğa özgü olanları denetler:

* notlardan farklı her seçimde uygulamanın sayıları ile üretilen Python kodunun sayıları 1e-12 düzeyinde aynıdır, üretilen
  R kodu çalışır, seçim kendi adımını ve onu kullanan adımları değiştirir, adım metinleri ve ekran tabloları her seçimde
  doğru kalır;
* adımlar söyledikleri hesabı yapar ve bağımsız bir hesapla aynı sayıyı verir: grup eğimleri ve koşullu farklar, farkın
  işaret değiştirdiği nokta, merkezlemenin eşdeğerliği, eğim ve iki doğru testleri; HC0–HC3'ün açık sandviç formülü,
  Breusch–Pagan (Koenker n·R²) ve White testleri (kuklanın karesi bir kez sayılır, statsmodels ``het_white``'a
  dayanılmaz), dayanıklı Wald testinin F biçimi (W/q), kaldıraç sabitleri;
* HPRICE1'de arsa sonuçlarının tek bir büyük arsaya duyarlılığı (notlar §11.7 ve §12.6);
* etkileşim terimlerinin etiketleri (adsız açıklama başlığı olduğu gibi kalır) ve tam sayı sütunlarında tipografik eksi;
* Sezgi deneyleri kuramsal ilişkileri tutturur, Konu 12 Deney 1'in varsayılanı Tablo 12.6'yı ve Şekil 12.1–12.2'nin
  verisini üretir, metinler kaydırıcıların uçlarında ve köşe bileşimlerinde doğru kalır;
* denklem sorularının kabul ve ret yazımları; sorulardaki sayılar veriden hesaplanır.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

from core import wooldridge_data as W
from core.charts import CHART_TYPES, figure_for
from core.codegen.base import generator, render_script
from core.labs.konu11 import HOUSE_RANGES, KONU11_LAB
from core.labs.konu12 import KONU12_LAB, LEVERAGE
from core.labs.regression import WHITE_RCOND, hetero_test, white_design, white_test
from core.labs.runner import LabState, batchable, run_lab, run_operations, shown_frame
from core.labs.sezgi_konu11 import CENTERING, JOINT, KONU11_EXPERIMENTS, OMITTED, gap
from core.labs.sezgi_konu12 import HETERO, KONU12_EXPERIMENTS, SMALL, TESTS, average_sd
from core.labs.spec import (LabSpec, MonteCarlo, MultiChoice, NumberChoice, RegressionTable, ShowFrame, ShowModel,
                             interaction_label)
from core.quiz.konu11 import KONU11_QUIZ
from core.quiz.konu12 import KONU12_QUIZ
from core.quiz.model import grade
from topics.lab_ui import coefficient_display, display_table, frame_display, regression_display

LABS = (KONU11_LAB, KONU12_LAB)
EXPERIMENTS = KONU11_EXPERIMENTS + KONU12_EXPERIMENTS
WAGE_FORMULA = "lwage ~ female * educ12 + exper + tenure"
HOUSE_FORMULA = "price ~ colonial * lotsize10k + sqrft100 + bdrms"
LEVEL_FORMULA = "price ~ lotsize1000 + sqrft100 + bdrms"
LOG_FORMULA = "lprice ~ llotsize + lsqrft + bdrms"


def _printed(value: float, expected: float, decimals: int) -> bool:
    """Değer, notlarda ``decimals`` basamakla basılı sayıya yuvarlanır."""

    return abs(value - expected) <= 0.5 * 10 ** -decimals + 1e-12


@pytest.fixture(scope="module")
def wage1() -> pd.DataFrame:
    frame = W.load("wage1").copy()
    frame["educ12"] = frame["educ"] - 12
    return frame


@pytest.fixture(scope="module")
def hprice1() -> pd.DataFrame:
    frame = W.load("hprice1").copy()
    frame["lotsize10k"] = (frame["lotsize"] - 10000) / 1000
    frame["lotsize1000"] = frame["lotsize"] / 1000
    frame["sqrft100"] = frame["sqrft"] / 100
    return frame


# --- Etkileşimli adımlar ---------------------------------------------------------------------------------

def _number_values(control: NumberChoice) -> list[float]:
    """Kaydırıcının uçları ve ortası (varsayılandan farklı olanlar)."""

    middle = control.normalize(control.minimum + (control.maximum - control.minimum) / 2)
    values = {control.normalize(control.minimum), control.normalize(control.maximum), middle}
    return sorted(value for value in values if value != control.default)


def _multi_values(control: MultiChoice) -> list[tuple[str, ...]]:
    """Tek öğeli seçimler, varsayılana bir öğe eklenmiş ve varsayılandan bir öğe çıkarılmış hâller (sınırlar içinde)."""

    options = [value for value, _ in control.options]
    candidates = [(value,) for value in options]
    candidates += [tuple(control.default) + (value,) for value in options if value not in control.default]
    candidates += [tuple(item for item in control.default if item != removed) for removed in control.default]
    values = []
    for candidate in candidates:
        if len(candidate) < control.minimum or (control.maximum is not None and len(candidate) > control.maximum):
            continue
        normalized = control.normalize(candidate)
        if normalized != control.default and normalized not in values:
            values.append(normalized)
    return values


def _single_changes(spec: LabSpec) -> list[dict[str, object]]:
    changes = []
    for control in spec.controls:
        if isinstance(control, NumberChoice):
            values = _number_values(control)
        elif isinstance(control, MultiChoice):
            values = _multi_values(control)
        else:
            values = [value for value, _ in control.options if value != control.default]
        changes += [{control.key: value} for value in values]
    return changes


def test_notes_specification_is_the_default_and_passes_every_check() -> None:
    for spec in LABS:
        resolved = spec.resolve({})
        assert resolved.variant == () and resolved.steps == spec.steps
        assert run_lab(spec).all_passed
    assert sum(len(step.checks) for step in KONU11_LAB.steps) == 149
    assert sum(len(step.checks) for step in KONU12_LAB.steps) == 100
    assert [step.note.section for step in KONU11_LAB.steps] == ["11.1", "11.2", "11.3", "11.4", "11.6", "11.6", "11.7",
                                                                   "11.7", "11.8", "11.9"]
    assert [step.note.section for step in KONU12_LAB.steps] == ["12.4", "12.5", "12.6", "12.7", "12.7", "12.7", "12.8",
                                                                   "12.10"]


@pytest.mark.parametrize("spec", LABS, ids=lambda s: s.topic_key)
def test_a_changed_choice_marks_its_own_step_and_the_steps_that_use_it(spec: LabSpec) -> None:
    changes = _single_changes(spec)
    assert len(changes) >= 20
    for change in changes:
        (key,) = change
        marked = {step.number for step in spec.steps
                  if any(control.key == key for control in (*step.controls, *step.uses))}
        assert spec.resolve(change).variant == tuple(sorted(marked)), change


def _frames_equal(app: pd.DataFrame, script: pd.DataFrame) -> None:
    numeric = [column for column in app.columns if pd.api.types.is_numeric_dtype(app[column])]
    np.testing.assert_allclose(script[numeric].to_numpy(float), app[numeric].to_numpy(float), rtol=0, atol=1e-12)


def _run_python(spec: LabSpec) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    namespace: dict = {}
    show = plt.show
    plt.show = lambda *args, **kwargs: plt.close("all")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(generator(spec, "Python").script(), namespace)
    finally:
        plt.show = show
        plt.close("all")
    return namespace


@pytest.mark.parametrize("spec", LABS, ids=lambda s: s.topic_key)
def test_every_single_choice_gives_the_same_numbers_in_the_app_and_in_python(spec: LabSpec) -> None:
    for change in _single_changes(spec):
        resolved = spec.resolve(change)
        state = run_operations(tuple(op for step in resolved.steps for op in step.operations))
        namespace = _run_python(resolved)
        for name, table in state.tables.items():
            numeric = table.select_dtypes("number")
            np.testing.assert_allclose(namespace[name][numeric.columns].to_numpy(float), numeric.to_numpy(float),
                                       rtol=0, atol=1e-12, err_msg=f"{change} {name}")
        for name, value in state.scalars.items():
            assert float(namespace[name]) == pytest.approx(value, rel=1e-12, abs=1e-12), (change, name)
        for name, frame in state.frames.items():
            _frames_equal(frame, namespace[name])
        for name, model in state.models.items():
            np.testing.assert_allclose(namespace[name].params.to_numpy(), model.params.to_numpy(), rtol=0,
                                       atol=1e-12)
            np.testing.assert_allclose(namespace[name].bse.to_numpy(), model.bse.to_numpy(), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("spec", LABS, ids=lambda s: s.topic_key)
def test_generated_r_runs_for_the_last_option_of_every_control(spec: LabSpec, tmp_path: Path, rscript: str,
                                                               r_environment: dict[str, str]) -> None:
    last = {}
    for change in _single_changes(spec):
        last[next(iter(change))] = change
    for change in last.values():
        path = tmp_path / "secim.R"
        path.write_text(render_script(spec.resolve(change), "R"), encoding="utf-8")
        result = subprocess.run([rscript, str(path)], cwd=tmp_path, capture_output=True, encoding="utf-8",
                                errors="replace", timeout=300, env=r_environment)
        assert result.returncode == 0, (change, result.stderr[-1500:])
        assert "warning" not in (result.stdout + result.stderr).lower(), change
        assert "HATA" not in result.stdout, change
        assert sorted(item.name for item in tmp_path.iterdir()) == ["secim.R"], change


def _clean(text: str) -> bool:
    """Metin boş değil, "nan" içermiyor, eksi sıfır ("−0,00") ve ondalıklı (hesaplanan) sayıya Türkçe ek ("0,35'ye")
    yazmıyor."""

    return (bool(text) and not re.search(r"\bnan\b", text.lower()) and not re.search(r"−0,0+(?![0-9])", text)
            and not re.search(r"\d,\d+'", text))


def _screen(operations, state: LabState, label) -> None:
    """Ekrandaki tablolar ve grafikler kurulabilir; tablo sütun adları tekildir, "nan" yazılmaz."""

    for op in operations:
        shown = None
        if isinstance(op, ShowModel):
            shown = coefficient_display(op, state.models[op.model], label)
        elif isinstance(op, RegressionTable):
            shown = regression_display(op, state, label)
        elif hasattr(op, "result") and op.result in state.tables and not isinstance(op, MonteCarlo):
            shown = display_table(op, state.tables[op.result], label)
        if shown is not None:
            assert shown.columns.is_unique, (type(op).__name__, list(shown.columns))
            assert not shown.astype(str).apply(lambda column: column.str.contains("nan")).any().any(), op
        if isinstance(op, CHART_TYPES):
            figure_for(op, state, label)


@pytest.mark.parametrize("spec", LABS, ids=lambda s: s.topic_key)
def test_changed_steps_explain_the_choice_and_show_clean_tables(spec: LabSpec) -> None:
    for change in [{}] + _single_changes(spec):
        resolved = spec.resolve(change)
        choices = spec.normalize(change)
        state = run_operations(tuple(op for step in resolved.steps for op in step.operations))
        _screen(tuple(op for step in resolved.steps for op in step.operations), state, spec.label)
        for step in spec.steps:
            if step.note_for is not None:
                partial = run_operations(resolved.operations_through(step.number))
                assert _clean(step.note_for(partial, choices)), (change, step.number)


def _note(spec: LabSpec, change: dict, number: int) -> str:
    resolved = spec.resolve(change)
    return spec.step(number).note_for(run_operations(resolved.operations_through(number)), spec.normalize(change))


def test_konu11_notes_follow_the_chosen_specification() -> None:
    assert "0,0903 + (−0,0072) = 0,0831 (yuvarlanmamış katsayılarla 0,0830; Tablo 11.3)" in _note(KONU11_LAB, {}, 2)
    text = _note(KONU11_LAB, {}, 4)
    assert "Eğitim 12 yılda" in text and "−0,2973" in text
    assert "çalışan yoktur: bu noktadaki fark tümüyle doğrusal model varsayımına dayanır" in _note(
        KONU11_LAB, {"adim4_c": 1}, 4)
    assert "yaklaşık 11,5 yılda işaret değiştirir" in _note(KONU11_LAB, {"adim5_kukla": "nonwhite"}, 6)
    text = _note(KONU11_LAB, {}, 7)
    assert "tek bir gözleme çok duyarlıdır" in text and "92.681 fit²" in text and "1,5961" in text
    assert "tek bir gözleme" not in _note(KONU11_LAB, {"adim7_x": "sqrft100"}, 7)
    text = _note(KONU11_LAB, {}, 8)
    assert "88 konutun 82 tanesi bu aralıktadır" in text and "yalnız 4 konut" in text and "11.871" in text
    assert "`female * educ12`" in _note(KONU11_LAB, {"adim10_formul": "acik"}, 10)


def test_konu12_notes_follow_the_chosen_specification() -> None:
    text = _note(KONU12_LAB, {}, 3)
    assert "1,22265 × √(88/84) = 1,2514" in text and "yaklaşık 0,84" in text
    assert "iki hesapta da" in _note(KONU12_LAB, {}, 4)
    assert "bütün hesaplarda" in _note(KONU12_LAB, {"adim4_hc": "HC3"}, 4)
    assert "arsa katsayısında HC1/geleneksel SH oranı 1,08; düzey modelinde 1,95" in _note(KONU12_LAB, {}, 5)
    text = _note(KONU12_LAB, {"adim7_terimler": ("sqrft100",)}, 7)
    assert "t = 6,927, t² = 47,983 (yuvarlanmamış t ile 47,979 = F)" in text
    text = _note(KONU12_LAB, {"adim8_x": ("female",)}, 8)
    assert "White testi Breusch–Pagan testiyle aynıdır" in text
    assert "Breusch–Pagan testiyle aynıdır" not in _note(KONU12_LAB, {"adim8_x": ("educ",)}, 8)


def test_interaction_terms_get_the_labels_of_their_factors_and_unnamed_axes_stay_unnamed() -> None:
    assert interaction_label({"a": "A", "b": "B"}, "a:b") == "A × B"
    assert interaction_label({"a": "A", "b": "B", "a:b": "Özel ad"}, "a:b") == "Özel ad"
    assert KONU11_LAB.label("H₀: fark = 0") == "H₀: fark = 0"
    assert KONU11_LAB.label(None) is None  # adı olmayan tablo sütunları (grafik açıklama başlığı)
    assert all(experiment.label(None) is None for experiment in EXPERIMENTS)


def test_integer_columns_with_negative_values_are_shown_with_the_typographic_minus() -> None:
    """Adım 2'nin ilk sekiz çalışanı: merkezlenmiş eğitim (educ − 12) ve çarpım tam sayıdır ve negatif olabilir."""

    state = run_operations(KONU11_LAB.operations_through(2))
    (op,) = [op for op in KONU11_LAB.step(2).operations if isinstance(op, ShowFrame)]
    text = frame_display(shown_frame(op, state), KONU11_LAB.label).to_string()
    assert "−1" in text and "-" not in text
    years = frame_display(pd.DataFrame({"Yıl": [1980, 1987], "Fark": [-2, 3]}), str).to_string()
    assert "1980" in years and "−2" in years  # binlik ayırıcı eklenmez


# --- Konu 11: adımlar söyledikleri hesabı yapar --------------------------------------------------------------------

def test_konu11_group_slopes_gaps_and_tests(wage1, hprice1) -> None:
    wage = smf.ols(WAGE_FORMULA, data=wage1).fit()
    b = wage.params
    assert tuple(wage.params.index) == ("Intercept", "female", "educ12", "female:educ12", "exper", "tenure")
    assert _printed(b["educ12"] + b["female:educ12"], 0.0830, 4)
    assert _printed(round(b["educ12"], 4) + round(b["female:educ12"], 4), 0.0831, 4)  # Tablo 11.3 notu
    assert _printed(round(b["female"], 4) - 4 * round(b["female:educ12"], 4), -0.2685, 4)  # Tablo 11.4 notu
    assert _printed(b["female"] - 4 * b["female:educ12"], -0.2683, 4)
    assert _printed(100 * math.expm1(b["female"]), -25.72, 2)
    # Merkezleme: farklı c'lerde aynı tahmin edilen değerler; kukla katsayısı c'deki farktır
    for center in (0, 8, 16):
        moved = smf.ols("lwage ~ female * educ_c + exper + tenure",
                        data=wage1.assign(educ_c=wage1["educ"] - center)).fit()
        np.testing.assert_allclose(moved.fittedvalues, wage.fittedvalues, rtol=0, atol=1e-10)
        assert moved.params["female"] == pytest.approx(b["female"] + (center - 12) * b["female:educ12"], abs=1e-10)
    house = smf.ols(HOUSE_FORMULA, data=hprice1).fit()
    h = house.params
    assert _printed(h["lotsize10k"] + h["colonial:lotsize10k"], 1.5960, 4)
    zero = 10000 - 1000 * h["colonial"] / h["colonial:lotsize10k"]
    assert round(zero) == 11871
    slope = house.f_test("colonial:lotsize10k = 0")
    assert _printed(float(slope.fvalue), 5.155, 3) and _printed(float(slope.pvalue), 0.0258, 4)
    lines = house.f_test("colonial = 0, colonial:lotsize10k = 0")
    assert _printed(float(lines.fvalue), 3.039, 3) and _printed(float(lines.pvalue), 0.0533, 4)
    assert _printed(float(wage.f_test("female = 0, female:educ12 = 0").fvalue), 32.785, 3)
    gaps = {educ: b["female"] + (educ - 12) * b["female:educ12"] for educ in (8, 12, 16, 18)}
    assert [round(value, 4) for value in gaps.values()] == [-0.2683, -0.2973, -0.3263, -0.3408]


def test_konu11_house_ranges_counts_and_the_largest_lot(hprice1) -> None:
    ranges = {"lotsize10k": "lotsize", "sqrft100": "sqrft", "bdrms": "bdrms"}
    for key, column in ranges.items():
        low, high = hprice1[column].min(), hprice1[column].max()
        unit = " oda" if key == "bdrms" else " fit²"
        text = f"{low:,.0f}–{high:,.0f}".replace(",", ".") + unit
        assert HOUSE_RANGES[key] == text
    lots = hprice1["lotsize"]
    assert (int(lots.between(3000, 20000).sum()), int((lots > 20000).sum()), int((lots < 3000).sum())) == (82, 4, 2)
    # §11.7: en büyük arsa kolonyal bir konuta aittir; çıkarılınca etkileşimin işareti değişir
    largest = lots.idxmax()
    assert lots[largest] == 92681 and hprice1.loc[largest, "colonial"] == 1
    full = smf.ols(HOUSE_FORMULA, data=hprice1).fit()
    without = smf.ols(HOUSE_FORMULA, data=hprice1.drop(index=largest)).fit()
    assert full.params["colonial:lotsize10k"] < 0 < without.params["colonial:lotsize10k"]
    assert (without.params["lotsize10k"] + without.params["colonial:lotsize10k"]
            > without.params["lotsize10k"])  # kolonyal eğimi daha yüksek çıkar


# --- Konu 12: dayanıklı kovaryans ve testler ----------------------------------------------------------------------

def _sandwich(model, kind: str) -> np.ndarray:
    """HC0–HC3 standart hataları açık formülle: (X'X)⁻¹ X' diag(ω) X (X'X)⁻¹."""

    x = model.model.exog
    residuals = model.resid.to_numpy()
    bread = np.linalg.inv(x.T @ x)
    leverage = np.einsum("ij,jk,ik->i", x, bread, x)
    n, p = x.shape
    omega = {"HC0": residuals**2, "HC1": residuals**2 * n / (n - p), "HC2": residuals**2 / (1 - leverage),
             "HC3": residuals**2 / (1 - leverage) ** 2}[kind]
    return np.sqrt(np.diag(bread @ (x.T * omega) @ x @ bread))


def test_konu12_robust_standard_errors_match_the_sandwich_formula(hprice1) -> None:
    level = smf.ols(LEVEL_FORMULA, data=hprice1).fit()
    state = run_operations(KONU12_LAB.operations_through(3))
    for kind in ("HC0", "HC1", "HC2", "HC3"):
        manual = _sandwich(level, kind)
        library = level.get_robustcov_results(cov_type=kind).bse
        np.testing.assert_allclose(manual, library, rtol=1e-10)
        assert state.scalars[f"sh3_{kind.lower()}"] == pytest.approx(manual[1], rel=1e-10)
    assert [round(value, 4) for value in _sandwich(level, "HC1")] == [37.1382, 1.2514, 1.7725, 8.4786]
    assert _printed(_sandwich(level, "HC3")[1], 7.1485, 4) and _printed(_sandwich(level, "HC0")[1], 1.2227, 4)
    # HC3 hiçbir katsayıda HC0'dan küçük değildir
    assert np.all(_sandwich(level, "HC3") >= _sandwich(level, "HC0"))


def test_konu12_leverage_constants_and_the_largest_lot(hprice1) -> None:
    largest = hprice1["lotsize"].idxmax()
    position = hprice1.index.get_loc(largest)
    for kind, formula in (("duzey", LEVEL_FORMULA), ("log", LOG_FORMULA)):
        model = smf.ols(formula, data=hprice1).fit()
        leverage = model.get_influence().hat_matrix_diag
        assert leverage.argmax() == position
        assert _printed(leverage[position], LEVERAGE[kind], 3)
    level = smf.ols(LEVEL_FORMULA, data=hprice1).fit()
    x, residuals = level.model.exog, level.resid.to_numpy()
    bread = np.linalg.inv(x.T @ x)
    leverage = level.get_influence().hat_matrix_diag
    weights = (x @ bread)[:, 1] ** 2 * residuals**2 / (1 - leverage) ** 2  # arsa katsayısının HC3 varyans katkıları
    assert weights[position] / weights.sum() > 0.99
    assert _printed((1 - leverage[position]) ** 2, 0.025, 3) and round(1 / (1 - leverage[position]) ** 2) == 40
    assert hprice1["lotsize"].drop(index=largest).max() == 31000
    without = smf.ols(LEVEL_FORMULA, data=hprice1.drop(index=largest)).fit()
    assert without.params["lotsize1000"] > 4 * level.params["lotsize1000"]  # arsa katsayısı bu gözleme duyarlı


def test_konu12_breusch_pagan_white_and_the_robust_wald_test(hprice1) -> None:
    level = smf.ols(LEVEL_FORMULA, data=hprice1).fit()
    squared = level.resid**2
    auxiliary = smf.ols("u2 ~ lotsize1000 + sqrft100 + bdrms", data=hprice1.assign(u2=squared)).fit()
    lm = len(hprice1) * auxiliary.rsquared
    assert lm == pytest.approx(het_breuschpagan(level.resid, level.model.exog)[0], rel=1e-10)
    assert _printed(lm, 14.092, 3) and _printed(stats.chi2.sf(lm, 3), 0.0028, 4)
    white = het_white(level.resid, level.model.exog)
    assert _printed(white[0], 33.732, 3) and white[1] < 0.001
    state = run_operations(KONU12_LAB.operations_through(2))
    assert state.scalars["lm_elle"] == pytest.approx(lm, rel=1e-10)
    robust = level.get_robustcov_results(cov_type="HC1")
    restriction = np.zeros((2, 4))
    restriction[0, 1] = restriction[1, 3] = 1
    beta = level.params.to_numpy()
    wald = float(beta @ restriction.T @ np.linalg.inv(restriction @ robust.cov_params() @ restriction.T)
                 @ restriction @ beta)
    test = robust.f_test("lotsize1000 = 0, bdrms = 0")
    assert float(test.fvalue) == pytest.approx(wald / 2, rel=1e-10)
    assert _printed(wald, 4.730, 3) and _printed(float(test.fvalue), 2.365, 3)
    assert _printed(float(test.pvalue), 0.1002, 4) and _printed(stats.chi2.sf(wald, 2), 0.094, 3)
    assert _printed(float(level.f_test("lotsize1000 = 0, bdrms = 0").fvalue), 6.610, 3)


def _white_reference(resid, exog) -> tuple[float, float, int]:
    """Bağımsız hesap: kuklanın karesi gibi aynı sütunları atar; statsmodels OLS ile n·R², q = kalan sütun − 1."""

    x = np.asarray(exog, dtype=float)
    first, second = np.triu_indices(x.shape[1])
    products = x[:, first] * x[:, second]
    kept: list[int] = []
    for column in range(products.shape[1]):
        if not any(np.array_equal(products[:, column], products[:, other]) for other in kept):
            kept.append(column)
    auxiliary = sm.OLS(np.asarray(resid) ** 2, products[:, kept]).fit()
    lm = len(products) * auxiliary.rsquared
    return lm, float(stats.chi2.sf(lm, len(kept) - 1)), len(kept) - 1


WHITE_SPECS = (
    ("wage1", "lwage ~ female", 1),
    ("wage1", "lwage ~ educ + exper + tenure + female", 13),
    ("wage1", "wage ~ female + married + nonwhite", 6),
    ("wage1", "lwage ~ educ + exper + tenure + female + married + nonwhite", 24),
    ("hprice1", LEVEL_FORMULA, 9),
    ("hprice1", LOG_FORMULA, 9),
)
"""(veri, formül, White yardımcı regresyonundaki bağımsız terim sayısı q = k(k + 3)/2 − kukla sayısı)."""


@pytest.mark.parametrize(("dataset", "formula", "q"), WHITE_SPECS, ids=[spec[1] for spec in WHITE_SPECS])
def test_white_test_counts_the_square_of_a_dummy_once(dataset, formula, q, wage1, hprice1) -> None:
    """0/1 kuklanın karesi kendisidir: yardımcı tasarım tam ranklı değildir, ama LM ve p bağımsız hesapla aynıdır."""

    model = smf.ols(formula, data={"wage1": wage1, "hprice1": hprice1}[dataset]).fit()
    lm, p = hetero_test(model, "white")
    reference_lm, reference_p, reference_q = _white_reference(model.resid, model.model.exog)
    assert reference_q == q
    assert lm == pytest.approx(reference_lm, rel=1e-9) and p == pytest.approx(reference_p, rel=1e-9)
    if dataset == "hprice1":  # kuklasız modelde het_white da aynı sayıyı verir (notlardaki Tablo 12.2)
        assert (lm, p) == pytest.approx(het_white(model.resid, model.model.exog)[:2], rel=1e-9)


def test_white_equals_breusch_pagan_for_a_single_dummy(wage1) -> None:
    """Tek 0/1 kuklada White'ın yardımcı regresyonu Breusch–Pagan'ınkiyle aynıdır (kuklanın karesi kendisidir)."""

    model = smf.ols("lwage ~ female", data=wage1).fit()
    assert hetero_test(model, "white") == pytest.approx(hetero_test(model, "bp"), rel=1e-9)


def test_white_design_drops_all_zero_columns_and_scales_the_rest_to_unit_length() -> None:
    exog = np.column_stack([np.ones(6), [1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]])  # birbirini dışlayan iki kukla
    design = white_design(exog)
    assert design.shape == (6, 5)  # sabit, d1, d2, d1², d2²: d1 × d2 yalnız sıfırdır
    np.testing.assert_allclose(np.linalg.norm(design, axis=0), 1.0, rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="sabit terim"):
        white_test(np.ones(5), np.column_stack([np.arange(5.0), np.arange(5.0) ** 2]))


def test_white_rank_decision_has_a_wide_margin_in_every_lab_design(wage1, hprice1) -> None:
    """Yinelenen sütunun tekil değeri ile gerçek terimlerinki arasında ``WHITE_RCOND`` her iki yandan en az 1e3 kat
    uzaktadır (ölçülen marj 1e5 ve 1e6; makineler arası gürültüye pay bırakılır). statsmodels ``het_white`` ise
    gürültüyü yüzde onluk bir farkla ayırıyordu (en küçük tekil değer 6,1e-16, eşik 6,7e-16)."""

    names = ("educ", "exper", "tenure", "female", "married", "nonwhite")
    models = [smf.ols("lwage ~ " + " + ".join(chosen), data=wage1) for size in range(1, len(names) + 1)
              for chosen in itertools.combinations(names, size)]
    models += [smf.ols(formula, data=hprice1) for formula in (LEVEL_FORMULA, LOG_FORMULA)]
    for model in models:
        singular = np.linalg.svd(white_design(model.exog), compute_uv=False)
        ratio = singular / singular[0]
        rank = int((ratio > WHITE_RCOND).sum())
        assert ratio[rank - 1] > 1e3 * WHITE_RCOND, model.formula
        if rank < len(ratio):
            assert ratio[rank] < WHITE_RCOND / 1e3, model.formula


def test_white_test_does_not_rely_on_statsmodels_het_white(monkeypatch) -> None:
    """statsmodels 0.14.6'daki ``het_white`` kukla varken rankı iki ayrı eşikle karşılaştıran bir ``assert`` içerir ve
    makineye göre AssertionError verebilir (Windows'ta Konu 12 Adım 8, açıklayıcı ``female``). Uygulama da üretilen
    Python kodu da ona dayanmaz: işlev çağrılırsa test düşer."""

    import statsmodels.stats.diagnostic as diagnostic

    def refuse(*args, **kwargs):
        raise AssertionError("het_white çağrılmamalı")

    monkeypatch.setattr(diagnostic, "het_white", refuse)
    spec = KONU12_LAB.resolve({"adim8_x": ("female",)})
    state = run_operations(tuple(op for step in spec.steps for op in step.operations))
    script = render_script(spec, "Python")
    assert not re.search(r"het_white\(", script) and "import het_breuschpagan, het_white" not in script
    assert "def white_testi(model):" in script
    namespace = _run_python(spec)
    for name in ("wh_wd", "p_wh_wd", "wh_wl", "p_wh_wl", "wh_wl_n", "p_wh_wl_n"):
        assert float(namespace[name]) == pytest.approx(state.scalars[name], rel=1e-12, abs=1e-12), name
    assert state.scalars["wh_wl"] == pytest.approx(state.scalars["bp_wl"], rel=1e-9)  # tek kukla: White = BP


# --- Sezgi deneyleri ------------------------------------------------------------------------------------

def _extremes(experiment) -> list[dict[str, float]]:
    settings = [experiment.defaults()]
    for parameter in experiment.parameters:
        for value in (parameter.minimum, parameter.maximum):
            settings.append(dict(experiment.defaults(), **{parameter.key: value}))
    return settings


def _corners(experiment) -> list[dict[str, float]]:
    ends = [(parameter.key, (parameter.minimum, parameter.maximum)) for parameter in experiment.parameters]
    return [dict(zip((key for key, _ in ends), values)) for values in itertools.product(*(pair for _, pair in ends))]


def test_block_simulations_use_the_batch_path() -> None:
    for experiment in EXPERIMENTS:
        for parameters in _extremes(experiment):
            simulations = [op for op in experiment.build(parameters) if isinstance(op, MonteCarlo)]
            assert simulations and all(batchable(op) for op in simulations), (experiment.key, parameters)


@pytest.mark.parametrize("experiment", EXPERIMENTS, ids=lambda e: e.key)
def test_experiment_texts_and_tables_are_correct_at_the_extremes_and_corners(experiment) -> None:
    for parameters in _extremes(experiment) + _corners(experiment):
        with np.errstate(all="ignore"):
            operations = experiment.build(parameters)
            state = run_operations(operations)
        metrics = experiment.metrics(state, parameters)
        assert len(metrics) == 4 and all(len(metric.value) <= 13 for metric in metrics), parameters
        assert all("nan" not in metric.value.lower() for metric in metrics), parameters
        assert _clean(experiment.takeaway(state, parameters)), parameters
        assert all(line for line in experiment.dgp(parameters))
        producers = {op.result: op for op in operations if hasattr(op, "result")}
        for entry in experiment.tables:
            producer = producers[entry[0]]
            shown = (regression_display(producer, state, experiment.label) if isinstance(producer, RegressionTable)
                     else display_table(producer, state.tables[entry[0]], experiment.label))
            assert shown.columns.is_unique, (experiment.key, list(shown.columns))
        for op in operations:
            if isinstance(op, CHART_TYPES):
                figure_for(op, state, experiment.label)


def test_konu12_experiment_1_reproduces_table_12_6_and_the_figures() -> None:
    state = run_operations(HETERO.build(HETERO.defaults()))
    s = state.scalars
    for name, printed, decimals in (("ort_b1", 1.986, 3), ("sd_b1", 0.734, 3), ("ort_sh_gel", 0.583, 3),
                                    ("ort_sh_hc1", 0.708, 3), ("ort_sh_hc3", 0.734, 3)):
        assert _printed(s[name], printed, decimals), name
    for name, printed in (("kap_gel", 88.10), ("kap_hc1", 93.45), ("kap_hc3", 94.35)):
        assert _printed(100 * s[name], printed, 2), name
    # Şekil 12.1–12.2: benzetimin ilk tekrarı (önce X, sonra Z; tohum 305)
    rng = np.random.default_rng(305)
    x = rng.uniform(0, 4, 60)
    z = rng.normal(0, 1, 60)
    sample = state.frames["orneklem"]
    np.testing.assert_array_equal(sample["x"].to_numpy(), x)
    np.testing.assert_array_equal(sample["z"].to_numpy(), z)
    np.testing.assert_allclose(sample["y"].to_numpy(), 1 + 2 * x + (0.3 + 0.7 * x**2) * z, rtol=0, atol=1e-12)
    sigma = average_sd(0.7)
    assert sigma == pytest.approx(math.sqrt(0.09 + 0.42 * 16 / 3 + 0.49 * 256 / 5), rel=1e-12)
    assert round(sigma, 2) == 5.24
    np.testing.assert_allclose(sample["y_homo"].to_numpy(), 1 + 2 * x + sigma * z, rtol=0, atol=1e-12)
    first = state.tables["tekrarlar"].iloc[0]
    assert first["b1"] == pytest.approx(smf.ols("y ~ x", data=sample).fit().params["x"], rel=1e-12)
    # Deney 1, c = 0: homoskedastik süreçte geleneksel aralık doğru formüldür
    s0 = run_operations(HETERO.build(dict(HETERO.defaults(), c=0.0))).scalars
    assert abs(100 * s0["kap_gel"] - 95) < 1.2 and s0["sd_b1"] == pytest.approx(s0["ort_sh_gel"], rel=0.06)


def test_konu12_experiments_match_their_theory() -> None:
    s = run_operations(TESTS.build(dict(TESTS.defaults(), gamma=0.0))).scalars
    assert abs(s["ret_bp"] - 0.05) < 0.015 and abs(s["ret_white"] - 0.05) < 0.015  # H₀ doğru: I. tür hata
    s = run_operations(TESTS.build(TESTS.defaults())).scalars  # U biçimli varyans: White BP'den güçlü
    assert s["ret_white"] > 0.9 > 0.5 > s["ret_bp"]
    s = run_operations(SMALL.build(SMALL.defaults())).scalars
    rates = [s[f"kap_{kind}"] for kind in ("hc0", "hc1", "hc2", "hc3")]
    assert rates == sorted(rates) and s["kap_gel"] < rates[-1]
    s = run_operations(SMALL.build(dict(SMALL.defaults(), n=200, delta=0.0))).scalars
    assert abs(100 * s["kap_gel"] - 95) < 1.2


def test_konu11_experiments_match_their_theory() -> None:
    assert gap(0.0, 5) == pytest.approx(-0.45) and gap(-0.05, 4) == pytest.approx(0.0, abs=1e-12)
    s = run_operations(CENTERING.build(CENTERING.defaults())).scalars
    assert s["sd_c"] > s["sd_13"]  # merkez verinin dışında: fark daha belirsiz
    s = run_operations(OMITTED.build(OMITTED.defaults())).scalars
    assert s["ort_add_d"] == pytest.approx(gap(OMITTED.defaults()["g1"], 13), abs=0.02)
    first = run_operations(JOINT.build(dict(JOINT.defaults(), c=0))).scalars
    second = run_operations(JOINT.build(dict(JOINT.defaults(), c=13))).scalars
    assert first["ret_reddet_f"] == second["ret_reddet_f"]  # F merkezlemeden bağımsız (aynı çekilişler)
    assert second["ret_reddet_d"] > first["ret_reddet_d"]  # veri ortasında γ₀ testi daha güçlü


# --- Sorular --------------------------------------------------------------------------------------------------

EQUATIONS = {
    ("konu11", "e01"): (("g0 + g1*mu", "\\hat\\gamma0 + \\hat\\gamma1\\mu", "γ̂0 + γ̂1μ", "g0 + g1*µ"),
                        ("g0 - g1*mu", "g0*mu + g1", "g1 + g0")),
    ("konu11", "e05"): (("g0 + g1*(L - 10)", "\\hat\\gamma0+\\hat\\gamma1(L-10)", "γ̂0 + γ̂1(L − 10)"),
                        ("g0 + g1*L", "g0 + g1*(L - 10000)/1000", "g0 + g1*(10 - L)")),
    ("konu12", "e02"): (("s0*sqrt(n/(n - k - 1))", "SE_HC0*sqrt(n/(n-k-1))", "s_0\\sqrt{n/(n-k-1)}"),
                        ("s0*sqrt(n/(n - k))", "s0*n/(n - k - 1)", "s0*sqrt((n - k - 1)/n)")),
    ("konu12", "e04"): (("1/X^2", "X^(-2)", "1/(X*X)", "exp(-2*log(X))"), ("1/X", "X^2", "1/sqrt(X)")),
    ("konu12", "e05"): (("sqrt((sb - a)/c)", "\\sqrt{(\\bar{σ}-a)/c}", "sqrt((\\overline{σ} - a)/c)",
                         "\\sqrt{\\frac{\\bar\\sigma - a}{c}}", "sqrt((σ̄-a)/c)"),
                        ("sqrt((a - sb)/c)", "(sb - a)/c", "sqrt(sb/c)")),
}


@pytest.mark.parametrize("topic, key", sorted(EQUATIONS))
def test_equation_questions_accept_equivalent_forms_and_reject_wrong_ones(topic: str, key: str) -> None:
    quiz = {"konu11": KONU11_QUIZ, "konu12": KONU12_QUIZ}[topic]
    question = next(item for item in quiz.questions if item.key == key)
    accepted, rejected = EQUATIONS[(topic, key)]
    for text in accepted:
        assert grade(question, text).correct, text
    for text in rejected:
        assert not grade(question, text).correct, text


def test_quiz_numbers_are_computed_not_typed(wage1, hprice1) -> None:
    house = smf.ols(HOUSE_FORMULA, data=hprice1).fit()
    h = house.params
    uncentered = smf.ols("price ~ colonial * lotsize1000 + sqrft100 + bdrms", data=hprice1).fit()
    assert _printed(uncentered.params["colonial"], 52.3081, 4) and _printed(uncentered.tvalues["colonial"], 2.356, 3)
    assert _printed(h["colonial"] / -h["colonial:lotsize10k"], 1.871, 3)
    wage = smf.ols(WAGE_FORMULA, data=wage1).fit()
    restricted = smf.ols("lwage ~ educ12 + exper + tenure", data=wage1).fit()
    assert (int(restricted.df_resid), int(wage.df_resid)) == (522, 520)  # Konu 11 b05
    level = smf.ols(LEVEL_FORMULA, data=hprice1).fit()
    bdrms = level.params["bdrms"]
    se = level.get_robustcov_results(cov_type="HC1").bse[3]
    critical = stats.t.ppf(0.975, 84)
    assert _printed(critical, 1.989, 3)
    assert _printed(13.8525 - 1.989 * 8.4786, -3.011, 3) and _printed(13.8525 + 1.989 * 8.4786, 30.716, 3)
    assert _printed(bdrms - critical * se, -3.008, 3) and _printed(bdrms + critical * se, 30.713, 3)
    s20 = run_operations(HETERO.build(dict(HETERO.defaults(), n=20))).scalars  # Konu 12 b04 ve d04
    assert (round(s20["sd_b1"], 3), round(s20["ort_sh_gel"], 3), round(s20["ort_sh_hc1"], 3),
            round(s20["ort_sh_hc3"], 3)) == (1.286, 0.993, 1.135, 1.274)
    s15 = run_operations(HETERO.build(dict(HETERO.defaults(), c=1.5))).scalars  # Konu 12 k05 ve d04
    assert (round(s15["ort_b1"], 3), round(s15["sd_b1"], 3), round(s15["ort_sh_hc3"], 3)) == (1.971, 1.544, 1.544)
    assert _printed(100 * s15["kap_hc3"], 94.35, 2) and _printed(s15["sd_b1"] / math.sqrt(4000), 0.024, 3)
