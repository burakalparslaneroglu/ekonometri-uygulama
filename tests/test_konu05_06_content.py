"""Konu 5–6 bloğu: etkileşimli spesifikasyonlar, motorun yeni işlemleri, toplu Monte Carlo, Sezgi deneyleri ve sorular.

Genel sözleşmeler (notlardaki her sayının uygulamada, üretilen Python'da ve R'de tutması; deneylerin üretilen Python
koduyla birebir aynı sayıları vermesi) ``test_all_labs.py``'dedir; bu dosya bloğa özgü olanları denetler:

* notlardan farklı her seçimde uygulamanın sayıları ile üretilen Python kodunun sayıları 1e-12 düzeyinde aynıdır ve
  üretilen R kodu çalışır; seçimin sonraki adımlara geçişi (``LabSpec.resolve``) doğrudur, veriyi yeniden yükleyen
  adımlar notlardaki gibi kalır; adım metinleri her seçimde doğru kalır;
* adımlar söyledikleri hesabı yapar ve bağımsız bir hesapla (statsmodels, numpy) aynı sayıyı verir: ceteris paribus
  karşılaştırma, tahmin ve artık, artıkların artıklara regresyonu, düzeltilmiş R², konut tahminleri, eksik değişken
  ayrıştırması, VIF;
* motorun yeni işlemleri: özet tablosu (``SummaryTable``), artık sütunu (``Residuals``), makale tablosunda düzeltilmiş
  R² ve terim basamağı, sonuç tablosundan istatistik, Monte Carlo gövdesinde ayrılmış adlar;
* toplu (vektörel) Monte Carlo yolu döngüyle aynı sayıları verir ve üreteci aynı durumda bırakır; eksik değerde
  döngüye döner;
* Sezgi deneyleri varsayılan ayarlarda notlardaki tabloları verir (Tablo 6.3, 6.5, 6.7, 6.8), kuramsal ilişkileri
  tutturur ve metinleri kaydırıcıların uçlarında da doğru kalır;
* denklem ve boşluk sorularının kabul ve ret yazımları; sorulardaki sayılar veriden hesaplanır.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from core import wooldridge_data as W
from core.codegen.base import generator, render_script
from core.labs import expr as E
from core.labs.konu05 import EXERCISE, HOUSES, KONU05_LAB, NOTES_X
from core.labs.konu06 import COLUMNS, KONU06_LAB, VIF_DEFAULT
from core.labs.runner import (
    LabState,
    batchable,
    execute,
    monte_carlo_batch,
    monte_carlo_loop,
    run_lab,
    run_operations,
)
from core.labs.sezgi_konu05 import COMPARE, FIT, KONU05_EXPERIMENTS, PARTIAL
from core.labs.sezgi_konu06 import COLLINEAR, KONU06_EXPERIMENTS, LEVELS, OMITTED, UNBIASED
from core.labs.spec import (
    INTERCEPT,
    OLS,
    CopyFrame,
    Derive,
    Draw,
    LabSpec,
    LoadWooldridge,
    MonteCarlo,
    MultiChoice,
    NewSample,
    NumberChoice,
    RegressionTable,
    Residuals,
    Statistic,
    SummaryTable,
    operation_reads,
    operation_writes,
)
from core.quiz.konu05 import KONU05_QUIZ
from core.quiz.konu06 import KONU06_QUIZ
from core.quiz.expression import FormulaError, Symbol, parse
from core.quiz.model import grade
from topics.lab_ui import display_table, regression_display

LABS = (KONU05_LAB, KONU06_LAB)
EXPERIMENTS = KONU05_EXPERIMENTS + KONU06_EXPERIMENTS


def _printed(value: float, expected: float, decimals: int) -> bool:
    """Değer, notlarda ``decimals`` basamakla basılı sayıya yuvarlanır."""

    return abs(value - expected) <= 0.5 * 10 ** -decimals + 1e-12


# --- Etkileşimli adımlar ---------------------------------------------------------------------------------

def _number_values(control: NumberChoice) -> list[float]:
    """Kaydırıcının uçları ve ortası (varsayılandan farklı olanlar)."""

    middle = control.normalize(control.minimum + (control.maximum - control.minimum) / 2)
    values = {control.normalize(control.minimum), control.normalize(control.maximum), middle}
    return sorted(value for value in values if value != control.default)


def _multi_values(control: MultiChoice) -> list[tuple[str, ...]]:
    """Tek öğeli seçimler (izin veriliyorsa), varsayılana bir öğe eklenmiş ve varsayılandan bir öğe çıkarılmış hâller."""

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
    assert sum(len(step.checks) for step in KONU05_LAB.steps) == 122
    assert sum(len(step.checks) for step in KONU06_LAB.steps) == 49
    assert [step.note.section for step in KONU05_LAB.steps] == ["5.3", "5.3", "5.4", "5.5", "5.6", "5.7", "5.8",
                                                                  "5.9", "5.10", "5.11"]
    assert [step.note.section for step in KONU06_LAB.steps] == ["6.8", "6.8", "6.8", "6.9", "6.12", "6.13"]


def test_a_changed_choice_marks_its_step_and_every_step_that_uses_it() -> None:
    # Adım 4 ve 7 veriyi yeniden yükler ya da başka veri kullanır; Adım 9 kendi modellerini kurar.
    for chosen in (("educ",), ("exper", "tenure"), ("educ", "exper", "tenure", "numdep")):
        resolved = KONU05_LAB.resolve({"adim1_x": chosen})
        assert resolved.variant == (1, 2, 3, 5, 6, 8), chosen
        for number in (4, 7, 9, 10):
            assert resolved.step(number).checks == KONU05_LAB.step(number).checks, (chosen, number)
    assert KONU06_LAB.resolve({"adim1_z": "tenure"}).variant == (1, 2)
    assert KONU06_LAB.resolve({"adim1_z": "numdep"}).step(3).checks == KONU06_LAB.step(3).checks
    shared = {"adim1_x", "adim1_z"}
    for spec in LABS:
        for change in _single_changes(spec):
            (key,) = change
            if key in shared:
                continue
            own = next(step.number for step in spec.steps if any(c.key == key for c in step.controls))
            assert spec.resolve(change).variant == (own,), change


def test_vif_needs_at_least_two_variables() -> None:
    with pytest.raises(ValueError, match="en az 2"):
        KONU06_LAB.resolve({"adim5_x": ("sqrft",)})


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
    changes = _single_changes(spec)
    assert len(changes) >= (30 if spec is KONU05_LAB else 12)
    for change in changes:
        resolved = spec.resolve(change)
        state = run_operations(tuple(op for step in resolved.steps for op in step.operations))
        namespace = _run_python(resolved)
        for name, table in state.tables.items():
            np.testing.assert_allclose(namespace[name].to_numpy(float), table.to_numpy(float), rtol=0, atol=1e-12,
                                       err_msg=f"{change} {name}")
        for name, value in state.scalars.items():
            assert float(namespace[name]) == pytest.approx(value, abs=1e-12), (change, name)
        for name, frame in state.frames.items():
            _frames_equal(frame, namespace[name])
        for name, model in state.models.items():
            np.testing.assert_allclose(namespace[name].params.to_numpy(), model.params.to_numpy(), rtol=0,
                                       atol=1e-12)


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
    """Metin boş değil, "nan" içermiyor ve eksi sıfır ("−0,00") yazmıyor."""

    return bool(text) and not re.search(r"\bnan\b", text.lower()) and not re.search(r"−0,0+(?![0-9])", text)


@pytest.mark.parametrize("spec", LABS, ids=lambda s: s.topic_key)
def test_changed_steps_explain_the_chosen_specification(spec: LabSpec) -> None:
    for change in [{}] + _single_changes(spec):
        resolved = spec.resolve(change)
        choices = spec.normalize(change)
        for step in spec.steps:
            if step.note_for is None:
                continue
            state = run_operations(resolved.operations_through(step.number))
            assert _clean(step.note_for(state, choices)), (change, step.number)


def _note(spec: LabSpec, change: dict, number: int) -> str:
    resolved = spec.resolve(change)
    return spec.step(number).note_for(run_operations(resolved.operations_through(number)), spec.normalize(change))


def test_notes_follow_the_chosen_specification() -> None:
    single = _note(KONU05_LAB, {"adim1_x": ("educ",)}, 1)
    assert "tek açıklayıcı değişken" in single and "farklı karşılaştırmalar" not in single
    assert "katsayı değişmez" in _note(KONU05_LAB, {"adim1_x": ("educ",)}, 6)
    assert "Eğitim modelde olmadığı için iki çalışanın tahmini aynıdır" in _note(KONU05_LAB, {"adim1_x": ("exper",)}, 2)
    assert "iç içe değildir" in _note(KONU05_LAB, {"adim1_x": ("exper", "tenure")}, 8)
    assert "iç içe değildir" not in _note(KONU05_LAB, {}, 8)
    rooms = _note(KONU05_LAB, {"adim7_x": ("bdrms",)}, 7)
    assert "diğer değişkenler aynıyken" not in rooms and "ikisi de modelde olmadığı için" in rooms
    assert "yalnız konut büyüklüğünün katkısıdır" in _note(KONU05_LAB, {"adim7_x": ("sqrft", "bdrms")}, 7)
    assert "iki tahmin aynıdır" in _note(KONU05_LAB, {"adim7_x": ("sqrft", "lotsize")}, 7)
    assert "birden fazla katsayının katkısıdır" in _note(KONU05_LAB, {}, 7)
    assert "Deneyim eklenince yükselir; kıdem eklenince düşer." in _note(KONU06_LAB, {}, 3)
    text = _note(KONU06_LAB, {"adim3_modeller": ("yalniz", "kidem")}, 3)
    assert "Kıdem eklenince yükselir." in text and "Deneyim" not in text
    assert "en az iki model" in _note(KONU06_LAB, {"adim3_modeller": ("tam",)}, 3)
    text = _note(KONU06_LAB, {"adim1_z": "tenure"}, 2)  # kıdem eğitimle negatif ilişkili, ücretle pozitif
    assert "işaretler farklı" in text and "aşağı yönlü" in text and "0,5414 = 0,5691 + (−0,0278)" in text
    text = _note(KONU06_LAB, {}, 2)  # notlar: deneyim eğitimle negatif ilişkili; kısa model eğitimi düşük bulur
    assert "1,468 yıl daha düşüktür" in text and "aşağı yönlü" in text and "0,5414 = 0,6443 + (−0,1029)" in text
    assert "Vergi değeri" in _note(KONU06_LAB, {"adim5_x": ("sqrft", "bdrms", "lotsize", "assess")}, 5)
    assert "tam bağlantı yoktur" in _note(KONU06_LAB, {"adim5_x": ("sqrft", "lsqrft")}, 5)


def test_figure_titles_name_only_the_variables_held_fixed() -> None:
    def titles(chosen):
        state = run_operations(KONU05_LAB.resolve({"adim1_x": chosen}).operations_through(2))
        return [key for key in state.plots if key.startswith("LineChart")]

    assert titles(NOTES_X) == ["LineChart:Şekil 5.1: kıdem medyan düzeyinde (2 yıl) sabitken üç deneyim düzeyinde "
                               "tahmin çizgileri"]
    (only,) = titles(("educ", "exper"))
    assert "kıdem" not in only and only.startswith("LineChart:Seçtiğiniz modelle Şekil 5.1")
    assert titles(("educ", "tenure")) == []
    wage1 = W.load("wage1")
    assert wage1["tenure"].median() == 2 and wage1["numdep"].median() == 1


# --- Adımların hesabı: bağımsız hesapla karşılaştırma -------------------------------------------------------

def test_konu05_wage_steps_compute_what_they_say() -> None:
    wage1 = W.load("wage1")
    for chosen in (NOTES_X, ("educ", "exper", "tenure", "numdep"), ("educ", "numdep"), ("tenure",)):
        change = {"adim1_x": chosen, "adim2_fark": 3, "adim3_egitim": 14, "adim3_ucret": 6.5}
        resolved = KONU05_LAB.resolve(change)
        s = run_operations(resolved.operations_through(8)).scalars
        fit = smf.ols("wage ~ " + " + ".join(chosen), data=wage1).fit()
        for name in chosen:
            assert s[f"b_{name}"] == pytest.approx(fit.params[name], rel=1e-10), (chosen, name)
        if "educ" in chosen:
            assert s["fark_BA"] == pytest.approx(3 * fit.params["educ"], rel=1e-10)
        else:
            assert s["fark_BA"] == pytest.approx(0, abs=1e-12)
        profile = pd.DataFrame({"educ": [14.0], "exper": [20.0], "tenure": [10.0], "numdep": [1.0]})
        predicted = float(fit.predict(profile)[0])
        assert s["birey_tahmin"] == pytest.approx(predicted, rel=1e-12)
        assert s["birey_artik"] == pytest.approx(6.5 - predicted, rel=1e-12)
        assert s["artik_toplami"] == pytest.approx(0, abs=1e-9)
        assert s["ort_tahmin"] == pytest.approx(s["ort_ucret"], rel=1e-12)
        k, n = len(chosen), 526
        assert s["r2d_kod"] == pytest.approx(1 - (1 - s["r2_kod"]) * (n - 1) / (n - k - 1), rel=1e-12)
        assert s["r2d_elle"] == pytest.approx(fit.rsquared_adj, rel=1e-10)
        assert s["r2d_coklu"] == pytest.approx(fit.rsquared_adj, rel=1e-12)
        if "educ" in chosen:  # iç içe modeller: değişken eklemek R²'yi azaltmaz
            assert s["r2_coklu"] >= s["r2_basit"] - 1e-12


def test_konu05_partial_regression_recovers_the_multiple_coefficient() -> None:
    for focus in NOTES_X:
        s = run_operations(KONU05_LAB.resolve({"adim4_x": focus}).operations_through(4)).scalars
        assert s["kismi_egim"] == pytest.approx(s["tam_katsayi"], abs=1e-10), focus
        assert abs(s["kismi_fark"]) < 1e-10


def test_konu05_houses_fit_and_article_steps_compute_what_they_say() -> None:
    hprice1 = W.load("hprice1")
    houses = pd.DataFrame(HOUSES, columns=["konut", "sqrft", "bdrms", "lotsize"])
    for chosen in (("sqrft", "bdrms", "lotsize"), ("bdrms",), ("sqrft", "lotsize")):
        s = run_operations(KONU05_LAB.resolve({"adim7_x": chosen}).operations_through(7)).scalars
        fit = smf.ols("price ~ " + " + ".join(chosen), data=hprice1).fit()
        for house, value in zip(houses["konut"], fit.predict(houses)):
            assert s[f"konut_{house}"] == pytest.approx(value, rel=1e-12), (chosen, house)
        expected = fit.params["bdrms"] if "bdrms" in chosen else 0.0
        assert s["konut_fark"] == pytest.approx(expected, rel=1e-10, abs=1e-12)
    for n in (20, 200, 2000):
        table = run_operations(KONU05_LAB.resolve({"adim8_n": n}).operations_through(8)).tables["abc"]["deger"]
        for label, r2, k in EXERCISE:
            assert table[label] == pytest.approx(1 - (1 - r2) * (n - 1) / (n - k - 1), rel=1e-12), (n, label)
    assert run_operations(KONU05_LAB.operations_through(8)).tables["abc"]["deger"].round(3).tolist() == [
        0.397, 0.421, 0.417]
    state = run_operations(KONU05_LAB.resolve({"adim9_sutunlar": ("log_basit", "deneyimli")}).operations_through(9))
    table = state.tables["makale"]
    assert list(table.columns) == ["(1) Ücret", "(2) ln(Ücret)"]  # seçenek sırası: deneyimli, log_basit
    assert list(table.index) == ["educ", "exper", "Intercept", "n", "r2", "adj_r2"]
    fit = smf.ols("lwage ~ educ", data=W.load("wage1")).fit()
    assert table.loc["adj_r2", "(2) ln(Ücret)"] == pytest.approx(fit.rsquared_adj, rel=1e-12)
    assert math.isnan(table.loc["exper", "(2) ln(Ücret)"])


def test_konu06_decomposition_holds_exactly_in_the_sample() -> None:
    wage1 = W.load("wage1")
    for z in ("exper", "tenure", "numdep"):
        s = run_operations(KONU06_LAB.resolve({"adim1_z": z}).operations_through(2)).scalars
        short = smf.ols("wage ~ educ", data=wage1).fit().params["educ"]
        long = smf.ols(f"wage ~ educ + {z}", data=wage1).fit().params
        delta = smf.ols(f"{z} ~ educ", data=wage1).fit().params["educ"]
        assert s["kisa_egitim"] == pytest.approx(short, rel=1e-12)
        assert s["uzun_egitim"] == pytest.approx(long["educ"], rel=1e-12)
        assert s["yardimci_egim"] == pytest.approx(delta, rel=1e-12)
        assert s["katki"] == pytest.approx(long[z] * delta, rel=1e-10)
        assert s["yeniden"] == pytest.approx(s["kisa_egitim"], abs=1e-10), z  # β̃₁ = β̂₁ + β̂₂δ̃₁ örneklemde tam


def test_konu06_controls_perfect_collinearity_and_vif_steps() -> None:
    wage1, hprice1 = W.load("wage1"), W.load("hprice1")
    for selected in (("yalniz", "deneyim", "tam", "kidem"), ("genis",)):  # en çok dört sütun
        s = run_operations(KONU06_LAB.resolve({"adim3_modeller": selected}).operations_through(3)).scalars
        for key in selected:
            formula = "wage ~ " + " + ".join(COLUMNS[key][1])
            assert s[f"egitim_{key}"] == pytest.approx(smf.ols(formula, data=wage1).fit().params["educ"], rel=1e-12)
    with pytest.raises(ValueError, match="en çok 4"):
        KONU06_LAB.resolve({"adim3_modeller": tuple(COLUMNS)})
    state = run_operations(KONU06_LAB.operations_through(4))
    np.testing.assert_allclose(state.frames["ciftler"]["birlesim"], 12.0, rtol=0, atol=1e-12)
    assert state.scalars["r_ay"] == pytest.approx(1.0, abs=1e-12)
    for chosen in (VIF_DEFAULT, ("sqrft", "lsqrft"), ("sqrft", "bdrms", "lotsize", "assess"),
                   ("bdrms", "lotsize", "assess", "lsqrft")):
        table = run_operations(KONU06_LAB.resolve({"adim5_x": chosen}).operations_through(5)).tables["vif"]["deger"]
        exog = sm.add_constant(hprice1[list(chosen)])
        expected = [variance_inflation_factor(exog.to_numpy(), index) for index in range(1, len(chosen) + 1)]
        np.testing.assert_allclose(table.to_numpy(), expected, rtol=1e-10, err_msg=str(chosen))
    table = run_operations(KONU06_LAB.operations_through(5)).tables["vif"]["deger"]
    assert table.round(3).tolist() == [1.419, 1.397, 1.037]


# --- Motor: yeni işlemler ---------------------------------------------------------------------------------

def test_summary_table_summarizes_result_tables_row_by_row() -> None:
    state = LabState()
    state.tables["a"] = pd.DataFrame({"b": [1.0, 2.0, 4.0], "c": [0.0, 1.0, 1.0]})
    state.tables["b"] = pd.DataFrame({"b": [3.0, 3.0], "c": [2.0, 6.0]})
    op = SummaryTable((("A", "a"), ("B", "b")), (("ort", "b", "mean"), ("sd", "c", "std"), ("n", "b", "count")),
                      "ozet", "Deneme", heading="Senaryo", column_decimals=(("sd", 1),))
    execute(op, state)
    table = state.tables["ozet"]
    assert list(table.index) == ["A", "B"] and list(table.columns) == ["ort", "sd", "n"]
    assert table.loc["A", "ort"] == pytest.approx(7 / 3) and table.loc["B", "sd"] == pytest.approx(math.sqrt(8))
    assert table.loc["A", "n"] == 3 and table.loc["B", "n"] == 2
    shown = display_table(op, table)
    assert list(shown.columns) == ["Senaryo", "ort", "sd", "n"]
    assert shown.loc[0, "sd"] == "0,6" and shown.loc[0, "ort"] == "2,333"  # sütun basamağı ve varsayılan basamak
    assert shown.loc[1, "n"] == "2"
    execute(Statistic("a", "b", "mean", "ort_a", "Sonuç tablosundan istatistik"), state)
    assert state.scalars["ort_a"] == pytest.approx(7 / 3)
    with pytest.raises(ValueError, match="en az bir satır"):
        SummaryTable((), (("ort", "b", "mean"),), "x", "x")
    with pytest.raises(ValueError, match="Desteklenmeyen"):
        SummaryTable((("A", "a"),), (("ort", "b", "mod"),), "x", "x")
    with pytest.raises(ValueError, match="column_decimals"):
        SummaryTable((("A", "a"),), (("ort", "b", "mean"),), "x", "x", column_decimals=(("sd", 1),))


def test_residuals_are_the_model_residuals_in_the_app_and_in_both_languages() -> None:
    wage1 = W.load("wage1")
    state = run_operations((LoadWooldridge("wage1", "veri"), OLS("m", "wage1", "wage", ("educ", "exper"), "m"),
                            Residuals("wage1", "artik", "m", "Artık")))
    fit = smf.ols("wage ~ educ + exper", data=wage1).fit()
    np.testing.assert_allclose(state.frames["wage1"]["artik"], fit.resid, rtol=0, atol=1e-12)
    python, r = render_script(KONU05_LAB, "Python"), render_script(KONU05_LAB, "R")
    assert 'wage1["ucret_artik"] = yardimci_ucret.resid' in python
    assert "wage1$ucret_artik <- resid(yardimci_ucret)" in r


def test_copy_frame_gives_an_independent_copy_in_the_app_and_in_both_languages() -> None:
    op = CopyFrame("kopya", "ozgun", "Kopya")
    assert operation_reads(op) == {"ozgun"} and operation_writes(op) == {"kopya"}
    state = run_operations((NewSample("ozgun", 5, 305), Draw("ozgun", "x", "normal", 0, 1, "x"), op,
                            Derive("kopya", "x", E.add(E.var("x"), 1), "x + 1")))
    np.testing.assert_allclose(state.frames["kopya"]["x"], state.frames["ozgun"]["x"] + 1, rtol=0, atol=1e-15)
    spec = COLLINEAR.spec(COLLINEAR.defaults())
    python, r = render_script(spec, "Python"), render_script(spec, "R")
    assert "degisik = orneklem.copy()" in python and python.count("default_rng(305)") == 7  # 6 düzey + tek örneklem
    assert "degisik <- orneklem" in r and r.count("set.seed(305)") == 7
    assert "Her tekrarda bir EKK tahmini yapılır" in python and "Her tekrarda bir EKK tahmini yapılır" in r


def test_regression_table_adds_adjusted_r2_and_term_decimals() -> None:
    hprice1 = W.load("hprice1")
    terms = ("sqrft", "bdrms", "lotsize", INTERCEPT)
    op = RegressionTable((("(1) Fiyat", "m"),), terms, "t", "Deneme", stars=False, decimals=4, standard_errors=False,
                         adj_r2=True, term_decimals=(("lotsize", 6),))
    state = run_operations((LoadWooldridge("hprice1", "veri"),
                            OLS("m", "hprice1", "price", ("sqrft", "bdrms", "lotsize"), "m"), op))
    table = state.tables["t"]
    fit = smf.ols("price ~ sqrft + bdrms + lotsize", data=hprice1).fit()
    assert list(table.index) == [*terms, "n", "r2", "adj_r2"]
    assert table.loc["adj_r2", "(1) Fiyat"] == pytest.approx(fit.rsquared_adj, rel=1e-12)
    shown = regression_display(op, state, lambda name: name).set_index("Değişken")
    assert shown.loc["lotsize", "(1) Fiyat"] == "0,002068" and shown.loc["sqrft", "(1) Fiyat"] == "0,1228"
    assert shown.loc["Düzeltilmiş R²", "(1) Fiyat"] == "0,6607"
    python, r = render_script(KONU05_LAB, "Python"), render_script(KONU05_LAB, "R")
    assert "sh=False, r2_duz=True" in python and "sh = FALSE, r2_duz = TRUE" in r
    with pytest.raises(ValueError, match="Düzeltilmiş R²"):
        RegressionTable((("(1)", "m"),), terms, "t", "t", stars=False, r2=False, adj_r2=True)
    with pytest.raises(ValueError, match="term_decimals"):
        RegressionTable((("(1)", "m"),), terms, "t", "t", term_decimals=(("educ", 3),))


def test_monte_carlo_rejects_the_names_of_the_generated_loop() -> None:
    def body(frame: str) -> tuple:
        return (NewSample(frame, 10, None), Draw(frame, "x", "normal", 0, 1, "x"),
                Statistic(frame, "x", "mean", "m", "ortalama"))

    MonteCarlo("sonuc", 5, 305, body("ornek"), (("m", E.ref("m")),), "geçerli")
    for result, frame in (("sonuc", "tekrar"), ("sonuclar", "ornek")):
        with pytest.raises(ValueError, match="ayrılmış ad"):
            MonteCarlo(result, 5, 305, body(frame), (("m", E.ref("m")),), "ayrılmış ad")


# --- Toplu Monte Carlo ------------------------------------------------------------------------------------

def _all_experiments() -> list:
    import importlib

    found = []
    for number in range(0, 13):
        try:
            module = importlib.import_module(f"core.labs.sezgi_konu{number:02d}")
        except ModuleNotFoundError:
            continue
        found.extend(getattr(module, f"KONU{number:02d}_EXPERIMENTS"))
    return found


def _extremes(experiment) -> list[dict[str, float]]:
    settings = [experiment.defaults()]
    for parameter in experiment.parameters:
        for value in (parameter.minimum, parameter.maximum):
            settings.append(dict(experiment.defaults(), **{parameter.key: value}))
    return settings


def _monte_carlos() -> list[tuple[str, MonteCarlo]]:
    found = []
    for experiment in _all_experiments():
        for parameters in _extremes(experiment):
            for op in experiment.build(parameters):
                if isinstance(op, MonteCarlo) and batchable(op):
                    # Az tekrar yeterli: yol aynıdır; bütün tekrarlarla birebirlik test_all_labs'tadır.
                    found.append((experiment.key, dataclasses.replace(op, reps=min(op.reps, 40))))
    return found


def test_batch_monte_carlo_matches_the_loop_and_leaves_the_generator_in_the_same_state() -> None:
    cases = _monte_carlos()
    assert {key for key, _ in cases} >= {"konu06_sezgi1", "konu06_sezgi3"}
    for key, op in cases:
        with np.errstate(all="ignore"):
            batch, batch_rng = monte_carlo_batch(op)
            loop, loop_rng = monte_carlo_loop(op)
        assert list(batch.columns) == list(loop.columns), key
        np.testing.assert_allclose(batch.to_numpy(float), loop.to_numpy(float), rtol=0, atol=1e-12,
                                   err_msg=f"{key} {op.result}")
        assert batch_rng.bit_generator.state == loop_rng.bit_generator.state, (key, op.result)


def test_konu06_experiments_use_the_batch_path() -> None:
    for experiment in (UNBIASED, COLLINEAR):
        simulations = [op for op in experiment.build(experiment.defaults()) if isinstance(op, MonteCarlo)]
        assert simulations and all(batchable(op) for op in simulations), experiment.key


def test_a_missing_value_sends_the_batch_back_to_the_loop() -> None:
    body = (NewSample("ornek", 6, None), Draw("ornek", "x", "normal", 0, 1, "x"),
            Derive("ornek", "lx", E.log(E.var("x")), "ln x (negatif X'te eksik değer)"),
            Statistic("ornek", "lx", "mean", "m", "ln x ortalaması"))
    op = MonteCarlo("sonuc", 25, 305, body, (("m", E.ref("m")),), "eksik değerli gövde")
    assert batchable(op)
    state = LabState()
    with np.errstate(all="ignore"):
        execute(op, state)
        loop, loop_rng = monte_carlo_loop(op)
    np.testing.assert_allclose(state.tables["sonuc"].to_numpy(float), loop.to_numpy(float), rtol=0, atol=1e-12)
    assert state.rng.bit_generator.state == loop_rng.bit_generator.state
    assert state.tables["sonuc"]["m"].notna().all()  # pandas eksik değeri atlar; toplu yol bunu taklit etmez


# --- Sezgi deneyleri ------------------------------------------------------------------------------------

@pytest.mark.parametrize("experiment", EXPERIMENTS, ids=lambda e: e.key)
def test_experiment_texts_are_correct_at_the_extremes(experiment) -> None:
    for parameters in _extremes(experiment):
        with np.errstate(all="ignore"):
            state = run_operations(experiment.build(parameters))
        metrics = experiment.metrics(state, parameters)
        assert len(metrics) == 4 and all(len(metric.value) <= 10 for metric in metrics), parameters
        assert all("nan" not in metric.value.lower() for metric in metrics), parameters
        assert _clean(experiment.takeaway(state, parameters)), parameters
        assert all(line for line in experiment.dgp(parameters))


def test_experiment_defaults_reproduce_the_notes_tables() -> None:
    state = run_operations(UNBIASED.build(UNBIASED.defaults()))
    table = state.tables["tablo63"]
    expected = ((1.501, 0.001, 0.117), (2.301, 0.801, 0.117))  # Tablo 6.3
    for row, values in zip(table.index, expected):
        for column, value in zip(("ortalama", "yanlilik", "std_sapma"), values):
            assert _printed(table.loc[row, column], value, 3), (row, column)
    assert list(table.index) == ["E(u | X) = 0", "E(u | X) = 0,8X"]
    s = run_operations(OMITTED.build(OMITTED.defaults())).tables["tablo65"]["deger"]
    for value, printed in zip(s, (2.000, 3.000, 0.689, 2.067, 4.067, 4.072, 2.014, 2.987)):  # Tablo 6.5
        assert _printed(value, printed, 3), (value, printed)
    state = run_operations(COLLINEAR.build(COLLINEAR.defaults()))
    table = state.tables["tablo67"]
    expected = (  # Tablo 6.7: ort. korelasyon, üç standart sapma (3 basamak), ort. VIF (1 basamak)
        (0.703, 0.129, 0.092, 0.090, 2.0), (0.893, 0.205, 0.184, 0.090, 5.1), (0.970, 0.379, 0.368, 0.090, 17.3),
        (0.993, 0.772, 0.767, 0.090, 71.7), (0.998, 1.536, 1.534, 0.090, 284.0), (1.000, 3.069, 3.068, 0.090, 1133.3),
    )
    columns = ("ort_korelasyon", "b1_std", "b2_std", "toplam_std", "ort_vif")
    for row, values in zip(table.index, expected):
        for column, value in zip(columns, values):
            assert _printed(table.loc[row, column], value, 1 if column == "ort_vif" else 3), (row, column)
    s = state.scalars
    for name, printed, decimals in (("b1_ozgun", 1.612, 3), ("b2_ozgun", 2.310, 3), ("toplam_ozgun", 3.923, 3),
                                    ("vif_ozgun", 383.5, 1), ("b1_degisik", 1.835, 3), ("b2_degisik", 2.086, 3),
                                    ("toplam_degisik", 3.921, 3), ("vif_degisik", 378.4, 1)):  # Tablo 6.8
        assert _printed(s[name], printed, decimals), name
    assert _printed(state.models["m_ozgun"].rsquared, 0.9343, 4)
    assert _printed(state.models["m_degisik"].rsquared, 0.9341, 4)
    assert _printed(s["r_ozgun"], 0.999, 3)


def test_unbiasedness_experiment_shifts_every_estimate_by_gamma() -> None:
    for gamma in (-1.5, 0.0, 0.8, 1.5):
        state = run_operations(UNBIASED.build(dict(UNBIASED.defaults(), gamma=gamma, n=40)))
        shift = state.tables["ihlal"]["b1"] - state.tables["gecerli"]["b1"]
        np.testing.assert_allclose(shift, gamma, rtol=0, atol=1e-10)  # aynı X ve ε: EKK eğimi tam γ kayar
        assert state.scalars["sd_ihlal"] == pytest.approx(state.scalars["sd_gecerli"], rel=1e-9)
    s = run_operations(UNBIASED.build(dict(UNBIASED.defaults(), n=400))).scalars
    assert s["ort_gecerli"] == pytest.approx(1.5, abs=4 * 0.05 / math.sqrt(3000))  # sd ≈ 1/√400


def test_omitted_variable_experiment_matches_the_formula() -> None:
    for delta in (-1.5, 0.0, 0.7):
        for beta_z in (-4.0, 0.0, 3.0):
            parameters = dict(OMITTED.defaults(), delta=delta, beta_z=beta_z)
            state = run_operations(OMITTED.build(parameters))
            s, frame = state.scalars, state.frames["veri"]
            gamma = state.models["uzun"].params["z"]
            aux = state.models["yardimci"].params["x"]
            # Örneklemde tam: kısa eğim = uzun eğim + uzun Z katsayısı × yardımcı eğim
            assert s["kisa_x"] == pytest.approx(s["uzun_x"] + gamma * aux, abs=1e-10), (delta, beta_z)
            assert s["yardimci_egim"] == pytest.approx(delta, abs=0.05)
            assert s["kisa_x"] == pytest.approx(2 + beta_z * delta, abs=0.1 + 0.03 * abs(beta_z))
            assert len(frame) == 5000


def test_collinearity_experiment_matches_its_formulas() -> None:
    state = run_operations(COLLINEAR.build(dict(COLLINEAR.defaults(), n=200)))
    table = state.tables["tablo67"]
    assert list(table["b1_std"]) == sorted(table["b1_std"])  # bağlantı arttıkça ayrı katsayılar dağılır
    assert list(table["ort_vif"]) == sorted(table["ort_vif"])
    assert table["toplam_std"].max() - table["toplam_std"].min() < 0.01  # toplam aynı bilgiyle tahmin edilir
    for index, sigma in enumerate(LEVELS, start=1):
        levels = state.tables[f"duzey_{index}"]
        np.testing.assert_allclose(levels["vif"], 1 / (1 - levels["r"] ** 2), rtol=1e-8)
        assert levels["r"].mean() == pytest.approx(1 / math.sqrt(1 + sigma ** 2), abs=0.02), sigma
    s = run_operations(COLLINEAR.build(dict(COLLINEAR.defaults(), delta=0.0))).scalars
    assert s["b1_ozgun"] == s["b1_degisik"] and s["vif_ozgun"] == s["vif_degisik"]  # δ = 0: aynı veri


def test_konu05_experiments_match_their_formulas() -> None:
    s = run_operations(COMPARE.build(dict(COMPARE.defaults(), n=1000))).scalars
    assert s["basit_b1"] == pytest.approx(1 + 2 * 0.8, abs=0.2)  # basit eğim ≈ β₁ + β₂·a
    assert s["coklu_b1"] == pytest.approx(1, abs=0.1) and s["coklu_b2"] == pytest.approx(2, abs=0.1)
    for a, beta_2 in ((0.0, 2.0), (0.8, 0.0)):
        s = run_operations(COMPARE.build(dict(COMPARE.defaults(), a=a, beta_2=beta_2, n=1000))).scalars
        assert abs(s["basit_b1"] - s["coklu_b1"]) < 0.15, (a, beta_2)
    for parameters in _extremes(PARTIAL):
        state = run_operations(PARTIAL.build(parameters))
        assert abs(state.scalars["fark"]) < 1e-10, parameters
        frame = state.frames["veri"]
        np.testing.assert_allclose(frame["x1_artik"], state.models["yardimci_x1"].resid, rtol=0, atol=1e-12)
    for parameters in _extremes(FIT):
        state = run_operations(FIT.build(parameters))
        n, k = int(parameters["n"]), int(parameters["k"])
        r2 = [state.scalars[f"r2_{size}"] for size in range(k + 1)]
        assert all(later >= earlier - 1e-12 for earlier, later in zip(r2, r2[1:])), parameters
        for size in range(k + 1):
            expected = 1 - (1 - r2[size]) * (n - 1) / (n - (size + 1) - 1)
            assert state.scalars[f"r2d_{size}"] == pytest.approx(expected, rel=1e-10), (parameters, size)


# --- Kendini sına: yazım çeşitleri ve sayılar -----------------------------------------------------------------

EQUATIONS = {
    ("konu05", "e01"): (("b1 d1 + b2 d2", "\\beta_1 d_1 + \\beta_2 d_2", "β₁d₁ + β₂d₂", "d1*b1 + d2*b2",
                         "beta1 d1 + beta_2 d_2", "b_1 d_1 + b_2 d_2", "B_1 D_1 + B_2 D_2"),
                        ("b1 + b2", "b1 d1", "b0 + b1 d1 + b2 d2", "b1 d1 - b2 d2")),
    ("konu05", "e02"): (("ybar - b1*x1bar - b2*x2bar", "\\bar{Y} - b_1\\bar{X}_1 - b_2\\bar{X}_2", "Ȳ − b₁X̄₁ − b₂X̄₂",
                         "\\bar Y - (b_1 \\bar X_1 + b_2 \\bar X_2)", "ybar - b1 X1bar - b2 X2bar",
                         "Y_bar - b1 X1_bar - b2 X2_bar"),
                        ("ybar + b1 x1bar + b2 x2bar", "ybar - b1 x1bar", "ybar", "ybar - b2 x1bar - b1 x2bar")),
    ("konu05", "e03"): (("x1 - g0 - g2 x2 - g3 x3", "X_1 - (g_0 + g_2 X_2 + g_3 X_3)", "X₁ − g₀ − g₂X₂ − g₃X₃",
                         "x_1 - g_0 - g_2 x_2 - g_3 x_3", "X_1 - G_0 - G_2 X_2 - G_3 X_3"),
                        ("g0 + g2 x2 + g3 x3 - x1", "x1 - g2 x2 - g3 x3", "x1", "x1 - g0")),
    ("konu05", "e04"): (("100*c2*a", "100 c2 a", "100 c_2 a", "a*c2*100", "100 a c₂", "100 C_2 A"),
                        ("c2 a", "100 c1 a", "100 c2", "100(c1 + c2 a)", "100 c2 a + c0")),
    ("konu05", "e05"): (("1 - (1 - R^2)(n-1)/(n-k-1)", "1-(1-R²)\\frac{n-1}{n-k-1}", "(R^2 (n-1) - k)/(n-k-1)",
                         "1 - (1-r2)*(n-1)/(n-k-1)"),
                        ("1 - (1-R^2)(n-k-1)/(n-1)", "R^2 (n-1)/(n-k-1)", "1 - (1 - R^2)(n-1)/(n-k)", "R^2")),
    ("konu06", "e01"): (("gamma(a + b x)", "\\gamma (a + bx)", "γa + γbx", "γ(a+b·x)", "γ(a + b·eğitim)",
                         "gamma*(a + b*egitim)", "\\gamma (a + b \\text{eğitim})", "γ(a + b·Eğitim)"),
                        ("a + b x", "gamma b x", "gamma (a + x)", "gamma a")),
    ("konu06", "e02"): (("b2 rho sz/sx", "\\beta_2 \\rho \\frac{\\sigma_Z}{\\sigma_X}", "β₂ρσ_Z/σ_X",
                         "beta2*rho*s_Z/s_X", "β₂ρσ_z/σ_x", "b2 rho_XZ sz/sx", "β₂ρ_{xz}σ_z/σ_x",
                         "b2 r_{XZ} s_Z/s_X"),
                        ("b2 rho sx/sz", "rho sz/sx", "b2 rho", "b2 sz/sx", "b2 r sz/sx")),
    ("konu06", "e03"): (("b0 + b2 d0", "\\beta_0 + \\beta_2\\delta_0", "β₀ + β₂δ₀", "beta0 + beta2*delta0"),
                        ("b0 + b2 d1", "b0", "b0 + b1 d0", "b0 + b2 d0 + b1")),
    ("konu06", "e04"): (("beta1 + b beta3", "\\beta_1 + b\\beta_3", "β₁ + bβ₃", "b*beta3 + beta1"),
                        ("beta1 + beta3", "beta1 + b", "beta1 + b beta3 + a beta3", "beta1 + c beta3")),
    ("konu06", "e05"): (("tkt/hkt", "TKT_j/HKT_j", "\\frac{\\text{TKT}_j}{\\text{HKT}_j}", "1/(1 - (1 - hkt/tkt))",
                         "\\mathrm{TKT}_j/\\mathrm{HKT}_j", "TKTj/HKTj"),
                        ("hkt/tkt", "1 - hkt/tkt", "tkt/(tkt - hkt)", "1/(1 - hkt/tkt)")),
}
QUIZZES = {"konu05": KONU05_QUIZ, "konu06": KONU06_QUIZ}


@pytest.mark.parametrize("topic, key", sorted(EQUATIONS))
def test_equation_questions_accept_equivalent_forms_and_reject_wrong_ones(topic: str, key: str) -> None:
    question = QUIZZES[topic].question(key)
    accepted, rejected = EQUATIONS[(topic, key)]
    for text in accepted:
        assert grade(question, text).correct, (topic, key, text)
    for text in rejected:
        assert not grade(question, text).correct, (topic, key, text)


def test_symbol_names_ignore_case_and_underscores_only_when_unambiguous() -> None:
    symbols = (Symbol("b1", "b_1", "eğim", 1, 2), Symbol("d1", "d_1", "fark", 1, 2))
    assert parse("B_1 D_1", symbols) == parse("b1*d1", symbols)
    twins = (Symbol("a_b1", "a", "birinci", 1, 2), Symbol("ab_1", "b", "ikinci", 1, 2))
    assert parse("A_B1", twins) == parse("a_b1", twins)  # önce yalnız harf farkı denenir
    with pytest.raises(FormulaError, match="tanımlı bir sembol değil"):
        parse("AB1", twins)  # alt çizgisiz yazım iki sembole de uyar


def test_every_equation_question_is_listed() -> None:
    listed = {key for _, key in EQUATIONS}
    for topic, quiz in QUIZZES.items():
        keys = {q.key for q in quiz.questions if q.kind == "denklem"}
        assert keys <= listed and {(topic, key) for key in keys} <= set(EQUATIONS), topic


BLANKS = {
    ("konu05", "b01"): ((["3,76", "-0,66"], ["3.7609", "−0.6609"]), (["3,76", "0,66"], ["3,10", "-0,66"])),
    ("konu05", "b02"): ((["0,6", "0,3"], ["0.60", "0.3"]), (["0,3", "0,6"], ["0,6", "-0,3"], ["2,4", "0,3"])),
    ("konu05", "b03"): ((["4966", "1014"], ["4966,3", "1014,4"]), (["4966", "1179,7"], ["2194", "1014"])),
    ("konu05", "b04"): ((["0,197", "0,291"], ["0.1971", "0.2911"]), (["0,300", "0,300"], ["0,291", "0,197"])),
    ("konu05", "b05"): ((["8", "45"], ["8,0", "45,00"]), (["1,78", "0,04"], ["8", "22,5"], ["3,56", "45"])),
    ("konu06", "b01"): ((["0,3", "-0,2"], ["0.30", "−0.20"]), (["-0,3", "0,2"], ["0,2", "-0,2"])),
    ("konu06", "b02"): ((["1,001", "0,117"], ["1", "0,117"]), (["2,001", "0,117"], ["1,001", "0,058"])),
    ("konu06", "b03"): ((["-0,133", "-2,133"], ["−0.133", "−2.133"]),
                        (["-0,128", "-2,1"], ["0,133", "2,133"], ["-0,133", "-2,1"])),
    ("konu06", "b04"): ((["-0,0278", "0,5691"], ["−0,0278", "0,5692"]), (["0,0278", "0,5136"], ["-0,1029", "0,6443"])),
    ("konu06", "b05"): ((["10,26", "50,25"], ["10,256", "50,251"]), (["20", "100"], ["10,26", "20"])),
}


@pytest.mark.parametrize("topic, key", sorted(BLANKS))
def test_blank_questions_read_turkish_numbers(topic: str, key: str) -> None:
    question = QUIZZES[topic].question(key)
    accepted, rejected = BLANKS[(topic, key)]
    for response in accepted:
        assert grade(question, response).correct, (topic, key, response)
    for response in rejected:
        assert not grade(question, response).correct, (topic, key, response)


def test_konu05_quiz_numbers_are_computed_not_typed() -> None:
    wage1, hprice1 = W.load("wage1"), W.load("hprice1")
    fit = smf.ols("wage ~ educ + exper + tenure", data=wage1).fit()
    printed = {"Intercept": -2.8727, "educ": 0.5990, "exper": 0.0223, "tenure": 0.1693}
    assert {name: round(value, 4) for name, value in fit.params.items()} == printed
    first = wage1.iloc[0]
    assert (first["educ"], first["exper"], first["tenure"], round(first["wage"], 2)) == (11, 2, 0, 3.10)
    predicted = printed["Intercept"] + printed["educ"] * 11 + printed["exper"] * 2
    assert round(predicted, 2) == 3.76 and round(3.10 - predicted, 2) == -0.66
    assert round(float(fit.predict(wage1.iloc[[0]]).iloc[0]), 2) == 3.76

    def ssr(educ: float) -> float:
        fitted = printed["Intercept"] + educ * wage1["educ"] + printed["exper"] * wage1["exper"] + \
            printed["tenure"] * wage1["tenure"]
        return float(((wage1["wage"] - fitted) ** 2).sum())

    assert round(ssr(0.5990), 1) == 4966.3 and round(ssr(0.62), 1) == 5004.7 and round(ssr(0.62) - ssr(0.5990), 1) == 38.4
    state = run_operations(KONU05_LAB.operations_through(4))
    assert abs(state.models["kismi"].params["Intercept"]) < 1e-10  # artıklar doğrusu orijinden geçer
    # b02: dört gözlemlik örnek tutarlıdır; deneyim (10, 8, 6, 12) iki artık serisine de diktir.
    exper = np.array([10.0, 8.0, 6.0, 12.0])
    educ_resid, wage_resid = np.array([1.0, 1.0, -1.0, -1.0]), np.array([0.9, 0.3, -0.5, -0.7])
    small = pd.DataFrame({"exper": exper, "educ": 12 + educ_resid, "wage": 6 + 0.1 * exper + wage_resid})
    np.testing.assert_allclose(smf.ols("educ ~ exper", data=small).fit().resid, educ_resid, atol=1e-12)
    np.testing.assert_allclose(smf.ols("wage ~ exper", data=small).fit().resid, wage_resid, atol=1e-12)
    slope = float(np.sum(educ_resid * wage_resid) / np.sum(educ_resid ** 2))
    assert round(slope, 1) == 0.6 == round(smf.ols("wage ~ educ + exper", data=small).fit().params["educ"], 12)
    assert round(wage_resid[0] - slope * educ_resid[0], 1) == 0.3
    tkt = float(((wage1["wage"] - wage1["wage"].mean()) ** 2).sum())
    simple = smf.ols("wage ~ educ", data=wage1).fit()
    assert round(tkt, 1) == 7160.4 and round(simple.ssr, 1) == 5980.7 and round(fit.rsquared, 4) == 0.3064
    assert abs(fit.ssr - 4966) <= 1 and abs((simple.ssr - fit.ssr) - 1014) <= 1
    assert abs(7160.4 * (1 - 0.3064) - 4966) <= 1 and abs(5980.7 - 7160.4 * (1 - 0.3064) - 1014) <= 1
    assert round(1 - 0.7 * 39 / 34, 3) == 0.197 and round(1 - 0.7 * 399 / 394, 3) == 0.291
    assert round(1 - 0.98 * 29 / 24, 2) == -0.18
    assert round(1 - (1 - 0.3064) * 525 / 522, 4) == 0.3024 == round(fit.rsquared_adj, 4)
    means = wage1[["wage", "educ", "exper", "tenure"]].mean()
    assert [round(means[name], 3) for name in ("wage", "educ", "exper", "tenure")] == [5.896, 12.563, 17.017, 5.105]
    assert round(5.896 - 0.5990 * 12.563 - 0.0223 * 17.017 - 0.1693 * 5.105, 3) == -2.873
    logs = smf.ols("lwage ~ educ + exper + tenure", data=wage1).fit().params
    assert (round(logs["educ"], 4), round(logs["tenure"], 4)) == (0.0920, 0.0221)
    assert round(30 * 0.0223, 2) == 0.67
    house = smf.ols("price ~ sqrft + bdrms + lotsize", data=hprice1).fit().params
    assert (round(house["sqrft"], 4), round(house["bdrms"], 4)) == (0.1228, 13.8525)
    assert round(300 * 0.1228, 2) == 36.84 and round(13.8525 + 300 * 0.1228, 2) == 50.69
    assert round(100 * 0.1228, 2) == 12.28 and round(house["bdrms"], 2) == 13.85
    assert round(3000 * house["lotsize"], 2) == 6.20 and round(house["lotsize"], 6) == 0.002068
    assert 1.6 * 4.5 / 0.9 == pytest.approx(8, abs=1e-12) and 0.9 * 2 / 0.04 == pytest.approx(45, abs=1e-12)  # b05
    assert round(1.6 / 0.9, 2) == 1.78


def test_konu06_quiz_numbers_are_computed_not_typed() -> None:
    wage1 = W.load("wage1")
    tenure = smf.ols("tenure ~ educ", data=wage1).fit().params["educ"]
    long = smf.ols("wage ~ educ + tenure", data=wage1).fit().params
    assert (round(tenure, 4), round(long["tenure"], 4), round(long["educ"], 4)) == (-0.1466, 0.1896, 0.5691)
    assert round(long["tenure"] * tenure, 4) == -0.0278 and round(0.1896 * -0.1466, 4) == -0.0278
    assert abs(0.5414 - 0.1896 * -0.1466 - 0.5691) <= 0.00015
    assert round(0.0701 * -1.4682, 4) == -0.1029 and round(0.6443 - 0.1029, 4) == 0.5414
    s = run_operations(UNBIASED.build(dict(UNBIASED.defaults(), gamma=-0.5))).scalars
    assert round(s["ort_ihlal"], 3) == 1.001 and round(s["sd_ihlal"], 3) == 0.117
    s = run_operations(UNBIASED.build(UNBIASED.defaults())).scalars
    assert round(s["ort_gecerli"], 3) == 1.501 and round(s["ort_ihlal"], 1) == 2.3  # Tablo 6.3
    assert round(2.014 + 2.987 * 0.689, 3) == 4.072 and round(2 + 3 * 0.689, 3) == 4.067  # d04: Tablo 6.5
    assert _printed(run_operations(OMITTED.build(OMITTED.defaults())).scalars["kisa_x"], 4.072, 3)
    s = run_operations(OMITTED.build(dict(OMITTED.defaults(), delta=-0.7))).scalars  # b03
    assert [round(s[name], 3) for name in ("yardimci_egim", "beklenen_kisa", "beklenen_katki", "kisa_x")] == \
        [-0.711, -0.133, -2.133, -0.128]
    assert round(2 + 3 * -0.711, 3) == -0.133
    assert round(1 / (1 - 0.95 ** 2), 2) == 10.26 and round(1 / (1 - 0.99 ** 2), 2) == 50.25
    assert round(100 / 1.419) == 70  # e05: bağımsız kalan değişkenlik ≈ %70
    s = run_operations(COLLINEAR.build(dict(COLLINEAR.defaults(), sigma=1.0))).scalars  # d06
    assert round(s["r_ozgun"], 1) == 0.7
    assert round(s["b1_ozgun"] - s["b1_degisik"], 3) == 0.002 and round(s["b2_ozgun"] - s["b2_degisik"], 4) == 0.0002
    s = run_operations(COLLINEAR.build(COLLINEAR.defaults())).scalars
    assert round(s["b1_degisik"] - s["b1_ozgun"], 2) == 0.22 and round(s["b2_ozgun"] - s["b2_degisik"], 2) == 0.22
