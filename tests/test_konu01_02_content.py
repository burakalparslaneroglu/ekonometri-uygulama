"""Konu 1–2 bloğu: veri katmanı, en küçük kareler, etkileşimli spesifikasyon, Sezgi deneyleri ve arayüz biçimi.

Genel sözleşmeler (notlardaki her sayının uygulamada, üretilen Python'da ve R'de tutması) ``test_all_labs.py``'dedir;
bu dosya bloğa özgü olanları denetler:

* ``wooldridge`` Python ve R paketleri aynı veriyi verir (satır, sütun, sütun toplamları);
* en küçük kareler sayıları açık formüllerle (numpy) aynıdır;
* notlardan farklı her seçimde uygulamanın sayıları ile üretilen Python kodunun sayıları 1e-12 düzeyinde aynıdır ve
  üretilen R kodu çalışır; seçimin sonraki adımlara geçişi (``LabSpec.resolve``) doğrudur;
* Sezgi deneyleri kuramsal değerleri verir; metinler kaydırıcıların uçlarında da doğru kalır;
* denklem sorularının kabul ve ret yazımları.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core import wooldridge_data as W
from core.codegen.base import generator, render_script
from core.labs import regression as RG
from core.labs.konu01 import KONU01_LAB
from core.labs.konu02 import KONU02_LAB
from core.labs.runner import LabState, execute, run_lab, run_operations
from core.labs.sezgi_konu01 import KONU01_EXPERIMENTS
from core.labs.sezgi_konu02 import KONU02_EXPERIMENTS
from core.labs.spec import (
    OLS,
    Choice,
    Derive,
    LabSpec,
    LabStep,
    LoadWooldridge,
    MultiChoice,
    NoteRef,
    NumberChoice,
    Shape,
    interactive_step,
)
from core.quiz.konu01 import KONU01_QUIZ
from core.quiz.konu02 import KONU02_QUIZ
from core.quiz.model import grade

R_ENVIRONMENT = dict(os.environ, LANG="C.UTF-8", LC_ALL="C.UTF-8")
LABS = (KONU01_LAB, KONU02_LAB)
EXPERIMENTS = KONU01_EXPERIMENTS + KONU02_EXPERIMENTS


# --- Veri katmanı ----------------------------------------------------------------------------------

DERIVED = {"cps78_85": {"wage"}}
"""Katalogda olup veri setinde olmayan, uygulamada türetilen değişkenler (CPS78_85: wage = exp(lwage))."""


def test_every_catalogued_variable_exists_and_every_column_has_a_label() -> None:
    for name, dataset in W.DATASETS.items():
        frame = W.load(name)
        assert set(dataset.variables) - DERIVED.get(name, set()) <= set(frame.columns), name
        assert set(frame.columns) <= set(dataset.variables), (name, sorted(set(frame.columns) - set(dataset.variables)))
    assert len(W.load("wage1")) == 526 and len(W.load("wagepan")) == 4360


def test_load_returns_an_independent_copy() -> None:
    first = W.load("jtrain2", ("train", "re78"))
    first.loc[0, "re78"] = -1.0
    assert W.load("jtrain2", ("train", "re78")).loc[0, "re78"] != -1.0
    with pytest.raises(KeyError, match="sütun yok"):
        W.load("wage1", ("wage", "yok"))


def test_python_and_r_packages_give_the_same_data(tmp_path: Path, rscript: str) -> None:
    names = sorted(W.DATASETS)
    lines = ['suppressMessages(library(wooldridge))']
    for name in names:
        lines += [
            f'data("{name}", package = "wooldridge")',
            f'd <- {name}; s <- sapply(d, function(x) sum(as.numeric(x), na.rm = TRUE))',
            f'cat("{name}", nrow(d), ncol(d), paste(names(s), sprintf("%.10f", s), sep = "=", collapse = ";"), '
            'sep = "|"); cat("\\n")',
        ]
    path = tmp_path / "veri.R"
    path.write_text("\n".join(lines), encoding="utf-8")
    result = subprocess.run([rscript, str(path)], capture_output=True, encoding="utf-8", timeout=300,
                            env=R_ENVIRONMENT)
    assert result.returncode == 0, result.stderr[-1500:]
    for line in result.stdout.strip().splitlines():
        name, rows, columns, sums = line.split("|")
        frame = W.load(name)
        assert (len(frame), frame.shape[1]) == (int(rows), int(columns)), name
        r_sums = dict(item.split("=") for item in sums.split(";"))
        assert list(r_sums) == list(frame.columns), name
        for column, value in r_sums.items():
            assert float(value) == pytest.approx(float(frame[column].sum()), rel=1e-12, abs=1e-8), (name, column)


# --- En küçük kareler: açık formüllerle karşılaştırma ------------------------------------------------------

@pytest.mark.parametrize("regressors", [("educ",), ("educ", "exper", "tenure"), ("female", "married")])
def test_ols_numbers_match_the_textbook_formulas(regressors: tuple[str, ...]) -> None:
    frame = W.load("wage1")
    result = RG.fit_ols(OLS("m", "wage1", "wage", regressors, "m"), frame)
    y = frame["wage"].to_numpy(float)
    x = np.column_stack([np.ones(len(frame)), frame[list(regressors)].to_numpy(float)])
    n, k = x.shape
    beta = np.linalg.solve(x.T @ x, x.T @ y)
    residuals = y - x @ beta
    sigma2 = residuals @ residuals / (n - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(x.T @ x)))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), n - k)
    critical = stats.t.ppf(0.975, n - k)
    r2 = 1 - residuals @ residuals / np.sum((y - y.mean()) ** 2)
    f = (r2 / (k - 1)) / ((1 - r2) / (n - k))
    table = RG.coefficient_table(result)
    np.testing.assert_allclose(table["coef"], beta, rtol=1e-10)
    np.testing.assert_allclose(table["se"], se, rtol=1e-10)
    np.testing.assert_allclose(table["t"], t, rtol=1e-9)
    np.testing.assert_allclose(table["p"], p, rtol=1e-8, atol=1e-300)
    np.testing.assert_allclose(table["ci_low"], beta - critical * se, rtol=1e-10)
    np.testing.assert_allclose(table["ci_high"], beta + critical * se, rtol=1e-10)
    assert RG.model_quantity(result, "r2") == pytest.approx(r2, rel=1e-12)
    assert RG.model_quantity(result, "f") == pytest.approx(f, rel=1e-10)
    assert RG.model_quantity(result, "nobs") == n and RG.model_quantity(result, "df_resid") == n - k


def test_ols_rejects_perfect_collinearity_and_missing_variables() -> None:
    frame = W.load("wage1").assign(educ2=lambda d: 2 * d["educ"])
    with pytest.raises(ValueError, match="doğrusal bağlantı"):
        RG.fit_ols(OLS("m", "wage1", "wage", ("educ", "educ2"), "m"), frame)
    with pytest.raises(ValueError, match="Veride olmayan"):
        RG.fit_ols(OLS("m", "wage1", "wage", ("yok",), "m"), frame)
    with pytest.raises(ValueError, match="en az bir"):
        OLS("m", "wage1", "wage", (), "m")


def test_stars_use_the_thresholds_printed_under_the_table() -> None:
    assert [RG.stars(p) for p in (0.0099, 0.01, 0.0499, 0.05, 0.0999, 0.10)] == ["***", "**", "**", "*", "*", ""]


# --- Etkileşimli spesifikasyon -----------------------------------------------------------------------

def _single_changes(spec: LabSpec) -> list[dict[str, object]]:
    """Her denetim için varsayılandan farklı her seçenek (çoklu seçimde tek öğe ve varsayılan eksi bir öğe)."""

    changes = []
    for control in spec.controls:
        if isinstance(control, MultiChoice):
            values = [(value,) for value, _ in control.options]
            values += [tuple(item for item in control.default if item != removed) for removed in control.default]
            values = [value for value in values if value and control.normalize(value) != control.default]
        else:
            values = [value for value, _ in control.options if value != control.default]
        changes += [{control.key: value} for value in values]
    return changes


def test_notes_specification_is_the_default_and_passes_every_check() -> None:
    for spec in LABS:
        resolved = spec.resolve({})
        assert resolved.variant == () and resolved.steps == spec.steps
        assert run_lab(spec).all_passed
    assert sum(len(step.checks) for step in KONU01_LAB.steps) == 49
    assert sum(len(step.checks) for step in KONU02_LAB.steps) == 50


def test_a_changed_choice_marks_its_step_and_every_step_that_uses_it() -> None:
    assert KONU01_LAB.resolve({"adim4_x": "tenure"}).variant == (4, 5)
    assert KONU01_LAB.resolve({"adim3_x": "exper"}).variant == (3,)
    assert KONU01_LAB.resolve({"adim2_degiskenler": ("wage",)}).variant == (2,)
    for change in _single_changes(KONU02_LAB):
        (key,) = change
        own = next(step.number for step in KONU02_LAB.steps if any(c.key == key for c in step.controls))
        assert KONU02_LAB.resolve(change).variant == (own,), change
    resolved = KONU01_LAB.resolve({"adim4_x": "exper"})
    assert all(not step.checks and not step.controls for step in resolved.steps if step.number in (4, 5))


def test_invalid_choices_are_reported_in_turkish() -> None:
    with pytest.raises(ValueError, match="geçersiz seçim"):
        KONU01_LAB.resolve({"adim4_x": "wage"})
    with pytest.raises(ValueError, match="en az 1"):
        KONU01_LAB.resolve({"adim2_degiskenler": ()})
    with pytest.raises(ValueError, match="Tanımsız denetim"):
        KONU01_LAB.resolve({"yok": 1})
    assert KONU02_LAB.normalize({"adim2_seriler": ["unem", "inf"]})["adim2_seriler"] == ("inf", "unem")
    number = NumberChoice("x0", "x₀", 0, 1, 0.5, 0.1)
    assert number.normalize(0.30000000000000004) == 0.3 and NumberChoice("n", "n", 1, 9, 5, 1, integer=True).normalize(
        4.6) == 5
    with pytest.raises(ValueError, match="aralığında"):
        number.normalize(2)


def test_step_definitions_must_agree_with_their_builders() -> None:
    choice = Choice("c", "Seçim", (("a", "A"), ("b", "B")), "a")

    def build(choices):
        return (LoadWooldridge("wage1", "veri"), Shape("wage1", "n", "k"))

    step = interactive_step(number=1, title="t", note=NoteRef("1.1"), explanation="t", controls=(choice,), build=build)
    assert step.operations == build({"c": "a"})
    with pytest.raises(ValueError, match="varsayılan seçimlerin"):
        LabStep(1, "t", NoteRef("1.1"), "t", operations=(), controls=(choice,), build=build)
    with pytest.raises(ValueError, match="build'i olmalıdır"):
        LabStep(1, "t", NoteRef("1.1"), "t", controls=(choice,))
    with pytest.raises(ValueError, match="tekil"):
        Choice("c", "Seçim", (("a", "A"), ("a", "B")), "a")
    later = LabStep(2, "t", NoteRef("1.1"), "t", operations=build({"c": "a"}), uses=(choice,), build=build)
    with pytest.raises(ValueError, match="önceki bir adımda"):
        LabSpec("konu01", "t", "1", (later,))


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
    return namespace


@pytest.mark.parametrize("spec", LABS, ids=lambda s: s.topic_key)
def test_every_single_choice_gives_the_same_numbers_in_the_app_and_in_python(spec: LabSpec) -> None:
    for change in _single_changes(spec):
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
def test_generated_r_runs_for_the_last_option_of_every_control(spec: LabSpec, tmp_path: Path, rscript: str) -> None:
    last = {}
    for change in _single_changes(spec):
        last[next(iter(change))] = change
    for change in last.values():
        path = tmp_path / "secim.R"
        path.write_text(render_script(spec.resolve(change), "R"), encoding="utf-8")
        result = subprocess.run([rscript, str(path)], cwd=tmp_path, capture_output=True, encoding="utf-8",
                                errors="replace", timeout=300, env=R_ENVIRONMENT)
        assert result.returncode == 0, (change, result.stderr[-1500:])
        assert "warning" not in (result.stdout + result.stderr).lower(), change
        assert "HATA" not in result.stdout, change
        assert sorted(item.name for item in tmp_path.iterdir()) == ["secim.R"], change


def test_changed_steps_explain_the_chosen_specification() -> None:
    for spec in LABS:
        for change in _single_changes(spec):
            resolved = spec.resolve(change)
            choices = spec.normalize(change)
            for number in resolved.variant:
                step = spec.step(number)
                if step.note_for is None:
                    continue
                state = run_operations(resolved.operations_through(number))
                text = step.note_for(state, choices)
                assert text and "nan" not in text.lower() and "−0," not in text.replace("−0,0", ""), (change, text)


def test_konu02_variants_compute_what_they_say() -> None:
    state = run_operations(KONU02_LAB.resolve({"adim2_sira": "unem"}).operations_through(2))
    phillips = state.frames["phillips"]
    assert phillips["unem"].is_monotonic_increasing and list(phillips["sira"]) == list(range(1, 57))
    assert list(phillips["year"].head(3)) == [1953, 1952, 1951]
    state = run_operations(KONU02_LAB.resolve({"adim3_gruplama": "year_female"}).operations_through(3))
    table = state.tables["donem_ozeti"]
    assert table["count"].sum() == 1084 and table.loc[(85, 1), "count"] == 245
    state = run_operations(KONU02_LAB.resolve({"adim5_degisken": "unem78"}).operations_through(5))
    assert state.scalars["fark"] == pytest.approx(0.24324324 - 0.35384615, abs=1e-8)


# --- Sezgi deneyleri -----------------------------------------------------------------------------

def _state(experiment, **parameters) -> LabState:
    values = experiment.defaults() | parameters
    state = LabState()
    for op in experiment.build(values):
        execute(op, state)
    return state


def test_konu01_experiment1_error_term_and_known_line() -> None:
    coffee = KONU01_EXPERIMENTS[0]
    exact = _state(coffee, olcek=0.0)
    assert exact.scalars["b1"] == pytest.approx(13.5, abs=1e-9) and exact.scalars["r2"] == pytest.approx(1.0)
    assert exact.scalars["b0"] == pytest.approx(3100, abs=1e-6)
    noisy = _state(coffee, olcek=1.0, n=365)
    frame = noisy.frames["gunler"]
    u = 560 * frame["hafta_sonu"] + 380 * frame["hava"] + frame["e"]
    np.testing.assert_allclose(frame["satis"], 3100 + 13.5 * frame["reklam"] + u, rtol=0, atol=1e-9)
    assert noisy.scalars["sabit_gercek"] == 3260  # 3100 + 560 · 2/7
    assert frame["hafta_sonu"].mean() == pytest.approx(2 / 7, abs=0.05)


def test_konu01_experiment1_degenerate_and_zero_slope_settings() -> None:
    """k = 0 ve β₁ = 0: satış sabittir, R² hesaplanmaz (uygulama ve üretilen kod). β₁ = 0: R² k'dan bağımsızdır."""

    coffee = KONU01_EXPERIMENTS[0]
    flat = coffee.defaults() | {"olcek": 0.0, "beta1": 0.0}
    state = _state(coffee, olcek=0.0, beta1=0.0)
    assert "r2" not in state.scalars
    assert dict((m.label, m.value) for m in coffee.metrics(state, flat))["R²"] == "tanımsız"
    for language in ("Python", "R"):
        code = generator(coffee.spec(flat), language).script()
        assert "R²" not in code and "r.squared" not in code and "rsquared" not in code, language
    r2 = [_state(coffee, olcek=k, beta1=0.0).scalars["r2"] for k in (0.25, 1.0, 3.0)]
    assert r2[0] == pytest.approx(r2[1], abs=1e-12) and r2[1] == pytest.approx(r2[2], abs=1e-12)
    text = coffee.takeaway(_state(coffee, beta1=0.0), coffee.defaults() | {"beta1": 0.0})
    assert "k'dan etkilenmez" in text


def test_konu01_experiment2_text_separates_the_shift_from_sampling_noise() -> None:
    from core.labs.sezgi_konu01 import expected_shift, sampling_dominates

    questions = KONU01_EXPERIMENTS[1]
    assert expected_shift(0.5) == pytest.approx(7.5)
    default = _state(questions)
    text = questions.takeaway(default, questions.defaults())
    assert "15·ρ = 7,50 TL/TL" in text
    noisy = questions.defaults() | {"rho": 0.1, "n": 50}
    state = _state(questions, rho=0.1, n=50)
    assert sampling_dominates(0.1, 13.5, state.scalars["b1"])
    assert "örneklem çekilişi baskındır" in questions.takeaway(state, noisy)


def test_konu01_experiment2_slope_tends_to_the_conditional_mean_not_the_causal_effect() -> None:
    questions = KONU01_EXPERIMENTS[1]
    big = _state(questions, rho=0.5, n=730)
    assert big.scalars["b1"] == pytest.approx(13.5 + 15 * 0.5, abs=1.5)
    assert big.scalars["nedensel_cevap"] == 135 and big.scalars["kosullu_ortalama"] == pytest.approx(
        3100 + 13.5 * 160 + 600 * 0.5)
    independent = _state(questions, rho=0.0, n=730)
    assert independent.scalars["b1"] == pytest.approx(13.5, abs=1.5)


def test_konu01_experiment3_true_mean_at_zero_education() -> None:
    intercept = KONU01_EXPERIMENTS[2]
    state = _state(intercept)
    assert state.scalars["gercek_sifir"] == pytest.approx(math.exp(0.4 + 0.45 ** 2 / 2))
    assert state.scalars["en_dusuk"] == 8 and state.frames["orneklem"]["educ"].max() == 18
    assert state.scalars["b0"] < 0 < state.scalars["gercek_sifir"]
    # Metnin WAGE1 cümlesi veriye dayanır: eğitimi 0 olan iki çalışan, ortalama eğitim 12,56 yıl.
    wage1 = W.load("wage1")
    assert (wage1["educ"] == 0).sum() == 2 and round(wage1["educ"].mean(), 2) == 12.56
    assert "yalnız iki çalışan" in intercept.takeaway(state, intercept.defaults())


def test_konu02_experiment1_random_assignment_versus_self_selection() -> None:
    assignment = KONU02_EXPERIMENTS[0]
    state = _state(assignment, n=3000, secilim=1.0, tau=1.8)
    s = state.scalars
    share, threshold = 0.4, stats.norm.ppf(0.6)
    # Gönüllü tasarımda seçilim farkı: 2 · (s/√(s²+1)) · φ(z)/(p(1 − p)), z = Φ⁻¹(0,6).
    expected_bias = 2 * (1 / math.sqrt(2)) * stats.norm.pdf(threshold) / (share * (1 - share))
    assert s["fark_g"] == pytest.approx(1.8 + expected_bias, abs=0.6)
    assert s["fark_r"] == pytest.approx(1.8, abs=0.6) and abs(s["fark_mr"]) < 0.15
    assert s["pay_r"] == pytest.approx(0.4, abs=0.03) and s["pay_g"] == pytest.approx(0.4, abs=0.03)


def test_konu02_experiment2_within_weather_comparisons_remove_the_confounder() -> None:
    confounder = KONU02_EXPERIMENTS[1]
    s = _state(confounder, n=2000, bag=0.6, tau=100).scalars
    assert s["fark_tum"] == pytest.approx(100 + 400 * 0.6, abs=30)
    assert s["fark_sicak"] == pytest.approx(100, abs=25) and s["fark_soguk"] == pytest.approx(100, abs=25)


def test_konu02_experiment3_common_trend_creates_correlation() -> None:
    trend = KONU02_EXPERIMENTS[2]
    assert _state(trend, egilim=2.0, T=100).scalars["r"] > 0.95
    assert abs(_state(trend, egilim=2.0, T=100).scalars["r_sapma"]) < 0.3
    frame = _state(trend, egilim=1.0).frames["yillar"]
    np.testing.assert_allclose(frame["A_sapma"], 4 * frame["eps"], atol=1e-9)
    weak = _state(trend, egilim=0.1, T=30)
    assert weak.scalars["r"] < 0
    assert "şansa bağlıdır" in trend.takeaway(weak, trend.defaults() | {"egilim": 0.1, "T": 30})


@pytest.mark.parametrize("experiment", EXPERIMENTS, ids=lambda e: e.key)
def test_experiment_texts_hold_at_the_slider_ends(experiment) -> None:
    for item in experiment.parameters:
        for value in (item.minimum, item.maximum):
            parameters = experiment.defaults() | {item.key: value}
            with np.errstate(all="ignore"):
                state = LabState()
                for op in experiment.build(parameters):
                    execute(op, state)
            text = experiment.takeaway(state, parameters)
            metrics = experiment.metrics(state, parameters)
            assert text and "nan" not in text.lower(), (experiment.key, item.key, value)
            assert all(len(metric.value) <= 12 for metric in metrics), (experiment.key, [m.value for m in metrics])
            assert all(line for line in experiment.dgp(parameters))


# --- Kendini sına: denklem yazımları ------------------------------------------------------------

@pytest.mark.parametrize("quiz, key, accepted, rejected", [
    (KONU01_QUIZ, "e01", ("((y1-p1)+(y2-p2))/2", "(y1+y2-p1-p2)/2", "(y_1 - p_1 + y_2 - p_2)/2",
                          "0,5(y1-p1+y2-p2)", "(y₁–p₁+y₂–p₂)/2"),
     ("((p1-y1)+(p2-y2))/2", "(y1-p1)+(y2-p2)", "(y1+y2)/2")),
    (KONU01_QUIZ, "e02", ("m*x+c", "c+mx", "c + m x", "c+m.x"), ("c*m*x", "c+m")),
    (KONU01_QUIZ, "e03", ("d*m", "m d"), ("m+d", "m/d")),
    (KONU01_QUIZ, "e04", ("h*m", "m·h"), ("m+h", "m/h")),
    (KONU01_QUIZ, "e05", ("y-(a+b*x)", "y - bx - a"), ("y-a+b*x", "a+b*x-y")),
    (KONU02_QUIZ, "e01", ("exp(l)", "EXP(l)", "e^l", "e**l", "e^(L)", "E^l", "e^{l}"), ("log(l)", "l", "e*l")),
    (KONU02_QUIZ, "e02", ("(n_1*m_1+n_2*m_2)/(n_1+n_2)", "(n1 m1 + n2 m2)/(n1+n2)", "(n1m1+n2m2)/(n1+n2)",
                          "(m1n1+m2n2)/(n1+n2)", "(n₁m₁+n₂m₂)/(n₁+n₂)", "(N1*M1+N2*M2)/(N1+N2)",
                          "(n1m1+n2m2):(n1+n2)"),
     ("(m1+m2)/2", "(n1m1+n2m2)/2")),
    (KONU02_QUIZ, "e03", ("100p1-100p0", "100*(p_1-p_0)"), ("p1-p0", "100*(p0-p1)")),
    (KONU02_QUIZ, "e04", ("b-a+1", "1+b-a"), ("b-a", "b-a-1")),
    (KONU02_QUIZ, "e05", ("N*T", "NT", "n*t", "T*N"), ("N+T", "N/T")),
])
def test_equation_answers_accept_equivalent_forms(quiz, key, accepted, rejected) -> None:
    question = quiz.question(key)
    assert all(grade(question, text).correct for text in accepted), key
    assert not any(grade(question, text).correct for text in rejected), key


def test_blank_grading_reads_turkish_numbers_and_spellings() -> None:
    from core.quiz.model import normalize_text, number_readings

    b02 = KONU02_QUIZ.question("b02")
    assert grade(b02, ["1.084", "49,26"]).correct and grade(b02, ["1084", "49.26"]).correct
    assert not grade(b02, ["1,084", "49,26"]).correct
    assert number_readings("4.360") == [4.36, 4360.0] and number_readings("1.084,5") == [1084.5]
    b04 = KONU02_QUIZ.question("b04")
    assert grade(b04, ["karistirici", "Ters nedensellik"]).correct
    assert grade(b04, ["Karıştırıcı faktor", "ters-nedensellik"]).correct
    assert normalize_text("Havuzlanmış yatay kesit") == normalize_text("havuzlanmis yatay-kesit")
    b01 = KONU02_QUIZ.question("b01")
    assert grade(b01, ["6,6", "–2,0"]).correct  # uzun tire eksi olarak okunur
    b02_k1 = KONU01_QUIZ.question("b02")  # tolerans basılı basamağa göre ve iki yönde simetrik
    assert grade(b02_k1, ["9,17", "36,72"]).correct and grade(b02_k1, ["9,19", "36,74"]).correct
    assert not grade(b02_k1, ["9,20", "36,73"]).correct and not grade(b02_k1, ["9,18", "36,80"]).correct


# --- Arayüz biçimi ------------------------------------------------------------------------------

def test_turkish_numbers_never_show_a_negative_zero() -> None:
    from core.charts import tr_number
    from core.labs.sezgi import plain

    assert tr_number(-0.0004, 3) == "0,000" and plain(-1e-15, 2) == "0,00"
    assert tr_number(-0.0006, 3) == "−0,001" and tr_number(0.25, 1, percent=True) == "%0,2"


def test_panel_and_group_tables_are_displayed_in_turkish_format() -> None:
    from topics.lab_ui import display_table

    state = run_operations(KONU02_LAB.operations_through(4))
    panel_op = KONU02_LAB.step(4).operations[-1]
    shown = display_table(panel_op, state.tables["panel"], KONU02_LAB.label)
    assert dict(zip(shown["Büyüklük"], shown["Değer"]))["İlk dönem"] == "1980"
    assert dict(zip(shown["Büyüklük"], shown["Değer"]))["Gözlem sayısı (satır)"] == "4.360"
    resolved = KONU02_LAB.resolve({"adim3_gruplama": "year_female"})
    state = run_operations(resolved.operations_through(3))
    group_op = resolved.step(3).operations[-1]
    shown = display_table(group_op, state.tables["donem_ozeti"], KONU02_LAB.label)
    assert list(shown.iloc[:, 0]) == ["78 · 0", "78 · 1", "85 · 0", "85 · 1"]
    assert list(shown["Ortalama"]) == ["6,83", "4,79", "9,99", "7,88"]


def test_a_check_with_a_numeric_row_is_written_correctly_in_both_languages() -> None:
    python, r = render_script(KONU02_LAB, "Python"), render_script(KONU02_LAB, "R")
    assert 'donem_ozeti.loc[78, "count"]' in python and 'donem_ozeti["78", "count"]' in r
    assert 'grup_ozeti.loc[0, "mean"]' in python and 'grup_ozeti["0", "mean"]' in r
    assert 'cps78_85["wage"] = np.exp(cps78_85["lwage"])' in python and "cps78_85$wage <- exp(cps78_85$lwage)" in r


def test_derived_wage_is_defined_before_it_is_summarised() -> None:
    operations = KONU02_LAB.step(3).operations
    kinds = [type(op).__name__ for op in operations]
    assert kinds.index("Derive") < kinds.index("GroupStats")
    assert isinstance(operations[1], Derive) and operations[1].name == "wage"
