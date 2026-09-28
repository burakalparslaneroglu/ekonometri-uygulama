"""Konu 0 bloğu: çözümlü örnekler, etkileşimli seçimler, Sezgi deneyleri ve soru yazımları.

Genel sözleşmeler (notlardaki her sayının uygulamada, üretilen Python'da ve R'de tutması) ``test_all_labs.py``'dedir;
bu dosya bloğa özgü olanları denetler:

* notlardan farklı her seçimde uygulamanın sayıları ile üretilen Python kodunun sayıları 1e-12 düzeyinde aynıdır ve
  üretilen R kodu çalışır; seçimin sonraki adımlara geçişi (``LabSpec.resolve``) doğrudur;
* adımlar söyledikleri hesabı yapar (ölçek ve kayma, gösterge kodlaması, iki koşullu ortalama, eğim = kovaryans /
  varyans, R² = r²);
* Sezgi deneyleri kuramsal değerleri verir; metinler kaydırıcıların uçlarında da doğru kalır;
* denklem ve boşluk sorularının kabul ve ret yazımları.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import wooldridge_data as W
from core.codegen.base import generator, render_script
from core.labs import regression as RG
from core.labs.konu00 import KONU00_LAB
from core.labs.runner import run_lab, run_operations
from core.labs.sezgi_konu00 import (
    CONDITIONAL,
    KONU00_EXPERIMENTS,
    LEVELS,
    LINEARITY,
    SAMPLING,
    VAR_SQUARE,
    VAR_X,
    conditional_mean,
    expected_mean,
    population_rho,
    sampled_sd,
    unconditional_mean,
)
from core.labs.spec import OLS, LabSpec, MultiChoice, NumberChoice
from core.quiz.konu00 import KONU00_QUIZ
from core.quiz.model import grade

R_ENVIRONMENT = dict(os.environ, LANG="C.UTF-8", LC_ALL="C.UTF-8")


# --- Etkileşimli adımlar ---------------------------------------------------------------------------------

def _number_values(control: NumberChoice) -> list[float]:
    """Kaydırıcının uçları ve ortası (varsayılandan farklı olanlar)."""

    middle = control.normalize(control.minimum + (control.maximum - control.minimum) / 2)
    values = {control.normalize(control.minimum), control.normalize(control.maximum), middle}
    return sorted(value for value in values if value != control.default)


def _single_changes(spec: LabSpec) -> list[dict[str, object]]:
    changes = []
    for control in spec.controls:
        if isinstance(control, NumberChoice):
            values = _number_values(control)
        elif isinstance(control, MultiChoice):
            values = [(value,) for value, _ in control.options if (value,) != control.default]
        else:
            values = [value for value, _ in control.options if value != control.default]
        changes += [{control.key: value} for value in values]
    return changes


def test_notes_specification_is_the_default_and_passes_every_check() -> None:
    resolved = KONU00_LAB.resolve({})
    assert resolved.variant == () and resolved.steps == KONU00_LAB.steps
    assert run_lab(KONU00_LAB).all_passed
    assert sum(len(step.checks) for step in KONU00_LAB.steps) == 79
    assert [step.note.section for step in KONU00_LAB.steps] == ["0.1", "0.3", "0.4", "0.5", "0.5", "0.6", "0.7",
                                                                  "0.8", "0.9"]


def test_a_changed_choice_marks_its_step_and_every_step_that_uses_it() -> None:
    assert KONU00_LAB.resolve({"adim2_degisken": "ucret"}).variant == (2, 3)
    assert KONU00_LAB.resolve({"adim3_a": 2}).variant == (3,)
    assert KONU00_LAB.resolve({"adim1_gosterge": "erkek"}).variant == (1,)
    assert KONU00_LAB.resolve({"adim9_x": "tenure"}).variant == (9,)
    # Seçim yalnız kendi çıktılarını değiştirir: veri yeniden yüklense de sonraki adımlar notlardaki gibi kalır.
    assert KONU00_LAB.resolve({"adim5_x": "exper"}).variant == (5,)
    assert KONU00_LAB.resolve({"adim7_egitim": "12"}).variant == (7,)
    assert KONU00_LAB.resolve({"adim4_desen": "karesel"}).variant == (4,)
    resolved = KONU00_LAB.resolve({"adim6_x1": 150})
    assert all(not step.checks and not step.controls for step in resolved.steps if step.number == 6)


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


def _compare(state, namespace, label) -> None:
    for name, table in state.tables.items():
        script = namespace[name]
        numeric = [column for column in table.columns if pd.api.types.is_numeric_dtype(table[column])]
        np.testing.assert_allclose(script[numeric].to_numpy(float), table[numeric].to_numpy(float), rtol=0,
                                   atol=1e-12, err_msg=f"{label} {name}")
        # Saklama türünü üretilen kod sütun olarak değil, dtypes (Python) ve str() (R) çıktısıyla gösterir.
        for column in set(table.columns) - set(numeric) - {"saklama"}:
            assert list(script[column].astype(str)) == list(table[column].astype(str)), (label, name, column)
    for name, value in state.scalars.items():
        assert float(namespace[name]) == pytest.approx(value, abs=1e-12), (label, name)
    for name, frame in state.frames.items():
        _frames_equal(frame, namespace[name])
    for name, model in state.models.items():
        np.testing.assert_allclose(namespace[name].params.to_numpy(), model.params.to_numpy(), rtol=0, atol=1e-12)


def test_every_single_choice_gives_the_same_numbers_in_the_app_and_in_python() -> None:
    changes = _single_changes(KONU00_LAB)
    assert len(changes) >= 30
    for change in changes:
        resolved = KONU00_LAB.resolve(change)
        state = run_operations(tuple(op for step in resolved.steps for op in step.operations))
        _compare(state, _run_python(resolved), change)


def test_generated_r_runs_for_the_last_option_of_every_control(tmp_path: Path, rscript: str) -> None:
    last = {}
    for change in _single_changes(KONU00_LAB):
        last[next(iter(change))] = change
    for change in last.values():
        path = tmp_path / "secim.R"
        path.write_text(render_script(KONU00_LAB.resolve(change), "R"), encoding="utf-8")
        result = subprocess.run([rscript, str(path)], cwd=tmp_path, capture_output=True, encoding="utf-8",
                                errors="replace", timeout=300, env=R_ENVIRONMENT)
        assert result.returncode == 0, (change, result.stderr[-1500:])
        assert "warning" not in (result.stdout + result.stderr).lower(), change
        assert "HATA" not in result.stdout, change
        assert sorted(item.name for item in tmp_path.iterdir()) == ["secim.R"], change


def _clean(text: str) -> bool:
    """Metin boş değil, "nan" içermiyor ve eksi sıfır ("−0,00") yazmıyor."""

    return bool(text) and not re.search(r"\bnan\b", text.lower()) and not re.search(r"−0,0+(?![0-9])", text)


def test_changed_steps_explain_the_chosen_specification() -> None:
    for change in [{}] + _single_changes(KONU00_LAB):
        resolved = KONU00_LAB.resolve(change)
        choices = KONU00_LAB.normalize(change)
        for step in KONU00_LAB.steps:
            if step.note_for is None:
                continue
            state = run_operations(resolved.operations_through(step.number))
            assert _clean(step.note_for(state, choices)), (change, step.number)


def test_steps_compute_what_they_say() -> None:
    state = run_operations(KONU00_LAB.resolve({"adim1_gosterge": "erkek"}).operations_through(1))
    assert state.scalars["erkek_payi"] == pytest.approx(0.6) and state.scalars["k"] == 4
    change = {"adim3_a": -2.0, "adim3_c": 5}
    state = run_operations(KONU00_LAB.resolve(change).operations_through(3))
    assert state.scalars["ort_y"] == pytest.approx(-2 * 10 + 5)
    assert state.scalars["std_y"] == pytest.approx(2 * state.scalars["std_B"], rel=1e-12)
    state = run_operations(KONU00_LAB.resolve({"adim2_degisken": "ucret"}).operations_through(3))
    assert state.scalars["toplam"] == pytest.approx(56.0) and state.scalars["ortalama"] == pytest.approx(11.2)
    assert state.scalars["sapma_toplami"] == pytest.approx(0, abs=1e-12)
    wage1 = W.load("wage1")
    for level in ("8", "12", "18"):
        state = run_operations(KONU00_LAB.resolve({"adim7_egitim": level}).operations_through(7))
        subset = wage1.loc[wage1["educ"] == int(level), "wage"]
        assert state.scalars["kosul_n"] == len(subset)
        assert state.scalars["kosullu_ortalama"] == pytest.approx(subset.mean(), rel=1e-12)
    state = run_operations(KONU00_LAB.resolve({}).operations_through(8))
    s = state.scalars
    assert (s["tahmin_tek"] + s["tahmin_cift"]) / 2 == pytest.approx(s["tahmin_tum"], rel=1e-12)
    assert s["tahmin_besinci"] == pytest.approx(wage1["wage"].iloc[::5].mean(), rel=1e-12)
    assert state.frames["wage1"]["besinci"].sum() == 106
    for x in ("educ", "exper", "tenure"):
        state = run_operations(KONU00_LAB.resolve({"adim9_x": x}).operations_through(9))
        slope = RG.coefficient(state.models["model"], x, "coef")
        assert state.scalars["egim_formul"] == pytest.approx(slope, rel=1e-12), x
        assert state.scalars["r_kare"] == pytest.approx(state.scalars["r2"], rel=1e-12), x


def _note(change: dict[str, object], number: int) -> str:
    resolved = KONU00_LAB.resolve(change)
    step = next(item for item in KONU00_LAB.steps if item.number == number)
    return step.note_for(run_operations(resolved.operations_through(number)), KONU00_LAB.normalize(change))


def test_notes_follow_the_chosen_scale_shift_and_rates() -> None:
    text = _note({"adim3_a": -3.0, "adim3_c": -10}, 3)
    assert "y = −3,0·B − 10" in text and "+ −" not in text and "+ -" not in text
    assert "≈ 18,97" in text and "6,3246" in text and "tersine döner" in text
    assert "standart sapma sıfırdır" in _note({"adim3_a": 0.0}, 3)
    assert "küçültür" in _note({"adim3_a": 0.5}, 3)
    assert "y = 1,0·B + 20" in _note({"adim3_c": 20}, 3)
    assert "fark 0 yüzde puandır" in _note({"adim6_oran1": 30}, 6)
    falling = _note({"adim6_oran1": 25}, 6)
    assert "5 yüzde puan düşer" in falling and "%−16,67" in falling
    large = _note({"adim6_x0": 10, "adim6_x1": 300}, 6)
    assert "%2900,00" in large and "%−96,67" in large  # kesirli sayılar binlik ayırıcısız (uygulama ve notlar)
    tiny = _note({"adim6_x0": 200, "adim6_x1": 201}, 6)
    assert "%0,5000" in tiny and "%−0,4975" in tiny  # iki basamakta simetrik görünmesin
    assert "bütün değerler −10 olur" in _note({"adim3_a": 0.0, "adim3_c": -10}, 3)
    assert "0,005" in _note({"adim6_x1": 101}, 6)  # 1,00 − 0,995: "0,00" yazılmaz
    small = _note({"adim7_egitim": "17"}, 7)
    assert "(12 gözlem)" in small and "Örneklemdeki 526 çalışanın" in small
    assert "eğitimi bir yıl daha uzun olan" in _note({}, 9) and "dolar daha yüksektir" in _note({}, 9)


def test_generated_code_names_what_was_compared_and_hides_stars() -> None:
    notes_python = render_script(KONU00_LAB, "Python")
    assert "Konu 00 uygulaması: Veri, Notasyon ve Temel İstatistik" in notes_python
    assert "Bütün değerler ders notlarıyla uyuşuyor." in notes_python
    variant = KONU00_LAB.resolve({"adim9_x": "exper"})
    python, r = render_script(variant, "Python"), render_script(variant, "R")
    assert "Karşılaştırılan adımlarda (Adım 1, 2, 3, 4, 5, 6, 7, 8) bütün değerler" in python
    assert "Karşılaştırılan adımlarda (Adım 1, 2, 3, 4, 5, 6, 7, 8) bütün değerler" in r
    assert "yildiz" not in python and "yildiz" not in r
    assert "signif.stars = FALSE" in r and 'nobs(model)' in r
    assert "Kod 0.1'deki model, seçtiğiniz değişkenle: wage ~ exper" in python
    sampling = render_script(SAMPLING.spec(SAMPLING.defaults()), "Python")
    assert '"yakin": float(' in sampling
    shifted = render_script(KONU00_LAB.resolve({"adim3_a": -2.5, "adim3_c": -10}), "Python")
    assert '-2.5 * iki_veri["B"] - 10' in shifted and "+ -10" not in shifted


def test_log_table_and_percent_formulas() -> None:
    state = run_operations(KONU00_LAB.resolve({}).operations_through(6))
    table = state.frames["log_tablo"]
    np.testing.assert_allclose(table["log_fark"], 100 * np.log(table["x1"] / table["x0"]), rtol=1e-13)
    assert (table["fark"] > 0).all()
    curve = state.frames["egri"]
    assert (curve["tam"] >= curve["log_fark"] - 1e-12).all()  # ln(1 + r) ≤ r


def test_ols_step_is_the_same_as_statsmodels_formula() -> None:
    import statsmodels.formula.api as smf

    frame = W.load("wage1")
    ours = RG.fit_ols(OLS("m", "wage1", "wage", ("educ",), "m"), frame)
    reference = smf.ols("wage ~ educ", data=frame).fit()
    np.testing.assert_allclose(ours.params.to_numpy(), reference.params.to_numpy(), rtol=1e-12)


# --- Sezgi deneyleri ---------------------------------------------------------------------------------------

def _extremes(experiment) -> list[dict[str, float]]:
    settings = [experiment.defaults()]
    for parameter in experiment.parameters:
        for value in (parameter.minimum, parameter.maximum):
            settings.append(dict(experiment.defaults(), **{parameter.key: value}))
    return settings


@pytest.mark.parametrize("experiment", KONU00_EXPERIMENTS, ids=lambda e: e.key)
def test_experiment_texts_are_correct_at_the_extremes(experiment) -> None:
    for parameters in _extremes(experiment):
        state = run_operations(experiment.build(parameters))
        metrics = experiment.metrics(state, parameters)
        assert len(metrics) == 4 and all(len(metric.value) <= 10 for metric in metrics), parameters
        assert all("nan" not in metric.value.lower() for metric in metrics), parameters
        assert _clean(experiment.takeaway(state, parameters)), parameters
        assert all(line for line in experiment.dgp(parameters))


def test_sampling_experiment_matches_its_formulas() -> None:
    for strength in (0.0, 0.5, 1.0):
        parameters = dict(SAMPLING.defaults(), n=100, secim=strength, tekrar=2000)
        table = run_operations(SAMPLING.build(parameters)).tables["tekrarlar"]
        sd_mean = sampled_sd(strength) / math.sqrt(100)
        assert table["xbar"].mean() == pytest.approx(expected_mean(strength), abs=4 * sd_mean / math.sqrt(2000))
        assert table["xbar"].std() == pytest.approx(sd_mean, rel=0.08)
    assert expected_mean(0) == 6 and expected_mean(1) == 7.5
    assert sampled_sd(0) == pytest.approx(math.sqrt(0.5 * (5.0625 + 14.0625) + 0.25 * 9))
    # Yanlılık n'den bağımsızdır: n büyüdükçe yayılım küçülür, merkez kaymış kalır.
    small = run_operations(SAMPLING.build(dict(SAMPLING.defaults(), n=10, secim=1, tekrar=1000))).tables["tekrarlar"]
    large = run_operations(SAMPLING.build(dict(SAMPLING.defaults(), n=500, secim=1, tekrar=1000))).tables["tekrarlar"]
    assert small["xbar"].mean() - 6 == pytest.approx(1.5, abs=0.1)
    assert large["xbar"].mean() - 6 == pytest.approx(1.5, abs=0.02)
    assert large["xbar"].std() < small["xbar"].std() / 5


def test_linearity_experiment_matches_its_formulas() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 5, 400_000)
    assert x.var() == pytest.approx(VAR_X, rel=0.01)
    assert ((x - 2.5) ** 2).var() == pytest.approx(VAR_SQUARE, rel=0.01)
    for beta, gamma, sigma in ((1, 0, 0.5), (0, 1, 0), (1, 1, 1), (-2, 0.5, 3)):
        y = beta * x + gamma * (x - 2.5) ** 2 + rng.normal(0, sigma, x.size)
        assert np.corrcoef(x, y)[0, 1] == pytest.approx(population_rho(beta, gamma, sigma), abs=0.005)
    assert population_rho(0, 0, 0) is None
    base = dict(LINEARITY.defaults(), beta=0.5, gamma=1.0, sigma=1.0)
    one = run_operations(LINEARITY.build(dict(base, olcek=1))).scalars
    hundred = run_operations(LINEARITY.build(dict(base, olcek=100))).scalars
    assert hundred["r"] == pytest.approx(one["r"], rel=1e-12)
    assert hundred["kovaryans"] == pytest.approx(100 * one["kovaryans"], rel=1e-12)
    constant = dict(LINEARITY.defaults(), beta=0, gamma=0, sigma=0)
    state = run_operations(LINEARITY.build(constant))
    assert "r" not in state.scalars and state.scalars["kovaryans"] == 0
    assert "tanımsız" in LINEARITY.takeaway(state, constant)


def test_conditional_experiment_matches_its_formulas() -> None:
    assert conditional_mean(16, 0) == 9 and conditional_mean(13, 0.2) == 7.5
    assert unconditional_mean(0) == pytest.approx(7.5) and unconditional_mean(0.12) == pytest.approx(7.5 + 1.4)
    parameters = dict(CONDITIONAL.defaults(), n=2000, sigma=0.5)
    frame = run_operations(CONDITIONAL.build(parameters)).frames["duzeyler"]
    assert frame["fark"].abs().max() < 0.1 and frame["gozlem"].sum() == 2000
    # Kaydırıcının her n değerinde her eğitim düzeyinde en az bir gözlem vardır (X yalnız n'ye bağlıdır).
    n_values = range(60, 2001, 20)
    for n in n_values:
        frame = run_operations(CONDITIONAL.build(dict(CONDITIONAL.defaults(), n=n))).frames["duzeyler"]
        assert (frame["gozlem"] > 0).all() and list(frame["x"]) == list(LEVELS), n


# --- Kendini sına: yazım çeşitleri --------------------------------------------------------------------------

EQUATIONS = {
    "e01": (("a*xbar + c", "a x̄ + c", "c + a·x̄", "a*x_bar+c", "a\\bar{x} + c", "a\\overline{x}+c", "a x̅ + c",
             "\\bar{y} = a\\bar{x} + c", "a\\bar{X} + c", "a\\overline{X}+c"),
            ("a*xbar", "xbar + c", "a*(xbar + c)", "sqrt(a^2)*xbar + c")),
    "e02": (("sxy/(sx*sy)", "s_xy/(s_x s_y)", "sxy/sx/sy", "s_{xy}/(s_x*s_y)", "S_xy/(S_x*S_y)", "syx/(sx*sy)",
             "\\frac{s_{xy}}{s_x s_y}", "r = sxy/(sx*sy)"), ("sxy/sx*sy", "sxy/(sx+sy)", "sxy")),
    "e03": (("100*(x1-x0)/x0", "100(x_1 - x_0)/x_0", "(x1/x0 - 1)*100", "100*(x₁−x₀)/x₀"),
            ("100*(x1-x0)/x1", "(x1-x0)/x0", "100*(x0-x1)/x0")),
    "e04": (("100*(ln(x1) - ln(x0))", "100 ln(x1/x0)", "100*log(x_1/x_0)", "100[ln(x1)-ln(x0)]",
             "100\\left(\\ln(x_1)-\\ln(x_0)\\right)", "%Δx ≈ 100*ln(x1/x0)", "100*(ln x1 - ln x0)",
             "100(\\ln x_1 - \\ln x_0)", "\\%\\Delta x \\approx 100\\ln(x_1/x_0)"),
            ("ln(x1-x0)*100", "100*(x1-x0)/x0", "ln(x1)-ln(x0)", "100*ln x1/x0", "ln x1 - ln x0")),
    "e05": (("sxy/sx^2", "s_xy/s_x²", "sxy/(sx*sx)", "s_{xy}/s_x^2", "sxy/s^2_x", "\\dfrac{S_{xy}}{S_x^{2}}"),
            ("sxy/sx", "sxy^2/sx", "sx^2/sxy")),
}


@pytest.mark.parametrize("key", sorted(EQUATIONS))
def test_equation_questions_accept_equivalent_forms_and_reject_wrong_ones(key: str) -> None:
    question = KONU00_QUIZ.question(key)
    accepted, rejected = EQUATIONS[key]
    for text in accepted:
        assert grade(question, text).correct, (key, text)
    for text in rejected:
        assert not grade(question, text).correct, (key, text)


def test_blank_questions_read_turkish_numbers() -> None:
    assert grade(KONU00_QUIZ.question("b04"), ["250", "3.000"]).correct
    assert grade(KONU00_QUIZ.question("b04"), ["250,0", "3000"]).correct
    assert not grade(KONU00_QUIZ.question("b04"), ["250", "300"]).correct
    assert grade(KONU00_QUIZ.question("b03"), ["25", "22,31"]).correct
    assert grade(KONU00_QUIZ.question("b03"), ["25", "22.31"]).correct
    assert not grade(KONU00_QUIZ.question("b03"), ["25", "22,4"]).correct
    assert grade(KONU00_QUIZ.question("b05"), ["0,25", "0,6"]).correct
    assert not grade(KONU00_QUIZ.question("b02"), ["2,67", "1,63"]).correct  # payda n: yanlış


def test_quiz_numbers_are_computed_not_typed() -> None:
    assert 100 * math.log(100 / 80) == pytest.approx(22.31, abs=0.005)
    assert np.var([4, 6, 8], ddof=1) == 4
    assert 2.5 * 100 * 12 == pytest.approx(3000)
    assert sum([12, 16, 19, 23]) / 4 == 17.5
    wage1 = W.load("wage1")
    group = wage1.loc[wage1["educ"] == 16, "wage"]
    assert len(group) == 68 and group.min() == pytest.approx(3.00) and group.max() == pytest.approx(22.86, abs=0.005)
    assert np.cov(wage1["wage"], wage1["educ"], ddof=1)[0, 1] / wage1["educ"].var() == pytest.approx(0.5414, abs=5e-5)
    assert wage1.loc[wage1["educ"] == 16, "wage"].mean() == pytest.approx(8.04, abs=0.005)
    assert 1.25 * 0.8 == pytest.approx(1.0)
