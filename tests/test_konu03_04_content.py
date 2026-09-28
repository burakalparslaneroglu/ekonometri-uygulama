"""Konu 3–4 bloğu: etkileşimli spesifikasyonlar, Sezgi deneyleri, soru yazımları ve motorun yeni seçenekleri.

Genel sözleşmeler (notlardaki her sayının uygulamada, üretilen Python'da ve R'de tutması) ``test_all_labs.py``'dedir;
bu dosya bloğa özgü olanları denetler:

* notlardan farklı her seçimde uygulamanın sayıları ile üretilen Python kodunun sayıları 1e-12 düzeyinde aynıdır ve
  üretilen R kodu çalışır; seçimin sonraki adımlara geçişi (``LabSpec.resolve``) doğrudur, veriyi yeniden yükleyen
  adım notlardaki gibi kalır;
* adımlar söyledikleri hesabı yapar (koşullu ortalama, küçük örnekte EKK, sıfır–bir değişken, birim dönüşümü, dört
  biçim, makale tablosu);
* Sezgi deneyleri kuramsal değerleri verir; metinler kaydırıcıların uçlarında da doğru kalır;
* denklem ve boşluk sorularının kabul ve ret yazımları; sorulardaki sayılar veriden hesaplanır;
* saçılım grafiğinde düzey ortalamaları ve standart hatasız / R²'siz makale tablosu.
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
import statsmodels.formula.api as smf

from core import wooldridge_data as W
from core.codegen.base import generator, render_script
from core.labs import regression as RG
from core.labs.konu03 import KONU03_LAB
from core.labs.konu04 import KONU04_LAB
from core.labs.runner import run_lab, run_operations
from core.labs.sezgi_konu03 import ERRORS, KONU03_EXPERIMENTS, LINES, SD_X, SEARCH, lines_axis
from core.labs.sezgi_konu04 import FORMS, KONU04_EXPERIMENTS, NOISE, SPREAD, VAR_UNIFORM_10, population_r2
from core.labs.spec import OLS, LabSpec, MultiChoice, NumberChoice, RegressionTable, ScatterPlot
from core.quiz.konu03 import KONU03_QUIZ
from core.quiz.konu04 import KONU04_QUIZ
from core.quiz.model import grade

R_ENVIRONMENT = dict(os.environ, LANG="C.UTF-8", LC_ALL="C.UTF-8")
LABS = (KONU03_LAB, KONU04_LAB)


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
    assert sum(len(step.checks) for step in KONU03_LAB.steps) == 80
    assert sum(len(step.checks) for step in KONU04_LAB.steps) == 74
    assert [step.note.section for step in KONU03_LAB.steps] == ["3.1", "3.3", "3.9", "3.10", "3.11", "3.11",
                                                                  "3.12", "3.13"]
    assert [step.note.section for step in KONU04_LAB.steps] == ["4.1", "4.2", "4.3", "4.4", "4.5", "4.7", "4.9",
                                                                  "4.10", "4.11", "4.12"]


def test_a_changed_choice_marks_its_step_and_every_step_that_uses_it() -> None:
    assert KONU03_LAB.resolve({"adim4_x": "exper"}).variant == (4, 5, 7)
    assert KONU03_LAB.resolve({"adim7_x0": 16, "adim7_y0": 8.75}).variant == (7,)
    # Adım 8 WAGE1'i yeniden yükler: Adım 1'deki seçim onu değiştirmez, denetimleri kalır.
    resolved = KONU04_LAB.resolve({"adim1_x": "tenure"})
    assert resolved.variant == (1, 2, 4, 5)
    assert resolved.step(8).checks == KONU04_LAB.step(8).checks
    for spec, shared in ((KONU03_LAB, "adim4_x"), (KONU04_LAB, "adim1_x")):
        for change in _single_changes(spec):
            (key,) = change
            if key == shared:
                continue
            own = next(step.number for step in spec.steps if any(c.key == key for c in step.controls))
            assert spec.resolve(change).variant == (own,), change


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
    assert len(changes) >= 15
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


def test_konu03_steps_compute_what_they_say() -> None:
    wage1, jtrain2 = W.load("wage1"), W.load("jtrain2")
    state = run_operations(KONU03_LAB.resolve({"adim1_x": "tenure"}).operations_through(1))
    means = wage1.groupby("tenure")["wage"].mean()
    np.testing.assert_allclose(state.tables["ortalamalar"]["mean"].to_numpy(), means.to_numpy(), rtol=1e-12)
    for level in (8, 17):
        state = run_operations(KONU03_LAB.resolve({"adim2_x": level}).operations_through(2))
        for name, value in (("ort_x", level), ("ort_x1", level + 1)):
            subset = wage1.loc[wage1["educ"] == value, "wage"]
            assert state.scalars[name] == pytest.approx(subset.mean(), rel=1e-12) and len(subset) > 0
    for score in (40, 88, 100):
        s = run_operations(KONU03_LAB.resolve({"adim3_y5": score}).operations_through(3)).scalars
        slope = 2.9 + 4 * (score - 78) / 40  # (X₅ − X̄)(Y₅ − 78)/Σ(Xᵢ − X̄)²
        assert s["b1_kucuk"] == pytest.approx(slope, abs=1e-12)
        assert s["b0_kucuk"] == pytest.approx((330 - 78 + score) / 5 - slope * 6, abs=1e-12)
        assert s["artik_toplami"] == pytest.approx(0, abs=1e-10)
    for d in ("black", "nodegree"):
        s = run_operations(KONU03_LAB.resolve({"adim6_d": d}).operations_through(6)).scalars
        groups = jtrain2.groupby(d)["re78"].mean()
        assert s["sabit_j"] == pytest.approx(groups[0], rel=1e-12)
        assert s["egim_j"] == pytest.approx(groups[1] - groups[0], rel=1e-10)
    s = run_operations(KONU03_LAB.resolve({"adim7_x0": 16, "adim7_y0": 8.75}).operations_through(7)).scalars
    assert s["tahmin_x0"] == pytest.approx(-0.9049 + 0.5414 * 16, abs=1e-12)
    assert s["artik_y0"] == pytest.approx(8.75 - (-0.9049 + 0.5414 * 16), abs=1e-12)
    s = run_operations(KONU03_LAB.resolve({"adim4_x": "tenure"}).operations_through(5)).scalars
    assert s["x_sifir"] == (wage1["tenure"] == 0).sum()


def test_konu04_steps_compute_what_they_say() -> None:
    hprice1 = W.load("hprice1")
    s = run_operations(KONU04_LAB.resolve({"adim1_x": "exper"}).operations_through(5)).scalars
    assert s["artik_toplami"] == pytest.approx(0, abs=1e-9)
    assert s["ort_tahmin"] == pytest.approx(s["ort_ucret"], rel=1e-12)
    assert s["dogru_xbar"] == pytest.approx(s["ort_ucret"], rel=1e-12)
    assert s["tkt"] == pytest.approx(s["mkt"] + s["hkt"], rel=1e-12)
    assert s["r2_mkt"] == pytest.approx(s["r2_hkt"], rel=1e-10) == pytest.approx(s["r_kare"], rel=1e-10)
    s = run_operations(KONU04_LAB.resolve({"adim3_ybar": 20.0, "adim3_yhat": 0.0, "adim3_y": 12.5})
                       .operations_through(3)).scalars
    assert (s["toplam_sapma"], s["model_sapma"], s["artik_sapma"]) == pytest.approx((-7.5, -20, 12.5))
    assert s["iki_bilesen"] == pytest.approx(s["toplam_sapma"])
    base = run_operations(KONU04_LAB.operations_through(6)).scalars
    for change, slope_factor, intercept_factor in (({"adim6_y": "dolar"}, 1000, 1000), ({"adim6_x": "yuz"}, 100, 1),
                                                   ({"adim6_y": "dolar", "adim6_x": "yuz"}, 100_000, 1000)):
        s = run_operations(KONU04_LAB.resolve(change).operations_through(6)).scalars
        assert s["egim_birim"] == pytest.approx(slope_factor * base["egim_birim"], rel=1e-10), change
        assert s["sabit_birim"] == pytest.approx(intercept_factor * base["sabit_birim"], rel=1e-10), change
        assert s["r2_birim"] == pytest.approx(base["r2_birim"], rel=1e-10), change
        assert s["bir_fit_dolar"] == pytest.approx(base["bir_fit_dolar"], rel=1e-10), change
    s = run_operations(KONU04_LAB.operations_through(7)).scalars
    for name, formula in (("egim_dd", "price ~ sqrft"), ("egim_ld", "lprice ~ sqrft"),
                          ("egim_dl", "price ~ lsqrft"), ("egim_ll", "lprice ~ lsqrft")):
        assert s[name] == pytest.approx(smf.ols(formula, data=hprice1).fit().params.iloc[1], rel=1e-10), name
    for x in ("exper", "tenure"):
        s = run_operations(KONU04_LAB.resolve({"adim8_x": x}).operations_through(8)).scalars
        assert s["b1_log"] == pytest.approx(smf.ols(f"lwage ~ {x}", data=W.load("wage1")).fit().params[x], rel=1e-10)
        assert s["yuzde_log"] == pytest.approx(100 * s["b1_log"], rel=1e-12)
    state = run_operations(KONU04_LAB.resolve({"adim9_sutunlar": ("log_log", "duzey_log")}).operations_through(9))
    table = state.tables["makale"]
    assert list(table.columns) == ["(1) Fiyat", "(2) ln(Fiyat)"]  # seçenek sırası
    assert list(table.index) == ["lsqrft", "Intercept", "n", "r2"]


def test_notes_follow_the_chosen_specification() -> None:
    def note(spec, change, number):
        resolved = spec.resolve(change)
        return spec.step(number).note_for(run_operations(resolved.operations_through(number)), spec.normalize(change))

    assert "Beşinci öğrencinin notu: 88 (notlarda 78)" in note(KONU03_LAB, {"adim3_y5": 88}, 3)
    assert "3,9" in note(KONU03_LAB, {"adim3_y5": 88}, 3)
    assert "nedensel etki olarak okunamaz" in note(KONU03_LAB, {"adim6_d": "married"}, 6)
    assert "rastgele atandığı için" in note(KONU03_LAB, {}, 6)
    assert "sıfır olan çalışan yok" in note(KONU03_LAB, {"adim4_x": "exper"}, 5)
    assert "sıfır olan 163 çalışan var" in note(KONU03_LAB, {"adim4_x": "tenure"}, 5)
    text = note(KONU04_LAB, {"adim8_x": "tenure"}, 8)
    assert "log–düzey model" in text and "karşılaştırılmaz" in text


def test_generated_code_hides_standard_errors_and_r2_where_the_notes_do() -> None:
    python = render_script(KONU03_LAB.resolve({"adim4_x": "exper"}), "Python")
    r = render_script(KONU03_LAB.resolve({"adim4_x": "exper"}), "R")
    assert "sh=False, r2=False" in python and "sh = FALSE, r2 = FALSE" in r
    notes_python, notes_r = render_script(KONU04_LAB, "Python"), render_script(KONU04_LAB, "R")
    assert "sh=False" in notes_python and "sh = FALSE" in notes_r
    assert "r2=False" not in notes_python and "r2 = FALSE" not in notes_r


# --- Motor: düzey ortalamaları ve seçenekli makale tablosu ------------------------------------------------

def test_scatter_plot_shows_level_means_and_the_table_hides_rows() -> None:
    frame = pd.DataFrame({"x": [1.0, 1.0, 2.0, 2.0, 2.0, 3.0], "y": [1.0, 3.0, 2.0, 4.0, 6.0, 5.0]})
    from core.labs.runner import LabState, execute, plot_key

    state = LabState()
    state.frames["veri"] = frame
    op = ScatterPlot("veri", "x", "y", "X", "Y", "Deneme", fit_line=True, means="Düzey ortalamaları")
    execute(op, state)
    means = state.plots[plot_key(op)]["ortalamalar"]
    assert list(means["x"]) == [1.0, 2.0, 3.0] and list(means["y"]) == [2.0, 4.0, 5.0]
    execute(OLS("m1", "veri", "y", ("x",), "m1"), state)
    table_op = RegressionTable((("(1)", "m1"),), ("x", "Intercept"), "tablo", "Deneme", stars=False,
                               standard_errors=False, r2=False)
    execute(table_op, state)
    assert list(state.tables["tablo"].index) == ["x", "Intercept", "n"]
    full = RegressionTable((("(1)", "m1"),), ("x", "Intercept"), "tam", "Deneme")
    execute(full, state)
    assert list(state.tables["tam"].index) == ["x", "x_sh", "Intercept", "Intercept_sh", "n", "r2"]
    with pytest.raises(ValueError, match="Yıldızlar"):
        RegressionTable((("(1)", "m1"),), ("x",), "t", "t", stars=True, standard_errors=False)


# --- Sezgi deneyleri ---------------------------------------------------------------------------------------

def _extremes(experiment) -> list[dict[str, float]]:
    settings = [experiment.defaults()]
    for parameter in experiment.parameters:
        for value in (parameter.minimum, parameter.maximum):
            settings.append(dict(experiment.defaults(), **{parameter.key: value}))
    return settings


@pytest.mark.parametrize("experiment", KONU03_EXPERIMENTS + KONU04_EXPERIMENTS, ids=lambda e: e.key)
def test_experiment_texts_are_correct_at_the_extremes(experiment) -> None:
    for parameters in _extremes(experiment):
        with np.errstate(all="ignore"):
            state = run_operations(experiment.build(parameters))
        metrics = experiment.metrics(state, parameters)
        assert len(metrics) == 4 and all(len(metric.value) <= 10 for metric in metrics), parameters
        assert all("nan" not in metric.value.lower() for metric in metrics), parameters
        assert _clean(experiment.takeaway(state, parameters)), parameters
        assert all(line for line in experiment.dgp(parameters))


def test_lines_experiment_matches_its_formulas() -> None:
    exact = run_operations(LINES.build(dict(LINES.defaults(), sigma=0.0))).tables["tekrarlar"]
    np.testing.assert_allclose(exact["b1"], 0.5, atol=1e-12)
    table = run_operations(LINES.build(dict(LINES.defaults(), n=100, sigma=3.0))).tables["tekrarlar"]
    # Eğim tahminlerinin standart sapması yaklaşık σ / (√n · s_X); 500 tekrarda ortalama 0,5 çevresinde.
    assert table["b1"].mean() == pytest.approx(0.5, abs=4 * 3 / (10 * SD_X) / math.sqrt(500))
    assert table["b1"].std() == pytest.approx(3 / (10 * SD_X), rel=0.12)
    narrow = run_operations(LINES.build(dict(LINES.defaults(), n=400, sigma=3.0))).tables["tekrarlar"]
    assert narrow["b1"].std() < table["b1"].std() / 1.6
    for sigma in np.arange(0.5, 6.01, 0.5):  # eksen en küçük n'de de bütün tahminleri gösterir
        low, high = lines_axis(float(sigma))
        wide = run_operations(LINES.build(dict(LINES.defaults(), n=10, sigma=float(sigma)))).tables["tekrarlar"]
        assert ((wide["b1"] >= low) & (wide["b1"] <= high)).all(), sigma


def test_search_experiment_matches_its_formulas() -> None:
    for slope in (-0.5, 0.0, 0.8, 2.0):
        for shift in (-3.0, 0.0, 1.5):
            s = run_operations(SEARCH.build(dict(SEARCH.defaults(), egim=slope, kayma=shift))).scalars
            assert s["aday_hkt"] >= s["ekk_hkt"] - 1e-9, (slope, shift)
            assert s["aday_toplam"] == pytest.approx(-20 * shift, abs=1e-9)  # Σ(artık) = −n·d
    best = run_operations(SEARCH.build(SEARCH.defaults())).scalars
    on_ols = run_operations(SEARCH.build(dict(SEARCH.defaults(), egim=round(best["b1"], 2), kayma=0.0))).scalars
    assert on_ols["aday_hkt"] - on_ols["ekk_hkt"] < 0.01 * on_ols["ekk_hkt"]
    text = SEARCH.takeaway(run_operations(SEARCH.build(dict(SEARCH.defaults(), egim=round(best["b1"], 2)))),
                           dict(SEARCH.defaults(), egim=round(best["b1"], 2)))
    assert "tek doğrudur" in text


def test_errors_experiment_matches_its_formulas() -> None:
    parameters = dict(ERRORS.defaults(), n=50, sigma=2.0)
    state = run_operations(ERRORS.build(parameters))
    frame, s = state.frames["orneklem"], state.scalars
    assert s["toplam_artik"] == pytest.approx(0, abs=1e-9)
    expected = (s["b0"] - (-1.0)) + (s["b1"] - 0.5) * frame["x"]  # u − û = (β̂₀ − β₀) + (β̂₁ − β₁)·X
    np.testing.assert_allclose(frame["fark"], expected, atol=1e-10)
    large = run_operations(ERRORS.build(dict(ERRORS.defaults(), n=300, sigma=2.0))).scalars
    small = run_operations(ERRORS.build(dict(ERRORS.defaults(), n=5, sigma=2.0))).scalars
    assert large["r_u_artik"] > 0.98 and large["r_u_artik"] > small["r_u_artik"]


def test_noise_and_spread_experiments_match_the_population_r2() -> None:
    assert population_r2(0.5, VAR_UNIFORM_10, 2.0) == pytest.approx(0.25 * 100 / 12 / (0.25 * 100 / 12 + 4))
    assert population_r2(0.5, VAR_UNIFORM_10, 0.0) == 1.0 and math.isnan(population_r2(0.0, 1.0, 0.0))
    exact = run_operations(NOISE.build(dict(NOISE.defaults(), sigma=0.0))).scalars
    assert exact["r2"] == pytest.approx(1.0) and exact["b1"] == pytest.approx(0.5, abs=1e-12)
    for sigma in (1.0, 2.0, 5.0):
        s = run_operations(NOISE.build(dict(NOISE.defaults(), sigma=sigma, n=500))).scalars
        assert s["r2"] == pytest.approx(population_r2(0.5, VAR_UNIFORM_10, sigma), abs=0.06), sigma
    r2 = [run_operations(SPREAD.build(dict(SPREAD.defaults(), genislik=w))).scalars["r2"] for w in (0.5, 5, 10, 20)]
    assert r2 == sorted(r2) and r2[0] < 0.05
    assert r2[-1] == pytest.approx(population_r2(0.5, 20 ** 2 / 12, 2.0), abs=0.06)
    assert population_r2(0.5, 20 ** 2 / 12, 2.0) == pytest.approx(100 / 12 / (100 / 12 + 4))


def test_forms_experiment_recovers_the_elasticity() -> None:
    for elasticity in (0.4, 0.8, 1.4):
        s = run_operations(FORMS.build(dict(FORMS.defaults(), esneklik=elasticity, n=500))).scalars
        assert s["b_log"] == pytest.approx(elasticity, abs=0.08), elasticity
    state = run_operations(FORMS.build(FORMS.defaults()))
    frame = state.frames["konutlar"]
    np.testing.assert_allclose(np.log(frame["fiyat"]), frame["ln_fiyat"], rtol=1e-12)
    assert abs(frame["ln_buyukluk"].mean() - math.log(2000)) < 0.1


def test_generated_texts_stay_true_across_the_sliders() -> None:
    small = KONU03_LAB.step(3).note_for
    for score in (40, 60, 100):
        state = run_operations(KONU03_LAB.resolve({"adim3_y5": score}).operations_through(3))
        text = small(state, KONU03_LAB.normalize({"adim3_y5": score}))
        assert "X = 4 noktası çevresinde döner" in text and "+ −" not in text, score
        assert state.frames["kucuk_ornek"].loc[1, "tahmin"] == pytest.approx(60.2, abs=1e-9)  # dönme noktası
    for elasticity in (0.2, 1.0):
        for n in (30, 36):
            parameters = dict(FORMS.defaults(), esneklik=elasticity, n=n)
            text = FORMS.takeaway(run_operations(FORMS.build(parameters)), parameters)
            assert "yüzde −" not in text and "değişen etki" not in text, (elasticity, n)
    noisy = dict(NOISE.defaults(), sigma=10.0, n=20)
    text = NOISE.takeaway(run_operations(NOISE.build(noisy)), noisy)
    assert "0,5 çevresinde kalır" not in text and "anakütle R²'si düşer" in text
    for n in (5, 90, 300):
        parameters = dict(ERRORS.defaults(), n=n, sigma=0.5)
        text = ERRORS.takeaway(run_operations(ERRORS.build(parameters)), parameters)
        assert "n büyüdükçe" not in text and "−0 " not in text and not text.endswith("−0"), n
    for slope in (0.65, 0.7):
        parameters = dict(SEARCH.defaults(), egim=slope, kayma=0.0)
        text = SEARCH.takeaway(run_operations(SEARCH.build(parameters)), parameters)
        assert "hangi yöne ayrılırsanız ayrılın" in text
        assert ("eğim 0,65 ve kayma 0" in text) == (slope == 0.7), slope


# --- Kendini sına: yazım çeşitleri ve sayılar -----------------------------------------------------------------

EQUATIONS = {
    ("konu03", "e01"): (("r*sy/sx", "r s_Y / s_X", "r\\frac{s_Y}{s_X}", "r_{XY} s_y/s_x", "sy*r/sx"),
                        ("r*sx/sy", "sy/sx", "r*sy*sx")),
    ("konu03", "e02"): (("ybar - sxy/sxx*xbar", "\\bar{Y} - \\frac{S_{XY}}{S_{XX}}\\bar{X}", "Ȳ - (S_XY/S_XX) X̄",
                         "ȳ - sxy xbar/sxx"),
                        ("ybar - sxx/sxy*xbar", "ybar + sxy/sxx*xbar", "sxy/sxx")),
    ("konu03", "e03"): (("sxy/syy", "S_{XY}/S_{YY}", "\\frac{S_{XY}}{S_{YY}}"), ("sxx/sxy", "sxy/sxx", "syy/sxy")),
    ("konu03", "e04"): (("m - b0 - b1 x", "m - (\\beta_0 + \\beta_1 x)", "m − β₀ − β₁x", "-b1*x + m - beta0"),
                        ("b0 + b1 x - m", "m - b0", "m - b1 x", "m")),
    ("konu03", "e05"): (("(b0 - bh0) + (b1 - bh1)x", "\\beta_0 - \\hat{\\beta}_0 + (\\beta_1 - \\hat{\\beta}_1) X_i",
                         "β₀ − β̂₀ + (β₁ − β̂₁)X_i", "b0 - b0hat + (b1 - b1hat)*x",
                         "\\beta_0 - \\widehat{\\beta}_0 + (\\beta_1 - \\widehat{\\beta}_1) X_i",
                         "b0 - bh0 + (b1 - bh1) Xi"),
                        ("(bh0 - b0) + (bh1 - b1)x", "b0 - bh0", "(b1 - bh1)x")),
    ("konu04", "e01"): (("(q - b0)/b1", "\\frac{q - b_0}{b_1}", "q/b1 - b0/b1", "(\\bar{Y} - b_0)/b_1"),
                        ("(q + b0)/b1", "q/b1", "(q - b1)/b0")),
    ("konu04", "e02"): (("b1 (x - xbar)", "\\hat{\\beta}_1 (X_i - \\bar{X})", "β̂₁(X_i − X̄)", "\\beta_1(X_i-\\bar X)",
                         "b1 x - b1 xbar", "b1(Xi - Xbar)", "b_1 (X_i - \\bar X)"),
                        ("b1 x", "x - xbar", "b1 (xbar - x)")),
    ("konu04", "e03"): (("tkt (1 - r^2)", "TKT(1 - r_{XY}^2)", "\\text{TKT}(1-r^2)", "TKT - TKT r²"),
                        ("tkt r^2", "tkt (1 - r)", "1 - r^2")),
    ("konu04", "e04"): (("c b0", "c*b_0", "b0 c"), ("c b0/k", "b0", "c b0 k", "c b1/k")),
    ("konu04", "e05"): (("b p", "b*p", "p·b"), ("100 b p", "b p/100", "b")),
}
QUIZZES = {"konu03": KONU03_QUIZ, "konu04": KONU04_QUIZ}


@pytest.mark.parametrize("topic, key", sorted(EQUATIONS))
def test_equation_questions_accept_equivalent_forms_and_reject_wrong_ones(topic: str, key: str) -> None:
    question = QUIZZES[topic].question(key)
    accepted, rejected = EQUATIONS[(topic, key)]
    for text in accepted:
        assert grade(question, text).correct, (topic, key, text)
    for text in rejected:
        assert not grade(question, text).correct, (topic, key, text)


def test_blank_questions_read_turkish_numbers_and_names() -> None:
    assert grade(KONU03_QUIZ.question("b02"), ["-0,70", "0.9"]).correct
    assert not grade(KONU03_QUIZ.question("b02"), ["0,70", "−0,90"]).correct
    assert grade(KONU03_QUIZ.question("b04"), ["Salary", "ROE"]).correct
    assert not grade(KONU03_QUIZ.question("b04"), ["roe", "salary"]).correct
    assert grade(KONU04_QUIZ.question("b04"), ["4,5113", "54.1359"]).correct
    assert grade(KONU04_QUIZ.question("b04"), ["4,51", "54,12"]).correct  # 12 × 4,51: istenen yuvarlamayla
    assert not grade(KONU04_QUIZ.question("b04"), ["4,5", "54"]).correct
    assert grade(KONU04_QUIZ.question("b05"), ["1.5767", "16,55"]).correct
    assert not grade(KONU04_QUIZ.question("b05"), ["1,6", "18,0"]).correct  # tam yüzde (Konu 9) beklenmez
    assert grade(KONU04_QUIZ.question("b03"), ["338", "336,40"]).correct


def test_quiz_numbers_are_computed_not_typed() -> None:
    wage1, hprice1 = W.load("wage1"), W.load("hprice1")
    level = smf.ols("wage ~ educ", data=wage1).fit().params
    at16 = level["Intercept"] + 16 * level["educ"]
    assert {6.00, 8.75} <= set(wage1.loc[wage1["educ"] == 16, "wage"].round(2))
    assert round(at16, 2) == 7.76 and round(8.75 - at16, 2) == 0.99 and round(6.00 - at16, 2) == -1.76
    x, y = wage1["educ"], wage1["wage"]
    sxy, sxx, syy = ((x - x.mean()) * (y - y.mean())).sum(), ((x - x.mean()) ** 2).sum(), ((y - y.mean()) ** 2).sum()
    assert round(sxy / syy, 3) == 0.304 and round(sxx / sxy, 3) == 1.847
    assert round(x.corr(y) * y.std() / x.std(), 3) == 0.541
    assert round(x.mean(), 3) == 12.563 and (x == 0).sum() == 2
    scores = np.array([55, 60, 65, 72, 88.0])
    hours = np.array([2, 4, 6, 8, 10.0])
    slope = np.polyfit(hours, scores, 1)
    assert slope == pytest.approx([3.9, 44.6])
    small = np.array([55, 60, 65, 72, 78.0])
    assert ((small - small.mean()) ** 2).sum() == 338 and 2.9 ** 2 * 40 == pytest.approx(336.4)
    logs = smf.ols("lwage ~ educ", data=wage1).fit().params
    assert round(logs["Intercept"] + 12 * logs["educ"], 2) == 1.58 and round(200 * logs["educ"], 2) == 16.55
    assert round(0.5838 + 0.0827 * 12, 2) == 1.58 and round(200 * 0.0827, 2) == 16.54
    assert round(level["educ"] * 100 / 12, 2) == 4.51 and round(level["educ"] * 100, 2) == 54.14
    four = {f: smf.ols(f, data=hprice1).fit() for f in ("price ~ sqrft", "lprice ~ sqrft", "price ~ lsqrft",
                                                         "lprice ~ lsqrft")}
    assert [round(m.params.iloc[1], 4) for m in four.values()] == [0.1402, 0.0004, 297.9113, 0.8727]
    assert round(four["lprice ~ sqrft"].rsquared, 4) == 0.5837 and round(four["lprice ~ lsqrft"].rsquared, 4) == 0.5530
    assert round(100 * math.log(1.05), 2) == 4.88 and round(100 * math.log(1.5), 2) == 40.55
    assert sum(r ** 2 for r in (1.2, -0.4, 0.9, -2.1, 0.4)) == pytest.approx(6.98)
    residuals = small - (48 + 3 * hours)  # Ŷ = 48 + 3X üç noktadan geçer ama EKK'den kötüdür
    assert list(residuals) == [1, 0, -1, 0, 0] and (residuals ** 2).sum() == 2 > 1.6
    fitted = level["Intercept"] + level["educ"] * wage1["educ"]
    worse = ((wage1["wage"] - fitted) ** 2 > (wage1["wage"] - wage1["wage"].mean()) ** 2).sum()
    assert worse == 220
    assert round((5.896 + 0.9049) / 0.5414, 2) == 12.56
    assert round(7160.414 * (1 - 0.4059 ** 2), 1) == 5980.7
    ceo = W.load("ceosal1") if "ceosal1" in W.DATASETS else __import__("wooldridge").data("ceosal1")
    assert round(smf.ols("lsalary ~ lsales", data=ceo).fit().params["lsales"], 4) == 0.2567
    assert {"salary", "roe"} <= set(ceo.columns)
    assert RG.stars(0.5) == ""
