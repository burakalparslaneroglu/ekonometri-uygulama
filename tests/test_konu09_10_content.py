"""Konu 9–10 bloğu: ölçekleme, logaritmik ve karesel biçimler, model seçimi; kukla değişkenler. Etkileşimli adımlar,
motorun yeni işlemleri, Sezgi deneyleri ve sorular.

Genel sözleşmeler (notlardaki her sayının uygulamada, üretilen Python'da ve R'de tutması; deneylerin üretilen Python
koduyla birebir aynı sayıları vermesi) ``test_all_labs.py``'dedir; bu dosya bloğa özgü olanları denetler:

* notlardan farklı her seçimde uygulamanın sayıları ile üretilen Python kodunun sayıları 1e-12 düzeyinde aynıdır, üretilen
  R kodu çalışır, seçim yalnız kendi adımını değiştirir, adım metinleri ve ekran tabloları her seçimde doğru kalır;
* adımlar söyledikleri hesabı yapar ve bağımsız bir hesapla (statsmodels) aynı sayıyı verir: birim dönüşümünde t, p ve
  R²'nin değişmemesi, standartlaştırılmış katsayı (β̂·s_X/s_Y), tam yüzde, dönüm noktaları, merkezlemenin eşdeğerliği,
  model karşılaştırması ve ortak F; kukla regresyonunun ortalama farkı olması, R² ve standart hata formülleri, referans
  değişikliğinde tahmin edilen değerlerin ve ortak F'nin değişmemesi, sabitsiz tam kukla modelinin eşdeğerliği ve kukla
  tuzağında rütbe kaybı;
* motorun yeni işlemleri: etiketli grup özeti, sütun ve satır basamaklı birleşik tablo, makale tablosunda boş ek hücre
  ve R² basamağı, saçılım grafiğinde eğri, çizgi grafiğinde dikey çizgi, yüzde biçimli katsayı grafiği; formül
  okuyucunun çok terimli üs, ``s\\sqrt{…}`` ve ``b_1c`` yazımları;
* Sezgi deneyleri kuramsal ilişkileri tutturur ve metinleri kaydırıcıların uçlarında da doğru kalır;
* denklem ve boşluk sorularının kabul ve ret yazımları; sorulardaki sayılar veriden hesaplanır.
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
from core.charts import CHART_TYPES, figure_for
from core.codegen.base import generator, render_script
from core.labs.konu09 import KONU09_LAB
from core.labs.konu10 import KONU10_LAB
from core.labs.runner import LabState, batchable, plot_key, run_lab, run_operations
from core.labs.sezgi_konu09 import FORM, KONU09_EXPERIMENTS, STANDARDIZED, TURNING
from core.labs.sezgi_konu10 import CODING, KONU10_EXPERIMENTS, LOG, RAW
from core.labs.spec import (
    CoefficientPlot,
    GroupSummary,
    JoinColumns,
    LabSpec,
    LineChart,
    MonteCarlo,
    MultiChoice,
    NumberChoice,
    RegressionTable,
    ScatterPlot,
    ShowModel,
)
from core.quiz.expression import Symbol, equivalent, parse
from core.quiz.konu09 import KONU09_QUIZ
from core.quiz.konu10 import KONU10_QUIZ
from core.quiz.model import grade
from topics.lab_ui import coefficient_display, display_table, regression_display

R_ENVIRONMENT = dict(os.environ, LANG="C.UTF-8", LC_ALL="C.UTF-8")
LABS = (KONU09_LAB, KONU10_LAB)
EXPERIMENTS = KONU09_EXPERIMENTS + KONU10_EXPERIMENTS
QUADRATIC = "lwage ~ educ + exper + expersq + tenure + tenursq"
REGIONS = {"Kuzeydoğu": "northeast", "Kuzey Merkez": "northcen", "Güney": "south", "Batı": "west"}


def _printed(value: float, expected: float, decimals: int) -> bool:
    """Değer, notlarda ``decimals`` basamakla basılı sayıya yuvarlanır."""

    return abs(value - expected) <= 0.5 * 10 ** -decimals + 1e-12


@pytest.fixture(scope="module")
def wage1() -> pd.DataFrame:
    frame = W.load("wage1").copy()
    frame["northeast"] = 1 - frame["northcen"] - frame["south"] - frame["west"]
    frame["region"] = np.select([frame["northcen"].eq(1), frame["south"].eq(1), frame["west"].eq(1)],
                                ["Kuzey Merkez", "Güney", "Batı"], default="Kuzeydoğu")
    return frame


@pytest.fixture(scope="module")
def quadratic(wage1):
    return smf.ols(QUADRATIC, data=wage1).fit()


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
    assert sum(len(step.checks) for step in KONU09_LAB.steps) == 145
    assert sum(len(step.checks) for step in KONU10_LAB.steps) == 135
    assert [step.note.section for step in KONU09_LAB.steps] == ["9.2", "9.2", "9.3", "9.5", "9.5", "9.6", "9.7", "9.8"]
    assert [step.note.section for step in KONU10_LAB.steps] == ["10.2", "10.3", "10.3", "10.4", "10.5", "10.5", "10.6",
                                                                   "10.7", "10.8", "10.8", "10.9"]


@pytest.mark.parametrize("spec", LABS, ids=lambda s: s.topic_key)
def test_a_changed_choice_marks_only_its_own_step(spec: LabSpec) -> None:
    changes = _single_changes(spec)
    assert len(changes) >= 20
    for change in changes:
        (key,) = change
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


def test_konu09_notes_follow_the_chosen_specification() -> None:
    text = _note(KONU09_LAB, {}, 1)
    assert "(−2,873 → −287,273)" in text and "(0,599 → 59,897)" in text and "t = 11,679" in text
    text = _note(KONU09_LAB, {"adim1_y": "bin", "adim1_x": "ay"}, 1)
    assert "(0,599 → 0,000599)" in text and "(0,0513 → 0,0000513)" in text and "katsayı 0,0499 olur" in text
    assert "%32,00, tam yorum %37,71" in _note(KONU09_LAB, {}, 3)
    assert "tam yorum %−99,33" in _note(KONU09_LAB, {"adim3_beta": -0.5, "adim3_dx": 10}, 3)
    text = _note(KONU09_LAB, {}, 5)
    assert "0,01746" in text and "%1,75" in text and "tam hesap %1,70" in text and "yalnız 6 çalışan" in text
    assert "negatiftir (−0,01804; yaklaşık %−1,80)" in _note(KONU09_LAB, {"adim5_x": 40}, 5)
    assert "0,017465" in _note(KONU09_LAB, {}, 6)
    text = _note(KONU09_LAB, {"adim6_c": 30}, 6)
    assert "−0,006207" in text and "β̂₂ + 2β̂₃·30" in text
    assert "M₄'tedir (0,3608)" in _note(KONU09_LAB, {}, 7)
    assert "0,3669 → 0,3696" in _note(KONU09_LAB, {"adim7_ek": "numdep"}, 7)
    text = _note(KONU09_LAB, {}, 8)
    assert "F = 20,887" in text and "(ör. −0,000592)" in text and "deneyim ve kıdem katsayıları" in text
    text = _note(KONU09_LAB, {"adim8_model": "m2"}, 8)
    assert "F = 35,339" in text and "M₄'tür: ortak F = 20,887" in text and "(ör. −0,000661)" in text
    text = _note(KONU09_LAB, {"adim8_model": "m3"}, 8)
    assert "kıdem katsayısı tek başına" in text and "β̂₃ + 2β̂₄x" in text and "deneyim katsayısı" not in text
    text = _note(KONU09_LAB, {"adim2_x": ("exper",)}, 2)
    assert "Tek açıklayıcıyla standartlaştırılmış katsayı (0,111)" in text and "korelasyonuna eşittir" in text
    assert "eğitim değişkenindedir (0,510)" in _note(KONU09_LAB, {"adim2_x": ("educ", "exper")}, 2)


def test_konu10_notes_follow_the_chosen_specification() -> None:
    text = _note(KONU10_LAB, {}, 1)
    assert "4,5877 − 7,0995 = −2,5118" in text and "erkek çalışanların" in text
    text = _note(KONU10_LAB, {"adim1_kod": "male"}, 1)
    assert "7,0995 − 4,5877 = 2,5118" in text and "kadın çalışanların" in text
    assert "(log birim)" in _note(KONU10_LAB, {"adim1_y": "lwage"}, 1)
    assert "kadın katsayısı −2,273" in _note(KONU10_LAB, {"adim2_x": ("educ",)}, 2)
    assert "(kıdem katsayısı 0,0174)" in _note(KONU10_LAB, {"adim3_x": "tenure"}, 3)
    assert "δ̂ = 0,1666, t = 3,988" in _note(KONU10_LAB, {"adim4_kukla": "married"}, 4)
    text = _note(KONU10_LAB, {"adim5_ref": "west"}, 5)
    assert "referans kategori Batı" in text and "%10,95 daha düşüktür" in text and "ayrışan bölge: Güney" in text
    assert "Kuzeydoğu katsayısı −0,0331" in _note(KONU10_LAB, {"adim6_ref": "west"}, 6)
    assert "tam hesapla %0,58 daha düşüktür" in _note(KONU10_LAB, {"adim8_kukla": "nonwhite"}, 8)
    text = _note(KONU10_LAB, {"adim9_ref": "south"}, 9)
    assert "Referans Güney iken F(3, 517) = 2,095, p = 0,100" in text
    assert "katsayı −0,361" in _note(KONU10_LAB, {"adim11_x": ("educ",)}, 11)


# --- Adımlar söyledikleri hesabı yapar -------------------------------------------------------------------------

def test_konu09_scaling_standardization_and_exact_percent(wage1) -> None:
    dollars = smf.ols("wage ~ educ + exper + tenure", data=wage1).fit()
    for frame, formula, scale in ((wage1.assign(wage=100 * wage1["wage"]), "wage ~ educ + exper + tenure", 100),
                                  (wage1.assign(educ=wage1["educ"] / 10), "wage ~ educ + exper + tenure", 10)):
        model = smf.ols(formula, data=frame).fit()
        assert model.params["educ"] == pytest.approx(scale * dollars.params["educ"], rel=1e-12)
        assert model.tvalues["educ"] == pytest.approx(dollars.tvalues["educ"], rel=1e-10)
        assert model.rsquared == pytest.approx(dollars.rsquared, rel=1e-12)
    logs = smf.ols("lwage ~ educ + exper + tenure", data=wage1).fit()
    z = wage1[["lwage", "educ", "exper", "tenure"]].apply(lambda column: (column - column.mean()) / column.std())
    standardized = smf.ols("lwage ~ educ + exper + tenure", data=z).fit()
    state = run_operations(KONU09_LAB.operations_through(2))
    for name, printed in (("educ", 0.479), ("exper", 0.105), ("tenure", 0.300)):
        ratio = logs.params[name] * wage1[name].std() / wage1["lwage"].std()
        assert standardized.params[name] == pytest.approx(ratio, rel=1e-10)
        assert state.scalars[f"bz_{name}"] == pytest.approx(ratio, rel=1e-10) and _printed(ratio, printed, 3)
        assert standardized.tvalues[name] == pytest.approx(logs.tvalues[name], rel=1e-9)  # t değişmez
    for beta, dx, approximate, exact in ((0.02, 1, 2.00, 2.02), (0.08, 1, 8.00, 8.33), (0.20, 1, 20.00, 22.14),
                                         (0.08, 4, 32.00, 37.71)):  # Tablo 9.4
        assert _printed(100 * beta * dx, approximate, 2) and _printed(100 * math.expm1(beta * dx), exact, 2)
    assert _printed(100 * math.expm1(0.0845), 8.82, 2)


def test_konu09_turning_points_centering_and_model_comparison(wage1, quadratic) -> None:
    b = quadratic.params
    experience, tenure = -b["exper"] / (2 * b["expersq"]), -b["tenure"] / (2 * b["tenursq"])
    assert (round(experience, 2), round(tenure, 2)) == (24.76, 30.15)
    assert round(0.0293 / (2 * 0.000592), 2) == 24.75  # notlardaki yuvarlanmış formül
    assert (int((wage1["exper"] > experience).sum()), int((wage1["tenure"] > tenure).sum())) == (147, 6)
    state = run_operations(KONU09_LAB.operations_through(5))
    assert state.scalars["donum_d"] == pytest.approx(experience, rel=1e-12)
    for c in (0, 10, 25.5):
        centered = wage1.assign(xc=wage1["exper"] - c, xcsq=(wage1["exper"] - c) ** 2)
        model = smf.ols("lwage ~ educ + xc + xcsq + tenure + tenursq", data=centered).fit()
        assert np.max(np.abs(model.fittedvalues - quadratic.fittedvalues)) < 1e-10
        assert model.params["xc"] == pytest.approx(b["exper"] + 2 * b["expersq"] * c, rel=1e-9, abs=1e-12)
        assert model.params["xcsq"] == pytest.approx(b["expersq"], rel=1e-9)
        assert model.ssr == pytest.approx(quadratic.ssr, rel=1e-12) and model.rsquared == pytest.approx(
            quadratic.rsquared, rel=1e-12)
        assert model.params["Intercept"] == pytest.approx(b["Intercept"] + b["exper"] * c + b["expersq"] * c ** 2,
                                                          rel=1e-9)  # soru e04
    formulas = ("lwage ~ educ + exper + tenure", "lwage ~ educ + exper + expersq + tenure",
                "lwage ~ educ + exper + tenure + tenursq", QUADRATIC)
    printed = ((0.3160, 0.3121, 101.456), (0.3595, 0.3545, 95.011), (0.3341, 0.3290, 98.774),
               (0.3669, 0.3608, 93.911))
    for formula, (r2, adjusted, ssr) in zip(formulas, printed):
        model = smf.ols(formula, data=wage1).fit()
        assert _printed(model.rsquared, r2, 4) and _printed(model.rsquared_adj, adjusted, 4) and _printed(
            model.ssr, ssr, 3), formula
    test = quadratic.f_test("expersq = 0, tenursq = 0")
    linear = smf.ols(formulas[0], data=wage1).fit()
    assert round(float(test.fvalue), 3) == 20.887
    assert float(test.fvalue) == pytest.approx(((linear.ssr - quadratic.ssr) / 2) / (quadratic.ssr / 520), rel=1e-10)


def test_konu10_dummy_regression_is_a_mean_difference(wage1) -> None:
    model = smf.ols("wage ~ female", data=wage1).fit()
    means = wage1.groupby("female")["wage"].mean()
    assert model.params["Intercept"] == pytest.approx(means[0], rel=1e-12)
    assert model.params["female"] == pytest.approx(means[1] - means[0], rel=1e-12)
    n, share = len(wage1), wage1["female"].mean()
    tkt = float(((wage1["wage"] - wage1["wage"].mean()) ** 2).sum())
    assert model.rsquared == pytest.approx(n * share * (1 - share) * model.params["female"] ** 2 / tkt, rel=1e-10)
    sigma = math.sqrt(model.ssr / model.df_resid)
    counts = wage1["female"].value_counts()
    assert model.bse["female"] == pytest.approx(sigma * math.sqrt(1 / counts[1] + 1 / counts[0]), rel=1e-10)
    state = run_operations(KONU10_LAB.operations_through(1))
    table = state.tables["tablo101"]
    assert list(table.index) == ["Erkek", "Kadın"]
    assert tuple(round(value, 2) for value in table.loc["Kadın"].iloc[1:]) == (4.59, 12.32, 16.43, 3.62)


def test_konu10_reference_change_and_the_dummy_trap(wage1) -> None:
    base = "lwage ~ educ + exper + expersq + tenure + tenursq"
    fits, tests = [], []
    for reference in REGIONS:
        model = smf.ols(f'{base} + C(region, Treatment(reference="{reference}"))', data=wage1).fit()
        others = [name for name in model.params.index if name.startswith("C(region")]
        tests.append(float(model.f_test(", ".join(f"{name} = 0" for name in others)).fvalue))
        fits.append(model)
    for model in fits[1:]:
        assert np.max(np.abs(model.fittedvalues - fits[0].fittedvalues)) < 1e-12
        assert model.rsquared == pytest.approx(fits[0].rsquared, rel=1e-12)
    assert max(tests) - min(tests) < 1e-9 and round(tests[0], 3) == 2.095
    northeast = fits[0].params
    south = fits[2].params
    assert south['C(region, Treatment(reference="Güney"))[T.Kuzeydoğu]'] == pytest.approx(
        -northeast['C(region, Treatment(reference="Kuzeydoğu"))[T.Güney]'], rel=1e-10)
    assert south['C(region, Treatment(reference="Güney"))[T.Batı]'] == pytest.approx(
        northeast['C(region, Treatment(reference="Kuzeydoğu"))[T.Batı]']
        - northeast['C(region, Treatment(reference="Kuzeydoğu"))[T.Güney]'], rel=1e-10)
    no_intercept = smf.ols(f"{base} + northeast + northcen + south + west - 1", data=wage1).fit()
    assert np.max(np.abs(no_intercept.fittedvalues - fits[0].fittedvalues)) < 1e-10
    design = np.column_stack([np.ones(len(wage1)), wage1[["northeast", "northcen", "south", "west"]].to_numpy(float)])
    assert np.linalg.matrix_rank(design) == 4  # sabit + dört kukla: beş sütun, rütbe dört (kukla tuzağı)
    industries = ("construc", "ndurman", "trcommpu", "trade", "services", "profserv")
    industry = smf.ols(f"{base} + " + " + ".join(industries), data=wage1).fit()
    joint = industry.f_test(", ".join(f"{name} = 0" for name in industries))
    assert (round(float(joint.fvalue), 3), int(joint.df_num), int(joint.df_denom)) == (8.242, 6, 514)


def test_konu10_log_dummy_exact_percent_and_interval(wage1) -> None:
    model = smf.ols("lwage ~ female + educ + exper + tenure", data=wage1).fit()
    delta = model.params["female"]
    low, high = model.conf_int().loc["female"]
    assert round(100 * math.expm1(delta), 2) == -26.00 and round(100 * delta, 2) == -30.11
    assert (round(100 * math.expm1(low), 2), round(100 * math.expm1(high), 2)) == (-31.22, -20.39)
    state = run_operations(KONU10_LAB.operations_through(8))
    assert state.scalars["tam_alt"] == pytest.approx(100 * math.expm1(low), rel=1e-10)


# --- Motorun yeni işlemleri ---------------------------------------------------------------------------------

def test_group_summary_labels_and_heading_in_the_app_and_in_both_languages() -> None:
    op = next(op for op in KONU10_LAB.step(1).operations if isinstance(op, GroupSummary))
    assert op.labels == ((0, "Erkek"), (1, "Kadın")) and op.heading == "Grup"
    state = run_operations(KONU10_LAB.operations_through(1))
    shown = display_table(op, state.tables[op.result], KONU10_LAB.label)
    assert list(shown["Grup"]) == ["Erkek", "Kadın"] and list(shown["Gözlem sayısı"]) == ["274", "252"]
    python, r = render_script(KONU10_LAB, "Python"), render_script(KONU10_LAB, "R")
    assert 'rename(index={0: "Erkek", 1: "Kadın"})' in python
    assert 'rownames(tablo101) <- c("Erkek", "Kadın")' in r
    with pytest.raises(ValueError):
        GroupSummary("veri", "kod", (("n", "y", "count"),), "g", (1, 2), labels=((2, "iki"), (1, "bir")))


def test_join_columns_decimals_and_p_columns() -> None:
    state = run_operations(KONU09_LAB.operations_through(6))
    tables = {op.result: op for step in KONU09_LAB.steps[:6] for op in step.operations if isinstance(op, JoinColumns)}
    shown = display_table(tables["tablo92"], state.tables["tablo92"], KONU09_LAB.label)
    assert list(shown["p"]) == ["< 0,001"] * 3 and list(shown["Eğitim katsayısı"]) == ["0,599", "59,897", "5,990"]
    shown = display_table(tables["tablo96"], state.tables["tablo96"], KONU09_LAB.label).set_index("Ölçüt")
    assert shown.loc["R²", "Ham model"] == "0,366875" and shown.loc["HKT", "Merkezlenmiş model"] == "93,9113"
    with pytest.raises(ValueError):
        JoinColumns("t", (("a", "x", "deger"),), column_decimals=(("yok", 2),))
    with pytest.raises(ValueError):
        JoinColumns("t", (("a", "x", "deger"),), p_columns=("yok",))


def test_regression_table_blank_extra_cells_and_fit_decimals() -> None:
    state = run_operations(KONU09_LAB.operations_through(8))
    op = next(op for op in KONU09_LAB.step(8).operations if isinstance(op, RegressionTable))
    shown = regression_display(op, state, KONU09_LAB.label).set_index("Değişken")
    linear, quadratic_column = (heading for heading, _ in op.models)
    assert shown.loc["Karesel terimler ortak F", linear] == "—" and shown.loc["Karesel terimler ortak F",
                                                                            quadratic_column] == "20,887"
    assert shown.loc["Ortak test p-değeri", quadratic_column] == "< 0,001"
    assert (shown.loc["R²", linear], shown.loc["Düzeltilmiş R²", quadratic_column]) == ("0,316", "0,361")
    assert shown.loc["Potansiyel deneyim (yıl)", linear] == "0,0041**"  # notlardaki gibi dört basamak
    resolved = KONU09_LAB.resolve({"adim8_model": "m3"})  # seçilen model ile notlardaki M₄ yan yana
    state = run_operations(resolved.operations_through(8))
    op = next(op for op in resolved.step(8).operations if isinstance(op, RegressionTable))
    shown = regression_display(op, state, KONU09_LAB.label).set_index("Değişken")
    assert [heading for heading, _ in op.models] == ["(1) Doğrusal", "(2) Seçiminiz: M₃", "(2) Notlar: M₄"]
    assert list(shown.loc["Karesel terim testi: F"]) == ["—", "14,145", "20,887"]
    python, r = render_script(KONU09_LAB, "Python"), render_script(KONU09_LAB, "R")
    assert "np.nan" in python and "NA" in r
    state = run_operations(KONU10_LAB.operations_through(6))
    op = next(op for op in KONU10_LAB.step(6).operations if isinstance(op, RegressionTable))
    shown = regression_display(op, state, KONU10_LAB.label).set_index("Değişken")
    assert list(shown.loc["R²"]) == ["0,374477", "0,374477"]


def test_scatter_curves_and_line_vertical_lines_in_the_app_and_in_both_languages() -> None:
    state = run_operations(KONU09_LAB.operations_through(5))
    scatter = next(op for op in KONU09_LAB.step(5).operations if isinstance(op, ScatterPlot))
    line = next(op for op in KONU09_LAB.step(5).operations if isinstance(op, LineChart))
    assert scatter.curves and line.vlines
    figure = figure_for(scatter, state, KONU09_LAB.label)
    curve = next(trace for trace in figure.data if trace.name == scatter.curves[0][3])
    frame = state.frames[scatter.curves[0][0]]
    np.testing.assert_allclose(curve.y, frame[scatter.curves[0][2]], rtol=0, atol=1e-12)
    figure = figure_for(line, state, KONU09_LAB.label)
    names = [trace.name for trace in figure.data]
    assert any("24,76" in (name or "") for name in names)
    python, r = render_script(KONU09_LAB, "Python"), render_script(KONU09_LAB, "R")
    assert "ax.axvline(" in python and "abline(v = " in r
    assert python.count("ax.plot(") >= 2 and "lines(" in r
    with pytest.raises(ValueError):
        LineChart("f", "x", "y", "x", "y", "t", series=(("z", "z"),), vlines=(("a", "a"),))


def test_coefficient_plot_in_percent() -> None:
    state = run_operations(KONU10_LAB.operations_through(5))
    op = next(op for op in KONU10_LAB.step(5).operations if isinstance(op, CoefficientPlot))
    assert op.percent
    data = state.plots[plot_key(op)].set_index("terim")
    model = state.models[op.model]
    interval = model.conf_int(alpha=round(1 - op.level, 10))
    for term in op.terms:
        assert data.loc[term, "tahmin"] == pytest.approx(100 * math.expm1(model.params[term]), rel=1e-12)
        assert data.loc[term, "alt"] == pytest.approx(100 * math.expm1(interval.loc[term, 0]), rel=1e-12)
        assert data.loc[term, "ust"] == pytest.approx(100 * math.expm1(interval.loc[term, 1]), rel=1e-12)
    figure_for(op, state, KONU10_LAB.label)
    python, r = render_script(KONU10_LAB, "Python"), render_script(KONU10_LAB, "R")
    assert "100 * (np.exp(" in python and "100 * (exp(" in r


def test_formula_reader_handles_multi_term_exponents_roots_and_subscripts() -> None:
    symbols = (Symbol("d", "\\widehat\\delta", "d", -1, 1, aliases=("\\hat\\delta", "\\widehat\\delta")),
               Symbol("s", "s", "s", 0.01, 0.2), Symbol("c", "c", "c", 1.6, 2.7))
    reference = parse("100*(exp(d - c*s) - 1)", symbols)
    for text in ("100(e^{\\hat\\delta - c s} - 1)", "100\\left(e^{\\widehat\\delta - c\\,s} - 1\\right)",
                 "100(e^{\\hat{\\delta}-c\\cdot s}-1)"):
        assert equivalent(parse(text, symbols), reference, symbols), text
    assert not equivalent(parse("100(e^{\\hat\\delta} - c s - 1)", symbols), reference, symbols)
    roots = (Symbol("s", "s", "s", 0.5, 5), Symbol("n1", "n_1", "n", 20, 300, aliases=("n_1",)),
             Symbol("n0", "n_0", "n", 20, 300, aliases=("n_0",)))
    assert equivalent(parse("s\\sqrt{\\frac{1}{n_1}+\\frac{1}{n_0}}", roots), parse("s*sqrt(1/n1 + 1/n0)", roots),
                      roots)
    terms = (Symbol("b0", "b_0", "b", 0, 5), Symbol("b1", "b_1", "b", 0.5, 3), Symbol("b2", "b_2", "b", -0.5, 0.5),
             Symbol("c", "c", "c", 1, 20))
    assert equivalent(parse("b_0+b_1c+b_2c^2", terms), parse("b0 + b1*c + b2*c^2", terms), terms)
    assert equivalent(parse("x^{2}", (Symbol("x", "x", "x"),)), parse("x*x", (Symbol("x", "x", "x"),)),
                      (Symbol("x", "x", "x"),))


# --- Sezgi deneyleri ------------------------------------------------------------------------------------

def _extremes(experiment) -> list[dict[str, float]]:
    settings = [experiment.defaults()]
    for parameter in experiment.parameters:
        for value in (parameter.minimum, parameter.maximum):
            settings.append(dict(experiment.defaults(), **{parameter.key: value}))
    return settings


def test_block_simulations_use_the_batch_path() -> None:
    for experiment in (TURNING, RAW):
        for parameters in _extremes(experiment):
            simulations = [op for op in experiment.build(parameters) if isinstance(op, MonteCarlo)]
            assert simulations and all(batchable(op) for op in simulations), experiment.key


@pytest.mark.parametrize("experiment", EXPERIMENTS, ids=lambda e: e.key)
def test_experiment_texts_and_tables_are_correct_at_the_extremes(experiment) -> None:
    for parameters in _extremes(experiment):
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
            shown = display_table(producers[entry[0]], state.tables[entry[0]], experiment.label)
            assert shown.columns.is_unique, (experiment.key, list(shown.columns))
        for op in operations:
            if isinstance(op, CHART_TYPES):
                figure_for(op, state, experiment.label)


def test_konu09_experiments_match_their_theory() -> None:
    s = run_operations(STANDARDIZED.build(dict(STANDARDIZED.defaults(), n=2000))).scalars
    assert (s["z1"], s["z2"]) == pytest.approx((s["g1"], s["g2"]), abs=0.03)
    assert s["z1"] == pytest.approx(s["b1"] * s["s_x1"] / s["s_y"], rel=1e-12)
    s = run_operations(TURNING.build(TURNING.defaults())).scalars
    assert s["med_a"] == pytest.approx(5, abs=0.2) and s["yakin_a"] > 0.9  # dönüm noktası verinin içinde
    assert s["yakin_b"] < 0.5 and s["aralik_b"] == 0  # x* = 20: veri aralığının dışında, isabetsiz
    state = run_operations(TURNING.build(dict(TURNING.defaults(), xstar=5)))  # B = A: aynı çekilişler, aynı x*
    pd.testing.assert_series_equal(state.tables["icerde"]["donum"], state.tables["secilen"]["donum"], check_names=False)
    state = run_operations(TURNING.build(TURNING.defaults()))
    shown = state.tables["donumler"]
    for column, row in (("icerde", 0), ("secilen", 1)):
        outside = ~shown[column].between(-10, 50)
        assert state.tables["donum_ozet"]["disarida"].iloc[row] == pytest.approx(outside.mean(), abs=1e-12)
    # Düz çizginin gerçek ortalamadan sapması: anakütlede γ·(100/27, −200/27, 100/27) (X ~ U(0, 10), üç eşit bölge)
    s = run_operations(FORM.build(dict(FORM.defaults(), n=1000))).scalars
    gamma = FORM.defaults()["gamma"]
    for name, share in (("dusuk", 100 / 27), ("orta", -200 / 27), ("yuksek", 100 / 27)):
        assert s[f"sap_{name}"] == pytest.approx(gamma * share, abs=0.15), name
        assert abs(s[f"sapk_{name}"]) < 0.15, name
    assert s["r2_k"] > s["r2_d"] and s["sap_orta"] < 0 < min(s["sap_dusuk"], s["sap_yuksek"])
    s = run_operations(FORM.build(dict(FORM.defaults(), gamma=0.0))).scalars
    assert abs(s["sap_orta"]) < 0.3 and s["r2_k"] == pytest.approx(s["r2_d"], abs=0.01)
    table = run_operations(STANDARDIZED.build(STANDARDIZED.defaults())).tables["katsayilar"]
    assert list(table.columns) == ["Gerçek eğim", "Ham eğim tahmini", "Standartlaştırılmış (anakütle)",
                                   "Standartlaştırılmış (örneklem)"] and (table["Gerçek eğim"] == 0.5).all()


def test_dgp_numbers_keep_their_trailing_zeros() -> None:
    """Sondaki sıfırları atan yardımcı tamsayıya dokunmaz (20 → "2" ve 100 → "1" hatası; Konu 2 ve 9 DGP'leri)."""

    import importlib

    for number in range(11):
        module = importlib.import_module(f"core.labs.sezgi_konu{number:02d}")
        for name in ("_tex", "_short"):
            helper = getattr(module, name, None)
            if helper is not None:
                assert helper(20.0, 0) == "20" and helper(100.0, 0) == "100" and helper(0.0, 0) == "0", (number, name)
                assert helper(0.5).replace("{,}", ",") == "0,5", (number, name)
    assert "x^* = 20;" in TURNING.dgp(TURNING.defaults())[2]


def test_konu10_experiments_match_their_theory() -> None:
    for parameters in _extremes(CODING) + [dict(CODING.defaults(), a=1.0, b=2.0)]:
        state = run_operations(CODING.build(parameters))
        s, groups = state.scalars, state.tables["gruplar"]["ortalama"]
        assert s["k2"] == pytest.approx(groups.iloc[1] - groups.iloc[0], rel=1e-10, abs=1e-12)
        assert s["k3"] == pytest.approx(groups.iloc[2] - groups.iloc[0], rel=1e-10, abs=1e-12)
        assert s["r2_kukla"] >= s["r2_sayisal"] - 1e-12  # sayısal kod, kukla modelinin kısıtlı hâlidir
    s = run_operations(CODING.build(dict(CODING.defaults(), a=1.0, b=2.0))).scalars
    assert s["r2_kukla"] - s["r2_sayisal"] < 0.005  # eşit adımlı farklar: iki model yakın
    for delta, gap in ((-1.0, 2.0), (1.0, -4.0), (0.5, 0.0)):
        s = run_operations(RAW.build(dict(RAW.defaults(), delta=delta, gap=gap))).scalars
        assert s["ort_ham"] == pytest.approx(delta + 0.5 * gap, abs=4 * s["ss_ham"] / math.sqrt(1000))
        assert s["ort_kontrol"] == pytest.approx(delta, abs=4 * s["ss_kontrol"] / math.sqrt(1000))
    for delta in (-1.0, -0.3, 0.0, 0.8):
        s = run_operations(LOG.build(dict(LOG.defaults(), delta=delta, sigma=0.2, n=10000))).scalars
        assert s["tam"] == pytest.approx(100 * math.expm1(s["d_hat"]), rel=1e-12, abs=1e-12)
        assert s["gercek_tam"] == pytest.approx(100 * math.expm1(delta), abs=1e-9)
        assert s["ortalama_fark"] == pytest.approx(s["gercek_tam"], abs=1.5)  # oran exp(δ)'yı tahmin eder
        assert s["tam"] == pytest.approx(s["gercek_tam"], abs=1.5)


# --- Kendini sına: yazım çeşitleri ve sayılar -----------------------------------------------------------------

EQUATIONS = {
    ("konu09", "e01"): (("b0 - b1^2/(4*b2)", "b_0 - \\frac{b_1^2}{4\\,b_2}", "\\beta_0 - \\frac{\\beta_1^2}{4\\beta_2}",
                         "b0 - (b1^2)/(4b2)"),
                        ("b0 + b1^2/(4*b2)", "-b1/(2*b2)", "b0 - b1^2/(2*b2)")),
    ("konu09", "e02"): (("-b1/(2*b2) - 1/2", "-b1/(2b2) - 0.5", "-(b_1 + b_2)/(2 b_2)", "-\\frac{b_1}{2b_2}-\\frac{1}{2}"),
                        ("-b1/(2*b2)", "-b1/(2*b2) + 1/2", "-b1/b2 - 1")),
    ("konu09", "e03"): (("100*((1 + p/100)^b - 1)", "100\\left[\\left(1 + \\frac{p}{100}\\right)^{b} - 1\\right]",
                         "100(exp(b*ln(1+p/100))-1)"),
                        ("b*p", "100*(exp(b*p/100) - 1)", "100*((1 + p)^b - 1)")),
    ("konu09", "e04"): (("b0 + b1*c + b2*c^2", "b_0 + b_1 c + b_2 c^2", "b_0+b_1c+b_2c^2", "b0+b1*c+b2*c²"),
                        ("b0 + b1*c", "b1 + 2*b2*c", "b0 - b1*c + b2*c^2")),
    ("konu09", "e05"): (("b1 + 2*b2*mu", "\\beta_1 + 2\\beta_2\\mu", "β₁ + 2β₂μ", "β1 + 2β2μ"),
                        ("b1 + b2*mu", "b1 + 2*b2", "2*b2*mu")),
    ("konu10", "e01"): (("n*p*(1 - p)*d^2/tkt", "n p (1-p) δ̂^2 / TKT", "\\frac{np(1-p)\\widehat\\delta^2}{\\text{TKT}}"),
                        ("p*(1 - p)*d^2/tkt", "n*p*d^2/tkt", "n*p*(1 - p)*d/tkt")),
    ("konu10", "e02"): (("d + b*g", "\\widehat\\delta + \\widehat b\\,\\Delta", "\\hat{\\delta}+\\hat{b}\\Delta",
                         "δ̂ + b̂Δ"),
                        ("d - b*g", "d + g", "b*g")),
    ("konu10", "e03"): (("s*sqrt(1/n1 + 1/n0)", "\\widehat\\sigma\\sqrt{\\frac{1}{n_1}+\\frac{1}{n_0}}",
                         "s\\sqrt{\\frac{1}{n_1}+\\frac{1}{n_0}}", "σ̂·sqrt((n1+n0)/(n1 n0))"),
                        ("s/sqrt(n1 + n0)", "s*sqrt(1/n1)", "s*(1/n1 + 1/n0)")),
    ("konu10", "e04"): (("100*(exp(d - c*s) - 1)", "100(e^{\\widehat\\delta - c s} - 1)",
                         "100\\left(e^{\\hat\\delta - c\\,s} - 1\\right)", "100(e^(d-cs)-1)"),
                        ("100*(d - c*s)", "100*(exp(d + c*s) - 1)", "100*exp(d - c*s)")),
    ("konu10", "e05"): (("log(1 + P/100)", "\\ln\\left(1 + \\frac{P}{100}\\right)", "ln(1+0.01P)"),
                        ("P/100", "exp(P/100) - 1", "log(P/100)")),
}
QUIZZES = {"konu09": KONU09_QUIZ, "konu10": KONU10_QUIZ}


@pytest.mark.parametrize("topic, key", sorted(EQUATIONS))
def test_equation_questions_accept_equivalent_forms_and_reject_wrong_ones(topic: str, key: str) -> None:
    question = QUIZZES[topic].question(key)
    accepted, rejected = EQUATIONS[(topic, key)]
    for text in accepted:
        assert grade(question, text).correct, (topic, key, text)
    for text in rejected:
        assert not grade(question, text).correct, (topic, key, text)


def test_every_equation_question_is_listed() -> None:
    for topic, quiz in QUIZZES.items():
        keys = {q.key for q in quiz.questions if q.kind == "denklem"}
        assert {(topic, key) for key in keys} == {item for item in EQUATIONS if item[0] == topic}, topic


BLANKS = {
    ("konu09", "b01"): ((["0,1746", "19,08"], ["0.1747", "19.07"]), (["0,2930", "19,08"], ["0,1746", "17,46"])),
    ("konu09", "b02"): ((["3,09", "0,63"], ["3,10", "0.63"]), (["2,48", "1,25"], ["3,09", "0,06"])),
    ("konu09", "b03"): ((["6,09", "2,47"], ["6.09", "2.47"]), (["20,89", "4,57"], ["6,09", "6,09"])),
    ("konu09", "b04"): ((["0,98", "0,00"], ["0.981", "0"]), (["0,96", "0,00"], ["0,98", "0,88"])),
    ("konu09", "b05"): ((["8,20", "11,83"], ["8.2", "11.83"]), (["11,83", "8,20"], ["8,45", "11,83"])),
    ("konu10", "b01"): ((["47", "52"], ["47,0", "52,0"]), (["49", "54"], ["45", "50"])),
    ("konu10", "b02"): ((["25,61", "45,35"], ["25.61", "45.40"]), (["−31,22", "−20,39"], ["22,80", "37,40"])),
    ("konu10", "b03"): ((["0,104", "10,96"], ["0.1045", "11.01"]), (["−0,038", "10,96"], ["0,104", "10,40"])),
    ("konu10", "b04"): ((["6", "7"], ["6", "7"]), (["7", "8"], ["6", "8"])),
    ("konu10", "b05"): ((["11", "5"], ["11", "5"]), (["12", "6"], ["11", "6"])),
}


@pytest.mark.parametrize("topic, key", sorted(BLANKS))
def test_blank_questions_read_turkish_numbers(topic: str, key: str) -> None:
    question = QUIZZES[topic].question(key)
    accepted, rejected = BLANKS[(topic, key)]
    for response in accepted:
        assert grade(question, response).correct, (topic, key, response)
    for response in rejected:
        assert not grade(question, response).correct, (topic, key, response)


def test_konu09_quiz_numbers_are_computed_not_typed(wage1, quadratic) -> None:
    b = quadratic.params
    assert (round(0.08 * 2 / 0.5, 2), round(0.08 * 4 / 0.5, 2)) == (0.32, 0.64)  # k01
    assert round(100 * math.expm1(-0.32), 2) == -27.39 and round(100 * math.expm1(0.08), 2) == 8.33  # k02
    assert (-0.5 / (2 * 0.02), 0.5 + 0.04 * 20) == (-12.5, 1.3)  # k03
    assert round(-b["tenure"] / (2 * b["tenursq"]), 2) == 30.15 and int(wage1["tenure"].max()) == 44  # k04
    assert round(quadratic.pvalues["tenursq"], 3) == 0.014
    assert round(b["exper"] + 20 * b["expersq"], 5) == 0.01746  # k05
    assert (round(b["expersq"], 6), round(quadratic.bse["expersq"], 6), round(quadratic.tvalues["expersq"], 3)) == (
        -0.000592, 0.000114, -5.189)  # k06
    assert round(np.corrcoef(wage1["exper"], wage1["expersq"])[0, 1], 2) == 0.96  # d01
    assert all(value ** 2 == 7 * value - 10 for value in (2, 5))
    assert round(b["Intercept"] + math.log(100), 3) == 4.807  # d02
    logs = smf.ols(QUADRATIC.replace("lwage", "np.log(wage)"), data=wage1).fit()  # lwage veri setinde yuvarlanmıştır
    scaled = smf.ols(QUADRATIC.replace("lwage", "np.log(100 * wage)"), data=wage1).fit()
    assert scaled.params["educ"] == pytest.approx(logs.params["educ"], rel=1e-10)
    assert scaled.params["Intercept"] == pytest.approx(logs.params["Intercept"] + math.log(100), rel=1e-10)
    assert scaled.bse["educ"] == pytest.approx(logs.bse["educ"], rel=1e-10)
    assert round(0.0293 * 10 - 0.000592 * 200, 4) == 0.1746 and round(100 * math.expm1(0.1746), 2) == 19.08  # b01
    assert round(100 * (b["exper"] * 10 + b["expersq"] * 200), 2) == pytest.approx(17.46, abs=0.01)
    assert (round(100 * (0.0371 - 2 * 0.000616 * 5), 2), round(100 * (0.0371 - 2 * 0.000616 * 25), 2)) == (3.09,
                                                                                                          0.63)  # b02
    m2 = smf.ols("lwage ~ educ + exper + expersq + tenure", data=wage1).fit()
    assert round(m2.ssr, 3) == 95.011 and round((95.011 - 93.911) / (93.911 / 520), 2) == 6.09  # b03
    assert round(abs(quadratic.tvalues["tenursq"]), 3) == 2.468
    x = np.arange(1, 6)
    assert round(np.corrcoef(x, x ** 2)[0, 1], 2) == 0.98 and np.corrcoef(x - 3, (x - 3) ** 2)[0, 1] == 0  # b04
    assert (round(math.log(2) / 0.0845, 2), round(100 / 8.45, 2)) == (8.20, 11.83)  # b05
    assert round(0.0293 ** 2 / (4 * 0.000592), 2) == 0.36  # e01
    assert b["exper"] + b["expersq"] * 49 > 0 > b["exper"] + b["expersq"] * 51  # e02: 24→25 artar, 25→26 azalır
    assert round(100 * math.expm1(b["exper"] + b["expersq"] * 51), 2) == -0.09
    assert round(100 * (1.5 ** 0.5 - 1), 2) == 22.47  # e03
    assert round(10 * 0.0293 - 100 * 0.000592, 3) == 0.234  # e04


def test_konu10_quiz_numbers_are_computed_not_typed(wage1) -> None:
    base = "lwage ~ educ + exper + expersq + tenure + tenursq"
    northeast = smf.ols(f'{base} + C(region, Treatment(reference="Kuzeydoğu"))', data=wage1).fit()
    south = smf.ols(f'{base} + C(region, Treatment(reference="Güney"))', data=wage1).fit()
    term = 'C(region, Treatment(reference="{}"))[T.{}]'
    west_se, south_se = northeast.bse[term.format("Kuzeydoğu", "Batı")], northeast.bse[term.format("Kuzeydoğu", "Güney")]
    assert (round(west_se, 4), round(south_se, 4)) == (0.0598, 0.0505)  # k03
    assert round(math.sqrt(west_se ** 2 + south_se ** 2), 4) == 0.0782
    assert round(south.bse[term.format("Güney", "Batı")], 4) == 0.0548
    assert round(south.pvalues[term.format("Güney", "Batı")], 3) == 0.035  # d05
    assert round(northeast.pvalues[term.format("Kuzeydoğu", "Batı")], 3) == 0.580
    covariance = northeast.cov_params().loc[term.format("Kuzeydoğu", "Batı"), term.format("Kuzeydoğu", "Güney")]
    assert round(covariance / (west_se * south_se), 2) == 0.52
    assert (round(100 * math.expm1(-0.30), 2), round(100 * math.expm1(0.10), 2), round(100 * math.expm1(-0.20), 2)) \
        == (-25.92, 10.52, -18.13)  # k04
    assert round(-25.92 + 10.52, 2) == -15.40
    logs = smf.ols("lwage ~ female + educ + exper + tenure", data=wage1).fit()
    low, high = logs.conf_int().loc["female"]
    assert (round(low, 3), round(high, 3)) == (-0.374, -0.228)  # b02
    assert (round(100 * math.expm1(0.228), 2), round(100 * math.expm1(0.374), 2)) == (25.61, 45.35)
    assert (round(100 * math.expm1(-high), 2), round(100 * math.expm1(-low), 2)) == (25.61, 45.40)
    assert round(100 * math.expm1(-logs.params["female"]), 2) == 35.14 and round(100 * math.expm1(0.3011), 2) == 35.13
    assert (round(-0.3011 - 1.96 * 0.0372, 3), round(-0.3011 + 1.96 * 0.0372, 3)) == (-0.374, -0.228)
    assert (0.033 - (-0.071), round(100 * math.expm1(0.104), 2)) == pytest.approx((0.104, 10.96))  # b03
    west, north_central = northeast.params[term.format("Kuzeydoğu", "Batı")], northeast.params[
        term.format("Kuzeydoğu", "Kuzey Merkez")]
    assert round(west - north_central, 4) == 0.1045 and round(100 * math.expm1(west - north_central), 2) == 11.01
    assert (49 - 0.4 * 5, 49 - 0.4 * 5 + 5) == (47.0, 52.0)  # b01
    assert ((5 - 1) + (3 - 1), 5 + (3 - 1)) == (6, 7)  # b04
    assert (526 - 514 - 1, 526 - 514 - 1 - 6) == (11, 5)  # b05
    simple = smf.ols("wage ~ female", data=wage1).fit()
    share = wage1["female"].mean()
    assert round(526 * share * (1 - share) * 2.5118 ** 2 / 7160.41, 3) == 0.116  # e01
    level = smf.ols("wage ~ female + educ + exper + tenure", data=wage1).fit()
    assert tuple(round(level.params[name], 3) for name in ("female", "educ", "exper", "tenure")) == (
        -1.811, 0.572, 0.025, 0.141)  # e02
    assert round(-1.811 + 0.572 * (-0.47) + 0.025 * (-1.13) + 0.141 * (-2.85), 2) == -2.51
    sigma = math.sqrt(simple.ssr / simple.df_resid)
    assert (round(sigma, 3), round(math.sqrt(1 / 252 + 1 / 274), 4)) == (3.476, 0.0873)  # e03
    assert round(sigma * math.sqrt(1 / 252 + 1 / 274), 4) == round(simple.bse["female"], 4) == 0.3034
    assert (round(-26.00 + 31.22, 2), round(-20.39 + 26.00, 2)) == (5.22, 5.61)  # e04
    assert (round(math.log(0.74), 3), round(math.log(0.8), 3)) == (-0.301, -0.223)  # e05
    assert round(0.004 * 100, 1) == 0.4  # k02
