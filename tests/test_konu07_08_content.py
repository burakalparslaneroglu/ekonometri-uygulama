"""Konu 7–8 bloğu: tek katsayı çıkarımı ve F testi; etkileşimli adımlar, motorun yeni işlemleri, Sezgi deneyleri ve
sorular.

Genel sözleşmeler (notlardaki her sayının uygulamada, üretilen Python'da ve R'de tutması; deneylerin üretilen Python
koduyla birebir aynı sayıları vermesi) ``test_all_labs.py``'dedir; bu dosya bloğa özgü olanları denetler:

* notlardan farklı her seçimde uygulamanın sayıları ile üretilen Python kodunun sayıları 1e-12 düzeyinde aynıdır, üretilen
  R kodu çalışır, seçim yalnız kendi adımını değiştirir, adım metinleri ve ekran tabloları her seçimde doğru kalır;
* adımlar söyledikleri hesabı yapar ve bağımsız bir hesapla (statsmodels, scipy) aynı sayıyı verir: t istatistiği,
  kritik değerler, p-değerleri (iki taraflı ve tek taraflı), güven aralıkları ve test–aralık eşdeğerliği, ortak F testi
  (``f_test``, SSR ve R² biçimleri), genel F, tek kısıtta F = t²;
* motorun yeni işlemleri: t ve F fonksiyonları, test grafiğinin ekseni ve alanları, çıkarım tablosu, ortak test, ek
  satırlı makale tablosu, p-değeri yazımı, başvuru çizgili aralık grafiği, sonuç tablosundan saçılım grafiği;
* Sezgi deneyleri varsayılan ayarlarda notlardaki tabloları verir (Tablo 7.1, Tablo 8.3), kuramsal ilişkileri tutturur
  (kapsama, boyut, güç, ailece hata, eksik değişkenin merkezi) ve metinleri kaydırıcıların uçlarında da doğru kalır;
* denklem ve boşluk sorularının kabul ve ret yazımları; sorulardaki sayılar veriden hesaplanır.
"""

from __future__ import annotations

import contextlib
import io
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import stats

from core import wooldridge_data as W
from core.charts import CHART_TYPES, figure_for, hypothesis_labels, p_text
from core.codegen.base import generator, render_script
from core.labs import expr as E
from core.labs import inference as I
from core.labs.konu07 import KONU07_LAB, REGRESSORS
from core.labs.konu08 import KONU08_LAB
from core.labs.runner import LabState, batchable, execute, plot_key, run_lab, run_operations
from core.labs.sezgi_konu07 import COVERAGE, ERRORS, IMPORTANCE, KONU07_EXPERIMENTS, theoretical_power
from core.labs.sezgi_konu08 import JOINT, KONU08_EXPERIMENTS, LARGE_SAMPLE, OMITTED, estimate_correlation
from core.labs.spec import (
    INTERCEPT,
    OLS,
    CoefficientTable,
    HypothesisPlot,
    IntervalPlot,
    JointTest,
    LabSpec,
    LoadWooldridge,
    MonteCarlo,
    MultiChoice,
    NumberChoice,
    RegressionTable,
    ShowModel,
    SummaryTable,
)
from core.quiz.konu07 import KONU07_QUIZ
from core.quiz.konu08 import KONU08_QUIZ
from core.quiz.model import grade
from topics.lab_ui import _model_stat_text, coefficient_display, display_table, regression_display

LABS = (KONU07_LAB, KONU08_LAB)
EXPERIMENTS = KONU07_EXPERIMENTS + KONU08_EXPERIMENTS


def _printed(value: float, expected: float, decimals: int) -> bool:
    """Değer, notlarda ``decimals`` basamakla basılı sayıya yuvarlanır."""

    return abs(value - expected) <= 0.5 * 10 ** -decimals + 1e-12


@pytest.fixture(scope="module")
def wage1() -> pd.DataFrame:
    return W.load("wage1")


@pytest.fixture(scope="module")
def hprice1() -> pd.DataFrame:
    return W.load("hprice1")


@pytest.fixture(scope="module")
def wage_model(wage1):
    return smf.ols("wage ~ educ + exper + tenure", data=wage1).fit()


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
    assert sum(len(step.checks) for step in KONU07_LAB.steps) == 119
    assert sum(len(step.checks) for step in KONU08_LAB.steps) == 81
    assert [step.note.section for step in KONU07_LAB.steps] == ["7.4", "7.4", "7.5", "7.6", "7.7", "7.8", "7.9",
                                                                  "7.10", "7.11", "7.13"]
    assert [step.note.section for step in KONU08_LAB.steps] == ["8.5", "8.6", "8.6", "8.7", "8.8", "8.9", "8.12"]


@pytest.mark.parametrize("spec", LABS, ids=lambda s: s.topic_key)
def test_a_changed_choice_marks_only_its_own_step(spec: LabSpec) -> None:
    changes = _single_changes(spec)
    assert len(changes) >= (40 if spec is KONU07_LAB else 18)
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
            np.testing.assert_allclose(namespace[name].to_numpy(float), table.to_numpy(float), rtol=0, atol=1e-12,
                                       err_msg=f"{change} {name}")
        for name, value in state.scalars.items():
            assert float(namespace[name]) == pytest.approx(value, rel=1e-12, abs=1e-12), (change, name)
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
    """Metin boş değil, "nan" içermiyor, eksi sıfır ("−0,00") ve ondalıklı (hesaplanan) sayıya Türkçe ek ("0,35'ye")
    yazmıyor."""

    return (bool(text) and not re.search(r"\bnan\b", text.lower()) and not re.search(r"−0,0+(?![0-9])", text)
            and not re.search(r"\d,\d+'", text))


def _screen(operations, state: LabState, label) -> None:
    """Ekrandaki tablolar ve grafikler kurulabilir; tablo sütun adları tekildir."""

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


def test_konu07_notes_follow_the_chosen_specification() -> None:
    text = _note(KONU07_LAB, {}, 1)
    assert "|t| = 2,50" in text and "H₀ reddedilir" in text and "t = 1 olurdu" in text
    text = _note(KONU07_LAB, {"adim1_se": 1.0}, 1)
    assert "|t| = 0,30" in text and "H₀ reddedilemez" in text and "t = 1 olurdu" not in text
    text = _note(KONU07_LAB, {}, 2)
    assert "t = 11,68: yüzde 5 iki taraflı testte H₀ reddedilir" in text
    assert "H₀: β = 0,50 için t = 1,93: H₀ reddedilemez" in text and "deneyim ve kıdem sabitken" in text
    text = _note(KONU07_LAB, {}, 3)
    assert "t = 1,853, iki taraflı p = 0,064" in text
    assert "%5 düzeyinde p ≥ α olduğundan H₀: β = 0 reddedilemez" in text
    assert "%10 düzeyinde reddedilir; %5, %1 düzeyinde reddedilemez" in text and "Reddedememek deneyimin" in text
    text = _note(KONU07_LAB, {"adim3_alfa": "0.1"}, 3)
    assert "p < α olduğundan H₀: β = 0 reddedilir" in text and "Reddedememek" not in text
    assert "p < 0,001" in _note(KONU07_LAB, {"adim3_terim": "educ"}, 3)
    assert "[0,065; 0,535]" in _note(KONU07_LAB, {}, 4) and "sıfırı kapsamaz" in _note(KONU07_LAB, {}, 4)
    assert "sıfırı kapsar" in _note(KONU07_LAB, {"adim4_se": 1.0}, 4)
    text = _note(KONU07_LAB, {}, 5)
    assert "[0,498; 0,700]" in text and "Sıfırı kapsayan aralık: deneyim" in text
    assert ("0 aralıkta olmadığı için H₀ reddedilir; 0,50 aralıkta olduğu için H₀ reddedilemez; 0,75 aralıkta "
            "olmadığı için H₀ reddedilir") in text
    assert "Hiçbir aralık sıfırı kapsamıyor." in _note(KONU07_LAB, {"adim5_duzey": "0.9"}, 5)  # deneyim p = 0,064
    text = _note(KONU07_LAB, {}, 6)
    assert "tek taraflı p = 0,032, iki taraflı p = 0,064" in text
    assert "tek taraflı testte H₀ reddedilir; iki taraflı testte reddedilemez" in text
    text = _note(KONU07_LAB, {"adim6_yon": "sol"}, 6)
    assert "p = 0,968" in text and "alternatifin tersi yönde" in text
    assert "iki taraflı testte p = 0,064" in _note(KONU07_LAB, {"adim6_yon": "iki"}, 6)
    text = _note(KONU07_LAB, {}, 7)
    assert "526 − 3 − 1 = 522" in text and "sıfırdan farklı eğimler: eğitim, mevcut işverendeki kıdem." in text
    assert "526 − 4 − 1 = 521" in _note(KONU07_LAB, {"adim7_x": ("educ", "exper", "tenure", "numdep")}, 7)
    text = _note(KONU07_LAB, {}, 9)
    assert "2,396 dolar, yüzde 95 aralığı [1,993; 2,799]" in text
    assert "13,85 bin dolar (p = 0,128), aralık [−4,07; 31,77]" in text
    text = _note(KONU07_LAB, {}, 10)
    assert "eğitim katsayısı 0,599 olarak tahmin edilmiştir (SH = 0,051; t = 11,68; p < 0,001; yüzde 95 GA " \
           "[0,498; 0,700])" in text
    assert "ortalama 0,599 dolar daha yüksek" in text and "2,40 dolardır" in text and "ayrışmamaktadır" not in text
    text = _note(KONU07_LAB, {"adim10_terim": "exper"}, 10)
    assert "p = 0,064" in text and "ayrışmamaktadır" in text and "eğitim ve kıdem sabitken" in text


def test_konu08_notes_follow_the_chosen_specification() -> None:
    text = _note(KONU08_LAB, {}, 1)
    assert "Gözlenen F = 53,31; %5 düzeyinde kritik değer 3,01" in text and "ortak p < 0,001" in text
    assert "hepsinin ayrı ayrı sıfırdan farklı olduğunu göstermez" in text
    text = _note(KONU08_LAB, {"adim1_x": ("exper",)}, 1)
    assert "F = t² (§8.8)" in text and "ortak p" not in text and "reddedilemez" in text  # p = 0,064
    assert "genel anlamlılık testidir" in _note(KONU08_LAB, {"adim1_x": ("educ", "exper", "tenure")}, 1)
    assert "%1 düzeyinde kritik değer 4,65" in _note(KONU08_LAB, {"adim1_alfa": "0.01"}, 1)
    text = _note(KONU08_LAB, {}, 3)
    assert "[(5980,682 − 4966,303)/2] / [4966,303/522] = 53,31" in text
    assert "[(0,30642 − 0,16476)/2] / [(1 − 0,30642)/522] = 53,31" in text
    text = _note(KONU08_LAB, {}, 4)
    assert "Genel test F(3, 522) = 76,87, p < 0,001: eğitim, deneyim ve kıdem katsayılarının tamamının birlikte" in text
    text = _note(KONU08_LAB, {"adim4_x": ("numdep",)}, 4)
    assert "F(1, 524)" in text and "t istatistiğinin karesidir" in text
    text = _note(KONU08_LAB, {}, 5)
    assert "t = 11,6795, t² = 136,41; aynı hipotezin F testi F(1, 522) = 136,41" in text
    assert "t testinde p < 0,0001, F testinde p < 0,0001" in text
    text = _note(KONU08_LAB, {}, 6)
    assert "F(2, 84) = 6,61, p = 0,0022: yüzde 5 düzeyinde H₀ reddedilir" in text
    assert "Konut büyüklüğü sabitken arsa büyüklüğü ve yatak odası sayısı birlikte fiyat modeline katkı sağlar" in text
    assert "Yatak odası katsayısının ayrı p = 0,128" in text
    text = _note(KONU08_LAB, {"adim6_x": ("bdrms",)}, 6)
    assert "reddedilemez" in text and "F = t² (§8.8)" in text


# --- Adımların hesabı: bağımsız hesapla karşılaştırma -------------------------------------------------------

def test_konu07_t_statistics_critical_values_and_intervals_are_what_they_say(wage_model) -> None:
    for b, se, a, alpha in ((0.30, 0.12, 0.0, "0.05"), (-1.2, 0.4, 0.5, "0.01"), (0.05, 1.0, -1.0, "0.1")):
        state = run_operations(KONU07_LAB.resolve({"adim1_b": b, "adim1_se": se, "adim1_a": a,
                                                   "adim1_alfa": alpha}).operations_through(1))
        q = 1 - float(alpha) / 2
        assert state.scalars["t_ornek"] == pytest.approx((b - a) / se, rel=1e-12)
        table = state.tables["kritik_degerler"]["deger"]
        for df in (10, 20, 30, 60, 120):
            assert table[str(df)] == pytest.approx(stats.t.ppf(q, df), rel=1e-12)
        assert table["Çok büyük"] == pytest.approx(stats.norm.ppf(q), rel=1e-12)
        assert list(table) == sorted(table, reverse=True)  # sd büyüdükçe kritik değer küçülür
    for term in REGRESSORS:
        for a in (-0.3, 0.5, 0.6):  # kaydırıcının adımı 0,01
            s = run_operations(KONU07_LAB.resolve({"adim2_terim": term, "adim2_a": a}).operations_through(2)).scalars
            assert s["ta_2"] == pytest.approx(float(np.squeeze(wage_model.t_test(f"{term} = {a}").tvalue)), rel=1e-10)
            assert s["t0_2"] == pytest.approx(wage_model.tvalues[term], rel=1e-12)
    assert s["kritik_2"] == pytest.approx(stats.t.ppf(0.975, 522), rel=1e-12)
    for b, se, level in ((0.30, 0.12, "0.95"), (0.30, 0.12, "0.9"), (-0.5, 0.3, "0.99")):
        s = run_operations(KONU07_LAB.resolve({"adim4_b": b, "adim4_se": se, "adim4_duzey": level})
                           .operations_through(4)).scalars
        z = stats.norm.ppf((1 + float(level)) / 2)
        assert (s["alt4"], s["ust4"]) == pytest.approx((b - z * se, b + z * se), rel=1e-12)


def test_konu07_p_values_and_the_test_interval_equivalence(wage_model) -> None:
    for term in REGRESSORS:
        s = run_operations(KONU07_LAB.resolve({"adim3_terim": term}).operations_through(3)).scalars
        assert s["p3"] == pytest.approx(2 * stats.t.sf(abs(wage_model.tvalues[term]), 522), rel=1e-10)
        assert s["p3"] == pytest.approx(wage_model.pvalues[term], rel=1e-10)
    for level in ("0.9", "0.95", "0.99"):
        state = run_operations(KONU07_LAB.resolve({"adim5_duzey": level}).operations_through(5))
        table = state.tables["tablo74"]
        interval = wage_model.conf_int(alpha=1 - float(level))
        for term in REGRESSORS:
            assert (table.loc[term, "alt"], table.loc[term, "ust"]) == pytest.approx(tuple(interval.loc[term]),
                                                                                     rel=1e-12)
            # Denklem 7.9: a aralığın dışındaysa iki taraflı H₀: β = a testi reddeder (ve tersi).
            for a in np.linspace(table.loc[term, "alt"] - 0.05, table.loc[term, "ust"] + 0.05, 23):
                p = float(np.squeeze(wage_model.t_test(f"{term} = {a}").pvalue))
                outside = not table.loc[term, "alt"] <= a <= table.loc[term, "ust"]
                if abs(p - (1 - float(level))) > 1e-9:
                    assert outside == (p < 1 - float(level)), (term, level, a)
        plot = state.plots[plot_key(next(op for op in KONU07_LAB.resolve({"adim5_duzey": level}).step(5).operations
                                        if type(op).__name__ == "CoefficientPlot"))]
        assert list(plot["terim"]) == list(REGRESSORS)
        np.testing.assert_allclose(plot[["alt", "ust"]].to_numpy(), table[["alt", "ust"]].to_numpy(), rtol=0, atol=0)
    for term in REGRESSORS:
        t = wage_model.tvalues[term]
        values = {}
        for direction in ("sag", "sol", "iki"):
            s = run_operations(KONU07_LAB.resolve({"adim6_terim": term, "adim6_yon": direction})
                               .operations_through(6)).scalars
            values[direction] = s["p_secilen"]
            assert s["p_iki6"] == pytest.approx(wage_model.pvalues[term], rel=1e-10)
        assert values["sag"] == pytest.approx(stats.t.sf(t, 522), rel=1e-10)
        assert values["sag"] + values["sol"] == pytest.approx(1, abs=1e-12)
        assert values["iki"] == pytest.approx(2 * min(values["sag"], values["sol"]), rel=1e-10)


def test_konu07_output_magnitude_and_report_steps_compute_what_they_say(wage1, hprice1, wage_model) -> None:
    for outcome, chosen in (("lwage", ("educ", "exper", "tenure")), ("wage", ("educ", "numdep")),
                            ("lwage", ("tenure",))):
        state = run_operations(KONU07_LAB.resolve({"adim7_bagimli": outcome, "adim7_x": chosen})
                               .operations_through(7))
        fit = smf.ols(f"{outcome} ~ " + " + ".join(chosen), data=wage1).fit()
        result = state.models["cikti"]
        for attribute in ("params", "bse", "tvalues", "pvalues"):
            np.testing.assert_allclose(getattr(result, attribute).to_numpy(), getattr(fit, attribute).to_numpy(),
                                       rtol=1e-10, atol=1e-14)
        assert int(result.df_resid) == 526 - len(chosen) - 1
    for years in (1, 4, 8):
        s = run_operations(KONU07_LAB.resolve({"adim9_fark": years}).operations_through(9)).scalars
        low, high = wage_model.conf_int().loc["educ"]
        assert (s["fark_tahmin"], s["fark_alt"], s["fark_ust"]) == pytest.approx(
            (years * wage_model.params["educ"], years * low, years * high), rel=1e-12)
    table = run_operations(KONU07_LAB.operations_through(9)).tables["tablo76"]
    house = smf.ols("price ~ lotsize + sqrft + bdrms", data=hprice1).fit()
    np.testing.assert_allclose(table["katsayi"], house.params[["lotsize", "sqrft", "bdrms"]], rtol=1e-12)
    np.testing.assert_allclose(table["t"], house.tvalues[["lotsize", "sqrft", "bdrms"]], rtol=1e-10)
    for term in REGRESSORS:
        s = run_operations(KONU07_LAB.resolve({"adim10_terim": term}).operations_through(10)).scalars
        assert (s["b10"], s["sh10"], s["p10"]) == pytest.approx(
            (wage_model.params[term], wage_model.bse[term], wage_model.pvalues[term]), rel=1e-10)
        assert s["dort10"] == pytest.approx(4 * wage_model.params[term], rel=1e-12)


def _restricted_f(frame: pd.DataFrame, outcome: str, regressors, dropped) -> tuple[float, float]:
    """SSR biçimi ile F ve p (Denklem 8.x): kısıtlı model aynı gözlemlerle yeniden tahmin edilir."""

    kept = [term for term in regressors if term not in dropped]
    full = smf.ols(f"{outcome} ~ " + " + ".join(regressors), data=frame).fit()
    small = smf.ols(f"{outcome} ~ " + (" + ".join(kept) if kept else "1"), data=frame).fit()
    q = len(dropped)
    f = ((small.ssr - full.ssr) / q) / (full.ssr / full.df_resid)
    return f, float(stats.f.sf(f, q, full.df_resid))


def test_konu08_joint_tests_match_f_test_and_the_ssr_formula(wage1, hprice1, wage_model) -> None:
    for terms in (("exper", "tenure"), ("educ",), ("exper",), ("educ", "exper", "tenure"), ("educ", "tenure")):
        for alpha in ("0.1", "0.05", "0.01"):
            s = run_operations(KONU08_LAB.resolve({"adim1_x": terms, "adim1_alfa": alpha})
                               .operations_through(1)).scalars
            reference = wage_model.f_test(", ".join(f"{term} = 0" for term in terms))
            assert s["F_ortak"] == pytest.approx(float(reference.fvalue), rel=1e-10), terms
            assert s["p_ortak"] == pytest.approx(float(reference.pvalue), rel=1e-8, abs=1e-300), terms
            f, p = _restricted_f(wage1, "wage", REGRESSORS, terms)
            assert s["F_ortak"] == pytest.approx(f, rel=1e-10) and s["p_ortak"] == pytest.approx(p, rel=1e-8)
            assert s["pay_sd"] == len(terms) and s["sd_artik"] == 522
            assert s["kritik_F"] == pytest.approx(stats.f.ppf(1 - float(alpha), len(terms), 522), rel=1e-12)
            if len(terms) == 1:
                assert s["F_ortak"] == pytest.approx(wage_model.tvalues[terms[0]] ** 2, rel=1e-10)
    s = run_operations(KONU08_LAB.operations_through(2)).scalars
    assert (s["genel_F"], s["genel_p"]) == pytest.approx((wage_model.fvalue, wage_model.f_pvalue), rel=1e-10)
    assert s["sd_model"] == 3 and s["genel_p_olcek"] == pytest.approx(1e41 * wage_model.f_pvalue, rel=1e-10)
    for dropped in (("exper", "tenure"), ("educ",), ("educ", "tenure"), ("tenure",)):
        s = run_operations(KONU08_LAB.resolve({"adim3_x": dropped}).operations_through(3)).scalars
        f, _ = _restricted_f(wage1, "wage", REGRESSORS, dropped)
        assert s["F_ssr"] == pytest.approx(f, rel=1e-10) and s["F_r2"] == pytest.approx(f, rel=1e-10), dropped
        assert s["ssr_k"] >= s["ssr_s"] and s["r2_k"] <= s["r2_s"]  # değişken çıkarmak uyumu artırmaz
    for regressors in (REGRESSORS, ("numdep",), ("educ", "numdep"), ("educ", "exper", "tenure", "numdep")):
        s = run_operations(KONU08_LAB.resolve({"adim4_x": regressors}).operations_through(4)).scalars
        fit = smf.ols("wage ~ " + " + ".join(regressors), data=wage1).fit()
        assert s["F_genel"] == pytest.approx(fit.fvalue, rel=1e-10)
        assert s["F_genel_r2"] == pytest.approx(fit.fvalue, rel=1e-10)
        f, p = _restricted_f(wage1, "wage", regressors, regressors)  # kısıtlı model yalnız sabitli
        assert s["F_genel"] == pytest.approx(f, rel=1e-10) and s["p_genel"] == pytest.approx(p, rel=1e-8)
    for term in REGRESSORS:
        s = run_operations(KONU08_LAB.resolve({"adim5_terim": term}).operations_through(5)).scalars
        assert s["F_tek"] == pytest.approx(s["t_kare"], rel=1e-10) and s["p_tek"] == pytest.approx(s["p_t_tek"],
                                                                                                    rel=1e-8)
    for dropped in (("lotsize", "bdrms"), ("sqrft",), ("bdrms",), ("lotsize", "sqrft")):
        s = run_operations(KONU08_LAB.resolve({"adim6_x": dropped}).operations_through(6)).scalars
        f, p = _restricted_f(hprice1, "price", ("lotsize", "sqrft", "bdrms"), dropped)
        assert s["F_h"] == pytest.approx(f, rel=1e-10) and s["p_h"] == pytest.approx(p, rel=1e-8), dropped
    s = run_operations(KONU08_LAB.operations_through(7)).scalars
    f, _ = _restricted_f(wage1, "lwage", REGRESSORS, ("exper", "tenure"))
    assert s["ortak_F2"] == pytest.approx(f, rel=1e-10) and round(f, 2) == 49.69  # eski metin 29,44 yazıyordu


# --- Motor: yeni işlemler ---------------------------------------------------------------------------------

def test_t_and_f_functions_match_scipy_and_render_in_both_languages() -> None:
    from core.codegen import python_gen, r_gen

    x, df, df2 = 1.7, 12.0, 30.0
    cases = {
        "tcdf": (E.tcdf(x, df), stats.t.cdf(x, df)), "tsf": (E.tsf(x, df), stats.t.sf(x, df)),
        "tinv": (E.tinv(0.975, df), stats.t.ppf(0.975, df)), "fsf": (E.fsf(x, df, df2), stats.f.sf(x, df, df2)),
        "finv": (E.finv(0.95, df, df2), stats.f.ppf(0.95, df, df2)), "norminv": (E.norminv(0.95), stats.norm.ppf(0.95)),
    }
    for name, (expression, expected) in cases.items():
        assert float(E.evaluate(expression)) == pytest.approx(expected, rel=1e-14), name
    python = E.Dialect(variable=str, functions=python_gen._FUNCTIONS, power="**")
    r = E.Dialect(variable=str, functions=r_gen._FUNCTIONS, power="^")
    assert E.render(E.tsf(E.var("t"), 522), python) == "stats.t.sf(t, 522)"
    assert E.render(E.tsf(E.var("t"), 522), r) == "pt(t, 522, lower.tail = FALSE)"
    assert E.render(E.fsf(E.var("f"), 2, 522), r) == "pf(f, 2, 522, lower.tail = FALSE)"
    assert E.render(E.finv(0.95, 2, 522), r) == "qf(0.95, 2, 522)"
    assert E.format_number(1e41) == "1e+41" and E.format_number(3.0) == "3" and E.format_number(0.975) == "0.975"
    assert float(eval(E.format_number(1e41))) == 1e41  # Python ve R'de aynı sayı


def test_hypothesis_plot_layout_regions_and_labels() -> None:
    def layout(distribution, statistic, alternative="iki", alpha=0.05, df=522, df2=None):
        op = HypothesisPlot(distribution, statistic, df, "Deneme", "x", alpha=alpha, alternative=alternative, df2=df2)
        return op, I.hypothesis_layout(op, statistic, df, df2)

    op, data = layout("t", 1.853)  # Şekil 7.2
    critical = stats.t.ppf(0.975, 522)
    assert data["kritik"] == pytest.approx((-critical, critical)) and data["sinirlar"] == (-4.0, 4.0)
    assert data["reddetme"] == [(-4.0, -critical), (critical, 4.0)]
    assert data["p_alani"] == [(-4.0, -1.853), (1.853, 4.0)] and not data["disarida"]
    assert data["p"] == pytest.approx(2 * stats.t.sf(1.853, 522))
    labels = hypothesis_labels(op, data)
    assert labels["kritik"] == "Kritik değer ±1,96" and labels["gozlenen"] == "Gözlenen t = 1,85"
    op, data = layout("t", 11.68)  # eğitim: eksenin dışında
    assert data["sinirlar"] == (-6.0, 6.0) and data["disarida"] and data["p_alani"] == []
    assert hypothesis_labels(op, data)["gozlenen"] == "Gözlenen t = 11,68 (eksenin dışında)"
    op, data = layout("t", 1.853, "sol", alpha=0.1)
    assert data["kritik"] == pytest.approx((stats.t.ppf(0.1, 522),)) and data["reddetme"][0][0] == -4.0
    assert data["p"] == pytest.approx(stats.t.cdf(1.853, 522)) and data["p_alani"] == [(-4.0, 1.853)]
    op, data = layout("f", 53.31, "sag", df=2, df2=522)  # Şekil 8.1
    f_critical = stats.f.ppf(0.95, 2, 522)
    assert data["kritik"] == pytest.approx((f_critical,)) and data["sinirlar"] == pytest.approx((0, 6 * f_critical))
    assert data["disarida"] and data["p"] == pytest.approx(stats.f.sf(53.31, 2, 522))
    assert hypothesis_labels(op, data)["kritik"] == "Kritik değer 3,01"
    op, data = layout("f", 2.0, "sag", df=2, df2=84)
    assert data["sinirlar"][1] == pytest.approx(2.4 * stats.f.ppf(0.95, 2, 84)) and not data["disarida"]
    for bad in ({"distribution": "z"}, {"distribution": "f", "alternative": "sag"},  # F: df2 gerekir
                {"distribution": "f", "df2": 5, "alternative": "iki"},  # F testi yalnız üst kuyruk
                {"alternative": "yukari"}, {"alpha": 1.5}):
        with pytest.raises(ValueError):
            HypothesisPlot(**{"distribution": "t", "statistic": 1.0, "df": 10, "title": "x", "x_label": "x", **bad})


def test_inference_operations_validate_their_input() -> None:
    with pytest.raises(ValueError, match="Güven düzeyi"):
        CoefficientTable("m", ("educ",), "t", "t", level=95)
    for terms in ((), ("educ", "educ"), (INTERCEPT,)):
        with pytest.raises(ValueError, match="Ortak testte"):
            JointTest("F", "p", "m", terms, "deneme")
    with pytest.raises(ValueError, match="her model sütunu"):
        RegressionTable((("(1)", "m"), ("(2)", "n")), ("educ",), "t", "t", extra=(("F", "F", ("F1",)),))
    state = run_operations((LoadWooldridge("wage1", "veri"), OLS("m", "wage1", "wage", ("educ",), "m")))
    with pytest.raises(KeyError, match="terim yok"):
        execute(JointTest("F", "p", "m", ("exper",), "modelde olmayan terim"), state)


def test_extra_rows_table_and_model_statistics_are_shown_in_turkish_format() -> None:
    state = run_operations(KONU08_LAB.operations_through(3))
    op = next(op for op in KONU08_LAB.step(3).operations if isinstance(op, RegressionTable))
    assert op.terms == () and op.extra_decimals == 3
    shown = regression_display(op, state, KONU08_LAB.label).set_index("Değişken")
    restricted, unrestricted = (heading for heading, _ in op.models)
    assert list(shown.index) == ["SSR", "R²", "Gözlem sayısı"]
    assert (shown.loc["SSR", restricted], shown.loc["SSR", unrestricted]) == ("5980,682", "4966,303")
    assert (shown.loc["R²", restricted], shown.loc["Gözlem sayısı", unrestricted]) == ("0,165", "526")
    state = run_operations(KONU08_LAB.operations_through(7))
    op = next(op for op in KONU08_LAB.step(7).operations if isinstance(op, RegressionTable))
    shown = regression_display(op, state, KONU08_LAB.label).set_index("Değişken")
    assert shown.loc["Deneyim ve kıdem ortak F", "(2) ln(Ücret)"] == "49,69"
    assert shown.loc["Ortak test p-değeri", "(1) Ücret"] == "< 0,001"
    python, r = render_script(KONU08_LAB, "Python"), render_script(KONU08_LAB, "R")
    assert 'F_ortak_test = m.f_test("exper = 0, tenure = 0")' in python
    assert "F_ortak_kisitli <- update(m, . ~ . - exper - tenure)" in r and "anova(F_ortak_kisitli, m)" in r
    assert "genel_p_olcek = genel_p * 1e+41" in python and "genel_p_olcek <- genel_p * 1e+41" in r
    assert _model_stat_text("f_p", 3.41e-41) == "< 0,001" and _model_stat_text("f", 76.8733) == "76,87"
    assert p_text(1e-5) == "< 0,001" and p_text(0.99999) == "> 0,999" and p_text(0.0641) == "0,064"
    assert p_text(0.00004, 4) == "< 0,0001" and p_text(0.9996) == "> 0,999" and p_text(0.9994) == "0,999"


def test_interval_plot_reference_line_in_the_app_and_in_both_languages() -> None:
    state = run_operations(IMPORTANCE.build(IMPORTANCE.defaults()))
    op = next(op for op in IMPORTANCE.build(IMPORTANCE.defaults()) if isinstance(op, IntervalPlot))
    assert op.reference == (0.0, "Sıfır: aralık sıfırı kesiyorsa H₀ reddedilemez")
    figure = figure_for(op, state)
    dotted = [shape for shape in figure.layout.shapes if shape.line.dash == "dot"]
    assert len(dotted) == 1 and dotted[0].x0 == 0
    assert any(trace.name == op.reference[1] for trace in figure.data)
    low, high = figure.layout.xaxis.range
    assert low < 0 < high  # başvuru çizgisi aralıkların dışında kalsa da eksende
    data = state.plots[plot_key(op)]
    table = state.tables[op.table].head(op.rows)
    np.testing.assert_array_equal(data["kapsiyor"], (table["alt"] <= op.truth) & (op.truth <= table["ust"]))
    python, r = render_script(IMPORTANCE.spec(IMPORTANCE.defaults()), "Python"), render_script(
        IMPORTANCE.spec(IMPORTANCE.defaults()), "R")
    assert python.count('ax.axvline(0, color="#6B4C9A", linestyle=":", linewidth=2, label="Sıfır:') == 2
    assert r.count('abline(v = 0, lty = 3, col = "#6B4C9A", lwd = 2)') == 2


def test_scatter_plot_reads_a_monte_carlo_result_table() -> None:
    state = run_operations(JOINT.build(JOINT.defaults()))
    op = next(op for op in JOINT.build(JOINT.defaults()) if type(op).__name__ == "ScatterPlot")
    table = state.tables["h0_yanlis"]
    figure = figure_for(op, state)
    points = next(trace for trace in figure.data if trace.mode == "markers")
    np.testing.assert_allclose(points.x, table["b1"], rtol=0, atol=0)
    np.testing.assert_allclose(points.y, table["b2"], rtol=0, atol=0)
    assert estimate_correlation(state) == pytest.approx(np.corrcoef(table["b1"], table["b2"])[0, 1])


# --- Sezgi deneyleri ------------------------------------------------------------------------------------

def _extremes(experiment) -> list[dict[str, float]]:
    settings = [experiment.defaults()]
    for parameter in experiment.parameters:
        for value in (parameter.minimum, parameter.maximum):
            settings.append(dict(experiment.defaults(), **{parameter.key: value}))
    return settings


def test_block_experiments_use_the_batch_path() -> None:
    for experiment in EXPERIMENTS:
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
        assert len(metrics) == 4 and all(len(metric.value) <= 10 for metric in metrics), parameters
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


def test_konu07_coverage_experiment_reproduces_table_7_1() -> None:
    state = run_operations(COVERAGE.build(COVERAGE.defaults()))
    table = state.tables["tablo71"]
    for column, printed, decimals in (("ortalama", 0.499, 3), ("std_sapma", 0.149, 3), ("ort_sh", 0.144, 3),
                                      ("kapsama", 0.9460, 4)):
        assert _printed(table[column].iloc[0], printed, decimals), column
    first = state.plots[plot_key(next(op for op in COVERAGE.build(COVERAGE.defaults()) if isinstance(op, IntervalPlot)))]
    assert int(first["kapsiyor"].sum()) == 24 and len(first) == 25  # Şekil 7.3
    runs = state.tables["tekrarlar"]
    critical = stats.t.ppf(0.975, 48)
    np.testing.assert_allclose(runs["ust"] - runs["b1"], critical * runs["sh"], rtol=1e-12)
    for n, sigma, level in ((20, 2.0, 90), (400, 0.5, 99)):
        s = run_operations(COVERAGE.build(dict(COVERAGE.defaults(), n=n, sigma=sigma, duzey=level))).scalars
        assert s["kapsama"] == pytest.approx(level / 100, abs=4 * math.sqrt(level / 100 * (1 - level / 100) / 5000))
        assert s["ort_sh"] == pytest.approx(s["sd_b1"], rel=0.06)  # raporlanan SH gerçek yayılımı izler


def _simulated_power(beta: float, n: int, alpha: float, reps: int = 20_000, chunk: int = 5_000) -> float:
    """Motordan bağımsız, vektörel benzetimle iki taraflı t testinin gücü (Y = βX + u, X, u ~ N(0, 1))."""

    rng = np.random.default_rng(20260929)
    critical = stats.t.ppf(1 - alpha / 2, n - 2)
    rejected = 0
    for _ in range(reps // chunk):
        x = rng.standard_normal((chunk, n))
        y = beta * x + rng.standard_normal((chunk, n))
        xc, yc = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
        sxx = np.einsum("ij,ij->i", xc, xc)
        slope = np.einsum("ij,ij->i", xc, yc) / sxx
        residual = yc - slope[:, None] * xc
        se = np.sqrt(np.einsum("ij,ij->i", residual, residual) / (n - 2) / sxx)
        rejected += int(np.sum(np.abs(slope / se) > critical))
    return rejected / reps


def test_theoretical_power_is_finite_and_matches_an_independent_simulation() -> None:
    for beta in np.linspace(-1, 1, 41):
        for n in (10, 30, 200):
            for alpha in (0.01, 0.05, 0.10):
                value = theoretical_power(float(beta), n, alpha)
                assert np.isfinite(value) and alpha - 1e-9 <= value <= 1 + 1e-12, (beta, n, alpha)
    assert theoretical_power(0.0, 30, 0.05) == pytest.approx(0.05, abs=1e-12)
    assert theoretical_power(0.3, 30, 0.05) == pytest.approx(theoretical_power(-0.3, 30, 0.05), abs=1e-15)
    for beta, n, printed in ((0.3, 30, 0.34), (0.3, 100, 0.83), (1.0, 10, 0.69)):
        value = theoretical_power(beta, n, 0.05)
        assert round(value, 2) == printed  # d07: ~%34 ve ~%83
        assert value == pytest.approx(_simulated_power(beta, n, 0.05), abs=0.012), (beta, n)


def test_konu07_error_and_importance_experiments_match_their_theory() -> None:
    s = run_operations(ERRORS.build(ERRORS.defaults())).scalars  # β₁ = 0: H₀ doğru
    assert s["ret"] == pytest.approx(0.05, abs=4 * math.sqrt(0.05 * 0.95 / 4000))
    state = run_operations(ERRORS.build(ERRORS.defaults()))
    assert state.tables["testler"]["medyan_p"].iloc[0] == pytest.approx(0.5, abs=0.03)  # p ~ düzgün
    for beta, n in ((0.3, 30), (0.3, 100), (-0.2, 200)):
        s = run_operations(ERRORS.build(dict(ERRORS.defaults(), beta=beta, n=n))).scalars
        expected = theoretical_power(beta, n, 0.05)
        assert s["ret"] == pytest.approx(expected, abs=4 * math.sqrt(expected * (1 - expected) / 4000)), (beta, n)
    state = run_operations(IMPORTANCE.build(IMPORTANCE.defaults()))
    table, s = state.tables["onem"], state.scalars
    assert (round(s["ret_a"], 3), round(s["ret_b"], 3)) == (0.894, 0.442)
    assert tuple(table["ort_genislik"].round(3)) == (0.062, 1.187)
    for rate, (beta, n) in ((s["ret_a"], (0.05, 4000)), (s["ret_b"], (0.5, 15))):
        expected = theoretical_power(beta, n, 0.05)
        assert rate == pytest.approx(expected, abs=4 * math.sqrt(expected * (1 - expected) / 500)), (beta, n)
    # Genişlik ≈ 2·t·σ/√n: A dar (büyük n), B geniş (küçük n)
    assert table["ort_genislik"].iloc[0] == pytest.approx(2 * 1.96 / math.sqrt(4000), rel=0.02)


def test_konu08_large_sample_experiment_reproduces_table_8_3() -> None:
    state = run_operations(LARGE_SAMPLE.build(LARGE_SAMPLE.defaults()))
    table = state.tables["tablo83"]
    expected = {"ret_orani": (0.046, 0.043, 0.053), "t_sd": (1.038, 0.985, 0.998), "carpiklik": (-0.006, -0.029, 0.047)}
    for column, values in expected.items():
        for row, value in zip(table.index, values):
            assert _printed(table.loc[row, column], value, 3), (row, column)
    assert list(table.index) == ["n = 25", "n = 100", "n = 500"] and list(table["tekrar"]) == [4000] * 3
    s = run_operations(LARGE_SAMPLE.build(dict(LARGE_SAMPLE.defaults(), k=0.25))).scalars
    for n in (25, 100, 500):
        assert s[f"ret_{n}"] == pytest.approx(0.05, abs=0.02), n  # çok çarpık hatada da yaklaşık geçerli


def test_konu08_joint_experiment_matches_its_theory() -> None:
    state = run_operations(JOINT.build(JOINT.defaults()))
    s = state.scalars
    assert (round(s["F_dogru"], 3), round(s["t_dogru"], 3)) == (0.049, 0.072)  # e04: ρ = 0,9'da %7,2
    assert (s["F_yanlis"], s["yalniz_yanlis"]) == pytest.approx((0.7155, 0.521), abs=0.0006)  # %71,5 ve %52,1
    assert estimate_correlation(state) == pytest.approx(-0.9, abs=0.03)  # d01: tahminler ters yönde
    s = run_operations(JOINT.build(dict(JOINT.defaults(), rho=0.0))).scalars
    familywise = 1 - 0.95 ** 2  # bağımsız iki test: e04
    assert s["t_dogru"] == pytest.approx(familywise, abs=4 * math.sqrt(familywise * (1 - familywise) / 2000))
    assert s["F_dogru"] == pytest.approx(0.05, abs=4 * math.sqrt(0.05 * 0.95 / 2000))


def test_konu08_omitted_variable_experiment_matches_its_theory() -> None:
    state = run_operations(OMITTED.build(OMITTED.defaults()))
    table = state.tables["buyuk_n"]
    assert tuple(table["kisa_kapsama"].round(3)) == (0.664, 0.114, 0.0, 0.0)  # d06
    assert all(value == pytest.approx(0.95, abs=0.03) for value in table["uzun_kapsama"])
    assert table["kisa_ort"].iloc[-1] == pytest.approx(0.5 + 0.5 * 0.5, abs=0.01)  # e05: β + γρ
    assert list(table["kisa_sd"]) == sorted(table["kisa_sd"], reverse=True)  # yayılım n ile küçülür
    summary = next(op for op in OMITTED.build(OMITTED.defaults()) if isinstance(op, SummaryTable))
    shown = display_table(summary, table, OMITTED.label)
    assert list(shown["n"]) == ["50", "200", "800", "2.000"] and list(shown["Örneklem"])[-1] == "n = 2.000"
    for gamma, rho in ((0.0, 0.5), (0.5, 0.0)):
        table = run_operations(OMITTED.build(dict(OMITTED.defaults(), gamma=gamma, rho=rho))).tables["buyuk_n"]
        assert all(value == pytest.approx(0.95, abs=0.03) for value in table["kisa_kapsama"]), (gamma, rho)


# --- Kendini sına: yazım çeşitleri ve sayılar -----------------------------------------------------------------

EQUATIONS = {
    ("konu07", "e01"): (("a + c*s", "a + c s", "c*s + a", "a + c\\,s", "a + c \\operatorname{se}(\\widehat\\beta)",
                         "a + c·SH", "a + c se(\\hat\\beta_1)", "a + c*sh"),
                        ("a - c*s", "c*s", "a + s", "a + c/s")),
    ("konu07", "e02"): (("(U - L)/(2*c)", "\\frac{U-L}{2c}", "(U-L)/2/c", "0.5*(U - L)/c", "(U − L)/(2c)"),
                        ("(U - L)/2*c", "(U + L)/(2*c)", "(U - L)/c", "U - L")),
    ("konu07", "e03"): (("sqrt(hkt/(n-k-1))", "\\sqrt{\\frac{\\text{HKT}}{n-k-1}}", "√(HKT/(n−k−1))",
                         "sqrt(SSR/(n - k - 1))", "(HKT/(n-k-1))^(1/2)"),
                        ("sqrt(hkt/(n-k))", "hkt/(n-k-1)", "sqrt(hkt/n)", "sqrt(hkt)/(n-k-1)")),
    ("konu07", "e04"): (("1 - p/2", "1 - p_2/2", "1-\\frac{p_2}{2}", "(2 - p2)/2", "1 − p₂/2"),
                        ("p/2", "1 - p", "2*p", "p")),
    ("konu07", "e05"): (("100*(b - c*s)*d", "100(\\widehat\\beta - c s)\\Delta x", "100 (β̂ − c·se) Δx",
                         "100*d*b - 100*d*c*s", "100 (\\hat\\beta_1 - c \\operatorname{se}(\\hat\\beta_1)) \\Delta x"),
                        ("100*(b + c*s)*d", "(b - c*s)*d", "100*(b - c*s)", "100*b*d")),
    ("konu08", "e01"): (("c*k/(c*k + n - k - 1)", "\\frac{ck}{ck + n - k - 1}", "(c k)/(c k + n - k - 1)",
                         "1/(1 + (n-k-1)/(c*k))"),
                        ("c*k/(n - k - 1)", "c*k/(c*k + n - k)", "k/(c*k + n - k - 1)")),
    ("konu08", "e02"): (("((u - r)/q)/((1 - u)/d)", "\\frac{(R^2_{UR} - R^2_{R})/q}{(1 - R^2_{UR})/d}",
                         "(R2_UR - R2_R)*d/(q*(1 - R2_UR))", "((R_ur^2 - R_r^2)/q)/((1-R_ur^2)/d)",
                         "\\frac{(R^2_{\\text{UR}} - R^2_{\\text{R}})/q}{(1 - R^2_{\\text{UR}})/d}"),
                        ("((u - r)/q)/((1 - r)/d)", "(u - r)/((1-u)/d)", "((r - u)/q)/((1-u)/d)")),
    ("konu08", "e03"): (("((b - a)/s)^2", "\\left(\\frac{\\widehat\\beta - a}{s}\\right)^2", "(b-a)^2/s^2",
                         "((β̂ − a)/se)^2", "((\\hat\\beta_1 - a)/\\operatorname{se}(\\hat\\beta_1))^2"),
                        ("(b - a)/s", "((b + a)/s)^2", "(b - a)^2/s")),
    ("konu08", "e04"): (("1 - (1 - alpha)^m", "1-(1-\\alpha)^m", "1 − (1 − α)^m"),
                        ("1 - (1 - alpha)*m", "alpha^m", "m*alpha", "(1 - alpha)^m")),
    ("konu08", "e05"): (("b + g*r", "\\beta + \\gamma\\rho", "β₁ + γρ", "beta_1 + gamma*rho", "\\beta_X + \\gamma \\rho"),
                        ("b + g", "b*g*r", "b + r", "g*r")),
}
QUIZZES = {"konu07": KONU07_QUIZ, "konu08": KONU08_QUIZ}


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
    ("konu07", "b01"): ((["−2,67", "14,00"], ["-2.67", "14"]), (["2,67", "14"], ["-2,67", "-14"])),
    ("konu07", "b02"): ((["0,064", "0,836"], ["0.064", "0.836"]), (["0,156", "0,744"], ["0,836", "0,064"])),
    ("konu07", "b03"): ((["0,020", "0,099"], ["0.02", "0.099"]), (["0,010", "0,049"], ["0,0099", "0,0495"])),
    ("konu07", "b04"): ((["-90", "-120"], ["−90", "−120"]), (["90", "120"], ["-90", "-60"])),
    ("konu07", "b05"): ((["0,040", "6400"], ["0.04", "6.400"]), (["0,020", "6400"], ["0,040", "3200"])),
    ("konu08", "b01"): ((["200", "9,09"], ["200", "9.09"]), (["201", "9,09"], ["200", "10,00"])),
    ("konu08", "b02"): ((["7160,4", "2194,1"], ["7.160,4", "2.194,1"]), (["4966,3", "2194,1"], ["7160,4", "1014,4"])),
    ("konu08", "b03"): ((["3,86", "6,68"], ["3.86", "6.68"]), (["1,96", "2,58"], ["3,84", "6,63"])),
    ("konu08", "b04"): ((["322955,9", "7,39"], ["322.956", "7,39"]), (["348053,4", "7,39"], ["322955,9", "15,74"])),
    ("konu08", "b05"): ((["0,10", "1,00"], ["0.1", "1"]), (["0,05", "1,00"], ["0,10", "0,50"])),
}


@pytest.mark.parametrize("topic, key", sorted(BLANKS))
def test_blank_questions_read_turkish_numbers(topic: str, key: str) -> None:
    question = QUIZZES[topic].question(key)
    accepted, rejected = BLANKS[(topic, key)]
    for response in accepted:
        assert grade(question, response).correct, (topic, key, response)
    for response in rejected:
        assert not grade(question, response).correct, (topic, key, response)


def test_konu07_quiz_numbers_are_computed_not_typed(wage1, hprice1, wage_model) -> None:
    sigma = math.sqrt(wage_model.ssr / wage_model.df_resid)
    assert (round(wage1["wage"].std(), 3), round(sigma, 3), round(wage_model.bse["educ"], 3)) == (3.693, 3.084, 0.051)
    assert round(wage_model.ssr, 3) == 4966.303 and round(math.sqrt(4966.303 / 522), 3) == 3.084  # e03
    assert [round(stats.t.ppf(0.975, df), 3) for df in (20, 25, 23)] == [2.086, 2.060, 2.069]  # k03
    assert round(stats.norm.ppf(0.95), 3) == 1.645 and round(stats.norm.sf(1.80), 3) == 0.036  # k05
    assert round(2 * stats.norm.sf(1.80), 3) == 0.072
    assert round(0.84 / 2.10, 2) == 0.40  # k06
    assert round((0.599 - 0.50) / 0.0513, 2) == 1.93 and round(wage_model.tvalues["educ"], 3) == 11.679  # d04
    house = smf.ols("price ~ lotsize + sqrft + bdrms", data=hprice1).fit()
    low, high = house.conf_int().loc["bdrms"]
    assert (round(house.pvalues["bdrms"], 3), round(low, 2), round(high, 2)) == (0.128, -4.07, 31.77)  # d05
    assert (round(0.95 ** 20, 2), round(1 - 0.95 ** 20, 2)) == (0.36, 0.64)  # d06
    assert round(1 - theoretical_power(0.3, 30, 0.05), 2) == pytest.approx(2 / 3, abs=0.02)  # d01: yaklaşık üçte iki
    assert round((0.84 - 1) / 0.06, 2) == -2.67 and round(0.84 / 0.06, 2) == 14.00  # b01
    assert (round(0.45 - 2.576 * 0.15, 3), round(0.45 + 2.576 * 0.15, 3)) == (0.064, 0.836)  # b02
    assert (round(0.45 - 1.96 * 0.15, 3), round(0.45 + 1.96 * 0.15, 3)) == (0.156, 0.744)
    assert (round(stats.norm.cdf(-2.33), 4), round(stats.norm.sf(1.65), 4)) == (0.0099, 0.0495)  # b03
    assert (round(2 * 0.0099, 3), round(2 * 0.0495, 3)) == (0.020, 0.099)
    assert round(0.080 / math.sqrt(1600 / 400), 3) == 0.040 and round(400 * (0.080 / 0.020) ** 2) == 6400  # b05
    assert round(1.96 * 0.12, 3) == 0.235  # e01
    assert round(stats.t.ppf(0.975, 522), 3) == 1.965 and round((0.700 - 0.498) / (2 * 1.965), 3) == 0.051  # e02
    assert round(1 - 0.064 / 2, 3) == 0.968  # e04
    logs = smf.ols("lwage ~ educ + exper + tenure", data=wage1).fit()
    assert (round(logs.params["tenure"], 3), round(logs.bse["tenure"], 3)) == (0.022, 0.003)  # e05: Tablo 7.5
    assert round(100 * 0.022 * 4, 1) == 8.8
    assert (round(100 * (0.022 - 1.965 * 0.003) * 4, 1), round(100 * (0.022 + 1.965 * 0.003) * 4, 1)) == (6.4, 11.2)
    state = run_operations(COVERAGE.build(COVERAGE.defaults()))
    assert round(state.scalars["kapsama"], 4) == 0.9460  # d02


def test_konu08_quiz_numbers_are_computed_not_typed(wage1, hprice1, wage_model) -> None:
    assert round(float(stats.f.sf(0.02, 2, 522)), 2) == 0.98  # k03
    equal = wage_model.f_test("exper = tenure")
    assert round(float(equal.fvalue), 2) == 24.58 and float(equal.pvalue) < 0.001  # k04
    assert (round(wage_model.params["exper"], 4), round(wage_model.params["tenure"], 4)) == (0.0223, 0.1693)
    house = smf.ols("price ~ lotsize + sqrft + bdrms", data=hprice1).fit()
    small = smf.ols("price ~ sqrft", data=hprice1).fit()
    assert (round(small.rsquared, 5), round(house.rsquared, 5)) == (0.62080, 0.67236)  # k05
    rounded = ((0.672 - 0.621) / 2) / ((1 - 0.672) / 84)
    exact = ((house.rsquared - small.rsquared) / 2) / ((1 - house.rsquared) / 84)
    assert (round(rounded, 2), round(exact, 2)) == (6.53, 6.61)
    assert round(100 * (exact - rounded) / exact) == 1  # "yaklaşık %1"
    overall = wage_model.f_test("educ = 0, exper = 0, tenure = 0")
    assert round(float(overall.fvalue), 2) == 76.87 == round(wage_model.fvalue, 2)  # k07
    assert round(4966.303 / 522, 2) == 9.51  # d03
    assert round((0.05 / 2) / (0.95 / 997), 1) == 26.2 and round(stats.f.ppf(0.95, 2, 997)) == 3  # d04
    assert round(wage_model.pvalues["exper"], 3) == 0.064 and round(wage_model.pvalues["exper"] / 2, 3) == 0.032  # d05
    assert round(float(wage_model.f_test("exper = 0").pvalue), 3) == 0.064
    assert round(((1250 - 1100) / 3) / (1100 / 200), 2) == 9.09  # b01
    tkt = float(((wage1["wage"] - wage1["wage"].mean()) ** 2).sum())
    assert round(tkt, 3) == 7160.414 and round(4966.303 / (1 - 0.30642), 1) == 7160.4  # b02
    assert round(4966.303 / (1 - 0.30642) - 4966.303, 1) == 2194.1
    assert round((2194.1 / 3) / (4966.303 / 522), 2) == 76.87
    assert (round(stats.t.ppf(0.975, 522), 4), round(stats.t.ppf(0.995, 522), 4)) == (1.9645, 2.5853)  # b03
    assert (round(1.9645 ** 2, 2), round(2.5853 ** 2, 2)) == (3.86, 6.68)
    assert round(stats.f.ppf(0.95, 1, 522), 2) == 3.86
    critical = stats.f.ppf(0.95, 2, 84)
    assert round(critical, 3) == 3.105  # b04
    assert abs(300723.805 * (1 + 2 * critical / 84) - 322955.9) < 15 and round(300723.805 * (1 + 2 * 3.105 / 84), 1) \
        == 322955.9
    assert round(100 * 2 * 3.105 / 84, 2) == 7.39 and round(100 * (348053.432 / 300723.805 - 1), 1) == 15.7
    assert (round(house.ssr, 3), round(small.ssr, 3)) == (300723.805, 348053.432)
    table = run_operations(LARGE_SAMPLE.build(LARGE_SAMPLE.defaults())).tables["tablo83"]
    assert tuple(table["t_sd"].round(3)) == (1.038, 0.985, 0.998)  # b05
    for n, k, printed in ((60, 2, 0.100), (60, 8, 0.250), (200, 2, 0.030)):  # e01
        c = stats.f.ppf(0.95, k, n - k - 1)
        assert round(c * k / (c * k + n - k - 1), 3) == pytest.approx(printed, abs=0.0015), (n, k)
    assert round((0.5990 / 0.05128) ** 2, 1) == 136.4 and round(11.6795 ** 2, 2) == 136.41  # e03
    assert round(wage_model.bse["educ"], 5) == 0.05128
    assert round(1 - 0.95 ** 2, 4) == 0.0975  # e04
    assert 0.5 + 0.5 * 0.5 == 0.75  # e05
