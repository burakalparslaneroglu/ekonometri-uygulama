"""Uygulama sekmesinin ek veri kaynakları: alternatif örnek ve "Kendi verini yükle" (``core.labs.ornekler``).

Kayıttaki her konu kendiliğinden kapsanır:

* alternatif örneğin adımları notlarla aynı numaralıdır ve aynı bölümlere bağlıdır;
* uygulama, üretilen Python ve R kodu aynı sayıları verir (varsayılan seçimde kontroller, her tek seçimde bütün
  tablolar, skalerler ve modeller);
* alternatif örneklerin sayıları bağımsız bir hesapla (pandas, numpy, statsmodels) doğrulanır;
* kendi verinde örnek dosya ve elle bozulmuş dosyalar iki dilde aynı okunur; rolü seçilmeyen adım neye ihtiyacı
  olduğunu yazar.

Dosya okuma ve temizleme kuralları İKT 217 uygulamasından taşınmıştır (``core.labs.kendi_veri``); testleri de buradadır.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import math
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from core import wooldridge_data as W
from core.charts import CHART_TYPES, figure_for
from core.codegen.base import LANGUAGES, generator, render_script, render_step, script_filename
from core.labs import kendi_veri as K
from core.labs.kurgusal_veri import PROGRAM_COLUMNS, PROGRAM_EFFECT, PROGRAM_ROWS, PROGRAM_UNEMPLOYMENT
from core.labs.ornek import SOURCE_LABELS, CustomChoices, custom_case, md, with_app_values
from core.labs.ornekler import VARIANTS
from core.labs.registry import LABS, get_lab
from core.labs.runner import LabState, evaluate_target, execute, run_lab, run_operations
from core.labs.spec import (
    BoxSummary,
    Check,
    Choice,
    CoefTarget,
    CompareBarChart,
    CompleteCases,
    CrossTab,
    FrequencyTable,
    InlineData,
    JoinColumns,
    LabSpec,
    LabStep,
    LoadWooldridge,
    MapCodes,
    ModelTarget,
    MonteCarlo,
    MultiChoice,
    NoteRef,
    NumberChoice,
    OLS,
    Outcomes,
    ReadFile,
    RegressionTable,
    ScalarTable,
    ScalarTarget,
    Shape,
    ShowFrame,
    ShowModel,
    Statistic,
    SummaryTable,
    TableTarget,
    TakeRows,
)
from topics.lab_ui import coefficient_display, display_table, frame_view, regression_display

TOPICS = sorted(VARIANTS)
XLSX = "ornek.xlsx"
SAMPLE_CHOICES = {
    "konu00": (None, CustomChoices(
        roles={"sonuc": "Yıllık kazanç (bin TL)", "aciklayici": "Eğitim yılı", "gosterge": "Cinsiyet"},
        extra=("Yaş", "Önceki yıllık kazanç (bin TL)"), picks={"gosterge": "Kadın"})),
    "konu01": (None, CustomChoices(
        roles={"sonuc": "Yıllık kazanç (bin TL)", "aciklayici": "Eğitim yılı"},
        extra=("Yaş", "Kadın (1/0)", "Evli (1/0)", "Önceki yıllık kazanç (bin TL)"))),
    "konu02_panel": ("Panel", CustomChoices(
        roles={"sayisal": "İşsizlik oranı (%)", "donem": "Yıl", "birim": "İl"},
        extra=("Büyüme (%)", "Kişi başı gelir (bin TL)"))),
    "konu02_deney": ("Deney", CustomChoices(
        roles={"sayisal": "Yıllık kazanç (bin TL)", "atama": "Grup", "deney_sonuc": "Yıllık kazanç (bin TL)"},
        extra=("Yaş", "Eğitim yılı", "Önceki yıllık kazanç (bin TL)"), picks={"atama": "Program"})),
    "konu03": (None, CustomChoices(
        roles={"sonuc": "Yıllık kazanç (bin TL)", "aciklayici": "Eğitim yılı", "gosterge": "Grup"},
        extra=("Yaş", "Önceki yıllık kazanç (bin TL)"), picks={"gosterge": "Program"})),
    **{topic: (None, CustomChoices(roles={"sonuc": "Yıllık kazanç (bin TL)", "aciklayici": "Eğitim yılı"},
                                   extra=("Yaş", "Önceki yıllık kazanç (bin TL)")))
       for topic in ("konu04", "konu05", "konu06", "konu07")},
}
AGG_SHOW = "ignore:FigureCanvasAgg is non-interactive:UserWarning"
"""Agg arka ucunda ``plt.show()`` bu uyarıyı basar; betiğin kendisinden değil test ortamından gelir."""


def _checks(spec: LabSpec) -> int:
    return sum(len(step.checks) for step in spec.steps)


def _operations(spec: LabSpec) -> tuple:
    return tuple(op for step in spec.steps for op in step.operations)


def _topic(name: str) -> str:
    return name.split("_")[0]


def _sample_data(name: str) -> bytes:
    return K.sample_excel(VARIANTS[_topic(name)].custom.sample())


def _sample_case(name: str, file_name: str = XLSX, data: bytes | None = None):
    sheet, choices = SAMPLE_CHOICES[name]
    custom = VARIANTS[_topic(name)].custom
    table = K.read_upload(file_name, data if data is not None else _sample_data(name), sheet)
    return custom_case(custom, table, choices)


def _sample_spec(name: str) -> LabSpec:
    case, _ = _sample_case(name)
    return VARIANTS[_topic(name)].custom.build(case)


def _sheet_csv(name: str) -> bytes:
    """Örnek dosyanın sayfası, Türkçe Excel'in "CSV (noktalı virgülle ayrılmış)" kaydı gibi."""

    sheet, _ = SAMPLE_CHOICES[name]
    sample = VARIANTS[_topic(name)].custom.sample()
    frame = sample[sheet] if isinstance(sample, dict) else sample
    return ("﻿" + frame.to_csv(sep=";", decimal=",", index=False)).encode("utf-8")


def _csv_spec(name: str) -> tuple[LabSpec, tuple[str, bytes]]:
    data = _sheet_csv(name)
    case, _ = _sample_case(name, "ornek.csv", data)
    return VARIANTS[_topic(name)].custom.build(case), ("ornek.csv", data)


def _read(spec: LabSpec) -> ReadFile:
    """Dosyayı okuyan ilk işlem (çoğu konuda Adım 1'in ilk işlemi; Konu 7'de Adım 2)."""

    return next(op for step in spec.steps for op in step.operations if isinstance(op, ReadFile))


def _python_environment() -> dict[str, str]:
    warnings = ",".join(item for item in (os.environ.get("PYTHONWARNINGS", ""), AGG_SHOW) if item)
    return dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="cp1254", PYTHONWARNINGS=warnings)


def _run_script(spec: LabSpec, language: str, folder: Path, command: list[str], environment: dict[str, str],
                data: tuple[str, bytes] | None = None):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / script_filename(spec, language)
    path.write_text(render_script(spec, language), encoding="utf-8")
    if data:
        (folder / data[0]).write_bytes(data[1])
    result = subprocess.run(command + [path.name], cwd=folder, capture_output=True, encoding="utf-8",
                            errors="replace", timeout=600, env=environment)
    return result, path


def _python(spec: LabSpec, folder: Path, data: tuple[str, bytes] | None = None):
    return _run_script(spec, "Python", folder, [sys.executable], _python_environment(), data)


def _reproduce(spec: LabSpec, result, path: Path, folder: Path, data: tuple[str, bytes] | None = None) -> None:
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert result.stdout.count("  OK   ") == _checks(spec)
    assert "HATA" not in result.stdout
    assert "warning" not in (result.stdout + result.stderr).lower(), (result.stdout + result.stderr)[-1500:]
    assert sorted(item.name for item in folder.iterdir()) == sorted([path.name, *([data[0]] if data else [])])


# --- Kayıt ve alternatif örnek ------------------------------------------------------------------------------

def test_registry_covers_topics_0_to_7_with_the_approved_labels() -> None:
    assert TOPICS == [f"konu{number:02d}" for number in range(8)]
    assert SOURCE_LABELS == {"notlar": "Notlardaki örnek", "alternatif": "Alternatif örnek",
                             "kendi": "Kendi verini yükle"}
    assert all(VARIANTS[topic].custom is not None and VARIANTS[topic].story for topic in TOPICS)


@pytest.mark.parametrize("topic", TOPICS)
def test_alternative_mirrors_the_steps_of_the_notes(topic: str) -> None:
    notes, alternative = get_lab(topic), VARIANTS[topic].alternative()
    assert alternative.source == "alternatif" and notes.source == "notlar"
    assert [step.number for step in alternative.steps] == [step.number for step in notes.steps]
    assert [step.note.section for step in alternative.steps] == [step.note.section for step in notes.steps]
    assert all(step.title and step.explanation for step in alternative.steps)
    assert _checks(alternative) >= 30
    assert [bool(step.operations) for step in alternative.steps] == [bool(step.operations) for step in notes.steps]
    assert alternative.resolve({}).variant == () and alternative.resolve({}).steps == alternative.steps


@pytest.mark.parametrize("topic", TOPICS)
def test_alternative_checks_point_to_computable_values(topic: str) -> None:
    """Beklenen değerler uygulamanın kendi hesabıdır; bu test yalnız her hedefin hesaplanabildiğini denetler. Sayıların
    doğruluğu bağımsız hesapla (aşağıdaki WAGE2, OKUN, KIELMC, CRIME4 testleri) ve iki dilde kodla doğrulanır."""

    assert run_lab(VARIANTS[topic].alternative()).all_passed


@pytest.mark.parametrize("topic", TOPICS)
def test_every_alternative_step_renders(topic: str) -> None:
    spec = VARIANTS[topic].alternative()
    for language in LANGUAGES:
        for step in spec.steps:
            assert bool(render_step(spec, step.number, language)) == bool(step.operations)
        assert script_filename(spec, language) == f"ikt305_{topic}_alternatif.{'py' if language == 'Python' else 'R'}"
    script = render_script(spec, "Python")
    assert "alternatif örnek" in script and "ders notlarındaki basılı" not in script
    assert "(uygulama: " in script and "(notlar: " not in script and "Uygulamayla karşılaştırma:" in script
    assert "wooldridge paketinden okunur" in script
    compile(script, f"{topic}.py", "exec")
    resolved = spec.resolve(_single_changes(spec)[0])
    assert script_filename(resolved, "R") == f"ikt305_{topic}_alternatif_secim.R"
    assert "varsayılandan farklı, seçilen spesifikasyon" in render_script(resolved, "R")


@pytest.mark.parametrize("topic", TOPICS)
def test_generated_python_reproduces_the_alternative(topic: str, tmp_path: Path) -> None:
    spec = VARIANTS[topic].alternative()
    result, path = _python(spec, tmp_path)
    _reproduce(spec, result, path, tmp_path)
    assert "Bütün değerler uygulamadaki sonuçlarla uyuşuyor." in result.stdout


@pytest.mark.parametrize("topic", TOPICS)
def test_generated_r_reproduces_the_alternative(topic: str, tmp_path: Path, rscript: str,
                                                r_environment: dict[str, str]) -> None:
    spec = VARIANTS[topic].alternative()
    result, path = _run_script(spec, "R", tmp_path, [rscript], r_environment)
    _reproduce(spec, result, path, tmp_path)


def test_notes_scripts_keep_their_wording_and_names() -> None:
    for topic in TOPICS:
        spec = get_lab(topic)
        script = render_script(spec, "Python")
        assert "ders notlarındaki basılı değerlerle karşılaştırılır" in script
        assert "(notlar: " in script and "(uygulama: " not in script
        assert script_filename(spec, "R") == f"ikt305_{topic}_uygulama.R"


# --- Seçimler: alternatif örnek ve kendi verin ---------------------------------------------------------------

def _number_values(control: NumberChoice) -> list[float]:
    middle = control.normalize(control.minimum + (control.maximum - control.minimum) / 2)
    values = {control.normalize(control.minimum), control.normalize(control.maximum), middle}
    return sorted(value for value in values if value != control.default)


def _multi_values(control: MultiChoice) -> list[tuple[str, ...]]:
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


def _single_changes(spec: LabSpec, limit: int = 8) -> list[dict[str, object]]:
    """Her denetimin varsayılandan farklı değerleri; çok seçenekli denetimde (ör. 47 yıl) uçlar ve aradan örnekler."""

    changes = []
    for control in spec.controls:
        if isinstance(control, NumberChoice):
            values = _number_values(control)
        elif isinstance(control, MultiChoice):
            values = _multi_values(control)
        else:
            values = [value for value, _ in control.options if value != control.default]
        if len(values) > limit:
            picks = np.unique(np.linspace(0, len(values) - 1, limit).round().astype(int))
            values = [values[index] for index in picks]
        changes += [{control.key: value} for value in values]
    return changes


def _specs() -> list[tuple[str, LabSpec]]:
    found = [(f"{topic}_alternatif", VARIANTS[topic].alternative()) for topic in TOPICS]
    found += [(f"{name}_kendi", _sample_spec(name)) for name in SAMPLE_CHOICES]
    return found


SPECS = _specs()


@pytest.mark.parametrize("spec", [spec for _, spec in SPECS], ids=[name for name, _ in SPECS])
def test_a_changed_choice_marks_its_own_step_and_the_steps_that_use_it(spec: LabSpec) -> None:
    changes = _single_changes(spec, limit=100)
    assert changes
    for change in changes:
        (key,) = change
        marked = {step.number for step in spec.steps
                  if any(control.key == key for control in (*step.controls, *step.uses))}
        assert spec.resolve(change).variant == tuple(sorted(marked)), change
        assert run_lab(spec.resolve(change)).all_passed, change


def _frames_equal(app: pd.DataFrame, script: pd.DataFrame, name: str) -> None:
    assert list(script.columns) == list(app.columns), name
    numeric = [column for column in app.columns if pd.api.types.is_numeric_dtype(app[column])]
    np.testing.assert_allclose(script[numeric].to_numpy(float), app[numeric].to_numpy(float), rtol=0, atol=1e-12,
                               err_msg=name)
    texts = [column for column in app.columns if column not in numeric]
    for column in texts:
        assert script[column].where(script[column].notna(), None).tolist() == \
               app[column].where(app[column].notna(), None).tolist(), (name, column)


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


def _same_index(script: pd.Index, app: pd.Index, name: str) -> None:
    """Satır etiketleri aynı: sayısal etiketlerde değer (0 ile 0.0 aynı), diğerlerinde metin."""

    assert len(script) == len(app), name
    if pd.api.types.is_numeric_dtype(script) and pd.api.types.is_numeric_dtype(app):
        np.testing.assert_allclose(script.to_numpy(float), app.to_numpy(float), rtol=0, atol=0, err_msg=name)
    else:
        assert [str(label) for label in script] == [str(label) for label in app], name


def _compare(spec: LabSpec, change: dict) -> None:
    resolved = spec.resolve(change)
    state = run_operations(_operations(resolved))
    namespace = _run_python(resolved)
    for name, table in state.tables.items():
        numeric = table.select_dtypes("number")
        _same_index(namespace[name].index, table.index, f"{change} {name}")
        np.testing.assert_allclose(namespace[name][numeric.columns].to_numpy(float), numeric.to_numpy(float),
                                   rtol=0, atol=1e-12, err_msg=f"{change} {name}")
    for name, value in state.scalars.items():
        assert float(namespace[name]) == pytest.approx(value, rel=1e-12, abs=1e-12), (change, name)
    for name, frame in state.frames.items():
        _frames_equal(frame, namespace[name], f"{change} {name}")
    for name, model in state.models.items():
        np.testing.assert_allclose(namespace[name].params.to_numpy(), model.params.to_numpy(), rtol=0, atol=1e-10)
        np.testing.assert_allclose(namespace[name].bse.to_numpy(), model.bse.to_numpy(), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("topic", TOPICS)
def test_every_single_choice_gives_the_same_numbers_in_the_app_and_in_python(topic: str) -> None:
    spec = VARIANTS[topic].alternative()
    for change in [{}] + _single_changes(spec):
        _compare(spec, change)


@pytest.mark.parametrize("name", list(SAMPLE_CHOICES))
def test_own_data_choices_give_the_same_numbers_in_the_app_and_in_python(name: str, tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Kendi verinde üretilen Python kodu dosyayı okur (çalışma klasöründe) ve uygulamanın bütün sayılarını verir."""

    spec = _sample_spec(name)
    (tmp_path / XLSX).write_bytes(_sample_data(name))
    monkeypatch.chdir(tmp_path)
    for change in [{}] + _single_changes(spec, limit=4):
        _compare(spec, change)


def _clean(text: str) -> bool:
    """Metin boş değil, "nan" içermiyor, eksi sıfır ("−0,00") ve ondalıklı sayıya Türkçe ek ("0,35'ye") yazmıyor."""

    return (bool(text) and not re.search(r"\bnan\b", text.lower()) and not re.search(r"−0,0+(?![0-9])", text)
            and not re.search(r"\d,\d+'", text))


def _screen(operations, state: LabState, label) -> None:
    """Ekrandaki her tablo: sütun adları tekil, hücrelerde "nan" yok; veri çerçeveleri de (kendi verinizin sütunları
    uygulamanın türettiği sütunlarla aynı tabloda) ve grafikler çizilir."""

    for op in operations:
        shown = None
        if isinstance(op, ShowModel):
            shown = coefficient_display(op, state.models[op.model], label)
        elif isinstance(op, RegressionTable):
            shown = regression_display(op, state, label)
        elif isinstance(op, (ShowFrame, MapCodes, ReadFile, TakeRows)):
            view = frame_view(op, state, label)
            frame = view if isinstance(view, pd.DataFrame) else view.data
            assert frame.columns.is_unique, (type(op).__name__, list(frame.columns))
            text = frame.to_string() if isinstance(view, pd.DataFrame) else view.to_string()
            assert not re.search(r"\bnan\b", text, flags=re.IGNORECASE), (type(op).__name__, text[:500])
        elif hasattr(op, "result") and op.result in state.tables and not isinstance(op, MonteCarlo):
            shown = display_table(op, state.tables[op.result], label)
        if shown is not None:
            assert shown.columns.is_unique, (type(op).__name__, list(shown.columns))
            assert not shown.astype(str).apply(lambda column: column.str.contains(r"\bnan\b", case=False)).any().any(), op
        if isinstance(op, CHART_TYPES):
            figure_for(op, state, label)


@pytest.mark.parametrize("spec", [spec for _, spec in SPECS], ids=[name for name, _ in SPECS])
def test_changed_steps_explain_the_choice_and_show_clean_tables(spec: LabSpec) -> None:
    for change in [{}] + _single_changes(spec):
        resolved = spec.resolve(change)
        choices = spec.normalize(change)
        state = run_operations(_operations(resolved))
        _screen(_operations(resolved), state, spec.label)
        for step in spec.steps:
            assert _clean(step.explanation), step.number
            if step.note_for is not None and step.operations:
                partial = run_operations(resolved.operations_through(step.number))
                assert _clean(step.note_for(partial, choices)), (change, step.number)
            elif step.takeaway:
                assert _clean(step.takeaway), step.number


def _labels(op) -> list[str] | None:
    """Üretilen Python kodunda sözlük anahtarı olan etiketler (aynı iki etiket tek satıra iner)."""

    if isinstance(op, (ScalarTable, SummaryTable)):
        return [label for label, _ in op.rows]
    if isinstance(op, RegressionTable):
        return [heading for heading, _ in op.models]
    if isinstance(op, JoinColumns):
        return [name for name, _, _ in op.columns]
    if isinstance(op, BoxSummary):
        return [label for _, _, label in op.series]
    if isinstance(op, Outcomes):
        return [name for name, _ in op.stages]
    if isinstance(op, CompareBarChart):
        return [label for label, _ in op.tables]
    if isinstance(op, MapCodes):
        return [label for label, _ in op.mapping]
    return None


def _coinciding_choices(spec: LabSpec) -> list[dict[str, object]]:
    """Aynı adımdaki iki denetim aynı değeri alır (ör. başlangıç ve yeni değer aynı grup)."""

    changes = []
    for step in spec.steps:
        for first, second in itertools.permutations(step.controls, 2):
            if isinstance(first, NumberChoice) and isinstance(second, NumberChoice):
                if first.minimum <= second.default <= first.maximum:
                    changes.append({first.key: second.default})
            elif type(first) is Choice and type(second) is Choice:
                if second.default in {value for value, _ in first.options}:
                    changes.append({first.key: second.default})
    return changes


LABELLED = [(f"{topic}_notlar", spec) for topic, spec in LABS.items()] + SPECS


@pytest.mark.parametrize("spec", [spec for _, spec in LABELLED], ids=[name for name, _ in LABELLED])
def test_labelled_tables_keep_every_row_when_two_choices_coincide(spec: LabSpec) -> None:
    for change in [{}] + _single_changes(spec, limit=100) + _coinciding_choices(spec):
        for op in _operations(spec.resolve(change)):
            labels = _labels(op)
            if labels is not None:
                assert len(labels) == len(set(labels)), (change, type(op).__name__, labels)


def test_equal_start_and_end_values_keep_both_directions() -> None:
    """Başlangıç ve yeni değer aynıyken yüzde değişim tablosu iki yönü de gösterir (notlarda ve alternatifte)."""

    for spec, change in ((get_lab("konu00"), {"adim6_x0": 120}), (VARIANTS["konu00"].alternative(), {"adim6_d0": "16"})):
        resolved = spec.resolve(change)
        state = run_operations(_operations(resolved))
        table = state.tables["yuzdeler"]
        assert len(table) == 6 and table.index.is_unique
        assert np.allclose(table["deger"].to_numpy()[:4], 0.0)
        assert [str(label) for label in _run_python(resolved)["yuzdeler"].index] == list(table.index)


def _chosen_with_checks(spec: LabSpec, change: dict) -> LabSpec:
    """Seçilen spesifikasyon, değişen adımlarda da kontrollerle: varsayılan adımın hedefleri seçimin hesabıyla
    doldurulur; artık hesaplanamayan hedefler (ör. modelden çıkan katsayı) atlanır. Böylece üretilen kod seçimde de
    uygulamanın bütün sayılarıyla karşılaştırılır."""

    resolved = spec.resolve(change)
    defaults = {step.number: step.checks for step in spec.steps}
    state = LabState()
    steps = []
    for step in resolved.steps:
        for op in step.operations:
            execute(op, state)
        checks = list(step.checks)
        if step.number in resolved.variant:
            checks = []
            for check in defaults[step.number]:
                try:
                    value = evaluate_target(check.target, state)
                except Exception:  # noqa: BLE001 - hedef seçimde yok
                    continue
                if math.isfinite(value):
                    checks.append(replace(check, expected=0.0 if abs(value) < 0.5 * 10 ** (-check.decimals) else value))
        steps.append(replace(step, checks=tuple(checks)))
    return replace(resolved, steps=tuple(steps))


def _last_changes(spec: LabSpec) -> list[dict]:
    last = {}
    for change in _single_changes(spec):
        last[next(iter(change))] = change
    return list(last.values())


R_CHOSEN = [(f"{topic}_alternatif", topic) for topic in TOPICS] + [(f"{name}_kendi", name) for name in SAMPLE_CHOICES]


@pytest.mark.parametrize("name, source", R_CHOSEN, ids=[name for name, _ in R_CHOSEN])
def test_generated_r_reproduces_the_last_option_of_every_control(name: str, source: str, tmp_path: Path,
                                                                 rscript: str, r_environment: dict[str, str]) -> None:
    """Her denetimin son seçeneğinde R kodu uygulamanın bütün sayılarını verir (değişen adımlarda da). Kendi verinde
    örnek dosyanın Türkçe CSV kaydı okunur."""

    if name.endswith("_alternatif"):
        spec, data = VARIANTS[source].alternative(), None
    else:
        spec, data = _csv_spec(source)
    for change in _last_changes(spec):
        chosen = _chosen_with_checks(spec, change)
        assert _checks(chosen) > 0
        folder = tmp_path / next(iter(change))
        result, path = _run_script(chosen, "R", folder, [rscript], r_environment, data)
        _reproduce(chosen, result, path, folder, data)


# --- Alternatif örneklerin sayıları: bağımsız hesap ------------------------------------------------------------

@pytest.fixture(scope="module")
def wage2() -> pd.DataFrame:
    return W.load("wage2")


def _scalars(spec: LabSpec, change: dict | None = None) -> dict[str, float]:
    resolved = spec.resolve(change or {})
    return run_operations(_operations(resolved)).scalars


def test_konu00_alternative_numbers_follow_wage2_and_okun(wage2: pd.DataFrame) -> None:
    spec = VARIANTS["konu00"].alternative()
    state = run_operations(_operations(spec))
    s = state.scalars
    excerpt = wage2.iloc[[0, 187, 374, 561, 748]]
    assert state.frames["ornek"]["gozlem"].tolist() == [1, 188, 375, 562, 749]
    assert excerpt["wage"].tolist() == [769, 962, 975, 685, 1000] and excerpt["educ"].tolist() == [12, 16, 13, 12, 13]
    assert (s["n"], s["k"], s["gosterge_payi"]) == (5, 4, pytest.approx(0.8))
    assert (s["toplam"], s["ortalama"]) == (66, pytest.approx(13.2))
    assert s["varyans"] == pytest.approx(excerpt["educ"].var()) == pytest.approx(2.7)
    assert (s["var_A"], s["var_B"]) == (pytest.approx(10.0), pytest.approx(90.0))
    okun = W.load("okun").dropna(subset=["cunem"])
    assert s["r_pozitif"] == pytest.approx(np.corrcoef(wage2["educ"], wage2["IQ"])[0, 1], abs=1e-12)
    assert s["r_negatif"] == pytest.approx(np.corrcoef(okun["pcrgdp"], okun["cunem"])[0, 1], abs=1e-12)
    assert round(s["r_pozitif"], 2) == 0.52 and round(s["r_negatif"], 2) == -0.84 and abs(s["r_zayif"]) < 0.01
    assert abs(s["r_karesel"]) < 1e-12
    assert s["kovaryans"] == pytest.approx(np.cov(wage2["wage"], wage2["educ"])[0, 1])
    assert s["kovaryans_100"] == pytest.approx(100 * s["kovaryans"])
    assert s["korelasyon_100"] == pytest.approx(s["korelasyon"]) and round(s["korelasyon"], 3) == 0.327
    means = wage2.groupby("educ")["wage"].mean()
    assert (s["x0"], s["x1"]) == (pytest.approx(means[12]), pytest.approx(means[16]))
    assert round(s["x0"], 2) == 862.67 and round(s["x1"], 2) == 1108.71 and round(s["yuzde_degisim"], 1) == 28.5
    assert s["log_farki"] == pytest.approx(100 * np.log(means[16] / means[12]))
    unemployment = W.load("okun").set_index("year")["unem"]
    assert (round(s["oran0"], 1), round(s["oran1"], 1), round(s["yuzde_puan"], 1)) == (5.8, 9.7, 3.9)
    assert s["goreli_degisim"] == pytest.approx(100 * (unemployment[1982] / unemployment[1979] - 1))
    table = state.tables["tablo03"]
    assert table.loc["12 → 18 yıl", "tam"] == pytest.approx(100 * (means[18] / means[12] - 1))
    np.testing.assert_allclose(state.tables["kosullu"]["mean"].to_numpy(), means.to_numpy(), rtol=0, atol=1e-9)
    assert s["kosullu_ortalama"] == pytest.approx(means[16]) and s["kosul_n"] == 150
    order = np.arange(1, len(wage2) + 1)
    assert s["tahmin_tek"] == pytest.approx(wage2["wage"][order % 2 == 1].mean())
    assert s["tahmin_besinci"] == pytest.approx(wage2["wage"][(order - 1) % 5 == 0].mean())
    assert (round(s["tahmin_tum"], 2), round(s["tahmin_tek"], 2), round(s["tahmin_cift"], 2),
            round(s["tahmin_besinci"], 2)) == (957.95, 972.68, 943.18, 1000.61)
    slope, intercept = np.polyfit(wage2["educ"], wage2["wage"], 1)
    model = state.models["model"]
    assert model.params["Intercept"] == pytest.approx(intercept) and s["b1"] == pytest.approx(slope)
    assert (round(model.params["Intercept"], 2), round(s["b1"], 2), round(s["r2"], 3)) == (146.95, 60.21, 0.107)
    assert s["r2"] == pytest.approx(s["r_yx"] ** 2)


def test_konu01_alternative_numbers_follow_wage2(wage2: pd.DataFrame) -> None:
    spec = VARIANTS["konu01"].alternative()
    state = run_operations(_operations(spec))
    summary = state.tables["ozet"]
    expected = wage2[["wage", "educ", "exper", "tenure"]].describe().T[["count", "mean", "std", "min", "max"]]
    np.testing.assert_allclose(summary.to_numpy(float), expected.to_numpy(float), rtol=0, atol=1e-9)
    assert (state.scalars["n"], state.scalars["k"]) == (935, 17)
    model = smf.ols("wage ~ educ", data=wage2).fit()
    assert state.scalars["b1"] == pytest.approx(model.params["educ"]) and state.scalars["r2"] == pytest.approx(model.rsquared)
    iq = _scalars(spec, {"adim4_x": "IQ"})
    assert iq["b1"] == pytest.approx(np.polyfit(wage2["IQ"], wage2["wage"], 1)[0])
    assert "r = 0,52" in spec.step(6).explanation and "60,21" in spec.step(6).explanation


def test_konu02_alternative_numbers_follow_the_four_data_sets(wage2: pd.DataFrame) -> None:
    spec = VARIANTS["konu02"].alternative()
    state = run_operations(_operations(spec))
    assert state.frames["wage2"].iloc[[0, 1, 2, 934]]["wage"].tolist() == [769, 808, 825, 1000]
    okun = W.load("okun")
    assert state.frames["okun"]["year"].head(6).tolist() == okun["year"].head(6).tolist()
    assert (state.scalars["ilk_yil"], state.scalars["son_yil"]) == (1959, 2005)
    kielmc = W.load("kielmc")
    counts = state.tables["donem_ozeti"]["count"]
    assert (counts[1978], counts[1981]) == (179, 142)
    means = kielmc.groupby("year")["price"].mean()
    assert state.tables["donem_ozeti"]["mean"].to_numpy() == pytest.approx(means.to_numpy())
    real = run_operations(spec.resolve({"adim3_degisken": "rprice"}).operations_through(3)).tables["donem_ozeti"]
    assert real.loc[1978, "mean"] == pytest.approx(means[1978]) and real.loc[1981, "mean"] < means[1981]
    panel = state.tables["panel"]["deger"]
    assert (panel["birim"], panel["gozlem"], panel["ilk_donem"], panel["son_donem"]) == (90, 630, 81, 87)
    assert panel["en_az_donem"] == panel["en_cok_donem"] == 7
    program = pd.DataFrame(list(PROGRAM_ROWS), columns=list(PROGRAM_COLUMNS))
    groups = program.groupby("program")["kazanc"].mean()
    assert (state.scalars["ort_kontrol"], state.scalars["ort_program"]) == (pytest.approx(groups[0]),
                                                                         pytest.approx(groups[1]))
    assert state.scalars["fark"] == pytest.approx(groups[1] - groups[0])
    assert state.tables["grup_ozeti"]["count"].tolist() == [100, 100]


def test_fictional_program_data_is_a_clean_randomized_experiment() -> None:
    frame = pd.DataFrame(list(PROGRAM_ROWS), columns=list(PROGRAM_COLUMNS))
    assert frame.shape == (200, 9) and frame["kisi"].tolist() == list(range(1, 201))
    assert frame["program"].value_counts().to_dict() == {0: 100, 1: 100}
    assert (frame.loc[frame["issiz"] == 1, "kazanc"] == 0).all() and (frame.loc[frame["issiz"] == 0, "kazanc"] >= 30).all()
    assert set(frame["egitim"]) <= {5, 8, 12, 14, 16} and frame["yas"].between(20, 55).all()
    assert frame[["kadin", "evli", "issiz"]].isin([0, 1]).all().all() and (frame["onceki_kazanc"] >= 0).all()
    gaps = frame.groupby("program")[["yas", "egitim", "onceki_kazanc"]].mean().diff().iloc[1].abs()
    assert gaps["yas"] < 2 and gaps["egitim"] < 1 and gaps["onceki_kazanc"] < 20


def test_the_stated_program_effect_is_the_dgp_average_effect() -> None:
    """Metindeki gerçek ortalama etki (parametre) DGP'nin anakütle ortalamasıdır; bu kuradaki tahmin ondan ayrıdır.

    İşsizlik kazançtan bağımsızdır: etki = 0,76·E[K(1)] − 0,70·E[K(0)]. K(0) ve K(1) aynı kişiler ve aynı e₂ ile
    çekilir (ortak rastgele sayılar); 2 milyon çekilişte standart hata yaklaşık 0,004'tür.
    """

    rng = np.random.default_rng(2026)
    n = 2_000_000
    kadin = rng.binomial(1, 0.45, n)
    yas = np.clip(np.rint(rng.normal(34, 8, n)), 20, 55)
    egitim = rng.choice([5, 8, 12, 14, 16], size=n, p=[0.10, 0.25, 0.35, 0.10, 0.20])
    onceki = np.round(np.maximum(0, 150 + 15 * (egitim - 8) + 3 * (yas - 34) - 25 * kadin + rng.normal(0, 60, n)), 1)
    base = 260 + 12 * (egitim - 8) + 2 * (yas - 34) - 30 * kadin + 0.25 * onceki + rng.normal(0, 70, n)
    control, treated = PROGRAM_UNEMPLOYMENT
    effect = (1 - treated) * np.round(np.maximum(30, base + 10), 1).mean() - \
        (1 - control) * np.round(np.maximum(30, base), 1).mean()
    assert effect == pytest.approx(PROGRAM_EFFECT, abs=0.05)
    spec = VARIANTS["konu02"].alternative()
    state = run_operations(spec.operations_through(5))
    estimate = state.scalars["fark"]
    assert estimate == pytest.approx(53.277, abs=1e-3) and abs(estimate - PROGRAM_EFFECT) > 20
    text = spec.step(5).note_for(state, spec.normalize({}))
    assert "53,277 bin TL bu kuradaki tahmindir" in text and "yaklaşık 27,6 bin TL" in text
    assert "0,67 yıl daha uzundur" in text and "kazancı yaklaşık bu kadar değiştirmiştir" not in text
    changed = spec.resolve({"adim5_degisken": "issiz"})
    shares = run_operations(changed.operations_through(5))
    assert "6 yüzde puan düşürür" in changed.step(5).note_for(shares, spec.normalize({"adim5_degisken": "issiz"}))


def test_fictional_program_rows_follow_the_documented_dgp() -> None:
    """Dondurulmuş satırlar modül belgesindeki DGP'nin tohum 305 ile çıktısıdır (bu NumPy sürümünde yeniden üretilir)."""

    rng = np.random.default_rng(305)
    n = 200
    kadin = rng.binomial(1, 0.45, n)
    evli = rng.binomial(1, 0.50, n)
    yas = np.clip(np.rint(rng.normal(34, 8, n)), 20, 55).astype(int)
    egitim = rng.choice([5, 8, 12, 14, 16], size=n, p=[0.10, 0.25, 0.35, 0.10, 0.20])
    onceki = np.round(np.maximum(0, 150 + 15 * (egitim - 8) + 3 * (yas - 34) - 25 * kadin + rng.normal(0, 60, n)), 1)
    kura = rng.permutation(n)
    program = np.zeros(n, dtype=int)
    program[kura[:100]] = 1
    issiz = rng.binomial(1, 0.30 - 0.06 * program)
    kazanc = np.maximum(30, 260 + 12 * (egitim - 8) + 2 * (yas - 34) - 30 * kadin + 0.25 * onceki + 10 * program
                        + rng.normal(0, 70, n))
    kazanc = np.where(issiz == 1, 0.0, np.round(kazanc, 1))
    frozen = pd.DataFrame(list(PROGRAM_ROWS), columns=list(PROGRAM_COLUMNS))
    assert frozen["program"].tolist() == program.tolist() and frozen["issiz"].tolist() == issiz.tolist()
    np.testing.assert_allclose(frozen["kazanc"], kazanc, rtol=0, atol=1e-9)
    np.testing.assert_allclose(frozen["onceki_kazanc"], onceki, rtol=0, atol=1e-9)
    assert frozen["yas"].tolist() == yas.tolist() and frozen["egitim"].tolist() == egitim.tolist()
    assert frozen["kadin"].tolist() == kadin.tolist() and frozen["evli"].tolist() == evli.tolist()


# --- Konu 3–7 alternatif örnekleri: bağımsız hesap ----------------------------------------------------------

@pytest.fixture(scope="module")
def houses() -> pd.DataFrame:
    """KIELMC'nin 1978 satışları: notlarda HPRICE1 kullanılan konut adımlarının alternatif verisi."""

    kielmc = W.load("kielmc")
    return kielmc[kielmc["year"] == 1978].reset_index(drop=True)


def _program() -> pd.DataFrame:
    return pd.DataFrame(list(PROGRAM_ROWS), columns=list(PROGRAM_COLUMNS))


def test_konu03_alternative_numbers_follow_wage2_and_the_program(wage2: pd.DataFrame) -> None:
    spec = VARIANTS["konu03"].alternative()
    state = run_operations(_operations(spec))
    s = state.scalars
    means = wage2.groupby("educ")["wage"].agg(["count", "mean"])
    np.testing.assert_allclose(state.tables["ortalamalar"][["count", "mean"]].to_numpy(float), means.to_numpy(float),
                               rtol=0, atol=1e-9)
    assert (s["n_x"], s["n_x1"]) == (393, 85)
    assert s["fark"] == pytest.approx(means.loc[13, "mean"] - means.loc[12, "mean"])
    # Adım 3: eğitime göre kararlı sıralamada 0, n/4, n/2, 3n/4 ve son konumdaki beş çalışan
    ordered = wage2.reset_index(drop=True).sort_values("educ", kind="stable")
    rows = ordered.iloc[[(index * (len(wage2) - 1)) // 4 for index in range(5)]]
    small = state.frames["kucuk_ornek"]
    assert small["gozlem"].tolist() == (rows.index + 1).tolist() == [112, 379, 907, 73, 901]
    assert small["wage"].tolist() == rows["wage"].tolist() and spec.step(3).controls[0].default == rows["wage"].iloc[4]
    slope, intercept = np.polyfit(rows["educ"], rows["wage"], 1)
    assert (s["b1_kucuk"], s["b0_kucuk"]) == (pytest.approx(slope), pytest.approx(intercept))
    assert abs(s["artik_toplami"]) < 1e-9
    model = smf.ols("wage ~ educ", data=wage2).fit()
    assert (s["b0"], s["b1"]) == (pytest.approx(model.params["Intercept"]), pytest.approx(model.params["educ"]))
    assert s["tahmin_x0"] == pytest.approx(model.params["Intercept"] + 12 * model.params["educ"])
    assert s["artik_y0"] == pytest.approx(900 - s["tahmin_x0"])
    program = _program()
    groups = program.groupby("program")["kazanc"].mean()
    fit = smf.ols("kazanc ~ program", data=program).fit()
    assert (s["ort_0"], s["ort_1"]) == (pytest.approx(groups[0]), pytest.approx(groups[1]))
    assert (s["sabit_j"], s["egim_j"]) == (pytest.approx(fit.params["Intercept"]), pytest.approx(groups[1] - groups[0]))
    text = spec.step(6).note_for(run_operations(spec.operations_through(6)), spec.normalize({}))
    assert "kurayla atandığı için" in text and "27,6 bin TL" in text
    change = {"adim6_d": "kadin"}
    women = run_operations(spec.resolve(change).operations_through(6))
    assert women.scalars["egim_j"] == pytest.approx(program.groupby("kadin")["kazanc"].mean().diff().iloc[1])
    text = spec.step(6).note_for(women, spec.normalize(change))
    assert "rastgele atanmadı" in text and "nedensel etki olarak okunamaz" in text and "27,6" not in text


def test_konu04_alternative_numbers_follow_wage2_and_kielmc(wage2: pd.DataFrame, houses: pd.DataFrame) -> None:
    spec = VARIANTS["konu04"].alternative()
    state = run_operations(_operations(spec))
    s = state.scalars
    model = smf.ols("wage ~ educ", data=wage2).fit()
    assert (s["b0"], s["b1"], s["n"]) == (pytest.approx(model.params["Intercept"]), pytest.approx(model.params["educ"]),
                                         935)
    assert s["ort_y"] == pytest.approx(wage2["wage"].mean())
    assert s["ort_tahmin"] == pytest.approx(s["ort_y"]) and s["dogru_xbar"] == pytest.approx(s["ort_y"])
    assert abs(s["artik_toplami"]) < 1e-6
    # Adım 3'ün sayıları verinin kendisinden: ortalama, ilk çalışanın tahmini ve kazancı (tam sayıya yuvarlanmış)
    defaults = {control.key: control.default for control in spec.step(3).controls}
    assert defaults == {"adim3_ybar": round(wage2["wage"].mean()), "adim3_yhat": round(model.fittedvalues.iloc[0]),
                        "adim3_y": wage2["wage"].iloc[0]} == {"adim3_ybar": 958, "adim3_yhat": 870, "adim3_y": 769}
    assert (s["toplam_sapma"], s["model_sapma"], s["artik_sapma"]) == (-189, -88, -101)
    assert (s["tkt"], s["mkt"], s["hkt"]) == (pytest.approx(model.centered_tss), pytest.approx(model.ess),
                                              pytest.approx(model.ssr))
    assert s["r2_yazilim"] == pytest.approx(model.rsquared)
    assert s["r"] == pytest.approx(np.corrcoef(wage2["wage"], wage2["educ"])[0, 1])
    level = smf.ols("price ~ area", data=houses).fit()
    assert len(houses) == 179 and s["r2_birim"] == pytest.approx(level.rsquared)
    assert (s["sabit_birim"], s["egim_birim"]) == (pytest.approx(level.params["Intercept"]), pytest.approx(level.params["area"]))
    assert s["yuz_fit_bin"] == pytest.approx(100 * level.params["area"] / 1000)
    scaled = _scalars(spec, {"adim6_y": "bin"})
    assert scaled["egim_birim"] == pytest.approx(level.params["area"] / 1000)
    assert scaled["sabit_birim"] == pytest.approx(level.params["Intercept"] / 1000)
    assert _scalars(spec, {"adim6_x": "yuz"})["egim_birim"] == pytest.approx(100 * level.params["area"])
    forms = {name: smf.ols(formula, data=houses).fit() for name, formula in (
        ("dd", "price ~ area"), ("ld", "lprice ~ area"), ("dl", "price ~ larea"), ("ll", "lprice ~ larea"))}
    assert s["egim_ld"] == pytest.approx(forms["ld"].params["area"])
    assert s["egim_dl"] == pytest.approx(forms["dl"].params["larea"])
    assert s["egim_ll"] == pytest.approx(forms["ll"].params["larea"])
    assert s["dd_fark"] == pytest.approx(100 * forms["dd"].params["area"])  # 100 fit²lik fark, dolar
    assert s["ld_fark"] == pytest.approx(100 * 100 * forms["ld"].params["area"])  # 100 fit², yaklaşık yüzde
    assert (s["dl_1"], s["ll_10"]) == (pytest.approx(forms["dl"].params["larea"] / 100),
                                       pytest.approx(10 * forms["ll"].params["larea"]))
    table = state.tables["dort_bicim"]
    assert table.loc["r2"].tolist() == pytest.approx([forms[name].rsquared for name in ("dd", "ld", "dl", "ll")])
    log_model = smf.ols("lwage ~ educ", data=wage2).fit()
    assert (s["b1_log"], s["r2_log"]) == (pytest.approx(log_model.params["educ"]), pytest.approx(log_model.rsquared))
    assert s["yuzde_log"] == pytest.approx(100 * log_model.params["educ"])
    assert s["makale_esneklik"] == pytest.approx(forms["ll"].params["larea"])


def test_konu05_alternative_numbers_follow_wage2_and_kielmc(wage2: pd.DataFrame, houses: pd.DataFrame) -> None:
    spec = VARIANTS["konu05"].alternative()
    state = run_operations(_operations(spec))
    s = state.scalars
    simple = smf.ols("wage ~ educ", data=wage2).fit()
    multi = smf.ols("wage ~ educ + exper + tenure", data=wage2).fit()
    assert s["basit_x"] == pytest.approx(simple.params["educ"]) and s["b_sabit"] == pytest.approx(multi.params["Intercept"])
    assert [s[f"b_{name}"] for name in ("educ", "exper", "tenure")] == pytest.approx(
        multi.params[["educ", "exper", "tenure"]].tolist())
    profiles = pd.DataFrame({"educ": [12, 16], "exper": [10, 10], "tenure": [5, 5]})
    predicted = multi.predict(profiles).tolist()
    assert [s["tahmin_A"], s["tahmin_B"]] == pytest.approx(predicted)
    assert s["fark_BA"] == pytest.approx(4 * multi.params["educ"]) and s["x_farki"] == pytest.approx(s["fark_BA"])
    assert s["birey_tahmin"] == pytest.approx(predicted[1]) and s["birey_artik"] == pytest.approx(1200 - predicted[1])
    x_residual = smf.ols("educ ~ exper + tenure", data=wage2).fit().resid
    y_residual = smf.ols("wage ~ exper + tenure", data=wage2).fit().resid
    assert s["kismi_egim"] == pytest.approx(np.polyfit(x_residual, y_residual, 1)[0])
    assert s["kismi_egim"] == pytest.approx(multi.params["educ"]) and abs(s["kismi_fark"]) < 1e-9
    log_multi = smf.ols("lwage ~ educ + exper + tenure", data=wage2).fit()
    assert (s["r2_kod"], s["r2d_kod"]) == (pytest.approx(multi.rsquared), pytest.approx(multi.rsquared_adj))
    assert s["log_x"] == pytest.approx(log_multi.params["educ"])
    assert s["basit_b_exper"] == pytest.approx(smf.ols("wage ~ exper", data=wage2).fit().params["exper"])
    assert s["basit_b_tenure"] == pytest.approx(smf.ols("wage ~ tenure", data=wage2).fit().params["tenure"])
    house = smf.ols("price ~ area + rooms + baths", data=houses).fit()
    assert [s[f"f_{name}"] for name in ("area", "rooms", "baths")] == pytest.approx(
        house.params[["area", "rooms", "baths"]].tolist())
    rows = pd.DataFrame({"area": [2000, 2000, 2250], "rooms": [7, 8, 8], "baths": [2, 2, 3]})
    assert [s["konut_A"], s["konut_B"], s["konut_C"]] == pytest.approx(house.predict(rows).tolist())
    assert s["konut_fark"] == pytest.approx(house.params["rooms"]) and s["yuz_fit"] == pytest.approx(100 * house.params["area"])
    wide = smf.ols("wage ~ educ + exper + tenure + IQ", data=wage2).fit()
    n = len(wage2)
    assert s["r2d_elle"] == pytest.approx(1 - (1 - multi.rsquared) * (n - 1) / (n - 3 - 1))
    assert s["r2d_elle"] == pytest.approx(multi.rsquared_adj) and s["r2_genis"] == pytest.approx(wide.rsquared)
    assert state.tables["abc"]["deger"].tolist() == pytest.approx(
        [simple.rsquared_adj, multi.rsquared_adj, wide.rsquared_adj])
    small = run_operations(_operations(spec.resolve({"adim8_n": 20}))).tables["abc"]["deger"]
    assert small.tolist() == pytest.approx([1 - (1 - fit.rsquared) * 19 / (19 - k)
                                            for fit, k in ((simple, 1), (multi, 3), (wide, 4))])


def test_konu06_alternative_numbers_follow_wage2_and_kielmc(wage2: pd.DataFrame, houses: pd.DataFrame) -> None:
    spec = VARIANTS["konu06"].alternative()
    state = run_operations(_operations(spec))
    s = state.scalars
    short = smf.ols("wage ~ educ", data=wage2).fit()
    long = smf.ols("wage ~ educ + IQ", data=wage2).fit()
    auxiliary = smf.ols("IQ ~ educ", data=wage2).fit()
    assert (s["kisa_x"], s["uzun_x"], s["uzun_z"]) == (pytest.approx(short.params["educ"]),
                                                      pytest.approx(long.params["educ"]), pytest.approx(long.params["IQ"]))
    assert (s["yardimci_sabit"], s["yardimci_egim"]) == (pytest.approx(auxiliary.params["Intercept"]),
                                                        pytest.approx(auxiliary.params["educ"]))
    assert s["katki"] == pytest.approx(long.params["IQ"] * auxiliary.params["educ"])
    assert s["yeniden"] == pytest.approx(short.params["educ"], abs=1e-9)  # kısa = uzun + β̂ · δ̂ (örneklemde tam)
    assert s["x_genis"] == pytest.approx(smf.ols("wage ~ educ + exper + tenure + IQ", data=wage2).fit().params["educ"])
    kww = run_operations(spec.resolve({"adim1_z": "KWW"}).operations_through(2)).scalars
    assert kww["katki"] == pytest.approx(kww["kisa_x"] - kww["uzun_x"])
    assert kww["uzun_z"] == pytest.approx(smf.ols("wage ~ educ + KWW", data=wage2).fit().params["KWW"])
    pairs = state.frames["ciftler"]
    assert (pairs["beta1"] + 12 * pairs["beta2"]).tolist() == pytest.approx([60.0] * len(pairs))
    # Bir değişkenle 12 katının korelasyonu 1'dir; kayan noktada 1 − 10⁻¹⁵ çıkabilir (Windows'ta 0,9999999999999991)
    assert s["r_ay"] == pytest.approx(1.0, abs=1e-12)
    assert "0,4153 ve 4,9833" in spec.step(4).explanation  # statsmodels'ın en küçük normlu çözümü: b/145, 12b/145
    assert short.params["educ"] / 145 == pytest.approx(0.41527, abs=5e-6)
    vif = state.tables["vif"]["deger"]
    for name, label in (("area", "Konut büyüklüğü"), ("rooms", "Oda sayısı"), ("baths", "Banyo sayısı")):
        others = " + ".join(item for item in ("area", "rooms", "baths") if item != name)
        r2 = smf.ols(f"{name} ~ {others}", data=houses).fit().rsquared
        assert s[f"r2_{name}"] == pytest.approx(r2) and vif[label] == pytest.approx(1 / (1 - r2))
    assert s["r_cift"] == pytest.approx(np.corrcoef(houses["area"], houses["baths"])[0, 1])


def test_konu07_alternative_numbers_follow_wage2_and_kielmc(wage2: pd.DataFrame, houses: pd.DataFrame) -> None:
    from scipy import stats

    spec = VARIANTS["konu07"].alternative()
    state = run_operations(_operations(spec))
    s = state.scalars
    model = smf.ols("wage ~ educ + exper + tenure", data=wage2).fit()
    # Sayısal örnekler p-değeri en büyük katsayıdan (kıdem) kurulur; değerleri yuvarlanmıştır
    assert model.pvalues.drop("Intercept").idxmax() == "tenure"
    controls = {control.key: control.default for control in spec.controls}
    assert (controls["adim1_b"], controls["adim1_se"]) == (round(model.params["tenure"], 2), round(model.bse["tenure"], 1))
    assert (s["t_ornek"], s["kritik_buyuk"]) == (pytest.approx(8.26 / 2.5), pytest.approx(stats.norm.ppf(0.975)))
    critical = state.tables["kritik_degerler"]["deger"]
    assert critical[["10", "20", "30", "60", "120"]].tolist() == pytest.approx(
        [stats.t.ppf(0.975, df) for df in (10, 20, 30, 60, 120)])
    b, se, df = model.params["educ"], model.bse["educ"], model.df_resid
    assert (s["b2"], s["sh2"], s["sd_artik"]) == (pytest.approx(b), pytest.approx(se), 931)
    assert (s["t0_2"], s["ta_2"]) == (pytest.approx(b / se), pytest.approx((b - 65) / se))
    assert s["kritik_2"] == pytest.approx(stats.t.ppf(0.975, df))
    assert (s["t3"], s["p3"]) == (pytest.approx(model.tvalues["tenure"]), pytest.approx(model.pvalues["tenure"]))
    assert (s["alt4"], s["ust4"]) == (pytest.approx(8.26 - stats.norm.ppf(0.975) * 2.5),
                                      pytest.approx(8.26 + stats.norm.ppf(0.975) * 2.5))
    table = state.tables["tablo74"]
    interval = model.conf_int()
    np.testing.assert_allclose(table[["katsayi", "sh", "t", "p"]].to_numpy(float),
                               np.column_stack([model.params, model.bse, model.tvalues, model.pvalues])[1:],
                               rtol=1e-10, atol=0)
    np.testing.assert_allclose(table[["alt", "ust"]].to_numpy(float), interval.iloc[1:].to_numpy(), rtol=1e-10, atol=0)
    assert s["p_secilen"] == pytest.approx(stats.t.sf(model.tvalues["tenure"], df))
    assert s["p_iki6"] == pytest.approx(model.pvalues["tenure"])
    left = run_operations(_operations(spec.resolve({"adim6_yon": "sol"}))).scalars
    assert left["p_secilen"] == pytest.approx(stats.t.cdf(model.tvalues["tenure"], df))
    logged = smf.ols("lwage ~ educ + exper + tenure", data=wage2).fit()
    article = state.tables["tablo75"]
    assert article.loc["educ", "(2) ln(Aylık kazanç)"] == pytest.approx(logged.params["educ"])
    assert article.loc["educ_sh", "(2) ln(Aylık kazanç)"] == pytest.approx(logged.bse["educ"])
    assert (s["fark_tahmin"], s["fark_alt"], s["fark_ust"]) == (
        pytest.approx(4 * b), pytest.approx(4 * interval.loc["educ", 0]), pytest.approx(4 * interval.loc["educ", 1]))
    house = smf.ols("price ~ area + rooms + baths", data=houses).fit()
    second = state.tables["tablo76"]
    assert second["p"].tolist() == pytest.approx(house.pvalues[["area", "rooms", "baths"]].tolist())
    assert house.pvalues.drop("Intercept").idxmax() == "rooms" and 0.2 < house.pvalues["rooms"] < 0.3
    text = spec.step(9).note_for(run_operations(spec.operations_through(9)), spec.normalize({}))
    assert "oda sayısı katsayısı" in text and "tahminin belirsiz olduğudur" in text
    assert (s["b10"], s["alt10"], s["ust10"]) == (pytest.approx(b), pytest.approx(interval.loc["educ", 0]),
                                                  pytest.approx(interval.loc["educ", 1]))


# --- Kendi verin: örnek dosya ----------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(SAMPLE_CHOICES))
def test_the_sample_file_runs_every_step_and_passes_its_checks(name: str) -> None:
    spec = _sample_spec(name)
    assert spec.source == "kendi" and run_lab(spec).all_passed
    assert [step.number for step in spec.steps] == [step.number for step in get_lab(_topic(name)).steps]
    # İlk veri işlemi dosyayı okur (Konu 7'nin ilk adımı notlardaki gibi verisiz bir sayısal örnektir)
    first = next(op for step in spec.steps for op in step.operations
                 if isinstance(op, (ReadFile, LoadWooldridge, InlineData)))
    assert isinstance(first, ReadFile)
    assert script_filename(spec, "Python") == f"ikt305_{_topic(name)}_kendi_verim.py"
    script = render_script(spec, "R")
    assert "kendi veriniz" in script and f'veri_dosyasi <- "{XLSX}"' in script and 'install.packages("readxl")' in script


@pytest.mark.parametrize("name", list(SAMPLE_CHOICES))
def test_generated_python_reads_the_uploaded_excel(name: str, tmp_path: Path) -> None:
    spec = _sample_spec(name)
    data = (XLSX, _sample_data(name))
    result, path = _python(spec, tmp_path, data)
    _reproduce(spec, result, path, tmp_path, data)


@pytest.mark.parametrize("name", list(SAMPLE_CHOICES))
def test_generated_r_reads_the_uploaded_excel(name: str, tmp_path: Path, rscript_readxl: str,
                                              r_environment: dict[str, str]) -> None:
    spec = _sample_spec(name)
    data = (XLSX, _sample_data(name))
    result, path = _run_script(spec, "R", tmp_path, [rscript_readxl], r_environment, data)
    _reproduce(spec, result, path, tmp_path, data)


@pytest.mark.parametrize("name", list(SAMPLE_CHOICES))
def test_a_turkish_csv_of_the_sample_gives_the_same_numbers_in_both_languages(name: str, tmp_path: Path, rscript: str,
                                                                            r_environment: dict[str, str]) -> None:
    spec, data = _csv_spec(name)
    excel = {(step.number, check.label): check.expected for step in _sample_spec(name).steps for check in step.checks}
    values = {(step.number, check.label): check.expected for step in spec.steps for check in step.checks}
    assert values.keys() == excel.keys()
    for key, value in values.items():
        assert value == pytest.approx(excel[key], rel=1e-12, abs=1e-12), key
    result, path = _python(spec, tmp_path / "python", data)
    _reproduce(spec, result, path, tmp_path / "python", data)
    result, path = _run_script(spec, "R", tmp_path / "r", [rscript], r_environment, data)
    _reproduce(spec, result, path, tmp_path / "r", data)


def _konu00_with_blanks() -> tuple[LabSpec, tuple[str, bytes], tuple[str, ...]]:
    """Ek değişkende boş hücre, sürekli açıklayıcı (25'ten çok değer) ve ayrı koşul sütunu; Türkçe CSV."""

    frame = VARIANTS["konu00"].custom.sample().copy()
    frame["Yaş"] = frame["Yaş"] + 0.5
    frame["Önceki yıllık kazanç (bin TL)"] = frame["Önceki yıllık kazanç (bin TL)"].astype(float)
    frame.loc[[3, 10, 20], "Önceki yıllık kazanç (bin TL)"] = np.nan
    frame.loc[[7], "Yıllık kazanç (bin TL)"] = np.nan
    data = ("﻿" + frame.to_csv(sep=";", decimal=",", index=False)).encode("utf-8")
    table = K.read_upload("bos.csv", data)
    choices = CustomChoices(roles={"sonuc": "Yıllık kazanç (bin TL)", "aciklayici": "Yaş", "kosul": "Eğitim yılı"},
                            extra=("Önceki yıllık kazanç (bin TL)",))
    case, notes = custom_case(VARIANTS["konu00"].custom, table, choices)
    return VARIANTS["konu00"].custom.build(case), ("bos.csv", data), notes


def test_blank_cells_drop_rows_only_for_required_roles_and_steps_use_complete_cases(tmp_path: Path, rscript: str,
                                                                                    r_environment) -> None:
    spec, data, notes = _konu00_with_blanks()
    assert spec.steps[0].operations[0].dropped == 1 and any("3 boş hücre" in note for note in notes)
    assert spec.step(8).checks[1].expected == pytest.approx(
        pd.read_csv(io.BytesIO(data[1]), sep=";", decimal=",")["Yıllık kazanç (bin TL)"].mean())
    complete = [op for step in spec.steps for op in step.operations if isinstance(op, CompleteCases)]
    assert complete and all("onceki_yillik_kazanc_bin" in op.columns for op in complete)
    changed = spec.resolve({"adim9_x": "onceki_yillik_kazanc_bin"})
    assert isinstance(changed.step(9).operations[0], CompleteCases)
    for folder, chosen in (("varsayilan", spec), ("secim", changed)):
        result, path = _python(chosen, tmp_path / folder / "python", data)
        _reproduce(chosen, result, path, tmp_path / folder / "python", data)
        result, path = _run_script(chosen, "R", tmp_path / folder / "r", [rscript], r_environment, data)
        _reproduce(chosen, result, path, tmp_path / folder / "r", data)


def test_steps_without_their_roles_say_what_they_need() -> None:
    frame = VARIANTS["konu00"].custom.sample()
    data = K.sample_excel(frame.assign(**{"Yaş": frame["Yaş"] + 0.5}))
    table = K.read_upload("veri.xlsx", data)
    case, _ = custom_case(VARIANTS["konu00"].custom, table, CustomChoices(
        roles={"sonuc": "Yıllık kazanç (bin TL)", "aciklayici": "Yaş"}))
    spec = VARIANTS["konu00"].custom.build(case)
    assert not spec.step(6).operations and "koşul sütunu" in spec.step(6).explanation
    assert not spec.step(7).operations and "koşul sütunu" in spec.step(7).explanation
    assert not spec.step(1).controls and "İki kategorili gösterge" in spec.step(1).note_for(
        run_operations(spec.operations_through(1)), {})
    custom = VARIANTS["konu02"].custom
    table = K.read_upload(XLSX, _sample_data("konu02_deney"), "Deney")
    case, _ = custom_case(custom, table, CustomChoices(roles={"sayisal": "Yaş"}))
    spec = custom.build(case)
    assert run_lab(spec).all_passed and spec.step(1).operations
    for number, need in ((2, "dönem"), (3, "dönem"), (4, "birim kimliği"), (5, "atama sütunu")):
        assert not spec.step(number).operations and need in spec.step(number).explanation, number


def test_role_rules_are_enforced() -> None:
    custom = VARIANTS["konu02"].custom
    table = K.read_upload(XLSX, _sample_data("konu02_panel"), "Panel")
    with pytest.raises(K.UploadError, match="için bir sütun seçin"):
        custom_case(custom, table, CustomChoices(roles={"sayisal": None}))
    with pytest.raises(K.UploadError, match="dönem sütunuyla birlikte"):
        custom_case(custom, table, CustomChoices(roles={"sayisal": "Büyüme (%)", "birim": "İl"}))
    deney = K.read_upload(XLSX, _sample_data("konu02_deney"), "Deney")
    with pytest.raises(K.UploadError, match="birlikte çalışır"):
        custom_case(custom, deney, CustomChoices(roles={"sayisal": "Yaş", "atama": "Grup"}))
    duplicated = K.read_upload("tekrar.csv", "İl;Yıl;Oran\nA;1;2\nA;1;3\nB;1;4\nB;2;5\nA;2;6\n".encode())
    with pytest.raises(K.UploadError, match="birden çok kez"):
        custom_case(custom, duplicated, CustomChoices(roles={"sayisal": "Oran", "donem": "Yıl", "birim": "İl"}))
    zero = VARIANTS["konu00"].custom
    sample = K.read_upload(XLSX, _sample_data("konu00"))
    with pytest.raises(K.UploadError, match="farklı sütunlar"):
        custom_case(zero, sample, CustomChoices(roles={"sonuc": "Yaş", "aciklayici": "Yaş"}))
    with pytest.raises(K.UploadError, match="en çok 25 farklı"):
        custom_case(zero, sample, CustomChoices(roles={"sonuc": "Yaş", "aciklayici": "Eğitim yılı",
                                                       "kosul": "Önceki yıllık kazanç (bin TL)"}))
    with pytest.raises(K.UploadError, match="En çok 4 ek sütun"):
        custom_case(zero, sample, CustomChoices(roles={"sonuc": "Yaş", "aciklayici": "Eğitim yılı"},
                                                extra=("Yıllık kazanç (bin TL)", "Önceki yıllık kazanç (bin TL)", "Cinsiyet",
                                                       "Yaş", "Eğitim yılı")))


def test_konu02_suggests_the_panel_and_experiment_roles() -> None:
    from core.labs.ornek_konu02 import suggest

    panel = K.read_upload(XLSX, _sample_data("konu02_panel"), "Panel")
    assert suggest(panel) == {"donem": "Yıl", "birim": "İl", "sayisal": "İşsizlik oranı (%)"}
    deney = K.read_upload(XLSX, _sample_data("konu02_deney"), "Deney")
    assert suggest(deney) == {"atama": "Grup", "deney_sonuc": "Yıllık kazanç (bin TL)",
                              "sayisal": "Yıllık kazanç (bin TL)"}
    assert K.time_like("Yıl") and K.time_like("Dönem no") and not K.time_like("Yıllık kazanç (bin TL)")


def test_user_names_are_escaped_in_texts() -> None:
    rows = ["Puan $;Not_1;Grup*"] + [f"{50 + 7 * index % 23};{index % 5 + 1};{'A' if index % 2 else 'B'}"
                                     for index in range(12)]
    table = K.read_upload("isaret.csv", ("\n".join(rows) + "\n").encode())
    case, _ = custom_case(VARIANTS["konu00"].custom, table, CustomChoices(
        roles={"sonuc": "Puan $", "aciklayici": "Not_1", "gosterge": "Grup*"}))
    spec = VARIANTS["konu00"].custom.build(case)
    text = " ".join(step.explanation for step in spec.steps)
    assert "Grup\\*" in text and "Not\\_1" in text and "Puan \\$" in text
    assert md("1. sınıf") == "1\\. sınıf" and md("a|b") == "a\\|b"


# --- Kendi verin: dağınık dosyalar uçtan uca ----------------------------------------------------------------

def _own(topic: str, frame: pd.DataFrame, roles: dict, extra: tuple = (), picks: dict | None = None):
    """Türkçe CSV (noktalı virgül, ondalık virgül) olarak yüklenen bir tablodan kendi veri uygulaması."""

    data = ("﻿" + frame.to_csv(sep=";", decimal=",", index=False)).encode("utf-8")
    table = K.read_upload("veri.csv", data)
    custom = VARIANTS[topic].custom
    case, _ = custom_case(custom, table, CustomChoices(roles=roles, extra=extra, picks=picks or {}))
    return custom.build(case), ("veri.csv", data)


def _every_step(spec: LabSpec, change: dict | None = None) -> LabState:
    """Bütün adımlar (``change`` seçimiyle) hesaplanır, her tablo ekranda temiz görünür ve her yorum metni yazılır."""

    resolved = spec.resolve(change or {})
    state = run_operations(_operations(resolved))
    _screen(_operations(resolved), state, spec.label)
    choices = spec.normalize(change or {})
    for step in spec.steps:
        assert _clean(step.explanation), step.number
        if step.note_for is not None and step.operations:
            partial = run_operations(resolved.operations_through(step.number))
            assert _clean(step.note_for(partial, choices)), (change, step.number)
    return state


def _scripts_agree(spec: LabSpec, data: tuple[str, bytes], folder: Path, rscript: str, r_environment,
                   warnings: bool = False) -> None:
    """İki dilde betik dosyayı okur ve bütün kontrolleri geçer; ``warnings``: R'nin uyarı yazmasına izin verilir
    (ör. tam uyumda "essentially perfect fit")."""

    for language, command, environment in (("Python", [sys.executable], _python_environment()),
                                           ("R", [rscript], r_environment)):
        result, path = _run_script(spec, language, folder / language, command, environment, data)
        if warnings:
            assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
            assert result.stdout.count("  OK   ") == _checks(spec) and "HATA" not in result.stdout, result.stdout[-1500:]
            assert sorted(item.name for item in (folder / language).iterdir()) == sorted([path.name, data[0]])
        else:
            _reproduce(spec, result, path, folder / language, data)


RESERVED = pd.DataFrame({
    "Gözlem": [3.5, 1.2, 4.8, 2.2, 5.1, 3.3, 2.9, 4.4],
    "Sapma": [10, 12, 9, 15, 11, 14, 13, 16],
    "Yüzde": [0.2, 0.5, 0.1, 0.9, 0.4, 0.8, 0.6, 0.7],
    "T": [1.5, 0.5, 1.0, 2.5, 0.0, 3.0, 1.0, 2.0],
    "Puan": [55.0, 61.5, 48.0, 72.5, 58.0, 69.0, 63.5, 75.0],
})


@pytest.mark.parametrize("topic, roles, extra", [
    ("konu00", {"sonuc": "Puan", "aciklayici": "Sapma"}, ("Gözlem", "Yüzde", "T")),
    ("konu01", {"sonuc": "Puan", "aciklayici": "Gözlem"}, ("Sapma", "Yüzde", "T")),
    ("konu02", {"sayisal": "Puan"}, ("Gözlem", "Yüzde", "Sapma")),
    *((topic, {"sonuc": "Puan", "aciklayici": "Gözlem"}, ("Sapma", "Yüzde", "T"))
      for topic in ("konu03", "konu04", "konu05", "konu06", "konu07")),
])
def test_columns_named_like_the_apps_own_columns_keep_their_names(topic, roles, extra, tmp_path: Path, rscript: str,
                                                                 r_environment) -> None:
    """"Gözlem", "Sapma", "Yüzde", "T" gibi adlar uygulamanın türettiği sütunlarla ve sabit etiketli tablo sütunlarıyla
    karışmaz: kod adları ``gozlem_2`` gibi olur, ekranda dosyadaki ad görünür, tablolarda sütun adları tekildir."""

    spec, data = _own(topic, RESERVED, roles, extra)
    read = _read(spec)
    codes = {original: name for name, original, _ in read.columns}
    assert codes["Gözlem"] == "gozlem_2" and codes["Sapma"] == "sapma_2" and codes["Yüzde"] == "yuzde_2"
    assert all(spec.label(name) == original for original, name in codes.items())
    assert not K.RESERVED_CODES & set(codes.values())
    _every_step(spec)
    _scripts_agree(spec, data, tmp_path, rscript, r_environment)


def test_reserved_code_names_cover_the_fixed_table_labels() -> None:
    from topics import lab_ui

    fixed = (set(lab_ui._COLUMN_LABELS) | set(lab_ui._DESCRIBE_LABELS) | set(lab_ui._PANEL_LABELS)
             | set(lab_ui._COEF_LABELS) | set(lab_ui._MODEL_LABELS) | set(lab_ui._BOX_LABELS))
    assert {name for name in fixed if name.islower()} <= K.RESERVED_CODES


def test_code_names_follow_the_file_not_the_roles() -> None:
    frame = pd.DataFrame({"Gelir (TL)": [5.0, 7, 6, 9, 8, 4], "Gelir TL": [1.0, 3, 2, 5, 4, 2],
                          "Yaş": [30, 41, 25, 52, 38, 29]})
    for roles, extra in (({"sonuc": "Gelir TL", "aciklayici": "Yaş"}, ("Gelir (TL)",)),
                         ({"sonuc": "Gelir (TL)", "aciklayici": "Yaş"}, ("Gelir TL",))):
        spec, _ = _own("konu01", frame, roles, extra)
        codes = {original: name for name, original, _ in _read(spec).columns}
        assert codes == {"Gelir (TL)": "gelir_tl", "Gelir TL": "gelir_tl_2", "Yaş": "yas"}


def test_a_constant_pair_is_not_offered_and_every_step_stays_clean(tmp_path: Path) -> None:
    """Ek değişkenin dolu olduğu gözlemlerde sonuç sabitse (5, 5, 5) korelasyon ve eğim tanımsızdır: o değişken
    ikinci değişken ya da açıklayıcı olarak sunulmaz; ekranda nan ya da sonsuz görünmez."""

    frame = pd.DataFrame({"Puan": [5.0, 5, 5, 1, 2, 3, 4, 6], "Saat": [1.0, 2, 3, 4, 5, 6, 7, 9],
                          "Ek": [1.0, 2, 3, None, None, None, None, None]})
    spec, _ = _own("konu00", frame, {"sonuc": "Puan", "aciklayici": "Saat"}, ("Ek",))
    controls = {control.key: control for control in spec.controls}
    assert [value for value, _ in controls["adim5_x"].options] == ["saat"]
    assert [value for value, _ in controls["adim9_x"].options] == ["saat"]
    assert not any("puan" in value and "ek" in value for value, _ in controls["adim4_desen"].options)
    _every_step(spec)
    for change in _single_changes(spec):
        _every_step(spec, change)
    spec, _ = _own("konu01", frame, {"sonuc": "Puan", "aciklayici": "Saat"}, ("Ek",))
    assert [value for value, _ in {c.key: c for c in spec.controls}["adim4_x"].options] == ["saat"]
    _every_step(spec)


@pytest.mark.parametrize("topic, step", [("konu00", 9), ("konu01", 4)])
def test_an_exact_fit_drops_the_standard_error_checks_and_says_why(topic, step, tmp_path: Path, rscript: str,
                                                                  r_environment) -> None:
    """y = 2x + 1: standart hata sıfıra çok yakın, t ve p yuvarlama gürültüsüdür; bu değerler karşılaştırılmaz."""

    x = [1.0, 2, 3, 5, 8, 13, 21, 34]
    frame = pd.DataFrame({"Y": [2 * value + 1 for value in x], "X": x})
    spec, data = _own(topic, frame, {"sonuc": "Y", "aciklayici": "X"})
    quantities = {check.target.quantity for check in spec.step(step).checks
                  if isinstance(check.target, (CoefTarget, ModelTarget))}
    assert quantities <= {"coef", "r2", "nobs"} and "coef" in quantities
    assert not any(str(getattr(check.target, "row", "")).endswith("_sh") for check in spec.step(step).checks)
    state = run_operations(spec.operations_through(step))
    assert "neredeyse tam bir doğrunun" in spec.step(step).note_for(state, spec.normalize({}))
    _scripts_agree(spec, data, tmp_path, rscript, r_environment, warnings=True)


def test_large_numbers_are_compared_with_a_relative_margin(tmp_path: Path, rscript: str, r_environment) -> None:
    """Gelir 10¹⁰ düzeyinde: kovaryans 10¹⁶ düzeyindedir ve iki yazılımın son basamak farkı mutlak payı aşar."""

    frame = pd.DataFrame({
        "Gelir (TL)": [5_200_000_000, 16_100_000_000, 9_850_000_000, 7_300_000_000, 12_400_000_000, 6_050_000_000,
                       14_900_000_000, 8_800_000_000],
        "Nüfus": [10_500_000, 88_000_000, 45_250_000, 31_000_000, 70_100_000, 22_300_000, 81_900_000, 39_400_000],
    })
    spec, data = _own("konu00", frame, {"sonuc": "Gelir (TL)", "aciklayici": "Nüfus"})
    assert abs(run_operations(_operations(spec)).scalars["kovaryans"]) > 1e16
    _scripts_agree(spec, data, tmp_path, rscript, r_environment)
    assert "1e-9 * abs(beklenen)" in render_script(spec, "Python") and "1e-9 * abs(beklenen)" in render_script(spec, "R")


def test_a_zero_slope_is_written_as_zero() -> None:
    frame = pd.DataFrame({"Y": [1.0, 3, 5, 7, 7, 5, 3, 1], "X": [1.0, 2, 3, 4, 5, 6, 7, 8]})
    spec, _ = _own("konu01", frame, {"sonuc": "Y", "aciklayici": "X"})
    state = run_operations(spec.operations_through(5))
    text = spec.step(5).note_for(state, spec.normalize({}))
    assert "Eğim dört basamakta sıfırdır" in text and "− 0,0000" not in text and "daha aynıdır" not in text
    assert "doğrusal bir ortalama ilişki görülmez" in spec.step(6).explanation
    spec, _ = _own("konu00", frame, {"sonuc": "Y", "aciklayici": "X"})
    state = run_operations(spec.operations_through(9))
    text = spec.step(9).note_for(state, spec.normalize({}))
    assert "eğim dört basamakta sıfırdır" in text and "daha aynıdır" not in text


# --- Kendi verin: Konu 3–7 -------------------------------------------------------------------------------------

REGRESSION_TOPICS = ("konu03", "konu04", "konu05", "konu06", "konu07")
EMPLOYED = {"sonuc": "Yıllık kazanç (bin TL)", "aciklayici": "Eğitim yılı"}


def _people(topic: str, roles: dict, extra: tuple = (), picks: dict | None = None) -> LabSpec:
    """Konu 3'ün örnek dosyası (200 kişi; işsiz kalanların kazancı 0, Grup sütunu) ile kendi veri uygulaması."""

    table = K.read_upload(XLSX, K.sample_excel(VARIANTS["konu03"].custom.sample()))
    custom = VARIANTS[topic].custom
    case, _ = custom_case(custom, table, CustomChoices(roles=roles, extra=extra, picks=picks or {}))
    return custom.build(case)


def test_konu03_to_07_steps_without_their_roles_say_what_they_need() -> None:
    """Sürekli açıklayıcı, iki kategorili sütun yok, ek değişken yok ya da sonuçta sıfır: o adımlar ne gerektiğini
    yazar; öteki adımlar kurulur ve kontrollerini geçer."""

    continuous = _people("konu03", {"sonuc": "Yıllık kazanç (bin TL)", "aciklayici": "Önceki yıllık kazanç (bin TL)"})
    assert not continuous.step(2).operations and "en çok 25 farklı değeri" in continuous.step(2).explanation
    assert not continuous.step(6).operations and "iki kategorili bir sütun" in continuous.step(6).explanation
    with_zeros = _people("konu04", EMPLOYED, ("Yaş",))
    for number in (7, 8, 9):
        assert not with_zeros.step(number).operations, number
    assert "bütün değerleri pozitif olmalı" in with_zeros.step(7).explanation
    assert "sonucun bütün değerleri pozitif olmalı" in with_zeros.step(8).explanation
    alone = _people("konu05", EMPLOYED)
    assert not alone.step(4).operations and "en az iki açıklayıcı değişken gerekir" in alone.step(4).explanation
    state = run_operations(alone.operations_through(5))
    assert "log sonuç sütunu kurulmaz" in alone.step(5).note_for(state, alone.normalize({}))
    omitted = _people("konu06", EMPLOYED)
    for number in (1, 2, 3):
        assert not omitted.step(number).operations and "en az bir ek sayısal değişken" in omitted.step(number).explanation
    assert not omitted.step(5).operations and "en az iki açıklayıcı değişken" in omitted.step(5).explanation
    testing = _people("konu07", EMPLOYED, ("Yaş",))
    outcome = {control.key: control for control in testing.controls}["adim7_bagimli"]
    assert [value for value, _ in outcome.options] == ["yillik_kazanc_bin_tl"]
    assert "log sonuç sütunu kurulmaz" in testing.step(8).takeaway
    assert "ml" not in {op.name for op in testing.step(8).operations if hasattr(op, "name")}
    for spec in (continuous, with_zeros, alone, omitted, testing):
        assert run_lab(spec).all_passed, spec.topic_key
        _every_step(spec)


def _blank_frame() -> pd.DataFrame:
    """Çalışanlar: ek değişkende üç, sonuçta bir boş hücre (dört ayrı satır); seçilmeyen sütundaki boşluk satır
    çıkarmaz."""

    frame = VARIANTS["konu05"].custom.sample().copy()
    frame["Önceki yıllık kazanç (bin TL)"] = frame["Önceki yıllık kazanç (bin TL)"].astype(float)
    frame.loc[[3, 10, 20], "Önceki yıllık kazanç (bin TL)"] = np.nan
    frame.loc[[7], "Yıllık kazanç (bin TL)"] = np.nan
    frame["Not"] = np.where(np.arange(len(frame)) % 4 == 0, np.nan, 1.0)
    return frame


@pytest.mark.parametrize("topic", REGRESSION_TOPICS)
def test_konu03_to_07_drop_rows_with_a_blank_in_any_selected_column(topic: str) -> None:
    frame = _blank_frame()
    roles = dict(EMPLOYED)
    spec, _ = _own(topic, frame, roles, ("Yaş", "Önceki yıllık kazanç (bin TL)"))
    read = _read(spec)
    assert read.dropped == 4 and set(read.required) == {code for code, _, _ in read.columns}
    assert len(read.rows) == len(frame) - 4 and "not" not in {code for code, _, _ in read.columns}
    complete = frame.dropna(subset=["Yıllık kazanç (bin TL)", "Eğitim yılı", "Yaş", "Önceki yıllık kazanç (bin TL)"])
    complete = complete.set_axis(["y", "x", "a", "o", "not"], axis=1)
    resolved = spec.resolve({})
    state = run_operations(_operations(resolved))
    fitted = [op for op in _operations(resolved) if isinstance(op, OLS) and op.frame in ("veri", "ikinci")]
    assert fitted and all(state.models[op.name].nobs == len(complete) for op in fitted)
    # Dosyanın sütunlarıyla kurulan her model, boş satırları çıkarılmış veriyle bağımsız hesapla aynıdır
    short = {"yillik_kazanc_bin_tl": "y", "egitim_yili": "x", "yas": "a", "onceki_yillik_kazanc_bin": "o"}
    raw = [op for op in fitted if {op.outcome, *op.regressors} <= set(short)]
    assert raw
    for op in raw:
        expected = smf.ols(f"{short[op.outcome]} ~ {' + '.join(short[name] for name in op.regressors)}",
                           data=complete).fit()
        np.testing.assert_allclose(state.models[op.name].params.to_numpy(), expected.params.to_numpy(), rtol=1e-9,
                                   atol=1e-9, err_msg=op.name)
    table = K.read_upload("veri.csv", ("﻿" + frame.to_csv(sep=";", decimal=",", index=False)).encode("utf-8"))
    _, notes = custom_case(VARIANTS[topic].custom, table, CustomChoices(roles=roles, extra=(
        "Yaş", "Önceki yıllık kazanç (bin TL)")))
    assert any("Seçilen sütunlardan" in note and "4 satır çıkarıldı" in note for note in notes), notes


@pytest.mark.parametrize("topic", ("konu05", "konu07"))
def test_konu05_and_07_blank_rows_are_dropped_the_same_way_in_both_languages(topic: str, tmp_path: Path, rscript: str,
                                                                            r_environment) -> None:
    spec, data = _own(topic, _blank_frame(), dict(EMPLOYED), ("Yaş", "Önceki yıllık kazanç (bin TL)"))
    _scripts_agree(spec, data, tmp_path, rscript, r_environment)


SPECIAL = pd.DataFrame({
    "Puan $": [52.0, 61, 47, 70, 58, 66, 49, 73, 55, 64, 60, 68, 51, 75, 57, 63],
    "Not_1": [1, 2, 1, 4, 2, 3, 1, 5, 2, 3, 3, 4, 1, 5, 2, 4],
    "Saat*": [2.5, 3.0, 1.5, 4.0, 2.0, 3.5, 3.0, 4.5, 1.0, 2.5, 4.0, 3.0, 2.0, 5.0, 1.5, 3.5],
    "Grup|": ["B*", "A_x", "A_x", "B*", "A_x", "B*", "A_x", "B*", "B*", "A_x", "A_x", "B*", "A_x", "B*", "B*", "A_x"],
})


@pytest.mark.parametrize("topic", REGRESSION_TOPICS)
def test_user_names_are_escaped_in_konu03_to_07_texts(topic: str) -> None:
    roles = {"sonuc": "Puan $", "aciklayici": "Not_1"}
    picks = None
    if topic == "konu03":
        roles["gosterge"], picks = "Grup|", {"gosterge": "B*"}
    spec, _ = _own(topic, SPECIAL, roles, ("Saat*",), picks)
    choices = spec.normalize({})
    texts = [step.explanation for step in spec.steps]
    for step in spec.steps:
        if step.note_for is not None and step.operations:
            texts.append(step.note_for(run_operations(spec.operations_through(step.number)), choices))
        elif step.takeaway:
            texts.append(step.takeaway)
    text = " ".join(texts)
    assert "Not\\_1" in text  # her konu temel açıklayıcıyı anar; sonucu ve ek değişkenleri anmayan konular var
    assert ("Puan \\$" in text) == (topic != "konu06") and ("Saat\\*" in text) == (topic in ("konu05", "konu06", "konu07"))
    for raw in ("Puan $", "Not_1", "Saat*", "Grup|", "B*", "A_x"):
        assert raw not in text, raw
    if topic == "konu03":
        assert "B\\* = 1" in text and "A\\_x = 0" in text
    _every_step(spec)


EXACT = pd.DataFrame({"X": [1.0, 2, 3, 5, 8, 13, 21, 34, 4, 9], "Z": [3.0, 1, 4, 1, 5, 9, 2, 6, 5, 3]})
EXACT["Y"] = 1 + 2 * EXACT["X"] + 0.5 * EXACT["Z"]
EXACT["Basit"] = 2 * EXACT["X"] + 1
FRAGILE_SCALARS = {"sh2", "t0_2", "ta_2", "sh3", "t3", "p3", "t6", "p_secilen", "p_iki6", "fark_alt", "fark_ust", "sh10",
                   "t10", "p10", "alt10", "ust10"}


def _fragile(check: Check) -> bool:
    target = check.target
    if isinstance(target, CoefTarget):
        return target.quantity in ("se", "t", "p", "ci_low", "ci_high")
    if isinstance(target, ModelTarget):
        return target.quantity in ("f", "f_p")
    if isinstance(target, ScalarTarget):
        return target.name in FRAGILE_SCALARS
    return isinstance(target, TableTarget) and (target.column in ("sh", "t", "p", "alt", "ust")
                                                or str(target.row).endswith("_sh"))


@pytest.mark.parametrize("topic, roles, extra, step, phrase", [
    ("konu03", {"sonuc": "Basit", "aciklayici": "X"}, (), 4, "neredeyse tam bir doğrunun"),
    ("konu05", {"sonuc": "Y", "aciklayici": "X"}, ("Z",), 5, "Model veriye neredeyse tam uyuyor"),
    ("konu06", {"sonuc": "Y", "aciklayici": "X"}, ("Z",), 3, None),
])
def test_konu03_05_06_exact_fit_drops_the_standard_error_checks(topic, roles, extra, step, phrase, tmp_path: Path,
                                                                 rscript: str, r_environment) -> None:
    spec, data = _own(topic, EXACT, roles, extra)
    assert not [check.label for item in spec.steps for check in item.checks if _fragile(check)]
    assert any(isinstance(check.target, CoefTarget) for check in spec.step(step).checks)
    if phrase:
        state = run_operations(spec.operations_through(step))
        assert phrase in spec.step(step).note_for(state, spec.normalize({}))
    _every_step(spec)
    _scripts_agree(spec, data, tmp_path, rscript, r_environment, warnings=True)


def test_konu07_exact_fit_keeps_the_numeric_examples_usable_and_says_why(tmp_path: Path, rscript: str,
                                                                        r_environment) -> None:
    """Uyum tamken standart hata sıfıra çok yakındır: sayısal örneklerin standart hatası katsayının yarısıdır, t
    "|t| > 10⁶" yazılır, standart hataya bağlı kontroller çıkarılır ve metinler nedenini söyler."""

    spec, data = _own("konu07", EXACT, {"sonuc": "Y", "aciklayici": "X"}, ("Z",))
    controls = {control.key: control for control in spec.controls}
    for prefix in ("adim1", "adim4"):
        assert controls[f"{prefix}_b"].default == pytest.approx(2) and controls[f"{prefix}_se"].default == pytest.approx(1)
        assert controls[f"{prefix}_se"].decimals <= 3 and controls[f"{prefix}_se"].minimum > 0
    assert controls["adim2_a"].default == pytest.approx(1) and controls["adim2_a"].decimals <= 3
    assert controls["adim3_terim"].default == "x_2" and "uyum tam olduğu için" in spec.step(1).explanation
    assert not [check.label for step in spec.steps for check in step.checks if _fragile(check)]
    choices = spec.normalize({})
    for number in (2, 3, 5, 6, 7, 9, 10):
        text = spec.step(number).note_for(run_operations(spec.operations_through(number)), choices)
        assert "Model veriye neredeyse tam uyuyor" in text, number
        assert not re.search(r"\d{7,}", text), (number, text)
    assert "|t| > 10⁶" in spec.step(2).note_for(run_operations(spec.operations_through(2)), choices)
    _every_step(spec)
    _scripts_agree(spec, data, tmp_path, rscript, r_environment, warnings=True)


def _scales() -> pd.DataFrame:
    """Ölçekleri çok farklı açıklayıcılar: sonuç bir oran (0–1), eğitim yılı, TL cinsinden gelir (eğimi 10⁻⁷ düzeyinde
    ama gerçek), sonuçla ilişkisiz bir gürültü sütunu."""

    rng = np.random.default_rng(305)
    n = 60
    egitim = rng.choice([5, 8, 11, 12, 14, 16], size=n)
    gelir = np.round(rng.uniform(20_000, 500_000, size=n), 0)
    oran = np.clip(0.1 + 0.02 * egitim + 6e-7 * gelir + rng.normal(0, 0.05, size=n), 0.01, 0.99)
    return pd.DataFrame({"Oran": np.round(oran, 4), "Eğitim yılı": egitim, "Hane geliri (TL)": gelir,
                         "Gürültü": np.round(rng.normal(10, 3, size=n), 2)})


def _texts_of(spec: LabSpec, change: dict | None = None) -> dict[int, str]:
    resolved, choices = spec.resolve(change or {}), spec.normalize(change or {})
    return {step.number: step.note_for(run_operations(resolved.operations_through(step.number)), choices)
            if step.note_for is not None and step.operations else step.takeaway for step in spec.steps}


SCALE_ROLES = {"sonuc": "Oran", "aciklayici": "Eğitim yılı"}


def test_small_unit_coefficients_keep_significant_digits_and_are_not_called_zero() -> None:
    """TL cinsinden gelirin eğimi 10⁻⁷ düzeyindedir: metin sıfır demez, en az üç anlamlı basamak yazar."""

    frame = _scales()
    income = "hane_geliri_tl"
    konu03, _ = _own("konu03", frame, SCALE_ROLES, ("Hane geliri (TL)", "Gürültü"))
    text = _texts_of(konu03, {"adim4_x": income})[5]
    assert "sıfırdır" not in text and re.search(r"yaklaşık 0,0000007\d\d birim daha yüksektir", text), text
    texts = _texts_of(_own("konu05", frame, SCALE_ROLES, ("Hane geliri (TL)", "Gürültü"))[0])
    assert "0,000000 birim" not in texts[1] and "daha aynıdır" not in texts[1]
    assert re.search(r"“Hane geliri \(TL\)” değeri bir birim daha yüksek olan gözlemlerin tahmin edilen “Oran” değeri "
                     r"yaklaşık 0,000000\d{3} birim daha yüksektir", texts[1]), texts[1]
    konu07, _ = _own("konu07", frame, SCALE_ROLES, ("Hane geliri (TL)", "Gürültü"))
    report = _texts_of(konu07, {"adim10_terim": income})[10]
    assert "katsayısı 0,000 " not in report and re.search(r"katsayısı 0,000000\d{3} olarak", report), report
    assert report.startswith("Örnek raporlama paragrafı:\n\n> ") and "rastgele atanmadıysa" in report


def test_konu03_prediction_point_follows_the_main_regressor_and_unequal_gaps_are_explained() -> None:
    spec, _ = _own("konu03", _scales(), SCALE_ROLES, ("Hane geliri (TL)", "Gürültü"))
    x0 = {control.key: control for control in spec.controls}["adim7_x0"]
    assert (x0.minimum, x0.maximum, x0.default, x0.step, x0.integer) == (2, 19, 11, 1, True)
    assert "kaydırıcı Eğitim yılı aralığına göre" in x0.label
    text = _texts_of(spec)[2]  # düzeyler 5, 8, 11, … : komşu iki düzey 3 birim uzakta
    assert "uzaklık 3 birim olduğu için anakütledeki fark 3 · β₁ olur" in text and "3 birimlik uzaklıkta" in text
    assert "her x için aynıdır ve" not in text
    people = VARIANTS["konu03"].custom.sample()
    x0 = {control.key: control for control in _sample_spec("konu03").controls}["adim7_x0"]
    assert x0.integer and (x0.minimum, x0.maximum) == (2, 19) and x0.default == people["Eğitim yılı"].median()


def test_konu07_interval_wording_follows_the_interval() -> None:
    spec, _ = _own("konu07", _scales(), SCALE_ROLES, ("Hane geliri (TL)", "Gürültü"))
    state = run_operations(spec.operations_through(9))
    row = state.tables["tablo76"].loc["gurultu"]
    assert row["p"] > 0.05 and row["alt"] < 0 < row["ust"] and max(-row["alt"], row["ust"]) < 2 * min(-row["alt"],
                                                                                                     row["ust"])
    text = _texts_of(spec)[9]
    assert "hem negatif hem pozitif değerler" in text and "küçük negatif" not in text
    gap = {control.key: control for control in spec.controls}["adim9_fark"]
    assert (gap.minimum, gap.default, gap.maximum) == (1, 4, 11)  # bir yıldan verideki aralığa (5–16)


def test_saturated_or_collinear_own_models_are_rejected_with_a_clear_message() -> None:
    frame = _scales().head(5)
    with pytest.raises(K.UploadError, match="en az 6 gözlem gerekir; analizde 5 gözlem var"):
        _own("konu05", frame.assign(Fazla=[1.0, 4, 2, 8, 3]), SCALE_ROLES, ("Hane geliri (TL)", "Gürültü", "Fazla"))
    twins = _scales().assign(**{"Gelir (bin TL)": lambda data: data["Hane geliri (TL)"] / 1000})
    for topic in ("konu05", "konu06", "konu07"):
        with pytest.raises(K.UploadError, match="tam doğrusal bağlantı"):
            _own(topic, twins, SCALE_ROLES, ("Hane geliri (TL)", "Gelir (bin TL)"))


def test_exact_fit_texts_draw_no_conclusions_from_rounding_noise() -> None:
    """Y = 3 + 2X tam; Z sonuçla ilişkisiz: Z'nin katsayısı yuvarlama gürültüsüdür ve işaretine göre yorum yapılmaz."""

    frame = EXACT.assign(Y=3 + 2 * EXACT["X"])
    konu06, _ = _own("konu06", frame, {"sonuc": "Y", "aciklayici": "X"}, ("Z",))
    texts = _texts_of(konu06)
    assert "hesap hassasiyetinde sıfır" in texts[2] and "yönlü etkileyebilir" not in texts[2]
    assert "değişmez" in texts[3] and "düşer" not in texts[3] and "yükselir" not in texts[3]
    assert "Model veriye neredeyse tam uyuyor" in texts[3]
    konu07, _ = _own("konu07", frame, {"sonuc": "Y", "aciklayici": "X"}, ("Z",))
    texts = _texts_of(konu07)
    for number in (5, 9):
        assert "yuvarlama hatasına bağlıdır" in texts[number] and "reddedilir" not in texts[number], number
        assert "sıfırdan ayrışır" not in texts[number] and "kapsayan aralık" not in texts[number], number


def test_large_log_coefficients_say_the_percent_approximation_is_rough() -> None:
    rng = np.random.default_rng(305)
    share = rng.uniform(0, 1, size=50)
    frame = pd.DataFrame({"Kazanç": np.round(np.exp(1 + 1.4 * share + rng.normal(0, 0.2, size=50)), 3),
                          "Pay": np.round(share, 3), "Yaş": rng.integers(20, 60, size=50)})
    roles = {"sonuc": "Kazanç", "aciklayici": "Pay"}
    caveat = "100 · β̂ yaklaşımı kabadır; tam dönüşüm Konu 9'da"
    assert caveat in _texts_of(_own("konu04", frame, roles, ("Yaş",))[0])[8]
    assert caveat in _texts_of(_own("konu05", frame, roles, ("Yaş",))[0])[5]
    assert caveat in _texts_of(_own("konu07", frame, roles, ("Yaş",))[0])[8]
    assert caveat not in _texts_of(_sample_spec("konu05"))[5]  # örnek dosyada log katsayısı küçük


def test_konu06_vif_text_names_the_most_correlated_pair() -> None:
    rng = np.random.default_rng(305)
    a = rng.normal(size=80)
    frame = pd.DataFrame({"Y": a + rng.normal(size=80), "A": a, "B": rng.normal(size=80),
                          "C": np.round(0.9 * a + 0.3 * rng.normal(size=80), 4), "D": rng.normal(size=80)})
    spec, _ = _own("konu06", frame, {"sonuc": "Y", "aciklayici": "A"}, ("B", "C", "D"))
    text = _texts_of(spec)[5]
    assert "en güçlü ikili doğrusal ilişki “A” ile “C” arasındadır" in text and "ilişkili olmaları olağandır" in text


def _panel_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Birim", "Yıl", "Değer", "Gelir"])


def test_konu02_structure_texts_follow_the_file(tmp_path: Path) -> None:
    once = _panel_frame([(f"F{index}", 2020 + index % 2, 10.0 + index, 3.0 * index + 1) for index in range(12)])
    spec, _ = _own("konu02", once, {"sayisal": "Değer", "donem": "Yıl", "birim": "Birim"}, ("Gelir",))
    assert not spec.step(4).operations and "yalnız bir kez görünür" in spec.step(4).explanation
    assert "aynı birimler birden çok dönemde gözlenir" not in spec.step(1).explanation
    assert spec.step(3).operations
    _every_step(spec)

    unbalanced = _panel_frame([("A", 2018, 10.0, 1.0), ("B", 2018, 40.0, 2.0), ("C", 2018, 70.0, 3.0),
                               ("A", 2019, 11.0, 1.5), ("B", 2019, 41.0, 2.5), ("C", 2019, 71.0, 3.5),
                               ("D", 2019, -80.0, 4.0)])
    spec, _ = _own("konu02", unbalanced, {"sayisal": "Değer", "donem": "Yıl", "birim": "Birim"}, ("Gelir",))
    state = run_operations(spec.operations_through(4))
    assert "eşit olmayabilir" in spec.step(3).note_for(state, spec.normalize({}))
    assert "dengesizdir" in spec.step(4).note_for(state, spec.normalize({}))
    single = spec.resolve({"adim4_gosterim": "D"})
    assert "yalnız bir dönemde" in single.step(4).note_for(run_operations(single.operations_through(4)),
                                                            spec.normalize({"adim4_gosterim": "D"}))
    _every_step(spec)

    by_period = _panel_frame([(unit, year, 10.0 * index + year - 2018, float(index))
                              for year in (2018, 2019, 2020) for index, unit in enumerate("ABCD")])
    spec, _ = _own("konu02", by_period, {"sayisal": "Değer", "donem": "Yıl", "birim": "Birim"}, ("Gelir",))
    state = run_operations(spec.operations_through(4))
    assert "dönemlere göre dizilmiş" in spec.step(4).note_for(state, spec.normalize({}))
    assert "dengeli bir paneldir" in spec.step(3).note_for(state, spec.normalize({}))
    assert "dengelidir" in spec.step(4).note_for(state, spec.normalize({}))
    _every_step(spec)


def test_konu02_blank_cells_in_shown_rows_and_role_conflicts(tmp_path: Path, rscript: str, r_environment) -> None:
    frame = pd.DataFrame({"Hane geliri": [12.0, 15, 9, 22, 18, 30, 11, 25],
                          "Hane büyüklüğü": [3.0, 4, 2, 5, 3, 6, 2, None],
                          "Kira": [4.0, 5, 3, 7, 6, 9, 4, 8]})
    spec, data = _own("konu02", frame, {"sayisal": "Hane geliri"}, ("Hane büyüklüğü", "Kira"))
    assert all(math.isfinite(check.expected) for step in spec.steps for check in step.checks)
    _every_step(spec)
    _scripts_agree(spec, data, tmp_path, rscript, r_environment)
    panel = _panel_frame([(unit, year, float(index + year), 1.0) for year in (1, 2) for index, unit in enumerate("ABC")])
    with pytest.raises(K.UploadError, match="Dönem sütunu sayısal değişkenlerden farklı"):
        _own("konu02", panel, {"sayisal": "Yıl", "donem": "Yıl", "birim": "Birim"})


def test_dot_decimals_in_a_semicolon_file_are_read_when_they_cannot_be_thousands(tmp_path: Path, rscript: str,
                                                                                  r_environment) -> None:
    text = "Gelir;Eğitim\n12.5;8\n10.25;12\n15.75;16\n9.5;8\n14;12\n11.125;11\n"
    table = K.read_upload("nokta.csv", text.encode())
    assert (table.separator, table.decimal) == (";", ",")
    case, _ = custom_case(VARIANTS["konu01"].custom, table, CustomChoices(roles={"sonuc": "Gelir",
                                                                                 "aciklayici": "Eğitim"}))
    assert case.data["gelir"].tolist() == [12.5, 10.25, 15.75, 9.5, 14.0, 11.125]
    spec = VARIANTS["konu01"].custom.build(case)
    _scripts_agree(spec, ("nokta.csv", text.encode()), tmp_path, rscript, r_environment)
    ambiguous = K.read_upload("binlik.csv", "Gelir;Eğitim\n1.250;8\n2.500;12\n3.125;16\n4.000;8\n5.500;12\n".encode())
    with pytest.raises(K.UploadError, match="nokta binlik ayırıcı olabilir"):
        custom_case(VARIANTS["konu01"].custom, ambiguous, CustomChoices(roles={"sonuc": "Gelir", "aciklayici": "Eğitim"}))


# --- Dosya okuma ve temizleme (İKT 217'den) -----------------------------------------------------------------

def _messy_frame(mark: str) -> pd.DataFrame:
    return pd.DataFrame({
        "Ödeme yöntemi": ["Kredi kartı", " Nakit", "Mobil ödeme", None, "Kredi kartı", "Nakit ", "Banka kartı", "T"],
        "Şube kodu": [1, 2, 1, 2, 1, np.nan, 2, 1],
        "Tutar (TL)": [f"12{mark}5", "30", 18, f"22{mark}5", "  ", f"40{mark}25", 15, 9],
    })


def _csv(frame: pd.DataFrame, separator: str, decimal: str, encoding: str) -> bytes:
    text = frame.to_csv(sep=separator, index=False, decimal=decimal)
    return ("﻿" + text).encode("utf-8") if encoding == "utf-8-sig" else text.encode(encoding)


def _messy_spec(table: K.UploadedTable):
    names = {"Ödeme yöntemi": "odeme_yontemi", "Şube kodu": "sube_kodu", "Tutar (TL)": "tutar_tl"}
    selections = [K.Selection(names["Ödeme yöntemi"], "Ödeme yöntemi", "kategorik"),
                  K.Selection(names["Şube kodu"], "Şube kodu", "kategorik"),
                  K.Selection(names["Tutar (TL)"], "Tutar (TL)", "sayisal")]
    prepared = K.prepare(table, selections)
    order = K.category_order(prepared.frame["odeme_yontemi"], "alfabetik")
    branches = K.category_order(prepared.frame["sube_kodu"], "alfabetik")
    step = LabStep(1, "Dosya", NoteRef("2.2"), "Deneme", operations=(
        prepared.read,
        FrequencyTable("veri", "odeme_yontemi", "tablo", order, relative=True, totals=True),
        CrossTab("veri", "sube_kodu", "odeme_yontemi", "capraz", branches, order, margins=True),
        Statistic("veri", "tutar_tl", "mean", "ortalama", "Ortalama tutar", decimals=3),
    ), checks=(*(Check(item, TableTarget("tablo", item, "frekans"), 0, 0) for item in order),
               Check("Ortalama", ScalarTarget("ortalama"), 0, 3)))
    return prepared, with_app_values(LabSpec("konu02", "Dosya", "2", (step,), source="kendi"))


def _frequency_spec(prepared: K.Prepared) -> LabSpec:
    name = prepared.read.columns[0][0]
    order = K.category_order(prepared.frame[name], "alfabetik")
    step = LabStep(1, "Dosya", NoteRef("2.2"), "Deneme", operations=(
        prepared.read, Shape("veri", "n", "k"),
        FrequencyTable("veri", name, "tablo", order, relative=False, totals=True),
    ), checks=(Check("n", ScalarTarget("n"), 0, 0), *(Check(item, TableTarget("tablo", item, "frekans"), 0, 0)
                                                      for item in order)))
    return with_app_values(LabSpec("konu02", "Dosya", "2", (step,), source="kendi"))


def _both(spec: LabSpec, folder: Path, data: tuple[str, bytes], rscript: str, r_environment) -> None:
    """Üretilen betik dosyayı okuyup uygulamanın bütün sayılarını iki dilde yeniden üretmeli."""

    result, path = _python(spec, folder / "python", data)
    _reproduce(spec, result, path, folder / "python", data)
    result, path = _run_script(spec, "R", folder / "r", [rscript], r_environment, data)
    _reproduce(spec, result, path, folder / "r", data)


@pytest.mark.parametrize("separator, decimal, encoding, mark", [
    (";", ",", "utf-8-sig", ","), (";", ",", "cp1254", ","), (",", ".", "utf-8-sig", "."),
    ("\t", ".", "utf-8-sig", ","),
], ids=["noktali-virgul-utf8", "noktali-virgul-cp1254", "virgul", "sekme-virgullu-sayilar"])
def test_csv_settings_are_detected_and_both_languages_read_the_same_values(separator, decimal, encoding, mark,
                                                                          tmp_path: Path, rscript: str,
                                                                          r_environment) -> None:
    data = _csv(_messy_frame(mark), separator, decimal, encoding)
    table = K.read_upload("verim.csv", data)
    assert (table.separator, table.decimal, table.encoding) == (separator, decimal, encoding)
    prepared, spec = _messy_spec(table)
    assert [kind for _, _, kind in prepared.read.columns] == ["metin", "kod", "sayi_metin"]
    assert prepared.read.dropped == 3 and len(prepared.frame) == 5
    assert prepared.frame["odeme_yontemi"].tolist() == ["Kredi kartı", "Nakit", "Mobil ödeme", "Banka kartı", "T"]
    assert prepared.frame["sube_kodu"].tolist() == ["1", "2", "1", "2", "1"]
    assert prepared.frame["tutar_tl"].tolist() == [12.5, 30.0, 18.0, 15.0, 9.0]
    _both(spec, tmp_path, ("verim.csv", data), rscript, r_environment)


def test_excel_cells_are_cleaned_like_the_generated_code(tmp_path: Path, rscript_readxl: str, r_environment) -> None:
    buffer = tmp_path / "kaynak.xlsx"
    _messy_frame(",").to_excel(buffer, index=False, sheet_name="Satışlar")
    data = buffer.read_bytes()
    buffer.unlink()
    table = K.read_upload("verim.xlsx", data)
    assert table.sheet == "Satışlar" and table.sheets == ("Satışlar",)
    prepared, spec = _messy_spec(table)
    assert [kind for _, _, kind in prepared.read.columns] == ["metin", "kod", "sayi_metin"]
    assert prepared.frame["tutar_tl"].tolist() == [12.5, 30.0, 18.0, 15.0, 9.0]
    _both(spec, tmp_path, ("verim.xlsx", data), rscript_readxl, r_environment)
    r_code = render_script(spec, "R")
    assert 'readxl::read_excel(veri_dosyasi, sheet = "Satışlar", na = c("", "NA")' in r_code
    assert 'install.packages("readxl")' in r_code and "Yalnız temel R" not in r_code


def test_text_cells_follow_one_rule_in_both_languages(tmp_path: Path, rscript: str, r_environment) -> None:
    """Bölünmez boşluk ve " NA " iki dilde aynı temizlenir."""

    rows = [("\xa0Kart", "A"), ("Kart\xa0", "B"), (" NA ", "A"), ("Nakit", "B"), ("Nakit", "A"), ("Kart", "B"),
            ("Mobil", "A"), ("NA", "B")]
    frame = pd.DataFrame(rows, columns=["Ödeme", "Grup"])
    data = _csv(frame, ";", ",", "utf-8-sig")
    table = K.read_upload("metin.csv", data)
    prepared = K.prepare(table, [K.Selection("odeme", "Ödeme", "kategorik", required=True),
                                 K.Selection("grup", "Grup", "kategorik")])
    assert prepared.frame["odeme"].tolist() == ["Kart", "Kart", "Nakit", "Nakit", "Kart", "Mobil"]
    assert prepared.read.dropped == 2
    _both(_frequency_spec(prepared), tmp_path, ("metin.csv", data), rscript, r_environment)


def test_header_names_are_cleaned_and_unusable_columns_are_left_out(tmp_path: Path, rscript: str,
                                                                   r_environment) -> None:
    text = (" Ödeme\xa0yöntemi ;Şube ;;NA;\"Not\"\"ı\"\n"
            "Nakit;Merkez;x;1;a\nKart;Sahil;y;2;b\nNakit;Sahil;z;3;c\nKart;Merkez;w;4;d\nMobil;Merkez;v;5;e\n")
    data = text.encode("utf-8")
    table = K.read_upload("baslik.csv", data)
    assert table.columns == ["Ödeme yöntemi", "Şube"] and table.strip_names
    assert any("Başlık hücresi boş" in note for note in table.notes)
    assert any("“NA”" in note for note in table.notes)
    prepared = K.prepare(table, [K.Selection("odeme_yontemi", "Ödeme yöntemi", "kategorik", required=True),
                                 K.Selection("sube", "Şube", "kategorik")])
    assert prepared.read.strip_names
    assert "ham.columns = [str(sutun).replace" in render_script(_frequency_spec(prepared), "Python")
    _both(_frequency_spec(prepared), tmp_path, ("baslik.csv", data), rscript, r_environment)


@pytest.mark.parametrize("text, message", [
    ("Ödeme,Ödeme ,Şube\nA,B,C\nA,B,C\n", "Aynı adı taşıyan"),
    ("Ödeme;Şube\n", "veri bulunamadı"),
    ("\nÖdeme;Şube\nA;B\n", None),
    ("   \nÖdeme;Şube\nA;B\n", "yalnız boşluk"),
    ("Ödeme;Şube\nA;B;C\nD;E;F\n", "daha az alan"),
    (";;\nA;B;C\n", "ilk satırı boş"),
    ('Ödeme;Not\nA;x"y\nB;z\n', "tırnak"),
])
def test_unusable_files_are_rejected(text: str, message: str | None) -> None:
    if message is None:
        assert K.read_upload("dosya.csv", text.encode()).columns == ["Ödeme", "Şube"]
        return
    with pytest.raises(K.UploadError, match=message):
        K.read_upload("dosya.csv", text.encode())


def test_size_rows_and_encoding_limits() -> None:
    with pytest.raises(K.UploadError, match="en çok 10.000 satır"):
        K.read_upload("uzun.csv", ("x\n" + "1\n" * (K.MAX_ROWS + 1)).encode())
    with pytest.raises(K.UploadError, match="MB'tan büyük"):
        K.read_upload("buyuk.csv", b"x" * (K.MAX_BYTES + 1))
    with pytest.raises(K.UploadError, match="UTF-16"):
        K.read_upload("utf16.csv", "Ödeme\nNakit\n".encode("utf-16"))
    with pytest.raises(K.UploadError, match="Yalnız Excel"):
        K.read_upload("veri.xls", b"x")


def test_encoding_is_detected_on_the_whole_file() -> None:
    prefix = "Şube;Not\nMerkez;"
    text = prefix + "a" * (65535 - len(prefix.encode())) + "ş\n" + "İzmir;b\n" * 3
    data = text.encode("utf-8")
    assert data[65535:65537] == "ş".encode()
    assert K.detect_csv(data)[0] == "utf-8-sig"
    assert K.read_upload("uzun.csv", data).frame["Şube"].iloc[-1] == "İzmir"


@pytest.mark.parametrize("text, message", [
    ("Tutar;Grup\n1.234;A\n12.000;B\n7;A\n", "noktalı sayılar"),
    ("Tutar;Grup\n1.234,5;A\n2;B\n3;A\n", "binlik ayırıcı"),
    ('Tutar,Grup\n"1,234",A\n"12,500",B\n7,A\n', "üç basamak"),
    ('Tutar,Grup\n"12,5",A\n13.5,B\n7,A\n', "hem noktalı"),
    ("Tutar;Grup\n12,5;A\nyok;B\n7;A\n", "sayı olmayan"),
    ("Tutar;Grup\n12,5;A\nInf;B\n7;A\n", "sonsuz"),
])
def test_numbers_that_could_be_misread_are_rejected(text: str, message: str) -> None:
    table = K.read_upload("sayilar.csv", text.encode())
    with pytest.raises(K.UploadError, match=message):
        K.column_kind(table, "Tutar", "sayisal")


def test_turkish_decimal_texts_are_read_as_numbers() -> None:
    table = K.read_upload("sayilar.csv", 'Tutar,Grup\n"12,5",A\n"7,25",B\n3,A\n'.encode())
    assert K.column_kind(table, "Tutar", "sayisal") == "sayi_metin"
    prepared = K.prepare(table, [K.Selection("tutar", "Tutar", "sayisal", required=True)])
    assert prepared.frame["tutar"].tolist() == [12.5, 7.25, 3.0]


@pytest.mark.parametrize("series, use, message", [
    (pd.Series(pd.to_datetime(["2024-01-01", "2024-02-01"])), "kategorik", "tarih"),
    (pd.Series([1.5, 2.0]), "kategorik", "ondalıklı sayılar içeriyor"),
    (pd.Series(["A", 12.5], dtype=object), "kategorik", "metinle birlikte ondalıklı"),
    (pd.Series(["12", "abc"], dtype=object), "sayisal", "sayı olmayan"),
    (pd.Series([1.0, np.inf]), "sayisal", "sonsuz"),
    (pd.Series([True, False]), "sayisal", "DOĞRU/YANLIŞ"),
    (pd.Series([True, False]), "serbest", "DOĞRU/YANLIŞ"),
])
def test_column_kinds_reject_unsuitable_columns(series, use, message) -> None:
    with pytest.raises(K.UploadError, match=message):
        K.series_kind(series, use, "Sütun")


def test_reserved_category_names_and_level_limits() -> None:
    table = K.read_upload("kategori.csv", "Kategori\nA\nToplam\nB\n".encode())
    with pytest.raises(K.UploadError, match="Toplam"):
        K.prepare(table, [K.Selection("kategori", "Kategori", "kategorik")])
    with pytest.raises(K.UploadError, match="en az 3"):
        K.check_levels(pd.Series(["a", "b", None]), "Sütun", 3, 5)
    with pytest.raises(K.UploadError, match="en çok 2"):
        K.check_levels(pd.Series(["a", "b", "c"]), "Sütun", 2, 2)


def test_mac_line_endings_and_single_column_files_are_read() -> None:
    mac = "Ödeme;Şube\rNakit;Merkez\rKart;Sahil\rNakit;Sahil\rKart;Merkez\rMobil;Merkez\r".encode()
    table = K.read_upload("mac.csv", mac)
    assert table.columns == ["Ödeme", "Şube"] and len(table.frame) == 5
    single = K.read_upload("tek.csv", "Puan\n12.5\n13\n14.25\n15\n16.5\n".encode())
    assert single.decimal == "." and K.column_kind(single, "Puan", "sayisal") == "sayi"
    assert K.read_upload("tek_virgul.csv", "Puan\n12,5\n13\n".encode()).decimal == ","


def test_integer_codes_are_the_same_text_in_both_languages(tmp_path: Path, rscript: str, r_environment) -> None:
    data = "Kod;Grup\n1;A\n-0;B\n2;A\n1;B\n2;A\n20230001001;B\n".encode()
    table = K.read_upload("kod.csv", data)
    prepared = K.prepare(table, [K.Selection("kod", "Kod", "kategorik", required=True),
                                 K.Selection("grup", "Grup", "kategorik")])
    assert sorted(set(prepared.frame["kod"])) == ["0", "1", "2", "20230001001"]
    _both(_frequency_spec(prepared), tmp_path, ("kod.csv", data), rscript, r_environment)
    huge = K.read_upload("buyuk.csv", "Kod;Grup\n9007199254740993;A\n1;B\n".encode())
    with pytest.raises(K.UploadError, match="çok büyük kodlar"):
        K.column_kind(huge, "Kod", "kategorik")


def test_code_names_are_ascii_unique_and_not_reserved() -> None:
    assert K.code_name("Ödeme yöntemi") == "odeme_yontemi"
    assert K.code_name("İl / İlçe (2024)") == "il_ilce_2024"
    assert K.code_name("2024 satış") == "degisken_2024_satis"
    assert K.code_name("if") == "if_" and K.code_name("TRUE") == "true"
    assert K.code_name("Şube", taken={"sube"}) == "sube_2"
    assert K.code_name("???") == "degisken"


def test_turkish_order_and_frequency_order() -> None:
    values = pd.Series(["Çay", "Su", "Ayran", "Şalgam", "Çay", "ılık", "İçecek", "10. sınıf", "2. sınıf", "Su", "Çay"])
    assert K.category_order(values, "alfabetik") == (
        "2. sınıf", "10. sınıf", "Ayran", "Çay", "ılık", "İçecek", "Su", "Şalgam")
    assert K.category_order(values, "frekans")[:2] == ("Çay", "Su")
    assert K.category_order(values, "dosya")[:3] == ("Çay", "Su", "Ayran")
