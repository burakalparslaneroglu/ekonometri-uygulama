"""Kayıtlı bütün uygulamalar ve sezgi deneyleri için ortak sözleşme ve iki dil testleri.

Yeni bir konu ``core/labs/registry.py``'ye (uygulama) veya ``core/labs/sezgi_konuNN.py``'ye (deneyler)
eklendiğinde bu testler onu kendiliğinden kapsar. R testleri Rscript ya da R ``wooldridge`` paketi yoksa
atlanır (``tests/conftest.py``).
"""

from __future__ import annotations

import contextlib
import importlib
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from core.codegen.base import LANGUAGES, generator, render_script, render_step
from core.labs.registry import LABS
from core.labs.runner import LabState, execute, run_lab
from core.labs.spec import MultiChoice, NumberChoice

SPECS = list(LABS.values())


def _experiments() -> list:
    found = []
    for number in range(0, 13):
        try:
            module = importlib.import_module(f"core.labs.sezgi_konu{number:02d}")
        except ModuleNotFoundError:
            continue
        found.extend(getattr(module, f"KONU{number:02d}_EXPERIMENTS"))
    return found


EXPERIMENTS = _experiments()
UTF8_OUTPUT = 'sys.stdout.reconfigure(encoding="utf-8")'
R_ENVIRONMENT = dict(os.environ, LANG="C.UTF-8", LC_ALL="C.UTF-8")


def _checks(spec) -> int:
    return sum(len(step.checks) for step in spec.steps)


def _chapter(spec) -> str:
    return str(int(spec.topic_key[-2:]))


# --- Tanım sözleşmesi -----------------------------------------------------------------

@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.topic_key)
def test_steps_are_numbered_and_tied_to_the_chapter(spec) -> None:
    assert spec.note_section == _chapter(spec)
    assert [step.number for step in spec.steps] == list(range(1, len(spec.steps) + 1))
    for step in spec.steps:
        assert step.note.section.startswith(spec.note_section + "."), step.number
        assert step.title and step.explanation
        # İşlemsiz adım yalnız son adım olabilir (notlardaki kavramsal bütünleştirici uygulama).
        assert step.operations or step is spec.steps[-1], step.number
    assert _checks(spec) >= 6


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.topic_key)
def test_every_step_renders_and_python_compiles(spec) -> None:
    for language in LANGUAGES:
        for step in spec.steps:
            assert bool(render_step(spec, step.number, language)) == bool(step.operations)
    compile(render_script(spec, "Python"), f"{spec.topic_key}.py", "exec")


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.topic_key)
def test_app_reproduces_every_number_in_the_notes(spec) -> None:
    run = run_lab(spec)
    failures = [f"{c.check.label}: {c.value}" for items in run.checks.values() for c in items if not c.passed]
    assert not failures, failures



def _single_changes(spec) -> list[dict[str, object]]:
    """Her denetimin notlardakinden farklı değerleri, birer birer: kaydırıcıda iki uç ve orta nokta, çoklu seçimde
    tek öğeli ve bir öğesi çıkarılmış kümeler."""

    changes = []
    for control in spec.controls:
        if isinstance(control, NumberChoice):
            span = control.maximum - control.minimum
            candidates = [control.minimum, control.maximum, control.minimum + span / 2]
        elif isinstance(control, MultiChoice):
            candidates = [(value,) for value, _ in control.options]
            candidates += [tuple(item for item in control.default if item != value) for value in control.default]
        else:
            candidates = [value for value, _ in control.options]
        default, seen = control.normalize(control.default), set()
        for candidate in candidates:
            try:
                value = control.normalize(candidate)
            except ValueError:
                continue
            if value != default and value not in seen:
                seen.add(value)
                changes.append({control.key: value})
    return changes


@pytest.mark.parametrize("spec", [spec for spec in SPECS if spec.controls], ids=lambda s: s.topic_key)
def test_a_choice_leaves_every_unmarked_step_at_the_notes(spec) -> None:
    """``LabSpec.resolve`` notlardan farklı saymadığı adımlar, seçim ne olursa olsun notlardaki sayıları verir."""

    changes = _single_changes(spec)
    assert changes
    for change in changes:
        resolved = spec.resolve(change)
        assert resolved.variant, change
        run = run_lab(resolved)
        failures = [f"Adım {number}: {c.check.label}" for number, items in run.checks.items() for c in items
                    if not c.passed]
        assert not failures, (change, resolved.variant, failures)


# --- Üretilen kod: Python -------------------------------------------------------------

@pytest.mark.parametrize(
    "spec", SPECS + [experiment.spec(experiment.defaults()) for experiment in EXPERIMENTS],
    ids=lambda s: f"{s.topic_key}-{s.kind}-{s.title[:20]}",
)
def test_python_scripts_write_utf8_before_any_output(spec) -> None:
    script = render_script(spec, "Python")
    assert "import sys" in script.splitlines()
    assert UTF8_OUTPUT in script
    first_print = script.find("print(")
    assert first_print == -1 or script.index(UTF8_OUTPUT) < first_print


def test_utf8_output_setup_survives_a_turkish_windows_code_page(tmp_path: Path) -> None:
    """Türkçe Windows'ta yönlendirilen çıktı cp1254'tür; θ, ∩, → bu kod sayfasında yoktur."""

    text = "θ = 360° × r; İktisat ∩ Dijital; x₁ → Şış Ğğ İı"
    setup = "\n".join(generator(SPECS[0], "Python").output_setup())
    environment = dict(os.environ, PYTHONIOENCODING="cp1254")
    for body, works in ((f"import sys\n{setup}\nprint({text!r})\n", True), (f"print({text!r})\n", False)):
        path = tmp_path / "cikti.py"
        path.write_text(body, encoding="utf-8")
        result = subprocess.run([sys.executable, str(path)], capture_output=True, encoding="utf-8",
                                errors="replace", timeout=60, env=environment)
        assert (result.returncode == 0) is works, result.stderr[-500:]
        if works:
            assert result.stdout.strip() == text


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.topic_key)
def test_generated_python_reproduces_the_notes(spec, tmp_path: Path) -> None:
    path = tmp_path / f"{spec.topic_key}.py"
    path.write_text(render_script(spec, "Python"), encoding="utf-8")
    # PYTHONIOENCODING=cp1254: Türkçe Windows'ta dosyaya/boruya yönlendirilen çıktının kod sayfası.
    result = subprocess.run([sys.executable, str(path)], cwd=tmp_path, capture_output=True, encoding="utf-8",
                            errors="replace", timeout=300,
                            env=dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="cp1254"))
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert result.stdout.count("  OK   ") == _checks(spec)
    assert "HATA" not in result.stdout
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]


@pytest.mark.parametrize("experiment", EXPERIMENTS, ids=lambda e: e.key)
def test_generated_python_reproduces_the_experiment_exactly(experiment, monkeypatch) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: plt.close("all"))
    parameters = experiment.defaults()
    state = LabState()
    for op in experiment.build(parameters):
        execute(op, state)
    namespace: dict = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(generator(experiment.spec(parameters), "Python").script(), namespace)
    for name, table in state.tables.items():
        np.testing.assert_allclose(namespace[name].to_numpy(dtype=float), table.to_numpy(dtype=float),
                                   rtol=0, atol=1e-12, err_msg=name)
    for name, value in state.scalars.items():
        if name in namespace:
            assert float(namespace[name]) == pytest.approx(value, abs=1e-12), name


def _balanced(latex: str) -> bool:
    """Kaçışsız süslü parantezler dengeli mi (``\\{`` ve ``\\}`` sayılmaz)."""

    depth, index = 0, 0
    while index < len(latex):
        if latex[index] == "\\" and latex[index + 1:index + 2] in ("{", "}"):
            index += 2
            continue
        depth += {"{": 1, "}": -1}.get(latex[index], 0)
        if depth < 0:
            return False
        index += 1
    return depth == 0


@pytest.mark.parametrize("experiment", EXPERIMENTS, ids=lambda e: e.key)
def test_dgp_lines_are_well_formed_latex_at_every_slider_end(experiment) -> None:
    """DGP satırları st.latex ile çizilir; dengesiz bir süslü parantez satırı ham metin olarak (kırmızı) gösterir."""

    settings = [experiment.defaults()]
    for parameter in experiment.parameters:
        for value in (parameter.minimum, parameter.maximum):
            settings.append(dict(experiment.defaults(), **{parameter.key: value}))
    for parameters in settings:
        for line in experiment.dgp(parameters):
            assert _balanced(line) and line.count("\\left") == line.count("\\right"), (parameters, line)


def test_every_experiment_renders_in_every_language_without_checks() -> None:
    for experiment in EXPERIMENTS:
        for language in LANGUAGES:
            code = generator(experiment.spec(experiment.defaults()), language).script()
            assert "sezgi deneyi" in code
            assert "kontrol_et" not in code


# --- Üretilen kod: R ----------------------------------------------------------------------

@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.topic_key)
def test_generated_r_reproduces_the_notes(spec, tmp_path: Path, rscript: str) -> None:
    path = tmp_path / f"{spec.topic_key}.R"
    path.write_text(render_script(spec, "R"), encoding="utf-8")
    result = subprocess.run([rscript, str(path)], cwd=tmp_path, capture_output=True, encoding="utf-8",
                            errors="replace", timeout=300, env=R_ENVIRONMENT)
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert result.stdout.count("  OK   ") == _checks(spec)
    assert "warning" not in (result.stdout + result.stderr).lower()
    # Rscript grafikleri çalışma klasörüne (Rplots.pdf) yazmamalı: OneDrive klasöründe artık dosya kalmaz.
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]


@pytest.mark.parametrize("experiment", EXPERIMENTS, ids=lambda e: e.key)
def test_generated_r_experiment_runs_cleanly(experiment, tmp_path: Path, rscript: str) -> None:
    path = tmp_path / "deney.R"
    path.write_text(generator(experiment.spec(experiment.defaults()), "R").script(), encoding="utf-8")
    result = subprocess.run([rscript, str(path)], cwd=tmp_path, capture_output=True, encoding="utf-8",
                            errors="replace", timeout=300, env=R_ENVIRONMENT)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "warning" not in (result.stdout + result.stderr).lower()
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]
