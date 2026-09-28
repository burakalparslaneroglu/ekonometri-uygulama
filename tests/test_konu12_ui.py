"""Konu 12 ekranının davranışsal AppTest denetimleri."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def _topic12() -> AppTest:
    app = AppTest.from_file(ROOT / "app.py")
    app.run(timeout=90)
    app.radio(key="selected_topic").set_value(app.radio(key="selected_topic").options[11]).run(timeout=120)
    return app


def _success_text(app: AppTest) -> str:
    return "\n".join(str(item.value) for item in app.success)


def test_konu12_renders_heading_decisions_and_questions() -> None:
    app = _topic12()
    assert not app.exception
    assert any("Heteroskedastisite" in str(item.value) for item in app.header)
    assert any(
        "KONU 12" in str(item.value) and "topic-badge" in str(item.value)
        for item in app.markdown
    )
    success = _success_text(app)
    assert "Breusch–Pagan kararı (α=0.05" in success
    assert "White testi kararı (α=0.05" in success
    assert app.button(key="konu12_show_answer")
    assert app.button(key="konu12_next_question")

    app.button(key="konu12_show_answer").click().run(timeout=120)
    assert app.session_state["konu12_answer_visible"] is True
    app.button(key="konu12_next_question").click().run(timeout=120)
    assert app.session_state["konu12_question_index"] == 1
    assert app.session_state["konu12_answer_visible"] is False


def test_konu12_alpha_covariance_and_diagnostic_controls_are_live() -> None:
    app = _topic12()
    app.select_slider(key="konu12_alpha").set_value(.01).run(timeout=120)
    success = _success_text(app)
    assert success.count("α=0.01") >= 2
    assert success.count("H₀ reddedilir") >= 2

    app.selectbox(key="konu12_covariance").set_value("HC3").run(timeout=120)
    assert any(str(metric.label) == "HC3 Wald/F" for metric in app.metric)
    assert "HC3" in " ".join(str(item.value) for item in app.caption)

    for diagnostic in ("Artık–tahmin", "Mutlak artık–tahmin", "Ölçek–konum"):
        app.selectbox(key="konu12_diagnostic").set_value(diagnostic).run(timeout=120)
        assert not app.exception
        assert app.session_state["konu12_diagnostic"] == diagnostic


def test_konu12_presentation_and_method_limits_are_explicit() -> None:
    app = _topic12()
    diagnostic_table = str(app.dataframe[2].value)
    assert "< 0.001" in diagnostic_table
    assert "0.0000" not in diagnostic_table

    app.selectbox(key="konu12_limit").set_value("içsellik").run(timeout=120)
    assert any("HC0–HC3 bu sorunu çözmez" in str(item.value) for item in app.error)

    app.selectbox(key="konu12_limit").set_value("heteroskedastisite").run(timeout=120)
    assert any("HC0–HC3 türü kovaryans çözümü uygun olabilir" in str(item.value) for item in app.success)
