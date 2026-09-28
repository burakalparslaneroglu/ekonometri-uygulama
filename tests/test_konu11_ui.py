"""Konu 11 ekranının davranışsal AppTest denetimleri."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
ZERO_TEXT = "sayısal tolerans içinde 0"


def _topic11() -> AppTest:
    app = AppTest.from_file(ROOT / "app.py")
    app.run(timeout=90)
    app.radio(key="selected_topic").set_value(app.radio(key="selected_topic").options[10]).run(timeout=120)
    return app


def _metric_value(app: AppTest, label: str) -> str:
    return str(next(metric.value for metric in app.metric if str(metric.label) == label))


def _visible_text(app: AppTest) -> str:
    groups = (app.markdown, app.caption, app.info, app.warning, app.success, app.text)
    return "\n".join(str(item.value) for group in groups for item in group)


def test_konu11_renders_heading_real_education_axis_and_questions() -> None:
    app = _topic11()
    assert not app.exception
    assert any("Etkileşim Terimleri" in str(item.value) for item in app.header)
    assert any(
        "KONU 11" in str(item.value) and "topic-badge" in str(item.value)
        for item in app.markdown
    )
    assert "gerçek eğitim yılı" in " ".join(str(item.value) for item in app.caption)
    assert app.button(key="konu11_show_answer")
    assert app.button(key="konu11_next_question")

    app.button(key="konu11_show_answer").click().run(timeout=120)
    assert app.session_state["konu11_answer_visible"] is True
    app.button(key="konu11_next_question").click().run(timeout=120)
    assert app.session_state["konu11_question_index"] == 1
    assert app.session_state["konu11_answer_visible"] is False


def test_konu11_structure_conditional_gap_and_centering_controls_update() -> None:
    app = _topic11()
    app.selectbox(key="konu11_structure").set_value("Aynı sabit – farklı eğim").run(timeout=120)
    assert "γ₀=0, γ₁≠0" in _visible_text(app)

    before = str(app.dataframe[1].value)
    app.slider(key="konu11_educ").set_value(20.0).run(timeout=120)
    after = str(app.dataframe[1].value)
    assert before != after
    assert app.session_state["konu11_educ"] == 20.0

    app.slider(key="konu11_center").set_value(16.0).run(timeout=120)
    assert any(str(metric.label) == "female katsayısı (educ=16)" for metric in app.metric)
    for label in ("Maks. fitted farkı", "Maks. artık farkı", "R² farkı", "SSR farkı"):
        assert _metric_value(app, label) == ZERO_TEXT


def test_konu11_hprice_conditional_difference_and_dynamic_decisions_update() -> None:
    app = _topic11()
    app.select_slider(key="konu11_lot").set_value(50000).run(timeout=120)
    visible = _visible_text(app)
    assert "50,000 kare fitte colonial − diğer farkı" in visible
    assert "α=0.05 (%5): H₀ reddedilemez" in visible
    assert "α=0.10 (%10): H₀ reddedilir" in visible
