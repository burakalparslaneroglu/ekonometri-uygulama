"""Konu 10 ekranının davranışsal AppTest denetimleri."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
ZERO_TEXT = "sayısal tolerans içinde 0"


def _topic10() -> AppTest:
    app = AppTest.from_file(ROOT / "app.py")
    app.run(timeout=90)
    app.radio[0].set_value(app.radio[0].options[9]).run(timeout=120)
    return app


def _metric_value(app: AppTest, label: str) -> str:
    return str(next(metric.value for metric in app.metric if str(metric.label) == label))


def test_konu10_renders_shared_heading_and_question_state() -> None:
    app = _topic10()
    assert not app.exception
    assert any("Kukla Değişkenler" in str(item.value) for item in app.header)
    assert any(
        "KONU 10" in str(item.value) and "topic-badge" in str(item.value)
        for item in app.markdown
    )
    assert app.button(key="konu10_show_answer")
    assert app.button(key="konu10_next_question")

    app.button(key="konu10_show_answer").click().run(timeout=120)
    assert app.session_state["konu10_answer_visible"] is True
    app.button(key="konu10_next_question").click().run(timeout=120)
    question_index = app.session_state["konu10_question_index"]
    assert question_index == 1
    assert app.session_state["konu10_answer_visible"] is False

    app.selectbox(key="text_scale_label").set_value("Çok büyük — %130").run(timeout=120)
    assert app.session_state["konu10_question_index"] == question_index


def test_konu10_coding_direction_and_reference_invariance_are_live() -> None:
    app = _topic10()
    app.radio(key="konu10_direction").set_value("D=1 erkek, D=0 kadın").run(timeout=120)
    assert not app.exception
    assert _metric_value(app, "Referans") == "Kadın"
    assert float(_metric_value(app, "δ̂")) > 0

    app.selectbox(key="konu10_region_reference").set_value("Güney").run(timeout=120)
    assert not app.exception
    captions = " ".join(str(item.value) for item in app.caption)
    assert "Güney ve Batı kodlamaları" in captions
    for label in ("Maks. fitted farkı", "Maks. artık farkı", "R² farkı", "SSR farkı"):
        assert _metric_value(app, label) == ZERO_TEXT


def test_konu10_rank_scenarios_change_identification_message() -> None:
    app = _topic10()
    rank_control = app.radio(key="konu10_rank_scenario")

    rank_control.set_value("Sabit + bütün kuklalar").run(timeout=120)
    assert _metric_value(app, "Tam rank") == "Hayır"
    assert _metric_value(app, "Tahmin") == "Sunulmaz"

    app.radio(key="konu10_rank_scenario").set_value("Sabit + m−1 kukla").run(timeout=120)
    assert _metric_value(app, "Tam rank") == "Evet"
    assert _metric_value(app, "Tahmin") == "Benzersiz"
