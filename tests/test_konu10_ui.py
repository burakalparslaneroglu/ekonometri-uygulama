"""Konu 10 ekranının temel AppTest kontrolü."""
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_konu10_renders_reference_invariance_and_shared_heading() -> None:
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/"app.py"); app.run(timeout=90)
    app.radio[0].set_value(app.radio[0].options[9]).run(timeout=120)
    assert not app.exception
    assert any("Kukla Değişkenler" in str(x.value) for x in app.header)
    assert any("KONU 10" in str(x.value) and "topic-badge" in str(x.value) for x in app.markdown)
    labels={str(metric.label) for metric in app.metric}
    assert {"Maks. fitted farkı","Maks. artık farkı","R² farkı","SSR farkı"}.issubset(labels)
    assert app.button(key="konu10_show_answer") and app.button(key="konu10_next_question")
    app.button(key="konu10_show_answer").click().run(timeout=120)
    assert app.session_state["konu10_answer_visible"] is True
    app.button(key="konu10_next_question").click().run(timeout=120)
    question_index=app.session_state["konu10_question_index"]
    assert question_index==1 and app.session_state["konu10_answer_visible"] is False
    app.selectbox(key="text_scale_label").set_value("Çok büyük — %130").run(timeout=120)
    assert app.session_state["konu10_question_index"]==question_index
