"""Konu 11 ekranının temel AppTest kontrolü."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


def test_konu11_renders_real_education_axis_and_dynamic_decisions() -> None:
    app=AppTest.from_file(ROOT/"app.py"); app.run(timeout=90)
    app.radio[0].set_value(app.radio[0].options[10]).run(timeout=120)
    assert not app.exception
    assert any("Etkileşim Terimleri" in str(x.value) for x in app.header)
    assert any("KONU 11" in str(x.value) and "topic-badge" in str(x.value) for x in app.markdown)
    captions=" ".join(str(x.value) for x in app.caption)
    assert "gerçek eğitim yılı" in captions
    labels={str(metric.label) for metric in app.metric}
    assert {"Maks. fitted farkı","Maks. artık farkı","R² farkı","SSR farkı"}.issubset(labels)
    source=(ROOT/"topics"/"konu11_etkilesimler_grup_farklari.py").read_text(encoding="utf-8")
    assert 'gap_grid["x"]+12.0' in source
    assert 'x_label="Eğitim yılı"' in source
    assert 'y_label="Koşullu log ücret farkı"' in source
    assert '"Kadın − erkek farkı"' in source
    assert '"Kadın − erkek tahmini (log puan)"' not in source
    assert "format_null_decision(h.joint_p_value,.05)" in source
    assert "%5'te ortak null reddedilemez" not in source
    assert app.button(key="konu11_show_answer") and app.button(key="konu11_next_question")
