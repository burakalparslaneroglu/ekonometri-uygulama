"""Konu 12 ekranının temel AppTest kontrolü."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


def test_konu12_renders_white_decision_and_shared_heading() -> None:
    app=AppTest.from_file(ROOT/"app.py"); app.run(timeout=90)
    app.radio[0].set_value(app.radio[0].options[11]).run(timeout=120)
    assert not app.exception
    assert any("Heteroskedastisite" in str(x.value) for x in app.header)
    assert any("KONU 12" in str(x.value) and "topic-badge" in str(x.value) for x in app.markdown)
    success_text=" ".join(str(x.value) for x in app.success)
    assert "White testi kararı (α=" in success_text
    source=(ROOT/"topics"/"konu12_heteroskedastisite.py").read_text(encoding="utf-8")
    assert "format_null_decision(white_details.p_value,alpha)" in source
    assert 'x_label="Tahmin edilen fiyat (bin ABD doları)"' in source
    assert "y_label=label" in source
    assert '"Ölçek–konum":("scale_location","√|studentize artık|")' in source
    assert app.button(key="konu12_show_answer") and app.button(key="konu12_next_question")
