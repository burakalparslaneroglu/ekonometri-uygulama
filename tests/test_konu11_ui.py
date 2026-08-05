"""Konu 11 ekranının temel AppTest kontrolü."""
from pathlib import Path
from streamlit.testing.v1 import AppTest
def test_konu11_renders() -> None:
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/"app.py"); app.run(timeout=90)
    app.radio[0].set_value(app.radio[0].options[10]).run(timeout=120)
    assert not app.exception and any("Etkileşim" in str(x.value) for x in app.subheader)
    assert app.button(key="konu11_show_answer")
