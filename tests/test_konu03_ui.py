from pathlib import Path

from streamlit.testing.v1 import AppTest


def _visible_text(app: AppTest) -> str:
    """Öğrenciye gösterilen metin bileşenlerini tek bir metinde toplar."""
    collections = (app.title, app.header, app.subheader, app.markdown, app.caption, app.info, app.metric)
    return "\n".join(str(item.value) for collection in collections for item in collection).lower()


def test_konu03_ui_does_not_show_konu04_content() -> None:
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.run(timeout=30)
    app.radio[0].set_value("Konu 03 — Basit Doğrusal Regresyon").run(timeout=30)
    assert len(app.exception) == 0
    visible_text = _visible_text(app)
    assert "r-kare" not in visible_text
    assert "artık" not in visible_text
    assert "tahmin edilen değerler ve artıklar" not in visible_text
    assert "belirli bir x değeri için tahmin" in visible_text
