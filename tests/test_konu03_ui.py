from pathlib import Path

from streamlit.testing.v1 import AppTest

from core.topic_registry import get_topic


def _visible_text(app: AppTest) -> str:
    """Öğrenciye gösterilen metin bileşenlerini tek bir metinde toplar."""
    collections = (app.title, app.header, app.subheader, app.markdown, app.caption, app.info, app.metric)
    return "\n".join(str(item.value) for collection in collections for item in collection).lower()


def test_konu03_ui_shows_fitted_values_and_residuals_without_konu04_content() -> None:
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.run(timeout=30)
    app.radio(key="selected_topic").set_value(get_topic("konu03").label).run(timeout=30)
    assert len(app.exception) == 0
    visible_text = _visible_text(app)
    assert "r-kare" not in visible_text
    assert "tkt" not in visible_text
    assert "mkt" not in visible_text
    assert "hkt" not in visible_text
    assert "bir gözlem için tahmin edilen değer ve artık" in visible_text
    assert "artık" in visible_text
    assert "belirli bir x değeri için tahmin" in visible_text
