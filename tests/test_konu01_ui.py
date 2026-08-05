from pathlib import Path
import unicodedata

from streamlit.testing.v1 import AppTest


FORBIDDEN_TEXT = (
    "r-kare",
    "p-değeri",
    "standart hata",
    "güven aralığı",
    "tahmin edilen değerler",
    "artıklar",
    "en küçük kareler tahmini",
)


def _visible_text(app: AppTest) -> str:
    """Öğrenci arayüzündeki metin bileşenlerini birleştirir."""
    collections = (app.title, app.header, app.subheader, app.markdown, app.caption, app.info, app.metric)
    text = "\n".join(str(item.value) for collection in collections for item in collection)
    return "".join(character for character in unicodedata.normalize("NFD", text.casefold()) if not unicodedata.combining(character))


def test_konu01_ui_respects_course_scope_and_branding() -> None:
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.run(timeout=30)
    assert len(app.exception) == 0
    visible_text = _visible_text(app)
    assert "ikt 305 ekonometri i" in visible_text
    assert "izmir bakırcay universitesi" in visible_text
    assert "wage1 veri gezgini" in visible_text
    for forbidden in FORBIDDEN_TEXT:
        assert forbidden not in visible_text
