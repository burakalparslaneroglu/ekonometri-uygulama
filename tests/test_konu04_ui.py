from pathlib import Path
import unicodedata

from streamlit.testing.v1 import AppTest


FORBIDDEN_TEXT = (
    "p-değeri", "standart hata", "t istatistiği", "güven aralığı", "hipotez testi",
    "istatistiksel anlamlılık", "f testi", "heteroskedastisite", "dayanıklı standart hata",
)


def _visible_text(app: AppTest) -> str:
    collections = (app.title, app.header, app.subheader, app.markdown, app.caption, app.info, app.warning, app.success, app.metric)
    text = "\n".join(str(item.value) for collection in collections for item in collection)
    return "".join(character for character in unicodedata.normalize("NFD", text.casefold()) if not unicodedata.combining(character))


def test_konu04_ui_loads_blocks_forms_and_questions() -> None:
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.run(timeout=30)
    app.radio[0].set_value(app.radio[0].options[3]).run(timeout=30)
    assert len(app.exception) == 0
    visible = _visible_text(app)
    for term in ("tkt", "mkt", "hkt", "r-kare", "olcu birimi", "fonksiyonel bicim"):
        assert term in visible
    for forbidden in FORBIDDEN_TEXT:
        assert forbidden not in visible
    assert set(app.selectbox(key="konu04_form").options) == {"düzey-düzey", "log-düzey", "düzey-log", "log-log"}
    for key in ("konu04_y_scale_choice", "konu04_x_scale_choice"):
        app.selectbox(key=key).set_value("Özel çarpan").run(timeout=30)
        assert len(app.exception) == 0
    app.number_input(key="konu04_y_scale_custom").set_value(0.001).run(timeout=30)
    app.number_input(key="konu04_x_scale_custom").set_value(100.0).run(timeout=30)
    assert len(app.exception) == 0
    app.button(key="konu04_show_answer").click().run(timeout=30)
    assert "cozum:" in _visible_text(app)
    app.button(key="konu04_next_question").click().run(timeout=30)
    assert "soru 2:" in _visible_text(app)
