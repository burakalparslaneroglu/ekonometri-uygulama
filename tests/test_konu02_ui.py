from pathlib import Path
import unicodedata

from streamlit.testing.v1 import AppTest


FORBIDDEN_TEXT = (
    "r-kare", "p-değeri", "standart hata", "güven aralığı", "t istatistiği",
    "tahmin edilen değerler", "artıklar", "regresyon katsayısı", "panel regresyon",
    "sabit etkiler", "rassal etkiler", "farkların farkı",
)


def _visible_text(app: AppTest) -> str:
    collections = (app.title, app.header, app.subheader, app.markdown, app.caption, app.info, app.warning, app.success, app.metric)
    text = "\n".join(str(item.value) for collection in collections for item in collection)
    return "".join(character for character in unicodedata.normalize("NFD", text.casefold()) if not unicodedata.combining(character))


def test_konu02_app_loads_laboratory_jtrain_and_questions() -> None:
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.run(timeout=30)
    app.radio[0].set_value("Konu 02 — Ekonomik Veri Türleri, Nedensellik ve Ceteris Paribus").run(timeout=30)
    assert len(app.exception) == 0
    visible = _visible_text(app)
    assert "ekonomik veri turleri" in visible
    assert "wage1" in visible
    assert "jtrain2" in visible
    assert len(app.metric) >= 8
    assert "1.794 bin abd doları" in visible
    for forbidden in FORBIDDEN_TEXT:
        assert forbidden not in visible
    for dataset_title in ("PHILLIPS", "CPS78_85", "WAGEPAN"):
        app.selectbox(key="konu02_dataset").set_value(dataset_title.lower()).run(timeout=30)
        assert len(app.exception) == 0
    app.button(key="konu02_show_answer").click().run(timeout=30)
    assert "cozum:" in _visible_text(app)
    app.button(key="konu02_next_question").click().run(timeout=30)
    assert "soru 2:" in _visible_text(app)
    app.radio[0].set_value("Konu 03 — Basit Doğrusal Regresyon").run(timeout=30)
    assert len(app.exception) == 0
    app.radio[0].set_value("Konu 01 — Ekonometri ve Ampirik Araştırma").run(timeout=30)
    assert len(app.exception) == 0
