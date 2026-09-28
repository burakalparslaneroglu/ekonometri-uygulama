"""Konu 09 Streamlit görünümünün temel etkileşim denetimleri."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_konu09_renders_and_reveals_question_answer() -> None:
    """Fonksiyonel biçim laboratuvarları ve kontrollü soru çözümü çalışır."""
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.run(timeout=90)
    app.radio(key="selected_topic").set_value(app.radio(key="selected_topic").options[8]).run(timeout=120)
    assert not app.exception
    text = "\n".join(str(item.value) for group in (app.header, app.subheader, app.markdown, app.caption) for item in group).casefold()
    for phrase in ("ölçekleme", "standartlaştırılmış", "karesel", "merkezleme", "yanlış fonksiyonel biçim"):
        assert phrase in text
    assert not any("çözüm" in str(item.value).casefold() for item in app.success)
    app.button(key="konu09_show_answer").click().run(timeout=120)
    assert any("çözüm" in str(item.value).casefold() for item in app.success)


def test_text_scale_persists_across_topic_navigation() -> None:
    """Görünüm seçimi sidebar'da bulunur ve konu değişiminde korunur."""
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.run(timeout=90)
    scale = app.selectbox(key="text_scale_label")
    assert len(scale.options) == 4 and scale.value == "Büyük — %110"
    scale.set_value("Çok büyük — %130").run(timeout=90)
    app.radio(key="selected_topic").set_value(app.radio(key="selected_topic").options[8]).run(timeout=120)
    assert app.selectbox(key="text_scale_label").value == "Çok büyük — %130"
    assert not app.exception
