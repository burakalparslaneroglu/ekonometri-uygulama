"""Konu 08 Streamlit görünümünün temel etkileşim denetimleri."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_konu08_renders_and_reveals_question_answer() -> None:
    """Ortak test laboratuvarı ve kontrollü çözüm görünürlüğü çalışır."""
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.run(timeout=90)
    app.radio[0].set_value(app.radio[0].options[7]).run(timeout=120)
    assert not app.exception
    text = "\n".join(str(item.value) for group in (app.header, app.subheader, app.markdown, app.caption) for item in group).casefold()
    for phrase in ("birden fazla kısıt", "wage1", "genel anlamlılık", "büyük örneklem", "kendini dene"):
        assert phrase in text
    assert not any("çözüm" in str(item.value).casefold() for item in app.success)
    app.button(key="konu08_show_answer").click().run(timeout=120)
    assert any("çözüm" in str(item.value).casefold() for item in app.success)


def test_konu08_custom_restriction_lab_updates_safely() -> None:
    """Preset, eşitlik ve bağımlı satır durumları arayüzde hatasız görünür."""
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.run(timeout=90)
    app.radio[0].set_value(app.radio[0].options[7]).run(timeout=120)
    app.selectbox(key="konu08_custom_preset_WAGE1").set_value("exper = tenure").run(timeout=120)
    assert not app.exception
    visible = "\n".join(str(item.value) for group in (app.subheader, app.markdown, app.caption, app.info) for item in group).casefold()
    assert "kullanıcı tanımlı" in visible and "matrix f" in visible
    app.selectbox(key="konu08_custom_preset_WAGE1").set_value("yinelenen/geçersiz kısıt örneği").run(timeout=120)
    assert not app.exception
    assert any("geçersiz" in str(item.value).casefold() for item in app.dataframe)
