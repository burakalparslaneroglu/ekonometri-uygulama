"""Konu 07 Streamlit görünümünün temel etkileşim denetimleri."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def _topic07() -> AppTest:
    """Uygulamayı Konu 07 seçili halde başlatır."""
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.run(timeout=60)
    app.radio[0].set_value(app.radio[0].options[6]).run(timeout=120)
    return app


def test_konu07_renders_and_reveals_question_answer() -> None:
    """Ana laboratuvarlar görünür; çözüm başlangıçta gizlidir."""
    app = _topic07()
    assert not app.exception
    text = "\n".join(str(item.value) for group in (app.header, app.subheader, app.markdown, app.caption) for item in group).casefold()
    for phrase in ("tek katsayı", "standart hata benzetimi", "hipotez kurma", "güven aralığı", "kapsama", "python çıktısı", "kendini dene"):
        assert phrase in text
    assert len(app.get("plotly_chart")) >= 4
    assert not any("çözüm" in str(item.value).casefold() for item in app.success)
    app.button(key="konu07_show_answer").click().run(timeout=120)
    assert any("çözüm" in str(item.value).casefold() for item in app.success)


def test_konu07_updates_model_and_test_controls() -> None:
    """Model, null ve test yönü değiştiğinde görünüm hatasız güncellenir."""
    app = _topic07()
    app.selectbox(key="konu07_model").set_value("H7-P").run(timeout=120)
    app.selectbox(key="konu07_coefficient").set_value("sqrft").run(timeout=120)
    app.number_input(key="konu07_null").set_value(0.1).run(timeout=120)
    app.selectbox(key="konu07_alternative").set_value("greater").run(timeout=120)
    assert not app.exception
    visible = "\n".join(str(item.value) for group in (app.markdown, app.caption, app.info) for item in group).casefold()
    assert "nonrobust" in visible
    assert "konu 08" in visible
