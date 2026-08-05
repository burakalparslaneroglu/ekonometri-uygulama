"""Konu 07 Streamlit görünümünün temel etkileşim denetimleri."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from core.data_registry import load_dataset
from core.regression_inference_utils import coefficient_test, fit_ols_inference
from topics.konu07_tekli_hipotez_testleri import _t_figure


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


def test_konu07_video_review_sections_and_large_t_annotation() -> None:
    """Video incelemesindeki görünür öğretim blokları ve büyük-t işareti denetlenir."""
    app = _topic07()
    text = "\n".join(str(item.value) for group in (app.header, app.subheader, app.markdown, app.caption, app.warning) for item in group).casefold()
    for phrase in ("iki tarafl", "seçilmi", "yanlış", "sonucu bütünle", "kritik", "standart hata türü"):
        assert phrase in text
    assert app.selectbox(key="konu07_hprice_change").value == 100
    app.selectbox(key="konu07_hprice_change").set_value(500).run(timeout=120)
    assert not app.exception
    result = fit_ols_inference(load_dataset("wage1"), "wage", ("educ", "exper", "tenure"))
    figure = _t_figure(coefficient_test(result, "educ"))
    assert max(abs(value) for value in figure.data[0].x) <= 5.0
    assert any("çizim alanının dışında" in annotation.text for annotation in figure.layout.annotations)
