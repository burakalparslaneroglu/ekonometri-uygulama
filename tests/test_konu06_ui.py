from pathlib import Path

from streamlit.testing.v1 import AppTest

from core.assumption_diagnostics_utils import coefficient_sensitivity, generate_near_collinearity_data, simulate_repeated_ols, wage1_ovb_decomposition
from core.data_registry import load_dataset
from topics.konu06_ols_varsayimlari_yanlilik import _distribution_figure, _sensitivity_table, _wage_table


def _topic06() -> AppTest:
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.run(timeout=60)
    app.radio(key="selected_topic").set_value(app.radio(key="selected_topic").options[5]).run(timeout=90)
    return app


def test_konu06_renders_laboratories_and_keeps_answer_hidden() -> None:
    app = _topic06()
    assert not app.exception
    text = "\n".join(str(item.value) for group in (app.header, app.subheader, app.markdown, app.caption) for item in group).casefold()
    for phrase in ("ekk varsayımları", "dört temel varsayım", "tekrarlı örnekleme laboratuvarı", "eksik değişken yanlılığı", "wage1", "yardımcı regresyon", "tam ve yüksek", "varyans şişirme faktörü", "kendini dene"):
        assert phrase in text
    assert len(app.get("plotly_chart")) >= 2
    assert not app.success
    app.button(key="konu06_show_answer").click().run(timeout=90)
    assert app.success
    app.button(key="konu06_next_question").click().run(timeout=90)
    assert not app.exception


def test_konu06_scenario_updates_and_no_inference_columns() -> None:
    app = _topic06()
    app.selectbox(key="konu06_sampling_scenario").set_value("bozulmus_sifir_kosullu_ortalama").run(timeout=90)
    assert not app.exception
    app.selectbox(key="konu06_near_scenario").set_value("very_high").run(timeout=90)
    assert not app.exception
    rendered_tables = "\n".join(str(item.value) for item in app.dataframe).casefold()
    for forbidden in ("std err", "p>|t|", "güven aralığı", "f testi"):
        assert forbidden not in rendered_tables
    assert "2.3" in "\n".join(str(item.value) for item in app.metric)


def test_konu06_presentation_helpers_are_student_friendly() -> None:
    wage = _wage_table(wage1_ovb_decomposition(load_dataset("wage1")))
    assert list(wage.columns) == ["Satır", "Yalnız eğitim", "Eğitim + deneyim", "Eğitim + deneyim + kıdem"]
    assert (wage == "—").any().any()
    assert not any(token in wage.to_string().casefold() for token in ("none", "nan", "null"))
    unbiased_names = [trace.name for trace in _distribution_figure(simulate_repeated_ols()).data if trace.name]
    assert unbiased_names == ["Gerçek eğim = hedef merkez: 1.5", "Ortalama tahmin: 1.500013"]
    biased_names = [trace.name for trace in _distribution_figure(simulate_repeated_ols(conditional_mean_loading=0.8)).data if trace.name]
    assert biased_names == ["Gerçek yapısal eğim: 1.5", "Tahmin dağılımının hedef merkezi: 2.3", "Ortalama tahmin: 2.300013"]
    sensitivity = _sensitivity_table(coefficient_sensitivity(generate_near_collinearity_data("near_exact").data))
    assert list(sensitivity.columns) == ["Büyüklük", "Özgün veri", "Küçük değişiklik sonrası", "Değişim"]
    assert not any("e-" in str(value) for value in sensitivity["Değişim"])
    assert "A4 altında doğru kalabilir" not in _topic06_text()


def _topic06_text() -> str:
    """Konu 06'nın görünür metnini birleştirir."""
    app = _topic06()
    return "\n".join(str(item.value) for group in (app.header, app.subheader, app.markdown, app.caption, app.text) for item in group)
