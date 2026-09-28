from pathlib import Path

from streamlit.testing.v1 import AppTest
from topics.konu05_coklu_regresyon import _article_table, _fit, _dataset
from core.data_registry import konu05_model_specs
from core.topic_registry import get_topic


def test_konu05_loads_multiple_models_and_hidden_answer() -> None:
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.run(timeout=40)
    app.radio(key="selected_topic").set_value(get_topic("konu05").label).run(timeout=40)
    assert not app.exception
    app.selectbox(key="konu05_model").set_value("W1-M").run(timeout=40)
    text = "\n".join(str(item.value) for group in (app.header, app.subheader, app.markdown, app.caption) for item in group).casefold()
    for term in ("çoklu regresyon", "ceteris paribus", "profil karşılaştırması", "kısmi ilişki", "makale tipi tablo", "basit ve çoklu model karşılaştırması", "düzeltilmiş r-kare"):
        assert term in text
    assert len(app.latex) >= 3
    assert len(app.get("plotly_chart")) >= 1
    app.selectbox(key="konu05_model").set_value("H1-M").run(timeout=40)
    assert not app.exception
    assert "sqrft100 = sqrft / 100" in "\n".join(str(item.value) for item in app.caption)
    app.button(key="konu05_show_answer").click().run(timeout=40)
    assert app.success


def test_article_table_contains_real_intercepts_and_adjusted_r_squared() -> None:
    specs = konu05_model_specs()
    results = [_fit(spec, _dataset("wage1"))[0] for spec in specs[:3]]
    table = _article_table(results)
    assert list(table.columns) == ["Satır", "(1) Ücret", "(2) Ücret", "(3) ln(Ücret)"]
    assert table.loc[table["Satır"] == "Sabit", "(1) Ücret"].item() != "—"
    assert "Düzeltilmiş R-kare" in set(table["Satır"])
    assert table.loc[table["Satır"] == "Deneyim", "(1) Ücret"].item() == "—"
