"""Uygulama kabuğu ve yeni mimariye taşınan konuların uçtan uca duman testleri."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from core.topic_registry import get_topic

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _run_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=60).run()


def _select(app: AppTest, topic: str) -> None:
    app.radio(key="selected_topic").set_value(get_topic(topic).label).run()


def _markdown(app: AppTest) -> str:
    return "\n".join(item.value for item in app.markdown)


def _metrics(app: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in app.metric}


def test_app_opens_on_topic_01_with_three_tabs() -> None:
    app = _run_app()
    assert not app.exception
    # app.py'nin açıklamaları Streamlit "magic" ile sayfaya yazılmamalı.
    shown = [item.value for item in app.markdown if not item.value.startswith("<style>")]
    assert not any("Kenar çubuğundaki konu adı" in text or "Çizilmediği çalıştırmalarda" in text for text in shown)
    assert app.radio(key="selected_topic").value == get_topic("konu01").label
    assert "Ekonometri ve Ampirik Araştırmanın Mantığı" in _markdown(app)
    assert [tab.label for tab in app.tabs][:3] == ["Uygulama", "Sezgi", "Kendini sına"]
    assert app.segmented_control(key="konu01_lab_step").value == 1
    assert {"Gözlem sayısı n": "526", "Değişken sayısı": "24"}.items() <= _metrics(app).items()


def test_every_lab_step_renders_in_both_languages() -> None:
    app = _run_app()
    for topic, steps in (("konu01", 6), ("konu02", 6)):
        _select(app, topic)
        for language in ("Python", "R"):
            app.segmented_control(key="code_language").set_value(language).run()
            for number in range(1, steps + 1):
                app.segmented_control(key=f"{topic}_lab_step").set_value(number).run()
                assert not app.exception, (topic, language, number)
                assert any(item.value.startswith(f"Adım {number}:") for item in app.subheader)


def test_lab_step_shows_the_notes_numbers_in_turkish_format() -> None:
    app = _run_app()
    app.segmented_control(key="konu01_lab_step").set_value(5).run()
    metrics = _metrics(app)
    assert metrics["Sabit terim β̂₀"] == "−0,9049"
    assert metrics["Eğim β̂₁ (educ)"] == "0,5414"
    assert metrics["R², yüzde olarak"] == "%16,5"


def test_a_choice_changes_the_model_carries_to_the_next_step_and_resets() -> None:
    app = _run_app()
    app.segmented_control(key="konu01_lab_step").set_value(4).run()
    app.segmented_control(key="konu01_secim_adim4_x").set_value("exper").run()
    assert not app.exception
    assert any("notlardan farklı" in item.value for item in app.info)
    app.segmented_control(key="konu01_lab_step").set_value(5).run()
    slope = _metrics(app)["Eğim β̂₁ (exper)"]
    assert slope != "0,5414" and slope.startswith("0,0")
    assert any("Adım 4" in item.value for item in app.info)
    app.button(key="konu01_notlara_don_5").click().run()
    assert _metrics(app)["Eğim β̂₁ (educ)"] == "0,5414"


def test_choices_survive_a_topic_switch() -> None:
    app = _run_app()
    app.segmented_control(key="konu01_lab_step").set_value(3).run()
    app.segmented_control(key="konu01_secim_adim3_x").set_value("tenure").run()
    _select(app, "konu02")
    app.segmented_control(key="konu02_lab_step").set_value(4).run()
    assert not app.exception
    _select(app, "konu01")
    assert app.segmented_control(key="konu01_lab_step").value == 3
    assert app.segmented_control(key="konu01_secim_adim3_x").value == "tenure"


def test_panel_step_writes_years_without_a_thousands_separator() -> None:
    app = _run_app()
    _select(app, "konu02")
    app.segmented_control(key="konu02_lab_step").set_value(4).run()
    tables = [frame.value for frame in app.dataframe]
    panel = next(table for table in tables if "Büyüklük" in table.columns)
    values = dict(zip(panel["Büyüklük"], panel["Değer"]))
    assert values["İlk dönem"] == "1980" and values["Son dönem"] == "1987"
    assert values["Gözlem sayısı (satır)"] == "4.360"


def test_every_experiment_runs_and_reacts_to_its_sliders() -> None:
    app = _run_app()
    app.slider(key="konu01_sezgi1_olcek").set_value(0.0).run()
    assert _metrics(app)["R²"] == "1,000"
    assert _metrics(app)["Tahmin β̂₁"] == _metrics(app)["Gerçek eğim β₁"] == "13,50"
    for topic in ("konu01", "konu02"):
        _select(app, topic)
        for number in (1, 2, 3):
            app.segmented_control(key=f"{topic}_sezgi_deney").set_value(number).run()
            assert not app.exception, (topic, number)


def test_quiz_checks_an_answer_and_lists_sections_to_review() -> None:
    app = _run_app()
    app.radio(key="konu01_quiz_k01").set_value(0).run()
    app.button(key="konu01_quiz_check_k01").click().run()
    assert not app.exception
    assert any("Tekrar edilecek bölümler" in item.value and "§1.1" in item.value for item in app.markdown)


def test_topic_switch_keeps_text_scale_and_code_language() -> None:
    app = _run_app()
    app.selectbox(key="text_scale_label").set_value("Daha büyük — %120").run()
    app.segmented_control(key="code_language").set_value("R").run()
    _select(app, "konu03")
    _select(app, "konu02")
    assert not app.exception
    assert app.session_state["text_scale"] == 1.2
    assert app.session_state["code_language"] == "R"
    assert "Ekonomik Veri Türleri, Nedensellik ve Ceteris Paribus" in _markdown(app)
