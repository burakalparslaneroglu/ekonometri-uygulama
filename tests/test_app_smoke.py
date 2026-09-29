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


def test_app_opens_on_topic_00_with_three_tabs() -> None:
    app = _run_app()
    assert not app.exception
    # app.py'nin açıklamaları Streamlit "magic" ile sayfaya yazılmamalı.
    shown = [item.value for item in app.markdown if not item.value.startswith("<style>")]
    assert not any("Kenar çubuğundaki konu adı" in text or "Çizilmediği çalıştırmalarda" in text for text in shown)
    assert app.radio(key="selected_topic").value == get_topic("konu00").label
    assert "Başlangıç Araç Kutusu: Veri, Notasyon ve Temel İstatistik" in _markdown(app)
    assert [tab.label for tab in app.tabs][:3] == ["Uygulama", "Sezgi", "Kendini sına"]
    assert app.segmented_control(key="konu00_lab_step").value == 1
    assert {"Gözlem sayısı n": "5", "Değişken sayısı": "4"}.items() <= _metrics(app).items()
    _select(app, "konu01")
    assert "Ekonometri ve Ampirik Araştırmanın Mantığı" in _markdown(app)
    assert {"Gözlem sayısı n": "526", "Değişken sayısı": "24"}.items() <= _metrics(app).items()


def test_every_lab_step_renders_in_both_languages() -> None:
    app = _run_app()
    for topic, steps in (("konu00", 9), ("konu01", 6), ("konu02", 6), ("konu03", 8), ("konu04", 10), ("konu05", 10),
                         ("konu06", 6), ("konu07", 10), ("konu08", 7), ("konu09", 8), ("konu10", 11)):
        _select(app, topic)
        for language in ("Python", "R"):
            app.segmented_control(key="code_language").set_value(language).run()
            for number in range(1, steps + 1):
                app.segmented_control(key=f"{topic}_lab_step").set_value(number).run()
                assert not app.exception, (topic, language, number)
                assert any(item.value.startswith(f"Adım {number}:") for item in app.subheader)


def test_step_buttons_move_one_step_and_are_disabled_at_the_ends() -> None:
    app = _run_app()
    _select(app, "konu04")
    assert app.button(key="konu04_lab_prev").disabled and not app.button(key="konu04_lab_next").disabled
    app.button(key="konu04_lab_next").click().run()
    assert app.segmented_control(key="konu04_lab_step").value == 2
    app.segmented_control(key="konu04_lab_step").set_value(10).run()
    assert app.button(key="konu04_lab_next").disabled and not app.button(key="konu04_lab_prev").disabled
    app.button(key="konu04_lab_prev").click().run()
    assert app.segmented_control(key="konu04_lab_step").value == 9
    assert not app.exception


def test_lab_step_shows_the_notes_numbers_in_turkish_format() -> None:
    app = _run_app()
    _select(app, "konu01")
    app.segmented_control(key="konu01_lab_step").set_value(5).run()
    metrics = _metrics(app)
    assert metrics["Sabit terim β̂₀"] == "−0,9049"
    assert metrics["Eğim β̂₁ (educ)"] == "0,5414"
    assert metrics["R², yüzde olarak"] == "%16,5"


def test_a_choice_changes_the_model_carries_to_the_next_step_and_resets() -> None:
    app = _run_app()
    _select(app, "konu01")
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
    _select(app, "konu01")
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
    _select(app, "konu01")
    olcek = app.select_slider(key="konu01_sezgi1_olcek")
    assert olcek.options[:2] == ["0,00", "0,25"] and olcek.value == 1.0  # ondalık virgül, değer sayı olarak kalır
    olcek.set_value(0.0).run()
    assert _metrics(app)["R²"] == "1,000"
    assert _metrics(app)["Tahmin β̂₁"] == _metrics(app)["Gerçek eğim β₁"] == "13,50"
    for topic in ("konu00", "konu01", "konu02", "konu03", "konu04", "konu05", "konu06", "konu07", "konu08", "konu09",
                  "konu10"):
        _select(app, topic)
        for number in (1, 2, 3):
            app.segmented_control(key=f"{topic}_sezgi_deney").set_value(number).run()
            assert not app.exception, (topic, number)


def test_quiz_checks_an_answer_and_lists_sections_to_review() -> None:
    app = _run_app()
    _select(app, "konu01")
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


def test_konu00_percent_step_reacts_to_its_sliders_and_notes_default_returns() -> None:
    app = _run_app()
    app.segmented_control(key="konu00_lab_step").set_value(6).run()
    metrics = _metrics(app)
    assert metrics["Yüzde değişim: 100 → 120"] == "%20,00"
    assert metrics["Geri dönüş: 120 → 100"] == "%−16,67"
    app.slider(key="konu00_secim_adim6_x1").set_value(105).run()
    assert not app.exception
    metrics = _metrics(app)
    assert metrics["Yüzde değişim: 100 → 105"] == "%5,00"
    assert metrics["100·[ln(x₁) − ln(x₀)]: 100 → 105"] == "4,88"
    assert any("notlardan farklı" in item.value for item in app.info)


def test_konu00_conditional_mean_step_reacts_to_the_condition() -> None:
    app = _run_app()
    app.segmented_control(key="konu00_lab_step").set_value(7).run()
    assert _metrics(app)["Koşullu ortalama: eğitim = 16"] == "8,04"
    app.selectbox(key="konu00_secim_adim7_egitim").set_value("12").run()
    assert not app.exception
    assert _metrics(app)["Koşullu ortalama: eğitim = 12"] == "5,37"
    assert _metrics(app)["Çalışan sayısı: eğitim = 12"] == "198"


def test_konu00_notice_names_only_the_step_the_result_depends_on() -> None:
    app = _run_app()
    app.segmented_control(key="konu00_secim_adim1_gosterge").set_value("erkek").run()
    app.segmented_control(key="konu00_lab_step").set_value(2).run()
    app.segmented_control(key="konu00_secim_adim2_degisken").set_value("ucret").run()
    app.segmented_control(key="konu00_lab_step").set_value(3).run()
    assert not app.exception
    notices = [item.value for item in app.info if "önceki bir adımdaki" in item.value]
    assert notices and "(Adım 2)" in notices[0], notices


def test_konu00_conditional_experiment_shows_its_level_table() -> None:
    app = _run_app()
    app.segmented_control(key="konu00_sezgi_deney").set_value(3).run()
    assert not app.exception
    columns = [set(map(str, frame.value.columns)) for frame in app.dataframe]
    assert any({"Gözlem sayısı", "Gerçek E(Y | X)"} <= found for found in columns), columns


def test_konu03_regressor_choice_carries_to_later_steps_and_resets() -> None:
    app = _run_app()
    _select(app, "konu03")
    assert "Basit Doğrusal Regresyon Modeli" in _markdown(app)
    app.segmented_control(key="konu03_lab_step").set_value(5).run()
    metrics = _metrics(app)
    assert metrics["Sabit terim β̂₀"] == "−0,9049" and metrics["Eğim β̂₁ (educ)"] == "0,5414"
    app.segmented_control(key="konu03_lab_step").set_value(4).run()
    app.segmented_control(key="konu03_secim_adim4_x").set_value("exper").run()
    assert not app.exception
    assert any("notlardan farklı" in item.value for item in app.info)
    app.segmented_control(key="konu03_lab_step").set_value(7).run()
    assert not app.exception
    assert any("(Adım 4)" in item.value for item in app.info)
    app.button(key="konu03_notlara_don_7").click().run()
    app.segmented_control(key="konu03_lab_step").set_value(5).run()
    assert _metrics(app)["Eğim β̂₁ (educ)"] == "0,5414"


def test_konu03_small_ols_step_reacts_to_the_fifth_score() -> None:
    app = _run_app()
    _select(app, "konu03")
    app.segmented_control(key="konu03_lab_step").set_value(3).run()
    assert _metrics(app)["Eğim β̂₁ = pay / payda (Denklem 3.8)"] == "2,9"
    app.slider(key="konu03_secim_adim3_y5").set_value(88).run()
    assert not app.exception
    assert _metrics(app)["Eğim β̂₁ = pay / payda (Denklem 3.8)"] == "3,9"


def test_konu04_four_forms_and_article_table_columns() -> None:
    app = _run_app()
    _select(app, "konu04")
    app.segmented_control(key="konu04_lab_step").set_value(7).run()
    metrics = _metrics(app)
    assert metrics["Düzey–düzey: 100 kare fit → bin dolar (100 β̂₁)"] == "14,02"
    assert metrics["Log–log eğim"] == "0,8727"
    app.segmented_control(key="konu04_lab_step").set_value(9).run()
    app.multiselect(key="konu04_secim_adim9_sutunlar").set_value(["duzey_duzey", "duzey_log"]).run()
    assert not app.exception
    assert any("notlardan farklı" in item.value for item in app.info)


def test_konu03_and_konu04_quizzes_grade_answers() -> None:
    app = _run_app()
    _select(app, "konu04")
    app.radio(key="konu04_quiz_k01").set_value(1).run()
    app.button(key="konu04_quiz_check_k01").click().run()
    assert not app.exception
    assert any(item.value == "Doğru." for item in app.success)
    _select(app, "konu03")
    app.radio(key="konu03_quiz_k01").set_value(0).run()
    app.button(key="konu03_quiz_check_k01").click().run()
    assert any("Tekrar edilecek bölümler" in item.value and "§3.1" in item.value for item in app.markdown)


def test_konu05_regressor_choice_carries_to_later_steps_and_resets() -> None:
    app = _run_app()
    _select(app, "konu05")
    assert "Çoklu Regresyon Modeli ve Ceteris Paribus Yorumu" in _markdown(app)
    assert _metrics(app)["Çoklu model: eğitim katsayısı"] == "0,5990"
    app.multiselect(key="konu05_secim_adim1_x").set_value(["educ", "exper"]).run()
    assert not app.exception
    assert any("notlardan farklı" in item.value for item in app.info)
    app.segmented_control(key="konu05_lab_step").set_value(2).run()
    assert _metrics(app)["Tahmin farkı B − A (dolar)"] == "2,577"
    assert any("(Adım 1)" in item.value for item in app.info)
    app.button(key="konu05_notlara_don_2").click().run()
    assert not app.exception
    assert _metrics(app)["Tahmin farkı B − A (dolar)"] == "2,396"


def test_konu06_omitted_variable_and_vif_choices() -> None:
    app = _run_app()
    _select(app, "konu06")
    assert _metrics(app)["Uzun model: eğitim katsayısı"] == "0,6443"
    app.segmented_control(key="konu06_secim_adim1_z").set_value("tenure").run()
    assert not app.exception
    assert _metrics(app)["Uzun model: eğitim katsayısı"] == "0,5691"
    app.segmented_control(key="konu06_lab_step").set_value(5).run()
    app.multiselect(key="konu06_secim_adim5_x").set_value(["sqrft", "bdrms"]).run()
    assert not app.exception
    assert any("notlardan farklı" in item.value for item in app.info)


def test_konu05_and_konu06_quizzes_grade_answers() -> None:
    app = _run_app()
    for topic, correct in (("konu05", 1), ("konu06", 2)):
        _select(app, topic)
        app.radio(key=f"{topic}_quiz_k01").set_value(correct).run()
        app.button(key=f"{topic}_quiz_check_k01").click().run()
        assert not app.exception, topic
        assert any(item.value == "Doğru." for item in app.success), topic


def test_konu07_and_konu08_choices_notice_and_quizzes() -> None:
    app = _run_app()
    _select(app, "konu07")
    assert "Tek Katsayı İçin Hipotez Testleri" in _markdown(app)
    app.segmented_control(key="konu07_lab_step").set_value(10).run()
    assert _metrics(app)["p-değeri"] == "< 0,001"
    app.segmented_control(key="konu07_secim_adim10_terim").set_value("exper").run()
    assert not app.exception
    assert _metrics(app)["p-değeri"] == "0,064"
    assert any("notlardan farklı" in item.value for item in app.info)
    app.radio(key="konu07_quiz_k01").set_value(1).run()
    app.button(key="konu07_quiz_check_k01").click().run()
    assert any(item.value == "Doğru." for item in app.success)
    _select(app, "konu08")
    assert _metrics(app)["Ortak F testi, H₀: β_deneyim = 0, β_kıdem = 0"] == "53,31"
    app.multiselect(key="konu08_secim_adim1_x").set_value(["educ", "exper", "tenure"]).run()
    assert not app.exception
    assert _metrics(app)["Ortak F testi, H₀: β_eğitim = 0, β_deneyim = 0, β_kıdem = 0"] == "76,87"
    app.radio(key="konu08_quiz_k01").set_value(2).run()
    app.button(key="konu08_quiz_check_k01").click().run()
    assert any(item.value == "Doğru." for item in app.success)


def test_konu09_and_konu10_choices_notice_and_quizzes() -> None:
    app = _run_app()
    _select(app, "konu09")
    assert "Ölçekleme, Logaritmik Modeller, Karesel Terimler ve Model Seçimi" in _markdown(app)
    assert _metrics(app)["Ücret sent, eğitim yıl: eğitim katsayısı"] == "59,897"
    app.segmented_control(key="konu09_secim_adim1_y").set_value("bin").run()
    assert not app.exception
    assert _metrics(app)["Ücret bin dolar, eğitim yıl: eğitim katsayısı"] == "0,000599"  # 0,001 yazılıp kaybolmaz
    assert any("notlardan farklı" in item.value for item in app.info)
    app.segmented_control(key="konu09_lab_step").set_value(5).run()
    assert _metrics(app)["Deneyim dönüm noktası −β̂₂ / (2β̂₃)"] == "24,76"
    app.radio(key="konu09_quiz_k01").set_value(2).run()
    app.button(key="konu09_quiz_check_k01").click().run()
    assert any(item.value == "Doğru." for item in app.success)
    _select(app, "konu10")
    assert "Kukla Değişkenler ve Kategorik Açıklayıcı Değişkenler" in _markdown(app)
    assert _metrics(app)["Kukla katsayısı: kadın − erkek farkı"] == "−2,5118"
    app.segmented_control(key="konu10_secim_adim1_kod").set_value("male").run()
    assert not app.exception
    assert _metrics(app)["Kukla katsayısı: erkek − kadın farkı"] == "2,5118"
    assert any("notlardan farklı" in item.value for item in app.info)
    app.radio(key="konu10_quiz_k01").set_value(2).run()
    app.button(key="konu10_quiz_check_k01").click().run()
    assert any(item.value == "Doğru." for item in app.success)
