"""Sunum URL'lerinin sınırları ve gerçek uygulamadaki oturum davranışı."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.navigation import experiments, initial_state, parse_route
from core.labs.registry import LABS
from core.topic_registry import get_topic

APP = Path(__file__).resolve().parents[1] / 'app.py'


def query(**values):
    return {key: [str(value)] for key, value in values.items()}


@pytest.mark.parametrize('topic', range(13))
def test_every_registered_target_is_valid(topic):
    key = f'konu{topic:02d}'
    for experiment in experiments(key):
        route = parse_route(query(konu=topic, sekme='sezgi', deney=experiment.number))
        assert dict(route.parameters) == experiment.defaults()
        assert initial_state(route)['selected_topic'] == get_topic(key).label
    for step in LABS[key].steps:
        assert parse_route(query(konu=topic, sekme='uygulama', adim=step.number)).step == step.number


@pytest.mark.parametrize('values', [
    {'konu': '13'}, {'konu': '-1'}, {'konu': 'nan'}, {'konu': '3', 'sekme': 'bilinmeyen'},
    {'konu': '3', 'sekme': 'sezgi', 'deney': '99'}, {'konu': '3', 'adim': '99'},
    {'konu': '3', 'kaynak': 'yanlis'}, {'konu': '3', 'sekme': 'sezgi', 'adim': '2'},
    {'konu': '3', 'sekme': 'sezgi', 'deney': '1', 'ayar_n': '30.5'},
    {'konu': '3', 'sekme': 'sezgi', 'deney': '1', 'ayar_sigma': 'nan'},
    {'konu': '3', 'sekme': 'sezgi', 'deney': '1', 'ayar_sigma': '100'},
    {'konu': '3', 'sekme': 'sezgi', 'deney': '1', 'ayar_sigma': '1.03'},
    {'konu': '3', 'sekme': 'sezgi', 'deney': '1', 'ayar_bilinmeyen': '1'},
    {'sekme': 'sezgi'}, {'konu': '3', 'sekme': 'sinama', 'deney': '1'},
])
def test_invalid_routes_are_rejected(values):
    with pytest.raises(ValueError):
        parse_route(query(**values))


def test_repeated_parameter_is_rejected():
    with pytest.raises(ValueError, match='yalnız bir kez'):
        parse_route({'konu': ['3', '4']})
    assert parse_route({}) is None


@pytest.mark.parametrize('topic', range(13))
def test_direct_urls_open_real_experiment_and_lab(topic):
    key = f'konu{topic:02d}'
    app = AppTest.from_file(APP, default_timeout=60)
    app.query_params.update(konu=str(topic), sekme='sezgi', deney='2')
    app.run()
    assert not app.exception and not app.error
    assert app.session_state[f'{key}_tab'] == 'Sezgi'
    assert app.segmented_control(key=f'{key}_sezgi_deney').value == 2
    assert any(s.value.startswith('Deney 2:') for s in app.subheader)
    app.query_params.clear()
    app.query_params.update(konu=str(topic), sekme='uygulama', adim=str(LABS[key].steps[-1].number))
    app.run()
    assert not app.exception and not app.error
    assert app.session_state[f'{key}_tab'] == 'Uygulama'
    assert app.segmented_control(key=f'{key}_lab_kaynak').value == 'notlar'
    assert app.segmented_control(key=f'{key}_lab_step').value == LABS[key].steps[-1].number


def test_starting_settings_apply_once_and_manual_navigation_survives():
    app = AppTest.from_file(APP, default_timeout=60)
    app.query_params.update(konu='03', sekme='sezgi', deney='1', ayar_n='50')
    app.run()
    assert not app.exception
    assert app.slider(key='konu03_sezgi1_n').value == 50
    app.slider(key='konu03_sezgi1_n').set_value(80).run()
    assert app.slider(key='konu03_sezgi1_n').value == 80
    app.segmented_control(key='konu03_sezgi_deney').set_value(3).run()
    assert app.segmented_control(key='konu03_sezgi_deney').value == 3
    app.session_state['konu03_tab'] = 'Uygulama'
    app.run()
    assert app.session_state['konu03_tab'] == 'Uygulama'
    app.radio(key='selected_topic').set_value(get_topic('konu04').label).run()
    assert app.radio(key='selected_topic').value == get_topic('konu04').label
    app.query_params.clear()
    app.run()
    app.query_params.update(konu='03', sekme='sezgi', deney='1', ayar_n='50')
    app.run()
    assert not app.exception
    assert app.slider(key='konu03_sezgi1_n').value == 50


def test_lab_url_resets_previous_source_and_notes_specification_once():
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.radio(key='selected_topic').set_value(get_topic('konu05').label).run()
    app.multiselect(key='konu05_secim_adim1_x').set_value(['educ', 'exper']).run()
    app.segmented_control(key='konu05_lab_kaynak').set_value('alternatif').run()
    app.query_params.update(konu='05', sekme='uygulama', adim='2')
    app.run()
    assert not app.exception
    assert app.segmented_control(key='konu05_lab_kaynak').value == 'notlar'
    assert app.session_state['konu05_secim_adim1_x'] == ['educ', 'exper', 'tenure']
    app.segmented_control(key='konu05_lab_step').set_value(3).run()
    assert app.segmented_control(key='konu05_lab_step').value == 3


def test_fractional_url_setting_and_slider_change_update_the_result():
    app = AppTest.from_file(APP, default_timeout=60)
    app.query_params.update(konu='03', sekme='sezgi', deney='2', ayar_egim='0.5')
    app.run()
    assert not app.exception
    assert app.select_slider(key='konu03_sezgi2_egim').value == 0.5
    before = {m.label:m.value for m in app.metric}['Aday: Σ(artık)²']
    app.select_slider(key='konu03_sezgi2_egim').set_value(0.8).run()
    assert not app.exception
    assert app.select_slider(key='konu03_sezgi2_egim').value == 0.8
    assert any('b = 0{,}8' in latex.value for latex in app.latex)
    assert {m.label:m.value for m in app.metric}['Aday: Σ(artık)²'] != before


def test_own_data_url_keeps_target_and_shows_upload_without_file():
    app = AppTest.from_file(APP, default_timeout=60)
    app.query_params.update(konu='03', sekme='uygulama', adim='4', kaynak='kendi')
    app.run()
    assert not app.exception
    assert app.segmented_control(key='konu03_lab_step').value == 4
    assert any(s.value.startswith('Adım 4:') for s in app.subheader)
    assert any('Başlamak için bir dosya yükleyin' in s.value for s in app.info)


def test_invalid_url_shows_error_without_initializing_wrong_experiment():
    app = AppTest.from_file(APP, default_timeout=60)
    app.query_params.update(konu='03', sekme='sezgi', deney='9')
    app.run()
    assert not app.exception
    assert any('Deney 9 bulunmuyor' in s.value for s in app.error)
    assert not app.subheader
