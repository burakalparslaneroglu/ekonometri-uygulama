"""Konu kaydı, konu modülleri ve yeni mimariye taşınan konuların sözleşmesi."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

from app import TOPIC_LABELS, TOPIC_RENDERERS
from core.labs.registry import LABS
from core.quiz.registry import QUIZZES
from core.topic_registry import get_topic, list_topics

TOPIC_MODULES = {
    "konu00": "topics.konu00_baslangic_arac_kutusu",
    "konu01": "topics.konu01_ampirik_arastirma",
    "konu02": "topics.konu02_veri_turleri_nedensellik",
    "konu03": "topics.konu03_basit_regresyon",
    "konu04": "topics.konu04_ols_cikti_fonksiyonel_bicimler",
    "konu05": "topics.konu05_coklu_regresyon",
    "konu06": "topics.konu06_ols_varsayimlari_yanlilik",
    "konu07": "topics.konu07_tekli_hipotez_testleri",
    "konu08": "topics.konu08_coklu_testler_buyuk_orneklem",
    "konu09": "topics.konu09_fonksiyonel_bicimler",
    "konu10": "topics.konu10_kukla_degiskenler",
    "konu11": "topics.konu11_etkilesimler_grup_farklari",
    "konu12": "topics.konu12_heteroskedastisite",
}

# Uygulama + Sezgi + Kendini sına yapısına geçmiş konular: kod iki dilde, tek tanımdan üretilir.
MIGRATED_TOPICS = {"konu00", "konu01", "konu02", "konu03", "konu04"}


def test_registry_has_exact_course_order() -> None:
    topics = list_topics()
    assert [topic.number for topic in topics] == list(range(0, 13))
    assert [topic.key for topic in topics] == list(TOPIC_MODULES)
    assert len({topic.title for topic in topics}) == 13
    assert all(topic.label.startswith(f"Konu {topic.number:02d} · ") for topic in topics)
    assert list(TOPIC_LABELS.values()) == list(TOPIC_MODULES)


def test_each_topic_module_exports_render() -> None:
    assert set(TOPIC_RENDERERS) == set(TOPIC_MODULES)
    for key, module_name in TOPIC_MODULES.items():
        module = importlib.import_module(module_name)
        assert callable(module.render), key
        assert TOPIC_RENDERERS[key] is module.render


def test_migrated_topics_use_three_tabs_and_single_source_definitions() -> None:
    assert set(LABS) == set(QUIZZES) == MIGRATED_TOPICS
    for key in MIGRATED_TOPICS:
        module = importlib.import_module(TOPIC_MODULES[key])
        source = inspect.getsource(module)
        assert importlib.import_module(f"core.labs.sezgi_{key}"), key
        assert '("Uygulama", "Sezgi", "Kendini sına")' in source, key
        assert "render_lab(" in source and "render_experiments(" in source and "render_quiz(" in source, key
        assert callable(module.widget_keys), key
        topic = get_topic(key)
        assert topic.guiding_question.endswith("?"), key
        assert source.splitlines()[0] == f'"""Konu {topic.number:02d}: {topic.title}."""', key


def test_legacy_modules_of_migrated_topics_are_gone() -> None:
    for name in ("core/research_question_utils.py", "core/scenario_registry.py", "core/data_structure_utils.py",
                 "core/group_comparison_utils.py", "tests/test_konu01_ui.py", "tests/test_konu02_ui.py",
                 "tests/test_konu01_questions.py", "tests/test_konu02_questions.py", "core/konu04_questions.py",
                 "tests/test_konu03_ui.py", "tests/test_konu04_ui.py", "tests/test_konu04_utils.py"):
        assert not Path(name).exists(), name
