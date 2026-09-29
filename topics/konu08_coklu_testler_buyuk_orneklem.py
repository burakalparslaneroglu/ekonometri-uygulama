"""Konu 08: Birden Fazla Kısıtın Sınanması: F Testi ve Büyük Örneklem Mantığı."""

from __future__ import annotations

import streamlit as st

from core.labs.registry import get_lab
from core.labs.sezgi_konu08 import KONU08_EXPERIMENTS
from core.quiz.registry import get_quiz
from topics.lab_ui import render_lab
from topics.lab_ui import widget_keys as lab_widget_keys
from topics.quiz_ui import render_quiz
from topics.shared import render_topic_header
from topics.sim_ui import render_experiments
from topics.sim_ui import widget_keys as experiment_widget_keys

TOPIC_KEY = "konu08"


def render() -> None:
    render_topic_header(TOPIC_KEY)
    application, intuition, self_test = st.tabs(("Uygulama", "Sezgi", "Kendini sına"))
    with application:
        render_lab(get_lab(TOPIC_KEY))
    with intuition:
        render_experiments(KONU08_EXPERIMENTS)
    with self_test:
        render_quiz(get_quiz(TOPIC_KEY))


def widget_keys() -> set[str]:
    """Konu çizilmediğinde de korunan seçimler: adım, spesifikasyon, deney ve kaydırıcılar."""

    return lab_widget_keys(get_lab(TOPIC_KEY)) | experiment_widget_keys(KONU08_EXPERIMENTS)
