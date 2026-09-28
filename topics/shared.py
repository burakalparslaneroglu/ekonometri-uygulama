"""Yeni mimariye taşınan konu sayfalarının ortak başlığı."""

from __future__ import annotations

from collections.abc import Iterable
from html import escape

import streamlit as st

from core.topic_registry import get_topic


def render_topic_header(topic_key: str) -> None:
    topic = get_topic(topic_key)
    st.markdown(
        "<section class='topic-band'>"
        f"<div class='topic-number'>Konu {topic.number:02d}</div>"
        f"<h2>{escape(topic.title)}</h2>"
        "</section>",
        unsafe_allow_html=True,
    )
    if topic.guiding_question:
        st.markdown(
            "<div class='guiding-question'>"
            "<div class='label'>Yönlendirici soru</div>"
            f"<p>{escape(topic.guiding_question)}</p>"
            "</div>",
            unsafe_allow_html=True,
        )


def keep_widget_state(keys: Iterable[str]) -> None:
    """Çizilmeyen widget'ların seçimlerini korur.

    Streamlit bir çalıştırmada çizilmeyen widget'ın değerini siler (ör. öğrenci Adım 4'te açıklayıcı değişkeni
    değiştirip Adım 5'e geçtiğinde Adım 4'ün denetimi çizilmez). Değeri Session State'e yeniden yazmak onu korur;
    seçim sonraki adımlara geçer ve öğrenci geri döndüğünde yerinde durur.
    """

    for key in keys:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
