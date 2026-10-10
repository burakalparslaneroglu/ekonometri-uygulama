"""Yeni mimariye taşınan konu sayfalarının ortak başlığı."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from html import escape

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from core.topic_registry import get_topic
from core.navigation import initial_state, parse_route


def apply_url_route() -> None:
    """URL değiştiğinde uygular; aynı URL'deki sonraki kullanıcı seçimlerini korur."""
    query = {key: st.query_params.get_all(key) for key in st.query_params}
    token = tuple(sorted((key, tuple(values)) for key, values in query.items()))
    try:
        route = parse_route(query)
    except ValueError as error:
        st.error(f"Sunum bağlantısı açılamadı: {error}")
        st.stop()
    if st.session_state.get("_sunum_url") != token:
        if route is not None:
            st.session_state.update(initial_state(route))
        st.session_state["_sunum_url"] = token


def topic_tabs(topic_key: str, labels: tuple[str, ...]) -> Sequence[DeltaGenerator]:
    """Sekme sırasını korur; URL ve normal sekme seçimi aynı widget'ı kullanır."""
    key = f"{topic_key}_tab"
    if st.session_state.get(key) not in labels:
        st.session_state[key] = labels[0]
    return st.tabs(labels, key=key, on_change="rerun")


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
