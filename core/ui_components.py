"""Konu sayfalarında tekrar kullanılan küçük Streamlit bileşenleri."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import streamlit as st

from core.session_utils import next_question, question_state_keys


def toggle_answer(state: MutableMapping[str, Any], topic_id: str) -> bool:
    """Konuya ait cevap görünürlüğünü açar veya kapatır."""
    _, _, answer_key = question_state_keys(topic_id)
    state[answer_key] = not bool(state.get(answer_key, False))
    return bool(state[answer_key])


def render_question_actions(*, topic_id: str) -> None:
    """Yakın, ölçeklenebilir primary/secondary soru eylemlerini gösterir."""
    _, _, answer_key = question_state_keys(topic_id)
    visible = bool(st.session_state.get(answer_key, False))
    with st.container(key=f"question_actions_{topic_id}"):
        show, new, _ = st.columns((1.0, 1.0, 2.4), gap="small")
        if show.button(
            "Cevabı gizle" if visible else "Cevabı göster",
            type="primary",
            width="stretch",
            key=f"{topic_id}_show_answer",
        ):
            toggle_answer(st.session_state, topic_id)
            st.rerun()
        if new.button("Yeni soru", type="secondary", width="stretch", key=f"{topic_id}_next_question"):
            next_question(st.session_state, topic_id)
            st.rerun()
