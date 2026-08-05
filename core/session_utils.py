"""Konu modüllerinde soru durumunu yönetmek için oturum yardımcıları."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


QUESTION_INDEX_KEY = "konu03_question_index"
QUESTION_MODEL_KEY = "konu03_question_model_id"
ANSWER_VISIBLE_KEY = "konu03_answer_visible"


def question_state_keys(topic_id: str) -> tuple[str, str, str]:
    """Konuya özgü soru durumu anahtarlarını kararlı biçimde üretir."""
    if not topic_id.strip():
        raise ValueError("Konu kimliği boş olamaz.")
    return (
        f"{topic_id}_question_index",
        f"{topic_id}_question_model_id",
        f"{topic_id}_answer_visible",
    )


def synchronize_question_state(state: MutableMapping[str, Any], model_id: str, topic_id: str = "konu03") -> bool:
    """Model değiştiğinde eski soru sırasını ve cevap görünürlüğünü sıfırlar."""
    index_key, model_key, answer_key = question_state_keys(topic_id)
    if state.get(model_key) == model_id:
        return False
    state[model_key] = model_id
    state[index_key] = 0
    state[answer_key] = False
    return True


def reveal_answer(state: MutableMapping[str, Any], topic_id: str = "konu03") -> None:
    """Mevcut sorunun cevabını görünür yapar."""
    _, _, answer_key = question_state_keys(topic_id)
    state[answer_key] = True


def next_question(state: MutableMapping[str, Any], topic_id: str = "konu03") -> int:
    """Soru sırasını artırır ve yeni cevabı gizler."""
    index_key, _, answer_key = question_state_keys(topic_id)
    current = int(state.get(index_key, 0))
    state[index_key] = current + 1
    state[answer_key] = False
    return current + 1


def reset_question_state(state: MutableMapping[str, Any], topic_id: str) -> None:
    """Konu değişiminde ilgili konunun soru durumunu başlangıca döndürür."""
    index_key, model_key, answer_key = question_state_keys(topic_id)
    state.pop(index_key, None)
    state.pop(model_key, None)
    state.pop(answer_key, None)


def synchronize_active_topic(state: MutableMapping[str, Any], topic_id: str) -> bool:
    """Konu değiştiğinde yeni konunun soru durumunu temizler."""
    active_topic_key = "active_topic_id"
    if state.get(active_topic_key) == topic_id:
        return False
    reset_question_state(state, topic_id)
    state[active_topic_key] = topic_id
    return True
