"""Konu modüllerinde soru durumunu yönetmek için oturum yardımcıları."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


QUESTION_INDEX_KEY = "konu03_question_index"
QUESTION_MODEL_KEY = "konu03_question_model_id"
ANSWER_VISIBLE_KEY = "konu03_answer_visible"


def synchronize_question_state(state: MutableMapping[str, Any], model_id: str) -> bool:
    """Model değiştiğinde eski soru sırasını ve cevap görünürlüğünü sıfırlar."""
    if state.get(QUESTION_MODEL_KEY) == model_id:
        return False
    state[QUESTION_MODEL_KEY] = model_id
    state[QUESTION_INDEX_KEY] = 0
    state[ANSWER_VISIBLE_KEY] = False
    return True


def reveal_answer(state: MutableMapping[str, Any]) -> None:
    """Mevcut sorunun cevabını görünür yapar."""
    state[ANSWER_VISIBLE_KEY] = True


def next_question(state: MutableMapping[str, Any]) -> int:
    """Soru sırasını artırır ve yeni cevabı gizler."""
    current = int(state.get(QUESTION_INDEX_KEY, 0))
    state[QUESTION_INDEX_KEY] = current + 1
    state[ANSWER_VISIBLE_KEY] = False
    return current + 1
