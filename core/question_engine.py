"""Eski sayfaların (Konu 5–12) ortak soru veri yapısı ve kararlı soru türü sırası."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratedQuestion:
    """Bir sorunun öğrenciye gösterilecek metnini ve gizli çözümünü tutar."""

    question_type: str
    prompt: str
    answer: str
    index: int


def _stable_number(model_id: str, question_index: int, salt: str = "") -> int:
    """Kimlikten süreçler arası sabit bir tamsayı üretir."""
    source = f"{model_id}|{question_index}|{salt}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(source).digest()[:8], "big")


def cycle_question_type(
    model_id: str,
    question_index: int,
    question_types: tuple[str, ...],
    *,
    namespace: str,
) -> str:
    """Bir konuya ait soru türlerini deterministik ve ardışık döndürür."""
    if question_index < 0:
        raise ValueError("Soru sırası negatif olamaz.")
    if not question_types:
        raise ValueError("En az bir soru türü tanımlanmalıdır.")
    start = _stable_number(model_id, 0, f"{namespace}:type") % len(question_types)
    return question_types[(start + question_index) % len(question_types)]
