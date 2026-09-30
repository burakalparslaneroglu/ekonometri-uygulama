"""Uygulama genelinde kullanılan değişmez metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopicMetadata:
    """Bir konu sayfasının tek kaynak metadata'sı; başlık ders notlarındaki bölüm adıdır."""

    key: str
    number: int
    title: str
    short_title: str
    guiding_question: str = ""
    """Konu başlığının altında gösterilen yönlendirici soru."""

    @property
    def label(self) -> str:
        return f"Konu {self.number:02d} · {self.short_title}"
