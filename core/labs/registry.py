"""Ders notu uygulamalarının kaydı. Yeni bir konu buraya eklendiğinde testler onu kendiliğinden kapsar."""

from __future__ import annotations

from core.labs.konu01 import KONU01_LAB
from core.labs.konu02 import KONU02_LAB
from core.labs.spec import LabSpec

LABS: dict[str, LabSpec] = {lab.topic_key: lab for lab in (KONU01_LAB, KONU02_LAB)}


def get_lab(topic_key: str) -> LabSpec | None:
    """Konunun uygulama tanımı; henüz hazırlanmadıysa ``None``."""

    return LABS.get(topic_key)
