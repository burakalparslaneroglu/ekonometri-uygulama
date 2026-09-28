"""Uygulama metinlerinde sayıya bağlı Türkçe ifadeler.

Kural: değişken bir sayıya Türkçe ek getirilmez ("0,54'lük" değil "0,54 birim"); yön sözcükleri sayıdan
ayrı yazılır.
"""

from __future__ import annotations


def tr_lower(text: str) -> str:
    """Türkçe küçük harf: I → ı, İ → i (``str.lower`` "İ"yi noktalı iki karaktere çevirir)."""

    return text.replace("I", "ı").replace("İ", "i").lower()


def signed_difference(value: float, tolerance: float = 1e-12) -> str:
    """Bir farkın yönü, yüklemle: "yüksektir", "düşüktür" ya da "aynıdır"."""

    if value > tolerance:
        return "yüksektir"
    if value < -tolerance:
        return "düşüktür"
    return "aynıdır"
