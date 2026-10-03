"""Uygulama metinlerinde sayıya bağlı Türkçe ifadeler.

Kural: değişken bir sayıya Türkçe ek getirilmez ("0,54'lük" değil "0,54 birim"); yön sözcükleri sayıdan
ayrı yazılır.
"""

from __future__ import annotations


def tr_lower(text: str) -> str:
    """Türkçe küçük harf: I → ı, İ → i (``str.lower`` "İ"yi noktalı iki karaktere çevirir)."""

    return text.replace("I", "ı").replace("İ", "i").lower()


def at_zero(value: float, decimals: int) -> bool:
    """Değer gösterilen basamakta sıfır mı (ör. dört basamakta |değer| < 0,00005)."""

    return abs(value) < 0.5 * 10 ** (-decimals)


def directions(start: str, end: str) -> tuple[str, str]:
    """İleri ve geri yönün etiketi ("12 → 16", "16 → 12").

    Başlangıç ile yeni değer aynıysa yön x₀/x₁ ile ayrılır; aynı etiketli iki satır üretilen Python kodunda
    (sözlükle kurulan tablo) birleşirdi.
    """

    if start == end:
        return f"x₀ → x₁ ({start} → {end})", f"x₁ → x₀ ({end} → {start})"
    return f"{start} → {end}", f"{end} → {start}"


def signed_difference(value: float, tolerance: float = 1e-12) -> str:
    """Bir farkın yönü, yüklemle: "yüksektir", "düşüktür" ya da "aynıdır"."""

    if value > tolerance:
        return "yüksektir"
    if value < -tolerance:
        return "düşüktür"
    return "aynıdır"
