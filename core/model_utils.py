"""Eski konu sayfalarının (Konu 10–11) öğrenci sayı biçimleri. Streamlit bağımlılığı yoktur.

Basit EKK, tahmin ve artık yardımcıları Konu 3–6'nın yeni yapıya taşınmasıyla kaldırıldı; yeni yapıda hesaplar
``core/labs`` tanımlarından yapılır.
"""

from __future__ import annotations

import numpy as np


def format_student_number(value: float, *, decimals: int = 6, zero_tolerance: float = 1e-10) -> str:
    """Öğrenci görünümü için sonlu sayıyı bilimsel gösterim kullanmadan yazar."""
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("Biçimlendirilecek değer sonlu olmalıdır.")
    if zero_tolerance < 0:
        raise ValueError("Sıfır toleransı negatif olamaz.")
    if abs(number) <= zero_tolerance:
        return "0"
    return f"{number:.{decimals}f}".rstrip("0").rstrip(".")


def format_numerical_difference(value: float, *, decimals: int = 6, zero_tolerance: float = 1e-10) -> str:
    """Cebirsel doğrulama farkını tolerans mesajıyla öğrenciye sunar."""
    number = float(value)
    if abs(number) <= zero_tolerance:
        return "sayısal tolerans içinde 0"
    return format_student_number(number, decimals=decimals, zero_tolerance=zero_tolerance)
