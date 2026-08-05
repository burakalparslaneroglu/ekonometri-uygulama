"""Uygulama geneli, kullanıcı kontrollü görünüm tercihleri."""

from __future__ import annotations

import math


TEXT_SCALE_OPTIONS: dict[str, float] = {
    "Normal — %100": 1.00,
    "Büyük — %110": 1.10,
    "Daha büyük — %120": 1.20,
    "Çok büyük — %130": 1.30,
}
DEFAULT_TEXT_SCALE_LABEL = "Büyük — %110"
DEFAULT_TEXT_SCALE = TEXT_SCALE_OPTIONS[DEFAULT_TEXT_SCALE_LABEL]


def normalize_text_scale(value: object, *, default: float = DEFAULT_TEXT_SCALE) -> float:
    """Tanımlı olmayan ya da sonlu olmayan ölçekleri güvenli varsayılana döndürür."""
    try:
        scale = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(scale):
        return default
    return scale if scale in TEXT_SCALE_OPTIONS.values() else default


def text_scale_css(scale: object) -> str:
    """Uygulama bileşenlerini tek CSS değişkeniyle ölçekleyen stil üretir."""
    value = normalize_text_scale(scale)
    return f"""
    <style>
    :root {{ --app-font-scale: {value:.2f}; }}
    .stApp, [data-testid="stSidebar"], [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"], [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    .stButton > button, [data-testid="stDataFrame"], [data-testid="stTable"],
    [data-testid="stCodeBlock"], [data-testid="stExpander"], [data-testid="stAlert"] {{
      font-size: calc(1em * var(--app-font-scale));
    }}
    .stApp h1 {{ font-size: clamp(1.85rem, calc(2.35rem * var(--app-font-scale)), 2.95rem); }}
    .stApp h2 {{ font-size: clamp(1.45rem, calc(1.80rem * var(--app-font-scale)), 2.30rem); }}
    .stApp h3 {{ font-size: clamp(1.18rem, calc(1.40rem * var(--app-font-scale)), 1.80rem); }}
    [data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] p {{ font-size: calc(1em * var(--app-font-scale)); }}
    .stButton > button {{ min-height: calc(2.5rem * var(--app-font-scale)); white-space: normal; line-height: 1.25; }}
    </style>
    """


def plot_font_sizes(scale: object) -> dict[str, int]:
    """Plotly için ölçeklenmiş ve güvenli yazı boyutlarını döndürür."""
    value = normalize_text_scale(scale)
    return {
        "base": max(10, round(14 * value)),
        "title": max(12, round(18 * value)),
        "annotation": max(10, round(13 * value)),
        "tick": max(9, round(12 * value)),
        "legend": max(9, round(12 * value)),
        "hover": max(10, round(13 * value)),
    }
