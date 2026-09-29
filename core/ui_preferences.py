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
    """Metni her bileşende yalnız bir kez, kök ``rem`` değerleriyle ölçekler."""
    value = normalize_text_scale(scale)
    sizes = {
        "body": 1.00 * value,
        "sidebar": 0.95 * value,
        "caption": 0.875 * value,
        "widget": 0.95 * value,
        "button": 0.95 * value,
        "metric_label": 0.85 * value,
        "metric_value": 1.55 * value,
        "table": 0.90 * value,
        "code": 0.90 * value,
        "panel": 0.95 * value,
        "badge": 0.78 * value,
        "lead": 1.08 * value,
        "h1": min(2.95, 2.35 * value),
        "h2": min(2.30, 1.80 * value),
        "h3": min(1.80, 1.40 * value),
    }
    return f"""
    <style>
    :root {{ --app-font-scale: {value:.2f}; }}
    .stApp {{ font-size: {sizes['body']:.3f}rem; }}
    .app-kicker, .topic-badge, .topic-band .topic-number, .guiding-question .label {{
      font-size: {sizes['badge']:.3f}rem; }}
    .guiding-question p {{ font-size: {sizes['lead']:.3f}rem; }}
    [data-testid="stSidebar"], [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] p {{ font-size: {sizes['sidebar']:.3f}rem; }}
    [data-testid="stCaptionContainer"] {{ font-size: {sizes['caption']:.3f}rem; }}
    [data-testid="stWidgetLabel"] {{ font-size: {sizes['widget']:.3f}rem; }}
    [data-testid="stMetricLabel"] {{ font-size: {sizes['metric_label']:.3f}rem; }}
    [data-testid="stMetricValue"] {{ font-size: {sizes['metric_value']:.3f}rem; }}
    .stButton > button {{ font-size: {sizes['button']:.3f}rem; min-height: 2.5rem; white-space: normal; line-height: 1.25; }}
    [data-testid="stDataFrame"], [data-testid="stTable"] {{ font-size: {sizes['table']:.3f}rem; }}
    [data-testid="stCodeBlock"] {{ font-size: {sizes['code']:.3f}rem; }}
    [data-testid="stExpander"], [data-testid="stAlert"] {{ font-size: {sizes['panel']:.3f}rem; }}
    .stApp h1 {{ font-size: {sizes['h1']:.3f}rem; }}
    .stApp h2 {{ font-size: {sizes['h2']:.3f}rem; }}
    .stApp h3 {{ font-size: {sizes['h3']:.3f}rem; }}
    </style>
    """
