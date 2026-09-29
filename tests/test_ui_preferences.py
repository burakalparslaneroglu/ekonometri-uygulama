"""Kullanıcı kontrollü uygulama geneli metin ölçeği denetimleri."""
from pathlib import Path

from core.ui_preferences import DEFAULT_TEXT_SCALE, TEXT_SCALE_OPTIONS, normalize_text_scale, text_scale_css

ROOT = Path(__file__).resolve().parents[1]


def test_text_scale_options_are_safe() -> None:
    """Dört ölçek ve varsayılan doğrulanır; geçersiz değer varsayılana döner."""
    assert tuple(TEXT_SCALE_OPTIONS.values()) == (1.0, 1.1, 1.2, 1.3)
    assert DEFAULT_TEXT_SCALE == 1.1
    assert normalize_text_scale(float("nan")) == DEFAULT_TEXT_SCALE
    assert normalize_text_scale(2.0) == DEFAULT_TEXT_SCALE


def test_text_scale_uses_single_root_relative_pass() -> None:
    """Alt bileşenler ebeveyn ``em`` değeriyle yeniden çarpılmaz."""
    css_100 = text_scale_css(1.0)
    css_130 = text_scale_css(1.3)
    static_css = (ROOT / "assets" / "styles.css").read_text(encoding="utf-8")
    assert "--app-font-scale: 1.30" in css_130
    assert "calc(1em" not in css_100 + css_130
    assert "font-size: calc" not in static_css
    assert ".stApp { font-size: 1.000rem; }" in css_100
    assert ".stApp { font-size: 1.300rem; }" in css_130
    for selector in ("stSidebar", "stWidgetLabel", "stMetricLabel", "stMetricValue", "stDataFrame", "stTable"):
        assert selector in css_130
