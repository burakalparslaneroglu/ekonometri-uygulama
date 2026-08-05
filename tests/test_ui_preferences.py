"""Kullanıcı kontrollü uygulama geneli metin ölçeği denetimleri."""

from core.ui_preferences import DEFAULT_TEXT_SCALE, TEXT_SCALE_OPTIONS, normalize_text_scale, plot_font_sizes, text_scale_css


def test_text_scale_options_css_and_plot_sizes_are_safe() -> None:
    """Dört ölçek, varsayılan, CSS değişkeni ve Plotly boyutları doğrulanır."""
    assert tuple(TEXT_SCALE_OPTIONS.values()) == (1.0, 1.1, 1.2, 1.3)
    assert DEFAULT_TEXT_SCALE == 1.1
    assert normalize_text_scale(float("nan")) == DEFAULT_TEXT_SCALE
    assert normalize_text_scale(2.0) == DEFAULT_TEXT_SCALE
    assert "--app-font-scale: 1.30" in text_scale_css(1.3)
    small, large = plot_font_sizes(1.0), plot_font_sizes(1.3)
    assert all(value > 0 for value in large.values())
    assert all(large[name] >= small[name] for name in small)
