"""Grafiklerin görünür eksen başlıklarını koruyan kaynak düzeyi testler."""

from __future__ import annotations

import ast
from pathlib import Path


NATIVE_CHART_METHODS = {"line_chart", "scatter_chart", "bar_chart", "area_chart"}


def test_all_streamlit_native_charts_define_axis_labels() -> None:
    """Konu sayfalarındaki yerel Streamlit grafikleri iki ekseni de adlandırmalıdır."""
    topics_dir = Path(__file__).resolve().parents[1] / "topics"
    missing: list[str] = []

    for path in sorted(topics_dir.glob("konu*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == "st"
                and function.attr in NATIVE_CHART_METHODS
            ):
                continue
            keywords = {item.arg for item in node.keywords if item.arg is not None}
            absent = {"x_label", "y_label"} - keywords
            if absent:
                missing.append(f"{path.name}:{node.lineno} ({', '.join(sorted(absent))})")

    assert not missing, "Eksen başlığı eksik Streamlit grafikleri: " + "; ".join(missing)
