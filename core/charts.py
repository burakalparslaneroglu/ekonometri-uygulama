"""Uygulama ve Sezgi sekmelerinin ortak grafik katmanı.

Grafikler üretilen Python ve R koduyla aynı veriden, aynı renk sırasıyla çizilir. ``show_figure`` bu
sekmelerde ``st.plotly_chart``'ı çağıran tek yerdir ve eksen adları olmayan grafiği reddeder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.codegen.base import HEAT_LOW, PALETTE, REFERENCE_COLORS
from core.labs import inference as I
from core.labs.runner import LabState, parameter, plot_key
from core.labs.spec import (
    CHARTS,
    BarChart,
    BoxPlot,
    ClassHistogram,
    CoefficientPlot,
    CompareBarChart,
    DensityCompare,
    DensityPlot,
    DotPlot,
    GroupedBarChart,
    HeatMap,
    Histogram,
    HypothesisPlot,
    IntervalPlot,
    LineChart,
    MosaicChart,
    Operation,
    PieChart,
    PmfWithDensity,
    ScatterPlot,
    TreeDiagram,
)
from core.labs.tables import boundary_label, density

CHART_TYPES = CHARTS


def tr_number(value: float, decimals: int = 0, percent: bool = False) -> str:
    """Türkçe sayı: ondalık virgül, tipografik eksi; yüzde işareti sayıdan önce."""

    if abs(value) < 0.5 * 10 ** (-decimals):  # yuvarlanınca sıfır: "−0,00" yazılmaz
        value = 0.0
    text = f"{value:.{decimals}f}".replace(".", ",").replace("-", "−")
    return "%" + text if percent else text


def style_figure(figure: go.Figure, *, title: str, x_title: str, y_title: str, legend_title: str = "") -> go.Figure:
    figure.update_layout(
        title={"text": title, "x": 0.0, "xanchor": "left", "font": {"size": 16}},
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title_text=legend_title,
        separators=",.",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 60, "b": 45},
        font={"family": "Source Sans Pro, sans-serif", "size": 14},
        hoverlabel={"font_size": 13},
    )
    return figure


def _has_axes(figure: go.Figure) -> bool:
    """Pasta grafiğinde ve eksenleri gizlenmiş grafiklerde (olasılık ağacı) eksen yoktur."""

    if any(isinstance(trace, go.Pie) for trace in figure.data):
        return False
    return figure.layout.xaxis.visible is not False or figure.layout.yaxis.visible is not False


def show_figure(figure: go.Figure, *, key: str | None = None) -> None:
    """Grafiği gösterir; eksenli bir grafikte eksen adı yoksa hatadır (pasta ve ağaçta eksen yoktur)."""

    if _has_axes(figure):
        x_title = figure.layout.xaxis.title.text or ""
        y_title = figure.layout.yaxis.title.text or ""
        if not x_title.strip() or not y_title.strip():
            raise ValueError("Her grafikte yatay ve dikey eksen adı açıkça belirtilmelidir.")
    st.plotly_chart(figure, width="stretch", key=key, config={"displaylogo": False})


# --- Grafik türleri ----------------------------------------------------------------

def _bar(op: BarChart, data: pd.DataFrame) -> go.Figure:
    labels = [tr_number(value, op.decimals, op.percent) for value in data["deger"]]
    if op.horizontal:
        trace = go.Bar(
            x=data["deger"], y=data["kategori"], orientation="h", marker_color=PALETTE[0], text=labels,
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{text}<extra></extra>",
        )
        figure = go.Figure(trace)
        figure.update_yaxes(autorange="reversed")
        figure.update_xaxes(rangemode="tozero")
        if op.y_range is not None:
            figure.update_xaxes(range=list(op.y_range))
        return style_figure(figure, title=op.title, x_title=op.y_label, y_title=op.x_label)
    trace = go.Bar(
        x=data["kategori"], y=data["deger"], marker_color=PALETTE[0], text=labels, textposition="outside",
        cliponaxis=False, hovertemplate="%{x}: %{text}<extra></extra>",
    )
    figure = go.Figure(trace)
    figure.update_xaxes(type="category")
    if op.y_range is not None:
        figure.update_yaxes(range=list(op.y_range))
    else:
        figure.update_yaxes(rangemode="tozero")
    return style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label)


def _series_bars(table: pd.DataFrame, *, stacked: bool, decimals: int, title: str, x_title: str,
                 y_title: str, legend_title: str, labels: bool = True) -> go.Figure:
    """Satırlar seriler, sütunlar yatay eksen: üretilen koddaki matrisle aynı düzen."""

    figure = go.Figure()
    categories = [str(column) for column in table.columns]
    for index, (series, row) in enumerate(table.iterrows()):
        values = row.to_numpy(dtype=float)
        shown = [tr_number(value, decimals) for value in values]
        figure.add_trace(
            go.Bar(
                name=str(series), x=categories, y=values, marker_color=PALETTE[index % len(PALETTE)],
                text=shown if labels else None, hovertext=shown,
                textposition="inside" if stacked else "outside", cliponaxis=False,
                insidetextfont={"color": "white"},
                hovertemplate=f"{series}<br>%{{x}}: %{{hovertext}}<extra></extra>",
            )
        )
    figure.update_layout(barmode="stack" if stacked else "group")
    figure.update_xaxes(type="category")
    figure.update_yaxes(rangemode="tozero")
    return style_figure(figure, title=title, x_title=x_title, y_title=y_title, legend_title=legend_title)


def _pie(op: PieChart, table: pd.DataFrame) -> go.Figure:
    values = table[op.column].to_numpy(dtype=float)
    # Plotly saat yönünün tersinde ilk dilimi ``rotation`` açısında BİTİRİR (açı 12 yönünden saat yönünde
    # ölçülür). İlk dilimin notlardaki gibi 0°'de (saat 3 yönü) BAŞLAMASI için onun açısı kadar geri dönülür.
    rotation = 90 - 360 * values[0] / values.sum()
    figure = go.Figure(
        go.Pie(
            labels=[str(label) for label in table.index], values=values, sort=False,
            direction="counterclockwise", rotation=rotation,
            marker={"colors": list(PALETTE[: len(values)]), "line": {"color": "white", "width": 1}},
            text=[tr_number(100 * value, 1, percent=True) for value in values], textinfo="label+text",
            hovertemplate="%{label}: %{text} · açı %{customdata:.1f}°<extra></extra>",
            customdata=table["aci"].to_numpy(dtype=float),
        )
    )
    figure.update_layout(showlegend=False)
    return style_figure(figure, title=op.title, x_title="", y_title="")


def _line(op: LineChart, data: pd.DataFrame, state: LabState) -> go.Figure:
    """Çizgi grafiği; ``references`` yatay başvuru çizgileridir (ör. gerçek olasılık) ve açıklamada görünür."""

    x = data[op.x].to_numpy(dtype=float)
    if op.series:
        figure = go.Figure()
        columns = [(op.y, op.legend or op.y_label), *op.series]
        for index, (column, label) in enumerate(columns):
            figure.add_trace(go.Scatter(
                x=x, y=data[column], mode="lines+markers" if op.markers else "lines", name=label,
                line={"color": PALETTE[index % len(PALETTE)], "width": 2.5}, marker={"size": 6},
                hovertemplate=f"{label}<br>%{{x}}: %{{y}}<extra></extra>",
            ))
        figure.update_layout(legend={"orientation": "h", "y": -0.25})
        return style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label)
    figure = go.Figure(
        go.Scatter(
            x=x, y=data[op.y], mode="lines+markers" if op.markers else "lines", name=op.y_label,
            line={"color": PALETTE[0], "width": 2.5}, marker={"size": 8},
            hovertemplate="%{x}: %{y}<extra></extra>" if op.markers else "%{x}: %{y:.3f}<extra></extra>",
        )
    )
    for index, (name, label) in enumerate(op.references):
        value = float(state.scalars[name])
        figure.add_trace(
            go.Scatter(
                x=[x.min(), x.max()], y=[value, value], mode="lines", name=label, hoverinfo="skip",
                line={"color": REFERENCE_COLORS[index % len(REFERENCE_COLORS)], "width": 2.5,
                      "dash": ("dash", "dot", "dashdot")[index % 3]},
            )
        )
    for column, label in op.bands:  # ör. μ ± 2σ/√n bandının kenarları; boş etiket açıklamada gösterilmez
        figure.add_trace(
            go.Scatter(
                x=x, y=data[column], mode="lines", name=label, showlegend=bool(label), hoverinfo="skip",
                line={"color": PALETTE[2], "width": 1.8, "dash": "dash"},
            )
        )
    if op.references or op.bands:
        figure.update_layout(legend={"orientation": "h", "y": -0.25})
    unique = np.unique(x)
    steps = np.diff(unique)
    if 1 < len(unique) <= 20 and np.allclose(steps, steps[0]):
        figure.update_xaxes(dtick=float(steps[0]))  # az sayıda eşit aralıklı değer: her değer bir işaret
    return style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label)


def _scatter(op: ScatterPlot, plot: dict) -> go.Figure:
    data = plot["veri"]
    figure = go.Figure(
        go.Scatter(
            x=data[op.x], y=data[op.y], mode="markers", name="Gözlemler",
            marker={"color": PALETTE[0], "size": op.size, "opacity": op.opacity},
            hovertemplate=f"{op.x_label}: %{{x}}<br>{op.y_label}: %{{y}}<extra></extra>",
        )
    )
    x = np.linspace(float(data[op.x].min()), float(data[op.x].max()), 100)
    if plot["dogru"] is not None:
        intercept, slope = plot["dogru"]
        figure.add_trace(go.Scatter(
            x=x, y=intercept + slope * x, mode="lines", name="Tahmin edilen doğru", hoverinfo="skip",
            line={"color": PALETTE[1], "width": 3},
        ))
    for index, (intercept, slope, label) in enumerate(plot.get("cizgiler", ())):
        figure.add_trace(go.Scatter(
            x=x, y=intercept + slope * x, mode="lines", name=label, hoverinfo="skip",
            line={"color": REFERENCE_COLORS[index % len(REFERENCE_COLORS)], "width": 3, "dash": "dash"},
        ))
    means = plot.get("ortalamalar")
    if means is not None:
        figure.add_trace(go.Scatter(
            x=means[op.x], y=means[op.y], mode="markers", name=op.means,
            marker={"color": PALETTE[3], "size": 11, "line": {"color": "white", "width": 1}},
            hovertemplate=f"{op.x_label}: %{{x}}<br>Ortalama: %{{y:.2f}}<extra></extra>",
        ))
    if plot["dogru"] is not None or plot.get("cizgiler") or means is not None:
        figure.update_layout(legend={"orientation": "h", "y": -0.2})
    else:
        figure.update_layout(showlegend=False)
    return style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label)


def _box(op: BoxPlot, boxes) -> go.Figure:
    """Hazır özetlerle yatay kutu grafiği: çeyrekler ders kuralıyla (Plotly'nin kendi çeyrek kuralı kullanılmaz).

    Kutular üretilen koddaki gibi 1, 2, … konumlarına çizilir (ilk seri altta); medyan kırmızı çizgidir.
    """

    figure = go.Figure()
    for position, (label, summary, outliers) in enumerate(boxes, start=1):
        figure.add_trace(
            go.Box(
                y=[position], q1=[summary["q1"]], median=[summary["medyan"]], q3=[summary["q3"]],
                lowerfence=[summary["alt_biyik"]], upperfence=[summary["ust_biyik"]], orientation="h", width=0.5,
                name=label, fillcolor="rgba(16, 124, 137, 0.2)", line={"color": PALETTE[0], "width": 2},
                showlegend=False, hovertemplate="%{x}<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[summary["medyan"]] * 2, y=[position - 0.25, position + 0.25], mode="lines", showlegend=False,
                line={"color": PALETTE[1], "width": 3}, hovertemplate=f"{label}: medyan %{{x}}<extra></extra>",
            )
        )
        if len(outliers):
            figure.add_trace(
                go.Scatter(
                    x=outliers, y=[position] * len(outliers), mode="markers", showlegend=False,
                    marker={"color": PALETTE[1], "size": 11}, hovertemplate="Aykırı değer adayı: %{x}<extra></extra>",
                )
            )
    figure.update_yaxes(tickvals=list(range(1, len(boxes) + 1)), ticktext=[label for label, _, _ in boxes],
                        range=[0.4, len(boxes) + 0.6], showgrid=False)
    return style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label)


def _histogram(op: Histogram, data: pd.DataFrame, state: LabState) -> go.Figure:
    """Üretilen kodla aynı kutular: [lower, upper] aralığında ``bins`` eşit genişlikte kutu."""

    edges = np.linspace(op.lower, op.upper, op.bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    figure = go.Figure()
    for index, (column, label) in enumerate(op.columns):
        counts, _ = np.histogram(data[column].to_numpy(dtype=float), bins=edges)
        figure.add_trace(
            go.Bar(
                x=centers, y=counts, width=np.diff(edges), name=label,
                marker={"color": PALETTE[index % len(PALETTE)], "opacity": 0.6, "line": {"width": 0}},
                hovertemplate=f"{label}<br>%{{x:.2f}} civarı: %{{y}} tekrar<extra></extra>",
            )
        )
    for index, (reference, label) in enumerate(op.references):
        value = float(state.scalars[reference]) if isinstance(reference, str) else float(reference)
        figure.add_trace(
            go.Scatter(
                x=[value, value], y=[0, 1], mode="lines", name=label, yaxis="y2", hoverinfo="skip",
                line={"color": REFERENCE_COLORS[index % len(REFERENCE_COLORS)], "width": 2.5, "dash": "dash"},
            )
        )
    if op.curves:  # beklenen sayı: gözlem sayısı × kutu genişliği × f(x)
        grid = np.linspace(op.lower, op.upper, 401)
        count = data[op.columns[0][0]].notna().sum()
        for index, (distribution, first, second, label) in enumerate(op.curves, start=len(op.columns)):
            values = density(distribution, parameter(first, state), parameter(second, state), grid)
            figure.add_trace(
                go.Scatter(
                    x=grid, y=count * (edges[1] - edges[0]) * values, mode="lines", name=label,
                    line={"color": PALETTE[index % len(PALETTE)], "width": 2.5},
                    hovertemplate=f"{label}<br>%{{x:.2f}}: %{{y:.1f}}<extra></extra>",
                )
            )
        figure.update_layout(legend={"orientation": "h", "y": -0.25})
    figure.update_layout(barmode="overlay", yaxis2={"overlaying": "y", "range": [0, 1], "visible": False})
    return style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label)


def _reference_lines(figure: go.Figure, references: list[tuple[float, str]]) -> None:
    """Dikey başvuru çizgileri; yardımcı eksende (0–1) tam boy çizilir ve açıklamada görünür."""

    for index, (value, label) in enumerate(references):
        figure.add_trace(
            go.Scatter(
                x=[value, value], y=[0, 1], mode="lines", name=f"{label}: {tr_number(value, 2)}", yaxis="y2",
                hoverinfo="skip",
                line={"color": REFERENCE_COLORS[index % len(REFERENCE_COLORS)], "width": 2.5,
                      "dash": ("dash", "dot", "dashdot")[index % 3]},
            )
        )
    if references:
        figure.update_layout(yaxis2={"overlaying": "y", "range": [0, 1], "visible": False})


def _class_histogram(op: ClassHistogram, data: pd.DataFrame) -> go.Figure:
    """Sınıflar sayısal eksende bitişik: dikdörtgenin genişliği sınıf genişliğidir."""

    lower, upper = data["alt"].to_numpy(dtype=float), data["ust"].to_numpy(dtype=float)
    values = data[op.y].to_numpy(dtype=float)
    labels = [f"{boundary_label(a)} ≤ x < {boundary_label(b)}" for a, b in zip(lower, upper)]
    shown = [tr_number(value, op.decimals, op.percent) for value in values]
    figure = go.Figure(
        go.Bar(
            x=(lower + upper) / 2, y=values, width=upper - lower, marker_color=PALETTE[0],
            marker_line={"color": "white", "width": 1}, customdata=labels, text=shown if op.labels else None,
            textposition="outside", cliponaxis=False, hovertext=shown,
            hovertemplate="%{customdata}: %{hovertext}<extra></extra>",
        )
    )
    edges = np.append(lower, upper[-1])
    figure.update_xaxes(tickvals=edges, ticktext=[boundary_label(edge) for edge in edges])
    figure.update_yaxes(rangemode="tozero")
    return style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label)


def _dot_plot(op: DotPlot, data: pd.DataFrame, state: LabState) -> go.Figure:
    figure = go.Figure(
        go.Scatter(
            x=data["deger"], y=data["yigin"], mode="markers", marker={"color": PALETTE[0], "size": 11},
            name="Gözlem", showlegend=False,
            hovertemplate="%{x}<extra></extra>",
        )
    )
    _reference_lines(figure, [(float(state.scalars[name]), label) for name, label in op.references])
    top = int(data["yigin"].max()) if len(data) else 1
    figure.update_layout(yaxis={"range": [0.3, top + 0.7], "tickvals": list(range(1, top + 1))},
                         legend={"orientation": "h", "y": -0.3})
    if op.x_range is not None:
        figure.update_xaxes(range=list(op.x_range))
    return style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label)


def _mosaic(op: MosaicChart, data: pd.DataFrame) -> go.Figure:
    """Sütun genişliği satırın marjinal payı, parça yüksekliği satır içindeki pay; parça alanı ortak olasılık."""

    values = data.to_numpy(dtype=float)
    widths = values.sum(axis=1) / values.sum()
    shares = values / values.sum(axis=1, keepdims=True)
    lefts = np.cumsum(widths) - widths
    centers = lefts + widths / 2
    rows = [str(item) for item in data.index]
    figure = go.Figure()
    bottom = np.zeros(len(rows))
    for index, column in reversed(list(enumerate(data.columns))):  # ilk sütun en üstte (notlardaki gibi)
        area = widths * shares[:, index]
        figure.add_trace(
            go.Bar(
                x=centers, y=shares[:, index], width=widths, base=bottom, name=str(column),
                marker={"color": PALETTE[index % len(PALETTE)], "line": {"color": "white", "width": 2}},
                text=[f"{column}<br>{tr_number(value, op.decimals)}" for value in area],
                textposition="inside", insidetextanchor="middle", textfont={"color": "white"},
                customdata=np.column_stack([rows, [tr_number(v, op.decimals) for v in area],
                                            [tr_number(v, op.decimals) for v in shares[:, index]]]),
                hovertemplate=(f"%{{customdata[0]}} ∩ {column}<br>alan (ortak olasılık): %{{customdata[1]}}"
                               "<br>sütun içindeki pay: %{customdata[2]}<extra></extra>"),
            )
        )
        bottom = bottom + shares[:, index]
    figure.update_layout(barmode="overlay", showlegend=False, bargap=0)
    figure.update_xaxes(range=[0, 1], tickvals=centers, ticktext=rows, showgrid=False)
    figure.update_yaxes(range=[0, 1], showgrid=False)
    return style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label)


def _heatmap(op: HeatMap, data: pd.DataFrame) -> go.Figure:
    """Beyazdan uygulamanın ana rengine koyulaşan hücreler; ilk satır en üstte (notlardaki tablo gibi)."""

    values = data.to_numpy(dtype=float)
    figure = go.Figure(
        go.Heatmap(
            z=values, x=[str(item) for item in data.columns], y=[str(item) for item in data.index],
            colorscale=[[0.0, HEAT_LOW], [1.0, PALETTE[0]]], zmin=0, zmax=float(values.max()), showscale=False,
            xgap=3, ygap=3, text=[[tr_number(value, op.decimals) for value in row] for row in values],
            texttemplate="%{text}", textfont={"size": 16},
            hovertemplate=f"{op.y_label}: %{{y}}<br>{op.x_label}: %{{x}}<br>değer: %{{text}}<extra></extra>",
        )
    )
    figure.update_xaxes(type="category", side="top", showgrid=False)
    figure.update_yaxes(type="category", autorange="reversed", showgrid=False)
    return style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label)


def _tree(op: TreeDiagram, data: pd.DataFrame) -> go.Figure:
    """Soldan sağa olasılık ağacı: dallarda (koşullu) olasılıklar, yol sonunda ortak olasılık."""

    firsts = data.drop_duplicates("ilk")
    root_y = float(firsts["y_ilk"].mean())
    figure = go.Figure()
    edges_x: list[float | None] = []
    edges_y: list[float | None] = []
    annotations = []

    def box(x: float, y: float, text: str) -> dict:
        return {"x": x, "y": y, "text": text, "showarrow": False, "bgcolor": "white", "bordercolor": PALETTE[0],
                "borderwidth": 1.5, "borderpad": 4, "font": {"size": 14}}

    def edge_label(x0: float, y0: float, x1: float, y1: float, value: float) -> dict:
        return {"x": (x0 + x1) / 2, "y": (y0 + y1) / 2, "text": tr_number(value, 2), "showarrow": False,
                "yshift": 11, "font": {"size": 13, "color": REFERENCE_COLORS[0]}}

    for _, row in firsts.iterrows():
        edges_x += [0, 1, None]
        edges_y += [root_y, row["y_ilk"], None]
        annotations += [edge_label(0, root_y, 1, row["y_ilk"], row["p_ilk"]), box(1, row["y_ilk"], row["ilk"])]
    for _, row in data.iterrows():
        edges_x += [1, 2, None]
        edges_y += [row["y_ilk"], row["y_yol"], None]
        annotations += [
            edge_label(1, row["y_ilk"], 2, row["y_yol"], row["p_ikinci"]),
            box(2, row["y_yol"], row["ikinci"]),
            {"x": 2.62, "y": row["y_yol"], "text": f"ortak {tr_number(row['ortak'], op.decimals)}",
             "showarrow": False, "font": {"size": 13, "color": PALETTE[1]}},
        ]
    annotations.append(box(0, root_y, op.root))
    figure.add_trace(go.Scatter(x=edges_x, y=edges_y, mode="lines", line={"color": PALETTE[0], "width": 2},
                                hoverinfo="skip", showlegend=False))
    figure.update_layout(annotations=annotations, height=140 + 70 * len(data))
    figure.update_xaxes(visible=False, range=[-0.35, 3.05])
    figure.update_yaxes(visible=False, range=[-0.6, len(data) - 0.4])
    return style_figure(figure, title=op.title, x_title="", y_title="")


def _rgba(color: str, alpha: float) -> str:
    red, green, blue = (int(color[index:index + 2], 16) for index in (1, 3, 5))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _density(op: DensityPlot, data: pd.DataFrame) -> go.Figure:
    """Yoğunluk eğrisi; boyalı aralıkların alanı olasılıktır (üretilen koddaki gibi her aralık ayrı ızgarayla)."""

    figure = go.Figure()
    for low, high in op.shade:
        x = np.linspace(low, high, 200)
        figure.add_trace(
            go.Scatter(
                x=np.r_[low, x, high], y=np.r_[0.0, density(op.distribution, op.first, op.second, x), 0.0],
                fill="toself", mode="lines", line={"width": 0}, fillcolor=_rgba(PALETTE[0], 0.3),
                name=f"{tr_number(low, 2).rstrip('0').rstrip(',')}–{tr_number(high, 2).rstrip('0').rstrip(',')}",
                hoverinfo="skip", showlegend=False,
            )
        )
    figure.add_trace(
        go.Scatter(
            x=data["x"], y=data["f"], mode="lines", line={"color": PALETTE[0], "width": 2.5}, name="f(x)",
            showlegend=False, hovertemplate="x = %{x:.2f}<br>f(x) = %{y:.4f}<extra></extra>",
        )
    )
    for index, (value, label) in enumerate(op.references):  # dikey çizgiler; etiket olduğu gibi açıklamada
        figure.add_trace(
            go.Scatter(
                x=[value, value], y=[0, 1], mode="lines", name=label, yaxis="y2", hoverinfo="skip",
                line={"color": REFERENCE_COLORS[index % len(REFERENCE_COLORS)], "width": 2.5,
                      "dash": ("dash", "dot", "dashdot")[index % 3]},
            )
        )
    if op.references:
        figure.update_layout(yaxis2={"overlaying": "y", "range": [0, 1], "visible": False},
                             legend={"orientation": "h", "y": -0.25})
    figure.update_xaxes(range=list(op.x_range))
    if op.y_max is None:
        figure.update_yaxes(rangemode="tozero")
    else:
        figure.update_yaxes(range=[0, op.y_max])
    return style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label)


def _density_compare(op: DensityCompare, data: pd.DataFrame) -> go.Figure:
    """Aynı eksende birden fazla yoğunluk; renkler üretilen koddaki sırayla."""

    figure = go.Figure()
    for index, (_, _, _, label) in enumerate(op.curves):
        figure.add_trace(
            go.Scatter(
                x=data["x"], y=data[f"f{index + 1}"], mode="lines", name=label,
                line={"color": PALETTE[index % len(PALETTE)], "width": 2.5},
                hovertemplate=f"{label}<br>x = %{{x:.2f}}<br>f(x) = %{{y:.4f}}<extra></extra>",
            )
        )
    figure.update_xaxes(range=list(op.x_range))
    if op.y_max is None:
        figure.update_yaxes(rangemode="tozero")
    else:
        figure.update_yaxes(range=[0, op.y_max])
    figure.update_layout(legend={"orientation": "h", "y": -0.25})
    return style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label)


def _pmf_density(op: PmfWithDensity, data: dict) -> go.Figure:
    """Kesikli olasılıklar çubuk, sürekli yaklaşım eğri; boyalı alan süreklilik düzeltmesinin aralığıdır."""

    bars, curve = data["cubuk"], data["egri"]
    figure = go.Figure(
        go.Bar(
            x=bars[op.x], y=bars[op.y], width=0.6, name=op.bar_label, marker={"color": _rgba(PALETTE[0], 0.8)},
            hovertemplate="x = %{x}<br>P(X = x) = %{y:.4f}<extra></extra>",
        )
    )
    for low, high, values in data["alan"]:
        x = np.linspace(low, high, 200)
        figure.add_trace(
            go.Scatter(
                x=np.r_[low, x, high], y=np.r_[0.0, values, 0.0], fill="toself", mode="lines", line={"width": 0},
                fillcolor=_rgba(PALETTE[1], 0.3), hoverinfo="skip", showlegend=False,
            )
        )
    figure.add_trace(
        go.Scatter(
            x=curve["x"], y=curve["f"], mode="lines", name=op.curve_label,
            line={"color": PALETTE[1], "width": 2.5}, hovertemplate="x = %{x:.2f}<br>f(x) = %{y:.4f}<extra></extra>",
        )
    )
    figure.update_yaxes(rangemode="tozero")
    figure.update_layout(legend={"orientation": "h", "y": -0.25}, bargap=0)
    return style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label)


def _vertical(figure: go.Figure, value: float, name: str, color: str, dash: str, *, legend: bool = True) -> None:
    """Tam boy dikey çizgi (yardımcı eksende 0–1); açıklamada ``name`` ile görünür."""

    figure.add_trace(go.Scatter(
        x=[value, value], y=[0, 1], mode="lines", name=name, yaxis="y2", hoverinfo="skip", showlegend=legend,
        line={"color": color, "width": 2.5, "dash": dash},
    ))


def _area(figure: go.Figure, x: np.ndarray, y: np.ndarray, *, name: str, color: str, legend: bool,
          pattern: bool = False) -> None:
    figure.add_trace(go.Scatter(
        x=np.r_[x[0], x, x[-1]], y=np.r_[0.0, y, 0.0], fill="toself", mode="lines", line={"width": 0},
        fillcolor=_rgba(color, 0.18 if pattern else 0.3), name=name, showlegend=legend, hoverinfo="skip",
        fillpattern={"shape": "/", "fgcolor": color, "solidity": 0.35} if pattern else None,
    ))


def hypothesis_labels(op: HypothesisPlot, data: dict) -> dict[str, str]:
    """Test grafiğinin açıklama metinleri (üretilen kodla aynı): reddetme bölgesi, p alanı, kritik ve gözlenen değer."""

    symbol = "F" if op.distribution == "f" else "t"
    level = tr_number(100 * op.alpha, 0 if float(100 * op.alpha).is_integer() else 1)
    critical = data["kritik"]
    if op.distribution == "t" and op.alternative == "iki":
        shown = f"±{tr_number(critical[1], 2)}"
    else:
        shown = tr_number(critical[0], 2)
    observed = f"Gözlenen {symbol} = {tr_number(data['gozlenen'], 2)}"
    if data["disarida"]:
        observed += " (eksenin dışında)"
    return {
        "reddetme": f"Yüzde {level} reddetme bölgesi",
        "p": "p-değeri alanı",
        "kritik": f"Kritik değer {shown}",
        "gozlenen": observed,
    }


def p_text(value: float, decimals: int = 3) -> str:
    """p-değerinin ekran yazımı: gösterim basamağında 0'a yuvarlanıyorsa "< 0,001", 1'e yuvarlanıyorsa "> 0,999"."""

    if value < 0.5 * 10 ** (-decimals):
        return "< " + tr_number(10 ** (-decimals), decimals)
    if value > 1 - 0.5 * 10 ** (-decimals):
        return "> " + tr_number(1 - 10 ** (-decimals), decimals)
    return tr_number(value, decimals)


def _hypothesis(op: HypothesisPlot, data: dict) -> go.Figure:
    """Test dağılımı: reddetme bölgesi boyalı, p-değeri alanı taralı; kritik değer kesikli, gözlenen değer düz çizgi."""

    labels = hypothesis_labels(op, data)
    df, df2 = data["sd"]
    figure = go.Figure()
    for index, (low, high) in enumerate(data["reddetme"]):
        x = np.linspace(low, high, 200)
        _area(figure, x, I.density(op.distribution, x, df, df2), name=labels["reddetme"], color=PALETTE[1],
              legend=index == 0)
    for index, (low, high) in enumerate(data["p_alani"]):
        x = np.linspace(low, high, 200)
        _area(figure, x, I.density(op.distribution, x, df, df2), name=labels["p"], color=REFERENCE_COLORS[0],
              legend=index == 0, pattern=True)
    figure.add_trace(go.Scatter(
        x=data["x"], y=data["f"], mode="lines", name="Yoğunluk", showlegend=False,
        line={"color": PALETTE[0], "width": 2.5}, hovertemplate="%{x:.2f}: %{y:.4f}<extra></extra>",
    ))
    for index, value in enumerate(data["kritik"]):
        _vertical(figure, value, labels["kritik"], PALETTE[1], "dash", legend=index == 0)
    if not data["disarida"]:
        _vertical(figure, data["gozlenen"], labels["gozlenen"], REFERENCE_COLORS[0], "solid")
    else:  # gözlenen değer eksende değil: açıklamada yazılır
        figure.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=labels["gozlenen"],
                                    line={"color": REFERENCE_COLORS[0], "width": 2.5}))
    figure.update_xaxes(range=list(data["sinirlar"]))
    figure.update_layout(yaxis={"rangemode": "tozero"}, yaxis2={"overlaying": "y", "range": [0, 1], "visible": False},
                         legend={"orientation": "h", "y": -0.25})
    return style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label)


def _coefficients(op: CoefficientPlot, data: pd.DataFrame, label) -> go.Figure:
    """Katsayılar ve güven aralıkları; ilk terim en üstte, sıfırda kesikli dikey çizgi."""

    names = [label(term) for term in data["terim"]]
    level = tr_number(100 * op.level, 0)
    figure = go.Figure(go.Scatter(
        x=data["tahmin"], y=names, mode="markers", name=f"Tahmin ve yüzde {level} güven aralığı",
        marker={"color": PALETTE[0], "size": 10},
        error_x={"type": "data", "symmetric": False, "array": data["ust"] - data["tahmin"],
                 "arrayminus": data["tahmin"] - data["alt"], "color": PALETTE[0], "thickness": 2.5, "width": 8},
        customdata=np.column_stack([data["alt"], data["ust"]]),
        hovertemplate="%{y}: %{x:.3f}<br>aralık [%{customdata[0]:.3f}; %{customdata[1]:.3f}]<extra></extra>",
    ))
    figure.add_vline(x=0, line={"color": REFERENCE_COLORS[0], "width": 2, "dash": "dash"})
    figure.update_yaxes(categoryorder="array", categoryarray=names[::-1])
    figure.update_layout(showlegend=False)
    return style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label)


def _intervals(op: IntervalPlot, data: pd.DataFrame, truth: float) -> go.Figure:
    """Tekrarların güven aralıkları: gerçek değeri kapsayanlar düz, kapsamayanlar kesikli ve ikinci renkte."""

    figure = go.Figure()
    for covered, name, color, dash in ((True, "Gerçek değeri kapsıyor", PALETTE[0], "solid"),
                                       (False, "Gerçek değeri kapsamıyor", PALETTE[1], "dash")):
        rows = data[data["kapsiyor"] == covered]
        if rows.empty:
            continue
        x = np.column_stack([rows["alt"], rows["ust"], np.full(len(rows), np.nan)]).ravel()
        y = np.column_stack([rows["tekrar"], rows["tekrar"], np.full(len(rows), np.nan)]).ravel()
        figure.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line={"color": color, "width": 2.5,
                                                                              "dash": dash}, hoverinfo="skip"))
        figure.add_trace(go.Scatter(
            x=rows["tahmin"], y=rows["tekrar"], mode="markers", showlegend=False, marker={"color": color, "size": 7},
            hovertemplate="Tekrar %{y}: tahmin %{x:.3f}<extra></extra>",
        ))
    figure.add_vline(x=truth, line={"color": REFERENCE_COLORS[0], "width": 2.5, "dash": "dash"})
    figure.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"Gerçek değer = {tr_number(truth, 2)}",
                                line={"color": REFERENCE_COLORS[0], "width": 2.5, "dash": "dash"}))
    if op.reference is not None:
        value, name = op.reference
        figure.add_vline(x=value, line={"color": REFERENCE_COLORS[1], "width": 2.5, "dash": "dot"})
        figure.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=name,
                                    line={"color": REFERENCE_COLORS[1], "width": 2.5, "dash": "dot"}))
        # Başvuru çizgisi aralıkların dışında kalsa da eksende görünür.
        low = min(float(data["alt"].min()), truth, value)
        high = max(float(data["ust"].max()), truth, value)
        pad = 0.05 * (high - low or 1.0)
        figure.update_xaxes(range=[low - pad, high + pad])
    figure.update_layout(legend={"orientation": "h", "y": -0.25})
    return style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label)


def figure_for(op: Operation, state: LabState, label=lambda name: name) -> go.Figure:
    """Bir grafik işleminin Plotly karşılığı; ``label`` seri başlıkları için Türkçe ad verir."""

    data = state.plots[plot_key(op)]
    if isinstance(op, BarChart):
        return _bar(op, data)
    if isinstance(op, GroupedBarChart):
        table = state.tables[op.table]
        legend = label(table.index.name if op.series == "satir" else table.columns.name)
        return _series_bars(data, stacked=op.stacked, decimals=op.decimals, title=op.title, x_title=op.x_label,
                            y_title=op.y_label, legend_title=legend or "", labels=op.labels)
    if isinstance(op, CompareBarChart):
        first = state.tables[op.tables[0][1]]
        return _series_bars(data, stacked=False, decimals=op.decimals, title=op.title, x_title=op.x_label,
                            y_title=op.y_label, legend_title=label(first.index.name) or "")
    if isinstance(op, PieChart):
        return _pie(op, data)
    if isinstance(op, LineChart):
        return _line(op, data, state)
    if isinstance(op, ScatterPlot):
        return _scatter(op, data)
    if isinstance(op, BoxPlot):
        return _box(op, data)
    if isinstance(op, Histogram):
        return _histogram(op, data, state)
    if isinstance(op, ClassHistogram):
        return _class_histogram(op, data)
    if isinstance(op, DotPlot):
        return _dot_plot(op, data, state)
    if isinstance(op, MosaicChart):
        return _mosaic(op, data)
    if isinstance(op, HeatMap):
        return _heatmap(op, data)
    if isinstance(op, TreeDiagram):
        return _tree(op, data)
    if isinstance(op, DensityPlot):
        return _density(op, data)
    if isinstance(op, DensityCompare):
        return _density_compare(op, data)
    if isinstance(op, PmfWithDensity):
        return _pmf_density(op, data)
    if isinstance(op, HypothesisPlot):
        return _hypothesis(op, data)
    if isinstance(op, CoefficientPlot):
        return _coefficients(op, data, label)
    if isinstance(op, IntervalPlot):
        return _intervals(op, data, parameter(op.truth, state))
    raise TypeError(f"Grafik türü tanınmıyor: {type(op).__name__}")
