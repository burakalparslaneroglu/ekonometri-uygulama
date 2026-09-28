"""Tanım şeması, tablo hesapları, kategorik çekiliş ve kod üreticisi yardımcıları."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.codegen.base import render_step, text, wrapped
from core.labs import expr as E
from core.labs import tables as T
from core.labs.runner import LabState, execute, run_operations
from core.labs.spec import (
    TOTAL,
    BoxPlot,
    Check,
    CrossTab,
    DotPlot,
    DrawCategory,
    FrequencyTable,
    FromCounts,
    Histogram,
    InlineData,
    JoinColumns,
    LineChart,
    NewSample,
    Scalar,
    ShowFrame,
    Statistic,
)


def test_frequency_table_follows_the_given_order_and_adds_totals() -> None:
    values = pd.Series(["b", "a", "b", "c", "b"], name="x")
    table = T.frequency_table(values, ("c", "b", "a"), relative=True, totals=True)
    assert list(table.index) == ["c", "b", "a", TOTAL]
    assert list(table["frekans"]) == [1, 3, 1, 5]
    assert table.loc["b", "goreli"] == pytest.approx(0.6)
    assert table.loc[TOTAL, "yuzde"] == pytest.approx(100.0)


def test_frequency_table_rejects_categories_missing_from_the_order() -> None:
    with pytest.raises(ValueError, match="Sırada olmayan"):
        T.frequency_table(pd.Series(["a", "z"]), ("a",))


def test_zero_frequency_categories_are_kept() -> None:
    table = T.frequency_table(pd.Series(["a", "a"]), ("a", "b"), relative=False)
    assert list(table["frekans"]) == [2, 0]


def test_from_counts_repeats_each_row_its_count_times() -> None:
    frame = T.from_counts(("g", "h"), (("x", "p", 2), ("y", "q", 0), ("y", "p", 3)))
    assert len(frame) == 5
    assert frame.groupby(["g", "h"]).size().to_dict() == {("x", "p"): 2, ("y", "p"): 3}


def test_crosstab_denominators() -> None:
    frame = T.from_counts(("r", "c"), (("A", "u", 1), ("A", "v", 3), ("B", "u", 2), ("B", "v", 2)))
    counts = T.crosstab(frame, "r", "c", ("A", "B"), ("u", "v"), margins=True)
    assert counts.loc[TOTAL, TOTAL] == 8 and counts.loc["A", TOTAL] == 4 and counts.loc[TOTAL, "u"] == 3
    rows = T.crosstab(frame, "r", "c", ("A", "B"), ("u", "v"), percent="satir", margins=True)
    assert rows.loc["A", "v"] == pytest.approx(75.0) and rows.loc["B", TOTAL] == pytest.approx(100.0)
    columns = T.crosstab(frame, "r", "c", ("A", "B"), ("u", "v"), percent="sutun", margins=True)
    assert columns.loc["B", "u"] == pytest.approx(200 / 3) and columns.loc[TOTAL, "v"] == pytest.approx(100.0)


def test_class_table_puts_boundary_values_in_the_upper_class_and_adds_totals() -> None:
    values = pd.Series([10.0, 19.0, 20.0, 29.0, 30.0])
    edges = T.class_edges(values, 10)
    assert list(edges) == [10.0, 20.0, 30.0, 40.0]
    table = T.class_table(values, edges, ("frekans", "yuzde", "kumulatif_frekans"), totals=True)
    assert list(table.index) == ["10 ≤ x < 20", "20 ≤ x < 30", "30 ≤ x < 40", TOTAL]
    assert list(table["frekans"]) == [2, 2, 1, 5]
    assert table.loc[TOTAL, "yuzde"] == pytest.approx(100.0)
    assert np.isnan(table.loc[TOTAL, "kumulatif_frekans"])
    cumulative = T.class_table(values, edges, ("kumulatif_yuzde",), row_labels="ust")
    assert list(cumulative.index) == ["x < 20", "x < 30", "x < 40"]
    assert list(cumulative["kumulatif_yuzde"]) == pytest.approx([40.0, 80.0, 100.0])


def test_class_table_rejects_unknown_columns_and_needs_a_class_count_with_a_lower_edge() -> None:
    with pytest.raises(ValueError, match="Tanınmayan"):
        T.class_table(pd.Series([1.0]), np.array([0.0, 10.0]), ("medyan",))
    with pytest.raises(ValueError, match="sınıf sayısı"):
        T.class_edges(pd.Series([1.0]), 10, lower=0)


def test_boundary_labels_use_a_decimal_comma_without_padding() -> None:
    assert T.boundary_label(12.5) == "12,5" and T.boundary_label(20.0) == "20"


def test_stem_leaf_keeps_empty_stems() -> None:
    stems = T.stem_leaf(pd.Series([12.0, 15.0, 31.0]))
    assert list(stems.index) == ["1", "2", "3"]
    assert list(stems["yapraklar"]) == ["2 5", "", "1"]
    assert list(stems["yaprak_sayisi"]) == [2, 0, 1]
    with pytest.raises(ValueError):
        T.stem_leaf(pd.Series([1.5]))


def test_percentile_locations_of_both_rules() -> None:
    assert T.percentile_location(12, 60) == pytest.approx(7.8)
    assert T.percentile_location(12, 60, "yazilim") == pytest.approx(7.6)
    with pytest.raises(ValueError):
        T.percentile_location(12, 60, "tip9")


def test_dot_plot_axis_range_is_shared_by_all_three_renderers() -> None:
    from core.charts import figure_for
    from core.labs.spec import LabSpec, LabStep, NoteRef

    spread = (5, 55)
    operations = (
        InlineData("a", ("x",), ((20.0,), (30.0,), (30.0,), (40.0,)), "Dar yayılım"),
        InlineData("b", ("x",), ((10.0,), (30.0,), (50.0,)), "Geniş yayılım"),
        DotPlot("a", "x", "Değer", "A", x_range=spread),
        DotPlot("b", "x", "Değer", "B", x_range=spread),
    )
    spec = LabSpec("konu01", "t", "1", (LabStep(1, "t", NoteRef("1.1"), "t", operations=operations),))
    state = run_operations(operations)
    assert all(tuple(figure_for(op, state).layout.xaxis.range) == spread for op in operations[2:])
    assert render_step(spec, 1, "Python").count("ax.set_xlim(5, 55)") == 2
    assert render_step(spec, 1, "R").count("xlim = c(5, 55)") == 2


def test_dot_plot_axis_range_must_cover_the_data() -> None:
    state = LabState()
    execute(FromCounts("f", ("x",), ((10.0, 1), (50.0, 1)), "iki gözlem"), state)
    with pytest.raises(ValueError, match="kapsamıyor"):
        execute(DotPlot("f", "x", "Değer", "Dar eksen", x_range=(20, 60)), state)


def test_thresholds_end_exactly_at_one_and_validate() -> None:
    edges = T.thresholds((0.1, 0.2, 0.7))
    assert edges[-1] == 1.0
    with pytest.raises(ValueError):
        T.thresholds((0.5, 0.6))


def test_category_draw_is_the_first_category_whose_cumulative_probability_exceeds_u() -> None:
    frame = pd.DataFrame({"g": ["a"] * 5})
    u = np.array([0.0, 0.0999, 0.1, 0.2999, 0.9999])
    drawn = T.draw_categories(frame, u, ("x", "y", "z"), (((), (0.1, 0.2, 0.7)),), ())
    assert list(drawn) == ["x", "x", "y", "y", "z"]


def test_conditional_category_draws_follow_their_own_probabilities() -> None:
    state = LabState()
    for op in (
        NewSample("s", 20000, 1),
        DrawCategory("s", "g", ("a", "b"), (((), (0.5, 0.5)),), "grup"),
        DrawCategory("s", "y", ("1", "0"), ((("a",), (0.9, 0.1)), (("b",), (0.2, 0.8))), "sonuç", by=("g",)),
    ):
        execute(op, state)
    frame = state.frames["s"]
    shares = frame.groupby("g")["y"].apply(lambda values: (values == "1").mean())
    assert shares["a"] == pytest.approx(0.9, abs=0.01) and shares["b"] == pytest.approx(0.2, abs=0.01)


def test_run_operations_matches_hand_counts() -> None:
    state = run_operations((
        FromCounts("f", ("k",), (("x", 3), ("y", 1)), "sayım"),
        FrequencyTable("f", "k", "t", ("x", "y"), totals=True),
        CrossTab("f", "k", "k", "c", ("x", "y"), ("x", "y"), margins=True),
    ))
    assert state.tables["t"].loc["x", "yuzde"] == pytest.approx(75.0)
    assert state.tables["c"].loc["x", "y"] == 0


def test_check_tolerance_is_half_the_last_printed_digit() -> None:
    assert Check("a", None, 0.625, 3).tolerance == pytest.approx(0.0005, abs=1e-11)
    assert Check("a", None, 64, 0).tolerance == 0.5


def test_expression_rendering_uses_minimal_parentheses() -> None:
    expression = E.mul(100, E.div(E.sub(E.ref("b"), E.ref("a")), E.ref("a")))
    dialect = E.Dialect(variable=lambda name: name, functions={}, power="^")
    assert E.render(expression, dialect) == "100 * (b - a) / a"
    assert E.evaluate(expression, scalar={"a": 72.0, "b": 78.0}.__getitem__) == pytest.approx(8.3333333)


def test_string_literals_are_escaped_for_both_languages() -> None:
    assert text('a "b" \\ c') == '"a \\"b\\" \\\\ c"'
    assert text(0.125) == "0.125" and text(3.0) == "3"


def test_wrapped_lists_keep_the_requested_items_per_line() -> None:
    lines = wrapped("x = [", [str(i) for i in range(7)], "]", per_line=3)
    assert lines == ["x = [", "    0, 1, 2,", "    3, 4, 5,", "    6", "]"]


# --- Konu 5–6 ile eklenen işlemler ----------------------------------------------------

def test_counting_functions_and_new_statistics() -> None:
    assert E.evaluate(E.comb(5, 2)) == 10 and E.evaluate(E.perm(5, 2)) == 20 and E.evaluate(E.factorial(5)) == 120
    frame = pd.DataFrame({"g": [0.0, 1.0, 0.0, 1.0]})
    assert list(E.evaluate(E.cummean(E.var("g")), frame)) == [0.0, 0.5, 1 / 3, 0.5]
    assert list(E.evaluate(E.seq(E.var("g")), frame)) == [1.0, 2.0, 3.0, 4.0]
    state = run_operations((
        InlineData("d", ("x",), ((4,), (6,), (8,), (10,), (12,)), "Tablo 5.1"),
        Statistic("d", "x", "var", "s2", "s²"),
        Statistic("d", "x", "nunique", "k", "farklı değer"),
    ))
    assert state.scalars["s2"] == 10 and state.scalars["k"] == 5
    python, r = render_step(_one_step(E.comb(5, 2)), 1, "Python"), render_step(_one_step(E.comb(5, 2)), 1, "R")
    assert "math.comb(5, 2)" in python and "import math" in python and "choose(5, 2)" in r


def _one_step(expression):
    from core.labs.spec import LabSpec, LabStep, NoteRef

    step = LabStep(1, "t", NoteRef("6.4"), "t", operations=(Scalar("c", expression, "C"),))
    return LabSpec("konu06", "t", "6", (step,))


def test_join_columns_keeps_the_first_table_order() -> None:
    state = run_operations((
        InlineData("d", ("x",), (("b",), ("a",), ("b",)), "veri"),
        FrequencyTable("d", "x", "f1", ("b", "a")),
        FrequencyTable("d", "x", "f2", ("b", "a"), relative=False),
        JoinColumns("j", (("oran", "f1", "goreli"), ("sayi", "f2", "frekans"))),
    ))
    joined = state.tables["j"]
    assert list(joined.index) == ["b", "a"] and list(joined["sayi"]) == [2, 1]
    assert joined.loc["a", "oran"] == pytest.approx(1 / 3)


def test_show_frame_requires_existing_columns() -> None:
    state = run_operations((InlineData("d", ("x",), ((1,),), "veri"),))
    execute(ShowFrame("d", ("x",), "göster"), state)
    with pytest.raises(ValueError, match="gösterilecek sütun yok"):
        execute(ShowFrame("d", ("x", "y"), "göster"), state)


def test_new_charts_use_state_values_and_course_quartiles() -> None:
    from core.charts import figure_for

    operations = (
        InlineData("d", ("k", "y"), tuple((k, 0.1 * (k % 3)) for k in range(1, 61)), "seri"),
        Scalar("p", E.const(0.1), "p"),
        Statistic("d", "y", "mean", "m", "ortalama"),
        LineChart("d", "k", "y", "k", "oran", "Uzun seri", references=(("p", "Gerçek p"),), markers=False),
        Histogram("d", (("y", "Değer"),), 5, 0, 0.5, "Histogram", "Değer", references=(("m", "Ortalama"),),
                  y_label="Gözlem sayısı"),
        BoxPlot((("d", "y", "Seri"),), "Değer", "Veri seti", "Kutu"),
    )
    state = run_operations(operations)
    line, histogram, box = (figure_for(op, state) for op in operations[3:])
    assert line.data[0].mode == "lines" and list(line.data[1].y) == [0.1, 0.1]
    assert line.layout.xaxis.dtick is None  # 60 farklı değer: her değere işaret konmaz
    assert histogram.data[-1].x[0] == pytest.approx(state.scalars["m"])
    assert histogram.layout.yaxis.title.text == "Gözlem sayısı"
    summary = T.box_summary(state.frames["d"]["y"])
    assert (box.data[0].q1[0], box.data[0].median[0], box.data[0].q3[0]) == (summary["q1"], summary["medyan"],
                                                                                summary["q3"])


# --- Konu 9–10 ile eklenen işlemler ---------------------------------------------------

def test_distribution_functions_agree_with_the_formulas() -> None:
    from math import comb, exp, factorial, pi, sqrt

    assert E.evaluate(E.dbinom(2, 8, 0.25)) == pytest.approx(comb(8, 2) * 0.25 ** 2 * 0.75 ** 6)
    assert E.evaluate(E.pbinom(0, 8, 0.25)) == pytest.approx(0.75 ** 8)
    assert E.evaluate(E.dpois(2, 3)) == pytest.approx(exp(-3) * 3 ** 2 / factorial(2))
    assert E.evaluate(E.ppois(1, 3)) == pytest.approx(4 * exp(-3))
    assert E.evaluate(E.dhyper(1, 20, 5, 4)) == pytest.approx(comb(5, 1) * comb(15, 3) / comb(20, 4))
    assert E.evaluate(E.phyper(0, 20, 5, 4)) == pytest.approx(comb(15, 4) / comb(20, 4))
    assert E.evaluate(E.dnorm(70, 70, 10)) == pytest.approx(1 / (10 * sqrt(2 * pi)))
    with pytest.raises(ValueError):
        E.validate(E.Call("dbinom", (E.const(1), E.const(2))))


def test_unary_minus_is_parenthesised_only_inside_operations() -> None:
    dialect = E.Dialect(variable=lambda name: name, functions={"exp": "exp"}, power="^")
    assert E.render(E.exp(E.neg(E.ref("lam"))), dialect) == "exp(-lam)"
    assert E.render(E.add(70, E.mul(E.neg(2), 10)), dialect) == "70 + (-2) * 10"
    assert E.render(E.neg(E.sub(E.ref("a"), E.ref("b"))), dialect) == "-(a - b)"
    assert E.render(E.power(E.neg(E.ref("a")), 2), dialect) == "(-a) ^ 2"
    assert E.render(E.sub(E.ref("b"), -2), dialect) == "b - -2"  # sabit eksi değerlerin yazımı değişmedi
    assert E.evaluate(E.add(70, E.mul(E.neg(2), 10))) == 50


def test_distribution_functions_in_both_languages() -> None:
    step = _one_step(E.add(E.dhyper(1, 20, 5, 4), E.dbinom(2, 8, 0.25)))
    python, r = render_step(step, 1, "Python"), render_step(step, 1, "R")
    assert "stats.hypergeom.pmf(1, 20, 5, 4) + stats.binom.pmf(2, 8, 0.25)" in python
    assert "from scipy import stats" in python
    assert "dhyper(1, 5, 20 - 5, 4) + dbinom(2, 8, 0.25)" in r  # R: dhyper(x, r, N − r, n)


def test_support_row_sum_and_rectangles() -> None:
    from core.labs.spec import Derive, Outcomes, Rectangles, RowSum, Support

    state = run_operations((
        Support("s", "x", 0, 3, "olası değerler"),
        Outcomes("d", (("y1", (0, 1)), ("y2", (0, 1)), ("y3", (0, 1))), "diziler"),
        RowSum("d", "x", ("y1", "y2", "y3"), "başarı sayısı"),
        Rectangles("r", "m", 10, 11, 0.25, "dikdörtgenler"),
        Rectangles("q", "m", 10, 11, 0.01, "ince dikdörtgenler"),
        Derive("q", "alan", E.mul(E.dnorm(E.var("m"), 10.5, 1), 0.01), "yükseklik × genişlik"),
    ))
    assert list(state.frames["s"]["x"]) == [0, 1, 2, 3]
    assert state.frames["d"]["x"].value_counts().sort_index().tolist() == [1, 3, 3, 1]
    assert list(state.frames["r"]["m"]) == pytest.approx([10.125, 10.375, 10.625, 10.875])
    assert state.frames["q"]["alan"].sum() == pytest.approx(0.382925, abs=2e-5)  # P(|Z| < 0,5)
    with pytest.raises(ValueError, match="tam katı"):
        Rectangles("r", "m", 0, 1, 0.3, "hatalı").count
    with pytest.raises(ValueError):
        T.support(2, 1)


def test_count_draws_follow_their_distributions_and_the_generated_call() -> None:
    from core.labs.spec import DrawCount

    rng = np.random.default_rng(305)
    binomial = T.draw_count(rng, "binomial", (10, 0.2), 20000)
    poisson = T.draw_count(rng, "poisson", (3,), 20000)
    hyper = T.draw_count(rng, "hypergeometric", (40, 4, 8), 20000)
    assert binomial.mean() == pytest.approx(2.0, abs=0.05) and binomial.var() == pytest.approx(1.6, abs=0.06)
    assert poisson.mean() == pytest.approx(3.0, abs=0.05) and poisson.var() == pytest.approx(3.0, abs=0.1)
    assert hyper.mean() == pytest.approx(0.8, abs=0.03) and hyper.max() <= 4
    with pytest.raises(ValueError, match="parametre"):
        T.draw_count(rng, "poisson", (3, 1), 5)
    # Uygulamadaki çekiliş, üretilen Python koduyla aynı çağrıdır: aynı tohum, aynı sayılar
    state = run_operations((NewSample("f", 5, 305), DrawCount("f", "x", "hypergeometric", (40, 4, 8), "çekiliş")))
    expected = np.random.default_rng(305).hypergeometric(ngood=4, nbad=36, nsample=8, size=5)
    assert list(state.frames["f"]["x"]) == list(expected.astype(float))


def test_density_plot_grid_codegen_and_figure() -> None:
    from core.charts import figure_for
    from core.codegen.base import render_script
    from core.labs.spec import DensityPlot, LabSpec, LabStep, NoteRef

    uniform = DensityPlot("uniform", 120, 140, (115, 145), "U(120, 140)", "Süre (dakika)", shade=((128, 136),))
    normal = DensityPlot("normal", 70, 10, (30, 110), "N(70, 10²)", "Puan", references=((85, "x = 85"),))
    state = run_operations((uniform, normal))
    grid = T.density_grid("uniform", 120, 140, (115, 145))
    assert grid["f"].max() == pytest.approx(0.05) and grid["f"].iloc[0] == 0
    figure = figure_for(uniform, state)
    assert any(trace.fill == "toself" for trace in figure.data)
    step = LabStep(1, "t", NoteRef("10.2"), "t", operations=(uniform, normal))
    spec = LabSpec("konu10", "t", "10", (step,))
    python, r = render_script(spec, "Python"), render_script(spec, "R")
    assert "stats.uniform.pdf(eksen, 120, 20)" in python and "stats.norm.pdf(eksen, 70, 10)" in python
    assert "dunif(" in r and "dnorm(" in r and "polygon(" in r and "abline(v = 85" in r
    with pytest.raises(ValueError):
        T.density("uniform", 2, 1, 0.5)


def test_numeric_group_summary_keeps_the_numeric_order_in_r() -> None:
    from core.labs.spec import GroupSummary

    state = run_operations((
        InlineData("d", ("x", "p"), ((1.0, 0.2), (0.0, 0.3), (1.0, 0.1), (2.0, 0.4)), "veri"),
        GroupSummary("d", "x", (("sayi", "p", "count"), ("toplam", "p", "sum")), "g", (0, 1, 2), decimals=4),
    ))
    table = state.tables["g"]
    assert list(table.index) == [0, 1, 2] and list(table["toplam"]) == pytest.approx([0.3, 0.3, 0.4])
    from core.labs.spec import LabSpec, LabStep, NoteRef

    step = LabStep(1, "t", NoteRef("9.3"), "t", operations=(
        InlineData("d", ("x", "p"), ((1.0, 0.2), (0.0, 0.3)), "veri"),
        GroupSummary("d", "x", (("toplam", "p", "sum"),), "g", (0, 1)),
    ))
    assert 'tapply(d$p, d$x, sum)[c("0", "1")]' in render_step(LabSpec("konu09", "t", "9", (step,)), 1, "R")


def test_konu11_12_operations_in_the_app_and_both_languages() -> None:
    """Tablo kuralı, üstel ve gamma yoğunlukları, üstel çekiliş, yoğunluk karşılaştırması, olasılık fonksiyonu ile
    yaklaşım eğrisi, histogramda beklenen sayı eğrisi ve çizgi grafiğinde bant."""

    from scipy import stats

    from core.charts import figure_for
    from core.codegen.base import render_script
    from core.labs.spec import DensityCompare, Draw, LabSpec, LabStep, NoteRef, PmfWithDensity, Support
    from core.labs.spec import Derive as DeriveOp

    assert E.evaluate(E.roundto(E.normcdf(E.roundto(0.8333, 2)), 4)) == 0.7967  # Φ(0,83): tablo kuralı
    x = np.linspace(0.1, 40, 50)
    assert T.density("exponential", 15, 15, x) == pytest.approx(stats.expon.pdf(x, scale=15))
    assert T.density("gamma", 5, 5, x[:10]) == pytest.approx(130.208333 * x[:10] ** 4 * np.exp(-5 * x[:10]),
                                                              rel=1e-6)
    with pytest.raises(ValueError, match="σ = μ"):
        T.density("exponential", 15, 10, x)
    ops = (
        NewSample("s", 60, 305),
        Draw("s", "t", "exponential", 5, 5, "Üstel(5)"),
        DeriveOp("s", "n", E.seq(E.var("t")), "n"),
        DeriveOp("s", "m", E.cummean(E.var("t")), "birikimli ortalama"),
        DeriveOp("s", "ust", E.add(5, E.div(10, E.sqrt(E.var("n")))), "üst"),
        DeriveOp("s", "alt", E.sub(5, E.div(10, E.sqrt(E.var("n")))), "alt"),
        Scalar("se", E.div(15, E.sqrt(4)), "σ/√n", decimals=2),
        DensityCompare((("normal", 50, 15, "X"), ("normal", 50, "se", "X̄, n = 4")), (0, 100), "Karşılaştırma", "x"),
        Support("d", "x", 0, 10, "x"),
        DeriveOp("d", "f", E.dbinom(E.var("x"), 10, 0.5), "f"),
        PmfWithDensity("d", "x", "f", "normal", 5, "se", "x", "Olasılık", "Bin ve normal", "Binom", "Normal",
                       shade=((5.5, 6.5),)),
        LineChart("s", "n", "m", "n", "Ortalama", "Yol", markers=False, bands=(("ust", "Bant"), ("alt", ""))),
        Histogram("s", (("t", "T"),), 30, 0, 30, "Süreler", "t", curves=(("exponential", 5, 5, "Beklenen"),)),
    )
    state = run_operations(ops)
    assert list(state.frames["s"]["t"]) == list(np.random.default_rng(305).exponential(5, size=60))
    compare = state.plots["DensityCompare:Karşılaştırma"]
    assert list(compare.columns) == ["x", "f1", "f2"] and len(compare) == 401
    assert compare["f2"].max() == pytest.approx(stats.norm.pdf(0, 0, 7.5), rel=1e-3)
    pmf = state.plots["PmfWithDensity:Bin ve normal"]
    assert set(pmf) == {"cubuk", "egri", "alan"} and pmf["alan"][0][:2] == (5.5, 6.5)
    assert list(state.plots["LineChart:Yol"].columns) == ["n", "m", "ust", "alt"]
    histogram = figure_for(ops[-1], state)
    curve = [trace for trace in histogram.data if trace.name == "Beklenen"][0]
    assert max(curve.y) == pytest.approx(60 * 1 * stats.expon.pdf(0, scale=5))  # gözlem × kutu genişliği × f(0)
    line = figure_for(ops[-2], state)
    assert [trace.showlegend for trace in line.data if trace.name in ("Bant", "")] == [True, False]
    with pytest.raises(ValueError, match="σ = μ"):
        run_operations((NewSample("s", 5, 1), Draw("s", "t", "exponential", 5, 4, "hatalı")))
    spec = LabSpec("konu11", "t", "11", (LabStep(1, "t", NoteRef("11.1"), "t", operations=ops),))
    python, r = render_script(spec, "Python"), render_script(spec, "R")
    assert "rng.exponential(5, size=len(s))" in python and "rexp(nrow(s), rate = 1 / 5)" in r
    assert "stats.norm.pdf(eksen, 50, se)" in python and "dnorm(eksen, 50, se)" in r
    assert "fill_between" in python and 'type = "h"' in r
    assert 'label="_nolegend_"' in python and "lty = 2" in r
    assert "beklenen * stats.expon.pdf(eksen, scale=5)" in python and "dexp(eksen, rate = 1 / 5)" in r
    rounded = LabSpec("konu11", "t", "11", (LabStep(1, "t", NoteRef("11.1"), "t", operations=(
        Scalar("phi", E.roundto(E.normcdf(1.25), 4), "Φ(1,25)", decimals=4),)),))
    assert "phi = np.round(stats.norm.cdf(1.25), 4)" in render_script(rounded, "Python")
    assert "phi <- round(pnorm(1.25), 4)" in render_script(rounded, "R")
