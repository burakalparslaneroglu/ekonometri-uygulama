"""Uygulama tanımını çalıştırır ve notlardaki sayılarla karşılaştırır.

Veriler Wooldridge (2020) veri setleridir (``core.wooldridge_data``) ya da tanımın içinde yazılı küçük veri
setleridir. Simülasyonlarda tek bir ``np.random.default_rng(seed)`` üreteci vardır ve bütün çekilişler işlem
sırasıyla ondan yapılır. Üretilen Python kodu aynı sırayla çektiği için aynı sayıları verir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core import wooldridge_data as W
from core.labs import expr as E
from core.labs import regression as R
from core.labs import tables as T
from core.labs.spec import (
    OLS,
    TOTAL,
    BarChart,
    BoxPlot,
    BoxSummary,
    CellTarget,
    Check,
    CoefTarget,
    Describe,
    GroupStats,
    LoadWooldridge,
    ModelTarget,
    ModelValue,
    PanelSummary,
    RegressionTable,
    ShowModel,
    SortRows,
    ClassHistogram,
    ClassTable,
    CompareBarChart,
    Count,
    CrossTab,
    Derive,
    DensityCompare,
    DensityPlot,
    DotPlot,
    Draw,
    DrawCategory,
    DrawCount,
    DrawDiscrete,
    Event,
    FrequencyTable,
    FromCounts,
    GroupedBarChart,
    Groups,
    GroupSummary,
    HeatMap,
    Histogram,
    InlineData,
    JoinColumns,
    LabSpec,
    LineChart,
    MapCodes,
    MonteCarlo,
    MosaicChart,
    NewSample,
    Operation,
    Outcomes,
    PairStatistic,
    Percentile,
    PieChart,
    PmfWithDensity,
    Rectangles,
    RowSum,
    Scalar,
    ScalarTable,
    ScalarTarget,
    ScatterPlot,
    Selections,
    Shape,
    ShowFrame,
    Statistic,
    StatTarget,
    StemLeaf,
    Support,
    TableTarget,
    TreeDiagram,
    VariableTypes,
)


@dataclass
class CheckResult:
    check: Check
    value: float
    passed: bool

    @property
    def difference(self) -> float:
        return self.value - self.check.expected


@dataclass
class LabState:
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)
    plots: dict[str, object] = field(default_factory=dict)
    models: dict[str, object] = field(default_factory=dict)
    rng: np.random.Generator | None = None


@dataclass
class LabRun:
    state: LabState
    checks: dict[int, list[CheckResult]]

    @property
    def all_passed(self) -> bool:
        return all(result.passed for items in self.checks.values() for result in items)

    def step_checks(self, number: int) -> list[CheckResult]:
        return self.checks.get(number, [])


# --- Yardımcılar ------------------------------------------------------------------

def plot_key(op) -> str:
    """Grafiğin uygulama durumundaki anahtarı; başlıklar bir tanım içinde tekildir."""

    return f"{type(op).__name__}:{op.title}"


def _subset(frame: pd.DataFrame, where: tuple[str, object] | None) -> pd.DataFrame:
    if where is None:
        return frame
    column, value = where
    return frame[frame[column] == value]


def statistic(series: pd.Series, stat: str) -> float:
    if stat == "count":
        return float(series.count())
    if stat == "sum":
        return float(series.sum())
    if stat == "mean":
        return float(series.mean())
    if stat == "median":
        return float(series.median())
    if stat == "mode":
        modes = series.mode()
        if len(modes) != 1:
            raise ValueError(f"Tek bir mod beklenirken {len(modes)} değer en yüksek frekansa sahip.")
        return float(modes.iloc[0])
    if stat == "mode_freq":
        return float(series.value_counts().max())
    if stat == "prod":
        return float(series.prod())
    if stat == "min":
        return float(series.min())
    if stat == "max":
        return float(series.max())
    if stat == "var":
        return float(series.var())  # payda n − 1 (pandas varsayılanı ddof=1)
    if stat == "std":
        return float(series.std())
    if stat == "nunique":
        return float(series.nunique())
    if stat == "value":
        if len(series) != 1:
            raise ValueError(f"Tek değer beklenirken {len(series)} gözlem bulundu.")
        return float(series.iloc[0])
    raise ValueError(f"Desteklenmeyen istatistik: {stat}")


def _scalar(state: LabState):
    def lookup(name: str) -> float:
        return float(state.scalars[name])

    return lookup


def parameter(value: float | str, state: LabState) -> float:
    """Grafik parametresi: sayı ya da önceden hesaplanmış bir skalerin adı."""

    return float(state.scalars[value]) if isinstance(value, str) else float(value)


def evaluate_scalar(expression: E.Expr, state: LabState) -> float:
    return float(E.evaluate(expression, scalar=_scalar(state)))


def without_total(table: pd.DataFrame) -> pd.DataFrame:
    """Grafiklerde ``Toplam`` satırı ve sütunu çizilmez."""

    return table.drop(index=TOTAL, columns=TOTAL, errors="ignore")


def _bar_data(op: BarChart, state: LabState) -> pd.DataFrame:
    if op.x is None:
        table = without_total(state.tables[op.source])
        data = pd.DataFrame({"kategori": table.index.astype(str), "deger": table[op.y].to_numpy(dtype=float)})
    else:
        frame = state.frames[op.source]
        categories = frame[op.x]
        # Sayısal değerler (ör. x = 0, 1, 2) kategori etiketi olur; ondalıksız değerde ".0" yazılmaz.
        labels = (categories.map(E.format_number) if pd.api.types.is_numeric_dtype(categories)
                  else categories.astype(str))
        data = pd.DataFrame({"kategori": labels.to_numpy(), "deger": frame[op.y].to_numpy(dtype=float)})
    if op.sort == "azalan":
        data = data.sort_values("deger", ascending=False, kind="stable")
    elif op.sort is not None:
        raise ValueError(f"Desteklenmeyen sıralama: {op.sort}")
    return data.reset_index(drop=True)


# --- İşlemler -----------------------------------------------------------------------

def _rows(frame: pd.DataFrame, op: ShowFrame) -> pd.DataFrame:
    """``ShowFrame``'in gösterdiği satırlar (üretilen kodla aynı seçim)."""

    if op.head:
        return frame.head(op.head)
    if op.rows:
        if max(op.rows) > len(frame) or min(op.rows) < 1:
            raise ValueError(f"{op.frame}: 1–{len(frame)} aralığı dışında gözlem numarası.")
        return frame.iloc[[row - 1 for row in op.rows]]
    return _subset(frame, op.where)


def shown_frame(op: ShowFrame, state: LabState) -> pd.DataFrame:
    return _rows(state.frames[op.frame], op)[list(op.columns)]


def group_stats(frame: pd.DataFrame, op: GroupStats) -> pd.DataFrame:
    by = op.by[0] if len(op.by) == 1 else list(op.by)
    table = frame.groupby(by)[op.variable].agg(list(op.stats))
    return table.astype(float)


def panel_summary(frame: pd.DataFrame, unit: str, time: str) -> pd.DataFrame:
    periods = frame.groupby(unit)[time].nunique()
    values = {
        "gozlem": len(frame), "birim": frame[unit].nunique(), "ilk_donem": frame[time].min(),
        "son_donem": frame[time].max(), "en_az_donem": periods.min(), "en_cok_donem": periods.max(),
    }
    return pd.DataFrame({"deger": [float(value) for value in values.values()]},
                        index=pd.Index(list(values), name="nicelik"))


def execute(op: Operation, state: LabState) -> None:
    if isinstance(op, LoadWooldridge):
        state.frames[op.frame] = W.load(op.dataset, op.columns)
    elif isinstance(op, SortRows):
        state.frames[op.frame] = state.frames[op.frame].sort_values(op.by, kind="stable").reset_index(drop=True)
    elif isinstance(op, Describe):
        frame = state.frames[op.frame]
        unknown = sorted(set(op.stats) - {"count", "mean", "std", "min", "max"})
        if unknown:
            raise ValueError(f"Desteklenmeyen betimsel ölçü: {unknown}")
        state.tables[op.result] = frame[list(op.variables)].describe().T[list(op.stats)].astype(float)
    elif isinstance(op, GroupStats):
        state.tables[op.result] = group_stats(state.frames[op.frame], op)
    elif isinstance(op, PanelSummary):
        state.tables[op.result] = panel_summary(state.frames[op.frame], op.unit, op.time)
    elif isinstance(op, OLS):
        state.models[op.name] = R.fit_ols(op, state.frames[op.frame])
    elif isinstance(op, ShowModel):
        if op.model not in state.models:
            raise ValueError(f"'{op.model}' modeli henüz tahmin edilmedi.")
    elif isinstance(op, ModelValue):
        result = state.models[op.model]
        state.scalars[op.name] = (R.coefficient(result, op.term, op.quantity) if op.term is not None
                                  else R.model_quantity(result, op.quantity))
    elif isinstance(op, RegressionTable):
        state.tables[op.result] = R.regression_table(op, state.models)
    elif isinstance(op, InlineData):
        state.frames[op.frame] = T.inline_frame(op.columns, op.rows)
    elif isinstance(op, FromCounts):
        state.frames[op.frame] = T.from_counts(op.columns, op.rows)
    elif isinstance(op, Outcomes):
        state.frames[op.frame] = T.outcomes(op.stages)
    elif isinstance(op, Selections):
        state.frames[op.frame] = T.selections(op.items, op.k, op.ordered, op.columns)
    elif isinstance(op, VariableTypes):
        frame = state.frames[op.frame]
        rows = []
        for variable, kind, detail in op.rows:
            stored = "sayı" if pd.api.types.is_numeric_dtype(frame[variable]) else "metin"
            rows.append({"degisken": variable, "saklama": stored, "tur": kind, "ayrinti": detail})
        state.tables[op.result] = pd.DataFrame(rows).set_index("degisken")
    elif isinstance(op, Event):
        frame = state.frames[op.frame]
        frame[op.name] = frame[op.column].isin(list(op.values)).astype(float)
    elif isinstance(op, ShowFrame):
        missing = sorted(set(op.columns) - set(state.frames[op.frame].columns))
        if missing:
            raise ValueError(f"{op.frame}: gösterilecek sütun yok: {', '.join(missing)}")
        _rows(state.frames[op.frame], op)  # satır seçimi geçerli mi
    elif isinstance(op, MapCodes):
        frame = state.frames[op.frame]
        frame[op.name] = T.map_codes(frame[op.source], op.mapping)
    elif isinstance(op, Groups):
        frame = state.frames[op.frame]
        if sum(op.sizes) != len(frame) or len(op.sizes) != len(op.labels):
            raise ValueError("Grup büyüklüklerinin toplamı gözlem sayısına eşit olmalıdır.")
        frame[op.name] = np.repeat(np.asarray(op.labels, dtype=object), op.sizes)
    elif isinstance(op, Support):
        state.frames[op.frame] = pd.DataFrame({op.name: T.support(op.lower, op.upper)})
    elif isinstance(op, Rectangles):
        state.frames[op.frame] = pd.DataFrame({op.name: T.rectangle_midpoints(op.lower, op.width, op.count)})
    elif isinstance(op, RowSum):
        frame = state.frames[op.frame]
        frame[op.name] = frame[list(op.columns)].sum(axis=1).astype(float)
    elif isinstance(op, Derive):
        frame = state.frames[op.frame]
        frame[op.name] = E.evaluate(op.expr, frame, scalar=_scalar(state))
    elif isinstance(op, NewSample):
        state.frames[op.frame] = pd.DataFrame({"id": np.arange(1, op.nobs + 1)})
        if op.seed is not None:
            state.rng = np.random.default_rng(op.seed)
        elif state.rng is None:
            raise ValueError("Tohumsuz örneklem yalnız Monte Carlo döngüsü içinde kullanılabilir.")
    elif isinstance(op, Draw):
        frame, rng = state.frames[op.frame], state.rng
        if op.distribution == "normal":
            frame[op.name] = rng.normal(op.first, op.second, size=len(frame))
        elif op.distribution == "uniform":
            frame[op.name] = rng.uniform(op.first, op.second, size=len(frame))
        elif op.distribution == "beta":
            frame[op.name] = rng.beta(op.first, op.second, size=len(frame))
        elif op.distribution == "gamma":
            frame[op.name] = rng.gamma(op.first, op.second, size=len(frame))
        elif op.distribution == "exponential":
            if op.second != op.first:
                raise ValueError("Üstel dağılımda σ = μ'dür.")
            frame[op.name] = rng.exponential(op.first, size=len(frame))
        else:
            raise ValueError(f"Desteklenmeyen dağılım: {op.distribution}")
    elif isinstance(op, DrawCount):
        frame = state.frames[op.frame]
        frame[op.name] = T.draw_count(state.rng, op.distribution, op.parameters, len(frame))
    elif isinstance(op, DrawCategory):
        frame = state.frames[op.frame]
        u = state.rng.random(len(frame))
        frame[op.name] = T.draw_categories(frame, u, op.categories, op.probabilities, op.by)
    elif isinstance(op, DrawDiscrete):
        frame = state.frames[op.frame]
        u = state.rng.random(len(frame))
        frame[op.name] = T.draw_discrete(u, op.values, op.probabilities)
    elif isinstance(op, Shape):
        frame = state.frames[op.frame]
        state.scalars[op.observations] = float(len(frame))
        state.scalars[op.variables] = float(frame.drop(columns=list(op.exclude)).shape[1])
    elif isinstance(op, Count):
        frame = state.frames[op.frame]
        state.scalars[op.name] = float((frame[op.column] == op.value).sum())
    elif isinstance(op, Statistic):
        series = _subset(state.frames[op.frame], op.where)[op.variable]
        state.scalars[op.name] = statistic(series, op.stat)
    elif isinstance(op, PairStatistic):
        frame = state.frames[op.frame]
        if op.stat == "cov":
            state.scalars[op.name] = float(frame[op.x].cov(frame[op.y]))
        elif op.stat == "corr":
            state.scalars[op.name] = float(frame[op.x].corr(frame[op.y]))
        else:
            raise ValueError(f"Desteklenmeyen iki değişkenli istatistik: {op.stat}")
    elif isinstance(op, Scalar):
        state.scalars[op.name] = evaluate_scalar(op.expr, state)
    elif isinstance(op, ScalarTable):
        state.tables[op.result] = pd.DataFrame(
            {"deger": [evaluate_scalar(expression, state) for _, expression in op.rows]},
            index=pd.Index([label for label, _ in op.rows], name="nicelik"),
        )
    elif isinstance(op, GroupSummary):
        grouped = state.frames[op.frame].groupby(op.by)
        table = pd.DataFrame({name: grouped[variable].agg(stat) for name, variable, stat in op.columns})
        state.tables[op.result] = table.reindex(list(op.order))
    elif isinstance(op, FrequencyTable):
        values = state.frames[op.frame][op.variable]
        state.tables[op.result] = T.frequency_table(values, op.order, relative=op.relative, totals=op.totals)
    elif isinstance(op, CrossTab):
        frame = _subset(state.frames[op.frame], op.where)
        state.tables[op.result] = T.crosstab(
            frame, op.row, op.column, op.row_order, op.column_order, percent=op.percent, margins=op.margins,
            weights=op.weights,
        )
    elif isinstance(op, JoinColumns):
        first = state.tables[op.columns[0][1]]
        state.tables[op.result] = pd.DataFrame(
            {name: state.tables[table][column].to_numpy(dtype=float) for name, table, column in op.columns},
            index=first.index,
        )
    elif isinstance(op, BoxSummary):
        state.tables[op.result] = pd.DataFrame(
            {label: T.box_summary(state.frames[frame][variable]) for frame, variable, label in op.series}
        )
    elif isinstance(op, ClassTable):
        values = state.frames[op.frame][op.variable]
        edges = T.class_edges(values, op.width, op.lower, op.classes)
        state.tables[op.result] = T.class_table(values, edges, op.columns, totals=op.totals,
                                                row_labels=op.row_labels)
    elif isinstance(op, StemLeaf):
        state.tables[op.result] = T.stem_leaf(state.frames[op.frame][op.variable])
    elif isinstance(op, Percentile):
        values = state.frames[op.frame][op.variable]
        state.scalars[op.name] = T.percentile(values, op.p, op.method)
        if op.location is not None:
            state.scalars[op.location] = T.percentile_location(len(values), op.p, op.method)
    elif isinstance(op, BarChart):
        state.plots[plot_key(op)] = _bar_data(op, state)
    elif isinstance(op, GroupedBarChart):
        table = without_total(state.tables[op.table])
        state.plots[plot_key(op)] = table if op.series == "satir" else table.T
    elif isinstance(op, CompareBarChart):
        state.plots[plot_key(op)] = pd.DataFrame(
            {label: state.tables[name][op.column] for label, name in op.tables}
        )
    elif isinstance(op, PieChart):
        values = state.tables[op.table].loc[list(op.order), op.column].astype(float)
        table = pd.DataFrame({op.column: values, "aci": 360 * values})
        state.tables[op.result] = table
        state.plots[plot_key(op)] = table
    elif isinstance(op, LineChart):
        # Kaynak bir veri çerçevesi ya da sonuç tablosu olabilir (ör. kümülatif yüzde eğrisi).
        frame = state.frames[op.frame] if op.frame in state.frames else without_total(state.tables[op.frame])
        columns = [op.x, op.y, *(column for column, _ in op.bands), *(column for column, _ in op.series)]
        state.plots[plot_key(op)] = frame[list(dict.fromkeys(columns))].copy()
    elif isinstance(op, ScatterPlot):
        data = state.frames[op.frame][[op.x, op.y]].dropna()
        line = None
        if op.fit_line:  # en küçük kareler doğrusu: eğim ve sabit (np.polyfit, derece 1)
            slope, intercept = np.polyfit(data[op.x].to_numpy(dtype=float), data[op.y].to_numpy(dtype=float), 1)
            line = (float(intercept), float(slope))
        known = [(parameter(first, state), parameter(second, state), label) for first, second, label in op.lines]
        means = None
        if op.means:  # aynı x değerindeki gözlemlerin y ortalaması (koşullu ortalamanın örneklem karşılığı)
            grouped = data.groupby(op.x)[op.y].mean()
            means = pd.DataFrame({op.x: grouped.index.to_numpy(dtype=float), op.y: grouped.to_numpy(dtype=float)})
        state.plots[plot_key(op)] = {"veri": data.copy(), "dogru": line, "cizgiler": known, "ortalamalar": means}
    elif isinstance(op, BoxPlot):
        boxes = []
        for frame, variable, label in op.series:
            values = state.frames[frame][variable]
            summary = T.box_summary(values)
            boxes.append((label, summary, T.outliers(values, summary)))
        state.plots[plot_key(op)] = boxes
    elif isinstance(op, Histogram):
        source = state.tables[op.table] if op.table in state.tables else state.frames[op.table]
        state.plots[plot_key(op)] = source[[column for column, _ in op.columns]].copy()
    elif isinstance(op, ClassHistogram):
        table = without_total(state.tables[op.table])
        state.plots[plot_key(op)] = table[["alt", "ust", op.y]].copy()
    elif isinstance(op, DotPlot):
        values = state.frames[op.frame][op.variable]
        if op.x_range is not None and not (op.x_range[0] < values.min() and values.max() < op.x_range[1]):
            raise ValueError(f"{op.title}: eksen sınırları {op.x_range} bütün gözlemleri kapsamıyor.")
        state.plots[plot_key(op)] = pd.DataFrame({
            "deger": values.to_numpy(dtype=float),
            "yigin": values.groupby(values).cumcount().to_numpy() + 1,
        })
    elif isinstance(op, (MosaicChart, HeatMap)):
        state.plots[plot_key(op)] = without_total(state.tables[op.table]).astype(float)
    elif isinstance(op, DensityPlot):
        state.plots[plot_key(op)] = T.density_grid(op.distribution, op.first, op.second, op.x_range)
    elif isinstance(op, DensityCompare):
        x = np.linspace(op.x_range[0], op.x_range[1], 401)
        curves = {"x": x}
        for index, (distribution, first, second, _) in enumerate(op.curves, start=1):
            curves[f"f{index}"] = T.density(distribution, parameter(first, state), parameter(second, state), x)
        state.plots[plot_key(op)] = pd.DataFrame(curves)
    elif isinstance(op, PmfWithDensity):
        frame = state.frames[op.frame]
        first, second = parameter(op.first, state), parameter(op.second, state)
        x = np.linspace(frame[op.x].min() - 0.5, frame[op.x].max() + 0.5, 401)
        state.plots[plot_key(op)] = {
            "cubuk": frame[[op.x, op.y]].copy(),
            "egri": pd.DataFrame({"x": x, "f": T.density(op.distribution, first, second, x)}),
            "alan": [(low, high, T.density(op.distribution, first, second, np.linspace(low, high, 200)))
                     for low, high in op.shade],
        }
    elif isinstance(op, TreeDiagram):
        state.plots[plot_key(op)] = T.tree_layout(state.frames[op.frame], op.first, op.second, op.first_p, op.second_p)
    elif isinstance(op, MonteCarlo):
        _monte_carlo(op, state)
    else:
        raise TypeError(f"Tanınmayan işlem: {type(op).__name__}")


def _monte_carlo(op: MonteCarlo, state: LabState) -> None:
    """Tekrar döngüsü: üreteç bir kez tohumlanır; her tekrar aynı üreteçten yeni çekiliş yapar."""

    rng = np.random.default_rng(op.seed)
    rows: list[list[float]] = []
    for _ in range(op.reps):
        local = LabState(rng=rng)
        for inner in op.body:
            execute(inner, local)
        rows.append([evaluate_scalar(expression, local) for _, expression in op.collect])
    state.rng = rng
    state.tables[op.result] = pd.DataFrame(rows, columns=[name for name, _ in op.collect])


# --- Notlarla karşılaştırma ---------------------------------------------------------

def evaluate_target(target, state: LabState) -> float:
    if isinstance(target, StatTarget):
        series = _subset(state.frames[target.frame], target.where)[target.variable]
        return statistic(series, target.stat)
    if isinstance(target, ScalarTarget):
        return float(state.scalars[target.name])
    if isinstance(target, TableTarget):
        return float(state.tables[target.table].loc[target.row, target.column])
    if isinstance(target, CellTarget):
        return float(state.frames[target.frame][target.column].iloc[target.row - 1])
    if isinstance(target, CoefTarget):
        return R.coefficient(state.models[target.model], target.term, target.quantity)
    if isinstance(target, ModelTarget):
        return R.model_quantity(state.models[target.model], target.quantity)
    raise TypeError(f"Tanınmayan hedef: {type(target).__name__}")


def run_operations(operations) -> LabState:
    state = LabState()
    for op in operations:
        execute(op, state)
    return state


def run_lab(spec: LabSpec) -> LabRun:
    state = LabState()
    checks: dict[int, list[CheckResult]] = {}
    for step in spec.steps:
        for op in step.operations:
            execute(op, state)
        results = []
        for check in step.checks:
            value = evaluate_target(check.target, state)
            passed = bool(np.isfinite(value)) and abs(value - check.expected) <= check.tolerance
            results.append(CheckResult(check=check, value=value, passed=passed))
        checks[step.number] = results
    return LabRun(state=state, checks=checks)
