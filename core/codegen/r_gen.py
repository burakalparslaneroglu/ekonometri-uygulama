"""Uygulama tanımından temel R kodu üretir (veri için yalnız ``wooldridge`` paketi gerekir).

Deterministik adımlarda sayılar Python ve uygulamayla ondalık düzeyinde aynıdır. Simülasyonlarda R'nin
rastgele sayı üreteci numpy'ninkinden farklıdır: aynı tohum aynı çekilişi vermez (yalnız dağılım aynıdır).
"""

from __future__ import annotations

from core.codegen.base import (
    HEAT_LOW,
    PALETTE,
    REFERENCE_COLORS,
    Generator,
    flatten,
    text,
    uses_charts,
    wrapped,
)
from core.labs import expr as E
from core.labs.spec import (
    TOTAL,
    BarChart,
    BoxPlot,
    BoxSummary,
    INTERCEPT,
    OLS,
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
    DensityCompare,
    DensityPlot,
    Derive,
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
from core.labs.tables import class_edges

_STAT = {"sum": "sum", "mean": "mean", "median": "median", "prod": "prod", "min": "min", "max": "max",
         "var": "var", "std": "sd", "count": "length"}
"""R'nin ``var`` ve ``sd`` fonksiyonları örneklem ölçüleridir (payda n − 1)."""
_FUNCTIONS = {
    "log": "log", "exp": "exp", "sqrt": "sqrt", "abs": "abs", "maximum": "pmax", "minimum": "pmin",
    "round": "round", "roundto": "round", "floor": "floor", "normcdf": "pnorm", "normpdf": "dnorm",
    "norminv": "qnorm",
    "cumprod": "cumprod", "cummean": "cumsum({0}) / seq_along({0})", "seq": "seq_along({0})",
    "factorial": "factorial({0})", "comb": "choose({0}, {1})", "perm": "factorial({0}) / factorial({0} - {1})",
    "dbinom": "dbinom", "pbinom": "pbinom", "dpois": "dpois", "ppois": "ppois", "dnorm": "dnorm",
    "dhyper_r": "dhyper", "phyper_r": "phyper",
    **{name: f"as.numeric({{0}} {symbol} {{1}})" for name, symbol in E.COMPARISONS.items()},
}
_REFERENCE_STYLES = tuple(zip(REFERENCE_COLORS, ("2", "3", "4")))


def _rewrite(expression: E.Expr) -> E.Expr:
    """R'nin hipergeometrik fonksiyonları (x, başarı, başarısızlık, seçim) sırasını ister: dhyper(x, r, N − r, n)."""

    if isinstance(expression, E.BinOp):
        return E.BinOp(expression.op, _rewrite(expression.left), _rewrite(expression.right))
    if isinstance(expression, E.Call):
        arguments = tuple(_rewrite(argument) for argument in expression.args)
        if expression.fn in ("dhyper", "phyper"):
            x, population, successes, draws = arguments
            return E.Call(f"{expression.fn}_r", (x, successes, E.sub(population, successes), draws))
        return E.Call(expression.fn, arguments)
    return expression


def _r(expression: E.Expr, dialect: E.Dialect) -> str:
    return E.render(_rewrite(expression), dialect)

_NUMBER_TEXT = [
    "# Grafik etiketleri için Türkçe sayı: ondalık virgül, yüzde işareti sayıdan önce",
    "sayi_metni <- function(deger, basamak = 0, yuzde = FALSE) {",
    '  metin <- formatC(deger, format = "f", digits = basamak, decimal.mark = ",")',
    '  if (yuzde) paste0("%", metin) else metin',
    "}",
]
_BOUNDARY_TEXT = [
    "# Sınıf sınırının yazımı: ondalık virgül (ör. 12,5)",
    'sinir_metni <- function(deger) trimws(formatC(deger, format = "fg", digits = 10, decimal.mark = ","))',
]
_PERCENTILE = [
    "# Ders kuralı: L_p = (p/100)(n + 1). L_p tam sayı değilse komşu iki gözlem arasında doğrusal ara değer;",
    "# L_p <= 1 ise en küçük, L_p >= n ise en büyük gözlem. quantile(x, p / 100, type = 6) aynı sonucu verir;",
    "# R'nin varsayılanı (type = 7) farklı bir kural kullanır.",
    "yuzdelik <- function(x, p) {",
    "  x <- sort(x)",
    "  n <- length(x)",
    "  konum <- p / 100 * (n + 1)",
    "  if (konum <= 1) return(x[1])",
    "  if (konum >= n) return(x[n])",
    "  k <- floor(konum)",
    "  x[k] + (konum - k) * (x[k + 1] - x[k])",
    "}",
]


_BOX_SUMMARY = [
    "# Kutu grafiği özeti: çeyrekler ders kuralıyla (yuzdelik); bıyıklar Q1 - 1,5·IQR ile Q3 + 1,5·IQR",
    "# sınırlarının içindeki en uç gözlemlere uzanır; sınırların dışındakiler aykırı değer adayıdır.",
    "kutu_ozeti <- function(x) {",
    "  x <- sort(x)",
    "  q1 <- yuzdelik(x, 25)",
    "  medyan <- yuzdelik(x, 50)",
    "  q3 <- yuzdelik(x, 75)",
    "  iqr <- q3 - q1",
    "  alt <- q1 - 1.5 * iqr",
    "  ust <- q3 + 1.5 * iqr",
    "  icerde <- x[x >= alt & x <= ust]",
    "  c(en_kucuk = x[1], q1 = q1, medyan = medyan, q3 = q3, en_buyuk = x[length(x)], iqr = iqr,",
    "    alt_sinir = alt, ust_sinir = ust, alt_biyik = min(icerde), ust_biyik = max(icerde),",
    "    aykiri_sayisi = sum(x < alt | x > ust))",
    "}",
]
_TREE_BOX = [
    "# Olasılık ağacında düğüm: metnin çevresinde çerçeve (temel R'de metin kutusu yoktur)",
    "kutu <- function(x, y, etiket) {",
    "  w <- strwidth(etiket) / 2 + 0.05",
    "  h <- strheight(etiket) / 2 + 0.12",
    f'  rect(x - w, y - h, x + w, y + h, col = "white", border = "{PALETTE[0]}")',
    "  text(x, y, etiket)",
    "}",
]
_ORDERED_SELECTIONS = [
    "# Sıralı seçimler (permütasyonlar), sözlük sırasıyla: k konumun bütün bileşimlerinden aynı öğeyi",
    "# birden fazla kez içerenler atılır.",
    "sirali_secimler <- function(x, k) {",
    "  g <- as.matrix(expand.grid(rep(list(seq_along(x)), k)))[, k:1, drop = FALSE]",
    "  g <- g[apply(g, 1, function(r) length(unique(r)) == k), , drop = FALSE]",
    "  matrix(x[g], ncol = k)",
    "}",
]


def _parameter(value) -> str:
    """Grafik parametresi: sayı ya da önceden hesaplanmış skalerin (aynı adlı değişken) adı."""

    return value if isinstance(value, str) else E.format_number(value)


def density_expression(distribution: str, first, second, x: str) -> str:
    """Yoğunluk f(x)'in R yazımı; ``first`` ve ``second`` sayı ya da değişken adıdır."""

    a, b = _parameter(first), _parameter(second)
    if distribution == "normal":
        return f"dnorm({x}, {a}, {b})"
    if distribution == "exponential":
        return f"dexp({x}, rate = 1 / {a})"
    if distribution == "gamma":
        return f"dgamma({x}, shape = {a}, rate = {b})"
    return f"dunif({x}, {a}, {b})"


def _vector(values) -> str:
    return "c(" + ", ".join(text(value) for value in values) + ")"


def _colors(count: int) -> str:
    return _vector(PALETTE[:count])


def _sprintf(value: str) -> str:
    """``sprintf`` biçim dizgesi içinde düz metin: tırnak, ters bölü ve yüzde işareti kaçırılır."""

    return value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")


def _quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _subset(frame: str, variable: str, where) -> str:
    """``frame$variable``; ``where`` verilirse yalnız koşulu sağlayan gözlemler."""

    values = f"{frame}${variable}"
    if where is None:
        return values
    column, value = where
    return f"{values}[{frame}${column} == {text(value)}]"


def _statistic(values: str, stat: str) -> str:
    if stat == "value":
        return values
    if stat == "count":
        return f"sum(!is.na({values}))"
    if stat == "mode":
        # En yüksek frekanslı değer(ler); tek mod beklenir
        return f"as.numeric(names(which(table({values}) == max(table({values})))))"
    if stat == "mode_freq":
        return f"max(table({values}))"
    if stat == "nunique":
        return f"length(unique({values}))"
    return f"{_STAT[stat]}({values})"


_R_STAT = {"count": "sum(!is.na(x))", "mean": "mean(x, na.rm = TRUE)", "std": "sd(x, na.rm = TRUE)",
           "min": "min(x, na.rm = TRUE)", "max": "max(x, na.rm = TRUE)", "sum": "sum(x, na.rm = TRUE)"}
"""Betimsel ve grup özetlerinin R yazımı; pandas gibi eksik değerleri dışarıda bırakır (sd: payda n − 1)."""
_COEF_COLUMNS = {"coef": "Estimate", "se": "Std. Error", "t": "t value", "p": "Pr(>|t|)"}
_REGRESSION_TABLE = [
    "# Makale tipi tablo: her terim için katsayı ve standart hata (terim_sh), sonra n ve R²",
    "makale_tablosu <- function(modeller, terimler) {",
    "  sutunlar <- lapply(modeller, function(m) {",
    "    kt <- summary(m)$coefficients",
    '    rownames(kt)[rownames(kt) == "(Intercept)"] <- "Intercept"',
    "    hucreler <- unlist(lapply(terimler, function(terim) {",
    '      if (terim %in% rownames(kt)) kt[terim, c("Estimate", "Std. Error")] else c(NA, NA)',
    "    }))",
    "    c(hucreler, nobs(m), summary(m)$r.squared)",
    "  })",
    "  tablo <- do.call(cbind, sutunlar)",
    '  rownames(tablo) <- c(as.vector(rbind(terimler, paste0(terimler, "_sh"))), "n", "r2")',
    "  tablo",
    "}",
]
_STARS = [
    "# Katsayının yanındaki yıldız: *** p < 0,01; ** p < 0,05; * p < 0,10",
    'yildiz <- function(p) if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""',
]


def r_term(term: str) -> str:
    return "(Intercept)" if term == INTERCEPT else term


def coef_expression(model: str, term: str, quantity: str) -> str:
    name = text(r_term(term))
    if quantity == "coef":
        return f"coef({model})[[{name}]]"
    if quantity in _COEF_COLUMNS:
        return f'summary({model})$coefficients[{name}, "{_COEF_COLUMNS[quantity]}"]'
    return f"confint({model})[{name}, {1 if quantity == 'ci_low' else 2}]"


def model_expression(model: str, quantity: str) -> str:
    return {
        "r2": f"summary({model})$r.squared",
        "adj_r2": f"summary({model})$adj.r.squared",
        "nobs": f"nobs({model})",
        "f": f'summary({model})$fstatistic[["value"]]',
        "f_p": (f'pf(summary({model})$fstatistic[["value"]], summary({model})$fstatistic[["numdf"]], '
                f'summary({model})$fstatistic[["dendf"]], lower.tail = FALSE)'),
        "ssr": f"sum(residuals({model})^2)",
        "df_resid": f"df.residual({model})",
    }[quantity]


def _stat_function(stats) -> list[str]:
    """``function(x) c(count = ..., mean = ...)`` gövdesi, okunur satırlara bölünmüş."""

    items = [f"{stat} = {_R_STAT[stat]}" for stat in stats]
    return ["function(x) {", *wrapped("  c(", items, ")", width=100), "}"]


class RGenerator(Generator):
    language = "R"
    comment = "#"

    def dialect(self, frame: str) -> E.Dialect:
        return E.Dialect(variable=lambda name: f"{frame}${name}", functions=_FUNCTIONS, power="^")

    # --- Başlık ve yardımcılar ------------------------------------------
    def imports(self, operations: tuple[Operation, ...], *, script: bool = False) -> list[str]:
        if not script:
            return []
        if any(isinstance(op, LoadWooldridge) for op in flatten(operations)):
            lines = [
                "# Temel R kullanılır; veri seti için yalnız wooldridge paketi gerekir (bir kez kurulur):",
                '# install.packages(c("wooldridge"))',
                "",
            ]
        else:
            lines = ["# Yalnız temel R kullanılır; ek paket gerekmez.", ""]
        if uses_charts(operations):
            lines += [
                "# Rscript ile (etkileşimsiz) çalıştırıldığında R grafikleri çalışma klasöründe Rplots.pdf",
                "# dosyasına yazar. Klasörde dosya oluşmasın diye grafikler R'nin geçici klasörüne yazılır;",
                "# R kapanınca bu klasör silinir. RStudio'da grafikler her zamanki gibi Plots panelinde görünür.",
                'if (!interactive()) png(file.path(tempdir(), "grafik%02d.png"))',
                "",
            ]
        return lines

    def helpers(self, operations: tuple[Operation, ...], *, with_checks: bool) -> list[str]:
        lines: list[str] = []
        flat = flatten(operations)
        labelled = (BarChart, CompareBarChart, PieChart, MosaicChart, TreeDiagram, HeatMap)
        if any(isinstance(op, labelled) or (isinstance(op, (ClassHistogram, GroupedBarChart)) and op.labels)
               for op in flat):
            lines += _NUMBER_TEXT + [""]
        if any(isinstance(op, (ClassTable, ClassHistogram)) for op in flat):
            lines += _BOUNDARY_TEXT + [""]
        boxes = any(isinstance(op, (BoxSummary, BoxPlot)) for op in flat)
        if boxes or any(isinstance(op, Percentile) and op.method == "ders" for op in flat):
            lines += _PERCENTILE + [""]
        if boxes:
            lines += _BOX_SUMMARY + [""]
        if any(isinstance(op, Selections) and op.ordered for op in flat):
            lines += _ORDERED_SELECTIONS + [""]
        if any(isinstance(op, TreeDiagram) for op in flat):
            lines += _TREE_BOX + [""]
        if any(isinstance(op, RegressionTable) for op in flat):
            lines += _REGRESSION_TABLE + [""]
            if any(isinstance(op, RegressionTable) and op.stars for op in flat):
                lines += _STARS + [""]
        if with_checks:
            lines += [
                "# Hesaplanan değeri ders notlarındaki basılı değerle karşılaştırır",
                "kontrol_et <- function(etiket, deger, beklenen, ondalik = 4) {",
                "  tolerans <- 0.5 * 10^(-ondalik) + 1e-12",
                '  durum <- if (abs(deger - beklenen) <= tolerans) "OK  " else "HATA"',
                '  cat(sprintf("  %s %s: %.*f  (notlar: %s)\\n", durum, etiket, ondalik, deger,',
                '              sprintf("%.*f", ondalik, beklenen)))',
                '  if (abs(deger - beklenen) > tolerans) stop(etiket, " notlarla uyuşmuyor.")',
                "}",
                "",
            ]
        return lines

    # --- İşlemler --------------------------------------------------------
    def operation(self, op: Operation) -> list[str]:
        lines = self._operation(op)
        if self.quiet:
            lines = [line for line in lines if not line.lstrip().startswith(("print(", "cat(", "str("))]
        return lines

    def _operation(self, op: Operation) -> list[str]:  # noqa: C901 - tek dağıtıcı, işlem türü başına bir blok
        if isinstance(op, LoadWooldridge):
            lines = [f"# {op.comment}", f'data("{op.dataset}", package = "wooldridge")']
            if op.columns:
                lines.append(f"{op.frame} <- {op.frame}[, {_vector(op.columns)}]")
            return lines
        if isinstance(op, SortRows):
            return [
                f"# {op.comment}",
                f"{op.frame} <- {op.frame}[order({op.frame}${op.by}), ]  # eşit değerlerde önceki sıra korunur",
                f"rownames({op.frame}) <- NULL",
            ]
        if isinstance(op, Describe):
            body = _stat_function(op.stats)
            return [
                f"# {op.comment}",
                f"{op.result} <- as.data.frame(t(sapply({op.frame}[, {_vector(op.variables)}], {body[0]}",
                *body[1:-1],
                "})))",
                f"print(round({op.result}, {op.decimals}))",
            ]
        if isinstance(op, GroupStats):
            values = f"{op.frame}${op.variable}"
            if len(op.by) == 1:
                groups = f"{op.frame}${op.by[0]}"
            else:  # grup bileşimleri pandas'taki gibi ilk değişkene, sonra ikinciye göre sıralı
                inner = ", ".join(f"{op.frame}${name}" for name in op.by)
                groups = f"interaction({inner}, lex.order = TRUE, drop = TRUE)"
            body = _stat_function(op.stats)
            return [
                f"# {op.comment}",
                f"{op.result} <- as.data.frame(t(sapply(split({values}, {groups}), {body[0]}",
                *body[1:-1],
                "})))",
                f"print(round({op.result}, {op.decimals}))",
            ]
        if isinstance(op, PanelSummary):
            periods = f"{op.result}_donem"
            frame = op.frame
            return [
                f"# {op.comment}",
                f"# Her birimin ({op.unit}) kaç farklı dönemde ({op.time}) gözlendiği",
                f"{periods} <- tapply({frame}${op.time}, {frame}${op.unit}, function(x) length(unique(x)))",
                f"print(summary({periods}))",
                f"{op.result} <- data.frame(deger = c(",
                f"  gozlem = nrow({frame}), birim = length(unique({frame}${op.unit})),",
                f"  ilk_donem = min({frame}${op.time}), son_donem = max({frame}${op.time}),",
                f"  en_az_donem = min({periods}), en_cok_donem = max({periods})",
                "))",
                f"print({op.result})",
            ]
        if isinstance(op, OLS):
            return [f"# {op.comment}", f"{op.name} <- lm({op.formula}, data = {op.frame})"]
        if isinstance(op, ShowModel):
            summary = f"summary({op.model})" if op.stars else f"summary({op.model}), signif.stars = FALSE"
            return [
                f"# {op.comment}",
                "# R özetindeki karşılıklar: Estimate = coef, Std. Error = std err, t value = t, Pr(>|t|) = P>|t|,",
                "# Multiple R-squared = R-squared. R özeti gözlem sayısını yazmaz; nobs() ile yazdırılır.",
                f"print({summary})",
                f"print(confint({op.model}))  # %95 güven aralıkları",
                f'cat("Gözlem sayısı:", nobs({op.model}), "\\n")',
            ]
        if isinstance(op, ModelValue):
            value = (coef_expression(op.model, op.term, op.quantity) if op.term is not None
                     else model_expression(op.model, op.quantity))
            return [
                f"# {op.comment}",
                f"{op.name} <- {value}",
                f'cat(sprintf("{_sprintf(op.comment)}: %.{op.decimals}f\\n", {op.name}))',
            ]
        if isinstance(op, RegressionTable):
            models = ", ".join(f"{text(heading)} = {name}" for heading, name in op.models)
            lines = [
                f"# {op.comment}",
                f"{op.result} <- makale_tablosu(list({models}), {_vector(op.terms)})",
                f"print(round({op.result}, {op.decimals}))",
            ]
            if op.stars:
                for heading, name in op.models:
                    terms = ", ".join(text(r_term(term)) for term in op.terms)
                    lines.append(
                        f"print(sapply(intersect(c({terms}), names(coef({name}))), "
                        f'function(terim) yildiz(summary({name})$coefficients[terim, "Pr(>|t|)"])))'
                    )
            return lines
        if isinstance(op, InlineData):
            return self._inline(op)
        if isinstance(op, FromCounts):
            return self._from_counts(op)
        if isinstance(op, Outcomes):
            return self._outcomes(op)
        if isinstance(op, Selections):
            return self._selections(op)
        if isinstance(op, VariableTypes):
            return self._variable_types(op)
        if isinstance(op, Event):
            members = ", ".join(str(value) for value in op.values)
            return [
                f"# {op.comment}: {op.column} ∈ {{{members}}} olan satırlar 1, diğerleri 0",
                f"{op.frame}${op.name} <- as.integer({op.frame}${op.column} %in% {_vector(op.values)})",
            ]
        if isinstance(op, ShowFrame):
            columns = _vector(op.columns)
            if op.head:
                return [f"# {op.comment}", f"print(head({op.frame}[, {columns}, drop = FALSE], {op.head}))"]
            if op.rows:
                rows = "c(" + ", ".join(str(row) for row in op.rows) + ")"
                return [f"# {op.comment}", f"print({op.frame}[{rows}, {columns}, drop = FALSE])"]
            if op.where is not None:
                column, value = op.where
                return [f"# {op.comment}",
                        f"print({op.frame}[{op.frame}${column} == {text(value)}, {columns}, drop = FALSE])"]
            if op.decimals is not None:
                return [f"# {op.comment}", f"print(round({op.frame}[, {columns}, drop = FALSE], {op.decimals}))"]
            return [f"# {op.comment}", f"print({op.frame}[, {columns}, drop = FALSE])"]
        if isinstance(op, MapCodes):
            pairs = ", ".join(f"{text(label)} = {text(code)}" for label, code in op.mapping)
            return [
                f"# {op.comment}",
                f"{op.frame}${op.name} <- unname(c({pairs})[{op.frame}${op.source}])",
                f'print(head({op.frame}[, c("{op.source}", "{op.name}")], 8))',
            ]
        if isinstance(op, Groups):
            return [
                f"# {op.comment}",
                f"{op.frame}${op.name} <- rep({_vector(op.labels)}, times = {_vector(op.sizes)})",
            ]
        if isinstance(op, Derive):
            return [f"# {op.comment}", f"{op.frame}${op.name} <- {_r(op.expr, self.dialect(op.frame))}"]
        if isinstance(op, Support):
            lower, upper = E.format_number(op.lower), E.format_number(op.upper)
            return [f"# {op.comment}", f"{op.frame} <- data.frame({op.name} = {lower}:{upper})"]
        if isinstance(op, RowSum):
            return [f"# {op.comment}", f"{op.frame}${op.name} <- rowSums({op.frame}[, {_vector(op.columns)}])"]
        if isinstance(op, Rectangles):
            lower, width = E.format_number(op.lower), E.format_number(op.width)
            return [
                f"# {op.comment}",
                f"# {op.count} dikdörtgen; {op.name}: dikdörtgenlerin orta noktaları, alt sınır + genişlik × (i - 0,5)",
                f"{op.frame} <- data.frame({op.name} = {lower} + {width} * (seq_len({op.count}) - 0.5))",
            ]
        if isinstance(op, NewSample):
            frame = f"{op.frame} <- data.frame(id = seq_len({op.nobs}))"
            if op.seed is None:
                return [frame]
            return [
                "# Sabit tohum: betik her çalıştırmada aynı veriyi üretir",
                f"set.seed({op.seed})",
                frame,
            ]
        if isinstance(op, Draw):
            a, b = E.format_number(op.first), E.format_number(op.second)
            call = {
                "normal": f"rnorm(nrow({op.frame}), mean = {a}, sd = {b})",
                "uniform": f"runif(nrow({op.frame}), min = {a}, max = {b})",
                "beta": f"rbeta(nrow({op.frame}), shape1 = {a}, shape2 = {b})",
                "gamma": f"rgamma(nrow({op.frame}), shape = {a}, scale = {b})",
                "exponential": f"rexp(nrow({op.frame}), rate = 1 / {a})",  # ortalama süre μ: hız 1/μ
            }[op.distribution]
            return [f"# {op.comment}", f"{op.frame}${op.name} <- {call}"]
        if isinstance(op, DrawCount):
            return self._draw_count(op)
        if isinstance(op, DrawCategory):
            return self._draw_category(op)
        if isinstance(op, DrawDiscrete):
            return [
                f"# {op.comment}",
                "# tekduze ~ Tek-düze(0, 1); X, birikimli olasılığı F(x) bu sayıyı ilk aşan değerdir "
                "(ters dağılım fonksiyonu)",
                f"tekduze <- runif(nrow({op.frame}))",
                f"{op.name}_degerler <- {_vector(op.values)}",
                f"esik <- cumsum({_vector(op.probabilities)})",
                "esik[length(esik)] <- 1  # yuvarlama hatasına karşı son eşik tam 1",
                f"{op.frame}${op.name} <- {op.name}_degerler[findInterval(tekduze, esik) + 1]",
            ]
        if isinstance(op, Shape):
            columns = (f"length(setdiff(names({op.frame}), {_vector(op.exclude)}))  # kimlik sütunu değişken sayılmaz"
                       if op.exclude else f"ncol({op.frame})")
            return [
                f"{op.observations} <- nrow({op.frame})  # gözlem sayısı",
                f"{op.variables} <- {columns}",
                f'cat("Gözlem sayısı:", {op.observations}, "| Değişken sayısı:", {op.variables}, "\\n")',
            ]
        if isinstance(op, Count):
            return [
                f"# {op.comment}",
                f"{op.name} <- sum({op.frame}${op.column} == {text(op.value)})",
                f'cat("{_quote(op.comment)}:", {op.name}, "\\n")',
            ]
        if isinstance(op, Statistic):
            values = _subset(op.frame, op.variable, op.where)
            return [
                f"# {op.comment}",
                f"{op.name} <- {_statistic(values, op.stat)}",
                f'cat(sprintf("{_sprintf(op.comment)}: %.{op.decimals}f\\n", {op.name}))',
            ]
        if isinstance(op, PairStatistic):
            function = {"cov": "cov", "corr": "cor"}[op.stat]
            note = "  # payda n - 1" if op.stat == "cov" else "  # Pearson korelasyonu"
            return [
                f"# {op.comment}",
                f"{op.name} <- {function}({op.frame}${op.x}, {op.frame}${op.y}){note}",
                f'cat(sprintf("{_sprintf(op.comment)}: %.{op.decimals}f\\n", {op.name}))',
            ]
        if isinstance(op, Scalar):
            rhs = _r(op.expr, self.dialect(""))
            shown = f"%%%.{op.decimals}f" if op.percent else f"%.{op.decimals}f"
            return [
                f"# {op.comment}",
                f"{op.name} <- {rhs}",
                f'cat(sprintf("{_sprintf(op.comment)}: {shown}\\n", {op.name}))',
            ]
        if isinstance(op, ScalarTable):
            dialect = self.dialect("")
            rows = [f"  {text(label)} = {_r(expression, dialect)}" for label, expression in op.rows]
            rows = [row + ("," if index < len(rows) - 1 else "") for index, row in enumerate(rows)]
            return [f"{op.result} <- data.frame(deger = c(", *rows, "))", f"print(round({op.result}, {op.decimals}))"]
        if isinstance(op, GroupSummary):
            return self._group_summary(op)
        if isinstance(op, FrequencyTable):
            return self._frequency(op)
        if isinstance(op, CrossTab):
            return self._crosstab(op)
        if isinstance(op, JoinColumns):
            items = [f'  {text(name)} = {table}[, "{column}"],' for name, table, column in op.columns]
            return [
                f"{op.result} <- data.frame(",
                *items,
                f"  row.names = rownames({op.columns[0][1]}),",
                "  check.names = FALSE",
                ")",
                f"print(round({op.result}, {op.decimals}))",
            ]
        if isinstance(op, BoxSummary):
            items = [f"  {text(label)} = kutu_ozeti({frame}${variable})," for frame, variable, label in op.series]
            return [
                "# Beş sayı özeti, IQR, aykırı değer sınırları ve bıyık uçları",
                f"{op.result} <- data.frame(",
                *items,
                "  check.names = FALSE",
                ")",
                f"print(round({op.result}, 3))",
            ]
        if isinstance(op, ClassTable):
            return self._class_table(op)
        if isinstance(op, StemLeaf):
            return self._stem_leaf(op)
        if isinstance(op, Percentile):
            return self._percentile(op)
        if isinstance(op, ClassHistogram):
            return self._class_histogram(op)
        if isinstance(op, DotPlot):
            return self._dot_plot(op)
        if isinstance(op, BarChart):
            return self._bar(op)
        if isinstance(op, GroupedBarChart):
            return self._grouped(op)
        if isinstance(op, CompareBarChart):
            return self._compare(op)
        if isinstance(op, PieChart):
            return self._pie(op)
        if isinstance(op, LineChart):
            return self._line(op)
        if isinstance(op, ScatterPlot):
            color = (f'"{PALETTE[0]}"' if op.opacity >= 1
                     else f'adjustcolor("{PALETTE[0]}", alpha.f = {E.format_number(op.opacity)})')
            size = "" if op.size == 11 else f", cex = {E.format_number(round(op.size / 11, 2))}"
            lines = [
                f"plot({op.frame}${op.x}, {op.frame}${op.y}, pch = 19, col = {color}{size},",
                f'     xlab = "{_quote(op.x_label)}", ylab = "{_quote(op.y_label)}", main = "{_quote(op.title)}")',
                "grid()",
            ]
            names, colors, types = [], [], []
            if op.fit_line:
                lines += [
                    "# Tahmin edilen doğru (sabit terim ve eğim veriden)",
                    f'abline(lm({op.y} ~ {op.x}, data = {op.frame}), col = "{PALETTE[1]}", lwd = 2.5)',
                ]
                names.append("Tahmin edilen doğru")
                colors.append(PALETTE[1])
                types.append("1")
            for index, (first, second, label) in enumerate(op.lines):
                color = REFERENCE_COLORS[index % len(REFERENCE_COLORS)]
                lines += [
                    f"# {label}: y = sabit + eğim · x (veri üretim sürecinden bilinen doğru)",
                    f'abline(a = {_parameter(first)}, b = {_parameter(second)}, col = "{color}", lty = 2, lwd = 2.5)',
                ]
                names.append(label)
                colors.append(color)
                types.append("2")
            if names:
                legend = ", ".join(f'"{_quote(name)}"' for name in names)
                palette = ", ".join(f'"{color}"' for color in colors)
                if len(names) == 1:
                    lines.append(f'legend("topleft", legend = {legend}, col = {palette}, lwd = 2.5, '
                                 f'lty = {types[0]}, bty = "n")')
                else:
                    lines += [
                        f'legend("topleft", legend = c({legend}),',
                        f'       col = c({palette}), lwd = 2.5, lty = c({", ".join(types)}), bty = "n")',
                    ]
            return lines
        if isinstance(op, BoxPlot):
            return self._box_plot(op)
        if isinstance(op, Histogram):
            return self._histogram(op)
        if isinstance(op, MosaicChart):
            return self._mosaic(op)
        if isinstance(op, TreeDiagram):
            return self._tree(op)
        if isinstance(op, HeatMap):
            return self._heatmap(op)
        if isinstance(op, DensityPlot):
            return self._density(op)
        if isinstance(op, DensityCompare):
            return self._density_compare(op)
        if isinstance(op, PmfWithDensity):
            return self._pmf_density(op)
        if isinstance(op, MonteCarlo):
            return self._monte_carlo(op)
        raise TypeError(f"R üreticisi bu işlemi tanımıyor: {type(op).__name__}")

    def _inline(self, op: InlineData) -> list[str]:
        lines = [f"# {op.comment}"]
        if op.layout and len(op.columns) == 1:
            name = f"{op.frame}_ham"
            items = [text(row[0]) for row in op.rows]
            lines += wrapped(f"{name} <- c(", items, ")", per_line=op.layout, indent="  ")
            lines.append(f"{op.frame} <- data.frame({op.columns[0]} = {name})")
        else:
            lines.append(f"{op.frame} <- data.frame(")
            for position, column in enumerate(op.columns):
                values = [text(row[position]) for row in op.rows]
                ending = "," if position < len(op.columns) - 1 else ""
                lines += wrapped(f"  {column} = c(", values, f"){ending}")
            lines.append(")")
        lines.append(f"print({op.frame})" if len(op.rows) <= 12 else f"print(head({op.frame}, 5))  # ilk beş gözlem")
        return lines

    def _from_counts(self, op: FromCounts) -> list[str]:
        name = f"{op.frame}_sayim"
        lines = [f"# {op.comment}", f"{name} <- data.frame("]
        names = (*op.columns, "sayi")
        for position, column in enumerate(names):
            values = [text(row[position]) for row in op.rows]
            ending = "," if position < len(names) - 1 else ""
            lines += wrapped(f"  {column} = c(", values, f"){ending}")
        lines += [
            ")",
            '# Her satır "sayi" kez tekrarlanır: bir satır = bir gözlem',
            f"satirlar <- rep(seq_len(nrow({name})), {name}$sayi)",
            f"{op.frame} <- {name}[satirlar, {_vector(op.columns)}, drop = FALSE]",
            f"rownames({op.frame}) <- NULL",
            f"print(nrow({op.frame}))  # gözlem sayısı",
        ]
        return lines

    def _outcomes(self, op: Outcomes) -> list[str]:
        if len(op.stages) == 1:
            name, values = op.stages[0]
            shown = f"print({op.frame})" if len(values) <= 12 else f"print(head({op.frame}, 5))  # ilk beş sonuç"
            return [
                f"# {op.comment}",
                f"{op.frame} <- data.frame({name} = {_vector(values)}, stringsAsFactors = FALSE)",
                shown,
                f'cat("Sonuç sayısı:", nrow({op.frame}), "\\n")',
            ]
        stages = [f"  {name} = {_vector(values)}" for name, values in op.stages]
        stages = [stage + ("," if index < len(stages) - 1 else "") for index, stage in enumerate(stages)]
        count = 1
        for _, values in op.stages:
            count *= len(values)
        shown = f"print({op.frame})" if count <= 12 else f"print(head({op.frame}, 5))  # ilk beş sonuç"
        return [
            f"# {op.comment}",
            f"{op.frame}_asamalar <- list(",
            *stages,
            ")",
            "# Bütün bileşimler (çarpım kuralı). expand.grid ilk aşamayı en hızlı değiştirir; aşamalar ters",
            "# sırayla verilip sütunlar yeniden dizildiği için son aşama en hızlı değişir (ağaç diyagramındaki sıra).",
            f"{op.frame} <- expand.grid(rev({op.frame}_asamalar), stringsAsFactors = FALSE)"
            f"[, names({op.frame}_asamalar), drop = FALSE]",
            shown,
            f'cat("Sonuç sayısı:", nrow({op.frame}), "\\n")',
        ]

    def _selections(self, op: Selections) -> list[str]:
        items = f"{op.frame}_ogeler"
        if op.ordered:
            rows, note = f"sirali_secimler({items}, {op.k})", "sıra önemli: permütasyonlar"
        else:
            rows, note = f"t(combn({items}, {op.k}))", "sıra önemsiz: kombinasyonlar"
        return [
            f"# {op.comment} ({note})",
            f"{items} <- {_vector(op.items)}",
            f"{op.frame} <- as.data.frame({rows}, stringsAsFactors = FALSE)",
            f"names({op.frame}) <- {_vector(op.columns)}",
            f"print(head({op.frame}, 10))",
            f'cat("Seçim sayısı:", nrow({op.frame}), "\\n")',
        ]

    def _variable_types(self, op: VariableTypes) -> list[str]:
        lines = [
            "# Yazılımın saklama türü: sayı (num) ya da metin (chr)",
            f"str({op.frame})",
            "# İstatistiksel tür yazılımdan değil, değişkenin anlamından gelir",
            f"{op.result} <- data.frame(",
        ]
        fields = (("degisken", 0), ("tur", 1), ("ayrinti", 2))
        for index, (field, position) in enumerate(fields):
            values = [text(row[position]) for row in op.rows]
            ending = "," if index < len(fields) - 1 else ""
            lines += wrapped(f"  {field} = c(", values, f"){ending}")
        return lines + [")", f"print({op.result})"]

    @staticmethod
    def _draw_count(op: DrawCount) -> list[str]:
        values = [E.format_number(value) for value in op.parameters]
        rows = f"nrow({op.frame})"
        if op.distribution == "binomial":
            return [f"# {op.comment} (n bağımsız Bernoulli denemesindeki başarı sayısı)",
                    f"{op.frame}${op.name} <- rbinom({rows}, size = {values[0]}, prob = {values[1]})"]
        if op.distribution == "poisson":
            return [f"# {op.comment} (aralıktaki olay sayısı)",
                    f"{op.frame}${op.name} <- rpois({rows}, lambda = {values[0]})"]
        population, successes, draws = (int(value) for value in op.parameters)
        return [
            f"# {op.comment} (N birimden yerine koymadan seçilen n birimdeki başarı sayısı)",
            "# R'nin adları: m başarı sayısı, n başarısızlık sayısı, k seçim sayısı",
            f"{op.frame}${op.name} <- rhyper({rows}, m = {successes}, n = {population - successes}, k = {draws})",
        ]

    def _draw_category(self, op: DrawCategory) -> list[str]:
        categories = f"{op.name}_kategoriler"
        lines = [
            f"# {op.comment}",
            "# u ~ Tek-düze(0, 1); kategori, birikimli olasılığı u'yu ilk aşan kategoridir",
            f"u <- runif(nrow({op.frame}))",
            f"{categories} <- {_vector(op.categories)}",
        ]
        if not op.by:
            probs = op.probabilities[0][1]
            return lines + [
                f"esik <- cumsum({_vector(probs)})",
                "esik[length(esik)] <- 1  # yuvarlama hatasına karşı son eşik tam 1",
                f"{op.frame}${op.name} <- {categories}[findInterval(u, esik) + 1]",
            ]
        table = f"{op.name}_olasilik"
        if len(op.by) == 1:
            entries = [f"  {text(condition[0])} = {_vector(probs)}" for condition, probs in op.probabilities]
            entries = [entry + ("," if index < len(entries) - 1 else "") for index, entry in enumerate(entries)]
            return lines + [
                f"{table} <- list(",
                *entries,
                ")",
                f"{op.frame}${op.name} <- NA_character_",
                f"for (kosul in names({table})) {{",
                f"  secili <- {op.frame}${op.by[0]} == kosul",
                f"  esik <- cumsum({table}[[kosul]])",
                "  esik[length(esik)] <- 1",
                f"  {op.frame}${op.name}[secili] <- {categories}[findInterval(u[secili], esik) + 1]",
                "}",
            ]
        entries = [
            f"  list(kosul = {_vector(condition)}, olasilik = {_vector(probs)})"
            for condition, probs in op.probabilities
        ]
        entries = [entry + ("," if index < len(entries) - 1 else "") for index, entry in enumerate(entries)]
        selection = " & ".join(
            f"{op.frame}${column} == satir$kosul[{index + 1}]" for index, column in enumerate(op.by)
        )
        return lines + [
            f"{table} <- list(",
            *entries,
            ")",
            f"{op.frame}${op.name} <- NA_character_",
            f"for (satir in {table}) {{",
            f"  secili <- {selection}",
            "  esik <- cumsum(satir$olasilik)",
            "  esik[length(esik)] <- 1",
            f"  {op.frame}${op.name}[secili] <- {categories}[findInterval(u[secili], esik) + 1]",
            "}",
        ]

    def _group_summary(self, op: GroupSummary) -> list[str]:
        # tapply sonucunun adları metindir; sayısal grup değerleri (0, 1, …) konumla değil adla seçilsin
        order = _vector(tuple(value if isinstance(value, str) else E.format_number(float(value)) for value in op.order))
        lines = [f"{op.result} <- data.frame("]
        for index, (name, variable, stat) in enumerate(op.columns):
            ending = "," if index < len(op.columns) - 1 else ""
            lines.append(
                f"  {name} = tapply({op.frame}${variable}, {op.frame}${op.by}, {_STAT[stat]})[{order}]{ending}"
            )
        return lines + [")", f"print(round({op.result}, 4))"]

    def _frequency(self, op: FrequencyTable) -> list[str]:
        order = f"{op.result}_sira"
        lines = [
            f"{order} <- {_vector(op.order)}",
            f"{op.result} <- data.frame(",
            f"  frekans = as.vector(table(factor({op.frame}${op.variable}, levels = {order}))),",
            f"  row.names = {order}",
            ")",
        ]
        if op.relative:
            lines += [
                f"{op.result}$goreli <- {op.result}$frekans / sum({op.result}$frekans)  # r = f / n",
                f"{op.result}$yuzde <- 100 * {op.result}$goreli  # p = 100 r",
            ]
        if op.totals:
            lines.append(f'{op.result}["{TOTAL}", ] <- colSums({op.result})')
        lines.append(f"print(round({op.result}, 3))")
        return lines

    def _crosstab(self, op: CrossTab) -> list[str]:
        lines: list[str] = []
        data = op.frame
        if op.where is not None:
            data = f"{op.result}_veri"
            column, value = op.where
            lines += [
                f"# Yalnız {column} = {value} olan gözlemler",
                f"{data} <- {op.frame}[{op.frame}${column} == {text(value)}, ]",
            ]
        factors = [
            f"  factor({data}${op.row}, levels = {_vector(op.row_order)}),",
            f"  factor({data}${op.column}, levels = {_vector(op.column_order)})",
        ]
        if op.weights is not None:
            terms = [
                f"  factor({data}${op.row}, levels = {_vector(op.row_order)}) +",
                f"  factor({data}${op.column}, levels = {_vector(op.column_order)})",
            ]
            lines += [
                f'# Hücreler gözlem sayısı değil, "{op.weights}" sütununun toplamıdır',
                f"{op.result} <- as.data.frame.matrix(xtabs({data}${op.weights} ~", *terms, "))",
            ]
            if op.margins:
                lines += [
                    f'{op.result}["{TOTAL}", ] <- colSums({op.result})  # sütun toplamları',
                    f"{op.result}${TOTAL} <- rowSums({op.result})  # satır toplamları",
                ]
            return lines + [f"print(round({op.result}, {op.decimals}))"]
        if op.percent is None:
            lines += [f"{op.result} <- as.data.frame.matrix(table(", *factors, "))"]
            if op.margins:
                lines += [
                    f'{op.result}["{TOTAL}", ] <- colSums({op.result})  # sütun toplamları',
                    f"{op.result}${TOTAL} <- rowSums({op.result})  # satır toplamları",
                ]
            return lines + [f"print({op.result})"]
        counts = f"{op.result}_sayi"
        lines += [f"{counts} <- table(", *factors, ")"]
        if op.percent == "satir":
            lines += [
                "# Satır yüzdesi: payda satır toplamıdır",
                f"{op.result} <- as.data.frame.matrix(100 * prop.table({counts}, 1))",
            ]
            if op.margins:
                lines.append(f"{op.result}${TOTAL} <- rowSums({op.result})")
        else:
            lines += [
                "# Sütun yüzdesi: payda sütun toplamıdır",
                f"{op.result} <- as.data.frame.matrix(100 * prop.table({counts}, 2))",
            ]
            if op.margins:
                lines.append(f'{op.result}["{TOTAL}", ] <- colSums({op.result})')
        return lines + [f"print(round({op.result}, {op.decimals}))"]

    def _class_table(self, op: ClassTable) -> list[str]:
        source = f"{op.frame}${op.variable}"
        r = op.result
        if op.lower is None:
            lines = [
                f"h <- {E.format_number(op.width)}  # sınıf genişliği",
                f"alt_sinir <- floor(min({source}) / h) * h  # en küçük değeri içeren h katı",
                f"k <- floor((max({source}) - alt_sinir) / h) + 1  # en büyük değeri de kapsayan sınıf sayısı",
                "kenarlar <- alt_sinir + h * (0:k)",
            ]
        else:
            edges = class_edges(None, op.width, op.lower, op.classes)
            lines = wrapped("kenarlar <- c(", [E.format_number(value) for value in edges], ")  # sınıf sınırları")
        if op.row_labels == "ust":
            labels = 'paste("x <", sinir_metni(kenarlar[-1]))'
        else:
            labels = 'paste(sinir_metni(head(kenarlar, -1)), "≤ x <", sinir_metni(kenarlar[-1]))'
        lines += [
            "# Sınıflar [alt, üst): alt sınır dahil, üst sınır hariç",
            f"siniflar <- cut({source}, breaks = kenarlar, right = FALSE)",
            "frekans <- as.vector(table(siniflar))  # her sınıftaki gözlem sayısı",
            "n_sinif <- sum(frekans)",
            f"{r} <- data.frame(alt = head(kenarlar, -1), ust = kenarlar[-1],",
            f"{' ' * len(r)}              row.names = {labels})",
        ]
        steps = {
            "orta_nokta": f"{r}$orta_nokta <- ({r}$alt + {r}$ust) / 2  # m = (alt + üst) / 2",
            "frekans": f"{r}$frekans <- frekans",
            "goreli": f"{r}$goreli <- frekans / n_sinif  # r = f / n",
            "yuzde": f"{r}$yuzde <- 100 * (frekans / n_sinif)  # p = 100 r",
            "kumulatif_frekans": f"{r}$kumulatif_frekans <- cumsum(frekans)  # F = f1 + ... + fj",
            "kumulatif_goreli": f"{r}$kumulatif_goreli <- cumsum(frekans) / n_sinif",
            "kumulatif_yuzde": f"{r}$kumulatif_yuzde <- 100 * (cumsum(frekans) / n_sinif)",
        }
        lines += [line for column, line in steps.items() if column in op.columns]
        if op.totals:
            summed = [column for column in ("frekans", "goreli", "yuzde") if column in op.columns]
            lines += [
                f"toplamlar <- colSums({r}[, {_vector(summed)}, drop = FALSE])",
                f'{r}["{TOTAL}", ] <- NA  # alt ve üst sınır toplanmaz',
                f'{r}["{TOTAL}", names(toplamlar)] <- toplamlar',
            ]
        return lines + [f"print(round({r}, 3))"]

    def _stem_leaf(self, op: StemLeaf) -> list[str]:
        r = op.result
        return [
            "# R'nin stem() fonksiyonu gövde ölçeğini kendisi seçer; notlardaki gösterimle aynı olsun diye",
            "# gövde (onlar basamağı) ve yaprak (birler basamağı) açıkça ayrılır.",
            f"sirali <- sort({op.frame}${op.variable})",
            "govde <- sirali %/% 10",
            "yaprak <- sirali %% 10",
            "govdeler <- seq(min(govde), max(govde))",
            f"{r} <- data.frame(",
            '  yapraklar = sapply(govdeler, function(g) paste(yaprak[govde == g], collapse = " ")),',
            "  yaprak_sayisi = sapply(govdeler, function(g) sum(govde == g)),",
            "  row.names = govdeler",
            ")",
            f'cat(paste(govdeler, "|", {r}$yapraklar), sep = "\\n")',
        ]

    def _percentile(self, op: Percentile) -> list[str]:
        p = E.format_number(op.p)
        source = f"{op.frame}${op.variable}"
        if op.method == "ders":
            lines = [f"# {op.comment}: ders kuralı L_p = (p/100)(n + 1)"]
            location = f"{p} / 100 * (length({source}) + 1)"
            value = f"yuzdelik({source}, {p})"
        else:
            lines = [f"# {op.comment}: R'nin varsayılanı (type = 7), konum 1 + (p/100)(n - 1)"]
            location = f"1 + {p} / 100 * (length({source}) - 1)"
            value = f"quantile({source}, {p} / 100, names = FALSE)"
        if op.location is not None:
            lines.append(f"{op.location} <- {location}")
        lines.append(f"{op.name} <- {value}")
        if op.location is not None:
            lines.append(
                f'cat(sprintf("{_sprintf(op.comment)}: %.{op.decimals}f (konum %.2f)\\n", {op.name}, {op.location}))'
            )
        else:
            lines.append(f'cat(sprintf("{_sprintf(op.comment)}: %.{op.decimals}f\\n", {op.name}))')
        return lines

    def _class_histogram(self, op: ClassHistogram) -> list[str]:
        rows, _ = self.totals.get(op.table, (False, False))
        source = f'{op.table}[rownames({op.table}) != "{TOTAL}", ]' if rows else op.table
        values = f"cizim${op.y}"
        lines = [
            f"cizim <- {source}",
            "genislik <- cizim$ust - cizim$alt",
            "# Bitişik dikdörtgenler (space = 0): genişlik sınıf genişliği, yükseklik sınıfın değeri",
            f'konum <- barplot({values}, width = genislik, space = 0, col = "{PALETTE[0]}", border = "white",',
            f"                 ylim = c(0, max({values}) * 1.15),",
            f'                 xlab = "{_quote(op.x_label)}", ylab = "{_quote(op.y_label)}",',
            f'                 main = "{_quote(op.title)}")',
            "axis(1, at = c(0, cumsum(genislik)), labels = sinir_metni(c(cizim$alt, cizim$ust[nrow(cizim)])))",
        ]
        if op.labels:
            percent = ", yuzde = TRUE" if op.percent else ""
            lines.append(
                f"text(konum, {values}, labels = sayi_metni({values}, {op.decimals}{percent}), pos = 3, xpd = TRUE)"
            )
        return lines

    def _dot_plot(self, op: DotPlot) -> list[str]:
        source = f"{op.frame}${op.variable}"
        limits = ""
        if op.x_range is not None:
            low, high = (E.format_number(value) for value in op.x_range)
            limits = f"xlim = c({low}, {high}), "  # karşılaştırılan grafiklerde aynı yatay eksen
        lines = [
            f"yigin <- ave({source}, {source}, FUN = seq_along)  # aynı değerdeki gözlemler üst üste",
            f'plot({source}, yigin, pch = 19, col = "{PALETTE[0]}", yaxt = "n", ylim = c(0.5, max(yigin) + 0.5),',
            f'     {limits}xlab = "{_quote(op.x_label)}", ylab = "{_quote(op.y_label)}", main = "{_quote(op.title)}")',
            "axis(2, at = seq_len(max(yigin)), las = 1)",
        ]
        if op.references:
            colors, ltys, labels = [], [], []
            for index, (name, label) in enumerate(op.references):
                color, lty = _REFERENCE_STYLES[index % len(_REFERENCE_STYLES)]
                lines.append(f'abline(v = {name}, col = "{color}", lty = {lty}, lwd = 2)')
                colors.append(f'"{color}"')
                ltys.append(lty)
                labels.append(text(label))
            lines.append(
                f'legend("topright", legend = c({", ".join(labels)}), col = c({", ".join(colors)}), '
                f'lty = c({", ".join(ltys)}), lwd = 2, bty = "n")'
            )
        return lines

    def _line(self, op: LineChart) -> list[str]:
        kind = "b" if op.markers else "l"
        if op.series:
            columns = [op.y, *(column for column, _ in op.series)]
            labels = [text(op.legend or op.y_label), *(text(label) for _, label in op.series)]
            colors = [f'"{PALETTE[index % len(PALETTE)]}"' for index in range(len(columns))]
            lines = [
                f"aralik <- range(c({', '.join(f'{op.frame}${column}' for column in columns)}), na.rm = TRUE)",
                f'plot({op.frame}${op.x}, {op.frame}${op.y}, type = "{kind}", pch = 19, lwd = 2, col = {colors[0]},',
                "     ylim = c(aralik[1], aralik[2] + 0.25 * diff(aralik)),",
                f'     xlab = "{_quote(op.x_label)}", ylab = "{_quote(op.y_label)}", main = "{_quote(op.title)}")',
            ]
            for column, color in zip(columns[1:], colors[1:]):
                lines.append(f'lines({op.frame}${op.x}, {op.frame}${column}, type = "{kind}", pch = 19, lwd = 2, '
                             f"col = {color})")
            lines += [
                "grid()",
                f'legend("top", legend = c({", ".join(labels)}), col = c({", ".join(colors)}), lwd = 2, '
                'horiz = TRUE, bty = "n")',
            ]
            return lines
        values = [f"{op.frame}${op.y}", *(name for name, _ in op.references),
                  *(f"{op.frame}${column}" for column, _ in op.bands)]
        legend = bool(op.references or op.bands)
        # Başvuru çizgileri ve bantlar dikey eksenin içinde kalsın: eksen hepsini kapsar. Üstteki boşluk
        # açıklama içindir; açıklama seriyi ve çizgileri örtmez.
        limits = ['     ylim = c(aralik[1], aralik[2] + 0.3 * diff(aralik)), yaxt = "n",'] if legend else []
        setup = [
            f"aralik <- range(c({', '.join(values)}))",
            "# Üstteki boşluk açıklama için: seri ve başvuru çizgileri alttaki aralıkta kalır",
        ] if legend else []
        lines = [
            *setup,
            f'plot({op.frame}${op.x}, {op.frame}${op.y}, type = "{kind}", pch = 19, lwd = 2, col = "{PALETTE[0]}",',
            *limits,
            f'     xlab = "{_quote(op.x_label)}", ylab = "{_quote(op.y_label)}", main = "{_quote(op.title)}")',
        ]
        if legend:
            lines.append("axis(2, at = pretty(aralik))  # eksen değerleri yalnız verinin aralığında")
            colors, ltys, labels = [f'"{PALETTE[0]}"'], ["1"], [text(op.y_label)]
            for index, (name, label) in enumerate(op.references):
                color, lty = _REFERENCE_STYLES[index % len(_REFERENCE_STYLES)]
                lines.append(f'abline(h = {name}, col = "{color}", lty = {lty}, lwd = 2)')
                colors.append(f'"{color}"')
                ltys.append(lty)
                labels.append(text(label))
            for column, label in op.bands:  # boş etiket açıklamada gösterilmez
                lines.append(f'lines({op.frame}${op.x}, {op.frame}${column}, col = "{PALETTE[2]}", lty = 2, lwd = 1.5)')
                if label:
                    colors.append(f'"{PALETTE[2]}"')
                    ltys.append("2")
                    labels.append(text(label))
            lines.append(
                f'legend("top", legend = c({", ".join(labels)}), col = c({", ".join(colors)}), '
                f'lty = c({", ".join(ltys)}), lwd = 2, bty = "n")'
            )
        return lines

    def _box_plot(self, op: BoxPlot) -> list[str]:
        items = [f"  {text(label)} = {frame}${variable}" for frame, variable, label in op.series]
        items = [item + ("," if index < len(items) - 1 else "") for index, item in enumerate(items)]
        return [
            "# Kutu: Q1'den Q3'e; çizgi: medyan; bıyıklar: sınırların içindeki en uç gözlemler; noktalar: aykırı",
            "seriler <- list(",
            *items,
            ")",
            "ozetler <- lapply(seriler, kutu_ozeti)",
            "aykiri <- c()",
            "grup <- c()",
            "for (i in seq_along(seriler)) {",
            '  secili <- seriler[[i]][seriler[[i]] < ozetler[[i]]["alt_sinir"] |',
            '                        seriler[[i]] > ozetler[[i]]["ust_sinir"]]',
            "  aykiri <- c(aykiri, secili)",
            "  grup <- c(grup, rep(i, length(secili)))",
            "}",
            "# bxp() kutuları verilen özetlerle çizer (boxplot() kendi çeyrek kuralını kullanırdı)",
            "eski_par <- par(mar = c(5, 10, 4, 2))  # uzun seri adları için geniş sol kenar",
            'bxp(list(stats = sapply(ozetler, function(k) k[c("alt_biyik", "q1", "medyan", "q3", "ust_biyik")]),',
            "         n = lengths(seriler), out = aykiri, group = grup, names = names(seriler)),",
            f'    horizontal = TRUE, boxwex = 0.5, boxfill = adjustcolor("{PALETTE[0]}", 0.2),',
            f'    medcol = "{PALETTE[1]}", medlwd = 3, whisklty = 1, outpch = 19, outcol = "{PALETTE[1]}", las = 1,',
            "    show.names = TRUE,",
            f'    xlab = "{_quote(op.x_label)}", main = "{_quote(op.title)}")',
            f'title(ylab = "{_quote(op.y_label)}", line = 8.5)',
            "par(eski_par)",
        ]

    def _chart_matrix(self, table: str) -> str:
        """Grafikte çizilecek sayılar: ``Toplam`` satırı ve sütunu çıkarılmış matris."""

        rows, columns = self.totals.get(table, (False, False))
        row_part = f'rownames({table}) != "{TOTAL}"' if rows else ""
        column_part = f'names({table}) != "{TOTAL}"' if columns else ""
        if rows or columns:
            return f"as.matrix({table}[{row_part}, {column_part}, drop = FALSE])"
        return f"as.matrix({table})"

    def _bar(self, op: BarChart) -> list[str]:
        if op.x is None:
            lines = [f"cizim <- setNames({op.source}${op.y}, rownames({op.source}))"]
            if self.totals.get(op.source, (False, False))[0]:
                lines.append(f'cizim <- cizim[names(cizim) != "{TOTAL}"]  # toplam çizilmez')
        else:
            lines = [f"cizim <- setNames({op.source}${op.y}, {op.source}${op.x})"]
        if op.sort == "azalan":
            lines.append("cizim <- cizim[order(-cizim)]  # yüksekten düşüğe")
        labels = f"sayi_metni(cizim, {op.decimals}{', yuzde = TRUE' if op.percent else ''})"
        x_label, y_label, title = _quote(op.x_label), _quote(op.y_label), _quote(op.title)
        if op.horizontal:
            limit = (f"c({E.format_number(op.y_range[0])}, {E.format_number(op.y_range[1])})"
                     if op.y_range is not None else "range(0, cizim) * 1.2")
            return lines + [
                "cizim <- rev(cizim)  # barplot yatayda ilk değeri en alta çizer; ilk kategori en üstte olsun",
                "eski_par <- par(mar = c(5, 10, 4, 2))  # uzun kategori adları için geniş sol kenar",
                f'konum <- barplot(cizim, horiz = TRUE, las = 1, col = "{PALETTE[0]}", border = NA,',
                f'                 xlim = {limit}, xlab = "{y_label}",',
                f'                 main = "{title}")',
                f'title(ylab = "{x_label}", line = 8.5)',
                f"text(cizim, konum, labels = {labels}, pos = 4, xpd = TRUE)",
                "par(eski_par)",
            ]
        if op.y_range is not None:
            limit = f"c({E.format_number(op.y_range[0])}, {E.format_number(op.y_range[1])})"
            return lines + [
                "# Değer ekseni verilen aralıkla sınırlanır; sütunlar alt sınırdan başlıyormuş gibi görünür",
                f'konum <- barplot(cizim, col = "{PALETTE[0]}", border = NA, ylim = {limit}, xpd = FALSE,',
                f'                 xlab = "{x_label}", ylab = "{y_label}",',
                f'                 main = "{title}")',
                "box(bty = \"l\")",
                f"text(konum, cizim, labels = {labels}, pos = 3, xpd = TRUE)",
            ]
        if op.x is not None and (op.source, op.y) in self.signed_columns:
            position = "ifelse(cizim < 0, 1, 3)"
            placement = ["# Etiket pozitif sütunun üstüne, negatif sütunun altına yazılır"]
        else:
            position, placement = "3", []
        return lines + [
            f'konum <- barplot(cizim, col = "{PALETTE[0]}", border = NA, ylim = range(0, cizim) * 1.15,',
            f'                 xlab = "{x_label}", ylab = "{y_label}",',
            f'                 main = "{title}")',
            *placement,
            f"text(konum, cizim, labels = {labels}, pos = {position}, xpd = TRUE)",
        ]

    def _bars_with_legend(self, x_label: str, y_label: str, title: str, stacked: bool, decimals: int,
                          labels: bool = True) -> list[str]:
        """``cizim`` matrisinin çok serili sütun grafiği: satırlar seriler, sütunlar yatay eksen."""

        x_label, y_label, title = _quote(x_label), _quote(y_label), _quote(title)
        colors = f"{_colors(len(PALETTE))}[seq_len(nrow(cizim))]"
        if stacked:
            middle = [
                "orta <- apply(cizim, 2, function(s) cumsum(s) - s / 2)  # her parçanın ortası",
                f'text(rep(konum, each = nrow(cizim)), orta, labels = sayi_metni(cizim, {decimals}), col = "white")',
            ]
            return [
                f"renkler <- {colors}",
                "konum <- barplot(cizim, col = renkler, border = NA, ylim = c(0, max(colSums(cizim)) * 1.3),",
                f'                 xlab = "{x_label}", ylab = "{y_label}",',
                f'                 main = "{title}")',
                *(middle if labels else []),
                'legend("top", legend = rownames(cizim), fill = renkler, border = NA, horiz = TRUE, bty = "n")',
            ]
        return [
            f"renkler <- {colors}",
            "konum <- barplot(cizim, beside = TRUE, col = renkler, border = NA, ylim = range(0, cizim) * 1.3,",
            f'                 xlab = "{x_label}", ylab = "{y_label}",',
            f'                 main = "{title}")',
            *([f"text(konum, cizim, labels = sayi_metni(cizim, {decimals}), pos = 3, xpd = TRUE)"] if labels else []),
            'legend("top", legend = rownames(cizim), fill = renkler, border = NA, horiz = TRUE, bty = "n")',
        ]

    def _grouped(self, op: GroupedBarChart) -> list[str]:
        matrix = self._chart_matrix(op.table)
        if op.series == "satir":
            lines = ["# Tablonun satırları seriler, sütunları yatay eksende", f"cizim <- {matrix}"]
        else:
            lines = ["# Tablonun sütunları seriler, satırları yatay eksende (matris devriği)", f"cizim <- t({matrix})"]
        return lines + self._bars_with_legend(op.x_label, op.y_label, op.title, op.stacked, op.decimals, op.labels)

    def _compare(self, op: CompareBarChart) -> list[str]:
        items = [f'  {text(label)} = {name}[, "{op.column}"]' for label, name in op.tables]
        items = [item + ("," if index < len(items) - 1 else "") for index, item in enumerate(items)]
        first = op.tables[0][1]
        return [
            "# Her tablonun aynı sütunu yan yana; sütunlar yatay eksende, satırlar seriler",
            "cizim <- cbind(",
            *items,
            ")",
            f"rownames(cizim) <- rownames({first})",
            *self._bars_with_legend(op.x_label, op.y_label, op.title, False, op.decimals),
        ]

    def _pie(self, op: PieChart) -> list[str]:
        order = f"{op.result}_sira"
        return [
            f"{order} <- {_vector(op.order)}  # dilimlerin sırası",
            f'{op.result} <- {op.table}[{order}, "{op.column}", drop = FALSE]',
            f"{op.result}$aci <- 360 * {op.result}${op.column}  # θ = 360° × r",
            f"print(round({op.result}, 3))",
            f"pie({op.result}${op.column},",
            f"    labels = paste(rownames({op.result}), sayi_metni(100 * {op.result}${op.column}, 1, yuzde = TRUE)),",
            f"    col = {_colors(len(op.order))}, clockwise = FALSE, init.angle = 0,",
            f'    main = "{_quote(op.title)}")',
        ]

    def _histogram(self, op: Histogram) -> list[str]:
        lower, upper = E.format_number(op.lower), E.format_number(op.upper)
        series = [f"{op.table}${column}" for column, _ in op.columns]
        lines = [
            f"# [{lower}, {upper}] aralığında {op.bins} eşit genişlikte kutu; aralık dışındaki değerler çizilmez",
            f"kutular <- seq({lower}, {upper}, length.out = {op.bins + 1})",
            f"aralikta <- function(x) x[x >= {lower} & x <= {upper}]",
            f"sayimlar <- sapply(list({', '.join(series)}),",
            "                   function(x) hist(aralikta(x), breaks = kutular, plot = FALSE)$counts)",
        ]
        fills, labels, ltys, colors = [], [], [], []
        curves = []
        if op.curves:
            lines += [
                "# Beklenen sayı eğrisi: gözlem sayısı × kutu genişliği × f(x)",
                f"eksen <- seq({lower}, {upper}, length.out = 401)",
                f"beklenen <- length({series[0]}) * (kutular[2] - kutular[1])",
            ]
            for index, (distribution, a, b, _) in enumerate(op.curves, start=1):
                lines.append(f"egri{index} <- beklenen * {density_expression(distribution, a, b, 'eksen')}")
                curves.append(f"egri{index}")
        top = f"max(sayimlar, {', '.join(curves)})" if curves else "max(sayimlar)"
        for index, (item, (_, label)) in enumerate(zip(series, op.columns)):
            color = f'adjustcolor("{PALETTE[index % len(PALETTE)]}", 0.55)'
            if index == 0:
                lines += [
                    f'hist(aralikta({item}), breaks = kutular, col = {color}, border = "white",',
                    f'     ylim = c(0, {top}), xlab = "{_quote(op.x_label)}", ylab = "{_quote(op.y_label)}",',
                    f'     main = "{_quote(op.title)}")',
                ]
            else:
                lines.append(f'hist(aralikta({item}), breaks = kutular, col = {color}, border = "white", add = TRUE)')
            fills.append(color)
            labels.append(text(label))
            ltys.append("NA")
            colors.append("NA")
        for index, (value, label) in enumerate(op.references):
            color, lty = _REFERENCE_STYLES[index % len(_REFERENCE_STYLES)]
            position = value if isinstance(value, str) else E.format_number(value)
            lines.append(f'abline(v = {position}, col = "{color}", lty = {lty}, lwd = 2)')
            fills.append("NA")
            labels.append(text(label))
            ltys.append(lty)
            colors.append(f'"{color}"')
        for index, (name, (_, _, _, label)) in enumerate(zip(curves, op.curves), start=len(op.columns)):
            color = PALETTE[index % len(PALETTE)]
            lines.append(f'lines(eksen, {name}, col = "{color}", lwd = 2)')
            fills.append("NA")
            labels.append(text(label))
            ltys.append("1")
            colors.append(f'"{color}"')
        return lines + [
            f"legend(\"topright\", legend = c({', '.join(labels)}),",
            f"       fill = c({', '.join(fills)}), border = NA,",
            f"       lty = c({', '.join(ltys)}), col = c({', '.join(colors)}), lwd = 2, bty = \"n\")",
        ]

    def _mosaic(self, op: MosaicChart) -> list[str]:
        x_label, y_label, title = _quote(op.x_label), _quote(op.y_label), _quote(op.title)
        return [
            f"cizim <- {self._chart_matrix(op.table)}  # hücreler: sayılar ya da ortak olasılıklar",
            "genislik <- rowSums(cizim) / sum(cizim)  # sütun genişliği: satırın marjinal payı",
            "pay <- cizim / rowSums(cizim)  # sütun içindeki pay: satır verildiğinde koşullu olasılık",
            "sol <- cumsum(genislik) - genislik  # sütunların sol kenarı",
            f"renkler <- {_colors(len(PALETTE))}[seq_len(ncol(cizim))]",
            'plot(NA, xlim = c(0, 1), ylim = c(0, 1), xaxs = "i", yaxs = "i", xaxt = "n",',
            f'     xlab = "{x_label}", ylab = "{y_label}", main = "{title}")',
            "axis(1, at = sol + genislik / 2, labels = rownames(cizim), tick = FALSE)",
            "taban <- rep(0, nrow(cizim))",
            "for (j in rev(seq_len(ncol(cizim)))) {  # ilk sütun en üstte",
            '  rect(sol, taban, sol + genislik, taban + pay[, j], col = renkler[j], border = "white", lwd = 2)',
            "  # parçanın alanı = ortak olasılık",
            "  text(sol + genislik / 2, taban + pay[, j] / 2,",
            f'       paste0(colnames(cizim)[j], "\\n", sayi_metni(genislik * pay[, j], {op.decimals})), col = "white")',
            "  taban <- taban + pay[, j]",
            "}",
        ]

    def _tree(self, op: TreeDiagram) -> list[str]:
        f, s, fp, sp = op.first, op.second, op.first_p, op.second_p
        return [
            "# Olasılık ağacı (soldan sağa): her satır bir yol, ilk yol en üstte",
            f"yollar <- {op.frame}",
            "n_yol <- nrow(yollar)",
            "y_yol <- rev(seq_len(n_yol) - 1)",
            f"ilk_dallar <- unique(yollar${f})",
            f"y_ilk <- sapply(ilk_dallar, function(dal) mean(y_yol[yollar${f} == dal]))",
            "y_kok <- mean(y_ilk)  # kök, ilk dalların ortasında",
            "plot.new()",
            "plot.window(xlim = c(-0.3, 3), ylim = c(-0.6, n_yol - 0.4))",
            f'title(main = "{_quote(op.title)}")',
            "for (i in seq_along(ilk_dallar)) {",
            f"  p <- yollar${fp}[yollar${f} == ilk_dallar[i]][1]",
            f'  segments(0, y_kok, 1, y_ilk[i], col = "{PALETTE[0]}", lwd = 2)',
            f'  text(0.5, (y_kok + y_ilk[i]) / 2, sayi_metni(p, 2), pos = 3, col = "{REFERENCE_COLORS[0]}")',
            "}",
            "for (i in seq_len(n_yol)) {",
            f"  j <- match(yollar${f}[i], ilk_dallar)",
            f'  segments(1, y_ilk[j], 2, y_yol[i], col = "{PALETTE[0]}", lwd = 2)',
            f"  text(1.5, (y_ilk[j] + y_yol[i]) / 2, sayi_metni(yollar${sp}[i], 2), pos = 3,",
            f'       col = "{REFERENCE_COLORS[0]}")',
            f"  kutu(2, y_yol[i], yollar${s}[i])",
            "  # yolun ortak olasılığı: dal olasılıklarının çarpımı",
            f'  text(2.3, y_yol[i], paste("ortak", sayi_metni(yollar${fp}[i] * yollar${sp}[i], {op.decimals})),',
            f'       pos = 4, col = "{PALETTE[1]}")',
            "}",
            "for (i in seq_along(ilk_dallar)) kutu(1, y_ilk[i], ilk_dallar[i])",
            f'kutu(0, y_kok, "{_quote(op.root)}")',
        ]

    def _heatmap(self, op: HeatMap) -> list[str]:
        x_label, y_label, title = _quote(op.x_label), _quote(op.y_label), _quote(op.title)
        return [
            f"cizim <- {self._chart_matrix(op.table)}",
            "satir_sayisi <- nrow(cizim)",
            "# image() ilk satırı en alta çizer; notlardaki gibi ilk satır üstte olsun diye satırlar ters çevrilir",
            "ters <- t(cizim[satir_sayisi:1, , drop = FALSE])",
            "# Sıfır açık tonda, en büyük değer ana renkte; sıfır hücreler de zeminden ayrılır",
            f'renkler <- colorRampPalette(c("{HEAT_LOW}", "{PALETTE[0]}"))(100)',
            "# Sütun adları ve eksen adı üstte (notlardaki tablo gibi), başlık onların üstünde: geniş üst kenar",
            "eski_par <- par(mar = c(1, 5, 6.5, 1))",
            "image(x = seq_len(ncol(cizim)), y = seq_len(satir_sayisi), z = ters, col = renkler,",
            f'      zlim = c(0, max(cizim)), axes = FALSE, xlab = "", ylab = "{y_label}")',
            'abline(v = seq_len(ncol(cizim) + 1) - 0.5, h = seq_len(satir_sayisi + 1) - 0.5, col = "white", lwd = 3)',
            "axis(3, at = seq_len(ncol(cizim)), labels = colnames(cizim), tick = FALSE)",
            f'mtext("{x_label}", side = 3, line = 2.5)',
            f'title(main = "{title}", line = 4.5)',
            "axis(2, at = seq_len(satir_sayisi), labels = rev(rownames(cizim)), las = 1, tick = FALSE)",
            'metin_rengi <- ifelse(ters > 0.6 * max(cizim), "white", "black")',
            "text(rep(seq_len(ncol(cizim)), times = satir_sayisi), rep(seq_len(satir_sayisi), each = ncol(cizim)),",
            f"     sayi_metni(as.vector(ters), {op.decimals}), col = as.vector(metin_rengi))",
            "par(eski_par)",
        ]

    @staticmethod
    def _density_call(op: DensityPlot, x: str) -> str:
        return density_expression(op.distribution, op.first, op.second, x)

    def _density(self, op: DensityPlot) -> list[str]:
        low, high = (E.format_number(value) for value in op.x_range)
        kind = {
            "normal": "N(μ, σ²): ortalama, standart sapma",
            "uniform": "U(a, b): alt ve üst sınır",
            "exponential": "üstel: ortalama süre μ (R'de hız 1/μ)",
            "gamma": "gamma: biçim ve oran",
        }[op.distribution]
        curve = self._density_call(op, "eksen")
        top = f"1.08 * max({curve})" if op.y_max is None else E.format_number(op.y_max)
        lines = [
            f"# Yoğunluk eğrisi, {kind}",
            f"eksen <- seq({low}, {high}, length.out = 401)",
            f'plot(eksen, {curve}, type = "l", lwd = 2, col = "{PALETTE[0]}", xaxs = "i",',
            f"     ylim = c(0, {top}),",
            f'     xlab = "{_quote(op.x_label)}", ylab = "{_quote(op.y_label)}", main = "{_quote(op.title)}")',
        ]
        if op.shade:
            pairs = ", ".join(f"c({E.format_number(a)}, {E.format_number(b)})" for a, b in op.shade)
            lines += [
                f"for (aralik in list({pairs})) {{  # olasılık = eğri altındaki alan",
                "  xa <- seq(aralik[1], aralik[2], length.out = 200)",
                f"  polygon(c(aralik[1], xa, aralik[2]), c(0, {self._density_call(op, 'xa')}, 0),",
                f'          col = adjustcolor("{PALETTE[0]}", 0.3), border = NA)',
                "}",
            ]
        if op.references:
            colors, ltys, labels = [], [], []
            for index, (value, label) in enumerate(op.references):
                color, lty = _REFERENCE_STYLES[index % len(_REFERENCE_STYLES)]
                lines.append(f'abline(v = {E.format_number(value)}, col = "{color}", lty = {lty}, lwd = 2)')
                colors.append(f'"{color}"')
                ltys.append(lty)
                labels.append(text(label))
            lines += [
                f'legend("topright", legend = c({", ".join(labels)}),',
                f'       col = c({", ".join(colors)}), lty = c({", ".join(ltys)}), lwd = 2, bty = "n")',
            ]
        return lines

    def _density_compare(self, op: DensityCompare) -> list[str]:
        low, high = (E.format_number(value) for value in op.x_range)
        curves = [density_expression(distribution, a, b, "eksen") for distribution, a, b, _ in op.curves]
        colors = _colors(len(op.curves))
        top = "1.08 * max(egriler)" if op.y_max is None else E.format_number(op.y_max)
        lines = [
            "# Yoğunluk eğrileri aynı eksende",
            f"eksen <- seq({low}, {high}, length.out = 401)",
            *wrapped("egriler <- cbind(", curves, ")"),
            f'matplot(eksen, egriler, type = "l", lty = 1, lwd = 2, col = {colors}, xaxs = "i",',
            f"        ylim = c(0, {top}),",
            f'        xlab = "{_quote(op.x_label)}", ylab = "{_quote(op.y_label)}", main = "{_quote(op.title)}")',
            f"legend(\"topright\", legend = c({', '.join(text(label) for *_, label in op.curves)}),",
            f'       col = {colors}, lty = 1, lwd = 2, bty = "n")',
        ]
        return lines

    def _pmf_density(self, op: PmfWithDensity) -> list[str]:
        x, y = f"{op.frame}${op.x}", f"{op.frame}${op.y}"
        lines = [
            f"eksen <- seq(min({x}) - 0.5, max({x}) + 0.5, length.out = 401)",
            f"egri <- {density_expression(op.distribution, op.first, op.second, 'eksen')}",
            f'plot({x}, {y}, type = "h", lwd = 8, lend = 1, col = adjustcolor("{PALETTE[0]}", 0.8),',
            f"     ylim = c(0, 1.08 * max(c({y}, egri))),",
            f'     xlab = "{_quote(op.x_label)}", ylab = "{_quote(op.y_label)}", main = "{_quote(op.title)}")',
            f'lines(eksen, egri, col = "{PALETTE[1]}", lwd = 2)',
        ]
        if op.shade:
            pairs = ", ".join(f"c({E.format_number(a)}, {E.format_number(b)})" for a, b in op.shade)
            curve = density_expression(op.distribution, op.first, op.second, "xa")
            lines += [
                f"for (aralik in list({pairs})) {{  # sürekli yaklaşımda tam sayı değerinin alanı",
                "  xa <- seq(aralik[1], aralik[2], length.out = 200)",
                f"  polygon(c(aralik[1], xa, aralik[2]), c(0, {curve}, 0),",
                f'          col = adjustcolor("{PALETTE[1]}", 0.3), border = NA)',
                "}",
            ]
        return lines + [
            f'legend("topright", legend = c({text(op.bar_label)}, {text(op.curve_label)}),',
            f'       col = c("{PALETTE[0]}", "{PALETTE[1]}"), lwd = c(8, 2), bty = "n")',
        ]

    def _monte_carlo(self, op: MonteCarlo) -> list[str]:
        lines = [
            f"# {op.comment}",
            f"# {op.reps} tekrar; tohum döngüden önce bir kez ayarlanır",
            f"set.seed({op.seed})",
            f'sonuclar <- vector("list", {op.reps})',
            f"for (tekrar in seq_len({op.reps})) {{",
        ]
        self.quiet = True
        try:
            for inner in op.body:
                lines += [f"  {line}" if line else "" for line in self.operation(inner)]
        finally:
            self.quiet = False
        dialect = self.dialect("")
        lines.append("  sonuclar[[tekrar]] <- c(")
        for index, (name, expression) in enumerate(op.collect):
            ending = "," if index < len(op.collect) - 1 else ""
            lines.append(f"    {name} = {_r(expression, dialect)}{ending}")
        return lines + [
            "  )",
            "}",
            f"{op.result} <- as.data.frame(do.call(rbind, sonuclar))",
            f"print(summary({op.result}))",
        ]

    # --- Notlarla karşılaştırma -----------------------------------------
    def target(self, target) -> str:
        if isinstance(target, StatTarget):
            return _statistic(_subset(target.frame, target.variable, target.where), target.stat)
        if isinstance(target, ScalarTarget):
            return target.name
        if isinstance(target, TableTarget):
            # Satır adı her zaman metin olarak verilir: tbl[7, ] yedinci satırı, tbl["7", ] adı 7 olan satırı seçer.
            # Sütun adı da metin olarak verilir: sayı olan bir sütun adı (ör. 2) konumla seçilirdi.
            row = target.row if isinstance(target.row, str) else E.format_number(float(target.row))
            column = target.column if isinstance(target.column, str) else E.format_number(float(target.column))
            return f"{target.table}[{text(row)}, {text(column)}]"
        if isinstance(target, CellTarget):
            return f"{target.frame}${target.column}[{target.row}]"
        if isinstance(target, CoefTarget):
            return coef_expression(target.model, target.term, target.quantity)
        if isinstance(target, ModelTarget):
            return model_expression(target.model, target.quantity)
        raise TypeError(f"Tanınmayan hedef: {type(target).__name__}")

    def check_lines(self, checks: tuple[Check, ...]) -> list[str]:
        lines = ['cat("Notlarla karşılaştırma:\\n")']
        for check in checks:
            expected = f"{check.expected:.{check.decimals}f}"
            target = self.target(check.target)
            lines.append(f'kontrol_et("{_quote(check.label)}", {target}, {expected}, {check.decimals})')
        return lines

    def closing(self) -> list[str]:
        return [f'cat("\\n{_quote(self.closing_message())}\\n")']
