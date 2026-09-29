"""Laboratuvar tanımlarında kullanılan küçük ifade dili.

Türetilmiş değişkenler ve skalerler bir kez burada tanımlanır; aynı ifade hem
uygulamada (numpy/pandas) değerlendirilir hem de Python ve R koduna çevrilir.
Böylece iki dildeki kod ile uygulamanın hesabı tek kaynaktan gelir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Var:
    """Veri çerçevesindeki bir değişken (sütun)."""

    name: str


@dataclass(frozen=True)
class Const:
    value: float


@dataclass(frozen=True)
class BinOp:
    op: str
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Call:
    fn: str
    args: tuple["Expr", ...]


@dataclass(frozen=True)
class Ref:
    """Daha önce hesaplanmış bir skaler; üretilen kodda aynı adlı değişken."""

    name: str


Expr = Union[Var, Const, BinOp, Call, Ref]

BINARY_OPS = ("+", "-", "*", "/", "^")
COMPARISONS = {"le": "<=", "lt": "<", "ge": ">=", "gt": ">", "eq": "==", "ne": "!="}
"""Karşılaştırma fonksiyonları: koşul sağlanırsa 1, değilse 0 (gösterge değişkeni)."""
DISTRIBUTION_FUNCTIONS = ("dbinom", "pbinom", "dpois", "ppois", "dhyper", "phyper", "dnorm")
"""Olasılık fonksiyonları ve birikimli olasılıklar (vektör üzerinde de çalışır): Python'da ``scipy.stats``, R'de
``dbinom``/``pbinom``, ``dpois``/``ppois``, ``dhyper``/``phyper`` ve ``dnorm``."""
TEST_FUNCTIONS = ("tcdf", "tsf", "tinv", "fsf", "finv")
"""Test istatistiklerinin dağılımları: t dağılımının birikimli olasılığı P(T ≤ x), üst kuyruğu P(T > x) ve ters
fonksiyonu (kritik değer), F dağılımının üst kuyruğu P(F > x) ve ters fonksiyonu. Python'da ``scipy.stats`` (``t``,
``f``), R'de ``pt``/``qt``/``pf``/``qf``. Üst kuyruk doğrudan hesaplanır: 1 − P(T ≤ x) çok küçük p-değerlerinde
(ör. 10⁻²²) sıfıra yuvarlanırdı."""
FUNCTIONS = (
    "neg", "log", "exp", "sqrt", "abs", "maximum", "minimum", "round", "roundto", "floor", "normcdf", "normpdf",
    "norminv",
    "cumprod", "cummean", "seq", "factorial", "comb", "perm", *DISTRIBUTION_FUNCTIONS, *TEST_FUNCTIONS, *COMPARISONS,
)
ARITY = {
    **{name: 2 for name in ("maximum", "minimum", "roundto", "comb", "perm", "dpois", "ppois", "tcdf", "tsf", "tinv",
                            *COMPARISONS)},
    **{name: 3 for name in ("dbinom", "pbinom", "dnorm", "fsf", "finv")},
    "dhyper": 4, "phyper": 4,
}
"""Birden fazla argüman alan fonksiyonların argüman sayısı; diğerleri tek argüman alır."""
COUNTING_FUNCTIONS = ("factorial", "comb", "perm")
"""Sayma fonksiyonları tam sayı ister; Python'da ``math`` modülüyle, R'de ``factorial``/``choose`` ile yazılır."""
_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}
_ATOM = 4


# --- Kurucular: tanım dosyalarının okunaklı kalması için -------------------

def var(name: str) -> Var:
    return Var(name)


def const(value: float) -> Const:
    return Const(float(value))


def _wrap(value: "Expr | float | int") -> "Expr":
    if isinstance(value, (int, float)):
        return Const(float(value))
    return value


def add(a, b) -> BinOp:
    return BinOp("+", _wrap(a), _wrap(b))


def sub(a, b) -> BinOp:
    return BinOp("-", _wrap(a), _wrap(b))


def mul(a, b) -> BinOp:
    return BinOp("*", _wrap(a), _wrap(b))


def div(a, b) -> BinOp:
    return BinOp("/", _wrap(a), _wrap(b))


def power(a, b) -> BinOp:
    return BinOp("^", _wrap(a), _wrap(b))


def neg(a) -> Call:
    """Tek terimli eksi: −a (ör. e^(−λ) için ``exp(neg(ref("lam")))``)."""

    return Call("neg", (_wrap(a),))


def log(a) -> Call:
    return Call("log", (_wrap(a),))


def exp(a) -> Call:
    return Call("exp", (_wrap(a),))


def sqrt(a) -> Call:
    return Call("sqrt", (_wrap(a),))


def absolute(a) -> Call:
    return Call("abs", (_wrap(a),))


def maximum(a, b) -> Call:
    return Call("maximum", (_wrap(a), _wrap(b)))


def minimum(a, b) -> Call:
    return Call("minimum", (_wrap(a), _wrap(b)))


def floor(a) -> Call:
    return Call("floor", (_wrap(a),))


def rounded(a) -> Call:
    """En yakın tam sayıya yuvarlama (sürekli çekilişlerde yarım değer olasılığı sıfırdır)."""

    return Call("round", (_wrap(a),))


def roundto(a, digits: int) -> Call:
    """``digits`` ondalık basamağa yuvarlama (ör. tablo değerleri: z iki, Φ(z) dört basamak)."""

    return Call("roundto", (_wrap(a), _wrap(digits)))


def normcdf(a) -> Call:
    """Standart normal dağılım fonksiyonu Φ(a)."""

    return Call("normcdf", (_wrap(a),))


def normpdf(a) -> Call:
    """Standart normal yoğunluk φ(a)."""

    return Call("normpdf", (_wrap(a),))


def norminv(a) -> Call:
    """Standart normal dağılımın ters fonksiyonu Φ⁻¹(a) (ör. Φ⁻¹(0,975) ≈ 1,96)."""

    return Call("norminv", (_wrap(a),))


def cumprod(a) -> Call:
    """Birikimli çarpım: bir değişkenin ilk gözlemden o gözleme kadar değerlerinin çarpımı."""

    return Call("cumprod", (_wrap(a),))


def cummean(a) -> Call:
    """Birikimli ortalama: k. değer ilk k gözlemin ortalamasıdır (ör. birikimli göreli frekans)."""

    return Call("cummean", (_wrap(a),))


def seq(a) -> Call:
    """Sıra numarası 1, 2, …, n (``a`` yalnız uzunluk için kullanılan bir değişkendir)."""

    return Call("seq", (_wrap(a),))


def factorial(a) -> Call:
    """N! = N(N − 1)⋯2·1; 0! = 1."""

    return Call("factorial", (_wrap(a),))


def comb(a, b) -> Call:
    """Kombinasyon C(N, n) = N! / (n!(N − n)!): sıra önemsiz seçimlerin sayısı."""

    return Call("comb", (_wrap(a), _wrap(b)))


def perm(a, b) -> Call:
    """Permütasyon P(N, n) = N! / (N − n)!: sıralı seçimlerin sayısı."""

    return Call("perm", (_wrap(a), _wrap(b)))


def dbinom(x, n, p) -> Call:
    """Binom olasılık fonksiyonu P(X = x), X ~ Bin(n, p)."""

    return Call("dbinom", (_wrap(x), _wrap(n), _wrap(p)))


def pbinom(x, n, p) -> Call:
    """Binom birikimli olasılığı P(X ≤ x), X ~ Bin(n, p)."""

    return Call("pbinom", (_wrap(x), _wrap(n), _wrap(p)))


def dpois(x, lam) -> Call:
    """Poisson olasılık fonksiyonu P(X = x), X ~ Pois(λ)."""

    return Call("dpois", (_wrap(x), _wrap(lam)))


def ppois(x, lam) -> Call:
    """Poisson birikimli olasılığı P(X ≤ x), X ~ Pois(λ)."""

    return Call("ppois", (_wrap(x), _wrap(lam)))


def dhyper(x, population, successes, draws) -> Call:
    """Hipergeometrik olasılık fonksiyonu P(X = x): N birimlik anakütlede r başarı, yerine koymadan n seçim."""

    return Call("dhyper", (_wrap(x), _wrap(population), _wrap(successes), _wrap(draws)))


def phyper(x, population, successes, draws) -> Call:
    """Hipergeometrik birikimli olasılık P(X ≤ x); argümanlar ``dhyper`` ile aynı sırada (x, N, r, n)."""

    return Call("phyper", (_wrap(x), _wrap(population), _wrap(successes), _wrap(draws)))


def dnorm(x, mean, sd) -> Call:
    """Normal yoğunluk f(x), X ~ N(μ, σ²): ortalama ``mean``, standart sapma ``sd``."""

    return Call("dnorm", (_wrap(x), _wrap(mean), _wrap(sd)))


def tcdf(x, df) -> Call:
    """t dağılımının birikimli olasılığı P(T ≤ x), serbestlik derecesi ``df`` (sol kuyruk p-değeri)."""

    return Call("tcdf", (_wrap(x), _wrap(df)))


def tsf(x, df) -> Call:
    """t dağılımının üst kuyruğu P(T > x) (sağ kuyruk p-değeri; iki taraflı p = 2·P(T > |t|))."""

    return Call("tsf", (_wrap(x), _wrap(df)))


def tinv(q, df) -> Call:
    """t dağılımının ters fonksiyonu: P(T ≤ c) = q olan c (ör. q = 0,975 ile yüzde 5 iki taraflı kritik değer)."""

    return Call("tinv", (_wrap(q), _wrap(df)))


def fsf(x, df1, df2) -> Call:
    """F(df1, df2) dağılımının üst kuyruğu P(F > x): F testinin p-değeri."""

    return Call("fsf", (_wrap(x), _wrap(df1), _wrap(df2)))


def finv(q, df1, df2) -> Call:
    """F(df1, df2) dağılımının ters fonksiyonu: P(F ≤ c) = q olan c (ör. q = 0,95 ile yüzde 5 kritik değeri)."""

    return Call("finv", (_wrap(q), _wrap(df1), _wrap(df2)))


def compare(name: str, a, b) -> Call:
    """Gösterge: ``a`` ile ``b`` karşılaştırması doğruysa 1, değilse 0 (``name``: le, lt, ge, gt, eq, ne)."""

    if name not in COMPARISONS:
        raise ValueError(f"Desteklenmeyen karşılaştırma: {name}")
    return Call(name, (_wrap(a), _wrap(b)))


def ref(name: str) -> Ref:
    return Ref(name)


# --- Doğrulama ---------------------------------------------------------------

def validate(expr: Expr) -> None:
    if isinstance(expr, (Var, Const, Ref)):
        return
    if isinstance(expr, BinOp):
        if expr.op not in BINARY_OPS:
            raise ValueError(f"Desteklenmeyen işlem: {expr.op}")
        validate(expr.left)
        validate(expr.right)
        return
    if isinstance(expr, Call):
        if expr.fn not in FUNCTIONS:
            raise ValueError(f"Desteklenmeyen fonksiyon: {expr.fn}")
        expected = ARITY.get(expr.fn, 1)
        if len(expr.args) != expected:
            raise ValueError(f"{expr.fn} {expected} argüman alır.")
        for argument in expr.args:
            validate(argument)
        return
    raise TypeError(f"Tanınmayan ifade düğümü: {type(expr).__name__}")


def variables(expr: Expr) -> set[str]:
    """İfadenin başvurduğu veri değişkenleri."""

    if isinstance(expr, Var):
        return {expr.name}
    if isinstance(expr, BinOp):
        return variables(expr.left) | variables(expr.right)
    if isinstance(expr, Call):
        found: set[str] = set()
        for argument in expr.args:
            found |= variables(argument)
        return found
    return set()


def references(expr: Expr) -> set[str]:
    """İfadenin başvurduğu skaler adları (``ref``)."""

    if isinstance(expr, Ref):
        return {expr.name}
    if isinstance(expr, BinOp):
        return references(expr.left) | references(expr.right)
    if isinstance(expr, Call):
        found: set[str] = set()
        for argument in expr.args:
            found |= references(argument)
        return found
    return set()


def functions_in(expr: Expr) -> set[str]:
    """İfadede geçen fonksiyon adları."""

    if isinstance(expr, Call):
        found = {expr.fn}
        for argument in expr.args:
            found |= functions_in(argument)
        return found
    if isinstance(expr, BinOp):
        return functions_in(expr.left) | functions_in(expr.right)
    return set()


# --- Değerlendirme -----------------------------------------------------------

def evaluate(
    expr: Expr,
    frame: pd.DataFrame | dict | None = None,
    scalar: Callable[[str], float] | None = None,
):
    """İfadeyi bir veri çerçevesi (vektör) veya skalerler üzerinde hesaplar."""

    def again(node: Expr):
        return evaluate(node, frame, scalar)

    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, Var):
        if frame is None:
            raise ValueError(f"'{expr.name}' için veri çerçevesi gerekir.")
        return np.asarray(frame[expr.name], dtype=float)
    if isinstance(expr, Ref):
        if scalar is None:
            raise ValueError(f"'{expr.name}' skaleri için hesaplanmış skalerler gerekir.")
        return scalar(expr.name)
    if isinstance(expr, BinOp):
        left = again(expr.left)
        right = again(expr.right)
        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            return left * right
        if expr.op == "/":
            return left / right
        return left**right
    if isinstance(expr, Call):
        values = [again(argument) for argument in expr.args]
        if expr.fn == "neg":
            return -values[0]
        if expr.fn == "log":
            return np.log(values[0])
        if expr.fn == "exp":
            return np.exp(values[0])
        if expr.fn == "sqrt":
            return np.sqrt(values[0])
        if expr.fn == "abs":
            return np.abs(values[0])
        if expr.fn == "round":
            return np.rint(values[0])
        if expr.fn == "roundto":
            return np.round(values[0], int(values[1]))
        if expr.fn == "floor":
            return np.floor(values[0])
        if expr.fn == "normcdf":
            return stats.norm.cdf(values[0])
        if expr.fn == "normpdf":
            return stats.norm.pdf(values[0])
        if expr.fn == "norminv":
            return stats.norm.ppf(values[0])
        if expr.fn == "cumprod":
            return np.cumprod(values[0])
        if expr.fn == "cummean":
            return np.cumsum(values[0]) / np.arange(1, len(values[0]) + 1)
        if expr.fn == "seq":
            return np.arange(1, len(values[0]) + 1).astype(float)
        if expr.fn == "factorial":
            return float(math.factorial(int(values[0])))
        if expr.fn == "comb":
            return float(math.comb(int(values[0]), int(values[1])))
        if expr.fn == "perm":
            return float(math.perm(int(values[0]), int(values[1])))
        if expr.fn == "dbinom":
            return stats.binom.pmf(*values)
        if expr.fn == "pbinom":
            return stats.binom.cdf(*values)
        if expr.fn == "dpois":
            return stats.poisson.pmf(*values)
        if expr.fn == "ppois":
            return stats.poisson.cdf(*values)
        if expr.fn == "dhyper":
            return stats.hypergeom.pmf(*values)  # (x, N, r, n): scipy'de (k, M, n, N) sırası aynıdır
        if expr.fn == "phyper":
            return stats.hypergeom.cdf(*values)
        if expr.fn == "dnorm":
            return stats.norm.pdf(*values)
        if expr.fn == "tcdf":
            return stats.t.cdf(*values)
        if expr.fn == "tsf":
            return stats.t.sf(*values)
        if expr.fn == "tinv":
            return stats.t.ppf(*values)
        if expr.fn == "fsf":
            return stats.f.sf(*values)
        if expr.fn == "finv":
            return stats.f.ppf(*values)
        if expr.fn in COMPARISONS:
            left, right = np.asarray(values[0]), np.asarray(values[1])
            outcome = {
                "le": left <= right, "lt": left < right, "ge": left >= right,
                "gt": left > right, "eq": left == right, "ne": left != right,
            }[expr.fn]
            return np.where(outcome, 1.0, 0.0)
        if expr.fn == "minimum":
            return np.minimum(values[0], values[1])
        return np.maximum(values[0], values[1])
    raise TypeError(f"Tanınmayan ifade düğümü: {type(expr).__name__}")


# --- Dile çevirme ------------------------------------------------------------

@dataclass(frozen=True)
class Dialect:
    """Bir hedef dilin ifade sözdizimi.

    ``functions`` değerleri ya bir ad (``np.exp`` → ``np.exp(x)``) ya da ``{0}``
    yer tutuculu bir kalıptır (``as.numeric({0} > {1})``).
    """

    variable: Callable[[str], str]
    functions: dict[str, str]
    power: str
    scalar: Callable[[str], str] | None = None


def format_number(value: float) -> str:
    """Koddaki sayı yazımı: tam sayı ondalıksız; çok büyük tam sayı (≥ 10¹⁶, ör. ölçek 10⁴¹) bilimsel gösterimle
    (``1e+41``; Python ve R'de aynı sayı)."""

    if float(value).is_integer() and abs(value) < 1e16:
        return str(int(value))
    return repr(float(value))


def _is_negation(expr: Expr) -> bool:
    """İşlem içindeki tek terimli eksi (``neg``) parantez içinde yazılır: (−a)², b − (−a)."""

    return isinstance(expr, Call) and expr.fn == "neg"


def _precedence(expr: Expr) -> int:
    return _PRECEDENCE[expr.op] if isinstance(expr, BinOp) else _ATOM


def render(expr: Expr, dialect: Dialect) -> str:
    """İfadeyi gereksiz parantez olmadan hedef dile yazar."""

    if isinstance(expr, Const):
        return format_number(expr.value)
    if isinstance(expr, Var):
        return dialect.variable(expr.name)
    if isinstance(expr, Ref):
        return dialect.scalar(expr.name) if dialect.scalar is not None else expr.name
    if isinstance(expr, Call):
        arguments = [render(argument, dialect) for argument in expr.args]
        if expr.fn == "neg":  # iki dilde de aynı yazım; bileşik terim parantez içinde
            return f"-({arguments[0]})" if isinstance(expr.args[0], BinOp) else f"-{arguments[0]}"
        spec = dialect.functions[expr.fn]
        if "{0}" in spec:
            return spec.format(*arguments)
        return f"{spec}({', '.join(arguments)})"
    parent = _PRECEDENCE[expr.op]
    left = render(expr.left, dialect)
    right = render(expr.right, dialect)
    if _is_negation(expr.left) or (expr.op == "^" and isinstance(expr.left, Const) and expr.left.value < 0):
        left = f"({left})"  # (−a)², (−2)²: üs, tek terimli eksiden önce uygulanırdı
    if _is_negation(expr.right):
        right = f"({right})"
    if _precedence(expr.left) < parent or (
        expr.op == "^" and _precedence(expr.left) == parent
    ):
        left = f"({left})"
    if _precedence(expr.right) < parent or (
        expr.op in ("-", "/") and _precedence(expr.right) == parent
    ):
        right = f"({right})"
    symbol = dialect.power if expr.op == "^" else expr.op
    return f"{left} {symbol} {right}"
