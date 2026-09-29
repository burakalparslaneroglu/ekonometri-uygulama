"""Öğrencinin yazdığı formülü güvenli biçimde okur, önizler ve doğru cevapla karşılaştırır.

Güvenlik: ``eval`` kullanılmaz. Metin Python'un sözdizimi ağacına çevrilir ve yalnız
izin verilen düğümler (sayılar, tanımlı semboller, + − × ÷ ^, exp/log/sqrt)
``core.labs.expr`` ifadelerine dönüştürülür; başka her şey reddedilir. Sembol olarak tanımlı değilse ``e``
Euler sayısıdır (``e^l`` = ``exp(l)``).

Eşdeğerlik: iki ifade, sembollerin rastgele seçilmiş değerlerinde sayısal olarak
karşılaştırılır. Böylece ``100g/n`` ile ``100*(g/n)`` veya ``g/n*100`` aynı kabul edilir.
"""

from __future__ import annotations

import ast
import math
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.labs import expr as E

FUNCTIONS = {"exp": "exp", "log": "log", "ln": "log", "sqrt": "sqrt"}


class FormulaError(ValueError):
    """Formül okunamadığında öğrenciye gösterilecek açıklamayla."""


@dataclass(frozen=True)
class Symbol:
    name: str
    latex: str
    meaning: str
    low: float = 0.5
    high: float = 3.0
    aliases: tuple[str, ...] = ()


def bar_aliases(letter: str) -> tuple[str, ...]:
    """Ortalama için öğrencinin yazabileceği biçimler: X̄, \\bar{X}, \\overline{X}, X_bar (büyük ve küçük harf)."""

    forms: list[str] = []
    for item in (letter.upper(), letter.lower()):
        forms += [f"{item}\u0304", f"{item}\u0305", f"\\bar{item}", f"\\bar {item}", f"\\overline{item}",
                  f"\\overline {item}", f"{item}_bar"]
        composed = unicodedata.normalize("NFC", f"{item}\u0304")  # Ȳ, ȳ tek karakterdir; X̄ değildir
        if len(composed) == 1:
            forms.append(composed)
    return tuple(forms)


def beta_aliases(index: int) -> tuple[str, ...]:
    """Anakütle parametresi için biçimler: β₀, β_0, \\beta_0, beta0."""

    number, subscript = str(index), "₀₁₂₃"[index]
    return (f"\\beta_{number}", f"\\beta{number}", f"β_{number}", f"β{number}", f"β{subscript}",
            f"beta_{number}", f"beta{number}")


def beta_hat_aliases(index: int) -> tuple[str, ...]:
    """Tahmin için biçimler: β̂₀, \\hat{\\beta}_0, \\widehat{\\beta}_0, b0hat. Süslü parantezler okunmadan önce
    silindiği için ``\\hat{\\beta}_0`` metni ``\\hat\\beta_0`` olarak eşleşir."""

    number, subscript, hat = str(index), "₀₁₂₃"[index], "\u0302"
    return (f"\\hat\\beta_{number}", f"\\widehat\\beta_{number}", f"\\hat\\beta{number}",
            f"\\widehat\\beta{number}", f"β{hat}_{number}", f"β{hat}{number}", f"β{hat}{subscript}",
            f"betahat_{number}", f"betahat{number}", f"beta{number}hat", f"bhat{number}", f"b{number}hat")


_TOKEN = re.compile(r"\s*(?:(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|([A-Za-z_]\w*)|(\*\*|[-+*/()]))")


_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
EULER = "e"
"""Sembol olarak tanımlı değilse ``e`` Euler sayısıdır."""


def _normalize(text: str) -> str:
    text = text.strip().translate(_SUBSCRIPTS)
    for old, new in (("−", "-"), ("–", "-"), ("—", "-"), ("×", "*"), ("·", "*"), ("÷", "/"), (":", "/"),
                     ("½", "(1/2)"), ("²", "^2"), ("³", "^3"), ("^", "**"), ("[", "("), ("]", ")"),
                     ("√", " sqrt ")):
        text = text.replace(old, new)
    text = re.sub(r"(?<=[A-Za-z_)])\.(?=[A-Za-z_(])", "*", text)  # okuldaki çarpma noktası: m.x, N.T
    return re.sub(r"(?<=\d),(?=\d)", ".", text)


def _resolve(name: str, known: set[str]) -> str:
    """Tanımlı bir sembolün büyük/küçük harf ya da alt çizgi farkıyla yazımı (``N1`` → ``n1``, ``B_1`` → ``b1``),
    yalnız tek bir sembole uyuyorsa. Önce yalnız harf farkı denenir; birden çok sembole uyan yazım olduğu gibi kalır."""

    if name in known:
        return name
    for fold in (str.lower, lambda text: text.replace("_", "").lower()):
        matches = [symbol for symbol in known if fold(symbol) == fold(name)]
        if matches:
            return matches[0] if len(matches) == 1 else name
    return name


def _split_products(name: str, known: set[str]) -> list[str] | None:
    """Tanımsız bir ad tanımlı sembollerin ardışık yazımıysa çarpım olarak okunur: ``mx`` → ``m``, ``x``;
    ``n1m1`` → ``n1``, ``m1``; ``b_1c`` → ``b1``, ``c``. Bölme en uzun sembolden başlar; tek bir okunuş bulunamazsa ad
    olduğu gibi kalır."""

    if _resolve(name, known) in known or name.lower() in FUNCTIONS:
        return None
    lookup = {symbol.lower(): symbol for symbol in known}
    if len(lookup) != len(known):  # yalnız büyük/küçük harfle ayrılan semboller varsa bölme harf duyarlıdır
        lookup = {symbol: symbol for symbol in known}
        text = name
    else:
        text = name.lower()
    longest = sorted(lookup, key=len, reverse=True)

    def split(rest: str) -> list[str] | None:
        if not rest:
            return []
        for symbol in longest:
            if rest.startswith(symbol):
                tail = split(rest[len(symbol):])
                if tail is not None:
                    return [lookup[symbol], *tail]
        return None

    parts = split(text)
    if parts is None and "_" in text:  # LaTeX alt indisleri: b_1c → b1, c
        parts = split(text.replace("_", ""))
    return parts if parts is not None and len(parts) > 1 else None


def _with_implicit_products(text: str, known: set[str] = frozenset()) -> str:
    """``100(exp(b)-1)`` → ``100*(exp(b)-1)``, ``2b`` → ``2*b``; ``known`` sembollerle ``mx`` → ``m*x``."""

    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if not match or match.end() == position:
            if text[position:].strip() == "":
                break
            raise FormulaError(f"Tanınmayan karakter: '{text[position:].strip()[0]}'")
        number, name, operator = match.groups()
        if number is not None:
            tokens.append(("sayi", number))
        elif name is not None:
            split = _split_products(name, known)
            tokens.extend(("ad", part) for part in (split or [_resolve(name, known)]))
        else:
            tokens.append(("islem", operator))
        position = match.end()
    pieces: list[str] = []
    for index, (kind, value) in enumerate(tokens):
        if index:
            previous_kind, previous = tokens[index - 1]
            left_closed = previous_kind == "sayi" or previous == ")" or (
                previous_kind == "ad" and previous.lower() not in FUNCTIONS
            )
            right_open = kind in ("sayi", "ad") or value == "("
            if left_closed and right_open:
                pieces.append("*")
        pieces.append(value)
    prepared = " ".join(pieces)
    # "ln x1", "\\ln x_1": parantezsiz fonksiyon yalnız hemen ardından gelen tek sembole ya da sayıya uygulanır.
    return re.sub(r"\b(exp|log|ln|sqrt)\s+(?!\()([A-Za-z_]\w*|\d+(?:\.\d+)?)", r"\1 ( \2 )", prepared)


def _convert(node: ast.AST, symbols: dict[str, Symbol]) -> E.Expr:
    if isinstance(node, ast.Expression):
        return _convert(node.body, symbols)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return E.Const(float(node.value))
    if isinstance(node, ast.Name):
        if node.id in symbols:
            return E.Var(node.id)
        if node.id in (EULER, EULER.upper()):
            return E.Const(math.e)
        allowed = ", ".join(symbols) or "yok"
        raise FormulaError(f"'{node.id}' tanımlı bir sembol değil. Kullanılabilecek semboller: {allowed}.")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _convert(node.operand, symbols)
        return E.BinOp("-", E.Const(0.0), operand) if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp):
        operators = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "^"}
        symbol = operators.get(type(node.op))
        if symbol is None:
            raise FormulaError("Yalnız + − * / ^ işlemleri kullanılabilir.")
        return E.BinOp(symbol, _convert(node.left, symbols), _convert(node.right, symbols))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        function = FUNCTIONS.get(node.func.id.lower())
        if function is None or len(node.args) != 1:
            raise FormulaError("Fonksiyon olarak yalnız exp(), ln() (ya da log()) ve sqrt() kullanılabilir.")
        return E.Call(function, (_convert(node.args[0], symbols),))
    raise FormulaError("Bu ifade okunamadı. Yalnız sayılar, semboller ve + − * / ^ ( ) kullanın.")


def _brace_group(text: str, start: int) -> tuple[str, int] | None:
    """``text[start]`` bir ``{`` ise eşleşen ``}``'e kadarki içerik ve ondan sonraki konum."""

    while start < len(text) and text[start] == " ":
        start += 1
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:index], index + 1
    return None


_LATEX_COMMANDS = (("\\left", ""), ("\\right", ""), ("\\cdot", "*"), ("\\times", "*"), ("\\ln", " ln"),
                   ("\\log", " log"), ("\\exp", " exp"), ("\\,", " "), ("\\;", " "), ("\\!", ""),
                   ("\\ ", " "))
_TEXT_COMMANDS = re.compile(r"\\(?:text|textrm|mathrm|mathit|operatorname)(?=\s*\{)")
"""Metin biçimi komutları (``\\text{UR}``, ``\\mathrm{se}``) içerikleri korunarak silinir: ``R^2_{\\text{UR}}`` ile
``R^2_{UR}`` aynı okunur."""


def _latex_lite(text: str) -> str:
    """Öğrencinin LaTeX alışkanlıkları: ``\\frac{a}{b}`` → ``((a)/(b))``, ``\\sqrt{a}`` → ``sqrt(a)``,
    ``\\cdot`` → ``*``, ``\\left(`` → ``(``, ``\\ln`` → ``ln``."""

    while True:
        match = re.search(r"\\[dt]?frac(?=\s*\{)", text)
        first = _brace_group(text, match.end()) if match else None
        second = _brace_group(text, first[1]) if first else None
        if second is None:
            break
        text = f"{text[:match.start()]}(({first[0]})/({second[0]})){text[second[1]:]}"
    while True:
        match = re.search(r"\\sqrt(?=\s*\{)", text)
        group = _brace_group(text, match.end()) if match else None
        if group is None:
            break
        text = f"{text[:match.start()]} sqrt({group[0]}){text[group[1]:]}"  # "s\\sqrt{…}" → "s sqrt(…)"
    for command, replacement in _LATEX_COMMANDS:
        text = text.replace(command, replacement)
    return _TEXT_COMMANDS.sub("", text)


def _exponent_groups(text: str) -> str:
    """Çok terimli üs süslü parantezler silinmeden önce parantezle korunur: ``e^{a - c s}`` → ``e^(a - c s)``. Tek
    terimli üs (``x^{2}``, ``x^{b}``, ``R^{2}_{UR}``) olduğu gibi kalır."""

    start = 0
    while (index := text.find("^", start)) >= 0:
        group = _brace_group(text, index + 1)
        if group is not None and not re.fullmatch(r"\s*[\w.,]+\s*", group[0]):
            text = f"{text[:index + 1]}({group[0]}){text[group[1]:]}"
        start = index + 1
    return text


def parse(text: str, symbols: tuple[Symbol, ...]) -> E.Expr:
    """Metni güvenli biçimde ifade ağacına çevirir; okunamazsa ``FormulaError``.

    Öğrenci denklemin sol tarafını da yazarsa (``r = …``) yalnız son ``=``, ``≈`` ya da ``\\approx`` işaretinden
    sonrası okunur.
    """

    if not text or not text.strip():
        raise FormulaError("Boş ifade.")
    if len(text) > 200:
        raise FormulaError("İfade çok uzun.")
    text = re.split(r"=|≈|\\approx", text)[-1]
    if not text.strip():
        raise FormulaError("Eşittir işaretinden sonra bir ifade yazın.")
    text = _exponent_groups(_latex_lite(text))
    text = text.replace("{", "").replace("}", "")  # LaTeX yazımı: y_{1}, e^{l}
    text = re.sub(r"([A-Za-z])\^(\d+)_([A-Za-z0-9]+)", r"\1_\3^\2", text)  # s^2_x → s_x^2
    replacements = sorted(
        ((alias, symbol.name) for symbol in symbols for alias in symbol.aliases),
        key=lambda pair: -len(pair[0]),
    )
    for alias, name in replacements:
        text = text.replace(alias, f" {name} ")
    prepared = _with_implicit_products(_normalize(text), {symbol.name for symbol in symbols})
    try:
        tree = ast.parse(prepared, mode="eval")
    except SyntaxError as error:
        raise FormulaError("Parantezleri ve işlemleri kontrol edin.") from error
    return _convert(tree, {symbol.name: symbol for symbol in symbols})


def equivalent(candidate: E.Expr, answer: E.Expr, symbols: tuple[Symbol, ...], draws: int = 16) -> bool:
    """İki ifadenin sembollerin rastgele değerlerinde aynı sonucu verip vermediği."""

    rng = np.random.default_rng(20260926)
    frame = pd.DataFrame(
        {symbol.name: rng.uniform(symbol.low, symbol.high, size=draws) for symbol in symbols}
    )
    with np.errstate(all="ignore"):
        try:
            got = np.asarray(E.evaluate(candidate, frame), dtype=float) * np.ones(draws)
            expected = np.asarray(E.evaluate(answer, frame), dtype=float) * np.ones(draws)
        except (ValueError, KeyError, ZeroDivisionError, OverflowError):
            return False
    usable = np.isfinite(expected)
    if not usable.any() or not np.isfinite(got[usable]).all():
        return False
    return bool(np.allclose(got[usable], expected[usable], rtol=1e-9, atol=1e-12))


def _latex_number(value: float) -> str:
    if value == math.e:
        return "e"
    text = f"{value:.10g}"
    return text.replace(".", "{,}")


_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}


def latex(expression: E.Expr, symbols: tuple[Symbol, ...]) -> str:
    """Önizleme için LaTeX; öğrenci yazdığının nasıl okunduğunu görür."""

    names = {symbol.name: symbol.latex for symbol in symbols}

    def level(node: E.Expr) -> int:
        return _PRECEDENCE[node.op] if isinstance(node, E.BinOp) else 4

    def wrap(node: E.Expr, minimum: int) -> str:
        text = render(node)
        return f"\\left({text}\\right)" if level(node) < minimum else text

    def render(node: E.Expr) -> str:
        if isinstance(node, E.Const):
            return _latex_number(node.value)
        if isinstance(node, E.Var):
            return names.get(node.name, node.name)
        if isinstance(node, E.Call):
            inner = render(node.args[0])
            if node.fn == "exp":
                return f"e^{{{inner}}}"
            if node.fn == "sqrt":
                return f"\\sqrt{{{inner}}}"
            return f"\\ln\\left({inner}\\right)"  # notlardaki gösterim: doğal logaritma ln
        if isinstance(node, E.BinOp):
            if node.op == "-" and isinstance(node.left, E.Const) and node.left.value == 0:
                return f"-{wrap(node.right, 2)}"
            if node.op == "/":
                return f"\\frac{{{render(node.left)}}}{{{render(node.right)}}}"
            if node.op == "^":
                return f"{{{wrap(node.left, 4)}}}^{{{render(node.right)}}}"
            if node.op == "*":
                return f"{wrap(node.left, 2)} \\cdot {wrap(node.right, 2)}"
            right = wrap(node.right, 2) if node.op == "-" else render(node.right)
            return f"{render(node.left)} {node.op} {right}"
        raise TypeError(type(node).__name__)

    return render(expression)
