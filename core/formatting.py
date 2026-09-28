from __future__ import annotations


def format_percent(value: float, digits: int = 1) -> str:
    return f"%{100 * value:.{digits}f}".replace(".", ",")


def format_number(value: float, digits: int = 1) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.{digits}f}".replace(".", ",")
