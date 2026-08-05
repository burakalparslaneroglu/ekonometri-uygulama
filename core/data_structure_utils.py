"""Ekonomik veri yapısını inceleyen Streamlit'ten bağımsız yardımcılar."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PanelBalanceSummary:
    """Bir panelin birim ve dönem kapsamını özetler."""

    unit_count: int
    period_count: int
    observations: int
    periods_per_unit: pd.Series
    balanced: bool
    expected_periods: tuple[object, ...]


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """İstenen sütunların veri çerçevesinde bulunduğunu doğrular."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Veri setinde bulunamayan sütun: {', '.join(missing)}")


def has_repeated_units(frame: pd.DataFrame, identifier: str) -> bool:
    """En az bir geçerli kimliğin birden çok satırda bulunup bulunmadığını döndürür."""
    _require_columns(frame, (identifier,))
    values = frame[identifier].dropna()
    if values.empty:
        raise ValueError(f"{identifier!r} sütununda geçerli kimlik yok.")
    return bool(values.duplicated().any())


def is_panel_structure(frame: pd.DataFrame, identifier: str, time_variable: str) -> bool:
    """Aynı birimlerin birden çok dönemde görünüp görünmediğini denetler."""
    _require_columns(frame, (identifier, time_variable))
    prepared = frame.loc[:, [identifier, time_variable]].dropna()
    if prepared.empty:
        raise ValueError("Panel kararı için kimlik ve dönem bilgisi gerekir.")
    distinct_periods = prepared[time_variable].nunique()
    repeated_periods = prepared.groupby(identifier, sort=False)[time_variable].nunique()
    return bool(distinct_periods > 1 and repeated_periods.gt(1).any())


def periods_per_unit(frame: pd.DataFrame, identifier: str, time_variable: str) -> pd.Series:
    """Her birim için benzersiz gözlem dönemi sayısını hesaplar."""
    _require_columns(frame, (identifier, time_variable))
    prepared = frame.loc[:, [identifier, time_variable]].dropna()
    if prepared.empty:
        raise ValueError("Kimlik ve dönem birlikte eksik olmayan gözlem içermelidir.")
    return prepared.groupby(identifier, sort=True)[time_variable].nunique().sort_index()


def summarize_panel_balance(frame: pd.DataFrame, identifier: str, time_variable: str) -> PanelBalanceSummary:
    """Her birimin dönem kümesini inceleyerek panelin dengeli olup olmadığını belirler."""
    _require_columns(frame, (identifier, time_variable))
    prepared = frame.loc[:, [identifier, time_variable]].dropna()
    if prepared.empty:
        raise ValueError("Panel özeti için kimlik ve dönem bilgisi gerekir.")
    if prepared.duplicated().any():
        raise ValueError("Bir kimlik-dönem birleşimi birden fazla kez bulunuyor.")
    counts = periods_per_unit(prepared, identifier, time_variable)
    expected_periods = tuple(sorted(prepared[time_variable].unique().tolist()))
    expected_set = set(expected_periods)
    unit_period_sets = prepared.groupby(identifier, sort=True)[time_variable].agg(lambda values: set(values))
    balanced = bool(unit_period_sets.map(lambda values: values == expected_set).all())
    return PanelBalanceSummary(
        unit_count=int(prepared[identifier].nunique()),
        period_count=len(expected_periods),
        observations=len(prepared),
        periods_per_unit=counts,
        balanced=balanced,
        expected_periods=expected_periods,
    )


def sort_time_series(frame: pd.DataFrame, time_variable: str) -> pd.DataFrame:
    """Benzersiz ve eksiksiz zaman indeksini doğrular, veriyi zamana göre sıralar."""
    _require_columns(frame, (time_variable,))
    if frame[time_variable].isna().any():
        raise ValueError(f"{time_variable!r} sütununda eksik zaman değeri var.")
    if frame[time_variable].duplicated().any():
        raise ValueError(f"{time_variable!r} sütununda yinelenen zaman değeri var.")
    return frame.sort_values(time_variable, kind="stable").reset_index(drop=True).copy()
