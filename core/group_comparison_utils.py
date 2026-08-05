"""İki grubun betimsel karşılaştırması için saf yardımcılar."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GroupComparison:
    """İki grubun sayısını, ortalamasını ve ortalama farkını tutar."""

    group_column: str
    outcome_column: str
    control_label: str
    treatment_label: str
    unit: str
    control_count: int
    treatment_count: int
    control_mean: float
    treatment_mean: float
    difference: float


def compare_two_groups(
    frame: pd.DataFrame,
    group_column: str,
    outcome_column: str,
    *,
    control_value: int | float,
    treatment_value: int | float,
    control_label: str,
    treatment_label: str,
    unit: str,
) -> GroupComparison:
    """İki tanımlı grup için gözlem sayısı, ortalama ve eğitim-kontrol farkını hesaplar."""
    missing = [column for column in (group_column, outcome_column) if column not in frame.columns]
    if missing:
        raise ValueError(f"Veri setinde bulunamayan sütun: {', '.join(missing)}")
    if control_value == treatment_value:
        raise ValueError("Kontrol ve eğitim grubu değerleri farklı olmalıdır.")
    prepared = frame.loc[:, [group_column, outcome_column]].dropna().copy()
    prepared[outcome_column] = pd.to_numeric(prepared[outcome_column], errors="coerce")
    prepared = prepared.dropna(subset=[outcome_column])
    allowed = prepared[group_column].isin((control_value, treatment_value))
    if not allowed.all():
        invalid = sorted(prepared.loc[~allowed, group_column].unique().tolist())
        raise ValueError(f"Beklenmeyen grup değeri: {invalid}")
    control = prepared.loc[prepared[group_column] == control_value, outcome_column]
    treatment = prepared.loc[prepared[group_column] == treatment_value, outcome_column]
    if control.empty or treatment.empty:
        raise ValueError("Her iki grup için en az bir geçerli sonuç gözlemi gerekir.")
    control_mean = float(control.mean())
    treatment_mean = float(treatment.mean())
    return GroupComparison(
        group_column=group_column,
        outcome_column=outcome_column,
        control_label=control_label,
        treatment_label=treatment_label,
        unit=unit,
        control_count=int(control.size),
        treatment_count=int(treatment.size),
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        difference=treatment_mean - control_mean,
    )
