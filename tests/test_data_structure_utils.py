import pandas as pd
import pytest

from core.data_registry import load_dataset
from core.data_structure_utils import has_repeated_units, is_panel_structure, periods_per_unit, sort_time_series, summarize_panel_balance


def test_wagepan_repeats_and_is_balanced_for_every_period() -> None:
    frame = load_dataset("wagepan")
    summary = summarize_panel_balance(frame, "nr", "year")
    assert has_repeated_units(frame, "nr")
    assert is_panel_structure(frame, "nr", "year")
    assert summary.unit_count == 545
    assert summary.period_count == 8
    assert summary.observations == 4360
    assert summary.balanced
    assert summary.expected_periods == tuple(range(1980, 1988))
    assert periods_per_unit(frame, "nr", "year").eq(8).all()


def test_id_and_year_columns_without_repeated_ids_are_not_panel_evidence() -> None:
    frame = pd.DataFrame({"id": [1, 2, 3], "year": [2020, 2020, 2020]})
    assert not has_repeated_units(frame, "id")
    assert not is_panel_structure(frame, "id", "year")


def test_time_series_is_sorted_and_duplicate_time_is_rejected() -> None:
    frame = load_dataset("phillips")
    ordered = sort_time_series(frame.sample(frac=1, random_state=202602), "year")
    assert ordered["year"].tolist() == sorted(frame["year"].tolist())
    with pytest.raises(ValueError, match="yinelenen"):
        sort_time_series(pd.DataFrame({"year": [2000, 2000]}), "year")
