import pytest

from core.data_registry import load_dataset
from core.group_comparison_utils import compare_two_groups


def test_jtrain2_group_summary_uses_real_data_and_preserves_unit() -> None:
    result = compare_two_groups(
        load_dataset("jtrain2"),
        "train",
        "re78",
        control_value=0,
        treatment_value=1,
        control_label="Kontrol grubu",
        treatment_label="Eğitim grubu",
        unit="bin ABD doları",
    )
    assert (result.control_count, result.treatment_count) == (260, 185)
    assert result.control_mean == pytest.approx(4.5548022841, abs=1e-10)
    assert result.treatment_mean == pytest.approx(6.3491453572, abs=1e-10)
    assert result.difference == pytest.approx(1.7943430731, abs=1e-10)
    assert result.unit == "bin ABD doları"
    assert result.difference > 0
    assert not hasattr(result, "standard_error")
    assert not hasattr(result, "p_value")
