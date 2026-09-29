"""Eski sayfaların (Konu 10–11) kullandığı öğrenci sayı biçimleri."""

import math

import pytest

from core.model_utils import format_numerical_difference, format_student_number


def test_student_number_has_no_scientific_notation_and_trims_zeros() -> None:
    assert format_student_number(0.5414) == "0.5414"
    assert format_student_number(-2.873, decimals=3) == "-2.873"
    assert format_student_number(1234567.25, decimals=2) == "1234567.25"
    assert format_student_number(3e-11) == "0"
    assert format_student_number(0.000001234, decimals=9) == "0.000001234"


def test_student_number_rejects_non_finite_values_and_negative_tolerance() -> None:
    for value in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError, match="sonlu"):
            format_student_number(value)
    with pytest.raises(ValueError, match="negatif"):
        format_student_number(1.0, zero_tolerance=-1e-12)


def test_numerical_difference_names_the_tolerance() -> None:
    assert format_numerical_difference(2e-13) == "sayısal tolerans içinde 0"
    assert format_numerical_difference(-2e-13) == "sayısal tolerans içinde 0"
    assert format_numerical_difference(0.0125) == "0.0125"
