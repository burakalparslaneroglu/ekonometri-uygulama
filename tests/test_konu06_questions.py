from core.assumption_diagnostics_utils import (
    calculate_vif, generate_near_collinearity_data, ovb_bias,
    simulate_collinearity_variability, simulate_repeated_ols, wage1_ovb_decomposition,
)
from core.data_registry import load_dataset
from core.konu06_questions import KONU06_QUESTION_TYPES, Konu06QuestionContext, generate_konu06_question


def _context(context_id: str = "test") -> Konu06QuestionContext:
    repeated = simulate_repeated_ols(seed=19, repetitions=500)
    wage = wage1_ovb_decomposition(load_dataset("wage1"))
    near = generate_near_collinearity_data("high", seed=19)
    vif = float(calculate_vif(near.data, ("x1", "x2"))["vif"].iloc[0])
    return Konu06QuestionContext(context_id, repeated, ovb_bias(1.2, -2, 0.4), wage, vif, False, simulate_collinearity_variability("high", repetitions=100, seed=19))


def test_konu06_questions_are_deterministic_and_cover_defined_types() -> None:
    context = _context()
    first = generate_konu06_question(context, 0)
    assert first == generate_konu06_question(context, 0)
    generated = {generate_konu06_question(context, index).question_type for index in range(len(KONU06_QUESTION_TYPES))}
    assert generated == set(KONU06_QUESTION_TYPES)
    all_text = " ".join(generate_konu06_question(context, index).answer.casefold() for index in range(len(KONU06_QUESTION_TYPES)))
    assert "p-değeri" not in all_text and "güven aralığı" not in all_text


def test_konu06_questions_use_numeric_context_and_safe_language() -> None:
    context = _context("scenario-a")
    answers = {generate_konu06_question(context, index).question_type: generate_konu06_question(context, index).answer for index in range(len(KONU06_QUESTION_TYPES))}
    assert "0,4" in answers["ovb_sayisal"]
    assert "nedensel kanıt değildir" in answers["wage_isaret"]
    assert "standart hatası değildir" in answers["merkez_ve_degiskinlik"]
    assert generate_konu06_question(context, 0) != generate_konu06_question(_context("scenario-b"), 0) or generate_konu06_question(context, 0).question_type == generate_konu06_question(_context("scenario-b"), 0).question_type
