"""Konu 07 soru motoru davranışı."""

from core.data_registry import load_dataset
from core.konu07_questions import Konu07QuestionContext, generate_konu07_question
from core.regression_inference_utils import coefficient_test, fit_ols_inference


def test_questions_are_deterministic_and_tied_to_test_result() -> None:
    """Aynı bağlam aynı soruyu, değişen indeks başka türü üretir."""
    result = fit_ols_inference(load_dataset("wage1"), "wage", ("educ", "exper", "tenure"))
    inference = coefficient_test(result, "educ", null_value=0.5)
    context = Konu07QuestionContext("wage:educ:half", "W7-W", result, inference, "Eğitim", "Saatlik ücret", "yıl")
    first, repeated, second = (generate_konu07_question(context, index) for index in (0, 0, 1))
    assert first == repeated
    assert first.question_type != second.question_type
    all_questions = [generate_konu07_question(context, index) for index in range(57)]
    all_text = " ".join(question.prompt + " " + question.answer for question in all_questions)
    assert "0.500" in all_text


def test_t_question_preserves_positive_and_negative_direction() -> None:
    """Soru çözümü t-istatistiğinin yönünü öğrenciye açıkça söyler."""
    result = fit_ols_inference(load_dataset("wage1"), "wage", ("educ", "exper", "tenure"))
    positive = Konu07QuestionContext("positive", "W7-W", result, coefficient_test(result, "educ"), "Eğitim", "Ücret", "yıl")
    negative = Konu07QuestionContext("negative", "W7-W", result, coefficient_test(result, "educ", null_value=0.75), "Eğitim", "Ücret", "yıl")
    positive_answer = next(generate_konu07_question(positive, index).answer for index in range(57) if generate_konu07_question(positive, index).question_type == "test_istatistigi")
    negative_answer = next(generate_konu07_question(negative, index).answer for index in range(57) if generate_konu07_question(negative, index).question_type == "test_istatistigi")
    assert "üzerindedir" in positive_answer
    assert "altındadır" in negative_answer
