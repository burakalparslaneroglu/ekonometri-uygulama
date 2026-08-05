"""Konu 08 soru motoru denetimi."""

from core.data_registry import load_dataset
from core.joint_inference_utils import nested_exclusion_f_test
from core.konu08_questions import Konu08QuestionContext, generate_konu08_question


def test_konu08_questions_are_deterministic() -> None:
    """Aynı bağlam soruyu sabit, indeks ise türü değişken tutar."""
    test = nested_exclusion_f_test(load_dataset("wage1"), "wage", ("educ", "exper", "tenure"), ("educ",))
    context = Konu08QuestionContext("wage", test)
    assert generate_konu08_question(context, 0) == generate_konu08_question(context, 0)
    assert len({generate_konu08_question(context, index).question_type for index in range(45)}) == 45
