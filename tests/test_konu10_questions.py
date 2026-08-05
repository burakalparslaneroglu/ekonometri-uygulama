"""Konu 10 soru belirlenim denetimi."""
from core.categorical_regression_utils import add_categorical_columns,compare_raw_and_controlled_dummy
from core.data_registry import load_dataset
from core.konu10_questions import Konu10QuestionContext,generate_konu10_question
def test_questions_are_deterministic() -> None:
    c=Konu10QuestionContext(compare_raw_and_controlled_dummy(add_categorical_columns(load_dataset("wage1"),"wage1")),.1)
    assert generate_konu10_question(c,"x",2)==generate_konu10_question(c,"x",2)
