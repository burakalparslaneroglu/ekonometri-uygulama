"""Konu 09 soru motoru denetimi."""

from core.data_registry import load_dataset
from core.functional_form_utils import wage1_model_comparison
from core.konu09_questions import Konu09QuestionContext, generate_konu09_question


def test_konu09_questions_are_deterministic() -> None:
    """Fonksiyonel biçim soruları gerçek M1–M4 bağlamına bağlıdır."""
    context = Konu09QuestionContext("m4", wage1_model_comparison(load_dataset("wage1")))
    first = generate_konu09_question(context, 0)
    assert first == generate_konu09_question(context, 0)
    assert len({generate_konu09_question(context, index).question_type for index in range(45)}) == 45
