from core.scenario_registry import KONU01_QUESTION_TYPES, generate_konu01_question
from core.session_utils import (
    next_question,
    question_state_keys,
    reveal_answer,
    synchronize_active_topic,
    synchronize_question_state,
)


def test_konu01_questions_are_deterministic_and_cover_registered_types() -> None:
    model_id = "konu01:wage1:ampirik_arastirma"
    first = generate_konu01_question(model_id, 3)
    assert first == generate_konu01_question(model_id, 3)
    types = {generate_konu01_question(model_id, index).question_type for index in range(len(KONU01_QUESTION_TYPES))}
    assert types == set(KONU01_QUESTION_TYPES)


def test_konu01_question_state_resets_when_topic_or_model_changes() -> None:
    state = {}
    assert synchronize_active_topic(state, "konu01")
    synchronize_question_state(state, "konu01:wage1", topic_id="konu01")
    index_key, _, answer_key = question_state_keys("konu01")
    reveal_answer(state, topic_id="konu01")
    next_question(state, topic_id="konu01")
    assert state[index_key] == 1
    assert state[answer_key] is False
    assert synchronize_question_state(state, "konu01:scenario-change", topic_id="konu01")
    assert state[index_key] == 0
    assert state[answer_key] is False
    next_question(state, topic_id="konu01")
    assert synchronize_active_topic(state, "konu03")
    assert synchronize_active_topic(state, "konu01")
    assert index_key not in state
    assert answer_key not in state
