from core.question_engine import cycle_question_type
from core.session_utils import ANSWER_VISIBLE_KEY, QUESTION_INDEX_KEY, next_question, synchronize_question_state

TYPES = ("a", "b", "c", "d")


def test_question_types_cycle_deterministically_and_cover_every_type() -> None:
    sequence = [cycle_question_type("wage1:wage:educ", index, TYPES, namespace="konu05") for index in range(8)]
    again = [cycle_question_type("wage1:wage:educ", index, TYPES, namespace="konu05") for index in range(8)]
    assert sequence == again
    assert set(sequence[:4]) == set(TYPES)
    assert sequence[:4] == sequence[4:]
    assert all(first != second for first, second in zip(sequence, sequence[1:]))


def test_question_state_resets_on_dataset_or_variable_pair_change() -> None:
    state = {}
    assert synchronize_question_state(state, "wage1:wage:educ")
    state[ANSWER_VISIBLE_KEY] = True
    next_question(state)
    assert state[QUESTION_INDEX_KEY] == 1
    assert synchronize_question_state(state, "wage1:wage:exper")
    assert state[QUESTION_INDEX_KEY] == 0
    assert state[ANSWER_VISIBLE_KEY] is False
    next_question(state)
    assert synchronize_question_state(state, "hprice1:price:sqrft")
    assert state[QUESTION_INDEX_KEY] == 0
    state[ANSWER_VISIBLE_KEY] = True
    assert synchronize_question_state(state, "hprice1:price:sqrft:observation:1")
    assert state[ANSWER_VISIBLE_KEY] is False
