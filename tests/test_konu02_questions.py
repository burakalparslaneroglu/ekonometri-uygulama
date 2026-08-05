from core.data_registry import load_dataset
from core.group_comparison_utils import compare_two_groups
from core.scenario_registry import KONU02_QUESTION_TYPES, generate_konu02_question
from core.session_utils import next_question, question_state_keys, reveal_answer, synchronize_active_topic, synchronize_question_state


def _comparison():
    return compare_two_groups(
        load_dataset("jtrain2"), "train", "re78", control_value=0, treatment_value=1,
        control_label="Kontrol grubu", treatment_label="Eğitim grubu", unit="bin ABD doları",
    )


def test_konu02_questions_are_deterministic_and_cover_registered_types() -> None:
    comparison = _comparison()
    model_id = "konu02:wage1:veri_nedensellik"
    assert generate_konu02_question(model_id, 4, comparison) == generate_konu02_question(model_id, 4, comparison)
    generated = {generate_konu02_question(model_id, index, comparison).question_type for index in range(len(KONU02_QUESTION_TYPES))}
    assert generated == set(KONU02_QUESTION_TYPES)


def test_konu02_question_state_resets_for_dataset_and_topic_change() -> None:
    state = {}
    assert synchronize_active_topic(state, "konu02")
    synchronize_question_state(state, "konu02:wage1", topic_id="konu02")
    index_key, _, answer_key = question_state_keys("konu02")
    assert state[answer_key] is False
    reveal_answer(state, topic_id="konu02")
    next_question(state, topic_id="konu02")
    assert state[index_key] == 1
    assert state[answer_key] is False
    assert synchronize_question_state(state, "konu02:wagepan", topic_id="konu02")
    assert state[index_key] == 0
    assert synchronize_active_topic(state, "konu01")
    assert synchronize_active_topic(state, "konu02")
    assert index_key not in state
