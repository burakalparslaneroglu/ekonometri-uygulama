import pandas as pd

from core.data_registry import load_dataset
from core.model_utils import fit_simple_ols
from core.question_engine import KONU03_QUESTION_TYPES, generate_question, question_type_for
from core.session_utils import ANSWER_VISIBLE_KEY, QUESTION_INDEX_KEY, next_question, synchronize_question_state


def _result():
    return fit_simple_ols(pd.DataFrame({"y": [1.0, 3.0, 5.0, 7.0], "x": [0.0, 1.0, 2.0, 3.0]}), "y", "x")


def test_question_is_reproducible_and_cycles_types() -> None:
    result = _result()
    first = generate_question(result, "toy:y:x:observation:2", 2, observation_position=2)
    second = generate_question(result, "toy:y:x:observation:2", 2, observation_position=2)
    assert first == second
    types = [question_type_for("toy:y:x", index) for index in range(len(KONU03_QUESTION_TYPES))]
    assert set(types) == set(KONU03_QUESTION_TYPES)
    assert "r_squared" not in KONU03_QUESTION_TYPES
    assert {"residual", "residual_sign", "relative_to_line", "absolute_residual"}.issubset(KONU03_QUESTION_TYPES)
    assert question_type_for("toy:y:x", 0) != question_type_for("toy:y:x", 1)


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


def test_wage1_and_hprice1_support_prediction_and_observation_questions() -> None:
    for dataset_key, dependent, explanatory in (("wage1", "wage", "educ"), ("hprice1", "price", "sqrft")):
        result = fit_simple_ols(load_dataset(dataset_key), dependent, explanatory)
        model_id = f"{dataset_key}:{dependent}:{explanatory}"
        questions = [
            generate_question(result, f"{model_id}:observation:0", index, observation_position=0)
            for index in range(len(KONU03_QUESTION_TYPES))
        ]
        generated_types = {question.question_type for question in questions}
        assert {"slope", "intercept", "prediction", "residual", "residual_sign"}.issubset(generated_types)
