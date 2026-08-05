from core.data_registry import konu05_model_specs, load_dataset, variable_metadata
from core.konu05_questions import KONU05_QUESTION_TYPES, generate_konu05_question
from core.multiple_regression_utils import fit_multiple_ols


def test_konu05_questions_are_deterministic_and_cover_types() -> None:
    spec = konu05_model_specs()[1]
    result = fit_multiple_ols(load_dataset("wage1"), spec.dependent, (spec.focal_explanatory, *spec.controls))
    metadata = {name: variable_metadata(spec.dataset_key, name) for name in (spec.dependent, spec.focal_explanatory, *spec.controls)}
    first = generate_konu05_question("W1-M", 0, result, spec, metadata)
    assert first == generate_konu05_question("W1-M", 0, result, spec, metadata)
    generated = {generate_konu05_question("W1-M", index, result, spec, metadata).question_type for index in range(len(KONU05_QUESTION_TYPES))}
    assert generated == set(KONU05_QUESTION_TYPES)
    assert "standart hata" not in first.answer.casefold()
