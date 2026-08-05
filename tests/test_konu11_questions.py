"""Konu 11 soru belirlenim denetimi."""
from core.data_registry import load_dataset
from core.interaction_utils import add_interaction_columns,conditional_group_difference,fit_interaction_model
from core.konu11_questions import Konu11QuestionContext,generate_konu11_question
def test_questions_are_deterministic() -> None:
    d=add_interaction_columns(load_dataset("wage1"),"wage1"); r=fit_interaction_model(d,"lwage","educ12","female","female_educ12",("exper","tenure")); c=Konu11QuestionContext(r,conditional_group_difference(r.result,"female","female_educ12",0))
    assert generate_konu11_question(c,"x",2)==generate_konu11_question(c,"x",2)
