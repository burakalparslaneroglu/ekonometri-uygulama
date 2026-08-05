"""Konu 12 soru belirlenim denetimi."""
from core.data_registry import load_dataset
from core.konu12_questions import Konu12QuestionContext,generate_konu12_question
from core.robust_inference_utils import fit_robust_inference,heteroskedasticity_tests,simulate_heteroskedastic_coverage
def test_questions_are_deterministic() -> None:
    d=load_dataset("hprice1").assign(lotsize1000=lambda x:x.lotsize/1000,sqrft100=lambda x:x.sqrft/100); r=fit_robust_inference(d,"price",("lotsize1000","sqrft100","bdrms")); c=Konu12QuestionContext(r,heteroskedasticity_tests(r.ols)[0],simulate_heteroskedastic_coverage(repetitions=20))
    assert generate_konu12_question(c,"x",2)==generate_konu12_question(c,"x",2)
