"""Konu 11 temel sayısal denetimleri."""
import pytest
from core.data_registry import load_dataset
from core.interaction_utils import add_interaction_columns,conditional_difference_grid,conditional_group_difference,fit_interaction_model,interaction_curve_data

def test_wage_and_hprice_interactions_match_notes() -> None:
    wage=add_interaction_columns(load_dataset("wage1"),"wage1")
    result=fit_interaction_model(wage,"lwage","educ12","female","female_educ12",("exper","tenure"))
    assert result.result.nobs==526
    assert (result.lines.slope_zero,result.lines.slope_one,result.result.p_values_two_sided_zero["female_educ12"],result.joint_f)==pytest.approx((.0903,.0830,.5935,32.785),abs=.002)
    gap=conditional_group_difference(result.result,"female","female_educ12",0)
    assert gap.estimate==pytest.approx(result.result.coefficients["female"])
    house=add_interaction_columns(load_dataset("hprice1"),"hprice1")
    model=fit_interaction_model(house,"price","lotsize10k","colonial","colonial_lotsize10k",("sqrft100","bdrms"))
    assert (model.result.p_values_two_sided_zero["colonial_lotsize10k"],model.joint_f)==pytest.approx((.0258,3.039),abs=.002)
    assert conditional_group_difference(model.result,"colonial","colonial_lotsize10k",0).estimate==pytest.approx(8.2437,abs=.001)

def test_curve_and_conditional_grid_include_covariance() -> None:
    import numpy as np
    wage=add_interaction_columns(load_dataset("wage1"),"wage1"); model=fit_interaction_model(wage,"lwage","educ12","female","female_educ12",("exper","tenure"))
    curve=interaction_curve_data(model.lines,np.array([-1.,0.,1.]))
    assert curve.loc[2,"D=1"]-curve.loc[1,"D=1"]==pytest.approx(model.lines.slope_one)
    grid=conditional_difference_grid(model.result,"female","female_educ12",np.array([-2.,0.,2.]),data_min=-4,data_max=4)
    point=conditional_group_difference(model.result,"female","female_educ12",2)
    assert grid.loc[2,"standard_error"]==pytest.approx(point.standard_error) and grid.in_sample_range.all()
