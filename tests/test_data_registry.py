from core.data_registry import allowed_explanatory_variables, get_dataset_metadata, load_dataset, variable_metadata


def test_wage1_catalogue_and_loader() -> None:
    metadata = get_dataset_metadata("wage1")
    frame = load_dataset("wage1")
    assert metadata.title.startswith("WAGE1")
    assert frame.shape[0] == 526
    assert {"wage", "educ", "exper", "tenure"}.issubset(frame.columns)
    assert allowed_explanatory_variables("wage1", "wage") == ("educ", "exper", "tenure")


def test_hprice_variable_metadata() -> None:
    variable = variable_metadata("hprice1", "sqrft")
    assert variable.label == "Konut alanı"
    assert variable.unit == "kare fit"
