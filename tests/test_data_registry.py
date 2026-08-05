from core.data_registry import allowed_explanatory_variables, get_dataset_metadata, load_dataset, variable_metadata


def test_wage1_catalogue_and_loader() -> None:
    metadata = get_dataset_metadata("wage1")
    frame = load_dataset("wage1")
    assert metadata.title.startswith("WAGE1")
    assert frame.shape[0] == 526
    assert {"wage", "educ", "exper", "tenure"}.issubset(frame.columns)
    assert allowed_explanatory_variables("wage1", "wage") == ("educ", "exper", "tenure")
    assert metadata.observation_unit == "Çalışan"
    assert variable_metadata("wage1", "wage").unit == "ABD doları/saat"


def test_hprice_variable_metadata() -> None:
    variable = variable_metadata("hprice1", "sqrft")
    assert variable.label == "Konut alanı"
    assert variable.unit == "kare fit"


def test_konu02_dataset_metadata_and_required_columns() -> None:
    expected = {
        "wage1": "yatay kesit verisi",
        "phillips": "zaman serisi verisi",
        "cps78_85": "havuzlanmış yatay kesit verisi",
        "wagepan": "panel veri",
    }
    for key, structure in expected.items():
        metadata = get_dataset_metadata(key)
        frame = load_dataset(key)
        assert metadata.data_structure == structure
        assert set(metadata.variables).issubset(frame.columns)
    jtrain = get_dataset_metadata("jtrain2")
    assert jtrain.collection_method == "deneysel"
    assert {"train", "re78"}.issubset(load_dataset("jtrain2").columns)
