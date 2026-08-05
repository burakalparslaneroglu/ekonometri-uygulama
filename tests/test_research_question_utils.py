import pytest

from core.research_question_utils import ResearchQuestion, format_research_question, is_measurable_question, missing_components


def test_research_question_validation_and_formatting() -> None:
    question = ResearchQuestion("çalışan", "saatlik ücret", "eğitim yılı", "incelenen örneklem", "İlişki")
    assert is_measurable_question(question)
    assert "eğitim yılı" in format_research_question(question)


def test_research_question_reports_missing_components() -> None:
    incomplete = ResearchQuestion("", "saatlik ücret", "", "", "İlişki")
    assert missing_components(incomplete) == ("gözlem birimi", "temel açıklayıcı değişken", "anakütle, yer ve dönem")
    with pytest.raises(ValueError, match="eksik bileşenler"):
        format_research_question(incomplete)
