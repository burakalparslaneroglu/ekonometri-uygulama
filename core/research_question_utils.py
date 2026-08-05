"""Ölçülebilir araştırma sorularını kurmak ve doğrulamak için yardımcılar."""

from __future__ import annotations

from dataclasses import dataclass


REQUIRED_RESEARCH_COMPONENTS = (
    "gözlem birimi",
    "sonuç değişkeni",
    "temel açıklayıcı değişken",
    "anakütle, yer ve dönem",
    "araştırma amacı",
)


@dataclass(frozen=True)
class ResearchQuestion:
    """Ölçülebilir bir araştırma sorusunun zorunlu bileşenleri."""

    observation_unit: str
    outcome_variable: str
    explanatory_variable: str
    population_scope: str
    purpose: str


def missing_components(question: ResearchQuestion) -> tuple[str, ...]:
    """Boş bırakılan zorunlu araştırma sorusu bileşenlerini döndürür."""
    values = (
        question.observation_unit,
        question.outcome_variable,
        question.explanatory_variable,
        question.population_scope,
        question.purpose,
    )
    return tuple(name for name, value in zip(REQUIRED_RESEARCH_COMPONENTS, values, strict=True) if not value.strip())


def is_measurable_question(question: ResearchQuestion) -> bool:
    """Sorunun tüm zorunlu bileşenleri içerip içermediğini bildirir."""
    return not missing_components(question)


def format_research_question(question: ResearchQuestion) -> str:
    """Bileşenlerden öğrencinin okuyabileceği ilişki sorusunu kurar."""
    missing = missing_components(question)
    if missing:
        raise ValueError(f"Araştırma sorusunda eksik bileşenler var: {', '.join(missing)}.")
    return (
        f"{question.population_scope} içindeki {question.observation_unit} için "
        f"{question.explanatory_variable} ile {question.outcome_variable} arasında "
        f"{question.purpose.lower()} amacıyla nasıl bir ilişki vardır?"
    )
