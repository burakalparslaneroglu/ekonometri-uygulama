"""Kayıtlı bütün "Kendini sına" setleri için ortak içerik sözleşmesi."""

from __future__ import annotations

from collections import Counter

import pytest

from core.quiz.model import (
    Equation,
    FillBlanks,
    MultipleChoice,
    NumberBlank,
    TrueFalse,
    correct_answer_text,
    grade,
)
from core.quiz.registry import QUIZZES

SETS = list(QUIZZES.values())
QUESTIONS_PER_SET = 24
KINDS = {"coktan": 7, "dogru_yanlis": 7, "bosluk": 5, "denklem": 5}
"""Her set: 7 çoktan seçmeli, 7 doğru–yanlış, 5 boşluk doldurma, 5 denklem yazma."""


@pytest.mark.parametrize("quiz", SETS, ids=lambda q: q.topic_key)
def test_rich_and_non_overlapping(quiz) -> None:
    assert len(quiz.questions) == QUESTIONS_PER_SET
    assert Counter(q.kind for q in quiz.questions) == KINDS
    for field in ("concept", "key", "prompt"):
        values = [getattr(q, field) for q in quiz.questions]
        assert len(set(values)) == len(values), field


@pytest.mark.parametrize("quiz", SETS, ids=lambda q: q.topic_key)
def test_every_numbered_section_of_the_chapter_is_covered(quiz) -> None:
    chapter = str(int(quiz.topic_key[-2:]))
    assert {q.note.section for q in quiz.questions} == set(quiz.sections)
    assert all(section.startswith(chapter + ".") for section in quiz.sections)
    numbers = sorted(int(section.split(".")[1]) for section in quiz.sections)
    assert numbers == list(range(1, len(numbers) + 1))


@pytest.mark.parametrize("quiz", SETS, ids=lambda q: q.topic_key)
def test_answer_keys_are_balanced(quiz) -> None:
    choices = [q.answer for q in quiz.questions if isinstance(q.answer, MultipleChoice)]
    assert all(len(set(a.options)) == 4 for a in choices)
    assert len({a.correct for a in choices}) == 4
    truths = [q.answer.statement_is_true for q in quiz.questions if isinstance(q.answer, TrueFalse)]
    assert sum(truths) >= 3 and len(truths) - sum(truths) >= 3


@pytest.mark.parametrize("quiz", SETS, ids=lambda q: q.topic_key)
def test_the_key_is_not_given_away_by_its_length(quiz) -> None:
    """Doğru seçenek, en uzun yanlış seçenekten belirgin biçimde uzun olmamalıdır (uzunluk ipucu)."""

    for question in quiz.questions:
        answer = question.answer
        if isinstance(answer, MultipleChoice):
            longest_wrong = max(len(option) for index, option in enumerate(answer.options) if index != answer.correct)
            assert len(answer.options[answer.correct]) <= 1.25 * longest_wrong, question.key


@pytest.mark.parametrize("quiz", SETS, ids=lambda q: q.topic_key)
def test_every_answer_key_grades_itself_and_cites_notes(quiz) -> None:
    for question in quiz.questions:
        answer = question.answer
        if isinstance(answer, MultipleChoice):
            response = answer.correct
        elif isinstance(answer, TrueFalse):
            response = answer.statement_is_true
        elif isinstance(answer, FillBlanks):
            response = tuple(b.shown if isinstance(b, NumberBlank) else b.accepted[0] for b in answer.blanks)
        else:
            response = answer.answer
        assert grade(question, response).correct, (quiz.topic_key, question.key)
        assert correct_answer_text(question)
        assert "§" in question.explanation and len(question.explanation) > 60, question.key


@pytest.mark.parametrize("quiz", SETS, ids=lambda q: q.topic_key)
def test_the_shown_formula_grades_as_correct(quiz) -> None:
    """Denklem sorularında cevap anahtarının gösterilen (LaTeX) biçimi de doğru okunur: öğrenci gösterilen cevabı
    aynen yazarsa doğru sayılır."""

    for question in quiz.questions:
        if isinstance(question.answer, Equation):
            assert grade(question, question.answer.shown).correct, (quiz.topic_key, question.key)


def test_formula_parser_reads_latex_habits_and_left_sides() -> None:
    """Öğrencinin LaTeX alışkanlıkları ve sol tarafı da yazması: aynı ifade okunur."""

    from core.quiz.expression import Symbol, equivalent, latex, parse

    symbols = (Symbol("a", "a", "a"), Symbol("sx", "s_x", "s", aliases=("s_x",)),
               Symbol("x1", "x_1", "x", 1, 5, aliases=("x_1",)))
    reference = parse("a/sx^2 + sqrt(x1)", symbols)
    for text in ("\\frac{a}{s_x^2} + \\sqrt{x_1}", "\\dfrac{a}{s^2_x}+\\sqrt{x_{1}}", "y = a/sx^2 + sqrt(x1)",
                 "\\left(\\frac{a}{s_x^{2}}\\right) + \\sqrt{x1}", "a \\cdot \\frac{1}{s_x^2} + sqrt(x1)"):
        assert equivalent(parse(text, symbols), reference, symbols), text
    assert equivalent(parse("\\ln(x_1) \\times 2", symbols), parse("2*log(x1)", symbols), symbols)
    assert latex(parse("ln(x1)", symbols), symbols) == "\\ln\\left(x_1\\right)"  # notlardaki gösterim
    with pytest.raises(ValueError):
        parse("a =", symbols)
