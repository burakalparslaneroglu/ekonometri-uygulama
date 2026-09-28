"""Ortak soru eylem bileşeninin kapsam ve davranış denetimleri."""
from pathlib import Path

import pytest

from core.session_utils import next_question,question_state_keys
from core.ui_components import format_null_decision,toggle_answer

ROOT=Path(__file__).resolve().parents[1]


def test_all_topics_use_shared_question_actions_and_badges() -> None:
    # Konu 1–2 Uygulama, Sezgi ve Kendini sına sekmelerine taşındı; ortak başlıkları topics/shared.py'dedir.
    for number in range(3,13):
        topic=next((ROOT/"topics").glob(f"konu{number:02d}_*.py"))
        source=topic.read_text(encoding="utf-8")
        assert "render_question_actions" in source
        assert "topic-badge" in source
        assert f"KONU {number:02d}" in source
        assert f'st.header("KONU {number:02d}")' not in source
    component=(ROOT/"core"/"ui_components.py").read_text(encoding="utf-8")
    css=(ROOT/"assets"/"styles.css").read_text(encoding="utf-8")
    assert "st.columns((1.0, 1.0)" in component and "2.4" not in component
    assert 'type="primary"' in component and 'type="secondary"' in component
    assert "st-key-question_actions_" in css and "flex-wrap: wrap" in css


def test_toggle_and_new_question_visibility() -> None:
    state={}; index,_,answer=question_state_keys("konu10")
    assert toggle_answer(state,"konu10") and state[answer]
    assert not toggle_answer(state,"konu10") and not state[answer]
    state[answer]=True; next_question(state,"konu10")
    assert state[index]==1 and not state[answer]


def test_null_decision_is_generated_from_p_value_and_alpha() -> None:
    assert format_null_decision(.0499,.05)=="H₀ reddedilir"
    assert format_null_decision(.05,.05)=="H₀ reddedilemez"
    assert format_null_decision(.0501,.05)=="H₀ reddedilemez"
    with pytest.raises(ValueError): format_null_decision(float("nan"),.05)
    with pytest.raises(ValueError): format_null_decision(.05,1.0)
