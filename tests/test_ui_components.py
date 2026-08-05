"""Ortak soru eylem bileşeninin kapsam ve davranış denetimleri."""
from pathlib import Path
from core.ui_components import toggle_answer
from core.session_utils import next_question,question_state_keys

ROOT=Path(__file__).resolve().parents[1]

def test_all_topics_use_shared_question_actions() -> None:
    for number in range(1,13):
        topic=next((ROOT/"topics").glob(f"konu{number:02d}_*.py"))
        assert "render_question_actions" in topic.read_text(encoding="utf-8")
    css=(ROOT/"assets"/"styles.css").read_text(encoding="utf-8")
    assert "st-key-question_actions_" in css and "flex-wrap: wrap" in css

def test_toggle_and_new_question_visibility() -> None:
    state={}; index,_,answer=question_state_keys("konu10")
    assert toggle_answer(state,"konu10") and state[answer]
    assert not toggle_answer(state,"konu10") and not state[answer]
    state[answer]=True; next_question(state,"konu10")
    assert state[index]==1 and not state[answer]
