"""Tests for intake-time hypothesis seeding (user hypothesis → top-ranked).

Covers Change 1: the user's stated hypothesis becomes one or more
``origin="user"`` entries ranked above the skill-derived ones, and the parse
prompt no longer injects tentative layers into the baseline.
"""

import json
from unittest.mock import MagicMock

from aure.nodes import intake
from aure.nodes.prompts import (
    format_sample_parse_prompt,
    format_structural_hypothesis_prompt,
)

_PARSED = {
    "substrate": {"name": "Si", "sld": 2.07},
    "layers": [{"name": "Cu", "thickness": 500, "sld": 6.5}],
    "ambient": {"name": "D2O", "sld": 6.19},
    "back_reflection": False,
}


def test_user_hypothesis_ranked_first(monkeypatch):
    """User-origin hypotheses are reordered to the top with stable ids."""
    monkeypatch.setattr(intake, "llm_available", lambda: True)
    monkeypatch.setattr(intake, "get_llm", lambda temperature=0: MagicMock())

    # LLM returns a skill hypothesis FIRST and the user one second; our code
    # must reorder the user-origin entry to the front and assign ids in rank.
    llm_json = json.dumps(
        [
            {
                "title": "Add native CuO",
                "rationale": "metal-oxide-interfaces",
                "change": "insert CuO",
                "skill_source": "metal-oxide-interfaces",
            },
            {
                "title": "Oxide on top (user)",
                "rationale": "user said so",
                "change": "insert oxide",
                "skill_source": "user",
            },
        ]
    )
    monkeypatch.setattr(
        intake, "invoke_with_timeout", lambda llm, msgs: MagicMock(content=llm_json)
    )

    out = intake.generate_structural_hypotheses_with_llm(
        sample_description="Si / Cu in D2O",
        parsed_sample=_PARSED,
        skill_context="",
        hypothesis="there may be an oxide on top",
    )
    assert len(out) == 2
    assert out[0]["origin"] == "user"
    assert out[0]["id"] == 1
    assert out[0]["title"] == "Oxide on top (user)"
    assert out[1]["origin"] == "skill"
    assert out[1]["id"] == 2
    # every entry carries the full provenance schema
    for h in out:
        assert "created_in_iteration" in h
        assert h["status"] == "pending"


def test_structural_prompt_includes_user_hypothesis():
    p = format_structural_hypothesis_prompt(
        sample_description="d",
        parsed_sample=_PARSED,
        skill_context="",
        hypothesis="there may be an oxide on top",
    )
    assert "User's Stated Hypothesis" in p
    assert "there may be an oxide on top" in p


def test_structural_prompt_no_hypothesis():
    p = format_structural_hypothesis_prompt(
        sample_description="d", parsed_sample=_PARSED, skill_context=""
    )
    assert "(none stated)" in p


def test_sample_parse_prompt_keeps_tentative_out_of_baseline():
    p = format_sample_parse_prompt("Si / Cu in D2O", hypothesis="maybe an oxide")
    assert "OUT of the baseline" in p
