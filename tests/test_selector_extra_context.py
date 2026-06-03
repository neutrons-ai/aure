"""Tests for the fit-evidence channel into skill selection (Change 3a).

``select_skills(..., extra_context=...)`` lets the evaluation node activate a
skill that only becomes relevant once the data reveals an artifact, while the
``None`` default reproduces the intake-time behavior.
"""

from unittest.mock import MagicMock, patch

from aure.skills import select_skills
from aure.skills.loader import SkillRegistry
from aure.skills.selector import _build_sample_info


def test_build_sample_info_includes_extra_context():
    info = _build_sample_info(
        "Si/Cu in D2O",
        {"layers": [{"name": "Cu"}]},
        extra_context="- Residual fringe ~40 A implies SEI",
    )
    assert "Observations from fitting so far" in info
    assert "Residual fringe ~40 A implies SEI" in info


def test_build_sample_info_backward_compatible():
    info = _build_sample_info("Si/Cu", None)
    assert "Observations" not in info


def test_select_skills_extra_context_can_surface_skill():
    mock_response = MagicMock()
    mock_response.content = '["sei-layer-analysis"]'
    with (
        patch("aure.llm.llm_available", return_value=True),
        patch("aure.llm.get_llm") as mock_get,
        patch("aure.llm.invoke_with_timeout", return_value=mock_response),
    ):
        mock_get.return_value = MagicMock()
        result = select_skills(
            "cycled Li cell",
            registry=SkillRegistry(),
            extra_context="- Residual fringe ~40 A implies an SEI",
        )
    assert "sei-layer-analysis" in result
    # always-on skills still present
    assert "neutron-reflectometry" in result
