"""Low-threshold inclusion of ``solvent-contrast-matching``.

Any liquid/solvent ambient should pull the skill in deterministically — even
when the LLM did not pick it and even when deuteration was never mentioned (an
unspecified solvent may be deuterated and the user simply forgot to say so).
"""

from unittest.mock import MagicMock, patch

import pytest

from aure.skills import select_skills
from aure.skills.loader import SkillRegistry
from aure.skills.selector import _has_liquid_ambient

SOLVENT = "solvent-contrast-matching"


def _resp(content):
    r = MagicMock()
    r.content = content
    return r


def _select(description, parsed=None, llm_content="[]"):
    with (
        patch("aure.llm.llm_available", return_value=True),
        patch("aure.llm.get_llm", return_value=MagicMock()),
        patch("aure.llm.invoke_with_timeout", return_value=_resp(llm_content)),
    ):
        return select_skills(
            description, parsed_sample=parsed, registry=SkillRegistry()
        )


def test_parsed_electrolyte_ambient_forces_solvent_skill():
    parsed = {
        "substrate": {"name": "silicon"},
        "layers": [{"name": "Cu"}],
        "ambient": {"name": "0.1 M NaHCO3 electrolyte", "sld": 0.0},
    }
    result = _select("Cu on Si in electrolyte at OCV", parsed=parsed, llm_content="[]")
    assert SOLVENT in result


def test_parsed_d2o_ambient_forces_skill_even_if_llm_omits_it():
    parsed = {"ambient": {"name": "D2O", "sld": 6.4}}
    result = _select(
        "a sample", parsed=parsed, llm_content='["metal-oxide-interfaces"]'
    )
    assert SOLVENT in result
    assert "metal-oxide-interfaces" in result


def test_description_electrolyte_forces_skill_before_parse():
    # parsed_sample=None reproduces the pre-parse selection call.
    result = _select(
        "Cu electrode in 0.1 M NaHCO3 electrolyte", parsed=None, llm_content="[]"
    )
    assert SOLVENT in result


def test_air_ambient_does_not_force_skill():
    parsed = {"ambient": {"name": "air", "sld": 0.0}, "layers": [{"name": "Cu"}]}
    result = _select("Cu film in air", parsed=parsed, llm_content="[]")
    assert SOLVENT not in result


def test_dry_sample_no_force():
    result = _select("polystyrene thin film on silicon", parsed=None, llm_content="[]")
    assert SOLVENT not in result


def test_forced_even_when_llm_unavailable():
    parsed = {"ambient": {"name": "electrolyte", "sld": 0.0}}
    with patch("aure.llm.llm_available", return_value=False):
        result = select_skills(
            "Cu in electrolyte", parsed_sample=parsed, registry=SkillRegistry()
        )
    assert SOLVENT in result
    assert "neutron-reflectometry" in result


def test_llm_selected_skill_not_duplicated():
    parsed = {"ambient": {"name": "D2O", "sld": 6.4}}
    result = _select("x", parsed=parsed, llm_content=f'["{SOLVENT}"]')
    assert result.count(SOLVENT) == 1


@pytest.mark.parametrize(
    "name,expected",
    [
        ("D2O", True),
        ("H2O", True),
        ("electrolyte", True),
        ("0.1 M NaHCO3 electrolyte", True),
        ("THF", True),
        ("air", False),
        ("vacuum", False),
        ("N2", False),
        ("", False),
        # Common solids that a misparse might drop into the ambient field.
        ("Si", False),
        ("silicon", False),
        ("Cu", False),
        ("Au", False),
        ("Ti", False),
        ("quartz", False),
    ],
)
def test_has_liquid_ambient_by_name(name, expected):
    assert _has_liquid_ambient("", {"ambient": {"name": name}}) is expected


def test_has_liquid_ambient_description_keywords():
    assert _has_liquid_ambient("measured in aqueous buffer", None)
    assert not _has_liquid_ambient("dry polymer film", None)


def test_ambient_matching_structural_material_is_not_liquid():
    # An exotic substrate name the static solids set doesn't know about, echoed
    # into the ambient field by an LLM misparse, must NOT read as a liquid.
    parsed = {
        "substrate": {"name": "MyAlloy"},
        "layers": [{"name": "Cu"}],
        "ambient": {"name": "MyAlloy"},
    }
    assert not _has_liquid_ambient("Cu on MyAlloy in air", parsed)


def test_solid_ambient_does_not_force_solvent_skill():
    parsed = {
        "substrate": {"name": "Si"},
        "layers": [{"name": "Cu"}],
        "ambient": {"name": "Si"},  # misparse
    }
    result = _select("Cu on Si", parsed=parsed, llm_content="[]")
    assert SOLVENT not in result


def test_per_state_liquid_ambient_forces_skill_when_top_level_air():
    parsed = {"ambient": {"name": "air", "sld": 0.0}, "layers": [{"name": "Cu"}]}
    states = [
        {"name": "wet", "ambient": {"name": "D2O", "sld": 6.4}},
        {"name": "dry", "ambient": {"name": "air", "sld": 0.0}},
    ]
    with (
        patch("aure.llm.llm_available", return_value=True),
        patch("aure.llm.get_llm", return_value=MagicMock()),
        patch("aure.llm.invoke_with_timeout", return_value=_resp("[]")),
    ):
        result = select_skills(
            "Cu film measured dry then wet",
            parsed_sample=parsed,
            registry=SkillRegistry(),
            states=states,
        )
    assert SOLVENT in result


def test_has_liquid_ambient_checks_per_state_ambients():
    parsed = {"ambient": {"name": "air"}, "layers": [{"name": "Cu"}]}
    states = [{"name": "s0", "ambient": {"name": "electrolyte"}}]
    assert _has_liquid_ambient("dry sample", parsed, states)
