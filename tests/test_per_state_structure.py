"""Per-state structure: the evidence for it, proposing it, and reporting it.

A multi-state co-refinement may need a layer in one state and not another
("sample != structure"). The builder has always supported that; these tests
cover the path that has to *decide* on it — the per-state residual evidence
reaching the prompts, a hypothesis being scoped to a subset of states, and the
resulting structural/tie change being reported.
"""

from __future__ import annotations

import pytest

from aure.nodes.hypotheses import (
    merge_structural_hypotheses,
    normalize_hypothesis_states,
)
from aure.nodes.modeling import _summarize_definition_changes
from aure.nodes.prompts import (
    format_fit_evaluation_prompt,
    format_hypothesis_revision_prompt,
    format_structural_hypothesis_prompt,
)


_PER_STATE = {
    "air": {
        "has_residual_fringes": True,
        "fringe_amplitude": 0.11,
        "n_residual_fringes": 3,
        "unmodeled_thicknesses": [
            {"thickness": 42.0, "uncertainty": 6.0, "confidence": "high"}
        ],
    },
    "reduced": {"has_residual_fringes": False},
}


def _eval_prompt(**kwargs):
    base = dict(
        sample_description="Cu on Si, in air then held at -0.5 V",
        hypothesis=None,
        chi_squared=5.0,
        method="lm",
        converged=True,
        parameters={"Cu thickness": 300.0},
        features={},
    )
    base.update(kwargs)
    return format_fit_evaluation_prompt(**base)


# ----------------------------------------------------------------------
# The evidence has to reach the prompt
# ----------------------------------------------------------------------


def test_per_state_fringes_reach_the_evaluation_prompt():
    """Multi-state runs leave the aggregate unset; without the per-state dict
    the evaluator was told no fringes were detected at all."""
    prompt = _eval_prompt(per_state_residual_analysis=_PER_STATE)
    section = prompt.split("## Residual Fringe Analysis")[1].split("## Parameters")[0]
    # The section is no longer the bare "nothing detected" line it rendered
    # when only the (always-unset) aggregate was consulted.
    assert section.strip() != "(no structured residual oscillations detected)"
    assert "State `air`" in section
    assert "~42" in section


def test_asymmetric_fringes_name_the_states_and_offer_both_readings():
    prompt = _eval_prompt(per_state_residual_analysis=_PER_STATE)
    section = prompt.split("## Residual Fringe Analysis")[1]
    assert "`air` but NOT in `reduced`" in section
    assert "UNTIED parameters" in section
    assert "differ in STRUCTURE" in section


def test_fringes_in_every_state_point_at_the_shared_template():
    every = {k: dict(_PER_STATE["air"]) for k in ("air", "reduced")}
    section = _eval_prompt(per_state_residual_analysis=every).split(
        "## Residual Fringe Analysis"
    )[1]
    assert "EVERY state" in section
    assert "SHARED template" in section


def test_aggregate_is_still_used_when_there_are_no_states():
    prompt = _eval_prompt(residual_analysis=_PER_STATE["air"])
    assert "~42" in prompt
    assert "State `air`" not in prompt


# ----------------------------------------------------------------------
# The per-file breakdown must say which state, and describe the right thing
# ----------------------------------------------------------------------


def test_per_file_chi2_labels_states_and_drops_the_q_segment_wording():
    prompt = _eval_prompt(
        per_file_results=[
            {"label": "REFL_1001", "state": "air", "chi_squared": 8.1},
            {"label": "REFL_1002", "state": "reduced", "chi_squared": 1.9},
        ]
    )
    assert "[air] REFL_1001" in prompt
    assert "[reduced] REFL_1002" in prompt
    assert "Q-range segments" not in prompt
    assert "physical STATES" in prompt


def test_per_file_chi2_keeps_the_multi_file_wording_without_states():
    prompt = _eval_prompt(
        per_file_results=[
            {"label": "low-Q", "chi_squared": 1.5},
            {"label": "high-Q", "chi_squared": 2.5},
        ]
    )
    assert "Q-range segments" in prompt
    assert "physical STATES" not in prompt


# ----------------------------------------------------------------------
# Scoping a hypothesis to a subset of states
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,known,expected",
    [
        (["air"], ["air", "reduced"], ["air"]),
        (["reduced", "air"], ["air", "reduced"], []),  # covers all -> global
        (["D2O"], ["air", "reduced"], []),  # unknown name dropped
        (["air", "D2O"], ["air", "reduced"], ["air"]),
        (["air"], ["air"], []),  # single state has nothing to scope
        (["air"], [], []),
        ("air", ["air", "reduced"], []),  # not a list
        (None, ["air", "reduced"], []),
    ],
)
def test_normalize_hypothesis_states(raw, known, expected):
    assert normalize_hypothesis_states(raw, known) == expected


def test_scope_survives_the_merge_and_cannot_be_changed_by_modeling():
    merged = merge_structural_hypotheses(
        prior=[],
        llm_returned=[
            {
                "title": "Add native CuO in air only",
                "change": "insert 10-50 A CuO above Cu",
                "skill_source": "metal-oxide-interfaces",
                "states": ["air"],
            }
        ],
        allow_new=True,
        current_iteration=1,
    )
    assert merged[0]["states"] == ["air"]

    # modeling may only touch status — re-scoping is a rename, and renames are
    # exactly what the guarded merge exists to refuse.
    back = merge_structural_hypotheses(
        prior=merged,
        llm_returned=[{"id": 1, "status": "tried", "states": ["reduced"]}],
        allow_new=False,
        current_iteration=2,
    )
    assert back[0]["states"] == ["air"]
    assert back[0]["status"] == "tried"


def test_hypothesis_prompts_expose_the_states_and_the_scope_field():
    parsed = {
        "substrate": {"name": "Si", "sld": 2.07},
        "layers": [{"name": "Cu", "thickness": 300, "sld": 6.55}],
        "ambient": {"name": "air", "sld": 0.0},
    }
    prompt = format_structural_hypothesis_prompt(
        sample_description="Cu on Si in air, then reduced",
        parsed_sample=parsed,
        skill_context="",
        state_names=["air", "reduced"],
    )
    assert "## Measurement States" in prompt
    assert "`air`" in prompt and "`reduced`" in prompt
    assert '"states": []' in prompt  # the field is in the emitted schema

    single = format_structural_hypothesis_prompt(
        sample_description="Cu on Si",
        parsed_sample=parsed,
        skill_context="",
        state_names=["state0"],
    )
    assert "single-state analysis" in single


def test_revision_prompt_renders_an_existing_scope():
    prompt = format_hypothesis_revision_prompt(
        sample_description="",
        current_model={},
        skill_context="",
        structural_hypotheses=[
            {"id": 1, "title": "Add CuO", "status": "pending", "states": ["air"]}
        ],
        fit_history=[],
        chi_squared=5.0,
        state_names=["air", "reduced"],
    )
    assert "applies to state(s): `air`" in prompt
    assert "## Measurement States" in prompt


# ----------------------------------------------------------------------
# Reporting the change
# ----------------------------------------------------------------------


def _two_state_models():
    template = [
        {"name": "SiO2", "sld": 3.47, "thickness": 15.0, "roughness": 3.0},
        {"name": "Cu", "sld": 6.55, "thickness": 300.0, "roughness": 8.0},
    ]
    old = {
        "substrate": {"name": "Si", "sld": 2.07},
        "ambient": {"name": "air", "sld": 0.0},
        "layers": template,
        "states": [{"name": "air"}, {"name": "reduced"}],
        "unshared_parameters": [],
    }
    new = dict(old)
    new["states"] = [
        {
            "name": "air",
            "layers": template
            + [{"name": "CuO", "sld": 4.8, "thickness": 30.0, "roughness": 6.0}],
        },
        {"name": "reduced"},
    ]
    new["unshared_parameters"] = ["Cu.interface"]
    return old, new


def test_per_state_layer_change_is_reported():
    """The top-level diff only sees the shared template, so a state-scoped
    structural change used to produce an empty change list."""
    old, new = _two_state_models()
    changes = _summarize_definition_changes(old, new)
    assert "State 'air': added layer(s) CuO" in changes
    assert not any("reduced" in c for c in changes)


def test_tie_change_is_reported():
    old, new = _two_state_models()
    changes = _summarize_definition_changes(old, new)
    assert any(c.startswith("Now untied (per-state): Cu.interface") for c in changes)


def test_state_without_own_layers_is_compared_against_the_template():
    """A state inheriting the template must not read as a structural change
    just because the JSON spells it differently."""
    old, _ = _two_state_models()
    new = dict(old)
    new["states"] = [
        {"name": "air", "layers": old["layers"]},  # explicit, but identical
        {"name": "reduced"},
    ]
    assert not any("State '" in c for c in _summarize_definition_changes(old, new))
