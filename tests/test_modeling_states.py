"""Tests for multi-state metadata in the modeling node (Ticket 06)."""

from __future__ import annotations

from typing import Any, Dict


def _parsed_sample() -> dict:
    return {
        "substrate": {
            "name": "silicon",
            "sld": 2.07,
            "roughness": 3.0,
            "roughness_max": 15.0,
        },
        "layers": [
            {
                "name": "Cu",
                "sld": 6.5,
                "sld_min": 4.0,
                "sld_max": 8.0,
                "thickness": 500.0,
                "thickness_min": 250.0,
                "thickness_max": 750.0,
                "roughness": 5.0,
                "roughness_max": 20.0,
            },
        ],
        "ambient": {"name": "air", "sld": 0.0},
        "back_reflection": False,
        "intensity": {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": False},
        "constraints": [],
    }


def _two_states() -> list[dict]:
    return [
        {
            "name": "D2O",
            "extra_description": "fully deuterated solvent",
            "data_files": [{"file": "/tmp/d2o.dat", "label": "D2O-comb"}],
            "ambient": {"name": "D2O", "sld": 6.4},
        },
        {
            "name": "H2O",
            "extra_description": "fully protonated solvent",
            "data_files": [{"file": "/tmp/h2o.dat", "label": "H2O-comb"}],
            "ambient": {"name": "H2O", "sld": -0.56},
        },
    ]


def _make_state(states_list, *, user_config=None) -> dict:
    return {
        "data_file": "/tmp/d2o.dat",
        "parsed_sample": _parsed_sample(),
        "extracted_features": {},
        "iteration": 0,
        "messages": [],
        "states": states_list,
        "user_config": user_config or {},
        "dq_is_fwhm": True,
    }


# ----------------------------------------------------------------------
# Initial build
# ----------------------------------------------------------------------


def test_initial_build_single_state_unchanged():
    """No multi-state metadata when states list is empty."""
    from aure.nodes.modeling import _build_initial_model

    state = _make_state([])
    out = _build_initial_model(state)

    assert "error" not in out
    model = out["current_model"]
    assert "states" not in model
    assert "shared_parameters" not in model


def test_initial_build_attaches_states_metadata():
    """Multi-state config: states copied verbatim, default tie spec used."""
    from aure.nodes.modeling import _build_initial_model

    state = _make_state(_two_states())
    out = _build_initial_model(state)

    assert "error" not in out, out.get("error")
    model = out["current_model"]
    assert len(model["states"]) == 2
    assert model["states"][0]["name"] == "D2O"
    assert model["states"][1]["ambient"]["sld"] == -0.56
    # No config override → empty lists (defaults applied at build time).
    assert model["shared_parameters"] == []
    assert model["unshared_parameters"] == []


def test_initial_build_user_config_shared_wins():
    """User-supplied shared_parameters land verbatim in the ModelDefinition."""
    from aure.nodes.modeling import _build_initial_model

    state = _make_state(
        _two_states(),
        user_config={"shared_parameters": ["Cu.thickness"]},
    )
    out = _build_initial_model(state)
    assert "error" not in out
    model = out["current_model"]
    assert model["shared_parameters"] == ["Cu.thickness"]
    assert model["unshared_parameters"] == []


def test_initial_build_user_config_unshared_wins():
    from aure.nodes.modeling import _build_initial_model

    state = _make_state(
        _two_states(),
        user_config={"unshared_parameters": ["Cu.interface"]},
    )
    out = _build_initial_model(state)
    assert "error" not in out
    model = out["current_model"]
    assert model["unshared_parameters"] == ["Cu.interface"]
    assert model["shared_parameters"] == []


def test_initial_build_invalid_layer_in_shared_parameters_errors():
    """Layer-name mismatch in user-supplied tie set surfaces as updates['error']."""
    from aure.nodes.modeling import _build_initial_model

    state = _make_state(
        _two_states(),
        user_config={"shared_parameters": ["Bogus.thickness"]},
    )
    out = _build_initial_model(state)
    assert "error" in out
    assert "Bogus" in out["error"] or "shared_parameters" in out["error"]


def test_initial_build_mutually_exclusive_errors():
    from aure.nodes.modeling import _build_initial_model

    state = _make_state(
        _two_states(),
        user_config={
            "shared_parameters": ["Cu.thickness"],
            "unshared_parameters": ["Cu.interface"],
        },
    )
    out = _build_initial_model(state)
    assert "error" in out
    assert "mutually exclusive" in out["error"]


# ----------------------------------------------------------------------
# Single-state nuisance must reach the FIRST model (regression)
# ----------------------------------------------------------------------


def _single_state(**nuisance) -> list[dict]:
    st = {
        "name": "state0",
        "data_files": [{"file": "/tmp/d2o.dat", "label": "d2o"}],
    }
    st.update(nuisance)
    return [st]


def test_initial_build_single_state_background_is_attached():
    """A single state that requests `background` must be attached so the first
    model AuRE fits includes it (regression: single-state nuisance was dropped
    by the `< 2 states` guard)."""
    from aure.nodes.model_builder import needs_states_problem
    from aure.nodes.modeling import _build_initial_model

    out = _build_initial_model(
        _make_state(_single_state(background={"init": 2e-6, "min": 0.0, "max": 1e-5}))
    )
    assert "error" not in out, out.get("error")
    model = out["current_model"]
    assert model.get("states"), "single-state nuisance not attached to first model"
    assert model["states"][0]["background"] == {"init": 2e-6, "min": 0.0, "max": 1e-5}
    # Routes through the per-state builder that ties the background.
    assert needs_states_problem(model) is True
    # A single state has no cross-state ties.
    assert model.get("shared_parameters", []) == []


def test_initial_build_single_state_theta_offset_is_attached():
    from aure.nodes.model_builder import needs_states_problem
    from aure.nodes.modeling import _build_initial_model

    out = _build_initial_model(
        _make_state(
            _single_state(theta_offset={"init": 0.0, "min": -0.02, "max": 0.02})
        )
    )
    assert "error" not in out, out.get("error")
    model = out["current_model"]
    assert model.get("states") and model["states"][0].get("theta_offset")
    assert needs_states_problem(model) is True


def test_initial_build_single_state_without_nuisance_not_attached():
    """A single state with no requested nuisance stays on the model-level build
    path — no states block is needed."""
    from aure.nodes.modeling import _build_initial_model

    out = _build_initial_model(_make_state(_single_state()))
    assert "error" not in out
    assert "states" not in out["current_model"]


# ----------------------------------------------------------------------
# _attach_state_metadata standalone
# ----------------------------------------------------------------------


def test_attach_state_metadata_noop_for_single_state():
    from aure.nodes.modeling import _attach_state_metadata

    model_def: Dict[str, Any] = {"layers": []}
    _attach_state_metadata(model_def, {"states": [], "user_config": {}})
    assert "states" not in model_def


def test_attach_state_metadata_validates_layers():
    import pytest
    from aure.nodes.modeling import _attach_state_metadata

    model_def: Dict[str, Any] = {
        "substrate": {"name": "silicon"},
        "layers": [{"name": "Cu"}],
        "ambient": {"name": "air"},
    }
    state = {
        "states": _two_states(),
        "user_config": {"shared_parameters": ["NotALayer.thickness"]},
    }
    with pytest.raises(ValueError, match="NotALayer"):
        _attach_state_metadata(model_def, state)


# ----------------------------------------------------------------------
# Refinement: carry-over of multi-state fields
# ----------------------------------------------------------------------


def _multi_state_current_model() -> dict:
    return {
        "substrate": {
            "name": "silicon",
            "sld": 2.07,
            "roughness": 3.0,
            "roughness_max": 15.0,
        },
        "layers": [
            {
                "name": "Cu",
                "sld": 6.5,
                "sld_min": 4.0,
                "sld_max": 8.0,
                "thickness": 500.0,
                "thickness_min": 250.0,
                "thickness_max": 750.0,
                "roughness": 5.0,
                "roughness_max": 20.0,
            }
        ],
        "ambient": {"name": "air", "sld": 0.0},
        "back_reflection": False,
        "constraints": [],
        "data_file": "/tmp/d2o.dat",
        "intensity": {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": False},
        "dq_is_fwhm": True,
        "states": _two_states(),
        "shared_parameters": ["Cu.thickness"],
        "unshared_parameters": [],
    }


def _refine_state(current_model: dict, *, user_config=None) -> dict:
    return {
        "data_file": current_model["data_file"],
        "parsed_sample": _parsed_sample(),
        "extracted_features": {},
        "iteration": 1,
        "current_model": current_model,
        "fit_results": [
            {
                "chi_squared": 99.9,
                "method": "lm",
                "converged": True,
                "issues": ["chi too high"],
                "suggestions": [],
                "parameters": {},
            }
        ],
        "messages": [],
        "user_config": user_config or {},
        "states": current_model.get("states") or [],
        "active_skills": [],
        "structural_hypotheses": [],
    }


def test_refine_carries_over_states_when_llm_omits(monkeypatch):
    """LLM returns a model without states/ties → previous values preserved."""
    import json
    from unittest.mock import MagicMock

    from aure.nodes import modeling

    monkeypatch.setattr(modeling, "llm_available", lambda: True)

    current = _multi_state_current_model()

    # LLM returns a "structural-only" model with no multi-state fields.
    llm_return = {
        k: v
        for k, v in current.items()
        if k not in ("states", "shared_parameters", "unshared_parameters")
    }
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps(llm_return))
    monkeypatch.setattr(modeling, "get_llm", lambda temperature=0: fake_llm)

    state = _refine_state(current)
    out = modeling._refine_model(state)

    assert "error" not in out, out.get("error")
    new_model = out["current_model"]
    assert new_model["states"] == current["states"]
    assert new_model["shared_parameters"] == ["Cu.thickness"]


def test_refine_user_config_overrides_llm_proposed_ties(monkeypatch):
    """When config has shared_parameters, it overrides whatever the LLM proposes."""
    import json
    from unittest.mock import MagicMock

    from aure.nodes import modeling

    monkeypatch.setattr(modeling, "llm_available", lambda: True)

    current = _multi_state_current_model()

    # LLM proposes a different tie set
    llm_return = dict(current)
    llm_return["shared_parameters"] = ["Cu.material.rho"]
    llm_return["unshared_parameters"] = []
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps(llm_return))
    monkeypatch.setattr(modeling, "get_llm", lambda temperature=0: fake_llm)

    state = _refine_state(current, user_config={"shared_parameters": ["Cu.thickness"]})
    out = modeling._refine_model(state)
    assert "error" not in out, out.get("error")
    assert out["current_model"]["shared_parameters"] == ["Cu.thickness"]


def test_refine_invalid_llm_tie_spec_falls_back_to_previous(monkeypatch, caplog):
    """LLM proposes ties referencing unknown layers → fall back to previous."""
    import json
    import logging
    from unittest.mock import MagicMock

    from aure.nodes import modeling

    monkeypatch.setattr(modeling, "llm_available", lambda: True)

    current = _multi_state_current_model()
    llm_return = dict(current)
    llm_return["shared_parameters"] = ["NotALayer.thickness"]
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps(llm_return))
    monkeypatch.setattr(modeling, "get_llm", lambda temperature=0: fake_llm)

    state = _refine_state(current)
    with caplog.at_level(logging.WARNING, logger="aure.nodes.modeling"):
        out = modeling._refine_model(state)

    assert "error" not in out
    assert out["current_model"]["shared_parameters"] == ["Cu.thickness"]
    assert any("invalid" in rec.message.lower() for rec in caplog.records)


def test_refine_single_state_unchanged(monkeypatch):
    """Single-state refinement is untouched by Ticket 06."""
    import json
    from unittest.mock import MagicMock

    from aure.nodes import modeling

    monkeypatch.setattr(modeling, "llm_available", lambda: True)

    current = _multi_state_current_model()
    # Strip multi-state fields → single-state model
    current.pop("states", None)
    current.pop("shared_parameters", None)
    current.pop("unshared_parameters", None)

    llm_return = dict(current)
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps(llm_return))
    monkeypatch.setattr(modeling, "get_llm", lambda temperature=0: fake_llm)

    state = _refine_state(current)
    state.pop("states", None)
    out = modeling._refine_model(state)

    assert "error" not in out, out.get("error")
    new_model = out["current_model"]
    assert "states" not in new_model or not new_model["states"]


def test_refine_hypothesis_writeback_is_status_only(monkeypatch):
    """Modeling may only update statuses — it cannot rename or add entries."""
    import json
    from unittest.mock import MagicMock

    from aure.nodes import modeling

    monkeypatch.setattr(modeling, "llm_available", lambda: True)

    current = _multi_state_current_model()
    llm_return = dict(current)
    # The LLM tries to (a) mark existing #1 tried but RENAME it, and
    # (b) fabricate a brand-new hypothesis #2. The merge guard must keep
    # the original identity and drop the fabricated entry.
    llm_return["structural_hypotheses"] = [
        {"id": 1, "title": "HIJACKED", "status": "tried", "tried_in_iteration": 1},
        {
            "id": 2,
            "title": "fabricated by modeling",
            "change": "x",
            "status": "pending",
        },
    ]
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps(llm_return))
    monkeypatch.setattr(modeling, "get_llm", lambda temperature=0: fake_llm)

    state = _refine_state(current)
    state["structural_hypotheses"] = [
        {
            "id": 1,
            "title": "Add CuO",
            "rationale": "r",
            "change": "c",
            "skill_source": "metal-oxide-interfaces",
            "origin": "skill",
            "status": "pending",
            "tried_in_iteration": None,
            "created_in_iteration": None,
            "notes": "",
        }
    ]

    out = modeling._refine_model(state)
    assert "error" not in out, out.get("error")
    hyps = out["structural_hypotheses"]
    assert len(hyps) == 1  # fabricated #2 dropped
    assert hyps[0]["title"] == "Add CuO"  # identity preserved (not HIJACKED)
    assert hyps[0]["status"] == "tried"  # status update applied


# ----------------------------------------------------------------------
# Natural-language cross-state tie extraction (unshared parameters)
# ----------------------------------------------------------------------

import json as _json_mod  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402


def _state_with_oxide(desc, user_config=None):
    parsed = {
        "substrate": {
            "name": "silicon",
            "sld": 2.07,
            "roughness": 3.0,
            "roughness_max": 15.0,
        },
        "layers": [
            {
                "name": "Cu oxide",
                "sld": 4.0,
                "sld_min": 2.0,
                "sld_max": 6.0,
                "thickness": 30.0,
                "thickness_min": 5.0,
                "thickness_max": 100.0,
                "roughness": 5.0,
                "roughness_max": 20.0,
            },
            {
                "name": "Cu",
                "sld": 6.5,
                "sld_min": 4.0,
                "sld_max": 8.0,
                "thickness": 500.0,
                "thickness_min": 250.0,
                "thickness_max": 750.0,
                "roughness": 5.0,
                "roughness_max": 20.0,
            },
        ],
        "ambient": {"name": "D2O", "sld": 6.4},
        "back_reflection": True,
        "intensity": {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": False},
        "constraints": [],
    }
    return {
        "data_file": "/tmp/REFL_230539.txt",
        "sample_description": desc,
        "parsed_sample": parsed,
        "extracted_features": {},
        "iteration": 0,
        "messages": [],
        "states": _two_states(),
        "user_config": user_config or {},
        "dq_is_fwhm": True,
    }


_OXIDE_DESC = (
    "D2O / Cu oxide / 50 nm Cu / 3 nm Ti on Si. The copper oxide thickness, SLD and "
    "interface are not shared across states."
)


def _fake_llm(monkeypatch, payload):
    from aure.nodes import modeling

    monkeypatch.setattr(modeling, "llm_available", lambda: True)
    fake = MagicMock()
    fake.invoke.return_value = MagicMock(content=_json_mod.dumps(payload))
    monkeypatch.setattr(modeling, "get_llm", lambda temperature=0: fake)
    return fake


def test_nl_unshared_extracted_and_unties_oxide(monkeypatch):
    from aure.nodes import modeling
    from aure.nodes.model_builder import _resolve_tied_set

    _fake_llm(
        monkeypatch,
        {
            "unshared_parameters": [
                "Cu oxide.thickness",
                "Cu oxide.material.rho",
                "Cu oxide.interface",
            ]
        },
    )

    out = modeling.modeling_node(_state_with_oxide(_OXIDE_DESC))
    assert "error" not in out, out.get("error")
    md = out["current_model"]
    assert set(md["unshared_parameters"]) == {
        "Cu oxide.thickness",
        "Cu oxide.material.rho",
        "Cu oxide.interface",
    }
    # persisted into user_config so it survives refinement
    assert set(out["user_config"]["unshared_parameters"]) == set(
        md["unshared_parameters"]
    )
    # the Cu oxide structural params are NOT tied across states
    tied = _resolve_tied_set(md)
    assert ("Cu oxide", "thickness") not in tied
    assert ("Cu oxide", "material.rho") not in tied
    assert ("Cu oxide", "interface") not in tied
    # Cu (not called out) stays tied by default
    assert ("Cu", "thickness") in tied


def test_nl_extraction_skipped_when_user_config_ties_present(monkeypatch):
    from aure.nodes import modeling

    # The fake returns neither per_state_absent nor unshared ties, so neither NL
    # extractor changes anything; the explicit tie config must win regardless.
    # (Structure extraction is orthogonal to ties and may still query the LLM.)
    _fake_llm(monkeypatch, {"unshared_parameters": []})
    st = _state_with_oxide(
        _OXIDE_DESC, user_config={"shared_parameters": ["Cu.thickness"]}
    )
    out = modeling.modeling_node(st)
    assert "error" not in out, out.get("error")
    assert out["current_model"]["shared_parameters"] == ["Cu.thickness"]
    assert not out["current_model"].get(
        "unshared_parameters"
    )  # NL tie extraction skipped


def test_nl_extraction_noop_without_llm(monkeypatch):
    from aure.nodes import modeling
    from aure.nodes.model_builder import _resolve_tied_set

    monkeypatch.setattr(modeling, "llm_available", lambda: False)
    out = modeling.modeling_node(_state_with_oxide(_OXIDE_DESC))
    assert "error" not in out, out.get("error")
    md = out["current_model"]
    assert not md.get("unshared_parameters")  # nothing derived without an LLM
    assert ("Cu oxide", "thickness") in _resolve_tied_set(md)  # default tying


def test_distinct_sample_threaded_onto_model(monkeypatch):
    """user_config.distinct_sample lands on the built model (identity, not physics)."""
    from aure.nodes import modeling

    monkeypatch.setattr(modeling, "llm_available", lambda: False)
    out = modeling.modeling_node(
        _state_with_oxide(_OXIDE_DESC, user_config={"distinct_sample": True})
    )
    assert "error" not in out, out.get("error")
    assert out["current_model"]["distinct_sample"] is True


def test_distinct_sample_absent_by_default(monkeypatch):
    """Without the flag, the model does not assert distinct samples."""
    from aure.nodes import modeling

    monkeypatch.setattr(modeling, "llm_available", lambda: False)
    out = modeling.modeling_node(_state_with_oxide(_OXIDE_DESC))
    assert "error" not in out, out.get("error")
    assert not out["current_model"].get("distinct_sample")


def test_refine_prunes_tie_for_removed_layer(monkeypatch):
    """Regression for the 230539_230543 crash: when a refine removes a layer,
    the now-dangling tie on it must be PRUNED, not carried into the fit (which
    previously aborted with 'unshared_parameters references unknown layer')."""
    import json
    from unittest.mock import MagicMock

    from aure.nodes import modeling
    from aure.nodes.model_builder import _resolve_tied_set

    monkeypatch.setattr(modeling, "llm_available", lambda: True)

    current = _multi_state_current_model()
    cu_oxide = {
        "name": "Cu oxide",
        "sld": 4.0,
        "sld_min": 2.0,
        "sld_max": 6.0,
        "thickness": 30.0,
        "thickness_min": 5.0,
        "thickness_max": 100.0,
        "roughness": 5.0,
        "roughness_max": 20.0,
    }
    current["layers"] = [cu_oxide, current["layers"][0]]  # [Cu oxide, Cu]
    current["shared_parameters"] = []
    current["unshared_parameters"] = ["Cu oxide.interface"]

    # LLM removes Cu oxide (structural change) and omits the tie spec.
    llm_return = {
        k: v
        for k, v in current.items()
        if k not in ("states", "shared_parameters", "unshared_parameters")
    }
    llm_return["layers"] = [current["layers"][1]]  # only Cu remains
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps(llm_return))
    monkeypatch.setattr(modeling, "get_llm", lambda temperature=0: fake_llm)

    out = modeling._refine_model(_refine_state(current))
    assert "error" not in out, out.get("error")
    nm = out["current_model"]
    # the dangling Cu oxide tie was pruned ...
    assert "Cu oxide.interface" not in (nm.get("unshared_parameters") or [])
    # ... so tie-resolution (which runs at fit time) no longer raises.
    _resolve_tied_set(nm)


def test_per_state_structure_extracted_from_description(monkeypatch):
    """sample != structure: 'the H2O state has no copper oxide' → that state's
    own layers drop the oxide while the other state keeps the template."""
    import json
    from unittest.mock import MagicMock

    from aure.nodes import modeling

    monkeypatch.setattr(modeling, "llm_available", lambda: True)

    st = _state_with_oxide(
        "D2O / Cu oxide / 50 nm Cu on Si. The H2O state has no copper oxide."
    )

    def fake_invoke(msgs):
        content = msgs[0].content
        if "per_state_absent" in content:  # structure-extraction prompt
            return MagicMock(
                content=json.dumps({"per_state_absent": {"H2O": ["Cu oxide"]}})
            )
        return MagicMock(content=json.dumps({"unshared_parameters": []}))  # tie prompt

    fake = MagicMock()
    fake.invoke.side_effect = fake_invoke
    monkeypatch.setattr(modeling, "get_llm", lambda temperature=0: fake)

    out = modeling.modeling_node(st)
    assert "error" not in out, out.get("error")
    by_name = {s["name"]: s for s in out["current_model"]["states"]}
    # H2O carries its own oxide-less stack; D2O inherits the template (has oxide).
    h2o_names = [layer["name"] for layer in by_name["H2O"]["layers"]]
    assert "Cu oxide" not in h2o_names and "Cu" in h2o_names
    assert not by_name["D2O"].get("layers")  # inherits template (oxide present)
