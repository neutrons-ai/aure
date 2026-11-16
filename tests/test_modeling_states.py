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


def _messages(out) -> str:
    return "\n".join(m["content"] for m in out.get("messages") or [])


def test_ties_without_declared_layers_warn_in_the_transcript():
    """Hand-written ties on a description-driven multi-state run rest on the
    parsed stack: names that vary per run, and one structure for every state.
    The run proceeds, and says so where it will be read."""
    from aure.nodes.modeling import _build_initial_model

    out = _build_initial_model(
        _make_state(_two_states(), user_config={"shared_parameters": ["Cu.thickness"]})
    )

    assert "error" not in out, out.get("error")
    text = _messages(out)
    assert "Cross-state ties are set (Cu.thickness)" in text
    assert "no state declares its own `layers`" in text
    assert "parsed from the sample description: Cu" in text
    assert "Not applied" not in text


def test_ties_that_match_no_parsed_layer_are_named_in_the_refusal():
    """A tie naming a layer the parse did not produce already stops the run;
    the refusal now names every such tie and says why the names did not match."""
    from aure.nodes.modeling import _build_initial_model

    out = _build_initial_model(
        _make_state(
            _two_states(),
            user_config={
                "unshared_parameters": [
                    "copper.thickness",
                    "Ti.interface",
                    "Cu.interface",
                ]
            },
        )
    )

    assert "unknown layer" in out["error"]
    assert "no state declares its own `layers`" in out["error"]
    assert "Not applied — no layer of that name in the parsed stack: " in out["error"]
    assert "copper.thickness, Ti.interface." in out["error"]


def test_no_tie_warning_when_a_state_declares_its_layers():
    from aure.nodes.modeling import _build_initial_model

    states = _two_states()
    states[1]["layers"] = [dict(_parsed_sample()["layers"][0])]
    out = _build_initial_model(
        _make_state(states, user_config={"shared_parameters": ["Cu.thickness"]})
    )

    assert "error" not in out, out.get("error")
    assert "Cross-state ties are set" not in _messages(out)


def test_no_tie_warning_without_hand_written_ties():
    """The default tied set (no spec) and the prose-derived untie are AuRE's
    own choices, made against the stack it parsed — nothing to warn about."""
    from aure.nodes.modeling import _build_initial_model

    out = _build_initial_model(_make_state(_two_states()))

    assert "error" not in out, out.get("error")
    assert "Cross-state ties are set" not in _messages(out)


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
    from aure.nodes.modeling import _build_initial_model

    out = _build_initial_model(
        _make_state(_single_state(background={"init": 2e-6, "min": 0.0, "max": 1e-5}))
    )
    assert "error" not in out, out.get("error")
    model = out["current_model"]
    assert model.get("states"), "single-state nuisance not attached to first model"
    assert model["states"][0]["background"] == {"init": 2e-6, "min": 0.0, "max": 1e-5}
    # A single state has no cross-state ties.
    assert model.get("shared_parameters", []) == []


def test_initial_build_single_state_theta_offset_is_attached():
    from aure.nodes.modeling import _build_initial_model

    out = _build_initial_model(
        _make_state(
            _single_state(theta_offset={"init": 0.0, "min": -0.02, "max": 0.02})
        )
    )
    assert "error" not in out, out.get("error")
    model = out["current_model"]
    assert model.get("states") and model["states"][0].get("theta_offset")


def test_initial_build_single_state_is_attached_without_nuisance():
    """A single state is attached whatever it carries: the state builder is
    the only reader of a state's overrides, so a model without its state is a
    model they were dropped from. (It used to need a nuisance key to qualify.)"""
    from aure.nodes.modeling import _build_initial_model

    out = _build_initial_model(_make_state(_single_state()))
    assert "error" not in out
    model = out["current_model"]
    assert [st["name"] for st in model["states"]] == ["state0"]
    # One state has nothing to tie.
    assert "shared_parameters" not in model


def test_single_state_is_attached_without_its_dataset_arrays():
    """The model is serialized into every refinement prompt and checkpoint;
    the builder reads the files, so the arrays have no business riding along."""
    from aure.nodes.modeling import _build_initial_model

    states = _single_state()
    states[0]["data_files"][0].update(Q=[0.01, 0.02], R=[1.0, 0.5], dR=[0.1, 0.1])
    out = _build_initial_model(_make_state(states))

    (ds,) = out["current_model"]["states"][0]["data_files"]
    assert ds == {"file": "/tmp/d2o.dat", "label": "d2o"}
    # The run's own copy keeps them for plotting.
    assert states[0]["data_files"][0]["Q"] == [0.01, 0.02]


def test_single_state_measurement_overrides_reach_the_model():
    """`ambient`, `intensity` and `back_reflection` on a lone state used to be
    read only when a nuisance key rode along. They stay on the state — which
    is what the builder reads, and what the refine LLM cannot edit — and the
    model-level fields are rewritten to match, so the model has one answer."""
    from aure.nodes.modeling import _build_initial_model

    states = _single_state(
        ambient={"rho": 6.4},
        intensity={"init": 1.0, "fixed": True},
        back_reflection=True,
    )
    out = _build_initial_model(_make_state(states))

    assert "error" not in out, out.get("error")
    model = out["current_model"]
    (attached,) = model["states"]
    assert attached["ambient"] == {"rho": 6.4}
    assert attached["intensity"] == {"init": 1.0, "fixed": True}
    assert attached["back_reflection"] is True
    assert model["ambient"] == {"name": "air", "sld": 6.4}
    assert model["back_reflection"] is True
    assert model["intensity"] == {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": True}


# ----------------------------------------------------------------------
# A single state's declared structure is the model's structure
# ----------------------------------------------------------------------

_DECLARED_COPPER = {
    "name": "copper",
    "sld": 6.48,
    "sld_min": 6.48,
    "sld_max": 6.48,
    "thickness": 480.0,
    "thickness_min": 400.0,
    "thickness_max": 600.0,
    "roughness": 5.0,
}


def test_single_state_declared_layers_become_the_model():
    """A single-state setup's `layers` were read only on the multi-state route,
    so the stack — and every bound in it — was dropped for the one parsed from
    the description (Cu, 6.5 over [4, 8] here)."""
    from aure.nodes.modeling import _build_initial_model

    states = _single_state(layers=[dict(_DECLARED_COPPER)])
    out = _build_initial_model(_make_state(states))

    assert "error" not in out, out.get("error")
    model = out["current_model"]
    # Verbatim: no default widening of the declared pin.
    assert model["layers"] == [_DECLARED_COPPER]
    assert out["baseline_model"]["layers"] == [_DECLARED_COPPER]
    assert out["model_history"][0]["n_layers"] == 1
    # The declaration lives on the model only; the attached state has no copy
    # to shadow it.
    (attached,) = model["states"]
    assert "layers" not in attached
    # The substrate was not declared, so the parsed one stands.
    assert model["substrate"]["name"] == "silicon"
    # And the run says which stack it used.
    assert any("declares its own layer stack" in m["content"] for m in out["messages"])
    # A copy: editing the model must not rewrite what the setup declared.
    model["layers"][0]["sld"] = 0.0
    assert states[0]["layers"][0]["sld"] == 6.48


def test_single_state_declared_layers_reach_the_fit_problem(tmp_path):
    import numpy as np

    from aure.nodes.model_builder import build_problem
    from aure.nodes.modeling import _build_initial_model

    q = np.linspace(0.01, 0.1, 40)
    data = tmp_path / "curve.dat"
    data.write_text(
        "# Q R dR dQ\n"
        + "\n".join(f"{a:.6f} {1e-3:.6e} {1e-4:.6e} {0.02 * a:.6e}" for a in q)
        + "\n"
    )
    states = _single_state(layers=[dict(_DECLARED_COPPER)])
    states[0]["data_files"] = [{"file": str(data), "label": "curve"}]
    state = _make_state(states)
    state["data_file"] = str(data)
    problem = build_problem(_build_initial_model(state)["current_model"])

    bounds = {p.name: (p.value, tuple(p.bounds)) for p in problem._parameters}
    # The declared pin is held: set to 6.48 and not a fit parameter at all.
    assert "copper rho" not in bounds
    assert next(iter(problem.models)).sample[1].material.rho.value == 6.48
    assert bounds["copper thickness"] == (480.0, (400.0, 600.0))


def test_single_state_declared_substrate_keeps_the_parsed_layers():
    from aure.nodes.modeling import _build_initial_model

    substrate = {"name": "quartz", "sld": 4.18, "roughness": 2.0, "roughness_max": 6.0}
    out = _build_initial_model(_make_state(_single_state(substrate=dict(substrate))))

    assert "error" not in out, out.get("error")
    model = out["current_model"]
    assert model["substrate"] == substrate
    assert [layer["name"] for layer in model["layers"]] == ["Cu"]


def test_single_state_declared_layer_may_omit_its_sld_window():
    """The builder defaults a missing window; the explanation must not demand
    one, or a declaration the builder accepts fails the modeling node."""
    from aure.nodes.modeling import _build_initial_model

    bare = {"name": "copper", "sld": 6.48, "thickness": 480.0, "roughness": 5.0}
    out = _build_initial_model(_make_state(_single_state(layers=[dict(bare)])))

    assert "error" not in out, out.get("error")
    assert out["current_model"]["layers"] == [bare]


def test_single_state_with_nuisance_declares_its_layers_once():
    """On the states route the declaration used to stay on the state while the
    model-level stack held the description's. It now lives in one place: left
    on the state too, it would shadow the model and make a refinement that
    edits the model's layers change nothing in the fit."""
    from aure.nodes.modeling import _build_initial_model

    states = _single_state(
        background={"init": 2e-6, "min": 0.0, "max": 1e-5},
        layers=[dict(_DECLARED_COPPER)],
    )
    out = _build_initial_model(_make_state(states))

    assert "error" not in out, out.get("error")
    model = out["current_model"]
    assert model["layers"] == [_DECLARED_COPPER]
    (attached,) = model["states"]
    assert "layers" not in attached
    assert attached["background"] == {"init": 2e-6, "min": 0.0, "max": 1e-5}
    # The run's own record of the setup is untouched.
    assert states[0]["layers"] == [_DECLARED_COPPER]


def test_multi_state_declared_layers_stay_on_their_states():
    """With two states the template/override split is the point — a state's
    own stack must not replace the shared one."""
    from aure.nodes.modeling import _build_initial_model

    states = _two_states()
    states[0]["layers"] = [dict(_DECLARED_COPPER)]
    out = _build_initial_model(_make_state(states))

    assert "error" not in out, out.get("error")
    model = out["current_model"]
    assert [layer["name"] for layer in model["layers"]] == ["Cu"]
    assert model["states"][0]["layers"] == [_DECLARED_COPPER]
    assert "layers" not in model["states"][1]


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


def _one_state_model(**state_fields) -> dict:
    current = _multi_state_current_model()
    current["states"] = [dict(current["states"][0], **state_fields)]
    current.pop("shared_parameters", None)
    current.pop("unshared_parameters", None)
    return current


def test_refine_cannot_undo_a_single_states_declared_measurement(monkeypatch):
    """The refine LLM re-emits the whole model, and rule 6 tells it to keep
    intensity varying. A state's own `ambient` / `intensity` are carried, not
    re-emitted, and the model-level fields follow them — so a value the setup
    declared outlives the first iteration."""
    from aure.nodes import modeling

    current = _one_state_model(
        ambient={"rho": 6.4}, intensity={"init": 1.0, "fixed": True}
    )
    modeling._fold_single_state(current)

    reply = _json_mod.loads(_json_mod.dumps(current))
    reply["ambient"] = {"name": "D2O", "sld": -0.56}
    reply["intensity"] = {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": False}
    _fake_llm(monkeypatch, reply)

    out = modeling._refine_model(_refine_state(current))

    assert "error" not in out, out.get("error")
    model = out["current_model"]
    assert model["ambient"]["sld"] == 6.4
    assert model["intensity"]["fixed"] is True
    assert model["states"][0]["ambient"] == {"rho": 6.4}
    assert model["states"][0]["intensity"] == {"init": 1.0, "fixed": True}


def test_refine_moves_structure_the_llm_put_on_a_single_state_up(monkeypatch):
    """Rule 12 invites the LLM to give a state its own `layers`. With one
    state that would leave the model-level stack as a second, unread
    description — so it is moved up and replaces it."""
    from aure.nodes import modeling

    current = _one_state_model()
    reply = _json_mod.loads(_json_mod.dumps(current))
    oxide = {"name": "CuO", "sld": 4.0, "thickness": 20.0, "roughness": 3.0}
    reply["states"][0]["layers"] = reply["layers"] + [oxide]
    _fake_llm(monkeypatch, reply)

    out = modeling._refine_model(_refine_state(current))

    assert "error" not in out, out.get("error")
    model = out["current_model"]
    assert [layer["name"] for layer in model["layers"]][-1] == "CuO"
    assert "layers" not in model["states"][0]


def _single_state_current_model() -> dict:
    current = _multi_state_current_model()
    for key in ("states", "shared_parameters", "unshared_parameters"):
        current.pop(key, None)
    return current


def _added_states_reply(current: dict) -> dict:
    """The refine LLM's model, with two states of runs it was never given."""
    return {
        **current,
        "states": [
            {"name": "234283_OCV", "data_files": ["/data/REFL_234283_1_234283.dat"]},
            {"name": "234324_OCV", "data_files": ["/data/REFL_234324_1_234324.dat"]},
        ],
        "unshared_parameters": ["Cu.thickness"],
    }


def test_refine_drops_states_the_llm_adds_to_a_single_state_model(monkeypatch):
    """A refinement cannot change which files a run fits."""
    import json
    from unittest.mock import MagicMock

    from aure.nodes import modeling

    monkeypatch.setattr(modeling, "llm_available", lambda: True)
    current = _single_state_current_model()
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(
        content=json.dumps(_added_states_reply(current))
    )
    monkeypatch.setattr(modeling, "get_llm", lambda temperature=0: fake_llm)
    state = _refine_state(current)
    state.pop("states", None)

    out = modeling._refine_model(state)

    assert "error" not in out, out.get("error")
    new_model = out["current_model"]
    assert "states" not in new_model
    assert "unshared_parameters" not in new_model
    assert new_model["data_file"] == current["data_file"]


def test_a_refinement_call_is_written_to_the_ledger(monkeypatch, tmp_path):
    """The call that refines the model is recorded like every other."""
    import json

    from aure.llm import ledger
    from aure.nodes import modeling

    current = _single_state_current_model()

    class _Reply:
        content = json.dumps(current)
        response_metadata: dict = {}
        usage_metadata: dict = {}

    class _LLM:
        model_name = "fake-model"
        temperature = 0

        def invoke(self, payload):
            return _Reply()

    monkeypatch.setattr(modeling, "llm_available", lambda: True)
    monkeypatch.setattr(modeling, "get_llm", lambda temperature=0: _LLM())
    monkeypatch.delenv("AURE_LLM_LOG", raising=False)
    monkeypatch.setattr(ledger, "_seq", 0)
    monkeypatch.setattr(ledger, "_node", None)
    ledger.set_sink(str(tmp_path / "llm_calls.jsonl"))
    ledger.set_node("modeling")
    state = _refine_state(current)
    state.pop("states", None)
    try:
        modeling._refine_model(state)
    finally:
        ledger.set_sink(None)

    lines = (tmp_path / "llm_calls.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert [(row["node"], row["model"]) for row in rows] == [("modeling", "fake-model")]


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
