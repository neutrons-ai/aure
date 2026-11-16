"""A value the user declares known is held: set, and not fitted.

"The copper is bulk copper, 6.48 — fix it" used to reach the fit on roughly
half the runs. There was no way to say it (the layer schema had bounds and no
flag), the initial build widened a zero-width window to ±2.5, a pin that did
get through was still a free parameter sitting on both of its bounds, and the
refine LLM re-emitted the model with nothing obliging it to copy the pin.

Two spellings mean "held": ``<key>_fixed: true``, and a window whose min
equals its max. These tests follow one through every place it can be lost.
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aure.nodes.model_builder import (
    bic_inputs,
    build_experiment,
    build_problem,
    build_states_problem,
    is_pinned,
)

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def data_file():
    Q = np.linspace(0.01, 0.10, 60)
    R = np.clip((0.0217 / (2 * Q)) ** 4, 1e-10, 1.0)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False)
    f.write("# Q R dR dQ\n")
    for q, r in zip(Q, R):
        f.write(f"{q:.6f} {r:.6e} {0.05 * r:.6e} {0.02 * q:.6e}\n")
    f.close()
    yield f.name
    try:
        os.unlink(f.name)
    except OSError:
        pass


COPPER = {"name": "copper", "sld": 6.48, "thickness": 500.0, "roughness": 5.0}
ZERO_WIDTH = {"sld_min": 6.48, "sld_max": 6.48}
FLAGGED = {"sld_fixed": True, "sld_min": 4.0, "sld_max": 9.0}


def _defn(data_file: str, layer: dict, **ambient) -> dict:
    return {
        "substrate": {"name": "Si", "sld": 2.07, "roughness": 3.0},
        "ambient": {"name": "dTHF", "sld": 6.2, **ambient},
        "layers": [layer],
        "data_file": data_file,
        "intensity": {"value": 1.0, "fixed": True},
    }


# ----------------------------------------------------------------------
# What "held" means
# ----------------------------------------------------------------------


def test_is_pinned_reads_both_spellings():
    assert is_pinned({"sld": 6.48, **ZERO_WIDTH}, "sld")
    assert is_pinned({"sld": 6.48, **FLAGGED}, "sld")
    assert is_pinned({"thickness": 15, "thickness_fixed": True}, "thickness")
    assert not is_pinned({"sld": 6.48, "sld_min": 6.0, "sld_max": 7.0}, "sld")
    assert not is_pinned({"sld": 6.48}, "sld")
    assert not is_pinned({"sld": 6.48, "sld_fixed": False}, "sld")
    # One bound alone is not a window.
    assert not is_pinned({"sld": 6.48, "sld_min": 6.48}, "sld")


# ----------------------------------------------------------------------
# The builder: a held parameter never enters the fit
# ----------------------------------------------------------------------


@pytest.mark.parametrize("spelling", [ZERO_WIDTH, FLAGGED], ids=["window", "flag"])
def test_a_held_sld_is_not_a_fit_parameter(data_file, spelling):
    free = build_problem(_defn(data_file, COPPER))
    held = build_problem(_defn(data_file, {**COPPER, **spelling}))

    assert "copper rho" in free.labels()
    assert "copper rho" not in held.labels()
    assert build_experiment(_defn(data_file, {**COPPER, **spelling})).sample[
        1
    ].material.rho.value == pytest.approx(6.48)
    # One fewer parameter for BIC to charge the model for.
    assert bic_inputs(held)["_n_free_params"] == bic_inputs(free)["_n_free_params"] - 1


def test_the_flag_holds_the_declared_value_whatever_the_window(data_file):
    """`sld_fixed` says the value is known. A window that excludes it is noise
    left over from an earlier edit, not a reason to move the value."""
    layer = {**COPPER, "sld_fixed": True, "sld_min": 7.0, "sld_max": 9.0}
    experiment = build_experiment(_defn(data_file, layer))
    assert experiment.sample[1].material.rho.value == pytest.approx(6.48)


def test_a_held_thickness_is_not_a_fit_parameter(data_file):
    for spelling in (
        {"thickness_fixed": True},
        {"thickness_min": 500.0, "thickness_max": 500.0},
    ):
        problem = build_problem(_defn(data_file, {**COPPER, **spelling}))
        assert "copper thickness" not in problem.labels()
        assert "copper rho" in problem.labels()


@pytest.mark.parametrize(
    "spelling",
    [{"sld_fixed": True}, {"sld_min": 6.2, "sld_max": 6.2}],
    ids=["flag", "window"],
)
def test_a_held_ambient_sld_is_not_a_fit_parameter(data_file, spelling):
    assert "dTHF rho" in build_problem(_defn(data_file, COPPER)).labels()
    held = _defn(data_file, COPPER, **spelling)
    assert "dTHF rho" not in build_problem(held).labels()
    assert build_experiment(held).sample[2].material.rho.value == pytest.approx(6.2)


def test_a_held_parameter_is_held_in_every_state(data_file):
    """Tied across states, a held parameter is one constant, not one free
    parameter shared."""
    defn = _defn(data_file, {**COPPER, **FLAGGED})
    defn["states"] = [
        {"name": "A", "data_files": [{"file": data_file, "label": "a"}]},
        {"name": "B", "data_files": [{"file": data_file, "label": "b"}]},
    ]
    problem, by_state, _ = build_states_problem(defn)

    assert not any("copper rho" in name for name in problem.labels())
    for experiments in by_state.values():
        assert experiments[0].sample[1].material.rho.value == pytest.approx(6.48)


# ----------------------------------------------------------------------
# The initial model, built from what intake parsed
# ----------------------------------------------------------------------


def test_the_initial_build_does_not_widen_a_pin():
    """A parsed window under 2.0 wide is expanded to ±2.5 "for flexibility".
    A window of one point is not a narrow range — it is the user's constant,
    and this is where it used to become 6.48 ± 2.5."""
    from aure.nodes.modeling import _build_layers

    parsed = {
        "layers": [
            {**COPPER, **ZERO_WIDTH},
            {**COPPER, "name": "narrow", "sld_min": 6.0, "sld_max": 7.0},
        ]
    }
    pinned, narrow = _build_layers(parsed, {})

    assert (pinned["sld_min"], pinned["sld_max"]) == (6.48, 6.48)
    assert (narrow["sld_min"], narrow["sld_max"]) == pytest.approx((3.98, 8.98))


def test_the_initial_build_carries_the_flags():
    from aure.nodes.modeling import _build_layers

    parsed = {"layers": [{**COPPER, "sld_fixed": True, "thickness_fixed": True}]}
    (layer,) = _build_layers(parsed, {})

    assert layer["sld_fixed"] is True
    assert layer["thickness_fixed"] is True
    # Absent stays absent: an unflagged layer does not grow a `false`.
    (plain,) = _build_layers({"layers": [dict(COPPER)]}, {})
    assert "sld_fixed" not in plain and "thickness_fixed" not in plain


def test_fringe_spacing_does_not_widen_a_held_thickness():
    """A lone layer's thickness window is stretched to cover the thickness the
    fringes suggest. Not when the thickness is declared known."""
    from aure.nodes.modeling import _build_layers

    features = {"estimated_total_thickness": 900.0}
    held = {**COPPER, "thickness_min": 500.0, "thickness_max": 500.0}

    (layer,) = _build_layers({"layers": [held]}, features)
    assert (layer["thickness_min"], layer["thickness_max"]) == (500.0, 500.0)

    (free,) = _build_layers({"layers": [dict(COPPER)]}, features)
    assert free["thickness_max"] == pytest.approx(1800.0)


def test_the_model_explanation_says_what_is_held():
    """Whether a "fix this" took effect has to be readable in the transcript,
    not only in the exported problem."""
    from aure.nodes.modeling import _explain_model

    text = _explain_model(
        [{**COPPER, **ZERO_WIDTH, "thickness_fixed": True}],
        {"name": "Si", "sld": 2.07},
        {"name": "dTHF", "sld": 6.2, "sld_fixed": True},
        {},
    )
    assert "SLD fixed at 6.48" in text
    assert "d fixed at 500 Å" in text
    assert "Ambient: SLD fixed at 6.20" in text


# ----------------------------------------------------------------------
# Wideners and counts
# ----------------------------------------------------------------------


@pytest.mark.parametrize("spelling", [ZERO_WIDTH, FLAGGED], ids=["window", "flag"])
def test_the_fallback_widener_leaves_a_held_parameter(spelling):
    from aure.nodes.modeling import _widen_all_bounds_json

    layer = {**COPPER, **spelling, "thickness_min": 400.0, "thickness_max": 600.0}
    (widened,) = _widen_all_bounds_json({"layers": [layer]})["layers"]

    assert (widened["sld_min"], widened["sld_max"]) == (
        spelling["sld_min"],
        spelling["sld_max"],
    )
    # The free parameter next to it is still widened.
    assert (widened["thickness_min"], widened["thickness_max"]) == (350.0, 650.0)


def test_a_boundary_hit_does_not_widen_a_held_parameter():
    """A held parameter is not in the fit, so a fit AuRE runs cannot report it
    — but a hit list from an older fit can, when a zero-width window sat on
    both bounds by construction."""
    from aure.nodes.evaluation import _expand_model_bounds

    model = {
        "layers": [
            {**COPPER, **FLAGGED, "thickness_min": 400.0, "thickness_max": 600.0}
        ]
    }
    hits = [
        {"name": "copper rho", "bound_hit": "upper"},
        {"name": "copper thickness", "bound_hit": "upper"},
    ]
    (layer,) = _expand_model_bounds(model, hits)["layers"]

    assert (layer["sld_min"], layer["sld_max"]) == (4.0, 9.0)
    assert layer["thickness_max"] == pytest.approx(700.0)


def test_the_free_parameter_count_leaves_out_what_is_held():
    from aure.nodes.evaluation import _count_free_params

    def count(layer, **ambient):
        return _count_free_params(
            {
                "layers": [layer],
                "ambient": {"name": "dTHF", "sld": 6.2, **ambient},
                "intensity": {"fixed": True},
            }
        )

    free = count(COPPER)
    assert count({**COPPER, **ZERO_WIDTH}) == free - 1
    assert count({**COPPER, **FLAGGED, "thickness_fixed": True}) == free - 2
    assert count(COPPER, sld_fixed=True) == free - 1


def test_mode_enumeration_does_not_reseed_a_held_sld():
    """It tries other SLDs for a thin layer. A held SLD is not a mode to pick."""
    from aure.nodes.fitting import _enumerate_thin_layer_modes

    thin = {"name": "oxide", "sld": 4.0, "thickness": 10.0, "roughness": 3.0}
    state = {"Q": np.linspace(0.01, 0.2, 50).tolist()}
    model = {"layers": [{**thin, "sld_fixed": True}], "data_file": "/nonexistent"}

    # No problem is built (the file does not exist): nothing was thin enough
    # to enumerate once the held layer is set aside.
    assert _enumerate_thin_layer_modes(model, state) is model


# ----------------------------------------------------------------------
# Refinement: what the baseline holds stays held
# ----------------------------------------------------------------------


def _baseline() -> dict:
    return {
        "substrate": {"name": "Si", "sld": 2.07, "roughness": 3.0},
        "layers": [
            {**COPPER, **ZERO_WIDTH, "thickness_min": 400.0, "thickness_max": 600.0},
            {
                "name": "oxide",
                "sld": 4.0,
                "sld_fixed": True,
                "thickness": 15.0,
                "thickness_fixed": True,
                "roughness": 3.0,
            },
        ],
        "ambient": {"name": "dTHF", "sld": 6.2, "sld_fixed": True},
        "data_file": "/tmp/curve.dat",
        "back_reflection": False,
        "intensity": {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": False},
    }


def test_restore_puts_back_what_the_llm_released_widened_or_moved():
    from aure.nodes.modeling import _restore_baseline_pins

    baseline = _baseline()
    model = json.loads(json.dumps(baseline))
    copper, oxide = model["layers"]
    copper.update(sld_min=6.3, sld_max=8.5, sld=6.55)  # widened and moved
    oxide.pop("sld_fixed")  # released
    oxide["thickness"] = 22.0  # moved, flag kept
    model["ambient"].update(sld=5.96, sld_fixed=False)  # released and moved
    copper["thickness"] = 480.0  # a free parameter: the LLM's to change

    restored = _restore_baseline_pins(baseline, model)

    assert sorted(restored) == [
        "copper sld",
        "dTHF sld",
        "oxide sld",
        "oxide thickness",
    ]
    assert (copper["sld"], copper["sld_min"], copper["sld_max"]) == (6.48, 6.48, 6.48)
    assert oxide["sld_fixed"] is True and oxide["thickness"] == 15.0
    assert model["ambient"] == baseline["ambient"]
    assert copper["thickness"] == 480.0


def test_restore_is_silent_when_nothing_moved():
    from aure.nodes.modeling import _restore_baseline_pins

    baseline = _baseline()
    model = json.loads(json.dumps(baseline))
    assert _restore_baseline_pins(baseline, model) == []
    assert model == baseline


def test_restore_leaves_a_renamed_or_removed_layer_alone():
    """A different name is a different material — realizing "the copper has
    oxidised through" must be possible."""
    from aure.nodes.modeling import _restore_baseline_pins

    baseline = _baseline()
    model = json.loads(json.dumps(baseline))
    model["layers"] = [
        {
            "name": "copper oxide",
            "sld": 5.1,
            "sld_min": 4.0,
            "sld_max": 6.5,
            "thickness": 480.0,
            "roughness": 5.0,
        }
    ]
    model["ambient"] = {"name": "THF", "sld": 0.18}

    assert _restore_baseline_pins(baseline, model) == []
    assert model["layers"][0]["sld_min"] == 4.0
    assert model["ambient"] == {"name": "THF", "sld": 0.18}


def test_a_pin_the_llm_added_is_not_sticky():
    """Only what the baseline declared is re-applied. A pin the refinement
    introduced is its own judgement, and it may take it back."""
    from aure.nodes.modeling import _restore_baseline_pins

    baseline = _baseline()
    baseline["layers"][0] = {**COPPER, "sld_min": 4.0, "sld_max": 9.0}
    model = json.loads(json.dumps(baseline))
    model["layers"][0].update(sld_min=6.0, sld_max=7.0)

    assert _restore_baseline_pins(baseline, model) == []
    assert model["layers"][0]["sld_min"] == 6.0


def test_restore_reaches_a_states_own_stack():
    from aure.nodes.modeling import _restore_baseline_pins

    baseline = _baseline()
    baseline["states"] = [
        {"name": "wet", "layers": [{**COPPER, **ZERO_WIDTH}]},
        {"name": "dry"},
    ]
    model = json.loads(json.dumps(baseline))
    model["states"][0]["layers"][0].update(sld_min=6.0, sld_max=7.0)

    assert _restore_baseline_pins(baseline, model) == ["[wet] copper sld"]
    assert model["states"][0]["layers"][0]["sld_min"] == 6.48


def test_refinement_cannot_release_a_declared_constant(monkeypatch):
    """End to end through the refine node: the LLM re-emits the model with the
    copper widened and the solvent unfixed, and both come back held — with a
    line in the transcript saying so."""
    from aure.nodes import modeling

    baseline = _baseline()
    reply = json.loads(json.dumps(baseline))
    reply["layers"][0].update(sld_min=6.3, sld_max=8.5)
    reply["ambient"].pop("sld_fixed")

    monkeypatch.setattr(modeling, "llm_available", lambda: True)
    fake = MagicMock()
    fake.invoke.return_value = MagicMock(content=json.dumps(reply))
    monkeypatch.setattr(modeling, "get_llm", lambda temperature=0: fake)

    out = modeling._refine_model(
        {
            "data_file": baseline["data_file"],
            "iteration": 1,
            "current_model": json.loads(json.dumps(baseline)),
            "baseline_model": baseline,
            "fit_results": [
                {
                    "chi_squared": 9.9,
                    "method": "lm",
                    "parameters": {},
                    "issues": [],
                    "suggestions": [],
                }
            ],
            "messages": [],
            "user_config": {},
            "active_skills": [],
            "structural_hypotheses": [],
        }
    )

    assert "error" not in out, out.get("error")
    model = out["current_model"]
    assert (model["layers"][0]["sld_min"], model["layers"][0]["sld_max"]) == (
        6.48,
        6.48,
    )
    assert model["ambient"]["sld_fixed"] is True
    transcript = out["messages"][0]["content"]
    assert "Kept fixed as declared" in transcript
    assert "copper sld" in transcript and "dTHF sld" in transcript


# ----------------------------------------------------------------------
# The prompts offer the slot, and rule 3 exempts it
# ----------------------------------------------------------------------


def test_the_intake_prompt_offers_a_way_to_say_fixed():
    from aure.nodes.prompts import format_sample_parse_prompt

    prompt = format_sample_parse_prompt("Cu on Si. The copper is 6.48, fix it.")
    assert '"sld_fixed"' in prompt and '"thickness_fixed"' in prompt
    assert "Known constants" in prompt


def test_the_refinement_prompt_exempts_held_parameters_from_widening():
    from aure.nodes.prompts import format_model_refinement_prompt_json

    prompt = format_model_refinement_prompt_json(
        current_model=_baseline(),
        sample_description="Cu on Si",
        fit_result={
            "chi_squared": 5.0,
            "parameters": {},
            "issues": [],
            "suggestions": [],
        },
        features={},
    )
    assert '"sld_fixed"' in prompt and '"thickness_fixed"' in prompt
    assert "NEVER for a parameter that is held constant" in prompt
    assert "3a. HELD CONSTANTS" in prompt


# ----------------------------------------------------------------------
# From a parsed description to the exported problem
# ----------------------------------------------------------------------


def test_a_fixed_sld_from_the_description_reaches_the_exported_problem(tmp_path):
    """The entry's own test, with the LLM's parse stubbed: "the copper is
    6.48 and the dTHF is 6.2 — fix both" must export a problem with no free
    parameter for either."""
    from bumps.serialize import load_file

    from aure.nodes.model_builder import save_problem_json
    from aure.workflow import run_prepare

    q = np.linspace(0.01, 0.25, 50)
    data = tmp_path / "curve.dat"
    data.write_text(
        "# Q R dR\n" + "\n".join(f"{v:.6f} 1.0e-3 1.0e-4" for v in q) + "\n"
    )

    def intake(state):
        return {
            "current_node": "intake",
            "Q": q.tolist(),
            "R": [1e-3] * len(q),
            "dR": [1e-4] * len(q),
            "parsed_sample": {
                "substrate": {"name": "silicon", "sld": 2.07, "roughness": 3.0},
                "layers": [
                    {**COPPER, "sld_min": 6.3, "sld_max": 8.5, "sld_fixed": True}
                ],
                "ambient": {"name": "dTHF", "sld": 6.2, "sld_fixed": True},
            },
            "messages": [],
        }

    def analysis(state):
        return {"current_node": "analysis", "extracted_features": {}, "messages": []}

    stubs = {"intake": intake, "analysis": analysis}
    with patch.dict("aure.workflow.runner.NODE_FUNCTIONS", stubs, clear=False):
        result = run_prepare(
            data_file=str(data),
            sample_description="Cu on Si in dTHF. Copper 6.48, dTHF 6.2. Fix both.",
            output_dir=str(tmp_path / "out"),
        )
    assert result.get("error") is None, result.get("error")

    out = Path(tmp_path / "problem.json")
    save_problem_json(result["current_model"], out)
    problem = load_file(str(out))
    free = problem.labels()
    assert not any("copper rho" in n or "dTHF rho" in n for n in free), free

    sample = next(iter(problem.models)).sample
    assert sample[1].material.rho.value == pytest.approx(6.48)
    assert sample[2].material.rho.value == pytest.approx(6.2)
    # And the baseline remembers them, which is what refinement re-applies.
    baseline = result["baseline_model"]
    assert is_pinned(baseline["layers"][0], "sld")
    assert is_pinned(baseline["ambient"], "sld")


# ----------------------------------------------------------------------
# Roughness is held the same way as every other parameter
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "spelling",
    [{"roughness_fixed": True}, {"roughness_min": 5.0, "roughness_max": 5.0}],
    ids=["flag", "window"],
)
def test_a_held_layer_roughness_is_not_a_fit_parameter(data_file, spelling):
    assert "copper interface" in build_problem(_defn(data_file, COPPER)).labels()

    held = _defn(data_file, {**COPPER, **spelling})
    assert "copper interface" not in build_problem(held).labels()
    assert build_experiment(held).sample[1].interface.value == pytest.approx(5.0)


def test_a_held_substrate_roughness_is_not_a_fit_parameter(data_file):
    """It used to be fitted from 0 whatever was declared — the one parameter
    with no way to hold it."""
    free = _defn(data_file, COPPER)
    assert "Si interface" in build_problem(free).labels()

    held = _defn(data_file, COPPER)
    held["substrate"]["roughness_fixed"] = True
    assert "Si interface" not in build_problem(held).labels()
    assert build_experiment(held).sample[0].interface.value == pytest.approx(3.0)


def test_back_reflection_holds_the_outer_surface_with_its_layer(data_file):
    """There the outer surface sits on the ambient slab and borrows the
    outermost layer's declared roughness: one declared value behind two
    parameters, so holding it holds both."""
    free = dict(_defn(data_file, COPPER), back_reflection=True)
    assert {"dTHF interface", "copper interface"} <= set(build_problem(free).labels())

    held = dict(
        _defn(data_file, {**COPPER, "roughness_fixed": True}), back_reflection=True
    )
    labels = build_problem(held).labels()
    assert "dTHF interface" not in labels and "copper interface" not in labels
    sample = build_experiment(held).sample
    assert sample[0].interface.value == pytest.approx(5.0)
    assert sample[1].interface.value == pytest.approx(5.0)


def test_a_held_roughness_wins_over_its_roughness_tie(data_file, caplog):
    """One says the roughness is a known number, the other that it follows
    the thickness. The declared constant wins, and says so."""
    tie = {"roughness_tie": {"fraction_max": 0.5}}
    tied = build_problem(_defn(data_file, {**COPPER, **tie}))
    assert "copper rough_frac" in tied.labels()

    with caplog.at_level("WARNING"):
        held = build_problem(
            _defn(data_file, {**COPPER, **tie, "roughness_fixed": True})
        )
    assert "copper rough_frac" not in held.labels()
    assert "copper interface" not in held.labels()
    assert "roughness_tie is ignored" in caplog.text


def test_a_named_interface_can_be_held(data_file):
    defn = _defn(data_file, COPPER)
    defn["interfaces"] = [
        {"below": "Si", "above": "copper", "roughness": 2.5, "roughness_fixed": True}
    ]
    assert "Si interface" not in build_problem(defn).labels()
    assert build_experiment(defn).sample[0].interface.value == pytest.approx(2.5)


def test_the_initial_build_leaves_a_held_roughness_alone():
    """The fringe-derived roughness replaces a larger declared one — not one
    declared known."""
    from aure.nodes.modeling import _build_layers

    features = {"estimated_roughness": 2.0}
    (free,) = _build_layers({"layers": [dict(COPPER)]}, features)
    (flagged,) = _build_layers(
        {"layers": [{**COPPER, "roughness_fixed": True}]}, features
    )
    (window,) = _build_layers(
        {"layers": [{**COPPER, "roughness_min": 5.0, "roughness_max": 5.0}]}, features
    )

    assert free["roughness"] == 2.0
    assert flagged["roughness"] == 5.0 and flagged["roughness_fixed"] is True
    assert (window["roughness"], window["roughness_min"]) == (5.0, 5.0)


def test_wideners_and_counts_leave_a_held_roughness():
    from aure.nodes.evaluation import _count_free_params, _expand_model_bounds
    from aure.nodes.modeling import _widen_all_bounds_json

    layer = {**COPPER, "roughness_min": 5.0, "roughness_max": 5.0}
    model = {
        "layers": [layer],
        "substrate": {
            "name": "Si",
            "sld": 2.07,
            "roughness": 3.0,
            "roughness_max": 15.0,
            "roughness_fixed": True,
        },
        "ambient": {"name": "air", "sld": 0.0},
        "intensity": {"fixed": True},
    }
    (widened,) = _widen_all_bounds_json(model)["layers"]
    assert (widened["roughness_min"], widened["roughness_max"]) == (5.0, 5.0)

    hits = [
        {"name": "copper interface", "bound_hit": "upper"},
        {"name": "Si interface", "bound_hit": "upper"},
    ]
    expanded = _expand_model_bounds(model, hits)
    assert expanded["layers"][0]["roughness_max"] == 5.0
    assert expanded["substrate"]["roughness_max"] == 15.0

    # sld + thickness free; the layer's and the substrate's roughness held.
    assert _count_free_params(model) == 2


def test_restore_covers_roughness_and_drops_a_tie_added_to_it():
    from aure.nodes.modeling import _restore_baseline_pins

    baseline = _baseline()
    baseline["layers"][0]["roughness_fixed"] = True
    baseline["substrate"]["roughness_fixed"] = True
    model = json.loads(json.dumps(baseline))
    model["layers"][0]["roughness_tie"] = {"fraction_max": 0.5}
    model["substrate"].update(roughness=9.0, roughness_fixed=False)

    restored = _restore_baseline_pins(baseline, model)

    assert sorted(restored) == ["Si roughness", "copper roughness"]
    assert "roughness_tie" not in model["layers"][0]
    assert model["substrate"] == baseline["substrate"]


def test_the_explanation_and_prompts_cover_roughness():
    from aure.nodes.modeling import _explain_model
    from aure.nodes.prompts import (
        format_model_refinement_prompt_json,
        format_sample_parse_prompt,
    )

    text = _explain_model(
        [{**COPPER, "sld_min": 4.0, "sld_max": 9.0, "roughness_fixed": True}],
        {"name": "Si", "sld": 2.07, "roughness": 3.0, "roughness_fixed": True},
        {"name": "air", "sld": 0.0},
        {},
    )
    assert "σ fixed at 5.0 Å" in text
    assert "Substrate: σ fixed at 3.0 Å" in text

    assert '"roughness_fixed"' in format_sample_parse_prompt("Cu on Si")
    refine = format_model_refinement_prompt_json(
        current_model=_baseline(),
        sample_description="Cu on Si",
        fit_result={
            "chi_squared": 5.0,
            "parameters": {},
            "issues": [],
            "suggestions": [],
        },
        features={},
    )
    assert '"roughness_fixed"' in refine
    assert "Never add it to a layer whose" in refine


# ----------------------------------------------------------------------
# Held and tied: the held value wins; two different held values are refused
# ----------------------------------------------------------------------

_FREE = {
    "name": "copper",
    "sld": 6.5,
    "sld_min": 5.0,
    "sld_max": 8.0,
    "thickness": 250.0,
    "thickness_min": 100.0,
    "thickness_max": 400.0,
    "roughness": 5.0,
}
_HELD_A = {
    **_FREE,
    "sld": 6.48,
    "sld_fixed": True,
    "thickness": 200.0,
    "thickness_fixed": True,
    "roughness": 4.0,
    "roughness_fixed": True,
}
_HELD_B = {
    **_FREE,
    "sld": 6.0,
    "sld_fixed": True,
    "thickness": 300.0,
    "thickness_fixed": True,
    "roughness": 8.0,
    "roughness_fixed": True,
}


def _two_states(data_file, a_layer=None, b_layer=None, **extra) -> dict:
    def state(name, layer):
        st = {"name": name, "data_files": [{"file": data_file, "label": name}]}
        if layer is not None:
            st["layers"] = [dict(layer)]
        return st

    return {
        "substrate": {"name": "Si", "sld": 2.07, "roughness": 3.0},
        "ambient": {"name": "air", "sld": 0.0},
        "layers": [dict(_FREE)],
        "intensity": {"value": 1.0, "fixed": True},
        "states": [state("A", a_layer), state("B", b_layer)],
        **extra,
    }


def _copper(by_state, name) -> tuple:
    slab = by_state[name][0].sample[1]
    return tuple(
        (par.value, bool(par.fixed))
        for par in (slab.material.rho, slab.thickness, slab.interface)
    )


@pytest.mark.parametrize(
    "a_layer, b_layer, expected",
    [
        (None, _HELD_B, ((6.0, True), (300.0, True), (8.0, True))),
        (_HELD_A, None, ((6.48, True), (200.0, True), (4.0, True))),
        (_HELD_A, _HELD_A, ((6.48, True), (200.0, True), (4.0, True))),
    ],
    ids=["held in a later state", "held in the first state", "held in both, agreeing"],
)
def test_a_value_held_in_any_tied_state_is_held_in_all(
    data_file, a_layer, b_layer, expected
):
    """A tie says one quantity; a held value says that quantity is known. It
    used to work only when the holding state came first — held in a later
    state, the pin was dropped and both states were fitted."""
    problem, by_state, _ = build_states_problem(
        _two_states(data_file, a_layer, b_layer)
    )

    assert _copper(by_state, "A") == expected
    assert _copper(by_state, "B") == expected
    assert not any("copper" in name for name in problem.labels())


def test_one_tied_parameter_held_at_two_values_is_refused(data_file):
    """The first state's value used to win without a word."""
    with pytest.raises(ValueError) as excinfo:
        build_states_problem(_two_states(data_file, _HELD_A, _HELD_B))

    message = str(excinfo.value)
    assert "tied across states but held at different values" in message
    assert "A=" in message and "B=" in message
    assert "unshared_parameters" in message


def test_an_untied_parameter_is_held_per_state(data_file):
    """Untying is the way to say the two states really differ."""
    defn = _two_states(
        data_file,
        _HELD_A,
        _HELD_B,
        unshared_parameters=[
            "copper.material.rho",
            "copper.thickness",
            "copper.interface",
        ],
    )
    _, by_state, _ = build_states_problem(defn)

    assert _copper(by_state, "A") == ((6.48, True), (200.0, True), (4.0, True))
    assert _copper(by_state, "B") == ((6.0, True), (300.0, True), (8.0, True))


def test_a_contradictory_setup_is_refused_when_the_model_is_made(data_file):
    """Not when the first fit is built: the modeling node already has what it
    needs to see it, and no data file has to be read."""
    from aure.nodes.modeling import _build_initial_model

    defn = _two_states("/nonexistent/never-read.dat", _HELD_A, _HELD_B)
    out = _build_initial_model(
        {
            "data_file": "/nonexistent/never-read.dat",
            "parsed_sample": {
                "substrate": defn["substrate"],
                "layers": [dict(_FREE)],
                "ambient": defn["ambient"],
            },
            "extracted_features": {},
            "states": defn["states"],
            "user_config": {},
            "messages": [],
        }
    )
    assert "held at different values" in out["error"]


def test_a_held_substrate_roughness_holds_its_boundary_in_back_reflection(data_file):
    """In back reflection the substrate boundary is carried by the bottom
    layer's slab, and the substrate slab's own interface is unused. A real
    intake run set `roughness_fixed` on the substrate there and nothing was
    held: the boundary stayed fitted at the bottom layer's declared value."""
    titanium = {"name": "titanium", "sld": -1.9, "thickness": 30.0, "roughness": 6.0}
    defn = dict(_defn(data_file, titanium), back_reflection=True)
    defn["layers"].append(dict(COPPER, roughness=8.0))

    free = build_problem(defn)
    assert "titanium interface" in free.labels()

    defn["substrate"]["roughness_fixed"] = True  # declared roughness: 3.0
    held = build_problem(defn)
    assert "titanium interface" not in held.labels()
    # stack is ambient-first: [dTHF, copper, titanium, Si]
    boundary = build_experiment(defn).sample[2].interface
    assert boundary.fixed and boundary.value == pytest.approx(3.0)
    # The other interfaces are untouched.
    assert {"dTHF interface", "copper interface"} <= set(held.labels())
