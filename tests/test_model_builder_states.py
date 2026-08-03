"""Tests for build_states_problem (cross-state co-refinement)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _make_data_file(q_min: float = 0.01, q_max: float = 0.10, n: int = 80) -> str:
    Q = np.linspace(q_min, q_max, n)
    Qc = 0.0217
    R = np.where(Q < Qc, 1.0, (Qc / (2 * Q)) ** 4)
    R *= 1 + 0.2 * np.cos(2 * Q * 100.0)
    R = np.clip(R, 1e-10, 1.0)
    dR = 0.05 * R
    dQ = 0.02 * Q
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False)
    f.write("# Q R dR dQ\n")
    for q, r, dr, dq in zip(Q, R, dR, dQ):
        f.write(f"{q:.6f}  {r:.6e}  {dr:.6e}  {dq:.6e}\n")
    f.close()
    return f.name


@pytest.fixture
def two_files():
    paths = [_make_data_file(0.01, 0.10, 60), _make_data_file(0.01, 0.10, 60)]
    yield paths
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


def _two_state_definition(files: list[str], **extra) -> dict:
    base = {
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
            {
                "name": "Ti",
                "sld": -1.95,
                "sld_min": -3.0,
                "sld_max": 0.0,
                "thickness": 30.0,
                "thickness_min": 10.0,
                "thickness_max": 60.0,
                "roughness": 3.0,
                "roughness_max": 15.0,
            },
        ],
        "ambient": {"name": "air", "sld": 0.0},
        "back_reflection": False,
        "intensity": {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": False},
        "states": [
            {
                "name": "D2O",
                "data_files": [{"file": files[0], "label": "D2O-comb"}],
                "ambient": {"name": "D2O", "sld": 6.4, "sld_min": 5.0, "sld_max": 6.6},
            },
            {
                "name": "H2O",
                "data_files": [{"file": files[1], "label": "H2O-comb"}],
                "ambient": {
                    "name": "H2O",
                    "sld": -0.56,
                    "sld_min": -0.6,
                    "sld_max": 0.0,
                },
            },
        ],
    }
    base.update(extra)
    return base


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_default_tied_set_aliases_structural_params(two_files):
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files)
    problem, by_state, by_files = build_states_problem(defn)

    assert set(by_state.keys()) == {"D2O", "H2O"}
    s0 = by_state["D2O"][0].sample
    s1 = by_state["H2O"][0].sample
    # samples themselves are distinct objects (different ambients)
    assert s0 is not s1
    # but the Cu thickness Parameter object is shared
    assert s1[1].thickness is s0[1].thickness
    assert s1[1].material.rho is s0[1].material.rho
    assert s1[1].interface is s0[1].interface
    # Ti
    assert s1[2].thickness is s0[2].thickness
    assert s1[2].material.rho is s0[2].material.rho
    # substrate.interface tied
    assert s1[0].interface is s0[0].interface
    # ambient SLD NOT tied (different solvents)
    assert s1[3].material.rho is not s0[3].material.rho


def test_shared_parameters_whitelist_replaces_default(two_files):
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files, shared_parameters=["Cu.thickness"])
    _, by_state, _ = build_states_problem(defn)
    s0 = by_state["D2O"][0].sample
    s1 = by_state["H2O"][0].sample

    # Cu.thickness tied
    assert s1[1].thickness is s0[1].thickness
    # everything else NOT tied
    assert s1[1].material.rho is not s0[1].material.rho
    assert s1[1].interface is not s0[1].interface
    assert s1[2].thickness is not s0[2].thickness


def test_unshared_parameters_subtracts_from_default(two_files):
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files, unshared_parameters=["Cu.thickness"])
    _, by_state, _ = build_states_problem(defn)
    s0 = by_state["D2O"][0].sample
    s1 = by_state["H2O"][0].sample

    # Cu.thickness explicitly not tied
    assert s1[1].thickness is not s0[1].thickness
    # but Cu.material.rho still tied (default)
    assert s1[1].material.rho is s0[1].material.rho


def test_unshared_substrate_alias_actually_unties(two_files):
    """`substrate.<attr>` in unshared_parameters must match the default tie.

    Regression: the default substrate tie is keyed by ``sub_name`` (e.g.
    ``"silicon"``), but users / docs spell it ``"substrate"``. Before
    normalization, the blacklist was silently ignored for the substrate.
    """
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files, unshared_parameters=["substrate.interface"])
    _, by_state, _ = build_states_problem(defn)
    s0 = by_state["D2O"][0].sample
    s1 = by_state["H2O"][0].sample

    # substrate interface explicitly NOT tied
    assert s1[0].interface is not s0[0].interface
    # but Cu.thickness still tied (default)
    assert s1[1].thickness is s0[1].thickness


def test_shared_substrate_alias_ties_and_keeps_canonical_name(two_files):
    """`substrate.<attr>` in shared_parameters ties and uses the canonical name.

    Regression: the rename pass keyed off ``sub_name``; with the alias
    spelling, the shared parameter was relabeled with a state prefix
    even though it was effectively shared.
    """
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files, shared_parameters=["substrate.interface"])
    _, by_state, _ = build_states_problem(defn)
    s0 = by_state["D2O"][0].sample
    s1 = by_state["H2O"][0].sample

    # substrate.interface tied via alias spelling
    assert s1[0].interface is s0[0].interface
    # and the shared parameter keeps the tied (state-unprefixed) name
    assert s0[0].interface.name == "silicon interface"


def test_unknown_layer_in_shared_parameters_raises(two_files):
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files, shared_parameters=["Aluminum.thickness"])
    with pytest.raises(ValueError, match="Aluminum"):
        build_states_problem(defn)


def test_shared_and_unshared_mutually_exclusive(two_files):
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(
        two_files,
        shared_parameters=["Cu.thickness"],
        unshared_parameters=["Cu.interface"],
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_states_problem(defn)


def test_returns_fit_problem_with_all_experiments(two_files):
    from aure.nodes.model_builder import build_states_problem
    from bumps.fitproblem import FitProblem

    defn = _two_state_definition(two_files)
    problem, by_state, _ = build_states_problem(defn)
    assert isinstance(problem, FitProblem)
    n = sum(len(v) for v in by_state.values())
    assert n == 2


def test_single_state_via_build_states_problem(two_files):
    from aure.nodes.model_builder import build_states_problem
    from bumps.fitproblem import FitProblem

    defn = _two_state_definition(two_files)
    defn["states"] = defn["states"][:1]
    problem, by_state, _ = build_states_problem(defn)
    assert isinstance(problem, FitProblem)
    assert list(by_state.keys()) == ["D2O"]


def test_save_problem_json_routes_multi_state(tmp_path, two_files):
    from aure.nodes.model_builder import save_problem_json

    defn = _two_state_definition(two_files)
    out = save_problem_json(defn, tmp_path / "problem.json")
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_save_problem_json_round_trip_with_bumps(tmp_path, two_files):
    from aure.nodes.model_builder import save_problem_json
    from bumps.serialize import load_file

    defn = _two_state_definition(two_files)
    out = save_problem_json(defn, tmp_path / "problem.json")
    loaded = load_file(out)
    assert loaded is not None


def test_ui_ambient_rho_override_merges_into_model_ambient(two_files):
    """A partial ambient override from the UI (``{rho: X}``) must merge
    into the model-level ambient and translate ``rho`` → ``sld``.

    Regression: ``_state_overrides`` used to replace ``eff["ambient"]``
    wholesale with the UI payload, leaving the resulting dict without
    ``name``/``sld``. ``_build_sample`` then raised ``KeyError`` reading
    ``ambient_info["name"]``.
    """
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files)
    # Simulate the UI shape: only `rho`, no `name`/`sld`.
    defn["states"][0]["ambient"] = {"rho": 6.4}
    defn["states"][1]["ambient"] = {"rho": -0.56}

    # Should build without KeyError.
    _, by_state, _ = build_states_problem(defn)
    s0 = by_state["D2O"][0].sample
    s1 = by_state["H2O"][0].sample

    # State-0 ambient SLD should reflect the override (6.4); state-1 (-0.56).
    # Ambient slot is sample[n+1] in normal orientation (n=2 layers).
    assert abs(float(s0[3].material.rho.value) - 6.4) < 1e-9
    assert abs(float(s1[3].material.rho.value) - (-0.56)) < 1e-9


def test_mixed_back_reflection_orientations_rejected(two_files):
    """Mixed back-reflection orientations across states are rejected.

    Refl1d ranges substrate.interface on sample[0] in the normal stack
    and on sample[n+1] in the back-reflection stack, so cross-state
    aliasing silently drops the range on one side. Reject up front.
    """
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files)
    defn["states"][1]["back_reflection"] = True
    with pytest.raises(ValueError, match="back_reflection"):
        build_states_problem(defn)


def test_back_reflection_substrate_interface_indexed_correctly(two_files):
    """Substrate interface must alias to sample[n+1] in back-reflection.

    Regression: ``_layer_index`` returned ``n`` for substrate in
    back-reflection geometry, colliding with the topmost layer
    (``sample[n - 0] = sample[n]``). The default tie ``(sub_name,
    "interface")`` would then silently alias the wrong layer's
    interface across states.

    Stack ordering for back_reflection with layers=[Cu, Ti] (n=2):
      sample[0]=ambient, sample[1]=Ti, sample[2]=Cu, sample[3]=substrate.
    """
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files, back_reflection=True)
    _, by_state, _ = build_states_problem(defn)
    s0 = by_state["D2O"][0].sample  # back: [amb, Ti, Cu, sub]
    s1 = by_state["H2O"][0].sample  # back: [amb, Ti, Cu, sub]

    # Substrate.interface tied: both at sample[3]
    assert s1[3].interface is s0[3].interface
    # And it must NOT have been incorrectly aliased onto Cu (sample[2]).
    assert s1[2].interface is not s0[3].interface
    # Cu.interface tied at sample[2]
    assert s1[2].interface is s0[2].interface
    # Ti.interface tied at sample[1]
    assert s1[1].interface is s0[1].interface


# ----------------------------------------------------------------------
# Per-state structure ("sample != structure")
# ----------------------------------------------------------------------

_OXIDE = {
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


def test_resolve_tied_set_union_does_not_raise_for_per_state_layer(two_files):
    """A layer present in only some states is a valid tie target (union); the
    default tied set spans the union and resolution must not raise."""
    from aure.nodes.model_builder import _resolve_tied_set, _valid_layer_names

    defn = _two_state_definition(two_files)
    defn["layers"] = [_OXIDE, *defn["layers"]]  # template: [Cu oxide, Cu, Ti]
    defn["states"][1]["layers"] = list(defn["layers"][1:])  # H2O omits the oxide
    valid = _valid_layer_names(defn)
    assert "Cu oxide" in valid  # still a valid name (present in D2O / template)

    # shared/unshared naming the per-state-absent oxide must NOT raise.
    defn_u = dict(defn, unshared_parameters=["Cu oxide.interface"])
    _resolve_tied_set(defn_u)  # no raise
    pairs = _resolve_tied_set(defn)
    assert ("Cu oxide", "thickness") in pairs  # default set spans the union
    assert ("Cu", "thickness") in pairs


def test_per_state_structure_oxide_absent_in_one_state(two_files):
    """state D2O has the oxide; H2O omits it. Build succeeds; layers shared by
    BOTH states tie by identity across the differing stacks; the oxide tie
    simply does not apply to H2O."""
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files)
    defn["layers"] = [_OXIDE, *defn["layers"]]  # [Cu oxide, Cu, Ti]
    defn["states"][1]["layers"] = list(defn["layers"][1:])  # H2O = [Cu, Ti] (no oxide)

    problem, by_state, _ = build_states_problem(defn)
    s0 = by_state["D2O"][0].sample  # [Si, Cu oxide, Cu, Ti, D2O]
    s1 = by_state["H2O"][0].sample  # [Si, Cu, Ti, H2O]

    assert len(s1) == len(s0) - 1  # H2O is missing the oxide layer
    # Cu and Ti exist in both states and are tied by IDENTITY despite the
    # oxide shifting D2O's indices (Cu: s0[2] vs s1[1]).
    assert s1[1].thickness is s0[2].thickness  # Cu tied
    assert s1[1].material.rho is s0[2].material.rho
    assert s1[2].thickness is s0[3].thickness  # Ti tied
    # substrate.interface still tied; ambient (different solvents) not tied
    assert s1[0].interface is s0[0].interface
    assert s1[-1].material.rho is not s0[-1].material.rho
