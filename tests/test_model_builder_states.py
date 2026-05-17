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


def test_back_reflection_per_state(two_files):
    """A state can override back_reflection independently of state 0."""
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files)
    defn["states"][1]["back_reflection"] = True
    # Should still build without crashing; aliasing still resolves layer
    # indices via per-state back_reflection.
    problem, by_state, _ = build_states_problem(defn)
    s0 = by_state["D2O"][0].sample
    s1 = by_state["H2O"][0].sample
    # In s1 (back_reflection=True) Cu is at index len(layers)-0 = 2; in s0 it's at 1.
    # Tied Cu.thickness must still be the same Parameter object.
    assert s0[1].thickness is s1[2].thickness
