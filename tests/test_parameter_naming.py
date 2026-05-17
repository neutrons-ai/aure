"""Tests for parameter naming and round-trip in multi-state co-refinement (Ticket 04)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest


def _make_data_file(q_min: float = 0.01, q_max: float = 0.10, n: int = 60) -> str:
    Q = np.linspace(q_min, q_max, n)
    Qc = 0.0217
    R = np.where(Q < Qc, 1.0, (Qc / (2 * Q)) ** 4)
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
    p = [_make_data_file(), _make_data_file()]
    yield p
    for f in p:
        try:
            os.unlink(f)
        except OSError:
            pass


def _two_state_def(files: list[str], **extra) -> dict:
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
                "name": "CuOx",
                "sld": 3.5,
                "sld_min": 1.0,
                "sld_max": 6.0,
                "thickness": 20.0,
                "thickness_min": 5.0,
                "thickness_max": 40.0,
                "roughness": 5.0,
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


def test_parameter_key_helper():
    from aure.nodes.model_builder import parameter_key

    assert parameter_key("D2O", "Cu", "thickness", tied=True) == "Cu thickness"
    assert parameter_key("D2O", "Cu", "thickness", tied=False) == "D2O Cu thickness"
    assert parameter_key("H2O", "Cu", "material.rho", tied=False) == "H2O Cu rho"
    assert (
        parameter_key("D2O", "substrate", "interface", tied=True)
        == "substrate interface"
    )


def test_unshared_creates_state_prefixed_names(two_files):
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_def(two_files, unshared_parameters=["CuOx.thickness"])
    problem, _, _ = build_states_problem(defn)
    names = [str(p.name) for p in problem._parameters]

    # tied params present once each
    assert names.count("Cu thickness") == 1
    assert names.count("Cu rho") == 1
    assert names.count("CuOx rho") == 1
    # untied CuOx.thickness present once per state, prefixed
    assert "D2O CuOx thickness" in names
    assert "H2O CuOx thickness" in names
    assert "CuOx thickness" not in names


def test_all_parameter_names_are_unique(two_files):
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_def(two_files, unshared_parameters=["CuOx.thickness"])
    problem, _, _ = build_states_problem(defn)
    names = [str(p.name) for p in problem._parameters]
    assert len(names) == len(set(names)), (
        f"Duplicates: {[n for n in names if names.count(n) > 1]}"
    )


def test_intensities_are_per_state_named(two_files):
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_def(two_files)
    problem, _, _ = build_states_problem(defn)
    names = [str(p.name) for p in problem._parameters]
    assert "D2O intensity" in names
    assert "H2O intensity" in names
    assert names.count("intensity") == 0


def test_extract_definition_round_trip(two_files):
    from aure.nodes.model_builder import build_states_problem, extract_definition

    defn = _two_state_def(two_files, unshared_parameters=["CuOx.thickness"])
    problem, _, _ = build_states_problem(defn)

    # Mutate fitted values: make D2O and H2O CuOx thicknesses different.
    for par in problem._parameters:
        if str(par.name) == "D2O CuOx thickness":
            par.value = 22.0
        elif str(par.name) == "H2O CuOx thickness":
            par.value = 17.0
        elif str(par.name) == "Cu thickness":
            par.value = 510.0  # tied — shared across states

    out = extract_definition(problem, defn)

    assert "states" in out
    d2o = next(s for s in out["states"] if s["name"] == "D2O")
    h2o = next(s for s in out["states"] if s["name"] == "H2O")
    # Untied CuOx thickness differs per state
    cuox_d2o = next(L for L in d2o["layers"] if L["name"] == "CuOx")
    cuox_h2o = next(L for L in h2o["layers"] if L["name"] == "CuOx")
    assert cuox_d2o["thickness"] == pytest.approx(22.0)
    assert cuox_h2o["thickness"] == pytest.approx(17.0)
    # Tied Cu thickness matches across states
    cu_d2o = next(L for L in d2o["layers"] if L["name"] == "Cu")
    cu_h2o = next(L for L in h2o["layers"] if L["name"] == "Cu")
    assert cu_d2o["thickness"] == cu_h2o["thickness"] == pytest.approx(510.0)
    # Top-level layers carry the tied baseline
    cu_top = next(L for L in out["layers"] if L["name"] == "Cu")
    assert cu_top["thickness"] == pytest.approx(510.0)


def test_apply_parameters_accepts_canonical_names(two_files):
    from aure.nodes.model_builder import apply_parameters, build_states_problem

    defn = _two_state_def(two_files, unshared_parameters=["CuOx.thickness"])
    problem, _, _ = build_states_problem(defn)
    apply_parameters(
        problem,
        {
            "Cu thickness": 480.0,
            "D2O CuOx thickness": 21.0,
            "H2O CuOx thickness": 16.5,
        },
    )
    by_name = {str(p.name): p for p in problem._parameters}
    assert by_name["Cu thickness"].value == pytest.approx(480.0)
    assert by_name["D2O CuOx thickness"].value == pytest.approx(21.0)
    assert by_name["H2O CuOx thickness"].value == pytest.approx(16.5)


def test_apply_parameters_legacy_short_name_broadcasts(two_files):
    """A legacy 'CuOx thickness' key (no state prefix) updates every untied match."""
    from aure.nodes.model_builder import apply_parameters, build_states_problem

    defn = _two_state_def(two_files, unshared_parameters=["CuOx.thickness"])
    problem, _, _ = build_states_problem(defn)
    apply_parameters(problem, {"CuOx thickness": 18.5})
    by_name = {str(p.name): p for p in problem._parameters}
    assert by_name["D2O CuOx thickness"].value == pytest.approx(18.5)
    assert by_name["H2O CuOx thickness"].value == pytest.approx(18.5)


def test_apply_parameters_idempotent(two_files):
    from aure.nodes.model_builder import apply_parameters, build_states_problem

    defn = _two_state_def(two_files, unshared_parameters=["CuOx.thickness"])
    problem, _, _ = build_states_problem(defn)
    fitted = {
        "Cu thickness": 495.0,
        "D2O CuOx thickness": 19.0,
        "H2O CuOx thickness": 21.0,
    }
    apply_parameters(problem, fitted)
    apply_parameters(problem, fitted)  # second application same result
    by_name = {str(p.name): p for p in problem._parameters}
    assert by_name["Cu thickness"].value == pytest.approx(495.0)
    assert by_name["D2O CuOx thickness"].value == pytest.approx(19.0)


def test_single_state_names_unchanged():
    """Regression: the legacy single-state build_problem is untouched."""
    from aure.nodes.model_builder import build_problem

    files = [_make_data_file()]
    try:
        defn = {
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
            "data_file": files[0],
            "intensity": {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": False},
        }
        problem = build_problem(defn)
        names = [str(p.name) for p in problem._parameters]
        assert "Cu thickness" in names
        assert "Cu rho" in names
        # No state prefix in single-state mode
        assert not any(n.startswith("default ") for n in names)
    finally:
        for f in files:
            os.unlink(f)
