"""A built parameter must start inside its own range, and be uniquely named.

Both invariants failed silently before: an out-of-range starting value makes
the whole problem infeasible (bumps short-circuits without evaluating the
model, and the fit reports `inf` before it starts), and two parameters sharing
a name collapse wherever results are keyed by name. Neither shows up as
anything wrong in the model JSON.
"""

from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
import pytest

from aure.nodes.model_builder import build_problem, build_states_problem

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


def _defn(data_file: str, layer: dict, **extra) -> dict:
    d = {
        "substrate": {"name": "Si", "sld": 2.07, "roughness": 3.0},
        "ambient": {"name": "air", "sld": 0.0},
        "layers": [layer],
        "data_file": data_file,
        "intensity": {"fixed": True},
    }
    d.update(extra)
    return d


def _param(problem, name: str):
    return next(p for p in problem._parameters if str(p.name) == name)


def _feasible(problem) -> bool:
    return not problem._nllf_components()[3]


# ----------------------------------------------------------------------
# The built-in roughness floor must not contradict a declared roughness
# ----------------------------------------------------------------------


def test_declared_roughness_below_the_default_floor_is_honoured(data_file):
    """A chemically sharp buried interface — "do not impose a roughness floor
    on it" — used to be built at 3 Å with bounds (5, 20): outside its own
    range, so the problem was infeasible before the optimizer ran."""
    problem = build_problem(
        _defn(data_file, {"name": "SiO2", "sld": 2.9, "thickness": 25.0,
                          "roughness": 3.0, "roughness_max": 20.0})
    )
    par = _param(problem, "SiO2 interface")
    assert par.value == pytest.approx(3.0)
    assert par.prior.limits[0] <= 3.0
    assert _feasible(problem)


def test_the_floor_still_applies_when_it_does_not_conflict(data_file):
    problem = build_problem(
        _defn(data_file, {"name": "film", "sld": 2.0, "thickness": 100.0,
                          "roughness": 8.0, "roughness_max": 30.0})
    )
    assert _param(problem, "film interface").prior.limits[0] == pytest.approx(5.0)


def test_an_explicit_roughness_min_is_obeyed(data_file):
    problem = build_problem(
        _defn(data_file, {"name": "film", "sld": 2.0, "thickness": 100.0,
                          "roughness": 8.0, "roughness_min": 0.0,
                          "roughness_max": 30.0})
    )
    assert _param(problem, "film interface").prior.limits[0] == pytest.approx(0.0)


def test_an_explicit_bound_excluding_the_start_clamps_and_stays_feasible(data_file):
    """The model contradicting itself: clamp into the range and log, rather
    than hand the optimizer a point it will refuse to evaluate."""
    problem = build_problem(
        _defn(data_file, {"name": "film", "sld": 2.0, "thickness": 100.0,
                          "roughness": 2.0, "roughness_min": 6.0,
                          "roughness_max": 30.0})
    )
    par = _param(problem, "film interface")
    assert par.value == pytest.approx(6.0)
    assert _feasible(problem)


@pytest.mark.parametrize(
    "layer",
    [
        {"name": "f", "sld": 2.0, "thickness": 100.0, "thickness_min": 150.0,
         "thickness_max": 400.0, "roughness": 5.0},
        {"name": "f", "sld": 2.0, "thickness": 100.0, "sld_min": 3.0,
         "sld_max": 6.0, "roughness": 5.0},
    ],
    ids=["thickness below its min", "sld below its min"],
)
def test_no_parameter_is_ever_built_outside_its_range(data_file, layer):
    problem = build_problem(_defn(data_file, layer))
    for par in problem._parameters:
        lo, hi = par.prior.limits
        assert lo <= par.value <= hi, f"{par.name} = {par.value} outside ({lo}, {hi})"
    assert _feasible(problem)


def test_negative_ambient_sld_does_not_invert_its_range(data_file):
    """H2O (-0.56) hits the multiplicative defaults backwards: min=-0.448,
    max=-0.672. The defaults are left alone (see the TODO at the site) but the
    range must not reach refl1d inverted."""
    problem = build_problem(
        _defn(data_file,
              {"name": "film", "sld": 2.0, "thickness": 100.0, "roughness": 5.0},
              ambient={"name": "H2O", "sld": -0.56})
    )
    par = _param(problem, "H2O rho")
    lo, hi = par.prior.limits
    assert lo < hi
    assert lo <= par.value <= hi


# ----------------------------------------------------------------------
# Every free parameter needs a unique name
# ----------------------------------------------------------------------


def test_back_reflection_states_do_not_share_an_ambient_interface_name(data_file):
    """In back reflection the ambient carries the outer interface, so it is a
    fitted parameter — one per state, and all called "<ambient> interface"
    until they are prefixed."""
    defn = {
        "substrate": {"name": "Si", "sld": 2.07, "roughness": 3.0,
                      "roughness_max": 15.0},
        "ambient": {"name": "dTHF", "sld": 6.2},
        "back_reflection": True,
        "layers": [{"name": "Cu", "sld": 6.55, "thickness": 500.0,
                    "roughness": 8.0, "roughness_max": 25.0}],
        "states": [
            {"name": "a", "back_reflection": True,
             "data_files": [{"file": data_file, "label": "a"}]},
            {"name": "b", "back_reflection": True,
             "data_files": [{"file": data_file, "label": "b"}]},
        ],
    }
    problem, _exps, _ = build_states_problem(defn)
    names = [str(p.name) for p in problem._parameters]
    dupes = [n for n in names if names.count(n) > 1]
    assert len(names) == len(set(names)), f"duplicates: {dupes}"
    assert {"a dTHF interface", "b dTHF interface"} <= set(names)
