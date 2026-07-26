"""Tests for the outermost-interface roughness ceiling (back reflection).

In back reflection the stack is assembled ambient-first, so
``sample[0].interface`` is the topmost layer's outer interface. Its bound used to
be hardcoded at 30 A, which cannot represent a diffuse solvent-swollen SEI
(expert reference fits of Cu-in-THF electrodes put it above 30 A in 40 of 51
runs, up to 209 A). It now follows the topmost layer's ``roughness_max``, with a
``ROUGHNESS_MAX_OUTER`` override, and keeps its 30 A default when unset.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from aure.nodes.model_builder import _outer_roughness_max, build_problem


def _make_data_file(q_min: float = 0.01, q_max: float = 0.12, n: int = 90) -> str:
    Q = np.linspace(q_min, q_max, n)
    Qc = 0.0217
    R = np.where(Q < Qc, 1.0, (Qc / (2 * Q)) ** 4)
    R *= 1 + 0.2 * np.cos(2 * Q * 300.0)
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
def data_file():
    p = _make_data_file()
    yield p
    try:
        os.unlink(p)
    except OSError:
        pass


@pytest.fixture(autouse=True)
def _clear_override():
    """Keep the env override from leaking between tests."""
    os.environ.pop("ROUGHNESS_MAX_OUTER", None)
    yield
    os.environ.pop("ROUGHNESS_MAX_OUTER", None)


def _definition(data_file: str, outer_roughness_max=None) -> dict:
    """Si | Ti | Cu | SEI in THF, measured through the substrate."""
    sei = {
        "name": "SEI",
        "sld": 2.5,
        "sld_min": 0.0,
        "sld_max": 5.0,
        "thickness": 200.0,
        "thickness_min": 50.0,
        "thickness_max": 500.0,
        "roughness": 20.0,
    }
    if outer_roughness_max is not None:
        sei["roughness_max"] = outer_roughness_max
    return {
        "substrate": {"name": "Si", "sld": 2.07, "roughness": 3.0},
        "ambient": {"name": "dTHF", "sld": 6.2},
        "back_reflection": True,
        "data_file": data_file,
        "layers": [
            {
                "name": "Ti",
                "sld": -2.0,
                "thickness": 50.0,
                "roughness": 6.0,
            },
            {
                "name": "Cu",
                "sld": 6.5,
                "thickness": 550.0,
                "roughness": 8.0,
            },
            sei,
        ],
    }


def _outer_bounds(problem):
    """(low, high) bounds of the ambient-side interface parameter."""
    bounds = problem.fitness.sample[0].interface.bounds
    limits = getattr(bounds, "limits", bounds)  # bumps version dependent
    return tuple(float(b) for b in limits)


# ── the helper in isolation ────────────────────────────────────────────


def test_defaults_to_30_when_unspecified():
    assert _outer_roughness_max([{"name": "SEI", "roughness": 20.0}]) == 30.0
    assert _outer_roughness_max([]) == 30.0


def test_follows_topmost_layer_roughness_max():
    layers = [{"name": "Cu", "roughness_max": 20.0},
              {"name": "SEI", "roughness_max": 220.0}]
    assert _outer_roughness_max(layers) == 220.0


def test_env_override_wins():
    os.environ["ROUGHNESS_MAX_OUTER"] = "150"
    layers = [{"name": "SEI", "roughness_max": 40.0}]
    assert _outer_roughness_max(layers) == 150.0


def test_non_numeric_values_fall_back_to_default():
    os.environ["ROUGHNESS_MAX_OUTER"] = "very rough"
    assert _outer_roughness_max([{"name": "SEI"}]) == 30.0
    os.environ.pop("ROUGHNESS_MAX_OUTER")
    assert _outer_roughness_max([{"name": "SEI", "roughness_max": None}]) == 30.0


# ── end-to-end through build_problem ──────────────────────────────────


def test_default_bound_unchanged(data_file):
    """Behaviour must not shift for models that never ask for a wider bound."""
    problem = build_problem(_definition(data_file))
    assert _outer_bounds(problem) == (0.0, 30.0)


def test_diffuse_sei_bound_is_honoured(data_file):
    problem = build_problem(_definition(data_file, outer_roughness_max=220.0))
    assert _outer_bounds(problem) == (0.0, 220.0)


def test_env_override_applies_end_to_end(data_file):
    os.environ["ROUGHNESS_MAX_OUTER"] = "180"
    problem = build_problem(_definition(data_file, outer_roughness_max=40.0))
    assert _outer_bounds(problem) == (0.0, 180.0)


def test_fit_can_reach_a_roughness_above_30(data_file):
    """The bound is what mattered: with it lifted the optimizer can go past 30 A."""
    from bumps.fitters import fit as bumps_fit

    problem = build_problem(_definition(data_file, outer_roughness_max=220.0))
    problem.fitness.sample[0].interface.value = 90.0
    bumps_fit(problem, method="amoeba", steps=200)
    assert float(problem.fitness.sample[0].interface.value) > 30.0


def test_front_reflection_still_uses_the_substrate_bound(data_file):
    """Only the back-reflection branch changes; front reflection is untouched."""
    defn = _definition(data_file, outer_roughness_max=220.0)
    defn["back_reflection"] = False
    defn["substrate"]["roughness_max"] = 12.0
    problem = build_problem(defn)
    assert _outer_bounds(problem) == (0.0, 12.0)
