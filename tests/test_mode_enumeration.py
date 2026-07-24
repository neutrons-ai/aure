"""Tests for gated thin-layer SLD mode enumeration (fitting)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from aure.nodes.fitting import (
    _mode_enumeration_enabled,
    _resolution_limit,
    _enumerate_thin_layer_modes,
)
from aure.nodes.model_builder import build_experiment, build_problem


def test_enabled_flag(monkeypatch):
    monkeypatch.delenv("MODE_ENUMERATION", raising=False)
    assert _mode_enumeration_enabled() is False
    monkeypatch.setenv("MODE_ENUMERATION", "1")
    assert _mode_enumeration_enabled() is True
    monkeypatch.setenv("MODE_ENUMERATION", "off")
    assert _mode_enumeration_enabled() is False


def test_resolution_limit():
    q = np.linspace(0.008, 0.25, 50)
    d = _resolution_limit({"Q": q.tolist()})
    assert d == pytest.approx(2 * np.pi / 0.25, rel=1e-6)
    assert _resolution_limit({"Q": []}) is None


def _truth_definition(data_file: str, thin_sld: float) -> dict:
    return {
        "substrate": {"name": "Si", "sld": 2.07, "roughness": 3.0, "roughness_max": 10.0},
        "ambient": {"name": "Si", "sld": 2.07},
        "back_reflection": False,
        "data_file": data_file,
        "intensity": {"value": 1.0, "fixed": True},
        "layers": [
            {
                "name": "Film",
                "sld": 4.5,
                "sld_min": 3.5,
                "sld_max": 5.5,
                "thickness": 400.0,
                "thickness_min": 380.0,
                "thickness_max": 420.0,
                "roughness": 6.0,
                "roughness_max": 12.0,
            },
            {
                "name": "Thin",
                "sld": thin_sld,
                "sld_min": -1.0,
                "sld_max": 5.0,
                "thickness": 15.0,
                "thickness_min": 12.0,
                "thickness_max": 18.0,
                "roughness": 5.0,
                "roughness_max": 7.0,
            },
        ],
    }


@pytest.fixture
def synthetic_case():
    """Generate reflectivity data from a stack whose thin layer has SLD=0.0,
    returning (data_file, Q). The thin layer is 15 Å — below the ~25 Å
    resolution limit at Q_max=0.25."""
    Q = np.linspace(0.008, 0.25, 220)
    # dummy file so build_experiment can load a probe on this Q grid
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False)
    f.write("# Q R dR dQ\n")
    for q in Q:
        f.write(f"{q:.6f}  {1.0:.6e}  {0.02:.6e}  {0.02 * q:.6e}\n")
    f.close()

    truth = _truth_definition(f.name, thin_sld=0.0)
    exp = build_experiment(truth)
    _, R_true = exp.reflectivity()
    dR = np.maximum(0.02 * R_true, 1e-9)
    with open(f.name, "w") as fh:
        fh.write("# Q R dR dQ\n")
        for q, r, dr in zip(Q, R_true, dR):
            fh.write(f"{q:.6f}  {r:.6e}  {dr:.6e}  {0.02 * q:.6e}\n")

    yield f.name, Q
    try:
        os.unlink(f.name)
    except OSError:
        pass


def test_enumeration_recovers_better_basin(synthetic_case):
    data_file, Q = synthetic_case
    # Seed the thin layer FAR from the truth (0.0), at the opposite end.
    bad = _truth_definition(data_file, thin_sld=4.5)
    chi2_before = build_problem(bad).chisq()

    reseeded = _enumerate_thin_layer_modes(bad, {"Q": Q.tolist()})
    thin = next(ly for ly in reseeded["layers"] if ly["name"] == "Thin")
    chi2_after = build_problem(reseeded).chisq()

    # Re-seeded toward the true basin and no worse in χ².
    assert abs(thin["sld"] - 0.0) < abs(4.5 - 0.0)
    assert chi2_after <= chi2_before + 1e-6


def test_thick_only_model_unchanged(synthetic_case):
    data_file, Q = synthetic_case
    # A model with no thin layers: enumeration is a no-op.
    defn = _truth_definition(data_file, thin_sld=0.0)
    defn["layers"] = [defn["layers"][0]]  # keep only the 400 Å film
    out = _enumerate_thin_layer_modes(defn, {"Q": Q.tolist()})
    assert out["layers"][0]["sld"] == defn["layers"][0]["sld"]
