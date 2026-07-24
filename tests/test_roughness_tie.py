"""Tests for tied roughness (sigma = fraction * thickness)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from aure.nodes.model_builder import build_problem, extract_definition


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


def _definition(data_file: str, tie: bool) -> dict:
    siox = {
        "name": "SiOx",
        "sld": 3.0,
        "sld_min": 2.2,
        "sld_max": 3.5,
        "thickness": 20.0,
        "thickness_min": 10.0,
        "thickness_max": 40.0,
        "roughness": 6.0,
        "roughness_max": 15.0,
    }
    if tie:
        siox["roughness_tie"] = {
            "fraction_init": 0.3,
            "fraction_min": 0.05,
            "fraction_max": 0.5,
        }
    return {
        "substrate": {"name": "Si", "sld": 2.07, "roughness": 3.0, "roughness_max": 15.0},
        "ambient": {"name": "Si", "sld": 2.07},
        "back_reflection": False,
        "data_file": data_file,
        "layers": [
            {
                "name": "Cu",
                "sld": 6.5,
                "sld_min": 4.5,
                "sld_max": 8.5,
                "thickness": 500.0,
                "thickness_min": 300.0,
                "thickness_max": 700.0,
                "roughness": 8.0,
                "roughness_max": 20.0,
            },
            siox,
        ],
    }


def test_tie_creates_fraction_parameter(data_file):
    problem = build_problem(_definition(data_file, tie=True))
    names = [str(p.name) for p in problem._parameters]
    assert "SiOx rough_frac" in names
    # The interface is now a derived parameter, not a free one.
    assert "SiOx interface" not in names


def test_tie_holds_after_fit(data_file):
    from bumps.fitters import fit as bumps_fit

    problem = build_problem(_definition(data_file, tie=True))
    bumps_fit(problem, method="amoeba", steps=200)

    # Locate the SiOx slab and confirm sigma == fraction * thickness.
    sample = problem.fitness.sample
    siox = next(s for s in sample if s.material.name == "SiOx")
    sigma = float(siox.interface.value)
    thick = float(siox.thickness.value)
    frac = next(
        float(p.value) for p in problem._parameters if str(p.name) == "SiOx rough_frac"
    )
    assert sigma == pytest.approx(frac * thick, rel=1e-6)
    # And it never outgrows half the layer (fraction_max = 0.5).
    assert sigma <= 0.5 * thick + 1e-6


def test_extract_definition_records_tied_roughness(data_file):
    from bumps.fitters import fit as bumps_fit

    base = _definition(data_file, tie=True)
    problem = build_problem(base)
    bumps_fit(problem, method="amoeba", steps=200)
    defn = extract_definition(problem, base)

    siox = next(layer for layer in defn["layers"] if layer["name"] == "SiOx")
    # roughness_tie metadata preserved for the next rebuild.
    assert "roughness_tie" in siox
    # numeric roughness captured = fraction * thickness.
    assert siox["roughness"] == pytest.approx(
        defn["fitted_parameters"]["SiOx rough_frac"] * siox["thickness"], rel=1e-6
    )


def test_untied_still_works(data_file):
    problem = build_problem(_definition(data_file, tie=False))
    names = [str(p.name) for p in problem._parameters]
    assert "SiOx interface" in names
    assert "SiOx rough_frac" not in names
