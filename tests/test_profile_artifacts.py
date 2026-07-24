"""Tests for SLD-profile artifact detection (feature_tools)."""

import numpy as np
from scipy.special import erf

from aure.tools.feature_tools import (
    detect_profile_artifacts,
    check_roughness_thickness_ratios,
    _turning_point_slds,
)


def _profile_from_slabs(interfaces):
    """Build a smooth SLD profile from a list of (z_boundary, rho_below, sigma).

    ``interfaces`` gives, from ambient (low z) to substrate (high z), the
    boundary position, the SLD *after* that boundary, and the interface
    roughness. The profile is the standard sum-of-error-functions model.
    Returns (z, rho).
    """
    z = np.linspace(-100, 1100, 4000)
    rho0 = interfaces[0][1]  # ambient SLD sits before the first boundary
    rho = np.full_like(z, rho0)
    prev = rho0
    for zb, rho_after, sigma in interfaces[1:]:
        s = max(sigma, 1e-3)
        rho = rho + 0.5 * (rho_after - prev) * (1 + erf((z - zb) / (s * np.sqrt(2))))
        prev = rho_after
    return z, rho


# Ordered slab SLDs (ambient -> substrate) for the Cu_K 207282 stack.
LAYER_RHOS = [6.13, 4.58, -0.03, 6.42, -1.61, 3.2, 2.07]


def test_turning_points():
    # interior extrema of the sequence: Plated min, Cu max, Ti min, SiOx max
    turns = sorted(_turning_point_slds(LAYER_RHOS))
    assert turns == sorted([-0.03, 6.42, -1.61, 3.2])
    # direction-agnostic
    assert sorted(_turning_point_slds(LAYER_RHOS[::-1])) == turns


def test_clean_profile_has_no_artifact():
    # A physical stack: each layer plateau in order, modest roughness.
    z, rho = _profile_from_slabs(
        [
            (None, 6.13, None),
            (100, 4.58, 15),
            (260, -0.03, 12),
            (300, 6.42, 10),
            (869, -1.61, 10),
            (925, 3.2, 8),
            (946, 2.07, 6),
        ]
    )
    res = detect_profile_artifacts(z, rho, LAYER_RHOS)
    assert res["has_artifact"] is False
    assert res["excursions"] == []


def test_erf_tail_dip_below_substrate_is_flagged():
    # Thin SiOx (20 Å) with a LARGE Ti/SiOx roughness: the Ti well's tail
    # reaches across SiOx and the profile dips below Si (2.07) before settling.
    z, rho = _profile_from_slabs(
        [
            (None, 6.13, None),
            (100, 4.58, 15),
            (260, -0.03, 12),
            (300, 6.42, 10),
            (869, -1.61, 10),
            (925, 3.2, 30),   # sigma >> half of the 20 Å SiOx layer
            (945, 2.07, 6),
        ]
    )
    # Confirm the synthetic profile actually undershoots below the substrate.
    tail = rho[z > 930]
    assert tail.min() < 2.07 - 0.05
    res = detect_profile_artifacts(z, rho, LAYER_RHOS)
    assert res["has_artifact"] is True
    assert any(e["kind"] == "min" and e["sld"] < 2.07 for e in res["excursions"])


def test_diffuse_but_physical_profile_not_flagged():
    # A deliberately diffuse single interface (sigma > half the "layer"): a
    # legitimate profile parametrization. Monotonic, no excursion -> no flag.
    seq = [6.35, 4.0, 2.07]  # ambient -> graded midpoint -> substrate
    z, rho = _profile_from_slabs(
        [
            (None, 6.35, None),
            (400, 4.0, 60),
            (460, 2.07, 60),
        ]
    )
    res = detect_profile_artifacts(z, rho, seq)
    assert res["has_artifact"] is False


def test_evaluation_veto_overrides_acceptable():
    """A non-physical excursion must flip acceptable=True -> False so the
    workflow refines instead of completing on χ² alone."""
    from aure.nodes.evaluation import _detect_profile_artifacts_into

    z, rho = _profile_from_slabs(
        [
            (None, 6.13, None),
            (100, 4.58, 15),
            (260, -0.03, 12),
            (300, 6.42, 10),
            (869, -1.61, 10),
            (925, 3.2, 30),
            (945, 2.07, 6),
        ]
    )
    model = {
        "ambient": {"name": "THF", "sld": 6.13},
        "substrate": {"name": "Si", "sld": 2.07},
        "layers": [
            {"name": "SEI", "sld": 4.58, "thickness": 160, "roughness": 15},
            {"name": "Plated", "sld": -0.03, "thickness": 56, "roughness": 8},
            {"name": "Cu", "sld": 6.42, "thickness": 569, "roughness": 10},
            {"name": "Ti", "sld": -1.61, "thickness": 56, "roughness": 10},
            {"name": "SiOx", "sld": 3.2, "thickness": 20, "roughness": 30},
        ],
    }
    fit = {"sld_z": z.tolist(), "sld_rho": rho.tolist(), "parameters": {}}
    analysis = {"acceptable": True, "issues": [], "suggestions": [], "physical_concerns": []}
    _detect_profile_artifacts_into(analysis, fit, model)
    assert analysis["acceptable"] is False
    assert any("excursion" in i.lower() for i in analysis["issues"])
    assert analysis["suggestions"], "a remedy suggestion should be added"


def test_evaluation_clean_profile_leaves_acceptable_untouched():
    from aure.nodes.evaluation import _detect_profile_artifacts_into

    z, rho = _profile_from_slabs(
        [
            (None, 6.13, None),
            (100, 4.58, 15),
            (260, -0.03, 12),
            (300, 6.42, 10),
            (869, -1.61, 10),
            (925, 3.2, 8),
            (946, 2.07, 6),
        ]
    )
    model = {
        "ambient": {"name": "THF", "sld": 6.13},
        "substrate": {"name": "Si", "sld": 2.07},
        "layers": [
            {"name": "SEI", "sld": 4.58, "thickness": 160, "roughness": 15},
            {"name": "Plated", "sld": -0.03, "thickness": 56, "roughness": 12},
            {"name": "Cu", "sld": 6.42, "thickness": 569, "roughness": 10},
            {"name": "Ti", "sld": -1.61, "thickness": 56, "roughness": 10},
            {"name": "SiOx", "sld": 3.2, "thickness": 20, "roughness": 8},
        ],
    }
    fit = {"sld_z": z.tolist(), "sld_rho": rho.tolist(), "parameters": {}}
    analysis = {"acceptable": True, "issues": [], "suggestions": [], "physical_concerns": []}
    _detect_profile_artifacts_into(analysis, fit, model)
    assert analysis["acceptable"] is True


def test_roughness_ratio_is_informational_only():
    model = {
        "layers": [
            {"name": "SiOx", "thickness": 20.0, "roughness": 17.0},
            {"name": "Cu", "thickness": 570.0, "roughness": 10.0},
        ]
    }
    notes = check_roughness_thickness_ratios(model)
    assert len(notes) == 1
    assert notes[0]["layer"] == "SiOx"
    assert notes[0]["ratio"] > 0.5
