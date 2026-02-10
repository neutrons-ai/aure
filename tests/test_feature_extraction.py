"""
Test feature extraction on synthetic reflectivity data.

Uses a pure-numpy Parratt recursion to generate ground truth curves
with known parameters, then tests if features can be recovered.
"""

import numpy as np
from aure.tools.feature_tools import extract_all_features, format_features_for_llm
from aure.database.materials import lookup_material


# ---------------------------------------------------------------------------
# Synthetic reflectivity (Parratt recursion, numpy only)
# ---------------------------------------------------------------------------

def _parratt(Q, sld, thickness, roughness):
    """
    Parratt recursion for specular reflectivity.

    Args:
        Q: 1-D array of Q values (Å⁻¹).
        sld: SLD of each slab (10⁻⁶ Å⁻²), ordered ambient → substrate.
        thickness: thickness of each slab (Å); ambient & substrate are 0.
        roughness: interfacial roughness (Å), length = len(sld) - 1.
                   roughness[i] is at the interface between slab i and i+1.

    Returns:
        Reflectivity array, same length as Q.
    """
    Q = np.asarray(Q, dtype=float)
    sld = np.asarray(sld, dtype=float)
    thickness = np.asarray(thickness, dtype=float)
    roughness = np.asarray(roughness, dtype=float)
    n = len(sld)

    R = np.empty(len(Q))
    for iq, q in enumerate(Q):
        if q <= 0:
            R[iq] = 1.0
            continue

        k0 = q / 2.0
        kz = np.sqrt(k0**2 - 4 * np.pi * sld * 1e-6 + 0j)

        # Build from the substrate upward
        r = 0.0 + 0j
        for j in range(n - 2, -1, -1):
            rj = (kz[j] - kz[j + 1]) / (kz[j] + kz[j + 1])
            rj *= np.exp(-2 * kz[j] * kz[j + 1] * roughness[j] ** 2)
            pj = np.exp(2j * kz[j + 1] * thickness[j + 1])
            r = (rj + r * pj) / (1 + rj * r * pj)

        R[iq] = np.abs(r) ** 2

    return R


def generate_synthetic_curve(n_layers, Q, params):
    """
    Generate synthetic reflectivity for 0-, 1-, or 2-layer systems.

    Parameters follow the same convention as the old torch-based helper
    so every existing test case works unchanged.
    """
    sub_sld = params["substrate_sld"]
    sub_rough = params["substrate_roughness"]

    if n_layers == 0:
        sld = [0.0, sub_sld]
        thick = [0.0, 0.0]
        rough = [sub_rough]
    elif n_layers == 1:
        sld = [0.0, params["sld1"], sub_sld]
        thick = [0.0, params["d1"], 0.0]
        rough = [params["rough1"], sub_rough]
    elif n_layers == 2:
        # Top layer first (ambient → layer2 → layer1 → substrate)
        sld = [0.0, params["sld2"], params["sld1"], sub_sld]
        thick = [0.0, params["d2"], params["d1"], 0.0]
        rough = [params["rough2"], params["rough1"], sub_rough]
    else:
        raise ValueError(f"n_layers must be 0, 1, or 2, got {n_layers}")

    return _parratt(Q, sld, thick, rough)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_0_layer():
    """Test feature extraction on Fresnel reflectivity (0 layers)."""
    params = {
        "substrate_sld": 2.07,      # Silicon
        "substrate_roughness": 3.0,
    }

    Q = np.linspace(0.005, 0.25, 200)
    R = generate_synthetic_curve(0, Q, params)

    features = extract_all_features(Q, R)

    # Expected Qc for silicon ≈ 0.0102 Å⁻¹
    expected_qc = 4 * np.sqrt(np.pi * params["substrate_sld"] * 1e-6)

    assert features["estimated_n_layers"] == 0, (
        f"Expected 0 layers, got {features['estimated_n_layers']}"
    )

    if features["critical_edges"]:
        extracted_qc = features["critical_edges"][0]["Qc"]
        qc_error = abs(extracted_qc - expected_qc) / expected_qc * 100
        assert qc_error < 20, f"Qc error too large: {qc_error:.1f}%"


def test_1_layer():
    """Test feature extraction on a 1-layer system (100 nm polymer on Si)."""
    params = {
        "d1": 1000.0,               # 1000 Å = 100 nm
        "sld1": 1.5,                # polymer
        "rough1": 5.0,
        "substrate_sld": 2.07,
        "substrate_roughness": 3.0,
    }

    Q = np.linspace(0.005, 0.25, 300)
    R = generate_synthetic_curve(1, Q, params)

    features = extract_all_features(Q, R)

    # Thickness should be recoverable from Kiessig fringes
    if features["estimated_total_thickness"]:
        thickness_error = (
            abs(features["estimated_total_thickness"] - params["d1"])
            / params["d1"]
            * 100
        )
        assert thickness_error < 20, f"Thickness error too large: {thickness_error:.1f}%"


def test_2_layer():
    """Test feature extraction on a 2-layer system."""
    params = {
        "d1": 120.0,
        "sld1": 4.0,
        "rough1": 5.0,
        "d2": 80.0,
        "sld2": 1.5,
        "rough2": 5.0,
        "substrate_sld": 2.07,
        "substrate_roughness": 3.0,
    }

    Q = np.linspace(0.005, 0.25, 300)
    R = generate_synthetic_curve(2, Q, params)

    features = extract_all_features(Q, R)
    true_total = params["d1"] + params["d2"]

    # Total thickness estimate should be in the right ballpark
    if features["estimated_total_thickness"]:
        thickness_error = (
            abs(features["estimated_total_thickness"] - true_total)
            / true_total
            * 100
        )
        # 2-layer beating makes thickness harder; use a generous tolerance
        assert thickness_error < 50, f"Thickness error too large: {thickness_error:.1f}%"


def test_material_lookup():
    """Test material database lookup."""
    test_queries = [
        ("Si", 2.07),
        ("silicon", 2.07),
        ("D2O", 6.37),
        ("heavy water", 6.37),
        ("gold", 4.66),
        ("native oxide", 3.47),
    ]

    for query, expected in test_queries:
        material = lookup_material(query)
        assert material is not None, f"lookup_material({query!r}) returned None"
        sld = material.get_sld()
        assert abs(sld - expected) < 0.1, (
            f"SLD mismatch for {query}: got {sld:.3f}, expected {expected}"
        )
