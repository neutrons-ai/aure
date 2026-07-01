"""Critical-edge Qc is estimated at the plateau half-height.

The previous "onset of the first drop" heuristic under-reported Qc — it fired on
plateau noise well before the real edge (e.g. 0.0111 for a true edge at ~0.0148).
"""

import numpy as np

from aure.tools.feature_tools import extract_critical_edges


def _plateau_then_drop(qc_true, plateau=0.9, width=0.0006, n=248, seed=0):
    """Synthetic total-reflection plateau that halves at ``qc_true``.

    R(Q) = plateau / (1 + exp((Q - qc)/w)); by construction R = plateau/2 at
    Q = qc_true, so the half-height estimator should recover qc_true.
    """
    Q = np.linspace(0.0104, 0.0358, n)
    R = plateau / (1.0 + np.exp((Q - qc_true) / width))
    return Q, R


def test_qc_at_half_plateau_matches_true_edge():
    qc_true = 0.0148
    Q, R = _plateau_then_drop(qc_true)
    edges = extract_critical_edges(Q, R)
    assert edges, "expected a critical edge"
    qc = edges[0]["Qc"]
    assert abs(qc - qc_true) / qc_true < 0.08, f"Qc={qc:.5f} vs {qc_true}"
    # And definitely not the old under-reported onset (~0.011).
    assert qc > 0.013


def test_qc_recovers_low_intensity_plateau():
    # Plateau well below 1.0 (imperfect normalization) must not shift Qc.
    qc_true = 0.0150
    Q, R = _plateau_then_drop(qc_true, plateau=0.6)
    edges = extract_critical_edges(Q, R)
    assert edges
    assert abs(edges[0]["Qc"] - qc_true) / qc_true < 0.08


def test_flat_data_gives_no_edge():
    Q = np.linspace(0.01, 0.05, 200)
    R = np.full_like(Q, 0.9)
    assert extract_critical_edges(Q, R) == []
