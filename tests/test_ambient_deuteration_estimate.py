"""Deterministic ambient-SLD estimate from the critical edge (deuteration hint).

Given the substrate SLD, the critical edge pins the ambient SLD; when that
implied value is far above the stated H-form the ambient is very likely
deuterated even if the user never said so.
"""

from aure.nodes.analysis import (
    _analyze_ambient,
    implied_ambient_sld_from_edge,
)
from aure.tools.feature_tools import format_critical_edge_line


# --- implied_ambient_sld_from_edge -----------------------------------------


def test_implied_high_branch_flags_deuteration():
    # Si (2.07), contrast ~4.36 (Qc ~0.0148), stated H2O (-0.56).
    est = implied_ambient_sld_from_edge(2.07, 4.36, -0.56)
    assert est["suggests_deuteration"] is True
    assert abs(est["implied_ambient_sld"] - 6.43) < 0.2


def test_h_form_edge_does_not_flag():
    # Genuine H2O front-reflection: contrast = Si - H2O = 2.63; low branch
    # matches the stated H-form, so no deuteration is implied.
    est = implied_ambient_sld_from_edge(2.07, 2.63, -0.56)
    assert est["suggests_deuteration"] is False


def test_already_deuterated_does_not_flag():
    # Ambient already stated as D2O (6.36): the high branch matches it.
    est = implied_ambient_sld_from_edge(2.07, 4.29, 6.36)
    assert est["suggests_deuteration"] is False


# --- _analyze_ambient ------------------------------------------------------


def _features(contrast, qc=0.0148, confidence="high"):
    return {
        "critical_edges": [
            {"Qc": qc, "estimated_SLD": contrast, "confidence": confidence}
        ]
    }


def test_generic_electrolyte_emits_hint_and_annotates_edge():
    parsed = {
        "substrate": {"name": "Si", "sld": 2.07},
        "ambient": {"name": "0.1 M NaHCO3 electrolyte", "sld": -0.56},
    }
    feats = _features(4.36)
    correction, msg = _analyze_ambient(parsed, feats)
    # Generic liquid → hint only, no auto-correction.
    assert correction is None
    assert "Deuteration hint" in msg
    # Edge annotated in place so the hint reaches downstream prompts.
    edge = feats["critical_edges"][0]
    assert edge["suggests_deuteration"] is True
    assert abs(edge["implied_ambient_sld"] - 6.43) < 0.2
    assert edge["substrate_sld"] == 2.07


def test_known_solvent_is_auto_corrected_and_constrained():
    parsed = {
        "substrate": {"name": "Si", "sld": 2.07},
        "ambient": {"name": "THF", "sld": 0.18},
    }
    # Contrast placing the high branch near d8-THF (6.35): 6.35 - 2.07 = 4.28.
    correction, msg = _analyze_ambient(parsed, _features(4.28))
    assert correction is not None
    assert correction["name"] == "d8-THF"
    assert abs(correction["sld"] - 6.35) < 0.01
    # SLD constrained near the deuterated value, not left across the H–D range.
    assert correction["sld_min"] > 4.0 and correction["sld_max"] < 8.0
    assert "d8-THF" in msg


def test_air_ambient_is_ignored():
    parsed = {
        "substrate": {"name": "Si", "sld": 2.07},
        "ambient": {"name": "air", "sld": 0.0},
    }
    assert _analyze_ambient(parsed, _features(4.36)) == (None, "")


def test_h_form_front_reflection_no_false_positive():
    parsed = {
        "substrate": {"name": "Si", "sld": 2.07},
        "ambient": {"name": "water", "sld": -0.56},
    }
    correction, msg = _analyze_ambient(parsed, _features(2.63))
    assert correction is None and msg == ""


def test_low_confidence_edge_ignored():
    parsed = {
        "substrate": {"name": "Si", "sld": 2.07},
        "ambient": {"name": "electrolyte", "sld": -0.56},
    }
    correction, msg = _analyze_ambient(parsed, _features(4.36, confidence="low"))
    assert correction is None and msg == ""


# --- format_critical_edge_line ---------------------------------------------


def test_edge_line_shows_deuteration_hint_when_annotated():
    edge = {
        "Qc": 0.0148,
        "estimated_SLD": 4.36,
        "confidence": "high",
        "implied_ambient_sld": 6.43,
        "substrate_sld": 2.07,
        "suggests_deuteration": True,
    }
    line = format_critical_edge_line(edge)
    assert "DEUTERATED" in line
    assert "6.4" in line


def test_edge_line_plain_without_annotation():
    line = format_critical_edge_line(
        {"Qc": 0.0102, "estimated_SLD": 2.07, "confidence": "high"}
    )
    assert "DEUTERATED" not in line
    assert "0.0102" in line
