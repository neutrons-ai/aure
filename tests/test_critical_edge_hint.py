"""The critical edge is a hint about what to test, never a value for the model.

It says some medium the beam reaches has a higher SLD than the incident one.
It does not say which medium, and its position is approximate, so the analysis
node only names the incident medium (from the geometry) and every prompt that
shows the edge says not to set an SLD or a bound from it.
"""

import numpy as np

from aure.nodes import analysis
from aure.nodes.analysis import annotate_critical_edges
from aure.nodes.prompts import (
    format_hypothesis_revision_prompt,
    format_structural_hypothesis_prompt,
)
from aure.tools.feature_tools import extract_critical_edges, format_critical_edge_line


def _features(contrast=5.3, qc=0.0163, **extra):
    edge = {"Qc": qc, "estimated_SLD": contrast, "confidence": "high", **extra}
    return {"critical_edges": [edge]}


def _fresnel(q, qc):
    r = np.ones_like(q)
    above = q > qc
    k = np.sqrt(1 - (qc / q[above]) ** 2)
    r[above] = ((1 - k) / (1 + k)) ** 2
    return r


# --- the incident medium follows the geometry ------------------------------


def test_back_reflection_measures_the_contrast_against_the_substrate():
    parsed = {
        "substrate": {"name": "silicon", "sld": 2.07},
        "ambient": {"name": "THF", "sld": 0.18},
        "back_reflection": True,
    }
    feats = _features()
    annotate_critical_edges(parsed, feats)
    assert feats["critical_edges"][0]["incident_medium"] == {
        "name": "silicon",
        "sld": 2.07,
        "side": "substrate",
    }


def test_front_reflection_measures_it_against_the_ambient():
    parsed = {
        "substrate": {"name": "silicon", "sld": 2.07},
        "ambient": {"name": "air", "sld": 0.0},
        "back_reflection": False,
    }
    feats = _features(2.07, qc=0.0102)
    annotate_critical_edges(parsed, feats)
    inc = feats["critical_edges"][0]["incident_medium"]
    assert (inc["name"], inc["side"]) == ("air", "ambient")


def test_annotation_tolerates_missing_pieces():
    feats = _features()
    annotate_critical_edges(None, feats)
    annotate_critical_edges(
        {"substrate": {"name": "Si"}, "back_reflection": True}, feats
    )
    assert "incident_medium" not in feats["critical_edges"][0]
    annotate_critical_edges({"ambient": {"name": "air", "sld": 0.0}}, None)


# --- the analysis node never rewrites the model from the edge -----------------


def test_a_solvent_stated_as_protonated_is_left_as_stated():
    """The edge used to rewrite "THF" as d8-THF. A thick copper film in
    protonated THF gives the same edge, so the node now leaves the model alone
    and the isotope is a hypothesis to test."""
    q = np.linspace(0.008, 0.2, 400)
    parsed = {
        "substrate": {"name": "silicon", "sld": 2.07},
        "layers": [{"name": "Cu", "thickness": 500, "sld": 6.5}],
        "ambient": {"name": "THF", "sld": 0.18},
        "back_reflection": True,
    }
    state = {
        "Q": q.tolist(),
        "R": _fresnel(q, 0.0150).tolist(),
        "dR": None,
        "parsed_sample": parsed,
    }
    out = analysis.analysis_node(state)
    assert "parsed_sample" not in out
    assert not any("Ambient correction" in m["content"] for m in out["messages"])
    edge = out["extracted_features"]["critical_edges"][0]
    assert edge["incident_medium"]["side"] == "substrate"
    assert "implied_ambient_sld" not in edge and "suggests_deuteration" not in edge
    assert not hasattr(analysis, "_analyze_ambient")


# --- what the edge line says --------------------------------------------------


def test_edge_line_is_a_hint_and_gives_no_absolute_sld():
    feats = _features(contrast_low=4.6, contrast_high=5.9)
    annotate_critical_edges(
        {"substrate": {"name": "silicon", "sld": 2.07}, "back_reflection": True}, feats
    )
    line = format_critical_edge_line(feats["critical_edges"][0])
    assert "about 5.3" in line and "silicon, SLD 2.07" in line
    assert "the beam enters through the substrate" in line
    assert "spans a contrast of 4.6–5.9" in line
    assert "far one or a thick layer" in line
    assert "not a value to set an SLD or its bounds from" in line
    assert "7.4" not in line and "DEUTERATED" not in line and "Constrain" not in line


def test_edge_line_survives_an_old_checkpoint():
    """A resumed run's edge may carry the retired annotations and none of the
    new ones; the line must still be the neutral hint."""
    old = {
        "Qc": 0.0163,
        "estimated_SLD": 5.3,
        "confidence": "high",
        "implied_ambient_sld": 7.37,
        "suggests_deuteration": True,
        "substrate_sld": 2.07,
    }
    line = format_critical_edge_line(old)
    assert "7.4" not in line and "Constrain" not in line
    assert "A hint about what to test" in line


def test_the_drop_width_is_reported_with_the_edge():
    q = np.linspace(0.006, 0.05, 300)
    edge = extract_critical_edges(q, _fresnel(q, 0.0150))[0]
    assert edge["contrast_low"] < edge["estimated_SLD"] < edge["contrast_high"]
    # A bare Fresnel edge sits just above the true contrast of 4.48.
    assert 4.3 < edge["contrast_low"] < 4.8


# --- the hint reaches both hypothesis prompts ---------------------------------


def test_the_intake_hypothesis_prompt_gets_the_hint():
    parsed = {
        "substrate": {"name": "silicon", "sld": 2.07},
        "layers": [],
        "ambient": {"name": "THF", "sld": 0.18},
        "back_reflection": True,
    }
    feats = _features()
    annotate_critical_edges(parsed, feats)
    prompt = format_structural_hypothesis_prompt(
        "Cu on Si in THF", parsed, "", features=feats
    )
    assert "## What the Data Show (hints, not measurements)" in prompt
    assert "Critical edge: Qc ≈ 0.0163" in prompt
    assert "do not copy its contrast into an SLD value" in prompt
    bare = format_structural_hypothesis_prompt("Cu on Si in THF", parsed, "")
    assert "no critical edge found" in bare


def test_the_revision_prompt_gets_the_hint():
    prompt = format_hypothesis_revision_prompt(
        sample_description="Cu on Si in THF",
        current_model={},
        skill_context="",
        structural_hypotheses=[],
        fit_history=[],
        chi_squared=4.0,
        features=_features(),
    )
    assert "Critical edge: Qc ≈ 0.0163" in prompt
