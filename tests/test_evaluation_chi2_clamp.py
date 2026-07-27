"""Tests for the deterministic χ² acceptance clamp in the evaluation node.

``CHI2_MAX`` used to be advisory (injected into the evaluation prompt, then the
LLM's boolean was trusted), so a fit that already met the threshold could still
be sent back around the refinement loop. The clamp makes the threshold binding:
a finite χ² at or below it completes the run regardless of the LLM's verdict.
The single documented exception is the SLD-profile artifact veto — a physically
impossible profile is invisible to χ² and must never be accepted on χ² alone.
"""

import numpy as np
import pytest
from scipy.special import erf

from aure.nodes import evaluation
from aure.nodes.evaluation import _clamp_acceptance_to_chi2


# ======================================================================
# Fixtures
# ======================================================================


def _analysis(acceptable=False, issues=None, **extra):
    a = {
        "acceptable": acceptable,
        "quality_assessment": "poor",
        "issues": list(issues or []),
        "suggestions": [],
        "physical_concerns": [],
        "next_action": "parameter_tweak",
        "_used_fallback": False,
    }
    a.update(extra)
    return a


def _model():
    return {
        "substrate": {"name": "silicon", "sld": 2.07, "roughness_max": 10.0},
        "layers": [
            {"name": "Cu", "thickness": 500.0, "sld": 6.42, "roughness": 8.0},
        ],
        "ambient": {"name": "air", "sld": 0.0},
        "intensity": {"value": 1.0, "fixed": True},
    }


def _state(chi2, **extra):
    state = {
        "iteration": 0,
        "fit_results": [
            {
                "chi_squared": chi2,
                "method": "amoeba",
                "converged": True,
                "parameters": {"Cu thickness": 500.0},
                "uncertainties": None,
                "bounds": None,
                "Q_fit": [],
                "R_fit": [],
                "residual_ratio": [],
                "per_file_results": [],
            }
        ],
        "current_model": _model(),
        "Q": list(np.linspace(0.01, 0.2, 50)),
        "messages": [],
        "active_skills": [],
        "user_config": {},
        "structural_hypotheses": [],
    }
    state.update(extra)
    return state


@pytest.fixture
def stub_llm(monkeypatch):
    """Make the evaluation node run offline with a scriptable LLM verdict.

    Returns a setter taking the analysis dict the fake LLM should hand back.
    The hypothesis-revision path is stubbed too so the not-acceptable branch
    never reaches a real provider.
    """
    monkeypatch.setattr(evaluation, "llm_available", lambda: True)
    monkeypatch.setattr(
        evaluation, "select_skills", lambda *a, **k: ["neutron-reflectometry"]
    )
    monkeypatch.setattr(evaluation, "load_skill_context", lambda names, registry: "ctx")
    monkeypatch.setattr(
        evaluation,
        "propose_hypothesis_revision_with_llm",
        lambda **kw: {"new_hypotheses": [], "ranking": []},
    )

    def _set(analysis):
        monkeypatch.setattr(
            evaluation,
            "analyze_fit_quality_with_llm",
            lambda **kwargs: dict(analysis),
        )

    return _set


def _profile_from_slabs(interfaces):
    """Sum-of-error-functions profile from (z_boundary, rho_below, sigma)."""
    z = np.linspace(-100, 1100, 4000)
    rho0 = interfaces[0][1]
    rho = np.full_like(z, rho0)
    prev = rho0
    for zb, rho_after, sigma in interfaces[1:]:
        s = max(sigma, 1e-3)
        rho = rho + 0.5 * (rho_after - prev) * (1 + erf((z - zb) / (s * np.sqrt(2))))
        prev = rho_after
    return z, rho


# The Cu_K 207282 stack with a 20 Å SiOx and an oversized Ti/SiOx roughness:
# the Ti well's erf tail reaches across SiOx and dips the profile below Si.
ARTIFACT_INTERFACES = [
    (None, 6.13, None),
    (100, 4.58, 15),
    (260, -0.03, 12),
    (300, 6.42, 10),
    (869, -1.61, 10),
    (925, 3.2, 30),
    (945, 2.07, 6),
]

ARTIFACT_MODEL = {
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


# ======================================================================
# Helper semantics
# ======================================================================


def test_clamp_flips_verdict_under_threshold():
    analysis = _analysis(acceptable=False, issues=["roughness looks large"])
    assert _clamp_acceptance_to_chi2(analysis, chi2=1.8, chi2_max=2.5) is True
    assert analysis["acceptable"] is True
    assert analysis["_chi2_clamped"] is True
    # The LLM's objections survive untouched.
    assert analysis["issues"] == ["roughness looks large"]


def test_clamp_accepts_exactly_at_threshold():
    analysis = _analysis(acceptable=False)
    assert _clamp_acceptance_to_chi2(analysis, chi2=2.5, chi2_max=2.5) is True


def test_clamp_noop_when_already_acceptable():
    analysis = _analysis(acceptable=True)
    assert _clamp_acceptance_to_chi2(analysis, chi2=1.0, chi2_max=2.5) is False
    assert analysis["acceptable"] is True
    assert "_chi2_clamped" not in analysis


def test_clamp_noop_above_threshold():
    analysis = _analysis(acceptable=False)
    assert _clamp_acceptance_to_chi2(analysis, chi2=2.51, chi2_max=2.5) is False
    assert analysis["acceptable"] is False
    assert "_chi2_clamped" not in analysis


@pytest.mark.parametrize(
    "chi2", [float("inf"), float("nan"), None, True, "1.0", np.nan]
)
def test_clamp_noop_on_nonfinite_chi2(chi2):
    analysis = _analysis(acceptable=False)
    assert _clamp_acceptance_to_chi2(analysis, chi2=chi2, chi2_max=2.5) is False
    assert analysis["acceptable"] is False


def test_clamp_yields_to_profile_artifact_veto():
    analysis = _analysis(acceptable=False, _profile_artifact=True)
    assert _clamp_acceptance_to_chi2(analysis, chi2=1.0, chi2_max=2.5) is False
    assert analysis["acceptable"] is False
    assert "_chi2_clamped" not in analysis


def test_artifact_detector_sets_the_veto_marker():
    """The marker the clamp reads is set by the detector, not by hand."""
    z, rho = _profile_from_slabs(ARTIFACT_INTERFACES)
    fit = {"sld_z": z.tolist(), "sld_rho": rho.tolist(), "parameters": {}}
    analysis = _analysis(acceptable=True)
    evaluation._detect_profile_artifacts_into(analysis, fit, ARTIFACT_MODEL)
    assert analysis["acceptable"] is False
    assert analysis["_profile_artifact"] is True


def test_clean_profile_leaves_no_veto_marker():
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
    fit = {"sld_z": z.tolist(), "sld_rho": rho.tolist(), "parameters": {}}
    analysis = _analysis(acceptable=False)
    evaluation._detect_profile_artifacts_into(analysis, fit, ARTIFACT_MODEL)
    assert "_profile_artifact" not in analysis


# ======================================================================
# Integration through evaluation_node
# ======================================================================


def test_node_completes_when_llm_rejects_a_passing_chi2(monkeypatch, stub_llm):
    monkeypatch.setenv("CHI2_MAX", "2.5")
    stub_llm(
        _analysis(
            acceptable=False,
            issues=["Cu roughness is larger than expected"],
            suggestions=["try a graded interface"],
        )
    )

    out = evaluation.evaluation_node(_state(1.7))

    assert out["workflow_complete"] is True
    content = out["messages"][0]["content"]
    assert "Fit Successful" in content
    assert "acceptance threshold" in content
    # The LLM's objection is reported, not dropped.
    assert "### Notes:" in content
    assert "Cu roughness is larger than expected" in content


def test_node_refines_when_chi2_exceeds_threshold(monkeypatch, stub_llm):
    monkeypatch.setenv("CHI2_MAX", "2.5")
    stub_llm(_analysis(acceptable=False, issues=["poor high-Q agreement"]))

    out = evaluation.evaluation_node(_state(9.0))

    assert "workflow_complete" not in out
    assert any("Fit Evaluation" in m["content"] for m in out["messages"])


def test_node_respects_an_llm_accept_above_threshold(monkeypatch, stub_llm):
    """No reverse clamp: a higher χ² the LLM accepts still completes."""
    monkeypatch.setenv("CHI2_MAX", "2.5")
    stub_llm(_analysis(acceptable=True))

    out = evaluation.evaluation_node(_state(7.0))

    assert out["workflow_complete"] is True
    # Not clamped, so no "stopped at the threshold" wording.
    assert "acceptance threshold" not in out["messages"][0]["content"]


def test_node_artifact_veto_beats_the_clamp(monkeypatch, stub_llm):
    """A passing χ² with a non-physical SLD profile must NOT complete."""
    monkeypatch.setenv("CHI2_MAX", "2.5")
    stub_llm(_analysis(acceptable=False))

    z, rho = _profile_from_slabs(ARTIFACT_INTERFACES)
    state = _state(1.2, current_model=ARTIFACT_MODEL)
    state["fit_results"][-1]["sld_z"] = z.tolist()
    state["fit_results"][-1]["sld_rho"] = rho.tolist()
    state["fit_results"][-1]["parameters"] = {}

    out = evaluation.evaluation_node(state)

    assert "workflow_complete" not in out
    assert any("excursion" in i.lower() for i in state["fit_results"][-1]["issues"])
