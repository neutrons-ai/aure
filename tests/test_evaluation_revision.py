"""Tests for the gated evaluation-time hypothesis revision (Change 3).

Covers the gate (`_should_revise_hypotheses`), the orchestration
(`_revise_hypotheses`: re-select skills, propose new hypotheses, re-rank),
and the ranking-ref resolver.
"""

from aure.nodes import evaluation


# ----------------------------------------------------------------------
# Gate
# ----------------------------------------------------------------------


def test_gate_fires_on_residual_fringes():
    lf = {"chi_squared": 9.0, "residual_analysis": {"has_residual_fringes": True}}
    assert evaluation._should_revise_hypotheses(lf, [{"status": "pending"}], [])


def test_gate_fires_when_no_pending_left():
    assert evaluation._should_revise_hypotheses(
        {"chi_squared": 9.0}, [{"status": "rejected"}], []
    )
    # empty list also counts as "no pending"
    assert evaluation._should_revise_hypotheses({"chi_squared": 9.0}, [], [])


def test_gate_fires_when_chi2_stalled():
    hist = [
        {"chi_squared": 10.0},
        {"chi_squared": 9.9},
        {"chi_squared": 9.8},
    ]
    assert evaluation._should_revise_hypotheses(
        {"chi_squared": 9.8}, [{"status": "pending"}], hist
    )


def test_gate_quiet_when_improving_and_pending_exists():
    hist = [
        {"chi_squared": 100.0},
        {"chi_squared": 50.0},
        {"chi_squared": 10.0},
    ]
    assert not evaluation._should_revise_hypotheses(
        {"chi_squared": 10.0}, [{"status": "pending"}], hist
    )


# ----------------------------------------------------------------------
# Ranking-ref resolver
# ----------------------------------------------------------------------


def test_resolve_ranking_maps_new_refs():
    # existing ids 1,3 ; new ids assigned [4,5]
    assert evaluation._resolve_ranking(["new1", 3, 1, "new2"], [4, 5]) == [4, 3, 1, 5]


def test_resolve_ranking_tolerates_garbage():
    assert evaluation._resolve_ranking([True, "2", "newX", 1], [9]) == [2, 1]


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------


def test_revise_appends_reranks_and_unions_skills(monkeypatch):
    monkeypatch.setattr(
        evaluation,
        "select_skills",
        lambda *a, **k: ["sei-layer-analysis", "neutron-reflectometry"],
    )
    monkeypatch.setattr(evaluation, "load_skill_context", lambda names, registry: "ctx")
    monkeypatch.setattr(
        evaluation,
        "propose_hypothesis_revision_with_llm",
        lambda **kw: {
            "new_hypotheses": [
                {
                    "title": "Add SEI",
                    "rationale": "sei",
                    "change": "insert SEI",
                    "skill_source": "sei-layer-analysis",
                }
            ],
            "ranking": ["new1", 1],
        },
    )

    prior = [
        {
            "id": 1,
            "title": "Add CuO",
            "rationale": "r",
            "change": "c",
            "skill_source": "metal-oxide-interfaces",
            "origin": "skill",
            "status": "pending",
            "tried_in_iteration": None,
            "created_in_iteration": None,
            "notes": "",
        }
    ]
    latest_fit = {
        "chi_squared": 9.0,
        "residual_analysis": {
            "has_residual_fringes": True,
            "unmodeled_thicknesses": [{"thickness": 40, "confidence": "high"}],
        },
    }

    rev = evaluation._revise_hypotheses(
        state={
            "sample_description": "Li cell",
            "parsed_sample": {},
            "current_model": {},
        },
        latest_fit=latest_fit,
        hypotheses=prior,
        iteration=3,
        bic=10.0,
        boundary_hits=[],
        analysis={"issues": [], "physical_concerns": []},
        fit_history=[latest_fit],
        registry=None,
        active_skills=["metal-oxide-interfaces", "neutron-reflectometry"],
    )

    assert rev["n_new"] == 1
    hyps = rev["hypotheses"]
    assert len(hyps) == 2
    new = next(h for h in hyps if h["id"] == 2)
    assert new["origin"] == "evaluation"
    assert new["title"] == "Add SEI"
    assert new["created_in_iteration"] == 3
    # ranking ["new1", 1] → new id 2 first, then id 1
    assert [h["id"] for h in hyps] == [2, 1]
    # skills unioned, intake skill never dropped
    assert set(rev["active_skills"]) >= {
        "metal-oxide-interfaces",
        "neutron-reflectometry",
        "sei-layer-analysis",
    }
    assert "sei-layer-analysis" in rev["added_skills"]
    assert rev["changed"]


def test_revise_no_new_no_drop_is_stable(monkeypatch):
    """No new hypotheses + same skills → membership and skills unchanged."""
    monkeypatch.setattr(
        evaluation, "select_skills", lambda *a, **k: ["neutron-reflectometry"]
    )
    monkeypatch.setattr(evaluation, "load_skill_context", lambda names, registry: "ctx")
    monkeypatch.setattr(
        evaluation,
        "propose_hypothesis_revision_with_llm",
        lambda **kw: {"new_hypotheses": [], "ranking": [1]},
    )
    prior = [
        {
            "id": 1,
            "title": "Add CuO",
            "rationale": "r",
            "change": "c",
            "skill_source": "metal-oxide-interfaces",
            "origin": "skill",
            "status": "pending",
            "tried_in_iteration": None,
            "created_in_iteration": None,
            "notes": "",
        }
    ]
    rev = evaluation._revise_hypotheses(
        state={"sample_description": "x", "parsed_sample": {}, "current_model": {}},
        latest_fit={"chi_squared": 9.0},
        hypotheses=prior,
        iteration=2,
        bic=None,
        boundary_hits=[],
        analysis={"issues": [], "physical_concerns": []},
        fit_history=[],
        registry=None,
        active_skills=["neutron-reflectometry"],
    )
    assert rev["n_new"] == 0
    assert [h["id"] for h in rev["hypotheses"]] == [1]
    assert rev["active_skills"] == ["neutron-reflectometry"]
