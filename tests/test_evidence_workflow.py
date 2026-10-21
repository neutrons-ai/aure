"""The posterior evidence inside the workflow (docs/evidence.md §5, §7).

* A real DREAM fit carries a JSON-safe ``posterior`` and leaves the problem as
  it found it.
* The guardrail decides on the evidence only when both fits are trusted, keeps
  BIC otherwise, and records a BIC revert the provisional evidence contests as
  inconclusive rather than as a rejection.
* ``fitting`` and ``evaluation`` apply complementary halves of one rule.
* ``finalize`` ranks by evidence only when every candidate has a trusted one.
* ``final_fit`` and the report surfaces disclose the posterior check.
"""

import copy
import json
import math
import os
import tempfile

import numpy as np
import pytest

from aure.nodes import evaluation, fitting
from aure.nodes import final_fit as ff
from aure.nodes.evaluation import BIC_FORMULA
from aure.nodes.finalize import finalize_node
from aure.nodes.prompts import _format_complexity_assessment
from aure.tools.posterior_tools import (
    POSTERIOR_FORMULA,
    compare_evidence,
    evidence_record,
    summarize_dream_fit,
)

N_DATA = 160
HALF_LN_N = 0.5 * math.log(N_DATA)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _post(lnz, *, err=0.2, trusted=True, k=8, chi2_total=None, method="bridge"):
    """A posterior summary carrying just what the workflow reads."""
    chi2_total = 140.0 if chi2_total is None else chi2_total
    lnz_bic = -0.5 * chi2_total - k * HALF_LN_N
    return {
        "formula": POSTERIOR_FORMULA,
        "method": "dream",
        "k": k,
        "n_data": N_DATA,
        "evidence_trusted": trusted,
        "fit_trusted": trusted,
        "log_evidence": lnz,
        "log_evidence_error": err,
        "log_evidence_method": method,
        "log_evidence_bic": lnz_bic,
        "bic_error": lnz_bic - lnz,
        "draw": {
            "rhat_max": 1.05 if trusted else 3.4,
            "n_unique": 900,
            "n_draws": 10000,
        },
        "checks": {"converged": trusted, "best_central": True},
        "thresholds": {"rhat_max": 1.2},
        "p_d": k - 0.5,
        "concerns": []
        if trusted
        else ["The chains did not mix: split-R̂ reaches 3.40."],
    }


def _model(n_layers=1):
    layers = [
        {"name": f"L{i}", "thickness": 50.0, "sld": 2.0, "roughness": 3.0}
        for i in range(n_layers)
    ]
    return {
        "substrate": {"name": "Si", "sld": 2.07, "roughness": 3.0},
        "layers": layers,
        "ambient": {"name": "air", "sld": 0.0},
        "intensity": {"value": 1.0, "fixed": True},
    }


def _fit(iteration, chi2_total, k, posterior=None, bounds=None):
    fit = {
        "iteration": iteration,
        "method": "dream",
        "chi_squared": chi2_total / (N_DATA - k),
        "converged": True,
        "parameters": {f"p{j}": 1.0 for j in range(k)},
        "bounds": bounds or {f"p{j}": [0.0, 10.0] for j in range(k)},
        "_chi2_total": chi2_total,
        "_n_data": N_DATA,
        "_n_free_params": k,
        "bic": chi2_total + k * math.log(N_DATA),
    }
    if posterior is not None:
        fit["posterior"] = posterior
    return fit


def _hypothesis(prev_iter):
    return {
        "id": 1,
        "title": "Add an interlayer",
        "status": "tried",
        "tried_in_iteration": prev_iter,
    }


def _state(base_fit, latest_fit, *, base_model, latest_model, evidence_baseline=True):
    """A state just after `fitting` scored *latest_fit* against *base_fit*."""
    state = {
        "iteration": 1,
        "fit_results": [base_fit, latest_fit],
        "current_model": latest_model,
        "Q": list(np.linspace(0.01, 0.2, N_DATA)),
        "messages": [],
        "best_chi2": latest_fit["chi_squared"],
        "best_model": latest_model,
        "best_bic": base_fit["bic"],
        "best_bic_model": base_model,
        "best_bic_chi2": base_fit["chi_squared"],
        "bic_formula": BIC_FORMULA,
        "structural_hypotheses": [_hypothesis(prev_iter=1)],
    }
    if evidence_baseline and base_fit.get("posterior", {}).get("evidence_trusted"):
        state["best_evidence"] = evidence_record(base_fit, 0)
        state["best_evidence_model"] = base_model
    if base_fit.get("posterior", {}).get("log_evidence_method") == "bridge":
        state["best_bic_evidence"] = evidence_record(base_fit, 0)
    return state


@pytest.fixture
def offline(monkeypatch):
    """Run the evaluation node with an LLM that always says "not acceptable"."""
    monkeypatch.setattr(evaluation, "llm_available", lambda: True)
    monkeypatch.setattr(evaluation, "load_skill_context", lambda names, registry: "")
    monkeypatch.setattr(evaluation, "select_skills", lambda *a, **k: [])
    monkeypatch.setattr(
        evaluation,
        "propose_hypothesis_revision_with_llm",
        lambda **kw: {"new_hypotheses": [], "ranking": []},
    )
    seen = {}

    def _verdict(**kwargs):
        seen.update(kwargs)
        return {
            "acceptable": False,
            "quality_assessment": "poor",
            "issues": [],
            "suggestions": ["try something"],
            "physical_concerns": [],
            "next_action": "parameter_tweak",
            "_used_fallback": False,
        }

    monkeypatch.setattr(evaluation, "analyze_fit_quality_with_llm", _verdict)
    monkeypatch.setenv("CHI2_MAX", "0.5")  # keep every fit above the accept window
    monkeypatch.setenv("CHI2_MIN", "0")
    return seen


def _with_comparison(state, latest_fit):
    """What `fitting` records when both fits are trusted."""
    base = state.get("best_evidence")
    if base is not None:
        latest_fit["model_comparison"] = compare_evidence(
            evidence_record(latest_fit, 1), base, margin=1.0
        )
    return state


# ----------------------------------------------------------------------
# A real DREAM fit
# ----------------------------------------------------------------------


@pytest.fixture
def film_data():
    """Synthetic film on Si: 220 points, 3 % noise."""
    from aure.nodes.model_builder import build_experiment

    q = np.linspace(0.01, 0.2, 120)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False)
    f.write("# Q R dR dQ\n")
    for qq in q:
        f.write(f"{qq:.6f}  1.0  0.02  {0.02 * qq:.6e}\n")
    f.close()
    truth = {
        "substrate": {
            "name": "Si",
            "sld": 2.07,
            "roughness": 3.0,
            "roughness_max": 10.0,
        },
        "ambient": {"name": "air", "sld": 0.0},
        "data_file": f.name,
        "intensity": {"value": 1.0, "fixed": True},
        "layers": [
            {
                "name": "Film",
                "sld": 4.0,
                "sld_min": 3.0,
                "sld_max": 5.0,
                "thickness": 150.0,
                "thickness_min": 120.0,
                "thickness_max": 180.0,
                "roughness": 5.0,
                "roughness_max": 12.0,
            }
        ],
    }
    _, r = build_experiment(truth).reflectivity()
    rng = np.random.default_rng(0)
    dr = 0.03 * r + 1e-9
    with open(f.name, "w") as fh:
        fh.write("# Q R dR dQ\n")
        for qq, rr, d in zip(q, r + rng.normal(0, dr), dr):
            fh.write(f"{qq:.6f}  {rr:.6e}  {d:.6e}  {0.02 * qq:.6e}\n")
    yield truth
    os.unlink(f.name)


def test_a_dream_fit_carries_a_json_safe_posterior(film_data, monkeypatch):
    monkeypatch.setenv("EVIDENCE_PROPOSALS", "300")
    fr = fitting.run_refl1d_fit(film_data, method="dream", steps=3000, burn=150)
    post = fr["posterior"]
    json.dumps(post, allow_nan=False)
    assert post["formula"] == POSTERIOR_FORMULA
    assert post["k"] == fr["_n_free_params"]
    assert set(post["mean"]) == set(fr["parameters"])
    assert post["log_evidence_bic"] == pytest.approx(
        -0.5 * fr["_chi2_total"] - 0.5 * fr["_n_free_params"] * math.log(fr["_n_data"]),
        abs=0.5,
    )
    assert post["log_evidence_method"] == "bridge"
    assert post["bridge"]["n_proposal"] == 300


def test_no_posterior_for_an_optimizer(film_data):
    fr = fitting.run_refl1d_fit(film_data, method="lm", steps=50)
    assert "posterior" not in fr


def test_bridge_sampling_leaves_the_problem_where_it_was(film_data, monkeypatch):
    from bumps.fitters import fit

    from aure.nodes.model_builder import build_problem

    problem = build_problem(film_data)
    result = fit(problem, **fitting._build_fit_options("dream", 2000, 100))
    problem.setp(result.x)
    before = problem.getp().copy()
    summarize_dream_fit(problem, result, n_proposal=200)
    assert np.array_equal(problem.getp(), before)


# ----------------------------------------------------------------------
# fitting: the evidence baseline
# ----------------------------------------------------------------------


def _fitting_state(**extra):
    state = {"fit_results": [], "best_evidence": None, "best_evidence_model": None}
    state.update(extra)
    return state


def test_the_first_trusted_fit_takes_the_evidence_baseline():
    fit = _fit(0, 150.0, 5, _post(-95.0, k=5))
    updates = {}
    fitting._update_evidence_baseline(fit, _fitting_state(), _model(), 0, updates)
    assert updates["best_evidence"]["log_evidence"] == -95.0
    assert "model_comparison" not in fit


def test_an_untrusted_fit_never_takes_it():
    fit = _fit(0, 150.0, 5, _post(-95.0, k=5, trusted=False))
    updates = {}
    fitting._update_evidence_baseline(fit, _fitting_state(), _model(), 0, updates)
    assert updates == {}


def test_a_better_fit_takes_it_and_records_the_comparison():
    base = _fit(0, 150.0, 5, _post(-95.0, k=5))
    state = _fitting_state(
        fit_results=[base],
        best_evidence=evidence_record(base, 0),
        best_evidence_model=_model(1),
    )
    latest = _fit(1, 135.0, 8, _post(-92.0, k=8))
    updates = {}
    fitting._update_evidence_baseline(latest, state, _model(2), 1, updates)
    assert latest["model_comparison"]["verdict"] == "better"
    assert updates["best_evidence"]["fit_index"] == 1
    assert updates["best_evidence_model"] == _model(2)


def test_an_inconclusive_more_complex_fit_leaves_it():
    base = _fit(0, 150.0, 5, _post(-95.0, k=5))
    state = _fitting_state(
        fit_results=[base],
        best_evidence=evidence_record(base, 0),
        best_evidence_model=_model(1),
    )
    latest = _fit(1, 135.0, 8, _post(-94.6, k=8))
    updates = {}
    fitting._update_evidence_baseline(latest, state, _model(2), 1, updates)
    assert latest["model_comparison"]["verdict"] == "inconclusive"
    assert latest["model_comparison"]["revert"] is True
    assert "best_evidence" not in updates


def _run_fitting_node(monkeypatch, state, result):
    """`fitting_node` with the fit itself replaced by a prepared FitResult."""
    monkeypatch.setenv("FIT_METHOD", "dream")
    monkeypatch.setattr(
        fitting, "run_fit_for_model", lambda **_k: copy.deepcopy(result)
    )
    out = fitting.fitting_node(state)
    return out, out["fit_results"][0]


def _aligned_state(base):
    """One trusted fit holding both baselines."""
    return {
        "current_model": _model(2),
        "data_files": [],
        "iteration": 1,
        "fit_results": [base],
        "best_chi2": base["chi_squared"],
        "best_model": _model(1),
        "best_bic": base["bic"],
        "best_bic_model": _model(1),
        "best_bic_chi2": base["chi_squared"],
        "bic_formula": BIC_FORMULA,
        "best_bic_evidence": evidence_record(base, 0),
        "best_evidence": evidence_record(base, 0),
        "best_evidence_model": _model(1),
    }


def test_the_bic_baseline_follows_an_evidence_promotion(monkeypatch):
    """Evidence better, BIC worse: both baselines move to the new fit."""
    base = _fit(0, 150.0, 5, _post(-95.5, k=5, chi2_total=150.0))
    latest = _fit(1, 137.0, 8, _post(-92.8, k=8, chi2_total=137.0))
    latest.pop("bic")
    out, fr = _run_fitting_node(monkeypatch, _aligned_state(base), latest)
    assert fr["bic"] > base["bic"]  # BIC alone would have kept the old baseline
    assert fr["model_comparison"]["verdict"] == "better"
    assert out["best_evidence"]["fit_index"] == 1
    assert out["best_bic"] == fr["bic"] and out["best_bic_model"] == _model(2)


def test_the_bic_baseline_stays_when_the_evidence_rejects(monkeypatch):
    """Evidence worse, BIC better: neither baseline moves, so they stay aligned."""
    base = _fit(0, 150.0, 5, _post(-95.0, k=5, chi2_total=150.0))
    latest = _fit(1, 120.0, 8, _post(-99.0, k=8, chi2_total=120.0))
    latest.pop("bic")
    out, fr = _run_fitting_node(monkeypatch, _aligned_state(base), latest)
    assert fr["bic"] < base["bic"]  # BIC alone would have promoted it
    assert fr["model_comparison"]["revert"] is True
    assert "best_bic" not in out and "best_evidence" not in out


def test_with_mixed_trust_bic_keeps_its_own_rule(monkeypatch):
    """The BIC baseline is an untrusted fit: BIC decides its own baseline."""
    untrusted = _fit(0, 150.0, 5, _post(-95.0, k=5, trusted=False))
    trusted = _fit(1, 149.0, 5, _post(-94.0, k=5))
    state = _aligned_state(trusted)
    state["fit_results"] = [untrusted, trusted]
    state.update(
        best_bic=untrusted["bic"],
        best_bic_model=_model(1),
        best_bic_chi2=untrusted["chi_squared"],
        best_bic_evidence=evidence_record(untrusted, 0),
        best_evidence=evidence_record(trusted, 1),
    )
    latest = _fit(2, 140.0, 8, _post(-99.0, k=8, chi2_total=140.0))
    latest.pop("bic")
    out, fr = _run_fitting_node(monkeypatch, state, latest)
    assert fr["model_comparison"]["verdict"] == "worse"  # against the trusted fit
    assert "best_evidence" not in out
    # BIC: 140 + 8 ln 160 > 150 + 5 ln 160, so BIC keeps the untrusted baseline too.
    assert "best_bic" not in out


# ----------------------------------------------------------------------
# evaluation: which criterion decides
# ----------------------------------------------------------------------


def test_trusted_evidence_overrides_a_bic_revert(offline):
    """BIC says the added layer costs too much; the evidence says it pays."""
    base = _fit(0, 150.0, 5, _post(-95.5, k=5, chi2_total=150.0))
    latest = _fit(1, 137.0, 8, _post(-92.8, k=8, chi2_total=137.0))
    assert latest["bic"] > base["bic"]  # BIC alone would revert
    state = _with_comparison(
        _state(base, latest, base_model=_model(1), latest_model=_model(2)), latest
    )
    state["best_evidence"] = evidence_record(latest, 1)  # fitting promoted it
    state["best_evidence_model"] = _model(2)

    out = evaluation.evaluation_node(state)

    assert "current_model" not in out
    cmp = state["fit_results"][-1]["model_comparison"]
    assert cmp["criterion"] == "evidence" and cmp["verdict"] == "better"
    assert cmp["bic_agrees"] is False
    hyp = out["structural_hypotheses"][0]
    # Not reverted, so the χ² bookkeeping records the improvement.
    assert hyp["status"] == "confirmed"
    assert "guardrail" not in (hyp.get("notes") or "")


def test_evidence_worse_reverts_to_the_evidence_baseline(offline):
    base = _fit(0, 150.0, 5, _post(-95.0, k=5, chi2_total=150.0))
    latest = _fit(1, 120.0, 8, _post(-99.0, k=8, chi2_total=120.0))
    assert latest["bic"] < base["bic"]  # BIC alone would keep it
    state = _with_comparison(
        _state(base, latest, base_model=_model(1), latest_model=_model(2)), latest
    )

    out = evaluation.evaluation_node(state)

    assert out["current_model"] == _model(1)
    assert "posterior evidence fell" in state["fit_results"][-1]["issues"][0]
    assert "BIC alone would have kept it" in state["fit_results"][-1]["issues"][0]
    hyp = out["structural_hypotheses"][0]
    assert hyp["status"] == "rejected"
    assert "Evidence guardrail" in hyp["notes"]


def test_inconclusive_evidence_reverts_by_parsimony_without_rejecting(offline):
    base = _fit(0, 150.0, 5, _post(-95.0, k=5, chi2_total=150.0))
    latest = _fit(1, 135.0, 8, _post(-95.4, k=8, chi2_total=135.0))
    state = _with_comparison(
        _state(base, latest, base_model=_model(1), latest_model=_model(2)), latest
    )

    out = evaluation.evaluation_node(state)

    assert out["current_model"] == _model(1)
    hyp = out["structural_hypotheses"][0]
    assert hyp["status"] == "tried"
    assert "Evidence inconclusive" in hyp["notes"]


def test_untrusted_fits_keep_the_bic_guardrail(offline):
    base = _fit(0, 150.0, 5, _post(-95.0, k=5, trusted=False, method="laplace"))
    latest = _fit(1, 137.0, 8, _post(-99.0, k=8, trusted=False, method="laplace"))
    state = _state(base, latest, base_model=_model(1), latest_model=_model(2))

    out = evaluation.evaluation_node(state)

    assert out["current_model"] == _model(1)  # BIC reverted, as before
    cmp = state["fit_results"][-1]["model_comparison"]
    assert cmp["criterion"] == "bic" and cmp["provisional_evidence"] is None
    hyp = out["structural_hypotheses"][0]
    assert hyp["status"] == "rejected"
    assert hyp["notes"] == "BIC guardrail reverted the structural change."


def test_a_bic_revert_contested_by_the_provisional_evidence_is_inconclusive(offline):
    """The sub-resolution layer case of docs/evidence.md §6: BIC rejects, the
    posterior favours it. The model still reverts — provisional evidence never
    decides — but the hypothesis is not recorded as ruled out."""
    base = _fit(0, 149.0, 5, _post(-95.4, k=5, trusted=True, chi2_total=149.0))
    latest = _fit(1, 137.0, 8, _post(-92.7, k=8, trusted=False, chi2_total=137.0))
    assert latest["bic"] > base["bic"]
    state = _state(base, latest, base_model=_model(1), latest_model=_model(2))

    out = evaluation.evaluation_node(state)

    assert out["current_model"] == _model(1)
    cmp = state["fit_results"][-1]["model_comparison"]
    assert cmp["criterion"] == "bic" and cmp["contested"] is True
    assert cmp["provisional_evidence"]["verdict"] == "better"
    issue = state["fit_results"][-1]["issues"][0]
    assert "posterior evidence disagrees" in issue
    hyp = out["structural_hypotheses"][0]
    assert hyp["status"] == "tried"
    assert "provisional posterior evidence favours it" in hyp["notes"]


def test_a_bic_revert_the_evidence_agrees_with_is_still_a_rejection(offline):
    base = _fit(0, 120.0, 8, _post(-94.2, k=8, trusted=False, chi2_total=120.0))
    latest = _fit(1, 119.0, 11, _post(-97.5, k=11, trusted=False, chi2_total=119.0))
    state = _state(base, latest, base_model=_model(2), latest_model=_model(3))

    out = evaluation.evaluation_node(state)

    assert out["current_model"] == _model(2)
    assert state["fit_results"][-1]["model_comparison"]["contested"] is False
    assert out["structural_hypotheses"][0]["status"] == "rejected"


def test_a_fit_that_just_set_the_bic_baseline_is_not_compared_with_itself(offline):
    base = _fit(0, 150.0, 5, _post(-95.0, k=5, trusted=False))
    latest = _fit(1, 120.0, 8, _post(-90.0, k=8, trusted=False))
    state = _state(base, latest, base_model=_model(1), latest_model=_model(2))
    # What `fitting` leaves behind when the latest fit wins the BIC baseline.
    state.update(
        best_bic=latest["bic"],
        best_bic_model=_model(2),
        best_bic_evidence=evidence_record(latest, 1),
    )
    out = evaluation.evaluation_node(state)
    cmp = state["fit_results"][-1]["model_comparison"]
    assert cmp["baseline_is_this_fit"] is True
    assert cmp["provisional_evidence"] is None and cmp["contested"] is False
    assert "current_model" not in out


def test_the_evaluator_is_told_about_the_posterior(offline):
    base = _fit(0, 149.0, 5, _post(-95.4, k=5, chi2_total=149.0))
    latest = _fit(1, 137.0, 8, _post(-92.7, k=8, trusted=False, chi2_total=137.0))
    state = _state(base, latest, base_model=_model(1), latest_model=_model(2))
    evaluation.evaluation_node(state)
    fit_seen = offline["fit_result"]
    assert fit_seen["posterior"]["log_evidence"] == -92.7
    assert fit_seen["model_comparison"]["contested"] is True


def test_complexity_assessment_renders_the_posterior_lines():
    post = _post(-92.7, k=8, trusted=False, chi2_total=137.0)
    post["occam"] = {"thin rho": 0.4, "thin thickness": 0.8, "poly rho": 5.7}
    post["occam_bic_per_param"] = HALF_LN_N
    cmp = {
        "criterion": "bic",
        "contested": True,
        "provisional_evidence": {
            "delta_log_evidence": 2.7,
            "error": 0.3,
            "verdict": "better",
        },
    }
    text = _format_complexity_assessment(
        bic=177.6,
        best_bic=174.4,
        n_params=8,
        n_layers=2,
        posterior=post,
        comparison=cmp,
    )
    assert "NOT converged" in text
    assert "provisional" in text
    assert "thin rho, thin thickness" in text
    assert "CONTESTS the BIC verdict" in text
    assert "BIC implies" in text


# ----------------------------------------------------------------------
# finalize: ranking by evidence
# ----------------------------------------------------------------------


def _history(*models):
    return [{"iteration": i, "definition": m} for i, m in enumerate(models)]


def test_finalize_ranks_by_evidence_when_every_candidate_is_trusted():
    simple = _fit(0, 149.0, 5, _post(-95.5, k=5))
    layered = _fit(1, 137.0, 8, _post(-92.8, k=8))
    state = {
        "fit_results": [simple, layered],
        "model_history": _history(_model(1), _model(2)),
        "current_model": _model(2),
        "iteration": 2,
    }
    sel = finalize_node(state)["final_selection"]
    assert sel["evidence_ranked"] is True
    assert sel["index"] == 1
    assert "posterior evidence" in sel["criterion"]


def test_finalize_breaks_an_evidence_tie_by_parsimony():
    simple = _fit(0, 149.0, 5, _post(-95.0, k=5))
    layered = _fit(1, 137.0, 8, _post(-94.7, k=8))
    state = {
        "fit_results": [simple, layered],
        "model_history": _history(_model(1), _model(2)),
        "current_model": _model(2),
        "iteration": 2,
    }
    sel = finalize_node(state)["final_selection"]
    assert sel["evidence_ranked"] is True
    assert sel["index"] == 0 and sel["parsimony_tiebreak"] is True


def test_finalize_keeps_the_chi2_rule_when_any_candidate_lacks_a_trusted_evidence():
    simple = _fit(0, 149.0, 5, _post(-95.5, k=5))
    layered = _fit(1, 137.0, 8, _post(-92.8, k=8, trusted=False))
    state = {
        "fit_results": [simple, layered],
        "model_history": _history(_model(1), _model(2)),
        "current_model": _model(2),
        "iteration": 2,
    }
    sel = finalize_node(state)["final_selection"]
    assert sel["evidence_ranked"] is False
    assert "lowest chi2" in sel["criterion"]


# ----------------------------------------------------------------------
# final_fit and the report surfaces
# ----------------------------------------------------------------------


def _full_posterior(trusted):
    """A real summary from a small Gaussian draw, so locate_point has data."""
    from aure.tools.posterior_tools import summarize_posterior

    rng = np.random.default_rng(0)
    g, c, k = 200, 20, 2
    pts = rng.normal(size=(g, c, k)) + np.array([5.0, 1.0])
    if not trusted:
        pts = pts * 0.05 + rng.normal(scale=2.0, size=(1, c, k)) + np.array([5.0, 1.0])
    x = pts.reshape(g * c, k)

    def ell(y):
        return -0.5 * np.sum((np.atleast_2d(y) - [5.0, 1.0]) ** 2, axis=1) - 60.0

    return summarize_posterior(
        x,
        ell(x),
        best_x=[5.0, 1.0],
        best_logp=-60.0,
        lower=[0, -9],
        upper=[10, 11],
        labels=["a", "b"],
        n_data=N_DATA,
        chain_ids=np.tile(np.arange(c), g),
        log_density=ell,
    )


def test_final_fit_places_the_exploration_optimum_in_the_dream_posterior(monkeypatch):
    monkeypatch.setenv("FIT_METHOD", "de")
    monkeypatch.setenv("FIT_METHOD_FINAL", "dream")
    post = _full_posterior(trusted=True)
    dream = {
        "iteration": 0,
        "method": "dream",
        "chi_squared": 0.70,
        "parameters": {"a": 5.0, "b": 1.0},
        "uncertainties": {"a": 1.0, "b": 1.0},
        "posterior": post,
    }
    monkeypatch.setattr(ff, "run_fit_for_model", lambda **_k: dream)
    explored = {
        "iteration": 0,
        "method": "de",
        "chi_squared": 0.80,
        "parameters": {"a": 9.0, "b": 1.0},  # 4σ off in a: a different basin
        "_chi2_total": 140.0,
    }
    state = {
        "current_model": _model(1),
        "current_chi2": 0.80,
        "fit_results": [explored],
        "final_selection": {"selected": True, "index": 0, "iteration": 0},
    }
    out = ff.final_fit_node(state)
    rec = out["final_fit"]
    assert rec["posterior"]["fit_trusted"] is True
    placed = rec["exploration_in_posterior"]
    assert placed["mahalanobis2_per_param"] > 1.0 and placed["superseded"] is True
    assert "not** at the minimum" in out["messages"][0]["content"]


def test_cli_report_and_json_payload_disclose_the_posterior():
    import click
    from click.testing import CliRunner

    from aure.cli import _print_analysis_results, _selection_payload

    post = _full_posterior(trusted=False)
    result = {
        "Q": [0.01, 0.02],
        "fit_results": [
            {
                "iteration": 0,
                "method": "dream",
                "chi_squared": 1.1,
                "parameters": {"a": 5.0},
                "posterior": post,
            }
        ],
        "final_selection": {"selected": True, "index": 0, "iteration": 0},
    }
    payload = _selection_payload(result)["posterior"]
    assert payload["source"] == "selected" and payload["fit_trusted"] is False
    json.dumps(payload, allow_nan=False)

    @click.command()
    def cmd():
        _print_analysis_results(result)

    text = CliRunner().invoke(cmd).output
    assert "Posterior Check (DREAM fit)" in text
    assert "did not converge" in text
    assert "Not trusted" in text


def test_web_parameters_payload_carries_the_posterior():
    from aure.web.data import RunData

    post = _full_posterior(trusted=True)
    explore = {"iteration": 0, "chi_squared": 1.0, "parameters": {}}
    dream = {"iteration": 0, "chi_squared": 0.9, "parameters": {}, "posterior": post}
    state = {
        "fit_results": [explore, dream],
        "final_selection": {"selected": True, "index": 0},
        "final_fit": {"ran": True},
    }
    # The displayed fit has no posterior of its own, but it is the selected
    # answer and the final DREAM polish characterized it.
    shown = RunData._posterior(state, state["fit_results"], 0, state["final_selection"])
    assert shown["source"] == "final_fit" and shown["fit_trusted"] is True
    own = RunData._posterior(state, state["fit_results"], 1, state["final_selection"])
    assert own["source"] == "displayed"
