"""Tests for the deterministic χ² acceptance clamp in the evaluation node.

``CHI2_MAX`` used to be advisory (injected into the evaluation prompt, then the
LLM's boolean was trusted), so a fit that already met the threshold could still
be sent back around the refinement loop. ``_clamp_acceptance_to_chi2`` makes the
threshold binding: a χ² inside the acceptance window completes the run
regardless of the LLM's verdict.

Two properties shape most of what is tested here:

* The clamp is a **floor on acceptance, not a ceiling** — it only flips
  ``False → True``. Above the ceiling the evaluator's verdict stands untouched
  and none of the guards below are consulted; only the pre-existing
  profile-artifact veto can lower an LLM "yes".
* Everything the threshold cannot see makes the clamp **stand down** — decline
  to force acceptance and let the LLM's verdict decide, exactly as before the
  clamp existed. That covers the SLD-profile artifact veto *and* the case where
  that veto could not reach an answer (a non-answer must not read as "clean"); a
  per-file/per-state χ² above the ceiling (``+inf`` is a deliberate "this
  dataset's fit failed" sentinel that must not hide under a passing aggregate);
  and a χ² below ``chi2_min`` (residuals smaller than the quoted uncertainties
  are evidence about the error model, not the structure).

**Known limitation:** the stop is inert for multi-state co-refinement. refl1d
writes one profile per model and only the top-level one reaches the detector, so
a co-refined fit is never marked verified and the clamp always stands down;
those runs still finish on the evaluator's verdict.
"""

import json
import logging

import numpy as np
import pytest
from scipy.special import erf

from aure.nodes import evaluation
from aure.nodes.evaluation import _clamp_acceptance_to_chi2


# --- Fixtures ---


def _clamp(analysis, chi2, chi2_max=2.5, **kw):
    return _clamp_acceptance_to_chi2(analysis, chi2=chi2, chi2_max=chi2_max, **kw)


def _analysis(acceptable=False, issues=None, **extra):
    """An LLM verdict dict, as ``analyze_fit_quality_with_llm`` returns it.

    Deliberately carries no ``_profile_checked``: that marker is set by the
    artifact detector, never by the LLM, so tests standing for "checked and
    clean" must pass it explicitly.
    """
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


def _checked(acceptable=False, **extra):
    """A verdict whose profile was checked and found clean."""
    return _analysis(acceptable=acceptable, _profile_checked=True, **extra)


def _model():
    return {
        "substrate": {"name": "silicon", "sld": 2.07, "roughness_max": 10.0},
        "layers": [{"name": "Cu", "thickness": 500.0, "sld": 6.42, "roughness": 8.0}],
        "ambient": {"name": "air", "sld": 0.0},
        "intensity": {"value": 1.0, "fixed": True},
    }


def _profile_from_slabs(interfaces):
    """Sum-of-error-functions profile from (z_boundary, rho_below, sigma)."""
    z = np.linspace(-100, 1100, 4000)
    rho = np.full_like(z, interfaces[0][1])
    prev = interfaces[0][1]
    for zb, rho_after, sigma in interfaces[1:]:
        s = max(sigma, 1e-3)
        rho = rho + 0.5 * (rho_after - prev) * (1 + erf((z - zb) / (s * np.sqrt(2))))
        prev = rho_after
    return z, rho


def _clean_profile():
    """A physically clean profile for ``_model()``: air → Cu → Si."""
    return _profile_from_slabs([(None, 0.0, None), (100, 6.42, 8), (600, 2.07, 6)])


def _fit_with_profile(z, rho):
    return {"sld_z": list(z), "sld_rho": list(rho), "parameters": {}}


# The Cu_K 207282 failure in minimal form: a thin low-SLD layer whose oversized
# neighbouring roughness lets its erf tail turn at an SLD no material provides.
ARTIFACT_INTERFACES = [
    (None, 0.0, None),
    (100, 6.42, 8),
    (600, -1.61, 10),
    (660, 2.07, 30),
]

ARTIFACT_MODEL = {
    "ambient": {"name": "air", "sld": 0.0},
    "substrate": {"name": "Si", "sld": 2.07, "roughness": 30},
    "layers": [
        {"name": "Cu", "sld": 6.42, "thickness": 500, "roughness": 8},
        {"name": "Ti", "sld": -1.61, "thickness": 60, "roughness": 10},
    ],
}


def _state(chi2, **extra):
    """A post-fitting state, carrying a clean exported SLD profile by default.

    A run only exports ``sld_z``/``sld_rho`` when it has an output directory and
    the clamp refuses a fit whose profile was never checked, so the default is a
    run that *did* export one; tests of the unchecked path pop the arrays.
    """
    z, rho = _clean_profile()
    state = {
        "iteration": 0,
        "fit_results": [
            {
                "chi_squared": chi2,
                "method": "amoeba",
                "converged": True,
                "parameters": {"Cu thickness": 500.0},
                "sld_z": z.tolist(),
                "sld_rho": rho.tolist(),
            }
        ],
        "current_model": _model(),
        "Q": list(np.linspace(0.01, 0.2, 50)),
        "messages": [],
    }
    state.update(extra)
    return state


def _per_file(**by_state):
    return [{"state": n, "label": n, "chi_squared": c} for n, c in by_state.items()]


def _offline(monkeypatch):
    monkeypatch.setattr(evaluation, "llm_available", lambda: True)
    monkeypatch.setattr(evaluation, "load_skill_context", lambda names, registry: "ctx")
    monkeypatch.setattr(
        evaluation, "select_skills", lambda *a, **k: ["neutron-reflectometry"]
    )
    # The not-acceptable branch reaches the hypothesis-revision step.
    monkeypatch.setattr(
        evaluation,
        "propose_hypothesis_revision_with_llm",
        lambda **kw: {"new_hypotheses": [], "ranking": []},
    )


@pytest.fixture
def stub_llm(monkeypatch):
    """Run the node offline; returns a setter for the verdict the LLM hands back.

    Re-set it between node calls in one test: the returned dict is a shallow copy,
    so the node's in-place appends to ``issues`` would otherwise leak forward.
    """
    _offline(monkeypatch)

    def _set(analysis):
        monkeypatch.setattr(
            evaluation, "analyze_fit_quality_with_llm", lambda **kwargs: dict(analysis)
        )

    return _set


@pytest.fixture
def stub_json_llm(monkeypatch):
    """Drive the REAL ``analyze_fit_quality_with_llm`` with a scriptable payload.

    ``stub_llm`` replaces that function wholesale, so it never reaches the parse
    step (or the prompt call) the tests using this fixture are about.
    """
    _offline(monkeypatch)
    box = {"payload": {}}

    class _LLM:
        def invoke(self, messages):
            class _R:
                content = "Sure:\n```json\n" + json.dumps(box["payload"]) + "\n```"

            return _R()

    monkeypatch.setattr(evaluation, "get_llm", lambda *a, **k: _LLM())
    return lambda payload: box.update(payload=payload)


# --- The acceptance window ---


def test_clamp_accepts_inside_the_window_inclusive_at_both_ends():
    analysis = _checked(issues=["roughness looks large"])
    assert _clamp(analysis, 1.8) is True
    assert analysis["acceptable"] is True
    assert analysis["_chi2_clamped"] is True
    # The LLM's objections survive untouched.
    assert analysis["issues"] == ["roughness looks large"]

    assert _clamp(_checked(), 2.5) is True  # at the ceiling
    assert _clamp(_checked(), 0.5, chi2_min=0.5) is True  # at the floor


def test_clamp_only_ever_flips_false_to_true_inside_the_window():
    already = _analysis(acceptable=True)
    assert _clamp(already, 1.0) is False
    assert already["acceptable"] is True
    assert "_chi2_clamped" not in already

    # Above the ceiling, and every χ² that is not a finite number.
    for chi2 in (2.51, float("inf"), float("nan"), None, True, "1.0"):
        outside = _checked()
        assert _clamp(outside, chi2) is False, chi2
        assert outside["acceptable"] is False
        assert "_chi2_clamped" not in outside


# --- The stand-down guards ---


def test_the_profile_detectors_two_verdicts_veto_and_permit_the_clamp():
    """The markers the clamp reads are set by the detector, not by hand."""
    bad = _analysis(acceptable=True)
    evaluation._detect_profile_artifacts_into(
        bad,
        _fit_with_profile(*_profile_from_slabs(ARTIFACT_INTERFACES)),
        ARTIFACT_MODEL,
    )
    # The pre-existing veto also lowers an LLM "yes".
    assert bad["acceptable"] is False
    assert bad["_profile_artifact"] is True
    assert _clamp(bad, 1.0) is False
    assert "_chi2_clamped" not in bad

    good = _analysis(acceptable=False)
    evaluation._detect_profile_artifacts_into(
        good, _fit_with_profile(*_clean_profile()), _model()
    )
    assert "_profile_artifact" not in good
    # "Checked and clean" is the positive statement the clamp requires.
    assert good["_profile_checked"] is True
    assert _clamp(good, 1.0) is True


@pytest.mark.parametrize(
    "fit,model,reason",
    [
        # No exported profile at all (library and MCP runs have none).
        ({}, _model(), "no exported SLD profile"),
        # Fewer than two resolvable media — no SLD range to test against.
        (_fit_with_profile(*_clean_profile()), {"layers": []}, "Fewer than two media"),
        # The detector's own "cannot check" return (conditions enumerated in
        # test_detector_reports_whether_it_actually_checked).
        (_fit_with_profile([0.0, 1.0, 2.0], [1.0, 46.3, 1.0]), _model(), "declined"),
    ],
)
def test_an_unevaluable_profile_is_not_marked_checked(fit, model, reason, caplog):
    """Every "cannot check" path leaves the fit unverified, says why, and stands
    the clamp down. The clamp's own log line points at "the preceding
    [EVALUATION] line for why", so each path must log its reason or that dangles.
    """
    analysis = _analysis(acceptable=False)
    with caplog.at_level(logging.INFO, logger="aure.nodes.evaluation"):
        evaluation._detect_profile_artifacts_into(analysis, dict(fit), model)
    assert not analysis.get("_profile_checked")
    assert reason in caplog.text
    assert _clamp(analysis, 0.9) is False


def test_a_co_refinement_is_never_marked_checked():
    """The documented limitation: only ``states[0]``'s profile reaches the
    detector, so a clean one is not evidence about the other states."""
    fit = _fit_with_profile(*_clean_profile())

    multi = _analysis(acceptable=False)
    two_states = dict(_model(), states=[{"name": "d2o"}, {"name": "h2o"}])
    evaluation._detect_profile_artifacts_into(multi, fit, two_states)
    assert "_profile_artifact" not in multi
    assert not multi.get("_profile_checked")
    assert _clamp(multi, 0.9) is False

    # One state, one profile, one verdict — the stand-down is multi-state only.
    single = _analysis(acceptable=False)
    evaluation._detect_profile_artifacts_into(
        single, fit, dict(_model(), states=[{"name": "d2o"}])
    )
    assert single["_profile_checked"] is True


def test_the_per_file_guard_blocks_bad_numbers_and_ignores_missing_ones(caplog):
    """``chi2`` is averaged over every model, so an unfitted contrast can hide
    under a passing aggregate. ``+inf`` is the deliberate "fit failed" sentinel,
    so it blocks too, and is named as one rather than printed as a real χ². A
    *missing* number is unknown, not bad, and must not block a legitimate clamp.
    """
    analysis = _checked()
    assert _clamp(analysis, 2.4, per_file_results=_per_file(d2o=0.9, h2o=18.4)) is False
    assert analysis["acceptable"] is False
    assert "_chi2_clamped" not in analysis

    failed = _per_file(d2o=0.85, h2o=float("inf"))
    over = evaluation._per_file_over_threshold(failed, 1.25)
    assert [label for label, _ in over] == ["h2o"]
    assert np.isinf(over[0][1])
    with caplog.at_level(logging.INFO, logger="aure.nodes.evaluation"):
        assert _clamp(_checked(), 0.9, chi2_max=1.25, per_file_results=failed) is False
    assert "h2o" in caplog.text
    assert "χ²=inf" in caplog.text

    unknown = [
        {"label": "low-Q"},
        {"label": "mid-Q", "chi_squared": None},
        {"label": "high-Q", "chi_squared": float("nan")},
        {"label": "flagged", "chi_squared": True},
        {"label": "combined", "chi_squared": 1.1},
        "not-a-dict",
    ]
    assert evaluation._per_file_over_threshold(unknown, 2.5) == []
    assert _clamp(_checked(), 1.4, per_file_results=unknown) is True


def test_the_floor_stands_the_clamp_down_and_can_be_switched_off():
    below = _checked()
    assert _clamp(below, 0.004, chi2_min=0.5) is False
    assert below["acceptable"] is False
    assert "_chi2_clamped" not in below

    # ``chi2_min=0`` is the documented off switch, and unit callers that pass no
    # floor keep the pre-floor behaviour.
    assert _clamp(_checked(), 0.004, chi2_min=0.0) is True
    assert _clamp(_checked(), 0.004) is True


def test_detector_reports_whether_it_actually_checked():
    """``has_artifact=False`` from a "cannot check" return means *unknown*, not
    *clean*, so the flag the clamp gates on is a separate positive statement."""
    from aure.tools.feature_tools import detect_profile_artifacts

    z, rho = _clean_profile()
    media = [0.0, 6.42, 2.07]
    assert detect_profile_artifacts(z, rho, media)["checked"] is True

    def declined(*args):
        res = detect_profile_artifacts(*args)
        return res["checked"] is False and res["has_artifact"] is False

    assert declined([0, 1, 2], [1.0, 46.3, 1.0], media)  # too few points
    assert declined(z[:50], rho[:40], media)  # mismatched lengths
    assert declined(z, rho, [2.07])  # fewer than two media
    assert declined(z, rho, [2.07, 2.07, 2.07])  # zero SLD span
    assert declined(z, np.full_like(rho, np.nan), media)  # diverged fit
    assert declined(np.where(z == z[3], np.nan, z), rho, media)  # NaN depth
    assert declined(z, rho, [0.0, float("nan"), 2.07])  # NaN medium


# --- Integration through ``evaluation_node`` ---


def test_node_stops_the_loop_on_a_passing_chi2_and_records_who_accepted(
    monkeypatch, stub_llm
):
    """The headline behaviour: an LLM "not acceptable" at a passing χ² still ends
    the run, reporting the objection rather than acting on it.
    ``chi2_clamp_accepted`` is written on every path — the runner's interactive
    pause reads it, and a restarted run must not inherit a stale True — and above
    the ceiling the LLM decides in both directions.
    """
    monkeypatch.setenv("CHI2_MAX", "1.25")

    stub_llm(
        _analysis(
            acceptable=False,
            issues=["Cu roughness is larger than expected"],
            suggestions=["try a graded interface"],
        )
    )
    clamped = evaluation.evaluation_node(_state(0.9))
    assert clamped["workflow_complete"] is True
    assert clamped["chi2_clamp_accepted"] is True
    content = clamped["messages"][0]["content"]
    assert "Fit Successful" in content
    assert "acceptance threshold" in content
    # The LLM's objection is reported, not dropped.
    assert "### Notes:" in content
    assert "Cu roughness is larger than expected" in content

    stub_llm(_analysis(acceptable=True))
    llm_accept = evaluation.evaluation_node(_state(9.0))
    assert llm_accept["workflow_complete"] is True
    assert llm_accept["chi2_clamp_accepted"] is False
    # Not clamped, so no "stopped at the threshold" wording.
    assert "acceptance threshold" not in llm_accept["messages"][0]["content"]

    stub_llm(_analysis(acceptable=False, issues=["poor high-Q agreement"]))
    refining = evaluation.evaluation_node(_state(9.0))
    assert "workflow_complete" not in refining
    assert refining["chi2_clamp_accepted"] is False
    assert any("Fit Evaluation" in m["content"] for m in refining["messages"])


def test_node_does_not_complete_on_a_non_physical_profile(monkeypatch, stub_llm):
    """A passing χ² with a non-physical SLD profile must NOT complete."""
    monkeypatch.setenv("CHI2_MAX", "2.5")
    stub_llm(_analysis(acceptable=False))

    state = _state(1.2, current_model=ARTIFACT_MODEL)
    state["fit_results"][-1].update(
        _fit_with_profile(*_profile_from_slabs(ARTIFACT_INTERFACES))
    )

    out = evaluation.evaluation_node(state)

    assert "workflow_complete" not in out
    assert any("excursion" in i.lower() for i in state["fit_results"][-1]["issues"])


def test_node_below_the_floor_leaves_the_verdict_to_the_llm(monkeypatch, stub_llm):
    """The gap the floor closes — χ² = 0.004 used to end the run on the spot —
    and the proof it is a stand-down and not a veto: a veto would send a good fit
    with conservative dR round the loop forever, the failure the clamp prevents.
    """
    monkeypatch.setenv("CHI2_MAX", "2.5")
    monkeypatch.delenv("CHI2_MIN", raising=False)  # the built-in 0.5 default

    stub_llm(_analysis(acceptable=False, issues=["dR looks overestimated"]))
    rejected = evaluation.evaluation_node(_state(0.004))
    assert "workflow_complete" not in rejected
    assert rejected["chi2_clamp_accepted"] is False

    stub_llm(_analysis(acceptable=True))
    accepted = evaluation.evaluation_node(_state(0.004))
    assert accepted["workflow_complete"] is True
    # Accepted by the LLM, not by the clamp.
    assert accepted["chi2_clamp_accepted"] is False
    assert any("Fit Successful" in m["content"] for m in accepted["messages"])


# --- Threshold validation and precedence ---


def test_threshold_validation_and_precedence(monkeypatch):
    """Hostile values are refused rather than trusted (``inf`` is the off switch
    users reach for and must not invert the clamp into "accept the first fit"),
    and state beats env beats default so a resumed run keeps its own window."""
    for raw in ("inf", "-inf", "nan", "-1", "abc", ""):
        monkeypatch.setenv("CHI2_MAX", raw)
        monkeypatch.setenv("CHI2_MIN", raw)
        assert evaluation._get_chi2_max() == 5.0, raw
        # The floor default is the same number ``_simple_evaluation`` calls
        # "Possible overfitting", on purpose, so the two cannot disagree.
        assert evaluation._get_chi2_min() == 0.5 == evaluation._CHI2_MIN_DEFAULT, raw

    monkeypatch.setenv("CHI2_MAX", "5.0")
    monkeypatch.setenv("CHI2_MIN", "0.9")
    assert evaluation._get_chi2_max({"chi2_max": 1.5}) == 1.5
    assert evaluation._get_chi2_min({"chi2_min": 0.2}) == 0.2

    # Absent / null / unusable in the state → the env var, which is what a
    # checkpoint written before the fields existed relies on.
    for st in ({}, {"chi2_max": None}, {"chi2_max": float("inf")}, {"chi2_max": 0}):
        assert evaluation._get_chi2_max(st) == 5.0
    for st in ({}, {"chi2_min": None}, {"chi2_min": float("nan")}, {"chi2_min": -1}):
        assert evaluation._get_chi2_min(st) == 0.9

    # ``0`` is unusable as a *ceiling* but is the floor's documented off value, so
    # a pinned or configured 0 floor must not fall through to the default.
    assert evaluation._get_chi2_min({"chi2_min": 0.0}) == 0.0
    monkeypatch.setenv("CHI2_MIN", "0")
    assert evaluation._get_chi2_min() == 0.0


def test_a_floor_not_below_the_ceiling_is_disabled_not_enforced(monkeypatch, caplog):
    """Such a pair leaves no acceptable χ² at all. Falling back to "no floor"
    keeps runs finishing; enforcing it would strand every one of them in the
    refinement loop. The setup loader rejects the pair, ``CHI2_MIN`` bypasses it.
    """
    monkeypatch.setenv("CHI2_MAX", "2.5")
    monkeypatch.setenv("CHI2_MIN", "2.5")
    with caplog.at_level(logging.WARNING, logger="aure.nodes.evaluation"):
        assert evaluation._get_chi2_min() == 0.0
    assert "not below the ceiling" in caplog.text

    # The ceiling it has to sit below is *this run's*, state-pinned or explicit.
    monkeypatch.setenv("CHI2_MIN", "2.0")
    assert evaluation._get_chi2_min({"chi2_max": 1.0}) == 0.0
    assert evaluation._get_chi2_min(chi2_max=1.0) == 0.0
    assert evaluation._get_chi2_min(chi2_max=9.0) == 2.0


def test_node_uses_the_state_window_not_the_ambient_env(monkeypatch, stub_llm):
    """A run launched with chi2_max=1.5 keeps refining at χ²=1.8 after resume,
    even though the resuming process has CHI2_MAX=5.0."""
    monkeypatch.setenv("CHI2_MAX", "5.0")
    stub_llm(_analysis(acceptable=False, issues=["residual fringes remain"]))

    assert "workflow_complete" not in evaluation.evaluation_node(
        _state(1.8, chi2_max=1.5)
    )

    out = evaluation.evaluation_node(_state(1.2, chi2_max=1.5))
    assert out["workflow_complete"] is True
    # …and the message quotes the window the run actually applied.
    assert "χ² ≤ 1.50" in out["messages"][0]["content"]


def test_runner_pins_the_window_and_keeps_a_resumed_one(monkeypatch):
    """The runner is where the window enters the state, so resumes inherit it."""
    from aure.workflow import runner

    def _run(initial):
        seen = {}

        def _capture(state):
            seen.update({k: state.get(k) for k in ("chi2_max", "chi2_min")})
            return {"workflow_complete": True}

        monkeypatch.setitem(runner.NODE_FUNCTIONS, "intake", _capture)
        monkeypatch.setitem(runner.NODE_FUNCTIONS, "finalize", lambda state: {})
        monkeypatch.setitem(runner.NODE_FUNCTIONS, "final_fit", lambda state: {})
        return seen, runner.run_workflow_with_checkpoints(initial_state=initial)

    monkeypatch.setenv("CHI2_MAX", "1.25")
    monkeypatch.setenv("CHI2_MIN", "0.3")
    seen, final = _run({"messages": []})
    assert seen == {"chi2_max": 1.25, "chi2_min": 0.3}
    assert (final["chi2_max"], final["chi2_min"]) == (1.25, 0.3)

    # A pinned 0.0 floor is a value, not an absence — which is why the runner
    # tests for it with ``is None``. Reading it as "unset" would re-read
    # ``CHI2_MIN`` and hand the run a floor it was launched without.
    monkeypatch.setenv("CHI2_MAX", "9.0")
    monkeypatch.setenv("CHI2_MIN", "2.0")
    seen, _ = _run({"messages": [], "chi2_max": 1.5, "chi2_min": 0.0})
    assert seen == {"chi2_max": 1.5, "chi2_min": 0.0}


def test_the_window_is_part_of_the_state_contract():
    """Not ad-hoc keys, and JSON-serializable so checkpoints round-trip."""
    from aure.state import ReflectivityState, create_initial_state

    for field in ("chi2_max", "chi2_min", "chi2_clamp_accepted"):
        assert field in ReflectivityState.__annotations__

    st = create_initial_state(data_file="d.txt", sample_description="Cu on Si")
    assert st["chi2_max"] is None
    assert st["chi2_min"] is None
    assert st["chi2_clamp_accepted"] is False
    assert json.loads(json.dumps({"chi2_min": st["chi2_min"]})) == {"chi2_min": None}


# --- What the run discloses ---


def test_the_sub_floor_finding_reaches_the_persisted_fit_result(monkeypatch, stub_llm):
    """``issues`` is the channel that reaches the user: it is copied onto the
    FitResult (hence the checkpoint, the success notes and the web tab), while
    ``physical_concerns`` stays node-local."""
    monkeypatch.setenv("CHI2_MAX", "2.5")
    monkeypatch.setenv("CHI2_MIN", "0.5")

    stub_llm(_analysis(acceptable=False))
    state = _state(0.004)
    evaluation.evaluation_node(state)

    low = [i for i in state["fit_results"][-1]["issues"] if "acceptance floor" in i]
    assert len(low) == 1
    assert "0.004" in low[0]
    assert "quoted uncertainties" in low[0]
    assert "dR" in low[0]

    # …and nothing is said about a floor a fit is comfortably above.
    stub_llm(_analysis(acceptable=False))
    above = _state(1.7)
    evaluation.evaluation_node(above)
    assert not [i for i in above["fit_results"][-1]["issues"] if "floor" in i]


def test_the_floor_is_disclosed_in_the_report_and_in_the_prompt():
    """A sub-floor χ² reaches the success message only because the evaluator
    accepted it on its own judgement, and the headline number invites the
    opposite reading. The evaluator, now deciding below the floor, has to be told
    what such a χ² says about the error model rather than guess. Both surfaces go
    quiet above the floor and when the floor is switched off — ``χ² ≥ 0`` would be
    noise, and worse, a bar every fit clears.
    """
    from aure.nodes.prompts import format_fit_evaluation_prompt

    fit = {"chi_squared": 0.004, "parameters": {}}
    ok = _analysis(acceptable=True)
    report = evaluation._format_success(fit, ok, chi2_max=2.5, chi2_min=0.5)
    assert "not a better fit" in report
    assert "acceptance floor" in report
    assert "quoted uncertainties" in report
    assert "The evaluator accepted it anyway" in report

    high = {"chi_squared": 1.7, "parameters": {}}
    assert "floor" not in evaluation._format_success(
        high, ok, chi2_max=2.5, chi2_min=0.5
    )
    assert "floor" not in evaluation._format_success(
        fit, ok, chi2_max=2.5, chi2_min=0.0
    )

    def _prompt(**kw):
        return format_fit_evaluation_prompt(
            sample_description="Cu on Si",
            hypothesis=None,
            chi_squared=0.004,
            method="amoeba",
            converged=True,
            parameters={},
            features={},
            chi2_max=2.5,
            **kw,
        )

    prompt = _prompt(chi2_min=0.5)
    assert "Acceptance floor: χ² ≥ 0.5" in prompt
    assert "quoted uncertainties" in prompt
    assert "dR" in prompt
    assert "Acceptance threshold: χ² ≤ 2.5" in prompt  # the ceiling block is untouched
    assert "floor" not in _prompt(chi2_min=0.0)
    assert "floor" not in _prompt()


def test_the_evaluator_is_told_the_floor_that_applies(monkeypatch, stub_json_llm):
    """From the node it is the run's pinned floor; from ``aure evaluate``, which
    passes none, ``None`` means "resolve from the environment"."""
    monkeypatch.setenv("CHI2_MAX", "2.5")
    monkeypatch.setenv("CHI2_MIN", "0.5")
    stub_json_llm({"acceptable": False, "issues": []})

    seen = {}
    monkeypatch.setattr(
        evaluation, "format_fit_evaluation_prompt", lambda **kw: seen.update(kw) or ""
    )

    evaluation.evaluation_node(_state(0.004, chi2_min=0.2))
    assert seen["chi2_min"] == 0.2

    seen.clear()
    evaluation.analyze_fit_quality_with_llm(
        fit_result={"chi_squared": 0.004},
        sample_description="Cu on Si",
        hypothesis=None,
        features=None,
        chi2_max=2.5,
    )
    assert seen["chi2_min"] == 0.5


def test_the_boundary_hit_note_matches_what_the_branch_actually_did(
    monkeypatch, stub_llm
):
    """Nothing is re-fitted after a clamped accept, so "auto-expanded" would lie
    there; on the refining path the expansion IS acted on, so the wording stands
    and the bounds-only shortcut is what matches on it."""
    monkeypatch.setenv("CHI2_MAX", "2.5")

    def _pinned_at_bound(chi2):
        state = _state(chi2)
        fit = state["fit_results"][-1]
        fit["parameters"] = {"Cu thickness": 800.0}
        fit["bounds"] = {"Cu thickness": [200.0, 800.0]}
        return state, fit

    stub_llm(_analysis(acceptable=False))
    state, fit = _pinned_at_bound(2.0)
    out = evaluation.evaluation_node(state)
    assert out["workflow_complete"] is True
    notes = "\n".join(fit["issues"]).lower()
    assert "auto-expanded" not in notes
    assert "auto-expanded" not in out["messages"][0]["content"].lower()
    # The pinned parameter is still reported — that is what the user needs.
    assert "pinned at its upper bound" in notes
    assert "unreliable" in notes
    # Nor is an unfittable bounds expansion published to the state.
    assert "current_model" not in out

    stub_llm(_analysis(acceptable=False))
    state, fit = _pinned_at_bound(9.0)
    out = evaluation.evaluation_node(state)
    assert "workflow_complete" not in out
    assert any("auto-expanded" in i.lower() for i in fit["issues"])
    assert isinstance(out["current_model"], dict)
    assert out["bounds_only_refinement"] is True


# --- Hypothesis outcomes on the accepting branch ---


def _hypotheses():
    return [
        {"id": 1, "title": "Native SiO2", "status": "tried", "tried_in_iteration": 1},
        {
            "id": 2,
            "title": "Graded Cu",
            "status": "pending",
            "tried_in_iteration": None,
        },
    ]


def test_accepted_fit_records_the_outcome_of_the_hypothesis_it_realized(
    monkeypatch, stub_llm
):
    """The accepting iteration is the *normal* terminus, so leaving it out of the
    bookkeeping reported the idea that worked as "tried, inconclusive"."""
    monkeypatch.setenv("CHI2_MAX", "2.5")
    stub_llm(_analysis(acceptable=False))

    hypotheses = _hypotheses()
    state = _state(1.4, iteration=1, best_chi2=1.4, structural_hypotheses=hypotheses)

    out = evaluation.evaluation_node(state)

    assert out["workflow_complete"] is True
    returned = out["structural_hypotheses"]
    assert returned[0]["status"] == "confirmed"
    assert "1.40" in returned[0]["notes"]
    # Untried entries are left alone so finalize can still list them.
    assert returned[1]["status"] == "pending"
    # Nodes mutate state only through the returned delta.
    assert hypotheses[0]["status"] == "tried"


@pytest.mark.parametrize(
    "chi2,best_chi2,accepted,expected",
    [
        # An accept waives the χ² comparison only when there is no baseline —
        # an accepting first iteration.
        (1.20, None, True, "confirmed"),
        # It must not waive a baseline that exists: the accepting iteration need
        # not be the run's best, and the accept branch never reaches the refine
        # branch's regression guardrail, so an unconditional bypass would confirm
        # a change that made the fit worse while finalize reports the earlier
        # model that lacks it.
        (1.20, 0.62, True, "tried"),
        (0.62, 0.62, True, "confirmed"),
        # The refine branch is unchanged by the new argument.
        (1.20, 0.62, False, "tried"),
        (0.60, 0.62, False, "confirmed"),
    ],
)
def test_accepted_does_not_loosen_the_outcome_rules(
    chi2, best_chi2, accepted, expected
):
    out = evaluation._update_hypothesis_outcomes(
        hypotheses=_hypotheses(),
        current_iteration=2,
        chi2=chi2,
        best_chi2=best_chi2,
        bic_reverted=False,
        accepted=accepted,
    )
    assert out[0]["status"] == expected


# --- The heuristic fallback reports; the clamp decides ---


def test_simple_evaluation_reports_without_asserting_acceptance():
    """Asserting ``chi2 <= chi2_max`` here short-circuited the clamp (which
    early-returns on an already-acceptable verdict), so a fit with no profile to
    check completed on χ² alone — on a path with no LLM judgement to defer to
    either. It stays False so the clamp is the single acceptance point, and its
    "Possible overfitting" flag reads the *configured* floor so the two agree.
    """
    good = evaluation._simple_evaluation({"chi_squared": 0.9}, chi2_max=1.25)
    assert good["acceptable"] is False
    assert good["quality_assessment"] == "good"
    assert good["issues"] == []

    def overfit(chi2, chi2_min):
        out = evaluation._simple_evaluation(
            {"chi_squared": chi2}, chi2_max=2.5, chi2_min=chi2_min
        )
        return any("Possible overfitting" in i for i in out["issues"])

    assert overfit(0.7, 1.0)
    assert not overfit(0.7, 0.5)
    assert not overfit(0.004, 0.0)  # the floor is off


# --- Interactive mode keeps its review pause on a clamped accept ---


@pytest.mark.parametrize("clamped,expected", [(True, ["evaluation"]), (False, [])])
def test_interactive_run_pauses_only_on_a_clamped_accept(
    monkeypatch, clamped, expected
):
    """A clamped accept overrode an evaluator that objected, and those objections
    go into the report unaddressed — exactly the decision an interactive run
    exists to put in front of the user, even though ``workflow_complete`` is set.
    A plain LLM accept owes no such confirmation point."""
    from aure.workflow import runner

    calls = []

    def _evaluation(state):
        calls.append(state.get("iteration"))
        return {
            "workflow_complete": True,
            "chi2_clamp_accepted": clamped,
            "iteration": 1,
            "fit_results": [{"chi_squared": 1.0, "issues": [], "suggestions": []}],
        }

    monkeypatch.setitem(runner.NODE_FUNCTIONS, "evaluation", _evaluation)
    monkeypatch.setitem(runner.NODE_FUNCTIONS, "finalize", lambda state: {})
    monkeypatch.setitem(runner.NODE_FUNCTIONS, "final_fit", lambda state: {})

    paused = []
    final = runner.run_workflow_with_checkpoints(
        initial_state={
            "messages": [],
            "interactive": True,
            "current_model": _model(),
            "iteration": 0,
            "max_iterations": 3,
            "chi2_max": 1.25,
        },
        start_node="evaluation",
        pause_callback=lambda state, node: paused.append(node),
    )

    assert paused == expected
    assert len(calls) == 1  # the pause does not re-run the node
    assert final["workflow_complete"] is True


# --- The evaluator's list fields are coerced once, at the parse site ---
#
# The node appends to ``analysis["issues"]`` on every sub-floor fit and on every
# boundary hit, and ``_detect_profile_artifacts_into`` extends all three fields.
# Those are LLM output, not a validated schema: ``"issues": "some text"`` raised
# ``AttributeError: 'str' object has no attribute 'append'`` and ``null`` raised
# the same on ``.extend``. The runner does not wrap the node call, so the whole
# run died — on a path that fires for every sub-floor fit.


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, []),
        ("   ", []),
        # A lone string becomes a one-element list rather than being dropped: an
        # evaluator reporting one issue as a bare string is reporting a real
        # finding, and losing it silently would be worse than the crash.
        ("  one issue  ", ["one issue"]),
        (("a", "b"), ["a", "b"]),
        # Consumers lower-case each element (the bounds-only shortcut,
        # ``_format_evaluation``), so the elements have to be strings too.
        ([None, "a", 1, "  "], ["a", "1"]),
        (0.25, ["0.25"]),
    ],
)
def test_as_text_list_semantics(raw, expected):
    assert evaluation._as_text_list(raw) == expected


def test_the_parse_site_coerces_every_field_so_a_sub_floor_fit_cannot_crash(
    monkeypatch, stub_json_llm
):
    """All three fields are normalized where the response is parsed, so every
    consumer downstream is safe — a dict is not iterated, which would yield its
    keys and lose the message."""
    monkeypatch.setenv("CHI2_MAX", "2.5")
    monkeypatch.setenv("CHI2_MIN", "0.5")
    stub_json_llm(
        {
            "acceptable": False,
            "issues": "dR looks overestimated",
            "suggestions": None,
            "physical_concerns": {"1": "the Ti well dips below Si"},
        }
    )

    out = evaluation.analyze_fit_quality_with_llm(
        fit_result={"chi_squared": 0.9, "parameters": {}},
        sample_description="Cu on Si",
        hypothesis=None,
        features=None,
        chi2_max=2.5,
        chi2_min=0.5,
    )
    assert out["issues"] == ["dR looks overestimated"]
    assert out["suggestions"] == []
    assert len(out["physical_concerns"]) == 1
    assert "the Ti well dips below Si" in out["physical_concerns"][0]

    # And the crash this closes, through the node: ``issues.append`` on a
    # sub-floor fit, with the node call unwrapped by the runner.
    state = _state(0.004)
    assert "workflow_complete" not in evaluation.evaluation_node(state)
    issues = state["fit_results"][-1]["issues"]
    assert "dR looks overestimated" in issues
    assert any("acceptance floor" in i for i in issues)


@pytest.fixture
def stub_unparseable_llm(monkeypatch):
    """Drive the REAL ``analyze_fit_quality_with_llm`` with unparseable output.

    ``_simple_evaluation`` only runs on that path, so the ``stub_llm`` fixture
    (which replaces ``analyze_fit_quality_with_llm`` wholesale) would skip the
    code under test entirely.
    """

    class _Msg:
        content = "The fit looks fine to me, but I am not answering in JSON."

    class _LLM:
        def invoke(self, messages):
            return _Msg()

    monkeypatch.setattr(evaluation, "llm_available", lambda: True)
    monkeypatch.setattr(evaluation, "get_llm", lambda *a, **k: _LLM())
    monkeypatch.setattr(evaluation, "load_skill_context", lambda names, registry: "ctx")
    monkeypatch.setattr(
        evaluation, "select_skills", lambda *a, **k: ["neutron-reflectometry"]
    )
    monkeypatch.setattr(
        evaluation,
        "propose_hypothesis_revision_with_llm",
        lambda **kw: {"new_hypotheses": [], "ranking": []},
    )


def test_the_fallback_path_cannot_complete_on_chi2_alone(
    monkeypatch, stub_unparseable_llm
):
    """The heuristic fallback reports; only the clamp accepts, guards and all.

    ``_simple_evaluation`` used to return ``acceptable = chi2 <= chi2_max``,
    which reached ``workflow_complete`` without consulting a single stand-down —
    so an unparseable evaluator reply accepted a fit whose profile was never
    checked. This is the only test that drives that real code path end to end.
    """
    monkeypatch.setenv("CHI2_MAX", "1.25")

    state = _state(0.9)
    state["fit_results"][-1].pop("sld_z")
    state["fit_results"][-1].pop("sld_rho")

    out = evaluation.evaluation_node(state)

    assert out["llm_calls"][0]["used_fallback"] is True
    assert "workflow_complete" not in out


# ======================================================================
# Co-refinement: every state's profile is checked, not just states[0]'s
# ======================================================================

_CONTRAST_MODEL = {
    "layers": [{"name": "Cu", "thickness": 500.0, "sld": 6.42, "roughness": 8.0}],
    "substrate": {"name": "Si", "sld": 2.07, "roughness": 5.0},
    "ambient": {"name": "solvent", "sld": 6.4},
    # A solvent-contrast series: the ambient SLD is a per-state override, so the
    # model-level template describes neither state.
    "states": [
        {"name": "D2O", "ambient": {"rho": 6.4}},
        {"name": "H2O", "ambient": {"rho": -0.56}},
    ],
}


def _contrast_fit(h2o_profile, *, drop_h2o=False):
    z, rho = _profile_from_slabs([(None, 6.4, None), (200, 6.42, 8), (700, 2.07, 5)])
    d2o = (list(z), list(rho))
    per_file = [
        {"state": "D2O", "chi_squared": 1.1, "sld_z": d2o[0], "sld_rho": d2o[1]},
        {
            "state": "H2O",
            "chi_squared": 1.3,
            **(
                {} if drop_h2o else {"sld_z": h2o_profile[0], "sld_rho": h2o_profile[1]}
            ),
        },
    ]
    return {
        # refl1d exports one profile per model, so the top-level pair is states[0]'s.
        "sld_z": d2o[0],
        "sld_rho": d2o[1],
        "parameters": {},
        "per_file_results": per_file,
    }


def _clean_h2o():
    z, rho = _profile_from_slabs([(None, -0.56, None), (200, 6.42, 8), (700, 2.07, 5)])
    return list(z), list(rho)


def test_a_co_refinement_with_every_state_clean_is_verified():
    """The payoff: the deterministic χ² stop was inert on every `states:` run,
    because only states[0]'s profile was ever read back. Note the clean H2O profile
    reaches -0.56, which would be flagged as an excursion if judged against the
    model-level ambient of 6.4 — hence the per-state media."""
    analysis = _analysis(acceptable=False)
    evaluation._detect_profile_artifacts_into(
        analysis, _contrast_fit(_clean_h2o()), _CONTRAST_MODEL
    )

    assert analysis.get("_profile_checked") is True
    assert not analysis.get("_profile_artifact")
    assert analysis["issues"] == []


def test_an_excursion_in_the_second_state_vetoes_and_names_it():
    """states[0] is clean, so checking only the top-level profile would miss this."""
    z, rho = _profile_from_slabs(
        [(None, -0.56, None), (200, 6.42, 60), (250, -3.0, 60), (700, 2.07, 5)]
    )
    bad_h2o = (list(z), list(rho))
    analysis = _analysis(acceptable=False)
    evaluation._detect_profile_artifacts_into(
        analysis, _contrast_fit(bad_h2o), _CONTRAST_MODEL
    )

    assert analysis["_profile_artifact"] is True
    assert analysis["acceptable"] is False
    assert not analysis.get("_profile_checked")
    assert "in state 'H2O'" in analysis["issues"][0]


def test_one_unreadable_state_leaves_the_whole_fit_unverified():
    """Fails closed: judging a co-refinement on the states that happened to export
    a profile is the "not checked read as clean" bug in another costume."""
    analysis = _analysis(acceptable=False)
    evaluation._detect_profile_artifacts_into(
        analysis, _contrast_fit(_clean_h2o(), drop_h2o=True), _CONTRAST_MODEL
    )

    assert not analysis.get("_profile_checked")
    assert not analysis.get("_profile_artifact")


# ======================================================================
# The refinement loop's regression baseline must not be a sub-floor fit
# ======================================================================


@pytest.mark.parametrize(
    "candidate,incumbent,wins,why",
    [
        (0.004, None, True, "nothing to compare against yet"),
        # The bug: one noise-absorbing fit became the bar every later honest fit
        # had to beat, so each read as a regression and got reverted.
        (3.50, 0.004, True, "an honest fit displaces a sub-floor baseline"),
        (0.004, 1.20, False, "a sub-floor fit cannot take it from an honest one"),
        (3.50, 1.20, False, "plain comparison still applies within the window"),
        (0.90, 1.20, True, "better in-window fit wins normally"),
        (0.004, 0.010, True, "all sub-floor: something must hold the baseline"),
    ],
)
def test_the_regression_baseline_prefers_an_in_window_fit(
    monkeypatch, candidate, incumbent, wins, why
):
    """`best_chi2`/`best_model` is what evaluation's guardrails revert *to*, so a
    fit the clamp refused to accept must not become it — but leaving the guardrails
    with no baseline at all would disable the check that stops the LLM refining an
    already-degraded model, so a sub-floor fit is recorded when nothing better
    exists."""
    from aure.nodes.fitting import _wins_baseline

    monkeypatch.setenv("CHI2_MIN", "0.5")
    monkeypatch.setenv("CHI2_MAX", "2.5")

    assert _wins_baseline(candidate, incumbent, {}) is wins, why


def test_a_disabled_floor_restores_plain_lowest_wins(monkeypatch):
    """`chi2_min: 0` is the documented off switch, so the baseline rule must reduce
    to exactly what it was before the floor existed."""
    from aure.nodes.fitting import _wins_baseline

    monkeypatch.setenv("CHI2_MIN", "0")
    monkeypatch.setenv("CHI2_MAX", "2.5")

    assert _wins_baseline(3.50, 0.004, {}) is False
    assert _wins_baseline(0.004, 1.20, {}) is True


def test_the_bic_baseline_is_ranked_by_bic_but_floored_on_chi2(monkeypatch):
    """BIC is monotone in χ², so a noise-absorbing fit wins on it for exactly the
    untrustworthy reason — and that guardrail has no slack and marks the tried
    hypothesis rejected, so it does more damage than the χ² one."""
    from aure.nodes.fitting import _wins_baseline

    monkeypatch.setenv("CHI2_MIN", "0.5")
    monkeypatch.setenv("CHI2_MAX", "2.5")

    # Worse BIC, but the incumbent baseline is sub-floor: trustworthy still wins.
    assert (
        _wins_baseline(3.50, 0.004, {}, candidate_score=-100.0, incumbent_score=-900.0)
        is True
    )
    # Both in-window: BIC decides, not χ².
    assert (
        _wins_baseline(1.90, 1.20, {}, candidate_score=-950.0, incumbent_score=-900.0)
        is True
    )


def test_a_sub_floor_state_blocks_the_clamp_and_is_reported(monkeypatch, stub_llm):
    """A contrast whose residuals are far below its quoted uncertainties constrains
    nothing, so the aggregate is carrying the co-refinement alone — the mirror of the
    over-ceiling guard, and the aggregate can look perfectly healthy. Live rather
    than theoretical now that the clamp fires on co-refinements at all."""
    from aure.nodes.evaluation import _per_file_under_floor

    monkeypatch.setenv("CHI2_MIN", "0.5")
    monkeypatch.setenv("CHI2_MAX", "2.5")

    per_file = [
        {"state": "d2o", "chi_squared": 0.004},
        {"state": "h2o", "chi_squared": 2.0},
    ]
    assert _per_file_under_floor(per_file, 0.5) == [("d2o", 0.004)]
    # Unknowns stay unknown, and ±inf cannot be below a finite floor.
    assert _per_file_under_floor([{"state": "a"}], 0.5) == []
    assert (
        _per_file_under_floor([{"state": "x", "chi_squared": float("inf")}], 0.5) == []
    )
    assert _per_file_under_floor(per_file, 0) == []  # floor disabled

    stub_llm(_analysis(acceptable=False))
    state = _state(1.0)
    state["fit_results"][-1]["per_file_results"] = per_file
    out = evaluation.evaluation_node(state)

    assert "workflow_complete" not in out
    assert out["chi2_clamp_accepted"] is False
    issues = state["fit_results"][-1]["issues"]
    assert any("d2o" in i and "no constraint" in i for i in issues)

    # Control: the same aggregate with both contrasts constraining the fit clamps,
    # so it is the sub-floor state doing the blocking and not something else.
    stub_llm(_analysis(acceptable=False))
    ok = _state(1.0)
    ok["fit_results"][-1]["per_file_results"] = [
        {"state": "d2o", "chi_squared": 1.1},
        {"state": "h2o", "chi_squared": 0.9},
    ]
    assert evaluation.evaluation_node(ok)["workflow_complete"] is True
