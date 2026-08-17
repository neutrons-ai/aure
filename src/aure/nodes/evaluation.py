"""
EVALUATION node: Assess fit quality and identify issues.

This node uses an LLM to analyze the fit results and determine:
- Is the fit acceptable (χ² close to 1)?
- Are there systematic residuals indicating model problems?
- Are parameters physically reasonable?
- What refinements might improve the fit?
"""

import copy
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import numpy as np
from langchain_core.messages import HumanMessage

from ..state import ReflectivityState, FitResult, Message, LLMCallRecord
from ..llm import llm_available, get_llm
from ..config import format_user_criteria
from ..skills import SkillRegistry, load_skill_context, select_skills
from .hypotheses import merge_structural_hypotheses, rerank_hypotheses
from .prompts import format_fit_evaluation_prompt, format_hypothesis_revision_prompt

logger = logging.getLogger(__name__)


_CHI2_MAX_DEFAULT = 5.0

# The lower end of the acceptance window. A reduced χ² this far below 1 says the
# residuals are much smaller than the quoted uncertainties — an overestimated
# ``dR`` column, or a model free enough to absorb the noise. That is evidence
# about the *error bars*, not the structure, so the accept clamp must not read it
# as a pass. 0.5 is deliberately the same number ``_simple_evaluation`` calls
# "Possible overfitting", so the two cannot contradict each other.
_CHI2_MIN_DEFAULT = 0.5


def _validated_threshold(
    raw: Any,
    source: str,
    *,
    default: float,
    relation: str,
    allow_zero: bool = False,
) -> Optional[float]:
    """Coerce a χ² threshold, returning None when the value is unusable.

    A non-finite threshold cannot be compared against usefully (``nan`` makes
    every comparison False, ``inf`` makes one side vacuous) and a negative one
    describes a χ² that cannot exist. *allow_zero* separates the two ends of the
    window: ``0`` is meaningless as a ceiling but is the documented off switch for
    the floor. ``setup.py`` rejects the same values for the YAML keys; the env
    vars and old checkpoints bypass that loader, so the rule is enforced here too.
    """
    if raw is None:
        return None
    value: Optional[float]
    if isinstance(raw, bool):
        value = None
    else:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = None
    unusable = (
        value is None
        or not np.isfinite(value)
        or (value < 0 if allow_zero else value <= 0)
    )
    if unusable:
        logger.warning(
            "[EVALUATION] %s must be a finite %s number, got %r — "
            "using χ² %s %.2f instead",
            source,
            "non-negative" if allow_zero else "positive",
            raw,
            relation,
            default,
        )
        return None
    return value


def _validated_chi2_max(raw: Any, source: str) -> Optional[float]:
    """Validate an acceptance *ceiling* (``chi2_max``); ``0`` is not usable."""
    return _validated_threshold(raw, source, default=_CHI2_MAX_DEFAULT, relation="≤")


def _validated_chi2_min(raw: Any, source: str) -> Optional[float]:
    """Validate an acceptance *floor* (``chi2_min``); ``0`` disables it."""
    return _validated_threshold(
        raw, source, default=_CHI2_MIN_DEFAULT, relation="≥", allow_zero=True
    )


def _get_chi2_max(state: Optional[dict] = None) -> float:
    """Return the χ² acceptance threshold for this run.

    ``state["chi2_max"]`` wins when present: the threshold now terminates the
    refinement loop deterministically, so a resumed run must keep the value it was
    launched with rather than whatever ``CHI2_MAX`` the resuming process carries.
    ``CHI2_MAX`` is the source for the first run (the runner pins it into the
    state) and the fallback for checkpoints predating the state field.
    """
    if isinstance(state, dict):
        from_state = _validated_chi2_max(state.get("chi2_max"), "state['chi2_max']")
        if from_state is not None:
            return from_state
    from_env = _validated_chi2_max(os.environ.get("CHI2_MAX"), "CHI2_MAX")
    return from_env if from_env is not None else _CHI2_MAX_DEFAULT


def _get_chi2_min(
    state: Optional[dict] = None, chi2_max: Optional[float] = None
) -> float:
    """Return the χ² acceptance *floor* for this run (``0`` = no floor).

    Precedence mirrors :func:`_get_chi2_max` (state, then ``CHI2_MIN``, then the
    default) and for the same reason — the floor changes where the loop stops.

    *chi2_max* is the run's effective ceiling, resolved here if the caller has not
    already. A floor at or above the ceiling leaves no acceptable χ² at all and
    would strand every run in the refinement loop, so such a pair is refused in
    favour of no floor — the failure mode that still lets a run finish. The setup
    loader rejects it up front, but ``CHI2_MIN`` in the shell bypasses that.
    """
    if chi2_max is None:
        chi2_max = _get_chi2_max(state)

    floor: Optional[float] = None
    if isinstance(state, dict):
        floor = _validated_chi2_min(state.get("chi2_min"), "state['chi2_min']")
    if floor is None:
        floor = _validated_chi2_min(os.environ.get("CHI2_MIN"), "CHI2_MIN")
    if floor is None:
        floor = _CHI2_MIN_DEFAULT

    if floor >= chi2_max:
        logger.warning(
            "[EVALUATION] χ² acceptance floor %.3f is not below the ceiling "
            "%.3f — no χ² could ever be accepted, so the floor is disabled for "
            "this run (set CHI2_MIN below CHI2_MAX to restore it)",
            floor,
            chi2_max,
        )
        return 0.0
    return floor


def _per_file_over_threshold(per_file_results: Optional[list], chi2_max: float) -> list:
    """Return ``(label, χ²)`` for every per-file result that fails *chi2_max*.

    A missing/``None``/``NaN`` χ² is *unknown* and ignored: no number is not
    evidence of a bad fit, and treating it as one would disable the clamp for every
    fit that reports no per-file χ² at all.

    ``+inf`` is the opposite — ``fitting.py`` and ``refl1d_import.py`` write it
    deliberately to mean "this dataset's fit is unusable", so it *blocks* the
    clamp: a contrast whose fit blew up must not hide under a passing aggregate.
    """
    over = []
    for pf in per_file_results or []:
        if not isinstance(pf, dict):
            continue
        c = pf.get("chi_squared")
        if isinstance(c, bool) or not isinstance(c, (int, float)):
            continue
        if np.isnan(c) or c <= chi2_max:
            continue
        label = pf.get("state") or pf.get("label") or pf.get("file") or "?"
        over.append((str(label), float(c)))
    return over


def _per_file_under_floor(per_file_results: Optional[list], chi2_min: float) -> list:
    """Return ``(label, χ²)`` for every per-file result *below* *chi2_min*.

    The mirror of :func:`_per_file_over_threshold`, and for a sharper reason than
    symmetry: a contrast whose residuals are far smaller than its quoted
    uncertainties is contributing essentially **no constraint** to the
    co-refinement. The aggregate can look healthy while one state's `dR` column is
    overestimated to the point that the fit is effectively ignoring it — you believe
    you co-refined against three contrasts and in fact fitted two, which is the
    failure a contrast series exists to prevent. Counting statistics genuinely
    differ between contrasts, so this is a realistic outcome rather than an exotic
    one.

    Unknown values (missing/``None``/``NaN``) are ignored for the same reason as in
    the ceiling guard. ``±inf`` cannot be below a finite floor, so the fit-failed
    sentinel is naturally excluded here and handled there. Returns ``[]`` when the
    floor is disabled.
    """
    if chi2_min <= 0:
        return []
    under = []
    for pf in per_file_results or []:
        if not isinstance(pf, dict):
            continue
        c = pf.get("chi_squared")
        if isinstance(c, bool) or not isinstance(c, (int, float)):
            continue
        if np.isnan(c) or c >= chi2_min:
            continue
        label = pf.get("state") or pf.get("label") or pf.get("file") or "?"
        under.append((str(label), float(c)))
    return under


def _format_per_file_failures(over: list) -> str:
    """Render ``_per_file_over_threshold`` output for the stand-down log line.

    ``+inf`` is a failure sentinel, not a measured value, so it is named as such
    rather than printed as a number that reads like a real χ².
    """
    parts = []
    for label, c in over:
        if np.isinf(c):
            parts.append(f"{label} χ²=inf (fit failed / no usable points)")
        else:
            parts.append(f"{label} χ²={c:.3f}")
    return ", ".join(parts)


def _clamp_acceptance_to_chi2(
    analysis: dict,
    *,
    chi2: float,
    chi2_max: float,
    chi2_min: float = 0.0,
    per_file_results: Optional[list] = None,
) -> bool:
    """Force acceptance when χ² already meets the run's threshold.

    ``chi2_max`` is the run's contract with the user, so a fit that meets it stops
    reproducibly rather than at the LLM's discretion — otherwise the refinement
    loop spends fit and LLM budget re-litigating a fit that already passed. The
    LLM's objections are not discarded: they stay in ``analysis["issues"]`` and are
    reported as notes, and the ideas the run never tried are listed by ``finalize``.

    This is a *floor* on acceptance: it only flips ``False → True``, never the
    reverse. Above ``chi2_max`` the verdict is entirely the LLM's, so none of the
    guards below apply there.

    Four things outrank the clamp, each a defect the aggregate χ² cannot see:

    * ``_profile_artifact`` — a physically impossible SLD profile must never be
      accepted on χ² alone;
    * ``_profile_checked`` *absent* — the profile veto is the clamp's only safety
      net and it is inert wherever :func:`_detect_profile_artifacts_into` could not
      reach a trustworthy answer. "Not checked" is unsafe, not clean;
    * a per-file χ² above ``chi2_max`` or carrying the ``+inf`` "fit failed"
      sentinel — ``chi2`` is averaged over every model, so an entirely unfitted
      dataset can hide under a passing aggregate;
    * ``chi2`` below *chi2_min* — see ``_CHI2_MIN_DEFAULT``. ``0`` disables it.
    * a per-file χ² *below* ``chi2_min`` — that dataset's residuals are far smaller
      than its quoted uncertainties, so it is contributing essentially no
      constraint and the aggregate is carrying the co-refinement alone.

    Each is a *stand-down*, not a veto: the clamp declines to force acceptance and
    the evaluator's verdict decides, which is the pre-clamp behaviour. Vetoing
    would re-introduce the endless refinement the clamp exists to stop, since a fit
    whose ``dR`` really is conservative can legitimately be accepted at a low χ².

    Returns:
        True if the verdict was flipped to acceptable.
    """
    if analysis.get("acceptable"):
        return False
    if isinstance(chi2, bool) or not isinstance(chi2, (int, float)):
        return False
    if not np.isfinite(chi2) or chi2 > chi2_max:
        return False
    if chi2_min > 0 and chi2 < chi2_min:
        logger.info(
            "[EVALUATION] Not accepting on χ²=%.4g alone: it is below the "
            "acceptance floor χ² ≥ %.3f, i.e. the residuals are smaller than the "
            "quoted uncertainties (an overestimated dR column, or a model free "
            "enough to absorb the noise), so the LLM's verdict on the error "
            "model stands and refinement continues",
            chi2,
            chi2_min,
        )
        return False
    if analysis.get("_profile_artifact"):
        return False
    if not analysis.get("_profile_checked"):
        # Deliberately does not name a cause: several are possible and only
        # `_detect_profile_artifacts_into` knows which applied, so it logs the
        # specific reason at the point it declines to set the marker.
        logger.info(
            "[EVALUATION] Not accepting on χ²=%.3f alone: the SLD profile was "
            "not verified for non-physical excursions (see the preceding "
            "[EVALUATION] line for why), so the LLM's verdict stands and "
            "refinement continues",
            chi2,
        )
        return False
    over = _per_file_over_threshold(per_file_results, chi2_max)
    if over:
        logger.info(
            "[EVALUATION] Not accepting on aggregate χ²=%.3f: %s above the "
            "threshold χ² ≤ %.3f",
            chi2,
            _format_per_file_failures(over),
            chi2_max,
        )
        return False
    under = _per_file_under_floor(per_file_results, chi2_min)
    if under:
        logger.info(
            "[EVALUATION] Not accepting on aggregate χ²=%.3f: %s below the "
            "acceptance floor χ² ≥ %.3f, so that data is contributing essentially "
            "no constraint — its uncertainties look overestimated. The LLM's "
            "verdict decides instead",
            chi2,
            _format_per_file_failures(under),
            chi2_min,
        )
        return False

    analysis["acceptable"] = True
    analysis["_chi2_clamped"] = True
    logger.warning(
        "[EVALUATION] Overriding acceptable=False: χ²=%.3f already meets the "
        "acceptance threshold χ² ≤ %.3f",
        chi2,
        chi2_max,
    )
    return True


def _count_free_params(model: dict) -> int:
    """Count the number of free parameters in a ModelDefinition dict."""
    n = 0
    # Each layer: thickness + SLD + roughness = 3
    n += 3 * len(model.get("layers", []))
    # Substrate roughness (if roughness_max is set, it's free)
    substrate = model.get("substrate", {})
    if substrate.get("roughness_max") is not None:
        n += 1
    # Ambient SLD (free if not air and non-zero SLD)
    ambient = model.get("ambient", {})
    if ambient.get("name", "").lower() != "air" and ambient.get("sld", 0) != 0:
        n += 1
    # Intensity (free unless fixed)
    intensity = model.get("intensity", {})
    if not intensity.get("fixed", False):
        n += 1
    return n


def _compute_bic(chi2: float, n_data: int, n_params: int) -> float:
    """Compute the Bayesian Information Criterion for a reflectivity fit.

    BIC = n·ln(χ²) + k·ln(n)

    Lower BIC indicates a better balance of fit quality and model simplicity.
    """
    import math

    if chi2 <= 0 or n_data <= 0:
        return float("inf")
    return n_data * math.log(chi2) + n_params * math.log(n_data)


def evaluation_node(state: ReflectivityState) -> Dict[str, Any]:
    """
    Evaluate fit quality and suggest improvements using LLM.

    Args:
        state: Current workflow state

    Returns:
        State updates including evaluation and suggestions
    """
    iteration = state.get("iteration", 0) + 1
    updates = {
        "current_node": "evaluation",
        "messages": [],
        "iteration": iteration,
        "llm_calls": [],
    }

    logger.info(f"[EVALUATION] Iteration {iteration} - Analyzing fit quality")

    fit_results = state.get("fit_results", [])
    if not fit_results:
        updates["error"] = "No fit results to evaluate"
        return updates

    # Get latest fit result
    latest_fit = fit_results[-1]
    chi2 = latest_fit.get("chi_squared", float("inf"))
    logger.info(f"[EVALUATION] Current χ² = {chi2:.3f}")

    # ========== BIC (Complexity Penalty) ==========
    current_model = state.get("current_model")
    n_data = len(state.get("Q", []))
    if isinstance(current_model, dict) and n_data > 0:
        n_params = _count_free_params(current_model)
        n_layers = len(current_model.get("layers", []))
        bic = _compute_bic(chi2, n_data, n_params)
        latest_fit["bic"] = bic
        logger.info(
            f"[EVALUATION] BIC = {bic:.1f} (k={n_params}, "
            f"layers={n_layers}, n={n_data})"
        )
    else:
        n_params = 0
        n_layers = 0
        bic = None

    # ========== Boundary-Hit Detection ==========
    # Check if any fitted parameters are pinned at their range boundaries.
    # If so, auto-expand those bounds and flag the issue.
    boundary_hits = _check_boundary_hits(latest_fit)
    expanded_model = None
    if boundary_hits:
        model = state.get("current_model")
        if isinstance(model, dict):
            # Published to the state only on the refining path below. An
            # accepting verdict ends the run without another fit, and bounds
            # that nothing will ever explore would misdescribe the model the
            # run reports.
            expanded_model = _expand_model_bounds(model, boundary_hits)
        for bh in boundary_hits:
            logger.warning(
                "[EVALUATION] Parameter '%s' hit %s bound (value=%.4f, bound=%.4f)",
                bh["name"],
                bh["bound_hit"],
                bh["value"],
                bh["bound_value"],
            )

    # ========== Residual Fringe Analysis ==========
    # When the fit is not great, look for unmodeled oscillations in the
    # residual that reveal missing layer thicknesses.
    chi2_max = _get_chi2_max(state)
    chi2_min = _get_chi2_min(state, chi2_max=chi2_max)
    residual_ratio = latest_fit.get("residual_ratio", [])
    Q = state.get("Q", [])
    per_file_results = latest_fit.get("per_file_results") or []
    states_in_fit = sorted(
        {pf.get("state") for pf in per_file_results if pf.get("state")}
    )
    if states_in_fit and chi2 > chi2_max:
        # Multi-state path: analyse each state's per-file ratios independently.
        try:
            from ..tools.feature_tools import analyze_residual_fringes
        except Exception as e:
            logger.debug(f"[EVALUATION] Residual fringe import failed: {e}")
            analyze_residual_fringes = None  # type: ignore[assignment]

        per_state_analysis: dict = {}
        if analyze_residual_fringes is not None:
            for st_name in states_in_fit:
                Qs: list = []
                Rs: list = []
                for pf in per_file_results:
                    if pf.get("state") != st_name:
                        continue
                    Qs.extend(pf.get("Q_fit") or [])
                    Rs.extend(pf.get("residual_ratio") or [])
                if not (Qs and Rs and len(Qs) == len(Rs)):
                    continue
                try:
                    ra = analyze_residual_fringes(np.array(Qs), np.array(Rs))
                    per_state_analysis[st_name] = ra
                    if ra.get("has_residual_fringes"):
                        for t in ra.get("unmodeled_thicknesses", []):
                            logger.info(
                                f"[EVALUATION] [{st_name}] residual fringes "
                                f"suggest unmodeled layer ~{t['thickness']:.0f} Å "
                                f"({t['confidence']} confidence)"
                            )
                except Exception as e:
                    logger.debug(
                        f"[EVALUATION] Per-state fringe analysis failed for "
                        f"{st_name}: {e}"
                    )
        if per_state_analysis:
            latest_fit["per_state_residual_analysis"] = per_state_analysis
    elif residual_ratio and Q and chi2 > chi2_max:
        try:
            from ..tools.feature_tools import analyze_residual_fringes

            residual_analysis = analyze_residual_fringes(
                np.array(Q),
                np.array(residual_ratio),
            )
            latest_fit["residual_analysis"] = residual_analysis
            if residual_analysis.get("has_residual_fringes"):
                for t in residual_analysis["unmodeled_thicknesses"]:
                    logger.info(
                        f"[EVALUATION] Residual fringes suggest unmodeled "
                        f"layer ~{t['thickness']:.0f} Å ({t['confidence']} confidence)"
                    )
        except Exception as e:
            logger.debug(f"[EVALUATION] Residual fringe analysis failed: {e}")

    # ========== Analyze Fit Quality ==========
    if not llm_available():
        updates["error"] = (
            "LLM is required for fit evaluation. Please configure LLM_PROVIDER."
        )
        return updates

    user_criteria = format_user_criteria(state.get("user_config"))
    # Load skill context
    registry = SkillRegistry()
    active_skills = state.get("active_skills", [])
    skill_context = load_skill_context(active_skills, registry)
    try:
        analysis = analyze_fit_quality_with_llm(
            fit_result=latest_fit,
            sample_description=state.get("sample_description"),
            hypothesis=state.get("hypothesis"),
            features=state.get("extracted_features"),
            chi2_max=chi2_max,
            chi2_min=chi2_min,
            user_criteria=user_criteria,
            residual_analysis=latest_fit.get("residual_analysis"),
            boundary_hits=boundary_hits,
            bic=bic,
            best_bic=state.get("best_bic"),
            n_params=n_params,
            n_layers=n_layers,
            skill_context=skill_context,
            per_file_results=latest_fit.get("per_file_results"),
            fit_history=fit_results,
            structural_hypotheses=state.get("structural_hypotheses", []),
        )
        used_fallback = analysis.pop("_used_fallback", False)
        updates["llm_calls"].append(
            LLMCallRecord(
                node="evaluation",
                timestamp=datetime.now(timezone.utc).isoformat(),
                success=True,
                used_fallback=used_fallback,
                fallback_reason="LLM response could not be parsed; used heuristic evaluation"
                if used_fallback
                else None,
                error=None,
            )
        )
    except Exception as e:
        error_msg = str(e).lower()
        if (
            "quota" in error_msg
            or "rate" in error_msg
            or "limit" in error_msg
            or "429" in str(e)
        ):
            updates["error"] = (
                "LLM quota/rate limit exceeded. Please wait or switch provider."
            )
        else:
            updates["error"] = f"LLM call failed: {str(e)[:200]}"
        logger.error(f"[EVALUATION] LLM error: {e}")
        updates["llm_calls"].append(
            LLMCallRecord(
                node="evaluation",
                timestamp=datetime.now(timezone.utc).isoformat(),
                success=False,
                used_fallback=False,
                fallback_reason=None,
                error=str(e)[:200],
            )
        )
        return updates

    # ========== SLD-Profile Artifact Detection ==========
    # A χ²-optimal fit can still produce a physically impossible SLD profile
    # (a roughness erf-tail dipping below/above the bounding media). Detect it
    # from the fitted profile and, if real, raise an issue that drives a
    # refinement pass. The σ/thickness ratio is surfaced as an informational
    # concern only — a large roughness is legitimate for a graded profile.
    _detect_profile_artifacts_into(analysis, latest_fit, current_model)

    # ========== Deterministic χ² Accept Clamp ==========
    # The threshold is enforced here, not left to the LLM's verdict. Must run
    # after the artifact detector so the profile veto keeps precedence.
    clamped = _clamp_acceptance_to_chi2(
        analysis,
        chi2=chi2,
        chi2_max=chi2_max,
        chi2_min=chi2_min,
        per_file_results=latest_fit.get("per_file_results"),
    )
    # Surface *who* accepted, so the runner can tell a clamp-driven completion from
    # an LLM accept: the interactive review pause is gated on the run not being
    # complete, which the clamp made unreachable on the very iteration it ends the
    # run. Written on every path, so a restarted run cannot inherit a stale True.
    updates["chi2_clamp_accepted"] = bool(clamped)

    # ========== Suspiciously Low χ² ==========
    # An ``issue`` rather than a ``physical_concern`` because only ``issues`` is
    # copied onto the FitResult below, and that is what reaches the user.
    if chi2_min > 0 and np.isfinite(chi2) and chi2 < chi2_min:
        analysis["issues"].append(
            f"χ² = {chi2:.4g} is below the acceptance floor χ² ≥ {chi2_min:.2f}. "
            f"The residuals are smaller than the quoted uncertainties, which "
            f"usually means the dR column is overestimated or the model has "
            f"enough free parameters to absorb the noise — evidence about the "
            f"error bars rather than confirmation of the structure. Check the "
            f"uncertainties and the parameter count before trusting this fit."
        )

    # The same finding for a single dataset of a co-refinement, where the aggregate
    # can look healthy while one contrast constrains nothing.
    sub_floor_states = _per_file_under_floor(
        latest_fit.get("per_file_results"), chi2_min
    )
    if sub_floor_states:
        analysis["issues"].append(
            f"{_format_per_file_failures(sub_floor_states)} — below the acceptance "
            f"floor χ² ≥ {chi2_min:.2f} while the aggregate is χ² = {chi2:.4g}. "
            f"That data's residuals are far smaller than its quoted uncertainties, "
            f"so it is contributing essentially no constraint and the rest of the "
            f"co-refinement is carrying the fit. Check its dR column."
        )

    # ========== Boundary-Hit Issues ==========
    # Reported after the verdict is settled because the two paths have different
    # truths: the expanded bounds are only adopted (and re-fitted) when the run
    # refines. On an accepting verdict the run ends here, so claiming the range was
    # expanded would hide what the user needs told — a reported parameter is pinned.
    if boundary_hits:
        for bh in boundary_hits:
            if analysis["acceptable"]:
                analysis["issues"].append(
                    f"Parameter '{bh['name']}' is pinned at its "
                    f"{bh['bound_hit']} bound "
                    f"({bh['value']:.4f} ≈ {bh['bound_value']:.4f}), so its "
                    f"value and uncertainty are unreliable and the true optimum "
                    f"may lie outside the range. The run stopped here, so the "
                    f"range was not re-fitted with the bound relaxed."
                )
            else:
                analysis["issues"].append(
                    f"Parameter '{bh['name']}' is at its {bh['bound_hit']} bound "
                    f"({bh['value']:.4f} ≈ {bh['bound_value']:.4f}). "
                    f"Range has been auto-expanded."
                )

    latest_fit["issues"] = analysis["issues"]
    latest_fit["suggestions"] = analysis["suggestions"]
    # Persist the verdict the clamp just read, so finalize and the report can tell
    # "checked and clean" from "never checked" instead of matching issue prose.
    # Written as a pair on every judged fit: the pair's presence is what marks the
    # fit as judged at all.
    latest_fit["profile_checked"] = bool(analysis.get("_profile_checked"))
    latest_fit["profile_artifact"] = bool(analysis.get("_profile_artifact"))
    latest_fit["next_action"] = analysis.get("next_action", "parameter_tweak")
    latest_fit["proposed_hypothesis_id"] = analysis.get("proposed_hypothesis_id")

    if analysis["issues"]:
        logger.info(f"[EVALUATION] Issues found: {analysis['issues']}")
    if analysis["suggestions"]:
        logger.info(f"[EVALUATION] Suggestions: {analysis['suggestions']}")

    # ========== Determine Next Action ==========
    if analysis["acceptable"]:
        logger.info("[EVALUATION] ✓ Fit acceptable - workflow complete")
        updates["workflow_complete"] = True
        # The accepted fit is the outcome of whatever hypothesis was realized in the
        # previous turn, so the bookkeeping runs on this branch too — the accepting
        # iteration is the *normal* terminus, and leaving it out reported the idea
        # that worked as "tried, inconclusive". Nothing was reverted here.
        accepted_hypotheses = list(state.get("structural_hypotheses", []) or [])
        if accepted_hypotheses:
            updates["structural_hypotheses"] = _update_hypothesis_outcomes(
                hypotheses=accepted_hypotheses,
                current_iteration=iteration,
                chi2=chi2,
                best_chi2=state.get("best_chi2"),
                bic_reverted=False,
                accepted=True,
            )
        updates["messages"] = [
            Message(
                role="assistant",
                content=_format_success(
                    latest_fit, analysis, chi2_max=chi2_max, chi2_min=chi2_min
                ),
                timestamp=None,
            )
        ]
    else:
        # The auto-expanded bounds are adopted here, where a re-fit follows.
        if expanded_model is not None:
            updates["current_model"] = expanded_model

        # ========== χ² Regression Guardrail ==========
        # If the current fit is worse than the best so far, revert to the
        # best model before sending it to the refinement loop. This prevents
        # the LLM from "refining" an already-degraded model.
        best_chi2 = state.get("best_chi2")
        best_model = state.get("best_model")
        chi2_reverted = False
        if best_chi2 is not None and best_model and chi2 > best_chi2 * 1.05:
            logger.warning(
                f"[EVALUATION] χ² regressed ({chi2:.3f} > best {best_chi2:.3f}) "
                f"— reverting to best model before refinement"
            )
            # Deepcopy so the next refine iteration's in-place edits of
            # current_model don't reach back into best_model.
            updates["current_model"] = copy.deepcopy(best_model)
            chi2_reverted = True
            analysis["issues"].insert(
                0,
                f"Previous refinement made the fit worse (χ² went from "
                f"{best_chi2:.2f} to {chi2:.2f}). Reverting to the best model "
                f"and trying a different approach.",
            )

        # ========== BIC Regression Guardrail ==========
        # If χ² improved (or stayed similar) but BIC worsened, the
        # added model complexity is not statistically justified.
        # Revert to the best BIC model (the simpler one).
        best_bic_val = state.get("best_bic")
        best_bic_mdl = state.get("best_bic_model")
        bic_reverted = False
        if (
            bic is not None
            and best_bic_val is not None
            and best_bic_mdl
            and bic > best_bic_val
            and not chi2_reverted
        ):
            logger.warning(
                f"[EVALUATION] BIC regressed ({bic:.1f} > best {best_bic_val:.1f}) "
                f"\u2014 added complexity not justified, reverting to simpler model"
            )
            updates["current_model"] = copy.deepcopy(best_bic_mdl)
            bic_reverted = True
            analysis["issues"].insert(
                0,
                f"Added layer(s) lowered χ² but increased BIC "
                f"({bic:.1f} vs best {best_bic_val:.1f}). The extra "
                f"complexity is not statistically justified. Reverting to "
                f"the simpler model and trying a different approach.",
            )

        # ========== Hypothesis status updates ==========
        # Mark the previously tried hypothesis (if any) as confirmed or
        # rejected based on how this iteration's result turned out. The
        # full updated list is then passed into modeling for the next turn.
        hypotheses = list(state.get("structural_hypotheses", []) or [])
        if hypotheses:
            hypotheses = _update_hypothesis_outcomes(
                hypotheses=hypotheses,
                current_iteration=iteration,
                chi2=chi2,
                best_chi2=best_chi2,
                bic_reverted=bic_reverted,
            )

        # ========== Hypothesis revision (gated) ==========
        # When the fit evidence warrants it, re-select skills from the
        # observed artifacts and ask the LLM to propose genuinely new
        # hypotheses and re-rank the list. This is the only place besides
        # intake that may grow the backlog (the modeling node is status-only).
        if _should_revise_hypotheses(latest_fit, hypotheses, fit_results):
            try:
                rev = _revise_hypotheses(
                    state=state,
                    latest_fit=latest_fit,
                    hypotheses=hypotheses,
                    iteration=iteration,
                    bic=bic,
                    boundary_hits=boundary_hits,
                    analysis=analysis,
                    fit_history=fit_results,
                    registry=registry,
                    active_skills=active_skills,
                )
                hypotheses = rev["hypotheses"]
                if rev["active_skills"] != list(active_skills):
                    updates["active_skills"] = rev["active_skills"]
                updates["llm_calls"].append(
                    LLMCallRecord(
                        node="evaluation",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        success=True,
                        used_fallback=False,
                        fallback_reason=None,
                        error=None,
                    )
                )
                if rev["changed"]:
                    bits = []
                    if rev["n_new"]:
                        bits.append(
                            f"added {rev['n_new']} new hypothesis(es) from fit evidence"
                        )
                    if rev["added_skills"]:
                        bits.append(
                            f"activated skill(s): {', '.join(rev['added_skills'])}"
                        )
                    bits.append("re-ranked the hypothesis list")
                    updates["messages"].append(
                        Message(
                            role="assistant",
                            content="**Hypothesis revision:** " + "; ".join(bits) + ".",
                            timestamp=None,
                        )
                    )
            except Exception as e:
                logger.warning("[EVALUATION] Hypothesis revision failed: %s", e)
                updates["llm_calls"].append(
                    LLMCallRecord(
                        node="evaluation",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        success=False,
                        used_fallback=True,
                        fallback_reason="Hypothesis revision failed; keeping existing list",
                        error=str(e)[:200],
                    )
                )

        if hypotheses:
            updates["structural_hypotheses"] = hypotheses

        # ========== Bounds-only refinement shortcut ==========
        # If the ONLY thing that changed this iteration is an auto-bound
        # expansion (no LLM-suggested refinements), skip the modeling node
        # and go straight back to fitting with the expanded model. This
        # avoids spending an entire LLM round on "widen bounds, re-fit".
        llm_issues_excl_bounds = [
            i
            for i in analysis.get("issues", [])
            if not ("bound" in i.lower() and "auto-expanded" in i.lower())
        ]
        llm_suggestions_excl_bounds = [
            s for s in analysis.get("suggestions", []) if "bound" not in s.lower()
        ]
        only_bounds = (
            bool(boundary_hits)
            and not llm_issues_excl_bounds
            and not llm_suggestions_excl_bounds
        )
        if only_bounds and not chi2_reverted and not bic_reverted:
            updates["bounds_only_refinement"] = True
            logger.info(
                "[EVALUATION] Only bound-expansion needed — routing directly to fitting"
            )
        else:
            updates["bounds_only_refinement"] = False

        logger.info("[EVALUATION] ✗ Fit not acceptable - proceeding to refinement")
        updates["messages"].append(
            Message(
                role="assistant",
                content=_format_evaluation(latest_fit, analysis, iteration=iteration),
                timestamp=None,
            )
        )

    return updates


def _update_hypothesis_outcomes(
    hypotheses: list,
    current_iteration: int,
    chi2: float,
    best_chi2: float | None,
    bic_reverted: bool,
    accepted: bool = False,
) -> list:
    """Update the status of a hypothesis that was tried in the previous turn.

    This is bookkeeping, not decision-making: the LLM at modeling time marked
    a hypothesis as ``tried`` for a specific iteration; now that the fit has
    been scored we record the outcome. The LLM itself remains in charge of
    choosing *which* hypothesis to try next.

    * If a hypothesis was marked ``tried`` in the previous iteration and the
      BIC guardrail just reverted the model, that hypothesis is marked
      ``rejected``.
    * If χ² improved relative to the best-so-far, the hypothesis is marked
      ``confirmed``.
    * Otherwise the status is left as ``tried`` so the LLM can decide.

    *accepted* (the run ends on this fit) waives the χ² comparison only when there
    is no baseline to compare against — ``best_chi2 is None``, i.e. an accepting
    first iteration. It must **not** waive a baseline that exists: the accepting
    iteration need not be the run's best one, and the accept branch never reaches
    the refine branch's χ² regression guardrail, so an unconditional bypass would
    mark a hypothesis that made the fit *worse* as ``confirmed`` while ``finalize``
    reports the earlier model that lacks that change.
    """
    prev_iter = current_iteration - 1
    updated = [dict(h) for h in hypotheses]
    for h in updated:
        if h.get("status") != "tried":
            continue
        if h.get("tried_in_iteration") != prev_iter:
            continue
        if bic_reverted:
            h["status"] = "rejected"
            h["notes"] = (
                (h.get("notes", "") + " ") if h.get("notes") else ""
            ) + "BIC guardrail reverted the structural change."
        elif (accepted and best_chi2 is None) or (
            best_chi2 is not None and chi2 <= best_chi2 * 1.01
        ):
            h["status"] = "confirmed"
            h["notes"] = (
                (h.get("notes", "") + " ") if h.get("notes") else ""
            ) + f"χ²={chi2:.2f} at iter {current_iteration}."
    return updated


def _chi2_series(fit_history: Optional[list]) -> list:
    """Extract finite χ² values from the fit history, in order."""
    out = []
    for fr in fit_history or []:
        c = fr.get("chi_squared")
        if isinstance(c, (int, float)) and c != float("inf"):
            out.append(float(c))
    return out


def _should_revise_hypotheses(
    latest_fit: FitResult,
    hypotheses: list,
    fit_history: Optional[list],
) -> bool:
    """Gate the (LLM-backed) hypothesis-revision step.

    Only worth an extra LLM call when there is a concrete signal that the
    intake-time hypothesis list may be incomplete:

    * residual fringes point to an unmodeled layer, or
    * no ``pending`` hypotheses remain (including an empty list), or
    * χ² has stalled (<5% improvement across the last two iterations).
    """
    ra = latest_fit.get("residual_analysis") or {}
    if ra.get("has_residual_fringes"):
        return True
    per_state = latest_fit.get("per_state_residual_analysis") or {}
    if any((v or {}).get("has_residual_fringes") for v in per_state.values()):
        return True

    if not any(h.get("status") == "pending" for h in (hypotheses or [])):
        return True

    chi2s = _chi2_series(fit_history)
    if len(chi2s) >= 3 and chi2s[-3] > 0:
        if (chi2s[-3] - chi2s[-1]) / chi2s[-3] < 0.05:
            return True
    return False


def _format_observations(
    latest_fit: FitResult,
    analysis: Dict[str, Any],
    boundary_hits: Optional[list],
    fit_history: Optional[list],
    bic: float | None,
) -> str:
    """Summarize fit evidence for the skill re-selection ``extra_context``."""
    lines = []
    chi2 = latest_fit.get("chi_squared")
    if isinstance(chi2, (int, float)):
        head = f"- Current χ²={chi2:.3f}"
        if bic is not None:
            head += f", BIC={bic:.1f}"
        lines.append(head)
    chi2s = _chi2_series(fit_history)
    if len(chi2s) >= 2:
        lines.append("- χ² trajectory: " + " → ".join(f"{c:.2f}" for c in chi2s[-4:]))

    ra = latest_fit.get("residual_analysis") or {}
    if ra.get("has_residual_fringes"):
        for t in ra.get("unmodeled_thicknesses", []):
            lines.append(
                f"- Residual fringe implies an unmodeled layer "
                f"~{t.get('thickness', 0):.0f} Å ({t.get('confidence', '?')} confidence)"
            )
    for bh in boundary_hits or []:
        lines.append(
            f"- Parameter '{bh['name']}' pinned at its {bh['bound_hit']} bound"
        )
    for c in analysis.get("physical_concerns") or []:
        lines.append(f"- Physical concern: {c}")
    for issue in (analysis.get("issues") or [])[:4]:
        lines.append(f"- Issue: {issue}")
    return "\n".join(lines) if lines else "(no specific artifacts)"


def propose_hypothesis_revision_with_llm(
    state: ReflectivityState,
    latest_fit: FitResult,
    hypotheses: list,
    bic: float | None,
    boundary_hits: Optional[list],
    analysis: Dict[str, Any],
    fit_history: Optional[list],
    skill_context: str,
) -> Dict[str, Any]:
    """Ask the LLM for new hypotheses + a re-ranking. Never raises.

    Returns ``{"new_hypotheses": [...], "ranking": [...]}`` (empty on any
    failure). New hypotheses carry no id; the ranking references existing
    hypotheses by integer id and new ones by ``"new1"``/``"new2"``/….
    """
    llm = get_llm(temperature=0)
    prompt = format_hypothesis_revision_prompt(
        sample_description=state.get("sample_description") or "",
        current_model=state.get("current_model") or {},
        skill_context=skill_context,
        structural_hypotheses=hypotheses,
        fit_history=fit_history,
        chi_squared=latest_fit.get("chi_squared", float("inf")),
        bic=bic,
        residual_analysis=latest_fit.get("residual_analysis"),
        boundary_hits=boundary_hits,
        concerns=(analysis.get("physical_concerns") or [])
        + (analysis.get("issues") or []),
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    match = re.search(r"\{[\s\S]*\}", response.content)
    if not match:
        return {"new_hypotheses": [], "ranking": []}
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError:
        logger.warning("[EVALUATION] Could not parse hypothesis-revision JSON")
        return {"new_hypotheses": [], "ranking": []}
    new_h = obj.get("new_hypotheses")
    ranking = obj.get("ranking")
    return {
        "new_hypotheses": new_h if isinstance(new_h, list) else [],
        "ranking": ranking if isinstance(ranking, list) else [],
    }


def _resolve_ranking(ranking_raw: list, new_ids: list) -> list:
    """Resolve a mixed ranking (ints + ``newK`` refs) to concrete ids."""
    ranked_ids = []
    for ref in ranking_raw:
        if isinstance(ref, bool):
            continue
        if isinstance(ref, int):
            ranked_ids.append(ref)
        elif isinstance(ref, str):
            m = re.match(r"\s*new\s*[:#-]?\s*(\d+)\s*$", ref, re.IGNORECASE)
            if m:
                k = int(m.group(1)) - 1
                if 0 <= k < len(new_ids):
                    ranked_ids.append(new_ids[k])
            else:
                try:
                    ranked_ids.append(int(ref.strip()))
                except (ValueError, AttributeError):
                    pass
    return ranked_ids


def _revise_hypotheses(
    *,
    state: ReflectivityState,
    latest_fit: FitResult,
    hypotheses: list,
    iteration: int,
    bic: float | None,
    boundary_hits: Optional[list],
    analysis: Dict[str, Any],
    fit_history: Optional[list],
    registry,
    active_skills: list,
) -> Dict[str, Any]:
    """Re-select skills from fit evidence, propose new hypotheses, and re-rank.

    Returns ``{hypotheses, active_skills, changed, n_new, added_skills}``.
    """
    observations = _format_observations(
        latest_fit, analysis, boundary_hits, fit_history, bic
    )
    try:
        reselected = select_skills(
            state.get("sample_description") or "",
            parsed_sample=state.get("parsed_sample"),
            registry=registry,
            extra_context=observations,
            states=state.get("states"),
        )
    except Exception as e:  # never let re-selection break evaluation
        logger.warning("[EVALUATION] Skill re-selection failed: %s", e)
        reselected = []
    # Union only — never drop a skill that intake activated.
    new_active_skills = sorted(set(active_skills) | set(reselected))
    skill_context = load_skill_context(new_active_skills, registry)

    revision = propose_hypothesis_revision_with_llm(
        state=state,
        latest_fit=latest_fit,
        hypotheses=hypotheses,
        bic=bic,
        boundary_hits=boundary_hits,
        analysis=analysis,
        fit_history=fit_history,
        skill_context=skill_context,
    )

    prior_ids = {h.get("id") for h in hypotheses}
    merged = merge_structural_hypotheses(
        prior=hypotheses,
        llm_returned=revision["new_hypotheses"],
        allow_new=True,
        current_iteration=iteration,
        default_origin="evaluation",
    )
    new_ids = [h["id"] for h in merged if h.get("id") not in prior_ids]
    ranked_ids = _resolve_ranking(revision["ranking"], new_ids)
    reranked = rerank_hypotheses(merged, ranked_ids)

    added_skills = sorted(set(new_active_skills) - set(active_skills))
    order_changed = [h.get("id") for h in reranked] != [h.get("id") for h in hypotheses]
    changed = bool(new_ids) or bool(added_skills) or order_changed
    return {
        "hypotheses": reranked,
        "active_skills": new_active_skills,
        "changed": changed,
        "n_new": len(new_ids),
        "added_skills": added_skills,
    }


def _as_text_list(raw: Any) -> list:
    """Normalize one LLM-supplied list field to a list of non-empty strings.

    ``issues`` / ``suggestions`` / ``physical_concerns`` are model output, not a
    validated schema, but every consumer treats them as lists of text — the node
    appends to ``issues`` on every sub-floor fit and every boundary hit. A bare
    string raised ``AttributeError: 'str' object has no attribute 'append'`` there
    and, since the runner does not wrap the node, ended the whole run.

    A single string becomes a one-element list rather than being dropped: an
    evaluator reporting one issue as a bare string is still reporting a real
    finding. Only ``None``/absent and whitespace mean "nothing to report".
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, (list, tuple)):
        out = []
        for item in raw:
            if item is None:
                continue
            text = (item if isinstance(item, str) else str(item)).strip()
            if text:
                out.append(text)
        return out
    return [str(raw)]


def analyze_fit_quality_with_llm(
    fit_result: FitResult,
    sample_description: Optional[str],
    hypothesis: Optional[str],
    features: Optional[Dict],
    chi2_max: float = 5.0,
    chi2_min: Optional[float] = None,
    user_criteria: str = "",
    residual_analysis: Optional[Dict] = None,
    boundary_hits: Optional[list] = None,
    bic: float | None = None,
    best_bic: float | None = None,
    n_params: int = 0,
    n_layers: int = 0,
    skill_context: str = "",
    per_file_results: Optional[list] = None,
    fit_history: Optional[list] = None,
    structural_hypotheses: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Use LLM to analyze fit quality in context.

    *chi2_min* is the acceptance floor (``0`` = none). ``None`` means "resolve it
    from the environment", for callers outside the workflow — ``aure evaluate``
    has no run state to pin it from, and the evaluator still has to be told what
    a χ² below the floor implies about the error model.

    Returns:
        Dictionary with acceptable, issues, suggestions, etc.
    """
    if chi2_min is None:
        chi2_min = _get_chi2_min(chi2_max=chi2_max)

    llm = get_llm(temperature=0)

    prompt = format_fit_evaluation_prompt(
        sample_description=sample_description or "",
        hypothesis=hypothesis,
        chi_squared=fit_result.get("chi_squared", float("inf")),
        method=fit_result.get("method", "unknown"),
        converged=fit_result.get("converged", False),
        parameters=fit_result.get("parameters", {}),
        features=features or {},
        chi2_max=chi2_max,
        chi2_min=chi2_min,
        user_criteria=user_criteria,
        boundary_hits=boundary_hits,
        residual_analysis=residual_analysis,
        bic=bic,
        best_bic=best_bic,
        n_params=n_params,
        n_layers=n_layers,
        skill_context=skill_context,
        per_file_results=per_file_results,
        fit_history=fit_history,
        structural_hypotheses=structural_hypotheses,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content

    # Extract JSON from response
    json_match = re.search(r"\{[\s\S]*\}", content)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return {
                "acceptable": result.get("acceptable", False),
                "quality_assessment": result.get("quality_assessment", "unknown"),
                "issues": _as_text_list(result.get("issues")),
                "suggestions": _as_text_list(result.get("suggestions")),
                "physical_concerns": _as_text_list(result.get("physical_concerns")),
                "hypothesis_addressed": result.get("hypothesis_addressed", ""),
                "needs_user_guidance": result.get("needs_user_guidance", False),
                "next_action": result.get("next_action", "parameter_tweak"),
                "proposed_hypothesis_id": result.get("proposed_hypothesis_id"),
                "chi_squared": fit_result.get("chi_squared", float("inf")),
                "_used_fallback": False,
            }
        except json.JSONDecodeError:
            logger.warning("[EVALUATION] Failed to parse LLM JSON response")

    # Fallback if LLM response can't be parsed
    fallback = _simple_evaluation(fit_result, chi2_max=chi2_max, chi2_min=chi2_min)
    fallback["_used_fallback"] = True
    return fallback


def _simple_evaluation(
    fit_result: FitResult,
    chi2_max: Optional[float] = None,
    chi2_min: Optional[float] = None,
) -> Dict[str, Any]:
    """Simple heuristic evaluation as fallback.

    Used when the evaluator LLM's response carries no parseable JSON. It
    **reports** — it does not decide acceptance.

    ``acceptable`` is always False here so ``_clamp_acceptance_to_chi2`` stays the
    single acceptance point and all its stand-down guards apply. Asserting
    ``chi2 <= chi2_max`` here short-circuited the clamp (it early-returns on an
    already-acceptable verdict), so a fit with no SLD profile to check, or an
    unfitted contrast under a passing aggregate, completed on χ² alone — on a path
    with no LLM judgement to defer to either. The clamp reaches the same decision
    once its guards are satisfied.

    *chi2_max* / *chi2_min* are the caller's effective window (state-pinned for a
    resumed run), falling back to the environment for callers with no state.
    """
    chi2 = fit_result.get("chi_squared", float("inf"))
    chi2_max = _get_chi2_max() if chi2_max is None else chi2_max
    # The overfitting flag below reads the *configured* floor rather than the literal
    # 0.5 it used to carry, so this heuristic and the clamp agree on where "χ² too
    # small to be a pass" starts.
    chi2_min = _get_chi2_min(chi2_max=chi2_max) if chi2_min is None else chi2_min

    issues = []
    suggestions = []

    if chi2 > 10:
        issues.append(f"Poor fit quality (χ² = {chi2:.1f})")
        suggestions.append("Consider modifying model structure")
    elif chi2 > chi2_max:
        issues.append(f"Marginal fit quality (χ² = {chi2:.1f}, threshold = {chi2_max})")
        suggestions.append("Try refining parameter bounds")
    elif chi2_min > 0 and chi2 < chi2_min:
        issues.append(f"Possible overfitting (χ² = {chi2:.2f})")

    return {
        "acceptable": False,
        "quality_assessment": "good" if chi2 < chi2_max else "poor",
        "issues": issues,
        "suggestions": suggestions,
        "physical_concerns": [],
        "hypothesis_addressed": "",
        "needs_user_guidance": False,
        "chi_squared": chi2,
    }


def _format_success(
    fit_result: FitResult,
    analysis: Dict,
    *,
    chi2_max: Optional[float] = None,
    chi2_min: Optional[float] = None,
) -> str:
    """Format success message.

    *chi2_max* / *chi2_min* are passed in rather than re-read from the environment
    so the reported window is the one the run actually applied (a resumed run keeps
    its original thresholds).
    """
    chi2 = fit_result["chi_squared"]
    lines = ["## ✓ Fit Successful!"]
    lines.append("")
    lines.append(f"**Final χ² = {chi2:.2f}**")
    lines.append("")

    if analysis.get("_chi2_clamped"):
        threshold = _get_chi2_max() if chi2_max is None else chi2_max
        lines.append(
            f"The run stopped here because χ² met the acceptance threshold "
            f"(χ² ≤ {threshold:.2f}). The notes below were raised during "
            f"evaluation but were not acted on."
        )
        lines.append("")

    # A χ² this low reaches this message only because the evaluator accepted it on
    # its own judgement — the clamp stands down below the floor. Said plainly here
    # because the headline number invites the opposite reading.
    floor = _get_chi2_min(chi2_max=chi2_max) if chi2_min is None else chi2_min
    if floor > 0 and np.isfinite(chi2) and chi2 < floor:
        lines.append(
            f"**Note:** χ² = {chi2:.4g} is *below* the acceptance floor "
            f"(χ² ≥ {floor:.2f}), so this is not a better fit than χ² ≈ 1 — the "
            f"residuals are smaller than the quoted uncertainties. That usually "
            f"means the dR column is overestimated or the model has enough free "
            f"parameters to absorb the noise. The evaluator accepted it anyway; "
            f"check the uncertainties and the parameter count."
        )
        lines.append("")

    if fit_result["parameters"]:
        lines.append("### Best-fit Structure:")
        lines.append("")

        # Group parameters by layer
        layers = {}
        for param, value in fit_result["parameters"].items():
            # Parse layer index from parameter name
            if "[" in param:
                idx = param.split("[")[1].split("]")[0]
                if idx not in layers:
                    layers[idx] = {}

                if "thickness" in param.lower():
                    layers[idx]["thickness"] = value
                elif "rho" in param.lower():
                    layers[idx]["sld"] = value
                elif "interface" in param.lower():
                    layers[idx]["roughness"] = value

        for idx in sorted(layers.keys()):
            layer = layers[idx]
            t = layer.get("thickness", "?")
            sld = layer.get("sld", "?")
            r = layer.get("roughness", "?")
            lines.append(
                f"- Layer {idx}: d = {t:.1f} Å, SLD = {sld:.2f}, σ = {r:.1f} Å"
            )

    if analysis.get("issues"):
        lines.append("")
        lines.append("### Notes:")
        for issue in analysis["issues"]:
            lines.append(f"- {issue}")

    return "\n".join(lines)


def _format_evaluation(
    fit_result: FitResult, analysis: Dict, *, iteration: int = 0
) -> str:
    """Format evaluation with issues and suggestions."""
    header = (
        f"## Fit Evaluation (iteration {iteration})"
        if iteration
        else "## Fit Evaluation"
    )
    lines = [header]
    lines.append("")
    lines.append(f"**χ² = {fit_result['chi_squared']:.2f}**")

    # Separate boundary-hit issues from other issues so they can be
    # collapsed into a single summary line.
    boundary_issues = []
    other_issues = []
    for issue in analysis.get("issues", []):
        if "bound" in issue.lower() and "auto-expanded" in issue.lower():
            boundary_issues.append(issue)
        else:
            other_issues.append(issue)

    if other_issues or boundary_issues:
        lines.append("")
        lines.append("### Issues Identified:")
        for issue in other_issues:
            lines.append(f"- ⚠️ {issue}")
        if boundary_issues:
            lines.append(
                f"- ⚠️ {len(boundary_issues)} parameter(s) hit range bounds "
                f"(auto-expanded)"
            )

    if analysis["suggestions"]:
        lines.append("")
        lines.append("### Suggested Improvements:")
        for i, suggestion in enumerate(analysis["suggestions"], 1):
            lines.append(f"{i}. {suggestion}")

    if analysis.get("needs_guidance"):
        lines.append("")
        lines.append(
            "*Would you like me to try one of these improvements, or do you have other guidance?*"
        )
    else:
        lines.append("")
        lines.append("*Attempting automatic refinement...*")

    return "\n".join(lines)


def _ordered_slds_for_artifacts(
    model: dict, parameters: dict, state_name: Optional[str] = None
) -> list:
    """Ordered SLD sequence ``[substrate, layers..., ambient]`` for artifact
    detection, preferring fitted values from ``parameters`` (refl1d names each
    material's SLD parameter ``"<name> rho"``) and falling back to model seeds.

    **The order is the physical stack order, and it matters.** A ModelDefinition
    lists its layers substrate-first (``prompts``: "Layers are listed from
    substrate to ambient"), and ``modeling`` emits the refl1d stack to match:
    ``substrate | material1 | ... | ambient`` in normal geometry, and the same
    adjacency written in beam order — ``ambient | materialN | ... | substrate``,
    with the layers *reversed* — in back reflection. Either way the substrate
    neighbours ``layers[0]`` and the ambient neighbours ``layers[-1]``.

    This used to emit ``[ambient, layers..., substrate]``, which is neither
    order: it kept the layers substrate-first while putting the terminals in
    beam order, so both ends were spliced onto the wrong neighbours. The
    interior is untouched, but the two media that decide whether the first and
    last layers are turning points are swapped.

    ``detect_profile_artifacts`` is direction-agnostic but not order-agnostic:
    it derives the set of *legitimate* extrema from exactly this sequence. With
    the terminals swapped it expected the wrong turning points and reported the
    real ones as excursions. On the 2026-08-17 cu_film sweep that vetoed every
    candidate in 38 of 51 runs — and since ``finalize`` ranks by that veto, it
    decided which model each run reported. Judged against this order the same
    profiles agree with the validation harness's independent detector.

    *state_name* is tried as a prefix first. ``build_states_problem`` renames
    untied parameters ``"<state> <material> rho"`` — the ambient SLD is untied by
    default in a contrast series — so without it a co-refinement silently falls
    back to the template seed and the profile gets judged against the wrong media.
    """
    if not isinstance(model, dict):
        return []
    params = parameters or {}

    def _sld(info):
        if not isinstance(info, dict):
            return None
        name = info.get("name")
        if name:
            for key in (
                f"{state_name} {name} rho" if state_name else None,
                f"{name} rho",
            ):
                if key and key in params:
                    return float(params[key])
        val = info.get("sld")
        return None if val is None else float(val)

    seq = []
    sub = _sld(model.get("substrate"))
    if sub is not None:
        seq.append(sub)
    for layer in model.get("layers") or []:
        v = _sld(layer)
        if v is not None:
            seq.append(v)
    amb = _sld(model.get("ambient"))
    if amb is not None:
        seq.append(amb)
    return seq


def _has_points(seq) -> bool:
    """True for a non-empty sequence. Length, not truthiness — a numpy array
    raises on ``bool()`` and these values arrive from JSON as lists but from
    in-memory callers as arrays."""
    try:
        return seq is not None and len(seq) > 0
    except TypeError:
        return False


def _profile_contexts(fit_result: FitResult, model) -> Optional[list]:
    """One ``(state_name, effective_model, z, rho)`` per state to be checked.

    Returns ``None`` when the model's shape cannot be resolved — every error path
    fails *closed*, because this function's whole job is to answer "what could this
    profile not cover", and answering "one state" on a resolution failure would
    hand the clamp a clean bill of health it has not earned.

    Single-state runs get one context built from the fit's own top-level profile,
    which is what refl1d exports; each state of a co-refinement gets the profile
    ``fitting`` read back from ``export_dir/state_<name>/`` and its own effective
    definition, since per-state ``ambient``/``layers`` overrides mean the
    model-level template describes no state in particular.
    """
    if not isinstance(model, dict):
        return None

    try:
        from ..state import iter_states
        from .model_builder import _state_overrides
    except Exception as e:  # pragma: no cover - import guard
        logger.debug(f"[EVALUATION] State resolution unavailable: {e}")
        return None

    declared = model.get("states")
    if not declared:
        return [(None, model, fit_result.get("sld_z"), fit_result.get("sld_rho"))]
    if not (isinstance(declared, list) and all(isinstance(s, dict) for s in declared)):
        logger.info(
            "[EVALUATION] `states` is not a list of mappings, so per-state profiles "
            "cannot be resolved — treating the profile as unverified"
        )
        return None

    try:
        states = iter_states(model)
    except Exception as e:
        logger.debug(f"[EVALUATION] iter_states failed: {e}")
        return None

    per_file = fit_result.get("per_file_results") or []
    contexts = []
    for i, st in enumerate(states):
        name = st.get("name") or f"state{i}"
        try:
            eff = _state_overrides(model, st)
        except Exception as e:
            logger.debug(f"[EVALUATION] _state_overrides failed for {name}: {e}")
            return None
        z = rho = None
        for pf in per_file:
            if (
                isinstance(pf, dict)
                and pf.get("state") == name
                and _has_points(pf.get("sld_z"))
            ):
                z, rho = pf.get("sld_z"), pf.get("sld_rho")
                break
        # states[0] is the model refl1d exported at the top level, so fall back to
        # it — that keeps a one-entry `states:` block behaving like a plain run.
        if not _has_points(z) and i == 0:
            z, rho = fit_result.get("sld_z"), fit_result.get("sld_rho")
        contexts.append((name, eff, z, rho))
    return contexts


def _detect_profile_artifacts_into(
    analysis: dict, fit_result: FitResult, model
) -> None:
    """Run the SLD-profile artifact detector and fold results into ``analysis``.

    A genuine non-physical excursion becomes an ``issue`` (so the workflow loops
    back to refinement) plus a targeted ``suggestion`` that offers both remedies
    — tie the roughness (keep a layer interpretation) or accept the diffuse
    transition as a profile parametrization. The extremum-count note and the
    σ/thickness ratios are added to ``physical_concerns`` only.

    Every state of a co-refinement is checked, each against its *own* effective
    media — per-state ``ambient``/``layers`` overrides mean the model-level
    template describes no state in particular, so judging one state's profile
    against it yields both false positives and false negatives.

    Two markers are left for the χ² accept clamp:

    * ``_profile_artifact`` — an excursion was found, in any state. A veto, and the
      issue names the state.
    * ``_profile_checked`` — a *positive* statement that this fit's profile was
      evaluated and the answer can be relied on. Set only when **every** state
      reported a profile the detector could evaluate. Left unset on every path that
      could not reach one: no exported profile (``sld_z``/``sld_rho`` are written
      only when the run has an output directory, so library and MCP runs have
      none), fewer than two resolvable media, a detector that returned
      ``checked=False``, an unresolvable model shape, or a co-refinement where any
      one state's profile is missing. Absent means "no evidence either way", which
      the clamp treats as unsafe — it stands down and the evaluator's verdict
      decides, as it did before the clamp existed.
    """
    try:
        from ..tools.feature_tools import (
            detect_profile_artifacts,
            check_roughness_thickness_ratios,
        )
    except Exception as e:  # pragma: no cover - import guard
        logger.debug(f"[EVALUATION] Artifact detector import failed: {e}")
        return

    analysis.setdefault("physical_concerns", [])
    analysis.setdefault("issues", [])
    analysis.setdefault("suggestions", [])

    contexts = _profile_contexts(fit_result, model)
    if contexts is None:
        logger.info(
            "[EVALUATION] Could not resolve the model's states, so the SLD profile "
            "cannot be attributed to one — treating the profile as unverified"
        )
        return

    multi = len(contexts) > 1
    params = fit_result.get("parameters")
    all_checked = True
    excursion = None  # (state_name, where, note) for the first one found
    eff_models = []

    for name, eff_model, z, rho in contexts:
        tag = f"[{name}] " if multi else ""
        eff_models.append((name, eff_model))
        if not (_has_points(z) and _has_points(rho)):
            all_checked = False
            logger.info(
                "[EVALUATION] %sThe fit carries no exported SLD profile (refl1d "
                "writes one only when the run has an output directory) — treating "
                "the profile as unverified",
                tag,
            )
            continue
        # Two orientation conventions meet here, and they do not agree:
        #
        #   * a ModelDefinition always lists its layers substrate-first, so
        #     `media` runs substrate -> ambient;
        #   * `z`/`rho` come from refl1d, which renders the stack in beam order
        #     — ambient -> substrate under back reflection.
        #
        # That mismatch is harmless *here* and only here: `detect_profile_artifacts`
        # compares extremum values against the turning-point values of `media`,
        # and a sequence's turning points are unchanged by reversing it, so
        # neither array needs flipping to match the other. What the detector
        # cannot survive is a sequence that is in *no* consistent order — which
        # is what `_ordered_slds_for_artifacts` used to return. Any future use of
        # this profile that reads position rather than value (a depth, a
        # centroid, a per-layer attribution) must orient it explicitly first.
        media = _ordered_slds_for_artifacts(eff_model, params, state_name=name)
        if len(media) < 2:
            all_checked = False
            logger.info(
                "[EVALUATION] %sFewer than two media resolvable (%d), so there is "
                "no SLD range to test the profile against — treating the profile "
                "as unverified",
                tag,
                len(media),
            )
            continue

        result = detect_profile_artifacts(z, rho, media)
        if not result.get("checked"):
            all_checked = False
            logger.info(
                "[EVALUATION] %sArtifact detector declined to check the profile "
                "(too few points, mismatched z/rho lengths, a non-finite sample, "
                "or a zero SLD span across the media) — treating the profile as "
                "unverified",
                tag,
            )
        for note in result.get("notes", []):
            analysis["physical_concerns"].append(f"{tag}{note}")
        if result.get("has_artifact") and excursion is None:
            exc = result["excursions"][0]
            where = ", ".join(
                f"{e['kind']} SLD {e['sld']:.2f} at z={e['z']:.0f} Å"
                for e in result["excursions"][:3]
            )
            excursion = (name if multi else None, where, exc["note"])

    if all_checked and excursion is None:
        analysis["_profile_checked"] = True

    if excursion is not None:
        state_name, where, note = excursion
        scope = f" in state '{state_name}'" if state_name else ""
        analysis["issues"].append(
            f"Non-physical SLD-profile excursion{scope} ({where}): "
            f"{note}. The reflectivity χ² does not see this."
        )
        # Veto acceptance: a physically impossible profile must not be accepted
        # on the strength of χ² alone. Forcing acceptable=False routes the loop
        # back to refinement so the excursion is resolved.
        if analysis.get("acceptable"):
            logger.info(
                "[EVALUATION] Overriding acceptable=True: SLD-profile artifact present"
            )
        analysis["acceptable"] = False
        analysis["_profile_artifact"] = True
        analysis["suggestions"].append(
            "Resolve the profile excursion: either constrain the offending "
            "interface roughness as a fraction of its layer thickness "
            "(roughness_tie with fraction ≤ 0.5, keeping a discrete-layer "
            "interpretation), or, if the diffuse transition is intended, keep "
            "the roughness free and treat those slabs as a profile "
            "parametrization (do not report them as distinct layers)."
        )
        logger.warning(f"[EVALUATION] SLD-profile artifact{scope}: {where}")

    # σ/thickness reads each state's own layers, since a state may override them.
    # Identical notes across states collapse to one unscoped line.
    seen = set()
    for name, eff_model in eff_models:
        tag = f"[{name}] " if multi else ""
        for r in check_roughness_thickness_ratios(eff_model):
            note = (
                f"roughness of '{r['layer']}' is {r['ratio']:.2f}× its thickness "
                f"(σ={r['roughness']:.1f} Å, t={r['thickness']:.1f} Å) — verify this "
                f"is an intended profile parametrization, not a spurious interface"
            )
            if note in seen:
                continue
            seen.add(note)
            analysis["physical_concerns"].append(f"{tag}{note}")


def _check_boundary_hits(
    fit_result: FitResult,
    tolerance: float = 0.01,
    sigma: float = 2.0,
    max_width_fraction: float = 0.75,
) -> list:
    """Check if any fitted parameters are at or near their range boundaries.

    Two ways a parameter can be pinned by its range:

    * its **value** is within *tolerance* (relative) of a bound edge; or
    * its **uncertainty interval** reaches the edge — ``value ± sigma·dx``
      crosses it — even though the point estimate does not.

    The second case is invisible to a point-estimate test and is common after
    a dream run, where the posterior is skewed towards the bound it is pressed
    against. A real example: ``CuOx interface`` fitted at 5.11 in ``[5, 11]``
    is 1.8 % from its floor, so a 1 % test passes it, while its posterior runs
    down to 5.03. The range is constraining the answer and the evaluator never
    hears about it.

    Uncertainty hits are only reported for parameters the data actually
    constrain. When ``2·sigma·dx`` covers more than *max_width_fraction* of
    the range the parameter is not pinned against an edge, it is
    unconstrained — and widening its bounds (see :func:`_expand_model_bounds`)
    makes the next fit worse rather than better, iteration after iteration.
    Those are skipped deliberately.

    The 0.75 default is calibrated rather than chosen for roundness: the
    ``CuOx interface`` case above has ``dx = 0.906`` on a span of 6, so its
    interval covers 60 % of the range while still being a genuine pin. A
    tighter guard rejects the very case this exists to catch.

    ``dx`` is unavailable for optimizers that do not estimate it, in which case
    this degrades to the point-estimate test alone.

    Returns a list of dicts:
    ``{name, value, bound_hit, bound_value, detected_by[, uncertainty]}``,
    where ``detected_by`` is ``"value"`` or ``"uncertainty"``.
    """
    params = fit_result.get("parameters") or {}
    bounds = fit_result.get("bounds") or {}
    uncertainties = fit_result.get("uncertainties") or {}
    hits = []
    for name, value in params.items():
        b = bounds.get(name)
        if not b or len(b) != 2:
            continue
        lo, hi = b
        span = hi - lo
        if span <= 0:
            continue
        if abs(value - lo) <= tolerance * span:
            hits.append(
                {
                    "name": name,
                    "value": value,
                    "bound_hit": "lower",
                    "bound_value": lo,
                    "detected_by": "value",
                }
            )
            continue
        if abs(value - hi) <= tolerance * span:
            hits.append(
                {
                    "name": name,
                    "value": value,
                    "bound_hit": "upper",
                    "bound_value": hi,
                    "detected_by": "value",
                }
            )
            continue

        dx = uncertainties.get(name)
        if not isinstance(dx, (int, float)) or dx <= 0 or np.isnan(dx):
            continue
        reach = sigma * dx
        if 2 * reach >= max_width_fraction * span:
            continue  # unconstrained, not pinned
        if value - reach <= lo:
            side, edge = "lower", lo
        elif value + reach >= hi:
            side, edge = "upper", hi
        else:
            continue
        hits.append(
            {
                "name": name,
                "value": value,
                "bound_hit": side,
                "bound_value": edge,
                "detected_by": "uncertainty",
                "uncertainty": dx,
            }
        )
    return hits


def _expand_model_bounds(model: dict, boundary_hits: list) -> dict:
    """Expand bounds in a ModelDefinition dict for parameters that hit a boundary.

    For each hit, the constrained side of the range is expanded by 50%.
    """
    import copy

    model = copy.deepcopy(model)

    # Build a lookup: param_name -> (layer_dict, key_lo, key_hi)
    _RANGE_KEYS = {
        "thickness": ("thickness_min", "thickness_max"),
        "rho": ("sld_min", "sld_max"),
        "irho": ("sld_min", "sld_max"),
        "interface": ("roughness_min", "roughness_max"),
    }

    for bh in boundary_hits:
        name = bh["name"]
        side = bh["bound_hit"]

        # Match parameter name to a model layer field.
        # refl1d names look like "copper thickness", "silicon interface", etc.
        matched = False
        for layer in model.get("layers", []):
            layer_name = layer.get("name", "").lower()
            for field, (lo_key, hi_key) in _RANGE_KEYS.items():
                if layer_name in name.lower() and field in name.lower():
                    if side == "lower" and lo_key and lo_key in layer:
                        lo_val = layer[lo_key]
                        hi_val = layer.get(hi_key, lo_val)
                        spread = hi_val - lo_val
                        layer[lo_key] = lo_val - spread * 0.5
                    elif side == "upper" and hi_key and hi_key in layer:
                        lo_val = layer.get(lo_key, 0)
                        hi_val = layer[hi_key]
                        spread = hi_val - lo_val
                        layer[hi_key] = hi_val + spread * 0.5
                    matched = True
                    break
            if matched:
                break

        # Also handle substrate roughness
        if not matched and "interface" in name.lower():
            substrate = model.get("substrate", {})
            sub_name = substrate.get("name", "").lower()
            if sub_name in name.lower():
                if side == "upper" and "roughness_max" in substrate:
                    r_max = substrate["roughness_max"]
                    substrate["roughness_max"] = r_max * 1.5

        # Handle intensity
        if not matched and "intensity" in name.lower():
            intensity = model.get("intensity", {})
            if side == "lower" and "min" in intensity:
                spread = intensity.get("max", 1.1) - intensity["min"]
                intensity["min"] = intensity["min"] - spread * 0.5
            elif side == "upper" and "max" in intensity:
                spread = intensity["max"] - intensity.get("min", 0.7)
                intensity["max"] = intensity["max"] + spread * 0.5

        # Handle sample_broadening
        if not matched and "sample_broadening" in name.lower():
            sb = model.get("sample_broadening", {})
            if sb.get("enabled"):
                if side == "upper" and "max" in sb:
                    spread = sb["max"] - sb.get("min", 0.0)
                    sb["max"] = sb["max"] + spread * 0.5

        # Handle theta_offset
        if not matched and "theta_offset" in name.lower():
            to = model.get("theta_offset", {})
            if to.get("enabled"):
                if side == "lower" and "min" in to:
                    spread = to.get("max", 0.02) - to["min"]
                    to["min"] = to["min"] - spread * 0.5
                elif side == "upper" and "max" in to:
                    spread = to["max"] - to.get("min", -0.02)
                    to["max"] = to["max"] + spread * 0.5

    return model
