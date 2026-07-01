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


def _get_chi2_max() -> float:
    """Return the χ² acceptance threshold from ``CHI2_MAX`` env var."""
    try:
        return float(os.environ.get("CHI2_MAX", "5.0"))
    except (TypeError, ValueError):
        return 5.0


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
    if boundary_hits:
        model = state.get("current_model")
        if isinstance(model, dict):
            model = _expand_model_bounds(model, boundary_hits)
            updates["current_model"] = model
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
    chi2_max = _get_chi2_max()
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

    # Update fit result with issues and suggestions
    # Inject boundary-hit issues into the analysis
    if boundary_hits:
        for bh in boundary_hits:
            analysis["issues"].append(
                f"Parameter '{bh['name']}' is at its {bh['bound_hit']} bound "
                f"({bh['value']:.4f} ≈ {bh['bound_value']:.4f}). "
                f"Range has been auto-expanded."
            )

    latest_fit["issues"] = analysis["issues"]
    latest_fit["suggestions"] = analysis["suggestions"]
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
        updates["messages"] = [
            Message(
                role="assistant",
                content=_format_success(latest_fit, analysis),
                timestamp=None,
            )
        ]
    else:
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
        elif best_chi2 is not None and chi2 <= best_chi2 * 1.01:
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


def analyze_fit_quality_with_llm(
    fit_result: FitResult,
    sample_description: Optional[str],
    hypothesis: Optional[str],
    features: Optional[Dict],
    chi2_max: float = 5.0,
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

    Returns:
        Dictionary with acceptable, issues, suggestions, etc.
    """
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
                "issues": result.get("issues", []),
                "suggestions": result.get("suggestions", []),
                "physical_concerns": result.get("physical_concerns", []),
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
    fallback = _simple_evaluation(fit_result)
    fallback["_used_fallback"] = True
    return fallback


def _simple_evaluation(fit_result: FitResult) -> Dict[str, Any]:
    """Simple heuristic evaluation as fallback."""
    chi2 = fit_result.get("chi_squared", float("inf"))
    chi2_max = _get_chi2_max()

    issues = []
    suggestions = []

    if chi2 > 10:
        issues.append(f"Poor fit quality (χ² = {chi2:.1f})")
        suggestions.append("Consider modifying model structure")
    elif chi2 > chi2_max:
        issues.append(f"Marginal fit quality (χ² = {chi2:.1f}, threshold = {chi2_max})")
        suggestions.append("Try refining parameter bounds")
    elif chi2 < 0.5:
        issues.append(f"Possible overfitting (χ² = {chi2:.2f})")

    return {
        "acceptable": chi2 <= chi2_max,
        "quality_assessment": "good" if chi2 < chi2_max else "poor",
        "issues": issues,
        "suggestions": suggestions,
        "physical_concerns": [],
        "hypothesis_addressed": "",
        "needs_user_guidance": False,
        "chi_squared": chi2,
    }


def _format_success(fit_result: FitResult, analysis: Dict) -> str:
    """Format success message."""
    lines = ["## ✓ Fit Successful!"]
    lines.append("")
    lines.append(f"**Final χ² = {fit_result['chi_squared']:.2f}**")
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


def _check_boundary_hits(
    fit_result: FitResult,
    tolerance: float = 0.01,
) -> list:
    """Check if any fitted parameters are at or near their range boundaries.

    A parameter is considered "at boundary" when its value is within
    *tolerance* (relative) of a bound edge.

    Returns a list of dicts: ``{name, value, bound_hit, bound_value}``.
    """
    params = fit_result.get("parameters") or {}
    bounds = fit_result.get("bounds") or {}
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
                {"name": name, "value": value, "bound_hit": "lower", "bound_value": lo}
            )
        elif abs(value - hi) <= tolerance * span:
            hits.append(
                {"name": name, "value": value, "bound_hit": "upper", "bound_value": hi}
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
