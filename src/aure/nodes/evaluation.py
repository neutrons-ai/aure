"""
EVALUATION node: Assess fit quality and identify issues.

This node uses an LLM to analyze the fit results and determine:
- Is the fit acceptable (χ² close to 1)?
- Are there systematic residuals indicating model problems?
- Are parameters physically reasonable?
- What refinements might improve the fit?
"""

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
from .prompts import format_fit_evaluation_prompt

logger = logging.getLogger(__name__)


def _get_chi2_max() -> float:
    """Return the χ² acceptance threshold from ``CHI2_MAX`` env var."""
    try:
        return float(os.environ.get("CHI2_MAX", "5.0"))
    except (TypeError, ValueError):
        return 5.0


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
    if residual_ratio and Q and chi2 > chi2_max:
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
        if best_chi2 is not None and best_model and chi2 > best_chi2 * 1.05:
            logger.warning(
                f"[EVALUATION] χ² regressed ({chi2:.3f} > best {best_chi2:.3f}) "
                f"— reverting to best model before refinement"
            )
            updates["current_model"] = best_model
            analysis["issues"].insert(
                0,
                f"Previous refinement made the fit worse (χ² went from "
                f"{best_chi2:.2f} to {chi2:.2f}). Reverting to the best model "
                f"and trying a different approach.",
            )

        logger.info("[EVALUATION] ✗ Fit not acceptable - proceeding to refinement")
        updates["messages"] = [
            Message(
                role="assistant",
                content=_format_evaluation(latest_fit, analysis),
                timestamp=None,
            )
        ]

    return updates


def analyze_fit_quality_with_llm(
    fit_result: FitResult,
    sample_description: Optional[str],
    hypothesis: Optional[str],
    features: Optional[Dict],
    chi2_max: float = 5.0,
    user_criteria: str = "",
    residual_analysis: Optional[Dict] = None,
    boundary_hits: Optional[list] = None,
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


def _format_evaluation(fit_result: FitResult, analysis: Dict) -> str:
    """Format evaluation with issues and suggestions."""
    lines = ["## Fit Evaluation"]
    lines.append("")
    lines.append(f"**χ² = {fit_result['chi_squared']:.2f}**")

    if analysis["issues"]:
        lines.append("")
        lines.append("### Issues Identified:")
        for issue in analysis["issues"]:
            lines.append(f"- ⚠️ {issue}")

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

    return model
