"""
ANALYSIS node: Extract physics features from reflectivity data.

This node analyzes the loaded data to extract:
- Critical edges (Qc values → SLD estimates)
- Oscillation periods (→ layer thicknesses)
- High-Q decay (→ roughness estimates)
- Layer count estimation
- Ambient SLD validation (deuterated vs protonated solvents)
"""

from typing import Dict, Any, Optional, Tuple
import re
import numpy as np

from ..state import ReflectivityState, ExtractedFeatures, Message
from ..tools.feature_tools import extract_all_features, format_features_for_llm


# Known solvents: name → {h_sld, d_sld, d_name}
_SOLVENT_VARIANTS = {
    "thf": {"h_sld": 0.18, "d_sld": 6.35, "d_name": "d8-THF"},
    "dthf": {"h_sld": 0.18, "d_sld": 6.35, "d_name": "d8-THF"},
    "d-thf": {"h_sld": 0.18, "d_sld": 6.35, "d_name": "d8-THF"},
    "d8-thf": {"h_sld": 0.18, "d_sld": 6.35, "d_name": "d8-THF"},
    "h2o": {"h_sld": -0.56, "d_sld": 6.36, "d_name": "D2O"},
    "d2o": {"h_sld": -0.56, "d_sld": 6.36, "d_name": "D2O"},
    "water": {"h_sld": -0.56, "d_sld": 6.36, "d_name": "D2O"},
    "toluene": {"h_sld": 0.94, "d_sld": 5.66, "d_name": "d-toluene"},
    "d-toluene": {"h_sld": 0.94, "d_sld": 5.66, "d_name": "d-toluene"},
    "cyclohexane": {"h_sld": -0.28, "d_sld": 6.70, "d_name": "d-cyclohexane"},
    "d-cyclohexane": {"h_sld": -0.28, "d_sld": 6.70, "d_name": "d-cyclohexane"},
}


def analysis_node(state: ReflectivityState) -> Dict[str, Any]:
    """
    Extract physics features from reflectivity data.

    Args:
        state: Current workflow state

    Returns:
        State updates including extracted features
    """
    updates = {
        "current_node": "analysis",
        "messages": [],
    }

    # Convert lists back to arrays
    Q = np.array(state["Q"])
    R = np.array(state["R"])
    dR = np.array(state["dR"]) if state["dR"] else None

    # ========== Extract Features ==========
    try:
        features = extract_all_features(Q, R, dR)
        updates["extracted_features"] = features

        # Format for user display
        summary = format_features_for_llm(features)
        updates["messages"] = [
            Message(
                role="assistant",
                content=f"**Feature Analysis:**\n\n{summary}",
                timestamp=None,
            )
        ]

    except Exception as e:
        updates["error"] = f"Feature extraction failed: {str(e)}"
        updates["messages"] = [
            Message(
                role="system",
                content=f"Error during feature extraction: {str(e)}",
                timestamp=None,
            )
        ]
        return updates

    # ========== Cross-check with parsed sample ==========
    if state.get("parsed_sample") and features:
        discrepancies = _check_consistency(state["parsed_sample"], features)
        if discrepancies:
            updates["messages"].append(
                Message(
                    role="assistant",
                    content=f"**Note:** {discrepancies}",
                    timestamp=None,
                )
            )

        # ========== Ambient SLD validation ==========
        new_ambient, ambient_msg = _check_ambient_sld(state["parsed_sample"], features)
        if new_ambient:
            # Update the parsed_sample with corrected ambient
            updated_parsed = dict(state["parsed_sample"])
            updated_parsed["ambient"] = new_ambient
            updates["parsed_sample"] = updated_parsed
            updates["messages"].append(
                Message(
                    role="assistant",
                    content=f"**Ambient correction:** {ambient_msg}",
                    timestamp=None,
                )
            )

    return updates


def _check_consistency(parsed: dict, features: ExtractedFeatures) -> str:
    """Check if extracted features match user's description."""
    issues = []

    # Check layer count
    described_layers = len(parsed.get("layers", []))
    estimated_layers = features.get("estimated_n_layers", 0)

    if described_layers > 0 and estimated_layers > 0:
        if abs(described_layers - estimated_layers) > 1:
            issues.append(
                f"You described {described_layers} layer(s), but features suggest "
                f"~{estimated_layers} layer(s). We'll start with your description."
            )

    # Check thickness consistency
    if parsed.get("layers") and features.get("estimated_total_thickness"):
        described_total = sum(l.get("thickness", 0) for l in parsed["layers"])
        estimated_total = features["estimated_total_thickness"]

        if (
            described_total > 0
            and abs(described_total - estimated_total) / described_total > 0.3
        ):
            issues.append(
                f"Described total thickness (~{described_total:.0f} Å) differs from "
                f"estimated (~{estimated_total:.0f} Å). Will use feature-based estimate."
            )

    return " ".join(issues) if issues else ""


def _check_ambient_sld(
    parsed: dict,
    features: ExtractedFeatures,
) -> Tuple[Optional[dict], str]:
    """
    Check whether the parsed ambient SLD is consistent with the critical-edge
    data.  In back-reflection geometry through a known substrate, a critical
    edge that can only be explained by a deuterated solvent indicates that the
    user's description (e.g. "THF") should be interpreted as deuterated.

    Returns
    -------
    (new_ambient, message)
        *new_ambient* is a dict ``{"name": ..., "sld": ...}`` to substitute
        into ``parsed_sample["ambient"]``, or ``None`` if no change is needed.
        *message* is a human-readable explanation (empty string if no change).
    """
    if not parsed.get("back_reflection"):
        return None, ""

    substrate = parsed.get("substrate", {})
    substrate_sld = substrate.get("sld", 2.07)
    ambient = parsed.get("ambient", {})
    ambient_name = ambient.get("name", "").lower().strip()
    # Strip parenthetical qualifiers added by the LLM (e.g. "THF (protonated)")
    ambient_name = re.sub(r"\s*\(.*?\)", "", ambient_name).strip()
    ambient_sld = ambient.get("sld", 0.0)

    # Only check for known solvents
    solvent = _SOLVENT_VARIANTS.get(ambient_name)
    if solvent is None:
        return None, ""

    # If the parsed SLD is already close to the deuterated value, nothing to do
    if abs(ambient_sld - solvent["d_sld"]) < 1.0:
        return None, ""

    # In back-reflection, critical-edge SLD_est = ρ_layer − ρ_substrate.
    # For the ambient/substrate interface:
    #   d_contrast  = ρ_d_ambient − ρ_substrate
    # If any high-confidence critical edge matches this contrast → deuterated.
    d_sld = solvent["d_sld"]
    d_contrast = d_sld - substrate_sld  # e.g. 6.35 − 2.07 = 4.28 for d-THF/Si

    # If the deuterated solvent SLD ≤ substrate, no critical edge expected.
    if d_contrast <= 0:
        return None, ""

    edges = features.get("critical_edges", [])
    high_conf = [e for e in edges if e.get("confidence") in ("high", "medium")]

    for edge in high_conf:
        est_sld = edge.get("estimated_SLD", 0)
        # Match: estimated contrast is close to the deuterated contrast
        if abs(est_sld - d_contrast) < 1.5:
            abs_sld = est_sld + substrate_sld
            new_ambient = {"name": solvent["d_name"], "sld": d_sld}
            msg = (
                f"Critical edge at Qc = {edge.get('Qc', 0):.4f} Å⁻¹ implies an "
                f"SLD contrast of ~{est_sld:.1f} × 10⁻⁶ Å⁻² with the substrate "
                f"({substrate.get('name', 'substrate')}, SLD = {substrate_sld}), "
                f"giving an absolute SLD of ~{abs_sld:.1f}. "
                f"This is consistent with **{solvent['d_name']}** (SLD = {d_sld}), "
                f"not protonated {ambient_name.upper()} (SLD = {solvent['h_sld']}). "
                f"Updating ambient to {solvent['d_name']}."
            )
            return new_ambient, msg

    return None, ""
