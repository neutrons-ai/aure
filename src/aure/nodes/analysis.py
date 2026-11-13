"""
ANALYSIS node: Extract physics features from reflectivity data.

This node analyzes the loaded data to extract:
- Critical edges (Qc values → SLD estimates)
- Oscillation periods (→ layer thicknesses)
- High-Q decay (→ roughness estimates)
- Layer count estimation

The critical edge is reported as a hint (see ``annotate_critical_edges``): it
is never used here to set or correct an SLD in the model.
"""

from typing import Dict, Any
import numpy as np

from ..state import ReflectivityState, ExtractedFeatures, Message
from ..tools.feature_tools import extract_all_features, format_features_for_llm


def annotate_critical_edges(parsed: dict | None, features: dict | None) -> None:
    """Name the incident medium on each critical edge, in place.

    A critical edge says some medium the beam reaches has a higher SLD than the
    one it comes from. Which medium the beam comes from is the geometry: the
    substrate in back reflection, the ambient otherwise. That reference is all
    this adds, so :func:`format_critical_edge_line` can state the contrast
    against the right medium.

    Deliberately no more than that. The edge does not say which medium carries
    the contrast (the far one, or a thick layer on the way), and its position is
    approximate, so it is fed to the hypothesis prompts as a hint and never used
    to set or correct an SLD here. An earlier version turned it into an "implied
    ambient SLD" and rewrote a solvent stated as protonated; on a copper film in
    protonated THF it reported a deuterated ambient near 7.4, because the 50 nm
    copper sets that edge.
    """
    if not isinstance(parsed, dict) or not isinstance(features, dict):
        return
    back = bool(parsed.get("back_reflection"))
    medium = parsed.get("substrate" if back else "ambient") or {}
    sld = medium.get("sld")
    if not medium.get("name") or not isinstance(sld, (int, float)):
        return
    for edge in features.get("critical_edges") or []:
        if isinstance(edge, dict):
            edge["incident_medium"] = {
                "name": str(medium["name"]),
                "sld": float(sld),
                "side": "substrate" if back else "ambient",
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
        # Name the medium the critical edge's contrast is measured against,
        # before the features are rendered anywhere.
        annotate_critical_edges(state.get("parsed_sample"), features)
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
                f"estimated (~{estimated_total:.0f} Å). Will use your described "
                f"value as starting point and widen the range to include both."
            )

    return " ".join(issues) if issues else ""
