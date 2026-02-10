"""
ANALYSIS node: Extract physics features from reflectivity data.

This node analyzes the loaded data to extract:
- Critical edges (Qc values → SLD estimates)
- Oscillation periods (→ layer thicknesses)
- High-Q decay (→ roughness estimates)
- Layer count estimation
"""

from typing import Dict, Any
import numpy as np

from ..state import ReflectivityState, ExtractedFeatures, Message
from ..tools.feature_tools import extract_all_features, format_features_for_llm


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
        updates["messages"] = [Message(
            role="assistant",
            content=f"**Feature Analysis:**\n\n{summary}",
            timestamp=None
        )]
        
    except Exception as e:
        updates["error"] = f"Feature extraction failed: {str(e)}"
        updates["messages"] = [Message(
            role="system",
            content=f"Error during feature extraction: {str(e)}",
            timestamp=None
        )]
        return updates
    
    # ========== Cross-check with parsed sample ==========
    if state.get("parsed_sample") and features:
        discrepancies = _check_consistency(state["parsed_sample"], features)
        if discrepancies:
            updates["messages"].append(Message(
                role="assistant",
                content=f"**Note:** {discrepancies}",
                timestamp=None
            ))
    
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
        
        if described_total > 0 and abs(described_total - estimated_total) / described_total > 0.3:
            issues.append(
                f"Described total thickness (~{described_total:.0f} Å) differs from "
                f"estimated (~{estimated_total:.0f} Å). Will use feature-based estimate."
            )
    
    return " ".join(issues) if issues else ""
