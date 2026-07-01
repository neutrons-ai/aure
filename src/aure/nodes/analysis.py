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

# Physically-admissible SLD window for a real ambient medium (×10⁻⁶ Å⁻²).
_MIN_REAL_SLD = -0.7  # just below H2O (-0.56)
_MAX_REAL_SLD = 8.0

# Ambient names that are gases/vacuum (not a solvent the deuteration check
# applies to). Everything else with a name is treated as a liquid medium.
_NON_LIQUID_AMBIENTS = frozenset(
    {"", "air", "vacuum", "gas", "none", "n2", "nitrogen", "argon", "ar", "he"}
)


def _is_liquid_ambient(ambient_name: str) -> bool:
    """True when *ambient_name* names a liquid medium (not gas/vacuum)."""
    return bool(ambient_name) and ambient_name not in _NON_LIQUID_AMBIENTS


def implied_ambient_sld_from_edge(
    substrate_sld: float,
    edge_contrast: float,
    stated_ambient_sld: float,
) -> dict:
    """Estimate the ambient SLD from a critical edge, given the substrate SLD.

    A critical edge sits at Qc where ``|ρ_ambient − ρ_substrate| = (Qc/4)²/π``
    — the total-reflection contrast between the two semi-infinite media.  A film
    of typical thickness only adds fringes *above* Qc, so it does not set the
    edge; the edge is fixed by the substrate/ambient contrast.  Hence
    ``ρ_ambient = ρ_substrate ± contrast``.  This is geometry-agnostic (front
    or back reflection give the same magnitude); the physical branch is chosen
    below.

    Returns ``{candidate_high, candidate_low, implied_ambient_sld,
    suggests_deuteration}``.  ``suggests_deuteration`` is True when neither
    branch matches the stated (H-form) ambient SLD but the high branch lands in
    the deuterated range — i.e. the edge can only be explained by an ambient SLD
    far above the stated value.
    """
    hi = substrate_sld + edge_contrast
    lo = substrate_sld - edge_contrast
    tol = 1.0
    matches_stated = (
        abs(lo - stated_ambient_sld) <= tol or abs(hi - stated_ambient_sld) <= tol
    )
    suggests = (not matches_stated) and (4.0 <= hi <= _MAX_REAL_SLD)
    return {
        "candidate_high": hi,
        "candidate_low": lo,
        "implied_ambient_sld": hi if suggests else None,
        "suggests_deuteration": suggests,
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
        # _analyze_ambient may annotate features["critical_edges"] in place with
        # the deterministic implied-ambient-SLD hint (features is the same object
        # stored in updates above), so the hint reaches downstream prompts.
        new_ambient, ambient_msg = _analyze_ambient(state["parsed_sample"], features)
        if new_ambient:
            # Known solvent: correct the parsed_sample ambient in place.
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
        elif ambient_msg:
            # Generic liquid: surface the deuteration hint (no auto-correction).
            updates["messages"].append(
                Message(role="assistant", content=ambient_msg, timestamp=None)
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


def _analyze_ambient(
    parsed: dict,
    features: ExtractedFeatures,
) -> Tuple[Optional[dict], str]:
    """
    Deterministic ambient-SLD check from the critical edge.

    Whenever the sample sits in a liquid, the critical edge — together with the
    known substrate SLD — pins the ambient SLD (see
    :func:`implied_ambient_sld_from_edge`).  When that implied SLD is far above
    the stated H-form value, the ambient is very likely deuterated even if the
    user never said so.  This is geometry-agnostic (no back-reflection gate) and
    works for generic electrolytes/solutions, not just the named solvents in
    ``_SOLVENT_VARIANTS``.

    Side effect: the top matching critical edge is annotated in place with
    ``implied_ambient_sld`` / ``candidate_high`` / ``candidate_low`` /
    ``substrate_sld`` / ``suggests_deuteration`` so the hint travels wherever
    critical edges are formatted (analysis display + evaluation/modeling
    prompts).

    Returns
    -------
    (correction, message)
        *correction* is a dict to substitute into ``parsed_sample["ambient"]``
        (with a constrained ``sld``/``sld_min``/``sld_max``), set only for a
        **known** solvent whose deuterated form the edge confidently implies;
        ``None`` for a generic liquid (hint only, left to the refinement loop).
        *message* is a human-readable explanation/hint (empty if nothing found).
    """
    ambient = parsed.get("ambient", {}) or {}
    ambient_name_raw = str(ambient.get("name", "")).strip()
    # Strip parenthetical qualifiers added by the LLM (e.g. "THF (protonated)").
    ambient_name = re.sub(r"\s*\(.*?\)", "", ambient_name_raw).lower().strip()
    if not _is_liquid_ambient(ambient_name):
        return None, ""

    stated_sld = ambient.get("sld", 0.0)
    if stated_sld is None:
        stated_sld = 0.0
    substrate = parsed.get("substrate", {}) or {}
    substrate_sld = substrate.get("sld", 2.07)
    if substrate_sld is None:
        substrate_sld = 2.07
    substrate_name = substrate.get("name", "substrate")

    edges = features.get("critical_edges", []) or []
    high_conf = [e for e in edges if e.get("confidence") in ("high", "medium")]
    if not high_conf:
        return None, ""

    edge = high_conf[0]
    contrast = edge.get("estimated_SLD", 0.0)
    est = implied_ambient_sld_from_edge(substrate_sld, contrast, stated_sld)
    if not est["suggests_deuteration"]:
        return None, ""

    implied = est["implied_ambient_sld"]
    qc = edge.get("Qc", 0.0)

    # Annotate the edge so the hint reaches the eval/modeling prompts.
    edge["implied_ambient_sld"] = float(implied)
    edge["candidate_high"] = float(est["candidate_high"])
    edge["candidate_low"] = float(est["candidate_low"])
    edge["substrate_sld"] = float(substrate_sld)
    edge["suggests_deuteration"] = True

    lo = max(_MIN_REAL_SLD, implied - 1.0)
    hi = min(_MAX_REAL_SLD, implied + 1.0)

    solvent = _SOLVENT_VARIANTS.get(ambient_name)
    if solvent is not None and abs(stated_sld - solvent["d_sld"]) >= 1.0:
        # Known solvent, stated as the H-form: correct to its deuterated form
        # and constrain the SLD near the tabulated D value.
        d_sld = solvent["d_sld"]
        correction = {
            "name": solvent["d_name"],
            "sld": d_sld,
            "sld_min": max(_MIN_REAL_SLD, d_sld - 1.0),
            "sld_max": min(_MAX_REAL_SLD, d_sld + 1.0),
        }
        msg = (
            f"Critical edge at Qc = {qc:.4f} Å⁻¹ (SLD contrast ~{contrast:.1f}) "
            f"with the {substrate_name} substrate (SLD {substrate_sld:.2f}) "
            f"implies an ambient SLD of ~{implied:.1f} × 10⁻⁶ Å⁻². This is "
            f"consistent with **{solvent['d_name']}** (SLD {d_sld}), not "
            f"protonated {ambient_name.upper()} (SLD {solvent['h_sld']}). "
            f"Setting ambient to {solvent['d_name']} and constraining its SLD "
            f"near {d_sld}."
        )
        return correction, msg

    # Generic liquid (electrolyte/solution/buffer/…): emit a strong hint but
    # leave the switch to the deuterated-ambient hypothesis in the refinement
    # loop. Constraining guidance is included so the modeler does not leave the
    # ambient SLD wandering across the full H–D range.
    msg = (
        f"**Deuteration hint:** the critical edge at Qc = {qc:.4f} Å⁻¹ "
        f"(SLD contrast ~{contrast:.1f} × 10⁻⁶ Å⁻²) with the {substrate_name} "
        f"substrate (SLD {substrate_sld:.2f}) implies an ambient SLD of "
        f"~{implied:.1f} × 10⁻⁶ Å⁻² — far above the stated H-form value "
        f"(~{stated_sld:.1f}). The ambient is very likely deuterated. Realize "
        f"the deuterated-ambient hypothesis and constrain the ambient SLD near "
        f"~{implied:.1f} (e.g. [{lo:.1f}, {hi:.1f}]), not the full H–D range."
    )
    return None, msg
