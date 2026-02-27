"""
MODELING node: Build refl1d model from features and sample description.

This node combines:
- Extracted physics features (Qc, thicknesses, roughness)
- Parsed sample description (materials, structure)
- Domain knowledge (SLD values, typical ranges)

To generate a refl1d model script with appropriate parameter bounds.

When called after evaluation (refinement loop), it uses the LLM to
regenerate the complete model script based on evaluation feedback.
"""

import logging
import re
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage

from ..state import ReflectivityState, Message
from ..database import get_sld
from ..llm import llm_available, get_llm
from ..config import format_user_constraints
from .prompts import format_model_refinement_prompt

logger = logging.getLogger(__name__)


def modeling_node(state: ReflectivityState) -> Dict[str, Any]:
    """
    Build or refine a refl1d model.

    On first call: builds an initial model from features and sample description.
    On subsequent calls (after evaluation): uses LLM to regenerate the model
    based on fit results and evaluation feedback.

    Args:
        state: Current workflow state

    Returns:
        State updates including model script
    """
    # Check if this is a refinement iteration (fit results already exist)
    fit_results = state.get("fit_results", [])
    current_model = state.get("current_model")

    if fit_results and current_model:
        return _refine_model(state)

    return _build_initial_model(state)


def _refine_model(state: ReflectivityState) -> Dict[str, Any]:
    """
    Refine an existing model using LLM based on evaluation feedback.

    The LLM receives the current model script, fit parameters, issues,
    and suggestions, and generates a complete improved model script.
    """
    updates = {
        "current_node": "modeling",
        "messages": [],
        "model_history": [],
    }

    iteration = state.get("iteration", 0)
    logger.info(
        f"[MODELING] Refinement iteration {iteration} - regenerating model with LLM"
    )

    fit_results = state.get("fit_results", [])
    latest_fit = fit_results[-1]
    current_model = state.get("current_model", "")

    issues = latest_fit.get("issues", [])
    suggestions = latest_fit.get("suggestions", [])
    logger.info(f"[MODELING] Issues to address: {issues}")
    logger.info(f"[MODELING] Suggestions: {suggestions}")

    if not llm_available():
        updates["error"] = (
            "LLM is required for model refinement. Please configure LLM_PROVIDER."
        )
        return updates

    try:
        user_constraints = format_user_constraints(state.get("user_config"))
        prompt = format_model_refinement_prompt(
            current_model=current_model,
            sample_description=state.get("sample_description", ""),
            fit_result=latest_fit,
            features=state.get("extracted_features") or {},
            user_constraints=user_constraints,
        )

        llm = get_llm(temperature=0)
        response = llm.invoke([HumanMessage(content=prompt)])
        new_model = response.content.strip()

        # Strip markdown code fences if present
        new_model = _strip_code_fences(new_model)

        # Fix common LLM mistake: var.material.rho → sample[i].material.rho
        new_model = _fix_sld_attr_access(new_model)

        # Validate the generated script has required components
        if not _validate_model_script(new_model):
            logger.warning(
                "[MODELING] LLM-generated script missing required components, keeping current model with widened bounds"
            )
            new_model = _widen_all_bounds(current_model)

        updates["current_model"] = new_model
        updates["model_history"] = [
            {
                "iteration": iteration,
                "description": f"Refined model (iteration {iteration})",
                "refinement_issues": issues,
                "refinement_suggestions": suggestions,
                "script": new_model,
            }
        ]

        # Format explanation message
        changes = _summarize_model_changes(current_model, new_model)
        updates["messages"] = [
            Message(
                role="assistant",
                content=_format_refinement_explanation(changes, issues, suggestions),
                timestamp=None,
            )
        ]

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
            updates["error"] = f"Model refinement failed: {str(e)[:200]}"
        logger.error(f"[MODELING] LLM refinement error: {e}")

    return updates


def _build_initial_model(state: ReflectivityState) -> Dict[str, Any]:
    """
    Build the initial refl1d model from features and sample description.
    """
    updates = {
        "current_node": "modeling",
        "messages": [],
        "model_history": [],
    }

    features = state.get("extracted_features") or {}
    parsed = state.get("parsed_sample") or {}

    # ========== Determine Model Structure ==========
    # Combine information from description and features
    substrate = _get_substrate(parsed, features)
    ambient = _get_ambient(parsed)
    layers = _build_layers(parsed, features)

    # ========== Generate refl1d Script ==========
    # Check for back reflection geometry
    back_reflection = parsed.get("back_reflection", False)

    # Get intensity normalization settings (default: vary 0.7–1.1)
    intensity_settings = parsed.get("intensity", {})
    intensity = {
        "value": intensity_settings.get("value", 1.0),
        "min": intensity_settings.get("min", 0.7),
        "max": intensity_settings.get("max", 1.1),
        "fixed": intensity_settings.get("fixed", False),
    }

    try:
        model_script = build_refl1d_script(
            layers=layers,
            substrate=substrate,
            ambient=ambient,
            data_file=state["data_file"],
            back_reflection=back_reflection,
            intensity=intensity,
        )
        updates["current_model"] = model_script
        updates["model_history"] = [
            {
                "iteration": state.get("iteration", 0),
                "n_layers": len(layers),
                "description": f"{len(layers)}-layer model",
                "script": model_script,
            }
        ]

        # Generate explanation
        explanation = _explain_model(layers, substrate, ambient, features)
        updates["messages"] = [
            Message(role="assistant", content=explanation, timestamp=None)
        ]

    except Exception as e:
        updates["error"] = f"Model building failed: {str(e)}"
        updates["messages"] = [
            Message(
                role="system", content=f"Error building model: {str(e)}", timestamp=None
            )
        ]

    return updates


def _get_substrate(parsed: dict, features: dict) -> dict:
    """Determine substrate from description or features."""
    if parsed.get("substrate"):
        return parsed["substrate"]

    # Default to silicon if not specified
    return {
        "name": "silicon",
        "sld": get_sld("silicon"),
        "roughness": 3.0,
        "roughness_max": 15.0,
    }


def _get_ambient(parsed: dict) -> dict:
    """Determine ambient medium."""
    if parsed.get("ambient"):
        return parsed["ambient"]

    # Default to air
    return {
        "name": "air",
        "sld": 0.0,
    }


def _build_layers(parsed: dict, features: dict) -> List[dict]:
    """Build layer list from description and features."""
    layers = []

    # Use described layers as starting point
    if parsed.get("layers"):
        for i, layer in enumerate(parsed["layers"]):
            # Handle None values from LLM parsing with sensible defaults
            sld = layer.get("sld") if layer.get("sld") is not None else 2.0

            # SLD range: use provided values or calculate defaults
            # Ensure a minimum spread of ±1.5 around the expected value
            provided_sld_min = layer.get("sld_min")
            provided_sld_max = layer.get("sld_max")

            if provided_sld_min is not None and provided_sld_max is not None:
                # Check if the provided range is too narrow (less than 1.0 spread)
                if (provided_sld_max - provided_sld_min) < 2.0:
                    # Expand to ensure reasonable fitting flexibility
                    sld_min = max(sld - 2.5, -6.0)
                    sld_max = min(sld + 2.5, 10.0)
                else:
                    sld_min = provided_sld_min
                    sld_max = provided_sld_max
            else:
                # Default: ±2.5 around expected value, bounded by physical limits
                sld_min = max(sld - 2.5, -6.0)
                sld_max = min(sld + 2.5, 10.0)

            thickness = (
                layer.get("thickness") if layer.get("thickness") is not None else 100.0
            )
            thickness_min = (
                layer.get("thickness_min")
                if layer.get("thickness_min") is not None
                else thickness * 0.5
            )
            thickness_max = (
                layer.get("thickness_max")
                if layer.get("thickness_max") is not None
                else thickness * 2.0
            )
            roughness = (
                layer.get("roughness") if layer.get("roughness") is not None else 5.0
            )
            roughness_max = (
                layer.get("roughness_max")
                if layer.get("roughness_max") is not None
                else 30.0
            )

            layers.append(
                {
                    "name": layer.get("name", f"layer{i + 1}"),
                    "sld": sld,
                    "sld_min": sld_min,
                    "sld_max": sld_max,
                    "thickness": thickness,
                    "thickness_min": thickness_min,
                    "thickness_max": thickness_max,
                    "roughness": roughness,
                    "roughness_max": roughness_max,
                }
            )

    # If no layers described, use feature estimates
    elif features.get("estimated_n_layers", 0) > 0:
        n_layers = features["estimated_n_layers"]
        total_thickness = features.get("estimated_total_thickness", 100.0)
        avg_thickness = total_thickness / n_layers if n_layers > 0 else 100.0

        for i in range(n_layers):
            layers.append(
                {
                    "name": f"layer{i + 1}",
                    "sld": 2.0,  # Generic value
                    "sld_min": 0.0,
                    "sld_max": 7.0,
                    "thickness": avg_thickness,
                    "thickness_min": avg_thickness * 0.5,
                    "thickness_max": avg_thickness * 2.0,
                    "roughness": features.get("estimated_roughness", 5.0),
                    "roughness_max": 30.0,
                }
            )

    # Apply feature-based refinements
    if layers and features:
        # Use estimated roughness
        if features.get("estimated_roughness"):
            for layer in layers:
                layer["roughness"] = min(
                    layer["roughness"], features["estimated_roughness"]
                )

        # Use oscillation-based thickness if available
        if (
            features.get("oscillation_periods")
            and len(features["oscillation_periods"]) > 0
        ):
            # Use first oscillation period as thickness estimate
            period = features["oscillation_periods"][0]
            if "thickness" in period and len(layers) == 1:
                layers[0]["thickness"] = period["thickness"]
                layers[0]["thickness_min"] = period["thickness"] * 0.7
                layers[0]["thickness_max"] = period["thickness"] * 1.3

    return layers


def build_refl1d_script(
    layers: List[dict],
    substrate: dict,
    ambient: dict,
    data_file: str,
    back_reflection: bool = False,
    intensity: dict = None,
) -> str:
    """
    Generate refl1d Python script for the model.

    Args:
        layers: List of layer dictionaries
        substrate: Substrate info
        ambient: Ambient info
        data_file: Path to data file
        back_reflection: If True, neutrons come from substrate side
        intensity: Dict with value, min, max, fixed for probe intensity

    Returns:
        Python script string
    """
    # Default intensity settings
    if intensity is None:
        intensity = {"value": 1.0, "min": 0.9, "max": 1.1, "fixed": False}
    import os

    # Use absolute path so the model can be run from any directory
    abs_data_file = os.path.abspath(data_file)

    lines = [
        '"""',
        "Auto-generated refl1d model.",
        '"""',
        "",
        "import warnings",
        "from refl1d.names import *",
        "",
        "# Suppress refl1d deprecation warnings",
        'warnings.filterwarnings("ignore", category=UserWarning, module="refl1d")',
        "",
        "# ========== Load Data ==========",
        f'probe = load4("{abs_data_file}")',
    ]

    lines.extend(
        [
            "",
            "# ========== Materials ==========",
            f'substrate = SLD(name="{substrate["name"]}", rho={substrate["sld"]:.4f})',
            f'ambient = SLD(name="{ambient["name"]}", rho={ambient["sld"]:.4f})',
        ]
    )

    # Define layer materials
    for i, layer in enumerate(layers):
        lines.append(
            f'material{i + 1} = SLD(name="{layer["name"]}", rho={layer["sld"]:.4f})'
        )

    lines.extend(
        [
            "",
            "# ========== Sample Structure ==========",
        ]
    )

    # Build sample stack
    # Stack is always ordered in beam direction (first element is where beam enters)
    if back_reflection:
        # Back reflection: neutrons come from substrate side
        # Beam sees: ambient first, then layers (reversed), then substrate
        # Example: sample = THF(0, roughness) | Cu(...) | Ti(...) | Si
        lines.append("# Neutrons come from substrate side (back reflection)")
        lines.append(
            "# Stack ordered in beam direction: ambient -> layers -> substrate"
        )
        stack_parts = [
            "ambient(0, {:.1f})".format(layers[-1]["roughness"] if layers else 3.0)
        ]
        # Add layers in reverse order (furthest from substrate first, i.e. closest to ambient)
        for i in reversed(range(len(layers))):
            layer = layers[i]
            stack_parts.append(
                f"material{i + 1}({layer['thickness']:.1f}, {layer['roughness']:.1f})"
            )
        stack_parts.append(f"substrate(0, {substrate.get('roughness', 3.0):.1f})")
    else:
        # Normal geometry: neutrons come from ambient side
        # Stack: ambient | layer1 | layer2 | ... | substrate
        lines.append("# Built from ambient (top) to substrate (bottom)")
        stack_parts = ["ambient"]
        for i, layer in enumerate(layers):
            stack_parts.append(
                f"material{i + 1}({layer['thickness']:.1f}, {layer['roughness']:.1f})"
            )
        stack_parts.append(f"substrate(0, {substrate.get('roughness', 3.0):.1f})")

    lines.append(f"sample = {' | '.join(stack_parts)}")

    lines.extend(
        [
            "",
            "# ========== Fit Parameters ==========",
        ]
    )

    # Ambient SLD - allow to vary if not air
    if ambient.get("name", "").lower() != "air" and ambient.get("sld", 0) != 0:
        ambient_sld = ambient["sld"]
        # Allow ±20% variation around the expected SLD
        ambient_min = max(ambient_sld * 0.8, -1.0)
        ambient_max = ambient_sld * 1.2
        # Ambient is always first (index 0) in both geometries now
        lines.append(
            f"sample[0].material.rho.range({ambient_min:.2f}, {ambient_max:.2f})"
        )

    # Add parameter ranges for layers
    for i, layer in enumerate(layers):
        if back_reflection:
            # In back reflection: ambient is index 0, layers are reversed
            # layers[i] corresponds to material{i+1}, which is at index (n - i)
            # e.g., for 2 layers: layers[0]=material1 at idx 2, layers[1]=material2 at idx 1
            idx = len(layers) - i
        else:
            # In normal geometry: ambient is index 0, layers start at index 1
            idx = i + 1

        # Thickness
        t_min = layer.get("thickness_min", layer["thickness"] * 0.5)
        t_max = layer.get("thickness_max", layer["thickness"] * 2.0)
        lines.append(f"sample[{idx}].thickness.range({t_min:.1f}, {t_max:.1f})")

        # SLD
        lines.append(
            f"sample[{idx}].material.rho.range({layer['sld_min']:.2f}, {layer['sld_max']:.2f})"
        )

        # Roughness
        r_max = layer.get("roughness_max", 30.0)
        lines.append(f"sample[{idx}].interface.range(0, {r_max:.1f})")

    # Substrate roughness (substrate is always last)
    sub_rough_max = substrate.get("roughness_max", 15.0)
    lines.append(f"sample[{len(layers) + 1}].interface.range(0, {sub_rough_max:.1f})")
    # Probe intensity normalization
    if not intensity.get("fixed", False):
        lines.extend(
            [
                "",
                "# ========== Probe Intensity ===========",
                "# Allow intensity to vary to account for normalization uncertainty",
                f"probe.intensity.range({intensity['min']:.2f}, {intensity['max']:.2f})",
            ]
        )
    lines.extend(
        [
            "",
            "# ========== Experiment ==========",
            "experiment = Experiment(probe=probe, sample=sample)",
            "problem = FitProblem(experiment)",
        ]
    )

    return "\n".join(lines)


def _explain_model(layers: list, substrate: dict, ambient: dict, features: dict) -> str:
    """Generate human-readable model explanation."""
    lines = ["**Proposed Model:**"]
    lines.append("")
    lines.append(f"- **Ambient:** {ambient['name']} (SLD = {ambient['sld']:.2f})")

    for i, layer in enumerate(layers, 1):
        lines.append(
            f"- **Layer {i}:** {layer['name']} "
            f"(d = {layer['thickness']:.0f} Å, SLD = {layer['sld']:.2f}, "
            f"σ = {layer['roughness']:.1f} Å)"
        )

    lines.append(
        f"- **Substrate:** {substrate['name']} "
        f"(SLD = {substrate['sld']:.2f}, σ = {substrate.get('roughness', 3):.1f} Å)"
    )

    lines.append("")
    lines.append("**Parameter ranges:**")
    for i, layer in enumerate(layers, 1):
        t_min = layer.get("thickness_min", layer["thickness"] * 0.5)
        t_max = layer.get("thickness_max", layer["thickness"] * 2.0)
        lines.append(
            f"- Layer {i}: d ∈ [{t_min:.0f}, {t_max:.0f}] Å, "
            f"SLD ∈ [{layer['sld_min']:.1f}, {layer['sld_max']:.1f}], "
            f"σ ∈ [0, {layer.get('roughness_max', 30):.0f}] Å"
        )

    return "\n".join(lines)


# ============================================================================
# Refinement helpers
# ============================================================================


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    # Remove ```python ... ``` wrapping
    text = re.sub(r"^```python\s*\n", "", text)
    text = re.sub(r"^```\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def _validate_model_script(script: str) -> bool:
    """Check that a model script has the minimum required components."""
    required = [
        "load4(",  # Data loading
        "SLD(",  # Material definition
        "sample =",  # Sample stack
        "Experiment(",  # Experiment setup
        "FitProblem(",  # Problem definition
    ]
    # At least 4 out of 5 required (allow minor variations)
    matches = sum(1 for r in required if r in script)
    return matches >= 4


def _widen_all_bounds(model: str) -> str:
    """Fallback: widen all parameter bounds by 50% in the existing model."""
    range_pattern = r"\.range\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)"

    def widen(match):
        low, high = float(match.group(1)), float(match.group(2))
        spread = high - low
        new_low = low - spread * 0.25
        new_high = high + spread * 0.25
        return f".range({new_low:.2f}, {new_high:.2f})"

    return re.sub(range_pattern, widen, model)


def _fix_sld_attr_access(script: str) -> str:
    """Fix LLM-generated scripts that use ``var.material.rho`` or
    ``var.thickness`` instead of ``sample[i].material.rho`` etc.

    ``SLD(...)`` objects do not have ``.material``, ``.thickness``, or
    ``.interface`` — those attributes only exist on ``Slab`` objects inside
    the sample stack.  This function rewrites offending lines to use
    ``sample[i]`` indexing by matching variable names to sample stack order.
    """
    # 1. Build var_name → sample index mapping from the sample stack.
    #    Pattern: lines like ``  var(thickness, roughness)`` or ``| var(...)``
    stack_pat = re.compile(r"^\s*\|?\s*(\w+)\s*\(", re.MULTILINE)
    # Find the sample block by matching balanced parens (non-greedy `?`
    # would stop at the first `)` inside the nested calls).
    header = re.search(r"sample\s*=\s*\(", script)
    if not header:
        return script
    depth, start = 1, header.end()
    for pos in range(start, len(script)):
        if script[pos] == "(":
            depth += 1
        elif script[pos] == ")":
            depth -= 1
            if depth == 0:
                break
    sample_body = script[start:pos]

    var_to_idx: dict[str, int] = {}
    for idx, m in enumerate(stack_pat.finditer(sample_body)):
        var_to_idx[m.group(1)] = idx

    if not var_to_idx:
        return script

    # 2. Rewrite offending lines.
    #    Match:  <var>.<attr>.range(...)  or  <var>.material.<attr>.range(...)
    #    where <var> is one of the sample-stack variable names.
    #    Sort longest-first so "copper_oxide" matches before "copper".
    var_names = "|".join(
        re.escape(v) for v in sorted(var_to_idx, key=len, reverse=True)
    )
    bad_pat = re.compile(
        rf"^(\s*)({var_names})\.(material\.|)(thickness|interface|rho)"
        rf"(\.range\([^)]*\).*)",
        re.MULTILINE,
    )

    def _rewrite(m: re.Match) -> str:
        indent = m.group(1)
        var = m.group(2)
        mat_prefix = m.group(3)  # 'material.' or ''
        attr = m.group(4)
        rest = m.group(5)
        idx = var_to_idx[var]

        if attr == "rho":
            return f"{indent}sample[{idx}].material.rho{rest}"
        else:
            return f"{indent}sample[{idx}].{attr}{rest}"

    fixed = bad_pat.sub(_rewrite, script)
    if fixed != script:
        n = len(bad_pat.findall(script))
        logger.info(
            "[MODELING] Fixed %d SLD-attribute lines (var.attr → sample[i].attr)",
            n,
        )
    return fixed


def _summarize_model_changes(old_model: str, new_model: str) -> list[str]:
    """Summarize differences between old and new model scripts."""
    changes = []

    # Count layers
    old_layers = len(re.findall(r"material\d+\s*=\s*SLD\(", old_model))
    new_layers = len(re.findall(r"material\d+\s*=\s*SLD\(", new_model))
    if new_layers != old_layers:
        changes.append(f"Layer count changed: {old_layers} → {new_layers}")

    # Check for new materials
    old_materials = set(re.findall(r'name="([^"]+)"', old_model))
    new_materials = set(re.findall(r'name="([^"]+)"', new_model))
    added = new_materials - old_materials
    removed = old_materials - new_materials
    if added:
        changes.append(f"Added materials: {', '.join(added)}")
    if removed:
        changes.append(f"Removed materials: {', '.join(removed)}")

    # Check for bound changes
    old_ranges = re.findall(r"\.range\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)", old_model)
    new_ranges = re.findall(r"\.range\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)", new_model)
    if len(new_ranges) != len(old_ranges):
        changes.append(
            f"Number of fit parameters changed: {len(old_ranges)} → {len(new_ranges)}"
        )

    if not changes:
        changes.append("Parameter values and bounds adjusted")

    return changes


def _format_refinement_explanation(
    changes: list[str],
    issues: list[str],
    suggestions: list[str],
) -> str:
    """Format a human-readable explanation of the model refinement."""
    lines = ["**Model Refinement:**"]
    lines.append("")

    if issues:
        lines.append("**Issues addressed:**")
        for issue in issues:
            lines.append(f"- {issue}")
        lines.append("")

    lines.append("**Changes made:**")
    for change in changes:
        lines.append(f"- {change}")
    lines.append("")
    lines.append("*Re-running fit with refined model...*")

    return "\n".join(lines)
