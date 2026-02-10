"""
Refinement helper utilities for model modification.

These are used by the MCP server's modify_model tool to make
targeted edits to refl1d model scripts (widen bounds, adjust
roughness, add layers, set thickness/SLD).

Note: The refinement *node* has been removed from the workflow.
Refinement is now handled by the modeling node's _refine_model()
function, which uses the LLM to regenerate the full script.
"""

import re
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def _widen_bounds(model: str, params: Dict[str, float]) -> Tuple[str, bool]:
    """Widen parameter bounds by 50% if values are near limits."""
    modified = False
    new_model = model

    range_pattern = r'\.range\((\d+\.?\d*),\s*(\d+\.?\d*)\)'

    def widen_range(match):
        nonlocal modified
        low, high = float(match.group(1)), float(match.group(2))
        new_low = max(0, low * 0.5)
        new_high = high * 1.5
        modified = True
        return f'.range({new_low:.1f}, {new_high:.1f})'

    new_model = re.sub(range_pattern, widen_range, new_model)
    return new_model, modified


def _increase_roughness_bounds(model: str) -> Tuple[str, bool]:
    """Increase roughness bounds by 50%, capped at 50 Å."""
    modified = False
    new_model = model

    pattern = r'\.interface\.range\((\d+),\s*(\d+\.?\d*)\)'

    def increase_roughness(match):
        nonlocal modified
        low, high = float(match.group(1)), float(match.group(2))
        new_high = min(high * 1.5, 50)  # Cap at 50 Å
        if new_high > high:
            modified = True
            return f'.interface.range({int(low)}, {new_high:.1f})'
        return match.group(0)

    new_model = re.sub(pattern, increase_roughness, new_model)
    return new_model, modified


def _add_layer(model: str) -> Tuple[str, bool]:
    """
    Add an additional layer to the model.

    This is a placeholder — adding a layer requires knowledge of
    the sample structure, so it is better handled by the LLM-based
    refinement in the modeling node.
    """
    logger.warning("[REFINEMENT] _add_layer is a stub — use the modeling node for structural changes")
    return model, False


def _set_layer_thickness(model: str, layer_index: int, value: float) -> Tuple[str, bool]:
    """
    Set the thickness of a specific layer by index (0 = first from substrate).

    Finds thickness assignments in the model script and updates the one
    at the given index.
    """
    pattern = r'(\.thickness\.value\s*=\s*)(\d+\.?\d*)'
    matches = list(re.finditer(pattern, model))

    if layer_index < 0 or layer_index >= len(matches):
        logger.warning(f"[REFINEMENT] Layer index {layer_index} out of range (found {len(matches)} layers)")
        return model, False

    match = matches[layer_index]
    new_model = model[:match.start(2)] + f"{value:.1f}" + model[match.end(2):]
    return new_model, True


def _set_layer_sld(model: str, layer_index: int, value: float) -> Tuple[str, bool]:
    """
    Set the SLD of a specific layer by index (0 = first from substrate).

    Finds SLD (rho) assignments in the model script and updates the one
    at the given index.
    """
    pattern = r'(\.rho\.value\s*=\s*)([-]?\d+\.?\d*)'
    matches = list(re.finditer(pattern, model))

    if layer_index < 0 or layer_index >= len(matches):
        logger.warning(f"[REFINEMENT] Layer index {layer_index} out of range (found {len(matches)} layers)")
        return model, False

    match = matches[layer_index]
    new_model = model[:match.start(2)] + f"{value:.4f}" + model[match.end(2):]
    return new_model, True
