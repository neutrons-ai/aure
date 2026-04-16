"""
Skill selection: choose which skills to activate based on sample description.

``neutron-reflectometry`` is always activated. Other skills are activated when
keywords from their description overlap with the sample description or parsed
sample material/layer names.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Any, Optional

from .loader import SkillRegistry

logger = logging.getLogger(__name__)

# Always-on skill
_BASELINE_SKILL = "neutron-reflectometry"

# Keyword → skill mapping. Keys are lowercase tokens that, if found in the
# sample description or parsed sample, trigger the corresponding skill.
_KEYWORD_MAP: Dict[str, str] = {
    # sei-layer-analysis
    "sei": "sei-layer-analysis",
    "solid electrolyte interphase": "sei-layer-analysis",
    "battery": "sei-layer-analysis",
    "electrolyte": "sei-layer-analysis",
    "lithium plating": "sei-layer-analysis",
    "lithium metal": "sei-layer-analysis",
    "li plating": "sei-layer-analysis",
    "anode": "sei-layer-analysis",
    "cathode": "sei-layer-analysis",
    "electrochemical": "sei-layer-analysis",
    "electrochemistry": "sei-layer-analysis",
    # polymer-films
    "polymer": "polymer-films",
    "polystyrene": "polymer-films",
    "ps": "polymer-films",
    "d-ps": "polymer-films",
    "d-polystyrene": "polymer-films",
    "pmma": "polymer-films",
    "polyethylene": "polymer-films",
    "block copolymer": "polymer-films",
    "brush": "polymer-films",
    "thin film": "polymer-films",
    "spin coat": "polymer-films",
    "spin-coat": "polymer-films",
    # metal-oxide-interfaces
    "oxide": "metal-oxide-interfaces",
    "cuo": "metal-oxide-interfaces",
    "cu2o": "metal-oxide-interfaces",
    "tio2": "metal-oxide-interfaces",
    "native oxide": "metal-oxide-interfaces",
    "sio2": "metal-oxide-interfaces",
    "metal oxide": "metal-oxide-interfaces",
    "copper oxide": "metal-oxide-interfaces",
    "titanium oxide": "metal-oxide-interfaces",
    # solvent-contrast-matching
    "d2o": "solvent-contrast-matching",
    "h2o": "solvent-contrast-matching",
    "contrast": "solvent-contrast-matching",
    "contrast match": "solvent-contrast-matching",
    "contrast variation": "solvent-contrast-matching",
    "deuterated": "solvent-contrast-matching",
    "protonated": "solvent-contrast-matching",
    "solvent": "solvent-contrast-matching",
    "thf": "solvent-contrast-matching",
    "dthf": "solvent-contrast-matching",
    "d8-thf": "solvent-contrast-matching",
    "toluene": "solvent-contrast-matching",
    "d-toluene": "solvent-contrast-matching",
    "cyclohexane": "solvent-contrast-matching",
}


def select_skills(
    sample_description: str,
    parsed_sample: Optional[Dict[str, Any]] = None,
    registry: Optional[SkillRegistry] = None,
) -> list[str]:
    """Select skills to activate based on sample context.

    Parameters
    ----------
    sample_description
        Free-form sample description from the user.
    parsed_sample
        Parsed sample dict (substrate, layers, ambient, etc.), if available.
    registry
        Skill registry to validate names against. If *None*, all matched
        skill names are returned without validation.

    Returns
    -------
    list[str]
        Names of skills to activate (always includes ``neutron-reflectometry``).
    """
    activated = {_BASELINE_SKILL}

    # Build searchable text from description + parsed sample fields
    text_parts = [sample_description.lower()]
    if parsed_sample:
        for layer in parsed_sample.get("layers", []):
            text_parts.append(layer.get("name", "").lower())
        sub = parsed_sample.get("substrate", {})
        text_parts.append(sub.get("name", "").lower())
        amb = parsed_sample.get("ambient", {})
        text_parts.append(amb.get("name", "").lower())
        for c in parsed_sample.get("constraints", []):
            text_parts.append(c.lower())

    search_text = " ".join(text_parts)

    for keyword, skill_name in _KEYWORD_MAP.items():
        # Use word boundary for short keywords to avoid false positives
        if len(keyword) <= 3:
            if re.search(rf"\b{re.escape(keyword)}\b", search_text):
                activated.add(skill_name)
        else:
            if keyword in search_text:
                activated.add(skill_name)

    # Filter to actually available skills if registry is provided
    if registry is not None:
        available = set(registry.skill_names)
        activated = {s for s in activated if s in available}
        # Always keep baseline if available
        if _BASELINE_SKILL in available:
            activated.add(_BASELINE_SKILL)

    result = sorted(activated)
    logger.info("Selected skills: %s", result)
    return result


def load_skill_context(
    skill_names: list[str],
    registry: SkillRegistry,
) -> str:
    """Load and concatenate skill bodies for the given names.

    Parameters
    ----------
    skill_names
        List of skill names to load.
    registry
        The skill registry to load from.

    Returns
    -------
    str
        Combined Markdown text suitable for injection into an LLM prompt.
    """
    sections = []
    for name in skill_names:
        try:
            body = registry.load_body(name)
            if body:
                sections.append(f"### Skill: {name}\n\n{body}")
        except KeyError:
            logger.warning("Skill '%s' not found in registry", name)
    return "\n\n---\n\n".join(sections)
