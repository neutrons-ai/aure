"""
Skill selection: choose which skills to activate based on sample description.

``neutron-reflectometry`` is always activated.  Other skills are selected by
asking an LLM to match the sample description against the skill catalog
(name + description).  If the LLM is unavailable, only the baseline skill
is returned.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, Any, Optional

from .loader import SkillRegistry

logger = logging.getLogger(__name__)

# Always-on skill
_BASELINE_SKILL = "neutron-reflectometry"

# ---------------------------------------------------------------------------
# LLM skill-selection prompt
# ---------------------------------------------------------------------------

_SKILL_SELECTION_PROMPT = """\
You are a neutron reflectometry expert selecting domain-specific skill \
modules for a reflectometry analysis workflow.

Below is a catalog of available skills.  Each has a name and a short \
description.  Given the sample information, return **only** the names of \
skills whose domain knowledge is relevant to this sample.  Do NOT include \
a skill just because a keyword appears — the skill must actually provide \
useful guidance for the analysis.

<available_skills>
{catalog}
</available_skills>

<sample_information>
{sample_info}
</sample_information>

Return a JSON array of skill names, e.g. ["metal-oxide-interfaces", "polymer-films"].
Return an empty array [] if no additional skills are needed beyond the \
baseline neutron-reflectometry skill (which is always included automatically).
Return ONLY the JSON array, no other text.
"""


def _build_catalog(registry: SkillRegistry) -> str:
    """Build a compact skill catalog string from the registry."""
    lines = []
    for meta in registry.all_metadata():
        # Skip the baseline skill — it's always on
        if meta.name == _BASELINE_SKILL:
            continue
        desc = meta.description.replace("\n", " ").strip()
        lines.append(f"- **{meta.name}**: {desc}")
    return "\n".join(lines)


def _build_sample_info(
    sample_description: str,
    parsed_sample: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a sample information block for the LLM prompt."""
    parts = [f"Description: {sample_description}"]
    if parsed_sample:
        layers = parsed_sample.get("layers", [])
        if layers:
            names = [l.get("name", "?") for l in layers]
            parts.append(f"Layers: {', '.join(names)}")
        sub = parsed_sample.get("substrate", {})
        if sub.get("name"):
            parts.append(f"Substrate: {sub['name']}")
        amb = parsed_sample.get("ambient", {})
        if amb.get("name"):
            parts.append(f"Ambient: {amb['name']}")
    return "\n".join(parts)


def select_skills(
    sample_description: str,
    parsed_sample: Optional[Dict[str, Any]] = None,
    registry: Optional[SkillRegistry] = None,
) -> list[str]:
    """Select skills to activate based on sample context.

    Uses an LLM to match the sample against the skill catalog.
    Returns only the baseline skill if the LLM is unavailable or fails.

    Parameters
    ----------
    sample_description
        Free-form sample description from the user.
    parsed_sample
        Parsed sample dict (substrate, layers, ambient, etc.), if available.
    registry
        Skill registry to validate names against.

    Returns
    -------
    list[str]
        Names of skills to activate (always includes ``neutron-reflectometry``).
    """
    if registry is None:
        registry = SkillRegistry()

    from ..llm import llm_available  # local import to avoid circular deps

    if llm_available():
        try:
            selected = _select_skills_llm(sample_description, parsed_sample, registry)
            logger.info("LLM-selected skills: %s", selected)
            return selected
        except Exception as e:
            logger.warning("LLM skill selection failed: %s", e)

    result = [_BASELINE_SKILL]
    logger.info("LLM unavailable, using baseline skill only: %s", result)
    return result


def _select_skills_llm(
    sample_description: str,
    parsed_sample: Optional[Dict[str, Any]],
    registry: SkillRegistry,
) -> list[str]:
    """Use the LLM to select relevant skills from the catalog."""
    from langchain_core.messages import HumanMessage
    from .. import llm as _llm

    catalog = _build_catalog(registry)
    sample_info = _build_sample_info(sample_description, parsed_sample)
    prompt = _SKILL_SELECTION_PROMPT.format(catalog=catalog, sample_info=sample_info)

    model = _llm.get_llm(temperature=0)
    response = _llm.invoke_with_timeout(model, [HumanMessage(content=prompt)])
    content = response.content.strip()

    # Strip markdown fences if present
    content = re.sub(r"^```(?:json)?\s*\n?", "", content)
    content = re.sub(r"\n?```\s*$", "", content)
    content = content.strip()

    # Extract JSON array
    match = re.search(r"\[[\s\S]*?\]", content)
    if not match:
        raise ValueError(f"No JSON array in LLM response: {content[:200]}")

    names = json.loads(match.group())
    if not isinstance(names, list):
        raise ValueError(f"Expected list, got {type(names).__name__}")

    # Validate against registry and always include baseline
    available = set(registry.skill_names)
    activated = {_BASELINE_SKILL} | {n for n in names if n in available}

    return sorted(activated)


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
