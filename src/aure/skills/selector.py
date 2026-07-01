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

# Always-on skills (baseline + meta-skills)
_BASELINE_SKILL = "neutron-reflectometry"
_ALWAYS_ON = ("neutron-reflectometry", "structural-hypothesis-ranking")

# Low-threshold skill: any liquid/solvent ambient should pull in
# ``solvent-contrast-matching`` even when the LLM did not pick it and even when
# deuteration was never mentioned — an unspecified solvent may be deuterated and
# the user simply forgot to say so. See ``_has_liquid_ambient``.
_SOLVENT_SKILL = "solvent-contrast-matching"

# Ambient names that are NOT liquids; everything else with a name is treated as
# a solvent/electrolyte/solution medium for the purpose of the low threshold.
# Includes gases/vacuum plus the common solid substrate/film materials that an
# LLM might mistakenly drop into the ambient field (a misparse must not be read
# as "there is a solvent"). The generic guard against that misparse is the
# substrate/layer-name cross-check in ``_has_liquid_ambient``; this set is the
# backstop for solids that aren't otherwise in the stack.
_NON_LIQUID_AMBIENTS = frozenset(
    {
        "",
        "air",
        "vacuum",
        "gas",
        "none",
        "n2",
        "nitrogen",
        "argon",
        "ar",
        "helium",
        "he",
        "o2",
        "oxygen",
        "co2",
        # common solids
        "si",
        "silicon",
        "sio2",
        "quartz",
        "fused silica",
        "sapphire",
        "al2o3",
        "glass",
        "cu",
        "copper",
        "au",
        "gold",
        "ag",
        "silver",
        "ti",
        "titanium",
        "fe",
        "iron",
        "ni",
        "nickel",
        "pt",
        "platinum",
        "pd",
        "palladium",
        "al",
        "aluminum",
        "aluminium",
    }
)

# Free-text cues that a liquid medium is present, used before the sample has
# been parsed (when no ambient dict is available yet).
_LIQUID_DESC_KEYWORDS = (
    "solvent",
    "electrolyte",
    "solution",
    "buffer",
    "aqueous",
    "dissolv",
    "immers",
    "submerg",
    "in water",
    "in liquid",
    "d2o",
    "h2o",
    "deuterat",
    "protonat",
    "contrast match",
    "contrast variation",
    "thf",
    "toluene",
    "methanol",
    "ethanol",
    "cyclohexane",
)


def _structural_material_names(parsed_sample: Optional[Dict[str, Any]]) -> set:
    """Lowercased substrate + layer material names for this sample.

    Used to reject an ambient that merely repeats a structural material — the
    signature of an LLM misparse (a solid dropped into the ambient field), which
    must not be read as "there is a solvent".
    """
    names: set = set()
    if not parsed_sample:
        return names
    sub = (parsed_sample.get("substrate") or {}).get("name")
    if sub:
        names.add(str(sub).strip().lower())
    for layer in parsed_sample.get("layers") or []:
        n = (layer or {}).get("name")
        if n:
            names.add(str(n).strip().lower())
    return names


def _ambient_name_is_liquid(amb_name: Any, solid_names: set) -> bool:
    """True when *amb_name* looks like a liquid medium (not gas/vacuum/solid)."""
    n = str(amb_name or "").strip().lower()
    if not n or n in _NON_LIQUID_AMBIENTS:
        return False
    if n in solid_names:  # repeats a substrate/layer material → a misparse
        return False
    return True


def _has_liquid_ambient(
    sample_description: str,
    parsed_sample: Optional[Dict[str, Any]] = None,
    states: Optional[list] = None,
) -> bool:
    """Return True when the sample appears to sit in a liquid/solvent medium.

    The parsed ambient (when available) is the reliable signal: any named
    medium that is not a gas/vacuum/solid is treated as a liquid. Per-state
    ambients (multi-state co-refinement) are checked too, since the model-level
    ambient may be air while individual states are in solvent. Before parsing,
    fall back to free-text cues in the description. Intentionally permissive —
    the cost of a false positive is one extra (relevant) skill in the prompt,
    which is exactly the "low threshold" the solvent skill should have.
    """
    solid_names = _structural_material_names(parsed_sample)
    if parsed_sample:
        amb = parsed_sample.get("ambient") or {}
        if _ambient_name_is_liquid(amb.get("name"), solid_names):
            return True
    for st in states or []:
        st_amb = (st or {}).get("ambient") or {}
        if _ambient_name_is_liquid(st_amb.get("name"), solid_names):
            return True
    desc = (sample_description or "").lower()
    return any(kw in desc for kw in _LIQUID_DESC_KEYWORDS)


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

If observations from fitting are provided, treat them as strong evidence: an \
artifact seen in the data (e.g. an unexpected contrast step, residual fringes \
of a characteristic thickness, a parameter pinned at a bound) may point to a \
phenomenon whose skill was not obvious from the static description.  Include \
such a skill when the observation makes its guidance relevant.

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
        # Skip always-on skills — they're always active
        if meta.name in _ALWAYS_ON:
            continue
        desc = meta.description.replace("\n", " ").strip()
        lines.append(f"- **{meta.name}**: {desc}")
    return "\n".join(lines)


def _build_sample_info(
    sample_description: str,
    parsed_sample: Optional[Dict[str, Any]] = None,
    extra_context: Optional[str] = None,
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
    if extra_context and extra_context.strip():
        parts.append(f"\nObservations from fitting so far:\n{extra_context.strip()}")
    return "\n".join(parts)


def select_skills(
    sample_description: str,
    parsed_sample: Optional[Dict[str, Any]] = None,
    registry: Optional[SkillRegistry] = None,
    extra_context: Optional[str] = None,
    states: Optional[list] = None,
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
    extra_context
        Optional free-form observations from fitting (residual artifacts,
        boundary hits, χ²/BIC trajectory, the evaluator's concerns). When
        provided, lets the selector activate a skill that only became
        relevant once the data revealed an artifact. Backward-compatible:
        ``None`` reproduces the intake-time selection behavior.
    states
        Optional list of per-state dicts (multi-state co-refinement). Their
        per-state ``ambient`` overrides are scanned for the low-threshold
        solvent-skill check, so a run whose model-level ambient is air but
        whose states sit in solvent still activates the skill.

    Returns
    -------
    list[str]
        Names of skills to activate (always includes ``neutron-reflectometry``).
    """
    if registry is None:
        registry = SkillRegistry()

    from ..llm import llm_available  # local import to avoid circular deps

    selected: Optional[list[str]] = None
    if llm_available():
        try:
            selected = _select_skills_llm(
                sample_description, parsed_sample, registry, extra_context
            )
            logger.info("LLM-selected skills: %s", selected)
        except Exception as e:
            logger.warning("LLM skill selection failed: %s", e)
            selected = None

    if selected is None:
        selected = sorted(n for n in _ALWAYS_ON if registry.get_metadata(n))
        logger.info("LLM unavailable, using always-on skills only: %s", selected)

    return _augment_solvent_skill(
        selected, sample_description, parsed_sample, registry, states
    )


def _augment_solvent_skill(
    selected: list[str],
    sample_description: str,
    parsed_sample: Optional[Dict[str, Any]],
    registry: SkillRegistry,
    states: Optional[list] = None,
) -> list[str]:
    """Force-include ``solvent-contrast-matching`` for any liquid ambient.

    Applied to both the LLM and fallback selection paths so the solvent skill
    activates deterministically whenever the sample sits in a solvent /
    electrolyte / solution — independent of the LLM's judgement and even when
    deuteration is never mentioned. This is the "low threshold" the solvent
    skill should have, so the deuterated-solvent hypothesis is always on the
    table when a fit stalls.
    """
    result = set(selected)
    if (
        _SOLVENT_SKILL not in result
        and registry.get_metadata(_SOLVENT_SKILL)
        and _has_liquid_ambient(sample_description, parsed_sample, states)
    ):
        logger.info(
            "Liquid ambient detected — force-including %s (low threshold)",
            _SOLVENT_SKILL,
        )
        result.add(_SOLVENT_SKILL)
    return sorted(result)


def _select_skills_llm(
    sample_description: str,
    parsed_sample: Optional[Dict[str, Any]],
    registry: SkillRegistry,
    extra_context: Optional[str] = None,
) -> list[str]:
    """Use the LLM to select relevant skills from the catalog."""
    from langchain_core.messages import HumanMessage
    from .. import llm as _llm

    catalog = _build_catalog(registry)
    sample_info = _build_sample_info(sample_description, parsed_sample, extra_context)
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

    # Validate against registry and always include always-on skills that exist
    available = set(registry.skill_names)
    activated = (set(_ALWAYS_ON) & available) | {n for n in names if n in available}

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
