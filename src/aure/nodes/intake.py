"""
INTAKE node: Load data and parse sample description.

This is the first node in the workflow. It:
1. Loads and validates the reflectivity data file
2. Inspects file headers to determine dQ convention (FWHM vs 1-sigma)
3. Parses the sample description using an LLM
4. Populates the initial state for analysis
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Dict, Any

from ..state import ReflectivityState, Message, LLMCallRecord
from ..tools.data_tools import load_reflectivity_data, validate_reflectivity_data
from ..llm import llm_available, get_llm, invoke_with_timeout
from ..skills import SkillRegistry, select_skills, load_skill_context
from .prompts import format_sample_parse_prompt

logger = logging.getLogger(__name__)


# ============================================================================
# dQ convention detection
# ============================================================================

_MAX_HEADER_LINES = 40  # Read at most this many lines from each file


def _read_file_header(file_path: str) -> str:
    """Read the header/comment lines from a data file.

    Returns the first ``_MAX_HEADER_LINES`` lines of the file (or fewer
    if the file is shorter).  This provides enough context for the LLM
    to determine the dQ convention without reading the entire file.
    """
    lines = []
    try:
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                if i >= _MAX_HEADER_LINES:
                    break
                lines.append(line.rstrip("\n"))
    except Exception:
        pass
    return "\n".join(lines)


_DQ_INSPECTION_PROMPT = """\
You are inspecting a neutron reflectometry data file header to determine
how the resolution column (dQ) is encoded.

The file header and first few data lines are shown below:
```
{header}
```

The file has 4 columns: Q, R, dR, dQ.

Is the dQ column expressed as **FWHM** (full width at half maximum) or as
**1-sigma** (standard deviation)?

Most neutron reflectometry reduction software outputs dQ as FWHM. This is
the default assumption unless the header explicitly states otherwise (e.g.,
"dQ (1-sigma)", "sigma_Q", "dQ [sigma]", or similar).

Respond with ONLY a JSON object:
{{"dq_is_fwhm": true}}   if dQ is FWHM (or if the header is ambiguous)
{{"dq_is_fwhm": false}}  if dQ is explicitly stated as 1-sigma
"""


def detect_dq_convention(file_path: str) -> bool:
    """Determine whether the dQ column in a data file is FWHM.

    Uses the LLM to inspect the file header.  Falls back to True
    (FWHM) if the LLM is unavailable or parsing fails.

    Parameters
    ----------
    file_path
        Path to the reflectivity data file.

    Returns
    -------
    bool
        True if dQ is FWHM, False if 1-sigma.
    """
    header = _read_file_header(file_path)
    if not header:
        logger.debug("[INTAKE] Could not read header from %s; assuming FWHM", file_path)
        return True

    if not llm_available():
        logger.debug("[INTAKE] LLM not available for dQ inspection; assuming FWHM")
        return True

    try:
        from langchain_core.messages import HumanMessage

        prompt = _DQ_INSPECTION_PROMPT.format(header=header)
        llm = get_llm(temperature=0)
        response = invoke_with_timeout(llm, [HumanMessage(content=prompt)])
        content = response.content.strip()

        # Strip markdown fences
        content = re.sub(r"^```(?:json)?\s*\n", "", content)
        content = re.sub(r"\n```\s*$", "", content)
        content = content.strip()

        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            result = json.loads(match.group())
            is_fwhm = result.get("dq_is_fwhm", True)
            logger.info("[INTAKE] dQ convention for %s: %s",
                        file_path, "FWHM" if is_fwhm else "1-sigma")
            return bool(is_fwhm)
    except Exception as e:
        logger.warning("[INTAKE] dQ inspection failed for %s: %s; assuming FWHM",
                       file_path, e)

    return True


# ============================================================================
# Sample parsing
# ============================================================================


def parse_sample_with_llm(
    description: str,
    hypothesis: str | None = None,
    skill_context: str = "",
) -> Dict[str, Any]:
    """
    Parse sample description into structured format using the configured LLM.

    Args:
        description: Free-form sample description from the user
        hypothesis: Optional hypothesis to test

    Returns:
        Parsed sample dictionary with substrate, layers, ambient, etc.

    Raises:
        ValueError: If LLM is not available or parsing fails
        LLMTimeoutError: If the LLM call times out (likely quota issue)
    """
    if not llm_available():
        raise ValueError(
            "LLM is required for sample parsing. Please configure LLM_PROVIDER "
            "and appropriate API keys. See .env.example for options."
        )

    llm = get_llm(temperature=0)
    prompt = format_sample_parse_prompt(description, hypothesis, skill_context=skill_context)

    from langchain_core.messages import HumanMessage

    response = invoke_with_timeout(llm, [HumanMessage(content=prompt)])

    # Extract JSON from response
    content = response.content

    # Strip markdown code fences the LLM may wrap the JSON in
    content = re.sub(r"^```(?:json)?\s*\n", "", content)
    content = re.sub(r"\n```\s*$", "", content)
    content = content.strip()

    # Try to find JSON block in the response
    json_match = re.search(r"\{[\s\S]*\}", content)
    if json_match:
        raw_json = json_match.group()
        return json.loads(_fix_llm_json(raw_json))

    raise ValueError("Could not extract JSON from LLM response")


def _fix_llm_json(text: str) -> str:
    """Best-effort fix-up of common LLM JSON mistakes.

    Handles trailing commas before ``}`` or ``]`` and single-line
    ``// ...`` comments.
    """
    # Remove single-line comments (// ...)
    text = re.sub(r"//[^\n]*", "", text)
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def intake_node(state: ReflectivityState) -> Dict[str, Any]:
    """
    Load data and parse sample description.

    Args:
        state: Current workflow state

    Returns:
        State updates
    """
    updates = {
        "current_node": "intake",
        "messages": [],
        "llm_calls": [],
    }

    # ========== 1. Load Data ==========
    try:
        data = load_reflectivity_data(state["data_file"])
        updates["Q"] = data["Q"].tolist()
        updates["R"] = data["R"].tolist()
        updates["dR"] = data.get("dR", [0.0] * len(data["Q"])).tolist()

        # Validate
        validation = validate_reflectivity_data(data["Q"], data["R"], data.get("dR"))

        if validation["issues"]:
            updates["messages"] = [
                Message(
                    role="system",
                    content=f"Data validation warnings: {', '.join(validation['issues'])}",
                    timestamp=None,
                )
            ]

        # Build data_files list for multi-file co-refinement
        import os
        from pathlib import Path

        existing_data_files = state.get("data_files", [])
        if existing_data_files:
            # Multi-file mode: validate all additional files
            validated_files = []
            for ds in existing_data_files:
                try:
                    extra_data = load_reflectivity_data(ds["file"])
                    validate_reflectivity_data(
                        extra_data["Q"], extra_data["R"], extra_data.get("dR")
                    )
                    # Store experimental Q/R/dR for plotting
                    enriched = dict(ds)
                    enriched["Q"] = extra_data["Q"].tolist()
                    enriched["R"] = extra_data["R"].tolist()
                    enriched["dR"] = extra_data.get("dR", [0.0] * len(extra_data["Q"])).tolist() if extra_data.get("dR") is not None else [0.0] * len(extra_data["Q"])
                    validated_files.append(enriched)
                except Exception as e:
                    updates["messages"].append(
                        Message(
                            role="system",
                            content=f"Warning: could not load {ds['label']}: {e}",
                            timestamp=None,
                        )
                    )
            updates["data_files"] = validated_files
            if len(validated_files) > 1:
                labels = ", ".join(ds["label"] for ds in validated_files)
                updates["messages"].append(
                    Message(
                        role="system",
                        content=f"Multi-file co-refinement: {len(validated_files)} files loaded ({labels})",
                        timestamp=None,
                    )
                )

    except Exception as e:
        updates["error"] = f"Failed to load data: {str(e)}"
        updates["messages"] = [
            Message(
                role="system",
                content=f"Error loading data file: {str(e)}",
                timestamp=None,
            )
        ]
        return updates

    # ========== 2. Detect dQ Convention ==========
    # Inspect the primary data file header to determine if dQ is FWHM.
    # The result is stored in the state so model_builder can pass it to load4.
    primary_dq_fwhm = detect_dq_convention(state["data_file"])
    updates["dq_is_fwhm"] = primary_dq_fwhm

    # Apply to each data_files entry as well (each file may differ)
    if "data_files" in updates:
        for ds in updates["data_files"]:
            if "dq_is_fwhm" not in ds:
                ds["dq_is_fwhm"] = detect_dq_convention(ds["file"])

    # ========== 3. Parse Sample Description ==========
    if state["sample_description"]:
        # Select and load skills based on sample description
        registry = SkillRegistry()
        active_skills = select_skills(
            state["sample_description"],
            parsed_sample=None,
            registry=registry,
        )
        skill_context = load_skill_context(active_skills, registry)

        try:
            parsed = parse_sample_with_llm(
                state["sample_description"],
                hypothesis=state.get("hypothesis"),
                skill_context=skill_context,
            )
            updates["parsed_sample"] = parsed

            # Re-select skills now that we have parsed sample info
            active_skills = select_skills(
                state["sample_description"],
                parsed_sample=parsed,
                registry=registry,
            )
            updates["active_skills"] = active_skills
            updates["llm_calls"].append(
                LLMCallRecord(
                    node="intake",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    success=True,
                    used_fallback=False,
                    fallback_reason=None,
                    error=None,
                )
            )

            # Add confirmation message
            updates["messages"].append(
                Message(
                    role="assistant",
                    content=_format_parsed_summary(parsed),
                    timestamp=None,
                )
            )

        except Exception as e:
            # Non-fatal - we can still proceed with feature extraction
            updates["llm_calls"].append(
                LLMCallRecord(
                    node="intake",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    success=False,
                    used_fallback=True,
                    fallback_reason="Proceeding with feature extraction only",
                    error=str(e)[:200],
                )
            )
            updates["messages"].append(
                Message(
                    role="system",
                    content=f"Could not parse sample description: {str(e)}. Will rely on feature extraction.",
                    timestamp=None,
                )
            )

    return updates


def _format_parsed_summary(parsed: dict) -> str:
    """Format parsed sample info for display."""
    lines = ["**Understood sample structure:**"]

    if parsed.get("substrate"):
        sub = parsed["substrate"]
        lines.append(f"- Substrate: {sub['name']} (SLD = {sub['sld']:.2f})")

    if parsed.get("layers"):
        for i, layer in enumerate(parsed["layers"], 1):
            lines.append(
                f"- Layer {i}: {layer['name']} "
                f"(~{layer['thickness']:.0f} Å, SLD ≈ {layer['sld']:.2f})"
            )

    if parsed.get("ambient"):
        amb = parsed["ambient"]
        lines.append(f"- Ambient: {amb['name']} (SLD = {amb['sld']:.2f})")

    if parsed.get("constraints"):
        lines.append(f"- Constraints: {', '.join(parsed['constraints'])}")

    return "\n".join(lines)
