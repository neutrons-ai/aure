"""
INTAKE node: Load data and parse sample description.

This is the first node in the workflow. It:
1. Loads and validates the reflectivity data file
2. Parses the sample description using an LLM
3. Populates the initial state for analysis
"""

import json
import re
from typing import Dict, Any

from ..state import ReflectivityState, Message
from ..tools.data_tools import load_reflectivity_data, validate_reflectivity_data
from ..llm import llm_available, get_llm, invoke_with_timeout
from .prompts import format_sample_parse_prompt


def parse_sample_with_llm(description: str, hypothesis: str | None = None) -> Dict[str, Any]:
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
    prompt = format_sample_parse_prompt(description, hypothesis)
    
    from langchain_core.messages import HumanMessage
    response = invoke_with_timeout(llm, [HumanMessage(content=prompt)])
    
    # Extract JSON from response
    content = response.content
    
    # Try to find JSON block in the response
    json_match = re.search(r"\{[\s\S]*\}", content)
    if json_match:
        return json.loads(json_match.group())
    
    raise ValueError("Could not extract JSON from LLM response")


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
    }
    
    # ========== 1. Load Data ==========
    try:
        data = load_reflectivity_data(state["data_file"])
        updates["Q"] = data["Q"].tolist()
        updates["R"] = data["R"].tolist()
        updates["dR"] = data.get("dR", [0.0] * len(data["Q"])).tolist()
        
        # Validate
        validation = validate_reflectivity_data(
            data["Q"], data["R"], 
            data.get("dR")
        )
        
        if validation["issues"]:
            updates["messages"] = [Message(
                role="system",
                content=f"Data validation warnings: {', '.join(validation['issues'])}",
                timestamp=None
            )]
            
    except Exception as e:
        updates["error"] = f"Failed to load data: {str(e)}"
        updates["messages"] = [Message(
            role="system",
            content=f"Error loading data file: {str(e)}",
            timestamp=None
        )]
        return updates
    
    # ========== 2. Parse Sample Description ==========
    if state["sample_description"]:
        try:
            parsed = parse_sample_with_llm(
                state["sample_description"],
                hypothesis=state.get("hypothesis")
            )
            updates["parsed_sample"] = parsed
            
            # Add confirmation message
            updates["messages"].append(Message(
                role="assistant",
                content=_format_parsed_summary(parsed),
                timestamp=None
            ))
            
        except Exception as e:
            # Non-fatal - we can still proceed with feature extraction
            updates["messages"].append(Message(
                role="system",
                content=f"Could not parse sample description: {str(e)}. Will rely on feature extraction.",
                timestamp=None
            ))
    
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
