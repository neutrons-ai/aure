"""
FastMCP Server for Reflectivity Analysis Workflow.

This MCP server exposes the reflectivity analysis workflow as tools
that can be used by AI assistants like Claude to help users analyze
neutron reflectivity data.

To run:
    python -m aure.mcp_server

Or with the CLI:
    python -m aure.cli mcp-server
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file (for LangSmith tracing, LLM config, etc.)
load_dotenv()

from fastmcp import FastMCP

# Import workflow components
from .workflow import create_workflow, run_analysis
from .state import create_initial_state, ReflectivityState
from .database.materials import get_sld, lookup_material
from .tools.feature_tools import (
    estimate_total_thickness,
    estimate_roughness,
    extract_critical_edges,
)

# Create the FastMCP server
mcp = FastMCP(
    name="reflectivity-analysis",
    instructions="Tools for analyzing neutron reflectivity data and fitting models",
)

# Store active sessions
_sessions: dict[str, ReflectivityState] = {}


# ============================================================================
# Material Database Tools
# ============================================================================

@mcp.tool()
def lookup_material_sld(
    material: str,
    wavelength: float = 1.8,
) -> dict:
    """
    Look up the scattering length density (SLD) for a material.
    
    Args:
        material: Material name (e.g., "silicon", "gold", "D2O") or 
                  chemical formula (e.g., "SiO2", "Fe2O3")
        wavelength: Neutron wavelength in Angstroms (default 1.8 Å)
    
    Returns:
        Dictionary with SLD value and material info
    """
    try:
        info = lookup_material(material)
        sld = get_sld(material)
        return {
            "material": material,
            "sld": round(sld, 4),
            "sld_unit": "10^-6 Å^-2",
            "density": info.density if info else None,
            "formula": info.formula if info else material,
            "wavelength": wavelength,
        }
    except Exception as e:
        return {"error": str(e), "material": material}


@mcp.tool()
def compare_materials(
    materials: list[str],
    wavelength: float = 1.8,
) -> list[dict]:
    """
    Compare SLD values for multiple materials.
    
    Useful for understanding contrast between layers.
    
    Args:
        materials: List of material names or formulas
        wavelength: Neutron wavelength in Angstroms
    
    Returns:
        List of material info dictionaries sorted by SLD
    """
    results = []
    for mat in materials:
        result = lookup_material_sld(mat, wavelength)
        results.append(result)
    
    # Sort by SLD
    results.sort(key=lambda x: x.get("sld", 0))
    return results


# ============================================================================
# Data Analysis Tools
# ============================================================================

@mcp.tool()
def analyze_reflectivity_features(
    q_values: list[float],
    r_values: list[float],
) -> dict:
    """
    Extract physics features from reflectivity data.
    
    Analyzes the curve to estimate:
    - Critical edge position (Qc)
    - Total film thickness from fringe spacing
    - Surface/interface roughness from high-Q decay
    - Number of visible fringes
    
    Args:
        q_values: Q values in inverse Angstroms
        r_values: Reflectivity values (normalized, 0-1)
    
    Returns:
        Dictionary of extracted features
    """
    import numpy as np
    
    Q = np.array(q_values)
    R = np.array(r_values)
    
    features = {}
    
    # Find critical edge
    try:
        qc_result = extract_critical_edges(Q, R)
        if qc_result.get("critical_edges"):
            edge = qc_result["critical_edges"][0]
            features["critical_edge"] = {
                "Qc": round(edge["Qc"], 5),
                "estimated_SLD": round(edge.get("estimated_SLD", 0), 3),
                "confidence": edge.get("confidence", "unknown"),
            }
        else:
            features["critical_edge"] = {"error": "No critical edge found"}
    except Exception as e:
        features["critical_edge"] = {"error": str(e)}
    
    # Estimate thickness
    try:
        thickness_result = estimate_total_thickness(Q, R)
        features["thickness"] = {
            "estimated_total": round(thickness_result["estimated_total_thickness"], 1),
            "unit": "Angstrom",
            "n_fringes": thickness_result.get("n_fringes", 0),
            "confidence": thickness_result.get("confidence", "unknown"),
        }
    except Exception as e:
        features["thickness"] = {"error": str(e)}
    
    # Estimate roughness
    try:
        roughness_result = estimate_roughness(Q, R)
        features["roughness"] = {
            "estimated": round(roughness_result["estimated_roughness"], 1),
            "unit": "Angstrom",
            "confidence": roughness_result.get("confidence", "unknown"),
        }
    except Exception as e:
        features["roughness"] = {"error": str(e)}
    
    # Data range
    features["data_range"] = {
        "q_min": round(float(Q.min()), 5),
        "q_max": round(float(Q.max()), 5),
        "n_points": len(Q),
    }
    
    return features


# ============================================================================
# Workflow Session Tools
# ============================================================================

@mcp.tool()
def start_analysis_session(
    data_file: str,
    sample_description: str,
    hypothesis: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict:
    """
    Start a new reflectivity analysis session.
    
    This initializes the workflow state and runs the initial
    intake and analysis steps.
    
    Args:
        data_file: Path to the reflectivity data file
        sample_description: Natural language description of the sample
                           (e.g., "100 nm polystyrene on silicon")
        hypothesis: Optional hypothesis to test
        session_id: Optional session ID (auto-generated if not provided)
    
    Returns:
        Session info including extracted features and initial model
    """
    import uuid
    
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]
    
    # Check if data file exists
    if not os.path.exists(data_file):
        return {"error": f"Data file not found: {data_file}"}
    
    # Create initial state and run workflow (without fitting)
    try:
        state = create_initial_state(
            data_file=data_file,
            sample_description=sample_description,
            hypothesis=hypothesis,
        )
        
        workflow = create_workflow(include_fitting=False)
        final_state = workflow.invoke(state)
        
        # Store session
        _sessions[session_id] = final_state
        
        # Return summary
        return {
            "session_id": session_id,
            "status": "initialized",
            "data_loaded": len(final_state.get("Q", [])) > 0,
            "n_points": len(final_state.get("Q", [])),
            "parsed_sample": final_state.get("parsed_sample"),
            "extracted_features": _summarize_features(
                final_state.get("extracted_features")
            ),
            "model_generated": final_state.get("current_model") is not None,
            "error": final_state.get("error"),
        }
    except Exception as e:
        return {"error": str(e), "session_id": session_id}


@mcp.tool()
def get_session_model(session_id: str) -> dict:
    """
    Get the current refl1d model script for a session.
    
    Args:
        session_id: The session ID from start_analysis_session
    
    Returns:
        The model script and session info
    """
    if session_id not in _sessions:
        return {"error": f"Session not found: {session_id}"}
    
    state = _sessions[session_id]
    return {
        "session_id": session_id,
        "model_script": state.get("current_model"),
        "iteration": state.get("iteration", 0),
    }


@mcp.tool()
def run_fit(
    session_id: str,
    method: str = "lm",
    max_iterations: int = 200,
) -> dict:
    """
    Run the fitting algorithm on the current model.
    
    Args:
        session_id: The session ID
        method: Fitting method - 'lm' (fast), 'de' (global), 'dream' (MCMC)
        max_iterations: Maximum fitting iterations
    
    Returns:
        Fit results including chi-squared and parameters
    """
    if session_id not in _sessions:
        return {"error": f"Session not found: {session_id}"}
    
    state = _sessions[session_id]
    
    if state.get("current_model") is None:
        return {"error": "No model available. Run start_analysis_session first."}
    
    try:
        from .nodes.fitting import run_refl1d_fit
        
        result = run_refl1d_fit(
            model_script=state["current_model"],
            data_file=state["data_file"],
            method=method,
            max_iterations=max_iterations,
        )
        
        # Update state
        state["fit_result"] = result
        state["current_chi2"] = result.get("chi_squared")
        
        return {
            "session_id": session_id,
            "success": result.get("success", False),
            "chi_squared": result.get("chi_squared"),
            "method": method,
            "parameters": result.get("parameters"),
            "uncertainties": result.get("uncertainties"),
        }
    except Exception as e:
        return {"error": str(e), "session_id": session_id}


@mcp.tool()
def evaluate_fit(session_id: str) -> dict:
    """
    Evaluate the quality of the current fit.
    
    Checks chi-squared, residual patterns, and parameter reasonableness.
    
    Args:
        session_id: The session ID
    
    Returns:
        Evaluation results with issues and suggestions
    """
    if session_id not in _sessions:
        return {"error": f"Session not found: {session_id}"}
    
    state = _sessions[session_id]
    
    if state.get("fit_result") is None:
        return {"error": "No fit result. Run run_fit first."}
    
    try:
        from .nodes.evaluation import analyze_fit_quality
        
        evaluation = analyze_fit_quality(
            chi_squared=state["fit_result"].get("chi_squared", 999),
            residuals=state["fit_result"].get("residuals", []),
            parameters=state["fit_result"].get("parameters", {}),
            parsed_sample=state.get("parsed_sample", {}),
        )
        
        state["evaluation"] = evaluation
        
        return {
            "session_id": session_id,
            "acceptable": evaluation.get("acceptable", False),
            "chi_squared_quality": evaluation.get("chi_squared_quality"),
            "issues": evaluation.get("issues", []),
            "suggestions": evaluation.get("suggestions", []),
        }
    except Exception as e:
        return {"error": str(e), "session_id": session_id}


@mcp.tool()
def modify_model(
    session_id: str,
    modification: str,
    value: Optional[float] = None,
    layer_index: Optional[int] = None,
) -> dict:
    """
    Modify the current model.
    
    Available modifications:
    - "widen_bounds": Increase parameter ranges by 50%
    - "add_layer": Add an additional layer to the model
    - "increase_roughness": Allow higher roughness values
    - "set_thickness": Set layer thickness (requires value and layer_index)
    - "set_sld": Set layer SLD (requires value and layer_index)
    
    Args:
        session_id: The session ID
        modification: Type of modification
        value: Optional value for modifications that require it
        layer_index: Layer index for per-layer modifications (0 = first layer from substrate)
    
    Returns:
        Updated model info
    """
    if session_id not in _sessions:
        return {"error": f"Session not found: {session_id}"}
    
    state = _sessions[session_id]
    
    if state.get("current_model") is None:
        return {"error": "No model available."}
    
    try:
        from .nodes.refinement import (
            _widen_bounds,
            _increase_roughness_bounds,
            _add_layer,
            _set_layer_thickness,
            _set_layer_sld,
        )
        
        model = state["current_model"]
        
        if modification == "widen_bounds":
            model, success = _widen_bounds(model, state.get("fit_result", {}).get("parameters", {}))
            msg = "Widened parameter bounds by 50%" if success else "No changes made"
        elif modification == "increase_roughness":
            model, success = _increase_roughness_bounds(model)
            msg = "Increased roughness limits" if success else "No changes made"
        elif modification == "add_layer":
            model, success = _add_layer(model, state)
            msg = "Added additional layer" if success else "Could not add layer"
        elif modification == "set_thickness":
            if value is None or layer_index is None:
                return {"error": "set_thickness requires both 'value' and 'layer_index'"}
            model, success = _set_layer_thickness(model, layer_index, value)
            msg = f"Set layer {layer_index} thickness to {value} Å" if success else f"Could not find layer {layer_index}"
        elif modification == "set_sld":
            if value is None or layer_index is None:
                return {"error": "set_sld requires both 'value' and 'layer_index'"}
            model, success = _set_layer_sld(model, layer_index, value)
            msg = f"Set layer {layer_index} SLD to {value} × 10⁻⁶ Å⁻²" if success else f"Could not find layer {layer_index}"
        else:
            return {"error": f"Unknown modification: {modification}"}
        
        if not success:
            return {
                "session_id": session_id,
                "modification": modification,
                "message": msg,
                "success": False,
            }
        
        state["current_model"] = model
        state["iteration"] = state.get("iteration", 0) + 1
        
        return {
            "session_id": session_id,
            "modification": modification,
            "message": msg,
            "iteration": state["iteration"],
            "success": True,
        }
    except Exception as e:
        return {"error": str(e), "session_id": session_id}


@mcp.tool()
def list_sessions() -> dict:
    """
    List all active analysis sessions.
    
    Returns:
        Dictionary of session IDs and their status
    """
    sessions = {}
    for sid, state in _sessions.items():
        sessions[sid] = {
            "data_file": state.get("data_file"),
            "sample": state.get("sample_description", "")[:50],
            "iteration": state.get("iteration", 0),
            "chi_squared": state.get("current_chi2"),
            "has_fit": state.get("fit_result") is not None,
        }
    return {"sessions": sessions, "count": len(sessions)}


# ============================================================================
# Quick Analysis Tool
# ============================================================================

@mcp.tool()
def quick_analyze(
    data_file: str,
    sample_description: str,
    max_iterations: int = 5,
) -> dict:
    """
    Run a complete analysis in one step.
    
    This is a convenience function that runs the full workflow
    including fitting and refinement, returning a summary of results.
    
    Args:
        data_file: Path to reflectivity data file
        sample_description: Description of the sample
        max_iterations: Maximum number of refinement iterations (default: 5)
    
    Returns:
        Complete analysis results
    """
    try:
        result = run_analysis(
            data_file=data_file,
            sample_description=sample_description,
            max_iterations=max_iterations,
        )
        
        return {
            "success": result.get("error") is None,
            "n_points": len(result.get("Q", [])),
            "parsed_sample": result.get("parsed_sample"),
            "features": _summarize_features(result.get("extracted_features")),
            "model_generated": result.get("current_model") is not None,
            "fit_result": result.get("fit_result"),
            "evaluation": result.get("evaluation"),
            "error": result.get("error"),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# Helper Functions
# ============================================================================

def _summarize_features(features: Optional[dict]) -> Optional[dict]:
    """Create a concise summary of extracted features."""
    if features is None:
        return None
    
    return {
        "estimated_layers": features.get("estimated_n_layers"),
        "estimated_thickness_A": features.get("estimated_total_thickness"),
        "estimated_roughness_A": features.get("estimated_roughness"),
        "q_range": f"{features.get('q_min', 0):.4f} - {features.get('q_max', 0):.4f}",
        "n_fringes": features.get("n_fringes", 0),
    }


# ============================================================================
# Server Entry Point
# ============================================================================

def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
