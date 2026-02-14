"""
LLM prompts for the reflectivity analysis workflow.

This module contains prompt templates for:
- Parsing sample descriptions into structured data
- Generating model reasoning
- Interpreting fit results
- Suggesting refinements

Note: LLM invocation is handled separately in the workflow nodes using
the llm module. This file only contains prompt templates and formatting helpers.
"""

from typing import Dict, Any


# ============================================================================
# SAMPLE DESCRIPTION PARSING
# ============================================================================

SAMPLE_PARSE_PROMPT = """You are analyzing a neutron reflectivity experiment. 
The user describes their sample as:

"{description}"

{hypothesis_section}

Extract the following information in JSON format:
{{
    "substrate": {{
        "name": "material name",
        "sld": <SLD value in 10^-6 Å^-2>,
        "roughness": <estimated roughness in Å, default 3>
    }},
    "layers": [
        {{
            "name": "material name",
            "sld": <SLD value>,
            "sld_min": <minimum possible SLD>,
            "sld_max": <maximum possible SLD>,
            "thickness": <thickness in Å>,
            "roughness": <roughness in Å, default 5>
        }}
    ],
    "ambient": {{
        "name": "material name (air, D2O, H2O, THF, dTHF, etc.)",
        "sld": <SLD value>
    }},
    "constraints": ["list of any constraints mentioned"],
    "hypothesis": "the hypothesis to test, if any",
    "back_reflection": <true if neutrons come from substrate side, false otherwise>,
    "intensity": {{
        "value": <starting intensity normalization, default 1.0>,
        "min": <minimum intensity, default 0.9>,
        "max": <maximum intensity, default 1.1>,
        "fixed": <true if data is perfectly normalized and intensity should not vary>
    }}
}}

Intensity normalization:
- By default, allow intensity to vary ±10% (0.9 to 1.1) to account for normalization uncertainty
- If user says "data is perfectly normalized" or similar, set fixed=true
- If user says "data needs large normalization correction" or similar, expand the range (e.g., 0.7 to 1.3)

Common SLD values (10^-6 Å^-2):
- Silicon: 2.07
- D2O: 6.36
- H2O: -0.56
- Air: 0.0
- SiO2: 3.47
- Gold: 4.5
- Copper: 6.55
- Titanium: -1.95
- Polystyrene: 1.4
- d-Polystyrene: 6.4
- THF (protonated): 0.18
- dTHF (deuterated): 6.35
- Toluene: 0.94
- d-Toluene: 5.66

SLD RANGES:
- Set sld_min and sld_max to at least ±1.0 around the nominal SLD value for each layer.
  For example, for copper (SLD 6.55): sld_min = 5.5, sld_max = 7.5.
  For titanium (SLD -1.95): sld_min = -3.0, sld_max = -1.0.
- This allows the fitter enough freedom to find the correct values even when the
  material is not perfectly stoichiometric or has some intermixing.
- Never use ranges narrower than ±0.5.

IMPORTANT:
- If thickness is given in nm, convert to Å (1 nm = 10 Å).
- If the user mentions neutrons coming from the substrate side, or back reflection, 
  or measuring through the substrate, set back_reflection to true.
- Pay attention to what the ambient medium is - it may be a solvent like THF, not air.
- If a value is not specified, use reasonable defaults based on the material.
"""


def format_sample_parse_prompt(
    description: str,
    hypothesis: str | None = None,
) -> str:
    """
    Format the sample parsing prompt with the given description.
    
    Args:
        description: Free-form sample description from the user
        hypothesis: Optional hypothesis to test
    
    Returns:
        Formatted prompt string ready for LLM invocation
    """
    hypothesis_section = ""
    if hypothesis:
        hypothesis_section = f"The hypothesis to test is: {hypothesis}"
    
    return SAMPLE_PARSE_PROMPT.format(
        description=description,
        hypothesis_section=hypothesis_section,
    )


# ============================================================================
# FIT EVALUATION
# ============================================================================

FIT_EVALUATION_PROMPT = """You are evaluating the results of a neutron reflectivity fit.

## Sample Description
{sample_description}

## User's Hypothesis (if any)
{hypothesis}

## Fit Results
- χ² (chi-squared): {chi_squared:.3f}
- Method: {method}
- Converged: {converged}

## Best-fit Parameters
{parameters}

## Extracted Data Features
{features}

## Task
Analyze the fit quality and determine:
1. Is this fit acceptable for the user's goals?
2. Are the fitted parameters physically reasonable?
3. Are there any issues or concerns?
4. What specific improvements could be made?

Respond in JSON format:
{{
    "acceptable": <true/false - is this fit good enough to report?>,
    "quality_assessment": "<brief assessment: excellent/good/marginal/poor>",
    "issues": ["<list of specific issues identified>"],
    "suggestions": ["<list of actionable suggestions for improvement>"],
    "physical_concerns": ["<any physically unreasonable parameter values>"],
    "hypothesis_addressed": "<how well does this fit address the user's hypothesis, if any>",
    "needs_user_guidance": <true/false - should we ask the user before proceeding?>
}}

Guidelines for χ²:
- χ² ≈ 1: Ideal fit (model matches data within error bars)
- χ² < 0.5: Possible overfitting or overestimated errors
- χ² 1-2: Excellent fit
- χ² 2-5: Good fit, minor discrepancies
- χ² 5-10: Marginal fit, model may be missing features
- χ² > 10: Poor fit, significant model problems

Consider the sample description when evaluating if parameters make physical sense.

IMPORTANT CONSTRAINTS:
- NEVER suggest changing the fitting engine/method (e.g., switching to Levenberg-Marquardt, differential evolution, etc.). The fitting method is chosen by the workflow and is not a model issue.
- If suggesting adding a native oxide layer (e.g., SiO2 on silicon), constrain it to: thickness < 40 Å (4 nm) and SLD between 1.0 and 4.3 × 10⁻⁶ Å⁻².
- NEVER suggest reversing the layer order or changing the back-reflection geometry. The measurement geometry (which side the neutrons come from) is set by the user and must not be changed. If the sample description says neutrons come from the substrate side, that is correct.
- NEVER suggest changing error bars, resolution, or Q-range — these are experimental parameters that cannot be modified.
- If a metal layer (e.g., copper, gold, iron, nickel, aluminum) is in contact with the ambient medium (air, solvent, etc.) and no oxide layer is already present, suggest adding a thin native metal oxide layer (10–30 Å) between the metal and the ambient. Metals exposed to air or solvent almost always form a native oxide. Common examples: CuO or Cu₂O on copper, NiO on nickel, Al₂O₃ on aluminum, TiO₂ on titanium.
"""


def format_fit_evaluation_prompt(
    sample_description: str,
    hypothesis: str | None,
    chi_squared: float,
    method: str,
    converged: bool,
    parameters: Dict[str, float],
    features: Dict[str, Any],
) -> str:
    """
    Format the fit evaluation prompt.
    
    Args:
        sample_description: Original sample description from user
        hypothesis: User's hypothesis (if any)
        chi_squared: Fit chi-squared value
        method: Fitting method used
        converged: Whether fit converged
        parameters: Best-fit parameter values
        features: Extracted physics features
    
    Returns:
        Formatted prompt string
    """
    # Format parameters as readable string
    param_lines = []
    for name, value in parameters.items():
        param_lines.append(f"  - {name}: {value:.4f}")
    params_str = "\n".join(param_lines) if param_lines else "  (no parameters)"
    
    # Format features as readable string
    feature_lines = []
    if features:
        if features.get("estimated_total_thickness"):
            feature_lines.append(f"  - Estimated thickness: {features['estimated_total_thickness']:.1f} Å")
        if features.get("estimated_roughness"):
            feature_lines.append(f"  - Estimated roughness: {features['estimated_roughness']:.1f} Å")
        if features.get("estimated_n_layers"):
            feature_lines.append(f"  - Estimated layers: {features['estimated_n_layers']}")
        if features.get("critical_edges"):
            for edge in features["critical_edges"][:2]:
                feature_lines.append(f"  - Critical edge at Qc={edge.get('Qc', 0):.4f} Å⁻¹")
    features_str = "\n".join(feature_lines) if feature_lines else "  (no features extracted)"
    
    return FIT_EVALUATION_PROMPT.format(
        sample_description=sample_description or "(not provided)",
        hypothesis=hypothesis or "(none)",
        chi_squared=chi_squared,
        method=method,
        converged="Yes" if converged else "No",
        parameters=params_str,
        features=features_str,
    )


# ============================================================================
# MODEL REFINEMENT (LLM regenerates the full model script)
# ============================================================================

MODEL_REFINEMENT_PROMPT = """You are refining a neutron reflectivity model (refl1d script) that did not fit well enough.

## Sample Description
{sample_description}

## Current Model Script
```python
{current_model}
```

## Fit Results
- χ² (chi-squared): {chi_squared:.3f}
- Method: {method}
- Converged: {converged}

## Best-fit Parameters (from fitting)
{parameters}

## Issues Identified
{issues}

## Suggestions for Improvement
{suggestions}

## Physics Features from Data
{features}

## Task
Generate an IMPROVED refl1d model script that addresses the issues above.
You must output a COMPLETE, valid refl1d Python script (not a partial edit).

Rules:
1. Keep the same data file path and probe loading.
2. You may add layers, remove layers, change materials, adjust SLD values, change parameter bounds, or add constraints.
3. If parameters are hitting their bounds, widen those bounds.
4. If there are systematic residuals, consider adding an interface layer or adjusting the model structure.
5. Use the best-fit parameter values as starting points for the refined model where they are physically reasonable.
6. Always include `probe.intensity.range(...)` for normalization.
7. The script must end with `experiment = Experiment(probe=probe, sample=sample)` and `problem = FitProblem(experiment)`.
8. NEVER change the fitting engine/method. The fitting method is chosen by the workflow — focus only on the model.
9. If adding a native oxide layer (e.g., SiO2 on silicon), its thickness must be < 40 Å (4 nm) with SLD between 1.0 and 4.3 × 10⁻⁶ Å⁻².
10. NEVER change the back-reflection/measurement geometry. If the current model uses `back_reflectivity(...)` or `back_absorption(...)`, you MUST keep it. Do NOT reverse the layer order or swap the fronting/backing media. The geometry is determined by the physical experiment and is NOT a fitting parameter.
11. NEVER change error bars, resolution, or Q-range — these are experimental parameters.
12. Use SLD ranges of at least ±1.0 around nominal values for each material to give the fitter sufficient freedom.
13. If a metal layer is in contact with the ambient medium and no oxide layer is already present, add a thin native metal oxide layer (10–30 Å) between the metal and the ambient. Metals exposed to air or solvent almost always form a native oxide. Common examples: CuO or Cu₂O on copper (SLD ~4–6 ×10⁻⁶ Å⁻²), NiO on nickel, Al₂O₃ on aluminum, TiO₂ on titanium.
14. CRITICAL refl1d API rule: `SLD(...)` objects do NOT have `.material`, `.thickness`, or `.interface` attributes. Those attributes only exist on `Slab` objects inside the sample stack. You MUST set parameter bounds using `sample[i]` indexing, for example:
      sample[0].material.rho.range(5.5, 7.0)   # ambient SLD
      sample[1].thickness.range(10.0, 30.0)     # first layer thickness
      sample[1].material.rho.range(2.0, 4.0)    # first layer SLD
      sample[1].interface.range(0.0, 5.0)       # first layer roughness
    NEVER write `copper.material.rho.range(...)` or `ambient.material.rho.range(...)` — this will crash with "'SLD' object has no attribute 'material'".

Output ONLY the Python script, no markdown fences, no explanation — just the script itself.
"""


def format_model_refinement_prompt(
    current_model: str,
    sample_description: str,
    fit_result: dict,
    features: dict,
) -> str:
    """
    Format the model refinement prompt for the LLM.
    
    Args:
        current_model: Current refl1d model script
        sample_description: Original sample description from user
        fit_result: Latest fit result dict (chi2, parameters, issues, suggestions)
        features: Extracted physics features
        
    Returns:
        Formatted prompt string
    """
    # Format parameters
    params = fit_result.get("parameters", {})
    param_lines = [f"  - {name}: {value:.4f}" for name, value in params.items()]
    params_str = "\n".join(param_lines) if param_lines else "  (no parameters)"
    
    # Format issues
    issues = fit_result.get("issues", [])
    issues_str = "\n".join(f"  - {issue}" for issue in issues) if issues else "  (none)"
    
    # Format suggestions
    suggestions = fit_result.get("suggestions", [])
    suggestions_str = "\n".join(f"  - {s}" for s in suggestions) if suggestions else "  (none)"
    
    # Format features
    feature_lines = []
    if features:
        if features.get("estimated_total_thickness"):
            feature_lines.append(f"  - Estimated thickness: {features['estimated_total_thickness']:.1f} Å")
        if features.get("estimated_roughness"):
            feature_lines.append(f"  - Estimated roughness: {features['estimated_roughness']:.1f} Å")
        if features.get("estimated_n_layers"):
            feature_lines.append(f"  - Estimated layers: {features['estimated_n_layers']}")
        if features.get("critical_edges"):
            for edge in features["critical_edges"][:2]:
                feature_lines.append(f"  - Critical edge at Qc={edge.get('Qc', 0):.4f} Å⁻¹")
    features_str = "\n".join(feature_lines) if feature_lines else "  (no features)"
    
    return MODEL_REFINEMENT_PROMPT.format(
        sample_description=sample_description or "(not provided)",
        current_model=current_model,
        chi_squared=fit_result.get("chi_squared", float('inf')),
        method=fit_result.get("method", "unknown"),
        converged="Yes" if fit_result.get("converged", False) else "No",
        parameters=params_str,
        issues=issues_str,
        suggestions=suggestions_str,
        features=features_str,
    )
