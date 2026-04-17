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

## Domain Knowledge
{skill_context}

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
        "sld": <SLD value>,
        "sld_min": <minimum SLD if user specifies a range, otherwise omit>,
        "sld_max": <maximum SLD if user specifies a range, otherwise omit>
    }},
    "constraints": ["list of any constraints mentioned"],
    "hypothesis": "the hypothesis to test, if any",
    "back_reflection": <true if neutrons come from substrate side, false otherwise>,
    "intensity": {{
        "value": <starting intensity normalization, default 1.0>,
        "min": <minimum intensity, default 0.7>,
        "max": <maximum intensity, default 1.1>,
        "fixed": <true if data is perfectly normalized and intensity should not vary>
    }}
}}

Intensity normalization:
- By default, allow intensity to vary between 0.7 and 1.1 to account for normalization uncertainty
- If user says "data is perfectly normalized" or similar, set fixed=true
- If user says "data needs large normalization correction" or similar, expand the range (e.g., 0.5 to 1.3)

HYPOTHESIZED / EXPECTED LAYERS:
- If the description mentions expected, hypothesized, or likely layers (e.g., "we expect
  lithium plating", "likely to form an SEI layer", "oxide should reduce away"), you MUST
  include these as layers in the output with reasonable initial guesses.
- Mark hypothesized layers by appending "(hypothetical)" to their name.
- Place them in the physically correct position in the layer stack.
- These layers are initial starting guesses — the fitter will optimise them.
- Do NOT omit hypothesized layers just because their thickness or composition is uncertain.

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
    skill_context: str = "",
) -> str:
    """
    Format the sample parsing prompt with the given description.

    Args:
        description: Free-form sample description from the user
        hypothesis: Optional hypothesis to test
        skill_context: Domain knowledge injected from activated skills

    Returns:
        Formatted prompt string ready for LLM invocation
    """
    hypothesis_section = ""
    if hypothesis:
        hypothesis_section = f"The hypothesis to test is: {hypothesis}"

    return SAMPLE_PARSE_PROMPT.format(
        description=description,
        hypothesis_section=hypothesis_section,
        skill_context=skill_context or "(no additional domain knowledge)",
    )


# ============================================================================
# FIT EVALUATION
# ============================================================================

FIT_EVALUATION_PROMPT = """You are evaluating the results of a neutron reflectivity fit.

## Domain Knowledge
{skill_context}

## Sample Description
{sample_description}

## User's Hypothesis (if any)
{hypothesis}

## Fit Results
- χ² (chi-squared): {chi_squared:.3f}
- Method: {method}
- Converged: {converged}

## Per-File Fit Quality (multi-file co-refinement)
{per_file_chi2}

## Best-fit Parameters
{parameters}

## Extracted Data Features
{features}

## Residual Fringe Analysis
{residual_analysis}

## Parameters at Range Boundaries
{boundary_hits}

## Model Complexity
{complexity_assessment}

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

**Acceptance threshold: χ² ≤ {chi2_max}.**
A fit with χ² within this threshold should be marked "acceptable": true.

Consider the sample description when evaluating if parameters make physical sense.

{user_criteria}
"""


def _format_residual_analysis(residual_analysis: Dict[str, Any] | None) -> str:
    """Format residual fringe analysis results for LLM prompts."""
    if not residual_analysis or not residual_analysis.get("has_residual_fringes"):
        return "  (no structured residual oscillations detected)"

    lines = []
    amp = residual_analysis.get("fringe_amplitude", 0)
    lines.append(f"  - Residual fringe amplitude (RMS): {amp:.3f}")
    lines.append(
        f"  - Number of residual fringes: {residual_analysis.get('n_residual_fringes', 0)}"
    )

    for i, t in enumerate(residual_analysis.get("unmodeled_thicknesses", [])):
        thick = t["thickness"]
        unc = t.get("uncertainty", thick * 0.2)
        conf = t.get("confidence", "low")
        method = t.get("method", "unknown")
        lines.append(
            f"  - **Unmodeled thickness {i + 1}**: ~{thick:.0f} ± {unc:.0f} Å "
            f"({conf} confidence, {method})"
        )

    lines.append("")
    lines.append(
        "  The residual (R_data / R_fit) shows periodic oscillations that the "
        "current model does not explain. These fringes indicate one or more "
        "layers are missing from the model. Consider adding a layer with the "
        "detected thickness."
    )
    return "\n".join(lines)


def _format_boundary_hits(boundary_hits: list | None) -> str:
    """Format boundary-hit info for LLM prompts."""
    if not boundary_hits:
        return "  (no parameters at range boundaries)"

    lines = []
    for bh in boundary_hits:
        lines.append(
            f"  - **{bh['name']}**: value {bh['value']:.4f} hit "
            f"{bh['bound_hit']} bound ({bh['bound_value']:.4f}). "
            f"Range has been auto-expanded for the next fit."
        )
    lines.append("")
    lines.append(
        "  Parameters hitting their range boundaries may indicate the model "
        "needs wider bounds or a structural change. Consider whether the "
        "constrained parameter should be allowed a wider range, or whether "
        "the model structure itself should be revised."
    )
    return "\n".join(lines)


def _format_complexity_assessment(
    bic: float | None = None,
    best_bic: float | None = None,
    n_params: int = 0,
    n_layers: int = 0,
) -> str:
    """Format model complexity info (BIC) for LLM prompts."""
    if bic is None:
        return "  (not computed)"

    lines = [
        f"  - Layers: {n_layers}",
        f"  - Free parameters: {n_params}",
        f"  - BIC (Bayesian Information Criterion): {bic:.1f}",
    ]
    if best_bic is not None:
        lines.append(f"  - Best BIC so far: {best_bic:.1f}")
        if bic > best_bic:
            lines.append(
                "  - **Current model is MORE complex than justified.** "
                "The simpler model with fewer layers had a better "
                "complexity-adjusted score. Prefer parameter adjustments "
                "over adding layers."
            )
    lines.append("")
    lines.append(
        "  BIC penalizes unnecessary model parameters. A lower BIC is better. "
        "Adding a layer (3 extra parameters) must produce a substantial χ² "
        "improvement to lower BIC. Do NOT suggest adding layers unless the "
        "BIC would clearly improve."
    )
    return "\n".join(lines)


def _format_per_file_chi2(per_file_results: list | None) -> str:
    """Format per-file χ² breakdown for LLM prompts."""
    if not per_file_results:
        return "  (single-file fit — not applicable)"

    lines = []
    for pf in per_file_results:
        chi2 = pf.get("chi_squared", float("inf"))
        lines.append(f"  - **{pf.get('label', '?')}**: χ² = {chi2:.3f}")
    lines.append("")
    lines.append(
        "  The data was fit jointly across multiple Q-range segments. "
        "All structural parameters are shared; each file has its own "
        "intensity normalization. If one segment fits much worse than "
        "the others, focus suggestions on the Q-range where the fit is poor."
    )
    return "\n".join(lines)


def format_fit_evaluation_prompt(
    sample_description: str,
    hypothesis: str | None,
    chi_squared: float,
    method: str,
    converged: bool,
    parameters: Dict[str, float],
    features: Dict[str, Any],
    chi2_max: float = 5.0,
    user_criteria: str = "",
    residual_analysis: Dict[str, Any] | None = None,
    boundary_hits: list | None = None,
    bic: float | None = None,
    best_bic: float | None = None,
    n_params: int = 0,
    n_layers: int = 0,
    skill_context: str = "",
    per_file_results: list | None = None,
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
        chi2_max: χ² acceptance threshold (from ``CHI2_MAX`` env var)
        user_criteria: Formatted user-defined evaluation criteria

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
            feature_lines.append(
                f"  - Estimated thickness: {features['estimated_total_thickness']:.1f} Å"
            )
        if features.get("estimated_roughness"):
            feature_lines.append(
                f"  - Estimated roughness: {features['estimated_roughness']:.1f} Å"
            )
        if features.get("estimated_n_layers"):
            feature_lines.append(
                f"  - Estimated layers: {features['estimated_n_layers']}"
            )
        if features.get("critical_edges"):
            for edge in features["critical_edges"][:2]:
                feature_lines.append(
                    f"  - Critical edge at Qc={edge.get('Qc', 0):.4f} Å⁻¹"
                )
    features_str = (
        "\n".join(feature_lines) if feature_lines else "  (no features extracted)"
    )

    return FIT_EVALUATION_PROMPT.format(
        sample_description=sample_description or "(not provided)",
        hypothesis=hypothesis or "(none)",
        chi_squared=chi_squared,
        method=method,
        converged="Yes" if converged else "No",
        parameters=params_str,
        features=features_str,
        chi2_max=chi2_max,
        user_criteria=user_criteria,
        residual_analysis=_format_residual_analysis(residual_analysis),
        boundary_hits=_format_boundary_hits(boundary_hits),
        complexity_assessment=_format_complexity_assessment(
            bic=bic,
            best_bic=best_bic,
            n_params=n_params,
            n_layers=n_layers,
        ),
        skill_context=skill_context or "(no additional domain knowledge)",
        per_file_chi2=_format_per_file_chi2(per_file_results),
    )


# ============================================================================
# MODEL REFINEMENT (LLM regenerates the full model script)
# ============================================================================

MODEL_REFINEMENT_PROMPT = """You are refining a neutron reflectivity model (refl1d script) that did not fit well enough.

## Domain Knowledge
{skill_context}

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

## Residual Fringe Analysis
{residual_analysis}

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
9. NEVER change the back-reflection/measurement geometry. If the current model uses `back_reflectivity(...)` or `back_absorption(...)`, you MUST keep it. Do NOT reverse the layer order or swap the fronting/backing media.
10. NEVER change error bars, resolution, or Q-range — these are experimental parameters.
11. Apply all domain-specific rules from the Domain Knowledge section above.

{user_constraints}

IMPORTANT: If user feedback is provided below, it takes absolute priority over
any of the rules above.  The user is the domain expert — follow their
instructions even if they contradict a default rule.

Output ONLY the Python script, no markdown fences, no explanation — just the script itself.
"""


def format_model_refinement_prompt(
    current_model: str,
    sample_description: str,
    fit_result: dict,
    features: dict,
    user_constraints: str = "",
    user_feedback: str | None = None,
    skill_context: str = "",
) -> str:
    """
    Format the model refinement prompt for the LLM.

    Args:
        current_model: Current refl1d model script
        sample_description: Original sample description from user
        fit_result: Latest fit result dict (chi2, parameters, issues, suggestions)
        features: Extracted physics features
        user_constraints: Formatted user-defined model constraints
        user_feedback: Optional text feedback from the interactive user session

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
    suggestions_str = (
        "\n".join(f"  - {s}" for s in suggestions) if suggestions else "  (none)"
    )

    # Format features
    feature_lines = []
    if features:
        if features.get("estimated_total_thickness"):
            feature_lines.append(
                f"  - Estimated thickness: {features['estimated_total_thickness']:.1f} Å"
            )
        if features.get("estimated_roughness"):
            feature_lines.append(
                f"  - Estimated roughness: {features['estimated_roughness']:.1f} Å"
            )
        if features.get("estimated_n_layers"):
            feature_lines.append(
                f"  - Estimated layers: {features['estimated_n_layers']}"
            )
        if features.get("critical_edges"):
            for edge in features["critical_edges"][:2]:
                feature_lines.append(
                    f"  - Critical edge at Qc={edge.get('Qc', 0):.4f} Å⁻¹"
                )
    features_str = "\n".join(feature_lines) if feature_lines else "  (no features)"

    # Format user feedback (interactive mode)
    feedback_section = ""
    if user_feedback:
        feedback_section = (
            "\n## User Feedback (from the scientist running this analysis)\n"
            f"{user_feedback}\n\n"
            "IMPORTANT: The user's feedback above is authoritative. Follow it "
            "even if it conflicts with any of the numbered rules above. The "
            "user is the domain expert and their instructions override all "
            "default constraints.\n"
        )

    return (
        MODEL_REFINEMENT_PROMPT.format(
            sample_description=sample_description or "(not provided)",
            current_model=current_model,
            chi_squared=fit_result.get("chi_squared", float("inf")),
            method=fit_result.get("method", "unknown"),
            converged="Yes" if fit_result.get("converged", False) else "No",
            parameters=params_str,
            issues=issues_str,
            suggestions=suggestions_str,
            features=features_str,
            user_constraints=user_constraints,
            residual_analysis=_format_residual_analysis(
                fit_result.get("residual_analysis")
            ),
            skill_context=skill_context or "(no additional domain knowledge)",
        )
        + feedback_section
    )


# ============================================================================
# MODEL REFINEMENT — JSON-based  (new)
# ============================================================================

MODEL_REFINEMENT_JSON_PROMPT = """You are refining a neutron reflectivity model that did not fit well enough.

## Domain Knowledge
{skill_context}

## Sample Description
{sample_description}

## Current Model (JSON)
```json
{current_model_json}
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

## Residual Fringe Analysis
{residual_analysis}

## Task
Generate an IMPROVED model definition that addresses the issues above.
You must output a COMPLETE, valid JSON object matching this schema:

```json
{{
  "substrate": {{
    "name": "material name",
    "sld": <SLD value>,
    "roughness": <roughness Å>,
    "roughness_max": <max roughness Å>
  }},
  "layers": [
    {{
      "name": "material name",
      "sld": <SLD value>,
      "sld_min": <minimum SLD>,
      "sld_max": <maximum SLD>,
      "thickness": <thickness Å>,
      "thickness_min": <min thickness Å>,
      "thickness_max": <max thickness Å>,
      "roughness": <roughness Å>,
      "roughness_max": <max roughness Å>
    }}
  ],
  "ambient": {{
    "name": "material name",
    "sld": <SLD value>,
    "sld_min": <minimum SLD if constrained, otherwise omit>,
    "sld_max": <maximum SLD if constrained, otherwise omit>
  }},
  "constraints": ["list of constraints"],
  "back_reflection": <true/false>,
  "data_file": "{data_file}",
  "intensity": {{
    "value": <starting intensity>,
    "min": <min intensity>,
    "max": <max intensity>,
    "fixed": <true/false>
  }}
}}
```

Layers are listed from substrate to ambient (closest to substrate first).

Rules:
1. NEVER change data_file or back_reflection — these are set by the experiment.
2. You may add layers, remove layers, change materials, adjust SLD values, or change parameter bounds.
3. If parameters are hitting their bounds, widen those bounds (sld_min/sld_max, thickness_min/thickness_max).
4. If there are systematic residuals, consider adding a layer.
5. Use best-fit parameter values as starting points where physically reasonable.
6. Unless the data is stated as perfectly normalized, keep intensity varying (fixed: false).
7. Apply all domain-specific rules from the Domain Knowledge section above.

{user_constraints}

IMPORTANT: If user feedback is provided below, it takes absolute priority over
any of the rules above.

Output ONLY the JSON object, no markdown fences, no explanation.
"""


def format_model_refinement_prompt_json(
    current_model: dict,
    sample_description: str,
    fit_result: dict,
    features: dict,
    user_constraints: str = "",
    user_feedback: str | None = None,
    skill_context: str = "",
) -> str:
    """Format the JSON-based model refinement prompt for the LLM.

    Parameters
    ----------
    current_model
        A ``ModelDefinition`` dict.
    sample_description
        Original sample description from user.
    fit_result
        Latest fit result dict.
    features
        Extracted physics features.
    user_constraints
        Formatted user-defined model constraints.
    user_feedback
        Optional text feedback from the interactive user session.

    Returns
    -------
    str
        Formatted prompt string ready for LLM invocation.
    """
    import json

    # Format parameters
    params = fit_result.get("parameters", {})
    param_lines = [f"  - {name}: {value:.4f}" for name, value in params.items()]
    params_str = "\n".join(param_lines) if param_lines else "  (no parameters)"

    # Format issues
    issues = fit_result.get("issues", [])
    issues_str = "\n".join(f"  - {issue}" for issue in issues) if issues else "  (none)"

    # Format suggestions
    suggestions = fit_result.get("suggestions", [])
    suggestions_str = (
        "\n".join(f"  - {s}" for s in suggestions) if suggestions else "  (none)"
    )

    # Format features
    feature_lines = []
    if features:
        if features.get("estimated_total_thickness"):
            feature_lines.append(
                f"  - Estimated thickness: {features['estimated_total_thickness']:.1f} Å"
            )
        if features.get("estimated_roughness"):
            feature_lines.append(
                f"  - Estimated roughness: {features['estimated_roughness']:.1f} Å"
            )
        if features.get("estimated_n_layers"):
            feature_lines.append(
                f"  - Estimated layers: {features['estimated_n_layers']}"
            )
        if features.get("critical_edges"):
            for edge in features["critical_edges"][:2]:
                feature_lines.append(
                    f"  - Critical edge at Qc={edge.get('Qc', 0):.4f} Å⁻¹"
                )
    features_str = "\n".join(feature_lines) if feature_lines else "  (no features)"

    # Serialize current model (strip fitted params for cleaner prompt)
    model_for_prompt = {
        k: v
        for k, v in current_model.items()
        if k not in ("fitted_parameters", "fitted_uncertainties")
    }
    current_model_json = json.dumps(model_for_prompt, indent=2)

    data_file = current_model.get("data_file", "")

    # Format user feedback
    feedback_section = ""
    if user_feedback:
        feedback_section = (
            "\n## User Feedback (from the scientist running this analysis)\n"
            f"{user_feedback}\n\n"
            "IMPORTANT: The user's feedback above is authoritative. Follow it "
            "even if it conflicts with any of the numbered rules above. The "
            "user is the domain expert and their instructions override all "
            "default constraints.\n"
        )

    return (
        MODEL_REFINEMENT_JSON_PROMPT.format(
            sample_description=sample_description or "(not provided)",
            current_model_json=current_model_json,
            chi_squared=fit_result.get("chi_squared", float("inf")),
            method=fit_result.get("method", "unknown"),
            converged="Yes" if fit_result.get("converged", False) else "No",
            parameters=params_str,
            issues=issues_str,
            suggestions=suggestions_str,
            features=features_str,
            data_file=data_file,
            user_constraints=user_constraints,
            residual_analysis=_format_residual_analysis(
                fit_result.get("residual_analysis")
            ),
            skill_context=skill_context or "(no additional domain knowledge)",
        )
        + feedback_section
    )
