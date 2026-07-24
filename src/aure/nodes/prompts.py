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

from ..tools.feature_tools import format_critical_edge_line


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
    }},
    "background": {{
        "enabled": <true ONLY if the user asks to fit/add a flat background, else false>,
        "min": <minimum background, default 0.0>,
        "max": <maximum background, default 1e-5>
    }}
}}

Intensity normalization:
- By default, allow intensity to vary between 0.7 and 1.1 to account for normalization uncertainty
- If user says "data is perfectly normalized" or similar, set fixed=true
- If user says "data needs large normalization correction" or similar, expand the range (e.g., 0.5 to 1.3)

Flat background:
- A constant incoherent background added to the reflectivity (R_total = intensity*R + background).
- Set background.enabled=true ONLY when the user explicitly asks to fit or add a background
  (e.g. "fit a flat background", "account for the background", "the data isn't background-subtracted").
  Otherwise leave it disabled. When enabled, a single background is fit and tied across all
  data files of each measurement state.

EXPECTED / TENTATIVE LAYERS (keep these OUT of the baseline):
- The "layers" list is the BASELINE structure — include ONLY layers the user
  states are actually present.
- If the description or hypothesis merely suggests a layer MIGHT be present
  (e.g., "we expect lithium plating", "there may be a native oxide", "likely
  to form an SEI layer", "oxide should reduce away"), do NOT add it to
  "layers". Tentative layers are handled separately as ranked structural
  hypotheses and are tested only if the baseline fit needs them.
- When unsure whether a layer is confirmed-present or merely expected, leave
  it OUT of the baseline.

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
# STRUCTURAL HYPOTHESIS RANKING
# ============================================================================

STRUCTURAL_HYPOTHESIS_PROMPT = """You are producing a ranked list of candidate
structural changes for a neutron reflectivity analysis. The analysis will
start by fitting the baseline model below; this list is consulted only when
parameter-only refinement fails to reach the χ² acceptance threshold.

## Domain Knowledge (active skills)
{skill_context}

## Sample Description
{sample_description}

## Baseline Parsed Model
- Substrate: {substrate}
- Layers (substrate → ambient): {layers}
- Ambient: {ambient}
- Back-reflection geometry: {back_reflection}

## User's Stated Hypothesis
{user_hypothesis}

## Task

Follow the `structural-hypothesis-ranking` skill. Enumerate plausible
structural changes to this baseline model, ranked by expected value.

If the user stated a hypothesis or a tentative ("may be", "expected",
"likely to form") layer above, you MUST turn it into one (or more)
hypotheses and rank it at the TOP of the list (highest expected value),
reformatted into the standard fields below, with `skill_source` set to
"user". Reword it to align with the workflow but preserve the user's intent.

Hypotheses may also be *reinterpretations* — not every hypothesis adds or
removes a layer. A reinterpretation re-labels an EXISTING material's SLD/isotope
rather than changing the layer stack. The most important one: when the ambient
is a liquid whose isotope was not specified, propose reinterpreting it as a
**deuterated solvent** (ambient SLD ≈ 0 → the D-form value, over a wide range
spanning both). List this even when the layer stack already looks complete — the
missing piece is the ambient's isotope, not a layer. Rank it high when the data
could show a critical edge / strong low-Q feature.

For each hypothesis, produce:
- `title`: one short line
- `rationale`: one sentence citing an active skill by name (or the user)
- `change`: the concrete edit in neutral terms — for a layer change: insertion
  point, thickness range Å, SLD range 10⁻⁶ Å⁻², roughness range Å; for a
  reinterpretation: the material and its new SLD value + range (no layer added)
- `skill_source`: name of the skill motivating the hypothesis, or "user"

Return 2–6 hypotheses in rank order. If no structural change OR reinterpretation
is plausible (e.g. the user has specified a complete model in a non-liquid
ambient and all relevant skills are satisfied), return an empty list.

Apply the ranking criteria and the "avoid unless justified" list from the
`structural-hypothesis-ranking` skill. Do NOT include changes to the
back-reflection geometry, the stacking order, the data file, or fitting
method.

Respond with ONLY a JSON array of objects:

```json
[
  {{
    "title": "Add native CuO on top of Cu",
    "rationale": "metal-oxide-interfaces: an outermost Cu layer exposed to D2O develops a 10-50 A native oxide unless otherwise stated",
    "change": "insert a CuO layer of 10-50 A (SLD 4.5-5.5) between Cu and D2O, roughness 3-15 A",
    "skill_source": "metal-oxide-interfaces"
  }}
]
```

Output ONLY the JSON array, no markdown fences, no other text.
"""


def format_structural_hypothesis_prompt(
    sample_description: str,
    parsed_sample: dict,
    skill_context: str,
    hypothesis: str | None = None,
) -> str:
    """Format the prompt that asks the LLM to produce ranked structural
    hypotheses from the parsed sample and active skill bodies.

    The prompt is intentionally decoupled from the initial sample parse so
    that the LLM sees the concrete baseline model (not a free-form
    description) when reasoning about what might be missing.

    The user's stated ``hypothesis`` (if any) is surfaced as its own section
    so the LLM turns it into one or more top-ranked candidate changes rather
    than baking it into the baseline.
    """
    sub = parsed_sample.get("substrate", {}) or {}
    amb = parsed_sample.get("ambient", {}) or {}
    layers = parsed_sample.get("layers", []) or []
    layer_str = (
        ", ".join(
            f"{l.get('name', '?')} (~{l.get('thickness', '?')} Å, SLD {l.get('sld', '?')})"
            for l in layers
        )
        or "(none)"
    )
    return STRUCTURAL_HYPOTHESIS_PROMPT.format(
        skill_context=skill_context or "(no additional domain knowledge)",
        sample_description=sample_description or "(not provided)",
        substrate=f"{sub.get('name', '?')} (SLD {sub.get('sld', '?')})",
        layers=layer_str,
        ambient=f"{amb.get('name', '?')} (SLD {amb.get('sld', '?')})",
        back_reflection=parsed_sample.get("back_reflection", False),
        user_hypothesis=(hypothesis or "").strip() or "(none stated)",
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

## χ² / BIC Trajectory (iterations in order)
{trajectory}

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

## Ranked Structural Hypotheses
{structural_hypotheses}

## Task
Analyze the fit quality and determine:
1. Is this fit acceptable for the user's goals?
2. Are the fitted parameters physically reasonable?
3. Are there any issues or concerns?
4. What specific improvements could be made?

Use the trajectory and the hypothesis list to choose between a parameter
change and a structural change — follow the `structural-hypothesis-ranking`
skill. If parameter tweaks have stopped making meaningful progress and
there is a `pending` hypothesis, your suggestion should be to realize that
hypothesis (cite it by title). If a hypothesis was just `rejected` by the
BIC guardrail, do not re-propose it.

Respond in JSON format:
{{
    "acceptable": <true/false - is this fit good enough to report?>,
    "quality_assessment": "<brief assessment: excellent/good/marginal/poor>",
    "issues": ["<list of specific issues identified>"],
    "suggestions": ["<list of actionable suggestions for improvement>"],
    "physical_concerns": ["<any physically unreasonable parameter values>"],
    "hypothesis_addressed": "<how well does this fit address the user's hypothesis, if any>",
    "needs_user_guidance": <true/false - should we ask the user before proceeding?>,
    "next_action": "<one of: 'parameter_tweak', 'structural_change', 'accept'>",
    "proposed_hypothesis_id": <id of the hypothesis to try next, or null if next_action != 'structural_change'>
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


def _format_trajectory(
    fit_history: list | None, chi2_current: float, bic_current: float | None
) -> str:
    """Format the χ² and BIC trajectory across past iterations.

    ``fit_history`` is the state's ``fit_results`` list (one entry per past
    fit). The *current* fit's χ²/BIC are appended as the last row.
    """
    lines = []
    for i, fr in enumerate(fit_history or []):
        chi2 = fr.get("chi_squared", float("inf"))
        bic = fr.get("bic")
        bic_str = f"BIC={bic:.1f}" if bic is not None else "BIC=?"
        lines.append(f"  - iter {i}: χ²={chi2:.3f}, {bic_str}")
    # Append current fit as a distinct line if it is not already the last
    # history entry (it usually is).
    if lines:
        return "\n".join(lines)
    bic_str = f"BIC={bic_current:.1f}" if bic_current is not None else "BIC=?"
    return f"  - iter 0: χ²={chi2_current:.3f}, {bic_str}"


def _format_structural_hypotheses(hypotheses: list | None) -> str:
    """Format the structural hypothesis list for LLM prompts."""
    if not hypotheses:
        return "  (no structural hypotheses enumerated at intake)"
    lines = []
    for h in hypotheses:
        status = h.get("status", "pending")
        iter_ = h.get("tried_in_iteration")
        iter_str = f", tried iter {iter_}" if iter_ is not None else ""
        lines.append(
            f"  - #{h.get('id', '?')} [{status}{iter_str}] **{h.get('title', '?')}** "
            f"(source: {h.get('skill_source', '?')})"
        )
        if h.get("change"):
            lines.append(f"      change: {h['change']}")
        if h.get("rationale"):
            lines.append(f"      rationale: {h['rationale']}")
        if h.get("notes"):
            lines.append(f"      notes: {h['notes']}")
    return "\n".join(lines)


def _structural_skeleton(model: dict | None) -> dict:
    """Compact structural view of a model for baseline diffing/display.

    Keeps only the layer stack / materials (substrate, layers, ambient,
    geometry) and per-state structure overrides — that is what a "rewind to
    baseline" decision turns on. Drops fitting runtime (loaded Q/R/dR data
    arrays inside ``states[*].data_files``) and post-fit snapshots so the
    serialized form stays small.
    """
    if not isinstance(model, dict):
        return {}
    skel = {
        k: model.get(k)
        for k in ("substrate", "layers", "ambient", "back_reflection")
        if k in model
    }
    states = model.get("states")
    if states:
        st_list = []
        for st in states:
            st_skel: dict = {"name": st.get("name")}
            for k in ("ambient", "substrate", "layers"):
                if st.get(k) is not None:
                    st_skel[k] = st.get(k)
            st_list.append(st_skel)
        skel["states"] = st_list
    return skel


def _format_baseline_model_section(
    baseline_model: dict | None, current_model: dict | None
) -> str:
    """Render the baseline (intake) model block, or "" when not useful.

    Shown only once the working model has structurally diverged from the
    intake baseline — that is exactly when a "rewind" is meaningful. While the
    two are identical (early iterations) the section is omitted to keep the
    prompt lean.
    """
    if not baseline_model:
        return ""
    import json

    base_skel = _structural_skeleton(baseline_model)
    cur_skel = _structural_skeleton(current_model)
    if not base_skel or base_skel == cur_skel:
        return ""
    return (
        "\n## Baseline (Intake) Model — structural skeleton\n"
        "This is the clean structure first built at intake, before any "
        "refinement added or inflated layers. When you realize a "
        "*reinterpretation* hypothesis (see Rule 13), rebuild from THIS "
        "skeleton, not from the Current Model above.\n"
        "```json\n" + json.dumps(base_skel, indent=2) + "\n```\n"
    )


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
    fit_history: list | None = None,
    structural_hypotheses: list | None = None,
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
                    f"  - Critical edge: {format_critical_edge_line(edge)}"
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
        trajectory=_format_trajectory(fit_history, chi_squared, bic),
        structural_hypotheses=_format_structural_hypotheses(structural_hypotheses),
    )


# ============================================================================
# HYPOTHESIS REVISION (evaluation proposes new hypotheses + re-ranks)
# ============================================================================

HYPOTHESIS_REVISION_PROMPT = """You are revising the ranked list of candidate structural changes for a neutron reflectivity fit, partway through the refinement loop.

The baseline fit has not yet reached the acceptance threshold. New evidence has
accumulated since the hypotheses were first enumerated at intake — residual
structure, parameters pinned at bounds, the χ²/BIC trajectory, and the
evaluator's concerns. Reconsider the list in light of that evidence and the
(possibly newly-activated) domain skills below.

## Domain Knowledge (active skills, refreshed)
{skill_context}

## Sample Description
{sample_description}

## Current Model (JSON)
```json
{current_model_json}
```

## χ² / BIC Trajectory (iterations in order)
{trajectory}

## Residual Fringe Analysis
{residual_analysis}

## Parameters at Range Boundaries
{boundary_hits}

## Evaluator's Concerns
{concerns}

## Existing Ranked Hypotheses
{structural_hypotheses}

## Task
Following the `structural-hypothesis-ranking` skill:

1. Propose any NEW structural hypotheses that the accumulated evidence now
   justifies and that are NOT already covered by an existing entry above. For
   example: residual fringes of a characteristic thickness imply a missing
   layer; an artifact in the data may point to a phenomenon whose skill only
   became relevant once observed. Cite the motivating skill in `rationale` and
   give a concrete `change` (insertion point, thickness/SLD/roughness ranges).
   Do NOT duplicate an existing hypothesis. Return an empty list if nothing new
   is warranted.

2. Re-rank ALL live hypotheses — the existing `pending`/`tried` ones plus your
   new ones — by current expected value, best first. Do NOT resurrect or list
   `rejected` hypotheses. Reference existing hypotheses by their integer `id`;
   reference your new hypotheses by `"new1"`, `"new2"`, … matching their
   1-based position in `new_hypotheses`.

Respond with ONLY a JSON object:

```json
{{
  "new_hypotheses": [
    {{
      "title": "Add SEI layer on Li",
      "rationale": "sei-layer-analysis: residual fringes at ~40 Å in a cycled cell imply a solid-electrolyte interphase",
      "change": "insert a 30-60 Å SEI layer (SLD 0.5-2.0) between Li and electrolyte, roughness 5-20 Å",
      "skill_source": "sei-layer-analysis"
    }}
  ],
  "ranking": ["new1", 3, 1, 2]
}}
```

Output ONLY the JSON object, no markdown fences, no other text.
"""


def format_hypothesis_revision_prompt(
    sample_description: str,
    current_model: dict,
    skill_context: str,
    structural_hypotheses: list,
    fit_history: list,
    chi_squared: float,
    bic: float | None = None,
    residual_analysis: Dict[str, Any] | None = None,
    boundary_hits: list | None = None,
    concerns: list | None = None,
) -> str:
    """Format the prompt that asks the LLM to propose new hypotheses and re-rank.

    Reuses the same section formatters as the fit-evaluation prompt so the
    trajectory, residual, boundary, and hypothesis-list views are consistent
    across the two evaluation-time LLM calls.
    """
    import json

    model_for_prompt = {
        k: v
        for k, v in (current_model or {}).items()
        if k not in ("fitted_parameters", "fitted_uncertainties")
    }
    current_model_json = json.dumps(model_for_prompt, indent=2)

    concerns_str = (
        "\n".join(f"  - {c}" for c in concerns) if concerns else "  (none reported)"
    )

    return HYPOTHESIS_REVISION_PROMPT.format(
        skill_context=skill_context or "(no additional domain knowledge)",
        sample_description=sample_description or "(not provided)",
        current_model_json=current_model_json,
        trajectory=_format_trajectory(fit_history, chi_squared, bic),
        residual_analysis=_format_residual_analysis(residual_analysis),
        boundary_hits=_format_boundary_hits(boundary_hits),
        concerns=concerns_str,
        structural_hypotheses=_format_structural_hypotheses(structural_hypotheses),
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
                    f"  - Critical edge: {format_critical_edge_line(edge)}"
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

## Ranked Structural Hypotheses
{structural_hypotheses}
{baseline_model_section}
## Evaluator's Chosen Next Action
- `next_action`: {next_action}
- `proposed_hypothesis_id`: {proposed_hypothesis_id}

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
      "roughness_max": <max roughness Å>,
      "roughness_tie": {{"fraction_max": <≤ 0.5>}}
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
  }},
  "sample_broadening": {{
    "enabled": <true/false>,
    "min": <min broadening in degrees, default 0.0>,
    "max": <max broadening in degrees, default 0.5>
  }},
  "theta_offset": {{
    "enabled": <true/false>,
    "min": <min offset in degrees, default -0.02>,
    "max": <max offset in degrees, default 0.02>
  }},
  "background": {{
    "enabled": <true/false>,
    "min": <min background, default 0.0>,
    "max": <max background, default 1e-5>
  }}
}}
```

Layers are listed from substrate to ambient (closest to substrate first).

Rules:
1. NEVER change data_file or back_reflection — these are set by the experiment.
2. You may add layers, remove layers, change materials, adjust SLD values, or change parameter bounds.
3. If parameters are hitting their bounds, widen those bounds (sld_min/sld_max, thickness_min/thickness_max).
4. If there are systematic residuals, consider adding a layer.
4b. `roughness_tie` is OPTIONAL and should be OMITTED for normal layers. Add it
    only when an issue reports a non-physical SLD-profile excursion (an
    erf-tail artifact — the profile dipping below or overshooting above the
    range its bounding materials can produce) AND you intend the affected slab
    as a real discrete layer: then set `"roughness_tie": {{"fraction_max": 0.5}}`
    on that layer so its interface is fit as σ = fraction × thickness and can
    never outgrow the layer. If instead the diffuse transition is physically
    intended, leave roughness free and treat those slabs as a profile
    parametrization (do not add roughness_tie).
5. Use best-fit parameter values as starting points where physically reasonable.
6. Unless the data is stated as perfectly normalized, keep intensity varying (fixed: false).
7. Apply all domain-specific rules from the Domain Knowledge section above.
8. sample_broadening and theta_offset only work with angle-based probes (multi-segment data with theta info). Only set "enabled": true when angle info is available and the fit quality warrants it. These give each segment independent resolution/alignment corrections.
9. When sample_broadening or theta_offset are already enabled and hitting bounds, widen their ranges.
9b. background fits a constant incoherent background (works with any data). Only enable it if the user asked for it or if the high-Q residuals plateau above the model (a sign of unmodelled background). When enabled it is tied across each state's data files.
10. If the evaluator's `next_action` is `structural_change`, realize the
    specified hypothesis (`proposed_hypothesis_id`) exactly as described in
    its `change` field — insert/remove the layer at the correct position
    with the suggested bounds. Also return an updated `structural_hypotheses`
    list: mark that hypothesis with `status: "tried"` and stamp
    `tried_in_iteration` with the current iteration. Preserve all other
    hypotheses and their statuses verbatim.
11. If `next_action` is `parameter_tweak`, keep the layer stack unchanged
    and adjust only bounds/starting values. Return the `structural_hypotheses`
    list unchanged.
12. Multi-state co-refinement (sample != structure): if the model has a
    top-level `states` array, the top-level `layers`/`substrate` is the shared
    TEMPLATE. A single state may have a DIFFERENT structure (e.g. "the H2O
    state has no oxide"). To make one state differ, set that state's own
    complete `layers` array (and `substrate` if it differs) inside its entry of
    `states`, e.g. `{{"name": "H2O", ..., "layers": [<that state's full stack>]}}`.
    Leave the other states without a `layers` key so they keep inheriting the
    template. Removing a layer from ONE state means editing that state's
    `layers`, NOT the top-level template (which would change every state). Ties
    referencing a layer absent from a state simply don't apply there — you need
    not edit `shared_parameters`/`unshared_parameters` for a structural removal.
13. REWIND FOR REINTERPRETATION HYPOTHESES. Some hypotheses REINTERPRET an
    existing material rather than add structure — most importantly "the ambient
    solvent/electrolyte is actually deuterated" (its SLD is ~0 in the current
    model, but the data wants a high-SLD medium). Such a reinterpretation is
    MUTUALLY EXCLUSIVE with a speculative layer — or an inflated thickness/SLD —
    that an earlier hypothesis added to explain the SAME data feature (e.g. a
    critical edge / strong low-Q upturn, a layer SLD pinned toward the ambient,
    a thickness driven to ~2x its nominal value). When `next_action` is
    `structural_change` and the chosen hypothesis is such a reinterpretation,
    START FROM THE BASELINE (INTAKE) MODEL shown above — discard the speculative
    layers and inflated values accumulated by earlier hypotheses — and apply
    ONLY the reinterpretation (e.g. set the ambient to the deuterated SLD and
    let it vary over a wide range). Do NOT keep both explanations; prefer the
    simpler (lower-BIC) one. For ordinary ADDITIVE hypotheses (a genuinely
    missing layer), keep editing the Current Model as usual.

{user_constraints}

IMPORTANT: If user feedback is provided below, it takes absolute priority over
any of the rules above.

Output ONLY the JSON object, no markdown fences, no explanation. The object
may include an optional top-level `structural_hypotheses` array (the updated
hypothesis list); if present, it must be the complete list.
"""


def format_model_refinement_prompt_json(
    current_model: dict,
    sample_description: str,
    fit_result: dict,
    features: dict,
    user_constraints: str = "",
    user_feedback: str | None = None,
    skill_context: str = "",
    structural_hypotheses: list | None = None,
    next_action: str | None = None,
    proposed_hypothesis_id: int | None = None,
    baseline_model: dict | None = None,
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
                    f"  - Critical edge: {format_critical_edge_line(edge)}"
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
            structural_hypotheses=_format_structural_hypotheses(structural_hypotheses),
            baseline_model_section=_format_baseline_model_section(
                baseline_model, current_model
            ),
            next_action=next_action or "parameter_tweak",
            proposed_hypothesis_id=(
                proposed_hypothesis_id if proposed_hypothesis_id is not None else "null"
            ),
        )
        + feedback_section
    )


def format_cross_state_ties_prompt(
    sample_description: str, tieable_params: list
) -> str:
    """Prompt to extract per-state (unshared) parameters from a free-text description.

    In a multi-state co-refinement every structural parameter below is SHARED
    (tied to one fitted value) across all states by default. The user's sample
    description may say that some parameters should instead vary independently
    per state (e.g. a surface oxide that differs between an in-air state and an
    in-electrolyte state). The LLM maps that wording onto the dotted parameter
    names and returns the ones to leave UNshared.
    """
    params_block = "\n".join(f"  - {p}" for p in tieable_params)
    return (
        "You are configuring a multi-state neutron-reflectometry co-refinement.\n\n"
        "By default, every structural parameter listed below is SHARED (tied to a "
        "single fitted value) across all states. The sample description may say that "
        "certain parameters should NOT be shared — i.e. they vary independently per "
        "state.\n\n"
        "Sample description:\n"
        f'"""{sample_description}"""\n\n'
        'Tieable parameters — dotted "<layer>.<attr>" names; <attr> is one of '
        "`thickness`, `material.rho` (the SLD), or `interface` (the roughness):\n"
        f"{params_block}\n\n"
        "From the description ONLY, list the parameters that should NOT be shared "
        "across states. Map the user's wording onto the names above:\n"
        '  - "SLD" -> `.material.rho`; "thickness" -> `.thickness`; '
        '"interface"/"roughness" -> `.interface`.\n'
        '  - Match layer names case-insensitively (e.g. "copper oxide" -> a layer '
        'named "Cu oxide").\n'
        "Only include names that appear in the list above. If the description does "
        "not call out any per-state (unshared) parameters, return an empty list.\n\n"
        "Respond with ONLY a JSON object, no prose:\n"
        '{"unshared_parameters": ["<layer>.<attr>", ...]}'
    )


def format_per_state_structure_prompt(
    sample_description: str, state_names: list, template_layers: list
) -> str:
    """Prompt to extract per-state STRUCTURE differences from the description.

    In a multi-state co-refinement the states share a template layer stack by
    default, but the SAMPLE CAN DIFFER per state — a layer may be present in one
    state and absent in another ("sample != structure"), e.g. a surface oxide in
    air but not in electrolyte. The LLM lists, per state, which template layers
    are ABSENT.
    """
    states_block = "\n".join(f"  - {n}" for n in state_names)
    layers_block = "\n".join(f"  - {n}" for n in template_layers)
    return (
        "You are configuring a multi-state neutron-reflectometry co-refinement.\n\n"
        "All states share the template layer stack below by default, but the SAMPLE "
        "CAN DIFFER from state to state — a layer may be PRESENT in one state and "
        "ABSENT in another (sample != structure). For example a surface oxide may exist "
        "in an in-air state but not in an in-electrolyte state.\n\n"
        "Sample description:\n"
        f'"""{sample_description}"""\n\n'
        f"States:\n{states_block}\n\n"
        f"Template layers (the shared default stack):\n{layers_block}\n\n"
        "From the description ONLY, for each state list the template layers that are "
        "ABSENT in that state. Match layer names case-insensitively to the template "
        "names above (e.g. 'copper oxide' -> 'Cu oxide'). If the description does not "
        "say a state lacks a layer, return an empty list for it.\n\n"
        "Respond with ONLY a JSON object mapping state name to absent layer names:\n"
        '{"per_state_absent": {"<state name>": ["<layer name>", ...], ...}}'
    )
