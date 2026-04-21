"""
State definitions for the reflectivity modeling workflow.

The state tracks all information needed throughout the analysis:
- Input data and user description
- Extracted features and parsed sample info
- Current model and fit history
- Conversation with user
"""

from typing import TypedDict, List, Optional, Annotated
import operator


class LayerInfo(TypedDict):
    """Information about a single layer."""

    name: str
    sld: float
    sld_min: Optional[float]
    sld_max: Optional[float]
    thickness: float
    thickness_min: Optional[float]
    thickness_max: Optional[float]
    roughness: float
    roughness_max: Optional[float]


class SubstrateInfo(TypedDict):
    """Information about the substrate."""

    name: str
    sld: float
    roughness: float
    roughness_max: Optional[float]


class AmbientInfo(TypedDict):
    """Information about the ambient/fronting medium."""

    name: str
    sld: float


class IntensityInfo(TypedDict, total=False):
    """Probe intensity normalization settings."""

    value: float  # Starting intensity (default 1.0)
    min: float  # Lower bound (default 0.7)
    max: float  # Upper bound (default 1.1)
    fixed: bool  # If True, intensity is not a fit parameter


class ParsedSample(TypedDict):
    """Structured sample information parsed from user description."""

    substrate: SubstrateInfo
    layers: List[LayerInfo]
    ambient: AmbientInfo
    constraints: List[str]
    hypothesis: Optional[str]
    back_reflection: bool  # True if neutrons come from substrate side


class ModelDefinition(TypedDict, total=False):
    """
    Canonical JSON representation of a refl1d layer model.

    Extends ParsedSample with fields needed to build a FitProblem:
    data file path, intensity settings, and (after fitting) best-fit
    parameter values and uncertainties.

    The workflow stores this instead of a Python script string.
    Experiment/FitProblem objects are built on-the-fly from this dict
    by :func:`aure.nodes.model_builder.build_problem`.
    """

    # ---- Structure (same fields as ParsedSample) ----
    substrate: SubstrateInfo
    layers: List[LayerInfo]
    ambient: AmbientInfo
    constraints: List[str]
    back_reflection: bool

    # ---- Fitting context ----
    data_file: str  # Absolute path to reflectivity data
    intensity: IntensityInfo
    dq_is_fwhm: bool  # Whether dQ column is FWHM (True) or 1-sigma (False)

    # ---- Post-fit snapshots (populated after fitting) ----
    fitted_parameters: dict  # {param_name: value}
    fitted_uncertainties: dict  # {param_name: uncertainty}


class ExtractedFeatures(TypedDict):
    """Physics features extracted from reflectivity data."""

    # Critical edge information
    critical_edges: List[dict]  # [{Qc, estimated_SLD, confidence}]

    # Oscillation/fringe information
    oscillation_periods: List[dict]  # [{delta_Q, thickness, amplitude}]
    estimated_total_thickness: Optional[float]
    n_fringes: int

    # Roughness estimates
    estimated_roughness: float
    roughness_confidence: str  # 'low', 'medium', 'high'

    # Layer count estimation
    estimated_n_layers: int
    layer_count_confidence: str

    # Data quality
    q_min: float
    q_max: float
    n_points: int
    has_error_bars: bool
    normalization_ok: bool


class DatasetInfo(TypedDict, total=False):
    """Information about one data file in a multi-file co-refinement.

    ``file`` and ``label`` are the only fields required at construction time
    (e.g. from the CLI/web layer).  ``dq_is_fwhm`` and ``theta`` are
    populated later during the intake node and are therefore optional here.
    """

    file: str  # Absolute path to the data file
    label: str  # Short human-readable label (e.g. "low-Q", "file1")
    dq_is_fwhm: bool  # Whether dQ column is FWHM (True) or 1-sigma (False)
    theta: float  # Incident angle in degrees (half of TwoTheta from header)


class PerFileFitResult(TypedDict, total=False):
    """Per-file fit results in a multi-file co-refinement."""

    file: str
    label: str
    chi_squared: float
    Q_fit: List[float]
    R_fit: List[float]
    residuals: List[float]
    residual_ratio: List[float]


class FitResult(TypedDict):
    """Results from a refl1d fit."""

    iteration: int
    method: str  # 'lm', 'de', 'dream'
    chi_squared: float
    converged: bool

    # Best-fit parameters
    parameters: dict  # {param_name: value}
    uncertainties: Optional[dict]  # {param_name: uncertainty}
    bounds: Optional[dict]  # {param_name: [low, high]}

    # Curves for plotting
    Q_fit: List[float]
    R_fit: List[float]
    residuals: List[float]
    residual_ratio: List[float]  # R_data / R_fit for fringe analysis

    # SLD profile (from refl1d output)
    sld_z: Optional[List[float]]
    sld_rho: Optional[List[float]]

    # Per-file results (multi-file co-refinement)
    per_file_results: Optional[List[PerFileFitResult]]

    # Evaluation
    issues: List[str]
    suggestions: List[str]


class StructuralHypothesis(TypedDict, total=False):
    """A candidate structural change to the model, ranked at intake time.

    Produced during intake by an LLM reasoning over the active skills, this
    is a ranked list of structural changes (adding, removing, splitting, or
    reshaping layers) that the workflow should consider when parameter-only
    refinement stalls. Each hypothesis carries a rationale sourced from the
    active skills so the evaluator and refiner can reason about it.

    The list is updated in-place (fully replaced) by the modeling and
    evaluation nodes as hypotheses are tried and confirmed or rejected.

    Fields
    ------
    id : int
        Stable identifier within this run (1-based).
    title : str
        One-line description, e.g. "Add native CuO on top of Cu".
    rationale : str
        Why this hypothesis is plausible — cite the active skill.
    change : str
        Concrete structural edit in neutral terms, e.g.
        "insert a 10-30 Å CuO layer (SLD ~5.0) between Cu and D2O".
    skill_source : str
        Name of the skill that motivates this hypothesis.
    status : str
        One of: 'pending', 'tried', 'confirmed', 'rejected'.
    tried_in_iteration : int | None
        Iteration number when the hypothesis was realized.
    notes : str
        Free-form notes (e.g., outcome after trial).
    """

    id: int
    title: str
    rationale: str
    change: str
    skill_source: str
    status: str
    tried_in_iteration: Optional[int]
    notes: str


class LLMCallRecord(TypedDict):
    """Record of a single LLM invocation during the workflow."""

    node: str  # Which workflow node made the call
    timestamp: Optional[str]
    success: bool  # Did the LLM call itself succeed?
    used_fallback: bool  # Was a default/heuristic used instead of LLM output?
    fallback_reason: Optional[str]  # Why the fallback was needed
    error: Optional[str]  # Error message if the call failed


class Message(TypedDict):
    """A message in the conversation."""

    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: Optional[str]


class ReflectivityState(TypedDict):
    """
    Complete state for the reflectivity analysis workflow.

    This state is passed between nodes in the LangGraph workflow
    and accumulates information as the analysis progresses.
    """

    # ========== Input Data ==========
    data_file: str  # Primary data file (always set)
    data_files: List[DatasetInfo]  # All data files for multi-file co-refinement
    dq_is_fwhm: bool  # Whether dQ in primary data_file is FWHM (True) or 1-sigma
    Q: List[float]
    R: List[float]
    dR: List[float]

    # ========== User Input ==========
    sample_description: str
    hypothesis: Optional[str]

    # ========== Parsed Information ==========
    parsed_sample: Optional[ParsedSample]
    extracted_features: Optional[ExtractedFeatures]

    # ========== Model State ==========
    current_model: Optional[dict]  # ModelDefinition JSON (or legacy script str)
    model_history: Annotated[List[dict], operator.add]  # Accumulates models

    # ========== Fit Results ==========
    fit_results: Annotated[List[FitResult], operator.add]  # Accumulates fits
    current_chi2: Optional[float]
    best_chi2: Optional[float]
    best_model: Optional[dict]  # ModelDefinition that produced the best χ²
    best_bic: Optional[float]  # Lowest BIC (complexity-penalized score)
    best_bic_model: Optional[dict]  # ModelDefinition that produced the best BIC

    # ========== Conversation ==========
    messages: Annotated[List[Message], operator.add]

    # ========== LLM Call Tracking ==========
    llm_calls: Annotated[List[LLMCallRecord], operator.add]

    # ========== Interactive Session ==========
    interactive: bool
    pending_user_feedback: Optional[str]

    # ========== Skills ==========
    active_skills: List[str]  # Names of activated Agent Skills
    structural_hypotheses: List[
        StructuralHypothesis
    ]  # Ranked candidate structural changes

    # ========== Workflow Control ==========
    current_node: str
    iteration: int
    max_iterations: int
    workflow_complete: bool
    error: Optional[str]
    output_dir: Optional[str]
    user_config: Optional[dict]  # User-supplied YAML config (criteria & constraints)
    bounds_only_refinement: (
        bool  # Set by evaluation when only bound-expansion is needed
    )


def create_initial_state(
    data_file: str,
    sample_description: str,
    hypothesis: Optional[str] = None,
    max_iterations: int = 5,
    user_config: Optional[dict] = None,
    data_files: Optional[List[dict]] = None,
) -> ReflectivityState:
    """
    Create initial state for a new analysis workflow.

    Args:
        data_file: Path to reflectivity data file
        sample_description: User's description of the sample
        hypothesis: Optional hypothesis to test
        max_iterations: Maximum refinement iterations
        user_config: Optional user-supplied YAML configuration
        data_files: Optional list of DatasetInfo dicts for multi-file co-refinement

    Returns:
        Initial workflow state
    """
    return ReflectivityState(
        # Input data (to be filled by intake node)
        data_file=data_file,
        data_files=data_files or [],
        dq_is_fwhm=True,  # Default; overridden by intake after header inspection
        Q=[],
        R=[],
        dR=[],
        # User input
        sample_description=sample_description,
        hypothesis=hypothesis,
        # Parsed information (to be filled by analysis nodes)
        parsed_sample=None,
        extracted_features=None,
        # Model state
        current_model=None,
        model_history=[],
        # Fit results
        fit_results=[],
        current_chi2=None,
        best_chi2=None,
        best_model=None,
        best_bic=None,
        best_bic_model=None,
        # Conversation
        messages=[],
        # LLM call tracking
        llm_calls=[],
        # Interactive session
        interactive=False,
        pending_user_feedback=None,
        # Skills
        active_skills=[],
        structural_hypotheses=[],
        # Workflow control
        current_node="intake",
        iteration=0,
        max_iterations=max_iterations,
        workflow_complete=False,
        error=None,
        output_dir=None,
        user_config=user_config,
        bounds_only_refinement=False,
    )
