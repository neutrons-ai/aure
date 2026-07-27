"""
State definitions for the reflectivity modeling workflow.

The state tracks all information needed throughout the analysis:
- Input data and user description
- Extracted features and parsed sample info
- Current model and fit history
- Conversation with user
"""

from typing import TypedDict, List, Optional, Dict


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
    # Optional: tie this layer's interface roughness to its own thickness as
    # sigma = fraction * thickness, fitting `fraction` instead of `roughness`.
    # Keeps a thin layer's interface from outgrowing it (prevents erf-tail
    # artifacts) while letting the thickness move. Shape:
    # {"fraction_init": float, "fraction_min": float, "fraction_max": float}.
    roughness_tie: Optional[Dict[str, float]]


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

    # ---- Multi-state co-refinement ----
    # When ``states`` is present and non-empty it is the source of truth for
    # data files and per-measurement settings. ``data_file`` / ``intensity``
    # / ``back_reflection`` above remain as the legacy single-state shape and
    # are synthesised from ``states[0]`` when needed (and vice versa via
    # :func:`iter_states`).
    states: List["StateDefinition"]
    shared_parameters: List[str]  # Whitelist of "Layer.attr" tied across states
    unshared_parameters: List[str]  # Blacklist; mutually exclusive with above
    # Identity (NOT physics): whether the co-refined states are distinct
    # physical samples rather than one sample measured under several
    # conditions. Default ``False`` — co-refined states usually ARE the same
    # sample (e.g. one electrode in D2O then H2O), so they share one
    # downstream ``sample_id``. Set ``True`` when the states are genuinely
    # different samples that merely share a fitting strategy, so the
    # data-assembler assigns a distinct ``sample_id`` per state. Orthogonal
    # to per-state structure ("sample != structure"): a single sample can
    # change structure between states, and distinct samples can share one.
    distinct_sample: bool

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

    ``intensity`` is an optional per-probe override applied by
    :func:`~aure.nodes.model_builder.build_states_problem`. It takes
    precedence over the state-level intensity and is used by
    ``aure import-refl1d`` to preserve the post-fit intensity values
    for individual segments in a co-refinement.
    """

    file: str  # Absolute path to the data file
    label: str  # Short human-readable label (e.g. "low-Q", "file1")
    dq_is_fwhm: bool  # Whether dQ column is FWHM (True) or 1-sigma (False)
    theta: float  # Incident angle in degrees (half of TwoTheta from header)
    intensity: dict  # Per-probe intensity override: {value/init, min, max, fixed}


class PerFileFitResult(TypedDict, total=False):
    """Per-file fit results in a multi-file co-refinement."""

    file: str
    label: str
    state: str  # State name (multi-state co-refinement); absent or "" for single-state
    chi_squared: float
    Q_fit: List[float]
    R_fit: List[float]
    residuals: List[float]
    residual_ratio: List[float]


class StateDefinition(TypedDict, total=False):
    """One physical state of the sample in a multi-state co-refinement.

    A *state* groups one or more data files that share a single refl1d
    ``Sample`` — i.e. every layer parameter is tied across the state's
    files. Across states, only the attributes named in
    ``ModelDefinition.shared_parameters`` (or implied by the default
    tied set) are tied; all others vary independently per state.

    Within one state, all data files must be the same kind: either a
    single combined file (``REFL_{set}_combined_data_auto.txt``) or N
    partial files sharing one ``set_id``. Mixing combined and partial
    files within one state is rejected.

    Per-state nuisance parameters (``theta_offset``, ``sample_broadening``)
    are only meaningful for partials states. ``ambient``, ``intensity``,
    and ``back_reflection`` are always per-state (they describe the
    measurement, not the sample structure) but inherit the model-level
    defaults when omitted.

    **Sample ≠ structure:** a state may also carry its own ``layers`` /
    ``substrate`` — its COMPLETE structure, overriding the model-level template
    for that state. So a layer can be present in some states and absent in
    others (e.g. a surface oxide in air but not in electrolyte). When omitted,
    the state inherits ``ModelDefinition.layers``/``substrate``. Cross-state
    ties apply only where a layer exists in the relevant states.
    """

    name: str  # Unique within the model definition
    data_files: List[DatasetInfo]  # Combined OR partials of one set_id
    extra_description: str  # Appended to sample_description when prompting the LLM
    back_reflection: bool  # Per-state stack orientation
    theta_offset: dict  # {init, min, max} — partials only
    sample_broadening: dict  # {init, min, max} — partials only
    ambient: AmbientInfo  # Per-state override of the model-level ambient
    intensity: IntensityInfo  # Per-state override of the model-level intensity
    # Per-state STRUCTURE override (full stack). Absent → inherit the
    # model-level template (ModelDefinition.layers/substrate).
    layers: List[LayerInfo]
    substrate: SubstrateInfo


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

    The initial list is produced during intake by an LLM reasoning over the
    active skills (and the user's stated hypothesis). It is a ranked list of
    structural changes (adding, removing, splitting, or reshaping layers) that
    the workflow should consider when parameter-only refinement stalls. Each
    hypothesis carries a rationale sourced from the active skills (or the user)
    so the evaluator and refiner can reason about it.

    The list evolves through a single guarded merge (``nodes.hypotheses``):

    * **modeling** may only update the *status* of existing entries.
    * **evaluation** may, when fit evidence warrants, append *new* entries
      (``origin="evaluation"``) and re-rank the list.
    * **intake** seeds the list, including the user's hypothesis as
      ``origin="user"`` entries ranked at the top.

    Identity fields (``id``/``title``/``rationale``/``change``/``skill_source``/
    ``origin``) are immutable once created; only ``status``/``tried_in_iteration``/
    ``notes`` are mutable.

    Fields
    ------
    id : int
        Stable identifier within this run (1-based).
    title : str
        One-line description, e.g. "Add native CuO on top of Cu".
    rationale : str
        Why this hypothesis is plausible — cite the active skill (or "user").
    change : str
        Concrete structural edit in neutral terms, e.g.
        "insert a 10-30 Å CuO layer (SLD ~5.0) between Cu and D2O".
    skill_source : str
        Name of the skill that motivates this hypothesis, or "user" when it
        was derived from the user's stated hypothesis.
    origin : str
        Provenance of the entry. One of: 'user' (from the user's hypothesis),
        'skill' (enumerated at intake from the active skills), 'evaluation'
        (proposed mid-run from fit evidence).
    status : str
        One of: 'pending', 'tried', 'confirmed', 'rejected'.
    tried_in_iteration : int | None
        Iteration number when the hypothesis was realized.
    created_in_iteration : int | None
        Iteration number when the hypothesis was added (None for intake-time
        entries; set for hypotheses proposed later by the evaluation node).
    notes : str
        Free-form notes (e.g., outcome after trial).
    """

    id: int
    title: str
    rationale: str
    change: str
    skill_source: str
    origin: str
    status: str
    tried_in_iteration: Optional[int]
    created_in_iteration: Optional[int]
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

    This state is passed between nodes of the analysis workflow (runner.py) and
    accumulates information as the analysis progresses. The list fields that
    accumulate across nodes (model_history, fit_results, messages, llm_calls)
    are combined by ``runner._merge_state_updates``; they are plain lists here.
    """

    # ========== Input Data ==========
    data_file: str  # Primary data file (always set)
    data_files: List[DatasetInfo]  # All data files for multi-file co-refinement
    states: List[StateDefinition]  # Per-state grouping (empty for single-state)
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
    model_history: List[dict]  # Accumulates models (merged by _merge_state_updates)
    # Clean snapshot of the very first model built at intake (before any
    # refinement). Used as the "rewind" starting point when the refiner
    # realizes a *reinterpretation* hypothesis (e.g. "the solvent is actually
    # deuterated") that is mutually exclusive with speculative layers added by
    # earlier hypotheses — so the new explanation isn't stacked on top of the
    # discarded one. ``None`` until the initial model is built.
    baseline_model: Optional[dict]

    # ========== Fit Results ==========
    fit_results: List[FitResult]  # Accumulates fits (merged by _merge_state_updates)
    current_chi2: Optional[float]
    best_chi2: Optional[float]
    best_model: Optional[dict]  # ModelDefinition that produced the best χ²
    best_bic: Optional[float]  # Lowest BIC (complexity-penalized score)
    best_bic_model: Optional[dict]  # ModelDefinition that produced the best BIC
    # Set by the terminal `finalize` node once the refinement loop has stopped.
    # `finalized` makes that node idempotent (the runner also calls it
    # defensively on loop-exit paths that never route through a node);
    # `final_selection` records which fit iteration was chosen as the run's
    # answer and why, so the decision is auditable from a checkpoint.
    finalized: bool
    final_selection: Optional[dict]
    # Set by the optional `final_fit` step (runner terminal block) when
    # `FIT_METHOD_FINAL` requests an MCMC (dream) polish of the finalize-selected
    # model to attach uncertainties the fast exploration optimizer cannot
    # produce. Records whether it ran, the export directory (so problem.json can
    # follow it), and the before/after χ². None when the feature is disabled.
    final_fit: Optional[dict]

    # ========== Conversation ==========
    messages: List[Message]

    # ========== LLM Call Tracking ==========
    llm_calls: List[LLMCallRecord]

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
    model_name: Optional[str]  # Resolved basename for refl1d/bumps export files
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
    states: Optional[List[dict]] = None,
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
        states: Optional list of StateDefinition dicts for multi-state
            co-refinement.  When provided, ``data_files`` is auto-populated
            as the flattened concatenation of every state's ``data_files``.

    Returns:
        Initial workflow state
    """
    resolved_states: List[dict] = list(states or [])
    if resolved_states and not data_files:
        data_files = flatten_data_files(resolved_states)
    return ReflectivityState(
        # Input data (to be filled by intake node)
        data_file=data_file,
        data_files=data_files or [],
        states=resolved_states,
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
        baseline_model=None,
        # Fit results
        fit_results=[],
        current_chi2=None,
        best_chi2=None,
        best_model=None,
        best_bic=None,
        best_bic_model=None,
        finalized=False,
        final_selection=None,
        final_fit=None,
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


# ============================================================================
# State helpers (multi-state co-refinement)
# ============================================================================


_DEFAULT_STATE_NAME = "state0"


def iter_states(definition: dict) -> List[dict]:
    """Return the list of states for a model definition.

    Synthesises a single-state view from the legacy single-file shape when
    ``definition["states"]`` is absent or empty, so existing callers
    continue to work unchanged.

    Parameters
    ----------
    definition
        A :class:`ModelDefinition`-shaped dict, or a
        :class:`ReflectivityState`-shaped dict (only the relevant keys are
        read in either case).

    Returns
    -------
    list of dict
        A list of :class:`StateDefinition`-shaped dicts. The list always
        has at least one entry; when ``definition`` carries explicit
        ``states``, the list is returned verbatim (not copied).
    """
    states = definition.get("states") or []
    if states:
        return list(states)

    # Synthesise a 1-state view from the legacy fields.
    data_files = list(definition.get("data_files") or [])
    if not data_files:
        primary = definition.get("data_file")
        if primary:
            data_files = [{"file": primary, "label": "primary"}]

    synthetic: dict = {
        "name": _DEFAULT_STATE_NAME,
        "data_files": data_files,
    }
    if "intensity" in definition:
        synthetic["intensity"] = definition["intensity"]
    if "ambient" in definition:
        synthetic["ambient"] = definition["ambient"]
    if "back_reflection" in definition:
        synthetic["back_reflection"] = definition["back_reflection"]
    return [synthetic]


def flatten_data_files(states: List[dict]) -> List[dict]:
    """Flatten the per-state ``data_files`` lists into one ordered list.

    Order is state-by-state, then file-by-file within each state. This
    is the layout used by the multi-file fitting path and by web/plotting
    helpers that group by ``label``.
    """
    flat: List[dict] = []
    for state in states:
        for ds in state.get("data_files", []) or []:
            flat.append(ds)
    return flat


def is_multi_state(definition: dict) -> bool:
    """Return True if *definition* has more than one explicit state."""
    return len(definition.get("states") or []) > 1
