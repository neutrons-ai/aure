"""
State definitions for the reflectivity modeling workflow.

The state tracks all information needed throughout the analysis:
- Input data and user description
- Extracted features and parsed sample info
- Current model and fit history
- Conversation with user
"""

from typing import TypedDict, List, Optional, Annotated, Any
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


class ParsedSample(TypedDict):
    """Structured sample information parsed from user description."""
    substrate: SubstrateInfo
    layers: List[LayerInfo]
    ambient: AmbientInfo
    constraints: List[str]
    hypothesis: Optional[str]
    back_reflection: bool  # True if neutrons come from substrate side


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


class FitResult(TypedDict):
    """Results from a refl1d fit."""
    iteration: int
    method: str  # 'lm', 'de', 'dream'
    chi_squared: float
    converged: bool
    
    # Best-fit parameters
    parameters: dict  # {param_name: value}
    uncertainties: Optional[dict]  # {param_name: uncertainty}
    
    # Curves for plotting
    Q_fit: List[float]
    R_fit: List[float]
    residuals: List[float]
    
    # Evaluation
    issues: List[str]
    suggestions: List[str]


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
    data_file: str
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
    current_model: Optional[str]  # refl1d Python script
    model_history: Annotated[List[dict], operator.add]  # Accumulates models
    
    # ========== Fit Results ==========
    fit_results: Annotated[List[FitResult], operator.add]  # Accumulates fits
    current_chi2: Optional[float]
    best_chi2: Optional[float]
    best_model: Optional[str]  # Model script that produced the best χ²
    
    # ========== Conversation ==========
    messages: Annotated[List[Message], operator.add]
    
    # ========== Workflow Control ==========
    current_node: str
    iteration: int
    max_iterations: int
    workflow_complete: bool
    error: Optional[str]
    output_dir: Optional[str]
    user_config: Optional[dict]  # User-supplied YAML config (criteria & constraints)


def create_initial_state(
    data_file: str,
    sample_description: str,
    hypothesis: Optional[str] = None,
    max_iterations: int = 5,
    user_config: Optional[dict] = None,
) -> ReflectivityState:
    """
    Create initial state for a new analysis workflow.
    
    Args:
        data_file: Path to reflectivity data file
        sample_description: User's description of the sample
        hypothesis: Optional hypothesis to test
        max_iterations: Maximum refinement iterations
        user_config: Optional user-supplied YAML configuration
    
    Returns:
        Initial workflow state
    """
    return ReflectivityState(
        # Input data (to be filled by intake node)
        data_file=data_file,
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
        
        # Conversation
        messages=[],
        
        # Workflow control
        current_node='intake',
        iteration=0,
        max_iterations=max_iterations,
        workflow_complete=False,
        error=None,
        output_dir=None,
        user_config=user_config,
    )
