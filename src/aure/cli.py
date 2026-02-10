"""
Command-line interface for the Reflectivity Analysis Workflow.

Usage:
    python -m aure.cli analyze data.dat "100 nm polystyrene on silicon"
    python -m aure.cli lookup-sld silicon gold D2O
    python -m aure.cli mcp-server
"""

import warnings

# Suppress compatibility warnings before any imports that might trigger them
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible")

import json
import logging
import sys
from typing import Optional

import click
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .llm import get_llm_info, get_llm, invoke_with_timeout, LLMTimeoutError, get_llm_timeout


def _check_llm_status(quiet: bool = False, test_connection: bool = True) -> tuple[bool, str]:
    """
    Check and report LLM configuration status.
    
    Args:
        quiet: If True, suppress output (for JSON mode)
        test_connection: If True, test the LLM with a simple query
        
    Returns:
        Tuple of (success, message)
    """
    info = get_llm_info()
    
    if not quiet:
        click.echo(click.style("  LLM Configuration", fg="cyan", bold=True))
        click.echo(f"    Provider: {info['provider']}")
        click.echo(f"    Model: {info['model']}")
        if info.get('base_url'):
            click.echo(f"    Base URL: {info['base_url']}")
    
    if not info['available']:
        msg = "LLM not configured (missing API key or base URL)"
        if not quiet:
            click.echo(click.style(f"    Status: ✗ {msg}", fg="yellow"))
        return False, msg
    
    if test_connection:
        if not quiet:
            click.echo("    Testing connection...", nl=False)
        try:
            llm = get_llm()
            # Use timeout to prevent infinite retries on quota errors
            invoke_with_timeout(llm, "Reply with only the word 'OK'", timeout_seconds=min(30, get_llm_timeout()))
            if not quiet:
                click.echo(click.style(" ✓ Connected", fg="green"))
            return True, "LLM connected successfully"
        except LLMTimeoutError:
            short_msg = "API quota/rate limit exceeded (call timed out)"
            if not quiet:
                click.echo(click.style(f" ✗ {short_msg}", fg="red"))
                click.echo(click.style("    Try: Wait and retry, or switch to a different model/provider", fg="yellow"))
            return False, short_msg
        except Exception as e:
            error_msg = str(e)
            error_lower = error_msg.lower()
            
            # Detect common error types
            if "quota" in error_lower or "rate" in error_lower or "limit" in error_lower or "429" in error_msg:
                short_msg = "API quota/rate limit exceeded"
                if not quiet:
                    click.echo(click.style(f" ✗ {short_msg}", fg="red"))
                    click.echo(click.style("    Try: Wait and retry, or switch to a different model/provider", fg="yellow"))
                return False, short_msg
            elif "401" in error_msg or "unauthorized" in error_lower or "api key not valid" in error_lower or "api_key_invalid" in error_lower or ("invalid" in error_lower and "key" in error_lower):
                short_msg = "Invalid API key"
                if not quiet:
                    click.echo(click.style(f" ✗ {short_msg}", fg="red"))
                    click.echo(click.style("    Check your LLM_API_KEY in .env", fg="yellow"))
                return False, short_msg
            elif "not found" in error_lower or "404" in error_msg:
                short_msg = f"Model '{info['model']}' not found"
                if not quiet:
                    click.echo(click.style(f" ✗ {short_msg}", fg="red"))
                    click.echo(click.style("    Check LLM_MODEL in .env", fg="yellow"))
                return False, short_msg
            elif "connection" in error_lower or "connect" in error_lower:
                short_msg = "Connection failed"
                if not quiet:
                    click.echo(click.style(f" ✗ {short_msg}", fg="red"))
                    if info.get('base_url'):
                        click.echo(click.style(f"    Check if server is running at {info['base_url']}", fg="yellow"))
                return False, short_msg
            else:
                # Generic error - truncate long messages
                if len(error_msg) > 100:
                    error_msg = error_msg[:100] + "..."
                if not quiet:
                    click.echo(click.style(" ✗ Failed", fg="red"))
                    click.echo(click.style(f"    Error: {error_msg}", fg="red"))
                return False, f"Connection failed: {error_msg}"
    
    if not quiet:
        click.echo(click.style("    Status: ✓ Configured", fg="green"))
    return True, "LLM configured"


# ============================================================================
# Main CLI Group
# ============================================================================

@click.group()
@click.version_option(version="0.1.0", prog_name="aure")
def cli():
    """
    Reflectivity Analysis Workflow CLI.
    
    An intelligent assistant for analyzing neutron reflectivity data.
    Uses LangGraph to orchestrate data analysis, model building, and fitting.
    """
    pass


# ============================================================================
# Analysis Commands
# ============================================================================

@cli.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("sample_description")
@click.option(
    "--hypothesis", "-h",
    help="Optional hypothesis to test",
)
@click.option(
    "--max-refinements", "-m",
    default=5,
    type=int,
    help="Maximum number of refinement iterations (default: 5)",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    help="Output directory for checkpoints and results",
)
@click.option(
    "--json", "output_json",
    is_flag=True,
    help="Output results as JSON",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose logging to trace workflow progress",
)
def analyze(
    data_file: str,
    sample_description: str,
    hypothesis: Optional[str],
    max_refinements: int,
    output_dir: Optional[str],
    output_json: bool,
    verbose: bool,
):
    """
    Analyze a reflectivity data file with fitting and refinement.
    
    DATA_FILE: Path to the reflectivity data file (.dat, .txt, .refl)
    
    SAMPLE_DESCRIPTION: Natural language description of the sample
    
    The workflow generates a model, fits it to the data, evaluates the fit,
    and refines the model if needed (up to --max-refinements iterations).
    
    When --output-dir is specified, checkpoints are saved after each workflow
    node (intake, analysis, modeling, fitting, evaluation, refinement).
    These can be used to inspect intermediate results or resume the workflow.
    
    Examples:
    
        # Basic analysis with default 5 refinement iterations
        python -m aure.cli analyze data.dat "100 nm polystyrene on silicon"
        
        # Save checkpoints to a directory
        python -m aure.cli analyze data.dat "multilayer" -o ./results
        
        # Limit refinement iterations
        python -m aure.cli analyze data.dat "thin film" --max-refinements 3
        
        # Resume from a checkpoint
        python -m aure.cli resume ./results/checkpoints/004_fitting.json
    """
    # Configure logging if verbose
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            stream=sys.stderr,
        )
        # Set specific loggers
        for module in ['agent.nodes.fitting', 'agent.nodes.evaluation', 'agent.nodes.refinement']:
            logging.getLogger(module).setLevel(logging.INFO)
    
    from .workflow import run_analysis
    
    if not output_json:
        click.echo(click.style("═" * 60, fg="blue"))
        click.echo(click.style("  Reflectivity Analysis Workflow", fg="blue", bold=True))
        click.echo(click.style("═" * 60, fg="blue"))
        click.echo()
        
        # Check LLM status first
        llm_ok, llm_msg = _check_llm_status(quiet=False, test_connection=True)
        click.echo()
        
        click.echo(f"  Data file: {data_file}")
        click.echo(f"  Sample: {sample_description}")
        if hypothesis:
            click.echo(f"  Hypothesis: {hypothesis}")
        click.echo()
    else:
        # Still check LLM in quiet mode for JSON output
        llm_ok, llm_msg = _check_llm_status(quiet=True, test_connection=True)
    
    # Stop if LLM is not available
    if not llm_ok:
        if output_json:
            click.echo(json.dumps({"error": f"LLM not available: {llm_msg}", "llm": get_llm_info()}))
        else:
            click.echo(click.style(f"  Cannot proceed: {llm_msg}", fg="red"))
            click.echo("  Please configure a working LLM provider in .env")
        sys.exit(1)
    
    # Create checkpoint callback for progress reporting
    def checkpoint_callback(state, node_name):
        if not output_json:
            status = "✓" if not state.get("error") else "✗"
            click.echo(click.style(f"  [{status}] {node_name.title()}", fg="green" if status == "✓" else "red"))
    
    # Run analysis
    try:
        result = run_analysis(
            data_file=data_file,
            sample_description=sample_description,
            hypothesis=hypothesis,
            max_iterations=max_refinements,
            output_dir=output_dir,
            checkpoint_callback=checkpoint_callback if not output_json else None,
        )
    except Exception as e:
        if output_json:
            click.echo(json.dumps({"error": str(e)}))
        else:
            click.echo(click.style(f"Error: {e}", fg="red"))
        sys.exit(1)
    
    # Check for errors
    if result.get("error"):
        if output_json:
            click.echo(json.dumps({"error": result["error"]}))
        else:
            click.echo(click.style(f"Error: {result['error']}", fg="red"))
        sys.exit(1)
    
    # Output results
    if output_json:
        output_data = {
            "success": True,
            "llm": {
                "available": llm_ok,
                "info": get_llm_info(),
            },
            "output_dir": output_dir,
            "n_points": len(result.get("Q", [])),
            "parsed_sample": result.get("parsed_sample"),
            "extracted_features": result.get("extracted_features"),
            "model_generated": result.get("current_model") is not None,
        }
        if result.get("fit_results"):
            latest_fit = result["fit_results"][-1] if result["fit_results"] else {}
            output_data["fit_result"] = {
                "chi_squared": latest_fit.get("chi_squared"),
                "parameters": latest_fit.get("parameters"),
            }
        click.echo(json.dumps(output_data, indent=2))
    else:
        _print_analysis_results(result, output_dir)


def _print_analysis_results(result: dict, output_dir: Optional[str] = None):
    """Pretty-print analysis results."""
    click.echo()
    
    # Data loaded
    n_points = len(result.get("Q", []))
    click.echo(click.style("  Data Loaded", fg="cyan", bold=True))
    click.echo(f"    Points: {n_points}")
    
    if result.get("Q"):
        import numpy as np
        Q = np.array(result["Q"])
        click.echo(f"    Q range: {Q.min():.4f} - {Q.max():.4f} Å⁻¹")
    click.echo()
    
    # Parsed sample
    parsed = result.get("parsed_sample")
    if parsed:
        click.echo(click.style("  Sample Structure", fg="cyan", bold=True))
        click.echo(f"    Substrate: {parsed['substrate']['name']} "
                  f"(SLD={parsed['substrate']['sld']:.2f})")
        for i, layer in enumerate(parsed.get("layers", [])):
            click.echo(f"    Layer {i+1}: {layer['name']} - "
                      f"{layer['thickness']:.0f} Å "
                      f"(SLD={layer['sld']:.2f})")
        click.echo(f"    Ambient: {parsed['ambient']['name']} "
                  f"(SLD={parsed['ambient']['sld']:.2f})")
        click.echo()
    
    # Extracted features
    features = result.get("extracted_features")
    if features:
        click.echo(click.style("  Extracted Features", fg="cyan", bold=True))
        click.echo(f"    Estimated layers: {features.get('estimated_n_layers', '?')}")
        thickness = features.get("estimated_total_thickness")
        if thickness:
            click.echo(f"    Total thickness: {thickness:.0f} Å")
        roughness = features.get("estimated_roughness")
        if roughness:
            click.echo(f"    Surface roughness: {roughness:.1f} Å")
        click.echo()
    
    # Model
    if result.get("current_model"):
        click.echo(click.style("  Model Generated", fg="green", bold=True))
        model_lines = result["current_model"].split("\n")
        click.echo(f"    Lines: {len(model_lines)}")
        click.echo()
    
    # Fit results
    if result.get("fit_result"):
        fit = result["fit_result"]
        click.echo(click.style("  Fit Results", fg="cyan", bold=True))
        chi2 = fit.get("chi_squared", "N/A")
        click.echo(f"    χ²: {chi2:.3f}" if isinstance(chi2, float) else f"    χ²: {chi2}")
        click.echo(f"    Method: {fit.get('method', 'unknown')}")
        
        if fit.get("parameters"):
            click.echo("    Parameters:")
            for name, value in list(fit["parameters"].items())[:5]:
                unc = fit.get("uncertainties", {}).get(name)
                if unc:
                    click.echo(f"      {name}: {value:.3f} ± {unc:.3f}")
                else:
                    click.echo(f"      {name}: {value:.3f}")
        click.echo()
        
        # Evaluation
        evaluation = result.get("evaluation")
        if evaluation:
            quality = evaluation.get("chi_squared_quality", "unknown")
            acceptable = evaluation.get("acceptable", False)
            color = "green" if acceptable else "yellow"
            click.echo(click.style(f"  Fit Quality: {quality}", fg=color, bold=True))
            
            if evaluation.get("issues"):
                click.echo("    Issues:")
                for issue in evaluation["issues"]:
                    click.echo(f"      - {issue}")
            
            if evaluation.get("suggestions"):
                click.echo("    Suggestions:")
                for sug in evaluation["suggestions"]:
                    click.echo(f"      - {sug}")
    
    # Output directory info
    if output_dir:
        click.echo()
        click.echo(click.style("  Output Directory", fg="cyan", bold=True))
        click.echo(f"    Checkpoints: {output_dir}/checkpoints/")
        click.echo(f"    Models: {output_dir}/models/")
        click.echo(f"    Final state: {output_dir}/final_state.json")


# ============================================================================
# Checkpoint Commands
# ============================================================================

@cli.command()
@click.argument("checkpoint_path", type=click.Path(exists=True))
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    help="Output directory for new checkpoints (defaults to original)",
)
@click.option(
    "--fit/--no-fit",
    default=True,
    help="Include fitting in resumed workflow",
)
@click.option(
    "--json", "output_json",
    is_flag=True,
    help="Output results as JSON",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def resume(
    checkpoint_path: str,
    output_dir: Optional[str],
    fit: bool,
    output_json: bool,
    verbose: bool,
):
    """
    Resume a workflow from a checkpoint.
    
    CHECKPOINT_PATH: Path to a checkpoint JSON file
    
    This command loads a saved checkpoint and continues the workflow
    from where it left off. Useful for:
    
    - Retrying after a failure
    - Testing changes to specific nodes
    - Skipping early stages when iterating on later ones
    
    Examples:
    
        # Resume from after fitting
        python -m aure.cli resume output/checkpoints/004_fitting.json
        
        # Resume with new output directory
        python -m aure.cli resume output/checkpoints/003_modeling.json -o output_v2
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            stream=sys.stderr,
        )
    
    from .workflow import run_from_checkpoint, CheckpointManager
    
    if not output_json:
        click.echo(click.style("═" * 60, fg="blue"))
        click.echo(click.style("  Resume Workflow from Checkpoint", fg="blue", bold=True))
        click.echo(click.style("═" * 60, fg="blue"))
        click.echo()
        
        # Load and display checkpoint info
        checkpoint_data = CheckpointManager.load_checkpoint(checkpoint_path)
        click.echo(f"  Checkpoint: {checkpoint_path}")
        click.echo(f"  Node: {checkpoint_data['node']}")
        click.echo(f"  Iteration: {checkpoint_data.get('iteration', 0)}")
        click.echo(f"  Timestamp: {checkpoint_data.get('timestamp', 'unknown')}")
        click.echo()
        
        # Check LLM status
        llm_ok, llm_msg = _check_llm_status(quiet=False, test_connection=True)
        click.echo()
    else:
        llm_ok, llm_msg = _check_llm_status(quiet=True, test_connection=True)
    
    if not llm_ok:
        if output_json:
            click.echo(json.dumps({"error": f"LLM not available: {llm_msg}"}))
        else:
            click.echo(click.style(f"  Cannot proceed: {llm_msg}", fg="red"))
        sys.exit(1)
    
    # Create checkpoint callback for progress reporting
    def checkpoint_callback(state, node_name):
        if not output_json:
            status = "✓" if not state.get("error") else "✗"
            click.echo(click.style(f"  [{status}] {node_name.title()}", fg="green" if status == "✓" else "red"))
    
    try:
        result = run_from_checkpoint(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            include_fitting=fit,
            checkpoint_callback=checkpoint_callback if not output_json else None,
        )
    except Exception as e:
        if output_json:
            click.echo(json.dumps({"error": str(e)}))
        else:
            click.echo(click.style(f"Error: {e}", fg="red"))
        sys.exit(1)
    
    if result.get("error"):
        if output_json:
            click.echo(json.dumps({"error": result["error"]}))
        else:
            click.echo(click.style(f"Error: {result['error']}", fg="red"))
        sys.exit(1)
    
    if output_json:
        click.echo(json.dumps({"success": True, "output_dir": output_dir}, indent=2))
    else:
        click.echo()
        click.echo(click.style("  Workflow resumed successfully", fg="green", bold=True))


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True))
@click.option(
    "--json", "output_json",
    is_flag=True,
    help="Output as JSON",
)
def checkpoints(output_dir: str, output_json: bool):
    """
    List checkpoints in an output directory.
    
    OUTPUT_DIR: Path to the output directory containing checkpoints
    
    Examples:
    
        python -m aure.cli checkpoints ./output
    """
    from .workflow import CheckpointManager
    
    checkpoint_list = CheckpointManager.list_checkpoints(output_dir)
    
    if output_json:
        click.echo(json.dumps(checkpoint_list, indent=2))
    else:
        if not checkpoint_list:
            click.echo("No checkpoints found.")
            return
        
        click.echo(click.style("  Checkpoints", fg="cyan", bold=True))
        click.echo()
        
        for cp in checkpoint_list:
            node = cp.get("node", "unknown")
            iteration = cp.get("iteration", 0)
            timestamp = cp.get("timestamp", "")
            filename = cp.get("file", "")
            
            iter_str = f" (iter {iteration})" if iteration > 0 else ""
            click.echo(f"    {filename}")
            click.echo(f"      Node: {node}{iter_str}")
            click.echo(f"      Time: {timestamp}")
            click.echo()


@cli.command("inspect-checkpoint")
@click.argument("checkpoint_path", type=click.Path(exists=True))
@click.option(
    "--show-state", "-s",
    is_flag=True,
    help="Show full state (can be verbose)",
)
@click.option(
    "--json", "output_json",
    is_flag=True,
    help="Output as JSON",
)
def inspect_checkpoint(checkpoint_path: str, show_state: bool, output_json: bool):
    """
    Inspect a checkpoint file.
    
    CHECKPOINT_PATH: Path to a checkpoint JSON file
    
    Examples:
    
        python -m aure.cli inspect-checkpoint output/checkpoints/004_fitting.json
        python -m aure.cli inspect-checkpoint output/checkpoints/004_fitting.json -s
    """
    from .workflow import CheckpointManager
    
    checkpoint_data = CheckpointManager.load_checkpoint(checkpoint_path)
    
    if output_json:
        if show_state:
            click.echo(json.dumps(checkpoint_data, indent=2))
        else:
            # Exclude large state fields
            summary = {k: v for k, v in checkpoint_data.items() if k != "state"}
            summary["state_keys"] = list(checkpoint_data.get("state", {}).keys())
            click.echo(json.dumps(summary, indent=2))
    else:
        click.echo(click.style("  Checkpoint Details", fg="cyan", bold=True))
        click.echo()
        click.echo(f"    Node: {checkpoint_data.get('node')}")
        click.echo(f"    Checkpoint ID: {checkpoint_data.get('checkpoint_id')}")
        click.echo(f"    Iteration: {checkpoint_data.get('iteration', 0)}")
        click.echo(f"    Timestamp: {checkpoint_data.get('timestamp')}")
        click.echo()
        
        state = checkpoint_data.get("state", {})
        
        # Summary info
        click.echo(click.style("  State Summary", fg="cyan", bold=True))
        click.echo(f"    Data points: {len(state.get('Q', []))}")
        click.echo(f"    Sample: {state.get('sample_description', 'N/A')[:50]}")
        click.echo(f"    Has model: {state.get('current_model') is not None}")
        click.echo(f"    Fit results: {len(state.get('fit_results', []))}")
        click.echo(f"    Current χ²: {state.get('current_chi2', 'N/A')}")
        click.echo(f"    Error: {state.get('error', 'None')}")
        click.echo()
        
        if show_state:
            click.echo(click.style("  Full State", fg="cyan", bold=True))
            # Pretty print state, but truncate long arrays
            for key, value in state.items():
                if isinstance(value, list) and len(value) > 5:
                    click.echo(f"    {key}: [{len(value)} items]")
                elif isinstance(value, str) and len(value) > 100:
                    click.echo(f"    {key}: {value[:100]}...")
                else:
                    click.echo(f"    {key}: {value}")


# ============================================================================
# Plotting Commands
# ============================================================================

@cli.command("plot-results")
@click.argument("output_dir", type=click.Path(exists=True))
@click.option(
    "--save", "-s",
    type=click.Path(),
    help="Save plot to file (PNG, PDF, SVG)",
)
@click.option(
    "--offset", "-f",
    default=10.0,
    help="Vertical offset factor between R(Q) curves (default: 10)",
)
@click.option(
    "--no-show",
    is_flag=True,
    help="Don't display plot interactively (useful with --save)",
)
@click.option(
    "--workspace", "-w",
    type=click.Path(exists=True),
    help="Workspace directory where data files are located (default: current directory)",
)
def plot_results(output_dir: str, save: Optional[str], offset: float, no_show: bool, workspace: Optional[str]):
    """
    Plot reflectivity and SLD profiles from workflow checkpoints.
    
    OUTPUT_DIR: Path to the output directory containing checkpoints
    
    Creates a two-panel plot:
    - Left: R(Q) curves (log-log) with data and model predictions from each iteration
    - Right: SLD profiles for each model iteration
    
    Examples:
    
        # Interactive plot
        python -m aure.cli plot-results ./results
        
        # Save to file
        python -m aure.cli plot-results ./results -s results.png
        
        # Save without display (from a different directory)
        python -m aure.cli plot-results /path/to/results -s results.pdf --no-show -w /path/to/workspace
    """
    from pathlib import Path
    import os
    from .workflow import CheckpointManager
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        click.echo(click.style("Error: matplotlib is required for plotting", fg="red"))
        click.echo("Install with: pip install matplotlib")
        sys.exit(1)
    
    output_path = Path(output_dir)
    models_dir = output_path / "models"
    checkpoints_dir = output_path / "checkpoints"
    
    # Use provided workspace or current directory
    working_dir = Path(workspace) if workspace else Path.cwd()
    
    # Check for model files
    if not models_dir.exists():
        click.echo(click.style("No models directory found", fg="red"))
        sys.exit(1)
    
    model_files = sorted(models_dir.glob("*.py"))
    if not model_files:
        click.echo(click.style("No model files found in models/", fg="red"))
        sys.exit(1)
    
    click.echo(click.style(f"  Found {len(model_files)} model file(s)", fg="cyan"))
    
    # Get list of checkpoints for experimental data
    checkpoint_list = CheckpointManager.list_checkpoints(output_dir)
    
    if not checkpoint_list:
        click.echo(click.style("No checkpoints found in directory", fg="red"))
        sys.exit(1)
    
    # Load data from first checkpoint
    first_cp = CheckpointManager.load_checkpoint(
        str(checkpoints_dir / checkpoint_list[0]['file'])
    )
    state = first_cp['state']
    Q_data = np.array(state.get('Q', []))
    R_data = np.array(state.get('R', []))
    dR_data = np.array(state.get('dR', []))
    sample_desc = state.get('sample_description', 'Unknown sample')[:50]
    
    if len(Q_data) == 0:
        click.echo(click.style("No experimental data found in checkpoints", fg="red"))
        sys.exit(1)
    
    # Get chi-squared values from fitting checkpoints
    # Map by (node, iteration) for more precise matching
    chi2_by_node = {}
    for cp_info in checkpoint_list:
        if cp_info['node'] in ('fitting', 'evaluation', 'refinement'):
            cp_path = str(checkpoints_dir / cp_info['file'])
            cp_data = CheckpointManager.load_checkpoint(cp_path)
            cp_state = cp_data['state']
            chi2 = cp_state.get('current_chi2')
            node = cp_info['node']
            iteration = cp_info.get('iteration', 0)
            if chi2 is not None:
                chi2_by_node[(node, iteration)] = chi2
    
    # Load and execute each model file to get R(Q) and SLD
    model_data = []
    import re
    for model_file in model_files:
        try:
            # Parse iteration and node type from filename
            # Filenames like: model_initial.py, model_fitting_iter0.py, model_refined_iter1.py
            name = model_file.stem
            
            if 'initial' in name:
                iteration = 0
                node_type = 'initial'
                label = "Initial"
                sort_key = (0, 0)  # (iteration, sub-order)
            elif 'fitting' in name:
                match = re.search(r'iter(\d+)', name)
                iteration = int(match.group(1)) if match else 0
                node_type = 'fitting'
                label = f"Fit iter {iteration}"
                sort_key = (iteration, 1)
            elif 'evaluation' in name:
                match = re.search(r'iter(\d+)', name)
                iteration = int(match.group(1)) if match else 0
                node_type = 'evaluation'
                label = f"Eval iter {iteration}"
                sort_key = (iteration, 2)
            elif 'refined' in name:
                match = re.search(r'iter(\d+)', name)
                iteration = int(match.group(1)) if match else 0
                node_type = 'refinement'
                label = f"Refined iter {iteration}"
                sort_key = (iteration, 3)
            else:
                iteration = len(model_data)
                node_type = 'unknown'
                label = name
                sort_key = (iteration, 0)
            
            click.echo(f"    Loading {model_file.name}...")
            result = _execute_model_file(model_file, Q_data, working_dir=working_dir)
            
            if result:
                # Look up chi2 by node type and iteration
                # Only show checkpoint chi2 for fitted models (not initial)
                chi2 = None
                if node_type != 'initial':
                    chi2 = chi2_by_node.get((node_type, iteration))
                    # Fallback to fitting chi2 for evaluation/refinement same iteration
                    if chi2 is None:
                        chi2 = chi2_by_node.get(('fitting', iteration))
                
                model_data.append({
                    'iteration': iteration,
                    'sort_key': sort_key,
                    'label': label,
                    'Q': result['Q'],
                    'R': result['R'],
                    'z': result.get('z'),
                    'sld': result.get('sld'),
                    'chi2': chi2,
                    'file': model_file.name,
                })
        except Exception as e:
            click.echo(click.style(f"    Warning: Could not load {model_file.name}: {e}", fg="yellow"))
    
    if not model_data:
        click.echo(click.style("No models could be loaded", fg="red"))
        sys.exit(1)
    
    # Sort by sort_key (iteration, sub-order)
    model_data.sort(key=lambda x: x.get('sort_key', (x['iteration'], 0)))
    
    click.echo(click.style(f"  Plotting {len(model_data)} model(s)", fg="cyan"))
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ========== Left panel: R(Q) curves ==========
    n_models = len(model_data)
    colors = plt.cm.viridis(np.linspace(0, 0.85, n_models))
    
    # Plot data (at top with maximum offset)
    base_offset = offset ** n_models
    ax1.errorbar(
        Q_data, R_data * base_offset, yerr=dR_data * base_offset,
        fmt='o', markersize=2, color='gray', alpha=0.6,
        label='Data', capsize=0, zorder=1,
    )
    
    # Plot models with decreasing offsets
    for i, md in enumerate(model_data):
        offset_factor = offset ** (n_models - i - 1)
        
        label = md['label']
        if md['chi2'] is not None:
            label += f" (χ²={md['chi2']:.1f})"
        
        # Plot model curve
        ax1.plot(
            md['Q'], np.array(md['R']) * offset_factor,
            '-', color=colors[i], linewidth=1.5,
            label=label, zorder=2,
        )
        
        # Plot data at same offset (faded)
        if offset_factor != base_offset:
            ax1.errorbar(
                Q_data, R_data * offset_factor, yerr=dR_data * offset_factor,
                fmt='o', markersize=1.5, color='gray', alpha=0.3, capsize=0, zorder=0,
            )
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Q (Å⁻¹)')
    ax1.set_ylabel('Reflectivity (offset)')
    ax1.set_title(f'R(Q) - {sample_desc}')
    ax1.legend(loc='lower left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # ========== Right panel: SLD profiles ==========
    has_sld = any(md['z'] is not None for md in model_data)
    
    if has_sld:
        for i, md in enumerate(model_data):
            if md['z'] is not None and md['sld'] is not None:
                ax2.plot(
                    md['z'], md['sld'],
                    '-', color=colors[i], linewidth=1.5,
                    label=md['label'],
                )
        
        ax2.set_xlabel('Depth (Å)')
        ax2.set_ylabel('SLD (×10⁻⁶ Å⁻²)')
        ax2.set_title('SLD Profile')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'SLD profiles not available',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('SLD Profile')
    
    plt.tight_layout()
    
    # Save if requested
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        click.echo(click.style(f"  Plot saved to: {save}", fg="green"))
    
    # Show if requested
    if not no_show:
        plt.show()
    
    plt.close()


def _execute_model_file(model_file, Q_data, working_dir=None) -> Optional[dict]:
    """
    Execute a refl1d model file and extract R(Q) and SLD profile.
    
    Args:
        model_file: Path to the model .py file
        Q_data: Experimental Q values (for reference, may not be used)
        working_dir: Directory to run the model in (for relative paths)
    
    Returns:
        Dictionary with 'Q', 'R', 'z', 'sld', 'chi2' or None on failure
    """
    import numpy as np
    from pathlib import Path
    import os
    
    # Save current directory
    original_cwd = os.getcwd()
    
    try:
        model_script = Path(model_file).read_text()
        
        # Change to working directory for relative data paths
        if working_dir and Path(working_dir).exists():
            os.chdir(working_dir)
        
        # Execute model script to get the problem
        model_globals = {"__file__": str(model_file)}
        exec(compile(model_script, model_file, 'exec'), model_globals)
        
        # First check for experiment directly (simpler models)
        experiment = model_globals.get("experiment")
        problem = model_globals.get("problem")
        
        if experiment is None and problem is not None:
            # Get the experiment from the problem
            if hasattr(problem, 'fitness'):
                fitness = problem.fitness
            else:
                fitness = problem
            
            if hasattr(fitness, '_models'):
                experiment = fitness._models[0]
            elif hasattr(fitness, 'reflectivity'):
                experiment = fitness
        
        if experiment is None:
            return None
        
        # Compute reflectivity using the probe's Q values
        # reflectivity() returns (Q, R) tuple
        result = experiment.reflectivity()
        if isinstance(result, tuple) and len(result) == 2:
            Q, R = result
        else:
            # Some versions may return just R
            Q = experiment.probe.Q
            R = result
        
        Q = np.array(Q)
        R = np.array(R)
        
        # Get chi-squared if available
        chi2 = None
        if problem is not None:
            try:
                chi2 = problem.chisq()
            except Exception:
                pass
        
        # Extract SLD profile from the experiment (not sample)
        z, sld = None, None
        try:
            z_arr, sld_arr, _ = experiment.smooth_profile(dz=1.0)
            z = np.array(z_arr)
            sld = np.array(sld_arr)
        except Exception:
            pass
        
        return {
            'Q': Q.tolist(),
            'R': R.tolist() if hasattr(R, 'tolist') else list(R),
            'z': z.tolist() if z is not None else None,
            'sld': sld.tolist() if sld is not None else None,
            'chi2': chi2,
        }
    except Exception as e:
        raise RuntimeError(f"Model execution failed: {e}")
    finally:
        # Restore original directory
        os.chdir(original_cwd)


# ============================================================================
# Material Database Commands
# ============================================================================

@cli.command("lookup-sld")
@click.argument("materials", nargs=-1, required=True)
@click.option(
    "--wavelength", "-w",
    default=1.8,
    help="Neutron wavelength in Angstroms (default: 1.8)",
)
@click.option(
    "--json", "output_json",
    is_flag=True,
    help="Output as JSON",
)
def lookup_sld(materials: tuple, wavelength: float, output_json: bool):
    """
    Look up SLD values for materials.
    
    MATERIALS: One or more material names or chemical formulas
    
    Examples:
    
        python -m aure.cli lookup-sld silicon gold D2O
        
        python -m aure.cli lookup-sld SiO2 Fe2O3 TiO2
        
        python -m aure.cli lookup-sld polystyrene PMMA
    """
    from .database.materials import get_sld, lookup_material
    
    results = []
    for mat in materials:
        try:
            sld = get_sld(mat)
            info = lookup_material(mat)
            results.append({
                "material": mat,
                "sld": round(sld, 4),
                "density": info.density if info else None,
                "formula": info.formula if info else mat,
            })
        except Exception as e:
            results.append({"material": mat, "error": str(e)})
    
    if output_json:
        click.echo(json.dumps(results, indent=2))
    else:
        click.echo()
        click.echo(click.style("  Material SLD Values", fg="cyan", bold=True))
        click.echo(f"  Wavelength: {wavelength} Å")
        click.echo()
        
        # Table header
        click.echo(f"  {'Material':<20} {'SLD (10⁻⁶ Å⁻²)':<18} {'Formula'}")
        click.echo(f"  {'-'*20} {'-'*18} {'-'*20}")
        
        for r in results:
            if "error" in r:
                click.echo(f"  {r['material']:<20} "
                          + click.style(f"Error: {r['error']}", fg="red"))
            else:
                click.echo(f"  {r['material']:<20} {r['sld']:<18.4f} {r.get('formula', '')}")
        click.echo()


@cli.command("list-materials")
@click.option(
    "--category", "-c",
    type=click.Choice(["polymers", "metals", "substrates", "solvents", "all"]),
    default="all",
    help="Filter by category",
)
def list_materials(category: str):
    """
    List common materials in the database.
    
    Shows materials with their SLD values for quick reference.
    """
    from .database.materials import get_sld
    
    click.echo()
    click.echo(click.style("  Material Database", fg="cyan", bold=True))
    click.echo()
    
    # Categories
    categories = {
        "polymers": ["polystyrene", "d-polystyrene", "PMMA", "d-PMMA", "PEO", "PDMS"],
        "metals": ["gold", "silver", "nickel", "titanium", "copper", "iron"],
        "substrates": ["silicon", "sapphire", "quartz", "glass"],
        "solvents": ["air", "D2O", "H2O", "toluene", "ethanol"],
    }
    
    if category == "all":
        for cat_name, mats in categories.items():
            click.echo(click.style(f"  {cat_name.title()}", bold=True))
            for mat in mats:
                try:
                    sld = get_sld(mat)
                    click.echo(f"    {mat:<20} SLD = {sld:>7.3f}")
                except Exception:
                    pass
            click.echo()
    else:
        mats = categories.get(category, [])
        click.echo(click.style(f"  {category.title()}", bold=True))
        for mat in mats:
            try:
                sld = get_sld(mat)
                click.echo(f"    {mat:<20} SLD = {sld:>7.3f}")
            except Exception:
                pass
        click.echo()


# ============================================================================
# Feature Extraction Commands
# ============================================================================

@cli.command("extract-features")
@click.argument("data_file", type=click.Path(exists=True))
@click.option(
    "--json", "output_json",
    is_flag=True,
    help="Output as JSON",
)
def extract_features(data_file: str, output_json: bool):
    """
    Extract physics features from a reflectivity file.
    
    Analyzes the data to estimate thickness, roughness, and layer count
    without building a full model.
    
    DATA_FILE: Path to the reflectivity data file
    """
    from .tools.data_tools import load_reflectivity_data
    from .tools.feature_tools import (
        estimate_total_thickness,
        estimate_roughness,
        extract_critical_edges,
        estimate_layer_count,
    )
    import numpy as np
    
    # Load data
    try:
        Q, R, dR = load_reflectivity_data(data_file)
    except Exception as e:
        click.echo(click.style(f"Error loading data: {e}", fg="red"))
        sys.exit(1)
    
    Q = np.array(Q)
    R = np.array(R)
    
    features = {}
    
    # Critical edge
    try:
        qc = extract_critical_edges(Q, R)
        features["critical_edge"] = qc
    except Exception as e:
        features["critical_edge"] = {"error": str(e)}
    
    # Thickness
    try:
        thickness = estimate_total_thickness(Q, R)
        features["thickness"] = thickness
    except Exception as e:
        features["thickness"] = {"error": str(e)}
    
    # Roughness
    try:
        roughness = estimate_roughness(Q, R)
        features["roughness"] = roughness
    except Exception as e:
        features["roughness"] = {"error": str(e)}
    
    # Layer count
    try:
        layers = estimate_layer_count(Q, R)
        features["layer_count"] = layers
    except Exception as e:
        features["layer_count"] = {"error": str(e)}
    
    features["data"] = {
        "file": data_file,
        "n_points": len(Q),
        "q_min": float(Q.min()),
        "q_max": float(Q.max()),
    }
    
    if output_json:
        click.echo(json.dumps(features, indent=2))
    else:
        click.echo()
        click.echo(click.style("  Feature Extraction", fg="cyan", bold=True))
        click.echo(f"  File: {data_file}")
        click.echo(f"  Points: {len(Q)}")
        click.echo(f"  Q range: {Q.min():.4f} - {Q.max():.4f} Å⁻¹")
        click.echo()
        
        if "Qc" in features.get("critical_edge", {}):
            qc = features["critical_edge"]["Qc"]
            sld = features["critical_edge"].get("estimated_SLD", 0)
            click.echo(f"  Critical edge: Qc = {qc:.5f} Å⁻¹ (SLD ≈ {sld:.2f})")
        
        if "thickness" in features.get("thickness", {}):
            t = features["thickness"]["thickness"]
            n = features["thickness"].get("n_fringes", 0)
            click.echo(f"  Thickness: {t:.0f} Å ({n} fringes)")
        
        if "roughness" in features.get("roughness", {}):
            r = features["roughness"]["roughness"]
            click.echo(f"  Roughness: {r:.1f} Å")
        
        if "n_layers" in features.get("layer_count", {}):
            n = features["layer_count"]["n_layers"]
            conf = features["layer_count"].get("confidence", "unknown")
            click.echo(f"  Estimated layers: {n} ({conf} confidence)")
        
        click.echo()


# ============================================================================
# MCP Server Command
# ============================================================================

@cli.command("mcp-server")
@click.option(
    "--transport", "-t",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport protocol (stdio for Claude Desktop, sse for HTTP)",
)
@click.option(
    "--port", "-p",
    default=8000,
    help="Port for SSE transport (default: 8000)",
)
def mcp_server(transport: str, port: int):
    """
    Start the MCP server for AI assistant integration.
    
    This starts a Model Context Protocol server that allows AI assistants
    like Claude to interact with the reflectivity analysis workflow.
    
    For Claude Desktop, use stdio transport (default).
    For HTTP-based clients, use sse transport.
    
    Examples:
    
        python -m aure.cli mcp-server
        
        python -m aure.cli mcp-server --transport sse --port 8080
    """
    from .mcp_server import mcp
    
    click.echo(click.style("═" * 60, fg="blue"))
    click.echo(click.style("  Reflectivity Analysis MCP Server", fg="blue", bold=True))
    click.echo(click.style("═" * 60, fg="blue"))
    click.echo()
    click.echo(f"  Transport: {transport}")
    if transport == "sse":
        click.echo(f"  Port: {port}")
    click.echo()
    click.echo("  Available tools:")
    click.echo("    - lookup_material_sld")
    click.echo("    - compare_materials")
    click.echo("    - analyze_reflectivity_features")
    click.echo("    - start_analysis_session")
    click.echo("    - get_session_model")
    click.echo("    - run_fit")
    click.echo("    - evaluate_fit")
    click.echo("    - modify_model")
    click.echo("    - quick_analyze")
    click.echo()
    click.echo("  Starting server...")
    click.echo()
    
    if transport == "sse":
        mcp.run(transport="sse", port=port)
    else:
        mcp.run(transport="stdio")


# ============================================================================
# Web Viewer Command
# ============================================================================

@cli.command("serve")
@click.argument("output_dir", type=click.Path(exists=True))
@click.option(
    "--port", "-p",
    default=5000,
    type=int,
    help="Port to run the web server on (default: 5000)",
)
@click.option(
    "--no-browser",
    is_flag=True,
    help="Don't open a browser automatically",
)
def serve(output_dir: str, port: int, no_browser: bool):
    """
    Launch a web viewer for workflow results.

    OUTPUT_DIR: Path to the output directory from 'aure analyze -o'

    Opens a local Flask app with two tabs:

    \b
      History  – checkpoint timeline and χ² progression chart
      Results  – R(Q) plot, SLD profile, and fit parameter table

    Examples:

        aure serve ./output

        aure serve ./output --port 8080 --no-browser
    """
    from .web import create_app

    click.echo(click.style("═" * 60, fg="blue"))
    click.echo(click.style("  AuRE – Results Viewer", fg="blue", bold=True))
    click.echo(click.style("═" * 60, fg="blue"))
    click.echo()
    click.echo(f"  Output dir: {output_dir}")
    click.echo(f"  URL:        http://127.0.0.1:{port}")
    click.echo()

    app = create_app(output_dir)

    if not no_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, webbrowser.open, args=[f"http://127.0.0.1:{port}"]).start()

    app.run(host="127.0.0.1", port=port, debug=False)


# ============================================================================
# Interactive Mode (Future)
# ============================================================================

@cli.command("interactive")
@click.argument("data_file", type=click.Path(exists=True), required=False)
def interactive(data_file: Optional[str]):
    """
    Start an interactive analysis session.
    
    [Not yet implemented - placeholder for future REPL mode]
    """
    click.echo()
    click.echo(click.style("  Interactive mode not yet implemented.", fg="yellow"))
    click.echo("  Use the 'analyze' command or MCP server for now.")
    click.echo()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
