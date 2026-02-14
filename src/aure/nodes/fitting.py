"""
FITTING node: Run refl1d optimization.

This node executes the generated refl1d model script to fit the data.
Supports multiple fitting methods:
- 'lm': Levenberg-Marquardt (fast, local optimizer)
- 'de': Differential Evolution (global optimizer)
- 'dream': MCMC for uncertainty quantification
"""

import os
import re
import json
import logging
import tempfile
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..state import ReflectivityState, FitResult, Message

logger = logging.getLogger(__name__)


def fitting_node(state: ReflectivityState) -> Dict[str, Any]:
    """
    Run refl1d fit on current model.
    
    Args:
        state: Current workflow state
    
    Returns:
        State updates including fit results
    """
    updates = {
        "current_node": "fitting",
        "messages": [],
        "fit_results": [],
    }
    
    model_script = state.get("current_model")
    if not model_script:
        updates["error"] = "No model to fit"
        return updates
    
    iteration = state.get("iteration", 0)
    method = os.environ.get("FIT_METHOD", "dream").lower()
    steps = int(os.environ.get("FIT_STEPS", "1000"))
    burn = int(os.environ.get("FIT_BURN", "1000"))
    logger.info(f"[FITTING] Starting iteration {iteration}")
    
    # ========== Run Fit ==========
    try:
        logger.info(f"[FITTING] Running {method.upper()} optimization...")
        result = run_refl1d_fit(
            model_script=model_script,
            method=method,
            iteration=iteration,
            steps=steps,
            burn=burn,
        )
        
        updates["fit_results"] = [result]
        updates["current_chi2"] = result["chi_squared"]
        logger.info(f"[FITTING] Completed with χ² = {result['chi_squared']:.3f}")
        
        # Update best chi2 and save best model
        best = state.get("best_chi2")
        if best is None or result["chi_squared"] < best:
            updates["best_chi2"] = result["chi_squared"]
            updates["best_model"] = model_script
            logger.info(f"[FITTING] New best χ² = {result['chi_squared']:.3f}")
        
        # Format message
        updates["messages"] = [Message(
            role="assistant",
            content=_format_fit_result(result),
            timestamp=None
        )]
        
    except Exception as e:
        updates["error"] = f"Fitting failed: {str(e)}"
        updates["messages"] = [Message(
            role="system",
            content=f"Error during fitting: {str(e)}",
            timestamp=None
        )]
    
    return updates


def run_refl1d_fit(
    model_script: str,
    method: str = "lm",
    iteration: int = 0,
    steps: int = 1000,
    burn: int = 1000,
) -> FitResult:
    """
    Execute refl1d fit using bumps.fit directly.
    
    Args:
        model_script: Python script defining the model
        method: Fitting method ('lm', 'de', 'dream')
        iteration: Current iteration number
        steps: Number of steps for MCMC methods
        burn: Number of burn-in steps for MCMC
    
    Returns:
        FitResult dictionary
    """
    from bumps.fitters import fit as bumps_fit
    from bumps.fitproblem import FitProblem
    
    # Create temporary directory for model execution
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write model script
        model_file = Path(tmpdir) / "model.py"
        model_file.write_text(model_script)
        
        # Execute model script to get the problem object
        model_globals = {"__file__": str(model_file)}
        exec(compile(model_script, model_file, 'exec'), model_globals)
        
        # Get the problem from the executed script
        problem = model_globals.get("problem")
        if problem is None:
            raise ValueError("Model script must define a 'problem' variable")
        
        # Configure fit options based on method
        fit_options = {
            "method": method,
            "parallel": 0,
        }
        
        if method == "dream":
            fit_options["samples"] = steps
            fit_options["burn"] = burn
            fit_options["pop"] = 10  # Population multiplier
        elif method == "de":
            fit_options["steps"] = steps
            fit_options["pop"] = 10
        elif method == "lm":
            fit_options["steps"] = steps
        
        # Run the fit
        logger.info(f"[FITTING] Running {method.upper()} with bumps.fit...")
        result = bumps_fit(problem, **fit_options)
        
        # Extract results
        return _extract_bumps_results(
            problem=problem,
            fit_result=result,
            method=method,
            iteration=iteration,
        )


def _extract_bumps_results(
    problem,
    fit_result,
    method: str,
    iteration: int,
) -> FitResult:
    """Extract fit results from bumps problem and fit result."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="bumps")

    # Get chi-squared (bumps uses chisq method on problem)
    chi_squared = problem.chisq()

    # Compute theory curve at fitted parameter values
    Q_fit = []
    R_fit = []
    try:
        experiment = problem.fitness
        experiment.update()
        Q_arr, R_arr = experiment.reflectivity()
        Q_fit = Q_arr.tolist()
        R_fit = R_arr.tolist()
    except Exception as e:
        logger.warning(f"[FITTING] Could not compute theory curve: {e}")
    
    # Get parameter values and names from problem._parameters
    parameters = {}
    uncertainties = {}
    
    for i, par in enumerate(problem._parameters):
        name = str(par.name)
        parameters[name] = par.value
        
        # Get uncertainties from fit result if available
        if hasattr(fit_result, 'dx') and fit_result.dx is not None:
            try:
                if i < len(fit_result.dx):
                    uncertainties[name] = fit_result.dx[i]
            except (IndexError, TypeError):
                pass
    
    # Check convergence
    converged = chi_squared < 100  # Simple heuristic
    if hasattr(fit_result, 'success'):
        converged = fit_result.success
    
    logger.info(f"[FITTING] Fit complete: χ² = {chi_squared:.3f}")
    for name, value in parameters.items():
        unc_str = f" ± {uncertainties.get(name, 0):.3f}" if name in uncertainties else ""
        logger.info(f"[FITTING]   {name}: {value:.3f}{unc_str}")
    
    return FitResult(
        iteration=iteration,
        method=method,
        chi_squared=chi_squared,
        converged=converged,
        parameters=parameters,
        uncertainties=uncertainties if uncertainties else None,
        Q_fit=Q_fit,
        R_fit=R_fit,
        residuals=[],
        issues=[],
        suggestions=[],
    )


def _format_fit_result(result: FitResult) -> str:
    """Format fit result for display."""
    lines = ["**Fit Results:**"]
    lines.append("")
    
    chi2 = result["chi_squared"]
    if chi2 < 2:
        quality = "✓ Excellent"
    elif chi2 < 5:
        quality = "○ Good"
    elif chi2 < 10:
        quality = "△ Acceptable"
    else:
        quality = "✗ Poor"
    
    lines.append(f"- **χ² = {chi2:.2f}** ({quality})")
    lines.append(f"- Method: {result['method'].upper()}")
    lines.append(f"- Converged: {'Yes' if result['converged'] else 'No'}")
    
    if result["parameters"]:
        lines.append("")
        lines.append("**Best-fit parameters:**")
        for param, value in list(result["parameters"].items())[:10]:
            lines.append(f"- {param}: {value:.3f}")
    
    if result["uncertainties"]:
        lines.append("")
        lines.append("**Uncertainties (1σ):**")
        for param, unc in list(result["uncertainties"].items())[:10]:
            lines.append(f"- {param}: ±{unc:.3f}")
    
    return "\n".join(lines)
