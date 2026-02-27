"""
Batch validation runner.

Runs the aure workflow on every dataset in the validation inventory,
passing the appropriate experiment context as the hypothesis.
Results are saved to validation/results/<run>/.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()  # Ensure .env is loaded before any aure imports

from .inventory import build_inventory, ReferenceModel, DATA_DIR


RESULTS_DIR = Path(__file__).parent / "results"


def run_single(
    ref: ReferenceModel,
    output_dir: Path,
    max_iterations: int = 5,
    verbose: bool = False,
) -> dict:
    """
    Run the aure workflow on a single validation dataset.

    Returns a summary dict with keys: run, sample, experiment, success,
    chi_squared, n_iterations, error, elapsed_s.
    """
    from aure.workflow import run_analysis

    summary = {
        "run": ref.run,
        "sample": ref.sample,
        "experiment": ref.experiment,
        "success": False,
        "chi_squared": None,
        "best_chi2": None,
        "n_iterations": 0,
        "error": None,
        "elapsed_s": 0.0,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        result = run_analysis(
            data_file=str(ref.data_file),
            sample_description=ref.context,   # full context as description
            hypothesis=None,
            max_iterations=max_iterations,
            output_dir=str(output_dir),
        )

        summary["success"] = not bool(result.get("error"))
        summary["n_iterations"] = result.get("iteration", 0)
        summary["best_chi2"] = result.get("best_chi2")

        fit_results = result.get("fit_results") or []
        if fit_results:
            summary["chi_squared"] = fit_results[-1].get("chi_squared")

        if result.get("error"):
            summary["error"] = result["error"]

        # Persist the final state for later comparison
        _save_state_summary(result, output_dir / "state_summary.json")

    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        if verbose:
            traceback.print_exc()

    summary["elapsed_s"] = round(time.time() - t0, 1)
    return summary


def run_all(
    data_dir: Path = DATA_DIR,
    results_dir: Path = RESULTS_DIR,
    max_iterations: int = 5,
    runs: Optional[List[str]] = None,
    verbose: bool = False,
) -> List[dict]:
    """
    Run validation across all (or selected) datasets.

    Args:
        data_dir: Directory containing reference data & JSONs.
        results_dir: Root directory for output.
        max_iterations: Max refinement iterations per dataset.
        runs: If given, only run these run numbers.
        verbose: Print progress to stdout.

    Returns:
        List of per-run summary dicts.
    """
    inventory = build_inventory(data_dir)

    if runs:
        inventory = {r: ref for r, ref in inventory.items() if r in runs}

    if not inventory:
        print("No datasets matched.")
        return []

    results_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[dict] = []

    total = len(inventory)
    for idx, (run, ref) in enumerate(inventory.items(), 1):
        label = f"[{idx}/{total}] Run {run} ({ref.sample}, exp {ref.experiment})"
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  {label}")
            print(f"{'=' * 60}")

        out = results_dir / run
        summary = run_single(ref, out, max_iterations=max_iterations, verbose=verbose)
        summaries.append(summary)

        if verbose:
            status = "OK" if summary["success"] else "FAIL"
            chi = summary["chi_squared"]
            chi_str = f"χ²={chi:.3f}" if chi is not None else "no fit"
            print(f"  {status}  {chi_str}  ({summary['elapsed_s']}s)")

    # Save aggregate summary
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    if verbose:
        ok = sum(1 for s in summaries if s["success"])
        print(f"\nDone: {ok}/{total} succeeded.  Summary → {summary_path}")

    return summaries


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _save_state_summary(state: dict, path: Path) -> None:
    """Save a JSON-serialisable subset of the workflow state."""
    subset = {
        "sample_description": state.get("sample_description"),
        "hypothesis": state.get("hypothesis"),
        "parsed_sample": state.get("parsed_sample"),
        "current_model": state.get("current_model"),
        "best_model": state.get("best_model"),
        "current_chi2": state.get("current_chi2"),
        "best_chi2": state.get("best_chi2"),
        "iteration": state.get("iteration"),
        "workflow_complete": state.get("workflow_complete"),
        "error": state.get("error"),
    }
    # Include last fit result parameters
    fits = state.get("fit_results") or []
    if fits:
        last = fits[-1]
        subset["last_fit"] = {
            "chi_squared": last.get("chi_squared"),
            "parameters": last.get("parameters"),
            "uncertainties": last.get("uncertainties"),
        }
    with open(path, "w") as f:
        json.dump(subset, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run validation batch")
    parser.add_argument("--runs", nargs="*", help="Specific run numbers to process")
    parser.add_argument("--max-iter", type=int, default=5, help="Max refinement iterations")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    run_all(
        runs=args.runs,
        max_iterations=args.max_iter,
        verbose=args.verbose,
    )
