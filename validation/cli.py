"""
Validation CLI — diagnostics and comparison tools.

Usage:
    python -m validation.cli compare [--runs RUN ...]
    python -m validation.cli chi2    [--runs RUN ...] [--fix-summary]
    python -m validation.cli diagnose [--runs RUN ...]

Sub-commands
------------
compare     Compare fitted results against reference models (delegates to
            comparator.compare_all / compare_single).
chi2        Show χ² progression through checkpoints for each run.
            With --fix-summary, also creates missing state_summary.json
            from the best checkpoint.
diagnose    Detailed diagnostic: intake parse, ambient correction, initial
            model, χ² progression, final params, evaluation feedback.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import List, Optional

from .comparator import (
    RESULTS_DIR,
    compare_all,
    compare_single,
    compute_aggregate,
    print_aggregate,
    print_comparison,
)
from .inventory import DATA_DIR, build_inventory


# ── helpers ───────────────────────────────────────────────────────────────────

_STATE_SUMMARY_KEYS = (
    "sample_description",
    "hypothesis",
    "parsed_sample",
    "current_model",
    "best_model",
    "current_chi2",
    "best_chi2",
    "iteration",
    "workflow_complete",
    "error",
)


def _resolve_runs(
    requested: Optional[List[str]],
    results_dir: Path = RESULTS_DIR,
) -> List[str]:
    """Return sorted list of run IDs that have result directories."""
    available = sorted(
        d.name
        for d in results_dir.iterdir()
        if d.is_dir() and (d / "checkpoints").is_dir()
    )
    if not requested:
        return available
    missing = set(requested) - set(available)
    if missing:
        print(f"WARNING: no results for {', '.join(sorted(missing))}")
    return sorted(r for r in requested if r in available)


def _create_state_summary(run: str, results_dir: Path = RESULTS_DIR) -> Optional[Path]:
    """Create state_summary.json from the best checkpoint if missing.

    Returns the path if created, else None.
    """
    summary_path = results_dir / run / "state_summary.json"
    if summary_path.exists():
        return None

    cp_dir = results_dir / run / "checkpoints"
    if not cp_dir.is_dir():
        return None

    best_chi2 = float("inf")
    best_cp: Optional[str] = None
    for cp in sorted(os.listdir(cp_dir)):
        with open(cp_dir / cp) as f:
            data = json.load(f)
        state = data.get("state", {})
        fr = state.get("fit_results", [])
        chi2 = fr[-1].get("chi_squared") if fr else None
        if chi2 is not None and chi2 < best_chi2:
            best_chi2 = chi2
            best_cp = cp

    if best_cp is None:
        return None

    with open(cp_dir / best_cp) as f:
        state = json.load(f)["state"]

    subset = {k: state.get(k) for k in _STATE_SUMMARY_KEYS}
    fits = state.get("fit_results") or []
    if fits:
        last = fits[-1]
        subset["last_fit"] = {
            "chi_squared": last.get("chi_squared"),
            "parameters": last.get("parameters"),
            "uncertainties": last.get("uncertainties"),
        }

    with open(summary_path, "w") as f:
        json.dump(subset, f, indent=2, default=str)

    return summary_path


# ── sub-commands ──────────────────────────────────────────────────────────────


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare fitted results against reference models."""
    inv = build_inventory()
    runs = _resolve_runs(args.runs)

    if not runs:
        print("No results to compare.  Run the batch runner first.")
        return

    results: list = []
    for run in runs:
        if run not in inv:
            print(f"{run}: no reference model in inventory — skipped")
            continue
        comp = compare_single(inv[run], RESULTS_DIR)
        if comp is None:
            # attempt to fix a missing summary first
            created = _create_state_summary(run)
            if created:
                comp = compare_single(inv[run], RESULTS_DIR)
        if comp is not None:
            print_comparison(comp)
            results.append(comp)
        else:
            print(f"\n{run}: no state_summary.json (run may have been interrupted)")
        print()

    if len(results) > 1:
        stats = compute_aggregate(results)
        print_aggregate(stats)


def cmd_chi2(args: argparse.Namespace) -> None:
    """Show χ² progression through checkpoints."""
    runs = _resolve_runs(args.runs)

    for run in runs:
        cp_dir = RESULTS_DIR / run / "checkpoints"
        if not cp_dir.is_dir():
            continue

        print(f"\n{'=' * 60}")
        print(f"  {run}  χ² progression")
        print(f"{'=' * 60}")

        best_chi2 = float("inf")
        best_cp = None
        for cp in sorted(os.listdir(cp_dir)):
            with open(cp_dir / cp) as f:
                data = json.load(f)
            state = data.get("state", {})
            fr = state.get("fit_results", [])
            chi2 = fr[-1].get("chi_squared") if fr else None
            best = state.get("best_chi2")
            it = state.get("iteration", 0)
            chi2_s = f"{chi2:.3f}" if chi2 is not None else "---"
            best_s = f"{best:.3f}" if best is not None else "---"
            print(f"  {cp:40s}  chi2={chi2_s:>10s}  best={best_s:>10s}  iter={it}")
            if chi2 is not None and chi2 < best_chi2:
                best_chi2 = chi2
                best_cp = cp

        if args.fix_summary:
            created = _create_state_summary(run)
            if created:
                print(f"  → created {created.name} from {best_cp} (χ²={best_chi2:.3f})")


def cmd_diagnose(args: argparse.Namespace) -> None:
    """Detailed diagnostic for each run."""
    runs = _resolve_runs(args.runs)

    for run in runs:
        base = RESULTS_DIR / run
        cp_dir = base / "checkpoints"
        if not cp_dir.is_dir():
            continue

        print(f"\n{'=' * 60}")
        print(f"  RUN {run}")
        print(f"{'=' * 60}")

        # ── Intake ────────────────────────────────────────────
        intake = cp_dir / "001_intake.json"
        if intake.exists():
            state = json.load(open(intake))["state"]
            ps = state.get("parsed_sample") or {}
            print("\n--- INTAKE: parsed_sample ---")
            print(f"  ambient:   {ps.get('ambient')}")
            for lay in ps.get("layers", []):
                print(
                    f"  layer: {lay['name']:30s}  sld={lay.get('sld', '?'):>6}  thick={lay.get('thickness', '?')}"
                )
            print(f"  substrate: {ps.get('substrate')}")
            print(f"  intensity: {ps.get('intensity')}")

        # ── Analysis ──────────────────────────────────────────
        analysis = cp_dir / "002_analysis.json"
        if analysis.exists():
            state = json.load(open(analysis))["state"]
            ps = state.get("parsed_sample") or {}
            print(f"\n--- ANALYSIS: corrected ambient ---")
            print(f"  ambient: {ps.get('ambient')}")

        # ── Initial model key lines ───────────────────────────
        modeling = cp_dir / "003_modeling.json"
        if modeling.exists():
            state = json.load(open(modeling))["state"]
            model = state.get("current_model", "")
            print(f"\n--- INITIAL MODEL ---")
            keywords = ("sld(", "sample =", "range", "intensity", "material")
            for i, line in enumerate(model.split("\n"), 1):
                if line.strip() and any(k in line.strip().lower() for k in keywords):
                    print(f"  {i:3d}: {line}")

        # ── χ² progression ────────────────────────────────────
        checkpoints = sorted(cp_dir.glob("*.json"))
        print(f"\n--- CHI2 PROGRESSION ---")
        for cp in checkpoints:
            d = json.load(open(cp))
            s = d.get("state", d)
            chi2 = s.get("current_chi2")
            best = s.get("best_chi2")
            it = s.get("iteration", 0)
            if chi2 is not None:
                print(f"  {cp.name:40s}  chi2={chi2:8.2f}  best={best:8.2f}  iter={it}")

        # ── Final params ──────────────────────────────────────
        summary_file = base / "state_summary.json"
        if summary_file.exists():
            summary = json.load(open(summary_file))
        else:
            # Fall back to last checkpoint
            last_cp = checkpoints[-1] if checkpoints else None
            if last_cp:
                summary = json.load(open(last_cp)).get("state", {})
            else:
                summary = {}

        lf = summary.get("last_fit") or {}
        params = lf.get("parameters") or {}
        if params:
            print(f"\n--- FINAL PARAMS ---")
            for k, v in params.items():
                if isinstance(v, (int, float)):
                    print(f"  {k:45s}: {v:10.4f}")
                else:
                    print(f"  {k:45s}: {v}")

        # ── Evaluation feedback ───────────────────────────────
        for cp in checkpoints:
            if "evaluation" not in cp.name:
                continue
            d = json.load(open(cp))
            s = d.get("state", d)
            fr = s.get("fit_results", [])
            if not fr:
                continue
            last = fr[-1]
            issues = last.get("issues", [])
            suggs = last.get("suggestions", [])
            if issues or suggs:
                print(f"\n--- {cp.name} EVALUATION FEEDBACK ---")
                for iss in issues:
                    print(f"  ISSUE: {iss}")
                for sg in suggs:
                    print(f"  SUGGEST: {sg}")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m validation.cli",
        description="Validation diagnostics and comparison tools.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- compare --
    p_cmp = sub.add_parser("compare", help="Compare results to reference models")
    p_cmp.add_argument("--runs", nargs="*", help="Specific run numbers (default: all)")

    # -- chi2 --
    p_chi = sub.add_parser("chi2", help="Show χ² progression per run")
    p_chi.add_argument("--runs", nargs="*", help="Specific run numbers (default: all)")
    p_chi.add_argument(
        "--fix-summary",
        action="store_true",
        help="Create missing state_summary.json from best checkpoint",
    )

    # -- diagnose --
    p_diag = sub.add_parser("diagnose", help="Detailed per-run diagnostic")
    p_diag.add_argument("--runs", nargs="*", help="Specific run numbers (default: all)")

    args = parser.parse_args()

    dispatch = {
        "compare": cmd_compare,
        "chi2": cmd_chi2,
        "diagnose": cmd_diagnose,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
