"""
Model comparator.

Compares aure workflow output against hand-fitted reference models.
Produces per-dataset and aggregate metrics.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .inventory import ReferenceModel, build_inventory, DATA_DIR


RESULTS_DIR = Path(__file__).parent / "results"


# ── Metric containers ────────────────────────────────────────────────────────


@dataclass
class ParamComparison:
    """Comparison of a single parameter between reference and fitted model."""

    layer: str  # reference layer name
    param: str  # thickness, interface, rho
    ref_value: float
    ref_p95: Optional[List[float]]  # 95% credible interval from reference
    fit_value: Optional[float]
    abs_error: Optional[float]
    within_p95: Optional[bool]
    fit_layer: Optional[str] = None  # fitted layer name (from positional match)


@dataclass
class LayerComparison:
    """Comparison result for a single dataset."""

    run: str
    sample: str
    experiment: int

    # Structure
    ref_layer_names: List[str]
    fit_layer_names: List[str]
    layer_count_match: bool
    missing_layers: List[str]
    extra_layers: List[str]

    # Chi-squared
    ref_chi2: Optional[float] = None
    fit_chi2: Optional[float] = None

    # Per-parameter comparisons
    params: List[ParamComparison] = field(default_factory=list)

    @property
    def param_errors(self) -> List[float]:
        return [p.abs_error for p in self.params if p.abs_error is not None]

    @property
    def frac_within_p95(self) -> Optional[float]:
        checks = [p.within_p95 for p in self.params if p.within_p95 is not None]
        if not checks:
            return None
        return sum(checks) / len(checks)


# ── Comparison logic ─────────────────────────────────────────────────────────


def compare_single(
    ref: ReferenceModel,
    results_dir: Path,
) -> Optional[LayerComparison]:
    """
    Compare a single run's output to its reference model.

    Looks for ``results_dir/<run>/state_summary.json`` which is written
    by the batch runner.
    """
    state_file = results_dir / ref.run / "state_summary.json"
    if not state_file.exists():
        return None

    with open(state_file) as f:
        state = json.load(f)

    # ── Layer names from fitted model ─────────────────────────
    fit_layer_names = _extract_layer_names_from_state(state)
    ref_layer_names = ref.layer_names

    # Position-based matching: anchor ambient (first) and substrate (last),
    # then match interior layers top-down.
    matched, missing_pos, extra_pos = _match_layers_by_position(
        ref_layer_names, fit_layer_names
    )

    comp = LayerComparison(
        run=ref.run,
        sample=ref.sample,
        experiment=ref.experiment,
        ref_layer_names=ref_layer_names,
        fit_layer_names=fit_layer_names,
        layer_count_match=(len(ref_layer_names) == len(fit_layer_names)),
        missing_layers=missing_pos,
        extra_layers=extra_pos,
    )

    # ── χ² ────────────────────────────────────────────────────
    comp.ref_chi2 = ref.chisq
    # Prefer best_chi2 from state, fall back to last_fit
    last_fit = state.get("last_fit") or {}
    comp.fit_chi2 = state.get("best_chi2") or last_fit.get("chi_squared")

    # ── Intensity ────────────────────────────────────────────
    fit_params = last_fit.get("parameters") or {}
    ref_intensity = ref.probe.get("intensity", {})
    if isinstance(ref_intensity, dict) and not ref_intensity.get("fixed"):
        ref_int_val = ref_intensity.get("value")
        ref_int_p95 = ref_intensity.get("p95")
        # Find intensity in fit params
        fit_int_val = None
        for pname, pval in fit_params.items():
            if "intensity" in pname.lower():
                fit_int_val = float(pval)
                break
        if ref_int_val is not None:
            abs_err = (
                abs(fit_int_val - ref_int_val) if fit_int_val is not None else None
            )
            within = None
            if fit_int_val is not None and ref_int_p95 and len(ref_int_p95) == 2:
                within = ref_int_p95[0] <= fit_int_val <= ref_int_p95[1]
            comp.params.append(
                ParamComparison(
                    layer="probe",
                    param="intensity",
                    ref_value=ref_int_val,
                    ref_p95=ref_int_p95,
                    fit_value=fit_int_val,
                    abs_error=abs_err,
                    within_p95=within,
                    fit_layer="probe",
                )
            )

    # ── Parameter comparisons ───────────────────────────────
    for layer in ref.layers:
        ref_layer_name = layer["name"]
        fit_layer_name = matched.get(ref_layer_name)  # May be None

        for param_key in ("thickness", "interface", "rho"):
            pinfo = layer.get(param_key, {})
            if not isinstance(pinfo, dict):
                continue
            if pinfo.get("fixed"):
                continue

            ref_val = pinfo.get("value")
            if ref_val is None:
                continue

            ref_p95 = pinfo.get("p95")
            fit_val = None
            if fit_layer_name is not None:
                fit_val = _find_fit_param(fit_params, fit_layer_name, param_key)

            abs_err = abs(fit_val - ref_val) if fit_val is not None else None
            within = None
            if fit_val is not None and ref_p95 and len(ref_p95) == 2:
                within = ref_p95[0] <= fit_val <= ref_p95[1]

            comp.params.append(
                ParamComparison(
                    layer=ref_layer_name,
                    param=param_key,
                    ref_value=ref_val,
                    ref_p95=ref_p95,
                    fit_value=fit_val,
                    abs_error=abs_err,
                    within_p95=within,
                    fit_layer=fit_layer_name,
                )
            )

    return comp


def compare_all(
    data_dir: Path = DATA_DIR,
    results_dir: Path = RESULTS_DIR,
) -> List[LayerComparison]:
    """Compare all runs that have results."""
    inventory = build_inventory(data_dir)
    comparisons = []
    for run, ref in inventory.items():
        comp = compare_single(ref, results_dir)
        if comp is not None:
            comparisons.append(comp)
    return comparisons


# ── Aggregate statistics ──────────────────────────────────────────────────────


@dataclass
class AggregateStats:
    n_datasets: int = 0
    n_with_results: int = 0
    n_layer_match: int = 0
    mean_chi2: Optional[float] = None
    median_chi2: Optional[float] = None
    mean_ref_chi2: Optional[float] = None
    mean_chi2_ratio: Optional[float] = None
    mean_frac_within_p95: Optional[float] = None
    common_missing_layers: Dict[str, int] = field(default_factory=dict)
    common_extra_layers: Dict[str, int] = field(default_factory=dict)

    # Per-param-type aggregate errors
    thickness_mae: Optional[float] = None
    interface_mae: Optional[float] = None
    rho_mae: Optional[float] = None


def compute_aggregate(comparisons: List[LayerComparison]) -> AggregateStats:
    """Compute aggregate statistics across all comparisons."""
    stats = AggregateStats(n_with_results=len(comparisons))

    if not comparisons:
        return stats

    # Layer structure
    stats.n_layer_match = sum(1 for c in comparisons if c.layer_count_match)

    for c in comparisons:
        for name in c.missing_layers:
            stats.common_missing_layers[name] = (
                stats.common_missing_layers.get(name, 0) + 1
            )
        for name in c.extra_layers:
            stats.common_extra_layers[name] = stats.common_extra_layers.get(name, 0) + 1

    # Chi-squared
    chi2s = [c.fit_chi2 for c in comparisons if c.fit_chi2 is not None]
    if chi2s:
        stats.mean_chi2 = sum(chi2s) / len(chi2s)
        stats.median_chi2 = sorted(chi2s)[len(chi2s) // 2]

    ref_chi2s = [c.ref_chi2 for c in comparisons if c.ref_chi2 is not None]
    if ref_chi2s:
        stats.mean_ref_chi2 = sum(ref_chi2s) / len(ref_chi2s)

    ratios = [
        c.fit_chi2 / c.ref_chi2
        for c in comparisons
        if c.fit_chi2 is not None and c.ref_chi2 is not None and c.ref_chi2 > 0
    ]
    if ratios:
        stats.mean_chi2_ratio = sum(ratios) / len(ratios)

    # Fraction within p95
    fracs = [c.frac_within_p95 for c in comparisons if c.frac_within_p95 is not None]
    if fracs:
        stats.mean_frac_within_p95 = sum(fracs) / len(fracs)

    # Per parameter type MAE
    for ptype, attr in [
        ("thickness", "thickness_mae"),
        ("interface", "interface_mae"),
        ("rho", "rho_mae"),
    ]:
        errors = []
        for c in comparisons:
            for p in c.params:
                if p.param == ptype and p.abs_error is not None:
                    errors.append(p.abs_error)
        if errors:
            setattr(stats, attr, sum(errors) / len(errors))

    return stats


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_layer_names_from_state(state: dict) -> List[str]:
    """
    Extract layer names from the last_fit parameters in the state summary.

    Fit parameter keys look like "Cu thickness", "THF rho", etc.
    We collect the unique layer name prefixes in order.
    Falls back to parsed_sample if last_fit is unavailable.
    """
    last_fit = state.get("last_fit") or {}
    params = last_fit.get("parameters") or {}

    if params:
        seen = set()
        names = []
        skip_words = {"intensity", "background", "sample_broadening", "theta_offset"}
        for pname in params:
            parts = pname.rsplit(" ", 1)
            if len(parts) == 2:
                layer_part, _ = parts
            else:
                continue
            if layer_part.lower() in skip_words:
                continue
            if layer_part not in seen:
                seen.add(layer_part)
                names.append(layer_part)
        return names

    # Fallback: use parsed_sample
    parsed = state.get("parsed_sample") or {}
    names = []
    ambient = parsed.get("ambient", {})
    if ambient.get("name"):
        names.append(ambient["name"])
    for layer in parsed.get("layers", []):
        if layer.get("name"):
            names.append(layer["name"])
    substrate = parsed.get("substrate", {})
    if substrate.get("name"):
        names.append(substrate["name"])
    return names


def _match_layers_by_position(
    ref_layer_names: List[str],
    fit_layer_names: List[str],
) -> Tuple[Dict[str, str], List[str], List[str]]:
    """
    Match reference layers to fitted layers **by position** in the stack.

    Both lists are ordered ambient → … → substrate (top to bottom).
    The algorithm anchors the ambient (first element) and substrate (last
    element), then matches interior layers top-down.  Any surplus fitted
    interior layers are reported as *extra*; unmatched reference interior
    layers are reported as *missing*.

    Returns
    -------
    matched : dict
        ``{ref_layer_name: fit_layer_name}`` for every matched pair.
    missing : list
        Reference interior layer names that have no fitted counterpart.
    extra : list
        Fitted interior layer names that have no reference counterpart.
    """
    if not ref_layer_names or not fit_layer_names:
        return {}, list(ref_layer_names), list(fit_layer_names)

    matched: Dict[str, str] = {}

    # ── Always match ambient (position 0) ──────────────────────
    matched[ref_layer_names[0]] = fit_layer_names[0]

    # ── Always match substrate (last position) ─────────────────
    if len(ref_layer_names) > 1 and len(fit_layer_names) > 1:
        matched[ref_layer_names[-1]] = fit_layer_names[-1]

    # ── Interior layers: everything between ambient and substrate ──
    ref_interior = ref_layer_names[1:-1] if len(ref_layer_names) > 2 else []
    fit_interior = fit_layer_names[1:-1] if len(fit_layer_names) > 2 else []

    # Match from top (ambient side) downward
    n_match = min(len(ref_interior), len(fit_interior))
    for i in range(n_match):
        matched[ref_interior[i]] = fit_interior[i]

    # Unmatched tails
    missing = ref_interior[n_match:]  # ref layers with no fit counterpart
    extra = fit_interior[n_match:]  # fit layers with no ref counterpart

    return matched, missing, extra


def _find_fit_param(
    fit_params: dict,
    fit_layer_name: str,
    param_key: str,
) -> Optional[float]:
    """
    Search the fit parameters dict for a value matching a layer+param.

    Fit parameter keys are formatted as "<layer> <param>" e.g. "Cu thickness".
    """
    param_aliases = {
        "thickness": ["thickness"],
        "interface": ["interface", "roughness", "sigma"],
        "rho": ["rho", "sld"],
    }
    aliases = param_aliases.get(param_key.lower(), [param_key.lower()])

    for alias in aliases:
        key = f"{fit_layer_name} {alias}"
        if key in fit_params:
            val = fit_params[key]
            if isinstance(val, (int, float)):
                return float(val)

    # Fallback: substring match (handles slight naming variations)
    layer_lower = fit_layer_name.lower()
    for pname, pval in fit_params.items():
        pname_lower = pname.lower()
        if layer_lower in pname_lower:
            for alias in aliases:
                if alias in pname_lower:
                    if isinstance(pval, (int, float)):
                        return float(pval)
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────


def print_comparison(comp: LayerComparison) -> None:
    """Print a single comparison to stdout."""
    header = f"Run {comp.run} ({comp.sample}, exp {comp.experiment})"
    print(f"\n{header}")
    print("-" * len(header))

    # Show layer stacks side by side with positional mapping
    print(f"  Ref layers:  {' → '.join(comp.ref_layer_names)}")
    print(
        f"  Fit layers:  {' → '.join(comp.fit_layer_names) if comp.fit_layer_names else '(none)'}"
    )

    n_ref = len(comp.ref_layer_names)
    n_fit = len(comp.fit_layer_names)
    match_str = "YES" if comp.layer_count_match else "NO"
    print(f"  Layer count: ref={n_ref} fit={n_fit} match={match_str}", end="")
    if comp.extra_layers:
        print(f"  extra={comp.extra_layers}", end="")
    print()

    if comp.fit_chi2 is not None or comp.ref_chi2 is not None:
        parts = []
        if comp.fit_chi2 is not None:
            parts.append(f"fit={comp.fit_chi2:.3f}")
        if comp.ref_chi2 is not None:
            parts.append(f"ref={comp.ref_chi2:.3f}")
        if comp.fit_chi2 is not None and comp.ref_chi2 is not None:
            ratio = comp.fit_chi2 / comp.ref_chi2
            parts.append(f"ratio={ratio:.2f}×")
        print(f"  χ²: {', '.join(parts)}")

    if comp.params:
        print(
            f"  {'Ref layer':<14} {'Fit layer':<16} {'Param':<12} {'Ref':>10} {'Fit':>10} {'Error':>10} {'p95?':>5}"
        )
        for p in comp.params:
            fit_str = f"{p.fit_value:.3f}" if p.fit_value is not None else "   ---"
            err_str = f"{p.abs_error:.3f}" if p.abs_error is not None else "   ---"
            p95_str = (
                " YES"
                if p.within_p95
                else ("  NO" if p.within_p95 is False else " ---")
            )
            fit_name = p.fit_layer if hasattr(p, "fit_layer") and p.fit_layer else ""
            print(
                f"  {p.layer:<14} {fit_name:<16} {p.param:<12} {p.ref_value:>10.3f} {fit_str:>10} {err_str:>10} {p95_str:>5}"
            )

    frac = comp.frac_within_p95
    if frac is not None:
        print(f"  Params within ref 95% CI: {frac:.0%}")


def print_aggregate(stats: AggregateStats) -> None:
    """Print aggregate stats."""
    print(f"\n{'=' * 60}")
    print("  AGGREGATE STATISTICS")
    print(f"{'=' * 60}")
    print(f"  Datasets with results: {stats.n_with_results}")
    print(f"  Layer count match:     {stats.n_layer_match}/{stats.n_with_results}")

    if stats.mean_chi2 is not None:
        print(f"  Mean χ² (fit):         {stats.mean_chi2:.3f}")
        print(f"  Median χ² (fit):       {stats.median_chi2:.3f}")
    if stats.mean_ref_chi2 is not None:
        print(f"  Mean χ² (ref):         {stats.mean_ref_chi2:.3f}")
    if stats.mean_chi2_ratio is not None:
        print(f"  Mean χ² ratio (fit/ref): {stats.mean_chi2_ratio:.2f}×")

    if stats.mean_frac_within_p95 is not None:
        print(f"  Mean frac in ref 95%:  {stats.mean_frac_within_p95:.1%}")

    if stats.thickness_mae is not None:
        print(f"  Thickness MAE (Å):     {stats.thickness_mae:.1f}")
    if stats.interface_mae is not None:
        print(f"  Interface MAE (Å):     {stats.interface_mae:.1f}")
    if stats.rho_mae is not None:
        print(f"  SLD MAE (×10⁻⁶ Å⁻²):  {stats.rho_mae:.3f}")

    if stats.common_missing_layers:
        print(f"  Commonly missing layers: {stats.common_missing_layers}")
    if stats.common_extra_layers:
        print(f"  Commonly extra layers:   {stats.common_extra_layers}")


if __name__ == "__main__":
    comparisons = compare_all()
    if not comparisons:
        print("No results found. Run batch_runner first.")
    else:
        for comp in comparisons:
            print_comparison(comp)
        stats = compute_aggregate(comparisons)
        print_aggregate(stats)
