"""
Validation report generator.

Produces a markdown report comparing aure outputs to reference models.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List

from .comparator import (
    LayerComparison,
    AggregateStats,
    compare_all,
    compute_aggregate,
)
from .inventory import DATA_DIR


RESULTS_DIR = Path(__file__).parent / "results"


def generate_report(
    comparisons: List[LayerComparison],
    stats: AggregateStats,
    output_path: Path,
) -> None:
    """Write a markdown validation report."""
    lines: list[str] = []
    w = lines.append

    w(f"# Validation Report")
    w(f"")
    w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"")

    # ── Summary table ────────────────────────────────────────
    w("## Summary")
    w("")
    w(f"| Metric | Value |")
    w(f"|--------|-------|")
    w(f"| Datasets with results | {stats.n_with_results} |")
    w(f"| Layer count match | {stats.n_layer_match}/{stats.n_with_results} |")
    if stats.mean_chi2 is not None:
        w(f"| Mean χ² | {stats.mean_chi2:.3f} |")
        w(f"| Median χ² | {stats.median_chi2:.3f} |")
    if stats.mean_frac_within_p95 is not None:
        w(f"| Mean params within ref 95% CI | {stats.mean_frac_within_p95:.1%} |")
    if stats.thickness_mae is not None:
        w(f"| Thickness MAE (Å) | {stats.thickness_mae:.1f} |")
    if stats.interface_mae is not None:
        w(f"| Interface MAE (Å) | {stats.interface_mae:.1f} |")
    if stats.rho_mae is not None:
        w(f"| SLD MAE (×10⁻⁶ Å⁻²) | {stats.rho_mae:.3f} |")
    w("")

    # ── Systematic issues ────────────────────────────────────
    if stats.common_missing_layers or stats.common_extra_layers:
        w("## Systematic Issues")
        w("")
        if stats.common_missing_layers:
            w("**Commonly missing layers:**")
            w("")
            for name, count in sorted(
                stats.common_missing_layers.items(), key=lambda x: -x[1]
            ):
                w(f"- `{name}`: missing in {count}/{stats.n_with_results} datasets")
            w("")
        if stats.common_extra_layers:
            w("**Commonly extra layers:**")
            w("")
            for name, count in sorted(
                stats.common_extra_layers.items(), key=lambda x: -x[1]
            ):
                w(f"- `{name}`: extra in {count}/{stats.n_with_results} datasets")
            w("")

    # ── Overview table ────────────────────────────────────────
    w("## Per-Dataset Overview")
    w("")
    w("| Run | Sample | Exp | χ² | Layers OK | % in 95% CI |")
    w("|-----|--------|-----|-----|-----------|-------------|")
    for c in comparisons:
        chi_str = f"{c.fit_chi2:.2f}" if c.fit_chi2 is not None else "—"
        layers_ok = "✓" if c.layer_count_match else "✗"
        frac = c.frac_within_p95
        frac_str = f"{frac:.0%}" if frac is not None else "—"
        w(
            f"| {c.run} | {c.sample} | {c.experiment} | {chi_str} | {layers_ok} | {frac_str} |"
        )
    w("")

    # ── Per-dataset details ──────────────────────────────────
    w("## Per-Dataset Details")
    w("")
    for c in comparisons:
        w(f"### Run {c.run} — {c.sample} (experiment {c.experiment})")
        w("")
        w(f"- **Ref layers:** {' → '.join(c.ref_layer_names)}")
        w(
            f"- **Fit layers:** {' → '.join(c.fit_layer_names) if c.fit_layer_names else '(none)'}"
        )
        if c.missing_layers:
            w(f"- **Missing:** {', '.join(c.missing_layers)}")
        if c.extra_layers:
            w(f"- **Extra:** {', '.join(c.extra_layers)}")
        if c.fit_chi2 is not None:
            w(f"- **χ²:** {c.fit_chi2:.3f}")
        w("")

        if c.params:
            w("| Layer | Param | Reference | Fitted | Error | In 95% CI |")
            w("|-------|-------|-----------|--------|-------|-----------|")
            for p in c.params:
                fit_str = f"{p.fit_value:.3f}" if p.fit_value is not None else "—"
                err_str = f"{p.abs_error:.3f}" if p.abs_error is not None else "—"
                p95_str = (
                    "✓" if p.within_p95 else ("✗" if p.within_p95 is False else "—")
                )
                w(
                    f"| {p.layer} | {p.param} | {p.ref_value:.3f} | {fit_str} | {err_str} | {p95_str} |"
                )
            w("")

    # ── Write ─────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Report written to {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    comparisons = compare_all()
    if not comparisons:
        print("No results found. Run batch_runner first.")
    else:
        stats = compute_aggregate(comparisons)
        report_path = RESULTS_DIR / "validation_report.md"
        generate_report(comparisons, stats, report_path)
