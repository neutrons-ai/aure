"""The validation harness must score the fit the run reported.

`finalize` chooses the reported answer — it rejects regressions, prefers the
simpler model inside the χ² band, and sets profile-vetoed fits aside — so the last
iteration *fitted* is routinely one it discarded. The harness read `fit_results[-1]`
for its per-run score and, worse, took χ² from `best_chi2` while taking parameters
from `last_fit`, comparing a reference against two different iterations at once.
That corrupts the measurement used to judge whether a change to AuRE helped.
"""

from __future__ import annotations

import json
from pathlib import Path

from validation.batch_runner import _save_state_summary, _selected_fit
from validation.comparator import compare_single
from validation.inventory import ReferenceModel


def _fit(chi2: float, thickness: float, *, vetoed: bool = False) -> dict:
    return {
        "chi_squared": chi2,
        "parameters": {"Cu thickness": thickness},
        "uncertainties": None,
        "profile_artifact": vetoed,
        "issues": [],
    }


def _state(*, vetoed: bool = False) -> dict:
    """A run where finalize reported iteration 0, not the last one fitted."""
    return {
        "fit_results": [_fit(1.20, 495.0, vetoed=vetoed), _fit(4.70, 120.0)],
        "current_chi2": 1.20,
        "best_chi2": 1.20,
        "final_selection": {"selected": True, "index": 0, "iteration": 1},
    }


def _reference(tmp_path: Path) -> ReferenceModel:
    return ReferenceModel(
        run="R1",
        sample="cu",
        experiment=1,
        data_file=tmp_path / "d.txt",
        model_file=tmp_path / "m.json",
        layers=[
            {
                "name": "Cu",
                "thickness": {"value": 500.0, "p95": [480.0, 520.0]},
            }
        ],
        probe={},
        chisq=1.0,
    )


def test_the_harness_scores_the_reported_fit(tmp_path):
    """χ² and parameters must come from the same iteration — the one reported."""
    assert _selected_fit(_state())["chi_squared"] == 1.20  # not the last fit's 4.70

    run_dir = tmp_path / "R1"
    run_dir.mkdir()
    _save_state_summary(_state(), run_dir / "state_summary.json")

    comp = compare_single(_reference(tmp_path), tmp_path)

    assert comp is not None
    assert comp.fit_chi2 == 1.20
    # The old path scored best_chi2 (1.20) against last_fit's 120.0 thickness.
    thickness = next(p for p in comp.params if p.param == "thickness")
    assert thickness.fit_value == 495.0
    assert thickness.within_p95 is True


def test_a_vetoed_model_is_flagged_not_scored_silently(tmp_path):
    """The excursion buys χ², so a rejected model can look like the best run in the
    set. The comparison has to say so."""
    run_dir = tmp_path / "R1"
    run_dir.mkdir()
    _save_state_summary(_state(vetoed=True), run_dir / "state_summary.json")

    comp = compare_single(_reference(tmp_path), tmp_path)

    assert comp.fit_profile_artifact is True


def test_summaries_written_before_selected_fit_still_compare(tmp_path):
    """Old state_summary.json files on disk carry only best_chi2 / last_fit."""
    run_dir = tmp_path / "R1"
    run_dir.mkdir()
    (run_dir / "state_summary.json").write_text(
        json.dumps(
            {
                "best_chi2": 2.5,
                "last_fit": {
                    "chi_squared": 2.5,
                    "parameters": {"Cu thickness": 505.0},
                },
            }
        )
    )

    comp = compare_single(_reference(tmp_path), tmp_path)

    assert comp.fit_chi2 == 2.5
    assert comp.fit_profile_artifact is False
    thickness = next(p for p in comp.params if p.param == "thickness")
    assert thickness.fit_value == 505.0
