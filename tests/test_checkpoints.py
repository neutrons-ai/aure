"""Tests for CheckpointManager's best-fit problem.json resolution.

Regression guard: bumps names the exported FitProblem JSON after the problem
(``<model_name>.json``, or ``None.json`` when the problem is unnamed), not
literally ``problem.json``. The checkpoint copy must find it either way and
write ``<output_dir>/problem.json``.
"""

from __future__ import annotations

import json

from aure.workflow.checkpoints import CheckpointManager


def _fit_dir(mgr, iteration=0, method="dream"):
    d = mgr.refl1d_output_dir / f"fit_iter{iteration}_{method}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write(path, payload="{}"):
    path.write_text(payload)
    return path


def test_find_problem_json_prefers_model_named_file(tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    d = _fit_dir(mgr)
    _write(d / "sample5_ocv_226642.json", '{"big": "problem"}')
    _write(d / "sample5_ocv_226642-1-expt.json")  # sidecar, must be ignored

    state = {"user_config": {"model_name": "sample5_ocv_226642"}}
    found = CheckpointManager._find_problem_json(d, state)
    assert found is not None and found.name == "sample5_ocv_226642.json"


def test_find_problem_json_handles_none_named_export(tmp_path):
    """The original failure: model was unnamed, so bumps wrote None.json."""
    mgr = CheckpointManager(str(tmp_path))
    d = _fit_dir(mgr)
    big = _write(d / "None.json", json.dumps({"models": list(range(100))}))
    _write(d / "None-1-expt.json")
    _write(d / "None-2-expt.json")

    found = CheckpointManager._find_problem_json(d, {})
    assert found == big  # the non-expt JSON, regardless of stem


def test_find_problem_json_ignores_definition_sidecar(tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    d = _fit_dir(mgr)
    _write(d / "m_definition.json", '{"def": 1}')
    real = _write(d / "m.json", json.dumps({"models": list(range(50))}))
    found = CheckpointManager._find_problem_json(d, {"user_config": {"model_name": "m"}})
    assert found == real


def test_find_problem_json_missing_dir_returns_none(tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    missing = mgr.refl1d_output_dir / "fit_iter9_dream"
    assert CheckpointManager._find_problem_json(missing, {}) is None


def test_copy_best_problem_json_writes_problem_json(tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    d = _fit_dir(mgr, iteration=0, method="dream")
    _write(d / "None.json", json.dumps({"models": [1, 2, 3]}))

    state = {
        "fit_results": [{"chi_squared": 1.85, "iteration": 0, "method": "dream"}],
        "best_chi2": 1.85,
    }
    mgr._copy_best_problem_json(state)
    assert (mgr.output_dir / "problem.json").is_file()
