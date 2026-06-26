"""Tests for fitting node multi-state co-refinement (Ticket 07)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ----------------------------------------------------------------------
# helpers (same shape as test_model_builder_states.py)
# ----------------------------------------------------------------------


def _make_data_file(q_min: float = 0.01, q_max: float = 0.10, n: int = 60) -> str:
    Q = np.linspace(q_min, q_max, n)
    Qc = 0.0217
    R = np.where(Q < Qc, 1.0, (Qc / (2 * Q)) ** 4)
    R *= 1 + 0.2 * np.cos(2 * Q * 100.0)
    R = np.clip(R, 1e-10, 1.0)
    dR = 0.05 * R
    dQ = 0.02 * Q
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False)
    f.write("# Q R dR dQ\n")
    for q, r, dr, dq in zip(Q, R, dR, dQ):
        f.write(f"{q:.6f}  {r:.6e}  {dr:.6e}  {dq:.6e}\n")
    f.close()
    return f.name


@pytest.fixture
def two_files():
    paths = [_make_data_file(), _make_data_file()]
    yield paths
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


def _two_state_model(files: list[str], **extra) -> dict:
    base = {
        "substrate": {
            "name": "silicon",
            "sld": 2.07,
            "roughness": 3.0,
            "roughness_max": 15.0,
        },
        "layers": [
            {
                "name": "Cu",
                "sld": 6.5,
                "sld_min": 4.0,
                "sld_max": 8.0,
                "thickness": 500.0,
                "thickness_min": 250.0,
                "thickness_max": 750.0,
                "roughness": 5.0,
                "roughness_max": 20.0,
            },
        ],
        "ambient": {"name": "air", "sld": 0.0},
        "back_reflection": False,
        "intensity": {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": True},
        "data_file": files[0],
        "dq_is_fwhm": True,
        "states": [
            {
                "name": "D2O",
                "data_files": [{"file": files[0], "label": "D2O-comb"}],
                "ambient": {"name": "D2O", "sld": 6.4, "sld_min": 5.0, "sld_max": 6.6},
                "intensity": {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": True},
            },
            {
                "name": "H2O",
                "data_files": [{"file": files[1], "label": "H2O-comb"}],
                "ambient": {
                    "name": "H2O",
                    "sld": -0.56,
                    "sld_min": -0.6,
                    "sld_max": 0.0,
                },
                "intensity": {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": True},
            },
        ],
    }
    base.update(extra)
    return base


def _single_model(data_file: str, **extra) -> dict:
    base = {
        "substrate": {
            "name": "silicon",
            "sld": 2.07,
            "roughness": 3.0,
            "roughness_max": 15.0,
        },
        "layers": [
            {
                "name": "Cu",
                "sld": 6.5,
                "sld_min": 4.0,
                "sld_max": 8.0,
                "thickness": 500.0,
                "thickness_min": 250.0,
                "thickness_max": 750.0,
                "roughness": 5.0,
                "roughness_max": 20.0,
            },
        ],
        "ambient": {"name": "air", "sld": 0.0},
        "back_reflection": False,
        "intensity": {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": True},
        "data_file": data_file,
        "dq_is_fwhm": True,
    }
    base.update(extra)
    return base


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_run_states_refl1d_fit_produces_per_file_with_state(two_files, tmp_path):
    """Each dataset gets one PerFileFitResult tagged with its state name."""
    from aure.nodes.fitting import run_states_refl1d_fit

    model = _two_state_model(two_files)
    export_dir = str(tmp_path / "export")
    Path(export_dir).mkdir(parents=True, exist_ok=True)

    # Use 'lm' with 1 step → fast smoke test (we don't care about convergence)
    result = run_states_refl1d_fit(
        model_definition=model,
        method="lm",
        iteration=1,
        steps=1,
        burn=1,
        export_dir=export_dir,
    )

    per_file = result.get("per_file_results")
    assert per_file is not None and len(per_file) == 2
    states_seen = {pf["state"] for pf in per_file}
    assert states_seen == {"D2O", "H2O"}
    for pf in per_file:
        assert pf["chi_squared"] >= 0


def test_per_state_profile_dat_written(two_files, tmp_path):
    """Each state writes profile.dat under state_<name>/."""
    from aure.nodes.fitting import run_states_refl1d_fit

    model = _two_state_model(two_files)
    export_dir = str(tmp_path / "export")
    Path(export_dir).mkdir(parents=True, exist_ok=True)

    run_states_refl1d_fit(
        model_definition=model,
        method="lm",
        iteration=1,
        steps=1,
        burn=1,
        export_dir=export_dir,
    )

    assert (Path(export_dir) / "state_D2O" / "profile.dat").exists()
    assert (Path(export_dir) / "state_H2O" / "profile.dat").exists()
    # Sanity-check: file is 2-column z rho
    text = (Path(export_dir) / "state_D2O" / "profile.dat").read_text().splitlines()
    body = [ln for ln in text if not ln.startswith("#")]
    assert body, "profile.dat must contain data rows"
    parts = body[0].split()
    assert len(parts) == 2


def test_states_fit_dedup_free_param_count(two_files, tmp_path):
    """With default tied set, 2-state fit deduplicates shared Parameter
    objects: free-param count is *less than* what naive 2× counting would
    give."""
    from aure.nodes.fitting import run_states_refl1d_fit

    model = _two_state_model(two_files)
    result = run_states_refl1d_fit(
        model_definition=model,
        method="lm",
        iteration=1,
        steps=1,
        burn=1,
        export_dir=None,
    )
    n_free = result["_n_free_params"]
    # Naively a 1-layer/1-substrate per state would have ~6 params per state
    # → 12 total. With Cu.thickness, Cu.material.rho, Cu.interface,
    # substrate.interface tied, we expect strictly fewer than 2× single-state.
    assert n_free < 12
    assert (
        n_free >= 4
    )  # minimum: at least Cu.thickness, Cu.rho, Cu.interface, sub.interface


def test_fitting_node_dispatches_to_states_branch(two_files, tmp_path, monkeypatch):
    """fitting_node detects multi-state via model['states'] and calls
    run_states_refl1d_fit (not run_multi_refl1d_fit)."""
    from aure.nodes import fitting

    called = {"states": False, "multi": False, "single": False}

    def fake_states(*a, **kw):
        called["states"] = True
        return {
            "iteration": kw.get("iteration", 0),
            "method": kw.get("method", "lm"),
            "chi_squared": 1.0,
            "converged": True,
            "parameters": {},
            "uncertainties": None,
            "bounds": None,
            "Q_fit": [],
            "R_fit": [],
            "residuals": [],
            "residual_ratio": [],
            "sld_z": None,
            "sld_rho": None,
            "per_file_results": [],
            "issues": [],
            "suggestions": [],
            "_n_free_params": 5,
        }

    def fake_multi(*a, **kw):
        called["multi"] = True
        raise AssertionError("multi branch should not be used")

    def fake_single(*a, **kw):
        called["single"] = True
        raise AssertionError("single branch should not be used")

    monkeypatch.setattr(fitting, "run_states_refl1d_fit", fake_states)
    monkeypatch.setattr(fitting, "run_multi_refl1d_fit", fake_multi)
    monkeypatch.setattr(fitting, "run_refl1d_fit", fake_single)

    model = _two_state_model(two_files)
    state = {
        "current_model": model,
        "iteration": 1,
        "data_files": [{"file": two_files[0]}, {"file": two_files[1]}],
        "Q": [],
        "output_dir": str(tmp_path),
        "messages": [],
    }
    out = fitting.fitting_node(state)
    assert called["states"] is True
    assert called["multi"] is False
    assert called["single"] is False
    assert "error" not in out


# ----------------------------------------------------------------------
# Model-name resolution (regression: refl1d output written as None-*)
# ----------------------------------------------------------------------


def test_resolve_model_name_priority_chain(tmp_path):
    from aure.nodes import fitting

    # 1. explicit user_config model_name wins over everything else
    st = {
        "user_config": {"model_name": "cu_air_230536"},
        "model_name": "ignored",
        "output_dir": str(tmp_path / "230536"),
    }
    assert fitting._resolve_model_name(st, {"name": "ignored"}) == "cu_air_230536"

    # 2. state-level model_name
    assert fitting._resolve_model_name({"model_name": "foo"}, {}) == "foo"

    # 3. the model definition's own name fields
    assert fitting._resolve_model_name({}, {"model_name": "bar"}) == "bar"
    assert fitting._resolve_model_name({}, {"name": "baz"}) == "baz"

    # 4. THE REGRESSION: user_config is None -> fall back to output-dir basename
    st = {"user_config": None, "output_dir": str(tmp_path / "230536")}
    assert fitting._resolve_model_name(st, {}) == "230536"

    # 5. primary data-file stem
    st = {"data_files": [{"file": "/data/REFL_111_1_111_partial.txt"}]}
    assert fitting._resolve_model_name(st, {}) == "REFL_111_1_111_partial"

    # 6. last-resort literal — but never None / "None" / ""
    assert fitting._resolve_model_name({}, {}) == "model"
    assert (
        fitting._resolve_model_name({"user_config": {"model_name": "None"}}, {})
        == "model"
    )

    # candidate names are sanitised into safe filename stems
    assert (
        fitting._resolve_model_name({"model_name": "my model/v2"}, {}) == "my_model_v2"
    )


def test_fitting_node_never_uses_none_basename(tmp_path, monkeypatch):
    """Regression: a run with no user_config must still name its refl1d output
    after the run folder, not None (which produced None-*.dat / None.json)."""
    from aure.nodes import fitting

    captured = {}

    def fake_single(*a, **kw):
        captured["model_name"] = kw.get("model_name")
        return {
            "success": True,
            "chi_squared": 1.0,
            "method": "dream",
            "parameters": {},
            "uncertainties": {},
            "converged": True,
            "per_file_results": [],
            "issues": [],
            "suggestions": [],
            "_n_free_params": 3,
        }

    def _boom(branch):
        def _f(*a, **kw):
            raise AssertionError(f"{branch} branch should not run")

        return _f

    monkeypatch.setattr(fitting, "run_refl1d_fit", fake_single)
    monkeypatch.setattr(fitting, "run_multi_refl1d_fit", _boom("multi"))
    monkeypatch.setattr(fitting, "run_states_refl1d_fit", _boom("states"))

    data_file = _make_data_file()
    run_dir = tmp_path / "230536"
    run_dir.mkdir()
    state = {
        "current_model": {"substrate": {"material": "Si"}, "layers": [], "ambient": {}},
        "user_config": None,  # the regression condition
        "iteration": 0,
        "data_files": [{"file": data_file}],
        "Q": [0.01, 0.02, 0.03],
        "output_dir": str(run_dir),
        "messages": [],
    }
    out = fitting.fitting_node(state)
    assert "error" not in out, out.get("error")
    assert captured["model_name"] == "230536"
    assert captured["model_name"] not in (None, "None", "")
    # The resolved name is persisted onto the state for checkpoints / run_info.
    assert out["model_name"] == "230536"


# ----------------------------------------------------------------------
# Background parameter (per-state tied, fittable)
# ----------------------------------------------------------------------


def test_single_experiment_background_is_fittable():
    from aure.nodes.model_builder import build_problem

    model = _single_model(
        _make_data_file(), background={"init": 2e-6, "min": 0.0, "max": 1e-5}
    )
    problem = build_problem(model)
    free = [str(p.name) for p in problem._parameters]
    assert any("background" in n for n in free), free


def test_single_experiment_no_background_when_absent():
    from aure.nodes.model_builder import build_problem

    problem = build_problem(_single_model(_make_data_file()))
    free = [str(p.name) for p in problem._parameters]
    assert not any("background" in n for n in free), free


def test_multi_problem_background_tied_across_segments():
    from aure.nodes.model_builder import build_multi_problem

    files = [
        {"file": _make_data_file(), "label": "lo"},
        {"file": _make_data_file(), "label": "hi"},
    ]
    model = _single_model(
        files[0]["file"], background={"enabled": True, "min": 0.0, "max": 1e-5}
    )
    problem, _exps, _sorted = build_multi_problem(model, files)
    free = [str(p.name) for p in problem._parameters]
    # Tied -> exactly one shared background parameter across both segments.
    assert free.count("background") == 1, free


def test_states_background_tied_per_state_and_named():
    from aure.nodes.model_builder import build_states_problem

    d2o = [_make_data_file(), _make_data_file()]
    h2o = [_make_data_file(), _make_data_file()]
    model = {
        "substrate": {
            "name": "silicon",
            "sld": 2.07,
            "roughness": 3.0,
            "roughness_max": 15.0,
        },
        "layers": [
            {
                "name": "Cu",
                "sld": 6.5,
                "sld_min": 4.0,
                "sld_max": 8.0,
                "thickness": 500.0,
                "thickness_min": 250.0,
                "thickness_max": 750.0,
                "roughness": 5.0,
                "roughness_max": 20.0,
            },
        ],
        "ambient": {"name": "air", "sld": 0.0},
        "states": [
            {
                "name": "D2O",
                "data_files": [
                    {"file": f, "label": f"d2o{i}"} for i, f in enumerate(d2o)
                ],
                "background": {"init": 2e-6, "min": 0.0, "max": 1e-5},
            },
            {
                "name": "H2O",
                "data_files": [
                    {"file": f, "label": f"h2o{i}"} for i, f in enumerate(h2o)
                ],
                "background": {"init": 3e-6, "min": 0.0, "max": 1e-5},
            },
        ],
    }
    problem, _by_state, _sorted = build_states_problem(model)
    free = [str(p.name) for p in problem._parameters]
    bg = sorted(n for n in free if "background" in n)
    # One tied background per state (not one per file), correctly named.
    assert bg == ["D2O background", "H2O background"], free


def test_states_no_spurious_background_when_absent():
    from aure.nodes.model_builder import build_states_problem

    files = [_make_data_file(), _make_data_file()]
    problem, _by_state, _sorted = build_states_problem(_two_state_model(files))
    free = [str(p.name) for p in problem._parameters]
    assert not any("background" in n for n in free), free


def test_needs_states_problem_triggers_on_single_state_background():
    from aure.nodes.model_builder import needs_states_problem

    model = {
        "states": [
            {
                "name": "s0",
                "data_files": [{"file": "x"}],
                "background": {"min": 0.0, "max": 1e-5},
            }
        ]
    }
    assert needs_states_problem(model) is True


def test_states_background_surfaces_in_fit_results():
    """End-to-end: a per-state background is a free parameter that shows up in
    the extracted fit results under '<state> background'."""
    from aure.nodes.fitting import run_states_refl1d_fit

    model = _two_state_model([_make_data_file(), _make_data_file()])
    for st in model["states"]:
        st["background"] = {"init": 2e-6, "min": 0.0, "max": 1e-5}

    result = run_states_refl1d_fit(
        model_definition=model,
        method="lm",
        iteration=0,
        steps=1,
        burn=1,
        export_dir=None,
    )
    params = result["parameters"]
    assert "D2O background" in params, list(params)
    assert "H2O background" in params, list(params)


def test_states_all_three_nuisance_wire_for_single_state():
    """A single state can fit background + theta_offset + sample_broadening
    together (each tied across the state's angle-based probes). Regression:
    the initial model must include *all* user-specified nuisance, not one."""
    from aure.nodes.model_builder import build_states_problem

    f1, f2 = _make_data_file(), _make_data_file()
    model = {
        "substrate": {
            "name": "silicon",
            "sld": 2.07,
            "roughness": 3.0,
            "roughness_max": 15.0,
        },
        "layers": [
            {
                "name": "Cu",
                "sld": 6.5,
                "sld_min": 4.0,
                "sld_max": 8.0,
                "thickness": 500.0,
                "thickness_min": 250.0,
                "thickness_max": 750.0,
                "roughness": 5.0,
                "roughness_max": 20.0,
            },
        ],
        "ambient": {"name": "air", "sld": 0.0},
        "states": [
            {
                "name": "s0",
                # theta on each dataset -> angle-based NeutronProbe, which is
                # what theta_offset / sample_broadening require.
                "data_files": [
                    {"file": f1, "label": "a", "theta": 0.5},
                    {"file": f2, "label": "b", "theta": 0.7},
                ],
                "background": {"init": 2e-6, "min": 0.0, "max": 1e-5},
                "theta_offset": {"init": 0.0, "min": -0.02, "max": 0.02},
                "sample_broadening": {"init": 0.0, "min": 0.0, "max": 0.05},
            }
        ],
    }
    problem, _by_state, _sorted = build_states_problem(model)
    free = [str(p.name) for p in problem._parameters]
    assert "s0 background" in free, free
    assert "s0 theta_offset" in free, free
    assert "s0 sample_broadening" in free, free
    # Each is tied across the two probes -> exactly one of each.
    assert sum("background" in n for n in free) == 1
    assert sum("theta_offset" in n for n in free) == 1
    assert sum("sample_broadening" in n for n in free) == 1
