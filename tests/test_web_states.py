"""Web/data-layer tests for multi-state co-refinement (Ticket 10)."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

from aure.web.data import RunData


def _make_multi_state_final_state(data_file: str) -> dict:
    """Minimal final-state with two states and per-file fit results tagged."""
    return {
        "data_file": data_file,
        "Q": [0.01, 0.02, 0.03],
        "R": [1.0, 0.5, 0.1],
        "dR": [0.01, 0.01, 0.01],
        "data_files": [
            {
                "label": "REFL_1_combined",
                "file": data_file,
                "Q": [0.01, 0.02, 0.03],
                "R": [1.0, 0.5, 0.1],
                "dR": [0.01, 0.01, 0.01],
            },
            {
                "label": "REFL_2_combined",
                "file": data_file,
                "Q": [0.01, 0.02, 0.03],
                "R": [0.9, 0.4, 0.08],
                "dR": [0.01, 0.01, 0.01],
            },
        ],
        "states": [
            {
                "name": "D2O",
                "data_files": [{"label": "REFL_1_combined", "file": data_file}],
            },
            {
                "name": "H2O",
                "data_files": [{"label": "REFL_2_combined", "file": data_file}],
            },
        ],
        "fit_results": [
            {
                "iteration": 0,
                "method": "lm",
                "chi_squared": 1.5,
                "converged": True,
                "parameters": {},
                "uncertainties": {},
                "per_file_results": [
                    {
                        "label": "REFL_1_combined",
                        "state": "D2O",
                        "Q_fit": [0.01, 0.02, 0.03],
                        "R_fit": [1.0, 0.5, 0.1],
                        "chi_squared": 1.4,
                    },
                    {
                        "label": "REFL_2_combined",
                        "state": "H2O",
                        "Q_fit": [0.01, 0.02, 0.03],
                        "R_fit": [0.9, 0.4, 0.08],
                        "chi_squared": 1.6,
                    },
                ],
            }
        ],
    }


def test_reflectivity_curves_carry_state_field(tmp_path: Path) -> None:
    rd = RunData(str(tmp_path))
    state = _make_multi_state_final_state("/tmp/x.dat")
    with patch.object(rd, "get_final_state", return_value=state):
        out = rd.get_reflectivity_data()

    by_label = {m["file_label"]: m for m in out["models"]}
    assert by_label["REFL_1_combined"]["state"] == "D2O"
    assert by_label["REFL_2_combined"]["state"] == "H2O"

    df_by_label = {df["label"]: df for df in out["data_files"]}
    assert df_by_label["REFL_1_combined"]["state"] == "D2O"
    assert df_by_label["REFL_2_combined"]["state"] == "H2O"


def test_sld_profiles_dispatch_to_states(tmp_path: Path) -> None:
    rd = RunData(str(tmp_path))
    state = _make_multi_state_final_state("/tmp/x.dat")
    state["current_model"] = {
        "substrate": {"name": "Si"},
        "ambient": {"name": "air"},
        "layers": [],
        "states": state["states"],
    }
    with (
        patch.object(rd, "get_final_state", return_value=state),
        patch.object(
            rd, "_get_model_for_iteration", return_value=state["current_model"]
        ),
        patch(
            "aure.web.data._compute_states_sld",
            return_value=[
                {"state": "D2O", "z": [0.0, 1.0], "sld": [2.07, 6.5]},
                {"state": "H2O", "z": [0.0, 1.0], "sld": [-0.56, 6.5]},
            ],
        ) as mock_compute,
    ):
        out = rd.get_sld_profiles()

    assert mock_compute.called
    states_seen = {p["state"] for p in out["profiles"]}
    assert states_seen == {"D2O", "H2O"}
    assert all("iteration" in p for p in out["profiles"])


# ======================================================================
# UI-shape submit contract for /api/start-analysis (Ticket 15)
# ======================================================================


def _make_ui_client(tmp_output_dir):
    """Build a Flask test client with the web blueprint, idle run-state."""
    from flask import Flask

    from aure.web.routes import bp

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["OUTPUT_DIR"] = str(tmp_output_dir)
    app.config["RUN_LOCK"] = threading.Lock()
    app.config["RUN_STATE"] = {"status": "idle"}
    app.register_blueprint(bp)
    return app.test_client()


def _write_two_state_files(tmp_path: Path) -> tuple[str, str]:
    f1 = tmp_path / "REFL_1_combined_data.txt"
    f2 = tmp_path / "REFL_2_combined_data.txt"
    rows = "0.01 1.0 0.01\n0.02 0.5 0.01\n0.03 0.1 0.01\n"
    f1.write_text(rows)
    f2.write_text(rows)
    return str(f1), str(f2)


def _ui_body(f1: str, f2: str, output_dir: str, **extra) -> dict:
    """Mirror exactly what setup.js `_buildAnalysisBody` produces."""
    body = {
        "data_file": f1,
        "sample_description": "Cu film",
        "hypothesis": None,
        "output_dir": output_dir,
        "interactive": False,
        "max_iterations": 1,
        "states": [
            {
                "name": "D2O",
                "data_files": [{"file": f1, "label": "REFL_1_combined_data"}],
            },
            {
                "name": "H2O",
                "data_files": [{"file": f2, "label": "REFL_2_combined_data"}],
            },
        ],
    }
    body.update(extra)
    return body


def test_ui_shape_submit_auto_mode(tmp_path: Path) -> None:
    f1, f2 = _write_two_state_files(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    client = _make_ui_client(out)
    body = _ui_body(f1, f2, str(out))

    with patch("aure.web.routes.threading.Thread") as mock_thread:
        mock_thread.return_value.start.return_value = None
        resp = client.post("/api/start-analysis", json=body)

    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["status"] == "started"
    assert mock_thread.called


def test_ui_shape_submit_shared_parameters(tmp_path: Path) -> None:
    f1, f2 = _write_two_state_files(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    client = _make_ui_client(out)
    body = _ui_body(
        f1,
        f2,
        str(out),
        shared_parameters=["Cu.thickness", "Cu.material.rho", "substrate.interface"],
    )

    with patch("aure.web.routes.threading.Thread") as mock_thread:
        mock_thread.return_value.start.return_value = None
        resp = client.post("/api/start-analysis", json=body)

    assert resp.status_code == 200, resp.get_json()


def test_ui_shape_submit_rejects_both_shared_and_unshared(tmp_path: Path) -> None:
    f1, f2 = _write_two_state_files(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    client = _make_ui_client(out)
    body = _ui_body(
        f1,
        f2,
        str(out),
        shared_parameters=["Cu.thickness"],
        unshared_parameters=["CuOx.thickness"],
    )

    resp = client.post("/api/start-analysis", json=body)
    assert resp.status_code == 400
    errors = resp.get_json()["errors"]
    assert any("mutually exclusive" in e for e in errors)


def test_ui_shape_submit_rejects_missing_file(tmp_path: Path) -> None:
    f1, _ = _write_two_state_files(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    client = _make_ui_client(out)
    body = _ui_body(f1, str(tmp_path / "does-not-exist.txt"), str(out))

    resp = client.post("/api/start-analysis", json=body)
    assert resp.status_code == 400
    errors = resp.get_json()["errors"]
    assert any("does not exist" in e for e in errors)


def test_setup_page_prev_run_carries_states(tmp_path: Path) -> None:
    """Ticket 16: setup page exposes `states` from the previous run."""
    import json

    output_dir = tmp_path / "out"
    (output_dir / "checkpoints").mkdir(parents=True)
    final = {
        "state": {
            "data_file": "/tmp/x.dat",
            "sample_description": "Cu film",
            "states": [
                {
                    "name": "D2O",
                    "data_files": [{"label": "REFL_1", "file": "/tmp/x.dat"}],
                    "ambient": {"rho": 6.36},
                },
                {
                    "name": "H2O",
                    "data_files": [{"label": "REFL_2", "file": "/tmp/y.dat"}],
                    "ambient": {"rho": -0.56},
                },
            ],
            "current_model": {
                "shared_parameters": ["Cu.thickness", "Cu.material.rho"],
            },
        },
        "completed_at": "2026-05-17T00:00:00",
    }
    (output_dir / "final_state.json").write_text(json.dumps(final))
    (output_dir / "run_info.json").write_text(
        json.dumps(
            {
                "data_file": "/tmp/x.dat",
                "sample_description": "Cu film",
                "data_files": [
                    {"file": "/tmp/x.dat", "label": "REFL_1"},
                    {"file": "/tmp/y.dat", "label": "REFL_2"},
                ],
            }
        )
    )

    client = _make_ui_client(output_dir)
    resp = client.get("/setup")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "_prevRun" in html
    assert '"states"' in html
    assert "D2O" in html and "H2O" in html
    assert "Cu.thickness" in html


# ======================================================================
# /api/preview-structure (Ticket 17)
# ======================================================================


def _fake_prepare_result(layer_names):
    return {
        "current_model": {
            "layers": [
                {"name": n, "thickness": 100.0, "material": {"rho": 6.0}}
                for n in layer_names
            ],
            "substrate": {"name": "Si"},
            "ambient": {"name": "air"},
        }
    }


def test_preview_structure_happy_path(tmp_path: Path) -> None:
    f1, f2 = _write_two_state_files(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    client = _make_ui_client(out)
    body = _ui_body(f1, f2, str(out))

    with patch(
        "aure.workflow.runner.run_prepare",
        return_value=_fake_prepare_result(["Cu", "SiO2"]),
    ):
        resp = client.post("/api/preview-structure", json=body)

    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    assert [l["name"] for l in data["layers"]] == ["Cu", "SiO2"]
    assert "Cu.thickness" in data["parameters"]
    assert "Cu.material.rho" in data["parameters"]
    assert "SiO2.interface" in data["parameters"]
    assert "substrate.interface" in data["parameters"]


def test_preview_structure_rejects_missing_file(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    client = _make_ui_client(out)
    body = {
        "sample_description": "Cu film",
        "states": [
            {
                "name": "D2O",
                "data_files": [{"file": str(tmp_path / "nope.txt"), "label": "x"}],
            }
        ],
    }

    resp = client.post("/api/preview-structure", json=body)
    assert resp.status_code == 400
    assert any("does not exist" in e for e in resp.get_json()["errors"])


def test_preview_structure_rejects_both_shared_and_unshared(tmp_path: Path) -> None:
    f1, f2 = _write_two_state_files(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    client = _make_ui_client(out)
    body = _ui_body(
        f1,
        f2,
        str(out),
        shared_parameters=["Cu.thickness"],
        unshared_parameters=["Cu.material.rho"],
    )

    resp = client.post("/api/preview-structure", json=body)
    assert resp.status_code == 400


# ======================================================================
# /api/known-shared-params (Ticket 18)
# ======================================================================


def test_known_shared_params_aggregates_past_runs(tmp_path: Path) -> None:
    import json as _json

    root = tmp_path / "runs"
    root.mkdir()
    active = root / "active"
    (active / "checkpoints").mkdir(parents=True)

    # Past run A
    rA = root / "20260101_A"
    (rA / "checkpoints").mkdir(parents=True)
    (rA / "final_state.json").write_text(
        _json.dumps(
            {
                "state": {
                    "current_model": {
                        "shared_parameters": ["Cu.thickness", "Cu.material.rho"],
                    }
                }
            }
        )
    )

    # Past run B
    rB = root / "20260202_B"
    (rB / "checkpoints").mkdir(parents=True)
    (rB / "final_state.json").write_text(
        _json.dumps(
            {
                "state": {
                    "current_model": {
                        "unshared_parameters": ["SiO2.interface", "Cu.thickness"],
                    }
                }
            }
        )
    )

    client = _make_ui_client(active)
    resp = client.get("/api/known-shared-params")
    assert resp.status_code == 200
    params = resp.get_json()["parameters"]
    assert params == sorted(["Cu.thickness", "Cu.material.rho", "SiO2.interface"])


def test_known_shared_params_empty_when_no_history(tmp_path: Path) -> None:
    active = tmp_path / "lone"
    (active / "checkpoints").mkdir(parents=True)
    client = _make_ui_client(active)
    resp = client.get("/api/known-shared-params")
    assert resp.status_code == 200
    assert resp.get_json()["parameters"] == []
