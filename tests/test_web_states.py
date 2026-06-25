"""Web/data-layer tests for multi-state co-refinement (Ticket 10)."""

from __future__ import annotations

import threading
import json
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
    output_dir = tmp_path / "out"
    (output_dir / "checkpoints").mkdir(parents=True)
    final = {
        "state": {
            "data_file": "/tmp/x.dat",
            "sample_description": "Cu film",
            "states": [
                {
                    "name": "D2O",
                    "data_files": [
                        {
                            "label": "REFL_1",
                            "file": "/tmp/x.dat",
                            "Q": [0.01, 0.02],
                            "R": [1.0, 0.8],
                            "dR": [0.01, 0.01],
                        }
                    ],
                    "ambient": {"rho": 6.36},
                },
                {
                    "name": "H2O",
                    "data_files": [
                        {
                            "label": "REFL_2",
                            "file": "/tmp/y.dat",
                            "Q": [0.01, 0.02],
                            "R": [0.9, 0.7],
                            "dR": [0.01, 0.01],
                        }
                    ],
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
    payload = html.split("var _prevRun = ", 1)[1].split(";\n</script>", 1)[0]
    prev = json.loads(payload)
    for st in prev["states"]:
        for df in st["data_files"]:
            assert set(df.keys()) == {"file", "label"}


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


# ---------------------------------------------------------------------------
# Save Setup (/api/setup/export)
# ---------------------------------------------------------------------------


def _write_partial_files(tmp_path: Path) -> list[str]:
    """Three partial files sharing one set_id (one physical state)."""
    rows = "0.01 1.0 0.01\n0.02 0.5 0.01\n0.03 0.1 0.01\n"
    names = [
        "REFL_230536_1_230536_partial.txt",
        "REFL_230536_2_230537_partial.txt",
        "REFL_230536_3_230538_partial.txt",
    ]
    paths = []
    for n in names:
        p = tmp_path / n
        p.write_text(rows)
        paths.append(str(p))
    return paths


def test_setup_export_single_state_with_overrides_round_trips(tmp_path: Path) -> None:
    """A single named state (3 partial files + per-state nuisance params) must
    export to a valid states-only YAML that load_setup can read back.

    Regression: clicking "Save Setup" after loading a single-state setup
    previously failed with "at least one state must be declared under
    `states:`" because the form sent flat data_files, not `states`.
    """
    import yaml

    from aure.setup import load_setup

    files = _write_partial_files(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    client = _make_ui_client(out)

    # Mirror what setup.js _buildAnalysisBody now emits for a single named
    # state loaded from job_230536.yaml.
    # Mirror the real form body exactly, including the runtime-only keys the
    # endpoint must translate/strip (output_dir, interactive, max_iterations).
    body = {
        "data_file": files[0],  # legacy positional still sent by the form
        "sample_description": "air / CuOx / 50 nm Cu / 3 nm Ti on Si (back reflection)",
        "hypothesis": None,
        "output_dir": str(out),
        "interactive": False,
        "max_iterations": 7,
        "states": [
            {
                "name": "run_230536",
                "data_files": [{"file": f, "label": Path(f).stem} for f in files],
                "theta_offset": {"init": 0.0, "min": -0.02, "max": 0.02},
                "sample_broadening": {"init": 0.0, "min": 0.0, "max": 0.05},
                "back_reflection": True,
                "extra_description": "Measured in air with the cell empty.",
            }
        ],
    }

    resp = client.post("/api/setup/export", json=body)
    assert resp.status_code == 200, resp.get_json()
    yaml_text = resp.get_data(as_text=True)

    # Parse the exported YAML and confirm the state + overrides survived, the
    # runtime keys were stripped, and max_iterations mapped to max_refinements.
    dumped = yaml.safe_load(yaml_text)
    assert [s["name"] for s in dumped["states"]] == ["run_230536"]
    assert dumped.get("max_refinements") == 7
    assert "output_dir" not in dumped and "interactive" not in dumped
    assert "max_iterations" not in dumped
    st = dumped["states"][0]
    assert st["back_reflection"] is True
    assert st["theta_offset"] == {"init": 0.0, "min": -0.02, "max": 0.02}
    assert "Measured in air" in st["extra_description"]

    # And it must load back through the canonical loader without error.
    setup_path = out / "exported.yaml"
    setup_path.write_text(yaml_text)
    reloaded = load_setup(setup_path)
    assert [s["name"] for s in reloaded["states"]] == ["run_230536"]
    assert len(reloaded["states"][0]["data_files"]) == 3


def test_setup_export_synthesizes_state0_from_flat_data_files(tmp_path: Path) -> None:
    """An ad-hoc save with only flat data_files (no `states`) should succeed by
    synthesizing a single `state0` rather than 400-ing."""
    import yaml

    f1, f2 = _write_two_state_files(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    client = _make_ui_client(out)

    body = {
        "data_file": f1,
        "sample_description": "Cu film, two Q segments",
        "data_files": [
            {"file": f1, "label": "REFL_1_combined_data"},
            {"file": f2, "label": "REFL_2_combined_data"},
        ],
    }
    resp = client.post("/api/setup/export", json=body)
    assert resp.status_code == 200, resp.get_json()
    dumped = yaml.safe_load(resp.get_data(as_text=True))
    assert [s["name"] for s in dumped["states"]] == ["state0"]
    assert len(dumped["states"][0]["data_files"]) == 2


def test_setup_export_synthesizes_state0_from_single_data_file(tmp_path: Path) -> None:
    """The bare single-file ad-hoc case (only `data_file`) also exports."""
    import yaml

    f1, _ = _write_two_state_files(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    client = _make_ui_client(out)

    body = {"data_file": f1, "sample_description": "single file"}
    resp = client.post("/api/setup/export", json=body)
    assert resp.status_code == 200, resp.get_json()
    dumped = yaml.safe_load(resp.get_data(as_text=True))
    assert [s["name"] for s in dumped["states"]] == ["state0"]
    assert dumped["states"][0]["data_files"][0]["file"] == f1


def test_setup_export_still_400s_with_no_files_at_all(tmp_path: Path) -> None:
    """No states and no data files -> the original validation error stands."""
    out = tmp_path / "out"
    out.mkdir()
    client = _make_ui_client(out)

    resp = client.post(
        "/api/setup/export", json={"sample_description": "nothing to export"}
    )
    assert resp.status_code == 400
    assert "states" in resp.get_json()["errors"][0]


def test_setup_export_single_combined_state_with_ambient_and_back_reflection(
    tmp_path: Path,
) -> None:
    """The single-state overrides editor lets an ungrouped (combined-file) state
    carry ambient + back_reflection without faking a second state. The body it
    emits must export and round-trip."""
    import yaml

    f1, _ = _write_two_state_files(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    client = _make_ui_client(out)

    body = {
        "data_file": f1,
        "sample_description": "single combined state in D2O, back reflection",
        "output_dir": str(out),
        "interactive": False,
        "max_iterations": 3,
        "states": [
            {
                "name": "state0",
                "data_files": [{"file": f1, "label": "REFL_1_combined_data"}],
                "ambient": {"rho": 6.4},
                "back_reflection": True,
                "extra_description": "measured in D2O",
            }
        ],
    }
    resp = client.post("/api/setup/export", json=body)
    assert resp.status_code == 200, resp.get_json()

    reloaded = load_setup_text(out, resp.get_data(as_text=True))
    st = reloaded["states"][0]
    assert st["name"] == "state0"
    assert st["back_reflection"] is True
    assert st["ambient"] == {"rho": 6.4}
    assert reloaded.get("max_refinements") == 3
    # sanity: it really is the loader's output, parseable as YAML too
    assert yaml.safe_load(resp.get_data(as_text=True))["states"][0]["ambient"]


def test_setup_export_rejects_theta_offset_on_combined_state(tmp_path: Path) -> None:
    """theta_offset / sample_broadening are partials-only. The UI now hides
    them for combined-file states; the server must still reject them if sent,
    which is the contract that gating mirrors."""
    f1, _ = _write_two_state_files(tmp_path)  # *_combined_data.txt
    out = tmp_path / "out"
    out.mkdir()
    client = _make_ui_client(out)

    body = {
        "data_file": f1,
        "sample_description": "combined file, invalid theta_offset",
        "states": [
            {
                "name": "state0",
                "data_files": [{"file": f1, "label": "REFL_1_combined_data"}],
                "theta_offset": {"init": 0.0, "min": -0.02, "max": 0.02},
            }
        ],
    }
    resp = client.post("/api/setup/export", json=body)
    assert resp.status_code == 400
    assert "partials" in resp.get_json()["errors"][0].lower()


def load_setup_text(out_dir: Path, yaml_text: str):
    """Write exported YAML to disk and load it back through the canonical loader."""
    from aure.setup import load_setup

    p = out_dir / "roundtrip.yaml"
    p.write_text(yaml_text)
    return load_setup(p)
