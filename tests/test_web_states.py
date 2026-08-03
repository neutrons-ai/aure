"""Web/data-layer tests for multi-state co-refinement (Ticket 10)."""

from __future__ import annotations

import threading
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

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


def test_preview_structure_surfaces_per_state_divergence(tmp_path: Path) -> None:
    """When a state carries its own stack (sample != structure), the preview
    reports how it diverges from the template; homogeneous states are omitted."""
    f1, f2 = _write_two_state_files(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    client = _make_ui_client(out)
    body = _ui_body(f1, f2, str(out))

    result = {
        "current_model": {
            "layers": [
                {"name": n, "thickness": 100.0, "material": {"rho": 6.0}}
                for n in ["Cu oxide", "Cu", "Ti"]
            ],
            "substrate": {"name": "Si"},
            "ambient": {"name": "air"},
            "states": [
                {"name": "D2O"},  # inherits the template (no per-state layers)
                {
                    "name": "H2O",
                    "layers": [
                        {"name": "Cu", "thickness": 100.0, "material": {"rho": 6.0}},
                        {"name": "Ti", "thickness": 30.0, "material": {"rho": -1.9}},
                    ],
                },
            ],
        }
    }
    with patch("aure.workflow.runner.run_prepare", return_value=result):
        resp = client.post("/api/preview-structure", json=body)

    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    # Only the divergent state is surfaced.
    assert [s["name"] for s in data["states"]] == ["H2O"]
    h2o = data["states"][0]
    assert h2o["layers"] == ["Cu", "Ti"]
    assert h2o["omits"] == ["Cu oxide"]
    assert h2o["adds"] == []


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


def _write_angle_partial(path: Path, two_theta: float = 0.9, n: int = 80) -> str:
    """A partial file whose header carries one TwoTheta(deg) row so
    `_parse_theta_from_header` recovers theta = two_theta / 2."""
    Q = np.linspace(0.01, 0.12, n)
    R = np.clip((0.0217 / (2 * Q)) ** 4, 1e-9, 1.0)
    lines = ["# index TwoTheta(deg) wavelength", f"# 0 {two_theta} 5.0", "# Q R dR dQ"]
    for q, r in zip(Q, R):
        lines.append(f"{q:.6f}  {r:.6e}  {0.05 * r:.6e}  {0.02 * q:.6f}")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_states_simulation_backfills_theta_for_theta_offset(tmp_path: Path) -> None:
    """Regression: the Results-tab recompute must apply theta_offset even when
    the model's state dataset carries no `theta` (imported / older models). The
    angle is re-derived from the file header so the slider has an effect."""
    from aure.web.data import _compute_states_simulation

    f = _write_angle_partial(tmp_path / "REFL_1_1_1_partial.txt")
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
                "name": "state0",
                # No `theta` on the dataset -> must be re-derived from header.
                "data_files": [{"file": f, "label": "p"}],
                "theta_offset": {"init": 0.0, "min": -0.02, "max": 0.02},
            }
        ],
    }
    r0 = _compute_states_simulation(model, {"state0 theta_offset": 0.0})
    r1 = _compute_states_simulation(model, {"state0 theta_offset": 0.02})
    R0 = np.array(r0["per_file"][0]["R_fit"])
    R1 = np.array(r1["per_file"][0]["R_fit"])
    assert len(R0) and not np.allclose(R0, R1), "theta_offset had no effect"
    # The states response uses `sld_profiles` (the shape the Results SLD User
    # trace now reads), not flat sld_z/sld_rho.
    assert r0.get("sld_profiles")


def test_states_simulation_returns_corrected_data_q(tmp_path: Path) -> None:
    """theta_offset re-assigns each measured point's Q, so the recompute must
    return the experimental data at the corrected Q (Q_data) for the Results
    tab to replot the data markers. R_data (measured) is unchanged."""
    import numpy as np

    from aure.web.data import _compute_states_simulation

    f = _write_angle_partial(tmp_path / "REFL_1_1_1_partial.txt")
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
                "name": "state0",
                "data_files": [{"file": f, "label": "p"}],
                "theta_offset": {"init": 0.0, "min": -0.05, "max": 0.05},
            }
        ],
    }
    pf0 = _compute_states_simulation(model, {"state0 theta_offset": 0.0})["per_file"][0]
    pf1 = _compute_states_simulation(model, {"state0 theta_offset": 0.05})["per_file"][
        0
    ]
    assert "Q_data" in pf0 and "R_data" in pf0
    Qd0, Qd1 = np.array(pf0["Q_data"]), np.array(pf1["Q_data"])
    Rd0, Rd1 = np.array(pf0["R_data"]), np.array(pf1["R_data"])
    # The data's Q shifts with the offset; the measured R is unchanged.
    assert not np.allclose(Qd0, Qd1), "corrected data Q did not move with theta_offset"
    assert np.allclose(Rd0, Rd1), "measured R should not change"


def _veto_state() -> dict:
    """Two iterations: the lower-χ² one was vetoed, so finalize reported the other."""

    def fit(chi2, vetoed):
        return {
            "iteration": 1,
            "method": "amoeba",
            "chi_squared": chi2,
            "converged": True,
            "parameters": {"Cu thickness": 495.0},
            "uncertainties": None,
            "bounds": {"Cu thickness": [200.0, 800.0]},
            "profile_checked": True,
            "profile_artifact": vetoed,
            "issues": ["Non-physical SLD-profile excursion (min SLD -0.88): x"]
            if vetoed
            else [],
        }

    return {
        "fit_results": [fit(0.62, True), fit(1.20, False)],
        "current_chi2": 1.20,
        "final_selection": {
            "selected": True,
            "index": 1,
            "demoted_for_profile_artifact": True,
            "selected_has_profile_artifact": False,
        },
    }


def test_results_payload_flags_the_displayed_iteration_not_just_the_selected_one(
    tmp_path: Path,
) -> None:
    """The Results tab has an iteration dropdown, so a reader can navigate to the
    vetoed fit finalize deliberately set aside. Keying the flag on the selected
    iteration alone would render that one like any other answer."""
    rd = RunData(str(tmp_path))
    with patch.object(rd, "get_final_state", return_value=_veto_state()):
        default = rd.get_fit_parameters()["profile_veto"]
        vetoed = rd.get_fit_parameters(0)["profile_veto"]

    # Default is the clean selection: not itself invalid, but explain the χ².
    assert default["displayed_vetoed"] is False
    assert default["demoted"] is True
    assert default["vetoed_iterations"] == [0]

    # Navigating to the vetoed iteration flags it, with the detector's own words.
    assert vetoed["displayed_vetoed"] is True
    assert "Non-physical SLD-profile excursion" in vetoed["reason"]


def test_history_steps_mark_a_vetoed_fit(tmp_path: Path) -> None:
    """A vetoed fit is often the run's lowest χ², so an unmarked point reads as the
    best result. False on `fitting` and True on `evaluation` is correct — that is
    when the fit is actually judged."""
    rd = RunData(str(tmp_path))
    clean = {"chi_squared": 0.62, "profile_artifact": False, "issues": []}
    judged = {"chi_squared": 0.62, "profile_artifact": True, "issues": []}
    cps = [
        {
            "_info": {"node": "fitting", "iteration": 1},
            "state": {"current_chi2": 0.62, "fit_results": [clean]},
        },
        {
            "_info": {"node": "evaluation", "iteration": 1},
            "state": {"current_chi2": 0.62, "fit_results": [judged]},
        },
    ]
    with patch.object(rd, "_load_all_checkpoints", return_value=cps):
        steps = rd.get_chi2_progression()

    assert [s["vetoed"] for s in steps] == [False, True]


def test_the_setup_form_round_trips_the_chi2_window(tmp_path: Path) -> None:
    """`chi2_max`/`chi2_min` were SetupConfig keys with no field in the form, so a
    setup loaded through the web UI and saved again silently lost them."""
    from aure.web import create_app

    data = tmp_path / "REFL_1_combined_data.txt"
    data.write_text("# Q R dR\n0.01 1.0 0.05\n0.02 0.25 0.01\n")

    app = create_app(None)
    app.config["TESTING"] = True
    client = app.test_client()

    yaml_text = (
        "sample_description: Cu on Si\n"
        "chi2_max: 3.75\n"
        "chi2_min: 0.25\n"
        "states:\n"
        "  - name: state0\n"
        "    data_files:\n"
        f"      - file: {data}\n"
    )

    loaded = client.post("/api/setup/load", json={"yaml": yaml_text}).get_json()
    assert loaded["chi2_max"] == 3.75
    assert loaded["chi2_min"] == 0.25

    exported = client.post(
        "/api/setup/export",
        json={
            "sample_description": "Cu on Si",
            "chi2_max": loaded["chi2_max"],
            "chi2_min": loaded["chi2_min"],
            "states": [{"name": "state0", "data_files": [{"file": str(data)}]}],
        },
    ).get_data(as_text=True)
    assert "chi2_max: 3.75" in exported
    assert "chi2_min: 0.25" in exported


def test_a_seeded_window_beats_the_ambient_environment(monkeypatch) -> None:
    """The web UI runs the analysis on a background thread, so the window is seeded
    into the state rather than exported to the environment — mutating os.environ
    there would race any other request. The runner pins only what is unset, so a
    seeded value is what the run keeps."""
    from aure.workflow import runner

    monkeypatch.setenv("CHI2_MAX", "9.0")
    monkeypatch.setenv("CHI2_MIN", "0.1")

    seen = {}

    def _evaluation(state):
        seen.update(chi2_max=state.get("chi2_max"), chi2_min=state.get("chi2_min"))
        return {
            "workflow_complete": True,
            "fit_results": [{"chi_squared": 1.0, "issues": []}],
        }

    monkeypatch.setitem(runner.NODE_FUNCTIONS, "evaluation", _evaluation)
    monkeypatch.setitem(runner.NODE_FUNCTIONS, "finalize", lambda s: {})
    monkeypatch.setitem(runner.NODE_FUNCTIONS, "final_fit", lambda s: {})

    final = runner.run_workflow_with_checkpoints(
        initial_state={
            "messages": [],
            "iteration": 0,
            "max_iterations": 2,
            "chi2_max": 3.75,
            "chi2_min": 0.25,
        },
        start_node="evaluation",
    )

    assert seen == {"chi2_max": 3.75, "chi2_min": 0.25}
    assert final["chi2_max"] == 3.75  # and it is what the checkpoints record
