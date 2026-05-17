"""Web/data-layer tests for multi-state co-refinement (Ticket 10)."""

from __future__ import annotations

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
