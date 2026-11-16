"""
Tests for the `aure prepare` command and `run_prepare` helper.

Covers:
- run_prepare runs only intake → analysis → modeling
- save_problem_json produces a loadable JSON file
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from aure.workflow import run_prepare
from aure.workflow.runner import run_workflow_with_checkpoints
from aure.state import create_initial_state
from aure.nodes.model_builder import save_problem_json


# A minimal, deterministic ModelDefinition used to stub the modeling node.
_STUB_MODEL = {
    "substrate": {"name": "silicon", "sld": 2.07, "roughness": 3.0},
    "layers": [
        {
            "name": "film",
            "sld": 6.4,
            "thickness": 100.0,
            "roughness": 5.0,
        }
    ],
    "ambient": {"name": "air", "sld": 0.0},
    "data_file": "",
    "back_reflection": False,
}


def _stub_intake(state):
    import numpy as np

    return {
        "current_node": "intake",
        "Q": np.linspace(0.01, 0.25, 50).tolist(),
        "R": np.ones(50).tolist(),
        "dR": (np.ones(50) * 0.01).tolist(),
        "parsed_sample": {
            "substrate": _STUB_MODEL["substrate"],
            "layers": _STUB_MODEL["layers"],
            "ambient": _STUB_MODEL["ambient"],
        },
        "messages": [],
    }


def _stub_analysis(state):
    return {
        "current_node": "analysis",
        "extracted_features": {
            "estimated_n_layers": 1,
            "estimated_total_thickness": 100.0,
            "estimated_roughness": 5.0,
        },
        "messages": [],
    }


def _stub_modeling(state):
    model = dict(_STUB_MODEL)
    model["data_file"] = state.get("data_file", "")
    return {
        "current_node": "modeling",
        "current_model": model,
        "model_history": [model],
        "messages": [],
    }


def _write_tiny_data_file() -> str:
    """Write a trivial 3-column reflectivity file."""
    import numpy as np

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write("# Q R dR\n")
        for q in np.linspace(0.01, 0.25, 50):
            f.write(f"{q:.6f}  1.0e-3  1.0e-4\n")
        return f.name


def test_run_prepare_stops_after_modeling(tmp_path):
    """run_prepare should execute intake → analysis → modeling and stop."""
    data_file = _write_tiny_data_file()

    stub_funcs = {
        "intake": _stub_intake,
        "analysis": _stub_analysis,
        "modeling": _stub_modeling,
    }
    with patch.dict("aure.workflow.runner.NODE_FUNCTIONS", stub_funcs, clear=False):
        result = run_prepare(
            data_file=data_file,
            sample_description="100 Å film on silicon",
            output_dir=str(tmp_path),
        )

    assert result.get("error") is None
    assert result.get("current_model") is not None
    # No fitting/evaluation should have run
    assert not result.get("fit_results")
    assert result.get("current_chi2") is None
    # Checkpoints for the three nodes
    cp_files = sorted((tmp_path / "checkpoints").glob("*.json"))
    cp_nodes = [cp.stem.split("_", 1)[1] for cp in cp_files]
    assert cp_nodes == ["intake", "analysis", "modeling"]


def test_run_prepare_passes_data_files(tmp_path):
    """run_prepare should accept and preserve data_files for co-refinement."""
    data_file = _write_tiny_data_file()
    extra_file = _write_tiny_data_file()
    datasets = [
        {"file": data_file, "label": "primary"},
        {"file": extra_file, "label": "extra"},
    ]

    stub_funcs = {
        "intake": _stub_intake,
        "analysis": _stub_analysis,
        "modeling": _stub_modeling,
    }
    with patch.dict("aure.workflow.runner.NODE_FUNCTIONS", stub_funcs, clear=False):
        result = run_prepare(
            data_file=data_file,
            sample_description="100 A film on silicon",
            output_dir=str(tmp_path),
            data_files=datasets,
        )

    assert result.get("error") is None
    assert len(result.get("data_files", [])) == 2
    assert result["data_files"][0]["label"] == "primary"
    assert result["data_files"][1]["label"] == "extra"


def test_stop_after_parameter_halts_runner(tmp_path):
    """run_workflow_with_checkpoints(stop_after=...) halts at the named node."""
    data_file = _write_tiny_data_file()

    stub_funcs = {
        "intake": _stub_intake,
        "analysis": _stub_analysis,
        "modeling": _stub_modeling,
    }
    with patch.dict("aure.workflow.runner.NODE_FUNCTIONS", stub_funcs, clear=False):
        state = create_initial_state(
            data_file=data_file,
            sample_description="film",
            max_iterations=0,
        )
        result = run_workflow_with_checkpoints(
            initial_state=state,
            output_dir=str(tmp_path),
            stop_after="analysis",
        )

    assert result.get("current_model") is None  # modeling never ran
    assert result.get("extracted_features") is not None
    assert result.get("workflow_complete") is True


def test_save_problem_json_produces_loadable_file(tmp_path):
    """save_problem_json should write a JSON file that round-trips through bumps."""
    pytest.importorskip("bumps")
    pytest.importorskip("refl1d")

    data_file = _write_tiny_data_file()
    model = dict(_STUB_MODEL)
    model["data_file"] = data_file

    out = tmp_path / "problem.json"
    returned = save_problem_json(model, out)

    assert Path(returned).exists()
    data = json.loads(out.read_text())
    # Serialized bumps problems are JSON objects with some structure
    assert isinstance(data, dict)


def test_prepare_from_a_single_state_setup_honours_its_declared_layers(tmp_path):
    """A setup with one state and a declared stack, run as `aure prepare -c`
    runs it: the first modeling checkpoint and the exported problem must carry
    the declared pin, not the stack parsed from the description.

    Intake and analysis are stubbed (they need the LLM); modeling is real.
    """
    from bumps.serialize import load_file

    from aure.setup import load_setup

    data_file = Path(_write_tiny_data_file())
    setup_path = tmp_path / "setup.yaml"
    setup_path.write_text(
        "sample_description: 100 A film on silicon\n"
        "states:\n"
        "  - name: OCV\n"
        f"    data_files: [{data_file}]\n"
        "    layers:\n"
        "      - {name: copper, sld: 6.48, sld_min: 6.48, sld_max: 6.48,\n"
        "         thickness: 480, thickness_min: 400, thickness_max: 600,\n"
        "         roughness: 5}\n"
    )
    setup = load_setup(setup_path)

    stub_funcs = {"intake": _stub_intake, "analysis": _stub_analysis}
    with patch.dict("aure.workflow.runner.NODE_FUNCTIONS", stub_funcs, clear=False):
        result = run_prepare(
            data_file=str(data_file),
            sample_description=setup["sample_description"],
            output_dir=str(tmp_path / "out"),
            states=setup["states"],
        )
    assert result.get("error") is None, result.get("error")

    (checkpoint,) = (tmp_path / "out" / "checkpoints").glob("*_modeling.json")
    model = json.loads(checkpoint.read_text())["state"]["current_model"]
    (copper,) = model["layers"]
    assert (copper["name"], copper["sld"]) == ("copper", 6.48)
    assert (copper["sld_min"], copper["sld_max"]) == (6.48, 6.48)

    out = tmp_path / "problem.json"
    save_problem_json(result["current_model"], out)
    problem = load_file(str(out))
    bounds = {p.name: tuple(p.bounds) for p in problem._parameters}
    # The exported problem has no free parameter for the pinned SLD.
    assert "copper rho" not in bounds
    assert next(iter(problem.models)).sample[1].material.rho.value == 6.48
    assert bounds["copper thickness"] == (400.0, 600.0)


def test_prepare_from_a_single_state_setup_honours_its_measurement_overrides(tmp_path):
    """`ambient`, `intensity` and `back_reflection` on a lone state, with no
    nuisance key to drag it onto the states route: the exported problem must
    have the declared solvent, the back-reflected stack and no free intensity."""
    from bumps.serialize import load_file

    from aure.setup import load_setup

    data_file = Path(_write_tiny_data_file())
    setup_path = tmp_path / "setup.yaml"
    setup_path.write_text(
        "sample_description: 100 A film on silicon\n"
        "states:\n"
        "  - name: OCV\n"
        f"    data_files: [{data_file}]\n"
        "    ambient: {name: THF, rho: 6.4}\n"
        "    intensity: false\n"
        "    back_reflection: true\n"
    )
    setup = load_setup(setup_path)

    stub_funcs = {"intake": _stub_intake, "analysis": _stub_analysis}
    with patch.dict("aure.workflow.runner.NODE_FUNCTIONS", stub_funcs, clear=False):
        result = run_prepare(
            data_file=str(data_file),
            sample_description=setup["sample_description"],
            output_dir=str(tmp_path / "out"),
            states=setup["states"],
        )
    assert result.get("error") is None, result.get("error")

    model = result["current_model"]
    assert model["ambient"]["sld"] == 6.4
    assert model["back_reflection"] is True
    assert model["intensity"]["fixed"] is True

    out = tmp_path / "problem.json"
    save_problem_json(model, out)
    params = {p.name: p for p in load_file(str(out))._parameters}
    assert params["THF rho"].value == pytest.approx(6.4)
    # Back reflection puts the ambient at the bottom of the stack, where its
    # interface is a fitted parameter; a normal stack has no such parameter.
    assert "THF interface" in params
    assert not any("intensity" in name for name in params)
