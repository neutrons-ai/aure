"""Tests for ``aure import-refl1d``.

The pipeline under test:

    build_states_problem(def)  →  save_problem_json  →  bumps.load_file
                                                          ↓
                                                  import_refl1d
                                                          ↓
                                              <AuRE output directory>

Each test builds a synthetic refl1d ``problem.json`` from a known
ModelDefinition, runs the importer against it, and asserts that:

- the on-disk layout matches what the web UI / ``aure resume`` expect
- the recovered ``ModelDefinition`` round-trips cleanly through
  ``build_states_problem`` (chi² stable, parameter names align)
- multi-state structure recovery preserves cross-state aliasing
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ----------------------------------------------------------------------
# Fixtures: synthetic data files + model definitions
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
def one_file():
    p = _make_data_file()
    yield p
    try:
        os.unlink(p)
    except OSError:
        pass


@pytest.fixture
def two_files():
    paths = [_make_data_file(), _make_data_file()]
    yield paths
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


def _single_state_definition(file_path: str) -> dict:
    return {
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
        "intensity": {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": False},
        "data_file": file_path,
    }


def _two_state_definition(files: list[str]) -> dict:
    return {
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
            {
                "name": "Ti",
                "sld": -1.95,
                "sld_min": -3.0,
                "sld_max": 0.0,
                "thickness": 30.0,
                "thickness_min": 10.0,
                "thickness_max": 60.0,
                "roughness": 3.0,
                "roughness_max": 15.0,
            },
        ],
        "ambient": {"name": "air", "sld": 0.0},
        "back_reflection": False,
        "intensity": {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": False},
        "states": [
            {
                "name": "D2O",
                "data_files": [{"file": files[0], "label": "D2O-comb"}],
                "ambient": {"name": "D2O", "sld": 6.4, "sld_min": 5.0, "sld_max": 6.6},
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
            },
        ],
    }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _save_problem_to(tmp_path: Path, definition: dict) -> Path:
    """Serialise *definition* to a ``problem.json`` inside a fit_iter*_* dir."""
    from aure.nodes.model_builder import save_problem_json

    fit_dir = tmp_path / "refl1d_output" / "fit_iter0_lm"
    fit_dir.mkdir(parents=True)
    out = fit_dir / "problem.json"
    save_problem_json(definition, out)
    return fit_dir


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_single_state_round_trip_writes_expected_layout(tmp_path, one_file):
    from aure.refl1d_import import import_refl1d

    src = _save_problem_to(tmp_path, _single_state_definition(one_file))
    out = tmp_path / "imported"

    summary = import_refl1d(str(src), str(out))

    # Layout
    assert (out / "run_info.json").is_file()
    assert (out / "final_state.json").is_file()
    assert (out / "problem.json").is_file()
    assert (out / "data").is_dir()
    # Checkpoint naming mirrors a real run: fitting/evaluation get an
    # ``_iter{n}`` suffix when iteration > 0 (evaluation increments).
    cp_dir = out / "checkpoints"
    assert sorted(p.name for p in cp_dir.glob("*.json")) == [
        "001_intake.json",
        "002_analysis.json",
        "003_modeling.json",
        "004_fitting.json",
        "005_evaluation_iter1.json",
    ]
    assert (out / "refl1d_output" / "fit_iter0_lm" / "problem.json").is_file()

    # Summary
    assert summary["states"] == ["state0"]
    assert summary["n_files"] == 1
    assert summary["method"] == "lm"
    assert summary["chi_squared"] >= 0


def test_single_state_final_state_carries_recovered_model(tmp_path, one_file):
    from aure.refl1d_import import import_refl1d

    src = _save_problem_to(tmp_path, _single_state_definition(one_file))
    out = tmp_path / "imported"
    import_refl1d(str(src), str(out))

    final = json.loads((out / "final_state.json").read_text())
    state = final["state"]

    assert state["workflow_complete"] is True
    assert state["iteration"] == 1
    assert state["current_chi2"] is not None
    assert state["best_chi2"] == state["current_chi2"]

    model = state["current_model"]
    assert model["substrate"]["name"] == "silicon"
    assert len(model["layers"]) == 1
    assert model["layers"][0]["name"] == "Cu"
    # Always emit states explicitly, even for single-state.
    assert isinstance(model["states"], list)
    assert len(model["states"]) == 1
    assert model["states"][0]["name"] == "state0"


def test_multi_state_recovers_state_grouping(tmp_path, two_files):
    from aure.refl1d_import import import_refl1d

    src = _save_problem_to(tmp_path, _two_state_definition(two_files))
    out = tmp_path / "imported"
    summary = import_refl1d(
        str(src), str(out), state_names=["D2O", "H2O"]
    )

    assert summary["states"] == ["D2O", "H2O"]
    assert summary["n_files"] == 2

    final = json.loads((out / "final_state.json").read_text())
    state = final["state"]
    states = state["current_model"]["states"]
    assert [s["name"] for s in states] == ["D2O", "H2O"]
    # Per-state ambient should differ
    amb_names = {s["ambient"]["name"] for s in states}
    assert amb_names == {"D2O", "H2O"}


def test_multi_state_recovered_definition_rebuilds_a_problem(
    tmp_path, two_files
):
    """The recovered ModelDefinition must round-trip back into a valid
    multi-state FitProblem with the same χ² (parameter ties preserved).
    """
    from aure.nodes.model_builder import build_states_problem
    from aure.refl1d_import import import_refl1d

    src = _save_problem_to(tmp_path, _two_state_definition(two_files))
    out = tmp_path / "imported"
    import_refl1d(str(src), str(out), state_names=["D2O", "H2O"])

    final = json.loads((out / "final_state.json").read_text())
    model = final["state"]["current_model"]

    # Rebuild from the imported model — this exercises that the recovered
    # shared_parameters/unshared_parameters keep the problem well-formed.
    problem, by_state, _ = build_states_problem(model)
    assert set(by_state.keys()) == {"D2O", "H2O"}
    chi2 = float(problem.chisq())
    # χ² should match the chi² captured at import time (no fit happens here).
    assert chi2 == pytest.approx(final["state"]["current_chi2"], rel=1e-6)


def test_force_overwrites_existing_output(tmp_path, one_file):
    from aure.refl1d_import import import_refl1d

    src = _save_problem_to(tmp_path, _single_state_definition(one_file))
    out = tmp_path / "imported"
    out.mkdir()
    (out / "stale.txt").write_text("leftover")

    with pytest.raises(FileExistsError):
        import_refl1d(str(src), str(out))

    import_refl1d(str(src), str(out), force=True)
    assert not (out / "stale.txt").exists()
    assert (out / "final_state.json").exists()


def test_rejects_output_inside_source_dir(tmp_path, one_file):
    """Writing the AuRE workspace inside the refl1d source would make the
    refl1d-tree copy step recurse into itself. We must refuse with a
    clear message.
    """
    from aure.refl1d_import import import_refl1d

    src = _save_problem_to(tmp_path, _single_state_definition(one_file))
    out = src / "nested_import"  # inside the source dir

    with pytest.raises(ValueError, match="inside the source refl1d"):
        import_refl1d(str(src), str(out))


def test_resolves_parent_refl1d_output_dir(tmp_path, one_file):
    """Pointing at the parent ``refl1d_output/`` dir should pick the
    latest ``fit_iter*_*`` automatically (same heuristic as ``aure
    evaluate``)."""
    from aure.refl1d_import import import_refl1d

    src = _save_problem_to(tmp_path, _single_state_definition(one_file))
    parent = src.parent  # the refl1d_output/ dir
    summary = import_refl1d(str(parent), str(tmp_path / "imported"))
    assert summary["method"] == "lm"


def test_back_reflection_override_applies(tmp_path, one_file):
    from aure.refl1d_import import import_refl1d

    src = _save_problem_to(tmp_path, _single_state_definition(one_file))
    summary = import_refl1d(
        str(src), str(tmp_path / "imported"), back_reflection=True
    )
    assert summary["back_reflection"] is True


def test_synthesized_description_mentions_back_reflection(tmp_path, one_file):
    """Back-reflection runs must surface the entry side in the description
    — the LLM uses this on resume/refine to keep the substrate side straight.
    """
    from aure.refl1d_import import import_refl1d

    # Build the problem in back-reflection mode so the orientation is
    # genuine. The heuristic recognises ``silicon`` as a substrate and
    # picks the orientation without needing the explicit override.
    defn = _single_state_definition(one_file)
    defn["back_reflection"] = True
    src = _save_problem_to(tmp_path, defn)
    out = tmp_path / "imported"
    summary = import_refl1d(str(src), str(out))

    assert summary["back_reflection"] is True
    final = json.loads((out / "final_state.json").read_text())
    desc = final["state"]["sample_description"]
    assert "Neutrons enter from the silicon substrate side" in desc


def test_synthesized_description_orders_layers_top_down(tmp_path, two_files):
    """Layers in ParsedSample run bottom-up (substrate-adjacent first);
    the synthesized description should reverse them so the English
    reading is top-down (\"Cu on Ti on Si\")."""
    from aure.refl1d_import import import_refl1d

    src = _save_problem_to(tmp_path, _two_state_definition(two_files))
    out = tmp_path / "imported"
    import_refl1d(str(src), str(out), state_names=["D2O", "H2O"])

    final = json.loads((out / "final_state.json").read_text())
    desc = final["state"]["sample_description"]
    # Fixture layers are [Cu, Ti] (Cu adjacent to substrate, Ti adjacent
    # to ambient). Top-down reading is "Ti on Cu on silicon".
    cu_idx = desc.find("Cu")
    ti_idx = desc.find("Ti")
    assert ti_idx >= 0 and cu_idx >= 0
    assert ti_idx < cu_idx, (
        f"layers should read top-down but description says {desc!r}"
    )


def test_per_file_results_carry_state_name(tmp_path, two_files):
    """``PerFileFitResult.state`` is required by the multi-state web view."""
    from aure.refl1d_import import import_refl1d

    src = _save_problem_to(tmp_path, _two_state_definition(two_files))
    out = tmp_path / "imported"
    import_refl1d(str(src), str(out), state_names=["D2O", "H2O"])

    final = json.loads((out / "final_state.json").read_text())
    pf_results = final["state"]["fit_results"][0]["per_file_results"]
    assert pf_results is not None
    assert {pf["state"] for pf in pf_results} == {"D2O", "H2O"}


def test_multi_state_summary_lists_tied_and_untied(tmp_path, two_files):
    """The summary dict should split the default tied set into tied/untied,
    so the CLI can show them at a glance. With the default tied set
    fully active, every default pair is tied.
    """
    from aure.refl1d_import import import_refl1d

    src = _save_problem_to(tmp_path, _two_state_definition(two_files))
    out = tmp_path / "imported"
    summary = import_refl1d(str(src), str(out), state_names=["D2O", "H2O"])

    assert summary["untied_parameters"] == []
    tied = summary["tied_parameters"]
    # Default tied set covers thickness, material.rho, interface per layer
    # plus substrate.interface.
    assert "Cu.thickness" in tied
    assert "Cu.material.rho" in tied
    assert "Cu.interface" in tied
    assert "Ti.thickness" in tied
    assert "silicon.interface" in tied
    # And no warnings on a clean AuRE-shape import.
    assert summary["warnings"] == []


def test_multi_state_warns_when_no_ties_recovered(tmp_path, two_files):
    """If a multi-sample problem comes in with zero shared parameter
    objects (e.g. constraint-expression ties we can't recover), the
    summary should include a warning so the user investigates.
    """
    from aure.refl1d_import import import_refl1d

    defn = _two_state_definition(two_files)
    # Untie every default-tied pair so the imported problem has zero
    # shared parameter objects across the two samples.
    defn["unshared_parameters"] = [
        "Cu.thickness",
        "Cu.material.rho",
        "Cu.interface",
        "Ti.thickness",
        "Ti.material.rho",
        "Ti.interface",
        "silicon.interface",
    ]
    src = _save_problem_to(tmp_path, defn)
    out = tmp_path / "imported"
    summary = import_refl1d(str(src), str(out), state_names=["D2O", "H2O"])

    assert summary["tied_parameters"] == []
    assert summary["untied_parameters"]  # everything got recovered as untied
    assert summary["warnings"]
    assert any(
        "no shared parameter objects" in w for w in summary["warnings"]
    )


def test_single_state_summary_has_no_tie_metadata(tmp_path, one_file):
    """Single-state imports should report empty tie lists and no warnings."""
    from aure.refl1d_import import import_refl1d

    src = _save_problem_to(tmp_path, _single_state_definition(one_file))
    summary = import_refl1d(str(src), str(tmp_path / "imported"))

    assert summary["tied_parameters"] == []
    assert summary["untied_parameters"] == []
    assert summary["warnings"] == []


def test_data_files_are_self_contained(tmp_path, one_file):
    """After import, deleting the source data file should not break the
    imported run — probes have been dumped into the output's ``data/`` dir.
    """
    from aure.refl1d_import import import_refl1d
    from aure.nodes.model_builder import build_problem

    src = _save_problem_to(tmp_path, _single_state_definition(one_file))
    out = tmp_path / "imported"
    import_refl1d(str(src), str(out))

    # Kill the original source data file
    os.unlink(one_file)

    final = json.loads((out / "final_state.json").read_text())
    model = final["state"]["current_model"]
    # Importer flattens states into the legacy data_file when populating
    # the model dict, and writes copies under <out>/data/
    file_in_model = model["data_file"]
    assert Path(file_in_model).exists()
    # And build_problem should still work against it.
    problem = build_problem(model)
    assert problem.chisq() >= 0
