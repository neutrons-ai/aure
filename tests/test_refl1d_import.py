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

    # Layout — lean by design: no refl1d_output/ duplicate, SLD profile
    # and theory curves are inlined in final_state.json.
    assert (out / "run_info.json").is_file()
    assert (out / "final_state.json").is_file()
    assert (out / "problem.json").is_file()
    assert (out / "data").is_dir()
    assert not (out / "refl1d_output").exists()
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
    summary = import_refl1d(str(src), str(out), state_names=["D2O", "H2O"])

    assert summary["states"] == ["D2O", "H2O"]
    assert summary["n_files"] == 2

    final = json.loads((out / "final_state.json").read_text())
    state = final["state"]
    states = state["current_model"]["states"]
    assert [s["name"] for s in states] == ["D2O", "H2O"]
    # Per-state ambient should differ
    amb_names = {s["ambient"]["name"] for s in states}
    assert amb_names == {"D2O", "H2O"}


def _hetero_state_definition(files: list[str]) -> dict:
    """Two states sharing a Cu/Ti base, but the D2O state additionally has a
    'Cu oxide' surface layer that H2O lacks (sample != structure). H2O carries
    its own oxide-less ``layers``; D2O inherits the template (with the oxide)."""
    defn = _two_state_definition(files)
    oxide = {
        "name": "Cu oxide",
        "sld": 5.0,
        "sld_min": 3.0,
        "sld_max": 6.5,
        "thickness": 25.0,
        "thickness_min": 5.0,
        "thickness_max": 60.0,
        "roughness": 4.0,
        "roughness_max": 15.0,
    }
    base_layers = defn["layers"]  # [Cu, Ti]
    defn["layers"] = [oxide, *base_layers]  # template = [Cu oxide, Cu, Ti]
    defn["states"][1]["layers"] = [
        dict(layer) for layer in base_layers
    ]  # H2O: no oxide
    return defn


def test_multi_state_heterogeneous_structure_round_trips(tmp_path, two_files):
    """A co-refinement where one state lacks a layer must round-trip: the
    recovered model keeps that state's own oxide-less stack, and rebuilding
    yields heterogeneous samples (oxide present in the D2O state only)."""
    from aure.nodes.model_builder import build_states_problem
    from aure.refl1d_import import import_refl1d

    src = _save_problem_to(tmp_path, _hetero_state_definition(two_files))
    out = tmp_path / "imported"
    import_refl1d(str(src), str(out), state_names=["D2O", "H2O"])

    final = json.loads((out / "final_state.json").read_text())
    model = final["state"]["current_model"]
    by_name = {s["name"]: s for s in model["states"]}

    # Template (top-level) keeps the oxide; H2O carries its own oxide-less stack
    # while D2O inherits the template (no per-state override needed).
    assert any(layer["name"] == "Cu oxide" for layer in model["layers"])
    h2o_layers = by_name["H2O"].get("layers")
    assert h2o_layers is not None
    assert [layer["name"] for layer in h2o_layers] == ["Cu", "Ti"]
    assert not by_name["D2O"].get("layers")

    # Rebuild: heterogeneous samples — oxide present only in the D2O state.
    _problem, by_state, _ = build_states_problem(model)
    d2o_names = [str(s.material.name) for s in by_state["D2O"][0].sample]
    h2o_names = [str(s.material.name) for s in by_state["H2O"][0].sample]
    assert "Cu oxide" in d2o_names
    assert "Cu oxide" not in h2o_names


def test_multi_state_recovered_definition_rebuilds_a_problem(tmp_path, two_files):
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
    summary = import_refl1d(str(src), str(tmp_path / "imported"), back_reflection=True)
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


def test_multi_file_single_state_collapses_to_one_state(tmp_path, two_files):
    """A Q-segment co-refinement (single Sample shared across N probes)
    survives bumps round-trip with N distinct Sample objects. The
    importer should detect identical ambient + substrate + full default
    tying and collapse to ONE state with N runs — not invent N fake states.
    """
    from aure.nodes.model_builder import save_problem_json
    from aure.refl1d_import import import_refl1d

    # Build a single-state, two-file problem via the legacy
    # build_multi_problem path (one shared Sample).
    defn = _single_state_definition(two_files[0])
    data_files = [
        {"file": two_files[0], "label": "low-Q"},
        {"file": two_files[1], "label": "high-Q"},
    ]
    fit_dir = tmp_path / "refl1d_output" / "fit_iter0_lm"
    fit_dir.mkdir(parents=True)
    save_problem_json(defn, fit_dir / "problem.json", data_files=data_files)

    out = tmp_path / "imported"
    summary = import_refl1d(str(fit_dir), str(out))

    # ONE state, TWO runs — not two single-file states.
    assert summary["states"] == ["state0"]
    assert summary["n_files"] == 2

    final = json.loads((out / "final_state.json").read_text())
    state = final["state"]["current_model"]["states"][0]
    labels = [ds["label"] for ds in state["data_files"]]
    assert labels == ["run0", "run1"]


def test_state_names_override_bypasses_collapse(tmp_path, two_files):
    """Explicit ``--state-name`` arguments must force multi-state
    interpretation even when ambient + substrate match across samples.
    """
    from aure.nodes.model_builder import save_problem_json
    from aure.refl1d_import import import_refl1d

    defn = _single_state_definition(two_files[0])
    data_files = [
        {"file": two_files[0], "label": "a"},
        {"file": two_files[1], "label": "b"},
    ]
    fit_dir = tmp_path / "refl1d_output" / "fit_iter0_lm"
    fit_dir.mkdir(parents=True)
    save_problem_json(defn, fit_dir / "problem.json", data_files=data_files)

    out = tmp_path / "imported"
    summary = import_refl1d(str(fit_dir), str(out), state_names=["before", "after"])
    assert summary["states"] == ["before", "after"]
    assert summary["n_files"] == 2


def test_setup_drives_state_names_and_description(tmp_path, two_files):
    """When ``--setup`` points at a YAML with the original metadata, the
    importer uses the setup's state names, sample description, and
    original file paths verbatim — no auto-detection.
    """
    import yaml

    from aure.refl1d_import import import_refl1d

    src = _save_problem_to(tmp_path, _two_state_definition(two_files))

    setup_path = tmp_path / "plan.yaml"
    setup_path.write_text(
        yaml.safe_dump(
            {
                "sample_description": "Cu/Ti on Si in D2O and H2O (annealed)",
                "hypothesis": "Cu has a thin oxide skin",
                "states": [
                    {"name": "D2O", "data_files": [{"file": two_files[0]}]},
                    {"name": "H2O", "data_files": [{"file": two_files[1]}]},
                ],
            }
        )
    )

    out = tmp_path / "imported"
    summary = import_refl1d(str(src), str(out), setup_path=str(setup_path))

    assert summary["states"] == ["D2O", "H2O"]
    final = json.loads((out / "final_state.json").read_text())
    state = final["state"]
    assert state["sample_description"] == "Cu/Ti on Si in D2O and H2O (annealed)"
    assert state["hypothesis"] == "Cu has a thin oxide skin"
    # State names come from the setup, not from auto-defaults.
    state_names = [st["name"] for st in state["current_model"]["states"]]
    assert state_names == ["D2O", "H2O"]


def test_setup_references_original_data_files(tmp_path, two_files):
    """The imported model should point at the original setup files, not
    probe dumps under ``<output>/data/``.
    """
    import yaml

    from aure.refl1d_import import import_refl1d

    src = _save_problem_to(tmp_path, _two_state_definition(two_files))
    setup_path = tmp_path / "plan.yaml"
    setup_path.write_text(
        yaml.safe_dump(
            {
                "sample_description": "x",
                "states": [
                    {"name": "D2O", "data_files": [{"file": two_files[0]}]},
                    {"name": "H2O", "data_files": [{"file": two_files[1]}]},
                ],
            }
        )
    )

    out = tmp_path / "imported"
    import_refl1d(str(src), str(out), setup_path=str(setup_path))

    final = json.loads((out / "final_state.json").read_text())
    state = final["state"]
    files_in_model = sorted(
        ds["file"] for st in state["current_model"]["states"] for ds in st["data_files"]
    )
    # The setup paths win — the model points at the originals (resolved).
    assert files_in_model == sorted(os.path.realpath(p) for p in two_files)
    # No probe-dump directory was populated for the model's files.
    for path in files_in_model:
        assert not str(path).startswith(str(out / "data"))


def test_setup_mismatched_file_count_errors(tmp_path, two_files, one_file):
    """A setup describing a different number of files than the problem
    has experiments must error out clearly.
    """
    import yaml

    from aure.refl1d_import import import_refl1d

    # Problem has 2 experiments
    src = _save_problem_to(tmp_path, _two_state_definition(two_files))

    # Setup declares only 1 file
    setup_path = tmp_path / "wrong.yaml"
    setup_path.write_text(
        yaml.safe_dump(
            {
                "sample_description": "x",
                "states": [
                    {"name": "only_one", "data_files": [{"file": one_file}]},
                ],
            }
        )
    )

    out = tmp_path / "imported"
    with pytest.raises(ValueError, match="must describe the same problem"):
        import_refl1d(str(src), str(out), setup_path=str(setup_path))


def test_setup_with_state_names_cli_override_rejected(tmp_path, two_files):
    """``--state-name`` and ``--setup`` are mutually exclusive — state
    names always come from the setup when supplied.
    """
    import yaml

    from aure.refl1d_import import import_refl1d

    src = _save_problem_to(tmp_path, _two_state_definition(two_files))
    setup_path = tmp_path / "plan.yaml"
    setup_path.write_text(
        yaml.safe_dump(
            {
                "sample_description": "x",
                "states": [
                    {"name": "D2O", "data_files": [{"file": two_files[0]}]},
                    {"name": "H2O", "data_files": [{"file": two_files[1]}]},
                ],
            }
        )
    )

    out = tmp_path / "imported"
    with pytest.raises(ValueError, match="cannot be combined"):
        import_refl1d(
            str(src),
            str(out),
            setup_path=str(setup_path),
            state_names=["A", "B"],
        )


def test_setup_data_dir_overrides_path_resolution(tmp_path, two_files):
    """When the setup YAML references data files by bare name and the
    actual files live in a different directory (the analyzer's typical
    layout — YAML in ``plan/``, data in ``./``), ``--data-dir`` must
    redirect path resolution there.
    """
    import shutil

    import yaml

    from aure.refl1d_import import import_refl1d

    # Move the data files to a "data/" subdir; leave the YAML in "plan/"
    data_dir = tmp_path / "data_root"
    data_dir.mkdir()
    relocated = []
    for src_file in two_files:
        dst = data_dir / Path(src_file).name
        shutil.copy(src_file, dst)
        relocated.append(dst)

    refl1d_src = _save_problem_to(tmp_path, _two_state_definition(two_files))

    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    setup_path = plan_dir / "plan.yaml"
    setup_path.write_text(
        yaml.safe_dump(
            {
                "sample_description": "x",
                "states": [
                    {
                        "name": "D2O",
                        "data_files": [{"file": Path(two_files[0]).name}],
                    },
                    {
                        "name": "H2O",
                        "data_files": [{"file": Path(two_files[1]).name}],
                    },
                ],
            }
        )
    )

    out = tmp_path / "imported"
    # Without --data-dir, the loader would look for the files in plan/,
    # which doesn't contain them → ConfigError.
    from aure.config import ConfigError

    with pytest.raises(ConfigError, match="data file not found"):
        import_refl1d(str(refl1d_src), str(out), setup_path=str(setup_path))

    # With --data-dir, the loader looks in data_root/ instead.
    import_refl1d(
        str(refl1d_src),
        str(out),
        setup_path=str(setup_path),
        setup_data_dir=str(data_dir),
    )
    final = json.loads((out / "final_state.json").read_text())
    files_in_model = sorted(
        ds["file"]
        for st in final["state"]["current_model"]["states"]
        for ds in st["data_files"]
    )
    expected = sorted(str(p.resolve()) for p in relocated)
    assert files_in_model == expected


def test_setup_data_dir_without_setup_errors():
    """``setup_data_dir`` without ``setup_path`` is a programmer error."""
    from aure.refl1d_import import import_refl1d

    with pytest.raises(ValueError, match="setup_data_dir is meaningful"):
        import_refl1d("nowhere", "also_nowhere", setup_data_dir="/tmp")


def test_setup_mode_round_trip_preserves_per_probe_intensities(tmp_path):
    """Single-state co-refinement with per-probe intensities + per-state
    nuisance must round-trip cleanly: the rebuilt problem yields the
    same χ² as the original after applying the imported parameters.

    Regression: per-probe intensities used to collapse under the shared
    name "intensity" (lossy), per-state theta_offset / sample_broadening
    were treated as model-level overrides that ``build_multi_problem``
    ignores, and the dispatch picked the wrong builder.
    """
    import yaml

    from aure.nodes.model_builder import (
        apply_parameters,
        build_states_problem,
        needs_states_problem,
    )
    from aure.refl1d_import import import_refl1d

    # Build a 3-file partial setup so refl1d picks NeutronProbe; enable
    # per-state theta_offset / sample_broadening.
    files = [
        _make_partial_data_file(tmp_path, i + 1, theta_deg=0.5 * (i + 1))
        for i in range(3)
    ]

    defn = _single_state_definition(files[0])
    defn["back_reflection"] = True
    defn["states"] = [
        {
            "name": "run_A",
            "data_files": [
                {"file": f, "theta": 0.5 * (i + 1)} for i, f in enumerate(files)
            ],
            "back_reflection": True,
            "theta_offset": {"init": 0.0, "min": -0.01, "max": 0.01},
            "sample_broadening": {"init": 0.0, "min": 0.0, "max": 0.1},
        }
    ]

    # Build, mutate per-probe intensities to be distinct, save.
    problem, by_state, _ = build_states_problem(defn)
    intensities = [0.83, 1.15, 1.27]
    for exp, val in zip(by_state["run_A"], intensities):
        exp.probe.intensity.value = val
    # Set the nuisance shared params to non-default values too.
    first_exp = by_state["run_A"][0]
    first_exp.probe.theta_offset.value = -0.0034
    first_exp.probe.sample_broadening.value = 0.033
    expected_chi2 = float(problem.chisq())

    from bumps.serialize import save_file

    fit_dir = tmp_path / "refl1d_output" / "fit_iter0_lm"
    fit_dir.mkdir(parents=True)
    save_file(str(fit_dir / "problem.json"), problem)

    # Setup file points at the same originals.
    setup_path = tmp_path / "plan.yaml"
    setup_path.write_text(
        yaml.safe_dump(
            {
                "sample_description": "x",
                "states": [
                    {
                        "name": "run_A",
                        "back_reflection": True,
                        "theta_offset": {"init": 0.0, "min": -0.01, "max": 0.01},
                        "sample_broadening": {"init": 0.0, "min": 0.0, "max": 0.1},
                        "data_files": [{"file": f} for f in files],
                    }
                ],
            }
        )
    )

    out = tmp_path / "imported"
    import_refl1d(str(fit_dir), str(out), setup_path=str(setup_path))

    final = json.loads((out / "final_state.json").read_text())
    model = final["state"]["current_model"]
    params = final["state"]["fit_results"][0]["parameters"]

    # Dispatch must pick the states-problem path (single state + nuisance).
    assert needs_states_problem(model)

    # Per-probe intensities preserved (no collapse).
    intensity_keys = [k for k in params if "intensity" in k]
    assert len(intensity_keys) == 3, intensity_keys
    intensity_values = sorted(params[k] for k in intensity_keys)
    assert intensity_values == pytest.approx(sorted(intensities), rel=1e-9)

    # Round-trip χ² check: rebuilding the problem and re-applying the
    # imported parameters should reproduce the original χ² exactly.
    rebuilt, _, _ = build_states_problem(model)
    apply_parameters(rebuilt, params)
    assert float(rebuilt.chisq()) == pytest.approx(expected_chi2, rel=1e-6)


def _make_partial_data_file(
    tmp_path, idx: int, theta_deg: float, n: int = 60, set_id: int = 2222
) -> str:
    """Write a REF_L-style partial with a TwoTheta header so the
    importer treats it as a NeutronProbe-able partial. All partials
    of one sequence share a single ``set_id`` per the REF_L convention
    (validated by ``_parse_states``).
    """
    Q = np.linspace(0.01 + 0.02 * (idx - 1), 0.10 + 0.02 * (idx - 1), n)
    R = (0.0217 / (2 * Q)) ** 4
    R = np.clip(R, 1e-10, 1.0)
    dR = 0.05 * R
    dQ = 0.02 * Q
    path = tmp_path / f"REFL_{set_id}_{idx}_{set_id + idx}_partial.txt"
    with open(path, "w") as fh:
        fh.write("# DataRun  TwoTheta(deg)  Sequence_id\n")
        fh.write(f"# {set_id}_{idx}  {2 * theta_deg:.4f}  {set_id}_partials\n")
        for q, r, dr, dq in zip(Q, R, dR, dQ):
            fh.write(f"{q:.6f}  {r:.6e}  {dr:.6e}  {dq:.6e}\n")
    return str(path)


def test_setup_mode_recovers_theta_from_file_headers(tmp_path):
    """A setup-driven import must read ``theta`` from the data file
    header so a downstream refine builds a NeutronProbe (enabling
    per-probe ``theta_offset`` / ``sample_broadening``) instead of
    falling back to a QProbe.

    Regression: the importer used to hardcode ``theta=0`` for every
    dataset, even in setup mode, which forced the QProbe path on
    rebuild and dropped the per-segment nuisance parameters.
    """
    import yaml

    from aure.refl1d_import import import_refl1d

    # Build a REF_L-style partial file: header carries TwoTheta(deg),
    # followed by 4-column reflectivity rows. The deterministic theta
    # parser splits TwoTheta in half.
    partial = tmp_path / "REFL_226642_3_226644_partial.txt"
    Q = np.linspace(0.01, 0.10, 60)
    Qc = 0.0217
    R = np.where(Q < Qc, 1.0, (Qc / (2 * Q)) ** 4)
    R = np.clip(R, 1e-10, 1.0)
    dR = 0.05 * R
    dQ = 0.02 * Q
    with open(partial, "w") as fh:
        fh.write("# DataRun  TwoTheta(deg)  Sequence_id\n")
        fh.write("# 226642_3 1.5000         226642_3_partials\n")
        for q, r, dr, dq in zip(Q, R, dR, dQ):
            fh.write(f"{q:.6f}  {r:.6e}  {dr:.6e}  {dq:.6e}\n")

    defn = _single_state_definition(str(partial))
    src = _save_problem_to(tmp_path, defn)

    setup_path = tmp_path / "plan.yaml"
    setup_path.write_text(
        yaml.safe_dump(
            {
                "sample_description": "x",
                "states": [
                    {"name": "S", "data_files": [{"file": str(partial)}]},
                ],
            }
        )
    )

    out = tmp_path / "imported"
    import_refl1d(str(src), str(out), setup_path=str(setup_path))
    final = json.loads((out / "final_state.json").read_text())

    flat = final["state"]["data_files"]
    assert flat, "data_files should be populated"
    # TwoTheta = 1.5 → theta = 0.75
    assert flat[0]["theta"] == pytest.approx(0.75, rel=1e-9)
    # And dq_is_fwhm defaults to True (matches intake's default).
    assert flat[0]["dq_is_fwhm"] is True


def test_setup_extra_description_carried_through(tmp_path, two_files):
    """Per-state ``extra_description`` from the setup is preserved on
    the recovered state, so the LLM sees the scientist's per-state notes.
    """
    import yaml

    from aure.refl1d_import import import_refl1d

    src = _save_problem_to(tmp_path, _two_state_definition(two_files))
    setup_path = tmp_path / "plan.yaml"
    setup_path.write_text(
        yaml.safe_dump(
            {
                "sample_description": "x",
                "states": [
                    {
                        "name": "D2O",
                        "extra_description": "first measurement, fresh sample",
                        "data_files": [{"file": two_files[0]}],
                    },
                    {
                        "name": "H2O",
                        "extra_description": "24h later, same sample",
                        "data_files": [{"file": two_files[1]}],
                    },
                ],
            }
        )
    )

    out = tmp_path / "imported"
    import_refl1d(str(src), str(out), setup_path=str(setup_path))
    final = json.loads((out / "final_state.json").read_text())
    states = final["state"]["current_model"]["states"]
    by_name = {st["name"]: st for st in states}
    assert by_name["D2O"]["extra_description"] == "first measurement, fresh sample"
    assert by_name["H2O"]["extra_description"] == "24h later, same sample"


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
    assert ti_idx < cu_idx, f"layers should read top-down but description says {desc!r}"


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
    assert any("no shared parameter objects" in w for w in summary["warnings"])


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


# ----------------------------------------------------------------------
# Uncertainties recovered from a dream export
# ----------------------------------------------------------------------


def test_uncertainties_are_read_from_a_dream_err_json(tmp_path):
    """`extract_fit_result_from_problem` has no bumps fit_result to read `dx`
    from, so without this the field stays empty and every consumer of it --
    notably the uncertainty half of `_check_boundary_hits` -- silently does
    nothing on the `aure evaluate` path.
    """
    import json

    from aure.refl1d_import import _read_uncertainties

    (tmp_path / "model-err.json").write_text(
        json.dumps(
            {
                "CuOx interface": {"best": 5.11, "std": 0.906, "p95": [5.03, 8.38]},
                "no std here": {"best": 1.0},
                "not a mapping": "surprise",
            }
        )
    )

    assert _read_uncertainties(str(tmp_path)) == {"CuOx interface": 0.906}


def test_missing_uncertainties_are_absent_rather_than_zero(tmp_path):
    """An optimizer run writes no -err.json. Reporting 0.0 would claim
    certainty the fit never established."""
    from aure.refl1d_import import _read_uncertainties

    assert _read_uncertainties(str(tmp_path)) == {}
    assert _read_uncertainties(None) == {}
    assert _read_uncertainties(str(tmp_path / "nope")) == {}


def test_a_malformed_err_json_does_not_break_the_import(tmp_path):
    """The file comes from bumps, and a truncated write must degrade to "no
    uncertainties" rather than failing the whole extraction."""
    from aure.refl1d_import import _read_uncertainties

    (tmp_path / "model-err.json").write_text("{ truncated")

    assert _read_uncertainties(str(tmp_path)) == {}


def test_imported_probe_keeps_its_resolution(tmp_path, two_files):
    """The dumped file is declared FWHM, but refl1d's ``probe.dQ`` is 1-sigma.

    Writing sigma under the FWHM label made the reload convert a second time,
    so every imported run fitted with a resolution 2.35x too narrow. Nothing
    reported it — the numbers stayed plausible and only the smearing was wrong
    — and the round-trip test above could not see it while both sides of its
    chi-squared comparison were infinite.
    """
    import numpy as np

    from aure.nodes.model_builder import build_states_problem
    from aure.refl1d_import import import_refl1d

    definition = _two_state_definition(two_files)
    original, by_state_before, _ = build_states_problem(definition)

    src = _save_problem_to(tmp_path, definition)
    out = tmp_path / "imported"
    import_refl1d(str(src), str(out), state_names=["D2O", "H2O"])
    model = json.loads((out / "final_state.json").read_text())["state"]["current_model"]
    _rebuilt, by_state_after, _ = build_states_problem(model)

    for state in ("D2O", "H2O"):
        before = np.asarray(by_state_before[state][0].probe.dQ, dtype=float)
        after = np.asarray(by_state_after[state][0].probe.dQ, dtype=float)
        assert after == pytest.approx(before, rel=1e-9), (
            f"{state}: resolution changed on round-trip by a factor "
            f"{float(after[0] / before[0]):.4f}"
        )


def test_round_trip_chi2_comparison_is_not_vacuous(tmp_path, two_files):
    """Guard the guard: the chi-squared round-trip assertion above is only
    meaningful if the problem is actually evaluable. It passed for a long time
    with `inf == inf`, because a layer whose declared roughness sat below the
    default 5 A interface floor was built outside its own bounds."""
    from aure.nodes.model_builder import build_states_problem

    problem, _by_state, _ = build_states_problem(_two_state_definition(two_files))
    assert not problem._nllf_components()[3], "problem is infeasible at its start"
    assert np.isfinite(problem.chisq())
