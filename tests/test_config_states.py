"""Tests for the multi-state ``states:`` block in user configuration YAML."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from aure.config import ConfigError, load_user_config, states_from_config


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _write(tmp_path: Path, name: str, content: str = "data\n") -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def _write_yaml(tmp_path: Path, body: str) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(body))
    return cfg


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


def test_load_user_config_no_states_returns_empty_states(tmp_path: Path) -> None:
    cfg_path = _write_yaml(
        tmp_path,
        """
        evaluation_criteria:
          - "be reasonable"
        """,
    )
    cfg = load_user_config(cfg_path)
    assert cfg["states"] == []
    assert cfg["evaluation_criteria"] == ["be reasonable"]


def test_two_combined_states_round_trip(tmp_path: Path) -> None:
    f1 = _write(tmp_path, "REFL_226642_combined_data_auto.txt")
    f2 = _write(tmp_path, "REFL_226660_combined_data_auto.txt")
    cfg_path = _write_yaml(
        tmp_path,
        f"""
        sample_description: |
          50 nm Cu / 3 nm Ti on Si
        states:
          - name: D2O
            data_files: [{f1.name}]
          - name: H2O
            extra_description: 24 h later
            back_reflection: true
            data_files: [{f2.name}]
        shared_parameters:
          - Cu.thickness
          - Cu.material.rho
        """,
    )
    cfg = load_user_config(cfg_path)
    assert len(cfg["states"]) == 2
    assert [s["name"] for s in cfg["states"]] == ["D2O", "H2O"]
    assert all(s["_kind"] == "combined" for s in cfg["states"])
    assert cfg["states"][1]["back_reflection"] is True
    assert cfg["states"][1]["extra_description"] == "24 h later"
    assert cfg["sample_description"].startswith("50 nm Cu")
    assert cfg["shared_parameters"] == ["Cu.thickness", "Cu.material.rho"]
    assert cfg["unshared_parameters"] == []
    runner_states = states_from_config(cfg)
    assert len(runner_states) == 2
    # files were resolved to absolute paths
    assert Path(runner_states[0]["data_files"][0]["file"]).is_absolute()


def test_partials_state_with_theta_offset(tmp_path: Path) -> None:
    a = _write(tmp_path, "REFL_226642_1_2001_partial.txt")
    b = _write(tmp_path, "REFL_226642_2_2002_partial.txt")
    cfg_path = _write_yaml(
        tmp_path,
        f"""
        states:
          - name: as-deposited
            data_files: [{a.name}, {b.name}]
            theta_offset: true
            sample_broadening: {{init: 0.0, min: 0.0, max: 0.05}}
        """,
    )
    cfg = load_user_config(cfg_path)
    st = cfg["states"][0]
    assert st["_kind"] == "partials"
    assert st["theta_offset"] == {"init": 0.0, "min": -0.02, "max": 0.02}
    assert st["sample_broadening"]["max"] == 0.05


# ----------------------------------------------------------------------
# Validation errors
# ----------------------------------------------------------------------


def test_missing_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # chdir into the (clean) config dir so the cwd fallback can't pick up a
    # stray same-named file from the repo root.
    monkeypatch.chdir(tmp_path)
    cfg_path = _write_yaml(
        tmp_path,
        """
        states:
          - name: A
            data_files: [does_not_exist.txt]
        """,
    )
    with pytest.raises(ConfigError, match="not found"):
        load_user_config(cfg_path)


def test_duplicate_state_names_rejected(tmp_path: Path) -> None:
    f = _write(tmp_path, "REFL_1_combined_data_auto.txt")
    cfg_path = _write_yaml(
        tmp_path,
        f"""
        states:
          - name: A
            data_files: [{f.name}]
          - name: A
            data_files: [{f.name}]
        """,
    )
    with pytest.raises(ConfigError, match="Duplicate"):
        load_user_config(cfg_path)


def test_mixed_combined_and_partials_in_one_state_rejected(tmp_path: Path) -> None:
    f1 = _write(tmp_path, "REFL_1_combined_data_auto.txt")
    f2 = _write(tmp_path, "REFL_1_2_3_partial.txt")
    cfg_path = _write_yaml(
        tmp_path,
        f"""
        states:
          - name: A
            data_files: [{f1.name}, {f2.name}]
        """,
    )
    with pytest.raises(ConfigError, match="cannot mix"):
        load_user_config(cfg_path)


def test_partials_with_mixed_set_ids_rejected(tmp_path: Path) -> None:
    a = _write(tmp_path, "REFL_111_1_1_partial.txt")
    b = _write(tmp_path, "REFL_222_1_1_partial.txt")
    cfg_path = _write_yaml(
        tmp_path,
        f"""
        states:
          - name: A
            data_files: [{a.name}, {b.name}]
        """,
    )
    with pytest.raises(ConfigError, match="set_id"):
        load_user_config(cfg_path)


def test_theta_offset_on_combined_state_rejected(tmp_path: Path) -> None:
    f = _write(tmp_path, "REFL_1_combined_data_auto.txt")
    cfg_path = _write_yaml(
        tmp_path,
        f"""
        states:
          - name: A
            data_files: [{f.name}]
            theta_offset: true
        """,
    )
    with pytest.raises(ConfigError, match="partials"):
        load_user_config(cfg_path)


def test_background_allowed_on_combined_state_and_expands_true(tmp_path: Path) -> None:
    f = _write(tmp_path, "REFL_1_combined_data_auto.txt")
    cfg_path = _write_yaml(
        tmp_path,
        f"""
        states:
          - name: A
            data_files: [{f.name}]
            background: true
        """,
    )
    cfg = load_user_config(cfg_path)
    st = cfg["states"][0]
    # Not partials-only: a combined state accepts background, and `true`
    # expands to the default fittable {init, min, max} triplet.
    assert st["_kind"] == "combined"
    assert st["background"] == {"init": 1e-6, "min": 0.0, "max": 1e-5}


def test_background_dict_passthrough(tmp_path: Path) -> None:
    f = _write(tmp_path, "REFL_1_combined_data_auto.txt")
    cfg_path = _write_yaml(
        tmp_path,
        f"""
        states:
          - name: A
            data_files: [{f.name}]
            background:
              init: 2.0e-6
              min: 0.0
              max: 1.0e-5
        """,
    )
    cfg = load_user_config(cfg_path)
    assert cfg["states"][0]["background"] == {"init": 2e-6, "min": 0.0, "max": 1e-5}


def test_shared_and_unshared_mutually_exclusive(tmp_path: Path) -> None:
    f = _write(tmp_path, "REFL_1_combined_data_auto.txt")
    cfg_path = _write_yaml(
        tmp_path,
        f"""
        states:
          - name: A
            data_files: [{f.name}]
        shared_parameters: [Cu.thickness]
        unshared_parameters: [Cu.interface]
        """,
    )
    with pytest.raises(ConfigError, match="mutually exclusive"):
        load_user_config(cfg_path)


def test_states_must_be_a_list(tmp_path: Path) -> None:
    cfg_path = _write_yaml(
        tmp_path,
        """
        states:
          name: oops
        """,
    )
    with pytest.raises(ConfigError, match="must be a list"):
        load_user_config(cfg_path)


def test_state_without_files_rejected(tmp_path: Path) -> None:
    cfg_path = _write_yaml(
        tmp_path,
        """
        states:
          - name: A
            data_files: []
        """,
    )
    with pytest.raises(ConfigError, match="at least one file"):
        load_user_config(cfg_path)


# ----------------------------------------------------------------------
# Aliases / edge cases
# ----------------------------------------------------------------------


def test_description_alias_for_sample_description(tmp_path: Path) -> None:
    cfg_path = _write_yaml(
        tmp_path,
        """
        description: 1 nm Au on Si
        """,
    )
    cfg = load_user_config(cfg_path)
    assert cfg["sample_description"] == "1 nm Au on Si"


def test_states_from_config_handles_none() -> None:
    assert states_from_config(None) == []
    assert states_from_config({}) == []


def test_missing_config_returns_empty_states(tmp_path: Path) -> None:
    cfg = load_user_config(tmp_path / "nope.yaml")
    assert cfg["states"] == []
    assert cfg["evaluation_criteria"] == []


# ----------------------------------------------------------------------
# Data-file path resolution (candidate-root search)
# ----------------------------------------------------------------------


def test_falls_back_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # YAML lives in plan/, data lives in data/, the file is referenced by bare
    # name and is NOT next to the YAML. Running from the data dir should resolve
    # it via the cwd fallback.
    plan_dir = tmp_path / "plan"
    data_dir = tmp_path / "data"
    plan_dir.mkdir()
    data_dir.mkdir()
    data_file = data_dir / "REFL_1_combined_data_auto.txt"
    data_file.write_text("data\n")
    cfg_path = plan_dir / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            f"""
            states:
              - name: A
                data_files: [{data_file.name}]
            """
        )
    )
    monkeypatch.chdir(data_dir)
    cfg = load_user_config(cfg_path)
    resolved = Path(cfg["states"][0]["data_files"][0]["file"])
    assert resolved == data_file.resolve()


def test_not_found_error_lists_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    cfg_path = plan_dir / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            states:
              - name: A
                data_files: [nowhere.txt]
            """
        )
    )
    with pytest.raises(ConfigError) as excinfo:
        load_user_config(cfg_path)
    msg = str(excinfo.value)
    assert "data file not found" in msg
    assert "Searched (in priority order)" in msg
    # both the YAML dir and the cwd should be listed
    assert str(plan_dir.resolve()) in msg
    assert str(tmp_path.resolve()) in msg
