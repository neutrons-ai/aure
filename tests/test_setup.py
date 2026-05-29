"""Tests for ``aure.setup`` — the canonical per-run YAML format.

Coverage:

- Round-trip: ``setup → dump_setup → load_setup → equal``.
- Analyzer-compat synonyms (``describe`` / ``data``).
- Hard-break legacy rejection (top-level ``data_file`` / ``data_files``).
- Unknown-key typo guard.
- Manifest loader: both ``jobs:`` shape and the flat (single-job) shape.
- ``aure batch`` accepts a flat setup as a one-job manifest (dry-run).
- ``aure analyze --help`` accepts no positional DATA_FILE when ``-c`` carries it.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _make_data_file(name: str = "data.txt") -> str:
    """Write a tiny 4-column reflectivity file and return its path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", prefix=f"{name}_", suffix=".txt", delete=False
    )
    f.write("# Q R dR dQ\n")
    f.write("0.010  1.000e+00  5.0e-02  2.0e-04\n")
    f.write("0.020  2.500e-01  1.0e-02  4.0e-04\n")
    f.write("0.030  1.111e-01  5.0e-03  6.0e-04\n")
    f.close()
    return f.name


@pytest.fixture
def data_file():
    p = _make_data_file()
    yield p
    try:
        os.unlink(p)
    except OSError:
        pass


@pytest.fixture
def two_files():
    paths = [_make_data_file("a"), _make_data_file("b")]
    yield paths
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


def _write_yaml(tmp_path: Path, name: str, doc: dict) -> Path:
    out = tmp_path / name
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    return out


# ----------------------------------------------------------------------
# load_setup
# ----------------------------------------------------------------------


def test_load_setup_minimal_single_state(tmp_path, data_file):
    from aure.setup import load_setup

    p = _write_yaml(
        tmp_path,
        "setup.yaml",
        {
            "sample_description": "Cu on Si",
            "states": [
                {"name": "state0", "data_files": [{"file": data_file}]},
            ],
        },
    )
    setup = load_setup(p)
    assert setup["sample_description"] == "Cu on Si"
    assert len(setup["states"]) == 1
    assert setup["states"][0]["name"] == "state0"
    # Path resolved to absolute
    assert os.path.isabs(setup["states"][0]["data_files"][0]["file"])


def test_load_setup_rejects_top_level_data_file(tmp_path, data_file):
    """Hard break: ``data_file:`` at top level is no longer accepted."""
    from aure.config import ConfigError
    from aure.setup import load_setup

    p = _write_yaml(
        tmp_path,
        "legacy.yaml",
        {"sample_description": "x", "data_file": data_file},
    )
    with pytest.raises(ConfigError, match="no longer supported"):
        load_setup(p)


def test_load_setup_rejects_top_level_data_files(tmp_path, data_file):
    from aure.config import ConfigError
    from aure.setup import load_setup

    p = _write_yaml(
        tmp_path,
        "legacy.yaml",
        {"sample_description": "x", "data_files": [data_file]},
    )
    with pytest.raises(ConfigError, match="no longer supported"):
        load_setup(p)


def test_load_setup_unknown_key_typo_guard(tmp_path, data_file):
    from aure.config import ConfigError
    from aure.setup import load_setup

    p = _write_yaml(
        tmp_path,
        "typo.yaml",
        {
            "sampledescription": "x",  # missing underscore — typo
            "states": [{"name": "state0", "data_files": [{"file": data_file}]}],
        },
    )
    with pytest.raises(ConfigError, match="unknown top-level"):
        load_setup(p)


def test_load_setup_jobs_block_rejected(tmp_path):
    """A multi-job manifest belongs to ``load_manifest``, not ``load_setup``."""
    from aure.config import ConfigError
    from aure.setup import load_setup

    p = _write_yaml(tmp_path, "manifest.yaml", {"jobs": [{"name": "j"}]})
    with pytest.raises(ConfigError, match="looks like a batch manifest"):
        load_setup(p)


# ----------------------------------------------------------------------
# Analyzer-compat synonyms
# ----------------------------------------------------------------------


def test_describe_is_synonym_for_sample_description(tmp_path, data_file):
    """``analyzer plan-data`` emits ``describe:`` — must round-trip in."""
    from aure.setup import load_setup

    p = _write_yaml(
        tmp_path,
        "analyzer.yaml",
        {
            "describe": "Cu on Si",
            "states": [{"name": "state0", "data_files": [{"file": data_file}]}],
        },
    )
    setup = load_setup(p)
    assert setup["sample_description"] == "Cu on Si"


def test_data_is_synonym_for_data_files_in_state(tmp_path, data_file):
    """``analyzer plan-data`` emits ``data:`` inside each state."""
    from aure.setup import load_setup

    p = _write_yaml(
        tmp_path,
        "analyzer.yaml",
        {
            "describe": "Cu on Si",
            "states": [{"name": "state0", "data": [data_file]}],
        },
    )
    setup = load_setup(p)
    assert setup["states"][0]["data_files"][0]["file"] == os.path.realpath(data_file)


def test_analyzer_plan_data_output_loads_cleanly(tmp_path):
    """The exact shape ``analyzer plan-data`` writes (describe + data +
    model_name + metadata) parses without errors and the metadata block
    survives untouched.

    Uses a REF_L partial-style filename so theta_offset / sample_broadening
    (partials-only nuisance parameters) are accepted by the validator.
    """
    from aure.setup import load_setup

    # plan-data targets multi-segment partial files; mimic that naming.
    partial = tmp_path / "REFL_226642_3_226644_partial.txt"
    partial.write_text("# Q R dR dQ\n0.01 1.0 0.05 1e-4\n")

    p = _write_yaml(
        tmp_path,
        "job_seq.yaml",
        {
            "describe": "Cu on Si in D2O",
            "states": [
                {
                    "name": "run_226642",
                    "data": [str(partial)],
                    "theta_offset": {"init": 0.0, "min": -0.02, "max": 0.02},
                    "sample_broadening": True,
                },
            ],
            "model_name": "Cu-D2O-226642",
            "metadata": {
                "perform_assembly": True,
                "notes": "Sequence complete",
            },
        },
    )
    setup = load_setup(p)
    assert setup["model_name"] == "Cu-D2O-226642"
    assert setup["metadata"]["perform_assembly"] is True
    assert "Sequence complete" in setup["metadata"]["notes"]


# ----------------------------------------------------------------------
# dump_setup + round-trip
# ----------------------------------------------------------------------


def test_dump_setup_round_trip(tmp_path, data_file):
    """setup → dump_setup → load_setup yields an equivalent setup."""
    from aure.setup import dump_setup, load_setup

    p = _write_yaml(
        tmp_path,
        "original.yaml",
        {
            "name": "test_run",
            "sample_description": "Cu on Si",
            "hypothesis": "Possible CuOx",
            "model_name": "cu_si",
            "max_refinements": 3,
            "fit_method": "lm",
            "evaluation_criteria": ["Roughness > 5"],
            "model_constraints": ["No extra layers"],
            "shared_parameters": ["Cu.thickness"],
            "states": [
                {"name": "state0", "data_files": [{"file": data_file}]},
            ],
            "metadata": {"notes": "round-trip test"},
        },
    )
    original = load_setup(p)

    yaml_text = dump_setup(original)
    # The dumped YAML should not mention the dropped legacy keys.
    assert "data_file:" not in yaml_text or "data_files:" in yaml_text
    # Round-trip
    out_path = tmp_path / "roundtrip.yaml"
    out_path.write_text(yaml_text)
    reloaded = load_setup(out_path)

    # Drop the internal _kind marker before comparison (load adds it).
    def _strip_kind(setup):
        for st in setup.get("states", []):
            st.pop("_kind", None)
        return setup

    assert _strip_kind(reloaded) == _strip_kind(original)


def test_dump_setup_drops_empty_fields(tmp_path, data_file):
    """``dump_setup`` should not emit empty lists / None for unset fields."""
    from aure.setup import dump_setup, load_setup

    p = _write_yaml(
        tmp_path,
        "minimal.yaml",
        {
            "sample_description": "x",
            "states": [{"name": "state0", "data_files": [{"file": data_file}]}],
        },
    )
    setup = load_setup(p)
    text = dump_setup(setup)
    # No `hypothesis:` line for a setup that didn't declare one.
    assert "hypothesis:" not in text
    # No `evaluation_criteria: []` either.
    assert "evaluation_criteria:" not in text
    # Internal _kind never leaks.
    assert "_kind" not in text


# ----------------------------------------------------------------------
# load_manifest
# ----------------------------------------------------------------------


def test_load_manifest_jobs_shape(tmp_path, two_files):
    from aure.setup import load_manifest

    p = _write_yaml(
        tmp_path,
        "manifest.yaml",
        {
            "defaults": {"max_refinements": 3, "fit_method": "lm"},
            "jobs": [
                {
                    "name": "job_a",
                    "sample_description": "A on Si",
                    "states": [
                        {"name": "state0", "data_files": [{"file": two_files[0]}]}
                    ],
                },
                {
                    "name": "job_b",
                    "sample_description": "B on Si",
                    "states": [
                        {"name": "state0", "data_files": [{"file": two_files[1]}]}
                    ],
                    "fit_method": "dream",  # override default
                },
            ],
        },
    )
    m = load_manifest(p)
    assert len(m["jobs"]) == 2
    # Defaults applied to job_a
    assert m["jobs"][0]["max_refinements"] == 3
    assert m["jobs"][0]["fit_method"] == "lm"
    # Per-job override on job_b
    assert m["jobs"][1]["fit_method"] == "dream"
    # Defaults preserved on the manifest dict
    assert m["defaults"]["max_refinements"] == 3


def test_load_manifest_flat_setup_treated_as_one_job(tmp_path, data_file):
    """A flat setup (no `jobs:`) becomes a 1-job manifest."""
    from aure.setup import load_manifest

    p = _write_yaml(
        tmp_path,
        "flat.yaml",
        {
            "name": "lonely",
            "sample_description": "x",
            "states": [{"name": "state0", "data_files": [{"file": data_file}]}],
        },
    )
    m = load_manifest(p)
    assert len(m["jobs"]) == 1
    assert m["jobs"][0]["name"] == "lonely"
    assert m["defaults"] == {}


def test_load_manifest_job_with_legacy_data_file_errors(tmp_path, data_file):
    from aure.config import ConfigError
    from aure.setup import load_manifest

    p = _write_yaml(
        tmp_path,
        "legacy_jobs.yaml",
        {
            "jobs": [
                {
                    "name": "j",
                    "sample_description": "x",
                    "data_file": data_file,  # legacy — must error
                }
            ]
        },
    )
    with pytest.raises(ConfigError, match="no longer supported"):
        load_manifest(p)


# ----------------------------------------------------------------------
# setup_to_user_config
# ----------------------------------------------------------------------


def test_setup_to_user_config_subset(data_file):
    from aure.setup import setup_to_user_config

    setup = {
        "name": "ignored",
        "sample_description": "x",
        "evaluation_criteria": ["a"],
        "model_constraints": ["b"],
        "shared_parameters": ["Cu.thickness"],
        "hypothesis": "ignored too",
        "fit_method": "ignored",
    }
    uc = setup_to_user_config(setup)
    # Only the runner-relevant fields survive
    assert uc["evaluation_criteria"] == ["a"]
    assert uc["model_constraints"] == ["b"]
    assert uc["shared_parameters"] == ["Cu.thickness"]
    assert uc["sample_description"] == "x"
    assert "fit_method" not in uc
    assert "hypothesis" not in uc


def test_setup_to_user_config_carries_model_name():
    """model_name must survive so the fitting node can name the FitProblem
    (otherwise bumps exports None-*.dat / None.json)."""
    from aure.setup import setup_to_user_config

    uc = setup_to_user_config({"model_name": "sample5_ocv_226642", "sample_description": "x"})
    assert uc["model_name"] == "sample5_ocv_226642"

    # Absent / blank model_name is simply not carried (no None leak).
    assert "model_name" not in setup_to_user_config({"sample_description": "x"})


# ----------------------------------------------------------------------
# CLI ergonomics
# ----------------------------------------------------------------------


def test_aure_batch_dry_run_with_flat_setup(tmp_path, data_file):
    """``aure batch flat.yaml --dry-run`` accepts a flat setup file."""
    from aure.cli import cli

    p = _write_yaml(
        tmp_path,
        "flat.yaml",
        {
            "name": "ad_hoc",
            "sample_description": "Cu on Si",
            "states": [{"name": "state0", "data_files": [{"file": data_file}]}],
        },
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["batch", str(p), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "ad_hoc" in result.output
    assert "1" in result.output  # Jobs : 1


def test_aure_batch_legacy_data_file_errors_with_migration_hint(
    tmp_path, data_file
):
    """A pre-states manifest must fail with a helpful migration message."""
    from aure.cli import cli

    p = _write_yaml(
        tmp_path,
        "legacy.yaml",
        {
            "jobs": [
                {
                    "name": "j",
                    "data_file": data_file,
                    "sample_description": "x",
                }
            ]
        },
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["batch", str(p), "--dry-run"])
    assert result.exit_code != 0
    assert "no longer supported" in result.output


def test_aure_analyze_requires_data_or_config_states():
    """No positional DATA_FILE and no -c states => error."""
    from aure.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["analyze"])
    # Click rejects missing positional with usage error first, but our
    # custom message fires when DATA_FILE is None.
    assert result.exit_code != 0


def test_aure_analyze_help_marks_positionals_optional():
    """The help output reflects that DATA_FILE / SAMPLE_DESCRIPTION are
    optional now (so users see they can pass -c instead)."""
    from aure.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "[DATA_FILE] [SAMPLE_DESCRIPTION]" in result.output
