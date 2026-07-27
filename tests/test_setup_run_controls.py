"""Tests for the setup YAML's run-control keys.

Coverage:

- ``chi2_max`` and the four final-fit keys load with the right Python types.
- ``dump_setup`` round-trips them (dump → load → same values).
- Numeric validation: non-numeric, zero, negative and non-finite χ² thresholds
  are rejected at load time rather than silently disabling acceptance.
- ``_build_env_overrides`` maps ``chi2_max`` → ``CHI2_MAX`` and forwards only
  the keys the setup actually declares.
- The env-override context manager restores the prior environment, including
  unsetting keys that were absent (not leaving them as empty strings).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def data_file():
    f = tempfile.NamedTemporaryFile(mode="w", prefix="rc_", suffix=".txt", delete=False)
    f.write("# Q R dR dQ\n")
    f.write("0.010  1.000e+00  5.0e-02  2.0e-04\n")
    f.write("0.020  2.500e-01  1.0e-02  4.0e-04\n")
    f.close()
    yield f.name
    try:
        os.unlink(f.name)
    except OSError:
        pass


def _write_yaml(tmp_path: Path, name: str, doc: dict) -> Path:
    out = tmp_path / name
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    return out


def _setup_doc(data_file: str, **run_controls) -> dict:
    doc = {
        "sample_description": "Cu on Si",
        "states": [{"name": "state0", "data_files": [{"file": data_file}]}],
    }
    doc.update(run_controls)
    return doc


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def test_run_controls_load_with_correct_types(tmp_path, data_file):
    from aure.setup import load_setup

    p = _write_yaml(
        tmp_path,
        "controls.yaml",
        _setup_doc(
            data_file,
            max_refinements=3,
            chi2_max=2.5,
            fit_method="amoeba",
            fit_steps=500,
            fit_burn=200,
            fit_method_final="dream",
            fit_steps_final=10000,
            fit_burn_final=8000,
            final_fit_chi2_max=1.5,
        ),
    )
    setup = load_setup(p)

    assert setup["chi2_max"] == 2.5
    assert isinstance(setup["chi2_max"], float)
    assert setup["fit_method_final"] == "dream"
    assert setup["fit_steps_final"] == 10000
    assert setup["fit_burn_final"] == 8000
    assert isinstance(setup["fit_steps_final"], int)
    assert isinstance(setup["fit_burn_final"], int)
    assert setup["final_fit_chi2_max"] == 1.5
    assert isinstance(setup["final_fit_chi2_max"], float)


def test_chi2_max_accepts_integer_yaml_scalar(tmp_path, data_file):
    """`chi2_max: 3` is a YAML int — it must still land as a float."""
    from aure.setup import load_setup

    p = _write_yaml(tmp_path, "int.yaml", _setup_doc(data_file, chi2_max=3))
    setup = load_setup(p)
    assert setup["chi2_max"] == 3.0
    assert isinstance(setup["chi2_max"], float)


def test_run_controls_absent_are_not_defaulted(tmp_path, data_file):
    """Unspecified keys stay absent so the ambient env / .env still wins."""
    from aure.setup import load_setup

    p = _write_yaml(tmp_path, "bare.yaml", _setup_doc(data_file))
    setup = load_setup(p)
    for key in (
        "chi2_max",
        "fit_method_final",
        "fit_steps_final",
        "fit_burn_final",
        "final_fit_chi2_max",
    ):
        assert key not in setup


def test_example_config_run_controls_are_known_keys():
    """Every run-control key documented in the shipped example must load."""
    from aure.setup import _KNOWN_TOP_LEVEL

    for key in (
        "chi2_max",
        "fit_method_final",
        "fit_steps_final",
        "fit_burn_final",
        "final_fit_chi2_max",
    ):
        assert key in _KNOWN_TOP_LEVEL


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_non_numeric_chi2_max_rejected(tmp_path, data_file):
    from aure.config import ConfigError
    from aure.setup import load_setup

    p = _write_yaml(tmp_path, "bad.yaml", _setup_doc(data_file, chi2_max="tight"))
    with pytest.raises(ConfigError, match="`chi2_max` must be a number"):
        load_setup(p)


@pytest.mark.parametrize("bad", [0, 0.0, -1.5])
def test_non_positive_chi2_max_rejected(tmp_path, data_file, bad):
    """A threshold ≤ 0 can never be met — fail loudly instead."""
    from aure.config import ConfigError
    from aure.setup import load_setup

    p = _write_yaml(tmp_path, "bad.yaml", _setup_doc(data_file, chi2_max=bad))
    with pytest.raises(ConfigError, match="finite positive"):
        load_setup(p)


@pytest.mark.parametrize("bad", [".nan", ".inf", "-.inf"])
def test_non_finite_chi2_max_rejected(tmp_path, data_file, bad):
    from aure.config import ConfigError
    from aure.setup import load_setup

    p = tmp_path / "bad.yaml"
    p.write_text(
        f"sample_description: Cu on Si\nchi2_max: {bad}\n"
        f"states:\n  - name: state0\n    data_files:\n      - file: {data_file}\n"
    )
    with pytest.raises(ConfigError, match="finite positive"):
        load_setup(p)


@pytest.mark.parametrize("bad", [0, -2.0])
def test_non_positive_final_fit_chi2_max_rejected(tmp_path, data_file, bad):
    from aure.config import ConfigError
    from aure.setup import load_setup

    p = _write_yaml(tmp_path, "bad.yaml", _setup_doc(data_file, final_fit_chi2_max=bad))
    with pytest.raises(ConfigError, match="`final_fit_chi2_max`"):
        load_setup(p)


def test_non_integer_fit_steps_final_rejected(tmp_path, data_file):
    from aure.config import ConfigError
    from aure.setup import load_setup

    p = _write_yaml(tmp_path, "bad.yaml", _setup_doc(data_file, fit_steps_final="lots"))
    with pytest.raises(ConfigError, match="`fit_steps_final` must be an integer"):
        load_setup(p)


# ----------------------------------------------------------------------
# dump_setup round-trip
# ----------------------------------------------------------------------


def test_run_controls_round_trip(tmp_path, data_file):
    """Web-UI Save Setup must not drop the run controls."""
    from aure.setup import dump_setup, load_setup

    p = _write_yaml(
        tmp_path,
        "original.yaml",
        _setup_doc(
            data_file,
            max_refinements=4,
            chi2_max=2.5,
            fit_method="amoeba",
            fit_steps=500,
            fit_burn=200,
            fit_method_final="dream",
            fit_steps_final=10000,
            fit_burn_final=8000,
            final_fit_chi2_max=1.5,
        ),
    )
    original = load_setup(p)

    text = dump_setup(original)
    assert "chi2_max: 2.5" in text

    round_tripped_path = tmp_path / "round_trip.yaml"
    round_tripped_path.write_text(text)
    round_tripped = load_setup(round_tripped_path)

    for key in (
        "max_refinements",
        "chi2_max",
        "fit_method",
        "fit_steps",
        "fit_burn",
        "fit_method_final",
        "fit_steps_final",
        "fit_burn_final",
        "final_fit_chi2_max",
    ):
        assert round_tripped[key] == original[key]


# ----------------------------------------------------------------------
# Env-var mapping
# ----------------------------------------------------------------------


def test_build_env_overrides_maps_chi2_max():
    from aure.cli import _build_env_overrides

    overrides = _build_env_overrides({"chi2_max": 2.5, "fit_method_final": "dream"})
    assert overrides["CHI2_MAX"] == "2.5"
    assert overrides["FIT_METHOD_FINAL"] == "dream"


def test_build_env_overrides_omits_absent_keys():
    from aure.cli import _build_env_overrides

    overrides = _build_env_overrides({"chi2_max": 1.0})
    assert set(overrides) == {"CHI2_MAX"}


def test_applied_env_overrides_restores_previous_value(monkeypatch):
    from aure.cli import _applied_env_overrides

    monkeypatch.setenv("CHI2_MAX", "9.0")
    with _applied_env_overrides({"chi2_max": 1.25}):
        assert os.environ["CHI2_MAX"] == "1.25"
    assert os.environ["CHI2_MAX"] == "9.0"


def test_applied_env_overrides_restores_absent_as_absent(monkeypatch):
    """An absent key must be unset again, not left as an empty string."""
    from aure.cli import _applied_env_overrides

    monkeypatch.delenv("CHI2_MAX", raising=False)
    with _applied_env_overrides({"chi2_max": 1.25}):
        assert os.environ["CHI2_MAX"] == "1.25"
    assert "CHI2_MAX" not in os.environ


def test_applied_env_overrides_restores_on_exception(monkeypatch):
    from aure.cli import _applied_env_overrides

    monkeypatch.delenv("CHI2_MAX", raising=False)
    with pytest.raises(RuntimeError):
        with _applied_env_overrides({"chi2_max": 1.25}):
            raise RuntimeError("boom")
    assert "CHI2_MAX" not in os.environ


def test_applied_env_overrides_visible_to_evaluation_getter(monkeypatch):
    """The value the evaluation node reads is the setup's, not the env's."""
    from aure.cli import _applied_env_overrides
    from aure.nodes.evaluation import _get_chi2_max

    monkeypatch.setenv("CHI2_MAX", "5.0")
    with _applied_env_overrides({"chi2_max": 1.75}):
        assert _get_chi2_max() == 1.75
    assert _get_chi2_max() == 5.0


# ----------------------------------------------------------------------
# `aure analyze -c setup.yaml` actually applies the run controls
# ----------------------------------------------------------------------


def test_analyze_applies_setup_chi2_max(tmp_path, data_file, monkeypatch):
    """Regression: setup run controls used to be honoured by `batch` only."""
    from click.testing import CliRunner

    import aure.cli as cli_module
    import aure.workflow as workflow_module

    monkeypatch.setenv("CHI2_MAX", "5.0")
    monkeypatch.setattr(cli_module, "_check_llm_status", lambda **kw: (True, "ok"))

    seen: dict = {}

    def fake_run_analysis(**kwargs):
        seen["chi2_max"] = os.environ.get("CHI2_MAX")
        return {
            "Q": [0.01, 0.02],
            "structural_hypotheses": [
                {
                    "id": 1,
                    "title": "Add native CuOx on top of Cu",
                    "change": "insert a 10-30 Å CuOx layer (SLD ~5.0)",
                    "status": "pending",
                },
                {"id": 2, "title": "Free the Ti roughness", "status": "confirmed"},
            ],
        }

    monkeypatch.setattr(workflow_module, "run_analysis", fake_run_analysis)

    p = _write_yaml(tmp_path, "setup.yaml", _setup_doc(data_file, chi2_max=1.25))
    result = CliRunner().invoke(cli_module.cli, ["analyze", "-c", str(p)])

    assert result.exit_code == 0, result.output
    assert seen["chi2_max"] == "1.25"
    assert "Accept when χ² ≤ 1.25" in result.output
    # Untried hypotheses are the only surviving record of what was skipped.
    assert "Possible further improvements" in result.output
    assert "Add native CuOx on top of Cu" in result.output
    assert "insert a 10-30 Å CuOx layer" in result.output
    assert "Free the Ti roughness" not in result.output
    assert "(1 of 2 attempted — 1 confirmed, 0 rejected)" in result.output
    # And the run must not leak its override into the ambient environment.
    assert os.environ["CHI2_MAX"] == "5.0"


def test_analyze_report_omits_improvements_when_none_pending(
    tmp_path, data_file, monkeypatch
):
    from click.testing import CliRunner

    import aure.cli as cli_module
    import aure.workflow as workflow_module

    monkeypatch.setattr(cli_module, "_check_llm_status", lambda **kw: (True, "ok"))
    monkeypatch.setattr(
        workflow_module,
        "run_analysis",
        lambda **kw: {
            "Q": [0.01],
            "structural_hypotheses": [{"id": 1, "title": "x", "status": "rejected"}],
        },
    )

    p = _write_yaml(tmp_path, "setup.yaml", _setup_doc(data_file))
    result = CliRunner().invoke(cli_module.cli, ["analyze", "-c", str(p)])

    assert result.exit_code == 0, result.output
    assert "Possible further improvements" not in result.output
