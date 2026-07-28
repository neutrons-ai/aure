"""Tests for the setup YAML's χ² run controls and the untried-backlog report.

Coverage:

- ``chi2_max`` / ``chi2_min`` load with the right Python types, round-trip
  through ``dump_setup``, and are validated at load time: non-numeric, boolean,
  non-positive ceiling, negative floor, non-finite, and a floor at or above the
  ceiling are all rejected instead of silently disabling acceptance. ``0`` is the
  floor's documented off switch and must survive as an explicit ``0.0``.
- ``_build_env_overrides`` maps them to ``CHI2_MAX`` / ``CHI2_MIN``, and
  ``_applied_env_overrides`` puts a setup's run controls in force for the
  duration of an ``aure analyze`` / ``aure prepare`` run and then restores the
  environment. Without it a YAML ``chi2_max:`` was honoured by ``aure batch``
  only, and silently ignored everywhere else.
- The ``analyze`` banner records the acceptance window that was actually in
  force, floor included.
- The untried-hypothesis backlog reaches the surfaces that carry it: the human
  report, ``aure analyze --json`` and ``aure batch``.
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


def _write_raw(tmp_path: Path, data_file: str, key: str, literal: str) -> Path:
    """A setup written as text, so YAML scalars like ``.nan`` reach the loader."""
    p = tmp_path / "raw.yaml"
    p.write_text(
        f"sample_description: Cu on Si\n{key}: {literal}\n"
        f"states:\n  - name: state0\n    data_files:\n      - file: {data_file}\n"
    )
    return p


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def test_chi2_thresholds_load_as_floats(tmp_path, data_file):
    from aure.setup import load_setup

    p = _write_yaml(
        tmp_path,
        "controls.yaml",
        # `chi2_max: 3` is a YAML int — it must still land as a float.
        _setup_doc(data_file, chi2_max=3, chi2_min=0.4),
    )
    setup = load_setup(p)

    assert setup["chi2_max"] == 3.0
    assert isinstance(setup["chi2_max"], float)
    assert setup["chi2_min"] == 0.4
    assert isinstance(setup["chi2_min"], float)


@pytest.mark.parametrize("zero", [0, 0.0])
def test_chi2_min_zero_disables_the_floor(tmp_path, data_file, zero):
    """0 is a legal floor ("accept any χ² under the ceiling"), not a typo.

    It must survive as an explicit 0.0 rather than being dropped as falsy, or the
    run would silently fall back to the default 0.5 floor.
    """
    from aure.setup import load_setup

    p = _write_yaml(tmp_path, "off.yaml", _setup_doc(data_file, chi2_min=zero))
    setup = load_setup(p)
    assert setup["chi2_min"] == 0.0
    assert isinstance(setup["chi2_min"], float)


def test_chi2_thresholds_absent_are_not_defaulted(tmp_path, data_file):
    """Unspecified keys stay absent so the ambient env / .env still wins."""
    from aure.setup import load_setup

    setup = load_setup(_write_yaml(tmp_path, "bare.yaml", _setup_doc(data_file)))
    assert "chi2_max" not in setup
    assert "chi2_min" not in setup


def test_chi2_thresholds_round_trip(tmp_path, data_file):
    """Web-UI Save Setup must not drop the run controls."""
    from aure.setup import dump_setup, load_setup

    p = _write_yaml(
        tmp_path, "original.yaml", _setup_doc(data_file, chi2_max=2.5, chi2_min=0.4)
    )
    original = load_setup(p)

    text = dump_setup(original)
    assert "chi2_max: 2.5" in text
    assert "chi2_min: 0.4" in text

    round_tripped = load_setup(_write_yaml(tmp_path, "rt.yaml", yaml.safe_load(text)))
    assert round_tripped["chi2_max"] == original["chi2_max"]
    assert round_tripped["chi2_min"] == original["chi2_min"]


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,literal,match",
    [
        ("chi2_max", "tight", "`chi2_max` must be a number"),
        ("chi2_min", "loose", "`chi2_min` must be a number"),
        # `float(True)` is 1.0, so without a bool guard `chi2_min: true` loads as
        # a floor of 1 and `chi2_max: no` as a ceiling of 0 — both silent. YAML
        # 1.1 (what PyYAML implements) parses yes/no as booleans too.
        ("chi2_max", "true", "`chi2_max` must be a number"),
        ("chi2_min", "yes", "`chi2_min` must be a number"),
        ("chi2_min", "no", "`chi2_min` must be a number"),
        # A ceiling ≤ 0 or non-finite can never be met, which would disable the
        # deterministic acceptance stop for the whole run.
        ("chi2_max", "0", "finite positive"),
        ("chi2_max", "-1.5", "finite positive"),
        ("chi2_max", ".nan", "finite positive"),
        ("chi2_max", ".inf", "finite positive"),
        # The floor may be 0 ("no floor") but not negative and not non-finite.
        ("chi2_min", "-0.1", "finite non-negative"),
        ("chi2_min", ".nan", "finite non-negative"),
        ("chi2_min", "-.inf", "finite non-negative"),
    ],
)
def test_invalid_chi2_threshold_rejected(tmp_path, data_file, key, literal, match):
    from aure.config import ConfigError
    from aure.setup import load_setup

    with pytest.raises(ConfigError, match=match):
        load_setup(_write_raw(tmp_path, data_file, key, literal))


@pytest.mark.parametrize("floor", [2.5, 3.0])
def test_chi2_min_at_or_above_chi2_max_rejected(tmp_path, data_file, floor):
    """An empty acceptance window would disable the deterministic stop entirely."""
    from aure.config import ConfigError
    from aure.setup import load_setup

    p = _write_yaml(
        tmp_path, "bad.yaml", _setup_doc(data_file, chi2_max=2.5, chi2_min=floor)
    )
    with pytest.raises(ConfigError) as exc:
        load_setup(p)

    message = str(exc.value)
    # The message has to name both numbers and say why the pair is impossible.
    assert "`chi2_min`" in message and "`chi2_max`" in message
    assert f"{floor:g}" in message and "2.5" in message
    assert "never be satisfied" in message


def test_chi2_min_below_chi2_max_accepted(tmp_path, data_file):
    from aure.setup import load_setup

    p = _write_yaml(
        tmp_path, "ok.yaml", _setup_doc(data_file, chi2_max=2.5, chi2_min=2.49)
    )
    assert load_setup(p)["chi2_min"] == 2.49


def test_chi2_min_alone_is_not_compared_against_the_env(
    tmp_path, data_file, monkeypatch
):
    """A file's validity must not depend on the shell that loads it.

    A setup that pins only the floor is judged on its own; reconciling it against
    an env-supplied ceiling is the run-time resolver's job
    (``evaluation._get_chi2_min``), which disables the floor and warns.
    """
    from aure.setup import load_setup

    monkeypatch.setenv("CHI2_MAX", "1.0")
    p = _write_yaml(tmp_path, "floor_only.yaml", _setup_doc(data_file, chi2_min=3.0))
    assert load_setup(p)["chi2_min"] == 3.0


# ----------------------------------------------------------------------
# Env-var mapping
# ----------------------------------------------------------------------


def test_build_env_overrides_maps_the_chi2_thresholds():
    from aure.cli import _build_env_overrides

    overrides = _build_env_overrides({"chi2_max": 2.5, "chi2_min": 0.4})
    assert overrides["CHI2_MAX"] == "2.5"
    assert overrides["CHI2_MIN"] == "0.4"
    # 0 is a value, not an absence: it must be forwarded as "0" so the run
    # disables the floor instead of inheriting the default.
    assert _build_env_overrides({"chi2_min": 0})["CHI2_MIN"] == "0"


def test_build_env_overrides_omits_absent_keys():
    from aure.cli import _build_env_overrides

    assert set(_build_env_overrides({"chi2_max": 1.0})) == {"CHI2_MAX"}
    # …and the floor alone does not drag the ceiling along.
    assert set(_build_env_overrides({"chi2_min": 0.4})) == {"CHI2_MIN"}


def test_applied_env_overrides_restores_the_previous_environment(monkeypatch):
    """A key that was absent must be unset again, not left as an empty string."""
    from aure.cli import _applied_env_overrides

    monkeypatch.setenv("CHI2_MAX", "9.0")
    monkeypatch.delenv("CHI2_MIN", raising=False)

    with _applied_env_overrides({"chi2_max": 1.25, "chi2_min": 0.2}):
        assert os.environ["CHI2_MAX"] == "1.25"
        assert os.environ["CHI2_MIN"] == "0.2"

    assert os.environ["CHI2_MAX"] == "9.0"
    assert "CHI2_MIN" not in os.environ


def test_applied_env_overrides_restores_on_exception(monkeypatch):
    from aure.cli import _applied_env_overrides

    monkeypatch.delenv("CHI2_MAX", raising=False)
    with pytest.raises(RuntimeError):
        with _applied_env_overrides({"chi2_max": 1.25}):
            raise RuntimeError("boom")
    assert "CHI2_MAX" not in os.environ


def test_applied_env_overrides_visible_to_the_evaluation_getters(monkeypatch):
    """The values the evaluation node reads are the setup's, not the env's."""
    from aure.cli import _applied_env_overrides
    from aure.nodes.evaluation import _get_chi2_max, _get_chi2_min

    monkeypatch.setenv("CHI2_MAX", "5.0")
    monkeypatch.setenv("CHI2_MIN", "0.5")

    with _applied_env_overrides({"chi2_max": 1.75, "chi2_min": 0.2}):
        assert _get_chi2_max() == 1.75
        assert _get_chi2_min() == 0.2

    assert _get_chi2_max() == 5.0
    assert _get_chi2_min() == 0.5


# ----------------------------------------------------------------------
# `aure analyze -c setup.yaml` applies the run controls and reports the backlog
# ----------------------------------------------------------------------

_FOUR_STATUS_BACKLOG = [
    {
        "id": 1,
        "title": "Add native CuOx",
        "change": "insert a CuOx slab",
        "status": "pending",
    },
    {"id": 2, "title": "Free the Ti roughness", "status": "confirmed"},
    {"id": 3, "title": "Split Cu into two slabs", "status": "rejected"},
    {"id": 4, "title": "Tie the Cu roughness", "status": "tried"},
    {"id": 5, "title": "Widen the SLD bounds", "change": "±20%", "status": "pending"},
]


def _run_analyze(tmp_path, data_file, monkeypatch, result_state, args=(), **setup_keys):
    """Invoke `aure analyze -c setup.yaml` against a canned workflow result."""
    from click.testing import CliRunner

    import aure.cli as cli_module
    import aure.workflow as workflow_module

    monkeypatch.setattr(cli_module, "_check_llm_status", lambda **kw: (True, "ok"))
    monkeypatch.setattr(workflow_module, "run_analysis", lambda **kw: result_state)

    p = _write_yaml(tmp_path, "an.yaml", _setup_doc(data_file, **setup_keys))
    return CliRunner().invoke(cli_module.cli, ["analyze", "-c", str(p), *args])


def test_analyze_applies_setup_chi2_max(tmp_path, data_file, monkeypatch):
    """Regression: setup run controls used to be honoured by `batch` only."""
    from click.testing import CliRunner

    import aure.cli as cli_module
    import aure.workflow as workflow_module

    monkeypatch.setenv("CHI2_MAX", "5.0")
    monkeypatch.delenv("CHI2_MIN", raising=False)
    monkeypatch.setattr(cli_module, "_check_llm_status", lambda **kw: (True, "ok"))

    seen: dict = {}

    def fake_run_analysis(**kwargs):
        seen["chi2_max"] = os.environ.get("CHI2_MAX")
        return {"Q": [0.01, 0.02], "structural_hypotheses": _FOUR_STATUS_BACKLOG}

    monkeypatch.setattr(workflow_module, "run_analysis", fake_run_analysis)

    p = _write_yaml(tmp_path, "setup.yaml", _setup_doc(data_file, chi2_max=1.25))
    result = CliRunner().invoke(cli_module.cli, ["analyze", "-c", str(p)])

    assert result.exit_code == 0, result.output
    assert seen["chi2_max"] == "1.25"
    # The banner records the whole acceptance window, floor included (0.5
    # default), and says what the floor does — a χ² under it can still be
    # accepted by the evaluator, so the floor stands the stop down, not vetoes.
    assert "Accept when 0.5 ≤ χ² ≤ 1.25" in result.output
    assert "χ² < 0.5 is judged, not auto-accepted" in result.output
    # And the run must not leak its override into the ambient environment.
    assert os.environ["CHI2_MAX"] == "5.0"


def test_analyze_banner_reports_a_disabled_floor(tmp_path, data_file, monkeypatch):
    monkeypatch.setenv("CHI2_MAX", "5.0")
    monkeypatch.delenv("CHI2_MIN", raising=False)

    result = _run_analyze(tmp_path, data_file, monkeypatch, {"Q": [0.01]}, chi2_min=0)

    assert result.exit_code == 0, result.output
    assert "Accept when χ² ≤ 5  (χ² floor disabled)" in result.output
    assert "≤ χ²" not in result.output


def test_analyze_banner_uses_the_env_window_when_the_setup_omits_it(
    tmp_path, data_file, monkeypatch
):
    monkeypatch.setenv("CHI2_MAX", "3.0")
    monkeypatch.setenv("CHI2_MIN", "0.25")

    result = _run_analyze(tmp_path, data_file, monkeypatch, {"Q": [0.01]})

    assert result.exit_code == 0, result.output
    assert "Accept when 0.25 ≤ χ² ≤ 3" in result.output


def test_report_lists_the_untried_improvements(tmp_path, data_file, monkeypatch):
    """A run that clears the threshold stops with candidates still pending."""
    result = _run_analyze(
        tmp_path,
        data_file,
        monkeypatch,
        {"Q": [0.01], "structural_hypotheses": _FOUR_STATUS_BACKLOG},
    )

    assert result.exit_code == 0, result.output
    assert "Possible further improvements" in result.output
    assert "Add native CuOx" in result.output
    assert "insert a CuOx slab" in result.output
    assert "Widen the SLD bounds" in result.output
    # Attempted entries are not offered again…
    assert "Free the Ti roughness" not in result.output
    # …but they are tallied, and `tried` counts as attempted alongside
    # confirmed/rejected — nor is an inconclusive attempt called a failure.
    assert (
        "(3 of 5 attempted — confirmed (1); rejected (1); tried, inconclusive (1))"
        in result.output
    )
    assert "failed" not in result.output


def test_report_tally_survives_an_untouched_backlog(tmp_path, data_file, monkeypatch):
    result = _run_analyze(
        tmp_path,
        data_file,
        monkeypatch,
        {
            "Q": [0.01],
            "structural_hypotheses": [{"id": 1, "title": "x", "status": "pending"}],
        },
    )

    assert result.exit_code == 0, result.output
    assert "(0 of 1 attempted)" in result.output


def test_report_omits_improvements_when_none_pending(tmp_path, data_file, monkeypatch):
    result = _run_analyze(
        tmp_path,
        data_file,
        monkeypatch,
        {
            "Q": [0.01],
            "structural_hypotheses": [{"id": 1, "title": "x", "status": "rejected"}],
        },
    )

    assert result.exit_code == 0, result.output
    assert "Possible further improvements" not in result.output


def test_analyze_json_carries_the_pending_hypotheses(tmp_path, data_file, monkeypatch):
    """`--json` is the machine surface for "should I keep refining?"."""
    import json

    result = _run_analyze(
        tmp_path,
        data_file,
        monkeypatch,
        {"Q": [0.01], "structural_hypotheses": _FOUR_STATUS_BACKLOG},
        args=("--json",),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["pending_hypotheses"] == [
        {"id": 1, "title": "Add native CuOx", "change": "insert a CuOx slab"},
        {"id": 5, "title": "Widen the SLD bounds", "change": "±20%"},
    ]


def test_analyze_json_pending_hypotheses_is_always_present(
    tmp_path, data_file, monkeypatch
):
    """An empty list, never a missing key — absence would read as "old aure"."""
    import json

    result = _run_analyze(
        tmp_path, data_file, monkeypatch, {"Q": [0.01]}, args=("--json",)
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["pending_hypotheses"] == []


# ----------------------------------------------------------------------
# `aure batch`
# ----------------------------------------------------------------------


def _run_batch(tmp_path, data_file, monkeypatch, result_state, **setup_keys):
    """Invoke `aure batch flat_setup.yaml` against a canned workflow result."""
    from click.testing import CliRunner

    import aure.cli as cli_module
    import aure.workflow as workflow_module

    monkeypatch.setattr(workflow_module, "run_analysis", lambda **kw: result_state)

    doc = _setup_doc(
        data_file, name="j1", output_root=str(tmp_path / "out"), **setup_keys
    )
    p = _write_yaml(tmp_path, "flat.yaml", doc)
    return CliRunner().invoke(cli_module.cli, ["batch", str(p)])


def test_batch_prints_the_pending_hypotheses(tmp_path, data_file, monkeypatch):
    """`batch` never calls the human report, so it must print the backlog itself."""
    result = _run_batch(
        tmp_path,
        data_file,
        monkeypatch,
        {
            "fit_results": [{"chi_squared": 1.4}],
            "structural_hypotheses": _FOUR_STATUS_BACKLOG,
        },
    )

    assert result.exit_code == 0, result.output
    assert "Done – χ² = 1.400" in result.output
    assert "Possible further improvements" in result.output
    assert "[1] Add native CuOx" in result.output
    assert "[5] Widen the SLD bounds" in result.output
    # Compact: titles only, and nothing already attempted.
    assert "insert a CuOx slab" not in result.output
    assert "Split Cu into two slabs" not in result.output


@pytest.mark.parametrize(
    "backlog", [None, [], [{"id": 1, "title": "x", "status": "rejected"}]]
)
def test_batch_stays_quiet_with_an_empty_backlog(
    tmp_path, data_file, monkeypatch, backlog
):
    state: dict = {"fit_results": [{"chi_squared": 1.4}]}
    if backlog is not None:
        state["structural_hypotheses"] = backlog

    result = _run_batch(tmp_path, data_file, monkeypatch, state)

    assert result.exit_code == 0, result.output
    assert "Done – χ² = 1.400" in result.output
    assert "Possible further improvements" not in result.output


def test_final_fit_keys_load_and_round_trip(tmp_path, data_file):
    """The four final-fit keys are documented; the loader must accept them.

    ``aure_config.example.yaml`` and ``docs/finalization.md`` document these and
    ``cli._build_env_overrides`` maps them, but they were missing from
    ``_KNOWN_TOP_LEVEL`` — so following the example file's own documentation was
    rejected as unknown keys and the optional final uncertainty fit was
    unreachable from a setup YAML.
    """
    from aure.setup import dump_setup, load_setup

    p = _write_yaml(
        tmp_path,
        "final.yaml",
        _setup_doc(
            data_file,
            fit_method_final="dream",
            fit_steps_final=10000,
            fit_burn_final=5000,
            final_fit_chi2_max=1.5,
        ),
    )
    setup = load_setup(p)

    assert setup["fit_method_final"] == "dream"
    assert setup["fit_steps_final"] == 10000
    assert setup["fit_burn_final"] == 5000
    assert setup["final_fit_chi2_max"] == 1.5
    assert isinstance(setup["final_fit_chi2_max"], float)

    # Round-trip: the web UI's Save Setup dumps what Load Setup parsed.
    dumped = yaml.safe_load(dump_setup(setup))
    for key in (
        "fit_method_final",
        "fit_steps_final",
        "fit_burn_final",
        "final_fit_chi2_max",
    ):
        assert dumped[key] == setup[key], key


@pytest.mark.parametrize("bad", [0, -1, ".inf", ".nan"])
def test_final_fit_chi2_max_must_be_a_usable_gate(tmp_path, data_file, bad):
    """A gate ≤ 0 or non-finite can never be met, so it silently disables the
    final fit rather than loosening it — the same rule ``chi2_max`` carries."""
    from aure.config import ConfigError
    from aure.setup import load_setup

    p = _write_raw(tmp_path, data_file, "final_fit_chi2_max", str(bad))

    with pytest.raises(ConfigError, match="finite positive number"):
        load_setup(p)
