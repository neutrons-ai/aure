"""Declaring a file's header values in the setup, instead of parsing them.

A setup's ``data_files`` entry may carry ``theta`` and ``dq_is_fwhm``. This is
the escape hatch for a format no registered instrument understands yet: the
caller has read the file with a reader that knows its dialect, and AuRE's job
is to use that rather than to guess.

The motivating case is REF_L's ``_autoreduction.dat``, whose header states
``dQ (sigma)`` where the previous reduction wrote FWHM — a factor of 2.355
that a fit absorbs into roughness rather than reporting. nr-workbench had read
both the convention and the per-segment angles out of those headers and had
nowhere to put them, because the parser kept ``file`` and ``label`` and
dropped the rest without a word. See ``docs/plan-new-reduction-format.md``.
"""

import pytest

from aure.config import ConfigError, _parse_states
from aure.setup import dump_setup


@pytest.fixture
def data_file(tmp_path):
    """A file no registered instrument claims — the case this exists for.

    Deliberately not a REF_L name. The motivating format is now a built-in
    (``REF_L_autoreduction``), and a test that used one would be asserting
    about that instrument rather than about the declaration mechanism.
    """
    path = tmp_path / "d17_012345_reduced.mft"
    path.write_text(
        "# instrument: D17\n0.01 1.0 0.01 0.0001\n0.02 0.5 0.01 0.0002\n"
    )
    return path


def _parse(tmp_path, entry):
    return _parse_states(
        [{"name": "state0", "data_files": [entry]}], base_dir=tmp_path
    )


# ---------------------------------------------------------------------------
# Accepted and carried through
# ---------------------------------------------------------------------------


def test_declared_values_survive_parsing(tmp_path, data_file):
    """The whole point: both reach the DatasetInfo the builders read."""
    states = _parse(
        tmp_path, {"file": str(data_file), "theta": 3.5, "dq_is_fwhm": False}
    )

    ds = states[0]["data_files"][0]
    assert ds["theta"] == pytest.approx(3.5)
    assert ds["dq_is_fwhm"] is False


def test_an_entry_declaring_nothing_carries_nothing(tmp_path, data_file):
    """Absence must stay distinguishable from a declared default.

    ``intake._enrich_dataset`` fills these in with ``setdefault``, so a key
    present here means "the header parse is overridden". Defaulting them at
    parse time would silence the parse for every existing setup file.
    """
    states = _parse(tmp_path, {"file": str(data_file)})

    ds = states[0]["data_files"][0]
    assert "theta" not in ds
    assert "dq_is_fwhm" not in ds


def test_a_bare_path_entry_still_works(tmp_path, data_file):
    states = _parse(tmp_path, str(data_file))

    assert states[0]["data_files"][0]["label"] == data_file.stem


def test_zero_theta_is_kept_not_dropped(tmp_path, data_file):
    """0.0 is meaningful: it is what tells model_builder to build a QProbe."""
    states = _parse(tmp_path, {"file": str(data_file), "theta": 0})

    assert states[0]["data_files"][0]["theta"] == pytest.approx(0.0)


def test_declared_values_round_trip_through_dump(tmp_path, data_file):
    """The web UI and `aure batch` must not lose them on a save."""
    states = _parse(
        tmp_path, {"file": str(data_file), "theta": 1.251, "dq_is_fwhm": False}
    )

    text = dump_setup({"states": states})

    assert "theta: 1.251" in text
    assert "dq_is_fwhm: false" in text


# ---------------------------------------------------------------------------
# Refused rather than dropped
# ---------------------------------------------------------------------------


def test_an_unknown_key_is_an_error_not_a_silent_drop(tmp_path, data_file):
    """The failure this whole change is about.

    A `thetas:` typo that parses and vanishes leaves the run using a header
    value the caller believed it had overridden.
    """
    with pytest.raises(ConfigError, match="thetas"):
        _parse(tmp_path, {"file": str(data_file), "thetas": [3.5]})


def test_the_error_names_what_is_accepted(tmp_path, data_file):
    with pytest.raises(ConfigError, match="dq_is_fwhm"):
        _parse(tmp_path, {"file": str(data_file), "dq_convention": "sigma"})


@pytest.mark.parametrize("bad", ["3.5", None, True, [3.5]])
def test_a_non_numeric_theta_is_refused(tmp_path, data_file, bad):
    """Including `True`, which is an int in Python and would fit to 1 degree."""
    with pytest.raises(ConfigError, match="theta"):
        _parse(tmp_path, {"file": str(data_file), "theta": bad})


def test_a_negative_theta_is_refused(tmp_path, data_file):
    """REF_L's new dialect stores signed angles; the probe wants the magnitude.

    Refusing here means the caller resolves the sign, rather than AuRE
    silently taking `abs()` of a value that might be a genuine mistake.
    """
    with pytest.raises(ConfigError, match="non-negative"):
        _parse(tmp_path, {"file": str(data_file), "theta": -3.5})


@pytest.mark.parametrize("bad", ["false", 0, None])
def test_a_non_boolean_dq_is_fwhm_is_refused(tmp_path, data_file, bad):
    """`dq_is_fwhm: "false"` is truthy, and silently means FWHM — 2.355x off."""
    with pytest.raises(ConfigError, match="dq_is_fwhm"):
        _parse(tmp_path, {"file": str(data_file), "dq_is_fwhm": bad})


# ---------------------------------------------------------------------------
# The unrecognised-file warning has to describe what will actually happen
# ---------------------------------------------------------------------------


def test_the_warning_still_offers_the_escape_hatch(tmp_path, data_file, caplog):
    """No instrument, nothing declared: say so, and say what to do about it."""
    with caplog.at_level("WARNING"):
        _parse(tmp_path, {"file": str(data_file)})

    text = caplog.text
    assert "no registered instrument recognises" in text
    assert "dQ as FWHM" in text
    assert "dq_is_fwhm" in text  # names the override


def test_the_warning_does_not_announce_a_default_that_was_overridden(
    tmp_path, data_file, caplog
):
    """The failure mode this whole plan is about, in miniature.

    A warning that says "dQ as FWHM" over a file the setup declared as sigma
    is not merely noise: it sends the reader hunting a resolution bug that is
    not there, which is exactly the kind of message that cost this beamtime.
    """
    with caplog.at_level("WARNING"):
        _parse(tmp_path, {"file": str(data_file), "theta": 3.5, "dq_is_fwhm": False})

    text = caplog.text
    assert "no registered instrument recognises" in text  # still unrecognised
    assert "declared in the setup" in text
    assert "treating as a combined curve" not in text
