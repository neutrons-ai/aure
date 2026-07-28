"""Tests that refl1d's SLD profile is found whatever the FitProblem is named.

`_name_problem` names the problem after `model_name`, so refl1d writes
`<model_name>-1-profile.dat`. Looking only for `problem-1-profile.dat` found
nothing on every named run, leaving `sld_z`/`sld_rho` empty — which silently
disabled the SLD-profile artifact detector, since evaluation returns early
without a profile.
"""

from __future__ import annotations

from aure.nodes.fitting import _find_profile_dat, _read_profile_dat

PROFILE = """#      z (A) rho (1e-6/A2) irho (1e-6/A2)
-1.00000000   6.18380853   0.00000000
0.00000000    2.07000000   0.00000000
1.00000000    3.40000000   0.00000000
"""


def _write(tmp_path, name):
    p = tmp_path / name
    p.write_text(PROFILE)
    return p


def test_finds_the_documented_default_name(tmp_path):
    _write(tmp_path, "problem-1-profile.dat")
    assert _find_profile_dat(str(tmp_path)).name == "problem-1-profile.dat"


def test_finds_a_model_named_profile(tmp_path):
    """The real-world case: the problem was named, so the basename differs."""
    _write(tmp_path, "aure_output-1-profile.dat")
    assert _find_profile_dat(str(tmp_path)).name == "aure_output-1-profile.dat"


def test_prefers_the_default_when_both_exist(tmp_path):
    _write(tmp_path, "aure_output-1-profile.dat")
    _write(tmp_path, "problem-1-profile.dat")
    assert _find_profile_dat(str(tmp_path)).name == "problem-1-profile.dat"


def test_returns_none_when_absent(tmp_path):
    (tmp_path / "aure_output-1-refl.dat").write_text("# not a profile\n")
    assert _find_profile_dat(str(tmp_path)) is None


def test_read_profile_dat_reads_a_named_file(tmp_path):
    _write(tmp_path, "cu_film_201136-1-profile.dat")
    z, rho = _read_profile_dat(str(tmp_path))
    assert z == [-1.0, 0.0, 1.0]
    assert rho == [6.18380853, 2.07, 3.4]


def test_read_profile_dat_tolerates_no_export_dir():
    assert _read_profile_dat(None) == (None, None)


def test_read_profile_dat_tolerates_a_missing_profile(tmp_path):
    assert _read_profile_dat(str(tmp_path)) == (None, None)


def test_this_fits_own_export_beats_a_stale_one(tmp_path):
    """A stale export must not shadow the file the current fit just wrote. The old
    order was the fixed default then `sorted(glob)[0]`, so `problem-1-profile.dat`
    from an earlier unnamed run always won — and the artifact detector, which is the
    clamp's only safety valve, silently judged the wrong profile."""
    _write(tmp_path, "problem-1-profile.dat")  # stale, and alphabetically first
    _write(tmp_path, "cu_d2o-1-profile.dat")  # what this fit wrote

    assert _find_profile_dat(str(tmp_path), "cu_d2o").name == "cu_d2o-1-profile.dat"
    # Unnamed problem: the default *is* this fit's export, so it still wins.
    assert _find_profile_dat(str(tmp_path)).name == "problem-1-profile.dat"


def test_freshest_wins_when_no_name_identifies_a_file(tmp_path):
    """Neither the named nor the default file is present, so mtime is the only
    honest signal left — alphabetical order carried no information about which fit
    produced which file."""
    import os

    old = _write(tmp_path, "aaa-1-profile.dat")
    new = _write(tmp_path, "zzz-1-profile.dat")
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))

    assert _find_profile_dat(str(tmp_path), "absent").name == "zzz-1-profile.dat"
