"""``aure formats`` and ``/api/instruments/classify``.

The registry was already extensible, documented and tested — but there was no
way to *ask* it anything without starting a run. That is the gap this closes,
and it is the one nr-workbench paid most for on its own side: `nrw.toml`'s
`[conventions]` block looks like the place to configure formats and is read by
nothing, so editing it eliminates the filename as a suspect while changing
nothing. The equivalent mistake here would be a registry with no way to see
what is in it.
"""

import json

import pytest
from click.testing import CliRunner

from aure.cli import cli
from aure.web import create_app
from tests.test_instrument_autoreduction import RAGGED, ROWS


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def autored(tmp_path):
    path = tmp_path / "REFL_234277_3_234279_autoreduction.dat"
    path.write_text(RAGGED + ROWS)
    return str(path)


@pytest.fixture
def unclaimed(tmp_path):
    path = tmp_path / "d17_012345_reduced.mft"
    path.write_text("# instrument: D17\n" + ROWS)
    return str(path)


# ---------------------------------------------------------------------------
# With no arguments: what can this understand?
# ---------------------------------------------------------------------------


def test_it_lists_the_registered_instruments(runner):
    result = runner.invoke(cli, ["formats"])

    assert result.exit_code == 0
    for name in ("REF_L_autoreduction", "REF_L", "ORSO"):
        assert name in result.output


def test_it_says_that_order_is_priority(runner):
    """A user staring at two REF_L entries needs to know which one wins."""
    result = runner.invoke(cli, ["formats"])

    assert "first to claim a file wins" in result.output


def test_it_points_at_both_escape_hatches(runner):
    """The answer to "mine is not here" has to be in the output itself."""
    result = runner.invoke(cli, ["formats"])

    assert "docs/instruments.md" in result.output
    assert "AURE_INSTRUMENT" in result.output


def test_it_names_what_a_format_is_authoritative_about(runner):
    result = runner.invoke(cli, ["formats"])

    assert "authoritative: dq_is_fwhm, theta" in result.output


# ---------------------------------------------------------------------------
# With files: why did it do that?
# ---------------------------------------------------------------------------


def test_it_reports_what_was_read_from_a_file(runner, autored):
    result = runner.invoke(cli, ["formats", autored])

    assert result.exit_code == 0
    assert "REF_L_autoreduction (claimed by filename)" in result.output
    assert "3.5" in result.output  # the angle, title-matched
    assert "1-sigma" in result.output
    assert "stated by the format" in result.output


def test_an_unrecognised_file_says_so_plainly(runner, unclaimed):
    """Not "generic", which reads like a format rather than an absence."""
    result = runner.invoke(cli, ["formats", unclaimed])

    assert "unrecognised" in result.output


def test_a_header_defect_is_shown(runner, tmp_path):
    """The whole point of being able to ask before running."""
    path = tmp_path / "REFL_234277_1_234277_autoreduction.dat"
    path.write_text(RAGGED.replace("dQ (sigma)", "dQ (halfwidth)") + ROWS)

    result = runner.invoke(cli, ["formats", str(path)])

    assert "halfwidth" in result.output


def test_a_missing_file_is_reported_not_an_error(runner, tmp_path):
    """Asking about a path before the data lands is a reasonable thing to do."""
    result = runner.invoke(cli, ["formats", str(tmp_path / "REFL_1_1_1_partial.txt")])

    assert result.exit_code == 0
    assert "file not found" in result.output


def test_json_output_is_machine_readable(runner, autored):
    result = runner.invoke(cli, ["formats", autored, "--json"])

    payload = json.loads(result.output)
    assert [i["name"] for i in payload["instruments"]][0] == "REF_L_autoreduction"
    entry = payload["files"][0]
    assert entry["role"] == "partial"
    assert entry["theta"] == pytest.approx(3.5)
    assert entry["dq_is_fwhm"] is False
    assert entry["group_key"] == "234277"
    assert entry["run_title"] == "Sample1_air-234277-3."


def test_several_files_are_each_reported(runner, autored, unclaimed):
    result = runner.invoke(cli, ["formats", autored, unclaimed, "--json"])

    assert len(json.loads(result.output)["files"]) == 2


# ---------------------------------------------------------------------------
# The same answer, for the browser
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return create_app().test_client()


def test_the_endpoint_classifies_a_file(client, autored):
    r = client.get("/api/instruments/classify", query_string={"path": autored})

    entry = r.get_json()["files"][0]
    assert entry["role"] == "partial"
    assert entry["supports_nuisance"] is True


def test_the_endpoint_is_what_the_setup_tab_needs(client, unclaimed):
    """An unclaimed file must not offer theta_offset / sample_broadening.

    This is the question the Setup tab used to answer with its own copy of the
    REF_L filename regex — a copy the registry could not reach, so a format
    taught to AuRE stayed unknown to the browser.
    """
    r = client.get("/api/instruments/classify", query_string={"path": unclaimed})

    entry = r.get_json()["files"][0]
    assert entry["role"] == "unknown"
    assert entry["supports_nuisance"] is False


def test_the_endpoint_takes_several_paths(client, autored, unclaimed):
    r = client.get(
        "/api/instruments/classify",
        query_string=[("path", autored), ("path", unclaimed)],
    )

    assert len(r.get_json()["files"]) == 2


def test_a_bad_path_does_not_fail_the_request(client):
    """A browser panel must never be the thing that fails a session."""
    r = client.get("/api/instruments/classify", query_string={"path": "/no/such/file"})

    assert r.status_code == 200
    assert r.get_json()["files"][0]["exists"] is False


def test_the_javascript_no_longer_carries_its_own_rule():
    """Pinned because the copy is what made the browser disagree with the server.

    A regex fallback remains for the first render before the request returns;
    what must not come back is a second source of truth.
    """
    from pathlib import Path

    import aure.web as web

    js = (Path(web.__file__).parent / "static" / "setup.js").read_text()

    assert "/api/instruments/classify" in js
    assert "_PARTIAL_FILE_RE_FALLBACK" in js  # named as a fallback, not the rule
    assert "const _PARTIAL_FILE_RE =" not in js
