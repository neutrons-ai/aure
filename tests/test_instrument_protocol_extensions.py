"""The two optional protocol members: ``run_title`` and ``header_issues``.

Both are looked up by name rather than declared on :class:`Instrument`, so an
instrument written against an earlier version of the protocol keeps working
and behaves exactly as instruments did before they existed. These tests pin
that, and pin what each is for.
"""

import pytest

from aure import instruments
from aure.instruments import PARTIAL, UNKNOWN
from aure.instruments.ref_l import REFLv2Instrument
from aure.nodes import intake
from aure.nodes.intake import intake_node
from aure.state import create_initial_state
from tests.test_instrument_autoreduction import RAGGED, ROWS


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """`aure.cli` leaks a real LLM_PROVIDER into the process; see the plan doc."""
    monkeypatch.setattr(intake, "llm_available", lambda: False)


class _Minimal:
    """An instrument from before either member existed."""

    name = "MINIMAL"

    def matches(self, file_path, header=""):
        return file_path.endswith(".minimal")

    def file_role(self, file_path):
        return UNKNOWN

    def group_key(self, file_path):
        return None

    def header_metadata(self, file_path):
        return dict(instruments.DEFAULT_HEADER_METADATA)

    def role_supports_nuisance(self, role):
        return False


class _Raising:
    """One whose optional members are broken."""

    name = "RAISING"

    def matches(self, file_path, header=""):
        return file_path.endswith(".raising")

    def file_role(self, file_path):
        return PARTIAL

    def group_key(self, file_path):
        return None

    def header_metadata(self, file_path):
        return dict(instruments.DEFAULT_HEADER_METADATA)

    def role_supports_nuisance(self, role):
        return role == PARTIAL

    def run_title(self, file_path):
        raise RuntimeError("boom")

    def header_issues(self, file_path):
        raise RuntimeError("boom")


@pytest.fixture
def registered(monkeypatch):
    from aure.instruments import registry

    monkeypatch.setattr(registry, "_REGISTRY", list(registry._REGISTRY))
    instruments.register(_Minimal(), first=True)
    instruments.register(_Raising(), first=True)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_an_instrument_without_them_gets_the_old_behaviour(tmp_path, registered):
    path = tmp_path / "sample.minimal"
    path.write_text("# Run title: CuPt_d8-THF-218386-1.\n" + ROWS)

    assert instruments.resolve(str(path)).name == "MINIMAL"
    assert instruments.run_title(str(path)) == "CuPt_d8-THF-218386-1."
    assert instruments.header_issues(str(path)) == []


def test_the_older_ref_l_dialect_keeps_the_label_match(tmp_path):
    """REF_L declares neither, and its `# Run title:` line is its own title."""
    path = tmp_path / "REFL_218386_1_218386_partial.txt"
    path.write_text("# Run title: CuPt_d8-THF_FullQ-218386-1.\n" + ROWS)

    assert instruments.run_title(str(path)) == "CuPt_d8-THF_FullQ-218386-1."


def test_a_broken_optional_member_does_not_break_the_run(tmp_path, registered):
    """A third-party instrument must not be able to fail an analysis."""
    path = tmp_path / "sample.raising"
    path.write_text("# Run title: still readable-1.\n" + ROWS)

    assert instruments.run_title(str(path)) == "still readable-1."  # fell back
    assert instruments.header_issues(str(path)) == []


# ---------------------------------------------------------------------------
# run_title: the case the generic match gets wrong
# ---------------------------------------------------------------------------


def test_the_generic_match_would_capture_the_whole_json_array(tmp_path):
    """What this exists to stop, stated as the behaviour it replaces."""
    path = tmp_path / "REFL_234277_3_234279_autoreduction.dat"
    path.write_text(RAGGED + ROWS)

    assert instruments.generic_run_title(str(path)).startswith('{"title":')


def test_the_instrument_returns_this_segments_own_title(tmp_path):
    path = tmp_path / "REFL_234277_3_234279_autoreduction.dat"
    path.write_text(RAGGED + ROWS)

    assert instruments.run_title(str(path)) == "Sample1_air-234277-3."


def test_intake_records_the_instruments_title(tmp_path):
    """The value that reaches the checkpoint, and the hypothesis list."""
    path = tmp_path / "REFL_234277_2_234278_autoreduction.dat"
    path.write_text(RAGGED + ROWS)

    assert intake._parse_run_title_from_header(str(path)) == "Sample1_air-234277-2."


def test_a_title_is_capped_however_it_was_read(tmp_path, registered):
    """A pathological header must not dominate a prompt by either route."""

    class _Long(_Minimal):
        name = "LONG"

        def matches(self, file_path, header=""):
            return file_path.endswith(".long")

        def run_title(self, file_path):
            return "x" * 5000

    instruments.register(_Long(), first=True)
    path = tmp_path / "sample.long"
    path.write_text(ROWS)

    assert len(instruments.run_title(str(path))) == instruments.MAX_RUN_TITLE_LEN


# ---------------------------------------------------------------------------
# header_issues: a defect that is survivable, but not silently
# ---------------------------------------------------------------------------


def test_a_clean_header_reports_nothing(tmp_path):
    path = tmp_path / "REFL_234277_1_234277_autoreduction.dat"
    path.write_text(RAGGED + ROWS)

    assert instruments.header_issues(str(path)) == []


def test_a_mismatched_array_is_reported(tmp_path):
    path = tmp_path / "REFL_234277_1_234277_autoreduction.dat"
    broken = RAGGED.replace('"ThetaShift": [0, 0, 0]', '"ThetaShift": [0]')
    path.write_text(broken + ROWS)

    issues = instruments.header_issues(str(path))

    assert len(issues) == 1
    assert "ThetaShift" in issues[0]


def test_an_unreadable_dq_label_is_reported(tmp_path):
    """The 2.355x field. Assuming FWHM quietly is the failure being prevented."""
    path = tmp_path / "REFL_234277_1_234277_autoreduction.dat"
    path.write_text(RAGGED.replace("dQ (sigma)", "dQ (halfwidth)") + ROWS)

    issues = instruments.header_issues(str(path))

    assert any("halfwidth" in i for i in issues)


def test_issues_reach_the_run_as_messages(tmp_path):
    """A log line scrolls past; this is what a scientist reads afterwards."""
    path = str(tmp_path / "REFL_234277_1_234277_autoreduction.dat")
    with open(path, "w") as f:
        f.write(RAGGED.replace("dQ (sigma)", "dQ (halfwidth)") + ROWS)
    state = create_initial_state(
        data_file=path,
        sample_description="",
        states=[{"name": "state0", "data_files": [{"file": path, "label": "seg1"}]}],
    )

    result = intake_node(state)

    warnings = [m for m in result["messages"] if "Header warning" in m["content"]]
    assert warnings, [m["content"] for m in result["messages"]]
    assert "halfwidth" in warnings[0]["content"]
    assert "seg1" in warnings[0]["content"]


def test_issues_are_recorded_on_the_dataset(tmp_path):
    """So the checkpoint carries them, not only the console."""
    path = str(tmp_path / "REFL_234277_1_234277_autoreduction.dat")
    with open(path, "w") as f:
        f.write(RAGGED.replace("dQ (sigma)", "dQ (halfwidth)") + ROWS)
    state = create_initial_state(
        data_file=path,
        sample_description="",
        states=[{"name": "state0", "data_files": [{"file": path}]}],
    )

    result = intake_node(state)

    assert result["states"][0]["data_files"][0]["header_issues"]


def test_a_clean_run_carries_no_warnings(tmp_path):
    """Every real file in the beamtime is clean; noise here would be ignored."""
    path = str(tmp_path / "REFL_234277_1_234277_autoreduction.dat")
    with open(path, "w") as f:
        f.write(RAGGED + ROWS)
    state = create_initial_state(
        data_file=path,
        sample_description="",
        states=[{"name": "state0", "data_files": [{"file": path}]}],
    )

    result = intake_node(state)

    assert not [m for m in result["messages"] if "Header warning" in m["content"]]
    assert "header_issues" not in result["states"][0]["data_files"][0]


# ---------------------------------------------------------------------------
# A state of several unclaimed files
# ---------------------------------------------------------------------------


def test_several_unclaimed_files_are_described_as_such(tmp_path, caplog):
    """One unrecognised file and three used to read the same.

    Three are almost always angle segments of one measurement, and unclaimed
    they become three independent complete curves — no grouping check, no
    per-angle optics, each probe built from Q alone.
    """
    from aure.config import _parse_states

    files = []
    for i in (1, 2, 3):
        p = tmp_path / f"d17_01234{i}_reduced.mft"
        p.write_text("# instrument: D17\n" + ROWS)
        files.append({"file": str(p)})

    with caplog.at_level("WARNING"):
        _parse_states([{"name": "state0", "data_files": files}], base_dir=tmp_path)

    assert "all 3 as separate complete curves" in caplog.text
    assert "angle segments of one measurement" in caplog.text


def test_one_unclaimed_file_keeps_the_single_file_wording(tmp_path, caplog):
    from aure.config import _parse_states

    p = tmp_path / "d17_012345_reduced.mft"
    p.write_text("# instrument: D17\n" + ROWS)

    with caplog.at_level("WARNING"):
        _parse_states(
            [{"name": "state0", "data_files": [{"file": str(p)}]}], base_dir=tmp_path
        )

    assert "treating as a combined curve" in caplog.text


def test_the_autoreduction_instrument_declares_both_members():
    inst = REFLv2Instrument()

    assert callable(getattr(inst, "run_title", None))
    assert callable(getattr(inst, "header_issues", None))
