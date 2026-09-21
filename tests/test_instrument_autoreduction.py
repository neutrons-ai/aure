"""REF_L's ``new_reduction`` dialect — ``_autoreduction.dat``.

The header fixtures are trimmed from the IPTS-37740 beamtime and preserve the
details that matter exactly:

* ``Angles`` and ``Run Title`` are longer than the segment count, while every
  array under ``Config`` stays at the segment count;
* ``Config`` is JSON (``null``, ``false``) while ``DB`` and ``NR_runs`` are
  Python (``None``, single quotes), on different lines of one file;
* the angles are negative, and 1.251 is a measurement rather than the nominal
  1.2;
* the dQ column is **sigma**, where the older reduction writes FWHM.

The over-length arrays are not "a segment measured twice", which is how this
was first read. **The reduction appends to them on reprocess instead of
replacing them** — four of the beamtime's five runs carry whole repeated
``[1,2,3]`` blocks, which no amount of splitting a single segment produces.
See ``docs/plan-new-reduction-format.md``.
"""

import pytest

from aure import instruments
from aure.instruments import PARTIAL
from aure.instruments.ref_l import REFLAutoreductionInstrument

INST = REFLAutoreductionInstrument()

# Run 234277: one full pass over segments 1,2 then a partial reprocess of 2,3,
# giving `[1, 2, 2, 3]`. The only run in the beamtime that is NOT a whole-block
# repeat, and the only one where positional indexing goes wrong — segment 3
# would land on THS[2] = 1.251 instead of 3.5.
RAGGED = (
    "# NR_runs = [None, None, 234279]\n"
    '# Run Title: {"title": ["Sample1_air-234277-1.", "Sample1_air-234277-2.", '
    '"Sample1_air-234277-2.", "Sample1_air-234277-3."]}\n'
    "# DB = ['A1_Si.txt', 'A2_Si.txt', 'A3_Si.txt']\n"
    "# Lambda Range = 2.6500000000000004Å to 9.45Å\n"
    '# Angles: {"THS": [-0.45, -1.251, -1.251, -3.5], '
    '"THI": [-0.0, -0.0, -0.0, -0.0], '
    '"ThCen": [-0.45, -1.251, -1.251, -3.5]}\n'
    '# Config: {"experiment_id": "IPTS-37740", "DBname": ["A1_Si.txt", '
    '"A2_Si.txt", "A3_Si.txt"], "RBnum": [null, null, 234279], '
    '"ThetaShift": [0, 0, 0], "Normalize": false, "qmin": 0.001, '
    '"qmax": 0.5, "dqbin": 0.015}\n'
    "# columns = Q, R, dR, dQ (sigma)\n"
)

# Run 234283: two complete reduction passes, `[1, 2, 3, 1, 2, 3]`. The shape
# that rules out "a segment was measured in two pieces".
DOUBLED = (
    '# Run Title: {"title": ["S1_KHCO3-234283-1.", "S1_KHCO3-234283-2.", '
    '"S1_KHCO3-234283-3.", "S1_KHCO3-234283-1.", "S1_KHCO3-234283-2.", '
    '"S1_KHCO3-234283-3."]}\n'
    '# Angles: {"THS": [-0.45, -1.251, -3.5, -0.45, -1.251, -3.5]}\n'
    '# Config: {"experiment_id": "IPTS-37740", "ThetaShift": [0, 0, 0]}\n'
    "# columns = Q, R, dR, dQ (sigma)\n"
)

ROWS = "0.0100 1.00e+00 1.0e-02 1.0e-04\n0.0200 5.00e-01 1.0e-02 2.0e-04\n"


def write(tmp_path, header, segment=1, subrun=234277, run=234277, name=None):
    path = tmp_path / (name or f"REFL_{run}_{segment}_{subrun}_autoreduction.dat")
    path.write_text(header + ROWS)
    return str(path)


# ---------------------------------------------------------------------------
# Claiming the file
# ---------------------------------------------------------------------------


def test_the_registry_resolves_it_to_this_instrument(tmp_path):
    path = write(tmp_path, RAGGED)

    assert instruments.resolve(path).name == "REF_L_autoreduction"


def test_it_is_claimed_without_reading_the_file(tmp_path):
    """Resolution runs while parsing a setup, before the data need exist."""
    missing = str(tmp_path / "REFL_234277_1_234277_autoreduction.dat")

    assert instruments.resolve_by_name(missing).name == "REF_L_autoreduction"


def test_a_header_claims_a_file_whose_name_does_not(tmp_path):
    path = write(tmp_path, RAGGED, name="exported.txt")

    assert instruments.resolve(path).name == "REF_L_autoreduction"


def test_the_older_reduction_is_untouched(tmp_path):
    """Disjoint patterns: neither dialect may start claiming the other's files."""
    partial = tmp_path / "REFL_226642_1_2001_partial.txt"
    partial.write_text("# DataRun NormRun TwoTheta(deg)\n# 226642 226643 0.9\n" + ROWS)

    assert instruments.resolve(str(partial)).name == "REF_L"
    assert not INST.matches(str(partial), partial.read_text())


# ---------------------------------------------------------------------------
# Role and grouping
# ---------------------------------------------------------------------------


def test_the_dialect_is_always_per_segment(tmp_path):
    """It has no combined form, unlike the reduction it replaces."""
    assert INST.file_role(write(tmp_path, RAGGED)) == PARTIAL


def test_the_group_key_matches_the_older_dialect(tmp_path):
    """So a beamtime mid-migration can co-refine both in one state."""
    from aure.instruments.ref_l import REFLInstrument

    new = write(tmp_path, RAGGED, segment=2, subrun=234278)
    old = str(tmp_path / "REFL_234277_2_234278_partial.txt")

    assert INST.group_key(new) == REFLInstrument().group_key(old) == "234277"


def test_a_state_of_these_files_is_partials_not_combined(tmp_path):
    """Which is what makes theta_offset / sample_broadening available."""
    from aure.config import _detect_kind

    files = [
        {"file": write(tmp_path, RAGGED, segment=i, subrun=234276 + i)}
        for i in (1, 2, 3)
    ]

    assert _detect_kind("state0", files) == "partials"


def test_nuisance_parameters_are_allowed_on_them(tmp_path):
    assert INST.role_supports_nuisance(PARTIAL) is True


# ---------------------------------------------------------------------------
# The angle — the field positional indexing gets wrong
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "segment,subrun,expected", [(1, 234277, 0.45), (2, 234278, 1.251), (3, 234279, 3.5)]
)
def test_the_angle_is_matched_by_title_not_position(
    tmp_path, segment, subrun, expected
):
    """Segment 3 is the one that catches it: THS[2] is 1.251, not 3.5."""
    meta = INST.header_metadata(write(tmp_path, RAGGED, segment, subrun))

    assert meta["theta"] == pytest.approx(expected)


def test_the_angle_is_right_across_a_whole_repeated_block(tmp_path):
    """`[1,2,3,1,2,3]` — where positional indexing happens to be right.

    Both schemes agree here, which is exactly why the bug stayed hidden: only
    the one ragged run in the beamtime distinguished them.
    """
    meta = INST.header_metadata(write(tmp_path, DOUBLED, 3, 234285, run=234283))

    assert meta["theta"] == pytest.approx(3.5)


def test_the_magnitude_is_taken(tmp_path):
    """Stored negative on a back-reflection run; a probe wants the magnitude."""
    assert INST.header_metadata(write(tmp_path, RAGGED))["theta"] > 0


def test_the_measured_angle_is_not_rounded_to_the_nominal_one(tmp_path):
    """1.251 is the measurement. 1.2 is the setting, and is not in the file."""
    meta = INST.header_metadata(write(tmp_path, RAGGED, 2, 234278))

    assert meta["theta"] != pytest.approx(1.2)


def test_the_most_recent_pass_wins_when_two_disagree(tmp_path, caplog):
    """The arrays are appended to, so the *first* entry is the stale one.

    Every duplicate agrees in the beamtime's own data, so nothing is wrong
    today — but taking the first would silently discard a reprocess that
    corrected the angle, which is the whole reason this is read per pass.
    """
    header = (
        '# Run Title: {"title": ["S-234277-1.", "S-234277-1."]}\n'
        '# Angles: {"THS": [-0.45, -0.46]}\n'
        "# columns = Q, R, dR, dQ (sigma)\n"
    )
    with caplog.at_level("WARNING"):
        meta = INST.header_metadata(write(tmp_path, header))

    assert meta["theta"] == pytest.approx(0.46)
    assert "superseded" in caplog.text


def test_agreeing_duplicates_are_not_warned_about(tmp_path, caplog):
    """Every real file in the beamtime has them; warning would be noise."""
    with caplog.at_level("WARNING"):
        INST.header_metadata(write(tmp_path, DOUBLED, 2, 234284, run=234283))

    assert caplog.text == ""


def test_a_file_claimed_by_header_alone_reports_no_angle(tmp_path):
    """No filename, no segment, no slot to look up — and 0.0 says so.

    0.0 is what tells model_builder to build a Q-based probe, which is the
    right degradation: the resolution still comes from the dQ column.
    """
    meta = INST.header_metadata(write(tmp_path, RAGGED, name="exported.txt"))

    assert meta["theta"] == 0.0


# ---------------------------------------------------------------------------
# The dQ convention — the 2.355x field
# ---------------------------------------------------------------------------


def test_the_dq_column_is_read_as_sigma(tmp_path):
    """`(sigma)` in parentheses on a `columns =` line, not `[FWHM]`."""
    assert INST.header_metadata(write(tmp_path, RAGGED))["dq_is_fwhm"] is False


def test_an_fwhm_label_is_honoured_too(tmp_path):
    """The dialect is new; do not hard-code one answer for the whole format."""
    header = RAGGED.replace("dQ (sigma)", "dQ (FWHM)")

    assert INST.header_metadata(write(tmp_path, header))["dq_is_fwhm"] is True


def test_an_unknown_label_is_loud_about_assuming_fwhm(tmp_path, caplog):
    """Silence here is the 2.355x error, so the fallback has to announce itself."""
    header = RAGGED.replace("dQ (sigma)", "dQ (halfwidth)")

    with caplog.at_level("WARNING"):
        meta = INST.header_metadata(write(tmp_path, header))

    assert meta["dq_is_fwhm"] is True
    assert "halfwidth" in caplog.text
    assert "dq_is_fwhm: false" in caplog.text  # names the override


def test_both_fields_are_declared_authoritative(tmp_path):
    """The LLM header parse must not be able to overrule either.

    It is handed the header text and not the filename, so for `theta` it
    cannot succeed even in principle — it has no way to know which segment
    the file is.
    """
    assert set(instruments.authoritative_fields(INST)) == {"dq_is_fwhm", "theta"}


# ---------------------------------------------------------------------------
# Notation, counting, and malformed input
# ---------------------------------------------------------------------------


def test_json_and_python_notation_are_read_on_different_lines(tmp_path):
    """`Config` is JSON; `DB` is a Python list of single-quoted strings.

    Parsing with only one of them half-works, which is the dangerous outcome:
    `Angles` is valid under both, so the angle looks right while everything
    under `Config` quietly goes missing.
    """
    from aure.instruments.ref_l import _autoreduction_fields

    fields = _autoreduction_fields(RAGGED)

    assert fields["Config"]["experiment_id"] == "IPTS-37740"  # JSON line
    assert fields["DB"] == ["A1_Si.txt", "A2_Si.txt", "A3_Si.txt"]  # Python line
    assert fields["Angles"]["THS"][0] == -0.45  # valid as either
    assert isinstance(fields["Lambda Range"], str)  # neither; kept raw


def test_segments_are_counted_distinctly_not_by_array_length(tmp_path):
    """Four title entries, three segments. The array counts reduction passes."""
    assert INST.header_metadata(write(tmp_path, RAGGED))["num_segments"] == 3


def test_mismatched_per_segment_arrays_are_reported(tmp_path, caplog):
    """The early warning for the next mutation of this format.

    The long arrays must be indexed by title and the `Config` ones by segment;
    that only holds while the `Config` ones really are per segment.
    """
    header = RAGGED.replace('"ThetaShift": [0, 0, 0]', '"ThetaShift": [0, 0]')

    with caplog.at_level("WARNING"):
        INST.header_metadata(write(tmp_path, header))

    assert "ThetaShift" in caplog.text
    assert "no longer line up" in caplog.text


def test_a_consistent_header_produces_no_warnings(tmp_path, caplog):
    """Every real file must be silent, or the warnings become noise."""
    with caplog.at_level("WARNING"):
        INST.header_metadata(write(tmp_path, RAGGED, 3, 234279))

    assert caplog.text == ""


def test_a_truncated_header_still_yields_what_it_can(tmp_path):
    """A new dialect will drop keys; that must not make a file unreadable."""
    header = '# Angles: {"THS": [-0.45]}\n# columns = Q, R, dR, dQ (sigma)\n'

    meta = INST.header_metadata(write(tmp_path, header))

    assert meta["dq_is_fwhm"] is False
    assert meta["theta"] == 0.0  # no titles, so no slot
    assert meta["instrument"] == "REF_L_autoreduction"


def test_a_missing_file_is_not_an_error(tmp_path):
    """Instruments are resolved before a run starts; the data may not exist."""
    meta = INST.header_metadata(str(tmp_path / "REFL_1_1_1_autoreduction.dat"))

    assert meta["theta"] == 0.0
    assert meta["dq_is_fwhm"] is True


def test_the_run_title_is_this_segments_own(tmp_path):
    """Not the JSON array, which is what intake's generic regex captures."""
    assert INST.run_title(write(tmp_path, RAGGED, 3, 234279)) == "Sample1_air-234277-3."
