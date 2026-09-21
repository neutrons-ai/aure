"""ORNL REF_L (SNS Liquids Reflectometer).

**Two reductions are live**, and they are not two spellings of one format.
:class:`REFLInstrument` reads the established ``_partial.txt`` /
``_combined_data_auto.txt`` files; :class:`REFLAutoreductionInstrument` reads
the ``new_reduction`` pipeline's ``_autoreduction.dat``. They share only their
data columns and their run-numbering, so they share only ``group_key`` here —
a beamtime mid-migration can co-refine both in one state. Everything else
differs, most consequentially the dQ column, which is a FWHM in the first and
one sigma in the second. See ``docs/plan-new-reduction-format.md``.

The regexes below were previously duplicated in :mod:`aure.config`,
:mod:`aure.nodes.intake` and :mod:`aure.refl1d_import`. They are reproduced
here verbatim so that consolidating them changed no classification; the
golden table in ``tests/test_instruments.py`` pins that.

Two families of pattern, and the distinction matters:

* the **loose** patterns (:data:`_PARTIAL_RE`, :data:`_COMBINED_RE`) decide the
  file's *role*. They do not require the ``REFL_`` prefix, because
  ``config._detect_kind`` never did — a file named ``x_1_2_partial.txt`` has
  always been treated as a partial.
* the **strict** patterns additionally capture the set id, and do require the
  prefix. A file can therefore have a role but no group key, which is why
  :meth:`REFLInstrument.group_key` may return ``None`` for a file this
  instrument claims.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from .base import COMBINED, DEFAULT_HEADER_METADATA, PARTIAL, UNKNOWN, read_file_header

logger = logging.getLogger(__name__)

# Loose — role only. Kept deliberately permissive; intake re-validates from
# the actual file headers.
_PARTIAL_RE = re.compile(r"_(\d+)_(\d+)_partial\.txt$", re.IGNORECASE)
_COMBINED_RE = re.compile(r"_combined_data_auto\.txt$", re.IGNORECASE)

# Strict — role plus set id.
_PARTIAL_SETID_RE = re.compile(r"REFL_(\d+)_\d+_\d+_partial\.txt$", re.IGNORECASE)
_COMBINED_SETID_RE = re.compile(r"REFL_(\d+)_combined_data_auto\.txt$", re.IGNORECASE)

#: A REF_L header carries a per-segment metadata table with this column. It is
#: the marker :func:`_parse_theta_from_header` needs, so a file carrying it is
#: a REF_L file whatever it is called.
_HEADER_MARKER = "TwoTheta"


def _parse_theta_from_header(file_path: str) -> float:
    """Extract the incident angle from a REF_L header, or ``0.0``.

    Looks for the metadata table containing ``TwoTheta(deg)``. For a
    single-segment file that table has exactly one data row and theta is half
    of TwoTheta. A combined file has several rows and therefore no single
    incident angle, so it yields ``0.0`` — which is the signal
    ``model_builder`` uses to build a Q-based probe instead of an angle-based
    one.

    Moved verbatim from ``nodes.intake``.
    """
    header = read_file_header(file_path)
    if not header:
        return 0.0

    lines = header.split("\n")
    col_idx = -1
    header_line_idx = -1
    for i, line in enumerate(lines):
        if _HEADER_MARKER in line and line.startswith("#"):
            cols = line.lstrip("# ").split()
            for j, col in enumerate(cols):
                if col.startswith(_HEADER_MARKER):
                    col_idx = j
                    header_line_idx = i
                    break
            break

    if col_idx < 0:
        return 0.0

    data_rows = []
    for line in lines[header_line_idx + 1 :]:
        if not line.startswith("#"):
            break
        parts = line.lstrip("# ").split()
        if len(parts) > col_idx:
            try:
                float(parts[col_idx])
                data_rows.append(parts)
            except ValueError:
                continue

    if len(data_rows) != 1:
        # Combined file (multiple segments) or no data — no single theta.
        return 0.0

    try:
        two_theta = float(data_rows[0][col_idx])
        return two_theta / 2.0
    except (ValueError, IndexError):
        return 0.0


class REFLInstrument:
    """The ORNL REF_L conventions, as AuRE has always applied them."""

    name = "REF_L"

    def matches(self, file_path: str, header: str = "") -> bool:
        name = os.path.basename(file_path)
        if _PARTIAL_RE.search(name) or _COMBINED_RE.search(name):
            return True
        # A REF_L header identifies the file even when the name does not.
        # Before the registry, the theta heuristic ran on every file
        # regardless of name, so claiming these keeps that behaviour.
        return bool(header) and _HEADER_MARKER in header

    def file_role(self, file_path: str) -> str:
        name = os.path.basename(file_path)
        if _PARTIAL_RE.search(name):
            return PARTIAL
        if _COMBINED_RE.search(name):
            return COMBINED
        # Claimed by header alone: a single-segment header is one angle, a
        # multi-segment header is a combined curve.
        if _parse_theta_from_header(file_path) > 0:
            return PARTIAL
        return UNKNOWN

    def group_key(self, file_path: str) -> Optional[str]:
        name = os.path.basename(file_path)
        for pattern in (_COMBINED_SETID_RE, _PARTIAL_SETID_RE):
            m = pattern.search(name)
            if m:
                return m.group(1)
        return None

    def header_metadata(self, file_path: str) -> dict:
        meta = dict(DEFAULT_HEADER_METADATA)
        meta["theta"] = _parse_theta_from_header(file_path)
        meta["instrument"] = self.name
        return meta

    def role_supports_nuisance(self, role: str) -> bool:
        # theta_offset / sample_broadening describe one angle's optics, so
        # they are only meaningful on a partial.
        return role == PARTIAL


# ---------------------------------------------------------------------------
# The ``new_reduction`` dialect — ``_autoreduction.dat``
# ---------------------------------------------------------------------------

#: A file of the ``new_reduction`` dialect, with its run, segment and subrun.
#:
#: The segment must come from the *filename*: this header describes the whole
#: run and is byte-identical in every one of that run's files, so it does not
#: say which segment it is attached to.
_AUTORED_NAME_RE = re.compile(
    r"REFL_(?P<run>\d+)_(?P<seg>\d+)_(?P<subrun>\d+)_autoreduction\.dat$",
    re.IGNORECASE,
)

#: Loose — role only, mirroring :data:`_PARTIAL_RE`'s tolerance of a file that
#: does not carry the ``REFL_`` prefix.
_AUTORED_LOOSE_RE = re.compile(r"_autoreduction\.dat$", re.IGNORECASE)

#: Lines whose presence identifies the dialect whatever the file is called.
_AUTORED_MARKERS = ("# Angles:", "# Config:")

#: ``# Key = value`` and ``# Key: value``, this header's shape.
_AUTORED_KV_RE = re.compile(
    r"^#\s*(?P<key>[A-Za-z][A-Za-z0-9 _]*?)\s*[:=]\s*(?P<value>\S.*)$"
)

#: The dQ convention, stated on the ``columns`` line:
#:
#:     # columns = Q, R, dR, dQ (sigma)
#:
#: This is the only place on disk that says whether the fourth column is a
#: full width or a standard deviation, and the two differ by 2.355 — a factor
#: that broadens or sharpens every fringe and that a fit absorbs into
#: roughness rather than reporting.
_AUTORED_DQ_RE = re.compile(
    r"^#\s*columns\s*=.*\bdQ\b\s*\(\s*(?P<label>[^)]*?)\s*\)", re.IGNORECASE
)

#: Column labels meaning a full width, and one standard deviation.
_FWHM_LABELS = frozenset({"fwhm", "full width", "full width at half maximum"})
_SIGMA_LABELS = frozenset(
    {"sigma", "σ", "1-sigma", "1 sigma", "one sigma", "std", "stdev", "std dev"}
)

#: The trailing segment index in a run title, e.g. ``Sample1_air-234277-2.``
#:
#: This is how a file finds its own entry in the header's parallel arrays.
_TITLE_SEGMENT_RE = re.compile(r"-(?P<seg>\d+)\.?\s*$")

#: Keys under ``Config`` that carry one entry per segment. Their length is a
#: cross-check on the segment count; see :func:`_check_array_lengths`.
_PER_SEGMENT_CONFIG_KEYS = ("DBname", "RBnum", "ThetaShift", "method_per_run")


def _autoreduction_fields(header: str) -> dict:
    """Parse the ``# Key = value`` lines of a ``new_reduction`` header.

    **The writer mixes two notations, line by line**, so both are tried::

        # Config: {"a": null, ...}       JSON — `null` is not Python
        # NR_runs = [None, None, 3]      Python — `None` is not JSON
        # DB = ['A1_Si.txt']             Python — single quotes are not JSON
        # Angles: {"THS": [-0.45]}       either, being only numbers
        # Lambda Range = 2.65Å to 9.45Å  neither; kept as its raw string

    Parsing with only one of them looks like it works, which is the dangerous
    outcome: ``Angles`` succeeds under both, so the angle — the field most
    likely to be spot-checked — comes out right while ``Config`` silently
    degrades to a string and every field under it goes missing.

    :func:`ast.literal_eval` executes nothing, so neither path evaluates code
    out of a data file.
    """
    import ast
    import json

    fields: dict = {}
    for line in header.split("\n"):
        match = _AUTORED_KV_RE.match(line)
        if match is None:
            continue
        raw = match.group("value").strip()
        value = raw
        try:
            value = json.loads(raw)
        except ValueError:
            try:
                value = ast.literal_eval(raw)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                value = raw
        fields[match.group("key").strip()] = value
    return fields


def _segment_of(file_path: str) -> Optional[int]:
    """This file's 1-based segment number, from its name; ``None`` if unnamed."""
    match = _AUTORED_NAME_RE.search(os.path.basename(file_path))
    return int(match.group("seg")) if match else None


def _title_slots(fields: dict, segment: int) -> list:
    """Every slot in the parallel arrays belonging to *segment*.

    Positional indexing (``array[segment - 1]``) is wrong here, and wrong in a
    way that fits: on run 234277 it gives segment 3 an angle of 1.251 deg
    instead of 3.5 — a factor of ~2.8 in Q, which converges cleanly to a wrong
    thickness rather than failing. The run titles end in ``-<segment>.``, so
    they are what ties a slot to a segment.

    Returns *all* matching slots rather than one, because there are routinely
    several: the reduction **appends to these arrays on reprocess instead of
    replacing them**, so a run reduced twice carries two complete passes
    (``[1,2,3,1,2,3]``) and a partially re-run one carries a ragged mixture
    (``[1,2,2,3]``). The caller decides which to believe and whether they
    agree.
    """
    titles = fields.get("Run Title")
    titles = titles.get("title") if isinstance(titles, dict) else None
    if not isinstance(titles, list):
        return []
    slots = []
    for index, title in enumerate(titles):
        match = _TITLE_SEGMENT_RE.search(str(title))
        if match and int(match.group("seg")) == segment:
            slots.append(index)
    return slots


def _check_array_lengths(fields: dict, file_path: str) -> None:
    """Warn when the header's two array families disagree about the segments.

    The long arrays (``Run Title``, ``Angles``) are per *acquisition* and grow
    on every reprocess; the arrays under ``Config`` are built from the
    reduction template and are per *segment*. Nothing in the file reconciles
    them, so the two must be indexed by different rules — and that only works
    while the ``Config`` ones really do have one entry per segment.

    Checking it is cheap and silent on every file seen so far. It is the early
    warning for the next mutation of this format, which would otherwise
    surface as a quietly mis-assigned direct beam.
    """
    titles = fields.get("Run Title")
    titles = titles.get("title") if isinstance(titles, dict) else None
    if not isinstance(titles, list):
        return
    segments = set()
    for title in titles:
        match = _TITLE_SEGMENT_RE.search(str(title))
        if match:
            segments.add(int(match.group("seg")))
    if not segments:
        return
    config = fields.get("Config")
    config = config if isinstance(config, dict) else {}
    for key in _PER_SEGMENT_CONFIG_KEYS:
        series = config.get(key)
        if isinstance(series, list) and len(series) != len(segments):
            logger.warning(
                "[REF_L] %s: header names %d segment(s) but Config.%s has %d "
                "entr(y/ies) — the per-segment arrays no longer line up with "
                "the measurement; treat anything read from them as suspect",
                os.path.basename(file_path),
                len(segments),
                key,
                len(series),
            )


def _autoreduction_theta(fields: dict, file_path: str) -> float:
    """The incident angle in degrees, or ``0.0`` when it cannot be resolved.

    Takes the **last** slot naming this segment. Because the arrays are
    appended to on reprocess, the first slot is the *oldest* reduction — stale
    by construction, and silently so if a reprocess corrected ``ThetaShift``
    or switched ``useCalcTheta``. Every slot for the segment is compared, and
    a disagreement is reported rather than resolved quietly.
    """
    segment = _segment_of(file_path)
    if segment is None:
        # Claimed by its header alone: no segment, so no slot to look up.
        # 0.0 is the honest answer and builds a Q-based probe.
        return 0.0

    angles = fields.get("Angles")
    angles = angles if isinstance(angles, dict) else {}
    # THS is the sample angle and carries the setting; ThCen repeats it. Both
    # are signed — negative on a back-reflection run — and a probe wants the
    # magnitude.
    series = angles.get("THS") or angles.get("ThCen")
    if not isinstance(series, list):
        return 0.0

    values = []
    for slot in _title_slots(fields, segment):
        if slot < len(series) and isinstance(series[slot], (int, float)):
            values.append(abs(float(series[slot])))
    if not values:
        return 0.0

    if len(set(values)) > 1:
        logger.warning(
            "[REF_L] %s: segment %d is recorded at %s deg by different "
            "reduction passes; using the most recent (%s). The reduction "
            "appends to these arrays rather than replacing them, so the "
            "earlier value is superseded, not an alternative",
            os.path.basename(file_path),
            segment,
            ", ".join(f"{v:g}" for v in values),
            f"{values[-1]:g}",
        )
    return values[-1]


def _autoreduction_dq_is_fwhm(header: str, file_path: str) -> bool:
    """Whether the fourth column is a full width, per the ``columns`` line.

    Falls back to :data:`DEFAULT_HEADER_METADATA`'s ``True`` when the line is
    absent or its label is one this does not know, and says so — the fallback
    is the convention the *other* reduction uses, so being wrong here is the
    2.355x error this instrument exists to prevent.
    """
    for line in header.split("\n"):
        match = _AUTORED_DQ_RE.match(line)
        if match is None:
            continue
        label = match.group("label").strip().lower()
        if label in _SIGMA_LABELS:
            return False
        if label in _FWHM_LABELS:
            return True
        logger.warning(
            "[REF_L] %s: unrecognised dQ column label %r; assuming FWHM. If "
            "it means one sigma the resolution is 2.355x too narrow — declare "
            "`dq_is_fwhm: false` on this file in the setup (docs/instruments.md)",
            os.path.basename(file_path),
            match.group("label"),
        )
        return DEFAULT_HEADER_METADATA["dq_is_fwhm"]

    logger.warning(
        "[REF_L] %s: no `# columns = ... dQ (...)` line; assuming FWHM",
        os.path.basename(file_path),
    )
    return DEFAULT_HEADER_METADATA["dq_is_fwhm"]


class REFLAutoreductionInstrument:
    """REF_L's ``new_reduction`` output.

    Files are named ``REFL_<run>_<segment>_<subrun>_autoreduction.dat``.

    Registered ahead of :class:`REFLInstrument`, though the two cannot collide:
    their filename patterns are disjoint and neither header marker appears in
    the other's files. The order records which is more specific, not a
    conflict.
    """

    name = "REF_L_autoreduction"

    #: This format *states* both. ``dq_is_fwhm`` is written on the ``columns``
    #: line, and leaving it to be inferred is the error that made this
    #: instrument necessary. ``theta`` is stronger still: the LLM header parse
    #: is given the header text and **not the filename**, and in this dialect
    #: the angle lives in a run-wide array that can only be indexed by the
    #: segment number the filename carries — so that parse cannot succeed even
    #: in principle, and must not be allowed to supply a plausible wrong
    #: answer.
    authoritative_fields = ("dq_is_fwhm", "theta")

    def matches(self, file_path: str, header: str = "") -> bool:
        if _AUTORED_LOOSE_RE.search(os.path.basename(file_path)):
            return True
        return bool(header) and any(m in header for m in _AUTORED_MARKERS)

    def file_role(self, file_path: str) -> str:
        """Always :data:`PARTIAL`: the dialect is only written per segment.

        Unlike the older reduction it has no combined form, so there is no
        second case to distinguish. A file claimed by its header alone still
        describes one segment — it just cannot say which.
        """
        return PARTIAL

    def group_key(self, file_path: str) -> Optional[str]:
        """The run number, matching :meth:`REFLInstrument.group_key`.

        Deliberately the same scheme as the older dialect, so a state may mix
        the two during a migration and still be recognised as one measurement
        set.
        """
        match = _AUTORED_NAME_RE.search(os.path.basename(file_path))
        return match.group("run") if match else None

    def header_metadata(self, file_path: str) -> dict:
        meta = dict(DEFAULT_HEADER_METADATA)
        meta["instrument"] = self.name
        header = read_file_header(file_path)
        if not header:
            return meta

        fields = _autoreduction_fields(header)
        _check_array_lengths(fields, file_path)
        meta["dq_is_fwhm"] = _autoreduction_dq_is_fwhm(header, file_path)
        meta["theta"] = _autoreduction_theta(fields, file_path)

        # The number of distinct segments the run produced — not the length of
        # the title array, which counts reduction passes as well.
        titles = fields.get("Run Title")
        titles = titles.get("title") if isinstance(titles, dict) else None
        if isinstance(titles, list):
            segments = {
                int(m.group("seg"))
                for m in (_TITLE_SEGMENT_RE.search(str(t)) for t in titles)
                if m
            }
            meta["num_segments"] = len(segments)
        return meta

    def run_title(self, file_path: str) -> str:
        """This segment's own title, not the JSON array that holds it.

        ``intake``'s generic regex matches the ``# Run Title:`` line and takes
        the rest of it, which here is the whole array — recorded identically
        in every file of the run, so nothing downstream can tell that it is
        not a title. Consumed once the protocol carries this method (phase 2);
        harmless until then.
        """
        segment = _segment_of(file_path)
        if segment is None:
            return ""
        fields = _autoreduction_fields(read_file_header(file_path))
        titles = fields.get("Run Title")
        titles = titles.get("title") if isinstance(titles, dict) else None
        if not isinstance(titles, list):
            return ""
        slots = _title_slots(fields, segment)
        return str(titles[slots[-1]]) if slots else ""

    def role_supports_nuisance(self, role: str) -> bool:
        return role == PARTIAL
