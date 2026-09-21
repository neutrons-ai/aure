"""The instrument/format seam.

AuRE's file handling used to hard-code the ORNL REF_L conventions in three
places (``config``, ``nodes.intake`` and ``refl1d_import``). This package
holds that knowledge in one place behind a small protocol so a facility can
add its own instrument or file format without patching AuRE.

An :class:`Instrument` answers four questions about a data file:

``file_role``
    Is this one complete curve (``COMBINED``), one angle/segment of a
    measurement that must be co-refined with its siblings (``PARTIAL``), or
    can we not tell (``UNKNOWN``)?
``group_key``
    Which measurement set does this file belong to? Files sharing a key were
    measured together; ``None`` means the instrument encodes no grouping.
``header_metadata``
    The deterministic reading of the file header — incident angle, dQ
    convention, segment count. This is the *fallback* for the LLM-based
    header parse in :mod:`aure.nodes.intake`, and the only source when no
    LLM is configured.
``role_supports_nuisance``
    May a state built from files of this role carry per-file resolution
    parameters (``theta_offset`` / ``sample_broadening``)? Those describe one
    angle's optics, so they are meaningless on a pre-combined curve.

Roles are strings rather than an enum so they round-trip through the setup
YAML and the checkpoints unchanged.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

#: One complete curve — a full Q range in a single file.
COMBINED = "combined"
#: One angle / Q segment of a measurement set; needs its siblings.
PARTIAL = "partial"
#: The instrument cannot classify this file.
UNKNOWN = "unknown"

#: Read at most this many lines when sniffing a header. Matches the limit
#: ``nodes.intake`` has always used, so header-driven decisions see the same
#: text whichever path reaches them.
MAX_HEADER_LINES = 40


def read_file_header(file_path: str) -> str:
    """Return the first :data:`MAX_HEADER_LINES` lines of *file_path*.

    Never raises: an unreadable or missing file yields ``""``, which every
    caller must already treat as "no metadata available". Instrument
    resolution runs on paths that may not exist yet (a setup file is parsed
    before the run starts), so a silent empty string is the correct answer
    rather than an error.
    """
    lines: list[str] = []
    try:
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                if i >= MAX_HEADER_LINES:
                    break
                lines.append(line.rstrip("\n"))
    except Exception:
        pass
    return "\n".join(lines)


#: The metadata shape :meth:`Instrument.header_metadata` returns. ``dq_is_fwhm``
#: defaults to True because that is what AuRE has always assumed; an instrument
#: whose files declare a 1-sigma dQ must say so explicitly.
DEFAULT_HEADER_METADATA = {
    "dq_is_fwhm": True,
    "num_segments": 0,
    "theta": 0.0,
    "instrument": None,
}


@runtime_checkable
class Instrument(Protocol):
    """What AuRE needs to know about one instrument or file format.

    Implement this and register it (see :mod:`aure.instruments.registry`) to
    teach AuRE a new instrument. Every method must tolerate a path that does
    not exist — resolution happens during config parsing, before any run.
    """

    #: Short identifier, e.g. ``"REF_L"``. Recorded in checkpoints.
    name: str

    def matches(self, file_path: str, header: str = "") -> bool:
        """Whether this instrument recognises the file.

        *header* is the first lines of the file when available, ``""``
        otherwise. Match on the filename alone where possible so that
        classification does not depend on the file being readable.
        """
        ...

    def file_role(self, file_path: str) -> str:
        """One of :data:`COMBINED`, :data:`PARTIAL`, :data:`UNKNOWN`."""
        ...

    def group_key(self, file_path: str) -> Optional[str]:
        """Identity of the measurement set, or ``None`` if not encoded."""
        ...

    def header_metadata(self, file_path: str) -> dict:
        """Deterministic header reading; see :data:`DEFAULT_HEADER_METADATA`."""
        ...

    def role_supports_nuisance(self, role: str) -> bool:
        """Whether *role* may carry per-file resolution parameters."""
        ...

    # Optional, and looked up by name rather than required here so that an
    # instrument written against an earlier protocol keeps working:
    #
    #   authoritative_fields : tuple  — fields this format *defines*, which
    #       outrank the LLM header parse. See above.
    #   run_title(file_path) -> str  — this file's own run title, when the
    #       generic label match would find the wrong thing.
    #   header_issues(file_path) -> list[str]  — defects in what the header
    #       says, surfaced to the run as warnings.


#: Optional attribute name. An instrument may set ``authoritative_fields`` to
#: the :data:`DEFAULT_HEADER_METADATA` keys its format *defines* rather than
#: merely hints at. Those win over the LLM header parse in
#: :func:`aure.nodes.intake.parse_file_header`, because a standard that
#: specifies its own resolution convention is not something to guess about.
#: Absent or empty means "the LLM's reading wins", which is how AuRE has
#: always behaved.
AUTHORITATIVE_FIELDS_ATTR = "authoritative_fields"


def authoritative_fields(instrument) -> tuple:
    """The fields *instrument* is authoritative about; ``()`` if it says none."""
    return tuple(getattr(instrument, AUTHORITATIVE_FIELDS_ATTR, ()) or ())


# ---------------------------------------------------------------------------
# Optional extension points
#
# Both are looked up by name rather than declared on :class:`Instrument`, for
# the same reason ``authoritative_fields`` is: an instrument written against
# an earlier version of this protocol must keep working. An instrument that
# implements neither behaves exactly as instruments did before they existed.
# ---------------------------------------------------------------------------

#: Header label spellings that carry the operator's free-form run title, e.g.
#: ``# Run title: CuPt_d8-THF_FullQ-218386-1.``. Comment lines only, case
#: insensitively. Lived in ``nodes.intake`` until a format arrived whose
#: ``# Run Title:`` line holds a JSON array of every segment's title rather
#: than this file's own — matching it there captured the array as the title.
_RUN_TITLE_RE = re.compile(r"^#\s*(?:run\s+)?title\s*:\s*(.+?)\s*$", re.IGNORECASE)

#: Longest run title retained. A pathological header line must not be able to
#: dominate a downstream prompt.
MAX_RUN_TITLE_LEN = 200

#: Optional method name: ``run_title(file_path) -> str``.
RUN_TITLE_ATTR = "run_title"

#: Optional method name: ``header_issues(file_path) -> list[str]``.
HEADER_ISSUES_ATTR = "header_issues"


def generic_run_title(file_path: str) -> str:
    """The free-form run title from a header, by the label-matching rule.

    The fallback for an instrument that does not extract its own, and the
    behaviour AuRE has always had. Returns ``""`` when no title line is
    present, which is the common case outside REF_L.

    The value is kept verbatim apart from surrounding whitespace (trailing
    punctuation included) because it is provenance, not data — normalizing it
    would make the checkpoint disagree with the file. The first matching
    comment line wins.
    """
    header = read_file_header(file_path)
    if not header:
        return ""
    for line in header.split("\n"):
        if not line.strip():
            continue
        if not line.startswith("#"):
            break  # reached the data block; no title in the header
        m = _RUN_TITLE_RE.match(line)
        if m:
            return m.group(1)[:MAX_RUN_TITLE_LEN]
    return ""


def run_title(instrument, file_path: str) -> str:
    """*instrument*'s reading of the run title, or the generic one.

    A format whose title line does not hold this file's own title has to say
    so, because nothing downstream can tell a wrong title from a right one:
    the value is free text, and in a whole-run header it is identical in every
    file, so even a cross-file comparison finds no disagreement to report.
    """
    reader = getattr(instrument, RUN_TITLE_ATTR, None)
    if callable(reader):
        try:
            return str(reader(file_path) or "")[:MAX_RUN_TITLE_LEN]
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "[INSTRUMENTS] %s.run_title raised on %s: %s",
                getattr(instrument, "name", instrument),
                file_path,
                e,
            )
    return generic_run_title(file_path)


def header_issues(instrument, file_path: str) -> List[str]:
    """Problems *instrument* found in this file's header; ``[]`` if none.

    For what a header *says* that is wrong or self-contradictory — arrays that
    no longer line up, a resolution convention stated in terms this cannot
    read. Distinct from a file being unreadable, which every method already
    handles by returning defaults.

    These reach the run as warnings rather than errors. A header defect is
    frequently survivable, and refusing to load a file over one would be worse
    than proceeding with a stated caveat; what must not happen is proceeding
    *silently*, which is how a reduction convention change reached a fit
    unremarked in the first place.
    """
    reporter = getattr(instrument, HEADER_ISSUES_ATTR, None)
    if not callable(reporter):
        return []
    try:
        return [str(i) for i in (reporter(file_path) or [])]
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "[INSTRUMENTS] %s.header_issues raised on %s: %s",
            getattr(instrument, "name", instrument),
            file_path,
            e,
        )
        return []


class GenericInstrument:
    """Fallback for a file no registered instrument claims.

    Deliberately incurious: it reports :data:`UNKNOWN`, encodes no grouping,
    and offers no header metadata beyond the defaults. This reproduces
    exactly what AuRE did with an unrecognised filename before the registry
    existed, so adding the seam changed nothing for such files.

    It refuses nuisance parameters, matching the previous behaviour: those
    were gated on a state being classified ``partials``, which an
    unrecognised filename never was.
    """

    name = "generic"

    def matches(self, file_path: str, header: str = "") -> bool:
        # Never claims a file; the registry falls back to it explicitly.
        return False

    def file_role(self, file_path: str) -> str:
        return UNKNOWN

    def group_key(self, file_path: str) -> Optional[str]:
        return None

    def header_metadata(self, file_path: str) -> dict:
        return dict(DEFAULT_HEADER_METADATA)

    def run_title(self, file_path: str) -> str:
        return generic_run_title(file_path)

    def header_issues(self, file_path: str) -> List[str]:
        """Nothing. It cannot read the header, so it has no standing to judge it."""
        return []

    def role_supports_nuisance(self, role: str) -> bool:
        return False
