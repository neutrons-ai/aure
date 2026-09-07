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
from typing import Optional, Protocol, runtime_checkable

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

    def role_supports_nuisance(self, role: str) -> bool:
        return False
