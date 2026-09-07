"""ORSO ``.ort`` — the Open Reflectometry Standards Organisation text format.

The second instrument, and the one that shows the seam is real: it is a
community interchange format rather than a facility convention, so it answers
the protocol's questions from *structured header metadata* instead of from
filename patterns.

Two things this fixes as a side effect of implementing it properly:

* ``tools.data_tools.parse_ort_file`` reads the YAML header into a list and
  then discards it. The instrument parses it, so an ORSO file's declared
  incident angle and instrument name now reach the run.
* An ORSO ``sQz`` column is a **standard deviation**, not a FWHM — the ``s``
  prefix is the standard's notation for one sigma. AuRE's global default is
  ``dq_is_fwhm=True`` (a REF_L convention), which over-broadens an ORSO
  resolution by a factor of 2.35. This instrument declares the correct
  convention for its own files.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .base import COMBINED, DEFAULT_HEADER_METADATA, PARTIAL, read_file_header

logger = logging.getLogger(__name__)

_SUFFIXES = (".ort",)

#: Markers that identify an ORSO header regardless of the file's extension.
_HEADER_MARKERS = ("ORSO reflectivity data file", "data_source:")


def _header_yaml(file_path: str, header: str = "") -> dict:
    """Parse the commented YAML block at the top of an ORSO file.

    Returns ``{}`` on any failure. The header is a YAML document in which
    every line is prefixed with ``"# "``; stripping one level of that prefix
    yields the document. The banner line becomes a YAML comment once
    unprefixed, which is why it can be left in place.
    """
    text = header or read_file_header(file_path)
    if not text:
        return {}
    lines = []
    for line in text.split("\n"):
        if not line.startswith("#"):
            break
        stripped = line[1:]
        lines.append(stripped[1:] if stripped.startswith(" ") else stripped)
    if not lines:
        return {}
    try:
        import yaml

        parsed = yaml.safe_load("\n".join(lines))
    except Exception as e:
        logger.debug("[ORSO] header YAML unparsable in %s: %s", file_path, e)
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _dig(doc: Any, *keys: str) -> Any:
    """Walk nested mappings, returning ``None`` at the first missing key."""
    cur = doc
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _magnitude(value: Any) -> Optional[float]:
    """Read an ORSO ``Value`` (``{magnitude, unit}``) or a bare number.

    Returns ``None`` for a range (``magnitude`` given as a list), because a
    span of incident angles is not a single angle — the same reasoning that
    makes a REF_L combined file yield theta 0.
    """
    if isinstance(value, dict):
        value = value.get("magnitude")
    if isinstance(value, bool) or isinstance(value, (list, tuple)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


class ORSOInstrument:
    """ORSO ``.ort`` files, read from their declared metadata."""

    name = "ORSO"

    #: The standard defines ``sQz`` as one standard deviation, so the dQ
    #: convention is not a matter of interpretation and must not be left to
    #: the LLM header parse.
    authoritative_fields = ("dq_is_fwhm",)

    def matches(self, file_path: str, header: str = "") -> bool:
        if os.path.splitext(file_path)[1].lower() in _SUFFIXES:
            return True
        return bool(header) and any(m in header for m in _HEADER_MARKERS)

    def file_role(self, file_path: str) -> str:
        """A single declared incident angle is one angle; anything else is a curve.

        An ORSO file that states one ``incident_angle`` describes a single
        measurement geometry, which is what a REF_L partial is. A file
        declaring an angle *range* (or none at all) is a complete curve.
        """
        doc = _header_yaml(file_path)
        angle = _magnitude(
            _dig(
                doc,
                "data_source",
                "measurement",
                "instrument_settings",
                "incident_angle",
            )
        )
        return PARTIAL if angle is not None else COMBINED

    def group_key(self, file_path: str) -> Optional[str]:
        """``None`` — ORSO encodes no equivalent of a REF_L set id.

        The standard has no field meaning "these files are segments of one
        measurement that must be co-refined". Grouping is therefore the
        user's to declare through ``states:``, and claiming otherwise would
        invent a constraint the format does not express.
        """
        return None

    def header_metadata(self, file_path: str) -> dict:
        doc = _header_yaml(file_path)
        meta = dict(DEFAULT_HEADER_METADATA)
        meta["instrument"] = (
            _dig(doc, "data_source", "experiment", "instrument") or self.name
        )
        angle = _magnitude(
            _dig(
                doc,
                "data_source",
                "measurement",
                "instrument_settings",
                "incident_angle",
            )
        )
        if angle is not None:
            meta["theta"] = angle
        # ORSO writes the resolution as one sigma (the ``s`` in ``sQz``).
        meta["dq_is_fwhm"] = False
        return meta

    def role_supports_nuisance(self, role: str) -> bool:
        return role == PARTIAL
