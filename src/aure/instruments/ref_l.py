"""ORNL REF_L (SNS Liquids Reflectometer).

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

import os
import re
from typing import Optional

from .base import COMBINED, DEFAULT_HEADER_METADATA, PARTIAL, UNKNOWN, read_file_header

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
