"""Instrument and file-format support.

See :mod:`aure.instruments.base` for the protocol a new instrument
implements, and ``docs/instruments.md`` for a walkthrough.
"""

from .base import (
    COMBINED,
    authoritative_fields,
    DEFAULT_HEADER_METADATA,
    PARTIAL,
    UNKNOWN,
    GenericInstrument,
    Instrument,
    read_file_header,
)
from .orso import ORSOInstrument
from .ref_l import REFLInstrument
from .registry import (
    file_role,
    generic,
    group_key,
    header_metadata,
    register,
    registered,
    resolve,
    resolve_by_name,
)

# Built-ins, in priority order. REF_L first because its filename patterns are
# the most specific; ORSO claims by extension and would otherwise be
# indistinguishable for a ``.txt`` REF_L file.
register(REFLInstrument())
register(ORSOInstrument())

__all__ = [
    "COMBINED",
    "authoritative_fields",
    "PARTIAL",
    "UNKNOWN",
    "DEFAULT_HEADER_METADATA",
    "Instrument",
    "GenericInstrument",
    "REFLInstrument",
    "ORSOInstrument",
    "read_file_header",
    "register",
    "registered",
    "generic",
    "resolve",
    "resolve_by_name",
    "file_role",
    "group_key",
    "header_metadata",
]
