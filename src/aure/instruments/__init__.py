"""Instrument and file-format support.

See :mod:`aure.instruments.base` for the protocol a new instrument
implements, and ``docs/instruments.md`` for a walkthrough.
"""

from .base import (
    COMBINED,
    DEFAULT_HEADER_METADATA,
    PARTIAL,
    UNKNOWN,
    GenericInstrument,
    Instrument,
    read_file_header,
)
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

# Built-ins, in priority order.
register(REFLInstrument())

__all__ = [
    "COMBINED",
    "PARTIAL",
    "UNKNOWN",
    "DEFAULT_HEADER_METADATA",
    "Instrument",
    "GenericInstrument",
    "REFLInstrument",
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
