"""Instrument and file-format support.

See :mod:`aure.instruments.base` for the protocol a new instrument
implements, and ``docs/instruments.md`` for a walkthrough.
"""

from .base import (
    COMBINED,
    CURRENT,
    DEPRECATED,
    LEGACY,
    LIFECYCLES,
    MAX_RUN_TITLE_LEN,
    PROTOTYPE,
    aliases,
    authoritative_fields,
    format_version,
    generic_run_title,
    lifecycle,
    lifecycle_note,
    DEFAULT_HEADER_METADATA,
    PARTIAL,
    UNKNOWN,
    GenericInstrument,
    Instrument,
    read_file_header,
)
from .orso import ORSOInstrument
from .ref_l import (
    REFLAutoreductionInstrument,
    REFLInstrument,
    REFLv1Instrument,
    REFLv2Instrument,
)
from .registry import (
    file_role,
    generic,
    group_key,
    header_issues,
    header_metadata,
    run_title,
    register,
    registered,
    resolve,
    resolve_by_name,
)

# Built-ins, in priority order. The two REF_L reductions first because their
# filename patterns are the most specific; ORSO claims by extension and would
# otherwise be indistinguishable for a ``.txt`` REF_L file. The two REF_L
# entries cannot collide — disjoint patterns, disjoint header markers — so
# their order records which is more specific, not a conflict.
register(REFLv2Instrument())
register(REFLv1Instrument())
register(ORSOInstrument())

__all__ = [
    "COMBINED",
    "authoritative_fields",
    "PARTIAL",
    "UNKNOWN",
    "DEFAULT_HEADER_METADATA",
    "Instrument",
    "GenericInstrument",
    "REFLv1Instrument",
    "REFLv2Instrument",
    "REFLInstrument",
    "REFLAutoreductionInstrument",
    "ORSOInstrument",
    "read_file_header",
    "generic_run_title",
    "run_title",
    "header_issues",
    "MAX_RUN_TITLE_LEN",
    "CURRENT",
    "LEGACY",
    "PROTOTYPE",
    "DEPRECATED",
    "LIFECYCLES",
    "lifecycle",
    "lifecycle_note",
    "format_version",
    "aliases",
    "register",
    "registered",
    "generic",
    "resolve",
    "resolve_by_name",
    "file_role",
    "group_key",
    "header_metadata",
]
