"""Instrument resolution.

Two entry points, and the difference is deliberate:

:func:`resolve_by_name`
    Filename only. Used where AuRE has always classified by name and must not
    start depending on a file being readable — ``config`` parses a setup file
    before the run begins, and ``refl1d_import`` classifies probes whose data
    files it is still writing.

:func:`resolve`
    May read the file header to identify an instrument the filename does not
    reveal. Used for header metadata, where the file is present by definition.

Registration order is priority order; the first instrument to claim a file
wins. Built-ins are registered at import. Third-party instruments arrive
through the ``aure.instruments`` entry-point group, which is loaded lazily on
first resolution so that a broken plugin cannot stop AuRE from importing.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from .base import GenericInstrument, Instrument, read_file_header

logger = logging.getLogger(__name__)

#: Registered instruments, highest priority first.
_REGISTRY: List[Instrument] = []
#: The catch-all. Never in ``_REGISTRY`` — it claims nothing and is returned
#: only when no registered instrument matches.
_GENERIC = GenericInstrument()

_entry_points_loaded = False

#: Force a specific instrument for every file in this process, by name.
#: An escape hatch for data whose filenames lie about their origin.
_ENV_OVERRIDE = "AURE_INSTRUMENT"


def register(instrument: Instrument, *, first: bool = False) -> None:
    """Add *instrument* to the registry.

    ``first=True`` gives it priority over everything already registered,
    which is what a site-specific override of a built-in needs. Re-registering
    an instrument with the same ``name`` replaces the earlier one rather than
    shadowing it, so a plugin can override a built-in without duplication.
    """
    global _REGISTRY
    _REGISTRY = [i for i in _REGISTRY if i.name != instrument.name]
    if first:
        _REGISTRY.insert(0, instrument)
    else:
        _REGISTRY.append(instrument)
    logger.debug("[INSTRUMENTS] registered %s (first=%s)", instrument.name, first)


def registered() -> List[Instrument]:
    """The registry in priority order, entry points included."""
    _load_entry_points()
    return list(_REGISTRY)


def generic() -> Instrument:
    """The catch-all instrument."""
    return _GENERIC


def _load_entry_points() -> None:
    """Load third-party instruments from the ``aure.instruments`` group.

    Failures are logged and skipped: a broken or incompatible plugin must not
    be able to prevent AuRE from running on the instruments it does have.
    """
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="aure.instruments")
    except Exception as e:  # pragma: no cover - depends on the environment
        logger.debug("[INSTRUMENTS] entry-point discovery unavailable: %s", e)
        return
    for ep in eps:
        try:
            factory = ep.load()
            register(factory() if callable(factory) else factory)
            logger.info("[INSTRUMENTS] loaded plugin %r", ep.name)
        except Exception as e:
            logger.warning("[INSTRUMENTS] plugin %r failed to load: %s", ep.name, e)


def _override() -> Optional[Instrument]:
    """The instrument named by :data:`_ENV_OVERRIDE`, if set and known."""
    forced = (os.environ.get(_ENV_OVERRIDE) or "").strip()
    if not forced:
        return None
    for inst in registered():
        if inst.name.lower() == forced.lower():
            return inst
    if forced.lower() == _GENERIC.name:
        return _GENERIC
    logger.warning(
        "[INSTRUMENTS] %s=%r names no registered instrument; ignoring "
        "(known: %s)",
        _ENV_OVERRIDE,
        forced,
        ", ".join(i.name for i in registered()) or "none",
    )
    return None


def resolve_by_name(file_path: str) -> Instrument:
    """Resolve from the filename alone; never reads the file."""
    forced = _override()
    if forced is not None:
        return forced
    for inst in registered():
        try:
            if inst.matches(file_path, ""):
                return inst
        except Exception as e:
            logger.warning(
                "[INSTRUMENTS] %s.matches raised on %s: %s", inst.name, file_path, e
            )
    return _GENERIC


def resolve(file_path: str) -> Instrument:
    """Resolve from the filename, falling back to a header sniff.

    The name is tried first so that a file whose name identifies it costs no
    I/O; only an unclaimed file is opened.
    """
    forced = _override()
    if forced is not None:
        return forced
    by_name = resolve_by_name(file_path)
    if by_name is not _GENERIC:
        return by_name
    header = read_file_header(file_path)
    if not header:
        return _GENERIC
    for inst in registered():
        try:
            if inst.matches(file_path, header):
                logger.debug(
                    "[INSTRUMENTS] %s claimed %s by header", inst.name, file_path
                )
                return inst
        except Exception as e:
            logger.warning(
                "[INSTRUMENTS] %s.matches raised on %s: %s", inst.name, file_path, e
            )
    return _GENERIC


# ---------------------------------------------------------------------------
# Convenience wrappers — the surface most call sites want.
# ---------------------------------------------------------------------------


def file_role(file_path: str) -> str:
    """The role of *file_path* per its resolved instrument (filename only)."""
    return resolve_by_name(file_path).file_role(file_path)


def group_key(file_path: str) -> Optional[str]:
    """The measurement-set identity of *file_path*, or ``None``."""
    return resolve_by_name(file_path).group_key(file_path)


def header_metadata(file_path: str) -> dict:
    """Deterministic header metadata, header sniff included."""
    return resolve(file_path).header_metadata(file_path)
