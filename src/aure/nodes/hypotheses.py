"""Shared helpers for the structural-hypothesis list.

The ranked ``structural_hypotheses`` list is created at intake and then
evolves across the refinement loop. Because the state field is replaced
wholesale on every node return (no ``operator.add`` reducer), the only thing
keeping the list coherent is that every write goes through one guarded merge.

This module is that guard. It is intentionally LLM-free and pure so it can be
unit-tested in isolation and reused by the intake, evaluation, and modeling
nodes:

* ``merge_structural_hypotheses`` — fold an LLM-returned list back onto the
  authoritative prior list. Identity fields are immutable; only the mutable
  status fields are copied over. New entries are appended **only** when
  ``allow_new=True`` (evaluation), and are dropped (with a warning) otherwise
  (modeling) so a misbehaving LLM can never silently grow, drop, or rename
  the backlog.
* ``rerank_hypotheses`` — reorder the list by an explicit id ranking (rank is
  encoded by list position, which is how consumers read it).
* ``next_hypothesis_id`` — allocate the next stable id.
* ``normalize_hypothesis_states`` — validate an LLM-proposed per-state *scope*
  against the run's actual state names.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# The four valid lifecycle states for a hypothesis.
ALLOWED_STATUSES = {"pending", "tried", "confirmed", "rejected"}

# Fields the merge will copy from an LLM-returned entry onto an existing one.
# Everything else (id/title/rationale/change/skill_source/origin/states) is
# identity and immutable once the hypothesis exists. ``states`` belongs with
# identity, not with status: "add an oxide to the air state" and "add an oxide
# to every state" are different hypotheses that must be accepted or rejected
# on their own evidence, so re-scoping an existing entry is a rename.
_MUTABLE_FIELDS = ("status", "tried_in_iteration", "notes")


def normalize_hypothesis_states(raw: Any, known: Optional[List[str]]) -> List[str]:
    """Validate an LLM-proposed state scope for one hypothesis.

    Returns the subset of *raw* that names a real state, in the order the run
    declares them. An empty list means "applies to every state" — the shared
    template — and is also what a scope covering all of them collapses to, so
    consumers have exactly one spelling for "global" to test.

    Unknown names are dropped rather than raising: a hallucinated state name
    must not scope a change to a state that does not exist, and must not take
    down the run either. A single-state run (or an unknown *known* list) has
    nothing to scope, so it always yields the global spelling.
    """
    if not known or len(known) < 2 or not isinstance(raw, list):
        return []
    wanted = {str(x).strip() for x in raw if isinstance(x, (str, int, float))}
    if not wanted:
        return []
    scoped = [name for name in known if name in wanted]
    unknown = wanted - set(known)
    if unknown:
        logger.warning(
            "[HYPOTHESES] Dropped unknown state name(s) %s from a hypothesis "
            "scope; known states: %s",
            sorted(unknown),
            known,
        )
    if len(scoped) == len(known):
        return []  # covers everything — that is the global template case
    return scoped


def next_hypothesis_id(hypotheses: List[Dict[str, Any]]) -> int:
    """Return the next 1-based id not already used in *hypotheses*."""
    ids = [h.get("id") for h in hypotheses if isinstance(h.get("id"), int)]
    return (max(ids) + 1) if ids else 1


def _coerce_status(value: Any, *, hyp_id: Any, default: Optional[str]) -> Optional[str]:
    """Validate a status value; fall back to *default* and log on mismatch."""
    if isinstance(value, str) and value in ALLOWED_STATUSES:
        return value
    if value is not None:
        logger.warning(
            "[HYPOTHESES] Ignoring invalid status %r for hypothesis #%s", value, hyp_id
        )
    return default


def merge_structural_hypotheses(
    prior: Optional[List[Dict[str, Any]]],
    llm_returned: Optional[List[Dict[str, Any]]],
    *,
    allow_new: bool,
    current_iteration: int,
    default_origin: str = "evaluation",
) -> List[Dict[str, Any]]:
    """Merge an LLM-returned hypothesis list back onto the authoritative one.

    The *prior* list is the source of truth for membership and identity. For
    every prior entry, if the LLM returned an entry with the same ``id``, only
    its mutable fields (``status``/``tried_in_iteration``/``notes``) are copied
    over — title/rationale/change/skill_source/origin are preserved verbatim,
    so the LLM cannot rename or re-scope an existing hypothesis.

    Returned entries whose ``id`` is not in *prior* (or which carry no id) are
    treated as proposed *new* hypotheses:

    * when ``allow_new`` is True they are appended with a freshly allocated id,
      ``origin=default_origin`` and ``created_in_iteration=current_iteration``;
    * otherwise they are dropped and a warning is logged.

    Order of the prior entries is preserved (re-ranking is a separate step via
    :func:`rerank_hypotheses`); new entries are appended at the end.

    Parameters
    ----------
    prior
        The authoritative current list (may be None/empty).
    llm_returned
        Whatever the LLM produced (may be None/empty, or contain new entries).
    allow_new
        Whether unknown-id entries may be added. True for the evaluation
        revision step; False for the modeling write-back (status-only).
    current_iteration
        Stamped onto any newly added entry's ``created_in_iteration``.
    default_origin
        ``origin`` assigned to added entries that don't declare one.
    """
    prior = prior or []
    llm_returned = llm_returned or []

    returned_by_id: Dict[int, Dict[str, Any]] = {}
    for h in llm_returned:
        hid = h.get("id")
        if isinstance(hid, int):
            returned_by_id.setdefault(hid, h)

    merged: List[Dict[str, Any]] = []
    prior_ids = set()
    for ph in prior:
        entry = dict(ph)
        pid = entry.get("id")
        prior_ids.add(pid)
        ret = returned_by_id.get(pid)
        if ret:
            new_status = _coerce_status(
                ret.get("status"), hyp_id=pid, default=entry.get("status")
            )
            if new_status is not None:
                entry["status"] = new_status
            if "tried_in_iteration" in ret:
                entry["tried_in_iteration"] = ret["tried_in_iteration"]
            if "notes" in ret:
                entry["notes"] = ret["notes"]
        merged.append(entry)

    # Candidate new entries: returned items whose id is not in prior (or idless).
    next_id = next_hypothesis_id(merged)
    dropped = 0
    for h in llm_returned:
        hid = h.get("id")
        if isinstance(hid, int) and hid in prior_ids:
            continue  # already folded in as a status update above
        title = str(h.get("title", "")).strip()
        if not (allow_new and title):
            dropped += 1
            continue
        merged.append(
            {
                "id": next_id,
                "title": title,
                "rationale": str(h.get("rationale", "")).strip(),
                "change": str(h.get("change", "")).strip(),
                "skill_source": str(h.get("skill_source", "")).strip(),
                "origin": str(h.get("origin") or default_origin),
                # Already normalized by the caller (which knows the state
                # names); anything else is treated as unscoped.
                "states": (
                    list(h["states"]) if isinstance(h.get("states"), list) else []
                ),
                "status": _coerce_status(
                    h.get("status"), hyp_id=next_id, default="pending"
                ),
                "tried_in_iteration": h.get("tried_in_iteration"),
                "created_in_iteration": current_iteration,
                "notes": str(h.get("notes", "")).strip(),
            }
        )
        next_id += 1

    if dropped:
        logger.warning(
            "[HYPOTHESES] Dropped %d hypothesis entr%s the LLM tried to add or "
            "rename (allow_new=%s); membership is fixed outside intake/evaluation.",
            dropped,
            "y" if dropped == 1 else "ies",
            allow_new,
        )
    return merged


def rerank_hypotheses(
    hypotheses: List[Dict[str, Any]],
    ranked_ids: Optional[List[int]],
) -> List[Dict[str, Any]]:
    """Reorder *hypotheses* to match *ranked_ids* (rank = list position).

    Ids listed in *ranked_ids* come first, in that order. Any hypothesis whose
    id is not mentioned keeps its original relative order at the end. Unknown
    ids in *ranked_ids* are ignored. A falsy *ranked_ids* leaves order intact.
    """
    if not ranked_ids:
        return list(hypotheses)

    by_id = {h.get("id"): h for h in hypotheses}
    ordered: List[Dict[str, Any]] = []
    seen = set()
    for rid in ranked_ids:
        h = by_id.get(rid)
        if h is not None and rid not in seen:
            ordered.append(h)
            seen.add(rid)
    for h in hypotheses:
        if h.get("id") not in seen:
            ordered.append(h)
            seen.add(h.get("id"))
    return ordered
