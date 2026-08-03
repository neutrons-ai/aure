# Known issues (triage list)

Real, reproduced defects found while building the deterministic χ² stop
(`chi2_max` / `chi2_min` plus the untried-improvements report). They were
**deliberately not fixed** there — that change had already grown far past its
brief, and fixing them inline is what made it unreviewable. Almost all of them
predate it; where the new stop makes one *more likely to bite*, the entry says
so.

This is a triage list, not a design doc: what is wrong, how it shows up, roughly
where.

**One found defect was fixed instead of triaged**, because it broke part of what
shipped rather than merely sitting next to it: `_print_analysis_results` did
`fit.get("uncertainties", {}).get(name)` while `fitting.py` writes an explicit
`uncertainties=None` whenever the optimizer produced no `dx`, so every
`lm`/`amoeba` run aborted the terminal report with `AttributeError` — and the new
untried-improvements block prints *below* that point, making it unreachable. The
fix is the `or {}` idiom already used at `cli.py:2477`. It is recorded here only
so the finding is not lost; there is nothing left to do.

> **A working implementation of several of these already exists** — in the
> pre-split patch at
> `/private/tmp/claude-501/-Users-m2d-git-aure/067b89c9-ac89-475a-9351-ba996980d35e/scratchpad/backup/full-change.patch`
> (the whole ~12k-line change across 30 files, plus its test files alongside).
>
> **That is a session scratchpad. It is not durable storage — it will be reaped
> without warning.** Move it somewhere real (a branch, a tag, anywhere under
> `~`) before relying on it. Nothing here depends on it; it is a shortcut, not a
> record.

Line numbers were re-checked against the working tree while writing this;
`dc1dfca` is the pre-change baseline, so `git diff dc1dfca -- <file>` shows what
the shipped subset added.

---

## Unrelated to this change

### 12. Two MCP tools are dead

**Predates this change; wholly unrelated to it.**

- `mcp_server.evaluate_fit` imports `analyze_fit_quality` from `nodes.evaluation`
  ([`src/aure/mcp_server.py:350`](src/aure/mcp_server.py)). No such symbol
  exists anywhere in the package (`hasattr` → `False`) — `ImportError` on call.
- `mcp_server.run_fit` calls `run_refl1d_fit(model_script=…, data_file=…,
  method=…, max_iterations=…)` ([`src/aure/mcp_server.py:305`](src/aure/mcp_server.py)).
  The real signature is `run_refl1d_fit(model_definition, method, iteration,
  steps, burn, export_dir, model_name)`
  ([`src/aure/nodes/fitting.py:334`](src/aure/nodes/fitting.py)) — three of the
  four keywords do not exist, so it is a `TypeError` on call.

### 13. `ruff format` is not clean at baseline

**Predates this change.** With the pinned hook version (ruff 0.15.5), `ruff
format --check` on a clean `dc1dfca` checkout reports **7** files needing
reformatting: `src/aure/cli.py`, `src/aure/exporters/isaac.py`,
`src/aure/nodes/evaluation.py`, `src/aure/nodes/final_fit.py`,
`src/aure/skills/loader.py`, `src/aure/web/routes.py`,
`src/aure/workflow/runner.py`. Any hook run that touches one of them emits
reflows unrelated to whatever change is in flight — `isaac.py` in particular,
since it is otherwise untouched here. Reformat them in a standalone commit
rather than absorbing the churn into a feature branch.

---

## Gaps created by the split itself

Not pre-existing — these are things the shipped subset lacks because the rest of
the work was dropped.

### 14. Interactive review feedback at a clamped accept is collected and thrown away

[`runner.py:321`](src/aure/workflow/runner.py) deliberately keeps the review
pause alive when `chi2_clamp_accepted` is set (the one verdict where code
overrode an objecting evaluator is the one a human should see), but plain text
feedback only sets `pending_user_feedback` — and the loop then breaks at
[`runner.py:409`](src/aure/workflow/runner.py) on `workflow_complete`, so it is
never acted on. Only the `restart_checkpoint` path clears `workflow_complete`.
The four-line fix (clear `workflow_complete` and `chi2_clamp_accepted` when
feedback arrives at a clamped accept) is in the preserved patch.
