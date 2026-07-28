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

## The profile the detector actually reads

### 5. `_find_profile_dat` can be shadowed by a stale export

**Predates this change.** `fitting._find_profile_dat`
([`src/aure/nodes/fitting.py:1109`](src/aure/nodes/fitting.py)) prefers the
literal `problem-1-profile.dat`, then falls back to
`sorted(d.glob("*-1-profile.dat"))[0]`. An alphabetically-earlier stale export
left in the directory beats the file the current fit just wrote, so the artifact
detector — and hence the clamp's verification — can be fed the wrong profile.
Nothing checks mtime or the problem name.

---

## Which fit is "the answer"

### 6. `aure analyze --json` and `aure batch` report `fit_results[-1]`

**Predates this change.** The human report resolves `final_selection["index"]`
correctly ([`src/aure/cli.py:864`](src/aure/cli.py)), but three other surfaces
take the last fit *performed* instead:

- `aure analyze --json` — [`src/aure/cli.py:743`](src/aure/cli.py)
- `aure batch`'s per-job headline χ² and per-job JSON —
  [`src/aure/cli.py:1579`](src/aure/cli.py) (the number CI gates on)
- the ISAAC context parameter block — [`src/aure/exporters/isaac.py:76`](src/aure/exporters/isaac.py)

`fit_results[-1]` is routinely an iteration `finalize` rejected, so the JSON and
the terminal can disagree about which model the run is reporting.

### 7. The validation harness is veto-unaware and scores a different iteration

**Predates this change.** `validation/batch_runner.py:69` reads
`fit_results[-1]`; `validation/comparator.py:115` and
`validation/cli.py:175,244` read `state["best_chi2"]`. Neither is the reported
answer (`final_selection` / `current_chi2`), and neither knows about the profile
veto — so the harness can score an iteration the run never reported, or the
sub-floor fit that #10 let anchor the baseline.

---

## Acceptance thresholds

### 8. `aure evaluate` judges against the ambient threshold and applies no clamp

**Predates this change.** [`src/aure/cli.py:2434`](src/aure/cli.py) calls
`_get_chi2_max()` with no state — it works off a refl1d directory's
`problem.json` and never reads the run's `final_state.json`, so it uses the
ambient `CHI2_MAX` rather than the `chi2_max` the evaluated run pinned. It also
applies neither the acceptance clamp nor the SLD-profile check
(`grep -n '_clamp_acceptance_to_chi2\|_detect_profile_artifacts_into'
src/aure/cli.py` is empty), yet prints an `acceptable` field with no note that
it is advisory. `aure analyze` can complete on a fit `aure evaluate` calls
unacceptable, and the reverse.

### 9. The regression baseline is floor-blind

**Predates this change; the floor made it visible.** `fitting`
([`src/aure/nodes/fitting.py:123-127`](src/aure/nodes/fitting.py)) records
`best_chi2` / `best_model` as the lowest χ² **outright**, with no floor test. One
noise-absorbing fit (χ² ≪ 1 from an overestimated `dR` column or excess free
parameters) becomes the baseline that `evaluation`'s χ² and BIC guardrails
compare against ([`evaluation.py:637`](src/aure/nodes/evaluation.py) and
[`:657`](src/aure/nodes/evaluation.py)), so every later *honest* fit reads as a
regression and gets reverted. The BIC branch is the harsher one: `BIC` is
monotone in χ², it has no 5% slack, and it also marks the tried hypothesis
`rejected`.

### 10. Per-state χ² is checked against the ceiling but not the floor

**Introduced by this change** — the shipped clamp's own gap.
`evaluation._per_file_over_threshold`
([`src/aure/nodes/evaluation.py:155`](src/aure/nodes/evaluation.py)) blocks the
clamp when any per-file/per-state χ² exceeds `chi2_max` (and on the `+inf`
fit-failed sentinel), but there is no matching sub-floor test. Reproduced: with
`per_file_results` of `d2o` at χ² = 0.004 and `h2o` at 2.0, an aggregate of 1.0
is force-accepted — a contrast with overestimated `dR` hiding under a passing
total. Mostly latent today only because of #5: the clamp never fires on
co-refinements at all.

### 11. `finalize._select` does not consult the acceptance floor

**Predates the floor; the mirror of the veto gap just fixed for #1's sibling.**
`_select` ([`src/aure/nodes/finalize.py`](src/aure/nodes/finalize.py)) now sets
profile-vetoed fits aside, but it still ranks purely on χ² otherwise — so a fit
*below* `chi2_min`, which the clamp explicitly refused to accept as a pass, can
still win outright and be reported. The floor exists because a reduced χ² far
under 1 is evidence about the `dR` column rather than the structure, and an
overfitted iteration is exactly the kind that scores lowest. Same shape as the
veto: the stop condition gates acceptance but not the reported answer. The fix is
the same too — set sub-floor fits aside unless they are the whole field, reading
the floor from `state["chi2_min"]`.

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

### 15. The web Setup form cannot set the χ² acceptance window

`chi2_max` / `chi2_min` are `SetupConfig` keys and are in `setup._DUMP_ORDER`,
but `grep -rn 'chi2_m' src/aure/web/` is empty — the form has no field for
either, so a setup loaded through it and saved again silently loses them (the
same drop it already does for `fit_method`, `evaluation_criteria`,
`model_constraints`, …). Any prose claiming the form round-trips those two keys
needs correcting along with the form.
