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



## Gaps created by the split itself

Not pre-existing — these are things the shipped subset lacks because the rest of
the work was dropped.

### 15. The web Setup form cannot set the χ² acceptance window

`chi2_max` / `chi2_min` are `SetupConfig` keys and are in `setup._DUMP_ORDER`,
but `grep -rn 'chi2_m' src/aure/web/` is empty — the form has no field for
either, so a setup loaded through it and saved again silently loses them (the
same drop it already does for `fit_method`, `evaluation_criteria`,
`model_constraints`, …). Any prose claiming the form round-trips those two keys
needs correcting along with the form.
