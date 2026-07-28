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

## The SLD-profile veto stops at `evaluation`

`evaluation` refuses to accept a fit whose SLD profile leaves the range its
bounding media can produce. Nothing past that node knows the veto happened.

### 1. `finalize` can report a model the SLD-profile check rejected

**Predates this change. This is the headline one.**

`finalize._select` ([`src/aure/nodes/finalize.py:367`](src/aure/nodes/finalize.py))
ranks candidate fits on lowest χ² — with a parsimony tie-break inside
`FINAL_SELECTION_TOL` — and knows nothing about the profile veto.

Reproduced end to end: iteration 1 (SEI/Plated/Cu/Ti/SiOx on Si) fits to
χ² = 0.62, but its erf tail dips to SLD −0.88 at z = 890 Å — outside the range
its bounding media can produce — so `evaluation` vetoes acceptance and the loop
refines; iteration 2 (Cu on Si) reaches χ² = 1.20, is clean, and is accepted.
`finalize` then selects **iteration 1** and reports the vetoed five-layer stack.
Re-verified against the current tree by calling `finalize_node` on exactly that
two-iteration state: it returns `index 0`, `chi_squared 0.62`, promotes the
five-layer definition into `current_model`, and `final_selection` carries no
veto/demotion field at all. (The z/SLD numbers come from the original
end-to-end reproduction and were not re-run; the selection behaviour they
trigger is verified here.)

Not an edge case. The veto exists precisely because the erf-tail excursion
**buys** χ², so the vetoed iteration is routinely the run's lowest-χ² one — and
the new deterministic stop makes it more likely to be where the run ends.

**And the warning never reaches the reader.** `cli._print_analysis_results`'s
Issues/Suggestions block reads `result.get("evaluation")`
([`src/aure/cli.py:922`](src/aure/cli.py)), a state key no workflow node ever
writes (`grep -rn '"evaluation"' src/aure/` finds only `mcp_server.py`'s own
payload and the runner's node tables). The evaluator's `issues` live on the
judged `FitResult`. So on an `aure analyze` run that whole block is dead code
and the excursion text — like the new sub-floor-χ² note — is never printed. The
vetoed model is presented with no caveat whatsoever.

Root of this and of #2–#4: the verdict is never recorded structurally. Inside
the node the two facts exist as the private `analysis["_profile_checked"]` /
`["_profile_artifact"]` markers the χ² clamp reads, but they are not persisted
onto the `FitResult`, so "checked and clean" is indistinguishable from "never
checked" and downstream can only pattern-match issue prose.

*Fix in the preserved patch:* exclude vetoed fits from the candidate pool unless
nothing else qualifies, record the demotion in `final_selection`, surface it in
the report.

### 2. `final_fit` polishes and adopts a profile-vetoed model

**Predates this change.** The gate at
[`src/aure/nodes/final_fit.py:161`](src/aure/nodes/final_fit.py) is χ²-only
(`chi2_before > gate` → skip). So a vetoed selection — which by construction has
a *low* χ² — sails straight through, and the node spends a full MCMC budget
characterizing it, adopts the result, and repoints `problem.json` at that
export. The most expensive step of the run is the one least guarded.

### 3. The ISAAC export discloses nothing about a rejected profile

**Predates this change.** `exporters/isaac._generate_context_description`
([`src/aure/exporters/isaac.py:62`](src/aure/exporters/isaac.py)) returns
`state["sample_description"]` verbatim when no LLM is configured, and the
exporter's `warnings` list only ever collects operational failures (ingest
timeouts, converter errors). An exported ISAAC record therefore carries no hint
that the model's profile was rejected — and it outlives the terminal any banner
was printed in.

### 4. The web Results tab and History chart never mention the veto

**Predates this change.** `grep -rn 'veto\|artifact\|demot\|profile_checked'
src/aure/web/` is empty. The History tab's χ² progression can plot a vetoed
iteration as the run's best point, unannotated, and the Results tab renders its
model like any other.

---

## The profile the detector actually reads

### 5. Per-state SLD profiles are written but never read back

**Predates this change.** `fitting._write_state_profile`
([`src/aure/nodes/fitting.py:1042`](src/aure/nodes/fitting.py), called at 1011)
writes `export_dir/state_<name>/profile.dat` for every state of a
co-refinement. Nothing reads them — `_read_profile_dat` is the only consumer and
it reads the single top-level `*-1-profile.dat`, which is `states[0]`'s alone.

This is the direct cause of the shipped limitation:
`evaluation._detect_profile_artifacts_into` refuses to mark a multi-state fit
verified ([`src/aure/nodes/evaluation.py:1476`](src/aure/nodes/evaluation.py)),
and the clamp stands down on anything unverified — **so the deterministic χ²
stop is inert for every co-refinement**, and those runs still finish on the
evaluator's verdict. That is the safe direction (`states[0]`'s profile is still
checked, so a *veto* still fires) but it means `chi2_max` does nothing on a
`states:` run. Wiring the per-state files into the detector is the fix.

### 6. `_find_profile_dat` can be shadowed by a stale export

**Predates this change.** `fitting._find_profile_dat`
([`src/aure/nodes/fitting.py:1109`](src/aure/nodes/fitting.py)) prefers the
literal `problem-1-profile.dat`, then falls back to
`sorted(d.glob("*-1-profile.dat"))[0]`. An alphabetically-earlier stale export
left in the directory beats the file the current fit just wrote, so the artifact
detector — and hence the clamp's verification — can be fed the wrong profile.
Nothing checks mtime or the problem name.

---

## Which fit is "the answer"

### 7. `aure analyze --json` and `aure batch` report `fit_results[-1]`

**Predates this change.** The human report resolves `final_selection["index"]`
correctly ([`src/aure/cli.py:864`](src/aure/cli.py)), but three other surfaces
take the last fit *performed* instead:

- `aure analyze --json` — [`src/aure/cli.py:743`](src/aure/cli.py)
- `aure batch`'s per-job headline χ² and per-job JSON —
  [`src/aure/cli.py:1579`](src/aure/cli.py) (the number CI gates on)
- the ISAAC context parameter block — [`src/aure/exporters/isaac.py:76`](src/aure/exporters/isaac.py)

`fit_results[-1]` is routinely an iteration `finalize` rejected, so the JSON and
the terminal can disagree about which model the run is reporting.

### 8. The validation harness is veto-unaware and scores a different iteration

**Predates this change.** `validation/batch_runner.py:69` reads
`fit_results[-1]`; `validation/comparator.py:115` and
`validation/cli.py:175,244` read `state["best_chi2"]`. Neither is the reported
answer (`final_selection` / `current_chi2`), and neither knows about the profile
veto — so the harness can score an iteration the run never reported, or the
sub-floor fit that #10 let anchor the baseline.

---

## Acceptance thresholds

### 9. `aure evaluate` judges against the ambient threshold and applies no clamp

**Predates this change.** [`src/aure/cli.py:2434`](src/aure/cli.py) calls
`_get_chi2_max()` with no state — it works off a refl1d directory's
`problem.json` and never reads the run's `final_state.json`, so it uses the
ambient `CHI2_MAX` rather than the `chi2_max` the evaluated run pinned. It also
applies neither the acceptance clamp nor the SLD-profile check
(`grep -n '_clamp_acceptance_to_chi2\|_detect_profile_artifacts_into'
src/aure/cli.py` is empty), yet prints an `acceptable` field with no note that
it is advisory. `aure analyze` can complete on a fit `aure evaluate` calls
unacceptable, and the reverse.

### 10. The regression baseline is floor-blind

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

### 11. Per-state χ² is checked against the ceiling but not the floor

**Introduced by this change** — the shipped clamp's own gap.
`evaluation._per_file_over_threshold`
([`src/aure/nodes/evaluation.py:155`](src/aure/nodes/evaluation.py)) blocks the
clamp when any per-file/per-state χ² exceeds `chi2_max` (and on the `+inf`
fit-failed sentinel), but there is no matching sub-floor test. Reproduced: with
`per_file_results` of `d2o` at χ² = 0.004 and `h2o` at 2.0, an aggregate of 1.0
is force-accepted — a contrast with overestimated `dR` hiding under a passing
total. Mostly latent today only because of #5: the clamp never fires on
co-refinements at all.

---

## Configuration, and one unrelated pair

### 12. Four documented final-fit setup keys are rejected by the setup loader

**Predates this change.** `fit_method_final`, `fit_steps_final`,
`fit_burn_final` and `final_fit_chi2_max` are mapped to env vars by
`cli._build_env_overrides` ([`src/aure/cli.py:1681-1684`](src/aure/cli.py)) and
documented as YAML keys in `docs/finalization.md` and
`aure_config.example.yaml` — all three at baseline — but are **absent from
`setup._KNOWN_TOP_LEVEL`** ([`src/aure/setup.py:103`](src/aure/setup.py)).

A setup or batch job using any of them is rejected with `unknown top-level
key(s)`, so the optional final uncertainty fit is unreachable from a setup YAML.
Verified: the shipped `aure_config.example.yaml` **does** load — the four keys
are commented out there — but uncommenting them (i.e. following the file's own
documentation) fails with
`ConfigError: unknown top-level key(s): ['final_fit_chi2_max', 'fit_burn_final',
'fit_method_final', 'fit_steps_final']`. Adding the four names to
`_KNOWN_TOP_LEVEL` (and `_DUMP_ORDER`) is the whole fix.

### 13. Two MCP tools are dead

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

### 14. `ruff format` is not clean at baseline

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

### 15. Interactive review feedback at a clamped accept is collected and thrown away

[`runner.py:321`](src/aure/workflow/runner.py) deliberately keeps the review
pause alive when `chi2_clamp_accepted` is set (the one verdict where code
overrode an objecting evaluator is the one a human should see), but plain text
feedback only sets `pending_user_feedback` — and the loop then breaks at
[`runner.py:409`](src/aure/workflow/runner.py) on `workflow_complete`, so it is
never acted on. Only the `restart_checkpoint` path clears `workflow_complete`.
The four-line fix (clear `workflow_complete` and `chi2_clamp_accepted` when
feedback arrives at a clamped accept) is in the preserved patch.

### 16. The web Setup form cannot set the χ² acceptance window

`chi2_max` / `chi2_min` are `SetupConfig` keys and are in `setup._DUMP_ORDER`,
but `grep -rn 'chi2_m' src/aure/web/` is empty — the form has no field for
either, so a setup loaded through it and saved again silently loses them (the
same drop it already does for `fit_method`, `evaluation_criteria`,
`model_constraints`, …). Any prose claiming the form round-trips those two keys
needs correcting along with the form.
