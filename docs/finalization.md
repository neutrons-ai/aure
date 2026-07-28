# Finalization: what happens at the end of a run, and how the final model is reported

This document covers the part of AuRE that was never written down: what happens
**after** the refinement loop stops — how the run picks the model it reports,
how (optionally) it attaches uncertainties, and how `final_state.json` /
`problem.json` are produced. If you touch the runner, the finalize node, the
final-fit node, or the checkpoint writer, read this first.

> TL;DR — The end of a run is a terminal block in `run_workflow_with_checkpoints`,
> reached after the node loop stops:
> **finalize (select the winner) → final_fit (optional dream polish) → save_final_state (write artifacts)**.

---

## 1. One execution engine

AuRE runs the workflow as a single hand-written state machine —
`run_workflow_with_checkpoints` in [`workflow/runner.py`](../src/aure/workflow/runner.py).
There is no graph framework. Everything drives this one engine:

| Caller | Entry | What it runs |
|--------|-------|--------------|
| CLI `aure analyze` | `run_analysis` | full loop + terminal block |
| Web UI (Flask, background thread) | `run_workflow_with_checkpoints` | full loop + terminal block |
| CLI `aure resume` | `run_from_checkpoint` | resume + terminal block |
| MCP `start_analysis_session` | `run_prepare` | intake→analysis→modeling only (`stop_after="modeling"`, no terminal block) |

The runner is a plain Python loop over `NODE_ORDER` that follows the
`ROUTING_FUNCTIONS` to pick the next node, with its own state accumulation
(`_merge_state_updates`), per-node checkpointing, interactive pause/resume,
restart-from-checkpoint, and tracing.

**Consequence:** anything that must happen at the end of a run belongs in the
runner's terminal block.

---

## 2. The terminal block (where "the end" actually is)

The refinement loop (`intake → analysis → modeling → fitting → evaluation ⟲`)
has ~6 different exits — the `complete`/`error` routes, the interactive
`__STOP__`, the `stop_after` break, the `max_total_iterations` cap, and the
missing-router break — and `workflow_complete` can break out *before* routing
even runs. So finalization is not handled at each exit (that would miss most of
them). Instead the runner runs it **after** the loop, where every exit
converges:

```
run_workflow_with_checkpoints:
    … manual node loop breaks on ANY exit …

    if not stop_after:
        if not finalized:
            finalize_node(state)          # 1. SELECT the winning model
            save_checkpoint("finalize")
            checkpoint_callback("finalize")

        final_fit_node(state)             # 2. OPTIONAL dream polish (self-gating)
        if it returned updates:
            save_checkpoint("final_fit")
            checkpoint_callback("final_fit")

    save_final_state(state)               # 3. WRITE final_state.json + problem.json
```

`stop_after` (used by `prepare`, which stops at `modeling`) skips the whole
block: a prepare run has no fit to finalize.

---

## 3. Step 1 — `finalize`: select the model to report (no fitting)

[`nodes/finalize.py`](../src/aure/nodes/finalize.py) does **not run a fit.** The
refinement loop is a search, and the last iteration is not necessarily the best.
Finalize makes the choice explicit and auditable:

- **Selection rule:** profile-vetoed fits are set aside; then lowest χ², with a
  **parsimony tie-break** — among fits within `FINAL_SELECTION_TOL` (default 2%)
  of the lowest χ², prefer the fewest free parameters, then the earliest
  iteration. (BIC's idea without depending on the stored `bic`.)

  The veto exclusion exists because the excursion is often *what buys* the low
  χ², so the vetoed iteration is routinely the run's best-scoring one — ranking
  on χ² alone reported exactly the model `evaluation` had refused to accept. A
  vetoed fit is reported only when it is the whole field, and then the selection
  message says so. `final_selection` records `vetoed_iterations`,
  `demoted_for_profile_artifact` (the veto *changed the answer*, not merely
  fired) and `selected_has_profile_artifact`. The verdict is read from the
  `profile_checked` / `profile_artifact` flags `evaluation` stamps on each judged
  `FitResult`, falling back to matching the excursion prose for checkpoints
  written before those were persisted.

  Selection still does **not** consult the acceptance floor (§8), so a sub-floor
  χ² can win outright — the same shape of gap, triaged as
  [issues.md](../issues.md) #11. Read it before changing `_select`.
- Resolves the *ModelDefinition* that produced the winning iteration via
  `model_history` (handles interactive rewinds and bounds-only re-fits, which
  append a `fit_results` entry with no history entry).
- Writes that iteration's fitted parameter values **back into** the definition
  (`_apply_fitted_values`) → the promoted `current_model`.
- Sets `current_model` and `current_chi2` to the winner.
- Records `final_selection` (index, iteration, χ², tie-break metadata) for audit.
- **Never writes `best_model` / `best_chi2`** — those are the loop's regression
  baseline that `aure resume` compares against, and a different question from
  "what does the run report". `fitting` maintains them, preferring a fit inside
  the acceptance window: a sub-floor fit claiming the baseline made every later
  honest fit read as a regression. It is recorded only when nothing in-window
  exists, so the guardrails are never left without one.
- Emits a **second, reporting-only message** listing the `structural_hypotheses`
  still `pending` (the run stops as soon as χ² lands in the acceptance window, so
  the ranked backlog is normally not exhausted) plus a one-line tally of what was
  attempted. Statuses are reported exactly as they stand — never re-derived here.
  It is appended *after* the selection message and is the only message on the
  no-usable-fits early return, so index `updates["messages"]` by content, not by
  `[0]` / `[-1]`. Suppressed when the identical text is already in the transcript
  (a re-finalized / resumed run) and omitted on an empty backlog, so a resumed
  run's `final_state.json` can lack a block the CLI report still prints — that
  reads `structural_hypotheses` directly. `pending_hypotheses` /
  `format_attempted_counts` / `hypothesis_label` are exported so the report and
  `aure batch` render the same backlog from the same selector.
- Idempotent via the `finalized` flag.

After finalize, `current_model` / `current_chi2` **are** the reported answer —
"the last iteration fitted" is explicitly *not* what gets reported.

---

## 4. Step 2 — `final_fit`: optional MCMC polish for uncertainties

[`nodes/final_fit.py`](../src/aure/nodes/final_fit.py). Fast exploration
optimizers (`amoeba`, `de`) find the best structure cheaply but report no usable
parameter uncertainties. When you want error bars, set `FIT_METHOD_FINAL`
(typically `dream`) and this step runs **one** MCMC fit on the
finalize-selected model, seeded from its fitted values. amoeba finds the basin;
dream characterizes it — the regime dream is good at (a cold dream from a poor
start is what performs badly).

**Self-gating.** Returns an empty update — no state change, no checkpoint —
unless *all* hold:

1. `FIT_METHOD_FINAL` is set and differs from `FIT_METHOD` (nothing to gain if
   exploration already used it);
2. there is a dict `current_model` to polish;
3. the SLD-profile check did not veto the selected fit — precise uncertainties on
   a physically impossible model only lend it authority. Checked *before* the χ²
   gate, because the excursion is what buys the low χ² so a vetoed selection
   passes that gate by construction. The verdict comes from
   `final_selection["selected_has_profile_artifact"]`, falling back to the
   selected `FitResult`'s own flag so an imported workspace or a checkpoint
   written before that field existed is not silently exempt. Recorded as
   `final_fit["profile_veto"]`;
4. the selected χ² is finite and ≤ the gate (`FINAL_FIT_CHI2_MAX`, default
   `CHI2_MAX`) — don't spend a long MCMC characterizing a poor fit.

`final_fit` still does not read `chi2_min`, so a sub-floor selection is polished
and adopted — the same gap `_select` has ([issues.md](../issues.md) #11).

**Budget.** `FIT_STEPS_FINAL` (default **10000**) and `FIT_BURN_FINAL` (default
= steps). This is deliberately ~10× the exploration budget: a small step count
gives plausible-looking but meaningless error bars. (See the note in
[`config/providers.yaml`](../validation) of the validation harness — at
`FIT_STEPS=1000`, dream draws only ~7 generations.)

**Adopt-vs-keep (keeps the reported model correct).** The polish fits the *same
structure* finalize chose, so adopting its refined values does not conflict with
the parsimony selection:

- **dream holds/improves** (χ² within `FINAL_SELECTION_TOL` of the selected χ²)
  → **adopt**: write dream values into `current_model`, set `current_chi2` to the
  dream χ², append the dream `FitResult` (uncertainties + chains), and set
  `final_fit['adopted'] = True` (which repoints `problem.json`, §5).
- **dream comes back worse** (degenerate — rare, since dream evaluates the seed
  point) → **keep** the better finalize selection untouched; still append the
  dream `FitResult` so its uncertainties are on record, with
  `final_fit['adopted'] = False`.

**Never fatal.** A fit failure is caught and recorded (`ran=False`); the run
still reports the finalize-selected model.

**Uncertainties live in the `FitResult`** (`result['uncertainties']`) and the
refl1d `-err.json` export, **not** in the ModelDefinition — which has no field
for them, matching how a plain `FIT_METHOD=dream` run already reports them.
`best_model` / `best_chi2` are never written here.

---

## 5. Step 3 — `save_final_state`: the artifacts

[`workflow/checkpoints.py: save_final_state`](../src/aure/workflow/checkpoints.py).

**`final_state.json`** — top level: `final_chi2 = current_chi2` (the reported
χ² — dream's when adopted, else finalize's), plus the full serialized `state`
(`current_model`, `best_model`, `best_chi2`, `fit_results`, `final_selection`,
`final_fit`, …).

**`problem.json`** (`_copy_best_problem_json`) — the serialized bumps
`FitProblem`, copied to the top level so it can never disagree with
`current_model`. Preference order:

1. **the adopted `final_fit` export** (`refl1d_output/final_<method>/`) — when
   the dream polish was adopted, `current_model` carries its values, so
   `problem.json` must track the dream export, not the exploration one;
2. otherwise the **finalize-selected iteration**
   (`refl1d_output/fit_iter{N}_{method}/`);
3. otherwise the best-χ² iteration (legacy states with no `final_selection`);
4. otherwise the lowest-χ² fit outright.

---

## 6. Run directory after finalization

```
<output_root>/<run_name>/
  final_state.json                     # final_chi2 + full state (incl. final_selection, final_fit)
  problem.json                         # serialized winning FitProblem (tracks current_model)
  checkpoints/NNN_finalize.json
  checkpoints/NNN_final_fit.json       # only when the polish ran
  refl1d_output/
    fit_iter{N}_{method}/              # per-iteration exploration exports
    final_{method}/                    # the final MCMC polish: *-err.json (σ), *-chain.mc.gz, …
```

---

## 7. Invariants not to break

1. **Finalization lives in the runner terminal block**, after the node loop —
   the single place every loop exit converges. Keep it there; don't scatter
   end-of-run logic across the per-node routes.
2. **`finalize` selects; it never fits.** `final_fit` fits; it never re-selects
   the structure.
3. **`best_model` / `best_chi2` are the loop's regression baseline** — never
   written by finalize or final_fit.
4. **`problem.json` must track `current_model`.** If `final_fit` adopts, repoint
   `problem.json` to the `final_<method>` export (this is what
   `final_fit['adopted']` drives).
5. **`final_fit` is optional and non-fatal** — inert when `FIT_METHOD_FINAL` is
   unset, and a failure inside must never lose the finalize-selected model.
6. **The web UI sees final results through the checkpoint callback.** The
   terminal block fires `checkpoint_callback(state, "final_fit")` so the live
   view (`web/routes.py`) captures the appended `fit_results` / updated
   `current_chi2`; keep that call if you refactor.

---

## 8. Configuration reference

| Env var | YAML key | Default | Meaning |
|---|---|---|---|
| `CHI2_MAX` | `chi2_max` | `5.0` | Upper end of the χ² acceptance window — the refinement loop's stop condition, enforced deterministically in `evaluation`; pinned into the run state so `aure resume` inherits it. Only ever *raises* a verdict; above it the evaluator LLM decides (§9) |
| `CHI2_MIN` | `chi2_min` | `0.5` | Lower end of the same window. Below it the deterministic stop **stands down** — not a veto: the evaluator's verdict decides. A reduced χ² far under 1 says the residuals are much smaller than the quoted uncertainties (an overestimated `dR` column, or free parameters absorbing the noise), i.e. evidence about the error model rather than the structure, so it must not read as a pass. `0` disables the floor; must be finite and strictly below `chi2_max`. Pinned like `chi2_max` |
| `FIT_METHOD` | `fit_method` | `dream` | Exploration / refinement-loop method |
| `FIT_STEPS` / `FIT_BURN` | `fit_steps` / `fit_burn` | `1000` | Exploration budget |
| `FIT_METHOD_FINAL` | — | *(unset → off)* | Final polish method; must differ from `FIT_METHOD` to run |
| `FIT_STEPS_FINAL` | — | `10000` | Final-polish sample budget |
| `FIT_BURN_FINAL` | — | = steps | Final-polish burn-in |
| `FINAL_FIT_CHI2_MAX` | — | = `CHI2_MAX` | Skip the polish when the selected χ² exceeds this |
| `FINAL_SELECTION_TOL` | — | `0.02` | χ² band for the parsimony tie-break and the adopt-vs-keep decision |

`chi2_max` / `chi2_min` and the four final-fit names are all setup-YAML keys,
applied for one run as env overrides by `analyze`, `prepare` and `batch`.

---

## 9. Sharp edges of the deterministic stop

**A co-refinement is verified only when every state is.** `fitting` reads each
state's `profile.dat` back onto that state's `per_file_results`, and the detector
checks all of them, each against its *own* effective media — per-state
`ambient`/`layers` overrides mean the model-level template describes no state in
particular. An excursion in any state vetoes and names it; a state whose profile
could not be read leaves the *whole* fit unverified, so a partially-exported
co-refinement stands the clamp down rather than being judged on the states that
happened to report.

**The stop is one-directional.** It turns "keep refining" into "stop", never the
reverse. Above `chi2_max` the LLM's `acceptable` stands as-is and *none* of the
clamp's stand-down guards are consulted, so an LLM that accepts a χ² of 4200 on a
profileless fit with a failed (`+inf`) per-state χ² ends the run. Standing down
below `chi2_min` is likewise not a veto — the evaluator may accept a sub-floor
fit and the run completes. The SLD-profile veto remains the only deterministic
check that can lower an accepting LLM's verdict.

**Below `chi2_min` the stopping point is not reproducible.** The floor hands the
decision back to the LLM, so whether such a run ends on iteration 1 or burns its
whole `max_refinements` budget is the evaluator's judgement. That is deliberate:
making the floor a veto would mean a fit whose `dR` genuinely is conservative
could never stop — the exact failure the clamp exists to prevent. The mitigation
is disclosure: the evaluation prompt states the floor and what it implies, the
sub-floor χ² is recorded as an issue on the `FitResult` (so it reaches
`final_state.json`), and the success message repeats it under the headline
number. Set `chi2_min: 0` to accept any χ² at or below `chi2_max`.

**Which fit is *reported* is a separate decision** (§3). The profile veto now
reaches it — vetoed fits are set aside — but the floor does not, so a sub-floor
χ² the clamp refused to accept can still be the reported answer
([issues.md](../issues.md) #11), and the veto still reaches nothing downstream of
the report — it now travels to the CLI report, `--json`, both web tabs,
`final_fit`'s gate and the ISAAC export.
