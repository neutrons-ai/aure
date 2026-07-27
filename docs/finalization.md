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

- **Selection rule:** lowest χ², with a **parsimony tie-break** — among fits
  within `FINAL_SELECTION_TOL` (default 2%) of the lowest χ², prefer the fewest
  free parameters, then the earliest iteration. (BIC's idea without depending on
  the stored `bic`.)
- Resolves the *ModelDefinition* that produced the winning iteration via
  `model_history` (handles interactive rewinds and bounds-only re-fits, which
  append a `fit_results` entry with no history entry).
- Writes that iteration's fitted parameter values **back into** the definition
  (`_apply_fitted_values`) → the promoted `current_model`.
- Sets `current_model` and `current_chi2` to the winner.
- Records `final_selection` (index, iteration, χ², tie-break metadata) for audit.
- **Never writes `best_model` / `best_chi2`** — those are the loop's regression
  baseline that `aure resume` compares against.
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
3. the selected χ² is finite and ≤ the gate (`FINAL_FIT_CHI2_MAX`, default
   `CHI2_MAX`) — don't spend a long MCMC characterizing a poor fit.

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

Env vars (also settable as setup-YAML keys via `_build_env_overrides`, mapped
`fit_method_final → FIT_METHOD_FINAL`, etc.):

| Env var | YAML key | Default | Meaning |
|---|---|---|---|
| `FIT_METHOD` | `fit_method` | `dream` | Exploration / refinement-loop method |
| `FIT_STEPS` / `FIT_BURN` | `fit_steps` / `fit_burn` | `1000` | Exploration budget |
| `FIT_METHOD_FINAL` | `fit_method_final` | *(unset → off)* | Final polish method; must differ from `FIT_METHOD` to run |
| `FIT_STEPS_FINAL` | `fit_steps_final` | `10000` | Final-polish sample budget |
| `FIT_BURN_FINAL` | `fit_burn_final` | = steps | Final-polish burn-in |
| `FINAL_FIT_CHI2_MAX` | `final_fit_chi2_max` | = `CHI2_MAX` | Skip the polish when the selected χ² exceeds this |
| `FINAL_SELECTION_TOL` | — | `0.02` | χ² band for the parsimony tie-break and the adopt-vs-keep decision |
