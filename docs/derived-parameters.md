# Derived parameters: fitting a combination instead of a coordinate

This document covers `derived_parameters` — how to declare a functional
relationship between fit parameters, why you would, and what it changes about a
run. If you are launching AuRE with a data file and a sentence of English and
you want to say *"the surface excess is about 1.2 mg/m²"* or *"the film is 30%
polymer by volume"*, this is where that goes.

> **TL;DR** — Set `allow_derived_parameters: true` and add a top-level
> `derived_parameters:` block to a setup YAML, then pass it with
> `aure analyze -c setup.yaml`. Each entry makes one **combination** a free fit
> parameter and derives a raw parameter from it. There is no CLI flag and no
> constraints-only side-file: the block lives in a setup YAML, which means your
> data file has to move into that file's `states:` block too.

---

## 1. Why bother

Reflectivity does not constrain the parameters a model is written in; it
constrains certain *combinations* of them. A thin layer's
`(ρ_layer − ρ_ambient) · thickness` is pinned tightly while the SLD and the
thickness separately are not — the likelihood is a long narrow ridge, and an
optimizer has to crawl along it. AuRE has machinery that exists purely to fight
this: the thin-layer SLD **mode enumeration** in the fitting node, and the
`thin-layer-degeneracy` skill.

Independent knowledge has the same shape. A QCM-D measurement gives you an
adsorbed amount, not an SLD. A known polymer density plus a swelling
measurement gives you a volume fraction, not a thickness. Putting a bound on
`SEI.rho` when what you actually know is the excess is both too weak (any SLD
inside the box is equally good) and too strong (just outside it is impossible).

Reparametrizing fixes both at once. Fit the excess; derive the SLD from it. The
ridge is gone from the sampling geometry, and an ordinary `min`/`max` range on
the excess means exactly what you wrote.

## 2. Turning it on

**Reparametrization is off by default.** It asks more of the model driving the
run than a plain layer stack does — derived layer attributes are not fit
parameters, their numbers are computed rather than fitted, and the layers they
reference must not be refined away. A model that does not hold all of that in
mind will "correct" a derived SLD or refine the layer away, and the run
degrades in a way that is hard to read from the outside. So the whole feature —
the config key, the refinement prompt rule, the `functional-constraints` skill
— stays out of the way until it is asked for.

Enable it per run in the setup file:

```yaml
allow_derived_parameters: true
```

or in the environment, for a whole session:

```console
$ export ALLOW_DERIVED_PARAMETERS=1
```

The config key wins when both are set. Declaring `derived_parameters` without
enabling the feature is a **hard error**, not a silent skip — ignoring the
block would fit a model measurably different from the one described, with
nothing in the report to say why:

```console
$ aure analyze -c sei.yaml
  Setup error: sei.yaml: derived_parameters (Gamma_SEI) are declared, but
  reparametrization is off by default because it asks more of the model than a
  plain layer stack does. Enable it with `allow_derived_parameters: true` in
  this file, or ALLOW_DERIVED_PARAMETERS=1 in the environment.
```

With the flag off, nothing about this feature reaches any prompt: the
refinement rule is rendered only for a model that actually carries a block, and
the skill is stripped from the active set even if the selector picks it.

## 3. The shape of a declaration

```yaml
derived_parameters:
  - name: Gamma_SEI
    free: {init: -420, min: -600, max: -250}
    assign:
      SEI.rho: "dTHF.rho + Gamma_SEI / SEI.thickness"
    keep_physical: ["SEI.rho > -1.0", "SEI.rho < 6.4"]
    source: "QCM-D, 1.2 mg/m² at this coverage"
```

Read it as: *`Gamma_SEI` is a new free parameter ranging over −600…−250; the
SEI's SLD is no longer fitted, it is whatever this expression says; and the
result must stay between −1.0 and 6.4.*

| Key | Meaning |
|-----|---------|
| `name` | The new free parameter. Must not collide with a layer or material name. |
| `free` | `{init, min, max}`. `min`/`max` are **required** — a derived parameter has no bounds of its own to fall back on. |
| `assign` | `"<layer>.<attr>": "<expression>"`. Every raw parameter named here **stops being fitted**. |
| `keep_physical` | Guards on the *derived* value. Single comparisons (`< <= > >=`); write two entries rather than chaining. |
| `source` | Where the number came from. Not enforced, but a constraint that moves the answer should be auditable. |
| `tied` | Multi-state only, default `true`. See §5. |
| `states` | Multi-state only: apply in these states only (default: all). |

**Nothing does algebra.** You write the inverse yourself. AuRE will not take
`(SEI.rho - dTHF.rho) * SEI.thickness ≈ Γ₀` and solve it for you — you say which
parameter becomes the dependent one and what it equals. This is deliberate: it
removes every ambiguity about which coordinate was eliminated.

### What expressions may contain

Numbers; `<layer>.<attr>` names; unary minus; `+ - * / **`. That is all.
Attributes are `thickness`, `interface`, and `rho` (`material.rho` is accepted
as a synonym, matching the tie-spec spelling); `substrate` and `ambient` work as
aliases for the corresponding material names. Anything else — a function call,
an index, a comprehension, `%` — is rejected before evaluation. These strings
are run through a whitelisted AST walker
([`nodes/expressions.py`](../src/aure/nodes/expressions.py)), never `eval`,
because they can arrive from a config file you did not write.

## 4. How you actually launch it

**There is no `--derived-parameter` flag.** The block lives in a setup YAML, so
the question "how does someone running `aure analyze data.txt "..." -h "..."`
declare one?" has a specific answer: they write a small setup file and move the
data file into it.

A constraints-only side-file **will not work** — `load_setup` requires at least
one state:

```console
$ aure analyze data.txt "SEI on Li" -c only_constraints.yaml
  Setup error: only_constraints.yaml: at least one state must be declared under `states:`.
```

So the smallest working version of the "one dataset + a sentence" workflow is:

```yaml
# sei.yaml
allow_derived_parameters: true

sample_description: |
  Lithium metal on copper, measured in deuterated THF electrolyte after
  cycling. An SEI layer is present on the lithium.
hypothesis: the SEI accounts for the excess QCM-D measured at this coverage

states:
  - name: cycled
    data_files: [REFL_226642_combined_data_auto.txt]

derived_parameters:
  - name: Gamma_SEI
    free: {init: -420, min: -600, max: -250}
    assign:
      SEI.rho: "dTHF.rho + Gamma_SEI / SEI.thickness"
    keep_physical: ["SEI.rho > -1.0", "SEI.rho < 6.4"]
    source: "QCM-D, 1.2 mg/m²"
```

```console
$ aure analyze -c sei.yaml
```

The positional arguments still work alongside `-c` and take precedence, so you
can keep iterating on the wording from the shell while the constraint stays in
the file:

```console
$ aure analyze -c sei.yaml -h "the SEI is thinner than the QCM-D excess implies"
```

`--extra-data` is refused when the setup declares `states:` — put every file in
the states block.

## 5. The layer has to exist first

`assign` names layers of the **baseline model**, which for a text-driven run is
whatever the intake LLM parsed out of `sample_description`. Two consequences,
both of which fail loudly rather than silently:

**The name must match.** If you write `SEI.rho` and the parse produced a layer
called `SEI layer`, the run stops at the modeling node with

```
Reparametrization (derived_parameters): derived parameter 'Gamma_SEI':
assignment target 'SEI.rho' names no layer; known: ['Li', 'ambient', 'copper', 'dTHF', 'substrate']
```

To make the name certain rather than hoping, declare the stack explicitly in
the state (see the per-state `layers:` block in
[`aure_config.example.yaml`](../aure_config.example.yaml)) instead of leaving it
to the description.

**A merely *hypothesised* layer is not in the baseline.** AuRE deliberately
keeps tentative layers out of the initial model — "we expect an SEI to form"
produces a ranked structural *hypothesis*, not a layer — so a derived parameter
referring to that layer fails at startup. If you want to constrain a layer's
excess, the layer must be part of the model you start from, not one the
refinement loop might add later. This is a real limitation, not a subtlety of
phrasing: today you cannot say "*if* an SEI is added, constrain it this way".

A refinement that removes a referenced layer is handled differently, because
there the mistake is the workflow's rather than yours: the affected declaration
is **dropped** and the run continues, exactly as a cross-state tie whose layer
disappeared is pruned. The transcript records it —

```
Dropped reparametrization 'Gamma_SEI': assignment target 'SEI.rho' names no
layer — the structural change removed what it was written against
```

— and an auxiliary parameter left with nothing referencing it goes with it. The
refinement prompt tells the model not to remove such a layer in the first place
(rule 14, rendered only for models that have a block), so this is the backstop
rather than the expected path. You lose the constraint, not the run.

## 6. Multi-state: the case this was built for

In a co-refinement, a declaration is **tied across states by default** — one
free parameter for the whole problem, with `assign` re-evaluated in each state's
own namespace.

That is not a convenience; it is the only way to state what solvent-contrast
variation actually assumes. A solvated film's SLD is
`φ·ρ_dry + (1−φ)·ρ_solvent`, so it **differs in every contrast**. Tying
`film.material.rho` across contrasts asserts something false. Untying it throws
away the coupling that made the contrast series worth measuring. The invariants
are `φ` and `ρ_dry`, and neither is a layer attribute, so `shared_parameters`
cannot name them:

```yaml
states:
  - name: dTHF
    data_files: [REFL_1001_combined_data_auto.txt]
    ambient: {rho: 6.35}
  - name: hTHF
    data_files: [REFL_1002_combined_data_auto.txt]
    ambient: {rho: 0.18}

derived_parameters:
  - name: phi                                  # shared across states
    free: {init: 0.30, min: 0.05, max: 1.0}
    assign:
      film.rho: "phi * rho_dry + (1 - phi) * solvent.rho"
    keep_physical: ["film.rho > -1.0", "film.rho < 6.5"]
  - name: rho_dry                              # auxiliary: free, used above
    free: {init: 2.0, min: 0.5, max: 4.0}
```

gives one `phi`, one `rho_dry`, and a different `film.rho` in each state:

```
dTHF: solvent = 6.35  ->  film.rho = 5.045
hTHF: solvent = 0.18  ->  film.rho = 0.726
```

`rho_dry` above is an **auxiliary** parameter: free, no `assign` of its own,
reaching the model only through another entry's expression. That is allowed
precisely because two-parameter reparametrizations like this one need it. A
declaration that assigns nothing *and* is referenced by nothing is rejected —
it would be a free parameter the data cannot see.

`film.rho` is removed from cross-state tie aliasing automatically, at **both**
ends, so no `shared_parameters` / `unshared_parameters` edit is needed. Use
`tied: false` for one copy per state (named `"<state> phi"`), and
`states: [dTHF]` to apply a declaration in a subset of states.

> ⚠️ **Δρ·t is not the contrast invariant for a solvated layer.** The surface
> excess form in §2 is invariant only for a non-exchanging adsorbate whose own
> SLD does not change with the solvent. For a layer the solvent penetrates, use
> the volume-fraction form above. Getting this backwards produces a model that
> fits one contrast and fights the other.

## 7. What it changes about the run

- **Free-parameter count / BIC.** A one-for-one swap (excess for SLD) is
  BIC-neutral; solvation (two free, one derived) costs one parameter. Handled
  by `_derived_param_delta` in the evaluation node, so the model-complexity
  comparison stays honest.
- **Reported χ² is the data term only.** `FitProblem.chisq()` scales the model
  misfit *plus* the prior and constraint penalties, and bumps returns a model
  term of `0.0` when a constraint fails — so a violated guard would otherwise be
  reported as a perfect fit, and would land under the acceptance floor as if the
  error bars were wrong. `model_builder.data_chisq` reports the data term and
  returns `inf` for an infeasible point instead. The penalty still drives the
  optimizer; it is just never counted as goodness of fit.
- **`problem.json` export is refused.** `aure prepare` / `save_problem_json`
  raise rather than export, because bumps serialization does not round-trip
  expression parameters and the exported file would quietly be a *different*
  model — derived parameters back to free, constraints gone.
- **Refinement keeps the block.** The modeling LLM re-emits the whole model and
  knows nothing about `derived_parameters`, so the node carries it across
  iterations explicitly (your config wins). Without that, the first refinement
  would silently revert to the raw coordinates.
- **The web Setup tab does not carry it.** `aure serve`'s Load/Save round-trip
  reads a setup into a form that has no field for `derived_parameters`, so
  loading a YAML there and exporting it again **drops the block**. Edit these
  files by hand, or keep the authoritative copy outside the UI.

## 8. Where the pieces live

| Piece | Location |
|-------|----------|
| Schema + field docs | `DerivedParameter` in [`state.py`](../src/aure/state.py) |
| Expression evaluator | [`nodes/expressions.py`](../src/aure/nodes/expressions.py) |
| Application + constraints | `apply_derived_parameters`, `validate_derived_parameters` in [`nodes/model_builder.py`](../src/aure/nodes/model_builder.py) |
| Data-only χ² | `data_chisq` / `penalty_nllf`, same file |
| YAML shape check | `_parse_derived_parameters` in [`config.py`](../src/aure/config.py) |
| Worked examples | [`aure_config.example.yaml`](../aure_config.example.yaml) |
| Tests | [`tests/test_derived_parameters.py`](../tests/test_derived_parameters.py) |
