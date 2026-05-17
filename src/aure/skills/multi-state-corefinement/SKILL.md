---
name: multi-state-corefinement
description: >
  Cross-state co-refinement of neutron reflectometry data — when one sample is
  measured in two or more physical states (solvent contrast, anneal temperature,
  swelling step, applied potential, etc.) and the structural model is shared
  across states. Activated automatically whenever the workflow state has
  ``len(states) > 1``. Covers when to use it, the default tied parameter set,
  how to choose ``shared_parameters`` versus ``unshared_parameters``, and common
  experimental patterns.
metadata:
  author: aure
  version: "1.0"
---

## When this skill applies

Use multi-state co-refinement when the **same physical sample** is measured
under several conditions and you want a single structural model to describe
all of them. Typical patterns:

- **Solvent contrast variation** — the same buried film measured against D₂O
  and H₂O (and sometimes a contrast-matched mixture). Layer thicknesses and
  isotope-independent SLD components are shared; the ambient SLD differs per
  state.
- **Anneal / temperature series** — the same coupon measured before and after
  thermal treatment. Substrate parameters and the underlying capping layer are
  shared; the layer being annealed changes thickness/SLD per state.
- **Swelling / hydration series** — a polymer or hydrogel film exposed to
  changing humidity or solvent. Substrate and dry-component SLDs are shared;
  the film thickness and mixed-layer SLD vary per state.
- **Applied potential / electrochemistry** — the same electrode at different
  potentials. Substrate, electrode, and SEI scaffolding are shared; the
  mobile-species layer (Li, electrolyte ingress, etc.) varies per state.

If the measurements are different segments (incident angles) of one
state, this is **not** the right skill — use the standard multi-segment
co-refinement (`states:` is for *physical states*, not measurement segments).

## How to configure

You can configure a multi-state run two ways:

- **Web UI** (`aure web`, Setup tab): tick *Group files into states*, label
  each file with a state name, pick a ties mode (Auto / Shared / Unshared),
  optionally click *Preview structure* to see the available layer
  parameters as a checklist, and adjust per-state ambient / intensity /
  theta_offset / sample_broadening / back_reflection / extra_description
  overrides from the accordion. The UI submits the same JSON body as the
  CLI/Python path.
- **YAML config** (`aure analyze ... --config`): add a `states:` block to
  the user config:

```yaml
states:
  - name: D2O
    data_files:
      - /data/REFL_12345_combined_data_auto.txt
    extra_description: |
      In D₂O — high-contrast ambient.
  - name: H2O
    data_files:
      - /data/REFL_12350_combined_data_auto.txt
    extra_description: |
      In H₂O — null-reflecting ambient.

# Optional. Whitelist the layer attributes to tie across states.
# Mutually exclusive with `unshared_parameters`.
shared_parameters:
  - Cu.thickness
  - Cu.material.rho
  - Cu.interface
  - substrate.interface
```

Each `state` may also override `ambient`, `intensity`, `theta_offset`,
`sample_broadening`, and `back_reflection`. Per-state partials (multiple
files sharing one `set_id`) are supported just like the single-state
multi-segment workflow.

## Default tied set

When neither `shared_parameters` nor `unshared_parameters` is supplied,
AuRE ties the **structural** parameters of each layer:

- `<layer>.thickness`
- `<layer>.material.rho`
- `<layer>.interface`

plus `substrate.interface`. Everything else (ambient SLD, intensity per
file, theta_offset, sample_broadening) is left **per-state**. The default
tied set is the right starting point for solvent-contrast experiments.

## Choosing `shared_parameters` vs `unshared_parameters`

- **`shared_parameters`** is a **whitelist**: only the listed `Layer.attr`
  pairs are tied across states; everything else is per-state. Use this
  when most parameters should vary independently and only a few are truly
  invariant across states.
- **`unshared_parameters`** is a **blacklist**: starts from the default
  tied set and removes the listed pairs. Use this when most structural
  parameters should be tied and only one or two layers are expected to
  change with state.

The two are mutually exclusive. The user-config choice always wins over
any LLM proposal.

### Common patterns

| Experiment | What to tie | What to leave per-state |
| --- | --- | --- |
| Solvent contrast (D₂O / H₂O / CM) | All layer thicknesses, dry-component SLDs, all interfaces | Ambient SLD, intensity per probe |
| Wet→dry anneal | Substrate, capping layer | Annealed layer thickness *and* SLD |
| Swelling polymer series | Substrate, dry interlayer | Swelling layer thickness, mixed-SLD |
| Electrochemistry (Li-ion) | Substrate, electrode, SEI scaffold | Mobile-species layer thickness/SLD |

## What the workflow does

1. **Intake** groups data files by state, validates per-state set_ids,
   and loads Q/R/dR + headers (theta, dq_is_fwhm).
2. **Modeling** carries the user-supplied tie set verbatim, validates
   that every `Layer.attr` resolves to a known layer, and surfaces a
   clear error otherwise. It also injects per-state context
   (`name`, `extra_description`, ambient SLD) into the LLM prompt.
3. **Fitting** builds one `bumps.fitproblem.FitProblem` with one refl1d
   `Sample` per state. Cross-state ties are realized by aliasing the
   shared `bumps.Parameter` objects across samples. Per-file
   `PerFileFitResult` entries are emitted, each tagged with the state
   name. Per-state `profile.dat` is written under
   `output_dir/refl1d_output/fit_iter{i}_{method}/state_<name>/`.
4. **Evaluation** runs residual-fringe analysis per state, surfaces
   boundary hits with state-prefixed names for untied parameters, and
   reverts to the previous best `ModelDefinition` (states + tie set
   included) on χ²/BIC regression.

## Diagnostics

- Per-state χ² is logged at fit completion and is available on each
  `PerFileFitResult["chi_squared"]`.
- The deduplicated free-parameter count (used in BIC) is available on
  the aggregate `FitResult["_n_free_params"]`.
- Per-state residual fringe analysis lives on
  `latest_fit["per_state_residual_analysis"]`, keyed by state name.

If only one state shows residual fringes, the issue is almost always in
its **untied** parameters (per-state ambient, mixed SLD, or per-state
layer thickness for a swelling/annealing layer).
