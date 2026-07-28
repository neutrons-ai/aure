# How AuRE Works — A Guide for New Users

This document explains what AuRE does and *why it does it the way it does*,
from the ground up. It assumes no prior knowledge of reflectometry and no
prior experience with large language models (LLMs). It is longer than a
typical README section on purpose: it is meant to be the piece of writing you
read once to understand the whole system, the kind of write-up you would
expect to find in the appendix of a methods paper.

If you already know reflectometry and LLMs and just want to understand the
changes that were introduced in the "skill-driven hypothesis" version of the
refinement loop, skip to §6 ("The refinement problem and how AuRE solves
it").

---

## 1. The scientific problem

**Neutron reflectometry** is a technique for measuring the thickness, density,
and roughness of thin films — typically films that are a few nanometres to a
few hundred nanometres thick, deposited on a flat substrate such as a silicon
wafer. It works by firing a beam of neutrons at the sample at a very shallow
angle and measuring what fraction of the beam bounces back (is *reflected*)
at each angle or wavelength.

The raw measurement is a curve: reflectivity $R$ as a function of the
momentum transfer $Q$ (in units of inverse Ångström, Å⁻¹). $Q$ encodes how
steeply the neutrons are probing: small $Q$ probes length scales of hundreds
of Ångström, large $Q$ probes single-Ångström features. A typical curve
starts near $R \approx 1$ at very small $Q$ (*total external reflection*),
drops sharply at a *critical edge*, and then decays over many orders of
magnitude, often showing oscillations called *Kiessig fringes* whose spacing
encodes the total film thickness:

$$R(Q) \approx \text{(some function of how the scattering-length density varies with depth)}$$

The key inversion problem is:

> Given a 1D reflectivity curve $R(Q)$, reconstruct the depth profile of
> scattering-length density (SLD), $\rho(z)$, along the sample normal.

This inversion is **not unique**. Many SLD profiles fit the same curve
equally well. So in practice one does not freely invert the data — one
builds a *physical model* of the sample (a stack of homogeneous layers, each
with thickness, SLD, and roughness) and fits its parameters. Choosing *which
layers to put in the model* is where the human expertise lives. A model with
too few layers will fit poorly; a model with too many layers will fit well
but be statistically unjustified and have parameters that cannot be
distinguished from each other.

**In short**, reflectometry fitting is two intertwined problems:

1. A **structural** problem — which layers does the sample have? which are
   missing from the model? which extra ones are redundant?
2. A **parameter** problem — given a layer structure, what are the
   thicknesses, SLDs, and roughnesses? Are they all within physically
   reasonable bounds?

Standard fitting software ([Refl1D](https://refl1d.readthedocs.io),
[MOTOFIT](http://motofit.sourceforge.net), BornAgain, …) is very good at
solving the parameter problem once you have picked a structure. It does not
solve the structural problem for you.

---

## 2. What AuRE is

AuRE is an *agent* that solves both problems together for a given dataset.
It takes two inputs from the scientist:

- the reflectivity data file, and
- a plain-English description of the sample (e.g. *"100 nm polystyrene on
  silicon in air"*, or *"50 nm copper on 5 nm Ti on Si in D2O"*),

and it produces a fitted Refl1D model. Internally, it runs a loop that
alternates between **deciding what the model should look like** (a task
historically done by the scientist) and **running the optimizer to fit its
parameters** (a task done by Refl1D). Between iterations it evaluates the
fit and decides whether to stop or to modify the model.

The LLM does the "thinking" parts — parsing the sample description,
proposing model structures, judging the fit, deciding what to change next.
The non-LLM parts — reading data files, computing chi-squared, running the
Markov-chain optimizer — are ordinary Python code.

This is sometimes called an *agentic* workflow. What makes it an agent,
rather than just a one-shot LLM prompt, is that the LLM sees the *results
of its previous actions* (fit quality, residuals, parameter values at
bounds) and chooses its next action in response. AuRE runs this loop as a
hand-written state machine ([`workflow/runner.py`](../src/aure/workflow/runner.py)),
which passes state between steps and check-points progress.

---

## 3. A one-minute LLM primer

An **LLM** (large language model) is a program that, given some text, can
produce more text that looks like a plausible continuation. Modern LLMs —
GPT-4, Claude, Gemini, Llama, and so on — have been trained on enormous
amounts of scientific text and can, when asked, explain concepts, follow
instructions, generate structured data (like JSON), and reason about simple
numerical and physical problems.

For this project, you need to remember four things:

1. **An LLM is just a function from text to text.** We send it a prompt, we
   get back a response. We re-invoke it for each step of the workflow.
2. **LLMs can be asked for structured output.** We tell the model "respond
   with a JSON object matching this schema" and parse the result. Much of
   AuRE's machinery is about phrasing questions in a way that produces a
   *reliable* structured answer.
3. **LLMs do not remember anything between calls.** Every call is
   independent. If we want the model to "know" what happened in the
   previous iteration, we must explicitly include it in the new prompt.
4. **LLMs are easy to mislead with phrasing.** If a prompt says "try
   parameter tweaks first, structural changes as a last resort", the model
   will dutifully try parameter tweaks for a long time before giving up —
   even when the right answer is structural. (This is one of the key
   lessons in §6.)

AuRE treats the LLM as *one expert reviewer on a panel*. The expert is
opinionated and knowledgeable but not infallible. The rest of the system —
guardrails, state tracking, checkpoints — exists to keep the expert honest.

---

## 4. The AuRE workflow

AuRE's core is a state machine with five nodes. The state is a Python
dictionary ([`ReflectivityState`](../src/aure/state.py)) that accumulates as
the workflow proceeds.

```mermaid
flowchart LR
    S((Start)) --> Intake
    Intake --> Analysis
    Analysis --> Modeling
    Modeling --> Fitting
    Fitting --> Evaluation
    Evaluation -->|acceptable| E((Done))
    Evaluation -->|refine model| Modeling
    Evaluation -->|bounds only| Fitting
    style S fill:#6c757d,color:#fff,stroke:none
    style E fill:#198754,color:#fff,stroke:none
    style Intake fill:#0d6efd,color:#fff,stroke:none
    style Analysis fill:#0d6efd,color:#fff,stroke:none
    style Modeling fill:#0d6efd,color:#fff,stroke:none
    style Fitting fill:#0d6efd,color:#fff,stroke:none
    style Evaluation fill:#fd7e14,color:#fff,stroke:none
```

### 4.1 Intake

The **intake** node does three jobs:

1. **Load the data file** from disk and validate it (finite $Q$ and $R$,
   positive errors, sensible ranges).
2. **Parse the sample description** with an LLM call. The response is a
   structured JSON object: substrate, list of layers (each with a starting
   thickness, SLD, roughness, and initial bounds), ambient medium, and any
   constraints the user mentioned (e.g. *"the Cu layer is pinhole-free"*).
   SLDs are looked up from the `periodictable` library when the material is
   a simple formula; otherwise the LLM supplies a literature value and
   plausible bounds. The baseline layer list contains only layers the user
   states are *present*; tentative or "expected" layers (*"there may be an
   oxide on top"*) are deliberately **kept out of the baseline** and instead
   become high-priority entries in the hypothesis list below.
3. **Generate a ranked list of structural hypotheses** (see §6). This is a
   second LLM call, guided by an always-on skill called
   [`structural-hypothesis-ranking`](../src/aure/skills/structural-hypothesis-ranking/SKILL.md).
   The output is a list of candidate structural changes that the refinement
   loop should consider if the initial model does not fit well — for
   example, *"add a 10–30 Å native CuO between the Cu and the D₂O"*. The
   user's own hypothesis (the `-h` flag) is folded into this list as one or
   more **top-ranked** entries tagged `origin="user"`, reworded to fit the
   hypothesis shape; skill-enumerated entries are tagged `origin="skill"`.

The result of intake is stored as `parsed_sample`, the initial
`current_model`, and `structural_hypotheses` in the workflow state.

### 4.2 Analysis

The **analysis** node runs ordinary Python (no LLM) over the data to
extract physics features:

- the **critical edge** $Q_c$, from which the substrate SLD can be
  cross-checked;
- the **estimated total thickness**, from the Kiessig fringe spacing;
- the **estimated surface roughness**, from the high-$Q$ decay rate;
- an **estimated number of layers** based on how many distinct fringe
  frequencies are present in the Fourier spectrum of $R(Q)Q^4$.

These features are later included in LLM prompts so the model has
data-derived numbers to compare against its own guesses.

### 4.3 Modeling

The **modeling** node is the LLM's main creative step. On the first
iteration it simply confirms the intake-parsed model (possibly adjusting
starting values to match the analysis features). On subsequent iterations
it *refines* the model in response to the previous fit.

The modeling prompt contains:

- the current model as JSON,
- the best-fit parameters, χ², and convergence status from the last fit,
- the physics features from analysis,
- the residual-fringe analysis from evaluation (see §4.5),
- the **active skills' content** (see §5),
- the **structural hypotheses list** with statuses,
- the evaluator's explicit direction (`next_action` — either
  `parameter_tweak` or `structural_change` — and, if the latter, a
  `proposed_hypothesis_id` pointing into the hypothesis list).

The response is a complete JSON model definition. The node validates it,
preserves immutable fields (like the data-file path), and writes it back
into the state. If the LLM output is invalid JSON, the node falls back to
simply widening all the bounds and trying again — a safety net that
prevents a malformed response from derailing the run.

### 4.4 Fitting

The **fitting** node runs [Refl1D](https://refl1d.readthedocs.io)'s
optimizer (differential evolution followed by, optionally,
DREAM/MCMC) on the current model. It returns best-fit parameter values,
uncertainties, a final χ², the residuals, and whether the fit converged.
This is ordinary numerical optimization — no LLM involved.

Optionally (env `MODE_ENUMERATION=1`, single-file fits), the node runs a
**thin-layer SLD mode enumeration** step *before* the main fit. Layers thinner
than the resolution limit $2\pi/Q_\text{max}$ sit on a "contrast × thickness"
degeneracy ridge (§6.8): several very different SLD values fit almost equally
well, in separate local minima that a single optimizer run — even a global one
— will not cross. The step re-seeds each thin layer's SLD across a few discrete
levels spanning its allowed range, cheaply polishes each, and starts the main
fit from the lowest-χ² basin. It is off by default, logs every seed it tries
and the basin it picks, and never aborts the fit on error.

### 4.5 Evaluation

The **evaluation** node is the decision-maker. It does the following (the
numbering is topical, not chronological — `evaluation_node` runs 1 → 2 → 4 → 3 →
6, then, on the refining branch only, 5 → 7):

1. **Auto-expand stuck bounds.** If any fitted parameter ended at its
   bound, the bound is widened by a factor (e.g. ×1.5 outward) and the
   issue is logged. This is a purely mechanical step that saves the LLM
   from having to ask for it. The widened model is *computed* here but only
   **adopted** on the refining path (after step 6): a run that accepts and stops
   would otherwise report bounds nothing ever explored, so on that path the
   issue text says the parameter is pinned instead of claiming a re-fit.
2. **Residual-fringe analysis.** A small Fourier analysis of the
   residuals; if it shows a clear oscillation at a characteristic
   thickness, that thickness is reported to the LLM as a likely "missing
   layer" hint.
3. **SLD-profile artifact detection.** A χ²-optimal fit can still produce a
   *physically impossible* SLD profile — most often when a roughness's
   error-function tail reaches across a thin layer and the profile dips below
   (or overshoots above) the range its neighbouring materials can produce, e.g.
   a dip *below the substrate SLD* just before the substrate. This defect is
   invisible in χ² (it only shows up in the depth profile), so a deterministic
   check (`detect_profile_artifacts`) inspects the fitted profile for interior
   extrema that sit at an SLD no material in the stack provides. A genuine
   excursion is treated as a deterministic guardrail: it **vetoes acceptance**
   (overriding the LLM if it judged the fit acceptable on χ² alone) and becomes
   an issue that routes the loop back to refinement, with a
   two-branch suggestion — either tie the offending roughness to its layer
   thickness (§6.8), or, if the diffuse transition is intended, keep the
   roughness free and treat those slabs as a *profile parametrization* rather
   than discrete layers. The σ/thickness ratio itself is reported only as an
   informational concern, never an error, because a large roughness is
   legitimate under the parametrization reading (§6.8).
   On a co-refinement only `states[0]`'s profile is available to this check — the
   per-state `profile.dat` files are written but never read back
   ([issues.md](../issues.md) #4) — so a *veto* can still fire but a clean bill of
   health cannot be given, which is what makes step 6's stop inert there.
4. **Call the LLM for a judgement.** The LLM sees the fit result, the
   features, the residual analysis, the **χ² and BIC trajectory across
   all previous iterations**, and the **hypothesis list with statuses**.
   It responds with a JSON object containing `acceptable` (bool),
   `issues`, `suggestions`, `next_action`, and optionally a
   `proposed_hypothesis_id`.
5. **Apply guardrails.** Two statistical guardrails run
   *independently of the LLM's opinion*:
   - **χ² regression guardrail.** If the current χ² is significantly
     worse than the best χ² seen so far, the node reverts the
     `current_model` to the best one and injects an issue telling the
     next modeling step to try a different direction.
   - **BIC regression guardrail.** Adding a layer will almost always
     lower χ² because it gives the fit more freedom. What we actually
     want is a model that is *statistically justified*, measured by the
     Bayesian Information Criterion:
     $$\mathrm{BIC} = k \ln n + \chi^2$$
     where $k$ is the number of free parameters and $n$ is the number of
     data points. If the most recent iteration added parameters and the
     BIC went up despite χ² going down, the added complexity was not
     worth it. The node reverts to the best-BIC model and marks the
     hypothesis that was just tried as `rejected`.

6. **Clamp acceptance to the χ² acceptance window.** `CHI2_MAX` (or the setup's
   `chi2_max:`) is the run's contract with the user, so a **finite χ² inside the
   window `chi2_min ≤ χ² ≤ chi2_max` forces `acceptable = True`** and the run
   completes — otherwise the loop can spend its whole budget re-litigating a fit
   that already passed, at the LLM's discretion and irreproducibly. The LLM's
   objections are not discarded: they stay in `issues` and are reported as notes,
   and the hypotheses the run never got to are listed by the finalize node. An
   interactive run also still gets its review pause on a *clamped* accept (keyed
   off `state["chi2_clamp_accepted"]`) — the one verdict where code overrode an
   objecting evaluator is the one a human should see; feedback typed there is not
   yet acted on ([issues.md](../issues.md) #13).

   The clamp is **one-directional — a floor on stopping, not a ceiling.** It only
   raises a verdict (`False → True`); it never lowers one. *Above* `chi2_max` the
   LLM's `acceptable` is taken as-is and none of the stand-down conditions below
   are even evaluated, so an LLM that accepts a χ² of 4200 on a profileless fit
   with a failed (`+inf`) per-state χ² ends the run. Step 3's profile veto is the
   only thing that lowers a verdict.

   The clamp is deliberately the **last step that can change the verdict**
   (everything after it only *reads* the settled verdict), and it stands down in
   four cases — the first three a defect the aggregate χ² cannot see, the fourth a
   regime in which χ² is not evidence about the structure at all:
   - step 3 **vetoed** acceptance: a physically impossible profile is invisible
     to χ² and must never be accepted on χ² alone;
   - step 3 did not reach a trustworthy answer, which `_profile_checked` states
     positively: the fit carries no exported SLD profile (refl1d writes one only
     when the run has an output directory — an ad-hoc `run_analysis(...)`, MCP's
     `co_refine_states` without `output_dir`, or `quick_analyze`, which has no
     such parameter), the detector declined the profile it has (too few points,
     mismatched `z`/`rho` lengths, a non-finite sample, or a zero SLD span across
     the media), **or the fit is multi-state** — only `states[0]`'s profile is
     read back, so a co-refinement is never verified and **the χ² stop is inert
     there** ([issues.md](../issues.md) #4). "Not checked" is treated as unsafe,
     not as clean, so the LLM's verdict decides, exactly as it did before the
     threshold became binding;
   - a **per-file / per-state χ²** is above the threshold, or carries the `+inf`
     "fit failed" sentinel: the reported χ² is `problem.chisq()` averaged over
     every model of a co-refinement, so one completely unfitted contrast can hide
     under a passing aggregate;
   - χ² is **below `chi2_min`** (`CHI2_MIN` / the setup's `chi2_min:`, default
     `0.5`, `0` disables it, validated finite and required strictly below
     `chi2_max`). A reduced χ² that far under 1 is not a better answer: it says
     the residuals are much smaller than the quoted uncertainties, which in
     reflectometry almost always means the `dR` column is overestimated or the
     model carries enough free parameters to absorb the noise. That is evidence
     about the **error model**, not about the structure, so it must not force
     acceptance. The default is deliberately the same number
     `_simple_evaluation` has always called "Possible overfitting" — the
     heuristic now reads the configured floor, so the two can no longer
     contradict each other. (The `neutron-reflectometry` skill still quotes the
     literal 0.5 in its guidance table.)

   **A stand-down is not a veto.** In all four cases the clamp merely declines to
   *force* acceptance; the evaluator LLM's verdict then decides, as it did before
   the clamp existed. That matters most for the floor: a dataset whose `dR`
   genuinely is conservative can produce a low χ² on a correct model, and vetoing
   there would re-introduce the endless refinement the clamp was added to stop. So
   the floor hands the decision back *with the reasoning attached* — the prompt
   gains an acceptance-floor block, the node records the finding in
   `analysis["issues"]` (copied onto the `FitResult`, hence `final_state.json`),
   and the success message repeats the caveat under the headline χ².

   **Ordering invariant:** the clamp must stay *below* the artifact check (step 3)
   — it decides by reading the two flags that check leaves behind. Hoisting it
   above leaves both unset at clamp time, and "not checked" means stand down: the
   clamp becomes dead code and the χ² stop silently disappears, while still
   type-checking and passing a smoke test. The same reorder turns into the
   *opposite* failure — accepting impossible profiles — the moment anything sets
   `_profile_checked` earlier than the check itself, which is why that flag must
   stay the check's own positive statement.
7. **Revise the hypotheses when the evidence demands it (gated).** When
   the fit is not acceptable *and* there is a concrete signal that the
   intake-time hypothesis list may be incomplete — residual fringes
   pointing to a missing layer, χ² stalled for two or more iterations, or
   no `pending` hypotheses left — the node runs a second LLM call. It first
   re-selects skills using the *observed artifacts* (so a skill that was
   not obvious from the static description — say `sei-layer-analysis` — can
   activate mid-run), then asks the LLM to propose genuinely new hypotheses
   and re-rank the whole list. New entries are tagged `origin="evaluation"`.
   This is the only place besides intake that may *grow* the list, and the
   trigger is a cost-gate — it decides only whether the call is worth
   making, never what the answer should be (see §6.5).

The evaluator's LLM may *suggest* a revert, but the guardrails *enforce*
it deterministically — and, since step 6, **stopping is partly code's decision
too**: the LLM's `acceptable` is advisory *once χ² meets the threshold* (the
clamp raises it) and whenever the profile check fires (the veto lowers it).
Everywhere else — in particular for every fit above the threshold — `acceptable`
is still the LLM's alone. This is a deliberate division of labour: the LLM
reasons about the shape of the fit, and code enforces the statistical invariants
and the run's stop contract.

The deterministic steps are **an ordered chain, not independent post-LLM
cleanup**: the profile veto (3) is applied first, then the χ² clamp (6) reads its
outcome; the regression reverts (5) run only on the path where the verdict is
still "not acceptable". See [architecture.md](../architecture.md) §6 before
reordering them.

After evaluation, the router decides the next step. There are three
possible transitions:

- **`complete`** — the fit is acceptable, or the iteration budget is
  exhausted;
- **`fitting`** — the only issue was a bound hit and no LLM refinement is
  needed (this saves an iteration; see §6.4);
- **`modeling`** — the normal refinement path.

---

## 5. Agent Skills

An **Agent Skill** is a Markdown file with YAML frontmatter that encodes
domain expertise. Each skill lives in its own directory under
[`src/aure/skills/`](../src/aure/skills/). The frontmatter tells AuRE when
to activate the skill (by matching against the sample description); the
body is prose that is injected directly into the LLM's prompt whenever the
skill is active.

For example, the [`metal-oxide-interfaces`](../src/aure/skills/metal-oxide-interfaces/SKILL.md)
skill contains domain facts like *"copper exposed to air or water forms a
10–50 Å native CuO with SLD ≈ 5.0×10⁻⁶ Å⁻² and roughness 3–15 Å"*. When a
sample description mentions copper, this skill is activated and its text
is prepended to the modeling, evaluation, and refinement prompts.

The skills currently shipped:

| Skill | When it activates | What it knows |
|---|---|---|
| `neutron-reflectometry` | Always | Baseline SLD reference values, BIC guidance, Refl1D conventions, and the two interpretations of roughness (discrete layer vs. profile parametrization) with the interpretation-independent rule that the profile must stay within the range its bounding media can produce |
| `structural-hypothesis-ranking` | Always | How to generate and consume a ranked hypothesis list (§6) |
| `thin-layer-degeneracy` | Always | Why thin layers are multimodal (the Δρ·t ridge), why a BIC comparison can wrongly reject a real thin layer via a local minimum, discrete SLD mode enumeration, and using a cleaner sibling/time-series measurement as a prior (§6.8) |
| `metal-oxide-interfaces` | Samples with Cu, Ti, Fe, exposed metals | Native oxide thicknesses, SLDs, and when to add them |
| `polymer-films` | Samples with polymers, PS, PMMA, etc. | Polymer SLDs, typical roughness, glass-transition effects |
| `sei-layer-analysis` | Battery / electrolyte samples | Solid-electrolyte interphase layer conventions |
| `solvent-contrast-matching` | Samples in D₂O, H₂O, deuterated solvents | Solvent SLDs, contrast matching, isotope-confusion traps |

Skill activation is itself an LLM decision. At the start of a run, the
model is given the list of available skills and the user's sample
description, and asked which ones apply. Always-on skills are added
automatically regardless of the LLM's answer. When the selector LLM is
unavailable or returns an empty list, the always-on skills alone remain.

Skill selection is **not** frozen at intake. When the evaluation node
revises the hypotheses (§6.5), it re-runs the selector with the observed
fit artifacts passed as extra context, so a skill that only becomes
relevant once an artifact appears — an SEI signature in the residuals, an
unexpected contrast step — can activate mid-run. Skills are only ever
*added* this way, never removed.

**Why skills and not just one giant prompt?** Three reasons:

1. **Focus.** A prompt with only the domain knowledge that applies to the
   current sample fits in a smaller context window and the LLM pays more
   attention to it.
2. **Testability.** Skills can be unit-tested, versioned, and revised
   independently of the code. Fixing a wrong SLD is a Markdown edit, not
   a code change.
3. **Explainability.** Every action the agent takes can be traced to the
   skill that justified it. If the agent adds a CuO layer, the rationale
   will cite `metal-oxide-interfaces`.

Guidance inside skills is written as *descriptive prose for the LLM* —
not as conditional rules that code evaluates. This matters a lot: see §6.

---

## 6. The refinement problem and how AuRE solves it

This is the part of the design that is most specific to AuRE. It deserves
its own section because it is where the "agentic" aspects really matter
and where naive approaches fail.

### 6.1 The failure we set out to fix

Early versions of AuRE had a frustrating failure mode. The agent would be
given a sample like *"50 nm copper on 5 nm Ti on Si, measured in D₂O"* and
initially build a model with exactly those three layers. The first fit
would land at χ² of, say, 12 (poor). The evaluator would notice the fit
was not great. The modeling node would respond by *tweaking parameters* —
widening the Cu thickness bounds, nudging the Ti roughness, enabling
`sample_broadening` on one segment — and the next iteration would see χ²
drop slightly but still be poor.

This would repeat for five or six iterations. Each time, the model would
change *slightly* and χ² would improve *slightly*. The agent would
eventually run out of its iteration budget without ever realizing that
the real problem was structural: there is a ~20 Å layer of CuO (copper
oxide) on top of the copper — which always forms spontaneously when
copper is exposed to water — that was not in the model.

The root cause was a design mistake, not an LLM mistake. The skill files
contained phrases like *"if χ² > 10, consider adding a surface oxide"*
and *"structural changes are a last resort"*. Those phrases were read by
the LLM and dutifully obeyed. The agent would spend its entire budget on
parameter tweaks because we had told it, in so many words, that structural
changes come last.

### 6.2 The constraint we put on the fix

When we set out to improve this, we committed to one explicit constraint:

> **The fix must not be a heuristic in code.** We will not write rules
> like "after N iterations of no improvement, force a structural change"
> or "if χ² has not dropped by X%, add a layer". All decision-making
> about whether to do a structural change must live in the LLM's prompts
> and in the skills.

The reason is maintainability. Heuristic thresholds are wrong for some
sample and right for others; every time we add one, we have to debug it
later. The LLM, properly prompted with the trajectory and the available
hypotheses, can make the call much better than any fixed threshold can.

The fix is therefore mostly *prompt-engineering*, *state-shape*, and
*skill content*. The code that was added is deliberately confined to
*enforcing invariants* and *gating cost* — never to deciding the science:
a guarded merge that keeps the hypothesis list coherent (§6.5), some
bookkeeping around hypothesis status (§6.3), and two cost-gates — the
bound-only shortcut (§6.4) and the trigger that decides *when* it is worth
spending an LLM call to reconsider the hypotheses (§6.5). The *what* —
which hypotheses, which skills, which ranking — stays in the LLM.

### 6.3 Ranked structural hypotheses

The central new idea is the **ranked structural-hypothesis list**. At
intake time, after the sample description has been parsed, a second LLM
call — guided by the `structural-hypothesis-ranking` skill — enumerates
every structural change that might plausibly improve the fit. For the
copper example above, that list would contain an entry like:

```json
{
  "id": 1,
  "title": "Add native CuO on top of Cu",
  "rationale": "metal-oxide-interfaces: Cu exposed to D2O forms a 10–50 Å CuO with SLD ≈ 5.0",
  "change": "insert a 10–30 Å CuO layer (SLD 4.5–5.5) between Cu and D2O, σ 3–15 Å",
  "skill_source": "metal-oxide-interfaces",
  "origin": "skill",
  "status": "pending",
  "tried_in_iteration": null,
  "created_in_iteration": null,
  "notes": ""
}
```

along with a handful of other candidates — each ranked by

1. **prior probability** (native oxides on exposed metals in aqueous
   ambients are nearly certain),
2. **expected effect size** (a 20 Å CuO between Cu and D₂O produces a
   big low-$Q$ contrast step),
3. **cost in parameters / BIC** (fewer added parameters is better), and
4. **reversibility** (easy-to-revert changes rank higher than stack
   re-orderings).

This list is stored in the workflow state as `structural_hypotheses` and
is passed into *every* subsequent modeling and evaluation prompt. Each
entry carries a `status` field with one of four values:

- `pending` — not yet tried.
- `tried` — realized in the model in some iteration; outcome unknown.
- `confirmed` — was realized and the fit improved materially.
- `rejected` — was realized and the BIC guardrail reverted it.

Each entry also carries an `origin`: `user` (seeded from the `-h`
hypothesis, ranked at the top), `skill` (enumerated at intake), or
`evaluation` (proposed mid-run from fit evidence; see §6.5).

The statuses drive the agent's behaviour through prompts, not through
code. The evaluator's prompt says, in essence: *"here is the trajectory
of χ² and BIC across every iteration; here is the list of hypotheses and
their statuses; decide whether the next refinement should be a
parameter-tweak or a structural change, and if structural, cite the
hypothesis id."* The modeling prompt says: *"here is the hypothesis list;
if the evaluator picked one, realise it in the model and stamp its status
as `tried`; otherwise leave the list unchanged."* *Which* entry is realised
and *how* the list is ranked are LLM decisions; the **membership** of the
list, however, is protected by code (§6.5).

This makes the loop *explicit and auditable*. After a run you can open
the final checkpoint and see exactly which hypotheses were considered,
which were tried, which were confirmed, and which were rejected — with
the reasoning visible in the LLM's own words.

### 6.4 The bounds-only shortcut

A frequent intermediate state is: the fit ran, one parameter hit its
bound, the evaluation node auto-expanded the bound, and there are no
other issues. In earlier versions this still triggered a full LLM
refinement call (modeling → fitting → evaluation) even though the only
thing that needed to change was already done by the evaluation node.

The fix is a one-field optimisation. The evaluator sets
`bounds_only_refinement = True` when the *only* issue is an auto-expanded
bound. The router honours this by routing directly from `evaluation` back
to `fitting`, skipping the `modeling` node entirely and saving one LLM
call per such iteration. This is the only non-cosmetic heuristic in the
whole refinement system, and it is justified because it is purely an
optimisation — the bound has *already* been expanded deterministically
before this check runs; the skipped modeling call would have been a
no-op.

### 6.5 Guarding the list — and growing it when the data demands

Two things make the hypothesis list trustworthy as it evolves.

**A single guarded merge.** The `structural_hypotheses` field is replaced
wholesale on every node return (it has no append-reducer), so every write
goes through one function, `merge_structural_hypotheses`
([`nodes/hypotheses.py`](../src/aure/nodes/hypotheses.py)). It treats the
prior list as the source of truth for *identity*: an entry's
`id`/`title`/`change`/`skill_source`/`origin` are immutable once created,
and only `status`/`tried_in_iteration`/`notes` can change. The **modeling**
node calls it with `allow_new=False`, so when modeling realises a hypothesis
it can update statuses but can never silently add, drop, or rename an entry —
a misbehaving LLM that fabricates entries simply has them discarded and
logged. (Before this guard, the modeling write-back was unvalidated and the
list could drift run-to-run.)

**Adaptive revision at evaluation.** The intake list is a good plan, but
fitting sometimes reveals something nobody anticipated — residual fringes at
a characteristic thickness, an unexpected contrast step, a parameter that
will not leave its bound. When that happens (and *only* then — see the
cost-gate in §4.5), the evaluation node calls `merge_structural_hypotheses`
with `allow_new=True` to append LLM-proposed entries (tagged
`origin="evaluation"`) and then re-ranks the whole list. To propose well it
first re-selects skills from the *observed evidence* via
`select_skills(extra_context=…)` — the one channel by which fit artifacts,
not just the static description, can pull a domain skill into play. That
skill's knowledge then informs the proposed hypotheses. This is what lets the
agent bring prior knowledge to bear that the description alone did not
surface: see a ~40 Å fringe in a cycled battery cell, `sei-layer-analysis`
activates, and an SEI-layer hypothesis re-ranks to the top.

The split is deliberate: **intake** seeds the list (including the user's
hypotheses), **evaluation** grows and re-ranks it, **modeling** only updates
status.

### 6.6 Skill revisions

Alongside the code changes, two existing skills were rewritten to remove
prompt-level biases:

- In [`metal-oxide-interfaces`](../src/aure/skills/metal-oxide-interfaces/SKILL.md),
  the phrase *"consider adding a surface oxide if χ² > 10"* was replaced
  with guidance to *emit a high-ranked hypothesis at intake time*. This
  moves the decision from late in the loop (when iterations are already
  being wasted) to intake (where it costs nothing).
- In [`neutron-reflectometry`](../src/aure/skills/neutron-reflectometry/SKILL.md),
  the phrase *"structural changes are a last resort"* was replaced with a
  description of the hypothesis list as the canonical mechanism for
  structural changes. The BIC section now emphasises that there is an
  automatic guardrail and that the LLM does not need to be conservative
  about trying a hypothesis — if it was a bad idea, the guardrail will
  revert it and mark it `rejected`.
- In [`structural-hypothesis-ranking`](../src/aure/skills/structural-hypothesis-ranking/SKILL.md),
  the rule *"do not re-order entries; only append at the end"* was relaxed:
  the evaluation node may now re-rank the list and add entries, and the
  skill documents the `origin` provenance and the status-only restriction
  on the modeling node (§6.5).

Later, the roughness guidance in
[`neutron-reflectometry`](../src/aure/skills/neutron-reflectometry/SKILL.md)
was rewritten again — this time to drop the blunt *"roughness must be less than
half the adjacent thickness"* rule in favour of the two-interpretations framing
of §6.8, and a new always-on skill was added:

- [`thin-layer-degeneracy`](../src/aure/skills/thin-layer-degeneracy/SKILL.md)
  teaches the reliability lesson behind §6.8: thin layers are multimodal, a BIC
  "reject" of a physically-expected thin layer may be a missed global minimum
  rather than evidence of absence, the mode-enumeration escape, and using a
  cleaner sibling/time-series measurement as a prior. It is always-on because
  the lesson is general to almost every fit.

### 6.7 Division of labour between LLM and code

The table below summarises who decides what.

| Decision | Who makes it | How it is made |
|---|---|---|
| Parse the sample into a layer stack | LLM | `format_sample_parsing_prompt` |
| Generate ranked hypothesis list | LLM | `structural-hypothesis-ranking` skill |
| Seed the user's hypothesis as top-ranked entries | LLM | structural-hypothesis prompt (`origin="user"`) |
| Choose which skills apply | LLM | Skill selector prompt |
| Re-select skills from fit evidence (mid-run) | LLM | `select_skills(extra_context=…)` |
| Propose refinement direction (tweak vs structural) | LLM | Evaluation prompt with trajectory + hypotheses |
| Propose new hypotheses + re-rank (mid-run) | LLM | Evaluation revision prompt (`origin="evaluation"`) |
| Realize a structural change | LLM | Modeling refinement prompt |
| Update hypothesis status after realising a change | LLM | Modeling refinement prompt |
| Auto-expand stuck bounds | Code | `_expand_model_bounds` in `evaluation.py` |
| Detect non-physical SLD-profile excursions | Code | `detect_profile_artifacts` in `feature_tools.py` |
| **Stop when χ² meets the threshold** (LLM's `acceptable` is advisory *at or below* it only) | Code | `_clamp_acceptance_to_chi2` — runs *after* the profile veto, which outranks it; raises verdicts only, never lowers them |
| **Choose the model the run reports** | Code | `finalize._select` — profile-vetoed fits set aside, then lowest χ² with a parsimony tie-break inside the χ² band; still floor-blind, see [issues.md](../issues.md) #11 |
| Report the untried hypotheses at the end | Code | `finalize._format_remaining_improvements` (statuses reported, never re-derived) |
| Escape thin-layer local minima before fitting | Code | mode enumeration in `fitting.py` (gated by `MODE_ENUMERATION`) |
| Revert on χ² regression | Code | χ² regression guardrail |
| Revert on BIC regression + mark hypothesis `rejected` | Code | BIC regression guardrail |
| Enforce list membership (modeling = status-only) | Code | `merge_structural_hypotheses` guard |
| Decide *when* revision is worth an LLM call | Code | `_should_revise_hypotheses` (cost-gate) |
| Route to fitting on bounds-only iterations | Code | `route_after_evaluation` |

The pattern is: the LLM *proposes* and *explains*; code *enforces
invariants*, *reverts regressions*, and *bookkeeps deterministic
outcomes*.

### 6.8 Thin layers: degeneracy, profile artifacts, and tied roughness

Three related mechanisms address a family of failures that all trace back to
*thin* layers — layers thinner than the real-space resolution limit
$2\pi/Q_\text{max}$ (≈ 30 Å for a typical $Q_\text{max} = 0.2\,\text{Å}^{-1}$).

**Why thin layers are hard.** Reflectivity constrains a thin layer mainly
through the *product* of its contrast and thickness ($\Delta\rho \cdot t$), not
$\rho$ and $t$ separately. Below the resolution limit, many $(\rho, t)$ pairs on
a curve of constant $\Delta\rho \cdot t$ fit almost equally well, and they live
in *distinct local minima* separated by barriers the optimizer will not cross.
The consequence is dangerous for automated model selection: if the more complex
candidate model landed in the wrong basin, its χ² is too high, its BIC looks
too large, and a real layer gets rejected as "not justified." **A BIC verdict is
only as trustworthy as the optimization behind it.**

**Mode enumeration (fitting, §4.4).** The fix is not more optimizer effort but
*discrete* exploration: seed the thin layer's SLD at a few levels spanning its
range and polish each locally. This reliably finds the right basin where a
single global run does not. Signatures of the local-minimum trap (worth
recognising in the `thin-layer-degeneracy` skill) include an *adjacent* layer's
parameter pinned at a bound, a roughness at a bound, or a tiny χ² gain for the
added parameters. Notably, seeding the whole stack from a sibling measurement's
converged structure does *not* work — the optimizer still slides into the wrong
mode of the thin layer; the SLD mode must be enumerated explicitly.

**Two interpretations of roughness.** A slab model with a roughness larger than
half a layer's thickness is not automatically wrong. It means one of two things,
and you must decide which:

- *Discrete-layer interpretation* — the slab is a real layer with a physical
  thickness and SLD. Here a large roughness that makes one slab's error-function
  tail bleed across a thin neighbour is a modeling error.
- *Profile-parametrization interpretation* — the slab stack is being used as a
  flexible basis to represent a smoothly-varying / graded SLD profile. The
  affected slabs are then **not** layers; one reports only the profile shape,
  and a large roughness is expected and fine.

The rule that holds under *both*: the resulting SLD profile must never leave the
range physically reachable by its bounding media (no dip below / overshoot above
what a blend of the two adjacent materials can produce). That excursion — not
the roughness/thickness ratio — is the genuine artifact, and it is invisible in
χ². The evaluation-node detector (§4.5) flags exactly this.

**Tied roughness (the remedy, only under the layer interpretation).** When a
discrete-layer reading is intended, the fix is not to cap or shrink the
roughness (that just distorts the thickness) but to fit the *ratio*: a layer may
carry a `roughness_tie` so its interface is built as $\sigma = f \cdot t$ with
the fraction $f \le 0.5$ the fitted parameter. The interface then cannot outgrow
its layer as the thickness moves. If instead the diffuse transition is intended,
the roughness stays free and the region is re-labelled as a profile
parametrization — the detector's suggestion offers both branches and leaves the
choice to the modeling step.

---

## 7. Reading the output of a run

When you run `aure analyze` with `-o ./output`, AuRE writes one
checkpoint file per completed node, as well as a final `state.json` with
the full workflow state. The files you will typically want to inspect are:

- **`intake_result.json`** — parsed sample and **the full ranked
  hypothesis list**. Inspecting this file after an unsuccessful run tells
  you whether the structural problem was even in the candidate list.
- **`analysis_result.json`** — the data-derived physics features.
- **`modeling_iter*.json`** — the model that was sent to the fitter at
  each iteration.
- **`fitting_iter*.json`** — the best-fit parameters and χ² per iteration.
- **`evaluation_iter*.json`** — the LLM's judgement, including
  `next_action` and `proposed_hypothesis_id`; plus the updated
  `structural_hypotheses` with statuses. On iterations where the revision
  step fired (§6.5), this also shows any newly-added `origin="evaluation"`
  hypotheses, the re-ranked order, and any skills that were activated from
  the observed artifacts.

For multi-state co-refinement runs (`states:` block in the user config),
each fit iteration also produces a per-state `profile.dat` under
`output/refl1d_output/fit_iter{i}_{method}/state_<name>/`, and the
aggregated `FitResult` carries one `PerFileFitResult` entry per dataset
tagged with its `state` name. Those per-state profile files are never read
back, though — only the single top-level one, which belongs to `states[0]`
([issues.md](../issues.md) #4). See the `multi-state-corefinement` skill
for the experimental patterns this addresses and for guidance on choosing
`shared_parameters` vs `unshared_parameters`.

The web UI (`aure serve ./output`) reads these checkpoints and displays
the history interactively.

---

## 8. Extending AuRE with new skills

Adding a new piece of domain knowledge is usually a five-minute task:

1. Create a new directory under `src/aure/skills/<your-skill-name>/`.
2. Write a `SKILL.md` with YAML frontmatter (`name`, `description`, `metadata`)
   followed by a prose body.
3. The `description` is what the selector LLM sees when deciding whether to
   activate the skill; write it as if you were answering the question
   "when should this skill be active?".
4. Add `<your-skill-name>` to the always-on tuple in
   [`src/aure/skills/selector.py`](../src/aure/skills/selector.py) if it
   should run on every sample.
5. Add a test in [`tests/test_skills.py`](../tests/test_skills.py) if the
   skill should appear for specific sample descriptions.

Skills should speak to the LLM in the LLM's own language — prose, not
pseudocode. Imagine you are briefing a new colleague who has a physics
background but has not worked on this exact sample family before. Avoid
writing thresholds or conditional rules unless the threshold is a
physical constant (e.g. *"CuO SLD is 4.5–5.5"* is a physical fact,
*"if χ² > 10, do X"* is a workflow heuristic and belongs in the prompts
or in code, not in a skill).

---

## 9. Further reading

- **Reflectometry physics**:
  [Jens Als-Nielsen & Des McMorrow, *Elements of Modern X-ray Physics*](https://www.wiley.com/en-us/Elements+of+Modern+X+ray+Physics%2C+2nd+Edition-p-9781119970156),
  chapters on reflectivity.
- **Refl1D**: the [Refl1D documentation](https://refl1d.readthedocs.io)
  for the optimizer and model-description API AuRE builds on.
- **BIC**: Schwarz, G. (1978). "Estimating the Dimension of a Model".
  *The Annals of Statistics*. Explains why we prefer BIC over χ² for
  deciding whether to add a layer.
- **Agent Skills pattern**: the
  [`src/aure/skills/`](../src/aure/skills/) directory is the canonical
  reference for how skills are structured.
