# Plan: reading REF_L's `new_reduction` format, and making the next format cheaper

**Status:** proposed. Raised 2026-09-21 after a beamtime (IPTS-37740, runs
234277–234335) in which AuRE could not be used at all, because the reduction
had changed format between experiments.

## What happened

REF_L's `new_reduction` pipeline writes `REFL_<run>_<seg>_<subrun>_autoreduction.dat`
alongside the established `_partial.txt`. Both are live. They are not two
spellings of one format:

| | `_partial.txt` | `_autoreduction.dat` |
|---|---|---|
| header | one `# Meta: {json}` line | several `# Key = value` / `# Key: value` lines |
| notation | JSON | JSON **and** Python literals, mixed line by line |
| scope | **this segment** | **the whole run**, byte-identical in every segment file |
| incident angle | `TwoTheta` in this file's own table | a slot in a run-wide `Angles.THS` array |
| dQ column | **FWHM** | **sigma** |

The dQ change alone is a factor of 2.355. Applied the wrong way round the
model is under-smeared and chases fringe structure the measurement cannot
resolve, and the fit absorbs the error into roughness rather than reporting it.

AuRE recognised none of this. `nrw aure new` wrote a setup for sample1 and it
was never run to a result; the whole beamtime was fitted through hand-written
`nr-workbench` model specs instead. The two lines that made those specs work
are worth quoting, because they are the entire remedy:

```yaml
probe:
  dq_is_fwhm: false            # "READ FROM THE FILES, not chosen"
states:
  - thetas: [0.45, 1.251, 3.5] # "read from each file's header. Do not round them."
```

nr-workbench had already read both facts out of the header. AuRE had nowhere
to put them: a setup file's `data_files` entry keeps `file` and `label` and
silently drops everything else. **That is the finding this plan is organised
around.** The instrument work below is the durable fix, but the reason the
beamtime was lost is a dropped key, not a missing parser.

## What AuRE does with such a file today

Measured, not assumed — a real `_autoreduction.dat` header put through the
current code:

```
resolved instrument: generic
role:                unknown
group_key:           None
header_metadata:     {'dq_is_fwhm': True, 'theta': 0.0, 'num_segments': 0, ...}
run_title:           '{"title": ["Sample1_air-234277-1.", "Sample1_air-234277-2.", ...]}'
detect_kind:         combined
```

Five defects, in severity order:

1. **`dq_is_fwhm: True` on a sigma file.** `instruments/ref_l.py` matches
   `_partial\.txt$`, and its header sniff looks for `TwoTheta`, which this
   dialect does not have. Nothing claims the file, so the generic default
   applies — and the default is the one convention that is now wrong.
2. **The LLM header parse cannot get theta right, structurally.**
   `intake.parse_file_header` passes the LLM the header text and *not the
   filename*. In this dialect the angle lives in a run-wide array indexed by
   this file's segment number, which appears only in the filename. The LLM
   returns 0.0 or picks a slot positionally, and positional is wrong (below).
3. **The run-title regex captures a JSON blob.** `_RUN_TITLE_RE` matches
   `# Run Title:` and takes the rest of the line — 200 characters of JSON
   array, recorded identically in every file of the run, so
   `_reconcile_run_titles` sees no disagreement and flags nothing. Inert while
   `USE_RUN_TITLE` is off; with it on, that blob seeds `origin="header"`
   hypotheses.
4. **Role `unknown` means a three-angle measurement is classified
   `combined`.** Nuisance parameters (`theta_offset`, `sample_broadening`) are
   refused at config load, `group_key` is `None` so intake's mixed-set_id
   guard can never fire, and probes are built Q-based.
5. **The setup file drops what the caller already knows** —
   `config._parse_states` keeps `file` and `label` only, without a word, even
   though `model_builder` already honours a per-file `theta` and
   `dq_is_fwhm`.

And one duplication we own: `web/static/setup.js` carries a JavaScript copy of
the partial-file regex, which the registry cannot reach.

## The registry already exists

nr-workbench's own
[`docs/plan-reduced-format-registry.md`](../../nr-workbench/docs/plan-reduced-format-registry.md)
proposes building a format registry, and names `aure/instruments/ref_l.py` as
one of the four places that hardcode REF_L. On our side that registry is
already here — `src/aure/instruments/` has the protocol, priority-ordered
resolution, an `aure.instruments` entry-point group for third-party plugins,
an `AURE_INSTRUMENT` override, a walkthrough in `docs/instruments.md`, and a
golden parity table in `tests/test_instruments.py`.

So this is not a build-the-seam job. It is: add a format *through* the seam,
and close the four gaps this particular format proves the seam is missing.

## The header is defective, and a reader must know how

The `Angles` and `Run Title` arrays are longer than the number of segments.
nr-workbench discovered this and documented it as "a segment measured in two
pieces gets two entries". **That explanation is wrong**, and the correct one
changes what a reader should do. Every run in the beamtime:

| run | files on disk | `len(title)` = `len(THS)` | all `Config` arrays | segment pattern |
|---|---|---|---|---|
| 234277 | 3 | 4 | 3 | `[1, 2, 2, 3]` |
| 234280 | 3 | **7** | 3 | `[1, 2, 3, 1, 2, 3, 3]` |
| 234283 | 3 | 6 | 3 | `[1, 2, 3, 1, 2, 3]` |
| 234327 | 3 | 6 | 3 | `[1, 2, 3, 1, 2, 3]` |
| 234333 | 3 | 6 | 3 | `[1, 2, 3, 1, 2, 3]` |

Four of five are exact whole-block repeats of `[1,2,3]`; 234280 is two full
blocks plus a lone segment 3. A segment measured in two pieces produces
`[1,2,2,3]` once — it cannot produce `[1,2,3,1,2,3]`. **The reduction appends
to these arrays on reprocess instead of replacing them.** Confirming it: only
`Run Title.title` and `Angles.{THS,THI,ThCen}` grow, while every array under
`Config` (`DB`, `Scaling factors`, `RBnum`, `ThetaShift`, `RB_Ymin/Ymax`,
`method_per_run`, `tof_max`) stays at exactly 3 — the per-acquisition lists
accumulate, the ones built from the reduction template are rewritten.

Three consequences for any reader, including ours:

1. **Title-matching over positional indexing is right, and for a stronger
   reason than nrw gave.** Under whole-block repeats, positional indexing is
   *accidentally correct* and only fails on 234277 (`[1,2,2,3]`, where
   segment 3 lands on 1.251° instead of 3.5° — a factor of ~2.8 in Q that fits
   cleanly to a wrong thickness). Had that one run not been in the beamtime,
   the bug would have shipped invisibly.
2. **Take the last match, not the first.** The first slot naming a segment is
   the *oldest* reduction. Every duplicate agrees in this data, so nothing is
   wrong today — but it is stale by construction, and a reprocess that
   corrected `ThetaShift` or switched `useCalcTheta` would be silently
   discarded. Take the last, check that every match for that segment agrees,
   and complain loudly when they do not.
3. **The header is internally inconsistent, so validate before indexing.**
   The long arrays must be indexed by title and the short `Config` arrays
   positionally by segment — two schemes over arrays that no longer align.
   That holds only while the short ones stay at segment length. Check
   `len(Config array)` against the segment count and report rather than guess
   when it does not match.

**This belongs upstream with the REF_L reduction team.** The old `_partial.txt`
was better here: its `# Meta:` block describes only its own segment, so there
is nothing to index and nothing to accumulate. Replacing a per-file header
with a whole-run one that grows on every reprocess is a regression whatever
consumes it. Until that is fixed, every reader needs the rules above.

## The plan

### Phase 0 — accept what the caller already knows — **done**

The change that would have saved the beamtime, and it needs no new format
code.

- `config._parse_states` accepts `theta` and `dq_is_fwhm` on a `data_files`
  entry, validates them (`theta` a non-negative float in degrees,
  `dq_is_fwhm` a bool), and **rejects unknown keys** there with an error
  naming the key and the accepted set. Silently dropping `theta:` is how this
  failure stayed invisible; an unknown key in a file the user hand-edits is
  a mistake, not forward compatibility.
- `setup._dataset_for_dump` emits both when present, so the web UI and
  `aure batch` round-trip them.
- `intake._enrich_dataset` already uses `setdefault`, so a declared value wins
  over the header parse. Verify the same holds for the state-level
  `dq_is_fwhm` on the single-file path, which reads the state rather than the
  dataset.
- A declared value is logged at INFO with its origin, so a run says which
  numbers came from the setup and which from the header.

The companion one-liner is in nr-workbench: `nrw aure new` writes the two
values it has already read. Together, AuRE runs on this beamtime with no
instrument work at all.

This is also the general answer for any facility whose format we do not yet
read, and the reason it comes first: it is the escape hatch that makes a
missing instrument an inconvenience rather than a wall.

### Phase 1 — the format, through the seam — **done**

`REFLv2Instrument` (`REF_L_v2`) in `instruments/ref_l.py`, registered ahead of
`REFLv1Instrument` (`REF_L_v1`), sharing its `group_key` scheme so a beamtime
mid-migration can co-refine both dialects in one state.

The names are versions because v2 is a stop-gap ahead of a larger reduction
rewrite and will probably be superseded rather than settle. A name is written
into every checkpoint that touches a file, so it must still mean the same
thing years later: `REF_L_autoreduction` is wrong as soon as the pipeline is
renamed, and `REF_L_prototype` as soon as that judgement changes. The version
goes in the name; the judgement goes in `lifecycle` / `lifecycle_note`, and
`aliases` keeps `AURE_INSTRUMENT=REF_L` working.

- Field parsing per nr-workbench's `_autoreduction_fields`: JSON first, then
  `ast.literal_eval`, per line. Parsing with only one of them half-works —
  `Angles` is valid under both, so the angle (the field most likely to be
  checked) comes out right while everything under `Config` degrades to `None`.
- Segment number from the **filename**; the header does not carry its own.
- Angle by **last** title match, with the agreement check and the `Config`
  array-length validation from the section above.
- `authoritative_fields = ("dq_is_fwhm", "theta")`. Both are stated by this
  format. Neither is the LLM's to guess, and for `theta` the LLM cannot
  succeed even in principle.
- `q_range` and `dq_over_q` stay unset: `Config` carries what the reduction
  was *asked* for, not what it produced.
- Tests port nr-workbench's fixture verbatim. The four-entries-for-three-segments
  case is the one that matters; add a whole-block-repeat case and a
  disagreeing-duplicates case, neither of which nrw has. New golden rows only —
  the existing table is parity with the pre-registry behaviour and stays put.

Shipped in `tests/test_instrument_autoreduction.py`. The array-consistency and
disagreeing-pass checks log at WARNING for now; phase 2's `header_issues`
promotes them to run-visible issues. `run_title` is implemented on the
instrument and unused until phase 2 puts it on the protocol.

One behaviour change to know about: a state of these files is now classified
`partials` rather than `combined`, so its files must share one run number and
`theta_offset` / `sample_broadening` become available on it.

### Phase 2 — three protocol gaps this format exposes — **done**

Each optional, each defaulting to today's behaviour.

- `run_title(file_path)` on the protocol, with intake's current regex as the
  generic implementation and the autoreduction instrument returning its own
  segment's title. Fixes defect 3 and stops a REF_L assumption living in a
  node.
- `header_issues(file_path) -> list[str]`, surfaced by intake as a run
  warning. This is where "the arrays disagree in length" and "this file
  declares a dQ convention I cannot parse" go. nr-workbench raises on an
  unknown dQ label; we must not quietly fall back to FWHM.
- The multi-file-unknown case says so plainly: a state of several files that
  no instrument claims is currently called `combined` on a debug-level note.

Both members are looked up by name rather than declared on the `Instrument`
protocol, so an instrument that implements neither is unchanged and one whose
implementation raises is logged and skipped — a third-party instrument must
not be able to fail an analysis. `ref_l.py`'s checks now run through one
`_autoreduction_scan`, so `header_metadata` (which logs them) and
`header_issues` (which returns them) cannot disagree about what a file says.

### Phase 3 — discoverability — **done**

`aure formats`: list registered instruments, built-in and plugin, and given
paths show what each resolves to, whether by name or by header, and what that
implies — role, group key, dQ convention, theta. This is what a user at
another facility runs when their file is not recognised.

nr-workbench's hardest-won lesson was that `nrw.toml`'s `[conventions]` block
*looks* like the place to configure this and is read by nothing, so editing it
eliminates the filename as a suspect while changing nothing. The equivalent
mistake here would be a registry with no way to ask what it contains. Same
data behind `GET /api/instruments/classify`, so `setup.js` can drop its regex
copy.

Shipped as `aure formats` (`cli._classify_for_report` behind both surfaces) and
`GET /api/instruments/classify`. `setup.js` now caches a role per path from
that endpoint and re-renders the overrides panel when it arrives; the old
regex survives only as the fallback for the first render before the request
returns, renamed to say so. The report distinguishes claimed-by-filename from
claimed-by-header, because a file claimed only by its header still works but
its classification changes on a rename — and `resolve_by_name` is what runs
while a setup is parsed.

### Phase 4 — documentation and a skill — **done**

- `docs/instruments.md`: the new built-in, the new optional protocol members,
  `aure formats`, and a user-facing "my file is not recognised" section giving
  the three escape hatches in order of effort — declare `theta` / `dq_is_fwhm`
  in the setup (Phase 0), `AURE_INSTRUMENT=`, write a plugin.
- A developer skill at `.claude/skills/add-data-format/SKILL.md`. It must open
  by distinguishing itself from `src/aure/skills/`, which holds *science*
  skills injected into LLM prompts — wrong directory, easy mistake. Contents:
  get a real file and commit its header as a fixture before writing code; the
  questions the protocol asks; what to declare authoritative; add golden rows,
  never edit existing ones. Four traps, all met here:
  - sigma vs FWHM is 2.355 and a fit hides it in roughness;
  - **when a header's arrays disagree in length, establish which are
    per-acquisition and which are per-request before indexing either** — the
    generalisable form of this beamtime's bug, and not REF_L-specific;
  - a writer may mix JSON and Python notation line by line;
  - never round a measured angle to its nominal setting.

  And the negative test: **if you find yourself editing a file under
  `nodes/`, the seam is wrong and that is the bug to fix.**

Shipped as `.claude/skills/add-data-format/SKILL.md`, with
`tests/test_developer_skills.py` pinning the two-directory split — a file in
the wrong one fails silently in both directions, so it is not something to
leave to a convention. `docs/instruments.md` now opens its escape-hatch
section with the three options as a table, ordered by effort, and says why
none of them is a lesser version of the others.

### Phase 5 — `de` rather than amoeba for the exploration step

Not a format change, and tracked here because it came out of the same
beamtime. It ships as its own commit.

**The evidence.** On sample2's D2O state, a 23-parameter model started at
χ² 1074; amoeba stopped at 88.5 with the ionomer and hydrated-layer SLDs the
wrong way round, and `de` on the identical problem — same spec, same data,
same bounds — reached 17.7, with three seeds agreeing to within 0.06. No model
change was involved, so no model change could have been the fix. amoeba is a
simplex: it walks downhill from where it starts and stops at the first minimum
it reaches, so on many parameters and a distant starting point it reports a
local minimum with no sign that it is one. nr-workbench put `de` on its fitter
menu for this (`7ee93ea`); the ionomer project's
`samples/sample2/reports/why-sample2-will-not-fit-the-sample1-stack.md`
records the numbers.

**What that does *not* license**, and the reason the phase says this out loud:
`de` beating amoeba says the *search* was the problem. `de` reaching the same
bad minimum from every seed says the *model* is, and no further optimiser will
help. On the same sample2, the air run bottomed out near χ² 62 under a
240k-evaluation `de` search with every bound reopened — the stack was wrong,
not the fitter. Swapping the default must not make "try another optimiser"
feel like progress; AuRE's refinement loop is exactly where that could become
an automated version of the loop nr-workbench's fitters module exists to
prevent.

**Where amoeba actually appears in AuRE**, which is less than the phase title
suggests:

| Site | What it is |
|---|---|
| `nodes/fitting.py` `method="amoeba"` | hard-coded, the mode-enumeration cheap polish |
| `nodes/final_fit.py`, `docs/finalization.md`, `aure_config.example.yaml` | prose recommending amoeba as the exploration method |

`FIT_METHOD` itself defaults to `dream` (`.env.example`, `fitting.py`), and
`setup.py` documents the menu as `lm | de | dream` — so **amoeba is not on
AuRE's menu at all**, yet it is hard-coded in one hot loop and recommended in
three documents. That inconsistency is the first thing to fix, and it may be
most of the work.

The one judgement call is the mode-enumeration polish. It runs one fit *per
seed per thin layer*, so it is the place where amoeba's speed is actually
being bought — `de` there could turn a cheap pre-pass into the dominant cost
of a run. **Benchmark both on the ionomer runs before switching that one.**
The other sites are documentation and can change immediately.

**The shape this should take: the loop decides, rather than the default
forcing.** Not a global swap to `de`, but an escalation the refinement loop
chooses — `evaluation` already detects a stalled χ² (≥2 iterations without
improvement), which is exactly the signal that distinguishes a search problem
from a model problem, and it already routes on that signal to propose
structural hypotheses. Escalating the optimiser is a cheaper first move than
changing the structure, and it belongs in the same decision.

What makes this better than a new default is that **the escalation produces
the diagnostic**. Run `de` once on a stalled fit and the answer is
informative either way: a decisive improvement says the search was the limit
and the structure was fine; the same minimum reached again says the model is
the limit, which is the finding that tells the loop to stop trying optimisers
and start changing the stack. A forced default gets the first outcome and
throws away the second, because there is nothing to compare against. The
sample2 air run is the case in point — χ² 62 under a 240k-evaluation `de`
search with every bound reopened, where the answer was the stack.

So the escalation is recorded like any other attempt: which optimiser ran,
what it reached, and whether it agreed with its predecessor. That also keeps
the guard rail — one escalation per stall, not an optimiser menu the loop can
cycle through, which is the failure nr-workbench's `fitters.py` exists to
prevent and which an automated loop could reproduce far faster than a person.

## Decisions taken

**The ORNL dialect ships upstream in AuRE, not as an nr-workbench plugin.**
It is REF_L and we ship REF_L; nr-workbench deletes code on its next version
bump. The entry-point plugin path stays the documented answer for other
facilities, and `tests/test_instruments.py` already exercises it.

**Phase 0 may be enough on its own** for a beamtime in progress. Phases 1–4
are the durable fix and need not block anyone.

## Out of scope, recorded here

**The test suite can make live LLM calls, depending on import order.**
`aure.cli` calls `load_dotenv()` at import, so the first test module that
imports it leaves a developer's real `LLM_PROVIDER` / `LLM_API_KEY` in
`os.environ` for every test that runs after it in the same process. A test
touching `intake.parse_file_header` then reaches a live model, and its result
depends on which other files pytest collected. Found while writing
`tests/test_intake_declared_header_values.py`, which pins
`llm_available -> False` for exactly this reason; the general fix is an
autouse isolation fixture in `conftest.py`. nr-workbench hit the same class of
bug and fixed it in `d6f91ab`.

The `Neutron_ionomer` project excludes AuRE's `output/` from version control
because `final_state.json` carries absolute data paths and the full
prompt/response transcript, which makes the directory unshareable. That is an
adoption blocker independent of file formats and wants its own issue.
