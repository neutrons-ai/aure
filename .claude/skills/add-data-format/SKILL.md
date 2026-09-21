---
name: add-data-format
description: >
  Teach AuRE to read a new reflectivity data format, or a new dialect of one
  it already reads.
  USE FOR: a facility's files that `aure formats` calls unrecognised; a
  reduction that changed its header, its filename pattern, or what a column
  means; deciding what an instrument should and should not claim to know.
  DO NOT USE FOR: the science the numbers feed into, or column parsing
  (`tools/data_tools.py` and refl1d already do that).
metadata:
  author: aure
  version: "1.0"
  audience: developers
---

# Adding a data format to AuRE

**This is a developer skill, not a science skill.** It lives in
`.claude/skills/` and is for someone editing AuRE. The skills in
`src/aure/skills/` are a different thing entirely — domain knowledge injected
into LLM prompts at runtime. Putting this one there would feed a refactoring
checklist into a modelling prompt. Putting one of those here would stop it
reaching the model at all.

Reference: [`docs/instruments.md`](../../../docs/instruments.md) is the
protocol and the walkthrough. This is the checklist for doing it well, written
from what the REF_L `new_reduction` dialect cost — see
[`docs/plan-new-reduction-format.md`](../../../docs/plan-new-reduction-format.md).

## Before you write anything

**1. Get a real file. Not a description of one.** Every trap below was
invisible in the format's documentation and obvious in the bytes. If you have
only one file, you cannot see which header fields vary per segment and which
are per run — get at least two from the same measurement, and if the format
has segments, get all of them.

**2. Commit its header as a test fixture first.** Trim the data rows, keep the
header exactly: every oddity, in its original notation. Then write the
assertions you expect before the class exists. The fixture is the thing that
survives; your parser is an implementation of it.

**3. Run `aure formats` on the file** and record what it currently does. That
is the behaviour you are changing, and if you cannot state it, you are not
ready to change it.

**4. Ask what the format actually decides.** An instrument answers four
questions and no others: role, group key, header metadata, and whether
per-angle nuisance parameters apply. If you find yourself wanting to change
how the fit works, you are past the seam.

## The four traps

Each of these was met for real. They are ordered by how expensive they were.

### The resolution convention, which is a factor of 2.355

Whether a dQ column is a FWHM or one standard deviation is the single most
damaging thing to get wrong, because **nothing downstream will tell you.** The
fit converges either way; the error is absorbed into roughness and reported as
a structure. Two live reductions of the same instrument disagreed about this.

So: read it per file, from the header, every time. Never infer it from the
filename, the facility, or what the last format did. If the header states it,
say so with `authoritative_fields = ("dq_is_fwhm",)` — otherwise an LLM header
parse can overrule the file. If the header does *not* state it, do not guess
quietly: return the default and report it through `header_issues`.

### Arrays that disagree in length

**When a header's arrays have different lengths, find out which are
per-acquisition and which are per-request before indexing any of them.** This
is the generalisable form of the worst bug in this area, and it is not
specific to any facility.

In REF_L v2, `Angles.THS` and `Run Title.title` grow every time the data is
reprocessed — the reduction appends rather than replaces — while the arrays
under `Config` come from the reduction template and stay at the segment count.
Two array families, two different indexing rules, nothing in the file saying
so. Positional indexing gave segment 3 an angle of 1.251° instead of 3.5°, a
factor of ~2.8 in Q, which fits cleanly to a wrong thickness.

Three rules that follow:

- **Index by a key, not a position**, where the format gives you one (a title
  ending in `-<segment>.`, an explicit id).
- **Take the most recent match, not the first.** With append-on-reprocess the
  first entry is the oldest and is stale by construction.
- **Cross-check the lengths and report a mismatch.** Cheap, silent on healthy
  files, and the early warning for the next mutation of the format.

Note how this one hid: four of five runs carried whole repeated `[1,2,3]`
blocks, where positional indexing is *accidentally right*. One ragged run
exposed it. **If your fixture only has the well-behaved case, you have tested
nothing.**

### A writer that mixes notations

One header had JSON on one line (`null`, `false`) and Python literals on the
next (`None`, `'single quotes'`). Parsing with only one of them half-works,
which is worse than failing: the numeric fields parse under both, so the angle
— the thing you would spot-check — looks right while everything else silently
degrades.

Try both (`json.loads`, then `ast.literal_eval`, which executes nothing), per
line, and keep the raw string when neither works.

### A nominal value where a measured one belongs

`1.251°` is the measurement. `1.2°` is the setting it was requested at. Never
round one to the other, and never report a *requested* range as a measured one
— `Config.qmin/qmax` is what the reduction was asked for, not what it
produced. If the header only carries the request, leave the field unset rather
than filling it with a plausible number.

## Writing it

- **One class per format version**, even for dialects of one instrument. They
  share helpers, not a `matches`.
- **`name` is written into every checkpoint**, so it must still mean the same
  thing years from now. Name the *version* of the format (`REF_L_v2`). Keep
  the pipeline's name out of it — pipelines get renamed. Keep the judgement
  out of it — `lifecycle` / `lifecycle_note` carry that and can be revised
  without stranding old checkpoints. Add `aliases` if you rename one.
- **Match on the filename where you can.** Resolution runs while a setup file
  is parsed, before the data need exist. A header sniff is the fallback, not
  the rule.
- **Every method tolerates a path that does not exist.** Return defaults; do
  not raise.
- **Declare `authoritative_fields` only for what the format genuinely
  specifies.** For REF_L v2 that includes `theta`, because the LLM header
  parse is given the header text and not the filename, and that format's angle
  can only be found via the segment number in the filename — it cannot succeed
  even in principle, so it must not be allowed to supply a plausible wrong
  answer.
- **Use `header_issues` for what the header says that is wrong**, as distinct
  from a file being unreadable. These warn rather than fail: a header defect
  is usually survivable, and refusing to load over one is worse than
  proceeding with a stated caveat. What must not happen is proceeding
  silently.

## Testing it

- Add **new** rows to the golden table in `tests/test_instruments.py`. Do not
  edit existing ones; they pin what REF_L files have always meant.
- Test the ragged case, not just the clean one (see above).
- Test that a **healthy file produces no warnings**. Warnings that fire on
  normal data get ignored, which costs you the one time they matter.
- Test the missing-file and truncated-header paths.
- Check `aure formats <file>` by hand. It is what a user will run, and it is
  the fastest way to see you have mis-set a role.

## The negative test

**If you are editing a file under `src/aure/nodes/`, stop.** The seam exists
so that adding a format touches `instruments/` and nothing else. A node that
needs to know about your format is a sign that either the protocol is missing
a member — add it as an optional one, looked up by name, so older instruments
keep working — or you are solving the wrong problem.

The same applies to `config.py`, `setup.py` and the web layer: they ask the
registry. If your format needs them to ask something new, add it to the
registry's surface, not to theirs.

## When you are done

- `aure formats` lists your instrument with a sensible lifecycle and note.
- `aure formats <your files>` reports the right role, angle and convention.
- A setup naming those files loads, and `_detect_kind` classifies the state as
  you intend.
- `docs/instruments.md` describes the new built-in under **The built-ins**.
- Nothing under `nodes/` changed.
