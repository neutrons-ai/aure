# Adding an instrument or file format

AuRE reads reflectivity data from ORNL's REF_L and from ORSO `.ort` files out
of the box. Anything else needs an **instrument**: a small class that answers
four questions about a data file. Nothing else in AuRE has to change.

Every format-specific convention lives in
[`src/aure/instruments/`](../src/aure/instruments/) behind the protocol below,
so adding a format is a new class and a registration — not an edit to the
nodes that consume the data.

## What an instrument is for

AuRE does not ask an instrument to *read* the data — column parsing is
[`tools/data_tools.py`](../src/aure/tools/data_tools.py)'s job, and the fit
loads probes through refl1d. An instrument supplies the things AuRE cannot
infer from numbers alone:

| Question | Method | Why AuRE needs it |
|---|---|---|
| Is this one complete curve, or one angle of a set? | `file_role` | decides whether per-file resolution parameters are meaningful |
| Which measurement set does it belong to? | `group_key` | catches a state that mixes files from different measurements |
| What does the header say? | `header_metadata` | incident angle, dQ convention, segment count |
| May this role carry `theta_offset` / `sample_broadening`? | `role_supports_nuisance` | those describe one angle's optics |

## The protocol

```python
from aure.instruments import COMBINED, PARTIAL, UNKNOWN

class MyInstrument:
    name = "MYREF"                      # recorded in checkpoints — see
                                        # "Naming, versions and lifecycle"

    def matches(self, file_path, header=""):
        """Claim the file, by name where possible.

        `header` is the first 40 lines when available and "" otherwise —
        match on the filename when you can, so classification does not
        depend on the file being readable yet.
        """
        return file_path.endswith(".myref")

    def file_role(self, file_path):
        """COMBINED (one full curve), PARTIAL (one angle), or UNKNOWN."""
        return COMBINED

    def group_key(self, file_path):
        """Identity of the measurement set, or None if the format has no
        such concept. Files in one state must agree on it."""
        return None

    def header_metadata(self, file_path):
        """theta (0.0 if not a single angle), dq_is_fwhm, num_segments,
        instrument."""
        return {
            "theta": 0.0,
            "dq_is_fwhm": True,
            "num_segments": 0,
            "instrument": self.name,
        }

    def role_supports_nuisance(self, role):
        return role == PARTIAL
```

Every method must tolerate a path that does not exist — instruments are
resolved while parsing a setup file, before any run starts.

### Optional: `authoritative_fields`

`nodes.intake.parse_file_header` asks an LLM to read the header first and
falls back to your `header_metadata`. If your format *specifies* a value
rather than hinting at it, say so and the LLM's reading is overridden:

```python
    authoritative_fields = ("dq_is_fwhm",)
```

ORSO uses this: its `sQz` column is one standard deviation by specification,
so the convention is not something to guess about. REF_L_v2 declares both
`dq_is_fwhm` and `theta` — for `theta` the LLM parse cannot succeed even in
principle, since it is given the header text and not the filename, and that
format's angle can only be found via the segment number the filename carries.
REF_L_v1 declares nothing, leaving the LLM's reading in charge.

### Optional: `run_title`

AuRE records the operator's free-form run title from every file, as
provenance. By default it takes the first `# Run title: …` comment line.
Implement this if your format puts something else there:

```python
    def run_title(self, file_path):
        """This file's own title, or "" if it has none."""
        return ...
```

REF_L's `new_reduction` dialect needs it: its `# Run Title:` line holds a JSON
array of *every* segment's title, identical in every file of the run, so the
generic match captures the array — and because it is identical everywhere, the
cross-file consistency check finds no disagreement to report.

### Optional: `header_issues`

For what the header *says* that is wrong or self-contradictory — as distinct
from the file being unreadable, which every method already handles by
returning defaults:

```python
    def header_issues(self, file_path):
        """Defects worth telling the scientist about; [] if none."""
        return ["Config.ThetaShift has 2 entries for 3 segments"]
```

These reach the run as warnings and are recorded on the dataset in the
checkpoint. They do not stop the run: a header defect is usually survivable,
and refusing to load a file over one would be worse than proceeding with a
stated caveat. What must not happen is proceeding *silently* — a resolution
convention that changed without anyone noticing is how this whole area of the
code came to be written.

Both members are looked up by name rather than required by the protocol, so an
instrument that implements neither behaves exactly as instruments did before
they existed, and one whose implementation raises is logged and skipped.

## Registering it

Three ways, in increasing permanence.

**In-process** — for a script or a notebook:

```python
from aure.instruments import register
register(MyInstrument(), first=True)   # first=True outranks the built-ins
```

**As a plugin** — from your own package's `pyproject.toml`, so it is picked
up wherever AuRE runs:

```toml
[project.entry-points."aure.instruments"]
myref = "my_package.instruments:MyInstrument"
```

The entry point may be a class, a factory, or an instance. A plugin that
fails to load is logged and skipped — a broken instrument cannot stop AuRE
from running on the ones that work.

**Forcing one** — when filenames lie about their origin:

```bash
AURE_INSTRUMENT=MYREF aure analyze data.myref "..."
```

`AURE_INSTRUMENT=generic` disables instrument detection entirely.

## Resolution order

Registration order is priority order; the first instrument to claim a file
wins. Re-registering a `name` replaces the earlier entry rather than
shadowing it, so a plugin can override a built-in.

Two entry points, and the difference matters if you are calling them:

- `resolve_by_name(path)` never opens the file. Used where AuRE classifies
  before the data is necessarily present.
- `resolve(path)` tries the name, then sniffs the header. Used for header
  metadata.

A file nothing claims resolves to `GenericInstrument`: role `UNKNOWN`, no
grouping, default conventions. That is not silent — the run logs a warning
naming the files and the registered instruments, and a setup declaring
`theta_offset` or `sample_broadening` on such files fails at load rather
than proceeding on a guess.

## Asking what AuRE understands

```bash
aure formats                      # the registered instruments, in priority order
aure formats data/*.dat           # what it makes of these files
aure formats file.dat --json      # the same, for a script
```

With files, it reports for each which instrument claimed it and **whether by
filename or by header**, the role and set id, the incident angle and dQ
convention it read, the run title, whether `theta_offset` /
`sample_broadening` apply, and any defects the instrument found in the header.

The distinction between claimed-by-filename and claimed-by-header is worth
attention: a file claimed only by its header still works, but a rename — or
running before the data lands — changes the answer, because
`resolve_by_name` is what runs while a setup file is parsed.

This is the command for "why was my file not recognised?" and "why did it use
that angle?", and it answers both without starting a run. The Setup tab asks
the same question of the same code through `/api/instruments/classify`.

## When AuRE cannot read your format yet

Start with `aure formats <your files>`. If it says *unrecognised*, you have
three options, in increasing order of effort and permanence:

| | What it costs | When it is right |
|---|---|---|
| **1. Declare the values** in the setup file | a line per file | a one-off run, or another program already reads your files and can write the setup |
| **2. `AURE_INSTRUMENT=`** force an existing instrument | a shell variable | your files *are* a known format under an unfamiliar name |
| **3. Write an instrument** | a small class | the knowledge should outlive one setup file |

None of them is a workaround for the others. (1) is not a lesser (3): if
something upstream has already parsed your headers correctly, passing what it
read is better than having AuRE re-derive it. (3) is not overkill for one
beamtime: it is roughly a page of code, and it is the only one that makes the
next person's files work without them knowing any of this.

### 1. Declare the values

A setup file's `data_files` entry may declare the two values AuRE would
otherwise have read from the header, and a declared value wins over the header
parse:

```yaml
states:
  - name: state0
    data_files:
      - {file: seg1.dat, theta: 0.45,  dq_is_fwhm: false}
      - {file: seg2.dat, theta: 1.251, dq_is_fwhm: false}
```

`theta` is the incident angle in **degrees** — the value the header states,
not the nominal setting it was rounded from. `dq_is_fwhm` is `false` when the
fourth column is one standard deviation rather than a full width; the two
differ by 2.355, and a fit absorbs the difference into roughness rather than
reporting it.

Both keys are optional, and an entry that declares neither behaves exactly as
before. Any *other* key on a `data_files` entry is an error — a `thetas:` typo
that parsed and vanished would leave the run quietly using the header value
you believed you had overridden.

It needs no code in AuRE and no release. What it does not do is travel: the
next person with the same files starts from the same place.

### 2. Force an existing instrument

If your files are a format AuRE already knows under a name it does not
recognise — a facility's own naming on top of a standard layout, say:

```bash
AURE_INSTRUMENT=ORSO aure analyze data.txt "..."
```

That applies to *every* file in the process, so it suits a directory of one
kind of data and not a mixed one. `aure formats` lists the names, and
`AURE_INSTRUMENT=generic` turns detection off entirely.

### 3. Write an instrument

The protocol at the top of this page, plus a registration. Roughly a page of
code; the walkthrough is this whole document, and the checklist for doing it
well is the `add-data-format` skill in `.claude/skills/`.

This is the only option that makes the format work for someone who has never
read this page — which, if the format is your facility's, is most of the
people who will meet it.

## Naming, versions and lifecycle

A facility's format changes over time, and two versions of it are routinely
live at once. AuRE names an instrument after **which version of the format it
reads** — `REF_L_v1`, `REF_L_v2` — and keeps every changing judgement about
that version outside the name:

```python
class REFLv2Instrument:
    name = "REF_L_v2"              # stable: which format this is
    format_version = 2
    lifecycle = PROTOTYPE          # revisable: what we think of it
    lifecycle_note = "a stop-gap ahead of a larger reduction rewrite"
    aliases = ("REF_L_autoreduction",)
```

The split is not cosmetic. **`name` is written into every checkpoint that
touches a file**, so it has to mean the same thing years later. A name
describing the pipeline (`REF_L_autoreduction`) is wrong as soon as the
pipeline is renamed; a name carrying the judgement (`REF_L_legacy`) is wrong
as soon as the judgement changes — and correcting either means old checkpoints
refer to an instrument that no longer exists. Which format a file is in does
not change. What we think of that format does.

`lifecycle` is one of `current` (the default, for anything that says nothing),
`legacy`, `prototype` or `deprecated`, and `lifecycle_note` says why in one
line — an unexplained label invites the wrong inference. `legacy` on REF_L_v1
does **not** mean stop using it: that format is what the archive is full of
and what most beamtimes still produce. `prototype` on REF_L_v2 does mean do
not build on its details, and do not assume a v3 will resemble it.

`aliases` lists earlier names the instrument still answers to, so
`AURE_INSTRUMENT=REF_L` keeps working after a rename. Without it the failure
would be invisible: an ignored override just means resolution goes back to
normal, which looks like nothing being wrong.

`aure formats` shows all of it.

## The built-ins

**REF_L_v1** ([`ref_l.py`](../src/aure/instruments/ref_l.py)) — REF_L's
established reduction, and the conventions AuRE was built on. Classifies by
filename: `*_combined_data_auto.txt` is one curve, `*_<n>_<m>_partial.txt` is
one angle, and `REFL_<setid>_...` supplies the group key. A file may have a
role but no group key, since the role patterns do not require the `REFL_`
prefix. Reads theta from the header's `TwoTheta(deg)` table, halving it; a
multi-segment table yields `0.0`, which is what tells `model_builder` to
build a Q-based probe instead of an angle-based one.

**REF_L_v2** ([`ref_l.py`](../src/aure/instruments/ref_l.py)) — REF_L's
`new_reduction` pipeline, `REFL_<run>_<seg>_<subrun>_autoreduction.dat`.
A different dialect from the file above, not a rename: the header is
`# Key = value` lines mixing JSON and Python notation, it describes the **whole
run** and is byte-identical in every one of that run's segment files, and its
fourth column is **one sigma** where the older reduction writes a FWHM. Always
`PARTIAL` — the dialect has no combined form — and it shares REF_L's group key
so a beamtime mid-migration can co-refine both in one state. Declares both
`dq_is_fwhm` and `theta` authoritative; `theta` because the LLM header parse is
given the header text and not the filename, and the angle can only be found by
the segment number the filename carries.

The angle lookup is the part worth knowing about. `Angles.THS` and
`Run Title.title` are longer than the segment count, because the reduction
**appends to them on reprocess instead of replacing them** — a run reduced
twice carries two complete passes. So a file finds its angle by matching the
trailing `-<segment>.` in the title array, taking the **last** match (the first
is the oldest pass, stale by construction), and it warns if the passes
disagree. Positional indexing is accidentally correct for a whole repeated
block and wrong for a ragged one; on run 234277 it gives segment 3 an angle of
1.251° instead of 3.5°, a factor of ~2.8 in Q that fits cleanly to a wrong
thickness.

**ORSO** ([`orso.py`](../src/aure/instruments/orso.py)) — claims `.ort`, or
any file whose header carries the ORSO banner. Parses the commented YAML
header: a single declared `incident_angle` means one angle, an angle range
means a complete curve. Declares `dq_is_fwhm = False`. Returns no group key —
the standard has no field meaning "these files are segments of one
measurement", and inventing one would impose a constraint the format does not
express, so grouping stays the user's to declare through `states:`.

## Testing yours

[`tests/test_instruments.py`](../tests/test_instruments.py) has a
`_FakeInstrument` showing the shape, including the assertions worth copying:
that your files get the role and group key you expect, that registering
yours does not disturb the built-ins, and that `config._detect_kind` reaches
the verdict you intend for a multi-file state.

The file opens with a golden parity table pinning REF_L's classifications.
Leave it alone unless you mean to change what REF_L files mean.
