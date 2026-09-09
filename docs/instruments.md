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
    name = "MYREF"                      # recorded in checkpoints

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
so the convention is not something to guess about. REF_L declares nothing,
leaving the LLM's reading in charge.

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

## The built-ins

**REF_L** ([`ref_l.py`](../src/aure/instruments/ref_l.py)) — classifies by
filename: `*_combined_data_auto.txt` is one curve, `*_<n>_<m>_partial.txt` is
one angle, and `REFL_<setid>_...` supplies the group key. A file may have a
role but no group key, since the role patterns do not require the `REFL_`
prefix. Reads theta from the header's `TwoTheta(deg)` table, halving it; a
multi-segment table yields `0.0`, which is what tells `model_builder` to
build a Q-based probe instead of an angle-based one.

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
