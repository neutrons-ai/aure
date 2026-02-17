<p align="center">
  <img src="docs/logo.svg" alt="AuRE" width="360">
</p>

<h1 align="center">Automated Reflectivity Evaluator</h1>

AuRE is an intelligent agent for analyzing neutron and X-ray reflectivity data.
It uses an LLM-driven workflow (powered by [LangGraph](https://github.com/langchain-ai/langgraph))
to go from a raw data file and a plain-English sample description to a fitted
[Refl1D](https://refl1d.readthedocs.io) model — automatically.

## How it works

AuRE runs an iterative analysis pipeline:

```mermaid
flowchart LR
    S((Start)) --> Intake
    Intake --> Analysis
    Analysis --> Modeling
    Modeling --> Fitting
    Fitting --> Evaluation

    Evaluation -->|fit acceptable| E((Done))
    Evaluation -->|refine model| Modeling

    Intake  -.->|error| E
    Analysis -.->|error| E
    Modeling -.->|error| E
    Fitting  -.->|error| E

    style S fill:#6c757d,color:#fff,stroke:none
    style E fill:#198754,color:#fff,stroke:none
    style Intake fill:#0d6efd,color:#fff,stroke:none
    style Analysis fill:#0d6efd,color:#fff,stroke:none
    style Modeling fill:#0d6efd,color:#fff,stroke:none
    style Fitting fill:#0d6efd,color:#fff,stroke:none
    style Evaluation fill:#fd7e14,color:#fff,stroke:none
```

1. **Intake** — Loads the reflectivity data file and parses the sample
   description with an LLM to extract structured layer/substrate/ambient
   information (materials, thicknesses, SLDs via `periodictable`).
2. **Analysis** — Extracts physics features from the data: critical edge,
   total thickness from Kiessig fringes, estimated roughness, and layer count.
3. **Modeling** — The LLM generates a Refl1D model script informed by the
   parsed sample and the extracted features.
4. **Fitting** — Runs the generated model through Refl1D's optimizer.
5. **Evaluation** — Assesses the fit quality (χ², residual structure, parameter
   reasonableness) and decides whether the result is acceptable.
6. **Refinement** — If the fit is not good enough, the LLM modifies the model
   (adjusting bounds, adding/removing layers, changing constraints) and the
   loop repeats, up to a configurable number of iterations.

Checkpoints are saved after every stage so you can inspect intermediate results
or resume a run from any point.

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/aure.git
cd aure

# Create a virtual environment and install with the agent extras
python -m venv .venv
source .venv/bin/activate
pip install -e ".[agent]"
```

### Extras

| Extra     | What it adds                                      |
|-----------|---------------------------------------------------|
| `agent`   | LangGraph, LangChain, Click, FastMCP, periodictable — everything needed for the CLI and workflow |
| `alcf`    | `globus-sdk` — native Globus auth for ALCF inference endpoints |
| `dev`     | pytest                                            |
| `all`     | All of the above                                  |

### LLM configuration

AuRE reads its LLM settings from environment variables (or a `.env` file in the
project root).  See [.env.example](.env.example) for every available option.

```bash
LLM_PROVIDER=openai          # "openai", "gemini", "alcf", or "local"
LLM_MODEL=gpt-4o             # model name for your provider
LLM_API_KEY=sk-...           # API key
# LLM_BASE_URL=              # only needed for local / openai-compatible
```

#### ALCF inference endpoints

To use the [ALCF inference service](https://docs.alcf.anl.gov/services/inference-endpoints/)
at Argonne National Laboratory:

```bash
LLM_PROVIDER=alcf
ALCF_CLUSTER=sophia           # "sophia" (vLLM) or "metis" (SambaNova)
LLM_MODEL=gpt-oss-120b        # any model served on the cluster
# ALCF_ACCESS_TOKEN=...       # Globus token (optional – see below)
```

If `ALCF_ACCESS_TOKEN` is not set AuRE will try, in order:

1. **`globus_sdk`** (install with `pip install aure[alcf]`) — reuses cached
   Globus tokens; no subprocess needed.
2. **`inference_auth_token.py get_access_token`** — subprocess fallback.

See the [ALCF docs](https://docs.alcf.anl.gov/services/inference-endpoints/#2-authenticate)
for initial Globus authentication setup.

## CLI reference

After installation the `aure` command is available:

```
aure [OPTIONS] COMMAND [ARGS]...
```

### `aure analyze`

Run a full analysis workflow on a reflectivity data file.

```bash
aure analyze DATA_FILE SAMPLE_DESCRIPTION [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-o, --output-dir PATH` | Save checkpoints and model scripts to this directory |
| `-m, --max-refinements N` | Maximum refinement iterations (default: 5) |
| `-h, --hypothesis TEXT` | Optional hypothesis to test |
| `-v, --verbose` | Stream workflow progress to stderr |
| `--json` | Emit results as JSON |

**Examples:**

```bash
# Basic analysis
aure analyze data.txt "100 nm polystyrene on silicon"

# Save all outputs and increase refinement budget
aure analyze data.txt "Cu/Ti bilayer on Si in dTHF" -o ./output -m 8 -v
```

### `aure batch`

Run one or more analyses from a YAML manifest file.
Ideal for automated / CI workflows where the full configuration lives in
version control.

```bash
aure batch MANIFEST [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-j, --job NAME` | Run only the named job(s). Repeatable. Default: all |
| `--dry-run` | Validate the manifest and print the plan without running |

The manifest is a YAML file with a `defaults` section and a `jobs` list.
See [manifest.example.yaml](manifest.example.yaml) for the full schema.

**Examples:**

```bash
# Run every job in the manifest
aure batch manifest.yaml

# Run a single job
aure batch manifest.yaml -j copper_on_silicon

# Preview without executing
aure batch manifest.yaml --dry-run
```

### `aure resume`

Resume a workflow from a previously saved checkpoint.

```bash
aure resume CHECKPOINT_PATH [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-o, --output-dir PATH` | Write new checkpoints here (defaults to the original) |
| `--fit / --no-fit` | Include or skip the fitting step |
| `-v, --verbose` | Verbose logging |
| `--json` | JSON output |

### `aure checkpoints`

List the checkpoints in an output directory.

```bash
aure checkpoints OUTPUT_DIR [--json]
```

### `aure inspect-checkpoint`

Show details about a single checkpoint file.

```bash
aure inspect-checkpoint CHECKPOINT_PATH [-s] [--json]
```

`-s, --show-state` prints the full workflow state (can be large).

### `aure plot-results`

Plot R(Q) curves and SLD profiles from a completed run.

```bash
aure plot-results OUTPUT_DIR [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-s, --save PATH` | Save the figure (PNG, PDF, SVG) |
| `-f, --offset N` | Vertical offset between curves (default: 10) |
| `--no-show` | Don't open the interactive plot window |
| `-w, --workspace PATH` | Working directory for resolving data file paths |

### `aure extract-features`

Quickly extract physics features from a data file without running a full
analysis.

```bash
aure extract-features DATA_FILE [--json]
```

### `aure lookup-sld`

Look up neutron scattering length densities for one or more materials.

```bash
aure lookup-sld MATERIAL [MATERIAL ...] [-w WAVELENGTH] [--json]
```

```bash
aure lookup-sld silicon gold D2O
aure lookup-sld SiO2 polystyrene PMMA
```

### `aure list-materials`

List known materials in the built-in database.

```bash
aure list-materials [-c CATEGORY] [--json]
```

Categories: `polymers`, `metals`, `substrates`, `solvents`, `all` (default).

### `aure mcp-server`

Start a [Model Context Protocol](https://modelcontextprotocol.io/) server so AI
assistants (e.g. Claude) can drive the workflow interactively.

```bash
aure mcp-server                          # stdio (for Claude Desktop)
aure mcp-server --transport sse --port 8080  # HTTP/SSE
```

### `aure serve`

Launch a local web viewer to explore the results of a completed (or
in-progress) workflow run.

```bash
aure serve OUTPUT_DIR [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-p, --port N` | Port for the local server (default: 5000) |
| `--no-browser` | Don't open a browser automatically |

The viewer has two tabs:

- **History** — step-by-step checkpoint timeline and an interactive χ²
  progression chart (Plotly.js, zoomable).
- **Results** — log-log R(Q) plot with experimental data and model curves
  (zoomable/pannable), SLD depth profile, and a table of best-fit parameters
  with uncertainties.

```bash
aure serve ./output
aure serve ./output --port 8080 --no-browser
```

## Python API

```python
from aure import run_analysis

result = run_analysis(
    data_file="data.txt",
    sample_description="100 nm polystyrene on silicon",
    hypothesis="Single layer",
    max_iterations=5,
    output_dir="./results",
)
```

## Project layout

```
.env.example            # Environment variable reference
manifest.example.yaml   # Batch manifest reference
src/aure/
├── cli.py              # Click CLI (entry point)
├── mcp_server.py       # FastMCP server
├── state.py            # Workflow state definitions
├── database/
│   └── materials.py    # Material SLD lookups (periodictable)
├── llm/                # LLM abstraction layer
│   ├── config.py       # Env-var configuration & availability
│   ├── timeout.py      # Signal-based call timeout
│   └── providers/      # One module per backend
│       ├── openai.py
│       ├── gemini.py
│       ├── alcf.py     # Argonne ALCF inference endpoints
│       └── local.py    # Ollama / LM Studio / vLLM
├── nodes/
│   ├── intake.py       # Data loading & sample parsing
│   ├── analysis.py     # Feature extraction
│   ├── modeling.py     # Refl1D model generation
│   ├── fitting.py      # Model fitting
│   ├── evaluation.py   # Fit quality evaluation
│   ├── refinement.py   # Model refinement
│   ├── routing.py      # Workflow routing decisions
│   └── prompts.py      # LLM prompt templates
├── tools/
│   ├── data_tools.py   # Reflectivity data I/O
│   └── feature_tools.py# Physics feature extraction
├── web/                # Flask results viewer
│   ├── __init__.py     # create_app() factory
│   ├── data.py         # RunData – checkpoint/model reader
│   ├── routes.py       # Blueprint (pages + JSON API)
│   ├── templates/      # Jinja2 (base, history, results)
│   └── static/         # CSS
└── workflow/
    ├── graph.py        # LangGraph workflow definition
    ├── runner.py       # Execution orchestration
    ├── checkpoints.py  # Checkpoint save/load
    └── tracing.py      # LangSmith tracing support
```
