"""Flask blueprint – page routes and JSON API endpoints."""

import re
import threading
from pathlib import Path
from typing import Optional

from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from .data import RunData
from ..tools.data_tools import load_reflectivity_data

bp = Blueprint(
    "web",
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static/web",
)


def _run_data() -> Optional[RunData]:
    output_dir = current_app.config.get("OUTPUT_DIR")
    if not output_dir or not Path(output_dir).exists():
        return None
    return RunData(output_dir)


def _has_output() -> bool:
    """Return True when a valid output directory is configured."""
    rd = _run_data()
    return rd is not None and bool(rd.get_run_info())


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _extract_run_name(data_file: str) -> str:
    """
    Derive a short run name from the data-file header.

    Reads the first line looking for ``Run <number>``.  Falls back to
    extracting a 6-digit number from the filename, then the sanitised stem.
    """
    # Try to read the run number from the file header
    try:
        with open(data_file, "r") as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                m = re.search(r"\bRun\s+(\d+)", line)
                if m:
                    return m.group(1)
    except OSError:
        pass

    # Fallback: extract from filename
    stem = Path(data_file).stem
    m = re.search(r"(\d{6})", stem)
    if m:
        return m.group(1)
    return re.sub(r"[^\w\-]", "_", stem)


def _sanitize_dir_name(name: str) -> str:
    """Make a string safe for use as a directory name."""
    return re.sub(r"[^\w\-]", "_", str(name)).strip("_") or "run"


def _parse_optional_float(raw) -> "Optional[float]":
    """A form number that may be blank. Blank/absent means "use the default".

    Invalid text is treated as absent rather than rejected: the setup loader and
    ``_get_chi2_max``/``_get_chi2_min`` validate the value that actually reaches
    them, and failing a whole run because a field held junk would be worse than
    falling back to the configured default.
    """
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _derive_run_dir_name(states, data_files, data_file) -> str:
    """Derive the output/run folder name.

    - Single run/file → its run number, or the state name when unnumbered (never
      a bare sanitised stem if a state name is available).
    - Co-refinement → the per-state primary run numbers joined (e.g.
      ``230539_230543``), so the folder includes every state's run and does NOT
      overwrite an existing single-state fit named by one run. When a state has no
      run number, its name is used.

    Each *group* below is one state (or the flat file list for a single-state
    co-refinement); a group's id is the lowest run number among its files.
    """
    if states:
        groups = [(st.get("data_files") or [], st.get("name")) for st in states]
    elif data_files:
        groups = [(data_files, None)]
    else:
        return _extract_run_name(data_file)

    ids: list[str] = []
    for files, state_name in groups:
        paths = [(df.get("file") if isinstance(df, dict) else df) for df in files]
        run_names = [_extract_run_name(p) for p in paths if p]
        numeric = [int(n) for n in run_names if n.isdigit()]
        if numeric:
            ids.append(str(min(numeric)))
        elif state_name:
            ids.append(_sanitize_dir_name(state_name))
        elif run_names:
            ids.append(_sanitize_dir_name(run_names[0]))

    deduped = list(dict.fromkeys(ids))  # preserve order, drop dups
    if not deduped:
        return _extract_run_name(data_file)
    return "_".join(deduped)


def _apply_overrides_to_model_script(
    script: str,
    parameter_overrides: dict,
    bounds_overrides: dict,
) -> str:
    """Patch a refl1d model script with user parameter/bounds overrides.

    For parameter overrides, update the *initial* value used in the script
    by modifying ``SLD(name=..., rho=<VALUE>)`` and layer constructor args.
    For bounds overrides, update ``.range(lo, hi)`` calls.

    This is best-effort regex patching — the LLM will re-generate the
    model on restart anyway, but giving it updated starting values and
    bounds helps it converge faster.
    """
    # Map friendly names → script patterns.
    # Parameter names from refl1d look like:
    #   "copper thickness", "copper rho", "copper interface",
    #   "intensity REFL_...", "dTHF rho"
    # We inject overrides by updating .range() start values & bounds.

    # For bounds overrides: replace .range(old_lo, old_hi) patterns
    for name, pair in bounds_overrides.items():
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        lo, hi = pair
        # Try to find a .range() call near a comment or variable matching this param
        # Generic approach: find lines with the param name in a comment
        lines = script.split("\n")
        for i, line in enumerate(lines):
            if ".range(" in line and (
                name.lower().replace(" ", "_") in line.lower()
                or name.lower() in line.lower()
            ):
                lines[i] = re.sub(
                    r"\.range\([^)]*\)",
                    f".range({lo}, {hi})",
                    line,
                )
                break
        script = "\n".join(lines)

    return script


def _apply_overrides_to_model_definition(
    definition: dict,
    parameter_overrides: dict,
    bounds_overrides: dict,
) -> dict:
    """Apply user parameter/bounds overrides to a JSON ModelDefinition.

    Returns a new dict with updated starting values and bounds.  Parameter
    names follow the refl1d convention: ``"<material> <attribute>"``.
    """
    import copy

    defn = copy.deepcopy(definition)

    # Build a lookup: (material, attr) → value
    p_lookup: dict[tuple, float] = {}
    for name, val in parameter_overrides.items():
        parts = name.rsplit(" ", 1)
        if len(parts) == 2:
            try:
                p_lookup[(parts[0], parts[1])] = float(val)
            except (ValueError, TypeError):
                pass

    b_lookup: dict[tuple, list] = {}
    for name, pair in bounds_overrides.items():
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        parts = name.rsplit(" ", 1)
        if len(parts) == 2:
            try:
                b_lookup[(parts[0], parts[1])] = [float(pair[0]), float(pair[1])]
            except (ValueError, TypeError):
                pass

    # Apply to layers
    attr_map = {
        "rho": "sld",
        "thickness": "thickness",
        "interface": "roughness",
    }
    bound_suffixes = {
        "sld": ("sld_min", "sld_max"),
        "thickness": ("thickness_min", "thickness_max"),
        "roughness": ("roughness_min", "roughness_max"),
    }

    for layer in defn.get("layers", []):
        mat_name = layer.get("name", "")
        for refl1d_attr, json_key in attr_map.items():
            key = (mat_name, refl1d_attr)
            if key in p_lookup:
                layer[json_key] = p_lookup[key]
            b_key = key
            if b_key in b_lookup and json_key in bound_suffixes:
                lo_k, hi_k = bound_suffixes[json_key]
                layer[lo_k] = b_lookup[b_key][0]
                layer[hi_k] = b_lookup[b_key][1]

    # Apply to substrate
    sub = defn.get("substrate", {})
    sub_name = sub.get("name", "")
    if (sub_name, "interface") in p_lookup:
        sub["roughness"] = p_lookup[(sub_name, "interface")]
    if (sub_name, "interface") in b_lookup:
        sub["roughness_max"] = b_lookup[(sub_name, "interface")][1]

    return defn


# ------------------------------------------------------------------
# Page routes
# ------------------------------------------------------------------


def _minimal_state_prefill(states: list[dict]) -> list[dict]:
    """Return a lightweight states payload for setup-page prefill."""
    out = []
    for st in states:
        if not isinstance(st, dict):
            continue
        slim: dict = {"name": st.get("name", "")}
        data_files = []
        for df in st.get("data_files") or []:
            if not isinstance(df, dict):
                continue
            file_path = df.get("file")
            if not file_path:
                continue
            data_files.append({"file": file_path, "label": df.get("label", "")})
        slim["data_files"] = data_files
        ambient = st.get("ambient")
        if isinstance(ambient, dict) and "rho" in ambient:
            slim["ambient"] = {"rho": ambient["rho"]}
        for k in ("intensity", "theta_offset", "sample_broadening", "background"):
            v = st.get(k)
            if isinstance(v, dict):
                slim[k] = {sub: v[sub] for sub in ("init", "min", "max") if sub in v}
        if "back_reflection" in st:
            slim["back_reflection"] = bool(st["back_reflection"])
        if st.get("extra_description"):
            slim["extra_description"] = st["extra_description"]
        out.append(slim)
    return out


@bp.route("/")
def index():
    """Landing page – setup form or redirect to results."""
    output_dir = current_app.config.get("OUTPUT_DIR")
    run_state = current_app.config["RUN_STATE"]

    # If we have a pre-loaded output dir (legacy serve mode) redirect
    if output_dir and Path(output_dir).exists() and run_state["status"] == "idle":
        ri = _run_data()
        if ri and ri.get_run_info():
            return redirect(url_for("web.history"))

    return render_template("setup.html", active_tab="setup", run_state=run_state)


@bp.route("/setup")
def setup():
    """Setup page – always accessible, pre-populated from previous run."""
    run_state = current_app.config["RUN_STATE"]
    prev_run = {}
    rd = _run_data()
    if rd:
        ri = rd.get_run_info()
        if ri:
            prev_run = {
                "data_file": ri.get("data_file", ""),
                "sample_description": ri.get("sample_description", ""),
                "hypothesis": ri.get("hypothesis") or "",
                "output_dir": str(
                    Path(current_app.config.get("OUTPUT_DIR", "")).parent
                ),
                "data_files": ri.get("data_files", []),
            }
            # Pull states + shared/unshared from the final state for
            # multi-state prefill (Ticket 16).
            try:
                fs = rd.get_final_state() or {}
                states = fs.get("states") or []
                if states:
                    prev_run["states"] = _minimal_state_prefill(states)
                model = fs.get("current_model") or fs.get("best_model") or {}
                if isinstance(model, dict):
                    if model.get("shared_parameters"):
                        prev_run["shared_parameters"] = list(model["shared_parameters"])
                    if model.get("unshared_parameters"):
                        prev_run["unshared_parameters"] = list(
                            model["unshared_parameters"]
                        )
                    if model.get("distinct_sample"):
                        prev_run["distinct_sample"] = bool(model["distinct_sample"])
            except Exception:
                # Prefill is best-effort; never block the setup page.
                pass
    return render_template(
        "setup.html",
        active_tab="setup",
        run_state=run_state,
        prev_run=prev_run,
    )


@bp.route("/history")
def history():
    if not _has_output():
        flash("No analysis results yet – start one first.", "warning")
        return redirect(url_for("web.index"))
    rd = _run_data()
    return render_template(
        "history.html",
        run_info=rd.get_run_info(),
        active_tab="history",
    )


@bp.route("/results")
def results():
    if not _has_output():
        flash("No analysis results yet – start one first.", "warning")
        return redirect(url_for("web.index"))
    rd = _run_data()
    return render_template(
        "results.html",
        run_info=rd.get_run_info(),
        active_tab="results",
    )


# ------------------------------------------------------------------
# JSON API – existing data endpoints
# ------------------------------------------------------------------


@bp.route("/api/run-info")
def api_run_info():
    rd = _run_data()
    if not rd:
        return jsonify({})
    return jsonify(rd.get_run_info())


@bp.route("/api/chi2")
def api_chi2():
    rd = _run_data()
    if not rd:
        return jsonify([])
    return jsonify(rd.get_chi2_progression())


@bp.route("/api/reflectivity")
def api_reflectivity():
    rd = _run_data()
    if not rd:
        return jsonify({"Q": [], "R": [], "dR": [], "models": []})
    return jsonify(rd.get_reflectivity_data())


@bp.route("/api/reflectivity-file")
def api_reflectivity_file():
    """Load Q/R/dR arrays for an arbitrary reflectivity data file."""
    raw_path = (request.args.get("path") or "").strip()
    if not raw_path:
        return jsonify({"error": "path is required"}), 400

    path = _safe_path(raw_path)
    if path is None or not path.is_file():
        return jsonify({"error": "File does not exist"}), 400

    try:
        data = load_reflectivity_data(str(path))
    except Exception as exc:
        return jsonify({"error": f"Could not parse reflectivity file: {exc}"}), 400

    return jsonify(
        {
            "path": str(path),
            "Q": data["Q"].tolist() if "Q" in data else [],
            "R": data["R"].tolist() if "R" in data else [],
            "dR": data["dR"].tolist() if "dR" in data else [],
        }
    )


@bp.route("/api/sld")
def api_sld():
    rd = _run_data()
    if not rd:
        return jsonify({"profiles": []})
    return jsonify(rd.get_sld_profiles())


@bp.route("/api/parameters")
def api_parameters():
    rd = _run_data()
    if not rd:
        return jsonify({"parameters": []})
    iteration = request.args.get("iteration", default=None, type=int)
    return jsonify(rd.get_fit_parameters(iteration=iteration))


@bp.route("/api/intake-report")
def api_intake_report():
    """Return a summary of what the intake stage concluded about the data."""
    rd = _run_data()
    if not rd:
        return jsonify({"files": [], "warnings": [], "messages": []})
    return jsonify(rd.get_intake_report())


@bp.route("/api/simulate", methods=["POST"])
def api_simulate():
    """Compute reflectivity/SLD for user-adjusted parameters.

    Expects JSON body::

        {
            "parameters": {"param name": value, ...},
            "bounds": {"param name": [lo, hi], ...}   // optional
        }

    Returns ``{Q_fit, R_fit, sld_z, sld_rho, chi_squared}``.
    """
    rd = _run_data()
    if not rd:
        return jsonify({"error": "No analysis output found"}), 404

    body = request.get_json(silent=True) or {}
    parameters = body.get("parameters")
    if not parameters or not isinstance(parameters, dict):
        return jsonify({"error": "parameters dict is required"}), 400

    # Validate all values are numeric
    try:
        parameters = {str(k): float(v) for k, v in parameters.items()}
    except (ValueError, TypeError):
        return jsonify({"error": "All parameter values must be numeric"}), 400

    # Optional bounds overrides from the UI
    raw_bounds = body.get("bounds")
    bounds: dict | None = None
    if raw_bounds and isinstance(raw_bounds, dict):
        try:
            bounds = {
                str(k): [float(v[0]), float(v[1])]
                for k, v in raw_bounds.items()
                if isinstance(v, list) and len(v) == 2
            }
        except (ValueError, TypeError, IndexError):
            bounds = None

    iteration = body.get("iteration")
    if iteration is not None:
        try:
            iteration = int(iteration)
        except (ValueError, TypeError):
            iteration = None

    result = rd.simulate(parameters, bounds=bounds, iteration=iteration)
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result)


@bp.route("/api/export-model")
def api_export_model():
    """Export the best-fit model as a refl1d problem.json.

    Returns the ``problem.json`` from the best fit iteration, which can
    be loaded with ``bumps.serialize.deserialize()``.

    Returns ``{"problem": <json>}`` or an error.
    """
    rd = _run_data()
    if not rd:
        return jsonify({"error": "No analysis output found"}), 404

    # Try top-level problem.json first (copied by checkpoint manager)
    import json as _json

    top_level = rd.output_dir / "problem.json"
    if top_level.exists():
        with open(top_level) as f:
            return jsonify({"problem": _json.load(f)})

    # Fall back to searching refl1d_output/
    refl1d_dir = rd.output_dir / "refl1d_output"
    if refl1d_dir.exists():
        fit_dirs = sorted(refl1d_dir.glob("fit_iter*_*"))
        for fit_dir in reversed(fit_dirs):
            pj = fit_dir / "problem.json"
            if pj.exists():
                with open(pj) as f:
                    return jsonify({"problem": _json.load(f)})

    return jsonify({"error": "No problem.json found"}), 404


@bp.route("/api/llm-status")
def api_llm_status():
    rd = _run_data()
    if not rd:
        return jsonify(
            {
                "total": 0,
                "succeeded": 0,
                "failed": 0,
                "used_fallback": 0,
                "all_ok": True,
                "calls": [],
            }
        )
    return jsonify(rd.get_llm_summary())


# ------------------------------------------------------------------
# JSON API – server-side file / folder browsing
# ------------------------------------------------------------------


def _safe_path(raw: str) -> Optional[Path]:
    """Resolve and validate a path. Return None if unsafe."""
    try:
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            return None
        return p
    except Exception:
        return None


@bp.route("/api/browse-files")
def api_browse_files():
    """
    List files and directories at a given path.

    Query params:
        path  – directory to list (default: home dir)
        ext   – optional extension filter, e.g. ".txt" or ".txt,.refl"
    """
    raw = request.args.get("path", str(Path.home()))
    ext = request.args.get("ext", "")
    exts = {s.lower().strip() for s in ext.split(",") if s and s.strip()}
    target = _safe_path(raw)
    if target is None:
        return jsonify({"error": "Path does not exist"}), 400
    if not target.is_dir():
        target = target.parent

    entries = []
    try:
        for child in sorted(
            target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        ):
            if child.name.startswith("."):
                continue
            if child.is_dir():
                entries.append({"name": child.name, "is_dir": True, "path": str(child)})
            elif not exts or child.suffix.lower() in exts:
                entries.append(
                    {"name": child.name, "is_dir": False, "path": str(child)}
                )
    except PermissionError:
        return jsonify({"error": "Permission denied"}), 403

    parent = str(target.parent) if target.parent != target else None
    return jsonify({"current": str(target), "parent": parent, "entries": entries})


@bp.route("/api/browse-dirs")
def api_browse_dirs():
    """
    List only directories at a given path (for the output-folder picker).

    Query params:
        path – directory to list (default: cwd)
    """
    raw = request.args.get("path", str(Path.cwd()))
    target = _safe_path(raw)
    if target is None:
        return jsonify({"error": "Path does not exist"}), 400
    if not target.is_dir():
        target = target.parent

    entries = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if child.name.startswith("."):
                continue
            if child.is_dir():
                entries.append({"name": child.name, "path": str(child)})
    except PermissionError:
        return jsonify({"error": "Permission denied"}), 403

    parent = str(target.parent) if target.parent != target else None
    return jsonify({"current": str(target), "parent": parent, "entries": entries})


# ------------------------------------------------------------------
# JSON API – analysis lifecycle
# ------------------------------------------------------------------


_PREVIEW_LOCK = threading.Lock()


@bp.route("/api/preview-structure", methods=["POST"])
def api_preview_structure():
    """Intake → analysis → modeling preview.

    Returns the parsed layer names and parameter dotted-names that a
    multi-state co-refinement can tie across states. Does NOT touch the
    global run state — the UI can call this while staring at the setup
    page and have it complete in seconds.

    Expects the same JSON body shape as ``/api/start-analysis`` (states
    or data_file), but ``output_dir`` is ignored.

    Returns ``{layers: [{name}], parameters: [str, ...]}`` on success,
    ``{errors: [str, ...]}`` with HTTP 400 on validation failure, or a
    409 if another preview is already in flight.
    """
    if not _PREVIEW_LOCK.acquire(blocking=False):
        return jsonify({"errors": ["Another preview is already running."]}), 409
    try:
        body = request.get_json(silent=True) or {}
        data_file = (body.get("data_file") or "").strip()
        sample_description = (body.get("sample_description") or "").strip()
        hypothesis = (body.get("hypothesis") or "").strip() or None
        states_body = body.get("states")
        shared_parameters = body.get("shared_parameters")
        unshared_parameters = body.get("unshared_parameters")
        distinct_sample = body.get("distinct_sample")
        user_config_extra = body.get("user_config") or {}

        errors: list[str] = []
        if not sample_description:
            errors.append("sample_description is required")

        states = None
        states_data_files = None
        if states_body is not None:
            if not isinstance(states_body, list):
                return jsonify({"errors": ["states must be a list"]}), 400
            if shared_parameters is not None and unshared_parameters is not None:
                return jsonify(
                    {
                        "errors": [
                            "shared_parameters and unshared_parameters are mutually exclusive"
                        ]
                    }
                ), 400
            for sidx, st in enumerate(states_body):
                if (
                    not isinstance(st, dict)
                    or "name" not in st
                    or "data_files" not in st
                ):
                    errors.append(
                        f"states[{sidx}]: each entry must have 'name' and 'data_files'"
                    )
                    continue
                for fidx, df in enumerate(st.get("data_files") or []):
                    fpath = df.get("file") if isinstance(df, dict) else None
                    if not fpath or not Path(fpath).is_file():
                        errors.append(
                            f"states[{sidx}].data_files[{fidx}]: file does not exist: {fpath}"
                        )
            if errors:
                return jsonify({"errors": errors}), 400
            from ..config import states_from_config, ConfigError
            from ..state import flatten_data_files as _flatten

            cfg_for_states = {"states": states_body}
            if shared_parameters is not None:
                cfg_for_states["shared_parameters"] = shared_parameters
            if unshared_parameters is not None:
                cfg_for_states["unshared_parameters"] = unshared_parameters
            try:
                states = states_from_config(cfg_for_states)
            except ConfigError as exc:
                return jsonify({"errors": [f"states: {exc}"]}), 400
            flat = _flatten(states)
            if flat:
                data_file = flat[0].get("file") or data_file
            if shared_parameters is not None:
                user_config_extra = {
                    **user_config_extra,
                    "shared_parameters": shared_parameters,
                }
            if unshared_parameters is not None:
                user_config_extra = {
                    **user_config_extra,
                    "unshared_parameters": unshared_parameters,
                }
            if distinct_sample is not None:
                user_config_extra = {
                    **user_config_extra,
                    "distinct_sample": bool(distinct_sample),
                }
        else:
            if not data_file or not Path(data_file).is_file():
                errors.append("data_file: file does not exist")
            if errors:
                return jsonify({"errors": errors}), 400

        from ..workflow.runner import run_prepare

        try:
            result = run_prepare(
                data_file=data_file,
                sample_description=sample_description,
                hypothesis=hypothesis,
                output_dir=None,
                user_config=user_config_extra or None,
                data_files=states_data_files,
                states=states,
            )
        except Exception as exc:
            return jsonify({"errors": [f"preview failed: {exc}"]}), 400

        # Extract layer names + tieable parameter dotted-names from the
        # ModelDefinition produced by the modeling node.
        model = result.get("current_model") or {}
        layers = []
        params: list[str] = []
        states_out: list[dict] = []
        if isinstance(model, dict):

            def _layer_names(layer_list):
                return [
                    layer.get("name")
                    for layer in layer_list or []
                    if isinstance(layer, dict) and layer.get("name")
                ]

            template_names = _layer_names(model.get("layers"))
            for name in template_names:
                layers.append({"name": name})
                for attr in ("thickness", "material.rho", "interface"):
                    params.append(f"{name}.{attr}")
            params.append("substrate.interface")

            # Per-state structure (sample != structure): surface any state that
            # carries its own stack so the UI can show how it diverges from the
            # template (which layers it omits / adds).
            template_set = set(template_names)
            for st in model.get("states", []) or []:
                if not isinstance(st, dict) or not st.get("layers"):
                    continue
                names = _layer_names(st.get("layers"))
                name_set = set(names)
                states_out.append(
                    {
                        "name": st.get("name"),
                        "layers": names,
                        "omits": [n for n in template_names if n not in name_set],
                        "adds": [n for n in names if n not in template_set],
                    }
                )

        return jsonify(
            {
                "layers": layers,
                "parameters": params,
                "states": states_out,
                "errors": [],
            }
        )
    finally:
        _PREVIEW_LOCK.release()


@bp.route("/api/setup/load", methods=["POST"])
def api_setup_load():
    """Parse a setup YAML and return the prefill payload the form expects.

    Accepts EITHER a multipart upload (key ``file``) OR a JSON body
    ``{"yaml": "<text>"}``. Returns a dict shaped like the ``prev_run``
    payload built by :func:`setup` so the JS can reuse its existing
    prefill code unchanged.
    """
    import tempfile

    from ..config import ConfigError
    from ..setup import load_setup

    yaml_text: Optional[str] = None
    upload = request.files.get("file")
    if upload is not None:
        try:
            yaml_text = upload.read().decode("utf-8")
        except UnicodeDecodeError:
            return jsonify({"errors": ["uploaded file is not valid UTF-8"]}), 400
    else:
        body = request.get_json(silent=True) or {}
        yaml_text = body.get("yaml")

    if not yaml_text:
        return jsonify(
            {"errors": ["expected a `file` upload or {'yaml': <text>} JSON body"]}
        ), 400

    # load_setup wants a file path; write to a temp file then parse so
    # path resolution inside the YAML (relative `data_files`) keeps
    # working when the user uploads from elsewhere.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(yaml_text)
        tmp_path = tmp.name

    try:
        setup = load_setup(tmp_path)
    except ConfigError as exc:
        return jsonify({"errors": [str(exc)]}), 400
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass

    # Reshape into the prefill payload the JS prefill code knows about.
    states = setup.get("states") or []
    payload = {
        "sample_description": setup.get("sample_description", ""),
        "hypothesis": setup.get("hypothesis", "") or "",
        "data_files": [
            {"file": ds["file"], "label": ds.get("label", "")}
            for st in states
            for ds in st.get("data_files") or []
        ],
        "states": _minimal_state_prefill(states),
    }
    if setup.get("shared_parameters"):
        payload["shared_parameters"] = list(setup["shared_parameters"])  # type: ignore[arg-type]
    if setup.get("unshared_parameters"):
        payload["unshared_parameters"] = list(setup["unshared_parameters"])  # type: ignore[arg-type]
    if setup.get("distinct_sample"):
        payload["distinct_sample"] = bool(setup["distinct_sample"])
    if setup.get("model_name"):
        payload["model_name"] = setup["model_name"]
    if setup.get("max_refinements") is not None:
        payload["max_refinements"] = setup["max_refinements"]
    for key in ("chi2_max", "chi2_min"):
        if setup.get(key) is not None:
            payload[key] = setup[key]
    return jsonify(payload)


@bp.route("/api/setup/export", methods=["POST"])
def api_setup_export():
    """Render the current form state as a downloadable setup YAML.

    Accepts the same JSON body shape as ``/api/start-analysis`` (states +
    sample_description + hypothesis + …). Validates via
    :func:`aure.setup._setup_from_dict` then dumps with
    :func:`aure.setup.dump_setup`. Returns YAML as ``text/yaml``.
    """
    from ..config import ConfigError
    from ..setup import _setup_from_dict, dump_setup

    body = request.get_json(silent=True) or {}

    # The web form posts the same body as /api/start-analysis, which carries
    # runtime-only keys not part of the states-only setup schema. Translate:
    #   * max_iterations -> max_refinements (the setup-file field name)
    #   * synthesize a single `state0` from the flat data_file/data_files when
    #     no explicit `states` block is present (mirrors `aure analyze
    #     DATA_FILE`), so an ad-hoc save still produces a valid setup
    #   * drop the flat data keys and the runtime keys (output_dir, interactive)
    if "max_iterations" in body and "max_refinements" not in body:
        body = {**body, "max_refinements": body["max_iterations"]}

    if not body.get("states"):
        flat = body.get("data_files")
        if not flat and body.get("data_file"):
            flat = [{"file": body["data_file"]}]
        synthesized: list[dict] = []
        for df in flat or []:
            if isinstance(df, dict) and df.get("file"):
                synthesized.append({"file": df["file"], "label": df.get("label", "")})
            elif isinstance(df, str) and df:
                synthesized.append({"file": df})
        if synthesized:
            body = {**body, "states": [{"name": "state0", "data_files": synthesized}]}

    _runtime_only = (
        "data_file",
        "data_files",
        "output_dir",
        "interactive",
        "max_iterations",
    )
    body = {k: v for k, v in body.items() if k not in _runtime_only}

    if not body.get("states"):
        return jsonify(
            {"errors": ["at least one state must be declared under `states:`"]}
        ), 400

    try:
        setup = _setup_from_dict(body, base_dir=Path.cwd(), source="<web form>")
    except ConfigError as exc:
        return jsonify({"errors": [str(exc)]}), 400

    try:
        yaml_text = dump_setup(setup)
    except Exception as exc:
        return jsonify({"errors": [f"dump failed: {exc}"]}), 500

    # Render filename: <model_name>.yaml or "setup.yaml"
    fname = (setup.get("model_name") or setup.get("name") or "setup") + ".yaml"
    return (
        yaml_text,
        200,
        {
            "Content-Type": "text/yaml; charset=utf-8",
            "Content-Disposition": f'attachment; filename="{fname}"',
        },
    )


@bp.route("/api/known-shared-params", methods=["GET"])
def api_known_shared_params():
    """Return distinct shared/unshared parameter names from past runs.

    Walks the output root (parent of the active OUTPUT_DIR) looking for
    ``final_state.json`` files. Collects unique parameter names from
    ``current_model.shared_parameters`` and ``current_model.unshared_parameters``.
    Results are sorted and capped at 200 entries.
    """
    import json as _json

    active = current_app.config.get("OUTPUT_DIR") or ""
    if not active:
        return jsonify({"parameters": []})
    try:
        root = Path(active).expanduser().resolve().parent
    except Exception:
        return jsonify({"parameters": []})
    if not root.is_dir():
        return jsonify({"parameters": []})

    names: set[str] = set()
    try:
        candidates = sorted(root.glob("*/final_state.json"))
    except Exception:
        candidates = []
    for fs in candidates[:500]:
        try:
            # Safe-path check: must be inside `root`.
            resolved = fs.resolve()
            resolved.relative_to(root)
        except Exception:
            continue
        try:
            payload = _json.loads(fs.read_text())
        except Exception:
            continue
        state = payload.get("state") or {}
        cm = state.get("current_model") or {}
        bm = state.get("best_model") or {}
        for model in (cm, bm):
            if not isinstance(model, dict):
                continue
            for key in ("shared_parameters", "unshared_parameters"):
                for p in model.get(key, []) or []:
                    if isinstance(p, str) and p.strip():
                        names.add(p.strip())
        if len(names) >= 200:
            break

    return jsonify({"parameters": sorted(names)[:200]})


@bp.route("/api/start-analysis", methods=["POST"])
def api_start_analysis():
    """
    Launch a background analysis run.

    Expects JSON body::

        {
            "data_file": "/abs/path/to/data.txt",
            "sample_description": "...",
            "hypothesis": "...",       // optional
            "output_dir": "/abs/path"  // root output dir
        }
    """
    lock: threading.Lock = current_app.config["RUN_LOCK"]
    run_state: dict = current_app.config["RUN_STATE"]

    with lock:
        if run_state["status"] == "running":
            return jsonify({"error": "An analysis is already running"}), 409

    body = request.get_json(silent=True) or {}
    data_file = (body.get("data_file") or "").strip()
    sample_description = (body.get("sample_description") or "").strip()
    hypothesis = (body.get("hypothesis") or "").strip() or None
    output_root = (body.get("output_dir") or "").strip()

    # ---- Validation ------------------------------------------------
    states_body_present = body.get("states") is not None
    errors = []
    if not states_body_present and (not data_file or not Path(data_file).is_file()):
        errors.append("data_file: file does not exist")
    if not sample_description:
        errors.append("sample_description is required")
    if not output_root:
        errors.append("output_dir is required")
    if errors:
        return jsonify({"errors": errors}), 400

    # Determine run sub-directory
    data_files = body.get("data_files")  # list of {file, label} or None
    states_body = body.get("states")  # list of state dicts (multi-state) or None
    user_config_extra = body.get("user_config") or {}
    shared_parameters = body.get("shared_parameters")
    unshared_parameters = body.get("unshared_parameters")
    distinct_sample = body.get("distinct_sample")

    states = None
    if states_body is not None:
        if not isinstance(states_body, list):
            return jsonify({"errors": ["states must be a list"]}), 400
        if data_files is not None:
            return jsonify(
                {"errors": ["cannot combine `states` with `data_files`"]}
            ), 400
        if shared_parameters is not None and unshared_parameters is not None:
            return jsonify(
                {
                    "errors": [
                        "shared_parameters and unshared_parameters are mutually exclusive"
                    ]
                }
            ), 400
        # Validate every file exists
        st_errors = []
        for sidx, st in enumerate(states_body):
            if not isinstance(st, dict) or "name" not in st or "data_files" not in st:
                st_errors.append(
                    f"states[{sidx}]: each entry must have 'name' and 'data_files'"
                )
                continue
            for fidx, df in enumerate(st.get("data_files") or []):
                fpath = df.get("file") if isinstance(df, dict) else None
                if not fpath or not Path(fpath).is_file():
                    st_errors.append(
                        f"states[{sidx}].data_files[{fidx}]: file does not exist: {fpath}"
                    )
        if st_errors:
            return jsonify({"errors": st_errors}), 400

        # Normalise via states_from_config (mirrors the CLI YAML path).
        from ..config import states_from_config, ConfigError

        cfg_for_states = {"states": states_body}
        if shared_parameters is not None:
            cfg_for_states["shared_parameters"] = shared_parameters
        if unshared_parameters is not None:
            cfg_for_states["unshared_parameters"] = unshared_parameters
        try:
            states = states_from_config(cfg_for_states)
        except ConfigError as exc:
            return jsonify({"errors": [f"states: {exc}"]}), 400

        # Override the positional data_file with the first state's first file
        from ..state import flatten_data_files as _flatten

        flat = _flatten(states)
        if flat:
            data_file = flat[0].get("file") or data_file
        # Carry shared/unshared parameters into user_config so the modeling
        # node sees the user's tie set.
        if shared_parameters is not None:
            user_config_extra = {
                **user_config_extra,
                "shared_parameters": shared_parameters,
            }
        if unshared_parameters is not None:
            user_config_extra = {
                **user_config_extra,
                "unshared_parameters": unshared_parameters,
            }
        if distinct_sample is not None:
            user_config_extra = {
                **user_config_extra,
                "distinct_sample": bool(distinct_sample),
            }
    elif data_files is not None:
        # Validate data_files structure and file existence
        if not isinstance(data_files, list):
            return jsonify({"errors": ["data_files must be a list"]}), 400
        df_errors = []
        for idx, df in enumerate(data_files):
            if not isinstance(df, dict) or "file" not in df or "label" not in df:
                df_errors.append(
                    f"data_files[{idx}]: each entry must have 'file' and 'label' keys"
                )
            elif not Path(df["file"]).is_file():
                df_errors.append(
                    f"data_files[{idx}]: file does not exist: {df['file']}"
                )
        if df_errors:
            return jsonify({"errors": df_errors}), 400

    # Co-refinement folders include every state's run (e.g. "230539_230543") so
    # they never overwrite a single-state fit; unnumbered runs fall back to the
    # state name. See _derive_run_dir_name.
    run_name = _derive_run_dir_name(states, data_files, data_file)
    output_dir = str(Path(output_root).expanduser().resolve() / run_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    interactive = bool(body.get("interactive", False))
    max_iterations = int(body.get("max_iterations", 5))
    from math import isfinite

    chi2_max = _parse_optional_float(body.get("chi2_max"))
    chi2_min = _parse_optional_float(body.get("chi2_min"))

    if chi2_max is not None and (not isfinite(chi2_max) or chi2_max <= 0):
        return jsonify({"errors": ["chi2_max must be a positive, finite number"]}), 400
    if chi2_min is not None and (not isfinite(chi2_min) or chi2_min < 0):
        return jsonify({"errors": ["chi2_min must be a non-negative, finite number"]}), 400
    if chi2_max is not None and chi2_min is not None and chi2_min >= chi2_max:
        return jsonify({"errors": ["chi2_min must be below chi2_max"]}), 400

    # Reset run state
    with lock:
        run_state.update(
            {
                "status": "running",
                "output_dir": output_dir,
                "current_node": None,
                "iteration": 0,
                "checkpoints": [],
                "error": None,
                "interactive": interactive,
                "messages": [],
                "_pause_event": None,
                "_user_feedback": None,
                "_stop_requested": False,
            }
        )

    # Store the Flask app reference for the background thread
    app = current_app._get_current_object()

    def _run_in_background():
        from ..workflow.runner import run_analysis as _run_analysis

        def _checkpoint_cb(state, node_name):
            with lock:
                run_state["current_node"] = node_name
                run_state["iteration"] = state.get("iteration", 0)
                # Compute LLM calls for this step by diffing cumulative list
                all_llm = state.get("llm_calls", [])
                prev_count = sum(
                    len(cp.get("llm_calls", [])) for cp in run_state["checkpoints"]
                )
                step_llm = all_llm[prev_count:]
                run_state["checkpoints"].append(
                    {
                        "node": node_name,
                        "iteration": state.get("iteration", 0),
                        "chi2": state.get("current_chi2"),
                        "llm_calls": step_llm,
                    }
                )
                # Capture experimental data (once) and fit results for live plots
                if "Q" not in run_state and state.get("Q"):
                    run_state["Q"] = state["Q"]
                    run_state["R"] = state["R"]
                    run_state["dR"] = state.get("dR", [])
                if "data_files" not in run_state and state.get("data_files"):
                    run_state["data_files"] = state["data_files"]
                if state.get("fit_results"):
                    run_state["fit_results"] = list(state["fit_results"])

        pause_callback = None
        if interactive:
            pause_event = threading.Event()
            with lock:
                run_state["_pause_event"] = pause_event

            def _pause_cb(state, node_name):
                """Block until user submits feedback or continues."""
                # Collect messages for the chat panel
                msgs = [
                    {"role": m["role"], "content": m["content"]}
                    for m in state.get("messages", [])
                ]
                with lock:
                    run_state["status"] = "waiting_for_user"
                    run_state["messages"] = msgs
                    run_state["_user_feedback"] = None
                    pause_event.clear()

                pause_event.wait()  # block indefinitely

                with lock:
                    run_state["status"] = "running"
                    feedback = run_state.get("_user_feedback")
                    stop_requested = run_state.get("_stop_requested", False)
                if stop_requested:
                    return "__STOP__"  # sentinel recognized by runner
                return feedback

            pause_callback = _pause_cb

        try:
            _run_analysis(
                data_file=data_file,
                sample_description=sample_description,
                hypothesis=hypothesis,
                max_iterations=max_iterations,
                output_dir=output_dir,
                checkpoint_callback=_checkpoint_cb,
                interactive=interactive,
                pause_callback=pause_callback,
                data_files=data_files,
                states=states,
                user_config=user_config_extra or None,
                chi2_max=chi2_max,
                chi2_min=chi2_min,
            )
            with lock:
                run_state["status"] = "complete"
            # Update app config so history/results pages work
            app.config["OUTPUT_DIR"] = output_dir
        except Exception as exc:
            with lock:
                run_state["status"] = "error"
                run_state["error"] = str(exc)

    t = threading.Thread(target=_run_in_background, daemon=True)
    t.start()

    return jsonify({"status": "started", "output_dir": output_dir})


@bp.route("/api/live/results")
def api_live_results():
    """Return reflectivity, SLD, and parameter data from the live run."""
    lock: threading.Lock = current_app.config["RUN_LOCK"]
    run_state: dict = current_app.config["RUN_STATE"]

    with lock:
        fit_results = run_state.get("fit_results", [])
        Q = run_state.get("Q", [])
        R = run_state.get("R", [])
        dR = run_state.get("dR", [])
        data_files = run_state.get("data_files", [])

    if not fit_results:
        return jsonify(
            {
                "Q": [],
                "R": [],
                "dR": [],
                "models": [],
                "profiles": [],
                "parameters": [],
            }
        )

    has_multi = len(data_files) > 1 and any(ds.get("Q") for ds in data_files)

    # Build model curves
    models = []
    profiles = []
    for fr in fit_results:
        it = fr.get("iteration", 0)
        chi2 = fr.get("chi_squared")
        per_file = fr.get("per_file_results") or []

        if has_multi and per_file:
            for pf in per_file:
                pf_label = pf.get("label", "?")
                pf_chi2 = pf.get("chi_squared")
                label = f"{pf_label} \u2013 iter {it}"
                if pf_chi2 is not None:
                    label += f" (\u03c7\u00b2={pf_chi2:.2f})"
                if pf.get("Q_fit") and pf.get("R_fit"):
                    models.append(
                        {
                            "label": label,
                            "Q": pf["Q_fit"],
                            "R": pf["R_fit"],
                            "chi2": pf_chi2,
                            "iteration": it,
                        }
                    )
        else:
            label = f"Iteration {it}"
            if chi2 is not None:
                label += f" (\u03c7\u00b2={chi2:.2f})"
            if fr.get("Q_fit") and fr.get("R_fit"):
                models.append(
                    {
                        "label": label,
                        "Q": fr["Q_fit"],
                        "R": fr["R_fit"],
                        "chi2": chi2,
                        "iteration": it,
                    }
                )

        # In multi-state runs, fr.sld_z/sld_rho is state 0's profile only
        # (top-level bumps profile.dat). The dedicated /api/sld-profiles
        # endpoint emits proper per-state profiles, so skip this single
        # mislabeled curve here.
        if not has_multi and fr.get("sld_z") and fr.get("sld_rho"):
            sld_label = f"Iteration {it}"
            if chi2 is not None:
                sld_label += f" (\u03c7\u00b2={chi2:.2f})"
            profiles.append(
                {"label": sld_label, "z": fr["sld_z"], "sld": fr["sld_rho"]}
            )

    # Latest fit parameters
    latest = fit_results[-1]
    params_list = []
    params_dict = latest.get("parameters", {})
    unc_dict = latest.get("uncertainties") or {}
    for name, value in params_dict.items():
        params_list.append(
            {
                "name": name,
                "value": value,
                "uncertainty": unc_dict.get(name),
            }
        )

    result = {
        "Q": Q,
        "R": R,
        "dR": dR,
        "models": models,
        "profiles": profiles,
        "chi_squared": latest.get("chi_squared"),
        "method": latest.get("method"),
        "converged": latest.get("converged"),
        "parameters": params_list,
        "issues": latest.get("issues", []),
        "suggestions": latest.get("suggestions", []),
    }

    if has_multi:
        result["data_files"] = [
            {
                "label": ds.get("label", ""),
                "Q": ds.get("Q", []),
                "R": ds.get("R", []),
                "dR": ds.get("dR", []),
            }
            for ds in data_files
        ]

    return jsonify(result)


@bp.route("/api/user-feedback", methods=["POST"])
def api_user_feedback():
    """Submit user feedback during an interactive pause."""
    lock: threading.Lock = current_app.config["RUN_LOCK"]
    run_state: dict = current_app.config["RUN_STATE"]

    body = request.get_json(silent=True) or {}
    action = (body.get("action") or "continue").strip()
    feedback_text = (body.get("feedback") or "").strip() or None
    dream_steps = body.get("dream_steps")  # int or None
    restart_checkpoint = body.get("restart_checkpoint")  # str or None

    with lock:
        if run_state.get("status") != "waiting_for_user":
            return jsonify({"error": "Analysis is not waiting for feedback"}), 409
        pause_event: threading.Event | None = run_state.get("_pause_event")
        if pause_event is None:
            return jsonify({"error": "No pause event found"}), 500

        if action == "stop":
            run_state["_user_feedback"] = None
            run_state["_stop_requested"] = True
            run_state["status"] = "running"
        else:
            # Build structured feedback payload
            payload: dict | str | None = feedback_text
            has_advanced = (dream_steps is not None) or restart_checkpoint
            if has_advanced:
                payload = {
                    "feedback": feedback_text,
                    "dream_steps": int(dream_steps) if dream_steps else None,
                    "restart_checkpoint": restart_checkpoint or None,
                }
            run_state["_user_feedback"] = payload

        pause_event.set()

    return jsonify({"status": "ok"})


@bp.route("/api/restart-analysis", methods=["POST"])
def api_restart_analysis():
    """
    Restart an already-completed analysis with new user insight.

    Expects JSON body::

        {
            "insight": "Try adding an oxide interlayer between Fe and Si",
            "restart_from": "modeling"   // or "analysis"
        }

    The previous run's final state is loaded from disk, augmented with
    the user's insight, and the workflow is relaunched from the chosen node.
    """
    lock: threading.Lock = current_app.config["RUN_LOCK"]
    run_state: dict = current_app.config["RUN_STATE"]

    with lock:
        if run_state["status"] == "running":
            return jsonify({"error": "An analysis is already running"}), 409

    body = request.get_json(silent=True) or {}
    insight = (body.get("insight") or "").strip()
    restart_from = (body.get("restart_from") or "modeling").strip()
    dream_steps = body.get("dream_steps")  # int or None
    checkpoint_step = body.get("checkpoint_step")  # int or None (1-based step)
    checkpoint_iteration = body.get("checkpoint_iteration")  # int or None (legacy)
    fit_iteration = body.get("fit_iteration")  # int or None (0-based fit_results index)
    parameter_overrides = body.get("parameter_overrides")  # {name: value} or None
    bounds_overrides = body.get("bounds_overrides")  # {name: [lo, hi]} or None

    if restart_from not in ("modeling", "analysis", "fitting"):
        return jsonify(
            {"errors": ["restart_from must be 'modeling', 'analysis', or 'fitting'"]}
        ), 400
    if not insight and restart_from != "fitting":
        return jsonify({"errors": ["insight is required"]}), 400
    if restart_from == "fitting" and not insight:
        insight = "Refit with user-adjusted parameters"

    # ---- Load the completed state from disk -----------------------
    output_dir = current_app.config.get("OUTPUT_DIR")
    if not output_dir or not Path(output_dir).exists():
        return jsonify({"error": "No previous analysis output found"}), 404

    rd = _run_data()
    if not rd:
        return jsonify({"error": "No previous analysis output found"}), 404

    final_state = rd.get_final_state()
    if not final_state:
        return jsonify({"error": "Could not load final state from previous run"}), 404

    # ---- Read max iteration from all existing checkpoints ---------
    import json as _json

    run_info_path = Path(output_dir) / "run_info.json"
    max_iteration = final_state.get("iteration", 0)
    if run_info_path.exists():
        try:
            run_info_data = _json.loads(run_info_path.read_text())
            for cp_entry in run_info_data.get("checkpoints", []):
                cp_iter = cp_entry.get("iteration", 0)
                if cp_iter > max_iteration:
                    max_iteration = cp_iter
        except Exception:
            pass

    # ---- Optionally load from a specific checkpoint ---------------
    if checkpoint_step is not None:
        # Load by step number (1-based index into run_info checkpoints)
        cp_dir = Path(output_dir) / "checkpoints"
        cp_state = None
        if run_info_path.exists() and cp_dir.exists():
            try:
                ri = _json.loads(run_info_path.read_text())
                cp_list = ri.get("checkpoints", [])
                step_idx = int(checkpoint_step) - 1
                if 0 <= step_idx < len(cp_list):
                    cp_file = cp_dir / cp_list[step_idx]["file"]
                    if cp_file.exists():
                        cp_data = _json.loads(cp_file.read_text())
                        cp_state = cp_data.get("state", cp_data)
            except Exception:
                pass
        if cp_state:
            final_state = cp_state
        else:
            return jsonify(
                {"error": f"Checkpoint step {checkpoint_step} not found"}
            ), 404
    elif checkpoint_iteration is not None:
        # Legacy: load by iteration number (finds the last match)
        cp_dir = Path(output_dir) / "checkpoints"
        if cp_dir.exists():
            cp_state = None
            for cp_file in sorted(cp_dir.glob("*.json")):
                try:
                    cp_data = _json.loads(cp_file.read_text())
                    cp_st = cp_data.get("state", cp_data)
                    if cp_st.get("iteration") == int(checkpoint_iteration):
                        cp_state = cp_st
                except Exception:
                    continue
            if cp_state:
                final_state = cp_state
            else:
                return jsonify(
                    {
                        "error": f"Checkpoint for iteration {checkpoint_iteration} not found"
                    }
                ), 404

    # ---- Optionally restore model from a specific fit iteration ----
    if (
        fit_iteration is not None
        and checkpoint_step is None
        and checkpoint_iteration is None
    ):
        fit_results = final_state.get("fit_results") or []
        fit_idx = int(fit_iteration)
        if 0 <= fit_idx < len(fit_results):
            iter_num = fit_results[fit_idx].get("iteration", fit_idx)
            model_history = final_state.get("model_history") or []
            for entry in model_history:
                if entry.get("iteration") == iter_num:
                    defn = entry.get("definition")
                    if defn and isinstance(defn, dict):
                        final_state["current_model"] = defn
                        break
                    script = entry.get("script")
                    if script:
                        final_state["current_model"] = script
                        break

    # Override iteration to continue from max seen across all checkpoints
    final_state["iteration"] = max_iteration

    interactive = bool(body.get("interactive", False))

    # ---- Prepare state for restart --------------------------------
    from ..workflow.runner import prepare_state_for_restart

    restarted_state = prepare_state_for_restart(
        state=final_state,
        user_insight=insight,
        restart_from=restart_from,
        extra_iterations=1,
    )
    restarted_state["output_dir"] = output_dir
    if interactive:
        restarted_state["interactive"] = True

    # Apply DREAM steps override if specified
    if dream_steps is not None:
        restarted_state["fit_steps"] = int(dream_steps)
        restarted_state["fit_burn"] = int(dream_steps)

    # Apply user parameter / bounds overrides to the latest fit results
    # and the current model script so the restart uses updated values.
    if parameter_overrides and isinstance(parameter_overrides, dict):
        fit_results = restarted_state.get("fit_results") or []
        if fit_results:
            latest = fit_results[-1]
            params = latest.get("parameters", {})
            for name, val in parameter_overrides.items():
                if name in params:
                    try:
                        params[name] = float(val)
                    except (ValueError, TypeError):
                        pass
            latest["parameters"] = params

    if bounds_overrides and isinstance(bounds_overrides, dict):
        fit_results = restarted_state.get("fit_results") or []
        if fit_results:
            latest = fit_results[-1]
            bounds = latest.get("bounds") or {}
            for name, pair in bounds_overrides.items():
                if isinstance(pair, list) and len(pair) == 2:
                    try:
                        bounds[name] = [float(pair[0]), float(pair[1])]
                    except (ValueError, TypeError):
                        pass
            latest["bounds"] = bounds

    # Update current_model with overrides
    current_model = restarted_state.get("current_model")
    if (parameter_overrides or bounds_overrides) and current_model:
        if isinstance(current_model, dict):
            # JSON ModelDefinition — apply overrides directly to the dict
            restarted_state["current_model"] = _apply_overrides_to_model_definition(
                current_model,
                parameter_overrides or {},
                bounds_overrides or {},
            )
        else:
            restarted_state["current_model"] = _apply_overrides_to_model_script(
                current_model,
                parameter_overrides or {},
                bounds_overrides or {},
            )

    # ---- Update run_info.json with restart metadata ---------------
    from datetime import datetime

    if run_info_path.exists():
        run_info = _json.loads(run_info_path.read_text())
        restarts = run_info.setdefault("restarts", [])
        restarts.append(
            {
                "restarted_at": datetime.now().isoformat(),
                "restart_from": restart_from,
                "insight": insight,
                "iteration_at_restart": max_iteration,
            }
        )
        run_info_path.write_text(_json.dumps(run_info, indent=2, default=str))

    # ---- Reset run state and launch background thread -------------
    with lock:
        run_state.update(
            {
                "status": "running",
                "output_dir": output_dir,
                "current_node": None,
                "iteration": restarted_state.get("iteration", 0),
                "checkpoints": run_state.get("checkpoints", []),
                "error": None,
                "interactive": interactive,
                "messages": [],
                "restarted": True,
                "restart_insight": insight,
                "_pause_event": None,
                "_user_feedback": None,
                "_stop_requested": False,
            }
        )
        # Preserve experimental data from previous run
        if "Q" not in run_state and final_state.get("Q"):
            run_state["Q"] = final_state["Q"]
            run_state["R"] = final_state["R"]
            run_state["dR"] = final_state.get("dR", [])
        if final_state.get("fit_results"):
            run_state["fit_results"] = list(final_state["fit_results"])

    app = current_app._get_current_object()

    def _run_restart_in_background():
        from ..workflow.runner import run_workflow_with_checkpoints

        def _checkpoint_cb(state, node_name):
            with lock:
                run_state["current_node"] = node_name
                run_state["iteration"] = state.get("iteration", 0)
                all_llm = state.get("llm_calls", [])
                prev_count = sum(
                    len(cp.get("llm_calls", [])) for cp in run_state["checkpoints"]
                )
                step_llm = all_llm[prev_count:]
                run_state["checkpoints"].append(
                    {
                        "node": node_name,
                        "iteration": state.get("iteration", 0),
                        "chi2": state.get("current_chi2"),
                        "llm_calls": step_llm,
                    }
                )
                if state.get("fit_results"):
                    run_state["fit_results"] = list(state["fit_results"])

        pause_callback = None
        if interactive:
            pause_event = threading.Event()
            with lock:
                run_state["_pause_event"] = pause_event

            def _pause_cb(state, node_name):
                msgs = [
                    {"role": m["role"], "content": m["content"]}
                    for m in state.get("messages", [])
                ]
                with lock:
                    run_state["status"] = "waiting_for_user"
                    run_state["messages"] = msgs
                    run_state["_user_feedback"] = None
                    pause_event.clear()

                pause_event.wait()

                with lock:
                    run_state["status"] = "running"
                    feedback = run_state.get("_user_feedback")
                    stop_requested = run_state.get("_stop_requested", False)
                if stop_requested:
                    return "__STOP__"
                return feedback

            pause_callback = _pause_cb

        try:
            run_workflow_with_checkpoints(
                initial_state=restarted_state,
                output_dir=output_dir,
                checkpoint_callback=_checkpoint_cb,
                start_node=restart_from,
                pause_callback=pause_callback,
            )
            with lock:
                run_state["status"] = "complete"
            app.config["OUTPUT_DIR"] = output_dir
        except Exception as exc:
            with lock:
                run_state["status"] = "error"
                run_state["error"] = str(exc)

    t = threading.Thread(target=_run_restart_in_background, daemon=True)
    t.start()

    return jsonify(
        {"status": "restarted", "output_dir": output_dir, "restart_from": restart_from}
    )


@bp.route("/api/analysis-status")
def api_analysis_status():
    """Return current analysis run state."""
    lock: threading.Lock = current_app.config["RUN_LOCK"]
    run_state: dict = current_app.config["RUN_STATE"]
    with lock:
        # Exclude internal objects from the JSON response
        return jsonify({k: v for k, v in run_state.items() if not k.startswith("_")})


# ------------------------------------------------------------------
# JSON API – data export
# ------------------------------------------------------------------


@bp.route("/api/export-info")
def api_export_info():
    """Return export availability and format metadata.

    Used by the Results page to conditionally show the Export button.
    """
    from ..exporters import get_exporter

    exporter = get_exporter()
    if exporter is None:
        return jsonify({"available": False})
    return jsonify(
        {
            "available": True,
            "format": exporter.format_id,
            "name": exporter.name,
        }
    )


@bp.route("/api/export", methods=["POST"])
def api_export():
    """Run the configured data exporter on the current results.

    Returns JSON with ``success``, ``output_path``, ``errors``, and ``warnings``.
    """
    from ..exporters import get_exporter

    exporter = get_exporter()
    if exporter is None:
        return jsonify({"error": "No exporter configured"}), 400

    body = request.get_json(silent=True) or {}
    user_context = (body.get("user_context") or "").strip() or None

    output_dir = current_app.config.get("OUTPUT_DIR")
    if not output_dir or not Path(output_dir).exists():
        return jsonify({"error": "No analysis output found"}), 404

    rd = _run_data()
    if not rd:
        return jsonify({"error": "No analysis output found"}), 404

    state = rd.get_final_state()
    run_info = rd.get_run_info()
    if not state:
        return jsonify({"error": "Could not load final state"}), 404

    lock: threading.Lock = current_app.config["RUN_LOCK"]
    run_state: dict = current_app.config["RUN_STATE"]
    with lock:
        if run_state.get("status") == "running":
            return jsonify({"error": "Cannot export while analysis is running"}), 409

    try:
        result = exporter.export(
            output_dir=Path(output_dir),
            state=state,
            run_info=run_info,
            user_context=user_context,
        )
        return jsonify(
            {
                "success": result.success,
                "output_path": str(result.output_path) if result.output_path else None,
                "errors": result.errors,
                "warnings": result.warnings,
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Export failed: {exc}"}), 500
