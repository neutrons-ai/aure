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


@bp.route("/api/export-script")
def api_export_script():
    """Export the model as a standalone refl1d Python script.

    Query parameters:
        iteration (int, optional) — fit iteration to use for parameter values.
            Defaults to the best-chi² iteration.
        include_ranges (bool, optional) — include ``.range()`` calls.
            Defaults to ``false``.

    Returns ``{"script": "<python source code>"}`` or an error.
    """
    rd = _run_data()
    if not rd:
        return jsonify({"error": "No analysis output found"}), 404

    state = rd.get_final_state()
    model = state.get("best_model") or state.get("current_model")
    if not model:
        return jsonify({"error": "No model available"}), 404

    if not isinstance(model, dict):
        # Legacy: try to read model_final.py from disk
        final_py = rd.output_dir / "models" / "model_final.py"
        if final_py.exists():
            return jsonify({"script": final_py.read_text()})
        return jsonify({"error": "Legacy model — no JSON definition available"}), 404

    from aure.nodes.model_builder import export_model_script

    iteration = request.args.get("iteration", default=None, type=int)
    include_ranges = request.args.get("include_ranges", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    fit_results = state.get("fit_results", [])
    fitted_params: dict | None = None
    uncertainties: dict | None = None

    if fit_results:
        if iteration is not None:
            selected = next(
                (fr for fr in fit_results if fr.get("iteration") == iteration),
                None,
            )
        else:
            selected = min(
                fit_results, key=lambda f: f.get("chi_squared", float("inf"))
            )
        if selected:
            fitted_params = selected.get("parameters")
            uncertainties = selected.get("uncertainties")

    script = export_model_script(
        model,
        fitted_params=fitted_params,
        uncertainties=uncertainties,
        include_ranges=include_ranges,
    )
    return jsonify({"script": script})


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
        ext   – optional extension filter, e.g. ".txt"
    """
    raw = request.args.get("path", str(Path.home()))
    ext = request.args.get("ext", "")
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
            elif not ext or child.suffix.lower() == ext.lower():
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
    errors = []
    if not data_file or not Path(data_file).is_file():
        errors.append("data_file: file does not exist")
    if not sample_description:
        errors.append("sample_description is required")
    if not output_root:
        errors.append("output_dir is required")
    if errors:
        return jsonify({"errors": errors}), 400

    # Determine run sub-directory
    run_name = _extract_run_name(data_file)
    output_dir = str(Path(output_root).expanduser().resolve() / run_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    interactive = bool(body.get("interactive", False))
    max_iterations = int(body.get("max_iterations", 5))

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

    # Build model curves
    models = []
    profiles = []
    for fr in fit_results:
        it = fr.get("iteration", 0)
        chi2 = fr.get("chi_squared")
        label = f"Iteration {it}"
        if chi2 is not None:
            label += f" (\u03c7\u00b2={chi2:.2f})"
        if fr.get("Q_fit") and fr.get("R_fit"):
            models.append(
                {"label": label, "Q": fr["Q_fit"], "R": fr["R_fit"], "chi2": chi2}
            )
        if fr.get("sld_z") and fr.get("sld_rho"):
            profiles.append({"label": label, "z": fr["sld_z"], "sld": fr["sld_rho"]})

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

    return jsonify(
        {
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
    )


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
