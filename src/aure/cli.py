"""
Command-line interface for the Reflectivity Analysis Workflow.

Usage:
    python -m aure.cli analyze data.dat "100 nm polystyrene on silicon"
    python -m aure.cli lookup-sld silicon gold D2O
    python -m aure.cli mcp-server
"""

import contextlib
import json
import logging
import math
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Optional

import click

from .env_config import load_env

# Load environment variables from .env then ~/.aure (project .env wins)
load_env()

from .llm import (  # noqa: E402
    get_llm_info,
    get_llm,
    invoke_with_timeout,
    LLMTimeoutError,
    get_llm_timeout,
)


_ALCF_AUTH_SCRIPT_URL = (
    "https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/"
    "refs/heads/main/inference_auth_token.py"
)


def _alcf_authenticate() -> bool:
    """Download the ALCF auth helper and run ``authenticate``.

    Returns True if the script appeared to succeed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "inference_auth_token.py"
        click.echo()
        click.echo("    Downloading inference_auth_token.py …", nl=False)
        try:
            urllib.request.urlretrieve(_ALCF_AUTH_SCRIPT_URL, script)
            click.echo(click.style(" done", fg="green"))
        except Exception as exc:
            click.echo(click.style(f" failed: {exc}", fg="red"))
            return False

        click.echo("    Launching Globus authentication (a browser window may open)…")
        click.echo()
        result = subprocess.run(
            [sys.executable, str(script), "authenticate"],
        )
        if result.returncode != 0:
            click.echo()
            click.echo(
                click.style("    Authentication script exited with an error.", fg="red")
            )
            return False

        click.echo()
        click.echo(click.style("    ✓ Authentication complete.", fg="green"))
        click.echo("    You can now obtain a token by running:")
        click.echo(f"      python {script.name} get_access_token")
        click.echo("    or set ALCF_ACCESS_TOKEN in your environment.")
        return True


def _show_alcf_auth_hint(*, offer_fix: bool = False) -> None:
    """Print ALCF authentication instructions, optionally offering to run them."""
    if offer_fix:
        if click.confirm(
            "    Download and run the ALCF auth script now?", default=True
        ):
            _alcf_authenticate()
            return

    click.echo("      # Download the authentication helper script")
    click.echo(f"      wget {_ALCF_AUTH_SCRIPT_URL}")
    click.echo()
    click.echo("      # Authenticate with your Globus account")
    click.echo("      python inference_auth_token.py authenticate")


def _check_llm_status(
    quiet: bool = False, test_connection: bool = True
) -> tuple[bool, str]:
    """
    Check and report LLM configuration status.

    Args:
        quiet: If True, suppress output (for JSON mode)
        test_connection: If True, test the LLM with a simple query

    Returns:
        Tuple of (success, message)
    """
    info = get_llm_info()

    if not quiet:
        click.echo(click.style("  LLM Configuration", fg="cyan", bold=True))
        click.echo(f"    Provider: {info['provider']}")
        click.echo(f"    Model: {info['model']}")
        if info.get("base_url"):
            click.echo(f"    Base URL: {info['base_url']}")

    if not info["available"]:
        msg = "LLM not configured (missing API key or base URL)"
        if not quiet:
            click.echo(click.style(f"    Status: ✗ {msg}", fg="yellow"))
        return False, msg

    if test_connection:
        if not quiet:
            click.echo("    Testing connection...", nl=False)
        # Suppress noisy provider-level warnings during the test call
        # (e.g. ALCF globus_sdk fallback messages).
        _llm_logger = logging.getLogger("aure.llm")
        _old_level = _llm_logger.level
        _llm_logger.setLevel(logging.CRITICAL)
        try:
            llm = get_llm()
            # Use timeout to prevent infinite retries on quota errors
            response = invoke_with_timeout(
                llm,
                "Reply with only the word 'OK'",
                timeout_seconds=min(30, get_llm_timeout()),
            )
            # Validate we got a real response back
            content = getattr(response, "content", None) or ""
            if not content.strip():
                msg = "LLM returned an empty response"
                if not quiet:
                    click.echo(click.style(f" ✗ {msg}", fg="red"))
                return False, msg
            if not quiet:
                click.echo(click.style(" ✓ Connected", fg="green"))
            return True, "LLM connected successfully"
        except LLMTimeoutError:
            short_msg = "API quota/rate limit exceeded (call timed out)"
            if not quiet:
                click.echo(click.style(f" ✗ {short_msg}", fg="red"))
                click.echo(
                    click.style(
                        "    Try: Wait and retry, or switch to a different model/provider",
                        fg="yellow",
                    )
                )
            return False, short_msg
        except Exception as e:
            error_msg = str(e)
            error_lower = error_msg.lower()

            # Detect common error types
            if (
                "quota" in error_lower
                or "rate" in error_lower
                or "limit" in error_lower
                or "429" in error_msg
            ):
                short_msg = "API quota/rate limit exceeded"
                if not quiet:
                    click.echo(click.style(f" ✗ {short_msg}", fg="red"))
                    click.echo(
                        click.style(
                            "    Try: Wait and retry, or switch to a different model/provider",
                            fg="yellow",
                        )
                    )
                return False, short_msg
            elif (
                "401" in error_msg
                or "unauthorized" in error_lower
                or "api key not valid" in error_lower
                or "api_key_invalid" in error_lower
                or ("invalid" in error_lower and "key" in error_lower)
            ):
                short_msg = "Invalid API key"
                if not quiet:
                    click.echo(click.style(f" ✗ {short_msg}", fg="red"))
                    click.echo(
                        click.style("    Check your LLM_API_KEY in .env", fg="yellow")
                    )
                return False, short_msg
            elif "not found" in error_lower or "404" in error_msg:
                short_msg = f"Model '{info['model']}' not found"
                if not quiet:
                    click.echo(click.style(f" ✗ {short_msg}", fg="red"))
                    click.echo(click.style("    Check LLM_MODEL in .env", fg="yellow"))
                return False, short_msg
            elif "connection" in error_lower or "connect" in error_lower:
                short_msg = "Connection failed"
                if not quiet:
                    click.echo(click.style(f" ✗ {short_msg}", fg="red"))
                    if info.get("base_url"):
                        click.echo(
                            click.style(
                                f"    Check if server is running at {info['base_url']}",
                                fg="yellow",
                            )
                        )
                return False, short_msg
            else:
                # Generic error - truncate long messages
                if len(error_msg) > 100:
                    error_msg = error_msg[:100] + "..."
                if not quiet:
                    click.echo(click.style(" ✗ Failed", fg="red"))
                    click.echo(click.style(f"    Error: {error_msg}", fg="red"))
                return False, f"Connection failed: {error_msg}"
        finally:
            _llm_logger.setLevel(_old_level)

    if not quiet:
        click.echo(click.style("    Status: ✓ Configured", fg="green"))
    return True, "LLM configured"


# ============================================================================
# Main CLI Group
# ============================================================================


@click.group()
@click.version_option(version="0.1.0", prog_name="aure")
def cli():
    """
    Reflectivity Analysis Workflow CLI.

    An intelligent assistant for analyzing neutron reflectivity data.
    Orchestrates data analysis, model building, and fitting.
    """
    pass


# ============================================================================
# Diagnostics
# ============================================================================


@cli.command("check-llm")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--no-test", is_flag=True, help="Skip the live connection test")
@click.option(
    "--fix", is_flag=True, help="Attempt to fix issues (e.g. download ALCF auth script)"
)
def check_llm(output_json: bool, no_test: bool, fix: bool):
    """
    Check LLM configuration and connectivity.

    Shows which provider and model are configured, whether credentials
    are present, and (unless --no-test) sends a tiny test prompt to
    verify the connection works end-to-end.

    Use --fix to automatically download and run the ALCF authentication
    helper when the provider is 'alcf' and credentials are missing or
    expired.

    \b
    Environment variables used:
        LLM_PROVIDER     openai | gemini | alcf | local
        LLM_MODEL        model name (default depends on provider)
        LLM_API_KEY      API key (or OPENAI_API_KEY / GEMINI_API_KEY)
        LLM_BASE_URL     base URL (required for 'local' provider)
        LLM_TIMEOUT      call timeout in seconds (default 120)
        LLM_TEMPERATURE  sampling temperature (default 0.0)

    \b
    Examples:
        aure check-llm
        aure check-llm --json
        aure check-llm --no-test
        aure check-llm --fix
    """
    from .llm.config import get_llm_config
    import os

    config = get_llm_config()
    info = get_llm_info()
    has_key = bool(config.get("api_key"))

    # For ALCF the credential is an access token, not an API key
    if config["provider"] == "alcf":
        has_credential = bool(os.environ.get("ALCF_ACCESS_TOKEN"))
    else:
        has_credential = has_key

    if not output_json:
        click.echo()
        click.echo(click.style("  LLM Configuration Check", fg="blue", bold=True))
        click.echo(click.style("  " + "─" * 40, fg="blue"))
        click.echo()
        click.echo(f"    Provider:    {config['provider'] or '(not set)'}")
        click.echo(f"    Model:       {config['model']}")
        if config["provider"] == "alcf":
            has_token = bool(os.environ.get("ALCF_ACCESS_TOKEN"))
            click.echo(
                f"    Token:       {'••••' + os.environ['ALCF_ACCESS_TOKEN'][-4:] if has_token else click.style('NOT SET', fg='red')}"
            )
            click.echo(f"    ALCF cluster: {config.get('alcf_cluster', 'sophia')}")
            click.echo(f"    Base URL:    {info.get('base_url', '(unknown)')}")
        else:
            click.echo(
                f"    API key:     {'••••' + config['api_key'][-4:] if has_key else click.style('NOT SET', fg='red')}"
            )
            if config.get("base_url"):
                click.echo(f"    Base URL:    {config['base_url']}")
        click.echo(f"    Timeout:     {get_llm_timeout()}s")
        click.echo(f"    Temperature: {config['temperature']}")
        click.echo()

    if not info["available"]:
        if output_json:
            ok, msg = False, "LLM not configured (missing API key or base URL)"
        else:
            ok, msg = False, "LLM not available"
            click.echo(click.style("  ✗ LLM not available", fg="red", bold=True))
            click.echo()
            if config["provider"] == "alcf":
                click.echo("    Authenticate with ALCF to obtain an access token:")
                click.echo()
                _show_alcf_auth_hint(offer_fix=fix)
                click.echo()
                click.echo("    Then set the token:")
                click.echo("      export ALCF_ACCESS_TOKEN=<your-token>")
            elif config["provider"] in ("openai", "gemini") and not has_key:
                click.echo("    Set an API key:")
                click.echo("      export LLM_API_KEY=<your-key>")
                click.echo("    or add to .env:")
                click.echo(f"      LLM_PROVIDER={config['provider']}")
                click.echo("      LLM_API_KEY=<your-key>")
            elif config["provider"] == "local" and not config.get("base_url"):
                click.echo("    Set a base URL for local provider:")
                click.echo("      export LLM_BASE_URL=http://localhost:11434/v1")
            click.echo()
    elif no_test:
        ok, msg = True, "Credentials present (skipped live test)"
        if not output_json:
            click.echo(
                click.style("  ✓ Credentials present (skipped live test)", fg="green")
            )
            click.echo()
    else:
        if not output_json:
            click.echo("    Testing connection...", nl=False)
        ok, msg = _check_llm_status(quiet=True, test_connection=True)
        if not output_json:
            if ok:
                click.echo(click.style(" ✓ Connected", fg="green"))
            else:
                click.echo(click.style(f" ✗ {msg}", fg="red"))
                if config["provider"] == "alcf":
                    click.echo()
                    click.echo(
                        click.style(
                            "    ALCF tokens expire periodically. Re-authenticate:",
                            fg="yellow",
                        )
                    )
                    click.echo()
                    _show_alcf_auth_hint(offer_fix=fix)
            click.echo()

    if output_json:
        result = {
            **info,
            "has_api_key": has_credential,
            "timeout": get_llm_timeout(),
            "temperature": config["temperature"],
            "ok": ok,
            "message": msg,
        }
        click.echo(json.dumps(result, indent=2))

    sys.exit(0 if ok else 1)


# ============================================================================
# Analysis Commands
# ============================================================================


@cli.command()
@click.argument("data_file", type=click.Path(exists=True), required=False)
@click.argument("sample_description", required=False)
@click.option(
    "--hypothesis",
    "-h",
    help="Optional hypothesis to test",
)
@click.option(
    "--extra-data",
    "-d",
    multiple=True,
    type=click.Path(exists=True),
    help="Additional data files for multi-file co-refinement (single state)",
)
@click.option(
    "--max-refinements",
    "-m",
    default=None,
    type=int,
    help="Maximum number of refinement iterations (default: 5)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Output directory for checkpoints and results",
)
@click.option(
    "--model-name",
    "-n",
    default=None,
    help="Basename for the exported refl1d/bumps files (default: the "
    "output-dir name, then the data-file stem)",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging to trace workflow progress",
)
@click.option(
    "--config",
    "-c",
    "config_file",
    type=click.Path(exists=True),
    help="Setup YAML file (states, evaluation criteria, model constraints, …)",
)
@click.option(
    "--data-dir",
    "data_dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Directory to resolve relative data_files against (highest priority). "
    "Search order: this dir → the config file's dir → the current directory.",
)
def analyze(
    data_file: Optional[str],
    sample_description: Optional[str],
    hypothesis: Optional[str],
    extra_data: tuple,
    max_refinements: Optional[int],
    output_dir: Optional[str],
    model_name: Optional[str],
    output_json: bool,
    verbose: bool,
    config_file: Optional[str],
    data_dir: Optional[str],
):
    """
    Analyze a reflectivity data file with fitting and refinement.

    DATA_FILE and SAMPLE_DESCRIPTION are the ad-hoc single-file shortcut.
    For multi-state co-refinement or finer control, pass a setup YAML via
    ``-c setup.yaml`` and omit the positional arguments. Positional
    arguments override anything declared in the setup.

    When --output-dir is specified, checkpoints are saved after each
    workflow node (intake, analysis, modeling, fitting, evaluation).

    \b
    Examples:
        aure analyze data.dat "100 nm polystyrene on silicon"
        aure analyze low-Q.dat "multilayer" -d mid-Q.dat -d high-Q.dat
        aure analyze -c setup.yaml -o ./results
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            stream=sys.stderr,
        )
        for module in [
            "agent.nodes.fitting",
            "agent.nodes.evaluation",
            "agent.nodes.refinement",
        ]:
            logging.getLogger(module).setLevel(logging.INFO)

    from .setup import (
        SetupConfig,
        load_setup,
        primary_data_file,
        setup_to_user_config,
    )
    from .workflow import run_analysis

    # ── Load setup ────────────────────────────────────────────────
    setup: SetupConfig = {}  # type: ignore[assignment]
    if config_file:
        try:
            setup = load_setup(config_file, data_dir=data_dir)
        except Exception as e:
            click.echo(click.style(f"  Setup error: {e}", fg="red"))
            sys.exit(2)
    elif data_dir:
        click.echo(
            click.style(
                "  --data-dir has no effect without --config; ignoring.",
                fg="yellow",
            )
        )

    # ── Merge positional overrides into the setup ─────────────────
    # Rule: positional args win over setup values, but warn if both supplied.
    if sample_description:
        if (
            setup.get("sample_description")
            and setup["sample_description"] != sample_description
        ):
            click.echo(
                click.style(
                    "  Note: positional SAMPLE_DESCRIPTION overrides the setup file.",
                    fg="yellow",
                )
            )
        setup["sample_description"] = sample_description
    if hypothesis:
        setup["hypothesis"] = hypothesis
    if max_refinements is not None:
        setup["max_refinements"] = max_refinements
    if model_name:
        # Highest-priority candidate in _resolve_model_name, so the refl1d
        # export files are <model_name>-* instead of being named after the
        # output directory (or, unnamed, None-*).
        setup["model_name"] = model_name

    # ── Resolve data files ────────────────────────────────────────
    # Three mutually exclusive paths:
    #   1. Setup carries states  → use them as-is. --extra-data forbidden.
    #   2. Positional DATA_FILE  → synthesize a single state from it (+ --extra-data).
    #   3. Neither               → error.
    states = setup.get("states") or []
    if states:
        if extra_data:
            click.echo(
                click.style(
                    "  Cannot combine --extra-data with `states:` in the setup file. "
                    "Move all data files into the setup's states block.",
                    fg="red",
                )
            )
            sys.exit(2)
        if data_file:
            from pathlib import Path as _Path
            from .state import flatten_data_files as _flatten

            first = _flatten(states)[0]["file"]
            if _Path(data_file).resolve() != _Path(first).resolve():
                click.echo(
                    click.style(
                        "  Note: positional DATA_FILE ignored; using files from setup.",
                        fg="yellow",
                    )
                )
    else:
        if not data_file:
            click.echo(
                click.style(
                    "  Missing data file. Pass it as the positional argument "
                    "or define a `states:` block in --config.",
                    fg="red",
                )
            )
            sys.exit(2)
        from pathlib import Path as _Path

        all_files = [data_file] + list(extra_data)
        synthetic_state = {
            "name": "state0",
            "data_files": [
                {"file": str(_Path(f).resolve()), "label": _Path(f).stem}
                for f in all_files
            ],
        }
        states = [synthetic_state]
        setup["states"] = states

    if not setup.get("sample_description"):
        click.echo(
            click.style(
                "  Missing sample description. Provide it as the positional "
                "argument or set `sample_description:` in --config.",
                fg="red",
            )
        )
        sys.exit(2)

    # ── Derive runner kwargs ──────────────────────────────────────
    sample_description = setup["sample_description"]
    hypothesis = setup.get("hypothesis")
    max_refinements = setup.get("max_refinements", 5)
    user_config = setup_to_user_config(setup)
    data_file = primary_data_file(setup)
    data_files = None  # multi-state path: runner reads from `states`

    # The setup's run controls (chi2_max, fit budgets, LLM settings) reach the
    # workflow as environment variables, so they must be in force before the
    # banner reads them back and before the runner starts.
    with _applied_env_overrides(setup):
        if not output_json:
            click.echo(click.style("═" * 60, fg="blue"))
            click.echo(
                click.style("  Reflectivity Analysis Workflow", fg="blue", bold=True)
            )
            click.echo(click.style("═" * 60, fg="blue"))
            click.echo()

            # Check LLM status first
            llm_ok, llm_msg = _check_llm_status(quiet=False, test_connection=True)
            click.echo()

            click.echo(f"  Data file: {data_file}")
            n_files = sum(len(s.get("data_files", [])) for s in states)
            if len(states) > 1:
                for st in states:
                    names = ", ".join(ds["label"] for ds in st.get("data_files", []))
                    click.echo(f"  State {st['name']}: {names}")
                click.echo(
                    f"  Multi-state co-refinement: {len(states)} states, "
                    f"{n_files} files"
                )
            elif n_files > 1:
                for ds in states[0].get("data_files", []):
                    click.echo(f"  + {ds['file']}")
                click.echo(f"  Co-refinement: {n_files} files (shared structure)")
            click.echo(f"  Sample: {sample_description}")
            if hypothesis:
                click.echo(f"  Hypothesis: {hypothesis}")

            from .nodes.evaluation import _get_chi2_max, _get_chi2_min

            # Echo both bounds, so the run log records the acceptance window
            # that was actually in force. The floor is not a second bar to
            # clear: below it the deterministic stop stands down and the
            # evaluator decides, because a χ² that far under 1 is evidence the
            # quoted dR is too large rather than evidence the structure is right.
            chi2_max = _get_chi2_max()
            chi2_min = _get_chi2_min(chi2_max=chi2_max)
            if chi2_min > 0:
                click.echo(
                    f"  Accept when {chi2_min:g} ≤ χ² ≤ {chi2_max:g}"
                    f"  (χ² < {chi2_min:g} is judged, not auto-accepted)"
                )
            else:
                click.echo(f"  Accept when χ² ≤ {chi2_max:g}  (χ² floor disabled)")
            click.echo()
        else:
            # Still check LLM in quiet mode for JSON output
            llm_ok, llm_msg = _check_llm_status(quiet=True, test_connection=True)

        # Captured while the setup's overrides are still in force: the --json
        # payload is emitted after they are restored, and must report the
        # provider/model the run actually used, not the ambient one.
        llm_info = get_llm_info()

        # Stop if LLM is not available
        if not llm_ok:
            if output_json:
                click.echo(
                    json.dumps(
                        {"error": f"LLM not available: {llm_msg}", "llm": llm_info}
                    )
                )
            else:
                click.echo(click.style(f"  Cannot proceed: {llm_msg}", fg="red"))
                click.echo("  Please configure a working LLM provider in .env")
            sys.exit(1)

        # Create checkpoint callback for progress reporting
        def checkpoint_callback(state, node_name):
            if not output_json:
                status = "✓" if not state.get("error") else "✗"
                click.echo(
                    click.style(
                        f"  [{status}] {node_name.title()}",
                        fg="green" if status == "✓" else "red",
                    )
                )

        # Run analysis
        # Open the per-call LLM ledger alongside the checkpoints. Measurement
        # only: it records what each call cost and never influences one. See
        # aure.llm.ledger; AURE_LLM_LOG overrides the destination.
        from .llm import ledger

        ledger.set_sink(
            str(Path(output_dir) / "llm_calls.jsonl") if output_dir else None
        )

        try:
            result = run_analysis(
                data_file=data_file,
                sample_description=sample_description,
                hypothesis=hypothesis,
                max_iterations=max_refinements,
                output_dir=output_dir,
                checkpoint_callback=checkpoint_callback if not output_json else None,
                user_config=user_config,
                data_files=data_files,
                states=states,
            )
        except Exception as e:
            if output_json:
                click.echo(json.dumps({"error": str(e)}))
            else:
                click.echo(click.style(f"Error: {e}", fg="red"))
            sys.exit(1)

    # Check for errors
    if result.get("error"):
        if output_json:
            click.echo(json.dumps({"error": result["error"]}))
        else:
            click.echo(click.style(f"Error: {result['error']}", fg="red"))
        sys.exit(1)

    # Output results
    if output_json:
        output_data = {
            "success": True,
            "llm": {
                "available": llm_ok,
                "info": llm_info,
            },
            "output_dir": output_dir,
            "n_points": len(result.get("Q", [])),
            "parsed_sample": result.get("parsed_sample"),
            "extracted_features": result.get("extracted_features"),
            "model_generated": result.get("current_model") is not None,
            # Always present, empty list included: a script deciding whether to
            # keep refining should not have to tell "no backlog" from "old aure".
            "pending_hypotheses": _pending_hypotheses_payload(
                result.get("structural_hypotheses")
            ),
        }
        if result.get("fit_results"):
            _, _, selected = _selected_fit(result)
            output_data["fit_result"] = {
                "chi_squared": (selected or {}).get("chi_squared"),
                "parameters": (selected or {}).get("parameters"),
            }
            output_data["selection"] = _selection_payload(result)
        click.echo(json.dumps(output_data, indent=2))
    else:
        _print_analysis_results(result, output_dir)


def _run_pinned_state(refl1d_path: Path) -> tuple:
    """``(state, source)`` for the run that produced a refl1d export directory.

    The export lives at ``<run>/refl1d_output/fit_iterN_method``, so the run's
    ``final_state.json`` is two levels up. It carries the ``chi2_max`` / ``chi2_min``
    the run was launched with, which is what its fits should be judged against.

    Best-effort by design: a directory inspected out of context, a truncated file, or
    one predating the pinned window all fall back to the ambient environment rather
    than failing the command.
    """
    candidate = refl1d_path.parent.parent / "final_state.json"
    try:
        if not candidate.is_file():
            return None, None
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    state = payload.get("state", payload)
    if not isinstance(state, dict) or state.get("chi2_max") is None:
        return None, None
    return state, str(candidate)


def _selected_fit(result: dict) -> tuple:
    """``(fit_results, final_selection, selected_fit)`` for a finished run.

    The answer is the iteration ``finalize`` chose, which is routinely *not* the
    last one performed — it rejects regressions, prefers the simpler model inside
    the χ² band, and sets profile-vetoed fits aside. Every surface that reports a
    run has to resolve it the same way or they disagree about which model the run
    produced. Falls back to the last fit for a legacy state with no selection.
    """
    fit_list = result.get("fit_results") or []
    selection = result.get("final_selection") or {}
    if not fit_list:
        return fit_list, selection, None
    index = selection.get("index") if selection.get("selected") else None
    if isinstance(index, int) and 0 <= index < len(fit_list):
        return fit_list, selection, fit_list[index]
    return fit_list, selection, fit_list[-1]


def _selection_payload(result: dict) -> dict:
    """Machine-readable account of which fit is being reported, and why.

    Without this a script cannot tell that the χ² it is gating on belongs to an
    earlier iteration than the last one fitted, nor that the reported model failed
    the SLD-profile check — so it would ingest a physically impossible model with
    no signal at all.
    """
    from .nodes.finalize import has_profile_artifact

    fit_list, selection, fit = _selected_fit(result)
    return {
        "iteration": selection.get("iteration"),
        "index": selection.get("index"),
        "of_iterations_fitted": len(fit_list),
        "superseded_last_iteration": bool(selection.get("superseded_last_iteration")),
        "profile_artifact": has_profile_artifact(fit),
        "demoted_for_profile_artifact": bool(
            selection.get("demoted_for_profile_artifact")
        ),
    }


def _pending_hypotheses_payload(hypotheses: Optional[list]) -> list:
    """The untried backlog, reduced to the fields a machine consumer needs."""
    from .nodes.finalize import pending_hypotheses

    return [
        {"id": h.get("id"), "title": h.get("title"), "change": h.get("change")}
        for h in pending_hypotheses(hypotheses)
    ]


def _echo_pending_hypotheses(
    hypotheses: Optional[list],
    *,
    indent: str,
    details: bool = False,
    lead_blank: bool = False,
) -> None:
    """Print the untried structural hypotheses, or nothing when there are none.

    Shared by the human report and ``aure batch`` — via the same selector and
    bucket labels the ``finalize`` node's own message uses — so the surfaces
    cannot disagree about one run's backlog. ``details`` adds the concrete
    change and the attempted tally the full report has room for; batch output
    stays a title list. ``lead_blank`` separates the block from whatever
    precedes it, and is suppressed along with the block itself.
    """
    from .nodes.finalize import (
        format_attempted_counts,
        hypothesis_label,
        pending_hypotheses,
    )

    pending = pending_hypotheses(hypotheses)
    if not pending:
        return
    if lead_blank:
        click.echo()
    click.echo(
        click.style(f"{indent}Possible further improvements", fg="cyan", bold=True)
    )
    for hyp in pending:
        click.echo(f"{indent}  {hypothesis_label(hyp)}")
        if details and hyp.get("change"):
            click.echo(f"{indent}      → {hyp['change']}")
    if details:
        click.echo(f"{indent}  ({format_attempted_counts(hypotheses)})")


def _fmt_num(value: Any, spec: str = ".2f") -> str:
    """Format a number that may be missing, None, non-numeric or non-finite.

    Not every field the workflow reports is guaranteed to carry a value: a
    layer whose thickness the sample description never pinned down reaches the
    summary as None. Formatting that directly raises TypeError, which would
    abort the run report — and the run's exit status — after the fit has
    already succeeded and been written to disk.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value):
            return format(value, spec)
    return "n/a"


def _print_analysis_results(result: dict, output_dir: Optional[str] = None):
    """Pretty-print analysis results."""
    click.echo()

    # Data loaded
    n_points = len(result.get("Q", []))
    click.echo(click.style("  Data Loaded", fg="cyan", bold=True))
    click.echo(f"    Points: {n_points}")

    if result.get("Q"):
        import numpy as np

        Q = np.array(result["Q"])
        click.echo(f"    Q range: {Q.min():.4f} - {Q.max():.4f} Å⁻¹")
    click.echo()

    # Parsed sample
    parsed = result.get("parsed_sample")
    if parsed:
        click.echo(click.style("  Sample Structure", fg="cyan", bold=True))
        substrate = parsed.get("substrate") or {}
        click.echo(
            f"    Substrate: {substrate.get('name', 'unknown')} "
            f"(SLD={_fmt_num(substrate.get('sld'))})"
        )
        for i, layer in enumerate(parsed.get("layers", [])):
            click.echo(
                f"    Layer {i + 1}: {layer.get('name', 'unknown')} - "
                f"{_fmt_num(layer.get('thickness'), '.0f')} Å "
                f"(SLD={_fmt_num(layer.get('sld'))})"
            )
        ambient = parsed.get("ambient") or {}
        click.echo(
            f"    Ambient: {ambient.get('name', 'unknown')} "
            f"(SLD={_fmt_num(ambient.get('sld'))})"
        )
        click.echo()

    # Extracted features
    features = result.get("extracted_features")
    if features:
        click.echo(click.style("  Extracted Features", fg="cyan", bold=True))
        click.echo(f"    Estimated layers: {features.get('estimated_n_layers', '?')}")
        thickness = features.get("estimated_total_thickness")
        if thickness:
            click.echo(f"    Total thickness: {thickness:.0f} Å")
        roughness = features.get("estimated_roughness")
        if roughness:
            click.echo(f"    Surface roughness: {roughness:.1f} Å")
        click.echo()

    # Model
    if result.get("current_model"):
        click.echo(click.style("  Final Model", fg="green", bold=True))
        click.echo(result.get("current_model"))
        #        model_lines = result["current_model"].split("\n")
        #        click.echo(f"    Script: {len(model_lines)} lines")
        click.echo()

    fit_list, selection, fit = _selected_fit(result)

    # Also grab the top-level chi² fields set by the workflow
    current_chi2 = result.get("current_chi2")
    best_chi2 = result.get("best_chi2")

    if fit:
        click.echo(click.style("  Fit Results", fg="cyan", bold=True))
        chi2 = fit.get("chi_squared", "N/A")
        click.echo(
            f"    χ²: {chi2:.4f}"
            if isinstance(chi2, (int, float))
            else f"    χ²: {chi2}"
        )
        if best_chi2 is not None and best_chi2 != chi2:
            click.echo(f"    Best χ²: {best_chi2:.4f}")
        click.echo(f"    Method: {fit.get('method', 'unknown')}")
        click.echo(f"    Iterations: {len(fit_list)}")
        if selection.get("selected"):
            click.echo(f"    Selected iteration: {selection.get('iteration')}")
            if selection.get("superseded_last_iteration"):
                from .nodes.finalize import _fmt_chi2

                last_chi2 = _fmt_chi2(selection.get("last_iteration_chi2"))
                click.echo(
                    click.style(
                        f"    ↳ chosen over the last iteration fitted "
                        f"(χ² = {last_chi2})",
                        fg="yellow",
                    )
                )

        if fit.get("parameters"):
            click.echo("    Parameters:")
            for name, value in fit["parameters"].items():
                # `or {}` not a `{}` default: fitting.py writes an explicit
                # `uncertainties=None` whenever the optimizer produced none
                # (every `lm`/`amoeba` run), so the key exists and holds None.
                # Guarding only the missing key aborted the whole report — and
                # with it the untried-improvements block printed further down.
                unc = (fit.get("uncertainties") or {}).get(name)
                if unc:
                    click.echo(f"      {name}: {value:.4f} ± {unc:.4f}")
                else:
                    click.echo(f"      {name}: {value:.4f}")
        click.echo()
    elif current_chi2 is not None:
        click.echo(click.style("  Fit Results", fg="cyan", bold=True))
        click.echo(f"    χ²: {current_chi2:.4f}")
        if best_chi2 is not None and best_chi2 != current_chi2:
            click.echo(f"    Best χ²: {best_chi2:.4f}")
        click.echo()

    # The SLD-profile veto, which χ² cannot show. Loudest thing in the report when
    # the model being reported is the one the check rejected.
    from .nodes.finalize import has_profile_artifact

    if selection.get("selected_has_profile_artifact") or has_profile_artifact(fit):
        click.echo(
            click.style(
                "  ⚠  THE REPORTED MODEL IS NOT PHYSICALLY VALID", fg="red", bold=True
            )
        )
        click.echo(
            click.style(
                "     Its SLD profile leaves the range its bounding media can "
                "produce.\n     Do not report these slabs as distinct layers.",
                fg="red",
            )
        )
        click.echo()
    elif selection.get("demoted_for_profile_artifact"):
        for v in selection.get("vetoed_iterations") or []:
            from .nodes.finalize import _fmt_chi2 as _fc

            click.echo(
                click.style(
                    f"    ↳ iteration {v.get('iteration')} fitted better "
                    f"(χ² = {_fc(v.get('chi_squared'))}) but was vetoed by the "
                    f"SLD-profile check",
                    fg="yellow",
                )
            )
        click.echo()

    # Evaluation. `result["evaluation"]` is only ever set by the MCP server; a
    # workflow run records the evaluator's findings on the judged FitResult, so
    # reading only the former made this whole block dead code for `aure analyze`
    # and the excursion text above never reached the reader.
    evaluation = result.get("evaluation")
    if evaluation:
        quality = evaluation.get("chi_squared_quality", "unknown")
        acceptable = evaluation.get("acceptable", False)
        color = "green" if acceptable else "yellow"
        click.echo(click.style(f"  Fit Quality: {quality}", fg=color, bold=True))
    else:
        evaluation = fit or {}
        if evaluation.get("issues") or evaluation.get("suggestions"):
            click.echo(click.style("  Fit Quality Notes", fg="cyan", bold=True))

    if evaluation.get("issues"):
        click.echo("    Issues:")
        for issue in evaluation["issues"]:
            click.echo(f"      - {issue}")

    if evaluation.get("suggestions"):
        click.echo("    Suggestions:")
        for sug in evaluation["suggestions"]:
            click.echo(f"      - {sug}")

    # Untried structural hypotheses. A run that clears the χ² threshold stops
    # with candidates still on the backlog, and this report is where a reader
    # finds out what the run chose not to spend its budget on.
    _echo_pending_hypotheses(
        result.get("structural_hypotheses"),
        indent="  ",
        details=True,
        lead_blank=True,
    )

    # Output directory info
    if output_dir:
        click.echo()
        click.echo(click.style("  Output Directory", fg="cyan", bold=True))
        click.echo(f"    Checkpoints: {output_dir}/checkpoints/")
        click.echo(f"    Final state: {output_dir}/final_state.json")


# ============================================================================
# Prepare Command (intake → analysis → modeling only)
# ============================================================================


@cli.command()
@click.argument("data_file", type=click.Path(exists=True), required=False)
@click.argument("sample_description", required=False)
@click.option(
    "--hypothesis",
    "-h",
    help="Optional hypothesis to test",
)
@click.option(
    "--extra-data",
    "-d",
    multiple=True,
    type=click.Path(exists=True),
    help="Additional data files for multi-file co-refinement (single state)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Output directory for checkpoints, models, and problem.json "
    "(default: ./output/<model-name>)",
)
@click.option(
    "--model-name",
    "-n",
    help="Base name for artifacts and the generated problem.json "
    "(default: derived from the data file stem)",
)
@click.option(
    "--config",
    "-c",
    "config_file",
    type=click.Path(exists=True),
    help="Setup YAML file (states, model constraints, …)",
)
@click.option(
    "--data-dir",
    "data_dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Directory to resolve relative data_files against (highest priority). "
    "Search order: this dir → the config file's dir → the current directory.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def prepare(
    data_file: Optional[str],
    sample_description: Optional[str],
    hypothesis: Optional[str],
    extra_data: tuple,
    output_dir: Optional[str],
    model_name: Optional[str],
    config_file: Optional[str],
    data_dir: Optional[str],
    output_json: bool,
    verbose: bool,
):
    """
    Run intake, analysis, and modeling only; emit a refl1d-ready problem.json.

    DATA_FILE and SAMPLE_DESCRIPTION are the ad-hoc single-file shortcut.
    For multi-state co-refinement, pass a setup YAML via ``-c setup.yaml``
    and omit the positional arguments.

    This stops before fitting, producing a ModelDefinition and a
    ``problem.json`` that can be loaded directly by refl1d (``refl1d
    <file>.json``) or submitted to a remote fit service.

    \b
    Examples:
        aure prepare data.dat "100 nm polystyrene on silicon"
        aure prepare data.dat "multilayer" -o ./out -n my_model
        aure prepare low-Q.dat "multilayer" -d mid-Q.dat -d high-Q.dat
        aure prepare -c setup.yaml -n my_model
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            stream=sys.stderr,
        )

    from .setup import (
        SetupConfig,
        load_setup,
        primary_data_file,
        setup_to_user_config,
    )
    from .workflow import run_prepare
    from .nodes.model_builder import save_problem_json

    # ── Load setup ────────────────────────────────────────────────
    setup: SetupConfig = {}  # type: ignore[assignment]
    if config_file:
        try:
            setup = load_setup(config_file, data_dir=data_dir)
        except Exception as e:
            click.echo(click.style(f"  Setup error: {e}", fg="red"))
            sys.exit(2)
    elif data_dir:
        click.echo(
            click.style(
                "  --data-dir has no effect without --config; ignoring.",
                fg="yellow",
            )
        )

    # ── Merge positional overrides ────────────────────────────────
    if sample_description:
        setup["sample_description"] = sample_description
    if hypothesis:
        setup["hypothesis"] = hypothesis
    if model_name:
        setup["model_name"] = model_name

    # ── Resolve data files ────────────────────────────────────────
    states = setup.get("states") or []
    if states:
        if extra_data:
            click.echo(
                click.style(
                    "  Cannot combine --extra-data with `states:` in the setup file.",
                    fg="red",
                )
            )
            sys.exit(2)
    else:
        if not data_file:
            click.echo(
                click.style(
                    "  Missing data file. Pass it as the positional argument "
                    "or define a `states:` block in --config.",
                    fg="red",
                )
            )
            sys.exit(2)
        from pathlib import Path as _Path

        all_files = [data_file] + list(extra_data)
        states = [
            {
                "name": "state0",
                "data_files": [
                    {"file": str(_Path(f).resolve()), "label": _Path(f).stem}
                    for f in all_files
                ],
            }
        ]
        setup["states"] = states

    if not setup.get("sample_description"):
        click.echo(
            click.style(
                "  Missing sample description. Provide it as the positional "
                "argument or set `sample_description:` in --config.",
                fg="red",
            )
        )
        sys.exit(2)

    # ── Derive runner kwargs ──────────────────────────────────────
    sample_description = setup["sample_description"]
    hypothesis = setup.get("hypothesis")
    user_config = setup_to_user_config(setup)
    data_file = primary_data_file(setup)
    # Multi-state runs use the `states` arg; legacy `data_files` is left empty.
    data_files = None

    resolved_model_name = setup.get("model_name") or Path(data_file).stem
    resolved_output_dir = output_dir or str(Path("output") / resolved_model_name)

    # Same reason as `analyze`: the setup's run controls reach the workflow as
    # environment variables, so they must be in force before the runner starts.
    with _applied_env_overrides(setup):
        if not output_json:
            click.echo(click.style("═" * 60, fg="blue"))
            click.echo(
                click.style("  Reflectivity Analysis — Prepare", fg="blue", bold=True)
            )
            click.echo(click.style("═" * 60, fg="blue"))
            click.echo()

            llm_ok, llm_msg = _check_llm_status(quiet=False, test_connection=True)
            click.echo()

            click.echo(f"  Data file:  {data_file}")
            n_files = sum(len(s.get("data_files", [])) for s in states)
            if len(states) > 1:
                for st in states:
                    names = ", ".join(ds["label"] for ds in st.get("data_files", []))
                    click.echo(f"  State {st['name']}: {names}")
                click.echo(
                    f"  Multi-state co-refinement: {len(states)} states, "
                    f"{n_files} files"
                )
            elif n_files > 1:
                for ds in states[0].get("data_files", []):
                    click.echo(f"  + {ds['file']}")
                click.echo(f"  Co-refinement: {n_files} files (shared structure)")
            click.echo(f"  Sample:     {sample_description}")
            if hypothesis:
                click.echo(f"  Hypothesis: {hypothesis}")
            click.echo(f"  Model name: {resolved_model_name}")
            click.echo(f"  Output dir: {resolved_output_dir}")
            click.echo()
        else:
            llm_ok, llm_msg = _check_llm_status(quiet=True, test_connection=True)

        if not llm_ok:
            if output_json:
                click.echo(
                    json.dumps(
                        {
                            "error": f"LLM not available: {llm_msg}",
                            "llm": get_llm_info(),
                        }
                    )
                )
            else:
                click.echo(click.style(f"  Cannot proceed: {llm_msg}", fg="red"))
            sys.exit(1)

        def checkpoint_callback(state, node_name):
            if not output_json:
                status = "✓" if not state.get("error") else "✗"
                click.echo(
                    click.style(
                        f"  [{status}] {node_name.title()}",
                        fg="green" if status == "✓" else "red",
                    )
                )

        try:
            result = run_prepare(
                data_file=data_file,
                sample_description=sample_description,
                hypothesis=hypothesis,
                output_dir=resolved_output_dir,
                checkpoint_callback=checkpoint_callback if not output_json else None,
                user_config=user_config,
                data_files=data_files,
                states=states or None,
            )
        except Exception as e:
            if output_json:
                click.echo(json.dumps({"error": str(e)}))
            else:
                click.echo(click.style(f"Error: {e}", fg="red"))
            sys.exit(1)

    if result.get("error"):
        if output_json:
            click.echo(json.dumps({"error": result["error"]}))
        else:
            click.echo(click.style(f"Error: {result['error']}", fg="red"))
        sys.exit(1)

    model = result.get("current_model")
    if not model:
        msg = "Modeling node did not produce a model"
        if output_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(click.style(f"Error: {msg}", fg="red"))
        sys.exit(1)

    # Write problem.json (bumps-serialised) and _definition.json (raw ModelDefinition)
    problem_path = Path(resolved_output_dir) / f"{resolved_model_name}.json"
    definition_path = (
        Path(resolved_output_dir) / f"{resolved_model_name}_definition.json"
    )
    try:
        if isinstance(model, str):
            raise RuntimeError(
                "Cannot serialize a legacy script-string model to problem.json. "
                "Re-run with an LLM that produces JSON ModelDefinitions."
            )
        loaded_data_files = result.get("data_files") or []
        save_problem_json(
            model,
            problem_path,
            data_files=loaded_data_files if len(loaded_data_files) > 1 else None,
        )
        sidecar = dict(model)
        if loaded_data_files:
            # Strip the Q/R/dR arrays intake attaches; keep only metadata
            # needed to rebuild the co-refinement.
            sidecar["data_files"] = [
                {k: v for k, v in ds.items() if k not in {"Q", "R", "dR"}}
                for ds in loaded_data_files
            ]
        definition_path.parent.mkdir(parents=True, exist_ok=True)
        definition_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    except Exception as e:
        if output_json:
            click.echo(json.dumps({"error": f"Failed to write problem.json: {e}"}))
        else:
            click.echo(click.style(f"Error writing problem.json: {e}", fg="red"))
        sys.exit(1)

    if output_json:
        click.echo(
            json.dumps(
                {
                    "success": True,
                    "output_dir": resolved_output_dir,
                    "model_name": resolved_model_name,
                    "problem_json": str(problem_path),
                    "definition_json": str(definition_path),
                    "n_points": len(result.get("Q", [])),
                    "parsed_sample": result.get("parsed_sample"),
                    "extracted_features": result.get("extracted_features"),
                },
                indent=2,
            )
        )
    else:
        _print_analysis_results(result, resolved_output_dir)
        click.echo()
        click.echo(click.style("  Problem JSON", fg="green", bold=True))
        click.echo(f"    {problem_path}")
        click.echo(f"    {definition_path}  (raw ModelDefinition)")
        click.echo()
        click.echo(click.style("  Next steps:", fg="cyan"))
        click.echo(f"    refl1d {problem_path}")


# ============================================================================
# Batch / Manifest Command
# ============================================================================


@cli.command()
@click.argument("manifest", type=click.Path(exists=True))
@click.option(
    "--job",
    "-j",
    multiple=True,
    help="Run only the named job(s).  May be repeated.  Default: all jobs.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate the manifest and print the job plan without running anything.",
)
@click.option(
    "--data-dir",
    "data_dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Directory to resolve relative data_files against, applied to every "
    "job (overrides per-job and defaults `data_dir:` keys). Search order: "
    "this dir → the manifest's dir → the current directory.",
)
def batch(manifest: str, job: tuple, dry_run: bool, data_dir: Optional[str]):
    """
    Run one or more jobs from a YAML manifest file.

    MANIFEST: Path to a YAML file describing batch jobs.

    Two manifest shapes are accepted:

    1. **Multi-job**: top-level ``jobs:`` list + optional ``defaults:`` block.
    2. **Flat setup**: a single setup file (no ``jobs:`` wrapper). Treated
       as a one-job manifest.

    See manifest.example.yaml for the full schema.

    \b
    Examples:
        aure batch manifest.yaml
        aure batch manifest.yaml -j copper_on_silicon
        aure batch manifest.yaml --dry-run
        aure batch setup.yaml                # flat single-job manifest
    """
    import os
    from .setup import (
        load_manifest,
        primary_data_file,
        setup_to_user_config,
    )

    manifest_path = Path(manifest).resolve()

    # ── Load manifest (also accepts a flat setup file) ────────────
    try:
        loaded = load_manifest(manifest_path, data_dir=data_dir)
    except Exception as exc:
        raise click.BadParameter(str(exc)) from exc

    jobs = loaded["jobs"]

    if job and len(jobs) == 1 and not loaded["defaults"]:
        # A flat setup has nothing to filter; refuse politely.
        raise click.BadParameter("--job filter is only valid for multi-job manifests.")

    # Filter by --job if specified
    if job:
        selected = {n for n in job}
        jobs = [j for j in jobs if j.get("name") in selected]
        missing = selected - {j.get("name", "") for j in jobs}
        if missing:
            raise click.BadParameter(
                f"Job(s) not found in manifest: {', '.join(sorted(missing))}"
            )

    # Every job needs a name (we use it for the output subdir). Default
    # to "job{i}" if missing — typically only happens for flat setups
    # that didn't declare one.
    for i, j in enumerate(jobs):
        j.setdefault("name", f"job{i}")
        command = str(j.get("command", "analyze")).strip().lower()
        if command not in {"analyze", "prepare"}:
            raise click.BadParameter(
                f"Job '{j['name']}' has invalid command '{command}'. "
                "Use 'analyze' or 'prepare'."
            )
        j["command"] = command

    manifest_dir = manifest_path.parent

    # ── Dry-run report ─────────────────────────────────────────
    click.echo(click.style("═" * 60, fg="blue"))
    click.echo(click.style("  AuRE Batch Runner", fg="blue", bold=True))
    click.echo(click.style("═" * 60, fg="blue"))
    click.echo(f"  Manifest : {manifest_path}")
    click.echo(f"  Jobs     : {len(jobs)}")
    click.echo()

    for j in jobs:
        name = j["name"]
        command = j["command"]
        output_root = _resolve_path(j.get("output_root", "./output"), manifest_dir)
        output_dir = str(Path(output_root) / name)
        states = j.get("states", [])
        click.echo(f"  • {name}")
        click.echo(f"      mode   : {command}")
        n_files = sum(len(s.get("data_files") or []) for s in states)
        if len(states) > 1:
            for st in states:
                files = st.get("data_files") or []
                names = ", ".join(Path(ds["file"]).name for ds in files)
                click.echo(f"      state  : {st.get('name', '?')} ({names})")
            click.echo(f"      files  : {len(states)} states, {n_files} files")
        else:
            for ds in states[0].get("data_files", []):
                click.echo(f"      data   : {ds['file']}")
        click.echo(f"      sample : {j.get('sample_description', '')[:72]}")
        click.echo(f"      output : {output_dir}")
        if j.get("hypothesis"):
            click.echo(f"      hypothesis : {j['hypothesis'][:72]}")
        if command == "analyze":
            click.echo(
                f"      fit    : {j.get('fit_method', 'dream')} "
                f"steps={j.get('fit_steps', 1000)} "
                f"burn={j.get('fit_burn', 1000)}"
            )
            if j.get("fit_method_final"):
                click.echo(
                    f"      final  : {j['fit_method_final']} "
                    f"steps={j.get('fit_steps_final', 10000)} "
                    f"burn={j.get('fit_burn_final', j.get('fit_steps_final', 10000))}"
                )
            click.echo(f"      refine : max {j.get('max_refinements', 5)}")
        else:
            click.echo(f"      model  : {j.get('model_name', name)}")
        click.echo()

    if dry_run:
        click.echo(click.style("  (dry-run – nothing executed)", fg="yellow"))
        return

    # ── Execute ────────────────────────────────────────────────
    from .workflow import run_analysis, run_prepare
    from .nodes.model_builder import save_problem_json

    results_summary: list[dict] = []

    for idx, j in enumerate(jobs, 1):
        name = j["name"]
        command = j["command"]
        output_root = _resolve_path(j.get("output_root", "./output"), manifest_dir)
        output_dir = str(Path(output_root) / name)

        states = j.get("states") or []
        data_file = primary_data_file(j)
        job_user_config = setup_to_user_config(j) or None

        click.echo(click.style(f"  [{idx}/{len(jobs)}] {name}", fg="cyan", bold=True))

        env_overrides = _build_env_overrides(j)
        saved_env = {k: os.environ.get(k) for k in env_overrides}
        os.environ.update(env_overrides)

        verbose = j.get("verbose", False)
        output_json = j.get("json", False)

        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(message)s",
                stream=sys.stderr,
            )

        def checkpoint_cb(state, node_name):
            if not output_json:
                status = "✓" if not state.get("error") else "✗"
                click.echo(
                    click.style(
                        f"    [{status}] {node_name.title()}",
                        fg="green" if status == "✓" else "red",
                    )
                )

        try:
            if command == "prepare":
                result = run_prepare(
                    data_file=data_file,
                    sample_description=j["sample_description"],
                    hypothesis=j.get("hypothesis"),
                    output_dir=output_dir,
                    checkpoint_callback=checkpoint_cb if not output_json else None,
                    states=states or None,
                    user_config=job_user_config,
                )
                model = result.get("current_model")
                if not model:
                    raise RuntimeError("Modeling node did not produce a model")
                if isinstance(model, str):
                    raise RuntimeError(
                        "Cannot serialize a legacy script-string model to problem.json"
                    )

                resolved_model_name = j.get("model_name", name)
                problem_path = Path(output_dir) / f"{resolved_model_name}.json"
                definition_path = (
                    Path(output_dir) / f"{resolved_model_name}_definition.json"
                )
                loaded_data_files = result.get("data_files") or []
                save_problem_json(
                    model,
                    problem_path,
                    data_files=loaded_data_files
                    if len(loaded_data_files) > 1
                    else None,
                )
                sidecar = dict(model)
                if loaded_data_files:
                    sidecar["data_files"] = [
                        {k: v for k, v in ds.items() if k not in {"Q", "R", "dR"}}
                        for ds in loaded_data_files
                    ]
                definition_path.parent.mkdir(parents=True, exist_ok=True)
                definition_path.write_text(
                    json.dumps(sidecar, indent=2), encoding="utf-8"
                )

                results_summary.append(
                    {
                        "name": name,
                        "command": command,
                        "success": True,
                        "problem_json": str(problem_path),
                        "definition_json": str(definition_path),
                        "output_dir": output_dir,
                    }
                )

                if output_json:
                    click.echo(
                        json.dumps(
                            {
                                "job": name,
                                "command": command,
                                "success": True,
                                "problem_json": str(problem_path),
                                "definition_json": str(definition_path),
                                "output_dir": output_dir,
                            }
                        )
                    )
                else:
                    click.echo(
                        click.style(
                            f"    Done – wrote {problem_path.name}",
                            fg="green",
                        )
                    )
            else:
                result = run_analysis(
                    data_file=data_file,
                    sample_description=j["sample_description"],
                    hypothesis=j.get("hypothesis"),
                    max_iterations=int(j.get("max_refinements", 5)),
                    output_dir=output_dir,
                    checkpoint_callback=checkpoint_cb if not output_json else None,
                    states=states or None,
                    user_config=job_user_config,
                )
                # The χ² a CI gate reads. It must be the iteration finalize chose,
                # not the last one fitted — those differ whenever finalize rejects a
                # regression, prefers a simpler model, or sets a vetoed fit aside.
                chi2 = None
                job_selection = None
                if result.get("fit_results"):
                    _, _, selected = _selected_fit(result)
                    chi2 = (selected or {}).get("chi_squared")
                    job_selection = _selection_payload(result)

                results_summary.append(
                    {
                        "name": name,
                        "command": command,
                        "success": True,
                        "chi_squared": chi2,
                        "selection": job_selection,
                        "output_dir": output_dir,
                    }
                )

                if output_json:
                    click.echo(
                        json.dumps(
                            {
                                "job": name,
                                "command": command,
                                "success": True,
                                "chi_squared": chi2,
                                "selection": job_selection,
                                "output_dir": output_dir,
                            }
                        )
                    )
                else:
                    chi_str = f"χ² = {chi2:.3f}" if chi2 is not None else "no fit"
                    click.echo(click.style(f"    Done – {chi_str}", fg="green"))
                    if (job_selection or {}).get("profile_artifact"):
                        click.echo(
                            click.style(
                                "    ⚠  NOT PHYSICALLY VALID — the reported model "
                                "failed the SLD-profile check",
                                fg="red",
                                bold=True,
                            )
                        )
                    # Batch never calls `_print_analysis_results`, so without
                    # this the backlog a stopped-early run left behind is
                    # invisible outside final_state.json. Titles only.
                    _echo_pending_hypotheses(
                        result.get("structural_hypotheses"), indent="    "
                    )

        except Exception as e:
            results_summary.append(
                {
                    "name": name,
                    "success": False,
                    "error": str(e),
                }
            )
            if output_json:
                click.echo(json.dumps({"job": name, "success": False, "error": str(e)}))
            else:
                click.echo(click.style(f"    Error: {e}", fg="red"))
        finally:
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        click.echo()

    # ── Summary ────────────────────────────────────────────────
    ok = sum(1 for r in results_summary if r["success"])
    fail = len(results_summary) - ok
    click.echo(click.style("═" * 60, fg="blue"))
    color = "green" if fail == 0 else "yellow"
    click.echo(
        click.style(
            f"  Batch complete: {ok} succeeded, {fail} failed",
            fg=color,
            bold=True,
        )
    )
    click.echo(click.style("═" * 60, fg="blue"))

    if fail:
        sys.exit(1)


def _resolve_path(p: str, base: Path) -> str:
    """Resolve a path relative to *base* unless it is already absolute."""
    path = Path(p)
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def _build_env_overrides(merged: dict) -> dict[str, str]:
    """
    Build a dict of environment-variable overrides from the merged job config.

    Only keys that are explicitly present in the manifest are forwarded so that
    the .env / ambient environment is left alone for anything unspecified.
    """
    mapping = {
        # χ² acceptance threshold — the refinement loop's stop condition.
        "chi2_max": "CHI2_MAX",
        # Acceptance floor. A reduced χ² far below 1 means the residuals are much
        # smaller than the quoted uncertainties — an overestimated `dR` column or
        # free parameters absorbing the noise. That is evidence about the error
        # bars, not about the structure, so it must not read as a pass.
        "chi2_min": "CHI2_MIN",
        "fit_method": "FIT_METHOD",
        "fit_steps": "FIT_STEPS",
        "fit_burn": "FIT_BURN",
        # Optional final uncertainty fit: run this method (typically "dream")
        # once on the finalize-selected model, with its own larger budget, to
        # attach uncertainties the fast exploration method does not produce.
        "fit_method_final": "FIT_METHOD_FINAL",
        "fit_steps_final": "FIT_STEPS_FINAL",
        "fit_burn_final": "FIT_BURN_FINAL",
        "final_fit_chi2_max": "FINAL_FIT_CHI2_MAX",
        "llm_provider": "LLM_PROVIDER",
        "llm_model": "LLM_MODEL",
        "llm_api_key": "LLM_API_KEY",
        "llm_base_url": "LLM_BASE_URL",
        "llm_temperature": "LLM_TEMPERATURE",
        "llm_timeout": "LLM_TIMEOUT",
        "alcf_cluster": "ALCF_CLUSTER",
        "alcf_access_token": "ALCF_ACCESS_TOKEN",
    }
    overrides: dict[str, str] = {}
    for yaml_key, env_key in mapping.items():
        if yaml_key in merged:
            overrides[env_key] = str(merged[yaml_key])
    return overrides


@contextlib.contextmanager
def _applied_env_overrides(setup: dict):
    """Apply a setup's env-mapped run controls for the duration of a run.

    The previous environment is restored on exit — keys that were absent are
    unset again, not left as empty strings — so an in-process caller (tests,
    the web UI, a follow-up command) never inherits one run's overrides.
    """
    import os

    overrides = _build_env_overrides(setup)
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        # Inside the try: a mid-update failure leaves the environment partially
        # overridden, and only the finally below can undo that.
        os.environ.update(overrides)
        yield overrides
    finally:
        for key, previous in saved.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


# ============================================================================
# Checkpoint Commands
# ============================================================================


@cli.command()
@click.argument("checkpoint_path", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Output directory for new checkpoints (defaults to original)",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def resume(
    checkpoint_path: str,
    output_dir: Optional[str],
    output_json: bool,
    verbose: bool,
):
    """
    Resume a workflow from a checkpoint.

    CHECKPOINT_PATH: Path to a checkpoint JSON file

    This command loads a saved checkpoint and continues the workflow
    from where it left off. Useful for:

    - Retrying after a failure
    - Testing changes to specific nodes
    - Skipping early stages when iterating on later ones

    Examples:

        # Resume from after fitting
        python -m aure.cli resume output/checkpoints/004_fitting.json

        # Resume with new output directory
        python -m aure.cli resume output/checkpoints/003_modeling.json -o output_v2
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            stream=sys.stderr,
        )

    from .workflow import run_from_checkpoint, CheckpointManager

    if not output_json:
        click.echo(click.style("═" * 60, fg="blue"))
        click.echo(
            click.style("  Resume Workflow from Checkpoint", fg="blue", bold=True)
        )
        click.echo(click.style("═" * 60, fg="blue"))
        click.echo()

        # Load and display checkpoint info
        checkpoint_data = CheckpointManager.load_checkpoint(checkpoint_path)
        click.echo(f"  Checkpoint: {checkpoint_path}")
        click.echo(f"  Node: {checkpoint_data['node']}")
        click.echo(f"  Iteration: {checkpoint_data.get('iteration', 0)}")
        click.echo(f"  Timestamp: {checkpoint_data.get('timestamp', 'unknown')}")
        click.echo()

        # Check LLM status
        llm_ok, llm_msg = _check_llm_status(quiet=False, test_connection=True)
        click.echo()
    else:
        llm_ok, llm_msg = _check_llm_status(quiet=True, test_connection=True)

    if not llm_ok:
        if output_json:
            click.echo(json.dumps({"error": f"LLM not available: {llm_msg}"}))
        else:
            click.echo(click.style(f"  Cannot proceed: {llm_msg}", fg="red"))
        sys.exit(1)

    # Create checkpoint callback for progress reporting
    def checkpoint_callback(state, node_name):
        if not output_json:
            status = "✓" if not state.get("error") else "✗"
            click.echo(
                click.style(
                    f"  [{status}] {node_name.title()}",
                    fg="green" if status == "✓" else "red",
                )
            )

    try:
        result = run_from_checkpoint(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            checkpoint_callback=checkpoint_callback if not output_json else None,
        )
    except Exception as e:
        if output_json:
            click.echo(json.dumps({"error": str(e)}))
        else:
            click.echo(click.style(f"Error: {e}", fg="red"))
        sys.exit(1)

    if result.get("error"):
        if output_json:
            click.echo(json.dumps({"error": result["error"]}))
        else:
            click.echo(click.style(f"Error: {result['error']}", fg="red"))
        sys.exit(1)

    if output_json:
        click.echo(json.dumps({"success": True, "output_dir": output_dir}, indent=2))
    else:
        click.echo()
        click.echo(
            click.style("  Workflow resumed successfully", fg="green", bold=True)
        )


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True))
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
def checkpoints(output_dir: str, output_json: bool):
    """
    List checkpoints in an output directory.

    OUTPUT_DIR: Path to the output directory containing checkpoints

    Examples:

        python -m aure.cli checkpoints ./output
    """
    from .workflow import CheckpointManager

    checkpoint_list = CheckpointManager.list_checkpoints(output_dir)

    if output_json:
        click.echo(json.dumps(checkpoint_list, indent=2))
    else:
        if not checkpoint_list:
            click.echo("No checkpoints found.")
            return

        click.echo(click.style("  Checkpoints", fg="cyan", bold=True))
        click.echo()

        for cp in checkpoint_list:
            node = cp.get("node", "unknown")
            iteration = cp.get("iteration", 0)
            timestamp = cp.get("timestamp", "")
            filename = cp.get("file", "")

            iter_str = f" (iter {iteration})" if iteration > 0 else ""
            click.echo(f"    {filename}")
            click.echo(f"      Node: {node}{iter_str}")
            click.echo(f"      Time: {timestamp}")
            click.echo()


@cli.command("inspect-checkpoint")
@click.argument("checkpoint_path", type=click.Path(exists=True))
@click.option(
    "--show-state",
    "-s",
    is_flag=True,
    help="Show full state (can be verbose)",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
def inspect_checkpoint(checkpoint_path: str, show_state: bool, output_json: bool):
    """
    Inspect a checkpoint file.

    CHECKPOINT_PATH: Path to a checkpoint JSON file

    Examples:

        python -m aure.cli inspect-checkpoint output/checkpoints/004_fitting.json
        python -m aure.cli inspect-checkpoint output/checkpoints/004_fitting.json -s
    """
    from .workflow import CheckpointManager

    checkpoint_data = CheckpointManager.load_checkpoint(checkpoint_path)

    if output_json:
        if show_state:
            click.echo(json.dumps(checkpoint_data, indent=2))
        else:
            # Exclude large state fields
            summary = {k: v for k, v in checkpoint_data.items() if k != "state"}
            summary["state_keys"] = list(checkpoint_data.get("state", {}).keys())
            click.echo(json.dumps(summary, indent=2))
    else:
        click.echo(click.style("  Checkpoint Details", fg="cyan", bold=True))
        click.echo()
        click.echo(f"    Node: {checkpoint_data.get('node')}")
        click.echo(f"    Checkpoint ID: {checkpoint_data.get('checkpoint_id')}")
        click.echo(f"    Iteration: {checkpoint_data.get('iteration', 0)}")
        click.echo(f"    Timestamp: {checkpoint_data.get('timestamp')}")
        click.echo()

        state = checkpoint_data.get("state", {})

        # Summary info
        click.echo(click.style("  State Summary", fg="cyan", bold=True))
        click.echo(f"    Data points: {len(state.get('Q', []))}")
        click.echo(f"    Sample: {state.get('sample_description', 'N/A')[:50]}")
        click.echo(f"    Has model: {state.get('current_model') is not None}")
        click.echo(f"    Fit results: {len(state.get('fit_results', []))}")
        click.echo(f"    Current χ²: {state.get('current_chi2', 'N/A')}")
        click.echo(f"    Error: {state.get('error', 'None')}")
        click.echo()

        if show_state:
            click.echo(click.style("  Full State", fg="cyan", bold=True))
            # Pretty print state, but truncate long arrays
            for key, value in state.items():
                if isinstance(value, list) and len(value) > 5:
                    click.echo(f"    {key}: [{len(value)} items]")
                elif isinstance(value, str) and len(value) > 100:
                    click.echo(f"    {key}: {value[:100]}...")
                else:
                    click.echo(f"    {key}: {value}")


# ============================================================================
# Plotting Commands
# ============================================================================


@cli.command("plot-results")
@click.argument("output_dir", type=click.Path(exists=True))
@click.option(
    "--save",
    "-s",
    type=click.Path(),
    help="Save plot to file (PNG, PDF, SVG)",
)
@click.option(
    "--offset",
    "-f",
    default=10.0,
    help="Vertical offset factor between R(Q) curves (default: 10)",
)
@click.option(
    "--no-show",
    is_flag=True,
    help="Don't display plot interactively (useful with --save)",
)
def plot_results(
    output_dir: str,
    save: Optional[str],
    offset: float,
    no_show: bool,
):
    """
    Plot reflectivity and SLD profiles from workflow results.

    OUTPUT_DIR: Path to the output directory containing refl1d_output

    Creates a two-panel plot:
    - Left: R(Q) curves (log-log) with data and model predictions from each iteration
    - Right: SLD profiles for each model iteration

    Examples:

        # Interactive plot
        python -m aure.cli plot-results ./results

        # Save to file
        python -m aure.cli plot-results ./results -s results.png

        # Save without display
        python -m aure.cli plot-results /path/to/results -s results.pdf --no-show
    """
    from pathlib import Path
    from .workflow import CheckpointManager

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        click.echo(click.style("Error: matplotlib is required for plotting", fg="red"))
        click.echo("Install with: pip install matplotlib")
        sys.exit(1)

    output_path = Path(output_dir)
    refl1d_dir = output_path / "refl1d_output"
    checkpoints_dir = output_path / "checkpoints"

    # Check for refl1d output
    if not refl1d_dir.exists():
        click.echo(click.style("No refl1d_output directory found", fg="red"))
        sys.exit(1)

    # Find problem.json files from fit iterations
    fit_dirs = sorted(refl1d_dir.glob("fit_iter*_*"))
    problem_files = [
        (d, d / "problem.json") for d in fit_dirs if (d / "problem.json").exists()
    ]
    if not problem_files:
        click.echo(
            click.style("No problem.json files found in refl1d_output/", fg="red")
        )
        sys.exit(1)

    click.echo(click.style(f"  Found {len(problem_files)} fit result(s)", fg="cyan"))

    # Get list of checkpoints for experimental data
    checkpoint_list = CheckpointManager.list_checkpoints(output_dir)

    if not checkpoint_list:
        click.echo(click.style("No checkpoints found in directory", fg="red"))
        sys.exit(1)

    # Load data from first checkpoint
    first_cp = CheckpointManager.load_checkpoint(
        str(checkpoints_dir / checkpoint_list[0]["file"])
    )
    state = first_cp["state"]
    Q_data = np.array(state.get("Q", []))
    R_data = np.array(state.get("R", []))
    dR_data = np.array(state.get("dR", []))
    sample_desc = state.get("sample_description", "Unknown sample")[:50]

    if len(Q_data) == 0:
        click.echo(click.style("No experimental data found in checkpoints", fg="red"))
        sys.exit(1)

    # Get chi-squared values from fitting checkpoints
    # Map by (node, iteration) for more precise matching
    chi2_by_node = {}
    for cp_info in checkpoint_list:
        if cp_info["node"] in ("fitting", "evaluation", "refinement"):
            cp_path = str(checkpoints_dir / cp_info["file"])
            cp_data = CheckpointManager.load_checkpoint(cp_path)
            cp_state = cp_data["state"]
            chi2 = cp_state.get("current_chi2")
            node = cp_info["node"]
            iteration = cp_info.get("iteration", 0)
            if chi2 is not None:
                chi2_by_node[(node, iteration)] = chi2

    # Load and deserialize each problem.json to get R(Q) and SLD
    model_data = []
    import json as _json
    import re

    for fit_dir, problem_file in problem_files:
        try:
            # Parse iteration from directory name (e.g. fit_iter0_dream)
            match = re.search(r"fit_iter(\d+)_(\w+)", fit_dir.name)
            iteration = int(match.group(1)) if match else len(model_data)
            method = match.group(2) if match else "unknown"
            label = f"Fit iter {iteration} ({method})"
            sort_key = (iteration, 0)

            click.echo(f"    Loading {fit_dir.name}/problem.json...")

            from bumps.serialize import deserialize

            from .nodes.model_builder import data_chisq

            with open(problem_file) as f:
                problem = deserialize(_json.load(f))

            fitness = problem.fitness
            experiments = fitness._models if hasattr(fitness, "_models") else [fitness]

            # Use the first experiment for reflectivity and SLD
            exp = experiments[0]
            exp.update()
            Q_arr, R_arr = exp.reflectivity()

            z, sld = None, None
            try:
                z_arr, sld_arr, _ = exp.smooth_profile(dz=1.0)
                z = np.array(z_arr).tolist()
                sld = np.array(sld_arr).tolist()
            except Exception:
                pass

            chi2 = None
            try:
                chi2 = float(data_chisq(problem))
            except Exception:
                pass

            model_data.append(
                {
                    "iteration": iteration,
                    "sort_key": sort_key,
                    "label": label,
                    "Q": np.array(Q_arr).tolist(),
                    "R": np.array(R_arr).tolist(),
                    "z": z,
                    "sld": sld,
                    "chi2": chi2,
                    "file": fit_dir.name,
                }
            )
        except Exception as e:
            click.echo(
                click.style(
                    f"    Warning: Could not load {fit_dir.name}: {e}", fg="yellow"
                )
            )

    if not model_data:
        click.echo(click.style("No models could be loaded", fg="red"))
        sys.exit(1)

    # Sort by sort_key (iteration, sub-order)
    model_data.sort(key=lambda x: x.get("sort_key", (x["iteration"], 0)))

    click.echo(click.style(f"  Plotting {len(model_data)} model(s)", fg="cyan"))

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ========== Left panel: R(Q) curves ==========
    n_models = len(model_data)
    colors = plt.cm.viridis(np.linspace(0, 0.85, n_models))

    # Plot data (at top with maximum offset)
    base_offset = offset**n_models
    ax1.errorbar(
        Q_data,
        R_data * base_offset,
        yerr=dR_data * base_offset,
        fmt="o",
        markersize=2,
        color="gray",
        alpha=0.6,
        label="Data",
        capsize=0,
        zorder=1,
    )

    # Plot models with decreasing offsets
    for i, md in enumerate(model_data):
        offset_factor = offset ** (n_models - i - 1)

        label = md["label"]
        if md["chi2"] is not None:
            label += f" (χ²={md['chi2']:.1f})"

        # Plot model curve
        ax1.plot(
            md["Q"],
            np.array(md["R"]) * offset_factor,
            "-",
            color=colors[i],
            linewidth=1.5,
            label=label,
            zorder=2,
        )

        # Plot data at same offset (faded)
        if offset_factor != base_offset:
            ax1.errorbar(
                Q_data,
                R_data * offset_factor,
                yerr=dR_data * offset_factor,
                fmt="o",
                markersize=1.5,
                color="gray",
                alpha=0.3,
                capsize=0,
                zorder=0,
            )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Q (Å⁻¹)")
    ax1.set_ylabel("Reflectivity (offset)")
    ax1.set_title(f"R(Q) - {sample_desc}")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ========== Right panel: SLD profiles ==========
    has_sld = any(md["z"] is not None for md in model_data)

    if has_sld:
        for i, md in enumerate(model_data):
            if md["z"] is not None and md["sld"] is not None:
                ax2.plot(
                    md["z"],
                    md["sld"],
                    "-",
                    color=colors[i],
                    linewidth=1.5,
                    label=md["label"],
                )

        ax2.set_xlabel("Depth (Å)")
        ax2.set_ylabel("SLD (×10⁻⁶ Å⁻²)")
        ax2.set_title("SLD Profile")
        ax2.legend(loc="best", fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "SLD profiles not available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("SLD Profile")

    plt.tight_layout()

    # Save if requested
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        click.echo(click.style(f"  Plot saved to: {save}", fg="green"))

    # Show if requested
    if not no_show:
        plt.show()

    plt.close()


# ============================================================================
# Standalone Evaluation Command
# ============================================================================


@cli.command()
@click.argument("refl1d_dir", type=click.Path(exists=True))
@click.option(
    "--context",
    "-c",
    "context_prompt",
    default=None,
    help="Optional description of the sample / model to give the LLM context",
)
@click.option(
    "--hypothesis",
    "-h",
    default=None,
    help="Optional hypothesis being tested",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def evaluate(
    refl1d_dir: str,
    context_prompt: Optional[str],
    hypothesis: Optional[str],
    output_json: bool,
    verbose: bool,
):
    """
    Evaluate a refl1d fit result using LLM analysis.

    REFL1D_DIR: Path to a refl1d output directory containing problem.json
                (e.g. output/refl1d_output/fit_iter0_dream)

    This command loads a serialised bumps FitProblem, extracts fit results
    (χ², parameters, theory curves, SLD profile, residuals), and asks the
    LLM to assess the fit quality — without re-running the full workflow.

    The reported `acceptable` is ADVISORY: this command runs neither the
    SLD-profile check nor the deterministic χ² stop, so its verdict can differ
    from what `aure analyze` decided for the same fit. Judged against the
    `chi2_max` the run was launched with when its `final_state.json` can be
    found, otherwise against the ambient `CHI2_MAX`.

    Use --context / -c to provide a natural-language description of the
    sample so the LLM can judge physical plausibility.

    \b
    Examples:
        aure evaluate output/refl1d_output/fit_iter0_dream

        aure evaluate output/refl1d_output/fit_iter0_dream \\
            -c "100 nm polystyrene film on silicon"

        aure evaluate output/refl1d_output/fit_iter0_dream --json
    """
    import re as _re

    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            stream=sys.stderr,
        )

    refl1d_path = Path(refl1d_dir)

    # ── Locate problem.json ────────────────────────────────────
    problem_file = refl1d_path / "problem.json"
    if not problem_file.exists():
        # Maybe the user pointed at the parent refl1d_output dir;
        # pick the latest fit_iter* subdirectory.
        fit_dirs = sorted(refl1d_path.glob("fit_iter*_*/problem.json"))
        if fit_dirs:
            problem_file = fit_dirs[-1]
            refl1d_path = problem_file.parent
        else:
            click.echo(click.style(f"No problem.json found in {refl1d_dir}", fg="red"))
            sys.exit(1)

    # Parse iteration & method from directory name
    match = _re.search(r"fit_iter(\d+)_(\w+)", refl1d_path.name)
    iteration = int(match.group(1)) if match else 0
    method = match.group(2) if match else "unknown"

    if not output_json:
        click.echo(click.style("═" * 60, fg="blue"))
        click.echo(click.style("  Evaluate Refl1D Fit Result", fg="blue", bold=True))
        click.echo(click.style("═" * 60, fg="blue"))
        click.echo()
        click.echo(f"  Directory: {refl1d_path}")
        click.echo(f"  Iteration: {iteration}  Method: {method}")
        if context_prompt:
            click.echo(f"  Context: {context_prompt}")
        click.echo()

    # ── Check LLM ──────────────────────────────────────────────
    llm_ok, llm_msg = _check_llm_status(quiet=output_json, test_connection=True)
    if not output_json:
        click.echo()
    if not llm_ok:
        if output_json:
            click.echo(json.dumps({"error": f"LLM not available: {llm_msg}"}))
        else:
            click.echo(click.style(f"  Cannot proceed: {llm_msg}", fg="red"))
        sys.exit(1)

    # ── Deserialise problem.json ───────────────────────────────
    if not output_json:
        click.echo("  Loading problem.json...", nl=False)
    try:
        import json as _json
        from bumps.serialize import deserialize

        with open(problem_file) as f:
            problem = deserialize(_json.load(f))
        if not output_json:
            click.echo(click.style(" done", fg="green"))
    except Exception as e:
        if not output_json:
            click.echo(click.style(f" failed: {e}", fg="red"))
        else:
            click.echo(json.dumps({"error": f"Failed to load problem.json: {e}"}))
        sys.exit(1)

    # ── Extract FitResult ──────────────────────────────────────
    if not output_json:
        click.echo("  Extracting fit results...", nl=False)
    try:
        from .refl1d_import import extract_fit_result_from_problem

        fit_result = extract_fit_result_from_problem(
            problem,
            method=method,
            iteration=iteration,
            export_dir=str(refl1d_path),
        )
        if not output_json:
            click.echo(click.style(" done", fg="green"))
    except Exception as e:
        if not output_json:
            click.echo(click.style(f" failed: {e}", fg="red"))
        else:
            click.echo(json.dumps({"error": f"Failed to extract fit results: {e}"}))
        sys.exit(1)

    chi2 = fit_result["chi_squared"]
    if not output_json:
        click.echo(f"  χ² = {chi2:.4f}")
        click.echo()

    # ── Run LLM evaluation ─────────────────────────────────────
    if not output_json:
        click.echo("  Running LLM evaluation...", nl=False)

    from .nodes.evaluation import (
        analyze_fit_quality_with_llm,
        _check_boundary_hits,
        _get_chi2_max,
    )

    # Judge against the threshold the evaluated run was launched with, not whatever
    # CHI2_MAX this shell happens to carry — otherwise the verdict is measured
    # against a bar the run never used.
    run_state, state_source = _run_pinned_state(refl1d_path)
    chi2_max = _get_chi2_max(run_state)
    if not output_json:
        origin = f"pinned by the run ({state_source})" if run_state else "ambient"
        click.echo(f"  Acceptance threshold: χ² ≤ {chi2_max:g}  [{origin}]")
    boundary_hits = _check_boundary_hits(fit_result)

    try:
        analysis = analyze_fit_quality_with_llm(
            fit_result=fit_result,
            sample_description=context_prompt,
            hypothesis=hypothesis,
            features=None,
            chi2_max=chi2_max,
            boundary_hits=boundary_hits,
            per_file_results=fit_result.get("per_file_results"),
        )
        if not output_json:
            click.echo(click.style(" done", fg="green"))
    except Exception as e:
        if not output_json:
            click.echo(click.style(f" failed: {e}", fg="red"))
        else:
            click.echo(json.dumps({"error": f"LLM evaluation failed: {e}"}))
        sys.exit(1)

    # ── Output ─────────────────────────────────────────────────
    if output_json:
        result = {
            "directory": str(refl1d_path),
            "iteration": iteration,
            "method": method,
            "chi_squared": chi2,
            "parameters": fit_result.get("parameters", {}),
            "uncertainties": fit_result.get("uncertainties"),
            "acceptable": analysis.get("acceptable", False),
            # `acceptable` is the evaluator's opinion on this one exported fit. No
            # SLD-profile check and no deterministic χ² stop run here, so it can
            # disagree with what `aure analyze` decided; gate on the run's own
            # final_state.json / --json `selection` block instead.
            "acceptable_is_advisory": True,
            "chi2_max": chi2_max,
            "chi2_max_source": state_source or "environment",
            "quality_assessment": analysis.get("quality_assessment", "unknown"),
            "issues": analysis.get("issues", []),
            "suggestions": analysis.get("suggestions", []),
            "physical_concerns": analysis.get("physical_concerns", []),
        }
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo()
        # Parameters
        click.echo(click.style("  Fit Parameters", fg="cyan", bold=True))
        for name, value in fit_result["parameters"].items():
            unc = (fit_result.get("uncertainties") or {}).get(name)
            if unc:
                click.echo(f"    {name}: {value:.4f} ± {unc:.4f}")
            else:
                click.echo(f"    {name}: {value:.4f}")
        click.echo()

        # Per-file chi2
        per_file = fit_result.get("per_file_results")
        if per_file:
            click.echo(click.style("  Per-file χ²", fg="cyan", bold=True))
            for pf in per_file:
                click.echo(f"    {pf['label']}: χ² = {pf['chi_squared']:.3f}")
            click.echo()

        # Evaluation
        acceptable = analysis.get("acceptable", False)
        quality = analysis.get("quality_assessment", "unknown")
        color = "green" if acceptable else "yellow"
        click.echo(
            click.style(
                f"  Fit Quality: {quality} (χ² = {chi2:.3f})",
                fg=color,
                bold=True,
            )
        )

        if analysis.get("issues"):
            click.echo()
            click.echo(click.style("  Issues:", fg="yellow"))
            for issue in analysis["issues"]:
                click.echo(f"    - {issue}")

        if analysis.get("suggestions"):
            click.echo()
            click.echo(click.style("  Suggestions:", fg="cyan"))
            for sug in analysis["suggestions"]:
                click.echo(f"    - {sug}")

        if analysis.get("physical_concerns"):
            click.echo()
            click.echo(click.style("  Physical Concerns:", fg="yellow"))
            for concern in analysis["physical_concerns"]:
                click.echo(f"    - {concern}")

        click.echo()
        verdict = (
            click.style("  ✓ Fit ACCEPTABLE", fg="green", bold=True)
            if acceptable
            else click.style("  ✗ Fit NOT acceptable", fg="red", bold=True)
        )
        click.echo(verdict)
        # This command inspects one exported fit; it does not run the SLD-profile
        # check or the deterministic χ² stop, so its verdict is the evaluator's
        # opinion and can differ from what `aure analyze` would have decided on the
        # same fit. Applying the clamp here would force acceptance on χ² alone with
        # no profile check, which is the failure mode the clamp's guards exist for.
        click.echo(
            click.style(
                "  (advisory — the evaluator's judgement, not `aure analyze`'s "
                "acceptance decision: no SLD-profile check, no χ² stop)",
                fg="bright_black",
            )
        )
        click.echo()


def _extract_fit_result_from_problem(
    problem, method: str, iteration: int, export_dir: str
) -> dict:
    """Backwards-compatible re-export.

    The real implementation lives in :mod:`aure.refl1d_import`.
    """
    from .refl1d_import import extract_fit_result_from_problem

    return extract_fit_result_from_problem(
        problem, method=method, iteration=iteration, export_dir=export_dir
    )


# ============================================================================
# Import a refl1d fit into AuRE
# ============================================================================


@cli.command("import-refl1d")
@click.argument("refl1d_dir", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=None,
    help="Target AuRE output directory (default: a sibling of REFL1D_DIR "
    "named <REFL1D_DIR>_aure_import)",
)
@click.option(
    "--context",
    "-c",
    "sample_description",
    default=None,
    help="Sample description recorded on the imported run",
)
@click.option(
    "--hypothesis",
    "-h",
    default=None,
    help="Optional hypothesis to attach to the imported run",
)
@click.option(
    "--state-name",
    "state_names",
    multiple=True,
    help="Override recovered state names (repeatable; one per distinct sample). "
    "Cannot be combined with --setup.",
)
@click.option(
    "--setup",
    "setup_path",
    type=click.Path(exists=True),
    default=None,
    help="Setup YAML describing the original problem (analyzer plan-data "
    "output or hand-written). Provides authoritative sample description, "
    "state names, and original data file paths.",
)
@click.option(
    "--data-dir",
    "data_dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Directory holding the data files referenced (by name) in --setup. "
    "Use when the YAML lists bare filenames but the data lives elsewhere "
    "(e.g. plan in ./plan/, data in ./).",
)
@click.option(
    "--back-reflection/--no-back-reflection",
    "back_reflection",
    default=None,
    help="Force stack orientation. Default: auto-detect from material names.",
)
@click.option("--force", is_flag=True, help="Overwrite OUTPUT_DIR if it exists")
@click.option("--json", "output_json", is_flag=True, help="Machine-readable summary")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def import_refl1d_cmd(
    refl1d_dir: str,
    output_dir: Optional[str],
    sample_description: Optional[str],
    hypothesis: Optional[str],
    state_names: tuple,
    setup_path: Optional[str],
    data_dir: Optional[str],
    back_reflection: Optional[bool],
    force: bool,
    output_json: bool,
    verbose: bool,
):
    """Load a refl1d ``problem.json`` into an AuRE output directory.

    REFL1D_DIR may be a specific ``fit_iter*_*`` directory or its parent
    (the latest fit iteration is then picked automatically). The imported
    run can be opened with ``aure serve OUTPUT_DIR`` or extended with
    ``aure resume OUTPUT_DIR/checkpoints/005_evaluation.json``.

    Pass ``--setup`` when the original setup YAML (e.g. from analyzer's
    ``plan-data``) is still around. The setup becomes the source of truth
    for state names / sample description / original data paths, while the
    refl1d output supplies the fitted numbers. Without it, the importer
    auto-detects everything from the deserialised problem.

    \b
    Examples:
        aure import-refl1d ./refl1d_output/fit_iter0_dream -o ./imported
        aure import-refl1d ./refl1d_output -o ./imported -c "Cu/Ti on Si in D2O"
        aure import-refl1d ./fit_iter0_dream -o ./imported --state-name D2O --state-name H2O
        aure import-refl1d ./Cu-D2O-226642 --setup ./plan/job_Cu-D2O-226642.yaml
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            stream=sys.stderr,
        )

    from .refl1d_import import import_refl1d

    # Default to a SIBLING of the user-typed source, not a child — putting
    # the workspace inside the source would make the refl1d-tree copy step
    # recurse into the output directory.
    if output_dir:
        target = output_dir
    else:
        src = Path(refl1d_dir).resolve()
        target = str(src.parent / f"{src.name}_aure_import")

    if not output_json:
        click.echo(click.style("═" * 60, fg="blue"))
        click.echo(click.style("  Import refl1d → AuRE", fg="blue", bold=True))
        click.echo(click.style("═" * 60, fg="blue"))
        click.echo()
        click.echo(f"  Source:  {refl1d_dir}")
        click.echo(f"  Target:  {target}")
        click.echo()

    if data_dir and not setup_path:
        click.echo(
            click.style(
                "  --data-dir has no effect without --setup; ignoring.",
                fg="yellow",
            )
        )

    try:
        summary = import_refl1d(
            refl1d_dir,
            target,
            setup_path=setup_path,
            setup_data_dir=data_dir,
            sample_description=sample_description,
            hypothesis=hypothesis,
            state_names=list(state_names) or None,
            back_reflection=back_reflection,
            force=force,
        )
    except FileExistsError as e:
        msg = str(e)
        if output_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(click.style(f"  {msg}", fg="red"))
        sys.exit(2)
    except Exception as e:
        if output_json:
            click.echo(json.dumps({"error": f"Import failed: {e}"}))
        else:
            click.echo(click.style(f"  Import failed: {e}", fg="red"))
        sys.exit(1)

    if output_json:
        click.echo(json.dumps(summary, indent=2))
        return

    click.echo(click.style("  Imported successfully", fg="green", bold=True))
    click.echo(
        f"    states     : {len(summary['states'])} ({', '.join(summary['states'])})"
    )
    click.echo(f"    files      : {summary['n_files']}")
    click.echo(f"    χ²         : {summary['chi_squared']:.4f}")
    click.echo(f"    method     : {summary['method']}")
    click.echo(f"    back-refl. : {summary['back_reflection']}")

    # Tie summary (only emit when there's more than one state)
    if len(summary["states"]) > 1:
        tied = summary.get("tied_parameters") or []
        untied = summary.get("untied_parameters") or []
        click.echo(f"    tied       : {', '.join(tied) if tied else '(none)'}")
        click.echo(f"    untied     : {', '.join(untied) if untied else '(none)'}")

    # Warnings (constraint-expression ties, etc.)
    for w in summary.get("warnings") or []:
        click.echo()
        click.echo(click.style("  ⚠ " + w, fg="yellow"))

    click.echo()
    click.echo(click.style("  Next:", fg="cyan", bold=True))
    click.echo(f"    aure serve {summary['output_dir']}")
    click.echo(
        f"    aure resume {summary['output_dir']}/checkpoints/005_evaluation.json"
    )


# ============================================================================
# Material Database Commands
# ============================================================================


@cli.command("lookup-sld")
@click.argument("materials", nargs=-1, required=True)
@click.option(
    "--wavelength",
    "-w",
    default=1.8,
    help="Neutron wavelength in Angstroms (default: 1.8)",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
def lookup_sld(materials: tuple, wavelength: float, output_json: bool):
    """
    Look up SLD values for materials.

    MATERIALS: One or more material names or chemical formulas

    Examples:

        python -m aure.cli lookup-sld silicon gold D2O

        python -m aure.cli lookup-sld SiO2 Fe2O3 TiO2

        python -m aure.cli lookup-sld polystyrene PMMA
    """
    from .database.materials import get_sld, lookup_material

    results = []
    for mat in materials:
        try:
            sld = get_sld(mat)
            info = lookup_material(mat)
            results.append(
                {
                    "material": mat,
                    "sld": round(sld, 4),
                    "density": info.density if info else None,
                    "formula": info.formula if info else mat,
                }
            )
        except Exception as e:
            results.append({"material": mat, "error": str(e)})

    if output_json:
        click.echo(json.dumps(results, indent=2))
    else:
        click.echo()
        click.echo(click.style("  Material SLD Values", fg="cyan", bold=True))
        click.echo(f"  Wavelength: {wavelength} Å")
        click.echo()

        # Table header
        click.echo(f"  {'Material':<20} {'SLD (10⁻⁶ Å⁻²)':<18} {'Formula'}")
        click.echo(f"  {'-' * 20} {'-' * 18} {'-' * 20}")

        for r in results:
            if "error" in r:
                click.echo(
                    f"  {r['material']:<20} "
                    + click.style(f"Error: {r['error']}", fg="red")
                )
            else:
                click.echo(
                    f"  {r['material']:<20} {r['sld']:<18.4f} {r.get('formula', '')}"
                )
        click.echo()


@cli.command("list-materials")
@click.option(
    "--category",
    "-c",
    type=click.Choice(["polymers", "metals", "substrates", "solvents", "all"]),
    default="all",
    help="Filter by category",
)
def list_materials(category: str):
    """
    List common materials in the database.

    Shows materials with their SLD values for quick reference.
    """
    from .database.materials import get_sld

    click.echo()
    click.echo(click.style("  Material Database", fg="cyan", bold=True))
    click.echo()

    # Categories
    categories = {
        "polymers": ["polystyrene", "d-polystyrene", "PMMA", "d-PMMA", "PEO", "PDMS"],
        "metals": ["gold", "silver", "nickel", "titanium", "copper", "iron"],
        "substrates": ["silicon", "sapphire", "quartz", "glass"],
        "solvents": ["air", "D2O", "H2O", "toluene", "ethanol"],
    }

    if category == "all":
        for cat_name, mats in categories.items():
            click.echo(click.style(f"  {cat_name.title()}", bold=True))
            for mat in mats:
                try:
                    sld = get_sld(mat)
                    click.echo(f"    {mat:<20} SLD = {sld:>7.3f}")
                except Exception:
                    pass
            click.echo()
    else:
        mats = categories.get(category, [])
        click.echo(click.style(f"  {category.title()}", bold=True))
        for mat in mats:
            try:
                sld = get_sld(mat)
                click.echo(f"    {mat:<20} SLD = {sld:>7.3f}")
            except Exception:
                pass
        click.echo()


# ============================================================================
# Feature Extraction Commands
# ============================================================================


@cli.command("extract-features")
@click.argument("data_file", type=click.Path(exists=True))
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
def extract_features(data_file: str, output_json: bool):
    """
    Extract physics features from a reflectivity file.

    Analyzes the data to estimate thickness, roughness, and layer count
    without building a full model.

    DATA_FILE: Path to the reflectivity data file
    """
    from .tools.data_tools import load_reflectivity_data
    from .tools.feature_tools import (
        estimate_total_thickness,
        estimate_roughness,
        extract_critical_edges,
        estimate_layer_count,
    )
    import numpy as np

    # Load data
    try:
        Q, R, dR = load_reflectivity_data(data_file)
    except Exception as e:
        click.echo(click.style(f"Error loading data: {e}", fg="red"))
        sys.exit(1)

    Q = np.array(Q)
    R = np.array(R)

    features = {}

    # Critical edge
    try:
        qc = extract_critical_edges(Q, R)
        features["critical_edge"] = qc
    except Exception as e:
        features["critical_edge"] = {"error": str(e)}

    # Thickness
    try:
        thickness = estimate_total_thickness(Q, R)
        features["thickness"] = thickness
    except Exception as e:
        features["thickness"] = {"error": str(e)}

    # Roughness
    try:
        roughness = estimate_roughness(Q, R)
        features["roughness"] = roughness
    except Exception as e:
        features["roughness"] = {"error": str(e)}

    # Layer count
    try:
        layers = estimate_layer_count(Q, R)
        features["layer_count"] = layers
    except Exception as e:
        features["layer_count"] = {"error": str(e)}

    features["data"] = {
        "file": data_file,
        "n_points": len(Q),
        "q_min": float(Q.min()),
        "q_max": float(Q.max()),
    }

    if output_json:
        click.echo(json.dumps(features, indent=2))
    else:
        click.echo()
        click.echo(click.style("  Feature Extraction", fg="cyan", bold=True))
        click.echo(f"  File: {data_file}")
        click.echo(f"  Points: {len(Q)}")
        click.echo(f"  Q range: {Q.min():.4f} - {Q.max():.4f} Å⁻¹")
        click.echo()

        if "Qc" in features.get("critical_edge", {}):
            qc = features["critical_edge"]["Qc"]
            sld = features["critical_edge"].get("estimated_SLD", 0)
            click.echo(f"  Critical edge: Qc = {qc:.5f} Å⁻¹ (SLD ≈ {sld:.2f})")

        if "thickness" in features.get("thickness", {}):
            t = features["thickness"]["thickness"]
            n = features["thickness"].get("n_fringes", 0)
            click.echo(f"  Thickness: {t:.0f} Å ({n} fringes)")

        if "roughness" in features.get("roughness", {}):
            r = features["roughness"]["roughness"]
            click.echo(f"  Roughness: {r:.1f} Å")

        if "n_layers" in features.get("layer_count", {}):
            n = features["layer_count"]["n_layers"]
            conf = features["layer_count"].get("confidence", "unknown")
            click.echo(f"  Estimated layers: {n} ({conf} confidence)")

        click.echo()


# ============================================================================
# MCP Server Command
# ============================================================================


@cli.command("mcp-server")
@click.option(
    "--transport",
    "-t",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport protocol (stdio for Claude Desktop, sse for HTTP)",
)
@click.option(
    "--port",
    "-p",
    default=8000,
    help="Port for SSE transport (default: 8000)",
)
def mcp_server(transport: str, port: int):
    """
    Start the MCP server for AI assistant integration.

    This starts a Model Context Protocol server that allows AI assistants
    like Claude to interact with the reflectivity analysis workflow.

    For Claude Desktop, use stdio transport (default).
    For HTTP-based clients, use sse transport.

    Examples:

        python -m aure.cli mcp-server

        python -m aure.cli mcp-server --transport sse --port 8080
    """
    from .mcp_server import mcp

    click.echo(click.style("═" * 60, fg="blue"))
    click.echo(click.style("  Reflectivity Analysis MCP Server", fg="blue", bold=True))
    click.echo(click.style("═" * 60, fg="blue"))
    click.echo()
    click.echo(f"  Transport: {transport}")
    if transport == "sse":
        click.echo(f"  Port: {port}")
    click.echo()
    click.echo("  Available tools:")
    click.echo("    - lookup_material_sld")
    click.echo("    - compare_materials")
    click.echo("    - analyze_reflectivity_features")
    click.echo("    - start_analysis_session")
    click.echo("    - get_session_model")
    click.echo("    - run_fit")
    click.echo("    - evaluate_fit")
    click.echo("    - modify_model")
    click.echo("    - quick_analyze")
    click.echo()
    click.echo("  Starting server...")
    click.echo()

    if transport == "sse":
        mcp.run(transport="sse", port=port)
    else:
        mcp.run(transport="stdio")


# ============================================================================
# Web Viewer Command
# ============================================================================


@cli.command("serve")
@click.argument(
    "output_dir", type=click.Path(exists=True), required=False, default=None
)
@click.option(
    "--port",
    "-p",
    default=5000,
    type=int,
    help="Port to run the web server on (default: 5000)",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help=(
        "Interface to bind to. Use 0.0.0.0 to listen on all interfaces "
        "(e.g. inside Docker). Only use this on trusted networks or behind "
        "proper network controls, as the web UI is not authenticated."
    ),
)
@click.option(
    "--no-browser",
    is_flag=True,
    help="Don't open a browser automatically",
)
def serve(output_dir: Optional[str], port: int, host: str, no_browser: bool):
    """
    Launch the AuRE web interface.

    When OUTPUT_DIR is given the app opens in read-only viewer mode.
    When omitted it starts in interactive setup mode where you can
    pick a data file, describe the sample, and launch an analysis
    from the browser.

    \b
      History  – checkpoint timeline and χ² progression chart
      Results  – R(Q) plot, SLD profile, and fit parameter table

    Examples:

        aure serve               # interactive mode

        aure serve ./output      # viewer mode

        aure serve ./output --port 8080 --no-browser
    """
    from .web import create_app

    click.echo(click.style("═" * 60, fg="blue"))
    if output_dir:
        click.echo(click.style("  AuRE – Results Viewer", fg="blue", bold=True))
    else:
        click.echo(click.style("  AuRE – Interactive Mode", fg="blue", bold=True))
    click.echo(click.style("═" * 60, fg="blue"))
    click.echo()
    if output_dir:
        click.echo(f"  Output dir: {output_dir}")
    else:
        click.echo("  Mode:       interactive setup")
    display_host = "localhost" if host == "0.0.0.0" else host
    click.echo(f"  URL:        http://{display_host}:{port}")
    click.echo()

    app = create_app(output_dir)

    if not no_browser:
        import threading
        import webbrowser

        threading.Timer(
            1.0, webbrowser.open, args=[f"http://{display_host}:{port}"]
        ).start()

    app.run(host=host, port=port, debug=False)


# ============================================================================
# Interactive Mode (Future)
# ============================================================================


@cli.command("interactive")
@click.argument("data_file", type=click.Path(exists=True), required=False)
@click.option("--port", "-p", default=5000, type=int, help="Port (default: 5000)")
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help=(
        "Interface to bind to. Use 0.0.0.0 to listen on all interfaces "
        "(e.g. inside Docker). Warning: this exposes the unauthenticated "
        "web UI/API and should only be used with appropriate network "
        "restrictions."
    ),
)
def interactive(data_file: Optional[str], port: int, host: str):
    """
    Start an interactive analysis session in the browser.

    Alias for ``aure serve`` in interactive setup mode.
    """
    from .web import create_app

    click.echo(click.style("═" * 60, fg="blue"))
    click.echo(click.style("  AuRE – Interactive Mode", fg="blue", bold=True))
    click.echo(click.style("═" * 60, fg="blue"))
    click.echo()
    display_host = "localhost" if host == "0.0.0.0" else host
    click.echo(f"  URL: http://{display_host}:{port}")
    click.echo()

    app = create_app()

    import threading
    import webbrowser

    threading.Timer(
        1.0, webbrowser.open, args=[f"http://{display_host}:{port}"]
    ).start()

    app.run(host=host, port=port, debug=False)


# ============================================================================
# Entry Point
# ============================================================================


def main():
    """CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
