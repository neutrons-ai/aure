"""
ISAAC AI-Ready Data exporter.

Drives the canonical *pull* pipeline: points the data-assembler at the workflow
run directory (``data-assembler ingest-workflow``) to produce neutral, typed
records, then ``nr-isaac-format convert-ingest`` to map those records into a
validated ISAAC JSON record.

AuRE no longer authors an ISAAC manifest or classifies environments — the
data-assembler is the structured-truth layer (it pulls the fitted model with
uncertainties, χ², and experimental conditions straight from the run dir) and
nr-isaac-format owns the ISAAC schema mapping. AuRE's only job here is to point
the assembler at the run directory and supply a human-readable context summary.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .base import BaseExporter, ExportResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM context generation
# ---------------------------------------------------------------------------

_CONTEXT_PROMPT_TEMPLATE = """\
You are a scientific data curator.  Summarise the following neutron
reflectometry analysis session into a single concise paragraph suitable
for a metadata record.  The paragraph should describe:
- The sample (composition, structure, substrate, ambient medium)
- The measurement technique and conditions
- Key analysis results (final chi-squared, number of layers, notable findings)

Keep it factual and under 200 words.  Do NOT use bullet points and do NOT
include headings.

## Sample Description
{sample_description}

## Hypothesis
{hypothesis}

## Final Fit
chi² = {chi2}
Parameters:
{param_summary}

## Conversation Highlights
{messages_summary}
"""


def _generate_context_description(state: dict) -> str:
    """Use the LLM to generate a context paragraph from the workflow state.

    Falls back to ``state["sample_description"]`` if the LLM is unavailable.
    """
    sample_description = state.get("sample_description", "Not specified")
    hypothesis = state.get("hypothesis") or "None provided"
    fallback = sample_description

    # Collect final fit info
    fit_results = state.get("fit_results") or []
    chi2 = state.get("current_chi2") or state.get("best_chi2") or "N/A"
    param_summary = "N/A"
    if fit_results:
        latest = fit_results[-1]
        params = latest.get("parameters", {})
        if params:
            lines = [f"  {k}: {v}" for k, v in list(params.items())[:15]]
            param_summary = "\n".join(lines)

    # Summarise messages (keep it short — last 10 non-system messages)
    messages = state.get("messages") or []
    relevant = [m for m in messages if m.get("role") in ("user", "assistant")][-10:]
    if relevant:
        msg_lines = [f"  [{m['role']}] {m['content'][:300]}" for m in relevant]
        messages_summary = "\n".join(msg_lines)
    else:
        messages_summary = "No conversation recorded."

    prompt = _CONTEXT_PROMPT_TEMPLATE.format(
        sample_description=sample_description,
        hypothesis=hypothesis,
        chi2=chi2,
        param_summary=param_summary,
        messages_summary=messages_summary,
    )

    try:
        from ..llm import get_llm, invoke_with_timeout

        llm = get_llm(temperature=0)
        response = invoke_with_timeout(llm, prompt, timeout_seconds=60)
        text = (
            response.content.strip()
            if hasattr(response, "content")
            else str(response).strip()
        )
        if text:
            return text
    except Exception as exc:
        logger.warning("[ISAAC-EXPORT] LLM context generation failed: %s", exc)

    return fallback


# ---------------------------------------------------------------------------
# IsaacExporter
# ---------------------------------------------------------------------------


class IsaacExporter(BaseExporter):
    """Export workflow results to ISAAC AI-Ready Record format (pull pipeline)."""

    @property
    def name(self) -> str:
        return "ISAAC AI-Ready Format"

    @property
    def format_id(self) -> str:
        return "isaac"

    def export(
        self,
        output_dir: Path,
        state: dict,
        run_info: dict,
        user_context: Optional[str] = None,
    ) -> ExportResult:
        errors: List[str] = []
        warnings: List[str] = []

        run_dir = Path(output_dir)
        ai_dir = run_dir / "ai-ready-data"
        ai_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Step A – ensure run_info.json is on disk for the assembler to pull.
        # The workflow normally writes it; only materialise it if missing so we
        # never clobber the canonical copy.
        # ------------------------------------------------------------------
        run_info_path = run_dir / "run_info.json"
        if not run_info_path.is_file() and run_info:
            try:
                run_info_path.write_text(json.dumps(run_info, default=str), encoding="utf-8")
            except OSError as exc:
                warnings.append(f"Could not write run_info.json: {exc}")

        # ------------------------------------------------------------------
        # Step B – human-readable context summary. Saved as an artifact and
        # passed to the record as the measurement notes.
        # ------------------------------------------------------------------
        context_description = _generate_context_description(state)
        if user_context:
            context_description = user_context + "\n\n" + context_description
        (ai_dir / "context.txt").write_text(context_description, encoding="utf-8")
        logger.info("[ISAAC-EXPORT] Wrote context description → %s", ai_dir / "context.txt")

        # ------------------------------------------------------------------
        # Step C – data-assembler pulls everything from the run dir → records.
        # ------------------------------------------------------------------
        if not self._run_ingest_workflow(run_dir, ai_dir, warnings):
            errors.append("data-assembler ingest-workflow failed; see warnings.")
            return ExportResult(
                success=False, output_path=ai_dir, errors=errors, warnings=warnings
            )

        # ------------------------------------------------------------------
        # Step D – nr-isaac-format maps the records → ISAAC JSON, with the
        # context summary as the measurement notes.
        # ------------------------------------------------------------------
        out_dir = ai_dir / "output"
        convert_ok = self._run_convert_ingest(ai_dir, out_dir, context_description, warnings)

        # ------------------------------------------------------------------
        # Step E – validate the produced record(s).
        # ------------------------------------------------------------------
        if convert_ok:
            self._run_validate(out_dir, warnings)

        return ExportResult(
            success=len(errors) == 0,
            output_path=ai_dir,
            errors=errors,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # CLI wrappers
    # ------------------------------------------------------------------

    def _run_ingest_workflow(self, run_dir: Path, ai_dir: Path, warnings: list[str]) -> bool:
        """Run ``data-assembler ingest-workflow <run_dir> -o <ai_dir> --json``."""
        cmd = self._find_assembler_command()
        if cmd is None:
            warnings.append(
                "data-assembler is not installed. "
                "Install with: pip install 'aure[export]' or "
                "pip install git+https://github.com/isaac-neutrons/data-assembler.git"
            )
            return False
        try:
            result = subprocess.run(
                [*cmd, "ingest-workflow", str(run_dir), "-o", str(ai_dir), "--json"],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode != 0:
                msg = result.stderr.strip() or result.stdout.strip()
                warnings.append(f"data-assembler ingest-workflow failed: {msg}")
                logger.warning("[ISAAC-EXPORT] ingest-workflow failed: %s", msg)
                return False
            logger.info("[ISAAC-EXPORT] ingest-workflow succeeded:\n%s", result.stdout.strip())
            return True
        except subprocess.TimeoutExpired:
            warnings.append("data-assembler ingest-workflow timed out after 180 s")
            return False
        except Exception as exc:
            warnings.append(f"data-assembler ingest-workflow error: {exc}")
            return False

    def _run_convert_ingest(
        self, ingest_dir: Path, out_dir: Path, context: str, warnings: list[str]
    ) -> bool:
        """Run ``nr-isaac-format convert-ingest <ingest_dir> -o <out_dir>``."""
        cmd = self._find_cli_command()
        if cmd is None:
            warnings.append(
                "nr-isaac-format is not installed. "
                "Install with: pip install 'aure[export]' or "
                "pip install git+https://github.com/isaac-neutrons/nr-isaac-format.git"
            )
            return False
        argv = [*cmd, "convert-ingest", str(ingest_dir), "-o", str(out_dir)]
        if context:
            argv += ["--context", context]
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                msg = result.stderr.strip() or result.stdout.strip()
                warnings.append(f"nr-isaac-format convert-ingest failed: {msg}")
                logger.warning("[ISAAC-EXPORT] convert-ingest failed: %s", msg)
                return False
            logger.info("[ISAAC-EXPORT] convert-ingest succeeded:\n%s", result.stdout.strip())
            return True
        except subprocess.TimeoutExpired:
            warnings.append("nr-isaac-format convert-ingest timed out after 120 s")
            return False
        except Exception as exc:
            warnings.append(f"nr-isaac-format convert-ingest error: {exc}")
            return False

    def _run_validate(self, output_dir: Path, warnings: list[str]) -> bool:
        """Run ``nr-isaac-format validate`` on each JSON in the output dir."""
        if not output_dir.is_dir():
            return False

        json_files = sorted(output_dir.glob("*.json"))
        if not json_files:
            warnings.append("No ISAAC JSON records found to validate.")
            return False

        cmd = self._find_cli_command()
        if cmd is None:
            warnings.append("nr-isaac-format CLI not found — skipping validation.")
            return False

        all_valid = True
        for jf in json_files:
            try:
                result = subprocess.run(
                    [*cmd, "validate", str(jf)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    msg = result.stderr.strip() or result.stdout.strip()
                    warnings.append(f"Validation failed for {jf.name}: {msg}")
                    all_valid = False
                else:
                    logger.info("[ISAAC-EXPORT] Validated: %s", jf.name)
            except Exception as exc:
                warnings.append(f"Validation error for {jf.name}: {exc}")
                all_valid = False

        return all_valid

    # ------------------------------------------------------------------
    # CLI discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _find_cli_command() -> Optional[list[str]]:
        """Locate the ``nr-isaac-format`` CLI, or *None* if not installed."""
        exe = shutil.which("nr-isaac-format")
        if exe:
            return [exe]
        try:
            from nr_isaac_format import cli as _cli  # noqa: F401

            return [sys.executable, "-c", "from nr_isaac_format.cli import main; main()"]
        except ImportError:
            return None

    @staticmethod
    def _find_assembler_command() -> Optional[list[str]]:
        """Locate the ``data-assembler`` CLI, or *None* if not installed."""
        exe = shutil.which("data-assembler")
        if exe:
            return [exe]
        try:
            from assembler.cli.main import main as _m  # noqa: F401

            return [sys.executable, "-c", "from assembler.cli.main import main; main()"]
        except ImportError:
            return None
