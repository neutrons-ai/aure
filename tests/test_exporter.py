"""
Tests for the data exporter system — base registry and ISAAC exporter.
"""

import json
import os
from unittest import mock

import pytest


# ---------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------


class TestExporterRegistry:
    """Tests for ``aure.exporters.get_exporter`` / ``is_export_available``."""

    def test_no_env_returns_none(self):
        from aure.exporters import get_exporter

        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("EXPORT_FORMAT", None)
            assert get_exporter() is None

    def test_empty_env_returns_none(self):
        from aure.exporters import get_exporter

        with mock.patch.dict(os.environ, {"EXPORT_FORMAT": ""}):
            assert get_exporter() is None

    def test_unknown_format_returns_none(self):
        from aure.exporters import get_exporter

        with mock.patch.dict(os.environ, {"EXPORT_FORMAT": "unsupported_xyz"}):
            assert get_exporter() is None

    def test_isaac_format_returns_exporter(self):
        from aure.exporters import get_exporter

        with mock.patch.dict(os.environ, {"EXPORT_FORMAT": "isaac"}):
            exp = get_exporter()
            assert exp is not None
            assert exp.format_id == "isaac"
            assert exp.name == "ISAAC AI-Ready Format"

    def test_is_export_available_true(self):
        from aure.exporters import is_export_available

        with mock.patch.dict(os.environ, {"EXPORT_FORMAT": "isaac"}):
            assert is_export_available() is True

    def test_is_export_available_false(self):
        from aure.exporters import is_export_available

        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("EXPORT_FORMAT", None)
            assert is_export_available() is False


# ---------------------------------------------------------------
# ISAAC exporter tests
# ---------------------------------------------------------------


def _make_state(output_dir: str, data_file: str) -> dict:
    """Build a minimal state dict resembling a completed workflow."""
    return {
        "data_file": data_file,
        "Q": [0.01, 0.02, 0.03, 0.04, 0.05],
        "R": [1.0, 0.8, 0.5, 0.2, 0.05],
        "dR": [0.01, 0.01, 0.01, 0.01, 0.005],
        "sample_description": "50 nm Cu on 5 nm Ti on Si substrate in dTHF",
        "hypothesis": "Copper layer with titanium adhesion layer",
        "parsed_sample": {
            "substrate": {"name": "silicon", "sld": 2.07, "roughness": 5.0},
            "layers": [
                {"name": "Ti", "sld": -1.95, "thickness": 50.0, "roughness": 5.0},
                {"name": "Cu", "sld": 6.55, "thickness": 500.0, "roughness": 8.0},
            ],
            "ambient": {"name": "dTHF", "sld": 6.36},
        },
        "current_chi2": 1.45,
        "best_chi2": 1.45,
        "fit_results": [
            {
                "iteration": 0,
                "method": "dream",
                "chi_squared": 1.45,
                "converged": True,
                "parameters": {
                    "Cu thickness": 502.3,
                    "Cu roughness": 8.1,
                    "Ti thickness": 48.7,
                },
                "uncertainties": {
                    "Cu thickness": 1.2,
                    "Cu roughness": 0.5,
                    "Ti thickness": 0.8,
                },
            }
        ],
        "messages": [
            {
                "role": "system",
                "content": "Data loaded successfully.",
                "timestamp": None,
            },
            {
                "role": "assistant",
                "content": "Building a 2-layer model: Cu/Ti on Si.",
                "timestamp": None,
            },
            {
                "role": "assistant",
                "content": "Fit converged with chi²=1.45.",
                "timestamp": None,
            },
        ],
        "output_dir": output_dir,
        "iteration": 1,
    }


def _make_run_info(data_file: str) -> dict:
    return {
        "run_id": "test_run_123",
        "started_at": "2026-03-07T10:00:00",
        "data_file": data_file,
        "sample_description": "50 nm Cu on 5 nm Ti on Si substrate in dTHF",
        "hypothesis": "Copper layer with titanium adhesion layer",
    }


class TestIsaacExporter:
    """Tests for ``aure.exporters.isaac.IsaacExporter``."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Create a temporary output directory structure."""
        self.output_dir = tmp_path / "output"
        self.output_dir.mkdir()

        # Create a fake reduced data file
        self.data_file = tmp_path / "REFL_218386_combined_data_auto.txt"
        self.data_file.write_text(
            "# Run 218386\n# Q (1/Å)  R  dR\n0.01  1.0  0.01\n0.02  0.8  0.01\n"
        )

        # Create a fake problem.json (best-fit model)
        problem = {"$schema": "bumps-draft-03", "object": {"name": "test"}}
        (self.output_dir / "problem.json").write_text(json.dumps(problem))

        self.state = _make_state(str(self.output_dir), str(self.data_file))
        self.run_info = _make_run_info(str(self.data_file))

    def _export_with_mocks(self, exporter, ingest_ok=True):
        """Run export with the LLM and both subprocess steps mocked out.

        Returns (result, ingest_mock, convert_mock).
        """
        from aure.exporters.isaac import IsaacExporter

        with (
            mock.patch(
                "aure.exporters.isaac._generate_context_description",
                return_value="Test context.",
            ),
            mock.patch.object(
                IsaacExporter,
                "_run_ingest_workflow",
                autospec=True,
                return_value=ingest_ok,
            ) as ingest,
            mock.patch.object(
                IsaacExporter, "_run_convert_ingest", autospec=True, return_value=True
            ) as convert,
            mock.patch.object(
                IsaacExporter, "_run_validate", autospec=True, return_value=True
            ),
        ):
            result = exporter.export(self.output_dir, self.state, self.run_info)
        return result, ingest, convert

    def test_export_creates_ai_ready_dir_and_context(self):
        from aure.exporters.isaac import IsaacExporter

        exporter = IsaacExporter()
        result, _, _ = self._export_with_mocks(exporter)

        ai_dir = self.output_dir / "ai-ready-data"
        assert ai_dir.is_dir()
        assert (ai_dir / "context.txt").read_text() == "Test context."
        assert result.output_path == ai_dir
        assert result.success is True

    def test_export_materializes_run_info(self):
        from aure.exporters.isaac import IsaacExporter

        # The test fixture has no run_info.json on disk; export should write one
        # (from the provided dict) so the assembler can pull it.
        assert not (self.output_dir / "run_info.json").exists()
        exporter = IsaacExporter()
        self._export_with_mocks(exporter)

        ri = self.output_dir / "run_info.json"
        assert ri.is_file()
        assert (
            json.loads(ri.read_text())["sample_description"]
            == self.run_info["sample_description"]
        )

    def test_export_invokes_pull_pipeline(self):
        from aure.exporters.isaac import IsaacExporter

        exporter = IsaacExporter()
        _, ingest, convert = self._export_with_mocks(exporter)

        ingest.assert_called_once()
        convert.assert_called_once()
        # ingest-workflow is pointed at the run dir, writing into ai-ready-data
        _self, run_dir_arg, ai_dir_arg, _warnings = ingest.call_args.args
        assert run_dir_arg == self.output_dir
        assert ai_dir_arg == self.output_dir / "ai-ready-data"

    def test_export_writes_no_manifest(self):
        from aure.exporters.isaac import IsaacExporter

        exporter = IsaacExporter()
        self._export_with_mocks(exporter)
        # The pull pipeline does not author an ISAAC manifest.
        assert not (self.output_dir / "ai-ready-data" / "manifest.yaml").exists()

    def test_export_fails_when_ingest_fails(self):
        from aure.exporters.isaac import IsaacExporter

        exporter = IsaacExporter()
        result, _, convert = self._export_with_mocks(exporter, ingest_ok=False)

        assert result.success is False
        assert any("ingest-workflow" in e for e in result.errors)
        # convert is short-circuited when the ingest step fails
        convert.assert_not_called()

    def test_export_result_has_correct_output_path(self):
        from aure.exporters.isaac import IsaacExporter

        exporter = IsaacExporter()
        result, _, _ = self._export_with_mocks(exporter)
        assert result.output_path == self.output_dir / "ai-ready-data"

    # ---------------------------------------------------------------
    # LLM context generation
    # ---------------------------------------------------------------

    def test_a_vetoed_profile_is_disclosed_in_the_record(self):
        """An exported record outlives the terminal any banner was printed in, and
        is the artifact that gets shared onward — so a rejected profile has to
        travel with it. Independent of the LLM: the no-provider path returns
        `sample_description` verbatim and would otherwise disclose nothing."""
        from aure.exporters.isaac import IsaacExporter

        self.state["fit_results"] = [
            {
                "chi_squared": 1.2,
                "parameters": {"Cu thickness": 495.0},
                "profile_artifact": True,
                "issues": [],
            }
        ]
        self.state["final_selection"] = {
            "selected": True,
            "index": 0,
            "selected_has_profile_artifact": True,
            "profile_veto_reason": "min SLD -0.88 at z=890 Å",
        }

        exporter = IsaacExporter()
        result, _, convert = self._export_with_mocks(exporter)

        context = (self.output_dir / "ai-ready-data" / "context.txt").read_text()
        assert context.startswith("WARNING: the fitted model reported here failed")
        assert "min SLD -0.88 at z=890 Å" in context
        assert "Test context." in context  # the summary itself is preserved
        # Also reaches the caller, and the ISAAC record's own --context.
        assert any("SLD-profile check" in w for w in result.warnings)
        assert "SLD-profile check" in convert.call_args[0][3]

    def test_a_clean_run_is_not_decorated(self):
        from aure.exporters.isaac import IsaacExporter

        exporter = IsaacExporter()
        result, _, _ = self._export_with_mocks(exporter)

        assert (
            self.output_dir / "ai-ready-data" / "context.txt"
        ).read_text() == "Test context."
        assert result.warnings == []


class TestContextGeneration:
    def test_fallback_on_llm_failure(self):
        from aure.exporters.isaac import _generate_context_description

        state = {
            "sample_description": "Test sample on silicon",
            "hypothesis": None,
            "current_chi2": 2.0,
            "fit_results": [],
            "messages": [],
        }

        # Mock LLM to raise an exception
        with mock.patch("aure.llm.get_llm", side_effect=Exception("No API key")):
            result = _generate_context_description(state)

        assert result == "Test sample on silicon"

    def test_uses_llm_response_when_available(self):
        from aure.exporters.isaac import _generate_context_description

        state = {
            "sample_description": "Cu on Si",
            "hypothesis": "Single Cu layer",
            "current_chi2": 1.5,
            "fit_results": [{"parameters": {"Cu thickness": 500}, "chi_squared": 1.5}],
            "messages": [
                {"role": "assistant", "content": "Model built.", "timestamp": None}
            ],
        }

        mock_response = mock.MagicMock()
        mock_response.content = "A neutron reflectometry study of Cu thin film on Si."

        mock_llm = mock.MagicMock()
        with mock.patch("aure.llm.get_llm", return_value=mock_llm):
            with mock.patch(
                "aure.llm.invoke_with_timeout",
                return_value=mock_response,
            ):
                result = _generate_context_description(state)

        assert "Cu" in result
        assert "Si" in result

    def test_context_parameters_come_from_the_selected_fit(self):
        """χ² is taken from the finalize-selected fit, so the parameters must be
        too — reading `fit_results[-1]` paired the selected χ² with another
        iteration's values whenever finalize did not pick the last one."""
        from aure.exporters.isaac import _selected_fit

        state = {
            "final_selection": {"selected": True, "index": 0},
            "fit_results": [
                {"chi_squared": 1.2, "parameters": {"Cu thickness": 495.0}},
                {"chi_squared": 4.7, "parameters": {"Cu thickness": 120.0}},
            ],
        }
        assert _selected_fit(state)["parameters"] == {"Cu thickness": 495.0}

        # No selection record (legacy / imported): fall back to the last fit.
        state["final_selection"] = {}
        assert _selected_fit(state)["parameters"] == {"Cu thickness": 120.0}
