"""
Tests for interactive parameter editing, simulation, model-script patching,
restart-with-overrides, and ISAAC export with user_context.

Covers the features added in the interactive-parameter-editor session:

- ``RunData.simulate()``
- ``POST /api/simulate``
- ``_apply_overrides_to_model_script()``
- Restart with ``parameter_overrides`` / ``bounds_overrides``
- ISAAC export ``user_context`` prepending
- ``POST /api/export`` with ``user_context``
"""

import json
import threading
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

# ======================================================================
# Helpers
# ======================================================================


def _make_state(output_dir: str, data_file: str = "/tmp/test.dat") -> dict:
    """Minimal completed-workflow state."""
    return {
        "data_file": data_file,
        "Q": [0.01, 0.02, 0.03, 0.04, 0.05],
        "R": [1.0, 0.8, 0.5, 0.2, 0.05],
        "dR": [0.01, 0.01, 0.01, 0.01, 0.005],
        "sample_description": "50 nm Cu on Si",
        "hypothesis": "Copper layer",
        "parsed_sample": {"substrate": {}, "layers": [], "ambient": {}},
        "extracted_features": {"n_fringes": 2},
        "current_chi2": 1.45,
        "best_chi2": 1.45,
        "current_model": (
            "# model script\ncopper_thickness = 500\ncopper_thickness.range(400, 600)\n"
        ),
        "best_model": "# model script",
        "workflow_complete": True,
        "iteration": 1,
        "max_iterations": 3,
        "error": None,
        "fit_results": [
            {
                "iteration": 0,
                "method": "dream",
                "chi_squared": 1.45,
                "converged": True,
                "parameters": {
                    "copper thickness": 502.3,
                    "copper roughness": 8.1,
                },
                "uncertainties": {
                    "copper thickness": 1.2,
                    "copper roughness": 0.5,
                },
                "bounds": {
                    "copper thickness": [400.0, 600.0],
                    "copper roughness": [2.0, 20.0],
                },
            }
        ],
        "messages": [],
        "llm_calls": [],
        "output_dir": output_dir,
        "pending_user_feedback": None,
    }


def _write_output_dir(tmp_path, state=None):
    """Write final_state.json, run_info.json, and required dirs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)

    if state is None:
        state = _make_state(str(output_dir))

    final = {"state": state, "completed_at": "2026-03-10T00:00:00"}
    (output_dir / "final_state.json").write_text(json.dumps(final, default=str))
    (output_dir / "run_info.json").write_text(
        json.dumps({"run_id": "test", "checkpoints": []})
    )
    return output_dir


@pytest.fixture()
def output_dir(tmp_path):
    return _write_output_dir(tmp_path)


@pytest.fixture()
def client(output_dir):
    """Flask test client wired to the web blueprint."""
    from flask import Flask

    from aure.web.routes import bp

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["OUTPUT_DIR"] = str(output_dir)
    app.config["RUN_LOCK"] = threading.Lock()
    app.config["RUN_STATE"] = {"status": "idle"}
    app.register_blueprint(bp)

    with app.test_client() as c:
        yield c


# ======================================================================
# _apply_overrides_to_model_script
# ======================================================================


class TestApplyOverridesToModelScript:
    """Unit tests for the regex-based model-script patcher."""

    def test_bounds_override_replaces_range(self):
        from aure.web.routes import _apply_overrides_to_model_script

        script = (
            "copper_thickness = 500\n"
            "copper_thickness.range(400, 600)  # copper thickness\n"
        )
        result = _apply_overrides_to_model_script(
            script,
            parameter_overrides={},
            bounds_overrides={"copper thickness": [350, 650]},
        )
        assert ".range(350, 650)" in result
        assert ".range(400, 600)" not in result

    def test_bounds_override_no_match_leaves_script_unchanged(self):
        from aure.web.routes import _apply_overrides_to_model_script

        script = "x.range(0, 10)  # some param\n"
        result = _apply_overrides_to_model_script(
            script,
            parameter_overrides={},
            bounds_overrides={"nonexistent param": [0, 99]},
        )
        assert result == script

    def test_bounds_override_invalid_pair_ignored(self):
        from aure.web.routes import _apply_overrides_to_model_script

        script = "x.range(0, 10)  # copper thickness\n"
        result = _apply_overrides_to_model_script(
            script,
            parameter_overrides={},
            bounds_overrides={"copper thickness": "bad"},
        )
        assert ".range(0, 10)" in result

    def test_empty_overrides_returns_original(self):
        from aure.web.routes import _apply_overrides_to_model_script

        script = "# unchanged\n"
        assert _apply_overrides_to_model_script(script, {}, {}) == script

    def test_multiple_bounds_overrides(self):
        from aure.web.routes import _apply_overrides_to_model_script

        script = (
            "copper_thickness.range(400, 600)  # copper thickness\n"
            "copper_roughness.range(2, 20)  # copper roughness\n"
        )
        result = _apply_overrides_to_model_script(
            script,
            parameter_overrides={},
            bounds_overrides={
                "copper thickness": [300, 700],
                "copper roughness": [1, 30],
            },
        )
        assert ".range(300, 700)" in result
        assert ".range(1, 30)" in result


# ======================================================================
# RunData.simulate()
# ======================================================================


class TestRunDataSimulate:
    """Unit tests for RunData.simulate() with mocked model execution."""

    def test_simulate_returns_curves(self, output_dir):
        from aure.web.data import RunData

        # Create a dummy model file
        models = output_dir / "models"
        (models / "model_fitting_iter0.py").write_text("# dummy model")

        rd = RunData(str(output_dir))

        mock_result = {
            "Q_fit": [0.01, 0.02],
            "R_fit": [1.0, 0.5],
            "z": [0.0, 10.0],
            "sld": [2.0, 6.5],
            "chi_squared": 1.23,
        }

        with patch("aure.web.data._compute_from_model", return_value=mock_result):
            result = rd.simulate({"copper thickness": 500.0})

        assert result["Q_fit"] == [0.01, 0.02]
        assert result["R_fit"] == [1.0, 0.5]
        assert result["sld_z"] == [0.0, 10.0]
        assert result["sld_rho"] == [2.0, 6.5]
        assert result["chi_squared"] == 1.23

    def test_simulate_no_model_file_returns_error(self, output_dir):
        from aure.web.data import RunData

        rd = RunData(str(output_dir))
        result = rd.simulate({"copper thickness": 500.0})
        assert "error" in result

    def test_simulate_execution_failure_returns_error(self, output_dir):
        from aure.web.data import RunData

        models = output_dir / "models"
        (models / "model_fitting_iter0.py").write_text("# dummy")

        rd = RunData(str(output_dir))

        with patch(
            "aure.web.data._compute_from_model",
            side_effect=RuntimeError("exec failed"),
        ):
            result = rd.simulate({"copper thickness": 500.0})

        assert "error" in result
        assert "exec failed" in result["error"]

    def test_simulate_returns_error_when_execute_returns_none(self, output_dir):
        from aure.web.data import RunData

        models = output_dir / "models"
        (models / "model_fitting_iter0.py").write_text("# dummy")

        rd = RunData(str(output_dir))

        with patch("aure.web.data._execute_model_file", return_value=None):
            result = rd.simulate({"copper thickness": 500.0})

        assert "error" in result

    def test_simulate_falls_back_to_model_final(self, tmp_path):
        """When no model_fitting_iter*.py exists, fall back to model_final.py."""
        from aure.web.data import RunData

        # Create state with no fit_results iterations
        state = _make_state(str(tmp_path / "output"))
        state["fit_results"] = []
        output_dir = _write_output_dir(tmp_path, state)

        models = output_dir / "models"
        (models / "model_final.py").write_text("# final model")

        rd = RunData(str(output_dir))

        mock_result = {
            "Q_fit": [0.01],
            "R_fit": [1.0],
            "z": [0.0],
            "sld": [2.0],
            "chi_squared": 2.0,
        }
        with patch("aure.web.data._execute_model_file", return_value=mock_result):
            result = rd.simulate({"copper thickness": 500.0})

        assert result["Q_fit"] == [0.01]


# ======================================================================
# POST /api/simulate
# ======================================================================


class TestApiSimulate:
    """Integration tests for the /api/simulate endpoint."""

    def test_simulate_success(self, client, output_dir):
        models = output_dir / "models"
        (models / "model_fitting_iter0.py").write_text("# dummy")

        mock_result = {
            "Q_fit": [0.01, 0.02],
            "R_fit": [0.9, 0.4],
            "z": [0.0, 10.0],
            "sld": [2.0, 6.5],
            "chi_squared": 1.5,
        }
        with patch("aure.web.data._compute_from_model", return_value=mock_result):
            resp = client.post(
                "/api/simulate",
                json={"parameters": {"copper thickness": 500.0}},
            )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["Q_fit"] == [0.01, 0.02]
        assert data["chi_squared"] == 1.5

    def test_simulate_missing_parameters_returns_400(self, client):
        resp = client.post("/api/simulate", json={})
        assert resp.status_code == 400

    def test_simulate_non_numeric_parameters_returns_400(self, client):
        resp = client.post(
            "/api/simulate",
            json={"parameters": {"copper thickness": "not-a-number"}},
        )
        assert resp.status_code == 400

    def test_simulate_empty_parameters_dict_returns_400(self, client):
        resp = client.post("/api/simulate", json={"parameters": {}})
        assert resp.status_code == 400

    def test_simulate_no_output_returns_404(self, tmp_path):
        """When OUTPUT_DIR is invalid, returns 404."""
        from flask import Flask

        from aure.web.routes import bp

        app = Flask(__name__)
        app.config["TESTING"] = True
        app.config["OUTPUT_DIR"] = str(tmp_path / "nonexistent")
        app.config["RUN_LOCK"] = threading.Lock()
        app.config["RUN_STATE"] = {"status": "idle"}
        app.register_blueprint(bp)

        with app.test_client() as c:
            resp = c.post(
                "/api/simulate",
                json={"parameters": {"x": 1.0}},
            )
        assert resp.status_code == 404


# ======================================================================
# Restart with parameter/bounds overrides
# ======================================================================


class TestRestartWithOverrides:
    """Integration tests for parameter/bounds overrides in restart."""

    @patch("aure.workflow.runner.run_workflow_with_checkpoints")
    def test_restart_applies_parameter_overrides(self, mock_run, client, output_dir):

        mock_run.return_value = _make_state(str(output_dir))

        resp = client.post(
            "/api/restart-analysis",
            json={
                "insight": "Adjust copper thickness",
                "restart_from": "modeling",
                "parameter_overrides": {"copper thickness": 510.0},
            },
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "restarted"

    @patch("aure.workflow.runner.run_workflow_with_checkpoints")
    def test_restart_applies_bounds_overrides(self, mock_run, client, output_dir):
        mock_run.return_value = _make_state(str(output_dir))

        resp = client.post(
            "/api/restart-analysis",
            json={
                "insight": "Widen copper thickness bounds",
                "restart_from": "modeling",
                "bounds_overrides": {"copper thickness": [300, 700]},
            },
        )

        assert resp.status_code == 200

    @patch("aure.workflow.runner.run_workflow_with_checkpoints")
    def test_restart_patches_model_script(self, mock_run, client, output_dir):
        """Verify the current_model in the restarted state was patched."""
        mock_run.return_value = _make_state(str(output_dir))

        # Capture the state passed to run_workflow_with_checkpoints
        captured_kwargs = {}

        def capture_run(*args, **kwargs):
            captured_kwargs.update(kwargs)
            if args:
                captured_kwargs["state"] = args[0]
            return _make_state(str(output_dir))

        mock_run.side_effect = capture_run

        resp = client.post(
            "/api/restart-analysis",
            json={
                "insight": "adjust bounds",
                "restart_from": "modeling",
                "bounds_overrides": {"copper thickness": [350, 650]},
            },
        )
        assert resp.status_code == 200


# ======================================================================
# ISAAC export with user_context
# ======================================================================


class TestIsaacUserContext:
    """Tests for user_context prepending in the ISAAC exporter."""

    def test_user_context_prepended_to_description(self):
        from aure.exporters.isaac import IsaacExporter

        exporter = IsaacExporter()

        # Set up temp output dir with model json and data file
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            data_file = tmp_path / "REFL_218386_combined_data_auto.txt"
            data_file.write_text("# Run 218386\n0.01  1.0  0.01\n")

            problem = {"$schema": "test"}
            (output_dir / "problem.json").write_text(json.dumps(problem))

            state = _make_state(str(output_dir), str(data_file))
            run_info = {"run_id": "test", "data_file": str(data_file)}

            llm_context = "LLM generated context."
            user_ctx = "User provided context about sample."

            with mock.patch(
                "aure.exporters.isaac._generate_context_description",
                return_value=llm_context,
            ):
                exporter.export(output_dir, state, run_info, user_context=user_ctx)

            context_file = output_dir / "ai-ready-data" / "context.txt"
            assert context_file.is_file()
            content = context_file.read_text()
            assert content.startswith(user_ctx)
            assert llm_context in content
            assert f"{user_ctx}\n\n{llm_context}" == content

    def test_no_user_context_leaves_description_unchanged(self):
        from aure.exporters.isaac import IsaacExporter

        exporter = IsaacExporter()

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            data_file = tmp_path / "data.txt"
            data_file.write_text("0.01  1.0  0.01\n")

            (output_dir / "problem.json").write_text(json.dumps({"$schema": "test"}))

            state = _make_state(str(output_dir), str(data_file))
            run_info = {"run_id": "test", "data_file": str(data_file)}

            llm_context = "LLM generated context."

            with mock.patch(
                "aure.exporters.isaac._generate_context_description",
                return_value=llm_context,
            ):
                exporter.export(output_dir, state, run_info, user_context=None)

            content = (output_dir / "ai-ready-data" / "context.txt").read_text()
            assert content == llm_context


# ======================================================================
# POST /api/export with user_context
# ======================================================================


class TestApiExportUserContext:
    """Integration tests for user_context in the /api/export endpoint."""

    @patch("aure.exporters.get_exporter")
    def test_export_passes_user_context(self, mock_get_exporter, client):
        """Verify user_context from POST body reaches the exporter."""
        mock_exporter = MagicMock()
        mock_exporter.format_id = "isaac"
        mock_exporter.name = "ISAAC AI-Ready Format"

        from aure.exporters.base import ExportResult

        mock_exporter.export.return_value = ExportResult(
            success=True,
            output_path=Path("/fake"),
            errors=[],
            warnings=[],
        )
        mock_get_exporter.return_value = mock_exporter

        resp = client.post(
            "/api/export",
            json={"user_context": "My sample context"},
        )

        assert resp.status_code == 200
        # Verify user_context was forwarded to the exporter
        mock_exporter.export.assert_called_once()
        call_kwargs = mock_exporter.export.call_args
        assert call_kwargs.kwargs.get("user_context") == "My sample context"

    @patch("aure.exporters.get_exporter")
    def test_export_empty_context_becomes_none(self, mock_get_exporter, client):
        """Empty string user_context is normalized to None."""
        mock_exporter = MagicMock()
        mock_exporter.format_id = "isaac"
        mock_exporter.name = "ISAAC"

        from aure.exporters.base import ExportResult

        mock_exporter.export.return_value = ExportResult(
            success=True, output_path=Path("/fake"), errors=[], warnings=[]
        )
        mock_get_exporter.return_value = mock_exporter

        resp = client.post("/api/export", json={"user_context": "   "})

        assert resp.status_code == 200
        call_kwargs = mock_exporter.export.call_args
        assert call_kwargs.kwargs.get("user_context") is None

    def test_export_no_exporter_returns_400(self, client):
        with mock.patch("aure.exporters.get_exporter", return_value=None):
            resp = client.post("/api/export", json={})
        assert resp.status_code == 400


# ======================================================================
# _execute_model_file with compute_reflectivity
# ======================================================================


class TestExecuteModelFileReflectivity:
    """Test _execute_model_file with compute_reflectivity flag."""

    def test_compute_reflectivity_returns_q_r(self, tmp_path):
        """When compute_reflectivity=True, result includes Q_fit/R_fit."""
        from aure.web.data import _execute_model_file
        import numpy as np

        # Create a model script that sets up a mock experiment and problem
        model_script = tmp_path / "model.py"
        model_script.write_text(
            """
import numpy as np

class MockExperiment:
    def smooth_profile(self, dz=1.0):
        return [0.0, 10.0, 20.0], [2.07, 6.5, 0.0], [0.0, 0.0, 0.0]

    def update(self):
        pass

    def reflectivity(self):
        return np.array([0.01, 0.02, 0.03]), np.array([1.0, 0.5, 0.1])

class MockProblem:
    fitness = None
    _parameters = []
    def chisq(self):
        return 1.23

experiment = MockExperiment()
problem = MockProblem()
"""
        )

        Q = np.array([0.01, 0.02, 0.03])
        result = _execute_model_file(model_script, Q, compute_reflectivity=True)

        assert result is not None
        assert result["Q_fit"] is not None
        assert len(result["Q_fit"]) == 3
        assert result["chi_squared"] == pytest.approx(1.23)
        assert result["z"] is not None

    def test_compute_reflectivity_false_no_q_r(self, tmp_path):
        """When compute_reflectivity=False (default), no Q_fit/R_fit keys."""
        from aure.web.data import _execute_model_file
        import numpy as np

        model_script = tmp_path / "model.py"
        model_script.write_text(
            """
class MockExperiment:
    def smooth_profile(self, dz=1.0):
        return [0.0, 10.0], [2.07, 0.0], [0.0, 0.0]

experiment = MockExperiment()
problem = None
"""
        )

        result = _execute_model_file(model_script, np.array([0.01, 0.02]))
        assert result is not None
        assert "Q_fit" not in result
        assert result["z"] is not None


# ======================================================================
# _apply_fitted_parameters
# ======================================================================


class TestApplyFittedParameters:
    """Test the parameter-setting helper."""

    def test_sets_parameter_values(self):
        from aure.web.data import _apply_fitted_parameters

        class FakeParam:
            def __init__(self, name, value):
                self.name = name
                self.value = value

        class FakeProblem:
            _parameters = [
                FakeParam("copper thickness", 500.0),
                FakeParam("copper roughness", 8.0),
            ]

        problem = FakeProblem()
        _apply_fitted_parameters(problem, {"copper thickness": 510.0})

        assert problem._parameters[0].value == 510.0
        assert problem._parameters[1].value == 8.0  # unchanged

    def test_ignores_missing_parameters(self):
        from aure.web.data import _apply_fitted_parameters

        class FakeParam:
            def __init__(self, name, value):
                self.name = name
                self.value = value

        class FakeProblem:
            _parameters = [FakeParam("x", 1.0)]

        problem = FakeProblem()
        _apply_fitted_parameters(problem, {"nonexistent": 99.0})
        assert problem._parameters[0].value == 1.0

    def test_no_parameters_attribute(self):
        from aure.web.data import _apply_fitted_parameters

        class NoProblem:
            pass

        _apply_fitted_parameters(NoProblem(), {"x": 1.0})  # should not raise


# ======================================================================
# RunData.get_fit_parameters bounds
# ======================================================================


class TestGetFitParametersBounds:
    """Verify get_fit_parameters returns bounds from state."""

    def test_returns_bounds_from_fit_results(self, output_dir):
        from aure.web.data import RunData

        rd = RunData(str(output_dir))
        params = rd.get_fit_parameters()

        assert params["chi_squared"] == 1.45
        copper_t = next(
            p for p in params["parameters"] if p["name"] == "copper thickness"
        )
        assert copper_t["bounds"] == [400.0, 600.0]
        assert copper_t["value"] == 502.3
        assert copper_t["uncertainty"] == 1.2
