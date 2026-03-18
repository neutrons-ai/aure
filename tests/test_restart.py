"""
Tests for the restart-with-new-insight feature.

Covers:
- prepare_state_for_restart (state preparation logic)
- /api/restart-analysis endpoint (via Flask test client)
"""

import json
from unittest.mock import patch

import pytest

from aure.state import create_initial_state, Message
from aure.workflow.runner import prepare_state_for_restart


# ======================================================================
# prepare_state_for_restart
# ======================================================================


def _make_completed_state(**overrides) -> dict:
    """Build a minimal completed-workflow state dict for testing."""
    state = dict(
        create_initial_state(
            data_file="/tmp/test.dat",
            sample_description="Si/SiO2 test sample",
            max_iterations=3,
        )
    )
    state.update(
        {
            "workflow_complete": True,
            "iteration": 2,
            "current_chi2": 3.5,
            "best_chi2": 3.5,
            "current_model": "# refl1d model script",
            "best_model": "# refl1d model script",
            "fit_results": [{"chi_squared": 3.5, "iteration": 2}],
            "parsed_sample": {"substrate": {}, "layers": [], "ambient": {}},
            "extracted_features": {"n_fringes": 2},
            "messages": [
                Message(role="assistant", content="Fit complete.", timestamp=None)
            ],
        }
    )
    state.update(overrides)
    return state


class TestPrepareStateForRestart:
    """Unit tests for prepare_state_for_restart."""

    def test_clears_completion_and_error(self):
        state = _make_completed_state(error="old error")
        result = prepare_state_for_restart(state, "try wider bounds")

        assert result["workflow_complete"] is False
        assert result["error"] is None

    def test_grants_extra_iterations(self):
        state = _make_completed_state(iteration=2)
        result = prepare_state_for_restart(state, "try again", extra_iterations=4)

        # max_iterations should be iteration + extra_iterations = 2 + 4
        assert result["max_iterations"] == 6

    def test_default_extra_iterations(self):
        state = _make_completed_state(iteration=2)
        result = prepare_state_for_restart(state, "try again")

        # default extra_iterations=1 → 2 + 1 = 3
        assert result["max_iterations"] == 3

    def test_injects_user_insight_into_messages(self):
        state = _make_completed_state()
        result = prepare_state_for_restart(state, "Add oxide layer")

        # Last message should contain the insight
        last_msg = result["messages"][-1]
        assert last_msg["role"] == "user"
        assert "Add oxide layer" in last_msg["content"]

    def test_sets_pending_user_feedback(self):
        state = _make_completed_state()
        result = prepare_state_for_restart(state, "Increase roughness")

        assert result["pending_user_feedback"] == "Increase roughness"

    def test_preserves_existing_messages(self):
        state = _make_completed_state()
        original_count = len(state["messages"])
        result = prepare_state_for_restart(state, "hint")

        # Should have original messages + 1 new one
        assert len(result["messages"]) == original_count + 1

    def test_restart_from_analysis_clears_features(self):
        state = _make_completed_state()
        assert state["parsed_sample"] is not None
        assert state["extracted_features"] is not None

        result = prepare_state_for_restart(
            state, "re-examine data", restart_from="analysis"
        )

        assert result["parsed_sample"] is None
        assert result["extracted_features"] is None

    def test_restart_from_modeling_keeps_features(self):
        state = _make_completed_state()
        result = prepare_state_for_restart(
            state, "try different model", restart_from="modeling"
        )

        assert result["parsed_sample"] is not None
        assert result["extracted_features"] is not None

    def test_invalid_restart_from_defaults_to_modeling(self):
        state = _make_completed_state()
        result = prepare_state_for_restart(
            state, "hint", restart_from="nonexistent_node"
        )

        # Should not crash; parsed_sample should remain (modeling behaviour)
        assert result["parsed_sample"] is not None

    def test_does_not_mutate_original_state(self):
        state = _make_completed_state()
        original_complete = state["workflow_complete"]
        _ = prepare_state_for_restart(state, "hint")

        assert state["workflow_complete"] == original_complete


# ======================================================================
# /api/restart-analysis endpoint
# ======================================================================


@pytest.fixture()
def _write_final_state(tmp_path):
    """Write a minimal final_state.json and run_info.json for the test app."""
    state = _make_completed_state()
    final = {"state": state, "completed_at": "2026-01-01T00:00:00"}
    (tmp_path / "final_state.json").write_text(json.dumps(final, default=str))
    (tmp_path / "run_info.json").write_text(
        json.dumps({"run_id": "test", "checkpoints": []})
    )
    (tmp_path / "checkpoints").mkdir(exist_ok=True)
    (tmp_path / "models").mkdir(exist_ok=True)
    return tmp_path


@pytest.fixture()
def client(_write_final_state, tmp_path):
    """Create a Flask test client with the web blueprint registered."""
    from flask import Flask
    import threading

    from aure.web.routes import bp

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["OUTPUT_DIR"] = str(_write_final_state)
    app.config["RUN_LOCK"] = threading.Lock()
    app.config["RUN_STATE"] = {"status": "idle"}
    app.register_blueprint(bp)

    with app.test_client() as c:
        yield c


class TestRestartAnalysisAPI:
    """Integration tests for the /api/restart-analysis route."""

    @patch("aure.workflow.runner.run_workflow_with_checkpoints")
    def test_restart_success(self, mock_run, client):
        """POST with valid insight returns 200 and launches the workflow."""
        # Make run_workflow_with_checkpoints a no-op that completes quickly
        mock_run.return_value = _make_completed_state()

        resp = client.post(
            "/api/restart-analysis",
            json={"insight": "Try adding a buffer layer", "restart_from": "modeling"},
        )
        data = resp.get_json()

        assert resp.status_code == 200
        assert data["status"] == "restarted"
        assert data["restart_from"] == "modeling"

    def test_restart_missing_insight(self, client):
        resp = client.post(
            "/api/restart-analysis",
            json={"insight": "", "restart_from": "modeling"},
        )
        assert resp.status_code == 400

    def test_restart_invalid_restart_from(self, client):
        resp = client.post(
            "/api/restart-analysis",
            json={"insight": "hint", "restart_from": "invalid_node"},
        )
        assert resp.status_code == 400

    @patch("aure.workflow.runner.run_workflow_with_checkpoints")
    def test_restart_blocked_while_running(self, mock_run, client):
        """Cannot restart while another analysis is running."""
        # Simulate a running state
        with client.application.app_context():
            lock = client.application.config["RUN_LOCK"]
            with lock:
                client.application.config["RUN_STATE"]["status"] = "running"

        resp = client.post(
            "/api/restart-analysis",
            json={"insight": "hint", "restart_from": "modeling"},
        )
        assert resp.status_code == 409

        # Restore for cleanup
        with client.application.app_context():
            lock = client.application.config["RUN_LOCK"]
            with lock:
                client.application.config["RUN_STATE"]["status"] = "idle"
