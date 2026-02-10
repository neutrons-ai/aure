"""
Checkpoint system for the reflectivity analysis workflow.

This module provides functionality to save and load workflow state checkpoints,
enabling:
- Saving state after each node for debugging and analysis
- Restarting workflows from a specific checkpoint
- Reviewing intermediate results
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)

# Node execution order for reference
NODE_ORDER = ["intake", "analysis", "modeling", "fitting", "evaluation"]


class CheckpointManager:
    """
    Manages workflow checkpoints.
    
    Checkpoints are saved as JSON files in a structured directory:
    
        output_dir/
        ├── run_info.json           # Metadata about the run
        ├── checkpoints/
        │   ├── 001_intake.json
        │   ├── 002_analysis.json
        │   ├── 003_modeling.json
        │   ├── 004_fitting.json
        │   ├── 005_evaluation.json
        │   └── 006_refinement_iter1.json
        ├── models/
        │   ├── model_initial.py
        │   └── model_refined_iter1.py
        └── final_state.json
    """
    
    def __init__(self, output_dir: str, run_id: Optional[str] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Directory for checkpoints and results
            run_id: Optional run identifier. If not provided, uses timestamp.
        """
        self.output_dir = Path(output_dir)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.models_dir = self.output_dir / "models"
        
        self._checkpoint_counter = 0
        self._initialized = False
    
    def initialize(self, initial_state: Dict[str, Any], data_file: str, sample_description: str):
        """
        Initialize the checkpoint directory for a new run.
        
        Args:
            initial_state: Initial workflow state
            data_file: Path to data file
            sample_description: User's sample description
        """
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Save run info
        run_info = {
            "run_id": self.run_id,
            "started_at": datetime.now().isoformat(),
            "data_file": str(data_file),
            "sample_description": sample_description,
            "hypothesis": initial_state.get("hypothesis"),
            "checkpoints": [],
        }
        self._save_json(self.output_dir / "run_info.json", run_info)
        
        self._initialized = True
        logger.info(f"[CHECKPOINT] Initialized checkpoint directory: {self.output_dir}")
    
    def initialize_for_resume(self, state: Dict[str, Any], start_node: str):
        """
        Initialize the checkpoint directory for resuming from a checkpoint.
        
        Args:
            state: State loaded from checkpoint
            start_node: Node to start from
        """
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Load existing run_info or create new one
        run_info_path = self.output_dir / "run_info.json"
        if run_info_path.exists():
            with open(run_info_path) as f:
                run_info = json.load(f)
            # Update checkpoint counter based on existing checkpoints
            self._checkpoint_counter = len(run_info.get("checkpoints", []))
        else:
            run_info = {
                "run_id": self.run_id,
                "started_at": datetime.now().isoformat(),
                "resumed_at": datetime.now().isoformat(),
                "data_file": str(state.get("data_file", "")),
                "sample_description": state.get("sample_description", ""),
                "hypothesis": state.get("hypothesis"),
                "checkpoints": [],
                "resumed_from_node": start_node,
            }
            self._save_json(run_info_path, run_info)
        
        self._initialized = True
        logger.info(f"[CHECKPOINT] Initialized for resume from {start_node}: {self.output_dir}")
    
    def save_checkpoint(self, state: Dict[str, Any], node_name: str) -> str:
        """
        Save a checkpoint after a node completes.
        
        Args:
            state: Current workflow state
            node_name: Name of the node that just completed
            
        Returns:
            Path to saved checkpoint file
        """
        if not self._initialized:
            raise RuntimeError("CheckpointManager not initialized. Call initialize() first.")
        
        self._checkpoint_counter += 1
        iteration = state.get("iteration", 0)
        
        # Create checkpoint filename
        if node_name in ("fitting", "evaluation", "refinement") and iteration > 0:
            filename = f"{self._checkpoint_counter:03d}_{node_name}_iter{iteration}.json"
        else:
            filename = f"{self._checkpoint_counter:03d}_{node_name}.json"
        
        checkpoint_path = self.checkpoints_dir / filename
        
        # Prepare checkpoint data
        checkpoint_data = {
            "checkpoint_id": self._checkpoint_counter,
            "node": node_name,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "state": self._serialize_state(state),
        }
        
        # Save checkpoint
        self._save_json(checkpoint_path, checkpoint_data)
        
        # Update run info
        self._update_run_info(filename, node_name, iteration)
        
        # Save model if present
        if state.get("current_model"):
            self._save_model(state["current_model"], node_name, iteration)
        
        logger.info(f"[CHECKPOINT] Saved: {filename}")
        return str(checkpoint_path)
    
    def save_final_state(self, state: Dict[str, Any]):
        """Save the final workflow state."""
        if not self._initialized:
            return
        
        final_path = self.output_dir / "final_state.json"
        final_data = {
            "completed_at": datetime.now().isoformat(),
            "success": not state.get("error"),
            "error": state.get("error"),
            "iterations": state.get("iteration", 0),
            "final_chi2": state.get("current_chi2"),
            "state": self._serialize_state(state),
        }
        self._save_json(final_path, final_data)
        logger.info(f"[CHECKPOINT] Saved final state: {final_path}")
    
    def _save_model(self, model_script: str, node_name: str, iteration: int):
        """Save model script to models directory."""
        if node_name == "modeling" and iteration == 0:
            filename = "model_initial.py"
        elif node_name == "refinement":
            filename = f"model_refined_iter{iteration}.py"
        else:
            filename = f"model_{node_name}_iter{iteration}.py"
        
        model_path = self.models_dir / filename
        model_path.write_text(model_script)
    
    def _update_run_info(self, checkpoint_file: str, node_name: str, iteration: int):
        """Update run_info.json with new checkpoint."""
        run_info_path = self.output_dir / "run_info.json"
        run_info = json.loads(run_info_path.read_text())
        
        run_info["checkpoints"].append({
            "file": checkpoint_file,
            "node": node_name,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
        })
        run_info["last_updated"] = datetime.now().isoformat()
        
        self._save_json(run_info_path, run_info)
    
    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert state to JSON-serializable format."""
        serialized = {}
        
        for key, value in state.items():
            if value is None:
                serialized[key] = None
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                serialized[key] = [
                    v.tolist() if isinstance(v, np.ndarray) else v 
                    for v in value
                ]
            elif isinstance(value, dict):
                serialized[key] = self._serialize_dict(value)
            elif isinstance(value, (int, float, str, bool)):
                serialized[key] = value
            else:
                # Try to convert, fall back to string representation
                try:
                    serialized[key] = json.loads(json.dumps(value))
                except (TypeError, ValueError):
                    serialized[key] = str(value)
        
        return serialized
    
    def _serialize_dict(self, d: Dict) -> Dict:
        """Recursively serialize a dictionary."""
        result = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                result[k] = v.tolist()
            elif isinstance(v, dict):
                result[k] = self._serialize_dict(v)
            elif isinstance(v, (list, tuple)):
                result[k] = [
                    x.tolist() if isinstance(x, np.ndarray) else x 
                    for x in v
                ]
            else:
                result[k] = v
        return result
    
    def _save_json(self, path: Path, data: Dict):
        """Save data as JSON with pretty formatting."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint JSON file
            
        Returns:
            Checkpoint data including state
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        with open(path) as f:
            checkpoint_data = json.load(f)
        
        # State is already JSON-compatible from loading
        logger.info(f"[CHECKPOINT] Loaded: {checkpoint_path}")
        return checkpoint_data
    
    @classmethod
    def list_checkpoints(cls, output_dir: str) -> List[Dict[str, Any]]:
        """
        List all checkpoints in a run directory.
        
        Args:
            output_dir: Path to output directory
            
        Returns:
            List of checkpoint info dictionaries
        """
        run_info_path = Path(output_dir) / "run_info.json"
        if not run_info_path.exists():
            return []
        
        with open(run_info_path) as f:
            run_info = json.load(f)
        
        return run_info.get("checkpoints", [])
    
    @classmethod
    def get_checkpoint_for_node(cls, output_dir: str, node_name: str, 
                                 iteration: int = 0) -> Optional[str]:
        """
        Find the checkpoint file for a specific node.
        
        Args:
            output_dir: Path to output directory
            node_name: Name of the node (e.g., "fitting", "evaluation")
            iteration: Iteration number (for nodes that run multiple times)
            
        Returns:
            Path to checkpoint file, or None if not found
        """
        checkpoints = cls.list_checkpoints(output_dir)
        
        for cp in checkpoints:
            if cp["node"] == node_name and cp.get("iteration", 0) == iteration:
                return str(Path(output_dir) / "checkpoints" / cp["file"])
        
        return None


def get_restart_state(checkpoint_path: str) -> Dict[str, Any]:
    """
    Get state suitable for restarting workflow from a checkpoint.
    
    This loads the checkpoint and prepares the state for continuing
    from the next node in the workflow.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        State dictionary ready for workflow restart
    """
    checkpoint_data = CheckpointManager.load_checkpoint(checkpoint_path)
    state = checkpoint_data["state"]
    
    # Clear any error state
    state["error"] = None
    
    # The state is ready to continue from where it left off
    # The current_node field indicates where we are
    
    logger.info(f"[CHECKPOINT] Prepared restart state from node: {checkpoint_data['node']}")
    return state


def get_node_after(node_name: str) -> Optional[str]:
    """
    Get the next node in the workflow after the given node.
    
    Args:
        node_name: Current node name
        
    Returns:
        Next node name, or None if at end
    """
    try:
        idx = NODE_ORDER.index(node_name)
        if idx < len(NODE_ORDER) - 1:
            return NODE_ORDER[idx + 1]
    except ValueError:
        pass
    return None
