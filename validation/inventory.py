"""
Validation dataset inventory.

Scans the reference data directory and builds a mapping of
run numbers to data files, reference model files, experiment types,
and sample names.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


DATA_DIR = Path.home() / "git" / "experiments-2024" / "val-sep24" / "data"


@dataclass
class ReferenceModel:
    """A single reference model parsed from the _model.json file."""

    run: str
    sample: str
    experiment: int
    data_file: Path
    model_file: Path
    layers: List[dict] = field(default_factory=list)
    probe: dict = field(default_factory=dict)

    # Convenience accessors ---------------------------------------------------

    @property
    def layer_names(self) -> List[str]:
        return [l["name"] for l in self.layers]

    @property
    def n_layers(self) -> int:
        """Number of layers excluding semi-infinite fronting/backing."""
        return len([l for l in self.layers if not l.get("thickness", {}).get("fixed")])

    @property
    def context_file(self) -> Path:
        return self.data_file.parent / f"context-experiment-{self.experiment}.md"

    @property
    def context(self) -> str:
        return self.context_file.read_text().strip()

    def param_value(self, layer_name: str, param: str) -> Optional[float]:
        """Get the best-fit value for a layer parameter."""
        for layer in self.layers:
            if layer["name"] == layer_name:
                p = layer.get(param, {})
                if isinstance(p, dict):
                    return p.get("value")
                return p
        return None

    def param_bounds(self, layer_name: str, param: str) -> Optional[list]:
        """Get the 95% credible interval for a layer parameter."""
        for layer in self.layers:
            if layer["name"] == layer_name:
                p = layer.get(param, {})
                if isinstance(p, dict):
                    return p.get("p95")
        return None


def load_reference(model_json: Path) -> ReferenceModel:
    """Load a single reference model from a JSON file."""
    with open(model_json) as f:
        data = json.load(f)

    data_filename = data["data_file"]
    data_path = model_json.parent / data_filename

    # Extract run number from filename  REFL_<run>_combined_data_auto_model.json
    run = model_json.stem.split("_")[1]

    return ReferenceModel(
        run=run,
        sample=data["sample"],
        experiment=data["experiment"],
        data_file=data_path,
        model_file=model_json,
        layers=data.get("layers", []),
        probe=data.get("probe", {}),
    )


def build_inventory(data_dir: Path = DATA_DIR) -> Dict[str, ReferenceModel]:
    """
    Scan the data directory and return {run_number: ReferenceModel}.

    Only returns entries where both data file and JSON exist.
    """
    inventory: Dict[str, ReferenceModel] = {}

    for model_json in sorted(data_dir.glob("REFL_*_combined_data_auto_model.json")):
        ref = load_reference(model_json)
        if ref.data_file.exists():
            inventory[ref.run] = ref
        else:
            print(f"  WARNING: data file missing for {model_json.name}")

    return inventory


def print_inventory(inventory: Dict[str, ReferenceModel]) -> None:
    """Print a human-readable summary of the inventory."""
    print(f"{'Run':>8}  {'Sample':>6}  {'Exp':>3}  {'Layers':>6}  Layer names")
    print("-" * 65)
    for run, ref in inventory.items():
        names = " → ".join(ref.layer_names)
        print(f"{run:>8}  {ref.sample:>6}  {ref.experiment:>3}  {ref.n_layers:>6}  {names}")


# ---------------------------------------------------------------------------
# CLI entry point for quick testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    inv = build_inventory()
    print(f"\nFound {len(inv)} validation datasets in {DATA_DIR}\n")
    print_inventory(inv)
