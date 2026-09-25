"""Prepare the matched Phase-C experiment; execution is deliberately gated."""
from __future__ import annotations
import json
from .config import ADAPTER, OUTPUT_ROOT, SEED, TINY_OUT
from .io import atomic_json, atomic_text

MAPPING_JUSTIFICATION = """# Alpha-to-Tchebycheff mapping

For objective-loss vector `L=(L_help,L_safe)` and the user's preference
`alpha=(alpha_help,alpha_safe)`, the probe uses the identity mapping `w=alpha`:

`smoothmax_i [w_i (L_i-z_i)/s_i] + rho sum_i w_i (L_i-z_i)/s_i`.

The reference `z` and positive scales `s` must be frozen from training-only
pilot batches before either arm starts. Identity is chosen because it preserves
the semantics and endpoints of the original PARM preference, maps 0.5 to equal
importance, is permutation-equivariant, and adds no tuned preference warp.
The smooth maximum uses log-sum-exp temperature 20 and augmentation rho=0.05.
These are probe constants shared across all steps, not validation-tuned values.
"""

def prepare() -> dict:
    common = {"starting_checkpoint": str(ADAPTER), "seed": SEED, "steps": 100, "optional_max_steps": 200,
              "optimizer": "paged_adamw_32bit", "learning_rate": 0.0005, "data_order": "sha256(seed,sample_id)",
              "trainable_parameters": "pblora_* only", "status": "PREPARED_NOT_RUN"}
    for name, scalarization in (("linear_control", "linear"), ("nonlinear_probe", "smooth_augmented_tchebycheff")):
        directory = TINY_OUT / name; directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "experiment_config.json", {**common, "experiment": name, "scalarization": scalarization})
    atomic_text(TINY_OUT / "SCALARIZATION_MAPPING.md", MAPPING_JUSTIFICATION)
    result = {"status": "PREPARED_NOT_RUN", "automatic_execution": False, "requires_decision": "HYPOTHESIS_SUPPORTED_FOR_TINY_TRAIN",
              "matched_fields": ["starting_checkpoint", "data_order", "batches", "seed", "optimizer", "learning_rate", "steps"]}
    atomic_json(TINY_OUT / "preparation.json", result); return result

def assert_authorized() -> None:
    decision_path = OUTPUT_ROOT / "decision.json"
    if not decision_path.is_file() or json.loads(decision_path.read_text()).get("decision") != "HYPOTHESIS_SUPPORTED_FOR_TINY_TRAIN":
        raise RuntimeError("Phase C blocked: Phase A+B did not support tiny training")
    raise RuntimeError("Phase C is prepared but requires an explicit user execution request; it is never launched by the A+B command")
