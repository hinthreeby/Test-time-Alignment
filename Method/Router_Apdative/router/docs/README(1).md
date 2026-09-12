# RAD Adaptive Router — AI Implementation Guides

This folder contains phase-by-phase Markdown instructions for implementing a minimal TARO-like token-level adaptive router on RAD.

## Files

- `00_FULL_CONTEXT.md` — complete project context and constraints.
- `01_DATA_PREPARATION.md` — build router train/validation datasets and prevent leakage.
- `02_MODEL_AND_TOKENIZER_VALIDATION.md` — validate GPT-2 Large and RAD reward model compatibility.
- `03_OFFLINE_FEATURE_CACHE.md` — create teacher-forced token-level feature shards.
- `04_ROUTER_IMPLEMENTATION.md` — implement the MLP 40 → 64 → 1 router.
- `05_ROUTER_TRAINING.md` — train with candidate-level NLL and early stopping.
- `06_RAD_INFERENCE_INTEGRATION.md` — integrate learned beta into RAD decoding.
- `07_EVALUATION_AND_BASELINES.md` — compare fixed, heuristic, and learned weighting.
- `08_REPRODUCIBILITY_AND_EXPERIMENT_MANAGEMENT.md` — provenance, seeds, hashes, and final audit.
- `09_MASTER_EXECUTION_PLAN.md` — phase order and stopping criteria.

## Recommended use

For each coding phase, provide the AI with:

1. `00_FULL_CONTEXT.md`;
2. exactly one phase file;
3. the relevant RAD source files;
4. outputs produced by earlier completed phases.

Begin with `01_DATA_PREPARATION.md`.
