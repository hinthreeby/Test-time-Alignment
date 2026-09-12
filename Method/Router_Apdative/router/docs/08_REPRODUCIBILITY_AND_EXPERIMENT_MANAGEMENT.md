# Stage 8 — Reproducibility, Experiment Tracking, and Final Audit

## Goal

Make every result reproducible from a clean environment and prevent accidental data leakage or configuration drift.

## Repository structure

```text
project/
├── configs/
├── data/
│   ├── router/
│   └── benchmark/
├── cache/router_features/
├── router/
├── router/scripts/
├── router/tests/
├── checkpoints/
├── runs/
├── results/
└── router/docs/
```

## Required configuration files

Create separate version-controlled YAML files for:

- data preparation;
- feature extraction;
- router training;
- fixed-beta sweep;
- heuristic baselines;
- learned-router generation;
- evaluation.

Every generated artifact must record the exact configuration used.

## Run metadata

Each run directory should include:

```text
config.yaml
environment.txt
git_commit.txt
model_metadata.json
dataset_metadata.json
stdout.log
metrics.json
```

Record:

- Python and package versions;
- CUDA/PyTorch versions if applicable;
- GPU/CPU hardware;
- random seeds;
- model names and revisions;
- checkpoint hashes;
- dataset hashes;
- command line;
- Git commit and dirty state.

## Seed policy

Use multiple seeds for final comparisons, such as:

```text
42, 43, 44
```

Use a single seed only for development and debugging.

## Data audit

Before final evaluation:

1. Re-run leakage detection.
2. Verify benchmark files are unchanged.
3. Confirm best fixed beta was selected only from validation.
4. Confirm no benchmark samples entered router training cache.
5. Save a final audit report.

## Model-freezing audit

After training, compare hashes or checksums of base LM and RM parameters before and after training. They must be identical.

## Minimal command chain

The project must document commands equivalent to:

```bash
python router/scripts/prepare_router_data.py --config configs/data.yaml
python router/scripts/validate_models.py --config configs/models.yaml
python router/scripts/cache_router_features.py --config configs/cache.yaml
python router/scripts/train_router.py --config configs/router_train.yaml
python router/scripts/run_fixed_beta_sweep.py --config configs/fixed_sweep.yaml
python router/scripts/run_learned_router.py --config configs/learned_router.yaml
python router/scripts/evaluate.py --config configs/evaluation.yaml
```

## Final reproducibility test

On a clean environment:

- install dependencies;
- execute the documented command chain;
- reproduce validation metrics within tolerance;
- reproduce selected benchmark aggregates within stochastic confidence bounds.

## Acceptance criteria

- Every result links to a config, code commit, dataset hash, and checkpoint hash.
- Base and reward model hashes remain unchanged.
- Final audit reports no leakage.
- Clean-environment reproduction succeeds.
- The README contains one complete end-to-end example.

## Prompt for the coding AI

Create the experiment-management and reproducibility layer for the adaptive RAD router project. Add versioned YAML configs, deterministic seeds, run metadata, model and dataset hashes, Git-state capture, data-leakage audits, frozen-model checksum verification, and a documented clean-environment command chain. Produce a final audit report and fail loudly when provenance is missing.
