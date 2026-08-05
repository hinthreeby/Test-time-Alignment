# Master Execution Plan

## Objective

Build the first stable, minimal token-level adaptive router for RAD before introducing more advanced optimization.

## Execution order

### Phase 1 — Data

Read and execute:

1. `00_FULL_CONTEXT.md`
2. `01_DATA_PREPARATION.md`

Exit condition:

- clean train/validation files;
- leakage report passes.

### Phase 2 — Model compatibility

Read and execute:

1. `00_FULL_CONTEXT.md`
2. `02_MODEL_AND_TOKENIZER_VALIDATION.md`

Exit condition:

- tokenizer mapping matches;
- base LM and RM are frozen;
- reward direction is verified.

### Phase 3 — Offline cache

Read and execute:

1. `00_FULL_CONTEXT.md`
2. `03_OFFLINE_FEATURE_CACHE.md`

Exit condition:

- token-level cache passes integrity checks;
- gold token is always present;
- no prompt token contributes to loss.

### Phase 4 — Router code

Read and execute:

1. `00_FULL_CONTEXT.md`
2. `04_ROUTER_IMPLEMENTATION.md`

Exit condition:

- MLP is exactly 40 → 64 → 1;
- all unit tests pass.

### Phase 5 — Router training

Read and execute:

1. `00_FULL_CONTEXT.md`
2. `05_ROUTER_TRAINING.md`

Exit condition:

- validation NLL improves;
- best checkpoint saved;
- beta statistics show no silent collapse.

### Phase 6 — RAD integration

Read and execute:

1. `00_FULL_CONTEXT.md`
2. `06_RAD_INFERENCE_INTEGRATION.md`

Exit condition:

- learned-router generation works;
- fixed-beta regression passes;
- beta histories are complete.

### Phase 7 — Evaluation

Read and execute:

1. `00_FULL_CONTEXT.md`
2. `07_EVALUATION_AND_BASELINES.md`

Exit condition:

- all systems evaluated on matched prompts and seeds;
- independent sentiment evaluator used;
- trade-off plots generated.

### Phase 8 — Final reproducibility audit

Read and execute:

1. `00_FULL_CONTEXT.md`
2. `08_REPRODUCIBILITY_AND_EXPERIMENT_MANAGEMENT.md`

Exit condition:

- clean-environment reproduction succeeds;
- no data leakage;
- frozen-model hashes unchanged.

## AI working rule

For each phase, give the coding AI only:

1. `00_FULL_CONTEXT.md`;
2. the current phase file;
3. the relevant repository files;
4. outputs from completed earlier phases.

Do not give all implementation tasks at once. Require the AI to:

- inspect existing code before modifying it;
- provide a file-by-file implementation plan;
- preserve backward compatibility;
- write tests before declaring completion;
- report assumptions and unresolved issues;
- stop when acceptance criteria are not met.

## First actionable task

Start with `01_DATA_PREPARATION.md`. Do not implement the router until the train/validation dataset and leakage audit are complete.
