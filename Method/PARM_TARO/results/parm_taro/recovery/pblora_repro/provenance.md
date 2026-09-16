# Historical PBLoRA provenance audit

## Scope and decision

This audit asks which backbone the PBLoRA adapter used by Stage 9/10 was
trained against. It does not claim that the lost checkpoint can be recreated
byte-for-byte: the historical launch command and RNG seed were not retained.

The historical Stage-10 PBLoRA was a two-objective PBLORA adapter attached to
the project's Tulu-2-7B backbone and tokenizer. It was not the unmodified
Alpaca-7B artifact named by the PARM author's example `run.sh`.

## Evidence hierarchy

| Priority | Evidence | Finding |
|---|---|---|
| 1 | `PARM_TARO/reports/stage9_prerequisite_audit.json` | This report inspected the adapter while it still existed. It records `frozen_two_objective_pblora_present=true`, `peft_type=PBLORA`, `obj_num=2`, and `guide_base_model=/home/jupyter-iec2024se10/Reward Decoding/models/tulu-2-7b`. |
| 1 | `results/parm_taro/training/taro/frozen_audit_before.json` | Both base and guide resolve to the same Tulu directory and have identical aggregate checkpoint SHA256 `cd816175990bef4dad02accc8aa73b4c57d31d9e7b85c16da8b00c15ceb7f59e`. The then-present adapter is recorded with aggregate SHA256 `002b6fb7ca107a3c7ca1eb511c6c946f41e4adcf362f5eedf12b09aaeb1d172f`. |
| 1 | Same frozen audit | The lost `adapter_config.json` was 521 bytes, SHA256 `a32ef80e2d487abee84a10cf1d87ea9da2aa15636f4eec61050f35c08f4aa69e`; `adapter_model.safetensors` was 25,233,088 bytes, SHA256 `1012790985d4eeb5229efa75a5cdee124fdf526ff226ddb239d1a2d29018ac93`. These are direct historical observations, not reconstructions. |
| 1 | Historical protected-artifact manifest | The full adapter tree was approximately 52.8 MB and had tree SHA256 `f26682e3cf51f8e09921873119fc371d329669cf9448d59ae0025fd40b23ffef`. |
| 2 | `PARM_TARO/reports/stage9_tokenization_boundary_audit.json` | The Stage-9 tokenizer was `LlamaTokenizerFast`, vocabulary size 32,000, at the same Tulu path; its semantic SHA256 was `25cd4342...`. |
| 2 | `results/parm_taro/pblora_author_runtime/data/{train,dev}.json` | Git-preserved symlink targets point to the project `dataset/parm_taro/{train,validation}.json`, proving that the author trainer was wired to the project split rather than an ad-hoc test split. The old absolute symlink root moved, but its target semantics are unambiguous. |
| 2 | `dataset/parm_taro/manifest.json` | The source is PKU-SafeRLHF-10K. The deterministic prompt-group split is 8,000/500/1,500 with seed label `parm_taro_pku_v1_seed_2026`; train SHA256 is `9a1ca7c8...`, validation SHA256 is `cdbb9144...`. Test is not used by the recovery wrapper. |
| 3 | `models/tulu-2-7b/.tta_artifact_revision.json` and frozen shard hashes | The restored public model is `allenai/tulu-2-7b` revision `3c6e328ae91fabdd0daf09de16887de9615c1f66`. Its content hashes match the Stage-9 frozen model audit, so this is the correct recovered content identity. |

## Author setup versus historical project setup

| Item | PARM author safety example | Historical project evidence / recovery choice |
|---|---|---|
| Base model | `PKU-Alignment/alpaca-7b-reproduced` | **Tulu-2-7B**, established by direct checkpoint-era audits |
| Current public identity | Not applicable | `allenai/tulu-2-7b@3c6e328ae91fabdd0daf09de16887de9615c1f66` |
| Dataset | PKU-SafeRLHF | PKU-SafeRLHF-10K processed into deterministic 8,000 train / 500 validation / 1,500 test |
| PBLoRA type | PBLORA, two objectives | Directly confirmed: `PBLORA`, `obj_num=2` |
| Objective order | Trainer constructs `safe`, then `help` | Project manifest explicitly records author-vector order `[harmlessness, helpfulness]` |
| Rank | `r1=4`, `r2=4` | No retained launch log; recovered from the vendored author launch recipe and consistent with the historical adapter size |
| Target modules | `q_proj`, `v_proj`, `k_proj` | Constructed unconditionally by `train_pref_arm.py`; expected 32 layers x 3 projections |
| LoRA alpha/dropout | alpha 8; dropout defaults to 0.05 | Reproduction uses 8 / 0.05; no surviving adapter config bytes to independently re-read |
| Preference sampling | 0.5 | Reproduction uses author recipe value 0.5 |
| Objective betas | safe 0.01, help 0.01 | Reproduction uses author recipe values |
| DPO/ARM beta | 0.5 | Reproduction uses author recipe value |
| LR / epochs / effective batch | 5e-4 / 2 / 32 | Reproduction uses author recipe values; exact historical launch log is absent |
| Seed | Not passed; Transformers default is normally 42 | **Not historically recorded.** Recovery makes fallback seed 42 explicit and overridable with `PBLORA_SEED`. |

## Exactness boundary

Resolved facts:

- The historical adapter's backbone/tokenizer provenance is Tulu-2-7B.
- Current Tulu files are the same content identity used by the frozen Stage-9
  audit and are pinned to the exact public revision above.
- The historical adapter was two-objective PBLORA and was wired to the project
  PKU train/validation data.
- The vendored trainer always attaches PBLoRA to Q/K/V projections.

Not recoverable from retained artifacts:

- the byte contents of the lost adapter config and weights;
- the exact historical CLI, package runtime state, RNG seed, worker ordering,
  and optimizer trajectory;
- a guarantee of byte-identical retraining.

Accordingly, “resolved” means the model provenance question is answered and a
scientifically labeled reproduction can proceed. Paper/author defaults used in
the recovery wrapper are explicitly marked as fallbacks rather than fabricated
historical facts.

## Reproduction specification prepared (not executed)

The recovery wrapper is
`PARM_TARO/recovery/scripts/train_pblora_repro.sh`. It reuses
`PARM/code/training/train_pref_arm.py` and the vendored PBLoRA implementation;
no protected source is changed. It provides:

- deterministic `--train-subset` and `--max-steps` controls;
- `--mode smoke`, `--mode one-epoch`, and `--mode full`;
- automatic `--resume` from the latest checkpoint containing
  `trainer_state.json`;
- optimizer/scheduler/RNG state through standard Trainer checkpoints;
- checkpoints every 25 steps for smoke and every 50 steps for longer runs;
- atomic `last` and validation-loss `best` pointers without deleting checkpoint
  directories;
- a pre-training trainable-parameter/frozen-base audit and ongoing finite-loss
  and PBLoRA-gradient evidence.

For Tulu-2-7B (32 layers, Q/K/V, `r1=r2=4`, two objectives), the expected
trainable count is **6,296,064 parameters**. This is derived from the vendored
PBLoRA tensor definitions and must still be verified at runtime before the
first optimizer step.

No training, GPU workload, validation generation, or test-set access was
performed during this audit.

PBLORA_PROVENANCE_RESOLVED
