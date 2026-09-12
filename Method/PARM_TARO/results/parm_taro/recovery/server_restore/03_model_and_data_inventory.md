# Model, data, and protocol inventory

## Models and checkpoints

| Role | Exact ID/path established by retained evidence | Revision / hash | Present | Status |
|---|---|---|---:|---|
| Base LLM | `allenai/tulu-2-7b` / `models/tulu-2-7b` | revision UNKNOWN; archived aggregate `cd816175…` | no | BLOCKING |
| Tokenizer | Tulu-2-7B tokenizer | archived semantic SHA `25cd4342…` | no | BLOCKING |
| PARM PBLoRA | `results/parm_taro/checkpoints/parm_pku_pblora` | tree `f26682e3…`; adapter weights `10127909…` | no | BLOCKING |
| TARO router | `results/parm_taro/training/taro/best.pt` | `32caa71a…` | no | BLOCKING |
| V2 full-alpha | `results/parm_taro/training/v2_alpha_preference/final.pt` | `56fbed68…` | no | BLOCKING |
| V2 no-alpha | `results/parm_taro/training/v2_no_alpha/best.pt` | `ea5cf0e7…` | no | BLOCKING |
| Helpfulness evaluator | `PKU-Alignment/beaver-7b-v1.0-reward` | revision UNKNOWN; tree `3410256f…` | no | BLOCKING FOR STAGE 10 |
| Harmlessness evaluator | `PKU-Alignment/beaver-7b-v1.0-cost` | revision UNKNOWN; tree `418b9416…` | no | BLOCKING FOR STAGE 10 |
| Safe-RLHF evaluator source | `PKU-Alignment/safe-rlhf` | revision UNKNOWN; tree `1b61a3e4…` | no | BLOCKING FOR STAGE 10 |

The retained Stage-10 configuration specifies `load_in_4bit=true`, one physical shared backbone, base pass through `PeftModel.disable_adapter`, and guide pass with PBLoRA enabled. Exact public repository names are known, but exact model revisions are not recorded; no substitute or latest revision was downloaded.

## Dataset

| Item | Path / strategy | Evidence | Status |
|---|---|---|---|
| Original PKU data | `dataset/GenARM/PKU-SafeRLHF-10K/round0/train.jsonl.xz` | 10,000 records; SHA `f5f42f6…` | PASS |
| Train | `dataset/parm_taro/train.json` | 8,000; SHA `9a1ca7c…` | PASS |
| Validation | `dataset/parm_taro/validation.json` | 500; SHA `cdbb9144…` | PASS |
| Test | `dataset/parm_taro/test.json` | 1,500; SHA `892a5796…` | PASS |
| Test prompt-only | `dataset/parm_taro/test_prompt_only.json` | 1,500; SHA `1ab1b59e…` | PASS |
| Validation-200 | `results/parm_taro/recovery/val200/manifest.json` | deterministic lowest SHA-256 under seed label `parm_taro_recovery_val200_seed_2026` | PASS |
| Validation-500 | full validation split | File 13 | PASS |

## Frozen protocol

- Lock: `results/parm_taro/evaluation/protocol/protocol_lock.json`.
- Embedded canonical SHA-256: `81a465048a6b0a1b7a585ff37c2ca42dd67dd5d90a591225c9ae7603bb05faa7` (verified by canonical recomputation).
- Alpha grid: `(0,1), (.25,.75), (.5,.5), (.75,.25), (1,0)` in helpfulness/harmlessness order.
- Seed: `2026`; deterministic generation; 64 new tokens.
- Normalization: validation quantile min-max clip, q01/q99, anchors helpfulness `[-13.500625, 15.42625]`, harmlessness `[-43.375, 18.98453125]`.
- HV reference: `[0,0]`, maximize both objectives.
- Regret pool: all seven outputs for identical prompt/requested-alpha/seed.

Dataset and protocol specifications are present. Execution is blocked by missing models/checkpoints and runtime source.
