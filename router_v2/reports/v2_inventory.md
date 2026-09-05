# Router V2 Phase 0 - Baseline and Workspace Inventory

Inventory time: `2026-08-14` (UTC)

Workspace: `/home/jupyter-iec2024se10/Reward Decoding`

Scope: baseline hash and read-only inventory only. No Router V2 or TARO code was implemented, no model/data was downloaded, and no V1/PARM artifact was overwritten.

## 1. Decision summary

- **PKU local data: usable without download.** `train.jsonl.xz` has exactly 10,000 valid rows, all seven required semantic fields are present in every row, and there are no empty prompt/response/label fields.
- **PKU is not directly file-compatible with author PARM scripts.** The local source is compressed JSONL, while PARM training expects JSON files at `PARM/code/data/train.json` and `PARM/code/data/dev.json`. A deterministic conversion/split must be written later to a new path such as `dataset/parm_taro/`, never into `PARM/`.
- **RAD inputs and primary local checkpoints are present.** GPT-2 Large, GPT-2 Small, RAD sentiment RM, V1 router checkpoint, and the independent sentiment evaluator all exist and have matching hashes.
- **Do not start full TARO yet.** The only schema-v2 feature cache is a 20-train/20-validation smoke cache under `/tmp`; the persistent `dataset/router_cache/rad/` cache is a different legacy schema. GPU availability must be verified on the user-run host, not inferred from this coding sandbox.
- **Do not start PARM-TARO yet.** The exact author base model, Beaver reward/cost models, trained PARM adapter, `safe-rlhf`, and a configured PARM environment are missing.

## 2. Created namespaces

The requested isolated paths now exist:

```text
router_v2/
router_v2/reports/
dataset/router_v2_cache/
PARM_TARO/
results/router_v2/
results/parm_taro/
```

Only these two files were added:

```text
router_v2/reports/baseline_hashes.json
router_v2/reports/v2_inventory.md
```

All other new namespaces are empty. `dataset/router_v2_train/` and `dataset/parm_taro/` were not created because this phase does not preprocess or train data.

## 3. Git and baseline protection

Current Git HEAD:

```text
55a7678144f408d0e0d318b3a66848ef60eca0f0
```

The worktree was already dirty before this inventory. Many old tracked paths are recorded as deleted while current `router/`, `PARM/`, and documents are largely untracked. Therefore Git HEAD alone is not a valid V1/PARM baseline. The authoritative baseline is the current on-disk content recorded in `router_v2/reports/baseline_hashes.json`.

Tree hash definition: SHA-256 of the sorted sequence
`relative_path<TAB>bytes<TAB>file_sha256<LF>` for every regular file in the root.

| Read-only root | Files | Bytes | Tree SHA-256 |
|---|---:|---:|---|
| `router/` | 100 | 521,788,719 | `7b5ceafc493c605eb13346c557ad8f768399fcecfe3dceb6ee1137dbf518b127` |
| `router/evaluation/` | 32 | 520,908,467 | `1af858c224e2d196af925e047b97f8ec1392db7289670ce97dbea2c0e7482fd9` |
| `PARM/` | 156 | 1,506,962 | `aef1aa5fbc30816a97c1c1bb82d7dbc6554138f8e1a267630b6498d14a02289e` |
| `Method/RAD/` | 110 | 114,269,650 | `3012ada626eabd9dc72dfde349cd4ffc7d0041ec0404cb796e253974b2f5de31` |
| `dataset/RAD_train/amazon_polarity/` | 14 | 1,863,162,461 | `8e2baf28258d48e4a22998fcf8fc16c83213a82da3f8923dacb9812137506c64` |
| `dataset/RAD_train/router_amazon_polarity/` | 5 | 33,903,172 | `8c7f376f8eb06ed920affcd98abb9773832ab1d6239ede8b869ab9a422cfc3ee` |
| `dataset/router_cache/rad/` | 2 | 57,134,618 | `7922216808b1e4614b982d522b7ac73f0c85c1eaf1006484e09d01f03909a36d` |
| `dataset/router_train/rad/` | 2 | 509,313 | `665bef1b2949c45a1ac6d02889ae861723ac16a3e008c809706991c45e4b2ab6` |
| `dataset/rad_benchmark/` | 4 | 8,717,638 | `46ed6143daa76fc66a5f6f06e1e3901dde36c4c283698d33e3517accca3880a6` |

Critical source hashes are stored individually in `baseline_hashes.json`. Key examples:

| File | SHA-256 |
|---|---|
| `router/model.py` | `9e12b46dd890893e0af0c4946b190c20caa83234b85932bd2adbb6dfc59dc284` |
| `router/features.py` | `34d9f43e2e86af07b01484ce0603b0b405610c92a04c260cf57203a873e541ea` |
| `router/training.py` | `2e45fda484c4c16016ade090d34de1cc60c1cdb05ed8ab43a8a41d8a0c7f4cd4` |
| `router/scripts/extract_router_features.py` | `5a0f98d3c6b5247b3601cc3ec741a3d52fd3ecee6f649a2d5542fee1a8cd1b32` |
| `PARM/code/data/relabel.py` | `cf9504e5df853529e73a2405568ef56a916acd34e8f14540314c1a0320c00363` |
| `PARM/code/training/train_pref_arm.py` | `5dec6057203dd9a5a6f728355f303bb8e5b39fadfc8a6c5dab5952870d1f147c` |
| `PARM/code/training/pref_arm_trainer.py` | `d7be821e41d7d389484904abafe55df09ab8f955449b74c2416302311aab919d` |
| `PARM/code/evaluation/generate_outputs.py` | `c990a43c66403fd4e3c0bcdf96b5690e569ad3f3c8920d297b6e7bfcff2c3a41` |

## 4. Router V1 implementation and results

V1 model and feature contract:

- Class: `RADTokenRouter` in `router/model.py`.
- Feature: normalized top-20 base logits concatenated with normalized top-20 RAD reward scores, dimension 40.
- Network: `Linear(40,64) -> Tanh -> Linear(64,1)`.
- Output: `beta = beta_max * sigmoid(logit)`; current evaluation uses `beta_max=30`.
- Training entrypoint: `router/scripts/train_router.py`, delegating to `router/training.py`.
- Feature extraction: `router/scripts/extract_router_features.py`, schema version 2.
- RAD inference integration: `router/rad/router_integration.py` and `router/scripts/run_adaptive_rad.py`.

V1 checkpoint currently exists outside the workspace:

| Artifact | Bytes | SHA-256 / status |
|---|---:|---|
| `/tmp/router_train_smoke/best.pt` | 50,097 | `bace99153fc4dc3183b36085c292dc532a570a224dd8f5885ff906a07bd1c937` |
| `/tmp/router_train_smoke/last.pt` | 50,097 | present |
| `/tmp/router_train_smoke/training_log.json` | 71,044 | present |
| `/tmp/router_train_smoke/` complete tree | 171,238 | `74ee8585ec64ff4386495e66a002ea0d00cd40bd5899a212c290ba2735827849` |

The best-checkpoint hash exactly matches the hash embedded in the completed V1 evaluation report. It is nevertheless volatile because it is stored under `/tmp`.

Checkpoint limitations:

- It was trained on only 200 train token-steps and 238 validation token-steps from 20 samples per split.
- Training ran 31 epochs on CPU and reported best validation NLL `3.2312343861876416`.
- This is explicitly a smoke-scale checkpoint, not a full-cache trained checkpoint.

V1 evaluation inventory:

| Output | Status | Main count/result |
|---|---|---|
| `router/evaluation/pilot/` | complete | 300 output records, 0 failed generations, 2-token generation |
| `router/evaluation/full_32tokens/` | config only | no result tables in this directory |
| `router/evaluation/report_run_32tokens/` | complete | 8,400 output records, 0 failed generations, 32-token generation |

The completed 32-token report selected fixed beta `30.0` on validation and concluded that learned-router performance was `statistically_indistinguishable_from_best_fixed_beta`. Its evaluator and checkpoint hashes still match current local artifacts.

The report records `missing_values=8400` despite zero failed generations. This should be audited metric-by-metric before claiming improvements, but no V1 result was changed in this phase.

## 5. RAD data inventory

### 5.1 Amazon Polarity source

`dataset/RAD_train/amazon_polarity/` is a Hugging Face saved Arrow dataset.

| Split | Rows | Schema |
|---|---:|---|
| train | 3,600,000 | `label: ClassLabel[negative,positive]`, `title: string`, `content: string` |
| test | 400,000 | same |

The five main Arrow shards were re-hashed and all match `data_report.json`:

| File | SHA-256 |
|---|---|
| `train/data-00000-of-00004.arrow` | `3b326b7cddeeaeb8d9229bde1b4e69ffcba6aa939ca2401d9daf2ca50c346410` |
| `train/data-00001-of-00004.arrow` | `a5bc080dd3bd892043676ae696d8104f1ca6fa9dc79fd63fd4d5e716d2de5b38` |
| `train/data-00002-of-00004.arrow` | `22e44d31082ef20d8c52c2b18478af915f18285b70b2e71edf424dbefcb73781` |
| `train/data-00003-of-00004.arrow` | `c16746db5f3fe56ea415cc86eaaa7257150ce61dafa706c0cf17345a66d47966` |
| `test/data-00000-of-00001.arrow` | `404683c77cd9330d5e7c60dd28e4ceaca435bdb72e082182b93dc3aab8226e92` |

### 5.2 Processed router examples

Schema for train/validation/dev JSONL:

```text
continuation: string
id: string
label: int
prompt: string
source: string
source_index: int
source_split: string
```

| File | Rows | SHA-256 |
|---|---:|---|
| `dataset/RAD_train/router_amazon_polarity/train.jsonl` | 20,000 | `92486a32e7c451422682afdd3d0c55a47a3016cc977542cb600a6172937c924b` |
| `dataset/RAD_train/router_amazon_polarity/validation.jsonl` | 2,000 | `70ee7d2fa35e311f4e0fa4a493564a9188313586668d2986c4e9038eaccac3fd` |
| `dataset/RAD_train/router_amazon_polarity/dev.jsonl` | 500 | `da38aa8f9760f355d605b0774bfa043692bc4afd1ad30e39bd15b417c4e0a691` |
| `dataset/RAD_train/router_amazon_polarity/manifest.jsonl` | 22,500 | `130d7c9b23f2a9793d3daa1edca8d9c358b7641d652bd7135c3c3e3cea617091` |

All rows parsed as valid JSON. No benchmark leakage was reported by the existing preprocessing report.

### 5.3 Persistent legacy V1 train/cache

| File | Records | SHA-256 |
|---|---:|---|
| `dataset/router_train/rad/train.jsonl` | 1,000 | `8a806ece54ecab73e4b48fe81b7673eeca357f085689683219097b7e30a096dc` |
| `dataset/router_train/rad/validation.jsonl` | 200 | `d1226cdd2c5ef9f14acbd5b1d671ab928e1b14d8e3a23ba7251cb5e7d0711b72` |
| `dataset/router_cache/rad/train.pt` | 46,712 token records | `2e42ff44efff6edc6de41b7f938dc3fb0e4d362e88b16924ad7359868cc925b5` |
| `dataset/router_cache/rad/validation.pt` | 9,184 token records | `a08ff632abfd7ce1473efe8cfc8f129d7a088de43f65e1c04bdd0f09574481dd` |

Legacy `.pt` records have:

```text
base_logits: Float16[20]
signal_scores: Float32[20]
candidate_ids: Int64[20]
gold_token_id: int
gold_index: int
position: int
sample_id: int
```

This legacy cache was generated by the older tracked `Method/Router/core/cache.py` contract. It is **not directly compatible** with current `router/training.py`, which expects schema-v2 manifests and shard fields including `rad_reward_scores`.

### 5.4 Modern V1 smoke feature cache

The only schema-v2 cache is `/tmp/router_feature_trainval20_short/`:

| Split | Samples | Token records | Manifest SHA-256 |
|---|---:|---:|---|
| train | 20 | 200 | `66a3e5890c15b04b51a6f1a5254a820f485294e4f078b1a21c764841b7db2b39` |
| validation | 20 | 238 | `6e368a765956bc8296b5e8033baff5c22c5f381a657d0f5a00f88e83bbf01082` |

The seven-file cache tree hash is `589ca89040b2d86d53290c5da25a177fcac0cc7ba0f5855424f13ad2b7f78b06`.

Schema-v2 shard fields are:

```text
schema_version, split, shard_index, top_k, num_records,
reward_transform_name, sample_ids, position, candidate_ids,
base_logits, raw_reward_scores, rad_reward_scores, gold_token_id,
gold_index, gold_was_in_topk, attention_length,
reward_effective_length, reward_transform_name_per_record,
original_gold_rank, base_entropy, reward_range
```

This cache is valid and its shard integrity samples passed, but it is smoke-scale and volatile. It must not be treated as the full V2 training cache.

### 5.5 RAD benchmarks

| File | Rows | SHA-256 |
|---|---:|---|
| `negative_prompts.jsonl` | 2,500 | `b82729cf20e3ee339b2b80e62adab365676010044ad29ad722ecb3ceb5a86950` |
| `neutral_prompts.jsonl` | 5,000 | `ebb612696e4e5336c978f8e50f874f80acd8a298e2c1f1b0194a4a889dffb0c8` |
| `positive_prompts.jsonl` | 2,500 | `e53c7cdebe1056ea7780687ffbc64a8861cd29a35247cf9b28c31b4309e5cb79` |
| `nontoxic_prompts-10k.jsonl` | 10,000 | `4f2e60805528e5fbf8b3484c1eeb89b1daead5c75cf1b4fd4abf80e9935e035b` |

## 6. PKU-SafeRLHF local inventory

Path: `dataset/GenARM/PKU-SafeRLHF-10K/round0/train.jsonl.xz`

| Property | Value |
|---|---|
| rows | 10,000 |
| compressed bytes | 1,678,716 |
| invalid JSON | 0 |
| SHA-256 | `f5f42f6f08a1fadbc3fd7c36e627f651782ce0f2f8178361483defa62b79ca06` |
| missing/empty required fields | 0 |

Schema, present in all 10,000 rows:

```text
prompt: string
response_0: string
response_1: string
is_response_0_safe: bool
is_response_1_safe: bool
better_response_id: int
safer_response_id: int
```

Label distribution:

| Field | Value 0 / false | Value 1 / true |
|---|---:|---:|
| `better_response_id` | 5,451 | 4,549 |
| `safer_response_id` | 5,206 | 4,794 |
| `is_response_0_safe` | 5,743 false | 4,257 true |
| `is_response_1_safe` | 5,773 false | 4,227 true |

Compatibility conclusion:

- The local copy is semantically sufficient for preference-aware helpfulness/safety training; no PKU download is needed.
- `PARM/code/training/train_pref_arm.py` only consumes prompt, two responses, `better_response_id`, and `safer_response_id`. All are already local.
- A format/split conversion is still required because author code hard-codes `../data/train.json` and `../data/dev.json`.
- Author `relabel.py` additionally generates Beaver help/harm scores, but the training script does not consume those numeric scores. It can be bypassed if official local labels are accepted.
- Strict reproduction of author relabeling still requires the missing Beaver reward/cost checkpoints and `safe_rlhf` package.

## 7. HH-RLHF candidate data

| File | Rows | Schema | SHA-256 |
|---|---:|---|---|
| `full-hh-rlhf/data/train-00000-of-00001-8349d0765e6718df.parquet` | 112,052 | `prompt,response,chosen,rejected: string` | `fd464e8ef54c2c4937904265c3a3e7638abb34481af8237754a781dadb6188b7` |
| `full-hh-rlhf/data/test-00000-of-00001-ec71e9262143a91c.parquet` | 12,451 | same | `903da5d09608e9dd4eeeba1e34164bd2110f2c9f2bdff4f1f55a05783b6a24b2` |

This is directly useful as a chosen/rejected helpfulness preference source, but it does not carry separate helpfulness and harmlessness objective labels, so it is not a direct replacement for the PKU multi-objective schema.

## 8. PARM author-code inventory

Author code remains untouched under `PARM/`.

Expected author workflow:

- Preprocessing: `PARM/code/data/relabel.py`.
- Expected generated files: `all.json`, `train.json` (first 8,000 shuffled rows), `dev.json` (500), `test.json` (remaining 1,500), and `test_prompt_only.json`.
- Training: `PARM/code/training/run.sh` -> `train_pref_arm.py` -> `PrefARMTrainer`.
- Training base ID: `PKU-Alignment/alpaca-7b-reproduced`.
- Preference dataset key: `PKU_SafeRLHF`.
- Objective order in training: `safe`, then `help`.
- Default run: PBLora `r1=4`, `r2=4`, alpha 8, beta safe/help `1e-2`, 2 epochs, global batch 32, bf16, 8 GPUs.
- Output expected at `PARM/code/training/exp/` and `exp/final_checkpoint/`.
- Generation: `PARM/code/evaluation/generate_outputs.py`, reading `../data/test_prompt_only.json` and a trained adapter containing `adapter_config.json` plus `adapter_model.safetensors`.
- Evaluation model IDs: `PKU-Alignment/beaver-7b-v1.0-reward` and `PKU-Alignment/beaver-7b-v1.0-cost`.

Current author-code artifacts:

- No processed `train.json`, `dev.json`, `test.json`, or `test_prompt_only.json` exists under `PARM/code/data/`.
- No trained PARM/PBLora adapter or checkpoint exists under `PARM/` or `models/`.
- The vendored `PARM/peft/` contains PBLora support, but it is not installed in any confirmed PARM environment.
- `PARM/language-model-arithmetic/` is present; the `genarm` conda environment has `model-arithmetic==1.1.0`.

Author-code reproducibility caveat: `relabel.py` calls `random.shuffle` without setting a seed. A new deterministic PARM-TARO preprocessing path must explicitly set and record its split seed.

## 9. Local model/checkpoint inventory

`models/` uses about 52 GB of workspace disk plus external Hugging Face blob targets reached through symlinks. All listed symlink targets currently resolve, but those targets are outside the workspace and are not protected by this repository.

### 9.1 RAD/V1-ready models

| Role | Path | Status | Primary weight SHA-256 |
|---|---|---|---|
| base LM | `models/gpt2-large/` | complete, local | `5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce` |
| reward base | `models/gpt2-small/` | complete, symlink resolves | `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707` |
| RAD sentiment RM | `models/rad_rm_sentiment/` | complete, local | `9fafd202d3a322f77ce03e682bec9a9443b19bb24f461003268966ddaa2fd298` |
| independent sentiment evaluator | `models/sentiment-roberta-large-english/` | complete, local | `805688de73481b1175b1e06f05a971144dceb7e0064b64a84b78a3685eb3209d` |
| alternative SST-2 evaluator | `models/sentiment-rm-sst2/` | complete, symlink resolves | safetensors blob `7c3919835e442510166d267fe7cbe847e0c51cd26d9ba07b89a57b952b49b8aa` |

The first four primary weight hashes were calculated directly from the current files. The SST-2 value is its resolved content-addressed Hugging Face blob name.

### 9.2 Multi-objective/GenARM candidates

| Path | Status | Checkpoint identity/hash |
|---|---|---|
| `models/tulu-2-7b/` | complete; 2 external symlink shards | `6d90e5350a50e3a1ae608eadf4de08d5336b387561c44094e558789c0e1480d6`, `3eb1b1833fa5b5b8bc9f0ffde237012fead0c18536047fa4c2fcfb8abc3c6425` |
| `models/AutoregressiveRM-tulu2-7b/` | complete; 3 external symlink shards | `7bfdce63b299a66b48481c32a5e7ccf7a00a1b057572b44315167dce6ec371a8`, `c012ad707ece4471c157ea341c4a8eee8f5ca70d8ff0023c18c4901d2f83dd92`, `56ba8eb4d645769f5684dd7f4c47a93786d65bfcc94506d70badfac1d84d47d0` |
| `models/genarm-tulu2-hh/` | complete LoRA adapter | `e5121712e702a36cccde1c085fd50fea0807f0d3dd14dc5ac46c15460266714c` |
| `models/gpt2-medium/` | complete base, multiple formats | safetensors metadata hash `fc5a354a19255ad494f3d71549390baca1ccf61d1d822b9408971705c687c9cd` |
| `models/genarm-gpt2-medium-hh/` | complete merged model | `ebd116adc8b5d7d5e30861f7f5886ce2197261067803295326e4b75481b91b19` |
| `models/genarm-gpt2-medium-hh-adapter/` | complete LoRA adapter | `0ffeefa2b63d69e01859f21fa94a1904fd393f2249b004501e20dcb9016b2fc3` |
| `models/genarm-gpt2-medium-sentiment-adapter/` | empty directory | unusable |

The Tulu and AutoregressiveRM values are full 64-hex content-addressed Hugging Face blob names from resolved symlink targets.

### 9.3 Base/evaluator candidates

| Path | Status | Primary weight hash/source |
|---|---|---|
| `models/llama-2-7b-hf/` | complete in both safetensors and PyTorch formats | safetensors shards `4ec71fd53e99766de38f24753b30c9e8942630e9e576a1ba27b0ec531e87be41`, `41780b5dac322ac35598737e99208d90bdc632a1ba3389ebedbb46a1d8385a7f` |
| `models/Llama-2-7b-hf/` | docs/license only | unusable duplicate with different case |
| `models/gpt2-large-helpful-rm/` | complete helpful reward classifier | `7e79f03e20870598bf264a18f7453c97e991f3538576b6cfb78aee52cedd44ef` |
| `models/helpfulness-deberta-v3-large/` | complete helpfulness evaluator | `bb3306045ebdcf84e3dbd51a55338b6ed255a832cfe179bbd65051b282c5e244` |
| `models/toxic-bert/` | complete toxicity evaluator | `2c272885d24138df70bff1b3cd944a999bd6b41dad33209730aa8ba074f6ad09` |
| `models/cd_prefix_scorer/` | complete CD scorer, not PARM | backbone `0222ca428f2af8c36b2afb34712b11a03f433be060a16b65a6be098ece4635b2` |

No local directory matches the exact PARM author base ID `PKU-Alignment/alpaca-7b-reproduced`, the two Beaver evaluator IDs, or a trained PARM adapter. Local Llama-2/Tulu models are possible adaptation candidates but cannot be called an author-baseline reproduction without code/config changes and a new baseline definition.

## 10. Runtime inventory

Current base shell:

```text
Python 3.13.5
torch 2.11.0
transformers 5.5.3
datasets 5.0.1
```

Importing base-environment `torch` did not complete within 30 seconds. The existing `genarm` environment is the closest reusable environment:

```text
Python 3.10.20
torch 2.2.2+cu121
transformers 4.39.3
datasets 2.18.0
trl 0.9.6
peft 0.10.0
accelerate 0.29.2
bitsandbytes 0.43.1
model-arithmetic 1.1.0
safe-rlhf: missing
```

`torch` imports successfully in `genarm`, but this coding sandbox reports `torch.cuda.is_available() == False`; sandbox `nvidia-smi` also cannot reach the driver. These observations only mean that GPU execution cannot be verified from this environment. They do **not** imply that the target machine is CPU-only or that its driver is broken.

User-confirmed target runtime: NVIDIA RTX 3080 10 GB with working CUDA when commands are run directly on the host. Future code must expose `device=auto|cuda|cpu`, choose CUDA for `auto` when available, and fall back safely to CPU. GPU stages that cannot be executed in this sandbox must be handed off with exact host commands for GPU verification, execution, monitoring, and result inspection.

There is no dedicated `parm` conda environment. The local vendored PBLora-enabled PEFT is not the same as the installed `peft==0.10.0` unless explicitly installed or placed first on `PYTHONPATH` in a new isolated environment.

## 11. Missing prerequisites

### Before TARO baseline on RAD

1. Run a host-side CUDA preflight before the full stage. The target RTX 3080/CUDA runtime is user-confirmed; sandbox CUDA visibility is not a blocker and no driver repair is inferred.
2. Select and freeze a Python environment. `genarm` is the closest match to V1 dependencies; the base Python 3.13 environment should not be used while `torch` import hangs.
3. Preserve the volatile V1 checkpoint and its schema-v2 smoke manifests outside `/tmp`, under a new V2/results namespace, while retaining their recorded hashes. Do not move or edit the originals.
4. Build a persistent V2 feature cache under `dataset/router_v2_cache/` from the existing 20,000/2,000 processed RAD splits. Do not reuse the legacy `.pt` files as though they were schema-v2 shards.
5. Define a V2 cache manifest that pins model/tokenizer hashes, reward transform, top-k, source-data hashes, split IDs, and leakage checks before training.
6. Decide whether TARO baseline training is smoke-scale for implementation validation or full-scale for scientific comparison. The current V1 learned checkpoint is smoke-scale, which must remain explicit in any comparison.

### Before PARM-TARO

1. Create deterministic processed PKU splits in a new data namespace; never write author-generated files into `PARM/code/data/`.
2. Decide strict author reproduction versus local-model adaptation. Strict reproduction requires `PKU-Alignment/alpaca-7b-reproduced`; local Llama-2/Tulu changes the baseline.
3. Provide or explicitly replace the missing Beaver reward/cost checkpoints used by author relabeling and evaluation.
4. Create an isolated PARM-TARO environment with the vendored PBLora PEFT, TRL-compatible versions, model-arithmetic, `safe-rlhf`, and a validated CUDA stack.
5. Train or provide a PARM adapter checkpoint. None exists locally.
6. Define independent helpfulness and harmlessness evaluators and pin their hashes. Existing DeBERTa/toxic-BERT models are candidates, not drop-in Beaver-equivalent evaluators.

## 12. Exact next recommended phase

Stop here for Phase 0/1.

The next phase should be a **TARO RAD baseline design and smoke validation**, not PARM training:

1. Snapshot the volatile V1 checkpoint/manifests into a new V2 results namespace with copied hashes.
2. Freeze the `genarm`-style Python dependency set and validate `device=auto|cuda|cpu` behavior on the host, including safe CPU fallback.
3. Specify the TARO interpolation contract and schema-v2 cache contract in `router_v2/` without importing or modifying V1 modules at runtime.
4. Generate a tiny new cache under `dataset/router_v2_cache/` and run invariant tests first.
5. Only after the smoke path passes, generate the full persistent cache and train/evaluate against immutable V1 outputs.

PARM-TARO should remain deferred until the RAD TARO baseline is reproducible and the strict-reproduction versus local-adaptation model choice is resolved.
