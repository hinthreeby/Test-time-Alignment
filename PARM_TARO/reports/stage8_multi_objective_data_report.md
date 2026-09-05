# Stage 8 PARM Multi-Objective Data Report

## Status

**STAGE 8 PASS - deterministic local multi-objective data materialized**

No data was downloaded and no reward model was run. All outputs are isolated
under `dataset/parm_taro/`; the GenARM source and original PARM tree remain
unchanged.

## Source Inventory

Inspected before processing:

```text
dataset/GenARM/PKU-SafeRLHF-10K/round0/train.jsonl.xz
PARM/code/data/
```

The local PKU source contains exactly 10,000 records with one consistent schema
and no missing or invalid labels:

```text
prompt
response_0
response_1
better_response_id
safer_response_id
is_response_0_safe
is_response_1_safe
```

Source properties:

```text
SHA256: f5f42f6f08a1fadbc3fd7c36e627f651782ce0f2f8178361483defa62b79ca06
unique prompts: 5,929
rows in repeated-prompt groups: 6,843
unique exact rows: 9,992
helpfulness preference counts: response 0 = 5,451; response 1 = 4,549
harmlessness preference counts: response 0 = 5,206; response 1 = 4,794
objective agreement/disagreement: 4,605 / 5,395
```

`PARM/code/data/` contains only `relabel.py`; author-preprocessed
`train.json`, `dev.json`, `test.json`, and `test_prompt_only.json` are absent.
The author script uses an unseeded shuffle and requires external reward/cost
models. The local source already contains both required preference labels, so
Priority B was selected and neither download nor relabeling was justified.

## Objective Semantics

Processed records retain every original source field and add explicit aliases:

```text
helpfulness_preferred_response_id = better_response_id
harmlessness_preferred_response_id = safer_response_id
```

No unavailable reward scores were synthesized. The manifest distinguishes:

```text
named processed order:       [helpfulness, harmlessness]
PARM author PBLORA order:    [harmlessness, helpfulness]
```

The second order follows `train_pref_arm.py` and `generate_outputs.py` and must
be used when constructing PARM `pref_vec`.

## Deterministic Split

Prompts are grouped first, ordered by SHA-256 of the fixed seed and prompt, then
assigned without splitting a prompt group:

```text
seed: parm_taro_pku_v1_seed_2026
train:       8,000 records / 4,755 prompt groups
validation:    500 records /   291 prompt groups
test:         1,500 records /   883 prompt groups
prompt overlap across splits: 0
split assignment SHA256: 16618dad25fb4b3561979aec857372381f60e05b4a68de1d18ce900a2d994422
```

Every source index occurs exactly once. Exact duplicates are retained as
separate examples with stable IDs such as `pku_round0_00000`.

## Artifacts

```text
train.json             9a1ca7c8dc559cb6a78d4dc773c40eee085b61588f6edc8f0f87a1f9f9875285
validation.json        cdbb914443591817d056e76ecd3cc59e28c7e95de526c5c0397c35763d2fc9ba
test.json              892a5796acb364fe3b2ab41b20e80151df8ee844b3a7cab2d351242c4e50b8d2
test_prompt_only.json  1ab1b59e3127d0077ed6908e538072bdfe541266affae51267d90d2ec8255472
source_report.json     bc54f7f66f66810fca24a3f9bc6c36ea7c66293e8a4fc5df00fa9e9115b71f59
manifest.json          dff8b3ff4762d1c864763e365188df4efb8a52adb02309cf2d5712deb423e564
```

`manifest.json` records source provenance, schema, split statistics, file
hashes, objective semantics, and invariant checks. `source_report.json` records
the source/PARM inventory and the no-download decision.

## Verification

```text
PARM-TARO tests:            27 passed
Router V2 Stage 2-6 tests: 120 passed
Combined:                  147 passed
```

The data tests cover deterministic bytes, prompt-group isolation, exact source
coverage, preservation of both objectives and duplicate rows, no-overwrite,
source immutability, manifest assignment tampering, file tampering, and full
workspace validation.

Protected artifacts after processing:

```text
GenARM source SHA256: f5f42f6f08a1fadbc3fd7c36e627f651782ce0f2f8178361483defa62b79ca06
PARM relabel.py SHA256: cf9504e5df853529e73a2405568ef56a916acd34e8f14540314c1a0320c00363
PARM tree SHA256: dcbaae05b0c3467e2d9ff3255ca6fc9f2296bc4e97f9a22b650e38f666bac23a
```

## Commands

```bash
cd "/home/jupyter-iec2024se10/Reward Decoding"
export PY=/home/jupyter-iec2024se10/miniconda3/envs/cd/bin/python
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Validate existing materialized files.
"$PY" -m PARM_TARO.scripts.prepare_multi_objective_data --validate-only

# Re-materialization is protected by default. Use --overwrite only after an
# explicit decision to replace this deterministic Stage 8 artifact.
"$PY" -m PARM_TARO.scripts.prepare_multi_objective_data --overwrite
```
