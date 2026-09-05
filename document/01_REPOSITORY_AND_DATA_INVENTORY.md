# 01 — Repository and Data Inventory

## Goal

Xác nhận workspace hiện tại trước khi code Router V2.

Root:

```text
/home/jupyter-iec2024se10/Reward Decoding
```

## 1. V1 baseline inventory

Inspect:

```text
router/
router/evaluation/
router/reports/
router/rad/
router/tests/
```

Phải tìm:
- V1 model class;
- V1 training script;
- V1 checkpoints;
- full evaluation configs/results;
- current feature/cache schema.

Không sửa.

## 2. RAD source inventory

Inspect:

```text
Method/RAD/
dataset/RAD_train/router_amazon_polarity/
dataset/router_cache/rad/
dataset/router_train/rad/
dataset/rad_benchmark/
models/gpt2-*
models/rad_rm_sentiment/
```

Report exact counts và hashes.

## 3. PARM inventory

Inspect:

```text
PARM/code/data/
PARM/code/training/
PARM/code/evaluation/
PARM/language-model-arithmetic/
PARM/peft/
```

Tìm:
- expected dataset filename;
- preprocessing scripts;
- training configs;
- checkpoint paths;
- base model IDs;
- reward/oracle model IDs;
- generation code.

Không sửa.

## 4. Multi-objective data inventory

Workspace đã có:

```text
dataset/GenARM/PKU-SafeRLHF-10K/round0/
dataset/GenARM/full-hh-rlhf/data/
```

AI phải kiểm tra:
- file formats;
- row counts;
- fields;
- có đủ prompt/response/preference labels không;
- có tương thích trực tiếp với PARM preprocessing không.

Không download PKU-SafeRLHF nếu local copy đủ.

## 5. Model inventory

Inspect `models/` và report exact available checkpoints relevant to:
- base LLM;
- ARM/PARM;
- helpfulness evaluator;
- harmlessness/cost evaluator;
- independent evaluator.

## 6. New paths proposed

```text
router_v2/
dataset/router_v2_cache/
dataset/router_v2_train/
PARM_TARO/
results/router_v2/
results/parm_taro/
```

Không cần `Method/RouterV2/` trừ khi implementation thực tế yêu cầu.

## 7. Required report

Create:

```text
router_v2/reports/v2_inventory.md
```

Nếu `router_v2/` chưa tồn tại, chỉ tạo directory/report; chưa implement model.

Report:
- V1 files and hashes;
- PARM files and hashes;
- existing data counts;
- usable local PKU-SafeRLHF status;
- model/checkpoint availability;
- missing prerequisites;
- exact next recommended phase.

## Acceptance criteria

- xác nhận PKU local có dùng được hay không;
- không thay đổi V1/PARM;
- no new downloads unless truly missing;
- hashes stored for baseline protection.

---

## Prompt cho AI

Inspect toàn workspace và tạo `router_v2/reports/v2_inventory.md`. Đặc biệt kiểm tra `router/`, `PARM/`, `dataset/GenARM/PKU-SafeRLHF-10K/round0`, RAD router data/cache và `models/`. Không sửa code, không download data nếu local copy đủ; ghi counts, schemas, hashes, checkpoints và missing prerequisites.
