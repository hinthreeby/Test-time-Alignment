# 11 — Master Execution Plan for Current Workspace

Root:

```text
/home/jupyter-iec2024se10/Reward Decoding
```

## Phase 0 — Freeze baselines

Hash/read-only:
```text
router/
router/evaluation/
PARM/
```

Create:
```text
router_v2/reports/baseline_hashes.json
```

## Phase 1 — Inventory

Inspect:
```text
router/
PARM/
dataset/RAD_train/router_amazon_polarity/
dataset/router_cache/rad/
dataset/router_train/rad/
dataset/rad_benchmark/
dataset/GenARM/PKU-SafeRLHF-10K/round0/
models/
```

Output:
```text
router_v2/reports/v2_inventory.md
```

## Phase 2 — TARO baseline

Create:
```text
router_v2/
```

Do not create/modify `router/`.

## Phase 3 — V2 cache

Create:
```text
dataset/router_v2_cache/
dataset/router_v2_train/
```

Do not modify:
```text
dataset/router_cache/rad/
dataset/router_train/rad/
```

## Phase 4 — Smart Router V2

Increment:

```text
TARO Top-K
→ +confidence
→ +disagreement
→ +position
→ +history
```

## Phase 5 — RAD validation

Read:
```text
06_RAD_V2_VALIDATION.md
```

Write new outputs to:

```text
results/router_v2/rad/
```

Do not overwrite `router/evaluation/`.

## Phase 6 — PARM data

Inspect local:

```text
dataset/GenARM/PKU-SafeRLHF-10K/round0/
PARM/code/data/
```

If processed data needed, create:

```text
dataset/parm_taro/
```

## Phase 7 — PARM_TARO

Create:

```text
PARM_TARO/
```

Do not edit `PARM/`.

## Phase 8 — preference-aware Router V2

Train:
```text
PARM static
PARM + TARO
PARM + V2 no-alpha
PARM + V2 alpha
```

## Phase 9 — final evaluation

Write:

```text
results/parm_taro/
```

Report:
- Pareto;
- HV;
- MIP;
- PPL/coherence;
- latency;
- lambda analysis;
- ablations;
- statistics.

## First exact AI task

1. Create `router_v2/reports/` only.
2. Hash V1/PARM.
3. Inventory all data/model/checkpoint paths.
4. Inspect local PKU dataset.
5. Create `router_v2/reports/v2_inventory.md`.
6. Stop before implementing Router V2.

---

## Prompt cho AI

Bắt đầu tại workspace hiện tại bằng baseline hash + inventory. Không code Router V2 ngay. Giữ `router/`, `router/evaluation/`, `PARM/`, V1 data/cache bất biến. Kiểm tra local PKU data và model checkpoints; tạo `router_v2/reports/v2_inventory.md`, sau đó dừng và báo prerequisites trước khi sang TARO baseline.
