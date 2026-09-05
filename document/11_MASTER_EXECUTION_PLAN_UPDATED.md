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

Prepare and freeze:
- alpha grid;
- objective/reward normalization statistics;
- Pareto/HV reference point;
- Preference Regret oracle/candidate-pool protocol.

Các protocol này phải dùng chung cho mọi method.

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

Alpha phải là preference vector thật từ multi-objective setup; không dùng constant/synthetic alpha để claim preference-aware result.

## Phase 9 — final evaluation

Write:

```text
results/parm_taro/
```

Report:
- Pareto;
- HV;
- MIP;
- PCS (Preference Cosine Similarity);
- Preference Regret;
- PPL/coherence;
- latency;
- lambda analysis;
- ablations;
- same-average-lambda controls;
- alpha controls;
- statistics.

### Required preference-aware metrics

#### MIP

\[
MIP
=
\frac{1}{N}
\sum_j
\alpha_j^\top r_j.
\]

Đo reward đạt được sau khi weighted theo preference user.

#### PCS

\[
PCS
=
\frac{1}{N}
\sum_j
\cos(\alpha_j,\tilde r_j).
\]

Đo direction của normalized objective vector có bám sát preference direction của user hay không.

#### Preference Regret

\[
Preference\ Regret
=
\frac{1}{N}
\sum_j
\left(
\alpha_j^\top r_j^{best}
-
\alpha_j^\top r_j^{model}
\right).
\]

Càng thấp càng tốt.

### Required controls

```text
PARM original static
PARM + faithful TARO
PARM + V2 no-alpha
PARM + V2 full-alpha
V2 full-alpha same-average-lambda
V2 full-alpha shuffled-alpha
V2 full-alpha fixed/mean-alpha
```

### Final scientific claim gate

Không claim preference-aware improvement chỉ vì alignment hoặc một objective tăng.

Strong claim chỉ được phép khi full-alpha model cho thấy hợp lý trên nhiều góc:
- Pareto/HV không xấu đi đáng kể và tốt hơn ở target comparison;
- MIP tăng;
- PCS tăng;
- Preference Regret giảm;
- quality/PPL/coherence vẫn trong trade-off hợp lý;
- full-alpha tốt hơn no-alpha và shuffled-alpha;
- adaptive tốt hơn same-average-lambda;
- statistical evidence đủ mạnh.

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

Ở Phase 9, giữ đầy đủ Pareto/HV/MIP hiện có và bổ sung PCS + Preference Regret. Tất cả method phải dùng cùng objective normalization, cùng HV reference point và cùng oracle/candidate-pool protocol cho regret. Không được dùng metric mới để thay thế HV/MIP; PCS và Preference Regret chỉ là metric bổ sung nhằm đo trực tiếp mức độ bám sát preference của user.
