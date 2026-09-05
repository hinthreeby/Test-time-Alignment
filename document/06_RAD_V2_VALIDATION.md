# 06 — Validate Router V2 on RAD

## Goal

So Router V2 với V1 trên đúng workspace hiện tại.

## Existing V1 outputs — READ ONLY

```text
router/evaluation/
├── pilot/
├── full_32tokens/
└── report_run_32tokens/
```

Không overwrite.

## New results

```text
results/router_v2/rad/
```

hoặc:

```text
router_v2/evaluation/rad/
```

Chọn một convention và giữ nhất quán.

## Baselines

```text
base
fixed
heuristic
V1 from router/
TARO Router V2
Smart Router V2
```

V1 phải được load như baseline, không modify/import side effects.

## RAD score

\[
s_{t,i}=z_{t,i}^{base}+\lambda_t r_{t,i}.
\]

## Test data

Reuse:

```text
dataset/rad_benchmark/
```

## Metrics

- independent sentiment alignment;
- target probability;
- PPL;
- coherence;
- repetition;
- distinct-n;
- generation length;
- latency/token;
- router lambda diagnostics.

## Pareto

Primary:

\[
x=PPL\ degradation,\quad y=alignment.
\]

## Required regression

- V1 reproduced from existing checkpoint/config.
- Existing V1 result files unchanged.
- Same-average-lambda fixed baseline included.
- 32+ token generations.

## Exit condition

Chỉ lên PARM nếu Router V2 stable và không collapse.

---

## Prompt cho AI

Evaluate Router V2 trên `dataset/rad_benchmark/`, dùng V1 trong `router/` và các output cũ chỉ làm read-only baseline. Ghi mọi kết quả mới sang `results/router_v2/rad/` hoặc `router_v2/evaluation/rad/`. So Base/Fixed/Heuristic/V1/TARO/V2 bằng alignment, PPL, coherence, latency, Pareto và same-average-lambda.
