# 13 — Staged Re-Evaluation After Low-Lambda Recovery: 200 → 500 → 1500

Root:

```text
/home/jupyter-iec2024se10/Reward Decoding
```

Read first:

```text
12_ROUTER_LOW_LAMBDA_DIAGNOSIS_AND_FIX.md
10_EVALUATION_AND_ABLATIONS_UPDATED.md
11_MASTER_EXECUTION_PLAN_UPDATED.md
```

## Goal

Sau khi root cause của low-lambda collapse đã được xác nhận và fix trên **training/validation only**, chạy evaluation theo ba gate:

```text
200 validation prompts
→ 500 validation prompts
→ freeze configuration
→ 1500 test prompts
```

Không nhảy thẳng lên 1,500.

Mục tiêu là tránh tốn compute cho một fix chưa ổn và tránh tiếp tục tuning trực tiếp trên test set.

---

## 1. Data policy

### 200 prompts

Lấy deterministic subset từ validation split:

```text
seed fixed
first/sample IDs stored in manifest
same prompts for every method
same 5 alpha vectors
```

Do not use test prompts.

### 500 prompts

Use full validation split.

### 1500 prompts

Use test split **only after all architecture/loss/lambda-range decisions are frozen**.

If the previous Stage-10 1,500 test set has already influenced the redesign, the final paper should disclose this and ideally reserve/create an untouched final holdout if available under the frozen data protocol. Do not silently call a repeatedly tuned test set untouched.

---

## 2. Preserve Stage-10 protocol

Unless diagnosis proves a protocol bug, keep identical across old/new methods:

```text
5 preference vectors
objective normalization
HV reference point
Preference Regret candidate-pool/oracle protocol
generation settings
reward evaluators
PPL/coherence implementation
paired sample IDs
```

Do not improve the new method by changing evaluation rules.

---

## 3. Methods by stage

### Gate A — 200 validation prompts

Run a compact diagnostic set:

```text
1. PARM static
2. old V2 full-alpha Stage-10 checkpoint
3. recovered V2 full-alpha
4. recovered same-average-lambda
5. recovered shuffled-alpha
6. recovered no-alpha
```

Also reuse constant-lambda sweep results from File 12.

Records with 5 alpha vectors:

```text
6 methods × 200 prompts × 5 alpha = 6,000 generations
```

If compute is still expensive, PARM static and old V2 outputs may be reused only when generation/evaluation protocol is byte-for-byte identical and prompt IDs match.

### Gate B — 500 validation prompts

Run all required methods:

```text
PARM static
PARM + TARO
V2 no-alpha
V2 full-alpha recovered
V2 same-average-lambda
V2 shuffled-alpha
V2 fixed/mean-alpha
```

Records:

```text
7 × 500 × 5 = 17,500 generations
```

### Gate C — 1500 test prompts

After freeze, run the same seven methods.

Records:

```text
7 × 1500 × 5 = 52,500 generations
```

---

## 4. Metrics required at every gate

Primary:

```text
HV ↑
MIP ↑
PCS ↑
Preference Regret ↓
PPL ↓
Coherence ↑
```

Secondary:

```text
distinct-1/2/3 ↑
repeated-4gram rate ↓
generation length
latency/token ↓
```

Router diagnostics:

```text
lambda mean/std/min/max
p05/p25/p50/p75/p95
fraction near floor
fraction near ceiling
lambda by token position
lambda by alpha
lambda vs base entropy
lambda vs JS disagreement
same-state alpha counterfactual sensitivity
```

---

## 5. Gate A — 200-prompt acceptance criteria

This gate is for **sanity + direction**, not final significance claims.

### Required functional checks

Must all pass:

```text
lambda=1 static-equivalence test passes
no NaN/Inf
all methods produce complete records
recovered router does not collapse to a constant
recovered router does not saturate almost entirely at floor/ceiling
correct alpha changes lambda measurably on same-state counterfactuals
```

### Alignment recovery check

Compare recovered full-alpha to old Stage-10 full-alpha:

```text
HV must increase
MIP must increase
PCS must increase
Regret must decrease
```

At least three of the four must improve in the expected direction; HV may not materially worsen.

### Static-gap recovery

Report how much of the Stage-10 gap to PARM static is recovered:

\[
Recovery(metric)
=
\frac{
metric_{new}-metric_{oldV2}
}{
metric_{static}-metric_{oldV2}
}.
\]

For Regret use reversed direction.

Do not hard-code a lambda target such as `mean lambda = 1`.

The correct lambda region is the region supported by the validation constant-lambda sweep.

### Quality guardrail

Recovery must not simply reproduce PARM static by losing all quality gains.

Report:

```text
Delta PPL vs old V2
Delta coherence vs old V2
Delta PPL vs PARM static
Delta coherence vs PARM static
```

### Alpha check

Required direction:

```text
correct-alpha MIP/PCS > shuffled-alpha
correct-alpha Regret < shuffled-alpha
```

If correct vs shuffled is essentially indistinguishable again:

> STOP before 500. Return to alpha-path diagnosis.

### Adaptive check

Required direction:

```text
adaptive full-alpha > same-average on preference metrics
```

If same-average matches or beats adaptive across HV/MIP/PCS:

> token-level adaptation is not yet demonstrated; do not scale up.

---

## 6. Gate B — 500-prompt acceptance criteria

Use the full validation set.

At this stage, require stronger evidence.

### Target comparisons

```text
recovered full-alpha vs PARM static
recovered full-alpha vs TARO
recovered full-alpha vs no-alpha
recovered full-alpha vs same-average
recovered full-alpha vs shuffled-alpha
recovered full-alpha vs fixed/mean-alpha
```

### Required behavior

Before test freeze, recovered full-alpha should show:

- materially recovered HV/MIP/PCS relative to Stage-10 full-alpha;
- Regret materially reduced;
- better preference metrics than TARO and no-alpha;
- correct-alpha better than shuffled/fixed-alpha;
- adaptive better than same-average-lambda;
- PPL/coherence remain inside an explicitly documented trade-off relative to PARM static.

### Statistics

Run:

```text
paired bootstrap CI
paired permutation test
effect size
Holm correction
```

At minimum, the **direction** of all primary target comparisons must be stable across bootstrap samples.

For promotion to 1,500, require statistically credible improvement on the comparisons that support the main novelty:

```text
full-alpha vs no-alpha
full-alpha vs shuffled-alpha
full-alpha vs same-average-lambda
```

If those fail, do not spend the 1,500-prompt test budget yet.

---

## 7. Freeze before Gate C

After 500 validation prompts:

Freeze and hash:

```text
router checkpoint
router config
lambda parameterization
loss weights
alpha grid
generation config
objective normalization
HV reference point
regret protocol
evaluator checkpoints
```

Create:

```text
results/parm_taro/recovery/protocol_lock.json
```

No architecture/loss/hyperparameter changes after this point.

---

## 8. Gate C — 1500-prompt final test

Run seven required methods under the frozen protocol.

Write only to:

```text
results/parm_taro/recovery/test1500/
```

Do not overwrite Stage 10:

```text
results/parm_taro/evaluation/full/
```

Required artifacts:

```text
generation_records.jsonl
scored_records.jsonl
raw_objective_vectors.jsonl
pareto_points.csv
pareto_points.json
method_aggregates.json
paired_statistics.csv
paired_statistics.json
recovery_stage10_report.json
```

---

## 9. Final scientific claim gate

Strong claim only if recovered full-alpha shows a coherent result across:

```text
Pareto/HV
MIP
PCS
Preference Regret
PPL/coherence
alpha controls
same-average control
statistics
lambda diagnostics
```

Specifically:

- alignment loss relative to PARM static is substantially recovered or the new alignment-quality trade-off is demonstrably superior;
- full-alpha beats TARO/no-alpha on preference metrics;
- correct-alpha beats shuffled/fixed-alpha;
- adaptive beats same-average-lambda;
- lambda is non-collapsed and meaningfully conditioned on decoding state;
- alpha has measurable counterfactual effect on routing;
- baseline hashes remain unchanged.

Do not claim “preference-aware adaptive improvement” if shuffled-alpha remains equivalent or if same-average remains equivalent.

---

## 10. Failure routing

```text
200 fails static equivalence
    -> integration fix

200 alignment does not recover
    -> return to lambda scale/objective diagnosis

200 correct ~= shuffled
    -> return to alpha-path training

200 adaptive ~= same-average
    -> return to token-level feature/history training

500 gains unstable
    -> do not run 1500

500 passes
    -> freeze all configs
    -> run 1500 once
```

---

## 11. Report table template

For each stage:

| Method | HV ↑ | MIP ↑ | PCS ↑ | Regret ↓ | PPL ↓ | Coh ↑ | mean λ | std λ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PARM static | | | | | | | 1.0 | 0 |
| TARO | | | | | | | | |
| V2 no-alpha | | | | | | | | |
| V2 full-alpha | | | | | | | | |
| same-average | | | | | | | | 0 |
| shuffled-alpha | | | | | | | | |
| fixed-alpha | | | | | | | | |

Also report:

```text
Delta full vs static
Delta full vs TARO
Delta full vs no-alpha
Delta full vs same-average
Delta full vs shuffled
Delta full vs fixed-alpha
```

---

## Acceptance criteria

- 200 uses validation only and passes sanity/directional gates;
- 500 uses full validation and supports the novelty comparisons;
- configuration is frozen before test;
- 1500 test run uses the frozen configuration once;
- Stage-10 artifacts remain untouched;
- original PARM/Router V1 hashes remain unchanged;
- all seven methods share the same protocol;
- raw records are preserved for independent recomputation.

---

## Prompt cho AI

Sau khi `12_ROUTER_LOW_LAMBDA_DIAGNOSIS_AND_FIX.md` xác nhận root cause và tạo recovered Router checkpoint, chạy staged re-evaluation 200→500→1500. Dùng deterministic 200-prompt subset của validation trước, sau đó full 500 validation prompts. Giữ cùng 5 alpha, normalization, HV reference, regret candidate-pool và evaluator protocol giữa mọi method. Ở 200, xác nhận recovered full-alpha tăng HV/MIP/PCS và giảm Regret so với old full-alpha, correct-alpha tốt hơn shuffled-alpha theo đúng direction, và adaptive tốt hơn same-average. Nếu fail thì dừng và quay lại diagnosis. Nếu 200 pass, chạy đủ bảy methods trên 500 validation, paired bootstrap/permutation/effect size/Holm; chỉ khi full-alpha có evidence đáng tin hơn no-alpha, shuffled-alpha và same-average mới freeze checkpoint/config/protocol. Sau freeze, chạy đúng một full evaluation trên 1,500 test prompts × 5 alpha × 7 methods và ghi toàn bộ outputs sang `results/parm_taro/recovery/test1500/`, không overwrite Stage 10. Giữ `PARM/`, `router/`, PBLoRA và artifacts cũ read-only, kiểm tra hashes trước/sau.
