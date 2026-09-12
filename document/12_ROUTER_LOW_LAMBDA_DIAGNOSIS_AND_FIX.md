# 12 — Diagnose Low-Lambda Collapse and Recover PARM Alignment

Root:

```text
/home/jupyter-iec2024se10/Reward Decoding
```

## Goal

Stage 10 đã PASS về mặt kỹ thuật, nhưng Router V2 trên PARM đang học guidance strength quá thấp và làm mất nhiều preference alignment.

Observed Stage-10 evidence:

```text
PARM static
lambda mean = 1.000000
HV          = 0.425525
MIP         = 0.634510
PCS         = 0.850811
Regret      = 0.042023
PPL         = 2.343498

V2 full-alpha
lambda mean = 0.010731
lambda std  = 0.004691
lambda max  = 0.046627
HV          = 0.316785
MIP         = 0.572453
PCS         = 0.799186
Regret      = 0.104080
PPL         = 2.079136

V2 no-alpha
lambda mean = 0.000765

PARM + TARO
lambda mean = 0.001442
```

V2 full-alpha tốt hơn TARO và no-alpha, nhưng vẫn kém PARM static đáng kể về HV/MIP/PCS/Regret. Correct-alpha cũng chỉ khác rất ít shuffled/fixed-alpha.

Mục tiêu của phase recovery này là:

1. xác định **nguyên nhân thật sự** khiến \(\lambda_t\) collapse về gần 0;
2. không sửa mò bằng cách nhân \(\lambda\) với một constant hậu nghiệm;
3. fix đúng nguyên nhân trong Router V2/PARM_TARO;
4. giữ nguyên `PARM/`, `router/`, Stage-10 artifacts và test protocol;
5. chỉ chạy lại full 1,500 test prompts sau khi fix đã vượt validation gates.

---

## 1. Scientific interpretation of the failure

PARM static sử dụng:

\[
\log \tilde\pi_t(v)
=
\log \pi_{base,t}(v)
+
\lambda\log\pi_{\theta(\alpha),t}(v),
\qquad \lambda=1.
\]

PARM quyết định reward direction/trade-off theo preference vector \(\alpha\), còn Router V2 quyết định token-level guidance strength \(\lambda_t\).

Router V2 hiện có:

\[
\bar\lambda\approx 0.0107,
\]

chỉ khoảng 1/93 của static PARM strength. Vì vậy model gần với base LM hơn, phù hợp với quan sát:

```text
PPL/coherence tốt hơn
alignment metrics giảm mạnh
```

Đây là **symptom**, chưa được phép coi là root cause trước khi audit các mục bên dưới.

---

## 2. Most likely root-cause hypotheses

### H1 — Token-NLL objective naturally pushes lambda toward zero

Current Router V2 main loss:

\[
\mathcal L_{NLL}
=
-\sum_t\log p_t^\lambda(y_t^\star).
\]

Nếu base LM thường cho gold token xác suất/rank tốt hơn PARM guide trên teacher-forced prefixes, NLL sẽ giảm bằng cách giảm reward guidance.

Với:

\[
s_t(v)=z^b_t(v)+\lambda_t z^g_t(v),
\]

đạo hàm của token NLL theo \(\lambda\) là:

\[
\frac{\partial \mathcal L_t}{\partial \lambda}
=
\mathbb E_{v\sim p_t^\lambda}[z^g_t(v)]
-
z^g_t(y_t^\star).
\]

Nếu giá trị này thường dương tại \(\lambda\approx0\), gradient sẽ trực tiếp đẩy \(\lambda\) xuống.

TARO paper cũng ghi nhận trường hợp base model nhất quán hơn có thể làm router dự đoán alpha nhỏ một cách nhất quán và under-use reward model. Vì vậy H1 là hypothesis ưu tiên kiểm tra đầu tiên.

### H2 — Lambda parameterization / scale is incompatible with PARM static scale

Audit exact implementation:

```text
raw router output
sigmoid gate g_t
lambda_max
lambda_min / clamp
output bias initialization
final lambda equation
```

Phải phân biệt rõ hai trường hợp:

```text
A. g_t thực sự gần 0
B. g_t bình thường nhưng lambda_max quá nhỏ
```

Stage-10 `lambda max = 0.0466` là tín hiệu cần audit, nhưng không được suy ra `lambda_max=0.05` nếu chưa đọc config/code/checkpoint.

### H3 — Train/inference equation mismatch

TARO paper-style interpolation:

\[
z_t^{guided}
=
(1-\hat\alpha_t)z_t^{base}
+
\hat\alpha_t z_t^{reward}.
\]

PARM_TARO wrapper target equation:

\[
\log \tilde\pi_t
=
\log\pi_{base,t}
+
\lambda_t\log\pi_{parm,t}.
\]

Nếu router được trained/normalized theo một mixing convention nhưng evaluation sử dụng convention khác, numeric meaning của routing coefficient có thể bị lệch mạnh.

### H4 — Optional regularization shrinks lambda

Stage-5 design có optional:

\[
\mathcal L_{strength}
=
\sum_t(\lambda_t/\lambda_{max})^2,
\]

và entropy/smoothness regularizers.

Nếu `strength` đang bật hoặc entropy làm router commit về phía base model, chúng có thể trực tiếp tạo low-lambda collapse.

Phải audit config thực tế của Stage 9, không suy đoán từ design document.

### H5 — PARM guide is useful at sequence level but weak for gold-token NLL

PARM được tối ưu để điều khiển multi-objective preference trade-off, không nhất thiết để tăng xác suất token của một single gold response ở mọi prefix.

Có thể xảy ra:

```text
PARM static -> sequence-level HV/MIP tốt
NLL training -> base token prediction tốt hơn
Router -> học giảm PARM guidance
```

Nếu đúng, đây là **objective mismatch**, không phải architecture failure.

### H6 — Alpha path is weak or ignored

Stage-10:

```text
full-alpha MIP      = 0.572453
shuffled-alpha MIP  = 0.571685
fixed-alpha MIP     = 0.571680
```

Khoảng cách nhỏ cho thấy preference input chưa tạo tác động đủ lớn.

Possible causes:
- alpha encoder gradient quá nhỏ;
- context/logit features dominate fusion;
- training batches không tạo đủ counterfactual pressure giữa nhiều alpha cho cùng state;
- alpha tới PARM guide nhưng không ảnh hưởng đủ tới router;
- history path che mất alpha signal.

### H7 — Feature/logit scale mismatch

Audit distribution của:

```text
base logits
PARM logits/log-probs
Top-K values
normalized features
confidence/disagreement
```

Nếu scale giữa base và PARM guide quá khác, router có thể học một coefficient rất nhỏ chỉ để cân bằng magnitude.

Không normalize final PARM equation một cách tùy tiện vì constant `lambda=1` phải còn reproduce static PARM.

### H8 — Teacher-forcing / autoregressive state mismatch

Training dùng teacher-forced prefixes nhưng inference dùng model-generated prefixes và history state online.

Nếu history/position/confidence distribution shift mạnh, router có thể output thấp ngoài training distribution.

---

## 3. Mandatory diagnosis order

Không sửa model trước khi hoàn thành D0-D7.

### D0 — Freeze and preserve current evidence

READ ONLY:

```text
router/
PARM/
Method/RAD/
results/parm_taro/evaluation/full/
results/parm_taro/training/
```

Create new only:

```text
results/parm_taro/recovery/
PARM_TARO/recovery/
router_v2/recovery/
```

Save hashes before any change.

### D1 — Static-equivalence regression

Must verify numerically:

```text
lambda = 0 -> base distribution
lambda = 1 -> PARM static distribution
```

For at least 100 deterministic token states:

```text
max_abs_logit_diff
max_abs_prob_diff
selected_token_match_rate
```

Required:

```text
lambda=1 adaptive wrapper == PARM static within floating-point tolerance
```

If this fails:

> STOP. Root cause is integration/equation mismatch. Do not retrain router yet.

### D2 — Inspect the actual Stage-9 lambda parameterization

Report:

```text
lambda_max
lambda_min
raw output mean/std/p05/p50/p95
sigmoid output mean/std/p05/p50/p95
final lambda mean/std/p05/p50/p95
output-layer bias
checkpoint/config path
```

Decision:

```text
if raw gate normal but final lambda tiny:
    parameterization/scale bug
elif raw gate itself near zero:
    training objective/regularizer/data pushes collapse
```

### D3 — Gold-token guidance utility audit

On training + validation token cache, compute for every token:

```text
base gold log-prob
PARM guide gold log-prob
base gold rank
PARM guide gold rank
NLL(lambda=0)
NLL(lambda=0.01)
NLL(lambda=0.05)
NLL(lambda=0.1)
NLL(lambda=0.25)
NLL(lambda=0.5)
NLL(lambda=1.0)
dL/dlambda at lambda=0
```

Aggregate by:

```text
alpha bucket
position bucket
base entropy bucket
JS disagreement bucket
helpfulness/harmlessness preference region
```

Required root-cause statistic:

```text
fraction(dL/dlambda > 0 at lambda=0)
```

If this fraction is dominant, token NLL intrinsically prefers low guidance.

### D4 — Constant-lambda sweep on validation only

Use the same frozen base/PARM and same 5 alpha grid.

Sweep:

```text
lambda ∈ {0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25}
```

First run only on the deterministic 200-prompt validation subset defined in File 13.

Measure:

```text
HV
MIP
PCS
Preference Regret
PPL
coherence
repetition
distinct-n
latency
```

Purpose:

> Determine the actual alignment-quality operating region before changing router scale.

Do not choose lambda from test data.

### D5 — Regularizer ablation

Re-evaluate/retrain short diagnostic variants:

```text
NLL only
NLL + entropy
NLL + smoothness
NLL + strength   # only if currently used
current Stage-9 objective
```

Record lambda distribution after the same number of optimization steps.

If adding a regularizer causes the major lambda drop, isolate/remove or retune it.

### D6 — Alpha counterfactual sensitivity

For the exact same prefix/state, run all real preference vectors through the router:

```text
alpha_1 ... alpha_5
```

Measure:

```text
std(lambda | same state, varying alpha)
mean pairwise |delta lambda|
rank correlation alpha_help vs lambda
rank correlation alpha_safe vs lambda
alpha-encoder gradient norm
fusion-layer alpha-path gradient norm
```

Also repeat with shuffled/fixed alpha.

If lambda is almost unchanged under counterfactual alpha, alpha conditioning is functionally ignored.

### D7 — Teacher-forced vs generated-prefix audit

Compare router feature distributions and lambda distributions on:

```text
teacher-forced validation prefixes
autoregressive generated prefixes
```

Report distribution shift for:

```text
entropy
JS
top1 agreement
position/history state
lambda
```

---

## 4. Root-cause decision table

| Finding | Root cause | Required fix |
|---|---|---|
| `lambda=1` does not reproduce PARM static | Integration/equation bug | Fix wrapper before retraining |
| `g_t` normal but final lambda tiny | Lambda scale/parameterization bug | Correct lambda mapping/range |
| `dL/dlambda > 0` for most tokens near 0 | Token-NLL objective mismatch | Change training objective/anchor to PARM |
| NLL optimum near 0 but HV/MIP optimum much larger | Token-vs-sequence objective mismatch | Preference-aware/sequence-level training signal |
| Strength regularizer causes shrinkage | Over-regularization | Disable/retune strength cost |
| Entropy commits mostly to base | Entropy-sensitive collapse | Reduce/remove entropy or change initialization |
| Correct/shuffled alpha produce same lambda | Alpha ignored | Counterfactual alpha training + stronger alpha path |
| Train lambda normal, generated-prefix lambda low | Exposure/history distribution shift | Sequential/on-policy state training |
| PARM/base logit scales differ strongly | Scale imbalance | Calibrate routing representation without breaking static equivalence |

---

## 5. Preferred fix strategy after diagnosis

Do **not** apply every fix simultaneously.

### Fix A — Residual lambda around the trusted PARM static baseline

If D1 passes but NLL drives lambda down, replace a zero-centered universal-strength parameterization with a residual parameterization around the scientifically validated PARM operating point:

\[
\lambda_t
=
\lambda_0\exp\left(\delta_{max}\tanh u_t\right),
\qquad \lambda_0=1.
\]

Initialize final router head so:

\[
u_t\approx0\Rightarrow\lambda_t\approx1.
\]

Alternative bounded form:

\[
\lambda_t
=
\lambda_0(1+\rho\tanh u_t),
\]

with bounds selected **only from training/validation**, not test.

Why preferred:
- starts from PARM instead of base;
- router learns token-specific deviations from a strong baseline;
- prevents accidental near-zero collapse at initialization;
- keeps the interpretation “PARM chooses direction, Router adjusts strength”.

### Fix B — Add a PARM-preservation trust region if NLL conflicts with alignment

If D3/D4 show clear objective mismatch, use:

\[
\mathcal L
=
\mathcal L_{router}
+
\eta\,D_{KL}
\left(
\pi_{PARM-static}\;\|\;\pi_{adaptive}
\right),
\]

with schedule:

```text
start eta > 0
reduce gradually after router is stable
```

Purpose: prevent router from obtaining lower token NLL simply by turning PARM off.

Do not tune `eta` on test.

### Fix C — Preserve real multi-objective preference structure

If alpha is ignored:

- keep only real alpha from the predefined preference simplex/grid;
- batch the **same context** under multiple real alpha values;
- ensure alpha reaches both PARM and Router;
- log alpha-path gradients;
- use pairwise multi-objective preference structure already present in PARM data rather than inventing synthetic labels;
- require correct-alpha behavior to differ from shuffled/fixed-alpha on validation before full test.

### Fix D — Remove guidance-strength penalty when it is the cause

If `L_strength` is active and D5 shows collapse:

```text
disable it for recovery baseline
```

Only reintroduce after adaptive benefit is established.

### Fix E — Sequential state training if exposure shift is the cause

If D7 confirms teacher-forced/generated-prefix shift:

```text
teacher-forced warmup
→ mixed sequential prefixes
→ autoregressive router-state fine-tuning
```

Keep base/PARM/PBLoRA frozen.

---

## 6. What NOT to do

Do not:

```text
multiply Stage-10 lambda by 50/100 after generation
pick a scale because it looks good on 1,500 test prompts
modify PARM original
modify Router V1
change objective normalization/HV reference/oracle only for new method
change alpha grid only for the new method
remove shuffled/fixed/no-alpha controls
claim success because mean lambda becomes larger
```

A larger lambda is not itself the goal.

Goal:

> recover alignment while preserving acceptable quality, and demonstrate that token-level/adaptive/preference-aware routing adds value beyond static controls.

---

## 7. Required recovery artifacts

Create:

```text
results/parm_taro/recovery/
├── diagnosis/
│   ├── static_equivalence.json
│   ├── lambda_parameterization.json
│   ├── token_gradient_audit.csv
│   ├── gold_token_utility.csv
│   ├── constant_lambda_sweep.csv
│   ├── regularizer_ablation.csv
│   ├── alpha_counterfactual.csv
│   ├── exposure_shift.json
│   └── root_cause_report.md
├── checkpoints/
├── val200/
├── val500/
└── test1500/
```

`root_cause_report.md` must state one of:

```text
CONFIRMED
PARTIALLY_CONFIRMED
REJECTED
```

for H1-H8, with direct measurements.

---

## 8. Exit condition before re-evaluation

Do not proceed to 200-prompt generation until:

- static equivalence passes;
- exact lambda mapping is documented;
- low-lambda cause is identified with measurements;
- proposed fix changes only new Router V2/PARM_TARO code/config;
- training/validation data only are used for tuning;
- hashes of `PARM/`, `router/`, PBLoRA and previous Stage-9/10 artifacts remain unchanged.

Then follow `13_STAGED_REEVALUATION_200_500_1500.md`.

---

## Prompt cho AI

Làm recovery diagnosis cho PARM-TARO sau Stage 10. Không sửa `PARM/`, `router/`, PBLoRA hoặc bất kỳ Stage-10 artifact nào. Trước khi retrain, kiểm tra static equivalence (`lambda=0 -> base`, `lambda=1 -> PARM static`), audit exact `lambda_max/min`, raw router output, sigmoid gate, output bias và final lambda mapping. Trên train/validation, tính gold-token utility và `dL/dlambda` để xác định liệu gold-token NLL có đang đẩy lambda về 0; chạy constant-lambda sweep, regularizer ablation, alpha counterfactual sensitivity và teacher-forced vs generated-prefix audit. Viết `results/parm_taro/recovery/diagnosis/root_cause_report.md` với evidence cho từng hypothesis H1-H8. Chỉ sau khi root cause được xác nhận mới áp dụng đúng một nhóm fix tối thiểu, ưu tiên residual lambda quanh static PARM (`lambda≈1` tại initialization) và/hoặc PARM-preservation trust region nếu objective mismatch được chứng minh. Không tune trên test. Sau đó chuyển sang staged validation 200→500→1500 theo File 13.
