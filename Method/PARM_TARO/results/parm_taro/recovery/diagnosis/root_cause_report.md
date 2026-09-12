# PARM-TARO low-lambda recovery: root-cause report

## Decision

**RECOVERY TRAINING: BLOCKED. Gate A (val200): NOT STARTED.**

The retained Stage-9/10 evidence confirms that gold-token NLL and PARM's
sequence-level alignment objective disagree. It also shows that the alpha path
is weak. However, this checkout does not contain the ignored Tulu-2-7B model,
PBLoRA, three Stage-9 checkpoints, a PyTorch environment, or a working GPU.
Consequently the mandatory fresh D1, per-token D3, sequence-level D4,
equal-step D5, and feature-level D7 measurements cannot be completed here.

File 12 explicitly requires D0-D7 before changing the model and requires a
numeric static-equivalence pass before retraining. No checkpoint was trained,
no lambda range was selected, and no test data was used for tuning. The staged
200 -> 500 -> 1500 evaluation therefore correctly stops before 200.

## D0: preservation boundary

The current checkout was hashed into `protected_artifacts_before.json` before
adding recovery code. The snapshot has source trees but omits ignored large
files, so its tree hashes cannot be equated with the archived full-workspace
hashes. The archived Stage-10 report records these last complete hashes:

| Protected artifact | Archived Stage-10 SHA-256 |
|---|---|
| `PARM/` | `dcbaae05b0c3467e2d9ff3255ca6fc9f2296bc4e97f9a22b650e38f666bac23a` |
| `router/` | `8595fb3e48c2b1771a07a6efba09c29560785a581d5490f582286c58fbf5ffd3` |
| `Method/RAD/` | `e0c8907885b80cde9c12f0e1f6e3134b0781e811ee8edafb9f0d690988d36bbe` |
| PBLoRA tree | `f26682e3cf51f8e09921873119fc371d329669cf9448d59ae0025fd40b23ffef` |
| full-alpha Stage-9 checkpoint | `56fbed68168537bd84066a5b86fbfefc8525cb5311c06d8668012e2cecca1e20` |

This recovery changed only `PARM_TARO/recovery/` and
`results/parm_taro/recovery/`. It did not modify Stage-9/10 artifacts.

## D1: static equivalence

**Numeric status: BLOCKED (0/100 required real states).** Source-level audit:

```text
log_softmax(log_softmax(base_logits) + lambda * log_softmax(guide_logits))
```

The static and adaptive decoders call the same implementation. Algebraically,
lambda=0 reduces to the base distribution and lambda=1 is the static PARM
formula. This is not substituted for the required 100-state numeric test.
See `static_equivalence.json`.

## D2: exact Stage-9 lambda mapping

The checked-in production router config and source establish:

```text
lambda_max          = 1.0
lambda_eps          = 1e-6
exact lambda floor  = 1e-6
exact lambda ceiling= 0.999999
initial_gate        = 0.5
initial output bias = logit(0.5) = 0.0
gate                = clamp(sigmoid(raw), 1e-6, 1-1e-6)
final lambda        = 1.0 * gate
```

Stage-10 generated output was mean 0.0107313, std 0.0046911, min 0.0065502,
max 0.0466272 (461,733 tokens). Because `lambda_max=1`, gate and lambda are
identical: this is a genuinely low sigmoid gate, not a hidden 0.05 cap. The
observed extrema imply raw logits about -5.0217 to -3.0178. Exact raw
mean/quantiles and trained output bias require the missing checkpoint and are
left null rather than fabricated in `lambda_parameterization.json`.

## D3: gold-token utility and dL/dlambda

**Exact per-token audit: BLOCKED.** The full-vocabulary token distributions
were not retained in Git, so the requested fraction
`dL/dlambda > 0 at lambda=0` cannot be reconstructed from aggregates.

Direct archived validation evidence is nevertheless strong. Across all five
real alpha values and 500 validation examples per alpha, the fixed-lambda
gold-token NLL optimum was exactly 0. For alpha_helpfulness 0, 0.25, 0.5,
0.75, and 1, mean NLL rose respectively:

| alpha_help | NLL lambda=0 | NLL lambda=0.01 | NLL lambda=1 |
|---:|---:|---:|---:|
| 0.00 | 1.888601 | 1.897598 | 3.774154 |
| 0.25 | 1.879254 | 1.887763 | 3.601291 |
| 0.50 | 1.869907 | 1.877608 | 3.433806 |
| 0.75 | 1.860561 | 1.867251 | 3.305572 |
| 1.00 | 1.851214 | 1.856831 | 3.198342 |

`gold_token_utility.csv` preserves the full archived sweep. The new exact
implementation in `PARM_TARO/recovery/diagnostics.py` computes
`E_base[log p_guide] - log p_guide(gold)` once the runtime is restored.

## D4: constant-lambda validation sweep

**Sequence-level val200 sweep: BLOCKED.** `constant_lambda_sweep.csv` contains
only the archived teacher-forced NLL sweep and labels its scope explicitly.
It has no HV/MIP/PCS/Regret entries and must not be used to select recovery
bounds. Stage-10 test values are used only to diagnose the already-observed
failure, never to tune a replacement.

## D5: regularizer ablation

**Required equal-step ablation: BLOCKED.** The archived objective contains a
lower-floor strength term `relu(0.01 - mean_lambda)^2`; it pushes lambda up
below the floor and has zero gradient above it. Production calibration reports
zero strength gradient in all 16 calibration samples. No entropy or smoothness
term appears in the production objective/config. Thus there is no evidence
that a shrinkage regularizer caused Stage-10 collapse, but the exact File-12
NLL/entropy/smoothness/strength equal-step experiment remains to be run.

## D6: alpha counterfactual sensitivity

Archived same-state validation contains 382,640 token states. After production
training, mean endpoint-alpha lambda delta was 0.002626 and mean correct-vs-
shuffled delta was 0.001052, against mean lambda 0.009850. The earlier feature
audit measured preference-to-state first-layer contribution ratio 0.04960;
preference encoder and fusion gradient norms under the preference loss were
0.04115 and 1.47767. Alpha is not completely disconnected, but its expressed
effect is small. Stage-10 sequence metrics agree: correct-vs-shuffled MIP was
only +0.000768 and PCS only +0.000689.

## D7: teacher-forced versus generated prefixes

**Feature-level audit: BLOCKED.** The available aggregates do not show a lambda
collapse unique to generation: teacher-forced validation mean was 0.0098503
and generated Stage-10 mean was 0.0107313 (ratio 1.0894). This argues against
H8 as the primary cause, but entropy/JS/top-1/history distributions cannot be
compared without generation records and model runtime. See
`exposure_shift.json`.

## H1-H8 adjudication

| Hypothesis | Status | Evidence and interpretation |
|---|---|---|
| H1: token NLL drives lambda down | **CONFIRMED** | Archived validation fixed-lambda NLL is minimized at 0 for all 5 alphas and increases monotonically through 1. Exact positive-gradient fraction is still a required rerun. |
| H2: lambda mapping/scale bug | **REJECTED** | Exact range is [1e-6, 0.999999], `lambda_max=1`; observed lambda equals the low sigmoid gate. Sigmoid saturation amplifies recovery difficulty but a hidden small multiplier does not explain the value. |
| H3: train/inference equation mismatch | **REJECTED** | Training routing and adaptive generation both use base log-prob + lambda * guide log-prob, and static/adaptive share the oracle function. Fresh 100-state numeric proof remains blocked, so this rejection is conditional on D1. |
| H4: shrinkage regularizer | **REJECTED** | Production strength loss is floor-promoting, had zero calibration gradient, and entropy/smoothness losses are absent. Equal-step ablation remains blocked. |
| H5: sequence alignment vs token-NLL mismatch | **CONFIRMED** | Lambda=1 has much worse gold-token NLL yet Stage-10 static PARM is far better: HV 0.425525 vs 0.316785, MIP 0.634510 vs 0.572453, PCS 0.850811 vs 0.799186, Regret 0.042023 vs 0.104080. |
| H6: alpha path weak/ignored | **PARTIALLY_CONFIRMED** | Same-state lambda changes with alpha, so the path is not disconnected; contribution ratio is only 4.96%, and correct/shuffled Stage-10 preference metrics are nearly identical. |
| H7: feature/logit scale mismatch | **PARTIALLY_CONFIRMED** | Preference features are numerically drowned by state features (4.96% first-layer contribution) and sigmoid derivative is small near collapse. Base-vs-guide full-logit scale statistics are unavailable, so the broader H7 is not fully established. |
| H8: teacher/generated exposure shift | **REJECTED** | Available lambda means are similar and generated is 8.9% higher, not lower. Rejection is provisional because the required feature-distribution audit is blocked. |

## Root cause and permitted fix family

The primary root cause is **objective mismatch (H1 + H5)**: gold-token NLL can
improve by turning off the very PARM guidance that delivers sequence-level
preference alignment. A secondary issue is weak alpha expression (H6/H7).

The only prepared fix family is isolated under `PARM_TARO/recovery/`:

```text
lambda = lambda_0 * (1 + rho * tanh(raw)), lambda_0 = 1
optional KL(PARM-static || adaptive) trust region
```

This is candidate code, not an applied or tuned fix. `rho` and any KL weight
must be selected using train/validation only after D1-D7 are complete. No
post-hoc multiplier is present.

## Unblock and stage order

Restore the exact assets matching archived hashes, restore the original
PyTorch/CUDA environment, then run in order:

1. Re-hash all protected full trees and checkpoints; abort on any mismatch.
2. Run numeric D1 on at least 100 deterministic real token states.
3. Emit exact raw/gate/lambda/bias quantiles from the production checkpoint.
4. Run per-token D3 on train and validation, including required buckets.
5. Run the sequence-level constant-lambda sweep on deterministic val200.
6. Run equal-step D5, same-state D6, and feature-level D7.
7. Only if those gates corroborate this report, train one residual/trust-region
   family; do not combine unrelated fixes.
8. Apply File 13: val200, stop on failure; val500 plus statistics, stop on
   failure; freeze/hash; then exactly one test1500 run.

