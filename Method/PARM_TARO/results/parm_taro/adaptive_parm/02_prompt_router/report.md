# Prompt-level Adaptive PARM router

Verdict: **FAIL**

## Leakage boundary

All features are computed from the raw prompt and requested alpha before the
first generated token. Reward/cost scorer values are labels only and never
inference features. Evaluation is leave-one-unique-prompt-out: all five alpha
cases for a held-out prompt stay in the same fold. No random row split is used.

## Offline policy result

| Policy | MIP | Gain vs global | Oracle-gap capture | Distinct weights |
|---|---:|---:|---:|---:|
| Global fixed | 0.646876824 | 0 | 0% | 1 |
| Per-alpha fixed | 0.653787492 | 0.006910668 | 9.52% | — |
| Logistic classifier | 0.597575055 | -0.049301769 | -67.90% | 7 |
| Ridge utility predictor | 0.617495234 | -0.029381590 | -40.46% | 7 |
| Oracle | 0.719490238 | 0.072613414 | 100% | — |

## Utility prediction diagnostics

- MSE: `0.0226105133`
- MAE: `0.119025781`
- Mean within-case Spearman: `0.006789`
- Mean within-case pairwise ranking accuracy: `0.502581`
- Oracle action accuracy: `20.00%`
- Oracle-set accuracy (tie-aware): `25.00%`
- Top-2 action accuracy: `31.67%`

## Limitations

Only 12 prompt groups are available. These are offline candidate-policy results,
not statistical or online generation evidence. The preferred method is the
predeclared small Ridge utility predictor; no large hidden-state MLP or token
router was trained.
