# Stage 9 Alpha Collapse Diagnostic

## Status

```text
PASS
```

## Fixed-Lambda Sweep

| alpha_h | lambda* | base NLL | optimal NLL | preferred margin at lambda* |
|---:|---:|---:|---:|---:|
| 0.00 | 0 | 1.88860055 | 1.88860055 | -0.00212656 |
| 0.25 | 0 | 1.87925391 | 1.87925391 | -0.00212656 |
| 0.50 | 0 | 1.86990728 | 1.86990728 | 0.08642152 |
| 0.75 | 0 | 1.86056064 | 1.86056064 | 0.07264652 |
| 1.00 | 0 | 1.85121401 | 1.85121401 | 0.07264652 |

## Sensitivity

- PBLORA responds to alpha: `True`
- Existing Router uses alpha materially: `False`
- All optimal lambdas near zero: `True`
- Objective collapse diagnosed: `True`

Detailed sweep cells, distribution sensitivity, finite-difference
Router derivatives, preference activation/gradient norms, provenance,
and protected hashes are recorded in the adjacent JSON report.

## Scientific Decision

The frozen PBLORA responds to alpha, but fixed-lambda gold-token NLL is minimized near zero for every alpha. Preserve NLL as a quality term and test a pairwise preference-sensitive Router objective.
