# Epoch 1 vs Epoch 2 PBLoRA control

Verdict: **IRREGULARITY_PERSISTS_AFTER_EPOCH2**

| Metric | Epoch 1 | Epoch 2 | Delta |
|---|---:|---:|---:|
| MIP | 0.586098 | 0.598125 | +0.0120273 |
| HV | 0.371519 | 0.42042 | +0.0489013 |
| helpfulness_spearman | 0.254545 | 0.415584 | +0.161039 |
| harmlessness_spearman | -0.94026 | -0.932468 | +0.00779221 |
| helpfulness_monotonicity_violations | 10 | 8 | -2 |
| harmlessness_monotonicity_violations | 7 | 6 | -1 |
| nondominated_points | 6 | 4 | -2 |
| unsupported_nondominated_points | 2 | 2 | +0 |

The runs use the identical 20-prompt manifest, 21-point alpha grid, generation protocol, Beaver scorers, and normalization.
