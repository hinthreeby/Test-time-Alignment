# Stage 9 Alpha Pilot V1 Failure Diagnostic

## Status

```text
PASS
```

## Root Cause

collapsed initialization suppresses sigmoid gradients; squared strength term is too weak after weighting; the original pairwise score includes the base response margin instead of isolating alpha-conditioned guidance gain.

## Initialization Ablation

| Initialization | Mean lambda | Mean sigmoid derivative | Endpoint delta |
|---|---:|---:|---:|
| `collapsed_v2_alpha` | 0.00072977 | 0.00072891 | 0.00000251 |
| `no_alpha_transplant_fresh_preference_head` | 0.03420873 | 0.03303817 | 0.00000525 |
| `fresh_alpha_target_gate` | 0.03021964 | 0.02930641 | 0.00000080 |

## Recommendation

- Initialization: `no_alpha_transplant_fresh_preference_head`
- Pairwise logit scale: `50`
- Use base-relative preference gain, a linear per-token lambda floor,
  endpoint sensitivity, and preference-only warm-up before NLL.
- Keep `min_control_lambda_delta=0.001` unchanged.

Full gradient tables, ratios, direction checks, exact sequence-score
sensitivities, hashes, and per-initialization measurements are in JSON.
