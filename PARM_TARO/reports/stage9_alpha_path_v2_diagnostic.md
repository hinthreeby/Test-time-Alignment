# Stage 9 Pilot V2 Alpha-Path Diagnostic

## Status

```text
PASS
```

## Feature Scale

| Metric | Mean |
|---|---:|
| Preference activation norm | 0.44692644 |
| Non-preference feature norm | 19.31342838 |
| Preference/state first-layer ratio | 0.04959710 |

## Logit Sensitivity

- Endpoint pre-sigmoid delta: `0.02894928`
- Endpoint lambda delta: `0.00086049`
- Lambda/logit compression ratio: `0.02966733`

## Conclusion

preference contribution is small relative to state fusion; sigmoid compression attenuates the expressed logit delta; lambda-space sensitivity spends most gradient on the shared final head instead of a dedicated alpha pathway.

Pilot V3 uses the V2 checkpoint unchanged as its source, adds a
normalized learned alpha-state residual before sigmoid, and moves
endpoint sensitivity to logit space. Full retraining is not authorized.
