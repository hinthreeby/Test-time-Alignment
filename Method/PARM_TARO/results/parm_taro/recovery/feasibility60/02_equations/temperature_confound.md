# Temperature/scale confound

**FUSION_SCALE_CONFOUND**

- Synthetic states: 64 (vocabulary 257, seed 2026).
- `max|F4(lambda=1)-F2| = 0.0`.
- `max|F3(lambda=1)-F2| = 3.6538293989184556`.
- F1 paper beta=1 and F3 lambda=1 coincide.
- F2 author and F4 normalized lambda=1 coincide.
- Division by `1+lambda` changes effective temperature while preserving ranking
  at fixed lambda; it is not generally removed by log-probability normalization.

Real five-prompt equation traces were not run because the exact PBLORA runtime
gate did not pass. Production source was not modified.
