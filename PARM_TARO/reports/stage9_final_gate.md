# Stage 9 Final Gate

## Status

```text
STAGE 9 PASS
```

Pilot V3 authorizes the production recipe but cannot satisfy the final gate. Only the isolated 8,000-train/500-validation production run is accepted.
A NOT_PASS production report may be corrected only by the read-only non-identical constant-alpha audit with the unchanged 0.001 threshold.

## Checks

- `taro_pass`: `true`
- `v2_no_alpha_pass`: `true`
- `pilot_v3_pass_authorization`: `true`
- `v2_alpha_preference_production_present`: `true`
- `v2_alpha_preference_production_pass`: `true`
- `production_required_checks_pass`: `true`
- `production_checkpoint_hash_valid`: `true`
- `production_train_count_8000`: `true`
- `production_validation_count_500`: `true`
- `test_split_unused`: `true`
- `parm_tree_unchanged`: `true`

The failed legacy `v2_alpha` run and all three pilots remain read-only evidence and are never promoted by this gate.
