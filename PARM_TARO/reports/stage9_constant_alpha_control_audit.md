# Stage 9 Constant-Alpha Control Audit

## Status

```text
PASS
```

Constant alpha: `[0.5, 0.5]`
Identical tasks: `500`
Non-identical tasks: `2000`
All-task token-weighted delta: `0.0008731123`
Non-identical token-weighted delta: `0.0010913904`
Unchanged threshold: `0.001`

The checkpoint, source run status, production output tree, frozen models, PBLORA, PARM tree, and Stage 8 data are read-only inputs to this audit.

## Checks

- `source_failed_only_constant_control`: `true`
- `constant_alpha_is_half_half`: `true`
- `full_validation_tasks_recomputed`: `true`
- `identical_task_count_is_500`: `true`
- `non_identical_task_count_is_2000`: `true`
- `identical_alpha_delta_approximately_zero`: `true`
- `midpoint_group_recognized_as_identical`: `true`
- `other_alpha_groups_are_informative`: `true`
- `all_task_delta_matches_original_report`: `true`
- `non_identical_constant_delta_passes_unchanged_threshold`: `true`
- `all_task_gate_was_diluted_by_identical_controls`: `true`
- `shuffled_gate_remains_passed`: `true`
- `endpoint_gate_remains_passed`: `true`
- `checkpoint_unchanged`: `true`
- `source_run_status_unchanged`: `true`
- `production_tree_unchanged`: `true`
- `frozen_artifacts_unchanged`: `true`
- `all_gradients_none`: `true`
- `test_split_unused`: `true`
