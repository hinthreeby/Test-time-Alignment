# Offline Adaptive-PARM oracle dataset

Status: **PASS**

This dataset was derived only from frozen Phase 08 generations and Phase 09
scores. No generation, model loading, rescoring, or renormalization occurred.
The primary per-case utility is Phase 09 MIP. Hypervolume remains a global
set-level evaluation metric and is not included as a decision label.

## Results

- Cases: 60 across 12 prompts and 5 alphas
- Candidate weights per case: 8
- Best global fixed weight: `1.00000000` (`normalized_guide_only`)
- MIP(best global): `0.646876824`
- MIP(best per-alpha): `0.653787492`
- MIP(per-case oracle): `0.719490238`
- Oracle gap: `0.072613414`
- Relative oracle headroom: `11.2252%`

The recomputed values agree with the earlier approximate sanity values
(`~0.647` global, `~0.720` oracle); no discrepancy requires explanation.

## Best fixed weight per alpha

| Alpha (helpfulness, harmlessness) | Weight | Mean MIP | Source candidate |
|---|---:|---:|---|
| (1.00, 0.00) | 1.00000000 | 0.619475451 | normalized_guide_only |
| (0.75, 0.25) | 1.00000000 | 0.604304529 | normalized_guide_only |
| (0.50, 0.50) | 0.33333333 | 0.612926633 | normalized_0.5 |
| (0.25, 0.75) | 1.00000000 | 0.688217205 | normalized_guide_only |
| (0.00, 1.00) | 1.00000000 | 0.744013642 | normalized_guide_only |

## Audits

- `cases_60`: PASS
- `unique_prompts_12`: PASS
- `five_alpha_values`: PASS
- `twelve_cases_per_alpha`: PASS
- `eight_weights_per_case`: PASS
- `no_duplicate_case_weight`: PASS
- `all_scores_finite`: PASS
- `alpha_order_helpfulness_harmlessness`: PASS
- `phase08_phase09_generation_identity`: PASS
- `method_lambda_weight_mapping`: PASS
- `phase09_mip_formula_exact`: PASS
- `phase09_candidate_regret_exact`: PASS
- `phase09_harmlessness_is_negative_cost`: PASS

Oracle ties are retained in `oracle_by_case.csv`; the primary label uses the
minimum-trust weight among exact/tolerance ties. `oracle_gap_capture` is stored
for every fixed candidate in `best_global_weight.csv` using the requested
global formula. Normalization metadata and hashes of all source artifacts are
frozen in `oracle_summary.json`.
