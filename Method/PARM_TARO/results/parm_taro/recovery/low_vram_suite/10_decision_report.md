# PARM-TARO low-VRAM decision

Decision level: **GO ROUTER**

Meaningful scored oracle MIP headroom

Teacher-forced NLL is diagnostic only. Candidate-pool regret is algebraically dependent on candidate MIP.

## Root causes

| RC | Hypothesis | Status | Evidence |
|---|---|---|---|
| RC1 | PBLoRA missing/wrong | NOT SUPPORTED | 01_artifact_audit.json |
| RC2 | Alpha not entering PBLoRA | NOT SUPPORTED | ../pblora_repro/epoch1_validation/alpha_probe_summary.json |
| RC3 | Weak guide/preference ranking | NOT SUPPORTED | 02_guide_utility_summary.json |
| RC4 | Model/tokenizer/checkpoint provenance wrong | NOT SUPPORTED | 01_artifact_audit.json |
| RC5 | Fusion mismatch with author PARM | CONFIRMED | 03_summary.json |
| RC6 | Scale/temperature confound | WEAKLY SUPPORTED | 03_fusion_scale.csv |
| RC7 | Router checkpoint wrong/random | NOT TESTED | historical Stage-10 only |
| RC8 | Router ignores alpha | NOT TESTED | historical Stage-10 only |
| RC9 | NLL pushes lambda to zero | STRONGLY SUPPORTED | 03_gradient_summary.csv |
| RC10 | Gate saturation | NOT TESTED | router checkpoint unavailable |
| RC11 | Feature normalization/cache issue | NOT TESTED | router cache/router checkpoint unavailable |
| RC12 | Teacher-forced/autoregressive exposure shift | NOT SUPPORTED | 05_exposure_shift_summary.json |
| RC13 | No adaptive headroom | NOT TESTED | 04_headroom_summary.json; 09_metrics_summary.json |
| RC14 | Headroom exists but router fails | WEAKLY SUPPORTED | 04_headroom_summary.json |
| RC15 | Optimal lambda varies by alpha | WEAKLY SUPPORTED | 04_headroom_summary.json |
| RC16 | Evaluator wiring bad | NOT SUPPORTED | 07_scorer_summary.json |
