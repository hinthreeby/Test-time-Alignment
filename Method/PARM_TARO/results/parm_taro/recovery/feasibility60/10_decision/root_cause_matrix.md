# Root-cause matrix

| Hypothesis | Evidence | Status | Confidence | Relevant artifact | Recommended action |
|---|---|---|---|---|---|
| RC1: PBLoRA checkpoint missing/load wrong | No PBLORA adapter weights found under project or /home search. | **CONFIRMED** | HIGH | `01_pblora/pblora_load.json` | Reproduce exact custom PBLORA under recovered protocol before any method claim. |
| RC2: Alpha does not enter PBLORA | Runtime probe forbidden by RC1. | **UNTESTABLE** | HIGH | `01_pblora/pblora_alpha_probe.csv` | Run five-alpha same-prefix probe after RC1 clears. |
| RC3: PBLORA guide weak/misranks | 20-pair utility audit blocked by RC1. | **UNTESTABLE** | HIGH | `03_guide_utility/report.md` | Measure validation preference accuracy and margins. |
| RC4: Model/tokenizer/template/provenance wrong | Tulu files match pinned public revision; custom adapter provenance absent; author and Stage-10 templates differ. | **PARTIAL** | HIGH | `00_preflight/path_audit.json` | Freeze exact adapter provenance and explicitly select template. |
| RC5: Author/PARM-TARO fusion mismatch | Author F2 divides summed log-probs by 2; current F3 does not. | **CONFIRMED** | HIGH | `02_equations/temperature_confound.md` | Treat paper, author, and current formulations as distinct comparators. |
| RC6: Lambda confounds guidance and temperature | F3 changes coefficient sum with lambda; F4 removes this scale change. Synthetic entropy matrix confirms distinct distributions. | **CONFIRMED** | HIGH | `02_equations/equation_matrix.csv` | Test F4 diagnostically after PBLORA gate; do not patch production yet. |
| RC7: Router checkpoint missing/wrong/random | All expected TARO/V2 checkpoint files are absent; retained JSON logs are not weights. | **CONFIRMED** | HIGH | `04_router/router_checkpoint_audit.json` | Reproduce/load checkpoint before router claims. |
| RC8: Router ignores alpha | No valid router checkpoint to probe. | **UNTESTABLE** | HIGH | `06_alpha/report.md` | Run same-state alpha counterfactual after RC7 clears. |
| RC9: Gold-token NLL pushes lambda to zero | Historical evidence suggests this, but requested fresh 20-case gradient audit is blocked. | **LIKELY** | MEDIUM | `05_gradients/report.md` | Re-test both F3 and F4 gradients on validation. |
| RC10: Sigmoid/lambda saturation or scale error | Source gives bounded sigmoid; trained raw/bias unavailable without checkpoint. | **PARTIAL** | MEDIUM | `04_router/router_checkpoint_audit.json` | Audit raw gate, bias, floor/max, saturation after RC7 clears. |
| RC11: Feature normalization/cache scale wrong | Reconstructed cache tests exist, but no matching trained checkpoint/state distribution can be checked. | **UNTESTABLE** | MEDIUM | `04_router/router_checkpoint_audit.json` | Compare training cache statistics to live features. |
| RC12: Teacher-forcing exposure shift | Both required model artifacts are absent. | **UNTESTABLE** | HIGH | `08_exposure/report.md` | Run 10-prompt paired-prefix audit last. |
| RC13: Little adaptive headroom | Fatal PBLORA gate prevents fixed-lambda/oracle sweep. | **UNTESTABLE** | HIGH | `07_lambda_sweep/headroom.md` | Do not conclude NO-GO; run validation-60 only after all gates. |
| RC14: Headroom exists but router cannot learn | Neither oracle headroom nor router policy can be measured. | **UNTESTABLE** | HIGH | `07_lambda_sweep/headroom.md` | Compare oracle, best global, dynamic, same-average. |
| RC15: Optimum varies mainly by alpha | Per-alpha sweep blocked. | **UNTESTABLE** | HIGH | `07_lambda_sweep/oracle_analysis.csv` | Estimate per-alpha optimum on fixed manifest. |
| RC16: Scorer wiring/model wrong | Reward weights verified; cost weights/tree complete but revision marker absent; Safe-RLHF source commit/runtime unavailable. | **PARTIAL** | HIGH | `09_scores/scorer_sanity.json` | Restore exact Safe-RLHF provenance and run deterministic sanity pairs. |
