# Custom artifact reproduction plan

All historical custom checkpoints are classified **NEEDS_REPRODUCTION**, not replaceable by public lookalikes and not byte-reproducible from backup.

| Artifact | Historical output | Retained training entry/config | Resolved protocol | Resume design | Remaining uncertainty |
|---|---|---|---|---|---|
| PBLoRA | `results/parm_taro/checkpoints/parm_pku_pblora/` | protected `PARM/code/training/train_pref_arm.py`; `PARM/code/training/run.sh` | PKU SafeRLHF, Tulu path from actual retained adapter audit, LoRA r/r2=4, alpha=8, dropout=.05, safe/help betas=.01, DPO beta=.5, LR 5e-4, 2 epochs, batch 4, grad-accum 8, cosine, warmup 20, wd .05 | Recovery entrypoint passes HF Trainer checkpoint resume; periodic `checkpoint-*` contains model/optimizer/scheduler/trainer/RNG state | Historical seed absent; author script default model conflicts with retained Tulu-based adapter, so explicit protocol approval is required |
| TARO | `results/parm_taro/training/taro/best.pt` | `train_parm_router`; `train_stage9_taro.json` | seed 2026; train 8000/validation 500; 1 epoch; LR 1e-4; grad-accum 8; wd 0; save every 250; max length 512/continuation 128; 4-bit NF4 double; shared backbone | Existing trainer's `--resume`; atomic `latest.pt`, model/optimizer/scheduler/epoch/global-step/best/RNG payload | Requires reproduced PBLoRA and public base/data |
| V2 no-alpha | `results/parm_taro/training/v2_no_alpha/best.pt` | `train_parm_router`; `train_stage9_v2_no_alpha.json` | seed 2026 and same retained data/runtime family as TARO | Existing `--resume`; recovery-only output and `last.pt` alias | Requires reproduced PBLoRA and cache/data validation |
| V2 full-alpha, old NLL objective | `results/parm_taro/training/v2_alpha_preference/final.pt` | `train_alpha_preference_production`; retained production/resolved configs | seed 2026; warmup 1 + quality 1 epoch; LR 1e-4; grad-accum 4; wd 0; checkpoint every 100 | Existing production `--resume`; recovery-only output | Depends on missing no-alpha and pilot V2/V3 checkpoints; exact recovery config remains gated |
| Recovered V2 | new recovery output only | future recovery entry/config | residual lambda/trust region must not be selected before fresh D1-D7 | wrapper is deliberately gated but already accepts `--resume` | Fix family/config not authorized in this task |

Reproduction order is fixed: public restore → PBLoRA → PARM static validation → TARO old → V2 no-alpha old → V2 full-alpha old → reproduce low-lambda collapse → fresh D1-D7 → recovered router → val200 → val500 → freeze → one final test evaluation.

Historical behavioral checks are comparison targets, not tuning targets: static PARM HV≈0.425525, MIP≈0.634510, PCS≈0.850811, Regret≈0.042023; old full-alpha mean lambda≈0.010731. If the old objective does not recover the low-lambda phenomenon, stop before applying a fix.

No training, model inference, staged validation, or test evaluation was executed.
