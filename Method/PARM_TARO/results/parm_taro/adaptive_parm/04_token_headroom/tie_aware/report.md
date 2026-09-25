# Selective Adaptive PARM: tie-aware offline analysis

Scientific verdict: **TOKEN_HEADROOM_EXISTS_BUT_NOT_PREDICTABLE**

## Tie-aware oracle (`epsilon=1e-06`)

- Tied states: 225 / 250
- Unique states: 25 / 250
- Raw `w=0` labels caused only by exact argmax ties: 200 / 222
- Fraction with `w=1` in oracle set: 85.60%

## Actionable states

- Count: 64
- Fixed `w=1` utility: 0.601782081
- Oracle utility: 0.657897686
- Absolute gain: 0.056115605; relative gain: 9.32%
- Prompt-grouped bootstrap CI95: [0.026902224784064035, 0.0856529343546345]

## Tie-aware structure

- Generic informative pairs: 296 (11.84%)
- Generic pairwise accuracy: 0.472973
- Generic mean defined-state Spearman: -0.031168
- Undefined Spearman fraction: 74.40%
- Generic oracle-set accuracy / top-2 coverage: 87.20% / 94.40%
- Stage-B actionable advantage pairwise accuracy: 0.405405
- Stage-B defined-state Spearman: -0.129514

## Selective feasibility

- Stage-A logistic AUROC/AUPRC/balanced accuracy: 1.000000 / 1.000000 / 1.000000
- Fixed / generic / selective / oracle MIP: 0.606883681 / 0.611260984 / 0.602883839 / 0.621249276
- Selective gain: -0.003999842; gap capture: -27.84%
- Intervention rate: 13.20%
- Beneficial / harmful interventions: 36.36% / 60.61%

All features precede scoring. Splits are grouped by unique prompt, and each
outer-fold delta is selected only from inner OOF predictions over training
prompts. This remains a small 10-prompt offline diagnostic.
