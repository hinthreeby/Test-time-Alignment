# Token-level Adaptive PARM headroom

Verdict: **TOKEN_LEVEL_HEADROOM_WEAK**

- Prompts/cases/states: 10 / 50 / 250
- Mean fixed `w=1` utility: `0.606883681`
- Mean per-state oracle utility: `0.621249276`
- Absolute gain: `0.014365595`
- Relative headroom: `2.37%`
- States preferring `w != 1`: `98.80%`
- States where all weights choose the same next token: `74.40%`
- Prompt-cluster bootstrap gain CI95: `[0.006197225140054439, 0.026451572628612707]`

## Structure probe

- Spearman: `-0.007979` (failed prompt-level baseline `0.006789`)
- Pairwise ranking accuracy: `0.867667` (prompt-level `0.502581`)
- Oracle action accuracy: `43.20%`
- Top-2 action accuracy: `53.20%`
- Learned offline policy MIP: `0.611260984`
- Learned oracle-gap capture: `30.47%`

All bootstrap samples resample unique prompts, not token states. Reward/cost are
labels only. No router or PBLoRA parameter was trained during state collection.
