# Selective Lookahead PARM probe

## Facts

- Status: `COMPLETE`
- Primary subset: 64 actionable states; secondary subset: all 250 states.
- Best internal score/horizon: `base`, H=2.
- Pairwise accuracy: 0.527027; defined-state Spearman: 0.079861.
- Policy utility: 0.620104; fixed w=1: 0.601782.
- Gain: 0.018322; oracle-gap capture: 32.65%.
- Harmful decision fraction: 0.4.

## Interpretation

Verdict: **LOOKAHEAD_PARM_WEAK**. Longer horizons are evaluated as direct, parameter-free internal branch scores, not learned predictors.

## Limitations

This reuses 10 prompts and existing deterministic one-step counterfactual rollouts. Internal likelihood is not an external alignment evaluator, and token-work estimates are not latency measurements.
