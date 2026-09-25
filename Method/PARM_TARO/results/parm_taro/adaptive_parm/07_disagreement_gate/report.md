# Disagreement-gated PARM

- Prompts: 200 fresh validation prompts; cases: 1000.
- Gate MIP: 0.564791; fixed PARM MIP: 0.603275.
- Difference: -0.038484; prompt-bootstrap CI95: [-0.05749304594069633, -0.019928065202511377].
- Gate HV: 0.306488; fixed PARM HV: 0.374429.
- Latency/token ratio: 1.644; intervention rate: 16.06%.
- Greedy gate output identical to Base: `True` (an algebraic consequence of the frozen rule).

Verdict: **ADAPTIVE_PARM_NO_GO**. The primary rule has no learned or tuned parameters.
