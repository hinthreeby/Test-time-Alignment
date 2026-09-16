# Lambda=1 parity micro-diagnostic

## Conclusion

**LAMBDA1_PARITY_FAIL**

The original PARM author path and PARM-TARO are not exact normalized-distribution
equivalents at lambda=1. Author `ModelArithmetic` evaluates `M_base + M_reward`
as `log_softmax((log p_base + log p_guide) / 2)`, while PARM-TARO evaluates
`log_softmax(log p_base + log p_guide)`. The common factor 1/2 preserves ranking
and therefore normally preserves greedy tokens, but changes probabilities.

Stage-10 `parm_static` is a different comparator: it uses the PARM-TARO decoder
with `FixedLambdaProvider(1.0)`. Its parity with router-bypass lambda=1 is exact
by construction; that does not establish parity with the original PARM author
generation distribution.

## Measurements

- Mode: `deterministic_cpu_source_equation_probe`
- Real model executed: `False`
- max_abs_logprob_diff: `2.141449130237178`
- mean_abs_logprob_diff: `0.7032898737311515`
- top1_agreement: `True`
- topk_overlap: `1.0`
- numerical tolerance: `1e-05`
- Full tensor trace: `results/parm_taro/recovery/quick100/lambda1_token_traces.pt`

## Gate

Router experiments must remain stopped for claims that require original-PARM
distribution parity. No protected source was changed and no retraining ran.
