# Stage 6 — Integrate the Learned Router into RAD Decoding

## Goal

Add a learned-router decoding mode to RAD without changing the behavior of the original fixed-beta implementation.

## Required modes

```text
base
fixed_beta
heuristic_adaptive
learned_router
```

The existing fixed-beta path must remain reproducible and unchanged.

## Learned-router decoding step

At each generation step:

1. Run GPT-2 Large on the current prefix.
2. Select top-20 candidate tokens.
3. Score each candidate with the frozen sentiment reward model.
4. Normalize candidate base logits and reward scores for router input.
5. Predict \(\beta_t\).
6. Compute RAD guided scores:

\[
s_{t,i}=\ell_{t,i}+\beta_t r_{t,i}.
\]

7. Apply the same sampling/selection procedure as the original RAD implementation.
8. Append the selected token.
9. Save detailed step diagnostics.

## Required beta history

For every output, save:

```json
{
  "step": 0,
  "beta": 31.42,
  "gate": 0.4189,
  "selected_token_id": 123,
  "selected_token": " great",
  "base_entropy": 2.73,
  "reward_mean": 0.11,
  "reward_std": 0.07,
  "reward_gap": 0.14
}
```

At minimum, `step`, `beta`, selected token ID, and selected token text are mandatory.

## CLI additions

Suggested arguments:

```text
--decoding-mode learned_router
--router-checkpoint path/to/best.pt
--beta-max 75
--save-beta-history
--top-k 20
```

Validate that checkpoint configuration matches runtime `top_k`, model IDs, and tokenizer hash.

## Performance considerations

- Use inference mode for both frozen models.
- Batch candidate reward scoring.
- Avoid CPU–GPU transfers inside the token loop.
- Record latency separately for:
  - base LM forward;
  - reward scoring;
  - router forward;
  - total decoding step.

## Regression tests

1. `fixed_beta` output matches the original code under identical seeds.
2. Setting router beta to a constant reproduces fixed-beta decoding.
3. Router checkpoint mismatch produces an explicit error.
4. Beta history length matches generated token count.
5. All saved beta values lie within range.

## Deliverables

```text
run_learned_router.py
router/rad/router_integration.py
router/tests/test_inference_integration.py
```

## Acceptance criteria

- Original RAD behavior is preserved.
- Learned router runs end to end.
- Full beta histories are saved.
- Constant-router equivalence test passes.
- Runtime model/checkpoint compatibility is validated.

## Prompt for the coding AI

Integrate the trained token-level router into the RAD decoding loop as a new mode while preserving the original fixed-beta implementation. At every token, compute top-20 base candidates, reward scores, router beta, and guided scores. Save full beta histories and detailed latency. Add constant-router equivalence and fixed-beta regression tests.
