# Stage 4 — Implement the Minimal TARO-Like RAD Router

## Goal

Implement the simplest trainable token-level router that is close to TARO while remaining native to RAD candidate scoring.

## Architecture

Input features:

\[
h_t=[\operatorname{Norm}(\ell_t);\operatorname{Norm}(r_t)]\in\mathbb{R}^{40}.
\]

Router:

\[
u_t=\tanh(W_1h_t+b_1),
\]

\[
a_t=\sigma(W_2u_t+b_2),
\]

\[
\beta_t=\beta_{max}a_t.
\]

Dimensions:

```text
40 → 64 → 1
```

## Reference implementation

```python
class RADTokenRouter(nn.Module):
    def __init__(self, beta_max: float):
        super().__init__()
        self.beta_max = beta_max
        self.mlp = nn.Sequential(
            nn.Linear(40, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, base_logits, reward_scores):
        base = standardize(base_logits)
        reward = standardize(reward_scores)
        features = torch.cat([base, reward], dim=-1)
        gate = torch.sigmoid(self.mlp(features))
        beta = self.beta_max * gate
        return beta, gate
```

The output shape must broadcast over the 20 candidate scores.

## Guided scoring

\[
s_{t,i}=\ell_{t,i}+\beta_t r_{t,i}.
\]

Implementation:

```python
beta, gate = router(base_logits, reward_scores)
guided_scores = base_logits + beta * reward_scores
```

Do not normalize the values used in the guided-score equation unless RAD itself requires it. Normalization is only for router input in version 1.

## Initialization

Initialize the final bias so the initial beta is near a reasonable fixed baseline rather than exactly `beta_max / 2` if desired.

For a target initial coefficient `beta_init`:

\[
b_2=\operatorname{logit}(\beta_{init}/\beta_{max}).
\]

Keep this configurable and document the choice.

## Required tests

1. Output beta is in `[0, beta_max]`.
2. Input and output shapes are correct.
3. Constant shifts of all base logits do not change normalized router features.
4. Constant shifts of all reward scores do not change normalized router features.
5. Gradients reach router parameters.
6. No gradient reaches cached features or frozen models.
7. Saved and loaded checkpoints produce identical outputs.

## Deliverables

```text
router/model.py
router/features.py
router/checkpoint.py
router/tests/test_router.py
```

## Acceptance criteria

- Architecture is exactly 40 → 64 → 1.
- Tanh hidden activation and sigmoid gate are used.
- `beta_max` is configurable.
- Router is independent of GPT-2 internals and consumes cached tensors.
- All unit tests pass.

## Prompt for the coding AI

Implement the minimal TARO-like RAD router with normalized 20-dimensional base logits, normalized 20-dimensional reward scores, an MLP 40→64→1, Tanh, and Sigmoid multiplied by `beta_max`. Keep normalization separate from guided-score computation. Add robust shape, range, gradient, invariance, and checkpoint tests. Do not add entropy, recurrence, KL control, or extra features.
