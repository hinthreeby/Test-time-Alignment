# Stage 5 — Train the Token-Level Router

## Goal

Train only the router on cached token-level features using negative log-likelihood of the gold continuation token.

## Inputs

- Train feature-cache manifest and shards.
- Validation feature-cache manifest and shards.
- Router architecture from Stage 4.
- `beta_max` selected from the fixed-beta validation sweep or a predefined safe bound.

## Loss

For each cached token step:

\[
s_i=\ell_i+\beta r_i,
\]

\[
p_i=\operatorname{softmax}(s)_i,
\]

\[
\mathcal{L}_{NLL}=-\log p_{gold\_index}.
\]

Use only NLL in version 1.

## Optimizer and initial hyperparameters

```yaml
optimizer: AdamW
learning_rate: 0.001
weight_decay: 0.0001
batch_size: 256
epochs: 10
gradient_clip_norm: 1.0
early_stopping_patience: 2
seed: 42
```

Run a small learning-rate sweep only after the base configuration works:

```text
1e-4, 3e-4, 1e-3
```

## Training loop requirements

For every epoch:

1. Train on cached train steps.
2. Evaluate on cached validation steps.
3. Record metrics.
4. Save last checkpoint.
5. Save best checkpoint by validation NLL.
6. Stop early after patience is exceeded.

## Required logs

- train NLL;
- validation NLL;
- train and validation accuracy over candidate set;
- beta mean/std/min/max;
- fraction of beta near zero;
- fraction of beta near beta_max;
- gate mean/std;
- gradient norm;
- learning rate;
- epoch duration.

Suggested collapse thresholds:

```text
near zero: beta < 0.05 * beta_max
near max:  beta > 0.95 * beta_max
```

## Checkpoint contents

```python
{
    "router_state_dict": ...,
    "optimizer_state_dict": ...,
    "epoch": ...,
    "global_step": ...,
    "best_validation_nll": ...,
    "config": ...,
    "dataset_manifest_hashes": ...,
    "model_identifiers": ...,
}
```

## Sanity baselines during training

Evaluate cached validation NLL under:

- beta = 0;
- several fixed beta values;
- router beta.

This reveals whether the router merely learns a constant coefficient.

## Acceptance criteria

- Base LM and RM parameters are absent from optimizer.
- Validation NLL improves over initialization.
- Best checkpoint is saved deterministically.
- Resume training reproduces the same trajectory within expected numerical tolerance.
- Beta statistics are logged per epoch.
- The router does not produce NaNs or collapse silently.

## Prompt for the coding AI

Implement router training from offline feature shards. Optimize only the 40→64→1 router using candidate-level NLL, AdamW, early stopping, gradient clipping, deterministic seeds, best/last checkpoints, and comprehensive beta statistics. Compare validation NLL against fixed-beta baselines during training. Add resume support and tests proving the frozen models are not part of optimization.
