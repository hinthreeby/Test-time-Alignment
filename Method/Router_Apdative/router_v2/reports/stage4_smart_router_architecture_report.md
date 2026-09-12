# Stage 4 Smart Router V2 Architecture Report

Generated: 2026-08-15
Reviewed: 2026-08-16

## Final Status

**STAGE 4 PASS - ready for Stage 5 training**

This status covers architecture, feature toggles, checkpoint contracts, and
synthetic tests only. Router training, RAD evaluation, and PARM-TARO integration
remain later stages.

Stage 3 remains PASS with 228 real FP32 cache shards. No cache shard or Stage 3
manifest was modified for Stage 4.

## Architecture

For each independently selected base/guide Top-K candidate:

~~~text
u_base_i  = [base_logit_i; token_embedding(base_token_i)]
u_guide_j = [guide_logit_j; token_embedding(guide_token_j)]

e_base_i  = phi_base(u_base_i)
e_guide_j = phi_guide(u_guide_j)
~~~

`phi_base` and `phi_guide` are separate candidate encoders. Each stream is
pooled with mean and max, then concatenated. The default candidate output is:

~~~text
4 * candidate_hidden_dim = 4 * 64 = 256 dimensions
~~~

The confidence/disagreement group contains exactly 11 current-token values:

~~~text
base entropy, guide entropy
base margin, guide margin
JS divergence
Top-1 agreement, Top-K overlap
base std, guide std
base range, guide range
~~~

Position is a separate scalar `position / max_position`. The cached
rank-correlation field is intentionally not inserted into this documented
11-value group.

The router-only state is:

~~~text
q_t = GRUCell(
  [lambda_(t-1), selected_score_(t-1), base_entropy_(t-1), JS_(t-1)],
  q_(t-1)
)
~~~

`selected_score[t]` supplied for a training sequence is shifted internally and
can affect only token `t+1` and later. If confidence is disabled in a custom
ablation, entropy and JS are also zero-masked in the GRU input.

Preference alpha is encoded by a separate two-layer MLP. It may be supplied per
sequence or per token. No default PARM behavior or PARM model loading is part of
Stage 4. `preference_dim` is not fixed to one: it must be set to `len(alpha)`
for later multi-objective use, such as two for a two-objective alpha. The
current RAD `v2_topk_state_history` preset disables preference input entirely.

Enabled groups are concatenated and routed through:

~~~text
LayerNorm
Linear(feature_dim, 128) -> GELU
Linear(128, 64) -> GELU
Linear(64, 1) -> Sigmoid
~~~

Tanh remains available by config. LayerNorm becomes identity for a valid
single-scalar custom ablation so that the scalar cannot be normalized away.

The universal routing strength and TARO-compatible full-logit equation are:

~~~text
gate_t   = sigmoid(fusion_t)
lambda_t = lambda_max * gate_t
guided_t = base_t + lambda_t * (guide_t - base_t)
~~~

`lambda_t` has shape `[..., 1]`; the same scalar is broadcast over every
vocabulary item. Base/guide full logits and Top-K logits are detached, so only
Smart Router parameters receive gradients.

This interpolation equation is retained specifically for TARO/RAD. A later
`PARM_TARO/` integration must not reuse it blindly; PARM requires its own
log-probability guidance semantics in that separate implementation.

## Feature Toggles

Named presets pin these cumulative variants:

| Variant | Candidate | Confidence | Position | History | Preference |
|---|---:|---:|---:|---:|---:|
| `taro_topk` | faithful flattened TARO | no | no | no | no |
| `v2_topk_confidence` | token-aware | yes | no | no | no |
| `v2_topk_confidence_position` | token-aware | yes | yes | no | no |
| `v2_topk_state_history` | token-aware | yes | yes | yes | no |
| `v2_topk_state_history_alpha` | token-aware | yes | yes | yes | yes |

`taro_topk` remains the existing `TAROTokenRouter`; it was not reimplemented or
changed. Arbitrary group combinations use `variant=custom`. Candidate mode can
be `disabled`, `taro_flatten`, or `token_aware_mean_max`.

## Dimensions And Parameters

Production config values use vocabulary 50,257, token embedding 32, candidate
hidden 64, GRU hidden 32, and preference embedding 16.

| Smart preset | Fusion input | Parameters |
|---|---:|---:|
| `v2_topk_confidence` | 267 | 1,655,735 |
| `v2_topk_confidence_position` | 268 | 1,655,865 |
| `v2_topk_state_history` | 300 | 1,663,673 |
| `v2_topk_state_history_alpha` | 316 | 1,666,057 |

Most parameters are the isolated 50,257 by 32 token embedding. No base/guide
backbone parameter or hidden state is included.

## Checkpoint Contract

Smart Router uses a separate atomic, no-overwrite checkpoint:

~~~text
format: router_v2.smart_router_checkpoint
schema_version: 1
model_class: SmartTokenRouter
method_label: SMART_ROUTER_V2
~~~

The payload mirrors variant, feature toggles, lambda maximum, full config,
state dict, training state, and metadata. TARO checkpoints fail Smart Router
validation clearly and are never silently loaded.

## Files Added Or Changed

~~~text
router_v2/smart_config.py
router_v2/smart_model.py
router_v2/smart_checkpoint.py
router_v2/configs/v2_topk_confidence.json
router_v2/configs/v2_topk_confidence_position.json
router_v2/configs/v2_topk_state_history.json
router_v2/configs/v2_topk_state_history_alpha.json
router_v2/schemas/smart_router_config.schema.json
router_v2/schemas/smart_router_checkpoint.schema.json
router_v2/scripts/smoke_smart_router.py
router_v2/tests/test_smart_router.py
router_v2/tests/test_checkpoint_config_device.py
router_v2/__init__.py
router_v2/README.md
router_v2/reports/stage4_smart_router_architecture_report.md
~~~

## Verification

Full CPU suite:

~~~text
Ran 89 tests in 4.098s
OK
~~~

The 19 Smart Router tests cover every group on/off, token identity, candidate
permutation, confidence perturbation, position perturbation, causal history,
future-score isolation, confidence masking inside history, configurable
multi-objective preference dimensions, production smoke config dimensions,
alpha shuffle, universal lambda, frozen inputs, separate checkpoints,
no-overwrite, and TARO checkpoint rejection.

Production-shape CPU forward/backward smoke:

~~~text
status: PASS
variant: v2_topk_state_history_alpha
feature_dim: 316
parameters: 1,666,057
lambda shape: [2, 4, 1]
router state shape: [2, 4, 32]
guided logits shape: [2, 4, 50,257]
base/guide input gradients: None
~~~

Protected regression hashes still match the recorded baseline:

| Protected tree | SHA-256 |
|---|---|
| `router/` | `7b5ceafc493c605eb13346c557ad8f768399fcecfe3dceb6ee1137dbf518b127` |
| `router/evaluation/` | `1af858c224e2d196af925e047b97f8ec1392db7289670ce97dbea2c0e7482fd9` |
| `PARM/` | `aef1aa5fbc30816a97c1c1bb82d7dbc6554138f8e1a267630b6498d14a02289e` |
| `Method/RAD/` | `3012ada626eabd9dc72dfde349cd4ffc7d0041ec0404cb796e253974b2f5de31` |
| `dataset/router_cache/rad/` | `7922216808b1e4614b982d522b7ac73f0c85c1eaf1006484e09d01f03909a36d` |
| `dataset/router_train/rad/` | `665bef1b2949c45a1ac6d02889ae861723ac16a3e008c809706991c45e4b2ab6` |
| `dataset/rad_benchmark/` | `46ed6143daa76fc66a5f6f06e1e3901dde36c4c283698d33e3517accca3880a6` |

## Host Commands

~~~bash
cd "/home/jupyter-iec2024se10/Reward Decoding"
export PY=/home/jupyter-iec2024se10/miniconda3/envs/cd/bin/python
export PYTHONDONTWRITEBYTECODE=1

CUDA_VISIBLE_DEVICES="" "$PY" -m unittest discover -s router_v2/tests -v

nvidia-smi
"$PY" -m router_v2.scripts.smoke_smart_router \
  --device cuda --no-cpu-fallback

watch -n 1 nvidia-smi
~~~

## Deferred By Scope

- Smart Router training and regularizers;
- cache-backed sequence dataset assembly;
- RAD generation/evaluation;
- PARM alpha integration;
- same-average-lambda controls.

No Smart Router training result is claimed in Stage 4.
