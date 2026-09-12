# Stage 2 Faithful TARO Top-K Baseline Report

Date: 2026-08-14

## Final status

**STAGE 2 PASS - faithful TARO Top-K baseline implemented**

Stage 2 stops here. Smart Router V2, PARM-TARO, full training, and cache generation were not started.

## Scope and read-only boundaries

Only files under `router_v2/` were modified. The protected trees `router/`, `router/evaluation/`, `PARM/`, and `Method/RAD/` were read only. Existing V1 data/cache trees were also left unchanged.

## Files changed

- `router_v2/__init__.py`
- `router_v2/README.md`
- `router_v2/candidates.py`
- `router_v2/model.py`
- `router_v2/objective.py`
- `router_v2/config.py`
- `router_v2/checkpoint.py`
- `router_v2/configs/taro_topk_nll.json`
- `router_v2/configs/taro_topk_nll_entropy.json`
- `router_v2/schemas/taro_config.schema.json`
- `router_v2/schemas/taro_checkpoint.schema.json`
- `router_v2/tests/test_taro_router.py`
- `router_v2/tests/test_checkpoint_config_device.py`
- `router_v2/reports/stage2_taro_baseline_report.md`

`router_v2/tests/test_v1_regression.py` and `router_v2/reports/baseline_hashes.json` were reused unchanged as the regression oracle.

## Final equations

Gate:

```text
alpha_t = sigmoid(W2 * tanh(W1 * h_topk_t + b1) + b2)
```

Full-vocabulary routing:

```text
z_guided_t = (1 - alpha_t) * z_base_t + alpha_t * z_reward_t
```

Gold-token NLL:

```text
L_NLL = -sum_t log softmax(z_guided_t)[gold_t]
```

Optional Bernoulli entropy:

```text
H(alpha_t) = -alpha_t*log(alpha_t) - (1-alpha_t)*log(1-alpha_t)
L = L_NLL + lambda_H * sum_t H(alpha_t)
```

`taro_topk_nll` requires `entropy_weight=0`. `taro_topk_nll_entropy` adds the configured positive entropy contribution.

## Exact Top-K feature construction

Base and reward candidates are selected independently:

```text
C_base   = TopK(z_base, K)
C_reward = TopK(z_reward, K)
```

No gold token is accepted by or inserted into the selector. Gold IDs are passed only to the objective after full-vocabulary guided logits exist.

For each selected candidate:

```text
u_base_i   = [base_logit_i;   E(base_token_id_i)]
u_reward_j = [reward_logit_j; E(reward_token_id_j)]
```

Pairs preserve `torch.topk` order. Base pairs are flattened first, reward pairs second:

```text
h_topk = concat(flatten(u_base_1..K), flatten(u_reward_1..K))
feature_dim = 2 * K * (d + 1)
```

There is no candidate-wise nonlinear encoder, mean pooling, GELU, LayerNorm, confidence feature, disagreement feature, or history state.

## Dimensions and parameter count

Default config: `K=20`, token embedding dimension `d=32`, vocabulary size `V=50257`, hidden size `128`.

```text
input dimension = 2 * 20 * (32 + 1) = 1320
token embedding = 50257 * 32            = 1,608,224
W1 and b1       = 1320 * 128 + 128      =   169,088
W2 and b2       = 128 * 1 + 1           =       129
total trainable parameters              = 1,777,441
```

State dict keys are exactly `token_embedding.weight`, `gate_mlp.0.weight`, `gate_mlp.0.bias`, `gate_mlp.2.weight`, and `gate_mlp.2.bias`.

## Checkpoint and config schema

Config and checkpoint schemas were incremented from 1 to 2 because the pooled/GELU state dict is structurally incompatible with the flatten/Tanh model. The checkpoint format is now `router_v2.taro_topk_checkpoint`.

Checkpoint schema 2 stores the required method, mode, vocabulary, Top-K, embedding, hidden, entropy, state, training state, and metadata fields. Mirrored architecture fields are validated against the embedded config. Atomic save and no-overwrite behavior remain.

A schema-1 checkpoint fails before model construction with an explicit incompatible-schema error; it is never silently loaded.

## Test results

CPU command:

```bash
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home/jupyter-iec2024se10/miniconda3/envs/genarm/bin/python -m unittest discover -s router_v2/tests -v
```

Result: **31 tests passed in 4.020 seconds**.

- 27 candidate, feature, architecture, routing, objective, gradient, config, device, and checkpoint tests passed.
- 4 V1 regression tests passed, including full tree rehash and V1 output isolation.
- Gradient assertions confirm source base/reward logits receive `grad=None`; token embeddings, W1, and W2 receive non-zero gradients.
- Full-vocabulary cross entropy passes even when gold is outside both Top-K sets.

## Baseline hash verification

All hashes were recomputed from the immutable inventory and matched exactly:

| Tree | Files | SHA-256 |
|---|---:|---|
| `router/` | 100 | `7b5ceafc493c605eb13346c557ad8f768399fcecfe3dceb6ee1137dbf518b127` |
| `router/evaluation/` | 32 | `1af858c224e2d196af925e047b97f8ec1392db7289670ce97dbea2c0e7482fd9` |
| `PARM/` | 156 | `aef1aa5fbc30816a97c1c1bb82d7dbc6554138f8e1a267630b6498d14a02289e` |
| `Method/RAD/` | 110 | `3012ada626eabd9dc72dfde349cd4ffc7d0041ec0404cb796e253974b2f5de31` |
| `dataset/router_cache/rad/` | 2 | `7922216808b1e4614b982d522b7ac73f0c85c1eaf1006484e09d01f03909a36d` |
| `dataset/router_train/rad/` | 2 | `665bef1b2949c45a1ac6d02889ae861723ac16a3e008c809706991c45e4b2ab6` |
| `dataset/rad_benchmark/` | 4 | `46ed6143daa76fc66a5f6f06e1e3901dde36c4c283698d33e3517accca3880a6` |

The regression suite also matched the two recorded `dataset/RAD_train/` trees and every critical-file hash in the baseline inventory.

## Differences from the previous Stage 2 implementation

- Replaced one base-selected shared token set with independent base and reward Top-K sets.
- Removed gold insertion and removed gold from model/selector signatures.
- Replaced `[token_embedding, base_logit, reward_logit]` candidate encoding with separate `[logit, token_embedding]` streams.
- Removed candidate Linear/GELU/LayerNorm and mean pooling.
- Replaced the pooled GELU gate with `Linear(2*K*(d+1),128) -> Tanh -> Linear(128,1)`.
- Exposed base and reward token IDs/logits separately through `TAROTopKBatch` and `TARORouterOutput.topk`.
- Added an explicit entropy contribution in objective diagnostics.
- Upgraded incompatible config/checkpoint schemas to version 2.

## Remaining deviations

Within the corrected Stage 2 definition, no known architecture or objective deviation remains. Full TARO training, production cache integration, official-result reproduction, and physical-GPU validation are intentionally outside this stage. The agent runtime did not verify the host RTX 3080.

## Exact host commands

Run from the workspace:

### 1. NVIDIA driver/GPU

```bash
cd '/home/jupyter-iec2024se10/Reward Decoding'
nvidia-smi
```

### 2. PyTorch CUDA verification

```bash
PY=/home/jupyter-iec2024se10/miniconda3/envs/genarm/bin/python
$PY -c "import torch; print({'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'available': torch.cuda.is_available(), 'count': torch.cuda.device_count(), 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
```

### 3. CPU unit and regression tests

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 $PY -m unittest discover -s router_v2/tests -v
```

### 4. CUDA forward/backward smoke test

```bash
$PY - <<'PY'
from dataclasses import replace
import torch
from router_v2.config import TARORouterConfig
from router_v2.model import build_taro_router
from router_v2.objective import compute_taro_objective

cfg = TARORouterConfig.load_json('router_v2/configs/taro_topk_nll.json')
cfg = replace(cfg, device='cuda', allow_cpu_fallback=False)
model = build_taro_router(cfg).train()
device = next(model.parameters()).device
base = torch.randn(2, cfg.vocab_size, device=device, requires_grad=True)
reward = torch.randn(2, cfg.vocab_size, device=device, requires_grad=True)
gold = torch.tensor([10, 20], device=device)
output = model.forward_from_full_logits(base, reward)
objective = compute_taro_objective(output, gold, cfg)
objective.total_loss.backward()
router_grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
print({'device': str(device), 'loss': objective.total_loss.item(), 'feature_dim': model.feature_dim, 'base_grad': base.grad, 'reward_grad': reward.grad, 'router_grad_l1': router_grad})
PY
```

Expected: device is `cuda:0`, source grads are `None`, and `router_grad_l1` is positive.

### 5. CUDA checkpoint save/load smoke test

```bash
$PY - <<'PY'
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from router_v2.checkpoint import load_taro_checkpoint, save_taro_checkpoint
from router_v2.config import TARORouterConfig
from router_v2.model import build_taro_router

cfg = replace(TARORouterConfig.load_json('router_v2/configs/taro_topk_nll.json'), device='cuda', allow_cpu_fallback=False)
model = build_taro_router(cfg).eval()
base = torch.randn(2, cfg.vocab_size, device='cuda')
reward = torch.randn_like(base)
before = model.forward_from_full_logits(base, reward)
with TemporaryDirectory() as directory:
    path = Path(directory) / 'taro-schema2.pt'
    save_taro_checkpoint(model, path, training_state={'step': 0}, metadata={'smoke': True})
    loaded, payload = load_taro_checkpoint(path, map_location='cuda')
    after = loaded.eval().forward_from_full_logits(base, reward)
    torch.testing.assert_close(after.alpha, before.alpha, rtol=0, atol=0)
    torch.testing.assert_close(after.guided_logits, before.guided_logits, rtol=0, atol=0)
    print({'schema': payload['schema_version'], 'format': payload['format'], 'device': str(next(loaded.parameters()).device)})
PY
```

### 6. GPU monitoring

```bash
watch -n 1 nvidia-smi
# or
nvidia-smi dmon -s pucm -d 1
```
