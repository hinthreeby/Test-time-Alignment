# Adaptive PARM Fusion foundation test

Status: **PASS**

Canonical implementation:

```text
z = (1 - w) * base_logprobs + w * parm_logprobs
p_adaptive = softmax(z)
```

- CPU only; no Tulu, PBLoRA, Beaver, or router checkpoint loaded.
- 17/17 unit tests passed.
- Scalar, per-batch, and per-token/prefix weights passed.
- FP16, BF16, and FP32 finite-output checks passed.
- Input non-mutation and finite weight-gradient checks passed.
- Maximum FP32 parity error: `4.76837158203125e-07`.
- Midpoint parity error: `0.0`.
- Logit-magnitude equivariance error: `0.0`.
- Protected paths `PARM/`, `router/`, and `Method/RAD/` remained untouched.
