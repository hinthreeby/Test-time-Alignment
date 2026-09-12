# PARM-TARO server restore summary

Audit commit: `d2912fa11170a272973130696f9bbeef3e6c8af3` on branch `main`.

## Decision

**NOT READY — MISSING ARTIFACTS**

No retraining, 200/500/1500 evaluation, bulk generation, package installation, or model download was performed. Protected baseline paths and historical Stage-9/10 artifacts were not modified.

## Gate results

| Gate | Status | Main evidence |
|---|---|---|
| Source | PARTIAL | Main trees present and Git-clean, but ignored `router_v2/cache` implementation is absent |
| Protected artifacts | FAIL | PBLoRA and three Stage-9 checkpoints missing; archived full tree hashes not reproducible |
| Models | FAIL | Tulu-2-7B/tokenizer and evaluators absent |
| Dataset | PASS | PKU source and deterministic 8000/500/1500 processed splits present |
| Protocol | PASS | Frozen protocol and deterministic val200 manifest present; canonical lock hash verifies |
| Python environment | FAIL | Candidate env differs from historical torch/Python and lacks bitsandbytes |
| GPU | FAIL | RTX 5090-class PCI device present, but no `/dev/nvidia*`; `nvidia-smi` and CUDA matmul fail |
| Real model load | NOT RUN | Blocked upstream |
| Static equivalence | NOT RUN | `BLOCKED_STATIC_EQUIVALENCE` |
| Fresh D1-D7 | NOT RUN | Required Phases 0-10 did not pass |

See `04_missing_artifacts.md` for exact restoration requirements. The final conclusion is driven by both missing irreplaceable checkpoints/source and an unusable GPU/runtime; artifact recovery is the first non-substitutable blocker.
