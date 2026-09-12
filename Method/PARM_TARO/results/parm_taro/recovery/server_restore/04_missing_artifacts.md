# Missing artifacts and exact unblock requirements

## Blocking artifacts

1. Restore the original `router_v2/cache/` package, at minimum `__init__.py`, `features.py`, `io.py`, `schema.py`, and `inventory.py`. It is referenced throughout the project but was ignored by the broad `cache/` Git rule and is absent from Git history, remote `main`, local mounts, and caches.
2. Restore `models/tulu-2-7b/` matching the archived model shard hashes (index `e572e08c…`, shard 1 `6d90e535…`, shard 2 `3eb1b183…`). The exact Hugging Face revision is UNKNOWN.
3. Restore PBLoRA at `results/parm_taro/checkpoints/parm_pku_pblora/`, tree SHA `f26682e3…`; its `adapter_model.safetensors` must be 25,233,088 bytes and SHA `10127909…`.
4. Restore the three Stage-9 checkpoints with exact hashes listed in `02_protected_artifact_audit.json`.
5. Before Stage-10 reproduction, restore both Beaver evaluators and Safe-RLHF source matching the frozen tree hashes.
6. Restore host access to `/dev/nvidia*` and make `nvidia-smi` work.
7. Resolve the environment only after GPU repair. Retained provenance identifies Python 3.10.20, torch 2.2.2+cu121, transformers 4.39.3, PEFT 0.10.0, accelerate 0.29.2, and bitsandbytes 0.43.1. Because this server appears to use a newer RTX 5090/GB202 device, do not blindly install the old cu121 torch build: first prove architecture support, or record and validate a minimal cu128 compatibility exception in an isolated environment.

Do not download a current/latest model revision. Supply either the old model directories or the exact revisions plus files that verify against the archived hashes. The root filesystem currently has only about 9.5 GiB free, insufficient for these model assets; free or mount adequate storage first.

## Verification commands after placing a backup

```bash
cd /home/jupyter-iec2024se10/Test-time-Alignment
sha256sum results/parm_taro/training/taro/best.pt \
  results/parm_taro/training/v2_alpha_preference/final.pt \
  results/parm_taro/training/v2_no_alpha/best.pt
/home/jupyter-iec2024se10/miniforge3/envs/tta/bin/python \
  PARM_TARO/recovery/server_restore_audit.py
```

For tree artifacts, rerun the audit rather than assuming a copied directory is correct.
