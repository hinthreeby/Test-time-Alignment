# Feasibility-60 (prepared, not executed)

This is a deterministic 60-case **validation-only** manifest: the same 12
prefixes are crossed with five alpha values, enabling paired comparisons. It
does not reference the Stage-10 1,500-case test split.

Prepared diagnostics cover RC3 guide quality, RC5 fusion mismatch, RC6 scale
confounding, RC8 alpha sensitivity, RC9 NLL gradient direction, RC13 adaptive
headroom, RC14 router failure despite headroom, and RC15 alpha-specific optima.

Do not launch the heavy generation/scoring job until the PBLoRA alpha probe
passes, free GPU memory is at least 14,336 MiB, and the user explicitly starts
it. Reward and cost evaluators must not be co-loaded with the low-VRAM probe.

The existing resumable entrypoint remains:

```bash
FEASIBILITY_PYTHON=/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10 \
CUDA_VISIBLE_DEVICES=0 bash PARM_TARO/recovery/scripts/run_feasibility_gpu.sh \
  --phase all --resume
```

The command above is documentation only and was not executed during epoch-1
validation preparation. The feasibility runner still requires integration of
this recovered adapter/config before scientific execution.
