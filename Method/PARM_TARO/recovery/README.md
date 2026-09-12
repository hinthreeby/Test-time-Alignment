# PARM-TARO recovery boundary

This package is the only code location for the post-Stage-10 recovery work.
It does not modify `PARM/`, `router/`, PBLoRA, or old Stage-9/10 artifacts.

`diagnostics.py` contains the exact static-equivalence and gold-token gradient
calculations required by File 12. `objectives.py` contains the *candidate* single
fix family: residual lambda around static PARM, plus an optional PARM-preservation
KL term. These utilities do not authorize training by themselves. Training must
remain blocked until D0-D7 have complete measured artifacts and the validation
protocol is available.

The checked-in workspace snapshot omits the ignored model/checkpoint assets and
has no usable PyTorch runtime/GPU. See
`results/parm_taro/recovery/diagnosis/root_cause_report.md` for the exact blocked
preconditions and the archived evidence that can still be audited.
