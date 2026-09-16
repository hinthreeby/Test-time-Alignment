# Genarm GPU environment and wrapper wiring

## Environment resolution

- The non-login agent shell has no active Conda command and no `python` on
  `PATH`, so no active environment was inferred.
- Located genarm executable:
  `/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python`.
- Resolved real path:
  `/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10`.
- Python: 3.10.21.
- CPU reconstruction remains allowed to use
  `/home/jupyter-iec2024se10/miniforge3/envs/tta/bin/python`.
- GPU training/inference wrappers reject an interpreter whose resolved path is
  under `/envs/tta/`.

## Genarm package audit

| Package | Version/status |
|---|---|
| torch | `2.7.1+cu128` |
| CUDA runtime compiled into torch | `12.8` |
| transformers | `4.39.3` |
| peft | `0.10.0` |
| trl | `0.9.6` |
| accelerate | `0.29.2` |
| bitsandbytes | `0.50.2` (installed into genarm during this task) |
| datasets | `2.18.0` |
| rich | import PASS |
| `trl.DPOTrainer`, `trl.DPOConfig` | import PASS |
| `pip check` | PASS |

`bitsandbytes==0.50.2` was selected as a compatibility exception rather than
the historical 0.43.1 pin: the installed PyTorch is CUDA 12.8 and the target
RTX 5090 is Blackwell (`sm_120`), which current official wheels support. No
existing package was upgraded.

## CUDA visibility caveat

The agent process reported `torch.cuda.is_available() == False`, GPU count 0,
and could not initialize NVML. This is the expected agent/sandbox visibility
restriction and is not classified as a host GPU failure. The user has confirmed
that the host RTX 5090 works.

Real wrapper execution is nevertheless fail-closed: before any model load it
imports the required stack, prints all requested versions and device fields,
requires CUDA availability, and requires the visible device name to contain
`5090`. PBLoRA additionally requires the TRL DPO imports. `--dry-run` prints the
same audit but tolerates hidden CUDA solely so Codex can validate command wiring
without starting training.

## Wrapper changes

- `gpu_env.sh`: centralized configurable interpreter selection and fail-closed
  CUDA/package preflight.
- `train_pblora_repro.sh`: prioritizes `PBLORA_PYTHON`; no tta fallback.
- TARO wrapper: prioritizes `TARO_PYTHON`, then `TTA_GPU_PYTHON`.
- All V2 wrappers: prioritize `V2_PYTHON`, then `TTA_GPU_PYTHON`.
- Staged evaluation wrapper: prioritizes `EVALUATION_PYTHON`, then
  `TTA_GPU_PYTHON`.
- `run_feasibility_gpu.sh`: new genarm-only GPU entrypoint using
  `FEASIBILITY_PYTHON`, then `TTA_GPU_PYTHON`.

Hard-coded tta references that remain are CPU reconstruction reports, logs,
state snapshots, and CPU audit utilities. No GPU wrapper contains a tta Python
hard-code.

## Verification

- Shell syntax: PASS.
- Python entrypoint syntax: PASS.
- Genarm imports and TRL DPO imports: PASS.
- `pip check`: PASS.
- PBLoRA smoke `--dry-run`: PASS; rendered command uses
  `/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10` and does not
  start training.
- Host-side CUDA preflight: pending user execution because CUDA is intentionally
  hidden from the agent.
