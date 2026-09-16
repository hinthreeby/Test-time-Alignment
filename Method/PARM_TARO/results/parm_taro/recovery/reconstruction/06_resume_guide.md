# Resume guide

## Reconstruction/public restore

Resume all unfinished phases:

```bash
cd /home/jupyter-iec2024se10/Test-time-Alignment
CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false \
  /home/jupyter-iec2024se10/miniforge3/envs/tta/bin/python \
  PARM_TARO/recovery/reconstruct_and_restore.py --phase all --device cpu --resume
```

After mounting/clearing at least 55 GiB, rerun the public downloader with the same command or directly:

```bash
bash results/parm_taro/recovery/reconstruction/scripts/download_public_artifacts.sh
```

It will resume/skip verified Hub files. Safe-RLHF still requires its exact historical commit.

## Common safe job control

For each command below use: START `nohup ... > LOG 2>&1 & echo $! > PID`; STATUS `ps -fp "$(cat PID)"`; LOG `tail -f LOG`; STOP-SAFELY `kill -SIGTERM "$(cat PID)"`. Do not use SIGKILL except an emergency. Recovery output directories are separate from Stage 9/10.

## PBLoRA reproduction

GPU jobs must use the `genarm` interpreter. The historical base is resolved as
Tulu-2-7B; seed 42 is an explicitly labeled fallback because the historical
launch seed was not retained:

```bash
PBLORA_PYTHON=/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10 PBLORA_SEED=42 CUDA_VISIBLE_DEVICES=0 nohup bash PARM_TARO/recovery/scripts/train_pblora_repro.sh --mode smoke > results/parm_taro/recovery/pblora_repro/smoke.log 2>&1 & echo $! > results/parm_taro/recovery/pblora_repro/smoke.pid
PBLORA_PYTHON=/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10 PBLORA_SEED=42 CUDA_VISIBLE_DEVICES=0 nohup bash PARM_TARO/recovery/scripts/train_pblora_repro.sh --mode smoke --resume > results/parm_taro/recovery/pblora_repro/smoke_resume.log 2>&1 & echo $! > results/parm_taro/recovery/pblora_repro/smoke.pid
```

HF Trainer periodic checkpoints contain model, optimizer, scheduler, trainer/global-step, and RNG state; `--resume` selects the latest numeric checkpoint.

## TARO and V2 reproduction

Start/resume by replacing `<wrapper>` and `<job>` with one of `train_taro_repro.sh`/`taro`, `train_v2_no_alpha_repro.sh`/`v2_no_alpha`, or—after its provenance gate is resolved—`train_v2_full_alpha_old_repro.sh`/`v2_alpha_preference`:

```bash
TTA_GPU_PYTHON=/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10 CUDA_VISIBLE_DEVICES=0 nohup bash PARM_TARO/recovery/scripts/<wrapper> > results/parm_taro/recovery/reproduced/<job>.log 2>&1 & echo $! > results/parm_taro/recovery/reproduced/<job>.pid
TTA_GPU_PYTHON=/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10 CUDA_VISIBLE_DEVICES=0 nohup bash PARM_TARO/recovery/scripts/<wrapper> --resume > results/parm_taro/recovery/reproduced/<job>_resume.log 2>&1 & echo $! > results/parm_taro/recovery/reproduced/<job>.pid
```

The retained trainers atomically write periodic/latest checkpoint payloads including model, optimizer, scheduler, epoch, global step, best metric, and RNG state. The recovered-router wrapper is present but intentionally exits until fresh D1-D7 authorizes its entry/config.

## Future val200 / val500 / test1500

Do not run these now. After checkpoint/config/protocol authorization, the staged wrapper requires both `--resume` and `--skip-existing`; its append-only engine deduplicates `(phase, prompt_id, alpha_index, method, seed)`:

```bash
EVALUATION_PYTHON=/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10 CUDA_VISIBLE_DEVICES=0 nohup bash PARM_TARO/recovery/scripts/evaluate_recovery_stage.sh --stage val200 --resume --skip-existing > results/parm_taro/recovery/val200/run.log 2>&1 & echo $! > results/parm_taro/recovery/val200/run.pid
EVALUATION_PYTHON=/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10 CUDA_VISIBLE_DEVICES=0 nohup bash PARM_TARO/recovery/scripts/evaluate_recovery_stage.sh --stage val500 --resume --skip-existing > results/parm_taro/recovery/val500/run.log 2>&1 & echo $! > results/parm_taro/recovery/val500/run.pid
EVALUATION_PYTHON=/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10 CUDA_VISIBLE_DEVICES=0 nohup bash PARM_TARO/recovery/scripts/evaluate_recovery_stage.sh --stage test1500 --resume --skip-existing > results/parm_taro/recovery/test1500/run.log 2>&1 & echo $! > results/parm_taro/recovery/test1500/run.pid
```

The wrapper refuses to start unless the stage-specific frozen config exists. Test1500 must remain locked until val200/val500 pass and checkpoint, config, and protocol hashes are frozen.
