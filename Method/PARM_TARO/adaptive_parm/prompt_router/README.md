# Prompt-level Adaptive PARM router

This phase predicts one canonical trust weight per `(prompt, alpha)`. It uses
only scalar prompt-end Base/PBLoRA features available before generation. The
PBLoRA checkpoint is loaded frozen and is never trained.

The historical teacher-forced cache is deliberately not reused for features:
concatenated prompt/response tokenization permits a boundary-crossing BPE token,
so its first stored continuation logit can depend on response text.

## CPU dry run

```bash
cd /home/jupyter-iec2024se10/Test-time-Alignment
/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10 \
  -m PARM_TARO.adaptive_parm.prompt_router.run_prompt_router \
  --phase all --device cuda --resume --dry-run
```

## Shared-GPU launch

```bash
cd /home/jupyter-iec2024se10/Test-time-Alignment
mkdir -p results/parm_taro/adaptive_parm/02_prompt_router

nohup /home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10 \
  PARM_TARO/adaptive_parm/scripts/wait_and_run_prompt_router.py \
  > results/parm_taro/adaptive_parm/02_prompt_router/watcher.log 2>&1 &

echo $! > results/parm_taro/adaptive_parm/02_prompt_router/watcher.pid
```

Rerunning the same command is resume-safe. Status:

```bash
cat results/parm_taro/adaptive_parm/02_prompt_router/watcher.pid
tail -f results/parm_taro/adaptive_parm/02_prompt_router/watcher.log
nvidia-smi --query-gpu=memory.free,memory.used,utilization.gpu --format=csv
```
