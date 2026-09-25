# Token-level Adaptive-PARM headroom diagnostic

This is a diagnostic experiment, not a trained router. It uses the frozen
PBLoRA checkpoint and canonical fusion

`z = (1-w) * log(p_base) + w * log(p_PARM)`.

The frozen validation manifest contains 10 unique prompts and all five alpha
values (50 cases). A guide-only (`w_ref=1`) trajectory supplies at most five
causal states per case. At each state, every `w` in
`[0, .25, .5, .75, 1]` controls one next-token action; the remaining 31 tokens
use `w_ref=1`.

The entry point launches four isolated processes in order:

1. Tulu/PBLoRA generation and leakage audit;
2. Beaver reward scoring;
3. Beaver cost scoring;
4. CPU analysis and grouped leave-one-prompt-out structure probe.

This process isolation prevents Tulu, reward, and cost models from sharing GPU
memory. Every case and score is checkpointed atomically and `--resume` skips
completed work.

The launcher also bootstraps the relocated source trees in this order:
`Method/Router_Apdative`, `Method/PARM_TARO`, then the project root. The same
ordered paths are propagated through `PYTHONPATH` to every isolated child and
the shared-GPU watcher; no manual `PYTHONPATH` export is required.

## Direct foreground run

```bash
cd /home/jupyter-iec2024se10/Test-time-Alignment
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GPU_PYTHON=/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10
$GPU_PYTHON PARM_TARO/adaptive_parm/token_headroom/run_token_headroom.py \
  --num-prompts 10 --resume
```

## Shared-GPU watcher

```bash
cd /home/jupyter-iec2024se10/Test-time-Alignment
mkdir -p results/parm_taro/adaptive_parm/04_token_headroom
GPU_PYTHON=/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10 \
MIN_FREE_MIB=18000 CHECK_INTERVAL=5 STABLE_CHECKS=2 \
nohup /home/jupyter-iec2024se10/.conda/envs/genarm/bin/python3.10 \
  PARM_TARO/adaptive_parm/scripts/wait_and_run_04_token_headroom.py \
  --num-prompts 10 --resume \
  > results/parm_taro/adaptive_parm/04_token_headroom/watcher.log 2>&1 &
echo $! > results/parm_taro/adaptive_parm/04_token_headroom/watcher.pid
```

Inspect with
`ps -fp "$(cat results/parm_taro/adaptive_parm/04_token_headroom/watcher.pid)"`,
`tail -f results/parm_taro/adaptive_parm/04_token_headroom/watcher.log`, and
`nvidia-smi`. Stop safely with
`kill -SIGTERM "$(cat results/parm_taro/adaptive_parm/04_token_headroom/watcher.pid)"`.

The runtime is workload-dependent. A reasonable planning range is roughly
4–10 hours on an otherwise free RTX 5090; resume state prevents completed
cases from being repeated after interruption.
