# Source inventory

Project root: `/home/jupyter-iec2024se10/Test-time-Alignment`

Git branch/commit: `main` / `d2912fa11170a272973130696f9bbeef3e6c8af3`

The protected tracked trees have no Git modifications. Recovery files and Files 12-13 are untracked; no checkout/reset/rebase was performed.

| Component | Expected path | Exists? | Size | Git tracked? | Modified? | Required? | Action |
|---|---|---:|---:|---:|---|---:|---|
| PARM baseline | `PARM` | yes | 1.4 MiB | yes | `clean` | yes | Preserve; tracked source clean |
| PARM-TARO | `PARM_TARO` | yes | 989.0 KiB | yes | `?? PARM_TARO/recovery/` | yes | Recovery code present |
| Router V1 | `router` | yes | 5.1 MiB | yes | `clean` | yes | Preserve; tracked source clean |
| Router V2 | `router_v2` | yes | 757.5 KiB | yes | `clean` | yes | Incomplete without cache package |
| Router V2 cache package | `router_v2/cache` | no | 0.0 B | no | `clean` | yes | Restore exact source from backup |
| RAD baseline | `Method/RAD` | yes | 117.9 KiB | yes | `clean` | yes | Preserve; tracked source clean |
| Dataset | `dataset` | yes | 58.2 MiB | yes | `clean` | yes | Processed and source PKU data present |
| PARM-TARO results | `results/parm_taro` | yes | 23.1 MiB | yes | `?? results/parm_taro/recovery/` | yes | Historical reports present |
| Recovery code | `PARM_TARO/recovery` | yes | 66.0 KiB | no | `?? PARM_TARO/recovery/` | yes | Present; untracked recovery-only code |
| Recovery evidence | `results/parm_taro/recovery` | yes | 69.5 KiB | no | `?? results/parm_taro/recovery/` | yes | Present; historical files preserved |
| Stage 9 training records | `results/parm_taro/training` | yes | 189.0 KiB | yes | `clean` | yes | JSON records present; weights absent |
| Stage 10 evaluation | `results/parm_taro/evaluation` | yes | 22.8 MiB | yes | `clean` | yes | Historical outputs present |
| Checkpoint root | `results/parm_taro/checkpoints` | no | 0.0 B | no | `clean` | yes | Restore from backup |

## Git status

```text
?? PARM_TARO/recovery/
?? document/12_ROUTER_LOW_LAMBDA_DIAGNOSIS_AND_FIX.md
?? document/13_STAGED_REEVALUATION_200_500_1500.md
?? results/parm_taro/recovery/
```

## Remotes

```text
origin	https://github.com/hinthreeby/Test-time-Alignment.git (fetch)
origin	https://github.com/hinthreeby/Test-time-Alignment.git (push)
```
