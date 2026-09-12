# Dependency audit

Selected historical evidence: `Method/GenARM/requirements.txt` plus Stage-9 provenance (Python 3.10.20 and torch 2.2.2+cu121). `PARM/requirements.txt` contains later/conflicting pins, so packages were not changed automatically.

Current candidate environment: `/home/jupyter-iec2024se10/miniforge3/envs/tta` (Python 3.10.21).

| Package | Required version | Installed version | Status | Evidence/source |
|---|---|---|---|---|
| torch | `2.2.2+cu121` | `2.7.1+cu128` | FAIL | Method/GenARM lock / Stage-9 provenance |
| transformers | `4.39.3` | `4.39.3` | PASS | Method/GenARM lock / Stage-9 provenance |
| peft | `0.10.0` | `0.10.0` | PASS | Method/GenARM lock / Stage-9 provenance |
| accelerate | `0.29.2` | `0.29.2` | PASS | Method/GenARM lock / Stage-9 provenance |
| bitsandbytes | `0.43.1` | `MISSING` | FAIL | Method/GenARM lock / Stage-9 provenance |
| datasets | `2.18.0` | `2.18.0` | PASS | Method/GenARM lock / Stage-9 provenance |
| tokenizers | `0.15.2` | `0.15.2` | PASS | Method/GenARM lock / Stage-9 provenance |
| numpy | `1.26.4` | `1.26.4` | PASS | Method/GenARM lock / Stage-9 provenance |
| pandas | `2.2.2` | `2.2.2` | PASS | Method/GenARM lock / Stage-9 provenance |
| scipy | `1.13.0` | `1.13.0` | PASS | Method/GenARM lock / Stage-9 provenance |
| scikit-learn | `1.4.2` | `1.4.2` | PASS | Method/GenARM lock / Stage-9 provenance |
| tqdm | `4.66.2` | `4.66.2` | PASS | Method/GenARM lock / Stage-9 provenance |
| safetensors | `0.4.3` | `0.4.3` | PASS | Method/GenARM lock / Stage-9 provenance |
| huggingface-hub | `0.22.2` | `0.36.2` | FAIL | Method/GenARM lock / Stage-9 provenance |

Import scanning also reveals an internal source dependency, `router_v2.cache`, which is missing and cannot be fixed by pip.

The historical torch 2.2.2+cu121 lock may not support the newer GB202 GPU. The existing torch 2.7.1+cu128 appears intentional for this hardware, but it cannot be accepted until CUDA device access is repaired and a real model smoke test passes. No package was changed during this audit.
