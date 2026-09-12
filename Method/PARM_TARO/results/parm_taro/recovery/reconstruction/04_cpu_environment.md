# CPU environment

- Host: GPU is externally confirmed **AVAILABLE** (`NVIDIA GeForce RTX 5090`, driver `580.105.08`, host-reported CUDA `13.0`).
- Agent: `AGENT_GPU_ACCESS=UNAVAILABLE_BY_DESIGN`; all commands set `CUDA_VISIBLE_DEVICES=""`. `torch.cuda.is_available() == False` is expected and is not an environment failure.
- Isolated interpreter: `/home/jupyter-iec2024se10/miniforge3/envs/tta/bin/python`, Python 3.10.21. No global install was performed.
- Core versions: torch 2.7.1+cu128 (CPU execution here), transformers 4.39.3, peft 0.10.0, accelerate 0.29.2, tokenizers 0.15.2, datasets 2.18.0, numpy 1.26.4, pandas 2.2.2, scipy 1.13.0, scikit-learn 1.4.2, safetensors 0.4.3.
- Repo-pinned additions needed by source preflight: pytest 8.3.5, loguru 0.7.1, trl 0.9.6, fuzzywuzzy 0.18.0, python-Levenshtein 0.21.1, google-api-python-client 2.98.0, python-dotenv 1.0.0. The Google/protobuf stack was aligned to `PARM/requirements.txt`, including protobuf 5.29.4.
- `python -m pip check`: **No broken requirements found**.

This is a CPU source/preflight environment only. It does not certify RTX 5090 kernel compatibility, 4-bit bitsandbytes execution, or 7B model load; those are user-side GPU gates after public artifacts are restored.
