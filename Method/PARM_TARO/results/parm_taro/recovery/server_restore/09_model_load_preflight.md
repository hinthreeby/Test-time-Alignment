# Real model-load preflight

Status: **NOT RUN**. The exact Tulu-2-7B files, tokenizer, PBLoRA, Stage-9 checkpoints, required internal `router_v2.cache` source, and usable CUDA device are unavailable. Import success alone was not accepted as model readiness. No substitute model and no bulk generation were used.

Expected Stage-10 logic remains documented as one 4-bit shared backbone, base pass with adapters disabled, and guide pass with PBLoRA enabled; it cannot be validated dynamically in the present state.
