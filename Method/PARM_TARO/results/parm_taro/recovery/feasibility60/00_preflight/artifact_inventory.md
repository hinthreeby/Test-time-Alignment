# Artifact inventory

| Artifact | Path | Status | Revision |
|---|---|---|---|
| base_model | `models/tulu-2-7b/config.json` | PASS | `3c6e328ae91fabdd0daf09de16887de9615c1f66` |
| base_weights | `models/tulu-2-7b/pytorch_model.bin.index.json` | PASS | `3c6e328ae91fabdd0daf09de16887de9615c1f66` |
| tokenizer | `models/tulu-2-7b/tokenizer.model` | PASS | `3c6e328ae91fabdd0daf09de16887de9615c1f66` |
| pblora | `results/parm_taro/checkpoints/parm_pku_pblora` | MISSING | `None` |
| taro_checkpoint | `results/parm_taro/training/taro/best.pt` | MISSING | `None` |
| v2_no_alpha | `results/parm_taro/training/v2_no_alpha/best.pt` | MISSING | `None` |
| v2_full_alpha | `results/parm_taro/training/v2_alpha_preference/final.pt` | MISSING | `None` |
| beaver_reward | `models/beaver-7b-v1.0-reward/model.safetensors.index.json` | PASS | `375cd6a9f0d7e339d2199b05ba129a4a8906596d` |
| beaver_cost | `models/beaver-7b-v1.0-cost/model.safetensors.index.json` | AMBIGUOUS | `None` |
| safe_rlhf | `models/safe-rlhf-source/safe_rlhf/__init__.py` | MISSING | `None` |
| dataset_manifest | `dataset/parm_taro/manifest.json` | PASS | `None` |
| validation | `dataset/parm_taro/validation.json` | PASS | `None` |
| protocol_lock | `results/parm_taro/evaluation/protocol/protocol_lock.json` | MOVED_BUT_RESOLVED | `None` |
| normalization_source | `results/parm_taro/evaluation/protocol/validation_calibration_raw.jsonl` | MOVED_BUT_RESOLVED | `None` |

## Full-server custom-artifact search

- PBLORA candidates: `[]`
- PARM-TARO router checkpoint candidates: `[]`

No model was loaded and no test split was read.
