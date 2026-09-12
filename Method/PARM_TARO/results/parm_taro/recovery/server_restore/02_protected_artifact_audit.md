# Protected artifact audit

Status: **FAIL** — none of the seven complete historical artifacts can be verified in this restored snapshot.

| Artifact | Path | Expected SHA-256 | Actual SHA-256 | Status |
|---|---|---|---|---|
| PARM original | `PARM` | `dcbaae05b0c3467e2d9ff3255ca6fc9f2296bc4e97f9a22b650e38f666bac23a` | `1264b399566c6126b86d98a6e468ec416a72e27469f92d7619f9934b85b2d42f` | **MISMATCH** |
| router V1 | `router` | `8595fb3e48c2b1771a07a6efba09c29560785a581d5490f582286c58fbf5ffd3` | `2575589fe7eebf021cd454b8c8c440b93ad54f2c852fd198b82be90e9bbe2f79` | **MISMATCH** |
| Method/RAD | `Method/RAD` | `e0c8907885b80cde9c12f0e1f6e3134b0781e811ee8edafb9f0d690988d36bbe` | `4dfccbd9afa8808ac3614ae31705616956811504112df7bc371dde5808177522` | **MISMATCH** |
| PBLoRA | `results/parm_taro/checkpoints/parm_pku_pblora` | `f26682e3cf51f8e09921873119fc371d329669cf9448d59ae0025fd40b23ffef` | `—` | **MISSING** |
| Stage 9 TARO | `results/parm_taro/training/taro/best.pt` | `32caa71a0e1462e5f31a6c587c735b19c22d109f16f729430cd63c7910731881` | `—` | **MISSING** |
| Stage 9 V2 full-alpha | `results/parm_taro/training/v2_alpha_preference/final.pt` | `56fbed68168537bd84066a5b86fbfefc8525cb5311c06d8668012e2cecca1e20` | `—` | **MISSING** |
| Stage 9 V2 no-alpha | `results/parm_taro/training/v2_no_alpha/best.pt` | `ea5cf0e7b33ccdef1f4882afba0679a09feaaf2b27c8a8bc12ad7c058cb0d7e3` | `—` | **MISSING** |

`PARM/`, `router/`, and `Method/RAD/` are Git-clean at the current commit. Their MISMATCH status means the archived hash covered a fuller old workspace (including now-missing ignored/runtime files); it does not prove the tracked files were edited. PBLoRA and all three Stage-9 weight files are genuinely missing.
