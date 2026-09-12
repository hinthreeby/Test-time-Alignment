# Public artifact discovery

Discovery was based on retained configs, logs, frozen audits, source paths, historical file hashes, then public repository metadata. No approximate substitute was selected.

| Artifact | Exact ID | Revision | Public | Evidence | Target | Status |
|---|---|---|---|---|---|---|
| Tulu-2-7B and tokenizer | `allenai/tulu-2-7b` | `3c6e328ae91fabdd0daf09de16887de9615c1f66` | yes | Both weight-shard hashes, index hash, and tokenizer hash match retained audit | `models/tulu-2-7b/` | INCOMPLETE |
| Beaver reward evaluator | `PKU-Alignment/beaver-7b-v1.0-reward` | `375cd6a9f0d7e339d2199b05ba129a4a8906596d` | yes | Remote metadata reconstructs retained 14-file tree SHA-256 `3410256f542b2e06225d2652102d5dff1ce356a4cd092249643ab03a999a47b3` | `models/beaver-7b-v1.0-reward/` | INCOMPLETE |
| Beaver cost evaluator | `PKU-Alignment/beaver-7b-v1.0-cost` | `c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28` | yes | Remote metadata reconstructs retained 14-file tree SHA-256 `418b941682ced967d71194c059beb967ebd225c2b6713a713ccdcea1fa6d8454` | `models/beaver-7b-v1.0-cost/` | INCOMPLETE |
| Safe-RLHF source | `https://github.com/PKU-Alignment/safe-rlhf.git` | **UNKNOWN** | yes | Repo identity is explicit, but retained artifact had no commit; its archived hash included non-reproducible `.git` state | `models/safe-rlhf-source/` | UNKNOWN_REVISION |

No artifact was downloaded. The filesystem has about **9.3 GiB free**, while the guarded downloader requires 55 GiB for three 7B snapshots and temporary files. It exits before download and never removes partial data. Safe-RLHF also remains disabled until its historical commit is established.

Resume-safe tooling:

- `scripts/download_public_artifacts.sh` pins all established revisions, verifies before skipping, and is safe to rerun.
- `scripts/check_downloads.py` checks required config/tokenizer/index/shards, retained hashes, total size, and `.tmp`/`.lock`/`.incomplete` files.

Public source pages used for identity confirmation: Hugging Face Tulu-2-7B, Hugging Face Beaver reward/cost repositories, and the official PKU-Alignment Safe-RLHF repository. The exact revision/hash claims above are preserved in `state/public_artifacts.json` and the checker.
