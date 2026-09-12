# Source reconstruction

Classification: **RECONSTRUCTED_FROM_REPO_EVIDENCE**. These files are compatible reconstructions, not claimed historical originals.

| File | Reconstructed interface | Primary retained evidence |
|---|---|---|
| `router_v2/cache/__init__.py` | Public cache API/re-exports | Imports in `router_v2`, `PARM_TARO`, and tests |
| `router_v2/cache/schema.py` | `CacheRecord`, `DerivedFeatures`, validation and deterministic dictionaries | `router_v2/tests/test_feature_cache.py`; cache manifests; File 03 |
| `router_v2/cache/features.py` | independent Top-K candidates and finite per-token derived features | Existing feature tests; SmartRouter inputs; File 03 equations |
| `router_v2/cache/io.py` | JSON/JSONL atomic writes, strict validation, no overwrite | Existing save/load tests and historical cache layout |
| `router_v2/cache/inventory.py` | deterministic source/schema inventory | `SourceInventoryTests`; retained inventory JSON |

All five source files carry the required four-line reconstruction notice. No caller was modified to weaken the expected interface. The cache contains current-position source logits, candidate IDs, gold target, and optional preference only. It rejects history/future-token fields and does not force the gold token into either Top-K candidate set.

Derived values implemented from retained equations/callers: base/guide entropy and margin, Jensen-Shannon divergence, Top-1 agreement, Top-K overlap, standard deviation/range, rank correlation, and normalized position. All are checked finite; JS is non-negative, overlap lies in `[0,1]`, and agreement is binary.

Validation results:

- New recovery contract suite: **11 passed**.
- Retained historical cache suite: **16 passed**.
- Vendored PARM source/equation/import spot checks: **3 passed**.
- A wider historical suite previously produced 232 passes plus artifact/hash-dependent failures caused by permanently missing checkpoints/runtime files; none was a cache contract failure.

`.gitignore` was narrowed so generated cache contents remain ignored while `router_v2/cache/*.py` can be tracked. Protected `PARM/`, `router/`, and `Method/RAD/` were not edited.

Runner phases `inventory`, `reconstruct_cache`, `verify_source`, and `import_test` are idempotent and state-driven. State updates use a temporary file, `fsync`, and atomic rename.
