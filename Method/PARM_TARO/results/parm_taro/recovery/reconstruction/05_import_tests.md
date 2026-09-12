# CPU import and static tests

Status: **PASS** for the scoped source/import/schema/unit preflight.

CPU imports passed for torch, transformers, peft, accelerate, `router_v2`, `PARM_TARO`, and all reconstructed cache modules. Vendored runtime resolution also passed and points to protected sources under `PARM/peft/src/peft` and `PARM/language-model-arithmetic/src/model_arithmetic`.

Synthetic checks passed without a real 7B model:

- SmartRouter V2 builds and forwards finite CPU tensors.
- Old sigmoid gate remains within configured epsilon bounds.
- Residual equation `lambda0 * (1 + rho * tanh(raw))` is finite and equals `lambda0=1` at raw zero.
- Diagnostic static endpoints select the expected base/static distributions on synthetic logits.
- Vendored static PARM raw-sum equation matches the current adapter equation.

Test totals used for the gate:

- reconstruction tests: 11/11 passed;
- retained cache tests: 16/16 passed;
- vendored PARM import/equation spot checks: 3/3 passed.

Logs: `logs/cache_tests.log`, `logs/source_verification.log`, and `logs/import_tests.log`. CUDA availability is false by design. A real 7B/PBLoRA load was not attempted because artifacts are absent and CPU inference would not be practical.
