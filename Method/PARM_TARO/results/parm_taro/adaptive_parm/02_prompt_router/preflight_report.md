# Prompt-router preflight

The implementation and CPU tests pass, but the scientific experiment has not
yet run because Codex cannot access the host GPU. No PASS/WEAK/FAIL method
verdict is assigned by this preflight.

The Phase 03 teacher-forced cache was rejected as an inference-feature source.
Its concatenated prompt/response tokenization can contain a boundary-crossing
BPE token, making the first cached distribution response-dependent. The maximum
response-branch differences were `0.0386744` for Base and `0.0633774` for
PBLoRA.

The host job instead performs fresh raw-prompt-only forwards with one physical
4-bit backbone, a frozen PBLoRA adapter, no generation, and no scorer. It then
runs CPU leave-one-prompt-out training for the logistic action classifier and
Ridge utility predictor.
