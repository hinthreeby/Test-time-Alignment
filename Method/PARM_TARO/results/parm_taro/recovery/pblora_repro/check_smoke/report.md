# PBLoRA smoke checkpoint report

1. **Smoke checkpoint valid?** PASS.

2. **Reload deterministic?** PASS (max diff 0.000e+00).

3. **Trainable parameter count correct?** PASS: 6,296,064.

4. **Base frozen?** PASS.

5. **Alpha tensor reaches PBLoRA?** True.

6. **Guide distribution changes with alpha?** max JS=3.359e-02; top-1 change=0.080.

7. **Alpha effect strong / weak / broken?** PBLORA_ALPHA_CONDITIONING_PASS.

8. **Preference signal exists?** overall=0.3; helpfulness=0.2; harmlessness=0.4.

9. **Actual seconds/step?** 6.4222 s.

10. **ETA 1 epoch?** 1605.6 s (26.8 min), excluding eval/save.

11. **ETA 2 epochs?** 3211.1 s (53.5 min), excluding eval/save.

12. **Safe to start one-epoch reproduction?** NO; complete/investigate alpha gate first.

Runtime architecture: total parameters=3,506,709,184; PBLoRA trainable parameters=6,296,064 (0.1795%); inference `requires_grad` count=0. All trainable paths were PBLoRA Q/K/V.

Diagnostic peak VRAM: allocated=7891.7 MiB; reserved=8032.0 MiB.

The audit is read-only, uses validation only, and performs no generation or training.

PBLORA_SMOKE_PASS — ALPHA EFFECT WEAK, INVESTIGATE FIRST
