# PARM-TARO feasibility-60 summary

## 1. Is PBLoRA actually available and loadable?

No. Exact PBLORA checkpoint is unavailable; load was not attempted.

## 2. Does PBLoRA respond to alpha?

UNTESTABLE until PBLORA exists.

## 3. Is the guide useful?

UNTESTABLE; guide utility was not inferred from historical metrics.

## 4. Are base/PBLoRA/model paths correct?

Tulu base/tokenizer are present at the pinned public revision; PBLORA provenance is absent.

## 5. Is PARM-TARO fusion mismatched with author PARM?

Yes. Author F2 divides by 2, current F3 does not.

## 6. Is there a temperature/scale confound?

Yes, algebraically and in the synthetic entropy matrix.

## 7. Is V2 checkpoint actually loaded?

No. Expected V2/TARO weight files are absent.

## 8. Does router respond to alpha?

UNTESTABLE without a valid router checkpoint.

## 9. Does NLL push lambda toward zero?

LIKELY historically, but fresh validation gradient evidence is blocked.

## 10. Is sigmoid/parameterization saturated?

UNTESTABLE for trained gates; source mapping alone is insufficient.

## 11. Is there teacher-forcing exposure shift?

UNTESTABLE.

## 12. What is best global fixed lambda?

UNTESTABLE.

## 13. What is best lambda per alpha?

UNTESTABLE.

## 14. What is oracle per-prompt performance?

UNTESTABLE.

## 15. How large is adaptive headroom?

UNTESTABLE.

## 16. Does dynamic beat same-average fixed?

UNTESTABLE.

## 17. Main root cause ranked #1/#2/#3.

#1 infrastructure/provenance (missing PBLORA); #2 confirmed fusion-scale confound; #3 likely NLL objective mismatch, pending fresh validation proof.

## 18. Is this research direction worth continuing?

Cannot decide GO/NO-GO scientifically until PBLORA, scorer runtime, and validation oracle-headroom sweep pass.

No retraining, validation generation, test tuning, or protected-source modification occurred.

BLOCKED — PBLORA / MODEL / EVALUATOR NOT READY
