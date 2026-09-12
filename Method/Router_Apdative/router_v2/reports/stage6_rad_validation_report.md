# Stage 6 RAD V2 Validation Implementation Report

Generated: 2026-08-16

## Status

**HOST FINALIZATION REQUIRED - all 7,200 full test records exist and pass the
corrected length audit, but `stage6_report.json` must be regenerated before
Stage 6 is marked PASS.**

All three Stage 5 checkpoints currently report `PASS`:

| Stage | Validation NLL | Frozen audit |
|---|---:|---|
| TARO | 3.2901245693 | PASS |
| V2 state | 3.2902039169 | PASS |
| V2 history | 3.2900684848 | PASS |

## Full-Evaluation Prompt ID Blocker Fix

The full test selection contains 600 rows: 200 each from negative, neutral, and
positive, beginning at per-class offset 50. The source reuses
`md5_hash=733a365ae53cd4f4fcd11514b9b9f8d8` for two different rows:

~~~text
neutral source index 163: Want to dedicate an entire weekend...
positive source index 223: This is what makes Cybrary different and allows us...
~~~

There are no duplicate selected hashes within an individual class. The collision
appeared only after concatenating classes because the previous V2 loader used the
bare source hash as `prompt_id`. The 2/2 smoke selection never reached either
colliding row.

V2 now constructs deterministic IDs as:

~~~text
rad:{split}:{class}:{zero_based_source_index}
~~~

It retains the original V1 hash as `source_prompt_id`. No prompt is removed or
reordered. V1 import matches read-only rows by source ID, class, and exact prompt
text, then normalizes them to the same collision-free V2 ID. A real full import
found all 2,400 requested V1 rows (600 prompts x 4 methods), with 2,400 unique
`(prompt_id, method, seed)` keys. The reused source hash maps to two distinct
canonical IDs as intended.

## Generation-Length Blocker Fix

The completed full run contains 100 realized generations shorter than 32 tokens:

~~~text
v1_base:        14
v1_fixed:       41
v1_heuristic:   23
v1_router:      22
all V2 methods:  0
~~~

Every short row ends in the GPT-2 EOS token `50256`. V1 decoding legitimately
stops on EOS, while its immutable report builder incorrectly hard-coded
`eos_generated=false`. The adapter now infers EOS termination from the final
selected token without changing the V1 source. GPT-2 Large and GPT-2 Medium use
byte-identical tokenizer assets, so tokenizer mismatch is not involved.

`max_new_tokens` is a generation budget, not a minimum realized length. The
correct gate requires a budget of at least 32, exact agreement between the
recorded length and selected token IDs, positive lengths no greater than the
budget, and EOS termination for every short record. The existing 7,200 records
pass all of these checks and can be reused unchanged. Full statistics and the
exact 100 affected IDs are in
`router_v2/reports/stage6_generation_length_diagnostic.md`.

The existing V1 32-token report, V1 checkpoint, V1 source files, benchmark, and
independent sentiment classifier were inspected read-only. Their recorded hashes
match current files, the V1 report contains zero generation failures, and its
checkpoint loads and completes a deterministic CPU forward through current V1
code.

## Evaluation Protocol

New results use only:

~~~text
results/router_v2/rad/smoke/
results/router_v2/rad/full/
~~~

Validation selects:

~~~text
best fixed lambda from [0, 0.25, 0.5, 0.75, 1]
best heuristic from entropy-gap and JS heuristics
mean TARO lambda across validation
mean V2-history lambda across validation
~~~

Selection uses sentiment success on negative validation prompts, then target
probability and PPL as deterministic tie breakers. Same-average controls use the
validation mean and are fixed before held-out test generation.

Held-out methods are:

~~~text
V1 Base / Fixed / Heuristic / learned router (read-only imported rows)
V2 Base
V2 selected Fixed
V2 selected Heuristic
TARO
V2 state
V2 history (primary Smart V2)
TARO same-average fixed lambda
V2-history same-average fixed lambda
~~~

Every V2 guided step computes independent base/guide Top-K features and applies
the router's scalar lambda to the complete vocabulary:

~~~text
guided_logits = base_logits + lambda_t * (guide_logits - base_logits)
~~~

Base generation skips guide and router forwards so its latency is not inflated.
History selected score at step `t` is stored only after token selection and can
affect lambda at `t+1` or later. Full logits are transient and never written.

## Metrics

The evaluator records:

~~~text
independent sentiment success and target probability
PPL
PPL degradation relative to same-family base
coherence proxy
unigram/bigram/trigram repetition
distinct-1/2/3
generation length
latency per token and total latency
lambda distribution and lambda by position
alignment-vs-PPL-degradation Pareto frontier
~~~

V1 PPL is the existing GPT-2 Large prompt-plus-generation score. V2 PPL is
continuation-conditional under the frozen GPT-2 Medium base. Raw values are
reported with explicit `ppl_model_family` and `ppl_protocol`, but are not treated
as cross-family comparable. Primary Pareto analysis uses family-relative PPL
degradation.

## Integrity And Resume

Before and after evaluation, checksums cover:

~~~text
router/
PARM/
Method/RAD/
dataset/rad_benchmark/
router/evaluation/report_run_32tokens/
~~~

JSONL generation is fsynced per completed job. Resume keys are
`(prompt_id, method, seed)` and duplicate/unplanned records fail closed. Final
status requires every method, zero failures, a configured budget of at least 32,
valid EOS termination for any shorter realized output, finite primary metrics,
nonconstant V1/TARO/state/history routing, both same-average controls, passing
provenance, and unchanged protected hashes. It never requires padding after EOS.

## Files Added Or Changed

~~~text
router_v2/evaluation/__init__.py
router_v2/evaluation/rad/__init__.py
router_v2/evaluation/rad/config.py
router_v2/evaluation/rad/data.py
router_v2/evaluation/rad/decoding.py
router_v2/evaluation/rad/scoring.py
router_v2/evaluation/rad/metrics.py
router_v2/evaluation/rad/legacy.py
router_v2/evaluation/rad/engine.py
router_v2/scripts/evaluate_rad_v2.py
router_v2/scripts/diagnose_stage6_lengths.py
router_v2/configs/evaluate_stage6_rad_smoke.json
router_v2/configs/evaluate_stage6_rad_full.json
router_v2/schemas/rad_evaluation_config.schema.json
router_v2/tests/test_rad_v2_evaluation.py
router_v2/README.md
router_v2/reports/stage6_rad_validation_report.md
router_v2/reports/stage6_generation_length_diagnostic.md
~~~

## CPU Verification

~~~text
Ran 120 tests in 4.155s
OK
~~~

The 20 Stage 6 tests additionally cover full/validation ID uniqueness, exact row
retention, deterministic ID reproduction, full resume-key uniqueness, preserved
V1 offsets/source IDs, disambiguation of reused V1 hashes, V1 EOS metadata
repair, valid short-EOS acceptance, and rejection of short non-EOS or mismatched
length records. Existing Stage 2-5 and protected-tree tests also pass.

The protected snapshot recorded immediately before the failed full preflight was
recomputed after the fix. `router/`, `PARM/`, `Method/RAD/`,
`dataset/rad_benchmark/`, and the old V1 result directory all match exactly.

Real artifact audit:

~~~text
V1 provenance: PASS
V1 checkpoint load/forward: PASS
V1 smoke rows found: 24/24
TARO checkpoint: PASS, 0 trainable inference parameters
state checkpoint: PASS, 0 trainable inference parameters
history checkpoint: PASS, 0 trainable inference parameters
~~~

## Host Setup

~~~bash
cd "/home/jupyter-iec2024se10/Reward Decoding"
set -o pipefail

export PY=/home/jupyter-iec2024se10/miniconda3/envs/cd/bin/python
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p results/router_v2/logs
nvidia-smi
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'
~~~

## CPU Tests

~~~bash
CUDA_VISIBLE_DEVICES="" "$PY" -m unittest discover -s router_v2/tests -v \
  2>&1 | tee results/router_v2/logs/stage6_cpu_tests.log
~~~

## GPU Smoke

The smoke uses 2 validation and 2 held-out prompts per class, one seed, and 32
generated tokens. It still imports matching V1 prompts from offset 50.

~~~bash
"$PY" -m router_v2.scripts.evaluate_rad_v2 \
  --config router_v2/configs/evaluate_stage6_rad_smoke.json \
  --device cuda --no-cpu-fallback \
  2>&1 | tee results/router_v2/logs/stage6_rad_smoke.log

"$PY" -m json.tool results/router_v2/rad/smoke/stage6_report.json
~~~

Resume smoke after interruption:

~~~bash
"$PY" -m router_v2.scripts.evaluate_rad_v2 \
  --config router_v2/configs/evaluate_stage6_rad_smoke.json \
  --device cuda --no-cpu-fallback --resume \
  2>&1 | tee -a results/router_v2/logs/stage6_rad_smoke.log
~~~

## Full Finalization

All 4,800 V2 and 2,400 imported V1 test records already exist. Resume reuses
them, performs zero generation jobs, and regenerates scoring/report artifacts
with the corrected gate:

~~~bash
"$PY" -m router_v2.scripts.evaluate_rad_v2 \
  --config router_v2/configs/evaluate_stage6_rad_full.json \
  --device cuda --no-cpu-fallback --resume \
  2>&1 | tee -a results/router_v2/logs/stage6_rad_full.log

"$PY" -m json.tool results/router_v2/rad/full/stage6_report.json
~~~

Resume full evaluation:

~~~bash
"$PY" -m router_v2.scripts.evaluate_rad_v2 \
  --config router_v2/configs/evaluate_stage6_rad_full.json \
  --device cuda --no-cpu-fallback --resume \
  2>&1 | tee -a results/router_v2/logs/stage6_rad_full.log
~~~

## Monitoring And Inspection

~~~bash
watch -n 1 nvidia-smi
~~~

~~~bash
tail -F results/router_v2/logs/stage6_rad_smoke.log \
  results/router_v2/logs/stage6_rad_full.log
~~~

~~~bash
"$PY" -m json.tool results/router_v2/rad/full/validation_selection.json
sed -n '1,40p' results/router_v2/rad/full/pareto_points.csv
sed -n '1,40p' results/router_v2/rad/full/lambda_diagnostics.csv
~~~

After smoke, return these files before starting the full run if any check fails:

~~~text
results/router_v2/logs/stage6_rad_smoke.log
results/router_v2/rad/smoke/stage6_report.json
results/router_v2/rad/smoke/validation_selection.json
results/router_v2/rad/smoke/protected_audit_after.json
results/router_v2/rad/smoke/v1_baseline_audit.json
results/router_v2/rad/smoke/pareto_points.csv
~~~

Do not begin PARM-TARO until the real Stage 6 report is complete and stable.
