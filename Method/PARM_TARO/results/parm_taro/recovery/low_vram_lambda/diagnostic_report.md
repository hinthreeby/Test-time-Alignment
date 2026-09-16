# Low-VRAM lambda diagnostic

## Facts

- Validation cases: 20 (first four existing feasibility60 cases per alpha; no Stage-10 test samples).
- Proxy objective: alpha-weighted mean continuation NLL over both labelled responses, matching Stage-9 source semantics.
- Base quantity `a`: normalized base log-probability; guide quantity `b`: normalized PBLoRA log-probability.
- Current best global lambda: 0.0 (NLL 1.730839).
- Normalized best global lambda: 0.0 (NLL 1.730839).
- Current oracle: NLL 1.730654; absolute gain 0.000185; relative gain 0.011%.
- Normalized oracle: NLL 1.706372; absolute gain 0.024467; relative gain 1.414%.
- Current gradient at lambda=0.01: mean 1.06241, median 1.20105, positive fraction 0.750.
- Normalized gradient at lambda=0.01: mean 0.584424, median 0.877007, positive fraction 0.700.
- Author parity max absolute log-probability difference: 0.
- Paper parity max absolute log-probability difference: 0.
- Peak process allocation during cache build: 4046.8 MiB.

### NLL-proxy optimum by alpha

| Fusion | Alpha (help,safe) | Best lambda | Mean NLL |
|---|---:|---:|---:|
| current | 1.00,0.00 | 0.0 | 1.718851 |
| current | 0.75,0.25 | 0.0 | 1.724845 |
| current | 0.50,0.50 | 0.0 | 1.730839 |
| current | 0.25,0.75 | 0.0 | 1.736833 |
| current | 0.00,1.00 | 0.0 | 1.742827 |
| normalized | 1.00,0.00 | 0.0 | 1.718851 |
| normalized | 0.75,0.25 | 0.0 | 1.724845 |
| normalized | 0.50,0.50 | 0.0 | 1.730839 |
| normalized | 0.25,0.75 | 0.0 | 1.736833 |
| normalized | 0.00,1.00 | 0.0 | 1.742827 |

## Interpretation

- Under gradient descent, a positive `dNLL/dlambda` pushes lambda downward. The full lambda/alpha breakdown is in `gradient_summary.csv`.
- Changes in entropy, maximum probability, and score standard deviation in `fusion_scale_summary.csv` quantify the scale/temperature confound; normalized fusion is only a diagnostic control.
- Oracle gain is a finite-grid, per-case upper bound under teacher forcing. Lambda diversity and per-alpha optima indicate proxy headroom, not a deployable routing result.
- Preliminary classification: **LOW-VRAM DIAGNOSTIC SUGGESTS LAMBDA COLLAPSE OBJECTIVE ISSUE**.

## Limitations

- Only 20 validation cases.
- Teacher-forced prefixes and continuation NLL only.
- NLL is a cheap proxy, not final preference-alignment evidence.
- No autoregressive generation was performed.
- No Beaver reward/cost evaluator was loaded.
- This result alone must not be used for a final scientific GO/NO-GO decision.

LOW-VRAM DIAGNOSTIC SUGGESTS LAMBDA COLLAPSE OBJECTIVE ISSUE
