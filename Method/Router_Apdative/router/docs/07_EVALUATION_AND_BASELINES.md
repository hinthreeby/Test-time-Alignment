# Stage 7 — Evaluate Fixed, Heuristic, and Learned Weighting

## Goal

Determine whether learned token-level adaptive weighting improves the alignment–quality trade-off over fixed beta and heuristic schedules.

## Evaluation datasets

Use three held-out RAD benchmark groups:

- negative prompts;
- neutral prompts;
- positive prompts.

Primary analysis should emphasize negative-to-positive steering, while the other groups test robustness and over-steering.

## Systems to compare

1. Base GPT-2 Large.
2. Fixed-beta RAD sweep.
3. Best fixed beta selected on validation only.
4. Every existing heuristic adaptive baseline.
5. Learned token-level router.

Recommended fixed-beta grid:

```text
0, 5, 10, 20, 30, 50, 75, 100
```

Adjust only if validation shows the useful range lies elsewhere. Do not select beta using benchmark results.

## Required metrics

### Alignment

- RAD reward-model score for diagnostics only.
- Independent sentiment evaluator score.
- Positive-label rate.
- Pairwise win rate against base and best fixed beta.

### Fluency and coherence

- Perplexity from an evaluator LM not used as the base generator when possible.
- Independent coherence score or judge.
- Grammar/fluency score if available.

### Degeneration

- repetition rate;
- repeated n-grams;
- Distinct-1/2/3;
- average generated length;
- premature EOS rate.

### Efficiency

- total latency;
- latency per generated token;
- reward-model scoring time;
- router overhead;
- throughput.

### Router behavior

- beta mean/std/min/max;
- beta by generation position;
- beta by prompt class;
- beta by selected-token sentiment;
- beta versus base entropy;
- beta versus reward gap;
- fraction near zero/max.

## Independent evaluator rule

The main reported sentiment result must come from an evaluator that is not the same reward model used during RAD decoding. Results from the decoding RM may be reported only as internal diagnostics.

## Statistical reporting

- Use the same prompts and random seeds across systems.
- Report mean, standard deviation, and bootstrap confidence intervals.
- Use paired significance tests where appropriate.
- Save per-sample metrics, not only aggregate averages.

## Core research plot

Create a Pareto-style plot:

```text
x-axis: fluency degradation, perplexity, or coherence loss
y-axis: independent alignment reward / positive rate
```

The learned router is successful when it moves the frontier outward relative to fixed and heuristic methods.

## Output files

```text
results/
├── generations.jsonl
├── per_sample_metrics.jsonl
├── aggregate_metrics.csv
├── beta_analysis.csv
├── latency_breakdown.csv
└── plots/
```

## Acceptance criteria

- Best fixed beta is chosen without benchmark leakage.
- Independent sentiment evaluation is included.
- All systems use matched prompts and seeds.
- Per-sample outputs and beta histories are retained.
- Confidence intervals are reported.
- At least one trade-off plot is generated.

## Prompt for the coding AI

Build a reproducible evaluation suite comparing base LM, a fixed-beta sweep, best fixed beta, heuristic adaptive schedules, and the learned router on held-out negative, neutral, and positive RAD prompts. Use an independent sentiment evaluator, measure fluency, coherence, repetition, diversity, length, latency, and router behavior, and produce per-sample outputs, aggregate CSVs, confidence intervals, significance tests, and Pareto trade-off plots.
