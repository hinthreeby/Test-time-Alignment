# Full Project Context — Token-Level Adaptive Router for RAD

## 1. Research objective

Design and evaluate a token-level adaptive router for reward-guided decoding. The router predicts a reward coefficient \(\beta_t\) at each decoding step to improve alignment reward while preserving coherence, fluency, diversity, and latency.

Primary research question:

> Can adaptive weighting adjust the reward coefficient at each decoding step to maximize alignment reward while maintaining coherence and fluency better than fixed weighting and heuristic schedules?

## 2. Starting point

The implementation starts from the RAD repository and its positive-sentiment control setup.

Initial experimental scope:

- Task: positive sentiment steering.
- Base language model: GPT-2 Large.
- Reward model: GPT-2 Small RAD sentiment reward model.
- Base LM and reward model are frozen.
- Only the router is trainable.
- RAD candidate size: top-k = 20.
- Router input: 20 base logits + 20 candidate reward scores.
- Router architecture: MLP 40 → 64 → 1.
- Output: \(\beta_t = \beta_{max}\,\sigma(g_\theta(h_t))\).

## 3. Existing data layout

The project already contains Amazon Polarity locally. Do not download another copy and do not use SST-2 for version 1.

Expected project layout:

```text
dataset/
├── rad_benchmark/
│   ├── negative_prompts.jsonl
│   ├── neutral_prompts.jsonl
│   ├── positive_prompts.jsonl
│   └── nontoxic_prompts-10k.jsonl
└── RAD_train/
    ├── amazon_polarity/
    └── sst2/
```

Data roles:

- `dataset/RAD_train/amazon_polarity/`: source corpus for router train and validation data.
- `dataset/rad_benchmark/negative_prompts.jsonl`: primary held-out benchmark for steering negative prefixes toward positive continuations.
- `dataset/rad_benchmark/neutral_prompts.jsonl`: held-out robustness benchmark.
- `dataset/rad_benchmark/positive_prompts.jsonl`: held-out preservation benchmark.
- `dataset/rad_benchmark/nontoxic_prompts-10k.jsonl`: reserved for later detoxification experiments; do not use in the first sentiment-router experiment.
- `dataset/RAD_train/sst2/`: not used in version 1; it may be retained for later ablations.

The Amazon Polarity source format must be inspected before preprocessing. The preparation script must support the actual files already present rather than assuming a new download layout.

## 4. Router-training data policy

Use only positive Amazon Polarity examples to create teacher-forcing prompt–continuation pairs for version 1.

Recommended target:

```text
Train:       20,000 examples
Validation:   2,000 examples
Optional dev:   500 examples
```

The exact counts may be reduced for an initial smoke test, but the final split must be deterministic and documented.

Required processed outputs:

```text
dataset/RAD_train/router_amazon_polarity/
├── train.jsonl
├── validation.jsonl
├── dev.jsonl                 # optional
└── data_report.json
```

Every benchmark file under `dataset/rad_benchmark/` must remain read-only and held out from router training, validation, split selection, and hyperparameter tuning.

## 5. Core decoding equation

For candidate token \(v_i\) at decoding step \(t\):

\[
S_t(v_i)=\ell_t(v_i)+\beta_t r_t(v_i),
\]

where:

- \(\ell_t(v_i)\): base-model logit or log-probability;
- \(r_t(v_i)\): reward-model score for appending candidate \(v_i\);
- \(\beta_t\): router-predicted adaptive reward coefficient.

The guided distribution is:

\[
p_t(v_i)=\operatorname{softmax}(S_t(v_i)).
\]

## 6. Router definition

At each token step:

1. Obtain top-k candidate tokens from the frozen base LM.
2. Score those candidates with the frozen RAD reward model.
3. Normalize base candidate logits and reward scores separately.
4. Concatenate them into a 40-dimensional feature vector.
5. Pass the feature vector through an MLP:

\[
h_t=[\operatorname{Norm}(\ell_t);\operatorname{Norm}(r_t)]\in\mathbb{R}^{40},
\]

\[
a_t=\sigma\left(W_2\tanh(W_1h_t+b_1)+b_2\right),
\]

\[
\beta_t=\beta_{max}a_t.
\]

Initial architecture:

```text
Input:   40
Hidden:  64
Output:   1
Act.:    Tanh
Gate:    Sigmoid
```

## 7. Training strategy

Use teacher forcing with prompt–continuation pairs produced from positive Amazon Polarity examples.

At each continuation token:

- Feed prompt plus gold prefix into both frozen models.
- Get top-k candidates from the base LM.
- Always add the gold token to the training candidate set if absent.
- Compute reward scores for all candidates.
- Run the router to obtain \(\beta_t\).
- Compute guided candidate scores.
- Compute negative log-likelihood for the gold candidate.
- Backpropagate only through the router.

Initial loss:

\[
\mathcal{L}_{router}=\mathcal{L}_{NLL}.
\]

Do not add entropy, KL, smoothness, RL, or extra evaluators until the minimal pipeline works reliably.

## 8. Data separation and leakage control

Maintain strict separation among:

- Reward-model training data.
- Amazon Polarity examples used for router training.
- Amazon Polarity examples used for router validation/dev.
- Final RAD benchmark prompts.

No normalized prompt, continuation, or complete source text from router data may overlap with the RAD benchmark.

Leakage checks must include at least:

- exact normalized-string matching;
- source-row identity checks where metadata exists;
- train/validation duplicate checks;
- near-duplicate checks when practical.

Do not create the final test benchmark from Amazon Polarity. The existing RAD benchmark remains the held-out evaluation set.

## 9. Offline feature cache

Because reward scoring is expensive, extract token-level training features once and cache them.

Each cached token step must include at least:

```text
base_logits        float[k]
reward_scores      float[k]
candidate_ids      int[k]
gold_token_id      int
gold_index         int
position           int
sample_id          str/int
continuation_mask  bool
```

Preferred format: `.pt`, Arrow, or another compact tensor format. Avoid JSON for large floating-point arrays.

## 10. Evaluation protocol

Compare:

1. Base LM.
2. Several fixed-beta RAD settings.
3. Best fixed beta selected on validation only.
4. Existing heuristic adaptive weighting methods.
5. Learned token-level router.

Evaluation prompt groups:

- Primary: `negative_prompts.jsonl`.
- Secondary: `neutral_prompts.jsonl`.
- Preservation check: `positive_prompts.jsonl`.
- Excluded in version 1: `nontoxic_prompts-10k.jsonl`.

Required metrics:

- Alignment reward.
- Independent sentiment score.
- Perplexity or independent fluency score.
- Repetition.
- Distinct-1/2/3.
- Output length.
- Latency and throughput.
- Full `beta_history` for each generated sample.

The independent sentiment evaluator must not be the same reward model used during decoding.

## 11. Reproducibility requirements

- Fixed random seeds.
- Saved configuration for each run.
- Exact model identifiers and checkpoint hashes when available.
- Amazon Polarity source-file hashes and processed split statistics.
- Validation-only hyperparameter selection.
- No tuning on the held-out RAD benchmark.
- Save best router checkpoint and last checkpoint.
- Save logs, metrics, and generated outputs.

## 12. Definition of success for version 1

Version 1 is successful when all conditions hold:

1. Amazon Polarity is converted reproducibly into clean router train/validation data.
2. End-to-end training runs without updating the base LM or reward model.
3. Router validation NLL improves over initialization.
4. \(\beta_t\) is not constant across every token and sample.
5. Learned routing outperforms at least one fixed-beta baseline.
6. Evaluation outputs include full beta histories and all required metrics.
7. Results are reproducible from a clean environment using documented commands.

## 13. Non-goals for version 1

Do not implement yet:

- SST-2 as a second router-training corpus.
- Detoxification using `nontoxic_prompts-10k.jsonl`.
- PPO, GRPO, or REINFORCE.
- Multiple reward models.
- GRU/LSTM/Transformer router.
- Dynamic KL budgets.
- Human-in-the-loop labels for beta.
- Joint fine-tuning of the base LM or reward model.
- Method-agnostic routing across RAD, GenARM, CD, and ARGS.

These belong to later research stages after the minimal TARO-like RAD router is stable.
