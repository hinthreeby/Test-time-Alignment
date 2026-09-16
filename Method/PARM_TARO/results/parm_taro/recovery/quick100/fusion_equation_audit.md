# Fusion equation audit

This audit reads the executable source, not the README. No protected file was
modified. There are two distinct things called “PARM static” in this project;
they must not be conflated.

## Exact equations

| Path | Input quantities | Implemented fusion | Class |
|---|---|---|---|
| Original PARM author generation | `PromptedLLM` emits normalized log-probabilities | `log_softmax((log p_base + log p_guide) / 2)` because `formula = M_base + M_reward`, `Sum.norm() = 2`, and `Operator.normalize()` divides by the norm | C (normalized additive product-of-experts) |
| Stage-10 `parm_static` | base logits and PBLORA-guide logits, each converted to log-probabilities | `log_softmax(log p_base + 1 * log p_guide)` | A |
| PARM-TARO generation (TARO or V2 provider) | base/guide logits; router sees their log-probabilities | `log_softmax(log p_base + lambda_t * log p_guide)` | A |
| PARM-TARO training route | cached/live base and guide log-probabilities | `log_softmax(log p_base + lambda_t * log p_guide)` | A |
| Standalone TARO/RAD router | raw base and guide logits | `(1-lambda_t) z_base + lambda_t z_guide` | B |
| Standalone Smart Router V2/RAD | raw base and guide logits | `z_base + lambda_t (z_guide-z_base)` | B |

The standalone TARO/V2 convex-mixing helpers are not used to fuse PARM-TARO
generation. In PARM-TARO, those router classes supply only `lambda_t`; the
integration layer applies PARM-specific equation A.

## Consequence at lambda=1

Original PARM author generation and PARM-TARO do **not** have equal normalized
log-probabilities:

```text
original author: log_softmax((log p_base + log p_guide) / 2)
PARM-TARO:       log_softmax( log p_base + log p_guide)
```

The factor 1/2 is not a vocabulary-wide additive constant, so normalization
does not remove it. It does preserve token ordering, hence top-1, top-k sets,
and greedy output normally agree when all other inputs are identical. Stage-10
`parm_static`, however, is `FixedLambdaProvider(1.0)` passed into the same
`ParmTaroDecoder`; it is exactly equal to router-bypass lambda=1 by construction.

## Ordering and controls

| Item | Original PARM author path | PARM-TARO Stage 10 path |
|---|---|---|
| Base quantity | `log_softmax(base logits)` | `log_softmax(base logits)` |
| Guide quantity | `log_softmax(PBLORA logits)` | `log_softmax(PBLORA logits)` |
| Beta | No explicit inference-time beta. `beta_safe=beta_help=0.5` belongs to PBLORA preference training and is embodied in adapter weights. | No explicit inference-time beta; same frozen PBLORA adapter. |
| Lambda | No router lambda; guide coefficient is 1. | Applied to guide log-probabilities before final normalization. |
| Temperature | Formula normalizes first. Author script uses `temperature=0` for greedy; optional `normalize_logit` sets 1. | Fused normalized log-probabilities are divided by temperature (Stage 10: 1.0), then decoded. |
| top-k/top-p | Applied after formula fusion; historical author call uses `top_k=0`, `top_p=1`. | For sampling, top-k is applied after fusion/temperature. For greedy, `argmax` is direct and top-k is not applied. No top-p implementation in this decoder. |
| Prompt template | `BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:` for the non-Alpaca-65B branch. | Decoder tokenizes exactly the string supplied. Stage-10 engine supplies the dataset prompt without adding the author template. |
| Alpha injection | Before load, author script copies adapter files to a cache directory and rewrites `pref_vec_init=[harmlessness, helpfulness]`. | In memory, `set_parm_preference` writes every `pref_vec`; named `[helpfulness, harmlessness]` is reordered to `[harmlessness, helpfulness]`. Requested alpha always controls the guide; shuffled/fixed alpha only changes the router input. |
| Base/guide adapter state | Separate plain base model plus PBLORA guide model. | One physical backbone: base pass is inside `PeftModel.disable_adapter()`; guide pass has PBLORA enabled. The wrapper checks and restores adapter state after every base pass. |
| Vocabulary | Author `PromptedLLM` truncates model outputs to tokenizer length. | `PARMTokenAlignment` validates base vocabulary and truncates the guide to the base vocabulary prefix. |

## Source evidence

- `PARM/code/evaluation/generate_outputs.py:125-140,161-183`
- `PARM/language-model-arithmetic/src/model_arithmetic/runnable_operators.py:124-138,507-522`
- `PARM/language-model-arithmetic/src/model_arithmetic/operators.py:105-120,384-407`
- `PARM_TARO/decoding/static_regression.py:9-32`
- `PARM_TARO/decoding/adaptive.py:155-201`
- `PARM_TARO/evaluation/generation.py:22-43,153-177`
- `PARM_TARO/training/routing.py:24-35,58-105`
- `PARM_TARO/training/runtime.py:348-421`
- `router_v2/model.py:115-129`
- `router_v2/smart_model.py:262-287`

## Diagnostic protocol

`PARM_TARO/recovery/run_lambda1_parity.py` independently evaluates the author
equation and the PARM-TARO fixed-lambda equation. The CPU source probe is safe
to run now. The real-model mode selects the first 8 validation records,
applies one shared deterministic prompt template and alpha, uses greedy decoding
for at most 16 new tokens, and writes full per-token tensors to
`lambda1_token_traces.pt` plus summaries to `lambda1_parity.json` and Markdown.

The CPU source probe has been executed and proves normalized-distribution
mismatch without loading either model. The requested real 8-prompt trace was
not run by the agent: GPU access is unavailable by design, and the exact custom
PBLORA checkpoint at `results/parm_taro/checkpoints/parm_pku_pblora/` is not
present. Public Tulu weights alone cannot replace that checkpoint.

CPU source-equation rerun:

```bash
CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false \
/home/jupyter-iec2024se10/miniforge3/envs/tta/bin/python \
  PARM_TARO/recovery/run_lambda1_parity.py
```

Real validation micro-run after restoring/reproducing the exact PBLORA:

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
/home/jupyter-iec2024se10/miniforge3/envs/tta/bin/python \
  PARM_TARO/recovery/run_lambda1_parity.py \
  --run-model --device cuda --num-prompts 8 --max-new-tokens 16 \
  --alpha 0.5 0.5 --prompt-template parm_author
```
