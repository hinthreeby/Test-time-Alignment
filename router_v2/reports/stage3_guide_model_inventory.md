# Stage 3 Autoregressive Guide Model Inventory

Generated: 2026-08-15

## Decision

No existing local checkpoint is both:

1. an autoregressive model that emits independent next-token logits over the
   base vocabulary; and
2. trained for the RAD sentiment objective.

The selected scientific path is therefore:

~~~text
base:
  models/gpt2-medium

new guide architecture:
  GPT2LMHeadModel + LoRA

new guide output:
  results/router_v2/sentiment_guide/final_adapter

objective:
  positive-continuation causal NLL

training source:
  dataset/RAD_train/router_amazon_polarity/train.jsonl

validation source:
  dataset/RAD_train/router_amazon_polarity/validation.jsonl

negative direction audit:
  dataset/RAD_train/amazon_polarity/test/data-00000-of-00001.arrow
  official test rows with label = negative
~~~

The new guide does not exist yet. Host GPU training and validation are required.

## Hash Definitions

- "tokenizer hash" below is SHA-256 of canonical sorted (token, token_id)
  entries returned by get_vocab().
- Single-file checkpoint hashes are direct SHA-256 hashes of the runtime weight
  file.
- Sharded checkpoints list the index hash and every weight shard hash.
- Hugging Face blob symlink names are content-addressed SHA-256 identifiers;
  all referenced targets exist locally.

## Tokenizer Families

### GPT-2

~~~text
vocab size:       50,257
vocab SHA-256:    79ff2372d75e36ad401b44e1a72214636dfdcc4ddeeefa498463b0a3b2a5fd76
bos token ID:     50,256
eos token ID:     50,256
raw base padding: none
runtime padding:  EOS, right padding, padding labels masked
~~~

Programmatic equality:

~~~text
models/gpt2-medium.get_vocab()
  == models/genarm-gpt2-medium-hh.get_vocab()
  == models/genarm-gpt2-medium-hh-adapter.get_vocab()
  == models/gpt2-small.get_vocab()
  == models/gpt2-large.get_vocab()
~~~

All comparisons returned True.

### Tulu/Llama

Tulu base and the local Tulu HH adapter use:

~~~text
vocab size:     32,000
vocab SHA-256:  d182db49d05efb7a55421d033bd6d75099b1d7af5b34b7ff2a55215195e39d7a
bos token ID:   1
eos token ID:   2
~~~

models/AutoregressiveRM-tulu2-7b instead uses:

~~~text
vocab size:     32,001
vocab SHA-256:  4ac8ca3556785ef05bd7226986750b251fc8019bafee2fadeec03f495fab3088
extra token:    <pad>, ID 32,000
~~~

Programmatic equality between Tulu base and the full autoregressive RM returned
False. Stage 3 does not permit a silent token mapping.

## Autoregressive Candidates

### models/gpt2-medium/

~~~text
model architecture:          GPT2LMHeadModel
tokenizer path:              models/gpt2-medium
vocab size:                  50,257
tokenizer hash:              79ff2372d75e36ad401b44e1a72214636dfdcc4ddeeefa498463b0a3b2a5fd76
base-model compatibility:    selected base
actual forward input shape:  [1, 3]
actual output tensor shape:  [1, 3, 50,257]
actual last-token shape:     [1, 50,257]
autoregressive:              yes
training objective:          generic GPT-2 causal language modeling
checkpoint hash:             fc5a354a19255ad494f3d71549390baca1ccf61d1d822b9408971705c687c9cd
usable_for_faithful_taro:    base only; no as sentiment guide
reason:                      no independent sentiment alignment
~~~

This is the selected base and initialization for the new guide.

### models/genarm-gpt2-medium-hh/

~~~text
model architecture:          GPT2LMHeadModel
tokenizer path:              models/genarm-gpt2-medium-hh
vocab size:                  50,257
tokenizer hash:              79ff2372d75e36ad401b44e1a72214636dfdcc4ddeeefa498463b0a3b2a5fd76
base-model compatibility:    exact GPT-2 token IDs
actual forward input shape:  [1, 3]
actual output tensor shape:  [1, 3, 50,257]
actual last-token shape:     [1, 50,257]
autoregressive:              yes
training objective:          GenARM preference loss on HH-RLHF
attribute:                   HH/helpfulness preference
checkpoint hash:             ebd116adc8b5d7d5e30861f7f5886ce2197261067803295326e4b75481b91b19
usable_for_faithful_taro:    engineering HH smoke only; no for RAD sentiment
reason:                      task/attribute mismatch
~~~

The full-vocabulary output contract is real and finite. Calling this checkpoint
a RAD sentiment guide would be scientifically invalid.

### models/genarm-gpt2-medium-hh-adapter/

~~~text
model architecture:          PeftModelForCausalLM over GPT2LMHeadModel
tokenizer path:              models/genarm-gpt2-medium-hh-adapter
vocab size:                  50,257
tokenizer hash:              79ff2372d75e36ad401b44e1a72214636dfdcc4ddeeefa498463b0a3b2a5fd76
base-model compatibility:    exact; base path is models/gpt2-medium
output tensor shape:         [B, T, 50,257], config/code verified
autoregressive:              yes
training objective:          GenARM preference loss on HH-RLHF
attribute:                   HH/helpfulness preference
adapter checkpoint hash:     0ffeefa2b63d69e01859f21fa94a1904fd393f2249b004501e20dcb9016b2fc3
adapter config hash:         dec6dd5b6a95517231608700fd25558e0e7d5ec68f1d1e927e0324b470330ee1
usable_for_faithful_taro:    engineering HH smoke only; no for RAD sentiment
reason:                      task/attribute mismatch
~~~

### models/genarm-gpt2-medium-sentiment-adapter/

~~~text
model architecture:          absent
tokenizer path:              absent
vocab size:                  unknown
tokenizer hash:              unavailable
base-model compatibility:    untestable
output tensor shape:         unavailable
autoregressive:              unavailable
training objective:          no metadata
checkpoint hash:             unavailable
usable_for_faithful_taro:    no
reason:                      directory is empty (4 KB directory entry only)
~~~

This empty directory is not a checkpoint and is not modified by the V2
pipeline.

### models/tulu-2-7b/

~~~text
model architecture:          LlamaForCausalLM
tokenizer path:              models/tulu-2-7b
vocab size:                  32,000
tokenizer hash:              d182db49d05efb7a55421d033bd6d75099b1d7af5b34b7ff2a55215195e39d7a
base-model compatibility:    self-compatible
output tensor shape:         [B, T, 32,000], config/index verified
autoregressive:              yes
training objective:          Tulu instruction causal LM
checkpoint index hash:       e572e08c4d4e81c7916197f6fcd2956a2f05e5919f28d72c9ba4f351efae1e29
weight shard hashes:         6d90e5350a50e3a1ae608eadf4de08d5336b387561c44094e558789c0e1480d6
                             3eb1b1833fa5b5b8bc9f0ffde237012fead0c18536047fa4c2fcfb8abc3c6425
usable_for_faithful_taro:    base only; no as sentiment guide
reason:                      generic instruction LM, not sentiment aligned
~~~

### models/AutoregressiveRM-tulu2-7b/

~~~text
model architecture:          LlamaForCausalLM
tokenizer path:              models/AutoregressiveRM-tulu2-7b
vocab size:                  32,001
tokenizer hash:              4ac8ca3556785ef05bd7226986750b251fc8019bafee2fadeec03f495fab3088
base-model compatibility:    fails exact compatibility with Tulu base
output tensor shape:         [B, T, 32,001], config/index verified
autoregressive:              yes
training objective:          GenARM/ARM preference training on UltraFeedback
attribute:                   general helpfulness/preference
checkpoint index hash:       d7080046cf82f1afea412b65fee47b0802999f0132e66ef4668c9d8ce4614573
weight shard hashes:         7bfdce63b299a66b48481c32a5e7ccf7a00a1b057572b44315167dce6ec371a8
                             c012ad707ece4471c157ea341c4a8eee8f5ca70d8ff0023c18c4901d2f83dd92
                             56ba8eb4d645769f5684dd7f4c47a93786d65bfcc94506d70badfac1d84d47d0
usable_for_faithful_taro:    no
reason:                      vocab mismatch and non-sentiment objective
~~~

### models/genarm-tulu2-hh/

~~~text
model architecture:          PeftModelForCausalLM over LlamaForCausalLM
tokenizer path:              models/genarm-tulu2-hh
vocab size:                  32,000
tokenizer hash:              d182db49d05efb7a55421d033bd6d75099b1d7af5b34b7ff2a55215195e39d7a
base-model compatibility:    exact with models/tulu-2-7b
output tensor shape:         [B, T, 32,000], config/code verified
autoregressive:              yes
training objective:          GenARM preference loss on HH-RLHF
attribute:                   HH/helpfulness preference
adapter checkpoint hash:     e5121712e702a36cccde1c085fd50fea0807f0d3dd14dc5ac46c15460266714c
adapter config hash:         3c0256a759c936ebe39dadb876520647831338d374b50fe06781893b49a07485
usable_for_faithful_taro:    engineering HH smoke only; no for RAD sentiment
reason:                      task mismatch and unnecessary 7B runtime cost
~~~

### Other generic local causal LMs

| Path | Architecture | Vocab / tokenizer hash | Output | Checkpoint hash | Sentiment guide decision |
|---|---|---|---|---|---|
| models/gpt2-small | GPT2LMHeadModel | 50,257 / 79ff2372... | [B,T,50257] | 248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707 | no; generic LM |
| models/gpt2-large | GPT2LMHeadModel | 50,257 / 79ff2372... | [B,T,50257] | 5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce | no; generic LM and larger than needed |
| models/llama-2-7b-hf | LlamaForCausalLM | 32,000 / d182db49... | [B,T,32000] | shards 4ec71fd53e99766de38f24753b30c9e8942630e9e576a1ba27b0ec531e87be41, 41780b5dac322ac35598737e99208d90bdc632a1ba3389ebedbb46a1d8385a7f | no; generic LM |

All are valid causal language models, but none supplies an independent
sentiment-aligned distribution without new training.

The separate case-sensitive path models/Llama-2-7b-hf contains only LICENSE
and README files (36 KB total). It has no config, tokenizer, or weights and is
not a runnable model candidate.

## GenARM Code and Data Inspection

Method/GenARM/training_trl/train_arm_gpt2_medium.py and its ARM trainer optimize
an autoregressive preference objective from chosen/rejected pairs.
Method/GenARM/generate_arm_gpt2_medium.py confirms that its output is consumed
as causal-LM vocabulary logits. This is the closest existing method family to
the required guide contract.

The only local paired GenARM dataset is dataset/GenARM/full-hh-rlhf, whose
attribute is HH/helpfulness. PKU-SafeRLHF is a safety/preference source for a
different track. Neither is a sentiment training source. The sentiment scripts
under language-model-arithmetic compose generic LMs with external classifier
scores at inference time; they do not provide a local, independently trained
sentiment causal-LM checkpoint.

Therefore the V2 wrapper reuses the causal-LM plus LoRA architecture but does
not claim to reproduce pairwise ARM training. It uses the explicitly documented
positive-continuation objective allowed for this Stage 3 prerequisite.

## Rejected Non-Autoregressive Models

### models/rad_rm_sentiment/

~~~text
model implementation:        custom GPT2RewardModel
tokenizer hash:              79ff2372d75e36ad401b44e1a72214636dfdcc4ddeeefa498463b0a3b2a5fd76
runtime output:              scalar sequence reward
checkpoint hash:             9fafd202d3a322f77ce03e682bec9a9443b19bb24f461003268966ddaa2fd298
usable_for_faithful_taro:    no
reason:                      no vocabulary-sized token distribution
~~~

The saved config names GPT2LMHeadModel, but the RAD loader replaces the LM
head with a scalar reward head. The runtime implementation, not the config
label, determines acceptance.

### Other rejected heads

| Path | Architecture/output | Tokenizer hash | Checkpoint hash | Rejection |
|---|---|---|---|---|
| models/gpt2-large-helpful-rm | GPT2ForSequenceClassification, scalar | 79ff2372... | 7e79f03e20870598bf264a18f7453c97e991f3538576b6cfb78aee52cedd44ef | scalar helpfulness |
| models/cd_prefix_scorer | GPT2Model plus scalar value head | 79ff2372... | backbone 0222ca428f2af8c36b2afb34712b11a03f433be060a16b65a6be098ece4635b2 | no LM head |
| models/sentiment-rm-sst2 | DistilBERT classifier | 904452de... | 7c3919835e442510166d267fe7cbe847e0c51cd26d9ba07b89a57b952b49b8aa | classifier probabilities |
| models/sentiment-roberta-large-english | RoBERTa classifier | f37ce98f... | 805688de73481b1175b1e06f05a971144dceb7e0064b64a84b78a3685eb3209d | classifier probabilities |
| models/helpfulness-deberta-v3-large | DeBERTa classifier | e20d4af1... | bb3306045ebdcf84e3dbd51a55338b6ed255a832cfe179bbd65051b282c5e244 | classifier probabilities |
| models/toxic-bert | BERT classifier | 904452de... | 2c272885d24138df70bff1b3cd944a999bd6b41dad33209730aa8ba074f6ad09 | classifier probabilities |

No scalar/classifier output is repeated, broadcast, or reconstructed into fake
vocabulary logits.

## Data and Objective Inventory

The processed sentiment source contains only positive examples:

| Split | Rows | Label counts | Token-boundary mismatches |
|---|---:|---|---:|
| train | 20,000 | {1: 20000} | 0 |
| validation | 2,000 | {1: 2000} | 0 |
| dev | 500 | {1: 500} | 0 |

Every prompt/continuation pair satisfies:

~~~python
encode(prompt) + encode(continuation) == encode(prompt + continuation)
~~~

The source builder originally split normalized positive Amazon reviews at GPT-2
token boundaries. The new objective is therefore:

~~~text
L_guide =
  -sum over positive training samples
   sum over continuation tokens plus EOS
    log pi_guide(y_t | prompt, y_<t)
~~~

Prompt labels and right-padding labels are masked. The validation split is not
used for training. This is positive-domain causal LM adaptation, not the
pairwise GenARM ARM loss; the pairwise objective is unavailable because the
processed source contains no rejected continuation paired to the same prompt.

The direction audit uses only label-0 reviews from the official Amazon
Polarity test split, not the heterogeneous `rad_benchmark` negative bucket.
Its Arrow SHA-256 is
`404683c77cd9330d5e7c60dd28e4ceaca435bdb72e082182b93dc3aab8226e92`.

This choice follows the Stage 3 allowance for an autoregressive model trained
on positive continuations and retains the GenARM causal-LM/LoRA architecture.

## Acceptance Gate Before Cache Extraction

The new guide remains unusable until
router_v2/reports/sentiment_guide_validation.json reports all of:

- actual output [B,T,50257] and next-token output [B,50257];
- exact get_vocab() equality and matching BOS/EOS/padding behavior;
- separate base and adapter checkpoint hashes;
- both models frozen under torch.inference_mode();
- positive validation NLL lower than the unaligned base by more than 0.01
  nats per supervised token;
- positive alignment delta greater than the held-out negative alignment delta
  by more than 0.01 nats per supervised token.

Until then, Stage 3 status is:

~~~text
HOST EXECUTION REQUIRED
~~~
