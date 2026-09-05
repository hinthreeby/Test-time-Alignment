# Reward Decoding Project

Repo này triển khai nhiều phương pháp decoding và alignment dựa trên reward model / reward-guided generation.

Các phương pháp chính trong repo:

- CD (Controlled Decoding)
- RAD (Reward-Augmented Decoding)
- GenARM (Generative Autoregressive Reward Model)
- ARGS (Arguments / reward-based decoding utilities)
- MultiSignal (kết hợp nhiều signal bằng controller)

---

## Tổng hợp training dataset, model và trạng thái train/finetune

| Method | Dataset dùng để train / eval | Model / checkpoint chính trong repo | Có train / finetune model không |
| --- | --- | --- | --- |
| CD | `dataset/cd_train/hh_prompts.jsonl` để train prefix scorer; script dùng prompt generation từ HH-style data | `models/gpt2-large` (base LM), `models/gpt2-small` (scorer backbone), `models/sentiment-roberta-large-english` (reward model), `models/cd_prefix_scorer` | Có. Prefix scorer được train từ đầu bằng script `Method/CD/train/train_fudge.py` và `Method/CD/train/train_cdq.py`; checkpoint sinh ra ở `Method/CD_new/checkpoints/` |
| RAD | Reward model được train bằng YAML config trong `Method/RAD` với dữ liệu sentiment / toxicity / reward datasets; eval dùng `dataset/rad_benchmark` và prompt benchmark | `models/gpt2-large` (base LM), `models/gpt2-small` (base RM), `models/rad_rm_sentiment` (reward model checkpoint), `models/sentiment-rm-sst2`, `models/sentiment-roberta-large-english`, `models/toxic-bert` | Có. Reward model có thể train/finetune; decoder RAD không có model riêng mới trong `models/` ngoài base LM và RM |
| GenARM | Training scripts dùng external HF datasets như HH, SafeRLHF, Alpaca; không có 1 file dataset cố định trong repo | `models/genarm-gpt2-medium-hh`, `models/genarm-gpt2-medium-hh-adapter`, `models/genarm-gpt2-medium-sentiment-adapter`, `models/genarm-tulu2-hh`, `models/AutoregressiveRM-tulu2-7b` | Có. Có model GenARM đã được train sẵn trong `models/`; các script `Method/GenARM/...` chạy generation hoặc train mới nếu cần |
| ARGS | Dùng external datasets từ HuggingFace như `Dahoas/full-hh-rlhf` và `stanfordnlp/SHP`; không có dataset local cố định trong repo | Không có một model ARGS riêng trong `models/`; chạy dựa trên LM + RM bên ngoài / config từ CLI | Không có model ARGS riêng được bundle trong `models/`; chủ yếu là generation script + reward steering |
| MultiSignal | `dataset/multisignal_train/train.jsonl` và `dataset/multisignal_train/validation.jsonl` cho cache/train; `dataset/rad_benchmark/...` cho generation test | `models/gpt2-large` (base LM), controller checkpoint `Method/MultiSignal/checkpoints/controller_4signal.pt` | Có. Controller MLP là model được train bằng `Method/MultiSignal/core/train.py`; nếu chưa train thì checkpoint chưa tồn tại |

### Ghi chú nhanh

- Nếu một method không có dataset hay model trong repo, ta ghi là `không có` / `external HF dataset` / `không bundle model riêng`.
- Đối với các method có base LM dùng chung, như `gpt2-large`, `gpt2-small`, các model này được dùng như backbone cho nhiều method khác nhau.
- Nếu bạn chỉ cần tổng hợp để report paper, nên dùng dạng trên để phân biệt rõ: `base backbone`, `reward model`, `trained controller`, `external dataset`.

---

## 1. Môi trường

Kích hoạt môi trường phù hợp:

```bash
conda activate base
# hoặc
conda activate multisignal
```

Vào thư mục project:

```bash
cd "/home/jupyter-iec2024se10/Reward Decoding"
```

---

## 2. CD (Controlled Decoding)

Thư mục triển khai:

- [Method/CD](Method/CD)
- [Method/CD/generate_cd.py](Method/CD/generate_cd.py)

### Cấu trúc chính

```text
Method/CD/
├── decoding/
│   ├── tokenwise.py
│   └── blockwise.py
├── train/
│   ├── train_fudge.py
│   └── train_cdq.py
├── generate_cd.py
└── models/
```

### Train CD-FUDGE

```bash
python Method/CD/train/train_fudge.py
```

### Sinh CD-FUDGE tokenwise

```bash
python Method/CD/generate_cd.py --scorer fudge --mode tokenwise --num-prompts 100
```

### Sinh CD-FUDGE blockwise

```bash
python Method/CD/generate_cd.py --scorer fudge --mode blockwise --num-prompts 100
```

### Train CD-Q

```bash
python Method/CD/train/train_cdq.py
```

### Sinh CD-Q tokenwise

```bash
python Method/CD/generate_cd.py --scorer cdq --mode tokenwise --num-prompts 100
```

### Sinh CD-Q blockwise

```bash
python Method/CD/generate_cd.py --scorer cdq --mode blockwise --num-prompts 100
```

### Kết quả lưu về

```text
results/
├── fudge/
│   ├── fudge_tokenwise.json
│   └── fudge_blockwise.json
├── cdq/
│   ├── cdq_tokenwise.json
│   └── cdq_blockwise.json
```

---

## 3. RAD (Reward-Augmented Decoding)

Thư mục triển khai:

- [Method/RAD](Method/RAD)
- [Method/RAD/generate.py](Method/RAD/generate.py)
- [Method/RAD/eval_sentiment.py](Method/RAD/eval_sentiment.py)
- [Method/RAD/eval_toxicity.py](Method/RAD/eval_toxicity.py)

### Train reward model cho RAD

```bash
cd Method/RAD
python reward_modeling/trainer_rm.py \
  --configs rm_sentiment gpt2-small \
  --wandb_entity WANDB_ID
```

### Chạy sentiment experiment

```bash
cd Method/RAD
DATASET=positive
BATCH_SIZE=4
LANGUAGE_MODEL=gpt2-large
TOPK=20
BETA=50
INVERSE=True

python eval_sentiment.py \
    --dataset $DATASET \
    --batch_size $BATCH_SIZE \
    --lm $LANGUAGE_MODEL \
    --topk $TOPK \
    --beta $BETA \
    --inverse $INVERSE
```

### Chạy toxicity experiment

```bash
cd Method/RAD
BATCH_SIZE=4
LANGUAGE_MODEL=gpt2-large
TOPK=20
BETA=50
INVERSE=True

python eval_toxicity.py \
    --batch_size $BATCH_SIZE \
    --lm $LANGUAGE_MODEL \
    --topk $TOPK \
    --beta $BETA \
    --inverse $INVERSE
```

### Custom generation

```bash
cd Method/RAD
python generate.py
```

---

## 4. GenARM (Autoregressive Reward Model)

Thư mục triển khai:

- [Method/GenARM](Method/GenARM)
- [Method/GenARM/generate_genarm.py](Method/GenARM/generate_genarm.py)
- [Method/GenARM/generate_arm_gpt2_medium.py](Method/GenARM/generate_arm_gpt2_medium.py)

### Chạy GenARM generation

```bash
python Method/GenARM/generate_genarm.py
```

### Chạy GenARM GPT-2 medium

```bash
python Method/GenARM/generate_arm_gpt2_medium.py
```

### Train GenARM

```bash
cd Method/GenARM
bash training_trl/train_arm_llama_HH.sh
```

Hoặc dùng script tương ứng cho Alpaca / SafeRLHF:

```bash
cd Method/GenARM
bash training_trl/train_arm_alpaca_SafeRLHF.sh
```

---

## 5. ARGS

Thư mục triển khai:

- [Method/args](Method/args)
- [Method/args/generate_args.py](Method/args/generate_args.py)

### Chạy generation cho ARGS

```bash
python Method/args/generate_args.py
```

### Thu thập output model

```bash
python Method/args/collect_model_outs.py
```

### Tính reward / metrics

```bash
python Method/args/measure_reward.py
python Method/args/metrics.py
```

---

## 6. MultiSignal

Thư mục triển khai:

- [Method/MultiSignal](Method/MultiSignal)
- [Method/MultiSignal/core/cache.py](Method/MultiSignal/core/cache.py)
- [Method/MultiSignal/core/train.py](Method/MultiSignal/core/train.py)
- [Method/MultiSignal/core/fusion.py](Method/MultiSignal/core/fusion.py)

### Build cache

```bash
python Method/MultiSignal/core/cache.py --split train
python Method/MultiSignal/core/cache.py --split validation
```

### Train controller

```bash
python Method/MultiSignal/core/train.py --cache dataset/multisignal_cache/train.pt
```

### Ý tưởng

MultiSignal học cách kết hợp 4 signal:

- CD
- GenARM
- RAD
- ARGS

thông qua một controller MLP và dùng trọng số đã học để cộng vào logits gốc của model.

---

## 7. Dataset benchmark gốc

Một số dataset benchmark có sẵn:

```text
dataset/
├── cd_train/
├── multisignal_train/
├── rad_benchmark/
└── ...
```

Ví dụ chạy 100 prompt đầu trên benchmark RAD:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path

path = Path('Method/generate_base.py')
spec = importlib.util.spec_from_file_location('genbase', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.NUM_PROMPTS = 100
mod.INPUT_FILE = Path('dataset/rad_benchmark/negative_prompts.jsonl')
mod.OUTPUT_FILE = Path('results/base_test_100.json')
mod.main()
PY
```

---

## 8. Thứ tự chạy khuyến nghị

```text
1. CD baseline
2. RAD baseline
3. GenARM baseline
4. ARGS baseline
5. MultiSignal cache + train
6. So sánh output trên cùng tập benchmark
```

---

## 9. Lưu ý

- RAD / GenARM / MultiSignal có thể yêu cầu GPU mạnh hơn.
- MultiSignal hiện đang phụ thuộc vào controller và cache preprocessing.
- Nếu chạy trên CPU, nên giảm batch size hoặc số prompt test.
- Với GPU, nên dùng `torch.cuda.is_available()` để kiểm tra runtime.

---

## 10. Tóm tắt ngắn

- CD: focus vào controlled decoding bằng prefix scorer / contrastive signal.
- RAD: reward-augmented decoding bằng reward model.
- GenARM: autoregressive reward model để steer generation.
- ARGS: utility / generation / evaluation với arg-based rewards.
- MultiSignal: kết hợp các signal theo trọng số học được.

Đây là các phương pháp chính mà repo đang triển khai.
