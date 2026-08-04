# Controlled Decoding (CD) — Reimplementation

Đây là bản cài đặt lại **Controlled Decoding (CD)** theo paper:

> **Controlled Decoding from Language Models**  
> ICML 2024

Bản hiện tại tập trung vào **CD-FUDGE** trước, gồm:

- Tạo rollout bằng Base LM
- Chấm reward bằng sentiment classifier
- Train Prefix Scorer
- Tokenwise decoding
- Blockwise best-of-K decoding
- Lưu kết quả theo format dùng chung với ARGS, RAD và GenARM

---

## 1. Cấu trúc thư mục

```text
Reward Decoding/
├── dataset/
│   ├── cd_train/
│   │   ├── prompts.jsonl
│   │   └── fudge_rollouts.jsonl
│   └── rad_benchmark/
│       └── negative_prompts.jsonl
│
├── models/
│   ├── gpt2-large/
│   ├── gpt2-small/
│   ├── sentiment-rm-sst2/
│   └── cd_fudge_prefix_scorer/
│
├── results/
│   └── cd.json
│
└── Method/
    └── CD/
        ├── requirements.txt
        ├── generate_cd.py
        ├── models/
        │   ├── __init__.py
        │   └── prefix_scorer.py
        ├── training/
        │   ├── __init__.py
        │   ├── build_fudge_rollouts.py
        │   └── train_cd_fudge.py
        └── decoding/
            ├── __init__.py
            └── cd_decoder.py
```

---

## 2. Tạo môi trường Conda

```bash
conda create -n cd python=3.10 -y
conda activate cd
```

Cài dependency:

```bash
cd "/home/jupyter-iec2024se10/Reward Decoding/Method/CD"

pip install -r requirements.txt
```

Kiểm tra GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Kỳ vọng:

```text
2.2.2 True
```

---

## 3. Chuẩn bị model

Các model được dùng:

```text
Base LM:
models/gpt2-large

Reward Model:
models/sentiment-rm-sst2

Prefix Scorer backbone:
models/gpt2-small
```

Reward Model là sentiment classifier đã fine-tune trên SST-2.

Prefix Scorer sẽ được train ở bước sau.

---

## 4. Chuẩn bị prompt train

Tạo file:

```text
dataset/cd_train/prompts.jsonl
```

Mỗi dòng có dạng:

```json
{"prompt": {"text": "It made my hair feel flat and uncooperative"}}
```

Không nên dùng chính 100 prompt benchmark để train Prefix Scorer vì sẽ gây rò rỉ dữ liệu.

Nên dùng một tập prompt riêng cho training.

---

## 5. Tạo rollout cho CD-FUDGE

File sử dụng:

```text
Method/CD/training/build_fudge_rollouts.py
```

Các tham số đã được đặt trực tiếp trong code:

```python
NUM_SAMPLES = 2000
MAX_NEW_TOKENS = 64
TEMPERATURE = 1.0
TOP_P = 0.95
SEED = 42
```

Chạy từ thư mục gốc:

```bash
cd "/home/jupyter-iec2024se10/Reward Decoding"

conda activate cd

python Method/CD/training/build_fudge_rollouts.py
```

Output:

```text
dataset/cd_train/fudge_rollouts.jsonl
```

Mỗi dòng có dạng:

```json
{
  "prompt": "It made my hair feel flat and uncooperative",
  "response": "but after a while it became easier to manage.",
  "reward": 0.8732
}
```

Kiểm tra nhanh:

```bash
head -3 dataset/cd_train/fudge_rollouts.jsonl
```

---

## 6. Train Prefix Scorer

Prefix Scorer học:

```text
prompt + partial response
→ expected final reward
```

Chạy:

```bash
cd "/home/jupyter-iec2024se10/Reward Decoding"

conda activate cd

python Method/CD/training/train_cd_fudge.py \
  --train-file dataset/cd_train/fudge_rollouts.jsonl \
  --model models/gpt2-small \
  --output-dir models/cd_fudge_prefix_scorer \
  --max-length 256 \
  --batch-size 8 \
  --epochs 1
```

Nếu bị CUDA OOM, giảm:

```bash
--batch-size 4
```

hoặc:

```bash
--batch-size 2
```

Model tốt nhất sẽ được lưu tại:

```text
models/cd_fudge_prefix_scorer/
```

Kiểm tra:

```bash
find models/cd_fudge_prefix_scorer -maxdepth 2 -type f
```

---

## 7. Tokenwise Controlled Decoding

Tokenwise CD thực hiện:

```text
aligned score
=
Base LM logits
+
lambda × Prefix Scorer value
```

Chạy trên 100 prompt RAD:

```bash
cd "/home/jupyter-iec2024se10/Reward Decoding"

conda activate cd

python Method/CD/generate_cd.py \
  --input dataset/rad_benchmark/negative_prompts.jsonl \
  --output results/cd.json \
  --base-model models/gpt2-large \
  --prefix-scorer models/cd_fudge_prefix_scorer \
  --num-prompts 100 \
  --mode tokenwise \
  --lambda-weight 1.0 \
  --top-k 20 \
  --max-new-tokens 64 \
  --temperature 1.0 \
  --sampling greedy
```

Output:

```text
results/cd.json
```

---

## 8. Blockwise Controlled Decoding

Blockwise CD hoạt động như sau:

```text
1. Sinh K candidate block
2. Mỗi block dài M token
3. Prefix Scorer chấm từng candidate
4. Chọn block có value cao nhất
5. Lặp đến khi đủ token hoặc gặp EOS
```

Chạy:

```bash
python Method/CD/generate_cd.py \
  --input dataset/rad_benchmark/negative_prompts.jsonl \
  --output results/cd_blockwise.json \
  --base-model models/gpt2-large \
  --prefix-scorer models/cd_fudge_prefix_scorer \
  --num-prompts 100 \
  --mode blockwise \
  --block-size 8 \
  --num-candidates 4 \
  --max-new-tokens 64 \
  --temperature 1.0 \
  --top-p 0.95
```

---

## 9. Kiểm tra output

```bash
python - <<'PY'
import json

path = "results/cd.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Samples:", len(data))
print(json.dumps(data[0], indent=2, ensure_ascii=False))
PY
```

Mỗi kết quả có dạng:

```json
{
  "id": 0,
  "md5_hash": "...",
  "prompt": "...",
  "reference": "...",
  "response": "...",
  "num_positive": 0,
  "method": "CD",
  "decoding_mode": "tokenwise",
  "lambda_weight": 1.0,
  "top_k": 20,
  "max_new_tokens": 64,
  "latency": 1.2345,
  "status": "success",
  "error": null
}
```

---

## 10. Chạy evaluator chung

Sau khi có:

```text
results/args.json
results/rad.json
results/cd.json
results/genarm.json
```

chạy:

```bash
cd "/home/jupyter-iec2024se10/Reward Decoding"

python eval/evaluate_all.py \
  --results-dir results
```

Kết quả:

```text
results/metrics.csv
results/metrics.json
```

Các metric:

- Positive Rate
- Perplexity
- Average Sentiment
- Average Length
- Dist-1
- Dist-2
- Dist-3
- Repetition
- Latency

---

## 11. Các tham số quan trọng

### `lambda_weight`

Điều chỉnh mức độ ảnh hưởng của Prefix Scorer:

```text
lambda = 0
→ gần Base LM

lambda lớn
→ reward mạnh hơn nhưng có thể giảm fluency
```

Nên thử:

```text
0.0
0.5
1.0
2.0
4.0
```

### `top_k`

Số candidate token được Prefix Scorer chấm ở mỗi bước:

```text
top_k nhỏ
→ nhanh hơn

top_k lớn
→ tìm kiếm tốt hơn nhưng chậm hơn
```

Nên thử:

```text
5
10
20
50
```

### Blockwise

```text
block_size = M
num_candidates = K
```

Khởi đầu:

```text
M = 8
K = 4
```

---

## 12. Pipeline tổng thể

```text
Training prompts
        ↓
GPT-2 Large rollout
        ↓
Sentiment RM chấm final reward
        ↓
fudge_rollouts.jsonl
        ↓
Train GPT-2 Prefix Scorer
        ↓
Tokenwise hoặc Blockwise CD
        ↓
results/cd.json
        ↓
Shared evaluator
```

---

## 13. Phạm vi hiện tại

Bản này triển khai:

```text
CD-FUDGE
```

Chưa triển khai đầy đủ:

```text
CD-Q
```

CD-Q cần:

- frozen Base LM trong quá trình train;
- Bellman target;
- expectation trên next-token distribution;
- stop-gradient target;
- target network hoặc cơ chế ổn định tương tự DQN.

Nên xác nhận CD-FUDGE chạy đúng trước, rồi mới xây CD-Q.
