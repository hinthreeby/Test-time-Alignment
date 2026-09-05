# MultiSignal

MultiSignal học một controller nhỏ để kết hợp nhiều reward-decoding signal trong lúc sinh token.

Signal hiện dùng:

- `genarm`: autoregressive reward signal
- `rad`: prefix/outcome reward signal
- `cdq`: prefix value signal
- `args`: outcome reward-search signal

Ở mỗi bước decoding, base LM tạo top-k candidate token. Các adapter chấm điểm từng candidate, controller dự đoán:

- `weights`: nên tin signal nào nhiều hơn
- `strength`: reward guidance nên mạnh bao nhiêu
- `trust/base_weight`: khi nào nên giảm reward và quay gần về base LM

## Cấu Trúc

```text
Method/MultiSignal/
├── configs/default.json
├── core/
│   ├── cache.py         # build dataset/multisignal_cache/*.pt
│   ├── fusion.py        # lm_logits + strength * fused_reward
│   ├── generate.py      # fused decoding
│   ├── oracle_study.py  # fixed-best vs oracle routing gap
│   └── train.py         # train MultiSignalController
├── models/
│   └── controller.py    # MLP weights/strength/trust
└── router/
    ├── adapters/
    │   ├── fallback_adapter.py
    │   └── registry.py
    └── configs/
        ├── args.json
        ├── cdq.json
        ├── genarm.json
        └── rad.json
```

## Adapter Hiện Tại

Adapter mặc định dùng model/checkpoint thật nếu local file tồn tại:

| Signal | Nguồn thật |
| --- | --- |
| `genarm` | `models/genarm-gpt2-medium-hh` |
| `rad` | `models/rad_rm_sentiment` + `models/gpt2-small` |
| `cdq` | `Method/CD/checkpoints/cd_q.pt` + `models/gpt2-small` |
| `args` | `models/sentiment-roberta-large-english` |

Nếu thiếu tài nguyên, adapter fallback về pseudo-signal từ base LM logits và ghi rõ:

```json
"signal_sources": ["fallback_lm_logits", "..."]
```

Kết quả dùng cho paper nên kiểm tra `signal_sources` và chỉ dùng run có signal thật.

## Dataset

Train/cache cần:

```text
dataset/multisignal_train/train.jsonl
dataset/multisignal_train/validation.jsonl
```

Mỗi dòng:

```json
{"prompt": "...", "response": "..."}
```

Benchmark generation thường dùng:

```text
dataset/rad_benchmark/negative_prompts.jsonl
dataset/rad_benchmark/positive_prompts.jsonl
dataset/rad_benchmark/neutral_prompts.jsonl
dataset/rad_benchmark/all.jsonl
```

## Pipeline Chính

Chạy từ root repo:

```bash
cd "/home/jupyter-iec2024se10/Reward Decoding"
```

### 1. Build cache

```bash
python Method/run_method.py \
  --method multisignal \
  --action cache \
  --split train \
  --max-response-tokens 32 \
  --top-k 20
```

Smoke cache nhỏ:

```bash
python Method/run_method.py \
  --method multisignal \
  --action cache \
  --split train \
  --max-samples 1 \
  --max-response-tokens 8 \
  --top-k 10
```

Cache được lưu ở:

```text
dataset/multisignal_cache/train.pt
dataset/multisignal_cache/validation.pt
```

### 2. Train controller

```bash
python Method/run_method.py \
  --method multisignal \
  --action train \
  --cache dataset/multisignal_cache/train.pt
```

Checkpoint được lưu ở:

```text
Method/MultiSignal/checkpoints/controller_4signal.pt
```

### 3. Generate

Generate một prompt:

```bash
python Method/run_method.py \
  --method multisignal \
  --action generate \
  --prompt "This movie was" \
  --max-new-tokens 32 \
  --top-k 20
```

Generate benchmark:

```bash
python Method/run_method.py \
  --method multisignal \
  --action generate \
  --input dataset/rad_benchmark/negative_prompts.jsonl \
  --output results/multi_signal.jsonl \
  --num-prompts 1000 \
  --max-new-tokens 32 \
  --top-k 20
```

Output JSONL có các trường quan trọng:

- `response`
- `signals`
- `signal_sources`
- `mean_weights`
- `mean_strength`
- `mean_trust`
- `mean_base_weight`
- `latency`

### 4. Smoke test nhẹ

Lệnh này không dùng signal thật, chỉ kiểm tra controller/generation nhanh:

```bash
python Method/MultiSignal/core/generate.py \
  --prompt "hello world" \
  --max-new-tokens 8 \
  --top-k 10 \
  --fallback-signals \
  --output /tmp/multisignal_smoke.jsonl
```

## Đánh Giá

Evaluator chung của repo đọc `results/multi_signal.jsonl`:

```bash
python eval/evaluate.py --methods multi-signal --device auto
```

Hoặc chỉ định file:

```bash
python eval/evaluate.py --files multi_signal.jsonl --device auto
```

## Oracle Study

Oracle study dùng để kiểm tra câu hỏi nghiên cứu quan trọng nhất:

> Nếu luôn chọn đúng method tốt nhất cho từng prompt, kết quả có hơn fixed-best method không?

Sau khi có nhiều output đã được evaluator độc lập gắn score cấp sample, chạy:

```bash
python Method/MultiSignal/core/oracle_study.py \
  --results results/base_scored.jsonl results/rad_scored.jsonl results/genarm_scored.jsonl results/multi_signal_scored.jsonl \
  --score-fields score reward rm_reward sentiment \
  --output Method/MultiSignal/reports/oracle_study.csv
```

File CSV sẽ ghi:

- số prompt chung giữa các method
- fixed-best method
- fixed-best score
- oracle score
- oracle gap
- số lần mỗi method thắng oracle

Oracle gap lớn là bằng chứng rằng routing giữa nhiều signal có ý nghĩa.

## Config

Config chính:

```text
Method/MultiSignal/configs/default.json
```

Tham số thường chỉnh:

- `controller.hidden_dim`: kích thước MLP
- `controller.signal_costs`: chi phí tương đối của từng signal theo thứ tự `genarm, rad, cdq, args`
- `training.batch_size`: giảm nếu OOM
- `training.epochs`: số epoch train controller
- `training.lr`: learning rate
- `training.num_workers`: để `0` nếu môi trường notebook/sandbox chặn multiprocessing
- `training.mixed_precision`: dùng AMP khi CUDA hợp lệ

`top_k` và `max_response_tokens` được truyền qua CLI khi build cache/generate.

## CUDA

Code tự chọn:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Mặc định runner dùng cấu hình low-VRAM:

- base LM: `--base-device auto`
- signal models: `--signal-device cpu`

Như vậy GPU 10GB chỉ cần chứa base LM chính, còn reward/scorer models chạy trên CPU. Nếu có GPU lớn hơn, có thể ép signal chạy CUDA:

```bash
python Method/run_method.py \
  --method multisignal \
  --action cache \
  --split train \
  --signal-device cuda
```

Nếu vẫn OOM, giảm:

- `--top-k 10`
- `--max-response-tokens 16`
- `--max-samples 200`

Nếu driver CUDA không tương thích, PyTorch sẽ rơi về CPU.

## Kiểm Tra Checkpoint

```bash
python - <<'PY'
import torch
ckpt = torch.load('Method/MultiSignal/checkpoints/controller_4signal.pt', map_location='cpu', weights_only=False)
print(ckpt.keys())
print(ckpt['signals'])
print(ckpt.get('signal_costs'))
PY
```

## Ghi Chú Research

Hiện code đã có skeleton phù hợp proposal: multi-signal adapters, token-level controller, trust gate, fused decoding, telemetry và oracle-study utility.

Để dùng cho paper Q1, cần chạy lại full cache/generation bằng signal thật, kiểm tra `signal_sources`, rồi làm ablation:

- base LM
- từng expert riêng
- uniform fusion
- prompt/router baseline
- MultiSignal token-level controller
- TARo nếu có
- oracle routing
