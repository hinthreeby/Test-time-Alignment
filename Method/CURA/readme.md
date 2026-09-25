# CURA

> **Calibrated Uncertainty-aware Reward Arbitration for Test-Time LLM Alignment**

`CURA` là tên làm việc thay cho `MultiSignal`. CURA giữ ý tưởng cốt lõi dùng một controller nhỏ để kết hợp `GenARM`, `RAD`, `CD-Q` và `ARGS`, nhưng nâng cấp bài toán từ **adaptive fusion** thành **confidence-aware reward arbitration**.

Tên và acronym cần được kiểm tra trùng lặp lần cuối trước khi gửi paper.

## Trạng thái triển khai

Code hiện có hai protocol tách biệt:

- `configs/sentiment.json`: CURA-MVP trainable với robust calibration và teacher-forced token loss;
- `configs/sentiment_paper.json`: protocol nghiên cứu dùng held-out rollout targets, learned
  heteroscedastic calibration, token-level candidate-ranking loss, utility-derived gate target
  và learned budget selector.

Các thành phần đã triển khai gồm kiểm tra checkpoint tồn tại và objective khai báo;
cache v2 theo shard với
atomic save, checksum, fingerprint và resume; target annotation bằng nhiều rollout;
candidate-specific uncertainty; disagreement-aware policy fusion; KL projection;
selector có thể thật sự bỏ qua expert để giảm số signal call; validation/best-checkpoint;
oracle token study; corrupted/leave-one-out ablations; paired bootstrap; reproducible
generation provenance và paper-run validator.

Checkpoint GenARM local hiện đo HH/harmlessness trong khi ba signal còn lại đo
sentiment. Cấu hình train mặc định `configs/sentiment.json` vì vậy dùng ba signal
tương thích `rad`, `cdq`, `args` và không silently fallback. Cấu hình chẩn đoán
`configs/mixed_objective_mvp.json` mới dùng đủ bốn signal và bắt buộc cờ
`--allow-objective-mismatch`; output đó không đủ điều kiện đưa vào bảng paper.
Full four-signal paper run vẫn cần một GenARM checkpoint cùng objective. Code hoàn chỉnh
không đồng nghĩa đã có bằng chứng A*: các benchmark nhiều seed, baseline, ablation và
transfer experiment bên dưới vẫn phải được chạy và báo cáo.

### Protocol A* khuyến nghị

`configs/sentiment_astar.json` là cấu hình chính để cải thiện kết quả sentiment. Cấu
hình này giữ `RAD` và `CD-Q` làm signal, đồng thời dành
`models/sentiment-roberta-large-english` làm rollout evaluator độc lập. `ARGS` bị loại
khỏi signal set vì nó dùng chính classifier đó; dùng cả hai sẽ gây target leakage.

Phải dùng cache mới: paper cache giữ nguyên natural top-k thay vì chèn gold token ngoài
support; rollout evaluator chỉ chấm response-so-far thay vì để negative prompt làm lệch
target; và checkpoint lưu provenance của train/validation rollout targets.

```bash
python Method/CURA/core/prepare_astar_data.py \
  --output-dir dataset/cura_astar_source \
  --train-per-class 2000 --validation-per-class 400

python Method/run_method.py --method cura --action cache --split train \
  --config Method/CURA/configs/sentiment_astar.json \
  --input dataset/cura_astar_source/train.jsonl \
  --cache-dir dataset/cura_cache_astar/train --shard-size 100 \
  --max-response-tokens 32 --top-k 20 --resume
python Method/run_method.py --method cura --action cache --split validation \
  --config Method/CURA/configs/sentiment_astar.json \
  --input dataset/cura_astar_source/validation.jsonl \
  --cache-dir dataset/cura_cache_astar/validation --shard-size 50 \
  --max-response-tokens 32 --top-k 20 --resume

python Method/run_method.py --method cura --action annotate-targets --split train \
  --cache-dir dataset/cura_cache_astar/train \
  --evaluator models/sentiment-roberta-large-english \
  --rollouts 2 --rollout-tokens 16 --candidate-batch-size 10
python Method/run_method.py --method cura --action annotate-targets --split validation \
  --cache-dir dataset/cura_cache_astar/validation \
  --evaluator models/sentiment-roberta-large-english \
  --rollouts 2 --rollout-tokens 16 --candidate-batch-size 10

python Method/run_method.py --method cura --action train \
  --config Method/CURA/configs/sentiment_astar.json \
  --cache-dir dataset/cura_cache_astar/train \
  --validation-cache-dir dataset/cura_cache_astar/validation \
  --output Method/CURA/checkpoints/cura_astar.pt

python Method/run_method.py --method cura --action diagnose \
  --checkpoint Method/CURA/checkpoints/cura_astar.pt \
  --cache-dir dataset/cura_cache_astar/validation \
  --output Method/CURA/reports/astar_diagnostic.json
```

Trước khi chạy full benchmark, dùng pipeline inference-tuning mặc định trên 250 prompt
validation không trùng train/final test. Pipeline so sánh learned CURA, `fixed-lambda`
2/3/4, `fixed-gate=0.75` và `no-gate`, sau đó ghi báo cáo riêng dưới
`results/experiments/cura_inference_tune_v1/`. Chỉ mở rộng run khi diagnostic cho thấy
`cura_gain_over_base > 0` và khoảng cách tới token oracle còn đủ lớn.
`configs/sentiment_stable_mvp.json` có thể dùng cho vòng smoke test không có rollout
targets, nhưng không đủ điều kiện làm kết quả A* chính.

Để train, build cả train và validation cache rồi gọi trainer:

```bash
python Method/run_method.py --method cura --action cache --split train \
  --cache-dir dataset/cura_cache/train --shard-size 100 --resume \
  --max-response-tokens 32 --top-k 20
python Method/run_method.py --method cura --action cache --split validation \
  --cache-dir dataset/cura_cache/validation --shard-size 50 --resume \
  --max-response-tokens 32 --top-k 20
python Method/run_method.py --method cura --action train \
  --cache-dir dataset/cura_cache/train \
  --validation-cache-dir dataset/cura_cache/validation
```

Đây là lệnh MVP. Với paper protocol, sau khi cache xong phải annotate **cả hai split**
bằng evaluator độc lập (không trùng bất kỳ signal model nào), sau đó train với
`configs/sentiment_paper.json`.

### Ma trận mức hoàn thiện

| Hạng mục | Code | Bằng chứng thực nghiệm |
| --- | --- | --- |
| Sharded/resumable cache | Có | Unit/integration test |
| Learned calibration và uncertainty | Có | Chưa chạy full held-out benchmark |
| Token ranking + utility + KL + gate loss | Có | Chưa tune nhiều seed |
| Conditional signal budget | Có | Chưa lập quality/latency Pareto |
| Objective declaration audit + cache-time tokenizer check | Có | Cần kiểm tra trên toàn bộ paper run |
| Fixed/token oracle + corruption/leave-one-out | Có CLI | Prompt oracle chưa có; chưa có bảng kết quả |
| Unseen-base/domain transfer | Protocol sẵn | Chưa chạy |
| Kết luận A* | Không thể kết luận từ code | Cần kết quả và peer review |

## 1. Mục tiêu nghiên cứu

Các reward-decoding method không tạo ra cùng một loại bằng chứng:

| Signal | Thông tin chính |
| --- | --- |
| `genarm` | Reward tự hồi quy ở cấp token |
| `rad` | Reward của prefix sau khi nối candidate |
| `cdq` | Giá trị kỳ vọng dài hạn của prefix |
| `args` | Outcome reward được dùng trực tiếp để tìm token |

Ở cùng một bước decoding, các signal có thể:

- dùng thang điểm khác nhau;
- đồng thuận hoặc mâu thuẫn;
- đáng tin ở những loại prefix khác nhau;
- trở nên thiếu tin cậy khi gặp dữ liệu ngoài miền;
- làm giảm fluency nếu guidance quá mạnh.

Câu hỏi nghiên cứu của CURA là:

> Có thể hiệu chỉnh và ước lượng độ tin cậy của các reward signal không đồng nhất, sau đó phân xử chúng theo từng token để cải thiện alignment mà vẫn giới hạn độ lệch khỏi base LM hay không?

## 2. Khác biệt so với MultiSignal cũ

| MultiSignal cũ | CURA |
| --- | --- |
| Dùng reward score thô hoặc chuẩn hóa chung | Mỗi signal có calibrator riêng |
| Controller học trộn score | Controller học độ tin cậy rồi mới trộn |
| Không mô hình hóa bất đồng rõ ràng | Có disagreement features và uncertainty |
| `strength` và `trust` có thể trùng vai trò | `strength` điều chỉnh reward; `gate` trộn hai policy |
| Oracle chủ yếu ở cấp prompt | Có fixed-best/global và token-candidate oracle; prompt oracle còn thiếu |
| Có thể dùng checkpoint khác mục tiêu | Paper run bắt buộc kiểm tra objective compatibility |

## 3. Giả thuyết nghiên cứu

- **H1 — Complementarity:** Không có một reward-decoding signal duy nhất tốt nhất trên mọi prompt và mọi bước sinh.
- **H2 — Calibration:** Hiệu chỉnh riêng từng signal tốt hơn cộng trực tiếp hoặc z-score đơn thuần.
- **H3 — Reliability:** Uncertainty và disagreement giúp controller giảm trọng số của signal không đáng tin.
- **H4 — Token adaptivity:** Routing theo token tốt hơn trọng số cố định và routing theo prompt.
- **H5 — Safe fallback:** Khi reward evidence yếu hoặc mâu thuẫn, quay gần về base LM giúp giữ fluency và giảm reward failure.
- **H6 — Efficiency:** Cost-aware routing có thể giữ phần lớn chất lượng với chi phí thấp hơn chạy mọi signal ở mọi bước.

## 4. Điều kiện bắt buộc về objective

Không được mặc định rằng bốn checkpoint hiện tại đo cùng một mục tiêu.

Ví dụ hiện tại:

- `models/genarm-gpt2-small-hh`: helpfulness/harmlessness;
- `models/rad_rm_sentiment`: sentiment;
- `models/sentiment-roberta-large-english`: sentiment;
- `Method/CD/checkpoints/cd_q.pt`: phụ thuộc reward dùng khi train.

Trước khi fusion, mỗi experiment phải chọn một trong hai chế độ.

### 4.1. Chế độ A — một objective, nhiều estimator

Tất cả `GenARM`, `RAD`, `CD-Q`, `ARGS` phải cùng đo một objective, ví dụ sentiment hoặc harmlessness. Đây là chế độ nên triển khai đầu tiên vì dễ kiểm chứng câu hỏi arbitration.

#### Signal set mục tiêu cho CURA sentiment

Signal set được chọn cho experiment CURA sentiment tiếp theo là:

```text
RAD + CD-Q + sentiment-GenARM
```

Ba signal phải có cùng objective `sentiment` nhưng cung cấp bằng chứng khác nhau:

- `rad`: reward của từng prefix sau khi nối candidate;
- `cdq`: long-horizon prefix value;
- `genarm`: autoregressive sentiment reward ở cấp token.

Config logic tương ứng phải khai báo đúng thứ tự (thứ tự này cũng quyết định thứ tự
calibrator, controller head và `signal_costs`):

```json
{
  "signals": ["rad", "cdq", "genarm"],
  "objective": "sentiment",
  "signal_objectives": {
    "rad": "sentiment",
    "cdq": "sentiment",
    "genarm": "sentiment"
  },
  "score_directions": {
    "rad": "higher_is_better",
    "cdq": "higher_is_better",
    "genarm": "higher_is_better"
  },
  "controller": {
    "signal_costs": [1.0, 0.8, 1.2]
  }
}
```

**Trạng thái artifact hiện tại:** signal set này chưa thể dùng cho paper run. Repo hiện
chỉ có `models/genarm-gpt2-small-hh`, được train cho helpfulness/harmlessness; đây
không phải sentiment-GenARM và không được đổi nhãn thành `sentiment`. Cần train hoặc
cung cấp một autoregressive reward checkpoint sentiment riêng, ví dụ
`models/genarm-gpt2-small-sentiment`, rồi trỏ GenARM adapter tới đúng artifact đó.
`audit-signals` phải kiểm tra chính đường dẫn checkpoint thực tế, không chỉ kiểm tra
chuỗi `signal_objectives` trong JSON.

Checkpoint `Method/CURA/checkpoints/cura_astar_full_v1.pt` chỉ chứa hai signal
`["rad", "cdq"]`. Không được nạp checkpoint này với config ba signal. Khi có
sentiment-GenARM, bắt buộc phải:

1. tạo config ba signal mới, không sửa ngược config/checkpoint của run cũ;
2. build train và validation cache mới với đúng thứ tự `rad, cdq, genarm`;
3. annotate rollout targets bằng evaluator sentiment độc lập với cả ba signal;
4. fit lại calibrator và controller rồi lưu checkpoint mới;
5. chạy `signal-budget=3` ở baseline đầu tiên để đo complementarity thật; chỉ thử
   budget 2 sau đó như một ablation quality/latency;
6. đánh giá final bằng bộ evaluator local đã khóa revision, tách khỏi evaluator tạo
   training targets.

Không tái sử dụng cache, calibration, checkpoint hoặc output của cấu hình hai signal
cho cấu hình ba signal. Mỗi output phải ghi đủ `signal_sources`,
`signal_objectives`, config fingerprint và checkpoint hash để phát hiện nhầm artifact.

Preference dataset cho sentiment-GenARM được tạo hoàn toàn local bằng GPT-2 Large và
teacher `sentiment-roberta-large-english`. Script tự loại prompt trùng final benchmark,
chấm response-only, lọc theo score margin/độ dài và hỗ trợ resume:

```bash
python Method/GenARM/training_trl/build_sentiment_preferences.py \
  --split train --device cuda

python Method/GenARM/training_trl/build_sentiment_preferences.py \
  --split validation --device cuda
```

Output DPO/ARM-compatible nằm tại:

```text
dataset/GenARM/sentiment_preferences/train/preferences.jsonl
dataset/GenARM/sentiment_preferences/validation/preferences.jsonl
```

Không dùng `distilbert-sst2` để pseudo-label; checkpoint đó được giữ riêng cho final
evaluation. Không đổi generation/threshold trong cùng output directory vì script sẽ
chặn fingerprint không tương thích; dùng một thư mục output mới cho mỗi protocol.

### 4.2. Chế độ B — nhiều objective, nhiều estimator

Score có hai chỉ số:

```text
score[objective][signal][candidate]
```

Controller phân xử estimator trong từng objective trước, sau đó kết hợp các objective bằng preference vector `alpha`. Chỉ triển khai sau khi chế độ A hoạt động đúng.

Mọi output phải ghi:

```json
{
  "objective": "sentiment",
  "signal_objectives": {
    "genarm": "sentiment",
    "rad": "sentiment",
    "cdq": "sentiment",
    "args": "sentiment"
  }
}
```

Nếu các objective không tương thích, run không được dùng làm kết quả chính của paper.

## 5. Ký hiệu và công thức

Tại bước `t`, base LM sinh tập candidate top-k:

```text
C_t = {v_1, ..., v_K}
```

Base logits:

```text
l_base[t, k]
```

Reward thô của signal `j`:

```text
s_raw[t, j, k]
```

### 5.1. Signal calibration

Mỗi signal có calibrator riêng. Với calibrator học được, input hiện gồm raw score và
các thống kê robust trong candidate set; code chưa đưa toàn bộ prefix/context embedding
vào calibrator:

```text
mu[t, j, k], log_var[t, j, k] = Calibrator_j(s_raw, candidate_set_statistics)
```

Trong đó:

- `mu`: reward đã đưa về utility chung;
- `var = exp(log_var)`: uncertainty của signal đối với candidate.

MVP có thể dùng robust normalization:

```text
z = (score - median(score)) / (MAD(score) + eps)
```

Paper mode dùng rollout target độc lập trên **train** để jointly train calibrator cùng
controller. Validation rollout target chỉ dùng để tính validation loss và chọn best
checkpoint. Không được fit calibrator hoặc controller trên test set. Lệnh `calibrate`
hiện chỉ kiểm tra cache có đủ target cho calibration mode, không tạo checkpoint riêng.

### 5.2. Disagreement

Candidate-level disagreement:

```text
D[t, k] = weighted_variance_j(mu[t, j, k])
```

Step-level disagreement:

```text
D_step[t] = mean_k(D[t, k])
```

Code hiện dùng weighted variance ở bước fusion và một scalar unweighted disagreement
trong controller features. Pairwise Spearman correlation chưa được tính hoặc lưu.

### 5.3. Controller

Controller nhận:

- base entropy, margin và top-k probability mass;
- mean/std của `mu` và mean `var` theo candidate cho từng signal;
- một step-level disagreement scalar;
- vị trí token và độ dài prefix;
- signal availability mask.

Signal cost chưa phải input feature. Nó được dùng làm fixed bias trong weight/selector
logits và trong cost regularization. `preference_alpha` và multi-objective routing chưa
được triển khai; code hiện tại là single-objective.

Controller xuất:

```text
w_t      = softmax(weight_head(h_t))
lambda_t = lambda_max * sigmoid(strength_head(h_t))
g_t      = sigmoid(gate_head(h_t))
```

Ý nghĩa:

- `w_t[j]`: mức tin signal `j`, tổng bằng 1;
- `lambda_t`: cường độ reward bên trong guided policy;
- `g_t`: mức dùng guided policy thay cho base policy.

### 5.4. Risk-adjusted fusion

```text
risk_reward[t, k]
  = sum_j w_t[j] * (mu[t, j, k] - kappa * sqrt(var[t, j, k]))
    - delta * D[t, k]
```

Guided policy:

```text
p_guided = softmax(l_base + lambda_t * risk_reward)
```

Final policy:

```text
p_final = (1 - g_t) * p_base + g_t * p_guided
```

Cách viết này giúp `lambda_t` và `g_t` có vai trò khác nhau:

- `lambda_t` quyết định reward làm sắc guided policy đến đâu;
- `g_t` quyết định có nên dùng guided policy hay fallback về base.

Thêm ràng buộc:

```text
KL(p_final || p_base) <= epsilon_kl
```

Nếu vượt ngân sách, giảm `lambda_t` bằng binary search hoặc projection.

## 6. Kiến trúc mã nguồn hiện tại

```text
Method/CURA/
├── readme.md
├── configs/
│   ├── default.json
│   ├── sentiment.json
│   ├── sentiment_paper.json
│   ├── sentiment_astar.json
│   ├── sentiment_stable_mvp.json
│   └── mixed_objective_mvp.json
├── core/
│   ├── cache.py
│   ├── cache_io.py
│   ├── cache_status.py
│   ├── annotate_targets.py
│   ├── calibrate.py      # readiness check, không fit model riêng
│   ├── features.py
│   ├── fusion.py
│   ├── generate.py
│   ├── train.py
│   ├── oracle_study.py
│   ├── diagnose_checkpoint.py
│   └── validate_run.py
├── models/
│   ├── calibrator.py
│   └── controller.py
├── router/
│   ├── adapters/
│   │   ├── base.py
│   │   └── registry.py
├── checkpoints/
└── reports/
```

`registry.py` bọc lại adapter đã có trong `Method/MultiSignal`; CURA chưa tách mỗi
signal thành một file adapter riêng.

## 7. Interface bắt buộc của adapter

Mọi adapter phải trả về cùng schema:

```python
SignalOutput(
    name: str,
    objective: str,
    scores: Tensor,          # [batch, top_k]
    available: bool,
    source: str,             # real checkpoint hoặc fallback
    cost_ms: float,
    metadata: dict,
)
```

Quy ước:

- Score lớn hơn luôn phải mang nghĩa candidate tốt hơn.
- Adapter chịu trách nhiệm xử lý tokenizer khác nhau.
- Candidate được tạo bởi tokenizer của base LM; adapter không được tự thay đổi candidate set.
- Nếu candidate sau decode/re-encode không tương đương, phải ghi `tokenization_mismatch=true`.
- Fallback chỉ dùng cho smoke test, không dùng trong paper tables.

## 8. Cache schema v2

```python
{
    "prompt_id": str,
    "step": int,
    "prefix_token_ids": Tensor[L],
    "prefix": str | None,          # chỉ có khi cache.store_text=true
    "candidate_token_ids": Tensor[K],
    "candidate_text": list[str],
    "base_logits": Tensor[K],
    "base_probs": Tensor[K],
    "raw_scores": Tensor[J, K],
    "signal_mask": Tensor[J],
    "signal_sources": list[str],
    "signal_objectives": list[str],
    "cost_ms": Tensor[J],
    "gold_index": int | None,
    "target_utilities": Tensor[K] | None,
    "preference_alpha": Tensor[M] | None,
}
```

`target_utilities` là rollout/evaluator score độc lập cho từng candidate và chỉ cần ở subset dùng train calibration hoặc oracle study.

### 8.1. Cache phải lưu theo shard

Không lưu toàn bộ split vào một file `train.pt` chỉ ở cuối quá trình. Nếu tiến trình hoặc server dừng trước lúc `torch.save`, toàn bộ phần đã tính sẽ bị mất. CURA phải lưu cache theo từng nhóm prompt:

```text
dataset/cura_cache/
├── train/
│   ├── manifest.json
│   ├── shards/
│   │   ├── shard_000000.pt
│   │   ├── shard_000001.pt
│   │   └── ...
│   ├── target_shards/   # copy-on-write shards có held-out rollout utility
│   └── errors.jsonl
└── validation/
    ├── manifest.json
    ├── shards/
    └── errors.jsonl
```

Mặc định:

```text
shard_size = 100 prompts
```

Nếu server dừng đột ngột, tối đa chỉ phải tính lại shard đang xử lý, không phải toàn bộ split. Có thể giảm `shard_size` xuống 25–50 nếu mỗi prompt mất nhiều thời gian.

### 8.2. Atomic save

Mỗi shard phải được ghi vào file tạm, kiểm tra đọc lại thành công, sau đó mới đổi tên thành `.pt`:

```python
tmp_path = shard_path.with_suffix(".pt.tmp")
torch.save(payload, tmp_path)

with open(tmp_path, "rb") as f:
    os.fsync(f.fileno())

loaded = torch.load(tmp_path, map_location="cpu", weights_only=False)
validate_shard(loaded)
os.replace(tmp_path, shard_path)  # atomic trên cùng filesystem
```

`manifest.json` cũng phải được cập nhật bằng cơ chế `manifest.json.tmp` rồi `os.replace`. Không đánh dấu shard là hoàn thành trước khi file shard đã được kiểm tra và đổi tên thành công.

### 8.3. Manifest schema

```json
{
  "schema_version": 2,
  "method": "cura",
  "split": "train",
  "dataset_path": "dataset/cura_train/train.jsonl",
  "dataset_fingerprint": "sha256:...",
  "config_fingerprint": "sha256:...",
  "signals": ["genarm", "rad", "cdq", "args"],
  "objective": "sentiment",
  "top_k": 20,
  "max_response_tokens": 32,
  "shard_size": 100,
  "total_prompts": 7500,
  "completed_prompts": 2300,
  "completed_shards": [
    {
      "id": 0,
      "file": "shards/shard_000000.pt",
      "start_index": 0,
      "end_index": 99,
      "num_prompts": 100,
      "num_steps": 2874,
      "sha256": "..."
    }
  ],
  "failed_prompt_ids": [],
  "status": "running"
}
```

`dataset_fingerprint` phải phụ thuộc vào nội dung dataset. `config_fingerprint` phải bao gồm base model, signal checkpoints, objective, `top_k`, maximum length, dtype và các tùy chọn ảnh hưởng đến score.

### 8.4. Resume rules

Khi chạy với `--resume`, `cache.py` phải:

1. Đọc `manifest.json`.
2. So sánh dataset/config fingerprint với lần chạy hiện tại.
3. Kiểm tra tất cả shard đã ghi trong manifest tồn tại, đọc được và có checksum đúng.
4. Bỏ qua các prompt thuộc shard hoàn chỉnh.
5. Bỏ qua file `.tmp` còn sót lại và tính lại shard chưa hoàn chỉnh.
6. Tiếp tục từ shard đầu tiên chưa hoàn thành.

Nếu fingerprint khác, chương trình phải dừng với thông báo rõ ràng; không được nối dữ liệu mới vào cache cũ. Người dùng phải chọn cache directory mới hoặc chủ động chạy chế độ restart.

### 8.5. Error handling

- Lỗi cuối cùng của một prompt được ghi vào `errors.jsonl` cùng `prompt_id`, index,
  exception và traceback; code chưa lưu riêng số lần thử.
- Cho phép `max_retries` cho lỗi tạm thời.
- Sau khi hoàn thành các prompt hợp lệ, chạy `--retry-errors` để xử lý lại prompt lỗi.
- Shard vẫn có descriptor cho toàn bộ khoảng prompt ngay cả khi có lỗi; prompt lỗi được
  theo dõi bằng `failed_prompt_ids`. Trainer mặc định từ chối cache còn prompt lỗi.
- Không silently thay signal thật bằng fallback khi một adapter lỗi trong paper mode.

### 8.6. Shard index và chế độ nạp dữ liệu

`ShardedCuraDataset` tạo index `(shard_file, local_index)` và mặc định chỉ giữ một shard
trong RAM:

```python
dataset = ShardedCuraDataset(cache_dir="dataset/cura_cache/train", preload=False)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

Trainer hiện dùng `preload=True` vì random shuffle với cache một shard sẽ gây reload
liên tục; do đó toàn bộ các row train/validation được nạp vào RAM khi train. Cache vẫn
được lưu và verify theo shard, không merge thành `train.pt`.

## 9. Dữ liệu huấn luyện

### 9.1. Smoke/imitation mode

Cho phép schema cũ:

```json
{"prompt": "...", "response": "..."}
```

Chế độ này dùng teacher forcing để kiểm tra pipeline, nhưng không đủ làm bằng chứng chính cho paper.

### 9.2. Paper mode

Input hiện vẫn là một response cho mỗi prompt:

```json
{
  "prompt": "...",
  "response": "...",
  "objective": "sentiment"
}
```

Cache giữ natural top-k ở mỗi teacher-forced step. Sau đó `annotate-targets` rollout
từng candidate và thêm:

```text
target_utilities[t, k] = mean held-out evaluator score over rollouts
```

Schema response-level `chosen/rejected` và trường `objective_scores` chưa được loader
sử dụng. Train, validation và test vẫn phải tách theo prompt. Evaluator tạo target không
được trùng model dùng làm signal chính.

## 10. Training objectives

### 10.1. Teacher-forced token loss

MVP tối ưu negative log-likelihood của gold token trong candidate support:

```text
L_nll = -log p_final(gold_token)
```

Paper cache giữ natural top-k. Khi `ignore_forced_gold=true`, những step mà gold token
không nằm trong natural top-k không đóng góp vào `L_nll`.

### 10.2. Candidate utility và ranking loss

Với rollout utility `q*` trên top-k candidate:

```text
L_utility = mean(max_k q*[k] - sum_k p_final[k] q*[k])

k+ = argmax_k q*[k]
k- = argmin_k q*[k]
L_pref = softplus(-beta * (log p_final[k+] - log p_final[k-]))
```

Đây là token-level candidate-ranking loss, không phải sequence preference loss trên
một cặp response `chosen/rejected`.

### 10.3. Calibration/uncertainty loss

Nếu có candidate rollout utility `q*`:

```text
L_cal = 0.5 * mean((q* - mu)^2 / exp(log_var) + log_var)
```

### 10.4. KL regularization

```text
L_kl = mean_t KL(p_final_t || p_base_t)
```

### 10.5. Cost regularization

```text
L_cost = mean_t sum_j w_t[j] * signal_cost[j]
```

### 10.6. Selector loss

Với mỗi signal, code đo mean squared error giữa calibrated utility và rollout target.
Target phân phối của selector ưu tiên signal có error thấp và cost nhỏ:

```text
e_j = mean_k (mu[j,k] - q*[k])^2
selector_target = softmax(-e_j / tau_selector - eta_cost * cost_j)
L_selector = CE(selector_logits(base_features), selector_target)
```

Selector chỉ dùng tám base-routing features để quyết định signal nào được gọi trước
khi reward score tồn tại.

### 10.7. Gate regularization

Khi có rollout target, gate target được suy ra từ chênh lệch expected utility giữa
guided và base policy, sau khi trừ uncertainty/disagreement penalty:

```text
gate_target = sigmoid(
    (U_guided - U_base
     - eta_u * uncertainty
     - eta_d * disagreement) / tau_gate
)
L_gate = BCEWithLogits(gate_logit, stop_gradient(gate_target))
```

Khi không có rollout target, MVP dùng heuristic theo median trong batch:

```text
uncertain_step = high_uncertainty OR high_disagreement
L_gate = BCE(g_t, 1 - uncertain_step)
```

Tổng loss:

```text
L_total = gamma_nll      * L_nll
        + gamma_utility  * L_utility
        + gamma_pref     * L_pref
        + gamma_cal      * L_cal
        + gamma_kl       * L_kl
        + gamma_cost     * L_cost
        + gamma_gate     * L_gate
        + gamma_selector * L_selector
```

Mỗi thành phần chỉ có tác dụng khi hệ số tương ứng khác 0. Nếu chưa có rollout targets,
`L_cal`, `L_utility`, `L_pref` và `L_selector` bằng 0; gate dùng heuristic. Kết quả này
phải được ghi là CURA-MVP, không phải full CURA.

## 11. Config đại diện cho code hiện tại

Đoạn dưới là dạng rút gọn sau khi merge `sentiment_paper.json` với config cha. Các
trường atomic save/checksum/resume là hành vi của code, không phải feature flag trong
config. Đây là config generic cũ dùng `ARGS`; không phải signal set mục tiêu mới.
Checkpoint full hiện có `cura_astar_full_v1.pt` dùng `rad, cdq`, còn experiment kế
tiếp chỉ chuyển sang `rad, cdq, genarm` sau khi có checkpoint sentiment-GenARM đúng
objective và đã rebuild toàn bộ cache/controller như mô tả ở mục 4.1.

```json
{
  "method": "cura",
  "signals": ["rad", "cdq", "args"],
  "objective": "sentiment",
  "paper_mode": true,
  "signal_objectives": {
    "rad": "sentiment",
    "cdq": "sentiment",
    "args": "sentiment"
  },
  "controller": {
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.1,
    "lambda_max": 3.0,
    "kappa": 0.5,
    "disagreement_penalty": 0.1,
    "epsilon_kl": 0.15,
    "signal_costs": [1.0, 0.8, 1.1]
  },
  "calibration": {
    "mode": "learned_heteroscedastic",
    "hidden_dim": 32,
    "epsilon": 0.000001
  },
  "cache": {
    "schema_version": 2,
    "shard_size": 100,
    "dtype": "float16",
    "max_retries": 2,
    "store_text": false
  },
  "training": {
    "batch_size": 32,
    "epochs": 10,
    "lr": 0.0001,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "gamma_nll": 0.25,
    "gamma_utility": 1.0,
    "gamma_pref": 1.0,
    "gamma_cal": 1.0,
    "gamma_kl": 0.1,
    "gamma_cost": 0.01,
    "gamma_gate": 0.1,
    "gamma_selector": 0.25,
    "preference_beta": 1.0,
    "selector_temperature": 0.25,
    "selector_cost_weight": 0.01,
    "gate_temperature": 0.1,
    "gate_uncertainty_penalty": 0.05,
    "gate_disagreement_penalty": 0.05,
    "ignore_forced_gold": true,
    "num_workers": 0,
    "mixed_precision": false
  }
}
```

Các giá trị trên là mặc định hiện tại của `sentiment_paper.json`, không phải
hyperparameter đã được chứng minh tối ưu.

## 12. Pipeline CLI đề xuất

Chạy từ repo root:

```bash
cd "/home/jupyter-iec2024se10/Test-time-Alignment"
```

### 12.1. Audit checkpoint và objective

```bash
python Method/run_method.py \
  --method cura \
  --action audit-signals \
  --config Method/CURA/configs/sentiment.json
```

Audit phải fail nếu:

- thiếu checkpoint thật;
- objective không khớp;
- score direction không được khai báo là `higher_is_better`.

Tokenizer mapping chưa được kiểm tra bởi action `audit-signals`. Mismatch được đo khi
adapter chấm candidate và paper cache sẽ fail nếu mismatch rate vượt threshold.

### 12.2. Build cache

```bash
python Method/run_method.py \
  --method cura \
  --action cache \
  --config Method/CURA/configs/sentiment_paper.json \
  --split train \
  --cache-dir dataset/cura_cache/train \
  --shard-size 100 \
  --resume \
  --max-response-tokens 32 \
  --top-k 20
```

Lệnh trên có thể chạy lại nguyên vẹn sau khi server dừng. Các shard hoàn chỉnh sẽ được giữ lại và bỏ qua.

Kiểm tra tiến độ mà không chạy cache:

```bash
python Method/run_method.py \
  --method cura \
  --action cache-status \
  --cache-dir dataset/cura_cache/train
```

Kiểm tra toàn bộ shard và checksum:

```bash
python Method/run_method.py \
  --method cura \
  --action cache-verify \
  --cache-dir dataset/cura_cache/train
```

Chạy lại các prompt lỗi sau khi cache chính hoàn thành:

```bash
python Method/run_method.py \
  --method cura \
  --action cache \
  --cache-dir dataset/cura_cache/train \
  --resume \
  --retry-errors
```

### 12.3. Tạo held-out rollout targets và kiểm tra calibrator

Không dùng `models/rad_rm_sentiment` hoặc
`models/sentiment-roberta-large-english` làm evaluator vì chúng đã là signal.
`HELD_OUT_EVALUATOR` phải là checkpoint độc lập do người chạy cung cấp.

```bash
python Method/run_method.py --method cura --action annotate-targets \
  --split train --cache-dir dataset/cura_cache/train \
  --evaluator "$HELD_OUT_EVALUATOR" --rollouts 2 --rollout-tokens 16
python Method/run_method.py --method cura --action annotate-targets \
  --split validation --cache-dir dataset/cura_cache/validation \
  --evaluator "$HELD_OUT_EVALUATOR" --rollouts 2 --rollout-tokens 16
```

Target shards được ghi copy-on-write vào `target_shards/`; base shards không bị
overwrite và manifest chỉ trỏ sang target shard sau khi atomic verification thành công.

```bash
python Method/run_method.py \
  --method cura \
  --action calibrate \
  --config Method/CURA/configs/sentiment_paper.json \
  --cache-dir dataset/cura_cache/validation
```

Lệnh `calibrate` trên chỉ verify shard và kiểm tra mọi row validation có
`target_utilities` khi dùng learned calibration. Calibrator được train cùng controller
ở action `train`, không được fit hoặc lưu riêng bởi action này.

### 12.4. Train controller

```bash
python Method/run_method.py \
  --method cura \
  --action train \
  --config Method/CURA/configs/sentiment_paper.json \
  --cache-dir dataset/cura_cache/train \
  --validation-cache-dir dataset/cura_cache/validation
```

### 12.5. Generate

```bash
python Method/run_method.py \
  --method cura \
  --action generate \
  --input dataset/rad_benchmark/all.jsonl \
  --output results/cura.json \
  --num-prompts 1000 \
  --max-new-tokens 32 \
  --top-k 20
```

Giảm chi phí thật bằng cách chỉ gọi hai expert được selector chọn ở mỗi token:

```bash
python Method/run_method.py --method cura --action generate \
  --input dataset/rad_benchmark/all.jsonl --output results/cura_budget2.jsonl \
  --num-prompts 10000 --signal-budget 2 --seed 42
```

Unseen-base transfer dùng `--base-model`. Candidate luôn được decode bằng tokenizer
của base rồi re-tokenize độc lập trong từng expert; output ghi mismatch rate:

```bash
python Method/run_method.py --method cura --action generate \
  --base-model models/ANOTHER_CAUSAL_LM --output results/cura_transfer.jsonl \
  --num-prompts 10000 --signal-budget 2 --seed 42
```

### 12.6. Validate paper run

```bash
python Method/CURA/core/validate_run.py \
  --input results/cura.json \
  --paper-mode
```

### 12.7. Oracle, robustness và thống kê

```bash
python Method/run_method.py --method cura --action oracle \
  --cache-dir dataset/cura_cache/validation
python Method/run_method.py --method cura --action generate \
  --output results/cura_corrupt_rad.jsonl --corrupt-signal rad --corrupt-std 1.0
python Method/run_method.py --method cura --action generate \
  --output results/cura_no_uncertainty.jsonl --ablation no-uncertainty
python Method/run_method.py --method cura --action generate \
  --output results/cura_leave_rad.jsonl --leave-out rad
```

Sau khi evaluator độc lập thêm cùng một trường score vào mỗi JSONL:

```bash
python Method/CURA/core/analyze_results.py \
  --files base=results/base_scored.jsonl cura=results/cura_scored.jsonl \
  --baseline base --score-field score --bootstrap 10000
```

## 13. Output telemetry

Mỗi sample thành công hiện chứa metadata đầu vào (`id`, `md5_hash`, `prompt`,
`reference`) cùng các trường generation sau. Ví dụ rút gọn cho config ba signal:

```json
{
  "id": "...",
  "md5_hash": "...",
  "response": "...",
  "method": "cura",
  "objective": "sentiment",
  "signal_sources": ["rad_reward_model", "cd_prefix_value_scorer", "skipped_by_selector"],
  "signal_objectives": {"rad": "sentiment", "cdq": "sentiment", "args": "sentiment"},
  "mean_weights": {"rad": 0.4, "cdq": 0.35, "args": 0.25},
  "mean_uncertainty": {"rad": 0.1, "cdq": 0.2, "args": 0.3},
  "selection_rate": {"rad": 1.0, "cdq": 1.0, "args": 0.0},
  "tokenization_mismatch_rate": {"rad": 0.0, "cdq": 0.0, "args": 0.0},
  "mean_disagreement": 0.12,
  "mean_strength": 1.1,
  "mean_gate": 0.76,
  "mean_base_weight": 0.24,
  "mean_kl": 0.08,
  "latency_ms": 123.4,
  "checkpoint_sha256": "...",
  "config_fingerprint": "sha256:...",
  "status": "success",
  "error": null
}
```

`signal_sources` hiện lấy từ decoding step cuối cùng; các thống kê `mean_*` và
`selection_rate` mới được aggregate trên toàn response. Code chưa ghi token trace hoặc
`token_trace_path`.

## 14. Oracle study đúng cho CURA

### 14.1. Fixed-best/global expert (đã có CLI)

CLI tính utility trung bình của từng expert và chọn một expert tốt nhất trên toàn bộ
token rows có rollout target.

### 14.2. Prompt oracle (chưa triển khai trong CURA CLI)

Mục tiêu dự kiến là chọn expert có final reward tốt nhất cho từng prompt. Action
`oracle` hiện chưa group token rows theo prompt và chưa tính prompt-oracle gap.

### 14.3. Token/step oracle (đã có CLI)

Tại một subset nhỏ, rollout mỗi candidate bằng cùng continuation policy và chấm bằng held-out evaluator:

```text
q*(prefix, candidate) = mean final evaluator score over N rollouts
```

CLI hiện báo cáo base greedy utility, utility của candidate do từng expert chọn,
fixed-best expert, candidate-level token oracle và số step mỗi expert thắng. Các metric
sau vẫn là phần mở rộng chưa triển khai:

- prompt-oracle to token-oracle gap;
- CURA regret so với từng oracle;
- hiệu năng theo mức disagreement.

Không đưa output của CURA vào tập expert khi tính baseline oracle chính, vì điều này làm thay đổi ý nghĩa của oracle gap.

## 15. Baselines và ablations

### 15.1. Baselines

- Base LM;
- ARGS;
- RAD;
- CD-Q;
- GenARM;
- Uniform fusion;
- Best fixed weights;
- Prompt-level router;
- TARo;
- PARM nếu chạy multi-objective;
- Oracle selector.

### 15.2. Ablations bắt buộc

- CURA without calibration;
- CURA without uncertainty;
- CURA without disagreement features;
- CURA without gate;
- CURA without KL constraint;
- CURA with fixed `lambda`;
- CURA with mean token weights;
- CURA with shuffled uncertainty;
- CURA with one corrupted/noisy signal;
- Leave-one-signal-out;
- Heuristic inverse-uncertainty weighting thay learned controller.

Hai ablation quan trọng nhất:

1. **Corrupted-signal test:** cố ý thêm noise/bias vào một signal và kiểm tra controller có giảm trọng số signal đó không.
2. **Unseen-base transfer:** train controller trên một base LM, đánh giá trên base LM chưa thấy khi train.

## 16. Metrics

### Alignment

- Held-out reward;
- preference win rate;
- task success;
- worst-group hoặc worst-objective reward.

### Generation quality

- perplexity;
- repetition;
- diversity;
- length;
- KL so với base LM.

### Calibration và routing

- ECE/Brier score nếu có probability target;
- negative log-likelihood của calibrator;
- uncertainty-error correlation;
- risk-coverage curve;
- oracle regret;
- weight specialization;
- gate rate theo uncertainty bucket.

### Efficiency

- latency/token;
- throughput;
- peak VRAM/RAM;
- số signal calls/token.

Kết quả chính phải báo cáo mean, standard deviation hoặc 95% confidence interval trên nhiều seed/prompt bootstrap.

## 17. Quy tắc paper-quality

Một run chỉ được đưa vào bảng chính khi:

- toàn bộ `signal_sources` là checkpoint thật;
- objective của các signal tương thích;
- test prompts không dùng để fit calibrator/controller;
- evaluator độc lập với signal models;
- không silently fallback;
- cùng decoding budget giữa các baseline;
- cùng base LM, top-k, maximum length và sampling seed protocol;
- latency được đo trên cùng phần cứng;
- output và config được lưu để tái lập.

## 18. Lộ trình triển khai

Trạng thái dưới đây tách **code path** khỏi **experiment**. Stage 1–5 đã có code path
chính và một số synthetic test; Stage 6 có một phần CLI nhưng chưa có kết quả full-scale;
Stage 7 chưa chạy. Không được đánh dấu paper-ready chỉ dựa trên unit test.

### Stage 0 — Giữ baseline cũ

- Đóng băng kết quả MultiSignal hiện tại.
- Lưu commit/config/checkpoint để làm baseline `learned-fusion`.

### Stage 1 — Rename và interface (code hoàn thành)

- Tạo `Method/CURA/` từ MultiSignal.
- Đổi runner từ `multisignal` sang `cura`.
- Chuẩn hóa adapter interface và metadata.

### Stage 1.5 — Fault-tolerant cache (code hoàn thành)

- Chuyển cache từ một file `.pt` sang sharded cache.
- Thêm atomic save cho shard và manifest.
- Thêm dataset/config fingerprint.
- Thêm `--resume`, `cache-status`, `cache-verify` và `--retry-errors`.
- Dataset hỗ trợ lazy one-shard loading; trainer hiện preload toàn bộ row để shuffle hiệu quả.
- Resume/checksum đã có code path; fault-injection test cho dừng giữa shard vẫn còn thiếu.

### Stage 2 — Objective audit (code hoàn thành; full four-signal artifact còn thiếu)

- Kiểm tra artifact khai báo có tồn tại, objective metadata có khớp và score direction
  có được khai báo là `higher_is_better`.
- Code chưa tự suy luận training objective từ nội dung checkpoint; độ đúng của metadata
  vẫn cần người chạy xác nhận.
- Chọn một objective chung cho thí nghiệm đầu tiên.
- Không tiếp tục full run nếu objective chưa khớp.

### Stage 3 — Calibration (code hoàn thành; benchmark calibration còn thiếu)

- Triển khai `robust_zscore` trước.
- Sau đó triển khai learned calibrator và uncertainty.
- Vẽ reliability/calibration plots trên validation.

### Stage 4 — CURA controller (code hoàn thành)

- Thêm base confidence, uncertainty và disagreement features.
- Tách `strength` và `gate` theo công thức policy mixture.
- Thêm KL projection.

### Stage 5 — Training và sanity checks (synthetic train path hoàn thành một phần)

- Overfit một batch nhỏ.
- Kiểm tra weights tổng bằng 1.
- Kiểm tra probabilities hữu hạn và tổng bằng 1.
- Khi `g=0`, output phải bằng base policy.
- Khi `lambda=0`, guided policy phải bằng base policy (invariant từ công thức, chưa có test riêng).
- Monotonic relation giữa uncertainty và learned weight chưa được đảm bảo bởi kiến trúc;
  cần đánh giá thực nghiệm thay vì coi là invariant.

### Stage 6 — Oracle và ablation (CLI hoàn thành một phần; experiment chưa chạy)

- Fixed-best/global expert và token-candidate oracle đã có CLI.
- Prompt oracle, CURA oracle regret và disagreement-bucket report chưa có.
- Chạy corrupted-signal và leave-one-out.

### Stage 7 — Scale-up (chưa chạy)

- Nhiều seed;
- nhiều dataset;
- nhiều base-model family;
- 2,500–7,500 prompts sau khi correctness đã được xác nhận.

## 19. Unit tests tối thiểu

Test suite hiện kiểm tra math invariants, candidate-specific learned uncertainty,
selector budget, KL bound, cache atomic/lazy read, gradient flow của preference và
calibration loss, cùng một train CLI end-to-end trên synthetic shards. Danh sách
dưới đây là acceptance suite đầy đủ cần đạt trước khi freeze paper artifact; các
test phụ thuộc model/GPU và fault injection vẫn phải chạy trong CI integration.

```text
test_adapter_score_shape
test_adapter_score_direction
test_candidate_tokenizer_roundtrip
test_calibrator_no_test_leakage
test_uncertainty_positive
test_controller_weights_sum_to_one
test_gate_range
test_lambda_range
test_fallback_equals_base
test_kl_budget
test_no_paper_fallback_signal
test_objective_compatibility
test_cache_schema_version
test_cache_atomic_write
test_cache_resume_skips_completed_shards
test_cache_resume_recomputes_incomplete_shard
test_cache_rejects_dataset_fingerprint_mismatch
test_cache_rejects_config_fingerprint_mismatch
test_cache_detects_corrupted_shard
test_cache_manifest_updated_after_shard_commit
test_cache_retry_errors
test_sharded_dataset_lazy_loading
test_reproducible_generation
```

## 20. Contribution dự kiến của paper

Không viết contribution là “kết hợp bốn reward models bằng MLP”. Nên viết:

1. Đặt bài toán **heterogeneous reward arbitration** cho token-level test-time alignment.
2. Đề xuất signal-specific calibration và uncertainty estimation để so sánh reward evidence khác loại.
3. Đề xuất disagreement-aware controller với explicit fallback và KL-bounded decoding.
4. Xây dựng protocol đánh giá complementarity, corrupted-signal robustness và oracle regret.
5. Thực nghiệm trên nhiều objective, dataset và base-model family với evaluator độc lập.

## 21. Tiêu chí quyết định có tiếp tục hướng này

Trước khi scale lên hàng nghìn prompt, cần đạt ba điều kiện:

1. Prompt-oracle gap lớn và ổn định: các expert thật sự bổ sung cho nhau.
2. CURA vượt uniform fusion và best fixed weights có ý nghĩa thống kê.
3. Khi một signal bị làm nhiễu, CURA giảm trọng số của nó và ít suy giảm hơn controller không có uncertainty.

Nếu không đạt điều kiện 1, vấn đề không nằm ở controller mà ở việc các expert thiếu tính bổ sung. Khi đó cần thay reward signals hoặc thay objective trước khi scale.

## 22. Tên paper gợi ý

> **CURA: Calibrated Uncertainty-aware Reward Arbitration for Test-Time Language Model Alignment**

Tên ngắn cho code:

```text
method = cura
folder = Method/CURA
checkpoint = cura_controller.pt
result = results/cura.json
```
