# DynaCAP-PARM — Hướng dẫn chạy

Môi trường: **genarm**
Output: `results/dynacap_parm.json` (cùng schema với các method khác)

---

## Chạy tiếp từ trạng thái hiện tại

Oracle-w hiện có verdict `TOKEN_LEVEL_HEADROOM_WEAK`. Vì vậy quy trình dưới
đây chỉ tạo checkpoint để **smoke-test kỹ thuật**, chưa được dùng để tuyên bố
Gate E/F đã pass.

### 1. Xuất dataset huấn luyện TokenRouter

Không dùng `results/oracle_probe_train.jsonl`: file đó chứa objective probe,
không chứa nhãn can thiệp `w_t`.

```bash
cd /home/jupyter-iec2024se10/Test-time-Alignment

/home/jupyter-iec2024se10/miniforge3/bin/conda run -n genarm python3 \
    Method/CAP_PARM_MATO/scripts/build_w_oracle.py \
    --out results/w_oracle_train.jsonl \
    --allow-weak-gate
```

Kết quả hiện tại dự kiến có 61 trạng thái không-tie: 27 nhãn `invoke` và 34
nhãn `skip`.

### 2. Train TokenRouter

```bash
/home/jupyter-iec2024se10/miniforge3/bin/conda run -n genarm python3 \
    Method/CAP_PARM_MATO/scripts/train_token_router.py \
    --oracle-data results/w_oracle_train.jsonl \
    --ckpt-out results/token_router.pt \
    --epochs 100 \
    --patience 15 \
    --seed 42
```

Hai file đầu ra:

- `results/token_router.pt`
- `results/token_router.pt.metrics.json`

### 3. Smoke-test DynW trên 2 prompt

```bash
/home/jupyter-iec2024se10/miniforge3/bin/conda run -n genarm python3 \
    Method/CAP_PARM_MATO/scripts/generate.py \
    --num-prompts 2 \
    --max-new-tokens 16 \
    --alpha-help 0.5 \
    --alpha-harm 0.5 \
    --w-fixed 1.0 \
    --dynamic-w \
    --router-ckpt results/token_router.pt \
    --output-path results/dynacap_parm_dynw_smoke.json
```

Kiểm tra nhanh kết quả:

```bash
/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("results/dynacap_parm_dynw_smoke.json")
rows = json.loads(path.read_text(encoding="utf-8"))
for row in rows:
    print({
        "id": row["id"],
        "status": row["status"],
        "parm_calls": row["parm_calls"],
        "parm_processed_tokens": row["parm_processed_tokens"],
        "replay_tokens": row["replay_tokens"],
        "intervention_rate": row["intervention_rate"],
        "error": row["error"],
    })
PY
```

Không dùng `conda run ... python3 -` cho heredoc này vì `conda run` có thể
không forward stdin, khiến command kết thúc mà không in kết quả.

Chỉ chuyển sang bước tiếp theo nếu cả hai record có `status="success"`.

### 4. Chạy DynW đầy đủ

```bash
/home/jupyter-iec2024se10/miniforge3/bin/conda run -n genarm python3 \
    Method/CAP_PARM_MATO/scripts/generate.py \
    --num-prompts 1000 \
    --max-new-tokens 64 \
    --alpha-help 0.5 \
    --alpha-harm 0.5 \
    --w-fixed 1.0 \
    --dynamic-w \
    --router-ckpt results/token_router.pt \
    --output-path results/dynacap_parm_dynw.json
```

Để có kết quả khoa học chính thức, bỏ quy trình override ở trên, mở rộng/rerun
Oracle-w cho đến khi Gate E pass, rồi mới train checkpoint chính thức.

---

## Các chế độ chạy

| Chế độ | dynamic_alpha | dynamic_w | Mô tả |
|---|---|---|---|
| **Fixed** | ✗ | ✗ | PARM với alpha và w cố định (baseline) |
| **DynAlpha** | ✓ | ✗ | MATO-style alpha cập nhật động |
| **DynW** | ✗ | ✓ | TARo-style token router chọn w_t |
| **Joint** | ✓ | ✓ | Cả hai động (full DynaCAP-PARM) |

---

## 1. Chế độ Fixed (baseline — không cần train)

```bash
cd /home/jupyter-iec2024se10/Test-time-Alignment

/home/jupyter-iec2024se10/miniforge3/bin/conda run -n genarm python3 \
    Method/CAP_PARM_MATO/scripts/generate.py \
    --num-prompts 1000 \
    --alpha-help 0.5 \
    --alpha-harm 0.5 \
    --w-fixed 1.0 \
    --fusion-mode parm_product \
    --output-path results/dynacap_parm_fixed.json
```

Đây là baseline theo công thức PARM `log p_base + log p_PARM`. Các experiment
controller dùng `--fusion-mode convex` và phải được báo cáo tách biệt.

---

## 2. Chế độ DynAlpha (không cần train — oracle probing)

```bash
cd /home/jupyter-iec2024se10/Test-time-Alignment

/home/jupyter-iec2024se10/miniforge3/bin/conda run -n genarm python3 \
    Method/CAP_PARM_MATO/scripts/generate.py \
    --num-prompts 1000 \
    --alpha-help 0.5 \
    --alpha-harm 0.5 \
    --w-fixed 0.5 \
    --dynamic-alpha \
    --update-interval 8 \
    --temperature 0.5 \
    --smoothing 0.25 \
    --kl-budget 0.1 \
    --output-path results/dynacap_parm_dynalpha.json
```

> Nếu chưa có `--tracker-ckpt`, script sẽ dùng oracle one-hot probing (chậm hơn nhưng không cần train).

Sau khi Gate C pass, có thể build probe schema v2 và train tracker:

```bash
/home/jupyter-iec2024se10/miniforge3/bin/conda run -n genarm python3 \
    Method/CAP_PARM_MATO/scripts/build_dynamic_alpha_oracle.py \
    --num-prompts 200 \
    --out results/oracle_probe_train_v2.jsonl

/home/jupyter-iec2024se10/miniforge3/bin/conda run -n genarm python3 \
    Method/CAP_PARM_MATO/scripts/train_objective_tracker.py \
    --oracle-data results/oracle_probe_train_v2.jsonl \
    --ckpt-out results/objective_tracker.pt
```

Sau đó truyền `--tracker-ckpt results/objective_tracker.pt`. Không train tracker
từ `oracle_probe_train.jsonl` legacy.

File legacy này dùng cache neutral-α từ implementation trước; schema v2 replay
đầy đủ prefix riêng cho từng α để tránh dùng KV-cache sai preference.

---

## 3. Chế độ DynW (cần train TokenRouter trước)

### Bước 1: Build dataset Oracle-w cho TokenRouter

Nguồn mặc định là kết quả counterfactual đã được chấm độc lập tại
`Method/PARM_TARO/results/parm_taro/adaptive_parm/04_token_headroom/`.

```bash
cd /home/jupyter-iec2024se10/Test-time-Alignment

/home/jupyter-iec2024se10/miniforge3/bin/conda run -n genarm python3 \
    Method/CAP_PARM_MATO/scripts/build_w_oracle.py \
    --out results/w_oracle_train.jsonl
```

Kết quả oracle hiện tại có verdict `TOKEN_LEVEL_HEADROOM_WEAK`, vì vậy command
trên sẽ dừng theo stage gate trong `readme.md`. Chỉ để smoke-test kỹ thuật
(không dùng làm kết quả khoa học), thêm `--allow-weak-gate`.

### Bước 2: Train TokenRouter

*(Chỉ chạy chính thức sau khi Gate E pass — xem `readme.md`.)*

```bash
cd /home/jupyter-iec2024se10/Test-time-Alignment

/home/jupyter-iec2024se10/miniforge3/bin/conda run -n genarm python3 \
    Method/CAP_PARM_MATO/scripts/train_token_router.py \
    --oracle-data results/w_oracle_train.jsonl \
    --ckpt-out results/token_router.pt
```

Trainer tách validation theo `sample_id`, bỏ tie, cân bằng BCE và ghi metrics
tại `results/token_router.pt.metrics.json`.

### Bước 3: Chạy với router đã train

```bash
cd /home/jupyter-iec2024se10/Test-time-Alignment

/home/jupyter-iec2024se10/miniforge3/bin/conda run -n genarm python3 \
    Method/CAP_PARM_MATO/scripts/generate.py \
    --num-prompts 1000 \
    --alpha-help 0.5 \
    --alpha-harm 0.5 \
    --w-fixed 1.0 \
    --dynamic-w \
    --router-ckpt results/token_router.pt \
    --output-path results/dynacap_parm_dynw.json
```

---

## 4. Chế độ Joint (dynamic alpha + dynamic w)

*(Chỉ chạy sau khi cả Gate C và Gate E pass)*

```bash
cd /home/jupyter-iec2024se10/Test-time-Alignment

/home/jupyter-iec2024se10/miniforge3/bin/conda run -n genarm python3 \
    Method/CAP_PARM_MATO/scripts/generate.py \
    --num-prompts 1000 \
    --alpha-help 0.5 \
    --alpha-harm 0.5 \
    --w-fixed 1.0 \
    --dynamic-alpha \
    --dynamic-w \
    --router-ckpt results/token_router.pt \
    --update-interval 8 \
    --output-path results/dynacap_parm_joint.json
```

---

## Tham số quan trọng

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--num-prompts` | `1000` | Số prompt |
| `--alpha-help` | `0.5` | Tỷ trọng Helpfulness |
| `--alpha-harm` | `0.5` | Tỷ trọng Harmlessness |
| `--w-fixed` | `1.0` | w khi router quyết định bật PARM |
| `--dynamic-alpha` | `False` | Bật MATO alpha update |
| `--dynamic-w` | `False` | Bật TARo token router |
| `--update-interval` | `8` | Cập nhật alpha mỗi N token |
| `--temperature` | `0.5` | MATO temperature τ |
| `--kl-budget` | `0.1` | KL trust region ρ |
| `--tracker-ckpt` | `None` | Checkpoint ObjectiveTracker (nếu có) |
| `--router-ckpt` | `None` | Checkpoint TokenRouter |
| `--output-path` | `results/dynacap_parm.json` | File output |

---

## Cần train không?

| Chế độ | Cần train | Ghi chú |
|---|---|---|
| Fixed | ❌ | Chạy ngay |
| DynAlpha (oracle) | ❌ | Chạy ngay, dùng one-hot probing |
| DynAlpha (tracker) | ✅ | Train ObjectiveTracker (MLP nhỏ) |
| DynW | ✅ | Train TokenRouter (MLP nhỏ) |
| Joint | ✅ | Cần cả hai |

---

## Output schema (mỗi record)

```json
{
  "id": 0,
  "prompt": "...",
  "reference": "...",
  "response": "...",
  "method": "DynaCAP-PARM[dyn_alpha+dyn_w]",
  "fusion_mode": "convex",
  "w_fixed": 1.0,
  "alpha_helpfulness": 0.5,
  "alpha_harmlessness": 0.5,
  "dynamic_alpha": true,
  "dynamic_w": true,
  "parm_calls": 48,
  "intervention_rate": 0.75,
  "latency": 4.21,
  "status": "success",
  "error": null
}
```

---

## Cấu trúc thư mục

```
Method/CAP_PARM_MATO/
├── scripts/
│   ├── generate.py                  ← Script chạy chính (tất cả chế độ)
│   ├── build_dynamic_alpha_oracle.py ← Sinh objective probe cho dynamic alpha
│   ├── build_w_oracle.py             ← Chuyển rollout đã chấm thành nhãn w
│   ├── train_objective_tracker.py     ← Train prefix objective tracker
│   └── train_token_router.py          ← Train binary TokenRouter
├── src/
│   ├── models/
│   │   ├── base_adapter.py
│   │   ├── parm_adapter.py          ← Thêm probe_objectives
│   │   ├── objective_tracker.py     ← MLP ước tính reward
│   │   └── token_router.py          ← Binary gate
│   ├── controllers/
│   │   └── dynamic_preference.py    ← MATO entropic mirror descent
│   └── decoding/
│       └── fusion.py
└── HOWTO.md
```
