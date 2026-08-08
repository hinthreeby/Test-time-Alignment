# Router Framework - How to Run

Framework hiện hỗ trợ:

```text
rad
args
cd-fudge
cdq
genarm
```

Mỗi method chạy theo đúng thứ tự:

```text
1. Build router training data
2. Cache token-level features
3. Train router
4. Generate với router
```

---

# 1. RAD + Router

## Bước 1 - Tạo training data

```bash
python Method/Router/core/build_data.py --method rad
```

Output:

```text
dataset/router_train/rad/train.jsonl
dataset/router_train/rad/validation.jsonl
```

## Bước 2 - Cache feature

```bash
python Method/Router/core/cache.py --method rad
```

Output:

```text
dataset/router_cache/rad/train.pt
dataset/router_cache/rad/validation.pt
```

## Bước 3 - Train router

```bash
python Method/Router/core/train.py --method rad
```

Checkpoint:

```text
Method/Router/checkpoints/rad_router.pt
```

## Bước 4 - Generate

Test nhanh 5 prompts:

```bash
python Method/Router/core/generate.py --method rad --num-prompts 5
```

Chạy benchmark theo config:

```bash
python Method/Router/core/generate.py --method rad
```

Output:

```text
results/rad_router.json
```

---

# 2. ARGS + Router

Chạy lần lượt:

```bash
python Method/Router/core/build_data.py --method args
python Method/Router/core/cache.py --method args
python Method/Router/core/train.py --method args
python Method/Router/core/generate.py --method args --num-prompts 5
python Method/Router/core/generate.py --method args
```

Checkpoint:

```text
Method/Router/checkpoints/args_router.pt
```

Output:

```text
results/args_router.json
```

---

# 3. CD-FUDGE + Router

Chạy lần lượt:

```bash
python Method/Router/core/build_data.py --method cd-fudge
python Method/Router/core/cache.py --method cd-fudge
python Method/Router/core/train.py --method cd-fudge
python Method/Router/core/generate.py --method cd-fudge --num-prompts 5
python Method/Router/core/generate.py --method cd-fudge
```

Checkpoint:

```text
Method/Router/checkpoints/cd-fudge_router.pt
```

Output:

```text
results/cd-fudge_router.json
```

---

# 4. CD-Q + Router

Chạy lần lượt:

```bash
python Method/Router/core/build_data.py --method cdq
python Method/Router/core/cache.py --method cdq
python Method/Router/core/train.py --method cdq
python Method/Router/core/generate.py --method cdq --num-prompts 5
python Method/Router/core/generate.py --method cdq
```

Checkpoint:

```text
Method/Router/checkpoints/cdq_router.pt
```

Output:

```text
results/cdq_router.json
```

---

# 5. GenARM + Router

Chạy lần lượt:

```bash
python Method/Router/core/build_data.py --method genarm
python Method/Router/core/cache.py --method genarm
python Method/Router/core/train.py --method genarm
python Method/Router/core/generate.py --method genarm --num-prompts 5
python Method/Router/core/generate.py --method genarm
```

Checkpoint:

```text
Method/Router/checkpoints/genarm_router.pt
```

Output:

```text
results/genarm_router.json
```

---

# 6. Pipeline chung

Có thể thay `METHOD` bằng:

```text
rad
args
cd-fudge
cdq
genarm
```

Sau đó chạy:

```bash
python Method/Router/core/build_data.py --method METHOD
python Method/Router/core/cache.py --method METHOD
python Method/Router/core/train.py --method METHOD
python Method/Router/core/generate.py --method METHOD --num-prompts 5
python Method/Router/core/generate.py --method METHOD
```

---

# 7. Khi đã có training data

Nếu đã có:

```text
dataset/router_train/METHOD/train.jsonl
dataset/router_train/METHOD/validation.jsonl
```

thì bỏ qua `build_data.py`.

Chạy từ:

```bash
python Method/Router/core/cache.py --method METHOD
```

---

# 8. Khi đã có cache

Nếu đã có:

```text
dataset/router_cache/METHOD/train.pt
dataset/router_cache/METHOD/validation.pt
```

thì không cần cache lại.

Chạy:

```bash
python Method/Router/core/train.py --method METHOD
```

---

# 9. Khi đã train router

Nếu checkpoint đã tồn tại:

```text
Method/Router/checkpoints/METHOD_router.pt
```

thì có thể generate trực tiếp:

```bash
python Method/Router/core/generate.py --method METHOD --num-prompts 5
```

Sau khi test ổn:

```bash
python Method/Router/core/generate.py --method METHOD
```

---

# 10. Kiểm tra các adapter

```bash
python -m py_compile \
Method/Router/adapters/rad.py \
Method/Router/adapters/args.py \
Method/Router/adapters/cd.py \
Method/Router/adapters/genarm.py \
Method/Router/adapters/registry.py
```

Kiểm tra registry:

```bash
python - <<'PY'
from Method.Router.adapters.registry import ADAPTERS
print("Available router methods:", list(ADAPTERS))
PY
```

Kết quả mong đợi:

```text
Available router methods: ['rad', 'args', 'cd-fudge', 'cdq', 'genarm']
```

---

# 11. Xem config

```bash
cat Method/Router/configs/rad.json
cat Method/Router/configs/args.json
cat Method/Router/configs/cd-fudge.json
cat Method/Router/configs/cdq.json
cat Method/Router/configs/genarm.json
```

Các tham số thường chỉnh:

```text
top_k
max_response_tokens
max_new_tokens

router.hidden_dim
router.output_min
router.output_max

training.epochs
training.lr

router_data.num_train
router_data.num_val

benchmark.num_prompts
```

---

# 12. Thứ tự khuyến nghị khi test lần đầu

Test RAD trước:

```bash
python Method/Router/core/build_data.py --method rad
python Method/Router/core/cache.py --method rad
python Method/Router/core/train.py --method rad
python Method/Router/core/generate.py --method rad --num-prompts 5
```

Nếu RAD chạy ổn, tiếp tục:

```text
ARGS
↓
CD-FUDGE
↓
CD-Q
↓
GenARM
```

Mỗi method đều theo pipeline:

```text
build_data
↓
cache
↓
train
↓
generate test
↓
generate full
```

---

# 13. File và thư mục output

Training data:

```text
dataset/router_train/<method>/
```

Feature cache:

```text
dataset/router_cache/<method>/
```

Router checkpoint:

```text
Method/Router/checkpoints/
```

Generation results:

```text
results/
```

---

# 14. Lệnh nhanh cho RAD từ đầu đến cuối

```bash
python Method/Router/core/build_data.py --method rad && \
python Method/Router/core/cache.py --method rad && \
python Method/Router/core/train.py --method rad && \
python Method/Router/core/generate.py --method rad --num-prompts 5
```

Không nên nối tất cả lệnh như trên trong lần chạy đầu nếu chưa kiểm tra từng bước.
