# Controlled Decoding (CD)

Thư mục triển khai các phương pháp Controlled Decoding gồm:

- CD-FUDGE
- CD-Q
- Tokenwise decoding
- Blockwise decoding

## Cấu trúc thư mục

```text
Method/CD_new/
├── models/
│   └── prefix_scorer.py
├── train/
│   ├── train_fudge.py
│   └── train_cdq.py
├── decoding/
│   ├── tokenwise.py
│   └── blockwise.py
├── checkpoints/
│   ├── cd_fudge.pt
│   └── cd_q.pt
└── generate_cd.py
```

Kết quả sinh được lưu theo cấu trúc:

```text
results/
├── fudge/
│   ├── fudge_tokenwise.json
│   └── fudge_blockwise.json
└── cdq/
    ├── cdq_tokenwise.json
    └── cdq_blockwise.json
```

## Môi trường

Kích hoạt môi trường Conda:

```bash
conda activate cd
```

Di chuyển vào thư mục project:

```bash
cd "/home/jupyter-iec2024se10/Reward Decoding"
```

## 1. Test Prefix Scorer

```bash
python Method/CD_new/models/prefix_scorer.py
```

Kết quả mong đợi:

```text
Shape: torch.Size([2])
Values: tensor([...])
```

Các giá trị ban đầu chưa có ý nghĩa vì value head chưa được train.

## 2. Train CD-FUDGE

Cấu hình số prompt trong `Method/CD_new/train/train_fudge.py`:

```python
NUM_PROMPTS = 1000
EPOCHS = 1
```

Chạy train:

```bash
python Method/CD_new/train/train_fudge.py
```

Checkpoint được lưu tại:

```text
Method/CD_new/checkpoints/cd_fudge.pt
```

Pipeline:

```text
Prompt → Base LM rollout → Reward Model chấm reward cuối
→ tạo nhiều prefix → Prefix Scorer dự đoán reward → MSE loss
```

## 3. Sinh kết quả CD-FUDGE Tokenwise

```bash
python Method/CD_new/generate_cd.py --scorer fudge --mode tokenwise --num-prompts 100
```

Output:

```text
results/fudge/fudge_tokenwise.json
```

Tokenwise decoding điều chỉnh từng token theo:

```text
aligned_logit = base_lm_logit + lambda × prefix_value
```

## 4. Sinh kết quả CD-FUDGE Blockwise

```bash
python Method/CD_new/generate_cd.py --scorer fudge --mode blockwise --num-prompts 100
```

Output:

```text
results/fudge/fudge_blockwise.json
```

Blockwise decoding sinh nhiều block ứng viên rồi chọn block có prefix value cao nhất.

## 5. Train CD-Q

Cấu hình số prompt trong `Method/CD_new/train/train_cdq.py`:

```python
NUM_PROMPTS = 1000
EPOCHS = 1
```

Chạy train:

```bash
python Method/CD_new/train/train_cdq.py
```

Checkpoint được lưu tại:

```text
Method/CD_new/checkpoints/cd_q.pt
```

Pipeline:

```text
Prompt → Base LM rollout → tạo cặp prefix hiện tại/prefix kế tiếp
→ Bellman target từ target scorer
→ prefix cuối học final reward
```

## 6. Sinh kết quả CD-Q Tokenwise

```bash
python Method/CD_new/generate_cd.py --scorer cdq --mode tokenwise --num-prompts 100
```

Output:

```text
results/cdq/cdq_tokenwise.json
```

## 7. Sinh kết quả CD-Q Blockwise

```bash
python Method/CD_new/generate_cd.py --scorer cdq --mode blockwise --num-prompts 100
```

Output:

```text
results/cdq/cdq_blockwise.json
```

## Thứ tự chạy khuyến nghị

```text
1. Test PrefixScorer
2. Train CD-FUDGE
3. Test FUDGE tokenwise
4. Test FUDGE blockwise
5. Train CD-Q
6. Test CD-Q tokenwise
7. Test CD-Q blockwise
```

Các lệnh tương ứng:

```bash
python Method/CD_new/models/prefix_scorer.py
python Method/CD_new/train/train_fudge.py
python Method/CD_new/generate_cd.py --scorer fudge --mode tokenwise --num-prompts 100
python Method/CD_new/generate_cd.py --scorer fudge --mode blockwise --num-prompts 100
python Method/CD_new/train/train_cdq.py
python Method/CD_new/generate_cd.py --scorer cdq --mode tokenwise --num-prompts 100
python Method/CD_new/generate_cd.py --scorer cdq --mode blockwise --num-prompts 100
```

## Dataset

Hiện tại prompt train có thể được đọc từ:

```text
dataset/cd_train/hh_prompts.jsonl
```

File này được tạo từ dataset Anthropic HH-RLHF.

Ví dụ cấu hình:

```python
DATASET_PATH = PROJECT_ROOT / "dataset" / "cd_train" / "hh_prompts.jsonl"
```

## Lưu ý về GPU

Với GPU khoảng 10 GB, không nên giữ đồng thời GPT-2 Large, Reward Model và Prefix Scorer trên GPU.

Cách đang dùng:

```text
Base LM lên GPU để rollout → chuyển về CPU
Reward Model lên GPU để chấm reward → chuyển về CPU
Prefix Scorer ở GPU để train
```

Có thể bật cấu hình giảm phân mảnh bộ nhớ CUDA:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Kiểm tra kết quả JSON

Xem nội dung file:

```bash
python -m json.tool results/fudge/fudge_tokenwise.json | head -n 60
```

Hoặc:

```bash
python -m json.tool results/fudge/fudge_blockwise.json | head -n 60
```

## Ý nghĩa các model

```text
Reward Model:
Chấm reward của response hoàn chỉnh.

Prefix Scorer:
Ước lượng reward cuối kỳ vọng từ một prefix chưa hoàn chỉnh.
```

Trong quá trình decoding, Prefix Scorer được dùng để hướng dẫn lựa chọn token hoặc block thay cho việc gọi Reward Model ở từng bước.
