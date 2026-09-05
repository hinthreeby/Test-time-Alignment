# 03 — Router V2 Feature Cache

## Goal

Tạo cache mới cho Router V2 mà không đụng V1 cache.

## Existing V1 cache — READ ONLY

```text
dataset/router_cache/rad/
```

## New V2 cache

```text
dataset/router_v2_cache/
├── rad/
│   ├── train/
│   └── validation/
└── parm/
    ├── train/
    └── validation/
```

Nếu cần processed training metadata riêng:

```text
dataset/router_v2_train/
```

## Raw token record

```python
{
  "sample_id": str,
  "position": int,
  "candidate_ids": LongTensor[K],
  "base_logits": FloatTensor[K],
  "guide_logits": FloatTensor[K],
  "gold_token_id": int,
  "gold_index": int,
  "preference_vector": Optional[FloatTensor],
}
```

## Derived features

\[
H_b=-\sum_i p_i^b\log p_i^b
\]

\[
H_g=-\sum_i p_i^g\log p_i^g
\]

\[
M_b=z^b_{(1)}-z^b_{(2)}
\]

\[
M_g=z^g_{(1)}-z^g_{(2)}
\]

\[
JS(p^b,p^g)
\]

Thêm:
- top1 agreement;
- Top-K overlap;
- rank correlation;
- std/range;
- normalized position.

## Data reuse

RAD source:
```text
dataset/RAD_train/router_amazon_polarity/
```

PARM candidate source:
```text
dataset/GenARM/PKU-SafeRLHF-10K/round0/
```

Nhưng PARM cache chỉ tạo sau khi inventory xác nhận exact schema/checkpoint.

## History

`previous_lambda` phụ thuộc router state, nên không bake cố định vào static cache.

History được compute online hoặc sequentially trong collator/training loop.

## Acceptance criteria

- V1 cache unchanged;
- V2 cache isolated;
- raw values reconstruct features;
- no future-token leakage;
- alpha/objective metadata correct.

---

## Prompt cho AI

Tạo cache mới dưới `dataset/router_v2_cache/` và optional metadata dưới `dataset/router_v2_train/`; tuyệt đối không sửa `dataset/router_cache/rad/`. Reuse `dataset/RAD_train/router_amazon_polarity/` cho RAD và chỉ dùng PKU-SafeRLHF local sau khi schema được inventory. Lưu raw logits/token IDs/gold targets cùng derived confidence-disagreement features và tests chống leakage.
