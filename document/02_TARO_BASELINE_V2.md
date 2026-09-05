# 02 — Router V2: Faithful TARO Baseline

## Goal

Tạo Router V2 mới gần TARO hơn, độc lập hoàn toàn với V1.

## New code only

Ưu tiên:

```text
Method/RouterV2/
router_v2/
```

Không import rồi monkey-patch V1 classes nếu điều đó có thể thay behavior V1.

## 1. TARO routing

\[
\hat{\alpha}_t=\sigma(g_\theta(h_t)).
\]

\[
z_t^{guided}
=
(1-\hat{\alpha}_t)z_t^{base}
+
\hat{\alpha}_t z_t^{reward}.
\]

## 2. Top-K token-aware input

Router phải giữ token identity.

Với candidate \(i\):

```text
token_id
base_logit
reward_logit
token_embedding
```

Không chỉ feed hai vector scalar đã sort.

## 3. Architecture

Tạo paper-style config, ví dụ:

```text
Top-K token-aware encoder
→ pooled representation
→ MLP hidden 128
→ sigmoid gate
```

Không ép kiến trúc phải giống V1 `40→64→1`.

## 4. Training loss

\[
\mathcal{L}_{NLL}
=
-\sum_t
\log
\pi_{guided}(y_t^\star|x,y_{<t}).
\]

Optional TARO entropy:

\[
H(\hat{\alpha}_t)
=
-\hat{\alpha}_t\log\hat{\alpha}_t
-(1-\hat{\alpha}_t)\log(1-\hat{\alpha}_t).
\]

\[
\mathcal{L}
=
\mathcal{L}_{NLL}
+
\lambda_H
\sum_t H(\hat{\alpha}_t).
\]

Không cần gold alpha.

## 5. Required modes

```text
taro_topk_nll
taro_topk_nll_entropy
```

Nếu full-logit implementation khả thi:

```text
taro_full_logits
```

## 6. Validation

Required tests:
- alpha in \((0,1)\);
- candidate-token identity preserved;
- candidate permutation consistency;
- gold token always scoreable;
- entropy on/off;
- gradients only router;
- checkpoint deterministic;
- V1 outputs unchanged before/after installation of Router V2.

## Acceptance criteria

- Router V2 có TARO-specific implementation riêng;
- V1 hash không đổi;
- TARO equations được unit-test;
- có config + checkpoint schema riêng;
- results ghi label `TARO`, không `TARO-like`.

---

## Prompt cho AI

Tạo Router V2 riêng để implement TARO token-level routing với Top-K logits, token-index embeddings, MLP hidden 128, gold-token NLL và optional Bernoulli entropy. Tuyệt đối không sửa `Method/Router` hoặc `router` V1. Viết unit tests, config/checkpoint schema riêng và regression test chứng minh V1 không đổi.
