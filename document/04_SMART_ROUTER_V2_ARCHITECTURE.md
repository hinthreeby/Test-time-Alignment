# 04 — Smart Router V2 Architecture

## Goal

Sau TARO baseline, tạo router "thông minh hơn" bằng feature/state conditioning, không chỉ tăng số tham số.

## 1. Token-aware candidate encoder

Candidate \(i\):

\[
e_{t,i}^{base}
=
\phi_b([z^b_{t,i};E(token_i)])
\]

\[
e_{t,i}^{guide}
=
\phi_g([z^g_{t,i};E(token_i)])
\]

Pool bằng lightweight attention hoặc mean+max.

## 2. Confidence/disagreement

\[
c_t=[
H_b,H_g,
M_b,M_g,
JS,
Top1Agree,
Overlap_K,
Std_b,Std_g,
Range_b,Range_g
].
\]

## 3. Position

\[
p_t = t/T_{max}
\]

hoặc embedding vị trí router riêng.

## 4. History

Small GRU only on router-level state:

\[
q_t=
GRU(
[\lambda_{t-1},
score_{t-1}^{selected},
H_{b,t-1},
JS_{t-1}],
q_{t-1}
).
\]

Không feed backbone hidden state ở default V2.

## 5. Preference encoder

Cho PARM:

\[
e_\alpha=MLP_\alpha(\alpha).
\]

RAD single-objective có thể dùng constant vector `[1.0]`.

## 6. Fusion

\[
u_t=
[e_t^{cand};c_t;p_t;q_t;e_\alpha].
\]

Ví dụ:

```text
LayerNorm
Linear → 128
GELU/Tanh
Linear → 64
GELU/Tanh
Linear → 1
Sigmoid
```

\[
g_t=\sigma(f_\phi(u_t)).
\]

Dùng universal strength:

\[
\lambda_t=\lambda_{\max}g_t.
\]

## 7. Variants

Bắt buộc để ablation:

```text
taro_topk
v2_topk_confidence
v2_topk_confidence_position
v2_topk_state_history
v2_topk_state_history_alpha
```

## 8. Không thêm trước khi ablation xong

- transformer router lớn;
- backbone hidden states;
- lookahead tree search;
- extra external LLM judge inside token loop.

## Acceptance criteria

- architecture mới nằm hoàn toàn trong Router V2;
- mỗi feature group toggle bằng config;
- synthetic tests chứng minh feature group có thể ảnh hưởng \(\lambda_t\);
- alpha-shuffle / history perturbation test có thể chạy.

---

## Prompt cho AI

Mở rộng TARO Router V2 bằng token-aware Top-K encoder, confidence/disagreement, position, small router-state GRU và preference encoder alpha. Dùng universal `lambda_t`, không backbone hidden states. Mỗi feature group phải bật/tắt bằng config để ablation và không thay bất kỳ code V1 nào.
