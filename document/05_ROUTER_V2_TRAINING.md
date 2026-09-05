# 05 — Router V2 Training

## Goal

Train TARO/V2 router không cần label \(\lambda_t\).

## 1. Main loss

\[
\mathcal{L}_{NLL}
=
-\sum_t
\log
p_t^\lambda(y_t^\star).
\]

Router học guidance strength gián tiếp qua khả năng tăng xác suất gold token.

## 2. TARO entropy

\[
H(g_t)
=
-g_t\log g_t-(1-g_t)\log(1-g_t)
\]

\[
\mathcal{L}_{entropy}
=
\sum_t H(g_t).
\]

Ablate:

```text
NLL only
NLL + entropy
```

## 3. Smart V2 optional regularizers

Smoothness:

\[
\mathcal{L}_{smooth}
=
\sum_t(\lambda_t-\lambda_{t-1})^2
\]

Guidance cost:

\[
\mathcal{L}_{strength}
=
\sum_t(\lambda_t/\lambda_{\max})^2.
\]

Không bật tất cả ở first run.

## 4. Training sequence

```text
A. TARO baseline
B. V2 confidence/state
C. V2 + history
D. V2 + alpha on PARM
```

## 5. Required diagnostics

```text
train/val NLL
token accuracy
lambda mean/std/min/max
p05/p25/p50/p75/p95
fraction near 0/max
lambda by position
lambda vs base entropy
lambda vs JS
lambda by alpha bucket
```

## 6. Same-average-lambda control

Nếu adaptive router mean:

\[
\bar{\lambda}=c
\]

phải chạy fixed:

\[
\lambda=c.
\]

Nếu adaptive không tốt hơn same-mean fixed, chưa chứng minh được token-level adaptation có giá trị.

## 7. Frozen audit

Router training không được update:
- base LM;
- reward model;
- PARM/PBLoRA.

Hash/checksum before/after.

## Acceptance criteria

- V1 untouched;
- validation NLL improves;
- no constant collapse;
- same-average fixed control available;
- frozen model hashes unchanged.

---

## Prompt cho AI

Implement training riêng cho Router V2 với gold-token NLL, optional TARO entropy, optional smoothness/strength regularizers và đầy đủ lambda diagnostics. Train theo thứ tự TARO→V2 state→history→alpha. Thêm same-average-lambda baseline và checksum audit để đảm bảo base/reward/PARM không bị update.
