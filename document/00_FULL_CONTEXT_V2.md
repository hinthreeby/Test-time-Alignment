# 00 — Full Context V2: Router V1 → TARO Router V2 → PARM-TARO

## 1. Workspace

Project root:

```text
/home/jupyter-iec2024se10/Reward Decoding
```

Router V1 hiện nằm tại:

```text
router/
```

với các phần:

```text
router/
├── docs/
├── evaluation/
├── rad/
├── reports/
├── scripts/data/
└── tests/
```

Đây là baseline khoa học và **READ ONLY**.

PARM gốc nằm tại:

```text
PARM/
```

và cũng **READ ONLY**.

## 2. Router V1 hiện tại

V1:

\[
h_t=
[\operatorname{Norm}(\ell_t);
 \operatorname{Norm}(r_t)]
\in\mathbb{R}^{40}
\]

\[
40\rightarrow64\rightarrow1
\]

\[
\beta_t=\beta_{\max}\sigma(g_\theta(h_t)).
\]

Kết quả V1 đang nằm dưới:

```text
router/evaluation/
├── pilot/
├── full_32tokens/
└── report_run_32tokens/
```

Không sửa hoặc overwrite các output này.

## 3. Router V2 mới

Tạo:

```text
router_v2/
```

Không đặt V2 vào `router/`.

TARO baseline V2 dùng:

\[
\hat{\alpha}_t=\sigma(g_\theta(h_t))
\]

\[
z_t^{guided}
=
(1-\hat{\alpha}_t)z_t^{base}
+
\hat{\alpha}_t z_t^{reward}.
\]

Sau đó Smart V2 mở rộng:

\[
\lambda_t=
f_\phi(
z_t^{base},
z_t^{guide},
\text{confidence},
\text{disagreement},
\text{history},
\alpha
).
\]

## 4. Data hiện có

RAD/router data:

```text
dataset/RAD_train/amazon_polarity/
dataset/RAD_train/router_amazon_polarity/
dataset/router_cache/rad/
dataset/router_train/rad/
dataset/rad_benchmark/
```

Multi-objective data candidate:

```text
dataset/GenARM/PKU-SafeRLHF-10K/round0/
dataset/GenARM/full-hh-rlhf/data/
```

Vì PKU-SafeRLHF đã có local trong workspace, **không download lại mặc định**.

## 5. Models hiện có

Có local checkpoints cho:
- GPT-2;
- RAD sentiment RM;
- GenARM;
- Llama-2-7B;
- Tulu-2-7B;
- autoregressive RM;
- helpfulness evaluator;
- sentiment evaluator;
- toxicity evaluator.

AI phải inventory exact paths trước khi hard-code.

## 6. PARM original

PARM nằm tại:

```text
PARM/
├── code/data/
├── code/evaluation/
├── code/training/
├── language-model-arithmetic/
└── peft/
```

Không sửa.

Tạo mới:

```text
PARM_TARO/
```

## 7. PARM-TARO equation

PARM static:

\[
\tilde{\pi}_t(v)
\propto
\pi_{base,t}(v)
[
\pi_{\theta(\alpha),t}(v)
]^{1/\beta}.
\]

Adaptive wrapper:

\[
\boxed{
\log\tilde{\pi}_t(v)
=
\log\pi_{base,t}(v)
+
\lambda_t
\log\pi_{\theta(\alpha),t}(v)
}
\]

với:

\[
\lambda_t=f_\phi(state_t,\alpha).
\]

PARM quyết định **reward direction/trade-off** theo preference vector.

Router V2 quyết định **guidance strength theo token**.

## 8. Boundary

READ ONLY:

```text
router/
PARM/
Method/RAD/
dataset/router_cache/rad/
dataset/router_train/rad/
router/evaluation/
```

NEW:

```text
router_v2/
dataset/router_v2_cache/
dataset/router_v2_train/
PARM_TARO/
results/router_v2/
results/parm_taro/
```

## 9. Success

- V1 outputs/hashes không đổi.
- PARM source/hashes không đổi.
- TARO baseline V2 chạy riêng.
- Smart V2 tốt hơn V1/TARO về Pareto trade-off.
- PARM-TARO constant lambda reproduce PARM static.
- full preference-aware V2 cải thiện HV/MIP/Pareto hoặc quality trade-off.

---

## Prompt cho AI

Làm việc tại `/home/jupyter-iec2024se10/Reward Decoding`. Xem `router/` là V1 read-only và `PARM/` là author code read-only. Tạo `router_v2/`, `dataset/router_v2_cache/`, `PARM_TARO/`, `results/router_v2/`, `results/parm_taro/`. Reuse data/model local hiện có và không overwrite bất kỳ artifact V1/PARM nào.
