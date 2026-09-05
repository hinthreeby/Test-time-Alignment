# 07 — Create PARM_TARO Without Modifying PARM

## Goal

Workspace có PARM tại:

```text
PARM/
```

Tạo mới song song:

```text
PARM_TARO/
```

## 1. Original PARM — READ ONLY

Không sửa:

```text
PARM/code/
PARM/language-model-arithmetic/
PARM/peft/
```

## 2. Wrapper layout

Recommended:

```text
PARM_TARO/
├── adapters/
│   ├── parm_adapter.py
│   └── token_alignment.py
├── decoding/
│   ├── static_regression.py
│   └── adaptive.py
├── configs/
├── scripts/
├── tests/
├── reports/
└── README.md
```

## 3. Import strategy

Ưu tiên load/import PARM original từ local workspace.

Không copy toàn bộ PARM source vào PARM_TARO.

Nếu PARM uses vendored `peft` / `language-model-arithmetic`, wrapper phải resolve local imports mà không sửa chúng.

## 4. Equation

\[
\log\tilde{\pi}_t(v)
=
\log\pi_{base,t}(v)
+
\lambda_t
\log\pi_{\theta(\alpha),t}(v).
\]

## 5. Router dependency

PARM_TARO load checkpoint/code từ:

```text
router_v2/
```

Không dùng V1 trừ baseline ablation.

## 6. Dataset

Ưu tiên local:

```text
dataset/GenARM/PKU-SafeRLHF-10K/round0/
```

hoặc data đã preprocess đúng PARM nếu tồn tại dưới `PARM/code/data/`.

## Regression tests

- original PARM hashes unchanged;
- lambda=0 -> base;
- constant lambda -> static PARM-equivalent;
- alpha changes PARM guide;
- alpha reaches Router V2.

## Acceptance criteria

- `PARM/` zero modifications;
- all new code in `PARM_TARO/`;
- PARM local dependencies resolve;
- adaptive generation works.

---

## Prompt cho AI

Tạo `PARM_TARO/` song song với `PARM/`. Reuse/import toàn bộ PARM original và vendored dependencies mà không sửa chúng. Load Router V2 từ `router_v2/`, thay static \(1/\beta\) bằng adaptive `lambda_t`, thêm static-regression tests và checksum để chứng minh `PARM/` hoàn toàn không đổi.
