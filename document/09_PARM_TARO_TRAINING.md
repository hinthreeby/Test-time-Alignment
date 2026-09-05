# 09 — Train Preference-Aware Router for PARM-TARO

## Goal

Train một Router V2 checkpoint dùng cho toàn preference simplex.

## 1. Alpha

\[
\alpha=(\alpha_{help},\alpha_{safe}),
\quad
\alpha_{help}+\alpha_{safe}=1.
\]

Sample alpha trong training.

## 2. Frozen components

Freeze:
- base LLM;
- PARM base weights;
- PBLoRA/PARM checkpoint.

Train:
- Router V2 only.

## 3. Token-level training

At each teacher-forced step:
1. base model distribution;
2. PARM distribution conditioned on alpha;
3. Router V2 receives current state + alpha;
4. router outputs \(\lambda_t\);
5. form guided distribution;
6. compute preference/gold-token loss;
7. backprop only router.

## 4. Conflicting preferences

Không tự chọn một "gold response" khi helpfulness và harmlessness disagree mà không có rule.

Ưu tiên formulation giữ pairwise multi-objective preference structure.

Nếu scalarization:

\[
q_\alpha=
\sum_i\alpha_i s_i
\]

thì:
- rule phải config;
- tie/near-tie phải log;
- phải có ablation.

## 5. Required comparison

```text
PARM static
PARM + TARO router
PARM + Smart V2 no-alpha
PARM + Smart V2 alpha
```

## 6. Alpha controls

Bắt buộc:
```text
correct alpha
shuffled alpha
constant alpha
no alpha
```

Nếu shuffled alpha không làm giảm/change behavior, router có thể đang bỏ qua preference input.

## Acceptance criteria

- one checkpoint works for many alpha;
- no retraining per user preference;
- router responds to alpha;
- PARM hashes unchanged;
- no test tuning.

---

## Prompt cho AI

Train Router V2 cho PARM_TARO với frozen base/PARM/PBLoRA, sample preference vector alpha trong training và optimize token-level guided loss. So sánh static PARM, TARO router, V2 no-alpha và V2 alpha; thêm shuffled/constant-alpha controls để chứng minh router thật sự dùng preference. Không update hoặc sửa PARM original.
