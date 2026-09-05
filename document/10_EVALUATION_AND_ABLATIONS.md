# 10 — Evaluation, Pareto, HV/MIP and Ablations

## A. RAD

So sánh:
```text
Base
Fixed
V1
TARO V2
Smart V2
```

Primary:
- alignment;
- PPL;
- coherence;
- latency;
- Pareto frontier.

## B. PARM-TARO

Eval trên alpha grid.

Mỗi alpha:
- generate response;
- helpfulness score;
- harmlessness score;
- PPL/quality;
- lambda history.

## 1. Pareto front

Một solution non-dominated nếu không có solution khác tốt hơn đồng thời trên mọi objective.

## 2. Hypervolume

HV cao hơn = Pareto set bao phủ vùng objective tốt hơn/rộng hơn.

PARM paper dùng HV để đánh giá chất lượng learned Pareto front.

## 3. MIP

\[
MIP
=
\frac{1}{N}
\sum_j
\alpha_j^\top r_j.
\]

MIP cao = response match requested preference vector tốt hơn.

## 4. Required methods

```text
PARM original static
PARM_TARO + faithful TARO
PARM_TARO + V2 no-alpha
PARM_TARO + V2 full-alpha
```

## 5. Architecture ablations

```text
TARO Top-K
+ confidence
+ disagreement
+ position
+ history
+ alpha
Full
```

## 6. Same-average-lambda

\[
\bar{\lambda}
=
\frac1T\sum_t\lambda_t.
\]

Run fixed lambda = mean.

Adaptive phải có benefit ngoài việc đơn giản dùng average guidance lớn hơn.

## 7. Statistical evaluation

- paired bootstrap CI;
- paired permutation test;
- effect size;
- correction cho multiple comparisons;
- deterministic repeated seeds không tính như independent observations.

## 8. Protect original baselines

Trước final report, checksum:
```text
Router V1
PARM original
```

phải giống trước experiment.

## Strong claim target

> PARM-TARO preserves PARM's preference-aware multi-objective guidance while learning token-specific guidance strength, expanding the alignment-quality Pareto frontier and improving preference matching without modifying the frozen base model or original PARM implementation.

## Acceptance criteria

- raw Pareto points saved;
- HV/MIP code validated;
- same-average control;
- alpha controls;
- V1/PARM unchanged;
- router latency overhead reported.

---

## Prompt cho AI

Xây evaluation suite so sánh Base/Fixed/V1/TARO/Smart V2 trên RAD và PARM original/PARM_TARO variants trên multi-objective data. Tính Pareto, HV, MIP, PPL/coherence, latency, same-average-lambda và alpha controls; chạy paired statistics và kiểm tra checksum để xác nhận V1/PARM original không đổi.
