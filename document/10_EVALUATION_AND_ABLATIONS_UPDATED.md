# 10 — Evaluation, Pareto, HV/MIP, Preference Matching and Ablations

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

> Lưu ý: HV/MIP/PCS/Preference Regret chủ yếu dùng cho PARM-TARO multi-objective evaluation, không ép dùng cho RAD single-objective nếu không có preference vector thật.

## B. PARM-TARO

Eval trên alpha grid.

Mỗi alpha:
- generate response;
- helpfulness score;
- harmlessness score;
- PPL/quality;
- lambda history;
- preference-matching metrics.

## 1. Pareto front

Một solution non-dominated nếu không có solution khác tốt hơn đồng thời trên mọi objective.

Lưu toàn bộ raw objective vectors để có thể tái dựng Pareto front.

## 2. Hypervolume (HV)

HV cao hơn = Pareto set bao phủ vùng objective tốt hơn/rộng hơn.

PARM paper dùng HV để đánh giá chất lượng learned Pareto front.

Yêu cầu:
- dùng cùng objective normalization cho mọi method;
- dùng cùng reference point cho mọi method trong cùng experiment;
- lưu reference point và normalized Pareto points vào report.

## 3. Mean Inner Product (MIP)

\[
MIP
=
\frac{1}{N}
\sum_j
\alpha_j^\top r_j.
\]

MIP cao = response có reward lớn theo đúng trọng số preference mà user yêu cầu.

Trong đó:
- \(\alpha_j\) là preference vector của user;
- \(r_j\) là reward/objective vector của response.

MIP vừa phản ánh direction preference vừa bị ảnh hưởng bởi magnitude của reward.

## 4. Preference Cosine Similarity (PCS)

Bổ sung metric đo riêng mức độ response đi đúng **hướng trade-off** mà user yêu cầu:

\[
PCS
=
\frac{1}{N}
\sum_j
\cos(\alpha_j,\tilde r_j)
=
\frac{1}{N}
\sum_j
\frac{
\alpha_j^\top \tilde r_j
}{
\|\alpha_j\|_2 \|\tilde r_j\|_2
}.
\]

Trong đó \(\tilde r_j\) là objective/reward vector sau khi normalize từng objective bằng cùng statistics/protocol giữa tất cả methods.

Ý nghĩa:
- PCS cao = trade-off giữa objectives bám sát preference direction của user;
- PCS thấp = model có thể đạt reward cao nhưng phân bổ sai giữa các objectives.

PCS là metric bổ sung, không thay thế MIP.

## 5. Preference Regret

Bổ sung metric phụ để đo khoảng cách giữa response của model và response tốt nhất theo preference hiện tại:

\[
Regret_j
=
\alpha_j^\top r_j^{best}
-
\alpha_j^\top r_j^{model},
\]

\[
Preference\ Regret
=
\frac{1}{N}
\sum_j Regret_j.
\]

Càng thấp càng tốt.

\(r_j^{best}\) phải được định nghĩa rõ và cố định trước khi so sánh, ví dụ:
- best attainable candidate trong cùng candidate/evaluation pool; hoặc
- oracle trên tập Pareto candidates dùng chung cho tất cả methods.

Không được dùng oracle khác nhau cho từng method.

## 6. Required methods

```text
PARM original static
PARM_TARO + faithful TARO
PARM_TARO + V2 no-alpha
PARM_TARO + V2 full-alpha
```

## 7. Architecture ablations

```text
TARO Top-K
+ confidence
+ disagreement
+ position
+ history
+ alpha
Full
```

## 8. Same-average-lambda

\[
\bar{\lambda}
=
\frac1T\sum_t\lambda_t.
\]

Run fixed lambda = mean.

Adaptive phải có benefit ngoài việc đơn giản dùng average guidance lớn hơn.

Report cho mỗi adaptive method:
- adaptive result;
- same-average fixed-\(\lambda\);
- delta HV;
- delta MIP;
- delta PCS;
- delta Preference Regret;
- delta PPL/coherence.

## 9. Alpha controls

Bắt buộc có control để chứng minh router thực sự sử dụng preference vector:

```text
correct alpha
shuffled alpha
fixed/mean alpha
no-alpha router
```

Kỳ vọng:
- correct alpha có MIP/PCS tốt hơn shuffled/fixed alpha;
- Preference Regret thấp hơn;
- nếu không khác biệt đáng kể thì không claim router preference-aware.

## 10. Statistical evaluation

- paired bootstrap CI;
- paired permutation test;
- effect size;
- correction cho multiple comparisons;
- deterministic repeated seeds không tính như independent observations.

Primary statistical comparisons cho PARM-TARO:
```text
V2 full-alpha vs PARM original
V2 full-alpha vs TARO
V2 full-alpha vs V2 no-alpha
V2 full-alpha vs same-average-lambda
```

Primary metrics:
```text
HV
MIP
PCS
Preference Regret
PPL / coherence
```

## 11. Protect original baselines

Trước final report, checksum:
```text
Router V1
PARM original
```

phải giống trước experiment.

## Strong claim target

> PARM-TARO preserves PARM's preference-aware multi-objective guidance while learning token-specific guidance strength, expanding the alignment-quality Pareto frontier and improving both reward-weighted preference satisfaction and preference-direction matching without modifying the frozen base model or original PARM implementation.

## Acceptance criteria

- raw Pareto points saved;
- HV/MIP code validated;
- PCS code validated;
- Preference Regret oracle/reference protocol fixed and documented;
- same-average control;
- alpha controls;
- objective normalization shared across methods;
- V1/PARM unchanged;
- router latency overhead reported;
- paired statistics reported for the main preference-aware comparisons.

---

## Prompt cho AI

Xây evaluation suite so sánh Base/Fixed/V1/TARO/Smart V2 trên RAD và PARM original/PARM_TARO variants trên multi-objective data. Giữ toàn bộ metric hiện có gồm Pareto, HV, MIP, PPL/coherence, latency, same-average-lambda, alpha controls và statistics. Bổ sung Preference Cosine Similarity (PCS) để đo direction matching giữa preference vector alpha và normalized reward vector, cùng Preference Regret để đo khoảng cách tới oracle/best candidate theo cùng alpha. Dùng cùng normalization/reference/oracle protocol cho mọi method, lưu raw points, chạy paired statistics và kiểm tra checksum để xác nhận V1/PARM original không đổi.
