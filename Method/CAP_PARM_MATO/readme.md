# DynaCAP-PARM Implementation Specification

## Dynamic Objective Balancing and Selective Reward Guidance for Multi-Objective Test-Time Alignment

Tài liệu này mô tả phiên bản mới của CAP-PARM trong đó nhánh PRO-style prompt router được thay bằng MATO-style dynamic objective balancing.

Backbone:

- PARM cung cấp một Preference-Aware Autoregressive Reward Model.
- MATO-inspired controller điều chỉnh objective weights alpha_t trong lúc sinh.
- TARo-inspired controller điều chỉnh mức PARM can thiệp w_t tại từng token.

Đây là đặc tả để Codex audit repository, triển khai theo từng phase, viết test và tạo evaluation pipeline. Không triển khai toàn bộ hệ thống trước khi các oracle test vượt stage gate.

---

## 0. Quy tắc dành cho Codex

Trước khi sửa code:

1. Đọc toàn bộ README.
2. Chạy rg --files để hiểu repository.
3. Xác định model loading, PBLoRA conditioning, fusion, generation, scoring và metrics hiện tại.
4. Tái sử dụng module có sẵn thay vì tạo pipeline trùng lặp.
5. Giữ nguyên checkpoint PBLoRA/PARM đã được xác nhận.
6. Reproduce fixed PARM trước mọi controller experiment.
7. Chạy oracle dynamic-alpha trước khi train objective tracker.
8. Chạy oracle-w trước khi train token router.
9. Không dùng test set để chọn hyperparameter.
10. Không dùng chính guide signal làm evaluator duy nhất.
11. Mọi experiment phải lưu config, seed, commit hash và output.
12. Không xóa kết quả cũ; mỗi experiment có run directory riêng.

Thứ tự bắt buộc:

1. Reproduce fixed PARM.
2. Xác minh objective reward signals.
3. Oracle MATO-style dynamic alpha.
4. Oracle token-level w.
5. Train objective tracker nếu oracle-alpha pass.
6. Train token router nếu oracle-w pass.
7. Ghép dynamic alpha_t và w_t.

---

## 1. Research question

PARM nhận một preference vector cố định:

\[
\alpha^{user}=(\alpha_1,\ldots,\alpha_K),\qquad
\sum_k\alpha_k=1.
\]

PARM giữ alpha cố định trong toàn bộ response. Điều này không quan sát được objective nào đang được đáp ứng tốt hoặc bị bỏ quên khi response đang hình thành.

DynaCAP-PARM đặt hai câu hỏi:

1. Có thể cập nhật objective weights dựa trên accumulated per-objective reward của prefix hay không?
2. Có thể chỉ kích hoạt PARM tại những token mà guidance đem lại expected utility gain hay không?

Hai biến điều khiển:

| Biến | Phạm vi | Vai trò |
|---|---|---|
| alpha_t | Prefix/token | Objective nào đang cần được ưu tiên |
| w_t | Token | PARM nên can thiệp mạnh đến đâu |

Phân biệt:

- alpha_t trộn các objective bên trong preference condition.
- w_t trộn Base LLM và PARM.

---

## 2. Phạm vi

### Trong phạm vi

- Frozen base LLM.
- Frozen PARM sau khi backbone được huấn luyện.
- Dynamic objective tracking.
- MATO-inspired exponentiated weight update.
- TARo-inspired token-level guidance controller.
- Quality-cost constrained decoding.
- Independent evaluation.

### Chưa làm trong phiên bản đầu

- RL fine-tuning base LLM.
- Thay đổi kiến trúc PBLoRA.
- FTRL nhiều vòng tại mỗi token như reproduction đầy đủ của MATO.
- Tree search hoặc beam search lớn.
- Joint end-to-end training toàn hệ thống ngay từ đầu.
- External evaluator reward model tại inference thật.

---

## 3. Trạng thái hiện tại

Historical results:

| Experiment | Kết quả |
|---|---:|
| Best fixed MIP | 0.64688 |
| Oracle per-case MIP | 0.71949 |
| Relative headroom | 11.23% |
| Best per-alpha MIP | 0.65379 |
| Prompt router MIP | 0.61750 |
| Prompt-router Spearman | gần 0 |
| Prompt-router pairwise accuracy | gần 0.50 |
| Token states đã phân tích | 250 |
| Tied states | 225/250 |
| States cùng next token | 74.4% |
| Actionable states | 64/250 |

Kết luận:

- Không tiếp tục PRO-style prompt-only router làm core.
- Optimal behavior khó suy ra chỉ từ prompt.
- MATO-style feedback dùng trạng thái response đang sinh nên có cơ sở tốt hơn.
- Token-level signal vẫn thưa, cần tie-aware training và abstention.

---

## 4. Ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| x | Prompt |
| y_<t | Prefix trước bước t |
| v | Candidate token |
| K | Số objective |
| alpha_user | Preference ban đầu của người dùng |
| alpha_t | Preference động tại bước t |
| a_t(v) | Base score cho token v |
| b_t(v, alpha_t) | PARM score theo alpha_t |
| r_k,t | Estimated incremental reward objective k |
| R_k,<t | Accumulated reward objective k |
| w_t | Mức guidance của PARM |
| q(x,y) | Final independent objective vector |

Simplex:

\[
\Delta^{K-1}=
\left\{
\alpha\in\mathbb R^K:
\alpha_k\ge0,\ \sum_k\alpha_k=1
\right\}.
\]

---

## 5. PARM backbone

PARM tạo phân phối:

\[
\pi_{\theta(\alpha)}(y_t\mid x,y_{<t}).
\]

Autoregressive reward:

\[
r(x,y,\alpha)=
\sum_t
\log\pi_{\theta(\alpha)}(y_t\mid x,y_{<t}).
\]

PBLoRA:

\[
\theta(\alpha)=\theta_0+sBW(\alpha)A.
\]

PARM gốc:

\[
\widetilde\pi(y_t)
\propto
\pi_{base}(y_t)
\pi_{\theta(\alpha)}(y_t)^{1/\beta}.
\]

Biểu thức trên là phép nhân phân phối. Dạng log:

\[
\log\widetilde\pi(y_t)
=
\log\pi_{base}(y_t)
+\frac{1}{\beta}\log\pi_{\theta(\alpha)}(y_t)
-\log Z.
\]

Project giữ thêm canonical convex fusion:

\[
z_t(v)=
(1-w_t)a_t(v)+w_tb_t(v\mid\alpha_t).
\]

Fusion modes:

- convex: mặc định của project.
- parm_product: baseline đúng theo công thức PARM.

Không trộn hai mode trong cùng result nếu không ghi rõ.

---

## 6. MATO-style dynamic objective balancing

### 6.1. Accumulated objective reward

Objective tracker ước lượng reward vector ở mỗi prefix:

\[
\hat{\mathbf r}_t=
[\hat r_{1,t},\ldots,\hat r_{K,t}].
\]

Accumulated reward:

\[
\hat R_{k,<t}
=
\sum_{j=1}^{t-1}\hat r_{k,j}.
\]

Cần normalize để các objective có scale tương đương:

\[
\widetilde R_{k,<t}
=
\frac{\hat R_{k,<t}-\mu_k(t)}
{\sigma_k(t)+\epsilon}.
\]

Normalization statistics chỉ fit trên train/validation.

### 6.2. Preference update

Dynamic alpha được định nghĩa bằng entropic mirror descent:

\[
\widetilde\alpha_t
=
\arg\min_{\alpha\in\Delta^{K-1}}
\left[
\alpha^\top\widetilde{\mathbf R}_{<t}
+\tau D_{KL}(\alpha\Vert\alpha^{user})
\right].
\]

Closed-form:

\[
\widetilde\alpha_{k,t}
\propto
\alpha^{user}_k
\exp\left(
-\frac{\widetilde R_{k,<t}}{\tau}
\right).
\]

Objective có accumulated reward thấp sẽ nhận trọng số cao hơn.

Đây là MATO-inspired update, nhưng alpha_user vẫn là prior để không làm mất preference của người dùng.

### 6.3. Stabilization

Để tránh alpha dao động mạnh:

\[
\alpha_t=
(1-\eta)\alpha_{t-1}
+\eta\widetilde\alpha_t.
\]

Sau đó project về simplex và trust region:

\[
D_{KL}(\alpha_t\Vert\alpha^{user})\le\rho.
\]

Có thể chỉ update mỗi H token:

\[
t\in\{H,2H,3H,\ldots\}.
\]

Config ban đầu:

- update_interval H thuộc {4, 8, 16}.
- smoothing eta thuộc {0.1, 0.25, 0.5}.
- temperature tau thuộc {0.1, 0.5, 1.0}.
- KL budget rho chọn bằng validation.

### 6.4. Không thay đổi evaluation preference

Response luôn được đánh giá bằng alpha_user:

\[
U(y)=
(\alpha^{user})^\top q(x,y).
\]

Không dùng alpha_t cuối cùng để tính MIP hoặc utility. Alpha_t là biến điều khiển nội bộ, không phải preference mới của người dùng.

---

## 7. Vấn đề quan trọng: lấy per-objective reward ở đâu?

PARM thống nhất không tự động xuất vector:

\[
[R_1,\ldots,R_K].
\]

PARM chỉ xuất phân phối đã condition theo một alpha. Vì vậy phải kiểm tra ba phương án.

### Phương án A: One-hot PARM probes

Với objective k, đặt:

\[
\alpha=e_k.
\]

Sau đó lấy:

\[
\hat r_{k,t}(v)
=
\log
\pi_{\theta(e_k)}
(v\mid x,y_{<t}).
\]

Ưu điểm:

- Không cần train objective tracker mới.
- Gần với behavior đã học của PARM.
- Phù hợp để làm oracle feasibility.

Nhược điểm:

- Cần K PARM configurations/passes mỗi update.
- Không phù hợp inference hiệu quả nếu K lớn.
- PBLoRA switching và KV cache phải được kiểm tra cẩn thận.

Chỉ dùng phương án này cho oracle hoặc small-scale experiment.

### Phương án B: Lightweight prefix objective tracker

Train một model nhẹ:

\[
g_\omega(x,y_{<t})
\rightarrow
[\hat r_{1,t},\ldots,\hat r_{K,t}].
\]

Input có thể gồm:

- Base hidden state.
- Base top-k log probabilities.
- Prefix pooled representation.
- Token position.
- Alpha_user.

Output:

- K incremental rewards; hoặc
- K remaining-return/value estimates.

Ưu điểm:

- Một forward nhẹ.
- Có thể dùng trước PARM call.
- Phù hợp với joint compute-saving controller.

Nhược điểm:

- Cần label đáng tin cậy.
- Dễ reward hacking.
- Phải kiểm tra calibration.

Đây là phương án production nếu oracle dynamic-alpha pass.

### Phương án C: External objective RMs

Dùng K evaluator RMs để chấm prefix/rollout.

Chỉ dùng cho:

- Offline label generation.
- Oracle analysis.
- Final response evaluation.

Không dùng tại inference thật vì chi phí cao và làm sai claim test-time efficiency.

### Quyết định

1. Oracle dùng A hoặc C.
2. Nếu oracle không có gain: dừng dynamic-alpha.
3. Nếu oracle có gain: train B.
4. Chỉ tuyên bố efficiency khi inference dùng B hoặc tín hiệu rẻ tương đương.

---

## 8. TARo-style token guidance

Token controller dự đoán:

\[
w_t=g_\psi(f_t).
\]

Binary version:

\[
w_t\in\{0,w_{fixed}\}.
\]

Sau khi binary gate thành công:

\[
w_t\in\{0,0.5,1\}
\quad\text{hoặc}\quad
w_t\in[0,1].
\]

Feature compute-saving chỉ được dùng thông tin trước PARM:

- Base entropy.
- Base top-1/top-2 margin.
- Base top-k log probabilities.
- Position.
- alpha_t.
- Objective deficit vector.
- Previous gate state.
- Objective-tracker uncertainty.

Nếu dùng cả Base và PARM logits như TARo gốc, router có thể cải thiện quality nhưng không tiết kiệm PARM forward.

Phải tách hai setting:

- quality router: có thể nhìn cả hai logits.
- selective router: chỉ nhìn base-side features trước PARM.

---

## 9. Joint decoding

Tại step t:

1. Base LLM tạo a_t.
2. Objective tracker cập nhật accumulated reward.
3. Dynamic controller tính alpha_t.
4. Token router tính w_t.
5. Nếu w_t = 0, sample từ base.
6. Nếu w_t > 0, PARM tạo b_t theo alpha_t.
7. Fuse và sample token.

Formula:

\[
\alpha_t
=
\operatorname{DynamicBalance}
(\alpha^{user},\widetilde{\mathbf R}_{<t}),
\]

\[
w_t
=
g_\psi
(f_t,\alpha_t,\widetilde{\mathbf R}_{<t}),
\]

\[
z_t(v)=
(1-w_t)a_t(v)
+w_tb_t(v\mid\alpha_t).
\]

---

## 10. Unified objective

\[
\max
\mathbb E
\left[
(\alpha^{user})^\top q(x,y)
+\mu\min_k\bar R_k(x,y)
-\lambda C(y)
-\tau_\alpha
\frac{1}{T}\sum_t
D_{KL}(\alpha_t\Vert\alpha^{user})
-\xi
\frac{1}{T}\sum_t
\|\alpha_t-\alpha_{t-1}\|_2^2
\right].
\]

Trong đó:

- Utility term giữ đúng preference người dùng.
- Worst-objective term ngăn một objective bị bỏ quên.
- C(y) là PARM compute cost.
- KL term giới hạn deviation.
- Stability term ngăn oscillation.

Compute:

\[
C(y)=
\frac{1}{T}
\sum_t\mathbb I[w_t>0].
\]

---

## 11. Kiến trúc code đề xuất

~~~text
dynacap_parm/
├── configs/
│   ├── reproduce_parm.yaml
│   ├── oracle_dynamic_alpha.yaml
│   ├── oracle_w.yaml
│   ├── train_objective_tracker.yaml
│   ├── train_token_router.yaml
│   └── evaluate_joint.yaml
├── src/
│   ├── models/
│   │   ├── base_adapter.py
│   │   ├── parm_adapter.py
│   │   ├── objective_tracker.py
│   │   ├── token_router.py
│   │   └── calibration.py
│   ├── controllers/
│   │   ├── dynamic_preference.py
│   │   ├── trust_region.py
│   │   └── budget_controller.py
│   ├── decoding/
│   │   ├── fusion.py
│   │   ├── dynamic_decoder.py
│   │   └── cache_manager.py
│   ├── oracle/
│   │   ├── objective_probe.py
│   │   ├── dynamic_alpha_search.py
│   │   ├── intervention_search.py
│   │   └── rollout.py
│   ├── training/
│   │   ├── train_objective_tracker.py
│   │   ├── train_token_router.py
│   │   └── losses.py
│   ├── evaluation/
│   │   ├── score_outputs.py
│   │   ├── metrics.py
│   │   ├── trajectory_metrics.py
│   │   └── bootstrap.py
│   └── data/
│       ├── schemas.py
│       ├── jsonl.py
│       └── splits.py
├── scripts/
│   ├── reproduce_fixed_parm.py
│   ├── build_dynamic_alpha_oracle.py
│   ├── build_w_oracle.py
│   ├── build_tracker_dataset.py
│   ├── train_objective_tracker.py
│   ├── train_token_router.py
│   ├── generate.py
│   ├── score.py
│   └── evaluate.py
└── tests/
    ├── test_dynamic_preference.py
    ├── test_objective_tracker.py
    ├── test_fusion.py
    ├── test_lazy_parm.py
    ├── test_cache_switching.py
    ├── test_schemas.py
    └── test_metrics.py
~~~

Nếu repository đã có module tương đương, tích hợp vào cấu trúc hiện tại thay vì tạo tree mới.

---

## 12. Interfaces

### Base adapter

~~~python
class BaseModelAdapter:
    def prefill(self, input_ids, attention_mask=None):
        """Return next scores, optional features, and KV cache."""

    def step(self, token_id, past_key_values):
        """Advance one token."""
~~~

### PARM adapter

~~~python
class PARMAdapter:
    def set_preference(self, alpha):
        """Validate simplex and configure PBLoRA."""

    def prefill(self, input_ids, alpha, attention_mask=None):
        """Return next scores and KV cache."""

    def step(self, token_id, alpha, past_key_values):
        """Advance one token under alpha."""

    def probe_objectives(self, input_ids, objective_vectors):
        """Oracle-only one-hot objective probes."""
~~~

### Objective tracker

~~~python
class ObjectiveTracker:
    def forward(self, base_features, alpha_user, step_index):
        """
        Return incremental reward vector, uncertainty,
        and optional remaining-return estimate.
        """
~~~

### Dynamic preference controller

~~~python
class DynamicPreferenceController:
    def update(
        self,
        alpha_user,
        alpha_previous,
        accumulated_rewards,
        uncertainty=None,
    ):
        """Return stabilized alpha_t and diagnostic values."""
~~~

### Token router

~~~python
class TokenRouter:
    def forward(
        self,
        base_features,
        alpha_t,
        objective_deficits,
        step_index,
    ):
        """Return gate probability, w_t, and uncertainty."""
~~~

---

## 13. Dynamic preference algorithm

~~~python
def update_dynamic_alpha(
    alpha_user,
    alpha_previous,
    cumulative_rewards,
    temperature,
    smoothing,
    kl_budget,
    eps=1e-8,
):
    alpha_user = validate_simplex(alpha_user)
    rewards = normalize_objective_rewards(cumulative_rewards)

    logits = log(alpha_user + eps) - rewards / temperature
    alpha_candidate = softmax(logits)

    alpha_smoothed = (
        (1.0 - smoothing) * alpha_previous
        + smoothing * alpha_candidate
    )

    alpha_projected = project_to_kl_ball(
        alpha_smoothed,
        center=alpha_user,
        radius=kl_budget,
    )

    return normalize_simplex(alpha_projected)
~~~

Required diagnostics:

- alpha_candidate.
- alpha_t.
- KL to alpha_user.
- L1 shift.
- Dominant objective.
- Objective deficit vector.
- Whether update was clipped.

---

## 14. Joint inference pseudocode

~~~python
def generate_dynacap_parm(prompt, alpha_user, cfg):
    alpha_user = validate_simplex(alpha_user)
    alpha_t = alpha_user.clone()

    cumulative_rewards = zeros_like(alpha_user)
    output_tokens = []
    trace = []

    base_state = base_model.prefill(prompt)
    parm_state = None
    pending_tokens = []

    for step in range(cfg.generation.max_new_tokens):
        base_scores = base_state.next_scores
        base_features = build_base_features(base_state, step)

        reward_vector, reward_uncertainty = objective_tracker(
            base_features,
            alpha_user,
            step,
        )
        cumulative_rewards += reward_vector

        if step > 0 and step % cfg.dynamic_alpha.update_interval == 0:
            alpha_t = dynamic_controller.update(
                alpha_user=alpha_user,
                alpha_previous=alpha_t,
                accumulated_rewards=cumulative_rewards,
                uncertainty=reward_uncertainty,
            )

        objective_deficits = compute_deficits(cumulative_rewards)

        gate_prob, w_t, gate_uncertainty = token_router(
            base_features,
            alpha_t,
            objective_deficits,
            step,
        )

        if w_t > 0:
            parm_state = synchronize_parm_cache(
                prompt=prompt,
                generated_tokens=output_tokens,
                pending_tokens=pending_tokens,
                state=parm_state,
                alpha=alpha_t,
            )
            parm_scores = parm_state.next_scores
            final_scores = fuse_scores(
                base_scores,
                parm_scores,
                w_t,
                mode=cfg.fusion.mode,
            )
            pending_tokens = []
            parm_called = True
        else:
            final_scores = base_scores
            parm_called = False

        token = sample(final_scores, cfg.generation)
        output_tokens.append(token)
        pending_tokens.append(token)
        base_state = base_model.step(token, base_state.cache)

        trace.append({
            "step": step,
            "alpha_t": alpha_t.tolist(),
            "reward_vector": reward_vector.tolist(),
            "cumulative_rewards": cumulative_rewards.tolist(),
            "w_t": float(w_t),
            "gate_prob": float(gate_prob),
            "parm_called": parm_called,
            "token_id": int(token),
        })

        if token == eos_token_id:
            break

    return output_tokens, trace
~~~

---

## 15. KV-cache constraints

### Dynamic alpha

PBLoRA parameters phụ thuộc alpha. Nếu alpha thay đổi giữa các token, cached hidden states của PARM có thể đã được tạo dưới alpha cũ.

Không được giả định KV cache còn chính xác khi đổi alpha.

Phải kiểm tra ba strategy:

1. Full recompute khi alpha thay đổi.
2. Update alpha theo block và reset/replay PARM cache tại block boundary.
3. Nếu implementation PBLoRA cho phép, verify cache invariance bằng numerical test.

Không tuyên bố dynamic alpha hiệu quả nếu phải full-recompute toàn prefix mỗi token.

Khuyến nghị phiên bản đầu:

- Update alpha theo block H token.
- Khi alpha đổi, replay prefix hoặc đoạn cần thiết.
- Đo replay cost rõ ràng.

### Selective w

Nếu bỏ PARM nhiều token:

- always-update cache không tiết kiệm compute thật.
- lazy replay có thể tiết kiệm nếu skip span đủ dài.

Phải báo cáo:

- PARM forward calls.
- PARM processed tokens.
- Replay tokens.
- Net latency.

---

## 16. Training objective tracker

### Label sources

Ưu tiên:

1. One-hot PARM oracle signal.
2. Independent objective RM delta trên rollout.
3. Final objective reward distributed bằng return-to-go.

Không dùng final reward lặp nguyên cho mọi token mà không kiểm tra bias.

### Targets

Incremental:

\[
\hat r_{k,t}\approx
R_k(x,y_{\le t})-R_k(x,y_{<t}).
\]

Hoặc remaining return:

\[
\hat V_{k,t}\approx
\mathbb E[R_k(x,y)\mid x,y_{<t}].
\]

Loss:

\[
\mathcal L_{tracker}
=
\lambda_1
\|\hat{\mathbf r}_t-\mathbf r_t^*\|_1
+\lambda_2\mathcal L_{rank}
+\lambda_3\mathcal L_{cal}.
\]

Evaluation:

- MAE từng objective.
- Spearman.
- Pairwise accuracy.
- Calibration error.
- Dynamic-alpha regret khi dùng predicted signal thay oracle.

Nếu tracker score tốt nhưng downstream dynamic-alpha không cải thiện, không giữ tracker.

---

## 17. Training token router

Tại sampled prefix, thử:

\[
\mathcal W=\{0,w_{fixed}\}
\]

hoặc:

\[
\mathcal W=\{0,0.5,1\}.
\]

Final utility:

\[
U(y)=
(\alpha^{user})^\top q(x,y).
\]

Advantage:

\[
\Delta_t(w)=U(y^{(w)})-U(y^{(0)}).
\]

Tie:

\[
|\Delta_t(w)|\le\epsilon_{tie}.
\]

Binary label:

\[
g_t^*
=
\mathbb I[
\max_w\Delta_t(w)
>\lambda C_t+\epsilon_{tie}
].
\]

Loss:

\[
\mathcal L_w
=
\mathcal L_{BCE/focal}
+\lambda_{budget}\mathcal L_{budget}
+\lambda_{cal}\mathcal L_{cal}.
\]

Không ép tie thành positive hoặc negative.

---

## 18. Oracle experiments

### Oracle A: Per-objective signal

Mục tiêu:

- Xác nhận one-hot alpha tạo objective-specific signal.
- Đo correlation với independent evaluator.

Pass khi:

- Signal của objective k tương quan dương với evaluator k.
- Cross-objective signal không hoàn toàn giống nhau.
- Alpha trajectory không chỉ phản ánh logit scale.

### Oracle B: Dynamic alpha

So sánh:

- Fixed alpha_user.
- MATO closed-form alpha_t.
- Oracle best alpha theo block.
- Random dynamic alpha.

Đánh giá:

- MIP.
- HV.
- Worst-objective reward.
- User utility.
- KL deviation.
- Alpha stability.
- Runtime.

### Oracle C: Selective w

So sánh:

- Fixed w.
- Per-state oracle w.
- Random gate cùng call rate.
- Entropy gate.

### Oracle D: Joint

Chỉ chạy nếu B và C pass.

So sánh:

- Fixed alpha + fixed w.
- Dynamic alpha + fixed w.
- Fixed alpha + dynamic w.
- Dynamic alpha + dynamic w.
- Oracle joint.

---

## 19. Data schemas

### generation_records.jsonl

~~~json
{
  "run_id": "string",
  "sample_id": "string",
  "prompt": "string",
  "alpha_user": [0.7, 0.3],
  "method": "dynacap_parm",
  "response": "string",
  "seed": 42,
  "num_new_tokens": 128,
  "latency_seconds": 0.0,
  "parm_calls": 0,
  "parm_processed_tokens": 0,
  "replay_tokens": 0,
  "intervention_rate": 0.0,
  "mean_kl_alpha": 0.0,
  "mean_alpha_shift": 0.0,
  "trace_path": "traces/sample_id.jsonl"
}
~~~

### token trace

~~~json
{
  "sample_id": "string",
  "step": 12,
  "alpha_t": [0.62, 0.38],
  "incremental_rewards": [0.1, -0.03],
  "cumulative_rewards": [1.2, 0.4],
  "objective_deficits": [-0.4, 0.4],
  "tracker_uncertainty": 0.1,
  "w_t": 1.0,
  "gate_probability": 0.8,
  "parm_called": true,
  "token_id": 123
}
~~~

### scored_records.jsonl

~~~json
{
  "run_id": "string",
  "sample_id": "string",
  "alpha_user": [0.7, 0.3],
  "raw_rewards": {
    "objective_1": 0.0,
    "objective_2": 0.0
  },
  "normalized_rewards": {
    "objective_1": 0.0,
    "objective_2": 0.0
  },
  "user_utility": 0.0,
  "worst_objective": 0.0,
  "latency_seconds": 0.0,
  "parm_calls": 0
}
~~~

JSONL writer phải:

- Atomic hoặc flush định kỳ.
- Resume theo sample_id.
- Không duplicate.
- Xử lý corrupt final line.

---

## 20. Evaluation metrics

### MIP

\[
MIP=
\frac{1}{N}
\sum_{n=1}^{N}
(\alpha_n^{user})^\top q_n.
\]

Không thay alpha_user bằng alpha_t.

### Hypervolume

Dùng cùng reference point cho tất cả method. Reference point được cố định trước khi chạy test.

### Objective balance

- Mean từng objective.
- Minimum/worst objective.
- Reward variance giữa objectives.
- Percentage prompts có objective dưới threshold.

### Dynamic-alpha diagnostics

- Mean KL:

\[
\frac{1}{T}\sum_t
D_{KL}(\alpha_t\Vert\alpha^{user}).
\]

- Mean step shift:

\[
\frac{1}{T}
\sum_t
\|\alpha_t-\alpha_{t-1}\|_1.
\]

- Dominant-objective switch count.
- Alpha entropy.
- Fraction updates bị trust-region clipping.

### Efficiency

- End-to-end latency.
- Tokens/second.
- PARM calls.
- PARM processed tokens.
- Replay tokens.
- Objective tracker time.
- Dynamic controller time.
- Peak memory.

### Statistics

- Paired bootstrap theo prompt.
- 95% confidence interval.
- Ít nhất ba seeds cho main table nếu có thể.
- Báo absolute và relative gain.

---

## 21. Baselines

### Bắt buộc

- Base LLM.
- Fixed PARM.
- Fixed alpha + best fixed w.
- MATO-style dynamic alpha + fixed w.
- Fixed alpha + learned w.
- Joint dynamic alpha + learned w.
- Oracle dynamic alpha.
- Oracle w.

### Multi-objective baselines

- Rewarded Soups.
- MOD/MOD-w2s.
- GenARM.
- PARM.
- RMOD nếu reproducible.
- MATO nếu code khả dụng.
- PRO có thể giữ trong bảng như prompt-level adaptive-weight baseline, nhưng không nằm trong method.

### Routing baselines

- Random gate.
- Periodic gate.
- Entropy threshold.
- Margin threshold.
- TARo-style router.

Fairness:

- Cùng base model.
- Cùng prompts và alpha_user.
- Cùng max tokens.
- Cùng evaluator.
- Cùng sampling seeds khi có thể.
- Báo compute budget.

---

## 22. Unit tests

### Dynamic alpha

- Output thuộc simplex.
- alpha không âm.
- Sum bằng 1.
- Equal rewards giữ gần alpha_user.
- Objective reward thấp hơn nhận weight cao hơn.
- Temperature lớn làm update mềm hơn.
- eta = 0 giữ alpha_previous.
- eta = 1 dùng candidate alpha.
- KL không vượt budget.
- Không NaN khi alpha_user chứa zero; dùng epsilon đúng.

### Objective tracker

- Output shape batch x K.
- Không NaN/Inf.
- Save/load parity.
- Mask padding đúng.
- Uncertainty hữu hạn.

### PARM probing

- One-hot vector hợp lệ.
- Probe k khác preference k khác trong ít nhất fixture kiểm soát.
- Cache không bị reuse sai giữa alpha values.

### Fusion

- w = 0 trả base.
- w = 1 trả PARM.
- w = 0.5 trả midpoint.
- parm_product khớp reference.
- Shape mismatch raise error.

### Cache

- Alpha đổi phải trigger strategy đã cấu hình.
- Replay khớp full recompute trong tolerance.
- w = 0 không tăng PARM call counter.
- EOS dừng chính xác.

### Metrics

- MIP manual parity.
- HV toy parity.
- Dynamic-alpha diagnostics manual parity.
- Dominated points không tăng HV.
- Bootstrap deterministic.

---

## 23. Config mẫu

~~~yaml
experiment:
  name: dynacap_parm
  seed: 42
  output_dir: results
  resume: true

models:
  base_model: REPLACE_BASE
  parm_model: REPLACE_PARM
  objective_tracker: REPLACE_TRACKER_OR_NULL
  evaluators:
    objective_1: REPLACE_EVAL_1
    objective_2: REPLACE_EVAL_2

objectives:
  names: [helpfulness, harmlessness]
  normalization_source: validation

dynamic_alpha:
  enabled: true
  signal_source: objective_tracker
  update_interval: 8
  temperature: 0.5
  smoothing: 0.25
  kl_budget: 0.1
  epsilon: 1.0e-8

token_router:
  enabled: true
  type: binary
  threshold: 0.5
  candidate_weights: [0.0, 1.0]
  use_parm_logits: false

features:
  base_top_k: 32
  use_entropy: true
  use_margin: true
  use_position: true
  use_objective_deficits: true
  use_tracker_uncertainty: true

fusion:
  mode: convex
  score_space: logits
  fixed_weight: 0.5

cache:
  alpha_change_strategy: block_replay
  lazy_parm: true

generation:
  max_new_tokens: 128
  do_sample: true
  temperature: 0.7
  top_p: 0.9

evaluation:
  bootstrap_samples: 1000
  confidence_level: 0.95
  hv_reference_point: REPLACE_FIXED_REFERENCE
~~~

Config loader phải fail-fast nếu còn placeholder.

---

## 24. CLI dự kiến

~~~bash
python scripts/reproduce_fixed_parm.py \
  --config configs/reproduce_parm.yaml

python scripts/build_dynamic_alpha_oracle.py \
  --config configs/oracle_dynamic_alpha.yaml

python scripts/build_w_oracle.py \
  --config configs/oracle_w.yaml

python scripts/build_tracker_dataset.py \
  --config configs/train_objective_tracker.yaml

python scripts/train_objective_tracker.py \
  --config configs/train_objective_tracker.yaml

python scripts/train_token_router.py \
  --config configs/train_token_router.yaml

python scripts/generate.py \
  --config configs/evaluate_joint.yaml

python scripts/score.py \
  --run-dir results/RUN_ID

python scripts/evaluate.py \
  --run-dir results/RUN_ID
~~~

Mỗi command cần:

- --help.
- --limit.
- --resume.
- --dry-run nếu phù hợp.
- Clear error khi thiếu model/config.

---

## 25. Stage gates

### Gate A: Baseline

Pass khi:

- Fixed PARM reproduce trong tolerance.
- Alpha conditioning hoạt động.
- Fusion tests pass.

### Gate B: Objective signal

Pass khi:

- Per-objective prefix signal khác nhau có ý nghĩa.
- Signal tương quan với independent objective evaluation.
- Không chỉ phản ánh response length hoặc logit scale.

### Gate C: Oracle dynamic alpha

Pass nếu:

- HV/MIP hoặc worst-objective tăng khoảng 3-5%; và
- User utility không giảm đáng kể; và
- KL deviation nằm trong budget.

Nếu fail, bỏ MATO branch và quay về w-only.

### Gate D: Objective tracker

Pass khi:

- Predicted signal có ranking tốt hơn random.
- Dynamic alpha dùng predicted signal vẫn vượt fixed alpha.
- Gain tồn tại trên unseen prompts.

### Gate E: Oracle w

Pass nếu:

- Giảm khoảng 30-40% PARM compute với quality loss không quá 1%; hoặc
- Tăng quality khoảng 5% tại cùng budget.

### Gate F: Learned w

Pass khi:

- Vượt random, periodic và entropy gate tại cùng call rate.
- Gần oracle quality-cost frontier hơn fixed w.

### Gate G: Joint model

Pass khi:

- Joint vượt cả dynamic-alpha-only và w-only.
- Gain tồn tại với independent evaluators.
- Net latency tính cả tracker, replay và alpha update vẫn hợp lý.

---

## 26. Rủi ro

### PARM cache không hợp lệ khi alpha đổi

Đây là rủi ro kỹ thuật lớn nhất.

Xử lý:

- Block-wise alpha update.
- Recompute/replay tại boundary.
- Numerical cache parity tests.
- Đo runtime thực.

### One-hot probes quá tốn

Chỉ dùng oracle. Không dùng production nếu cần K passes mỗi step.

### Objective tracker không chính xác

Nếu downstream regret cao:

- Thêm uncertainty.
- Không update alpha khi uncertainty vượt threshold.
- Tăng update interval.
- Dùng conservative step size.

### Dynamic alpha vi phạm user intent

Xử lý:

- alpha_user là prior.
- KL trust region.
- Log toàn bộ alpha trajectory.
- Có strict mode tắt adaptation.

### Alpha oscillation

Xử lý:

- Smoothing.
- Update theo block.
- Stability penalty.
- Hysteresis cho dominant objective.

### Hai controller xung đột

Ví dụ alpha_t tăng safety nhưng w_t lại về 0.

Xử lý:

- Cho token router nhìn objective deficit.
- Train sequentially.
- Kiểm tra conditional intervention rate theo từng objective.
- Không joint-train từ đầu.

### Reward hacking

Xử lý:

- Independent evaluators.
- Multiple evaluator families.
- Pairwise judge trên subset.
- Length/repetition controls.

---

## 27. Definition of Done

Implementation hoàn thành khi:

- Fixed PARM reproduce.
- One-hot objective signal được kiểm chứng.
- Oracle dynamic-alpha report hoàn chỉnh.
- Objective tracker chỉ được train nếu oracle pass.
- Oracle-w và learned w report hoàn chỉnh.
- Cache behavior khi đổi alpha được test.
- Lazy PARM computation được đo thật.
- Generation, scoring và aggregation tách riêng.
- JSONL schema ổn định và resume được.
- HV, MIP, objective balance và efficiency được báo cáo.
- Confidence intervals được tính.
- Main results dùng independent evaluators.
- Tất cả unit test và smoke test pass.
- Có command tái tạo mỗi bảng chính.

---

## 28. Paper framing

Không viết:

> Chúng tôi ghép PARM, MATO và TARo.

Nên viết:

> Chúng tôi xây dựng một dual-control framework cho multi-objective test-time alignment. Một entropic feedback controller điều chỉnh objective priorities dựa trên prefix-level satisfaction, trong khi một budget-aware intervention controller quyết định khi nào reward guidance đáng chi phí.

Hai câu hỏi:

- What objective is currently underserved?
- Is reward guidance worth invoking at this token?

Potential claim, chỉ dùng nếu kết quả hỗ trợ:

> DynaCAP-PARM improves the multi-objective quality-cost frontier by dynamically correcting neglected objectives while selectively invoking preference-aware reward guidance.

Nếu dynamic alpha fail:

- Thu hẹp về selective w-only.

Nếu w fail nhưng dynamic alpha pass:

- Thu hẹp về dynamic objective balancing for PARM.

Nếu cả hai pass nhưng joint không pass:

- Báo hai controller riêng; không ép thành joint contribution.

---

## 29. Tài liệu nền

- PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model.
- MATO: dynamic test-time multi-objective alignment using reward discovery, objective balancing and online optimization.
- TARo: Token-level Adaptive Routing for LLM Test-time Alignment.
- GenARM: Reward Guided Generation with Autoregressive Reward Model.
- MOD/MOD-w2s: Multi-Objective Decoding.
- RMOD: robust multi-objective decoding.
- PRO: prompt-level Preference Orchestrator; giữ làm baseline, không còn là thành phần của method.

Khi triển khai reproduction, kiểm tra paper và official code. Phần dynamic alpha trong tài liệu này là adaptation có ràng buộc user preference, không được gọi là reproduction nguyên bản của MATO.

---

## 30. Prompt bắt đầu cho Codex

> Đọc README_CAP_PARM_MATO.md và audit repository mà chưa sửa code. Xác định model loading, PBLoRA alpha conditioning, KV-cache behavior khi alpha thay đổi, fusion, generation, scoring và metrics. Sau đó lập plan chỉ cho Gate A và Gate B: reproduce fixed PARM và kiểm chứng one-hot per-objective prefix signals. Không train objective tracker hoặc token router trước khi hai gate này pass.

