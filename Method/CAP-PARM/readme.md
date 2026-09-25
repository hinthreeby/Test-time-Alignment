# CAP-PARM Implementation Specification

## Context-Adaptive Preference Allocation and Selective Reward Guidance for Multi-Objective Test-Time Alignment

Tài liệu này là đặc tả kỹ thuật để Codex hoặc một lập trình viên khác có thể triển khai, kiểm thử và đánh giá CAP-PARM mà không phải suy đoán lại mục tiêu nghiên cứu.

---

## 0. Quy tắc thực thi dành cho Codex

Trước khi viết code:

1. Đọc toàn bộ README này.
2. Khảo sát cấu trúc repository hiện tại bằng rg --files.
3. Tìm pipeline PARM/PBLoRA, generation, scoring và evaluation đang có.
4. Tái sử dụng code hiện tại; không tạo pipeline song song nếu chức năng tương đương đã tồn tại.
5. Không thay đổi checkpoint PBLoRA epoch-1 đã được xác nhận nếu chưa có yêu cầu.
6. Không huấn luyện full CAP-PARM trước khi các oracle test tương ứng vượt qua tiêu chí go/no-go.
7. Không dùng evaluator đang dẫn đường làm evaluator chính thức cho kết quả cuối.
8. Mọi experiment phải lưu config, seed, commit hash, output và thời gian chạy.
9. Mỗi stage phải có unit test và smoke test trước khi chạy full dataset.
10. Không xóa hoặc ghi đè kết quả cũ; tạo run directory mới.

Thứ tự ưu tiên:

1. Reproduce fixed PARM.
2. Oracle token-level intervention.
3. Train selective token controller.
4. Oracle prompt-specific preference.
5. Chỉ train preference orchestrator nếu Oracle-alpha vượt gate.
6. Cuối cùng mới thử joint alpha + w.

Quyết định phương pháp hiện tại:

- **Core path:** PARM + token-level w_t, lấy cảm hứng từ TARo nhưng controller phải quyết định trước PARM call nếu muốn tiết kiệm compute.
- **Optional path:** PRO-style prompt-level alpha_x, chỉ triển khai sau khi Oracle-alpha chứng minh có gain.
- **Không dùng PRO làm core ngay lúc này:** prompt router trước đó đạt MIP 0.61750, thấp hơn fixed MIP 0.64688, với Spearman gần 0 và pairwise accuracy gần 0.50.

---

## 1. Mục tiêu nghiên cứu

PARM dùng một Preference-Aware Autoregressive Reward Model để hướng dẫn frozen base LLM theo vector preference của người dùng.

CAP-PARM mở rộng PARM bằng hai biến điều khiển:

| Biến | Phạm vi | Ý nghĩa |
|---|---|---|
| alpha_x | Một lần cho mỗi prompt | Điều chỉnh tỷ trọng objective theo nội dung prompt |
| w_t | Mỗi decoding step | Điều chỉnh mức can thiệp của PARM |

Câu hỏi nghiên cứu:

> Có thể cải thiện multi-objective alignment và giảm chi phí inference bằng cách tự động quyết định objective nào cần ưu tiên và token nào thực sự cần reward guidance hay không?

Mục tiêu chính:

- Cải thiện hoặc giữ nguyên HV và MIP.
- Giảm số lần forward PARM.
- Bảo toàn preference do người dùng cung cấp.
- Giữ base LLM và PARM frozen sau khi backbone đã được huấn luyện.

---

## 2. Những gì không thuộc phạm vi ban đầu

Không triển khai trong phiên bản đầu:

- RL fine-tuning cho base LLM.
- Thay đổi kiến trúc PBLoRA.
- Huấn luyện lại PARM bằng dataset mới trước khi baseline được reproduce.
- Controller lớn ngang với base LLM hoặc PARM.
- Tree search, beam search lớn hoặc nhiều rollout tại inference thật.
- Dùng evaluator reward model ở inference.
- Dynamic objective discovery như MATO.
- Tự động thay hoàn toàn preference rõ ràng của người dùng.

---

## 3. Cơ sở phương pháp

### 3.1. Ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| x | Prompt |
| y_<t | Prefix trước token t |
| v | Candidate token |
| K | Số objective |
| alpha_user | Preference vector do người dùng cung cấp |
| alpha_x | Preference vector sau prompt adaptation |
| a_t(v) | Base-model score cho token v |
| b_t(v, alpha_x) | PARM score cho token v dưới alpha_x |
| w_t | Mức can thiệp PARM tại token t |
| R_k(x,y) | Independent evaluator score cho objective k |
| q | Vector objective đã chuẩn hóa |

Mọi preference vector phải thuộc simplex:

\[
\alpha_k \ge 0,\qquad \sum_{k=1}^{K}\alpha_k=1.
\]

### 3.2. PARM gốc

PARM nhận alpha và tạo phân phối autoregressive:

\[
\pi_{\theta(\alpha)}(y_t\mid x,y_{<t}).
\]

Reward của response:

\[
r(x,y,\alpha)=\sum_t\log\pi_{\theta(\alpha)}(y_t\mid x,y_{<t}).
\]

PBLoRA condition model theo alpha:

\[
\theta(\alpha)=\theta_0+sBW(\alpha)A.
\]

Paper PARM sử dụng guided decoding:

\[
\widetilde{\pi}(y_t)\propto
\pi_{base}(y_t)
\pi_{\theta(\alpha)}(y_t)^{1/\beta}.
\]

Dạng log-space:

\[
\log\widetilde{\pi}(y_t)
=
\log\pi_{base}(y_t)
+\frac{1}{\beta}\log\pi_{\theta(\alpha)}(y_t)
-\log Z.
\]

### 3.3. Fusion chuẩn của project

Project hiện sử dụng canonical interpolation:

\[
z_t(v)=(1-w_t)a_t(v)+w_tb_t(v\mid\alpha_x),
\qquad w_t\in[0,1].
\]

Ý nghĩa:

- w_t = 0: base-only.
- w_t = 1: PARM-only score.
- w_t = 0.5: midpoint chính xác giữa hai score.

Không quay lại công thức a + lambda b trong implementation chính vì lambda vừa thay đổi tỷ lệ Base/PARM vừa làm đổi logit scale.

Hai fusion mode phải được hỗ trợ để đối chiếu:

1. convex: công thức nội suy phía trên; đây là mặc định của project.
2. parm_product: công thức gốc của PARM; chỉ dùng làm baseline.

Không trộn kết quả của hai fusion mode trong cùng một bảng nếu không ghi rõ.

### 3.4. Prompt-conditioned preference

Preference orchestrator dự đoán residual:

\[
\alpha_x=
\operatorname{softmax}
\left(
\log(\alpha_{user}+\epsilon)
+s_\alpha\delta_\phi(x,\alpha_{user})
\right).
\]

Yêu cầu:

- alpha_x luôn nằm trên simplex.
- Có residual scale s_alpha để giới hạn mức điều chỉnh.
- Có KL constraint để bảo toàn ý người dùng.

\[
\mathcal L_{KL}=D_{KL}(\alpha_x\Vert\alpha_{user}).
\]

Không diễn giải alpha là target reward. Alpha là trọng số preference, không phải vector reward cần đạt.

### 3.5. Token-level intervention

Controller dự đoán:

\[
w_t=g_\psi(f_t).
\]

Version đầu dùng binary gate:

\[
w_t\in\{0,1\}.
\]

Sau khi binary gate vượt baseline mới mở rộng:

\[
w_t\in\{0,0.5,1\}
\quad\text{hoặc}\quad
w_t\in[0,1].
\]

Nếu mục tiêu là giảm compute, feature f_t phải được tính trước khi gọi PARM. Không được dùng PARM logits làm feature bắt buộc cho binary skip gate.

Feature khả thi:

- Base top-k log probabilities.
- Base entropy.
- Top-1/top-2 margin.
- Token position hoặc normalized position.
- Prompt embedding đã pooling.
- Current alpha_x.
- Previous gate decision.
- Running statistics của base confidence.

Hidden state của base model chỉ dùng nếu đã có sẵn và không cần thêm forward pass.

### 3.6. Joint constrained objective

\[
\max_{\phi,\psi}\;
\mathbb E\left[
\alpha_{user}^{\top}\bar{\mathbf R}(x,y)
-\tau D_{KL}(\alpha_x\Vert\alpha_{user})
-\lambda\frac{1}{T}\sum_t\mathbb I[w_t>0]
\right].
\]

Ba thành phần:

1. Multi-objective utility.
2. Preference fidelity.
3. PARM inference cost.

Có thể thay cost penalty bằng ràng buộc:

\[
\frac{1}{T}\sum_t\mathbb I[w_t>0]\le B.
\]

---

## 4. Trạng thái thực nghiệm hiện tại

Các số sau phải được lưu làm historical reference, không xem là final result:

| Experiment | Kết quả |
|---|---:|
| Best fixed MIP | 0.64688 |
| Oracle per-case MIP | 0.71949 |
| Relative oracle headroom | 11.23% |
| Best per-alpha MIP | 0.65379 |
| Best prompt router MIP | 0.61750 |
| Prompt-router Spearman | xấp xỉ 0 |
| Prompt-router pairwise accuracy | xấp xỉ 0.50 |
| Token states đã phân tích | 250 |
| Tied token states | 225/250 |
| States có cùng next token | 74.4% |
| Actionable states | 64/250 |

Kết luận hiện tại:

- Prompt-only router cũ đã thất bại và không được dùng làm bằng chứng cho adaptive alpha.
- Alpha đơn lẻ chỉ giải thích một phần nhỏ oracle headroom.
- Token-level signal thưa; controller phải xử lý imbalance và tie.
- Hướng khả thi hơn trước mắt là value-of-steering hoặc selective w_t.
- Full CAP-PARM chỉ được tiếp tục nếu oracle-alpha và oracle-w cho thấy headroom thật.

---

## 5. Kiến trúc phần mềm đề xuất

~~~text
cap_parm/
├── configs/
│   ├── base.yaml
│   ├── reproduce_parm.yaml
│   ├── oracle_alpha.yaml
│   ├── oracle_w.yaml
│   ├── train_alpha_router.yaml
│   ├── train_w_router.yaml
│   └── evaluate_joint.yaml
├── src/
│   ├── models/
│   │   ├── base_adapter.py
│   │   ├── parm_adapter.py
│   │   ├── alpha_router.py
│   │   ├── token_router.py
│   │   └── calibration.py
│   ├── decoding/
│   │   ├── fusion.py
│   │   ├── selective_decoder.py
│   │   └── cache_manager.py
│   ├── oracle/
│   │   ├── alpha_search.py
│   │   ├── intervention_search.py
│   │   └── rollout.py
│   ├── data/
│   │   ├── schemas.py
│   │   ├── jsonl.py
│   │   ├── datasets.py
│   │   └── splits.py
│   ├── training/
│   │   ├── train_alpha_router.py
│   │   ├── train_token_router.py
│   │   └── losses.py
│   ├── evaluation/
│   │   ├── score_outputs.py
│   │   ├── metrics.py
│   │   ├── bootstrap.py
│   │   └── report.py
│   └── utils/
│       ├── config.py
│       ├── seed.py
│       ├── logging.py
│       └── run_manifest.py
├── scripts/
│   ├── reproduce_fixed_parm.py
│   ├── build_alpha_oracle.py
│   ├── build_w_oracle.py
│   ├── train_alpha_router.py
│   ├── train_token_router.py
│   ├── generate.py
│   ├── score.py
│   └── evaluate.py
├── tests/
│   ├── test_fusion.py
│   ├── test_alpha_simplex.py
│   ├── test_router.py
│   ├── test_lazy_parm.py
│   ├── test_kv_cache.py
│   ├── test_jsonl_schema.py
│   └── test_metrics.py
└── results/
    └── RUN_ID/
        ├── config.yaml
        ├── manifest.json
        ├── generation_records.jsonl
        ├── scored_records.jsonl
        ├── metrics.json
        ├── per_alpha_metrics.csv
        └── figures/
~~~

Nếu repository đã có cấu trúc tương đương, map các module vào code hiện tại thay vì tạo lại toàn bộ tree.

---

## 6. Interface bắt buộc

### 6.1. Base model adapter

~~~python
class BaseModelAdapter:
    def prefill(self, input_ids, attention_mask=None):
        """Return next-token logits, hidden features, and KV cache."""

    def step(self, token_id, past_key_values):
        """Advance one token and return logits/features/new cache."""
~~~

### 6.2. PARM adapter

~~~python
class PARMAdapter:
    def set_preference(self, alpha):
        """Validate alpha and configure PBLoRA conditioning."""

    def prefill(self, input_ids, alpha, attention_mask=None):
        """Return PARM logits and KV cache."""

    def step(self, token_id, alpha, past_key_values):
        """Advance PARM cache for one generated token."""
~~~

### 6.3. Alpha router

~~~python
class AlphaRouter:
    def forward(self, prompt_features, alpha_user):
        """Return alpha_x on the K-dimensional simplex."""
~~~

### 6.4. Token router

~~~python
class TokenRouter:
    def forward(self, base_features, alpha_x, step_index):
        """Return gate probability, selected w_t, and uncertainty."""
~~~

### 6.5. Fusion

~~~python
def fuse_scores(base_scores, parm_scores, weight, mode="convex"):
    """
    convex: (1 - weight) * base_scores + weight * parm_scores
    parm_product: base_log_probs + weight * parm_log_probs
    """
~~~

Yêu cầu shape:

- base_scores: batch x vocab.
- parm_scores: batch x vocab.
- weight: batch x 1 hoặc scalar.
- output: batch x vocab.

Không broadcast ngầm nếu batch size không khớp.

---

## 7. Calibration

Raw logits của base và PARM có thể có scale khác nhau. Hỗ trợ ba mode:

1. none: dùng để reproduce kết quả hiện tại.
2. temperature: chia từng source cho temperature học trên validation.
3. logprob: dùng log_softmax trước fusion.

Config:

~~~yaml
fusion:
  mode: convex
  score_space: logits
  base_temperature: 1.0
  parm_temperature: 1.0
~~~

Quy tắc:

- Mặc định reproduce phải giữ nguyên score_space hiện tại.
- Calibration parameters chỉ fit trên validation set.
- Không fit temperature trên test set.
- Báo cáo calibration ablation riêng.

---

## 8. Pipeline triển khai

### Phase 0: Audit repository

Codex phải tạo báo cáo ngắn gồm:

- Entry point hiện tại.
- Model/checkpoint paths.
- Dataset loader.
- Fusion implementation.
- Generation schema.
- Evaluation schema.
- Các test đang có.
- Những module có thể tái sử dụng.

Không chỉnh sửa code ở phase này.

### Phase 1: Reproduce fixed PARM

Mục tiêu:

- Reproduce best fixed MIP khoảng 0.64688 trong tolerance hợp lý.
- Xác nhận alpha conditioning hoạt động.
- Xác nhận w endpoint và midpoint parity.

Chạy grid:

\[
w\in\{0,0.1,\ldots,1.0\}.
\]

Lưu output riêng cho mỗi alpha và mỗi w.

Nếu baseline không reproduce, dừng và debug trước khi làm router.

### Phase 2: Oracle-w

Mục tiêu là đo headroom thật của selective intervention.

Version đầu không tìm w cho mọi token trong full response. Dùng sampled states:

1. Chạy fixed baseline để thu prefix states.
2. Chọn state theo stratified sampling:
   - low entropy,
   - medium entropy,
   - high entropy,
   - early/middle/late position.
3. Tại mỗi state, thử candidate action:

\[
\mathcal W=\{0,w_{fixed}\}
\]

hoặc:

\[
\mathcal W=\{0,0.5,1\}.
\]

4. Roll out completion bằng cùng decoding seed khi có thể.
5. Chấm final utility bằng independent evaluators.
6. Tính advantage:

\[
\Delta_t(w)=U(y^{(w)})-U(y^{(0)}).
\]

Tie rule:

\[
|\Delta_t(w)|\le\epsilon_{tie}
\Rightarrow \text{tie}.
\]

Không ép tie thành positive hoặc negative.

Output của phase này là oracle dataset và oracle quality-cost curve.

### Phase 3: Train binary token router

Ưu tiên binary gate:

\[
w_t\in\{0,w_{fixed}\}.
\]

Feature chỉ dùng thông tin có trước PARM call.

Target:

\[
g_t^*=\mathbb I[\max_w\Delta_t(w)>\lambda C_t+\epsilon_{tie}].
\]

Do positive state thưa, hỗ trợ:

- Weighted BCE.
- Focal loss.
- Pairwise ranking.
- Balanced batch sampler.
- Calibration bằng validation set.

Loss mặc định:

\[
\mathcal L_w=
\mathcal L_{BCE}
+\lambda_{cal}\mathcal L_{cal}
+\lambda_{budget}\max(0,\hat c-B).
\]

Không dùng token gold-NLL làm objective duy nhất vì mục tiêu là final multi-objective utility, không chỉ bắt chước gold token.

### Phase 4: Lazy PARM execution

Muốn giảm compute thật, decoder phải:

1. Chạy base step.
2. Tạo feature.
3. Router quyết định gate.
4. Nếu gate = 0, không chạy PARM logits cho decision đó.
5. Nếu gate > 0, chạy PARM và fuse.

Vấn đề cache:

- PARM vẫn cần prefix đầy đủ khi được kích hoạt lại.
- Cần một trong hai chiến lược:
  1. Always-update cache: vẫn forward PARM mỗi token; không tiết kiệm FLOPs thật.
  2. Lazy replay: khi kích hoạt lại, replay các token bị bỏ qua vào PARM.

Version đầu phải đo cả hai:

- always-update để kiểm tra chất lượng/router.
- lazy-replay để đo compute thực.

Không tuyên bố giảm inference cost nếu vẫn forward PARM ở mọi token để duy trì cache.

Tối ưu lazy replay:

- Batch replay đoạn token bị bỏ qua.
- Lưu last_parm_position.
- Dùng attention mask đúng.
- Kiểm tra logits sau replay khớp với always-update trong tolerance.

### Phase 5: Oracle-alpha

Chỉ chạy sau khi fixed baseline ổn định.

Candidate alpha grid phải phủ simplex. Với K = 2:

\[
\alpha_1\in\{0,0.1,\ldots,1\},
\qquad \alpha_2=1-\alpha_1.
\]

Với K = 3, dùng simplex lattice hoặc Dirichlet samples cố định.

Với mỗi prompt:

1. Nhận alpha_user.
2. Sinh response dưới từng candidate alpha.
3. Chấm objective vector q.
4. Chọn:

\[
\alpha^*=
\arg\max_\alpha
\left[
\alpha_{user}^{\top}q(x,y_\alpha)
-\tau D_{KL}(\alpha\Vert\alpha_{user})
\right].
\]

Oracle-alpha report phải có:

- Mean utility gain.
- HV gain.
- MIP gain.
- Percentage prompts có alpha khác alpha_user.
- Distribution của KL shift.
- Bootstrap confidence interval.

### Phase 6: Train alpha router

Chỉ thực hiện nếu oracle-alpha vượt gate.

Input:

- Prompt tokens hoặc pooled prompt embedding.
- alpha_user.
- Optional task/domain ID nếu có thật trong dataset.

Output:

- Residual preference vector.

Loss:

\[
\mathcal L_\alpha
=
\mathcal L_{rank}
+\lambda_{KL}D_{KL}(\alpha_x\Vert\alpha_{user})
+\lambda_{reg}\|\delta_\phi\|_2^2.
\]

Ưu tiên ranking candidate alpha hơn regression trực tiếp vào alpha_star nếu nhiều candidate có utility gần bằng nhau.

Phải báo cáo:

- Spearman giữa predicted ranking và oracle utility.
- Pairwise accuracy.
- Regret so với oracle.
- Gain so với fixed alpha.
- KL shift.

Nếu router không vượt fixed baseline trên validation, không ghép vào joint model.

### Phase 7: Joint alpha + w

Chỉ thực hiện nếu:

- token router vượt fixed-w quality-cost curve; và
- alpha router vượt fixed-alpha baseline.

Inference:

1. Compute alpha_x một lần từ prompt.
2. Prefill base.
3. Tại mỗi step, token router chọn w_t.
4. PARM dùng alpha_x khi được kích hoạt.
5. Fuse và sample token.
6. Lưu toàn bộ decisions để audit.

Không joint fine-tune cả hai controller ngay từ đầu. Trình tự:

1. Freeze alpha router, train w router.
2. Freeze w router, kiểm tra alpha router.
3. Optional low-learning-rate joint calibration.

---

## 9. Pseudocode inference

~~~python
def generate_cap_parm(prompt, alpha_user, cfg):
    alpha_user = validate_simplex(alpha_user)

    if cfg.alpha_router.enabled:
        alpha_x = alpha_router(prompt, alpha_user)
    else:
        alpha_x = alpha_user

    base_state = base_model.prefill(prompt)
    parm_state = None
    pending_tokens = []
    output_tokens = []
    trace = []

    for step in range(cfg.generation.max_new_tokens):
        base_scores = base_state.next_scores
        features = build_router_features(
            base_scores=base_scores,
            hidden=base_state.hidden,
            alpha=alpha_x,
            step=step,
        )

        gate_prob, w_t, uncertainty = token_router(features)

        if w_t > 0:
            parm_state = synchronize_parm_cache(
                prompt=prompt,
                generated_tokens=output_tokens,
                state=parm_state,
                pending_tokens=pending_tokens,
                alpha=alpha_x,
            )
            parm_scores = parm_state.next_scores
            fused_scores = fuse_scores(
                base_scores,
                parm_scores,
                w_t,
                mode=cfg.fusion.mode,
            )
            pending_tokens = []
            parm_called = True
        else:
            fused_scores = base_scores
            parm_called = False

        next_token = sample(fused_scores, cfg.generation)
        output_tokens.append(next_token)
        pending_tokens.append(next_token)
        base_state = base_model.step(next_token, base_state.cache)

        trace.append({
            "step": step,
            "w_t": float(w_t),
            "gate_prob": float(gate_prob),
            "uncertainty": float(uncertainty),
            "parm_called": parm_called,
            "token_id": int(next_token),
        })

        if next_token == eos_token_id:
            break

    return output_tokens, alpha_x, trace
~~~

---

## 10. Data schemas

### 10.1. generation_records.jsonl

~~~json
{
  "run_id": "string",
  "sample_id": "string",
  "prompt": "string",
  "alpha_user": [0.7, 0.3],
  "alpha_used": [0.65, 0.35],
  "method": "cap_parm",
  "fusion_mode": "convex",
  "seed": 42,
  "response": "string",
  "response_token_ids": [1, 2, 3],
  "num_new_tokens": 3,
  "latency_seconds": 0.0,
  "parm_calls": 2,
  "total_steps": 3,
  "intervention_rate": 0.6667,
  "trace_path": "relative/path/sample_id.jsonl"
}
~~~

### 10.2. scored_records.jsonl

~~~json
{
  "run_id": "string",
  "sample_id": "string",
  "alpha_user": [0.7, 0.3],
  "alpha_used": [0.65, 0.35],
  "response": "string",
  "raw_rewards": {
    "objective_1": 0.0,
    "objective_2": 0.0
  },
  "normalized_rewards": {
    "objective_1": 0.0,
    "objective_2": 0.0
  },
  "utility": 0.0,
  "latency_seconds": 0.0,
  "parm_calls": 0,
  "total_steps": 0
}
~~~

### 10.3. alpha_oracle.jsonl

~~~json
{
  "sample_id": "string",
  "alpha_user": [0.7, 0.3],
  "candidates": [
    {
      "alpha": [0.6, 0.4],
      "normalized_rewards": [0.0, 0.0],
      "utility": 0.0,
      "kl_to_user": 0.0
    }
  ],
  "best_alpha": [0.6, 0.4],
  "best_utility": 0.0,
  "fixed_utility": 0.0,
  "oracle_gain": 0.0
}
~~~

### 10.4. intervention_oracle.jsonl

~~~json
{
  "sample_id": "string",
  "state_id": "string",
  "step": 12,
  "prefix_token_ids": [1, 2, 3],
  "alpha_used": [0.7, 0.3],
  "base_features": {
    "entropy": 0.0,
    "top1_top2_margin": 0.0,
    "topk_logprobs": []
  },
  "actions": [
    {
      "w": 0.0,
      "utility": 0.0,
      "cost": 0.0
    },
    {
      "w": 1.0,
      "utility": 0.0,
      "cost": 1.0
    }
  ],
  "best_w": 1.0,
  "advantage": 0.0,
  "is_tie": false
}
~~~

Mọi JSONL writer phải flush định kỳ và hỗ trợ resume theo sample_id.

---

## 11. Dataset split và chống leakage

Tối thiểu:

- train: học router.
- validation: chọn threshold, temperature và hyperparameter.
- test: chỉ dùng một lần cho báo cáo cuối.

Yêu cầu:

- Split theo prompt ID trước khi tạo rollout.
- Không để prefix từ cùng một prompt xuất hiện ở cả train và test.
- Alpha candidates của test có thể chứa unseen combinations để đo generalization.
- Evaluator calibration không dùng test.
- Oracle label của test không được dùng để chọn model.

---

## 12. Evaluation

### 12.1. Objective scores

Với K objective:

\[
q(x,y)=
[\bar R_1(x,y),\ldots,\bar R_K(x,y)].
\]

Normalization parameters phải cố định từ train hoặc validation và dùng chung cho mọi method.

### 12.2. Mean Inner Product

Theo PARM:

\[
MIP=
\frac{1}{N}\sum_{n=1}^{N}
\alpha_n^\top q_n.
\]

MIP đo mức kết quả tương ứng với preference được yêu cầu. Báo cáo rõ q là raw hay normalized.

### 12.3. Hypervolume

Cho tập objective vectors S và reference point z:

\[
HV_z(S)=
\Lambda\left(
\{p\mid\exists q\in S:q\preceq p\preceq z\}
\right).
\]

Yêu cầu:

- Dùng cùng reference point cho mọi method.
- Không chọn reference point riêng để có lợi cho method.
- Với convention maximize, chuyển dấu hoặc dùng implementation nhất quán.
- Có unit test bằng các Pareto set nhỏ có kết quả biết trước.

### 12.4. Quality metrics

Báo cáo:

- HV.
- MIP.
- Mean reward từng objective.
- Worst-objective reward.
- Preference regret so với oracle.
- Pairwise win rate với fixed PARM nếu có independent judge.

### 12.5. Efficiency metrics

Báo cáo:

- End-to-end latency.
- Tokens per second.
- PARM call count.
- PARM call rate.
- Number of replayed tokens.
- Peak GPU memory.
- Total model forward time.
- Quality tại cùng call budget.

Call rate:

\[
CallRate=
\frac{\sum_t\mathbb I[w_t>0]}{T}.
\]

Nếu lazy replay xử lý nhiều token trong một batch, báo cáo cả số forward call và số token PARM xử lý.

### 12.6. Statistical reporting

- Ít nhất ba seeds cho kết quả chính nếu ngân sách cho phép.
- Bootstrap 95% confidence interval theo prompt.
- Paired bootstrap khi so cùng prompt.
- Báo cả absolute gain và relative gain.
- Không kết luận thắng nếu confidence interval của difference chứa 0, trừ khi ghi rõ exploratory.

---

## 13. Baselines

### Bắt buộc

1. Base LLM.
2. Fixed PARM với best validation w.
3. PARM với w = 0.5.
4. Oracle per-case w.
5. Learned token router.
6. Joint model nếu vượt stage gate.

### Multi-objective SOTA

- Rewarded Soups.
- MOD/MOD-w2s.
- GenARM.
- PARM.
- RMOD nếu code/checkpoint khả dụng.
- MATO nếu reproduction khả thi.
- PRO nếu đánh giá prompt-conditioned preference.

### Routing comparison

- Random gate với cùng call rate.
- Periodic gate với cùng call rate.
- Entropy threshold.
- Margin threshold.
- TARo-style router dùng cả base và PARM logits.

TARo-style router dùng cả hai logits không phải compute-saving baseline vì PARM đã được chạy trước khi routing. Ghi rõ điều này.

---

## 14. Unit tests bắt buộc

### 14.1. Alpha tests

- Tổng alpha bằng 1 trong tolerance.
- Không có alpha âm.
- Uniform input hoạt động.
- One-hot input hoạt động.
- KL bằng 0 khi alpha_x = alpha_user.
- Residual bằng 0 trả đúng alpha_user.

### 14.2. Fusion tests

- w = 0 trả đúng base scores.
- w = 1 trả đúng PARM scores.
- w = 0.5 trả đúng midpoint.
- Batch weight broadcast chính xác.
- Sai vocab size phải raise error.
- Không xuất hiện NaN/Inf.
- parm_product khớp log-space reference.

### 14.3. Router tests

- Output nằm trong miền hợp lệ.
- Binary threshold deterministic.
- State dict save/load không đổi output.
- Masked features không làm thay đổi shape.
- Uncertainty finite.

### 14.4. Lazy execution tests

- Khi w = 0, PARM forward counter không tăng.
- Khi kích hoạt lại, replay đúng số token.
- Replayed cache logits khớp always-update cache.
- EOS dừng cả hai model.
- Batch prompts dài khác nhau hoạt động.

### 14.5. Data tests

- JSONL schema validation.
- Resume không ghi duplicate sample_id.
- Partial/corrupt last line được xử lý an toàn.
- Mọi record có config hash và run ID.

### 14.6. Metric tests

- MIP khớp manual calculation.
- HV khớp Pareto set toy.
- Dominated points không làm tăng HV.
- Reference point được cố định.
- Bootstrap deterministic theo seed.

---

## 15. Smoke tests

Trước full run:

1. Hai prompts.
2. Hai alpha vectors.
3. Tối đa 16 new tokens.
4. Chạy base, fixed PARM và selective decoder.
5. Score bằng evaluator.
6. Tạo metrics.json.
7. Resume run và xác nhận không duplicate.

Sau đó chạy mini feasibility:

- 50 prompts cho pipeline correctness.
- 250 prefix states cho oracle-w parity với phân tích cũ.
- 100-200 prompts cho oracle-alpha sơ bộ.

Chỉ chạy full dataset khi toàn bộ smoke test pass.

---

## 16. Config mẫu

~~~yaml
experiment:
  name: cap_parm_binary_gate
  seed: 42
  output_dir: results
  resume: true

models:
  base_model: REPLACE_WITH_BASE_MODEL
  parm_model: REPLACE_WITH_PARM_CHECKPOINT
  evaluator_models:
    objective_1: REPLACE_WITH_EVALUATOR_1
    objective_2: REPLACE_WITH_EVALUATOR_2

objectives:
  names: [helpfulness, harmlessness]
  normalize: true
  normalization_source: validation

alpha_router:
  enabled: false
  hidden_size: 256
  residual_scale: 0.1
  kl_weight: 1.0

token_router:
  enabled: true
  type: binary
  hidden_size: 128
  threshold: 0.5
  candidate_weights: [0.0, 1.0]
  uncertainty: true

features:
  base_top_k: 32
  use_entropy: true
  use_margin: true
  use_position: true
  use_hidden_state: false
  use_parm_logits: false

fusion:
  mode: convex
  score_space: logits
  fixed_weight: 0.5
  base_temperature: 1.0
  parm_temperature: 1.0

generation:
  max_new_tokens: 128
  do_sample: true
  temperature: 0.7
  top_p: 0.9

oracle:
  tie_epsilon: 0.001
  rollout_seeds: [42]
  alpha_grid_step: 0.1

evaluation:
  bootstrap_samples: 1000
  confidence_level: 0.95
  hv_reference_point: REPLACE_WITH_FIXED_REFERENCE
~~~

Không để placeholder tồn tại khi bắt đầu run; config loader phải fail-fast.

---

## 17. CLI dự kiến

Tên argument có thể map theo repository hiện tại.

~~~bash
python scripts/reproduce_fixed_parm.py \
  --config configs/reproduce_parm.yaml

python scripts/build_w_oracle.py \
  --config configs/oracle_w.yaml

python scripts/train_token_router.py \
  --config configs/train_w_router.yaml

python scripts/build_alpha_oracle.py \
  --config configs/oracle_alpha.yaml

python scripts/train_alpha_router.py \
  --config configs/train_alpha_router.yaml

python scripts/generate.py \
  --config configs/evaluate_joint.yaml

python scripts/score.py \
  --run-dir results/RUN_ID

python scripts/evaluate.py \
  --run-dir results/RUN_ID
~~~

Mỗi command phải có:

- --help.
- --dry-run nếu phù hợp.
- --resume.
- --limit cho smoke test.
- Clear error khi thiếu checkpoint hoặc config.

---

## 18. Run manifest

Mỗi run lưu manifest.json:

~~~json
{
  "run_id": "timestamp-name-hash",
  "git_commit": "string",
  "git_dirty": false,
  "config_sha256": "string",
  "started_at": "ISO-8601",
  "finished_at": "ISO-8601",
  "hostname": "string",
  "device": "string",
  "torch_version": "string",
  "transformers_version": "string",
  "base_model": "string",
  "parm_checkpoint": "string",
  "dataset": "string",
  "split": "test",
  "status": "completed"
}
~~~

Không lưu secret, access token hoặc absolute private path trong manifest công khai.

---

## 19. Stage gates

### Gate A: Baseline reproduction

Pass khi:

- Fixed PARM result nằm trong tolerance xác định trước so với historical result.
- Alpha conditioning thay đổi output/reward theo hướng hợp lý.
- 17/17 fusion tests cũ và test mới đều pass.

### Gate B: Oracle-w

Pass nếu đạt ít nhất một điều:

- Giảm khoảng 30-40% PARM processing với giảm HV/MIP không quá 1%; hoặc
- Tăng quality khoảng 5% tại cùng compute budget.

Ngưỡng cuối phải được đăng ký trước full test.

### Gate C: Learned token router

Pass khi:

- Vượt random, periodic và confidence threshold tại cùng call rate.
- Nằm gần oracle frontier hơn fixed-w baseline.
- Gain giữ được trên test và ít nhất hai seeds.

### Gate D: Oracle-alpha

Pass khi:

- Prompt-specific alpha cải thiện HV/MIP khoảng 3-5% với CI hợp lý; và
- Gain không đến từ việc vi phạm preference của người dùng.

### Gate E: Learned alpha router

Pass khi:

- MIP/utility vượt fixed alpha.
- Ranking metrics vượt random rõ ràng.
- Regret so với oracle giảm.
- KL shift nằm trong budget.

### Gate F: Joint model

Pass khi:

- Alpha + w vượt alpha-only và w-only.
- Cải thiện không chỉ xuất hiện ở evaluator đang dùng để train label.
- Quality-cost frontier tốt hơn fixed PARM.

---

## 20. Bảng kết quả mục tiêu

~~~text
Method          HV↑   MIP↑   Worst-R↑   Calls↓   Tok/s↑   Latency↓
Base
Fixed PARM
Random Gate
Periodic Gate
Entropy Gate
TARo-style
CAP-PARM w-only
CAP-PARM alpha-only
CAP-PARM joint
Oracle-w
Oracle-alpha
Oracle-joint
~~~

Bảng ablation:

~~~text
Variant                     HV↑   MIP↑   Calls↓
No calibration
Temperature calibration
Log-prob fusion
Without uncertainty
Without budget penalty
Without KL constraint
Binary w
Three-level w
Continuous w
~~~

---

## 21. Rủi ro và cách xử lý

### Prompt router không học được alpha

Nguyên nhân có thể:

- Prompt không chứa đủ signal về optimal alpha.
- Evaluator noise.
- Candidate alpha có utility gần nhau.
- Alpha labels không ổn định theo decoding seed.

Xử lý:

- Kiểm tra oracle gain trước.
- Dùng ranking và soft labels.
- Aggregate nhiều rollout seeds.
- Báo alpha routing là negative result nếu không có signal.

### Token states phần lớn tie

Xử lý:

- Không train trên tie như label cứng.
- Dùng abstention.
- Oversample actionable states nhưng giữ prior khi calibration.
- Tối ưu precision của intervention thay vì accuracy tổng.

### Router chạy PARM trước khi quyết định

Đây không phải compute-saving. Phải tách:

- Quality router: có thể dùng cả hai logits.
- Compute-saving router: chỉ dùng base-side features trước PARM call.

### Lazy replay làm mất lợi ích tốc độ

Đo:

- Skip span length.
- Replay batch size.
- Replay time.
- Net wall-clock saving.

Nếu replay cost quá cao, chuyển claim từ FLOP saving sang adaptive quality control; không báo giảm compute.

### Reward hacking

Xử lý:

- Tách guide model và evaluator.
- Dùng nhiều evaluator.
- Thêm pairwise judge hoặc human check trên subset.
- Kiểm tra response length và repetition.

### Alpha adaptation vi phạm ý người dùng

Xử lý:

- Residual nhỏ.
- KL bound.
- Báo alpha_user và alpha_used.
- Có chế độ strict-user-preference tắt adaptation.

---

## 22. Definition of Done

Implementation chỉ được xem là hoàn thành khi:

- Baseline fixed PARM reproduce thành công.
- Fusion modes có test đầy đủ.
- Oracle-w dataset có thể resume và tái lập.
- Token router có quality-cost curve.
- Lazy execution thực sự giảm PARM computation hoặc claim được sửa trung thực.
- Oracle-alpha report hoàn chỉnh.
- Alpha router chỉ được đưa vào nếu vượt gate.
- Generation, scoring và aggregation tách riêng.
- Independent evaluator được sử dụng.
- JSONL schemas ổn định.
- Tất cả unit test và smoke test pass.
- Kết quả có confidence interval.
- README chạy thực tế đã thay toàn bộ placeholder bằng thông tin repository.
- Một command hoặc script có thể tái tạo từng bảng chính.

---

## 23. Deliverables

Codex cần tạo theo từng phase:

1. Repository audit report.
2. Baseline reproduction report.
3. Oracle-w dataset và report.
4. Token-router checkpoint và report.
5. Lazy-decoding benchmark.
6. Oracle-alpha dataset và report.
7. Alpha-router checkpoint nếu Gate D pass.
8. Joint CAP-PARM checkpoint nếu Gate E pass.
9. Final evaluation tables.
10. Reproduction commands và environment lock file.

---

## 24. Paper framing

Không viết:

> Chúng tôi kết hợp PARM, PRO và TARo.

Nên viết:

> Chúng tôi xây dựng một bài toán điều khiển hai timescale cho multi-objective test-time alignment: prompt-level preference allocation quyết định what to optimize, trong khi budget-aware token intervention quyết định when reward guidance is worth its cost.

Claim chính chỉ dùng nếu thực nghiệm hỗ trợ:

> CAP-PARM cải thiện quality-cost frontier bằng cách điều chỉnh preference theo prompt và chỉ kích hoạt reward guidance tại các token có expected utility gain lớn hơn chi phí can thiệp.

Nếu adaptive alpha thất bại, paper phải thu hẹp thành:

> Budgeted Selective PARM: learning when autoregressive reward guidance is worth its inference cost.

Hướng w-only vẫn là đóng góp hợp lệ nếu có oracle analysis, controller tốt, compute measurement trung thực và quality-cost frontier mạnh.

---

## 25. Tài liệu nền

- PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model.
- GenARM: Reward Guided Generation with Autoregressive Reward Model for Test-Time Alignment.
- TARo: Token-level Adaptive Routing for LLM Test-time Alignment.
- MOD: Multi-Objective Decoding.
- PRO: Preference Orchestrator.
- MATO: test-time multi-objective alignment with dynamic weight optimization.

Các PDF PARM, GenARM, TARo, MOD và các controlled-decoding baselines đã được cung cấp trong project sources. Khi triển khai công thức gốc, phải kiểm tra lại paper và code chính thức thay vì dựa vào tên phương pháp.

---

## 26. Chỉ dẫn bắt đầu cho Codex

Prompt thực thi đề xuất:

> Đọc README.md, sau đó audit repository mà chưa sửa code. Báo cáo vị trí của model loading, PARM/PBLoRA conditioning, fusion, generation, scoring và metrics. Tiếp theo lập plan chỉ cho Phase 1: reproduce fixed PARM và bổ sung các unit test cần thiết. Không triển khai router hoặc chạy training cho đến khi baseline reproduction và fusion tests vượt Gate A.
