# Router V2 + PARM-TARO — AI Implementation Guides

## Mục tiêu

Bộ tài liệu này định nghĩa hướng phát triển mới với ba nguyên tắc bắt buộc:

1. **Router V1 là baseline bất biến** — không sửa code, checkpoint, config hay evaluation hiện tại.
2. **Router V2 được tạo mới** và phải gần TARO hơn V1, sau đó mới mở rộng thành smart router.
3. **PARM gốc của tác giả được giữ nguyên**; tạo một phiên bản tích hợp mới `PARM_TARO/` để gắn adaptive router vào decoding.

## Workspace hiện tại

Root:

```text
/home/jupyter-iec2024se10/Reward Decoding
```

Các vùng quan trọng hiện có:

```text
Method/
├── CD/
├── GenARM/
├── MultiSignal/
├── RAD/
├── SafeDecoding/
└── args/

PARM/
├── code/
│   ├── data/
│   ├── evaluation/
│   └── training/
├── language-model-arithmetic/
└── peft/

dataset/
├── GenARM/
│   ├── PKU-SafeRLHF-10K/round0/
│   └── full-hh-rlhf/data/
├── RAD_train/
│   ├── amazon_polarity/
│   ├── router_amazon_polarity/
│   └── sst2/
├── rad_benchmark/
├── router_cache/rad/
└── router_train/rad/

models/
results/
results-100/

router/
├── docs/
├── evaluation/
│   ├── full_32tokens/
│   ├── pilot/
│   └── report_run_32tokens/
├── rad/
├── reports/
├── scripts/data/
└── tests/
```

## Boundary bắt buộc

Giữ nguyên:

```text
router/          # Router V1 hiện tại — READ ONLY
PARM/            # code tác giả — READ ONLY
Method/RAD/      # dependency/baseline hiện tại
```

Tạo mới:

```text
router_v2/       # Router V2
PARM_TARO/       # adaptive PARM wrapper/integration
```

Không cần tạo `Method/RouterV2/` nếu workspace hiện tại chưa có `Method/Router/`. Router V1 thực tế nằm dưới `router/`, nên Router V2 nên đặt song song thành `router_v2/`.

## Data reuse

Có sẵn:

```text
dataset/RAD_train/router_amazon_polarity/
dataset/router_cache/rad/
dataset/router_train/rad/
dataset/rad_benchmark/
dataset/GenARM/PKU-SafeRLHF-10K/round0/
```

Do đó AI phải **inventory và reuse data hiện có trước**, không download hoặc preprocess lại mặc định.

## Thứ tự đọc

1. `00_FULL_CONTEXT_V2.md`
2. `01_REPOSITORY_AND_DATA_INVENTORY.md`
3. `02_TARO_BASELINE_V2.md`
4. `03_ROUTER_V2_FEATURE_CACHE.md`
5. `04_SMART_ROUTER_V2_ARCHITECTURE.md`
6. `05_ROUTER_V2_TRAINING.md`
7. `06_RAD_V2_VALIDATION.md`
8. `07_PARM_TARO_INTEGRATION.md`
9. `08_PARM_MULTI_OBJECTIVE_DATA.md`
10. `09_PARM_TARO_TRAINING.md`
11. `10_EVALUATION_AND_ABLATIONS.md`
12. `11_MASTER_EXECUTION_PLAN.md`

---

## Prompt cho AI

Đọc bộ hướng dẫn như specification của workspace `/home/jupyter-iec2024se10/Reward Decoding`. Giữ `router/` và `PARM/` bất biến; tạo `router_v2/` và `PARM_TARO/` mới. Reuse dataset/checkpoint hiện có trước khi tải lại. Thực hiện từng phase, kiểm tra code hiện tại trước khi sửa, viết test, báo cáo file thay đổi và dừng khi acceptance criteria chưa đạt.
