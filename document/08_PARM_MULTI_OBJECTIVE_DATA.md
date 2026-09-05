# 08 — PARM Multi-Objective Data in Current Workspace

## Goal

Reuse local PKU-SafeRLHF data trước khi tải/relabel lại.

## Existing local source

Workspace đã có:

```text
dataset/GenARM/PKU-SafeRLHF-10K/round0/
```

Đây là source đầu tiên phải inspect.

Ngoài ra kiểm tra:

```text
PARM/code/data/
```

để xem author-preprocessed `train/dev/test` đã tồn tại hay chưa.

## 1. Priority order

### Priority A

Nếu `PARM/code/data/` đã có đúng processed files:
- reuse;
- hash;
- không regenerate.

### Priority B

Nếu chưa có, nhưng `dataset/GenARM/PKU-SafeRLHF-10K/round0/` đủ field:
- tạo deterministic wrapper ngoài PARM;
- không download lại.

### Priority C

Chỉ khi local dataset thiếu:
- load `PKU-Alignment/PKU-SafeRLHF-10K`.

## 2. New prepared data path

Không ghi vào dataset GenARM.

Tạo:

```text
dataset/parm_taro/
├── train.json
├── validation.json
├── test.json
├── test_prompt_only.json
├── manifest.json
└── source_report.json
```

## 3. Objectives

Bảo toàn helpfulness và harmlessness fields.

Preference vector:

\[
\alpha=(\alpha_{help},\alpha_{safe}).
\]

## 4. Split

Mục tiêu tương thích PARM:

```text
8000 train
500 validation
~1500 test
```

Nếu local source đã có official/preexisting split thì ưu tiên reuse, không resplit tùy tiện.

## 5. No test tuning

Test chỉ dùng final evaluation.

## Acceptance criteria

- local PKU inspected first;
- no unnecessary download;
- deterministic provenance;
- no modification to `dataset/GenARM/...`;
- processed output isolated under `dataset/parm_taro/`.

---

## Prompt cho AI

Inspect `dataset/GenARM/PKU-SafeRLHF-10K/round0/` và `PARM/code/data/` trước. Reuse data local nếu đủ; chỉ download khi thiếu. Tạo processed deterministic data riêng dưới `dataset/parm_taro/`, giữ helpfulness/harmlessness, manifest/hashes và train/validation/test separation. Không sửa data GenARM hay PARM original.
