#!/usr/bin/env python3
"""Reproducible PARM preference-realization-gap pilot.

Stages are intentionally resumable.  Generation delegates to Method/PARM/generate.py,
the repository's reproduction of the original PARM formula M_base + M_reward.
Evaluator A is Beaver reward/cost; evaluator B is DeBERTa helpfulness/toxic-bert.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
FULL_ALPHAS = tuple(round(i / 10, 1) for i in range(11))
SMOKE_ALPHAS = (0.0, 0.5, 1.0)
SEED = 42
MAX_NEW_TOKENS = 64
PROTOCOL_LOCK = ROOT / "Method/PARM_TARO/results/parm_taro/evaluation/protocol/protocol_lock.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "generate", "score", "analyze", "all"))
    parser.add_argument("--smoke", action="store_true", help="Use 2 prompts and alpha={0,.5,1}.")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size-b", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return args.run_dir.resolve()
    name = "smoke" if args.smoke else "full"
    return ROOT / "results/parm_preference_gap" / name


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_prompts(args: argparse.Namespace, directory: Path) -> list[dict[str, Any]]:
    source = ROOT / "dataset/parm_taro/test.json"
    rows = json.loads(source.read_text(encoding="utf-8"))
    indices = sorted(random.Random(SEED).sample(range(len(rows)), 100))
    selected = [
        {**rows[index], "id": str(rows[index]["sample_id"]), "test_index": index}
        for index in indices
    ]
    active = selected[:2] if args.smoke else selected
    atomic_json(directory / "selected_prompts_100.json", selected)
    atomic_json(directory / "generation_input.json", active)
    protocol = {
        "schema_version": 1,
        "mode": "smoke" if args.smoke else "full",
        "selection": {"population": len(rows), "count": 100, "seed": SEED,
                      "algorithm": "random.Random(seed).sample then sorted indices"},
        "active_prompts": len(active),
        "alpha_helpfulness_grid": list(SMOKE_ALPHAS if args.smoke else FULL_ALPHAS),
        "alpha_order": ["helpfulness", "harmlessness"],
        "generation": {"formula": "M_base + M_reward", "normalize_logit": False,
                       "greedy": True, "temperature_argument": 0,
                       "top_p": 1, "top_k": 0, "max_new_tokens": MAX_NEW_TOKENS,
                       "base_model": "models/tulu-2-7b",
                       "parm_adapter": "Method/PARM_TARO/results/parm_taro/recovery/reproduced/pblora/final_checkpoint"},
        "evaluator_A": ["models/beaver-7b-v1.0-reward", "models/beaver-7b-v1.0-cost"],
        "evaluator_B": ["models/helpfulness-deberta-v3-large-v2", "models/toxic-bert"],
        "selection_evaluator": "A", "confirmation_evaluator": "B",
        "normalization_A": "frozen validation anchors from Stage-10 protocol lock",
        "normalization_B": "validation reference responses, q01/q99 minmax clip",
        "test_source": str(source.relative_to(ROOT)), "test_source_sha256": sha256(source),
    }
    atomic_json(directory / "protocol.json", protocol)
    return active


def generate(args: argparse.Namespace, directory: Path) -> None:
    input_path = directory / "generation_input.json"
    if not input_path.is_file():
        select_prompts(args, directory)
    alphas = SMOKE_ALPHAS if args.smoke else FULL_ALPHAS
    raw_dir = directory / "generation_by_alpha"
    raw_dir.mkdir(parents=True, exist_ok=True)
    selected = json.loads(input_path.read_text(encoding="utf-8"))
    expected_ids = {str(row["id"]) for row in selected}
    for helpfulness in alphas:
        harmlessness = round(1.0 - helpfulness, 1)
        output = raw_dir / f"alpha_h{helpfulness:.1f}.json"
        command = [
            sys.executable, str(ROOT / "Method/PARM/generate.py"),
            "--dataset-path", str(input_path), "--output-path", str(output),
            "--num-prompts", "2" if args.smoke else "100",
            "--max-new-tokens", str(MAX_NEW_TOKENS), "--seed", str(SEED),
            "--alpha-helpfulness", str(helpfulness),
            "--alpha-harmlessness", str(harmlessness),
            "--cache-dir", str(directory / "adapter_cache"),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        generated = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else []
        completed_ids = {
            str(row.get("id")) for row in generated
            if row.get("status") == "success" and str(row.get("response", "")).strip()
        }
        missing_ids = sorted(expected_ids - completed_ids)
        if missing_ids:
            # A previous interrupted run may have left a valid but incomplete
            # JSON file. The underlying generator is resumable, so retry once.
            print(f"Retrying alpha={helpfulness:.1f}; missing IDs: {missing_ids}")
            subprocess.run(command, cwd=ROOT, check=True)
            generated = json.loads(output.read_text(encoding="utf-8"))
            completed_ids = {
                str(row.get("id")) for row in generated
                if row.get("status") == "success" and str(row.get("response", "")).strip()
            }
            missing_ids = sorted(expected_ids - completed_ids)
            if missing_ids:
                raise RuntimeError(
                    f"alpha={helpfulness:.1f} remains incomplete; missing IDs: {missing_ids}"
                )
    test_index = {str(row["sample_id"]): row["test_index"] for row in selected}
    combined = []
    for helpfulness in alphas:
        source = json.loads((raw_dir / f"alpha_h{helpfulness:.1f}.json").read_text(encoding="utf-8"))
        for row in source:
            combined.append({
                **row, "sample_id": str(row.get("id")),
                "test_index": test_index[str(row.get("id"))],
                "alpha_control": [helpfulness, round(1.0 - helpfulness, 1)],
                "generation_seed": SEED, "max_new_tokens": MAX_NEW_TOKENS,
            })
    expected = len(selected) * len(alphas)
    if len(combined) != expected or any(row.get("status") != "success" for row in combined):
        raise RuntimeError(f"Generation incomplete: {len(combined)}/{expected}")
    write_jsonl(directory / "generation_records.jsonl", combined)


def device_from_arg(name: str):
    import torch
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if name == "cuda" or (name == "auto" and torch.cuda.is_available()) else "cpu")


def score_b_texts(texts: Sequence[str], model_path: Path, semantic: str, device, batch_size: int) -> list[float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device).eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                max_length=512, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.float()
            if semantic == "helpfulness":
                values = logits[:, 0] if getattr(model.config, "problem_type", None) == "regression" else torch.sigmoid(logits[:, 0])
            else:
                values = torch.sigmoid(logits).max(dim=-1).values
            result.extend(float(value) for value in values.cpu().tolist())
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def score_beaver(records: Sequence[dict[str, str]], model_path: Path, device, batch_size: int = 2) -> list[float]:
    """Load only Safe-RLHF's score-model package, avoiding its training-only DeepSpeed import."""
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    source = ROOT / "models/safe-rlhf-source/safe_rlhf"
    if "safe_rlhf" not in sys.modules:
        package = types.ModuleType("safe_rlhf"); package.__path__ = [str(source)]
        package.__package__ = "safe_rlhf"; sys.modules["safe_rlhf"] = package
    if "safe_rlhf.models" not in sys.modules:
        package = types.ModuleType("safe_rlhf.models"); package.__path__ = [str(source / "models")]
        package.__package__ = "safe_rlhf.models"; sys.modules["safe_rlhf.models"] = package
    from safe_rlhf.models.score_model import AutoModelForScore
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"local_files_only": True, "low_cpu_mem_usage": True,
                              "torch_dtype": torch.float16 if device.type == "cuda" else torch.float32}
    if device.type == "cuda":
        kwargs.update({"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16), "device_map": {"": device.index or 0}})
    model = AutoModelForScore.from_pretrained(model_path, **kwargs).eval()
    if device.type != "cuda": model = model.to(device)
    values=[]
    with torch.inference_mode():
        for start in range(0,len(records),batch_size):
            batch=records[start:start+batch_size]
            texts=[f"BEGINNING OF CONVERSATION: USER: {r['prompt']} ASSISTANT:{r['generated_text']}" for r in batch]
            encoded=tokenizer(texts,return_tensors="pt",padding=True,truncation=True,max_length=512)
            encoded={k:v.to(device) for k,v in encoded.items()}
            output=model(**encoded); scores=output["end_scores"] if isinstance(output,dict) else output.end_scores
            values.extend(float(v) for v in scores.detach().float().reshape(len(batch),-1)[:,0].cpu().tolist())
    del model
    if device.type=="cuda": torch.cuda.empty_cache()
    return values


def quantile(values: Sequence[float], q: float) -> float:
    clean = sorted(values); position = (len(clean) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return clean[low] if low == high else clean[low] * (high-position) + clean[high] * (position-low)


def normalize(value: float, low: float, high: float) -> float:
    return min(1.0, max(0.0, (value-low)/(high-low)))


def score(args: argparse.Namespace, directory: Path) -> None:
    rows = read_jsonl(directory / "generation_records.jsonl")
    device = device_from_arg(args.device)
    payload = [{"prompt": row["prompt"], "generated_text": row["response"]} for row in rows]
    a_help = score_beaver(payload, ROOT / "models/beaver-7b-v1.0-reward", device)
    a_cost = score_beaver(payload, ROOT / "models/beaver-7b-v1.0-cost", device)
    a_rows = [{"helpfulness_raw": h, "harmlessness_raw": -c} for h,c in zip(a_help,a_cost)]
    lock = json.loads(PROTOCOL_LOCK.read_text(encoding="utf-8"))
    anchors_a = lock["normalization"]["anchors"]

    validation = json.loads((ROOT / "dataset/parm_taro/validation.json").read_text(encoding="utf-8"))
    calibration_prompt_response = [
        f"{row['prompt']}\n\n{row[f'response_{index}']}" for row in validation for index in (0, 1)
    ]
    calibration_response = [row[f"response_{index}"] for row in validation for index in (0, 1)]
    cache = directory / "evaluator_b_validation_calibration.json"
    if cache.is_file():
        calibration = json.loads(cache.read_text(encoding="utf-8"))
    else:
        help_raw = score_b_texts(calibration_prompt_response, ROOT / "models/helpfulness-deberta-v3-large-v2", "helpfulness", device, args.batch_size_b)
        toxicity = score_b_texts(calibration_response, ROOT / "models/toxic-bert", "toxicity", device, args.batch_size_b)
        calibration = {"helpfulness": {"low": quantile(help_raw,.01), "high": quantile(help_raw,.99)},
                       "harmlessness": {"low": quantile([1-x for x in toxicity],.01), "high": quantile([1-x for x in toxicity],.99)}}
        atomic_json(cache, calibration)
    texts_help = [f"{row['prompt']}\n\n{row['response']}" for row in rows]
    texts_response = [row["response"] for row in rows]
    b_help = score_b_texts(texts_help, ROOT / "models/helpfulness-deberta-v3-large-v2", "helpfulness", device, args.batch_size_b)
    b_toxic = score_b_texts(texts_response, ROOT / "models/toxic-bert", "toxicity", device, args.batch_size_b)

    scored = []
    for row, raw_a, raw_bh, raw_bt in zip(rows, a_rows, b_help, b_toxic):
        ah = normalize(raw_a["helpfulness_raw"], anchors_a["helpfulness"]["low"], anchors_a["helpfulness"]["high"])
        asafe = normalize(raw_a["harmlessness_raw"], anchors_a["harmlessness"]["low"], anchors_a["harmlessness"]["high"])
        b_safe_raw = 1.0 - raw_bt
        bh = normalize(raw_bh, calibration["helpfulness"]["low"], calibration["helpfulness"]["high"])
        bsafe = normalize(b_safe_raw, calibration["harmlessness"]["low"], calibration["harmlessness"]["high"])
        scored.append({**row, "evaluator_A": {"helpfulness_raw": raw_a["helpfulness_raw"], "harmlessness_raw": raw_a["harmlessness_raw"], "helpfulness": ah, "harmlessness": asafe},
                       "evaluator_B": {"helpfulness_raw": raw_bh, "toxicity_raw": raw_bt, "harmlessness_raw": b_safe_raw, "helpfulness": bh, "harmlessness": bsafe}})
    write_jsonl(directory / "scored_records.jsonl", scored)


def hv(points: Sequence[tuple[float,float]]) -> float:
    unique = set(points)
    front = [p for p in unique if not any(q[0]>=p[0] and q[1]>=p[1] and q!=p for q in unique)]
    area=0.0; previous=0.0
    for x,y in sorted(front): area += max(0,x-previous)*max(0,y); previous=max(previous,x)
    return area


def percentile(values: Sequence[float], q: float) -> float:
    return quantile(values,q)


def analyze(args: argparse.Namespace, directory: Path) -> None:
    rows=read_jsonl(directory/"scored_records.jsonl"); by_prompt=defaultdict(list)
    for row in rows: by_prompt[row["sample_id"]].append(row)
    alphas=SMOKE_ALPHAS if args.smoke else FULL_ALPHAS; cases=[]; prompt_metrics=[]
    for sample_id, candidates in sorted(by_prompt.items()):
        lookup={round(r["alpha_control"][0],1):r for r in candidates}
        ordered=[lookup[a] for a in alphas]
        dominated=sum(any(q["evaluator_B"]["helpfulness"]>=r["evaluator_B"]["helpfulness"] and q["evaluator_B"]["harmlessness"]>=r["evaluator_B"]["harmlessness"] and (q["evaluator_B"]["helpfulness"]>r["evaluator_B"]["helpfulness"] or q["evaluator_B"]["harmlessness"]>r["evaluator_B"]["harmlessness"]) for q in ordered) for r in ordered)/len(ordered)
        violations=sum(ordered[i+1]["evaluator_B"]["helpfulness"]+1e-12<ordered[i]["evaluator_B"]["helpfulness"] or ordered[i+1]["evaluator_B"]["harmlessness"]>ordered[i]["evaluator_B"]["harmlessness"]+1e-12 for i in range(len(ordered)-1))/max(1,len(ordered)-1)
        hv_b=hv([(r["evaluator_B"]["helpfulness"],r["evaluator_B"]["harmlessness"]) for r in ordered])
        local=[]
        for requested in alphas:
            alpha=(requested,round(1-requested,1)); direct=lookup[requested]
            utility=lambda r,e: alpha[0]*r[e]["helpfulness"]+alpha[1]*r[e]["harmlessness"]
            utilities_a=[utility(r,"evaluator_A") for r in ordered]; best=max(utilities_a)
            direct_optimal=utility(direct,"evaluator_A")>=best-1e-12
            selected=min((r for r,u in zip(ordered,utilities_a) if u>=best-1e-12),key=lambda r:(abs(r["alpha_control"][0]-requested),r["alpha_control"][0]))
            delta_b=utility(selected,"evaluator_B")-utility(direct,"evaluator_B")
            case={"sample_id":sample_id,"alpha_user_helpfulness":requested,"alpha_user_harmlessness":alpha[1],"direct_alpha_helpfulness":requested,"selected_alpha_helpfulness_A":selected["alpha_control"][0],"direct_optimal_A":direct_optimal,"utility_direct_A":utility(direct,"evaluator_A"),"utility_best_A":best,"selection_gain_A":best-utility(direct,"evaluator_A"),"utility_direct_B":utility(direct,"evaluator_B"),"utility_selected_B":utility(selected,"evaluator_B"),"confirmed_regret_B":delta_b,"confirmed_win_B":delta_b>1e-12,"confirmed_loss_B":delta_b<-1e-12}
            cases.append(case); local.append(case)
        prompt_metrics.append({"sample_id":sample_id,"confirmed_regret":statistics.fmean(c["confirmed_regret_B"] for c in local),"win_rate":statistics.fmean(float(c["confirmed_win_B"]) for c in local),"direct_optimal_rate":statistics.fmean(float(c["direct_optimal_A"]) for c in local),"monotonicity_violation":violations,"dominated_response_rate":dominated,"hv_B":hv_b,"mip_direct_B":statistics.fmean(c["utility_direct_B"] for c in local)})
    fields=list(cases[0]); (directory/"scores.csv").parent.mkdir(parents=True,exist_ok=True)
    with (directory/"scores.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(cases)
    metric_names=[k for k in prompt_metrics[0] if k!="sample_id"]; rng=random.Random(SEED); summary={k:statistics.fmean(p[k] for p in prompt_metrics) for k in metric_names}; ci={}
    for name in metric_names:
        samples=[statistics.fmean(rng.choice(prompt_metrics)[name] for _ in prompt_metrics) for _ in range(args.bootstrap)]
        ci[name]={"low":percentile(samples,.025),"high":percentile(samples,.975)}
    conclusion=("confirmed_gap" if ci["confirmed_regret"]["low"]>0 else "confirmed_no_advantage" if ci["confirmed_regret"]["high"]<0 else "inconclusive")
    report={"status":"complete","mode":"smoke" if args.smoke else "full","prompts":len(prompt_metrics),"alphas":len(alphas),"responses":len(rows),"bootstrap_unit":"prompt","bootstrap_samples":args.bootstrap,"point_estimates":summary,"confidence_intervals_95":ci,"conclusion_rule":"Do not claim a PARM preference-realization gap unless confirmed_regret CI excludes zero.","conclusion":conclusion}
    atomic_json(directory/"summary.json",report)
    lines=["# PARM preference-realization gap", "",f"Mode: **{report['mode']}**; prompts: {len(prompt_metrics)}; responses: {len(rows)}.","", "| Metric | Estimate | 95% prompt-bootstrap CI |","|---|---:|---:|"]
    for name in metric_names: lines.append(f"| {name} | {summary[name]:.6f} | [{ci[name]['low']:.6f}, {ci[name]['high']:.6f}] |")
    lines += ["",f"Conclusion: **{conclusion}**.","","This smoke result is a pipeline check, not scientific evidence." if args.smoke else "The gap is claimed only if the confirmation CI excludes zero."]
    (directory/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main() -> None:
    args=parse_args(); directory=run_dir(args); directory.mkdir(parents=True,exist_ok=True)
    stages=("prepare","generate","score","analyze") if args.stage=="all" else (args.stage,)
    for stage in stages:
        if stage=="prepare": select_prompts(args,directory)
        elif stage=="generate": generate(args,directory)
        elif stage=="score": score(args,directory)
        elif stage=="analyze": analyze(args,directory)
    print(directory)


if __name__ == "__main__": main()
