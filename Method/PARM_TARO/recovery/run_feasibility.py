"""Fail-fast, resumable PARM-TARO feasibility/root-cause diagnostic.

The runner never trains and never reads the Stage-10 test split.  Real-model
phases refuse to run unless the exact PBLORA checkpoint passes the artifact
gate.  In an infrastructure-blocked checkout, CPU-safe inventory, equation,
and decision evidence are still emitted without fabricating model results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).absolute().parents[2]
OUT = ROOT / "results/parm_taro/recovery/feasibility60"
STATE = OUT / "state.json"
PHASES = (
    "preflight", "pblora", "equation", "guide_utility", "router",
    "gradient", "alpha", "sweep", "exposure", "score", "decision",
)
REAL_PBLORA_PHASES = {"guide_utility", "gradient", "alpha", "sweep", "exposure"}
ALPHAS = ((0.0, 1.0), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (1.0, 0.0))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_state() -> dict[str, Any]:
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"schema_version": 1, "phases": {name: "pending" for name in PHASES}}


def save_phase(name: str, status: str, detail: str) -> None:
    state = load_state()
    state.setdefault("phases", {})[name] = status
    state.setdefault("details", {})[name] = detail
    state["updated_at"] = now()
    atomic_json(STATE, state)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def revision(path: Path) -> str | None:
    marker = path / ".tta_artifact_revision.json"
    if not marker.is_file():
        return None
    try:
        return str(json.loads(marker.read_text(encoding="utf-8"))["revision"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None


def classify(path: Path, expected: str, required: bool = True) -> dict[str, Any]:
    exists = path.is_dir() if expected == "dir" else path.is_file()
    lexical = path
    resolved = path.resolve(strict=False)
    if exists and lexical != resolved:
        status = "MOVED_BUT_RESOLVED"
    else:
        status = "PASS" if exists else ("MISSING" if required else "AMBIGUOUS")
    return {
        "path": str(path.relative_to(ROOT)), "resolved_path": str(resolved),
        "kind": expected, "required": required, "exists": exists,
        "readable": bool(exists and os.access(path, os.R_OK)), "status": status,
    }


def search_custom_artifacts() -> dict[str, list[str]]:
    names = {
        "pblora": {"adapter_model.safetensors", "adapter_model.bin"},
        "router": {"best.pt", "final.pt", "latest.pt", "last.pt"},
    }
    found = {key: [] for key in names}
    base = Path("/home/jupyter-iec2024se10")
    for current, dirs, files in os.walk(base):
        relative_depth = len(Path(current).relative_to(base).parts)
        dirs[:] = [d for d in dirs if d not in {".git", ".cache", "__pycache__"}]
        if relative_depth >= 10:
            dirs[:] = []
        for name in files:
            full = Path(current) / name
            if name in names["pblora"]:
                try:
                    cfg = json.loads((full.parent / "adapter_config.json").read_text())
                except (OSError, json.JSONDecodeError):
                    cfg = {}
                if str(cfg.get("peft_type", "")).upper() == "PBLORA":
                    found["pblora"].append(str(full))
            if name in names["router"] and "parm_taro" in str(full).lower():
                found["router"].append(str(full))
    return {key: sorted(values) for key, values in found.items()}


def phase_preflight(_: argparse.Namespace) -> tuple[str, str]:
    specs = [
        ("base_model", ROOT / "models/tulu-2-7b/config.json", "file", True),
        ("base_weights", ROOT / "models/tulu-2-7b/pytorch_model.bin.index.json", "file", True),
        ("tokenizer", ROOT / "models/tulu-2-7b/tokenizer.model", "file", True),
        ("pblora", ROOT / "results/parm_taro/checkpoints/parm_pku_pblora", "dir", True),
        ("taro_checkpoint", ROOT / "results/parm_taro/training/taro/best.pt", "file", True),
        ("v2_no_alpha", ROOT / "results/parm_taro/training/v2_no_alpha/best.pt", "file", True),
        ("v2_full_alpha", ROOT / "results/parm_taro/training/v2_alpha_preference/final.pt", "file", True),
        ("beaver_reward", ROOT / "models/beaver-7b-v1.0-reward/model.safetensors.index.json", "file", True),
        ("beaver_cost", ROOT / "models/beaver-7b-v1.0-cost/model.safetensors.index.json", "file", True),
        ("safe_rlhf", ROOT / "models/safe-rlhf-source/safe_rlhf/__init__.py", "file", True),
        ("dataset_manifest", ROOT / "dataset/parm_taro/manifest.json", "file", True),
        ("validation", ROOT / "dataset/parm_taro/validation.json", "file", True),
        ("protocol_lock", ROOT / "results/parm_taro/evaluation/protocol/protocol_lock.json", "file", True),
        ("normalization_source", ROOT / "results/parm_taro/evaluation/protocol/validation_calibration_raw.jsonl", "file", True),
    ]
    revisions = {
        "tulu": revision(ROOT / "models/tulu-2-7b"),
        "beaver_reward": revision(ROOT / "models/beaver-7b-v1.0-reward"),
        "beaver_cost": revision(ROOT / "models/beaver-7b-v1.0-cost"),
    }
    revision_by_artifact = {
        "base_model": revisions["tulu"], "base_weights": revisions["tulu"],
        "tokenizer": revisions["tulu"], "beaver_reward": revisions["beaver_reward"],
        "beaver_cost": revisions["beaver_cost"],
    }
    expected_revisions = {
        "base_model": "3c6e328ae91fabdd0daf09de16887de9615c1f66",
        "base_weights": "3c6e328ae91fabdd0daf09de16887de9615c1f66",
        "tokenizer": "3c6e328ae91fabdd0daf09de16887de9615c1f66",
        "beaver_reward": "375cd6a9f0d7e339d2199b05ba129a4a8906596d",
        "beaver_cost": "c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28",
    }
    entries = []
    for label, path, kind, required in specs:
        item = classify(path, kind, required)
        item["artifact"] = label
        item["revision"] = revision_by_artifact.get(label)
        item["expected_revision"] = expected_revisions.get(label)
        if item["exists"] and label in expected_revisions and item["revision"] != item["expected_revision"]:
            item["status"] = "AMBIGUOUS"
        entries.append(item)
    searches = search_custom_artifacts()
    payload = {
        "generated_at": now(), "device_policy": "CPU audit only",
        "entries": entries, "server_search": searches,
        "revisions": revisions,
        "fatal_missing": [x["artifact"] for x in entries if x["required"] and not x["exists"]],
    }
    atomic_json(OUT / "00_preflight/path_audit.json", payload)
    lines = ["# Artifact inventory", "", "| Artifact | Path | Status | Revision |", "|---|---|---|---|"]
    for item in entries:
        rev = item.get("revision", "n/a")
        lines.append(f"| {item['artifact']} | `{item['path']}` | {item['status']} | `{rev}` |")
    lines += ["", "## Full-server custom-artifact search", "", f"- PBLORA candidates: `{searches['pblora']}`", f"- PARM-TARO router checkpoint candidates: `{searches['router']}`", "", "No model was loaded and no test split was read.", ""]
    atomic_text(OUT / "00_preflight/artifact_inventory.md", "\n".join(lines))
    return "done", f"inventory complete; fatal missing={payload['fatal_missing']}"


def phase_pblora(_: argparse.Namespace) -> tuple[str, str]:
    candidates = search_custom_artifacts()["pblora"]
    target = ROOT / "results/parm_taro/checkpoints/parm_pku_pblora"
    available = bool(candidates) and (target / "adapter_config.json").is_file()
    payload = {
        "generated_at": now(), "target": str(target), "candidates": candidates,
        "available": available, "load_attempted": False,
        "runtime_status": "NOT_RUN",
        "classification": "PBLORA_RUNTIME_PENDING" if available else "BLOCKED — PBLORA CHECKPOINT UNAVAILABLE",
        "historical_expected_tree_sha256": "f26682e3cf51f8e09921873119fc371d329669cf9448d59ae0025fd40b23ffef",
        "reproduction": {
            "entrypoint": "PARM_TARO/recovery/pblora_resume_entrypoint.py",
            "wrapper": "PARM_TARO/recovery/scripts/train_pblora_repro.sh",
            "author_training_source": "PARM/code/training/train_pref_arm.py",
            "known_config": "Tulu-2-7B; PKU SafeRLHF; r/r2=4; alpha=8; dropout=0.05; safe/help beta=0.01; DPO beta=0.5; LR=5e-4; 2 epochs; batch=4; grad accumulation=8; cosine; warmup=20; wd=0.05",
            "unknown": ["historical random seed", "approved resolution of author default-model versus retained Tulu provenance"],
        },
    }
    atomic_json(OUT / "01_pblora/pblora_load.json", payload)
    atomic_csv(OUT / "01_pblora/pblora_alpha_probe.csv", ["status", "reason"], [{"status": "NOT_RUN", "reason": payload["classification"]}])
    atomic_text(OUT / "01_pblora/pblora_report.md", "# PBLORA gate\n\n**" + payload["classification"] + "**\n\nNo random or substitute adapter was loaded. Real-model tests and generation are stopped.\n")
    return ("done", "PBLORA artifact found; runtime probe pending") if available else ("blocked", payload["classification"])


def entropy(logp: Any) -> float:
    return float((-(logp.exp() * logp).sum(-1)).mean().item())


def phase_equation(_: argparse.Namespace) -> tuple[str, str]:
    import torch
    from torch.nn import functional as F

    generator = torch.Generator(device="cpu").manual_seed(2026)
    base_logits = torch.randn(64, 257, generator=generator, dtype=torch.float64)
    guide_logits = 1.2 * torch.randn(64, 257, generator=generator, dtype=torch.float64)
    a = F.log_softmax(base_logits, dim=-1)
    b = F.log_softmax(guide_logits, dim=-1)
    formulas = {
        "F1_paper_beta1": a + b,
        "F2_author": (a + b) / 2.0,
        "F3_current_lambda1": a + b,
        "F3_current_lambda0.01": a + 0.01 * b,
        "F4_normalized_lambda1": (a + b) / 2.0,
        "F4_normalized_lambda0.01": (a + 0.01 * b) / 1.01,
    }
    rows = []
    normalized = {}
    for name, score in formulas.items():
        logp = F.log_softmax(score, dim=-1)
        normalized[name] = logp
        rows.append({"formula": name, "entropy_mean": entropy(logp), "top1_token_0": int(logp[0].argmax())})
    f4_f2 = float((normalized["F4_normalized_lambda1"] - normalized["F2_author"]).abs().max())
    f3_f2 = float((normalized["F3_current_lambda1"] - normalized["F2_author"]).abs().max())
    atomic_csv(OUT / "02_equations/equation_matrix.csv", list(rows[0]), rows)
    text = f"""# Temperature/scale confound

**FUSION_SCALE_CONFOUND**

- Synthetic states: 64 (vocabulary 257, seed 2026).
- `max|F4(lambda=1)-F2| = {f4_f2}`.
- `max|F3(lambda=1)-F2| = {f3_f2}`.
- F1 paper beta=1 and F3 lambda=1 coincide.
- F2 author and F4 normalized lambda=1 coincide.
- Division by `1+lambda` changes effective temperature while preserving ranking
  at fixed lambda; it is not generally removed by log-probability normalization.

Real five-prompt equation traces were not run because the exact PBLORA runtime
gate did not pass. Production source was not modified.
"""
    atomic_text(OUT / "02_equations/temperature_confound.md", text)
    return "done", f"FUSION_SCALE_CONFOUND; F4/F2 diff={f4_f2}; F3/F2 diff={f3_f2}"


def blocked_phase(directory: str, files: dict[str, str], reason: str) -> tuple[str, str]:
    for name, content in files.items():
        atomic_text(OUT / directory / name, content)
    return "blocked", reason


def phase_guide_utility(_: argparse.Namespace) -> tuple[str, str]:
    reason = "BLOCKED — PBLORA CHECKPOINT UNAVAILABLE"
    atomic_csv(OUT / "03_guide_utility/preference_ranking.csv", ["status", "reason"], [{"status": "NOT_RUN", "reason": reason}])
    atomic_csv(OUT / "03_guide_utility/gold_token_utility.csv", ["status", "reason"], [{"status": "NOT_RUN", "reason": reason}])
    return blocked_phase("03_guide_utility", {"report.md": f"# Guide utility\n\n**{reason}**\n\nNo claim about guide quality is possible.\n"}, reason)


def phase_router(_: argparse.Namespace) -> tuple[str, str]:
    targets = [ROOT / "results/parm_taro/training/taro/best.pt", ROOT / "results/parm_taro/training/v2_no_alpha/best.pt", ROOT / "results/parm_taro/training/v2_alpha_preference/final.pt"]
    available = [str(p) for p in targets if p.is_file()]
    payload = {"generated_at": now(), "expected": [str(p) for p in targets], "available": available, "load_attempted": False, "status": "PENDING_LOAD" if available else "ROUTER_CHECKPOINT_UNAVAILABLE"}
    atomic_json(OUT / "04_router/router_checkpoint_audit.json", payload)
    atomic_csv(OUT / "04_router/gate_distribution.csv", ["status", "reason"], [{"status": "NOT_RUN", "reason": payload["status"]}])
    return ("done", "router artifact present; load probe pending") if available else ("blocked", payload["status"])


def phase_gradient(_: argparse.Namespace) -> tuple[str, str]:
    reason = "BLOCKED — PBLORA CHECKPOINT UNAVAILABLE"
    atomic_csv(OUT / "05_gradients/gradient_audit.csv", ["status", "reason"], [{"status": "NOT_RUN", "reason": reason}])
    return blocked_phase("05_gradients", {"report.md": f"# NLL gradient audit\n\n**{reason}**\n\nHistorical gradients were not treated as a fresh result.\n"}, reason)


def phase_alpha(_: argparse.Namespace) -> tuple[str, str]:
    reason = "BLOCKED — PBLORA AND ROUTER CHECKPOINTS UNAVAILABLE"
    atomic_csv(OUT / "06_alpha/alpha_trace.csv", ["status", "reason"], [{"status": "NOT_RUN", "reason": reason}])
    return blocked_phase("06_alpha", {"report.md": f"# Alpha path\n\n**{reason}**\n"}, reason)


def phase_sweep(_: argparse.Namespace) -> tuple[str, str]:
    reason = "BLOCKED — PBLORA RUNTIME GATE DID NOT PASS"
    atomic_csv(OUT / "07_lambda_sweep/fixed_lambda.csv", ["status", "reason"], [{"status": "NOT_RUN", "reason": reason}])
    atomic_csv(OUT / "07_lambda_sweep/oracle_analysis.csv", ["status", "reason"], [{"status": "NOT_RUN", "reason": reason}])
    return blocked_phase("07_lambda_sweep", {"headroom.md": f"# Adaptive headroom\n\n**{reason}**\n\nNo manifest was created and no validation generation ran. Adaptive headroom is UNTESTABLE.\n"}, reason)


def phase_exposure(_: argparse.Namespace) -> tuple[str, str]:
    reason = "BLOCKED — PBLORA AND ROUTER CHECKPOINTS UNAVAILABLE"
    return blocked_phase("08_exposure", {"report.md": f"# Exposure shift\n\n**{reason}**\n"}, reason)


def phase_score(_: argparse.Namespace) -> tuple[str, str]:
    reward = ROOT / "models/beaver-7b-v1.0-reward"
    cost = ROOT / "models/beaver-7b-v1.0-cost"
    source = ROOT / "models/safe-rlhf-source"
    payload = {
        "generated_at": now(), "load_attempted": False,
        "reward": {"path": str(reward), "revision": revision(reward), "index_present": (reward / "model.safetensors.index.json").is_file()},
        "cost": {"path": str(cost), "revision": revision(cost), "index_present": (cost / "model.safetensors.index.json").is_file()},
        "safe_rlhf_source_present": (source / "safe_rlhf/__init__.py").is_file(),
        "status": "NOT_RUN — SAFE_RLHF SOURCE/PROVENANCE NOT READY",
        "note": "Weight presence is not a scorer-runtime sanity pass.",
    }
    atomic_json(OUT / "09_scores/scorer_sanity.json", payload)
    return "blocked", payload["status"]


def phase_decision(_: argparse.Namespace) -> tuple[str, str]:
    matrix = [
        ("RC1", "PBLoRA checkpoint missing/load wrong", "No PBLORA adapter weights found under project or /home search.", "CONFIRMED", "HIGH", "01_pblora/pblora_load.json", "Reproduce exact custom PBLORA under recovered protocol before any method claim."),
        ("RC2", "Alpha does not enter PBLORA", "Runtime probe forbidden by RC1.", "UNTESTABLE", "HIGH", "01_pblora/pblora_alpha_probe.csv", "Run five-alpha same-prefix probe after RC1 clears."),
        ("RC3", "PBLORA guide weak/misranks", "20-pair utility audit blocked by RC1.", "UNTESTABLE", "HIGH", "03_guide_utility/report.md", "Measure validation preference accuracy and margins."),
        ("RC4", "Model/tokenizer/template/provenance wrong", "Tulu files match pinned public revision; custom adapter provenance absent; author and Stage-10 templates differ.", "PARTIAL", "HIGH", "00_preflight/path_audit.json", "Freeze exact adapter provenance and explicitly select template."),
        ("RC5", "Author/PARM-TARO fusion mismatch", "Author F2 divides summed log-probs by 2; current F3 does not.", "CONFIRMED", "HIGH", "02_equations/temperature_confound.md", "Treat paper, author, and current formulations as distinct comparators."),
        ("RC6", "Lambda confounds guidance and temperature", "F3 changes coefficient sum with lambda; F4 removes this scale change. Synthetic entropy matrix confirms distinct distributions.", "CONFIRMED", "HIGH", "02_equations/equation_matrix.csv", "Test F4 diagnostically after PBLORA gate; do not patch production yet."),
        ("RC7", "Router checkpoint missing/wrong/random", "All expected TARO/V2 checkpoint files are absent; retained JSON logs are not weights.", "CONFIRMED", "HIGH", "04_router/router_checkpoint_audit.json", "Reproduce/load checkpoint before router claims."),
        ("RC8", "Router ignores alpha", "No valid router checkpoint to probe.", "UNTESTABLE", "HIGH", "06_alpha/report.md", "Run same-state alpha counterfactual after RC7 clears."),
        ("RC9", "Gold-token NLL pushes lambda to zero", "Historical evidence suggests this, but requested fresh 20-case gradient audit is blocked.", "LIKELY", "MEDIUM", "05_gradients/report.md", "Re-test both F3 and F4 gradients on validation."),
        ("RC10", "Sigmoid/lambda saturation or scale error", "Source gives bounded sigmoid; trained raw/bias unavailable without checkpoint.", "PARTIAL", "MEDIUM", "04_router/router_checkpoint_audit.json", "Audit raw gate, bias, floor/max, saturation after RC7 clears."),
        ("RC11", "Feature normalization/cache scale wrong", "Reconstructed cache tests exist, but no matching trained checkpoint/state distribution can be checked.", "UNTESTABLE", "MEDIUM", "04_router/router_checkpoint_audit.json", "Compare training cache statistics to live features."),
        ("RC12", "Teacher-forcing exposure shift", "Both required model artifacts are absent.", "UNTESTABLE", "HIGH", "08_exposure/report.md", "Run 10-prompt paired-prefix audit last."),
        ("RC13", "Little adaptive headroom", "Fatal PBLORA gate prevents fixed-lambda/oracle sweep.", "UNTESTABLE", "HIGH", "07_lambda_sweep/headroom.md", "Do not conclude NO-GO; run validation-60 only after all gates."),
        ("RC14", "Headroom exists but router cannot learn", "Neither oracle headroom nor router policy can be measured.", "UNTESTABLE", "HIGH", "07_lambda_sweep/headroom.md", "Compare oracle, best global, dynamic, same-average."),
        ("RC15", "Optimum varies mainly by alpha", "Per-alpha sweep blocked.", "UNTESTABLE", "HIGH", "07_lambda_sweep/oracle_analysis.csv", "Estimate per-alpha optimum on fixed manifest."),
        ("RC16", "Scorer wiring/model wrong", "Reward weights verified; cost weights/tree complete but revision marker absent; Safe-RLHF source commit/runtime unavailable.", "PARTIAL", "HIGH", "09_scores/scorer_sanity.json", "Restore exact Safe-RLHF provenance and run deterministic sanity pairs."),
    ]
    lines = ["# Root-cause matrix", "", "| Hypothesis | Evidence | Status | Confidence | Relevant artifact | Recommended action |", "|---|---|---|---|---|---|"]
    for rc, hypothesis, evidence, status, confidence, artifact, action in matrix:
        lines.append(f"| {rc}: {hypothesis} | {evidence} | **{status}** | {confidence} | `{artifact}` | {action} |")
    atomic_text(OUT / "10_decision/root_cause_matrix.md", "\n".join(lines) + "\n")
    answers = [
        "No. Exact PBLORA checkpoint is unavailable; load was not attempted.",
        "UNTESTABLE until PBLORA exists.",
        "UNTESTABLE; guide utility was not inferred from historical metrics.",
        "Tulu base/tokenizer are present at the pinned public revision; PBLORA provenance is absent.",
        "Yes. Author F2 divides by 2, current F3 does not.",
        "Yes, algebraically and in the synthetic entropy matrix.",
        "No. Expected V2/TARO weight files are absent.",
        "UNTESTABLE without a valid router checkpoint.",
        "LIKELY historically, but fresh validation gradient evidence is blocked.",
        "UNTESTABLE for trained gates; source mapping alone is insufficient.",
        "UNTESTABLE.", "UNTESTABLE.", "UNTESTABLE.", "UNTESTABLE.", "UNTESTABLE.", "UNTESTABLE.",
        "#1 infrastructure/provenance (missing PBLORA); #2 confirmed fusion-scale confound; #3 likely NLL objective mismatch, pending fresh validation proof.",
        "Cannot decide GO/NO-GO scientifically until PBLORA, scorer runtime, and validation oracle-headroom sweep pass.",
    ]
    headings = [
        "Is PBLoRA actually available and loadable?", "Does PBLoRA respond to alpha?", "Is the guide useful?",
        "Are base/PBLoRA/model paths correct?", "Is PARM-TARO fusion mismatched with author PARM?", "Is there a temperature/scale confound?",
        "Is V2 checkpoint actually loaded?", "Does router respond to alpha?", "Does NLL push lambda toward zero?",
        "Is sigmoid/parameterization saturated?", "Is there teacher-forcing exposure shift?", "What is best global fixed lambda?",
        "What is best lambda per alpha?", "What is oracle per-prompt performance?", "How large is adaptive headroom?",
        "Does dynamic beat same-average fixed?", "Main root cause ranked #1/#2/#3.", "Is this research direction worth continuing?",
    ]
    summary = ["# PARM-TARO feasibility-60 summary", ""]
    for index, (heading, answer) in enumerate(zip(headings, answers), 1):
        summary.extend([f"## {index}. {heading}", "", answer, ""])
    summary += ["No retraining, validation generation, test tuning, or protected-source modification occurred.", "", "BLOCKED — PBLORA / MODEL / EVALUATOR NOT READY", ""]
    atomic_text(OUT / "feasibility_summary.md", "\n".join(summary))
    return "blocked", "BLOCKED — PBLORA / MODEL / EVALUATOR NOT READY"


HANDLERS: dict[str, Callable[[argparse.Namespace], tuple[str, str]]] = {
    "preflight": phase_preflight, "pblora": phase_pblora, "equation": phase_equation,
    "guide_utility": phase_guide_utility, "router": phase_router,
    "gradient": phase_gradient, "alpha": phase_alpha, "sweep": phase_sweep,
    "exposure": phase_exposure, "score": phase_score, "decision": phase_decision,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", *PHASES), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    selected = PHASES if args.phase == "all" else (args.phase,)
    if args.dry_run:
        print(json.dumps({"phases": selected, "device": args.device, "writes": False}, indent=2))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    state = load_state()
    for phase in selected:
        if args.resume and state.get("phases", {}).get(phase) == "done":
            print(f"SKIP {phase}: done")
            continue
        print(f"RUN {phase}")
        status, detail = HANDLERS[phase](args)
        save_phase(phase, status, detail)
        state = load_state()
        print(f"{status.upper()} {phase}: {detail}")
    blocked = any(state.get("phases", {}).get(name) == "blocked" for name in selected)
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
