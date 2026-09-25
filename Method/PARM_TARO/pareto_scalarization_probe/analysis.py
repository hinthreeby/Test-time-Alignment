from __future__ import annotations
import csv, json, math
from collections import defaultdict
from statistics import fmean
from PARM_TARO.evaluation.metrics import hypervolume_2d
from .config import ALPHA_GRID, DENSE_OUT, PROTOCOL
from .geometry import local_curvature, nondominated, spearman, support_interval
from .io import atomic_csv, atomic_json, read_jsonl

def normalize(value: float, low: float, high: float) -> float: return max(0., min(1., (value-low)/(high-low)))
def _score_map(kind: str) -> dict[str, float]:
    with (DENSE_OUT / f"{kind}_scores.csv").open(newline="", encoding="utf-8") as handle:
        return {row["case_id"]: float(row[f"{kind}_score"]) for row in csv.DictReader(handle)}

def run() -> dict:
    generations = read_jsonl(DENSE_OUT / "generations.jsonl"); reward, cost = _score_map("reward"), _score_map("cost")
    lock = json.loads(PROTOCOL.read_text()); anchors = lock["normalization"]["anchors"]; records = []
    for row in generations:
        h = normalize(reward[row["case_id"]], float(anchors["helpfulness"]["low"]), float(anchors["helpfulness"]["high"]))
        sraw = -cost[row["case_id"]]; s = normalize(sraw, float(anchors["harmlessness"]["low"]), float(anchors["harmlessness"]["high"]))
        ah, safe = map(float, row["requested_alpha"]); mip = ah*h + safe*s; denom = math.hypot(ah, safe)*math.hypot(h, s)
        records.append({"case_id": row["case_id"], "sample_id": row["sample_id"], "prompt_sha256": row["prompt_sha256"],
                        "alpha_helpfulness": ah, "alpha_harmlessness": safe, "helpfulness_raw": reward[row["case_id"]],
                        "harmlessness_raw": sraw, "helpfulness": h, "harmlessness": s, "MIP": mip, "PCS": 0. if denom == 0 else mip/denom})
    atomic_csv(DENSE_OUT / "dense_alpha_records.csv", records)
    grouped = defaultdict(list)
    for row in records: grouped[row["alpha_helpfulness"]].append(row)
    summary_rows = []
    for ah, safe in ALPHA_GRID:
        group = grouped[ah]; summary_rows.append({"alpha_helpfulness": ah, "alpha_harmlessness": safe,
            "mean_helpfulness": fmean(r["helpfulness"] for r in group), "mean_harmlessness": fmean(r["harmlessness"] for r in group),
            "mean_MIP": fmean(r["MIP"] for r in group), "mean_PCS": fmean(r["PCS"] for r in group), "records": len(group)})
    points = [(r["mean_helpfulness"], r["mean_harmlessness"]) for r in summary_rows]; nd = nondominated(points); curves = local_curvature(points)
    geometry = []
    for i, row in enumerate(summary_rows):
        interval = support_interval(i, points) if nd[i] else None
        strictly_positive = interval is not None and max(interval[0], 0.0) < min(interval[1], 1.0)
        if i == 0: dh = ds = None
        else:
            step = row["alpha_helpfulness"] - summary_rows[i-1]["alpha_helpfulness"]
            dh = (row["mean_helpfulness"]-summary_rows[i-1]["mean_helpfulness"])/step
            ds = (row["mean_harmlessness"]-summary_rows[i-1]["mean_harmlessness"])/step
        geometry.append({**row, "d_helpfulness_d_alpha": dh, "d_harmlessness_d_alpha": ds,
                         "helpfulness_monotonicity_violation": dh is not None and dh < 0,
                         "harmlessness_monotonicity_violation": ds is not None and ds > 0,
                         "low_sensitivity": dh is not None and abs(dh) < .02 and abs(ds) < .02,
                         "nondominated": nd[i], "supported_positive_linear": strictly_positive,
                         "support_weight_min": None if interval is None else interval[0], "support_weight_max": None if interval is None else interval[1], **curves[i]})
    atomic_csv(DENSE_OUT / "pareto_geometry.csv", geometry)
    supported = [r for r in geometry if r["nondominated"]]
    atomic_csv(DENSE_OUT / "supported_points.csv", supported)
    violation_h = sum(bool(r["helpfulness_monotonicity_violation"]) for r in geometry); violation_s = sum(bool(r["harmlessness_monotonicity_violation"]) for r in geometry)
    flat = sum(bool(r["low_sensitivity"]) for r in geometry); unsupported = sum(r["nondominated"] and not r["supported_positive_linear"] for r in geometry)
    hv = hypervolume_2d(points, lock["pareto_hypervolume"]["reference_point_normalized"])
    result = {"status": "COMPLETE", "prompts": len({r["sample_id"] for r in records}), "cases": len(records),
              "spearman_alpha_helpfulness": spearman([r["alpha_helpfulness"] for r in summary_rows], [r["mean_helpfulness"] for r in summary_rows]),
              "spearman_alpha_harmlessness": spearman([r["alpha_helpfulness"] for r in summary_rows], [r["mean_harmlessness"] for r in summary_rows]),
              "monotonicity_violations": {"helpfulness": violation_h, "harmlessness": violation_s}, "low_sensitivity_intervals": flat,
              "nondominated_points": sum(nd), "unsupported_nondominated_points": unsupported, "dense_alpha_hv": hv,
              "normalization_source": str(PROTOCOL), "low_sensitivity_definition": "both absolute finite-difference slopes < 0.02"}
    atomic_json(DENSE_OUT / "dense_alpha_summary.json", result); return result
