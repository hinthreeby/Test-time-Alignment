from __future__ import annotations
import json
from .config import DENSE_OUT, GRADIENT_OUT, OUTPUT_ROOT
from .io import atomic_json, atomic_text

def run() -> dict:
    dense = json.loads((DENSE_OUT / "dense_alpha_summary.json").read_text()); gradient = json.loads((GRADIENT_OUT / "gradient_summary.json").read_text())
    intervals = 20
    h1_flags = {"monotonicity_violation_fraction_at_least_0.20": (dense["monotonicity_violations"]["helpfulness"] + dense["monotonicity_violations"]["harmlessness"]) / (2*intervals) >= .20,
                "low_sensitivity_fraction_at_least_0.25": dense["low_sensitivity_intervals"] / intervals >= .25,
                "unsupported_nondominated_point_exists": dense["unsupported_nondominated_points"] > 0}
    h2_flags = {"negative_cosine_majority": gradient["fraction_cosine_below_zero"] >= .5,
                "midpoint_median_cancellation_below_0.80": gradient["cancellation_by_alpha"]["0.5"]["median"] < .8}
    h1, h2 = any(h1_flags.values()), all(h2_flags.values()); supported = h1 and h2
    decision = "HYPOTHESIS_SUPPORTED_FOR_TINY_TRAIN" if supported else "LINEAR_SCALARIZATION_BOTTLENECK_NOT_ESTABLISHED"
    result = {"decision": decision, "h1_supported": h1, "h2_supported": h2, "h1_evidence": h1_flags, "h2_evidence": h2_flags,
              "gate_policy": "Exploratory fail-fast thresholds declared in code; evidence is reported continuously and conclusions are not forced.",
              "phase_c_started": False}
    atomic_json(OUTPUT_ROOT / "decision.json", result)
    atomic_text(OUTPUT_ROOT / "report.md", f"# Pareto scalarization feasibility probe\n\n## Phase A\n\n- Dense-alpha HV: {dense['dense_alpha_hv']:.6f}\n- Spearman alpha/helpfulness: {dense['spearman_alpha_helpfulness']:.4f}\n- Spearman alpha/harmlessness: {dense['spearman_alpha_harmlessness']:.4f}\n- Monotonicity violations (help/safe): {dense['monotonicity_violations']['helpfulness']}/{dense['monotonicity_violations']['harmlessness']}\n- Low-sensitivity intervals: {dense['low_sensitivity_intervals']}\n- Nondominated / unsupported nondominated: {dense['nondominated_points']} / {dense['unsupported_nondominated_points']}\n\n## Phase B\n\n- Mean/median gradient cosine: {gradient['cosine_similarity']['mean']:.4f} / {gradient['cosine_similarity']['median']:.4f}\n- Fraction cosine < 0: {gradient['fraction_cosine_below_zero']:.2%}\n- Midpoint median cancellation: {gradient['cancellation_by_alpha']['0.5']['median']:.4f}\n- Optimizer steps: 0\n\n## Decision\n\n**{decision}**\n\nPhase C was not started.\n")
    return result
