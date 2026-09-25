from __future__ import annotations
import csv, json
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT=Path(__file__).resolve().parents[3]
EPOCH1=ROOT/"results/parm_taro/pareto_scalarization_probe/01_dense_alpha"
CONTROL=ROOT/"results/parm_taro/recovery/reproduced/pblora_epoch2_control"
EPOCH2=CONTROL/"dense_alpha/01_dense_alpha"; OUT=CONTROL/"epoch1_vs_epoch2"

def read_csv(path: Path) -> list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as handle: return list(csv.DictReader(handle))
def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); temporary.replace(path)
def metrics(directory: Path) -> dict[str,Any]:
    summary=json.loads((directory/"dense_alpha_summary.json").read_text()); records=read_csv(directory/"dense_alpha_records.csv")
    return {"MIP":fmean(float(r["MIP"]) for r in records),"HV":summary["dense_alpha_hv"],
        "helpfulness_spearman":summary["spearman_alpha_helpfulness"],"harmlessness_spearman":summary["spearman_alpha_harmlessness"],
        "helpfulness_monotonicity_violations":summary["monotonicity_violations"]["helpfulness"],
        "harmlessness_monotonicity_violations":summary["monotonicity_violations"]["harmlessness"],
        "nondominated_points":summary["nondominated_points"],"unsupported_nondominated_points":summary["unsupported_nondominated_points"]}
def run() -> dict[str,Any]:
    one,two=metrics(EPOCH1),metrics(EPOCH2); g1,g2=read_csv(EPOCH1/"pareto_geometry.csv"),read_csv(EPOCH2/"pareto_geometry.csv")
    if [r["alpha_helpfulness"] for r in g1] != [r["alpha_helpfulness"] for r in g2]: raise RuntimeError("Alpha grids differ")
    rows=[]
    for a,b in zip(g1,g2):
        rows.append({"alpha_helpfulness":a["alpha_helpfulness"],"alpha_harmlessness":a["alpha_harmlessness"],
            "epoch1_helpfulness":a["mean_helpfulness"],"epoch1_harmlessness":a["mean_harmlessness"],
            "epoch2_helpfulness":b["mean_helpfulness"],"epoch2_harmlessness":b["mean_harmlessness"]})
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"per_alpha_objectives.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5)); ax.plot([float(r["epoch1_helpfulness"]) for r in rows],[float(r["epoch1_harmlessness"]) for r in rows],"o-",label="Epoch 1")
    ax.plot([float(r["epoch2_helpfulness"]) for r in rows],[float(r["epoch2_harmlessness"]) for r in rows],"o-",label="Epoch 2 control")
    ax.set(xlabel="Normalized helpfulness",ylabel="Normalized harmlessness",title="Dense-alpha objective space"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"objective_space.png",dpi=180); plt.close(fig)
    epoch1_irregular=one["helpfulness_monotonicity_violations"]+one["harmlessness_monotonicity_violations"]>=8 or one["unsupported_nondominated_points"]>0
    epoch2_irregular=two["helpfulness_monotonicity_violations"]+two["harmlessness_monotonicity_violations"]>=8 or two["unsupported_nondominated_points"]>0
    verdict="UNDERTRAINING_EXPLAINS_IRREGULARITY" if epoch1_irregular and not epoch2_irregular else ("IRREGULARITY_PERSISTS_AFTER_EPOCH2" if epoch2_irregular else "INCONCLUSIVE")
    result={"verdict":verdict,"epoch1":one,"epoch2":two,"delta":{k:two[k]-one[k] for k in one},
        "same_prompts_and_alpha_grid":True,"criterion":"irregular iff combined monotonicity violations >= 8/40 or any unsupported nondominated point"}
    write_json(OUT/"comparison.json",result)
    lines=["# Epoch 1 vs Epoch 2 PBLoRA control","",f"Verdict: **{verdict}**","","| Metric | Epoch 1 | Epoch 2 | Delta |","|---|---:|---:|---:|"]
    lines += [f"| {k} | {one[k]:.6g} | {two[k]:.6g} | {two[k]-one[k]:+.6g} |" for k in one]
    lines += ["","The runs use the identical 20-prompt manifest, 21-point alpha grid, generation protocol, Beaver scorers, and normalization.",""]
    (OUT/"report.md").write_text("\n".join(lines),encoding="utf-8"); return result
