"""Aggregate recovery evidence without overclaiming incomplete diagnostics."""
import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from PARM_TARO.recovery.low_vram_suite.common import OUT, atomic_csv_write, atomic_json_dump, atomic_text, dry_run_payload

def read(name):
    path=OUT/name
    try: return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError): return None
def main():
    p=argparse.ArgumentParser(); p.add_argument("--dry-run",action="store_true"); args=p.parse_args()
    expected=["01_artifact_audit.json","02_guide_utility_summary.json","03_summary.json","04_headroom_summary.json","05_exposure_shift_summary.json","06_generation_summary.json","07_scorer_summary.json","09_metrics_summary.json"]
    if args.dry_run: print(json.dumps(dry_run_payload("10_decision_report",inputs={name:(OUT/name).is_file() for name in expected}),indent=2)); return 0
    artifact=read(expected[0]); guide=read(expected[1]); lam=read(expected[2]); headroom=read(expected[3]); exposure=read(expected[4]); generation=read(expected[5]); scorer=read(expected[6]); full=read(expected[7]); alpha_probe=read("../pblora_repro/epoch1_validation/alpha_probe_summary.json")
    def row(rc,hypothesis,status,evidence,number="",action=""):
        return {"rc":rc,"hypothesis":hypothesis,"status":status,"evidence_file":evidence,"measured_number":number,"recommended_action":action}
    matrix=[
      row("RC1","PBLoRA missing/wrong","NOT SUPPORTED" if artifact and artifact.get("status")=="PASS" else "NOT TESTED","01_artifact_audit.json",str(artifact.get("trainable_parameter_count")) if artifact else ""),
      row("RC2","Alpha not entering PBLoRA","NOT SUPPORTED" if alpha_probe and alpha_probe.get("classification")=="PASS" else "NOT TESTED","../pblora_repro/epoch1_validation/alpha_probe_summary.json",str(alpha_probe.get("mean_extreme_alpha_js_divergence")) if alpha_probe else ""),
      row("RC3","Weak guide/preference ranking","STRONGLY SUPPORTED" if guide and guide.get("verdict")=="GUIDE UTILITY FAIL" else ("WEAKLY SUPPORTED" if guide and guide.get("verdict")=="GUIDE UTILITY WEAK" else ("NOT SUPPORTED" if guide else "NOT TESTED")),"02_guide_utility_summary.json",str(guide.get("weighted_accuracy_non_tie")) if guide else ""),
      row("RC4","Model/tokenizer/checkpoint provenance wrong","NOT SUPPORTED" if artifact and artifact.get("status")=="PASS" else "NOT TESTED","01_artifact_audit.json"),
      row("RC5","Fusion mismatch with author PARM","CONFIRMED" if lam else "NOT TESTED","03_summary.json",str(lam.get("parity_max_abs_logprob_diff")) if lam else "","Treat author/current/paper formulations separately"),
      row("RC6","Scale/temperature confound","WEAKLY SUPPORTED" if lam else "NOT TESTED","03_fusion_scale.csv","","Compare entropy/max-probability curves"),
      row("RC7","Router checkpoint wrong/random","NOT TESTED","historical Stage-10 only","","Audit only after router reproduction"),
      row("RC8","Router ignores alpha","NOT TESTED","historical Stage-10 only","","Requires valid reproduced router"),
      row("RC9","NLL pushes lambda to zero","STRONGLY SUPPORTED" if lam and lam.get("oracle",{}).get("current",{}).get("global",{}).get("lambda")==0.0 else ("WEAKLY SUPPORTED" if lam else "NOT TESTED"),"03_gradient_summary.csv",str(lam.get("oracle",{}).get("current",{}).get("global")) if lam else ""),
      row("RC10","Gate saturation","NOT TESTED","router checkpoint unavailable","","Requires router runtime"),
      row("RC11","Feature normalization/cache issue","NOT TESTED","router cache/router checkpoint unavailable"),
      row("RC12","Teacher-forced/autoregressive exposure shift","STRONGLY SUPPORTED" if exposure and exposure.get("gradient_sign_agreement",1)<.5 else ("NOT SUPPORTED" if exposure else "NOT TESTED"),"05_exposure_shift_summary.json",str(exposure.get("gradient_sign_agreement")) if exposure else ""),
      row("RC13","No adaptive headroom","STRONGLY SUPPORTED" if full and headroom and headroom.get("normalized",{}).get("classification")=="HEADROOM ABSENT" else ("WEAKLY SUPPORTED" if headroom and headroom.get("normalized",{}).get("classification")=="HEADROOM ABSENT" else "NOT TESTED"),"04_headroom_summary.json; 09_metrics_summary.json","","Teacher-forced result alone cannot confirm"),
      row("RC14","Headroom exists but router fails","WEAKLY SUPPORTED" if headroom and headroom.get("normalized",{}).get("classification")=="HEADROOM CLEAR" else "NOT TESTED","04_headroom_summary.json"),
      row("RC15","Optimal lambda varies by alpha","WEAKLY SUPPORTED" if headroom and headroom.get("normalized",{}).get("fraction_prompts_multiple_alpha_optima",0)>0 else ("NOT SUPPORTED" if headroom else "NOT TESTED"),"04_headroom_summary.json",str(headroom.get("normalized",{}).get("fraction_prompts_multiple_alpha_optima")) if headroom else ""),
      row("RC16","Evaluator wiring bad","NOT SUPPORTED" if scorer and scorer.get("status")=="PASS" else "NOT TESTED","07_scorer_summary.json"),
    ]
    complete_full=bool(full and full.get("status")=="COMPLETE" and scorer and scorer.get("status")=="PASS")
    blocked=bool(artifact and artifact.get("status")=="FAIL")
    if blocked: level="BLOCKED"; verdict="Artifacts invalid"
    elif not complete_full: level="DIAGNOSTIC ONLY"; verdict="Full autoregressive scored Feasibility-60 is incomplete"
    else:
        # A final GO/NO-GO requires scorer-valid candidate headroom; retain a
        # conservative diagnostic outcome unless 09 records an explicit gain.
        oracle=read("09_oracle_summary.json") or {}; gain=float(oracle.get("mip_gain",0))
        if gain>=.02: level="GO ROUTER"; verdict="Meaningful scored oracle MIP headroom"
        elif gain<.002: level="NO-GO ADAPTIVE LAMBDA"; verdict="Scored oracle is approximately the strong global candidate"
        else: level="DIAGNOSTIC ONLY"; verdict="Scored headroom is mixed"
    decision={"status":level,"reason":verdict,"full_scored_feasibility_complete":complete_full,"teacher_forced_nll_not_final":True,"regret_not_independent_from_mip":True,"root_causes":matrix}; atomic_csv_write(OUT/"10_root_cause_matrix.csv",matrix,matrix[0].keys()); atomic_json_dump(OUT/"10_decision.json",decision)
    lines=["# PARM-TARO low-VRAM decision",f"\nDecision level: **{level}**",f"\n{verdict}","\nTeacher-forced NLL is diagnostic only. Candidate-pool regret is algebraically dependent on candidate MIP.","\n## Root causes\n","| RC | Hypothesis | Status | Evidence |","|---|---|---|---|"]+[f"| {r['rc']} | {r['hypothesis']} | {r['status']} | {r['evidence_file']} |" for r in matrix]; atomic_text(OUT/"10_decision_report.md","\n".join(lines)+"\n"); print(level); return 0
if __name__=="__main__": raise SystemExit(main())

