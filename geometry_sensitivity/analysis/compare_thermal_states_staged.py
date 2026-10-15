#!/usr/bin/env python3
import json, hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path("/home/yqs/p31-b3-boundary/geometry-sensitivity-20261012")
BASE=ROOT/"inputs/reference_states"
CASES={"gasa":(BASE/"gasa_baseline",["distance_only","azimuth_only","joint"]),"kaiser":(BASE/"kaiser_baseline",["distance_only","azimuth_only","joint"])}
FIELDS=["final_surface_temperature_k","final_frost_mass_kg_m2","maximum_frost_mass_kg_m2"]
def load_run(run_dir):
 out={k:[] for k in ["scenario_id"]+[f"{prefix}_{field}" for prefix in ("baseline","shadow") for field in FIELDS]}
 for s in range(4):
  for f in sorted((run_dir/f"shard{s}"/"states").glob("chunk-*.pt")):
   d=torch.load(f,map_location="cpu",weights_only=False)
   for k in out: out[k].append(d[k].detach().cpu().numpy())
 data={k:np.concatenate(v) for k,v in out.items()}
 ix=np.argsort(data["scenario_id"])
 return {k:v[ix] for k,v in data.items()}
def metrics(a,b,eps):
 d=np.asarray(b,dtype=np.float64)-np.asarray(a,dtype=np.float64); ad=np.abs(d)
 return {"n":int(d.size),"mean_signed_delta":float(np.mean(d)),"rmse":float(np.sqrt(np.mean(d*d))),"p50_abs":float(np.quantile(ad,.50)),"p95_abs":float(np.quantile(ad,.95)),"p99_abs":float(np.quantile(ad,.99)),"max_abs":float(np.max(ad)),"fraction_abs_gt_epsilon":float(np.mean(ad>eps))}
report={"schema_version":1,"purpose":"Compare final thermal state under geometry variants against archived formal 24-azimuth/128-cell baseline on identical valid scenario IDs.","baseline_pairs":{},"reference_integrity_audit":str(ROOT/"qa/reference_input_integrity.json"),"cases":{},"tolerances_only_for_machine_equality":{"temperature_k":1e-10,"frost_kg_m2":1e-12}}
for site,(bdir,variants) in CASES.items():
 b=load_run(bdir)
 report["baseline_pairs"][site]={"run_dir":str(bdir),"scenario_count":int(b["scenario_id"].size),"scenario_id_min":int(b["scenario_id"].min()),"scenario_id_max":int(b["scenario_id"].max()),"scenario_ids_unique":bool(np.unique(b["scenario_id"]).size==b["scenario_id"].size)}
 report["cases"][site]={}
 for v in variants:
  rdir=ROOT/"runs"/f"{site}_{v}"/"attempt-001"; r=load_run(rdir)
  same=np.array_equal(b["scenario_id"],r["scenario_id"])
  if not same: raise RuntimeError(f"Scenario ID mismatch {site}/{v}")
  vr={"run_dir":str(rdir),"scenario_count":int(r["scenario_id"].size),"scenario_ids_match":same,"baseline_B1_against_archived":{},"shadow_B3_against_archived":{},"frost_presence_flip_count":0}
  for fld in FIELDS:
   e=1e-10 if fld.endswith("temperature_k") else 1e-12
   vr["baseline_B1_against_archived"][fld]=metrics(b[f"baseline_{fld}"],r[f"baseline_{fld}"],e)
   vr["shadow_B3_against_archived"][fld]=metrics(b[f"shadow_{fld}"],r[f"shadow_{fld}"],e)
  old=b["shadow_final_frost_mass_kg_m2"]>1e-10; new=r["shadow_final_frost_mass_kg_m2"]>1e-10
  vr["frost_presence_flip_count"]=int(np.count_nonzero(old!=new)); vr["frost_presence_flip_fraction"]=float(np.mean(old!=new))
  report["cases"][site][v]=vr
out=ROOT/"results/thermal_state_geometry_comparison_v2_staged.json"; out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"output":str(out),"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"summary":{site:{v:{"T_RMSE":d["shadow_B3_against_archived"]["final_surface_temperature_k"]["rmse"],"T_p95":d["shadow_B3_against_archived"]["final_surface_temperature_k"]["p95_abs"],"T_max":d["shadow_B3_against_archived"]["final_surface_temperature_k"]["max_abs"],"frost_RMSE":d["shadow_B3_against_archived"]["final_frost_mass_kg_m2"]["rmse"],"frost_max":d["shadow_B3_against_archived"]["final_frost_mass_kg_m2"]["max_abs"],"frost_flip_count":d["frost_presence_flip_count"]} for v,d in variants.items()} for site,variants in report["cases"].items()}},indent=2))
