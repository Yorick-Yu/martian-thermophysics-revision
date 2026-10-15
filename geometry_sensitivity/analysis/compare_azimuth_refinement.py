import json, hashlib
from pathlib import Path
import numpy as np, torch
R=Path("/home/yqs/p31-b3-boundary/geometry-sensitivity-20261012")
BASE=R/"inputs/reference_states"
def load(d):
 out={k:[] for k in ["scenario_id","baseline_final_surface_temperature_k","shadow_final_surface_temperature_k","baseline_final_frost_mass_kg_m2","shadow_final_frost_mass_kg_m2"]}
 for s in range(4):
  for f in sorted((d/f"shard{s}"/"states").glob("chunk-*.pt")):
   x=torch.load(f,map_location="cpu",weights_only=False)
   for k in out: out[k].append(x[k].detach().cpu().numpy())
 z={k:np.concatenate(v) for k,v in out.items()}; ix=np.argsort(z["scenario_id"]); return {k:v[ix] for k,v in z.items()}
def m(a,b):
 d=np.asarray(b,dtype=np.float64)-np.asarray(a,dtype=np.float64); ad=np.abs(d)
 return {"rmse":float(np.sqrt(np.mean(d*d))),"p95_abs":float(np.quantile(ad,.95)),"p99_abs":float(np.quantile(ad,.99)),"max_abs":float(ad.max()),"mean_signed":float(d.mean())}
def path(site,n):
 if n==24:return BASE/f"{site}_baseline"
 if n==192:return R/f"runs/{site}_azimuth_only/attempt-001"
 return R/f"runs/azimuth_refinement/{site}_azimuth_{n}/attempt-002"
report={"schema_version":1,"purpose":"Consecutive horizon azimuth sampling sensitivity on identical scenario IDs; state differences do not constitute radiance uncertainty tests.","azimuths":{}}
for site in ("gasa","kaiser"):
 xs={n:load(path(site,n)) for n in (24,48,96,192)}
 ids=xs[24]["scenario_id"]
 if not all(np.array_equal(ids,x["scenario_id"]) for x in xs.values()): raise RuntimeError("scenario ids mismatch")
 q={"scenario_count":int(ids.size),"scenario_ids_identical":True,"adjacent":{},"to_24":{}}
 for n in (48,96,192):
  old=xs[n//2] if n>48 else xs[24]; new=xs[n]
  rec={}
  for fld in ("shadow_final_surface_temperature_k","shadow_final_frost_mass_kg_m2"):
   rec[fld]=m(old[fld],new[fld])
  a=old["shadow_final_frost_mass_kg_m2"]>1e-10;b=new["shadow_final_frost_mass_kg_m2"]>1e-10
  rec["frost_presence_flip_count"]=int(np.count_nonzero(a!=b))
  q["adjacent"][f"{n//2}_to_{n}"]=rec
  q["to_24"][str(n)]={fld:m(xs[24][fld],new[fld]) for fld in ("shadow_final_surface_temperature_k","shadow_final_frost_mass_kg_m2")}
 report["azimuths"][site]=q
out=R/"results/thermal_state_azimuth_refinement_v1.json";out.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"output":str(out),"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"summary":{s:{k:{"temperature_rmse":v["shadow_final_surface_temperature_k"]["rmse"],"temperature_p95":v["shadow_final_surface_temperature_k"]["p95_abs"],"frost_rmse":v["shadow_final_frost_mass_kg_m2"]["rmse"],"frost_flips":v["frost_presence_flip_count"]} for k,v in q["adjacent"].items()} for s,q in report["azimuths"].items()}},indent=2))
