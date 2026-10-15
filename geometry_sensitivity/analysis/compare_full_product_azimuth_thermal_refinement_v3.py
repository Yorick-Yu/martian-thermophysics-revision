import hashlib,json,pathlib
import numpy as np, torch
W=pathlib.Path("/home/yqs/p31-b3-boundary/geometry-sensitivity-20261012")
base_counts=(24,48,96,192)
fields=("baseline_final_surface_temperature_k","shadow_final_surface_temperature_k","baseline_final_frost_mass_kg_m2","shadow_final_frost_mass_kg_m2","shadow_maximum_frost_mass_kg_m2")
def paths(site,n):
 if n==24:return W/f"runs/full_product/{site}_distance_362/attempt-002"
 return W/f"runs/full_product_azimuth/{site}_az{n}_distance_362/attempt-001"
def config(site,n):
 if n==384:return W/"configs/full_product_azimuth_attempt006/gasa_az384_distance_362.json"
 if n==768:return W/"configs/full_product_azimuth_attempt007/gasa_az768_distance_362.json"
 if n==24:return W/f"configs/full_product_attempt002/{site}_distance_362.json"
 return W/f"configs/full_product_azimuth_attempt001/{site}_az{n}_distance_362.json"
def load(path, shard_paths=None):
 d={"scenario_id":[]}; d.update({f:[] for f in fields})
 for sh in range(4):
  shard_path=(path/f"shard{sh}") if shard_paths is None else shard_paths[sh]
  rp=shard_path/"result.json"
  if json.load(open(rp))["status"]!="PASS": raise RuntimeError(f"not PASS {rp}")
  fs=sorted((shard_path/"states").glob("chunk-*.pt"))
  if not fs: raise RuntimeError(f"no chunks {shard_path}")
  for f in fs:
   c=torch.load(f,map_location="cpu",weights_only=False)
   for k in d:
    if k not in c: raise RuntimeError(f"missing {k} {f}")
    d[k].append(c[k].detach().cpu().numpy())
 out={k:np.concatenate(v) for k,v in d.items()}; ix=np.argsort(out["scenario_id"])
 return {k:v[ix] for k,v in out.items()}
def metric(a,b):
 delta=np.asarray(b,dtype=np.float64)-np.asarray(a,dtype=np.float64); ad=np.abs(delta)
 return {"mean_signed_new_minus_old":float(delta.mean()),"rmse":float(np.sqrt(np.mean(delta**2))),"p95_abs":float(np.quantile(ad,.95)),"p99_abs":float(np.quantile(ad,.99)),"max_abs":float(ad.max())}
report={"schema_version":1,"purpose":"Full official PDS-product DTM azimuth refinement at 362 cells through Gasa 768 azimuths. Compare final fields on identical scenario IDs and preserve all non-pass run files.","azimuth_counts":{"gasa":[24,48,96,192,384,768],"kaiser":[24,48,96,192]},"sites":{}}
for site in ("gasa","kaiser"):
 counts=base_counts+((384,768) if site=="gasa" else ())
 cfg={n:json.load(open(config(site,n))) for n in counts}
 dtms={cfg[n]["dtm_sha256"] for n in counts}
 if len(dtms)!=1: raise RuntimeError(f"DTM SHA differs {site}")
 if any(cfg[n]["horizon_maximum_distance_cells"]!=362 or cfg[n]["horizon_azimuth_count"]!=n for n in counts): raise RuntimeError(f"config mismatch {site}")
 runs={}
 for n in counts:
  if n==384:
   shard_paths=[W/f"runs/full_product_azimuth/gasa_az384_distance_362/attempt-{a}/shard{sh}" for sh,a in [(0,"005"),(1,"005"),(2,"006"),(3,"006")]]
   runs[n]=load(W/"runs/full_product_azimuth/gasa_az384_distance_362/attempt-006",shard_paths)
  else:
   runs[n]=load(paths(site,n))
 ids=runs[24]["scenario_id"]
 if not all(np.array_equal(ids,runs[n]["scenario_id"]) for n in counts): raise RuntimeError(f"scenario IDs differ {site}")
 item={"common_valid_scenario_count":int(ids.size),"scenario_ids_identical":True,"dtm_sha256":next(iter(dtms)),"grid_shape":cfg[24].get("grid_shape"),"adjacent_changes":{},"changes_vs_24":{}}
 for n in counts[1:]:
  prev=n//2; r={}
  for f in fields:r[f]=metric(runs[prev][f],runs[n][f])
  r["shadow_frost_presence_flip_count"]=int(np.count_nonzero((runs[prev]["shadow_final_frost_mass_kg_m2"]>1e-10)!=(runs[n]["shadow_final_frost_mass_kg_m2"]>1e-10)))
  item["adjacent_changes"][f"{prev}_to_{n}"]=r
  v={f:metric(runs[24][f],runs[n][f]) for f in fields}
  v["shadow_frost_presence_flip_count"]=int(np.count_nonzero((runs[24]["shadow_final_frost_mass_kg_m2"]>1e-10)!=(runs[n]["shadow_final_frost_mass_kg_m2"]>1e-10)))
  item["changes_vs_24"][str(n)]=v
 report["sites"][site]=item
out=W/"results/full_product_azimuth_thermal_refinement_v3.json"
if out.exists():raise RuntimeError(f"refusing overwrite {out}")
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
summary={s:{k:{"temp_rmse":v["shadow_final_surface_temperature_k"]["rmse"],"temp_p95":v["shadow_final_surface_temperature_k"]["p95_abs"],"frost_rmse":v["shadow_final_frost_mass_kg_m2"]["rmse"],"frost_flips":v["shadow_frost_presence_flip_count"]} for k,v in d["adjacent_changes"].items()} for s,d in report["sites"].items()}
print(json.dumps({"output":str(out),"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"summary":summary},indent=2))
