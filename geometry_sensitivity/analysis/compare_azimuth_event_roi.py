import json,hashlib
from pathlib import Path
import numpy as np,torch
R=Path("/home/yqs/p31-b3-boundary/geometry-sensitivity-20261012")
def load(d):
 vals={"scenario_id":[],"shadow_final_surface_temperature_k":[],"shadow_final_frost_mass_kg_m2":[]}
 for s in range(4):
  for f in sorted((d/f"shard{s}"/"states").glob("chunk-*.pt")):
   x=torch.load(f,map_location="cpu",weights_only=False)
   for k in vals:vals[k].append(x[k].detach().cpu().numpy())
 z={k:np.concatenate(v) for k,v in vals.items()};o=np.argsort(z["scenario_id"]);return {k:v[o] for k,v in z.items()}
def metrics(a,b):
 d=np.asarray(b)-np.asarray(a);q=np.abs(d)
 return {"n":int(d.size),"rmse":float(np.sqrt(np.mean(d*d))),"p95_abs":float(np.quantile(q,.95)),"p99_abs":float(np.quantile(q,.99)),"max_abs":float(q.max())}
report={"schema_version":1,"purpose":"Final thermal-field angular sensitivity within the event-centered radius on common valid cells. This uses same 256x256 DTM and model input except horizon azimuth count.","roi_radius_cells":80,"sites":{}}
for site,patt in [("gasa","*GASA_EVENT_256x256*"),("kaiser","*EVENT_256x256*")]:
 f=next(x for x in (R/"inputs"/site).glob("*.npz") if "256x256" in x.name);z=np.load(f,allow_pickle=False);meta=json.loads(str(z["metadata_json"]));valid=z["fully_valid"].astype(bool)
 row=(meta["event_center_line_zero_based"]-meta["downloaded_row_start_zero_based"])/8
 col=(meta["event_center_sample_zero_based"]-meta["raw_crop_column_start"])/8
 rr,cc=np.indices(valid.shape);roi=valid&((rr-row)**2+(cc-col)**2<=80**2)
 ids=np.flatnonzero(valid);selected=np.flatnonzero(roi);mask=np.isin(ids,selected)
 dirs={24:R/f"inputs/reference_states/{site}_baseline",48:R/f"runs/azimuth_refinement/{site}_azimuth_48/attempt-002",96:R/f"runs/azimuth_refinement/{site}_azimuth_96/attempt-002",192:R/f"runs/{site}_azimuth_only/attempt-001"}
 xs={a:load(p) for a,p in dirs.items()};baseids=xs[24]["scenario_id"]
 for a,x in xs.items():
  if not np.array_equal(baseids,x["scenario_id"]):raise RuntimeError(f"ID mismatch {site}/{a}")
 # scenario ordering is row-major valid cells for the one parameter combination.
 siteout={"roi_valid_cell_count":int(roi.sum()),"roi_center_row_col":[row,col],"adjacent":{}}
 for a,b in ((24,48),(48,96),(96,192)):
  vals={}
  for field in ("shadow_final_surface_temperature_k","shadow_final_frost_mass_kg_m2"):
   vals[field]=metrics(xs[a][field][mask],xs[b][field][mask])
  vals["frost_presence_flip_count"]=int(np.count_nonzero((xs[a]["shadow_final_frost_mass_kg_m2"][mask]>1e-10)!=(xs[b]["shadow_final_frost_mass_kg_m2"][mask]>1e-10)))
  siteout["adjacent"][f"{a}_to_{b}"]=vals
 report["sites"][site]=siteout
out=R/"results/thermal_state_azimuth_event_roi_v1.json";out.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"path":str(out),"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"summary":{s:{k:{"T_RMSE":v["shadow_final_surface_temperature_k"]["rmse"],"T_p95":v["shadow_final_surface_temperature_k"]["p95_abs"],"frost_RMSE":v["shadow_final_frost_mass_kg_m2"]["rmse"],"flips":v["frost_presence_flip_count"]} for k,v in x["adjacent"].items()} for s,x in report["sites"].items()}},indent=2))
