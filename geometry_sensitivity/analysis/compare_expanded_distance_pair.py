import json, hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path("/home/yqs/p31-b3-boundary/geometry-sensitivity-20261012")
FIELDS=["final_surface_temperature_k","final_frost_mass_kg_m2","maximum_frost_mass_kg_m2"]
PREFIXES=["baseline","shadow"]
def load_run(path):
    store={"scenario_id":[]}
    for pref in PREFIXES:
        for fld in FIELDS: store[f"{pref}_{fld}"]=[]
    for shard in range(4):
        files=sorted((path/f"shard{shard}"/"states").glob("chunk-*.pt"))
        if not files: raise RuntimeError(f"No state chunks in {path}/shard{shard}")
        for file in files:
            chunk=torch.load(file,map_location="cpu",weights_only=False)
            for key in store:
                if key not in chunk: raise RuntimeError(f"Missing {key} in {file}")
                store[key].append(chunk[key].detach().cpu().numpy())
    data={key:np.concatenate(parts) for key,parts in store.items()}
    order=np.argsort(data["scenario_id"])
    return {key:val[order] for key,val in data.items()}
def metric(a,b):
    delta=np.asarray(b,dtype=np.float64)-np.asarray(a,dtype=np.float64)
    absolute=np.abs(delta)
    return {"mean_signed_362_minus_128":float(delta.mean()),"rmse":float(np.sqrt(np.mean(delta**2))),"p50_abs":float(np.quantile(absolute,.5)),"p95_abs":float(np.quantile(absolute,.95)),"p99_abs":float(np.quantile(absolute,.99)),"max_abs":float(absolute.max()),"fraction_abs_gt_1e-10":float(np.mean(absolute>1e-10))}
report={"schema_version":1,"purpose":"Compare final thermal fields at maximum horizon distances 128 and 362 cells for the same expanded HiRISE DTM, using the common valid terrain scenario IDs.","distance_cells":{"short":128,"long":362},"sites":{}}
for site in ("gasa","kaiser"):
    cfgs={}
    runs={}
    for distance in (128,362):
        name=f"{site}_distance_{distance}"
        cfg=json.loads((ROOT/"configs/expanded_range"/f"{name}.json").read_text())
        cfgs[distance]=cfg
        runs[distance]=load_run(ROOT/"runs"/"expanded_range"/name/"attempt-001")
        if not all(json.loads((ROOT/"runs"/"expanded_range"/name/"attempt-001"/f"shard{i}"/"result.json").read_text())["status"]=="PASS" for i in range(4)):
            raise RuntimeError(f"Non-PASS shard in {name}")
    ids_a=runs[128]["scenario_id"]; ids_b=runs[362]["scenario_id"]
    same=np.array_equal(ids_a,ids_b)
    same_dtm=cfgs[128]["dtm_sha256"]==cfgs[362]["dtm_sha256"]
    if not same or not same_dtm: raise RuntimeError(f"Incomparable terrain support for {site}")
    item={"scenario_count_common_valid_cells":int(ids_a.size),"scenario_ids_identical":same,"scenario_ids_unique":bool(np.unique(ids_a).size==ids_a.size),"dtm_sha256_identical":same_dtm,"dtm_sha256":cfgs[128]["dtm_sha256"],"grid_shape":cfgs[128].get("grid_shape"),"distance_pair_thermal_changes":{}}
    for pref in PREFIXES:
        for fld in FIELDS:
            key=f"{pref}_{fld}"
            item["distance_pair_thermal_changes"][key]=metric(runs[128][key],runs[362][key])
        a=runs[128][f"{pref}_final_frost_mass_kg_m2"]>1e-10
        b=runs[362][f"{pref}_final_frost_mass_kg_m2"]>1e-10
        item["distance_pair_thermal_changes"][f"{pref}_frost_presence_flip_count"]=int(np.count_nonzero(a!=b))
    report["sites"][site]=item
out=ROOT/"results/expanded_range_thermal_distance_pair_v1.json"
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"output":str(out),"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"summary":{s:{k:{"rmse":v.get("rmse"),"p95_abs":v.get("p95_abs"),"max_abs":v.get("max_abs")} for k,v in d["distance_pair_thermal_changes"].items() if isinstance(v,dict) and ("shadow_" in k)} for s,d in report["sites"].items()}},indent=2))
