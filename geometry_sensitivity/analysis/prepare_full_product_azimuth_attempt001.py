import hashlib,json,pathlib,sys
W=pathlib.Path("/home/yqs/p31-b3-boundary/geometry-sensitivity-20261012")
sys.path.insert(0,str(W/"code"))
import run_torch_gasa_shadow as runner
out=W/"configs/full_product_azimuth_attempt001"; out.mkdir(parents=True,exist_ok=True)
for site in ("gasa","kaiser"):
 for az in (48,96,192):
  name=f"{site}_az{az}_distance_362"
  base=json.load(open(W/f"configs/full_product_attempt002/{site}_distance_362.json"))
  base["horizon_azimuth_count"]=az
  base["run_id"]=base.get("run_id","GEOM-SENS")+"-AZ"+str(az)+"-ATTEMPT-001"
  runner._validate_config(base)
  dem=pathlib.Path(base["dtm_path"])
  if hashlib.sha256(dem.read_bytes()).hexdigest()!=base["dtm_sha256"]: raise ValueError(f"DTM hash mismatch {name}")
  target=out/(name+".json")
  target.write_text(json.dumps(base,indent=2,ensure_ascii=False,sort_keys=False)+"\n")
  runner._validate_config(json.load(open(target)))
  print(f"THERMAL_CONFIG_PASS {target} az={az} dist={base['horizon_maximum_distance_cells']} dtm={base['dtm_sha256']}")
pout=W/"configs/full_product_azimuth_projection_attempt001"; pout.mkdir(parents=True,exist_ok=True)
for site in ("gasa","kaiser"):
 for az in (48,96,192):
  name=f"{site}_az{az}_distance_362"
  source=json.load(open(W/f"configs/full_product_projection_attempt002_v2/{site}_distance_362.json"))
  source["run_id"] += f"-AZ{az}-ATTEMPT-001"
  source["output_dir"]=str(W/f"results/full_product_azimuth_projection_attempt001/{name}")
  for item in source["validation_products"]:
   item["state_globs"]=[str(W/f"runs/full_product_azimuth/{name}/attempt-001/shard*/states/chunk-*.pt")]
  target=pout/(name+".json")
  target.write_text(json.dumps(source,indent=2,ensure_ascii=False,sort_keys=False)+"\n")
  chk=json.load(open(target))
  if chk["frozen_gain"]!=json.load(open(W/f"configs/full_product_projection_attempt002_v2/{site}_distance_362.json"))["frozen_gain"]:
   raise ValueError(f"frozen gain changed {name}")
  if "attempt-001/shard*" not in chk["validation_products"][0]["state_globs"][0]: raise ValueError(f"state glob mismatch {name}")
  print(f"PROJECTION_CONFIG_PASS {target} output={chk['output_dir']}")
