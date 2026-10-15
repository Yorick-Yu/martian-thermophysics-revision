import hashlib,json,pathlib
W=pathlib.Path("/home/yqs/p31-b3-boundary/geometry-sensitivity-20261012")
out=W/"configs/full_product_projection_attempt002"; out.mkdir(parents=True,exist_ok=True)
for site in ("gasa","kaiser"):
 for dist in (128,362):
  name=f"{site}_distance_{dist}"
  src=W/"configs/full_product_projection"/(name+".json")
  cfg=json.load(open(src))
  dem=pathlib.Path(cfg["dtm_path"])
  got=hashlib.sha256(dem.read_bytes()).hexdigest()
  if got != cfg["dtm_sha256"]: raise ValueError(f"DTM hash mismatch {name}")
  for item in cfg["validation_products"]:
   if item["product_id"].lower().startswith(site):
    item["state_globs"]=[str(W/f"runs/full_product/{name}/attempt-002/shard*/states/chunk-*.pt")]
    cube=pathlib.Path(item["cube_path"])
    if not cube.is_file(): raise FileNotFoundError(cube)
  cfg["output_dir"]=str(W/f"results/full_product_projection_attempt002/{name}")
  cfg["run_id"] += "-ATTEMPT-002"
  target=out/(name+".json")
  target.write_text(json.dumps(cfg,indent=2,ensure_ascii=False,sort_keys=False)+"\n")
  chk=json.load(open(target))
  if chk["frozen_gain"] != json.load(open(src))["frozen_gain"] or chk["frozen_offset"] != json.load(open(src))["frozen_offset"]:
   raise ValueError(f"frozen calibration changed {name}")
  print(f"PASS {target} dem_sha256={got} state_glob={chk['validation_products'][0]['state_globs'][0]} output={chk['output_dir']}")
