import hashlib, json, pathlib, sys
W=pathlib.Path("/home/yqs/p31-b3-boundary/geometry-sensitivity-20261012")
sys.path.insert(0,str(W/"code"))
import run_torch_gasa_shadow as runner
axis_names=runner.AXIS_NAMES
outdir=W/"configs/full_product_attempt002"
outdir.mkdir(parents=True,exist_ok=True)
for site in ("gasa","kaiser"):
 for dist in (128,362):
  name=f"{site}_distance_{dist}"
  src=W/"configs/full_product"/(name+".json")
  cfg=json.load(open(src))
  old=cfg["parameter_axes"]
  cfg["parameter_axes"]={k:old[k] for k in axis_names}
  if "run_id" in cfg: cfg["run_id"] += "-ATTEMPT-002"
  runner._validate_config(cfg)
  dem=pathlib.Path(cfg["dtm_path"])
  if not dem.is_file(): raise FileNotFoundError(dem)
  h=hashlib.sha256(dem.read_bytes()).hexdigest()
  if h != cfg["dtm_sha256"]: raise ValueError(f"DTM hash mismatch {name}: {h}")
  target=outdir/(name+".json")
  target.write_text(json.dumps(cfg,indent=2,ensure_ascii=False,sort_keys=False)+"\n")
  reread=json.load(open(target)); runner._validate_config(reread)
  print(f"PASS {target} axes={list(reread['parameter_axes'])} dtm_sha256={h} config_sha256={hashlib.sha256(target.read_bytes()).hexdigest()}")
