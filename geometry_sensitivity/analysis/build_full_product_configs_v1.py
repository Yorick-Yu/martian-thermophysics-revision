import copy, hashlib, json
from pathlib import Path
W=Path("/home/yqs/p31-b3-boundary/geometry-sensitivity-20261012")
audit=json.loads((W/"qa/full_product_dtm_build_audit_v1.json").read_text())
for site in ("gasa","kaiser"):
  rec=audit["sites"][site]
  dtm=Path(rec["dtm_path"])
  assert hashlib.sha256(dtm.read_bytes()).hexdigest()==rec["dtm_sha256"]
  for distance in (128,362):
    name=f"{site}_distance_{distance}"
    base=json.loads((W/"configs/expanded_range"/f"{name}.json").read_text())
    base["experiment"]=f"same official full-product HiRISE DTM range sensitivity {site}, fixed 24 azimuth, {distance}-cell horizon"
    base["dtm_path"]=str(dtm); base["dtm_sha256"]=rec["dtm_sha256"]; base["dtm_size_bytes"]=dtm.stat().st_size
    base["model_scope"]=f"full_product_{site}_DTM_24az_{distance}cells_paired_B1_B3"
    base["claim_boundary"]="Full available official PDS DTM product at factor-8 engineering resolution. Directional product-edge and no-data limits are audited in qa/full_product_dtm_build_audit_v1.json; no claim extends beyond that source."
    dest=W/"configs/full_product"/f"{name}.json"
    if dest.exists(): raise RuntimeError(f"refusing to overwrite {dest}")
    dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(json.dumps(base,indent=2,sort_keys=True)+"\n")
    proj=copy.deepcopy(json.loads((W/"configs/expanded_range_projection"/f"{name}.json").read_text()))
    proj["dtm_path"]=str(dtm); proj["dtm_sha256"]=rec["dtm_sha256"]
    proj["claim_boundary"]="Frozen observation projection of full available official PDS DTM simulation. Product-edge and internal no-data limits are documented in qa/full_product_dtm_build_audit_v1.json."
    proj["run_id"]=f"GEOM-SENS-20261013-{site.upper()}-FULL-PRODUCT-24AZ-{distance}CELL-FROZEN-PROJECTION"
    proj["output_dir"]=str(W/"results/full_product_projection"/name)
    for item in proj["validation_products"]:
      item["state_globs"]=[str(W/"runs/full_product"/name/"attempt-001"/"shard*"/"states"/"chunk-*.pt")]
    pdest=W/"configs/full_product_projection"/f"{name}.json"
    if pdest.exists(): raise RuntimeError(f"refusing to overwrite {pdest}")
    pdest.parent.mkdir(parents=True,exist_ok=True); pdest.write_text(json.dumps(proj,indent=2,sort_keys=True)+"\n")
print("Created four full-product thermal configs and four frozen-projection configs.")
