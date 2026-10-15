import hashlib,json,pathlib,sys
W=pathlib.Path("/home/yqs/p31-b3-boundary/geometry-sensitivity-20261012")
sys.path.insert(0,str(W/"code"))
import run_torch_gasa_shadow as runner
source=json.load(open(W/"configs/full_product_attempt002/gasa_distance_362.json"))
source["horizon_azimuth_count"]=384
source["experiment"]=source.get("experiment","")+"; 384-azimuth follow-up attempt 002"
runner._validate_config(source)
dem=pathlib.Path(source["dtm_path"])
if hashlib.sha256(dem.read_bytes()).hexdigest()!=source["dtm_sha256"]:raise ValueError("DTM hash mismatch")
out=W/"configs/full_product_azimuth_attempt002";out.mkdir(parents=True,exist_ok=True)
p=out/"gasa_az384_distance_362.json";p.write_text(json.dumps(source,indent=2,ensure_ascii=False,sort_keys=False)+"\n")
runner._validate_config(json.load(open(p)))
psource=json.load(open(W/"configs/full_product_azimuth_projection_attempt001/gasa_az192_distance_362.json"))
psource["run_id"]+="-AZ384-ATTEMPT-002"
psource["output_dir"]=str(W/"results/full_product_azimuth_projection_attempt002/gasa_az384_distance_362")
psource["validation_products"][0]["state_globs"]=[str(W/"runs/full_product_azimuth/gasa_az384_distance_362/attempt-002/shard*/states/chunk-*.pt")]
po=W/"configs/full_product_azimuth_projection_attempt002";po.mkdir(parents=True,exist_ok=True)
q=po/"gasa_az384_distance_362.json";q.write_text(json.dumps(psource,indent=2,ensure_ascii=False,sort_keys=False)+"\n")
print("THERMAL_CONFIG_PASS",p,"az=384","DTM_SHA256",source["dtm_sha256"])
print("PROJECTION_CONFIG_PASS",q,"output",psource["output_dir"])
