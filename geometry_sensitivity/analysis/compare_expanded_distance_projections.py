import json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path("/home/yqs/p31-b3-boundary/geometry-sensitivity-20261012")
def metrics(a,b):
    delta=np.asarray(b,dtype=np.float64)-np.asarray(a,dtype=np.float64)
    absolute=np.abs(delta)
    return {"mean_signed_362_minus_128":float(delta.mean()),"rmse":float(np.sqrt(np.mean(delta**2))),"p95_abs":float(np.quantile(absolute,.95)),"p99_abs":float(np.quantile(absolute,.99)),"max_abs":float(absolute.max())}
def classify(gain, lower, corr, u):
    if lower>u and corr>0: return "BEYOND_UNCERTAINTY"
    if gain>0 and lower>0 and corr>0: return "POSITIVE_WITHIN_UNCERTAINTY"
    if gain>0 and lower<=0 and corr>0: return "SIGN_UNCERTAIN"
    return "NO_POSITIVE_EVIDENCE"
report={"schema_version":1,"purpose":"Compare frozen observation projections at 128 versus 362 horizon cells for identical expanded same-product DTM; no refitting or changed projection/calibration.","sites":{}}
for site in ("gasa","kaiser"):
    pair={}
    for distance in (128,362):
        name=f"{site}_distance_{distance}"
        base=ROOT/"results/expanded_range_projection"/name
        result=json.loads((base/"result.json").read_text())
        npz=next(base.glob("*_local_fields.npz"))
        pair[distance]=(result,np.load(npz),npz)
    r0,z0,p0=pair[128]; r1,z1,p1=pair[362]
    if r0["u_val_conservative"]!=r1["u_val_conservative"]: raise RuntimeError(f"Uval changed for {site}")
    prod0=list(r0["by_product"]); prod1=list(r1["by_product"])
    if prod0!=prod1: raise RuntimeError(f"Product mismatch for {site}")
    u=float(r0["u_val_conservative"])
    products={}
    for product in prod0:
        a=r0["by_product"][product]; b=r1["by_product"][product]
        pa=a["state_paths"]; pb=b["state_paths"]
        if len(pa)!=len(pb): raise RuntimeError("state shard count mismatch")
        radiance={}
        for field in ("calibrated_b3","g_b3","observed","g_observed","valid"):
            va=z0[field]; vb=z1[field]
            if va.shape!=vb.shape: raise RuntimeError(f"shape mismatch {site}/{field}")
            if field in ("observed","g_observed","valid") and not np.array_equal(va,vb): raise RuntimeError(f"frozen observation/support changed for {site}/{field}")
            if field in ("calibrated_b3","g_b3"):
                radiance[field]=metrics(va,vb)
        four0=classify(float(a["mean_gain_g"]),float(a["gain_g_lower_95_one_sided"]),float(a["correlation_lower_95_one_sided"]),u)
        four1=classify(float(b["mean_gain_g"]),float(b["gain_g_lower_95_one_sided"]),float(b["correlation_lower_95_one_sided"]),u)
        products[product]={
            "four_state_128":four0,"four_state_362":four1,"four_state_unchanged":four0==four1,
            "mean_gain_g_128":a["mean_gain_g"],"mean_gain_g_362":b["mean_gain_g"],"delta_mean_gain_g":float(b["mean_gain_g"]-a["mean_gain_g"]),
            "gain_lower_95_one_sided_128":a["gain_g_lower_95_one_sided"],"gain_lower_95_one_sided_362":b["gain_g_lower_95_one_sided"],"delta_gain_lower_95_one_sided":float(b["gain_g_lower_95_one_sided"]-a["gain_g_lower_95_one_sided"]),
            "gain_lower_sign_128":int(np.sign(a["gain_g_lower_95_one_sided"])),"gain_lower_sign_362":int(np.sign(b["gain_g_lower_95_one_sided"])),"gain_lower_sign_unchanged":bool(np.sign(a["gain_g_lower_95_one_sided"])==np.sign(b["gain_g_lower_95_one_sided"])),
            "correlation_lower_95_one_sided_128":a["correlation_lower_95_one_sided"],"correlation_lower_95_one_sided_362":b["correlation_lower_95_one_sided"],
            "u_val_conservative":u,"radiance_field_change":radiance,"projection_report_status_128":a["status"],"projection_report_status_362":b["status"],
        }
    report["sites"][site]={
        "dtm_sha256_identical":json.loads((ROOT/"configs/expanded_range_projection"/f"{site}_distance_128.json").read_text())["dtm_sha256"]==json.loads((ROOT/"configs/expanded_range_projection"/f"{site}_distance_362.json").read_text())["dtm_sha256"],
        "frozen_parameters_identical":r0["frozen_design_operator"]==r1["frozen_design_operator"],
        "result_sha256":{"128":hashlib.sha256((ROOT/"results/expanded_range_projection"/f"{site}_distance_128"/"result.json").read_bytes()).hexdigest(),"362":hashlib.sha256((ROOT/"results/expanded_range_projection"/f"{site}_distance_362"/"result.json").read_bytes()).hexdigest()},
        "by_product":products,
    }
out=ROOT/"results/expanded_range_projection_distance_pair_v1.json"
if out.exists(): raise RuntimeError(f"Refusing to overwrite {out}")
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"output":str(out),"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"summary":{s:{p:{"class128":v["four_state_128"],"class362":v["four_state_362"],"class_unchanged":v["four_state_unchanged"],"gain_lb_sign_same":v["gain_lower_sign_unchanged"],"delta_gain":v["delta_mean_gain_g"],"delta_lcb":v["delta_gain_lower_95_one_sided"],"u_val":v["u_val_conservative"],"g_b3_rmse":v["radiance_field_change"]["g_b3"]["rmse"],"g_b3_p95":v["radiance_field_change"]["g_b3"]["p95_abs"],"g_b3_max":v["radiance_field_change"]["g_b3"]["max_abs"]} for p,v in d["by_product"].items()} for s,d in report["sites"].items()}},indent=2))
