import hashlib,json,pathlib
import numpy as np
W=pathlib.Path("/home/yqs/p31-b3-boundary/geometry-sensitivity-20261012")
base_counts=(24,48,96,192)
def result_paths(site,n):
 if n==384:return W/"results/full_product_azimuth_projection_attempt006/gasa_az384_distance_362"
 if n==768:return W/"results/full_product_azimuth_projection_attempt007/gasa_az768_distance_362"
 if n==24:return W/f"results/full_product_projection_attempt002_v2/{site}_distance_362"
 return W/f"results/full_product_azimuth_projection_attempt001/{site}_az{n}_distance_362"
def config_path(site,n):
 if n==384:return W/"configs/full_product_azimuth_projection_attempt006/gasa_az384_distance_362.json"
 if n==768:return W/"configs/full_product_azimuth_projection_attempt007/gasa_az768_distance_362.json"
 if n==24:return W/f"configs/full_product_projection_attempt002_v2/{site}_distance_362.json"
 return W/f"configs/full_product_azimuth_projection_attempt001/{site}_az{n}_distance_362.json"
def met(a,b):
 d=np.asarray(b,dtype=np.float64)-np.asarray(a,dtype=np.float64); z=np.abs(d)
 return {"mean_signed_new_minus_old":float(d.mean()),"rmse":float(np.sqrt(np.mean(d*d))),"p95_abs":float(np.quantile(z,.95)),"p99_abs":float(np.quantile(z,.99)),"max_abs":float(z.max())}
def classify(gain,lb,corr,u):
 if lb>u and corr>0:return "BEYOND_UNCERTAINTY"
 if gain>0 and lb>0 and corr>0:return "POSITIVE_WITHIN_UNCERTAINTY"
 if gain>0 and lb<=0 and corr>0:return "SIGN_UNCERTAIN"
 return "NO_POSITIVE_EVIDENCE"
report={"schema_version":1,"purpose":"Frozen projection across full-product DTM horizon-azimuth refinements at 362 cells through Gasa 768 azimuths; compare local fields, conservative uncertainty, one-sided lower bound and four-state classification. No retraining or refitting.","azimuth_counts":{"gasa":[24,48,96,192,384,768],"kaiser":[24,48,96,192]},"sites":{}}
for site in ("gasa","kaiser"):
 counts=base_counts+((384,768) if site=="gasa" else ())
 vals={}
 for n in counts:
  p=result_paths(site,n); r=json.load(open(p/"result.json")); npz=next(p.glob("*_local_fields.npz")); z=np.load(npz)
  cfg=json.load(open(config_path(site,n)))
  if cfg["validation_products"][0]["state_globs"][0].find("full_product")<0:raise RuntimeError(f"wrong glob {site}/{n}")
  vals[n]=(r,z,cfg)
 uvals={float(vals[n][0]["u_val_conservative"]) for n in counts}
 if len(uvals)!=1:raise RuntimeError(f"Uval differs {site}")
 products=list(vals[24][0]["by_product"])
 if any(list(vals[n][0]["by_product"])!=products for n in counts):raise RuntimeError(f"products differ {site}")
 u=next(iter(uvals)); site_item={"u_val_conservative":u,"by_product":{}}
 for prod in products:
  pr={}
  for n in counts:
   r,z,cfg=vals[n]
   if r["by_product"][prod]["status"]=="ERROR":raise RuntimeError(f"projection error {site}/{n}")
   pr[n]=(r["by_product"][prod],z)
  fields={}
  for fld in ("calibrated_b3","g_b3","observed","g_observed","valid"):
   arr={n:pr[n][1][fld] for n in counts}
   if any(arr[n].shape!=arr[24].shape for n in counts):raise RuntimeError(f"shape {site}/{fld}")
   if fld in ("observed","g_observed","valid") and any(not np.array_equal(arr[n],arr[24]) for n in counts[1:]):raise RuntimeError(f"frozen obs changed {site}/{fld}")
   if fld in ("calibrated_b3","g_b3"):
    fields[fld]={"adjacent":{},"vs_24":{}}
    for n in counts[1:]:
     fields[fld]["adjacent"][f"{n//2}_to_{n}"]=met(arr[n//2],arr[n])
     fields[fld]["vs_24"][str(n)]=met(arr[24],arr[n])
  cls={}
  for n in counts:
   a=pr[n][0]; cls[n]=classify(float(a["mean_gain_g"]),float(a["gain_g_lower_95_one_sided"]),float(a["correlation_lower_95_one_sided"]),u)
  evidence={"classification":{str(n):cls[n] for n in counts},"classification_unchanged":len(set(cls.values()))==1,"metrics":{}}
  for n in counts:
   a=pr[n][0]
   evidence["metrics"][str(n)]={"mean_gain_g":a["mean_gain_g"],"gain_lower_95_one_sided":a["gain_g_lower_95_one_sided"],"gain_lower_sign":int(np.sign(a["gain_g_lower_95_one_sided"])),"correlation_lower_95_one_sided":a["correlation_lower_95_one_sided"]}
  evidence["gain_lower_sign_unchanged"]=len({int(np.sign(pr[n][0]["gain_g_lower_95_one_sided"])) for n in counts})==1
  transitions={}
  for n in counts[1:]:
   a,b=pr[n//2][0],pr[n][0]
   transitions[f"{n//2}_to_{n}"]={"delta_mean_gain_g":float(b["mean_gain_g"]-a["mean_gain_g"]),"delta_gain_lower_95_one_sided":float(b["gain_g_lower_95_one_sided"]-a["gain_g_lower_95_one_sided"]),"gain_lower_sign_unchanged":bool(np.sign(a["gain_g_lower_95_one_sided"])==np.sign(b["gain_g_lower_95_one_sided"])),"classification_unchanged":cls[n//2]==cls[n],"u_val_conservative":u}
  site_item["by_product"][prod]={"radiance_field_change":fields,"evidence":evidence,"adjacent_evidence_changes":transitions}
 report["sites"][site]=site_item
out=W/"results/full_product_azimuth_projection_refinement_v3.json"
if out.exists():raise RuntimeError(f"refusing overwrite {out}")
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
summary={s:{p:{"classes":v["evidence"]["classification"],"lb_sign_same":v["evidence"]["gain_lower_sign_unchanged"],"g_b3_adjacent":{k:{"rmse":d["rmse"],"p95":d["p95_abs"],"max":d["max_abs"]} for k,d in v["radiance_field_change"]["g_b3"]["adjacent"].items()}} for p,v in x["by_product"].items()} for s,x in report["sites"].items()}
print(json.dumps({"output":str(out),"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"summary":summary},indent=2))
