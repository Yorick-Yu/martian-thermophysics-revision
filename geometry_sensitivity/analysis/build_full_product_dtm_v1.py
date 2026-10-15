#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
import numpy as np
from mars_frost.pds_hirise import read_pds3_dtm_spec, bounded_window_start, block_mean_valid
W=Path("/home/yqs/p31-b3-boundary/geometry-sensitivity-20261012")
SITES={
 "gasa":{"raw":W/"inputs/wider_dem/gasa/DTEEC_021584_1440_022217_1440_A01.IMG","baseline":W/"inputs/gasa/DTEEC_021584_1440_022217_1440_A01.GASA_EVENT_256x256_formal_v1.npz","expected_raw_sha256":"f9f438e27fe9bb8f63da2a5778598cc89f2007a7cbb8b4076f29e96b3d9ba485","crop_rows":9616,"crop_columns":4800,"out":W/"inputs/wider_dem/gasa/DTEEC_021584_1440_022217_1440_A01.GASA_FULL_PRODUCT_v1.npz"},
 "kaiser":{"raw":W/"inputs/wider_dem/kaiser/DTEEC_013017_1325_013083_1325_U01.IMG","baseline":W/"inputs/kaiser/DTEEC_013017_1325_013083_1325_U01.KAISER_CRATER_DUNES_EVENT_256x256_formal_v1.npz","expected_raw_sha256":"d0b6fe516453a9d3687bdf596ae9dca153eb9dfe5487b5c99c519b3c61f236f1","crop_rows":11672,"crop_columns":6512,"out":W/"inputs/wider_dem/kaiser/DTEEC_013017_1325_013083_1325_U01.KAISER_FULL_PRODUCT_v1.npz"}}
audit={"schema_version":1,"purpose":"Recover the maximum factor-8 engineering grid available from each identical official HiRISE PDS DTM product, preserving the existing block-mean/validity rule and exact formal-crop reproduction.","sites":{}}
for site,c in SITES.items():
 raw=c["raw"]; spec=read_pds3_dtm_spec(raw); raw_sha=hashlib.sha256(raw.read_bytes()).hexdigest()
 if raw_sha!=c["expected_raw_sha256"]: raise RuntimeError(f"raw source hash mismatch {site}")
 if raw.stat().st_size!=spec.image_offset_bytes+spec.lines*spec.record_bytes: raise RuntimeError(f"source IMG extent mismatch {site}")
 if c["out"].exists(): raise RuntimeError(f"refusing to overwrite {c['out']}")
 with np.load(c["baseline"],allow_pickle=False) as z:
  base_e=z["elevation_m"].copy(); base_v=z["fully_valid"].copy(); meta=json.loads(str(z["metadata_json"].item()))
 if spec.product_id!=meta["product_id"]: raise RuntimeError(f"product ID mismatch {site}")
 full=np.memmap(raw,mode="r",dtype=spec.numpy_dtype,offset=spec.image_offset_bytes,shape=(spec.lines,spec.line_samples),order="C")
 factor=int(meta["downsample_factor"]); old_side=int(meta["raw_side_pixels"])
 old_r=int(meta["downloaded_row_start_zero_based"])+int(meta["raw_crop_row_start_in_download"]); old_c=int(meta["raw_crop_column_start"])
 old_raw=np.asarray(full[old_r:old_r+old_side,old_c:old_c+old_side]).copy()
 old_e,old_v=block_mean_valid(old_raw,factor=factor,valid_minimum_m=spec.valid_minimum_m,valid_maximum_m=spec.valid_maximum_m)
 if old_e.shape!=base_e.shape or not np.array_equal(old_v,base_v) or not np.allclose(old_e[old_v],base_e[base_v],rtol=0,atol=1e-10): raise RuntimeError(f"formal crop does not reproduce for {site}")
 nr,nc=c["crop_rows"],c["crop_columns"]
 if nr%factor or nc%factor or nr>spec.lines or nc>spec.line_samples: raise RuntimeError(f"bad source crop dims for {site}")
 r0=bounded_window_start(float(meta["event_center_line_zero_based"]),side_pixels=nr,extent_pixels=spec.lines)
 c0=bounded_window_start(float(meta["event_center_sample_zero_based"]),side_pixels=nc,extent_pixels=spec.line_samples)
 raw_crop=np.asarray(full[r0:r0+nr,c0:c0+nc]).copy()
 elev,valid=block_mean_valid(raw_crop,factor=factor,valid_minimum_m=spec.valid_minimum_m,valid_maximum_m=spec.valid_maximum_m)
 newmeta=dict(meta)
 newmeta.update({"claim_boundary":"Maximum factor-8 engineering grid reconstructed from the same full official HiRISE PDS DTM product; product-edge limitations remain and are explicitly audited.","downloaded_row_start_zero_based":0,"raw_crop_row_start_in_download":r0,"raw_crop_column_start":c0,"raw_crop_shape":[nr,nc],"engineering_grid_shape":list(elev.shape),"engineering_spacing_m":spec.map_scale_m_per_pixel*factor,"fully_valid_cell_fraction":float(valid.mean()),"raw_full_grid_shape":[spec.lines,spec.line_samples],"raw_pds_sha256":raw_sha,"expanded_crop_window_row_start_zero_based":r0,"expanded_crop_window_column_start_zero_based":c0,"expanded_crop_window_rows":nr,"expanded_crop_window_columns":nc,"grid_role":"FORMAL_E3_FULL_PRODUCT_SENSITIVITY","source_url":meta.get("source_url")})
 newmeta.pop("raw_side_pixels",None)
 np.savez_compressed(c["out"],elevation_m=elev,fully_valid=valid,metadata_json=np.asarray(json.dumps(newmeta,sort_keys=True)))
 row=(float(meta["event_center_line_zero_based"])-r0-0.5*(factor-1))/factor
 col=(float(meta["event_center_sample_zero_based"])-c0-0.5*(factor-1))/factor
 cov=[]
 for abin in range(24):
  az=2*np.pi*abin/24; dist=np.arange(1,363,dtype=float)
  rr=np.rint(row-np.cos(az)*dist).astype(int); cc=np.rint(col+np.sin(az)*dist).astype(int)
  inside=(rr>=0)&(rr<valid.shape[0])&(cc>=0)&(cc<valid.shape[1])
  ok=np.zeros_like(inside); ok[inside]=valid[rr[inside],cc[inside]]
  idx=np.flatnonzero(ok)
  cov.append({"azimuth_bin":abin,"azimuth_deg_clockwise_from_north":abin*15,"nearest_valid_distance_cells":int(idx[0]+1) if idx.size else None,"farthest_valid_distance_cells":int(idx[-1]+1) if idx.size else None,"valid_samples_within_362":int(idx.size),"all_362_samples_valid":bool(ok.all())})
 audit["sites"][site]={"raw_path":str(raw),"raw_size_bytes":raw.stat().st_size,"raw_sha256":raw_sha,"product_id":spec.product_id,"raw_shape":[spec.lines,spec.line_samples],"image_offset_bytes":spec.image_offset_bytes,"record_bytes":spec.record_bytes,"formal_crop_reproduced_exactly":True,"full_crop_start_row":r0,"full_crop_start_column":c0,"full_crop_raw_shape":[nr,nc],"full_crop_engineering_shape":list(elev.shape),"spacing_m":spec.map_scale_m_per_pixel*factor,"valid_fraction":float(valid.mean()),"event_center_engineering_row_col":[row,col],"dtm_path":str(c["out"]),"dtm_sha256":hashlib.sha256(c["out"].read_bytes()).hexdigest(),"dtm_size_bytes":c["out"].stat().st_size,"directional_coverage_to_362":cov,"all_directions_full_362_coverage":all(x["all_362_samples_valid"] for x in cov)}
 del full
out=W/"qa/full_product_dtm_build_audit_v1.json"
if out.exists(): raise RuntimeError(f"refusing to overwrite {out}")
out.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
print(json.dumps({s:{"shape":v["full_crop_engineering_shape"],"valid_fraction":v["valid_fraction"],"center":v["event_center_engineering_row_col"],"min_nearest_cells":min(x["nearest_valid_distance_cells"] or 0 for x in v["directional_coverage_to_362"]),"min_farthest_cells":min(x["farthest_valid_distance_cells"] or 0 for x in v["directional_coverage_to_362"]),"all_directions_full_362":v["all_directions_full_362_coverage"],"dtm_sha256":v["dtm_sha256"]} for s,v in audit["sites"].items()},indent=2))
