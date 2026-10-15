#!/usr/bin/env python3
import json,hashlib
from pathlib import Path
import numpy as np
from mars_frost.pds_hirise import read_pds3_dtm_spec,bounded_window_start,block_mean_valid
W=Path("/home/yqs/p31-b3-boundary/geometry-sensitivity-20261012")
SITES={
 "gasa":{"raw":W/"inputs/wider_dem/gasa/DTEEC_021584_1440_022217_1440_A01.IMG","baseline":W/"inputs/gasa/DTEEC_021584_1440_022217_1440_A01.GASA_EVENT_256x256_formal_v1.npz","crop_rows":4096,"crop_columns":4800,"out":W/"inputs/wider_dem/gasa/DTEEC_021584_1440_022217_1440_A01.GASA_EVENT_512x600_expanded_v1.npz"},
 "kaiser":{"raw":W/"inputs/wider_dem/kaiser/DTEEC_013017_1325_013083_1325_U01.IMG","baseline":W/"inputs/kaiser/DTEEC_013017_1325_013083_1325_U01.KAISER_CRATER_DUNES_EVENT_256x256_formal_v1.npz","crop_rows":6400,"crop_columns":6400,"out":W/"inputs/wider_dem/kaiser/DTEEC_013017_1325_013083_1325_U01.KAISER_CRATER_DUNES_EVENT_800x800_expanded_v1.npz"}}
audit={"schema_version":1,"method":"Read original official PDS HiRISE DTM IMG, retain factor-8 block mean and fully-valid rule, center expanded crop using same source event pixel. Verify the existing 256x256 crop reproduces exactly before writing expanded maps.","sites":{}}
for site,c in SITES.items():
 raw=c["raw"]; baseline=c["baseline"]; spec=read_pds3_dtm_spec(raw)
 raw_sha=hashlib.sha256(raw.read_bytes()).hexdigest()
 if raw.stat().st_size != spec.image_offset_bytes+spec.lines*spec.record_bytes: raise RuntimeError(f"raw size/label mismatch for {site}")
 with np.load(baseline,allow_pickle=False) as z:
  base_e=z["elevation_m"].copy();base_v=z["fully_valid"].copy();meta=json.loads(str(z["metadata_json"].item()))
 if spec.product_id != meta["product_id"]:raise RuntimeError(f"product ID mismatch {site}")
 if spec.lines != int(meta["input_inventory"]["source_lines"]) if "source_lines" in meta["input_inventory"] else False:raise RuntimeError("unexpected source lines")
 full=np.memmap(raw,mode="r",dtype=spec.numpy_dtype,offset=spec.image_offset_bytes,shape=(spec.lines,spec.line_samples),order="C")
 factor=int(meta["downsample_factor"])
 old_rows=int(meta["raw_side_pixels"])
 old_r=int(meta["downloaded_row_start_zero_based"])+int(meta["raw_crop_row_start_in_download"])
 old_c=int(meta["raw_crop_column_start"])
 old_crop=np.asarray(full[old_r:old_r+old_rows,old_c:old_c+old_rows]).copy()
 old_e,old_v=block_mean_valid(old_crop,factor=factor,valid_minimum_m=spec.valid_minimum_m,valid_maximum_m=spec.valid_maximum_m)
 if old_e.shape!=base_e.shape or not np.array_equal(old_v,base_v) or not np.allclose(old_e[old_v],base_e[base_v],rtol=0,atol=1e-10):raise RuntimeError(f"original crop failed exact source reproduction for {site}")
 nr,nc=int(c["crop_rows"]),int(c["crop_columns"])
 if nr%factor or nc%factor or nr>spec.lines or nc>spec.line_samples:raise RuntimeError("expanded crop dimensions invalid")
 center_line=float(meta["event_center_line_zero_based"]);center_sample=float(meta["event_center_sample_zero_based"])
 r0=bounded_window_start(center_line,side_pixels=nr,extent_pixels=spec.lines)
 c0=bounded_window_start(center_sample,side_pixels=nc,extent_pixels=spec.line_samples)
 crop=np.asarray(full[r0:r0+nr,c0:c0+nc]).copy()
 elev,valid=block_mean_valid(crop,factor=factor,valid_minimum_m=spec.valid_minimum_m,valid_maximum_m=spec.valid_maximum_m)
 newmeta=dict(meta)
 newmeta.update({"claim_boundary":"Expanded engineering crop reconstructed from the same full official HiRISE PDS DTM product; not a different terrain source.","downloaded_row_start_zero_based":0,"raw_crop_row_start_in_download":r0,"raw_crop_column_start":c0,"raw_crop_shape":[nr,nc],"engineering_grid_shape":list(elev.shape),"engineering_spacing_m":spec.map_scale_m_per_pixel*factor,"fully_valid_cell_fraction":float(valid.mean()),"raw_full_grid_shape":[spec.lines,spec.line_samples],"raw_pds_sha256":raw_sha,"expanded_crop_window_row_start_zero_based":r0,"expanded_crop_window_column_start_zero_based":c0,"expanded_crop_window_rows":nr,"expanded_crop_window_columns":nc,"source_url":meta.get("source_url")})
 newmeta.pop("raw_side_pixels",None)
 out=c["out"];out.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(out,elevation_m=elev,fully_valid=valid,metadata_json=np.asarray(json.dumps(newmeta,sort_keys=True)))
 audit["sites"][site]={"raw_path":str(raw),"raw_size_bytes":raw.stat().st_size,"raw_sha256":raw_sha,"product_id":spec.product_id,"raw_shape":[spec.lines,spec.line_samples],"image_offset_bytes":spec.image_offset_bytes,"record_bytes":spec.record_bytes,"old_crop_exact_reproduction":True,"expanded_crop_row_start":r0,"expanded_crop_column_start":c0,"expanded_raw_shape":[nr,nc],"expanded_engineering_shape":list(elev.shape),"engineering_spacing_m":spec.map_scale_m_per_pixel*factor,"valid_fraction":float(valid.mean()),"event_center_engineering_xy":[(center_line-r0-0.5*(factor-1))/factor,(center_sample-c0-0.5*(factor-1))/factor],"expanded_dtm_path":str(out),"expanded_dtm_size_bytes":out.stat().st_size,"expanded_dtm_sha256":hashlib.sha256(out.read_bytes()).hexdigest()}
 del full
p=W/"qa/expanded_dtm_build_audit.json";p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
print(json.dumps(audit,indent=2))
