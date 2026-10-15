import hashlib,json,pathlib,sys
W=pathlib.Path('/home/yqs/p31-b3-boundary/geometry-sensitivity-20261012')
sys.path.insert(0,str(W/'code'))
import run_torch_gasa_shadow as runner
source=json.load(open(W/'configs/full_product_azimuth_attempt002/gasa_az384_distance_362.json'))
source['experiment']=source['experiment'].replace('attempt 002','attempt 003')
runner._validate_config(source)
dem=pathlib.Path(source['dtm_path'])
if hashlib.sha256(dem.read_bytes()).hexdigest()!=source['dtm_sha256']:
    raise ValueError('DTM hash mismatch')
THERM=W/'configs/full_product_azimuth_attempt003/gasa_az384_distance_362.json'
THERM.parent.mkdir(parents=True,exist_ok=True)
THERM.write_text(json.dumps(source,indent=2,ensure_ascii=False,sort_keys=False)+'\n')
runner._validate_config(json.load(open(THERM)))
psource=json.load(open(W/'configs/full_product_azimuth_projection_attempt002/gasa_az384_distance_362.json'))
psource['run_id']=psource['run_id'].replace('ATTEMPT-002','ATTEMPT-003')
psource['output_dir']=str(W/'results/full_product_azimuth_projection_attempt003/gasa_az384_distance_362')
psource['validation_products'][0]['state_globs']=[str(W/'runs/full_product_azimuth/gasa_az384_distance_362/attempt-003/shard*/states/chunk-*.pt')]
PROJ=W/'configs/full_product_azimuth_projection_attempt003/gasa_az384_distance_362.json'
PROJ.parent.mkdir(parents=True,exist_ok=True)
PROJ.write_text(json.dumps(psource,indent=2,ensure_ascii=False,sort_keys=False)+'\n')
print('THERMAL_CONFIG_PASS',THERM,'azimuth',source['horizon_azimuth_count'],'distance',source['horizon_maximum_distance_cells'],'dtm_sha256',source['dtm_sha256'])
print('PROJECTION_CONFIG_WRITTEN',PROJ,'run_id',psource['run_id'],'output',psource['output_dir'])
