from pathlib import Path
import numpy as np,json,math,hashlib
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);args=parser.parse_args();root=args.data/'audit'
def hp(x,s):
 r=max(2,math.ceil(4*s));a=np.arange(-r,r+1);k=np.exp(-.5*(a/s)**2);k/=k.sum();z=x.copy()
 for ax in [-1,-2]:
  pad=[(0,0)]*3;pad[ax]=(r,r);z=np.apply_along_axis(lambda a:np.convolve(a,k,mode='valid'),ax,np.pad(z,pad,mode='reflect'))
 return x-z

def fit(p,o,v):
 x=p[:,v];y=o[:,v];xx=x-x.mean(1)[:,None];yy=y-y.mean(1)[:,None];a=np.clip((xx*yy).sum(1)/np.maximum((xx*xx).sum(1),1e-24),0,1);b=np.maximum(y.mean(1)-a*x.mean(1),0);return a,b

def score(m,o,v):
 m=m[:,v];o=o[:,v];g=np.sqrt((o*o).mean(1))-np.sqrt(((m-o)**2).mean(1));c=[np.corrcoef(a,b)[0,1] for a,b in zip(m,o)];return {'G':float(g.mean()),'C':float(np.mean(c)),'G_bands':g.tolist(),'C_bands':c,'rmse':float(np.sqrt(((m-o)**2).mean(1)).mean())}

def boot(m,o,v,seed,b=None):
 starts=[(i,j) for i in range(15) for j in range(15) if v[i:i+3,j:j+3].all()];draws=math.ceil(v.sum()/9);ch=np.random.default_rng(seed).integers(0,len(starts),(10000,draws));n=draws*9
 def moments(x):return np.stack([x[:,i:i+3,j:j+3].sum((1,2)) for i,j in starts])[ch].sum(1)/n
 mm=moments(m);om=moments(o);m2=moments(m*m);o2=moments(o*o);mo=moments(m*o);err=moments((m-o)**2)
 g=(np.sqrt(o2)-np.sqrt(err)).mean(1);c=((mo-mm*om)/np.sqrt(np.maximum((m2-mm*mm)*(o2-om*om),1e-48))).mean(1)
 gp=np.sqrt(o2.mean(1))-np.sqrt(err.mean(1));mp=mm.mean(1);op=om.mean(1);cp=(mo.mean(1)-mp*op)/np.sqrt((m2.mean(1)-mp*mp)*(o2.mean(1)-op*op))
 q=lambda x:[float(t) for t in np.quantile(x,[.025,.05,.975])]
 r={'gain_quantiles':q(g),'corr_quantiles':q(c),'old_pooled_gain_quantiles':q(gp),'old_pooled_corr_quantiles':q(cp),'blocks':len(starts),'valid_pixels':int(v.sum()),'seed':seed,'resamples':10000}
 if b is not None:r['paired_rmse_improvement_quantiles']=q((np.sqrt(moments((b-o)**2))-np.sqrt(err)).mean(1))
 return r

def state(g,lg,lc,u):
 if lc<=0:return 'not demonstrated'
 if g<=0:return 'pattern only'
 if lg>u:return 'supported beyond uncertainty'
 return 'positive but not separable'

data=json.loads((root/'raw_projection.json').read_text());out={}
local=args.data/'results'
for site,d in data.items():
 obs=np.array(d['observed']);nom={k:np.array(v['prediction']) for k,v in d['nominal'].items()};valid=np.array(d['nominal']['shadow']['valid']);coeff={k:fit(v,obs,valid) for k,v in nom.items()};cal={k:a[:,None,None]*nom[k]+b[:,None,None] for k,(a,b) in coeff.items()};frozen=d['validation_config'];aa=np.array(frozen['frozen_gain']);bb=np.array(frozen['frozen_offset']);assert np.max(abs(coeff['shadow'][0]-aa))<1e-12
 prod=aa[:,None,None]*np.array(d['production_shadow']['prediction'])+bb[:,None,None];common=valid&np.array(d['production_shadow']['valid']);sigmanom=frozen['smoothing_sigma_pixels']
 result={'coefficients':{k:[a.tolist(),b.tolist()] for k,(a,b) in coeff.items()},'calibration_reproduction_gain_error':float(np.max(abs(aa-coeff['shadow'][0]))),'uncertainty':[],'validation':[]}
 for s in [1,1.5,2,3]:
  gn=score(hp(cal['shadow'],s),hp(obs,s),valid)['G'];pert=[score(hp(np.array(p['calibrated']),s),hp(obs,s),np.array(p['valid']))['G'] for p in d['perturbations']];us=float(np.quantile(np.abs(np.array(pert)-gn),.95));un=float(np.sqrt(np.mean((hp(prod,s)[:,common]-hp(cal['shadow'],s)[:,common])**2)));result['uncertainty'].append({'sigma':s,'U_sys':us,'U_num':un,'U_val':us+un})
 for i,item in enumerate(d['validation']):
  ob=np.array(item['observed']);v=np.array(item['shadow']['valid']);assert np.array_equal(v,np.array(item['baseline']['valid']));pred={k:coeff[k][0][:,None,None]*np.array(item[k]['prediction'])+coeff[k][1][:,None,None] for k in ['baseline','shadow']};shared=aa[:,None,None]*np.array(item['baseline']['prediction'])+bb[:,None,None]
  matches=list(local.rglob(item['product']+'*_local_fields.npz'))
  assert len(matches)==1,(item['product'],matches)
  old=np.load(matches[0]);assert np.array_equal(v,old['valid']);diff=float(np.max(abs(old['calibrated_b3']-pred['shadow'])));assert diff<1e-15,diff
  row={'product':item['product'],'reproduction_max_error':diff,'scale_results':[]}
  for s,u in zip([1,1.5,2,3],result['uncertainty']):
   m=hp(pred['shadow'],s);o=hp(ob,s);b=hp(pred['baseline'],s);metrics=score(m,o,v);bm=score(b,o,v);sb=score(hp(shared,s),o,v);bt=boot(m,o,v,int(frozen['bootstrap_seed'])+i,b);row['scale_results'].append({'sigma':s,**metrics,**bt,'U_val':u['U_val'],'state':state(metrics['G'],bt['gain_quantiles'][1],bt['corr_quantiles'][1],u['U_val']),'baseline':bm,'shared_affine_baseline':sb,'paired_rmse_improvement':bm['rmse']-metrics['rmse'],'shared_affine_improvement':sb['rmse']-metrics['rmse']})
  result['validation'].append(row)
 out[site]=result
(args.data/'analysis_recomputed.json').write_text(json.dumps(out,indent=2))
for site,r in out.items():
 print(site,'COEFFICIENTS',r['coefficients'],'U',r['uncertainty'])
 for x in r['validation']:
  print(x['product'],'reproduction error',x['reproduction_max_error'])
  for z in x['scale_results']:print(z['sigma'],z['G'],z['gain_quantiles'][1],z['corr_quantiles'][1],z['U_val'],z['state'],'paired',z['paired_rmse_improvement'],z['paired_rmse_improvement_quantiles'],'shared',z['shared_affine_improvement'])

# Recompute the three archived 128-grid Kaiser checks with the same estimator.
config=json.loads((root/'extra/config.json').read_text());r128={}
for i,path in enumerate(sorted((root/'extra').glob('*_local_fields.npz'))):
 z=np.load(path);m=z['g_b3'];o=z['g_observed'];v=z['valid'].astype(bool);r128[path.name.replace('_local_fields.npz','')]={**score(m,o,v),**boot(m,o,v,int(config['bootstrap_seed'])+i)}
(args.data/'bootstrap_128_recomputed.json').write_text(json.dumps(r128,indent=2))
