from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();R=Path(__file__).resolve().parent
D=args.data/'results'
A=args.data/'audit'
OUT=args.output;OUT.mkdir(exist_ok=True,parents=True)
a=json.load(open(A/'analysis.json'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':400,'pdf.fonttype':42,'figure.facecolor':'white','axes.edgecolor':'#30343A','text.color':'#202124','axes.labelcolor':'#202124'})
BLUE='#314858';STEEL='#536F82';PALE='#BBC8D0';RED='#D94332';ORANGE='#F0A21A';GRID='#D9DEE1'
def save(f,n):
 f.savefig(OUT/f'Figure_{n:02d}.pdf',bbox_inches='tight',pad_inches=.09)
 f.savefig(OUT/f'Figure_{n:02d}.png',bbox_inches='tight',pad_inches=.09)
 plt.close(f)
def panel(ax,s):ax.text(-.08,1.06,s,transform=ax.transAxes,fontweight='bold',fontsize=11)
def box(ax,x,y,w,h,t):
 ax.add_patch(Rectangle((x,y),w,h,fc='white',ec=STEEL,lw=1.3));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=11,linespacing=1.5)
def arrow(ax,p,q):ax.annotate('',q,p,arrowprops={'arrowstyle':'-|>','color':BLUE,'lw':1.8})
f,ax=plt.subplots(figsize=(9,5.5));ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off')
box(ax,.2,4.5,2.5,1,'HiRISE terrain\nSolar geometry')
box(ax,3.45,4.5,2.8,1,'Vertical thermal columns\nLocal incidence\nTerrain-horizon shadow')
box(ax,7,4.5,2.8,1,'Saved temperature fields\nTwo model branches')
arrow(ax,(2.7,5),(3.45,5));arrow(ax,(6.25,5),(7,5))
box(ax,7,2.5,2.8,1.1,'Radiance conversion\nPSF + pixel integration\nRegistration')
arrow(ax,(8.4,4.5),(8.4,3.6))
box(ax,3.45,2.5,2.8,1.1,'Calibration image\nAffine correction\nLocal-field scale')
arrow(ax,(6.25,3.05),(7,3.05))
box(ax,.2,2.5,2.5,1.1,'THEMIS observations\n100 m pixels\nTwo infrared bands')
arrow(ax,(2.7,3.05),(3.45,3.05))
box(ax,3.45,.2,6.35,1.3,'Fixed-parameter evaluation\nLocal-pattern gain versus zero field\nPaired comparison versus local-slope baseline')
arrow(ax,(8.4,2.5),(8.4,1.5));arrow(ax,(1.45,2.5),(1.45,.85));arrow(ax,(1.45,.85),(3.45,.85));save(f,1)
# Numerical diagnostics
s=json.load(open(D/'e3_satellite_scale_q32.json'))
f,axs=plt.subplots(1,2,figsize=(8.6,3.6),layout='constrained')
ax=axs[0];ax.bar([0,1],[.4284268242,.15380557],color=[PALE,BLUE]);ax.set(xticks=[0,1],xticklabels=['300 s','150 s'],ylabel='Surface temperature P95 error (K)',xlabel='Time step; 75 s reference');panel(ax,'a)')
ax=axs[1];x=np.arange(2);raw=[s['by_grid_relative_to_256'][str(k)]['relative_raw_radiance_p95']*100 for k in [64,128]];loc=[s['by_grid_relative_to_256'][str(k)]['relative_local_structure_p95']*100 for k in [64,128]]
ax.bar(x-.17,raw,.34,color=BLUE,label='Radiance');ax.bar(x+.17,loc,.34,color=PALE,label='Local field');ax.axhline(5,color=RED,ls='--',label='5% criterion');ax.set(xticks=x,xticklabels=['64 vs 256','128 vs 256'],ylabel='Relative P95 difference (%)',xlabel='Cells per side');ax.legend(frameon=False,fontsize=10);panel(ax,'b)');save(f,2)
# Local radiance arrays, shared observed/model range per row; absolute residual.
def fields(rows,n):
 f,axs=plt.subplots(len(rows),3,figsize=(8.8,2.7*len(rows)),squeeze=False,layout='constrained')
 for i,(path,band,label) in enumerate(rows):
  p=np.load(path);v=p['valid'].astype(bool);o=np.where(v,p['g_observed'][band]*1e6,np.nan);m=np.where(v,p['g_b3'][band]*1e6,np.nan);lo=np.nanmin([o,m]);hi=np.nanmax([o,m]);res=np.abs(m-o)
  for j,z in enumerate([o,m,res]):
   ax=axs[i,j];im=ax.imshow(z,cmap='viridis' if j<2 else 'magma',vmin=lo if j<2 else 0,vmax=hi if j<2 else np.nanmax(res),origin='lower',extent=[0,1.7,0,1.7]);ax.set(xlabel='Window x (km)',ylabel='Window y (km)');ax.set_title(f'{chr(97+i*3+j)}) '+['Observed','Terrain-shadow model','Absolute residual'][j],loc='left',fontsize=11)
   cb=f.colorbar(im,ax=ax,fraction=.046,pad=.025);cb.ax.tick_params(labelsize=10);cb.set_label(r'$10^{-6}$ W cm$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$',fontsize=10)
  axs[i,0].text(0,1.19,label,transform=axs[i,0].transAxes,fontsize=10,fontweight='bold')
 save(f,n)
fields([(D/'e4_fields/gasa_design_local_fields.npz',0,'Gasa calibration, band 1'),(D/'e4_fields/gasa_design_local_fields.npz',1,'Gasa calibration, band 2')],3)
fields([(D/'e5_fields/I97665004SNU_local_fields.npz',0,'Gasa I97665004SNU'),(D/'e5_fields/I88618002SNU_local_fields.npz',0,'Gasa I88618002SNU')],4)
# Archived degradation (no bootstrap reinterpretation)
bd=json.load(open(D/'e6_gasa_boundary_expanded.json'));rec=bd['records']
zero=lambda v:abs(v)<1e-8
worst=lambda r:min(x['gain_g_lower_95'] for x in r['by_product'].values())*1e6
pix=[r for r in rec if zero(r['additional_blur_sigma_m']) and zero(r['registration_east_m']) and zero(r['registration_south_m']) and zero(r['noise_ratio'])]
noise=[r for r in rec if r['pixel_scale_m']==100 and zero(r['additional_blur_sigma_m']) and zero(r['registration_east_m']) and zero(r['registration_south_m'])]
reg=[r for r in rec if r['pixel_scale_m']==100 and zero(r['additional_blur_sigma_m']) and zero(r['noise_ratio'])]
f,axs=plt.subplots(2,2,figsize=(8.7,6.8),layout='constrained')
for ax,rs,key,lab in [(axs[0,0],pix,'pixel_scale_m','Effective pixel scale (m)'),(axs[0,1],noise,'noise_ratio','Added noise / local-signal SD')]:
 rs.sort(key=lambda r:r[key]);ax.plot([r[key]for r in rs],[worst(r)for r in rs],'-o',color=BLUE);ax.axhline(0,color='black',lw=.8);ax.set(xlabel=lab,ylabel=r'Minimum gain statistic ($10^{-6}$)')
ax=axs[1,0];vals=[worst(r) for r in reg];ax.scatter([np.hypot(r['registration_east_m'],r['registration_south_m'])for r in reg],vals,c=[BLUE if v>0 else RED for v in vals],s=30);ax.axhline(0,color='black',lw=.8);ax.set(xlabel='Registration displacement (m)',ylabel=r'Minimum gain statistic ($10^{-6}$)')
ax=axs[1,1];rows=[x['scale_results'][-1]for x in a['Kaiser']['validation']];rat=[x['gain_quantiles'][1]/x['U_val']for x in rows];ax.bar(range(3),rat,color=[RED,BLUE,RED]);ax.axhline(1,color='black',ls='--');ax.axhline(0,color='black',lw=.8);ax.set(xticks=range(3),xticklabels=['Morning','Matched\nevening','Later\nevening'],ylabel=r'Corrected $L_G/U_{val}$')
for i,ax in enumerate(axs.flat):panel(ax,f'{chr(97+i)})');ax.grid(axis='y',color=GRID,lw=.5);ax.set_axisbelow(True)
save(f,5)
# Terrain comparison
ter=json.load(open(D/'terrain_comparison.json'))
f,axs=plt.subplots(2,2,figsize=(8.7,6.5),layout='constrained')
for i,site in enumerate(['gasa','kaiser']):
 p=np.load(D/('e4_gasa_3d/gasa_3d_b3_fields.npz' if i==0 else 'e6_kaiser_3d/kaiser_3d_b3_fields.npz'));z=np.where(p['valid'],p['elevation_m'],np.nan);z-=np.nanmin(z);ax=axs[0,i];im=ax.imshow(z,origin='lower',extent=[0,2.056 if i==0 else 2.046,0,2.056 if i==0 else 2.046],cmap='viridis',vmin=0,vmax=950);ax.set(xlabel='Terrain x (km)',ylabel='Terrain y (km)',title=f'{chr(97+i)}) {site.title()}');f.colorbar(im,ax=ax,label='Relief above local minimum (m)',fraction=.046,pad=.025)
x=np.arange(2)
for i,(key,lab) in enumerate([('b3_local_field_coherence_length_m_by_band','Coherence length (m)'),('b3_local_field_abs_p95_by_band',r'Amplitude ($10^{-6}$ radiance units)')]):
 ax=axs[1,i]
 for j,site in enumerate(['gasa','kaiser']):
  v=np.array(ter[site][key])*(1e6 if i else 1);ax.bar(x+(-.18 if j==0 else .18),v,.36,color=[PALE,BLUE][j],label=site.title())
 ax.set(xticks=x,xticklabels=['Band 1','Band 2'],ylabel=lab);ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(0,1.20),ncol=2,fontsize=9);panel(ax,f'{chr(99+i)})')
save(f,6)
kfiles=sorted((D/'e6_kaiser_fields').glob('IA*_local_fields.npz'))
fields([(p,0,lab)for p,lab in zip(kfiles,['Kaiser morning','Kaiser matched evening','Kaiser later evening'])],7)
# Direct baseline paired confidence intervals
labels=[];rs=[]
for site in ['Gasa','Kaiser']:
 for item in a[site]['validation']:
  labels.append(site+' '+item['product'].split('_')[0]);rs.append(item['scale_results'][2 if site=='Gasa' else 3])
f,ax=plt.subplots(figsize=(8.5,4.4),layout='constrained');yy=np.arange(5);point=np.array([r['paired_rmse_improvement']for r in rs])*1e7;lo=np.array([r['paired_rmse_improvement_quantiles'][0]for r in rs])*1e7;hi=np.array([r['paired_rmse_improvement_quantiles'][2]for r in rs])*1e7
ax.errorbar(point,yy,xerr=[point-lo,hi-point],fmt='o',color=BLUE,capsize=4);ax.scatter([r['paired_rmse_improvement_quantiles'][1]*1e7 for r in rs],yy,marker='|',color=RED,s=160,label='One-sided 95% lower bound');ax.axvline(0,color='black',ls='--',lw=1);ax.set(yticks=yy,yticklabels=labels,xlabel=r'Baseline RMSE minus terrain-shadow RMSE ($10^{-7}$ radiance units)');ax.invert_yaxis();ax.grid(axis='x',color=GRID);f.legend(*ax.get_legend_handles_labels(),frameon=False,loc='outside lower center',fontsize=10);save(f,8)
# scale sensitivity
f,axs=plt.subplots(1,2,figsize=(8.8,4.8),layout='constrained')
for site in ['Gasa','Kaiser']:
 for i,item in enumerate(a[site]['validation']):
  rs=item['scale_results'];axs[0].plot([r['sigma']for r in rs],[r['gain_quantiles'][1]/r['U_val']for r in rs],marker=['o','s','^','D','v'][len(axs[0].lines)],color=[BLUE,STEEL,'#89949B',RED,'#667E8B'][len(axs[0].lines)],label=site+' '+item['product'].split('_')[0],lw=1.6)
axs[0].axhline(1,color='black',ls='--');axs[0].set(xlabel='Gaussian local-field scale (pixels)',ylabel=r'$L_G/U_{val}$',xticks=[1,1.5,2,3]);f.legend(*axs[0].get_legend_handles_labels(),fontsize=9,frameon=False,loc='outside lower center',ncol=2);panel(axs[0],'a)')
rs=a['Kaiser']['validation'][1]['scale_results'];ax=axs[1];ax.plot([r['sigma']for r in rs],[r['gain_quantiles'][1]*1e6 for r in rs],'-o',color=BLUE,label=r'Evening $L_G$');ax.plot([r['sigma']for r in rs],[r['U_val']*1e6 for r in rs],'--s',color=RED,label=r'$U_{val}$');ax.set(xlabel='Gaussian local-field scale (pixels)',ylabel=r'Radiance ($10^{-6}$ units)',xticks=[1,1.5,2,3]);ax.legend(frameon=False);panel(ax,'b)');save(f,9)
# model schematic; all physically meaningful arrows are local vertical conduction.
f,axs=plt.subplots(1,2,figsize=(8.9,4.8),layout='constrained')
for i,ax in enumerate(axs):
 ax.set(xlim=(0,10),ylim=(0,8));ax.axis('off');ax.text(0,7.5,f'{chr(97+i)}) '+['Local-slope column baseline','Terrain-shadow column model'][i],fontsize=12,fontweight='bold')
 x=np.array([.5,2.5,4.5,6.5,8.5]);y=np.array([5,4,5.8,4.2,4.6]);ax.plot(x,y,color=BLUE,lw=2)
 for j in range(4):
  xx=(x[j]+x[j+1])/2;yy=(y[j]+y[j+1])/2
  for k in range(5):ax.add_patch(Rectangle((xx-.55,yy-.48*(k+1)),1.1,.45,fc=PALE if k%2 else '#E1E7EB',ec=STEEL,lw=.7))
  ax.annotate('',(xx,yy-.65),(xx,yy-1.9),arrowprops={'arrowstyle':'<->','color':BLUE,'lw':1.1})
  if not(i==1 and j==2):ax.annotate('',(xx,yy+.1),(xx-.85,yy+1.25),arrowprops={'arrowstyle':'-|>','color':ORANGE,'lw':1.8})
  else:ax.text(xx+.25,yy+.7,'Shadowed',ha='center',fontsize=9,color=BLUE)
 ax.text(.6,.35,'Independent vertical conduction\nSurface heat capacity + frost balance',fontsize=10,linespacing=1.5)
 if i:ax.text(.5,6.8,'Horizon test modifies direct sunlight',fontsize=10)
 else:ax.text(.5,6.8,'Direct sunlight uses the local normal',fontsize=10)
save(f,10)
print('Created 10 figures')
