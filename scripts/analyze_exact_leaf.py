#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

def fwhm(x,dx):
 y=np.abs(x)**2
 if not np.isfinite(y).all() or y.max()<=0:return float('nan')
 y/=y.max();p=int(np.argmax(y));l=p;r=p
 while l>0 and y[l]>=.5:l-=1
 while r<len(y)-1 and y[r]>=.5:r+=1
 if l==p or r==p:return float('nan')
 def c(a,b):return a+(.5-y[a])/(y[b]-y[a]) if y[b]!=y[a] else float(a)
 return float((c(r-1,r)-c(l,l+1))*dx)
def cscale(a,b):
 den=np.vdot(b,b);return np.vdot(b,a)/den if abs(den)>0 else 0j
def cm(a,b):
 a=np.asarray(a).ravel();b=np.asarray(b).ravel();s=cscale(a,b);bb=s*b;den=np.linalg.norm(a)*np.linalg.norm(bb)
 return {'corr':float(abs(np.vdot(a,bb))/den) if den>0 else float('nan'),'nmse':float(np.linalg.norm(a-bb)**2/max(np.linalg.norm(a)**2,1e-30)),'phase_rmse':float(np.sqrt(np.mean(np.angle(a*np.conj(bb))**2)))}
def choose(pred,full,ii):
 target=full-np.mean(full[ii]);best=None
 for name,s in [('direct',pred),('conj',np.conj(pred)),('reverse',pred[::-1]),('conj_reverse',np.conj(pred[::-1]))]:
  for st in range(len(s)-len(target)+1):
   block=s[st:st+len(target)];scale=cscale(target[ii],block[ii]);loss=np.linalg.norm(target[ii]-scale*block[ii])**2/max(np.linalg.norm(target[ii])**2,1e-30)
   if best is None or loss<best[0]:best=(loss,name,st,scale,block,target)
 return best

def main():
 d=loadmat(sys.argv[1],squeeze_me=True);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True);K=int(d['K']);dx=810/K;ii=np.asarray(d['ii'],int).ravel()-1
 rows=[]
 for j in range(d['dft_profile'].shape[1]):
  m=choose(d['miaa_spectrum'][:,j],d['normalized_full'][:,j],ii);_,ori,st,scale,block,target=m;held=np.ones(200,bool);held[ii]=False;met=cm(target[held],scale*block[held])
  row={'line':j,'group':int(np.ravel(d['group_id'])[j]),'score':float(np.ravel(d['selection_score'])[j]),'coordinate_yx0':np.asarray(d['coords_yx0'])[j].astype(int).tolist(),'mapping_orientation':ori,'mapping_start':int(st),**{f'holdout_{k}':v for k,v in met.items()}}
  for key in ['dft_profile','normalized_dft_profile','rfiaa_profile','rfiaa_map_profile','miaa_profile']:row[f'fwhm_{key}_um']=fwhm(d[key][:,j],dx)
  rows.append(row)
 df=pd.DataFrame([{k:(json.dumps(v) if isinstance(v,list) else v) for k,v in r.items()} for r in rows]);df.to_csv(out/'per_line.csv',index=False)
 ag=df.groupby('group').median(numeric_only=True).reset_index();ag.to_csv(out/'group_medians.csv',index=False)
 report={'experiment':'exact_author_FIAA_MIAA_on_deterministic_leaf_A_lines','boundary':'phase-corrected leaf Cscan; score-stratified A-lines, not the exact Figure-5 fit line','groups':{'1':'low score','2':'middle score','3':'high score'},'group_medians':ag.to_dict(orient='records'),'all_median':df.median(numeric_only=True).to_dict(),'per_line':rows,'interpretation':'Generalization is supported only where holdout complex prediction and profile behavior remain stable across score groups; narrow FWHM alone is not accepted.'}
 (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 plt.figure(figsize=(8.5,5.5));
 for g,label in [(1,'low'),(2,'mid'),(3,'high')]:
  q=df[df.group==g];plt.scatter(q.score,q.holdout_corr,label=label)
 plt.xscale('log');plt.xlabel('leaf A-line score');plt.ylabel('128→200 holdout complex correlation');plt.title('Exact MIAA representation prediction versus leaf signal level');plt.legend();plt.tight_layout();plt.savefig(out/'holdout_vs_signal.png',dpi=180);plt.close()
 print(json.dumps(report['group_medians'],indent=2))
if __name__=='__main__':main()
