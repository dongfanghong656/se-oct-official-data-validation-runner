#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

def fwhm(a,dx):
 y=np.abs(a)**2;y=np.maximum(y-np.percentile(y,5),0);y/=max(y.max(),1e-30);p=int(np.argmax(y));l=p;r=p
 while l>0 and y[l]>=.5:l-=1
 while r<len(y)-1 and y[r]>=.5:r+=1
 if l==p or r==p:return float('nan')
 def c(i,j):return i+(.5-y[i])/(y[j]-y[i]) if y[j]!=y[i] else i
 return float((c(r-1,r)-c(l,l+1))*dx)
def main():
 d=loadmat(sys.argv[1],squeeze_me=True);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True);sup=np.asarray(d['supers'],int).ravel();nz=int(d['Nz']);rows=[]
 for si,s in enumerate(sup):
  K=nz*s;dx=810/K
  for line in range(d['rfiaa'].shape[1]):
   rows.append({'super':int(s),'line':line,'rfiaa_fwhm_um':fwhm(d['rfiaa'][:K,line,si],dx),'map_fwhm_um':fwhm(d['rfmap'][:K,line,si],dx),'miaa_fwhm_um':fwhm(d['miaa'][:K,line,si],dx)})
 df=pd.DataFrame(rows);df.to_csv(out/'per_line.csv',index=False);med=df.groupby('super').median(numeric_only=True).reset_index();med.to_csv(out/'medians.csv',index=False)
 report={'experiment':'exact_author_super_factor_sensitivity','definition':'super selects K=200*super and the extrapolated MIAA support; all A-line data and q_i=10 are fixed','medians':med.to_dict(orient='records'),'per_line':rows,'boundary':'Peak width dependence on chosen model support is not evidence that any particular outside-200 spectrum is physically true.'}
 (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 plt.figure(figsize=(8.5,5.3))
 for col,label in [('rfiaa_fwhm_um','RFIAA'),('map_fwhm_um','RFIAA+MAP'),('miaa_fwhm_um','MIAA')]:plt.plot(med.super,med[col],marker='o',label=label)
 plt.xlabel('author super factor');plt.ylabel('median FWHM (µm)');plt.title('Exact-author peak width versus selected extrapolation factor');plt.legend();plt.tight_layout();plt.savefig(out/'superfactor_width.png',dpi=180);plt.close();print(json.dumps(report,indent=2))
if __name__=='__main__':main()
