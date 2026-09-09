#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks

def cscale(r,p):
 d=np.vdot(p,p);return np.vdot(p,r)/d if abs(d)>0 else 0j
def corr(r,p):
 d=np.linalg.norm(r)*np.linalg.norm(p);return float(abs(np.vdot(r,p))/d) if d>0 else np.nan
def nmse(r,p):
 s=cscale(r,p);return float(np.linalg.norm(r-s*p)**2/max(np.linalg.norm(r)**2,1e-30))
def cd(a,b,K):
 d=abs(float(a)-float(b))%K;return min(d,K-d)
def resolve(v,z1,z2,K):
 y=np.abs(v)**2;y/=max(float(y.max()),1e-30)
 pk,_=find_peaks(y,height=.04,prominence=.025,distance=1);pk=pk[np.argsort(y[pk])[::-1]]
 used=set();sel=[];errs=[]
 for t in (z1,z2):
  q=[(cd(p,t,K),i,int(p)) for i,p in enumerate(pk) if i not in used]
  if not q:errs.append(np.inf);continue
  e,i,p=min(q);used.add(i);sel.append(p);errs.append(e)
 match=len(sel)==2 and max(errs)<=2 and len(set(sel))==2;valley=False
 if match:
  a,b=sorted(sel);seg=y[a:b+1] if b-a<=K//2 else np.r_[y[b:],y[:a+1]]
  valley=float(seg.min())<=.85*min(float(y[a]),float(y[b]))
 extra=max(int(np.sum(y[pk]>.2))-(2 if match else 0),0)
 return match and valley and extra==0,match,extra,float(max(errs))
def main():
 d=loadmat(sys.argv[1],squeeze_me=True);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
 K=int(d['K']);z1=int(d['z1']);meta=np.atleast_2d(d['meta']);rows=[]
 for method,key in [('DFT','prof_dft'),('IAA','prof_iaa'),('MIAA','prof_miaa')]:
  arr=np.asarray(d[key],complex)
  for j,m in enumerate(meta):
   sep,ph,ratio,snr,seed=m;ok,match,extra,err=resolve(arr[:,j],z1,z1+int(sep),K)
   rows.append({'method':method,'separation_bins':int(sep),'separation_over_dft_resolution':float(sep/(K/int(d['Ng']))),'phase_rad':float(ph),'amplitude_ratio':float(ratio),'snr_db':float(snr),'seed':int(seed),'resolved':bool(ok),'position_match':bool(match),'extra_strong_peaks':int(extra),'max_position_error_bins':err})
 df=pd.DataFrame(rows);df.to_csv(out/'per_case.csv',index=False);summary=[]
 for (method,sep),q in df.groupby(['method','separation_bins']):
  summary.append({'method':method,'separation_bins':int(sep),'separation_over_dft_resolution':float(q.separation_over_dft_resolution.iloc[0]),'n':int(len(q)),'resolve_rate':float(q.resolved.mean()),'position_match_rate':float(q.position_match.mean()),'false_extra_rate':float((q.extra_strong_peaks>0).mean())})
 am=np.atleast_2d(d['add_meta']);pair=np.asarray(d['add_out_pair'],complex);summ=np.asarray(d['add_out_sum'],complex);add=[]
 for j,m in enumerate(am):add.append({'separation_bins':int(m[0]),'amplitude_ratio':float(m[1]),'phase_rad':float(m[2]),'aligned_nmse':nmse(pair[:,j],summ[:,j]),'complex_corr':corr(pair[:,j],summ[:,j])})
 def threshold(method):
  vals=[r['separation_over_dft_resolution'] for r in summary if r['method']==method and r['resolve_rate']>=.8]
  return min(vals) if vals else None
 report={'schema':'exact-author-two-target-audit-v1','status':'VERIFIED_SYNTHETIC','experiment':'deposited FIAA plus paper MMSE MIAA','K':K,'Ng':int(d['Ng']),'Nfull':int(d['Nfull']),'summary':summary,'additivity':add,'adjudication':{'dft_80pct_threshold_over_dft_resolution':threshold('DFT'),'iaa_80pct_threshold_over_dft_resolution':threshold('IAA'),'miaa_80pct_threshold_over_dft_resolution':threshold('MIAA'),'maximum_miaa_additivity_nmse':float(max(x['aligned_nmse'] for x in add)),'universal_linear_psf_supported':bool(max(x['aligned_nmse'] for x in add)<1e-3),'single_point_fwhm_is_resolution_proof':False},'boundary':'Synthetic normalized spectra; exact deposited FIAA estimates the power state and the paper MMSE equation predicts missing data.'}
 (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
