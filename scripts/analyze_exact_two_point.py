#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks

def cscale(r,p):
    d=np.vdot(p,p); return np.vdot(p,r)/d if abs(d)>0 else 0j

def corr(r,p):
    d=np.linalg.norm(r)*np.linalg.norm(p); return float(abs(np.vdot(r,p))/d) if d>0 else float('nan')

def nmse(r,p):
    s=cscale(r,p); return float(np.linalg.norm(r-s*p)**2/max(np.linalg.norm(r)**2,1e-30))

def circular_distance(a,b,K):
    d=abs(float(a)-float(b))%K; return min(d,K-d)

def resolve(v,z1,z2,K):
    y=np.abs(v)**2; y/=max(float(y.max()),1e-30)
    peaks,_=find_peaks(y,height=.04,prominence=.025,distance=1); peaks=peaks[np.argsort(y[peaks])[::-1]]
    used=set(); selected=[]; errors=[]
    for target in (z1,z2):
        candidates=[(circular_distance(p,target,K),i,int(p)) for i,p in enumerate(peaks) if i not in used]
        if not candidates: errors.append(np.inf); continue
        error,i,p=min(candidates); used.add(i); selected.append(p); errors.append(error)
    matched=len(selected)==2 and max(errors)<=2 and len(set(selected))==2
    valley_ok=False
    if matched:
        a,b=sorted(selected); segment=y[a:b+1] if b-a<=K//2 else np.r_[y[b:],y[:a+1]]
        valley_ok=float(segment.min())<=.85*min(float(y[a]),float(y[b]))
    extras=max(int(np.sum(y[peaks]>.2))-(2 if matched else 0),0)
    return bool(matched and valley_ok and extras==0),bool(matched),int(extras),float(max(errors))

def main():
    data=loadmat(sys.argv[1],squeeze_me=True); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
    K=int(data['K']); Ng=int(data['Ng']); z1=int(data['z1']); meta=np.atleast_2d(data['meta'])
    rows=[]
    for method,key in [('DFT','prof_dft'),('IAA','prof_iaa'),('MIAA','prof_miaa')]:
        arr=np.asarray(data[key],complex)
        for j,m in enumerate(meta):
            sep,phase,ratio,snr,seed=m; ok,match,extra,error=resolve(arr[:,j],z1,z1+int(sep),K)
            rows.append({'method':method,'separation_bins':int(sep),'separation_over_dft_resolution':float(sep/(K/Ng)),'relative_phase_rad':float(phase),'amplitude_ratio':float(ratio),'snr_db':float(snr),'seed':int(seed),'resolved':ok,'position_match':match,'extra_strong_peaks':extra,'max_position_error_bins':error})
    frame=pd.DataFrame(rows); frame.to_csv(out/'per_case.csv',index=False)
    summary=[]
    for (method,sep),group in frame.groupby(['method','separation_bins']):
        summary.append({'method':method,'separation_bins':int(sep),'separation_over_dft_resolution':float(group.separation_over_dft_resolution.iloc[0]),'n':int(len(group)),'resolve_rate':float(group.resolved.mean()),'position_match_rate':float(group.position_match.mean()),'false_extra_rate':float((group.extra_strong_peaks>0).mean())})
    add_meta=np.atleast_2d(data['add_meta']); pair=np.asarray(data['add_out_pair'],complex); summed=np.asarray(data['add_out_sum'],complex)
    additivity=[{'separation_bins':int(m[0]),'amplitude_ratio':float(m[1]),'relative_phase_rad':float(m[2]),'aligned_nmse':nmse(pair[:,j],summed[:,j]),'complex_corr':corr(pair[:,j],summed[:,j])} for j,m in enumerate(add_meta)]
    report={'schema':'exact-author-two-target-audit-v1','status':'VERIFIED_SYNTHETIC','experiment':'deposited_fiaa_plus_paper_mmse_miaa_two_target','K':K,'Ng':Ng,'summary':summary,'additivity':additivity,'decisions':{'single_point_fwhm_is_resolution_proof':False,'universal_linear_psf_supported':bool(max(x['aligned_nmse'] for x in additivity)<1e-3)},'boundary':'Synthetic normalized spectra; deposited fiaa_oct_c1 supplies the IAA state and paper Eq. 10-12 supplies missing-data prediction.'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    plt.figure(figsize=(8.4,5.2))
    for method in ('DFT','IAA','MIAA'):
        q=[x for x in summary if x['method']==method]
        plt.plot([x['separation_over_dft_resolution'] for x in q],[x['resolve_rate'] for x in q],marker='o',label=method)
    plt.ylim(-.03,1.03); plt.xlabel('separation / conventional DFT resolution'); plt.ylabel('registered resolve rate'); plt.title('Two-target resolution, not single-point FWHM'); plt.legend(); plt.tight_layout(); plt.savefig(out/'two_target_resolution.png',dpi=180); plt.close()
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
