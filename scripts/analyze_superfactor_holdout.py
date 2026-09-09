#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat


def scale(ref,pred,w=None):
    ref=np.asarray(ref);pred=np.asarray(pred)
    if w is None:
        den=np.vdot(pred,pred);return np.vdot(pred,ref)/den if abs(den)>0 else 0j
    w=np.asarray(w,float);den=np.sum(w*np.conj(pred)*pred);return np.sum(w*np.conj(pred)*ref)/den if abs(den)>0 else 0j

def score(ref,pred,w=None):
    ref=np.asarray(ref).reshape(-1);pred=np.asarray(pred).reshape(-1)
    if w is None:
      s=scale(ref,pred);q=s*pred;den=np.linalg.norm(ref)*np.linalg.norm(q)
      return {'corr':float(abs(np.vdot(ref,q))/den) if den>0 else np.nan,
              'nmse':float(np.linalg.norm(ref-q)**2/max(np.linalg.norm(ref)**2,1e-30))}
    w=np.asarray(w,float).reshape(-1);s=scale(ref,pred,w);q=s*pred
    den=np.sqrt(np.sum(w*np.abs(ref)**2)*np.sum(w*np.abs(q)**2))
    return {'corr':float(abs(np.sum(w*np.conj(ref)*q))/den) if den>0 else np.nan,
            'nmse':float(np.sum(w*np.abs(ref-q)**2)/max(np.sum(w*np.abs(ref)**2),1e-30))}

def register_from_input(series,full,ii):
    target=full-np.mean(full[ii,:],axis=0,keepdims=True)
    variants={'direct':series,'conjugate':np.conj(series),'reverse':series[::-1,:],'conjugate_reverse':np.conj(series[::-1,:])}
    best=None
    for name,v in variants.items():
      for st in range(v.shape[0]-200+1):
        b=v[st:st+200,:];loss=[]
        for j in range(b.shape[1]):
          s=scale(target[ii,j],b[ii,j]);loss.append(np.linalg.norm(target[ii,j]-s*b[ii,j])**2/max(np.linalg.norm(target[ii,j])**2,1e-30))
        val=float(np.median(loss))
        if best is None or val<best[0]:best=(val,name,st,b)
    val,name,st,b=best
    return {'orientation':name,'start':int(st),'median_train_nmse':val},b,target

def width(v,db,dx):
    y=np.abs(v)**2;p=int(np.argmax(y));h=y[p]*10**(-db/10);l=p;r=p
    while l>0 and y[l]>=h:l-=1
    while r<len(y)-1 and y[r]>=h:r+=1
    if l==p or r==p:return np.nan
    def cr(a,b):return a+(h-y[a])/(y[b]-y[a]) if y[b]!=y[a] else float(a)
    return float((cr(r-1,r)-cr(l,l+1))*dx)

def main():
    src=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
    d=loadmat(src,squeeze_me=True);Nz=int(d['Nz']);supers=np.asarray(d['supers'],int).reshape(-1);ii=np.asarray(d['ii'],int).reshape(-1)-1
    sk=np.asarray(d['sk'],float).reshape(-1);skn=np.abs(sk)/max(float(np.max(np.abs(sk))),1e-30)
    full=np.asarray(d['normalized_full'],complex);cube=np.asarray(d['spectra'],complex)
    idx=np.arange(Nz);held=np.ones(Nz,bool);held[ii]=False
    masks={'all72':held,'near8':held&(((idx>=ii[0]-8)&(idx<ii[0]))|((idx>ii[-1])&(idx<=ii[-1]+8))),
           'source_ge_5pct':held&(skn>=.05),'source_lt_5pct':held&(skn<.05),'left':idx<ii[0],'right':idx>ii[-1]}
    rows=[];registrations=[]
    for si,su in enumerate(supers):
      K=Nz*int(su);series=cube[:K,:,si]
      reg,block,target=register_from_input(series,full,ii);reg|={'super':int(su),'K':K,'support_to_input_ratio':K/len(ii)};registrations.append(reg)
      edge=min(100,K//4);h=np.hanning(2*edge);win=np.ones(K);win[:edge]=h[:edge];win[-edge:]=h[edge:]
      for j in range(series.shape[1]):
        s=scale(target[ii,j],block[ii,j]);pred=s*block[:,j]
        prof=np.fft.ifft(series[:,j]*win)
        base={'super':int(su),'K':K,'support_to_input_ratio':K/len(ii),'line':j,'train_nmse':score(target[ii,j],pred[ii])['nmse'],
              'fwhm3_um':width(prof,3,810.0/K),'width10_um':width(prof,10,810.0/K),'width20_um':width(prof,20,810.0/K),'width40_um':width(prof,40,810.0/K)}
        for name,m in masks.items():
          nm=score(target[m,j],pred[m]);raw=score(target[m,j]*sk[m],pred[m]*sk[m]);weighted=score(target[m,j],pred[m],skn[m]**2)
          rows.append(base|{'region':name,'n_bins':int(m.sum()),'normalized_corr':nm['corr'],'normalized_nmse':nm['nmse'],
                            'raw_corr':raw['corr'],'raw_nmse':raw['nmse'],'source_weighted_corr':weighted['corr'],'source_weighted_nmse':weighted['nmse']})
    df=pd.DataFrame(rows);df.to_csv(out/'per_line.csv',index=False)
    summary=[]
    for (su,reg),q in df.groupby(['super','region']):
      rec={'super':int(su),'region':reg,'n_lines':int(q.line.nunique()),'n_bins':int(q.n_bins.iloc[0]),'support_to_input_ratio':float(q.support_to_input_ratio.iloc[0])}
      for c in ['train_nmse','normalized_corr','normalized_nmse','raw_corr','raw_nmse','source_weighted_corr','source_weighted_nmse','fwhm3_um','width10_um','width20_um','width40_um']:
        rec[c+'_median']=float(np.nanmedian(q[c]))
      summary.append(rec)
    report={'experiment':'exact_author_MIAA_target_support_vs_measured_holdout','registrations':registrations,'summary':summary,
            'boundary':{'heldout':'72 bins are from the deposited 200-bin phase-corrected ROI representation, not raw independent wideband camera samples.',
                        'registration':'orientation/start selected only from the 128 input bins jointly across four A-lines.',
                        'interpretation':'A target support that narrows FWHM without a unique measured-holdout optimum is a model choice, not measured bandwidth.'}}
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
