#!/usr/bin/env python3
"""Exact-author RFIAA traversal and MIAA target-support audit.

The 72 non-input bins belong to the same phase-corrected finite-depth representation; they are
an internal consistency check, not independent raw-camera or wider-band truth.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

EPS=1e-30

def cscale(r,p,w=None):
 r=np.asarray(r).ravel();p=np.asarray(p).ravel()
 if w is None:
  d=np.vdot(p,p);return np.vdot(p,r)/d if abs(d)>0 else 0j
 w=np.asarray(w,float).ravel();d=np.sum(w*np.conj(p)*p)
 return np.sum(w*np.conj(p)*r)/d if abs(d)>0 else 0j

def cm(r,p,w=None,align=True):
 r=np.asarray(r).ravel();p=np.asarray(p).ravel()
 if not len(r):return {k:np.nan for k in ('corr','nmse','phase_rmse_rad','amplitude_nrmse')}
 w=None if w is None else np.asarray(w,float).ravel();q=(cscale(r,p,w) if align else 1)*p
 if w is None:
  er=np.sum(abs(r)**2);ep=np.sum(abs(q)**2);cross=np.vdot(r,q);err=np.sum(abs(r-q)**2);ae=np.sum((abs(r)-abs(q))**2);pr=np.sqrt(np.mean(np.angle(r*np.conj(q))**2))
 else:
  er=np.sum(w*abs(r)**2);ep=np.sum(w*abs(q)**2);cross=np.sum(w*np.conj(r)*q);err=np.sum(w*abs(r-q)**2);ae=np.sum(w*(abs(r)-abs(q))**2);pr=np.sqrt(np.sum(w*np.angle(r*np.conj(q))**2)/max(np.sum(w),EPS))
 return {'corr':float(abs(cross)/np.sqrt(max(er*ep,EPS))),'nmse':float(err/max(er,EPS)),'phase_rmse_rad':float(pr),'amplitude_nrmse':float(np.sqrt(ae/max(er,EPS)))}

def orient(a,name):
 return {'direct':a,'conjugate':np.conj(a),'reverse':a[::-1,...],'conjugate_reverse':np.conj(a[::-1,...])}[name]

def refview(a,n,offset):
 j=np.arange(n);k=j+offset;valid=(k>=0)&(k<a.shape[0]);b=np.full((n,)+a.shape[1:],np.nan+1j*np.nan);b[valid]=a[k[valid]];return b,valid,k

def register(spectrum,target,ii):
 if spectrum.ndim==1:spectrum=spectrum[:,None]
 if target.ndim==1:target=target[:,None]
 best=[];K=spectrum.shape[0]
 for name in ('direct','conjugate','reverse','conjugate_reverse'):
  a=orient(spectrum,name)
  for off in range(-int(ii[0]),K-int(ii[-1])):
   losses=[]
   for line in range(target.shape[1]):
    p=a[ii+off,line];s=cscale(target[ii,line],p);losses.append(np.linalg.norm(target[ii,line]-s*p)**2/max(np.linalg.norm(target[ii,line])**2,EPS))
   best.append((float(np.median(losses)),float(np.quantile(losses,.9)),name,off))
 best.sort();med,p90,name,off=best[0];a=orient(spectrum,name);b,valid,k=refview(a,target.shape[0],off)
 return {'orientation':name,'offset_zero_based':int(off),'train_nmse_median':med,'train_nmse_p90':p90,'valid_reference_bins':int(valid.sum()),'input_bins_all_valid':bool(valid[ii].all()),'runner_up_median':float(best[1][0]),'selection_basis':'input bins only, jointly across lines'},b,a,valid,k

def masks(sk,ii):
 n=len(sk);j=np.arange(n);held=np.ones(n,bool);held[ii]=0;s=abs(sk)/max(abs(sk).max(),EPS);near=held&(((j>=max(0,ii[0]-8))&(j<ii[0]))|((j>ii[-1])&(j<=min(n-1,ii[-1]+8))))
 return {'input_128':~held,'heldout_all_72':held,'heldout_near_16':near,'heldout_far_56':held&~near,'heldout_source_ge_5pct':held&(s>=.05),'heldout_source_lt_5pct':held&(s<.05),'heldout_left':j<ii[0],'heldout_right':j>ii[-1]}

def addmetrics(row,r,p,sk,m):
 r=r[m];p=p[m];sn=abs(sk[m])/max(abs(sk).max(),EPS);rw=sn**2*abs(r)**2;rw/=max(rw.max() if len(rw) else 0,EPS)
 for prefix,v in [('normalized',cm(r,p)),('source_restored',cm(r*sk[m],p*sk[m])),('source_weighted',cm(r,p,sn**2)),('signal_source_weighted',cm(r,p,rw))]:
  for k,x in v.items():row[prefix+'_'+k]=x

def q(a):
 a=np.asarray(a,float);a=a[np.isfinite(a)]
 return {k:(float(v) if len(a) else np.nan) for k,v in zip(('min','p10','median','p90','max'),np.quantile(a,[0,.1,.5,.9,1]) if len(a) else [np.nan]*5)}

def summary(df,groups,metrics):
 out=[]
 for keys,g in df.groupby(groups,dropna=False):
  if not isinstance(keys,tuple):keys=(keys,)
  r=dict(zip(groups,keys));r['n_rows']=len(g);r['n_lines']=g.line.nunique() if 'line' in g else len(g)
  for m in metrics:r[m]=q(g[m])
  out.append(r)
 return out

def taper(n,e):
 e=min(max(int(e),0),n//2)
 if not e:return np.ones(n)
 h=np.hanning(2*e);w=np.ones(n);w[:e]=h[:e];w[-e:]=h[e:];return w

def width(v,dx,db):
 y=abs(np.asarray(v).ravel())**2
 if not np.isfinite(y).all() or y.max()<=0:return np.nan
 y=np.maximum(y-np.quantile(y,.01),0);p=int(y.argmax());h=y[p]*10**(-db/10);l=p;r=p
 while l>0 and y[l]>=h:l-=1
 while r<len(y)-1 and y[r]>=h:r+=1
 if l==p or r==p:return np.nan
 def c(a,b):return a+(h-y[a])/(y[b]-y[a]) if y[b]!=y[a] else float(a)
 return float((c(r-1,r)-c(l,l+1))*dx)

def expected(K,Nz,su,ii):
 first=int(ii[0])+1;shift=round(Nz*((su-1)/2))+first-1
 return {'orientation':'reverse','start_zero_based':int(K-shift-first),'shift_amount_matlab':int(shift)}

def analyze_order(path):
 d=loadmat(path,squeeze_me=True);Nz=int(d['Nz']);K=int(d['K']);su=int(d['super']);ii=np.asarray(d['ii'],int).ravel()-1;sk=np.asarray(d['sk'],float).ravel();target=np.asarray(d['anC'],complex);target-=target[ii].mean(axis=0,keepdims=True)
 spectra={'forward_whole_strip':np.asarray(d['spectra_forward'],complex),'reverse_whole_strip':np.asarray(d['spectra_reverse'],complex),'independent_q10':np.asarray(d['spectra_independent'],complex)}
 chunk=bool(int(d['chunk_exact_ok']))
 if chunk:spectra|={'forward_author_L4':np.asarray(d['spectra_chunk_forward'],complex),'reverse_author_L4':np.asarray(d['spectra_chunk_reverse'],complex)}
 mp,_,_,valid,k=register(spectra['independent_q10'],target,ii);ex=expected(K,Nz,su,ii);mp|={'author_formula_expectation':ex,'matches_author_formula':mp['orientation']==ex['orientation'] and mp['offset_zero_based']==ex['start_zero_based']}
 oriented={n:orient(a,mp['orientation']) for n,a in spectra.items()};blocks={n:refview(a,Nz,mp['offset_zero_based'])[0] for n,a in oriented.items()};base={n:m&valid for n,m in masks(sk,ii).items()}
 scores=np.asarray(d['line_scores'],float).ravel();x=np.asarray(d['x_indices_zero_based'],int).ravel();strong=int(d['strong_idx'])-1;hold=[]
 for line in range(target.shape[1]):
  for mode,b in blocks.items():
   est=cscale(target[ii,line],b[ii,line])*b[:,line]
   for region,m in base.items():
    if not m.any():continue
    r={'line':line,'x_zero_based':int(x[line]),'mode':mode,'region':region,'n_bins':int(m.sum()),'signal_score':float(scores[line]),'position_class':'edge' if min(line,target.shape[1]-1-line)<6 else 'interior','side_of_strong_line':'before' if line<strong else ('after' if line>strong else 'strong')};addmetrics(r,target[:,line],est,sk,m);hold.append(r)
 embedded=np.zeros(K,bool);embedded[k[valid]]=1;inp=np.zeros(K,bool);inp[ii+mp['offset_zero_based']]=1;regions={'input_128':inp,'embedded_200':embedded,'heldout_72':embedded&~inp,'extrapolated_outside_200':~embedded,'full_K':np.ones(K,bool)}
 comps=[('whole_forward_vs_reverse','forward_whole_strip','reverse_whole_strip'),('whole_forward_vs_independent','forward_whole_strip','independent_q10'),('whole_reverse_vs_independent','reverse_whole_strip','independent_q10')]
 if chunk:comps += [('author_L4_forward_vs_reverse','forward_author_L4','reverse_author_L4'),('author_L4_forward_vs_independent','forward_author_L4','independent_q10'),('author_L4_reverse_vs_independent','reverse_author_L4','independent_q10')]
 pairs=[]
 for line in range(target.shape[1]):
  for label,rn,pn in comps:
   ref=oriented[rn][:,line];pred=oriented[pn][:,line];pred=cscale(ref[inp],pred[inp])*pred
   for region,m in regions.items():pairs.append({'line':line,'comparison':label,'region':region,'position_class':'edge' if min(line,target.shape[1]-1-line)<6 else 'interior','side_of_strong_line':'before' if line<strong else ('after' if line>strong else 'strong'),**cm(ref[m],pred[m],align=False)})
 h=pd.DataFrame(hold);p=pd.DataFrame(pairs);hm=['normalized_corr','normalized_nmse','source_restored_corr','source_restored_nmse','source_weighted_corr','source_weighted_nmse','signal_source_weighted_corr','signal_source_weighted_nmse'];pm=['corr','nmse','phase_rmse_rad','amplitude_nrmse']
 result={'experiment':'exact_author_RFIAA_traversal_order_on_committed_TiO2_strip','boundary':{'input':'checksum-traced phase-corrected public TiO2 subset','holdout':'same-pipeline finite-ROI bins, not independent physical wideband truth'},'configuration':{'Nz':Nz,'K':K,'super':su,'strip_lines':target.shape[1],'strong_line_zero_based':strong,'mapping':mp,'chunk_exact_ok':chunk,'chunk_error':str(d.get('chunk_exact_error',''))},'holdout_summary':summary(h,['mode','region'],hm),'pair_summary':summary(p,['comparison','region','position_class'],pm),'decision_rules':['Input fit is necessary but does not validate extrapolation.','Forward/reverse disagreement after input alignment is traversal-history dependence.','FWHM is not an order-invariance criterion.']}
 return result,h,p

def analyze_super(path):
 d=loadmat(path,squeeze_me=True);Nz=int(d['Nz']);supers=np.asarray(d['supers'],int).ravel();ii=np.asarray(d['ii'],int).ravel()-1;sk=np.asarray(d['sk'],float).ravel();target=np.asarray(d['anC4'],complex);target-=target[ii].mean(axis=0,keepdims=True);cube=np.asarray(d['spectra_super'],complex);base=masks(sk,ii);items={}
 for si,su0 in enumerate(supers):
  su=int(su0);K=Nz*su;sp=cube[:K,:,si];mp,b,_,valid,_=register(sp,target,ii);ex=expected(K,Nz,su,ii);mp|={'super':su,'K':K,'author_formula_expectation':ex,'matches_author_formula':mp['orientation']==ex['orientation'] and mp['offset_zero_based']==ex['start_zero_based']};items[su]=(sp,b,valid,mp)
 common=np.logical_and.reduce([v[2] for v in items.values()]);rows=[]
 for su,(sp,b,valid,mp) in items.items():
  K=Nz*su;wins={'none':np.ones(K),'fixed100':taper(K,100),'proportional12p5':taper(K,round(K/8))};evalm={**{n:m&valid for n,m in base.items()},**{'common_'+n:m&common for n,m in base.items()}}
  for line in range(target.shape[1]):
   est=cscale(target[ii,line],b[ii,line])*b[:,line];wd={}
   for wn,w in wins.items():
    pr=np.fft.ifft(sp[:,line]*w)
    for db in (3,10,20,40):wd[f'{wn}_w{db}db_um']=width(pr,810/K,db)
   for region,m in evalm.items():
    if not m.any():continue
    r={'super':su,'K':K,'line':line,'region':region,'n_bins':int(m.sum()),'support_to_input_ratio':K/len(ii),**wd};addmetrics(r,target[:,line],est,sk,m);rows.append(r)
 f=pd.DataFrame(rows);metrics=['source_restored_corr','source_restored_nmse','source_weighted_corr','source_weighted_nmse','signal_source_weighted_corr','signal_source_weighted_nmse','fixed100_w3db_um','fixed100_w20db_um','fixed100_w40db_um','proportional12p5_w3db_um'];sm=summary(f,['super','region'],metrics);g=f[f.region=='common_heldout_source_ge_5pct'];cand=[]
 for su,x in g.groupby('super'):cand.append({'super':int(su),'n_common_bins':int(x.n_bins.iloc[0]),'source_restored_nmse_median':float(np.median(x.source_restored_nmse)),'signal_source_weighted_nmse_median':float(np.median(x.signal_source_weighted_nmse)),'fixed_taper_fwhm_median_um':float(np.median(x.fixed100_w3db_um))})
 rr=sorted(cand,key=lambda x:x['source_restored_nmse_median']);rs=sorted(cand,key=lambda x:x['signal_source_weighted_nmse_median'])
 def margin(a,k):return (a[1][k]-a[0][k])/max(abs(a[0][k]),EPS) if len(a)>1 else None
 both=rr[0]['super']==4 and rs[0]['super']==4;sel={'evaluation_region':'common_heldout_source_ge_5pct','candidates':cand,'best_super_source_restored':rr[0]['super'],'best_super_signal_source_weighted':rs[0]['super'],'relative_margin_source_restored':margin(rr,'source_restored_nmse_median'),'relative_margin_signal_weighted':margin(rs,'signal_source_weighted_nmse_median'),'super4_selected_by_both':both,'super4_uniquely_selected_by_5pct_rule':both and margin(rr,'source_restored_nmse_median')>=.05 and margin(rs,'signal_source_weighted_nmse_median')>=.05,'scientific_status':'INTERNAL_SUPPORT_DIAGNOSTIC_ONLY'}
 return {'experiment':'exact_author_MIAA_target_support_on_committed_TiO2_lines','common_valid_reference_bins':int(common.sum()),'common_valid_heldout_bins':int((common&base['heldout_all_72']).sum()),'registrations':[v[3] for v in items.values()],'summary':sm,'support_selection_diagnostic':sel,'boundary':{'holdout':'same-pipeline finite-ROI bins, not independent wider-band truth','registration':'input bins only','comparison':'same common evaluable bins for all support factors','spacing':'approximately 810-um pre-ISAM ROI divided by K'}},f

def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.integer):return int(x)
 if isinstance(x,(np.floating,float)):
  v=float(x);return v if math.isfinite(v) else None
 return x

def write(path,x):path.write_text(json.dumps(clean(x),indent=2,allow_nan=False),encoding='utf-8')

def selftest():
 rng=np.random.default_rng(7);ii=np.arange(23,151);t=rng.normal(size=(200,3))+1j*rng.normal(size=(200,3))
 for K,off in ((400,100),(400,-10),(800,453)):
  a=rng.normal(size=(K,3))+1j*rng.normal(size=(K,3));j=np.arange(200);v=(j+off>=0)&(j+off<K);a[j[v]+off]=t[v];mp,b,_,valid,_=register(a[::-1],t,ii);assert mp['orientation']=='reverse' and mp['offset_zero_based']==off and valid[ii].all() and np.allclose(b[ii],t[ii])
 print('SELF_TEST_OK')

def main():
 if len(sys.argv)==2 and sys.argv[1]=='--self-test':selftest();return
 if len(sys.argv)!=4:raise SystemExit('usage: analyze_committed_subset_audit.py ORDER_MAT SUPER_MAT OUT | --self-test')
 out=Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True);o,h,p=analyze_order(Path(sys.argv[1]));s,f=analyze_super(Path(sys.argv[2]));h.to_csv(out/'order_holdout_per_line.csv',index=False);p.to_csv(out/'order_pair_per_line.csv',index=False);f.to_csv(out/'superfactor_per_line.csv',index=False);write(out/'order_metrics.json',o);write(out/'superfactor_metrics.json',s);write(out/'metrics.json',{'schema_version':'2.0','order_audit':o,'superfactor_audit':s});print(json.dumps(clean({'order_mapping':o['configuration']['mapping'],'support_selection':s['support_selection_diagnostic']}),indent=2))
if __name__=='__main__':main()
