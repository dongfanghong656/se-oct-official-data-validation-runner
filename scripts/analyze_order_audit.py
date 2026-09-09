#!/usr/bin/env python3
"""Analyze exact-author RFIAA/MIAA recursion order and robust held-out fidelity."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat


def cscale(ref: np.ndarray, pred: np.ndarray, w: np.ndarray | None = None) -> complex:
    ref=np.asarray(ref); pred=np.asarray(pred)
    if w is None:
        den=np.vdot(pred,pred); return np.vdot(pred,ref)/den if abs(den)>0 else 0j
    w=np.asarray(w,float)
    den=np.sum(w*np.conj(pred)*pred)
    return np.sum(w*np.conj(pred)*ref)/den if abs(den)>0 else 0j


def metrics(ref: np.ndarray, pred: np.ndarray, w: np.ndarray | None = None) -> dict[str,float]:
    ref=np.asarray(ref).reshape(-1); pred=np.asarray(pred).reshape(-1)
    if w is not None: w=np.asarray(w,float).reshape(-1)
    s=cscale(ref,pred,w); q=s*pred
    if w is None:
        den=np.linalg.norm(ref)*np.linalg.norm(q)
        corr=abs(np.vdot(ref,q))/den if den>0 else np.nan
        nmse=np.linalg.norm(ref-q)**2/max(np.linalg.norm(ref)**2,1e-30)
        phase=np.sqrt(np.mean(np.angle(ref*np.conj(q))**2))
        amp=np.linalg.norm(np.abs(ref)-np.abs(q))/max(np.linalg.norm(np.abs(ref)),1e-30)
    else:
        sw=max(float(np.sum(w)),1e-30)
        den=np.sqrt(np.sum(w*np.abs(ref)**2)*np.sum(w*np.abs(q)**2))
        corr=abs(np.sum(w*np.conj(ref)*q))/den if den>0 else np.nan
        nmse=np.sum(w*np.abs(ref-q)**2)/max(np.sum(w*np.abs(ref)**2),1e-30)
        phase=np.sqrt(np.sum(w*np.angle(ref*np.conj(q))**2)/sw)
        amp=np.sqrt(np.sum(w*(np.abs(ref)-np.abs(q))**2)/max(np.sum(w*np.abs(ref)**2),1e-30))
    return {'corr':float(corr),'nmse':float(nmse),'phase_rmse_rad':float(phase),'amplitude_nrmse':float(amp),
            'scale_real':float(np.real(s)),'scale_imag':float(np.imag(s))}


def fixed_mapping(K: int, Nz: int, ii: np.ndarray, pred: np.ndarray, full: np.ndarray) -> dict[str,object]:
    """Use the indexing implied by the deposited circshift/flip construction; do not fit it per line."""
    ii_first_1based=int(ii[0])+1
    shift_amount=round(Nz*((4-1)/2))+ii_first_1based-1
    start=K-shift_amount-ii_first_1based
    mapping={'orientation':'reverse','start':int(start),'ii_first_1based':ii_first_1based,
             'shift_amount':int(shift_amount),'derivation':'start = K - shift_amount - ii_first_1based'}
    block=pred[::-1,:][start:start+200,:]
    target=full-np.mean(full[ii,:],axis=0,keepdims=True)
    losses=[]
    for j in range(block.shape[1]):
        sc=cscale(target[ii,j],block[ii,j])
        losses.append(np.linalg.norm(target[ii,j]-sc*block[ii,j])**2/max(np.linalg.norm(target[ii,j])**2,1e-30))
    mapping['median_train_nmse']=float(np.median(losses))
    mapping['max_train_nmse']=float(np.max(losses))
    return mapping

def mapped_block(pred: np.ndarray, mapping: dict[str,object]) -> np.ndarray:
    name=str(mapping['orientation'])
    series={'direct':pred,'conjugate':np.conj(pred),'reverse':pred[::-1,:],
            'conjugate_reverse':np.conj(pred[::-1,:])}[name]
    st=int(mapping['start'])
    return series[st:st+200,:]


def bootstrap_ci(values: list[float], seed: int=12345, nboot: int=4000) -> list[float]:
    a=np.asarray([x for x in values if np.isfinite(x)],float)
    if a.size<2: return [float('nan'),float('nan')]
    rng=np.random.default_rng(seed)
    med=np.median(a[rng.integers(0,a.size,size=(nboot,a.size))],axis=1)
    return [float(np.quantile(med,.025)),float(np.quantile(med,.975))]


def analyze(path: Path) -> dict[str,object]:
    d=loadmat(path,squeeze_me=True,struct_as_record=False)
    kind=str(d['kind']).strip()
    K=int(d['K']); ii=np.asarray(d['ii'],int).reshape(-1)-1
    sk=np.asarray(d['sk'],float).reshape(-1); skn=np.abs(sk)/max(float(np.max(np.abs(sk))),1e-30)
    full=np.asarray(d['normalized_full'],complex)
    if full.ndim==2: full=full[:,:,None]
    modes={m:np.asarray(d[f'spectra_{m}'],complex) for m in ('forward','reverse','independent')}
    for m in modes:
        if modes[m].ndim==2: modes[m]=modes[m][:,:,None]
    nstrip=full.shape[1]; nstrips=full.shape[2]
    full2=full.reshape(200,-1,order='F')
    modes2={m:v.reshape(K,-1,order='F') for m,v in modes.items()}
    line_scores=np.asarray(d['line_scores'],float)
    if line_scores.ndim==1: line_scores=line_scores[:,None]
    scores=line_scores.reshape(-1,order='F')
    groups=np.repeat(np.asarray(d['coords_meta'].y_group,int).reshape(-1),nstrip)
    strips=np.repeat(np.arange(nstrips),nstrip)
    line_in_strip=np.tile(np.arange(nstrip),nstrips)
    edge_distance=np.minimum(line_in_strip,nstrip-1-line_in_strip)
    position_class=np.where(edge_distance>=5,'interior','edge')

    mapping=fixed_mapping(K,200,ii,modes2['independent'],full2)
    blocks={m:mapped_block(v,mapping) for m,v in modes2.items()}
    idx=np.arange(200); held=np.ones(200,bool); held[ii]=False
    near=held & (((idx>=ii[0]-8)&(idx<ii[0])) | ((idx>ii[-1])&(idx<=ii[-1]+8)))
    region_masks={
      'heldout_all':held,
      'heldout_near8':near,
      'heldout_far':held & ~near,
      'heldout_source_ge_5pct':held & (skn>=.05),
      'heldout_source_lt_5pct':held & (skn<.05),
      'heldout_left':idx<ii[0],
      'heldout_right':idx>ii[-1],
    }
    rows=[]
    for j in range(full2.shape[1]):
        target=full2[:,j]-np.mean(full2[ii,j])
        for mode,block in blocks.items():
            s=cscale(target[ii],block[ii,j]); pred=s*block[:,j]
            train=metrics(target[ii],pred[ii])
            base={'dataset':kind,'flat_line':j,'strip':int(strips[j]),'line_in_strip':int(line_in_strip[j]),
                  'score_group':int(groups[j]),'signal_score':float(scores[j]),'edge_distance':int(edge_distance[j]),
                  'position_class':str(position_class[j]),'mode':mode,'train_nmse':train['nmse']}
            for region,mask in region_masks.items():
                if not np.any(mask): continue
                nmet=metrics(target[mask],pred[mask])
                rmet=metrics(target[mask]*sk[mask],pred[mask]*sk[mask])
                wmet=metrics(target[mask],pred[mask],w=skn[mask]**2)
                rows.append(base|{'region':region,'n_bins':int(mask.sum()),
                          'normalized_corr':nmet['corr'],'normalized_nmse':nmet['nmse'],
                          'normalized_phase_rmse_rad':nmet['phase_rmse_rad'],
                          'raw_corr':rmet['corr'],'raw_nmse':rmet['nmse'],
                          'raw_phase_rmse_rad':rmet['phase_rmse_rad'],
                          'source_weighted_corr':wmet['corr'],'source_weighted_nmse':wmet['nmse'],
                          'source_weighted_phase_rmse_rad':wmet['phase_rmse_rad']})
    df=pd.DataFrame(rows)

    pair_rows=[]
    pair_defs=[('forward_vs_reverse','forward','reverse'),('forward_vs_independent','forward','independent'),
               ('reverse_vs_independent','reverse','independent')]
    for j in range(full2.shape[1]):
      for label,a,b in pair_defs:
        pair_rows.append({'dataset':kind,'flat_line':j,'strip':int(strips[j]),'line_in_strip':int(line_in_strip[j]),
                          'score_group':int(groups[j]),'signal_score':float(scores[j]),'edge_distance':int(edge_distance[j]),
                          'position_class':str(position_class[j]),'comparison':label,
                          **metrics(modes2[a][:,j],modes2[b][:,j])})
    pdf=pd.DataFrame(pair_rows)

    summary=[]
    for (mode,region),q in df.groupby(['mode','region']):
      rec={'mode':mode,'region':region,'n_lines':int(q.flat_line.nunique()),'n_bins':int(q.n_bins.iloc[0])}
      for col in ['normalized_corr','normalized_nmse','normalized_phase_rmse_rad','raw_corr','raw_nmse','raw_phase_rmse_rad',
                  'source_weighted_corr','source_weighted_nmse','source_weighted_phase_rmse_rad','train_nmse']:
        vals=q[col].astype(float).to_list(); rec[f'{col}_median']=float(np.nanmedian(vals)); rec[f'{col}_median_ci95']=bootstrap_ci(vals)
      summary.append(rec)
    pair_summary=[]
    for (label,pos),q in pdf.groupby(['comparison','position_class']):
      rec={'comparison':label,'position_class':pos,'n_lines':int(len(q))}
      for col in ['corr','nmse','phase_rmse_rad','amplitude_nrmse']:
        vals=q[col].astype(float).to_list(); rec[f'{col}_median']=float(np.nanmedian(vals)); rec[f'{col}_median_ci95']=bootstrap_ci(vals,seed=54321)
      pair_summary.append(rec)
    holdout_group_summary=[]
    q0=df[(df['mode']=='independent') & (df['region']=='heldout_all')]
    for group,q in q0.groupby('score_group'):
      rec={'score_group':int(group),'n_lines':int(len(q))}
      for col in ['normalized_corr','raw_corr','source_weighted_corr','normalized_phase_rmse_rad','raw_phase_rmse_rad']:
        vals=q[col].astype(float).to_list(); rec[f'{col}_median']=float(np.nanmedian(vals)); rec[f'{col}_median_ci95']=bootstrap_ci(vals,seed=67890)
      holdout_group_summary.append(rec)
    order_group_summary=[]
    for (label,group),q in pdf.groupby(['comparison','score_group']):
      rec={'comparison':label,'score_group':int(group),'n_lines':int(len(q))}
      for col in ['corr','nmse','phase_rmse_rad','amplitude_nrmse']:
        vals=q[col].astype(float).to_list(); rec[f'{col}_median']=float(np.nanmedian(vals)); rec[f'{col}_median_ci95']=bootstrap_ci(vals,seed=9876)
      order_group_summary.append(rec)

    return {'dataset':kind,'K':K,'n_strips':int(nstrips),'lines_per_strip':int(nstrip),'n_lines':int(full2.shape[1]),
            'input_window_zero_based':[int(ii[0]),int(ii[-1])], 'fixed_mapping':mapping,
            'source_heldout_counts':{name:int(mask.sum()) for name,mask in region_masks.items()},
            'heldout_summary':summary,'holdout_group_summary':holdout_group_summary,
            'order_pair_summary':pair_summary,'order_group_summary':order_group_summary,
            '_line_rows':rows,'_order_pair_rows':pair_rows,
            'boundary':{'input':'phase-corrected complex Cscan',
                        'mapping':'indexing derived from the deposited circshift/flip construction and only validated on independent-mode input bins',
                        'holdout':'72 deposited bins are lower-source-envelope measurements, not wider-band raw-camera truth',
                        'order':'finite selected strips test recursive warm-start sensitivity, not byte-identical full-B-scan history'}}


def main() -> None:
    out=Path(sys.argv[-1]); out.mkdir(parents=True,exist_ok=True)
    reports=[analyze(Path(p)) for p in sys.argv[1:-1]]
    compact_reports=[]
    for r in reports:
      pd.DataFrame(r.pop('_line_rows')).to_csv(out/f"{r['dataset']}_holdout_per_line.csv",index=False)
      pd.DataFrame(r.pop('_order_pair_rows')).to_csv(out/f"{r['dataset']}_order_per_line.csv",index=False)
      compact_reports.append(r)
    combined={'experiment':'exact_author_RFIAA_MIAA_order_and_holdout_audit_v2','datasets':compact_reports,
              'decision_rules':{
                'order_invariance':'A two-iteration recursive warm start should approximate independent ten-iteration FIAA; materially different forward/reverse spectra indicate traversal dependence.',
                'holdout':'Report normalized, raw-interference-domain, and source-weighted fidelity separately; do not treat low-source normalized bins as noiseless truth.',
                'fwhm':'Peak width is not an acceptance criterion for this audit.'}}
    (out/'metrics.json').write_text(json.dumps(combined,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(combined,indent=2),encoding='utf-8')
    print(json.dumps(combined,indent=2))

if __name__=='__main__': main()
