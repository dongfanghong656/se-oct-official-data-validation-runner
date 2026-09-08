#!/usr/bin/env python3
"""Analyze exact Octave execution of the deposited FIAA/MIAA functions."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def intensity(x: np.ndarray) -> np.ndarray:
    y = np.abs(np.asarray(x).reshape(-1)) ** 2
    return y / max(float(np.max(y)), 1e-30)


def fwhm(y: np.ndarray, dx: float) -> float:
    v = intensity(y)
    p = int(np.argmax(v))
    half = 0.5
    left = p
    while left > 0 and v[left] >= half:
        left -= 1
    right = p
    while right < v.size - 1 and v[right] >= half:
        right += 1
    if left == p or right == p:
        return float('nan')
    def crossing(i0: int, i1: int) -> float:
        y0, y1 = v[i0], v[i1]
        if y1 == y0:
            return float(i0)
        return i0 + (half-y0)/(y1-y0)
    xl = crossing(left, left+1)
    xr = crossing(right-1, right)
    return float((xr-xl)*dx)


def cscale(ref: np.ndarray, pred: np.ndarray) -> complex:
    den = np.vdot(pred, pred)
    return np.vdot(pred, ref) / den if abs(den) > 0 else 0j


def corr(ref: np.ndarray, pred: np.ndarray) -> float:
    r = np.asarray(ref).reshape(-1)
    p = np.asarray(pred).reshape(-1)
    den = np.linalg.norm(r)*np.linalg.norm(p)
    return float(abs(np.vdot(r,p))/den) if den > 0 else float('nan')


def aligned_nmse(ref: np.ndarray, pred: np.ndarray) -> float:
    a = cscale(ref,pred)
    return float(np.linalg.norm(ref-a*pred)**2 / max(np.linalg.norm(ref)**2,1e-30))


def phase_rmse(ref: np.ndarray, pred: np.ndarray) -> float:
    a = cscale(ref,pred)
    d = np.angle(ref*np.conj(a*pred))
    return float(np.sqrt(np.mean(d*d)))


def choose_mapping(pred800: np.ndarray, full200: np.ndarray, ii0: np.ndarray) -> dict[str, object]:
    """Choose alignment only on the 128 given bins, then score the 72 held-out bins."""
    target = full200 - np.mean(full200[ii0])
    orientations = {
        'direct': pred800,
        'conjugate': np.conj(pred800),
        'reverse': pred800[::-1],
        'conjugate_reverse': np.conj(pred800[::-1]),
    }
    best = None
    for name, series in orientations.items():
        for start in range(series.size-target.size+1):
            block = series[start:start+target.size]
            train_ref = target[ii0]
            train_pred = block[ii0]
            a = cscale(train_ref,train_pred)
            loss = np.linalg.norm(train_ref-a*train_pred)**2/max(np.linalg.norm(train_ref)**2,1e-30)
            if best is None or loss < best[0]:
                best = (float(loss),name,start,a,block)
    assert best is not None
    loss,name,start,a,block = best
    held = np.ones(target.size,dtype=bool)
    held[ii0] = False
    # Avoid bins where the exact deposited source is near zero; those are noise-amplified by C/sk.
    ref = target[held]
    pr = (a*block)[held]
    return {
        'orientation': name,
        'start_in_800': int(start),
        'train_aligned_nmse': loss,
        'heldout_complex_correlation': corr(ref,pr),
        'heldout_aligned_nmse': aligned_nmse(ref,pr),
        'heldout_phase_rmse_rad': phase_rmse(ref,pr),
        'scale_real': float(np.real(a)),
        'scale_imag': float(np.imag(a)),
    }


def main() -> None:
    mat_path = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True,exist_ok=True)
    d = loadmat(mat_path,squeeze_me=True)
    K = int(d['K'])
    nlines = d['dft_profile'].shape[1]
    dx = 810.0/K
    ii0 = np.asarray(d['ii'],dtype=int).reshape(-1)-1
    methods = ['dft_profile','rfiaa_profile','rfiaa_map_profile','miaa_profile',
               'dispersed_miaa_profile','corrected_miaa_profile']
    lines=[]
    for j in range(nlines):
        item={'line':j,'coordinate_yx0':np.asarray(d['coords_yx0'])[j].astype(int).tolist(),
              'fwhm_um':{m:fwhm(d[m][:,j],dx) for m in methods},
              'mapping':choose_mapping(d['miaa_spectrum'][:,j],d['normalized_full'][:,j],ii0)}
        item['baseline_to_dispersed_complex_correlation']=corr(d['miaa_profile'][:,j],d['dispersed_miaa_profile'][:,j])
        item['baseline_to_explicitly_corrected_complex_correlation']=corr(d['miaa_profile'][:,j],d['corrected_miaa_profile'][:,j])
        lines.append(item)
    def med(path):
        vals=[]
        for x in lines:
            cur=x
            for k in path: cur=cur[k]
            if np.isfinite(cur): vals.append(float(cur))
        return float(np.median(vals)) if vals else float('nan')
    aggregate={
      'median_fwhm_um':{m:med(('fwhm_um',m)) for m in methods},
      'median_mapping_train_nmse':med(('mapping','train_aligned_nmse')),
      'median_heldout_complex_correlation':med(('mapping','heldout_complex_correlation')),
      'median_heldout_aligned_nmse':med(('mapping','heldout_aligned_nmse')),
      'median_heldout_phase_rmse_rad':med(('mapping','heldout_phase_rmse_rad')),
      'median_baseline_to_dispersed_complex_correlation':med(('baseline_to_dispersed_complex_correlation',)),
      'median_baseline_to_explicitly_corrected_complex_correlation':med(('baseline_to_explicitly_corrected_complex_correlation',)),
    }
    report={
      'experiment':'exact_deposited_FIAA_MAP_MIAA_selected_A_lines',
      'source_boundary':'Input is the authors phase-corrected Cscan, not raw camera interferograms.',
      'authors_code_execution':'fiaa_oct_c1.m verified by Zenodo MD5; preprocessing and MIAA mapping copied from MIAA_ISAM_processing.m.',
      'K':K,'axial_spacing_um':dx,'ii_zero_based':ii0.tolist(),'lines':lines,'aggregate':aggregate,
      'interpretation':{
        'dispersion':'MIAA is not dispersion correction if injected phase changes the output and explicit inverse phase restores it.',
        'image_optimization':'MIAA is not validated merely by FWHM; held-out measured complex spectral bins are the primary mechanism target.',
        'spectral_extrapolation_boundary':'Predictions outside all 200 deposited bins remain unverified without wider measured data.'
      }
    }
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    x=np.arange(K)*dx
    j=0
    plt.figure(figsize=(10,5.5))
    for m,label in [('dft_profile','DFT 200'),('rfiaa_profile','FIAA 128'),('rfiaa_map_profile','FIAA+MAP 128'),('miaa_profile','exact MIAA spectrum→IFFT')]:
        plt.plot(x,intensity(d[m][:,j]),label=label)
    p=int(np.argmax(intensity(d['dft_profile'][:,j])))
    plt.xlim(max(0,(p-35)*dx),min(K*dx,(p+35)*dx))
    plt.xlabel('axial coordinate (µm)'); plt.ylabel('normalized intensity')
    plt.title('Exact deposited FIAA/MIAA functions on official phase-corrected A-line')
    plt.legend(); plt.tight_layout(); plt.savefig(out/'exact_aline_profiles.png',dpi=180); plt.close()
    plt.figure(figsize=(10,5.5))
    for m,label in [('miaa_profile','baseline exact MIAA'),('dispersed_miaa_profile','injected quadratic phase + exact MIAA'),('corrected_miaa_profile','explicit inverse phase + exact MIAA')]:
        plt.plot(x,intensity(d[m][:,j]),label=label)
    plt.xlim(max(0,(p-50)*dx),min(K*dx,(p+80)*dx))
    plt.xlabel('axial coordinate (µm)'); plt.ylabel('normalized intensity')
    plt.title('Exact-code dispersion causal control')
    plt.legend(); plt.tight_layout(); plt.savefig(out/'exact_dispersion_control.png',dpi=180); plt.close()
    print(json.dumps(aggregate,indent=2))

if __name__=='__main__':
    main()
