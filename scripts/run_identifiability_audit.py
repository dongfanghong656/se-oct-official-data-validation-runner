#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seoct_uncertainty import single_reflector_localization_crlb, two_reflector_separation_crlb

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'identifiability_audit'; OUT.mkdir(parents=True,exist_ok=True)

def conventional_psf_fwhm(k):
    z=np.linspace(-20,20,200001); field=np.exp(1j*2*k[:,None]*z[None,:]).sum(axis=0)
    intensity=np.abs(field)**2; intensity/=intensity.max(); center=int(np.argmax(intensity)); left=center; right=center
    while left>0 and intensity[left]>=.5:left-=1
    while right<intensity.size-1 and intensity[right]>=.5:right+=1
    def crossing(i0,i1):
        y0,y1=intensity[i0],intensity[i1]
        return float(z[i0]) if y1==y0 else float(z[i0]+(.5-y0)/(y1-y0)*(z[i1]-z[i0]))
    return crossing(right-1,right)-crossing(left,left+1)

def main():
    k=np.linspace(10,11,64); psf=conventional_psf_fwhm(k)
    single=[]
    for snr_db in (0,5,10,15,20,25,30,35,40,50):
        result=single_reflector_localization_crlb(k,1+.0j,.4,np.eye(k.size)*10**(-snr_db/10))
        single.append({'snr_db_per_sample':snr_db,'localization_standard_deviation':result.z_standard_deviation,'localization_std_over_conventional_psf_fwhm':result.z_standard_deviation/psf,'fisher_condition_number':result.condition_number})
    separations=np.linspace(.02,3*psf,120); two=[]
    for phase in (0,np.pi/2,np.pi):
        for ratio in (1,.5):
            for separation in separations:
                result=two_reflector_separation_crlb(k,1,ratio*np.exp(1j*phase),.4,float(separation),np.eye(k.size)*1e-3)
                two.append({'relative_phase_rad':float(phase),'amplitude_ratio':ratio,'separation':float(separation),'separation_over_conventional_psf_fwhm':float(separation/psf),'separation_standard_deviation':result.separation_standard_deviation,'separation_std_over_true_separation':float(result.separation_standard_deviation/max(separation,1e-30)),'fisher_condition_number':result.condition_number})
    report={'schema':'se-oct-identifiability-audit-v1','status':'VERIFIED_SYNTHETIC','conventional_psf_fwhm_normalized_units':psf,'single_reflector':single,'two_reflector':two,'decisions':{'single_reflector_localization_is_same_as_two_reflector_resolution':False,'single_point_fwhm_proves_universal_resolution':False,'crlb_is_physical_wideband_validation':False},'boundary':'Optimistic proper-complex Gaussian CRLB with correct target count and forward model; excludes model selection, multiple scattering, chromatic transfer and false peaks.'}
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    plt.figure(figsize=(8.4,5.2)); plt.semilogy([r['snr_db_per_sample'] for r in single],[r['localization_std_over_conventional_psf_fwhm'] for r in single],marker='o'); plt.xlabel('per-sample SNR (dB)'); plt.ylabel('localization std / DFT PSF FWHM'); plt.title('Single-target localization can be finer than the Fourier main lobe'); plt.tight_layout(); plt.savefig(OUT/'single_target_localization_crlb.png',dpi=180); plt.close()
    plt.figure(figsize=(8.4,5.2))
    for phase in (0,np.pi/2,np.pi):
        rows=[r for r in two if r['amplitude_ratio']==1 and r['relative_phase_rad']==float(phase)]
        plt.semilogy([r['separation_over_conventional_psf_fwhm'] for r in rows],[r['separation_std_over_true_separation'] for r in rows],label=f'phase={phase/np.pi:.1f}pi')
    plt.xlabel('true separation / conventional DFT PSF FWHM'); plt.ylabel('CRLB separation std / true separation'); plt.title('Two-target separation becomes ill-conditioned at small spacing'); plt.ylim(1e-3,1e4); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'two_target_separation_crlb.png',dpi=180); plt.close()
if __name__=='__main__':main()
