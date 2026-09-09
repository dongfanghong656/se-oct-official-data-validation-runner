#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from seoct_uncertainty import combine_additive_and_calibration_noise,conditional_miaa,evaluate_phase_fit,fit_phase_polynomial
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports'/'calibration_uncertainty_audit'; OUT.mkdir(parents=True,exist_ok=True)

def fourier(indices,n):
    z=np.arange(n); return np.exp(-1j*2*np.pi*indices[:,None]*z[None,:]/n)

def main():
    rng=np.random.default_rng(20260909); k_cal=np.linspace(-1,1,121); k_eval=np.linspace(-4,4,401)
    ref_std=.02; ref_obs=.2+1.4*k_cal+rng.normal(scale=ref_std,size=k_cal.size)
    ref_fit=fit_phase_polynomial(k_cal,ref_obs,degree=1,phase_noise_variance=ref_std**2,center=0,scale=1); _,_,ref_phase_std=evaluate_phase_fit(ref_fit,k_eval)
    disp_truth=.1+.25*k_cal**2-.08*k_cal**3+.03*k_cal**4; disp_std=.01; disp_obs=disp_truth+rng.normal(scale=disp_std,size=k_cal.size)
    disp_fit=fit_phase_polynomial(k_cal,disp_obs,degree=8,phase_noise_variance=disp_std**2,center=0,scale=1); _,_,disp_phase_std=evaluate_phase_fit(disp_fit,k_eval)
    table=[]
    for x in (0,.5,1,1.5,2,3,4):
        i=int(np.argmin(np.abs(k_eval-x))); table.append({'normalized_distance':x,'linear_reference_phase_std_rad':float(ref_phase_std[i]),'degree8_dispersion_phase_std_rad':float(disp_phase_std[i])})
    n=64; given=np.arange(-8,8); missing=np.r_[np.arange(-32,-8),np.arange(8,32)]; fg=fourier(given,n); fm=fourier(missing,n)
    reflectivity=np.zeros(n,complex); reflectivity[[18,39]]=[1,.45*np.exp(.7j)]; power=np.abs(reflectivity)**2; y=fg@reflectivity; additive=np.eye(given.size)*2e-3
    xg=given/max(abs(given)); design=np.column_stack([np.ones(given.size),xg]); phase_cov=design@ref_fit.coefficient_covariance@design.T
    augmented=combine_additive_and_calibration_noise(additive,y,phase_cov)
    baseline=conditional_miaa(y,fg,fm,power,additive); calibrated=conditional_miaa(y,fg,fm,power,augmented)
    report={'schema':'phase-calibration-uncertainty-audit-v1','status':'VERIFIED_SYNTHETIC','calibrated_interval':[-1,1],'reference_linear_fit':{'degree':1,'phase_noise_std_rad':ref_std,'coefficient_condition_number':ref_fit.condition_number},'dispersion_fit':{'degree':8,'phase_noise_std_rad':disp_std,'coefficient_condition_number':disp_fit.condition_number},'phase_uncertainty_by_distance':table,'miaa_confidence_effect':{'without_calibration_uncertainty_median_predictability':float(np.median(baseline.predictability)),'with_reference_fit_uncertainty_median_predictability':float(np.median(calibrated.predictability)),'without_calibration_uncertainty_fraction_q_ge_0_5':float(np.mean(baseline.predictability>=.5)),'with_reference_fit_uncertainty_fraction_q_ge_0_5':float(np.mean(calibrated.predictability>=.5))},'decisions':{'phase_calibration_uncertainty_should_be_propagated':True,'high_order_polynomial_should_be_extrapolated_beyond_calibration_support':False,'phase_correction_point_estimate_is_exact':False},'boundary':'Synthetic statistical fit uncertainty only. Authors did not publish raw mirror/coverslip phase residuals; actual error is not estimated.'}
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    plt.figure(figsize=(8.6,5.3)); plt.semilogy(k_eval,np.maximum(ref_phase_std,1e-8),label='linear reference phase'); plt.semilogy(k_eval,np.maximum(disp_phase_std,1e-8),label='degree-8 dispersion fit'); plt.axvspan(-1,1,alpha=.15,label='calibration support'); plt.xlabel('normalized spectral coordinate'); plt.ylabel('phase standard deviation (rad)'); plt.title('Phase-model uncertainty grows outside calibrated support'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'phase_uncertainty_extrapolation.png',dpi=180); plt.close()
if __name__=='__main__':main()
