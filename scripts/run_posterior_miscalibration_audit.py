#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from seoct_uncertainty import conditional_miaa
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports'/'posterior_miscalibration_audit';OUT.mkdir(parents=True,exist_ok=True)

def fourier(indices,n):
    z=np.arange(n);return np.exp(-1j*2*np.pi*indices[:,None]*z[None,:]/n)

def main():
    rng=np.random.default_rng(20260909);n=128;given=np.arange(-12,12);all_indices=np.arange(-48,48);missing_mask=~np.isin(all_indices,given);missing=all_indices[missing_mask]
    fg=fourier(given,n);fm=fourier(missing,n);r=np.zeros(n,complex);positions=np.array([36,57,89]);amps=np.array([1,.42*np.exp(.8j),.22*np.exp(-1.2j)]);r[positions]=amps;true_power=np.abs(r)**2
    missing_weak=true_power.copy();missing_weak[positions[-1]]=0
    shifted=np.zeros_like(true_power);shifted_positions=(positions+np.array([0,1,-1]))%n;shifted[shifted_positions]=np.abs(amps)**2
    scenarios={'correct_model':{'power':true_power,'transfer':np.ones(missing.size,complex)},'missing_weak_reflector':{'power':missing_weak,'transfer':np.ones(missing.size,complex)},'one_bin_grid_mismatch':{'power':shifted,'transfer':np.ones(missing.size,complex)},'unmodelled_out_of_band_phase':{'power':true_power,'transfer':np.exp(1j*2.5*(missing/np.max(np.abs(all_indices)))**2)}}
    yg=fg@r;base_missing=fm@r;noise_variance=2e-3;sigma=np.eye(given.size)*noise_variance;trials=500;coverage_threshold=-np.log(.05);rows=[];curves={}
    for name,scenario in scenarios.items():
        truth=base_missing*scenario['transfer'];errors=np.zeros((trials,missing.size));reported=None
        for trial in range(trials):
            noise=np.sqrt(noise_variance/2)*(rng.standard_normal(given.size)+1j*rng.standard_normal(given.size));result=conditional_miaa(yg+noise,fg,fm,scenario['power'],sigma);errors[trial]=np.abs(result.mean-truth)**2
            if reported is None:reported=result.posterior_variance.copy()
        mse=np.mean(errors,axis=0);ratio=mse/np.maximum(reported,1e-30);normalized=errors/np.maximum(reported[None,:],1e-30);coverage=np.mean(normalized<=coverage_threshold,axis=0)
        rows.append({'scenario':name,'median_empirical_mse_over_reported_variance':float(np.median(ratio)),'q10_empirical_mse_over_reported_variance':float(np.quantile(ratio,.1)),'q90_empirical_mse_over_reported_variance':float(np.quantile(ratio,.9)),'median_nominal_95pct_coverage':float(np.median(coverage)),'minimum_nominal_95pct_coverage':float(np.min(coverage)),'fraction_bins_with_coverage_below_0_8':float(np.mean(coverage<.8))})
        curves[name]={'missing_indices':missing.tolist(),'mse_variance_ratio':ratio.tolist(),'coverage':coverage.tolist()}
    report={'schema':'conditional-miaa-posterior-miscalibration-v1','status':'VERIFIED_SYNTHETIC','trials':trials,'noise_variance':noise_variance,'rows':rows,'curves':curves,'decisions':{'plug_in_conditional_covariance_includes_model_uncertainty':False,'correct_model_calibration_implies_model_mismatch_calibration':False,'physical_validation_can_be_replaced_by_posterior_variance':False},'boundary':'Synthetic plug-in empirical-Bayes audit. Misspecification scenarios are examples, not estimates of the authors actual error.'}
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    plt.figure(figsize=(8.7,5.3))
    for row in rows:
        curve=curves[row['scenario']];plt.plot(curve['missing_indices'],np.minimum(curve['mse_variance_ratio'],100),label=row['scenario'])
    plt.axhline(1);plt.yscale('log');plt.xlabel('missing spectral-bin index');plt.ylabel('empirical MSE / reported conditional variance');plt.title('Plug-in posterior can be overconfident under model mismatch');plt.legend();plt.tight_layout();plt.savefig(OUT/'model_mismatch_calibration_ratio.png',dpi=180);plt.close()
if __name__=='__main__':main()
