from __future__ import annotations

import numpy as np

from seoct_uncertainty.calibration import combine_additive_and_calibration_noise,evaluate_phase_fit,fit_phase_polynomial,multiplicative_phase_noise_covariance,phase_error_covariance


def test_linear_phase_fit_and_psd_covariance():
    k=np.linspace(10,11,101); center=k.mean(); scale=np.max(np.abs(k-center)); x=(k-center)/scale; phase=.3+1.7*x
    fit=fit_phase_polynomial(k,phase,degree=1,phase_noise_variance=1e-4,center=center,scale=scale)
    np.testing.assert_allclose(fit.coefficients,[.3,1.7],atol=1e-10)
    covariance=phase_error_covariance(fit,k); assert np.min(np.linalg.eigvalsh(covariance))>=-1e-12


def test_phase_uncertainty_grows_outside_calibration_interval():
    k=np.linspace(-1,1,101); fit=fit_phase_polynomial(k,.2+.7*k,degree=1,phase_noise_variance=.01,center=0,scale=1)
    _,_,inside=evaluate_phase_fit(fit,np.array([0.,1.])); _,_,outside=evaluate_phase_fit(fit,np.array([2.,4.]))
    assert outside[-1]>outside[0]>inside[0] and outside[-1]>inside[-1]


def test_multiplicative_phase_covariance_and_combination():
    signal=np.array([1+0j,2j]); phase_cov=np.array([[.1,.02],[.02,.2]])
    got=multiplicative_phase_noise_covariance(signal,phase_cov); expected=signal[:,None]*phase_cov*signal.conj()[None,:]
    np.testing.assert_allclose(got,expected)
    additive=np.eye(2)*.01; combined=combine_additive_and_calibration_noise(additive,signal,phase_cov)
    assert np.all(np.real(np.diag(combined))>=np.diag(additive))
