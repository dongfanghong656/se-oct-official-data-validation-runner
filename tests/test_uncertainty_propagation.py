from __future__ import annotations

import numpy as np

from seoct_uncertainty import conditional_miaa, propagate_conditional_monte_carlo, sample_conditional_spectra


def fourier(indices: np.ndarray, object_grid: int) -> np.ndarray:
    z=np.arange(object_grid); return np.exp(-1j*2*np.pi*indices[:,None]*z[None,:]/object_grid)


def base_result():
    n=16; fg=fourier(np.arange(-2,2),n); fm=fourier(np.arange(2,6),n)
    power=np.zeros(n); power[[4,9]]=[1.0,0.25]; y=fg@np.sqrt(power)
    return conditional_miaa(y,fg,fm,power,np.eye(4)*5e-3)


def test_complex_samples_match_moments() -> None:
    result=base_result(); samples=sample_conditional_spectra(result,sample_count=15000,seed=17)
    empirical_mean=samples.mean(axis=0); centered=samples-empirical_mean
    empirical_cov=(centered.conj().T@centered/samples.shape[0]).T
    np.testing.assert_allclose(empirical_mean,result.mean,rtol=.06,atol=.025)
    np.testing.assert_allclose(empirical_cov,result.covariance,rtol=.15,atol=.02)


def test_linear_propagation_matches_exact_variance() -> None:
    result=base_result(); transform=np.array([[1,.5j,-.2,.1],[.3j,1,.4,-.2j]],complex)
    propagated=propagate_conditional_monte_carlo(result,lambda x:transform@x,sample_count=18000,seed=21)
    exact_mean=transform@result.mean; exact_cov=transform@result.covariance@transform.conj().T
    np.testing.assert_allclose(propagated.mean,exact_mean,rtol=.06,atol=.025)
    np.testing.assert_allclose(propagated.variance,np.real(np.diag(exact_cov)),rtol=.12,atol=.025)
