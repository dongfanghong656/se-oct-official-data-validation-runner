from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

ComplexArray = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PhasePolynomialFit:
    coefficients: RealArray
    coefficient_covariance: RealArray
    center: float
    scale: float
    degree: int
    residual_variance: float
    condition_number: float


def polynomial_phase_design(wavenumbers: npt.ArrayLike, *, degree: int, center: float | None = None, scale: float | None = None) -> tuple[RealArray,float,float]:
    k=np.asarray(wavenumbers,dtype=np.float64).reshape(-1)
    if degree<0: raise ValueError('degree must be non-negative')
    if np.any(~np.isfinite(k)): raise ValueError('wavenumbers contain non-finite values')
    used_center=float(np.mean(k)) if center is None else float(center)
    used_scale=max(float(np.max(np.abs(k-used_center))),1e-30) if scale is None else float(scale)
    if not np.isfinite(used_scale) or used_scale<=0: raise ValueError('scale must be finite and positive')
    x=(k-used_center)/used_scale
    return np.asarray(np.column_stack([x**order for order in range(degree+1)]),dtype=np.float64),used_center,used_scale


def fit_phase_polynomial(wavenumbers:npt.ArrayLike,unwrapped_phase:npt.ArrayLike,*,degree:int,phase_noise_variance:npt.ArrayLike|None=None,center:float|None=None,scale:float|None=None)->PhasePolynomialFit:
    phase=np.asarray(unwrapped_phase,dtype=np.float64).reshape(-1)
    design,used_center,used_scale=polynomial_phase_design(wavenumbers,degree=degree,center=center,scale=scale)
    if phase.size!=design.shape[0]: raise ValueError('phase and wavenumber lengths differ')
    if np.any(~np.isfinite(phase)): raise ValueError('phase contains non-finite values')
    if phase_noise_variance is None:
        variance=np.ones(phase.size); variance_known=False
    else:
        variance=np.asarray(phase_noise_variance,dtype=np.float64)
        if variance.ndim==0: variance=np.full(phase.size,float(variance))
        variance=variance.reshape(-1); variance_known=True
        if variance.size!=phase.size or np.any(~np.isfinite(variance)) or np.any(variance<=0): raise ValueError('invalid phase_noise_variance')
    weight=1/variance; normal=design.T@(weight[:,None]*design); rhs=design.T@(weight*phase)
    coefficients=np.linalg.solve(normal,rhs); residual=phase-design@coefficients; dof=max(phase.size-(degree+1),1)
    residual_variance=float(np.sum(weight*residual**2)/dof); covariance=np.linalg.inv(normal)
    if not variance_known: covariance*=residual_variance
    return PhasePolynomialFit(np.asarray(coefficients),np.asarray(covariance),used_center,used_scale,degree,residual_variance,float(np.linalg.cond(normal)))


def evaluate_phase_fit(fit:PhasePolynomialFit,wavenumbers:npt.ArrayLike)->tuple[RealArray,RealArray,RealArray]:
    design,_,_=polynomial_phase_design(wavenumbers,degree=fit.degree,center=fit.center,scale=fit.scale)
    phase=design@fit.coefficients; covariance=design@fit.coefficient_covariance@design.T; covariance=.5*(covariance+covariance.T)
    variance=np.maximum(np.real(np.diag(covariance)),0)
    return np.asarray(phase),np.asarray(variance),np.sqrt(variance)


def phase_error_covariance(fit:PhasePolynomialFit,wavenumbers:npt.ArrayLike)->RealArray:
    design,_,_=polynomial_phase_design(wavenumbers,degree=fit.degree,center=fit.center,scale=fit.scale)
    covariance=design@fit.coefficient_covariance@design.T
    return np.asarray(.5*(covariance+covariance.T),dtype=np.float64)


def multiplicative_phase_noise_covariance(complex_signal:npt.ArrayLike,phase_covariance:npt.ArrayLike)->ComplexArray:
    signal=np.asarray(complex_signal,dtype=np.complex128).reshape(-1); covariance=np.asarray(phase_covariance,dtype=np.float64)
    if covariance.shape!=(signal.size,signal.size): raise ValueError('phase_covariance has wrong shape')
    if np.any(~np.isfinite(covariance)) or np.any(~np.isfinite(signal)): raise ValueError('inputs contain non-finite values')
    covariance=.5*(covariance+covariance.T)
    if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-9*max(float(np.linalg.norm(covariance)),1.0): raise ValueError('phase_covariance is not positive semidefinite')
    result=signal[:,None]*covariance*signal.conj()[None,:]
    return np.asarray(.5*(result+result.conj().T),dtype=np.complex128)


def combine_additive_and_calibration_noise(additive_covariance:npt.ArrayLike,complex_signal:npt.ArrayLike,phase_covariance:npt.ArrayLike)->ComplexArray:
    additive=np.asarray(additive_covariance,dtype=np.complex128); phase_noise=multiplicative_phase_noise_covariance(complex_signal,phase_covariance)
    if additive.shape!=phase_noise.shape: raise ValueError('additive_covariance has wrong shape')
    if np.any(~np.isfinite(additive)): raise ValueError('additive_covariance contains non-finite values')
    return np.asarray(.5*(additive+additive.conj().T)+phase_noise,dtype=np.complex128)
