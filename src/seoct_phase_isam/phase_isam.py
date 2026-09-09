from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

RealArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


def ewald_kz(k: npt.ArrayLike, qx: npt.ArrayLike, qy: npt.ArrayLike | float = 0.0) -> RealArray:
    """Backscatter Ewald-shell axial frequency sqrt(4 k^2-qx^2-qy^2)."""
    kk, xx, yy = np.broadcast_arrays(
        np.asarray(k, dtype=float),
        np.asarray(qx, dtype=float),
        np.asarray(qy, dtype=float),
    )
    radicand = 4.0 * kk**2 - xx**2 - yy**2
    if np.any(radicand < -1e-12 * max(float(np.max(4.0 * kk**2)), 1.0)):
        raise ValueError("requested transverse frequency lies outside the propagating Ewald shell")
    return np.sqrt(np.maximum(radicand, 0.0))


def paraxial_ewald_kz(k: npt.ArrayLike, q: npt.ArrayLike) -> RealArray:
    """Paraxial expansion 2k-q^2/(4k)."""
    kk, qq = np.broadcast_arrays(np.asarray(k, float), np.asarray(q, float))
    if np.any(kk <= 0):
        raise ValueError("k must be positive")
    return 2.0 * kk - qq**2 / (4.0 * kk)


def gaussian_phase_coherence(phase_rms_rad: npt.ArrayLike) -> tuple[RealArray, RealArray]:
    """Expected coherent amplitude and intensity factors for zero-mean Gaussian phase error."""
    sigma = np.asarray(phase_rms_rad, dtype=float)
    if np.any(sigma < 0) or np.any(~np.isfinite(sigma)):
        raise ValueError("phase_rms_rad must be finite and non-negative")
    amplitude = np.exp(-0.5 * sigma**2)
    return amplitude, amplitude**2


def phase_rms_from_one_way_opl(opl_rms_um: npt.ArrayLike, wavelength_um: float) -> RealArray:
    """OCT phase RMS for one-way optical-path-coordinate jitter in exp(-i 2 k z)."""
    if wavelength_um <= 0:
        raise ValueError("wavelength_um must be positive")
    opl = np.asarray(opl_rms_um, dtype=float)
    if np.any(opl < 0) or np.any(~np.isfinite(opl)):
        raise ValueError("opl_rms_um must be finite and non-negative")
    return 4.0 * np.pi * opl / wavelength_um


def opl_rms_for_intensity_coherence(target_intensity: float, wavelength_um: float) -> float:
    """Maximum one-way OPL RMS for expected coherent intensity above target."""
    if not 0.0 < target_intensity <= 1.0:
        raise ValueError("target_intensity must lie in (0,1]")
    return float(np.sqrt(-np.log(target_intensity)) * wavelength_um / (4.0 * np.pi))


def miaa_linear_mmse(
    y_given: npt.ArrayLike,
    fourier_given: npt.ArrayLike,
    fourier_missing: npt.ArrayLike,
    power: npt.ArrayLike,
    noise_covariance: npt.ArrayLike,
) -> ComplexArray:
    """MIAA conditional mean R_mg R_gg^-1 y_g for a fixed power model."""
    y = np.asarray(y_given, dtype=np.complex128).reshape(-1)
    fg = np.asarray(fourier_given, dtype=np.complex128)
    fm = np.asarray(fourier_missing, dtype=np.complex128)
    p = np.asarray(power, dtype=float).reshape(-1)
    sigma = np.asarray(noise_covariance, dtype=np.complex128)
    if fg.shape[0] != y.size or fg.shape[1] != fm.shape[1] or p.size != fg.shape[1]:
        raise ValueError("incompatible MIAA dimensions")
    if sigma.shape != (y.size, y.size):
        raise ValueError("noise_covariance shape mismatch")
    if np.any(p < 0) or np.any(~np.isfinite(p)):
        raise ValueError("power must be finite and non-negative")
    rgg = (fg * p[None, :]) @ fg.conj().T + sigma
    rmg = (fm * p[None, :]) @ fg.conj().T
    return rmg @ np.linalg.solve(rgg, y)


def effective_coherent_sample_count(weights: npt.ArrayLike) -> float:
    """Kish effective count for non-negative coherent aperture weights."""
    w = np.abs(np.asarray(weights, dtype=np.complex128).reshape(-1))
    if w.size == 0 or np.sum(w) <= 0:
        raise ValueError("weights must be non-empty and nonzero")
    return float(np.sum(w) ** 2 / np.sum(w**2))


def expected_finite_aperture_peak_intensity(
    phase_rms_rad: npt.ArrayLike,
    weights: npt.ArrayLike,
) -> RealArray:
    """Expected peak intensity including the finite-aperture incoherent floor."""
    _, coherent = gaussian_phase_coherence(phase_rms_rad)
    neff = effective_coherent_sample_count(weights)
    return coherent + (1.0 - coherent) / neff


def lateral_phase_mix(signal_xk: npt.ArrayLike, phase_xk: npt.ArrayLike) -> ComplexArray:
    """Apply a lateral phase screen then transform x to q."""
    signal = np.asarray(signal_xk, dtype=np.complex128)
    phase = np.asarray(phase_xk, dtype=float)
    if signal.shape != phase.shape:
        raise ValueError("signal_xk and phase_xk must have the same shape")
    corrupted = signal * np.exp(1j * phase)
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(corrupted, axes=0), axis=0), axes=0)


@dataclass(frozen=True)
class SupportGeometry:
    k0_um_inv: float
    measured_k_width_um_inv: float
    predicted_k_width_um_inv: float
    q_max_um_inv: float
    point_object_ewald_downshift_um_inv: float
    measured_on_axis_kz_width_um_inv: float
    predicted_on_axis_kz_width_um_inv: float
    measured_point_union_width_um_inv: float
    predicted_point_union_width_um_inv: float


def support_geometry(
    *,
    k0_um_inv: float,
    measured_k_width_um_inv: float,
    predicted_k_width_um_inv: float,
    q_max_um_inv: float,
) -> SupportGeometry:
    """First-order support accounting for measured and predicted radial bandwidth."""
    if min(k0_um_inv, measured_k_width_um_inv, predicted_k_width_um_inv) <= 0:
        raise ValueError("k0 and bandwidths must be positive")
    if q_max_um_inv < 0 or q_max_um_inv >= 2 * k0_um_inv:
        raise ValueError("q_max must lie on the propagating shell")
    downshift = float(2 * k0_um_inv - ewald_kz(k0_um_inv, q_max_um_inv))
    measured_kz = 2.0 * measured_k_width_um_inv
    predicted_kz = 2.0 * predicted_k_width_um_inv
    return SupportGeometry(
        k0_um_inv=float(k0_um_inv),
        measured_k_width_um_inv=float(measured_k_width_um_inv),
        predicted_k_width_um_inv=float(predicted_k_width_um_inv),
        q_max_um_inv=float(q_max_um_inv),
        point_object_ewald_downshift_um_inv=downshift,
        measured_on_axis_kz_width_um_inv=measured_kz,
        predicted_on_axis_kz_width_um_inv=predicted_kz,
        measured_point_union_width_um_inv=measured_kz + downshift,
        predicted_point_union_width_um_inv=predicted_kz + downshift,
    )


def iaa_reflectivity_power(
    y_given: npt.ArrayLike,
    fourier_given: npt.ArrayLike,
    *,
    noise_variance: float | None = None,
    iterations: int = 10,
    power_floor: float = 1e-20,
) -> tuple[ComplexArray, RealArray, float]:
    """Transparent direct IAA for mechanism audits, not the fast Toeplitz implementation."""
    y = np.asarray(y_given, dtype=np.complex128).reshape(-1)
    fg = np.asarray(fourier_given, dtype=np.complex128)
    if fg.ndim != 2 or fg.shape[0] != y.size:
        raise ValueError("fourier_given row count must match y_given")
    if iterations < 1 or power_floor <= 0:
        raise ValueError("iterations and power_floor must be positive")
    signal_power = max(float(np.mean(np.abs(y) ** 2)), 1e-30)
    eta = float(noise_variance) if noise_variance is not None else 1e-8 * signal_power
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("noise_variance must be finite and non-negative")
    eta = max(eta, 1e-15 * signal_power)
    norm2 = np.sum(np.abs(fg) ** 2, axis=0)
    reflectivity = fg.conj().T @ y / np.maximum(norm2, 1e-30)
    power = np.maximum(np.abs(reflectivity) ** 2, power_floor)
    identity = np.eye(y.size, dtype=np.complex128)
    for _ in range(iterations):
        covariance = (fg * power[None, :]) @ fg.conj().T + eta * identity
        solved_y = np.linalg.solve(covariance, y)
        solved_dictionary = np.linalg.solve(covariance, fg)
        numerator = fg.conj().T @ solved_y
        denominator = np.sum(fg.conj() * solved_dictionary, axis=0)
        reflectivity = numerator / np.where(np.abs(denominator) > 1e-30, denominator, 1e-30)
        power = np.maximum(np.abs(reflectivity) ** 2, power_floor)
    return np.asarray(reflectivity), np.asarray(power), eta


def iaa_miaa_predict(
    y_given: npt.ArrayLike,
    fourier_given: npt.ArrayLike,
    fourier_target: npt.ArrayLike,
    *,
    noise_variance: float | None = None,
    iterations: int = 10,
) -> tuple[ComplexArray, ComplexArray, RealArray, float]:
    """Estimate IAA power then compute the MIAA conditional target spectrum."""
    reflectivity, power, eta = iaa_reflectivity_power(
        y_given, fourier_given, noise_variance=noise_variance, iterations=iterations
    )
    y_target = miaa_linear_mmse(
        y_given,
        fourier_given,
        fourier_target,
        power,
        np.eye(np.asarray(y_given).size, dtype=np.complex128) * eta,
    )
    return y_target, reflectivity, power, eta
