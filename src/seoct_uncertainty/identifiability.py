from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SingleReflectorCRLB:
    z_variance: float
    z_standard_deviation: float
    fisher_information: np.ndarray
    condition_number: float


@dataclass(frozen=True)
class TwoReflectorSeparationCRLB:
    separation_variance: float
    separation_standard_deviation: float
    fisher_information: np.ndarray
    condition_number: float


def single_reflector_localization_crlb(
    wavenumbers: npt.ArrayLike,
    complex_amplitude: complex,
    axial_position: float,
    noise_covariance: npt.ArrayLike,
) -> SingleReflectorCRLB:
    """CRLB for one reflector with unknown complex amplitude and position."""

    k = np.asarray(wavenumbers, dtype=np.float64).reshape(-1)
    covariance = np.asarray(noise_covariance, dtype=np.complex128)
    if covariance.shape != (k.size, k.size):
        raise ValueError("noise_covariance has wrong shape")
    if np.any(~np.isfinite(k)) or np.any(~np.isfinite(covariance)):
        raise ValueError("inputs contain non-finite values")
    h = np.exp(-1j * 2.0 * k * axial_position)
    derivatives = np.column_stack([
        h,
        1j * h,
        (-1j * 2.0 * k) * complex_amplitude * h,
    ])
    solved = np.linalg.solve(covariance, derivatives)
    fisher = 2.0 * np.real(derivatives.conj().T @ solved)
    condition = float(np.linalg.cond(fisher))
    parameter_covariance = np.linalg.pinv(fisher, rcond=1e-12)
    variance = float(max(parameter_covariance[2, 2], 0.0))
    return SingleReflectorCRLB(
        z_variance=variance,
        z_standard_deviation=float(np.sqrt(variance)),
        fisher_information=fisher,
        condition_number=condition,
    )


def two_reflector_separation_crlb(
    wavenumbers: npt.ArrayLike,
    complex_amplitude_1: complex,
    complex_amplitude_2: complex,
    center_position: float,
    separation: float,
    noise_covariance: npt.ArrayLike,
) -> TwoReflectorSeparationCRLB:
    """Optimistic CRLB for two reflectors with unknown amplitudes and center."""

    k = np.asarray(wavenumbers, dtype=np.float64).reshape(-1)
    covariance = np.asarray(noise_covariance, dtype=np.complex128)
    if covariance.shape != (k.size, k.size):
        raise ValueError("noise_covariance has wrong shape")
    if separation < 0:
        raise ValueError("separation must be non-negative")
    if np.any(~np.isfinite(k)) or np.any(~np.isfinite(covariance)):
        raise ValueError("inputs contain non-finite values")
    z1 = center_position - separation / 2.0
    z2 = center_position + separation / 2.0
    h1 = np.exp(-1j * 2.0 * k * z1)
    h2 = np.exp(-1j * 2.0 * k * z2)
    derivative_center = (-1j * 2.0 * k) * (
        complex_amplitude_1 * h1 + complex_amplitude_2 * h2
    )
    derivative_separation = (
        1j * k * complex_amplitude_1 * h1
        - 1j * k * complex_amplitude_2 * h2
    )
    derivatives = np.column_stack([
        h1,
        1j * h1,
        h2,
        1j * h2,
        derivative_center,
        derivative_separation,
    ])
    solved = np.linalg.solve(covariance, derivatives)
    fisher = 2.0 * np.real(derivatives.conj().T @ solved)
    condition = float(np.linalg.cond(fisher))
    parameter_covariance = np.linalg.pinv(fisher, rcond=1e-12)
    variance = float(max(parameter_covariance[5, 5], 0.0))
    return TwoReflectorSeparationCRLB(
        separation_variance=variance,
        separation_standard_deviation=float(np.sqrt(variance)),
        fisher_information=fisher,
        condition_number=condition,
    )
