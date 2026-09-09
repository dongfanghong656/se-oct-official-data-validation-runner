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
    effective_rank: int
    identifiable: bool


@dataclass(frozen=True)
class TwoReflectorSeparationCRLB:
    separation_variance: float
    separation_standard_deviation: float
    fisher_information: np.ndarray
    condition_number: float
    effective_rank: int
    identifiable: bool


def _parameter_covariance_or_infinite(
    fisher: np.ndarray,
    parameter_index: int,
    *,
    relative_rank_tolerance: float,
) -> tuple[float, float, int, bool]:
    fisher = 0.5 * (fisher + fisher.T)
    eigenvalues = np.linalg.eigvalsh(fisher)
    maximum = max(float(np.max(np.abs(eigenvalues))), 1e-30)
    tolerance = relative_rank_tolerance * maximum
    rank = int(np.sum(eigenvalues > tolerance))
    condition = float(np.linalg.cond(fisher))
    identifiable = rank == fisher.shape[0] and np.isfinite(condition)
    if not identifiable:
        return float("inf"), condition, rank, False
    covariance = np.linalg.inv(fisher)
    variance = float(max(covariance[parameter_index, parameter_index], 0.0))
    return variance, condition, rank, True


def single_reflector_localization_crlb(
    wavenumbers: npt.ArrayLike,
    complex_amplitude: complex,
    axial_position: float,
    noise_covariance: npt.ArrayLike,
    *,
    relative_rank_tolerance: float = 1e-10,
) -> SingleReflectorCRLB:
    """CRLB for one reflector with unknown complex amplitude and position.

    A finite bound is returned only when the full nuisance-parameter Fisher
    matrix is identifiable. This is a localization bound, not a two-target
    resolution criterion.
    """

    k = np.asarray(wavenumbers, dtype=np.float64).reshape(-1)
    covariance = np.asarray(noise_covariance, dtype=np.complex128)
    if covariance.shape != (k.size, k.size):
        raise ValueError("noise_covariance has wrong shape")
    if relative_rank_tolerance <= 0:
        raise ValueError("relative_rank_tolerance must be positive")
    if np.any(~np.isfinite(k)) or np.any(~np.isfinite(covariance)):
        raise ValueError("inputs contain non-finite values")

    h = np.exp(-1j * 2.0 * k * axial_position)
    derivatives = np.column_stack(
        [
            h,
            1j * h,
            (-1j * 2.0 * k) * complex_amplitude * h,
        ]
    )
    solved = np.linalg.solve(covariance, derivatives)
    fisher = 2.0 * np.real(derivatives.conj().T @ solved)
    variance, condition, rank, identifiable = _parameter_covariance_or_infinite(
        fisher,
        2,
        relative_rank_tolerance=relative_rank_tolerance,
    )
    return SingleReflectorCRLB(
        z_variance=variance,
        z_standard_deviation=float(np.sqrt(variance)),
        fisher_information=fisher,
        condition_number=condition,
        effective_rank=rank,
        identifiable=identifiable,
    )


def two_reflector_separation_crlb(
    wavenumbers: npt.ArrayLike,
    complex_amplitude_1: complex,
    complex_amplitude_2: complex,
    center_position: float,
    separation: float,
    noise_covariance: npt.ArrayLike,
    *,
    relative_rank_tolerance: float = 1e-10,
) -> TwoReflectorSeparationCRLB:
    """Optimistic CRLB for two reflectors with unknown amplitudes and center.

    Parameters are ``[Re(a1), Im(a1), Re(a2), Im(a2), center, separation]``.
    At zero or near-zero separation the nuisance-parameter Fisher matrix loses
    rank or becomes severely ill-conditioned. A finite single-target bound
    therefore does not imply the same two-target resolution.
    """

    k = np.asarray(wavenumbers, dtype=np.float64).reshape(-1)
    covariance = np.asarray(noise_covariance, dtype=np.complex128)
    if covariance.shape != (k.size, k.size):
        raise ValueError("noise_covariance has wrong shape")
    if separation < 0:
        raise ValueError("separation must be non-negative")
    if relative_rank_tolerance <= 0:
        raise ValueError("relative_rank_tolerance must be positive")
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
    derivatives = np.column_stack(
        [
            h1,
            1j * h1,
            h2,
            1j * h2,
            derivative_center,
            derivative_separation,
        ]
    )
    solved = np.linalg.solve(covariance, derivatives)
    fisher = 2.0 * np.real(derivatives.conj().T @ solved)
    variance, condition, rank, identifiable = _parameter_covariance_or_infinite(
        fisher,
        5,
        relative_rank_tolerance=relative_rank_tolerance,
    )
    return TwoReflectorSeparationCRLB(
        separation_variance=variance,
        separation_standard_deviation=float(np.sqrt(variance)),
        fisher_information=fisher,
        condition_number=condition,
        effective_rank=rank,
        identifiable=identifiable,
    )
