from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

ComplexArray = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ConditionalMIAAResult:
    """Conditional mean and uncertainty of MIAA missing-data prediction.

    This is conditional on the current IAA power/noise model. It is not a
    guarantee that the unmeasured physical spectrum is correct.
    """

    mean: ComplexArray
    covariance: ComplexArray
    posterior_variance: RealArray
    prior_variance: RealArray
    predictability: RealArray
    predicted_snr: RealArray
    condition_number: float
    jitter_used: float


def _vector(value: npt.ArrayLike, name: str) -> ComplexArray:
    out = np.asarray(value, dtype=np.complex128).reshape(-1)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    return out


def _matrix(value: npt.ArrayLike, name: str) -> ComplexArray:
    out = np.asarray(value, dtype=np.complex128)
    if out.ndim != 2:
        raise ValueError(f"{name} must be 2-D; got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    return out


def _hermitian(value: ComplexArray) -> ComplexArray:
    return 0.5 * (value + value.conj().T)


def _covariance(
    value: npt.ArrayLike,
    size: int,
    name: str,
    *,
    tolerance: float,
) -> ComplexArray:
    cov = _matrix(value, name)
    if cov.shape != (size, size):
        raise ValueError(f"{name} must have shape {(size, size)}; got {cov.shape}")
    scale = max(float(np.linalg.norm(cov)), 1.0)
    if float(np.linalg.norm(cov - cov.conj().T)) > tolerance * scale:
        raise ValueError(f"{name} is not Hermitian")
    cov = _hermitian(cov)
    if float(np.min(np.linalg.eigvalsh(cov))) < -tolerance * scale:
        raise ValueError(f"{name} is not positive semidefinite")
    return cov


def normalized_noise_covariance(
    source_spectrum: npt.ArrayLike,
    *,
    raw_noise_variance: Optional[npt.ArrayLike] = None,
    raw_noise_covariance: Optional[npt.ArrayLike] = None,
    relative_floor: float = 1e-3,
) -> ComplexArray:
    """Propagate interference-domain noise through division by S(k).

    If ``y_norm = y_raw / S``, then
    ``Sigma_norm = D(1/S) Sigma_raw D(1/S)^H``.
    """

    source = _vector(source_spectrum, "source_spectrum")
    if relative_floor <= 0:
        raise ValueError("relative_floor must be positive")
    if raw_noise_variance is not None and raw_noise_covariance is not None:
        raise ValueError("provide variance or covariance, not both")

    floor = relative_floor * max(float(np.max(np.abs(source))), 1e-30)
    safe = source.copy()
    weak = np.abs(safe) < floor
    if np.any(weak):
        phase = np.exp(1j * np.angle(safe[weak]))
        phase[np.abs(phase) == 0] = 1.0
        safe[weak] = floor * phase
    inv = 1.0 / safe

    if raw_noise_covariance is not None:
        raw_cov = _covariance(
            raw_noise_covariance,
            source.size,
            "raw_noise_covariance",
            tolerance=1e-10,
        )
    else:
        if raw_noise_variance is None:
            variance = np.ones(source.size, dtype=np.float64)
        else:
            variance = np.asarray(raw_noise_variance, dtype=np.float64)
            if variance.ndim == 0:
                variance = np.full(source.size, float(variance))
            variance = variance.reshape(-1)
            if variance.size != source.size:
                raise ValueError("raw_noise_variance has the wrong length")
            if np.any(~np.isfinite(variance)) or np.any(variance < 0):
                raise ValueError("raw_noise_variance must be finite and non-negative")
        raw_cov = np.diag(variance.astype(np.complex128))

    return _hermitian(inv[:, None] * raw_cov * inv.conj()[None, :])


def conditional_miaa(
    y_given: npt.ArrayLike,
    fourier_given: npt.ArrayLike,
    fourier_missing: npt.ArrayLike,
    reflectivity_power: npt.ArrayLike,
    noise_covariance_given: npt.ArrayLike,
    *,
    noise_covariance_missing: Optional[npt.ArrayLike] = None,
    initial_relative_jitter: float = 1e-12,
    max_condition_number: float = 1e12,
    hermitian_tolerance: float = 1e-9,
) -> ConditionalMIAAResult:
    """Return MMSE-MIAA mean and conditional missing-data covariance.

    ``mean = R_mg R_gg^-1 y_g`` and
    ``C_m|g = R_mm - R_mg R_gg^-1 R_gm``.
    """

    y = _vector(y_given, "y_given")
    fg = _matrix(fourier_given, "fourier_given")
    fm = _matrix(fourier_missing, "fourier_missing")
    power = np.asarray(reflectivity_power, dtype=np.float64).reshape(-1)
    if fg.shape[0] != y.size:
        raise ValueError("fourier_given rows must match y_given")
    if fg.shape[1] != fm.shape[1]:
        raise ValueError("given and missing Fourier matrices need common columns")
    if power.size != fg.shape[1]:
        raise ValueError("reflectivity_power length is inconsistent")
    if np.any(~np.isfinite(power)) or np.any(power < 0):
        raise ValueError("reflectivity_power must be finite and non-negative")

    sigma_g = _covariance(
        noise_covariance_given,
        y.size,
        "noise_covariance_given",
        tolerance=hermitian_tolerance,
    )
    sigma_m = (
        np.zeros((fm.shape[0], fm.shape[0]), dtype=np.complex128)
        if noise_covariance_missing is None
        else _covariance(
            noise_covariance_missing,
            fm.shape[0],
            "noise_covariance_missing",
            tolerance=hermitian_tolerance,
        )
    )

    r_gg = _hermitian((fg * power[None, :]) @ fg.conj().T + sigma_g)
    r_mg = (fm * power[None, :]) @ fg.conj().T
    r_mm = _hermitian((fm * power[None, :]) @ fm.conj().T + sigma_m)

    scale = max(float(np.real(np.trace(r_gg)) / max(r_gg.shape[0], 1)), 1e-30)
    jitter = max(initial_relative_jitter * scale, 0.0)
    identity = np.eye(r_gg.shape[0], dtype=np.complex128)
    for _ in range(12):
        regularized = r_gg + jitter * identity
        condition = float(np.linalg.cond(regularized))
        if np.isfinite(condition) and condition <= max_condition_number:
            try:
                solved_y = np.linalg.solve(regularized, y)
                solved_cross = np.linalg.solve(regularized, r_mg.conj().T)
                break
            except np.linalg.LinAlgError:
                pass
        jitter = max(jitter * 10.0, 1e-15 * scale)
    else:
        raise np.linalg.LinAlgError("unable to stabilize R_gg")

    mean = r_mg @ solved_y
    covariance = _hermitian(r_mm - r_mg @ solved_cross)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    reference = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -1e-7 * reference:
        raise np.linalg.LinAlgError("conditional covariance is materially indefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    covariance = _hermitian(
        (eigenvectors * eigenvalues[None, :]) @ eigenvectors.conj().T
    )

    posterior = np.maximum(np.real(np.diag(covariance)), 0.0)
    prior = np.maximum(np.real(np.diag(r_mm)), 0.0)
    predictability = np.zeros_like(prior)
    valid = prior > 1e-30
    predictability[valid] = 1.0 - posterior[valid] / prior[valid]
    predictability = np.clip(predictability, 0.0, 1.0)
    predicted_snr = np.abs(mean) ** 2 / np.maximum(posterior, 1e-30)

    return ConditionalMIAAResult(
        mean=np.asarray(mean, dtype=np.complex128),
        covariance=np.asarray(covariance, dtype=np.complex128),
        posterior_variance=np.asarray(posterior, dtype=np.float64),
        prior_variance=np.asarray(prior, dtype=np.float64),
        predictability=np.asarray(predictability, dtype=np.float64),
        predicted_snr=np.asarray(predicted_snr, dtype=np.float64),
        condition_number=condition,
        jitter_used=float(jitter),
    )


def confidence_taper(
    predictability: npt.ArrayLike,
    *,
    reject_below: float = 0.25,
    accept_above: float = 0.80,
) -> RealArray:
    values = np.asarray(predictability, dtype=np.float64).reshape(-1)
    if not 0 <= reject_below < accept_above <= 1:
        raise ValueError("require 0 <= reject_below < accept_above <= 1")
    t = np.clip((values - reject_below) / (accept_above - reject_below), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def select_confident_missing_bins(
    result: ConditionalMIAAResult,
    *,
    min_predictability: float = 0.5,
    min_predicted_snr: float = 1.0,
) -> npt.NDArray[np.bool_]:
    if not 0 <= min_predictability <= 1:
        raise ValueError("min_predictability must lie in [0, 1]")
    if min_predicted_snr < 0:
        raise ValueError("min_predicted_snr must be non-negative")
    return (result.predictability >= min_predictability) & (
        result.predicted_snr >= min_predicted_snr
    )
