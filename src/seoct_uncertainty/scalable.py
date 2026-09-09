from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from .conditional import _covariance, _hermitian, _matrix, _vector

ComplexArray = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ConditionalMIAADiagonalResult:
    """Scalable conditional MIAA moments without an Nm x Nm covariance."""

    mean: ComplexArray
    posterior_variance: RealArray
    prior_variance: RealArray
    predictability: RealArray
    predicted_snr: RealArray
    condition_number: float
    jitter_used: float


def _noise_diagonal(value: Optional[npt.ArrayLike], size: int, name: str) -> RealArray:
    if value is None:
        return np.zeros(size, dtype=np.float64)
    array = np.asarray(value)
    if array.ndim == 2:
        if array.shape != (size, size):
            raise ValueError(f"{name} covariance has wrong shape")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")
        diagonal = np.real(np.diag(array))
        off_diagonal = array - np.diag(np.diag(array))
        if np.linalg.norm(off_diagonal) > 1e-10 * max(np.linalg.norm(array), 1.0):
            raise ValueError(f"{name} is correlated; diagonal-only propagation would be invalid")
    else:
        diagonal = np.asarray(value, dtype=np.float64)
        if diagonal.ndim == 0:
            diagonal = np.full(size, float(diagonal))
        diagonal = diagonal.reshape(-1)
        if diagonal.size != size:
            raise ValueError(f"{name} has wrong length")
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    return np.asarray(diagonal, dtype=np.float64)


def conditional_miaa_diagonal(
    y_given: npt.ArrayLike,
    fourier_given: npt.ArrayLike,
    fourier_missing: npt.ArrayLike,
    reflectivity_power: npt.ArrayLike,
    noise_covariance_given: npt.ArrayLike,
    *,
    missing_noise_variance: Optional[npt.ArrayLike] = None,
    initial_relative_jitter: float = 1e-12,
    max_condition_number: float = 1e12,
    hermitian_tolerance: float = 1e-9,
) -> ConditionalMIAADiagonalResult:
    """Return conditional mean and per-bin variance at O(Nm*Ng) storage.

    The diagonal is
    ``diag(R_mm) - diag(R_mg R_gg^-1 R_gm)``. Use the full conditional
    interface when cross-bin covariance is needed for coherent propagation.
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

    sigma_g = _covariance(noise_covariance_given, y.size, "noise_covariance_given", tolerance=hermitian_tolerance)
    sigma_m_diag = _noise_diagonal(missing_noise_variance, fm.shape[0], "missing_noise_variance")
    r_gg = _hermitian((fg * power[None, :]) @ fg.conj().T + sigma_g)
    r_mg = (fm * power[None, :]) @ fg.conj().T
    prior = np.sum(np.abs(fm) ** 2 * power[None, :], axis=1).real
    prior = np.maximum(prior + sigma_m_diag, 0.0)

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
    reduction = np.real(np.einsum("ij,ji->i", r_mg, solved_cross))
    posterior = prior - reduction
    reference = max(float(np.max(np.abs(prior))), 1.0)
    if float(np.min(posterior)) < -1e-7 * reference:
        raise np.linalg.LinAlgError("diagonal conditional variance is materially negative")
    posterior = np.maximum(posterior, 0.0)
    predictability = np.zeros_like(prior)
    valid = prior > 1e-30
    predictability[valid] = 1.0 - posterior[valid] / prior[valid]
    predictability = np.clip(predictability, 0.0, 1.0)
    predicted_snr = np.abs(mean) ** 2 / np.maximum(posterior, 1e-30)
    return ConditionalMIAADiagonalResult(
        mean=np.asarray(mean, dtype=np.complex128),
        posterior_variance=np.asarray(posterior, dtype=np.float64),
        prior_variance=np.asarray(prior, dtype=np.float64),
        predictability=np.asarray(predictability, dtype=np.float64),
        predicted_snr=np.asarray(predicted_snr, dtype=np.float64),
        condition_number=condition,
        jitter_used=float(jitter),
    )


def conservative_confidence_envelope(
    missing_indices: npt.ArrayLike,
    confidence: npt.ArrayLike,
    *,
    given_min: int,
    given_max: int,
) -> RealArray:
    """Prevent confidence from increasing after moving farther out."""

    indices = np.asarray(missing_indices, dtype=int).reshape(-1)
    values = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if indices.size != values.size:
        raise ValueError("missing_indices and confidence must have same length")
    if len(np.unique(indices)) != indices.size:
        raise ValueError("missing_indices must be unique")
    if np.any(~np.isfinite(values)):
        raise ValueError("confidence contains non-finite values")
    output = values.copy()

    left = np.where(indices < given_min)[0]
    left = left[np.argsort(indices[left])[::-1]]
    running = np.inf
    previous = given_min
    for position in left:
        current = int(indices[position])
        if previous - current != 1:
            running = min(running, 0.0)
        running = min(running, float(values[position]))
        output[position] = running
        previous = current

    right = np.where(indices > given_max)[0]
    right = right[np.argsort(indices[right])]
    running = np.inf
    previous = given_max
    for position in right:
        current = int(indices[position])
        if current - previous != 1:
            running = min(running, 0.0)
        running = min(running, float(values[position]))
        output[position] = running
        previous = current

    output[(indices >= given_min) & (indices <= given_max)] = 0.0
    return np.clip(output, 0.0, 1.0)


def contiguous_confident_support(
    missing_indices: npt.ArrayLike,
    accepted: npt.ArrayLike,
    *,
    given_min: int,
    given_max: int,
) -> npt.NDArray[np.bool_]:
    """Keep only contiguous accepted bins adjacent to measured support."""

    indices = np.asarray(missing_indices, dtype=int).reshape(-1)
    accepted_mask = np.asarray(accepted, dtype=bool).reshape(-1)
    if indices.size != accepted_mask.size:
        raise ValueError("missing_indices and accepted must have same length")
    confidence = conservative_confidence_envelope(
        indices,
        accepted_mask.astype(float),
        given_min=given_min,
        given_max=given_max,
    )
    return confidence >= 1.0
