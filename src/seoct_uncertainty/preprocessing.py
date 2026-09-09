from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt

ComplexArray = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class BackgroundDemodulationResult:
    coefficient: complex
    background_subtracted: ComplexArray
    normalized: ComplexArray
    demodulated: ComplexArray


def _complex_vector(value: npt.ArrayLike, name: str) -> ComplexArray:
    out = np.asarray(value, dtype=np.complex128).reshape(-1)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    return out


def _indices(indices: Optional[npt.ArrayLike], size: int) -> npt.NDArray[np.int64]:
    if indices is None:
        return np.arange(size, dtype=np.int64)
    out = np.asarray(indices, dtype=np.int64).reshape(-1)
    if out.size == 0 or np.any(out < 0) or np.any(out >= size):
        raise ValueError("indices must be nonempty and within source range")
    if np.unique(out).size != out.size:
        raise ValueError("indices must be unique")
    return out


def safe_source_spectrum(
    source_spectrum: npt.ArrayLike,
    *,
    relative_floor: float = 1e-6,
) -> ComplexArray:
    source = _complex_vector(source_spectrum, "source_spectrum")
    if relative_floor <= 0:
        raise ValueError("relative_floor must be positive")
    floor = relative_floor * max(float(np.max(np.abs(source))), 1e-30)
    safe = source.copy()
    weak = np.abs(safe) < floor
    if np.any(weak):
        phase = np.exp(1j * np.angle(safe[weak]))
        phase[np.abs(phase) == 0] = 1.0
        safe[weak] = floor * phase
    return safe


def centering_matrix(size: int, *, weights: Optional[npt.ArrayLike] = None) -> ComplexArray:
    """Return the linear operator that subtracts a scalar mean.

    With no weights this matches MATLAB ``x - mean(x)``. With normalized
    non-negative weights summing to one it returns ``I - 1 w^T``.
    """

    if size <= 0:
        raise ValueError("size must be positive")
    if weights is None:
        w = np.full(size, 1.0 / size, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.size != size or np.any(~np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite, non-negative, and match size")
        total = float(np.sum(w))
        if total <= 0:
            raise ValueError("weights must have positive sum")
        w = w / total
    return np.eye(size, dtype=np.complex128) - np.ones((size, 1)) @ w[None, :]


def background_coefficient(
    raw_spectrum: npt.ArrayLike,
    source_spectrum: npt.ArrayLike,
    *,
    rule: Literal["author", "complex_ls"] = "author",
) -> complex:
    """Return the scalar used to subtract a source-shaped background.

    ``author`` reproduces ``raw' * sk / (sk' * sk)`` from the deposited
    MATLAB code. ``complex_ls`` is the standard complex least-squares
    coefficient ``sk' * raw / (sk' * sk)``.
    """

    raw = _complex_vector(raw_spectrum, "raw_spectrum")
    source = _complex_vector(source_spectrum, "source_spectrum")
    if raw.size != source.size:
        raise ValueError("raw_spectrum and source_spectrum must match")
    denominator = np.vdot(source, source)
    if abs(denominator) <= 1e-30:
        raise ValueError("source_spectrum has zero norm")
    if rule == "author":
        return complex(np.vdot(raw, source) / denominator)
    if rule == "complex_ls":
        return complex(np.vdot(source, raw) / denominator)
    raise ValueError(f"unknown rule: {rule}")


def preprocess_for_miaa(
    raw_spectrum: npt.ArrayLike,
    source_spectrum: npt.ArrayLike,
    *,
    input_indices: Optional[npt.ArrayLike] = None,
    background_rule: Literal["author", "complex_ls"] = "author",
    relative_floor: float = 1e-6,
    mean_weights: Optional[npt.ArrayLike] = None,
) -> BackgroundDemodulationResult:
    """Reproduce source subtraction, division, selection, and mean removal."""

    raw = _complex_vector(raw_spectrum, "raw_spectrum")
    source = _complex_vector(source_spectrum, "source_spectrum")
    if raw.size != source.size:
        raise ValueError("raw_spectrum and source_spectrum must match")
    idx = _indices(input_indices, raw.size)
    safe = safe_source_spectrum(source, relative_floor=relative_floor)
    coefficient = background_coefficient(raw, source, rule=background_rule)
    corrected = raw - coefficient * source
    normalized = corrected / safe
    selected = normalized[idx]
    center = centering_matrix(selected.size, weights=mean_weights)
    demodulated = center @ selected
    return BackgroundDemodulationResult(
        coefficient=coefficient,
        background_subtracted=corrected,
        normalized=normalized,
        demodulated=demodulated,
    )


def demodulation_operator(
    source_spectrum: npt.ArrayLike,
    *,
    input_indices: Optional[npt.ArrayLike] = None,
    relative_floor: float = 1e-6,
    mean_weights: Optional[npt.ArrayLike] = None,
) -> ComplexArray:
    """Map pre-subtraction spectral samples directly to MIAA input.

    Because a source-shaped scalar subtraction becomes a constant after
    division by the same source spectrum, the subsequent mean subtraction
    removes it exactly. Hence the MIAA input operator is independent of the
    scalar background coefficient rule:

        H = Q D(1/S_selected) P_selected.
    """

    source = _complex_vector(source_spectrum, "source_spectrum")
    idx = _indices(input_indices, source.size)
    safe = safe_source_spectrum(source, relative_floor=relative_floor)
    selection = np.zeros((idx.size, source.size), dtype=np.complex128)
    selection[np.arange(idx.size), idx] = 1.0
    division = np.diag(1.0 / safe[idx])
    center = centering_matrix(idx.size, weights=mean_weights)
    return center @ division @ selection


def demodulated_noise_covariance(
    source_spectrum: npt.ArrayLike,
    *,
    raw_noise_variance: Optional[npt.ArrayLike] = None,
    raw_noise_covariance: Optional[npt.ArrayLike] = None,
    input_indices: Optional[npt.ArrayLike] = None,
    relative_floor: float = 1e-6,
    mean_weights: Optional[npt.ArrayLike] = None,
) -> ComplexArray:
    """Propagate spectral noise through the exact MIAA demodulation chain.

    ``Sigma_y = H Sigma_raw H^H``, where ``H`` includes source division,
    input-band selection, and per-A-line mean subtraction. The last operation
    introduces correlations and removes one complex mode, so even white raw
    noise does not remain ``eta I``.
    """

    source = _complex_vector(source_spectrum, "source_spectrum")
    if raw_noise_variance is not None and raw_noise_covariance is not None:
        raise ValueError("provide raw_noise_variance or raw_noise_covariance, not both")
    if raw_noise_covariance is None:
        if raw_noise_variance is None:
            variance = np.ones(source.size, dtype=np.float64)
        else:
            variance = np.asarray(raw_noise_variance, dtype=np.float64)
            if variance.ndim == 0:
                variance = np.full(source.size, float(variance))
            variance = variance.reshape(-1)
            if variance.size != source.size:
                raise ValueError("raw_noise_variance has wrong length")
            if np.any(~np.isfinite(variance)) or np.any(variance < 0):
                raise ValueError("raw_noise_variance must be finite and non-negative")
        raw_cov = np.diag(variance.astype(np.complex128))
    else:
        raw_cov = np.asarray(raw_noise_covariance, dtype=np.complex128)
        if raw_cov.shape != (source.size, source.size):
            raise ValueError("raw_noise_covariance has wrong shape")
        if not np.all(np.isfinite(raw_cov)):
            raise ValueError("raw_noise_covariance contains non-finite values")
        raw_cov = 0.5 * (raw_cov + raw_cov.conj().T)
        eig = np.linalg.eigvalsh(raw_cov)
        scale = max(float(np.max(np.abs(eig))), 1.0)
        if float(np.min(eig)) < -1e-10 * scale:
            raise ValueError("raw_noise_covariance must be positive semidefinite")
    operator = demodulation_operator(
        source,
        input_indices=input_indices,
        relative_floor=relative_floor,
        mean_weights=mean_weights,
    )
    covariance = operator @ raw_cov @ operator.conj().T
    return 0.5 * (covariance + covariance.conj().T)


def scalar_covariance_approximation(covariance: npt.ArrayLike) -> tuple[ComplexArray, float, float]:
    """Return trace-matched ``eta I`` and its relative Frobenius mismatch."""

    cov = np.asarray(covariance, dtype=np.complex128)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be square")
    eta = float(np.real(np.trace(cov)) / cov.shape[0])
    scalar = np.eye(cov.shape[0], dtype=np.complex128) * eta
    mismatch = float(np.linalg.norm(cov - scalar, "fro") / max(np.linalg.norm(cov, "fro"), 1e-30))
    return scalar, eta, mismatch
