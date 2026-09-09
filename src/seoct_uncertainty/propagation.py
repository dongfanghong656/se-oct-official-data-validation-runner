from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from .conditional import ConditionalMIAAResult

ComplexArray = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class MonteCarloPropagationResult:
    mean: ComplexArray
    variance: RealArray
    phase_standard_deviation: RealArray
    sample_count: int
    seed: int


def sample_conditional_spectra(
    result: ConditionalMIAAResult,
    *,
    sample_count: int,
    seed: int = 0,
    eigenvalue_tolerance: float = 1e-10,
) -> ComplexArray:
    """Draw proper-complex Gaussian spectra from conditional moments."""

    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    covariance = 0.5 * (result.covariance + result.covariance.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    reference = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -eigenvalue_tolerance * reference:
        raise ValueError("covariance is materially indefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    transform = eigenvectors * np.sqrt(eigenvalues)[None, :]
    rng = np.random.default_rng(seed)
    standard = (
        rng.standard_normal((sample_count, result.mean.size))
        + 1j * rng.standard_normal((sample_count, result.mean.size))
    ) / np.sqrt(2.0)
    return result.mean[None, :] + standard @ transform.T


def propagate_conditional_monte_carlo(
    result: ConditionalMIAAResult,
    operator: Callable[[ComplexArray], npt.ArrayLike],
    *,
    sample_count: int = 256,
    seed: int = 0,
    minimum_resultant_length: float = 1e-8,
) -> MonteCarloPropagationResult:
    """Propagate full spectral covariance through a fixed coherent operator."""

    samples = sample_conditional_spectra(result, sample_count=sample_count, seed=seed)
    outputs = [np.asarray(operator(sample), dtype=np.complex128) for sample in samples]
    shape = outputs[0].shape
    if any(output.shape != shape for output in outputs):
        raise ValueError("operator returned inconsistent shapes")
    stack = np.stack(outputs, axis=0)
    mean = np.mean(stack, axis=0)
    variance = np.mean(np.abs(stack - mean[None, ...]) ** 2, axis=0)
    resultant = np.mean(np.exp(1j * np.angle(stack)), axis=0)
    length = np.clip(np.abs(resultant), minimum_resultant_length, 1.0)
    phase_std = np.sqrt(np.maximum(-2.0 * np.log(length), 0.0))
    return MonteCarloPropagationResult(
        mean=np.asarray(mean, dtype=np.complex128),
        variance=np.asarray(variance, dtype=np.float64),
        phase_standard_deviation=np.asarray(phase_std, dtype=np.float64),
        sample_count=sample_count,
        seed=seed,
    )
