from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt

from .conditional import ConditionalMIAAResult, select_confident_missing_bins

RealArray = npt.NDArray[np.float64]


def approximate_phase_standard_deviation(
    result: ConditionalMIAAResult,
    *,
    minimum_amplitude: float = 1e-15,
) -> RealArray:
    """Approximate phase uncertainty for coherent post-processing.

    For proper circular complex Gaussian error at sufficiently high SNR,
    ``Var(phi) ~= C_nn / (2 |mu_n|^2)``. Tiny-mean bins return infinity and
    fail the coherent-support gate.
    """

    if minimum_amplitude <= 0:
        raise ValueError("minimum_amplitude must be positive")
    amplitude_squared = np.abs(result.mean) ** 2
    phase_variance = np.full_like(result.posterior_variance, np.inf)
    valid = amplitude_squared >= minimum_amplitude**2
    phase_variance[valid] = result.posterior_variance[valid] / np.maximum(
        2.0 * amplitude_squared[valid], 1e-30
    )
    return np.sqrt(np.maximum(phase_variance, 0.0))


def select_coherent_missing_bins(
    result: ConditionalMIAAResult,
    *,
    min_predictability: float = 0.5,
    min_predicted_snr: float = 1.0,
    max_phase_standard_deviation: float = 0.3,
) -> npt.NDArray[np.bool_]:
    """Select bins credible enough for coherent ISAM or CAO use."""

    if max_phase_standard_deviation <= 0:
        raise ValueError("max_phase_standard_deviation must be positive")
    base = select_confident_missing_bins(
        result,
        min_predictability=min_predictability,
        min_predicted_snr=min_predicted_snr,
    )
    return base & (
        approximate_phase_standard_deviation(result)
        <= max_phase_standard_deviation
    )


def combine_conditional_results(
    results: list[ConditionalMIAAResult],
    *,
    weights: Optional[npt.ArrayLike] = None,
) -> ConditionalMIAAResult:
    """Combine model variants with the law of total covariance.

    For model index b,
    ``C_total = sum w_b [C_b + (mu_b-mu)(mu_b-mu)^H]``.
    This exposes disagreement among forward/reverse RFIAA traversal,
    bootstrap perturbations, or registered hyperparameters rather than
    hiding it behind one narrow point estimate.
    """

    if not results:
        raise ValueError("results must be non-empty")
    size = results[0].mean.size
    for result in results:
        if result.mean.size != size or result.covariance.shape != (size, size):
            raise ValueError("all results must share the missing-data shape")

    if weights is None:
        weight = np.full(len(results), 1.0 / len(results), dtype=np.float64)
    else:
        weight = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weight.size != len(results):
            raise ValueError("weights length must match results")
        if np.any(~np.isfinite(weight)) or np.any(weight < 0) or float(np.sum(weight)) <= 0:
            raise ValueError("weights must be finite, non-negative and nonzero")
        weight /= np.sum(weight)

    means = np.stack([result.mean for result in results], axis=0)
    mean = np.sum(weight[:, None] * means, axis=0)
    covariance = np.zeros((size, size), dtype=np.complex128)
    prior = np.zeros(size, dtype=np.float64)
    for w, result in zip(weight, results):
        delta = result.mean - mean
        covariance += w * (
            result.covariance + np.outer(delta, delta.conj())
        )
        prior += w * result.prior_variance
    covariance = 0.5 * (covariance + covariance.conj().T)
    posterior = np.maximum(np.real(np.diag(covariance)), 0.0)
    prior = np.maximum(prior, 0.0)
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
        condition_number=max(result.condition_number for result in results),
        jitter_used=max(result.jitter_used for result in results),
    )
