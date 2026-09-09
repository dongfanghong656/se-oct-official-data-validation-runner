from __future__ import annotations

import numpy as np
import pytest

from seoct_uncertainty import (
    conditional_miaa,
    conditional_miaa_diagonal,
    conservative_confidence_envelope,
    contiguous_confident_support,
)


def fourier(indices: np.ndarray, object_grid: int) -> np.ndarray:
    z = np.arange(object_grid)
    return np.exp(-1j * 2 * np.pi * indices[:, None] * z[None, :] / object_grid)


def test_diagonal_matches_full_covariance() -> None:
    n = 32
    fg = fourier(np.arange(-4, 4), n)
    fm = fourier(np.arange(4, 15), n)
    power = np.zeros(n); power[[7, 13, 21]] = [1.0, 0.4, 0.15]
    y = fg @ np.sqrt(power)
    sigma = np.diag(np.linspace(1e-4, 8e-4, fg.shape[0]))
    full = conditional_miaa(y, fg, fm, power, sigma)
    diagonal = conditional_miaa_diagonal(y, fg, fm, power, sigma)
    np.testing.assert_allclose(diagonal.mean, full.mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(diagonal.posterior_variance, full.posterior_variance, rtol=1e-9, atol=1e-10)


def test_diagonal_rejects_correlated_missing_noise() -> None:
    n = 16
    fg = fourier(np.arange(-2, 2), n); fm = fourier(np.arange(2, 6), n)
    power = np.zeros(n); power[4] = 1.0; y = fg @ np.sqrt(power)
    correlated = np.eye(4); correlated[0, 1] = correlated[1, 0] = 0.1
    with pytest.raises(ValueError, match="correlated"):
        conditional_miaa_diagonal(y, fg, fm, power, np.eye(4)*1e-3, missing_noise_variance=correlated)


def test_confidence_envelope_and_contiguous_support() -> None:
    indices = np.array([-3, -2, -1, 8, 9, 10])
    confidence = np.array([0.9, 0.2, 0.8, 0.9, 0.3, 0.95])
    envelope = conservative_confidence_envelope(indices, confidence, given_min=0, given_max=7)
    np.testing.assert_allclose(envelope, [0.2, 0.2, 0.8, 0.9, 0.3, 0.3])
    accepted = np.array([True, False, True, True, False, True])
    keep = contiguous_confident_support(indices, accepted, given_min=0, given_max=7)
    np.testing.assert_array_equal(keep, [False, False, True, True, False, False])
