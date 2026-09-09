from __future__ import annotations

import numpy as np
import pytest

from seoct_uncertainty import (
    conditional_miaa,
    confidence_taper,
    normalized_noise_covariance,
)


def fourier(indices: np.ndarray, object_grid: int) -> np.ndarray:
    z = np.arange(object_grid)
    return np.exp(-1j * 2 * np.pi * indices[:, None] * z[None, :] / object_grid)


def test_noise_change_of_variables() -> None:
    source = np.array([1.0, 0.5, 0.25])
    got = normalized_noise_covariance(source, raw_noise_variance=2.0)
    np.testing.assert_allclose(got, np.diag(2.0 / source**2))


def test_conditional_covariance_is_psd_and_below_prior() -> None:
    n = 32
    fg = fourier(np.arange(-4, 4), n)
    fm = fourier(np.arange(4, 12), n)
    power = np.zeros(n)
    power[[7, 13]] = [1.0, 0.4]
    y = fg @ np.sqrt(power)
    result = conditional_miaa(y, fg, fm, power, np.eye(8) * 1e-3)
    assert np.min(np.linalg.eigvalsh(result.covariance)) >= -1e-10
    assert np.all(result.posterior_variance <= result.prior_variance + 1e-9)
    assert np.all((0 <= result.predictability) & (result.predictability <= 1))


def test_more_noise_reduces_predictability() -> None:
    n = 32
    fg = fourier(np.arange(-4, 4), n)
    fm = fourier(np.arange(4, 12), n)
    power = np.zeros(n)
    power[10] = 1.0
    y = fg @ np.sqrt(power)
    low = conditional_miaa(y, fg, fm, power, np.eye(8) * 1e-5)
    high = conditional_miaa(y, fg, fm, power, np.eye(8) * 1e-1)
    assert float(np.mean(low.predictability)) > float(np.mean(high.predictability))


def test_confidence_taper_is_monotone() -> None:
    values = np.linspace(0, 1, 11)
    weights = confidence_taper(values, reject_below=0.2, accept_above=0.8)
    assert np.all(np.diff(weights) >= -1e-12)
    assert weights[0] == 0
    assert weights[-1] == 1


def test_shape_validation_fails_closed() -> None:
    with pytest.raises(ValueError):
        conditional_miaa(
            np.ones(3),
            np.ones((4, 5)),
            np.ones((2, 5)),
            np.ones(5),
            np.eye(3),
        )
