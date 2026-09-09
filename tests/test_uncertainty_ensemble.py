from __future__ import annotations

from dataclasses import replace

import numpy as np

from seoct_uncertainty import (
    approximate_phase_standard_deviation,
    combine_conditional_results,
    conditional_miaa,
    select_coherent_missing_bins,
)


def fourier(indices: np.ndarray, object_grid: int) -> np.ndarray:
    z = np.arange(object_grid)
    return np.exp(-1j * 2 * np.pi * indices[:, None] * z[None, :] / object_grid)


def base_result():
    n = 16
    fg = fourier(np.arange(-2, 2), n)
    fm = fourier(np.arange(2, 6), n)
    power = np.zeros(n)
    power[5] = 1.0
    y = fg @ np.sqrt(power)
    return conditional_miaa(y, fg, fm, power, np.eye(4) * 1e-3)


def test_phase_standard_deviation_formula() -> None:
    result = base_result()
    got = approximate_phase_standard_deviation(result)
    expected = np.sqrt(
        result.posterior_variance
        / np.maximum(2 * np.abs(result.mean) ** 2, 1e-30)
    )
    np.testing.assert_allclose(got, expected)


def test_ensemble_adds_between_model_uncertainty() -> None:
    base = base_result()
    shifted = replace(base, mean=base.mean * np.exp(0.4j))
    ensemble = combine_conditional_results([base, shifted])
    assert np.all(ensemble.posterior_variance >= base.posterior_variance - 1e-12)
    assert float(np.mean(ensemble.predictability)) < float(np.mean(base.predictability))


def test_coherent_gate_rejects_large_phase_uncertainty() -> None:
    base = base_result()
    inflated = replace(
        base,
        covariance=base.covariance * 1e6,
        posterior_variance=base.posterior_variance * 1e6,
        predicted_snr=base.predicted_snr / 1e6,
    )
    accepted = select_coherent_missing_bins(
        inflated,
        min_predictability=0.0,
        min_predicted_snr=0.0,
        max_phase_standard_deviation=0.3,
    )
    assert not np.any(accepted)
