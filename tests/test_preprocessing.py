from __future__ import annotations

import numpy as np

from seoct_uncertainty.preprocessing import (
    background_coefficient,
    demodulated_noise_covariance,
    demodulation_operator,
    preprocess_for_miaa,
    scalar_covariance_approximation,
)


def test_author_and_ls_background_rules_cancel_after_normalize_and_center() -> None:
    rng = np.random.default_rng(11)
    source = np.linspace(0.2, 1.0, 40)
    raw = rng.normal(size=40) + 1j * rng.normal(size=40)
    idx = np.arange(5, 35)
    author = preprocess_for_miaa(raw, source, input_indices=idx, background_rule="author")
    least_squares = preprocess_for_miaa(raw, source, input_indices=idx, background_rule="complex_ls")
    assert not np.isclose(author.coefficient, least_squares.coefficient)
    np.testing.assert_allclose(author.demodulated, least_squares.demodulated, atol=1e-12, rtol=1e-12)


def test_demodulation_operator_matches_explicit_processing() -> None:
    rng = np.random.default_rng(12)
    source = np.linspace(0.3, 1.2, 24)
    raw = rng.normal(size=24) + 1j * rng.normal(size=24)
    idx = np.arange(3, 21)
    explicit = preprocess_for_miaa(raw, source, input_indices=idx).demodulated
    operator = demodulation_operator(source, input_indices=idx)
    np.testing.assert_allclose(operator @ raw, explicit, atol=1e-12, rtol=1e-12)


def test_demodulated_covariance_matches_monte_carlo() -> None:
    rng = np.random.default_rng(13)
    source = np.linspace(0.2, 1.0, 18)
    idx = np.arange(2, 16)
    variance = np.linspace(0.4, 1.4, source.size)
    expected = demodulated_noise_covariance(source, raw_noise_variance=variance, input_indices=idx)
    operator = demodulation_operator(source, input_indices=idx)
    trials = 120_000
    noise = (rng.normal(size=(source.size, trials)) + 1j * rng.normal(size=(source.size, trials))) * np.sqrt(variance[:, None] / 2)
    transformed = operator @ noise
    empirical = transformed @ transformed.conj().T / trials
    np.testing.assert_allclose(empirical, expected, rtol=0.025, atol=0.025 * np.max(np.diag(expected).real))


def test_mean_subtraction_creates_one_null_mode() -> None:
    source = np.linspace(0.2, 1.0, 20)
    covariance = demodulated_noise_covariance(source)
    eigenvalues = np.linalg.eigvalsh(covariance)
    assert eigenvalues[0] <= 1e-10 * eigenvalues[-1]
    assert np.sum(eigenvalues > 1e-10 * eigenvalues[-1]) == source.size - 1


def test_scalar_eta_is_not_exact_for_nonflat_source() -> None:
    source = np.geomspace(0.1, 1.0, 32)
    covariance = demodulated_noise_covariance(source)
    _, eta, mismatch = scalar_covariance_approximation(covariance)
    assert eta > 0
    assert mismatch > 0.2


def test_complex_ls_coefficient_orthogonalizes_source() -> None:
    rng = np.random.default_rng(14)
    source = np.linspace(0.1, 1.0, 50)
    raw = rng.normal(size=50) + 1j * rng.normal(size=50)
    coefficient = background_coefficient(raw, source, rule="complex_ls")
    residual = raw - coefficient * source
    assert abs(np.vdot(source, residual)) < 1e-11 * np.linalg.norm(source) * np.linalg.norm(raw)
