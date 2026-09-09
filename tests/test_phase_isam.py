from __future__ import annotations

import numpy as np

from seoct_phase_isam import (
    ewald_kz,
    expected_finite_aperture_peak_intensity,
    gaussian_phase_coherence,
    miaa_linear_mmse,
    opl_rms_for_intensity_coherence,
    phase_rms_from_one_way_opl,
    support_geometry,
)


def test_ewald_mapping_and_paraxial_direction() -> None:
    k = 16.0
    assert np.isclose(ewald_kz(k, 0.0), 2 * k)
    assert ewald_kz(k, 4.0) < 2 * k


def test_gaussian_phase_coherence_monte_carlo() -> None:
    rng = np.random.default_rng(42)
    sigma = 0.7
    empirical = abs(np.mean(np.exp(1j * rng.normal(scale=sigma, size=400_000))))
    expected, _ = gaussian_phase_coherence(sigma)
    assert abs(empirical - float(expected)) < 4e-3


def test_round_trip_opl_conversion() -> None:
    wavelength = 0.510
    opl = opl_rms_for_intensity_coherence(0.9, wavelength)
    phase = phase_rms_from_one_way_opl(opl, wavelength)
    assert np.isclose(gaussian_phase_coherence(phase)[1], 0.9)


def test_miaa_preserves_even_wrong_global_phase() -> None:
    rng = np.random.default_rng(7)
    ng, nm, nz = 12, 9, 24
    fg = rng.normal(size=(ng, nz)) + 1j * rng.normal(size=(ng, nz))
    fm = rng.normal(size=(nm, nz)) + 1j * rng.normal(size=(nm, nz))
    power = np.exp(rng.normal(size=nz))
    y = rng.normal(size=ng) + 1j * rng.normal(size=ng)
    sigma = np.eye(ng) * 0.2
    theta = 1.13
    base = miaa_linear_mmse(y, fg, fm, power, sigma)
    rotated = miaa_linear_mmse(np.exp(1j * theta) * y, fg, fm, power, sigma)
    np.testing.assert_allclose(rotated, np.exp(1j * theta) * base, rtol=1e-12, atol=1e-12)


def test_finite_aperture_has_incoherent_floor() -> None:
    assert np.isclose(expected_finite_aperture_peak_intensity(10.0, np.ones(100)), 0.01, rtol=1e-3)


def test_miaa_support_changes_axial_not_lateral_support() -> None:
    geometry = support_geometry(
        k0_um_inv=16.4,
        measured_k_width_um_inv=0.45,
        predicted_k_width_um_inv=1.8,
        q_max_um_inv=4.0,
    )
    assert np.isclose(geometry.predicted_on_axis_kz_width_um_inv, 4 * geometry.measured_on_axis_kz_width_um_inv)
    assert geometry.q_max_um_inv == 4.0


def test_direct_iaa_miaa_recovers_single_exponential_target_spectrum() -> None:
    from seoct_phase_isam import iaa_miaa_predict
    z_grid = np.linspace(0.0, 40.0, 161)
    k_given = np.linspace(-0.25, 0.25, 33)
    k_target = np.linspace(-1.0, 1.0, 129)
    fg = np.exp(-1j * 2 * k_given[:, None] * z_grid[None, :])
    ft = np.exp(-1j * 2 * k_target[:, None] * z_grid[None, :])
    amplitude = 0.8 * np.exp(0.4j)
    y = amplitude * fg[:, 73]
    truth = amplitude * ft[:, 73]
    predicted, _, _, _ = iaa_miaa_predict(y, fg, ft, noise_variance=1e-10, iterations=5)
    correlation = abs(np.vdot(truth, predicted)) / (np.linalg.norm(truth) * np.linalg.norm(predicted))
    assert correlation > 0.999


def test_full_iaa_miaa_is_global_phase_equivariant() -> None:
    from seoct_phase_isam import iaa_miaa_predict
    z_grid = np.linspace(0.0, 50.0, 201)
    k_given = np.linspace(-0.25, 0.25, 33)
    k_target = np.linspace(-1.0, 1.0, 129)
    fg = np.exp(-1j * 2 * k_given[:, None] * z_grid[None, :])
    ft = np.exp(-1j * 2 * k_target[:, None] * z_grid[None, :])
    y = 0.9 * np.exp(0.3j) * fg[:, 61] + 0.35 * np.exp(-0.9j) * fg[:, 123]
    theta = 1.271
    base, reflectivity, power, eta = iaa_miaa_predict(y, fg, ft, noise_variance=1e-8, iterations=5)
    rotated, reflectivity_rotated, power_rotated, eta_rotated = iaa_miaa_predict(
        np.exp(1j * theta) * y, fg, ft, noise_variance=1e-8, iterations=5
    )
    phase = np.exp(1j * theta)
    assert np.linalg.norm(rotated - phase * base) / np.linalg.norm(base) < 1e-7
    assert (
        np.linalg.norm(reflectivity_rotated - phase * reflectivity)
        / np.linalg.norm(reflectivity)
        < 1e-7
    )
    assert np.linalg.norm(power_rotated - power) / np.linalg.norm(power) < 1e-7
    assert np.isclose(eta_rotated, eta)
