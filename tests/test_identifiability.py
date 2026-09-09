from __future__ import annotations

import numpy as np

from seoct_uncertainty import (
    single_reflector_localization_crlb,
    two_reflector_separation_crlb,
)


def test_localization_improves_with_lower_noise() -> None:
    k = np.linspace(10.0, 11.0, 64)
    high = single_reflector_localization_crlb(k, 1.0 + 0.2j, 0.7, np.eye(64) * 1e-2)
    low = single_reflector_localization_crlb(k, 1.0 + 0.2j, 0.7, np.eye(64) * 1e-3)
    assert low.identifiable and high.identifiable
    assert low.z_standard_deviation < high.z_standard_deviation


def test_localization_improves_with_bandwidth() -> None:
    narrow_k = np.linspace(10.45, 10.55, 64)
    wide_k = np.linspace(10.0, 11.0, 64)
    narrow = single_reflector_localization_crlb(narrow_k, 1.0, 0.4, np.eye(64) * 1e-3)
    wide = single_reflector_localization_crlb(wide_k, 1.0, 0.4, np.eye(64) * 1e-3)
    assert narrow.identifiable and wide.identifiable
    assert wide.z_standard_deviation < narrow.z_standard_deviation


def test_two_reflector_separation_becomes_ill_conditioned_when_close() -> None:
    k = np.linspace(10.0, 11.0, 64)
    covariance = np.eye(64) * 1e-3
    close = two_reflector_separation_crlb(k, 1.0, 0.8 * np.exp(0.3j), 0.5, 0.02, covariance)
    separated = two_reflector_separation_crlb(k, 1.0, 0.8 * np.exp(0.3j), 0.5, 0.8, covariance)
    assert separated.identifiable
    assert (not close.identifiable) or close.condition_number > separated.condition_number
    assert (not close.identifiable) or close.separation_standard_deviation > separated.separation_standard_deviation


def test_exactly_coincident_two_reflectors_are_not_identifiable() -> None:
    k = np.linspace(10.0, 11.0, 64)
    result = two_reflector_separation_crlb(k, 1.0, 0.8 * np.exp(0.3j), 0.5, 0.0, np.eye(64) * 1e-3)
    assert not result.identifiable
    assert result.effective_rank < 6
    assert np.isinf(result.separation_standard_deviation)
