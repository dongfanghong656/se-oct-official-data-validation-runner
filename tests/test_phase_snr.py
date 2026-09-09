from __future__ import annotations

import numpy as np
import pytest

from seoct_uncertainty import phase_variance_from_complex_snr


def test_phase_variance_high_snr_formula_and_low_snr_gate() -> None:
    snr = np.array([0.5, 1.0, 10.0, 100.0])
    variance = phase_variance_from_complex_snr(snr)
    assert np.isinf(variance[0])
    np.testing.assert_allclose(variance[1:], 1.0 / (2.0 * snr[1:]))


def test_phase_variance_rejects_invalid_snr() -> None:
    with pytest.raises(ValueError):
        phase_variance_from_complex_snr(np.array([1.0, -1.0]))
