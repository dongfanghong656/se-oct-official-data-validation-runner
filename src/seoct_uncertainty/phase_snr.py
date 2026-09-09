from __future__ import annotations

import numpy as np
import numpy.typing as npt


def phase_variance_from_complex_snr(
    complex_snr: npt.ArrayLike,
    *,
    low_snr_infinite_below: float = 1.0,
) -> npt.NDArray[np.float64]:
    """High-SNR phase-variance approximation for a noisy complex phasor.

    For proper circular complex noise, ``Var(phi) ~= 1/(2*SNR)``. The
    approximation is not trusted below ``low_snr_infinite_below``; those bins
    return infinity and must fail a coherent phase-confidence gate.
    """

    snr = np.asarray(complex_snr, dtype=np.float64)
    if np.any(~np.isfinite(snr)) or np.any(snr < 0):
        raise ValueError("complex_snr must be finite and non-negative")
    if low_snr_infinite_below <= 0:
        raise ValueError("low_snr_infinite_below must be positive")
    variance = np.full_like(snr, np.inf)
    valid = snr >= low_snr_infinite_below
    variance[valid] = 1.0 / np.maximum(2.0 * snr[valid], 1e-30)
    return variance
