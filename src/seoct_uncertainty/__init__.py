"""Uncertainty-aware utilities for phase-preserving SE-OCT / MIAA audits."""

from .calibration import (
    PhasePolynomialFit,
    combine_additive_and_calibration_noise,
    evaluate_phase_fit,
    fit_phase_polynomial,
    multiplicative_phase_noise_covariance,
    phase_error_covariance,
)
from .conditional import (
    ConditionalMIAAResult,
    conditional_miaa,
    confidence_taper,
    normalized_noise_covariance,
    select_confident_missing_bins,
)
from .ensemble import (
    approximate_phase_standard_deviation,
    combine_conditional_results,
    select_coherent_missing_bins,
)
from .identifiability import (
    SingleReflectorCRLB,
    TwoReflectorSeparationCRLB,
    single_reflector_localization_crlb,
    two_reflector_separation_crlb,
)
from .phase_snr import phase_variance_from_complex_snr
from .propagation import (
    MonteCarloPropagationResult,
    propagate_conditional_monte_carlo,
    sample_conditional_spectra,
)

from .preprocessing import (
    BackgroundDemodulationResult,
    background_coefficient,
    centering_matrix,
    demodulated_noise_covariance,
    demodulation_operator,
    preprocess_for_miaa,
    safe_source_spectrum,
    scalar_covariance_approximation,
)

from .scalable import (
    ConditionalMIAADiagonalResult,
    conditional_miaa_diagonal,
    conservative_confidence_envelope,
    contiguous_confident_support,
)

__all__ = [
    "PhasePolynomialFit",
    "combine_additive_and_calibration_noise",
    "evaluate_phase_fit",
    "fit_phase_polynomial",
    "multiplicative_phase_noise_covariance",
    "phase_error_covariance",
    "ConditionalMIAAResult",
    "conditional_miaa",
    "confidence_taper",
    "normalized_noise_covariance",
    "select_confident_missing_bins",
    "approximate_phase_standard_deviation",
    "combine_conditional_results",
    "select_coherent_missing_bins",
    "SingleReflectorCRLB",
    "TwoReflectorSeparationCRLB",
    "single_reflector_localization_crlb",
    "two_reflector_separation_crlb",
    "phase_variance_from_complex_snr",
    "MonteCarloPropagationResult",
    "propagate_conditional_monte_carlo",
    "sample_conditional_spectra",
    "ConditionalMIAADiagonalResult",
    "conditional_miaa_diagonal",
    "conservative_confidence_envelope",
    "contiguous_confident_support",
    "BackgroundDemodulationResult",
    "background_coefficient",
    "centering_matrix",
    "demodulated_noise_covariance",
    "demodulation_operator",
    "preprocess_for_miaa",
    "safe_source_spectrum",
    "scalar_covariance_approximation",
]
