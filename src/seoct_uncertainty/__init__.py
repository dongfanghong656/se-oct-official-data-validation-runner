"""Uncertainty-aware utilities for phase-preserving SE-OCT / MIAA audits."""

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
from .propagation import (
    MonteCarloPropagationResult,
    propagate_conditional_monte_carlo,
    sample_conditional_spectra,
)
from .scalable import (
    ConditionalMIAADiagonalResult,
    conditional_miaa_diagonal,
    conservative_confidence_envelope,
    contiguous_confident_support,
)

__all__ = [
    "ConditionalMIAAResult",
    "ConditionalMIAADiagonalResult",
    "MonteCarloPropagationResult",
    "SingleReflectorCRLB",
    "TwoReflectorSeparationCRLB",
    "approximate_phase_standard_deviation",
    "combine_conditional_results",
    "conditional_miaa",
    "conditional_miaa_diagonal",
    "confidence_taper",
    "conservative_confidence_envelope",
    "contiguous_confident_support",
    "normalized_noise_covariance",
    "propagate_conditional_monte_carlo",
    "sample_conditional_spectra",
    "select_coherent_missing_bins",
    "select_confident_missing_bins",
    "single_reflector_localization_crlb",
    "two_reflector_separation_crlb",
]
