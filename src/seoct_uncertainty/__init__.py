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

__all__ = [
    "ConditionalMIAAResult",
    "approximate_phase_standard_deviation",
    "combine_conditional_results",
    "conditional_miaa",
    "confidence_taper",
    "normalized_noise_covariance",
    "select_coherent_missing_bins",
    "select_confident_missing_bins",
]
