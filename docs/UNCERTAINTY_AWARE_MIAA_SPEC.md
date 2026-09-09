# Uncertainty-aware MIAA specification

## Purpose

MIAA is an MMSE conditional mean estimator. A scientifically complete output must also report the conditional missing-data covariance

```
C_m|g = R_mm - R_mg R_gg^-1 R_gm
```

rather than presenting every extrapolated bin as equally certain.

## Required outputs

For each missing spectral bin: complex mean, prior variance, posterior variance, fractional variance reduction (`predictability`), predicted SNR, confidence weight, and effective-support decision.

For each A-line: condition number, applied jitter, noise-covariance provenance, input/output support, IAA iteration metadata, and failure reason.

## Noise model

For `y_norm = y_raw / S`, propagate noise as

```
Sigma_norm = D(1/S) Sigma_raw D(1/S)^H
```

Repeated backgrounds are preferable to a scalar-noise assumption. Interpolation and depth gating can introduce correlated noise and should be included or prewhitened.

## Effective support

Do not equate a fixed target length with measured bandwidth. A missing bin enters coherent ISAM only when registered predictability and predicted-SNR thresholds are passed. Preserve unweighted output for audit.

## Boundaries

The covariance is conditional on the estimated IAA power model. It omits power-estimation uncertainty, wavelength-dependent sample scattering, chromatic optical transfer, multiple scattering, and model mismatch. Confidence weighting cannot replace independent physical wideband validation.
