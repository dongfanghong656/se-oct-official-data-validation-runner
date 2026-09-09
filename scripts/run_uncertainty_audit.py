#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seoct_uncertainty import conditional_miaa, confidence_taper, normalized_noise_covariance

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "uncertainty_audit"
OUT.mkdir(parents=True, exist_ok=True)


def fourier(indices: np.ndarray, object_grid: int) -> np.ndarray:
    z = np.arange(object_grid)
    return np.exp(-1j * 2 * np.pi * indices[:, None] * z[None, :] / object_grid)


def corr(ref: np.ndarray, pred: np.ndarray) -> float:
    den = np.linalg.norm(ref) * np.linalg.norm(pred)
    return float(abs(np.vdot(ref, pred)) / den) if den > 0 else float("nan")


def main() -> None:
    rng = np.random.default_rng(20260909)
    n = 256
    given_indices = np.arange(-16, 16)
    all_indices = np.arange(-64, 64)
    missing = ~np.isin(all_indices, given_indices)
    fg = fourier(given_indices, n)
    fm = fourier(all_indices[missing], n)

    reflectivity = np.zeros(n, dtype=np.complex128)
    reflectivity[[86, 104, 151]] = [1.0, 0.55 * np.exp(0.7j), 0.30 * np.exp(-1.1j)]
    power = np.abs(reflectivity) ** 2
    source = np.exp(-0.5 * (given_indices / 13.0) ** 2)
    raw_variance = np.full(given_indices.size, 2.5e-4)
    sigma = normalized_noise_covariance(
        source, raw_noise_variance=raw_variance, relative_floor=0.03
    )
    y_clean = fg @ reflectivity
    raw_noise = np.sqrt(raw_variance / 2) * (
        rng.standard_normal(given_indices.size) + 1j * rng.standard_normal(given_indices.size)
    )
    y = y_clean + raw_noise / source
    result = conditional_miaa(y, fg, fm, power, sigma, max_condition_number=1e10)
    truth = fm @ reflectivity
    weight = confidence_taper(result.predictability, reject_below=0.15, accept_above=0.75)

    squared_error = np.zeros((300, truth.size))
    for trial in range(squared_error.shape[0]):
        noise = np.sqrt(raw_variance / 2) * (
            rng.standard_normal(given_indices.size) + 1j * rng.standard_normal(given_indices.size)
        )
        trial_result = conditional_miaa(
            y_clean + noise / source,
            fg,
            fm,
            power,
            sigma,
            max_condition_number=1e10,
        )
        squared_error[trial] = np.abs(trial_result.mean - truth) ** 2
    empirical_mse = np.mean(squared_error, axis=0)
    ratio = empirical_mse / np.maximum(result.posterior_variance, 1e-30)

    report = {
        "schema": "uncertainty-aware-miaa-audit-v1",
        "status": "VERIFIED_SYNTHETIC",
        "missing_band_metrics": {
            "point_estimate_complex_correlation": corr(truth, result.mean),
            "uncertainty_weighted_complex_correlation": corr(truth, weight * result.mean),
            "median_predictability": float(np.median(result.predictability)),
            "minimum_predictability": float(np.min(result.predictability)),
            "maximum_predictability": float(np.max(result.predictability)),
            "fraction_predictability_ge_0_5": float(np.mean(result.predictability >= 0.5)),
            "condition_number": result.condition_number,
            "jitter_used": result.jitter_used,
        },
        "posterior_variance_calibration": {
            "trials": int(squared_error.shape[0]),
            "median_empirical_mse_over_reported_variance": float(np.median(ratio)),
            "q10_empirical_mse_over_reported_variance": float(np.quantile(ratio, 0.1)),
            "q90_empirical_mse_over_reported_variance": float(np.quantile(ratio, 0.9)),
        },
        "profile_metrics": {
            "note": "Profile-width and false-peak acceptance remain separate registered audits."
        },
        "decisions": {
            "posterior_uncertainty_should_be_reported": True,
            "confidence_weighting_replaces_physical_validation": False,
            "heteroscedastic_noise_should_be_modelled": True,
        },
        "boundary": "Known-power synthetic Gaussian audit; model/power uncertainty is not included.",
    }
    (OUT / "uncertainty_audit_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    order = np.argsort(all_indices[missing])
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(all_indices[missing][order], result.predictability[order], label="predictability")
    plt.plot(all_indices[missing][order], weight[order], label="confidence taper")
    plt.xlabel("missing spectral-bin index")
    plt.ylabel("confidence")
    plt.ylim(-0.03, 1.03)
    plt.title("MIAA extrapolated bins do not have uniform confidence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "predictability_and_taper.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    plt.scatter(result.posterior_variance, empirical_mse, s=18)
    vmax = max(float(np.max(result.posterior_variance)), float(np.max(empirical_mse)))
    plt.plot([0, vmax], [0, vmax])
    plt.xlabel("reported conditional variance")
    plt.ylabel("empirical MSE")
    plt.title("Conditional-variance calibration under the assumed model")
    plt.tight_layout()
    plt.savefig(OUT / "posterior_variance_calibration.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
