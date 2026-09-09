#!/usr/bin/env python3
"""Registered model-misspecification calibration for MIAA uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from uncertainty_miaa_audit import calibration_metrics, combine_results, conditional_miaa, fourier


def variants(true_power: np.ndarray) -> dict[str, np.ndarray]:
    maximum = max(float(np.max(true_power)), 1e-30)
    missing = true_power.copy(); missing[82] = 0.0
    shifted = np.roll(true_power, 1)
    smoothed = np.convolve(true_power, np.array([0.2, 0.6, 0.2]), mode="same")
    spurious = true_power.copy(); spurious[69] += 0.22 * maximum
    thresholded = true_power.copy(); thresholded[thresholded < 0.1 * maximum] = 0.0
    return {
        "oracle_power": true_power,
        "miss_weak_reflector": missing,
        "shifted_support_plus1": shifted,
        "smoothed_support": smoothed,
        "spurious_reflector": spurious,
        "threshold_10pct": thresholded,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=1200)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260909)
    object_grid = 128
    given_indices = np.arange(-12, 12)
    missing_indices = np.concatenate([np.arange(-48, -12), np.arange(12, 48)])
    fg = fourier(given_indices, object_grid); fm = fourier(missing_indices, object_grid)
    true_power = np.zeros(object_grid); true_power[31] = 1.0; true_power[53] = 0.45**2; true_power[82] = 0.24**2
    powers = variants(true_power)
    noise_variance = 4e-4; sigma = np.eye(given_indices.size) * noise_variance
    errors = {name: [] for name in powers}; variances = {name: [] for name in powers}
    inclusive_errors = []; inclusive_variances = []; non_oracle_errors = []; non_oracle_variances = []
    for _ in range(args.trials):
        coefficients = np.sqrt(true_power / 2) * (rng.standard_normal(object_grid) + 1j * rng.standard_normal(object_grid))
        true_given = fg @ coefficients; true_missing = fm @ coefficients
        observed = true_given + np.sqrt(noise_variance / 2) * (rng.standard_normal(given_indices.size) + 1j * rng.standard_normal(given_indices.size))
        results = {}
        for name, power in powers.items():
            result = conditional_miaa(observed, fg, fm, power, sigma, max_condition_number=1e11)
            results[name] = result
            errors[name].append(np.abs(result.mean - true_missing) ** 2)
            variances[name].append(result.posterior_variance)
        inclusive = combine_results(results.values())
        inclusive_errors.append(np.abs(inclusive.mean - true_missing) ** 2); inclusive_variances.append(inclusive.posterior_variance)
        non_oracle = combine_results(result for name, result in results.items() if name != "oracle_power")
        non_oracle_errors.append(np.abs(non_oracle.mean - true_missing) ** 2); non_oracle_variances.append(non_oracle.posterior_variance)
    rows = []
    for name in powers:
        row = {"model": name}; row.update(calibration_metrics(np.asarray(errors[name]), np.mean(np.asarray(variances[name]), axis=0))); rows.append(row)
    for name, err, var in (("oracle_inclusive_ensemble", inclusive_errors, inclusive_variances), ("non_oracle_ensemble", non_oracle_errors, non_oracle_variances)):
        row = {"model": name}; row.update(calibration_metrics(np.asarray(err), np.mean(np.asarray(var), axis=0))); rows.append(row)
    by_name = {row["model"]: row for row in rows}
    oracle = by_name["oracle_power"]; missing = by_name["miss_weak_reflector"]; non_oracle = by_name["non_oracle_ensemble"]
    report = {
        "schema": "miaa-model-misspecification-stress-v2",
        "status": "VERIFIED_SYNTHETIC",
        "trials": args.trials,
        "generative_model": "Each trial draws complex-Gaussian reflectivity from the registered true diagonal power prior and adds proper complex Gaussian measurement noise.",
        "models": rows,
        "decisions": {
            "oracle_model_calibrated": 0.8 <= oracle["median_empirical_mse_over_reported_variance"] <= 1.25 and oracle["nominal_95pct_complex_error_coverage"] >= 0.92,
            "misspecified_fixed_model_overconfident": missing["median_empirical_mse_over_reported_variance"] > 1.5 or missing["nominal_95pct_complex_error_coverage"] < 0.90,
            "non_oracle_ensemble_improves_over_missing_model": non_oracle["nominal_95pct_complex_error_coverage"] > missing["nominal_95pct_complex_error_coverage"],
            "ensemble_replaces_physical_wideband_validation": False,
        },
        "boundary": "The registered ensemble is incomplete; unrepresented sample physics can still create undercoverage.",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
